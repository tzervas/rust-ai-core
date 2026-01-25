// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Memory estimation and tracking utilities for GPU operations.
//!
//! ## Why This Module Exists
//!
//! GPU memory (VRAM) is a precious and limited resource. Unlike CPU memory, there's no
//! swap space fallback when VRAM runs out - operations simply fail. This module provides
//! utilities to estimate memory requirements before allocation and track usage during
//! execution, enabling crates to:
//!
//! 1. **Pre-flight checks**: Verify sufficient VRAM before starting expensive operations
//! 2. **Batch size optimization**: Automatically adjust batch sizes to fit available memory
//! 3. **Memory budgeting**: Track allocations across multiple operations
//! 4. **Debugging**: Identify memory leaks or unexpected allocations
//!
//! ## Design Decisions
//!
//! - **Conservative estimation**: Estimates include overhead buffers because running out
//!   of memory mid-operation is worse than slightly underutilizing VRAM.
//!
//! - **No global state**: `MemoryTracker` is an explicit struct, not a global singleton,
//!   because different parts of an application may need independent tracking.
//!
//! - **Candle-agnostic sizes**: Functions work with shapes and dtypes directly, not just
//!   Candle tensors, enabling estimation before tensor creation.

use crate::error::{CoreError, Result};
use candle_core::DType;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Default overhead factor applied to memory estimates.
///
/// Why 1.1x: CUDA allocators have alignment requirements and fragmentation overhead.
/// A 10% buffer prevents edge-case OOM errors when estimates are exact.
pub const DEFAULT_OVERHEAD_FACTOR: f64 = 1.1;

/// Estimate the memory required to store a tensor with given shape and dtype.
///
/// This function calculates the raw memory requirement without overhead. Use
/// [`MemoryTracker::estimate_with_overhead`] for production estimates.
///
/// ## Arguments
///
/// * `shape` - Tensor dimensions (e.g., `[batch, seq_len, hidden_dim]`)
/// * `dtype` - Data type determining bytes per element
///
/// ## Returns
///
/// Memory requirement in bytes.
///
/// ## Why This Function
///
/// Pre-computing memory requirements allows batch size optimization and preflight
/// checks before committing to expensive allocations.
///
/// ## Example
///
/// ```rust
/// use rust_ai_core::estimate_tensor_bytes;
/// use candle_core::DType;
///
/// // LLaMA-2 7B attention output: [batch, heads, seq, head_dim]
/// let bytes = estimate_tensor_bytes(&[1, 32, 4096, 128], DType::BF16);
/// assert_eq!(bytes, 1 * 32 * 4096 * 128 * 2); // 32 MB
/// ```
#[must_use]
pub fn estimate_tensor_bytes(shape: &[usize], dtype: DType) -> usize {
    let numel: usize = shape.iter().product();
    numel * dtype.size_in_bytes()
}

/// Estimate memory for attention computation.
///
/// Attention requires storing Q, K, V tensors plus the attention weights matrix.
/// This function estimates the total memory for a single attention layer.
///
/// ## Arguments
///
/// * `batch_size` - Number of sequences in the batch
/// * `num_heads` - Number of attention heads
/// * `seq_len` - Sequence length (context window)
/// * `head_dim` - Dimension per attention head
/// * `dtype` - Data type for all tensors
///
/// ## Returns
///
/// Estimated memory in bytes for one attention layer.
///
/// ## Why This Function
///
/// Attention is the primary memory consumer in transformers. The attention weights
/// matrix scales with `O(seq_len²)`, making it the bottleneck for long sequences.
/// This estimate helps determine maximum context length for a given VRAM budget.
///
/// ## Memory Breakdown
///
/// - Q, K, V: 3 × (batch × heads × seq × `head_dim`)
/// - Attention weights: batch × heads × seq × seq
/// - Output: batch × heads × seq × `head_dim`
#[must_use]
pub fn estimate_attention_memory(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    dtype: DType,
) -> usize {
    let bytes_per_elem = dtype.size_in_bytes();

    // Q, K, V tensors
    let qkv_bytes = 3 * batch_size * num_heads * seq_len * head_dim * bytes_per_elem;

    // Attention weights matrix (the O(n²) component)
    let attn_weights_bytes = batch_size * num_heads * seq_len * seq_len * bytes_per_elem;

    // Output tensor
    let output_bytes = batch_size * num_heads * seq_len * head_dim * bytes_per_elem;

    qkv_bytes + attn_weights_bytes + output_bytes
}

/// Memory usage tracker for GPU operations.
///
/// Tracks allocated and peak memory usage across operations. Thread-safe via atomics.
///
/// ## Why This Struct
///
/// Unlike CPU memory which is managed by the OS, GPU memory requires explicit tracking
/// because:
///
/// 1. **No swap**: When VRAM runs out, allocations fail immediately
/// 2. **Fragmentation**: Repeated allocations can fragment the heap
/// 3. **Debugging**: Memory leaks on GPU are harder to diagnose than CPU leaks
///
/// ## Usage Pattern
///
/// ```rust
/// use rust_ai_core::MemoryTracker;
///
/// let tracker = MemoryTracker::new();
///
/// // Before allocation
/// tracker.allocate(1024 * 1024).expect("allocation should succeed"); // 1 MB
///
/// // After freeing
/// tracker.deallocate(1024 * 1024);
///
/// println!("Peak usage: {} bytes", tracker.peak_bytes());
/// ```
#[derive(Debug)]
pub struct MemoryTracker {
    /// Currently allocated bytes.
    allocated: AtomicUsize,
    /// Peak allocation during lifetime.
    peak: AtomicUsize,
    /// Optional memory limit (0 = unlimited).
    limit: AtomicUsize,
    /// Overhead factor for estimates.
    overhead_factor: f64,
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryTracker {
    /// Create a new memory tracker with no limit.
    ///
    /// ## Why No Default Limit
    ///
    /// Different GPUs have vastly different VRAM capacities (4GB to 80GB+).
    /// Setting a default limit would either be too restrictive for high-end cards
    /// or meaningless for consumer cards. Users should set limits explicitly.
    #[must_use]
    pub fn new() -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            limit: AtomicUsize::new(0),
            overhead_factor: DEFAULT_OVERHEAD_FACTOR,
        }
    }

    /// Create a tracker with a memory limit.
    ///
    /// ## Arguments
    ///
    /// * `limit_bytes` - Maximum allowed allocation in bytes
    ///
    /// ## Why Memory Limits
    ///
    /// Setting a limit below actual VRAM capacity reserves space for:
    /// - CUDA context overhead (~200-500 MB)
    /// - Framework allocations (PyTorch/Candle tensor cache)
    /// - Other processes sharing the GPU
    #[must_use]
    pub fn with_limit(limit_bytes: usize) -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            limit: AtomicUsize::new(limit_bytes),
            overhead_factor: DEFAULT_OVERHEAD_FACTOR,
        }
    }

    /// Set a custom overhead factor for estimates.
    ///
    /// ## Arguments
    ///
    /// * `factor` - Multiplier applied to estimates (default: 1.1)
    #[must_use]
    pub fn with_overhead_factor(mut self, factor: f64) -> Self {
        self.overhead_factor = factor;
        self
    }

    /// Record a memory allocation.
    ///
    /// ## Arguments
    ///
    /// * `bytes` - Number of bytes allocated
    ///
    /// ## Returns
    ///
    /// `Ok(())` if allocation is within limits.
    ///
    /// ## Errors
    ///
    /// Returns `CoreError::OutOfMemory` if allocation would exceed the limit.
    ///
    /// ## Why Track Allocations
    ///
    /// Explicit tracking catches memory issues early. Without tracking, OOM errors
    /// occur deep in CUDA kernels with unhelpful error messages.
    pub fn allocate(&self, bytes: usize) -> Result<()> {
        // Check limit BEFORE updating state to avoid partial updates on failure
        let limit = self.limit.load(Ordering::SeqCst);
        let current = self.allocated.load(Ordering::SeqCst);
        let new_allocated = current + bytes;

        if limit > 0 && new_allocated > limit {
            return Err(CoreError::oom(format!(
                "allocation of {bytes} bytes would exceed limit of {limit} bytes \
                 (current: {current} bytes)"
            )));
        }

        // Update allocated (actual update happens here)
        let actual_new = self.allocated.fetch_add(bytes, Ordering::SeqCst) + bytes;

        // Update peak
        let mut current_peak = self.peak.load(Ordering::SeqCst);
        while actual_new > current_peak {
            match self.peak.compare_exchange_weak(
                current_peak,
                actual_new,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(p) => current_peak = p,
            }
        }

        Ok(())
    }

    /// Record a memory deallocation.
    ///
    /// ## Arguments
    ///
    /// * `bytes` - Number of bytes freed
    pub fn deallocate(&self, bytes: usize) {
        self.allocated.fetch_sub(bytes, Ordering::SeqCst);
    }

    /// Get currently allocated bytes.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated.load(Ordering::SeqCst)
    }

    /// Get peak allocation during tracker lifetime.
    ///
    /// ## Why Track Peak
    ///
    /// Peak usage is more useful than current usage for capacity planning.
    /// It shows the high-water mark needed to complete a workload.
    #[must_use]
    pub fn peak_bytes(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }

    /// Get configured memory limit (0 = unlimited).
    #[must_use]
    pub fn limit_bytes(&self) -> usize {
        self.limit.load(Ordering::SeqCst)
    }

    /// Estimate required memory with overhead factor applied.
    ///
    /// ## Arguments
    ///
    /// * `shape` - Tensor dimensions
    /// * `dtype` - Data type
    ///
    /// ## Returns
    ///
    /// Estimated bytes including overhead buffer.
    #[must_use]
    pub fn estimate_with_overhead(&self, shape: &[usize], dtype: DType) -> usize {
        let raw = estimate_tensor_bytes(shape, dtype);
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        {
            (raw as f64 * self.overhead_factor) as usize
        }
    }

    /// Check if an allocation would fit within limits.
    ///
    /// ## Arguments
    ///
    /// * `bytes` - Proposed allocation size
    ///
    /// ## Returns
    ///
    /// `true` if allocation would succeed.
    #[must_use]
    pub fn would_fit(&self, bytes: usize) -> bool {
        let limit = self.limit.load(Ordering::SeqCst);
        if limit == 0 {
            return true; // No limit
        }
        self.allocated.load(Ordering::SeqCst) + bytes <= limit
    }

    /// Reset the tracker to initial state.
    ///
    /// ## Why Reset
    ///
    /// Between training epochs or inference batches, resetting allows tracking
    /// per-phase memory usage without creating new tracker instances.
    pub fn reset(&self) {
        self.allocated.store(0, Ordering::SeqCst);
        self.peak.store(0, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tensor_bytes() {
        // 1000 f32 elements = 4000 bytes
        assert_eq!(estimate_tensor_bytes(&[10, 100], DType::F32), 4000);

        // 1000 f16 elements = 2000 bytes
        assert_eq!(estimate_tensor_bytes(&[10, 100], DType::F16), 2000);

        // Empty tensor = 0 bytes
        assert_eq!(estimate_tensor_bytes(&[0], DType::F32), 0);
    }

    #[test]
    fn test_estimate_attention_memory() {
        // Simple case: 1 batch, 1 head, 4 seq, 2 head_dim, f32
        let bytes = estimate_attention_memory(1, 1, 4, 2, DType::F32);
        // QKV: 3 * 1 * 1 * 4 * 2 * 4 = 96
        // Attn: 1 * 1 * 4 * 4 * 4 = 64
        // Out: 1 * 1 * 4 * 2 * 4 = 32
        // Total: 192
        assert_eq!(bytes, 192);
    }

    #[test]
    fn test_memory_tracker_allocation() {
        let tracker = MemoryTracker::with_limit(1000);

        // Successful allocation
        assert!(tracker.allocate(500).is_ok());
        assert_eq!(tracker.allocated_bytes(), 500);

        // Second allocation
        assert!(tracker.allocate(400).is_ok());
        assert_eq!(tracker.allocated_bytes(), 900);

        // Exceeds limit
        assert!(tracker.allocate(200).is_err());
        assert_eq!(tracker.allocated_bytes(), 900); // Unchanged

        // Deallocation
        tracker.deallocate(400);
        assert_eq!(tracker.allocated_bytes(), 500);

        // Now fits
        assert!(tracker.allocate(200).is_ok());
    }

    #[test]
    fn test_memory_tracker_peak() {
        let tracker = MemoryTracker::new();

        tracker.allocate(100).unwrap();
        tracker.allocate(200).unwrap();
        assert_eq!(tracker.peak_bytes(), 300);

        tracker.deallocate(200);
        assert_eq!(tracker.allocated_bytes(), 100);
        assert_eq!(tracker.peak_bytes(), 300); // Peak unchanged

        tracker.allocate(50).unwrap();
        assert_eq!(tracker.peak_bytes(), 300); // Still 300

        tracker.allocate(300).unwrap();
        assert_eq!(tracker.peak_bytes(), 450); // New peak
    }

    #[test]
    fn test_would_fit() {
        let tracker = MemoryTracker::with_limit(1000);
        tracker.allocate(500).unwrap();

        assert!(tracker.would_fit(400));
        assert!(tracker.would_fit(500));
        assert!(!tracker.would_fit(501));

        // Unlimited tracker
        let unlimited = MemoryTracker::new();
        assert!(unlimited.would_fit(usize::MAX));
    }
}
