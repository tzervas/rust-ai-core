// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Common types and utilities for TypeScript bindings.

// Allow casts for JS interop - JavaScript numbers are limited to f64/u32 ranges anyway
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::missing_errors_doc)] // Errors documented in JS docstrings

use std::sync::{Arc, Mutex};

use crate::dtype;
use crate::memory::MemoryTracker;

/// Thread-safe wrapper around [`MemoryTracker`] for JavaScript bindings.
#[derive(Clone)]
pub struct JsMemoryTracker {
    /// Inner memory tracker protected by mutex for thread-safe mutation.
    pub inner: Arc<Mutex<MemoryTracker>>,
}

impl JsMemoryTracker {
    /// Create a new memory tracker with the specified limit and overhead factor.
    #[must_use]
    pub fn new(limit_gb: f64, overhead_factor: f64) -> Self {
        let limit_bytes = (limit_gb * 1024.0 * 1024.0 * 1024.0) as usize;
        let tracker = MemoryTracker::with_limit(limit_bytes).with_overhead_factor(overhead_factor);
        Self {
            inner: Arc::new(Mutex::new(tracker)),
        }
    }

    /// Check if an allocation would fit within the tracker's limit.
    pub fn would_fit(&self, bytes: usize) -> Result<bool, String> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {e}"))?;
        Ok(inner.would_fit(bytes))
    }

    /// Record an allocation in the tracker.
    pub fn allocate(&self, bytes: usize) -> Result<(), String> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {e}"))?;
        inner
            .allocate(bytes)
            .map_err(|e| format!("Allocation failed: {e}"))
    }

    /// Record a deallocation in the tracker.
    pub fn deallocate(&self, bytes: usize) -> Result<(), String> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {e}"))?;
        inner.deallocate(bytes);
        Ok(())
    }

    /// Get current allocation in bytes.
    pub fn allocated_bytes(&self) -> Result<usize, String> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {e}"))?;
        Ok(inner.allocated_bytes())
    }

    /// Get peak allocation in bytes.
    pub fn peak_bytes(&self) -> Result<usize, String> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {e}"))?;
        Ok(inner.peak_bytes())
    }

    /// Get the memory limit in bytes.
    pub fn limit_bytes(&self) -> Result<usize, String> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {e}"))?;
        Ok(inner.limit_bytes())
    }

    /// Estimate bytes with overhead factor applied.
    pub fn estimate_with_overhead(&self, shape: &[usize], dtype: &str) -> Result<usize, String> {
        let candle_dtype = parse_dtype(dtype)?;
        let inner = self
            .inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {e}"))?;
        Ok(inner.estimate_with_overhead(shape, candle_dtype))
    }

    /// Reset the tracker to initial state.
    pub fn reset(&self) -> Result<(), String> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {e}"))?;
        inner.reset();
        Ok(())
    }
}

/// Device information returned by `getDeviceInfo()`.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device type: "cuda", "metal", or "cpu".
    pub device_type: String,
    /// Device ordinal (GPU index), `None` for CPU.
    pub ordinal: Option<usize>,
    /// Human-readable device name.
    pub name: String,
}

/// Parse a dtype string into a Candle DType.
pub fn parse_dtype(dtype: &str) -> Result<candle_core::DType, String> {
    match dtype.to_lowercase().as_str() {
        "f16" | "float16" => Ok(candle_core::DType::F16),
        "bf16" | "bfloat16" => Ok(candle_core::DType::BF16),
        "f32" | "float32" | "float" => Ok(candle_core::DType::F32),
        "f64" | "float64" | "double" => Ok(candle_core::DType::F64),
        "u8" | "uint8" => Ok(candle_core::DType::U8),
        "u32" | "uint32" => Ok(candle_core::DType::U32),
        "i16" | "int16" => Ok(candle_core::DType::I16),
        "i32" | "int32" | "int" => Ok(candle_core::DType::I32),
        "i64" | "int64" | "long" => Ok(candle_core::DType::I64),
        _ => Err(format!(
            "Unknown dtype: {dtype}. Supported: f16, bf16, f32, f64, u8, u32, i16, i32, i64"
        )),
    }
}

/// Convert a Candle DType back to its string representation.
#[must_use]
pub fn dtype_to_string(dtype: candle_core::DType) -> &'static str {
    match dtype {
        candle_core::DType::F16 => "f16",
        candle_core::DType::BF16 => "bf16",
        candle_core::DType::F32 => "f32",
        candle_core::DType::F64 => "f64",
        candle_core::DType::U8 => "u8",
        candle_core::DType::U32 => "u32",
        candle_core::DType::I16 => "i16",
        candle_core::DType::I32 => "i32",
        candle_core::DType::I64 => "i64",
        candle_core::DType::F8E4M3 => "f8e4m3",
        candle_core::DType::F6E2M3 => "f6e2m3",
        candle_core::DType::F6E3M2 => "f6e3m2",
        candle_core::DType::F4 => "f4",
        candle_core::DType::F8E8M0 => "f8e8m0",
    }
}

/// Get bytes per element for a dtype.
pub fn bytes_per_dtype(dtype: &str) -> Result<usize, String> {
    let candle_dtype = parse_dtype(dtype)?;
    Ok(dtype::bytes_per_element(candle_dtype))
}

/// Check if a dtype is floating point.
pub fn is_floating_point_dtype(dtype: &str) -> Result<bool, String> {
    let candle_dtype = parse_dtype(dtype)?;
    Ok(dtype::is_floating_point(candle_dtype))
}

/// Get the accumulator dtype for a given dtype.
pub fn accumulator_dtype(dtype: &str) -> Result<String, String> {
    use crate::dtype::DTypeExt;
    let candle_dtype = parse_dtype(dtype)?;
    let acc_dtype = candle_dtype.accumulator_dtype();
    Ok(dtype_to_string(acc_dtype).to_string())
}

/// Get all supported dtypes.
#[must_use]
pub fn supported_dtypes() -> Vec<&'static str> {
    vec![
        "f16", "bf16", "f32", "f64", "u8", "u32", "i16", "i32", "i64",
    ]
}

/// Default overhead factor for memory estimation.
#[must_use]
pub fn default_overhead_factor() -> f64 {
    crate::memory::DEFAULT_OVERHEAD_FACTOR
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dtype() {
        assert!(matches!(
            parse_dtype("f32").unwrap(),
            candle_core::DType::F32
        ));
        assert!(matches!(
            parse_dtype("bf16").unwrap(),
            candle_core::DType::BF16
        ));
        assert!(parse_dtype("invalid").is_err());
    }

    #[test]
    fn test_js_memory_tracker() {
        let tracker = JsMemoryTracker::new(1.0, 1.1);
        assert_eq!(tracker.allocated_bytes().unwrap(), 0);
        tracker.allocate(1024).unwrap();
        assert_eq!(tracker.allocated_bytes().unwrap(), 1024);
        tracker.deallocate(512).unwrap();
        assert_eq!(tracker.allocated_bytes().unwrap(), 512);
    }
}
