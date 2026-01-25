// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! CubeCL ↔ Candle tensor interoperability.
//!
//! This module provides utilities for converting between Candle tensors and
//! CubeCL buffer handles, enabling seamless integration between the two frameworks.
//!
//! ## Overview
//!
//! Candle provides high-level tensor operations using a PyTorch-like API, while
//! CubeCL provides low-level GPU kernel execution. This module bridges the gap,
//! allowing you to:
//!
//! 1. Convert Candle tensors to raw byte buffers for CubeCL kernels
//! 2. Create Candle tensors from CubeCL kernel outputs
//! 3. Check CUDA runtime availability
//!
//! ## Key Functions
//!
//! - [`candle_to_cubecl_handle`] - Convert contiguous Candle tensor to CubeCL buffer
//! - [`cubecl_to_candle_tensor`] - Convert CubeCL output back to Candle tensor
//! - [`has_cubecl_cuda_support`] - Check if CUDA runtime is available
//! - [`allocate_output_buffer`] - Pre-allocate output buffer for kernel results
//!
//! ## Memory Management
//!
//! The conversion functions handle:
//! - Ensuring tensor contiguity (required for raw pointer access)
//! - Dtype-specific byte encoding (f32, f16, bf16)
//! - Shape and size validation
//! - Zero-copy where possible for contiguous tensors
//!
//! ## Typical Usage Pattern
//!
//! ```rust,ignore
//! use rust_ai_core::{candle_to_cubecl_handle, cubecl_to_candle_tensor};
//! use cubecl::Runtime;
//!
//! // 1. Convert Candle tensor to buffer
//! let input_buffer = candle_to_cubecl_handle(&input_tensor)?;
//!
//! // 2. Create CubeCL handle and launch kernel
//! let input_handle = client.create(&input_buffer.bytes);
//! let output_handle = my_kernel.launch(&input_handle);
//!
//! // 3. Read kernel output
//! let output_bytes = client.read(&output_handle);
//! let output_buffer = TensorBuffer::new(output_bytes, output_shape, dtype);
//!
//! // 4. Convert back to Candle tensor
//! let output_tensor = cubecl_to_candle_tensor(&output_buffer, &device)?;
//! ```
//!
//! ## Performance Tips
//!
//! - **Pre-allocate buffers**: Use [`allocate_output_buffer`] outside loops
//! - **Ensure contiguity**: Non-contiguous tensors require a copy operation
//! - **Reuse buffers**: Minimize allocations in hot paths
//!
//! ## Feature Gate
//!
//! This module is only available when the `cuda` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! rust-ai-core = { version = "0.1", features = ["cuda"] }
//! ```

mod interop;

pub use interop::{
    allocate_output_buffer, candle_to_cubecl_handle, cubecl_to_candle_tensor,
    has_cubecl_cuda_support, TensorBuffer,
};
