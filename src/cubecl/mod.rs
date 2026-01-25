// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! CubeCL ↔ Candle tensor interoperability.
//!
//! This module provides utilities for converting between Candle tensors and
//! CubeCL buffer handles, enabling seamless integration between the two frameworks.
//!
//! ## Key Functions
//!
//! - [`candle_to_cubecl_handle`] - Convert contiguous Candle tensor to CubeCL buffer
//! - [`cubecl_to_candle_tensor`] - Convert CubeCL output back to Candle tensor
//! - [`has_cubecl_cuda_support`] - Check if CUDA runtime is available
//!
//! ## Memory Management
//!
//! The conversion functions handle:
//! - Ensuring tensor contiguity (required for raw pointer access)
//! - Buffer creation via `client.create(bytes)`
//! - Buffer reuse where possible to minimize allocations
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
    candle_to_cubecl_handle, cubecl_to_candle_tensor, has_cubecl_cuda_support, TensorBuffer,
};
