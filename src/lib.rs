// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! # rust-ai-core
//!
//! Shared core utilities for the rust-ai ecosystem, providing unified abstractions
//! for device selection, error handling, configuration validation, and `CubeCL` interop.
//!
//! ## Design Philosophy
//!
//! **CUDA-first**: All operations prefer GPU execution. CPU is a fallback that emits
//! warnings, not a silent alternative. This ensures users are aware when they're not
//! getting optimal performance.
//!
//! ## Modules
//!
//! - [`device`] - CUDA-first device selection with environment variable overrides
//! - [`error`] - Unified error types across all rust-ai crates
//! - [`traits`] - Common traits for configs, quantization, and GPU dispatch
//! - `cubecl` - `CubeCL` ↔ Candle tensor interoperability (feature-gated with `cuda` feature)
//!
//! ## Quick Start
//!
//! ```rust
//! use rust_ai_core::{get_device, DeviceConfig, CoreError, Result};
//!
//! fn main() -> Result<()> {
//!     // Get CUDA device with automatic fallback + warning
//!     let device = get_device(&DeviceConfig::default())?;
//!     
//!     // Or with explicit configuration
//!     let config = DeviceConfig::new()
//!         .with_cuda_device(0)
//!         .with_force_cpu(false);
//!     let device = get_device(&config)?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! - `cuda` - Enable CUDA support via Candle and `CubeCL` kernels
//!
//! ## Crate Integration
//!
//! All rust-ai crates should depend on rust-ai-core and use its shared types:
//!
//! ```toml
//! [dependencies]
//! rust-ai-core = { version = "0.1", features = ["cuda"] }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod device;
pub mod error;
pub mod traits;

#[cfg(feature = "cuda")]
pub mod cubecl;

// Re-exports for convenience
pub use device::{get_device, warn_if_cpu, DeviceConfig};
pub use error::{CoreError, Result};
pub use traits::{Dequantize, GpuDispatchable, Quantize, ValidatableConfig};

#[cfg(feature = "cuda")]
pub use cubecl::{
    allocate_output_buffer, candle_to_cubecl_handle, cubecl_to_candle_tensor,
    has_cubecl_cuda_support, TensorBuffer,
};
