// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! # rust-ai-core
//!
//! Shared core utilities for the rust-ai ecosystem, providing unified abstractions
//! for device selection, error handling, configuration validation, and `CubeCL` interop.
//!
//! ## Why This Crate Exists
//!
//! The rust-ai ecosystem consists of multiple specialized crates (peft-rs, qlora-rs,
//! unsloth-rs, etc.) that share common patterns. Without a foundation layer, each crate
//! would independently implement device selection, error types, and GPU dispatch logic,
//! leading to inconsistency and code duplication.
//!
//! rust-ai-core consolidates these patterns into a single, well-tested foundation that
//! ensures consistency across the ecosystem and provides a unified user experience.
//!
//! ## Design Philosophy
//!
//! **CUDA-first**: All operations prefer GPU execution. CPU is a fallback that emits
//! warnings, not a silent alternative. This design ensures users are immediately aware
//! when they're not getting optimal performance, rather than silently running slower.
//!
//! **Zero-cost abstractions**: Traits compile to static dispatch with no vtable overhead.
//! The abstraction layer adds no runtime cost compared to direct implementations.
//!
//! **Fail-fast validation**: Configuration errors are caught at construction time, not
//! deep in a training loop. This saves users from wasted compute time.
//!
//! ## Modules
//!
//! - [`device`] - CUDA-first device selection with environment variable overrides
//! - [`mod@error`] - Unified error types across all rust-ai crates
//! - [`traits`] - Common traits for configs, quantization, and GPU dispatch
//! - [`memory`] - Memory estimation and tracking utilities
//! - [`dtype`] - Data type utilities and precision helpers
//! - [`logging`] - Unified logging and observability
//! - `cubecl` - `CubeCL` ↔ Candle tensor interoperability (requires `cuda` feature)
//!
//! ## Quick Start
//!
//! ```rust
//! use rust_ai_core::{get_device, DeviceConfig, CoreError, Result};
//!
//! fn main() -> Result<()> {
//!     // Get CUDA device with automatic fallback + warning
//!     // Why: Users should know when they're not getting GPU acceleration
//!     let device = get_device(&DeviceConfig::default())?;
//!
//!     // Or with explicit configuration from environment
//!     // Why: Production deployments need runtime configuration without recompilation
//!     let config = DeviceConfig::from_env();
//!     let device = get_device(&config)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Description | Dependencies |
//! |---------|-------------|--------------|
//! | `cuda` | Enable CUDA support via Candle and `CubeCL` kernels | cubecl, cubecl-cuda |
//! | `python` | Enable Python bindings via `PyO3` | pyo3, numpy |
//!
//! ## Crate Integration
//!
//! All rust-ai crates should depend on rust-ai-core and use its shared types.
//! This ensures consistent error handling, device selection, and trait implementations
//! across the ecosystem.
//!
//! ```toml
//! [dependencies]
//! rust-ai-core = { version = "0.1", features = ["cuda"] }
//! ```
//!
//! ## Environment Variables
//!
//! | Variable | Description | Example |
//! |----------|-------------|---------|
//! | `RUST_AI_FORCE_CPU` | Force CPU execution | `1` or `true` |
//! | `RUST_AI_CUDA_DEVICE` | Select CUDA device ordinal | `0`, `1` |
//! | `RUST_LOG` | Control logging level | `rust_ai_core=debug` |

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod device;
pub mod dtype;
pub mod error;
pub mod logging;
pub mod memory;
pub mod traits;

#[cfg(feature = "cuda")]
pub mod cubecl;

#[cfg(feature = "python")]
pub mod python;

// Re-exports for convenience.
//
// Why: Users shouldn't need to navigate the module hierarchy for common types.
// Flat re-exports provide a clean, discoverable API surface.
pub use device::{get_device, warn_if_cpu, DeviceConfig};
pub use dtype::{bytes_per_element, is_floating_point, DTypeExt};
pub use error::{CoreError, Result};
pub use logging::{
    debug, error, info, init_logging, log_memory_usage, log_training_step, trace, warn, LogConfig,
};
pub use memory::{
    estimate_attention_memory, estimate_tensor_bytes, MemoryTracker, DEFAULT_OVERHEAD_FACTOR,
};
pub use traits::{Dequantize, GpuDispatchable, Quantize, ValidatableConfig};

#[cfg(feature = "cuda")]
pub use cubecl::{
    allocate_output_buffer, candle_to_cubecl_handle, cubecl_to_candle_tensor,
    has_cubecl_cuda_support, TensorBuffer,
};

/// Crate version for runtime version checking.
///
/// Why: Dependent crates may need to verify compatibility at runtime,
/// especially when loading serialized data that includes version info.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
