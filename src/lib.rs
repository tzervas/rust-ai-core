// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! # rust-ai-core
//!
//! Unified AI engineering toolkit that orchestrates the complete rust-ai ecosystem.
//!
//! This crate integrates 8 specialized AI/ML crates into a cohesive toolkit:
//!
//! | Crate | Purpose |
//! |-------|---------|
//! | **peft-rs** | LoRA, DoRA, AdaLoRA adapters for parameter-efficient fine-tuning |
//! | **qlora-rs** | 4-bit quantized LoRA for memory-efficient fine-tuning |
//! | **unsloth-rs** | Optimized transformer blocks (attention, FFN, normalization) |
//! | **axolotl-rs** | YAML-driven training orchestration and configuration |
//! | **bitnet-quantize** | Microsoft BitNet b1.58 ternary quantization |
//! | **trit-vsa** | Balanced ternary arithmetic and VSA operations |
//! | **vsa-optim-rs** | VSA-based deterministic training optimization |
//! | **tritter-accel** | GPU-accelerated ternary operations |
//!
//! ## Quick Start
//!
//! ### Using the Unified API
//!
//! ```rust,ignore
//! use rust_ai_core::{RustAI, RustAIConfig};
//!
//! // Initialize the unified API
//! let ai = RustAI::new(RustAIConfig::default())?;
//!
//! // Configure fine-tuning with LoRA
//! let finetune_config = ai.finetune()
//!     .model("meta-llama/Llama-2-7b")
//!     .rank(64)
//!     .alpha(16.0)
//!     .build()?;
//!
//! // Configure quantization
//! let quant_config = ai.quantize()
//!     .method(QuantizeMethod::Nf4)
//!     .bits(4)
//!     .build();
//!
//! // Configure VSA operations
//! let vsa_config = ai.vsa()
//!     .dimension(10000)
//!     .build();
//! ```
//!
//! ### Direct Crate Access
//!
//! ```rust,ignore
//! use rust_ai_core::ecosystem::peft::{LoraConfig, LoraLinear};
//! use rust_ai_core::ecosystem::qlora::QLoraConfig;
//! use rust_ai_core::ecosystem::bitnet::TernaryLinear;
//! use rust_ai_core::ecosystem::trit::TritVector;
//! ```
//!
//! ### Using Foundation Types
//!
//! ```rust
//! use rust_ai_core::{get_device, DeviceConfig, CoreError, Result};
//!
//! fn main() -> Result<()> {
//!     // Get CUDA device with automatic fallback + warning
//!     let device = get_device(&DeviceConfig::default())?;
//!
//!     // Or configure from environment variables
//!     let config = DeviceConfig::from_env();
//!     let device = get_device(&config)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Design Philosophy
//!
//! **CUDA-first**: All operations prefer GPU execution. CPU is a fallback that emits
//! warnings, not a silent alternative. Users are immediately aware when they're not
//! getting optimal performance.
//!
//! **Zero-cost abstractions**: Traits compile to static dispatch with no vtable overhead.
//! The abstraction layer adds no runtime cost compared to direct implementations.
//!
//! **Fail-fast validation**: Configuration errors are caught at construction time, not
//! deep in a training loop. This saves users from wasted compute time.
//!
//! **Unified API**: The [`RustAI`] facade provides a single entry point for all AI
//! engineering tasks, automatically composing the right ecosystem crates.
//!
//! ## Modules
//!
//! ### Foundation Modules
//!
//! - [`device`] - CUDA-first device selection with environment variable overrides
//! - [`error`] - Unified error types across all rust-ai crates
//! - [`traits`] - Common traits for configs, quantization, and GPU dispatch
//! - [`memory`] - Memory estimation and tracking utilities
//! - [`dtype`] - Data type utilities and precision helpers
//! - [`logging`] - Unified logging and observability
//! - [`cubecl`] - CubeCL ↔ Candle tensor interoperability (requires `cuda` feature)
//!
//! ### Ecosystem Modules
//!
//! - [`ecosystem`] - Re-exports from all 8 rust-ai ecosystem crates
//! - [`facade`] - High-level unified API ([`RustAI`])
//!
//! ## Feature Flags
//!
//! | Feature | Description | Dependencies |
//! |---------|-------------|--------------|
//! | `cuda` | Enable CUDA support via Candle and CubeCL | cubecl, cubecl-cuda |
//! | `python` | Enable Python bindings via PyO3 | pyo3, numpy |
//! | `full` | Enable both CUDA and Python | cuda, python |
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

// =============================================================================
// FOUNDATION MODULES
// =============================================================================

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

// =============================================================================
// ECOSYSTEM INTEGRATION MODULES
// =============================================================================

/// Unified re-exports from all rust-ai ecosystem crates.
///
/// Access individual crates through their submodules:
/// - `ecosystem::peft` - LoRA, DoRA, AdaLoRA
/// - `ecosystem::qlora` - 4-bit quantized LoRA
/// - `ecosystem::unsloth` - Optimized transformers
/// - `ecosystem::axolotl` - Training orchestration
/// - `ecosystem::bitnet` - 1.58-bit quantization
/// - `ecosystem::trit` - Ternary VSA
/// - `ecosystem::vsa_optim` - VSA optimization
/// - `ecosystem::tritter` - Ternary acceleration
pub mod ecosystem;

/// High-level unified API facade.
///
/// The [`RustAI`] struct provides a simplified interface for common AI
/// engineering tasks, orchestrating multiple ecosystem crates automatically.
pub mod facade;

// =============================================================================
// RE-EXPORTS
// =============================================================================

// Foundation re-exports for convenience.
pub use device::{get_device, warn_if_cpu, DeviceConfig};
pub use dtype::{bytes_per_element, is_floating_point, DTypeExt};
pub use error::{CoreError, Result};
pub use logging::{init_logging, LogConfig};
pub use memory::{estimate_tensor_bytes, MemoryTracker};
pub use traits::{Dequantize, GpuDispatchable, Quantize, ValidatableConfig};

// CubeCL interop (CUDA only)
#[cfg(feature = "cuda")]
pub use cubecl::{
    allocate_output_buffer, candle_to_cubecl_handle, cubecl_to_candle_tensor,
    has_cubecl_cuda_support, TensorBuffer,
};

// Unified API facade
pub use facade::{
    AdapterType, FinetuneBuilder, FinetuneConfig, QuantizeBuilder, QuantizeConfig,
    QuantizeMethod, RustAI, RustAIConfig, RustAIInfo, TrainBuilder, TrainConfig,
    VsaBuilder, VsaConfig,
};

// Ecosystem information
pub use ecosystem::EcosystemInfo;

/// Crate version for runtime version checking.
///
/// Why: Dependent crates may need to verify compatibility at runtime,
/// especially when loading serialized data that includes version info.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
