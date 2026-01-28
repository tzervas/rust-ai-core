// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Unified re-exports from the rust-ai ecosystem crates.
//!
//! This module provides convenient access to all rust-ai ecosystem crates
//! through a single import path. All crates are always available as they
//! are required dependencies of rust-ai-core.
//!
//! ## Available Modules
//!
//! | Module | Crate | Description |
//! |--------|-------|-------------|
//! | [`peft`] | peft-rs | LoRA, DoRA, AdaLoRA adapters |
//! | [`qlora`] | qlora-rs | 4-bit quantized fine-tuning |
//! | [`unsloth`] | unsloth-rs | Optimized transformer blocks |
//! | [`axolotl`] | axolotl-rs | Fine-tuning orchestration |
//! | [`bitnet`] | bitnet-quantize | BitNet 1.58-bit quantization |
//! | [`trit`] | trit-vsa | Ternary VSA operations |
//! | [`vsa_optim`] | vsa-optim-rs | VSA-based optimization |
//! | [`tritter`] | tritter-accel | Ternary GPU acceleration |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use rust_ai_core::ecosystem::peft::{LoraConfig, LoraLinear};
//! use rust_ai_core::ecosystem::qlora::QLoraConfig;
//! use rust_ai_core::ecosystem::bitnet::TernaryLinear;
//! ```
//!
//! Or use the top-level facade for common operations:
//!
//! ```rust,ignore
//! use rust_ai_core::RustAI;
//!
//! let ai = RustAI::new(RustAIConfig::default())?;
//! let config = ai.finetune()
//!     .model("meta-llama/Llama-2-7b")
//!     .rank(64)
//!     .build()?;
//! ```

// =============================================================================
// PEFT (Parameter-Efficient Fine-Tuning)
// =============================================================================

/// LoRA, DoRA, and AdaLoRA adapter implementations.
///
/// Re-exports from `peft-rs` crate.
///
/// ## Key Types
///
/// - `LoraConfig` - Configuration for LoRA adapters
/// - `LoraLinear` - LoRA-wrapped linear layer
/// - `DoraConfig` - Configuration for DoRA (Weight-Decomposed LoRA)
/// - `AdaLoraConfig` - Configuration for AdaLoRA (Adaptive Budget)
///
/// ## Example
///
/// ```rust,ignore
/// use rust_ai_core::ecosystem::peft::{LoraConfig, LoraLinear};
///
/// let config = LoraConfig::new(64, 16.0); // rank=64, alpha=16.0
/// let lora_layer = LoraLinear::new(base_linear, &config)?;
/// ```
pub mod peft {
    pub use peft_rs::*;
}

// =============================================================================
// QLoRA (Quantized LoRA)
// =============================================================================

/// 4-bit quantized LoRA for memory-efficient fine-tuning.
///
/// Re-exports from `qlora-rs` crate.
///
/// ## Key Types
///
/// - `QLoraConfig` - Combined quantization and LoRA configuration
/// - `Nf4Quantizer` - NF4 (Normal Float 4-bit) quantizer
/// - `QuantizedLinear` - Quantized linear layer with LoRA
///
/// ## Example
///
/// ```rust,ignore
/// use rust_ai_core::ecosystem::qlora::{QLoraConfig, QuantizedLinear};
///
/// let config = QLoraConfig::default()
///     .with_lora_rank(32)
///     .with_bits(4);
/// let qlora_layer = QuantizedLinear::new(weights, &config)?;
/// ```
pub mod qlora {
    pub use qlora_rs::*;
}

// =============================================================================
// Unsloth (Optimized Transformers)
// =============================================================================

/// Optimized transformer building blocks.
///
/// Re-exports from `unsloth-rs` crate.
///
/// ## Key Types
///
/// - `FlashAttention` - Memory-efficient attention implementation
/// - `SwiGLU` - SwiGLU activation (used in Llama models)
/// - `RMSNorm` - Root Mean Square layer normalization
///
/// ## Example
///
/// ```rust,ignore
/// use rust_ai_core::ecosystem::unsloth::{FlashAttention, AttentionConfig};
///
/// let attn = FlashAttention::new(&config, device)?;
/// let output = attn.forward(&q, &k, &v, mask)?;
/// ```
pub mod unsloth {
    pub use unsloth_rs::*;
}

// =============================================================================
// Axolotl (Fine-Tuning Orchestration)
// =============================================================================

/// YAML-driven fine-tuning configuration and orchestration.
///
/// Re-exports from `axolotl-rs` crate.
///
/// ## Key Types
///
/// - `AxolotlConfig` - Main configuration struct (loadable from YAML)
/// - `TrainingPipeline` - Orchestrates the training workflow
/// - `DatasetConfig` - Dataset loading and preprocessing configuration
///
/// ## Example
///
/// ```rust,ignore
/// use rust_ai_core::ecosystem::axolotl::{AxolotlConfig, TrainingPipeline};
///
/// let config = AxolotlConfig::from_yaml("config.yaml")?;
/// let pipeline = TrainingPipeline::new(config)?;
/// pipeline.run()?;
/// ```
pub mod axolotl {
    pub use axolotl_rs::*;
}

// =============================================================================
// BitNet (1.58-bit Quantization)
// =============================================================================

/// Microsoft BitNet b1.58 quantization and inference.
///
/// Re-exports from `bitnet-quantize` crate.
///
/// ## Key Types
///
/// - `BitNetConfig` - Configuration for BitNet quantization
/// - `TernaryLinear` - Linear layer with ternary weights (-1, 0, +1)
/// - `BitNetQuantizer` - Quantizes weights to 1.58-bit representation
///
/// ## Example
///
/// ```rust,ignore
/// use rust_ai_core::ecosystem::bitnet::{BitNetConfig, TernaryLinear};
///
/// let config = BitNetConfig::default();
/// let ternary_layer = TernaryLinear::from_linear(linear, &config)?;
/// ```
pub mod bitnet {
    pub use bitnet_quantize::*;
}

// =============================================================================
// Trit-VSA (Ternary Vector Symbolic Architectures)
// =============================================================================

/// Balanced ternary arithmetic with bitsliced storage.
///
/// Re-exports from `trit-vsa` crate.
///
/// ## Key Types
///
/// - `TritVector` - Balanced ternary vector (-1, 0, +1)
/// - `TritSlice` - Bitsliced storage for efficient operations
/// - `HdcEncoder` - Hyperdimensional computing encoder
///
/// ## Example
///
/// ```rust,ignore
/// use rust_ai_core::ecosystem::trit::{TritVector, TritOps};
///
/// let a = TritVector::random(10000);
/// let b = TritVector::random(10000);
/// let bound = a.bind(&b); // Multiplication in VSA
/// ```
pub mod trit {
    pub use trit_vsa::*;
}

// =============================================================================
// VSA-Optim (VSA-Based Optimization)
// =============================================================================

/// Deterministic training optimization using VSA compression.
///
/// Re-exports from `vsa-optim-rs` crate.
///
/// ## Key Types
///
/// - `VsaOptimizer` - VSA-based optimizer with gradient prediction
/// - `CompressionConfig` - Configuration for gradient compression
/// - `GradientPredictor` - Closed-form gradient prediction
///
/// ## Example
///
/// ```rust,ignore
/// use rust_ai_core::ecosystem::vsa_optim::{VsaOptimizer, VsaConfig};
///
/// let config = VsaConfig::default()
///     .with_dimension(10000)
///     .with_compression_ratio(0.1);
/// let optimizer = VsaOptimizer::new(model.parameters(), config)?;
/// ```
pub mod vsa_optim {
    pub use vsa_optim_rs::*;
}

// =============================================================================
// Tritter-Accel (Ternary GPU Acceleration)
// =============================================================================

/// GPU-accelerated ternary operations for BitNet and VSA.
///
/// Re-exports from `tritter-accel` crate.
///
/// ## Key Types
///
/// - `TritterRuntime` - GPU runtime for ternary operations
/// - `TernaryMatmul` - Optimized ternary matrix multiplication
/// - `PackedTernary` - Memory-efficient ternary storage
///
/// ## Example
///
/// ```rust,ignore
/// use rust_ai_core::ecosystem::tritter::{TritterRuntime, TernaryMatmul};
///
/// let runtime = TritterRuntime::new(device)?;
/// let matmul = TernaryMatmul::new(&runtime);
/// let output = matmul.forward(&weights, &input)?;
/// ```
pub mod tritter {
    pub use tritter_accel::*;
}

// =============================================================================
// ECOSYSTEM INFO
// =============================================================================

/// Information about the rust-ai ecosystem crates.
///
/// Provides version information and capability detection for all ecosystem crates.
#[derive(Debug, Clone)]
pub struct EcosystemInfo {
    /// peft-rs version
    pub peft_version: &'static str,
    /// qlora-rs version
    pub qlora_version: &'static str,
    /// unsloth-rs version
    pub unsloth_version: &'static str,
    /// axolotl-rs version
    pub axolotl_version: &'static str,
    /// bitnet-quantize version
    pub bitnet_version: &'static str,
    /// trit-vsa version
    pub trit_version: &'static str,
    /// vsa-optim-rs version
    pub vsa_optim_version: &'static str,
    /// tritter-accel version
    pub tritter_version: &'static str,
}

impl Default for EcosystemInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl EcosystemInfo {
    /// Get ecosystem version information.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            peft_version: "1.0",
            qlora_version: "1.0",
            unsloth_version: "1.0",
            axolotl_version: "1.1",
            bitnet_version: "0.1",
            trit_version: "0.1",
            vsa_optim_version: "0.1",
            tritter_version: "0.1",
        }
    }

    /// List all ecosystem crate names.
    #[must_use]
    pub const fn crate_names() -> &'static [&'static str] {
        &[
            "peft-rs",
            "qlora-rs",
            "unsloth-rs",
            "axolotl-rs",
            "bitnet-quantize",
            "trit-vsa",
            "vsa-optim-rs",
            "tritter-accel",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ecosystem_info() {
        let info = EcosystemInfo::new();
        assert!(!info.peft_version.is_empty());
        assert!(!info.qlora_version.is_empty());
        assert!(!info.tritter_version.is_empty());
    }

    #[test]
    fn test_crate_names() {
        let names = EcosystemInfo::crate_names();
        assert_eq!(names.len(), 8);
        assert!(names.contains(&"peft-rs"));
        assert!(names.contains(&"tritter-accel"));
    }
}
