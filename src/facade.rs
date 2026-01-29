// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! High-level unified API for the rust-ai ecosystem.
//!
//! The `RustAI` struct provides a facade over all rust-ai ecosystem crates,
//! offering a simplified interface for common AI engineering tasks.
//!
//! ## Design Philosophy
//!
//! Rather than requiring users to understand each individual crate's API,
//! `RustAI` provides high-level workflows that compose the right crates
//! automatically based on the task at hand.
//!
//! ## Example
//!
//! ```rust,ignore
//! use rust_ai_core::{RustAI, RustAIConfig};
//!
//! // Initialize with sensible defaults
//! let ai = RustAI::new(RustAIConfig::default())?;
//!
//! // Fine-tune a model with LoRA
//! let config = ai.finetune()
//!     .model("meta-llama/Llama-2-7b")
//!     .rank(64)
//!     .build()?;
//!
//! // Quantize for deployment
//! let quant_config = ai.quantize()
//!     .method(QuantizeMethod::Nf4)
//!     .bits(4)
//!     .build();
//! ```

use crate::device::{get_device, DeviceConfig};
use crate::ecosystem::EcosystemInfo;
use crate::error::{CoreError, Result};
use candle_core::Device;

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for the RustAI facade.
///
/// This centralizes all configuration options and provides sensible defaults
/// for the unified API.
#[derive(Debug, Clone)]
pub struct RustAIConfig {
    /// Device configuration (CUDA selection, CPU fallback)
    pub device: DeviceConfig,
    /// Enable verbose logging
    pub verbose: bool,
    /// Memory limit in bytes (0 = no limit)
    pub memory_limit: usize,
}

impl Default for RustAIConfig {
    fn default() -> Self {
        Self {
            device: DeviceConfig::from_env(),
            verbose: false,
            memory_limit: 0,
        }
    }
}

impl RustAIConfig {
    /// Create a new configuration with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set verbose mode.
    #[must_use]
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set memory limit in bytes.
    #[must_use]
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = limit;
        self
    }

    /// Force CPU execution.
    #[must_use]
    pub fn with_cpu(mut self) -> Self {
        self.device = self.device.with_force_cpu(true);
        self
    }

    /// Select a specific CUDA device.
    #[must_use]
    pub fn with_cuda_device(mut self, ordinal: usize) -> Self {
        self.device = self.device.with_cuda_device(ordinal);
        self
    }
}

// =============================================================================
// RUST-AI FACADE
// =============================================================================

/// Unified facade for the rust-ai ecosystem.
///
/// Provides high-level APIs that orchestrate multiple ecosystem crates
/// to accomplish common AI engineering tasks.
///
/// ## Capabilities
///
/// | Workflow | Description |
/// |----------|-------------|
/// | `finetune()` | LoRA, DoRA, AdaLoRA adapter creation |
/// | `quantize()` | 4-bit (NF4/FP4) and 1.58-bit (BitNet) quantization |
/// | `vsa()` | VSA-based operations and optimization |
/// | `train()` | YAML-driven training pipelines |
///
/// ## Example
///
/// ```rust
/// use rust_ai_core::{RustAI, RustAIConfig};
///
/// let config = RustAIConfig::new()
///     .with_verbose(true);
///
/// let ai = RustAI::new(config).unwrap();
/// println!("Device: {:?}", ai.device());
/// println!("Ecosystem: {:?}", ai.ecosystem());
/// ```
pub struct RustAI {
    config: RustAIConfig,
    device: Device,
    ecosystem: EcosystemInfo,
}

impl RustAI {
    /// Create a new RustAI instance.
    ///
    /// Initializes the device and ecosystem information.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration options
    ///
    /// # Returns
    ///
    /// A configured `RustAI` instance.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::DeviceNotAvailable` if device initialization fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rust_ai_core::{RustAI, RustAIConfig};
    ///
    /// let ai = RustAI::new(RustAIConfig::default())?;
    /// # Ok::<(), rust_ai_core::CoreError>(())
    /// ```
    pub fn new(config: RustAIConfig) -> Result<Self> {
        let device = get_device(&config.device)?;
        let ecosystem = EcosystemInfo::new();

        if config.verbose {
            tracing::info!("RustAI initialized");
            tracing::info!("Device: {:?}", device);
            tracing::info!("Ecosystem crates: {:?}", EcosystemInfo::crate_names());
        }

        Ok(Self {
            config,
            device,
            ecosystem,
        })
    }

    /// Get the active device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get ecosystem information.
    #[must_use]
    pub fn ecosystem(&self) -> &EcosystemInfo {
        &self.ecosystem
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &RustAIConfig {
        &self.config
    }

    /// Check if CUDA is available and active.
    #[must_use]
    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }

    /// Start a fine-tuning workflow.
    ///
    /// Returns a builder for configuring and running fine-tuning.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = ai.finetune()
    ///     .model("meta-llama/Llama-2-7b")
    ///     .adapter(AdapterType::Lora)
    ///     .rank(64)
    ///     .build()?;
    /// ```
    #[must_use]
    pub fn finetune(&self) -> FinetuneBuilder<'_> {
        FinetuneBuilder::new(self)
    }

    /// Start a quantization workflow.
    ///
    /// Returns a builder for configuring and running quantization.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = ai.quantize()
    ///     .method(QuantizeMethod::Nf4)
    ///     .bits(4)
    ///     .build();
    /// ```
    #[must_use]
    pub fn quantize(&self) -> QuantizeBuilder<'_> {
        QuantizeBuilder::new(self)
    }

    /// Start a VSA workflow.
    ///
    /// Returns a builder for VSA-based operations.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = ai.vsa()
    ///     .dimension(10000)
    ///     .build();
    /// ```
    #[must_use]
    pub fn vsa(&self) -> VsaBuilder<'_> {
        VsaBuilder::new(self)
    }

    /// Start an Axolotl training pipeline.
    ///
    /// Returns a builder for YAML-driven training configuration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = ai.train()
    ///     .config_file("config.yaml")
    ///     .build()?;
    /// ```
    #[must_use]
    pub fn train(&self) -> TrainBuilder<'_> {
        TrainBuilder::new(self)
    }

    /// Get information about the RustAI environment.
    ///
    /// Returns a struct containing version info, ecosystem, and device details.
    #[must_use]
    pub fn info(&self) -> RustAIInfo {
        RustAIInfo {
            version: crate::VERSION.to_string(),
            device: format!("{:?}", self.device),
            ecosystem_crates: EcosystemInfo::crate_names()
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            cuda_available: self.is_cuda(),
            memory_limit: self.config.memory_limit,
        }
    }
}

/// Information about the RustAI environment.
#[derive(Debug, Clone)]
pub struct RustAIInfo {
    /// Crate version
    pub version: String,
    /// Active device description
    pub device: String,
    /// List of ecosystem crates
    pub ecosystem_crates: Vec<String>,
    /// Whether CUDA is available
    pub cuda_available: bool,
    /// Memory limit (0 = unlimited)
    pub memory_limit: usize,
}

// =============================================================================
// FINE-TUNING WORKFLOW
// =============================================================================

/// Builder for fine-tuning workflows.
pub struct FinetuneBuilder<'a> {
    ai: &'a RustAI,
    model_path: Option<String>,
    adapter_type: AdapterType,
    rank: usize,
    alpha: f32,
    dropout: f32,
    target_modules: Vec<String>,
}

/// Type of PEFT adapter.
#[derive(Debug, Clone, Copy, Default)]
pub enum AdapterType {
    /// Low-Rank Adaptation
    #[default]
    Lora,
    /// Weight-Decomposed Low-Rank Adaptation
    Dora,
    /// Adaptive Budget Low-Rank Adaptation
    AdaLora,
}

impl<'a> FinetuneBuilder<'a> {
    fn new(ai: &'a RustAI) -> Self {
        Self {
            ai,
            model_path: None,
            adapter_type: AdapterType::Lora,
            rank: 64,
            alpha: 16.0,
            dropout: 0.1,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
        }
    }

    /// Set the model path or identifier.
    #[must_use]
    pub fn model(mut self, path: impl Into<String>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Set the adapter type.
    #[must_use]
    pub fn adapter(mut self, adapter: AdapterType) -> Self {
        self.adapter_type = adapter;
        self
    }

    /// Set the LoRA rank.
    #[must_use]
    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    /// Set the LoRA alpha scaling factor.
    #[must_use]
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the dropout rate.
    #[must_use]
    pub fn dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set the target module names to adapt.
    #[must_use]
    pub fn target_modules(mut self, modules: Vec<String>) -> Self {
        self.target_modules = modules;
        self
    }

    /// Build the fine-tuning configuration.
    ///
    /// # Errors
    ///
    /// Returns error if model path is not specified.
    pub fn build(self) -> Result<FinetuneConfig> {
        let model_path = self.model_path.ok_or_else(|| {
            CoreError::invalid_config("model path is required for fine-tuning")
        })?;

        Ok(FinetuneConfig {
            model_path,
            adapter_type: self.adapter_type,
            rank: self.rank,
            alpha: self.alpha,
            dropout: self.dropout,
            target_modules: self.target_modules,
        })
    }
}

/// Configuration for fine-tuning.
#[derive(Debug, Clone)]
pub struct FinetuneConfig {
    /// Path to the model
    pub model_path: String,
    /// Type of adapter
    pub adapter_type: AdapterType,
    /// LoRA rank
    pub rank: usize,
    /// LoRA alpha
    pub alpha: f32,
    /// Dropout rate
    pub dropout: f32,
    /// Target modules
    pub target_modules: Vec<String>,
}

// =============================================================================
// QUANTIZATION WORKFLOW
// =============================================================================

/// Builder for quantization workflows.
pub struct QuantizeBuilder<'a> {
    #[allow(dead_code)]
    ai: &'a RustAI,
    method: QuantizeMethod,
    bits: u8,
    group_size: usize,
}

/// Quantization method.
#[derive(Debug, Clone, Copy, Default)]
pub enum QuantizeMethod {
    /// NF4 (Normal Float 4-bit) - used in QLoRA
    #[default]
    Nf4,
    /// FP4 (Floating Point 4-bit)
    Fp4,
    /// BitNet 1.58-bit ternary quantization
    BitNet,
    /// Standard INT8 quantization
    Int8,
}

impl<'a> QuantizeBuilder<'a> {
    fn new(ai: &'a RustAI) -> Self {
        Self {
            ai,
            method: QuantizeMethod::Nf4,
            bits: 4,
            group_size: 64,
        }
    }

    /// Set the quantization method.
    #[must_use]
    pub fn method(mut self, method: QuantizeMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the number of bits (for non-BitNet methods).
    #[must_use]
    pub fn bits(mut self, bits: u8) -> Self {
        self.bits = bits;
        self
    }

    /// Set the group size for group-wise quantization.
    #[must_use]
    pub fn group_size(mut self, size: usize) -> Self {
        self.group_size = size;
        self
    }

    /// Build the quantization configuration.
    #[must_use]
    pub fn build(self) -> QuantizeConfig {
        QuantizeConfig {
            method: self.method,
            bits: self.bits,
            group_size: self.group_size,
        }
    }
}

/// Configuration for quantization.
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// Quantization method
    pub method: QuantizeMethod,
    /// Number of bits
    pub bits: u8,
    /// Group size
    pub group_size: usize,
}

// =============================================================================
// VSA WORKFLOW
// =============================================================================

/// Builder for VSA workflows.
pub struct VsaBuilder<'a> {
    #[allow(dead_code)]
    ai: &'a RustAI,
    dimension: usize,
}

impl<'a> VsaBuilder<'a> {
    fn new(ai: &'a RustAI) -> Self {
        Self {
            ai,
            dimension: 10000,
        }
    }

    /// Set the VSA dimension.
    #[must_use]
    pub fn dimension(mut self, dim: usize) -> Self {
        self.dimension = dim;
        self
    }

    /// Build the VSA configuration.
    #[must_use]
    pub fn build(self) -> VsaConfig {
        VsaConfig {
            dimension: self.dimension,
        }
    }
}

/// Configuration for VSA operations.
#[derive(Debug, Clone)]
pub struct VsaConfig {
    /// VSA dimension
    pub dimension: usize,
}

// =============================================================================
// TRAINING WORKFLOW
// =============================================================================

/// Builder for Axolotl training pipelines.
pub struct TrainBuilder<'a> {
    #[allow(dead_code)]
    ai: &'a RustAI,
    config_path: Option<String>,
}

impl<'a> TrainBuilder<'a> {
    fn new(ai: &'a RustAI) -> Self {
        Self {
            ai,
            config_path: None,
        }
    }

    /// Set the YAML configuration file path.
    #[must_use]
    pub fn config_file(mut self, path: impl Into<String>) -> Self {
        self.config_path = Some(path.into());
        self
    }

    /// Build the training configuration.
    ///
    /// # Errors
    ///
    /// Returns error if config file is not specified.
    pub fn build(self) -> Result<TrainConfig> {
        let config_path = self.config_path.ok_or_else(|| {
            CoreError::invalid_config("config file path is required")
        })?;

        Ok(TrainConfig { config_path })
    }
}

/// Configuration for Axolotl training.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Path to YAML configuration
    pub config_path: String,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rustai_config_default() {
        let config = RustAIConfig::default();
        assert!(!config.verbose);
        assert_eq!(config.memory_limit, 0);
    }

    #[test]
    fn test_rustai_config_builder() {
        let config = RustAIConfig::new()
            .with_verbose(true)
            .with_memory_limit(1024 * 1024 * 1024)
            .with_cpu();

        assert!(config.verbose);
        assert_eq!(config.memory_limit, 1024 * 1024 * 1024);
        assert!(config.device.force_cpu);
    }

    #[test]
    fn test_rustai_new() {
        let config = RustAIConfig::new().with_cpu();
        let ai = RustAI::new(config).unwrap();
        assert!(!ai.is_cuda());
        assert_eq!(EcosystemInfo::crate_names().len(), 8);
    }

    #[test]
    fn test_rustai_info() {
        let config = RustAIConfig::new().with_cpu();
        let ai = RustAI::new(config).unwrap();
        let info = ai.info();
        assert!(!info.version.is_empty());
        assert!(!info.cuda_available);
        assert_eq!(info.ecosystem_crates.len(), 8);
    }

    #[test]
    fn test_finetune_builder() {
        let config = RustAIConfig::new().with_cpu();
        let ai = RustAI::new(config).unwrap();

        let finetune_config = ai.finetune()
            .model("test-model")
            .rank(32)
            .alpha(8.0)
            .build()
            .unwrap();

        assert_eq!(finetune_config.model_path, "test-model");
        assert_eq!(finetune_config.rank, 32);
        assert!((finetune_config.alpha - 8.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quantize_builder() {
        let config = RustAIConfig::new().with_cpu();
        let ai = RustAI::new(config).unwrap();

        let quant_config = ai.quantize()
            .method(QuantizeMethod::BitNet)
            .bits(2)
            .group_size(128)
            .build();

        assert!(matches!(quant_config.method, QuantizeMethod::BitNet));
        assert_eq!(quant_config.bits, 2);
        assert_eq!(quant_config.group_size, 128);
    }

    #[test]
    fn test_vsa_builder() {
        let config = RustAIConfig::new().with_cpu();
        let ai = RustAI::new(config).unwrap();

        let vsa_config = ai.vsa()
            .dimension(8192)
            .build();

        assert_eq!(vsa_config.dimension, 8192);
    }

    #[test]
    fn test_train_builder() {
        let config = RustAIConfig::new().with_cpu();
        let ai = RustAI::new(config).unwrap();

        let train_config = ai.train()
            .config_file("train.yaml")
            .build()
            .unwrap();

        assert_eq!(train_config.config_path, "train.yaml");
    }
}
