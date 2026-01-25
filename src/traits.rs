// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Common traits for the rust-ai ecosystem.
//!
//! This module defines shared trait interfaces that enable consistent APIs
//! across all rust-ai crates. Implementing these traits ensures interoperability
//! and allows crates to be composed together seamlessly.
//!
//! ## Core Traits
//!
//! - [`ValidatableConfig`] - Configuration validation interface
//! - [`Quantize`] - Tensor quantization (full precision → quantized)
//! - [`Dequantize`] - Tensor dequantization (quantized → full precision)
//! - [`GpuDispatchable`] - GPU/CPU kernel dispatch pattern
//!
//! ## Implementation Guidelines
//!
//! When implementing these traits:
//!
//! 1. **Validation**: Use `ValidatableConfig::validate()` in constructors
//! 2. **GPU-first**: `GpuDispatchable` should prefer GPU, warn on CPU
//! 3. **Error handling**: Return `CoreError` variants appropriately

use crate::error::{CoreError, Result};
use candle_core::{Device, Tensor};

/// Configuration validation trait.
///
/// All configuration structs should implement this trait to provide
/// consistent validation across the ecosystem.
///
/// # Example
///
/// ```rust
/// use rust_ai_core::{ValidatableConfig, CoreError, Result};
///
/// #[derive(Clone)]
/// struct LoraConfig {
///     rank: usize,
///     alpha: f32,
/// }
///
/// impl ValidatableConfig for LoraConfig {
///     fn validate(&self) -> Result<()> {
///         if self.rank == 0 {
///             return Err(CoreError::invalid_config("rank must be > 0"));
///         }
///         if self.alpha <= 0.0 {
///             return Err(CoreError::invalid_config("alpha must be positive"));
///         }
///         Ok(())
///     }
/// }
/// ```
pub trait ValidatableConfig: Clone + Send + Sync {
    /// Validate the configuration parameters.
    ///
    /// # Returns
    ///
    /// `Ok(())` if configuration is valid.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::InvalidConfig` if validation fails.
    fn validate(&self) -> Result<()>;
}

/// Tensor quantization trait.
///
/// Converts full-precision tensors to quantized representation.
/// Implementations may use various quantization schemes (NF4, FP4, ternary, etc.).
///
/// # Type Parameters
///
/// - `Q`: The quantized tensor type (e.g., `QuantizedTensor`, `TernaryVector`)
///
/// # Example
///
/// ```rust,ignore
/// use rust_ai_core::Quantize;
///
/// struct Nf4Quantizer;
///
/// impl Quantize<Nf4Tensor> for Nf4Quantizer {
///     fn quantize(&self, tensor: &Tensor, device: &Device) -> Result<Nf4Tensor> {
///         // Quantize to NF4 format
///     }
/// }
/// ```
pub trait Quantize<Q>: Send + Sync {
    /// Quantize a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Full-precision input tensor
    /// * `device` - Target device for the quantized output
    ///
    /// # Returns
    ///
    /// Quantized representation of the input tensor.
    ///
    /// # Errors
    ///
    /// May return errors for unsupported dtypes, shapes, or device issues.
    fn quantize(&self, tensor: &Tensor, device: &Device) -> Result<Q>;
}

/// Tensor dequantization trait.
///
/// Converts quantized tensors back to full precision for computation.
///
/// # Type Parameters
///
/// - `Q`: The quantized tensor type
///
/// # Example
///
/// ```rust,ignore
/// use rust_ai_core::Dequantize;
///
/// impl Dequantize<Nf4Tensor> for Nf4Quantizer {
///     fn dequantize(&self, quantized: &Nf4Tensor, device: &Device) -> Result<Tensor> {
///         // Restore to f32/f16/bf16
///     }
/// }
/// ```
pub trait Dequantize<Q>: Send + Sync {
    /// Dequantize a tensor.
    ///
    /// # Arguments
    ///
    /// * `quantized` - Quantized input tensor
    /// * `device` - Target device for the dequantized output
    ///
    /// # Returns
    ///
    /// Full-precision tensor.
    ///
    /// # Errors
    ///
    /// May return errors for corrupted quantized data or device issues.
    fn dequantize(&self, quantized: &Q, device: &Device) -> Result<Tensor>;
}

/// GPU/CPU dispatch trait for operations with both implementations.
///
/// This trait enables the CUDA-first pattern: operations that have both
/// GPU (`CubeCL`) and CPU implementations should implement this trait to
/// automatically route to the appropriate backend.
///
/// # Design Pattern
///
/// ```rust,ignore
/// use rust_ai_core::{GpuDispatchable, warn_if_cpu};
///
/// struct FlashAttention;
///
/// impl GpuDispatchable for FlashAttention {
///     type Input = (Tensor, Tensor, Tensor); // Q, K, V
///     type Output = Tensor;
///
///     fn dispatch_gpu(&self, input: &Self::Input, device: &Device) -> Result<Self::Output> {
///         // CubeCL Flash Attention kernel
///     }
///
///     fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> Result<Self::Output> {
///         // Candle-based fallback
///         warn_if_cpu(device, "unsloth-rs");
///         // ... fallback implementation
///     }
/// }
/// ```
pub trait GpuDispatchable: Send + Sync {
    /// Input type for the operation.
    type Input;

    /// Output type for the operation.
    type Output;

    /// Execute operation on GPU using `CubeCL` kernels.
    ///
    /// # Arguments
    ///
    /// * `input` - Operation input
    /// * `device` - Must be a CUDA device
    ///
    /// # Errors
    ///
    /// Returns `CoreError::KernelError` if kernel execution fails.
    fn dispatch_gpu(&self, input: &Self::Input, device: &Device) -> Result<Self::Output>;

    /// Execute operation on CPU (fallback).
    ///
    /// This should emit a warning via `warn_if_cpu()` before execution.
    ///
    /// # Arguments
    ///
    /// * `input` - Operation input
    /// * `device` - CPU device
    ///
    /// # Errors
    ///
    /// Returns appropriate error if operation fails.
    fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> Result<Self::Output>;

    /// Automatically dispatch to GPU or CPU based on device.
    ///
    /// This is the primary entry point. It checks the device type and
    /// routes to the appropriate implementation.
    ///
    /// # Arguments
    ///
    /// * `input` - Operation input
    /// * `device` - Target device (CUDA or CPU)
    ///
    /// # Returns
    ///
    /// Operation result from GPU or CPU path.
    fn dispatch(&self, input: &Self::Input, device: &Device) -> Result<Self::Output> {
        match device {
            Device::Cuda(_) => self.dispatch_gpu(input, device),
            Device::Cpu => self.dispatch_cpu(input, device),
            #[allow(unreachable_patterns)]
            _ => Err(CoreError::device_not_available(format!("{device:?}"))),
        }
    }

    /// Check if GPU dispatch is available for this operation.
    ///
    /// Default implementation checks if CUDA feature is enabled and
    /// a CUDA device is available.
    fn gpu_available(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            matches!(Device::cuda_if_available(0), Ok(Device::Cuda(_)))
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestConfig {
        value: i32,
    }

    impl ValidatableConfig for TestConfig {
        fn validate(&self) -> Result<()> {
            if self.value < 0 {
                return Err(CoreError::invalid_config("value must be non-negative"));
            }
            Ok(())
        }
    }

    #[test]
    fn test_validatable_config() {
        let valid = TestConfig { value: 10 };
        assert!(valid.validate().is_ok());

        let invalid = TestConfig { value: -1 };
        assert!(invalid.validate().is_err());
    }
}
