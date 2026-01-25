// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Data type utilities and precision helpers.
//!
//! ## Why This Module Exists
//!
//! Working with multiple data types (f32, f16, bf16, quantized) is common in ML workloads.
//! This module provides:
//!
//! 1. **Type queries**: Check dtype properties without pattern matching everywhere
//! 2. **Conversion helpers**: Safely convert between compatible dtypes
//! 3. **Precision utilities**: Choose appropriate dtypes for different operations
//!
//! ## Design Decisions
//!
//! - **Extension trait pattern**: `DTypeExt` extends `candle_core::DType` rather than
//!   wrapping it, providing zero-cost access to additional methods.
//!
//! - **No implicit conversions**: All dtype changes are explicit. Silent precision
//!   loss causes subtle training bugs that are hard to diagnose.

use candle_core::DType;

/// Get the size in bytes for a single element of the given dtype.
///
/// ## Arguments
///
/// * `dtype` - The data type to query
///
/// ## Returns
///
/// Number of bytes per element.
///
/// ## Why This Function
///
/// Memory calculations require knowing element sizes. This centralizes the logic
/// rather than scattering `dtype.size_in_bytes()` calls throughout the codebase.
///
/// ## Example
///
/// ```rust
/// use rust_ai_core::bytes_per_element;
/// use candle_core::DType;
///
/// assert_eq!(bytes_per_element(DType::F32), 4);
/// assert_eq!(bytes_per_element(DType::BF16), 2);
/// ```
#[must_use]
pub fn bytes_per_element(dtype: DType) -> usize {
    dtype.size_in_bytes()
}

/// Check if a dtype is a floating-point type.
///
/// ## Arguments
///
/// * `dtype` - The data type to check
///
/// ## Returns
///
/// `true` for F16, BF16, F32, F64; `false` for integer types.
///
/// ## Why This Function
///
/// Many ML operations only make sense on floating-point tensors. This check
/// prevents runtime errors from attempting operations like softmax on integers.
///
/// ## Example
///
/// ```rust
/// use rust_ai_core::is_floating_point;
/// use candle_core::DType;
///
/// assert!(is_floating_point(DType::F32));
/// assert!(is_floating_point(DType::BF16));
/// assert!(!is_floating_point(DType::I64));
/// ```
#[must_use]
pub fn is_floating_point(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
}

/// Extension trait adding utility methods to `candle_core::DType`.
///
/// ## Why An Extension Trait
///
/// Extension traits allow adding methods to external types without wrapping them.
/// Users can call these methods naturally on any `DType` value after importing
/// the trait.
///
/// ## Example
///
/// ```rust
/// use rust_ai_core::DTypeExt;
/// use candle_core::DType;
///
/// let dtype = DType::BF16;
/// assert!(dtype.is_half_precision());
/// assert!(dtype.is_training_dtype());
/// ```
pub trait DTypeExt {
    /// Check if this dtype is a half-precision float (f16 or bf16).
    ///
    /// ## Why This Method
    ///
    /// Half-precision types require special handling for numerical stability
    /// (loss scaling, careful accumulation). This check identifies when those
    /// precautions are needed.
    fn is_half_precision(&self) -> bool;

    /// Check if this dtype is suitable for training (f16, bf16, f32).
    ///
    /// ## Why This Method
    ///
    /// Training requires floating-point math. f64 is technically valid but
    /// rarely used due to memory cost. Integer dtypes cannot be used for
    /// gradient computation.
    fn is_training_dtype(&self) -> bool;

    /// Check if this dtype is an integer type.
    fn is_integer(&self) -> bool;

    /// Get a human-readable name for this dtype.
    ///
    /// ## Why This Method
    ///
    /// Error messages and logs are more readable with "f32" than "`DType::F32`".
    fn name(&self) -> &'static str;

    /// Get the recommended accumulator dtype for this compute dtype.
    ///
    /// ## Why This Method
    ///
    /// When summing many values (e.g., in matrix multiplication), using a
    /// higher-precision accumulator prevents numerical overflow. This returns
    /// the recommended accumulator for each compute dtype:
    ///
    /// - f16/bf16 -> f32 (standard mixed-precision pattern)
    /// - f32 -> f32 (already high precision)
    /// - f64 -> f64 (already highest precision)
    fn accumulator_dtype(&self) -> DType;
}

impl DTypeExt for DType {
    fn is_half_precision(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16)
    }

    fn is_training_dtype(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32)
    }

    fn is_integer(&self) -> bool {
        matches!(
            self,
            DType::U8 | DType::U32 | DType::I16 | DType::I32 | DType::I64
        )
    }

    fn name(&self) -> &'static str {
        match self {
            DType::U8 => "u8",
            DType::U32 => "u32",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            // Exotic formats (MX, F8, etc.)
            _ => "exotic",
        }
    }

    #[allow(clippy::match_same_arms)] // Explicit arms for documentation; wildcard for future dtypes
    fn accumulator_dtype(&self) -> DType {
        match self {
            DType::F16 | DType::BF16 | DType::F32 => DType::F32,
            DType::F64 => DType::F64,
            // Integer accumulators: use i64 for safety
            DType::U8 | DType::U32 | DType::I16 | DType::I32 | DType::I64 => DType::I64,
            // Exotic formats: accumulate in f32
            _ => DType::F32,
        }
    }
}

/// Precision mode for mixed-precision training.
///
/// ## Why This Enum
///
/// Mixed-precision training uses different dtypes for different operations:
/// - Forward pass: often f16/bf16 for speed
/// - Accumulation: f32 for numerical stability
/// - Master weights: f32 for gradient updates
///
/// This enum captures common configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrecisionMode {
    /// Full precision (f32 everywhere).
    ///
    /// Use for debugging, final fine-tuning, or when VRAM is not constrained.
    #[default]
    Full,

    /// `BFloat16` compute with f32 accumulation.
    ///
    /// Recommended for training on Ampere+ GPUs (RTX 30xx, A100, H100).
    /// BF16 has the same exponent range as f32, making it numerically stable.
    Bf16,

    /// Float16 compute with f32 accumulation.
    ///
    /// Use on older GPUs (Turing, Volta) or when BF16 is not available.
    /// Requires loss scaling due to limited exponent range.
    Fp16,
}

impl PrecisionMode {
    /// Get the compute dtype for this precision mode.
    #[must_use]
    pub fn compute_dtype(&self) -> DType {
        match self {
            Self::Full => DType::F32,
            Self::Bf16 => DType::BF16,
            Self::Fp16 => DType::F16,
        }
    }

    /// Get the master weight dtype for this precision mode.
    ///
    /// ## Why Master Weights
    ///
    /// In mixed-precision training, optimizer states and gradient updates
    /// happen in f32 to maintain precision. The f32 weights are then cast
    /// to the compute dtype for forward/backward passes.
    #[must_use]
    pub fn master_weight_dtype(&self) -> DType {
        DType::F32
    }

    /// Check if this precision mode requires loss scaling.
    ///
    /// ## Why Loss Scaling
    ///
    /// FP16 has limited exponent range (-14 to 15). Small gradients underflow
    /// to zero. Loss scaling multiplies the loss before backward pass, then
    /// divides gradients after, keeping values in FP16's representable range.
    #[must_use]
    pub fn requires_loss_scaling(&self) -> bool {
        matches!(self, Self::Fp16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_per_element() {
        assert_eq!(bytes_per_element(DType::F32), 4);
        assert_eq!(bytes_per_element(DType::F16), 2);
        assert_eq!(bytes_per_element(DType::BF16), 2);
        assert_eq!(bytes_per_element(DType::F64), 8);
        assert_eq!(bytes_per_element(DType::U8), 1);
        assert_eq!(bytes_per_element(DType::I64), 8);
    }

    #[test]
    fn test_is_floating_point() {
        assert!(is_floating_point(DType::F32));
        assert!(is_floating_point(DType::F16));
        assert!(is_floating_point(DType::BF16));
        assert!(is_floating_point(DType::F64));
        assert!(!is_floating_point(DType::U8));
        assert!(!is_floating_point(DType::I64));
    }

    #[test]
    fn test_dtype_ext() {
        assert!(DType::F16.is_half_precision());
        assert!(DType::BF16.is_half_precision());
        assert!(!DType::F32.is_half_precision());

        assert!(DType::F32.is_training_dtype());
        assert!(DType::BF16.is_training_dtype());
        assert!(!DType::I64.is_training_dtype());

        assert!(DType::I32.is_integer());
        assert!(!DType::F32.is_integer());

        assert_eq!(DType::F32.name(), "f32");
        assert_eq!(DType::BF16.name(), "bf16");
    }

    #[test]
    fn test_accumulator_dtype() {
        assert_eq!(DType::F16.accumulator_dtype(), DType::F32);
        assert_eq!(DType::BF16.accumulator_dtype(), DType::F32);
        assert_eq!(DType::F32.accumulator_dtype(), DType::F32);
        assert_eq!(DType::I32.accumulator_dtype(), DType::I64);
    }

    #[test]
    fn test_precision_mode() {
        assert_eq!(PrecisionMode::Full.compute_dtype(), DType::F32);
        assert_eq!(PrecisionMode::Bf16.compute_dtype(), DType::BF16);
        assert_eq!(PrecisionMode::Fp16.compute_dtype(), DType::F16);

        assert!(!PrecisionMode::Bf16.requires_loss_scaling());
        assert!(PrecisionMode::Fp16.requires_loss_scaling());

        // All modes use f32 master weights
        assert_eq!(PrecisionMode::Full.master_weight_dtype(), DType::F32);
        assert_eq!(PrecisionMode::Bf16.master_weight_dtype(), DType::F32);
    }
}
