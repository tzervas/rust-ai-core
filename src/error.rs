// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Unified error types for the rust-ai ecosystem.
//!
//! This module provides common error types that are shared across all rust-ai crates.
//! Each crate can extend these with domain-specific variants while maintaining
//! compatibility for error conversion.
//!
//! ## Error Hierarchy
//!
//! ```text
//! CoreError
//! ├── InvalidConfig       - Configuration validation failures
//! ├── ShapeMismatch       - Tensor shape incompatibilities
//! ├── DimensionMismatch   - Dimension count mismatches
//! ├── DeviceNotAvailable  - Requested device unavailable
//! ├── DeviceMismatch      - Tensors on different devices
//! ├── OutOfMemory         - GPU/CPU memory exhausted
//! ├── KernelError         - GPU kernel launch/execution failure
//! ├── NotImplemented      - Feature not yet implemented
//! ├── Io                  - File/network I/O errors
//! └── Candle              - Underlying Candle errors
//! ```
//!
//! ## Crate-Specific Errors
//!
//! Crates should define their own error types that wrap `CoreError`:
//!
//! ```rust
//! use rust_ai_core::CoreError;
//! use thiserror::Error;
//!
//! #[derive(Error, Debug)]
//! pub enum MyError {
//!     #[error("adapter not found: {0}")]
//!     AdapterNotFound(String),
//!     
//!     #[error(transparent)]
//!     Core(#[from] CoreError),
//! }
//! ```

use thiserror::Error;

/// Result type alias for rust-ai-core operations.
pub type Result<T> = std::result::Result<T, CoreError>;

/// Core errors shared across the rust-ai ecosystem.
///
/// These errors represent common failure modes that can occur in any crate.
/// Domain-specific errors should wrap these variants.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum CoreError {
    /// Invalid configuration parameter.
    ///
    /// Raised when a configuration value is out of bounds, incompatible,
    /// or otherwise invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Tensor shape mismatch.
    ///
    /// Raised when an operation expects tensors of specific shapes but
    /// receives tensors with incompatible shapes.
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape received.
        actual: Vec<usize>,
    },

    /// Dimension count mismatch.
    ///
    /// Raised when tensors have different numbers of dimensions.
    #[error("dimension mismatch: {message}")]
    DimensionMismatch {
        /// Descriptive error message.
        message: String,
    },

    /// Requested device not available.
    ///
    /// Raised when attempting to use a device (e.g., CUDA:1) that doesn't
    /// exist or isn't accessible.
    #[error("device not available: {device}")]
    DeviceNotAvailable {
        /// Description of the unavailable device.
        device: String,
    },

    /// Device mismatch between tensors.
    ///
    /// Raised when an operation requires tensors on the same device but
    /// they reside on different devices.
    #[error("device mismatch: tensors must be on the same device")]
    DeviceMismatch,

    /// Out of memory.
    ///
    /// Raised when GPU or CPU memory allocation fails.
    #[error("out of memory: {message}")]
    OutOfMemory {
        /// Descriptive error message.
        message: String,
    },

    /// GPU kernel error.
    ///
    /// Raised when a CUDA/CubeCL kernel fails to launch or execute.
    #[error("kernel error: {message}")]
    KernelError {
        /// Descriptive error message.
        message: String,
    },

    /// Feature not implemented.
    ///
    /// Raised when a requested feature or code path is not yet implemented.
    #[error("not implemented: {feature}")]
    NotImplemented {
        /// Description of the unimplemented feature.
        feature: String,
    },

    /// I/O error.
    ///
    /// Raised for file operations, network errors, serialization failures, etc.
    #[error("I/O error: {0}")]
    Io(String),

    /// Underlying Candle error.
    ///
    /// Wraps errors from the Candle tensor library.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

impl CoreError {
    /// Create an invalid configuration error.
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }

    /// Create a shape mismatch error.
    pub fn shape_mismatch(expected: impl Into<Vec<usize>>, actual: impl Into<Vec<usize>>) -> Self {
        Self::ShapeMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create a dimension mismatch error.
    pub fn dim_mismatch(msg: impl Into<String>) -> Self {
        Self::DimensionMismatch {
            message: msg.into(),
        }
    }

    /// Create a device not available error.
    pub fn device_not_available(device: impl Into<String>) -> Self {
        Self::DeviceNotAvailable {
            device: device.into(),
        }
    }

    /// Create an out of memory error.
    pub fn oom(msg: impl Into<String>) -> Self {
        Self::OutOfMemory {
            message: msg.into(),
        }
    }

    /// Create a kernel error.
    pub fn kernel(msg: impl Into<String>) -> Self {
        Self::KernelError {
            message: msg.into(),
        }
    }

    /// Create a not implemented error.
    pub fn not_implemented(feature: impl Into<String>) -> Self {
        Self::NotImplemented {
            feature: feature.into(),
        }
    }

    /// Create an I/O error.
    pub fn io(msg: impl Into<String>) -> Self {
        Self::Io(msg.into())
    }
}

impl From<std::io::Error> for CoreError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CoreError::invalid_config("rank must be positive");
        assert_eq!(err.to_string(), "invalid configuration: rank must be positive");

        let err = CoreError::shape_mismatch(vec![2, 3], vec![3, 2]);
        assert!(err.to_string().contains("shape mismatch"));

        let err = CoreError::device_not_available("CUDA:5");
        assert!(err.to_string().contains("CUDA:5"));
    }

    #[test]
    fn test_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let core_err: CoreError = io_err.into();
        assert!(matches!(core_err, CoreError::Io(_)));
    }
}
