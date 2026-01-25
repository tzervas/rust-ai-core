// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Python bindings for rust-ai-core.
//!
//! This module provides `PyO3` bindings to expose rust-ai-core's utilities
//! to Python developers, enabling:
//!
//! - Memory estimation for AI training planning
//! - Device detection and configuration
//! - Data type utilities
//! - Logging configuration
//!
//! # Python Usage
//!
//! ```python
//! from rust_ai_core_bindings import (
//!     # Memory utilities
//!     estimate_tensor_bytes,
//!     estimate_attention_memory,
//!     create_memory_tracker,
//!     tracker_allocate,
//!     tracker_would_fit,
//!
//!     # Device utilities
//!     get_device_info,
//!     cuda_available,
//!
//!     # DType utilities
//!     bytes_per_dtype,
//!     is_floating_point,
//!
//!     # Logging
//!     init_logging,
//!
//!     # Version
//!     version,
//! )
//!
//! # Estimate memory for a large tensor
//! bytes_needed = estimate_tensor_bytes([1, 512, 4096], "f32")
//! print(f"Tensor needs {bytes_needed / 1024**2:.2f} MB")
//!
//! # Check if GPU is available
//! if cuda_available():
//!     device = get_device_info()
//!     print(f"Using device: {device['type']}")
//!
//! # Track memory usage during training
//! tracker = create_memory_tracker(limit_gb=8.0)
//! if tracker_would_fit(tracker, bytes_needed):
//!     tracker_allocate(tracker, bytes_needed)
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::useless_conversion)] // PyO3 macro generates these
#![allow(clippy::missing_errors_doc)] // Python bindings - errors are documented in docstrings
#![allow(clippy::needless_pass_by_value)] // PyO3 requires owned types for Python arguments

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::device::{get_device as rust_get_device, DeviceConfig};
use crate::dtype;
use crate::logging::{init_logging as rust_init_logging, LogConfig, LogLevel};
use crate::memory::{self, MemoryTracker};

// =============================================================================
// MEMORY TRACKER WRAPPER
// =============================================================================

/// Opaque handle for Python to reference a `MemoryTracker`.
#[pyclass(name = "MemoryTracker")]
#[derive(Clone)]
pub struct PyMemoryTracker {
    inner: Arc<Mutex<MemoryTracker>>,
}

// =============================================================================
// MEMORY ESTIMATION FUNCTIONS
// =============================================================================

/// Estimate memory required for a tensor with given shape and dtype.
///
/// # Arguments
/// * `shape` - List of dimension sizes (e.g., `[1, 512, 4096]`)
/// * `dtype` - Data type string: "f32", "f16", "bf16", "f64", "i32", "i64", "u8", "u32"
///
/// # Returns
/// Number of bytes required for the tensor.
///
/// # Example
/// ```python
/// bytes_needed = estimate_tensor_bytes([1, 512, 4096], "f32")
/// print(f"Tensor needs {bytes_needed / 1024**2:.2f} MB")
/// ```
#[pyfunction]
#[pyo3(signature = (shape, dtype="f32"))]
fn estimate_tensor_bytes(shape: Vec<usize>, dtype: &str) -> PyResult<usize> {
    let candle_dtype = parse_dtype(dtype)?;
    Ok(memory::estimate_tensor_bytes(&shape, candle_dtype))
}

/// Estimate memory for attention computation (O(n^2) attention scores).
///
/// # Arguments
/// * `batch_size` - Batch size
/// * `num_heads` - Number of attention heads
/// * `seq_len` - Sequence length
/// * `head_dim` - Dimension per head
/// * `dtype` - Data type string (default: "bf16")
///
/// # Returns
/// Total bytes for attention computation (Q, K, V, and attention scores).
///
/// # Example
/// ```python
/// # Estimate memory for 4K context with 32 heads
/// mem = estimate_attention_memory(1, 32, 4096, 128, "bf16")
/// print(f"Attention needs {mem / 1024**3:.2f} GB")
/// ```
#[pyfunction]
#[pyo3(signature = (batch_size, num_heads, seq_len, head_dim, dtype="bf16"))]
fn estimate_attention_memory(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    dtype: &str,
) -> PyResult<usize> {
    let candle_dtype = parse_dtype(dtype)?;
    Ok(memory::estimate_attention_memory(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        candle_dtype,
    ))
}

// =============================================================================
// MEMORY TRACKER FUNCTIONS
// =============================================================================

/// Create a memory tracker with a limit.
///
/// # Arguments
/// * `limit_gb` - Memory limit in gigabytes (default: 8.0)
/// * `overhead_factor` - Overhead multiplier for allocations (default: 1.1)
///
/// # Returns
/// A `MemoryTracker` handle for tracking allocations.
///
/// # Example
/// ```python
/// tracker = create_memory_tracker(limit_gb=8.0, overhead_factor=1.1)
/// ```
#[pyfunction]
#[pyo3(signature = (limit_gb=8.0, overhead_factor=1.1))]
fn create_memory_tracker(limit_gb: f64, overhead_factor: f64) -> PyMemoryTracker {
    let limit_bytes = (limit_gb * 1024.0 * 1024.0 * 1024.0) as usize;
    let tracker = MemoryTracker::with_limit(limit_bytes).with_overhead_factor(overhead_factor);
    PyMemoryTracker {
        inner: Arc::new(Mutex::new(tracker)),
    }
}

/// Check if an allocation would fit within the tracker's limit.
///
/// # Arguments
/// * `tracker` - The memory tracker handle
/// * `bytes` - Number of bytes to check
///
/// # Returns
/// `True` if the allocation would fit, `False` otherwise.
#[pyfunction]
fn tracker_would_fit(tracker: &PyMemoryTracker, bytes: usize) -> PyResult<bool> {
    let inner = tracker
        .inner
        .lock()
        .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
    Ok(inner.would_fit(bytes))
}

/// Record an allocation in the tracker.
///
/// # Arguments
/// * `tracker` - The memory tracker handle
/// * `bytes` - Number of bytes allocated
///
/// # Raises
/// `RuntimeError` if the allocation exceeds the limit.
#[pyfunction]
fn tracker_allocate(tracker: &PyMemoryTracker, bytes: usize) -> PyResult<()> {
    let inner = tracker
        .inner
        .lock()
        .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
    inner
        .allocate(bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("Allocation failed: {e}")))
}

/// Record a deallocation in the tracker.
///
/// # Arguments
/// * `tracker` - The memory tracker handle
/// * `bytes` - Number of bytes deallocated
#[pyfunction]
fn tracker_deallocate(tracker: &PyMemoryTracker, bytes: usize) -> PyResult<()> {
    let inner = tracker
        .inner
        .lock()
        .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
    inner.deallocate(bytes);
    Ok(())
}

/// Get current allocation from the tracker.
///
/// # Arguments
/// * `tracker` - The memory tracker handle
///
/// # Returns
/// Current allocated bytes.
#[pyfunction]
fn tracker_allocated_bytes(tracker: &PyMemoryTracker) -> PyResult<usize> {
    let inner = tracker
        .inner
        .lock()
        .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
    Ok(inner.allocated_bytes())
}

/// Get peak allocation from the tracker.
///
/// # Arguments
/// * `tracker` - The memory tracker handle
///
/// # Returns
/// Peak allocated bytes since creation.
#[pyfunction]
fn tracker_peak_bytes(tracker: &PyMemoryTracker) -> PyResult<usize> {
    let inner = tracker
        .inner
        .lock()
        .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
    Ok(inner.peak_bytes())
}

/// Get the memory limit from the tracker.
///
/// # Arguments
/// * `tracker` - The memory tracker handle
///
/// # Returns
/// Memory limit in bytes.
#[pyfunction]
fn tracker_limit_bytes(tracker: &PyMemoryTracker) -> PyResult<usize> {
    let inner = tracker
        .inner
        .lock()
        .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
    Ok(inner.limit_bytes())
}

/// Estimate bytes with overhead factor.
///
/// # Arguments
/// * `tracker` - The memory tracker handle
/// * `shape` - Tensor shape
/// * `dtype` - Data type string
///
/// # Returns
/// Estimated bytes including overhead.
#[pyfunction]
#[pyo3(signature = (tracker, shape, dtype="f32"))]
fn tracker_estimate_with_overhead(
    tracker: &PyMemoryTracker,
    shape: Vec<usize>,
    dtype: &str,
) -> PyResult<usize> {
    let candle_dtype = parse_dtype(dtype)?;
    let inner = tracker
        .inner
        .lock()
        .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
    Ok(inner.estimate_with_overhead(&shape, candle_dtype))
}

/// Reset the memory tracker to initial state.
///
/// # Arguments
/// * `tracker` - The memory tracker handle
#[pyfunction]
fn tracker_reset(tracker: &PyMemoryTracker) -> PyResult<()> {
    let inner = tracker
        .inner
        .lock()
        .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
    inner.reset();
    Ok(())
}

// =============================================================================
// DEVICE FUNCTIONS
// =============================================================================

/// Check if CUDA is available.
///
/// # Returns
/// `True` if CUDA is available, `False` otherwise.
///
/// # Example
/// ```python
/// if cuda_available():
///     print("CUDA is available!")
/// else:
///     print("Running on CPU")
/// ```
#[pyfunction]
fn cuda_available() -> bool {
    candle_core::Device::cuda_if_available(0)
        .map(|d| matches!(d, candle_core::Device::Cuda(_)))
        .unwrap_or(false)
}

/// Get device information.
///
/// # Arguments
/// * `force_cpu` - Force CPU device (default: `False`)
/// * `cuda_device` - CUDA device ordinal (default: 0)
///
/// # Returns
/// Dictionary with device information:
/// - `type`: "cuda" or "cpu"
/// - `ordinal`: CUDA device ordinal (if CUDA)
/// - `name`: Human-readable device name
///
/// # Example
/// ```python
/// device = get_device_info()
/// print(f"Using {device['type']} device")
/// ```
#[pyfunction]
#[pyo3(signature = (force_cpu=false, cuda_device=0))]
fn get_device_info(force_cpu: bool, cuda_device: usize) -> PyResult<HashMap<String, PyObject>> {
    Python::with_gil(|py| {
        let config = DeviceConfig::new()
            .with_force_cpu(force_cpu)
            .with_cuda_device(cuda_device);

        let device = rust_get_device(&config)
            .map_err(|e| PyRuntimeError::new_err(format!("Device error: {e}")))?;

        let mut result: HashMap<String, PyObject> = HashMap::new();

        match device {
            candle_core::Device::Cuda(_cuda_dev) => {
                result.insert("type".to_string(), "cuda".into_py(py));
                // Use the configured cuda_device ordinal (Candle doesn't expose it from CudaDevice directly)
                result.insert("ordinal".to_string(), cuda_device.into_py(py));
                result.insert(
                    "name".to_string(),
                    format!("CUDA:{cuda_device}").into_py(py),
                );
            }
            candle_core::Device::Cpu => {
                result.insert("type".to_string(), "cpu".into_py(py));
                result.insert("ordinal".to_string(), py.None());
                result.insert("name".to_string(), "CPU".into_py(py));
            }
            candle_core::Device::Metal(_metal_dev) => {
                result.insert("type".to_string(), "metal".into_py(py));
                // Metal device ordinal isn't directly exposed
                result.insert("ordinal".to_string(), 0_usize.into_py(py));
                result.insert("name".to_string(), "Metal:0".into_py(py));
            }
        }

        Ok(result)
    })
}

// =============================================================================
// DTYPE FUNCTIONS
// =============================================================================

/// Get bytes per element for a data type.
///
/// # Arguments
/// * `dtype` - Data type string: "f32", "f16", "bf16", "f64", "i32", "i64", "u8", "u32"
///
/// # Returns
/// Number of bytes per element.
///
/// # Example
/// ```python
/// print(f"f32 uses {bytes_per_dtype('f32')} bytes per element")
/// ```
#[pyfunction]
fn bytes_per_dtype(dtype: &str) -> PyResult<usize> {
    let candle_dtype = parse_dtype(dtype)?;
    Ok(dtype::bytes_per_element(candle_dtype))
}

/// Check if a data type is floating point.
///
/// # Arguments
/// * `dtype` - Data type string
///
/// # Returns
/// `True` if the dtype is floating point (f16, bf16, f32, f64).
///
/// # Example
/// ```python
/// assert is_floating_point("f32") == True
/// assert is_floating_point("i32") == False
/// ```
#[pyfunction]
fn is_floating_point_dtype(dtype: &str) -> PyResult<bool> {
    let candle_dtype = parse_dtype(dtype)?;
    Ok(dtype::is_floating_point(candle_dtype))
}

/// Get the accumulator data type for a given dtype.
///
/// For reduced precision types (f16, bf16), returns f32.
/// For integer types, returns i64.
///
/// # Arguments
/// * `dtype` - Data type string
///
/// # Returns
/// Accumulator dtype string.
///
/// # Example
/// ```python
/// assert accumulator_dtype("bf16") == "f32"
/// assert accumulator_dtype("i32") == "i64"
/// ```
#[pyfunction]
fn accumulator_dtype(dtype: &str) -> PyResult<String> {
    use crate::dtype::DTypeExt;
    let candle_dtype = parse_dtype(dtype)?;
    let acc_dtype = candle_dtype.accumulator_dtype();
    Ok(dtype_to_string(acc_dtype))
}

/// Get all supported data types.
///
/// # Returns
/// List of supported dtype strings.
#[pyfunction]
fn supported_dtypes() -> Vec<&'static str> {
    vec![
        "f16", "bf16", "f32", "f64", "u8", "u32", "i16", "i32", "i64",
    ]
}

// =============================================================================
// LOGGING FUNCTIONS
// =============================================================================

/// Initialize logging with configuration.
///
/// # Arguments
/// * `level` - Log level: "trace", "debug", "info", "warn", "error" (default: "info")
/// * `timestamps` - Include timestamps (default: `True`)
/// * `ansi` - Use ANSI colors (default: `True`)
///
/// # Example
/// ```python
/// init_logging(level="debug", timestamps=True, ansi=True)
/// ```
#[pyfunction]
#[pyo3(signature = (level="info", timestamps=true, ansi=true))]
fn init_logging(level: &str, timestamps: bool, ansi: bool) -> PyResult<()> {
    let log_level = match level.to_lowercase().as_str() {
        "trace" => LogLevel::Trace,
        "debug" => LogLevel::Debug,
        "info" => LogLevel::Info,
        "warn" | "warning" => LogLevel::Warn,
        "error" => LogLevel::Error,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid log level: {level}. Use: trace, debug, info, warn, error"
            )))
        }
    };

    let config = LogConfig::new()
        .with_level(log_level)
        .with_timestamps(timestamps)
        .with_ansi(ansi);

    rust_init_logging(&config);
    Ok(())
}

// =============================================================================
// VERSION AND UTILITIES
// =============================================================================

/// Get rust-ai-core version.
///
/// # Returns
/// Version string (e.g., "0.2.0").
#[pyfunction]
fn version() -> &'static str {
    crate::VERSION
}

/// Get the default overhead factor for memory estimation.
///
/// # Returns
/// Default overhead factor (1.1 = 10% overhead).
#[pyfunction]
fn default_overhead_factor() -> f64 {
    memory::DEFAULT_OVERHEAD_FACTOR
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn parse_dtype(dtype: &str) -> PyResult<candle_core::DType> {
    match dtype.to_lowercase().as_str() {
        "f16" | "float16" => Ok(candle_core::DType::F16),
        "bf16" | "bfloat16" => Ok(candle_core::DType::BF16),
        "f32" | "float32" | "float" => Ok(candle_core::DType::F32),
        "f64" | "float64" | "double" => Ok(candle_core::DType::F64),
        "u8" | "uint8" => Ok(candle_core::DType::U8),
        "u32" | "uint32" => Ok(candle_core::DType::U32),
        "i16" | "int16" => Ok(candle_core::DType::I16),
        "i32" | "int32" | "int" => Ok(candle_core::DType::I32),
        "i64" | "int64" | "long" => Ok(candle_core::DType::I64),
        _ => Err(PyValueError::new_err(format!(
            "Unknown dtype: {dtype}. Supported: f16, bf16, f32, f64, u8, u32, i16, i32, i64"
        ))),
    }
}

fn dtype_to_string(dtype: candle_core::DType) -> String {
    match dtype {
        candle_core::DType::F16 => "f16".to_string(),
        candle_core::DType::BF16 => "bf16".to_string(),
        candle_core::DType::F32 => "f32".to_string(),
        candle_core::DType::F64 => "f64".to_string(),
        candle_core::DType::U8 => "u8".to_string(),
        candle_core::DType::U32 => "u32".to_string(),
        candle_core::DType::I16 => "i16".to_string(),
        candle_core::DType::I32 => "i32".to_string(),
        candle_core::DType::I64 => "i64".to_string(),
        // Newer exotic dtypes
        candle_core::DType::F8E4M3 => "f8e4m3".to_string(),
        candle_core::DType::F6E2M3 => "f6e2m3".to_string(),
        candle_core::DType::F6E3M2 => "f6e3m2".to_string(),
        candle_core::DType::F4 => "f4".to_string(),
        candle_core::DType::F8E8M0 => "f8e8m0".to_string(),
    }
}

// =============================================================================
// PYTHON MODULE DEFINITION
// =============================================================================

/// Python module for rust-ai-core bindings.
#[pymodule]
pub fn rust_ai_core_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register class types
    m.add_class::<PyMemoryTracker>()?;

    // Memory estimation functions
    m.add_function(wrap_pyfunction!(estimate_tensor_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_attention_memory, m)?)?;

    // Memory tracker functions
    m.add_function(wrap_pyfunction!(create_memory_tracker, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_would_fit, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_allocate, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_deallocate, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_allocated_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_peak_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_limit_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_estimate_with_overhead, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_reset, m)?)?;

    // Device functions
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_info, m)?)?;

    // DType functions
    m.add_function(wrap_pyfunction!(bytes_per_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(is_floating_point_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(accumulator_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(supported_dtypes, m)?)?;

    // Logging functions
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;

    // Utilities
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(default_overhead_factor, m)?)?;

    Ok(())
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dtype() {
        assert!(matches!(
            parse_dtype("f32").unwrap(),
            candle_core::DType::F32
        ));
        assert!(matches!(
            parse_dtype("bf16").unwrap(),
            candle_core::DType::BF16
        ));
        assert!(matches!(
            parse_dtype("i64").unwrap(),
            candle_core::DType::I64
        ));
        assert!(parse_dtype("invalid").is_err());
    }

    #[test]
    fn test_dtype_to_string() {
        assert_eq!(dtype_to_string(candle_core::DType::F32), "f32");
        assert_eq!(dtype_to_string(candle_core::DType::BF16), "bf16");
    }

    #[test]
    fn test_estimate_tensor_bytes_internal() {
        let bytes = memory::estimate_tensor_bytes(&[1, 512, 4096], candle_core::DType::F32);
        assert_eq!(bytes, 1 * 512 * 4096 * 4);
    }
}
