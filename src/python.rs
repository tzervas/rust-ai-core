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

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::useless_conversion)] // PyO3 macro generates these
#![allow(clippy::missing_errors_doc)] // Python bindings - errors are documented in docstrings
#![allow(clippy::needless_pass_by_value)] // PyO3 requires owned types for Python arguments
#![allow(deprecated)] // PyO3 0.24 deprecates into_py, will migrate to IntoPyObject in future

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::device::{get_device as rust_get_device, DeviceConfig};
use crate::dtype;
use crate::logging::{init_logging as rust_init_logging, LogConfig, LogLevel};
use crate::memory::{self, MemoryTracker};

/// Opaque handle for Python to reference a `MemoryTracker`.
#[pyclass(name = "MemoryTracker")]
#[derive(Clone)]
pub struct PyMemoryTracker {
    inner: Arc<Mutex<MemoryTracker>>,
}

/// Estimate memory required for a tensor with given shape and dtype.
#[pyfunction]
#[pyo3(signature = (shape, dtype="f32"))]
fn estimate_tensor_bytes(shape: Vec<usize>, dtype: &str) -> PyResult<usize> {
    let candle_dtype = parse_dtype(dtype)?;
    Ok(memory::estimate_tensor_bytes(&shape, candle_dtype))
}

/// Estimate memory for attention computation (O(n^2) attention scores).
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

/// Create a memory tracker with a limit.
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
#[pyfunction]
fn tracker_would_fit(tracker: &PyMemoryTracker, bytes: usize) -> PyResult<bool> {
    let inner = tracker
        .inner
        .lock()
        .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
    Ok(inner.would_fit(bytes))
}

/// Record an allocation in the tracker.
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
#[pyfunction]
fn tracker_allocated_bytes(tracker: &PyMemoryTracker) -> PyResult<usize> {
    let inner = tracker
        .inner
        .lock()
        .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
    Ok(inner.allocated_bytes())
}

/// Get peak allocation from the tracker.
#[pyfunction]
fn tracker_peak_bytes(tracker: &PyMemoryTracker) -> PyResult<usize> {
    let inner = tracker
        .inner
        .lock()
        .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
    Ok(inner.peak_bytes())
}

/// Get the memory limit from the tracker.
#[pyfunction]
fn tracker_limit_bytes(tracker: &PyMemoryTracker) -> PyResult<usize> {
    let inner = tracker
        .inner
        .lock()
        .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
    Ok(inner.limit_bytes())
}

/// Estimate bytes with overhead factor.
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
#[pyfunction]
fn tracker_reset(tracker: &PyMemoryTracker) -> PyResult<()> {
    let inner = tracker
        .inner
        .lock()
        .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
    inner.reset();
    Ok(())
}

/// Check if CUDA is available.
#[pyfunction]
fn cuda_available() -> bool {
    candle_core::Device::cuda_if_available(0)
        .map(|d| matches!(d, candle_core::Device::Cuda(_)))
        .unwrap_or(false)
}

/// Get device information.
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
                result.insert("ordinal".to_string(), 0_usize.into_py(py));
                result.insert("name".to_string(), "Metal:0".into_py(py));
            }
        }

        Ok(result)
    })
}

/// Get bytes per element for a data type.
#[pyfunction]
fn bytes_per_dtype(dtype: &str) -> PyResult<usize> {
    let candle_dtype = parse_dtype(dtype)?;
    Ok(dtype::bytes_per_element(candle_dtype))
}

/// Check if a data type is floating point.
#[pyfunction]
fn is_floating_point_dtype(dtype: &str) -> PyResult<bool> {
    let candle_dtype = parse_dtype(dtype)?;
    Ok(dtype::is_floating_point(candle_dtype))
}

/// Get the accumulator data type for a given dtype.
#[pyfunction]
fn accumulator_dtype(dtype: &str) -> PyResult<String> {
    use crate::dtype::DTypeExt;
    let candle_dtype = parse_dtype(dtype)?;
    let acc_dtype = candle_dtype.accumulator_dtype();
    Ok(dtype_to_string(acc_dtype))
}

/// Get all supported data types.
#[pyfunction]
fn supported_dtypes() -> Vec<&'static str> {
    vec![
        "f16", "bf16", "f32", "f64", "u8", "u32", "i16", "i32", "i64",
    ]
}

/// Initialize logging with configuration.
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

/// Get rust-ai-core version.
#[pyfunction]
fn version() -> &'static str {
    crate::VERSION
}

/// Get the default overhead factor for memory estimation.
#[pyfunction]
fn default_overhead_factor() -> f64 {
    memory::DEFAULT_OVERHEAD_FACTOR
}

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
        candle_core::DType::F8E4M3 => "f8e4m3".to_string(),
        candle_core::DType::F6E2M3 => "f6e2m3".to_string(),
        candle_core::DType::F6E3M2 => "f6e3m2".to_string(),
        candle_core::DType::F4 => "f4".to_string(),
        candle_core::DType::F8E8M0 => "f8e8m0".to_string(),
    }
}

/// Python module for rust-ai-core bindings.
#[pymodule]
pub fn rust_ai_core_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemoryTracker>()?;

    m.add_function(wrap_pyfunction!(estimate_tensor_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_attention_memory, m)?)?;

    m.add_function(wrap_pyfunction!(create_memory_tracker, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_would_fit, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_allocate, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_deallocate, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_allocated_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_peak_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_limit_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_estimate_with_overhead, m)?)?;
    m.add_function(wrap_pyfunction!(tracker_reset, m)?)?;

    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_info, m)?)?;

    m.add_function(wrap_pyfunction!(bytes_per_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(is_floating_point_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(accumulator_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(supported_dtypes, m)?)?;

    m.add_function(wrap_pyfunction!(init_logging, m)?)?;

    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(default_overhead_factor, m)?)?;

    Ok(())
}

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
        assert!(parse_dtype("invalid").is_err());
    }

    #[test]
    fn test_dtype_to_string() {
        assert_eq!(dtype_to_string(candle_core::DType::F32), "f32");
        assert_eq!(dtype_to_string(candle_core::DType::BF16), "bf16");
    }
}
