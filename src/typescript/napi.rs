// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Node.js native bindings via napi-rs.

#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::redundant_closure)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

use super::common::{self, DeviceInfo, JsMemoryTracker};
use crate::device::{get_device as rust_get_device, DeviceConfig};
use crate::logging::{init_logging as rust_init_logging, LogConfig, LogLevel};
use crate::memory;

/// Memory tracker for managing GPU/CPU memory allocations.
#[napi]
pub struct MemoryTracker {
    inner: JsMemoryTracker,
}

#[napi]
impl MemoryTracker {
    #[napi(constructor)]
    pub fn new(limit_gb: Option<f64>, overhead_factor: Option<f64>) -> Self {
        Self {
            inner: JsMemoryTracker::new(
                limit_gb.unwrap_or(8.0),
                overhead_factor.unwrap_or(common::default_overhead_factor()),
            ),
        }
    }

    #[napi]
    pub fn would_fit(&self, bytes: u32) -> Result<bool> {
        self.inner
            .would_fit(bytes as usize)
            .map_err(|e| Error::from_reason(e))
    }

    #[napi]
    pub fn allocate(&self, bytes: u32) -> Result<()> {
        self.inner
            .allocate(bytes as usize)
            .map_err(|e| Error::from_reason(e))
    }

    #[napi]
    pub fn deallocate(&self, bytes: u32) -> Result<()> {
        self.inner
            .deallocate(bytes as usize)
            .map_err(|e| Error::from_reason(e))
    }

    #[napi]
    pub fn allocated_bytes(&self) -> Result<u32> {
        self.inner
            .allocated_bytes()
            .map(|b| b as u32)
            .map_err(|e| Error::from_reason(e))
    }

    #[napi]
    pub fn peak_bytes(&self) -> Result<u32> {
        self.inner
            .peak_bytes()
            .map(|b| b as u32)
            .map_err(|e| Error::from_reason(e))
    }

    #[napi]
    pub fn limit_bytes(&self) -> Result<u32> {
        self.inner
            .limit_bytes()
            .map(|b| b as u32)
            .map_err(|e| Error::from_reason(e))
    }

    #[napi]
    pub fn estimate_with_overhead(&self, shape: Vec<u32>, dtype: Option<String>) -> Result<u32> {
        let shape_usize: Vec<usize> = shape.into_iter().map(|s| s as usize).collect();
        let dtype_str = dtype.as_deref().unwrap_or("f32");
        self.inner
            .estimate_with_overhead(&shape_usize, dtype_str)
            .map(|b| b as u32)
            .map_err(|e| Error::from_reason(e))
    }

    #[napi]
    pub fn reset(&self) -> Result<()> {
        self.inner.reset().map_err(|e| Error::from_reason(e))
    }
}

#[napi(object)]
pub struct JsDeviceInfo {
    #[napi(js_name = "type")]
    pub device_type: String,
    pub ordinal: Option<u32>,
    pub name: String,
}

impl From<DeviceInfo> for JsDeviceInfo {
    fn from(info: DeviceInfo) -> Self {
        Self {
            device_type: info.device_type,
            ordinal: info.ordinal.map(|o| o as u32),
            name: info.name,
        }
    }
}

#[napi]
pub fn estimate_tensor_bytes(shape: Vec<u32>, dtype: Option<String>) -> Result<u32> {
    let shape_usize: Vec<usize> = shape.into_iter().map(|s| s as usize).collect();
    let dtype_str = dtype.as_deref().unwrap_or("f32");
    let candle_dtype = common::parse_dtype(dtype_str).map_err(|e| Error::from_reason(e))?;
    Ok(memory::estimate_tensor_bytes(&shape_usize, candle_dtype) as u32)
}

#[napi]
pub fn estimate_attention_memory(
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    dtype: Option<String>,
) -> Result<u32> {
    let dtype_str = dtype.as_deref().unwrap_or("bf16");
    let candle_dtype = common::parse_dtype(dtype_str).map_err(|e| Error::from_reason(e))?;
    Ok(memory::estimate_attention_memory(
        batch_size as usize,
        num_heads as usize,
        seq_len as usize,
        head_dim as usize,
        candle_dtype,
    ) as u32)
}

#[napi]
pub fn cuda_available() -> bool {
    candle_core::Device::cuda_if_available(0)
        .map(|d| matches!(d, candle_core::Device::Cuda(_)))
        .unwrap_or(false)
}

#[napi]
pub fn get_device_info(force_cpu: Option<bool>, cuda_device: Option<u32>) -> Result<JsDeviceInfo> {
    let config = DeviceConfig::new()
        .with_force_cpu(force_cpu.unwrap_or(false))
        .with_cuda_device(cuda_device.unwrap_or(0) as usize);

    let device =
        rust_get_device(&config).map_err(|e| Error::from_reason(format!("Device error: {e}")))?;

    let info = match device {
        candle_core::Device::Cuda(_) => DeviceInfo {
            device_type: "cuda".to_string(),
            ordinal: Some(cuda_device.unwrap_or(0) as usize),
            name: format!("CUDA:{}", cuda_device.unwrap_or(0)),
        },
        candle_core::Device::Metal(_) => DeviceInfo {
            device_type: "metal".to_string(),
            ordinal: Some(0),
            name: "Metal:0".to_string(),
        },
        candle_core::Device::Cpu => DeviceInfo {
            device_type: "cpu".to_string(),
            ordinal: None,
            name: "CPU".to_string(),
        },
    };

    Ok(info.into())
}

#[napi]
pub fn bytes_per_dtype(dtype: String) -> Result<u32> {
    common::bytes_per_dtype(&dtype)
        .map(|b| b as u32)
        .map_err(|e| Error::from_reason(e))
}

#[napi]
pub fn is_floating_point_dtype(dtype: String) -> Result<bool> {
    common::is_floating_point_dtype(&dtype).map_err(|e| Error::from_reason(e))
}

#[napi]
pub fn accumulator_dtype(dtype: String) -> Result<String> {
    common::accumulator_dtype(&dtype).map_err(|e| Error::from_reason(e))
}

#[napi]
pub fn supported_dtypes() -> Vec<String> {
    common::supported_dtypes()
        .into_iter()
        .map(String::from)
        .collect()
}

#[napi]
pub fn init_logging(
    level: Option<String>,
    timestamps: Option<bool>,
    ansi: Option<bool>,
) -> Result<()> {
    let level_str = level.as_deref().unwrap_or("info");
    let log_level = match level_str.to_lowercase().as_str() {
        "trace" => LogLevel::Trace,
        "debug" => LogLevel::Debug,
        "info" => LogLevel::Info,
        "warn" | "warning" => LogLevel::Warn,
        "error" => LogLevel::Error,
        _ => {
            return Err(Error::from_reason(format!(
                "Invalid log level: {level_str}. Use: trace, debug, info, warn, error"
            )))
        }
    };

    let config = LogConfig::new()
        .with_level(log_level)
        .with_timestamps(timestamps.unwrap_or(true))
        .with_ansi(ansi.unwrap_or(true));

    rust_init_logging(&config);
    Ok(())
}

#[napi]
pub fn version() -> &'static str {
    crate::VERSION
}

#[napi]
pub fn default_overhead_factor() -> f64 {
    common::default_overhead_factor()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracker_napi() {
        let tracker = MemoryTracker::new(Some(1.0), Some(1.1));
        assert_eq!(tracker.allocated_bytes().unwrap(), 0);
        tracker.allocate(1024).unwrap();
        assert_eq!(tracker.allocated_bytes().unwrap(), 1024);
    }

    #[test]
    fn test_estimate_tensor_bytes_napi() {
        let bytes = estimate_tensor_bytes(vec![32, 32], Some("f32".to_string())).unwrap();
        assert_eq!(bytes, 4096);
    }
}
