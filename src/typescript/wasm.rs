// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! WebAssembly bindings via wasm-bindgen.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::needless_pass_by_value)]

use wasm_bindgen::prelude::*;

use super::common::{self, JsMemoryTracker};
use crate::memory;

/// Memory tracker for managing memory allocations (WASM version).
#[wasm_bindgen]
pub struct MemoryTrackerWasm {
    inner: JsMemoryTracker,
}

#[wasm_bindgen]
impl MemoryTrackerWasm {
    #[wasm_bindgen(constructor)]
    pub fn create(limit_gb: Option<f64>, overhead_factor: Option<f64>) -> Self {
        Self {
            inner: JsMemoryTracker::new(
                limit_gb.unwrap_or(2.0),
                overhead_factor.unwrap_or(common::default_overhead_factor()),
            ),
        }
    }

    #[wasm_bindgen(js_name = wouldFit)]
    pub fn would_fit(&self, bytes: u32) -> Result<bool, JsError> {
        self.inner
            .would_fit(bytes as usize)
            .map_err(|e| JsError::new(&e))
    }

    #[wasm_bindgen]
    pub fn allocate(&self, bytes: u32) -> Result<(), JsError> {
        self.inner
            .allocate(bytes as usize)
            .map_err(|e| JsError::new(&e))
    }

    #[wasm_bindgen]
    pub fn deallocate(&self, bytes: u32) -> Result<(), JsError> {
        self.inner
            .deallocate(bytes as usize)
            .map_err(|e| JsError::new(&e))
    }

    #[wasm_bindgen(js_name = allocatedBytes)]
    pub fn allocated_bytes(&self) -> Result<u32, JsError> {
        self.inner
            .allocated_bytes()
            .map(|b| b as u32)
            .map_err(|e| JsError::new(&e))
    }

    #[wasm_bindgen(js_name = peakBytes)]
    pub fn peak_bytes(&self) -> Result<u32, JsError> {
        self.inner
            .peak_bytes()
            .map(|b| b as u32)
            .map_err(|e| JsError::new(&e))
    }

    #[wasm_bindgen(js_name = limitBytes)]
    pub fn limit_bytes(&self) -> Result<u32, JsError> {
        self.inner
            .limit_bytes()
            .map(|b| b as u32)
            .map_err(|e| JsError::new(&e))
    }

    #[wasm_bindgen(js_name = estimateWithOverhead)]
    pub fn estimate_with_overhead(
        &self,
        shape: &[u32],
        dtype: Option<String>,
    ) -> Result<u32, JsError> {
        let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
        let dtype_str = dtype.as_deref().unwrap_or("f32");
        self.inner
            .estimate_with_overhead(&shape_usize, dtype_str)
            .map(|b| b as u32)
            .map_err(|e| JsError::new(&e))
    }

    #[wasm_bindgen]
    pub fn reset(&self) -> Result<(), JsError> {
        self.inner.reset().map_err(|e| JsError::new(&e))
    }
}

#[wasm_bindgen(js_name = estimateTensorBytesWasm)]
pub fn estimate_tensor_bytes_wasm(shape: &[u32], dtype: Option<String>) -> Result<u32, JsError> {
    let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    let dtype_str = dtype.as_deref().unwrap_or("f32");
    let candle_dtype = common::parse_dtype(dtype_str).map_err(|e| JsError::new(&e))?;
    Ok(memory::estimate_tensor_bytes(&shape_usize, candle_dtype) as u32)
}

#[wasm_bindgen(js_name = estimateAttentionMemoryWasm)]
pub fn estimate_attention_memory_wasm(
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    dtype: Option<String>,
) -> Result<u32, JsError> {
    let dtype_str = dtype.as_deref().unwrap_or("bf16");
    let candle_dtype = common::parse_dtype(dtype_str).map_err(|e| JsError::new(&e))?;
    Ok(memory::estimate_attention_memory(
        batch_size as usize,
        num_heads as usize,
        seq_len as usize,
        head_dim as usize,
        candle_dtype,
    ) as u32)
}

#[wasm_bindgen(js_name = cudaAvailableWasm)]
pub fn cuda_available_wasm() -> bool {
    false // CUDA is never available in WASM
}

#[wasm_bindgen(js_name = getDeviceInfoWasm)]
pub fn get_device_info_wasm() -> js_sys::Object {
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(&obj, &"type".into(), &"cpu".into()).unwrap_or_default();
    js_sys::Reflect::set(&obj, &"ordinal".into(), &JsValue::NULL).unwrap_or_default();
    js_sys::Reflect::set(&obj, &"name".into(), &"CPU (WASM)".into()).unwrap_or_default();
    obj
}

#[wasm_bindgen(js_name = bytesPerDtypeWasm)]
pub fn bytes_per_dtype_wasm(dtype: String) -> Result<u32, JsError> {
    common::bytes_per_dtype(&dtype)
        .map(|b| b as u32)
        .map_err(|e| JsError::new(&e))
}

#[wasm_bindgen(js_name = isFloatingPointDtypeWasm)]
pub fn is_floating_point_dtype_wasm(dtype: String) -> Result<bool, JsError> {
    common::is_floating_point_dtype(&dtype).map_err(|e| JsError::new(&e))
}

#[wasm_bindgen(js_name = accumulatorDtypeWasm)]
pub fn accumulator_dtype_wasm(dtype: String) -> Result<String, JsError> {
    common::accumulator_dtype(&dtype).map_err(|e| JsError::new(&e))
}

#[wasm_bindgen(js_name = supportedDtypesWasm)]
pub fn supported_dtypes_wasm() -> js_sys::Array {
    let arr = js_sys::Array::new();
    for dtype in common::supported_dtypes() {
        arr.push(&JsValue::from_str(dtype));
    }
    arr
}

#[wasm_bindgen(js_name = versionWasm)]
pub fn version_wasm() -> String {
    crate::VERSION.to_string()
}

#[wasm_bindgen(js_name = defaultOverheadFactorWasm)]
pub fn default_overhead_factor_wasm() -> f64 {
    common::default_overhead_factor()
}

#[wasm_bindgen(js_name = initLoggingWasm)]
pub fn init_logging_wasm(level: Option<String>) -> Result<(), JsError> {
    let level_str = level.as_deref().unwrap_or("info");
    match level_str.to_lowercase().as_str() {
        "trace" | "debug" | "info" | "warn" | "warning" | "error" => Ok(()),
        _ => Err(JsError::new(&format!(
            "Invalid log level: {level_str}. Use: trace, debug, info, warn, error"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tensor_bytes_wasm() {
        let bytes = estimate_tensor_bytes_wasm(&[32, 32], Some("f32".to_string())).unwrap();
        assert_eq!(bytes, 4096);
    }

    #[test]
    fn test_cuda_available_wasm() {
        assert!(!cuda_available_wasm());
    }
}
