// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Candle ↔ CubeCL tensor conversion utilities.
//!
//! This module provides the core interop layer between Candle's tensor representation
//! and CubeCL's buffer handles. It handles memory layout, device synchronization,
//! and dtype conversion.

use crate::error::{CoreError, Result};
use candle_core::{DType, Device, Tensor};

/// Intermediate buffer representation for CubeCL kernels.
///
/// Contains the raw bytes, shape, and dtype needed to create a CubeCL
/// buffer handle and reconstruct a Candle tensor after kernel execution.
#[derive(Debug, Clone)]
pub struct TensorBuffer {
    /// Raw tensor data as little-endian bytes.
    pub bytes: Vec<u8>,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Data type.
    pub dtype: DType,
}

impl TensorBuffer {
    /// Create a new tensor buffer.
    pub fn new(bytes: Vec<u8>, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            bytes,
            shape,
            dtype,
        }
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.bytes.len()
    }
}

/// Check if CubeCL CUDA runtime support is available.
///
/// This checks:
/// 1. The `cuda` feature is enabled at compile time
/// 2. A CUDA-capable device is detected at runtime
///
/// # Returns
///
/// `true` if CubeCL CUDA kernels can be launched, `false` otherwise.
///
/// # Example
///
/// ```rust
/// use rust_ai_core::has_cubecl_cuda_support;
///
/// if has_cubecl_cuda_support() {
///     println!("CubeCL CUDA acceleration available!");
/// } else {
///     println!("Falling back to Candle backend");
/// }
/// ```
#[must_use]
pub fn has_cubecl_cuda_support() -> bool {
    // Check if Candle can see a CUDA device as a proxy for CubeCL support
    // CubeCL uses the same CUDA driver stack
    matches!(Device::cuda_if_available(0), Ok(Device::Cuda(_)))
}

/// Convert a Candle tensor to a CubeCL-compatible buffer.
///
/// The tensor must be contiguous in memory. If not, it will be made contiguous
/// (which may involve a copy).
///
/// # Arguments
///
/// * `tensor` - The Candle tensor to convert
///
/// # Returns
///
/// A [`TensorBuffer`] containing raw bytes, shape, and dtype that can be used
/// to create a CubeCL buffer handle via `client.create(&buffer.bytes)`.
///
/// # Errors
///
/// Returns error if:
/// - Tensor is not on a CUDA device
/// - Tensor dtype is not supported (only f32, f16, bf16 currently)
///
/// # Example
///
/// ```rust,ignore
/// use rust_ai_core::candle_to_cubecl_handle;
///
/// let tensor = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &Device::cuda_if_available(0)?)?;
/// let buffer = candle_to_cubecl_handle(&tensor)?;
///
/// // Use with CubeCL:
/// // let handle = client.create(&buffer.bytes);
/// ```
pub fn candle_to_cubecl_handle(tensor: &Tensor) -> Result<TensorBuffer> {
    // Ensure tensor is on CUDA
    if !matches!(tensor.device(), Device::Cuda(_)) {
        return Err(CoreError::invalid_config(
            "candle_to_cubecl_handle requires CUDA tensor",
        ));
    }

    // Ensure contiguous memory layout
    let tensor = tensor.contiguous()?;

    // Get shape and dtype
    let shape = tensor.dims().to_vec();
    let dtype = tensor.dtype();

    // Extract raw bytes based on dtype
    let bytes = match dtype {
        DType::F32 => {
            let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
            data.iter().flat_map(|f| f.to_le_bytes()).collect()
        }
        DType::F16 => {
            let data: Vec<half::f16> = tensor.flatten_all()?.to_vec1()?;
            data.iter().flat_map(|f| f.to_le_bytes()).collect()
        }
        DType::BF16 => {
            let data: Vec<half::bf16> = tensor.flatten_all()?.to_vec1()?;
            data.iter().flat_map(|f| f.to_le_bytes()).collect()
        }
        _ => {
            return Err(CoreError::invalid_config(format!(
                "candle_to_cubecl_handle does not support dtype {dtype:?}"
            )));
        }
    };

    Ok(TensorBuffer::new(bytes, shape, dtype))
}

/// Convert a CubeCL buffer back to a Candle tensor.
///
/// # Arguments
///
/// * `buffer` - Buffer containing raw bytes from CubeCL kernel output
/// * `device` - Target Candle device (must be CUDA)
///
/// # Returns
///
/// A Candle tensor with the specified shape on the target device.
///
/// # Errors
///
/// Returns error if:
/// - Device is not CUDA
/// - Buffer size doesn't match shape * dtype size
/// - Dtype is not supported
///
/// # Example
///
/// ```rust,ignore
/// use rust_ai_core::cubecl_to_candle_tensor;
///
/// // After kernel execution:
/// // let output_bytes = client.read(&output_handle);
/// let buffer = TensorBuffer::new(output_bytes, vec![2, 4, 8, 64], DType::F32);
/// let tensor = cubecl_to_candle_tensor(&buffer, &device)?;
/// ```
pub fn cubecl_to_candle_tensor(buffer: &TensorBuffer, device: &Device) -> Result<Tensor> {
    // Validate device
    if !matches!(device, Device::Cuda(_)) {
        return Err(CoreError::invalid_config(
            "cubecl_to_candle_tensor requires CUDA device",
        ));
    }

    let numel = buffer.numel();
    let expected_bytes = numel * buffer.dtype.size_in_bytes();

    if buffer.bytes.len() != expected_bytes {
        return Err(CoreError::shape_mismatch(
            vec![expected_bytes],
            vec![buffer.bytes.len()],
        ));
    }

    // Reconstruct tensor based on dtype
    let tensor = match buffer.dtype {
        DType::F32 => {
            let data: Vec<f32> = buffer
                .bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Tensor::from_vec(data, buffer.shape.as_slice(), device)?
        }
        DType::F16 => {
            let data: Vec<half::f16> = buffer
                .bytes
                .chunks_exact(2)
                .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();
            Tensor::from_vec(data, buffer.shape.as_slice(), device)?
        }
        DType::BF16 => {
            let data: Vec<half::bf16> = buffer
                .bytes
                .chunks_exact(2)
                .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();
            Tensor::from_vec(data, buffer.shape.as_slice(), device)?
        }
        _ => {
            return Err(CoreError::invalid_config(format!(
                "cubecl_to_candle_tensor does not support dtype {:?}",
                buffer.dtype
            )));
        }
    };

    Ok(tensor)
}

/// Allocate an output buffer for a CubeCL kernel.
///
/// Creates a zero-initialized buffer of the appropriate size.
///
/// # Arguments
///
/// * `shape` - Output tensor shape
/// * `dtype` - Output data type
///
/// # Returns
///
/// A [`TensorBuffer`] with zero-initialized bytes.
pub fn allocate_output_buffer(shape: &[usize], dtype: DType) -> Result<TensorBuffer> {
    let numel: usize = shape.iter().product();
    let size_bytes = numel * dtype.size_in_bytes();
    let bytes = vec![0u8; size_bytes];
    Ok(TensorBuffer::new(bytes, shape.to_vec(), dtype))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_buffer() {
        let buffer = TensorBuffer::new(vec![0u8; 64], vec![2, 8], DType::F32);
        assert_eq!(buffer.numel(), 16);
        assert_eq!(buffer.size_bytes(), 64);
    }

    #[test]
    fn test_allocate_output_buffer() {
        let buffer = allocate_output_buffer(&[4, 8, 16], DType::F32).unwrap();
        assert_eq!(buffer.numel(), 512);
        assert_eq!(buffer.size_bytes(), 2048); // 512 * 4 bytes
    }
}
