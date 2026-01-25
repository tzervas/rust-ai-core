// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Integration tests for rust-ai-core.
//!
//! These tests verify the public API works correctly as a cohesive system.

#![allow(clippy::similar_names)] // quantizer/quantized are intentionally similar
#![allow(clippy::cast_possible_truncation)] // Test quantization - precision loss is expected
#![allow(clippy::cast_sign_loss)] // Test quantization uses intentional casts

use candle_core::{DType, Device, Tensor};
use rust_ai_core::{
    bytes_per_element, estimate_tensor_bytes, get_device, is_floating_point, warn_if_cpu,
    CoreError, DTypeExt, Dequantize, DeviceConfig, GpuDispatchable, LogConfig, MemoryTracker,
    Quantize, Result, ValidatableConfig,
};

// ============================================================================
// Device Selection Tests
// ============================================================================

#[test]
fn test_device_config_from_env_respects_force_cpu() {
    // Set environment variable
    std::env::set_var("RUST_AI_FORCE_CPU", "1");

    let config = DeviceConfig::from_env();
    assert!(config.force_cpu);

    // Clean up
    std::env::remove_var("RUST_AI_FORCE_CPU");
}

#[test]
fn test_get_device_with_force_cpu() {
    let config = DeviceConfig::new()
        .with_force_cpu(true)
        .with_crate_name("test");

    let device = get_device(&config).expect("Should always succeed with force_cpu");
    assert!(matches!(device, Device::Cpu));
}

#[test]
fn test_device_config_builder_chain() {
    let config = DeviceConfig::new()
        .with_cuda_device(2)
        .with_force_cpu(false)
        .with_crate_name("integration-test");

    assert_eq!(config.cuda_device, 2);
    assert!(!config.force_cpu);
    assert_eq!(config.crate_name.as_deref(), Some("integration-test"));
}

#[test]
fn test_warn_if_cpu_does_not_panic() {
    // Verify warning mechanism doesn't panic on repeated calls
    let device = Device::Cpu;
    warn_if_cpu(&device, "test-crate");
    warn_if_cpu(&device, "test-crate"); // Second call should be silenced
}

// ============================================================================
// Error Type Tests
// ============================================================================

#[test]
fn test_error_helper_constructors() {
    let err = CoreError::invalid_config("test error");
    assert!(err.to_string().contains("test error"));

    let err = CoreError::shape_mismatch(vec![1, 2], vec![3, 4]);
    assert!(err.to_string().contains("shape mismatch"));

    let err = CoreError::dim_mismatch("dimensions don't match");
    assert!(err.to_string().contains("dimensions don't match"));

    let err = CoreError::device_not_available("CUDA:99");
    assert!(err.to_string().contains("CUDA:99"));

    let err = CoreError::oom("failed to allocate 16GB");
    assert!(err.to_string().contains("16GB"));

    let err = CoreError::kernel("kernel launch failed");
    assert!(err.to_string().contains("kernel launch failed"));

    let err = CoreError::not_implemented("async training");
    assert!(err.to_string().contains("async training"));

    let err = CoreError::io("file not found");
    assert!(err.to_string().contains("file not found"));
}

#[test]
fn test_error_from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
    let core_err: CoreError = io_err.into();

    assert!(matches!(core_err, CoreError::Io(_)));
    assert!(core_err.to_string().contains("access denied"));
}

#[test]
fn test_error_from_candle_error() {
    // Create a Candle error by attempting an invalid operation
    let tensor_result: std::result::Result<Tensor, candle_core::Error> =
        Tensor::zeros(&[0], DType::F32, &Device::Cpu); // Empty tensor

    if let Err(candle_err) = tensor_result {
        let core_err: CoreError = candle_err.into();
        assert!(matches!(core_err, CoreError::Candle(_)));
    }
}

// ============================================================================
// Trait Tests
// ============================================================================

#[derive(Clone)]
struct TestConfig {
    rank: usize,
    alpha: f32,
}

impl ValidatableConfig for TestConfig {
    fn validate(&self) -> Result<()> {
        if self.rank == 0 {
            return Err(CoreError::invalid_config("rank must be > 0"));
        }
        if self.alpha <= 0.0 {
            return Err(CoreError::invalid_config("alpha must be positive"));
        }
        Ok(())
    }
}

#[test]
fn test_validatable_config_valid() {
    let config = TestConfig {
        rank: 16,
        alpha: 32.0,
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_validatable_config_invalid_rank() {
    let config = TestConfig {
        rank: 0,
        alpha: 32.0,
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("rank"));
}

#[test]
fn test_validatable_config_invalid_alpha() {
    let config = TestConfig {
        rank: 16,
        alpha: -1.0,
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("alpha"));
}

// ============================================================================
// Memory Management Tests
// ============================================================================

#[test]
fn test_memory_estimation_basic() {
    // 1000 f32 elements = 4000 bytes
    assert_eq!(estimate_tensor_bytes(&[10, 100], DType::F32), 4000);

    // 1000 bf16 elements = 2000 bytes
    assert_eq!(estimate_tensor_bytes(&[10, 100], DType::BF16), 2000);

    // 3D tensor: 2 * 4 * 8 * 4 = 256 bytes for f32
    assert_eq!(estimate_tensor_bytes(&[2, 4, 8], DType::F32), 256);
}

#[test]
fn test_memory_tracker_lifecycle() {
    let tracker = MemoryTracker::with_limit(10_000);

    // Initial state
    assert_eq!(tracker.allocated_bytes(), 0);
    assert_eq!(tracker.peak_bytes(), 0);

    // Allocate
    tracker.allocate(5_000).expect("Should fit");
    assert_eq!(tracker.allocated_bytes(), 5_000);
    assert_eq!(tracker.peak_bytes(), 5_000);

    // More allocation
    tracker.allocate(3_000).expect("Should fit");
    assert_eq!(tracker.allocated_bytes(), 8_000);
    assert_eq!(tracker.peak_bytes(), 8_000);

    // Would exceed limit
    let result = tracker.allocate(5_000);
    assert!(result.is_err());
    assert_eq!(tracker.allocated_bytes(), 8_000); // Unchanged

    // Deallocate
    tracker.deallocate(3_000);
    assert_eq!(tracker.allocated_bytes(), 5_000);
    assert_eq!(tracker.peak_bytes(), 8_000); // Peak unchanged

    // Now fits
    tracker.allocate(4_000).expect("Should fit after dealloc");
    assert_eq!(tracker.allocated_bytes(), 9_000);
    assert_eq!(tracker.peak_bytes(), 9_000);
}

#[test]
fn test_memory_tracker_would_fit() {
    let tracker = MemoryTracker::with_limit(1000);
    tracker.allocate(500).unwrap();

    assert!(tracker.would_fit(400));
    assert!(tracker.would_fit(500));
    assert!(!tracker.would_fit(501));
}

#[test]
fn test_memory_tracker_reset() {
    let tracker = MemoryTracker::new();
    tracker.allocate(1000).unwrap();
    tracker.allocate(2000).unwrap();

    assert_eq!(tracker.allocated_bytes(), 3000);
    assert_eq!(tracker.peak_bytes(), 3000);

    tracker.reset();

    assert_eq!(tracker.allocated_bytes(), 0);
    assert_eq!(tracker.peak_bytes(), 0);
}

// ============================================================================
// DType Utilities Tests
// ============================================================================

#[test]
fn test_bytes_per_element_all_types() {
    assert_eq!(bytes_per_element(DType::U8), 1);
    assert_eq!(bytes_per_element(DType::U32), 4);
    assert_eq!(bytes_per_element(DType::I16), 2);
    assert_eq!(bytes_per_element(DType::I32), 4);
    assert_eq!(bytes_per_element(DType::I64), 8);
    assert_eq!(bytes_per_element(DType::F16), 2);
    assert_eq!(bytes_per_element(DType::BF16), 2);
    assert_eq!(bytes_per_element(DType::F32), 4);
    assert_eq!(bytes_per_element(DType::F64), 8);
}

#[test]
fn test_is_floating_point() {
    assert!(is_floating_point(DType::F16));
    assert!(is_floating_point(DType::BF16));
    assert!(is_floating_point(DType::F32));
    assert!(is_floating_point(DType::F64));

    assert!(!is_floating_point(DType::U8));
    assert!(!is_floating_point(DType::U32));
    assert!(!is_floating_point(DType::I64));
}

#[test]
fn test_dtype_ext_methods() {
    // Half precision
    assert!(DType::F16.is_half_precision());
    assert!(DType::BF16.is_half_precision());
    assert!(!DType::F32.is_half_precision());
    assert!(!DType::F64.is_half_precision());

    // Training dtype
    assert!(DType::F16.is_training_dtype());
    assert!(DType::BF16.is_training_dtype());
    assert!(DType::F32.is_training_dtype());
    assert!(!DType::F64.is_training_dtype()); // Too expensive for training
    assert!(!DType::I64.is_training_dtype());

    // Integer
    assert!(DType::U8.is_integer());
    assert!(DType::I64.is_integer());
    assert!(!DType::F32.is_integer());

    // Accumulator dtype
    assert_eq!(DType::F16.accumulator_dtype(), DType::F32);
    assert_eq!(DType::BF16.accumulator_dtype(), DType::F32);
    assert_eq!(DType::F32.accumulator_dtype(), DType::F32);
    assert_eq!(DType::I64.accumulator_dtype(), DType::I64);
}

// ============================================================================
// Logging Tests
// ============================================================================

#[test]
fn test_log_config_presets() {
    let dev = LogConfig::development();
    assert!(dev.with_file_line);
    assert!(dev.with_ansi);

    let prod = LogConfig::production();
    assert!(!prod.with_file_line);
    assert!(!prod.with_ansi);

    let test = LogConfig::testing();
    assert!(!test.with_timestamps);
}

#[test]
fn test_log_config_builder() {
    let config = LogConfig::new()
        .with_level(rust_ai_core::logging::LogLevel::Debug)
        .with_timestamps(false)
        .with_ansi(false);

    assert!(!config.with_timestamps);
    assert!(!config.with_ansi);
}

// ============================================================================
// End-to-End Workflow Tests
// ============================================================================

#[test]
fn test_memory_aware_tensor_creation() {
    let tracker = MemoryTracker::with_limit(10_000_000); // 10 MB limit

    // Estimate memory for a tensor
    let shape = [32, 64, 128];
    let estimated = tracker.estimate_with_overhead(&shape, DType::F32);

    // Check if it fits
    assert!(tracker.would_fit(estimated));

    // Record allocation
    tracker.allocate(estimated).expect("Should fit");

    // Create the tensor (on CPU for testing)
    let tensor = Tensor::zeros(&shape, DType::F32, &Device::Cpu).expect("Tensor creation");

    // Verify shape matches
    assert_eq!(tensor.dims(), shape);

    // Clean up
    tracker.deallocate(estimated);
}

#[test]
fn test_device_aware_workflow() {
    // Simulate a typical workflow: get device, create tensor, check dtype

    // 1. Get device (forced to CPU for test reliability)
    let config = DeviceConfig::new().with_force_cpu(true);
    let device = get_device(&config).expect("Device selection");

    // 2. Create tensor on device
    let tensor = Tensor::randn(0f32, 1f32, &[4, 8], &device).expect("Tensor creation");

    // 3. Verify dtype properties
    let dtype = tensor.dtype();
    assert!(dtype.is_training_dtype());
    assert!(!dtype.is_half_precision());
    assert_eq!(dtype.accumulator_dtype(), DType::F32);

    // 4. Estimate memory
    let bytes = estimate_tensor_bytes(tensor.dims(), dtype);
    assert_eq!(bytes, 4 * 8 * 4); // 4 * 8 elements * 4 bytes
}

// ============================================================================
// GPU Tests (Ignored by Default)
// ============================================================================

#[test]
#[ignore = "Requires CUDA GPU"]
fn test_get_device_returns_cuda_when_available() {
    let config = DeviceConfig::new().with_crate_name("gpu-test");
    let device = get_device(&config).expect("Device selection");

    // On a machine with CUDA, this should return a CUDA device
    assert!(matches!(device, Device::Cuda(_)));
}

#[test]
#[ignore = "Requires CUDA GPU"]
fn test_cuda_device_tensor_operations() {
    let config = DeviceConfig::new();
    let device = get_device(&config).expect("Device selection");

    if matches!(device, Device::Cuda(_)) {
        // Create tensor on GPU
        let tensor = Tensor::randn(0f32, 1f32, &[32, 64], &device).expect("GPU tensor");
        assert!(matches!(tensor.device(), Device::Cuda(_)));

        // Perform operation
        let result = tensor.sqr().expect("GPU operation");
        assert_eq!(result.dims(), &[32, 64]);
    }
}

// ============================================================================
// GpuDispatchable Trait Tests
// ============================================================================

/// Test implementation of `GpuDispatchable` for verification.
struct MockGpuOp {
    gpu_calls: std::sync::atomic::AtomicUsize,
    cpu_calls: std::sync::atomic::AtomicUsize,
}

impl MockGpuOp {
    fn new() -> Self {
        Self {
            gpu_calls: std::sync::atomic::AtomicUsize::new(0),
            cpu_calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn gpu_call_count(&self) -> usize {
        self.gpu_calls.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn cpu_call_count(&self) -> usize {
        self.cpu_calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl GpuDispatchable for MockGpuOp {
    type Input = Tensor;
    type Output = Tensor;

    fn dispatch_gpu(&self, input: &Self::Input, _device: &Device) -> Result<Self::Output> {
        self.gpu_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // Simulate GPU operation
        Ok((input * 2.0)?)
    }

    fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> Result<Self::Output> {
        self.cpu_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        warn_if_cpu(device, "mock-gpu-op");
        // Simulate CPU fallback
        Ok((input * 2.0)?)
    }
}

#[test]
fn test_gpu_dispatchable_cpu_routing() {
    let device = Device::Cpu;
    let op = MockGpuOp::new();
    let input = Tensor::new(&[1.0f32, 2.0, 3.0], &device).expect("input tensor");

    // dispatch() should route to CPU on CPU device
    let result = op.dispatch(&input, &device).expect("dispatch");

    assert_eq!(op.cpu_call_count(), 1);
    assert_eq!(op.gpu_call_count(), 0);

    let values: Vec<f32> = result.to_vec1().expect("result values");
    assert_eq!(values, vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_gpu_dispatchable_gpu_availability() {
    let op = MockGpuOp::new();

    // gpu_available() returns true only when compiled with cuda feature
    // Without CUDA, should return false
    #[cfg(not(feature = "cuda"))]
    assert!(!op.gpu_available());
}

// ============================================================================
// Quantize/Dequantize Trait Tests
// ============================================================================

/// Simple int8 quantized representation for testing.
#[derive(Debug)]
struct MockQuantized {
    data: Vec<i8>,
    scale: f32,
    zero_point: f32,
    shape: Vec<usize>,
}

/// Mock quantizer implementing both Quantize and Dequantize.
struct MockQuantizer;

impl Quantize<MockQuantized> for MockQuantizer {
    fn quantize(&self, tensor: &Tensor, _device: &Device) -> Result<MockQuantized> {
        let flat = tensor.flatten_all()?;
        let values: Vec<f32> = flat.to_vec1()?;

        // Simple min-max quantization using unsigned range [0, 255]
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let scale = if (max - min).abs() < 1e-8 {
            1.0
        } else {
            (max - min) / 255.0
        };
        let zero_point = min;

        // Map to unsigned [0, 255] then shift to signed [-128, 127]
        let data: Vec<i8> = values
            .iter()
            .map(|&v| {
                let unsigned = ((v - zero_point) / scale).round().clamp(0.0, 255.0) as u8;
                (i16::from(unsigned) - 128) as i8
            })
            .collect();

        Ok(MockQuantized {
            data,
            scale,
            zero_point,
            shape: tensor.shape().dims().to_vec(),
        })
    }
}

impl Dequantize<MockQuantized> for MockQuantizer {
    fn dequantize(&self, quantized: &MockQuantized, device: &Device) -> Result<Tensor> {
        // Reverse: signed [-128, 127] back to unsigned [0, 255] then to float
        let values: Vec<f32> = quantized
            .data
            .iter()
            .map(|&v| {
                let unsigned = f32::from(i16::from(v) + 128);
                unsigned * quantized.scale + quantized.zero_point
            })
            .collect();

        let tensor = Tensor::from_vec(values, quantized.shape.as_slice(), device)?;
        Ok(tensor)
    }
}

#[test]
fn test_quantize_trait_basic() {
    let device = Device::Cpu;
    let quantizer = MockQuantizer;

    // Create a simple tensor
    let original = Tensor::new(&[0.0f32, 0.5, 1.0, 1.5, 2.0], &device).expect("input tensor");

    // Quantize
    let quantized = quantizer
        .quantize(&original, &device)
        .expect("quantization");

    assert_eq!(quantized.data.len(), 5);
    assert_eq!(quantized.shape, vec![5]);
    assert!(quantized.scale > 0.0);
}

#[test]
fn test_dequantize_trait_basic() {
    let device = Device::Cpu;
    let quantizer = MockQuantizer;

    // Create input
    let original = Tensor::new(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &device).expect("input tensor");
    let original_values: Vec<f32> = original.to_vec1().expect("original values");

    // Quantize then dequantize
    let quantized = quantizer
        .quantize(&original, &device)
        .expect("quantization");
    let restored = quantizer
        .dequantize(&quantized, &device)
        .expect("dequantization");
    let restored_values: Vec<f32> = restored.to_vec1().expect("restored values");

    // Values should be close (with some quantization error)
    for (o, r) in original_values.iter().zip(restored_values.iter()) {
        assert!(
            (o - r).abs() < 0.1,
            "Original {o} differs from restored {r} by more than 0.1"
        );
    }
}

#[test]
fn test_quantize_dequantize_round_trip_shape() {
    let device = Device::Cpu;
    let quantizer = MockQuantizer;

    // Test with 2D tensor
    let original = Tensor::randn(0f32, 1.0, &[4, 8], &device).expect("input tensor");

    let quantized = quantizer
        .quantize(&original, &device)
        .expect("quantization");
    let restored = quantizer
        .dequantize(&quantized, &device)
        .expect("dequantization");

    // Shape should be preserved
    assert_eq!(original.dims(), restored.dims());
}

#[test]
fn test_quantize_constant_tensor() {
    let device = Device::Cpu;
    let quantizer = MockQuantizer;

    // Constant tensor (all same value) - edge case for scale calculation
    let constant = Tensor::new(&[5.0f32, 5.0, 5.0, 5.0], &device).expect("constant tensor");

    let quantized = quantizer
        .quantize(&constant, &device)
        .expect("quantization");
    let restored = quantizer
        .dequantize(&quantized, &device)
        .expect("dequantization");

    let restored_values: Vec<f32> = restored.to_vec1().expect("restored values");

    // All values should be approximately 5.0
    for v in restored_values {
        assert!((v - 5.0).abs() < 0.1, "Expected ~5.0, got {v}");
    }
}

// ============================================================================
// CubeCL Interop Tests (Ignored - Requires CUDA)
// ============================================================================

#[test]
#[ignore = "Requires CUDA GPU"]
fn test_cubecl_candle_to_cubecl_handle_f32() {
    #[cfg(feature = "cuda")]
    {
        use rust_ai_core::{candle_to_cubecl_handle, cubecl_to_candle_tensor};

        let device = get_device(&DeviceConfig::new()).expect("device");
        if matches!(device, Device::Cuda(_)) {
            let tensor = Tensor::randn(0f32, 1.0, &[4, 8], &device).expect("GPU tensor");
            let buffer = candle_to_cubecl_handle(&tensor).expect("conversion to CubeCL");

            assert_eq!(buffer.shape, vec![4, 8]);
            assert_eq!(buffer.dtype, DType::F32);
            assert_eq!(buffer.bytes.len(), 4 * 8 * 4); // 4*8 elements * 4 bytes/f32
        }
    }
}

#[test]
#[ignore = "Requires CUDA GPU"]
fn test_cubecl_round_trip_f32() {
    #[cfg(feature = "cuda")]
    {
        use rust_ai_core::{candle_to_cubecl_handle, cubecl_to_candle_tensor};

        let device = get_device(&DeviceConfig::new()).expect("device");
        if matches!(device, Device::Cuda(_)) {
            let original = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device).expect("GPU tensor");
            let original_values: Vec<f32> = original.to_vec1().expect("original values");

            // Convert to CubeCL buffer and back
            let buffer = candle_to_cubecl_handle(&original).expect("to CubeCL");
            let restored = cubecl_to_candle_tensor(&buffer, &device).expect("from CubeCL");
            let restored_values: Vec<f32> = restored.to_vec1().expect("restored values");

            assert_eq!(original_values, restored_values);
        }
    }
}

#[test]
#[ignore = "Requires CUDA GPU"]
fn test_cubecl_round_trip_bf16() {
    #[cfg(feature = "cuda")]
    {
        use rust_ai_core::{candle_to_cubecl_handle, cubecl_to_candle_tensor};

        let device = get_device(&DeviceConfig::new()).expect("device");
        if matches!(device, Device::Cuda(_)) {
            // Create bf16 tensor
            let f32_tensor = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device).expect("f32 tensor");
            let bf16_tensor = f32_tensor.to_dtype(DType::BF16).expect("convert to bf16");

            let buffer = candle_to_cubecl_handle(&bf16_tensor).expect("to CubeCL");

            assert_eq!(buffer.dtype, DType::BF16);
            assert_eq!(buffer.bytes.len(), 4 * 2); // 4 elements * 2 bytes/bf16

            let restored = cubecl_to_candle_tensor(&buffer, &device).expect("from CubeCL");
            assert_eq!(restored.dtype(), DType::BF16);
        }
    }
}

#[test]
#[ignore = "Requires CUDA GPU"]
fn test_cubecl_allocate_output_buffer() {
    #[cfg(feature = "cuda")]
    {
        use rust_ai_core::allocate_output_buffer;

        let buffer = allocate_output_buffer(&[16, 32], DType::F32).expect("allocate buffer");

        assert_eq!(buffer.shape, vec![16, 32]);
        assert_eq!(buffer.dtype, DType::F32);
        assert_eq!(buffer.bytes.len(), 16 * 32 * 4);

        // Buffer should be zero-initialized
        for &byte in &buffer.bytes {
            assert_eq!(byte, 0);
        }
    }
}

#[test]
#[ignore = "Requires CUDA GPU"]
fn test_cubecl_has_cuda_support() {
    #[cfg(feature = "cuda")]
    {
        use rust_ai_core::has_cubecl_cuda_support;

        // When cuda feature is enabled and CUDA is present, should return true
        // This test is ignored by default; when run on CUDA hardware, should pass
        let has_cuda = has_cubecl_cuda_support();
        assert!(has_cuda, "Expected CUDA support when running ignored test");
    }
}
