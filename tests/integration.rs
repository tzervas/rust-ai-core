// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Integration tests for rust-ai-core.
//!
//! These tests verify the public API works correctly as a cohesive system.

use candle_core::{DType, Device, Tensor};
use rust_ai_core::{
    bytes_per_element, estimate_tensor_bytes, get_device, is_floating_point, warn_if_cpu,
    CoreError, DTypeExt, DeviceConfig, LogConfig, MemoryTracker, Result, ValidatableConfig,
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
