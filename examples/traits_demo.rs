//! Example: Traits Demonstration
//!
//! This example demonstrates how to implement and use rust-ai-core's
//! trait interfaces for configuration validation, quantization, and GPU dispatch.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example traits_demo
//! ```

#![allow(clippy::similar_names)] // quantizer/quantized are intentionally similar
#![allow(clippy::cast_possible_truncation)] // Example quantization - precision loss is expected

use candle_core::{Device, Tensor};
use rust_ai_core::{
    warn_if_cpu, CoreError, Dequantize, GpuDispatchable, Quantize, Result, ValidatableConfig,
};

// ============================================================================
// Example 1: ValidatableConfig trait
// ============================================================================

/// Configuration for a hypothetical `LoRA` adapter.
#[derive(Clone, Debug)]
struct LoraConfig {
    rank: usize,
    alpha: f32,
    dropout: f32,
}

impl ValidatableConfig for LoraConfig {
    fn validate(&self) -> Result<()> {
        if self.rank == 0 {
            return Err(CoreError::invalid_config("rank must be positive"));
        }
        if self.rank > 256 {
            return Err(CoreError::invalid_config("rank must be <= 256"));
        }
        if self.alpha <= 0.0 {
            return Err(CoreError::invalid_config("alpha must be positive"));
        }
        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(CoreError::invalid_config("dropout must be in [0, 1]"));
        }
        Ok(())
    }
}

// ============================================================================
// Example 2: Quantize/Dequantize traits
// ============================================================================

/// A simple 8-bit quantized tensor representation.
#[derive(Debug)]
struct Int8Tensor {
    data: Vec<i8>,
    scale: f32,
    shape: Vec<usize>,
}

/// Simple quantizer using min-max scaling.
struct SimpleQuantizer;

impl Quantize<Int8Tensor> for SimpleQuantizer {
    fn quantize(&self, tensor: &Tensor, _device: &Device) -> Result<Int8Tensor> {
        // Flatten and convert to f32
        let flat = tensor.flatten_all()?;
        let values: Vec<f32> = flat.to_vec1()?;

        // Find min/max for scaling
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = (max - min) / 255.0;

        // Quantize to int8
        #[allow(clippy::cast_possible_truncation)]
        let data: Vec<i8> = values
            .iter()
            .map(|&v| ((v - min) / scale - 128.0) as i8)
            .collect();

        Ok(Int8Tensor {
            data,
            scale,
            shape: tensor.shape().dims().to_vec(),
        })
    }
}

impl Dequantize<Int8Tensor> for SimpleQuantizer {
    fn dequantize(&self, quantized: &Int8Tensor, device: &Device) -> Result<Tensor> {
        // Dequantize back to f32
        let values: Vec<f32> = quantized
            .data
            .iter()
            .map(|&v| (f32::from(v) + 128.0) * quantized.scale)
            .collect();

        // Reconstruct tensor
        let tensor = Tensor::from_vec(values, quantized.shape.as_slice(), device)?;
        Ok(tensor)
    }
}

// ============================================================================
// Example 3: GpuDispatchable trait
// ============================================================================

/// A simple operation that can run on GPU or CPU.
struct VectorAdd;

impl GpuDispatchable for VectorAdd {
    type Input = (Tensor, Tensor);
    type Output = Tensor;

    fn dispatch_gpu(&self, (a, b): &Self::Input, _device: &Device) -> Result<Self::Output> {
        // In real code, this would use CubeCL kernels
        // For this example, we use Candle's built-in ops
        let result = (a + b)?;
        Ok(result)
    }

    fn dispatch_cpu(&self, (a, b): &Self::Input, device: &Device) -> Result<Self::Output> {
        // Emit warning when using CPU fallback
        warn_if_cpu(device, "traits_demo");

        // CPU implementation using Candle
        let result = (a + b)?;
        Ok(result)
    }
}

// ============================================================================
// Main demonstration
// ============================================================================

fn main() -> Result<()> {
    println!("=== Traits Demonstration ===\n");

    // Get a CPU device for this example
    let device = Device::Cpu;

    // ========================================================================
    // Demo 1: ValidatableConfig
    // ========================================================================
    println!("1. ValidatableConfig trait:");

    // Valid configuration
    let valid_config = LoraConfig {
        rank: 16,
        alpha: 32.0,
        dropout: 0.1,
    };
    match valid_config.validate() {
        Ok(()) => println!("   Valid config: {valid_config:?}"),
        Err(e) => println!("   Error: {e}"),
    }

    // Invalid configurations
    let invalid_configs = [
        LoraConfig {
            rank: 0,
            alpha: 32.0,
            dropout: 0.1,
        },
        LoraConfig {
            rank: 16,
            alpha: -1.0,
            dropout: 0.1,
        },
        LoraConfig {
            rank: 16,
            alpha: 32.0,
            dropout: 1.5,
        },
    ];
    for cfg in &invalid_configs {
        match cfg.validate() {
            Ok(()) => println!("   Unexpected: {cfg:?} should fail"),
            Err(e) => println!("   Expected error: {e}"),
        }
    }
    println!();

    // ========================================================================
    // Demo 2: Quantize/Dequantize
    // ========================================================================
    println!("2. Quantize/Dequantize traits:");

    let original = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &device)?;
    println!("   Original tensor: {:?}", original.to_vec1::<f32>()?);

    let quantizer = SimpleQuantizer;
    let quantized = quantizer.quantize(&original, &device)?;
    println!(
        "   Quantized (scale={}): {:?}",
        quantized.scale, quantized.data
    );

    let dequantized = quantizer.dequantize(&quantized, &device)?;
    println!("   Dequantized: {:?}", dequantized.to_vec1::<f32>()?);
    println!("   (Note: Some precision loss is expected with int8 quantization)");
    println!();

    // ========================================================================
    // Demo 3: GpuDispatchable
    // ========================================================================
    println!("3. GpuDispatchable trait:");

    let a = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
    let b = Tensor::new(&[4.0f32, 5.0, 6.0], &device)?;

    let op = VectorAdd;

    // Check GPU availability
    println!("   GPU available: {}", op.gpu_available());

    // Use dispatch() which auto-routes based on device
    let result = op.dispatch(&(a.clone(), b.clone()), &device)?;
    println!("   a = {:?}", a.to_vec1::<f32>()?);
    println!("   b = {:?}", b.to_vec1::<f32>()?);
    println!("   a + b = {:?}", result.to_vec1::<f32>()?);
    println!();

    // ========================================================================
    // Demo 4: Combining traits in a real workflow
    // ========================================================================
    println!("4. Combined workflow (config -> quantize -> compute):");

    // 1. Validate configuration
    let config = LoraConfig {
        rank: 8,
        alpha: 16.0,
        dropout: 0.0,
    };
    config.validate()?;
    println!("   Config validated: rank={}", config.rank);

    // 2. Create and quantize weights
    let weights = Tensor::randn(0f32, 1.0, (4, 4), &device)?;
    let quantized_weights = quantizer.quantize(&weights, &device)?;
    println!(
        "   Weights quantized: {} elements -> {} bytes",
        quantized_weights.data.len(),
        quantized_weights.data.len()
    );

    // 3. Dequantize for computation
    let restored_weights = quantizer.dequantize(&quantized_weights, &device)?;
    println!("   Weights restored for computation");

    // 4. Run computation with dispatch
    let input = Tensor::randn(0f32, 1.0, (4, 4), &device)?;
    let output = op.dispatch(&(input, restored_weights), &device)?;
    println!("   Computation complete: output shape {:?}", output.shape());
    println!();

    println!("=== Example Complete ===");
    Ok(())
}
