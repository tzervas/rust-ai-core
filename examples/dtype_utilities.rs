//! Example: Data Type Utilities
//!
//! This example demonstrates how to use rust-ai-core's dtype utilities
//! for memory calculation, precision management, and dtype introspection.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example dtype_utilities
//! ```

#![allow(clippy::cast_precision_loss)] // Example code - precision loss in display is acceptable

use candle_core::DType;
use rust_ai_core::{bytes_per_element, is_floating_point, DTypeExt};

fn main() {
    println!("=== Data Type Utilities Example ===\n");

    // 1. Basic dtype introspection
    println!("1. Bytes per element:");
    let dtypes = [DType::F32, DType::F16, DType::BF16, DType::U8, DType::I64];
    for dtype in dtypes {
        println!("   {dtype:?}: {} bytes", bytes_per_element(dtype));
    }
    println!();

    // 2. Floating point detection
    println!("2. Floating point detection:");
    let all_dtypes = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::U8,
        DType::U32,
        DType::I64,
    ];
    for dtype in all_dtypes {
        let is_fp = is_floating_point(dtype);
        println!("   {dtype:?}: floating_point={is_fp}");
    }
    println!();

    // 3. DTypeExt trait methods
    println!("3. DTypeExt extension trait:");
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::I64] {
        println!("   {dtype:?}:");
        println!("      name: {}", dtype.name());
        println!("      half_precision: {}", dtype.is_half_precision());
        println!("      training_dtype: {}", dtype.is_training_dtype());
        println!("      integer: {}", dtype.is_integer());
        println!("      accumulator: {:?}", dtype.accumulator_dtype());
    }
    println!();

    // 4. Practical use case: memory calculation for a model layer
    println!("4. Memory calculation for model layer:");
    let hidden_dim = 4096;
    let intermediate_dim = 11008;
    let batch_size = 8;
    let seq_len = 2048;

    // Calculate memory for different precisions
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        let bytes = bytes_per_element(dtype);

        // Weight memory (hidden_dim x intermediate_dim)
        let weight_mem = hidden_dim * intermediate_dim * bytes;

        // Activation memory (batch x seq x hidden)
        let activation_mem = batch_size * seq_len * hidden_dim * bytes;

        let weight_mb = weight_mem as f64 / 1_000_000.0;
        let activation_mb = activation_mem as f64 / 1_000_000.0;
        println!("   {dtype:?}: weights={weight_mb:.1} MB, activations={activation_mb:.1} MB");
    }
    println!();

    // 5. Mixed precision accumulator selection
    println!("5. Mixed precision accumulators:");
    println!("   Why: Half-precision types need FP32 accumulators to prevent overflow");
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        let acc = dtype.accumulator_dtype();
        println!("   {dtype:?} -> accumulator {acc:?} (for numerical stability)");
    }
    println!();

    println!("=== Example Complete ===");
}
