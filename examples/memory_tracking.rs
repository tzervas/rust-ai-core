//! Example: Memory Tracking
//!
//! This example demonstrates how to track GPU memory usage during
//! tensor operations to prevent out-of-memory errors.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example memory_tracking
//! ```

#![allow(clippy::cast_precision_loss)] // Example code - precision loss in MB display is acceptable

use candle_core::{DType, Device, Tensor};
use rust_ai_core::{
    estimate_tensor_bytes,
    memory::{estimate_attention_memory, DEFAULT_OVERHEAD_FACTOR},
    MemoryTracker, Result,
};

fn main() -> Result<()> {
    println!("=== Memory Tracking Example ===\n");

    // Create a tracker with 1GB limit (simulating a GPU budget)
    let one_gb = 1024 * 1024 * 1024;
    let tracker = MemoryTracker::with_limit(one_gb).with_overhead_factor(DEFAULT_OVERHEAD_FACTOR);

    println!(
        "Memory budget: {} MB\n",
        tracker.limit_bytes() / (1024 * 1024)
    );

    // Estimate memory for various tensor sizes
    println!("--- Tensor Memory Estimation ---");

    let shapes = [
        (
            &[1, 512, 4096][..],
            "Embedding (batch=1, seq=512, dim=4096)",
        ),
        (
            &[1, 32, 512, 128][..],
            "Attention QKV (batch=1, heads=32, seq=512, head_dim=128)",
        ),
        (
            &[1, 4096, 11008][..],
            "MLP hidden (batch=1, seq=4096, hidden=11008)",
        ),
    ];

    for (shape, desc) in shapes {
        let bytes_f32 = estimate_tensor_bytes(shape, DType::F32);
        let bytes_bf16 = estimate_tensor_bytes(shape, DType::BF16);
        let overhead = tracker.estimate_with_overhead(shape, DType::F32);

        println!("{desc}:");
        println!("  F32:  {:>8.2} MB", bytes_f32 as f64 / (1024.0 * 1024.0));
        println!("  BF16: {:>8.2} MB", bytes_bf16 as f64 / (1024.0 * 1024.0));
        println!(
            "  With overhead: {:>8.2} MB\n",
            overhead as f64 / (1024.0 * 1024.0)
        );
    }

    // Estimate attention memory (the O(n²) component)
    println!("--- Attention Memory Scaling ---");
    for seq_len in [512, 1024, 2048, 4096] {
        let attn_bytes = estimate_attention_memory(
            1,  // batch
            32, // heads
            seq_len,
            128, // head_dim
            DType::BF16,
        );
        println!(
            "seq_len={seq_len}: {:>8.2} MB",
            attn_bytes as f64 / (1024.0 * 1024.0)
        );
    }
    println!();

    // Simulate a training step with memory tracking
    println!("--- Simulated Training Step ---");
    let device = Device::Cpu; // Use CPU for example

    // Step 1: Allocate embedding output
    let embed_shape = [1, 512, 4096];
    let embed_bytes = tracker.estimate_with_overhead(&embed_shape, DType::F32);

    if tracker.would_fit(embed_bytes) {
        tracker.allocate(embed_bytes)?;
        println!(
            "Allocated embedding: {:.2} MB",
            embed_bytes as f64 / (1024.0 * 1024.0)
        );
    }

    // Step 2: Allocate attention tensors
    let attn_shape = [1, 32, 512, 128];
    let attn_bytes = tracker.estimate_with_overhead(&attn_shape, DType::F32);

    // Allocate Q, K, V
    for name in ["Q", "K", "V"] {
        if tracker.would_fit(attn_bytes) {
            tracker.allocate(attn_bytes)?;
            println!(
                "Allocated {name}: {:.2} MB",
                attn_bytes as f64 / (1024.0 * 1024.0)
            );
        }
    }

    // Check current usage
    println!(
        "\nCurrent allocation: {:.2} MB",
        tracker.allocated_bytes() as f64 / (1024.0 * 1024.0)
    );
    println!(
        "Peak allocation: {:.2} MB",
        tracker.peak_bytes() as f64 / (1024.0 * 1024.0)
    );

    // Simulate freeing after forward pass
    tracker.deallocate(attn_bytes * 2); // Free K, V
    println!(
        "After freeing K, V: {:.2} MB\n",
        tracker.allocated_bytes() as f64 / (1024.0 * 1024.0)
    );

    // Create actual tensors to verify estimates
    println!("--- Actual Tensor Creation ---");
    let tensor = Tensor::zeros(&[1, 512, 4096], DType::F32, &device)?;
    println!(
        "Created tensor shape: {:?}, dtype: {:?}",
        tensor.dims(),
        tensor.dtype()
    );

    println!("\n=== Example Complete ===");
    Ok(())
}
