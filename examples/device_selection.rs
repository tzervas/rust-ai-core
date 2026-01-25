//! Example: Device Selection
//!
//! This example demonstrates how to use rust-ai-core's device selection
//! with CUDA-first philosophy and environment variable configuration.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example device_selection
//! RUST_AI_FORCE_CPU=1 cargo run --example device_selection
//! RUST_AI_CUDA_DEVICE=1 cargo run --example device_selection
//! ```

use rust_ai_core::{get_device, warn_if_cpu, DeviceConfig, Result};

fn main() -> Result<()> {
    println!("=== Device Selection Example ===\n");

    // Method 1: Default configuration (CUDA device 0, auto-fallback)
    println!("1. Default configuration:");
    let config = DeviceConfig::default();
    let device = get_device(&config)?;
    println!("   Device: {device:?}\n");

    // Method 2: Explicit builder pattern
    println!("2. Builder pattern:");
    let config = DeviceConfig::new()
        .with_cuda_device(0)
        .with_crate_name("my-app");
    let device = get_device(&config)?;
    println!("   Device: {device:?}\n");

    // Method 3: From environment variables
    println!("3. From environment (check RUST_AI_FORCE_CPU, RUST_AI_CUDA_DEVICE):");
    let config = DeviceConfig::from_env();
    println!("   force_cpu: {}", config.force_cpu);
    println!("   cuda_device: {}", config.cuda_device);
    let device = get_device(&config)?;
    println!("   Device: {device:?}\n");

    // Method 4: Force CPU for testing
    println!("4. Force CPU (for testing):");
    let config = DeviceConfig::new().with_force_cpu(true);
    let device = get_device(&config)?;
    println!("   Device: {device:?}\n");

    // Demonstrate the warning mechanism
    println!("5. Warning on CPU usage:");
    warn_if_cpu(&device, "example-app");
    warn_if_cpu(&device, "example-app"); // Second call is silenced
    println!("   (Warning printed only once)\n");

    println!("=== Example Complete ===");
    Ok(())
}
