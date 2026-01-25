//! Example: Logging Setup
//!
//! This example demonstrates how to use rust-ai-core's logging utilities
//! for configuring tracing, logging training progress, and memory usage.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example logging_setup
//! RUST_LOG=debug cargo run --example logging_setup
//! ```

#![allow(clippy::cast_precision_loss)] // Example code - precision loss acceptable for display

use rust_ai_core::{
    debug, error, info, init_logging, log_memory_usage, log_training_step, trace, warn, LogConfig,
};

fn main() {
    println!("=== Logging Setup Example ===\n");

    // 1. Default logging configuration
    println!("1. Default configuration:");
    let config = LogConfig::default();
    println!("   level: info");
    println!("   timestamps: true");
    println!("   ansi colors: true\n");

    // 2. Builder pattern for custom configuration
    println!("2. Builder pattern:");
    let _config = LogConfig::new()
        .with_level(rust_ai_core::logging::LogLevel::Debug)
        .with_timestamps(true)
        .with_ansi(true);
    println!("   LogConfig::new().with_level(Debug).with_timestamps(true)\n");

    // 3. Preset configurations
    println!("3. Preset configurations:");
    let _dev = LogConfig::development();
    println!("   LogConfig::development() - Debug level, timestamps, colors");
    let _prod = LogConfig::production();
    println!("   LogConfig::production() - Info level, timestamps, no colors");
    let _test = LogConfig::testing();
    println!("   LogConfig::testing() - Warn level, no timestamps, no colors\n");

    // 4. Initialize logging (only do this once per process)
    println!("4. Initializing logging (this affects the rest of the example):");
    init_logging(&config);
    println!("   init_logging(&config) called\n");

    // 5. Using tracing macros
    println!("5. Using tracing macros (output appears in stderr):");
    trace!("This is a trace message (usually hidden)");
    debug!("This is a debug message");
    info!("This is an info message");
    warn!("This is a warning message");
    error!("This is an error message");
    println!();

    // 6. Structured logging with fields
    println!("6. Structured logging with fields:");
    let batch_size = 32;
    let learning_rate = 0.001;
    info!(batch_size, learning_rate, "Training configuration");
    println!();

    // 7. Training step logging helper
    println!("7. Training step logging:");
    for step in [0, 100, 500, 1000] {
        let loss = 2.5 - (step as f64 * 0.002);
        let lr = 0.001 * (1.0 - step as f64 / 1000.0);
        log_training_step(step, 1000, loss, lr);
    }
    println!();

    // 8. Memory usage logging helper
    println!("8. Memory usage logging:");
    let allocated = 4 * 1024 * 1024 * 1024; // 4 GB
    let peak = 6 * 1024 * 1024 * 1024; // 6 GB peak
    log_memory_usage(allocated, peak, "after forward pass");
    println!();

    // 9. Environment variable override
    println!("9. Environment variable configuration:");
    println!("   Set RUST_LOG to override log levels:");
    println!("     RUST_LOG=trace           - all messages");
    println!("     RUST_LOG=debug           - debug and above");
    println!("     RUST_LOG=info            - info and above (default)");
    println!("     RUST_LOG=rust_ai_core=debug,warn - per-module control");
    println!();

    println!("=== Example Complete ===");
}
