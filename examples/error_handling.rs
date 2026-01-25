//! Example: Error Handling
//!
//! This example demonstrates proper error handling patterns
//! using rust-ai-core's unified error types.
//!
//! Run with:
//! ```bash
//! cargo run --example error_handling
//! ```

#![allow(clippy::items_after_statements)]

use rust_ai_core::{CoreError, Result, ValidatableConfig};

/// Example configuration struct for a `LoRA` adapter.
#[derive(Clone)]
struct LoraConfig {
    rank: usize,
    alpha: f32,
    dropout: f32,
}

impl ValidatableConfig for LoraConfig {
    fn validate(&self) -> Result<()> {
        if self.rank == 0 {
            return Err(CoreError::invalid_config(
                "rank must be greater than 0 (typical values: 4, 8, 16, 32)",
            ));
        }
        if self.alpha <= 0.0 {
            return Err(CoreError::invalid_config(
                "alpha must be positive (typical values: rank * 2)",
            ));
        }
        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(CoreError::invalid_config(
                "dropout must be between 0.0 and 1.0",
            ));
        }
        Ok(())
    }
}

/// Example of a domain-specific error that wraps `CoreError`.
#[derive(Debug, thiserror::Error)]
#[allow(dead_code)] // Some variants only shown for documentation
enum AdapterError {
    #[error("adapter not found: {0}")]
    NotFound(String),

    #[error("adapter already exists: {0}")]
    AlreadyExists(String),

    #[error(transparent)]
    Core(#[from] CoreError),
}

fn create_adapter(config: &LoraConfig) -> std::result::Result<(), AdapterError> {
    // Validate configuration
    config.validate()?; // CoreError auto-converts to AdapterError

    // Simulate adapter creation
    println!(
        "Created adapter with rank={}, alpha={}",
        config.rank, config.alpha
    );
    Ok(())
}

fn load_adapter(name: &str) -> std::result::Result<LoraConfig, AdapterError> {
    if name == "missing" {
        return Err(AdapterError::NotFound(name.to_string()));
    }

    Ok(LoraConfig {
        rank: 16,
        alpha: 32.0,
        dropout: 0.1,
    })
}

fn main() {
    println!("=== Error Handling Example ===\n");

    // Example 1: Valid configuration
    println!("1. Valid configuration:");
    let valid_config = LoraConfig {
        rank: 16,
        alpha: 32.0,
        dropout: 0.1,
    };
    match create_adapter(&valid_config) {
        Ok(()) => println!("   Success!\n"),
        Err(e) => println!("   Error: {e}\n"),
    }

    // Example 2: Invalid rank
    println!("2. Invalid rank (0):");
    let invalid_rank = LoraConfig {
        rank: 0,
        alpha: 32.0,
        dropout: 0.1,
    };
    match create_adapter(&invalid_rank) {
        Ok(()) => println!("   Success!\n"),
        Err(e) => println!("   Error: {e}\n"),
    }

    // Example 3: Invalid alpha
    println!("3. Invalid alpha (negative):");
    let invalid_alpha = LoraConfig {
        rank: 16,
        alpha: -1.0,
        dropout: 0.1,
    };
    match create_adapter(&invalid_alpha) {
        Ok(()) => println!("   Success!\n"),
        Err(e) => println!("   Error: {e}\n"),
    }

    // Example 4: Domain-specific error
    println!("4. Domain-specific error (adapter not found):");
    match load_adapter("missing") {
        Ok(_) => println!("   Loaded!\n"),
        Err(e) => println!("   Error: {e}\n"),
    }

    // Example 5: Error helper constructors
    println!("5. Error helper constructors:");
    let errors = [
        CoreError::shape_mismatch(vec![2, 3], vec![3, 2]),
        CoreError::dim_mismatch("expected 3D tensor, got 2D"),
        CoreError::device_not_available("CUDA:1"),
        CoreError::oom("failed to allocate 16GB"),
        CoreError::kernel("kernel launch failed: invalid grid size"),
        CoreError::not_implemented("gradient checkpointing"),
        CoreError::io("config.yaml: file not found"),
    ];

    for err in errors {
        println!("   {err}");
    }
    println!();

    // Example 6: Error matching
    println!("6. Error pattern matching:");
    let err = CoreError::shape_mismatch(vec![1, 2, 3], vec![1, 3, 2]);
    match &err {
        CoreError::ShapeMismatch { expected, actual } => {
            println!("   Shape mismatch!");
            println!("   Expected: {expected:?}");
            println!("   Actual: {actual:?}");
        }
        _ => println!("   Other error: {err}"),
    }
    println!();

    // Example 7: Using the ? operator with Result
    println!("7. Using ? operator:");
    fn process_config(config: &LoraConfig) -> Result<String> {
        config.validate()?;
        Ok(format!(
            "Valid config: rank={}, alpha={}",
            config.rank, config.alpha
        ))
    }

    match process_config(&LoraConfig {
        rank: 8,
        alpha: 16.0,
        dropout: 0.0,
    }) {
        Ok(msg) => println!("   {msg}"),
        Err(e) => println!("   Error: {e}"),
    }

    println!("\n=== Example Complete ===");
}
