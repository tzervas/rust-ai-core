// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Unified logging and observability for the rust-ai ecosystem.
//!
//! ## Why This Module Exists
//!
//! ML training and inference generate massive amounts of diagnostic data:
//! - Training progress (loss, learning rate, throughput)
//! - Memory usage and allocation patterns
//! - Device information and kernel timings
//! - Errors and warnings
//!
//! Without consistent logging configuration, each crate would implement its own
//! approach, leading to inconsistent output formats and configuration mechanisms.
//!
//! This module provides:
//!
//! 1. **Unified log initialization**: Single function to configure logging for all crates
//! 2. **Structured logging helpers**: Consistent field names for metrics and events
//! 3. **Progress tracking**: Training loop progress bars and ETA estimation
//!
//! ## Design Decisions
//!
//! - **tracing-based**: Uses the `tracing` ecosystem for structured logging with spans
//! - **Environment-driven**: Log levels configured via `RUST_LOG` environment variable
//! - **Zero-cost when disabled**: All logging compiles to no-ops when level is filtered

use std::sync::Once;

/// Configuration for logging initialization.
///
/// ## Why This Struct
///
/// Logging configuration often needs to vary between development and production:
/// - Development: verbose, colored output to terminal
/// - Production: JSON structured logs to file/stdout
/// - Testing: minimal output, captured by test harness
///
/// This struct captures these variations.
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Default log level when `RUST_LOG` is not set.
    pub default_level: LogLevel,
    /// Include timestamps in log output.
    pub with_timestamps: bool,
    /// Include target (module path) in log output.
    pub with_target: bool,
    /// Include source file and line numbers.
    pub with_file_line: bool,
    /// Use ANSI colors (disable for file output).
    pub with_ansi: bool,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            default_level: LogLevel::Info,
            with_timestamps: true,
            with_target: true,
            with_file_line: false,
            with_ansi: true,
        }
    }
}

impl LogConfig {
    /// Create a new logging configuration with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the default log level.
    #[must_use]
    pub fn with_level(mut self, level: LogLevel) -> Self {
        self.default_level = level;
        self
    }

    /// Enable or disable timestamps.
    #[must_use]
    pub fn with_timestamps(mut self, enable: bool) -> Self {
        self.with_timestamps = enable;
        self
    }

    /// Enable or disable ANSI colors.
    #[must_use]
    pub fn with_ansi(mut self, enable: bool) -> Self {
        self.with_ansi = enable;
        self
    }

    /// Configuration preset for development.
    ///
    /// Verbose output with colors, file/line info for debugging.
    #[must_use]
    pub fn development() -> Self {
        Self {
            default_level: LogLevel::Debug,
            with_timestamps: true,
            with_target: true,
            with_file_line: true,
            with_ansi: true,
        }
    }

    /// Configuration preset for production.
    ///
    /// Clean output without colors (for structured log ingestion).
    #[must_use]
    pub fn production() -> Self {
        Self {
            default_level: LogLevel::Info,
            with_timestamps: true,
            with_target: false,
            with_file_line: false,
            with_ansi: false,
        }
    }

    /// Configuration preset for testing.
    ///
    /// Minimal output, captured by test harness.
    #[must_use]
    pub fn testing() -> Self {
        Self {
            default_level: LogLevel::Warn,
            with_timestamps: false,
            with_target: false,
            with_file_line: false,
            with_ansi: false,
        }
    }
}

/// Log level enumeration.
///
/// Maps to tracing levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogLevel {
    /// Errors only.
    Error,
    /// Warnings and above.
    Warn,
    /// Informational messages and above.
    #[default]
    Info,
    /// Debug messages and above.
    Debug,
    /// All messages including trace.
    Trace,
}

impl LogLevel {
    /// Convert to a tracing filter string.
    fn as_filter_str(&self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}

/// Guard ensuring logging is only initialized once.
static INIT_LOGGING: Once = Once::new();

/// Initialize logging for the rust-ai ecosystem.
///
/// This function should be called once at application startup. It configures
/// the global tracing subscriber based on the provided configuration.
///
/// ## Arguments
///
/// * `config` - Logging configuration
///
/// ## Why Single Initialization
///
/// The tracing subscriber is global. Multiple initialization attempts would
/// either panic or silently fail. This function uses `Once` to ensure safe
/// idempotent calls.
///
/// ## Environment Override
///
/// The `RUST_LOG` environment variable always takes precedence over the
/// config's default level. This allows runtime tuning without recompilation.
///
/// ## Example
///
/// ```rust
/// use rust_ai_core::{init_logging, LogConfig};
///
/// // Initialize with defaults
/// init_logging(&LogConfig::default());
///
/// // Or with explicit config
/// init_logging(&LogConfig::development());
/// ```
pub fn init_logging(config: &LogConfig) {
    INIT_LOGGING.call_once(|| {
        // Use RUST_LOG if set, otherwise fall back to config default
        let filter = std::env::var("RUST_LOG")
            .unwrap_or_else(|_| config.default_level.as_filter_str().to_string());

        // Build the subscriber
        let builder = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_ansi(config.with_ansi)
            .with_target(config.with_target)
            .with_file(config.with_file_line)
            .with_line_number(config.with_file_line);

        // Apply timestamp configuration
        if config.with_timestamps {
            builder.init();
        } else {
            builder.without_time().init();
        }
    });
}

/// Macro to log a training metric with consistent field names.
///
/// ## Why This Macro
///
/// Training metrics need consistent field names for downstream processing
/// (`TensorBoard`, Weights & Biases, etc.). This macro ensures all crates
/// use the same schema.
///
/// ## Example
///
/// ```rust,ignore
/// log_metric!(
///     step = 1000,
///     loss = 2.5,
///     lr = 1e-4,
///     throughput_tokens_sec = 50000.0
/// );
/// ```
#[macro_export]
macro_rules! log_metric {
    ($($field:ident = $value:expr),+ $(,)?) => {
        tracing::info!(
            target: "rust_ai::metrics",
            $($field = $value),+
        );
    };
}

/// Log a training step with standard fields.
///
/// ## Arguments
///
/// * `step` - Current training step
/// * `total_steps` - Total steps for progress calculation
/// * `loss` - Current loss value
/// * `lr` - Current learning rate
///
/// ## Why This Function
///
/// Every training loop logs steps. Having a dedicated function ensures
/// consistent formatting and field names across all crates.
pub fn log_training_step(step: usize, total_steps: usize, loss: f64, lr: f64) {
    let progress_pct = if total_steps > 0 {
        (step as f64 / total_steps as f64) * 100.0
    } else {
        0.0
    };

    tracing::info!(
        target: "rust_ai::training",
        step,
        total_steps,
        progress_pct = format!("{progress_pct:.1}"),
        loss = format!("{loss:.6}"),
        lr = format!("{lr:.2e}"),
        "Training step"
    );
}

/// Log memory usage.
///
/// ## Arguments
///
/// * `allocated_bytes` - Currently allocated bytes
/// * `peak_bytes` - Peak allocation
/// * `context` - Description of what operation is being tracked
pub fn log_memory_usage(allocated_bytes: usize, peak_bytes: usize, context: &str) {
    let allocated_mb = allocated_bytes as f64 / (1024.0 * 1024.0);
    let peak_mb = peak_bytes as f64 / (1024.0 * 1024.0);

    tracing::debug!(
        target: "rust_ai::memory",
        allocated_mb = format!("{allocated_mb:.2}"),
        peak_mb = format!("{peak_mb:.2}"),
        context,
        "Memory usage"
    );
}

// Re-export tracing macros for convenience so crates don't need to depend on tracing directly
pub use tracing::{debug, error, info, trace, warn};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_config_default() {
        let config = LogConfig::default();
        assert!(matches!(config.default_level, LogLevel::Info));
        assert!(config.with_timestamps);
        assert!(config.with_ansi);
    }

    #[test]
    fn test_log_config_builder() {
        let config = LogConfig::new()
            .with_level(LogLevel::Debug)
            .with_timestamps(false)
            .with_ansi(false);

        assert!(matches!(config.default_level, LogLevel::Debug));
        assert!(!config.with_timestamps);
        assert!(!config.with_ansi);
    }

    #[test]
    fn test_log_config_presets() {
        let dev = LogConfig::development();
        assert!(matches!(dev.default_level, LogLevel::Debug));
        assert!(dev.with_file_line);

        let prod = LogConfig::production();
        assert!(matches!(prod.default_level, LogLevel::Info));
        assert!(!prod.with_ansi);

        let test = LogConfig::testing();
        assert!(matches!(test.default_level, LogLevel::Warn));
        assert!(!test.with_timestamps);
    }

    #[test]
    fn test_log_level_filter_str() {
        assert_eq!(LogLevel::Error.as_filter_str(), "error");
        assert_eq!(LogLevel::Warn.as_filter_str(), "warn");
        assert_eq!(LogLevel::Info.as_filter_str(), "info");
        assert_eq!(LogLevel::Debug.as_filter_str(), "debug");
        assert_eq!(LogLevel::Trace.as_filter_str(), "trace");
    }
}
