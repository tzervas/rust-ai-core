// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! CUDA-first device selection with environment variable overrides.
//!
//! This module provides unified device selection logic across all rust-ai crates.
//! The philosophy is **CUDA-first**: GPU is always preferred, and CPU fallback
//! triggers a warning to alert users they're not getting optimal performance.
//!
//! ## Environment Variables
//!
//! - `RUST_AI_FORCE_CPU` - Set to `1` or `true` to force CPU execution
//! - `RUST_AI_CUDA_DEVICE` - Set to device ordinal (e.g., `0`, `1`) to select GPU
//!
//! Legacy variables are also supported for backwards compatibility:
//! - `AXOLOTL_FORCE_CPU`, `VSA_OPTIM_FORCE_CPU`
//! - `AXOLOTL_CUDA_DEVICE`, `VSA_OPTIM_CUDA_DEVICE`
//!
//! ## Example
//!
//! ```rust
//! use rust_ai_core::{get_device, DeviceConfig};
//!
//! // Default: CUDA device 0 with auto-fallback
//! let device = get_device(&DeviceConfig::default())?;
//!
//! // Explicit GPU selection
//! let config = DeviceConfig::new().with_cuda_device(1);
//! let device = get_device(&config)?;
//!
//! // Force CPU (for testing)
//! let config = DeviceConfig::new().with_force_cpu(true);
//! let device = get_device(&config)?;
//! # Ok::<(), rust_ai_core::CoreError>(())
//! ```

use crate::error::Result;
use candle_core::Device;
use std::sync::Once;

/// Configuration for device selection.
///
/// # Fields
///
/// - `cuda_device`: Preferred CUDA device ordinal (default: 0)
/// - `force_cpu`: Force CPU execution regardless of GPU availability
/// - `crate_name`: Name of the crate for logging (used in warnings)
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// Preferred CUDA device ordinal.
    pub cuda_device: usize,
    /// Force CPU execution (disables GPU).
    pub force_cpu: bool,
    /// Crate name for logging (appears in warnings).
    pub crate_name: Option<String>,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            cuda_device: 0,
            force_cpu: false,
            crate_name: None,
        }
    }
}

impl DeviceConfig {
    /// Create a new device configuration with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the preferred CUDA device ordinal.
    #[must_use]
    pub fn with_cuda_device(mut self, ordinal: usize) -> Self {
        self.cuda_device = ordinal;
        self
    }

    /// Force CPU execution.
    #[must_use]
    pub fn with_force_cpu(mut self, force: bool) -> Self {
        self.force_cpu = force;
        self
    }

    /// Set crate name for logging.
    #[must_use]
    pub fn with_crate_name(mut self, name: impl Into<String>) -> Self {
        self.crate_name = Some(name.into());
        self
    }

    /// Build configuration from environment variables.
    ///
    /// Checks these environment variables (in order):
    /// 1. `RUST_AI_FORCE_CPU` / `RUST_AI_CUDA_DEVICE`
    /// 2. `AXOLOTL_FORCE_CPU` / `AXOLOTL_CUDA_DEVICE` (legacy)
    /// 3. `VSA_OPTIM_FORCE_CPU` / `VSA_OPTIM_CUDA_DEVICE` (legacy)
    #[must_use]
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Check force CPU flags
        let force_cpu_vars = [
            "RUST_AI_FORCE_CPU",
            "AXOLOTL_FORCE_CPU",
            "VSA_OPTIM_FORCE_CPU",
        ];
        for var in force_cpu_vars {
            if let Ok(val) = std::env::var(var) {
                if val == "1" || val.to_lowercase() == "true" {
                    config.force_cpu = true;
                    break;
                }
            }
        }

        // Check CUDA device selection
        let cuda_device_vars = [
            "RUST_AI_CUDA_DEVICE",
            "AXOLOTL_CUDA_DEVICE",
            "VSA_OPTIM_CUDA_DEVICE",
        ];
        for var in cuda_device_vars {
            if let Ok(val) = std::env::var(var) {
                if let Ok(ordinal) = val.parse::<usize>() {
                    config.cuda_device = ordinal;
                    break;
                }
            }
        }

        config
    }
}

/// Get a device according to configuration, preferring CUDA.
///
/// This function implements the CUDA-first philosophy:
/// 1. If `force_cpu` is set, returns CPU device with warning
/// 2. Otherwise, attempts to get CUDA device at specified ordinal
/// 3. Falls back to CPU with warning if CUDA unavailable
///
/// # Arguments
///
/// * `config` - Device configuration specifying preferences
///
/// # Returns
///
/// The selected Candle `Device`.
///
/// # Errors
///
/// Returns error only if device creation fails entirely (rare).
///
/// # Example
///
/// ```rust
/// use rust_ai_core::{get_device, DeviceConfig};
///
/// let device = get_device(&DeviceConfig::from_env())?;
/// println!("Using device: {:?}", device);
/// # Ok::<(), rust_ai_core::CoreError>(())
/// ```
pub fn get_device(config: &DeviceConfig) -> Result<Device> {
    let crate_name = config.crate_name.as_deref().unwrap_or("rust-ai");

    if config.force_cpu {
        tracing::warn!(
            "{}: CPU device forced via configuration. \
             CUDA is the intended default for optimal performance.",
            crate_name
        );
        return Ok(Device::Cpu);
    }

    // Try to get CUDA device
    match Device::cuda_if_available(config.cuda_device) {
        Ok(Device::Cuda(cuda)) => {
            tracing::info!(
                "{}: Using CUDA device {} for GPU-accelerated execution",
                crate_name,
                config.cuda_device
            );
            Ok(Device::Cuda(cuda))
        }
        Ok(Device::Cpu) | Err(_) => {
            // CUDA not available, fall back with warning
            warn_if_cpu_internal(&Device::Cpu, crate_name);
            Ok(Device::Cpu)
        }
        Ok(device) => Ok(device), // Metal or other
    }
}

/// Emit a one-time warning if running on CPU.
///
/// This function should be called when entering performance-critical code paths
/// to remind users that CUDA is preferred. The warning is emitted only once per
/// process to avoid log spam.
///
/// # Arguments
///
/// * `device` - The current device
/// * `crate_name` - Name of the crate for the warning message
///
/// # Example
///
/// ```rust
/// use rust_ai_core::warn_if_cpu;
/// use candle_core::Device;
///
/// fn expensive_operation(device: &Device) {
///     warn_if_cpu(device, "my-crate");
///     // ... perform operation
/// }
/// ```
pub fn warn_if_cpu(device: &Device, crate_name: &str) {
    warn_if_cpu_internal(device, crate_name);
}

/// Internal warning implementation with static once-flag.
fn warn_if_cpu_internal(device: &Device, crate_name: &str) {
    static WARN_ONCE: Once = Once::new();

    if matches!(device, Device::Cpu) {
        WARN_ONCE.call_once(|| {
            tracing::warn!(
                "{crate_name}: CPU device in use. CUDA is the intended default; \
                 CPU mode exists only as a compatibility fallback. \
                 For production workloads, ensure CUDA is available. \
                 Set RUST_AI_FORCE_CPU=1 to silence this warning."
            );
            eprintln!(
                "WARNING: {crate_name}: CPU device in use. CUDA is the intended default; \
                 CPU mode exists only as a compatibility fallback."
            );
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_config_default() {
        let config = DeviceConfig::default();
        assert_eq!(config.cuda_device, 0);
        assert!(!config.force_cpu);
        assert!(config.crate_name.is_none());
    }

    #[test]
    fn test_device_config_builder() {
        let config = DeviceConfig::new()
            .with_cuda_device(1)
            .with_force_cpu(true)
            .with_crate_name("test-crate");

        assert_eq!(config.cuda_device, 1);
        assert!(config.force_cpu);
        assert_eq!(config.crate_name.as_deref(), Some("test-crate"));
    }

    #[test]
    fn test_force_cpu_returns_cpu() {
        let config = DeviceConfig::new().with_force_cpu(true);
        let device = get_device(&config).unwrap();
        assert!(matches!(device, Device::Cpu));
    }
}
