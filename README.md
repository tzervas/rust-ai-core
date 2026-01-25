# rust-ai-core

[![Crates.io](https://img.shields.io/crates/v/rust-ai-core.svg)](https://crates.io/crates/rust-ai-core) [![Documentation](https://docs.rs/rust-ai-core/badge.svg)](https://docs.rs/rust-ai-core) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-MIT)

Shared core utilities for the rust-ai ecosystem, providing unified abstractions for device selection, error handling, configuration validation, and CubeCL interop.

## Design Philosophy

**CUDA-first**: All operations prefer GPU execution. CPU is a fallback that emits warnings, not a silent alternative. This ensures users are aware when they're not getting optimal performance.

## Features

- **Unified Device Selection**: CUDA-first with environment variable overrides
- **Common Error Types**: `CoreError` hierarchy shared across all crates
- **Trait Interfaces**: `ValidatableConfig`, `Quantize`, `Dequantize`, `GpuDispatchable`
- **CubeCL Interop**: Candle ↔ CubeCL tensor conversion utilities

## Installation

```toml
[dependencies]
rust-ai-core = "0.1"

# With CUDA support
rust-ai-core = { version = "0.1", features = ["cuda"] }
```

## Quick Start

```rust
use rust_ai_core::{get_device, DeviceConfig, CoreError, Result};

fn main() -> Result<()> {
    // Get CUDA device with automatic fallback + warning
    let device = get_device(&DeviceConfig::default())?;
    
    // Or with environment-based configuration
    let config = DeviceConfig::from_env();
    let device = get_device(&config)?;
    
    Ok(())
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUST_AI_FORCE_CPU` | Set to `1` or `true` to force CPU execution |
| `RUST_AI_CUDA_DEVICE` | CUDA device ordinal (default: 0) |

Legacy variables from individual crates are also supported:
- `AXOLOTL_FORCE_CPU`, `AXOLOTL_CUDA_DEVICE`
- `VSA_OPTIM_FORCE_CPU`, `VSA_OPTIM_CUDA_DEVICE`

## Modules

### `device`

CUDA-first device selection with fallback warnings.

```rust
use rust_ai_core::{get_device, DeviceConfig, warn_if_cpu};

// Explicit configuration
let config = DeviceConfig::new()
    .with_cuda_device(0)
    .with_force_cpu(false)
    .with_crate_name("my-crate");

let device = get_device(&config)?;

// In hot paths, warn if on CPU
warn_if_cpu(&device, "my-crate");
```

### `error`

Common error types shared across the ecosystem.

```rust
use rust_ai_core::{CoreError, Result};

fn my_function() -> Result<()> {
    // Use convenient constructors
    if rank == 0 {
        return Err(CoreError::invalid_config("rank must be positive"));
    }
    
    if shape_a != shape_b {
        return Err(CoreError::shape_mismatch(shape_a, shape_b));
    }
    
    Ok(())
}
```

### `traits`

Common trait interfaces for configuration and GPU dispatch.

```rust
use rust_ai_core::{ValidatableConfig, GpuDispatchable, CoreError, Result};

#[derive(Clone)]
struct MyConfig {
    rank: usize,
}

impl ValidatableConfig for MyConfig {
    fn validate(&self) -> Result<()> {
        if self.rank == 0 {
            return Err(CoreError::invalid_config("rank must be > 0"));
        }
        Ok(())
    }
}
```

### `cubecl` (feature: `cuda`)

CubeCL ↔ Candle tensor interop.

```rust
use rust_ai_core::{has_cubecl_cuda_support, candle_to_cubecl_handle, cubecl_to_candle_tensor};

if has_cubecl_cuda_support() {
    let buffer = candle_to_cubecl_handle(&tensor)?;
    // ... launch CubeCL kernel with buffer.bytes ...
    let output = cubecl_to_candle_tensor(&output_buffer, &device)?;
}
```

## Crate Integration

All rust-ai crates depend on rust-ai-core:

```
rust-ai-core ──┬── trit-vsa
               ├── bitnet-quantize
               ├── peft-rs
               ├── qlora-rs
               ├── unsloth-rs
               ├── vsa-optim-rs
               ├── axolotl-rs
               └── tritter-accel
```

## License

MIT License - see [LICENSE-MIT](LICENSE-MIT)

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.
