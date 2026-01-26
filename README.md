# rust-ai-core

[![Crates.io](https://img.shields.io/crates/v/rust-ai-core.svg)](https://crates.io/crates/rust-ai-core) [![PyPI](https://img.shields.io/pypi/v/rust-ai-core-bindings.svg)](https://pypi.org/project/rust-ai-core-bindings/) [![Documentation](https://docs.rs/rust-ai-core/badge.svg)](https://docs.rs/rust-ai-core) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-MIT)

**Foundation layer for the rust-ai ecosystem**, providing unified abstractions for device selection, error handling, configuration validation, and CubeCL interop.

rust-ai-core is the shared foundation that enables a future AI framework built on transparency, traceability, performance, ease of use, repeatability, and customization depth.

## Design Philosophy

**CUDA-first**: All operations prefer GPU execution. CPU is a fallback that emits warnings, not a silent alternative. This ensures users are aware when they're not getting optimal performance.

**Ecosystem Integration**: rust-ai-core serves as the foundation for all rust-ai crates, ensuring consistent behavior, unified error handling, and seamless interoperability across the entire stack.

## Features

- **Unified Device Selection**: CUDA-first with environment variable overrides
- **Common Error Types**: `CoreError` hierarchy shared across all crates
- **Trait Interfaces**: `ValidatableConfig`, `Quantize`, `Dequantize`, `GpuDispatchable`
- **CubeCL Interop**: Candle ↔ CubeCL tensor conversion utilities

## Installation

### Rust

```toml
[dependencies]
rust-ai-core = "0.2"

# With CUDA support
rust-ai-core = { version = "0.2", features = ["cuda"] }
```

### Python

```bash
pip install rust-ai-core-bindings
```

The Python package provides bindings for memory estimation, device detection, and dtype utilities.

## Quick Start

### Rust

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

### Python

```python
import rust_ai_core_bindings as rac

# Memory estimation for AI training planning
batch, heads, seq_len, head_dim = 1, 32, 4096, 128
attention_bytes = rac.estimate_attention_memory(batch, heads, seq_len, head_dim, "bf16")
print(f"Attention layer: {attention_bytes / 1024**2:.1f} MB")

# Tensor memory estimation
shape = [1, 512, 4096]
tensor_bytes = rac.estimate_tensor_bytes(shape, "f32")
print(f"Tensor: {tensor_bytes / 1024**2:.1f} MB")

# Memory tracking for GPU budget management
tracker = rac.create_memory_tracker(limit_bytes=8 * 1024**3)  # 8 GB limit
rac.tracker_allocate(tracker, tensor_bytes)
print(f"Current: {rac.tracker_allocated_bytes(tracker)} bytes")
print(f"Peak: {rac.tracker_peak_bytes(tracker)} bytes")

# Device detection
if rac.cuda_available():
    device_info = rac.get_device_info()
    print(f"Device: {device_info['name']}")

# Data type utilities
print(f"f32 size: {rac.bytes_per_dtype('f32')} bytes")
print(f"bf16 is float: {rac.is_floating_point_dtype('bf16')}")
print(f"bf16 accumulator: {rac.accumulator_dtype('bf16')}")
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

All rust-ai crates depend on rust-ai-core as their foundation:

```
rust-ai-core (Foundation Layer)
    │
    ├── trit-vsa          - Ternary Vector Symbolic Architectures
    ├── bitnet-quantize   - 1.58-bit quantization
    ├── peft-rs           - LoRA, DoRA, AdaLoRA adapters
    ├── qlora-rs          - 4-bit quantization + QLoRA
    ├── unsloth-rs        - GPU-optimized transformer kernels
    ├── vsa-optim-rs      - VSA optimizers and operations
    ├── axolotl-rs        - High-level fine-tuning orchestration
    └── tritter-accel     - Ternary GPU acceleration
```

Each crate uses rust-ai-core's:
- **Device selection**: Consistent CUDA-first device logic
- **Error types**: Shared `CoreError` hierarchy with domain-specific extensions
- **Traits**: Common interfaces (`ValidatableConfig`, `Quantize`, etc.)
- **CubeCL interop**: Unified Candle ↔ CubeCL tensor conversion

## Public API Reference

### Core Types

- **`DeviceConfig`** - Configuration builder for device selection
- **`CoreError`** - Unified error type with domain-specific variants
- **`Result<T>`** - Type alias for `std::result::Result<T, CoreError>`
- **`TensorBuffer`** - Intermediate representation for Candle ↔ CubeCL conversion

### Traits

- **`ValidatableConfig`** - Configuration validation interface
  ```rust
  trait ValidatableConfig: Clone + Send + Sync {
      fn validate(&self) -> Result<()>;
  }
  ```

- **`Quantize<Q>`** - Tensor quantization (full precision → quantized)
  ```rust
  trait Quantize<Q>: Send + Sync {
      fn quantize(&self, tensor: &Tensor, device: &Device) -> Result<Q>;
  }
  ```

- **`Dequantize<Q>`** - Tensor dequantization (quantized → full precision)
  ```rust
  trait Dequantize<Q>: Send + Sync {
      fn dequantize(&self, quantized: &Q, device: &Device) -> Result<Tensor>;
  }
  ```

- **`GpuDispatchable`** - GPU/CPU dispatch pattern for operations with both implementations
  ```rust
  trait GpuDispatchable: Send + Sync {
      type Input;
      type Output;

      fn dispatch_gpu(&self, input: &Self::Input, device: &Device) -> Result<Self::Output>;
      fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> Result<Self::Output>;
      fn dispatch(&self, input: &Self::Input, device: &Device) -> Result<Self::Output>;
      fn gpu_available(&self) -> bool;
  }
  ```

### Device Selection Functions

- **`get_device(config: &DeviceConfig) -> Result<Device>`** - Get device with CUDA-first fallback
- **`warn_if_cpu(device: &Device, crate_name: &str)`** - Emit one-time CPU warning

### CubeCL Interop (feature: `cuda`)

- **`has_cubecl_cuda_support() -> bool`** - Check if CubeCL CUDA runtime is available
- **`candle_to_cubecl_handle(tensor: &Tensor) -> Result<TensorBuffer>`** - Convert Candle tensor to CubeCL buffer
- **`cubecl_to_candle_tensor(buffer: &TensorBuffer, device: &Device) -> Result<Tensor>`** - Convert CubeCL buffer to Candle tensor
- **`allocate_output_buffer(shape: &[usize], dtype: DType) -> Result<TensorBuffer>`** - Pre-allocate CubeCL output buffer

## Error Handling Philosophy

rust-ai-core provides a structured error hierarchy that balances specificity with ergonomics:

```rust
use rust_ai_core::{CoreError, Result};

fn validate_shapes(a: &[usize], b: &[usize]) -> Result<()> {
    if a.len() != b.len() {
        return Err(CoreError::dim_mismatch(
            format!("expected {} dims, got {}", a.len(), b.len())
        ));
    }
    if a != b {
        return Err(CoreError::shape_mismatch(a, b));
    }
    Ok(())
}
```

Crates should extend `CoreError` with domain-specific variants:

```rust
#[derive(Error, Debug)]
pub enum PeftError {
    #[error("adapter '{0}' not found")]
    AdapterNotFound(String),

    #[error(transparent)]
    Core(#[from] CoreError),
}
```

## Future Framework Goals

rust-ai-core is designed to enable a future AI framework with these principles:

1. **Transparency** - Clear, understandable operations at every level
2. **Traceability** - Track what happens at each step with detailed logging
3. **Performance** - GPU-accelerated, optimized for production workloads
4. **Ease of use** - Simple high-level API with sensible defaults
5. **Repeatability** - Deterministic, reproducible results
6. **Customization depth** - Users can go as deep as they want, from high-level APIs to custom kernels

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions and extension points.

## Python API Reference

The `rust-ai-core-bindings` package exposes the following functions:

### Memory Estimation

```python
# Estimate tensor memory
estimate_tensor_bytes(shape: list[int], dtype: str) -> int

# Estimate attention layer memory (Q, K, V + attention weights + output)
estimate_attention_memory(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: str
) -> int
```

### Memory Tracking

```python
# Create a memory tracker with optional limit
create_memory_tracker(limit_bytes: int = 0) -> MemoryTracker

# Record allocation (raises if exceeds limit)
tracker_allocate(tracker: MemoryTracker, bytes: int)

# Record deallocation
tracker_deallocate(tracker: MemoryTracker, bytes: int)

# Query tracker state
tracker_allocated_bytes(tracker: MemoryTracker) -> int
tracker_peak_bytes(tracker: MemoryTracker) -> int
tracker_would_fit(tracker: MemoryTracker, bytes: int) -> bool

# Reset tracker to initial state
tracker_reset(tracker: MemoryTracker)
```

### Device Detection

```python
# Check CUDA availability
cuda_available() -> bool

# Get device information (returns dict with type, ordinal, name)
get_device_info(force_cpu: bool = False, cuda_device: int = 0) -> dict
```

### Data Type Utilities

```python
# Get bytes per element for dtype
bytes_per_dtype(dtype: str) -> int

# Check if dtype is floating point
is_floating_point_dtype(dtype: str) -> bool

# Get accumulator dtype for mixed precision
accumulator_dtype(dtype: str) -> str
```

### Supported Data Types

- `"f32"` - 32-bit float
- `"f16"` - 16-bit float
- `"bf16"` - Brain float 16
- `"f64"` - 64-bit float
- `"i64"` - 64-bit integer
- `"u32"` - 32-bit unsigned integer
- `"u8"` - 8-bit unsigned integer

### Logging

```python
# Initialize logging (call once at startup)
init_logging(level: str = "info")  # debug, info, warn, error
```

## License

MIT License - see [LICENSE-MIT](LICENSE-MIT)

## Contributing

Contributions welcome! Please ensure:
- All public items have documentation
- Tests pass: `cargo test`
- Lints pass: `cargo clippy --all-targets --all-features`
- Code is formatted: `cargo fmt`
