# rust-ai-core

[![Crates.io](https://img.shields.io/crates/v/rust-ai-core.svg)](https://crates.io/crates/rust-ai-core) [![PyPI](https://img.shields.io/pypi/v/rust-ai-core-bindings.svg)](https://pypi.org/project/rust-ai-core-bindings/) [![Documentation](https://docs.rs/rust-ai-core/badge.svg)](https://docs.rs/rust-ai-core) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-MIT)

**Unified AI engineering toolkit** that orchestrates the complete rust-ai ecosystem into a cohesive API for fine-tuning, quantization, and GPU-accelerated AI operations.

## What is rust-ai-core?

rust-ai-core is the central hub for 8 specialized AI/ML crates, providing:

- **Unified API**: Single entry point (`RustAI`) for all AI engineering tasks
- **Foundation Layer**: Device selection, error handling, memory tracking
- **Ecosystem Re-exports**: Direct access to all component crates

### Integrated Ecosystem

| Crate | Purpose | Version |
|-------|---------|---------|
| [peft-rs](https://crates.io/crates/peft-rs) | LoRA, DoRA, AdaLoRA adapters | 1.0.3 |
| [qlora-rs](https://crates.io/crates/qlora-rs) | 4-bit quantized LoRA | 1.0.5 |
| [unsloth-rs](https://crates.io/crates/unsloth-rs) | Optimized transformer blocks | 1.0 |
| [axolotl-rs](https://crates.io/crates/axolotl-rs) | YAML-driven fine-tuning | 1.1 |
| [bitnet-quantize](https://crates.io/crates/bitnet-quantize) | BitNet 1.58-bit quantization | 0.2 |
| [trit-vsa](https://crates.io/crates/trit-vsa) | Ternary VSA operations | 0.2 |
| [vsa-optim-rs](https://crates.io/crates/vsa-optim-rs) | VSA-based optimization | 0.1 |
| [tritter-accel](https://crates.io/crates/tritter-accel) | Ternary GPU acceleration | 0.1 |

## Installation

### Rust

```toml
[dependencies]
# Full ecosystem integration
rust-ai-core = "0.3"

# With CUDA support
rust-ai-core = { version = "0.3", features = ["cuda"] }

# With Python bindings
rust-ai-core = { version = "0.3", features = ["python"] }

# Everything
rust-ai-core = { version = "0.3", features = ["full"] }
```

### Python

```bash
pip install rust-ai-core-bindings
```

## Quick Start Guide

### Option 1: Unified API (Recommended)

The `RustAI` facade provides a simplified interface for common tasks:

```rust
use rust_ai_core::{RustAI, RustAIConfig, AdapterType, QuantizeMethod};

fn main() -> rust_ai_core::Result<()> {
    // Initialize with sensible defaults
    let ai = RustAI::new(RustAIConfig::default())?;

    println!("Device: {:?}", ai.device());
    println!("Ecosystem crates: {:?}", ai.ecosystem());

    // Configure fine-tuning with LoRA
    let finetune = ai.finetune()
        .model("meta-llama/Llama-2-7b")
        .adapter(AdapterType::Lora)
        .rank(64)
        .alpha(16.0)
        .build()?;

    // Configure quantization
    let quant = ai.quantize()
        .method(QuantizeMethod::Nf4)
        .bits(4)
        .group_size(64)
        .build();

    // Configure VSA operations
    let vsa = ai.vsa()
        .dimension(10000)
        .build();

    Ok(())
}
```

### Option 2: Direct Crate Access

Access individual crates through the `ecosystem` module:

```rust
use rust_ai_core::ecosystem::peft::{LoraConfig, LoraLinear};
use rust_ai_core::ecosystem::qlora::{QLoraConfig, QuantizedTensor};
use rust_ai_core::ecosystem::bitnet::{BitNetConfig, TernaryLinear};
use rust_ai_core::ecosystem::trit::{TritVector, PackedTritVec};

fn main() -> rust_ai_core::Result<()> {
    // Use peft-rs directly
    let lora_config = LoraConfig::new(64, 16.0);

    // Use trit-vsa directly
    let vec = TritVector::random(10000);

    Ok(())
}
```

### Option 3: Foundation Layer Only

Use just the core utilities without ecosystem crates:

```rust
use rust_ai_core::{get_device, DeviceConfig, CoreError, Result};
use rust_ai_core::{estimate_tensor_bytes, MemoryTracker};

fn main() -> Result<()> {
    // CUDA-first device selection
    let device = get_device(&DeviceConfig::default())?;

    // Memory estimation
    let shape = [1, 4096, 4096];
    let bytes = estimate_tensor_bytes(&shape, candle_core::DType::F32);
    println!("Tensor size: {} MB", bytes / 1024 / 1024);

    // Memory tracking
    let tracker = MemoryTracker::new(8 * 1024 * 1024 * 1024); // 8 GB
    tracker.try_allocate(bytes)?;

    Ok(())
}
```

## API Reference

### Unified API (`RustAI`)

```rust
// Initialize
let ai = RustAI::new(RustAIConfig::default())?;

// Workflows
ai.finetune()   // -> FinetuneBuilder (LoRA, DoRA, AdaLoRA)
ai.quantize()   // -> QuantizeBuilder (NF4, FP4, BitNet, INT8)
ai.vsa()        // -> VsaBuilder (Vector Symbolic Architectures)
ai.train()      // -> TrainBuilder (Axolotl-style YAML config)

// Info
ai.device()     // Active device (CPU or CUDA)
ai.ecosystem()  // Ecosystem crate versions
ai.is_cuda()    // Whether CUDA is active
ai.info()       // Full environment info
```

### Configuration Options

```rust
let config = RustAIConfig::new()
    .with_verbose(true)                    // Enable verbose logging
    .with_memory_limit(8 * 1024 * 1024 * 1024)  // 8 GB limit
    .with_cpu()                            // Force CPU execution
    .with_cuda_device(0);                  // Select CUDA device
```

### Ecosystem Modules

| Module | Crate | Key Types |
|--------|-------|-----------|
| `ecosystem::peft` | peft-rs | `LoraConfig`, `LoraLinear`, `DoraConfig` |
| `ecosystem::qlora` | qlora-rs | `QLoraConfig`, `QuantizedTensor`, `Nf4Quantizer` |
| `ecosystem::unsloth` | unsloth-rs | `FlashAttention`, `SwiGLU`, `RMSNorm` |
| `ecosystem::axolotl` | axolotl-rs | `AxolotlConfig`, `TrainingPipeline` |
| `ecosystem::bitnet` | bitnet-quantize | `BitNetConfig`, `TernaryLinear` |
| `ecosystem::trit` | trit-vsa | `TritVector`, `PackedTritVec`, `HdcEncoder` |
| `ecosystem::vsa_optim` | vsa-optim-rs | `VsaOptimizer`, `GradientPredictor` |
| `ecosystem::tritter` | tritter-accel | `TritterRuntime`, `TernaryMatmul` |

### Foundation Utilities

| Function | Purpose |
|----------|---------|
| `get_device()` | CUDA-first device selection with fallback |
| `warn_if_cpu()` | One-time warning when running on CPU |
| `estimate_tensor_bytes()` | Memory estimation for tensor shapes |
| `estimate_attention_memory()` | O(n^2) attention memory estimation |
| `MemoryTracker::new()` | Thread-safe memory tracking with limits |
| `init_logging()` | Initialize tracing with env filter |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_AI_FORCE_CPU` | Force CPU execution | `false` |
| `RUST_AI_CUDA_DEVICE` | CUDA device ordinal | `0` |
| `RUST_LOG` | Logging level | `info` |

## Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `cuda` | CUDA support via CubeCL | cubecl, cubecl-cuda |
| `python` | Python bindings via PyO3 | pyo3, numpy |
| `full` | All features enabled | cuda, python |

## Design Philosophy

- **CUDA-first**: GPU preferred, CPU fallback with warnings
- **Zero-cost abstractions**: Traits compile to static dispatch
- **Fail-fast validation**: Configuration errors caught at construction
- **Unified API**: Single entry point for all AI engineering tasks
- **Direct access**: Full crate APIs available when needed

## Examples

See the [examples/](examples/) directory:

- `device_selection.rs` - Device configuration patterns
- `memory_tracking.rs` - Memory estimation and tracking
- `error_handling.rs` - Error handling patterns

## Python Bindings

```python
import rust_ai_core_bindings as rac

# Memory estimation
bytes = rac.estimate_tensor_bytes([1, 4096, 4096], "f32")
print(f"Tensor: {bytes / 1024**2:.1f} MB")

# Attention memory (for model planning)
attn_bytes = rac.estimate_attention_memory(1, 32, 4096, 128, "bf16")
print(f"Attention: {attn_bytes / 1024**2:.1f} MB")

# Device detection
if rac.cuda_available():
    info = rac.get_device_info()
    print(f"CUDA device: {info['name']}")

# Memory tracking
tracker = rac.create_memory_tracker(8 * 1024**3)  # 8 GB limit
rac.tracker_allocate(tracker, bytes)
print(f"Allocated: {rac.tracker_allocated_bytes(tracker)} bytes")
```

## License

MIT License - see [LICENSE-MIT](LICENSE-MIT)

## Contributing

Contributions welcome! Please ensure:
- All public items have documentation
- Tests pass: `cargo test`
- Lints pass: `cargo clippy --all-targets --all-features`
- Code is formatted: `cargo fmt`

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Design decisions and extension points
- [docs.rs/rust-ai-core](https://docs.rs/rust-ai-core) - Full API documentation
