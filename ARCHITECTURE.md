# rust-ai-core Architecture

This document describes the design decisions, architectural patterns, and extension points in rust-ai-core, the foundation layer for the rust-ai ecosystem.

## Table of Contents

- [Overview](#overview)
- [Design Principles](#design-principles)
- [Module Architecture](#module-architecture)
- [CUDA-First Device Selection](#cuda-first-device-selection)
- [Error Handling Philosophy](#error-handling-philosophy)
- [Trait Design](#trait-design)
- [CubeCL Interop Layer](#cubecl-interop-layer)
- [Extension Points](#extension-points)
- [Performance Considerations](#performance-considerations)
- [Future Framework Integration](#future-framework-integration)

## Overview

rust-ai-core provides the shared foundation for the entire rust-ai ecosystem. It establishes:

- **Unified device selection** with CUDA-first philosophy
- **Common error types** that all crates extend
- **Trait interfaces** for configuration, quantization, and GPU dispatch
- **CubeCL interop** for Candle ↔ CubeCL tensor conversion

All rust-ai crates depend on this foundation, ensuring consistent behavior across:
- peft-rs (PEFT adapters)
- qlora-rs (4-bit quantization)
- unsloth-rs (GPU kernels)
- axolotl-rs (fine-tuning orchestration)
- trit-vsa, bitnet-quantize, vsa-optim-rs, tritter-accel (specialized components)

## Design Principles

### 1. CUDA-First Philosophy

**Rationale**: Production AI workloads require GPU acceleration. CPU execution is a compatibility fallback, not a first-class citizen.

**Implementation**:
- Default behavior: attempt CUDA, fall back to CPU with warning
- Explicit opt-in for CPU via environment variable or config
- Warning emitted exactly once per process to avoid log spam
- Clear messaging to users about performance implications

**Why this matters**: Silent CPU fallbacks hide performance issues. Explicit warnings ensure users know when they're not getting optimal performance.

### 2. Unified Error Handling

**Rationale**: Consistent error types across all crates improve ergonomics and reduce boilerplate.

**Implementation**:
- `CoreError` provides common variants (shape mismatch, device errors, etc.)
- Crates extend via domain-specific error types that wrap `CoreError`
- Helper constructors (`invalid_config()`, `shape_mismatch()`, etc.) for ergonomics
- All errors implement `thiserror::Error` for good error messages

**Why this matters**: Users work across multiple crates. Consistent error handling reduces cognitive load.

### 3. Trait-Based Extensibility

**Rationale**: Common interfaces enable interoperability and composition.

**Implementation**:
- `ValidatableConfig` for configuration validation
- `Quantize` / `Dequantize` for quantization schemes
- `GpuDispatchable` for operations with GPU/CPU implementations
- All traits are `Send + Sync` for thread safety

**Why this matters**: Crates can be composed together seamlessly. New quantization schemes integrate easily.

### 4. Zero-Cost Abstractions

**Rationale**: Performance is critical for AI/ML workloads.

**Implementation**:
- Trait methods inline where possible
- No runtime overhead for device selection
- Direct pointer access for tensor conversions
- Feature gates (`cuda`) compile out unused code

**Why this matters**: Framework overhead must be negligible compared to computation costs.

## Module Architecture

```
rust-ai-core/
├── src/
│   ├── lib.rs          # Public API surface, re-exports
│   ├── device.rs       # CUDA-first device selection
│   ├── dtype.rs        # Data type utilities and precision helpers
│   ├── error.rs        # Unified error types
│   ├── logging.rs      # Unified logging and observability
│   ├── memory.rs       # Memory estimation and tracking
│   ├── traits.rs       # Common trait interfaces
│   └── cubecl/
│       ├── mod.rs      # CubeCL module exports
│       └── interop.rs  # Candle ↔ CubeCL conversion
├── tests/              # Integration tests
├── benches/            # Criterion benchmarks
└── examples/           # Usage examples
```

### Module Responsibilities

#### `device.rs`
- **Purpose**: Centralized device selection logic
- **Exports**: `DeviceConfig`, `get_device()`, `warn_if_cpu()`
- **Key features**:
  - Builder pattern for configuration
  - Environment variable support (with legacy compatibility)
  - One-time warning mechanism
  - Tracing integration for observability

#### `error.rs`
- **Purpose**: Common error types shared across all crates
- **Exports**: `CoreError`, `Result<T>`
- **Key features**:
  - Structured variants for common failures
  - Helper constructors for ergonomics
  - Conversion from `std::io::Error` and `candle_core::Error`
  - Non-exhaustive enum for forward compatibility

#### `traits.rs`
- **Purpose**: Common trait interfaces for interoperability
- **Exports**: `ValidatableConfig`, `Quantize`, `Dequantize`, `GpuDispatchable`
- **Key features**:
  - Generic over quantized types (`Quantize<Q>`)
  - Default implementation for `GpuDispatchable::dispatch()`
  - Runtime GPU availability checking

#### `dtype.rs`
- **Purpose**: Data type utilities and precision helpers
- **Exports**: `bytes_per_element()`, `is_floating_point()`, `DTypeExt`
- **Key features**:
  - Extension trait for `candle_core::DType`
  - Mixed-precision support (accumulator dtype selection)
  - Training dtype validation

#### `logging.rs`
- **Purpose**: Unified logging and observability
- **Exports**: `LogConfig`, `init_logging()`, `log_training_step()`, `log_memory_usage()`, tracing macros
- **Key features**:
  - Builder pattern for log configuration
  - Preset configurations (development, production, testing)
  - Training-specific logging helpers

#### `memory.rs`
- **Purpose**: Memory estimation and tracking utilities
- **Exports**: `MemoryTracker`, `estimate_tensor_bytes()`, `estimate_attention_memory()`
- **Key features**:
  - Thread-safe memory tracking with atomics
  - Attention memory estimation (O(n²) scaling)
  - Configurable overhead factors

#### `cubecl/interop.rs`
- **Purpose**: Candle ↔ CubeCL tensor conversion
- **Exports**: `TensorBuffer`, `candle_to_cubecl_handle()`, `cubecl_to_candle_tensor()`
- **Key features**:
  - Zero-copy where possible (contiguous tensors)
  - Support for f32, f16, bf16 dtypes
  - Memory layout validation
  - Pre-allocation helpers

## CUDA-First Device Selection

### Strategy

```rust
pub fn get_device(config: &DeviceConfig) -> Result<Device> {
    if config.force_cpu {
        warn_and_return_cpu();
    }

    match Device::cuda_if_available(config.cuda_device) {
        Ok(Device::Cuda(cuda)) => Ok(Device::Cuda(cuda)),
        Ok(Device::Cpu) | Err(_) => {
            warn_and_return_cpu();
        }
        Ok(device) => Ok(device), // Metal, etc.
    }
}
```

### Decision Tree

```
Start
  │
  ├─ force_cpu set? ──Yes──> CPU + warning
  │         │
  │         No
  │         │
  ├─ CUDA available? ──Yes──> CUDA device
  │         │
  │         No
  │         │
  └────> CPU + warning
```

### Warning Strategy

Warnings are emitted **exactly once per process** using `std::sync::Once`:

```rust
static WARN_ONCE: Once = Once::new();

WARN_ONCE.call_once(|| {
    tracing::warn!("CPU device in use...");
    eprintln!("WARNING: CPU device in use...");
});
```

**Rationale**:
- Avoids log spam in hot loops
- Dual output (tracing + stderr) ensures visibility
- Clear actionable message with silencing option

### Environment Variables

Priority order (first match wins):

1. `RUST_AI_FORCE_CPU` / `RUST_AI_CUDA_DEVICE` (current)
2. `AXOLOTL_FORCE_CPU` / `AXOLOTL_CUDA_DEVICE` (legacy)
3. `VSA_OPTIM_FORCE_CPU` / `VSA_OPTIM_CUDA_DEVICE` (legacy)

**Rationale**: Backwards compatibility while migrating to unified variables.

## Error Handling Philosophy

### Error Hierarchy

```
CoreError (shared foundation)
    ├── InvalidConfig       - Validation failures
    ├── ShapeMismatch       - Tensor shape incompatibilities
    ├── DimensionMismatch   - Dimension count mismatches
    ├── DeviceNotAvailable  - Requested device unavailable
    ├── DeviceMismatch      - Tensors on different devices
    ├── OutOfMemory         - Memory allocation failures
    ├── KernelError         - GPU kernel failures
    ├── NotImplemented      - Unimplemented features
    ├── Io                  - I/O errors
    └── Candle              - Wrapped Candle errors

Domain-Specific Errors (per crate)
    └── wraps CoreError + adds crate-specific variants
```

### Design Rationale

**Non-exhaustive enum**: Future-proofs the API. New error variants can be added without breaking existing code.

**Structured variants with fields**: Programmatic error inspection. Callers can extract shape values, device names, etc.

**Helper constructors**: Ergonomics. Compare:

```rust
// Without helpers (verbose)
Err(CoreError::InvalidConfig("rank must be positive".to_string()))

// With helpers (concise)
Err(CoreError::invalid_config("rank must be positive"))
```

**Transparent Candle errors**: Preserves full error context from underlying tensor library.

### Crate Extension Pattern

Each crate defines its own error type that wraps `CoreError`:

```rust
// In peft-rs
#[derive(Error, Debug)]
pub enum PeftError {
    #[error("adapter '{0}' not found")]
    AdapterNotFound(String),

    #[error("rank {rank} exceeds max {max}")]
    RankTooLarge { rank: usize, max: usize },

    #[error(transparent)]
    Core(#[from] CoreError),
}
```

**Benefits**:
- Domain-specific variants for clarity
- Automatic conversion from `CoreError` via `#[from]`
- Consistent error handling across crates
- Type safety at crate boundaries

## Trait Design

### `ValidatableConfig`

**Purpose**: Standardized configuration validation.

**Design**:
```rust
pub trait ValidatableConfig: Clone + Send + Sync {
    fn validate(&self) -> Result<()>;
}
```

**Bounds**:
- `Clone`: Configs are typically small, copy-on-write friendly
- `Send + Sync`: Thread-safe for concurrent training

**Usage pattern**:
```rust
impl MyStruct {
    pub fn new(config: MyConfig) -> Result<Self> {
        config.validate()?;  // Validate in constructor
        Ok(Self { config })
    }
}
```

### `Quantize<Q>` / `Dequantize<Q>`

**Purpose**: Unified interface for quantization schemes.

**Design**:
```rust
pub trait Quantize<Q>: Send + Sync {
    fn quantize(&self, tensor: &Tensor, device: &Device) -> Result<Q>;
}

pub trait Dequantize<Q>: Send + Sync {
    fn dequantize(&self, quantized: &Q, device: &Device) -> Result<Tensor>;
}
```

**Generic parameter `Q`**: Quantized type (e.g., `Nf4Tensor`, `TernaryVector`, `Int8Tensor`).

**Separate traits**: Not all quantizers support dequantization (e.g., one-way compression).

**Device parameter**: Allows in-place quantization or device transfer.

**Examples**:
- **NF4 quantization** (qlora-rs): `Q = Nf4Tensor`
- **Ternary quantization** (trit-vsa): `Q = TernaryVector`
- **Int8 quantization** (bitnet-quantize): `Q = Int8Tensor`

### `GpuDispatchable`

**Purpose**: Automatic GPU/CPU dispatch for operations with both implementations.

**Design**:
```rust
pub trait GpuDispatchable: Send + Sync {
    type Input;
    type Output;

    fn dispatch_gpu(&self, input: &Self::Input, device: &Device) -> Result<Self::Output>;
    fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> Result<Self::Output>;

    fn dispatch(&self, input: &Self::Input, device: &Device) -> Result<Self::Output> {
        match device {
            Device::Cuda(_) => self.dispatch_gpu(input, device),
            Device::Cpu => self.dispatch_cpu(input, device),
            _ => Err(CoreError::device_not_available(format!("{device:?}"))),
        }
    }

    fn gpu_available(&self) -> bool { /* ... */ }
}
```

**Associated types**: Flexible input/output types per operation.

**Default `dispatch()` implementation**: Implementors only write `dispatch_gpu` and `dispatch_cpu`.

**CUDA-first pattern**: GPU path first, CPU as fallback.

**Usage pattern**:
```rust
// In unsloth-rs
impl GpuDispatchable for FlashAttention {
    type Input = (Tensor, Tensor, Tensor);  // Q, K, V
    type Output = Tensor;

    fn dispatch_gpu(&self, (q, k, v): &Self::Input, device: &Device) -> Result<Tensor> {
        // Launch CubeCL flash attention kernel
    }

    fn dispatch_cpu(&self, (q, k, v): &Self::Input, device: &Device) -> Result<Tensor> {
        warn_if_cpu(device, "unsloth-rs");
        // Candle fallback implementation
    }
}

// Usage
let attn = FlashAttention::new();
let output = attn.dispatch(&(q, k, v), &device)?;  // Auto-routes
```

## CubeCL Interop Layer

### Problem Statement

Candle provides high-level tensor operations, CubeCL provides low-level GPU kernels. We need seamless conversion between them.

### Solution Architecture

```
Candle Tensor (Device::Cuda)
    │
    ├─> candle_to_cubecl_handle()
    │       │
    │       ├─ Ensure contiguity
    │       ├─ Extract raw bytes
    │       └─> TensorBuffer { bytes, shape, dtype }
    │               │
    │               └─> client.create(&bytes) → CubeCL Handle
    │
    ├─> CubeCL Kernel Execution
    │       │
    │       └─> CubeCL Output Handle
    │               │
    │               └─> client.read(&handle) → bytes
    │                       │
    │                       └─> TensorBuffer { bytes, shape, dtype }
    │
    └─> cubecl_to_candle_tensor()
            │
            └─> Candle Tensor (Device::Cuda)
```

### Memory Management

**TensorBuffer design**:
```rust
pub struct TensorBuffer {
    pub bytes: Vec<u8>,      // Raw little-endian bytes
    pub shape: Vec<usize>,   // Tensor dimensions
    pub dtype: DType,        // Element type
}
```

**Contiguity requirement**: Candle tensors must be contiguous for raw pointer access. Non-contiguous tensors are made contiguous (involves copy).

**Zero-copy path**: Contiguous tensors → direct memory access → no allocation.

**Supported dtypes**: f32, f16, bf16 (common for ML). Others return error.

### Validation

```rust
pub fn cubecl_to_candle_tensor(buffer: &TensorBuffer, device: &Device) -> Result<Tensor> {
    // 1. Device validation
    if !matches!(device, Device::Cuda(_)) {
        return Err(CoreError::invalid_config("requires CUDA device"));
    }

    // 2. Size validation
    let expected = buffer.numel() * buffer.dtype.size_in_bytes();
    if buffer.bytes.len() != expected {
        return Err(CoreError::shape_mismatch(...));
    }

    // 3. Reconstruct tensor
    match buffer.dtype {
        DType::F32 => { /* ... */ }
        DType::F16 => { /* ... */ }
        DType::BF16 => { /* ... */ }
        _ => Err(CoreError::invalid_config("unsupported dtype")),
    }
}
```

### Performance Considerations

**Pre-allocation**: Use `allocate_output_buffer()` to avoid allocations in hot loops.

```rust
// Outside loop
let mut output_buffer = allocate_output_buffer(&output_shape, DType::F32)?;

// In loop (reuse buffer)
for batch in batches {
    let result = launch_kernel(&input, &mut output_buffer)?;
    // ... use result
}
```

**Contiguity check cost**: Negligible compared to kernel launch overhead.

## Extension Points

### 1. New Quantization Schemes

To add a new quantization scheme (e.g., FP8):

```rust
// 1. Define quantized type
pub struct Fp8Tensor {
    data: Vec<u8>,
    scale: f32,
    shape: Vec<usize>,
}

// 2. Implement Quantize trait
pub struct Fp8Quantizer;

impl Quantize<Fp8Tensor> for Fp8Quantizer {
    fn quantize(&self, tensor: &Tensor, device: &Device) -> Result<Fp8Tensor> {
        // Quantization logic
    }
}

// 3. Implement Dequantize trait
impl Dequantize<Fp8Tensor> for Fp8Quantizer {
    fn dequantize(&self, quantized: &Fp8Tensor, device: &Device) -> Result<Tensor> {
        // Dequantization logic
    }
}
```

### 2. New Device Types

To support new device types (e.g., Metal):

```rust
pub fn get_device(config: &DeviceConfig) -> Result<Device> {
    // ... existing logic ...

    // Add Metal support
    #[cfg(feature = "metal")]
    if config.prefer_metal {
        if let Ok(device) = Device::metal_if_available(0) {
            return Ok(device);
        }
    }

    // ... fallback logic ...
}
```

### 3. New Error Variants

Add to `CoreError` (non-breaking due to `#[non_exhaustive]`):

```rust
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum CoreError {
    // ... existing variants ...

    #[error("quantization error: {0}")]
    QuantizationError(String),
}

impl CoreError {
    pub fn quantization(msg: impl Into<String>) -> Self {
        Self::QuantizationError(msg.into())
    }
}
```

### 4. Custom GPU Operations

Implement `GpuDispatchable` for new operations:

```rust
pub struct MyCustomOp {
    // ... state ...
}

impl GpuDispatchable for MyCustomOp {
    type Input = MyInput;
    type Output = MyOutput;

    fn dispatch_gpu(&self, input: &Self::Input, device: &Device) -> Result<Self::Output> {
        // CubeCL kernel path
    }

    fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> Result<Self::Output> {
        warn_if_cpu(device, "my-crate");
        // Candle fallback
    }
}
```

## Performance Considerations

### Device Selection Overhead

**Cost**: Negligible. Device selection happens once per operation, warning only once per process.

**Benchmark**: `get_device()` takes < 1μs on typical hardware.

### Error Handling Overhead

**Cost**: Zero in happy path (errors are exceptional).

**Design**: `Result<T>` is zero-cost when `Ok`, same size as `T` for common cases.

### Trait Dispatch Overhead

**Cost**: Zero due to monomorphization.

**Explanation**: Rust traits compile to static dispatch. No vtable lookups, no runtime overhead.

### CubeCL Conversion Overhead

**Cost**: Dominated by kernel launch latency (10-50μs).

**Optimization**: Pre-allocate buffers, ensure tensor contiguity before hot loops.

**Benchmark**: Conversion adds < 1% overhead for typical kernel operations.

## Future Framework Integration

rust-ai-core is designed to support a future AI framework with these goals:

### 1. Transparency

**Foundation support**:
- Tracing integration throughout (`tracing::info`, `tracing::warn`)
- Structured errors with detailed context
- Clear naming conventions

**Future additions**:
- Operation-level tracing hooks
- Computation graph visualization
- Step-by-step execution logging

### 2. Traceability

**Foundation support**:
- Device selection logged
- Error provenance tracked (source location, call stack)
- Configuration validation recorded

**Future additions**:
- Tensor provenance tracking (what operations created this tensor?)
- Execution timeline recording
- Reproducibility metadata (seeds, versions, configurations)

### 3. Performance

**Foundation support**:
- Zero-cost abstractions
- CUDA-first philosophy
- Efficient CubeCL interop

**Future additions**:
- Kernel fusion hints
- Memory pool management
- Multi-GPU coordination

### 4. Ease of Use

**Foundation support**:
- Builder patterns for configuration
- Helpful error messages
- Sensible defaults

**Future additions**:
- High-level API wrappers
- Auto-configuration (detect hardware, tune parameters)
- Progress bars and status updates

### 5. Repeatability

**Foundation support**:
- Deterministic device selection (given same config)
- No hidden state in core utilities
- Explicit configuration over magic

**Future additions**:
- Seed management
- Checkpoint/restore
- Experiment tracking integration

### 6. Customization Depth

**Foundation support**:
- Trait-based extension points
- Feature flags for optional components
- Public internal APIs where appropriate

**Future additions**:
- Custom kernel registration
- Operation graph manipulation
- Plugin system for new backends

## Summary

rust-ai-core provides a solid, performant, and extensible foundation for the rust-ai ecosystem. Its design enables:

- **Consistency**: Unified device selection, errors, and traits across all crates
- **Performance**: Zero-cost abstractions, GPU-first execution
- **Extensibility**: Trait-based interfaces for new quantization schemes, operations, and devices
- **Developer experience**: Clear errors, helpful warnings, comprehensive documentation
- **Future-proofing**: Extension points and non-exhaustive types for evolution

The architecture balances immediate practical needs with long-term framework goals, providing a stable foundation while enabling future innovation.
