# rust-ai-core - Claude Code Instructions

This crate is the **foundation layer** for the entire rust-ai ecosystem. All changes here impact multiple dependent crates. Handle with care.

## Crate Role

rust-ai-core provides:
1. **Unified device selection** - CUDA-first philosophy across all crates
2. **Common error types** - `CoreError` hierarchy shared by all rust-ai crates
3. **Trait interfaces** - `ValidatableConfig`, `Quantize`, `Dequantize`, `GpuDispatchable`
4. **CubeCL interop** - Candle ↔ CubeCL tensor conversion utilities

## Dependencies

**Downstream crates** (all depend on rust-ai-core):
- peft-rs
- qlora-rs
- unsloth-rs
- axolotl-rs
- trit-vsa
- bitnet-quantize
- vsa-optim-rs
- tritter-accel

**Critical**: Changes to public API affect all downstream crates. Always check compatibility.

## Code Style & Conventions

### Error Handling

**Always use helper constructors**:
```rust
// Good
return Err(CoreError::invalid_config("rank must be positive"));
return Err(CoreError::shape_mismatch(expected, actual));

// Bad (verbose)
return Err(CoreError::InvalidConfig("rank must be positive".to_string()));
```

**Extend CoreError for domain-specific errors**:
```rust
#[derive(Error, Debug)]
pub enum MyError {
    #[error("domain-specific error")]
    MyVariant,

    #[error(transparent)]
    Core(#[from] CoreError),
}
```

### Device Selection

**Always use CUDA-first pattern**:
```rust
// Good - warns when on CPU
let device = get_device(&DeviceConfig::from_env())?;

// In hot paths
warn_if_cpu(&device, "my-crate");

// Bad - doesn't warn users
let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
```

### Traits

**Implement common traits for interoperability**:
```rust
// Configuration structs
impl ValidatableConfig for MyConfig {
    fn validate(&self) -> Result<()> {
        // Validation logic
        Ok(())
    }
}

// Quantization
impl Quantize<MyQuantizedType> for MyQuantizer {
    fn quantize(&self, tensor: &Tensor, device: &Device) -> Result<MyQuantizedType> {
        // Quantization logic
    }
}

// GPU operations
impl GpuDispatchable for MyOp {
    type Input = MyInput;
    type Output = MyOutput;

    fn dispatch_gpu(&self, input: &Self::Input, device: &Device) -> Result<Self::Output> {
        // CubeCL kernel
    }

    fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> Result<Self::Output> {
        warn_if_cpu(device, "my-crate");
        // Fallback
    }
}
```

### Documentation

**All public items must have doc comments**:
```rust
/// Brief description (one line).
///
/// Longer description with details.
///
/// # Arguments
///
/// * `arg` - Description
///
/// # Returns
///
/// Description of return value.
///
/// # Errors
///
/// When this function returns an error.
///
/// # Example
///
/// ```rust
/// use rust_ai_core::function_name;
/// let result = function_name(42)?;
/// # Ok::<(), rust_ai_core::CoreError>(())
/// ```
pub fn function_name(arg: i32) -> Result<()> {
    // ...
}
```

## Testing Strategy

### Unit Tests

Place in same file as implementation:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CoreError::invalid_config("test");
        assert_eq!(err.to_string(), "invalid configuration: test");
    }
}
```

### Integration Tests

Place in `tests/` directory:
```rust
// tests/integration_test.rs
use rust_ai_core::{get_device, DeviceConfig};

#[test]
fn test_device_selection() {
    let config = DeviceConfig::new().with_force_cpu(true);
    let device = get_device(&config).unwrap();
    assert!(matches!(device, candle_core::Device::Cpu));
}
```

### GPU Tests

Mark with `#[ignore]` for manual execution:
```rust
#[test]
#[ignore]
fn test_cuda_device() {
    let device = get_device(&DeviceConfig::default()).unwrap();
    assert!(matches!(device, candle_core::Device::Cuda(_)));
}
```

Run with: `cargo test -- --ignored`

## Common Tasks

### Adding a New Error Variant

1. Add to `CoreError` enum in `src/error.rs`:
```rust
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum CoreError {
    // ... existing variants ...

    #[error("my new error: {0}")]
    MyNewError(String),
}
```

2. Add helper constructor:
```rust
impl CoreError {
    pub fn my_new_error(msg: impl Into<String>) -> Self {
        Self::MyNewError(msg.into())
    }
}
```

3. Add test:
```rust
#[test]
fn test_my_new_error() {
    let err = CoreError::my_new_error("test");
    assert!(err.to_string().contains("my new error"));
}
```

4. **Check downstream crates**: Ensure new variant doesn't conflict.

### Adding a New Trait

1. Define in `src/traits.rs`:
```rust
/// Brief description.
///
/// # Example
///
/// ```rust,ignore
/// impl MyNewTrait for MyType { /* ... */ }
/// ```
pub trait MyNewTrait: Send + Sync {
    fn my_method(&self) -> Result<()>;
}
```

2. Export in `src/lib.rs`:
```rust
pub use traits::{ValidatableConfig, Quantize, Dequantize, GpuDispatchable, MyNewTrait};
```

3. Add example implementation in trait docs.

4. **Check downstream crates**: Identify which crates should implement this trait.

### Extending CubeCL Interop

1. Add new dtype support in `src/cubecl/interop.rs`:
```rust
pub fn candle_to_cubecl_handle(tensor: &Tensor) -> Result<TensorBuffer> {
    // ... existing code ...

    let bytes = match dtype {
        DType::F32 => { /* ... */ }
        DType::F16 => { /* ... */ }
        DType::BF16 => { /* ... */ }
        DType::U8 => {  // New dtype
            let data: Vec<u8> = tensor.flatten_all()?.to_vec1()?;
            data  // Already bytes
        }
        _ => return Err(CoreError::invalid_config(/* ... */)),
    };
}
```

2. Update `cubecl_to_candle_tensor()` with matching logic.

3. Add tests for new dtype.

### Modifying DeviceConfig

**BREAKING CHANGE - coordinate with downstream crates!**

1. Update `DeviceConfig` struct in `src/device.rs`.
2. Update builder methods.
3. Update `from_env()` if adding new environment variables.
4. Update documentation in `src/lib.rs` and `README.md`.
5. **Notify maintainers of dependent crates** (peft-rs, qlora-rs, etc.).

## Breaking Change Protocol

rust-ai-core is foundational. Breaking changes require coordination:

1. **Identify affected crates**: Check which downstream crates use the API.
2. **Create compatibility layer** (if possible): Deprecate old API, add new API.
3. **Update dependent crates**: Fix compilation in peft-rs, qlora-rs, etc.
4. **Bump version**: Follow semantic versioning (0.x.y → 0.x+1.0 for breaking).
5. **Update CHANGELOG.md**: Document breaking changes clearly.
6. **Announce**: Notify users via release notes.

## Feature Flags

- `default = []` - Core functionality only
- `cuda = ["candle-core/cuda", "dep:cubecl", "dep:cubecl-cuda"]` - Enable GPU support

**When adding features**:
- Gate optional dependencies with `dep:` prefix
- Update `Cargo.toml` and documentation
- Ensure tests pass with `--all-features` and `--no-default-features`

## Build & Test Commands

```bash
# Full check (all features)
cargo check --all-features

# Test without GPU
cargo test

# Test with GPU (requires CUDA)
cargo test --features cuda

# GPU tests only (manual, requires hardware)
cargo test --features cuda -- --ignored

# Clippy (zero warnings required)
cargo clippy --all-targets --all-features

# Documentation
cargo doc --no-deps --open

# Format
cargo fmt
```

## Pre-Commit Checklist

Before committing changes to rust-ai-core:

- [ ] All public items have documentation
- [ ] Tests added for new functionality
- [ ] `cargo test` passes
- [ ] `cargo clippy --all-features` has zero warnings
- [ ] `cargo fmt` applied
- [ ] Dependent crates checked (if API changed)
- [ ] CHANGELOG.md updated (if user-facing change)
- [ ] Documentation builds: `cargo doc --no-deps`

## When Working on Dependent Crates

If modifying code in peft-rs, qlora-rs, unsloth-rs, axolotl-rs, etc., and you need to change rust-ai-core:

1. Make changes to rust-ai-core first
2. Test in isolation: `cargo test -p rust-ai-core`
3. Update dependent crate to use new API
4. Test dependent crate: `cargo test -p <crate-name>`
5. Test workspace: `cargo test --workspace`
6. Commit both changes together (atomic)

## Performance Guidelines

### Critical Paths

- `get_device()` - Called frequently, must be fast
- `candle_to_cubecl_handle()` / `cubecl_to_candle_tensor()` - In kernel launch path
- Trait methods - Must inline for zero-cost abstraction

### Optimization Techniques

**Inline hot paths**:
```rust
#[inline]
pub fn get_device(config: &DeviceConfig) -> Result<Device> {
    // ...
}
```

**Avoid allocations in loops**:
```rust
// Bad
for _ in 0..n {
    let buffer = allocate_output_buffer(&shape, dtype)?;
}

// Good
let mut buffer = allocate_output_buffer(&shape, dtype)?;
for _ in 0..n {
    // Reuse buffer
}
```

**Benchmark critical paths**:
```rust
// benches/benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_get_device(c: &mut Criterion) {
    c.bench_function("get_device", |b| {
        let config = DeviceConfig::default();
        b.iter(|| get_device(black_box(&config)));
    });
}

criterion_group!(benches, bench_get_device);
criterion_main!(benches);
```

## Debugging Tips

### Tracing

All modules use `tracing` for logging:
```rust
use tracing::{info, warn, error, debug, trace};

info!("CUDA device {ordinal} selected");
warn!("CPU fallback in use");
debug!("tensor shape: {:?}", shape);
```

Enable with `RUST_LOG=rust_ai_core=debug`.

### Error Context

Add context to errors:
```rust
// Good
tensor.contiguous()
    .map_err(|e| CoreError::kernel(format!("failed to make tensor contiguous: {e}")))?;

// Bad (loses context)
tensor.contiguous()?;
```

### Device Debugging

Check device at runtime:
```rust
tracing::debug!("device type: {:?}", device);
match device {
    Device::Cuda(cuda) => tracing::debug!("CUDA device: {cuda:?}"),
    Device::Cpu => tracing::warn!("CPU device in use"),
    _ => tracing::info!("other device: {device:?}"),
}
```

## Anti-Patterns to Avoid

### Silent CPU Fallback
```rust
// BAD - silently falls back to CPU
let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

// GOOD - warns user about CPU usage
let device = get_device(&DeviceConfig::from_env())?;
```

### Magic Numbers
```rust
// BAD
let device = get_device(&DeviceConfig::new().with_cuda_device(0))?;

// GOOD - use from_env() to respect user configuration
let device = get_device(&DeviceConfig::from_env())?;
```

### Ignoring Validation
```rust
// BAD - no validation
pub fn new(config: MyConfig) -> Self {
    Self { config }
}

// GOOD - validate in constructor
pub fn new(config: MyConfig) -> Result<Self> {
    config.validate()?;
    Ok(Self { config })
}
```

### Non-Contiguous Tensor Assumptions
```rust
// BAD - assumes contiguous
let ptr = tensor.data_ptr();

// GOOD - ensure contiguous
let tensor = tensor.contiguous()?;
let ptr = tensor.data_ptr();
```

## Resources

- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions
- **API Docs**: Run `cargo doc --open`
- **Examples**: See dependent crates (peft-rs, qlora-rs) for usage examples
- **Issues**: Check GitHub issues for known problems

## Contact

For questions about rust-ai-core architecture or breaking changes, consult:
- ARCHITECTURE.md (this repo)
- Workspace CLAUDE.md at `/home/kang/Documents/projects/rust-ai/CLAUDE.md`
- Dependent crate maintainers (if cross-crate coordination needed)
