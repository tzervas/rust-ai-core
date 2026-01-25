# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-01-25

### Added

#### Python Bindings (`python` feature)
- **`rust-ai-core-bindings`** PyPI package for Python developers
- Memory estimation functions exposed to Python
- Memory tracker with full API access
- Device utilities (CUDA availability, device info)
- DType utilities (bytes_per_dtype, is_floating_point, accumulator_dtype)
- Logging initialization
- Type stubs (`.pyi`) for IDE support

#### New Modules
- **`memory` module** - Memory estimation and tracking utilities
  - `estimate_tensor_bytes()` - Calculate memory for tensor shapes
  - `estimate_attention_memory()` - O(n²) attention memory estimation
  - `MemoryTracker` - Thread-safe allocation tracking with limits
  - `DEFAULT_OVERHEAD_FACTOR` - Conservative 1.1x buffer for CUDA overhead

- **`dtype` module** - Data type utilities and precision helpers
  - `bytes_per_element()` - Get size in bytes for any DType
  - `is_floating_point()` - Check if DType is floating-point
  - `DTypeExt` trait - Extension methods for `candle_core::DType`
  - `PrecisionMode` enum - Mixed-precision training configurations

- **`logging` module** - Unified logging and observability
  - `LogConfig` - Configuration builder with presets (development, production, testing)
  - `init_logging()` - Single-call logging initialization
  - `log_training_step()` - Consistent training step logging
  - `log_memory_usage()` - Memory usage logging helper
  - Re-exports of tracing macros (`info!`, `warn!`, `debug!`, etc.)

#### Infrastructure
- **GitHub Actions workflows**
  - `ci.yml` - Continuous integration (test, lint, docs, Python tests)
  - `release.yml` - Automated releases to crates.io, PyPI, and GitHub
- **Integration tests** (`tests/integration.rs`) - Comprehensive public API tests
- **Examples directory** with 6 runnable examples:
  - `device_selection.rs` - Device selection patterns
  - `memory_tracking.rs` - Memory estimation and tracking
  - `error_handling.rs` - Error handling patterns
  - `dtype_utilities.rs` - DType utilities demonstration
  - `logging_setup.rs` - Logging configuration patterns
  - `traits_demo.rs` - Trait implementation examples
- **Benchmark suite** (`benches/core_ops.rs`) - Performance benchmarks
- **CHANGELOG.md** - Version history documentation
- **pyproject.toml** - Python package configuration for maturin

### Changed
- Enhanced documentation with Google-style docstrings
- Added "why" explanations to all module and function documentation
- Updated `lib.rs` with comprehensive crate-level documentation
- All public APIs now have examples in doc comments

### Technical Details
- Added `tracing-subscriber` dependency for logging infrastructure
- Added `tokio` dev-dependency for async test support
- Handles all Candle 0.9.2 DType variants (including exotic MX formats)

## [0.1.0] - 2026-01-24

### Added

#### Core Modules
- **`device` module** - CUDA-first device selection
  - `DeviceConfig` struct with builder pattern
  - `get_device()` - Device selection with automatic fallback
  - `warn_if_cpu()` - One-time CPU usage warning
  - Environment variable support (`RUST_AI_FORCE_CPU`, `RUST_AI_CUDA_DEVICE`)
  - Legacy env var compatibility (AXOLOTL_*, VSA_OPTIM_*)

- **`error` module** - Unified error types
  - `CoreError` enum with 9 variants (non-exhaustive for future compatibility)
  - Helper constructors for ergonomic error creation
  - `Result<T>` type alias
  - `From` implementations for `std::io::Error` and `candle_core::Error`

- **`traits` module** - Common trait interfaces
  - `ValidatableConfig` - Configuration validation
  - `Quantize<Q>` - Tensor quantization
  - `Dequantize<Q>` - Tensor dequantization
  - `GpuDispatchable` - GPU/CPU dispatch pattern

- **`cubecl` module** (feature-gated with `cuda`)
  - `TensorBuffer` - Intermediate buffer representation
  - `candle_to_cubecl_handle()` - Candle → CubeCL conversion
  - `cubecl_to_candle_tensor()` - CubeCL → Candle conversion
  - `has_cubecl_cuda_support()` - Runtime CUDA check
  - `allocate_output_buffer()` - Pre-allocation helper

#### Documentation
- README.md with quick start and API reference
- CLAUDE.md for Claude Code development workflow
- ARCHITECTURE.md explaining design decisions

### Technical Details
- Minimum Rust version: 1.92
- Candle version: 0.9.2
- CubeCL version: 0.9 (optional, cuda feature)
- Zero unsafe code
- Thread-safe (all traits are Send + Sync)

### Design Philosophy
- **CUDA-first**: GPU preferred, CPU is fallback with warnings
- **Zero-cost abstractions**: Traits compile to static dispatch
- **Fail-fast validation**: Configuration errors caught at construction

[Unreleased]: https://github.com/tzervas/rust-ai-core/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/tzervas/rust-ai-core/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tzervas/rust-ai-core/releases/tag/v0.1.0
