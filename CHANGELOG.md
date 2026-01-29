# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] - 2026-01-29

### Added

#### TypeScript/JavaScript Bindings
- **Node.js bindings via napi-rs** (`napi` feature)
  - Native performance for ML training orchestration in Node.js
  - Memory tracker, device selection, and dtype utilities
  - Stable ABI across Node.js versions via N-API v4
- **WebAssembly bindings via wasm-bindgen** (`wasm` feature)
  - Browser/WASM compilation support
  - Isomorphic support for TensorFlow.js/ONNX.js interop
  - Zero-copy array sharing via TypedArrays
- **`typescript` module** - Shared implementation for NAPI and WASM bindings
- **`full-bindings` feature flag** - Enables all binding types (CUDA, Python, NAPI, WASM)

### Dependencies

- napi = "2.16" (optional, `napi` feature)
- napi-derive = "2.16" (optional, `napi` feature)
- wasm-bindgen = "0.2" (optional, `wasm` feature)
- js-sys = "0.3" (optional, `wasm` feature)

## [0.3.1] - 2026-01-28

### Changed
- Updated PyO3 from 0.22 to 0.27 for Python 3.13/3.14 compatibility
- Updated numpy bindings from 0.22 to 0.27 (aligned with PyO3)
- Aligned with tritter-accel v0.1.3 for ecosystem version consistency

### Dependencies
- pyo3 = "0.27" (previously 0.22)
- numpy = "0.27" (previously 0.22)
- tritter-accel = "0.1" (resolves to 0.1.3)

## [0.3.0] - 2026-01-28

### Added

#### Ecosystem Integration
- **Full rust-ai ecosystem orchestration** - rust-ai-core now integrates all 8 ecosystem crates
- **`ecosystem` module** - Unified re-exports from all ecosystem crates:
  - `ecosystem::peft` - LoRA, DoRA, AdaLoRA adapters (peft-rs 1.0.3)
  - `ecosystem::qlora` - 4-bit quantized LoRA (qlora-rs 1.0.5)
  - `ecosystem::unsloth` - Optimized transformer blocks (unsloth-rs 1.0)
  - `ecosystem::axolotl` - YAML-driven fine-tuning (axolotl-rs 1.1)
  - `ecosystem::bitnet` - BitNet 1.58-bit quantization (bitnet-quantize 0.2)
  - `ecosystem::trit` - Ternary VSA operations (trit-vsa 0.2)
  - `ecosystem::vsa_optim` - VSA-based optimization (vsa-optim-rs 0.1)
  - `ecosystem::tritter` - Ternary GPU acceleration (tritter-accel 0.1)

#### Unified API Facade
- **`RustAI` struct** - Single entry point for all AI engineering tasks
- **`RustAIConfig`** - Configuration with builder pattern
- **Workflow builders**:
  - `FinetuneBuilder` - Configure LoRA, DoRA, AdaLoRA adapters
  - `QuantizeBuilder` - Configure NF4, FP4, BitNet, INT8 quantization
  - `VsaBuilder` - Configure VSA operations
  - `TrainBuilder` - Configure Axolotl-style YAML pipelines
- **`EcosystemInfo`** - Version info for all integrated crates

### Changed
- Updated description to reflect unified toolkit role
- Updated keywords to include fine-tuning and quantization
- Upgraded PyO3 to 0.27 for Python 3.13/3.14 support
- Updated bitnet-quantize to 0.2, trit-vsa to 0.2

### Dependencies
- peft-rs = "1.0" (resolves to 1.0.3)
- qlora-rs = "1.0" (resolves to 1.0.4)
- unsloth-rs = "1.0"
- axolotl-rs = "1.1"
- bitnet-quantize = "0.2"
- trit-vsa = "0.2"
- vsa-optim-rs = "0.1"
- tritter-accel = "0.1"

### Removed
- Removed obsolete "REQUIRED UPSTREAM FIXES" documentation (peft-rs 1.0.3 resolved safetensors compatibility)

## [0.2.0] - 2026-01-25

### Added

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
- **Integration tests** (`tests/integration.rs`) - Comprehensive public API tests
- **Examples directory** with 3 runnable examples:
  - `device_selection.rs` - Device selection patterns
  - `memory_tracking.rs` - Memory estimation and tracking
  - `error_handling.rs` - Error handling patterns
- **Benchmark suite** (`benches/core_ops.rs`) - Performance benchmarks
- **CHANGELOG.md** - Version history documentation

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

[Unreleased]: https://github.com/tzervas/rust-ai-core/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/tzervas/rust-ai-core/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/tzervas/rust-ai-core/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/tzervas/rust-ai-core/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/tzervas/rust-ai-core/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tzervas/rust-ai-core/releases/tag/v0.1.0
