// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

// Allow missing docs for macro-generated code in submodules
#![allow(missing_docs)]

//! TypeScript/JavaScript bindings for rust-ai-core.
//!
//! This module provides bindings to expose rust-ai-core's utilities to
//! TypeScript and JavaScript developers via two complementary approaches:
//!
//! ## Binding Strategies
//!
//! | Approach | Runtime | Use Case |
//! |----------|---------|----------|
//! | **napi-rs** | Node.js | Native performance for ML training orchestration |
//! | **wasm-bindgen** | Browser | Isomorphic support, TensorFlow.js/ONNX.js interop |
//!
//! ## Why Two Approaches?
//!
//! - **napi-rs**: Compiles to native Node.js addon (`.node` file). Provides
//!   peak performance and direct access to system resources (CUDA, file I/O).
//!   Ideal for training scripts and server-side ML pipelines.
//!
//! - **wasm-bindgen**: Compiles to WebAssembly. Runs in any browser without
//!   native dependencies. Enables memory estimation and planning tools in
//!   web-based ML IDEs and notebooks.
//!
//! ## API Surface
//!
//! Both bindings expose the same core API:
//!
//! | Function | Description |
//! |----------|-------------|
//! | `estimateTensorBytes(shape, dtype?)` | Memory estimation for tensors |
//! | `estimateAttentionMemory(...)` | Attention memory calculation |
//! | `MemoryTracker` class | Track allocations with limits |
//! | `getDeviceInfo(...)` | Device detection (CUDA/Metal/CPU) |
//! | `cudaAvailable()` | CUDA availability check |
//! | `bytesPerDtype(dtype)` | Element size lookup |
//! | `isFloatingPointDtype(dtype)` | Type classification |
//! | `supportedDtypes()` | List all supported dtypes |
//! | `initLogging(...)` | Configure logging |
//! | `version()` | Get rust-ai-core version |
//!
//! ## Feature Flags
//!
//! - `napi`: Enable Node.js native bindings
//! - `wasm`: Enable WebAssembly browser bindings

// Common types and utilities shared between napi and wasm bindings
pub mod common;

// Node.js native bindings via napi-rs (N-API v4+)
#[cfg(feature = "napi")]
pub mod napi;

// WebAssembly bindings via wasm-bindgen
#[cfg(feature = "wasm")]
pub mod wasm;

// Re-export common types for internal use
pub use common::*;
