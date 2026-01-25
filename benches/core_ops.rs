//! Benchmarks for rust-ai-core performance-critical paths.

use candle_core::DType;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rust_ai_core::{
    estimate_tensor_bytes, get_device,
    memory::{estimate_attention_memory, MemoryTracker},
    DTypeExt, DeviceConfig,
};
use std::hint::black_box;

/// Benchmark device selection (should be very fast).
fn bench_device_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_selection");

    // Default configuration
    group.bench_function("default_config", |b| {
        b.iter(|| {
            let config = DeviceConfig::default();
            black_box(config)
        })
    });

    // From environment (involves env var lookups)
    group.bench_function("from_env", |b| {
        b.iter(|| {
            let config = DeviceConfig::from_env();
            black_box(config)
        })
    });

    // Builder pattern
    group.bench_function("builder_chain", |b| {
        b.iter(|| {
            let config = DeviceConfig::new()
                .with_cuda_device(0)
                .with_force_cpu(false)
                .with_crate_name("bench");
            black_box(config)
        })
    });

    // Full get_device call (forced CPU for consistency)
    group.bench_function("get_device_cpu", |b| {
        let config = DeviceConfig::new().with_force_cpu(true);
        b.iter(|| {
            let device = get_device(black_box(&config)).unwrap();
            black_box(device)
        })
    });

    group.finish();
}

/// Benchmark memory estimation functions.
fn bench_memory_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_estimation");

    // Simple tensor estimation
    for dims in [2, 3, 4] {
        let shape: Vec<usize> = (0..dims).map(|_| 64).collect();
        group.bench_with_input(
            BenchmarkId::new("estimate_tensor_bytes", dims),
            &shape,
            |b, shape| b.iter(|| black_box(estimate_tensor_bytes(black_box(shape), DType::F32))),
        );
    }

    // Attention memory estimation
    for seq_len in [512, 1024, 2048, 4096] {
        group.bench_with_input(
            BenchmarkId::new("estimate_attention", seq_len),
            &seq_len,
            |b, &seq_len| {
                b.iter(|| {
                    black_box(estimate_attention_memory(
                        black_box(1),  // batch
                        black_box(32), // heads
                        black_box(seq_len),
                        black_box(128), // head_dim
                        DType::BF16,
                    ))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory tracker operations.
fn bench_memory_tracker(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_tracker");

    // Allocation tracking (hot path during training)
    group.bench_function("allocate_deallocate", |b| {
        let tracker = MemoryTracker::with_limit(1_000_000_000);
        b.iter(|| {
            tracker.allocate(black_box(1_000_000)).unwrap();
            tracker.deallocate(black_box(1_000_000));
        })
    });

    // would_fit check (used for preflight checks)
    group.bench_function("would_fit", |b| {
        let tracker = MemoryTracker::with_limit(1_000_000_000);
        tracker.allocate(500_000_000).unwrap();
        b.iter(|| black_box(tracker.would_fit(black_box(400_000_000))))
    });

    // Estimate with overhead
    group.bench_function("estimate_with_overhead", |b| {
        let tracker = MemoryTracker::new();
        let shape = [32, 64, 128];
        b.iter(|| black_box(tracker.estimate_with_overhead(black_box(&shape), DType::F32)))
    });

    group.finish();
}

/// Benchmark DType operations.
fn bench_dtype_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_ops");

    // is_half_precision (used for mixed precision checks)
    group.bench_function("is_half_precision", |b| {
        let dtype = DType::BF16;
        b.iter(|| black_box(dtype.is_half_precision()))
    });

    // is_training_dtype
    group.bench_function("is_training_dtype", |b| {
        let dtype = DType::F32;
        b.iter(|| black_box(dtype.is_training_dtype()))
    });

    // accumulator_dtype
    group.bench_function("accumulator_dtype", |b| {
        let dtype = DType::BF16;
        b.iter(|| black_box(dtype.accumulator_dtype()))
    });

    // name (string lookup)
    group.bench_function("name", |b| {
        let dtype = DType::F32;
        b.iter(|| black_box(dtype.name()))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_device_selection,
    bench_memory_estimation,
    bench_memory_tracker,
    bench_dtype_operations,
);
criterion_main!(benches);
