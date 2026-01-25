"""
rust-ai-core-bindings: Python bindings for rust-ai-core ML utilities.

This package provides high-performance Rust implementations of common ML utilities:

- Memory estimation for AI training planning
- Device detection and configuration
- Data type utilities
- Logging configuration
"""

from __future__ import annotations

from .rust_ai_core_bindings import (
    MemoryTracker,
    estimate_tensor_bytes,
    estimate_attention_memory,
    create_memory_tracker,
    tracker_would_fit,
    tracker_allocate,
    tracker_deallocate,
    tracker_allocated_bytes,
    tracker_peak_bytes,
    tracker_limit_bytes,
    tracker_estimate_with_overhead,
    tracker_reset,
    cuda_available,
    get_device_info,
    bytes_per_dtype,
    is_floating_point_dtype,
    accumulator_dtype,
    supported_dtypes,
    init_logging,
    version,
    default_overhead_factor,
)

__version__ = version()
__all__ = [
    "MemoryTracker",
    "estimate_tensor_bytes",
    "estimate_attention_memory",
    "create_memory_tracker",
    "tracker_would_fit",
    "tracker_allocate",
    "tracker_deallocate",
    "tracker_allocated_bytes",
    "tracker_peak_bytes",
    "tracker_limit_bytes",
    "tracker_estimate_with_overhead",
    "tracker_reset",
    "cuda_available",
    "get_device_info",
    "bytes_per_dtype",
    "is_floating_point_dtype",
    "accumulator_dtype",
    "supported_dtypes",
    "init_logging",
    "version",
    "default_overhead_factor",
    "__version__",
]
