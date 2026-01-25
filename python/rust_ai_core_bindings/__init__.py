"""
rust-ai-core-bindings: Python bindings for rust-ai-core ML utilities.

This package provides high-performance Rust implementations of common ML utilities:

- Memory estimation for AI training planning
- Device detection and configuration
- Data type utilities
- Logging configuration

Example
-------
>>> from rust_ai_core_bindings import (
...     estimate_tensor_bytes,
...     estimate_attention_memory,
...     create_memory_tracker,
...     cuda_available,
...     get_device_info,
... )
>>>
>>> # Estimate memory for a large tensor
>>> bytes_needed = estimate_tensor_bytes([1, 512, 4096], "f32")
>>> print(f"Tensor needs {bytes_needed / 1024**2:.2f} MB")
Tensor needs 8.00 MB
>>>
>>> # Check device availability
>>> if cuda_available():
...     device = get_device_info()
...     print(f"Using {device['type']} device")
...
>>> # Track memory during training
>>> tracker = create_memory_tracker(limit_gb=8.0)
>>> print(f"Memory limit: {tracker_limit_bytes(tracker) / 1024**3:.2f} GB")
Memory limit: 8.00 GB
"""

from __future__ import annotations

# Import all functions from the Rust extension
from .rust_ai_core_bindings import (
    # Classes
    MemoryTracker,
    # Memory estimation
    estimate_tensor_bytes,
    estimate_attention_memory,
    # Memory tracker
    create_memory_tracker,
    tracker_would_fit,
    tracker_allocate,
    tracker_deallocate,
    tracker_allocated_bytes,
    tracker_peak_bytes,
    tracker_limit_bytes,
    tracker_estimate_with_overhead,
    tracker_reset,
    # Device utilities
    cuda_available,
    get_device_info,
    # DType utilities
    bytes_per_dtype,
    is_floating_point_dtype,
    accumulator_dtype,
    supported_dtypes,
    # Logging
    init_logging,
    # Utilities
    version,
    default_overhead_factor,
)

__version__ = version()
__all__ = [
    # Classes
    "MemoryTracker",
    # Memory estimation
    "estimate_tensor_bytes",
    "estimate_attention_memory",
    # Memory tracker
    "create_memory_tracker",
    "tracker_would_fit",
    "tracker_allocate",
    "tracker_deallocate",
    "tracker_allocated_bytes",
    "tracker_peak_bytes",
    "tracker_limit_bytes",
    "tracker_estimate_with_overhead",
    "tracker_reset",
    # Device utilities
    "cuda_available",
    "get_device_info",
    # DType utilities
    "bytes_per_dtype",
    "is_floating_point_dtype",
    "accumulator_dtype",
    "supported_dtypes",
    # Logging
    "init_logging",
    # Utilities
    "version",
    "default_overhead_factor",
    # Version
    "__version__",
]
