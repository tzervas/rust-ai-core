"""Type stubs for rust_ai_core_bindings."""

from __future__ import annotations
from typing import Optional, Dict, Any, List

__version__: str

class MemoryTracker:
    """Opaque handle for tracking memory allocations."""
    ...

# Memory estimation functions
def estimate_tensor_bytes(shape: List[int], dtype: str = "f32") -> int:
    """
    Estimate memory required for a tensor with given shape and dtype.

    Args:
        shape: List of dimension sizes (e.g., [1, 512, 4096])
        dtype: Data type string: "f32", "f16", "bf16", "f64", "i32", "i64", "u8", "u32"

    Returns:
        Number of bytes required for the tensor.
    """
    ...

def estimate_attention_memory(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: str = "bf16",
) -> int:
    """
    Estimate memory for attention computation (O(n^2) attention scores).

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension per head
        dtype: Data type string (default: "bf16")

    Returns:
        Total bytes for attention computation.
    """
    ...

# Memory tracker functions
def create_memory_tracker(limit_gb: float = 8.0, overhead_factor: float = 1.1) -> MemoryTracker:
    """
    Create a memory tracker with a limit.

    Args:
        limit_gb: Memory limit in gigabytes
        overhead_factor: Overhead multiplier for allocations

    Returns:
        A MemoryTracker handle.
    """
    ...

def tracker_would_fit(tracker: MemoryTracker, bytes: int) -> bool:
    """Check if an allocation would fit within the tracker's limit."""
    ...

def tracker_allocate(tracker: MemoryTracker, bytes: int) -> None:
    """Record an allocation in the tracker."""
    ...

def tracker_deallocate(tracker: MemoryTracker, bytes: int) -> None:
    """Record a deallocation in the tracker."""
    ...

def tracker_allocated_bytes(tracker: MemoryTracker) -> int:
    """Get current allocation from the tracker."""
    ...

def tracker_peak_bytes(tracker: MemoryTracker) -> int:
    """Get peak allocation from the tracker."""
    ...

def tracker_limit_bytes(tracker: MemoryTracker) -> int:
    """Get the memory limit from the tracker."""
    ...

def tracker_estimate_with_overhead(
    tracker: MemoryTracker,
    shape: List[int],
    dtype: str = "f32",
) -> int:
    """Estimate bytes with overhead factor."""
    ...

def tracker_reset(tracker: MemoryTracker) -> None:
    """Reset the memory tracker to initial state."""
    ...

# Device functions
def cuda_available() -> bool:
    """Check if CUDA is available."""
    ...

def get_device_info(force_cpu: bool = False, cuda_device: int = 0) -> Dict[str, Any]:
    """
    Get device information.

    Args:
        force_cpu: Force CPU device
        cuda_device: CUDA device ordinal

    Returns:
        Dictionary with keys: 'type', 'ordinal', 'name'
    """
    ...

# DType functions
def bytes_per_dtype(dtype: str) -> int:
    """Get bytes per element for a data type."""
    ...

def is_floating_point_dtype(dtype: str) -> bool:
    """Check if a data type is floating point."""
    ...

def accumulator_dtype(dtype: str) -> str:
    """Get the accumulator data type for a given dtype."""
    ...

def supported_dtypes() -> List[str]:
    """Get all supported data types."""
    ...

# Logging functions
def init_logging(level: str = "info", timestamps: bool = True, ansi: bool = True) -> None:
    """
    Initialize logging with configuration.

    Args:
        level: Log level: "trace", "debug", "info", "warn", "error"
        timestamps: Include timestamps
        ansi: Use ANSI colors
    """
    ...

# Utilities
def version() -> str:
    """Get rust-ai-core version."""
    ...

def default_overhead_factor() -> float:
    """Get the default overhead factor for memory estimation."""
    ...
