"""Dinkster's fork-backed execution provider."""

from .pool import ResidentPool, VaeSource, default_pool, memory_consumers

__all__ = [
    "ResidentPool",
    "VaeSource",
    "default_pool",
    "memory_consumers",
]
