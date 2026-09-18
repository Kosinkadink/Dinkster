"""Torch-free accelerator reserve and budget policy."""

from __future__ import annotations

from dataclasses import dataclass

MiB = 1024**2
GiB = 1024**3

DEFAULT_ACCELERATOR_HEADROOM_BYTES = 256 * MiB
DEFAULT_INFERENCE_RESERVE_BYTES = int(0.8 * GiB)


class AcceleratorMemoryPolicyError(ValueError):
    """An accelerator reserve policy input is malformed."""


def _non_negative_bytes(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise AcceleratorMemoryPolicyError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class ResolvedAcceleratorMemoryPolicy:
    """One policy resolved against a physical device total."""

    total_bytes: int
    physical_headroom_bytes: int
    inference_reserve_bytes: int
    hard_budget_bytes: int | None
    effective_budget_bytes: int
    budget_headroom_bytes: int
    residency_capacity_bytes: int

    @property
    def minimum_free_bytes(self) -> int:
        return self.physical_headroom_bytes + self.inference_reserve_bytes

    @property
    def insufficient_total(self) -> bool:
        return self.minimum_free_bytes > self.total_bytes

    @property
    def budget_exceeds_total(self) -> bool:
        return self.hard_budget_bytes is not None and self.hard_budget_bytes > self.total_bytes


@dataclass(frozen=True)
class AcceleratorMemoryPolicy:
    """Physical headroom, inference reserve, and hard-budget derivation."""

    physical_headroom_bytes: int = DEFAULT_ACCELERATOR_HEADROOM_BYTES
    inference_reserve_bytes: int = DEFAULT_INFERENCE_RESERVE_BYTES

    def __post_init__(self) -> None:
        _non_negative_bytes(self.physical_headroom_bytes, "physical accelerator headroom")
        _non_negative_bytes(self.inference_reserve_bytes, "inference working reserve")

    @property
    def minimum_free_bytes(self) -> int:
        return self.physical_headroom_bytes + self.inference_reserve_bytes

    def resolve(
        self,
        total_bytes: int,
        hard_budget_bytes: int | None = None,
    ) -> ResolvedAcceleratorMemoryPolicy:
        if type(total_bytes) is not int or total_bytes < 0:
            raise AcceleratorMemoryPolicyError("accelerator total must be a non-negative integer")
        if hard_budget_bytes is not None:
            hard_budget_bytes = _non_negative_bytes(
                hard_budget_bytes,
                "hard accelerator budget",
            )
        effective_budget = (
            total_bytes if hard_budget_bytes is None else min(total_bytes, hard_budget_bytes)
        )
        budget_headroom = total_bytes - effective_budget
        return ResolvedAcceleratorMemoryPolicy(
            total_bytes=total_bytes,
            physical_headroom_bytes=self.physical_headroom_bytes,
            inference_reserve_bytes=self.inference_reserve_bytes,
            hard_budget_bytes=hard_budget_bytes,
            effective_budget_bytes=effective_budget,
            budget_headroom_bytes=budget_headroom,
            residency_capacity_bytes=max(0, effective_budget - self.minimum_free_bytes),
        )


__all__ = [
    "AcceleratorMemoryPolicy",
    "AcceleratorMemoryPolicyError",
    "DEFAULT_ACCELERATOR_HEADROOM_BYTES",
    "DEFAULT_INFERENCE_RESERVE_BYTES",
    "ResolvedAcceleratorMemoryPolicy",
]
