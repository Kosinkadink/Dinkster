"""Generation-scoped adapters for family-owned conditioning payloads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .conditioning_wire import ConditioningCarrier


@dataclass(frozen=True)
class ConditioningAdapter:
    """Prepare and release one family's payloads without changing their carrier."""

    id: str
    prepare: Callable[[ConditioningCarrier], ConditioningCarrier]
    release: Callable[[ConditioningCarrier], None]
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not callable(self.prepare):
            raise TypeError("prepare must be callable")
        if not callable(self.release):
            raise TypeError("release must be callable")


__all__ = ["ConditioningAdapter"]
