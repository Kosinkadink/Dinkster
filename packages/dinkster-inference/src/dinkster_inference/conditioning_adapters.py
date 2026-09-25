"""Generation-scoped adapters for family-owned conditioning payloads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .conditioning_wire import ConditioningCarrier


class ConditioningAdapterLookup(Protocol):
    """Generation-local lookup needed to prepare conditioning payloads."""

    def get(self, id_or_alias: str) -> ConditioningAdapter | None: ...


class ConditioningLayoutIncompatibility(ValueError):
    """A carrier combines token layouts that no family adapter can own."""


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


def conditioning_family_id(carrier: ConditioningCarrier) -> str | None:
    """Return the sole declared token-layout family, or refuse a mixed carrier."""

    family_ids = tuple(
        sorted(
            {
                record.token_layout.family_id
                for record in carrier.conditioning.records
                if record.token_layout is not None
            }
        )
    )
    if len(family_ids) > 1:
        raise ConditioningLayoutIncompatibility(
            "conditioning combines incompatible token-layout families: " + ", ".join(family_ids)
        )
    return family_ids[0] if family_ids else None


def prepare_conditioning(
    carrier: ConditioningCarrier,
    adapters: ConditioningAdapterLookup,
) -> ConditioningCarrier:
    """Run the matching family adapter while preserving canonical records."""

    family_id = conditioning_family_id(carrier)
    adapter = adapters.get(family_id) if family_id is not None else None
    if adapter is None:
        return carrier
    prepared = adapter.prepare(carrier)
    if type(prepared) is not ConditioningCarrier:
        raise TypeError(f"conditioning adapter {adapter.id!r} must return ConditioningCarrier")
    if prepared.conditioning != carrier.conditioning:
        raise ConditioningLayoutIncompatibility(
            f"conditioning adapter {adapter.id!r} changed canonical conditioning records"
        )
    return prepared


def release_conditioning(
    carrier: ConditioningCarrier,
    adapters: ConditioningAdapterLookup,
) -> None:
    """Release one prepared carrier through its family-owned adapter."""

    family_id = conditioning_family_id(carrier)
    adapter = adapters.get(family_id) if family_id is not None else None
    if adapter is not None:
        adapter.release(carrier)


__all__ = [
    "ConditioningAdapter",
    "ConditioningAdapterLookup",
    "ConditioningLayoutIncompatibility",
    "conditioning_family_id",
    "prepare_conditioning",
    "release_conditioning",
]
