"""The consumer detail contract: what a memory panel renders (DESIGN 3.10).

The mandatory Shedder surface is one number per device - enough for
admission, useless for a frontend that wants to show *which models* hold
the bytes. This contract is the optional second level: a consumer that
knows its contents as items (a model pool does; a byte-blob cache may not)
describes them with stable IDs, display names, and byte decompositions -
and the governor aggregates without interpreting, staying model-agnostic.

Shapes are dictated by the failure modes of the scraping they replace
(kijai's ComfyUI-MemoryVisualization against ComfyUI internals):

- ``item_id`` is stable across mutations, so "unload this item" cannot
  race the list it addresses the way a mutable index does.
- ``display_name`` is the checkpoint filename or asset identity - never a
  Python class name, which collides the moment two checkpoints share an
  architecture.
- ``PageMap`` carries its own geometry (page size, one flag per page), so
  no client ever hardcodes what the server knows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PageMap:
    """Page-granular residency, geometry included.

    ``flags`` holds one value per page; 0 means not device-resident, and
    nonzero values are consumer-defined states (resident, pinned, faulting
    in) that clients render, never interpret. Page count is ``len(flags)``
    - derived, not stored twice.
    """

    page_bytes: int
    flags: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.page_bytes <= 0:
            raise ValueError("page_bytes must be > 0")

    @property
    def page_count(self) -> int:
        return len(self.flags)


@dataclass(frozen=True)
class ConsumerItem:
    """One named thing a consumer holds: a loaded model, a pinned buffer.

    ``bytes_by_residency`` uses the same residency-class keys budgets and
    COST_META_KEY use (``"vram:cuda:0"``, ``"ram"``, ``"disk"``) - one
    vocabulary from envelope to panel. An item may decompose across
    several classes at once (device-resident pages plus a pinned host
    copy); classes absent from the map hold zero.
    """

    item_id: str
    display_name: str
    bytes_by_residency: Mapping[str, int] = field(default_factory=dict[str, int])
    pages: PageMap | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bytes_by_residency", dict(self.bytes_by_residency))
        if not self.item_id:
            raise ValueError("item_id must be non-empty")
        if not self.display_name:
            raise ValueError("display_name must be non-empty")
        for residency, nbytes in self.bytes_by_residency.items():
            if nbytes < 0:
                raise ValueError(f"bytes for {residency!r} must be >= 0, got {nbytes}")


@runtime_checkable
class DetailedConsumer(Protocol):
    """A Shedder that can also name what it holds.

    Implementing this contract is what makes a consumer targetable by
    item: the governor routes item-scoped pressure (``PressureSignal.items``)
    only to consumers that expose details, because "unload item X" sent to
    a consumer that cannot resolve X must free nothing, not something.
    """

    def details(self) -> Sequence[ConsumerItem]:
        """Current items, stable IDs included. Called on demand (a status
        query), never on the admission path - may be moderately costly."""
        ...
