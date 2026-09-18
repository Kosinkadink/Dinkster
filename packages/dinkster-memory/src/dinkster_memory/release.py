"""Two-phase release for consumers whose references live elsewhere.

``Shedder.shed`` is one honest number - fine when freeing is local and
safe by construction (advisory vram unloading, a cache dropping its own
entries). Releasing a *referenced* resource (the ram lane: dropping a
resident's strong reference) is not: the holder of the references (the
engine process) must invalidate its caches and check its pins between
selection and the actual drop. This contract splits shed into the two
halves that gate straddles:

- ``propose_release`` names candidates - stable item ids, the resource
  ids their envelopes carry, declared bytes, and a use-clock token.
- ``release`` drops exactly the still-valid candidates: one whose token
  no longer matches was used since it was proposed and is refused. The
  outcome maps item id -> bytes freed; a candidate absent from the
  mapping was refused, and the caller must roll back whatever it staked
  on that release.

Both halves are consumer-local and make no transport claims; the worker
host and memory relay carry them across the process boundary and add the
in-flight and pinning gates there (DESIGN 3.10, open questions).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from .governor import PressureSignal

__all__ = [
    "FullReleaseConsumer",
    "FullReleaseCommitResult",
    "FullReleaseResult",
    "FullReleasableConsumer",
    "ReleasableConsumer",
    "ReleaseCandidate",
]

FullReleaseStatus = Literal["complete", "busy", "unsupported", "error"]


@dataclass(frozen=True)
class FullReleaseResult:
    """Terminal outcome of one consumer's normal full-release operation."""

    status: FullReleaseStatus
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status == "error":
            if not self.error:
                raise ValueError("an error full-release result requires an error message")
        elif self.error is not None:
            raise ValueError("only an error full-release result may carry an error message")


@runtime_checkable
class FullReleaseConsumer(Protocol):
    """A consumer that can terminally release every volatile holding."""

    async def full_release(self) -> FullReleaseResult: ...


@dataclass(frozen=True)
class FullReleaseCommitResult:
    """Released candidates and the consumer's complete maintenance outcome."""

    released: Mapping[str, int]
    result: FullReleaseResult


@dataclass(frozen=True)
class ReleaseCandidate:
    """One thing a consumer would release under the proposed pressure.

    ``token`` is the consumer's use-clock reading at proposal time; the
    consumer refuses the release if the item was used since (the same
    compare-and-drop discipline everywhere: a stale decision must not
    destroy fresh state).
    """

    item_id: str
    resource_id: str
    nbytes: int
    token: str


@runtime_checkable
class ReleasableConsumer(Protocol):
    """A Shedder whose ram-lane shedding can be gated by a reference holder.

    Implementations keep ``shed`` for local pressure; the split surface
    exists so a *remote* gate (cache invalidation and pinning in another
    process) can run between selection and drop.
    """

    def propose_release(self, pressure: PressureSignal) -> Sequence[ReleaseCandidate]:
        """Select what the consumer would release; free nothing yet.

        Contract: an item's token must advance whenever the item is named
        in an invocation result (serializing a reference counts as a use).
        The cross-process gate depends on it - a candidate proposed before
        the result registered its in-flight hold must fail the token check
        at release(), or the gate has a window where a resident whose stub
        the parent just decoded gets dropped. ResidentPool satisfies this
        naturally: encoding a resident reference touches its use clock.
        """
        ...

    async def release(
        self, device: str, candidates: Sequence[ReleaseCandidate]
    ) -> Mapping[str, int]:
        """Drop the still-valid candidates. Returns item id -> bytes freed;
        a candidate missing from the mapping was refused (used since
        proposal, or already gone) and remains resident."""
        ...


@runtime_checkable
class FullReleasableConsumer(ReleasableConsumer, Protocol):
    """A referenced consumer that exposes every holding to a remote gate."""

    def propose_full_release(self) -> Sequence[ReleaseCandidate]:
        """Return every current holding, including items with zero declared cost."""
        ...

    async def release_full(self, candidates: Sequence[ReleaseCandidate]) -> FullReleaseCommitResult:
        """Release gated candidates and report whether any holding or failure remains."""
        ...
