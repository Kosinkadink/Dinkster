"""ResourcePins: liveness for resource references held by live runs.

A value carrying RESOURCE_ID_META_KEY is a reference to owner-resolved
state, and the owner may live in another process. Releasing that state
(the ram lane of --cache-ram) is only safe when nothing on this side can
still use a reference: caches are invalidated by the release path, but a
*run* holds envelopes in plain Python variables between invocations, and
those are invisible to any cache. This registry is where they become
visible: the engine pins every resource-referencing envelope it routes
for the duration of the run, and release consults the pins.

Release itself needs one more move: a reference can be *in flight* -
retrieved but not yet pinned - across a genuine suspension point (an
orphaned single-flight future whose owner run already ended, a cache
whose get() truly suspends). ``condemn()`` closes that window: it
atomically refuses if the resource is pinned, and while a resource is
condemned every ``pin()`` returns False, so a late arrival is *detected*
and the caller recomputes through the ordinary miss path instead of
using a stub that is about to dangle. A refused release is rolled back
with ``absolve()``; a confirmed one stays condemned forever - resource
ids are never reused, so the tombstone can only ever refuse a genuinely
stale reference.

Event-loop confined, like the engine that feeds it: not thread-safe by
design. Node authors never see this type - pinning is host plumbing
(DESIGN 3.10, hazard H2).
"""

from __future__ import annotations

__all__ = ["ResourcePins"]


class ResourcePins:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._condemned: set[str] = set()

    def pin(self, resource_id: str) -> bool:
        """Register one live reference. False refuses the pin: the resource
        is condemned (released, or mid-release) - the caller must treat
        whatever carried the reference as invalid and recompute."""
        if resource_id in self._condemned:
            return False
        self._counts[resource_id] = self._counts.get(resource_id, 0) + 1
        return True

    def unpin(self, resource_id: str) -> None:
        count = self._counts.get(resource_id, 0)
        if count <= 1:
            self._counts.pop(resource_id, None)
        else:
            self._counts[resource_id] = count - 1

    def pinned(self, resource_id: str) -> bool:
        return self._counts.get(resource_id, 0) > 0

    def condemn(self, resource_id: str) -> bool:
        """Atomically mark a resource as being released. False means it is
        pinned by a live run (or already condemned by a concurrent
        release) and must not be released. While condemned, pin() refuses,
        so no new reference can slip in between this check and the owner
        dropping the resource."""
        if self.pinned(resource_id) or resource_id in self._condemned:
            return False
        self._condemned.add(resource_id)
        return True

    def absolve(self, resource_id: str) -> None:
        """Roll back a condemnation: the owner refused the release (the
        resident was used since it was proposed), so references stay
        valid. Never called after a confirmed release - that tombstone is
        permanent."""
        self._condemned.discard(resource_id)

    def condemned(self, resource_id: str) -> bool:
        return resource_id in self._condemned
