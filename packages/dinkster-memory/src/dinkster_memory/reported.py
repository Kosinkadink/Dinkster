"""ReportedTelemetry: worker-reported measurements, parent side.

Workers own measurement (only the child's interpreter has torch and can
ask the device), the engine owns policy - so measurements cross the
boundary as plain numbers and land here, a per-source snapshot store the
governor's TelemetryProbe seat reads from. Each source is one live
worker session: its snapshot is replaced whole on every report and
dropped whole when the session ends, so a dead worker can never leave a
stale "free" number behind.

INFORMATIONAL ONLY: nothing here feeds admission. Physical free is
global while allocator-reclaimable bytes are process-local; measured
free entering admission math would double-count and admit memory another
process cannot actually release (Oracle review, 2026-07-26 - see
ROADMAP "Memory governance (interprocess)"). Measurements exist so
/memory/status shows ground truth beside declared budgets, nothing more.
"""

from __future__ import annotations

import time
from collections.abc import Mapping

from .governor import MeasuredMemory


class ReportedTelemetry:
    """Per-source measured-memory snapshots, freshest-wins on overlap.

    Sources are opaque objects (parent-side worker sessions), compared by
    identity. Two workers with the same DeviceMap-translated device key
    genuinely see the same silicon, so serving the freshest snapshot is
    the honest aggregation: both read the same driver counter, and newer
    beats staler.
    """

    def __init__(self) -> None:
        # Keyed by the source object itself (identity hash), never id():
        # a recycled id must not let a new session inherit a dead one's
        # snapshot. Sessions clear themselves on close AND read-loop
        # death, so entries do not outlive their workers.
        self._snapshots: dict[object, tuple[float, int, dict[str, MeasuredMemory]]] = {}
        self._sequence = 0

    def update(self, source: object, measurements: Mapping[str, MeasuredMemory]) -> None:
        """Replace ``source``'s snapshot whole - a report is a moment in
        time, so a device absent from the new snapshot is no longer
        measured by that source."""
        self._sequence += 1
        self._snapshots[source] = (time.monotonic(), self._sequence, dict(measurements))

    def clear(self, source: object) -> None:
        """Drop ``source``'s snapshot (its worker closed or died). Unknown
        sources are ignored - cleanup paths must not raise over
        already-gone state."""
        self._snapshots.pop(source, None)

    def probe(self, device: str) -> MeasuredMemory | None:
        """The freshest measurement any source reports for ``device``;
        None when nobody measures it (honest absence, never a fake
        zero). This is the governor's TelemetryProbe shape."""
        freshest: tuple[float, int, MeasuredMemory] | None = None
        for stamp, sequence, snapshot in self._snapshots.values():
            measured = snapshot.get(device)
            if measured is None:
                continue
            if freshest is None or (stamp, sequence) > freshest[:2]:
                freshest = (stamp, sequence, measured)
        return None if freshest is None else freshest[2]

    def devices(self) -> frozenset[str]:
        """Every device any source currently measures - the device
        universe the governor's status() unions in, so measured but
        unbudgeted devices are visible without inventing a budget."""
        keys: set[str] = set()
        for _, _, snapshot in self._snapshots.values():
            keys.update(snapshot)
        return frozenset(keys)
