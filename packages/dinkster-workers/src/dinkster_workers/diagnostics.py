"""Boundary diagnostics: make what a crossing costs visible (DESIGN 3.9).

Sane defaults plus an oblivious authoring path only stay healthy if
developers can see what the defaults cost. Every IsolatedWorker invocation
produces one BoundaryDiagnostic - execute time vs boundary time, transport
and payload size per edge, and whether the codec used was declared or the
correct-but-possibly-slow default fallback. Consumers (dev mode, `dinkster
doctor`, frontends) decide what to surface; the worker just reports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeCost:
    """One value's crossing, attributed to its input or output id.

    ``codec_ms`` totals the codec/copy work at transfer time on both sides
    of the boundary where known (lazy decode-on-resolve is not included).
    ``declared_codec`` False means the type crossed via the default fallback
    codec - correct, but a declaration would make it faster. It is only
    meaningful where the type is registered. ``reused`` means encoded bytes
    were relayed without re-encoding - no codec ran at all, so
    ``declared_codec`` says nothing about such a crossing.
    ``network_bytes``/``transfer_ms`` are payload bytes the persistentCas
    transport actually streamed for this edge and the time spent streaming
    them - zero on a store hit. ``reused`` is conversation-local;
    ``network_bytes == 0`` on a persistentCas edge is the persistent-store
    hit signal, which survives reconnects and re-runs."""

    edge_id: str
    type_id: str
    transport: str
    size_bytes: int
    codec_ms: float
    declared_codec: bool
    reused: bool
    network_bytes: int = 0
    transfer_ms: float = 0.0


@dataclass(frozen=True)
class BoundaryDiagnostic:
    """The cost breakdown of one invocation across a process boundary."""

    invocation_id: str
    node_id: str
    node_type: str
    pack: str
    inputs: tuple[EdgeCost, ...]
    outputs: tuple[EdgeCost, ...]
    execute_ms: float
    round_trip_ms: float

    @property
    def boundary_ms(self) -> float:
        """Round-trip time not spent executing: framing, transfer, codecs."""
        return max(self.round_trip_ms - self.execute_ms, 0.0)


DiagnosticListener = Callable[[BoundaryDiagnostic], None]
