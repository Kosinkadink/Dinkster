"""Structured engine events: the substrate for progress reporting and
dev-mode diagnostics (DESIGN 3.8). Everything the engine knows, surfaced."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

EventKind = Literal[
    "run_started",
    "node_cached",
    "cache_miss",
    "node_started",
    "node_event",
    "node_finished",
    "node_failed",
    "node_skipped",
    "value_diagnostics",
    "region_expanded",
    "region_iteration_started",
    "region_iteration_finished",
    "region_finished",
    "run_finished",
]


@dataclass(frozen=True)
class EngineEvent:
    kind: EventKind
    run_id: str
    node_id: str | None = None
    detail: Mapping[str, object] = field(default_factory=dict[str, object])


EventListener = Callable[[EngineEvent], None]
