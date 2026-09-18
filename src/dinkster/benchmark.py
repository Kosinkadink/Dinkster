"""Native performance benchmarking, first slice (DESIGN 3.9): one record
per run, assembled from the events the engine and boundary already emit.

comfyui-benchmark had to monkey-patch ~ten ComfyUI internals to see phase
timings; Dinkster has the seams natively, so this module is assembly, not
surgery. The BenchmarkAssembler is a passive EventListener tee - it never
participates in execution (H15's spirit: no signature, no cache key, no
scheduling decision reads benchmark state), and benchmark mode off means
no listener is attached at all - zero work on the hot path.

What a record contains and where each part comes from:

- per-occurrence timings: ``node_finished.duration_ms`` (engine-measured,
  authoritative) plus host-side start/end stamps on the same monotonic
  clock the hardware sampler uses;
- cache attribution: ``node_cached`` (hit, coalesced or not) and
  ``cache_miss`` explanations (reason + changed input IDS - never values);
- boundary costs: BoundaryDiagnostic per isolated invocation (execute vs
  boundary ms, per-edge transport/size/codec, fallback-codec markers);
- hardware timeline: a host-owned sampler thread (shared system-memory
  provider and NVML when importable) on ``time.perf_counter()``.

Privacy rule (DESIGN 3.9): records carry identities and costs, never
payloads. Output summaries are stripped to typeId/length (the inline
``value`` channel for small scalars is deliberately dropped), miss
explanations carry input IDs only, and no event detail is copied
wholesale - every record field is an explicit pick.

Known first-slice limits, documented rather than papered over: a run that
fails emits no ``run_finished``, so no record is cut for it (open runs are
bounded and evicted oldest-first); boundary diagnostics carry no run id,
so correlation is by runtime node id - two concurrent runs executing the
same node id simultaneously could cross-attribute a boundary breakdown.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import sys
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dinkster_engine import Engine, EngineEvent, EventListener
from dinkster_memory import system_memory_snapshot
from dinkster_workers import BoundaryDiagnostic, EdgeCost

__all__ = [
    "RECORD_VERSION",
    "BenchmarkAssembler",
    "HardwareSample",
    "HardwareSampler",
    "default_hardware_probe",
    "default_environment",
    "instrument_engine_factory",
    "record_to_json",
    "write_record",
]

RECORD_VERSION = "dinkster.benchmark/3"

# Open (started, never finished) runs kept while waiting for run_finished.
# A failed run never emits one, so this bounds the leak, not any cache.
_OPEN_RUN_CAP = 64

# The output-summary keys a record may carry. "value" (inline small
# scalars for frontend badges) is deliberately NOT here: benchmark
# records never capture prompt or result values, however small.
_SAFE_OUTPUT_KEYS = ("typeId", "length")

_MISS_ID_FIELDS = ("changed_inputs", "added_inputs", "removed_inputs")


def _rel_ms(t: float, t0: float) -> float:
    return round((t - t0) * 1000.0, 3)


# ---------------------------------------------------------------------------
# Hardware sampling (host policy, like logging configuration - H23)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareSample:
    """One probe reading at a monotonic instant (``time.perf_counter()``
    seconds - the same clock the assembler stamps events with)."""

    t: float
    metrics: Mapping[str, object]


_nvml_handles: list[object] | None = None
_nvml_failed = False


def _nvml_devices() -> list[object]:
    """Lazy NVML init, once; any failure disables GPU sampling quietly."""
    global _nvml_handles, _nvml_failed
    if _nvml_failed:
        return []
    if _nvml_handles is None:
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            _nvml_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())
            ]
        except Exception:
            _nvml_failed = True
            return []
    return _nvml_handles


def default_hardware_probe() -> dict[str, object]:
    """Best-effort process, effective system-memory, and GPU metrics."""
    metrics: dict[str, object] = {}
    try:
        import psutil  # type: ignore[import-not-found]

        proc = psutil.Process()
        with proc.oneshot():
            metrics["rssBytes"] = proc.memory_info().rss
            metrics["cpuPercent"] = proc.cpu_percent(interval=None)
    except Exception:
        pass
    try:
        memory = system_memory_snapshot(include_swap=True)
        metrics["systemUsedBytes"] = memory.effective_used_bytes
        metrics["systemAvailableBytes"] = memory.effective_available_bytes
        metrics["systemTotalBytes"] = memory.effective_total_bytes
        metrics["systemMemoryProvenance"] = list(memory.provenance)
        if (
            memory.effective_swap_total_bytes is not None
            and memory.effective_swap_available_bytes is not None
        ):
            metrics["systemSwapUsedBytes"] = (
                memory.effective_swap_total_bytes - memory.effective_swap_available_bytes
            )
            metrics["systemSwapAvailableBytes"] = memory.effective_swap_available_bytes
            metrics["systemSwapTotalBytes"] = memory.effective_swap_total_bytes
    except Exception:
        pass
    try:
        import pynvml  # type: ignore[import-not-found]

        gpus: list[dict[str, object]] = []
        for index, handle in enumerate(_nvml_devices()):
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            entry: dict[str, object] = {
                "index": index,
                "vramUsedBytes": int(mem.used),
                "vramTotalBytes": int(mem.total),
            }
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                entry["utilizationPercent"] = int(util.gpu)
            except Exception:
                pass
            gpus.append(entry)
        if gpus:
            metrics["gpus"] = gpus
    except Exception:
        pass
    return metrics


class HardwareSampler:
    """Host-owned background sampler on the shared monotonic clock.

    Packs never poll hardware themselves (DESIGN 3.9); the host runs one
    of these and the assembler windows its samples per run. The buffer is
    bounded (oldest samples fall off), so a long-lived server never grows
    without bound. ``probe`` is injectable for tests and for hosts with
    their own telemetry source.
    """

    def __init__(
        self,
        *,
        interval_s: float = 0.5,
        probe: Callable[[], Mapping[str, object]] | None = None,
        capacity: int = 4096,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._interval = interval_s
        self._probe = probe if probe is not None else default_hardware_probe
        self._clock = clock
        self._samples: deque[HardwareSample] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> None:
        """Take one reading now; a failing probe records nothing."""
        try:
            metrics = dict(self._probe())
        except Exception:
            return
        with self._lock:
            self._samples.append(HardwareSample(self._clock(), metrics))

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.wait(self._interval):
                self.sample_once()

        self._thread = threading.Thread(target=loop, name="dinkster-benchmark-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None

    def samples_between(self, t0: float, t1: float) -> list[HardwareSample]:
        with self._lock:
            return [s for s in self._samples if t0 <= s.t <= t1]


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------


@dataclass
class _OpenRun:
    t0: float
    started_at_utc: str
    planned: tuple[str, ...]
    occurrences: OrderedDict[str, dict[str, object]] = field(default_factory=OrderedDict)
    failed: int = 0


def default_environment() -> dict[str, object]:
    """The comparability envelope (DESIGN 3.9): what machine/build produced
    this record. Hosts extend it (pack set, GPU names) before runs start."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


class BenchmarkAssembler:
    """Folds engine events + boundary diagnostics into one record per run.

    Both callbacks are synchronous and cheap (dict writes); the engine's
    emit path and the worker's diagnostic path stay non-blocking. Not
    thread-safe by design - engine events and boundary diagnostics both
    arrive on the host loop's thread, matching EventHub's model.
    """

    def __init__(
        self,
        on_record: Callable[[dict[str, object]], None],
        *,
        sampler: HardwareSampler | None = None,
        environment: Mapping[str, object] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._on_record = on_record
        self._sampler = sampler
        self._clock = clock
        #: Mutable on purpose: hosts add context (pack ids, device names)
        #: after construction but before runs start.
        self.environment: dict[str, object] = (
            dict(environment) if environment is not None else default_environment()
        )
        self._runs: OrderedDict[str, _OpenRun] = OrderedDict()
        # node_id -> boundary breakdown, pending until that node finishes.
        # Diagnostics carry no run id, so this is assembler-global (see the
        # module docstring's concurrency caveat).
        self._pending_boundary: dict[str, dict[str, object]] = {}

    # -- engine events ------------------------------------------------------

    def on_engine_event(self, event: EngineEvent) -> None:
        handler = getattr(self, f"_on_{event.kind}", None)
        if handler is not None:
            handler(event)

    def _occurrence(self, event: EngineEvent) -> dict[str, object] | None:
        run = self._runs.get(event.run_id)
        if run is None or event.node_id is None:
            return None
        return run.occurrences.setdefault(event.node_id, {"nodeId": event.node_id})

    def _on_run_started(self, event: EngineEvent) -> None:
        planned = event.detail.get("planned")
        self._runs[event.run_id] = _OpenRun(
            t0=self._clock(),
            started_at_utc=_dt.datetime.now(_dt.UTC).isoformat(),
            planned=tuple(planned) if isinstance(planned, (list, tuple)) else (),
        )
        while len(self._runs) > _OPEN_RUN_CAP:
            self._runs.popitem(last=False)  # failed runs never finalize

    def _on_node_started(self, event: EngineEvent) -> None:
        occurrence = self._occurrence(event)
        run = self._runs.get(event.run_id)
        if occurrence is not None and run is not None:
            occurrence["startMs"] = _rel_ms(self._clock(), run.t0)

    def _on_cache_miss(self, event: EngineEvent) -> None:
        occurrence = self._occurrence(event)
        if occurrence is None:
            return
        miss: dict[str, object] = {}
        reason = event.detail.get("reason")
        if isinstance(reason, str):
            miss["reason"] = reason
        for fieldname in _MISS_ID_FIELDS:
            ids = event.detail.get(fieldname)
            if isinstance(ids, (list, tuple)) and ids:
                miss[fieldname] = list(ids)  # input IDS, never values
        occurrence["miss"] = miss

    def _on_node_finished(self, event: EngineEvent) -> None:
        occurrence = self._occurrence(event)
        run = self._runs.get(event.run_id)
        if occurrence is None or run is None:
            return
        occurrence["state"] = "executed"
        occurrence["endMs"] = _rel_ms(self._clock(), run.t0)
        duration = event.detail.get("duration_ms")
        if isinstance(duration, (int, float)):
            occurrence["durationMs"] = float(duration)
        self._pick_identity(occurrence, event)
        boundary = self._pending_boundary.pop(event.node_id or "", None)
        if boundary is not None:
            occurrence["boundary"] = boundary

    def _on_node_cached(self, event: EngineEvent) -> None:
        occurrence = self._occurrence(event)
        run = self._runs.get(event.run_id)
        if occurrence is None or run is None:
            return
        occurrence["state"] = "cached"
        occurrence["endMs"] = _rel_ms(self._clock(), run.t0)
        if event.detail.get("coalesced") is True:
            occurrence["coalesced"] = True
        cache_layer = event.detail.get("cacheLayer")
        if isinstance(cache_layer, str):
            occurrence["cacheLayer"] = cache_layer
        self._pick_identity(occurrence, event)

    def _on_node_skipped(self, event: EngineEvent) -> None:
        occurrence = self._occurrence(event)
        run = self._runs.get(event.run_id)
        if occurrence is None or run is None:
            return
        occurrence["state"] = "skipped"
        occurrence["endMs"] = _rel_ms(self._clock(), run.t0)
        skip: dict[str, object] = {}
        for key in ("input", "origin", "reason"):
            value = event.detail.get(key)
            if isinstance(value, str):
                skip[key] = value
        if skip:
            occurrence["skip"] = skip

    def _on_node_failed(self, event: EngineEvent) -> None:
        occurrence = self._occurrence(event)
        run = self._runs.get(event.run_id)
        if occurrence is None or run is None:
            return
        occurrence["state"] = "failed"
        occurrence["endMs"] = _rel_ms(self._clock(), run.t0)
        message = event.detail.get("message")
        if isinstance(message, str):
            occurrence["error"] = message
        run.failed += 1

    def _on_region_expanded(self, event: EngineEvent) -> None:
        occurrence = self._occurrence(event)
        run = self._runs.get(event.run_id)
        if occurrence is None or run is None:
            return
        occurrence["state"] = "region"
        occurrence["startMs"] = _rel_ms(self._clock(), run.t0)
        kind = event.detail.get("kind")
        if isinstance(kind, str):
            occurrence["regionKind"] = kind

    def _on_region_finished(self, event: EngineEvent) -> None:
        occurrence = self._occurrence(event)
        run = self._runs.get(event.run_id)
        if occurrence is None or run is None:
            return
        occurrence["state"] = "region"
        occurrence["endMs"] = _rel_ms(self._clock(), run.t0)
        iterations = event.detail.get("iterations")
        if isinstance(iterations, int):
            occurrence["iterations"] = iterations

    def _on_run_finished(self, event: EngineEvent) -> None:
        run = self._runs.pop(event.run_id, None)
        if run is None:
            return
        t1 = self._clock()
        totals: dict[str, object] = {"failed": run.failed}
        for key in ("executed", "cached", "skipped"):
            count = event.detail.get(key)
            if isinstance(count, int):
                totals[key] = count
        record: dict[str, object] = {
            "record": RECORD_VERSION,
            "runId": event.run_id,
            "startedAtUtc": run.started_at_utc,
            "durationMs": _rel_ms(t1, run.t0),
            "environment": dict(self.environment),
            "totals": totals,
            "planned": list(run.planned),
            "occurrences": list(run.occurrences.values()),
        }
        if self._sampler is not None:
            record["hardwareSamples"] = [
                {"tMs": _rel_ms(sample.t, run.t0), **dict(sample.metrics)}
                for sample in self._sampler.samples_between(run.t0, t1)
            ]
        self._on_record(record)

    @staticmethod
    def _pick_identity(occurrence: dict[str, object], event: EngineEvent) -> None:
        """Cache identity + sanitized output shape. The cache key IS the
        comparability key (H4: schema signature + input fingerprints), so
        'same computation, different build/hardware' comparisons are
        principled. Output summaries keep type/length only - the inline
        scalar ``value`` channel never enters a benchmark record."""
        key = event.detail.get("cache_key")
        if isinstance(key, str):
            occurrence["cacheKey"] = key
        outputs = event.detail.get("outputs")
        if isinstance(outputs, Mapping):
            occurrence["outputs"] = {
                str(output_id): {safe: entry[safe] for safe in _SAFE_OUTPUT_KEYS if safe in entry}
                for output_id, entry in outputs.items()
                if isinstance(entry, Mapping)
            }

    # -- boundary diagnostics ------------------------------------------------

    def on_boundary_diagnostic(self, diagnostic: BoundaryDiagnostic) -> None:
        """Correlated by runtime node id: the worker reports before the
        engine emits node_finished for the same invocation, so the pending
        entry is picked up immediately after."""

        def edges(costs: tuple[EdgeCost, ...]) -> list[dict[str, object]]:
            return [
                {
                    "edgeId": edge.edge_id,
                    "typeId": edge.type_id,
                    "transport": edge.transport,
                    "sizeBytes": edge.size_bytes,
                    "codecMs": edge.codec_ms,
                    "declaredCodec": edge.declared_codec,
                    "reused": edge.reused,
                    "networkBytes": edge.network_bytes,
                    "transferMs": edge.transfer_ms,
                }
                for edge in costs
            ]

        self._pending_boundary[diagnostic.node_id] = {
            "pack": diagnostic.pack,
            "executeMs": diagnostic.execute_ms,
            "roundTripMs": diagnostic.round_trip_ms,
            "boundaryMs": round(diagnostic.boundary_ms, 3),
            "inputs": edges(diagnostic.inputs),
            "outputs": edges(diagnostic.outputs),
        }


# ---------------------------------------------------------------------------
# Host wiring helpers
# ---------------------------------------------------------------------------


def instrument_engine_factory(
    make_engine: Callable[[EventListener], Engine],
    assembler: BenchmarkAssembler,
) -> Callable[[EventListener], Engine]:
    """Tee an engine factory's event stream through the assembler first,
    then the host's own listener - observation, never interposition."""

    def factory(on_event: EventListener) -> Engine:
        def tee(event: EngineEvent) -> None:
            assembler.on_engine_event(event)
            on_event(event)

        return make_engine(tee)

    return factory


def record_to_json(record: Mapping[str, object]) -> str:
    """Deterministic serialization (sorted keys, fixed separators): the
    same record always produces the same bytes, so artifacts diff cleanly
    in CI regression gates."""
    return json.dumps(record, sort_keys=True, indent=2, separators=(",", ": "))


def write_record(record: Mapping[str, object], directory: Path | str) -> Path:
    """One versioned JSON artifact per job under ``directory``. Fixed
    output paths are a first-slice pragmatism; routing records through the
    asset/storage abstractions is the tracked follow-up (DESIGN 3.9)."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    started = str(record.get("startedAtUtc", ""))
    stamp = "".join(char for char in started if char.isascii() and char.isalnum())[:15]
    run_id = str(record.get("runId", "unknown"))
    run_digest = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    path = target / f"bench-{stamp or 'unknown'}-{run_digest}.json"
    path.write_text(record_to_json(record) + "\n", encoding="utf-8")
    return path
