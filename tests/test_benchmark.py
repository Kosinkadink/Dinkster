"""Native benchmarking, first slice (DESIGN 3.9): records assemble from
the events the engine already emits, sampling is host-owned and bounded,
and records carry identities and costs - never values."""

from __future__ import annotations

import asyncio
import json
from typing import cast

from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent
from dinkster_graph import Graph, GraphNode, Link
from dinkster_schema import build_node_types, build_schemas
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import BoundaryDiagnostic, EdgeCost, InProcessWorker
from scaffold_nodes import SCAFFOLD_NODES, register_scaffold_types

from dinkster.benchmark import (
    RECORD_VERSION,
    BenchmarkAssembler,
    HardwareSampler,
    default_hardware_probe,
    instrument_engine_factory,
    record_to_json,
    write_record,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def collect() -> tuple[list[dict[str, object]], BenchmarkAssembler, FakeClock]:
    records: list[dict[str, object]] = []
    clock = FakeClock()
    assembler = BenchmarkAssembler(records.append, clock=clock)
    return records, assembler, clock


def occurrences(record: dict[str, object]) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", record["occurrences"])


def by_node_id(record: dict[str, object]) -> dict[str, dict[str, object]]:
    return {cast(str, occ["nodeId"]): occ for occ in occurrences(record)}


# -- assembly from a synthetic event stream ---------------------------------


def test_record_assembles_states_timings_and_miss_attribution() -> None:
    records, assembler, clock = collect()
    run = "run-1"
    assembler.on_engine_event(EngineEvent("run_started", run, detail={"planned": ["a", "b", "c"]}))
    clock.now = 100.1
    assembler.on_engine_event(
        EngineEvent(
            "cache_miss",
            run,
            "a",
            {"cache_key": "k-a", "reason": "inputs-changed", "changed_inputs": ["x"]},
        )
    )
    assembler.on_engine_event(EngineEvent("node_started", run, "a"))
    clock.now = 100.3
    assembler.on_engine_event(
        EngineEvent(
            "node_finished",
            run,
            "a",
            {
                "duration_ms": 42.5,
                "cache_key": "k-a",
                "outputs": {"out": {"typeId": "core.int", "value": 7}},
            },
        )
    )
    assembler.on_engine_event(
        EngineEvent(
            "node_cached",
            run,
            "b",
            {
                "cache_key": "k-b",
                "coalesced": True,
                "cacheLayer": "disk",
                "outputs": {"out": {"typeId": "core.list<core.int>", "length": 3}},
            },
        )
    )
    assembler.on_engine_event(
        EngineEvent(
            "node_skipped",
            run,
            "c",
            {"input": "image", "origin": "b", "reason": "no-face-found"},
        )
    )
    clock.now = 100.5
    assembler.on_engine_event(
        EngineEvent("run_finished", run, detail={"executed": 1, "cached": 1, "skipped": 1})
    )

    assert len(records) == 1
    record = records[0]
    assert RECORD_VERSION == "dinkster.benchmark/3"
    assert record["record"] == RECORD_VERSION
    assert record["runId"] == run
    assert record["durationMs"] == 500.0
    assert record["planned"] == ["a", "b", "c"]
    assert record["totals"] == {"executed": 1, "cached": 1, "skipped": 1, "failed": 0}

    by_id = by_node_id(record)
    a = by_id["a"]
    assert a["state"] == "executed"
    assert a["startMs"] == 100.0
    assert a["endMs"] == 300.0
    assert a["durationMs"] == 42.5  # engine-measured, authoritative
    assert a["cacheKey"] == "k-a"
    assert a["miss"] == {"reason": "inputs-changed", "changed_inputs": ["x"]}
    b = by_id["b"]
    assert b["state"] == "cached"
    assert b["coalesced"] is True
    assert b["cacheLayer"] == "disk"
    assert b["outputs"] == {"out": {"typeId": "core.list<core.int>", "length": 3}}
    c = by_id["c"]
    assert c["state"] == "skipped"
    assert c["skip"] == {"input": "image", "origin": "b", "reason": "no-face-found"}


def test_inline_scalar_values_never_enter_a_record() -> None:
    """Privacy rule: the outputs summary's inline ``value`` channel (small
    scalars for frontend badges) is stripped - records carry type/length
    identities, never payloads."""
    records, assembler, _ = collect()
    assembler.on_engine_event(EngineEvent("run_started", "r", detail={"planned": []}))
    assembler.on_engine_event(
        EngineEvent(
            "node_finished",
            "r",
            "n",
            {
                "duration_ms": 1.0,
                "outputs": {
                    "count": {"typeId": "core.int", "value": 42},
                    "name": {"typeId": "core.string", "value": "secret prompt"},
                },
            },
        )
    )
    assembler.on_engine_event(EngineEvent("run_finished", "r", detail={}))
    (record,) = records
    assert "secret prompt" not in json.dumps(record)
    (occurrence,) = occurrences(record)
    assert occurrence["outputs"] == {
        "count": {"typeId": "core.int"},
        "name": {"typeId": "core.string"},
    }


def test_failed_node_and_region_occurrences() -> None:
    records, assembler, _ = collect()
    assembler.on_engine_event(EngineEvent("run_started", "r", detail={"planned": []}))
    assembler.on_engine_event(
        EngineEvent("region_expanded", "r", "loop", {"kind": "map", "iterations": 2})
    )
    assembler.on_engine_event(EngineEvent("node_failed", "r", "loop[1]/n", {"message": "boom"}))
    assembler.on_engine_event(EngineEvent("region_finished", "r", "loop", {"iterations": 2}))
    assembler.on_engine_event(EngineEvent("run_finished", "r", detail={"executed": 0}))
    (record,) = records
    by_id = by_node_id(record)
    assert by_id["loop"]["state"] == "region"
    assert by_id["loop"]["regionKind"] == "map"
    assert by_id["loop"]["iterations"] == 2
    assert by_id["loop[1]/n"]["state"] == "failed"
    assert by_id["loop[1]/n"]["error"] == "boom"
    assert record["totals"] == {"failed": 1, "executed": 0}


def test_boundary_diagnostic_attaches_to_its_occurrence() -> None:
    records, assembler, _ = collect()
    assembler.on_engine_event(EngineEvent("run_started", "r", detail={"planned": []}))
    # The worker reports its breakdown before the engine emits node_finished.
    assembler.on_boundary_diagnostic(
        BoundaryDiagnostic(
            invocation_id="inv-1",
            node_id="n",
            node_type="mypack.foo",
            pack="mypack",
            inputs=(
                EdgeCost(
                    edge_id="image",
                    type_id="mypack.image",
                    transport="shm",
                    size_bytes=1024,
                    codec_ms=0.4,
                    declared_codec=False,
                    reused=False,
                ),
            ),
            outputs=(),
            execute_ms=10.0,
            round_trip_ms=12.5,
        )
    )
    assembler.on_engine_event(EngineEvent("node_finished", "r", "n", {"duration_ms": 12.5}))
    assembler.on_engine_event(EngineEvent("run_finished", "r", detail={"executed": 1}))
    (record,) = records
    (occurrence,) = occurrences(record)
    boundary = cast("dict[str, object]", occurrence["boundary"])
    assert boundary["pack"] == "mypack"
    assert boundary["executeMs"] == 10.0
    assert boundary["boundaryMs"] == 2.5
    assert boundary["inputs"] == [
        {
            "edgeId": "image",
            "typeId": "mypack.image",
            "transport": "shm",
            "sizeBytes": 1024,
            "codecMs": 0.4,
            "declaredCodec": False,
            "reused": False,
            "networkBytes": 0,
            "transferMs": 0.0,
        }
    ]
    # Consumed on attach: nothing pending to mis-attribute to a later run.
    assert assembler._pending_boundary == {}


def test_open_runs_are_bounded() -> None:
    """A failed run never emits run_finished; the assembler must not leak."""
    records, assembler, _ = collect()
    for i in range(200):
        assembler.on_engine_event(EngineEvent("run_started", f"r{i}", detail={"planned": []}))
    assert len(assembler._runs) <= 64
    # The evicted run's late events are ignored, not crashed on.
    assembler.on_engine_event(EngineEvent("node_finished", "r0", "n", {}))
    assembler.on_engine_event(EngineEvent("run_finished", "r0", detail={}))
    assert records == []


# -- hardware sampler ---------------------------------------------------------


def test_default_hardware_probe_reports_effective_system_memory() -> None:
    metrics = default_hardware_probe()
    assert cast(int, metrics["rssBytes"]) > 0
    assert cast(int, metrics["systemTotalBytes"]) > 0
    assert 0 <= cast(int, metrics["systemAvailableBytes"]) <= cast(int, metrics["systemTotalBytes"])
    assert (
        0
        <= cast(int, metrics["systemSwapAvailableBytes"])
        <= cast(int, metrics["systemSwapTotalBytes"])
    )
    provenance = cast(list[str], metrics["systemMemoryProvenance"])
    assert provenance[0] == "psutil"


def test_sampler_is_bounded_and_windows_by_time() -> None:
    clock = FakeClock()
    readings = iter(range(1000))
    sampler = HardwareSampler(probe=lambda: {"rssBytes": next(readings)}, capacity=5, clock=clock)
    for step in range(10):
        clock.now = 100.0 + step
        sampler.sample_once()
    # Bounded: only the newest 5 survive.
    kept = sampler.samples_between(0.0, 1e9)
    assert [s.metrics["rssBytes"] for s in kept] == [5, 6, 7, 8, 9]
    # Windowed: inclusive bounds on the shared monotonic clock.
    window = sampler.samples_between(106.0, 108.0)
    assert [s.metrics["rssBytes"] for s in window] == [6, 7, 8]


def test_failing_probe_records_nothing() -> None:
    def probe() -> dict[str, object]:
        raise RuntimeError("nvml fell over")

    sampler = HardwareSampler(probe=probe)
    sampler.sample_once()
    assert sampler.samples_between(0.0, 1e9) == []


def test_samples_attach_to_records_relative_to_run_start() -> None:
    clock = FakeClock()
    sampler = HardwareSampler(probe=lambda: {"rssBytes": 1}, clock=clock)
    records: list[dict[str, object]] = []
    assembler = BenchmarkAssembler(records.append, sampler=sampler, clock=clock)
    clock.now = 50.0
    sampler.sample_once()  # before the run: outside the window
    clock.now = 100.0
    assembler.on_engine_event(EngineEvent("run_started", "r", detail={"planned": []}))
    clock.now = 100.25
    sampler.sample_once()
    clock.now = 100.5
    assembler.on_engine_event(EngineEvent("run_finished", "r", detail={}))
    (record,) = records
    assert record["hardwareSamples"] == [{"tMs": 250.0, "rssBytes": 1}]


# -- serialization -------------------------------------------------------------


def test_record_json_is_deterministic_and_write_names_stably(tmp_path) -> None:
    record = {
        "record": RECORD_VERSION,
        "runId": "abcdef1234567890",
        "startedAtUtc": "2026-07-20T12:00:00.123456+00:00",
        "b": 1,
        "a": 2,
    }
    first = record_to_json(record)
    second = record_to_json(dict(reversed(list(record.items()))))
    assert first == second  # key order never leaks into the artifact
    path = write_record(record, tmp_path)
    assert path.name == "bench-20260720T120000-840881e18cbe4007.json"
    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_write_record_distinguishes_runs_with_the_same_timestamp_prefix(tmp_path) -> None:
    common = {
        "record": RECORD_VERSION,
        "startedAtUtc": "2026-08-13T09:36:11.123456+00:00",
    }
    first = {**common, "runId": "019ffa7aab9fc610d715b56b0fc08c84"}
    second = {**common, "runId": "019ffa7aab9fb5b5ad48b8ecc7a04ee1"}

    first_path = write_record(first, tmp_path)
    second_path = write_record(second, tmp_path)

    assert first_path != second_path
    assert json.loads(first_path.read_text(encoding="utf-8")) == first
    assert json.loads(second_path.read_text(encoding="utf-8")) == second


def test_write_record_keeps_untrusted_fields_inside_directory(tmp_path) -> None:
    record = {
        "record": RECORD_VERSION,
        "runId": "../../outside",
        "startedAtUtc": "../../also-outside",
    }

    path = write_record(record, tmp_path)

    assert path.parent == tmp_path
    assert json.loads(path.read_text(encoding="utf-8")) == record


# -- end to end through the real engine ----------------------------------------


def _image_graph() -> Graph:
    return Graph(
        nodes={
            "g": GraphNode("dev.image.gradient", {"width": 16, "height": 8}),
            "i": GraphNode("dev.image.invert", {"image": Link("g", "image")}),
            "s": GraphNode("dev.image.stats", {"image": Link("i", "image")}),
        }
    )


def test_end_to_end_records_through_real_engine() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_scaffold_types(registry)
        records: list[dict[str, object]] = []
        assembler = BenchmarkAssembler(records.append)
        seen: list[EngineEvent] = []

        def make_engine(on_event):
            return Engine(
                schemas=build_schemas(SCAFFOLD_NODES),
                registry=registry,
                worker=InProcessWorker(build_node_types(SCAFFOLD_NODES), registry),
                cache=MemoryLRUCache(),
                on_event=on_event,
                explain_misses=True,
            )

        engine = instrument_engine_factory(make_engine, assembler)(seen.append)
        await engine.run(_image_graph(), ["s"])
        await engine.run(_image_graph(), ["s"])

        # Observation, never interposition: the host listener saw everything.
        assert any(ev.kind == "run_finished" for ev in seen)

        assert len(records) == 2
        first, second = records
        first_by_id = by_node_id(first)
        assert set(first_by_id) == {"g", "i", "s"}
        for occurrence in first_by_id.values():
            assert occurrence["state"] == "executed"
            duration = occurrence["durationMs"]
            assert isinstance(duration, float) and duration >= 0.0
            start, end = occurrence["startMs"], occurrence["endMs"]
            assert isinstance(start, float) and isinstance(end, float)
            assert end >= start
            miss = cast("dict[str, object]", occurrence["miss"])
            assert miss["reason"] == "first-seen"
            assert isinstance(occurrence["cacheKey"], str)
        second_by_id = by_node_id(second)
        assert {occ["state"] for occ in second_by_id.values()} == {"cached"}
        # Cache identity is the comparability key: stable across the runs.
        for node_id, occurrence in second_by_id.items():
            assert occurrence["cacheKey"] == first_by_id[node_id]["cacheKey"]
        # The whole artifact is JSON-serializable, deterministically.
        assert record_to_json(first) == record_to_json(json.loads(record_to_json(first)))

    asyncio.run(scenario())
