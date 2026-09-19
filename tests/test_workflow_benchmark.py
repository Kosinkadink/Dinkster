"""Exercise the family-independent harness, with real HTTP and owned subprocesses."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.test_compare_benchmark_reports import compare_benchmark_reports as comparator
from tools import workflow_benchmark as benchmark
from tools import workflow_benchmark_observer as observer
from tools.evidence_paths import EVIDENCE_ROOT
from tools.workflow_benchmark_observer import (
    ObserverInstallation,
    validate_free_response,
    validate_window,
)
from tools.workflow_benchmark_process import OwnedServer
from tools.workflow_benchmark_report import (
    ALLOCATOR_PEAK_SCOPE,
    comfyui_history_status,
    compare_workflows,
    json_digest,
    seeded_workflow,
    summary,
    validate_workflow_files,
    validate_workflow_report,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "tests/fixtures/lumina2-workflow-api.json"


def workload() -> dict[str, Any]:
    graph = json.loads(WORKFLOW.read_text())
    digest = hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()
    return {
        "graph": graph,
        "api_sha256": digest,
        "graph_sha256": json_digest(graph),
        "seed_inputs": ["48:33.seed"],
        "seeds": list(range(1064, 1070)),
        "provenance": {
            "template_commit": "1" * 40,
            "template_sha256": "2" * 64,
            "template_path": "templates/image_netayume_lumina_t2i.json",
            "api_sha256": digest,
            "exporter": "ComfyUI frontend app.graphToPrompt",
        },
    }


def complete_report(system: str = "dinkster") -> dict[str, Any]:
    specification = workload()
    rows = []
    for index, seed in enumerate(specification["seeds"]):
        rows.append(
            {
                "seed": seed,
                "phase": "cold" if index == 0 else "warm",
                "completed": True,
                "job_id": str(index),
                "start_monotonic_seconds": 10.0,
                "end_monotonic_seconds": 12.0,
                "client_wall_seconds": 2.0,
                "submitted_sha256": json_digest(
                    seeded_workflow(specification["graph"], ["48:33.seed"], seed)
                ),
                "submitted_file": "submitted.json",
                "accepted_file": "accepted.json",
                "history_file": "history.json",
                "outputs": [{"file": "output.png", "bytes": 3, "sha256": "a" * 64}],
            }
        )
    return {
        "report_kind": "workflow_http",
        "report_version": 1,
        "system": system,
        "all_ok": True,
        "errors": [],
        "cleanup_verified": True,
        "runs": rows,
        "sources": {
            name: {"commit": "3" * 40, "tree": "4" * 40, "clean": True}
            for name in ("harness", "dinkster", "comfyui")
        },
        "workload": specification,
        "warm_summary": summary([2.0] * 5),
        "memory": {
            "sampled_peak_tree_rss_bytes": 100,
            "sampled_peak_device_used_bytes": None,
            "allocator_peak_bytes": None,
            "unload_residual_bytes": None,
            "samples_file": "memory.jsonl",
            "sampled_peaks_are_lower_bounds": True,
            "count": 10,
            "errors": [],
            "cleanup_verified": True,
        },
        "artifacts": [
            {
                "name": "model.safetensors",
                "category": "checkpoints",
                "sha256": "a" * 64,
                "bytes": 100,
            }
        ],
        "device": {"kind": "cpu"},
        "machine": {"hostname": "test"},
        "poll_interval_seconds": 0.1,
        "runtime": {"packages": {"pytest": "test"}},
    }


@pytest.mark.parametrize("label", ["lumina2", "another-model", "", None])
def test_family_names_are_diagnostic_not_admission(label: str | None) -> None:
    ours, theirs = complete_report(), complete_report("comfyui")
    ours["family_hint"] = label
    ours["family_observations"] = [{"family": "never-seen-by-this-harness"}]
    assert validate_workflow_report(ours) == ()
    assert comparator.gate_problems(ours, "dinkster") == ()
    assert comparator.comparability_problems(ours, theirs) == ()
    comparison = comparator.build_comparison(ours, theirs)
    assert comparison["cold_client_wall_seconds"]["ratio"] == 1
    assert "workflow_http" in comparator.format_comparison(comparison)


@pytest.mark.parametrize(
    "key,value",
    [
        ("cleanup_verified", False),
        ("all_ok", False),
        ("runs", []),
        ("memory", None),
        ("artifacts", []),
        ("sources", {}),
        ("runtime", None),
        ("report_kind", "unknown"),
        ("poll_interval_seconds", float("nan")),
    ],
)
def test_missing_or_invalid_evidence_cannot_pass(key: str, value: Any) -> None:
    report = complete_report()
    report[key] = value
    assert validate_workflow_report(report)


@pytest.mark.parametrize("field", ["allocator_peak_bytes", "unload_residual_bytes"])
@pytest.mark.parametrize("value", [pytest.param(None, id="missing"), 0, 1])
def test_unmeasured_memory_fields_must_be_present_and_null(field: str, value: int | None) -> None:
    report = complete_report()
    assert report["memory"][field] is None
    assert validate_workflow_report(report) == ()
    if value is None:
        del report["memory"][field]
    else:
        report["memory"][field] = value
    assert validate_workflow_report(report)
    assert comparator.gate_problems(report, "dinkster")


def allocator_records(
    *,
    nonce: str = "a" * 48,
    job_ref: str = "job",
    attempt: int = 1,
    process: str | None = None,
    visible: str = "GPU-test",
    include_invocation: bool = True,
) -> list[dict[str, Any]]:
    process = process or "worker-instance-1"
    common = {
        "source_sha256": "b" * 64,
        "process_instance": process,
        "worker": "pack-a" if process == "worker-instance-1" else "pack-b",
    }
    identity = {
        **common,
        "nonce": nonce,
        "job_ref": job_ref,
        "attempt": attempt,
        "requested_device": {"kind": "cuda", "uuid": "GPU-test"},
        "visible_devices": visible,
        "logical_device": "0",
        "physical_device_uuid": "GPU-test",
        "allocator_error": None,
    }
    counters = {
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
        "allocated_bytes": 50,
        "reserved_bytes": 75,
    }
    records = [
        {**common, "event": "registration_ack"},
        {
            **identity,
            "event": "reset_ack",
            "job_ref": None,
            "attempt": None,
            "counters": dict(counters),
        },
    ]
    if include_invocation:
        records.append(
            {
                **common,
                "event": "invocation_ack",
                "nonce": nonce,
                "job_ref": job_ref,
                "attempt": attempt,
                "invocation_id": "invoke-" + process,
            }
        )
    records.append(
        {
            **identity,
            "event": "read_ack",
            "counters": dict(counters),
        }
    )
    return records


def validate_allocator_records(records: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    arguments = {
        "nonce": "a" * 48,
        "job_ref": "job",
        "attempt": 1,
        "device": {"kind": "cuda", "uuid": "GPU-test"},
        "all_cached": False,
        "source_sha256": "b" * 64,
        **overrides,
    }
    return validate_window(records, **arguments)


def test_allocator_window_accepts_multiple_real_worker_processes() -> None:
    first = allocator_records()
    second = allocator_records(process="worker-instance-2")
    records = [*first, *second]
    window = validate_allocator_records(records)
    assert window["status"] == "acknowledged"
    assert set(window["workers"]) == {"worker-instance-1", "worker-instance-2"}
    assert window["peak_allocated_bytes"] == 200
    assert window["peak_reserved_bytes"] == 400
    assert window["raw_records"] == [row for row in records if row.get("nonce") == "a" * 48]


def test_allocator_window_allows_idle_worker_without_invocation() -> None:
    records = [
        *allocator_records(),
        *allocator_records(process="worker-instance-2", include_invocation=False),
    ]
    window = validate_allocator_records(records)
    assert window["status"] == "acknowledged"
    assert set(window["workers"]) == {"worker-instance-1", "worker-instance-2"}


@pytest.mark.parametrize("event", ["reset_ack", "read_ack"])
@pytest.mark.parametrize(
    "defect", ["allocator-error", "missing-counters", "missing-uuid", "missing-logical"]
)
def test_allocator_window_rejects_one_invalid_worker_ack(event: str, defect: str) -> None:
    records = [*allocator_records(), *allocator_records(process="worker-instance-2")]
    row = next(
        record
        for record in records
        if record.get("event") == event and record.get("process_instance") == "worker-instance-2"
    )
    if defect == "allocator-error":
        row["allocator_error"] = "allocator unavailable"
    elif defect == "missing-counters":
        row["counters"] = None
    elif defect == "missing-uuid":
        row["physical_device_uuid"] = None
    else:
        row["logical_device"] = None
    window = validate_allocator_records(records)
    assert window["status"] == "invalid"
    assert window["problems"]
    assert window["peak_allocated_bytes"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-reset",
        "missing-read",
        "duplicate-read",
        "wrong-job",
        "wrong-attempt",
        "wrong-device",
        "wrong-physical-device",
        "worker-restart",
        "observer-error",
    ],
)
def test_allocator_window_fails_closed_on_misattribution(mutation: str) -> None:
    records = allocator_records()
    if mutation == "missing-reset":
        records.pop(1)
    elif mutation == "missing-read":
        records.pop()
    elif mutation == "duplicate-read":
        records.append(dict(records[-1]))
    elif mutation == "wrong-job":
        records[-1]["job_ref"] = "other"
    elif mutation == "wrong-attempt":
        records[-1]["attempt"] = 2
    elif mutation == "wrong-device":
        records[-1]["requested_device"] = {"kind": "cuda", "uuid": "GPU-other"}
    elif mutation == "wrong-physical-device":
        records[-1]["physical_device_uuid"] = "GPU-other"
    elif mutation == "worker-restart":
        records[-1]["process_instance"] = "100:9:restarted"
    else:
        records.append(
            {
                "event": "read_error",
                "nonce": "a" * 48,
                "process_instance": "100:1:boot",
            }
        )
    window = validate_allocator_records(records)
    assert window["status"] == "invalid"
    assert window["problems"]


def test_cached_window_accepts_current_active_reset_and_terminal_read() -> None:
    records = allocator_records(include_invocation=False)
    window = validate_allocator_records(records, all_cached=True)
    assert window["status"] == "acknowledged"
    assert window["peak_allocated_bytes"] == 100
    missing = validate_allocator_records(records[:-1], all_cached=True)
    assert missing["status"] == "invalid"


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong-job",
        "wrong-attempt",
        "duplicate-id",
        "unregistered-process",
        "missing-noncached",
        "unexpected-cached",
    ],
)
def test_allocator_window_fails_closed_on_invocation_contract(mutation: str) -> None:
    all_cached = mutation == "unexpected-cached"
    records = (
        [*allocator_records(), *allocator_records(process="worker-instance-2")]
        if mutation == "duplicate-id"
        else allocator_records(include_invocation=mutation != "missing-noncached")
    )
    invocations = [row for row in records if row.get("event") == "invocation_ack"]
    if mutation == "wrong-job":
        invocations[0]["job_ref"] = "other"
    elif mutation == "wrong-attempt":
        invocations[0]["attempt"] = 2
    elif mutation == "duplicate-id":
        invocations[1]["invocation_id"] = invocations[0]["invocation_id"]
    elif mutation == "unregistered-process":
        invocations[0]["process_instance"] = "unregistered"
    window = validate_allocator_records(records, all_cached=all_cached)
    assert window["status"] == "invalid"
    assert window["problems"]
    assert window["raw_records"] == [row for row in records if row.get("nonce") == "a" * 48]


def test_allocator_window_uses_actual_attempt_identity() -> None:
    records = allocator_records(attempt=3)
    assert validate_allocator_records(records, attempt=3)["status"] == "acknowledged"
    assert validate_allocator_records(records, attempt=1)["status"] == "invalid"


@pytest.mark.parametrize("system", ["dinkster", "comfyui"])
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_job_timer_starts_before_allocator_reset_and_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str, warm: bool
) -> None:
    events: list[str] = []
    device = {"kind": "cuda", "uuid": "GPU-test"}
    graph = {"node": {"inputs": {"seed": 1}}}
    report = {
        "workload": {"graph": graph, "seed_inputs": ["node.seed"], "seeds": [1]},
        "device": device,
        "runs": [],
    }
    args: Any = SimpleNamespace(
        output=tmp_path,
        system=system,
        gpu_uuid="GPU-test",
        port=1234,
        job_timeout=30,
    )

    class SubmissionReached(RuntimeError):
        pass

    class Server:
        def assert_listener(self, port: int) -> None:
            assert port == args.port
            events.append("listener")

    class Sampler:
        def check(self) -> None:
            pass

    class Observer:
        def begin(self, requested: dict[str, Any], *, require_existing: bool) -> str:
            assert requested == device
            assert require_existing is warm
            events.append("begin")
            return "a" * 48

    def save_event(path: Path, value: Any) -> None:
        del value
        events.append("save:" + path.name)

    def request_event(base: str, path: str, body: Any = None, **kwargs: Any) -> Any:
        del base, kwargs
        if path.endswith("?dryRun=1"):
            events.append("dry-run")
            return {}
        if path == "/dinkster_benchmark/reset":
            events.append("reset")
            return {
                "event": "reset_ack",
                "nonce": body["nonce"],
                "device": device,
                "physical_device_uuid": "GPU-test",
                "process_instance": "comfy-instance",
            }
        events.append("submit")
        raise SubmissionReached

    monkeypatch.setattr(benchmark, "save", save_event)
    monkeypatch.setattr(benchmark, "request", request_event)
    monkeypatch.setattr(benchmark, "output_snapshot", lambda path: set())
    if warm:
        monkeypatch.setattr(
            benchmark, "enumerate", lambda values: [(1, value) for value in values], raising=False
        )
    monkeypatch.setattr(
        benchmark.time,
        "monotonic",
        lambda: events.append("start") or 10.0,
    )
    with pytest.raises(SubmissionReached):
        server: Any = Server()
        sampler: Any = Sampler()
        observer: Any = Observer() if system == "dinkster" else None
        benchmark.run_jobs(
            args,
            report,
            server,
            "http://server",
            sampler,
            observer,
        )

    allocator_event = "begin" if system == "dinkster" else "reset"
    prefix = "1" if warm else "0"
    assert events.index(f"save:{prefix}-submitted.json") < events.index("start")
    if system == "dinkster":
        assert events.index("dry-run") < events.index("start")
    assert events.index("start") < events.index(allocator_event) < events.index("submit")


def test_full_free_reconciles_worker_instances_device_map_and_residue() -> None:
    records = [
        *allocator_records(),
        *allocator_records(process="local", include_invocation=False),
    ]
    for read in [row for row in records if row.get("event") == "read_ack"]:
        records.append(
            {
                **read,
                "event": "residue_ack",
                "counters": {
                    "peak_allocated_bytes": 100,
                    "peak_reserved_bytes": 200,
                    "allocated_bytes": 10,
                    "reserved_bytes": 20,
                },
            }
        )
    for row in records:
        if row.get("process_instance") == "local":
            row["worker"] = "local"
    window = validate_allocator_records(records)
    response = {
        "requestId": "free-1",
        "completed": True,
        "queuePaused": True,
        "workers": [
            {
                "worker": "pack-a",
                "workerInstance": "worker-instance-1",
                "deviceMap": {"mapping": {"cuda:0": "cuda:0"}, "qualifier": None},
                "status": "complete",
                "consumers": [{"consumer": "models", "status": "complete"}],
            },
            {
                "worker": "local",
                "workerInstance": "local",
                "deviceMap": {"mapping": {}, "qualifier": None},
                "status": "complete",
                "consumers": [
                    {"consumer": "execution-cache", "status": "complete"},
                    {"consumer": "comfy-models", "status": "complete"},
                ],
            },
            {
                "worker": "pack-b",
                "workerInstance": "worker-instance-1",
                "deviceMap": {"mapping": {"cuda:0": "cuda:0"}, "qualifier": None},
                "status": "complete",
                "consumers": [],
            },
        ],
    }
    evidence = validate_free_response(
        response,
        records,
        request_id="free-1",
        nonce="a" * 48,
        device={"kind": "cuda", "uuid": "GPU-test"},
        windows=[window],
        source_sha256="b" * 64,
    )
    assert evidence["problems"] == []
    assert evidence["residual_allocated_bytes"] == 20


@pytest.mark.parametrize(
    "defect", ["allocator-error", "missing-counters", "missing-uuid", "wrong-logical"]
)
def test_full_free_rejects_one_invalid_worker_residue(defect: str) -> None:
    records = [*allocator_records(), *allocator_records(process="worker-instance-2")]
    for read in [row for row in records if row.get("event") == "read_ack"]:
        records.append({**read, "event": "residue_ack"})
    bad = next(
        row
        for row in records
        if row.get("event") == "residue_ack" and row.get("process_instance") == "worker-instance-2"
    )
    if defect == "allocator-error":
        bad["allocator_error"] = "allocator unavailable"
    elif defect == "missing-counters":
        bad["counters"] = None
    elif defect == "missing-uuid":
        bad["physical_device_uuid"] = None
    else:
        bad["logical_device"] = "1"
    window = validate_allocator_records(records)
    workers = []
    for instance, worker in (("worker-instance-1", "pack-a"), ("worker-instance-2", "pack-b")):
        workers.append(
            {
                "worker": worker,
                "workerInstance": instance,
                "deviceMap": {"mapping": {}, "qualifier": None},
                "status": "complete",
                "consumers": [{"consumer": "models", "status": "complete"}],
            }
        )
    evidence = validate_free_response(
        {
            "requestId": "free-1",
            "completed": True,
            "queuePaused": True,
            "workers": workers,
        },
        records,
        request_id="free-1",
        nonce="a" * 48,
        device={"kind": "cuda", "uuid": "GPU-test"},
        windows=[window],
        source_sha256="b" * 64,
    )
    assert evidence["problems"]
    assert evidence["residual_allocated_bytes"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "unobserved-worker",
        "restarted-worker",
        "wrong-map",
        "incomplete",
        "worker-error-on-complete",
        "consumer-error-on-complete",
        "duplicate-consumer",
        "missing-worker-name",
        "missing-consumer-name",
        "invalid-map-entry",
        "unpaused",
    ],
)
def test_full_free_fails_closed_on_worker_contract_mismatch(mutation: str) -> None:
    records = allocator_records()
    records.append({**records[-1], "event": "residue_ack"})
    window = validate_allocator_records(records)
    worker = {
        "worker": "pack-a",
        "workerInstance": "worker-instance-1",
        "deviceMap": {"mapping": {"cuda:0": "cuda:0"}, "qualifier": None},
        "status": "complete",
        "consumers": [{"consumer": "models", "status": "complete"}],
    }
    response = {
        "requestId": "free-1",
        "completed": True,
        "queuePaused": True,
        "workers": [worker],
    }
    if mutation == "unobserved-worker":
        response["workers"].append({**worker, "worker": "pack-b", "workerInstance": "missing"})
    elif mutation == "restarted-worker":
        worker["workerInstance"] = "restarted"
    elif mutation == "wrong-map":
        worker["deviceMap"] = {"mapping": {"cuda:0": "cuda:1"}, "qualifier": None}
    elif mutation == "incomplete":
        worker["consumers"][0]["status"] = "failed"
    elif mutation == "worker-error-on-complete":
        worker["error"] = "forbidden"
    elif mutation == "consumer-error-on-complete":
        worker["consumers"][0]["error"] = "forbidden"
    elif mutation == "duplicate-consumer":
        worker["consumers"].append(dict(worker["consumers"][0]))
    elif mutation == "missing-worker-name":
        worker["worker"] = ""
    elif mutation == "missing-consumer-name":
        worker["consumers"][0]["consumer"] = ""
    elif mutation == "invalid-map-entry":
        worker["deviceMap"] = {"mapping": {"": "cuda:0"}, "qualifier": None}
    else:
        response["queuePaused"] = False
    evidence = validate_free_response(
        response,
        records,
        request_id="free-1",
        nonce="a" * 48,
        device={"kind": "cuda", "uuid": "GPU-test"},
        windows=[window],
        source_sha256="b" * 64,
    )
    assert evidence["problems"]


def test_observer_installation_uses_unique_windows_and_removes_startup_hook(tmp_path: Path) -> None:
    installation = ObserverInstallation.create(Path(sys.executable), tmp_path)
    try:
        assert installation.pth_path.is_file()
        assert len(installation.source_sha256) == 64
        subprocess.run(
            [sys.executable, "-c", "pass"],
            env={**os.environ, **installation.environment(), "PYTHONNOUSERSITE": "1"},
            check=True,
            timeout=20,
        )
        loads = installation.records()
        assert len(loads) == 1
        assert loads[0]["event"] == "observer_loaded"
        assert loads[0]["role"] == "unclassified"
        assert loads[0]["process_instance"] is None
        assert loads[0]["source_sha256"] == installation.source_sha256
    finally:
        installation.close()
    assert not installation.pth_path.exists()
    assert not installation.control_path.exists()


def test_observer_close_retries_a_transient_windows_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installation = ObserverInstallation(
        source_sha256="b" * 64,
        control_path=tmp_path / "allocator-control.json",
        records_path=tmp_path / "allocator-records.jsonl",
        pth_path=tmp_path / "observer.pth",
    )
    installation.control_path.write_text("{}")
    installation.pth_path.write_text("")
    original_unlink = Path.unlink
    control_attempts = 0

    def unlink(path: Path, *, missing_ok: bool = False) -> None:
        nonlocal control_attempts
        if path == installation.control_path:
            control_attempts += 1
            if control_attempts < 3:
                raise PermissionError(32, "file is in use", str(path))
        original_unlink(path, missing_ok=missing_ok)

    sleeps: list[float] = []
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(observer.time, "sleep", sleeps.append)

    installation.close()

    assert control_attempts == 3
    assert sleeps == [0.01, 0.01]
    assert not installation.pth_path.exists()
    assert not installation.control_path.exists()


def test_active_control_waits_for_exact_worker_instance_acknowledgements(tmp_path: Path) -> None:
    installation = ObserverInstallation(
        source_sha256="b" * 64,
        control_path=tmp_path / "control.json",
        records_path=tmp_path / "records.jsonl",
        pth_path=tmp_path / "unused.pth",
    )

    def worker() -> None:
        while not installation.control_path.exists():
            time.sleep(0.001)
        command = json.loads(installation.control_path.read_text())
        installation.records_path.write_text(
            json.dumps(
                {
                    "event": "reset_ack",
                    "command_id": command["command_id"],
                    "process_instance": "worker-instance-1",
                }
            )
            + "\n"
        )

    thread = threading.Thread(target=worker)
    thread.start()
    rows = installation.command(
        "reset",
        nonce="a" * 48,
        device={"kind": "cuda", "uuid": "GPU-test"},
        expected_instances={"worker-instance-1"},
        timeout=5,
    )
    thread.join()
    assert [row["process_instance"] for row in rows] == ["worker-instance-1"]


def test_begin_arms_reset_before_a_lazy_worker_registers(tmp_path: Path) -> None:
    installation = ObserverInstallation(
        source_sha256="b" * 64,
        control_path=tmp_path / "control.json",
        records_path=tmp_path / "records.jsonl",
        pth_path=tmp_path / "unused.pth",
    )
    nonce = installation.begin({"kind": "cuda", "uuid": "GPU-test"})
    command = json.loads(installation.control_path.read_text())
    assert len(nonce) == 48
    assert command["nonce"] == nonce
    assert command["operation"] == "reset"
    with pytest.raises(RuntimeError, match="no live registered workers"):
        installation.begin({"kind": "cuda", "uuid": "GPU-test"}, require_existing=True)


def test_live_registrations_use_cross_platform_process_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installation = ObserverInstallation(
        source_sha256="b" * 64,
        control_path=tmp_path / "control.json",
        records_path=tmp_path / "records.jsonl",
        pth_path=tmp_path / "unused.pth",
    )
    installation.records_path.write_text(
        "\n".join(json.dumps({"event": "registration_ack", "pid": pid}) for pid in (123, 456))
        + "\n"
    )
    checked: list[int] = []

    def pid_exists(pid: int) -> bool:
        checked.append(pid)
        return pid == 123

    monkeypatch.setattr(observer.psutil, "pid_exists", pid_exists)

    assert installation.live_registrations() == [{"event": "registration_ack", "pid": 123}]
    assert checked == [123, 456]


def test_worker_hello_uses_product_process_token_and_starts_active_control(tmp_path: Path) -> None:
    installation = ObserverInstallation.create(Path(sys.executable), tmp_path)
    nonce = installation.begin({"kind": "cuda", "uuid": "GPU-test"})
    reset_command = json.loads(installation.control_path.read_text())
    script = """
import asyncio
import json
import time
from dinkster_values import process_instance_token
from dinkster_workers import boundary

class Writer:
    def write(self, data):
        pass

    async def drain(self):
        pass

async def main():
    await boundary.write_frame(
        Writer(),
        {"type": "hello", "pack": "pack-a", "workerInstance": process_instance_token()},
        [],
    )
    reader = asyncio.StreamReader()
    header = {
        "type": "invoke",
        "jobRef": "job",
        "attemptId": 3,
        "invocationId": "invoke-1",
        "blobs": [],
    }
    data = json.dumps(header).encode()
    reader.feed_data(len(data).to_bytes(4, "big") + data)
    reader.feed_eof()
    await boundary.read_frame(reader)

asyncio.run(main())
time.sleep(10)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        env={**os.environ, **installation.environment(), "PYTHONNOUSERSITE": "1"},
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            events = [row.get("event") for row in installation.records()]
            if "registration_ack" in events and "invocation_ack" in events:
                break
            time.sleep(0.01)
        registration = installation.registrations()[0]
        assert registration["process_instance"] == registration["announced_process_instance"]
        assert len(registration["process_instance"]) == 32
        records = installation.records()
        reset_index = next(i for i, row in enumerate(records) if row.get("event") == "reset_ack")
        invocation_index = next(
            i for i, row in enumerate(records) if row.get("event") == "invocation_ack"
        )
        assert reset_index < invocation_index
        assert records[reset_index]["command_id"] == reset_command["command_id"]
        assert records[reset_index]["nonce"] == nonce
        assert records[invocation_index]["attempt"] == 3
        rows = installation.command(
            "read",
            nonce=nonce,
            device={"kind": "cuda", "uuid": "GPU-test"},
            expected_instances={registration["process_instance"]},
            job_ref="job",
            attempt=3,
            timeout=5,
        )
        assert rows[0]["event"] == "read_ack"
        assert rows[0]["job_ref"] == "job"
        assert rows[0]["attempt"] == 3
        assert rows[0]["counters"] is None
        assert rows[0]["allocator_error"]
    finally:
        process.terminate()
        process.wait(timeout=5)
        installation.close()


def test_server_process_registers_local_full_free_worker(tmp_path: Path) -> None:
    installation = ObserverInstallation.create(Path(sys.executable), tmp_path)
    nonce = installation.begin({"kind": "cuda", "uuid": "GPU-test"})
    process = subprocess.Popen(
        [sys.executable, "-c", "import dinkster_server.app, time; time.sleep(10)"],
        env={**os.environ, **installation.environment(), "PYTHONNOUSERSITE": "1"},
    )
    try:
        registration = installation.wait_for_registration("local", timeout=5)
        assert registration["process_instance"] == registration["announced_process_instance"]
        deadline = time.monotonic() + 5
        resets: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            resets = [
                row
                for row in installation.records()
                if row.get("event") == "reset_ack" and row.get("nonce") == nonce
            ]
            if resets:
                break
            time.sleep(0.01)
        assert len(resets) == 1
        assert resets[0]["process_instance"] == registration["process_instance"]
    finally:
        process.terminate()
        process.wait(timeout=5)
        installation.close()


def allocator_report(system: str = "dinkster") -> dict[str, Any]:
    report = complete_report(system)
    report["report_version"] = 2
    report["device"] = {
        "kind": "cuda",
        "uuid": "GPU-test",
        "name": "Test GPU",
        "driver": "test",
    }
    report["memory"].update(
        {
            "sampled_peak_device_used_bytes": 300,
            "allocator_peak_scope": ALLOCATOR_PEAK_SCOPE,
            "allocator_peak_bytes": 100,
            "allocator_peak_reserved_bytes": 200,
            "unload_residual_bytes": 0,
            "unload_residual_reserved_bytes": 16,
        }
    )
    if system == "dinkster":
        report["allocator_observer"] = {
            "source_sha256": "b" * 64,
            "records_file": "allocator-records.jsonl",
        }
    for index, row in enumerate(report["runs"]):
        nonce = f"{index:048x}"
        row["allocator_window_nonce"] = nonce
        row["allocator_file"] = f"{index}-allocator.json"
        if system == "dinkster":
            attempt = index + 1
            row["all_cached"] = False
            records = allocator_records(nonce=nonce, job_ref=row["job_id"], attempt=attempt)
            for record in records:
                if record.get("event") in ("reset_ack", "read_ack"):
                    record["requested_device"] = report["device"]
            row["allocator"] = validate_window(
                records,
                nonce=nonce,
                job_ref=row["job_id"],
                attempt=attempt,
                device=report["device"],
                all_cached=False,
                source_sha256="b" * 64,
            )
        else:
            process = "comfyui-process"
            device = report["device"]
            attempt = index + 1
            row["allocator"] = {
                "nonce": nonce,
                "job_ref": row["job_id"],
                "status": "acknowledged",
                "problems": [],
                "attempt": attempt,
                "peak_allocated_bytes": 100,
                "peak_reserved_bytes": 200,
                "raw_records": [
                    {
                        "event": "reset_ack",
                        "nonce": nonce,
                        "device": device,
                        "physical_device_uuid": device["uuid"],
                        "process_instance": process,
                    },
                    {
                        "event": "read_ack",
                        "nonce": nonce,
                        "job_ref": row["job_id"],
                        "attempt": attempt,
                        "device": device,
                        "logical_device": "cuda:0",
                        "physical_device_uuid": device["uuid"],
                        "process_instance": process,
                        "peak_allocated_bytes": 100,
                        "peak_reserved_bytes": 200,
                        "allocated_bytes": 50,
                        "reserved_bytes": 75,
                    },
                ],
            }
    if system == "comfyui":
        last = report["runs"][-1]
        report["memory"]["unload_record"] = {
            "event": "unload_ack",
            "nonce": last["allocator_window_nonce"],
            "job_ref": last["job_id"],
            "process_instance": "comfyui-process",
            "device": report["device"],
            "residual_allocated_bytes": 0,
            "residual_reserved_bytes": 16,
        }
        return report
    response = {
        "requestId": "free-1",
        "completed": True,
        "queuePaused": True,
        "workers": [
            {
                "worker": "pack-a",
                "workerInstance": "worker-instance-1",
                "status": "complete",
                "deviceMap": {"mapping": {"cuda:0": "cuda:0"}, "qualifier": None},
                "consumers": [],
            }
        ],
    }
    residue = {
        **report["runs"][-1]["allocator"]["raw_records"][-1],
        "event": "residue_ack",
        "job_ref": None,
        "attempt": None,
        "counters": {
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
            "allocated_bytes": 0,
            "reserved_bytes": 16,
        },
    }
    records = [
        *report["runs"][0]["allocator"]["registrations"],
        *(record for row in report["runs"] for record in row["allocator"]["raw_records"]),
        residue,
    ]
    report["memory"]["unload_record"] = validate_free_response(
        response,
        records,
        request_id="free-1",
        nonce=report["runs"][-1]["allocator_window_nonce"],
        device=report["device"],
        windows=[row["allocator"] for row in report["runs"]],
        source_sha256="b" * 64,
    )
    return report


def test_allocator_report_cannot_be_downgraded_to_cpu_schema() -> None:
    report = allocator_report()
    report["report_version"] = 1
    report["memory"]["allocator_peak_bytes"] = None
    report["memory"]["unload_residual_bytes"] = None
    problems = validate_workflow_report(report)
    assert "workflow report version does not match device kind" in problems


def test_cpu_report_cannot_claim_allocator_schema() -> None:
    report = complete_report()
    report["report_version"] = 2
    problems = validate_workflow_report(report)
    assert "workflow report version does not match device kind" in problems


def test_allocator_report_requires_unique_complete_raw_windows() -> None:
    assert validate_workflow_report(allocator_report()) == ()
    for mutation in (
        "duplicate",
        "missing",
        "invalid",
        "forged-raw-records",
        "wrong-source",
        "no-invocation",
        "memory",
        "full-free",
        "full-free-schema",
    ):
        report = allocator_report()
        if mutation == "duplicate":
            report["runs"][1]["allocator_window_nonce"] = report["runs"][0][
                "allocator_window_nonce"
            ]
            report["runs"][1]["allocator"]["nonce"] = report["runs"][0]["allocator_window_nonce"]
        elif mutation == "missing":
            del report["runs"][0]["allocator"]["raw_records"]
        elif mutation == "invalid":
            report["runs"][0]["allocator"]["status"] = "invalid"
        elif mutation == "forged-raw-records":
            report["runs"][0]["allocator"]["raw_records"] = [{"event": "read_ack"}]
        elif mutation == "wrong-source":
            report["runs"][0]["allocator"]["raw_records"][0]["source_sha256"] = "c" * 64
        elif mutation == "no-invocation":
            report["runs"][0]["allocator"]["status"] = "no_invocation"
        elif mutation == "memory":
            report["memory"]["unload_residual_bytes"] = None
        elif mutation == "full-free":
            report["memory"]["unload_record"]["response"]["queuePaused"] = False
        else:
            report["memory"]["unload_record"]["response"]["workers"][0]["error"] = "forbidden"
        assert validate_workflow_report(report)
    ours, theirs = allocator_report(), allocator_report("comfyui")
    comparison = compare_workflows(ours, theirs)
    assert comparison["allocator_peak_bytes"]["ratio"] == 1
    assert comparison["allocator_peak_bytes"]["scope"] == ALLOCATOR_PEAK_SCOPE
    assert comparison["unload_residual_bytes"]["ratio"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "c" * 64),
        ("physical_device_uuid", "GPU-wrong"),
        ("allocator_error", "allocator unavailable"),
        ("counters", None),
    ],
)
def test_dinkster_allocator_report_revalidates_residue_records(field: str, value: Any) -> None:
    report = allocator_report()
    report["memory"]["unload_record"]["residue_records"][0][field] = value
    assert validate_workflow_report(report)


def test_dinkster_allocator_report_rejects_duplicate_full_free_consumers() -> None:
    report = allocator_report()
    report["memory"]["unload_record"]["response"]["workers"][0]["consumers"] = [
        {"consumer": "models", "status": "complete"},
        {"consumer": "models", "status": "complete"},
    ]
    assert validate_workflow_report(report)


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("unload", "event", "wrong"),
        ("unload", "nonce", "f" * 48),
        ("unload", "job_ref", "wrong"),
        ("unload", "process_instance", "wrong"),
        ("unload", "device", {"kind": "cuda", "uuid": "GPU-wrong"}),
        ("unload", "residual_reserved_bytes", 17),
        ("read", "attempt", 999),
        ("read", "physical_device_uuid", "GPU-wrong"),
        ("read", "allocated_bytes", -1),
    ],
)
def test_comfyui_allocator_report_rejects_changed_window_or_unload_identity(
    target: str, field: str, value: Any
) -> None:
    report = allocator_report("comfyui")
    if target == "unload":
        report["memory"]["unload_record"][field] = value
    else:
        report["runs"][-1]["allocator"]["raw_records"][-1][field] = value
    assert validate_workflow_report(report)


@pytest.mark.parametrize("response", [None, {}, {"paused": True}, {"paused": None}])
def test_dinkster_queue_resume_rejects_bad_status(response: Any) -> None:
    assert not benchmark.queue_resumed(response)


def test_dinkster_queue_resume_accepts_explicit_unpaused_status() -> None:
    assert benchmark.queue_resumed({"paused": False})


@pytest.mark.parametrize("response", [None, {}, {"paused": False}, {"paused": None}])
def test_dinkster_queue_pause_rejects_bad_status(response: Any) -> None:
    assert not benchmark.queue_paused(response)


@pytest.mark.parametrize("pause_failure", [RuntimeError("lost response"), {}])
def test_dinkster_failed_pause_acknowledgement_attempts_resume(
    pause_failure: BaseException | dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def request(_base: str, path: str, _body: Any) -> Any:
        events.append(path)
        if path.endswith("pause"):
            if isinstance(pause_failure, BaseException):
                raise pause_failure
            return pause_failure
        return {"paused": False}

    monkeypatch.setattr(benchmark, "request", request)
    with pytest.raises(RuntimeError, match="lost response|pause acknowledgement is invalid"):
        with benchmark.paused_queue("http://server"):
            raise AssertionError("unreachable")
    assert events == ["/api/queue/pause", "/api/queue/resume"]


def test_dinkster_paused_queue_resumes_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []

    def request(_base: str, path: str, _body: Any) -> Any:
        events.append(path)
        return {"paused": path.endswith("pause")}

    monkeypatch.setattr(benchmark, "request", request)
    with benchmark.paused_queue("http://server"):
        events.append("full-free")
    assert events == ["/api/queue/pause", "full-free", "/api/queue/resume"]


def test_dinkster_resume_failure_does_not_mask_full_free_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def request(_base: str, path: str, _body: Any) -> Any:
        return {"paused": True} if path.endswith("pause") else {}

    monkeypatch.setattr(benchmark, "request", request)
    with pytest.raises(ValueError, match="primary full-free failure") as caught:
        with benchmark.paused_queue("http://server"):
            raise ValueError("primary full-free failure")
    assert caught.value.__notes__ == [
        "Dinkster queue resume also failed: RuntimeError: "
        "Dinkster queue resume acknowledgement is invalid"
    ]


def test_journal_replay_uses_terminal_history_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    routes: list[str] = []

    def request(_base: str, route: str) -> dict[str, Any]:
        routes.append(route)
        return {
            "records": [
                {
                    "streamId": "execution-run/job",
                    "seq": 1,
                    "name": "job_state",
                    "payload": {"jobRef": "job", "state": "completed"},
                }
            ],
            "latestSeq": 1,
            "coalescedBelow": 0,
        }

    monkeypatch.setattr(benchmark, "request", request)
    benchmark.collect_journal("http://server", "job", "local", tmp_path / "journal.json")
    assert routes == ["/api/runs/job/journal?scope=local&limit=1000&after=0"]
    with pytest.raises(ValueError, match="valid journal scope"):
        benchmark.collect_journal("http://server", "job", "bad scope", tmp_path / "bad.json")


def test_dinkster_observer_canonicalizes_torch_cuda_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_properties=lambda device: SimpleNamespace(uuid="test"),
        synchronize=lambda device: None,
        reset_peak_memory_stats=lambda device: None,
        max_memory_allocated=lambda device: 100,
        max_memory_reserved=lambda device: 200,
        memory_allocated=lambda device: 50,
        memory_reserved=lambda device: 75,
    )
    torch = SimpleNamespace(cuda=module)
    monkeypatch.setattr(observer.importlib, "import_module", lambda name: torch)
    counters, logical_device, physical_uuid, error = observer._allocator_snapshot(reset=True)
    assert counters == {
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
        "allocated_bytes": 50,
        "reserved_bytes": 75,
    }
    assert logical_device == "0"
    assert physical_uuid == "GPU-test"
    assert error is None


def test_comfyui_allocator_window_uses_real_free_path_without_extra_cache_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_benchmark_comfyui import _load_shim_with_stubs

    harness = _load_shim_with_stubs(monkeypatch)
    module = harness.shim.torch.cuda
    module.memory_reserved = lambda device: 16
    module.max_memory_allocated = lambda device: 100
    module.max_memory_reserved = lambda device: 200
    module.reset_peak_memory_stats = lambda device: None
    module.mem_get_info = lambda device: (900, 1000)
    module.get_device_properties = lambda device: SimpleNamespace(uuid="test")
    module.empty_cache = lambda: harness.events.append("empty_cache")
    harness.shim.gc.collect = lambda: harness.events.append("gc_collect")
    harness.shim.torch._C._cuda_clearCublasWorkspaces = lambda: harness.events.append(
        "clear_workspaces"
    )
    harness.queue.get_history = lambda prompt_id: {
        prompt_id: {"prompt": [7, prompt_id, {}, {}, []]}
    }
    reset = harness.handlers[("POST", "/dinkster_benchmark/reset")]
    memory = harness.handlers[("GET", "/dinkster_benchmark/memory")]
    unload = harness.handlers[("POST", "/dinkster_benchmark/unload")]
    nonce = "a" * 48
    device = {"kind": "cuda", "uuid": "GPU-test"}

    async def scenario() -> tuple[Any, Any, Any]:
        async def body(value: dict[str, Any]) -> dict[str, Any]:
            return value

        reset_response = await reset(
            SimpleNamespace(json=lambda: body({"nonce": nonce, "device": device}))
        )
        harness.shim._EVENTS.append({"prompt_id": "job", "event": "node_finish"})
        memory_response = await memory(
            SimpleNamespace(rel_url=SimpleNamespace(query={"nonce": nonce, "job_ref": "job"}))
        )

        async def prompt_worker() -> None:
            while not harness.queue.get_flags(reset=False):
                await asyncio.sleep(0.01)
            harness.queue.get_flags()
            harness.model_management.soft_empty_cache()
            harness.executor.reset()
            harness.model_management.soft_empty_cache()

        worker = asyncio.create_task(prompt_worker())
        unload_response = await unload(
            SimpleNamespace(json=lambda: body({"nonce": nonce, "job_ref": "job"}))
        )
        await worker
        return reset_response, memory_response, unload_response

    reset_response, memory_response, unload_response = asyncio.run(scenario())
    assert reset_response.payload["event"] == "reset_ack"
    assert memory_response.payload["event"] == "read_ack"
    assert memory_response.payload["attempt"] == 7
    assert memory_response.payload["peak_allocated_bytes"] == 100
    assert memory_response.payload["peak_reserved_bytes"] == 200
    assert unload_response.payload["event"] == "unload_ack"
    assert unload_response.payload["nonce"] == nonce
    assert unload_response.payload["residual_allocated_bytes"] == 0
    assert unload_response.payload["residual_reserved_bytes"] == 16
    assert harness.events.count("soft_empty_cache") == 2
    assert "gc_collect" not in harness.events
    assert "clear_workspaces" not in harness.events
    assert "empty_cache" not in harness.events


def test_comfyui_legacy_unload_preserves_cleanup_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_benchmark_comfyui import _load_shim_with_stubs

    harness = _load_shim_with_stubs(monkeypatch)
    module = harness.shim.torch.cuda
    module.memory_allocated = lambda device: 0
    module.empty_cache = lambda: harness.events.append("empty_cache")
    harness.shim.gc.collect = lambda: harness.events.append("gc_collect")
    harness.shim.torch._C._cuda_clearCublasWorkspaces = lambda: harness.events.append(
        "clear_workspaces"
    )
    unload = harness.handlers[("POST", "/dinkster_benchmark/unload")]

    async def scenario() -> Any:
        async def prompt_worker() -> None:
            while not harness.queue.get_flags(reset=False):
                await asyncio.sleep(0.01)
            harness.queue.get_flags()
            harness.model_management.soft_empty_cache()
            harness.executor.reset()
            harness.model_management.soft_empty_cache()

        worker = asyncio.create_task(prompt_worker())
        response = await unload(
            SimpleNamespace(json=lambda: asyncio.sleep(0, result={"drain_target_bytes": 0}))
        )
        await worker
        return response

    response = asyncio.run(scenario())
    assert response.payload == {"residual_allocated_bytes": 0}
    assert harness.events == [
        "soft_empty_cache",
        "executor_reset",
        "soft_empty_cache",
        "gc_collect",
        "clear_workspaces",
        "empty_cache",
    ]


def test_comfyui_memory_rejects_query_attempt_without_queue_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_benchmark_comfyui import _load_shim_with_stubs

    harness = _load_shim_with_stubs(monkeypatch)
    reset = harness.handlers[("POST", "/dinkster_benchmark/reset")]
    memory = harness.handlers[("GET", "/dinkster_benchmark/memory")]
    harness.shim.torch.cuda.get_device_properties = lambda device: SimpleNamespace(uuid="GPU-test")
    harness.shim.torch.cuda.reset_peak_memory_stats = lambda device: None
    harness.shim.torch.cuda.mem_get_info = lambda device: (900, 1000)
    harness.queue.get_history = lambda prompt_id: {}

    async def scenario() -> Any:
        await reset(
            SimpleNamespace(
                json=lambda: asyncio.sleep(
                    0,
                    result={
                        "nonce": "a" * 48,
                        "device": {"kind": "cuda", "uuid": "GPU-test"},
                    },
                )
            )
        )
        harness.shim._EVENTS.append({"prompt_id": "job", "event": "node_finish"})
        return await memory(
            SimpleNamespace(
                rel_url=SimpleNamespace(
                    query={"nonce": "a" * 48, "job_ref": "job", "attempt": "forged"}
                )
            )
        )

    response = asyncio.run(scenario())
    assert response.status == 409


@pytest.mark.parametrize(
    "path,value",
    [
        (("runs", 1, "submitted_sha256"), "f" * 64),
        (("runs", 1, "job_id"), "0"),
        (("runs", 1, "client_wall_seconds"), float("inf")),
        (("runs", 1, "client_wall_seconds"), True),
        (("runs", 1, "outputs"), {}),
        (("workload", "seeds"), [1, 1]),
        (("workload", "provenance", "api_sha256"), "f" * 64),
        (("sources", "comfyui", "commit"), "short"),
    ],
)
def test_tampered_workload_timing_and_provenance_fail(path: tuple, value: Any) -> None:
    report = complete_report()
    cursor = report
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    assert validate_workflow_report(report)


def test_reference_revision_is_provenance_not_a_fixed_pin() -> None:
    report = complete_report()
    for revision in ("15eb748b3ec5f8a0a2d470b7fb280e2d7579f916", "e" * 40):
        report["sources"]["comfyui"]["commit"] = revision
        assert validate_workflow_report(report) == ()
    assert comparator.comparability_problems(report, complete_report("comfyui"))
    assert comparator.comparability_problems(report, {"system": "comfyui"})


def test_lumina_graph_changes_only_explicit_seed_binding() -> None:
    graph = workload()["graph"]
    original = copy.deepcopy(graph)
    changed = seeded_workflow(graph, ["48:33.seed"], 1064)
    changed["48:33"]["inputs"]["seed"] = graph["48:33"]["inputs"]["seed"]
    assert changed == original == graph
    assert graph["48:33"]["inputs"]["sampler_name"] == "res_multistep"
    for binding in (["missing.seed"], ["48:33.seed", "48:33.seed"], ["48:33.model"]):
        with pytest.raises(ValueError):
            seeded_workflow(graph, binding, 1)


@pytest.mark.parametrize(
    "submitted,observed,accepted",
    [
        (4, 4.0, True),
        (2**53 + 1, 2**53 + 1, True),
        (2**64 - 1, 2**64 - 1, True),
        (True, 1, False),
        (True, 1.0, False),
        (1, True, False),
        ("1", 1, False),
        (4, 5, False),
        (2**53 + 1, float(2**53 + 1), False),
        (2**64 - 1, float(2**64 - 1), False),
        (1.0, float("inf"), False),
        (1.0, float("nan"), False),
        (float("inf"), float("inf"), False),
        ({"value": [1, 2]}, {"value": [1]}, False),
        ({"value": 1}, {"value": 1, "extra": 2}, False),
    ],
)
def test_comfyui_history_graph_comparison_is_typed_and_lossless(
    submitted: Any, observed: Any, accepted: bool
) -> None:
    expected_graph = {"node": {"inputs": {"value": submitted}}}
    observed_graph = {"node": {"inputs": {"value": observed}}}
    expected_before = repr(expected_graph)
    observed_before = repr(observed_graph)
    history = {
        "prompt": [0, "job", observed_graph, {}, []],
        "status": {"completed": True, "status_str": "success"},
    }
    if accepted:
        assert comfyui_history_status(history, "job", expected_graph) == (True, True)
    else:
        with pytest.raises(ValueError, match="history graph differs"):
            comfyui_history_status(history, "job", expected_graph)
    assert repr(expected_graph) == expected_before
    assert repr(observed_graph) == observed_before


def arguments(tmp_path: Path) -> list[str]:
    model = tmp_path / "weights.bin"
    model.write_bytes(b"authenticated test model bytes")
    benchmark.save(
        tmp_path / "artifacts.json",
        [
            {
                "path": str(model),
                "category": "checkpoints",
                "name": "model.bin",
                "sha256": benchmark.file_digest(model),
            }
        ],
    )
    benchmark.save(tmp_path / "provenance.json", workload()["provenance"])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return [
        "--workflow",
        str(WORKFLOW),
        "--workflow-provenance",
        str(tmp_path / "provenance.json"),
        "--artifacts",
        str(tmp_path / "artifacts.json"),
        "--repo",
        str(ROOT),
        "--server-python",
        sys.executable,
        "--reference-root",
        str(ROOT),
        "--reference-commit",
        "e" * 40,
        "--output",
        str(tmp_path / "evidence"),
        "--seed-input",
        "48:33.seed",
        "--family",
        "not-a-registered-family",
        "--port",
        str(port),
        "--cpu",
        "--poll-interval",
        "0.01",
        "--startup-timeout",
        "20",
    ]


@pytest.mark.parametrize("script", ["benchmark_inference.py", "benchmark_comfyui.py"])
@pytest.mark.parametrize("equals", [False, True])
def test_actual_cli_workflow_dispatch_precedes_torch_and_aimdo(script: str, equals: bool) -> None:
    code = """
import importlib.abc, runpy, sys
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('torch', 'dinkster_aimdo', 'dinkster_workers.aimdo_bootstrap')):
            raise AssertionError('execution import during workflow parser: ' + fullname)
sys.meta_path.insert(0, Refuse())
script, *args = sys.argv[1:]
sys.argv = [script, *args]
runpy.run_path(script, run_name='__main__')
"""
    flag = ["--workflow=x"] if equals else ["--workflow", "x"]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(EVIDENCE_ROOT / "scripts" / script),
            *flag,
            "--family",
            "unknown",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--reference-commit" in result.stdout


@pytest.mark.skipif(os.name != "posix", reason="workflow process containment uses POSIX sessions")
@pytest.mark.parametrize("system", ["dinkster", "comfyui"])
@pytest.mark.parametrize(
    "mode", ["success", "failed", "timeout", "lost-response", "no-output", "exit"]
)
def test_real_http_execution_evidence_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str, mode: str
) -> None:
    result, report, directory = run_http_peer(tmp_path, monkeypatch, system, mode)
    assert report["cleanup_verified"]
    assert report["memory"]["cleanup_verified"]
    assert report["server_exit_code"] is not None
    assert result == (0 if mode == "success" else 1), report
    assert report["all_ok"] == (mode == "success"), report
    received = [
        json.loads(line) for line in (directory / "received.jsonl").read_text().splitlines()
    ]
    assert len(received) == (6 if mode == "success" else 1)
    if mode == "success":
        assert validate_workflow_report(report) == ()
        assert validate_workflow_files(report, directory) == ()
        for index, row in enumerate(report["runs"]):
            expected = seeded_workflow(workload()["graph"], ["48:33.seed"], 1064 + index)
            assert received[index]["prompt"] == expected
            submitted = json.loads((directory / row["submitted_file"]).read_text())["prompt"]
            assert submitted == expected
            assert row["submitted_sha256"] == json_digest(expected)
            if system == "comfyui":
                observed = json.loads((directory / row["history_file"]).read_text())["prompt"][2]
                for node, name in (
                    ("48:32", "shift"),
                    ("48:33", "cfg"),
                    ("48:33", "denoise"),
                ):
                    assert type(submitted[node]["inputs"][name]) is int
                    assert type(observed[node]["inputs"][name]) is float
                    assert submitted[node]["inputs"][name] == observed[node]["inputs"][name]
            for output in row["outputs"]:
                assert benchmark.file_digest(directory / output["file"]) == output["sha256"]
        (directory / report["runs"][0]["outputs"][0]["file"]).write_bytes(b"changed output")
        assert "output digest/size differs" in str(validate_workflow_files(report, directory))


def run_http_peer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str, mode: str
) -> tuple[int, dict[str, Any], Path]:
    argv = arguments(tmp_path)
    if mode == "timeout":
        argv += ["--job-timeout", "0.15"]
    actual_command = benchmark.server_command

    def command(args: Any, artifacts: Any) -> list[str]:
        actual_command(args, artifacts)
        return [
            sys.executable,
            str(ROOT / "tests/workflow_benchmark_server.py"),
            system,
            str(args.output),
            str(args.port),
            mode,
        ]

    monkeypatch.setattr(benchmark, "server_command", command)
    monkeypatch.setattr(
        benchmark,
        "source_identity",
        lambda *args: {"commit": "e" * 40, "tree": "f" * 40, "clean": True},
    )
    monkeypatch.setattr(
        benchmark, "runtime_identity", lambda *args: {"packages": {"python": "test"}}
    )
    result = benchmark.main(system, argv)
    directory = tmp_path / "evidence"
    report = json.loads((directory / "report.json").read_text())
    return result, report, directory


@pytest.mark.skipif(os.name != "posix", reason="workflow process containment uses POSIX sessions")
@pytest.mark.parametrize(
    "system,mode,problem",
    [
        ("comfyui", "wrong-comfyui-job", "history belongs to another job"),
        ("comfyui", "wrong-comfyui-graph", "history graph differs"),
        ("comfyui", "missing-comfyui-prompt", "history prompt is missing or malformed"),
        ("comfyui", "malformed-comfyui-prompt", "history prompt is missing or malformed"),
        ("dinkster", "wrong-dinkster-job", "journal job state belongs to another job"),
        ("dinkster", "wrong-dinkster-stream", "journal stream belongs to another job"),
    ],
)
def test_bad_peer_evidence_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    mode: str,
    problem: str,
) -> None:
    result, report, _ = run_http_peer(tmp_path, monkeypatch, system, mode)
    assert result == 1
    assert not report["all_ok"]
    assert report["cleanup_verified"]
    assert any(problem in error for error in report["errors"])


def test_native_server_command_respects_owned_cpu_policy(tmp_path: Path) -> None:
    from dinkster_server.settings import validate_comfy_args

    args = benchmark.parse_arguments("dinkster", arguments(tmp_path))
    command = benchmark.server_command(args, benchmark.authenticate_artifacts(args.artifacts))
    worker_args = tuple(
        arg.removeprefix("--comfy-arg=") for arg in command if arg.startswith("--comfy-arg=")
    )
    assert validate_comfy_args(worker_args) == worker_args
    assert "--cpu" not in worker_args
    assert "--disable-all-custom-nodes" in worker_args
    assert not (args.output / "base/models/checkpoints/model.bin").is_symlink()


def test_comfyui_server_command_loads_only_benchmark_custom_nodes(tmp_path: Path) -> None:
    args = benchmark.parse_arguments("comfyui", arguments(tmp_path))
    command = benchmark.server_command(args, benchmark.authenticate_artifacts(args.artifacts))
    model_paths = json.loads((args.output / "models.yaml").read_text())["benchmark"]
    assert (args.output / "base/custom_nodes").is_dir()
    assert (args.output / "base/models/audio_encoders").is_dir()
    assert (args.output / "base/models/t2i_adapter").is_dir()
    assert (args.output / "images/vae").is_dir()
    assert model_paths["custom_nodes"] == str(ROOT / "scripts/comfyui_benchmark_nodes")
    assert "--disable-all-custom-nodes" in command
    assert command[command.index("--whitelist-custom-nodes") + 1] == "dinkster_benchmark_shim"
    assert "--disable-api-nodes" in command


def test_actual_interpreter_provenance_and_wrong_checkout_refusal(tmp_path: Path) -> None:
    identity = benchmark.runtime_identity(Path(sys.executable), ROOT, dict(os.environ), "dinkster")
    assert identity["packages"]
    assert Path(identity["dinkster_origin"]).is_relative_to(ROOT)
    with pytest.raises(ValueError, match="recorded checkout"):
        benchmark.runtime_identity(Path(sys.executable), tmp_path, dict(os.environ), "dinkster")


def test_source_identity_asserts_supplied_revision_and_clean_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {
        ("rev-parse", "HEAD"): "e" * 40,
        ("rev-parse", "HEAD^{tree}"): "f" * 40,
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
    }
    monkeypatch.setattr(
        subprocess, "check_output", lambda command, **kwargs: responses[tuple(command[3:])]
    )
    assert benchmark.source_identity(ROOT, "e" * 40)["commit"] == "e" * 40
    with pytest.raises(ValueError, match="revision"):
        benchmark.source_identity(ROOT, "d" * 40)
    responses[("status", "--porcelain=v1", "--untracked-files=all")] = " M tracked.py"
    with pytest.raises(ValueError, match="dirty"):
        benchmark.source_identity(ROOT)


def write_retained_workflow_report(directory: Path, system: str) -> tuple[dict[str, Any], Path]:
    directory.mkdir()
    report = complete_report(system)
    for index, row in enumerate(report["runs"]):
        identity = row["job_id"]
        graph = seeded_workflow(
            report["workload"]["graph"], report["workload"]["seed_inputs"], row["seed"]
        )
        row["submitted_file"] = f"{index}-submitted.json"
        row["accepted_file"] = f"{index}-accepted.json"
        row["history_file"] = f"{index}-history.json"
        benchmark.save(directory / row["submitted_file"], {"prompt": graph})
        benchmark.save(
            directory / row["accepted_file"],
            {"prompt_id" if system == "comfyui" else "jobRef": identity},
        )
        if system == "comfyui":
            observed_graph = copy.deepcopy(graph)
            for node, name in (
                ("48:32", "shift"),
                ("48:33", "cfg"),
                ("48:33", "denoise"),
            ):
                observed_graph[node]["inputs"][name] = float(observed_graph[node]["inputs"][name])
            history = {
                "prompt": [0, identity, observed_graph, {}, []],
                "status": {"completed": True, "status_str": "success"},
            }
        else:
            history = {"jobRef": identity, "state": "completed"}
            row["journal_file"] = f"{index}-journal.json"
            benchmark.save(
                directory / row["journal_file"],
                {
                    "records": [
                        {
                            "streamId": f"execution-run/{identity}",
                            "seq": 1,
                            "name": "node_started",
                            "payload": {"nodeId": "test"},
                        },
                        {
                            "streamId": f"execution-run/{identity}",
                            "seq": 2,
                            "name": "job_state",
                            "payload": {"jobRef": identity, "state": "completed"},
                        },
                    ]
                },
            )
        benchmark.save(directory / row["history_file"], history)
        output = directory / f"{index}-output.bin"
        output.write_bytes(f"{system}-{identity}".encode())
        row["outputs"] = [
            {
                "file": output.name,
                "bytes": output.stat().st_size,
                "sha256": benchmark.file_digest(output),
            }
        ]
    (directory / "memory.jsonl").write_text(
        "".join(json.dumps({"tree_rss_bytes": 100, "device": None}) + "\n" for _ in range(10))
    )
    path = directory / "report.json"
    benchmark.save(path, report)
    assert validate_workflow_report(report) == ()
    assert validate_workflow_files(report, directory) == ()
    return report, path


def test_reference_float_normalization_passes_retained_validation_and_cli(tmp_path: Path) -> None:
    _, dinkster_path = write_retained_workflow_report(tmp_path / "dinkster", "dinkster")
    report, comfyui_path = write_retained_workflow_report(tmp_path / "comfyui", "comfyui")
    row = report["runs"][0]
    submitted = json.loads((comfyui_path.parent / row["submitted_file"]).read_text())["prompt"]
    observed = json.loads((comfyui_path.parent / row["history_file"]).read_text())["prompt"][2]
    assert row["submitted_sha256"] == json_digest(submitted)
    for node, name in (
        ("48:32", "shift"),
        ("48:33", "cfg"),
        ("48:33", "denoise"),
    ):
        assert type(submitted[node]["inputs"][name]) is int
        assert type(observed[node]["inputs"][name]) is float
    assert validate_workflow_files(report, comfyui_path.parent) == ()
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/compare_benchmark_reports.py"),
            "--dinkster",
            str(dinkster_path),
            "--comfyui",
            str(comfyui_path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "workflow_http" in result.stdout


@pytest.mark.parametrize(
    "system,corruption",
    [
        ("comfyui", "wrong-job"),
        ("comfyui", "wrong-graph"),
        ("comfyui", "missing-prompt"),
        ("comfyui", "malformed-prompt"),
        ("comfyui", "swapped-histories"),
        ("dinkster", "wrong-job-ref"),
        ("dinkster", "wrong-stream"),
        ("dinkster", "foreign-terminal"),
        ("dinkster", "swapped-journals"),
    ],
)
def test_retained_job_evidence_corruption_is_rejected_by_validator_and_cli(
    tmp_path: Path, system: str, corruption: str
) -> None:
    dinkster, dinkster_path = write_retained_workflow_report(tmp_path / "dinkster", "dinkster")
    comfyui, comfyui_path = write_retained_workflow_report(tmp_path / "comfyui", "comfyui")
    report = comfyui if system == "comfyui" else dinkster
    directory = comfyui_path.parent if system == "comfyui" else dinkster_path.parent
    first = report["runs"][0]
    second = report["runs"][1]
    if corruption == "swapped-histories":
        first_path = directory / first["history_file"]
        second_path = directory / second["history_file"]
        first_bytes, second_bytes = first_path.read_bytes(), second_path.read_bytes()
        first_path.write_bytes(second_bytes)
        second_path.write_bytes(first_bytes)
    elif corruption == "swapped-journals":
        first_path = directory / first["journal_file"]
        second_path = directory / second["journal_file"]
        first_bytes, second_bytes = first_path.read_bytes(), second_path.read_bytes()
        first_path.write_bytes(second_bytes)
        second_path.write_bytes(first_bytes)
    elif system == "comfyui":
        path = directory / first["history_file"]
        history = json.loads(path.read_text())
        if corruption == "wrong-job":
            history["prompt"][1] = "wrong-job"
        elif corruption == "wrong-graph":
            history["prompt"][2] = {"wrong": {"class_type": "Other", "inputs": {}}}
        elif corruption == "missing-prompt":
            del history["prompt"]
        else:
            history["prompt"] = {}
        benchmark.save(path, history)
    else:
        path = directory / first["journal_file"]
        journal = json.loads(path.read_text())
        terminal = next(row for row in journal["records"] if row["name"] == "job_state")
        if corruption == "wrong-job-ref":
            terminal["payload"]["jobRef"] = "wrong-job"
        elif corruption == "wrong-stream":
            terminal["streamId"] = "execution-run/wrong-job"
        else:
            foreign = copy.deepcopy(terminal)
            foreign["seq"] = 3
            foreign["streamId"] = "execution-run/wrong-job"
            foreign["payload"]["jobRef"] = "wrong-job"
            journal["records"].append(foreign)
        benchmark.save(path, journal)
    assert validate_workflow_files(report, directory)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/compare_benchmark_reports.py"),
            "--dinkster",
            str(dinkster_path),
            "--comfyui",
            str(comfyui_path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert "invalid retained workflow evidence" in result.stderr


def test_comparison_cli_rejects_missing_retained_evidence(tmp_path: Path) -> None:
    ours, theirs = tmp_path / "dinkster.json", tmp_path / "comfyui.json"
    benchmark.save(ours, complete_report())
    benchmark.save(theirs, complete_report("comfyui"))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/compare_benchmark_reports.py"),
            "--dinkster",
            str(ours),
            "--comfyui",
            str(theirs),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert "invalid retained workflow evidence" in result.stderr


def test_offline_unknown_family_does_not_start_a_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = arguments(tmp_path)
    monkeypatch.setattr(
        benchmark,
        "source_identity",
        lambda *args: {"commit": "e" * 40, "tree": "f" * 40, "clean": True},
    )
    monkeypatch.setattr(
        benchmark, "OwnedServer", lambda *args: pytest.fail("offline server launch")
    )
    monkeypatch.setattr(benchmark, "NvmlDevice", lambda *args: pytest.fail("offline GPU query"))
    assert benchmark.main("dinkster", [*argv, "--offline-check"]) == 0
    report = json.loads((tmp_path / "evidence/report.json").read_text())
    assert report["preflight_ok"] and not report["all_ok"]
    assert report["sources"]["comfyui"]["commit"] == "e" * 40


def test_input_digest_and_path_escape_refusals(tmp_path: Path) -> None:
    arguments(tmp_path)
    manifest = tmp_path / "artifacts.json"
    entry = json.loads(manifest.read_text())[0]
    for field, value in (("name", "../outside"), ("category", "/absolute"), ("sha256", "0" * 64)):
        benchmark.save(manifest, [{**entry, field: value}])
        with pytest.raises(ValueError):
            benchmark.authenticate_artifacts(manifest)


@pytest.mark.skipif(os.name != "posix", reason="workflow process containment uses POSIX sessions")
@pytest.mark.parametrize("exit_before_inspection", [False, True])
def test_cleanup_includes_orphaned_child_not_foreign_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_before_inspection: bool
) -> None:
    marker = tmp_path / "child.pid"
    code = (
        "import subprocess,sys,time; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(0.2)"
    )
    if exit_before_inspection:
        real_popen = subprocess.Popen

        def launch_then_wait(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
            process = real_popen(*args, **kwargs)
            process.wait(timeout=20)
            return process

        monkeypatch.setattr(subprocess, "Popen", launch_then_wait)
    server = OwnedServer(
        [sys.executable, "-c", code, str(marker)],
        tmp_path,
        dict(os.environ),
        tmp_path / "server.log",
    )
    try:
        server.process.wait(timeout=20)
        child_pid = int(marker.read_text())
        assert child_pid in {p.pid for p in server.members()}
        assert os.getpid() not in {p.pid for p in server.members()}
    finally:
        assert server.stop()
