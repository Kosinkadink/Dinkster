"""Workflow HTTP evidence, separate from the historical per-family cell schema.

These reports measure complete submitted graphs, including saving outputs. GPU
v2 reports require allocator windows; reports do not assert numerical parity,
spill avoidance, or cold disks. Family names are observations, never admission keys.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from tools.workflow_benchmark_observer import validate_free_response, validate_window

REPORT_KIND = "workflow_http"
ALLOCATOR_PEAK_SCOPE = (
    "sum of worker-local window peaks; not a measured simultaneous device-wide peak"
)


def _completed_free_worker(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    device_map = row.get("deviceMap")
    consumers = row.get("consumers")
    mapping = device_map.get("mapping") if isinstance(device_map, dict) else None
    qualifier = device_map.get("qualifier") if isinstance(device_map, dict) else None
    return (
        isinstance(row.get("worker"), str)
        and bool(row["worker"])
        and isinstance(row.get("workerInstance"), str)
        and bool(row["workerInstance"])
        and row.get("status") == "complete"
        and "error" not in row
        and isinstance(mapping, dict)
        and all(
            isinstance(child, str) and bool(child) and isinstance(parent, str) and bool(parent)
            for child, parent in mapping.items()
        )
        and (qualifier is None or isinstance(qualifier, str) and bool(qualifier))
        and isinstance(consumers, list)
        and all(
            isinstance(consumer, dict)
            and isinstance(consumer.get("consumer"), str)
            and bool(consumer["consumer"])
            and consumer.get("status") == "complete"
            and "error" not in consumer
            for consumer in consumers
        )
        and len({consumer["consumer"] for consumer in consumers}) == len(consumers)
    )


def _valid_comfyui_window(row: dict[str, Any], device: Any) -> bool:
    allocator = row.get("allocator")
    records = allocator.get("raw_records") if isinstance(allocator, dict) else None
    if (
        not isinstance(device, dict)
        or not isinstance(records, list)
        or len(records) != 2
        or not all(isinstance(record, dict) for record in records)
    ):
        return False
    reset, read = records
    nonce = row.get("allocator_window_nonce")
    process = reset.get("process_instance")
    attempt = allocator.get("attempt")
    counters = (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "allocated_bytes",
        "reserved_bytes",
    )
    return (
        reset.get("event") == "reset_ack"
        and reset.get("nonce") == nonce
        and reset.get("device") == device
        and reset.get("physical_device_uuid") == device.get("uuid")
        and isinstance(process, str)
        and bool(process)
        and read.get("event") == "read_ack"
        and read.get("nonce") == nonce
        and read.get("job_ref") == row.get("job_id")
        and type(attempt) in (int, float)
        and read.get("attempt") == attempt
        and read.get("process_instance") == process
        and read.get("device") == device
        and isinstance(read.get("logical_device"), str)
        and bool(read["logical_device"])
        and read.get("physical_device_uuid") == device.get("uuid")
        and all(type(read.get(name)) is int and read[name] >= 0 for name in counters)
        and allocator.get("peak_allocated_bytes") == read.get("peak_allocated_bytes")
        and allocator.get("peak_reserved_bytes") == read.get("peak_reserved_bytes")
    )


def _valid_comfyui_unload(memory: dict[str, Any], last: dict[str, Any], device: Any) -> bool:
    unload = memory.get("unload_record")
    allocator = last.get("allocator")
    records = allocator.get("raw_records") if isinstance(allocator, dict) else None
    read = records[-1] if isinstance(records, list) and records else None
    return (
        isinstance(unload, dict)
        and isinstance(read, dict)
        and unload.get("event") == "unload_ack"
        and unload.get("nonce") == last.get("allocator_window_nonce")
        and unload.get("job_ref") == last.get("job_id")
        and unload.get("process_instance") == read.get("process_instance")
        and unload.get("device") == device
        and unload.get("residual_allocated_bytes") == memory.get("unload_residual_bytes")
        and unload.get("residual_reserved_bytes") == memory.get("unload_residual_reserved_bytes")
    )


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def seeded_workflow(graph: dict[str, Any], bindings: list[str], seed: int) -> dict[str, Any]:
    result = copy.deepcopy(graph)
    if not bindings or len(set(bindings)) != len(bindings):
        raise ValueError("seed input bindings must be nonempty and unique")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    for binding in bindings:
        node, separator, name = binding.rpartition(".")
        if not separator or node not in result or name not in result[node].get("inputs", {}):
            raise ValueError(f"missing literal seed input: {binding}")
        if type(result[node]["inputs"][name]) is not int:
            raise ValueError(f"seed input is not a literal integer: {binding}")
        result[node]["inputs"][name] = seed
    return result


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered or any(not math.isfinite(value) or value <= 0 for value in ordered):
        raise ValueError("timing samples must be positive and finite")
    return {
        "median_seconds": statistics.median(ordered),
        "p95_seconds": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "max_seconds": ordered[-1],
    }


def _history_graph_matches(submitted: Any, observed: Any) -> bool:
    if type(submitted) is dict:
        return (
            type(observed) is dict
            and submitted.keys() == observed.keys()
            and all(
                _history_graph_matches(value, observed[key]) for key, value in submitted.items()
            )
        )
    if type(submitted) is list:
        return (
            type(observed) is list
            and len(submitted) == len(observed)
            and all(
                _history_graph_matches(expected, actual)
                for expected, actual in zip(submitted, observed, strict=True)
            )
        )
    if type(submitted) is int and type(observed) is float:
        return math.isfinite(observed) and submitted == observed
    if type(submitted) is not type(observed):
        return False
    if type(submitted) is float:
        return math.isfinite(submitted) and math.isfinite(observed) and submitted == observed
    if type(submitted) in (str, int, bool, type(None)):
        return submitted == observed
    return False


def comfyui_history_status(
    history: dict[str, Any], expected_job_id: str, expected_graph: dict[str, Any]
) -> tuple[bool, bool]:
    prompt = history.get("prompt")
    if not isinstance(prompt, list) or len(prompt) < 3:
        raise ValueError("ComfyUI history prompt is missing or malformed")
    if prompt[1] != expected_job_id:
        raise ValueError("ComfyUI history belongs to another job")
    if not _history_graph_matches(expected_graph, prompt[2]):
        raise ValueError("ComfyUI history graph differs from the submitted graph")
    status = history.get("status")
    if not isinstance(status, dict):
        raise ValueError("ComfyUI history status is missing or malformed")
    done = status.get("completed") is True or status.get("status_str") == "error"
    success = status.get("completed") is True and status.get("status_str") == "success"
    return done, success


def dinkster_journal_completed(records: Any, expected_job_id: str) -> bool:
    if not isinstance(records, list):
        raise ValueError("Dinkster journal records are missing or malformed")
    expected_stream = "execution-run/" + expected_job_id
    completed = False
    for record in records:
        if not isinstance(record, dict) or record.get("name") != "job_state":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("Dinkster job-state payload is missing or malformed")
        if record.get("streamId") != expected_stream:
            raise ValueError("Dinkster job-state journal stream belongs to another job")
        if payload.get("jobRef") != expected_job_id:
            raise ValueError("Dinkster journal job state belongs to another job")
        if payload.get("state") in ("completed", "failed", "cancelled"):
            if payload["state"] != "completed":
                raise ValueError("Dinkster journal records a failed terminal job state")
            completed = True
    return completed


def _hex(value: Any, size: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == size
        and all(c in "0123456789abcdef" for c in value)
    )


def _positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_workload(workload: dict[str, Any]) -> None:
    graph, bindings, seeds = workload["graph"], workload["seed_inputs"], workload["seeds"]
    if not isinstance(graph, dict) or not graph:
        raise ValueError("workflow graph must be nonempty")
    if not isinstance(bindings, list) or not all(isinstance(b, str) for b in bindings):
        raise ValueError("invalid seed inputs")
    if not isinstance(seeds, list) or len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("need a cold seed and distinct warm seeds")
    for seed in seeds:
        seeded_workflow(graph, bindings, seed)
    if not _hex(workload["api_sha256"], 64):
        raise ValueError("invalid workflow export digest")
    if workload["graph_sha256"] != json_digest(graph):
        raise ValueError("workflow graph digest differs")
    provenance = workload["provenance"]
    if (
        provenance.get("api_sha256") != workload["api_sha256"]
        or not _hex(provenance.get("template_commit"), 40)
        or not _hex(provenance.get("template_sha256"), 64)
        or not isinstance(provenance.get("template_path"), str)
        or not provenance["template_path"]
        or not isinstance(provenance.get("exporter"), str)
        or not provenance["exporter"]
    ):
        raise ValueError("workflow export provenance is incomplete")


def validate_workflow_report(report: dict[str, Any]) -> tuple[str, ...]:
    """Validate evidence structure without a family table or a fixed revision."""
    problems: list[str] = []
    if (
        report.get("report_kind") != REPORT_KIND
        or type(report.get("report_version")) is not int
        or report["report_version"] not in (1, 2)
    ):
        problems.append("unsupported workflow report kind/version")
    version = report.get("report_version")
    device = report.get("device")
    if isinstance(device, dict) and (version, device.get("kind")) not in (
        (1, "cpu"),
        (2, "cuda"),
    ):
        problems.append("workflow report version does not match device kind")
    if report.get("system") not in ("dinkster", "comfyui"):
        problems.append("unknown execution system")
    if report.get("all_ok") is not True or report.get("errors") != []:
        problems.append("run failed or is incomplete")
    if report.get("cleanup_verified") is not True:
        problems.append("owned process cleanup is not verified")
    sources = report.get("sources")
    if not isinstance(sources, dict):
        problems.append("missing source provenance")
    else:
        for name in ("harness", "dinkster", "comfyui"):
            source = sources.get(name)
            if (
                not isinstance(source, dict)
                or not _hex(source.get("commit"), 40)
                or not _hex(source.get("tree"), 40)
                or source.get("clean") is not True
            ):
                problems.append(f"invalid {name} source provenance")
    workload = report.get("workload")
    rows = report.get("runs")
    try:
        if not isinstance(workload, dict) or not isinstance(rows, list):
            raise ValueError("missing workload/runs")
        validate_workload(workload)
        graph, bindings, seeds = workload["graph"], workload["seed_inputs"], workload["seeds"]
        if len(rows) != len(seeds):
            raise ValueError("run count differs from workload")
        identities = set()
        window_nonces = set()
        observer_sha256: str | None = None
        dinkster_registrations: list[dict[str, Any]] | None = None
        dinkster_records: list[dict[str, Any]] = []
        validated_dinkster_windows: list[dict[str, Any]] = []
        if report.get("report_version") == 2 and report.get("system") == "dinkster":
            observer = report.get("allocator_observer")
            if not isinstance(observer, dict) or not _hex(observer.get("source_sha256"), 64):
                raise ValueError("Dinkster allocator observer identity is missing")
            observer_sha256 = observer["source_sha256"]
        for index, (row, seed) in enumerate(zip(rows, seeds, strict=True)):
            expected = seeded_workflow(graph, bindings, seed)
            if row["seed"] != seed or row["submitted_sha256"] != json_digest(expected):
                raise ValueError("submitted graph/seed differs from workload")
            if row["phase"] != ("cold" if index == 0 else "warm"):
                raise ValueError("run phase differs")
            if row["job_id"] in identities or not row["job_id"]:
                raise ValueError("job identity is missing or reused")
            identities.add(row["job_id"])
            if report["report_version"] == 2:
                allocator = row.get("allocator")
                nonce = row.get("allocator_window_nonce")
                if (
                    not _hex(nonce, 48)
                    or nonce in window_nonces
                    or not isinstance(allocator, dict)
                    or allocator.get("nonce") != nonce
                    or allocator.get("job_ref") != row["job_id"]
                    or allocator.get("status") != "acknowledged"
                    or allocator.get("problems") != []
                    or not isinstance(allocator.get("raw_records"), list)
                    or not row.get("allocator_file")
                ):
                    raise ValueError("allocator window is missing, stale, duplicate, or invalid")
                if report.get("system") == "dinkster":
                    registrations = allocator.get("registrations")
                    if (
                        type(row.get("all_cached")) is not bool
                        or not isinstance(registrations, list)
                        or not all(isinstance(record, dict) for record in registrations)
                        or not all(isinstance(record, dict) for record in allocator["raw_records"])
                    ):
                        raise ValueError("Dinkster allocator raw evidence is incomplete")
                    registration_records = list(registrations)
                    if dinkster_registrations is None:
                        dinkster_registrations = registration_records
                    elif registration_records != dinkster_registrations:
                        raise ValueError("Dinkster allocator registrations changed between windows")
                    assert observer_sha256 is not None
                    validated = validate_window(
                        [*registration_records, *allocator["raw_records"]],
                        nonce=nonce,
                        job_ref=row["job_id"],
                        attempt=allocator.get("attempt"),
                        device=report.get("device"),
                        all_cached=row["all_cached"],
                        source_sha256=observer_sha256,
                    )
                    if validated != allocator:
                        raise ValueError("Dinkster allocator raw evidence differs from its summary")
                    validated_dinkster_windows.append(validated)
                    dinkster_records.extend(allocator["raw_records"])
                elif not _valid_comfyui_window(row, report.get("device")):
                    raise ValueError("ComfyUI allocator window identity or counters are invalid")
                window_nonces.add(nonce)
            if not _positive(row["client_wall_seconds"]):
                raise ValueError("invalid client wall interval")
            if (
                row["end_monotonic_seconds"] - row["start_monotonic_seconds"]
                != row["client_wall_seconds"]
            ):
                raise ValueError("client wall interval differs from raw clocks")
            if row["completed"] is not True:
                raise ValueError("job did not succeed")
            if not isinstance(row["outputs"], list) or not row["outputs"]:
                raise ValueError("job produced no retained output files")
            for output in row["outputs"]:
                if not _hex(output["sha256"], 64) or not _positive(output["bytes"]):
                    raise ValueError("invalid retained output digest/size")
            if not all(row.get(key) for key in ("submitted_file", "accepted_file", "history_file")):
                raise ValueError("raw job evidence references are missing")
        expected_summary = summary([row["client_wall_seconds"] for row in rows[1:]])
        if report.get("warm_summary") != expected_summary:
            raise ValueError("warm summary differs from raw samples")
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        problems.append(f"invalid workflow evidence: {error}")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        problems.append("missing authenticated input artifacts")
    else:
        keys = set()
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                problems.append("invalid artifact entry")
                continue
            key = (artifact.get("category"), artifact.get("name"))
            if not all(isinstance(part, str) and part for part in key):
                problems.append("invalid artifact category/name")
                continue
            if key in keys:
                problems.append("duplicate input artifact")
            keys.add(key)
            if not _hex(artifact.get("sha256"), 64) or not _positive(artifact.get("bytes")):
                problems.append("invalid artifact digest/size")
    memory = report.get("memory")
    version = report.get("report_version")
    base_memory_invalid = (
        not isinstance(memory, dict)
        or not _positive(memory.get("sampled_peak_tree_rss_bytes"))
        or not memory.get("samples_file")
        or memory.get("sampled_peaks_are_lower_bounds") is not True
        or memory.get("cleanup_verified") is not True
        or memory.get("errors") != []
        or not _positive(memory.get("count"))
    )
    if version == 1:
        allocator_memory_invalid = (
            not isinstance(memory, dict)
            or not {"allocator_peak_bytes", "unload_residual_bytes"}.issubset(memory)
            or memory.get("allocator_peak_bytes") is not None
            or memory.get("unload_residual_bytes") is not None
        )
    else:
        allocator_memory_invalid = (
            not isinstance(memory, dict)
            or memory.get("allocator_peak_scope") != ALLOCATOR_PEAK_SCOPE
            or any(
                type(memory.get(name)) is not int or memory[name] < 0
                for name in (
                    "allocator_peak_bytes",
                    "allocator_peak_reserved_bytes",
                    "unload_residual_bytes",
                    "unload_residual_reserved_bytes",
                )
            )
        )
        if not allocator_memory_invalid and report.get("system") == "comfyui":
            allocator_memory_invalid = (
                not isinstance(rows, list)
                or not rows
                or not _valid_comfyui_unload(memory, rows[-1], report.get("device"))
            )
        elif not allocator_memory_invalid and report.get("system") == "dinkster":
            unload = memory.get("unload_record")
            response = unload.get("response") if isinstance(unload, dict) else None
            request = unload.get("request") if isinstance(unload, dict) else None
            residue = unload.get("residue_records") if isinstance(unload, dict) else None
            workers = response.get("workers") if isinstance(response, dict) else None
            workers_complete = (
                isinstance(workers, list)
                and bool(workers)
                and all(_completed_free_worker(worker) for worker in workers)
            )
            instances = (
                [worker.get("workerInstance") for worker in workers]
                if isinstance(workers, list) and all(isinstance(worker, dict) for worker in workers)
                else []
            )
            residue_instances = (
                [row.get("process_instance") for row in residue]
                if isinstance(residue, list) and all(isinstance(row, dict) for row in residue)
                else []
            )
            valid_instances = bool(instances) and all(
                isinstance(instance, str) and instance for instance in instances
            )
            valid_residue_instances = bool(residue_instances) and all(
                isinstance(instance, str) and instance for instance in residue_instances
            )
            allocator_memory_invalid = (
                not valid_instances
                or not valid_residue_instances
                or not workers_complete
                or unload.get("problems") != []
                or set(instances) != set(residue_instances)
                or len(set(residue_instances)) != len(residue_instances)
                or not isinstance(request, dict)
                or request.get("requestId") != response.get("requestId")
                or response.get("completed") is not True
                or response.get("queuePaused") is not True
                or dinkster_registrations is None
                or observer_sha256 is None
                or not isinstance(rows, list)
                or len(validated_dinkster_windows) != len(rows)
            )
            if not allocator_memory_invalid:
                assert isinstance(unload, dict)
                assert isinstance(response, dict)
                assert isinstance(request, dict)
                assert isinstance(residue, list)
                revalidated = validate_free_response(
                    response,
                    [*dinkster_registrations, *dinkster_records, *residue],
                    request_id=request["requestId"],
                    nonce=rows[-1]["allocator_window_nonce"],
                    device=device,
                    windows=validated_dinkster_windows,
                    source_sha256=observer_sha256,
                )
                allocator_memory_invalid = revalidated != unload
    if base_memory_invalid or allocator_memory_invalid:
        problems.append("missing sampled memory evidence")
    if (
        not isinstance(device, dict)
        or device.get("kind") not in ("cpu", "cuda")
        or not report.get("machine")
    ):
        problems.append("missing machine/device provenance")
    elif device["kind"] == "cuda" and (
        not all(isinstance(device.get(k), str) and device[k] for k in ("uuid", "name", "driver"))
        or not isinstance(memory, dict)
        or not _positive(memory.get("sampled_peak_device_used_bytes"))
    ):
        problems.append("missing observed GPU identity/memory")
    if not isinstance(report.get("runtime"), dict) or not report["runtime"].get("packages"):
        problems.append("missing interpreter/package provenance")
    if not _positive(report.get("poll_interval_seconds")):
        problems.append("missing polling resolution")
    return tuple(problems)


def validate_workflow_files(report: dict[str, Any], directory: Path) -> tuple[str, ...]:
    """Check the retained bytes as well as the summary's self-consistency."""

    def local(name: str) -> Path:
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("evidence path escapes its report directory")
        return path

    try:
        system = report["system"]
        for row in report["runs"]:
            body = json.loads(local(row["submitted_file"]).read_text())
            if json_digest(body["prompt"]) != row["submitted_sha256"]:
                raise ValueError("retained submitted graph differs")
            accepted = json.loads(local(row["accepted_file"]).read_text())
            if accepted["prompt_id" if system == "comfyui" else "jobRef"] != row["job_id"]:
                raise ValueError("retained acceptance identity differs")
            history = json.loads(local(row["history_file"]).read_text())
            if system == "comfyui":
                _, success = comfyui_history_status(history, row["job_id"], body["prompt"])
            else:
                success = (
                    history.get("state") == "completed" and history.get("jobRef") == row["job_id"]
                )
                journal = json.loads(local(row["journal_file"]).read_text())
                if not dinkster_journal_completed(journal.get("records"), row["job_id"]):
                    raise ValueError("terminal journal evidence is missing")
            if not success:
                raise ValueError("retained history is not successful")
            for output in row["outputs"]:
                path = local(output["file"])
                if path.stat().st_size != output["bytes"] or file_digest(path) != output["sha256"]:
                    raise ValueError("retained output digest/size differs")
            if report["report_version"] == 2:
                allocator = json.loads(local(row["allocator_file"]).read_text())
                if allocator != row["allocator"]:
                    raise ValueError("retained allocator window differs")
                if system == "dinkster":
                    all_cached = bool(history.get("nodeStates")) and all(
                        state == "cached" for state in history["nodeStates"].values()
                    )
                    if row.get("all_cached") is not all_cached or allocator.get(
                        "attempt"
                    ) != accepted.get("attemptId"):
                        raise ValueError("retained Dinkster allocator terminal identity differs")
        if system == "dinkster" and report["report_version"] == 2:
            full_free = json.loads(local(report["allocator_full_free_file"]).read_text())
            unload = report["memory"]["unload_record"]
            if full_free != {"request": unload["request"], "response": unload["response"]}:
                raise ValueError("retained full-free response differs")
        count, rss, device_used = 0, 0, None
        with local(report["memory"]["samples_file"]).open() as stream:
            for line in stream:
                sample = json.loads(line)
                count += 1
                rss = max(rss, sample["tree_rss_bytes"])
                if sample["device"] is not None:
                    device_used = max(device_used or 0, sample["device"]["used"])
        memory = report["memory"]
        if (
            count != memory["count"]
            or rss != memory["sampled_peak_tree_rss_bytes"]
            or device_used != memory["sampled_peak_device_used_bytes"]
        ):
            raise ValueError("sampled memory summary differs from retained readings")
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as error:
        return (f"invalid retained workflow evidence: {error}",)
    return ()


def workflow_comparability_problems(
    ours: dict[str, Any], theirs: dict[str, Any]
) -> tuple[str, ...]:
    problems = []
    if ours.get("system") != "dinkster" or theirs.get("system") != "comfyui":
        problems.append("expected Dinkster and ComfyUI arms, respectively")
    for key in (
        "report_kind",
        "report_version",
        "workload",
        "device",
        "machine",
        "poll_interval_seconds",
    ):
        if ours.get(key) != theirs.get(key):
            problems.append(f"{key} differs")
    for name in ("harness", "dinkster", "comfyui"):
        if ours["sources"][name]["commit"] != theirs["sources"][name]["commit"]:
            problems.append(f"{name} comparison revision differs")

    def inputs(report: dict[str, Any]) -> list[tuple[str, str, str, int]]:
        return sorted(
            (a["category"], a["name"], a["sha256"], a["bytes"]) for a in report["artifacts"]
        )

    if inputs(ours) != inputs(theirs):
        problems.append("authenticated inputs differ")
    return tuple(problems)


def compare_workflows(ours: dict[str, Any], theirs: dict[str, Any]) -> dict[str, Any]:
    def pair(a: float, b: float) -> dict[str, Any]:
        return {"dinkster": a, "comfyui": b, "ratio": None if b == 0 else a / b}

    comparison = {
        "report_kind": REPORT_KIND,
        "sources": ours["sources"],
        "workflow_sha256": ours["workload"]["graph_sha256"],
        "workflow_provenance": ours["workload"]["provenance"],
        "device": ours["device"],
        "runtime": {"dinkster": ours["runtime"], "comfyui": theirs["runtime"]},
        "cold_client_wall_seconds": pair(
            ours["runs"][0]["client_wall_seconds"], theirs["runs"][0]["client_wall_seconds"]
        ),
        "warm_client_wall_seconds": {
            key: pair(ours["warm_summary"][key], theirs["warm_summary"][key])
            for key in ours["warm_summary"]
        },
        "sampled_peak_tree_rss_bytes": pair(
            ours["memory"]["sampled_peak_tree_rss_bytes"],
            theirs["memory"]["sampled_peak_tree_rss_bytes"],
        ),
        "sampled_peak_device_used_bytes": {
            "dinkster": ours["memory"].get("sampled_peak_device_used_bytes"),
            "comfyui": theirs["memory"].get("sampled_peak_device_used_bytes"),
            "scope": "NVML device-global; includes unrelated allocations, no ratio",
        },
        "notes": [
            "Ratios are Dinkster / ComfyUI; complete HTTP graphs include output "
            "saving and polling.",
            "Cold means first job in a fresh server, not cold disk/page caches.",
            "Memory peaks are sampled lower bounds; tree RSS includes shared-page double counting.",
            "Output digests are evidence, not a parity verdict; PNG metadata differs.",
            "Selected kernels and spill avoidance are unassessed.",
        ],
    }
    if ours["report_version"] == 2:
        comparison.update(
            {
                "allocator_peak_bytes": pair(
                    ours["memory"]["allocator_peak_bytes"],
                    theirs["memory"]["allocator_peak_bytes"],
                ),
                "allocator_peak_reserved_bytes": pair(
                    ours["memory"]["allocator_peak_reserved_bytes"],
                    theirs["memory"]["allocator_peak_reserved_bytes"],
                ),
                "unload_residual_bytes": pair(
                    ours["memory"]["unload_residual_bytes"],
                    theirs["memory"]["unload_residual_bytes"],
                ),
                "unload_residual_reserved_bytes": pair(
                    ours["memory"]["unload_residual_reserved_bytes"],
                    theirs["memory"]["unload_residual_reserved_bytes"],
                ),
            }
        )
        comparison["allocator_peak_bytes"]["scope"] = ALLOCATOR_PEAK_SCOPE
        comparison["allocator_peak_reserved_bytes"]["scope"] = ALLOCATOR_PEAK_SCOPE
        comparison["notes"].append(
            "Allocator peaks sum worker-local windows and do not measure simultaneous "
            "device-wide concurrency."
        )
    else:
        comparison["notes"].append("Allocator peaks and unload residue are unassessed in v1.")
    return comparison
