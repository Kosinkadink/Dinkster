"""Counterbalanced fresh-process ordinary-versus-scheduled GPU evidence.

The committed Blackwell record predates this counterbalanced acceptance method
and is intentionally immutable. Future runs write a distinct schema-2 record.
Each family gets sixteen adjacent fresh-process pairs in the fixed
OS,SO,SO,OS order repeated four times. Synchronized ``perf_counter_ns`` wall
time is the acceptance clock; CUDA events and timestamped telemetry are
diagnostic companions only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
CURRENT_RECORD = (
    REPO / "packages/dinkster-inference-torch/tests/performance/scheduled_runtime_blackwell.json"
)
OUT = REPO / "scheduled_runtime_counterbalanced.json"
ENVELOPE = 0.15
PAIR_ORDERS = ("OS", "SO", "SO", "OS") * 4
PAIR_COUNT = 16
RANK_ONE_BASED = 12
ONE_SIDED_COVERAGE = 0.9615936279296875
FAMILIES = ("flux", "sd")
PHASES = ("warmup", "real")
FAMILY_IDS = {"flux": "dinkster.flux_dev", "sd": "dinkster.sd15"}
GPU_TELEMETRY_FIELDS = (
    "timestamp",
    "pstate",
    "clocks.current.graphics",
    "clocks.current.memory",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
    "utilization.memory",
    "clocks_throttle_reasons.active",
)
GPU_NUMERIC_TELEMETRY_FIELDS = (
    "clocks.current.graphics",
    "clocks.current.memory",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
    "utilization.memory",
)
ENVIRONMENT_FIELDS = ("python", "torch", "cuda", "device_name", "device_uuid", "driver")
RUNTIME_SOURCES = (
    REPO / "packages/dinkster-inference-torch/src/dinkster_inference_torch/scheduled_sampling.py",
    REPO / "packages/dinkster-inference-torch/src/dinkster_inference_torch/wiring.py",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _digest_tensor(value: Any) -> str:
    import torch

    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return _sha256_bytes(raw)


def _gpu_telemetry(device_uuid: str, phase: str, boundary: str) -> dict[str, object]:
    import resource

    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id=GPU-{device_uuid}",
            "--query-gpu=" + ",".join(GPU_TELEMETRY_FIELDS),
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    if len(values) != len(GPU_TELEMETRY_FIELDS) or result.stderr:
        raise RuntimeError("GPU telemetry query returned an incomplete record")
    return {
        "captured_at": _utc_now(),
        "phase": phase,
        "boundary": boundary,
        "gpu": dict(zip(GPU_TELEMETRY_FIELDS, values, strict=True)),
        "host": {
            "load_average": list(os.getloadavg()),
            "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    }


def _numeric_telemetry_values(samples: list[dict[str, object]], field: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        gpu = sample.get("gpu")
        if not isinstance(gpu, dict):
            continue
        value = gpu.get(field)
        if not isinstance(value, str):
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def _telemetry_summary(samples: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"sample_count": len(samples)}
    for field in (
        "clocks.current.graphics",
        "clocks.current.memory",
        "temperature.gpu",
        "power.draw",
        "utilization.gpu",
        "utilization.memory",
    ):
        values = _numeric_telemetry_values(samples, field)
        summary[field] = {"minimum": min(values), "maximum": max(values)} if values else None
    summary["pstates"] = sorted(
        {
            value
            for sample in samples
            if isinstance((gpu := sample.get("gpu")), dict)
            and isinstance((value := gpu.get("pstate")), str)
        }
    )
    summary["active_throttle_reasons"] = sorted(
        {
            value
            for sample in samples
            if isinstance((gpu := sample.get("gpu")), dict)
            and isinstance((value := gpu.get("clocks_throttle_reasons.active")), str)
        }
    )
    return summary


def _worker(engine: str, family: str) -> dict[str, object]:
    import resource

    import torch

    tests = REPO / "packages/dinkster-inference-torch/tests"
    sys.path.insert(0, str(tests))
    from dinkster_inference import FLUX_DEV, SD15, SamplingGuidance
    from test_denoise import tiny_cond as flux_cond
    from test_denoise import tiny_latent as flux_latent
    from test_scheduled_sampling import _conditioning_carrier, _runtime
    from test_sd_denoise import tiny_cond as sd_cond
    from test_sd_denoise import tiny_latent as sd_latent

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("worker requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    runtime = _runtime(family)
    runtime.assembled.diffusion.to(device)
    if family == "flux":
        family_id = FLUX_DEV.id
        latent = flux_latent().to(device)
        cond = flux_cond("performance-cond")
        uncond = flux_cond("performance-uncond")
    else:
        family_id = SD15.id
        latent = sd_latent().to(device)
        cond = sd_cond("performance-cond")
        uncond = sd_cond("performance-uncond")
    cond_carrier = _conditioning_carrier(family_id, cond)
    uncond_carrier = _conditioning_carrier(family_id, uncond)
    properties = torch.cuda.get_device_properties(device)
    device_uuid = str(properties.uuid)
    started_at = _utc_now()
    telemetry: list[dict[str, object]] = []
    phases: dict[str, object] = {}
    for phase in PHASES:
        telemetry.append(_gpu_telemetry(device_uuid, phase, "before"))
        before_ref = len(telemetry) - 1
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        cuda_started = torch.cuda.Event(enable_timing=True)
        cuda_finished = torch.cuda.Event(enable_timing=True)
        cuda_started.record()
        torch.cuda.synchronize(device)
        wall_started = time.perf_counter_ns()
        if engine == "ordinary":
            output = runtime.sample(
                latent,
                cond=cond,
                cfg=SamplingGuidance(uncond, 7.0),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=10,
                seed=4,
                compute_dtype=torch.float32,
                device=device,
            )
        else:
            output = runtime.sample_scheduled(
                latent,
                cond=cond_carrier,
                cfg=SamplingGuidance(uncond_carrier, 7.0),
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=10,
                seed=4,
                compute_dtype=torch.float32,
                device=device,
            )
        torch.cuda.synchronize(device)
        wall_elapsed_ns = time.perf_counter_ns() - wall_started
        cuda_finished.record()
        torch.cuda.synchronize(device)
        cuda_elapsed_ns = round(cuda_started.elapsed_time(cuda_finished) * 1_000_000)
        telemetry.append(_gpu_telemetry(device_uuid, phase, "after"))
        after_ref = len(telemetry) - 1
        phases[phase] = {
            "output_sha256": _digest_tensor(output),
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype).removeprefix("torch."),
            "wall_elapsed_ns": wall_elapsed_ns,
            "cuda_elapsed_ns": cuda_elapsed_ns,
            "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
            "telemetry_refs": [before_ref, after_ref],
        }
    driver = subprocess.run(
        [
            "nvidia-smi",
            f"--id=GPU-{device_uuid}",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    if driver.stderr:
        raise RuntimeError("driver query wrote to stderr")
    return {
        "worker_id": str(uuid.uuid4()),
        "pid": os.getpid(),
        "engine": engine,
        "family": family_id,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": properties.name,
            "device_uuid": device_uuid,
            "driver": driver.stdout.strip(),
            "visible_cuda_device_count": torch.cuda.device_count(),
        },
        "phase_order": list(PHASES),
        "phases": phases,
        "telemetry": {
            "clock_policy": "default-unmodified",
            "raw": telemetry,
            "summary": _telemetry_summary(telemetry),
        },
        "started_at": started_at,
        "finished_at": _utc_now(),
    }


def _run_plan() -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    invocation_index = 0
    for family in FAMILIES:
        for pair_index, order in enumerate(PAIR_ORDERS, start=1):
            for pair_position, letter in enumerate(order, start=1):
                invocation_index += 1
                plan.append(
                    {
                        "invocation_index": invocation_index,
                        "family": family,
                        "pair_index": pair_index,
                        "order": order,
                        "pair_position": pair_position,
                        "engine": "ordinary" if letter == "O" else "scheduled",
                    }
                )
    return plan


def _worker_command(engine: str, family: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        engine,
        "--family",
        family,
    ]


def _run_worker(plan_item: dict[str, object], ordinal: int) -> dict[str, object]:
    engine = str(plan_item["engine"])
    family = str(plan_item["family"])
    command = _worker_command(engine, family)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(ordinal)
    started_at = _utc_now()
    process = subprocess.Popen(
        command,
        cwd=REPO,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    finished_at = _utc_now()
    worker: object | None = None
    parse_error: str | None = None
    if process.returncode == 0 and not stderr:
        try:
            worker = json.loads(stdout)
        except (json.JSONDecodeError, UnicodeError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    return {
        **plan_item,
        "command": command,
        "cuda_visible_devices": str(ordinal),
        "pid": process.pid,
        "return_code": process.returncode,
        "stderr": stderr,
        "stdout_sha256": _sha256_bytes(stdout.encode("utf-8")),
        "parse_error": parse_error,
        "attempt": 1,
        "retry_count": 0,
        "discarded": False,
        "substituted": False,
        "replacement": False,
        "started_at": started_at,
        "finished_at": finished_at,
        "worker": worker,
    }


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _finite_numeric_string(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        number = float(value)
    except ValueError:
        return False
    return math.isfinite(number)


def _sha256_is_valid(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    return all(character in "0123456789abcdef" for character in value.removeprefix("sha256:"))


def _environment_is_complete(value: object) -> bool:
    return (
        isinstance(value, dict)
        and all(isinstance(value.get(field), str) and value[field] for field in ENVIRONMENT_FIELDS)
        and value.get("visible_cuda_device_count") == 1
    )


def _phase_is_complete(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    shape = value.get("output_shape")
    return (
        _sha256_is_valid(value.get("output_sha256"))
        and isinstance(shape, list)
        and bool(shape)
        and all(
            isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0
            for dimension in shape
        )
        and isinstance(value.get("output_dtype"), str)
        and bool(value["output_dtype"])
        and _positive_number(value.get("wall_elapsed_ns"))
        and _positive_number(value.get("cuda_elapsed_ns"))
        and _positive_number(value.get("peak_ram_bytes"))
        and _positive_number(value.get("peak_vram_bytes"))
    )


def _output_signature(value: object) -> tuple[str, tuple[int, ...], str] | None:
    if not _phase_is_complete(value):
        return None
    assert isinstance(value, dict)
    output_sha256 = value["output_sha256"]
    output_shape = value["output_shape"]
    output_dtype = value["output_dtype"]
    assert isinstance(output_sha256, str)
    assert isinstance(output_shape, list)
    assert all(isinstance(dimension, int) for dimension in output_shape)
    assert isinstance(output_dtype, str)
    return output_sha256, tuple(output_shape), output_dtype


def _ratio(candidate: object, baseline: object) -> float | None:
    if not _positive_number(candidate) or not _positive_number(baseline):
        return None
    return float(candidate) / float(baseline)


def _phase_payload(worker: object, phase: str) -> dict[str, object] | None:
    if not isinstance(worker, dict):
        return None
    phases = worker.get("phases")
    if not isinstance(phases, dict):
        return None
    payload = phases.get(phase)
    return payload if isinstance(payload, dict) else None


def _telemetry_is_complete(worker: object) -> bool:
    if not isinstance(worker, dict):
        return False
    telemetry = worker.get("telemetry")
    if not isinstance(telemetry, dict):
        return False
    raw = telemetry.get("raw")
    summary = telemetry.get("summary")
    if (
        telemetry.get("clock_policy") != "default-unmodified"
        or not isinstance(raw, list)
        or len(raw) != 4
        or not isinstance(summary, dict)
        or summary.get("sample_count") != 4
    ):
        return False
    for phase_index, phase in enumerate(PHASES):
        payload = _phase_payload(worker, phase)
        expected_refs = [phase_index * 2, phase_index * 2 + 1]
        if payload is None or payload.get("telemetry_refs") != expected_refs:
            return False
        for ref, boundary in zip(expected_refs, ("before", "after"), strict=True):
            sample = raw[ref]
            gpu = sample.get("gpu") if isinstance(sample, dict) else None
            host = sample.get("host") if isinstance(sample, dict) else None
            load_average = host.get("load_average") if isinstance(host, dict) else None
            if (
                not isinstance(sample, dict)
                or sample.get("phase") != phase
                or sample.get("boundary") != boundary
                or not isinstance(sample.get("captured_at"), str)
                or not isinstance(gpu, dict)
                or not all(isinstance(gpu.get(field), str) for field in GPU_TELEMETRY_FIELDS)
                or not all(str(gpu[field]).strip() for field in GPU_TELEMETRY_FIELDS)
                or not all(
                    _finite_numeric_string(gpu[field]) for field in GPU_NUMERIC_TELEMETRY_FIELDS
                )
                or not isinstance(host, dict)
                or not isinstance(load_average, list)
                or len(load_average) != 3
                or not all(_finite_number(value) for value in load_average)
                or not _positive_number(host.get("process_max_rss_bytes"))
            ):
                return False
    return summary == _telemetry_summary(raw)


def _rank_analysis(entries: list[dict[str, object]]) -> dict[str, object]:
    raw_ratios = [entry["ratio"] for entry in entries]
    sorted_ratios = sorted(float(value) for value in raw_ratios if isinstance(value, float))
    rank_value = sorted_ratios[RANK_ONE_BASED - 1] if len(sorted_ratios) == PAIR_COUNT else None
    strata = {
        order: [entry for entry in entries if entry["order"] == order] for order in ("OS", "SO")
    }
    return {
        "raw_ratios": entries,
        "strata": strata,
        "sorted_ratios": sorted_ratios,
        "rank": {"one_based": RANK_ONE_BASED, "value": rank_value},
        "one_sided_coverage": ONE_SIDED_COVERAGE,
        "within_15_percent": rank_value is not None and rank_value <= 1.0 + ENVELOPE,
    }


def _evaluate_invocations(
    invocations: list[dict[str, object]], *, device_ordinal: int = 0
) -> dict[str, object]:
    expected_plan = _run_plan()
    expected_keys = (
        "invocation_index",
        "family",
        "pair_index",
        "order",
        "pair_position",
        "engine",
    )
    checks: dict[str, bool] = {
        "worker_count_exact": len(invocations) == len(expected_plan),
        "plan_order_adjacency_family_exact": len(invocations) == len(expected_plan)
        and all(
            all(actual.get(key) == expected.get(key) for key in expected_keys)
            for actual, expected in zip(invocations, expected_plan, strict=True)
        ),
        "commands_exact": len(invocations) == len(expected_plan)
        and all(
            actual.get("command")
            == _worker_command(str(expected["engine"]), str(expected["family"]))
            for actual, expected in zip(invocations, expected_plan, strict=True)
        ),
        "return_codes_zero": all(item.get("return_code") == 0 for item in invocations),
        "stderr_empty": all(item.get("stderr") == "" for item in invocations),
        "timestamps_complete": all(
            isinstance(item.get("started_at"), str)
            and bool(item["started_at"])
            and isinstance(item.get("finished_at"), str)
            and bool(item["finished_at"])
            and isinstance((worker := item.get("worker")), dict)
            and isinstance(worker.get("started_at"), str)
            and bool(worker["started_at"])
            and isinstance(worker.get("finished_at"), str)
            and bool(worker["finished_at"])
            for item in invocations
        ),
        "visible_device_ordinal_exact": all(
            item.get("cuda_visible_devices") == str(device_ordinal) for item in invocations
        ),
        "worker_payloads_complete": all(
            item.get("parse_error") is None and isinstance(item.get("worker"), dict)
            for item in invocations
        ),
        "stdout_digests_valid": all(
            _sha256_is_valid(item.get("stdout_sha256")) for item in invocations
        ),
        "no_retry_or_discard": all(
            item.get("attempt") == 1
            and item.get("retry_count") == 0
            and item.get("discarded") is False
            and item.get("substituted") is False
            and item.get("replacement") is False
            for item in invocations
        ),
    }
    parent_pids = [item.get("pid") for item in invocations]
    worker_ids = [
        worker.get("worker_id") if isinstance((worker := item.get("worker")), dict) else None
        for item in invocations
    ]
    worker_pids = [
        worker.get("pid") if isinstance((worker := item.get("worker")), dict) else None
        for item in invocations
    ]
    checks["parent_pids_unique"] = all(
        isinstance(pid, int) and pid > 0 for pid in parent_pids
    ) and len(set(parent_pids)) == len(expected_plan)
    checks["worker_ids_unique"] = all(
        isinstance(identity, str) and identity for identity in worker_ids
    ) and len(set(worker_ids)) == len(expected_plan)
    checks["worker_pids_match_and_unique"] = (
        worker_pids == parent_pids
        and all(isinstance(pid, int) and pid > 0 for pid in worker_pids)
        and len(set(worker_pids)) == len(expected_plan)
    )
    checks["phase_integrity"] = all(
        isinstance((worker := item.get("worker")), dict)
        and worker.get("phase_order") == list(PHASES)
        and isinstance(worker.get("phases"), dict)
        and set(worker["phases"]) == set(PHASES)
        and all(_phase_is_complete(_phase_payload(worker, phase)) for phase in PHASES)
        for item in invocations
    )
    checks["workers_declared_no_failure"] = all(
        isinstance((worker := item.get("worker")), dict) and worker.get("failure") is None
        for item in invocations
    )
    checks["telemetry_complete"] = all(
        _telemetry_is_complete(item.get("worker")) for item in invocations
    )
    all_environments = [
        worker.get("environment") if isinstance((worker := item.get("worker")), dict) else None
        for item in invocations
    ]
    checks["environment_device_exact_all_invocations"] = (
        bool(all_environments)
        and _environment_is_complete(all_environments[0])
        and all(environment == all_environments[0] for environment in all_environments)
    )
    checks["warmup_real_output_exact_each_worker"] = all(
        (warmup := _phase_payload(item.get("worker"), "warmup")) is not None
        and (real := _phase_payload(item.get("worker"), "real")) is not None
        and _phase_is_complete(warmup)
        and _phase_is_complete(real)
        and all(
            warmup.get(key) == real.get(key)
            for key in ("output_sha256", "output_shape", "output_dtype")
        )
        for item in invocations
    )

    timing: dict[str, dict[str, object]] = {}
    resource_pairs: dict[str, list[dict[str, object]]] = {family: [] for family in FAMILIES}
    for family_index, family in enumerate(FAMILIES):
        timing[family] = {}
        family_offset = family_index * PAIR_COUNT * 2
        family_items = invocations[family_offset : family_offset + PAIR_COUNT * 2]
        family_workers = [item.get("worker") for item in family_items]
        family_environments = [
            worker.get("environment") if isinstance(worker, dict) else None
            for worker in family_workers
        ]
        reference_environment = family_environments[0] if family_environments else None
        checks[f"{family}_environment_device_exact_all_workers"] = (
            len(family_items) == PAIR_COUNT * 2
            and _environment_is_complete(reference_environment)
            and all(environment == reference_environment for environment in family_environments)
            and all(
                isinstance(worker, dict) and worker.get("family") == FAMILY_IDS[family]
                for worker in family_workers
            )
        )
        family_output_signatures = [
            _output_signature(_phase_payload(worker, phase))
            for worker in family_workers
            for phase in PHASES
        ]
        reference_output_signature = (
            family_output_signatures[0] if family_output_signatures else None
        )
        checks[f"{family}_output_exact_all_workers"] = (
            len(family_output_signatures) == PAIR_COUNT * 2 * len(PHASES)
            and reference_output_signature is not None
            and all(
                signature == reference_output_signature for signature in family_output_signatures
            )
        )
        phase_ratios: dict[str, list[dict[str, object]]] = {phase: [] for phase in PHASES}
        family_pairs_valid = True
        for pair_zero_index, order in enumerate(PAIR_ORDERS):
            start = family_offset + pair_zero_index * 2
            pair = invocations[start : start + 2]
            if len(pair) != 2:
                family_pairs_valid = False
                continue
            ordinary_item = next((item for item in pair if item.get("engine") == "ordinary"), None)
            scheduled_item = next(
                (item for item in pair if item.get("engine") == "scheduled"), None
            )
            if ordinary_item is None or scheduled_item is None:
                family_pairs_valid = False
                continue
            ordinary_worker = ordinary_item.get("worker")
            scheduled_worker = scheduled_item.get("worker")
            pair_number = pair_zero_index + 1
            expected_family_id = FAMILY_IDS[family]
            identity_exact = (
                isinstance(ordinary_worker, dict)
                and isinstance(scheduled_worker, dict)
                and ordinary_worker.get("engine") == "ordinary"
                and scheduled_worker.get("engine") == "scheduled"
                and ordinary_worker.get("family") == expected_family_id
                and scheduled_worker.get("family") == expected_family_id
                and ordinary_worker.get("environment") == scheduled_worker.get("environment")
            )
            checks[f"{family}_pair_{pair_number}_identity_exact"] = identity_exact
            family_pairs_valid = family_pairs_valid and identity_exact
            resource_entry: dict[str, object] = {
                "pair_index": pair_number,
                "order": order,
                "phases": {},
            }
            for phase in PHASES:
                ordinary_phase = _phase_payload(ordinary_worker, phase)
                scheduled_phase = _phase_payload(scheduled_worker, phase)
                output_exact = (
                    _phase_is_complete(ordinary_phase)
                    and _phase_is_complete(scheduled_phase)
                    and all(
                        ordinary_phase.get(key) == scheduled_phase.get(key)
                        for key in ("output_sha256", "output_shape", "output_dtype")
                    )
                )
                checks[f"{family}_pair_{pair_number}_{phase}_output_exact"] = output_exact
                wall_ratio = (
                    _ratio(
                        scheduled_phase.get("wall_elapsed_ns"),
                        ordinary_phase.get("wall_elapsed_ns"),
                    )
                    if ordinary_phase is not None and scheduled_phase is not None
                    else None
                )
                ram_ratio = (
                    _ratio(
                        scheduled_phase.get("peak_ram_bytes"),
                        ordinary_phase.get("peak_ram_bytes"),
                    )
                    if ordinary_phase is not None and scheduled_phase is not None
                    else None
                )
                vram_ratio = (
                    _ratio(
                        scheduled_phase.get("peak_vram_bytes"),
                        ordinary_phase.get("peak_vram_bytes"),
                    )
                    if ordinary_phase is not None and scheduled_phase is not None
                    else None
                )
                cuda_times_valid = (
                    ordinary_phase is not None
                    and scheduled_phase is not None
                    and _positive_number(ordinary_phase.get("cuda_elapsed_ns"))
                    and _positive_number(scheduled_phase.get("cuda_elapsed_ns"))
                )
                checks[f"{family}_pair_{pair_number}_{phase}_cuda_diagnostic_valid"] = (
                    cuda_times_valid
                )
                checks[f"{family}_pair_{pair_number}_{phase}_ram_within_15_percent"] = (
                    ram_ratio is not None and ram_ratio <= 1.0 + ENVELOPE
                )
                checks[f"{family}_pair_{pair_number}_{phase}_vram_within_15_percent"] = (
                    vram_ratio is not None and vram_ratio <= 1.0 + ENVELOPE
                )
                if wall_ratio is not None:
                    phase_ratios[phase].append(
                        {"pair_index": pair_number, "order": order, "ratio": wall_ratio}
                    )
                phases = resource_entry["phases"]
                assert isinstance(phases, dict)
                phases[phase] = {
                    "ram_ratio": ram_ratio,
                    "vram_ratio": vram_ratio,
                    "ordinary_cuda_elapsed_ns": (
                        ordinary_phase.get("cuda_elapsed_ns")
                        if ordinary_phase is not None
                        else None
                    ),
                    "scheduled_cuda_elapsed_ns": (
                        scheduled_phase.get("cuda_elapsed_ns")
                        if scheduled_phase is not None
                        else None
                    ),
                }
                family_pairs_valid = family_pairs_valid and output_exact
            resource_pairs[family].append(resource_entry)
        checks[f"{family}_all_pair_invariants"] = family_pairs_valid
        for phase in PHASES:
            analysis = _rank_analysis(phase_ratios[phase])
            timing[family][phase] = analysis
            checks[f"{family}_{phase}_sixteen_ratios"] = len(phase_ratios[phase]) == PAIR_COUNT
            checks[f"{family}_{phase}_r12_within_15_percent"] = bool(analysis["within_15_percent"])
    return {
        "timing": timing,
        "resource_pairs": resource_pairs,
        "checks": checks,
        "overall_pass": all(checks.values()),
    }


def _source_hashes() -> dict[str, object]:
    runtime = {str(path.relative_to(REPO)): _sha256_file(path) for path in RUNTIME_SOURCES}
    return {
        "tool_script": _sha256_file(Path(__file__).resolve()),
        "runtime_files": runtime,
        "runtime_combined": _sha256_bytes(b"".join(path.read_bytes() for path in RUNTIME_SOURCES)),
    }


def _build_document(
    invocations: list[dict[str, object]],
    *,
    device_ordinal: int,
    measured_commit: str,
    source_base_commit: str,
    provenance_label: str,
    source_hashes: dict[str, object],
) -> dict[str, object]:
    analysis = _evaluate_invocations(invocations, device_ordinal=device_ordinal)
    return {
        "schema": 2,
        "kind": "counterbalanced-current-hardware-scheduled-runtime-companion",
        "provenance": {
            "measured_commit": measured_commit,
            "source_base_commit": source_base_commit,
            "source_base_label": provenance_label,
            "ordinary_baseline": "adjacent fresh-process ordinary invocation at measured_commit",
        },
        "source_hashes": source_hashes,
        "device_ordinal": device_ordinal,
        "envelope": ENVELOPE,
        "acceptance": {
            "primary_clock": "synchronized perf_counter_ns wall time",
            "pair_orders": list(PAIR_ORDERS),
            "pairs_per_family": PAIR_COUNT,
            "workers_per_family": PAIR_COUNT * 2,
            "rank_one_based": RANK_ONE_BASED,
            "one_sided_coverage": ONE_SIDED_COVERAGE,
            "time_pass_rule": "sorted one-based r(12) <= 1.15 independently per family/phase",
            "cuda_events_are_diagnostic_only": True,
            "telemetry_is_diagnostic_only": True,
            "retry_discard_substitution_replacement_allowed": False,
        },
        "workload": {
            "models": ["deterministic tiny production Flux", "deterministic tiny SD15 UNet"],
            "sampler": "dinkster.euler",
            "scheduler": "dinkster.normal",
            "steps": 10,
            "cfg": 7.0,
            "seed": 4,
            "compute_dtype": "float32",
            "processes": "fresh per invocation; adjacent within each family pair",
            "requests": ["one measured warmup", "one primary real"],
        },
        "telemetry": {
            "clock_policy": "default-unmodified",
            "raw_reference": "invocations[*].worker.telemetry.raw",
            "summary_reference": "invocations[*].worker.telemetry.summary",
            "captured_fields": [
                "clock",
                "temperature",
                "power",
                "pstate",
                "throttle reasons",
                "GPU/memory utilization",
                "host load",
                "process RSS",
            ],
        },
        "invocations": invocations,
        "analysis": {"timing": analysis["timing"], "resource_pairs": analysis["resource_pairs"]},
        "checks": analysis["checks"],
        "overall_pass": analysis["overall_pass"],
    }


def _resolve_commit(value: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--verify", f"{value}^{{commit}}"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _worktree_status() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def _measurement_source_is_stable(
    *,
    measured_commit: str,
    measured_hashes: dict[str, object],
    current_commit: str,
    current_hashes: dict[str, object],
    worktree_status: str,
) -> bool:
    return (
        not worktree_status
        and measured_commit == current_commit
        and measured_hashes == current_hashes
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("ordinary", "scheduled"))
    parser.add_argument("--family", choices=FAMILIES, default="flux")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--base-commit")
    parser.add_argument("--provenance-label")
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(_worker(args.worker, args.family), sort_keys=True))
        return
    if not args.base_commit or not args.provenance_label:
        parser.error("--base-commit and --provenance-label are required for a measurement run")
    if args.output.resolve() == CURRENT_RECORD.resolve():
        parser.error("the committed schema-1 performance record is immutable")
    if _worktree_status():
        parser.error("measurement runs require a clean worktree")
    measured_commit = _resolve_commit("HEAD")
    source_base_commit = _resolve_commit(args.base_commit)
    measured_hashes = _source_hashes()
    invocations = [_run_worker(item, args.device) for item in _run_plan()]
    if not _measurement_source_is_stable(
        measured_commit=measured_commit,
        measured_hashes=measured_hashes,
        current_commit=_resolve_commit("HEAD"),
        current_hashes=_source_hashes(),
        worktree_status=_worktree_status(),
    ):
        raise SystemExit("measurement source changed during the run; no record was written")
    document = _build_document(
        invocations,
        device_ordinal=args.device,
        measured_commit=measured_commit,
        source_base_commit=source_base_commit,
        provenance_label=args.provenance_label,
        source_hashes=measured_hashes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "checks": document["checks"],
                "overall_pass": document["overall_pass"],
            },
            sort_keys=True,
        )
    )
    if not document["overall_pass"]:
        raise SystemExit("scheduled runtime performance/resource proof failed")


if __name__ == "__main__":
    main()
