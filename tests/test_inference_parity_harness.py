"""Acceptance harness schemas, pins, serialization, and phase gates."""

from __future__ import annotations

import ast
import ctypes
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import time
import zlib
from argparse import Namespace
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, NoReturn, cast

import numpy as np
import pytest
from tools.inference_parity import harness
from tools.inference_parity import nvfp4_flux_dinkster_adapter as nvfp4_adapter
from tools.inference_parity.comfyui_adapter import (
    _normal_sigmas_on_cpu,
    _set_text_precision,
    _set_vae_precision,
)
from tools.inference_parity.harness import (
    TIMING_ENGINE_ORDER,
    TIMING_PHASES,
    HarnessError,
    _adapter_outputs,
    _adapter_reply,
    _adapter_runtime,
    _png_rgba_contract,
    _records_path,
    _reply_integer,
    _timing_correctness,
    _validate_safetensors_contract,
    _write_adapter_logs,
    canonical_bytes,
    compare_records,
    digest_bytes,
    digest_file,
    main,
    phase_requests,
    run_timing,
    run_workload,
    summarize_timing,
    timing_requests,
    validate_acceptance,
    validate_manifest,
    validate_pins,
    validate_timing_history,
    write_json,
)
from tools.inference_parity.ovis_comfyui_adapter import _configure_precision
from tools.inference_parity.ovis_decode_calibration import _regular_decode_without_fallback
from tools.inference_parity.sdxl_edm_vpred_comfyui_adapter import (
    _configure_precision as _configure_sdxl_edm_vpred_precision,
)
from tools.inference_parity.sdxl_vpred_comfyui_adapter import (
    _configure_precision as _configure_sdxl_vpred_precision,
)

from tools.evidence_paths import EVIDENCE_ROOT

CORE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def evidence_working_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    # Harness manifests and adapter commands are relative to the evidence checkout.
    monkeypatch.chdir(EVIDENCE_ROOT)


def test_records_path_resolves_relative_paths_only() -> None:
    root = Path("/external/records")
    assert _records_path(root, Path("workload/verdict.json")) == root / "workload/verdict.json"
    assert _records_path(root, Path("/tmp/verdict.json")) == Path("/tmp/verdict.json")
    assert _records_path(None, Path("workload/verdict.json")) == Path("workload/verdict.json")


def test_records_environment_selects_external_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINKSTER_INFERENCE_PARITY_RECORDS", "/external/evidence/records")
    assert harness._records_default() == Path("/external/evidence/records")


def station_paths(environ: Mapping[str, str] = os.environ) -> dict[str, Path]:
    """Resolve the external ComfyUI prerequisites for adapter integration proofs.

    The defaults preserve the original acceptance-station layout; any other
    machine opts in by exporting DINKSTER_PARITY_COMFYUI_ROOT,
    DINKSTER_PARITY_COMFYUI_PYTHON, and DINKSTER_PARITY_ARTIFACT_ROOT.
    """
    comfyui = Path(environ.get("DINKSTER_PARITY_COMFYUI_ROOT", "/home/kosin/ComfyUI"))
    return {
        "artifact": Path(
            environ.get("DINKSTER_PARITY_ARTIFACT_ROOT", "/home/kosin/ComfyUI-Shared/models")
        ),
        "comfyui": comfyui,
        "python": Path(
            environ.get("DINKSTER_PARITY_COMFYUI_PYTHON", str(comfyui / "venv" / "bin" / "python"))
        ),
    }


def station_available(paths: Mapping[str, Path]) -> bool:
    """True when the torch-bearing ComfyUI interpreter, repo, and artifacts exist."""
    return (
        paths["python"].is_file()
        and (paths["comfyui"] / "main.py").is_file()
        and paths["artifact"].is_dir()
    )


_STATION = station_paths()

requires_comfyui_station = pytest.mark.skipif(
    not station_available(_STATION),
    reason=(
        "external ComfyUI station prerequisites unavailable; set"
        " DINKSTER_PARITY_COMFYUI_ROOT, DINKSTER_PARITY_COMFYUI_PYTHON, and"
        " DINKSTER_PARITY_ARTIFACT_ROOT to enable the adapter construction proofs"
    ),
)


def manifest() -> dict[str, Any]:
    return {
        "schema": 1,
        "workloads": [
            {
                "acceptance_manifest": {"digest": digest_bytes(canonical_bytes(acceptance()))},
                "artifacts": [
                    {"bytes": 1, "digest": "sha256:" + "0" * 64, "path": "m", "role": "model"}
                ],
                "engines": {},
                "execution": {},
                "graph": {},
                "id": "test",
                "purpose": "test",
            }
        ],
    }


def acceptance() -> dict[str, Any]:
    metrics = {
        "end_to_end_ns": "0.1",
        "generation_ns": "0.1",
        "peak_ram_bytes": "0.1",
        "peak_vram_bytes": "0.1",
        "throughput_megapixels_per_second": "0.1",
    }
    return {
        "schema": 1,
        "workloads": {
            "test": {
                "phases": {
                    "warmup": {
                        "metrics": {**metrics, "cold_load_ns": "0.1"},
                        "output": {"max_abs": "0", "mean_abs": "0", "ssim_minimum": "1"},
                    },
                    "real": {
                        "metrics": {**metrics, "warm_generation_ns": "0.1"},
                        "output": {"max_abs": "0", "mean_abs": "0", "ssim_minimum": "1"},
                    },
                }
            }
        },
    }


def record(engine: str, output: Path) -> dict[str, Any]:
    common = {
        "end_to_end_ns": 100,
        "generation_ns": 80,
        "peak_ram_bytes": 1000,
        "peak_vram_bytes": 2000,
        "throughput_megapixels_per_second": "1.0",
    }
    return {
        "acceptance_manifest_digest": digest_bytes(canonical_bytes(acceptance())),
        "engine": {"commit": "a" * 40, "id": engine},
        "warmup": {
            "metrics": {**common, "cold_load_ns": 20},
            "output": {"digest": digest_file(output), "path": str(output)},
            "runtime": {"torch": "2.9.1+cu130"},
        },
        "real": {
            "metrics": {**common, "warm_generation_ns": 80},
            "output": {"digest": digest_file(output), "path": str(output)},
            "runtime": {"torch": "2.9.1+cu130"},
        },
        "workload_id": "test",
    }


def workload(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    applied = acceptance() if policy is None else policy
    return {
        "acceptance_manifest": {"digest": digest_bytes(canonical_bytes(applied))},
        "id": "test",
    }


def timing_hardware_pin() -> dict[str, Any]:
    return {
        "device_ordinal": 0,
        "device_uuid": "GPU-test-0",
        "gpus": [
            {
                "driver": "595.84",
                "index": index,
                "memory_total_mib": 81920,
                "name": "GPU",
                "uuid": f"GPU-test-{index}",
            }
            for index in range(4)
        ],
        "hostname": "RipperPC",
    }


def timing_history(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "hardware": timing_hardware_pin(),
        "policy": {
            "prior_dinkster_warm_warn_ratio": "1.10",
            "regression_status": "WARN",
            "threshold_relation": "strictly-greater-than",
        },
        "records": [] if records is None else records,
        "schema": 1,
    }


def timing_workload() -> dict[str, Any]:
    return {
        "acceptance_manifest": {"digest": digest_bytes(canonical_bytes(acceptance()))},
        "artifacts": [
            {"bytes": 1, "digest": "sha256:" + "0" * 64, "path": "model", "role": "model"}
        ],
        "engines": {
            "comfyui": {"commit": "HEAD", "python": "{root}/python"},
            "dinkster": {"commit": "HEAD", "python": "{root}/python"},
        },
        "execution": {
            "batch": 1,
            "device_ordinal": 0,
            "height": 1,
            "seed": 1,
            "width": 1,
        },
        "graph": {},
        "id": "test",
        "purpose": "timing test",
    }


def timing_process(
    tmp_path: Path,
    engine: str,
    process_index: int,
    *,
    cold_ns: int,
    warm_ns: tuple[int, int],
    output: Path,
) -> dict[str, Any]:
    directory = tmp_path / f"{process_index + 1:02d}-{engine}"
    directory.mkdir(parents=True, exist_ok=True)
    stdout = "protocol\n"
    stderr = ""
    (directory / f"{engine}.stdout.txt").write_bytes(stdout.encode("utf-8"))
    (directory / f"{engine}.stderr.txt").write_bytes(stderr.encode("utf-8"))
    observations = {}
    for phase, elapsed in zip(TIMING_PHASES, (cold_ns, 1, *warm_ns), strict=True):
        observations[phase] = {
            "metrics": {"end_to_end_ns": elapsed},
            "output": {"digest": digest_file(output), "path": str(output)},
            "runtime": {
                "text_parameter_dtype": "float32",
                "torch": "2.13.0+cu130",
            },
        }
    hardware_pin = timing_hardware_pin()
    inventory = {
        "cpu": "cpu",
        "gpus": hardware_pin["gpus"],
        "hostname": hardware_pin["hostname"],
        "platform": "linux",
        "ram_bytes": 1,
    }
    return {
        "engine": {"commit": engine[0] * 40, "id": engine},
        "hardware": {
            "device_ordinal": 0,
            "device_uuid": hardware_pin["device_uuid"],
            "inventory": inventory,
        },
        "harness": {"commit": harness._git_output(CORE_ROOT, "rev-parse", "HEAD")},
        "observations": observations,
        "stderr_digest": digest_bytes(stderr.encode()),
        "stderr_path": f"{engine}.stderr.txt",
        "stdout_digest": digest_bytes(stdout.encode()),
        "stdout_path": f"{engine}.stdout.txt",
    }


def timing_summary(
    tmp_path: Path,
    values: list[tuple[int, tuple[int, int]]],
    history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    processes = [
        timing_process(
            tmp_path,
            engine,
            index,
            cold_ns=cold_ns,
            warm_ns=warm_ns,
            output=output,
        )
        for index, (engine, (cold_ns, warm_ns)) in enumerate(
            zip(TIMING_ENGINE_ORDER, values, strict=True)
        )
    ]
    selected = timing_workload()
    return summarize_timing(
        processes,
        {
            "acceptance_manifest_digest": selected["acceptance_manifest"]["digest"],
            "checks": [{} for _ in range(16)],
            "pass": True,
            "text_parameter_dtype": None,
            "torch": "2.13.0+cu130",
        },
        selected,
        timing_history() if history is None else history,
        {"artifact": tmp_path, "comfyui": tmp_path, "dinkster": CORE_ROOT},
        tmp_path,
        "2026-08-06T00:00:00+00:00",
    )


def test_timing_requests_freeze_four_phase_contract() -> None:
    requests = timing_requests(42)
    assert tuple(request["phase"] for request in requests) == TIMING_PHASES
    assert all(request["seed"] == 42 for request in requests)


def test_timing_hardware_binds_one_designated_gpu_without_requiring_other_gpus_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = timing_workload()
    pin = timing_hardware_pin()
    inventory = {"gpus": pin["gpus"]}
    monkeypatch.setattr(harness.platform, "node", lambda: "RipperPC")
    validated = harness.validate_timing_hardware(selected, inventory, "GPU-test-0", pin)
    assert validated == {
        "device_ordinal": 0,
        "device_uuid": "GPU-test-0",
        "inventory": {**inventory, "hostname": "RipperPC"},
    }
    with pytest.raises(HarnessError, match="CLI GPU UUID"):
        harness.validate_timing_hardware(selected, inventory, "GPU-wrong", pin)
    monkeypatch.setattr(harness.platform, "node", lambda: "other")
    with pytest.raises(HarnessError, match="hostname"):
        harness.validate_timing_hardware(selected, inventory, "GPU-test-0", pin)
    monkeypatch.setattr(harness.platform, "node", lambda: "RipperPC")
    wrong_gpus = {
        **inventory,
        "gpus": [*pin["gpus"][:-1], {**pin["gpus"][-1], "driver": "0"}],
    }
    with pytest.raises(HarnessError, match="GPU inventory"):
        harness.validate_timing_hardware(selected, wrong_gpus, "GPU-test-0", pin)
    selected["execution"]["device_ordinal"] = 1
    with pytest.raises(HarnessError, match="workload device ordinal"):
        harness.validate_timing_hardware(selected, inventory, "GPU-test-0", pin)

    commands: list[list[str]] = []

    def idle_query(command: list[str], **_kwargs: Any) -> Namespace:
        commands.append(command)
        return Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(harness.subprocess, "run", idle_query)
    harness.require_device_idle(0)
    assert commands == [
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    ]


@pytest.mark.parametrize(
    ("returncode", "stdout", "match"),
    [(1, "", "process query failed"), (0, "1234\n", "is not idle")],
)
def test_timing_gpu_isolation_query_fails_closed(
    monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str, match: str
) -> None:
    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *_args, **_kwargs: Namespace(returncode=returncode, stdout=stdout, stderr="query"),
    )
    with pytest.raises(HarnessError, match=match):
        harness.require_device_idle(0)


def test_timing_summary_retains_exact_values_and_first_baseline_is_null(tmp_path: Path) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    processes = [
        timing_process(tmp_path, "comfyui", 0, cold_ns=100, warm_ns=(80, 80), output=output),
        timing_process(tmp_path, "dinkster", 1, cold_ns=90, warm_ns=(70, 70), output=output),
        timing_process(tmp_path, "dinkster", 2, cold_ns=92, warm_ns=(72, 72), output=output),
        timing_process(tmp_path, "comfyui", 3, cold_ns=102, warm_ns=(82, 82), output=output),
    ]
    selected = timing_workload()
    roots = {"artifact": tmp_path, "comfyui": tmp_path, "dinkster": CORE_ROOT}
    correctness = {
        "acceptance_manifest_digest": selected["acceptance_manifest"]["digest"],
        "checks": [{} for _ in range(16)],
        "pass": True,
        "text_parameter_dtype": None,
        "torch": "2.13.0+cu130",
    }

    summary = summarize_timing(
        processes,
        correctness,
        selected,
        timing_history(),
        roots,
        tmp_path,
        "2026-08-06T00:00:00+00:00",
    )

    assert summary["harness"]["job_count"] == 16
    assert summary["harness"]["process_count"] == 4
    assert summary["harness"]["process_order"] == list(TIMING_ENGINE_ORDER)
    assert summary["harness"]["requests_per_process"] == list(TIMING_PHASES)
    assert summary["harness"]["external_clock"] == "perf_counter_ns request-to-reply wall time"
    assert summary["engines"]["comfyui"]["cold_ns"] == [100, 102]
    assert summary["engines"]["comfyui"]["warm_ns"] == [80, 80, 82, 82]
    assert summary["engines"]["comfyui"]["warm_median_ns"] == 81.0
    assert summary["engines"]["dinkster"]["cold_ns"] == [90, 92]
    assert summary["engines"]["dinkster"]["warm_ns"] == [70, 70, 72, 72]
    assert summary["engines"]["dinkster"]["warm_median_ns"] == 71.0
    assert summary["ratios"]["dinkster_over_comfyui_cold"] == format(91.0 / 101.0, ".9f")
    assert summary["ratios"]["dinkster_over_comfyui_warm"] == format(71.0 / 81.0, ".9f")
    assert summary["regression"] == {
        "equality_passes": True,
        "prior_dinkster_warm_median_ns": None,
        "prior_dinkster_warm_ratio": None,
        "prior_selection": "no-history",
        "status": "BASELINE",
        "warn_if_strictly_greater_than": "1.10",
    }
    assert summary["status"] == "BASELINE"


@pytest.mark.parametrize(
    ("dinkster_warm_ns", "expected_status"),
    [(110, "PASS"), (111, "WARN")],
)
def test_timing_regression_threshold_is_strictly_greater_than(
    tmp_path: Path, dinkster_warm_ns: int, expected_status: str
) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    processes = [
        timing_process(tmp_path, "comfyui", 0, cold_ns=100, warm_ns=(100, 100), output=output),
        timing_process(
            tmp_path,
            "dinkster",
            1,
            cold_ns=100,
            warm_ns=(dinkster_warm_ns, dinkster_warm_ns),
            output=output,
        ),
        timing_process(
            tmp_path,
            "dinkster",
            2,
            cold_ns=100,
            warm_ns=(dinkster_warm_ns, dinkster_warm_ns),
            output=output,
        ),
        timing_process(tmp_path, "comfyui", 3, cold_ns=100, warm_ns=(100, 100), output=output),
    ]
    selected = timing_workload()
    roots = {"artifact": tmp_path, "comfyui": tmp_path, "dinkster": CORE_ROOT}
    contract_digest = digest_bytes(canonical_bytes(selected))
    history = timing_history(
        [
            {
                "engines": {
                    "comfyui": {
                        "commit": "c" * 40,
                        "torch": "2.13.0+cu130",
                        "warm_median_ns": 100.0,
                    },
                    "dinkster": {
                        "commit": "d" * 40,
                        "torch": "2.13.0+cu130",
                        "warm_median_ns": 100.0,
                    },
                },
                "hardware": {
                    "device_uuid": "GPU-test-0",
                    "inventory": processes[0]["hardware"]["inventory"],
                },
                "schema": 1,
                "status": "BASELINE",
                "workload": {"contract_digest": contract_digest},
            }
        ]
    )
    summary = summarize_timing(
        processes,
        {
            "acceptance_manifest_digest": selected["acceptance_manifest"]["digest"],
            "checks": [{} for _ in range(16)],
            "pass": True,
            "text_parameter_dtype": None,
            "torch": "2.13.0+cu130",
        },
        selected,
        history,
        roots,
        tmp_path,
        "2026-08-06T00:00:00+00:00",
    )

    assert summary["regression"]["prior_dinkster_warm_ratio"] == format(
        dinkster_warm_ns / 100, ".9f"
    )
    assert summary["regression"]["prior_selection"] == "compatible-record"
    assert summary["regression"]["status"] == expected_status
    assert summary["status"] == expected_status


def test_timing_summary_discloses_incompatible_history_and_refuses_commit_drift(
    tmp_path: Path,
) -> None:
    values = [
        (100, (100, 100)),
        (100, (90, 90)),
        (100, (90, 90)),
        (100, (100, 100)),
    ]
    prior = timing_history(
        [
            {
                "engines": {
                    engine: {
                        "commit": engine[0] * 40,
                        "torch": "2.13.0+cu130",
                        "warm_median_ns": 100.0,
                    }
                    for engine in ("comfyui", "dinkster")
                },
                "hardware": {
                    "device_uuid": "GPU-test-0",
                    "inventory": {"different": "environment"},
                },
                "schema": 1,
                "status": "BASELINE",
                "workload": {"contract_digest": "sha256:" + "0" * 64},
            }
        ]
    )
    summary = timing_summary(tmp_path, values, prior)
    assert summary["regression"]["prior_dinkster_warm_median_ns"] is None
    assert summary["regression"]["prior_dinkster_warm_ratio"] is None
    assert summary["regression"]["prior_selection"] == "no-compatible-record"

    output = tmp_path / "other.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    processes = [
        timing_process(
            tmp_path,
            engine,
            index,
            cold_ns=cold_ns,
            warm_ns=warm_ns,
            output=output,
        )
        for index, (engine, (cold_ns, warm_ns)) in enumerate(
            zip(TIMING_ENGINE_ORDER, values, strict=True)
        )
    ]
    processes[2]["engine"]["commit"] = "e" * 40
    selected = timing_workload()
    with pytest.raises(HarnessError, match="dinkster commit changed"):
        summarize_timing(
            processes,
            {
                "acceptance_manifest_digest": selected["acceptance_manifest"]["digest"],
                "checks": [{} for _ in range(16)],
                "pass": True,
                "text_parameter_dtype": None,
                "torch": "2.13.0+cu130",
            },
            selected,
            timing_history(),
            {"artifact": tmp_path, "comfyui": tmp_path, "dinkster": CORE_ROOT},
            tmp_path,
            "2026-08-06T00:00:00+00:00",
        )

    processes[2]["engine"]["commit"] = "d" * 40
    processes[2]["harness"]["commit"] = "e" * 40
    with pytest.raises(HarnessError, match="harness commit changed"):
        summarize_timing(
            processes,
            {
                "acceptance_manifest_digest": selected["acceptance_manifest"]["digest"],
                "checks": [{} for _ in range(16)],
                "pass": True,
                "text_parameter_dtype": None,
                "torch": "2.13.0+cu130",
            },
            selected,
            timing_history(),
            {"artifact": tmp_path, "comfyui": tmp_path, "dinkster": CORE_ROOT},
            tmp_path,
            "2026-08-06T00:00:00+00:00",
        )


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        (
            [(100, (100, 100)), (100, (90, 90)), (100, (90, 90)), (100, (111, 111))],
            "comfyui warm max/min exceeds 1.10",
        ),
        (
            [(100, (100, 100)), (100, (90, 90)), (100, (90, 90)), (116, (100, 100))],
            "comfyui cold max/min exceeds 1.15",
        ),
        (
            [(100, (100, 100)), (100, (90, 90)), (100, (90, 90)), (100, (106, 106))],
            "comfyui process-position warm medians differ by more than 5 percent",
        ),
        (
            [(100, (100, 100)), (100, (104, 104)), (100, (96, 96)), (100, (100, 100))],
            "the two engine-order directions disagree on which engine is faster",
        ),
    ],
)
def test_timing_variance_and_order_triggers_are_indeterminate(
    tmp_path: Path,
    values: list[tuple[int, tuple[int, int]]],
    reason: str,
) -> None:
    summary = timing_summary(tmp_path, values)
    assert summary["escalation"]["triggered"] is True
    assert reason in summary["escalation"]["reasons"]
    assert summary["regression"]["status"] == "BASELINE"
    assert summary["status"] == "INDETERMINATE"


def test_timing_escalation_threshold_equalities_do_not_trigger(tmp_path: Path) -> None:
    summary = timing_summary(
        tmp_path,
        [
            (100, (100, 100)),
            (100, (100, 110)),
            (100, (100, 110)),
            (115, (100, 100)),
        ],
    )
    assert summary["escalation"] == {"reasons": [], "triggered": False}
    assert summary["status"] == "BASELINE"


@pytest.mark.parametrize("phase", ["resident-warmup", "warm-1"])
def test_timing_correctness_refuses_any_non_equivalent_output(tmp_path: Path, phase: str) -> None:
    zero = tmp_path / "zero.npy"
    one = tmp_path / "one.npy"
    np.save(zero, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    np.save(one, np.ones((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    processes = [
        timing_process(tmp_path, engine, index, cold_ns=1, warm_ns=(1, 1), output=zero)
        for index, engine in enumerate(TIMING_ENGINE_ORDER)
    ]
    processes[2]["observations"][phase]["output"] = {
        "digest": digest_file(one),
        "path": str(one),
    }
    with pytest.raises(HarnessError, match="timing output correctness failed"):
        _timing_correctness(processes, acceptance(), workload())


def test_timing_correctness_requires_actual_canonical_text_parameter_dtype(
    tmp_path: Path,
) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    processes = [
        timing_process(tmp_path, engine, index, cold_ns=1, warm_ns=(1, 1), output=output)
        for index, engine in enumerate(TIMING_ENGINE_ORDER)
    ]
    selected = workload()
    selected["id"] = "W0-HARNESS-SD15-TXT2IMG"
    selected["execution"] = {"precision": {"text": "float32"}}
    policy = acceptance()
    policy["workloads"][selected["id"]] = policy["workloads"].pop("test")
    correctness = _timing_correctness(processes, policy, selected)
    assert correctness["text_parameter_dtype"] == "float32"

    processes[2]["observations"]["resident-warmup"]["runtime"]["text_parameter_dtype"] = "float16"
    with pytest.raises(HarnessError, match="text parameter dtype does not match"):
        _timing_correctness(processes, policy, selected)


def test_run_timing_uses_exact_process_and_request_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = timing_workload()
    calls: list[tuple[str, tuple[str, ...]]] = []
    clean_checks: list[tuple[Path, tuple[str, ...]]] = []

    monkeypatch.setattr(
        harness,
        "hardware_inventory",
        lambda: {
            "gpus": [{"index": 0, "uuid": "GPU-test"}],
        },
    )
    monkeypatch.setattr(harness, "validate_timing_hardware", lambda *_args: {"device_ordinal": 0})
    monkeypatch.setattr(harness, "require_device_idle", lambda _index: None)

    def fake_git_output(root: Path, *args: str) -> str:
        clean_checks.append((root, args))
        return ""

    monkeypatch.setattr(harness, "_git_output", fake_git_output)

    def fake_run_workload(
        _selected: dict[str, Any],
        engine: str,
        _roots: dict[str, Path],
        _digest: str,
        _output_dir: Path,
        *,
        requests: list[dict[str, Any]] | None = None,
        timing_device_uuid: str | None = None,
        timing_hardware: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert timing_device_uuid == "GPU-test-0"
        assert timing_hardware == timing_hardware_pin()
        assert requests is not None
        calls.append((engine, tuple(request["phase"] for request in requests)))
        return {"engine": {"id": engine}}

    monkeypatch.setattr(harness, "run_workload", fake_run_workload)
    monkeypatch.setattr(harness, "_timing_correctness", lambda *_args: {"pass": True})
    monkeypatch.setattr(harness, "summarize_timing", lambda *_args: {"status": "BASELINE"})

    result = run_timing(
        selected,
        {"artifact": tmp_path, "comfyui": tmp_path, "dinkster": tmp_path},
        acceptance(),
        selected["acceptance_manifest"]["digest"],
        timing_history(),
        tmp_path / "attempt",
        "GPU-test-0",
    )

    assert result == {"status": "BASELINE"}
    assert [engine for engine, _phases in calls] == list(TIMING_ENGINE_ORDER)
    assert all(phases == TIMING_PHASES for _engine, phases in calls)
    assert sum(len(phases) for _engine, phases in calls) == 16
    assert clean_checks == [(tmp_path, ("status", "--porcelain"))] * 4


def test_run_timing_refuses_dirty_dinkster_before_comfyui_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        harness,
        "hardware_inventory",
        lambda: {"gpus": [{"index": 0, "uuid": "GPU-test"}]},
    )
    monkeypatch.setattr(harness, "validate_timing_hardware", lambda *_args: {"device_ordinal": 0})
    monkeypatch.setattr(harness, "_git_output", lambda *_args: " M comfyui_adapter.py")
    monkeypatch.setattr(
        harness,
        "run_workload",
        lambda *_args, **_kwargs: pytest.fail("dirty Dinkster checkout reached ComfyUI process"),
    )
    attempt = tmp_path / "attempt"

    with pytest.raises(HarnessError, match="dinkster checkout must be clean"):
        run_timing(
            timing_workload(),
            {"artifact": tmp_path, "comfyui": tmp_path, "dinkster": tmp_path},
            acceptance(),
            timing_workload()["acceptance_manifest"]["digest"],
            timing_history(),
            attempt,
            "GPU-test-0",
        )

    retained = json.loads((attempt / "attempt.json").read_text())
    assert retained["status"] == "INVALID"
    assert retained["processes"] == []


@pytest.mark.parametrize(
    "failure",
    ["artifact digest mismatch", "adapter returncode 1", "adapter cleanup failed"],
)
def test_run_timing_marks_pin_process_and_cleanup_failures_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setattr(
        harness,
        "hardware_inventory",
        lambda: {"gpus": [{"index": 0, "uuid": "GPU-test"}]},
    )
    monkeypatch.setattr(harness, "validate_timing_hardware", lambda *_args: {"device_ordinal": 0})
    monkeypatch.setattr(harness, "require_device_idle", lambda _index: None)
    monkeypatch.setattr(harness, "_git_output", lambda *_args: "")
    monkeypatch.setattr(
        harness,
        "run_workload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(HarnessError(failure)),
    )
    attempt = tmp_path / "attempt"

    with pytest.raises(HarnessError, match=failure):
        run_timing(
            timing_workload(),
            {"artifact": tmp_path, "comfyui": tmp_path, "dinkster": tmp_path},
            acceptance(),
            timing_workload()["acceptance_manifest"]["digest"],
            timing_history(),
            attempt,
            "GPU-test-0",
        )

    assert json.loads((attempt / "attempt.json").read_text())["status"] == "INVALID"


def test_timing_history_schema_refuses_policy_weakening() -> None:
    validate_timing_history(timing_history())
    weakened = timing_history()
    weakened["policy"]["threshold_relation"] = "greater-than-or-equal"
    with pytest.raises(HarnessError, match="timing history threshold relation"):
        validate_timing_history(weakened)


def test_timing_history_pins_exact_ripperpc_inventory() -> None:
    configured = json.loads(Path("tools/inference_parity/timings.json").read_text())
    validate_timing_history(configured)
    assert configured["hardware"] == {
        "device_ordinal": 0,
        "device_uuid": "GPU-ff7692e5-eec5-36b4-4c72-e879f32c5e98",
        "gpus": [
            {
                "driver": "595.84",
                "index": index,
                "memory_total_mib": 97887,
                "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                "uuid": uuid,
            }
            for index, uuid in enumerate(
                (
                    "GPU-ff7692e5-eec5-36b4-4c72-e879f32c5e98",
                    "GPU-73bdaf24-bdbd-c2ee-7442-08a7f4ab3b9d",
                    "GPU-ea539630-50c6-a154-5789-852977375e95",
                    "GPU-66e690fe-c321-d3f3-5192-17a757c711f5",
                )
            )
        ],
        "hostname": "RipperPC",
    }


def test_timing_history_refuses_incomplete_or_indeterminate_rows() -> None:
    incomplete = timing_history([{"schema": 1, "status": "PASS"}])
    with pytest.raises(HarnessError, match="workload contract digest"):
        validate_timing_history(incomplete)
    indeterminate = timing_history([{"schema": 1, "status": "INDETERMINATE"}])
    with pytest.raises(HarnessError, match="not an accepted schema-1 result"):
        validate_timing_history(indeterminate)


def test_manifest_and_acceptance_schema_refuse_missing_contracts() -> None:
    validate_manifest(manifest())
    validate_acceptance(acceptance())
    broken = manifest()
    broken["workloads"][0]["artifacts"][0].pop("digest")
    with pytest.raises(HarnessError, match="digest"):
        validate_manifest(broken)
    broken_acceptance = acceptance()
    broken_acceptance["workloads"]["test"]["phases"].pop("real")
    with pytest.raises(HarnessError, match="warmup and real"):
        validate_acceptance(broken_acceptance)


def test_digest_pinning_changes_with_content(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"a")
    first = digest_file(artifact)
    artifact.write_bytes(b"b")
    assert digest_file(artifact) != first


def test_serialization_is_ascii_sorted_deterministic_and_atomic(tmp_path: Path) -> None:
    value = {"z": "\N{SNOWMAN}", "a": 1}
    assert canonical_bytes(value) == canonical_bytes(value)
    assert canonical_bytes(value).isascii()
    output = tmp_path / "record.json"
    write_json(output, value)
    assert output.read_bytes() == canonical_bytes(value)
    assert not output.with_name("record.json.tmp").exists()
    assert output.read_text().index('"a"') < output.read_text().index('"z"')


def test_compare_gates_phases_independently(tmp_path: Path) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    baseline = record("comfyui", output)
    candidate = record("dinkster", output)
    candidate["warmup"]["metrics"]["peak_ram_bytes"] = 1200
    verdict = compare_records(baseline, candidate, acceptance(), workload())
    assert verdict["warmup"]["pass"] is False
    assert verdict["real"]["pass"] is True
    assert verdict["overall_pass"] is False
    assert verdict["real"]["output_comparison"]["observed"]["ssim"] == "1"


def test_compare_and_cli_return_nonzero_on_real_failure(tmp_path: Path) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    baseline = record("comfyui", output)
    candidate = record("dinkster", output)
    candidate["real"]["metrics"]["generation_ns"] = 200
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    acceptance_path = tmp_path / "acceptance.json"
    for path, value in (
        (baseline_path, baseline),
        (candidate_path, candidate),
        (acceptance_path, acceptance()),
        (tmp_path / "manifest.json", manifest()),
    ):
        write_json(path, value)
    assert (
        main(
            [
                "compare",
                "--baseline",
                str(baseline_path),
                "--candidate",
                str(candidate_path),
                "--acceptance",
                str(acceptance_path),
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--workload",
                "test",
                "--output",
                str(tmp_path / "verdict.json"),
            ]
        )
        == 1
    )
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["warmup"]["pass"] is True
    assert verdict["real"]["pass"] is False


def test_compare_refuses_stale_applied_acceptance_digest(tmp_path: Path) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    stale_workload = workload()
    stale_workload["acceptance_manifest"]["digest"] = "sha256:" + "0" * 64
    with pytest.raises(HarnessError, match="workload acceptance-manifest digest mismatch"):
        compare_records(
            record("comfyui", output), record("dinkster", output), acceptance(), stale_workload
        )


def test_compare_refuses_stale_candidate_even_when_digest_is_predecessor(
    tmp_path: Path,
) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    policy = acceptance()
    predecessor = "sha256:" + "2" * 64
    policy["workloads"]["test"]["calibration"] = {"pre_calibration_acceptance_digest": predecessor}
    current = digest_bytes(canonical_bytes(policy))
    baseline = record("comfyui", output)
    candidate = record("dinkster", output)
    baseline["acceptance_manifest_digest"] = current
    candidate["acceptance_manifest_digest"] = predecessor
    with pytest.raises(HarnessError, match="candidate record acceptance digest is stale"):
        compare_records(baseline, candidate, policy, workload(policy))


def test_compare_accepts_only_declared_baseline_predecessor(tmp_path: Path) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    policy = acceptance()
    predecessor = "sha256:" + "2" * 64
    policy["workloads"]["test"]["calibration"] = {"pre_calibration_acceptance_digest": predecessor}
    current = digest_bytes(canonical_bytes(policy))
    baseline = record("comfyui", output)
    candidate = record("dinkster", output)
    baseline["acceptance_manifest_digest"] = predecessor
    candidate["acceptance_manifest_digest"] = current
    assert compare_records(baseline, candidate, policy, workload(policy))["overall_pass"]
    baseline["acceptance_manifest_digest"] = "sha256:" + "3" * 64
    with pytest.raises(HarnessError, match="not current or declared predecessor"):
        compare_records(baseline, candidate, policy, workload(policy))


def test_compare_refuses_stale_baseline_without_declared_predecessor(tmp_path: Path) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    baseline = record("comfyui", output)
    baseline["acceptance_manifest_digest"] = "sha256:" + "2" * 64
    with pytest.raises(HarnessError, match="not current or declared predecessor"):
        compare_records(baseline, record("dinkster", output), acceptance(), workload())


def test_compare_refuses_null_baseline_without_declared_predecessor(tmp_path: Path) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    baseline = record("comfyui", output)
    baseline["acceptance_manifest_digest"] = None
    with pytest.raises(HarnessError, match="not current or declared predecessor"):
        compare_records(baseline, record("dinkster", output), acceptance(), workload())


def test_compare_refuses_records_for_different_selected_workload(tmp_path: Path) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    selected = workload()
    selected["id"] = "different-workload"
    with pytest.raises(HarnessError, match="does not match selected workload"):
        compare_records(
            record("comfyui", output), record("dinkster", output), acceptance(), selected
        )


def test_compare_refuses_calibration_pending_policy(tmp_path: Path) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    policy = acceptance()
    policy["workloads"]["test"]["status"] = "calibration_pending"
    current = digest_bytes(canonical_bytes(policy))
    baseline = record("comfyui", output)
    candidate = record("dinkster", output)
    baseline["acceptance_manifest_digest"] = current
    candidate["acceptance_manifest_digest"] = current
    with pytest.raises(HarnessError, match="calibration is pending"):
        compare_records(baseline, candidate, policy, workload(policy))


def test_compare_refuses_torch_version_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    candidate = record("dinkster", output)
    candidate["real"]["runtime"]["torch"] = "2.10.0"
    with pytest.raises(HarnessError, match="torch versions differ"):
        compare_records(record("comfyui", output), candidate, acceptance(), workload())


def test_compare_refuses_tampered_persisted_output(tmp_path: Path) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    baseline = record("comfyui", output)
    candidate = record("dinkster", output)
    np.save(output, np.ones((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    with pytest.raises(HarnessError, match="persisted output digest mismatch"):
        compare_records(baseline, candidate, acceptance(), workload())


def test_multi_output_compare_and_missing_latent_refusal(tmp_path: Path) -> None:
    image = tmp_path / "image.npy"
    latent = tmp_path / "latent.npy"
    np.save(image, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    np.save(latent, np.zeros((1, 4, 2, 2), dtype=np.float32), allow_pickle=False)
    policy = acceptance()
    for phase in ("warmup", "real"):
        policy["workloads"]["test"]["phases"][phase]["outputs"] = {
            "image": policy["workloads"]["test"]["phases"][phase].pop("output"),
            "latent": {"comparator": "exact-array/1", "max_abs": "0"},
        }
    current = digest_bytes(canonical_bytes(policy))
    baseline = record("comfyui", image)
    candidate = record("dinkster", image)
    for item in (baseline, candidate):
        item["acceptance_manifest_digest"] = current
        for phase in ("warmup", "real"):
            image_output = item[phase].pop("output")
            item[phase]["outputs"] = {
                "image": image_output,
                "latent": {"digest": digest_file(latent), "path": str(latent)},
            }
    verdict = compare_records(baseline, candidate, policy, workload(policy))
    assert verdict["real"]["output_comparisons"]["latent"]["pass"] is True
    candidate["real"]["outputs"].pop("latent")
    with pytest.raises(HarnessError, match="outputs do not match acceptance"):
        compare_records(baseline, candidate, policy, workload(policy))


def test_adapter_output_contract_refuses_missing_latent_and_wrong_decode_mode() -> None:
    pinned = {
        "execution": {
            "decode_modes": {"comfyui": "tiled", "dinkster": "tiled"},
            "outputs": ["image", "latent"],
        }
    }
    with pytest.raises(HarnessError, match="outputs mismatch"):
        _adapter_outputs(
            {"decode_mode": "tiled", "outputs": {"image": "/tmp/image.npy"}},
            pinned,
            "comfyui",
        )
    with pytest.raises(HarnessError, match="decode_mode mismatch"):
        _adapter_outputs(
            {
                "decode_mode": "regular",
                "outputs": {"image": "/tmp/image.npy", "latent": "/tmp/latent.npy"},
            },
            pinned,
            "dinkster",
        )
    with pytest.raises(HarnessError, match="decode_mode mismatch"):
        _adapter_outputs(
            {
                "decode_mode": "regular",
                "outputs": {"image": "/tmp/image.npy", "latent": "/tmp/latent.npy"},
            },
            pinned,
            "comfyui",
        )


def test_ovis_workload_pins_symmetric_tiled_decode_modes() -> None:
    workloads = json.loads(Path("tools/inference_parity/workloads.json").read_text())["workloads"]
    ovis = next(item for item in workloads if item["id"] == "W0-FLUX-GATED-OVIS-TXT2IMG")
    assert ovis["execution"]["decode_modes"] == {"comfyui": "tiled", "dinkster": "tiled"}


def test_sd15_inpaint_workload_pins_template_artifacts_and_decode_mode() -> None:
    manifest_path = Path("tools/inference_parity/workloads.json")
    acceptance_path = Path("tools/inference_parity/acceptance.json")
    pinned_manifest = json.loads(manifest_path.read_text())
    pinned_acceptance = json.loads(acceptance_path.read_text())
    validate_manifest(pinned_manifest)
    validate_acceptance(pinned_acceptance)
    item = next(
        workload for workload in pinned_manifest["workloads"] if workload["id"] == "W0-SD15-INPAINT"
    )
    assert item["acceptance_manifest"]["digest"] == digest_bytes(canonical_bytes(pinned_acceptance))
    assert item["execution"] == {
        **item["execution"],
        "decode_modes": {"comfyui": "regular", "dinkster": "regular"},
        "grow_mask_by": 6,
        "input_image": "inpaint_example_input_image.png",
        "megapixels": 0.25,
        "negative_prompt": "watermark, text\n",
        "sampler": "uni_pc_bh2",
        "scheduler": "normal",
        "seed": 808369199502636,
        "steps": 20,
        "upscale_method": "nearest-exact",
    }
    assert item["execution"]["precision"]["text"] == "float32"
    artifacts = {artifact["role"]: artifact for artifact in item["artifacts"]}
    assert artifacts["combined-sd15-inpaint-checkpoint"]["digest"] == (
        "sha256:7eca34abcebf1662d0e7021e4f6dd362d5e3e7ab3b5dfb4326b8b08b39b9145d"
    )
    assert artifacts["combined-sd15-inpaint-checkpoint"]["bytes"] == 4265203868
    assert artifacts["template-input-image"]["digest"] == (
        "sha256:ebbf24255b79907c90fe692696eea41fdbe9822d4bbc8ae0f660079f0a9b4559"
    )
    assert artifacts["historical-official-template"]["digest"] == (
        "sha256:b5cc720dcc1b41bb772fbadac2fbda1202c2bd955cd13bb61ce59b86006ba0b8"
    )
    assert item["provenance"]["template_commit"] == ("5b32eb71d46ca0b2fd6c1d48ad6e5a383f95d629")
    assert {graph["root"] for graph in item["graph"].values()} == {"artifact"}
    for phase in ("warmup", "real"):
        assert pinned_acceptance["workloads"]["W0-SD15-INPAINT"]["phases"][phase]["output"] == {
            "max_abs": "0",
            "mean_abs": "0",
            "ssim_minimum": "1",
        }


def test_sd15_inpaint_artifact_graph_digest_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned_manifest = json.loads(Path("tools/inference_parity/workloads.json").read_text())
    item = next(
        workload for workload in pinned_manifest["workloads"] if workload["id"] == "W0-SD15-INPAINT"
    )
    item["engines"]["comfyui"]["commit"] = "HEAD"
    item["graph"]["comfyui"]["digest"] = "sha256:" + "0" * 64
    monkeypatch.setattr(
        harness,
        "_git_output",
        lambda _root, *args: "" if args[0] == "status" else "a" * 40,
    )
    artifact_root = tmp_path / "artifact"
    graph_path = artifact_root / item["graph"]["comfyui"]["path"]
    graph_path.parent.mkdir(parents=True)
    graph_path.write_bytes(b"{}")
    roots = {
        "artifact": artifact_root,
        "comfyui": tmp_path / "comfyui",
        "dinkster": CORE_ROOT,
        "template": tmp_path / "template",
    }
    with pytest.raises(HarnessError, match="graph digest mismatch"):
        validate_pins(item, "comfyui", roots, item["acceptance_manifest"]["digest"])


@requires_comfyui_station
def test_sd15_inpaint_adapters_construct_with_cuda_hidden(tmp_path: Path) -> None:
    pinned_manifest = json.loads(Path("tools/inference_parity/workloads.json").read_text())
    item = next(
        workload for workload in pinned_manifest["workloads"] if workload["id"] == "W0-SD15-INPAINT"
    )
    common = [
        "--artifact-root",
        str(_STATION["artifact"]),
        "--output-dir",
        str(tmp_path),
        "--workload-json",
        json.dumps(item["execution"], sort_keys=True),
        "--construct-only",
    ]
    comfy = subprocess.run(
        [
            str(_STATION["python"]),
            "tools/inference_parity/sd15_inpaint_comfyui_adapter.py",
            "--repo",
            str(_STATION["comfyui"]),
            *common,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert comfy.returncode == 0, comfy.stderr
    assert "comfy_args.cpu=True; nodes imported" in comfy.stdout
    dinkster = subprocess.run(
        [
            str(_STATION["python"]),
            "tools/inference_parity/sd15_inpaint_dinkster_adapter.py",
            "--repo",
            str(CORE_ROOT),
            *common,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert dinkster.returncode == 0, dinkster.stderr
    assert "family=dinkster.sd15; latent=(1, 4, 64, 64); mask=(1, 1, 512, 512)" in dinkster.stdout


def test_sd15_inpaint_comfyui_adapter_prefers_checkout_tools_package(tmp_path: Path) -> None:
    hostile_package = tmp_path / "tools"
    hostile_package.mkdir()
    (hostile_package / "__init__.py").write_text(
        'raise RuntimeError("hostile foreign tools package imported")\n'
    )
    result = subprocess.run(
        [
            sys.executable,
            "tools/inference_parity/sd15_inpaint_comfyui_adapter.py",
            "--help",
        ],
        env={"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--construct-only" in result.stdout


def test_sd15_inpaint_adapters_pin_and_report_loaded_text_parameter_dtype() -> None:
    comfy_source = Path("tools/inference_parity/sd15_inpaint_comfyui_adapter.py").read_text()
    dinkster_source = Path("tools/inference_parity/sd15_inpaint_dinkster_adapter.py").read_text()
    assert "_set_text_precision(workload, comfy_args)" in comfy_source
    assert '"text_parameter_dtype": text_parameter_dtype' in comfy_source
    assert "text_dtype=torch.float32" in dinkster_source
    assert '"text_parameter_dtype": text_parameter_dtype' in dinkster_source


def test_sd15_inpaint_dinkster_adapter_places_only_diffusion_in_compute_dtype() -> None:
    source = Path("tools/inference_parity/sd15_inpaint_dinkster_adapter.py").read_text()
    assert "diffusion_dtype = torch.float16" in source
    assert "diffusion_dtype=diffusion_dtype" in source
    placement = (
        "runtime.assembled.diffusion.to(\n"
        "                device=device,\n"
        "                dtype=diffusion_dtype,"
    )
    assert placement in source
    assert "runtime.assembled.clip_l.to(device)" in source
    assert "runtime.assembled.vae.to(device)" in source
    assert "runtime.assembled.clip_l.to(device, dtype=" not in source
    assert "runtime.assembled.vae.to(device, dtype=" not in source


def test_sdxl_inpaint_workload_freezes_artifact_graph_execution_and_policy() -> None:
    pinned_manifest = json.loads(Path("tools/inference_parity/workloads.json").read_text())
    pinned_acceptance = json.loads(
        Path("tools/inference_parity/acceptance_sdxl_inpaint.json").read_text()
    )
    existing_acceptance = json.loads(Path("tools/inference_parity/acceptance.json").read_text())
    validate_manifest(pinned_manifest)
    validate_acceptance(pinned_acceptance)
    item = next(
        workload for workload in pinned_manifest["workloads"] if workload["id"] == "W0-SDXL-INPAINT"
    )
    assert item["acceptance_manifest"]["digest"] == digest_bytes(canonical_bytes(pinned_acceptance))
    checkpoint, fixture, template = item["artifacts"]
    assert checkpoint == {
        **checkpoint,
        "bytes": 6938069944,
        "digest": "sha256:fe1b97fe6544814eb6fc8ce53f04ad8d339ec6946b58b0afd566fcc47813fa8a",
        "license": "CreativeML Open RAIL++-M",
        "path": "checkpoints/sd_xl_base_1.0_inpainting_0.1.safetensors",
        "repository_revision": "2c65165f0d5c3d93f671ef7613f3a94f4b3ec30a",
        "source_url": (
            "https://huggingface.co/benjamin-paine/sd-xl-alternative-bases/resolve/"
            "2c65165f0d5c3d93f671ef7613f3a94f4b3ec30a/"
            "sd_xl_base_1.0_inpainting_0.1.safetensors"
        ),
    }
    assert checkpoint["layout"]["external_component_substitution"] is False
    assert fixture["digest"] == (
        "sha256:ebbf24255b79907c90fe692696eea41fdbe9822d4bbc8ae0f660079f0a9b4559"
    )
    assert fixture["bytes"] == 1296300
    assert template["digest"] == (
        "sha256:b5cc720dcc1b41bb772fbadac2fbda1202c2bd955cd13bb61ce59b86006ba0b8"
    )
    assert template["bytes"] == 11878
    assert item["graph"]["comfyui"] == item["graph"]["dinkster"]
    assert item["graph"]["comfyui"]["commit"] == ("5b32eb71d46ca0b2fd6c1d48ad6e5a383f95d629")
    assert item["engines"]["comfyui"]["commit"] == ("00d02f2854892ee5b9808bc2f6348b972017886a")
    execution = item["execution"]
    sd15 = next(
        workload for workload in pinned_manifest["workloads"] if workload["id"] == "W0-SD15-INPAINT"
    )["execution"]
    retained = {
        "cfg",
        "denoise",
        "grow_mask_by",
        "input_image",
        "negative_prompt",
        "precision",
        "prompt",
        "sampler",
        "scheduler",
        "seed",
        "steps",
        "upscale_method",
    }
    assert {key: execution[key] for key in retained} == {key: sd15[key] for key in retained}
    assert execution["megapixels"] == 1.0
    assert (execution["width"], execution["height"]) == (1024, 1024)
    assert execution["outputs"] == ["image", "latent"]
    assert execution["mask_polarity"] == "1-alpha"
    assert execution["no_crop"] is True
    exact = {"max_abs": "0", "mean_abs": "0", "ssim_minimum": "1"}
    for phase in ("warmup", "real"):
        policy = pinned_acceptance["workloads"]["W0-SDXL-INPAINT"]["phases"][phase]
        assert policy["outputs"] == {"image": exact, "latent": exact}
        assert (
            policy["metrics"]
            == existing_acceptance["workloads"]["W0-SD15-INPAINT"]["phases"][phase]["metrics"]
        )


def _write_test_safetensors(path: Path, entries: dict[str, dict[str, Any]], payload: bytes) -> None:
    header = json.dumps(entries, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def test_sdxl_inpaint_safetensors_preflight_validates_geometry_ranges_and_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.safetensors"
    entries = {
        "first": {"data_offsets": [0, 4], "dtype": "F16", "shape": [2]},
        "second": {"data_offsets": [4, 8], "dtype": "F32", "shape": [1]},
    }
    _write_test_safetensors(path, entries, b"12345678")
    contract = {
        "required_tensors": {
            "first": {"dtype": "F16", "shape": [2]},
            "second": {"dtype": "F32", "shape": [1]},
        }
    }
    result = _validate_safetensors_contract(path, contract)
    assert result["tensor_count"] == 2
    assert result["payload_size"] == 8
    broken = deepcopy(entries)
    broken["second"]["data_offsets"] = [3, 7]
    _write_test_safetensors(path, broken, b"12345678")
    with pytest.raises(HarnessError, match="overlap"):
        _validate_safetensors_contract(path, contract)
    broken = deepcopy(entries)
    broken["second"]["data_offsets"] = [5, 9]
    _write_test_safetensors(path, broken, b"123456789")
    with pytest.raises(HarnessError, match="gap"):
        _validate_safetensors_contract(path, contract)
    broken = deepcopy(entries)
    broken["first"]["data_offsets"] = [1, 5]
    broken["second"]["data_offsets"] = [5, 9]
    _write_test_safetensors(path, broken, b"123456789")
    with pytest.raises(HarnessError, match="gap"):
        _validate_safetensors_contract(path, contract)
    _write_test_safetensors(path, entries, b"123456789")
    with pytest.raises(HarnessError, match="complete payload"):
        _validate_safetensors_contract(path, contract)
    _write_test_safetensors(path, entries, b"12345678")
    contract["required_tensors"]["first"]["shape"] = [1]
    with pytest.raises(HarnessError, match="geometry mismatch"):
        _validate_safetensors_contract(path, contract)


def test_sdxl_inpaint_png_preflight_reads_rgba_shape_and_alpha_polarity(
    tmp_path: Path,
) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 2, 1, 8, 6, 0, 0, 0)
    pixels = b"\x00" + bytes((1, 2, 3, 0, 4, 5, 6, 255))
    path = tmp_path / "fixture.png"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )
    assert _png_rgba_contract(path) == {
        "alpha_mask_below_128_percent": "50.000",
        "mode": "RGBA",
        "size": "2x1",
    }


def test_sdxl_inpaint_adapters_are_named_multi_output_and_preflight_is_pure_stdlib() -> None:
    comfy = Path("tools/inference_parity/sdxl_inpaint_comfyui_adapter.py").read_text()
    dinkster = Path("tools/inference_parity/sdxl_inpaint_dinkster_adapter.py").read_text()
    harness_source = Path("tools/inference_parity/harness.py").read_text()
    for source in (comfy, dinkster):
        assert '"outputs": {' in source
        assert '"image": str(image_output)' in source
        assert '"latent": str(latent_output)' in source
        assert "--construct-only" in source
        assert 'workload["hardware"]["device_uuid"]' in source
    assert "runtime.assembled.clip_l.to(device)" in dinkster
    assert "runtime.assembled.clip_g.to(device)" in dinkster
    assert "diffusion_dtype = torch.float16" in dinkster
    preflight = ast.get_source_segment(
        harness_source,
        next(
            node
            for node in ast.parse(harness_source).body
            if isinstance(node, ast.FunctionDef) and node.name == "validate_sdxl_inpaint_preflight"
        ),
    )
    assert preflight is not None
    assert "import torch" not in preflight
    assert "import comfy" not in preflight


def test_sdxl_inpaint_hardware_pin_checks_uuid_driver_and_memory() -> None:
    manifest = json.loads(Path("tools/inference_parity/workloads.json").read_text())
    item = next(
        workload for workload in manifest["workloads"] if workload["id"] == "W0-SDXL-INPAINT"
    )
    expected = item["execution"]["hardware"]
    inventory = {
        "gpus": [
            {
                "driver": expected["driver"],
                "index": index,
                "memory_total_mib": expected["memory_total_mib"],
                "name": expected["gpu_names"][index],
                "uuid": expected["gpu_uuids"][index],
            }
            for index in range(expected["gpu_count"])
        ]
    }
    assert harness.validate_hardware(item, inventory)["device_uuid"] == expected["device_uuid"]
    for field, message in (
        ("uuid", "UUID"),
        ("driver", "driver"),
        ("memory_total_mib", "memory"),
    ):
        broken = deepcopy(inventory)
        broken["gpus"][-1][field] = "wrong" if field != "memory_total_mib" else 1
        with pytest.raises(HarnessError, match=message):
            harness.validate_hardware(item, broken)


def test_sdxl_inpaint_engine_and_manifest_pins_refuse_substitution() -> None:
    manifest_path = Path("tools/inference_parity/workloads.json").resolve()
    acceptance_path = Path("tools/inference_parity/acceptance_sdxl_inpaint.json").resolve()
    manifest = json.loads(manifest_path.read_text())
    workload = next(item for item in manifest["workloads"] if item["id"] == "W0-SDXL-INPAINT")
    interpreter = workload["preflight"]["interpreter"]["path"]
    for engine in ("comfyui", "dinkster"):
        assert workload["engines"][engine]["python"] == interpreter
        adapter = workload["engines"][engine]["adapter"]
        assert f"tools/inference_parity/{adapter}" in workload["preflight"]["source_files"]
    assert workload["acceptance_manifest"]["path"] == acceptance_path.name
    acceptance = json.loads(acceptance_path.read_text())
    assert acceptance_path.read_bytes() == canonical_bytes(acceptance)


def test_sdxl_vpred_workload_pins_template_artifact_and_execution() -> None:
    manifest_path = Path("tools/inference_parity/workloads.json")
    acceptance_path = Path("tools/inference_parity/acceptance.json")
    pinned_manifest = json.loads(manifest_path.read_text())
    pinned_acceptance = json.loads(acceptance_path.read_text())
    validate_manifest(pinned_manifest)
    validate_acceptance(pinned_acceptance)
    item = next(
        workload for workload in pinned_manifest["workloads"] if workload["id"] == "W0-SDXL-VPRED"
    )
    assert item["acceptance_manifest"] == {
        "digest": digest_bytes(canonical_bytes(pinned_acceptance)),
        "id": "w0-sdxl-vpred-v1",
        "path": "acceptance.json",
    }
    assert item["acceptance_manifest"]["id"] not in pinned_acceptance["workloads"]
    assert item["artifacts"] == [
        {
            **item["artifacts"][0],
            "bytes": 7105350110,
            "digest": ("sha256:ea349eeae87ca8d25ba902c93810f7ca83e5c82f920edf12f273af004ae02819"),
            "path": "checkpoints/NoobAI-XL-Vpred-v1.0.safetensors",
            "role": "combined-sdxl-vpred-ztsnr-checkpoint",
        }
    ]
    assert item["execution"] == {
        **item["execution"],
        "batch": 1,
        "cfg": 8.0,
        "decode_modes": {"comfyui": "regular", "dinkster": "regular"},
        "denoise": 1.0,
        "device_ordinal": 0,
        "height": 1024,
        "negative_prompt": "text, watermark",
        "precision": {"codec": "float32", "diffusion": "float16", "text": "float32"},
        "prompt": "beautiful scenery nature glass bottle landscape, purple galaxy bottle,",
        "sampler": "euler",
        "scheduler": "normal",
        "seed": 685468484323813,
        "steps": 20,
        "width": 1024,
    }
    assert item["graph"]["comfyui"] == {
        "commit": "aa3661d9fc1a493f8de6b029f5b8af27da3c5d08",
        "digest": "sha256:396fb34efa019c5f1cc242da22f89a7fcc43fce7240d014d75290278cb2f6dc0",
        "path": "templates/default.json",
        "root": "template",
    }
    assert item["graph"]["dinkster"]["digest"] == (
        "sha256:4a8bc366e27e3482b838c481d0d4ccb2dfd60a61eee81a740ef73e73aa05962c"
    )


def test_sdxl_vpred_dinkster_graph_is_canonical_txt2img_graph() -> None:
    canonical = Path("tools/inference_parity/graphs/sd15_text_to_image_dinkster.json")
    sdxl = Path("tools/inference_parity/graphs/sdxl_vpred_text_to_image_dinkster.json")
    assert sdxl.read_bytes() == canonical.read_bytes()
    assert digest_file(sdxl) == (
        "sha256:4a8bc366e27e3482b838c481d0d4ccb2dfd60a61eee81a740ef73e73aa05962c"
    )


def test_sdxl_vpred_graph_digest_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pinned_manifest = json.loads(Path("tools/inference_parity/workloads.json").read_text())
    item = next(
        workload for workload in pinned_manifest["workloads"] if workload["id"] == "W0-SDXL-VPRED"
    )
    item["graph"]["dinkster"]["digest"] = "sha256:" + "0" * 64
    monkeypatch.setattr(
        harness,
        "_git_output",
        lambda _root, *args: "" if args[0] == "status" else "a" * 40,
    )
    roots = {
        "artifact": tmp_path / "artifact",
        "comfyui": tmp_path / "comfyui",
        "dinkster": EVIDENCE_ROOT,
        "template": tmp_path / "template",
    }
    with pytest.raises(HarnessError, match="graph digest mismatch"):
        validate_pins(item, "dinkster", roots, item["acceptance_manifest"]["digest"])


@requires_comfyui_station
def test_sdxl_vpred_adapters_construct_with_cuda_hidden(tmp_path: Path) -> None:
    pinned_manifest = json.loads(Path("tools/inference_parity/workloads.json").read_text())
    item = next(
        workload for workload in pinned_manifest["workloads"] if workload["id"] == "W0-SDXL-VPRED"
    )
    common = [
        "--artifact-root",
        str(_STATION["artifact"]),
        "--output-dir",
        str(tmp_path),
        "--workload-json",
        json.dumps(item["execution"], sort_keys=True),
        "--construct-only",
    ]
    comfy = subprocess.run(
        [
            str(_STATION["python"]),
            "tools/inference_parity/sdxl_vpred_comfyui_adapter.py",
            "--repo",
            str(_STATION["comfyui"]),
            *common,
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"},
    )
    assert comfy.returncode == 0, comfy.stderr
    assert "READY comfyui cpu=True; nodes imported; cuda_initialized=False" in comfy.stdout
    dinkster = subprocess.run(
        [
            str(_STATION["python"]),
            "tools/inference_parity/sdxl_vpred_dinkster_adapter.py",
            "--repo",
            str(CORE_ROOT),
            *common,
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"},
    )
    assert dinkster.returncode == 0, dinkster.stderr
    assert (
        "READY dinkster family=dinkster.sdxl; parameterization=v_prediction; "
        "zsnr=True; cuda_initialized=False"
    ) in dinkster.stdout


def test_sdxl_vpred_adapters_pin_and_report_precision() -> None:
    comfy_source = Path("tools/inference_parity/sdxl_vpred_comfyui_adapter.py").read_text()
    dinkster_source = Path("tools/inference_parity/sdxl_vpred_dinkster_adapter.py").read_text()
    assert "_configure_precision(workload, comfy_args)" in comfy_source
    assert '"text_parameter_dtype": text_parameter_dtype' in comfy_source
    assert "diffusion_dtype = torch.float16" in dinkster_source
    assert "text_dtype = torch.float32" in dinkster_source
    assert "vae_dtype = torch.float32" in dinkster_source
    assert "diffusion_dtype=diffusion_dtype" in dinkster_source
    assert "text_dtype=text_dtype" in dinkster_source
    assert "vae_dtype=vae_dtype" in dinkster_source
    assert '"text_parameter_dtype": text_parameter_dtype' in dinkster_source
    placement = (
        "runtime.assembled.diffusion.to(\n"
        "                device=device,\n"
        "                dtype=diffusion_dtype,"
    )
    assert placement in dinkster_source
    assert "runtime.assembled.clip_l.to(device=device, dtype=text_dtype)" in dinkster_source
    assert "runtime.assembled.clip_g.to(device=device, dtype=text_dtype)" in dinkster_source
    assert "runtime.assembled.vae.to(device=device, dtype=vae_dtype)" in dinkster_source
    assert "Path(__file__)" not in comfy_source
    assert "Path(__file__)" not in dinkster_source


def test_sdxl_vpred_comfyui_adapter_selects_all_precision_pins() -> None:
    args = Namespace()
    _configure_sdxl_vpred_precision(
        {"precision": {"codec": "float32", "diffusion": "float16", "text": "float32"}},
        args,
    )
    assert {name for name, value in vars(args).items() if value} == {
        "fp16_unet",
        "fp32_text_enc",
        "fp32_vae",
    }


def test_sdxl_edm_vpred_workload_pins_artifact_template_and_acceptance() -> None:
    manifest_path = Path("tools/inference_parity/workloads.json")
    acceptance_path = Path("tools/inference_parity/acceptance.json")
    pinned_manifest = json.loads(manifest_path.read_text())
    pinned_acceptance = json.loads(acceptance_path.read_text())
    validate_manifest(pinned_manifest)
    validate_acceptance(pinned_acceptance)
    item = next(
        workload
        for workload in pinned_manifest["workloads"]
        if workload["id"] == "W0-SDXL-EDM-VPRED"
    )
    assert item["acceptance_manifest"] == {
        "digest": digest_bytes(canonical_bytes(pinned_acceptance)),
        "id": "w0-sdxl-edm-vpred-v1",
        "path": "acceptance.json",
    }
    assert {
        workload["acceptance_manifest"]["digest"]
        for workload in pinned_manifest["workloads"]
        if workload["id"] != "W0-SDXL-INPAINT"
    } == {digest_bytes(canonical_bytes(pinned_acceptance))}
    assert (
        pinned_acceptance["workloads"][item["id"]]
        == pinned_acceptance["workloads"]["W0-SDXL-VPRED"]
    )
    assert item["artifacts"] == [
        {
            **item["artifacts"][0],
            "bytes": 6938075892,
            "digest": ("sha256:0c2ff84b8dea2cea110dc71b62be29c6b417d6c50f31e89d9fb37596c77062c6"),
            "path": "checkpoints/cosxl.safetensors",
            "role": "combined-sdxl-edm-vpred-checkpoint",
        }
    ]
    assert item["execution"] == {
        **item["execution"],
        "batch": 1,
        "cfg": 8.0,
        "decode_modes": {"comfyui": "regular", "dinkster": "regular"},
        "denoise": 1.0,
        "device_ordinal": 0,
        "height": 1024,
        "negative_prompt": "text, watermark",
        "precision": {"codec": "float32", "diffusion": "float16", "text": "float32"},
        "prompt": "beautiful scenery nature glass bottle landscape, purple galaxy bottle,",
        "sampler": "euler",
        "scheduler": "normal",
        "seed": 685468484323813,
        "steps": 20,
        "width": 1024,
    }
    assert item["graph"]["comfyui"] == {
        "commit": "aa3661d9fc1a493f8de6b029f5b8af27da3c5d08",
        "digest": "sha256:396fb34efa019c5f1cc242da22f89a7fcc43fce7240d014d75290278cb2f6dc0",
        "path": "templates/default.json",
        "root": "template",
    }
    assert item["graph"]["dinkster"] == {
        "commit": "HEAD",
        "digest": "sha256:4a8bc366e27e3482b838c481d0d4ccb2dfd60a61eee81a740ef73e73aa05962c",
        "path": "tools/inference_parity/graphs/sdxl_vpred_text_to_image_dinkster.json",
        "root": "engine",
    }


@requires_comfyui_station
def test_sdxl_edm_vpred_adapters_construct_with_cuda_hidden(tmp_path: Path) -> None:
    pinned_manifest = json.loads(Path("tools/inference_parity/workloads.json").read_text())
    item = next(
        workload
        for workload in pinned_manifest["workloads"]
        if workload["id"] == "W0-SDXL-EDM-VPRED"
    )
    common = [
        "--artifact-root",
        str(_STATION["artifact"]),
        "--output-dir",
        str(tmp_path),
        "--workload-json",
        json.dumps(item["execution"], sort_keys=True),
        "--construct-only",
    ]
    environment = {"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"}
    comfy = subprocess.run(
        [
            str(_STATION["python"]),
            "tools/inference_parity/sdxl_edm_vpred_comfyui_adapter.py",
            "--repo",
            str(_STATION["comfyui"]),
            *common,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert comfy.returncode == 0, comfy.stderr
    assert (
        "READY comfyui cpu=True; nodes imported; edm_vpred markers pinned; cuda_initialized=False"
    ) in comfy.stdout
    dinkster = subprocess.run(
        [
            str(_STATION["python"]),
            "tools/inference_parity/sdxl_edm_vpred_dinkster_adapter.py",
            "--repo",
            str(CORE_ROOT),
            *common,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert dinkster.returncode == 0, dinkster.stderr
    assert (
        "READY dinkster family=dinkster.sdxl; parameterization=v_prediction; "
        "space=continuous_edm; sigma_min=0.0020000000949949026; sigma_max=120.0; "
        "cuda_initialized=False"
    ) in dinkster.stdout


def test_sdxl_edm_vpred_adapters_pin_and_report_precision() -> None:
    comfy_source = Path("tools/inference_parity/sdxl_edm_vpred_comfyui_adapter.py").read_text()
    dinkster_source = Path("tools/inference_parity/sdxl_edm_vpred_dinkster_adapter.py").read_text()
    assert "_configure_precision(workload, comfy_args)" in comfy_source
    assert '"text_parameter_dtype": text_parameter_dtype' in comfy_source
    assert "diffusion_dtype = torch.float16" in dinkster_source
    assert "text_dtype = torch.float32" in dinkster_source
    assert "vae_dtype = torch.float32" in dinkster_source
    assert "diffusion_dtype=diffusion_dtype" in dinkster_source
    assert "text_dtype=text_dtype" in dinkster_source
    assert "vae_dtype=vae_dtype" in dinkster_source
    assert '"text_parameter_dtype": text_parameter_dtype' in dinkster_source
    assert "runtime.assembled.clip_l.to(device=device, dtype=text_dtype)" in dinkster_source
    assert "runtime.assembled.clip_g.to(device=device, dtype=text_dtype)" in dinkster_source
    assert "runtime.assembled.vae.to(device=device, dtype=vae_dtype)" in dinkster_source


def test_sdxl_edm_vpred_comfyui_adapter_selects_all_precision_pins() -> None:
    args = Namespace()
    _configure_sdxl_edm_vpred_precision(
        {"precision": {"codec": "float32", "diffusion": "float16", "text": "float32"}},
        args,
    )
    assert {name for name, value in vars(args).items() if value} == {
        "fp16_unet",
        "fp32_text_enc",
        "fp32_vae",
    }


def test_compare_refuses_persisted_wrong_decode_mode(tmp_path: Path) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 2, 2, 3), dtype=np.float32), allow_pickle=False)
    pinned = workload()
    pinned["execution"] = {"decode_modes": {"comfyui": "tiled", "dinkster": "tiled"}}
    baseline = record("comfyui", output)
    candidate = record("dinkster", output)
    for phase in ("warmup", "real"):
        baseline[phase]["metrics"]["decode_mode"] = "tiled"
        candidate[phase]["metrics"]["decode_mode"] = "tiled"
    candidate["real"]["metrics"]["decode_mode"] = "regular"
    with pytest.raises(HarnessError, match="decode_mode does not match"):
        compare_records(baseline, candidate, acceptance(), pinned)


def test_ovis_comfyui_precision_pins_bf16_vae() -> None:
    args = Namespace()
    _configure_precision(args)
    assert args.bf16_vae is True
    assert args.fp16_vae is False
    assert args.fp32_vae is False


def test_ovis_comfyui_adapter_pins_explicit_tiled_decode_geometry() -> None:
    source = Path("tools/inference_parity/ovis_comfyui_adapter.py").read_text()
    assert "memory_used = vae.memory_used_decode(" in source
    assert "model_management.load_models_gpu(" in source
    assert "force_full_load=vae.disable_offload" in source
    tree = ast.parse(source)
    movedim_calls = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "movedim"
        and isinstance(call.func.value, ast.Call)
        and isinstance(call.func.value.func, ast.Attribute)
        and call.func.value.func.attr == "decode_tiled_"
        and isinstance(call.func.value.func.value, ast.Name)
        and call.func.value.func.value.id == "vae"
    ]
    assert len(movedim_calls) == 1
    movedim_call = movedim_calls[0]
    assert isinstance(movedim_call.func, ast.Attribute)
    assert isinstance(movedim_call.func.value, ast.Call)
    decode_call = movedim_call.func.value
    assert len(decode_call.args) == 1
    latent_arg = decode_call.args[0]
    assert isinstance(latent_arg, ast.Subscript)
    assert isinstance(latent_arg.value, ast.Name)
    assert latent_arg.value.id == "sampled"
    assert isinstance(latent_arg.slice, ast.Constant)
    assert latent_arg.slice.value == "samples"
    assert {keyword.arg: ast.literal_eval(keyword.value) for keyword in decode_call.keywords} == {
        "tile_x": 64,
        "tile_y": 64,
        "overlap": 16,
    }
    assert [ast.literal_eval(arg) for arg in movedim_call.args] == [1, -1]
    assert movedim_call.keywords == []
    assert '"decode_mode": "tiled"' in source


def test_ovis_dinkster_adapter_pins_text_compute_context_and_tiled_decode() -> None:
    path = Path("tools/inference_parity/ovis_dinkster_adapter.py")
    source = path.read_text()
    tree = ast.parse(source)
    load_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_runtime"
    ]
    assert len(load_calls) == 1
    text_dtype = next(
        keyword.value for keyword in load_calls[0].keywords if keyword.arg == "text_dtype"
    )
    assert ast.unparse(text_dtype) == "torch.float32"
    assert source.count("with torch.inference_mode():") == 2
    assert "sampled.to(torch.bfloat16)" in source
    assert "runtime.codec.decode_tiled(" in source
    assert 'output_device="cpu"' in source
    assert "dtype=torch.float32" in source


def test_decode_calibration_refuses_regular_side_tiled_fallback() -> None:
    class VAE:
        def decode_tiled_(self, latent: object) -> object:
            return latent

        def decode(self, latent: object) -> object:
            return self.decode_tiled_(latent)

    class Context:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    with pytest.raises(RuntimeError, match="fell back to tiled"):
        _regular_decode_without_fallback(VAE(), object(), Context)


def test_pin_validation_refuses_wrong_artifact_and_template_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact.bin"
    graph = tmp_path / "graph.json"
    artifact.write_bytes(b"artifact")
    graph.write_bytes(b"graph")
    current = "sha256:" + "1" * 64
    pinned = {
        "acceptance_manifest": {"digest": current},
        "artifacts": [
            {
                "bytes": artifact.stat().st_size,
                "digest": digest_file(artifact),
                "path": str(artifact),
                "role": "model",
            }
        ],
        "engines": {"comfyui": {"commit": "HEAD"}},
        "graph": {
            "comfyui": {
                "commit": "HEAD",
                "digest": digest_file(graph),
                "path": str(graph),
                "root": "template",
            }
        },
    }
    roots = {name: tmp_path for name in ("artifact", "comfyui", "dinkster", "template")}
    monkeypatch.setattr(
        harness,
        "_git_output",
        lambda _root, *args: "" if args[0] == "status" else "a" * 40,
    )
    result = validate_pins(pinned, "comfyui", roots, current)
    assert set(result) == {"artifacts", "engine_commit", "graph_commit", "graph_digest"}
    assert result["engine_commit"] == "a" * 40
    pinned["artifacts"][0]["digest"] = "sha256:" + "0" * 64
    with pytest.raises(HarnessError, match="artifact digest mismatch"):
        validate_pins(pinned, "comfyui", roots, current)
    pinned["artifacts"][0]["digest"] = digest_file(artifact)
    pinned["graph"]["comfyui"]["digest"] = "sha256:" + "0" * 64
    with pytest.raises(HarnessError, match="graph digest mismatch"):
        validate_pins(pinned, "comfyui", roots, current)


def test_station_prerequisites_default_override_and_probe(tmp_path: Path) -> None:
    default = station_paths({})
    assert default["artifact"] == Path("/home/kosin/ComfyUI-Shared/models")
    assert default["comfyui"] == Path("/home/kosin/ComfyUI")
    assert default["python"] == Path("/home/kosin/ComfyUI/venv/bin/python")
    overridden = station_paths(
        {
            "DINKSTER_PARITY_ARTIFACT_ROOT": str(tmp_path / "models"),
            "DINKSTER_PARITY_COMFYUI_ROOT": str(tmp_path / "comfy"),
        }
    )
    assert overridden["artifact"] == tmp_path / "models"
    assert overridden["comfyui"] == tmp_path / "comfy"
    assert overridden["python"] == tmp_path / "comfy" / "venv" / "bin" / "python"
    explicit = station_paths({"DINKSTER_PARITY_COMFYUI_PYTHON": str(tmp_path / "python")})
    assert explicit["python"] == tmp_path / "python"
    assert explicit["comfyui"] == Path("/home/kosin/ComfyUI")
    assert not station_available(overridden)
    (tmp_path / "comfy" / "venv" / "bin").mkdir(parents=True)
    (tmp_path / "comfy" / "venv" / "bin" / "python").write_bytes(b"")
    assert not station_available(overridden)
    (tmp_path / "comfy" / "main.py").write_text("")
    assert not station_available(overridden)
    (tmp_path / "models").mkdir()
    assert station_available(overridden)


def test_protocol_has_exactly_two_ordered_independently_gated_phases() -> None:
    from tools.inference_parity.harness import PHASES

    assert PHASES == ("warmup", "real")
    assert phase_requests(123) == (
        {"phase": "warmup", "seed": 123},
        {"phase": "real", "seed": 123},
    )


@pytest.mark.parametrize(
    ("codec", "selected"),
    (("float32", "fp32_vae"), ("bfloat16", "bf16_vae"), ("float16", "fp16_vae")),
)
def test_comfyui_adapter_selects_exactly_one_vae_precision(codec: str, selected: str) -> None:
    args = Namespace(fp32_vae=True, bf16_vae=True, fp16_vae=True)
    _set_vae_precision({"precision": {"codec": codec}}, args)
    assert {name for name, value in vars(args).items() if value} == {selected}


def test_comfyui_adapter_refuses_unknown_or_missing_codec_precision() -> None:
    args = Namespace(fp32_vae=False, bf16_vae=False, fp16_vae=False)
    with pytest.raises(ValueError, match="precision must be an object"):
        _set_vae_precision({}, args)
    with pytest.raises(ValueError, match="precision.codec must be one of"):
        _set_vae_precision({"precision": {"codec": "automatic"}}, args)


@pytest.mark.parametrize(
    ("text", "selected"),
    (
        ("float32", "fp32_text_enc"),
        ("bfloat16", "bf16_text_enc"),
        ("float16", "fp16_text_enc"),
    ),
)
def test_comfyui_adapter_selects_exactly_one_text_precision(text: str, selected: str) -> None:
    args = Namespace(fp32_text_enc=True, bf16_text_enc=True, fp16_text_enc=True)
    _set_text_precision({"precision": {"text": text}}, args)
    assert {name for name, value in vars(args).items() if value} == {selected}


def test_comfyui_adapter_refuses_unknown_or_missing_text_precision() -> None:
    args = Namespace(fp32_text_enc=False, bf16_text_enc=False, fp16_text_enc=False)
    with pytest.raises(ValueError, match="precision must be an object"):
        _set_text_precision({}, args)
    with pytest.raises(ValueError, match="precision.text must be one of"):
        _set_text_precision({"precision": {"text": "automatic"}}, args)


def test_canonical_sd15_text_precision_is_wired_before_checkpoint_load_and_reported() -> None:
    workloads = json.loads(Path("tools/inference_parity/workloads.json").read_text())["workloads"]
    canonical = next(item for item in workloads if item["id"] == "W0-HARNESS-SD15-TXT2IMG")
    args = Namespace(fp32_text_enc=False, bf16_text_enc=False, fp16_text_enc=False)
    before = vars(args).copy()
    _set_text_precision(canonical["execution"], args)
    assert canonical["execution"]["precision"]["text"] == "float32"
    assert vars(args) != before
    adapter_source = Path("tools/inference_parity/comfyui_adapter.py").read_text()
    main_source = ast.get_source_segment(
        adapter_source,
        next(
            node
            for node in ast.parse(adapter_source).body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        ),
    )
    assert main_source is not None
    assert main_source.index("_set_text_precision(workload, comfy_args)") < main_source.index(
        "load_checkpoint"
    )
    assert "next(clip.cond_stage_model.parameters()).dtype" in main_source
    assert '"text_parameter_dtype": text_parameter_dtype' in main_source

    dinkster_source = Path("tools/inference_parity/dinkster_adapter.py").read_text()
    dinkster_main = ast.get_source_segment(
        dinkster_source,
        next(
            node
            for node in ast.parse(dinkster_source).body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        ),
    )
    assert dinkster_main is not None
    assert dinkster_main.count("text_dtype = torch.float32") == 1
    assert "text_dtype=text_dtype" in dinkster_main
    placement = "runtime.assembled.clip_l.to(device=device, dtype=text_dtype)"
    assert placement in dinkster_main
    assert dinkster_main.index(placement) < dinkster_main.index(
        "next(runtime.assembled.clip_l.parameters()).dtype"
    )
    assert dinkster_main.index(placement) < dinkster_main.index("runtime.encode_text")
    assert "next(runtime.assembled.clip_l.parameters()).dtype" in dinkster_main
    assert '"text_parameter_dtype": text_parameter_dtype' in dinkster_main


def test_adapter_runtime_records_loaded_text_parameter_dtype_and_refuses_bad_values() -> None:
    for workload_id in (
        "W0-HARNESS-SD15-TXT2IMG",
        "W0-SD15-INPAINT",
        "W0-SDXL-VPRED",
    ):
        pinned = {
            "execution": {"precision": {"text": "float32"}},
            "id": workload_id,
        }
        assert _adapter_runtime(
            {"torch": "2.9.1+cu130", "text_parameter_dtype": "float32"},
            pinned,
            "comfyui",
        ) == {"torch": "2.9.1+cu130", "text_parameter_dtype": "float32"}
        with pytest.raises(HarnessError, match="comfyui text_parameter_dtype must be a string"):
            _adapter_runtime({"text_parameter_dtype": 32}, pinned, "comfyui")
        with pytest.raises(HarnessError, match="text_parameter_dtype must be a string"):
            _adapter_runtime({}, pinned, "comfyui")
        with pytest.raises(HarnessError, match="expected float32, got float16"):
            _adapter_runtime({"text_parameter_dtype": "float16"}, pinned, "comfyui")
    canonical = {"execution": {"precision": {"text": "float32"}}, "id": "txt2img"}
    assert _adapter_runtime({}, canonical, "comfyui") == {"torch": "unknown"}
    with pytest.raises(HarnessError, match="torch must be a string"):
        _adapter_runtime({"torch": []}, canonical, "comfyui")


def test_harness_retains_adapter_stdout_and_stderr_text(tmp_path: Path) -> None:
    stdout_path, stderr_path = _write_adapter_logs(
        tmp_path, "comfyui", '{"phase":"warmup"}\n', "Using pytorch attention in VAE\n"
    )
    assert stdout_path == Path("comfyui.stdout.txt")
    assert stderr_path == Path("comfyui.stderr.txt")
    assert (tmp_path / stdout_path).read_text() == '{"phase":"warmup"}\n'
    assert (tmp_path / stderr_path).read_text() == "Using pytorch attention in VAE\n"


def test_adapter_reply_and_timings_refuse_malformed_protocol() -> None:
    with pytest.raises(HarnessError, match="invalid JSON"):
        _adapter_reply("not-json\n", "comfyui", "warmup")
    with pytest.raises(HarnessError, match="must be an object"):
        _adapter_reply("[]\n", "comfyui", "warmup")
    with pytest.raises(HarnessError, match="phase protocol mismatch"):
        _adapter_reply('{"phase":"real"}\n', "comfyui", "warmup")
    with pytest.raises(HarnessError, match="positive integer"):
        _reply_integer({"generation_ns": "1"}, "generation_ns", "comfyui", "real", positive=True)
    with pytest.raises(HarnessError, match="non-negative integer"):
        _reply_integer({"cold_load_ns": -1}, "cold_load_ns", "comfyui", "warmup", positive=False)


class _AdapterProcess:
    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.pid = 2**31 - 1
        self.returncode = returncode

    def wait(self, timeout: int | None = None) -> int:
        del timeout
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def _protocol_workload(output: Path) -> dict[str, Any]:
    return {
        "engines": {"comfyui": {"commit": "HEAD"}},
        "execution": {
            "batch": 1,
            "height": 1,
            "precision": {"text": "float32"},
            "seed": 1,
            "width": 1,
        },
        "id": "W0-SD15-INPAINT",
        "output": str(output),
    }


def _adapter_line(output: Path, phase: str, dtype: str = "float32") -> str:
    return (
        json.dumps(
            {
                "cold_load_ns": 0,
                "generation_ns": 1,
                "output_path": str(output),
                "phase": phase,
                "text_parameter_dtype": dtype,
                "torch": "2.9.1+cu130",
            }
        )
        + "\n"
    )


def _run_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stdout: str,
    *,
    dirty_after_adapter: bool = False,
    extension_dirty_after_adapter: bool = False,
    post_engine_commit: str | None = None,
    post_extension_commit: str | None = None,
    post_harness_commit: str | None = None,
    stderr: str = "backend announce\n",
    returncode: int = 0,
    requests: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 1, 1, 3), dtype=np.float32), allow_pickle=False)
    process = _AdapterProcess(stdout, stderr, returncode)
    extension_root = tmp_path / "extension"
    extension_pins = (
        [
            {
                "commit": "a" * 40,
                "resolved_path": str(extension_root),
            }
        ]
        if extension_dirty_after_adapter or post_extension_commit is not None
        else []
    )
    monkeypatch.setattr(
        harness,
        "validate_pins",
        lambda *args: {"engine_commit": "a" * 40, "extensions": extension_pins},
    )
    monkeypatch.setattr(harness, "hardware_inventory", lambda: {})
    monkeypatch.setattr(
        harness,
        "validate_hardware",
        lambda *args: {"device_ordinal": 0},
    )
    monkeypatch.setattr(harness, "_runner_command", lambda *args: ["adapter"])
    monkeypatch.setattr(harness, "_sample_process", lambda *args: None)

    status_calls = 0
    harness_root = tmp_path / "harness" if post_harness_commit is not None else tmp_path
    harness_head_calls = 0

    def git_output(_root: Path, *args: str) -> str:
        nonlocal harness_head_calls, status_calls
        if args[0] == "status":
            if _root == extension_root:
                return " M extension.py" if extension_dirty_after_adapter else ""
            status_calls += 1
            return " M changed.py" if dirty_after_adapter and status_calls > 1 else ""
        if _root == extension_root:
            return post_extension_commit or "a" * 40
        if _root == harness_root and harness_root != tmp_path:
            harness_head_calls += 1
            assert post_harness_commit is not None
            return "a" * 40 if harness_head_calls == 1 else post_harness_commit
        return post_engine_commit or "a" * 40

    monkeypatch.setattr(
        harness,
        "_git_output",
        git_output,
    )
    monkeypatch.setattr(harness.subprocess, "Popen", lambda *args, **kwargs: process)
    if requests is not None:
        process.pid = 12345
        monkeypatch.setattr(
            harness,
            "validate_timing_hardware",
            lambda *_args: {"device_ordinal": 0},
        )
        monkeypatch.setattr(harness, "_kill_process_group", lambda *_args: None)
    return run_workload(
        _protocol_workload(output),
        "comfyui",
        {"comfyui": tmp_path, "dinkster": harness_root},
        "sha256:" + "a" * 64,
        tmp_path,
        requests=requests,
        timing_device_uuid=None if requests is None else "GPU-test-0",
        timing_hardware=None if requests is None else timing_hardware_pin(),
    )


def test_run_workload_retains_exact_protocol_text_and_relative_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "image.npy"
    stdout = _adapter_line(output, "warmup") + _adapter_line(output, "real")
    record = _run_protocol(monkeypatch, tmp_path, stdout)
    assert record["stdout_path"] == "comfyui.stdout.txt"
    assert record["stderr_path"] == "comfyui.stderr.txt"
    assert (tmp_path / record["stdout_path"]).read_text() == stdout
    assert (tmp_path / record["stderr_path"]).read_text() == "backend announce\n"
    assert record["stdout_digest"] == digest_bytes(stdout.encode())
    assert record["stderr_digest"] == digest_bytes(b"backend announce\n")
    assert record["engine"]["commit"] == "a" * 40
    assert record["harness"]["commit"] == "a" * 40


@pytest.mark.parametrize(
    ("post_engine_commit", "post_harness_commit", "dirty_after_adapter", "match"),
    (
        ("b" * 40, None, False, "comfyui checkout changed"),
        (None, None, True, "comfyui checkout became dirty"),
        (None, "b" * 40, False, "harness checkout changed"),
    ),
)
def test_run_workload_refuses_checkout_mutation_during_adapter_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    post_engine_commit: str | None,
    post_harness_commit: str | None,
    dirty_after_adapter: bool,
    match: str,
) -> None:
    output = tmp_path / "image.npy"
    stdout = _adapter_line(output, "warmup") + _adapter_line(output, "real")
    with pytest.raises(HarnessError, match=match):
        _run_protocol(
            monkeypatch,
            tmp_path,
            stdout,
            post_engine_commit=post_engine_commit,
            post_harness_commit=post_harness_commit,
            dirty_after_adapter=dirty_after_adapter,
        )


@pytest.mark.parametrize(
    ("post_extension_commit", "extension_dirty_after_adapter", "match"),
    (
        ("b" * 40, False, "extension checkout changed"),
        (None, True, "extension checkout became dirty"),
    ),
)
def test_run_workload_refuses_extension_mutation_during_adapter_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    post_extension_commit: str | None,
    extension_dirty_after_adapter: bool,
    match: str,
) -> None:
    output = tmp_path / "image.npy"
    stdout = _adapter_line(output, "warmup") + _adapter_line(output, "real")
    with pytest.raises(HarnessError, match=match):
        _run_protocol(
            monkeypatch,
            tmp_path,
            stdout,
            post_extension_commit=post_extension_commit,
            extension_dirty_after_adapter=extension_dirty_after_adapter,
        )


def test_timing_protocol_does_not_retain_resident_warmup_timing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "image.npy"
    requests = timing_requests(1)
    stdout = "".join(_adapter_line(output, request["phase"]) for request in requests)
    record = _run_protocol(monkeypatch, tmp_path, stdout, requests=requests)

    assert record["command"] == ["adapter"]
    assert record["pid"] == 12345
    assert record["observations"]["resident-warmup"]["metrics"] == {}
    assert "wall_clock" not in record["observations"]["resident-warmup"]
    for phase in ("cold", "warm-1", "warm-2"):
        assert record["observations"][phase]["metrics"]["end_to_end_ns"] > 0
        assert record["observations"][phase]["wall_clock"]["clock"] == "perf_counter_ns"


@pytest.mark.parametrize(
    ("stdout", "returncode", "match"),
    (
        ("not-json\n", 0, "invalid JSON"),
        ("[]\n", 0, "must be an object"),
        ("", 1, "exited during warmup"),
        ("wrong-dtype", 0, "expected float32, got float16"),
        ("extra", 0, "trailing stdout"),
        ("nonzero", 7, "adapter failed \\(7\\)"),
    ),
)
def test_run_workload_retains_logs_on_protocol_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stdout: str,
    returncode: int,
    match: str,
) -> None:
    output = tmp_path / "image.npy"
    if stdout == "wrong-dtype":
        stdout = _adapter_line(output, "warmup", "float16")
    elif stdout == "extra":
        stdout = _adapter_line(output, "warmup") + _adapter_line(output, "real") + "extra\n"
    elif stdout == "nonzero":
        stdout = _adapter_line(output, "warmup") + _adapter_line(output, "real")
    with pytest.raises(HarnessError, match=match):
        _run_protocol(
            monkeypatch,
            tmp_path,
            stdout,
            stderr="failure detail\n",
            returncode=returncode,
        )
    assert (tmp_path / "comfyui.stdout.txt").read_text() == stdout
    assert (tmp_path / "comfyui.stderr.txt").read_text() == "failure detail\n"


@pytest.mark.parametrize(
    ("stdout", "returncode", "match"),
    (
        ("nonzero", 7, "adapter failed \\(7\\)"),
        ("extra", 0, "trailing stdout"),
    ),
)
def test_run_workload_preserves_late_failure_when_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stdout: str,
    returncode: int,
    match: str,
) -> None:
    output = tmp_path / "image.npy"
    stdout = _adapter_line(output, "warmup") + _adapter_line(output, "real")
    if returncode == 0:
        stdout += "extra\n"
    monkeypatch.setattr(harness, "_kill_process_group", lambda process: "denied")
    with pytest.raises(HarnessError, match=match) as raised:
        _run_protocol(monkeypatch, tmp_path, stdout, returncode=returncode)
    assert any("cleanup errors" in note and "denied" in note for note in raised.value.__notes__)


@pytest.mark.parametrize("phase", ("warmup", "real"))
@pytest.mark.parametrize(
    ("value", "match"),
    ((None, "must be a string"), (32, "must be a string"), ("float16", "expected float32")),
)
def test_run_workload_requires_loaded_text_dtype_in_both_phases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    phase: str,
    value: object,
    match: str,
) -> None:
    output = tmp_path / "image.npy"
    replies = []
    for current in ("warmup", "real"):
        reply = json.loads(_adapter_line(output, current))
        if current == phase:
            if value is None:
                reply.pop("text_parameter_dtype")
            else:
                reply["text_parameter_dtype"] = value
        replies.append(json.dumps(reply) + "\n")
    stdout = "".join(replies)
    with pytest.raises(HarnessError, match=match):
        _run_protocol(monkeypatch, tmp_path, stdout)
    assert (tmp_path / "comfyui.stdout.txt").read_text() == stdout


def test_run_workload_refuses_non_string_torch_through_protocol(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "image.npy"
    reply = json.loads(_adapter_line(output, "warmup"))
    reply["torch"] = []
    stdout = json.dumps(reply) + "\n"
    with pytest.raises(HarnessError, match="torch must be a string"):
        _run_protocol(monkeypatch, tmp_path, stdout)
    assert (tmp_path / "comfyui.stdout.txt").read_text() == stdout


class _BrokenInput(io.StringIO):
    def write(self, value: str) -> int:
        del value
        raise BrokenPipeError


def test_run_workload_stops_sampler_and_retains_logs_on_broken_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "image.npy"
    process = _AdapterProcess("", "adapter already exited\n", 1)
    process.stdin = _BrokenInput()
    stopped: list[bool] = []

    def sample(_process: Any, _ordinal: int, stop: Any, _peaks: Any) -> None:
        stopped.append(stop.wait(1))

    monkeypatch.setattr(harness, "validate_pins", lambda *args: {"engine_commit": "a" * 40})
    monkeypatch.setattr(harness, "hardware_inventory", lambda: {})
    monkeypatch.setattr(harness, "validate_hardware", lambda *args: {"device_ordinal": 0})
    monkeypatch.setattr(harness, "_runner_command", lambda *args: ["adapter"])
    monkeypatch.setattr(harness, "_sample_process", sample)
    monkeypatch.setattr(
        harness,
        "_git_output",
        lambda _root, *args: "" if args[0] == "status" else "a" * 40,
    )
    monkeypatch.setattr(harness.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(HarnessError, match="input pipe failed"):
        run_workload(
            _protocol_workload(output),
            "comfyui",
            {"comfyui": tmp_path, "dinkster": tmp_path},
            "sha256:" + "a" * 64,
            tmp_path,
        )
    assert stopped == [True]
    assert (tmp_path / "comfyui.stderr.txt").read_text() == "adapter already exited\n"


def test_run_workload_drains_large_stderr_without_deadlock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 1, 1, 3), dtype=np.float32), allow_pickle=False)
    reply = {
        "cold_load_ns": 0,
        "generation_ns": 1,
        "output_path": str(output),
        "text_parameter_dtype": "float32",
        "torch": "test",
    }
    code = (
        "import json,sys\n"
        f"reply={reply!r}\n"
        "for phase in ('warmup','real'):\n"
        " sys.stdin.readline()\n"
        " sys.stderr.write('x'*200000); sys.stderr.flush()\n"
        " reply['phase']=phase; print(json.dumps(reply),flush=True)\n"
    )
    monkeypatch.setattr(harness, "validate_pins", lambda *args: {"engine_commit": "a" * 40})
    monkeypatch.setattr(harness, "hardware_inventory", lambda: {})
    monkeypatch.setattr(harness, "validate_hardware", lambda *args: {"device_ordinal": 0})
    monkeypatch.setattr(harness, "_runner_command", lambda *args: [sys.executable, "-c", code])
    monkeypatch.setattr(harness, "_sample_process", lambda *args: None)
    monkeypatch.setattr(
        harness,
        "_git_output",
        lambda _root, *args: "" if args[0] == "status" else "a" * 40,
    )
    record = run_workload(
        _protocol_workload(output),
        "comfyui",
        {"comfyui": tmp_path, "dinkster": tmp_path},
        "sha256:" + "a" * 64,
        tmp_path,
    )
    assert record["stderr_digest"] == digest_bytes(("x" * 400000).encode())
    assert (tmp_path / record["stderr_path"]).stat().st_size == 400000


def test_run_workload_times_out_silent_adapter_and_retains_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "image.npy"
    ready = tmp_path / "adapter.ready"
    code = (
        "import pathlib,sys,time; sys.stdout.write('partial'); sys.stdout.flush(); "
        "sys.stderr.write('waiting\\n'); sys.stderr.flush(); "
        f"pathlib.Path({str(ready)!r}).touch(); time.sleep(60)"
    )
    log_dir = tmp_path / "new" / "logs"
    monkeypatch.setattr(harness, "validate_pins", lambda *args: {"engine_commit": "a" * 40})
    monkeypatch.setattr(harness, "hardware_inventory", lambda: {})
    monkeypatch.setattr(harness, "validate_hardware", lambda *args: {"device_ordinal": 0})
    monkeypatch.setattr(harness, "_runner_command", lambda *args: [sys.executable, "-c", code])
    monkeypatch.setattr(harness, "_sample_process", lambda *args: None)
    monkeypatch.setattr(
        harness,
        "_git_output",
        lambda _root, *args: "" if args[0] == "status" else "a" * 40,
    )
    monkeypatch.setattr(harness, "ADAPTER_REPLY_TIMEOUT_SECONDS", 0.1)

    real_kill = harness._kill_process_group

    def kill_after_adapter_wrote(process: Any) -> str | None:
        # The 0.1s reply timeout can fire before the child interpreter
        # finishes starting; killing then would leave the pipes empty.
        # Wait for proof the adapter flushed its output first.
        deadline = time.monotonic() + 30
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        return real_kill(process)

    monkeypatch.setattr(harness, "_kill_process_group", kill_after_adapter_wrote)
    with pytest.raises(HarnessError, match="timed out during warmup"):
        run_workload(
            _protocol_workload(output),
            "comfyui",
            {"comfyui": tmp_path, "dinkster": tmp_path},
            "sha256:" + "a" * 64,
            log_dir,
        )
    assert (log_dir / "comfyui.stdout.txt").read_text() == "partial"
    assert (log_dir / "comfyui.stderr.txt").read_text() == "waiting\n"


def test_run_workload_terminates_successful_adapter_descendants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "image.npy"
    np.save(output, np.zeros((1, 1, 1, 3), dtype=np.float32), allow_pickle=False)
    reply = {
        "cold_load_ns": 0,
        "generation_ns": 1,
        "output_path": str(output),
        "text_parameter_dtype": "float32",
        "torch": "test",
    }
    code = (
        "import json,subprocess,sys\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        "print(f'child={child.pid}',file=sys.stderr,flush=True)\n"
        f"reply={reply!r}\n"
        "for phase in ('warmup','real'):\n"
        " sys.stdin.readline(); reply['phase']=phase; print(json.dumps(reply),flush=True)\n"
    )
    monkeypatch.setattr(harness, "validate_pins", lambda *args: {"engine_commit": "a" * 40})
    monkeypatch.setattr(harness, "hardware_inventory", lambda: {})
    monkeypatch.setattr(harness, "validate_hardware", lambda *args: {"device_ordinal": 0})
    monkeypatch.setattr(harness, "_runner_command", lambda *args: [sys.executable, "-c", code])
    monkeypatch.setattr(harness, "_sample_process", lambda *args: None)
    monkeypatch.setattr(
        harness,
        "_git_output",
        lambda _root, *args: "" if args[0] == "status" else "a" * 40,
    )
    record = run_workload(
        _protocol_workload(output),
        "comfyui",
        {"comfyui": tmp_path, "dinkster": tmp_path},
        "sha256:" + "a" * 64,
        tmp_path,
    )
    child_pid = int((tmp_path / record["stderr_path"]).read_text().split("=", 1)[1])
    for _ in range(50):
        if not Path(f"/proc/{child_pid}").exists():
            break
        time.sleep(0.01)
    assert not Path(f"/proc/{child_pid}").exists()


class _Buffer:
    def __init__(self, device: str) -> None:
        self.device = device


class _ModelSampling:
    def __init__(self, device: str) -> None:
        self.sigmas = _Buffer(device)
        self.log_sigmas = _Buffer(device)
        self.moves: list[str] = []

    def to(self, device: str) -> _ModelSampling:
        self.moves.append(device)
        self.sigmas.device = device
        self.log_sigmas.device = device
        return self


@pytest.mark.parametrize("raises", (False, True))
def test_comfyui_normal_sigmas_restore_all_buffers(raises: bool) -> None:
    model_sampling = _ModelSampling("cuda:0")
    result = object()

    def calculate_sigmas(model: _ModelSampling, scheduler: str, steps: int) -> object:
        assert model.sigmas.device == "cpu"
        assert model.log_sigmas.device == "cpu"
        assert scheduler == "normal"
        assert steps == 20
        if raises:
            raise RuntimeError("schedule failed")
        return result

    calibrated = _normal_sigmas_on_cpu(calculate_sigmas, "cpu")
    if raises:
        with pytest.raises(RuntimeError, match="schedule failed"):
            calibrated(model_sampling, "normal", 20)
    else:
        assert calibrated(model_sampling, "normal", 20) is result
    assert model_sampling.sigmas.device == "cuda:0"
    assert model_sampling.log_sigmas.device == "cuda:0"
    assert model_sampling.moves == ["cpu", "cuda:0"]


def test_comfyui_non_normal_sigmas_pass_through_without_moves() -> None:
    model_sampling = _ModelSampling("cuda:0")
    result = object()
    calls: list[tuple[_ModelSampling, str, int]] = []

    def calculate_sigmas(model: _ModelSampling, scheduler: str, steps: int) -> object:
        calls.append((model, scheduler, steps))
        return result

    calibrated = _normal_sigmas_on_cpu(calculate_sigmas, "cpu")
    assert calibrated(model_sampling, "karras", 7) is result
    assert calls == [(model_sampling, "karras", 7)]
    assert model_sampling.moves == []


def _nvfp4_request(tmp_path: Path) -> dict[str, Any]:
    contract = nvfp4_adapter.ADAPTER_CONTRACT_SHA256
    return {
        "schema_version": 1,
        "engine": "dinkster",
        "source": {"repo_path": str(tmp_path), "sha": "a" * 40, "tree": "b" * 40},
        "status_api": {
            "module": "dinkster_inference_torch._nvfp4_diagnostics",
            "symbol": "nvfp4_runtime_status",
            "contract_sha256": "c" * 64,
        },
        "artifacts": {
            role: {
                "path": str(tmp_path / f"misleading-{index}.bin"),
                "size": index + 1,
                "sha256": f"{index + 1:064x}",
                "blake3": f"blake3:{index + 11:064x}",
            }
            for index, role in enumerate(nvfp4_adapter.ROLES)
        },
        "gpu": {"index": 7, "uuid": "request-owned-gpu"},
        "workload": dict(nvfp4_adapter.WORKLOAD),
        "expected_evidence": json.loads(json.dumps(nvfp4_adapter.EXPECTED_EVIDENCE)),
        "adapter": {
            "module": "adapter.module",
            "symbol": "main",
            "contract_sha256": contract,
            "sha256": "d" * 64,
        },
    }


def test_nvfp4_adapter_request_is_exact_and_fail_closed(tmp_path: Path) -> None:
    request = _nvfp4_request(tmp_path)
    launcher_evidence = {
        "nvfp4_linear_layers": 152,
        "fp8_linear_layers": 114,
        "route_pre_sm10_calls": 304,
        "dequantize_nvfp4_calls": 304,
        "f_linear_nvfp4_calls": 304,
        "native_nvfp4_calls": 0,
        "quantize_nvfp4_calls": 0,
        "scaled_mm_nvfp4_calls": 0,
        "fallback_calls": 0,
        "error_calls": 0,
        "selected_backend": "dequantize_nvfp4_plus_f_linear",
        "complete_dequantize_and_f_linear_accounting": True,
        "representative_direct_kitchen_exact_equality": True,
        "option_a_phase_local_object_identity_exact": True,
        "option_a_cross_transition_bytes_and_logical_metadata_exact": True,
        "option_a_final_pre_release_exact": True,
        "option_a_single_handle_runtime_assembled_recorder_identity": True,
        "warmup_real_output_bytes_identical": True,
        "metrics": [
            "load_seconds",
            "generation_seconds",
            "peak_rss_bytes",
            "peak_vram_bytes",
        ],
        "cleanup": [
            "runtime_released",
            "cpu_offload_complete",
            "adapter_process_exited",
            "no_residual_gpu_process",
        ],
    }
    assert request["expected_evidence"] == nvfp4_adapter.EXPECTED_EVIDENCE == launcher_evidence
    assert set(request["expected_evidence"]) == {
        "nvfp4_linear_layers",
        "fp8_linear_layers",
        "route_pre_sm10_calls",
        "dequantize_nvfp4_calls",
        "f_linear_nvfp4_calls",
        "native_nvfp4_calls",
        "quantize_nvfp4_calls",
        "scaled_mm_nvfp4_calls",
        "fallback_calls",
        "error_calls",
        "selected_backend",
        "complete_dequantize_and_f_linear_accounting",
        "representative_direct_kitchen_exact_equality",
        "option_a_phase_local_object_identity_exact",
        "option_a_cross_transition_bytes_and_logical_metadata_exact",
        "option_a_final_pre_release_exact",
        "option_a_single_handle_runtime_assembled_recorder_identity",
        "warmup_real_output_bytes_identical",
        "metrics",
        "cleanup",
    }
    assert nvfp4_adapter.validate_request(request) is request
    for mutation in (
        lambda value: value.update(extra=True),
        lambda value: value["workload"].update(seed=43),
        lambda value: value["artifacts"].pop("vae"),
        lambda value: value["artifacts"]["diffusion"].update(blake3="0" * 64),
        lambda value: value["adapter"].update(contract_sha256="0" * 64),
        lambda value: value["expected_evidence"].pop("dequantize_nvfp4_calls"),
        lambda value: value["expected_evidence"].update(dequantize_nvfp4_calls=303),
        lambda value: value["expected_evidence"].update(nvfp4_linear_layers=151),
        lambda value: value["expected_evidence"].update(quantize_nvfp4_calls=1),
        lambda value: value["expected_evidence"].update(
            complete_dequantize_and_f_linear_accounting=False
        ),
        lambda value: value["expected_evidence"].update(metrics=["load_seconds"]),
        lambda value: value["expected_evidence"].update(
            packed_qdata_and_all_scale_digests_unchanged=True
        ),
        lambda value: value["expected_evidence"].update(parameter_identity_unchanged=True),
        lambda value: value["status_api"].update(module="wrong.module"),
        lambda value: value["status_api"].update(symbol="wrong_symbol"),
    ):
        candidate = json.loads(json.dumps(request))
        mutation(candidate)
        with pytest.raises(nvfp4_adapter.AdapterError):
            nvfp4_adapter.validate_request(candidate)


def test_nvfp4_adapter_builds_fixed_role_assets_without_filename_authority(
    tmp_path: Path,
) -> None:
    request = _nvfp4_request(tmp_path)

    class Asset:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    assets, resolver = nvfp4_adapter._assets(request, SimpleNamespace(AssetRef=Asset))
    assert tuple(assets) == nvfp4_adapter.ROLES
    for role in nvfp4_adapter.ROLES:
        pin = request["artifacts"][role]
        assert assets[role].name == role
        assert assets[role].digest == pin["blake3"]
        assert assets[role].size == pin["size"]
        assert assets[role].resolver is resolver
        assert resolver.resolve(pin["blake3"]) == Path(pin["path"])
    assert resolver.resolve("f" * 64) is None


def test_nvfp4_adapter_runtime_imports_public_native_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_compat_comfy import native_arm

    torch = ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", torch)

    api = nvfp4_adapter._runtime_imports()

    assert api.torch is torch
    assert api.load_native_runtime_handle is native_arm.load_native_runtime_handle


def test_nvfp4_adapter_uses_canonical_native_runtime_loader(tmp_path: Path) -> None:
    request = _nvfp4_request(tmp_path)
    events: list[object] = []

    class Asset:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class Handle:
        def __init__(self) -> None:
            self.released = False

        def terminal_release(self) -> None:
            self.released = True

    handle = Handle()

    class Context:
        def __init__(self, arm: str, identity: None, fp8_matmul: bool) -> None:
            events.append(("context", arm, identity, fp8_matmul))

    @contextmanager
    def use_context(_context: Any) -> Any:
        events.append("enter-context")
        yield
        events.append("exit-context")

    def load_native_runtime_handle(assets: dict[str, Any]) -> Handle:
        events.append(("load", tuple(assets)))
        return handle

    api = SimpleNamespace(
        AssetRef=Asset,
        ExecutionContext=Context,
        use_execution_context=use_context,
        load_native_runtime_handle=load_native_runtime_handle,
        NativeRuntimeHandle=Handle,
    )
    assert nvfp4_adapter._canonical_load(request, api) is handle
    assert [event[0] if isinstance(event, tuple) else event for event in events] == [
        "context",
        "enter-context",
        "load",
        "exit-context",
    ]
    assert events[0] == ("context", "compat@native", None, False)
    assert events[2] == ("load", nvfp4_adapter.ROLES)

    api.load_native_runtime_handle = lambda *_args: object()
    with pytest.raises(nvfp4_adapter.AdapterError, match="unexpected handle"):
        nvfp4_adapter._canonical_load(request, api)


def _nvfp4_snapshot(
    lifetime: dict[str, int],
    terminals: tuple[str, ...] = (),
    active: int = 0,
    counters: dict[str, int] | None = None,
) -> SimpleNamespace:
    invocation = counters or {"route_pre_sm10": 1, "dequantize_success": 1}
    return SimpleNamespace(
        lifetime=lifetime,
        completed=tuple(
            SimpleNamespace(terminal=value, counters=dict(invocation)) for value in terminals
        ),
        active=active,
    )


def test_nvfp4_adapter_status_delta_requires_exact_completed_ada_route() -> None:
    before = _nvfp4_snapshot({"route_pre_sm10": 100, "dequantize_success": 100}, ("success",) * 16)
    after = _nvfp4_snapshot({"route_pre_sm10": 404, "dequantize_success": 404}, ("success",) * 16)
    assert nvfp4_adapter._counter_delta(before, after, 304)["dequantize_success"] == 304
    for invalid in (
        _nvfp4_snapshot({"route_pre_sm10": 404, "dequantize_success": 403}, ("success",) * 16),
        _nvfp4_snapshot(
            {
                "route_pre_sm10": 404,
                "dequantize_success": 404,
                "quantize_success": 1,
            },
            ("success",) * 16,
        ),
        _nvfp4_snapshot(
            {
                "route_pre_sm10": 404,
                "dequantize_success": 404,
                "route_backend_fallback": 1,
            },
            ("success",) * 16,
        ),
        _nvfp4_snapshot(
            {"route_pre_sm10": 404, "dequantize_success": 404},
            ("success",) * 15 + ("error",),
        ),
        _nvfp4_snapshot(
            {"route_pre_sm10": 404, "dequantize_success": 404},
            ("success",) * 16,
            active=1,
        ),
        _nvfp4_snapshot(
            {"route_pre_sm10": 404, "dequantize_success": 404},
            ("success",) * 16,
            counters={"route_native": 1},
        ),
    ):
        with pytest.raises(nvfp4_adapter.AdapterError):
            nvfp4_adapter._counter_delta(before, invalid, 304)


def test_nvfp4_adapter_allows_transition_replacement_but_refuses_drift_and_loaded_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _nvfp4_request(tmp_path)
    recorder = object()
    assembled = SimpleNamespace(diffusion=SimpleNamespace(_nvfp4_diagnostics=recorder))
    runtime = SimpleNamespace(assembled=assembled)
    sources = tuple(
        SimpleNamespace(
            role=role,
            source=SimpleNamespace(digest=request["artifacts"][role]["blake3"]),
        )
        for role in sorted(nvfp4_adapter.ROLES)
    )
    handle = SimpleNamespace(runtime=runtime, recipe=SimpleNamespace(sources=sources))
    semantic = {
        "layer": {
            "geometry": (8, 4),
            "tensors": {
                "weight": {
                    "key": "layer.weight",
                    "shape": (8, 2),
                    "dtype": "torch.uint8",
                    "size": 16,
                    "sha256": "a" * 64,
                }
            },
        }
    }
    current = semantic
    monkeypatch.setattr(
        nvfp4_adapter, "_transition_snapshot", lambda _runtime, _progress=None: current
    )
    identity = nvfp4_adapter._ProcessIdentity.capture(handle, request)
    current = dict(semantic)
    identity.check(request, transition=True)


def test_nvfp4_adapter_process_identity_and_transition_drift_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _nvfp4_request(tmp_path)
    recorder = object()
    assembled = SimpleNamespace(diffusion=SimpleNamespace(_nvfp4_diagnostics=recorder))
    runtime = SimpleNamespace(assembled=assembled)
    sources = tuple(
        SimpleNamespace(
            role=role,
            source=SimpleNamespace(digest=request["artifacts"][role]["blake3"]),
        )
        for role in sorted(nvfp4_adapter.ROLES)
    )
    handle = SimpleNamespace(runtime=runtime, recipe=SimpleNamespace(sources=sources))
    baseline = {
        "layer": {
            "geometry": (8, 4),
            "tensors": {
                "weight": {
                    "key": "layer.weight",
                    "shape": (8, 2),
                    "dtype": "torch.uint8",
                    "size": 16,
                    "sha256": "a" * 64,
                }
            },
        }
    }
    state = baseline
    monkeypatch.setattr(
        nvfp4_adapter, "_transition_snapshot", lambda _runtime, _progress=None: state
    )
    identity = nvfp4_adapter._ProcessIdentity.capture(handle, request)
    for field, value in (
        ("geometry", (9, 4)),
        ("key", "other.weight"),
        ("shape", (9, 2)),
        ("dtype", "torch.int8"),
        ("sha256", "b" * 64),
    ):
        layer = baseline["layer"]
        tensor = layer["tensors"]["weight"]
        state = {
            "layer": {
                "geometry": value if field == "geometry" else layer["geometry"],
                "tensors": {
                    "weight": {
                        **tensor,
                        **({field: value} if field != "geometry" else {}),
                    }
                },
            }
        }
        with pytest.raises(nvfp4_adapter.AdapterError, match="transition state"):
            identity.check(request, transition=True)
    state = identity.baseline
    handle.runtime = SimpleNamespace(assembled=assembled)
    with pytest.raises(nvfp4_adapter.AdapterError, match="handle/runtime/assembled"):
        identity.check(request)
    handle.runtime = runtime
    assembled.diffusion._nvfp4_diagnostics = object()
    with pytest.raises(nvfp4_adapter.AdapterError, match="recorder identity"):
        identity.check(request)


def test_nvfp4_adapter_loaded_identity_and_terminal_release_are_literal() -> None:
    shared = object()
    nvfp4_adapter._require_same_loaded_objects({"q": shared}, {"q": shared})
    with pytest.raises(nvfp4_adapter.AdapterError, match="object identity"):
        nvfp4_adapter._require_same_loaded_objects({"q": shared}, {"q": object()})

    class Handle:
        released = False
        releases = 0

        def terminal_release(self) -> None:
            self.releases += 1
            self.released = True

    handle = Handle()
    nvfp4_adapter._terminal_release(handle)
    assert handle.released is True
    assert handle.releases == 1


class _CtypesTensor:
    def __init__(
        self,
        data: bytes,
        *,
        offset: int = 0,
        contiguous: _CtypesTensor | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: str = "torch.uint8",
    ) -> None:
        self._buffer = ctypes.create_string_buffer(b"x" * offset + data)
        self._offset = offset
        self._data = data
        self._contiguous = self if contiguous is None else contiguous
        self.shape = (len(data),) if shape is None else shape
        self.dtype = dtype

    def detach(self) -> _CtypesTensor:
        return self

    def contiguous(self) -> _CtypesTensor:
        return self._contiguous

    def cpu(self) -> _CtypesTensor:
        return self

    def data_ptr(self) -> int:
        return ctypes.addressof(self._buffer) + self._offset

    def numel(self) -> int:
        return len(self._data)

    def element_size(self) -> int:
        return 1

    def untyped_storage(self) -> NoReturn:
        raise AssertionError("slow storage iteration must not be called")


def test_nvfp4_adapter_raw_tensor_bytes_bulk_copies_exact_contiguous_view() -> None:
    offset = _CtypesTensor(b"offset-data", offset=7)
    normalized = _CtypesTensor(b"normalized")
    noncontiguous = _CtypesTensor(b"ignored", contiguous=normalized)
    empty = _CtypesTensor(b"", offset=3)

    assert nvfp4_adapter._raw_tensor_bytes(offset) == b"offset-data"
    assert nvfp4_adapter._raw_tensor_bytes(noncontiguous) == b"normalized"
    assert nvfp4_adapter._raw_tensor_bytes(empty) == b""


def test_nvfp4_adapter_real_transition_snapshot_covers_bytes_and_semantics() -> None:
    layer = type(
        "Nvfp4Linear",
        (),
        {
            "out_features": 8,
            "in_features": 4,
            "weight": _CtypesTensor(b"qdata", shape=(8, 2)),
            "weight_scale": _CtypesTensor(b"block", shape=(128, 4), dtype="torch.float8_e4m3fn"),
            "weight_scale_2": _CtypesTensor(b"tensor", shape=(), dtype="torch.float32"),
            "input_scale": _CtypesTensor(b"input", shape=(), dtype="torch.float32"),
            "pre_quant_scale": None,
        },
    )()
    diffusion = SimpleNamespace(named_modules=lambda: iter((("", object()), ("first", layer))))
    runtime = SimpleNamespace(assembled=SimpleNamespace(diffusion=diffusion))
    snapshot = nvfp4_adapter._transition_snapshot(runtime)
    assert snapshot["first"]["geometry"] == (8, 4)
    assert snapshot["first"]["tensors"]["weight"] == {
        "key": "first.weight",
        "shape": (8, 2),
        "dtype": "torch.uint8",
        "size": 5,
        "sha256": hashlib.sha256(b"qdata").hexdigest(),
    }
    assert set(snapshot["first"]["tensors"]) == {
        "weight",
        "weight_scale",
        "weight_scale_2",
        "input_scale",
    }
    baseline = nvfp4_adapter._transition_result(snapshot)
    assert len(baseline) == 6
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef") for value in baseline.values()
    )
    mutations = (
        ("packed_qdata_bytes_sha256", "weight", "sha256", "1" * 64),
        ("all_scale_bytes_sha256", "weight_scale", "sha256", "2" * 64),
        ("logical_keys_sha256", "weight", "key", "renamed.weight"),
        ("logical_shapes_sha256", "weight", "shape", (9, 2)),
        ("logical_dtypes_sha256", "weight", "dtype", "torch.int8"),
    )
    for expected_field, tensor_name, field, value in mutations:
        changed = deepcopy(snapshot)
        changed["first"]["tensors"][tensor_name][field] = value
        result = nvfp4_adapter._transition_result(changed)
        assert {key for key in baseline if baseline[key] != result[key]} == {expected_field}
    changed = deepcopy(snapshot)
    changed["first"]["geometry"] = (9, 4)
    result = nvfp4_adapter._transition_result(changed)
    assert {key for key in baseline if baseline[key] != result[key]} == {"logical_geometry_sha256"}


def test_nvfp4_adapter_representative_uses_direct_kitchen_dequant_and_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tensor(_CtypesTensor):
        def __getitem__(self, _item: object) -> Tensor:
            return self

    runtime_output = Tensor(b"exact")
    dequantized = Tensor(b"weight")
    calls: list[tuple[object, ...]] = []

    def dequantize(*args: object, **kwargs: object) -> Tensor:
        calls.append((*args, kwargs))
        return dequantized

    monkeypatch.setitem(
        sys.modules,
        "dinkster_kitchen",
        SimpleNamespace(dequantize_nvfp4=dequantize),
    )
    torch = SimpleNamespace(
        nn=SimpleNamespace(
            functional=SimpleNamespace(
                linear=lambda value, weight, bias: (
                    calls.append(("linear", value, weight, bias)) or runtime_output
                )
            )
        )
    )
    layer = SimpleNamespace(
        pre_quant_scale=None,
        weight=object(),
        weight_scale_2=object(),
        weight_scale=object(),
        compute_dtype="bfloat16",
        out_features=8,
        in_features=4,
        bias=None,
    )
    result = nvfp4_adapter._representative(
        "first", layer, {"input": Tensor(b"input"), "output": runtime_output}, torch
    )
    assert result["exact"] is True
    assert result["direct_sha256"] == hashlib.sha256(b"exact").hexdigest()
    assert result["runtime_sha256"] == result["direct_sha256"]
    assert calls[-1][0] == "linear"


def test_nvfp4_adapter_success_result_has_exact_phases_and_releases_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _nvfp4_request(tmp_path)

    class Blob:
        def permute(self, *_axes: int) -> Blob:
            return self

        def detach(self) -> Blob:
            return self

        def to(self, **_kwargs: object) -> Blob:
            return self

    class Layer:
        def register_forward_hook(self, _hook: Any) -> SimpleNamespace:
            return SimpleNamespace(remove=lambda: None)

    class Runtime:
        def encode_text(self, text: str) -> str:
            return text

        def sample(self, *_args: object, **_kwargs: object) -> Blob:
            return Blob()

        def decode_latent(self, _sampled: Blob) -> Blob:
            return Blob()

    class Mechanism:
        def __init__(self, handle: Handle) -> None:
            self.handle = handle

        def loaded_bytes(self) -> int:
            return 1 if self.handle.loaded else 0

    class Handle:
        def __init__(self) -> None:
            self.runtime = Runtime()
            self.load_device = "cuda:0"
            self.loaded = False
            self.released = False
            self.releases = 0
            self.mechanisms = (Mechanism(self),)

        @contextmanager
        def stage(self, role: str) -> Any:
            events.append(f"stage:{role}")
            self.loaded = True
            yield

        def advisory_unload(self) -> None:
            self.loaded = False

        def terminal_release(self) -> None:
            self.releases += 1
            self.released = True

    class Torch:
        float32 = "float32"
        bfloat16 = "bfloat16"
        cuda = SimpleNamespace(
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 12,
        )

        @staticmethod
        @contextmanager
        def inference_mode() -> Any:
            yield

        @staticmethod
        def zeros(*_args: object, **_kwargs: object) -> Blob:
            return Blob()

    semantic = {
        "layer": {
            "geometry": (8, 4),
            "tensors": {
                "weight": {
                    "key": "layer.weight",
                    "shape": (8, 2),
                    "dtype": "torch.uint8",
                    "size": 1,
                    "sha256": "a" * 64,
                },
                "weight_scale": {
                    "key": "layer.weight_scale",
                    "shape": (128, 4),
                    "dtype": "torch.float8_e4m3fn",
                    "size": 1,
                    "sha256": "b" * 64,
                },
                "weight_scale_2": {
                    "key": "layer.weight_scale_2",
                    "shape": (),
                    "dtype": "torch.float32",
                    "size": 1,
                    "sha256": "e" * 64,
                },
            },
        }
    }

    class Identity:
        def __init__(self, handle: Handle) -> None:
            self.handle = handle
            self.runtime = handle.runtime
            self.assembled = object()
            self.recorder = object()
            self.baseline = semantic

        def check(
            self,
            _request: Any,
            _progress: object = None,
            *,
            transition: bool = False,
        ) -> dict[str, Any]:
            del transition
            return semantic

        def status(self, _request: Any, _progress: object = None) -> SimpleNamespace:
            return SimpleNamespace()

    handle = Handle()
    layer = Layer()
    events: list[str] = []
    captured_identities: list[Identity] = []
    expected_counts: list[int] = []
    monkeypatch.setattr(
        nvfp4_adapter,
        "_runtime_imports",
        lambda: SimpleNamespace(
            torch=Torch,
            SamplingGuidance=lambda uncond, scale: SimpleNamespace(uncond=uncond, scale=scale),
        ),
    )
    monkeypatch.setattr(nvfp4_adapter, "_canonical_load", lambda *_args: handle)

    def capture_identity(
        _cls: object, value: Handle, _request: object, _progress: object = None
    ) -> Identity:
        events.append("capture")
        identity = Identity(value)
        captured_identities.append(identity)
        return identity

    monkeypatch.setattr(nvfp4_adapter._ProcessIdentity, "capture", classmethod(capture_identity))
    monkeypatch.setattr(
        nvfp4_adapter,
        "_inventory",
        lambda _runtime: {"nvfp4_linear_layers": 152, "fp8_linear_layers": 114},
    )
    monkeypatch.setattr(nvfp4_adapter, "_nvfp4_modules", lambda _runtime: ([("layer", layer)], []))
    warmup_objects = {
        "layer.weight": object(),
        "layer.weight_scale": object(),
        "layer.weight_scale_2": object(),
        "layer.input_scale": object(),
        "layer.pre_quant_scale": object(),
    }
    real_objects = {
        "layer.weight": object(),
        "layer.weight_scale": object(),
        "layer.weight_scale_2": object(),
        "layer.input_scale": object(),
        "layer.pre_quant_scale": object(),
    }
    loaded_calls = 0

    def loaded_identity(_runtime: object) -> tuple[str, dict[str, object]]:
        nonlocal loaded_calls
        phase = loaded_calls // 2
        loaded_calls += 1
        return (("c" if phase == 0 else "d") * 64, (warmup_objects, real_objects)[phase])

    monkeypatch.setattr(
        nvfp4_adapter,
        "_loaded_identity",
        loaded_identity,
    )
    monkeypatch.setattr(
        nvfp4_adapter,
        "_counter_delta",
        lambda _before, _after, expected: (
            expected_counts.append(expected)
            or {
                "route_pre_sm10": expected,
                "dequantize_success": expected,
                "quantize_success": 0,
                "scaled_mm_success": 0,
            }
        ),
    )
    monkeypatch.setattr(
        nvfp4_adapter,
        "_representative",
        lambda *_args: {
            "layer_id": "layer",
            "direct_sha256": "d" * 64,
            "runtime_sha256": "d" * 64,
            "exact": True,
        },
    )
    monkeypatch.setattr(nvfp4_adapter, "_raw_tensor_bytes", lambda _tensor: b"image")
    monkeypatch.setattr(nvfp4_adapter, "_peak_rss_bytes", lambda: 34)
    result = nvfp4_adapter._execute(request)
    assert result.keys() == {
        "schema_version",
        "engine",
        "status",
        "adapter",
        "status_api",
        "phases",
        "state",
    }
    assert [phase["name"] for phase in result["phases"]] == ["warmup", "real"]
    expected_delta = {
        "route_pre_sm10_calls": 304,
        "dequantize_nvfp4_calls": 304,
        "f_linear_nvfp4_calls": 304,
        "native_nvfp4_calls": 0,
        "quantize_nvfp4_calls": 0,
        "scaled_mm_nvfp4_calls": 0,
        "fallback_calls": 0,
        "error_calls": 0,
        "selected_backend": "dequantize_nvfp4_plus_f_linear",
        "complete": True,
    }
    assert [phase["adapter_lifetime_deltas"] for phase in result["phases"]] == [
        expected_delta,
        expected_delta,
    ]
    assert all("operations" not in phase and "state" not in phase for phase in result["phases"])
    state = result["state"]
    assert state.keys() == {
        "phase_local_loaded_object_identity",
        "cross_transition",
        "final_pre_release",
        "single_object_identity",
    }
    phase_state = state["phase_local_loaded_object_identity"]
    assert phase_state.keys() == {"warmup", "real"}
    endpoint_keys = {
        "qdata_object_identity_sha256",
        "block_scale_object_identity_sha256",
        "tensor_scale_object_identity_sha256",
        "all_scale_object_identity_sha256",
    }
    for phase_name in ("warmup", "real"):
        for endpoint in ("before", "after"):
            endpoint_state = phase_state[phase_name][endpoint]
            assert endpoint_state.keys() == endpoint_keys
            assert all(
                len(value) == 64 and set(value) <= set("0123456789abcdef")
                for value in endpoint_state.values()
            )

    def keyed_identity_digest(objects: dict[str, object], names: set[str]) -> str:
        digest = hashlib.sha256()
        for key, value in objects.items():
            if key.rsplit(".", 1)[-1] not in names:
                continue
            encoded_key = key.encode("ascii")
            encoded_identity = str(id(value)).encode("ascii")
            digest.update(len(encoded_key).to_bytes(8, "big"))
            digest.update(encoded_key)
            digest.update(len(encoded_identity).to_bytes(8, "big"))
            digest.update(encoded_identity)
        return digest.hexdigest()

    scale_names = {"weight_scale", "weight_scale_2", "input_scale", "pre_quant_scale"}
    assert phase_state["warmup"]["before"]["all_scale_object_identity_sha256"] == (
        keyed_identity_digest(warmup_objects, scale_names)
    )
    assert phase_state["real"]["before"]["all_scale_object_identity_sha256"] == (
        keyed_identity_digest(real_objects, scale_names)
    )
    assert phase_state["warmup"]["before"] == phase_state["warmup"]["after"]
    assert phase_state["real"]["before"] == phase_state["real"]["after"]
    assert phase_state["warmup"]["before"] != phase_state["real"]["before"]
    assert phase_state["warmup"]["exact"] is phase_state["real"]["exact"] is True
    transition = state["cross_transition"]
    assert transition["before"] == transition["after"] == state["final_pre_release"]
    assert transition["exact"] is True
    assert set(transition["before"]) == {
        "packed_qdata_bytes_sha256",
        "all_scale_bytes_sha256",
        "logical_keys_sha256",
        "logical_shapes_sha256",
        "logical_dtypes_sha256",
        "logical_geometry_sha256",
    }
    assert state["single_object_identity"].keys() == {
        "handle_identity_sha256",
        "runtime_identity_sha256",
        "assembled_identity_sha256",
        "recorder_identity_sha256",
        "exact",
    }
    assert state["single_object_identity"]["exact"] is True
    captured_identity = captured_identities[0]
    assert state["single_object_identity"] == {
        "handle_identity_sha256": hashlib.sha256(str(id(handle)).encode("ascii")).hexdigest(),
        "runtime_identity_sha256": hashlib.sha256(
            str(id(handle.runtime)).encode("ascii")
        ).hexdigest(),
        "assembled_identity_sha256": hashlib.sha256(
            str(id(captured_identity.assembled)).encode("ascii")
        ).hexdigest(),
        "recorder_identity_sha256": hashlib.sha256(
            str(id(captured_identity.recorder)).encode("ascii")
        ).hexdigest(),
        "exact": True,
    }
    assert expected_counts == [304, 304]
    assert events[:2] == ["capture", "stage:text"]
    assert handle.released is True
    assert handle.releases == 1


def test_nvfp4_adapter_result_is_canonical_ascii_and_atomic(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    value = {"z": "ASCII", "a": [1, True]}
    nvfp4_adapter._atomic_write(output, value)
    assert output.read_bytes() == b'{"a":[1,true],"z":"ASCII"}\n'
    assert not list(tmp_path.glob(".*.tmp"))


def test_nvfp4_adapter_progress_is_canonical_monotonic_and_secret_free(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "secret-artifact-name.safetensors"
    layers = []
    for index, data in enumerate((b"first", b"second")):
        layer = type(
            "Nvfp4Linear",
            (),
            {
                "out_features": 8,
                "in_features": 4,
                "weight": _CtypesTensor(data, shape=(8, 2)),
                "weight_scale": None,
                "weight_scale_2": None,
                "input_scale": None,
                "pre_quant_scale": None,
            },
        )()
        layers.append((f"layer-{index}-{secret}", layer))
    diffusion = SimpleNamespace(named_modules=lambda: iter(layers))
    runtime = SimpleNamespace(assembled=SimpleNamespace(diffusion=diffusion))
    path = tmp_path / "progress.json"

    nvfp4_adapter._transition_snapshot(runtime, nvfp4_adapter._ProgressSink(path))

    stderr = capsys.readouterr().err
    events = [json.loads(line) for line in stderr.splitlines()]
    assert json.loads(path.read_text(encoding="ascii")) == events[-1]
    assert all(set(event) == nvfp4_adapter.PROGRESS_KEYS for event in events)
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert [event["layers_complete"] for event in events] == [0, 1, 2, 2]
    assert [event["bytes_complete"] for event in events] == [0, 5, 11, 11]
    assert all(event["layers_total"] == 2 for event in events)
    assert [event["status"] for event in events] == [
        "started",
        "progress",
        "progress",
        "completed",
    ]
    assert secret not in stderr
    assert secret not in path.read_text(encoding="ascii")


def test_nvfp4_adapter_progress_terminal_passes_after_result_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result_path = tmp_path / "result.json"
    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(nvfp4_adapter, "_execute", lambda _request, _progress: {"ok": True})

    nvfp4_adapter._run({}, result_path, progress_path)

    assert json.loads(result_path.read_text(encoding="ascii")) == {"ok": True}
    event = json.loads(progress_path.read_text(encoding="ascii"))
    assert event["operation"] == "terminal"
    assert event["status"] == "passed"
    assert event["exception_type"] is None
    assert json.loads(capsys.readouterr().err) == event


def test_nvfp4_adapter_progress_failure_preserves_exception_and_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    failure = RuntimeError("forbidden-request-path")

    def fail(_request: object, _progress: object) -> NoReturn:
        raise failure

    monkeypatch.setattr(nvfp4_adapter, "_execute", fail)
    progress_path = tmp_path / "progress.json"
    with pytest.raises(RuntimeError) as caught:
        nvfp4_adapter._run({}, tmp_path / "result.json", progress_path)
    assert caught.value is failure
    assert caught.traceback[-1].name == "fail"
    event = json.loads(progress_path.read_text(encoding="ascii"))
    assert event["operation"] == "terminal"
    assert event["status"] == "failed"
    assert event["exception_type"] == "RuntimeError"
    stderr = capsys.readouterr().err
    assert json.loads(stderr) == event
    assert "forbidden-request-path" not in stderr
    assert "forbidden-request-path" not in progress_path.read_text(encoding="ascii")


def test_nvfp4_adapter_releases_handle_on_execution_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _nvfp4_request(tmp_path)

    class Torch:
        float32 = "float32"
        bfloat16 = "bfloat16"
        cuda = SimpleNamespace(reset_peak_memory_stats=lambda: None)

        @staticmethod
        @contextmanager
        def inference_mode() -> Any:
            yield

    class Handle:
        def __init__(self) -> None:
            self.released = False
            self.releases = 0
            self.load_device = "cuda:0"
            self.runtime = SimpleNamespace(
                encode_text=lambda _text: (_ for _ in ()).throw(RuntimeError("boom"))
            )

        @contextmanager
        def stage(self, role: str) -> Any:
            assert role == "text"
            yield

        def terminal_release(self) -> None:
            self.releases += 1
            self.released = True

    handle = Handle()
    monkeypatch.setattr(nvfp4_adapter, "_runtime_imports", lambda: SimpleNamespace(torch=Torch))
    monkeypatch.setattr(nvfp4_adapter, "_canonical_load", lambda _request, _api: handle)

    class Identity:
        def __init__(self) -> None:
            self.handle = handle
            self.runtime = handle.runtime
            self.assembled = object()
            self.recorder = object()
            self.baseline: dict[str, Any] = {}

        def check(self, *_args: object, **_kwargs: object) -> dict[str, Any]:
            return {}

    monkeypatch.setattr(
        nvfp4_adapter._ProcessIdentity,
        "capture",
        classmethod(lambda _cls, _handle, _request, _progress=None: Identity()),
    )
    with pytest.raises(RuntimeError, match="boom"):
        nvfp4_adapter._execute(request)
    assert handle.released
    assert handle.releases == 1


def test_nvfp4_adapter_releases_when_completed_load_progress_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _nvfp4_request(tmp_path)

    class Handle:
        released = False
        releases = 0

        def terminal_release(self) -> None:
            self.releases += 1
            self.released = True

    class Progress:
        calls = 0

        def emit(self, *_args: object, **_kwargs: object) -> None:
            self.calls += 1
            if self.calls == 2:
                raise OSError("progress unavailable")

    handle = Handle()
    torch = SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None))
    monkeypatch.setattr(nvfp4_adapter, "_runtime_imports", lambda: SimpleNamespace(torch=torch))
    monkeypatch.setattr(nvfp4_adapter, "_canonical_load", lambda *_args: handle)

    with pytest.raises(OSError, match="progress unavailable"):
        nvfp4_adapter._execute(request, Progress())  # type: ignore[arg-type]
    assert handle.released
    assert handle.releases == 1


def test_nvfp4_adapter_cleanup_failures_do_not_replace_execution_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _nvfp4_request(tmp_path)
    primary = RuntimeError("execution failed")

    class Handle:
        def __init__(self) -> None:
            self.released = False
            self.releases = 0
            self.runtime = object()

        def terminal_release(self) -> None:
            self.releases += 1
            raise OSError("release failed")

    class Identity:
        def __init__(self, handle: Handle) -> None:
            self.handle = handle
            self.runtime = handle.runtime
            self.assembled = object()
            self.recorder = object()
            self.baseline: dict[str, Any] = {}
            self.checks = 0

        def check(self, *_args: object, **_kwargs: object) -> dict[str, Any]:
            self.checks += 1
            if self.checks == 1:
                raise primary
            raise OSError("snapshot progress failed")

    handle = Handle()
    identity = Identity(handle)
    torch = SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None))
    monkeypatch.setattr(nvfp4_adapter, "_runtime_imports", lambda: SimpleNamespace(torch=torch))
    monkeypatch.setattr(nvfp4_adapter, "_canonical_load", lambda *_args: handle)
    monkeypatch.setattr(
        nvfp4_adapter._ProcessIdentity,
        "capture",
        classmethod(lambda *_args: identity),
    )

    with pytest.raises(RuntimeError) as caught:
        nvfp4_adapter._execute(request)
    assert caught.value is primary
    assert caught.traceback[-1].name == "check"
    assert handle.releases == 1
    notes = getattr(primary, "__notes__", ())
    assert len(notes) == 1
    assert "snapshot progress failed" in notes[0]
    assert "release failed" in notes[0]


def test_nvfp4_adapter_prepare_only_is_closed_exact_and_nonexecuting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = _nvfp4_request(tmp_path)
    calls = {"imports": 0, "load": 0, "capture": 0, "inventory": 0, "release": 0}
    forbidden_calls: list[str] = []

    class ForbiddenCuda:
        def __getattr__(self, name: str) -> NoReturn:
            forbidden_calls.append(f"cuda.{name}")
            raise AssertionError(name)

    class Runtime:
        assembled = object()

        def __getattr__(self, name: str) -> NoReturn:
            forbidden_calls.append(name)
            raise AssertionError(name)

    class Handle:
        def __init__(self) -> None:
            self.runtime = Runtime()
            self.released = False

        def stage(self, role: str) -> NoReturn:
            forbidden_calls.append(f"stage:{role}")
            raise AssertionError(role)

        def terminal_release(self) -> None:
            calls["release"] += 1
            self.released = True

    handle = Handle()
    baseline = {
        "layer": {
            "geometry": (8, 4),
            "tensors": {
                "weight": {
                    "key": "layer.weight",
                    "shape": (8, 2),
                    "dtype": "torch.uint8",
                    "size": 4,
                    "sha256": hashlib.sha256(b"data").hexdigest(),
                }
            },
        }
    }

    class Identity:
        def __init__(self, value: Handle, snapshot: dict[str, Any]) -> None:
            self.runtime = value.runtime
            self.assembled = value.runtime.assembled
            self.recorder = object()
            self.baseline = snapshot
            self.handle = value

    identity = Identity(handle, baseline)

    def imports() -> SimpleNamespace:
        calls["imports"] += 1
        return SimpleNamespace(torch=SimpleNamespace(cuda=ForbiddenCuda()))

    def load(_request: object, _api: object) -> Handle:
        calls["load"] += 1
        return handle

    def capture(_cls: object, value: Handle, supplied: object, progress: object) -> Identity:
        calls["capture"] += 1
        assert value is handle
        assert supplied is request
        assert isinstance(progress, nvfp4_adapter._ProgressSink)
        return identity

    def inventory(runtime: object) -> dict[str, int]:
        calls["inventory"] += 1
        assert runtime is handle.runtime
        return {"nvfp4_linear_layers": 152, "fp8_linear_layers": 114}

    monkeypatch.setattr(nvfp4_adapter, "_runtime_imports", imports)
    monkeypatch.setattr(nvfp4_adapter, "_canonical_load", load)
    monkeypatch.setattr(nvfp4_adapter._ProcessIdentity, "capture", classmethod(capture))
    monkeypatch.setattr(nvfp4_adapter, "_inventory", inventory)
    monkeypatch.setattr(nvfp4_adapter, "_peak_rss_bytes", lambda: 1234)
    for name in ("_status", "_counter_delta", "_representative"):
        monkeypatch.setattr(
            nvfp4_adapter,
            name,
            lambda *_args, forbidden=name, **_kwargs: forbidden_calls.append(forbidden),
        )
    result_path = tmp_path / "prepared.json"
    progress_path = tmp_path / "progress.json"

    nvfp4_adapter._run(
        request,
        result_path,
        progress_path,
        prepare_only=True,
    )

    result = json.loads(result_path.read_text(encoding="ascii"))
    assert set(result) == {
        "schema_version",
        "engine",
        "status",
        "adapter",
        "status_api",
        "inventory",
        "transition",
        "single_object_identity",
        "metrics",
        "cleanup",
    }
    assert result["schema_version"] == 1
    assert result["engine"] == "dinkster"
    assert result["status"] == "prepared"
    assert result["adapter"] == request["adapter"]
    assert result["status_api"] == request["status_api"]
    assert result["inventory"] == {"nvfp4_linear_layers": 152, "fp8_linear_layers": 114}
    assert result["transition"] == nvfp4_adapter._transition_result(baseline)
    assert set(result["transition"]) == {
        "packed_qdata_bytes_sha256",
        "all_scale_bytes_sha256",
        "logical_keys_sha256",
        "logical_shapes_sha256",
        "logical_dtypes_sha256",
        "logical_geometry_sha256",
    }
    assert result["single_object_identity"] == nvfp4_adapter._single_object_identity(
        cast(Any, identity)
    )
    assert set(result["metrics"]) == {"load_seconds", "snapshot_seconds", "peak_rss_bytes"}
    assert result["metrics"]["peak_rss_bytes"] == 1234
    assert result["cleanup"] == {"runtime_released": True}
    assert calls == {"imports": 1, "load": 1, "capture": 1, "inventory": 1, "release": 1}
    assert forbidden_calls == []
    assert result_path.read_bytes() == nvfp4_adapter._canonical_bytes(result)
    terminal = json.loads(progress_path.read_text(encoding="ascii"))
    assert terminal["operation"] == "terminal"
    assert terminal["status"] == "passed"
    assert json.loads(capsys.readouterr().err.splitlines()[-1]) == terminal


def test_nvfp4_adapter_prepare_failure_preserves_primary_and_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = _nvfp4_request(tmp_path)
    primary = RuntimeError("private request detail")

    class Handle:
        released = False
        releases = 0

        def terminal_release(self) -> None:
            self.releases += 1
            self.released = True

    handle = Handle()
    monkeypatch.setattr(nvfp4_adapter, "_runtime_imports", lambda: SimpleNamespace())
    monkeypatch.setattr(nvfp4_adapter, "_canonical_load", lambda *_args: handle)

    def fail_capture(*_args: object) -> NoReturn:
        raise primary

    monkeypatch.setattr(
        nvfp4_adapter._ProcessIdentity,
        "capture",
        classmethod(fail_capture),
    )
    progress_path = tmp_path / "progress.json"

    with pytest.raises(RuntimeError) as caught:
        nvfp4_adapter._run(
            request,
            tmp_path / "result.json",
            progress_path,
            prepare_only=True,
        )
    assert caught.value is primary
    assert caught.traceback[-1].name == "fail_capture"
    assert handle.released
    assert handle.releases == 1
    terminal = json.loads(progress_path.read_text(encoding="ascii"))
    assert terminal["operation"] == "terminal"
    assert terminal["status"] == "failed"
    assert terminal["exception_type"] == "RuntimeError"
    stderr = capsys.readouterr().err
    assert json.loads(stderr.splitlines()[-1]) == terminal
    assert "private request detail" not in stderr
    assert "private request detail" not in progress_path.read_text(encoding="ascii")


def test_nvfp4_adapter_prepare_cleanup_failure_does_not_replace_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _nvfp4_request(tmp_path)
    primary = RuntimeError("capture failed")

    class Handle:
        released = False
        releases = 0

        def terminal_release(self) -> None:
            self.releases += 1
            raise OSError("release failed")

    handle = Handle()
    monkeypatch.setattr(nvfp4_adapter, "_runtime_imports", lambda: SimpleNamespace())
    monkeypatch.setattr(nvfp4_adapter, "_canonical_load", lambda *_args: handle)
    monkeypatch.setattr(
        nvfp4_adapter._ProcessIdentity,
        "capture",
        classmethod(lambda *_args: (_ for _ in ()).throw(primary)),
    )

    with pytest.raises(RuntimeError) as caught:
        nvfp4_adapter._prepare(request)
    assert caught.value is primary
    assert handle.releases == 1
    notes = cast(list[str], getattr(primary, "__notes__", []))
    assert notes and "release failed" in notes[0]


def test_nvfp4_adapter_prepare_refuses_pre_released_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _nvfp4_request(tmp_path)

    class Handle:
        released = True
        releases = 0
        runtime = object()

        def terminal_release(self) -> None:
            self.releases += 1

    class Identity:
        def __init__(self, handle: Handle) -> None:
            self.handle = handle
            self.runtime = handle.runtime
            self.assembled = object()
            self.recorder = object()
            self.baseline: dict[str, Any] = {}

    handle = Handle()
    identity = Identity(handle)
    monkeypatch.setattr(nvfp4_adapter, "_runtime_imports", lambda: SimpleNamespace())
    monkeypatch.setattr(nvfp4_adapter, "_canonical_load", lambda *_args: handle)
    monkeypatch.setattr(
        nvfp4_adapter._ProcessIdentity,
        "capture",
        classmethod(lambda *_args: identity),
    )
    monkeypatch.setattr(
        nvfp4_adapter,
        "_inventory",
        lambda _runtime: {"nvfp4_linear_layers": 152, "fp8_linear_layers": 114},
    )

    with pytest.raises(nvfp4_adapter.AdapterError, match="released before final cleanup"):
        nvfp4_adapter._prepare(request)
    assert handle.releases == 0


def test_nvfp4_adapter_run_defaults_to_normal_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        nvfp4_adapter,
        "_execute",
        lambda _request, _progress: calls.append("execute") or {"mode": "normal"},
    )
    monkeypatch.setattr(
        nvfp4_adapter,
        "_prepare",
        lambda _request, _progress: calls.append("prepare") or {"mode": "prepare"},
    )
    result_path = tmp_path / "result.json"
    nvfp4_adapter._run({}, result_path, None)
    assert calls == ["execute"]
    assert json.loads(result_path.read_text(encoding="ascii")) == {"mode": "normal"}


@pytest.mark.parametrize("prepare_only", [False, True])
def test_nvfp4_adapter_cli_prepare_flag_is_additive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepare_only: bool,
) -> None:
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    arguments = [
        "nvfp4_flux_dinkster_adapter.py",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]
    if prepare_only:
        arguments.append("--prepare-only")
    observed: list[tuple[object, ...]] = []
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(
        nvfp4_adapter,
        "_run",
        lambda *args, **kwargs: observed.append((*args, kwargs)),
    )
    assert nvfp4_adapter.main() == 0
    assert observed == [
        (
            request_path,
            result_path,
            None,
            {"prepare_only": prepare_only},
        )
    ]


def test_nvfp4_adapter_source_keeps_status_and_authority_boundaries() -> None:
    source = Path("tools/inference_parity/nvfp4_flux_dinkster_adapter.py").read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "dinkster_inference_torch._nvfp4_diagnostics"
        for alias in node.names
    }
    assert imports == {"nvfp4_runtime_status"}
    assert "dinkster_adapter" not in source
    assert "ovis_dinkster_adapter" not in source
    assert "load_safetensors_header" not in source
    assert "load_runtime" not in source
    assert "reason" not in source
    assert "filename" not in source
    assert "GPU-" not in source
    assert "/home/" not in source
    assert ".safetensors" not in source
