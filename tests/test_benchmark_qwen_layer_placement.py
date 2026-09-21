"""Qwen layer-placement benchmark evidence validation."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest

import tools.benchmark_qwen_layer_placement as benchmark
from tools.evidence_paths import EVIDENCE_ROOT

EVIDENCE = EVIDENCE_ROOT / "benchmarks" / "qwen-generation"
DINKSTER_COMMIT = benchmark._git_head(Path(__file__).parents[1])


def _report(mode: str, *, rate: float, memory: int) -> dict[str, object]:
    devices = ("cuda:0",) if mode == "one-gpu" else ("cuda:0", "cuda:1")
    weights = {device: memory - index for index, device in enumerate(devices)}
    runs = tuple(
        benchmark.Run(
            benchmark.EXPECTED_GENERATED_TOKENS / rate,
            0.01,
            benchmark.EXPECTED_GENERATED_TOKENS,
            benchmark.EXPECTED_TEXT_SHA256,
            benchmark.EXPECTED_TOKEN_IDS_SHA256,
            {device: memory + 100 for device in devices},
            {device: memory + 200 for device in devices},
        )
        for _ in range(3)
    )
    return {
        "schema": "dinkster.qwen-layer-placement-benchmark.v1",
        "mode": mode,
        "artifact": {
            "path": "/models/qwen_3_06b_base.safetensors",
            "bytes": benchmark.MODEL_SIZE,
            "sha256": benchmark.MODEL_SHA256,
            "source_url": benchmark.MODEL_URL,
        },
        "workload": {
            "prompt": benchmark.DEFAULT_PROMPT,
            "max_new_tokens": 128,
            "sampler": "greedy",
            "warmups": 1,
            "repeats": 3,
        },
        "placement": {
            "split_layer": None if mode == "one-gpu" else benchmark.TWO_GPU_SPLIT_LAYER,
            "ranges": (
                [{"start": 0, "stop": benchmark.MODEL_LAYER_COUNT, "device": "cuda:0"}]
                if mode == "one-gpu"
                else [
                    {
                        "start": 0,
                        "stop": benchmark.TWO_GPU_SPLIT_LAYER,
                        "device": "cuda:0",
                    },
                    {
                        "start": benchmark.TWO_GPU_SPLIT_LAYER,
                        "stop": benchmark.MODEL_LAYER_COUNT,
                        "device": "cuda:1",
                    },
                ]
            ),
        },
        "environment": {
            "dinkster_commit": DINKSTER_COMMIT,
            "dinkster_dirty": False,
            "host": "benchmark-host",
            "python": "3.12.0",
            "torch": "2.13.0",
            "cuda": "13.0",
            "cuda_visible_devices": None,
            "model_dtype": "torch.bfloat16",
            "gpus": (
                [{"device": "cuda:0", "name": "RTX 4090", "memory_bytes": 24_000}]
                if mode == "one-gpu"
                else [
                    {"device": "cuda:0", "name": "RTX 4090", "memory_bytes": 24_000},
                    {"device": "cuda:1", "name": "RTX 4090", "memory_bytes": 24_000},
                ]
            ),
        },
        "runs": [asdict(run) for run in runs],
        "summary": benchmark._summary(runs, weights),
    }


def test_comparison_records_throughput_and_per_device_memory_reduction() -> None:
    reference = _report("one-gpu", rate=90.0, memory=1_200)
    candidate = _report("two-gpu", rate=70.0, memory=700)

    assert benchmark._compare_reports(reference, candidate) == {
        "throughput_ratio": cast("dict[str, float]", candidate["summary"])[
            "median_generated_tokens_per_s"
        ]
        / cast("dict[str, float]", reference["summary"])["median_generated_tokens_per_s"],
        "max_weight_memory_ratio": 700.0 / 1_200.0,
        "max_weight_memory_reduction": 1.0 - 700.0 / 1_200.0,
    }


def test_retained_two_gpu_evidence_preserves_outputs_and_reduces_weight_memory() -> None:
    reference = json.loads((EVIDENCE / "qwen-layer-placement-one-gpu.json").read_text())
    candidate = json.loads((EVIDENCE / "qwen-layer-placement-two-gpu.json").read_text())
    comparison = benchmark._compare_reports(reference, candidate)

    assert comparison == candidate["comparison"]
    assert comparison["throughput_ratio"] == pytest.approx(0.9555093901136775)
    assert comparison["max_weight_memory_reduction"] == pytest.approx(0.4985483269683054)


@pytest.mark.parametrize(
    "artifact_path",
    ("/models/qwen_3_06b_base.safetensors", r"C:\models\qwen_3_06b_base.safetensors"),
)
def test_comparison_accepts_absolute_artifact_paths_from_either_host_os(
    artifact_path: str,
) -> None:
    reference = _report("one-gpu", rate=90.0, memory=1_200)
    candidate = _report("two-gpu", rate=70.0, memory=700)
    for report in (reference, candidate):
        cast("dict[str, object]", report["artifact"])["path"] = artifact_path

    benchmark._compare_reports(reference, candidate)


def test_comparison_accepts_scrubbed_local_home_artifact_path() -> None:
    reference = _report("one-gpu", rate=90.0, memory=1_200)
    candidate = _report("two-gpu", rate=70.0, memory=700)
    for report in (reference, candidate):
        cast("dict[str, object]", report["artifact"])["path"] = (
            "<LOCAL_HOME>/models/qwen_3_06b_base.safetensors"
        )

    benchmark._compare_reports(reference, candidate)


def test_comparison_refuses_relative_artifact_path() -> None:
    reference = _report("one-gpu", rate=90.0, memory=1_200)
    candidate = _report("two-gpu", rate=70.0, memory=700)
    for report in (reference, candidate):
        cast("dict[str, object]", report["artifact"])["path"] = "models/model.safetensors"

    with pytest.raises(ValueError, match="artifact path must be absolute or use <LOCAL_HOME>"):
        benchmark._compare_reports(reference, candidate)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("artifact", {}, "artifact"),
        ("workload", {}, "workload"),
        ("placement", {}, "retained profile"),
        ("environment.dinkster_commit", "d" * 40, "environment dinkster_commit"),
        ("runs.generated_tokens", 25, "token count"),
        ("runs.text_sha256", "e" * 64, "output hashes"),
        ("runs.token_ids_sha256", "f" * 64, "output hashes"),
    ],
)
def test_comparison_refuses_unmatched_evidence(
    field: str,
    value: object,
    match: str,
) -> None:
    reference = _report("one-gpu", rate=90.0, memory=1_200)
    candidate = deepcopy(_report("two-gpu", rate=70.0, memory=700))
    if field in ("artifact", "workload", "placement"):
        candidate[field] = value
    elif field.startswith("environment."):
        environment = cast("dict[str, object]", candidate["environment"])
        environment[field.partition(".")[2]] = value
    else:
        run = cast("list[dict[str, object]]", candidate["runs"])[0]
        run[field.partition(".")[2]] = value

    with pytest.raises(ValueError, match=match):
        benchmark._compare_reports(reference, candidate)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("artifact", "artifact sha256"),
        ("dirty", "clean Dinkster checkout"),
        ("commit", "retained or current code"),
        ("runs", "run count"),
        ("placement", "retained profile"),
        ("summary", "summary does not match"),
    ),
)
def test_comparison_refuses_paired_or_self_consistent_tampering(
    mutation: str,
    match: str,
) -> None:
    reference = deepcopy(_report("one-gpu", rate=90.0, memory=1_200))
    candidate = deepcopy(_report("two-gpu", rate=70.0, memory=700))
    if mutation == "artifact":
        for report in (reference, candidate):
            cast("dict[str, object]", report["artifact"])["sha256"] = "0" * 64
    elif mutation == "dirty":
        for report in (reference, candidate):
            cast("dict[str, object]", report["environment"])["dinkster_dirty"] = True
    elif mutation == "commit":
        for report in (reference, candidate):
            cast("dict[str, object]", report["environment"])["dinkster_commit"] = "0" * 40
    elif mutation == "runs":
        for report in (reference, candidate):
            report["runs"] = []
    elif mutation == "placement":
        placement = cast("dict[str, object]", candidate["placement"])
        placement["split_layer"] = 1
        placement["ranges"] = [
            {"start": 0, "stop": 1, "device": "cuda:0"},
            {"start": 1, "stop": benchmark.MODEL_LAYER_COUNT, "device": "cuda:1"},
        ]
    elif mutation == "summary":
        summary = cast("dict[str, object]", candidate["summary"])
        summary["median_generated_tokens_per_s"] = 1_000_000.0
        summary["max_weight_allocated_bytes_per_device"] = 1

    with pytest.raises(ValueError, match=match):
        benchmark._compare_reports(reference, candidate)


def test_comparison_refuses_tampered_retained_comparison() -> None:
    reference = _report("one-gpu", rate=90.0, memory=1_200)
    candidate = _report("two-gpu", rate=70.0, memory=700)
    candidate["comparison"] = {
        "throughput_ratio": 1_000_000.0,
        "max_weight_memory_ratio": 0.0,
        "max_weight_memory_reduction": 1.0,
    }

    with pytest.raises(ValueError, match="comparison does not match"):
        benchmark._compare_reports(reference, candidate)


def test_comparison_requires_one_then_two_gpu_reports() -> None:
    with pytest.raises(ValueError, match="one-gpu reference"):
        benchmark._compare_reports(
            _report("two-gpu", rate=90.0, memory=1_200),
            _report("one-gpu", rate=70.0, memory=700),
        )
