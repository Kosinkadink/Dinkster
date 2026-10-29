"""Matched external Qwen generation benchmark report validation."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest

import tools.benchmark_qwen_generation as benchmark


def _report(backend: str, *, total_s: float, rate: float) -> dict[str, object]:
    return {
        "schema": "dinkster.qwen-generation-benchmark.v1",
        "backend": backend,
        "artifact": {
            "runtime_model": "publisher/model",
            "converted_path": "/models/model.gguf",
            "converted_bytes": 100,
            "converted_sha256": "b" * 64,
            "source_bytes": benchmark.MODEL_SIZE,
            "source_sha256": benchmark.MODEL_SHA256,
            "source_url": benchmark.MODEL_URL,
        },
        "workload": {
            "prompt": benchmark.DEFAULT_PROMPT,
            "max_new_tokens": 128,
            "sampler": "greedy",
            "warmups": 1,
            "repeats": 1,
        },
        "environment": {
            "client": backend,
            "compatibility": "openai",
            "dinkster_commit": "c" * 40,
            "dinkster_dirty": False,
            "host": "benchmark-host",
            "runtime": "LM Studio 0.3.31",
            "runtime_model": "publisher/model",
            "runtime_url": "http://127.0.0.1:1234/v1",
            "python": "3.14.0",
        },
        "runs": [
            {
                "total_s": total_s,
                "time_to_first_token_s": 0.01,
                "prompt_tokens": 12,
                "generated_tokens": 26,
                "text_sha256": "a" * 64,
                "token_ids_sha256": None,
            }
        ],
        "median": {
            "total_s": total_s,
            "time_to_first_token_s": 0.01,
            "generated_tokens_per_s": rate,
        },
    }


def test_external_report_comparison_records_matched_overhead() -> None:
    direct = _report("lm-studio", total_s=1.0, rate=26.0)
    provider = _report("lm-studio-provider", total_s=1.04, rate=25.0)

    assert benchmark._compare_external_reports(direct, provider) == {
        "reference_backend": "lm-studio",
        "latency_ratio": 1.04,
        "throughput_ratio": 25.0 / 26.0,
    }


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("artifact", {}, "artifact"),
        ("workload", {}, "workload"),
        ("environment.compatibility", "llama.cpp", "environment compatibility"),
        ("environment.dinkster_commit", "d" * 40, "environment dinkster_commit"),
        ("environment.dinkster_dirty", True, "environment dinkster_dirty"),
        ("environment.host", "another-host", "environment host"),
        ("environment.runtime", "another-runtime", "environment runtime"),
        ("environment.runtime_model", "another-model", "environment runtime_model"),
        ("environment.runtime_url", "http://127.0.0.1:9000/v1", "environment runtime_url"),
        ("runs.prompt_tokens", 13, "prompt_tokens"),
        ("runs.generated_tokens", 25, "generated_tokens"),
        ("runs.text_sha256", "b" * 64, "text_sha256"),
    ],
)
def test_external_report_comparison_refuses_unmatched_evidence(
    field: str,
    value: object,
    match: str,
) -> None:
    direct = _report("lm-studio", total_s=1.0, rate=26.0)
    provider = deepcopy(_report("lm-studio-provider", total_s=1.04, rate=25.0))
    if field == "artifact":
        provider[field] = value
    elif field == "workload":
        provider[field] = value
    elif field.startswith("environment."):
        environment = cast("dict[str, object]", provider["environment"])
        environment[field.partition(".")[2]] = value
    else:
        run = cast("list[dict[str, object]]", provider["runs"])[0]
        run[field.partition(".")[2]] = value

    with pytest.raises(ValueError, match=match):
        benchmark._compare_external_reports(direct, provider)


def test_external_report_comparison_rejects_nonfinite_timings() -> None:
    direct = _report("lm-studio", total_s=1.0, rate=26.0)
    provider = _report("lm-studio-provider", total_s=float("nan"), rate=25.0)

    with pytest.raises(ValueError, match="candidate total_s"):
        benchmark._compare_external_reports(direct, provider)


def test_external_report_comparison_requires_matching_backend_pair() -> None:
    direct = _report("openai", total_s=1.0, rate=26.0)
    provider = _report("lm-studio-provider", total_s=1.04, rate=25.0)

    with pytest.raises(ValueError, match="direct and provider"):
        benchmark._compare_external_reports(direct, provider)
