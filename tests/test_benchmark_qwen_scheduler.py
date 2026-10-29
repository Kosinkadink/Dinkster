"""Continuous Qwen benchmark evidence validation."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import replace

import pytest
from dinkster_inference import (
    GenerationFinishReason,
    GenerationResult,
    GenerationStats,
    GenerationTerminalEvent,
)

import tools.benchmark_qwen_scheduler as benchmark


def _batch(*, wall_s: float = 2.0, token_hash: str = "a" * 64) -> benchmark.BatchRun:
    request = benchmark.RequestRun(
        total_s=wall_s,
        time_to_first_token_s=0.1,
        prefill_time_s=0.1,
        decode_time_s=0.3,
        mean_inter_token_s=0.1,
        completion_s=wall_s,
        prompt_tokens=2,
        generated_tokens=4,
        text_sha256="b" * 64,
        token_ids_sha256=token_hash,
        inter_token_s=(0.08, 0.1, 0.12),
    )
    return benchmark.BatchRun(
        wall_s=wall_s,
        generated_tokens=8,
        allocated_before_bytes=100,
        peak_allocated_bytes=200,
        peak_reserved_bytes=300,
        requests=(request, replace(request, completion_s=wall_s + 0.25)),
    )


def test_summary_records_aggregate_rate_latency_fairness_and_memory() -> None:
    assert benchmark._summary((_batch(),)) == {
        "median_wall_s": 2.0,
        "median_generated_tokens_per_s": 4.0,
        "median_time_to_first_token_s": 0.1,
        "median_inter_token_s": 0.1,
        "inter_token_p50_s": 0.1,
        "inter_token_p95_s": 0.12,
        "median_prefill_tokens_per_s": 20.0,
        "median_decode_tokens_per_s": 10.0,
        "max_completion_spread_s": 0.25,
        "max_peak_allocated_bytes": 200.0,
        "max_peak_allocated_delta_bytes": 100.0,
        "max_peak_reserved_bytes": 300.0,
    }


def test_output_validation_requires_determinism_within_each_execution_mode() -> None:
    serial = _batch()
    continuous = _batch(wall_s=1.0)
    assert benchmark._validate_outputs((serial,), (continuous,)) is True

    changed_request = replace(continuous.requests[0], token_ids_sha256="c" * 64)
    changed = replace(continuous, requests=(changed_request, continuous.requests[1]))
    with pytest.raises(RuntimeError, match="continuous generation outputs are not deterministic"):
        benchmark._validate_outputs((serial,), (changed,))

    changed_mode = replace(
        continuous,
        requests=tuple(
            replace(request, token_ids_sha256="c" * 64) for request in continuous.requests
        ),
    )
    assert benchmark._validate_outputs((serial,), (changed_mode,)) is False


def test_percentile_interpolates_small_samples() -> None:
    assert benchmark._percentile((), 0.95) == 0.0
    assert benchmark._percentile((1.0,), 0.95) == 1.0
    assert benchmark._percentile((1.0, 2.0, 3.0), 0.75) == 2.5


def test_result_collection_requires_prompt_token_evidence() -> None:
    event = GenerationTerminalEvent(
        GenerationResult(
            "result",
            GenerationFinishReason.LENGTH,
            GenerationStats(
                0.5,
                generated_tokens=1,
                time_to_first_token_s=0.1,
                prefill_time_s=0.1,
                decode_time_s=0.2,
            ),
            token_ids=(1,),
        )
    )

    class Stream:
        def __enter__(self) -> Stream:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self) -> Iterator[GenerationTerminalEvent]:
            yield event

    with pytest.raises(RuntimeError, match="token or first-token evidence"):
        benchmark._collect_result(Stream(), time.perf_counter())


def test_serial_batch_submits_all_requests_before_draining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_run = _batch().requests[0]
    collected: list[int] = []
    closed: list[int] = []

    class Stream:
        def __init__(self, index: int) -> None:
            self.index = index

        def close(self) -> None:
            closed.append(self.index)

    class Provider:
        def __init__(self) -> None:
            self.streams: list[Stream] = []

        def generate(self, _request: object, *, cancelled: object) -> Stream:
            del cancelled
            stream = Stream(len(self.streams))
            self.streams.append(stream)
            return stream

    provider = Provider()

    def collect(stream: Stream, _started: float) -> benchmark.RequestRun:
        assert len(provider.streams) == 3
        collected.append(stream.index)
        return request_run

    monkeypatch.setattr(benchmark, "_collect_result", collect)

    assert benchmark._serial_batch(provider, object(), 3) == (request_run,) * 3
    assert collected == [0, 1, 2]
    assert closed == [0, 1, 2]
