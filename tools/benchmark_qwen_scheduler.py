"""Compare serialized and continuous Qwen3-0.6B generation on one model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from math import floor
from pathlib import Path
from typing import Any

MODEL_SIZE = 1_192_135_096
MODEL_SHA256 = "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
MODEL_URL = (
    "https://huggingface.co/circlestone-labs/Anima/resolve/"
    "e26179e4b23bcb3a9e91b4ad2961a76ab9644d43/"
    "split_files/text_encoders/qwen_3_06b_base.safetensors"
)
DEFAULT_PROMPT = "Write one concise sentence describing a quiet forest at sunrise."


@dataclass(frozen=True, slots=True)
class RequestRun:
    total_s: float
    time_to_first_token_s: float
    prefill_time_s: float
    decode_time_s: float
    mean_inter_token_s: float
    completion_s: float
    prompt_tokens: int
    generated_tokens: int
    text_sha256: str
    token_ids_sha256: str
    inter_token_s: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BatchRun:
    wall_s: float
    generated_tokens: int
    allocated_before_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    requests: tuple[RequestRun, ...]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model(path: Path) -> None:
    if path.stat().st_size != MODEL_SIZE:
        raise SystemExit(f"model size differs from pinned {MODEL_SIZE} bytes: {path}")
    digest = _digest(path)
    if digest != MODEL_SHA256:
        raise SystemExit(f"model sha256 differs from pinned {MODEL_SHA256}: {digest}")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _hash_ids(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _collect_result(stream: Any, started: float) -> RequestRun:
    from dinkster_inference import GenerationTerminalEvent, GenerationTokenEvent

    events = []
    token_times = []
    with stream:
        for event in stream:
            events.append(event)
            if isinstance(event, GenerationTokenEvent):
                token_times.append(time.perf_counter())
    completed = time.perf_counter() - started
    terminal = events[-1]
    if not isinstance(terminal, GenerationTerminalEvent):
        raise RuntimeError("Qwen generation ended without a terminal event")
    result = terminal.result
    if (
        result.token_ids is None
        or result.stats.time_to_first_token_s is None
        or result.stats.prefill_time_s is None
        or result.stats.decode_time_s is None
        or result.stats.prompt_tokens is None
    ):
        raise RuntimeError("Qwen generation did not report token or first-token evidence")
    inter_token = [
        later - earlier for earlier, later in zip(token_times, token_times[1:], strict=False)
    ]
    return RequestRun(
        result.stats.total_time_s,
        result.stats.time_to_first_token_s,
        result.stats.prefill_time_s,
        result.stats.decode_time_s,
        statistics.mean(inter_token) if inter_token else 0.0,
        completed,
        result.stats.prompt_tokens,
        len(result.token_ids),
        _hash_text(result.text),
        _hash_ids(result.token_ids),
        tuple(inter_token),
    )


def _measure(
    torch: Any,
    device: str,
    call: Callable[[], tuple[RequestRun, ...]],
) -> BatchRun:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated = torch.cuda.memory_allocated(device)
    started = time.perf_counter()
    requests = call()
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    return BatchRun(
        wall,
        sum(run.generated_tokens for run in requests),
        allocated,
        torch.cuda.max_memory_allocated(device),
        torch.cuda.max_memory_reserved(device),
        requests,
    )


def _serial_batch(provider: Any, request: Any, concurrency: int) -> tuple[RequestRun, ...]:
    streams = [provider.generate(request, cancelled=lambda: False) for _ in range(concurrency)]
    started = time.perf_counter()
    try:
        return tuple(_collect_result(stream, started) for stream in streams)
    finally:
        for stream in streams:
            stream.close()


def _continuous_batch(provider: Any, request: Any, concurrency: int) -> tuple[RequestRun, ...]:
    streams = [provider.generate(request, cancelled=lambda: False) for _ in range(concurrency)]
    barrier = threading.Barrier(concurrency)
    started = time.perf_counter()

    def collect(stream: Any) -> RequestRun:
        barrier.wait()
        return _collect_result(stream, started)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(collect, stream) for stream in streams]
        return tuple(future.result() for future in futures)


def _validate_outputs(serial: Sequence[BatchRun], continuous: Sequence[BatchRun]) -> bool:
    evidence_by_mode = {
        mode: {
            (request.generated_tokens, request.text_sha256, request.token_ids_sha256)
            for batch in runs
            for request in batch.requests
        }
        for mode, runs in (("serialized", serial), ("continuous", continuous))
    }
    for mode, evidence in evidence_by_mode.items():
        if len(evidence) != 1:
            raise RuntimeError(f"{mode} generation outputs are not deterministic")
    return evidence_by_mode["serialized"] == evidence_by_mode["continuous"]


def _percentile(samples: Sequence[float], fraction: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(runs: Sequence[BatchRun]) -> dict[str, float]:
    throughput = [run.generated_tokens / run.wall_s for run in runs]
    first_token = [request.time_to_first_token_s for run in runs for request in run.requests]
    inter_token = [request.mean_inter_token_s for run in runs for request in run.requests]
    inter_token_samples = [
        sample for run in runs for request in run.requests for sample in request.inter_token_s
    ]
    prefill_throughput = [
        request.prompt_tokens / request.prefill_time_s
        for run in runs
        for request in run.requests
        if request.prefill_time_s > 0.0
    ]
    decode_throughput = [
        (request.generated_tokens - 1) / request.decode_time_s
        for run in runs
        for request in run.requests
        if request.generated_tokens > 1 and request.decode_time_s > 0.0
    ]
    completion_spread = [
        max(request.completion_s for request in run.requests)
        - min(request.completion_s for request in run.requests)
        for run in runs
    ]
    return {
        "median_wall_s": statistics.median(run.wall_s for run in runs),
        "median_generated_tokens_per_s": statistics.median(throughput),
        "median_time_to_first_token_s": statistics.median(first_token),
        "median_inter_token_s": statistics.median(inter_token),
        "inter_token_p50_s": _percentile(inter_token_samples, 0.5),
        "inter_token_p95_s": _percentile(inter_token_samples, 0.95),
        "median_prefill_tokens_per_s": (
            statistics.median(prefill_throughput) if prefill_throughput else 0.0
        ),
        "median_decode_tokens_per_s": (
            statistics.median(decode_throughput) if decode_throughput else 0.0
        ),
        "max_completion_spread_s": max(completion_spread),
        "max_peak_allocated_bytes": float(max(run.peak_allocated_bytes for run in runs)),
        "max_peak_allocated_delta_bytes": float(
            max(run.peak_allocated_bytes - run.allocated_before_bytes for run in runs)
        ),
        "max_peak_reserved_bytes": float(max(run.peak_reserved_bytes for run in runs)),
    }


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty(root: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--slot-capacity", type=int, default=256)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--require-speedup", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.max_new_tokens < 1
        or args.concurrency < 2
        or args.slot_capacity < 1
        or args.prefill_chunk_tokens < 1
        or args.warmups < 0
        or args.repeats < 1
    ):
        parser.error("counts must be positive, concurrency at least two, and warmups non-negative")
    if not args.device.startswith("cuda"):
        parser.error("the scheduler benchmark requires a CUDA device")
    _verify_model(args.model)

    import torch
    from dinkster_inference import (
        ANIMA_QWEN3_06B_CONFIG,
        GenerationRequest,
        GenerationStopConditions,
        load_qwen_bpe,
    )
    from dinkster_inference_torch import (
        QwenContinuousGenerationProvider,
        QwenGenerationProvider,
        QwenTextModel,
    )
    from safetensors.torch import load_file

    state = load_file(args.model, device=args.device)
    tower = {key.removeprefix("model."): value for key, value in state.items()}
    if len(tower) != len(state) or any(not key.startswith("model.") for key in state):
        raise SystemExit("pinned Qwen checkpoint must contain only model.* tensors")
    with torch.device("meta"):
        model = QwenTextModel(ANIMA_QWEN3_06B_CONFIG)
    model.load_state_dict(tower, strict=True, assign=True)
    del state, tower
    tokenizer = load_qwen_bpe()
    prompt_tokens = len(tokenizer.encode(args.prompt))
    required = prompt_tokens + args.max_new_tokens
    if required > args.slot_capacity:
        parser.error(
            f"prompt and output need {required} positions, above slot capacity {args.slot_capacity}"
        )
    direct = QwenGenerationProvider(
        model,
        tokenizer,
        MODEL_SHA256,
        block_tokens=256,
        max_device_blocks=max(32, args.concurrency * 2),
    )
    request = GenerationRequest(
        direct.id,
        MODEL_SHA256,
        prompt=args.prompt,
        stop=GenerationStopConditions(args.max_new_tokens),
    )
    scheduled = QwenContinuousGenerationProvider(
        direct,
        max_batch_size=args.concurrency,
        slot_capacity=args.slot_capacity,
        prefill_chunk_tokens=args.prefill_chunk_tokens,
    )
    working_cache_bytes = scheduled.working_cache_bytes
    try:
        for _ in range(args.warmups):
            _serial_batch(direct, request, args.concurrency)
            _continuous_batch(scheduled, request, args.concurrency)
        serial: list[BatchRun] = []
        continuous: list[BatchRun] = []
        for index in range(args.repeats):
            ordered = (
                (
                    (serial, lambda: _serial_batch(direct, request, args.concurrency)),
                    (continuous, lambda: _continuous_batch(scheduled, request, args.concurrency)),
                )
                if index % 2 == 0
                else (
                    (continuous, lambda: _continuous_batch(scheduled, request, args.concurrency)),
                    (serial, lambda: _serial_batch(direct, request, args.concurrency)),
                )
            )
            for destination, call in ordered:
                destination.append(_measure(torch, args.device, call))
    finally:
        scheduled.close()
    cross_batch_output_match = _validate_outputs(serial, continuous)
    serial_summary = _summary(serial)
    continuous_summary = _summary(continuous)
    speedup = (
        continuous_summary["median_generated_tokens_per_s"]
        / serial_summary["median_generated_tokens_per_s"]
    )
    if args.require_speedup and speedup <= 1.0:
        raise SystemExit(
            f"continuous generation did not improve aggregate throughput: {speedup:.3f}x"
        )

    root = Path(__file__).resolve().parent.parent
    properties = torch.cuda.get_device_properties(args.device)
    report = {
        "schema": "dinkster.qwen-scheduler-benchmark.v1",
        "artifact": {
            "path": str(args.model.resolve()),
            "bytes": MODEL_SIZE,
            "sha256": MODEL_SHA256,
            "source_url": MODEL_URL,
        },
        "workload": {
            "prompt": args.prompt,
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "sampler": "greedy",
            "concurrency": args.concurrency,
            "slot_capacity": args.slot_capacity,
            "prefill_chunk_tokens": args.prefill_chunk_tokens,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": properties.name,
            "gpu_memory_bytes": properties.total_memory,
            "model_dtype": str(model.embed_tokens.weight.dtype),
            "dinkster_commit": _git_head(root),
            "dinkster_dirty": _git_dirty(root),
        },
        "working_cache_bytes": working_cache_bytes,
        "serial": {
            "runs": [asdict(run) for run in serial],
            "summary": serial_summary,
        },
        "continuous": {
            "runs": [asdict(run) for run in continuous],
            "summary": continuous_summary,
        },
        "cross_batch_output_match": cross_batch_output_match,
        "aggregate_throughput_speedup": speedup,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
