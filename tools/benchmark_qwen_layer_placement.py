"""Benchmark one- and two-GPU Qwen layer placement on a pinned model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

MODEL_SIZE = 1_192_135_096
MODEL_SHA256 = "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
MODEL_LAYER_COUNT = 28
MODEL_URL = (
    "https://huggingface.co/circlestone-labs/Anima/resolve/"
    "e26179e4b23bcb3a9e91b4ad2961a76ab9644d43/"
    "split_files/text_encoders/qwen_3_06b_base.safetensors"
)
DEFAULT_PROMPT = "Write one concise sentence describing a quiet forest at sunrise."
TWO_GPU_SPLIT_LAYER = 9
EXPECTED_GENERATED_TOKENS = 26
EXPECTED_TEXT_SHA256 = "b98237094bcc8cc1aaa102438b7b1581acb4caf91ebc75d541737c54bf585385"
EXPECTED_TOKEN_IDS_SHA256 = "72801d4b83a5860d7b67da2215cad64835a3130533cf805a14795944c8089f0c"
RETAINED_DINKSTER_COMMIT = "d2df01d3c42bf731a7fac544b0808222428dc47f"
_SCHEMA = "dinkster.qwen-layer-placement-benchmark.v1"
_REPORT_FIELDS = {
    "schema",
    "mode",
    "artifact",
    "workload",
    "placement",
    "environment",
    "runs",
    "summary",
}
_RUN_FIELDS = {
    "total_s",
    "time_to_first_token_s",
    "generated_tokens",
    "text_sha256",
    "token_ids_sha256",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
}


@dataclass(frozen=True, slots=True)
class Run:
    total_s: float
    time_to_first_token_s: float
    generated_tokens: int
    text_sha256: str
    token_ids_sha256: str
    peak_allocated_bytes: dict[str, int]
    peak_reserved_bytes: dict[str, int]


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
    encoded = json.dumps(list(token_ids), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def _report_object(value: object, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"benchmark report {name} must be an object")
    return cast("dict[str, object]", value)


def _report_runs(value: object) -> list[dict[str, object]]:
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise ValueError("benchmark report runs must be a list of objects")
    return cast("list[dict[str, object]]", value)


def _require_fields(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"benchmark report {name} fields do not match")


def _positive_number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"benchmark report {name} must be positive")
    result = float(cast("int | float", value))
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"benchmark report {name} must be positive")
    return result


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"benchmark report {name} must be a positive integer")
    return value


def _validate_placement(report: Mapping[str, object], mode: str) -> None:
    placement = _report_object(report.get("placement"), "placement")
    if mode == "one-gpu":
        expected = {
            "split_layer": None,
            "ranges": [{"start": 0, "stop": MODEL_LAYER_COUNT, "device": "cuda:0"}],
        }
    else:
        expected = {
            "split_layer": TWO_GPU_SPLIT_LAYER,
            "ranges": [
                {"start": 0, "stop": TWO_GPU_SPLIT_LAYER, "device": "cuda:0"},
                {
                    "start": TWO_GPU_SPLIT_LAYER,
                    "stop": MODEL_LAYER_COUNT,
                    "device": "cuda:1",
                },
            ],
        }
    if placement != expected:
        raise ValueError(f"{mode} benchmark placement does not match the retained profile")


def _validate_artifact(report: Mapping[str, object]) -> None:
    artifact = _report_object(report.get("artifact"), "artifact")
    _require_fields(artifact, {"path", "bytes", "sha256", "source_url"}, "artifact")
    path = artifact.get("path")
    if (
        type(path) is not str
        or not path
        or not (
            PurePosixPath(path).is_absolute()
            or PureWindowsPath(path).is_absolute()
            or path.startswith("<LOCAL_HOME>/")
        )
    ):
        raise ValueError("benchmark report artifact path must be absolute or use <LOCAL_HOME>")
    if artifact.get("bytes") != MODEL_SIZE:
        raise ValueError("benchmark report artifact size does not match the pinned model")
    if artifact.get("sha256") != MODEL_SHA256:
        raise ValueError("benchmark report artifact sha256 does not match the pinned model")
    if artifact.get("source_url") != MODEL_URL:
        raise ValueError("benchmark report artifact source does not match the pinned model")


def _validate_workload(report: Mapping[str, object]) -> None:
    workload = _report_object(report.get("workload"), "workload")
    expected = {
        "prompt": DEFAULT_PROMPT,
        "max_new_tokens": 128,
        "sampler": "greedy",
        "warmups": 1,
        "repeats": 3,
    }
    if workload != expected:
        raise ValueError("benchmark report workload does not match the retained contract")


def _validate_environment(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> None:
    reference_environment = _report_object(reference.get("environment"), "environment")
    candidate_environment = _report_object(candidate.get("environment"), "environment")
    fields = {
        "dinkster_commit",
        "dinkster_dirty",
        "host",
        "python",
        "torch",
        "cuda",
        "cuda_visible_devices",
        "model_dtype",
        "gpus",
    }
    _require_fields(reference_environment, fields, "environment")
    _require_fields(candidate_environment, fields, "environment")
    for field in fields - {"gpus"}:
        if reference_environment.get(field) != candidate_environment.get(field):
            raise ValueError(f"benchmark environment {field} does not match")
    if reference_environment.get("dinkster_dirty") is not False:
        raise ValueError("benchmark reports must come from a clean Dinkster checkout")
    commit = reference_environment.get("dinkster_commit")
    if (
        type(commit) is not str
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("benchmark environment Dinkster commit is invalid")
    root = Path(__file__).resolve().parent.parent
    if commit not in (RETAINED_DINKSTER_COMMIT, _git_head(root)):
        raise ValueError(
            "benchmark environment Dinkster commit is not bound to retained or current code"
        )
    for field in ("host", "python", "torch", "cuda"):
        if type(reference_environment.get(field)) is not str or not reference_environment[field]:
            raise ValueError(f"benchmark environment {field} is invalid")
    if reference_environment.get("model_dtype") != "torch.bfloat16":
        raise ValueError("benchmark environment model dtype does not match")
    visible = reference_environment.get("cuda_visible_devices")
    if visible is not None and type(visible) is not str:
        raise ValueError("benchmark environment CUDA visibility is invalid")
    reference_gpus = reference_environment.get("gpus")
    candidate_gpus = candidate_environment.get("gpus")
    if (
        type(reference_gpus) is not list
        or len(reference_gpus) != 1
        or type(candidate_gpus) is not list
        or len(candidate_gpus) != 2
    ):
        raise ValueError("benchmark GPU environments do not match")
    expected_devices = (("cuda:0",), ("cuda:0", "cuda:1"))
    for gpus, devices in zip((reference_gpus, candidate_gpus), expected_devices, strict=True):
        for gpu, device in zip(gpus, devices, strict=True):
            row = _report_object(gpu, "GPU")
            _require_fields(row, {"device", "name", "memory_bytes"}, "GPU")
            if row.get("device") != device or type(row.get("name")) is not str or not row["name"]:
                raise ValueError("benchmark GPU environment is invalid")
            _positive_integer(row.get("memory_bytes"), "GPU memory")
    if reference_gpus[0] != candidate_gpus[0] or (
        cast("dict[str, object]", candidate_gpus[0]).get("name")
        != cast("dict[str, object]", candidate_gpus[1]).get("name")
    ):
        raise ValueError("benchmark GPU environments do not match")


def _validate_runs_and_summary(report: Mapping[str, object], mode: str) -> None:
    devices = ("cuda:0",) if mode == "one-gpu" else ("cuda:0", "cuda:1")
    raw_runs = _report_runs(report.get("runs"))
    if len(raw_runs) != 3:
        raise ValueError("benchmark report run count does not match the retained workload")
    runs: list[Run] = []
    for raw in raw_runs:
        _require_fields(raw, _RUN_FIELDS, "run")
        total = _positive_number(raw.get("total_s"), "run total time")
        first = _positive_number(raw.get("time_to_first_token_s"), "run first-token time")
        if first > total:
            raise ValueError("benchmark run first-token time exceeds total time")
        if raw.get("generated_tokens") != EXPECTED_GENERATED_TOKENS:
            raise ValueError("benchmark run token count does not match the retained output")
        if raw.get("text_sha256") != EXPECTED_TEXT_SHA256 or (
            raw.get("token_ids_sha256") != EXPECTED_TOKEN_IDS_SHA256
        ):
            raise ValueError("benchmark run output hashes do not match the retained output")
        memory_maps: list[dict[str, int]] = []
        for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
            values = _report_object(raw.get(field), field)
            if set(values) != set(devices):
                raise ValueError(f"benchmark run {field} devices do not match placement")
            memory_maps.append(
                {device: _positive_integer(values[device], f"run {field}") for device in devices}
            )
        runs.append(
            Run(
                total,
                first,
                EXPECTED_GENERATED_TOKENS,
                EXPECTED_TEXT_SHA256,
                EXPECTED_TOKEN_IDS_SHA256,
                memory_maps[0],
                memory_maps[1],
            )
        )
    summary = _report_object(report.get("summary"), "summary")
    weights = _report_object(summary.get("weight_allocated_bytes"), "weight allocation")
    if set(weights) != set(devices):
        raise ValueError("benchmark weight allocation devices do not match placement")
    weight_values = {
        device: _positive_integer(weights[device], "weight allocation") for device in devices
    }
    for run in runs:
        for device in devices:
            if run.peak_allocated_bytes[device] < weight_values[device] or (
                run.peak_reserved_bytes[device] < run.peak_allocated_bytes[device]
            ):
                raise ValueError("benchmark run memory evidence is inconsistent")
    expected_summary = _summary(runs, weight_values)
    if summary != expected_summary:
        raise ValueError("benchmark summary does not match raw run evidence")


def _compare_reports(reference: object, candidate: object) -> dict[str, float]:
    reference_report = _report_object(reference, "reference")
    candidate_report = _report_object(candidate, "candidate")
    _require_fields(reference_report, _REPORT_FIELDS, "reference")
    candidate_fields = set(candidate_report)
    if candidate_fields not in (_REPORT_FIELDS, _REPORT_FIELDS | {"comparison"}):
        raise ValueError("benchmark report candidate fields do not match")
    if reference_report.get("schema") != _SCHEMA or candidate_report.get("schema") != _SCHEMA:
        raise ValueError("benchmark report schema does not match")
    if reference_report.get("mode") != "one-gpu" or candidate_report.get("mode") != "two-gpu":
        raise ValueError("comparison requires one-gpu reference and two-gpu candidate")
    _validate_placement(reference_report, "one-gpu")
    _validate_placement(candidate_report, "two-gpu")
    _validate_artifact(reference_report)
    _validate_artifact(candidate_report)
    _validate_workload(reference_report)
    _validate_workload(candidate_report)
    _validate_environment(reference_report, candidate_report)
    _validate_runs_and_summary(reference_report, "one-gpu")
    _validate_runs_and_summary(candidate_report, "two-gpu")
    reference_summary = _report_object(reference_report.get("summary"), "summary")
    candidate_summary = _report_object(candidate_report.get("summary"), "summary")
    reference_rate = _positive_number(
        reference_summary.get("median_generated_tokens_per_s"), "reference throughput"
    )
    candidate_rate = _positive_number(
        candidate_summary.get("median_generated_tokens_per_s"), "candidate throughput"
    )
    reference_memory = _positive_number(
        reference_summary.get("max_weight_allocated_bytes_per_device"), "reference memory"
    )
    candidate_memory = _positive_number(
        candidate_summary.get("max_weight_allocated_bytes_per_device"), "candidate memory"
    )
    comparison = {
        "throughput_ratio": candidate_rate / reference_rate,
        "max_weight_memory_ratio": candidate_memory / reference_memory,
        "max_weight_memory_reduction": 1.0 - candidate_memory / reference_memory,
    }
    if "comparison" in candidate_report:
        retained = _report_object(candidate_report["comparison"], "comparison")
        if retained != comparison:
            raise ValueError("benchmark comparison does not match raw evidence")
    return comparison


def _summary(
    runs: Sequence[Run],
    weight_allocated_bytes: dict[str, int],
) -> dict[str, object]:
    generated = statistics.median(run.generated_tokens for run in runs)
    return {
        "median_total_s": statistics.median(run.total_s for run in runs),
        "median_time_to_first_token_s": statistics.median(
            run.time_to_first_token_s for run in runs
        ),
        "median_generated_tokens_per_s": generated / statistics.median(run.total_s for run in runs),
        "weight_allocated_bytes": weight_allocated_bytes,
        "max_weight_allocated_bytes_per_device": max(weight_allocated_bytes.values()),
        "max_peak_allocated_bytes_per_device": max(
            value for run in runs for value in run.peak_allocated_bytes.values()
        ),
        "max_peak_reserved_bytes_per_device": max(
            value for run in runs for value in run.peak_reserved_bytes.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("one-gpu", "two-gpu"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--split-layer", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compare-with", type=Path)
    parser.add_argument("--require-memory-reduction", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("token and repeat counts must be positive and warmups non-negative")
    if args.require_memory_reduction and args.compare_with is None:
        parser.error("--require-memory-reduction requires --compare-with")
    _verify_model(args.model)

    import torch
    from dinkster_inference import (
        ANIMA_QWEN3_06B_CONFIG,
        GenerationRequest,
        GenerationStopConditions,
        GenerationTerminalEvent,
        load_qwen_bpe,
    )
    from dinkster_inference_torch import (
        QwenGenerationProvider,
        QwenLayerPlacement,
        QwenLayerRange,
        QwenTextModel,
        ResidencyManager,
        enroll_qwen_layer_placement,
        load_tensors,
    )

    if not torch.cuda.is_available() or (args.mode == "two-gpu" and torch.cuda.device_count() < 2):
        parser.error(f"{args.mode} needs the corresponding visible CUDA devices")
    layer_count = ANIMA_QWEN3_06B_CONFIG.num_hidden_layers
    if layer_count != MODEL_LAYER_COUNT:
        raise RuntimeError("pinned Qwen profile layer count changed")
    if args.mode == "two-gpu" and not 0 < args.split_layer < layer_count:
        parser.error(f"split layer must be between 1 and {layer_count - 1}")
    first = torch.device("cuda", 0)
    ranges = (
        (QwenLayerRange(0, layer_count, first),)
        if args.mode == "one-gpu"
        else (
            QwenLayerRange(0, args.split_layer, first),
            QwenLayerRange(args.split_layer, layer_count, torch.device("cuda", 1)),
        )
    )
    placement = QwenLayerPlacement(ranges)
    state = load_tensors(args.model)
    tower = {key.removeprefix("model."): value for key, value in state.items()}
    if len(tower) != len(state) or any(not key.startswith("model.") for key in state):
        raise SystemExit("pinned Qwen checkpoint must contain only model.* tensors")
    with torch.device("meta"):
        model = QwenTextModel(ANIMA_QWEN3_06B_CONFIG)
    model.load_state_dict(tower, strict=True, assign=True)
    del state, tower
    enrolled = enroll_qwen_layer_placement(model, placement, offload_device="cpu")
    ResidencyManager().load(enrolled.mechanisms, force_full_load=True)
    devices = placement.devices
    for device in devices:
        torch.cuda.synchronize(device)
    weight_allocated = {str(device): torch.cuda.memory_allocated(device) for device in devices}

    tokenizer = load_qwen_bpe()
    provider = QwenGenerationProvider(
        model,
        tokenizer,
        MODEL_SHA256,
        block_tokens=256,
        max_device_blocks=16,
    )
    request = GenerationRequest(
        provider.id,
        MODEL_SHA256,
        prompt=args.prompt,
        stop=GenerationStopConditions(args.max_new_tokens),
        open_session=True,
    )

    def call() -> Run:
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        with provider.generate(request, cancelled=lambda: False) as stream:
            events = tuple(stream)
        terminal = events[-1]
        if not isinstance(terminal, GenerationTerminalEvent):
            raise RuntimeError("Qwen generation ended without a terminal event")
        result = terminal.result
        if (
            result.token_ids is None
            or result.stats.time_to_first_token_s is None
            or result.continuation is None
        ):
            raise RuntimeError("Qwen generation did not return complete benchmark evidence")
        provider.close_session(result.continuation)
        return Run(
            result.stats.total_time_s,
            result.stats.time_to_first_token_s,
            len(result.token_ids),
            _hash_text(result.text),
            _hash_ids(result.token_ids),
            {str(device): torch.cuda.max_memory_allocated(device) for device in devices},
            {str(device): torch.cuda.max_memory_reserved(device) for device in devices},
        )

    for _ in range(args.warmups):
        call()
    runs = tuple(call() for _ in range(args.repeats))
    if len({(run.generated_tokens, run.text_sha256, run.token_ids_sha256) for run in runs}) != 1:
        raise RuntimeError("repeated generation outputs differ")

    root = Path(__file__).resolve().parent.parent
    properties = tuple(torch.cuda.get_device_properties(device) for device in devices)
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "mode": args.mode,
        "artifact": {
            "path": str(args.model.resolve()),
            "bytes": MODEL_SIZE,
            "sha256": MODEL_SHA256,
            "source_url": MODEL_URL,
        },
        "workload": {
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "sampler": "greedy",
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "placement": {
            "split_layer": None if args.mode == "one-gpu" else args.split_layer,
            "ranges": [asdict(item) | {"device": str(item.device)} for item in placement.ranges],
        },
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpus": [
                {"device": str(device), "name": prop.name, "memory_bytes": prop.total_memory}
                for device, prop in zip(devices, properties, strict=True)
            ],
            "model_dtype": str(model.embed_tokens.weight.dtype),
            "dinkster_commit": _git_head(root),
            "dinkster_dirty": _git_dirty(root),
        },
        "runs": [asdict(run) for run in runs],
        "summary": _summary(runs, weight_allocated),
    }
    if args.compare_with is not None:
        report["comparison"] = _compare_reports(
            json.loads(args.compare_with.read_text()),
            report,
        )
        if (
            args.require_memory_reduction
            and cast("dict[str, float]", report["comparison"])["max_weight_memory_reduction"] <= 0
        ):
            raise SystemExit("two-GPU placement did not reduce maximum per-device weight memory")
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
