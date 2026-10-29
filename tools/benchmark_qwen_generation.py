"""Run a matched raw-prompt Qwen3-0.6B generation benchmark.

Each invocation runs one backend so model weights do not overlap in VRAM.
Use the same prompt, token limit, warmup count, and repeat count for Dinkster,
the pinned ComfyUI checkout, and an OpenAI-compatible server hosting a
conversion of the same checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

MODEL_SIZE = 1_192_135_096
MODEL_SHA256 = "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
MODEL_URL = (
    "https://huggingface.co/circlestone-labs/Anima/resolve/"
    "e26179e4b23bcb3a9e91b4ad2961a76ab9644d43/"
    "split_files/text_encoders/qwen_3_06b_base.safetensors"
)
COMFYUI_COMMIT = "3ac5d7941dfa2504555260512132d1cb5664648d"
DEFAULT_PROMPT = "Write one concise sentence describing a quiet forest at sunrise."


@dataclass(frozen=True, slots=True)
class Run:
    total_s: float
    time_to_first_token_s: float | None
    prompt_tokens: int
    generated_tokens: int
    text_sha256: str
    token_ids_sha256: str | None


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


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty(path: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _hash_ids(token_ids: Sequence[int]) -> str:
    encoded = json.dumps(list(token_ids), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _collect(
    call: Callable[[], Run],
    *,
    warmups: int,
    repeats: int,
) -> list[Run]:
    for _ in range(warmups):
        call()
    return [call() for _ in range(repeats)]


def _load_state(path: Path, device: str) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise SystemExit(
            "benchmark backends need safetensors in the selected interpreter"
        ) from error
    return load_file(path, device=device)


def _torch_environment(torch: Any, device: str) -> dict[str, object]:
    selected = torch.device(device)
    environment: dict[str, object] = {
        "device": str(selected),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if selected.type == "cuda":
        properties = torch.cuda.get_device_properties(selected)
        environment.update(
            {
                "cuda": torch.version.cuda,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": properties.name,
                "gpu_memory_bytes": properties.total_memory,
            }
        )
    return environment


def _dinkster(
    args: argparse.Namespace,
) -> tuple[list[Run], dict[str, object]]:
    import torch
    from dinkster_inference import (
        ANIMA_QWEN3_06B_CONFIG,
        GenerationRequest,
        GenerationStopConditions,
        GenerationTerminalEvent,
        load_qwen_bpe,
    )
    from dinkster_inference_torch import QwenGenerationProvider, QwenTextModel

    state = _load_state(args.model, args.device)
    tower = {key.removeprefix("model."): value for key, value in state.items()}
    if len(tower) != len(state) or any(not key.startswith("model.") for key in state):
        raise SystemExit("pinned Qwen checkpoint must contain only model.* tensors")
    with torch.device("meta"):
        model = QwenTextModel(ANIMA_QWEN3_06B_CONFIG)
    model.load_state_dict(tower, strict=True, assign=True)
    del state, tower
    tokenizer = load_qwen_bpe()
    provider = QwenGenerationProvider(
        model,
        tokenizer,
        MODEL_SHA256,
        block_tokens=256,
        max_device_blocks=128,
    )
    request = GenerationRequest(
        provider.id,
        MODEL_SHA256,
        prompt=args.prompt,
        stop=GenerationStopConditions(args.max_new_tokens),
    )
    prompt_tokens = len(tokenizer.encode(args.prompt))

    def call() -> Run:
        with provider.generate(request, cancelled=lambda: False) as stream:
            events = tuple(stream)
        terminal = events[-1]
        if not isinstance(terminal, GenerationTerminalEvent):
            raise RuntimeError("Dinkster generation ended without a terminal event")
        result = terminal.result
        assert result.token_ids is not None
        assert result.stats.time_to_first_token_s is not None
        return Run(
            result.stats.total_time_s,
            result.stats.time_to_first_token_s,
            prompt_tokens,
            len(result.token_ids),
            _hash_text(result.text),
            _hash_ids(result.token_ids),
        )

    runs = _collect(call, warmups=args.warmups, repeats=args.repeats)
    environment = _torch_environment(torch, args.device)
    root = Path(__file__).resolve().parent.parent
    environment["dinkster_commit"] = _git_head(root)
    environment["dinkster_dirty"] = _git_dirty(root)
    environment["model_dtype"] = str(model.embed_tokens.weight.dtype)
    return runs, environment


def _comfyui(
    args: argparse.Namespace,
) -> tuple[list[Run], dict[str, object]]:
    if args.comfyui_root is None:
        raise SystemExit("--comfyui-root is required for the comfyui backend")
    root = args.comfyui_root.resolve()
    head = _git_head(root)
    if head != COMFYUI_COMMIT:
        raise SystemExit(f"ComfyUI must be detached at {COMFYUI_COMMIT}, found {head}")
    if _git_dirty(root):
        raise SystemExit("ComfyUI benchmark checkout must be clean")
    sys.path.insert(0, str(root))
    sys.argv = [sys.argv[0], "--disable-all-custom-nodes"]

    import torch
    from comfy import ops
    from comfy.text_encoders import llama
    from dinkster_inference import load_qwen_bpe

    state = _load_state(args.model, args.device)
    execution_dtype = state["model.embed_tokens.weight"].dtype
    with torch.device("meta"):
        model = llama.Qwen3_06B({}, execution_dtype, "meta", ops.disable_weight_init)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    tokenizer = load_qwen_bpe()
    prompt_ids = tokenizer.encode(args.prompt)
    ids = torch.tensor((prompt_ids,), dtype=torch.long, device=args.device)

    def call() -> Run:
        torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        with torch.inference_mode():
            embeds = model.model.embed_tokens(ids, out_dtype=execution_dtype)
            generated = model.generate(
                embeds=embeds,
                do_sample=False,
                max_length=args.max_new_tokens,
                stop_tokens=model.model.config.stop_tokens,
                initial_tokens=prompt_ids,
                execution_dtype=execution_dtype,
            )
        torch.cuda.synchronize(args.device)
        total = time.perf_counter() - started
        text = tokenizer.decode(generated)
        return Run(
            total,
            None,
            len(prompt_ids),
            len(generated),
            _hash_text(text),
            _hash_ids(generated),
        )

    runs = _collect(call, warmups=args.warmups, repeats=args.repeats)
    environment = _torch_environment(torch, args.device)
    environment["comfyui_commit"] = head
    environment["comfyui_dirty"] = _git_dirty(root)
    environment["model_dtype"] = str(execution_dtype)
    return runs, environment


def _external_direct(
    args: argparse.Namespace,
) -> tuple[list[Run], dict[str, object]]:
    if args.lm_model is None:
        raise SystemExit("--external-model is required for external backends")
    from dinkster_inference import load_qwen_bpe

    tokenizer = load_qwen_bpe()
    prompt_tokens = len(tokenizer.encode(args.prompt))

    def call() -> Run:
        payload = json.dumps(_external_greedy_payload(args)).encode()
        request = urllib.request.Request(
            args.lm_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        first: float | None = None
        text = ""
        reported_prompt_tokens: int | None = None
        reported_generated_tokens: int | None = None
        with urllib.request.urlopen(request, timeout=args.lm_timeout) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                packet = json.loads(line[6:])
                usage = packet.get("usage")
                if usage is not None:
                    reported_prompt_tokens = usage.get("prompt_tokens")
                    reported_generated_tokens = usage.get("completion_tokens")
                choices = packet.get("choices", ())
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("text", "")
                if delta and first is None:
                    first = time.perf_counter() - started
                text += delta
        total = time.perf_counter() - started
        generated = tokenizer.encode(text)
        return Run(
            total,
            first,
            prompt_tokens if reported_prompt_tokens is None else reported_prompt_tokens,
            len(generated) if reported_generated_tokens is None else reported_generated_tokens,
            _hash_text(text),
            None,
        )

    runs = _collect(call, warmups=args.warmups, repeats=args.repeats)
    root = Path(__file__).resolve().parent.parent
    return runs, {
        "client": "direct-urllib",
        "compatibility": _external_compatibility(args),
        "dinkster_commit": _git_head(root),
        "dinkster_dirty": _git_dirty(root),
        "host": platform.node(),
        "runtime": _external_runtime(args.backend, args.external_runtime),
        "runtime_model": args.lm_model,
        "runtime_url": _external_base_url(args.lm_url),
        "python": platform.python_version(),
    }


def _external_provider(
    args: argparse.Namespace,
) -> tuple[list[Run], dict[str, object]]:
    if args.lm_model is None:
        raise SystemExit("--external-model is required for external backends")
    from dinkster_inference import (
        GenerationRequest,
        GenerationStopConditions,
        GenerationTerminalEvent,
        OpenAICompatibility,
        OpenAIGenerationProvider,
        load_qwen_bpe,
    )

    tokenizer = load_qwen_bpe()
    fallback_prompt_tokens = len(tokenizer.encode(args.prompt))
    provider = OpenAIGenerationProvider(
        args.lm_base_url,
        args.lm_model,
        compatibility=OpenAICompatibility(_external_compatibility(args)),
        stream=True,
        timeout_s=args.lm_timeout,
    )
    request = GenerationRequest(
        provider.id,
        provider.model_identity,
        prompt=args.prompt,
        stop=GenerationStopConditions(args.max_new_tokens),
    )

    def call() -> Run:
        with provider.generate(request, cancelled=lambda: False) as stream:
            events = tuple(stream)
        terminal = events[-1]
        if not isinstance(terminal, GenerationTerminalEvent):
            raise RuntimeError("OpenAI provider generation ended without a terminal event")
        result = terminal.result
        generated = tokenizer.encode(result.text)
        return Run(
            result.stats.total_time_s,
            result.stats.time_to_first_token_s,
            (
                fallback_prompt_tokens
                if result.stats.prompt_tokens is None
                else result.stats.prompt_tokens
            ),
            (
                len(generated)
                if result.stats.generated_tokens is None
                else result.stats.generated_tokens
            ),
            _hash_text(result.text),
            None,
        )

    try:
        runs = _collect(call, warmups=args.warmups, repeats=args.repeats)
    finally:
        provider.close()
    root = Path(__file__).resolve().parent.parent
    return runs, {
        "client": "dinkster-openai-provider",
        "compatibility": _external_compatibility(args),
        "dinkster_commit": _git_head(root),
        "dinkster_dirty": _git_dirty(root),
        "host": platform.node(),
        "runtime": _external_runtime(args.backend, args.external_runtime),
        "runtime_model": args.lm_model,
        "runtime_url": args.lm_base_url.rstrip("/"),
        "python": platform.python_version(),
    }


def _external_runtime(backend: str, runtime: str | None) -> str:
    if backend.startswith("lm-studio"):
        return "LM Studio"
    if runtime is None or not runtime.strip():
        raise SystemExit("--external-runtime is required for openai backends")
    return runtime


def _external_greedy_payload(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": args.lm_model,
        "prompt": args.prompt,
        "max_tokens": args.max_new_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    if _external_compatibility(args) == "llama.cpp":
        payload.update(
            {
                "repeat_penalty": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "typical_p": 1.0,
                "samplers": ["temperature"],
            }
        )
    return payload


def _external_compatibility(args: argparse.Namespace) -> str:
    return "openai" if args.backend.startswith("lm-studio") else args.external_compatibility


def _external_base_url(completion_url: str) -> str:
    suffix = "/completions"
    normalized = completion_url.rstrip("/")
    if not normalized.endswith(suffix):
        raise SystemExit("--external-url must end in /completions")
    return normalized[: -len(suffix)]


def _external_artifact(args: argparse.Namespace) -> dict[str, object]:
    if args.external_artifact is None:
        raise SystemExit("--external-artifact is required for external backends")
    path = args.external_artifact.resolve()
    return {
        "runtime_model": args.lm_model,
        "converted_path": str(path),
        "converted_bytes": path.stat().st_size,
        "converted_sha256": _digest(path),
        "source_bytes": MODEL_SIZE,
        "source_sha256": MODEL_SHA256,
        "source_url": MODEL_URL,
    }


def _report(
    backend: str,
    args: argparse.Namespace,
    runs: list[Run],
    environment: dict[str, object],
) -> dict[str, object]:
    median_total = statistics.median(run.total_s for run in runs)
    median_generated = statistics.median(run.generated_tokens for run in runs)
    ttft = [run.time_to_first_token_s for run in runs if run.time_to_first_token_s is not None]
    return {
        "schema": "dinkster.qwen-generation-benchmark.v1",
        "backend": backend,
        "artifact": (
            {
                "path": str(args.model),
                "bytes": MODEL_SIZE,
                "sha256": MODEL_SHA256,
                "source_url": MODEL_URL,
            }
            if backend in ("dinkster", "comfyui")
            else _external_artifact(args)
        ),
        "workload": {
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "sampler": "greedy",
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "environment": environment,
        "runs": [asdict(run) for run in runs],
        "median": {
            "total_s": median_total,
            "time_to_first_token_s": statistics.median(ttft) if ttft else None,
            "generated_tokens_per_s": median_generated / median_total,
        },
    }


def _compare_external_reports(
    reference: object,
    candidate: object,
) -> dict[str, object]:
    if type(reference) is not dict or type(candidate) is not dict:
        raise ValueError("benchmark reports must be JSON objects")
    reference_report = cast("dict[str, object]", reference)
    candidate_report = cast("dict[str, object]", candidate)
    if reference_report.get("schema") != "dinkster.qwen-generation-benchmark.v1" or (
        candidate_report.get("schema") != "dinkster.qwen-generation-benchmark.v1"
    ):
        raise ValueError("benchmark report schema does not match")
    reference_backend = reference_report.get("backend")
    candidate_backend = candidate_report.get("backend")
    backend_pairs = {
        "lm-studio-provider": "lm-studio",
        "openai-provider": "openai",
    }
    if backend_pairs.get(candidate_backend) != reference_backend:
        raise ValueError("comparison requires direct and provider external backends")
    for field in ("artifact", "workload"):
        if reference_report.get(field) != candidate_report.get(field):
            raise ValueError(f"benchmark report {field} does not match")
    reference_environment = _report_object(reference_report.get("environment"), "environment")
    candidate_environment = _report_object(candidate_report.get("environment"), "environment")
    for field in (
        "compatibility",
        "dinkster_commit",
        "dinkster_dirty",
        "host",
        "runtime",
        "runtime_model",
        "runtime_url",
        "python",
    ):
        if reference_environment.get(field) != candidate_environment.get(field):
            raise ValueError(f"benchmark environment {field} does not match")
    reference_runs = _report_runs(reference_report.get("runs"))
    candidate_runs = _report_runs(candidate_report.get("runs"))
    if len(reference_runs) != len(candidate_runs):
        raise ValueError("benchmark run counts do not match")
    for index, (reference_run, candidate_run) in enumerate(
        zip(reference_runs, candidate_runs, strict=True)
    ):
        for field in ("prompt_tokens", "generated_tokens", "text_sha256"):
            if reference_run.get(field) != candidate_run.get(field):
                raise ValueError(f"benchmark run {index} {field} does not match")
    reference_median = _report_object(reference_report.get("median"), "median")
    candidate_median = _report_object(candidate_report.get("median"), "median")
    reference_total = _positive_number(reference_median.get("total_s"), "reference total_s")
    candidate_total = _positive_number(candidate_median.get("total_s"), "candidate total_s")
    reference_rate = _positive_number(
        reference_median.get("generated_tokens_per_s"),
        "reference generated_tokens_per_s",
    )
    candidate_rate = _positive_number(
        candidate_median.get("generated_tokens_per_s"),
        "candidate generated_tokens_per_s",
    )
    return {
        "reference_backend": reference_backend,
        "latency_ratio": candidate_total / reference_total,
        "throughput_ratio": candidate_rate / reference_rate,
    }


def _report_object(value: object, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"benchmark report {name} must be an object")
    return cast("dict[str, object]", value)


def _report_runs(value: object) -> list[dict[str, object]]:
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise ValueError("benchmark report runs must be a list of objects")
    return cast("list[dict[str, object]]", value)


def _positive_number(value: object, name: str) -> float:
    if type(value) is int:
        number = float(value)
    elif type(value) is float:
        number = value
    else:
        raise ValueError(f"benchmark report {name} must be positive")
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"benchmark report {name} must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "backend",
        choices=(
            "dinkster",
            "comfyui",
            "lm-studio",
            "lm-studio-provider",
            "openai",
            "openai-provider",
        ),
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--comfyui-root", type=Path)
    parser.add_argument(
        "--lm-url",
        "--external-url",
        dest="lm_url",
        default="http://127.0.0.1:1234/v1/completions",
    )
    parser.add_argument(
        "--lm-base-url",
        "--external-base-url",
        dest="lm_base_url",
        default="http://127.0.0.1:1234/v1",
    )
    parser.add_argument("--lm-model", "--external-model", dest="lm_model")
    parser.add_argument(
        "--lm-timeout", "--external-timeout", dest="lm_timeout", type=float, default=300.0
    )
    parser.add_argument("--external-runtime")
    parser.add_argument("--external-artifact", type=Path)
    parser.add_argument(
        "--external-compatibility",
        choices=("openai", "llama.cpp"),
        default="openai",
    )
    parser.add_argument("--compare-with", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("token count and repeats must be positive; warmups must be non-negative")
    if args.backend in ("dinkster", "comfyui"):
        if args.model is None:
            parser.error("--model is required for local backends")
        _verify_model(args.model)

    runners = {
        "dinkster": _dinkster,
        "comfyui": _comfyui,
        "lm-studio": _external_direct,
        "lm-studio-provider": _external_provider,
        "openai": _external_direct,
        "openai-provider": _external_provider,
    }
    runs, environment = runners[args.backend](args)
    report = _report(args.backend, args, runs, environment)
    if args.compare_with is not None:
        if args.backend not in ("lm-studio-provider", "openai-provider"):
            parser.error("--compare-with requires an external provider backend")
        try:
            reference = json.loads(args.compare_with.read_text())
            report["comparison"] = _compare_external_reports(reference, report)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            parser.error(f"external benchmark comparison failed: {error}")
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
