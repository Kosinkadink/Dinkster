"""Benchmark native Qwen3-30B-A3B direct and continuous generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__:
    from tools.benchmark_qwen_scheduler import (
        _continuous_batch,
        _git_dirty,
        _git_head,
        _measure,
        _serial_batch,
        _summary,
        _validate_outputs,
    )
else:
    from benchmark_qwen_scheduler import (
        _continuous_batch,
        _git_dirty,
        _git_head,
        _measure,
        _serial_batch,
        _summary,
        _validate_outputs,
    )

MODEL_REPOSITORY = "Qwen/Qwen3-30B-A3B"
MODEL_REVISION = "d47d535f78ec44bd57128f8e8aeba17eeb0285ea"
MODEL_TENSOR_BYTES = 61_064_245_248
MODEL_FILE_BYTES = 61_066_575_648
DEFAULT_PROMPT = "Write one concise sentence describing a quiet forest at sunrise."


@dataclass(frozen=True, slots=True)
class ArtifactShard:
    name: str
    size: int
    sha256: str

    @property
    def source_url(self) -> str:
        return f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{self.name}"


SHARDS = (
    ArtifactShard(
        "model-00001-of-00016.safetensors",
        3_999_417_504,
        "454e77b346a61bfb201d54df60e15158838cf930617ee135113556204f2802b5",
    ),
    ArtifactShard(
        "model-00002-of-00016.safetensors",
        3_999_974_192,
        "47f015d6e5bb1782a834d75c06113c5e8c77ddca1cb7daf89686a4ec0deda19e",
    ),
    ArtifactShard(
        "model-00003-of-00016.safetensors",
        3_997_360_832,
        "ac0bf5990f2da995c1e8b77a3149dee71900b4fe1b8a614231dea9647193d96b",
    ),
    ArtifactShard(
        "model-00004-of-00016.safetensors",
        3_999_975_056,
        "89b01fd34a683c70fdb7f22c2e7090538d65063335ba0db46f3654642b17d1e5",
    ),
    ArtifactShard(
        "model-00005-of-00016.safetensors",
        3_999_975_400,
        "9849eb3584d928e35bf5956ccfd91b64207467e1fad8304aae68dc4d92d6d6d9",
    ),
    ArtifactShard(
        "model-00006-of-00016.safetensors",
        3_999_975_400,
        "f7a0f1525557d740158fb692986ad43506e1b5901e7406ae65808c6457bc53ad",
    ),
    ArtifactShard(
        "model-00007-of-00016.safetensors",
        3_999_975_472,
        "920702a50f27a009e5f676da3b02978dbd794f6ad7854f301d709136154c1a84",
    ),
    ArtifactShard(
        "model-00008-of-00016.safetensors",
        3_997_362_064,
        "a85bf0cc8a8047c116172d4c08cf4792eb59d4375032708ba3e1a7ffca0f708a",
    ),
    ArtifactShard(
        "model-00009-of-00016.safetensors",
        3_999_975_408,
        "25cd8aaed86b8e17685efb152912f92448ea3fffb1bdc247f257f04d05907604",
    ),
    ArtifactShard(
        "model-00010-of-00016.safetensors",
        3_999_975_400,
        "7c3307ee214797c4bc61c0994f81d690e768150df5f14d18edf84e53ba4810dc",
    ),
    ArtifactShard(
        "model-00011-of-00016.safetensors",
        3_999_975_408,
        "c658cad2842d36fa4c7c7f726f8515dc5670a3d80c94e13257d4e322ffe61988",
    ),
    ArtifactShard(
        "model-00012-of-00016.safetensors",
        3_987_400_496,
        "8cb898bc5e78492600053d1105c9ff61d7a719f5cc2802df63e41535a652cc26",
    ),
    ArtifactShard(
        "model-00013-of-00016.safetensors",
        3_997_353_632,
        "599594421f314cb8e8ab4474db0eb490cc6310585067e82d73821254b93319ce",
    ),
    ArtifactShard(
        "model-00014-of-00016.safetensors",
        3_999_975_400,
        "66d3294e9976b5f01d117a0a0bed768f128a8898eb6937e224d051de2961d363",
    ),
    ArtifactShard(
        "model-00015-of-00016.safetensors",
        3_999_975_400,
        "2fe000da4fc7399ff6303bb0daec8c415a5d850f5ea53863392758c77fc37664",
    ),
    ArtifactShard(
        "model-00016-of-00016.safetensors",
        1_087_928_584,
        "0c979e314faf06e94a5e1841abd285d4e26e763c79c1abe7710d543192d9da48",
    ),
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_shards(directory: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for shard in SHARDS:
        path = directory / shard.name
        if not path.is_file():
            raise SystemExit(f"missing pinned Qwen3 MoE shard: {path}")
        if path.stat().st_size != shard.size:
            raise SystemExit(
                f"{shard.name} size differs from pinned {shard.size} bytes: {path.stat().st_size}"
            )
        digest = _digest(path)
        if digest != shard.sha256:
            raise SystemExit(f"{shard.name} sha256 differs from pinned {shard.sha256}: {digest}")
        paths.append(path)
    return tuple(paths)


def _start_rss_sampler(process: Any) -> tuple[threading.Event, threading.Thread, list[int]]:
    stop = threading.Event()
    samples = [process.memory_info().rss]

    def sample() -> None:
        while not stop.wait(0.002):
            samples.append(process.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    return stop, thread, samples


def _load_model(torch: Any, shards: tuple[Path, ...], device: str) -> tuple[Any, dict[str, Any]]:
    import psutil
    from dinkster_inference_torch import load_qwen3_moe_checkpoint

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    stop, sampler, rss_samples = _start_rss_sampler(process)
    started = time.perf_counter()
    try:
        model = load_qwen3_moe_checkpoint(shards, device=device)
    finally:
        stop.set()
        sampler.join()
    torch.cuda.synchronize(device)
    rss_after = process.memory_info().rss
    rss_samples.append(rss_after)
    evidence = {
        "elapsed_s": time.perf_counter() - started,
        "cuda_allocated_before_bytes": allocated_before,
        "cuda_allocated_after_bytes": torch.cuda.memory_allocated(device),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_reserved_before_bytes": reserved_before,
        "cuda_reserved_after_bytes": torch.cuda.memory_reserved(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_peak_bytes": max(rss_samples),
    }
    return model, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--slot-capacity", type=int, default=512)
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
        parser.error("the Qwen3 MoE generation benchmark requires a CUDA device")
    shards = _verify_shards(args.model_directory)

    import torch
    from dinkster_inference import GenerationRequest, GenerationStopConditions, load_qwen_bpe
    from dinkster_inference_torch import (
        QwenContinuousGenerationProvider,
        QwenGenerationProvider,
    )

    model, load_evidence = _load_model(torch, shards, args.device)
    tokenizer = load_qwen_bpe()
    prompt_tokens = len(tokenizer.encode(args.prompt))
    required = prompt_tokens + args.max_new_tokens
    if required > args.slot_capacity:
        parser.error(
            f"prompt and output need {required} positions, above slot capacity {args.slot_capacity}"
        )
    model_identity = f"{MODEL_REPOSITORY}@{MODEL_REVISION}"
    direct = QwenGenerationProvider(
        model,
        tokenizer,
        model_identity,
        eos_token_ids=(model.config.eos_token_id,),
        block_tokens=256,
        max_device_blocks=max(32, args.concurrency * 2),
    )
    request = GenerationRequest(
        direct.id,
        model_identity,
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
            _serial_batch(direct, request, 1)
            _serial_batch(direct, request, args.concurrency)
            _continuous_batch(scheduled, request, args.concurrency)
        singleton = [
            _measure(torch, args.device, lambda: _serial_batch(direct, request, 1))
            for _ in range(args.repeats)
        ]
        serial: list[Any] = []
        continuous: list[Any] = []
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
    cross_batch_output_match = _validate_outputs((*singleton, *serial), continuous)
    singleton_summary = _summary(singleton)
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
        "schema": "dinkster.qwen3-moe-generation-benchmark.v1",
        "artifact": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "tensor_bytes": MODEL_TENSOR_BYTES,
            "file_bytes": MODEL_FILE_BYTES,
            "shards": [
                {
                    **asdict(shard),
                    "path": str((args.model_directory / shard.name).resolve()),
                    "source_url": shard.source_url,
                }
                for shard in SHARDS
            ],
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
            "model_dtype": str(model.model.embed_tokens.weight.dtype),
            "dinkster_commit": _git_head(root),
            "dinkster_dirty": _git_dirty(root),
        },
        "model_load": load_evidence,
        "working_cache_bytes": working_cache_bytes,
        "error_count": 0,
        "singleton": {
            "runs": [asdict(run) for run in singleton],
            "summary": singleton_summary,
        },
        "serialized_concurrent": {
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
