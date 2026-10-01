"""Measure fork-backed MiniMax H3 single-job multi-GPU execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import secrets
import shlex
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

BASELINES = {
    ("RipperPC", "guidance", "sdpa", "short"): 1.974,
    ("RipperPC", "guidance", "sdpa", "production"): 1.802,
    ("RipperPC", "sequence", "sdpa", "production"): 1.5404,
    ("RipperPC", "sequence", "dinkster_kitchen_int8", "production"): 1.436792013659844,
}
INT8_WEIGHT_BASELINES = {
    ("X570", "sequence", "sdpa", "production"): 1.327,
}
POLICIES = ("sdpa", "dinkster_kitchen_int8")
MODES = ("guidance", "sequence", "sparse")
ATTEMPT_GROUP = "minimax-h3-multigpu-receipt"
ATTEMPT = 1


class ReceiptError(RuntimeError):
    pass


class _CollectiveTimer:
    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.events: list[tuple[str, Any, Any, int, int]] = []

    @staticmethod
    def _tensor_bytes(value: object) -> int:
        if isinstance(value, (list, tuple)):
            return sum(_CollectiveTimer._tensor_bytes(item) for item in value)
        numel = getattr(value, "numel", None)
        element_size = getattr(value, "element_size", None)
        if not callable(numel) or not callable(element_size):
            return 0
        return int(cast("Any", numel)() * cast("Any", element_size)())

    def _peer_bytes(self, name: str, inputs: tuple[object, ...]) -> tuple[int, int]:
        index = 1 if name in ("all_gather", "all_to_all") else 0
        payload = self._tensor_bytes(inputs[index]) if len(inputs) > index else 0
        world_size = int(self.torch.distributed.get_world_size())
        if name == "all_gather":
            peer_bytes = payload * (world_size - 1)
        elif name == "all_to_all":
            peer_bytes = payload * (world_size - 1) // world_size
        else:
            peer_bytes = payload * 2 * (world_size - 1) // world_size
        return payload, peer_bytes

    def install(self) -> None:
        for name in ("all_gather", "all_reduce", "all_to_all"):
            operation = getattr(self.torch.distributed, name)

            def timed(
                *args: object,
                _name: str = name,
                _operation: Any = operation,
                **kwargs: object,
            ) -> Any:
                started = self.torch.cuda.Event(enable_timing=True)
                finished = self.torch.cuda.Event(enable_timing=True)
                started.record()
                result = _operation(*args, **kwargs)
                finished.record()
                payload_bytes, peer_bytes = self._peer_bytes(_name, args)
                self.events.append((_name, started, finished, payload_bytes, peer_bytes))
                return result

            setattr(self.torch.distributed, name, timed)

    def reset(self) -> None:
        self.events.clear()

    def elapsed_seconds(self) -> float:
        return sum(start.elapsed_time(end) for _, start, end, _, _ in self.events) / 1000.0

    def evidence(self) -> dict[str, dict[str, float | int]]:
        evidence: dict[str, dict[str, float | int]] = {}
        for name in ("all_gather", "all_reduce", "all_to_all"):
            events = [event for event in self.events if event[0] == name]
            evidence[name] = {
                "calls": len(events),
                "seconds": sum(start.elapsed_time(end) for _, start, end, _, _ in events) / 1000.0,
                "input_payload_bytes": sum(event[3] for event in events),
                "logical_peer_bytes": sum(event[4] for event in events),
            }
        return evidence


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *arguments), text=True, timeout=30
    ).strip()


def _tensor_sha256(value: Any) -> str:
    torch = cast("Any", importlib.import_module("torch"))
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _latent_hashes(samples: Any) -> dict[str, str]:
    return {role: _tensor_sha256(samples.by_role(role)) for role in samples.roles}


def _asset(path: Path, digest: str) -> Any:
    from dinkster_assets import AssetRef

    return AssetRef(digest, path.name, path.stat().st_size, resolver=_Resolver(path))


def _model_artifact(path: Path) -> dict[str, Any]:
    dtype_bytes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    with path.open("rb") as model_file:
        header_length_bytes = model_file.read(8)
        if len(header_length_bytes) != 8:
            raise ReceiptError("model is not a safetensors file")
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        header = json.loads(model_file.read(header_length))
        data_start = 8 + header_length
        quantization = {}
        for key, value in header.items():
            if key.endswith(".comfy_quant"):
                start, end = value["data_offsets"]
                model_file.seek(data_start + start)
                quantization[key.removesuffix(".comfy_quant")] = json.loads(
                    model_file.read(end - start)
                )
        model_file.seek(0)
        sha256 = hashlib.file_digest(model_file, "sha256")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    dtype_counts = Counter(value["dtype"] for value in tensors.values())
    quantized_weights = [
        key
        for key, value in tensors.items()
        if key.endswith(".weight")
        and value["dtype"] == "I8"
        and f"{key.removesuffix('.weight')}.comfy_quant" in tensors
    ]
    logical_weight_bytes = sum(
        math.prod(value["shape"]) * dtype_bytes[value["dtype"]]
        for key, value in tensors.items()
        if not key.endswith(".comfy_quant")
    )
    weight_dtypes = {value["dtype"] for key, value in tensors.items() if key.endswith(".weight")}
    convrot_configs = [
        quantization[key.removesuffix(".weight")]
        for key in quantized_weights
        if quantization.get(key.removesuffix(".weight"), {}).get("format") == "int8_tensorwise"
        and quantization[key.removesuffix(".weight")].get("convrot") is True
    ]
    if quantized_weights and len(convrot_configs) == len(quantized_weights):
        provider = "int8-convrot"
    elif weight_dtypes == {"BF16"}:
        provider = "bf16-linear"
    else:
        provider = "mixed-linear"
    return {
        "sha256": sha256.hexdigest(),
        "provider": provider,
        "logical_weight_bytes": logical_weight_bytes,
        "tensor_dtype_counts": dict(sorted(dtype_counts.items())),
        "quantized_linear_weights": len(quantized_weights),
        "quantization_format": "int8_tensorwise" if convrot_configs else None,
        "convrot_group_sizes": sorted(
            {int(config["convrot_groupsize"]) for config in convrot_configs}
        ),
    }


def _conditioning(torch: Any, seed: int) -> list[list[object]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn((1, 32, 5120), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    return [[value, {}]]


def _route(policy: str) -> tuple[Any, Any]:
    from dinkster_native.attention import create_attention_runtime
    from dinkster_protocol import (
        AttentionPolicy,
        AttentionPolicyConfig,
        derive_attention_route_token,
    )

    runtime = create_attention_runtime()
    token = derive_attention_route_token(
        runtime.capabilities,
        AttentionPolicyConfig(requested_policy=cast("AttentionPolicy", policy)),
    )
    return runtime, token


def _inventory() -> dict[str, Any]:
    from dinkster.compose import default_pack_specs

    packs = []
    for spec in default_pack_specs():
        for pack, info in sorted((spec.packs or {}).items()):
            packs.append(
                {
                    "pack": pack,
                    "artifact_digest": info.artifact_digest,
                    "source": info.source,
                    "version": info.version,
                }
            )
    distributions = {}
    for name in ("comfy-kitchen", "dinkster-inference"):
        distributions[name] = importlib.metadata.version(name)
    return {
        "executable": sys.executable,
        "packs": packs,
        "distributions": distributions,
    }


def _run_sample(
    model: Any,
    latent: dict[str, object],
    positive: object,
    negative: object,
    cfg: float,
    steps: int,
    seed: int,
    tensor_output: Path | None = None,
    collective_timer: _CollectiveTimer | None = None,
    video_vae: Any = None,
    audio_vae: Any = None,
) -> tuple[dict[str, Any], float, float, dict[str, Any]]:
    from dinkster_native.fork_nodes import (
        GenerationKSampler,
        GenerationVAEDecode,
        NativeVAEDecodeAudio,
    )

    torch = cast("Any", importlib.import_module("torch"))
    if collective_timer is not None:
        collective_timer.reset()
    torch.cuda.synchronize()
    cuda_started = torch.cuda.Event(enable_timing=True)
    cuda_finished = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    cuda_started.record()
    sampled = cast(
        "dict[str, Any]",
        GenerationKSampler.execute(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name="euler",
            scheduler="simple",
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=1.0,
        )["latent"],
    )
    cuda_finished.record()
    fence_started = time.perf_counter()
    torch.cuda.synchronize()
    fence_seconds = time.perf_counter() - fence_started
    cuda_sample_seconds = cuda_started.elapsed_time(cuda_finished) / 1000.0
    samples = sampled["samples"]
    sample_seconds = time.perf_counter() - started
    hashes: dict[str, Any] = {"latents": _latent_hashes(samples)}
    tensor_evidence = None
    if tensor_output is not None:
        latent_evidence = {role: samples.by_role(role).detach().cpu() for role in samples.roles}
        tensor_evidence = (
            {"latents": latent_evidence}
            if video_vae is not None or audio_vae is not None
            else latent_evidence
        )
    decode_seconds = 0.0
    if video_vae is not None and audio_vae is not None:
        decode_started = time.perf_counter()
        video = cast("Any", GenerationVAEDecode.execute(samples=sampled, vae=video_vae)["image"])
        audio = cast(
            "dict[str, Any]",
            NativeVAEDecodeAudio.execute(samples=sampled, vae=audio_vae)["audio"],
        )["waveform"]
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started
        hashes["decoded"] = {"video": _tensor_sha256(video), "audio": _tensor_sha256(audio)}
        if tensor_evidence is not None:
            tensor_evidence["decoded"] = {
                "video": video.detach().cpu(),
                "audio": audio.detach().cpu(),
            }
    elif video_vae is not None or audio_vae is not None:
        raise ReceiptError("receipt decoding requires both video and audio VAEs")
    communication_sync_seconds = (
        collective_timer.elapsed_seconds() if collective_timer is not None else 0.0
    )
    compute_seconds = cuda_sample_seconds - communication_sync_seconds
    timing_breakdown = {
        "compute_seconds": compute_seconds,
        "communication_sync_seconds": communication_sync_seconds,
        "compute_share": compute_seconds / cuda_sample_seconds,
        "communication_sync_share": communication_sync_seconds / cuda_sample_seconds,
        "fence_seconds": fence_seconds,
        "collectives": collective_timer.evidence() if collective_timer is not None else {},
        "cuda_sample_seconds": cuda_sample_seconds,
        "sample_seconds": sample_seconds,
        "decode_seconds": decode_seconds,
        "whole_job_seconds": sample_seconds + decode_seconds,
    }
    evidence_started = time.perf_counter()
    if tensor_output is not None:
        assert tensor_evidence is not None
        torch.save(tensor_evidence, tensor_output)
    return (
        hashes,
        sample_seconds + decode_seconds,
        time.perf_counter() - evidence_started,
        timing_breakdown,
    )


def run_worker(args: argparse.Namespace) -> int:
    worker_started = time.perf_counter()
    inventory = _inventory()
    torch = cast("Any", importlib.import_module("torch"))
    from dinkster_native.attention import (
        activate_distributed_attention,
        release_distributed_attention,
    )
    from dinkster_native.fork_nodes import (
        GenerationLoadDiffusionModel,
        NativeBlockSparseAttention,
        NativeEmptyMiniMaxH3AV,
        NativeLoadVae,
    )
    from dinkster_workers import ExecutionContext, use_execution_context

    if not torch.cuda.is_available():
        raise ReceiptError("MiniMax H3 receipt worker requires CUDA")
    setup_started = time.perf_counter()
    runtime, token = _route(args.policy)
    sparse_backend_calls = 0
    if args.sparse_enabled:
        sparse_backend = runtime.registry["comfy_kitchen_sol_chunked"]

        def counted_sparse_backend(*inputs: object, **options: object) -> object:
            nonlocal sparse_backend_calls
            sparse_backend_calls += 1
            return sparse_backend(*inputs, **options)

        runtime.registry["comfy_kitchen_sol_chunked"] = counted_sparse_backend
    load_started = time.perf_counter()
    model = cast(
        "Any",
        GenerationLoadDiffusionModel.execute(
            diffusion_model=_asset(args.model, args.model_digest), weight_dtype="default"
        )["model"],
    )
    model_load_seconds = time.perf_counter() - load_started
    video_vae = (
        NativeLoadVae.execute(vae=_asset(args.video_vae, args.video_vae_digest))["vae"]
        if args.video_vae is not None
        else None
    )
    audio_vae = (
        NativeLoadVae.execute(vae=_asset(args.audio_vae, args.audio_vae_digest))["vae"]
        if args.audio_vae is not None
        else None
    )
    model_size = int(model.model_size())
    loading_route = {
        "loader": "dinkster_native.fork_nodes.GenerationLoadDiffusionModel",
        "assign_loaded_weights": True,
        "mmap_backed_state_dict": bool(model.fast_disk),
        "direct_offloaded_weight_pinning": bool(model.pin_offloaded_weights),
        "dynamic_weight_loading": bool(model.is_dynamic()),
    }
    if args.sparse_enabled:
        model = cast(
            "Any",
            NativeBlockSparseAttention.execute(
                model=model,
                selection="sol-attn",
                start_percent=0.2,
                end_percent=1.0,
                dense_blocks="",
                min_tokens=12_288,
                extra_tokens=256,
                sink_conditioning="exact_kv_and_rows",
                **{"selection.tau": 1.3},  # pyright: ignore[reportArgumentType]
            )["MODEL"],
        )
    latent = cast(
        "dict[str, object]",
        NativeEmptyMiniMaxH3AV.execute(
            width=args.width, height=args.height, frame_count=args.frames
        )["latent"],
    )
    positive = _conditioning(torch, 220)
    negative = _conditioning(torch, 221) if args.mode == "guidance" else []
    cfg = 2.0 if args.mode == "guidance" else 1.0
    distributed = "DINKSTER_SINGLE_JOB_RANK" in os.environ
    context = ExecutionContext(
        arm="native",
        expected_execution_identity=None,
        attention_policy=cast("Any", args.policy),
        attention_route_token=token,
        attention_capabilities=runtime.capabilities,
        attention_runtime=runtime,
    )
    setup_seconds = time.perf_counter() - setup_started
    collective_timer = None
    if distributed and args.mode == "sequence":
        collective_timer = _CollectiveTimer(torch)
        collective_timer.install()
    if distributed:
        activate_distributed_attention(ATTEMPT_GROUP, ATTEMPT)
    try:
        with use_execution_context(context):
            cold_hashes, cold_seconds, evidence_write_seconds, cold_timing_breakdown = _run_sample(
                model,
                latent,
                positive,
                negative,
                cfg,
                args.steps,
                args.seed,
                args.tensor_output,
                collective_timer,
                video_vae,
                audio_vae,
            )
            cold_workflow_seconds = time.perf_counter() - worker_started - evidence_write_seconds
            warmup_runs = [
                _run_sample(
                    model,
                    latent,
                    positive,
                    negative,
                    cfg,
                    args.steps,
                    args.seed,
                    collective_timer=collective_timer,
                    video_vae=video_vae,
                    audio_vae=audio_vae,
                )
                for _ in range(args.warmups)
            ]
            measured_runs = [
                _run_sample(
                    model,
                    latent,
                    positive,
                    negative,
                    cfg,
                    args.steps,
                    args.seed,
                    collective_timer=collective_timer,
                    video_vae=video_vae,
                    audio_vae=audio_vae,
                )
                for _ in range(args.repeats)
            ]
    finally:
        if distributed:
            release_distributed_attention(ATTEMPT_GROUP, ATTEMPT)
    warmup_sample_seconds = [float(run[3]["sample_seconds"]) for run in warmup_runs]
    warmup_whole_job_seconds = [float(run[3]["whole_job_seconds"]) for run in warmup_runs]
    sample_seconds = [float(run[3]["sample_seconds"]) for run in measured_runs]
    whole_job_seconds = [float(run[3]["whole_job_seconds"]) for run in measured_runs]
    measured_breakdowns = [run[3] for run in measured_runs]
    median_sample_seconds = statistics.median(sample_seconds)
    median_whole_job_seconds = statistics.median(whole_job_seconds)
    if args.sparse_enabled and sparse_backend_calls == 0:
        raise ReceiptError("sparse candidate never executed its registry backend")
    properties = torch.cuda.get_device_properties(0)
    resources = cast("Any", importlib.import_module("resource"))
    loaded_size = int(model.loaded_size())
    result = {
        "status": "PASS",
        "mode": args.mode,
        "policy": args.policy,
        "distributed": distributed,
        "rank": int(os.environ["DINKSTER_SINGLE_JOB_RANK"]) if distributed else None,
        "world_size": int(os.environ["DINKSTER_SINGLE_JOB_WORLD_SIZE"]) if distributed else 1,
        "output_hashes": {
            "cold": cold_hashes,
            "warmups": [run[0] for run in warmup_runs],
            "measured": [run[0] for run in measured_runs],
        },
        "timing": {
            "worker_wall_seconds": time.perf_counter() - worker_started,
            "setup_seconds": setup_seconds,
            "model_load_seconds": model_load_seconds,
            "cold_workflow_seconds": cold_workflow_seconds,
            "cold_sample_seconds": cold_timing_breakdown["sample_seconds"],
            "cold_decode_seconds": cold_timing_breakdown["decode_seconds"],
            "cold_whole_job_seconds": cold_seconds,
            "excluded_evidence_write_seconds": evidence_write_seconds,
            "warmup_sample_seconds": warmup_sample_seconds,
            "warmup_whole_job_seconds": warmup_whole_job_seconds,
            "sample_seconds": sample_seconds,
            "whole_job_seconds": whole_job_seconds,
            "cold_compute_communication": cold_timing_breakdown,
            "warmup_compute_communication": [run[3] for run in warmup_runs],
            "measured_compute_communication": measured_breakdowns,
            "median_compute_seconds": statistics.median(
                run["compute_seconds"] for run in measured_breakdowns
            ),
            "median_communication_sync_seconds": statistics.median(
                run["communication_sync_seconds"] for run in measured_breakdowns
            ),
            "median_compute_share": statistics.median(
                run["compute_share"] for run in measured_breakdowns
            ),
            "median_communication_sync_share": statistics.median(
                run["communication_sync_share"] for run in measured_breakdowns
            ),
            "median_warm_workflow_seconds": median_whole_job_seconds,
            "median_sample_seconds": median_sample_seconds,
            "median_whole_job_seconds": median_whole_job_seconds,
            "jobs_per_hour": 3600.0 / median_whole_job_seconds,
        },
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "peak_host_rss_bytes": resources.getrusage(resources.RUSAGE_SELF).ru_maxrss * 1024,
        "model_residency": {
            "logical_weight_bytes": model_size,
            "loaded_bytes": loaded_size,
            "offloaded_bytes": model_size - loaded_size,
        },
        "loading_route": loading_route,
        "sparse_backend_calls": sparse_backend_calls,
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability(0)
            ),
            "gpu": properties.name,
            "gpu_uuid": str(properties.uuid),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
        },
        "inventory": inventory,
    }
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


def _source_environment(dinkster_root: Path, fork_root: Path) -> dict[str, str]:
    package_sources = sorted(
        str(path) for path in (dinkster_root / "packages").glob("*/src") if path.is_dir()
    )
    sources = (str(dinkster_root / "src"), *package_sources, str(fork_root))
    current = os.environ.get("PYTHONPATH")
    return {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join((*sources, *((current,) if current else ()))),
    }


def _spawn_worker(
    args: argparse.Namespace,
    result: Path,
    log: Path,
    environment: dict[str, str],
) -> subprocess.Popen[bytes]:
    command = (
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--model",
        str(args.model),
        "--model-digest",
        args.model_digest,
        "--result",
        str(result),
        "--mode",
        args.mode,
        "--policy",
        args.policy,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--frames",
        str(args.frames),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        *(("--video-vae", str(args.video_vae)) if args.video_vae else ()),
        *(("--video-vae-digest", args.video_vae_digest) if args.video_vae else ()),
        *(("--audio-vae", str(args.audio_vae)) if args.audio_vae else ()),
        *(("--audio-vae-digest", args.audio_vae_digest) if args.audio_vae else ()),
        *(("--sparse-enabled",) if args.sparse_enabled else ()),
        *(("--tensor-output", str(args.tensor_output)) if args.tensor_output else ()),
    )
    with log.open("wb") as output:
        return subprocess.Popen(
            command,
            cwd=args.dinkster_root,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )


def _wait(workers: Sequence[tuple[str, subprocess.Popen[bytes], Path]]) -> None:
    failures = []
    for name, worker, log in workers:
        returncode = worker.wait()
        if returncode:
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
            failures.append(f"{name} exited {returncode}:\n{tail}")
    if failures:
        raise ReceiptError("; ".join(failures))


def _load(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text()))


def _run_serial(
    args: argparse.Namespace,
    scratch: Path,
    *,
    sparse_enabled: bool = False,
    name: str | None = None,
    gpu: str | None = None,
    repeats: int | None = None,
    warmups: int | None = None,
) -> dict[str, Any]:
    name = name or ("sparse" if sparse_enabled else "serial")
    result = scratch / f"{name}.json"
    log = scratch / f"{name}.log"
    environment = _source_environment(args.dinkster_root, args.fork_root)
    environment = {
        key: value
        for key, value in environment.items()
        if not key.startswith("DINKSTER_SINGLE_JOB_")
    }
    environment["CUDA_VISIBLE_DEVICES"] = gpu or args.gpu[0]
    worker_args = argparse.Namespace(**vars(args))
    worker_args.repeats = repeats if repeats is not None else args.repeats
    worker_args.warmups = warmups if warmups is not None else args.warmups
    worker_args.sparse_enabled = sparse_enabled
    worker_args.tensor_output = scratch / f"{name}-tensors.pt" if args.mode == "sparse" else None
    worker = _spawn_worker(worker_args, result, log, environment)
    _wait(((name, worker, log),))
    loaded = _load(result)
    loaded["worker_log"] = _log_receipt(log)
    if worker_args.tensor_output is not None:
        loaded["_tensor_output"] = str(worker_args.tensor_output)
    return loaded


def _run_distributed(
    args: argparse.Namespace,
    scratch: Path,
    *,
    name: str = "candidate",
    gpus: Sequence[str] | None = None,
    repeats: int | None = None,
    warmups: int | None = None,
) -> list[dict[str, Any]]:
    if args.mode == "sparse":
        raise ReceiptError("sparse attention is measured as a one-GPU registry route")
    token = secrets.token_hex(16)
    selected_gpus = tuple(gpus or args.gpu)
    rendezvous = scratch / f"{name}-rendezvous"
    workers = []
    for rank, gpu in enumerate(selected_gpus):
        result = scratch / f"{name}-rank{rank}.json"
        environment = _source_environment(args.dinkster_root, args.fork_root)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "DINKSTER_SINGLE_JOB_RANK": str(rank),
                "DINKSTER_SINGLE_JOB_WORLD_SIZE": str(len(selected_gpus)),
                "DINKSTER_SINGLE_JOB_MULTI_GPU_MODE": args.mode,
                "DINKSTER_SINGLE_JOB_RENDEZVOUS": f"file://{rendezvous}",
                "DINKSTER_SINGLE_JOB_TOKEN": token,
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,GRAPH",
            }
        )
        log = scratch / f"{name}-rank{rank}.log"
        worker_args = argparse.Namespace(**vars(args))
        worker_args.repeats = repeats if repeats is not None else args.repeats
        worker_args.warmups = warmups if warmups is not None else args.warmups
        worker_args.sparse_enabled = False
        workers.append(
            (f"{name}-rank{rank}", _spawn_worker(worker_args, result, log, environment), log)
        )
    _wait(workers)
    results = []
    for rank in range(len(selected_gpus)):
        loaded = _load(scratch / f"{name}-rank{rank}.json")
        loaded["worker_log"] = _log_receipt(scratch / f"{name}-rank{rank}.log")
        results.append(loaded)
    return results


def _log_receipt(path: Path) -> dict[str, Any]:
    contents = path.read_bytes()
    lines = contents.decode(errors="replace").splitlines()
    transport_terms = ("NCCL", "P2P", "NET/", "Channel", "NVLS", "SHM")
    return {
        "sha256": hashlib.sha256(contents).hexdigest(),
        "bytes": len(contents),
        "transport_lines": [
            line for line in lines if any(term in line for term in transport_terms)
        ][-500:],
    }


def _nvidia_smi(*arguments: str) -> str:
    return subprocess.check_output(("nvidia-smi", *arguments), text=True, timeout=30).strip()


def _classify_host(gpus: Sequence[dict[str, Any]]) -> str:
    names = {gpu["environment"]["gpu"] for gpu in gpus}
    if names == {"NVIDIA GeForce RTX 4090"}:
        return "X570"
    if names == {"NVIDIA RTX PRO 6000 Blackwell Workstation Edition"}:
        return "RipperPC"
    return os.uname().nodename


def _median_seconds(result: dict[str, Any]) -> float:
    timing = result["timing"]
    return float(timing.get("median_whole_job_seconds", timing["median_sample_seconds"]))


def _performance(
    reference: dict[str, Any], candidate: Sequence[dict[str, Any]]
) -> dict[str, float]:
    reference_seconds = _median_seconds(reference)
    slowest_candidate = max(candidate, key=_median_seconds)
    candidate_seconds = _median_seconds(slowest_candidate)
    candidate_timing = slowest_candidate["timing"]
    world_size = len(candidate)
    return {
        "reference_median_seconds": reference_seconds,
        "candidate_median_seconds": candidate_seconds,
        "candidate_compute_seconds": float(candidate_timing["median_compute_seconds"]),
        "candidate_communication_sync_seconds": float(
            candidate_timing["median_communication_sync_seconds"]
        ),
        "candidate_compute_share": float(candidate_timing["median_compute_share"]),
        "candidate_communication_sync_share": float(
            candidate_timing["median_communication_sync_share"]
        ),
        "speedup": reference_seconds / candidate_seconds,
        "reference_jobs_per_hour": 3600.0 / reference_seconds,
        "candidate_jobs_per_hour": 3600.0 / candidate_seconds,
        "reference_gpu_seconds_per_job": reference_seconds,
        "candidate_gpu_seconds_per_job": candidate_seconds * world_size,
    }


def _require_identical_matrix_hashes(segments: Sequence[dict[str, Any]]) -> None:
    expected: str | None = None
    expected_location = ""
    for segment_index, segment in enumerate(segments):
        for rank, result in enumerate(segment["ranks"]):
            hashes = result["output_hashes"]
            runs = (
                ("cold", hashes["cold"]),
                *((f"warmup-{index}", value) for index, value in enumerate(hashes["warmups"])),
                *((f"measured-{index}", value) for index, value in enumerate(hashes["measured"])),
            )
            for run, value in runs:
                location = f"segment {segment_index} {segment['arm']} rank {rank} {run}"
                encoded = json.dumps(value, sort_keys=True)
                if expected is None:
                    expected = encoded
                    expected_location = location
                elif encoded != expected:
                    raise ReceiptError(
                        f"output hashes differ: {location} does not match {expected_location}"
                    )


def _matrix_arm_summary(segments: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    arm_segments = [segment for segment in segments if segment["arm"] == arm]
    job_seconds: list[float] = []
    load_balance: list[dict[str, float]] = []
    for segment in arm_segments:
        rank_times = [result["timing"]["whole_job_seconds"] for result in segment["ranks"]]
        for run_times in zip(*rank_times, strict=True):
            values = [float(value) for value in run_times]
            slowest = max(values)
            fastest = min(values)
            job_seconds.append(slowest)
            load_balance.append(
                {
                    "slowest_rank_seconds": slowest,
                    "fastest_rank_seconds": fastest,
                    "spread_seconds": slowest - fastest,
                    "slowest_to_fastest_ratio": slowest / fastest,
                }
            )
    world_size = len(arm_segments[0]["ranks"])
    median_seconds = statistics.median(job_seconds)
    return {
        "world_size": world_size,
        "measured_whole_job_seconds": job_seconds,
        "median_whole_job_seconds": median_seconds,
        "jobs_per_hour": 3600.0 / median_seconds,
        "gpu_seconds_per_job": median_seconds * world_size,
        "load_balance": load_balance,
        "peak_allocated_bytes_by_rank": [
            int(result["peak_allocated_bytes"])
            for segment in arm_segments
            for result in segment["ranks"]
        ],
        "peak_reserved_bytes_by_rank": [
            int(result["peak_reserved_bytes"])
            for segment in arm_segments
            for result in segment["ranks"]
        ],
    }


def _run_sequence_matrix(
    args: argparse.Namespace, scratch: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    order = ("serial", "u2", "u4", "u4", "u2", "serial")
    first_repeats = (args.repeats + 1) // 2
    second_repeats = args.repeats // 2
    occurrences = Counter[str]()
    segments: list[dict[str, Any]] = []
    for block, arm in enumerate(order):
        occurrence = occurrences[arm]
        occurrences[arm] += 1
        repeats = first_repeats if occurrence == 0 else second_repeats
        warmups = args.warmups if occurrence == 0 else 0
        name = f"block{block}-{arm}"
        if arm == "serial":
            ranks = [
                _run_serial(
                    args,
                    scratch,
                    name=name,
                    gpu=args.gpu[0],
                    repeats=repeats,
                    warmups=warmups,
                )
            ]
            gpus = [args.gpu[0]]
        else:
            gpus = args.u2_gpu if arm == "u2" else args.gpu
            ranks = _run_distributed(
                args,
                scratch,
                name=name,
                gpus=gpus,
                repeats=repeats,
                warmups=warmups,
            )
        segments.append(
            {
                "block": block,
                "arm": arm,
                "gpu_uuids": list(gpus),
                "warmups": warmups,
                "repeats": repeats,
                "ranks": ranks,
            }
        )
    _require_identical_matrix_hashes(segments)
    summaries = {arm: _matrix_arm_summary(segments, arm) for arm in ("serial", "u2", "u4")}
    summaries["u2"]["speedup_vs_serial"] = (
        summaries["serial"]["median_whole_job_seconds"]
        / summaries["u2"]["median_whole_job_seconds"]
    )
    summaries["u4"]["speedup_vs_serial"] = (
        summaries["serial"]["median_whole_job_seconds"]
        / summaries["u4"]["median_whole_job_seconds"]
    )
    summaries["u4"]["speedup_vs_u2"] = (
        summaries["u2"]["median_whole_job_seconds"] / summaries["u4"]["median_whole_job_seconds"]
    )
    return segments, summaries


def _sparse_quality_oracle(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    torch = cast("Any", importlib.import_module("torch"))
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if not isinstance(reference, dict) or set(reference) != set(candidate):
        raise ReceiptError("sparse quality tensors have different roles")
    metrics = {}
    for role in sorted(reference):
        expected = reference[role]
        actual = candidate[role]
        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            raise ReceiptError("sparse quality output is not a tensor")
        if expected.shape != actual.shape:
            raise ReceiptError("sparse quality tensors have different shapes")
        expected = expected.float().flatten()
        actual = actual.float().flatten()
        difference = actual - expected
        reference_rms = float(expected.square().mean().sqrt())
        metrics[role] = {
            "shape": list(reference[role].shape),
            "max_abs": float(difference.abs().max()),
            "rmse": float(difference.square().mean().sqrt()),
            "relative_rmse": (
                float(difference.square().mean().sqrt()) / reference_rms if reference_rms else None
            ),
            "cosine_similarity": float(
                torch.nn.functional.cosine_similarity(expected, actual, dim=0)
            ),
        }
    return {
        "baseline": "same-session dense SDPA with identical model, inputs, sampler, and seed",
        "candidate": "fork registry comfy_kitchen_sol_chunked",
        "metrics_by_role": metrics,
    }


def _mint_sequence_matrix(args: argparse.Namespace, artifact: dict[str, Any]) -> int:
    with tempfile.TemporaryDirectory(prefix="minimax-h3-u4-matrix-") as directory:
        segments, summaries = _run_sequence_matrix(args, Path(directory))
    workload = (
        "production"
        if (args.width, args.height, args.frames, args.steps) == (1344, 768, 124, 20)
        else "short"
    )
    u4_beats_u2 = (
        summaries["u4"]["median_whole_job_seconds"] < summaries["u2"]["median_whole_job_seconds"]
    )
    receipt = {
        "schema": "dinkster.minimax-h3-sequence-matrix.v1",
        "status": "PASS",
        "invocation": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "working_directory": os.getcwd(),
        },
        "source": {
            "dinkster_head": _git(args.dinkster_root, "rev-parse", "HEAD"),
            "fork_head": _git(args.fork_root, "rev-parse", "HEAD"),
        },
        "artifacts": {
            "model": {
                "path": str(args.model),
                "bytes": args.model.stat().st_size,
                "blake3": args.model_digest.removeprefix("blake3:"),
                **artifact,
                "source_url": args.model_source_url,
                "source_revision": args.model_source_revision,
            },
            "video_vae": {
                "path": str(args.video_vae),
                "blake3": args.video_vae_digest.removeprefix("blake3:"),
            },
            "audio_vae": {
                "path": str(args.audio_vae),
                "blake3": args.audio_vae_digest.removeprefix("blake3:"),
            },
        },
        "workload": {
            "name": workload,
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "steps": args.steps,
            "seed": args.seed,
            "warmups_per_arm": args.warmups,
            "measured_repeats_per_arm": args.repeats,
            "sampler": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "guidance": 1.0,
            "conditioning": "synthetic 32-token bfloat16 context with seed 220",
        },
        "execution": {
            "mode": "sequence",
            "attention_policy": args.policy,
            "execution_provider": artifact["provider"],
            "compute_dtype": "bfloat16",
            "arm_order": [segment["arm"] for segment in segments],
            "order_balance": "mirrored serial,U2,U4,U4,U2,serial blocks",
            "requested_gpu_uuids": args.gpu,
            "u2_gpu_uuids": args.u2_gpu,
            "hash_contract": (
                "byte-identical video/audio latents and decoded outputs across every arm, "
                "rank, cold run, warmup, and measured repeat"
            ),
            "measurement_boundaries": {
                "whole_job": "one KSampler invocation followed by video and audio VAE decode",
                "sample": "synchronized KSampler wall time",
                "decode": "synchronized video and audio VAE decode wall time",
                "compute": "KSampler CUDA event time minus CUDA-event collective time",
                "collectives": (
                    "CUDA event time per operation; input payload bytes and logical peer bytes "
                    "use collective semantics rather than a provider-specific wire algorithm"
                ),
                "fence": "host wait in the CUDA synchronization after KSampler returns",
            },
            "arms": summaries,
            "provider_cells": {
                "ulysses4": {"status": "measured"},
                "ring4": {
                    "status": "provider-unavailable",
                    "reason": (
                        "selected dense provider does not expose online max and exp-sum "
                        "merge statistics"
                    ),
                },
                "ulysses2_ring2": {
                    "status": "provider-unavailable",
                    "reason": (
                        "selected dense provider does not expose online max and exp-sum "
                        "merge statistics"
                    ),
                },
            },
            "production_route": (
                "explicit-u4"
                if workload == "production" and u4_beats_u2
                else "u2-fallback"
                if workload == "production"
                else "not-decided-by-short-workload"
            ),
            "u4_beats_u2": u4_beats_u2,
        },
        "environment": {
            "hostname": os.uname().nodename,
            "nvidia_smi_topology": _nvidia_smi("topo", "-m"),
            "nvidia_smi_p2p_read": _nvidia_smi("topo", "-p2p", "r"),
            "nvidia_smi_p2p_write": _nvidia_smi("topo", "-p2p", "w"),
            "nvidia_smi_inventory": _nvidia_smi(
                "--query-gpu=index,name,uuid,pci.bus_id,memory.total",
                "--format=csv,noheader",
            ),
        },
        "segments": segments,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"PASS: {args.output}")
    return 0


def run_mint(args: argparse.Namespace) -> int:
    from dinkster_assets import digest_file

    args.dinkster_root = args.dinkster_root.resolve()
    args.fork_root = args.fork_root.resolve()
    args.model = args.model.resolve()
    args.matrix = getattr(args, "matrix", False)
    args.u2_gpu = getattr(args, "u2_gpu", None)
    args.video_vae = getattr(args, "video_vae", None)
    args.audio_vae = getattr(args, "audio_vae", None)
    if args.video_vae is not None:
        args.video_vae = args.video_vae.resolve()
    if args.audio_vae is not None:
        args.audio_vae = args.audio_vae.resolve()
    if args.matrix and args.mode != "sequence":
        raise ReceiptError("the serial/U2/U4 matrix requires sequence mode")
    if args.matrix and (len(args.gpu) != 4 or len(set(args.gpu)) != 4):
        raise ReceiptError("the serial/U2/U4 matrix requires exactly four unique GPU UUIDs")
    if args.matrix and (
        args.u2_gpu is None
        or len(args.u2_gpu) != 2
        or len(set(args.u2_gpu)) != 2
        or not set(args.u2_gpu).issubset(args.gpu)
    ):
        raise ReceiptError("the matrix requires two unique --u2-gpu UUIDs selected from --gpu")
    if args.matrix and (args.repeats < 5 or args.warmups != 1):
        raise ReceiptError("the matrix requires one warmup and at least five measured repeats")
    if args.matrix and (args.video_vae is None or args.audio_vae is None):
        raise ReceiptError("the matrix requires video and audio VAEs")
    if not args.matrix and len(args.gpu) != 2 and args.mode != "sparse":
        raise ReceiptError("guidance and U2R1 receipt minting require exactly two GPU UUIDs")
    if args.mode == "sparse" and len(args.gpu) != 1:
        raise ReceiptError("sparse attention receipt minting requires exactly one GPU UUID")
    if args.mode == "sparse" and args.policy != "sdpa":
        raise ReceiptError("sparse attention uses SDPA as its dense reference policy")
    if not args.model.is_file():
        raise ReceiptError(f"model does not exist: {args.model}")
    for name in ("video_vae", "audio_vae"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            raise ReceiptError(f"{name.replace('_', ' ')} does not exist: {path}")
    for root in (args.dinkster_root, args.fork_root):
        if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
            raise ReceiptError(f"source checkout is dirty: {root}")
    args.model_digest = digest_file(args.model)
    args.video_vae_digest = digest_file(args.video_vae) if args.video_vae is not None else None
    args.audio_vae_digest = digest_file(args.audio_vae) if args.audio_vae is not None else None
    artifact = _model_artifact(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.matrix:
        return _mint_sequence_matrix(args, artifact)
    with tempfile.TemporaryDirectory(prefix="minimax-h3-multigpu-") as directory:
        scratch = Path(directory)
        arm_order = (
            ("candidate", "reference") if args.candidate_first else ("reference", "candidate")
        )
        arms: dict[str, Any] = {}
        for arm in arm_order:
            if arm == "reference":
                arms[arm] = _run_serial(args, scratch)
            elif args.mode == "sparse":
                arms[arm] = [_run_serial(args, scratch, sparse_enabled=True)]
            else:
                arms[arm] = _run_distributed(args, scratch)
        serial = cast("dict[str, Any]", arms["reference"])
        candidate = cast("list[dict[str, Any]]", arms["candidate"])
        sparse_quality_oracle = (
            _sparse_quality_oracle(
                Path(serial.pop("_tensor_output")),
                Path(candidate[0].pop("_tensor_output")),
            )
            if args.mode == "sparse"
            else None
        )
    if args.mode != "sparse":
        hashes = {
            json.dumps(result["output_hashes"], sort_keys=True) for result in (serial, *candidate)
        }
        if len(hashes) != 1:
            raise ReceiptError("serial and distributed output hashes differ")
    performance = _performance(serial, candidate)
    speedup = performance["speedup"]
    host = _classify_host(candidate)
    workload = (
        "production"
        if (args.width, args.height, args.frames, args.steps) == (1344, 768, 124, 20)
        else "short"
    )
    baseline = (
        INT8_WEIGHT_BASELINES.get((host, args.mode, args.policy, workload))
        if artifact["provider"] == "int8-convrot"
        else BASELINES.get((host, args.mode, args.policy, workload))
    )
    receipt = {
        "schema": "dinkster.minimax-h3-multigpu-receipt.v3",
        "status": "PASS",
        "invocation": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "working_directory": os.getcwd(),
        },
        "source": {
            "dinkster_head": _git(args.dinkster_root, "rev-parse", "HEAD"),
            "fork_head": _git(args.fork_root, "rev-parse", "HEAD"),
        },
        "model": {
            "path": str(args.model),
            "bytes": args.model.stat().st_size,
            "blake3": args.model_digest.removeprefix("blake3:"),
            **artifact,
            "source_url": args.model_source_url,
            "source_revision": args.model_source_revision,
        },
        "workload": {
            "name": workload,
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "steps": args.steps,
            "seed": args.seed,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "sampler": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "guidance": 2.0 if args.mode == "guidance" else 1.0,
            "conditioning": "synthetic 32-token bfloat16 context with seed 220",
            "sparse_config": (
                {
                    "selection": "sol-attn",
                    "start_percent": 0.2,
                    "end_percent": 1.0,
                    "dense_blocks": [],
                    "min_tokens": 12_288,
                    "extra_tokens": 256,
                    "sink_conditioning": "exact_kv_and_rows",
                    "tau": 1.3,
                }
                if args.mode == "sparse"
                else None
            ),
        },
        "execution": {
            "mode": args.mode,
            "execution_provider": artifact["provider"],
            "compute_dtype": "bfloat16",
            "attention_policy": args.policy,
            "attention_provider": {
                "sdpa": "torch-sdpa-priority-v1",
                "dinkster_kitchen_int8": "comfy-kitchen-int8-attention-v1",
            }[args.policy],
            "sparse_backend": "comfy_kitchen_sol_chunked" if args.mode == "sparse" else None,
            "world_size": len(candidate),
            "arm_order": list(arm_order),
            "measurement_boundaries": {
                "cold_workflow": (
                    "worker process start through model load, setup, and first KSampler output"
                ),
                "warm_workflow": "one warm KSampler node invocation from resident inputs",
                "sampling_only": "the same KSampler invocation; VAE decode is excluded",
                "wall_time": "synchronized host wall clock around each KSampler invocation",
                "communication_sync": (
                    "CUDA event time around all_gather, all_reduce, and all_to_all collectives"
                ),
                "compute": "synchronized sampling wall time minus communication_sync",
            },
            "reference_output_hashes": serial["output_hashes"],
            "candidate_output_hashes": candidate[0]["output_hashes"],
            "hash_contract": (
                "all run hashes retained; sparse output may differ from dense reference"
                if args.mode == "sparse"
                else "bit-identical between serial and every distributed rank"
            ),
            "sparse_quality_oracle": sparse_quality_oracle,
            "performance": performance,
            "pre_reset_speedup": baseline,
            "speedup_ratio_to_pre_reset": speedup / baseline if speedup and baseline else None,
        },
        "environment": {
            "host": host,
            "hostname": os.uname().nodename,
            "nvidia_smi_topology": _nvidia_smi("topo", "-m"),
            "nvidia_smi_p2p_read": _nvidia_smi("topo", "-p2p", "r"),
            "nvidia_smi_p2p_write": _nvidia_smi("topo", "-p2p", "w"),
            "nvidia_smi_inventory": _nvidia_smi(
                "--query-gpu=index,name,uuid,pci.bus_id,memory.total",
                "--format=csv,noheader",
            ),
            "requested_gpu_uuids": args.gpu,
        },
        "reference": serial,
        "candidate_ranks": candidate,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"PASS: {args.output}")
    return 0


def _workload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video-vae", type=Path)
    parser.add_argument("--audio-vae", type=Path)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sparse-enabled", action="store_true", help=argparse.SUPPRESS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    _workload_arguments(worker)
    worker.add_argument("--model-digest", required=True)
    worker.add_argument("--video-vae-digest", help=argparse.SUPPRESS)
    worker.add_argument("--audio-vae-digest", help=argparse.SUPPRESS)
    worker.add_argument("--result", type=Path, required=True)
    worker.add_argument("--tensor-output", type=Path, help=argparse.SUPPRESS)
    worker.set_defaults(function=run_worker)
    mint = commands.add_parser("mint")
    _workload_arguments(mint)
    mint.add_argument("--dinkster-root", type=Path, required=True)
    mint.add_argument("--fork-root", type=Path, required=True)
    mint.add_argument("--model-source-url", required=True)
    mint.add_argument("--model-source-revision", required=True)
    mint.add_argument("--gpu", action="append", required=True)
    mint.add_argument("--u2-gpu", action="append")
    mint.add_argument("--output", type=Path, required=True)
    mint.add_argument("--candidate-first", action="store_true")
    mint.add_argument("--matrix", action="store_true")
    mint.set_defaults(function=run_mint, tensor_output=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
