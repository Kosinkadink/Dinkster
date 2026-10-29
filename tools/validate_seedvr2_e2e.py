"""Run paired official-weight SeedVR2 workflows in ComfyUI and Dinkster."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.validate_seedvr2_official import (  # noqa: E402
    COMFYUI_COMMIT,
    OFFICIAL_REVISION,
    OFFICIAL_SOURCE,
    FixedResolver,
    _artifact,
    _gpu1_claim,
    _identity,
    _load_components,
    _offline_launch,
    _persist,
    _tensor_record,
    _verify_comfy_root,
)

for source_root in sorted((REPO_ROOT / "packages").glob("*/src")):
    sys.path.insert(0, str(source_root))
sys.path.insert(0, str(REPO_ROOT / "src"))

THREAD_ID = "T-01a05194-5f4d-769b-8629-8c0545c5d174"
ACCELERATOR_HEADROOM_BYTES = 400 * 1024 * 1024
SEED = 959948902156062
STEPS = 1
CFG = 1.0
SAMPLER = "euler"
SCHEDULER = "simple"
DENOISE = 1.0
TILE_SIZE = 512
TILE_OVERLAP = 128
TEMPORAL_SIZE = 4096
TEMPORAL_OVERLAP = 8


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    path: Path
    sha256: str
    variant: str
    repository_path: str


@dataclass(frozen=True, slots=True)
class CaseSpec:
    name: str
    diffusion: str
    frames: int
    height: int
    width: int
    channels: int
    target_free_gib: int | None = None
    expected_oom: bool = False


def case_specs() -> tuple[CaseSpec, ...]:
    return (
        CaseSpec("3b_swiglu_image", "3b_swiglu", 1, 512, 768, 4),
        CaseSpec("3b_swiglu_video", "3b_swiglu", 5, 512, 768, 3),
        CaseSpec("7b_mlp_int8_image", "7b_mlp_int8", 1, 512, 768, 4),
        CaseSpec("7b_mlp_sharp_video", "7b_mlp_sharp", 5, 256, 384, 3),
        CaseSpec("7b_mlp_sharp_constrained_oom", "7b_mlp_sharp", 5, 256, 384, 3, 12, True),
    )


def _rss() -> dict[str, int]:
    fields: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition(":")
        if separator and name in ("VmRSS", "VmHWM"):
            fields[name] = int(value.split()[0]) * 1024
    if set(fields) != {"VmRSS", "VmHWM"}:
        raise RuntimeError("could not read process RSS and high-water mark")
    return {"current_bytes": fields["VmRSS"], "peak_bytes": fields["VmHWM"]}


def _trim_heap() -> bool:
    malloc_trim = ctypes.CDLL(None).malloc_trim
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    return bool(malloc_trim(0))


def _source(torch: Any, case: CaseSpec) -> Any:
    frame = torch.arange(case.frames, dtype=torch.int64)[:, None, None]
    y = torch.arange(case.height, dtype=torch.int64)[None, :, None]
    x = torch.arange(case.width, dtype=torch.int64)[None, None, :]
    channels = (
        (x * 3 + y * 5 + frame * 17) % 256,
        (x * 11 + y * 7 + frame * 29) % 256,
        (x * 13 + y * 19 + frame * 31) % 256,
    )
    if case.channels == 4:
        channels += ((x * 23 + y * 37 + frame * 41) % 256,)
    return torch.stack(channels, dim=-1).to(dtype=torch.float32).div_(255.0)


def _cuda_memory(torch: Any) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(0)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(0),
        "reserved_bytes": torch.cuda.memory_reserved(0),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "driver_free_bytes": free,
        "driver_total_bytes": total,
    }


def _mechanisms(handle: object) -> list[dict[str, object]]:
    return [
        {
            "type": type(mechanism).__name__,
            "demand_paged": bool(mechanism.demand_paged),
            "total_bytes": mechanism.total_bytes(),
            "loaded_bytes": mechanism.loaded_bytes(),
            "offloaded_bytes": mechanism.offloaded_bytes(),
        }
        for mechanism in handle.mechanisms  # type: ignore[attr-defined]
    ]


def _asset(path: Path, digest: str, size: int) -> object:
    from dinkster_assets import AssetRef

    resolved = path.resolve(strict=True)
    if resolved.stat().st_size != size:
        raise RuntimeError(f"artifact size changed for {resolved}")
    return AssetRef(
        digest=digest,
        name=resolved.name,
        size=size,
        media_type="application/x-safetensors",
        virtual_path=f"models/{resolved.name}",
        resolver=FixedResolver(resolved, digest),
    )


def _configure_comfy(
    comfy_root: Path,
    *,
    low_memory: bool,
    offline: bool = False,
) -> tuple[Any, Any, Any, Any]:
    root = _verify_comfy_root(comfy_root)
    sys.path.insert(0, str(root))
    saved_argv = sys.argv
    argv = [
        str(root / "main.py"),
        "--use-pytorch-cross-attention",
        "--async-offload",
        "1",
    ]
    if offline:
        argv.append("--cpu")
    elif low_memory:
        argv.append("--lowvram")
    sys.argv = argv
    try:
        import comfy.options as options

        options.enable_args_parsing()
        import comfy.model_management as model_management
        import comfy.sd as comfy_sd
        import comfy.utils as comfy_utils
        import nodes
    finally:
        sys.argv = saved_argv
    return comfy_sd, comfy_utils, model_management, nodes


def _load_comfy_components(
    comfy_sd: Any,
    comfy_utils: Any,
    diffusion_path: Path,
    vae_path: Path,
    torch: Any,
) -> tuple[object, object]:
    model = comfy_sd.load_diffusion_model(
        str(diffusion_path),
        model_options={"dtype": torch.bfloat16},
    )
    state, metadata = comfy_utils.load_torch_file(str(vae_path), return_metadata=True)
    vae = comfy_sd.VAE(sd=state, metadata=metadata, dtype=torch.float16)
    del state
    vae.throw_exception_if_invalid()
    return model, vae


def _comfy_preflight(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the ComfyUI launch preflight")
    comfy_sd, comfy_utils, _model_management, _nodes = _configure_comfy(
        args.comfy_root,
        low_memory=args.target_free_gib is not None,
        offline=True,
    )
    calls: list[dict[str, object]] = []
    model_sentinel = object()

    class VaeValue:
        @staticmethod
        def throw_exception_if_invalid() -> None:
            pass

    vae_sentinel = VaeValue()
    original_model = comfy_sd.load_diffusion_model
    original_file = comfy_utils.load_torch_file
    original_vae = comfy_sd.VAE

    def load_model(path: str, model_options: dict[str, object]) -> object:
        calls.append(
            {
                "boundary": "comfy.sd.load_diffusion_model",
                "path": path,
                "options": {name: str(value) for name, value in model_options.items()},
            }
        )
        return model_sentinel

    def load_file(
        path: str, *, return_metadata: bool
    ) -> tuple[dict[str, object], dict[str, object]]:
        calls.append(
            {
                "boundary": "comfy.utils.load_torch_file",
                "path": path,
                "return_metadata": return_metadata,
            }
        )
        return {}, {}

    class VaeSentinel:
        def __new__(cls, **kwargs: object) -> object:
            calls.append(
                {
                    "boundary": "comfy.sd.VAE",
                    "dtype": str(kwargs["dtype"]),
                    "state_keys": sorted(kwargs["sd"]),  # type: ignore[arg-type]
                }
            )
            return vae_sentinel

    comfy_sd.load_diffusion_model = load_model
    comfy_utils.load_torch_file = load_file
    comfy_sd.VAE = VaeSentinel
    try:
        model, vae = _load_comfy_components(
            comfy_sd,
            comfy_utils,
            args.diffusion,
            args.vae,
            torch,
        )
    finally:
        comfy_sd.load_diffusion_model = original_model
        comfy_utils.load_torch_file = original_file
        comfy_sd.VAE = original_vae
    if model is not model_sentinel or vae is not vae_sentinel:
        raise RuntimeError("ComfyUI preflight did not stop at the production load boundaries")
    if torch.cuda.is_initialized():
        raise RuntimeError("ComfyUI launch preflight initialized CUDA")
    return {
        "status": "passed",
        "cuda_initialized": False,
        "device_policy": "cpu substituted for the production CUDA policy",
        "requested_low_memory": args.target_free_gib is not None,
        "boundaries": calls,
    }


def _dinkster_preflight(args: argparse.Namespace) -> dict[str, object]:
    diffusion = _asset(args.diffusion, args.diffusion_digest, args.diffusion_size)
    vae = _asset(args.vae, args.vae_digest, args.vae_size)
    diffusion_identity, _ = _identity(diffusion, "diffusion")
    vae_identity, _ = _identity(vae, "vae")
    return _offline_launch(diffusion, vae, diffusion_identity, vae_identity)


def _timed_stage(
    torch: Any,
    timings: list[tuple[str, Any, Any, dict[str, int]]],
    name: str,
    call: Callable[[], Any],
) -> Any:
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    start.record()
    value = call()
    finish.record()
    timings.append((name, start, finish, _cuda_memory(torch)))
    return value


def _comfy_workflow(
    model: object,
    vae: object,
    source: object,
    nodes: Any,
) -> tuple[dict[str, object], list[tuple[str, Any, Any, dict[str, int]]]]:
    import torch
    from comfy_extras import nodes_custom_sampler, nodes_seedvr

    timings: list[tuple[str, Any, Any, dict[str, int]]] = []
    preprocessed = _timed_stage(
        torch, timings, "preprocess", lambda: nodes_seedvr.SeedVR2Preprocess.execute(source)[0]
    )
    latent = _timed_stage(
        torch,
        timings,
        "encode",
        lambda: nodes.VAEEncodeTiled().encode(
            vae,
            preprocessed,
            TILE_SIZE,
            TILE_OVERLAP,
            TEMPORAL_SIZE,
            TEMPORAL_OVERLAP,
        )[0],
    )
    positive, negative = _timed_stage(
        torch,
        timings,
        "conditioning",
        lambda: nodes_seedvr.SeedVR2Conditioning.execute(model, latent),
    )
    sigmas = _timed_stage(
        torch,
        timings,
        "schedule",
        lambda: nodes_custom_sampler.BasicScheduler.execute(model, SCHEDULER, STEPS, DENOISE)[0],
    )
    noise = _timed_stage(
        torch,
        timings,
        "noise",
        lambda: nodes_custom_sampler.RandomNoise.execute(SEED)[0].generate_noise(latent),
    )
    sampled = _timed_stage(
        torch,
        timings,
        "sample",
        lambda: nodes.KSampler().sample(
            model,
            SEED,
            STEPS,
            CFG,
            SAMPLER,
            SCHEDULER,
            positive,
            negative,
            latent,
            DENOISE,
        )[0],
    )
    decoded = _timed_stage(
        torch,
        timings,
        "decode",
        lambda: nodes.VAEDecodeTiled().decode(
            vae,
            sampled,
            TILE_SIZE,
            TILE_OVERLAP,
            TEMPORAL_SIZE,
            TEMPORAL_OVERLAP,
        )[0],
    )
    final = _timed_stage(
        torch,
        timings,
        "postprocess",
        lambda: nodes_seedvr.SeedVR2PostProcessing.execute(decoded, source, "none")[0],
    )
    return (
        {
            "input": source,
            "preprocessed": preprocessed,
            "encoded": latent["samples"],
            "conditioning": positive[0][1]["condition"],
            "noise": noise,
            "sigmas": sigmas,
            "sampled": sampled["samples"],
            "decoded": decoded,
            "final": final,
        },
        timings,
    )


def _dinkster_workflow(
    model: object, vae: object, source: object
) -> tuple[dict[str, object], list[tuple[str, Any, Any, dict[str, int]]]]:
    import dinkster_inference_torch
    import torch
    from dinkster_compat_comfy import native_arm as arm

    timings: list[tuple[str, Any, Any, dict[str, int]]] = []
    preprocessed = _timed_stage(
        torch,
        timings,
        "preprocess",
        lambda: arm.GenerationSeedVR2Preprocess.execute(resized_images=source)["images"],
    )
    latent = _timed_stage(
        torch,
        timings,
        "encode",
        lambda: arm.GenerationVAEEncodeTiled.execute(
            pixels=preprocessed,
            vae=vae,
            tile_size=TILE_SIZE,
            overlap=TILE_OVERLAP,
            temporal_size=TEMPORAL_SIZE,
            temporal_overlap=TEMPORAL_OVERLAP,
        )["latent"],
    )
    conditioning = _timed_stage(
        torch,
        timings,
        "conditioning",
        lambda: arm.GenerationSeedVR2Conditioning.execute(
            model=model,
            vae_conditioning=latent,
        ),
    )
    sigma_value = _timed_stage(
        torch,
        timings,
        "schedule",
        lambda: arm.GenerationBasicScheduler.execute(
            model=model,
            scheduler=SCHEDULER,
            steps=STEPS,
            denoise=DENOISE,
        )["sigmas"],
    )
    sigmas = torch.tensor(sigma_value.values, dtype=torch.float32)
    noise = _timed_stage(
        torch,
        timings,
        "noise",
        lambda: dinkster_inference_torch.prepare_noise(latent["samples"], SEED),
    )
    sampled = _timed_stage(
        torch,
        timings,
        "sample",
        lambda: arm.GenerationKSampler.execute(
            model=model,
            seed=SEED,
            steps=STEPS,
            cfg=CFG,
            sampler_name=SAMPLER,
            scheduler=SCHEDULER,
            positive=conditioning["positive"],
            negative=conditioning["negative"],
            latent_image=latent,
            denoise=DENOISE,
        )["latent"],
    )
    decoded = _timed_stage(
        torch,
        timings,
        "decode",
        lambda: arm.GenerationVAEDecodeTiled.execute(
            samples=sampled,
            vae=vae,
            tile_size=TILE_SIZE,
            overlap=TILE_OVERLAP,
            temporal_size=TEMPORAL_SIZE,
            temporal_overlap=TEMPORAL_OVERLAP,
        )["image"],
    )
    final = _timed_stage(
        torch,
        timings,
        "postprocess",
        lambda: arm.GenerationSeedVR2PostProcessing.execute(
            images=decoded,
            original_resized_images=source,
            color_correction_method="none",
        )["images"],
    )
    return (
        {
            "input": source,
            "preprocessed": preprocessed,
            "encoded": latent["samples"],
            "conditioning": conditioning["positive"][0][0],
            "noise": noise,
            "sigmas": sigmas,
            "sampled": sampled["samples"],
            "decoded": decoded,
            "final": final,
        },
        timings,
    )


def _canonical_stages(stages: dict[str, object]) -> dict[str, Any]:
    return {
        name: value.detach().float().cpu().contiguous()  # type: ignore[attr-defined]
        for name, value in stages.items()
    }


def _ballast(torch: Any, target_free_gib: int | None) -> Any | None:
    if target_free_gib is None:
        return None
    target = target_free_gib * 1024**3
    free, _total = torch.cuda.mem_get_info(0)
    size = max(0, free - target)
    ballast = torch.empty(size, dtype=torch.uint8, device="cuda")
    ballast.fill_(1)
    torch.cuda.synchronize(0)
    return ballast


def _comfy_fallback(model: object, vae: object) -> dict[str, object]:
    patchers = {"diffusion": model, "vae": vae.patcher}  # type: ignore[attr-defined]
    return {
        name: {
            "model_lowvram": bool(patcher.model.model_lowvram),
            "lowvram_patch_counter": patcher.model.lowvram_patch_counter,
            "total_bytes": patcher.model_size(),
            "loaded_bytes": patcher.loaded_size(),
            "offloaded_bytes": patcher.model_size() - patcher.loaded_size(),
        }
        for name, patcher in patchers.items()
    }


def _dinkster_fallback(model: object, vae: object) -> dict[str, object]:
    route = model.residency_route  # type: ignore[attr-defined]
    return {
        "diffusion": _mechanisms(model),
        "vae": _mechanisms(vae),
        "residency_route": None if route is None else asdict(route),
    }


def _run_child(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("GPU validation requires CUDA_VISIBLE_DEVICES=1")
    if args.backend == "dinkster":
        os.environ["DINKSTER_ACCELERATOR_HEADROOM_BYTES"] = str(ACCELERATOR_HEADROOM_BYTES)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GPU validation requires exactly physical GPU 1 to be visible")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    case = CaseSpec(
        args.case_name,
        args.variant,
        args.frames,
        args.height,
        args.width,
        args.channels,
        args.target_free_gib,
    )
    result: dict[str, object] = {
        "status": "running",
        "backend": args.backend,
        "case": asdict(case),
        "settings": {
            "seed": SEED,
            "steps": STEPS,
            "cfg": CFG,
            "sampler": SAMPLER,
            "scheduler": SCHEDULER,
            "denoise": DENOISE,
            "diffusion_dtype": "bfloat16",
            "vae_dtype": "float16",
            "attention": "pytorch_sdpa",
            "async_weight_transfer_streams": 1,
            "accelerator_headroom_bytes": ACCELERATOR_HEADROOM_BYTES,
            "tile_size": TILE_SIZE,
            "tile_overlap": TILE_OVERLAP,
            "temporal_size": TEMPORAL_SIZE,
            "temporal_overlap": TEMPORAL_OVERLAP,
        },
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "started_at": datetime.now(UTC).isoformat(),
    }
    _persist(args.output, result)
    source = _source(torch, case)
    model = None
    vae = None
    ballast = _ballast(torch, case.target_free_gib)
    torch.cuda.reset_peak_memory_stats(0)
    baseline_cuda = _cuda_memory(torch)
    baseline_host = _rss()
    result["baseline"] = {"cuda": baseline_cuda, "host": baseline_host}
    _persist(args.output, result)
    process_cuda_peaks: list[dict[str, int]] = [baseline_cuda]
    error: BaseException | None = None
    fallback: Callable[[], dict[str, object]] | None = None
    try:
        load_phases: dict[str, float] = {}
        load_started = time.perf_counter()
        if args.backend == "comfy":
            phase_started = time.perf_counter()
            comfy_sd, comfy_utils, model_management, nodes = _configure_comfy(
                args.comfy_root,
                low_memory=case.target_free_gib is not None,
            )
            load_phases["framework_setup_seconds"] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            model, vae = _load_comfy_components(
                comfy_sd,
                comfy_utils,
                args.diffusion,
                args.vae,
                torch,
            )
            load_phases["component_construction_seconds"] = time.perf_counter() - phase_started

            def workflow() -> tuple[dict[str, object], list[tuple[str, Any, Any, dict[str, int]]]]:
                return _comfy_workflow(model, vae, source, nodes)

            def fallback() -> dict[str, object]:
                return _comfy_fallback(model, vae)

        else:
            phase_started = time.perf_counter()
            diffusion_asset = _asset(
                args.diffusion,
                args.diffusion_digest,
                args.diffusion_size,
            )
            vae_asset = _asset(args.vae, args.vae_digest, args.vae_size)
            diffusion_identity, _ = _identity(diffusion_asset, "diffusion")
            vae_identity, _ = _identity(vae_asset, "vae")
            load_phases["artifact_planning_seconds"] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            model, vae = _load_components(
                diffusion_asset,
                vae_asset,
                diffusion_identity,
                vae_identity,
            )
            load_phases["component_construction_seconds"] = time.perf_counter() - phase_started

            def workflow() -> tuple[dict[str, object], list[tuple[str, Any, Any, dict[str, int]]]]:
                return _dinkster_workflow(model, vae, source)

            def fallback() -> dict[str, object]:
                return _dinkster_fallback(model, vae)

        load_seconds = time.perf_counter() - load_started
        torch.cuda.synchronize(0)
        load_memory = _cuda_memory(torch)
        process_cuda_peaks.append(load_memory)
        result["load"] = {
            "seconds": load_seconds,
            "phases": load_phases,
            "cuda": load_memory,
            "host": _rss(),
        }
        _persist(args.output, result)
        iterations: list[dict[str, object]] = []
        final_hashes: list[str] = []
        for index in range(args.warm_repeats + 1):
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            with torch.inference_mode():
                stages, stage_timings = workflow()
                torch.cuda.synchronize(0)
                seconds = time.perf_counter() - started
                canonical = _canonical_stages(stages)
            memory = _cuda_memory(torch)
            process_cuda_peaks.append(memory)
            records = {name: _tensor_record(value) for name, value in canonical.items()}
            final_hashes.append(str(records["final"]["sha256_float32"]))
            iteration = {
                "kind": "cold" if index == 0 else "warm",
                "seconds": seconds,
                "cuda": memory,
                "host": _rss(),
                "stage_seconds": {
                    name: start.elapsed_time(finish) / 1000.0
                    for name, start, finish, _memory in stage_timings
                },
                "stage_cuda": {name: memory for name, _start, _finish, memory in stage_timings},
                "stages": records,
            }
            iterations.append(iteration)
            result["iterations"] = iterations
            if index == 0:
                torch.save(canonical, args.tensors)
                result["tensor_file"] = str(args.tensors)
            _persist(args.output, result)
            del stages, stage_timings, canonical, records
            gc.collect()
        if len(set(final_hashes)) != 1:
            raise RuntimeError("repeated workflow outputs were not deterministic")
        warm_seconds = [float(item["seconds"]) for item in iterations[1:]]
        result["performance"] = {
            "load_seconds": load_seconds,
            "cold_workflow_seconds": float(iterations[0]["seconds"]),
            "cold_total_seconds": load_seconds + float(iterations[0]["seconds"]),
            "warm_seconds": warm_seconds,
            "warm_median_seconds": statistics.median(warm_seconds),
        }
        result["fallback"] = fallback()
        result["deterministic_final_sha256"] = final_hashes[0]
        result["status"] = "passed"
    except BaseException as caught:
        error = caught.with_traceback(None)
        result["status"] = "failed"
        result["error"] = {"type": type(caught).__name__, "message": str(caught)}
        result["failure"] = {"cuda": _cuda_memory(torch), "host": _rss()}
        if fallback is not None and model is not None and vae is not None:
            result["fallback"] = fallback()
    finally:
        try:
            if args.backend == "comfy" and model is not None:
                model_management.unload_all_models()
                model_management.soft_empty_cache(force=True)
            elif args.backend == "dinkster":
                if model is not None:
                    model.terminal_release()
                if vae is not None:
                    vae.terminal_release()
        finally:
            model = None
            vae = None
            source = None
            del ballast
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(0)
            untrimmed_host = _rss()
            heap_trimmed = _trim_heap()
            final_cuda = _cuda_memory(torch)
            process_cuda_peaks.append(final_cuda)
            result["process_peak_cuda"] = {
                "peak_allocated_bytes": max(
                    item["peak_allocated_bytes"] for item in process_cuda_peaks
                ),
                "peak_reserved_bytes": max(
                    item["peak_reserved_bytes"] for item in process_cuda_peaks
                ),
            }
            result["residual"] = {
                "cuda": final_cuda,
                "host": _rss(),
                "host_before_allocator_trim": untrimmed_host,
                "heap_trimmed": heap_trimmed,
            }
            result["finished_at"] = datetime.now(UTC).isoformat()
            _persist(args.output, result)
    if error is not None:
        raise error


def _compare_tensor(torch: Any, dinkster: Any, comfy: Any) -> dict[str, object]:
    if tuple(dinkster.shape) != tuple(comfy.shape):
        return {"exact": False, "error": "shape mismatch"}
    difference = (dinkster.float() - comfy.float()).abs()
    return {
        "exact": bool(torch.equal(dinkster, comfy)),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "dinkster_sha256": _tensor_record(dinkster)["sha256_float32"],
        "comfy_sha256": _tensor_record(comfy)["sha256_float32"],
    }


def evaluate_pair(dinkster: dict[str, Any], comfy: dict[str, Any]) -> dict[str, object]:
    comparisons = {
        "cold_total_seconds": (
            float(dinkster["performance"]["cold_total_seconds"]),
            float(comfy["performance"]["cold_total_seconds"]),
        ),
        "warm_median_seconds": (
            float(dinkster["performance"]["warm_median_seconds"]),
            float(comfy["performance"]["warm_median_seconds"]),
        ),
        "peak_allocated_bytes": (
            int(dinkster["process_peak_cuda"]["peak_allocated_bytes"]),
            int(comfy["process_peak_cuda"]["peak_allocated_bytes"]),
        ),
        "peak_reserved_bytes": (
            int(dinkster["process_peak_cuda"]["peak_reserved_bytes"]),
            int(comfy["process_peak_cuda"]["peak_reserved_bytes"]),
        ),
        "peak_host_delta_bytes": (
            int(dinkster["residual"]["host"]["peak_bytes"])
            - int(dinkster["baseline"]["host"]["current_bytes"]),
            int(comfy["residual"]["host"]["peak_bytes"])
            - int(comfy["baseline"]["host"]["current_bytes"]),
        ),
        "residual_allocated_bytes": (
            int(dinkster["residual"]["cuda"]["allocated_bytes"]),
            int(comfy["residual"]["cuda"]["allocated_bytes"]),
        ),
        "residual_reserved_bytes": (
            int(dinkster["residual"]["cuda"]["reserved_bytes"]),
            int(comfy["residual"]["cuda"]["reserved_bytes"]),
        ),
        "residual_host_delta_bytes": (
            int(dinkster["residual"]["host"]["current_bytes"])
            - int(dinkster["baseline"]["host"]["current_bytes"]),
            int(comfy["residual"]["host"]["current_bytes"])
            - int(comfy["baseline"]["host"]["current_bytes"]),
        ),
    }
    metrics = {
        name: {
            "dinkster": values[0],
            "comfy": values[1],
            "ratio": values[0] / values[1] if values[1] else (0.0 if not values[0] else None),
            "passed": values[0] <= values[1],
        }
        for name, values in comparisons.items()
    }
    return {
        "metrics": metrics,
        "performance_memory_passed": all(bool(value["passed"]) for value in metrics.values()),
    }


def _error_type(record: dict[str, Any]) -> object:
    error = record.get("error")
    return error.get("type") if isinstance(error, dict) else None


def evaluate_oom_pair(dinkster: dict[str, Any], comfy: dict[str, Any]) -> dict[str, object]:
    errors_match = all(
        record.get("status") == "failed" and _error_type(record) == "OutOfMemoryError"
        for record in (dinkster, comfy)
    )
    fallback_bytes = {
        "dinkster": sum(int(item["offloaded_bytes"]) for item in dinkster["fallback"]["diffusion"]),
        "comfy": int(comfy["fallback"]["diffusion"]["offloaded_bytes"]),
    }
    residuals = {
        "allocated_bytes": (
            int(dinkster["residual"]["cuda"]["allocated_bytes"]),
            int(comfy["residual"]["cuda"]["allocated_bytes"]),
        ),
        "reserved_bytes": (
            int(dinkster["residual"]["cuda"]["reserved_bytes"]),
            int(comfy["residual"]["cuda"]["reserved_bytes"]),
        ),
        "host_delta_bytes": (
            int(dinkster["residual"]["host"]["current_bytes"])
            - int(dinkster["baseline"]["host"]["current_bytes"]),
            int(comfy["residual"]["host"]["current_bytes"])
            - int(comfy["baseline"]["host"]["current_bytes"]),
        ),
    }
    residual_metrics = {
        name: {"dinkster": values[0], "comfy": values[1], "passed": values[0] <= values[1]}
        for name, values in residuals.items()
    }
    capacity_peaks = {
        "peak_allocated_bytes": {
            "dinkster": int(dinkster["process_peak_cuda"]["peak_allocated_bytes"]),
            "comfy": int(comfy["process_peak_cuda"]["peak_allocated_bytes"]),
        },
        "peak_reserved_bytes": {
            "dinkster": int(dinkster["process_peak_cuda"]["peak_reserved_bytes"]),
            "comfy": int(comfy["process_peak_cuda"]["peak_reserved_bytes"]),
        },
        "failure_driver_free_bytes": {
            "dinkster": int(dinkster["failure"]["cuda"]["driver_free_bytes"]),
            "comfy": int(comfy["failure"]["cuda"]["driver_free_bytes"]),
        },
    }
    fallback_passed = all(value > 0 for value in fallback_bytes.values())
    residual_passed = all(bool(value["passed"]) for value in residual_metrics.values())
    return {
        "expected_oom": True,
        "errors_match": errors_match,
        "fallback_offloaded_bytes": fallback_bytes,
        "fallback_passed": fallback_passed,
        "capacity_limited_peaks": capacity_peaks,
        "residual_metrics": residual_metrics,
        "residual_passed": residual_passed,
        "passed": errors_match and fallback_passed and residual_passed,
    }


def _variant(asset: object) -> str:
    from dinkster_inference import load_safetensors_header, plan_seedvr2_split_component

    path = asset.resolver.resolve(asset.digest)  # type: ignore[attr-defined]
    source = load_safetensors_header(
        path,
        asset_digest=asset.digest,  # type: ignore[attr-defined]
        asset_size=asset.size,  # type: ignore[attr-defined]
    )
    plan = plan_seedvr2_split_component(source, role="diffusion", path=path)
    return plan.config.variant  # type: ignore[union-attr]


def _warm_file_cache(path: Path) -> None:
    with path.open("rb") as handle:
        while handle.read(16 * 1024 * 1024):
            pass


def _child_command(
    args: argparse.Namespace,
    case: CaseSpec,
    backend: str,
    artifact: dict[str, object],
    vae: dict[str, object],
    output: Path,
    tensors: Path,
    mode: str,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        mode,
        "--backend",
        backend,
        "--case-name",
        case.name,
        "--variant",
        case.diffusion,
        "--frames",
        str(case.frames),
        "--height",
        str(case.height),
        "--width",
        str(case.width),
        "--channels",
        str(case.channels),
        "--diffusion",
        str(artifact["path"]),
        "--diffusion-digest",
        str(artifact["blake3"]),
        "--diffusion-size",
        str(artifact["bytes"]),
        "--vae",
        str(vae["path"]),
        "--vae-digest",
        str(vae["blake3"]),
        "--vae-size",
        str(vae["bytes"]),
        "--comfy-root",
        str(args.comfy_root),
        "--output",
        str(output),
        "--tensors",
        str(tensors),
        "--warm-repeats",
        str(args.warm_repeats),
    ]
    if case.target_free_gib is not None:
        command.extend(("--target-free-gib", str(case.target_free_gib)))
    return command


def _run_orchestrator(args: argparse.Namespace) -> None:
    import torch

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    artifacts = {
        "3b_swiglu": ArtifactSpec(
            args.diffusion_3b,
            args.diffusion_3b_sha256,
            "3b_swiglu",
            "diffusion_models/seedvr2_3b_int8_convrot.safetensors",
        ),
        "7b_mlp_int8": ArtifactSpec(
            args.diffusion_7b_int8,
            args.diffusion_7b_int8_sha256,
            "7b_mlp",
            "diffusion_models/seedvr2_7b_int8_convrot.safetensors",
        ),
        "7b_mlp_sharp": ArtifactSpec(
            args.diffusion_7b_sharp,
            args.diffusion_7b_sharp_sha256,
            "7b_mlp",
            "diffusion_models/seedvr2_7b_sharp_fp16.safetensors",
        ),
    }
    summary: dict[str, object] = {
        "status": "running",
        "validator": str(Path(__file__).relative_to(REPO_ROOT)),
        "dinkster_head": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, text=True
        ).strip(),
        "comfyui_commit": COMFYUI_COMMIT,
        "official_source": OFFICIAL_SOURCE,
        "official_revision": OFFICIAL_REVISION,
        "workflow_contract": {
            "current_template_source_commit": args.template_commit,
            "image_and_video": True,
            "spatial_tiling_exercised": True,
            "temporal_memory_state_exercised": True,
            "custom_sampling_runtime_via_ksampler_sugar": True,
            "architecture_reachability": {
                "3b_swiglu": "dedicated official artifact; image and 5-frame video",
                "7b_mlp": (
                    "official INT8 image, official sharp FP16 5-frame video, and matched "
                    "constrained-OOM execution"
                ),
                "7b_swiglu": (
                    "no released ByteDance or Comfy-Org artifact; retained ComfyUI compatibility "
                    "layout has synthetic exact-source goldens only"
                ),
                "shared_paths": (
                    "All variants use NaDiT.forward, SeedVR2DiffusionRuntime.sample_custom, "
                    "the causal SeedVR2 VAE, conditioning, scheduler, noise, KSampler sugar, "
                    "tiled codec nodes, and postprocessing."
                ),
            },
        },
        "cases": {},
        "started_at": datetime.now(UTC).isoformat(),
    }
    _persist(summary_path, summary)
    records: dict[str, dict[str, object]] = {}
    for key, spec in artifacts.items():
        asset, record = _artifact(spec.path, spec.sha256)
        variant = _variant(asset)
        if variant != spec.variant:
            raise RuntimeError(f"{spec.path.name} detected as {variant}, expected {spec.variant}")
        record["variant"] = variant
        record["immutable_url"] = (
            f"{OFFICIAL_SOURCE}/resolve/{OFFICIAL_REVISION}/{spec.repository_path}"
        )
        records[key] = record
        summary["diffusion_artifacts"] = records
        _persist(summary_path, summary)
    vae_asset, vae_record = _artifact(args.vae, args.vae_sha256)
    vae_record["immutable_url"] = (
        f"{OFFICIAL_SOURCE}/resolve/{OFFICIAL_REVISION}/vae/seedvr2_ema_vae_fp16.safetensors"
    )
    summary["vae_artifact"] = vae_record
    _persist(summary_path, summary)
    _verify_comfy_root(args.comfy_root)
    for path in (*[spec.path for spec in artifacts.values()], args.vae):
        _warm_file_cache(path)
    summary["page_cache_warmed"] = True
    _persist(summary_path, summary)
    cases = summary["cases"]
    assert isinstance(cases, dict)
    failed_cases: list[str] = []
    for case in case_specs():
        case_result: dict[str, object] = {"spec": asdict(case), "backends": {}}
        cases[case.name] = case_result
        _persist(summary_path, summary)
        backend_records: dict[str, dict[str, object]] = {}
        for backend in ("comfy", "dinkster"):
            raw = output_dir / f"{case.name}-{backend}.json"
            tensors = output_dir / f"{case.name}-{backend}.pt"
            preflight = output_dir / f"{case.name}-{backend}-preflight.json"
            preflight_command = _child_command(
                args,
                case,
                backend,
                records[case.diffusion],
                vae_record,
                preflight,
                tensors,
                "preflight",
            )
            preflight_environment = os.environ.copy()
            preflight_environment["CUDA_VISIBLE_DEVICES"] = ""
            if backend == "dinkster":
                preflight_environment["DINKSTER_ACCELERATOR_HEADROOM_BYTES"] = str(
                    ACCELERATOR_HEADROOM_BYTES
                )
            subprocess.run(
                preflight_command,
                cwd=REPO_ROOT,
                env=preflight_environment,
                check=True,
            )
            with _gpu1_claim(THREAD_ID, f"SeedVR2-E2E-{case.name}-{backend}") as lock_path:
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = "1"
                if backend == "dinkster":
                    environment["DINKSTER_ACCELERATOR_HEADROOM_BYTES"] = str(
                        ACCELERATOR_HEADROOM_BYTES
                    )
                run_command = _child_command(
                    args,
                    case,
                    backend,
                    records[case.diffusion],
                    vae_record,
                    raw,
                    tensors,
                    "run-child",
                )
                completed = subprocess.run(
                    run_command,
                    cwd=REPO_ROOT,
                    env=environment,
                    check=False,
                )
            record = json.loads(raw.read_text(encoding="utf-8"))
            record["preflight_file"] = str(preflight)
            record["claim"] = str(lock_path)
            backend_records[backend] = record
            case_result["backends"] = backend_records
            _persist(summary_path, summary)
            expected_oom = (
                case.expected_oom
                and completed.returncode != 0
                and record["status"] == "failed"
                and _error_type(record) == "OutOfMemoryError"
            )
            if not expected_oom and (completed.returncode != 0 or record["status"] != "passed"):
                failed_cases.append(case.name)
        if case.expected_oom:
            pair = evaluate_oom_pair(backend_records["dinkster"], backend_records["comfy"])
            case_result["comparison"] = pair
            _persist(summary_path, summary)
            if not pair["passed"]:
                failed_cases.append(case.name)
            continue
        if any(record["status"] != "passed" for record in backend_records.values()):
            case_result["comparison"] = {
                "passed": False,
                "error": "one or both backends failed",
            }
            _persist(summary_path, summary)
            continue
        dinkster_tensors = torch.load(
            output_dir / f"{case.name}-dinkster.pt", map_location="cpu", weights_only=True
        )
        comfy_tensors = torch.load(
            output_dir / f"{case.name}-comfy.pt", map_location="cpu", weights_only=True
        )
        if set(dinkster_tensors) != set(comfy_tensors):
            raise RuntimeError(f"{case.name} produced different intermediate sets")
        tensor_comparisons = {
            name: _compare_tensor(torch, dinkster_tensors[name], comfy_tensors[name])
            for name in sorted(dinkster_tensors)
        }
        pair = evaluate_pair(backend_records["dinkster"], backend_records["comfy"])
        pair["tensors"] = tensor_comparisons
        pair["correctness_passed"] = all(
            bool(value["exact"]) for value in tensor_comparisons.values()
        )
        pair["passed"] = bool(pair["correctness_passed"]) and bool(
            pair["performance_memory_passed"]
        )
        case_result["comparison"] = pair
        _persist(summary_path, summary)
        if not pair["passed"]:
            failed_cases.append(case.name)
    if failed_cases:
        summary["status"] = "failed"
        summary["failed_cases"] = sorted(set(failed_cases))
        summary["finished_at"] = datetime.now(UTC).isoformat()
        _persist(summary_path, summary)
        raise RuntimeError(
            "SeedVR2 cases did not meet correctness/performance/memory parity: "
            + ", ".join(sorted(set(failed_cases)))
        )
    summary["status"] = "passed"
    summary["finished_at"] = datetime.now(UTC).isoformat()
    _persist(summary_path, summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    orchestrate = subparsers.add_parser("orchestrate")
    orchestrate.add_argument("--diffusion-3b", type=Path, required=True)
    orchestrate.add_argument("--diffusion-3b-sha256", required=True)
    orchestrate.add_argument("--diffusion-7b-int8", type=Path, required=True)
    orchestrate.add_argument("--diffusion-7b-int8-sha256", required=True)
    orchestrate.add_argument("--diffusion-7b-sharp", type=Path, required=True)
    orchestrate.add_argument("--diffusion-7b-sharp-sha256", required=True)
    orchestrate.add_argument("--vae", type=Path, required=True)
    orchestrate.add_argument("--vae-sha256", required=True)
    orchestrate.add_argument("--comfy-root", type=Path, required=True)
    orchestrate.add_argument("--output-dir", type=Path, required=True)
    orchestrate.add_argument("--template-commit", required=True)
    orchestrate.add_argument("--warm-repeats", type=int, default=3)
    for mode in ("preflight", "run-child"):
        child = subparsers.add_parser(mode)
        child.add_argument("--backend", choices=("comfy", "dinkster"), required=True)
        child.add_argument("--case-name", required=True)
        child.add_argument("--variant", required=True)
        child.add_argument("--frames", type=int, required=True)
        child.add_argument("--height", type=int, required=True)
        child.add_argument("--width", type=int, required=True)
        child.add_argument("--channels", type=int, required=True)
        child.add_argument("--target-free-gib", type=int)
        child.add_argument("--diffusion", type=Path, required=True)
        child.add_argument("--diffusion-digest", required=True)
        child.add_argument("--diffusion-size", type=int, required=True)
        child.add_argument("--vae", type=Path, required=True)
        child.add_argument("--vae-digest", required=True)
        child.add_argument("--vae-size", type=int, required=True)
        child.add_argument("--comfy-root", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--tensors", type=Path, required=True)
        child.add_argument("--warm-repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "orchestrate":
        _run_orchestrator(args)
    elif args.mode == "preflight":
        result = _comfy_preflight(args) if args.backend == "comfy" else _dinkster_preflight(args)
        _persist(args.output, result)
    else:
        _run_child(args)


if __name__ == "__main__":
    main()
