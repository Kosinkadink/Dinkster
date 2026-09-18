"""Pinned Flux spatial-window comparison against ComfyUI-TiledDiffusion."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

COMFYUI_COMMIT = "947c2749dd04c51ef0e21b069544d8b0b4f9b411"
TILED_DIFFUSION_COMMIT = "a155b1bac39147381aeaa52b9be42e545626a44f"
PORTING_POLICY_COMMIT = "c73601518c38ad57e6ce20dbc43fc0e0974387d7"
DINKSTER_COMMIT = "ae0d5ae74813839ff31f871dd2a88f90a898fbad"
CHECKPOINT_BYTES = 17_246_524_772
CHECKPOINT_SHA256 = "8e91b68084b53a7fc44ed2a3756d821e355ac1a7b6fe29be760c1db532f3d88a"
CHECKPOINT_REVISION = "40a8a3d745c7d7adb077cb19879a975aa19c847b"
CHECKPOINT_URL = (
    "https://huggingface.co/Comfy-Org/flux1-dev/resolve/"
    f"{CHECKPOINT_REVISION}/flux1-dev-fp8.safetensors"
)

PROMPT = "A red fox sitting beside a mossy stone in a sunlit forest, detailed photograph"
SEED = 424242
STEPS = 4
WIDTH = 512
HEIGHT = 512
TILE_WIDTH = 384
TILE_HEIGHT = 512
TILE_OVERLAP = 256
MEASURED_RUNS = 5
PERFORMANCE_NOISE_FLOOR_FRACTION = 0.01
PERFORMANCE_NOISE_MAD_MULTIPLIER = 3.0
POSITION_ISOLATION_MIN_COSINE = 0.99
POSITION_ISOLATION_MIN_COSINE_GAIN = 0.05
COMFYUI_PRECISION = {
    "diffusion_actual": "bfloat16",
    "requested_text": "float32",
    "requested_vae": "float32",
    "text_actual": "float32",
    "vae_actual": "float32",
}
DINKSTER_PRECISION = {
    "clip_l_actual": "float32",
    "diffusion_actual": "bfloat16",
    "fp8_matmul": False,
    "requested_diffusion": "bfloat16",
    "requested_text": "float32",
    "requested_vae": "float32",
    "t5xxl_actual": "float32",
    "vae_actual": "float32",
}


class ComparisonError(RuntimeError):
    """The pinned benchmark environment or result is invalid."""


RunOutput = tuple[np.ndarray, np.ndarray, dict[str, Any]]


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _position_id_record(position_ids: Any) -> dict[str, Any]:
    values = position_ids.detach().float().cpu().numpy()
    if (
        values.ndim != 3
        or values.shape[-1] != 3
        or not np.isfinite(values).all()
        or not np.array_equal(values, np.rint(values))
        or np.count_nonzero(values[..., 0])
    ):
        raise ComparisonError("Flux image position IDs must be finite integer [batch, rows, 3]")
    return {
        "height_values": sorted({int(value) for value in values[..., 1].ravel()}),
        "sha256": array_sha256(values),
        "shape": list(values.shape),
        "width_values": sorted({int(value) for value in values[..., 2].ravel()}),
    }


def _summarize_position_ids(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        key = json.dumps(record, sort_keys=True)
        if key not in grouped:
            grouped[key] = {**record, "calls": 0}
        grouped[key]["calls"] += 1
    return list(grouped.values())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_source(root: Path, expected_commit: str, name: str) -> None:
    head = _git(root, "rev-parse", "HEAD")
    if head != expected_commit:
        raise ComparisonError(f"{name} must be checked out at {expected_commit}, got {head}")
    if _git(root, "status", "--porcelain"):
        raise ComparisonError(f"{name} checkout must be clean")


def _checkpoint_receipt(checkpoint: Path) -> dict[str, Any]:
    size = checkpoint.stat().st_size
    if size != CHECKPOINT_BYTES:
        raise ComparisonError(f"checkpoint size must be {CHECKPOINT_BYTES} bytes, got {size}")
    digest = _file_sha256(checkpoint)
    if digest != CHECKPOINT_SHA256:
        raise ComparisonError(f"checkpoint sha256 must be {CHECKPOINT_SHA256}, got {digest}")
    return {
        "bytes": size,
        "path": str(checkpoint.resolve()),
        "revision": CHECKPOINT_REVISION,
        "sha256": digest,
        "url": CHECKPOINT_URL,
    }


def _driver_version() -> str:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    versions = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        raise ComparisonError(f"expected one NVIDIA driver version, got {sorted(versions)}")
    return versions.pop()


def _device_receipt(torch: Any, device: Any) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise ComparisonError(
            f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}"
        )
    properties = torch.cuda.get_device_properties(device)
    return {
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "driver": _driver_version(),
        "gpu": properties.name,
        "gpu_total_memory": properties.total_memory,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }


def _run_metrics(
    torch: Any,
    device: Any,
    sample: Callable[[], Any],
    decode: Callable[[Any], Any],
    latent_array: Callable[[Any], np.ndarray],
    image_array: Callable[[Any], np.ndarray],
) -> RunOutput:
    torch.cuda.synchronize(device)
    sample_start = time.perf_counter_ns()
    with torch.inference_mode():
        sampled = sample()
    torch.cuda.synchronize(device)
    sample_end = time.perf_counter_ns()
    with torch.inference_mode():
        image = decode(sampled)
    torch.cuda.synchronize(device)
    end = time.perf_counter_ns()
    return (
        latent_array(sampled),
        image_array(image),
        {
            "decode_seconds": (end - sample_end) / 1e9,
            "end_to_end_seconds": (end - sample_start) / 1e9,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "sample_seconds": (sample_end - sample_start) / 1e9,
        },
    )


def _summarize_runs(
    window_run: Callable[[], RunOutput],
    full_run: Callable[[], RunOutput],
    native_run: Callable[[], RunOutput],
    *,
    process_start: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cold_latent, cold_image, cold_metrics = window_run()
    cold_metrics["load_text_sample_decode_seconds"] = (time.perf_counter_ns() - process_start) / 1e9
    warmup_latent, warmup_image, _ = window_run()

    measured: list[dict[str, Any]] = []
    measured_latent: np.ndarray | None = None
    measured_image: np.ndarray | None = None
    for index in range(MEASURED_RUNS):
        measured_latent, measured_image, metrics = window_run()
        metrics.update(
            {
                "image_sha256": array_sha256(measured_image),
                "index": index,
                "latent_sha256": array_sha256(measured_latent),
            }
        )
        measured.append(metrics)
    assert measured_latent is not None and measured_image is not None

    full_latent, full_image, full_metrics = full_run()
    native_latent, native_image, native_metrics = native_run()
    sample_values = [float(item["sample_seconds"]) for item in measured]
    end_values = [float(item["end_to_end_seconds"]) for item in measured]
    summary = {
        "cold": {
            **cold_metrics,
            "image_sha256": array_sha256(cold_image),
            "latent_sha256": array_sha256(cold_latent),
        },
        "end_to_end_range_seconds": [min(end_values), max(end_values)],
        "full_window_control": {
            **full_metrics,
            "image_sha256": array_sha256(full_image),
            "latent_sha256": array_sha256(full_latent),
        },
        "measured": measured,
        "median_end_to_end_seconds": statistics.median(end_values),
        "median_sample_seconds": statistics.median(sample_values),
        "native_control": {
            **native_metrics,
            "equals_full_window_image": bool(np.array_equal(native_image, full_image)),
            "equals_full_window_latent": bool(np.array_equal(native_latent, full_latent)),
            "image_sha256": array_sha256(native_image),
            "latent_sha256": array_sha256(native_latent),
        },
        "sample_range_seconds": [min(sample_values), max(sample_values)],
        "warmup": {
            "image_sha256": array_sha256(warmup_image),
            "latent_sha256": array_sha256(warmup_latent),
        },
        "window_output": {
            "image_sha256": array_sha256(measured_image),
            "image_shape": list(measured_image.shape),
            "latent_sha256": array_sha256(measured_latent),
            "latent_shape": list(measured_latent.shape),
        },
    }
    arrays = {
        "full-image": full_image,
        "full-latent": full_latent,
        "native-image": native_image,
        "native-latent": native_latent,
        "window-image": measured_image,
        "window-latent": measured_latent,
    }
    return summary, arrays


def _save_engine_result(
    output_dir: Path,
    prefix: str,
    result: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(output_dir / f"{prefix}-{name}.npy", array, allow_pickle=False)
    (output_dir / f"{prefix}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _load_custom_node(root: Path) -> Any:
    name = "comfyui_tiled_diffusion_dinkster_comparison"
    spec = importlib.util.spec_from_file_location(
        name,
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise ComparisonError("could not load the pinned TiledDiffusion package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_comfyui(args: argparse.Namespace) -> int:
    comfyui_root = args.comfyui_root.resolve()
    tiled_root = args.tiled_root.resolve()
    checkpoint = args.checkpoint.resolve()
    _require_source(comfyui_root, COMFYUI_COMMIT, "ComfyUI")
    _require_source(tiled_root, TILED_DIFFUSION_COMMIT, "ComfyUI-TiledDiffusion")
    checkpoint_receipt = _checkpoint_receipt(checkpoint)

    sys.path.insert(0, str(comfyui_root))
    sys.argv = [
        sys.argv[0],
        "--highvram",
        "--use-pytorch-cross-attention",
        "--fp32-text-enc",
        "--fp32-vae",
    ]

    import comfy.options

    comfy.options.enable_args_parsing()

    import comfy.cli_args
    import comfy.model_management as model_management
    import comfy.sample
    import comfy.samplers
    import folder_paths
    import node_helpers
    import nodes
    import torch

    if not (
        comfy.cli_args.args.fp32_text_enc
        and comfy.cli_args.args.fp32_vae
        and comfy.cli_args.args.use_pytorch_cross_attention
    ):
        raise ComparisonError("ComfyUI precision and attention arguments were not applied")
    device = torch.device("cuda:0")
    device_receipt = _device_receipt(torch, device)
    folder_paths.add_model_folder_path("checkpoints", str(checkpoint.parent), is_default=True)
    resolved = Path(folder_paths.get_full_path_or_raise("checkpoints", checkpoint.name))
    if not resolved.samefile(checkpoint):
        raise ComparisonError(f"ComfyUI resolved a different checkpoint: {resolved}")

    custom_node = _load_custom_node(tiled_root)
    tiled_type = custom_node.NODE_CLASS_MAPPINGS["TiledDiffusion"]
    process_start = time.perf_counter_ns()
    torch.cuda.reset_peak_memory_stats(device)
    model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(checkpoint.name)
    if vae.vae_dtype is not torch.float32:
        raise ComparisonError(f"ComfyUI VAE must be float32, got {vae.vae_dtype}")
    precision = {
        "diffusion_actual": str(model.model.get_dtype_inference()).removeprefix("torch."),
        "requested_text": "float32",
        "requested_vae": "float32",
        "text_actual": str(clip.patcher.object_patches.get("manual_cast_dtype")).removeprefix(
            "torch."
        ),
        "vae_actual": str(vae.vae_dtype).removeprefix("torch."),
    }
    if precision != COMFYUI_PRECISION:
        raise ComparisonError(f"ComfyUI precision must be {COMFYUI_PRECISION}, got {precision}")
    positive = nodes.CLIPTextEncode().encode(clip, PROMPT)[0]
    positive = node_helpers.conditioning_set_values(positive, {"guidance": 3.5})
    negative = nodes.CLIPTextEncode().encode(clip, "")[0]
    latent = {
        "samples": torch.zeros((1, 16, HEIGHT // 8, WIDTH // 8), dtype=torch.float32),
        "downscale_ratio_spacial": 8,
    }
    initial_noise = comfy.sample.prepare_noise(latent["samples"], SEED)
    ksampler = comfy.samplers.KSampler(
        model,
        STEPS,
        model.load_device,
        sampler="euler",
        scheduler="normal",
        denoise=1.0,
    )
    sigma_hex = [float(value).hex() for value in ksampler.sigmas.cpu().tolist()]

    tiled_node = tiled_type()
    tiled_model = tiled_node.apply(
        model,
        "MultiDiffusion",
        TILE_WIDTH,
        TILE_HEIGHT,
        TILE_OVERLAP,
        1,
    )[0]
    full_node = tiled_type()
    full_model = full_node.apply(model, "MultiDiffusion", WIDTH, HEIGHT, 0, 1)[0]

    def run(selected_model: Any) -> RunOutput:
        return _run_metrics(
            torch,
            device,
            lambda: nodes.common_ksampler(
                selected_model,
                SEED,
                STEPS,
                1.0,
                "euler",
                "normal",
                positive,
                negative,
                latent,
                denoise=1.0,
            )[0],
            lambda sampled: nodes.VAEDecode().decode(vae, sampled)[0],
            lambda sampled: sampled["samples"].detach().float().cpu().numpy(),
            lambda image: image.detach().float().cpu().numpy(),
        )

    summary, arrays = _summarize_runs(
        lambda: run(tiled_model),
        lambda: run(full_model),
        lambda: run(model),
        process_start=process_start,
    )
    position_records: list[dict[str, Any]] = []
    diffusion_model = model.model.diffusion_model
    original_process_img = diffusion_model.process_img

    def capture_process_img(*call_args: Any, **call_kwargs: Any) -> Any:
        result = original_process_img(*call_args, **call_kwargs)
        position_records.append(_position_id_record(result[1]))
        return result

    diffusion_model.process_img = capture_process_img
    try:
        diagnostic_latent, diagnostic_image, _ = run(tiled_model)
    finally:
        diffusion_model.process_img = original_process_img
    diagnostic_matches = bool(
        np.array_equal(diagnostic_latent, arrays["window-latent"])
        and np.array_equal(diagnostic_image, arrays["window-image"])
    )
    if not diagnostic_matches:
        raise ComparisonError("ComfyUI position diagnostic changed the deterministic window output")
    summary["position_id_diagnostic"] = {
        "actual_image_position_ids": _summarize_position_ids(position_records),
        "matches_measured_output": diagnostic_matches,
    }
    tile_boxes = [
        {"height": box.h, "width": box.w, "x": box.x, "y": box.y}
        for batch in tiled_node.impl.batched_bboxes
        for box in batch
    ]
    result = {
        **summary,
        "declared_windows": tile_boxes,
        "engine": "ComfyUI+ComfyUI-TiledDiffusion",
        "receipts": {
            **device_receipt,
            "attention_policy": "pytorch-sdpa (--use-pytorch-cross-attention)",
            "checkpoint": checkpoint_receipt,
            "comfyui_commit": COMFYUI_COMMIT,
            "initial_noise_sha256": array_sha256(initial_noise.cpu().numpy()),
            "normal_schedule_float_hex": sigma_hex,
            "precision": precision,
            "tiled_diffusion_commit": TILED_DIFFUSION_COMMIT,
        },
        "workload": _workload_receipt(),
    }
    _save_engine_result(args.output_dir, "comfyui", result, arrays)

    gc.collect()
    model_management.unload_all_models()
    model_management.soft_empty_cache(force=True)
    return 0


def _add_dinkster_sources(dinkster_root: Path) -> None:
    for source in sorted((dinkster_root / "packages").glob("*/src")):
        sys.path.insert(0, str(source))


def run_dinkster(args: argparse.Namespace) -> int:
    dinkster_root = args.dinkster_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    _require_source(dinkster_root, args.dinkster_commit, "Dinkster")
    checkpoint_receipt = _checkpoint_receipt(checkpoint_path)
    _add_dinkster_sources(dinkster_root)

    import dinkster_inference_torch.denoise as denoise_module
    import torch
    from dinkster_inference import (
        AccumulationDType,
        IntegerAffineIndexMap,
        KindAxisMap,
        LayerWindow,
        MediaAxis,
        MergeDeclaration,
        WindowIndexList,
        WindowKind,
        WindowPlanLayer,
        WindowWeightKind,
        WindowWeightProfile,
        compile_window_plan,
        load_safetensors_header,
    )
    from dinkster_inference_torch import load_runtime
    from dinkster_inference_torch.denoise import prepare_noise
    from dinkster_inference_torch.wiring import _flux_sampling_plan

    device = torch.device("cuda:0")
    device_receipt = _device_receipt(torch, device)
    width_axis = MediaAxis("width", WIDTH // 16)
    kinds = (
        WindowKind(
            "latent_image",
            (KindAxisMap("width", WIDTH // 16, IntegerAffineIndexMap(1)),),
        ),
        WindowKind("text", invariant_axes=("width",)),
    )

    def make_plan(windows: tuple[tuple[int, ...], ...]) -> Any:
        return compile_window_plan(
            axes=(width_axis,),
            kinds=kinds,
            layers=(
                WindowPlanLayer(
                    ("width",),
                    tuple(LayerWindow((WindowIndexList(indices),)) for indices in windows),
                    (WindowWeightProfile(WindowWeightKind.FLAT),),
                    MergeDeclaration(AccumulationDType.FLOAT32),
                ),
            ),
        )

    window_plan = make_plan((tuple(range(24)), tuple(range(8, 32))))
    full_plan = make_plan((tuple(range(32)),))
    process_start = time.perf_counter_ns()
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint = load_safetensors_header(checkpoint_path)
    runtime = load_runtime(
        checkpoint=checkpoint,
        diffusion_dtype=torch.bfloat16,
        text_dtype=torch.float32,
        vae_dtype=torch.float32,
        fp8_matmul=False,
        attention_policy="auto",
    )
    assert runtime.assembled.clip_l is not None
    assert runtime.assembled.t5xxl is not None
    runtime.assembled.clip_l.to(device)
    runtime.assembled.t5xxl.to(device)
    with torch.inference_mode():
        cond = runtime.encode_text(PROMPT)
    runtime.assembled.clip_l.to("cpu")
    runtime.assembled.t5xxl.to("cpu")
    torch.cuda.empty_cache()
    runtime.assembled.diffusion.to(device)
    runtime.assembled.vae.to(device)
    precision = {
        "clip_l_actual": str(runtime.assembled.compute_dtype("clip_l")).removeprefix("torch."),
        "diffusion_actual": str(runtime.assembled.compute_dtype("diffusion")).removeprefix(
            "torch."
        ),
        "fp8_matmul": False,
        "requested_diffusion": "bfloat16",
        "requested_text": "float32",
        "requested_vae": "float32",
        "t5xxl_actual": str(runtime.assembled.compute_dtype("t5xxl")).removeprefix("torch."),
        "vae_actual": str(runtime.assembled.compute_dtype("vae")).removeprefix("torch."),
    }
    if precision != DINKSTER_PRECISION:
        raise ComparisonError(f"Dinkster precision must be {DINKSTER_PRECISION}, got {precision}")
    latent = torch.zeros((1, 16, HEIGHT // 8, WIDTH // 8), dtype=torch.float32, device=device)
    initial_noise = prepare_noise(latent.cpu(), SEED)
    sampler = runtime._samplers.get("dinkster.euler")
    scheduler = runtime._schedulers.get("dinkster.normal")
    assert sampler is not None and scheduler is not None
    _, sigmas, _ = _flux_sampling_plan(
        runtime.family,
        sampler,
        scheduler,
        latent,
        steps=STEPS,
        denoise=1.0,
        seed=SEED,
        device=device,
    )

    def run(plan: Any | None) -> RunOutput:
        return _run_metrics(
            torch,
            device,
            lambda: runtime.sample(
                latent,
                cond=cond,
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.normal",
                steps=STEPS,
                denoise=1.0,
                seed=SEED,
                guidance=3.5,
                window_plan=plan,
                compute_dtype=torch.bfloat16,
                device=device,
            ),
            lambda sampled: runtime.decode_latent(sampled).permute(0, 2, 3, 1),
            lambda sampled: sampled.detach().float().cpu().numpy(),
            lambda image: image.detach().float().cpu().numpy(),
        )

    summary, arrays = _summarize_runs(
        lambda: run(window_plan),
        lambda: run(full_plan),
        lambda: run(None),
        process_start=process_start,
    )
    original_position_ids = denoise_module.flux_window_position_ids
    global_position_records: list[dict[str, Any]] = []
    local_position_records: list[dict[str, Any]] = []

    def capture_global_position_ids(**position_args: Any) -> Any:
        position_ids = original_position_ids(**position_args)
        global_position_records.append(_position_id_record(position_ids))
        return position_ids

    def local_position_ids(
        *,
        height_indices: tuple[int, ...],
        width_indices: tuple[int, ...],
        batch: int,
        device: Any,
    ) -> Any:
        position_ids = original_position_ids(
            height_indices=tuple(range(len(height_indices))),
            width_indices=tuple(range(len(width_indices))),
            batch=batch,
            device=device,
        )
        local_position_records.append(_position_id_record(position_ids))
        return position_ids

    denoise_module.flux_window_position_ids = capture_global_position_ids
    try:
        diagnostic_global_latent, diagnostic_global_image, _ = run(window_plan)
        denoise_module.flux_window_position_ids = local_position_ids
        diagnostic_local_latent, diagnostic_local_image, _ = run(window_plan)
    finally:
        denoise_module.flux_window_position_ids = original_position_ids
    global_matches = bool(
        np.array_equal(diagnostic_global_latent, arrays["window-latent"])
        and np.array_equal(diagnostic_global_image, arrays["window-image"])
    )
    if not global_matches:
        raise ComparisonError(
            "Dinkster position diagnostic changed the deterministic window output"
        )
    arrays["local-window-latent"] = diagnostic_local_latent
    arrays["local-window-image"] = diagnostic_local_image
    summary["position_id_diagnostic"] = {
        "global_actual_image_position_ids": _summarize_position_ids(global_position_records),
        "global_matches_measured_output": global_matches,
        "local_actual_image_position_ids": _summarize_position_ids(local_position_records),
        "local_image_sha256": array_sha256(diagnostic_local_image),
        "local_latent_sha256": array_sha256(diagnostic_local_latent),
        "controlled_change": "replace global packed image position IDs with window-local IDs",
    }
    status = runtime.attention_status["flux"]
    result = {
        **summary,
        "declared_windows": [
            {"packed_width_indices": list(range(24))},
            {"packed_width_indices": list(range(8, 32))},
        ],
        "engine": "Dinkster",
        "receipts": {
            **device_receipt,
            "attention_backend": status.primary,
            "attention_policy": status.requested_policy,
            "checkpoint": checkpoint_receipt,
            "dinkster_commit": args.dinkster_commit,
            "initial_noise_sha256": array_sha256(initial_noise.numpy()),
            "normal_schedule_float_hex": [float(value).hex() for value in sigmas],
            "precision": precision,
            "runtime_identity": runtime.runtime_identity,
            "window_plan_digest": window_plan.digest,
        },
        "workload": _workload_receipt(),
    }
    _save_engine_result(args.output_dir, "dinkster", result, arrays)

    runtime.assembled.diffusion.to("cpu")
    runtime.assembled.vae.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def _workload_receipt() -> dict[str, Any]:
    return {
        "cfg": 1.0,
        "flux_guidance": 3.5,
        "height": HEIGHT,
        "merge": "flat per-occurrence mean in float32",
        "prompt": PROMPT,
        "sampler": "Euler",
        "scheduler": "normal",
        "seed": SEED,
        "steps": STEPS,
        "tile_batch_size": 1,
        "tile_height": TILE_HEIGHT,
        "tile_overlap": TILE_OVERLAP,
        "tile_width": TILE_WIDTH,
        "width": WIDTH,
    }


def compare_arrays(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"left_shape": list(left.shape), "right_shape": list(right.shape)}
    left64 = left.astype(np.float64)
    right64 = right.astype(np.float64)
    delta = left64 - right64
    left_flat = left64.ravel()
    right_flat = right64.ravel()
    delta_flat = delta.ravel()
    left_norm = float(np.linalg.norm(left_flat))
    right_norm = float(np.linalg.norm(right_flat))
    delta_norm = float(np.linalg.norm(delta_flat))
    denominator = left_norm * right_norm
    mse = float(np.mean(delta * delta))
    return {
        "all_finite": bool(np.isfinite(left).all() and np.isfinite(right).all()),
        "cosine": None if denominator == 0 else float(np.dot(left_flat, right_flat) / denominator),
        "exact": bool(np.array_equal(left, right)),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "relative_l2_to_left": None if left_norm == 0 else delta_norm / left_norm,
        "rmse": math.sqrt(mse),
        "shape": list(left.shape),
    }


def _performance_metric(
    comfyui: dict[str, Any],
    dinkster: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    field = f"{metric}_seconds"
    reference_values = [float(item[field]) for item in comfyui["measured"]]
    candidate_values = [float(item[field]) for item in dinkster["measured"]]
    reference_median = statistics.median(reference_values)
    candidate_median = statistics.median(candidate_values)
    reference_mad = statistics.median([abs(value - reference_median) for value in reference_values])
    allowed_regression = max(
        reference_median * PERFORMANCE_NOISE_FLOOR_FRACTION,
        reference_mad * PERFORMANCE_NOISE_MAD_MULTIPLIER,
    )
    maximum_candidate_median = reference_median + allowed_regression
    return {
        "allowed_regression_seconds": allowed_regression,
        "candidate_median_seconds": candidate_median,
        "maximum_candidate_median_seconds": maximum_candidate_median,
        "pass": candidate_median <= maximum_candidate_median,
        "reference_mad_seconds": reference_mad,
        "reference_median_seconds": reference_median,
    }


def _position_id_evidence(comfyui: dict[str, Any], dinkster: dict[str, Any]) -> bool:
    expected_height = list(range(32))
    local_width = list(range(24))
    global_width = list(range(8, 32))
    comfy_records = comfyui["position_id_diagnostic"]["actual_image_position_ids"]
    global_records = dinkster["position_id_diagnostic"]["global_actual_image_position_ids"]
    local_records = dinkster["position_id_diagnostic"]["local_actual_image_position_ids"]

    def coordinates(records: list[dict[str, Any]]) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
        return {
            (tuple(record["height_values"]), tuple(record["width_values"])) for record in records
        }

    return bool(
        comfyui["position_id_diagnostic"]["matches_measured_output"]
        and dinkster["position_id_diagnostic"]["global_matches_measured_output"]
        and coordinates(comfy_records) == {(tuple(expected_height), tuple(local_width))}
        and coordinates(global_records)
        == {
            (tuple(expected_height), tuple(local_width)),
            (tuple(expected_height), tuple(global_width)),
        }
        and coordinates(local_records) == {(tuple(expected_height), tuple(local_width))}
    )


def build_verdict(
    comfyui: dict[str, Any],
    dinkster: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    dinkster_commit: str = DINKSTER_COMMIT,
) -> dict[str, Any]:
    comparisons = {
        "comfyui_window_vs_full_image": compare_arrays(
            arrays["comfyui-window-image"], arrays["comfyui-full-image"]
        ),
        "comfyui_window_vs_full_latent": compare_arrays(
            arrays["comfyui-window-latent"], arrays["comfyui-full-latent"]
        ),
        "cross_engine_full_image": compare_arrays(
            arrays["dinkster-full-image"], arrays["comfyui-full-image"]
        ),
        "cross_engine_full_latent": compare_arrays(
            arrays["dinkster-full-latent"], arrays["comfyui-full-latent"]
        ),
        "cross_engine_window_image": compare_arrays(
            arrays["dinkster-window-image"], arrays["comfyui-window-image"]
        ),
        "cross_engine_window_latent": compare_arrays(
            arrays["dinkster-window-latent"], arrays["comfyui-window-latent"]
        ),
        "cross_engine_local_id_window_image": compare_arrays(
            arrays["dinkster-local-window-image"], arrays["comfyui-window-image"]
        ),
        "cross_engine_local_id_window_latent": compare_arrays(
            arrays["dinkster-local-window-latent"], arrays["comfyui-window-latent"]
        ),
        "dinkster_window_vs_full_image": compare_arrays(
            arrays["dinkster-window-image"], arrays["dinkster-full-image"]
        ),
        "dinkster_window_vs_full_latent": compare_arrays(
            arrays["dinkster-window-latent"], arrays["dinkster-full-latent"]
        ),
    }
    sample_performance = _performance_metric(comfyui, dinkster, "sample")
    end_performance = _performance_metric(comfyui, dinkster, "end_to_end")
    measured_count = all(len(result["measured"]) == MEASURED_RUNS for result in (comfyui, dinkster))
    deterministic = measured_count and all(
        len({item[key] for item in result["measured"]}) == 1
        for result in (comfyui, dinkster)
        for key in ("image_sha256", "latent_sha256")
    )
    one_window_identity = all(
        result["native_control"][key]
        for result in (comfyui, dinkster)
        for key in ("equals_full_window_image", "equals_full_window_latent")
    )
    exact_noise = (
        comfyui["receipts"]["initial_noise_sha256"] == dinkster["receipts"]["initial_noise_sha256"]
    )
    exact_schedule = (
        comfyui["receipts"]["normal_schedule_float_hex"]
        == dinkster["receipts"]["normal_schedule_float_hex"]
        and len(comfyui["receipts"]["normal_schedule_float_hex"]) == STEPS + 1
        and comfyui["receipts"]["normal_schedule_float_hex"][-1] == "0x0.0p+0"
    )
    matching_workload = comfyui["workload"] == dinkster["workload"] == _workload_receipt()
    matching_geometry = (
        comfyui["declared_windows"]
        == [
            {"height": 64, "width": 48, "x": 0, "y": 0},
            {"height": 64, "width": 48, "x": 16, "y": 0},
        ]
        and dinkster["declared_windows"]
        == [
            {"packed_width_indices": list(range(24))},
            {"packed_width_indices": list(range(8, 32))},
        ]
        and all(
            result["window_output"]["latent_shape"] == [1, 16, 64, 64]
            and result["window_output"]["image_shape"] == [1, 512, 512, 3]
            for result in (comfyui, dinkster)
        )
    )
    pinned_sources = (
        comfyui["receipts"]["comfyui_commit"] == COMFYUI_COMMIT
        and comfyui["receipts"]["tiled_diffusion_commit"] == TILED_DIFFUSION_COMMIT
        and dinkster["receipts"]["dinkster_commit"] == dinkster_commit
        and comfyui["receipts"]["checkpoint"]["sha256"] == CHECKPOINT_SHA256
        and dinkster["receipts"]["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    )
    environment_keys = ("cuda", "driver", "gpu", "gpu_total_memory", "torch")
    matching_environment = all(
        comfyui["receipts"][key] == dinkster["receipts"][key] for key in environment_keys
    )
    matching_precision = (
        comfyui["receipts"]["precision"] == COMFYUI_PRECISION
        and dinkster["receipts"]["precision"] == DINKSTER_PRECISION
    )
    all_finite = all(value.get("all_finite", False) for value in comparisons.values())
    position_id_evidence = _position_id_evidence(comfyui, dinkster)
    global_position_comparison = comparisons["cross_engine_window_latent"]
    local_position_comparison = comparisons["cross_engine_local_id_window_latent"]
    global_cosine = global_position_comparison.get("cosine")
    local_cosine = local_position_comparison.get("cosine")
    position_isolation = bool(
        position_id_evidence
        and isinstance(global_cosine, float)
        and isinstance(local_cosine, float)
        and local_cosine >= POSITION_ISOLATION_MIN_COSINE
        and local_cosine - global_cosine >= POSITION_ISOLATION_MIN_COSINE_GAIN
    )
    overall_pass = all(
        (
            deterministic,
            one_window_identity,
            exact_noise,
            exact_schedule,
            matching_geometry,
            matching_workload,
            matching_environment,
            matching_precision,
            pinned_sources,
            all_finite,
            position_isolation,
            sample_performance["pass"],
            end_performance["pass"],
        )
    )
    return {
        "behavioral": {
            "all_outputs_finite": all_finite,
            "deterministic_measured_outputs": deterministic,
            "exact_initial_noise": exact_noise,
            "exact_normal_schedule": exact_schedule,
            "matching_environment": matching_environment,
            "matching_geometry": matching_geometry,
            "matching_precision": matching_precision,
            "matching_workload": matching_workload,
            "measured_run_count": measured_count,
            "one_window_equals_native_within_each_engine": one_window_identity,
            "pinned_sources": pinned_sources,
            "position_id_evidence": position_id_evidence,
            "position_id_isolation": position_isolation,
            "position_id_semantics": {
                "comfyui_tiled_diffusion": (
                    "each cropped Flux call derives local packed positions starting at zero"
                ),
                "dinkster": "each cropped Flux call uses its declared global packed positions",
            },
        },
        "comparisons": comparisons,
        "overall_pass": overall_pass,
        "performance": {
            "end_to_end": end_performance,
            "pass": sample_performance["pass"] and end_performance["pass"],
            "sample": sample_performance,
        },
        "policy": {
            "performance_noise_floor_fraction": PERFORMANCE_NOISE_FLOOR_FRACTION,
            "performance_noise_mad_multiplier": PERFORMANCE_NOISE_MAD_MULTIPLIER,
            "porting_commit": PORTING_POLICY_COMMIT,
            "position_isolation_min_cosine": POSITION_ISOLATION_MIN_COSINE,
            "position_isolation_min_cosine_gain": POSITION_ISOLATION_MIN_COSINE_GAIN,
            "tiled_diffusion_license": (
                "CC BY-NC-SA comparison reference only; no code was ported"
            ),
            "warmup_runs_discarded": 1,
            "measured_runs": MEASURED_RUNS,
        },
        "schema": 1,
    }


def run_compare(args: argparse.Namespace) -> int:
    input_dir = args.input_dir
    comfyui = json.loads((input_dir / "comfyui.json").read_text())
    dinkster = json.loads((input_dir / "dinkster.json").read_text())
    arrays = {
        f"{engine}-{kind}": np.load(input_dir / f"{engine}-{kind}.npy", allow_pickle=False)
        for engine in ("comfyui", "dinkster")
        for kind in (
            "full-image",
            "full-latent",
            "native-image",
            "native-latent",
            "window-image",
            "window-latent",
        )
    }
    for kind in ("local-window-image", "local-window-latent"):
        arrays[f"dinkster-{kind}"] = np.load(
            input_dir / f"dinkster-{kind}.npy",
            allow_pickle=False,
        )
    for engine, result in (("comfyui", comfyui), ("dinkster", dinkster)):
        for kind, section in (
            ("full", "full_window_control"),
            ("native", "native_control"),
            ("window", "window_output"),
        ):
            for value in ("image", "latent"):
                actual = array_sha256(arrays[f"{engine}-{kind}-{value}"])
                expected = result[section][f"{value}_sha256"]
                if actual != expected:
                    raise ComparisonError(
                        f"{engine}-{kind}-{value} digest must be {expected}, got {actual}"
                    )
    for value in ("image", "latent"):
        actual = array_sha256(arrays[f"dinkster-local-window-{value}"])
        expected = dinkster["position_id_diagnostic"][f"local_{value}_sha256"]
        if actual != expected:
            raise ComparisonError(
                f"dinkster-local-window-{value} digest must be {expected}, got {actual}"
            )
    verdict = build_verdict(comfyui, dinkster, arrays, dinkster_commit=args.dinkster_commit)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["overall_pass"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    comfyui = subparsers.add_parser("comfyui")
    comfyui.add_argument("--checkpoint", type=Path, required=True)
    comfyui.add_argument("--comfyui-root", type=Path, required=True)
    comfyui.add_argument("--output-dir", type=Path, required=True)
    comfyui.add_argument("--tiled-root", type=Path, required=True)
    comfyui.set_defaults(function=run_comfyui)

    dinkster = subparsers.add_parser("dinkster")
    dinkster.add_argument("--checkpoint", type=Path, required=True)
    dinkster.add_argument("--dinkster-commit", default=DINKSTER_COMMIT)
    dinkster.add_argument("--dinkster-root", type=Path, required=True)
    dinkster.add_argument("--output-dir", type=Path, required=True)
    dinkster.set_defaults(function=run_dinkster)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--dinkster-commit", default=DINKSTER_COMMIT)
    compare.add_argument("--input-dir", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(function=run_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
