"""Pinned plain Flux performance comparison against core ComfyUI."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
CHECKPOINT_SHA256 = "8e91b68084b53a7fc44ed2a3756d821e355ac1a7b6fe29be760c1db532f3d88a"
CHECKPOINT_SIZE = 17_246_524_772
DEVICE_UUID = "GPU-666d1242-9c20-341c-73ea-e63770947451"
PROMPT = "a red fox sitting beside a mountain lake at sunrise"
SEED = 424242
STEPS = 4
WIDTH = 512
HEIGHT = 512
MEASURED_RUNS = 5
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
    """The comparison contract or evidence is invalid."""


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_source(root: Path, expected_commit: str, name: str) -> None:
    actual = _git(root, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise ComparisonError(f"{name} must be at {expected_commit}, got {actual}")
    if _git(root, "status", "--porcelain"):
        raise ComparisonError(f"{name} source must be clean: {root}")


def _checkpoint_receipt(checkpoint: Path) -> dict[str, Any]:
    size = checkpoint.stat().st_size
    digest = _file_sha256(checkpoint)
    if size != CHECKPOINT_SIZE or digest != CHECKPOINT_SHA256:
        raise ComparisonError(
            f"checkpoint must be {CHECKPOINT_SIZE} bytes and sha256:{CHECKPOINT_SHA256}; "
            f"got {size} bytes and sha256:{digest}"
        )
    return {"path": str(checkpoint), "sha256": digest, "size_bytes": size}


def _driver_version() -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            f"--id={DEVICE_UUID}",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _device_receipt(torch: Any, device: Any) -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise ComparisonError(
            f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != DEVICE_UUID:
        raise ComparisonError(f"CUDA_VISIBLE_DEVICES must be {DEVICE_UUID}")
    properties = torch.cuda.get_device_properties(device)
    return {
        "cuda": torch.version.cuda,
        "device_uuid": DEVICE_UUID,
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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
    run: Callable[[], tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    *,
    process_start: int,
) -> dict[str, Any]:
    cold_latent, cold_image, cold_metrics = run()
    cold_metrics["load_text_sample_decode_seconds"] = (time.perf_counter_ns() - process_start) / 1e9
    warmup_latent, warmup_image, _ = run()
    measured: list[dict[str, Any]] = []
    for index in range(MEASURED_RUNS):
        latent, image, metrics = run()
        metrics.update(
            {
                "image_sha256": _array_sha256(image),
                "index": index,
                "latent_sha256": _array_sha256(latent),
            }
        )
        measured.append(metrics)
    sample_values = [float(item["sample_seconds"]) for item in measured]
    decode_values = [float(item["decode_seconds"]) for item in measured]
    end_values = [float(item["end_to_end_seconds"]) for item in measured]
    return {
        "cold": {
            **cold_metrics,
            "image_sha256": _array_sha256(cold_image),
            "latent_sha256": _array_sha256(cold_latent),
        },
        "measured": measured,
        "medians": {
            "decode_seconds": statistics.median(decode_values),
            "end_to_end_seconds": statistics.median(end_values),
            "sample_seconds": statistics.median(sample_values),
        },
        "ranges": {
            "decode_seconds": [min(decode_values), max(decode_values)],
            "end_to_end_seconds": [min(end_values), max(end_values)],
            "sample_seconds": [min(sample_values), max(sample_values)],
        },
        "warmup": {
            "image_sha256": _array_sha256(warmup_image),
            "latent_sha256": _array_sha256(warmup_latent),
        },
    }


def _workload_receipt(mode: str = "plain-native") -> dict[str, Any]:
    return {
        "guidance": 3.5,
        "height": HEIGHT,
        "measured_runs": MEASURED_RUNS,
        "mode": mode,
        "prompt": PROMPT,
        "sampler": "Euler",
        "scheduler": "normal",
        "seed": SEED,
        "steps": STEPS,
        "warmup_runs_discarded": 1,
        "width": WIDTH,
    }


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _profile_call(torch: Any, call: Callable[[], Any]) -> list[dict[str, Any]]:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as profile:
        with torch.inference_mode():
            call()
        torch.cuda.synchronize()
    events = sorted(
        profile.key_averages(group_by_input_shape=True),
        key=lambda event: event.self_device_time_total,
        reverse=True,
    )
    return [
        {
            "calls": event.count,
            "input_shapes": event.input_shapes,
            "key": event.key,
            "self_cpu_time_us": event.self_cpu_time_total,
            "self_device_time_us": event.self_device_time_total,
        }
        for event in events[:40]
    ]


def run_comfyui(args: argparse.Namespace) -> int:
    root = args.comfyui_root.resolve()
    checkpoint = args.checkpoint.resolve()
    _require_source(root, COMFYUI_COMMIT, "ComfyUI")
    checkpoint_receipt = _checkpoint_receipt(checkpoint)
    sys.path.insert(0, str(root))
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

    def sample() -> Any:
        return nodes.common_ksampler(
            model,
            SEED,
            STEPS,
            1.0,
            "euler",
            "normal",
            positive,
            negative,
            latent,
            denoise=1.0,
        )[0]

    def decode(sampled: Any) -> Any:
        return nodes.VAEDecode().decode(vae, sampled)[0]

    latest_sampled: list[Any] = []

    def capture_sample() -> Any:
        sampled = sample()
        latest_sampled[:] = [sampled]
        return sampled

    def run() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return _run_metrics(
            torch,
            device,
            capture_sample,
            decode,
            lambda sampled: sampled["samples"].detach().float().cpu().numpy(),
            lambda image: image.detach().float().cpu().numpy(),
        )

    summary = _summarize_runs(run, process_start=process_start)
    result = {
        **summary,
        "engine": "comfyui",
        "receipts": {
            **device_receipt,
            "attention_backend": "pytorch-sdpa",
            "checkpoint": checkpoint_receipt,
            "commit": COMFYUI_COMMIT,
            "initial_noise_sha256": _array_sha256(initial_noise.cpu().numpy()),
            "normal_schedule_float_hex": [
                float(value).hex() for value in ksampler.sigmas.cpu().tolist()
            ],
            "precision": precision,
        },
        "workload": _workload_receipt(),
    }
    _write_result(args.output, result)
    if args.profile_output is not None:
        profiles: dict[str, Any] = {}
        if args.profile_phase in ("decode", "both"):
            profiles["decode"] = _profile_call(torch, lambda: decode(latest_sampled[0]))
        if args.profile_phase in ("sample", "both"):
            profiles["sample"] = _profile_call(torch, sample)
        _write_result(
            args.profile_output,
            profiles,
        )
    gc.collect()
    model_management.unload_all_models()
    model_management.soft_empty_cache(force=True)
    return 0


def _add_dinkster_sources(root: Path) -> None:
    for source in sorted(root.glob("packages/*/src")):
        sys.path.insert(0, str(source))
    sys.path.insert(0, str(root / "src"))


def run_dinkster(args: argparse.Namespace) -> int:
    root = args.dinkster_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    _require_source(root, args.dinkster_commit, "Dinkster")
    checkpoint_receipt = _checkpoint_receipt(checkpoint_path)
    _add_dinkster_sources(root)

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
    latent = torch.zeros((1, 16, HEIGHT // 8, WIDTH // 8), dtype=torch.float32, device=device)
    window_plan = None
    if args.window_mode != "native":
        width_axis = MediaAxis("width", WIDTH // 16)
        kinds = (
            WindowKind(
                "latent_image",
                (KindAxisMap("width", WIDTH // 16, IntegerAffineIndexMap(1)),),
            ),
            WindowKind("text", invariant_axes=("width",)),
        )
        windows = (
            (tuple(range(32)),)
            if args.window_mode == "one-window"
            else (tuple(range(24)), tuple(range(8, 32)))
        )
        window_plan = compile_window_plan(
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

    def sample() -> Any:
        return runtime.sample(
            latent,
            cond=cond,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.normal",
            steps=STEPS,
            denoise=1.0,
            seed=SEED,
            guidance=3.5,
            window_plan=window_plan,
            compute_dtype=torch.bfloat16,
            device=device,
        )

    def decode(sampled: Any) -> Any:
        return runtime.decode_latent(sampled).permute(0, 2, 3, 1)

    latest_sampled: list[Any] = []

    def capture_sample() -> Any:
        sampled = sample()
        latest_sampled[:] = [sampled]
        return sampled

    def run() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return _run_metrics(
            torch,
            device,
            capture_sample,
            decode,
            lambda sampled: sampled.detach().float().cpu().numpy(),
            lambda image: image.detach().float().cpu().numpy(),
        )

    summary = _summarize_runs(run, process_start=process_start)
    status = runtime.attention_status["flux"]
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
    result = {
        **summary,
        "engine": "dinkster",
        "receipts": {
            **device_receipt,
            "attention_backend": status.primary,
            "attention_policy": status.requested_policy,
            "checkpoint": checkpoint_receipt,
            "commit": args.dinkster_commit,
            "initial_noise_sha256": _array_sha256(initial_noise.numpy()),
            "normal_schedule_float_hex": [float(value).hex() for value in sigmas],
            "precision": precision,
            "runtime_identity": runtime.runtime_identity,
            "window_plan_digest": None if window_plan is None else window_plan.digest,
        },
        "workload": _workload_receipt(
            "plain-native" if args.window_mode == "native" else f"dinkster-{args.window_mode}"
        ),
    }
    _write_result(args.output, result)
    if args.profile_output is not None:
        profiles = {}
        if args.profile_phase in ("decode", "both"):
            profiles["decode"] = _profile_call(torch, lambda: decode(latest_sampled[0]))
        if args.profile_phase in ("sample", "both"):
            profiles["sample"] = _profile_call(torch, sample)
        _write_result(
            args.profile_output,
            profiles,
        )
    runtime.assembled.diffusion.to("cpu")
    runtime.assembled.vae.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def _metric(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(run[key]) for record in records for run in record["measured"]]
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "mad_seconds": statistics.median(deviations),
        "median_seconds": median,
        "range_seconds": [min(values), max(values)],
        "values_seconds": values,
    }


def build_verdict(records: list[dict[str, Any]], *, dinkster_commit: str) -> dict[str, Any]:
    if [record["engine"] for record in records] != ["comfyui", "dinkster", "dinkster", "comfyui"]:
        raise ComparisonError("record order must be comfyui, dinkster, dinkster, comfyui")
    reference = records[0]
    for record in records:
        if record["workload"] != _workload_receipt():
            raise ComparisonError("workload receipt differs from the pinned plain-native workload")
        if len(record["measured"]) != MEASURED_RUNS:
            raise ComparisonError(f"each process must record exactly {MEASURED_RUNS} measured runs")
        warmup = record.get("warmup")
        if not isinstance(warmup, dict) or not all(
            isinstance(warmup.get(key), str) and warmup[key]
            for key in ("image_sha256", "latent_sha256")
        ):
            raise ComparisonError("each process must record one discarded resident warmup")
        schedule = record["receipts"].get("normal_schedule_float_hex")
        if (
            not isinstance(schedule, list)
            or len(schedule) != STEPS + 1
            or schedule[-1] != float(0).hex()
        ):
            raise ComparisonError(f"normal schedule must contain {STEPS + 1} values ending at zero")
        if (
            record["receipts"]["checkpoint"]["sha256"] != CHECKPOINT_SHA256
            or record["receipts"]["checkpoint"]["size_bytes"] != CHECKPOINT_SIZE
        ):
            raise ComparisonError("checkpoint receipt differs")
    for record in records[1:]:
        if record["workload"] != reference["workload"]:
            raise ComparisonError("workload receipts differ")
        for key in (
            "cuda",
            "device_uuid",
            "driver",
            "gpu",
            "gpu_total_memory",
            "python",
            "torch",
        ):
            if record["receipts"][key] != reference["receipts"][key]:
                raise ComparisonError(f"runtime receipt differs for {key}")
        if (
            record["receipts"]["initial_noise_sha256"]
            != reference["receipts"]["initial_noise_sha256"]
        ):
            raise ComparisonError("initial noise differs")
        if (
            record["receipts"]["normal_schedule_float_hex"]
            != reference["receipts"]["normal_schedule_float_hex"]
        ):
            raise ComparisonError("normal schedule differs")
    comfyui = [records[0], records[3]]
    dinkster = [records[1], records[2]]
    if any(
        record["receipts"]["commit"] != COMFYUI_COMMIT
        or record["receipts"]["attention_backend"] != "pytorch-sdpa"
        or record["receipts"]["precision"] != COMFYUI_PRECISION
        for record in comfyui
    ):
        raise ComparisonError("ComfyUI source, attention, or precision receipt differs")
    dinkster_commits = {record["receipts"]["commit"] for record in dinkster}
    if dinkster_commits != {dinkster_commit} or any(
        record["receipts"]["attention_backend"] != "sdpa"
        or record["receipts"]["attention_policy"] != "auto"
        or record["receipts"]["precision"] != DINKSTER_PRECISION
        or record["receipts"]["window_plan_digest"] is not None
        for record in dinkster
    ):
        raise ComparisonError(
            "Dinkster source, attention, precision, or native-mode receipt differs"
        )
    for engine_records in (comfyui, dinkster):
        engine = engine_records[0]["engine"]
        latent_hashes = {
            run["latent_sha256"] for record in engine_records for run in record["measured"]
        }
        image_hashes = {
            run["image_sha256"] for record in engine_records for run in record["measured"]
        }
        if len(latent_hashes) != 1 or len(image_hashes) != 1:
            raise ComparisonError(f"{engine} output is not deterministic across processes")

    metrics: dict[str, Any] = {}
    passes: list[bool] = []
    for key in ("sample_seconds", "decode_seconds", "end_to_end_seconds"):
        reference_metric = _metric(comfyui, key)
        candidate_metric = _metric(dinkster, key)
        allowance = max(
            reference_metric["median_seconds"] * 0.01,
            reference_metric["mad_seconds"] * 3.0,
        )
        passed = (
            candidate_metric["median_seconds"] <= reference_metric["median_seconds"] + allowance
        )
        passes.append(passed)
        metrics[key] = {
            "comfyui": reference_metric,
            "dinkster": candidate_metric,
            "dinkster_over_comfyui_ratio": (
                candidate_metric["median_seconds"] / reference_metric["median_seconds"]
            ),
            "noise_allowance_seconds": allowance,
            "pass": passed,
        }
    return {
        "metrics": metrics,
        "pass": all(passes),
        "policy": {
            "engine_order": ["comfyui", "dinkster", "dinkster", "comfyui"],
            "measured_runs_per_process": MEASURED_RUNS,
            "noise_floor_fraction": 0.01,
            "noise_mad_multiplier": 3.0,
            "warmup_runs_discarded_per_process": 1,
        },
        "process_records": records,
        "receipts": [record["receipts"] for record in records],
        "workload": reference["workload"],
    }


def run_compare(args: argparse.Namespace) -> int:
    records = [json.loads(path.read_text()) for path in args.inputs]
    verdict = build_verdict(records, dinkster_commit=args.dinkster_commit)
    _write_result(args.output, verdict)
    return 0 if verdict["pass"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    comfyui = commands.add_parser("comfyui")
    comfyui.add_argument("--checkpoint", type=Path, required=True)
    comfyui.add_argument("--comfyui-root", type=Path, required=True)
    comfyui.add_argument("--output", type=Path, required=True)
    comfyui.add_argument("--profile-output", type=Path)
    comfyui.add_argument("--profile-phase", choices=("sample", "decode", "both"), default="both")
    comfyui.set_defaults(func=run_comfyui)
    dinkster = commands.add_parser("dinkster")
    dinkster.add_argument("--checkpoint", type=Path, required=True)
    dinkster.add_argument("--dinkster-commit", required=True)
    dinkster.add_argument("--dinkster-root", type=Path, required=True)
    dinkster.add_argument("--output", type=Path, required=True)
    dinkster.add_argument("--profile-output", type=Path)
    dinkster.add_argument("--profile-phase", choices=("sample", "decode", "both"), default="both")
    dinkster.add_argument(
        "--window-mode", choices=("native", "one-window", "two-window"), default="native"
    )
    dinkster.set_defaults(func=run_dinkster)
    compare = commands.add_parser("compare")
    compare.add_argument("--dinkster-commit", required=True)
    compare.add_argument("--inputs", nargs=4, type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(func=run_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
