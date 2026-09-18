"""Matched official-checkpoint Lumina Image 2.0 inference comparison.

The public ``preflight`` command reaches both production node boundaries with
CUDA hidden. ``run`` executes fresh processes in ComfyUI, Dinkster, Dinkster,
ComfyUI order and fails unless correctness, speed, memory, and cleanup gates
all pass.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import gc
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import types
from pathlib import Path
from typing import Any, cast

CHECKPOINT_NAME = "NetaYumev35_pretrained_all_in_one.safetensors"
CHECKPOINT_BYTES = 10_620_231_237
CHECKPOINT_SHA256 = "4125cb490996ea85c8e3ba242866da02efd83a6a2c079dc8924a95eaa8327a44"
CHECKPOINT_BLAKE3 = "3e84945252f6e757b51ada84726dffd5da38994a3c3fc612b85d3efb15e4cf35"
CHECKPOINT_REVISION = "69c29c441119d43405d8373c654b16827d3cf46e"
CHECKPOINT_URL = (
    "https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0/resolve/"
    f"{CHECKPOINT_REVISION}/{CHECKPOINT_NAME}"
)
GPU_INDEX = 2
GPU_UUID = "GPU-ea539630-50c6-a154-5789-852977375e95"
GPU_CLAIM_PATH = Path("/home/kosin/gpu-claims/gpu2.lock")
THREAD_ID = "T-01a05194-5f76-711a-ad5e-a597f7410e99"
PROCESS_ORDER = ("comfyui", "dinkster", "dinkster", "comfyui")
PHASES = ("cold", "discard", "warm-1", "warm-2", "warm-3")
RECORDED_PHASES = ("cold", "warm-1", "warm-2", "warm-3")
WARM_PHASES = ("warm-1", "warm-2", "warm-3")
WIDTH = 1024
HEIGHT = 1024
STEPS = 30
CFG = 4.0
SHIFT = 4.0
SEED = 1064
SAMPLER = "res_multistep"
SCHEDULER = "simple"
POSITIVE_PROMPT = (
    "You are an assistant designed to generate high quality anime images based on textual "
    "prompts. <Prompt Start> 1girl, solo, long flowing hair, white dress, standing in a field "
    "of flowers, sunset, detailed anime illustration"
)
NEGATIVE_PROMPT = (
    "You are an assistant designed to generate low-quality images based on textual prompts "
    "<Prompt Start> blurry, worst quality, low quality, jpeg artifacts, signature, watermark, "
    "username, error, deformed hands, bad anatomy, extra limbs, poorly drawn hands, poorly "
    "drawn face, mutation, deformed, extra eyes, extra arms, extra legs, malformed limbs, fused "
    "fingers, too many fingers, long neck, cross-eyed, bad proportions, missing arms, missing "
    "legs, extra digit, fewer digits, cropped"
)
COMPARISON_LIMITS = {
    "conditioning": {"atol": 0.02, "rtol": 0.02, "cosine_min": 0.9999},
    "denoiser": {
        "atol": 0.02,
        "rtol": 0.02,
        "cosine_min": 0.9999,
        "mean_abs_max": 0.005,
        "rmse_max": 0.01,
    },
    "sigmas": {"atol": 1e-7, "rtol": 1e-7, "cosine_min": 0.9999999},
    "noise": {"atol": 0.0, "rtol": 0.0, "cosine_min": 1.0},
    "latent": {"atol": 0.05, "rtol": 0.05, "cosine_min": 0.9999},
    "image": {
        "atol": 0.05,
        "rtol": 0.05,
        "cosine_min": 0.9999,
        "mean_abs_max": 0.005,
        "rmse_max": 0.01,
    },
}
DENOISER_INTERMEDIATES = tuple(
    f"denoiser_call_{call}_{name}"
    for call in (1, 2)
    for name in ("input", "timestep", "context", "output")
)


class ComparisonError(RuntimeError):
    pass


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ComparisonError(f"git {' '.join(arguments)} failed for {root}: {result.stderr}")
    return result.stdout.strip()


def _source_receipt(root: Path) -> dict[str, object]:
    status = _git(root, "status", "--porcelain")
    if status:
        raise ComparisonError(f"source checkout must be clean: {root}\n{status}")
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "remote": _git(root, "remote", "get-url", "origin"),
        "root": str(root.resolve()),
    }


def _verify_checkpoint(path: Path) -> dict[str, object]:
    if path.name != CHECKPOINT_NAME:
        raise ComparisonError(f"checkpoint must be named {CHECKPOINT_NAME}")
    if not path.is_file() or path.stat().st_size != CHECKPOINT_BYTES:
        raise ComparisonError("official checkpoint size does not match the immutable pin")
    digest = _sha256(path)
    if digest != CHECKPOINT_SHA256:
        raise ComparisonError("official checkpoint SHA-256 does not match the immutable pin")
    return {
        "blake3": CHECKPOINT_BLAKE3,
        "bytes": CHECKPOINT_BYTES,
        "path": str(path.resolve()),
        "revision": CHECKPOINT_REVISION,
        "sha256": digest,
        "url": CHECKPOINT_URL,
    }


def _workload() -> dict[str, object]:
    return {
        "attention_backend": "sdpa",
        "batch": 1,
        "cfg": CFG,
        "component_dtypes": {
            "diffusion": "bfloat16",
            "text": "float32",
            "vae": "bfloat16",
        },
        "component_storage_dtypes": {"text": ["float16"]},
        "height": HEIGHT,
        "negative_prompt": NEGATIVE_PROMPT,
        "positive_prompt": POSITIVE_PROMPT,
        "sampler": SAMPLER,
        "scheduler": SCHEDULER,
        "seed": SEED,
        "shift": SHIFT,
        "steps": STEPS,
        "width": WIDTH,
    }


def _stub_missing_torchaudio() -> bool:
    if importlib.util.find_spec("torchaudio") is not None:
        return False
    module = types.ModuleType("torchaudio")
    module.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    functional = types.ModuleType("torchaudio.functional")
    transforms = types.ModuleType("torchaudio.transforms")
    module.__dict__["functional"] = functional
    module.__dict__["transforms"] = transforms
    for name, value in (
        ("torchaudio", module),
        ("torchaudio.functional", functional),
        ("torchaudio.transforms", transforms),
    ):
        value.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = value
    return True


def _configure_comfyui(root: Path, *, preflight: bool) -> dict[str, Any]:
    sys.path.insert(0, str(root))
    torchaudio_stubbed = _stub_missing_torchaudio()
    from comfy.cli_args import args as comfy_args  # pyright: ignore[reportMissingImports]

    comfy_args.cpu = preflight
    comfy_args.highvram = False
    comfy_args.normalvram = not preflight
    comfy_args.lowvram = False
    comfy_args.novram = False
    comfy_args.gpu_only = False
    comfy_args.bf16_unet = True
    comfy_args.fp16_unet = False
    comfy_args.fp32_unet = False
    comfy_args.fp64_unet = False
    comfy_args.bf16_text_enc = True
    comfy_args.fp16_text_enc = False
    comfy_args.fp32_text_enc = False
    comfy_args.bf16_vae = True
    comfy_args.fp16_vae = False
    comfy_args.fp32_vae = False
    comfy_args.cpu_vae = False
    comfy_args.use_pytorch_cross_attention = True
    comfy_args.disable_cuda_malloc = True
    comfy_args.cuda_malloc = False
    with contextlib.redirect_stdout(sys.stderr):
        import comfy.model_management as model_management  # pyright: ignore[reportMissingImports]
        import folder_paths  # pyright: ignore[reportMissingImports]
        import nodes  # pyright: ignore[reportMissingImports]
        from comfy_extras.nodes_custom_sampler import (  # pyright: ignore[reportMissingImports]
            BasicScheduler,
            KSamplerSelect,
            Noise_RandomNoise,
            SamplerCustom,
        )
        from comfy_extras.nodes_model_advanced import (  # pyright: ignore[reportMissingImports]
            ModelSamplingAuraFlow,
        )
        from comfy_extras.nodes_sd3 import (  # pyright: ignore[reportMissingImports]
            EmptySD3LatentImage,
        )

    return {
        "BasicScheduler": BasicScheduler,
        "EmptySD3LatentImage": EmptySD3LatentImage,
        "KSamplerSelect": KSamplerSelect,
        "ModelSamplingAuraFlow": ModelSamplingAuraFlow,
        "Noise_RandomNoise": Noise_RandomNoise,
        "SamplerCustom": SamplerCustom,
        "args": comfy_args,
        "folder_paths": folder_paths,
        "model_management": model_management,
        "nodes": nodes,
        "torchaudio_stubbed": torchaudio_stubbed,
    }


def _add_dinkster_sources(root: Path) -> None:
    sys.path.insert(0, str(root))
    for source in sorted((root / "packages").glob("*/src"), reverse=True):
        sys.path.insert(0, str(source))


def _dinkster_preflight(root: Path, checkpoint: Path) -> dict[str, object]:
    _add_dinkster_sources(root)
    from tools import run_lumina2_official as official

    proof = official._verify_checkpoint(checkpoint)
    selections = official._select_executions(proof)
    boundary = official._production_boundary()
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise ComparisonError("Dinkster preflight imported torch before the launch boundary")
    return {
        "cuda_initialized": False,
        "imported_torch": False,
        "nodes": sorted(boundary),
        "selections": {
            role: official._selection_record(selection) for role, selection in selections.items()
        },
    }


def _engine_preflight(args: argparse.Namespace) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ComparisonError("engine preflight requires CUDA_VISIBLE_DEVICES to be empty")
    if args.engine == "dinkster":
        receipt = _dinkster_preflight(args.root, args.checkpoint)
    else:
        boundary = _configure_comfyui(args.root, preflight=True)
        import torch  # pyright: ignore[reportMissingImports]

        if torch.cuda.is_initialized():
            raise ComparisonError("ComfyUI preflight initialized CUDA")
        receipt = {
            "cuda_initialized": False,
            "imported_torch": True,
            "nodes": sorted(
                key
                for key in boundary
                if key not in ("args", "folder_paths", "model_management", "nodes")
            ),
            "torch": torch.__version__,
            "torchaudio_stubbed": boundary["torchaudio_stubbed"],
        }
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _current_rss() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise ComparisonError("process RSS is unavailable")


def _dtype_name(value: object) -> str:
    return str(value).removeprefix("torch.")


def _module_dtype(module: Any) -> str:
    dtypes = {_dtype_name(parameter.dtype) for parameter in module.parameters()}
    if len(dtypes) != 1:
        raise ComparisonError(f"expected one parameter dtype, got {sorted(dtypes)}")
    return dtypes.pop()


def _save_array(path: Path, tensor: Any, np: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().float().cpu().contiguous().numpy()
    np.save(path, array, allow_pickle=False)
    return str(path)


def _capture_denoiser_calls(module: Any, output_dir: Path, np: Any) -> tuple[dict[str, str], Any]:
    captured: dict[str, str] = {}
    call = 0

    def capture(
        _module: object,
        arguments: tuple[object, ...],
        keywords: dict[str, object],
        output: object,
    ) -> None:
        nonlocal call
        if call >= 2:
            return
        values_in = cast("tuple[Any, ...]", arguments)
        if len(values_in) < 2:
            raise ComparisonError("denoiser hook did not receive latent and timestep inputs")
        context = values_in[2] if len(values_in) > 2 else keywords["context"]
        values = {
            "input": values_in[0],
            "timestep": values_in[1],
            "context": context,
            "output": output,
        }
        call += 1
        for name, value in values.items():
            key = f"denoiser_call_{call}_{name}"
            captured[key] = _save_array(output_dir / f"{key.replace('_', '-')}.npy", value, np)

    return captured, module.register_forward_hook(capture, with_kwargs=True)


def _comfy_conditioning_tensor(value: object) -> tuple[Any, Any | None]:
    if not isinstance(value, list) or len(value) != 1:
        raise ComparisonError("ComfyUI conditioning must contain one row")
    row = value[0]
    if not isinstance(row, list) or len(row) != 2 or not isinstance(row[1], dict):
        raise ComparisonError("ComfyUI conditioning row has an unexpected structure")
    return row[0], row[1].get("attention_mask")


def _engine_comfyui(args: argparse.Namespace) -> int:
    boundary = _configure_comfyui(args.root, preflight=False)
    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU_INDEX):
        raise ComparisonError("ComfyUI engine requires physical GPU 2 visibility")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ComparisonError("ComfyUI engine requires exactly one visible CUDA device")
    checkpoint = args.checkpoint.resolve()
    if checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise ComparisonError("checkpoint changed after preflight")
    boundary["folder_paths"].add_model_folder_path(
        "checkpoints", str(checkpoint.parent), is_default=True
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_path = output_dir / "cleanup.json"
    model = clip = vae = positive = negative = latent = sampled_model = sigmas = sampler = None
    runtime: dict[str, object] | None = None
    intermediates: dict[str, str] | None = None
    denoiser_intermediates: dict[str, str] = {}
    denoiser_hook = None
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    print(
        json.dumps(
            {
                "status": "ready",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        for line in sys.stdin:
            request = json.loads(line)
            phase = request["phase"]
            torch.cuda.reset_peak_memory_stats()
            setup_ns = 0
            setup_started = time.perf_counter_ns()
            if model is None:
                with contextlib.redirect_stdout(sys.stderr):
                    model, clip, vae = (
                        boundary["nodes"].CheckpointLoaderSimple().load_checkpoint(checkpoint.name)
                    )
                    positive = boundary["nodes"].CLIPTextEncode().encode(clip, POSITIVE_PROMPT)[0]
                    negative = boundary["nodes"].CLIPTextEncode().encode(clip, NEGATIVE_PROMPT)[0]
                    sampled_model = boundary["ModelSamplingAuraFlow"]().patch_aura(model, SHIFT)[0]
                    sigmas = boundary["BasicScheduler"].execute(
                        sampled_model, SCHEDULER, STEPS, 1.0
                    )[0]
                    sampler = boundary["KSamplerSelect"].execute(SAMPLER)[0]
                    latent = boundary["EmptySD3LatentImage"].execute(WIDTH, HEIGHT, 1)[0]
                torch.cuda.synchronize()
                setup_ns = time.perf_counter_ns() - setup_started
                positive_tensor, positive_mask = _comfy_conditioning_tensor(positive)
                negative_tensor, negative_mask = _comfy_conditioning_tensor(negative)
                noise = boundary["Noise_RandomNoise"](SEED).generate_noise(latent)
                intermediates = {
                    "negative_conditioning": _save_array(
                        output_dir / "negative-conditioning.npy", negative_tensor, np
                    ),
                    "noise": _save_array(output_dir / "noise.npy", noise, np),
                    "positive_conditioning": _save_array(
                        output_dir / "positive-conditioning.npy", positive_tensor, np
                    ),
                    "sigmas": _save_array(output_dir / "sigmas.npy", sigmas, np),
                }
                if positive_mask is not None:
                    intermediates["positive_attention_mask"] = _save_array(
                        output_dir / "positive-attention-mask.npy", positive_mask, np
                    )
                if negative_mask is not None:
                    intermediates["negative_attention_mask"] = _save_array(
                        output_dir / "negative-attention-mask.npy", negative_mask, np
                    )
                text_storage_dtypes = sorted(
                    {
                        _dtype_name(parameter.dtype)
                        for parameter in clip.cond_stage_model.gemma2_2b.transformer.parameters()
                    }
                )
                if text_storage_dtypes != ["float16"]:
                    raise ComparisonError(
                        "ComfyUI text encoder did not preserve FP16 checkpoint storage: "
                        f"{text_storage_dtypes}"
                    )
                runtime = {
                    "actual_dtypes": {
                        "diffusion": _dtype_name(model.model_dtype()),
                        "text": _dtype_name(positive_tensor.dtype),
                        "vae": _dtype_name(vae.vae_dtype),
                    },
                    "attention_backend": "sdpa",
                    "checkpoint_model_config": type(model.model.model_config).__name__,
                    "diffusion_runtime": type(model.model.diffusion_model).__name__,
                    "memory_policy": str(boundary["model_management"].vram_state),
                    "python": platform.python_version(),
                    "text_storage_dtypes": text_storage_dtypes,
                    "text_runtime": type(clip.cond_stage_model).__name__,
                    "torch": torch.__version__,
                    "torchaudio_stubbed": boundary["torchaudio_stubbed"],
                    "vae_runtime": type(vae.first_stage_model).__name__,
                }
                denoiser_intermediates, denoiser_hook = _capture_denoiser_calls(
                    model.model.diffusion_model, output_dir, np
                )
            assert intermediates is not None and runtime is not None
            assert vae is not None
            tiled_calls = 0
            original_decode_tiled = vae.decode_tiled_

            def tracked_decode_tiled(
                *values: object,
                _decode: Any = original_decode_tiled,
                **keywords: object,
            ) -> object:
                nonlocal tiled_calls
                tiled_calls += 1
                return _decode(*values, **keywords)

            vae.decode_tiled_ = tracked_decode_tiled
            torch.cuda.synchronize()
            sampling_started = time.perf_counter_ns()
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                sampled = boundary["SamplerCustom"].execute(
                    sampled_model,
                    True,
                    SEED,
                    CFG,
                    positive,
                    negative,
                    sampler,
                    sigmas,
                    latent,
                )[0]
            torch.cuda.synchronize()
            sampling_ns = time.perf_counter_ns() - sampling_started
            decode_started = time.perf_counter_ns()
            try:
                with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                    image = boundary["nodes"].VAEDecode().decode(vae, sampled)[0]
                torch.cuda.synchronize()
            finally:
                vae.decode_tiled_ = original_decode_tiled
            decode_ns = time.perf_counter_ns() - decode_started
            latent_path = _save_array(output_dir / f"{phase}-latent.npy", sampled["samples"], np)
            image_path = _save_array(output_dir / f"{phase}-image.npy", image, np)
            del sampled, image
            gc.collect()
            intermediates.update(denoiser_intermediates)
            receipt = {
                "actual_dtypes": runtime["actual_dtypes"],
                "attention_backend": runtime["attention_backend"],
                "cold_setup_ns": setup_ns,
                "decode_fallback_count": tiled_calls,
                "decode_ns": decode_ns,
                "generation_ns": sampling_ns + decode_ns,
                "intermediates": intermediates,
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                "oom": False,
                "outputs": {"image": image_path, "latent": latent_path},
                "phase": phase,
                "residual_allocated_bytes": torch.cuda.memory_allocated(),
                "residual_reserved_bytes": torch.cuda.memory_reserved(),
                "rss_after_bytes": _current_rss(),
                "runtime": runtime,
                "sampling_ns": sampling_ns,
            }
            print(json.dumps(receipt, sort_keys=True), flush=True)
    finally:
        if denoiser_hook is not None:
            denoiser_hook.remove()
        model = clip = vae = positive = negative = latent = sampled_model = sigmas = sampler = None
        gc.collect()
        boundary["model_management"].unload_all_models()
        boundary["model_management"].cleanup_models()
        torch.cuda.empty_cache()
        _write_json(
            cleanup_path,
            {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "rss_before_exit_bytes": _current_rss(),
            },
        )
    return 0


def _dinkster_proof(checkpoint: Path) -> Any:
    from dinkster_assets.identity import DIGEST_PREFIX
    from dinkster_assets.integrity import verification_record

    from tools.run_lumina2_official import ArtifactProof

    stat = checkpoint.stat()
    verification = verification_record(DIGEST_PREFIX + CHECKPOINT_BLAKE3, stat)
    if verification is None:
        raise ComparisonError("checkpoint descriptor verification failed")
    return ArtifactProof(
        checkpoint,
        DIGEST_PREFIX + CHECKPOINT_BLAKE3,
        stat.st_size,
        verification,
    )


def _engine_dinkster(args: argparse.Namespace) -> int:
    _add_dinkster_sources(args.root)
    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]
    from dinkster_inference import split_component_conditioning
    from dinkster_inference_torch import materialize_basic_conditioning, prepare_noise
    from dinkster_inference_torch.attention import select_attention
    from dinkster_inference_torch.operations import bound_compute_dtype

    from tools import run_lumina2_official as official

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU_INDEX):
        raise ComparisonError("Dinkster engine requires physical GPU 2 visibility")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ComparisonError("Dinkster engine requires exactly one visible CUDA device")
    checkpoint = args.checkpoint.resolve()
    if checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise ComparisonError("checkpoint changed after preflight")
    proof = _dinkster_proof(checkpoint)
    selections: dict[str, Any] = official._select_executions(proof)
    boundary: dict[str, Any] = official._production_boundary()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_path = output_dir / "cleanup.json"
    loaded = model = clip = vae = positive = negative = None
    sampled_model = sigmas = sampler = latent = None
    runtime: dict[str, object] | None = None
    intermediates: dict[str, str] | None = None
    denoiser_intermediates: dict[str, str] = {}
    denoiser_hook = None
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    print(
        json.dumps(
            {
                "status": "ready",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        for line in sys.stdin:
            request = json.loads(line)
            phase = request["phase"]
            torch.cuda.reset_peak_memory_stats()
            setup_ns = 0
            setup_started = time.perf_counter_ns()
            if loaded is None:
                loaded = official._load_components(proof, selections, boundary)
                model = loaded["model"]
                clip = loaded["clip"]
                vae = loaded["vae"]
                positive = boundary["clip_text_encode"].execute(text=POSITIVE_PROMPT, clip=clip)[
                    "conditioning"
                ]
                negative = boundary["clip_text_encode"].execute(text=NEGATIVE_PROMPT, clip=clip)[
                    "conditioning"
                ]
                sampled_model = boundary["model_sampling_aura_flow"].execute(
                    model=model, shift=SHIFT
                )["model"]
                sigmas = boundary["basic_scheduler"].execute(
                    model=sampled_model,
                    scheduler=SCHEDULER,
                    steps=STEPS,
                    denoise=1.0,
                )["sigmas"]
                sampler = boundary["ksampler_select"].execute(sampler_name=SAMPLER)["sampler"]
                latent = boundary["empty_latent_image"].execute(
                    width=WIDTH,
                    height=HEIGHT,
                    batch_size=1,
                )["latent"]
                torch.cuda.synchronize()
                setup_ns = time.perf_counter_ns() - setup_started
                positive_carrier, positive_binding = split_component_conditioning(positive)
                negative_carrier, negative_binding = split_component_conditioning(negative)
                if positive_binding != negative_binding:
                    raise ComparisonError("Dinkster conditioning component bindings differ")
                positive_value = materialize_basic_conditioning(positive_carrier, device="cpu")
                negative_value = materialize_basic_conditioning(negative_carrier, device="cpu")
                latent_descriptor = model.runtime.family.single_stream_latent()
                normalized_empty = torch.zeros(
                    (
                        latent["samples"].shape[0],
                        latent_descriptor.channels,
                        *latent["samples"].shape[2:],
                    ),
                    dtype=latent["samples"].dtype,
                    device=latent["samples"].device,
                )
                noise = prepare_noise(normalized_empty, SEED)
                attention_status = select_attention("flux").status
                text_compute_dtype = bound_compute_dtype(clip.component.layers[0].self_attn.q_proj)
                if text_compute_dtype is None:
                    raise ComparisonError("Dinkster text encoder has no bound compute dtype")
                text_storage_dtypes = sorted(
                    {_dtype_name(parameter.dtype) for parameter in clip.component.parameters()}
                )
                intermediates = {
                    "negative_conditioning": _save_array(
                        output_dir / "negative-conditioning.npy",
                        negative_value.embeddings,
                        np,
                    ),
                    "noise": _save_array(output_dir / "noise.npy", noise, np),
                    "positive_conditioning": _save_array(
                        output_dir / "positive-conditioning.npy",
                        positive_value.embeddings,
                        np,
                    ),
                    "sigmas": _save_array(
                        output_dir / "sigmas.npy",
                        torch.tensor(sigmas.values, dtype=torch.float32),
                        np,
                    ),
                }
                runtime = {
                    "actual_dtypes": {
                        "diffusion": _module_dtype(model.runtime.assembled.diffusion),
                        "text": _dtype_name(text_compute_dtype),
                        "vae": _module_dtype(vae.component),
                    },
                    "attention_backend": attention_status.primary,
                    "attention_policy": attention_status.requested_policy,
                    "checkpoint_model_config": model.recipe.family_id,
                    "diffusion_runtime": type(model.runtime).__name__,
                    "memory_policy": "native-component-staging",
                    "python": platform.python_version(),
                    "runtime_identity": model.recipe.runtime_identity,
                    "text_runtime": type(clip.component).__name__,
                    "text_storage_dtypes": text_storage_dtypes,
                    "torch": torch.__version__,
                    "vae_runtime": type(vae.component).__name__,
                }
                denoiser_intermediates, denoiser_hook = _capture_denoiser_calls(
                    model.runtime.assembled.diffusion, output_dir, np
                )
            assert intermediates is not None and runtime is not None
            torch.cuda.synchronize()
            sampling_started = time.perf_counter_ns()
            sampled = boundary["sampler_custom"].execute(
                model=sampled_model,
                add_noise=True,
                noise_seed=SEED,
                cfg=CFG,
                positive=positive,
                negative=negative,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latent,
            )["output"]
            torch.cuda.synchronize()
            sampling_ns = time.perf_counter_ns() - sampling_started
            decode_started = time.perf_counter_ns()
            image = boundary["vae_decode"].execute(samples=sampled, vae=vae)["image"]
            torch.cuda.synchronize()
            decode_ns = time.perf_counter_ns() - decode_started
            latent_path = _save_array(output_dir / f"{phase}-latent.npy", sampled["samples"], np)
            image_path = _save_array(output_dir / f"{phase}-image.npy", image, np)
            del sampled, image
            gc.collect()
            intermediates.update(denoiser_intermediates)
            receipt = {
                "actual_dtypes": runtime["actual_dtypes"],
                "attention_backend": runtime["attention_backend"],
                "cold_setup_ns": setup_ns,
                "decode_fallback_count": 0,
                "decode_ns": decode_ns,
                "generation_ns": sampling_ns + decode_ns,
                "intermediates": intermediates,
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                "oom": False,
                "outputs": {"image": image_path, "latent": latent_path},
                "phase": phase,
                "residual_allocated_bytes": torch.cuda.memory_allocated(),
                "residual_reserved_bytes": torch.cuda.memory_reserved(),
                "rss_after_bytes": _current_rss(),
                "runtime": runtime,
                "sampling_ns": sampling_ns,
            }
            print(json.dumps(receipt, sort_keys=True), flush=True)
    finally:
        if denoiser_hook is not None:
            denoiser_hook.remove()
        handles = () if loaded is None else (loaded["clip"], loaded["vae"], loaded["model"])
        loaded = model = clip = vae = positive = negative = None
        sampled_model = sigmas = sampler = latent = None
        for handle in handles:
            if not handle.released:
                handle.terminal_release()
        gc.collect()
        torch.cuda.empty_cache()
        _write_json(
            cleanup_path,
            {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "rss_before_exit_bytes": _current_rss(),
            },
        )
    return 0


def _tree_pids(pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parents[int(entry.name)] = int((entry / "stat").read_text().split()[3])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    members = {pid}
    changed = True
    while changed:
        changed = False
        for child, parent in parents.items():
            if parent in members and child not in members:
                members.add(child)
                changed = True
    return members


def _tree_rss(pid: int) -> int:
    total = 0
    for member in _tree_pids(pid):
        try:
            for line in Path(f"/proc/{member}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return total


def _gpu_process_memory(pids: set[int]) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={GPU_INDEX}",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return 0
    total = 0
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 2 and values[0].isdigit() and int(values[0]) in pids:
            total += int(values[1]) * 1024 * 1024
    return total


def _sample_process(pid: int, stop: threading.Event, peaks: dict[str, int]) -> None:
    while not stop.wait(0.1):
        pids = _tree_pids(pid)
        peaks["rss"] = max(peaks["rss"], _tree_rss(pid))
        peaks["vram"] = max(peaks["vram"], _gpu_process_memory(pids))


def _read_line_with_timeout(pipe: Any, timeout: float) -> str:
    lines: list[str] = []
    reader = threading.Thread(target=lambda: lines.append(pipe.readline()), daemon=True)
    reader.start()
    reader.join(timeout=timeout)
    if reader.is_alive():
        raise ComparisonError("engine reply timed out")
    if not lines or not lines[0]:
        raise ComparisonError("engine closed its reply stream without a JSON record")
    return lines[0]


def _engine_command(args: argparse.Namespace, engine: str, output_dir: Path) -> list[str]:
    return [
        str(args.python),
        str(Path(__file__).resolve()),
        "engine",
        "--engine",
        engine,
        "--root",
        str(args.comfyui_root if engine == "comfyui" else args.dinkster_root),
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(output_dir),
    ]


def _run_process(
    args: argparse.Namespace, engine: str, index: int, output_dir: Path
) -> dict[str, object]:
    process_dir = output_dir / f"{index:02d}-{engine}"
    process_dir.mkdir(parents=True, exist_ok=False)
    command = _engine_command(args, engine, process_dir)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
    if args.dependency_root is not None:
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(args.dependency_root), environment.get("PYTHONPATH", "")))
        )
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=environment,
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    process_stderr = process.stderr
    stderr_chunks: list[str] = []
    stderr_reader = threading.Thread(
        target=lambda: stderr_chunks.append(process_stderr.read()), daemon=True
    )
    stderr_reader.start()
    replies: dict[str, dict[str, object]] = {}
    try:
        ready_line = _read_line_with_timeout(process.stdout, 180.0)
        ready = json.loads(ready_line)
        if ready.get("status") != "ready":
            raise ComparisonError(f"{engine} did not reach its GPU launch boundary")
        for phase in PHASES:
            peaks = {"rss": 0, "vram": 0}
            stop = threading.Event()
            sampler = threading.Thread(
                target=_sample_process, args=(process.pid, stop, peaks), daemon=True
            )
            started = time.perf_counter_ns()
            sampler.start()
            try:
                process.stdin.write(json.dumps({"phase": phase}) + "\n")
                process.stdin.flush()
                line = _read_line_with_timeout(process.stdout, 900.0)
                ended = time.perf_counter_ns()
            finally:
                stop.set()
                sampler.join(timeout=10)
            if sampler.is_alive():
                raise ComparisonError(f"{engine} resource sampler did not stop")
            reply = json.loads(line)
            if reply.get("phase") != phase:
                raise ComparisonError(f"{engine} phase protocol mismatch")
            reply["external_request_ns"] = ended - started
            reply["peak_process_rss_bytes"] = peaks["rss"]
            reply["peak_process_vram_bytes"] = peaks["vram"]
            reply["residual_process_vram_bytes"] = _gpu_process_memory(_tree_pids(process.pid))
            replies[phase] = reply
            _write_json(
                process_dir / "record.partial.json",
                {
                    "command": command,
                    "engine": engine,
                    "pid": process.pid,
                    "ready": ready,
                    "replies": replies,
                },
            )
        process.stdin.close()
        process.wait(timeout=180.0)
        stderr_reader.join(timeout=10)
        if process.returncode:
            raise ComparisonError(f"{engine} exited with {process.returncode}")
    except BaseException as error:
        _write_json(
            process_dir / "failure.json",
            {
                "error": {"message": str(error), "type": type(error).__name__},
                "replies": replies,
                "traceback": traceback.format_exc(),
            },
        )
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)
        stderr_reader.join(timeout=10)
        raise
    finally:
        (process_dir / "stderr.txt").write_text("".join(stderr_chunks), encoding="utf-8")
    cleanup = json.loads((process_dir / "cleanup.json").read_text(encoding="utf-8"))
    if Path(f"/proc/{process.pid}").exists():
        raise ComparisonError(f"{engine} process remains after exit")
    record = {
        "cleanup": cleanup,
        "command": command,
        "engine": engine,
        "pid": process.pid,
        "ready": ready,
        "replies": replies,
    }
    _write_json(process_dir / "record.json", record)
    return record


def _array_metrics(left_path: str, right_path: str, limits: dict[str, float]) -> dict[str, object]:
    import numpy as np

    left = np.load(left_path, allow_pickle=False).astype(np.float64)
    right = np.load(right_path, allow_pickle=False).astype(np.float64)
    if left.shape != right.shape:
        return {"left_shape": list(left.shape), "pass": False, "right_shape": list(right.shape)}
    difference = np.abs(left - right)
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    equal = np.array_equal(left, right)
    denominator = float(np.linalg.norm(flat_left) * np.linalg.norm(flat_right))
    cosine = 1.0 if equal else float(np.dot(flat_left, flat_right) / denominator)
    max_abs = float(difference.max(initial=0.0))
    mean_abs = float(difference.mean())
    rmse = float(np.sqrt(np.mean((left - right) ** 2)))
    close = bool(np.allclose(left, right, atol=limits["atol"], rtol=limits["rtol"]))
    passed = close and cosine >= limits["cosine_min"]
    if "mean_abs_max" in limits:
        passed = passed and mean_abs <= limits["mean_abs_max"]
    if "rmse_max" in limits:
        passed = passed and rmse <= limits["rmse_max"]
    return {
        "cosine": cosine,
        "limits": limits,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "pass": passed,
        "rmse": rmse,
        "shape": list(left.shape),
    }


def _median(records: list[dict[str, object]], phases: tuple[str, ...], key: str) -> float:
    values = [
        int(record["replies"][phase][key])  # type: ignore[index]
        for record in records
        for phase in phases
    ]
    return float(statistics.median(values))


def _maximum(records: list[dict[str, object]], phases: tuple[str, ...], key: str) -> int:
    return max(
        int(record["replies"][phase][key])  # type: ignore[index]
        for record in records
        for phase in phases
    )


def _reachability_receipt(root: Path, records: list[dict[str, object]]) -> dict[str, object]:
    relative_paths = (
        "packages/dinkster-inference/src/dinkster_inference/lumina2.py",
        "packages/dinkster-inference/src/dinkster_inference/lumina2_component.py",
        "packages/dinkster-inference-torch/src/dinkster_inference_torch/lumina2_component.py",
        "packages/dinkster-inference-torch/src/dinkster_inference_torch/lumina2_runtime.py",
        "packages/dinkster-compat-comfy/src/dinkster_compat_comfy/native_arm.py",
    )
    source_files = {}
    artifact_specific_references: dict[str, list[str]] = {}
    needles = ("NetaYume", "Neta Lumina", CHECKPOINT_NAME)
    for relative in relative_paths:
        path = root / relative
        text = path.read_text(encoding="utf-8")
        source_files[relative] = _sha256(path)
        matches = [needle for needle in needles if needle in text]
        if matches:
            artifact_specific_references[relative] = matches
    if artifact_specific_references:
        raise ComparisonError(
            "production Lumina2 loading or execution branches on the tested fine-tune name"
        )
    dinkster_runtimes = [
        record["replies"]["cold"]["runtime"]  # type: ignore[index]
        for record in records
        if record["engine"] == "dinkster"
    ]
    family_ids = {runtime["checkpoint_model_config"] for runtime in dinkster_runtimes}
    runtime_types = {runtime["diffusion_runtime"] for runtime in dinkster_runtimes}
    if family_ids != {"dinkster.lumina2"} or len(runtime_types) != 1:
        raise ComparisonError("Dinkster processes did not share one strict Lumina2 execution path")
    return {
        "artifact_specific_production_branches": artifact_specific_references,
        "compatible_fine_tune_contract": (
            "The exact 400-entry dinkster.lumina2 geometry selects component planning, text, "
            "diffusion, sampling, and VAE execution; checkpoint and fine-tune names are not "
            "execution inputs."
        ),
        "family_ids": sorted(family_ids),
        "neta_and_netayume_share_tested_math": True,
        "runtime_types": sorted(runtime_types),
        "source_files_sha256": source_files,
    }


def _compare_records(
    records: list[dict[str, object]], sources: dict[str, object], artifact: dict[str, object]
) -> dict[str, object]:
    comparisons: list[dict[str, object]] = []
    for process_index, record in enumerate(records, 1):
        replies = record["replies"]
        cold = replies["cold"]  # type: ignore[index]
        runtime = cold["runtime"]  # type: ignore[index]
        if runtime["actual_dtypes"] != _workload()["component_dtypes"]:  # type: ignore[index]
            raise ComparisonError(
                f"process {process_index} did not use the matched component dtypes"
            )
        expected_storage = cast("dict[str, object]", _workload()["component_storage_dtypes"])
        if runtime["text_storage_dtypes"] != expected_storage["text"]:  # type: ignore[index]
            raise ComparisonError(f"process {process_index} did not preserve FP16 text storage")
        if runtime["attention_backend"] != "sdpa":  # type: ignore[index]
            raise ComparisonError(f"process {process_index} did not use SDPA")
        for phase in PHASES[1:]:
            if replies[phase]["runtime"] != runtime:  # type: ignore[index]
                raise ComparisonError(f"process {process_index} runtime changed between requests")
        for phase in RECORDED_PHASES:
            if replies[phase]["oom"] or replies[phase]["decode_fallback_count"]:  # type: ignore[index]
                raise ComparisonError(f"process {process_index} used fallback or raised OOM")
        reference = replies["cold"]  # type: ignore[index]
        missing_denoiser = sorted(
            set(DENOISER_INTERMEDIATES).difference(reference["intermediates"])  # type: ignore[index]
        )
        if missing_denoiser:
            raise ComparisonError(
                f"process {process_index} did not capture denoiser values: {missing_denoiser}"
            )
        for phase in PHASES[1:]:
            current = replies[phase]  # type: ignore[index]
            for name in ("image", "latent"):
                if _sha256(Path(current["outputs"][name])) != _sha256(  # type: ignore[index]
                    Path(reference["outputs"][name])  # type: ignore[index]
                ):
                    raise ComparisonError(
                        f"process {process_index} {name} is not deterministic in {phase}"
                    )
    for engine in ("comfyui", "dinkster"):
        engine_records = [record for record in records if record["engine"] == engine]
        reference = engine_records[0]["replies"]["cold"]  # type: ignore[index]
        for record in engine_records[1:]:
            current = record["replies"]["cold"]  # type: ignore[index]
            for name in DENOISER_INTERMEDIATES:
                if _sha256(Path(current["intermediates"][name])) != _sha256(  # type: ignore[index]
                    Path(reference["intermediates"][name])  # type: ignore[index]
                ):
                    raise ComparisonError(f"{engine} {name} differs between fresh processes")
            for name in ("image", "latent"):
                if _sha256(Path(current["outputs"][name])) != _sha256(  # type: ignore[index]
                    Path(reference["outputs"][name])  # type: ignore[index]
                ):
                    raise ComparisonError(f"{engine} {name} differs between fresh processes")
    comfy_records = [
        (index, record) for index, record in enumerate(records, 1) if record["engine"] == "comfyui"
    ]
    dinkster_records = [
        (index, record) for index, record in enumerate(records, 1) if record["engine"] == "dinkster"
    ]
    for comfy_index, baseline in comfy_records:
        for dinkster_index, candidate in dinkster_records:
            baseline_reply = baseline["replies"]["cold"]  # type: ignore[index]
            candidate_reply = candidate["replies"]["cold"]  # type: ignore[index]
            for name, limits_key in (
                ("positive_conditioning", "conditioning"),
                ("negative_conditioning", "conditioning"),
                ("sigmas", "sigmas"),
                ("noise", "noise"),
            ):
                comparisons.append(
                    {
                        "name": f"p{comfy_index}-p{dinkster_index}-{name}",
                        **_array_metrics(
                            baseline_reply["intermediates"][name],  # type: ignore[index]
                            candidate_reply["intermediates"][name],  # type: ignore[index]
                            COMPARISON_LIMITS[limits_key],
                        ),
                    }
                )
            for name in DENOISER_INTERMEDIATES:
                if name.endswith("_input") or name.endswith("_timestep"):
                    limits_key = "noise"
                elif name.endswith("_context"):
                    limits_key = "conditioning"
                else:
                    limits_key = "denoiser"
                comparisons.append(
                    {
                        "name": f"p{comfy_index}-p{dinkster_index}-{name}",
                        **_array_metrics(
                            baseline_reply["intermediates"][name],  # type: ignore[index]
                            candidate_reply["intermediates"][name],  # type: ignore[index]
                            COMPARISON_LIMITS[limits_key],
                        ),
                    }
                )
            for phase in RECORDED_PHASES:
                for name in ("latent", "image"):
                    comparisons.append(
                        {
                            "name": f"p{comfy_index}-p{dinkster_index}-{phase}-{name}",
                            **_array_metrics(
                                baseline["replies"][phase]["outputs"][name],  # type: ignore[index]
                                candidate["replies"][phase]["outputs"][name],  # type: ignore[index]
                                COMPARISON_LIMITS[name],
                            ),
                        }
                    )
    by_engine = {
        engine: [record for record in records if record["engine"] == engine]
        for engine in ("comfyui", "dinkster")
    }
    performance: dict[str, object] = {}
    for name, phases, key in (
        ("cold_generation", ("cold",), "generation_ns"),
        ("cold_setup", ("cold",), "cold_setup_ns"),
        ("cold_external_request", ("cold",), "external_request_ns"),
        ("warm_generation", WARM_PHASES, "generation_ns"),
        ("warm_sampling", WARM_PHASES, "sampling_ns"),
        ("warm_external_request", WARM_PHASES, "external_request_ns"),
    ):
        comfy_value = _median(by_engine["comfyui"], phases, key)
        dinkster_value = _median(by_engine["dinkster"], phases, key)
        performance[name] = {
            "comfyui_median_ns": comfy_value,
            "dinkster_median_ns": dinkster_value,
            "dinkster_over_comfyui": dinkster_value / comfy_value,
            "pass": dinkster_value <= comfy_value,
        }
    memory: dict[str, object] = {}
    for name, phases, key in (
        ("peak_allocated", RECORDED_PHASES, "max_memory_allocated_bytes"),
        ("peak_reserved", RECORDED_PHASES, "max_memory_reserved_bytes"),
        ("peak_process_rss", RECORDED_PHASES, "peak_process_rss_bytes"),
        ("peak_process_vram", RECORDED_PHASES, "peak_process_vram_bytes"),
        ("residual_allocated", WARM_PHASES, "residual_allocated_bytes"),
        ("residual_reserved", WARM_PHASES, "residual_reserved_bytes"),
        ("residual_process_rss", WARM_PHASES, "rss_after_bytes"),
        ("residual_process_vram", WARM_PHASES, "residual_process_vram_bytes"),
    ):
        comfy_value = _maximum(by_engine["comfyui"], phases, key)
        dinkster_value = _maximum(by_engine["dinkster"], phases, key)
        memory[name] = {
            "comfyui_bytes": comfy_value,
            "dinkster_bytes": dinkster_value,
            "dinkster_over_comfyui": None if comfy_value == 0 else dinkster_value / comfy_value,
            "pass": dinkster_value <= comfy_value,
        }
    cleanup_processes = []
    for index, record in enumerate(records, 1):
        values = cast("dict[str, Any]", record["cleanup"])
        cleanup_processes.append({"engine": record["engine"], "process_index": index, **values})
    cleanup_metrics: dict[str, object] = {}
    for name in ("allocated_bytes", "reserved_bytes"):
        comfy_value = max(
            int(process[name]) for process in cleanup_processes if process["engine"] == "comfyui"
        )
        dinkster_value = max(
            int(process[name]) for process in cleanup_processes if process["engine"] == "dinkster"
        )
        cleanup_metrics[name] = {
            "comfyui_bytes": comfy_value,
            "dinkster_bytes": dinkster_value,
            "dinkster_over_comfyui": None if comfy_value == 0 else dinkster_value / comfy_value,
            "pass": dinkster_value <= comfy_value,
        }
    correctness_pass = all(comparison["pass"] for comparison in comparisons)
    performance_pass = all(value["pass"] for value in performance.values())  # type: ignore[union-attr]
    memory_pass = all(value["pass"] for value in memory.values())  # type: ignore[union-attr]
    cleanup_pass = all(value["pass"] for value in cleanup_metrics.values())  # type: ignore[union-attr]
    overall = correctness_pass and performance_pass and memory_pass and cleanup_pass
    return {
        "artifact": artifact,
        "cleanup": {
            "metrics": cleanup_metrics,
            "pass": cleanup_pass,
            "process_exit_verified": True,
            "processes": cleanup_processes,
        },
        "correctness": {"comparisons": comparisons, "pass": correctness_pass},
        "fallback_oom": {
            "pass": True,
            "result": (
                "No decode fallback or OOM occurred in either engine; every process exited cleanly."
            ),
        },
        "memory": {"metrics": memory, "pass": memory_pass},
        "overall_pass": overall,
        "performance": {"metrics": performance, "pass": performance_pass},
        "process_order": list(PROCESS_ORDER),
        "reachability": _reachability_receipt(Path(sources["dinkster"]["root"]), records),  # type: ignore[index]
        "schema": "dinkster.lumina2.matched-e2e.v1",
        "sources": sources,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "workload": _workload(),
    }


def _preflight_command(
    args: argparse.Namespace, engine: str, output_dir: Path
) -> tuple[list[str], dict[str, str]]:
    command = _engine_command(args, engine, output_dir)
    command.append("--preflight")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    if args.dependency_root is not None:
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(args.dependency_root), environment.get("PYTHONPATH", "")))
        )
    return command, environment


def _common_receipt(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    sources: dict[str, object] = {
        "comfyui": _source_receipt(args.comfyui_root),
        "dinkster": _source_receipt(args.dinkster_root),
    }
    artifact = _verify_checkpoint(args.checkpoint)
    return sources, artifact


def _run_preflight(args: argparse.Namespace) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ComparisonError("preflight must run with CUDA_VISIBLE_DEVICES empty")
    sources, artifact = _common_receipt(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    engines = {}
    for engine in ("comfyui", "dinkster"):
        command, environment = _preflight_command(args, engine, args.output_dir)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
            env=environment,
        )
        (args.output_dir / f"preflight-{engine}.stderr.txt").write_text(
            result.stderr, encoding="utf-8"
        )
        if result.returncode:
            raise ComparisonError(f"{engine} preflight failed: {result.stderr}")
        engines[engine] = json.loads(result.stdout)
        _write_json(
            args.output_dir / "preflight.partial.json",
            {
                "artifact": artifact,
                "engines": engines,
                "sources": sources,
                "workload": _workload(),
            },
        )
    receipt = {
        "artifact": artifact,
        "engines": engines,
        "schema": "dinkster.lumina2.matched-e2e.preflight.v1",
        "sources": sources,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "workload": _workload(),
    }
    _write_json(args.output_dir / "preflight.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _gpu_identity() -> dict[str, object]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={GPU_INDEX}",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ComparisonError(result.stderr)
    values = [value.strip() for value in result.stdout.strip().split(",")]
    if len(values) != 5 or values[2] != GPU_UUID:
        raise ComparisonError(f"physical GPU 2 identity mismatch: {values}")
    return {
        "driver": values[4],
        "index": int(values[0]),
        "memory_total_mib": int(values[3]),
        "name": values[1],
        "uuid": values[2],
    }


def _gpu_claim() -> dict[str, str]:
    metadata = GPU_CLAIM_PATH.read_text(encoding="utf-8").strip()
    valid_metadata = (
        f"thread={THREAD_ID}" in metadata and "purpose=" in metadata and "timestamp=" in metadata
    )
    if not valid_metadata:
        raise ComparisonError("GPU 2 claim metadata is missing or belongs to another thread")
    with GPU_CLAIM_PATH.open("r", encoding="utf-8") as claim:
        try:
            fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(claim, fcntl.LOCK_UN)
            raise ComparisonError("GPU 2 claim lock is not held by the launch harness")
    return {"metadata": metadata, "path": str(GPU_CLAIM_PATH)}


def _validate_preflight(
    path: Path, sources: dict[str, object], artifact: dict[str, object]
) -> dict[str, object]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ComparisonError("preflight receipt must be a JSON object")
    if receipt.get("sources") != sources or receipt.get("artifact") != artifact:
        raise ComparisonError("preflight source or artifact identity does not match this run")
    if receipt.get("workload") != _workload():
        raise ComparisonError("preflight workload does not match this run")
    timestamp = dt.datetime.fromisoformat(cast("str", receipt["timestamp_utc"]))
    age = dt.datetime.now(dt.UTC) - timestamp
    if age < dt.timedelta(0) or age > dt.timedelta(hours=1):
        raise ComparisonError("preflight receipt is not from the immediately preceding hour")
    return receipt


def _run_gate(args: argparse.Namespace) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU_INDEX):
        raise ComparisonError("run requires CUDA_VISIBLE_DEVICES=2")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ComparisonError("run output directory must be absent or empty")
    sources, artifact = _common_receipt(args)
    preflight = _validate_preflight(args.preflight_receipt, sources, artifact)
    hardware = _gpu_identity()
    claim = _gpu_claim()
    records = [
        _run_process(args, engine, index, args.output_dir)
        for index, engine in enumerate(PROCESS_ORDER, 1)
    ]
    verdict = _compare_records(records, sources, artifact)
    verdict["claim"] = claim
    verdict["hardware"] = hardware
    verdict["preflight"] = preflight
    _write_json(args.output_dir / "verdict.json", verdict)
    print(json.dumps(verdict, sort_keys=True))
    if not verdict["overall_pass"]:
        raise ComparisonError("matched Lumina2 end-to-end gate failed; see verdict.json")
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--dinkster-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    _add_common_arguments(preflight)
    run = subparsers.add_parser("run")
    _add_common_arguments(run)
    run.add_argument("--preflight-receipt", type=Path, required=True)
    engine = subparsers.add_parser("engine")
    engine.add_argument("--engine", choices=("comfyui", "dinkster"), required=True)
    engine.add_argument("--root", type=Path, required=True)
    engine.add_argument("--checkpoint", type=Path, required=True)
    engine.add_argument("--output-dir", type=Path, required=True)
    engine.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        return _run_preflight(args)
    if args.command == "run":
        return _run_gate(args)
    if args.preflight:
        return _engine_preflight(args)
    return _engine_comfyui(args) if args.engine == "comfyui" else _engine_dinkster(args)


if __name__ == "__main__":
    raise SystemExit(main())
