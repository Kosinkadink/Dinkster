"""Persistent ComfyUI adapter for the SDXL GGUF parity workload."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _custom_package(repo: Path) -> Any:
    root = repo / "custom_nodes" / "ComfyUI-GGUF"
    spec = importlib.util.spec_from_file_location(
        "comfyui_gguf_reference", root / "__init__.py", submodule_search_locations=[str(root)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load custom node at {root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _regular_decode(vae: Any, latent: Any) -> Any:
    tiled_calls = 0
    original = vae.decode_tiled_

    def tracked(*args: Any, **kwargs: Any) -> Any:
        nonlocal tiled_calls
        tiled_calls += 1
        return original(*args, **kwargs)

    vae.decode_tiled_ = tracked
    try:
        image = vae.decode(latent)
    finally:
        vae.decode_tiled_ = original
    if tiled_calls:
        raise RuntimeError("regular GGUF workload decode fell back to tiled decode")
    return image


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _parameter_dtype(module: Any) -> str:
    dtypes = {_dtype_name(parameter.dtype) for parameter in module.parameters()}
    if len(dtypes) != 1:
        raise RuntimeError(f"expected one parameter dtype, got {sorted(dtypes)}")
    return dtypes.pop()


def _actual_dtypes(model: Any, clip: Any, vae: Any, dequant: str) -> dict[str, str]:
    return {
        "codec": _dtype_name(vae.vae_dtype),
        "dequant": dequant,
        "diffusion": _dtype_name(model.model_dtype()),
        "text": _parameter_dtype(clip.cond_stage_model),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workload-json", required=True)
    parser.add_argument("--reference-output", type=Path)
    args = parser.parse_args()
    workload = json.loads(args.workload_json)
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "" if args.reference_output else str(workload["device_ordinal"])
    )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(args.repo))
    with contextlib.redirect_stdout(sys.stderr):
        from comfy.cli_args import args as comfy_args  # pyright: ignore[reportMissingImports]

        comfy_args.highvram = False
        comfy_args.normalvram = not bool(args.reference_output)
        comfy_args.cpu = bool(args.reference_output)
        comfy_args.fp16_unet = not bool(args.reference_output)
        comfy_args.fp32_unet = bool(args.reference_output)
        comfy_args.fp32_text_enc = True
        comfy_args.fp32_vae = True
        comfy_args.use_pytorch_cross_attention = True
        import comfy.model_management as model_management  # pyright: ignore[reportMissingImports]
        import folder_paths  # pyright: ignore[reportMissingImports]
        import nodes  # pyright: ignore[reportMissingImports]
        import numpy as np
        import torch  # pyright: ignore[reportMissingImports]

        custom = _custom_package(args.repo)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        key: _path(args.artifact_root, workload[key])
        for key in ("checkpoint", "clip_l", "clip_g", "vae")
    }
    folder_paths.add_model_folder_path("diffusion_models", str(paths["checkpoint"].parent), True)
    folder_paths.add_model_folder_path("text_encoders", str(paths["clip_l"].parent), True)
    folder_paths.add_model_folder_path("vae", str(paths["vae"].parent), True)

    if args.reference_output:
        dequant_module: Any = sys.modules["comfyui_gguf_reference.dequant"]
        loader_module: Any = sys.modules["comfyui_gguf_reference.loader"]
        dequantize_tensor = dequant_module.dequantize_tensor
        gguf_sd_loader = loader_module.gguf_sd_loader

        state, metadata = gguf_sd_loader(str(paths["checkpoint"]))
        tensor_map = []
        digests = []
        for key, tensor in state.items():
            shape = list(tensor.tensor_shape)
            qtype = tensor.tensor_type.name
            tensor_map.append({"key": key, "type": qtype, "shape": shape})
            if qtype == "Q8_0":
                value = (
                    dequantize_tensor(tensor, dtype=torch.float32, dequant_dtype=torch.float32)
                    .cpu()
                    .contiguous()
                )
                digests.append(
                    {
                        "key": key,
                        "shape": shape,
                        "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                    }
                )
        model = custom.NODE_CLASS_MAPPINGS["UnetLoaderGGUFAdvanced"]().load_unet(
            paths["checkpoint"].name, "float32", "default", False
        )[0]
        clip = nodes.DualCLIPLoader().load_clip(paths["clip_l"].name, paths["clip_g"].name, "sdxl")[
            0
        ]
        vae = nodes.VAELoader().load_vae(paths["vae"].name)[0]
        reference = {**workload, **workload.get("reference", {})}
        positive = nodes.CLIPTextEncode().encode(clip, reference["prompt"])[0]
        negative = nodes.CLIPTextEncode().encode(clip, reference.get("negative_prompt", ""))[0]
        latent = nodes.EmptyLatentImage().generate(64, 64, 1)[0]
        with torch.inference_mode():
            sampled = nodes.common_ksampler(
                model,
                240,
                1,
                1.0,
                "euler",
                "simple",
                positive,
                negative,
                latent,
                denoise=reference.get("denoise", 1.0),
            )[0]
            image = _regular_decode(vae, sampled["samples"]).clamp(0, 1)
        image_path = args.reference_output.with_suffix(".image.npy")
        latent_path = args.reference_output.with_suffix(".latent.npy")
        np.save(image_path, image.cpu().float().numpy(), allow_pickle=False)
        np.save(latent_path, sampled["samples"].cpu().float().numpy(), allow_pickle=False)
        receipt = {
            "engine": "comfyui",
            "gguf": version("gguf"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "attention_backend": "pytorch-sdpa",
            "vram_state": str(model_management.vram_state),
            "actual_dtypes": _actual_dtypes(model, clip, vae, "float32"),
            "metadata": metadata,
            "image_path": str(image_path),
            "latent_path": str(latent_path),
            "q8_tensors": digests,
            "tensor_map": tensor_map,
        }
        args.reference_output.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        print(json.dumps(receipt, sort_keys=True), flush=True)
        return 0

    model = clip = vae = positive = negative = latent = None
    for line in sys.stdin:
        request = json.loads(line)
        load_ns = 0
        with contextlib.redirect_stdout(sys.stderr):
            if model is None:
                start = time.perf_counter_ns()
                model = custom.NODE_CLASS_MAPPINGS["UnetLoaderGGUFAdvanced"]().load_unet(
                    paths["checkpoint"].name, "float16", "default", False
                )[0]
                clip = nodes.DualCLIPLoader().load_clip(
                    paths["clip_l"].name, paths["clip_g"].name, "sdxl"
                )[0]
                vae = nodes.VAELoader().load_vae(paths["vae"].name)[0]
                positive = nodes.CLIPTextEncode().encode(clip, workload["prompt"])[0]
                negative = nodes.CLIPTextEncode().encode(clip, workload.get("negative_prompt", ""))[
                    0
                ]
                latent = nodes.EmptyLatentImage().generate(
                    workload["width"], workload["height"], workload["batch"]
                )[0]
                torch.cuda.synchronize()
                load_ns = time.perf_counter_ns() - start
            assert vae is not None
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            with torch.inference_mode():
                sampled = nodes.common_ksampler(
                    model,
                    request["seed"],
                    workload["steps"],
                    request.get("cfg", workload["cfg"]),
                    workload["sampler"],
                    workload["scheduler"],
                    positive,
                    negative,
                    latent,
                    denoise=workload["denoise"],
                )[0]
            torch.cuda.synchronize()
            sampling_ns = time.perf_counter_ns() - start
            start = time.perf_counter_ns()
            with torch.inference_mode():
                image = _regular_decode(vae, sampled["samples"]).clamp(0, 1)
            torch.cuda.synchronize()
            decode_ns = time.perf_counter_ns() - start
        output = args.output_dir / f"comfyui-{request['phase']}.npy"
        np.save(output, image.cpu().float().numpy(), allow_pickle=False)
        receipt = {
            "phase": request["phase"],
            "cold_load_ns": load_ns,
            "generation_ns": sampling_ns + decode_ns,
            "sampling_ns": sampling_ns,
            "decode_ns": decode_ns,
            "decode_mode": "regular",
            "output_path": str(output),
            "torch": torch.__version__,
            "gguf": version("gguf"),
            "python": platform.python_version(),
            "attention_backend": "pytorch-sdpa",
            "actual_dtypes": _actual_dtypes(model, clip, vae, "float16"),
            "memory_flags": {
                "cpu": comfy_args.cpu,
                "highvram": comfy_args.highvram,
                "normalvram": comfy_args.normalvram,
            },
            "memory_policy": str(model_management.vram_state),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
