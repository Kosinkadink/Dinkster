"""Persistent ComfyUI adapter for the canonical SDXL v-pred workload."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _configure_precision(workload: dict[str, Any], comfy_args: Any) -> None:
    precision = workload.get("precision")
    if not isinstance(precision, dict):
        raise ValueError("precision must be an object")
    groups = (
        (
            "diffusion",
            {
                "float32": "fp32_unet",
                "bfloat16": "bf16_unet",
                "float16": "fp16_unet",
            },
        ),
        (
            "text",
            {
                "float32": "fp32_text_enc",
                "bfloat16": "bf16_text_enc",
                "float16": "fp16_text_enc",
            },
        ),
        (
            "codec",
            {
                "float32": "fp32_vae",
                "bfloat16": "bf16_vae",
                "float16": "fp16_vae",
            },
        ),
    )
    for key, flags in groups:
        selected = precision.get(key)
        if selected not in flags:
            supported = ", ".join(sorted(flags))
            raise ValueError(f"precision.{key} must be one of {supported}, got {selected!r}")
        for flag in flags.values():
            setattr(comfy_args, flag, flag == flags[selected])


def _normal_sigmas_on_cpu(calculate_sigmas: Any, cpu_device: Any) -> Any:
    def calibrated_calculate_sigmas(model_sampling: Any, scheduler_name: str, steps: int) -> Any:
        if scheduler_name != "normal":
            return calculate_sigmas(model_sampling, scheduler_name, steps)
        original_device = model_sampling.sigmas.device
        try:
            model_sampling.to(cpu_device)
            return calculate_sigmas(model_sampling, scheduler_name, steps)
        finally:
            model_sampling.to(original_device)

    return calibrated_calculate_sigmas


def _regular_decode_without_fallback(vae: Any, latent: Any, inference_mode: Any) -> Any:
    tiled_calls = 0
    original_decode_tiled = vae.decode_tiled_

    def tracked_decode_tiled(
        *decode_args: Any,
        _decode: Any = original_decode_tiled,
        **decode_kwargs: Any,
    ) -> Any:
        nonlocal tiled_calls
        tiled_calls += 1
        return _decode(*decode_args, **decode_kwargs)

    vae.decode_tiled_ = tracked_decode_tiled
    try:
        with inference_mode():
            image = vae.decode(latent)
    finally:
        vae.decode_tiled_ = original_decode_tiled
    if tiled_calls:
        raise RuntimeError("regular workload decode fell back to tiled decode")
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workload-json", required=True)
    parser.add_argument("--construct-only", action="store_true")
    args = parser.parse_args()
    workload = json.loads(args.workload_json)
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "" if args.construct_only else str(workload["device_ordinal"])
    )
    sys.path.insert(0, str(args.repo))

    from comfy.cli_args import args as comfy_args  # pyright: ignore[reportMissingImports]

    if args.construct_only:
        comfy_args.cpu = True
    _configure_precision(workload, comfy_args)

    import comfy.samplers  # pyright: ignore[reportMissingImports]
    import folder_paths  # pyright: ignore[reportMissingImports]
    import nodes  # pyright: ignore[reportMissingImports]
    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]

    comfy.samplers.calculate_sigmas = _normal_sigmas_on_cpu(
        comfy.samplers.calculate_sigmas, torch.device("cpu")
    )
    checkpoint_path = args.artifact_root / workload["checkpoint"]
    folder_paths.add_model_folder_path("checkpoints", str(checkpoint_path.parent), True)
    if args.construct_only:
        print(
            "READY comfyui cpu=True; nodes imported; "
            f"cuda_initialized={torch.cuda.is_initialized()}",
            flush=True,
        )
        return 0

    checkpoint = None
    positive = None
    negative = None
    latent = None
    vae = None
    text_parameter_dtype = None
    for line in sys.stdin:
        # ComfyUI's server executor wraps node calls in inference_mode
        # (execution.py:709 @ f4b99bc). Keep load, encode, sample, and
        # decode under the same environment.
        with torch.inference_mode():
            request = json.loads(line)
            phase = request["phase"]
            load_ns = 0
            if checkpoint is None:
                start = time.perf_counter_ns()
                checkpoint = nodes.CheckpointLoaderSimple().load_checkpoint(checkpoint_path.name)
                model, clip, vae = checkpoint
                text_parameter_dtype = str(
                    next(clip.cond_stage_model.parameters()).dtype
                ).removeprefix("torch.")
                positive = nodes.CLIPTextEncode().encode(clip, workload["prompt"])[0]
                negative = nodes.CLIPTextEncode().encode(clip, workload["negative_prompt"])[0]
                latent = nodes.EmptyLatentImage().generate(
                    workload["width"], workload["height"], workload["batch"]
                )[0]
                torch.cuda.synchronize()
                load_ns = time.perf_counter_ns() - start
            assert checkpoint is not None and positive is not None and negative is not None
            assert latent is not None and vae is not None and text_parameter_dtype is not None
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            sampled = nodes.common_ksampler(
                checkpoint[0],
                request["seed"],
                workload["steps"],
                workload["cfg"],
                workload["sampler"],
                workload["scheduler"],
                positive,
                negative,
                latent,
                denoise=workload["denoise"],
            )[0]
            image = _regular_decode_without_fallback(vae, sampled["samples"], torch.inference_mode)
            torch.cuda.synchronize()
            generation_ns = time.perf_counter_ns() - start
            output = args.output_dir / f"comfyui-{phase}.npy"
            np.save(output, image.detach().float().cpu().numpy(), allow_pickle=False)
            print(
                json.dumps(
                    {
                        "cold_load_ns": load_ns,
                        "decode_mode": "regular",
                        "generation_ns": generation_ns,
                        "output_path": str(output),
                        "phase": phase,
                        "text_parameter_dtype": text_parameter_dtype,
                        "torch": torch.__version__,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
