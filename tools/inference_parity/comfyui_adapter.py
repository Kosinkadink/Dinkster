"""Persistent ComfyUI adapter for the canonical SD1.5 workload."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _set_vae_precision(workload: dict[str, Any], comfy_args: Any) -> None:
    precision = workload.get("precision")
    if not isinstance(precision, dict):
        raise ValueError("precision must be an object")
    codec_precision = precision.get("codec")
    vae_precision_flags = {
        "float32": "fp32_vae",
        "bfloat16": "bf16_vae",
        "float16": "fp16_vae",
    }
    if codec_precision not in vae_precision_flags:
        supported = ", ".join(sorted(vae_precision_flags))
        raise ValueError(f"precision.codec must be one of {supported}, got {codec_precision!r}")
    selected_flag = vae_precision_flags[codec_precision]
    for flag in vae_precision_flags.values():
        setattr(comfy_args, flag, flag == selected_flag)


def _set_text_precision(workload: dict[str, Any], comfy_args: Any) -> None:
    precision = workload.get("precision")
    if not isinstance(precision, dict):
        raise ValueError("precision must be an object")
    text_precision = precision.get("text")
    text_precision_flags = {
        "float32": "fp32_text_enc",
        "bfloat16": "bf16_text_enc",
        "float16": "fp16_text_enc",
    }
    if text_precision not in text_precision_flags:
        supported = ", ".join(sorted(text_precision_flags))
        raise ValueError(f"precision.text must be one of {supported}, got {text_precision!r}")
    selected_flag = text_precision_flags[text_precision]
    for flag in text_precision_flags.values():
        setattr(comfy_args, flag, flag == selected_flag)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workload-json", required=True)
    args = parser.parse_args()
    workload = json.loads(args.workload_json)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(workload["device_ordinal"])
    sys.path.insert(0, str(args.repo))

    import comfy.samplers  # pyright: ignore[reportMissingImports]
    import folder_paths  # pyright: ignore[reportMissingImports]
    import nodes  # pyright: ignore[reportMissingImports]
    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]
    from comfy.cli_args import args as comfy_args  # pyright: ignore[reportMissingImports]

    _set_text_precision(workload, comfy_args)
    _set_vae_precision(workload, comfy_args)
    comfy.samplers.calculate_sigmas = _normal_sigmas_on_cpu(
        comfy.samplers.calculate_sigmas, torch.device("cpu")
    )

    checkpoint = None
    positive = None
    negative = None
    latent = None
    vae = None
    text_parameter_dtype = None
    for line in sys.stdin:
        request = json.loads(line)
        phase = request["phase"]
        load_ns = 0
        if checkpoint is None:
            start = time.perf_counter_ns()
            checkpoint_path = args.artifact_root / workload["checkpoint"]
            registered_path = Path(
                folder_paths.get_full_path_or_raise("checkpoints", checkpoint_path.name)
            )
            if not checkpoint_path.samefile(registered_path):
                raise RuntimeError(
                    f"pinned checkpoint {checkpoint_path} is not ComfyUI's {registered_path}"
                )
            checkpoint = nodes.CheckpointLoaderSimple().load_checkpoint(checkpoint_path.name)
            model, clip, vae = checkpoint
            text_parameter_dtype = str(next(clip.cond_stage_model.parameters()).dtype).removeprefix(
                "torch."
            )
            positive = nodes.CLIPTextEncode().encode(clip, workload["prompt"])[0]
            negative = nodes.CLIPTextEncode().encode(clip, workload["negative_prompt"])[0]
            latent = nodes.EmptyLatentImage().generate(
                workload["width"], workload["height"], workload["batch"]
            )[0]
            torch.cuda.synchronize()
            load_ns = time.perf_counter_ns() - start
        assert (
            positive is not None
            and negative is not None
            and latent is not None
            and vae is not None
            and text_parameter_dtype is not None
        )
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
        image = nodes.VAEDecode().decode(vae, sampled)[0]
        torch.cuda.synchronize()
        generation_ns = time.perf_counter_ns() - start
        output = args.output_dir / f"comfyui-{phase}.npy"
        np.save(output, image.detach().float().cpu().numpy(), allow_pickle=False)
        print(
            json.dumps(
                {
                    "cold_load_ns": load_ns,
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
