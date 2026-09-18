"""Persistent ComfyUI adapter for the canonical Ovis workload."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _configure_precision(comfy_args: Any) -> None:
    for flag in (
        "fp32_unet",
        "fp64_unet",
        "fp16_unet",
        "fp8_e4m3fn_unet",
        "fp8_e5m2_unet",
        "fp8_e8m0fnu_unet",
    ):
        setattr(comfy_args, flag, False)
    comfy_args.bf16_unet = True
    for flag in (
        "fp8_e4m3fn_text_enc",
        "fp8_e5m2_text_enc",
        "fp32_text_enc",
        "bf16_text_enc",
    ):
        setattr(comfy_args, flag, False)
    comfy_args.fp16_text_enc = True
    comfy_args.fp16_vae = False
    comfy_args.bf16_vae = True
    comfy_args.fp32_vae = False


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

    import folder_paths  # pyright: ignore[reportMissingImports]

    for category, key in (
        ("diffusion_models", "diffusion_model"),
        ("text_encoders", "text_encoder"),
        ("vae", "codec"),
    ):
        path = Path(workload[key])
        folder_paths.add_model_folder_path(category, str(path.parent), True)

    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]
    from comfy.cli_args import args as comfy_args  # pyright: ignore[reportMissingImports]

    _configure_precision(comfy_args)
    import nodes  # pyright: ignore[reportMissingImports]
    from comfy import model_management  # pyright: ignore[reportMissingImports]
    from comfy_extras.nodes_model_advanced import (  # pyright: ignore[reportMissingImports]
        ModelSamplingAuraFlow,
    )

    model = None
    positive = None
    negative = None
    latent = None
    vae = None
    for line in sys.stdin:
        request = json.loads(line)
        phase = request["phase"]
        load_ns = 0
        if model is None:
            start = time.perf_counter_ns()
            model = nodes.UNETLoader().load_unet(Path(workload["diffusion_model"]).name, "default")[
                0
            ]
            model = ModelSamplingAuraFlow().patch_aura(model, workload["model_sampling_shift"])[0]
            clip = nodes.CLIPLoader().load_clip(
                Path(workload["text_encoder"]).name, "ovis", "default"
            )[0]
            vae = nodes.VAELoader().load_vae(Path(workload["codec"]).name)[0]
            if vae.vae_dtype != torch.bfloat16:
                raise RuntimeError(f"expected BF16 VAE, got {vae.vae_dtype}")
            positive = nodes.CLIPTextEncode().encode(clip, workload["prompt"])[0]
            negative = nodes.CLIPTextEncode().encode(clip, workload["negative_prompt"])[0]
            latent = {
                "samples": torch.zeros(
                    (
                        workload["batch"],
                        16,
                        workload["height"] // 8,
                        workload["width"] // 8,
                    ),
                    dtype=torch.float32,
                ),
                "downscale_ratio_spacial": 8,
            }
            torch.cuda.synchronize()
            load_ns = time.perf_counter_ns() - start
        assert positive is not None and negative is not None and latent is not None
        assert vae is not None and model is not None
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        sampled = nodes.common_ksampler(
            model,
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
        memory_used = vae.memory_used_decode(sampled["samples"].shape, vae.vae_dtype)
        model_management.load_models_gpu(
            [vae.patcher],
            memory_required=memory_used,
            force_full_load=vae.disable_offload,
        )
        with torch.inference_mode():
            image = vae.decode_tiled_(sampled["samples"], tile_x=64, tile_y=64, overlap=16).movedim(
                1, -1
            )
        torch.cuda.synchronize()
        generation_ns = time.perf_counter_ns() - start
        latent_output = args.output_dir / f"comfyui-{phase}-latent.npy"
        image_output = args.output_dir / f"comfyui-{phase}-image.npy"
        np.save(
            latent_output,
            sampled["samples"].detach().float().cpu().numpy(),
            allow_pickle=False,
        )
        np.save(image_output, image.detach().float().cpu().numpy(), allow_pickle=False)
        print(
            json.dumps(
                {
                    "cold_load_ns": load_ns,
                    "decode_mode": "tiled",
                    "generation_ns": generation_ns,
                    "outputs": {"image": str(image_output), "latent": str(latent_output)},
                    "phase": phase,
                    "torch": torch.__version__,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
