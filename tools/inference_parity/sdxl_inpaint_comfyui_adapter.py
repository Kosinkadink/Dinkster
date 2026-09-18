"""Persistent ComfyUI adapter for the canonical SDXL inpaint workload."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.inference_parity.comfyui_adapter import (
    _normal_sigmas_on_cpu,
    _set_text_precision,
    _set_vae_precision,
)
from tools.inference_parity.ovis_decode_calibration import (
    _regular_decode_without_fallback,
)


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
        "" if args.construct_only else workload["hardware"]["device_uuid"]
    )
    sys.path.insert(0, str(args.repo))

    from comfy.cli_args import args as comfy_args  # pyright: ignore[reportMissingImports]

    if args.construct_only:
        comfy_args.cpu = True
    import comfy.samplers  # pyright: ignore[reportMissingImports]
    import comfy.utils  # pyright: ignore[reportMissingImports]
    import folder_paths  # pyright: ignore[reportMissingImports]
    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]

    _set_vae_precision(workload, comfy_args)
    _set_text_precision(workload, comfy_args)
    import nodes  # pyright: ignore[reportMissingImports]

    comfy.samplers.calculate_sigmas = _normal_sigmas_on_cpu(
        comfy.samplers.calculate_sigmas, torch.device("cpu")
    )
    checkpoint_path = args.artifact_root / workload["checkpoint"]
    folder_paths.add_model_folder_path("checkpoints", str(checkpoint_path.parent), True)
    folder_paths.set_input_directory(str(args.artifact_root / "input"))
    if args.construct_only:
        print("comfy_args.cpu=True; nodes imported", flush=True)
        return 0

    checkpoint = None
    positive = None
    negative = None
    latent = None
    vae = None
    text_parameter_dtype = None
    for line in sys.stdin:
        with torch.inference_mode():
            request = json.loads(line)
            phase = request["phase"]
            load_ns = 0
            if checkpoint is None:
                start = time.perf_counter_ns()
                checkpoint = nodes.CheckpointLoaderSimple().load_checkpoint(checkpoint_path.name)
                model, clip, vae = checkpoint
                text_dtypes = {
                    str(parameter.dtype).removeprefix("torch.")
                    for parameter in clip.cond_stage_model.parameters()
                }
                if text_dtypes != {"float32"}:
                    raise RuntimeError(f"SDXL text parameter dtypes differ: {sorted(text_dtypes)}")
                text_parameter_dtype = text_dtypes.pop()
                positive = nodes.CLIPTextEncode().encode(clip, workload["prompt"])[0]
                negative = nodes.CLIPTextEncode().encode(clip, workload["negative_prompt"])[0]
                pixels, mask = nodes.LoadImage().load_image(workload["input_image"])
                samples = pixels.movedim(-1, 1)
                total = workload["megapixels"] * 1024 * 1024
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                scaled = comfy.utils.common_upscale(
                    samples, width, height, workload["upscale_method"], "disabled"
                ).movedim(1, -1)
                latent = nodes.VAEEncodeForInpaint().encode(
                    vae, scaled, mask, workload["grow_mask_by"]
                )[0]
                torch.cuda.synchronize()
                load_ns = time.perf_counter_ns() - start
            assert checkpoint is not None and positive is not None and negative is not None
            assert latent is not None and vae is not None
            assert text_parameter_dtype is not None
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
                        "decode_mode": "regular",
                        "generation_ns": generation_ns,
                        "outputs": {
                            "image": str(image_output),
                            "latent": str(latent_output),
                        },
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
