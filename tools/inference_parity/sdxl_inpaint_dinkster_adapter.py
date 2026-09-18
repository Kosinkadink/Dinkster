"""Persistent native Dinkster adapter for the canonical SDXL inpaint workload."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.inference_parity.sd15_inpaint_dinkster_adapter import _prepare_inpaint_inputs


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
    for source in sorted((args.repo / "packages").glob("*/src")):
        sys.path.insert(0, str(source))

    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]
    from dinkster_inference import (
        InpaintConditioning,
        SamplingGuidance,
        load_safetensors_header,
        probe_native,
    )
    from dinkster_inference_torch import SDRuntime, load_runtime
    from PIL import Image, ImageOps

    if args.construct_only:
        source = load_safetensors_header(args.artifact_root / workload["checkpoint"])
        capability = probe_native(source)
        if not capability.native or capability.family_id != "dinkster.sdxl":
            raise RuntimeError(f"checkpoint was not accepted as SDXL: {capability}")

        class ConstructionRuntime:
            class Assembled:
                class VAE:
                    class Config:
                        spatial_downscale = 8

                    config = Config()

                vae = VAE()

            assembled = Assembled()

            @staticmethod
            def encode_content(content: Any) -> Any:
                return torch.zeros(
                    (content.shape[0], 4, content.shape[2] // 8, content.shape[3] // 8)
                )

        latent, noise_mask, _ = _prepare_inpaint_inputs(
            args.artifact_root / "input" / workload["input_image"],
            ConstructionRuntime(),
            torch,
            np,
            Image,
            ImageOps,
            megapixels=workload["megapixels"],
            grow_mask_by=workload["grow_mask_by"],
            device=torch.device("cpu"),
        )
        conditioning = InpaintConditioning(mask=noise_mask, masked_image=latent)
        print(
            f"family={capability.family_id}; latent={tuple(conditioning.masked_image.shape)}; "
            f"mask={tuple(conditioning.mask.shape)}",
            flush=True,
        )
        return 0

    runtime = None
    cond = None
    uncond = None
    latent = None
    noise_mask = None
    inpaint = None
    text_parameter_dtype = None
    device = torch.device("cuda:0")
    diffusion_dtype = torch.float16
    for line in sys.stdin:
        request = json.loads(line)
        phase = request["phase"]
        load_ns = 0
        if runtime is None:
            start = time.perf_counter_ns()
            runtime = cast(
                SDRuntime,
                load_runtime(
                    checkpoint=load_safetensors_header(args.artifact_root / workload["checkpoint"]),
                    diffusion_dtype=diffusion_dtype,
                    text_dtype=torch.float32,
                    vae_dtype=torch.float32,
                ),
            )
            runtime.assembled.diffusion.to(device=device, dtype=diffusion_dtype)
            assert runtime.assembled.clip_l is not None
            assert runtime.assembled.clip_g is not None
            runtime.assembled.clip_l.to(device)
            runtime.assembled.clip_g.to(device)
            text_dtypes = {
                str(next(runtime.assembled.clip_l.parameters()).dtype).removeprefix("torch."),
                str(next(runtime.assembled.clip_g.parameters()).dtype).removeprefix("torch."),
            }
            if text_dtypes != {"float32"}:
                raise RuntimeError(f"SDXL text parameter dtypes differ: {sorted(text_dtypes)}")
            text_parameter_dtype = text_dtypes.pop()
            runtime.assembled.vae.to(device)
            with torch.inference_mode():
                cond = runtime.encode_text(workload["prompt"])
                uncond = runtime.encode_text(workload["negative_prompt"])
                latent, noise_mask, _ = _prepare_inpaint_inputs(
                    args.artifact_root / "input" / workload["input_image"],
                    runtime,
                    torch,
                    np,
                    Image,
                    ImageOps,
                    megapixels=workload["megapixels"],
                    grow_mask_by=workload["grow_mask_by"],
                    device=device,
                )
                inpaint = InpaintConditioning(mask=noise_mask, masked_image=latent)
            torch.cuda.synchronize()
            load_ns = time.perf_counter_ns() - start
        assert runtime is not None and cond is not None and uncond is not None
        assert latent is not None and noise_mask is not None and inpaint is not None
        assert text_parameter_dtype is not None
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        with torch.inference_mode():
            sampled = runtime.sample(
                latent,
                cond=cond,
                cfg=SamplingGuidance(uncond, workload["cfg"]),
                sampler_id=workload["sampler"],
                scheduler_id=workload["scheduler"],
                steps=workload["steps"],
                denoise=workload["denoise"],
                seed=request["seed"],
                denoise_mask=noise_mask,
                inpaint=inpaint,
                compute_dtype=torch.float16,
                device=device,
            )
            image = runtime.decode_latent(sampled).permute(0, 2, 3, 1)
        torch.cuda.synchronize()
        generation_ns = time.perf_counter_ns() - start
        latent_output = args.output_dir / f"dinkster-{phase}-latent.npy"
        image_output = args.output_dir / f"dinkster-{phase}-image.npy"
        np.save(latent_output, sampled.detach().float().cpu().numpy(), allow_pickle=False)
        np.save(image_output, image.detach().float().cpu().numpy(), allow_pickle=False)
        print(
            json.dumps(
                {
                    "cold_load_ns": load_ns,
                    "decode_mode": "regular",
                    "generation_ns": generation_ns,
                    "outputs": {"image": str(image_output), "latent": str(latent_output)},
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
