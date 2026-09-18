"""Persistent native Dinkster adapter for the canonical SD1.5 inpaint workload."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, cast


def _prepare_inpaint_inputs(
    image_path: Path,
    runtime: Any,
    torch: Any,
    np: Any,
    image_api: Any,
    image_ops: Any,
    *,
    megapixels: float,
    grow_mask_by: int,
    device: Any,
) -> tuple[Any, Any, Any]:
    """Transcribe the pinned ComfyUI image, scale, and inpaint nodes."""
    # LoadImage @ f4b99bc, nodes.py:1713-1759: EXIF transpose, RGB
    # float32/255 pixels, and mask = 1 - alpha.
    opened = image_ops.exif_transpose(image_api.open(image_path))
    pixels_np = np.array(opened.convert("RGB")).astype(np.float32) / 255.0
    pixels = torch.from_numpy(pixels_np)[None,]
    if "A" in opened.getbands():
        alpha = np.array(opened.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(alpha).unsqueeze(0)
    else:
        mask = torch.zeros((1, 64, 64), dtype=torch.float32)

    # ImageScaleToTotalPixels @ f4b99bc,
    # comfy_extras/nodes_post_processing.py:235-246; nearest-exact is
    # torch interpolate through comfy/utils.py:1003-1029.
    samples = pixels.movedim(-1, 1)
    total = megapixels * 1024 * 1024
    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    scaled = torch.nn.functional.interpolate(
        samples, size=(height, width), mode="nearest-exact"
    ).movedim(1, -1)

    # VAEEncodeForInpaint @ f4b99bc, nodes.py:393-422: bilinear mask
    # resize, ceil-padded all-ones convolution on the rounded mask,
    # neutral-gray masked pixels before regular VAE encode, and the
    # rounded grown noise mask.
    downscale = runtime.assembled.vae.config.spatial_downscale
    x = (scaled.shape[1] // downscale) * downscale
    y = (scaled.shape[2] // downscale) * downscale
    mask = torch.nn.functional.interpolate(
        mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
        size=(scaled.shape[1], scaled.shape[2]),
        mode="bilinear",
        align_corners=False,
    )
    scaled = scaled.clone()
    if scaled.shape[1] != x or scaled.shape[2] != y:
        x_offset = (scaled.shape[1] % downscale) // 2
        y_offset = (scaled.shape[2] % downscale) // 2
        scaled = scaled[:, x_offset : x + x_offset, y_offset : y + y_offset, :]
        mask = mask[:, :, x_offset : x + x_offset, y_offset : y + y_offset]
    if grow_mask_by == 0:
        mask_erosion = mask
    else:
        kernel = torch.ones((1, 1, grow_mask_by, grow_mask_by))
        padding = math.ceil((grow_mask_by - 1) / 2)
        mask_erosion = torch.clamp(
            torch.nn.functional.conv2d(mask.round(), kernel, padding=padding), 0, 1
        )
    keep = (1.0 - mask.round()).squeeze(1)
    for channel in range(3):
        scaled[:, :, :, channel] -= 0.5
        scaled[:, :, :, channel] *= keep
        scaled[:, :, :, channel] += 0.5
    latent = runtime.encode_content(scaled.movedim(-1, 1).to(device))
    noise_mask = mask_erosion[:, :, :x, :y].round().to(device)
    return latent, noise_mask, scaled


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
        if not capability.native or capability.family_id != "dinkster.sd15":
            raise RuntimeError(f"converted checkpoint was not accepted: {capability}")

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
            runtime.assembled.diffusion.to(
                device=device,
                dtype=diffusion_dtype,
            )
            assert runtime.assembled.clip_l is not None
            runtime.assembled.clip_l.to(device)
            parameter = next(runtime.assembled.clip_l.parameters())
            text_parameter_dtype = str(parameter.dtype).removeprefix("torch.")
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
        output = args.output_dir / f"dinkster-{phase}.npy"
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
