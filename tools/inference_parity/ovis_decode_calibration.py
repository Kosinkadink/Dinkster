"""Calibrate Ovis regular-versus-tiled ComfyUI VAE decode divergence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.inference_parity.harness import digest_file, image_ssim, write_json


def _regular_decode_without_fallback(vae: Any, latent: Any, inference_mode: Any) -> Any:
    regular_tiled_calls = 0
    original_decode_tiled = vae.decode_tiled_

    def tracked_decode_tiled(
        *decode_args: Any,
        _decode=original_decode_tiled,
        **decode_kwargs: Any,
    ) -> Any:
        nonlocal regular_tiled_calls
        regular_tiled_calls += 1
        return _decode(*decode_args, **decode_kwargs)

    vae.decode_tiled_ = tracked_decode_tiled
    try:
        with inference_mode():
            regular = vae.decode(latent)
    finally:
        vae.decode_tiled_ = original_decode_tiled
    if regular_tiled_calls:
        raise RuntimeError("regular calibration decode fell back to tiled decode")
    return regular


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-ordinal", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_ordinal)
    sys.path.insert(0, str(args.comfyui_root))

    import folder_paths  # pyright: ignore[reportMissingImports]

    folder_paths.add_model_folder_path("vae", str(args.codec.parent), True)

    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]
    from comfy.cli_args import args as comfy_args  # pyright: ignore[reportMissingImports]

    comfy_args.fp16_vae = False
    comfy_args.bf16_vae = True
    comfy_args.fp32_vae = False
    import nodes  # pyright: ignore[reportMissingImports]

    vae = nodes.VAELoader().load_vae(args.codec.name)[0]
    if vae.vae_dtype != torch.bfloat16:
        raise RuntimeError(f"expected BF16 VAE, got {vae.vae_dtype}")
    latent = torch.from_numpy(np.load(args.latent, allow_pickle=False))
    regular = _regular_decode_without_fallback(vae, latent, torch.inference_mode)
    with torch.inference_mode():
        tiled = vae.decode_tiled(latent)
    regular_array = regular.detach().float().cpu().numpy()
    tiled_array = tiled.detach().float().cpu().numpy()
    delta = np.abs(regular_array.astype(np.float64) - tiled_array.astype(np.float64))
    result = {
        "codec": {"digest": digest_file(args.codec), "path": str(args.codec)},
        "comparator": "image-max-mean-ssim/1",
        "latent": {"digest": digest_file(args.latent), "path": str(args.latent)},
        "method": "ComfyUI VAE-only regular decode versus forced tiled decode",
        "observed": {
            "max_abs": format(float(delta.max(initial=0)), ".12g"),
            "mean_abs": format(float(delta.mean()), ".12g"),
            "ssim": format(image_ssim(regular_array, tiled_array), ".12g"),
        },
        "regular_decode_mode": "regular",
        "torch": torch.__version__,
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
