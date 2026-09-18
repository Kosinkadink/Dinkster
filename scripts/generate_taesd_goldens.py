#!/usr/bin/env python3
"""Generate compact TAESD facade goldens with pinned ComfyUI.

Run with ``/home/kosin/ComfyUI/venv/bin/python
scripts/generate_taesd_goldens.py`` after installing the pinned reference's
``comfy-aimdo==0.4.10`` into ``../ComfyUI-goldenref/.venv``.
The script refuses a wrong or dirty checkout because both architecture and VAE
boundary transforms are the reference under test.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

PIN = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
COMFY = Path(__file__).resolve().parents[2] / "ComfyUI-goldenref"
WEIGHTS = Path("/tmp/dinkster-taesd-weights")
OUTPUT = (
    Path(__file__).parents[1] / "packages/dinkster-inference-torch/tests/goldens/taesd_goldens.json"
)
ARTIFACTS = {
    "taesd_decoder.pth": "02873377c3f4659cd9f9adb2f718dcb434ba4c7bba3af3ee7bb95cbdabe2d3cf",
    "taesd_encoder.pth": "15bc6128f0ac51c673d3427216082ebfa62402ffc89763af74edfe259b32d49d",
    "taesdxl_decoder.pth": "a3956b8a7a763f251c7357aad6375dacdb6d971121c73e97a3069fc33a94fe1e",
    "taesdxl_encoder.pth": "a5648de089aba6b641e505c0f132b2dc7ff70ab6070c31be5e624e807e711c5e",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(COMFY), *args], text=True).strip()


def compact(tensor: torch.Tensor) -> dict[str, object]:
    flat = tensor.detach().cpu().flatten()
    indexes = sorted({0, flat.numel() // 3, flat.numel() // 2, flat.numel() - 1})
    return {
        "shape": list(tensor.shape),
        "sum": float(flat.sum()),
        "samples": [[index, float(flat[index])] for index in indexes],
    }


def main() -> None:
    if git("rev-parse", "HEAD") != PIN or git("status", "--porcelain"):
        raise SystemExit(f"refusing: {COMFY} must be clean at {PIN}")
    for filename, expected in ARTIFACTS.items():
        observed = hashlib.sha256((WEIGHTS / filename).read_bytes()).hexdigest()
        if observed != expected:
            raise SystemExit(f"refusing: {filename} sha256 is {observed}, expected {expected}")
    reference_site = (
        COMFY
        / ".venv/lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    sys.path.insert(0, str(reference_site))
    sys.path.insert(0, str(COMFY))
    from comfy.sd import VAE

    content = torch.linspace(0.0, 1.0, 16 * 16 * 3, dtype=torch.float32).reshape(1, 16, 16, 3)
    result: dict[str, object] = {
        "_meta": {
            "artifacts": ARTIFACTS,
            "device": "cpu",
            "dtype": "float32",
            "reference_commit": PIN,
            "torch": torch.__version__,
        },
        "input": compact(content),
    }
    for family, stem in (("sd15", "taesd"), ("sdxl", "taesdxl")):
        encoder = torch.load(WEIGHTS / f"{stem}_encoder.pth", map_location="cpu", weights_only=True)
        decoder = torch.load(WEIGHTS / f"{stem}_decoder.pth", map_location="cpu", weights_only=True)
        state = {f"taesd_encoder.{key}": value for key, value in encoder.items()}
        state.update({f"taesd_decoder.{key}": value for key, value in decoder.items()})
        state["vae_scale"] = torch.tensor(0.18215 if family == "sd15" else 0.13025)
        state["vae_shift"] = torch.tensor(0.0)
        vae = VAE(sd=state, device=torch.device("cpu"), dtype=torch.float32)
        latent = vae.encode(content)
        decoded = vae.decode(latent)
        result[family] = {"encode": compact(latent), "decode": compact(decoded)}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
