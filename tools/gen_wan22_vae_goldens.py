"""Generate reduced Wan 2.2 VAE goldens from pinned ComfyUI 947c2749.

The source tree is materialized from the exact git object with ``git archive``;
the current ComfyUI checkout need not move. The reduced model retains every
official block type and the complete 16x spatial, 4x temporal topology.

Run from the Dinkster root with a torch interpreter that can import ComfyUI::

    python tools/gen_wan22_vae_goldens.py
    python tools/gen_wan22_vae_goldens.py --check
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
OUT = (
    REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "wan22_vae_goldens.json"
)

CONFIG: dict[str, Any] = {
    "dim": 2,
    "dec_dim": 2,
    "z_dim": 2,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 1,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
}


def _comfy_root() -> Path:
    configured = os.environ.get("DINKSTER_COMFYUI_ROOT")
    candidates = (
        Path(configured) if configured else None,
        REPO.parent / "ComfyUI",
        REPO.parent.parent / "ComfyUI",
    )
    for candidate in candidates:
        if candidate is not None and (candidate / ".git").exists():
            return candidate.resolve()
    raise SystemExit("set DINKSTER_COMFYUI_ROOT to a ComfyUI git checkout")


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def _tensor(value: Any) -> dict[str, Any]:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("golden values must be tensors")
    value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return {
        "shape": list(value.shape),
        "dtype": "float32",
        "data": value.flatten().tolist(),
    }


def _payload(source: Path) -> dict[str, Any]:
    loader = source / "comfy" / "sd.py"
    official_temporal_config = '"temperal_downsample": [False, True, True]'
    if official_temporal_config not in loader.read_text(encoding="utf-8"):
        raise RuntimeError("pinned ComfyUI loader does not contain the official Wan 2.2 VAE config")

    sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
    sys.path.insert(0, str(source))
    sys.argv[:] = [sys.argv[0], "--cpu"]

    import comfy.options
    import torch

    comfy.options.enable_args_parsing()

    import kl_fill
    from comfy.ldm.modules.diffusionmodules import model as attention_module
    from comfy.ldm.wan import vae as shared_vae_module
    from comfy.ldm.wan import vae2_2 as reference_module
    from comfy.ldm.wan.vae2_2 import WanVAE

    module_file = Path(reference_module.__file__ or "").resolve()
    if not module_file.is_relative_to(source):
        raise RuntimeError(f"Wan 2.2 VAE imported from {module_file}, not {source}")
    shared_vae_module.vae_attention = lambda: attention_module.pytorch_attention
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    model = WanVAE(**CONFIG)
    model.eval()
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    model.load_state_dict(kl_fill.fill_state_dict(entries), strict=True)

    content = torch.linspace(-0.9, 0.9, 1 * 3 * 5 * 16 * 16).reshape(1, 3, 5, 16, 16)
    latent = torch.linspace(-0.4, 0.4, 1 * 2 * 2).reshape(1, 2, 2, 1, 1)
    with torch.no_grad():
        encoded = model.encode(content)
        decoded = model.decode(latent)

    return {
        "reference": {
            "repo": "Comfy-Org/ComfyUI",
            "commit": REFERENCE_COMMIT,
            "file": "comfy/ldm/wan/vae2_2.py",
            "configuration_file": "comfy/sd.py",
            "torch": torch.__version__,
            "attention": "pytorch_attention",
        },
        "config": {
            "dim": CONFIG["dim"],
            "decoder_dim": CONFIG["dec_dim"],
            "z_dim": CONFIG["z_dim"],
            "dim_mult": CONFIG["dim_mult"],
            "num_res_blocks": CONFIG["num_res_blocks"],
            "attn_scales": CONFIG["attn_scales"],
            "temporal_downsample": CONFIG["temperal_downsample"],
            "image_channels": 3,
            "conv_out_channels": 3,
            "patch_size": 2,
            "dropout": CONFIG["dropout"],
        },
        "state_dict": entries,
        "content": _tensor(content),
        "encoded": _tensor(encoded),
        "latent": _tensor(latent),
        "decoded": _tensor(decoded),
    }


def _generate() -> bytes:
    comfy_root = _comfy_root()
    resolved = _git(comfy_root, "rev-parse", REFERENCE_COMMIT).decode().strip()
    if resolved != REFERENCE_COMMIT:
        raise SystemExit(f"ComfyUI does not contain {REFERENCE_COMMIT}")
    archive = _git(comfy_root, "archive", "--format=tar", REFERENCE_COMMIT)
    with tempfile.TemporaryDirectory(prefix="dinkster-wan22-vae-reference-") as directory:
        source = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as payload:
            payload.extractall(source, filter="data")
        result = _payload(source)
    return (json.dumps(result, indent=2) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = _generate()
    if args.check:
        if not OUT.is_file() or OUT.read_bytes() != generated:
            raise SystemExit(f"{OUT} is not stable; regenerate it")
        print(f"stable {OUT}")
        return
    OUT.write_bytes(generated)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
