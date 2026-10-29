r"""Generate official-weight bfloat16 VAE decode goldens in pinned ComfyUI.

Usage::

    DINKSTER_VAE_DTYPE_MODELS=/path/to/vae-dtype-parity \
        .venv-gpu/bin/python tools/gen_vae_dtype_goldens.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

COMFY_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
ARTIFACTS = {
    "sdxl": (
        "sdxl/sd_xl_base_1.0.safetensors",
        6_938_078_334,
        "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b",
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/"
        "462165984030d82259a11f4367a4eed129e94a7b/sd_xl_base_1.0.safetensors",
        (1, 4, 4, 4),
        "first_stage_model.",
    ),
    "z_image": (
        "zimage/ae.safetensors",
        335_304_388,
        "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/"
        "08d04455279082882deaabc8d0d09fc914c071e1/split_files/vae/ae.safetensors",
        (1, 16, 4, 4),
        "",
    ),
    "wan21": (
        "wan21/wan_2.1_vae.safetensors",
        253_815_318,
        "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/"
        "617a7633e636506f850e043bc4605f290a466a8e/split_files/vae/"
        "wan_2.1_vae.safetensors",
        (1, 16, 1, 2, 2),
        "",
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _comfy_root() -> Path:
    configured = os.environ.get("DINKSTER_COMFYUI_ROOT")
    candidates = (
        *((Path(configured).resolve(),) if configured else ()),
        _repo_root().parent / "ComfyUI",
        _repo_root().parent.parent / "ComfyUI",
    )
    for candidate in candidates:
        if (candidate / ".git").is_dir():
            return candidate
    raise RuntimeError("set DINKSTER_COMFYUI_ROOT to a ComfyUI checkout")


def _model_root() -> Path:
    configured = os.environ.get("DINKSTER_VAE_DTYPE_MODELS")
    if configured is None:
        raise RuntimeError("set DINKSTER_VAE_DTYPE_MODELS to the pinned artifact directory")
    root = Path(configured).resolve()
    if not all((root / values[0]).is_file() for values in ARTIFACTS.values()):
        raise RuntimeError("DINKSTER_VAE_DTYPE_MODELS is missing one or more pinned artifacts")
    return root


def _export(repo: Path, destination: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", COMFY_COMMIT], text=True
    ).strip()
    if revision != COMFY_COMMIT:
        raise RuntimeError(f"resolved ComfyUI revision {revision}, expected {COMFY_COMMIT}")
    destination.mkdir()
    archive = destination / "comfy.tar"
    subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", "-o", str(archive), COMFY_COMMIT],
        check=True,
    )
    subprocess.run(["tar", "-xf", str(archive), "-C", str(destination)], check=True)
    archive.unlink()


def _artifact_facts(root: Path) -> dict[str, dict[str, Any]]:
    facts = {}
    for name, (relative, size, digest, url, _shape, _prefix) in ARTIFACTS.items():
        path = root / relative
        if path.stat().st_size != size:
            raise RuntimeError(f"{name}: expected {size} bytes, found {path.stat().st_size}")
        with path.open("rb") as artifact:
            actual = hashlib.file_digest(artifact, "sha256").hexdigest()
        if actual != digest:
            raise RuntimeError(f"{name}: sha256 {actual}, expected {digest}")
        facts[name] = {"url": url, "byte_size": size, "sha256": digest}
    return facts


def _tensor(value: Any) -> dict[str, Any]:
    value = value.detach().float().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": "float32",
        "data": value.flatten().tolist(),
    }


def _run(source: Path, models: Path, artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sys.path.insert(0, str(source))
    import comfy.model_management as model_management
    import comfy.sd as comfy_sd
    import torch
    from safetensors import safe_open

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cases = {}
    for name, (relative, _size, _digest, _url, shape, prefix) in ARTIFACTS.items():
        state = {}
        with safe_open(models / relative, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key.startswith(prefix):
                    state[key.removeprefix(prefix)] = checkpoint.get_tensor(key)
        vae = comfy_sd.VAE(sd=state, device=torch.device("cuda:0"))
        if vae.vae_dtype is not torch.bfloat16:
            raise RuntimeError(f"{name}: ComfyUI selected {vae.vae_dtype}, expected bfloat16")
        latent = torch.linspace(-0.5, 0.5, int(torch.tensor(shape).prod())).reshape(shape)
        with torch.no_grad():
            decoded = vae.decode(latent).movedim(-1, 1)
        cases[name] = {"latent": _tensor(latent), "decoded": _tensor(decoded)}
        del decoded, latent, state, vae
        model_management.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "reference": {
            "repository": "https://github.com/Comfy-Org/ComfyUI",
            "commit": COMFY_COMMIT,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvidia_driver": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                    "-i",
                    "0",
                ],
                text=True,
            ).strip(),
            "host": platform.node(),
            "device": torch.cuda.get_device_name(0),
            "vae_dtype": "bfloat16",
            "output_dtype": "float32",
        },
        "artifacts": artifacts,
        "cases": cases,
    }


def main() -> None:
    models = _model_root()
    artifacts = _artifact_facts(models)
    with tempfile.TemporaryDirectory(prefix="dinkster-vae-dtype-") as directory:
        source = Path(directory) / "ComfyUI"
        _export(_comfy_root(), source)
        payload = _run(source, models, artifacts)
    output = (
        _repo_root()
        / "packages"
        / "dinkster-inference-torch"
        / "tests"
        / "goldens"
        / "vae_dtype_goldens.json"
    )
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
