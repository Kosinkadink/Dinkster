r"""Run the official Wan 2.1 1.3B pipeline in pinned ComfyUI.

This is the same-checkpoint acceptance oracle required by ``PORTING.md``.
It exports ComfyUI commit b78cec87 before importing it, runs the immutable
Comfy-Org Wan 2.1 artifacts on CUDA, and records the full small-shape outputs
consumed by Dinkster's capability-gated GPU test.

Usage::

    lanes/comfyui-krea2-parity/.venv/bin/python tools/gen_wan21_official_goldens.py
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
COMFY_KITCHEN_REF = "v0.2.31"
COMFY_AIMDO_REF = "v0.4.13"
ARTIFACT_REPOSITORY = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
ARTIFACT_REVISION = "617a7633e636506f850e043bc4605f290a466a8e"
ARTIFACTS = {
    "diffusion": (
        "diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
        2_838_303_560,
        "be531024cd9018cb5b48c40cfbb6a6191645b1c792eb8bf4f8c1c6e10f924dc5",
    ),
    "text_encoder": (
        "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        6_735_906_897,
        "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
    ),
    "vae": (
        "vae/wan_2.1_vae.safetensors",
        253_815_318,
        "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
    ),
}
PROMPT = "a red fox running through snow"
NEGATIVE_PROMPT = ""
SEED = 7


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
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("set DINKSTER_COMFYUI_ROOT to a ComfyUI checkout")


def _model_root() -> Path:
    configured = os.environ.get("DINKSTER_WAN21_MODELS")
    candidates = (
        *((Path(configured).resolve(),) if configured else ()),
        _repo_root().parents[1] / "installs" / "ComfyUI" / "models",
    )
    for candidate in candidates:
        if all((candidate / values[0]).is_file() for values in ARTIFACTS.values()):
            return candidate
    raise RuntimeError("set DINKSTER_WAN21_MODELS to the official split-model root")


def _export(repo: Path, destination: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", COMFY_COMMIT], text=True
    ).strip()
    if revision != COMFY_COMMIT:
        raise RuntimeError(f"resolved ComfyUI revision {revision}, expected {COMFY_COMMIT}")
    destination.mkdir()
    _export_ref(repo, COMFY_COMMIT, destination)


def _export_ref(repo: Path, ref: str, destination: Path) -> None:
    archive = destination / "comfy.tar"
    subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", "-o", str(archive), ref],
        check=True,
    )
    subprocess.run(["tar", "-xf", str(archive), "-C", str(destination)], check=True)
    archive.unlink()


def _artifact_facts(root: Path) -> dict[str, dict[str, Any]]:
    facts: dict[str, dict[str, Any]] = {}
    for role, (relative, expected_size, expected_digest) in ARTIFACTS.items():
        path = root / relative
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"{role} has the wrong byte size")
        with path.open("rb") as artifact:
            digest = hashlib.file_digest(artifact, "sha256").hexdigest()
        if digest != expected_digest:
            raise RuntimeError(f"{role} has the wrong SHA256")
        facts[role] = {
            "url": (
                f"https://huggingface.co/{ARTIFACT_REPOSITORY}/resolve/{ARTIFACT_REVISION}/"
                f"split_files/{relative}"
            ),
            "byte_size": expected_size,
            "sha256": expected_digest,
        }
    return facts


def _tensor(value: Any) -> dict[str, Any]:
    value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return {
        "shape": list(value.shape),
        "dtype": "float32",
        "data": value.flatten().tolist(),
    }


def _release() -> None:
    model_management.unload_all_models()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return result.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _run(
    source: Path,
    kitchen: Path,
    aimdo: Path,
    models: Path,
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sys.path[:0] = [str(source), str(kitchen), str(aimdo)]
    sys.argv = [
        sys.argv[0],
        "--highvram",
        "--disable-xformers",
        "--use-pytorch-cross-attention",
        "--bf16-unet",
        "--fp32-text-enc",
    ]

    import comfy.options

    comfy.options.enable_args_parsing()

    global torch, comfy_sample, comfy_samplers, comfy_sd, model_management
    import torch
    from comfy import model_management
    from comfy import sample as comfy_sample
    from comfy import samplers as comfy_samplers
    from comfy import sd as comfy_sd

    if not torch.cuda.is_available():
        raise RuntimeError("official Wan 2.1 acceptance requires CUDA")
    torch.cuda.reset_peak_memory_stats()

    text_path = models / ARTIFACTS["text_encoder"][0]
    clip = comfy_sd.load_clip([str(text_path)], clip_type=comfy_sd.CLIPType.WAN)
    positive_tokens = clip.tokenize(PROMPT)
    negative_tokens = clip.tokenize(NEGATIVE_PROMPT)
    prompt_ids = [pair[0] for pair in positive_tokens["umt5xxl"][0]]
    negative_ids = [pair[0] for pair in negative_tokens["umt5xxl"][0]]
    with torch.no_grad():
        positive_cond, positive_pooled = clip.encode_from_tokens(
            positive_tokens, return_pooled=True
        )
        negative_cond, negative_pooled = clip.encode_from_tokens(
            negative_tokens, return_pooled=True
        )
    if positive_pooled is not None or negative_pooled is not None:
        raise RuntimeError("Wan text conditioning unexpectedly produced pooled output")
    positive_active = _tensor(positive_cond[:, : sum(token != 0 for token in prompt_ids)])
    negative_active = _tensor(negative_cond[:, : sum(token != 0 for token in negative_ids)])
    positive = [[positive_cond, {"pooled_output": None}]]
    negative = [[negative_cond, {"pooled_output": None}]]
    del clip
    _release()

    diffusion_path = models / ARTIFACTS["diffusion"][0]
    diffusion = comfy_sd.load_diffusion_model(
        str(diffusion_path),
        model_options={"dtype": torch.bfloat16},
    )
    if diffusion.model_dtype() is not torch.bfloat16:
        raise RuntimeError(f"Wan diffusion loaded as {diffusion.model_dtype()}, expected bfloat16")
    latent = torch.zeros((1, 16, 1, 2, 2), dtype=torch.float32)
    noise = comfy_sample.prepare_noise(latent, SEED)
    sampler = "euler"
    scheduler = "simple"
    steps = 1
    cfg = 5.0
    schedule = comfy_samplers.KSampler(
        diffusion,
        steps=steps,
        device=torch.device("cuda:0"),
        sampler=sampler,
        scheduler=scheduler,
        denoise=1.0,
    ).sigmas
    with torch.no_grad():
        sampled = comfy_sample.sample(
            diffusion,
            noise,
            steps,
            cfg,
            sampler,
            scheduler,
            positive,
            negative,
            latent,
            denoise=1.0,
            disable_pbar=True,
            seed=SEED,
        ).cpu()
    del diffusion, negative, negative_cond, positive, positive_cond
    _release()

    vae_path = models / ARTIFACTS["vae"][0]
    vae_state, vae_metadata = comfy_sd.comfy.utils.load_torch_file(
        str(vae_path), return_metadata=True
    )
    vae = comfy_sd.VAE(sd=vae_state, metadata=vae_metadata)
    if vae.vae_dtype is not torch.bfloat16:
        raise RuntimeError(f"Wan VAE selected {vae.vae_dtype}, expected bfloat16 default")
    content_ncthw = torch.linspace(0.0, 1.0, 1 * 3 * 5 * 16 * 16).reshape(1, 3, 5, 16, 16)
    model_management.load_models_gpu([vae.patcher], force_full_load=True)
    vae_model = vae.first_stage_model
    with torch.no_grad():
        encoded = vae_model.encode((content_ncthw * 2.0 - 1.0).to(vae.device, vae.vae_dtype))
        # Mirror comfy.sd.VAE.decode: cast to the float32 output dtype before
        # normalizing, so the [0, 1] mapping is not quantized to the VAE dtype.
        decoded_ncthw = ((vae_model.decode(encoded).to(torch.float32) + 1.0) / 2.0).clamp_(0.0, 1.0)
        pipeline_decoded_ncthw = vae.decode(sampled).movedim(-1, 1)
    peak_memory = torch.cuda.max_memory_allocated()
    del vae, vae_state
    _release()

    return {
        "reference": {
            "repository": "https://github.com/Comfy-Org/ComfyUI",
            "commit": COMFY_COMMIT,
            "comfy_kitchen": COMFY_KITCHEN_REF,
            "comfy_aimdo": COMFY_AIMDO_REF,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvidia_driver": _nvidia_driver_version(),
            "host": platform.node(),
            "device": torch.cuda.get_device_name(0),
            "device_index_policy": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_device": "cuda:0",
            },
            "attention": "pytorch SDPA",
            "diffusion_dtype": "bfloat16",
            "text_dtype": "float32",
            "vae_dtype": "bfloat16",
            "peak_cuda_bytes": peak_memory,
        },
        "artifacts": artifacts,
        "conditioning": {
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "prompt_token_ids": prompt_ids,
            "negative_token_ids": negative_ids,
            "positive_active": positive_active,
            "negative_active": negative_active,
        },
        "pipeline": {
            "seed": SEED,
            "sampler": sampler,
            "scheduler": scheduler,
            "steps": steps,
            "denoise": 1.0,
            "cfg": cfg,
            "sigmas": [float(value) for value in schedule],
            "initial_latent": _tensor(latent),
            "noise": _tensor(noise),
            "sampled_latent": _tensor(sampled),
            "decoded_content": _tensor(pipeline_decoded_ncthw),
        },
        "causal_vae": {
            "content": _tensor(content_ncthw),
            "encoded": _tensor(encoded),
            "decoded": _tensor(decoded_ncthw),
        },
    }


def main() -> None:
    models = _model_root()
    artifacts = _artifact_facts(models)
    with tempfile.TemporaryDirectory(prefix="dinkster-wan21-official-") as directory:
        root = Path(directory)
        source = root / "ComfyUI"
        kitchen = root / "comfy-kitchen"
        aimdo = root / "comfy-aimdo"
        _export(_comfy_root(), source)
        kitchen.mkdir()
        aimdo.mkdir()
        _export_ref(_repo_root().parents[1] / "comfy-kitchen", COMFY_KITCHEN_REF, kitchen)
        _export_ref(_repo_root().parents[1] / "comfy-aimdo", COMFY_AIMDO_REF, aimdo)
        payload = _run(source, kitchen, aimdo, models, artifacts)
    output = (
        _repo_root()
        / "packages"
        / "dinkster-inference-torch"
        / "tests"
        / "goldens"
        / "wan21_official_goldens.json"
    )
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
