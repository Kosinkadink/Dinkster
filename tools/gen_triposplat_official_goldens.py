r"""Run the official TripoSplat pipeline in pinned ComfyUI.

This same-checkpoint acceptance oracle exports ComfyUI commit b78cec87,
authenticates the immutable VAST-AI/TripoSplat artifacts, and records the
bounded outputs consumed by Dinkster's capability-gated GPU test.

Usage::

    CUDA_VISIBLE_DEVICES="" /path/to/python tools/gen_triposplat_official_goldens.py --preflight
    CUDA_VISIBLE_DEVICES=0 /path/to/python tools/gen_triposplat_official_goldens.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
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
ARTIFACT_REPOSITORY = "VAST-AI/TripoSplat"
ARTIFACT_REVISION = "56a96e603204ec410c4da60c13ea4fa09a2169a9"
ARTIFACTS = {
    "diffusion": (
        "diffusion_models/triposplat_fp16.safetensors",
        741_106_994,
        "c870b97ac1d6bc9177608a5ec625e19ef9f3c5019aa68f64b0fb7803abcd6d20",
    ),
    "vision": (
        "clip_vision/dino_v3_vit_h.safetensors",
        1_681_247_696,
        "a29ef35101a16966972a0d50732a6f3a608ff7cfffb2afa9bbe9007cb842cc53",
    ),
    "reference_vae": (
        "vae/flux2-vae.safetensors",
        336_213_556,
        "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5",
    ),
    "gaussian_decoder": (
        "vae/triposplat_vae_decoder_fp16.safetensors",
        576_148_442,
        "ed0d0c3d43b599e326845d0ec70f3cf77be9a55e2d97627ac3b34d2830763cc8",
    ),
}
IMAGE_SEED = 19
IMAGE_HEIGHT = 288
IMAGE_WIDTH = 320
PREPROCESS_SIZE = 256
ERODE_RADIUS = 1
SAMPLE_SEED = 7
DECODE_SEED = 0
NUM_GAUSSIANS = 32_768
SLICE_VALUES = 4_096


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_roots(configured: str | None, name: str) -> tuple[Path, ...]:
    repo = _repo_root()
    return (
        *((Path(configured).resolve(),) if configured else ()),
        *(parent / name for parent in repo.parents),
    )


def _comfy_root() -> Path:
    for candidate in _candidate_roots(os.environ.get("DINKSTER_COMFYUI_ROOT"), "ComfyUI"):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("set DINKSTER_COMFYUI_ROOT to the pinned ComfyUI checkout")


def _source_repo(name: str, environment: str) -> Path:
    for candidate in _candidate_roots(os.environ.get(environment), name):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"set {environment} to the {name} checkout")


def _model_root() -> Path:
    configured = os.environ.get("DINKSTER_TRIPOSPLAT_MODELS")
    candidates = (
        *((Path(configured).resolve(),) if configured else ()),
        Path("/home/kosin/model-artifacts/dinkster-triposplat-e2e"),
    )
    for candidate in candidates:
        if all((candidate / values[0]).is_file() for values in ARTIFACTS.values()):
            return candidate
    raise RuntimeError("set DINKSTER_TRIPOSPLAT_MODELS to the official split-model root")


def _require_clean_pinned_checkout(repo: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != COMFY_COMMIT:
        raise RuntimeError(f"ComfyUI checkout is at {revision}, expected {COMFY_COMMIT}")
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise RuntimeError(f"ComfyUI checkout must be clean:\n{dirty}")


def _export(repo: Path, destination: Path) -> None:
    _require_clean_pinned_checkout(repo)
    destination.mkdir()
    _export_ref(repo, COMFY_COMMIT, destination)


def _export_ref(repo: Path, ref: str, destination: Path) -> None:
    archive = destination / "source.tar"
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
                f"{relative}"
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


def _moments(value: Any) -> dict[str, float]:
    flat = value.detach().to(device="cpu", dtype=torch.float32).flatten()
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


def _summary(value: Any) -> dict[str, Any]:
    flat = value.detach().to(device="cpu", dtype=torch.float32).contiguous().flatten()
    stride = max(1, flat.numel() // SLICE_VALUES)
    return {
        "shape": list(value.shape),
        "dtype": "float32",
        "moments": _moments(flat),
        "slice": {
            "stride": stride,
            "data": flat[::stride][:SLICE_VALUES].tolist(),
        },
    }


def _synthetic_input() -> tuple[Any, Any]:
    generator = torch.Generator(device="cpu").manual_seed(IMAGE_SEED)
    y = torch.linspace(0.0, 1.0, IMAGE_HEIGHT).view(IMAGE_HEIGHT, 1)
    x = torch.linspace(0.0, 1.0, IMAGE_WIDTH).view(1, IMAGE_WIDTH)
    image = torch.stack(
        (
            x.expand(IMAGE_HEIGHT, -1),
            y.expand(-1, IMAGE_WIDTH),
            (x + y).expand(IMAGE_HEIGHT, IMAGE_WIDTH) * 0.5,
        ),
        dim=-1,
    )
    image = (image + torch.rand(image.shape, generator=generator) * 0.01).clamp(0.0, 1.0)
    yy = torch.arange(IMAGE_HEIGHT).view(IMAGE_HEIGHT, 1) - (IMAGE_HEIGHT - 1) / 2
    xx = torch.arange(IMAGE_WIDTH).view(1, IMAGE_WIDTH) - (IMAGE_WIDTH - 1) / 2
    radius = min(IMAGE_HEIGHT, IMAGE_WIDTH) * 0.32
    mask = ((xx.square() + yy.square()) <= radius**2).to(torch.float32)
    return image.unsqueeze(0), mask.unsqueeze(0)


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


def _configure_reference(source: Path, kitchen: Path, aimdo: Path, *, preflight: bool) -> None:
    del kitchen
    if importlib.metadata.version("comfy-kitchen") != COMFY_KITCHEN_REF.removeprefix("v"):
        raise RuntimeError(f"installed comfy-kitchen must be {COMFY_KITCHEN_REF}")
    sys.path[:0] = [str(source), str(aimdo)]
    sys.argv = (
        [sys.argv[0], "--cpu"]
        if preflight
        else [
            sys.argv[0],
            "--highvram",
            "--disable-xformers",
            "--use-pytorch-cross-attention",
            "--bf16-unet",
            "--fp32-text-enc",
            "--fp32-vae",
        ]
    )

    import comfy.options

    comfy.options.enable_args_parsing()


def _preflight(source: Path, kitchen: Path, aimdo: Path) -> None:
    _configure_reference(source, kitchen, aimdo, preflight=True)

    global torch
    import torch
    from comfy import clip_vision, model_management, sample, samplers, sd
    from comfy.image_encoders import dino3
    from comfy.ldm.modules import attention
    from comfy.ldm.triposplat import model, vae
    from comfy_extras import nodes_triposplat

    del (
        attention,
        clip_vision,
        dino3,
        model,
        model_management,
        nodes_triposplat,
        sample,
        samplers,
        sd,
        vae,
    )
    if torch.cuda.is_initialized():
        raise RuntimeError("preflight initialized CUDA")
    print("preflight reached the official pipeline launch boundary with CUDA uninitialized")


def _run(
    source: Path,
    kitchen: Path,
    aimdo: Path,
    models: Path,
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    _configure_reference(source, kitchen, aimdo, preflight=False)

    global torch, model_management
    import torch
    import torch.nn.functional as functional
    from comfy import clip_vision, model_management, nested_tensor
    from comfy import sample as comfy_sample
    from comfy import samplers as comfy_samplers
    from comfy import sd as comfy_sd
    from comfy.image_encoders import dino3 as dino3_module
    from comfy.ldm.modules import attention as attention_module
    from comfy.ldm.triposplat import model as triposplat_module
    from comfy_extras.nodes_triposplat import _preprocess

    if not torch.cuda.is_available():
        raise RuntimeError("official TripoSplat acceptance requires CUDA")
    triposplat_module.optimized_attention = attention_module.attention_pytorch
    dino3_module.optimized_attention_for_device = lambda device, mask=False: (
        attention_module.attention_pytorch
    )
    if triposplat_module.attention.__globals__["optimized_attention"] is not (
        attention_module.attention_pytorch
    ):
        raise RuntimeError("TripoSplat did not select pytorch SDPA")
    torch.cuda.reset_peak_memory_stats()

    print("preprocessing synthetic input", flush=True)
    image, mask = _synthetic_input()
    with torch.no_grad():
        preprocessed = _preprocess(image[0], mask[0], ERODE_RADIUS, PREPROCESS_SIZE)

    print("encoding DINOv3 conditioning", flush=True)
    vision_path = models / ARTIFACTS["vision"][0]
    vision = clip_vision.load(str(vision_path))
    if vision is None:
        raise RuntimeError("DINOv3 artifact was not recognized")
    if vision.dtype is not torch.float32:
        raise RuntimeError(f"DINOv3 loaded as {vision.dtype}, expected float32")
    model_management.load_model_gpu(vision.patcher)
    pixel = preprocessed.movedim(-1, 1).to(vision.load_device)
    mean = pixel.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = pixel.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    with torch.no_grad():
        sequence = vision.model(pixel_values=((pixel - mean) / std).float())[0]
        features = functional.layer_norm(sequence.float(), sequence.shape[-1:]).cpu()
    del mean, pixel, sequence, std, vision
    _release()

    print("encoding Flux2 reference latent", flush=True)
    reference_path = models / ARTIFACTS["reference_vae"][0]
    reference_state, reference_metadata = comfy_sd.comfy.utils.load_torch_file(
        str(reference_path), return_metadata=True
    )
    reference_vae = comfy_sd.VAE(
        sd=reference_state, metadata=reference_metadata, dtype=torch.float32
    )
    if reference_vae.vae_dtype is not torch.float32:
        raise RuntimeError(
            f"Flux2 reference VAE loaded as {reference_vae.vae_dtype}, expected float32"
        )
    with torch.no_grad():
        reference_latent = reference_vae.encode(preprocessed).cpu()
    del reference_metadata, reference_state, reference_vae
    _release()

    print("sampling TripoSplat latent streams", flush=True)
    diffusion_path = models / ARTIFACTS["diffusion"][0]
    diffusion = comfy_sd.load_diffusion_model(
        str(diffusion_path), model_options={"dtype": torch.bfloat16}
    )
    if diffusion.model_dtype() is not torch.bfloat16:
        raise RuntimeError(
            f"TripoSplat diffusion loaded as {diffusion.model_dtype()}, expected bfloat16"
        )
    positive = [[features, {"reference_latents": [reference_latent]}]]
    negative = [
        [torch.zeros_like(features), {"reference_latents": [torch.zeros_like(reference_latent)]}]
    ]
    latent = nested_tensor.NestedTensor(
        (
            torch.zeros((1, 8192, 16), dtype=torch.float32),
            torch.zeros((1, 1, 5), dtype=torch.float32),
        )
    )
    noise = comfy_sample.prepare_noise(latent, SAMPLE_SEED)
    sampler = "euler"
    scheduler = "simple"
    steps = 4
    cfg = 4.0
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
            seed=SAMPLE_SEED,
        ).cpu()
    sampled_latent, sampled_camera = sampled.unbind()
    del diffusion, latent, negative, noise, positive
    _release()

    print("decoding Gaussian splat", flush=True)
    decoder_path = models / ARTIFACTS["gaussian_decoder"][0]
    decoder_state, decoder_metadata = comfy_sd.comfy.utils.load_torch_file(
        str(decoder_path), return_metadata=True
    )
    decoder_vae = comfy_sd.VAE(sd=decoder_state, metadata=decoder_metadata, dtype=torch.float32)
    if decoder_vae.vae_dtype is not torch.float32:
        raise RuntimeError(
            f"TripoSplat decoder loaded as {decoder_vae.vae_dtype}, expected float32"
        )
    model_management.load_models_gpu([decoder_vae.patcher], force_full_load=True)
    decoder = decoder_vae.first_stage_model
    generator = torch.Generator(device="cpu").manual_seed(DECODE_SEED)
    with torch.no_grad():
        parts = decoder.decode(
            sampled_latent.to(device=decoder_vae.device, dtype=decoder_vae.vae_dtype),
            num_gaussians=NUM_GAUSSIANS,
            generator=generator,
        )
        rendered = [part.render_tensors() for part in parts]
        decoded = {
            name: torch.stack(values).cpu()
            for name, values in zip(
                ("positions", "scales", "rotations", "opacities", "sh"),
                zip(*rendered, strict=True),
                strict=True,
            )
        }
    peak_memory = torch.cuda.max_memory_allocated()
    del decoder, decoder_metadata, decoder_state, decoder_vae, parts, rendered, sampled
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
            "vision_dtype": "float32",
            "reference_vae_dtype": "float32",
            "gaussian_decoder_dtype": "float32",
            "peak_cuda_bytes": peak_memory,
        },
        "artifacts": artifacts,
        "input": {
            "image_seed": IMAGE_SEED,
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "preprocess_size": PREPROCESS_SIZE,
            "erode_radius": ERODE_RADIUS,
        },
        "conditioning": {
            "preprocessed": _summary(preprocessed),
            "features": _summary(features),
            "reference_latent": _summary(reference_latent),
        },
        "pipeline": {
            "seed": SAMPLE_SEED,
            "sampler": sampler,
            "scheduler": scheduler,
            "steps": steps,
            "denoise": 1.0,
            "cfg": cfg,
            "sigmas": [float(value) for value in schedule],
            "sampled_latent": _tensor(sampled_latent),
            "sampled_camera": _tensor(sampled_camera),
        },
        "decode": {
            "seed": DECODE_SEED,
            "num_gaussians": NUM_GAUSSIANS,
            "tensors": {
                name: {
                    "first_64": _tensor(value[:, :64]),
                    "moments": _moments(value),
                }
                for name, value in decoded.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    models = _model_root()
    print(f"authenticating artifacts under {models}", flush=True)
    artifacts = _artifact_facts(models)
    with tempfile.TemporaryDirectory(prefix="dinkster-triposplat-official-") as directory:
        root = Path(directory)
        source = root / "ComfyUI"
        kitchen = root / "comfy-kitchen"
        aimdo = root / "comfy-aimdo"
        _export(_comfy_root(), source)
        kitchen.mkdir()
        aimdo.mkdir()
        _export_ref(
            _source_repo("comfy-kitchen", "DINKSTER_COMFY_KITCHEN_ROOT"),
            COMFY_KITCHEN_REF,
            kitchen,
        )
        _export_ref(
            _source_repo("comfy-aimdo", "DINKSTER_COMFY_AIMDO_ROOT"),
            COMFY_AIMDO_REF,
            aimdo,
        )
        if args.preflight:
            _preflight(source, kitchen, aimdo)
            return
        payload = _run(source, kitchen, aimdo, models, artifacts)
    output = (
        _repo_root()
        / "packages"
        / "dinkster-inference-torch"
        / "tests"
        / "goldens"
        / "triposplat_official_goldens.json"
    )
    output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
