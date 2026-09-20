r"""Generate Wan 2.1 pipeline goldens by executing pinned ComfyUI.

Run with the installed ComfyUI interpreter so its declared dependencies are
available::

    F:\workspaces\station1\stations\station14\installs\ComfyUI\.venv\Scripts\python.exe \
        tools\gen_wan21_pipeline_goldens.py

The generator exports ComfyUI commit b78cec87 to a temporary directory before
importing it. Tiny deterministic UMT5, Wan T2V/I2V, and causal Wan VAE models
keep the payload reviewable while exercising the same implementation boundaries
as the official models. The official repackaged UMT5 checkpoint supplies only
the pinned SentencePiece model used by ComfyUI's tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

if __package__:
    from .golden_platform import platform_variant_output_path
else:
    from golden_platform import platform_variant_output_path

COMFY_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
ARTIFACT_REPOSITORY = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
ARTIFACT_REVISION = "617a7633e636506f850e043bc4605f290a466a8e"
PROMPT = "A red fox runs through snow."
PROMPT_IDS = [320, 4062, 273, 56209, 48083, 3311, 45540, 274]
SEED = 94721

WAN_ARCH: dict[str, Any] = {
    "model_type": "t2v",
    "patch_size": [1, 2, 2],
    "text_len": 512,
    "in_dim": 16,
    "dim": 24,
    "ffn_dim": 48,
    "freq_dim": 8,
    "text_dim": 12,
    "out_dim": 16,
    "num_heads": 2,
    "num_layers": 2,
    "window_size": [-1, -1],
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
}
I2V_WAN_ARCH: dict[str, Any] = {
    **WAN_ARCH,
    "model_type": "i2v",
    "in_dim": 36,
}
UMT5_ARCH: dict[str, Any] = {
    "d_model": 12,
    "d_ff": 24,
    "d_kv": 6,
    "num_heads": 2,
    "num_layers": 2,
    "vocab_size": 256384,
    "dense_act_fn": "gelu_pytorch_tanh",
    "is_gated_act": True,
    "model_type": "umt5",
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "layer_norm_epsilon": 1e-6,
}
VAE_ARCH: dict[str, Any] = {
    "dim": 4,
    "z_dim": 16,
    "dim_mult": [1, 1, 1, 1],
    "num_res_blocks": 1,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "image_channels": 3,
    "conv_out_channels": 3,
    "dropout": 0.0,
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
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("set DINKSTER_COMFYUI_ROOT to a ComfyUI checkout")


def _text_encoder_path() -> Path:
    configured = os.environ.get("DINKSTER_WAN21_TEXT_ENCODER")
    candidates = (
        *((Path(configured).resolve(),) if configured else ()),
        _repo_root().parents[1]
        / "installs"
        / "ComfyUI"
        / "models"
        / "text_encoders"
        / "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("set DINKSTER_WAN21_TEXT_ENCODER to the official UMT5 checkpoint")


def _revision(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", COMFY_COMMIT], text=True
    ).strip()


def _export(repo: Path, destination: Path) -> None:
    archive = destination / "comfy.tar"
    subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", "-o", str(archive), COMFY_COMMIT],
        check=True,
    )
    subprocess.run(["tar", "-xf", str(archive), "-C", str(destination)], check=True)
    archive.unlink()


def _tensor(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "data": value.flatten().tolist(),
    }


def _entries(module: torch.nn.Module) -> list[tuple[str, list[int]]]:
    return sorted((key, list(value.shape)) for key, value in module.state_dict().items())


def _fill(
    module: torch.nn.Module,
    fill_state_dict: Any,
) -> list[tuple[str, list[int]]]:
    entries = _entries(module)
    module.load_state_dict(fill_state_dict(entries), strict=True)
    return entries


def _tokenizer_bytes(path: Path) -> bytes:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        data = checkpoint.get_tensor("spiece_model")
    if data.dtype != torch.uint8 or data.ndim != 1:
        raise RuntimeError("official UMT5 spiece_model must be rank-1 uint8")
    return data.contiguous().numpy().tobytes()


def _noise(shape: Sequence[int]) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    return torch.randn(tuple(shape), dtype=torch.float32, generator=generator)


def _content(shape: Sequence[int]) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    return torch.linspace(-0.9, 0.9, count, dtype=torch.float32).reshape(tuple(shape))


def _build_payload(source: Path, text_encoder: Path) -> dict[str, Any]:
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(_repo_root() / "packages" / "dinkster-inference-torch" / "tests"))

    import clip_fill
    import comfy.options
    import kl_fill
    import unet_fill

    # comfy.model_management asserts at import when torch has no CUDA
    # unless --cpu is parsed; every tensor here is built on CPU either way.
    if not torch.cuda.is_available():
        sys.argv = [sys.argv[0], "--cpu"]
    comfy.options.enable_args_parsing()
    from comfy import model_base, ops, supported_models
    from comfy.ldm.wan.vae import WanVAE
    from comfy.text_encoders.wan import WanT5Model, WanT5Tokenizer

    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    tokenizer_model = _tokenizer_bytes(text_encoder)
    tokenizer_tensor = torch.frombuffer(bytearray(tokenizer_model), dtype=torch.uint8)
    tokenizer = WanT5Tokenizer(tokenizer_data={"spiece_model": tokenizer_tensor})
    token_pairs = tokenizer.tokenize_with_weights(PROMPT)
    token_ids = [pair[0] for pair in token_pairs["umt5xxl"][0]]
    if token_ids[: len(PROMPT_IDS)] != PROMPT_IDS or token_ids[len(PROMPT_IDS)] != 1:
        raise RuntimeError("official ComfyUI tokenizer did not produce the pinned prompt IDs")

    text_model = WanT5Model(
        device="cpu",
        dtype=torch.float32,
        model_options={
            "custom_operations": ops.disable_weight_init,
            "umt5xxl_model_config": UMT5_ARCH,
        },
    )
    text_tower = text_model.umt5xxl.transformer
    text_entries = _fill(text_tower, clip_fill.fill_state_dict)
    encoded = text_model.encode_token_weights(token_pairs)[0].float()

    model_config = supported_models.WAN21_T2V(
        {
            **WAN_ARCH,
            "patch_size": tuple(WAN_ARCH["patch_size"]),
            "window_size": tuple(WAN_ARCH["window_size"]),
            "image_model": "wan2.1",
            "dtype": torch.float32,
        }
    )
    model_config.custom_operations = ops.disable_weight_init
    t2v_model = model_base.WAN21(model_config, device=torch.device("cpu"))
    diffusion_entries = _fill(t2v_model.diffusion_model, unet_fill.fill_state_dict)

    i2v_model_config = supported_models.WAN21_I2V(
        {
            **I2V_WAN_ARCH,
            "patch_size": tuple(I2V_WAN_ARCH["patch_size"]),
            "window_size": tuple(I2V_WAN_ARCH["window_size"]),
            "image_model": "wan2.1",
            "dtype": torch.float32,
        }
    )
    i2v_model_config.custom_operations = ops.disable_weight_init
    i2v_model = model_base.WAN21(
        i2v_model_config,
        image_to_video=True,
        device=torch.device("cpu"),
    )
    i2v_diffusion_entries = _fill(i2v_model.diffusion_model, unet_fill.fill_state_dict)

    vae = WanVAE(**VAE_ARCH)
    vae_entries = _fill(vae, kl_fill.fill_state_dict)

    vae_content = _content((1, 3, 5, 8, 8)) * 0.5 + 0.5
    vae_encoded = vae.encode(vae_content * 2.0 - 1.0)
    vae_latent = _content((1, 16, 2, 1, 1))
    vae_decoded = ((vae.decode(vae_latent) + 1.0) / 2.0).clamp_(0.0, 1.0)

    latent = torch.zeros((1, 16, 1, 2, 2), dtype=torch.float32)
    noise = _noise(latent.shape)
    sigma = torch.tensor([1.0], dtype=torch.float32)
    model_input = t2v_model.model_sampling.noise_scaling(
        sigma, noise, t2v_model.latent_format.process_in(latent), max_denoise=True
    )
    denoised = t2v_model.apply_model(model_input, sigma, c_crossattn=encoded)
    sampled = t2v_model.latent_format.process_out(denoised)
    decoded = ((vae.decode(sampled) + 1.0) / 2.0).clamp_(0.0, 1.0)

    start_content = _content((1, 3, 1, 16, 16)) * 0.5 + 0.5
    padded_content = torch.full((1, 3, 5, 16, 16), 0.5, dtype=torch.float32)
    padded_content[:, :, :1] = start_content
    reference_latent = vae.encode(padded_content * 2.0 - 1.0)
    i2v_latent = torch.zeros((1, 16, 2, 2, 2), dtype=torch.float32)
    i2v_noise = _noise(i2v_latent.shape)
    i2v_model_input = i2v_model.model_sampling.noise_scaling(
        sigma,
        i2v_noise,
        i2v_model.latent_format.process_in(i2v_latent),
        max_denoise=True,
    )
    concat_mask = torch.ones((1, 1, 2, 2, 2), dtype=torch.float32)
    concat_mask[:, :, :1] = 0.0
    model_concat = i2v_model.concat_cond(
        noise=i2v_model_input,
        concat_latent_image=reference_latent,
        concat_mask=concat_mask,
        device=torch.device("cpu"),
    )
    if model_concat is None or model_concat.shape != (1, 20, 2, 2, 2):
        raise RuntimeError("ComfyUI did not construct the expected Wan I2V concat channels")
    vision = unet_fill.hashed_input("i2v_pipeline:vision", (1, 257, 1280))
    i2v_denoised = i2v_model.apply_model(
        i2v_model_input,
        sigma,
        c_concat=model_concat,
        c_crossattn=encoded,
        clip_fea=vision,
    )
    i2v_sampled = i2v_model.latent_format.process_out(i2v_denoised)
    i2v_decoded = ((vae.decode(i2v_sampled) + 1.0) / 2.0).clamp_(0.0, 1.0)

    return {
        "source": {
            "repository": "https://github.com/Comfy-Org/ComfyUI",
            "commit": COMFY_COMMIT,
            "python": platform.python_version(),
            "torch": torch.__version__,
            **(
                {"os": platform.platform(), "platform": sys.platform}
                if not sys.platform.startswith("linux")
                else {}
            ),
        },
        "artifact": {
            "repository": ARTIFACT_REPOSITORY,
            "revision": ARTIFACT_REVISION,
            "url": (
                f"https://huggingface.co/{ARTIFACT_REPOSITORY}/resolve/{ARTIFACT_REVISION}/"
                "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
            ),
            "text_encoder_file": text_encoder.name,
            "text_encoder_byte_size": text_encoder.stat().st_size,
            "spiece_byte_size": len(tokenizer_model),
            "spiece_sha256": hashlib.sha256(tokenizer_model).hexdigest(),
        },
        "architecture": {
            "wan": WAN_ARCH,
            "wan_i2v": I2V_WAN_ARCH,
            "umt5": {key: value for key, value in UMT5_ARCH.items() if key != "layer_norm_epsilon"},
            "vae": {
                "dim": VAE_ARCH["dim"],
                "z_dim": VAE_ARCH["z_dim"],
                "dim_mult": VAE_ARCH["dim_mult"],
                "num_res_blocks": VAE_ARCH["num_res_blocks"],
                "attn_scales": VAE_ARCH["attn_scales"],
                "temporal_downsample": VAE_ARCH["temperal_downsample"],
                "image_channels": VAE_ARCH["image_channels"],
                "conv_out_channels": VAE_ARCH["conv_out_channels"],
                "dropout": VAE_ARCH["dropout"],
            },
        },
        "state": {
            "diffusion": diffusion_entries,
            "diffusion_i2v": i2v_diffusion_entries,
            "umt5": text_entries,
            "vae": vae_entries,
        },
        "tolerances": {
            "rtol": 0.0001,
            "atol": 0.00001,
            "reason": "CPU float32 SDPA and equivalent reordered tensor kernels",
        },
        "conditioning": {
            "prompt": PROMPT,
            "prompt_ids": PROMPT_IDS,
            "padded_token_ids": token_ids,
            "embedding": _tensor(encoded),
        },
        "causal_vae": {
            "content": _tensor(vae_content),
            "encoded": _tensor(vae_encoded),
            "latent": _tensor(vae_latent),
            "decoded": _tensor(vae_decoded),
        },
        "t2v_pipeline": {
            "seed": SEED,
            "sampler": "euler",
            "scheduler": "normal",
            "steps": 1,
            "flow_shift": 8.0,
            "sigmas": [1.0, 0.0],
            "initial_latent": _tensor(latent),
            "noise": _tensor(noise),
            "sampled_latent": _tensor(sampled),
            "decoded_content": _tensor(decoded),
        },
        "i2v_pipeline": {
            "seed": SEED,
            "sampler": "euler",
            "scheduler": "normal",
            "steps": 1,
            "flow_shift": 8.0,
            "sigmas": [1.0, 0.0],
            "start_content": _tensor(start_content),
            "padded_content": _tensor(padded_content),
            "reference_latent": _tensor(reference_latent),
            "concat_mask": _tensor(concat_mask),
            "model_concat": _tensor(model_concat),
            "vision_shape": list(vision.shape),
            "initial_latent": _tensor(i2v_latent),
            "noise": _tensor(i2v_noise),
            "sampled_latent": _tensor(i2v_sampled),
            "decoded_content": _tensor(i2v_decoded),
        },
    }


def main() -> None:
    repo = _comfy_root()
    revision = _revision(repo)
    if revision != COMFY_COMMIT:
        raise RuntimeError(f"resolved ComfyUI revision {revision}, expected {COMFY_COMMIT}")
    text_encoder = _text_encoder_path()
    with tempfile.TemporaryDirectory(prefix="dinkster-wan21-golden-") as directory:
        source = Path(directory)
        _export(repo, source)
        payload = _build_payload(source, text_encoder)
    base_output = (
        _repo_root()
        / "packages"
        / "dinkster-inference-torch"
        / "tests"
        / "goldens"
        / "wan21_pipeline_goldens.json"
    )
    runtime = f"py{platform.python_version()}-torch{torch.__version__}"
    key = runtime if sys.platform.startswith("linux") else f"{sys.platform}-{runtime}"
    output = platform_variant_output_path(base_output, key)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
