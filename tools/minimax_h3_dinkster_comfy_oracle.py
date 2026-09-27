"""Compare the registered Dinkster MiniMax H3 path with pinned ComfyUI."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

COMFYUI_REVISION = "b5cc8830279eae909a59de030af1e50761c36751"
PROMPT = "A lighthouse flashes across a calm sea."
SEED = 459
STEPS = 1
WIDTH = 320
HEIGHT = 192
LENGTH = 5


def _hash_tensor(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(
        tensor.view(cast("Any", importlib.import_module("torch")).uint8).numpy().tobytes()
    ).hexdigest()


def _hash_latent(samples: Any) -> list[str]:
    if hasattr(samples, "roles"):
        streams = tuple(samples.by_role(role) for role in samples.roles)
    else:
        streams = samples.unbind() if samples.is_nested else (samples,)
    return [_hash_tensor(stream) for stream in streams]


def _empty_latent(torch: Any, nested_type: Any, intermediate_device: object) -> dict[str, object]:
    frame_count = LENGTH
    while frame_count % 17 != 5:
        frame_count += 1
    video_frames = 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2
    audio_frames = round(frame_count / 24 * 40)
    return {
        "samples": nested_type(
            (
                torch.zeros(
                    (1, 24, video_frames, HEIGHT // 16, WIDTH // 16),
                    device=intermediate_device,
                ),
                torch.zeros((1, 32, 2, audio_frames), device=intermediate_device),
            )
        )
    }


def _decode(
    vae: Any, audio_vae: Any, samples: Any, torch: Any, model_management: Any
) -> dict[str, object]:
    model_management.unload_all_models()
    model_management.soft_empty_cache()
    video_latent, audio_latent = samples.unbind()
    video = vae.decode(video_latent)
    audio = audio_vae.decode(audio_latent).movedim(-1, 1)
    std = torch.std(audio, dim=(1, 2), keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio /= std
    return {"video": _hash_tensor(video), "audio": _hash_tensor(audio)}


def _load_stock(paths: dict[str, Path], root: Path) -> tuple[Any, Any, Any, Any]:
    sys.path.insert(0, str(root))
    sd = cast("Any", importlib.import_module("comfy.sd"))
    utils = cast("Any", importlib.import_module("comfy.utils"))
    model = sd.load_diffusion_model(str(paths["diffusion"]), model_options={})
    clip = sd.load_clip(
        ckpt_paths=[str(paths["text_encoder"])],
        embedding_directory=[],
        clip_type=sd.CLIPType.MINIMAX,
        model_options={},
    )

    def load_vae(path: Path) -> Any:
        state_dict, metadata = utils.load_torch_file(str(path), return_metadata=True)
        return sd.VAE(sd=state_dict, metadata=metadata)

    return model, clip, load_vae(paths["video_vae"]), load_vae(paths["audio_vae"])


def _run_stock(paths: dict[str, Path], root: Path) -> dict[str, object]:
    torch = cast("Any", importlib.import_module("torch"))
    model, clip, vae, audio_vae = _load_stock(paths, root)
    sample = cast("Any", importlib.import_module("comfy.sample"))
    model_management = cast("Any", importlib.import_module("comfy.model_management"))
    nested_type = importlib.import_module("comfy.nested_tensor").NestedTensor
    latent = _empty_latent(torch, nested_type, model_management.intermediate_device())
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT, images=[]))
    model_management.unload_model_and_clones(model)
    noise = sample.prepare_noise(latent["samples"], SEED, None)
    output = sample.sample(
        model,
        noise,
        STEPS,
        1.0,
        "res_multistep",
        "simple",
        positive,
        [],
        latent["samples"],
        denoise=1.0,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        noise_mask=None,
        callback=None,
        disable_pbar=True,
        seed=SEED,
    )
    return {
        "latent": _hash_latent(output),
        "decoded": _decode(vae, audio_vae, output, torch, model_management),
        "gpu": str(torch.cuda.get_device_properties(0).uuid),
    }


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


def _run_worker(paths: dict[str, Path]) -> dict[str, object]:
    from dinkster_assets import AssetRef, digest_file
    from dinkster_native.fork_nodes import (
        GenerationKSampler,
        GenerationLoadDiffusionModel,
        GenerationVAEDecode,
        NativeLoadClip,
        NativeLoadVae,
        NativeMiniMaxH3ImageToVideo,
        NativeVAEDecodeAudio,
    )

    def asset(path: Path) -> AssetRef:
        return AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=_Resolver(path))

    torch = cast("Any", importlib.import_module("torch"))
    model = GenerationLoadDiffusionModel.execute(
        diffusion_model=asset(paths["diffusion"]), weight_dtype="default"
    )["model"]
    clip = NativeLoadClip.execute(
        text_encoder=asset(paths["text_encoder"]), type="minimax", device="default"
    )["clip"]
    vae = NativeLoadVae.execute(vae=asset(paths["video_vae"]))["vae"]
    audio_vae = NativeLoadVae.execute(vae=asset(paths["audio_vae"]))["vae"]
    conditioned = NativeMiniMaxH3ImageToVideo.execute(
        clip=clip,
        vae=vae,
        prompt=PROMPT,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
    )
    sampled = cast(
        "dict[str, Any]",
        GenerationKSampler.execute(
            model=model,
            seed=SEED,
            steps=STEPS,
            cfg=1.0,
            sampler_name="res_multistep",
            scheduler="simple",
            positive=conditioned["positive"],
            negative=[],
            latent_image=conditioned["latent"],
            denoise=1.0,
        )["latent"],
    )
    video = GenerationVAEDecode.execute(samples=sampled, vae=vae)["image"]
    audio = cast(
        "dict[str, Any]", NativeVAEDecodeAudio.execute(samples=sampled, vae=audio_vae)["audio"]
    )["waveform"]
    return {
        "latent": _hash_latent(sampled["samples"]),
        "decoded": {"video": _hash_tensor(video), "audio": _hash_tensor(audio)},
        "gpu": str(torch.cuda.get_device_properties(0).uuid),
    }


def _child(kind: str, paths: dict[str, Path], stock_root: Path | None) -> None:
    result = _run_stock(paths, cast("Path", stock_root)) if kind == "stock" else _run_worker(paths)
    print(json.dumps(result, sort_keys=True))


def _run_child(kind: str, paths: dict[str, Path], stock_root: Path | None) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--child", kind]
    for name, path in paths.items():
        command.extend((f"--{name.replace('_', '-')}", str(path)))
    if stock_root is not None:
        command.extend(("--stock-root", str(stock_root)))
    completed = subprocess.run(command, check=True, capture_output=True, text=True, env=os.environ)
    return json.loads(completed.stdout.splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("stock", "worker"))
    parser.add_argument("--stock-root", type=Path)
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--video-vae", type=Path, required=True)
    parser.add_argument("--audio-vae", type=Path, required=True)
    arguments = parser.parse_args()
    paths = {
        "diffusion": arguments.diffusion,
        "text_encoder": arguments.text_encoder,
        "video_vae": arguments.video_vae,
        "audio_vae": arguments.audio_vae,
    }
    if any(not path.is_file() for path in paths.values()):
        raise SystemExit("all four MiniMax H3 model files are required")
    if arguments.child:
        _child(arguments.child, paths, arguments.stock_root)
        return
    if arguments.stock_root is None:
        raise SystemExit("--stock-root is required")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=arguments.stock_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != COMFYUI_REVISION:
        raise SystemExit(f"stock root is {revision}, expected {COMFYUI_REVISION}")
    stock = _run_child("stock", paths, arguments.stock_root)
    worker = _run_child("worker", paths, None)
    if worker != stock:
        raise SystemExit(
            json.dumps({"status": "mismatch", "stock": stock, "worker": worker}, indent=2)
        )
    print(json.dumps({"status": "pass", "stock": stock, "worker": worker}, indent=2))


if __name__ == "__main__":
    main()
