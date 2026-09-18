"""Benchmark real-weight Wan 2.1 FFN and RoPE routes on CUDA.

The caller must hold the matching ``~/gpu-claims/gpuN.lock`` for the entire
process. This harness deliberately does not acquire GPU claims itself.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

CHECKPOINT_REPOSITORY = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
CHECKPOINT_REVISION = "617a7633e636506f850e043bc4605f290a466a8e"
CHECKPOINT_SIZE = 2_838_303_560
CHECKPOINT_SHA256 = "be531024cd9018cb5b48c40cfbb6a6191645b1c792eb8bf4f8c1c6e10f924dc5"
DEFAULT_CHECKPOINT = Path(
    "/home/kosin/ComfyUI-Shared/models/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors"
)
DEFAULT_TEXT_ENCODER = Path(
    "/home/kosin/ComfyUI-Shared/models/text_encoders/umt5_xxl_fp16.safetensors"
)
DEFAULT_VAE = Path("/home/kosin/ComfyUI-Shared/models/vae/wan_2.1_vae.safetensors")
LATENT_SHAPE = (1, 16, 5, 32, 32)
CONTEXT_SHAPE = (1, 64, 4096)
TIMESTEP = 500.0
ROPE_ABSOLUTE_TOLERANCE = 1 / 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _driver_version() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return output.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _fixed_tensor(torch: Any, shape: tuple[int, ...], *, device: Any) -> Any:
    count = math.prod(shape)
    values = torch.arange(count, dtype=torch.float32).remainder(257).sub(128).div(128)
    return values.reshape(shape).to(device=device, dtype=torch.bfloat16)


def _summary(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "max_ms": ordered[-1],
    }


def _measure(
    torch: Any,
    device: Any,
    operation: Any,
    *,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    with torch.set_grad_enabled(False):
        for _ in range(warmups):
            torch.cuda.synchronize(device)
            operation()
            torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        raw_ms: list[float] = []
        for _ in range(samples):
            torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            operation()
            torch.cuda.synchronize(device)
            raw_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "raw_ms": raw_ms,
        "summary": _summary(raw_ms),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _parity(torch: Any, optimized: Any, reference: Any) -> dict[str, Any]:
    optimized_float = optimized.float()
    reference_float = reference.float()
    absolute = (optimized_float - reference_float).abs()
    relative = absolute / reference_float.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "bit_equal": bool(torch.equal(optimized, reference)),
        "max_absolute_drift": float(absolute.max().item()),
        "max_relative_drift": float(relative.max().item()),
        "dtype": str(optimized.dtype).removeprefix("torch."),
        "shape": list(optimized.shape),
    }


@contextmanager
def _reference_routes(wan21_model: Any, flux: Any) -> Iterator[None]:
    original_ffn = wan21_model.WanFeedForward.forward
    original_rope = wan21_model.apply_rope

    def reference_ffn(self: Any, input: Any) -> Any:
        return self[2](self[1](self[0](input)))

    try:
        wan21_model.WanFeedForward.forward = reference_ffn
        wan21_model.apply_rope = flux._apply_rope_torch
        yield
    finally:
        wan21_model.WanFeedForward.forward = original_ffn
        wan21_model.apply_rope = original_rope


def _load_model(torch: Any, checkpoint: Path, text_encoder: Path, vae: Path) -> tuple[Any, str]:
    from dinkster_assets import digest_file
    from dinkster_inference import load_safetensors_header, plan_wan21_assembly
    from dinkster_inference_torch import assemble_wan21

    asset_digest = digest_file(checkpoint)
    plan = plan_wan21_assembly(
        diffusion=load_safetensors_header(checkpoint),
        umt5xxl=load_safetensors_header(text_encoder),
        vae=load_safetensors_header(vae),
    )
    assembled = assemble_wan21(
        plan,
        diffusion_dtype=torch.bfloat16,
        text_dtype=torch.float16,
        vae_dtype=torch.bfloat16,
        fp8_matmul=False,
    )
    return assembled.diffusion, asset_digest


def _run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"checkpoint does not exist: {checkpoint}")
    size = checkpoint.stat().st_size
    if size != CHECKPOINT_SIZE:
        raise RuntimeError(f"checkpoint size is {size}, expected {CHECKPOINT_SIZE}")
    sha256 = _sha256(checkpoint)
    if sha256 != CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA-256 is {sha256}, expected {CHECKPOINT_SHA256}")
    text_encoder = args.text_encoder.resolve()
    vae = args.vae.resolve()
    for role, path in (("text encoder", text_encoder), ("VAE", vae)):
        if not path.is_file():
            raise RuntimeError(f"{role} does not exist: {path}")

    from dinkster_inference_torch import flux, wan21_model

    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model, asset_digest = _load_model(torch, checkpoint, text_encoder, vae)
    model.eval().to(device)

    latent = _fixed_tensor(torch, LATENT_SHAPE, device=device)
    context = _fixed_tensor(torch, CONTEXT_SHAPE, device=device)
    timestep = torch.tensor([TIMESTEP], device=device, dtype=torch.float32)

    def forward() -> Any:
        return model(latent, timestep, context)

    optimized = _measure(torch, device, forward, warmups=args.warmups, samples=args.samples)
    with _reference_routes(wan21_model, flux):
        reference = _measure(torch, device, forward, warmups=args.warmups, samples=args.samples)

    token_count = math.prod(LATENT_SHAPE[2:]) // math.prod(model.config.patch_size)
    ffn_input = _fixed_tensor(torch, (1, token_count, model.config.hidden_size), device=device)
    ffn = model.blocks[0].ffn
    with torch.set_grad_enabled(False):
        optimized_ffn = ffn(ffn_input.clone())
        with _reference_routes(wan21_model, flux):
            reference_ffn = ffn(ffn_input.clone())

    rope_input = _fixed_tensor(
        torch,
        (
            1,
            token_count,
            model.config.num_heads,
            model.config.hidden_size // model.config.num_heads,
        ),
        device=device,
    )
    rope_frequencies = model._rope(
        (
            LATENT_SHAPE[2] // model.config.patch_size[0],
            LATENT_SHAPE[3] // model.config.patch_size[1],
            LATENT_SHAPE[4] // model.config.patch_size[2],
        ),
        latent,
    )
    owned_rope = flux._probe_dinkster_apply_rope()
    owned_rope_eligible = owned_rope is not None and owned_rope.supported(
        rope_input, rope_input, rope_frequencies
    )
    kitchen_rope_available = flux._kitchen_apply_rope() is not None
    with torch.set_grad_enabled(False):
        optimized_q, optimized_k = wan21_model.apply_rope(
            rope_input.clone(), rope_input.clone(), rope_frequencies
        )
        reference_q, reference_k = flux._apply_rope_torch(
            rope_input.clone(), rope_input.clone(), rope_frequencies
        )

    ffn_parity = _parity(torch, optimized_ffn, reference_ffn)
    rope_q_parity = _parity(torch, optimized_q, reference_q)
    rope_k_parity = _parity(torch, optimized_k, reference_k)
    for parity in (rope_q_parity, rope_k_parity):
        parity["absolute_tolerance"] = ROPE_ABSOLUTE_TOLERANCE
        parity["within_tolerance"] = parity["max_absolute_drift"] <= ROPE_ABSOLUTE_TOLERANCE
    if not ffn_parity["bit_equal"]:
        raise RuntimeError("real-weight FFN parity is not bit-exact")
    if not rope_q_parity["within_tolerance"] or not rope_k_parity["within_tolerance"]:
        raise RuntimeError("real-weight paired RoPE parity exceeds the frozen BF16 tolerance")

    optimized_median = optimized["summary"]["median_ms"]
    reference_median = reference["summary"]["median_ms"]
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "repository": {
            "url": _git("remote", "get-url", "origin"),
            "head": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "artifact": {
            "path": str(checkpoint),
            "repository": CHECKPOINT_REPOSITORY,
            "revision": CHECKPOINT_REVISION,
            "url": (
                f"https://huggingface.co/{CHECKPOINT_REPOSITORY}/resolve/"
                f"{CHECKPOINT_REVISION}/split_files/diffusion_models/{checkpoint.name}"
            ),
            "byte_size": size,
            "sha256": sha256,
            "asset_digest": asset_digest,
            "assembly_text_encoder": str(text_encoder),
            "assembly_vae": str(vae),
        },
        "environment": {
            "host": platform.node(),
            "device": torch.cuda.get_device_name(index),
            "device_index": index,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "driver": _driver_version(),
        },
        "input": {
            "latent_shape": list(LATENT_SHAPE),
            "context_shape": list(CONTEXT_SHAPE),
            "timestep": TIMESTEP,
            "compute_dtype": "bfloat16",
            "patch_size": list(model.config.patch_size),
            "token_count": token_count,
            "hidden_size": model.config.hidden_size,
            "ffn_hidden_size": model.config.ffn_hidden_size,
            "num_heads": model.config.num_heads,
            "head_dim": model.config.hidden_size // model.config.num_heads,
            "num_layers": model.config.num_layers,
        },
        "routes": {
            "optimized": {
                "ffn": "production WanFeedForward.forward using linear_input_act gelu_tanh",
                "rope": "production wan21_model.apply_rope using its fastest eligible paired route",
                "owned_rope_eligible": owned_rope_eligible,
                "kitchen_rope_available": kitchen_rope_available,
            },
            "reference": {
                "ffn": "self[2](self[1](self[0](input)))",
                "rope": "flux._apply_rope_torch",
                "scope": "WanFeedForward.forward and wan21_model.apply_rope only",
            },
        },
        "parity": {
            "ffn": ffn_parity,
            "rope_query": rope_q_parity,
            "rope_key": rope_k_parity,
        },
        "benchmark": {
            "warmups_per_route": args.warmups,
            "samples_per_route": args.samples,
            "optimized": optimized,
            "reference": reference,
            "speedup_ratio": reference_median / optimized_median,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The caller must hold ~/gpu-claims/gpuN.lock matching --device; "
            "this harness never acquires the lock."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--text-encoder", type=Path, default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--vae", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        parser.error("--device must select CUDA")
    if args.warmups < 2:
        parser.error("--warmups must be at least 2")
    if args.samples < 10:
        parser.error("--samples must be at least 10")
    try:
        payload = _run(args)
    except Exception as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
