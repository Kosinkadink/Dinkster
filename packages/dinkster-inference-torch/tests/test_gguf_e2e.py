"""Artifact-backed native GGUF checkpoint acceptance."""

from __future__ import annotations

import hashlib
import os
import struct
import zlib
from io import BytesIO
from pathlib import Path

import pytest
import torch
from dinkster_inference import (
    FLOAT32,
    GGUFResidencyMode,
    build_runtime_identity,
    extend_runtime_identity,
    load_gguf_weight_source,
    load_safetensors_header,
    plan_sd_assembly,
)
from dinkster_inference_torch import AssembledSD, GgufEncodedLinear, SDRuntime, assemble_sd
from golden_files import reference_validation_enabled
from PIL import Image

_GGUF_REVISION = "f129031aa93c19cc656a4444734791e5fa8446de"
_SDXL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
_ARTIFACTS = {
    "sdxl_base_1.0_Q8_0.gguf": (
        2_758_748_128,
        "76c792c7f3bc36bebe62fb785effd62469d7620b580cc4553b90a2debf584c24",
        f"https://huggingface.co/HyperX-Sentience/SDXL-GGUF/resolve/{_GGUF_REVISION}/sdxl_base_1.0_Q8_0.gguf",
    ),
    "clip_l.fp16.safetensors": (
        246_144_152,
        "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
        f"https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/{_SDXL_REVISION}/text_encoder/model.fp16.safetensors",
    ),
    "clip_g.fp16.safetensors": (
        1_389_382_176,
        "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4",
        f"https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/{_SDXL_REVISION}/text_encoder_2/model.fp16.safetensors",
    ),
    "vae.fp16.safetensors": (
        167_335_342,
        "bcb60880a46b63dea58e9bc591abe15f8350bde47b405f9c38f4be70c6161e68",
        f"https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/{_SDXL_REVISION}/vae/diffusion_pytorch_model.fp16.safetensors",
    ),
}
_PROMPT = "a red square on a white background"
_EXPECTED_PIXEL_SHA256 = "76b2bbe98184d2d18eebf2928b1636b4475c66da067e5bf4b6695c4c3f0e5be8"
_EXPECTED_PNG_SHA256 = "20f5586ab12fbedf3ffe349567503454d894f378ad6fb7714dd93b4d19d2ddc5"
_EXPECTED_BASE_RUNTIME_IDENTITY = (
    "native:dinkster.sdxl:71438e1521da4d6fa3d56a1875f7352dcda3da9f4db65eec11cc8e775e82748a"
)
_EXPECTED_GGUF_RUNTIME_FACTS = (
    "gguf.artifact.file_sha256=76c792c7f3bc36bebe62fb785effd62469d7620b580cc4553b90a2debf584c24",
    "gguf.artifact.manifest_sha256=71fdc722ec155374068722aa849fa19251dce6803c95b1efc0b02741e756cd8a",
    "gguf.artifact.mapper_id=dinkster.gguf.diffusion.v1",
    "gguf.artifact.architecture=sdxl",
    "gguf.artifact.family_id=dinkster.sdxl",
    "gguf.artifact.component=diffusion",
    "gguf.route.provider_key=dinkster-gguf-cpu-reference",
    "gguf.route.implementation_version=v1",
    "gguf.route.kind=reference-decode",
    "gguf.route.device_kind=cpu",
    "gguf.route.device_capability=generic",
    "gguf.route.compute_dtype=float32",
    "gguf.route.accumulation_dtype=float32",
)
_EXPECTED_MEMORY_GGUF_RUNTIME_FACTS = (
    *_EXPECTED_GGUF_RUNTIME_FACTS[:6],
    "gguf.route.provider_key=dinkster-gguf-torch-onuse",
    "gguf.route.implementation_version=v1",
    "gguf.route.kind=bounded-decode",
    "gguf.route.device_kind=any",
    "gguf.route.device_capability=generic",
    "gguf.route.compute_dtype=float32",
    "gguf.route.accumulation_dtype=float32",
    "gguf.route.fused_matmul=auto",
)
_EXPECTED_BALANCED_GGUF_RUNTIME_FACTS = (
    *_EXPECTED_GGUF_RUNTIME_FACTS[:6],
    "gguf.route.provider_key=dinkster-gguf-torch-onuse",
    "gguf.route.implementation_version=v1",
    "gguf.route.kind=cached-decode",
    "gguf.route.device_kind=any",
    "gguf.route.device_capability=generic",
    "gguf.route.compute_dtype=float32",
    "gguf.route.accumulation_dtype=float32",
    "gguf.route.decoded_cache=auto",
)
_EXPECTED_RUNTIME_IDENTITY = extend_runtime_identity(
    _EXPECTED_BASE_RUNTIME_IDENTITY,
    _EXPECTED_GGUF_RUNTIME_FACTS,
)
_EXPECTED_MEMORY_RUNTIME_IDENTITY = extend_runtime_identity(
    _EXPECTED_BASE_RUNTIME_IDENTITY,
    _EXPECTED_MEMORY_GGUF_RUNTIME_FACTS,
)
_EXPECTED_BALANCED_RUNTIME_IDENTITY = extend_runtime_identity(
    _EXPECTED_BASE_RUNTIME_IDENTITY,
    _EXPECTED_BALANCED_GGUF_RUNTIME_FACTS,
)
_EXPECTED_ENCODED_LINEARS = 717


def _artifact_root() -> Path | None:
    configured = os.environ.get("DINKSTER_GGUF_E2E_ROOT")
    candidates = (
        *((Path(configured),) if configured else ()),
        Path("/home/jed/model-artifacts/dinkster-gguf-e2e-240"),
        Path("/home/kosin/model-artifacts/dinkster-gguf-e2e-240"),
    )
    return next(
        (root for root in candidates if all((root / name).is_file() for name in _ARTIFACTS)),
        None,
    )


ROOT = _artifact_root()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _encode_rgb_png(width: int, height: int, pixels: bytes) -> bytes:
    row_bytes = width * 3
    rows = b"".join(
        b"\0" + pixels[offset : offset + row_bytes] for offset in range(0, len(pixels), row_bytes)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows))
        + _png_chunk(b"IEND", b"")
    )


def _assert_rgb_png(encoded: bytes, width: int, height: int, pixels: bytes) -> None:
    with Image.open(BytesIO(encoded)) as image:
        image.verify()
    with Image.open(BytesIO(encoded)) as image:
        image.load()
        assert image.mode == "RGB"
        assert image.size == (width, height)
        assert image.tobytes() == pixels


def test_rgb_png_contract_rejects_truncated_payload() -> None:
    pixels = bytes((0, 64, 255, 255, 64, 0))
    encoded = _encode_rgb_png(2, 1, pixels)
    _assert_rgb_png(encoded, 2, 1, pixels)
    with pytest.raises(OSError):
        _assert_rgb_png(encoded[:29], 2, 1, pixels)


def _plan_and_assemble(residency_mode: GGUFResidencyMode) -> tuple[AssembledSD, str]:
    assert ROOT is not None
    plan = plan_sd_assembly(
        diffusion=load_gguf_weight_source(
            ROOT / "sdxl_base_1.0_Q8_0.gguf",
            residency_mode=residency_mode,
        ),
        clip_l=load_safetensors_header(ROOT / "clip_l.fp16.safetensors"),
        clip_g=load_safetensors_header(ROOT / "clip_g.fp16.safetensors"),
        vae=load_safetensors_header(ROOT / "vae.fp16.safetensors"),
    )
    assert plan.diffusion.source_format == "gguf"
    assert any(dtype is FLOAT32 for dtype in plan.diffusion.dtypes.values())
    expected_facts = {
        "speed": _EXPECTED_GGUF_RUNTIME_FACTS,
        "memory": _EXPECTED_MEMORY_GGUF_RUNTIME_FACTS,
        "balanced": _EXPECTED_BALANCED_GGUF_RUNTIME_FACTS,
    }[residency_mode]
    assert plan.diffusion.runtime_facts == expected_facts
    assembled = assemble_sd(
        plan,
        diffusion_dtype=torch.float32,
        text_dtype=torch.float32,
        vae_dtype=torch.float32,
    )
    identity = build_runtime_identity(
        plan.family.id,
        plan.identity_components,
        diffusion_dtype=FLOAT32,
        text_dtype=FLOAT32,
        vae_dtype=FLOAT32,
        fp8_matmul=False,
    )
    return assembled, identity


def _sample_pinned_image(assembled: AssembledSD, identity: str) -> bytes:
    runtime = SDRuntime(assembled, runtime_identity=identity)
    with torch.inference_mode():
        conditioning = runtime.encode_text(_PROMPT)
        latent = runtime.sample(
            torch.zeros(1, 4, 8, 8),
            cond=conditioning,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            seed=240,
            compute_dtype=torch.float32,
            device="cpu",
        )
        pixels = runtime.decode_latent(latent).clamp(0, 1)
        image_bytes = bytes(
            (pixels[0].movedim(0, -1) * 255).round().to(torch.uint8).reshape(-1).tolist()
        )
    assert latent.shape == (1, 4, 8, 8)
    assert len(image_bytes) == 64 * 64 * 3
    return image_bytes


@pytest.mark.skipif(ROOT is None, reason="digest-verified SDXL GGUF artifact set absent")
def test_real_quantized_sdxl_checkpoint_samples_an_image() -> None:
    assert ROOT is not None
    for name, (size, digest, _source) in _ARTIFACTS.items():
        path = ROOT / name
        assert path.stat().st_size == size
        assert _sha256(path) == digest

    assembled, identity = _plan_and_assemble("speed")
    assert identity == _EXPECTED_RUNTIME_IDENTITY
    assert not any(
        isinstance(module, GgufEncodedLinear) for module in assembled.diffusion.modules()
    )
    image_bytes = _sample_pinned_image(assembled, identity)

    assert min(image_bytes) < max(image_bytes)
    if reference_validation_enabled():
        assert hashlib.sha256(image_bytes).hexdigest() == _EXPECTED_PIXEL_SHA256
    encoded = _encode_rgb_png(64, 64, image_bytes)
    _assert_rgb_png(encoded, 64, 64, image_bytes)
    if reference_validation_enabled():
        assert hashlib.sha256(encoded).hexdigest() == _EXPECTED_PNG_SHA256


@pytest.mark.skipif(ROOT is None, reason="digest-verified SDXL GGUF artifact set absent")
def test_memory_residency_samples_the_same_image_bit_exactly() -> None:
    assert ROOT is not None
    assembled, identity = _plan_and_assemble("memory")
    assert identity == _EXPECTED_MEMORY_RUNTIME_IDENTITY
    assert identity != _EXPECTED_RUNTIME_IDENTITY
    encoded_linears = sum(
        1 for module in assembled.diffusion.modules() if isinstance(module, GgufEncodedLinear)
    )
    assert encoded_linears == _EXPECTED_ENCODED_LINEARS
    image_bytes = _sample_pinned_image(assembled, identity)

    # The residency contract is bit-identity with the eager decode-at-load
    # route, asserted against a same-host speed-mode sample so the check
    # holds even where the host's kernels drift from the pinned goldens
    # (the speed test pins those goldens).
    speed_assembled, speed_identity = _plan_and_assemble("speed")
    speed_image_bytes = _sample_pinned_image(speed_assembled, speed_identity)
    assert image_bytes == speed_image_bytes


@pytest.mark.skipif(ROOT is None, reason="digest-verified SDXL GGUF artifact set absent")
def test_balanced_residency_samples_the_same_image_bit_exactly() -> None:
    assert ROOT is not None
    assembled, identity = _plan_and_assemble("balanced")
    assert identity == _EXPECTED_BALANCED_RUNTIME_IDENTITY
    assert identity not in (_EXPECTED_RUNTIME_IDENTITY, _EXPECTED_MEMORY_RUNTIME_IDENTITY)
    encoded_modules = [
        module for module in assembled.diffusion.modules() if isinstance(module, GgufEncodedLinear)
    ]
    assert len(encoded_modules) == _EXPECTED_ENCODED_LINEARS
    caches = {module.decoded_cache for module in encoded_modules}
    (cache,) = caches
    assert cache is not None
    assert cache.used_bytes == 0
    image_bytes = _sample_pinned_image(assembled, identity)

    # The sampling forwards filled the live-admission auto cache.
    assert cache.budget_bytes is None
    assert cache.used_bytes > 0
    # Release the cached decoded weights before assembling the eager
    # float32 reference so both never occupy host memory at once.
    cache.clear()
    del assembled, encoded_modules

    # Bit-identity with the eager decode-at-load route, asserted against
    # a same-host speed-mode sample (see the memory-residency test).
    speed_assembled, speed_identity = _plan_and_assemble("speed")
    speed_image_bytes = _sample_pinned_image(speed_assembled, speed_identity)
    assert image_bytes == speed_image_bytes
