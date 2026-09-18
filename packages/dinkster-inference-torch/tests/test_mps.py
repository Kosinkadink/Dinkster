"""Apple Silicon (MPS) validation of real-weight native runtimes.

Everything here is capability-gated: without an MPS device the whole
module skips, so the CPU-only `.venv-torch` gate and the CUDA suite are
unaffected. Real checkpoints resolve from model-root environment variables
documented in this package's README ("MPS validation"); tests skip when
their root is unset or the required files are absent.

The flow mirrors the CUDA real-weight Z-Image test in test_gpu.py:
header load, native probe, bf16 runtime assembly, resident-vs-streamed
text encoding, VAE roundtrip across an unload/reload cycle, and a
one-step sample. Equality assertions are exact (rtol=0, atol=0): MPS
compute is bit-deterministic for these ops, and a mismatch is a real
finding to investigate, never a tolerance to widen.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch

mps_available = torch.backends.mps.is_available()

pytestmark = pytest.mark.skipif(not mps_available, reason="MPS device required (see README)")


def _native_residency_module() -> Any:
    root = Path(__file__).parents[3]
    for package_source in sorted((root / "packages").glob("*/src")):
        source = str(package_source)
        if source not in sys.path:
            sys.path.insert(0, source)
    return importlib.import_module("dinkster_compat_comfy.native_residency")


def _z_image_paths() -> dict[str, Path]:
    root = os.environ.get("DINKSTER_Z_IMAGE_MODELS")
    if not root:
        pytest.skip("DINKSTER_Z_IMAGE_MODELS is unset (see README, 'MPS validation')")
    base = Path(root)
    paths = {
        "diffusion": base / "diffusion_models/z_image_turbo_bf16.safetensors",
        "qwen3_4b": base / "text_encoders/qwen_3_4b.safetensors",
        "vae": base / "vae/ae.safetensors",
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("real Z-Image checkpoint set not present under DINKSTER_Z_IMAGE_MODELS")
    return paths


def _qwen_image_paths() -> dict[str, Path]:
    root = os.environ.get("DINKSTER_QWEN_IMAGE_MODELS")
    if not root:
        pytest.skip("DINKSTER_QWEN_IMAGE_MODELS is unset (see README, 'MPS validation')")
    base = Path(root)
    paths = {
        "diffusion": base / "diffusion_models/qwen_image_bf16.safetensors",
        "qwen2_5_vl_7b": base / "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "vae": base / "vae/qwen_image_vae.safetensors",
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip(
            "official Qwen Image checkpoint set not present under DINKSTER_QWEN_IMAGE_MODELS"
        )
    return paths


class _QwenImageArtifactResolver:
    def __init__(self, artifacts: dict[str, Path]) -> None:
        self._artifacts = artifacts

    def resolve(self, digest: str) -> Path | None:
        return self._artifacts.get(digest)


def test_real_z_image_turbo_encodes_and_samples_on_mps() -> None:
    from dinkster_inference import load_safetensors_header, probe_native
    from dinkster_inference_torch import (
        ZImageRuntime,
        enroll_assembled,
        load_runtime,
        soft_empty_cache,
    )

    paths = _z_image_paths()
    diffusion = load_safetensors_header(paths["diffusion"])
    qwen3_4b = load_safetensors_header(paths["qwen3_4b"])
    vae = load_safetensors_header(paths["vae"])
    capability = probe_native(diffusion=diffusion, qwen3_4b=qwen3_4b, vae=vae)
    assert capability.native and capability.family_id == "dinkster.z_image"
    runtime = load_runtime(
        diffusion=diffusion,
        qwen3_4b=qwen3_4b,
        vae=vae,
        diffusion_dtype=torch.bfloat16,
        text_dtype=torch.bfloat16,
        vae_dtype=torch.bfloat16,
    )
    assert isinstance(runtime, ZImageRuntime)
    assert runtime.runtime_identity.startswith("native:dinkster.z_image:")
    assert not hasattr(runtime, "streamed_residency_components")

    enrolled = enroll_assembled(
        runtime.assembled,
        load_device="mps",
        offload_device="cpu",
    )
    enrolled["qwen3_4b"].partially_load(None)
    with torch.no_grad():
        resident_cond = runtime.encode_text("a photo of a cat")
    enrolled["qwen3_4b"].unload()
    soft_empty_cache(torch.device("mps"))
    with torch.no_grad():
        cond = runtime.encode_text("a photo of a cat")
    torch.testing.assert_close(cond.embeddings, resident_cond.embeddings, rtol=0, atol=0)
    assert cond.embeddings.shape[0] == 1 and cond.embeddings.shape[2] == 2560
    assert bool(torch.isfinite(cond.embeddings).all())

    content = torch.linspace(0.0, 1.0, 3 * 32 * 32, device="mps").reshape(1, 3, 32, 32)
    enrolled["vae"].partially_load(None)
    assert enrolled["vae"].loaded_bytes() == enrolled["vae"].total_bytes()
    with torch.no_grad():
        resident_latent = runtime.encode_content(content)
        resident_decoded = runtime.decode_latent(resident_latent)
    enrolled["vae"].unload()
    assert enrolled["vae"].loaded_bytes() == 0
    soft_empty_cache(torch.device("mps"))
    enrolled["vae"].partially_load(None)
    with torch.no_grad():
        reloaded_latent = runtime.encode_content(content)
        reloaded_decoded = runtime.decode_latent(reloaded_latent)
    assert resident_latent.shape == (1, 16, 4, 4)
    assert resident_decoded.shape == content.shape
    assert resident_latent.device.type == resident_decoded.device.type == "mps"
    assert resident_latent.dtype == resident_decoded.dtype == torch.bfloat16
    assert reloaded_latent.device.type == reloaded_decoded.device.type == "mps"
    assert reloaded_latent.dtype == reloaded_decoded.dtype == torch.bfloat16
    assert bool(torch.isfinite(resident_latent).all())
    assert bool(torch.isfinite(resident_decoded).all())
    torch.testing.assert_close(reloaded_latent, resident_latent, rtol=0, atol=0)
    torch.testing.assert_close(reloaded_decoded, resident_decoded, rtol=0, atol=0)
    enrolled["vae"].unload()
    soft_empty_cache(torch.device("mps"))

    enrolled["diffusion"].partially_load(None)
    latent = torch.zeros(1, 16, 8, 8)
    with torch.no_grad():
        out = runtime.sample(
            latent,
            cond=cond,
            sampler_id="dinkster.euler",
            scheduler_id="dinkster.simple",
            steps=1,
            seed=7,
            device="mps",
        )
    assert out.shape == latent.shape
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all())
    assert float(out.std()) > 0.01

    for mechanism in enrolled.values():
        mechanism.unload()
    soft_empty_cache(torch.device("mps"))


def test_real_qwen_image_encodes_samples_and_decodes_on_mps() -> None:
    from dinkster_assets import AssetRef, digest_file
    from dinkster_inference import (
        BFLOAT16,
        QWEN_IMAGE,
        compose_execution,
        load_safetensors_header,
        plan_qwen_image_official_component,
        qwen_image_component_runtime_identity,
    )
    from dinkster_inference_torch import (
        AssembledQwenImage,
        QwenImageRuntime,
        ResidencyManager,
        load_qwen_image_component,
        soft_empty_cache,
    )

    paths = _qwen_image_paths()
    authorities = {role: (path.stat().st_size, digest_file(path)) for role, path in paths.items()}
    resolver = _QwenImageArtifactResolver(
        {digest: paths[role] for role, (_size, digest) in authorities.items()}
    )
    loaded: dict[str, Any] = {}
    identities: dict[str, str] = {}
    for role, path in paths.items():
        size, digest = authorities[role]
        source = load_safetensors_header(
            path,
            asset_digest=digest,
            asset_size=size,
        )
        plan = plan_qwen_image_official_component(
            source,
            role=cast(Any, role),
            path=path,
        )
        expected_identity = qwen_image_component_runtime_identity(
            plan,
            cast(Any, role),
            BFLOAT16,
        )
        loaded[role] = load_qwen_image_component(
            path,
            asset=AssetRef(
                digest,
                path.name,
                size,
                resolver=resolver,
            ),
            expected_role=cast(Any, role),
            expected_identity=expected_identity,
            compute_dtype=torch.bfloat16,
        )
        identities[role] = expected_identity
    assembled = AssembledQwenImage(
        diffusion=loaded["diffusion"].module,
        text=loaded["qwen2_5_vl_7b"].module,
        vae=loaded["vae"].module,
        family=QWEN_IMAGE,
        _component_compute_dtypes={
            "diffusion": torch.bfloat16,
            "text": torch.bfloat16,
            "vae": torch.bfloat16,
        },
    )
    execution = compose_execution(QWEN_IMAGE.id, identities)
    runtime = QwenImageRuntime(
        assembled,
        runtime_identity=execution.execution_identity,
    )
    native_residency = _native_residency_module()
    coordinator = native_residency.NativeResidencyCoordinator(ResidencyManager())
    enrolled = {}
    try:
        for role, component in loaded.items():
            enrolled[role] = coordinator.enroll_component(
                component.module,
                load_device="mps",
                offload_device="cpu",
            )
        with torch.no_grad():
            first_cond = runtime.encode_text("a photo of a cat")
            second_cond = runtime.encode_text("a photo of a cat")
        assert torch.equal(second_cond.embeddings, first_cond.embeddings)
        assert first_cond.embeddings.device.type == "mps"
        assert bool(torch.isfinite(first_cond.embeddings).all())
        soft_empty_cache(torch.device("mps"))

        latent = torch.zeros((1, 16, 1, 2, 2), device="mps")
        with torch.no_grad():
            first_sample = runtime.sample(
                latent,
                cond=first_cond,
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.simple",
                steps=1,
                seed=7,
            )
            second_sample = runtime.sample(
                latent,
                cond=first_cond,
                sampler_id="dinkster.euler",
                scheduler_id="dinkster.simple",
                steps=1,
                seed=7,
            )
        assert torch.equal(second_sample, first_sample)
        assert first_sample.shape == latent.shape
        assert first_sample.dtype == torch.float32
        assert bool(torch.isfinite(first_sample).all())
        assert float(first_sample.std()) > 0.01

        enrolled["vae"].partially_load(None)
        with torch.no_grad():
            decoded = runtime.decode_latent(first_sample)
        assert decoded.shape == (1, 3, 16, 16)
        assert decoded.device.type == "mps"
        assert decoded.dtype == torch.float32
        assert torch.mps.current_allocated_memory() > 0
        assert bool(torch.isfinite(decoded).all())
    finally:
        for mechanism in enrolled.values():
            mechanism.unload()
        soft_empty_cache(torch.device("mps"))
