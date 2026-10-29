from __future__ import annotations

import gc
import hashlib
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import (
    CHROMA,
    CHROMA_RADIANCE,
    ChromaComponentRole,
    Conditioning,
    CustomSamplingRequest,
    chroma_component_family_id,
    chroma_component_runtime_identity,
    default_diffusion_dtype,
    default_text_dtype,
    default_vae_dtype,
    load_safetensors_header,
    plan_chroma_split_component,
)
from dinkster_inference_torch import (
    AutoencoderKL,
    Chroma,
    ChromaDiffusionRuntime,
    ChromaRadiance,
    ChromaTextRuntime,
    Fp8Linear,
    T5TextModel,
    enroll_component,
    kl_codec_plugin,
    load_chroma_component,
    soft_empty_cache,
)
from dinkster_inference_torch.solvers import torch_sampler_registry
from gpu_test_gate import require_gpu_tests_enabled

_PREFLIGHT = os.environ.get("DINKSTER_CHROMA_OFFICIAL_PREFLIGHT") == "1"
pytestmark = pytest.mark.skipif(
    not _PREFLIGHT and not torch.cuda.is_available(), reason="CUDA GPU required (see README)"
)

_MODELS = Path(os.environ.get("DINKSTER_CHROMA_MODELS", "/home/kosin/ComfyUI-Shared/models"))
_ARTIFACTS = {
    "chroma": (
        _MODELS / "diffusion_models/Chroma1-HD-fp8mixed.safetensors",
        9_193_379_316,
        "a2928ca6075f308f4d5e2182e2b96120fa8ad270ec6ea9b1b5c724c85c49a575",
        "https://huggingface.co/Comfy-Org/Chroma1-HD_repackaged/resolve/"
        "47f45ad2f72b2bccaa808418aeedca8c49d67974/split_files/diffusion_models/"
        "Chroma1-HD-fp8mixed.safetensors",
        "diffusion",
    ),
    "radiance": (
        _MODELS / "diffusion_models/chroma-radiance-x0.safetensors",
        19_012_346_326,
        "086e11d033ccd7470e67fa80e00a29902df2868cc84e16df0b48853be3a8672a",
        "https://huggingface.co/Comfy-Org/Chroma1-Radiance_Repackaged/resolve/"
        "c030c66a6aa7ff42dfe5f7c1a1e9cdc2652701d1/split_files/diffusion_models/"
        "chroma-radiance-x0.safetensors",
        "diffusion",
    ),
    "t5xxl": (
        _MODELS / "text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors",
        5_157_348_688,
        "a498f0485dc9536735258018417c3fd7758dc3bccc0a645feaa472b34955557a",
        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/"
        "6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5/t5xxl_fp8_e4m3fn_scaled.safetensors",
        "t5xxl",
    ),
    "vae": (
        _MODELS / "vae/ae.safetensors",
        335_304_388,
        "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
        "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/"
        "5b072540ef86570fecb8249c505f23d5bdeb88cd/split_files/vae/ae.safetensors",
        "vae",
    ),
}


class _Resolver:
    def __init__(self, digest: str, path: Path) -> None:
        self.digest = digest
        self.path = path

    def resolve(self, digest: str) -> Path | None:
        return self.path if digest == self.digest else None


def _official_plans() -> tuple[
    dict[str, Any], dict[str, AssetRef], dict[str, str], dict[str, ChromaComponentRole]
]:
    plans: dict[str, Any] = {}
    assets: dict[str, AssetRef] = {}
    identities: dict[str, str] = {}
    roles: dict[str, ChromaComponentRole] = {}
    for name, (path, size, sha256, url, role_value) in _ARTIFACTS.items():
        assert url.startswith("https://huggingface.co/")
        assert path.stat().st_size == size
        with path.open("rb") as artifact:
            assert hashlib.file_digest(artifact, "sha256").hexdigest() == sha256
        digest = digest_file(path)
        asset = AssetRef(digest, path.name, size, resolver=_Resolver(digest, path))
        role = cast("ChromaComponentRole", role_value)
        source = load_safetensors_header(path, asset_digest=digest, asset_size=size)
        plan = plan_chroma_split_component(source, role=role, path=path)
        plans[name] = plan
        assets[name] = asset
        family_id = chroma_component_family_id(plan)
        compute_dtype = {
            "diffusion": default_diffusion_dtype(family_id),
            "t5xxl": default_text_dtype(family_id),
            "vae": default_vae_dtype(family_id),
        }[role]
        identities[name] = chroma_component_runtime_identity(plan, role, compute_dtype)
        roles[name] = role
    return plans, assets, identities, roles


def _release(mechanism: Any) -> None:
    mechanism.unload()
    gc.collect()
    soft_empty_cache(torch.device("cuda:0"))


@pytest.mark.skipif(
    not all(values[0].exists() for values in _ARTIFACTS.values()),
    reason="official Chroma artifact set not present (set DINKSTER_CHROMA_MODELS)",
)
def test_official_chroma_family_loads_text_codec_and_samples_on_cuda() -> None:
    require_gpu_tests_enabled()
    plans, assets, identities, roles = _official_plans()

    assert chroma_component_family_id(plans["chroma"]) == CHROMA.id
    assert chroma_component_family_id(plans["radiance"]) == CHROMA_RADIANCE.id
    assert chroma_component_family_id(plans["t5xxl"]) == CHROMA.id
    assert chroma_component_family_id(plans["vae"]) == CHROMA.id
    assert sum(quant.config is not None for quant in plans["chroma"].quant.values()) == 228
    assert not any(quant.config is not None for quant in plans["radiance"].quant.values())
    assert not any(quant.config is not None for quant in plans["t5xxl"].quant.values())
    assert identities["chroma"].startswith("native:dinkster.chroma:")
    assert identities["radiance"].startswith("native:dinkster.chroma_radiance:")
    if _PREFLIGHT:
        assert not torch.cuda.is_initialized()
        return

    assert torch.cuda.device_count() == 1
    sampler = torch_sampler_registry().get("dinkster.euler")
    assert sampler is not None
    request = CustomSamplingRequest(sampler, (), (1.0, 0.0))

    text_loaded = load_chroma_component(
        _ARTIFACTS["t5xxl"][0],
        asset=assets["t5xxl"],
        expected_role=roles["t5xxl"],
        expected_identity=identities["t5xxl"],
        compute_dtype=torch.float32,
        load_device=torch.device("cuda:0"),
        attention_backend="t5",
    )
    text_model = cast("T5TextModel", text_loaded.module)
    assert not any(
        module.fp8_matmul for module in text_model.modules() if isinstance(module, Fp8Linear)
    )
    text_mechanism = enroll_component(text_model, load_device="cuda:0", offload_device="cpu")
    text_mechanism.partially_load(None)
    with torch.inference_mode():
        encoded = ChromaTextRuntime(text_model).encode_text("a radiant glass sculpture")
    assert encoded.embeddings.shape[0] == 1 and encoded.embeddings.shape[2] == 4096
    assert bool(torch.isfinite(encoded.embeddings).all())
    conditioning = Conditioning(encoded.embeddings.cpu(), None)
    _release(text_mechanism)
    del encoded, text_model, text_loaded

    vae_loaded = load_chroma_component(
        _ARTIFACTS["vae"][0],
        asset=assets["vae"],
        expected_role=roles["vae"],
        expected_identity=identities["vae"],
        compute_dtype=torch.bfloat16,
        load_device=torch.device("cuda:0"),
        attention_backend="vae",
    )
    vae_model = cast("AutoencoderKL", vae_loaded.module)
    vae_mechanism = enroll_component(vae_model, load_device="cuda:0", offload_device="cpu")
    vae_mechanism.partially_load(None)
    codec = replace(kl_codec_plugin(vae_model), compute_dtype=torch.bfloat16)
    content = torch.linspace(0.0, 1.0, 3 * 32 * 32, device="cuda:0").reshape(1, 3, 32, 32)
    with torch.inference_mode():
        latent = codec.encode(content)
        decoded = codec.decode(latent)
    assert latent.shape == (1, 16, 4, 4)
    assert decoded.shape == content.shape
    assert bool(torch.isfinite(latent).all()) and bool(torch.isfinite(decoded).all())
    assert float(decoded.min()) >= 0.0 and float(decoded.max()) <= 1.0
    _release(vae_mechanism)
    del codec, content, decoded, latent, vae_model, vae_loaded

    cases = (
        ("chroma", Chroma, 16, 2, 3.5, True),
        ("radiance", ChromaRadiance, 3, 16, 0.0, False),
    )
    for name, model_type, channels, spatial, guidance, fully_resident in cases:
        loaded = load_chroma_component(
            _ARTIFACTS[name][0],
            asset=assets[name],
            expected_role=roles[name],
            expected_identity=identities[name],
            compute_dtype=torch.bfloat16,
            load_device=torch.device("cuda:0"),
            attention_backend="flux",
        )
        model = cast("Chroma | ChromaRadiance", loaded.module)
        assert isinstance(model, model_type)
        assert sum(
            module.fp8_matmul for module in model.modules() if isinstance(module, Fp8Linear)
        ) == (228 if name == "chroma" else 0)
        mechanism = enroll_component(model, load_device="cuda:0", offload_device="cpu")
        if fully_resident:
            mechanism.partially_load(None)
            assert mechanism.loaded_bytes() == mechanism.total_bytes()
        else:
            assert mechanism.loaded_bytes() == 0
        runtime = ChromaDiffusionRuntime(
            model,
            runtime_identity=loaded.runtime_identity,
            compute_dtype=torch.bfloat16,
            attention_status={"flux": loaded.attention_status},
        )
        assert runtime.attention_status["flux"].primary == "sdpa"
        sample_input = torch.zeros(
            (1, channels, spatial, spatial), device="cuda:0", dtype=torch.float32
        )
        with torch.inference_mode():
            result = runtime.sample_custom(
                sample_input,
                noise=torch.ones_like(sample_input),
                cond=conditioning,
                request=request,
                seed=7,
                guidance=guidance,
                device="cuda:0",
            )
        assert result.output.shape == sample_input.shape
        assert result.output.dtype == torch.float32
        assert bool(torch.isfinite(result.output).all())
        if not fully_resident:
            assert mechanism.loaded_bytes() == 0
        _release(mechanism)
        del loaded, mechanism, model, result, runtime, sample_input
