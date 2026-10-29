"""Torch-free LTX-2 split-component identity tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT8_E4M3,
    FLOAT32,
    GEMMA3_LTX_12B_CONFIG,
    LTX_LATENT_UPSAMPLER_CONFIG,
    LTX_TEXT_CONNECTOR_CONFIG,
    LTXAV,
    LTXAV_19B_AUDIO_VAE_CONFIG,
    LTXAV_19B_CONFIG,
    LTXAV_19B_VAE_CONFIG,
    LTXAV_19B_VOCODER_CONFIG,
    ComponentBinding,
    ComponentPlan,
    LayerQuant,
    LTXAVAudioCodecPlan,
    LTXAVStandaloneComponentPlan,
)
from dinkster_inference import ltxav_component as component
from dinkster_inference.weights import WeightEntry, WeightSource
from dinkster_protocol import ATTENTION_ROLES, AttentionRoute, AttentionRouteToken


@dataclass(frozen=True)
class _Source:
    path: Path
    asset_size: int | None
    asset_digest: str | None

    def keys(self) -> tuple[str, ...]:
        return ()

    def entry(self, key: str) -> WeightEntry:
        raise KeyError(key)

    def metadata(self) -> dict[str, str]:
        return {}


@dataclass(frozen=True)
class _BareSource:
    path: Path

    def keys(self) -> tuple[str, ...]:
        return ()

    def entry(self, key: str) -> WeightEntry:
        raise KeyError(key)

    def metadata(self) -> dict[str, str]:
        return {}


def _plan(path: Path) -> LTXAVAudioCodecPlan:
    audio_vae = ComponentPlan("audio_vae", path, LTXAV_19B_AUDIO_VAE_CONFIG, {}, {}, {})
    vocoder = ComponentPlan("vocoder", path, LTXAV_19B_VOCODER_CONFIG, {}, {}, {})
    return LTXAVAudioCodecPlan(LTXAV, audio_vae, vocoder)


def _component_plan(role: str, path: Path) -> LTXAVStandaloneComponentPlan:
    config = {
        "diffusion": LTXAV_19B_CONFIG,
        "gemma3_12b": GEMMA3_LTX_12B_CONFIG,
        "text_projection": "single_linear",
        "connectors": LTX_TEXT_CONNECTOR_CONFIG,
        "latent_upscaler": LTX_LATENT_UPSAMPLER_CONFIG,
        "vae": LTXAV_19B_VAE_CONFIG,
    }[role]
    component_plan = ComponentPlan(role, path, config, {}, {}, {})
    return LTXAVStandaloneComponentPlan(
        cast("Any", role),
        cast("Any", component_plan),
        "spiece_model" if role == "gemma3_12b" else "",
    )


def test_standalone_diffusion_plan_requires_an_exact_supported_profile(tmp_path: Path) -> None:
    path = tmp_path / "diffusion.safetensors"

    varied_multiplier = replace(LTXAV_19B_CONFIG, av_ca_timestep_scale_multiplier=250.0)
    LTXAVStandaloneComponentPlan(
        "diffusion", ComponentPlan("diffusion", path, varied_multiplier, {}, {}, {})
    )

    invalid = replace(LTXAV_19B_CONFIG, num_layers=LTXAV_19B_CONFIG.num_layers - 1)
    with pytest.raises(ValueError, match="exact supported roles"):
        LTXAVStandaloneComponentPlan(
            "diffusion", ComponentPlan("diffusion", path, invalid, {}, {}, {})
        )
    nonfinite = replace(LTXAV_19B_CONFIG, av_ca_timestep_scale_multiplier=float("inf"))
    with pytest.raises(ValueError, match="exact supported roles"):
        LTXAVStandaloneComponentPlan(
            "diffusion", ComponentPlan("diffusion", path, nonfinite, {}, {}, {})
        )


def test_standalone_diffusion_uses_checkpoint_selected_fp8_matmul(tmp_path: Path) -> None:
    quant = LayerQuant(
        "transformer_blocks.2.attn1.to_q",
        "float8_e4m3fn",
        "transformer_blocks.2.attn1.to_q.weight",
        "transformer_blocks.2.attn1.to_q.weight_scale",
        "transformer_blocks.2.attn1.to_q.input_scale",
    )
    weight_key = f"{quant.layer}.weight"
    component_plan = ComponentPlan(
        "diffusion",
        tmp_path / "diffusion.safetensors",
        LTXAV_19B_CONFIG,
        {weight_key: weight_key},
        {weight_key: FLOAT8_E4M3},
        {quant.layer: quant},
    )

    assert component.ltxav_component_uses_fp8_matmul("diffusion", component_plan)
    assert not component.ltxav_component_uses_fp8_matmul("connectors", component_plan)
    full_precision = replace(quant, full_precision_matmul=True)
    assert not component.ltxav_component_uses_fp8_matmul(
        "diffusion",
        replace(component_plan, quant={full_precision.layer: full_precision}),
    )


@pytest.mark.parametrize(
    "role",
    ("diffusion", "gemma3_12b", "text_projection", "connectors", "latent_upscaler", "vae"),
)
def test_split_component_binds_role_asset_identity_and_compute_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    path = tmp_path / f"{role}.safetensors"
    digest = "blake3:" + "1" * 64
    monkeypatch.setattr(
        component,
        "plan_ltxav_standalone_component",
        lambda _source, selected_role: _component_plan(selected_role, path),
    )

    planned = component.plan_ltxav_split_component(
        cast("WeightSource", _Source(path, 10, digest)),
        role=cast("Any", role),
        path=path,
    )
    identity = component.ltxav_component_runtime_identity(planned, BFLOAT16)
    token = AttentionRouteToken(
        version=1,
        routes=tuple(AttentionRoute(attention_role, "sdpa") for attention_role in ATTENTION_ROLES),
        provider_versions=(("torch", "2.13.0"),),
        adapter_contract_revision="dinkster.attention-kernel.v1",
        device_kind="cpu",
        device_sm=None,
        sdpa_torch_runtime="2.13.0",
        requested_policy="auto",
    )
    routed_identity = component.ltxav_component_runtime_identity(
        planned,
        BFLOAT16,
        attention_route_token=token,
    )

    assert planned.role == role
    assert f"asset_digest={digest}" in planned.component.identity_facts
    assert "asset_size=10" in planned.component.identity_facts
    assert identity != component.ltxav_component_runtime_identity(planned, FLOAT32)
    if role in ("diffusion", "gemma3_12b", "connectors"):
        assert routed_identity != identity
    else:
        assert routed_identity == identity
    ComponentBinding(role, "dinkster.ltxav", identity)


def test_split_component_refuses_path_mismatch_and_missing_asset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "diffusion.safetensors"
    monkeypatch.setattr(
        component,
        "plan_ltxav_standalone_component",
        lambda _source, role: _component_plan(role, path),
    )

    with pytest.raises(component.LTXAVComponentAssemblyError, match="path differs"):
        component.plan_ltxav_split_component(
            cast("WeightSource", _Source(path, 5, "blake3:" + "3" * 64)),
            role="diffusion",
            path=tmp_path / "other.safetensors",
        )
    with pytest.raises(component.LTXAVComponentAssemblyError, match="must carry asset identity"):
        component.plan_ltxav_split_component(
            cast("WeightSource", _BareSource(path)), role="diffusion", path=path
        )


def test_split_audio_codec_binds_shared_asset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "audio-codec.safetensors"
    digest = "blake3:" + "1" * 64
    monkeypatch.setattr(component, "plan_ltxav_audio_codec", lambda _source: _plan(path))

    planned = component.plan_ltxav_split_audio_codec(
        cast("WeightSource", _Source(path, 10, digest)), path=path
    )
    identity = component.ltxav_audio_codec_runtime_identity(planned)

    for plan in planned.identity_components:
        assert f"asset_digest={digest}" in plan.identity_facts
        assert "asset_size=10" in plan.identity_facts
    assert identity.startswith("native:dinkster.ltxav:")


def test_same_geometry_with_different_audio_codec_bytes_has_different_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "audio-codec.safetensors"
    monkeypatch.setattr(component, "plan_ltxav_audio_codec", lambda _source: _plan(path))
    first = component.plan_ltxav_split_audio_codec(
        cast("WeightSource", _Source(path, 10, "blake3:" + "1" * 64)), path=path
    )
    second = component.plan_ltxav_split_audio_codec(
        cast("WeightSource", _Source(path, 10, "blake3:" + "2" * 64)), path=path
    )

    assert component.ltxav_audio_codec_runtime_identity(
        first
    ) != component.ltxav_audio_codec_runtime_identity(second)


def test_split_audio_codec_refuses_path_mismatch_and_missing_asset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "audio-codec.safetensors"
    monkeypatch.setattr(component, "plan_ltxav_audio_codec", lambda _source: _plan(path))

    with pytest.raises(component.LTXAVAudioCodecAssemblyError, match="path differs"):
        component.plan_ltxav_split_audio_codec(
            cast("WeightSource", _Source(path, 5, "blake3:" + "3" * 64)),
            path=tmp_path / "other.safetensors",
        )
    with pytest.raises(component.LTXAVAudioCodecAssemblyError, match="must carry asset identity"):
        component.plan_ltxav_split_audio_codec(cast("WeightSource", _BareSource(path)), path=path)


def test_split_audio_codec_wraps_layout_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "audio-codec.safetensors"

    def refuse(_source: object) -> object:
        raise ValueError("not an exact LTX-2 audio codec")

    monkeypatch.setattr(component, "plan_ltxav_audio_codec", refuse)

    with pytest.raises(component.LTXAVAudioCodecAssemblyError, match="exact LTX-2 audio codec"):
        component.plan_ltxav_split_audio_codec(
            cast("WeightSource", _Source(path, 5, "blake3:" + "4" * 64)), path=path
        )
