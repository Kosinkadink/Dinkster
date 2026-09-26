"""Torch-free host native-dispatch policy tests."""

from __future__ import annotations

import asyncio
import json
import struct
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import dinkster_inference as inference
import pytest
from dinkster_assets import (
    AssetIntegrityError,
    AssetVault,
    LibraryStore,
    digest_bytes,
    digest_file,
)
from dinkster_compat_comfy.native import (
    NATIVE_NODES,
    EmptyMiniMaxH3AV,
    EmptyMiniMaxMusic3LatentAudio,
    LoadZImageControlPatch,
    MiniMaxH3ImageToVideo,
    MiniMaxH3ReferenceToVideo,
)
from dinkster_compat_comfy.native_arm import NATIVE_ARM_NODES, NATIVE_SCHEDULING_NODES
from dinkster_engine import ExecutionSelection
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    LTXAV_19B_CONFIG,
    LTXAV_19B_VAE_CONFIG,
    LTXV_2B_V09_CONFIG,
    LTXV_2B_V09_VAE_CONFIG,
    UMT5_XXL_CONFIG,
    WAN21_T2V_14B,
    WAN21_VAE_CONFIG,
    MiniMaxH3DiTRole,
    MiniMaxMusic3ComponentRole,
    NativeRefusalError,
    TensorGeometry,
    build_runtime_identity,
    default_diffusion_dtype,
    default_text_dtype,
    default_vae_dtype,
    minimax_h3_component_descriptor,
    minimax_h3_dit_layout,
    minimax_h3_dit_provider_facts,
    minimax_h3_dit_runtime_identity,
    triposplat_component_runtime_identity,
)
from dinkster_inference.assembly import (
    ComponentPlan,
    Flux2ComponentRole,
    QwenImageComponentRole,
    TripoSplatComponentRole,
    Wan21StandaloneComponentPlan,
    Wan21StandaloneComponentRole,
)
from dinkster_inference.clip_text import CLIP_L_TEXT_CONFIG
from dinkster_inference.devices import DType
from dinkster_inference.minimax_h3_assembly import (
    MiniMaxH3CommonComponentRole,
    minimax_h3_component_runtime_identity,
)
from dinkster_protocol import ATTENTION_ROLES, AttentionRoute, AttentionRouteToken
from dinkster_schema import NodeSchema, schema_from_wire, schema_signature, schema_to_wire
from dinkster_server import ServerLibrary
from dinkster_values import (
    RESOURCE_HANDLE_TYPE,
    RESOURCE_PRODUCER_ARM_META_KEY,
    ListPayload,
    ResourceHandle,
    TypeRegistry,
    Value,
    ValueMeta,
    register_core_types,
    register_resource_handle_type,
)
from dinkster_values.lists import list_fingerprint
from dinkster_values.model import PyObjPayload

import dinkster.native_policy as native_policy_module
from dinkster.compose import ServingComposer
from dinkster.native_policy import NativeDispatchPolicy, NativePolicyDiagnostic
from dinkster.serve import (
    _native_asset_locator,
    _schedule_legacy_checkpoint_conversion,
)

ARMS = ("compat", {"compat": "compat-tag", "compat@native": "native-default"})
NATIVE_DISPATCH_SCHEMAS = {
    schema.node_type: schema
    for schema in (
        *(node.schema() for node in NATIVE_SCHEDULING_NODES),
        EmptyMiniMaxH3AV.schema(),
        EmptyMiniMaxMusic3LatentAudio.schema(),
        LoadZImageControlPatch.schema(),
        MiniMaxH3ImageToVideo.schema(),
        MiniMaxH3ReferenceToVideo.schema(),
    )
}


def _ignore_diagnostic(_diagnostic: NativePolicyDiagnostic) -> None:
    pass


def _value(value: object) -> Value:
    return Value(
        type_id="dinkster.string",
        fingerprint=str(value),
        meta=ValueMeta(),
        payload=PyObjPayload(value),
    )


def _asset(digest: object, name: object = "model.safetensors") -> Value:
    return Value(
        type_id="dinkster.asset",
        fingerprint=str(digest),
        meta=ValueMeta({"digest": digest, "name": name}),
        payload=PyObjPayload(None),
    )


def test_native_body_may_come_from_cross_pack_provider() -> None:
    policy = NativeDispatchPolicy(
        lambda _digest: None,
        _ignore_diagnostic,
        schemas=lambda: NATIVE_DISPATCH_SCHEMAS,
    )
    selection = asyncio.run(
        policy.select(
            "dinkster.create_hook_lora",
            {},
            (
                "schema-owner",
                {
                    "schema-owner": "schema-tag",
                    "body-provider": "compat-tag",
                    "body-provider@native": "native-tag",
                },
            ),
        )
    )

    assert selection is not None
    assert selection.target == "body-provider@native"
    assert selection.cache_tag == "native-tag"


def _sdpa_route_token(device_sm: int) -> AttentionRouteToken:
    return AttentionRouteToken(
        1,
        tuple(AttentionRoute(role, "sdpa") for role in ATTENTION_ROLES),
        (("torch", "2.13.0"),),
        "dinkster.attention-kernel.v1",
        "cuda",
        device_sm,
        "2.13.0",
        "auto",
    )


def test_load_diffusion_model_preserves_non_h3_owner_fallback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"not-h3")
    digest = digest_file(path)
    policy = NativeDispatchPolicy(
        lambda found: path if found == digest else None, _ignore_diagnostic
    )

    selection = asyncio.run(
        policy.select("dinkster.load_diffusion_model", {"diffusion_model": _asset(digest)}, ARMS)
    )

    assert selection is not None
    assert selection.target == "compat"
    assert selection.cache_tag == "compat-tag"


def _h3_component_plan(role: str) -> ComponentPlan[object]:
    component = {
        "qwen3vl-32b-conditioner": "conditioner",
        "video-vae": "video_vae",
        "audio-vae": "audio_vae",
    }[role]
    return ComponentPlan(
        component=component,
        path=Path(f"/{role}.safetensors"),
        config=None,
        keys={},
        dtypes={},
        quant={},
        identity_facts=(f"role={role}",),
    )


@pytest.mark.parametrize(
    ("node_type", "role", "input_name", "extra", "dtype"),
    (
        (
            "dinkster.load_clip",
            "qwen3vl-32b-conditioner",
            "text_encoder",
            {"type": _value("minimax")},
            FLOAT16,
        ),
        (
            "dinkster.load_clip",
            "qwen3vl-32b-conditioner",
            "text_encoder",
            {"type": _value("wan")},
            FLOAT16,
        ),
        ("dinkster.load_vae", "video-vae", "vae", {}, FLOAT16),
        ("dinkster.load_vae", "audio-vae", "vae", {}, FLOAT32),
    ),
)
def test_generic_loader_selects_native_h3_component_with_plan_identity(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
    role: MiniMaxH3CommonComponentRole,
    input_name: str,
    extra: dict[str, Value],
    dtype: DType,
) -> None:
    digest = "blake3:" + "1" * 64
    plan = _h3_component_plan(role)
    policy = NativeDispatchPolicy(
        lambda _digest: None,
        _ignore_diagnostic,
        minimax_h3_runtime_versions=lambda: {"torch": "2.13.0", "dinkster-kitchen": "0.1.0"},
    )
    monkeypatch.setattr(
        minimax_h3_component_descriptor,
        "plan_minimax_h3_common_component",
        lambda *_args, **_kwargs: plan,
    )
    _stub_registry_probe(
        monkeypatch,
        policy,
        "dinkster.minimax_h3",
        role,
        minimax_h3_component_descriptor.H3ComponentCandidate(
            cast("Any", object()), plan.path, role, digest, 8
        ),
    )

    selection = asyncio.run(policy.select(node_type, {input_name: _asset(digest), **extra}, ARMS))

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == minimax_h3_component_runtime_identity(
        plan,
        role,
        dtype,
    )
    assert selection.diffusion_dtype == "unloaded"
    assert selection.text_dtype == ("float16" if node_type == "dinkster.load_clip" else "unloaded")
    assert selection.vae_dtype == (dtype.name if node_type == "dinkster.load_vae" else "unloaded")


def _music3_component_plan(role: MiniMaxMusic3ComponentRole) -> ComponentPlan[object]:
    return ComponentPlan(
        component=role,
        path=Path(f"/{role}.safetensors"),
        config=None,
        keys={},
        dtypes={},
        quant={},
        identity_facts=(f"role={role}",),
    )


@pytest.mark.parametrize(
    ("node_type", "role", "input_name", "extra", "dtype", "dtype_field"),
    (
        (
            "dinkster.load_diffusion_model",
            "diffusion",
            "diffusion_model",
            {},
            BFLOAT16,
            "diffusion_dtype",
        ),
        (
            "dinkster.load_clip",
            "text",
            "text_encoder",
            {"type": _value("minimax")},
            BFLOAT16,
            "text_dtype",
        ),
        ("dinkster.load_vae", "vae", "vae", {}, FLOAT32, "vae_dtype"),
    ),
)
def test_generic_loaders_select_native_music3_components(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
    role: MiniMaxMusic3ComponentRole,
    input_name: str,
    extra: dict[str, Value],
    dtype: DType,
    dtype_field: str,
) -> None:
    digest = "blake3:" + "8" * 64
    plan = _music3_component_plan(role)
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_registry_probe(monkeypatch, policy, "dinkster.minimax_music3", role, plan)

    selection = asyncio.run(policy.select(node_type, {input_name: _asset(digest), **extra}, ARMS))

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.minimax_music3_component_runtime_identity(
        plan,
        role,
        dtype,
    )
    assert getattr(selection, dtype_field) == dtype.name
    assert (
        sum(
            value != "unloaded"
            for value in (
                selection.diffusion_dtype,
                selection.text_dtype,
                selection.vae_dtype,
            )
        )
        == 1
    )


@pytest.mark.parametrize(
    ("node_type", "role", "input_name", "extra", "message"),
    (
        (
            "dinkster.load_diffusion_model",
            "text",
            "diffusion_model",
            {},
            "role mismatch: detected MiniMax Music 3 text",
        ),
        (
            "dinkster.load_clip",
            "vae",
            "text_encoder",
            {"type": _value("minimax")},
            "no matching component architecture; detected MiniMax Music 3 vae",
        ),
        (
            "dinkster.load_vae",
            "text",
            "vae",
            {},
            "no matching component architecture; detected MiniMax Music 3 text",
        ),
    ),
)
def test_generic_loaders_refuse_wrong_music3_component_roles(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
    role: MiniMaxMusic3ComponentRole,
    input_name: str,
    extra: dict[str, Value],
    message: str,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_registry_probe(
        monkeypatch, policy, "dinkster.minimax_music3", role, _music3_component_plan(role)
    )

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(
            policy.select(
                node_type,
                {input_name: _asset("blake3:" + "9" * 64), **extra},
                ARMS,
            )
        )


def _qwen_component_plan(role: QwenImageComponentRole) -> ComponentPlan[object]:
    return ComponentPlan(
        component=role,
        path=Path(f"/{role}.safetensors"),
        config=None,
        keys={},
        dtypes={},
        quant={},
        identity_facts=(f"role={role}",),
    )


@pytest.mark.parametrize(
    ("node_type", "role", "input_name", "extra", "dtype_field"),
    (
        (
            "dinkster.load_clip",
            "qwen2_5_vl_7b",
            "text_encoder",
            {"type": _value("qwen_image")},
            "text_dtype",
        ),
        ("dinkster.load_vae", "vae", "vae", {}, "vae_dtype"),
    ),
)
def test_generic_loader_selects_native_qwen_component_with_plan_identity(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
    role: QwenImageComponentRole,
    input_name: str,
    extra: dict[str, Value],
    dtype_field: str,
) -> None:
    digest = "blake3:" + "8" * 64
    plan = _qwen_component_plan(role)
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_registry_probe(monkeypatch, policy, "dinkster.qwen_image", role, plan)

    selection = asyncio.run(policy.select(node_type, {input_name: _asset(digest), **extra}, ARMS))

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.qwen_image_component_runtime_identity(
        plan, role, BFLOAT16
    )
    assert getattr(selection, dtype_field) == "bfloat16"


def test_anima_shared_vae_keeps_canonical_qwen_provider_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "blake3:" + "7" * 64
    qwen_plan = _qwen_component_plan("vae")
    wan_plan = _wan21_component_plan("vae")
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    from dinkster_inference.component_catalog import default_component_registry
    from dinkster_inference.component_registry import DetectedComponents

    registry = default_component_registry()
    qwen = registry.get("dinkster.qwen_image")
    wan = registry.get("dinkster.wan21")
    assert qwen is not None and wan is not None
    monkeypatch.setattr(
        policy,
        "_probe_components_transaction",
        lambda _digest: (
            DetectedComponents(wan, (("vae", wan_plan),)),
            DetectedComponents(qwen, (("vae", qwen_plan),)),
        ),
    )

    selection = asyncio.run(
        policy.select(
            "dinkster.load_vae",
            {"vae": _asset(digest, "qwen_image_vae.safetensors")},
            ARMS,
        )
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.qwen_image_component_runtime_identity(
        qwen_plan, "vae", BFLOAT16
    )
    assert selection.cache_tag != inference.wan21_component_runtime_identity(wan_plan, BFLOAT16)
    assert selection.diffusion_dtype == selection.text_dtype == "unloaded"
    assert selection.vae_dtype == "bfloat16"


def test_generic_diffusion_loader_selects_native_qwen_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "blake3:" + "9" * 64
    plan = _qwen_component_plan("diffusion")
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_registry_probe(monkeypatch, policy, "dinkster.qwen_image", "diffusion", plan)

    selection = asyncio.run(
        policy.select("dinkster.load_diffusion_model", {"diffusion_model": _asset(digest)}, ARMS)
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.qwen_image_component_runtime_identity(
        plan, "diffusion", BFLOAT16
    )
    assert selection.diffusion_dtype == "bfloat16"
    assert selection.text_dtype == selection.vae_dtype == "unloaded"


def _wan21_component_plan(role: Wan21StandaloneComponentRole) -> Wan21StandaloneComponentPlan:
    config = {
        "diffusion": WAN21_T2V_14B,
        "umt5xxl": UMT5_XXL_CONFIG,
        "vae": WAN21_VAE_CONFIG,
    }[role]
    plan = ComponentPlan(
        component=role,
        path=Path(f"/{role}.safetensors"),
        config=config,
        keys={},
        dtypes={},
        quant={},
        identity_facts=(f"role={role}",),
    )
    return Wan21StandaloneComponentPlan(
        role,
        cast("Any", plan),
        "spiece_model" if role == "umt5xxl" else "",
    )


@pytest.mark.parametrize(
    ("node_type", "role", "input_name", "extra", "dtype_field", "dtype"),
    (
        (
            "dinkster.load_clip",
            "umt5xxl",
            "text_encoder",
            {"type": _value("wan")},
            "text_dtype",
            FLOAT32,
        ),
        ("dinkster.load_vae", "vae", "vae", {}, "vae_dtype", BFLOAT16),
    ),
)
def test_generic_loader_selects_native_wan21_component_with_asset_identity(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
    role: Wan21StandaloneComponentRole,
    input_name: str,
    extra: dict[str, Value],
    dtype_field: str,
    dtype: DType,
) -> None:
    digest = "blake3:" + "a" * 64
    planned = _wan21_component_plan(role)
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_registry_probe(monkeypatch, policy, "dinkster.wan21", role, planned)

    selection = asyncio.run(policy.select(node_type, {input_name: _asset(digest), **extra}, ARMS))

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.wan21_component_runtime_identity(planned, dtype)
    assert getattr(selection, dtype_field) == dtype.name


def test_generic_diffusion_loader_selects_exact_wan21_t2v_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "blake3:" + "b" * 64
    planned = _wan21_component_plan("diffusion")
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_registry_probe(monkeypatch, policy, "dinkster.wan21", "diffusion", planned)

    selection = asyncio.run(
        policy.select("dinkster.load_diffusion_model", {"diffusion_model": _asset(digest)}, ARMS)
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.wan21_component_runtime_identity(planned, BFLOAT16)
    assert selection.diffusion_dtype == "bfloat16"
    assert selection.text_dtype == selection.vae_dtype == "unloaded"


def test_qwen_component_probe_fails_closed_for_wrong_asset_digest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "truncated.safetensors"
    path.write_bytes(b"truncated")
    digest = "blake3:" + "7" * 64
    policy = NativeDispatchPolicy(
        lambda found: path if found == digest else None, _ignore_diagnostic
    )

    with pytest.raises(AssetIntegrityError):
        asyncio.run(
            policy.select(
                "dinkster.load_diffusion_model",
                {"diffusion_model": _asset(digest)},
                ARMS,
            )
        )


@pytest.mark.parametrize(
    ("node_type", "inputs"),
    (
        (
            "dinkster.load_clip",
            {"text_encoder": _asset("blake3:" + "2" * 64), "type": _value("wan")},
        ),
        ("dinkster.load_vae", {"vae": _asset("blake3:" + "3" * 64)}),
    ),
)
def test_generic_loader_preserves_non_h3_owner_fallback(
    node_type: str,
    inputs: dict[str, Value],
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)

    selection = asyncio.run(policy.select(node_type, inputs, ARMS))

    assert selection is not None
    assert selection.target == "compat"
    assert selection.cache_tag == "compat-tag"


@pytest.mark.parametrize(
    ("node_type", "inputs", "role", "message"),
    (
        (
            "dinkster.load_clip",
            {
                "text_encoder": _asset("blake3:" + "5" * 64),
                "type": _value("minimax"),
            },
            "video-vae",
            "dinkster.load_clip: text loading: no matching component architecture; "
            "detected MiniMax H3 video-vae",
        ),
        (
            "dinkster.load_vae",
            {"vae": _asset("blake3:" + "7" * 64)},
            "qwen3vl-32b-conditioner",
            "dinkster.load_vae: codec loading: no matching component architecture; "
            "detected MiniMax H3 qwen3vl-32b-conditioner",
        ),
    ),
)
def test_generic_loader_reports_h3_role_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
    inputs: dict[str, Value],
    role: str,
    message: str,
) -> None:
    plan = _h3_component_plan(role)
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_registry_probe(monkeypatch, policy, "dinkster.minimax_h3", role, plan)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(policy.select(node_type, inputs, ARMS))
    assert str(caught.value) == message


def _flux2_planned(role: str, family_id: str) -> inference.Flux2PlannedComponent:
    return inference.Flux2PlannedComponent(
        cast("Flux2ComponentRole", role),
        family_id,
        ComponentPlan(
            component=role,
            path=Path(f"/{role}.safetensors"),
            config=None,
            keys={},
            dtypes={},
            quant={},
            identity_facts=(f"role={role}",),
        ),
    )


def _anima_planned(role: str) -> ComponentPlan[object]:
    return ComponentPlan(
        component=role,
        path=Path(f"/{role}.safetensors"),
        config=None,
        keys={},
        dtypes={},
        quant={},
        identity_facts=(f"role={role}",),
    )


def _chroma_planned(role: str, family_id: str = "dinkster.chroma") -> ComponentPlan[object]:
    return ComponentPlan(
        component=role,
        path=Path(f"/{role}.safetensors"),
        config=SimpleNamespace(family_id=family_id),
        keys={},
        dtypes={},
        quant={},
        identity_facts=(f"family={family_id}", f"role={role}"),
    )


def _stub_registry_probe(
    monkeypatch: pytest.MonkeyPatch,
    policy: NativeDispatchPolicy,
    family_id: str,
    role: str | None,
    planned: Any,
) -> None:
    from dinkster_inference.component_catalog import default_component_registry
    from dinkster_inference.component_registry import DetectedComponents

    descriptor = default_component_registry().get(family_id)
    assert descriptor is not None
    matches = () if role is None else (DetectedComponents(descriptor, ((role, planned),)),)
    monkeypatch.setattr(policy, "_probe_components_transaction", lambda _digest: matches)


def _stub_chroma_probe(
    monkeypatch: pytest.MonkeyPatch,
    policy: NativeDispatchPolicy,
    role: str | None,
    family_id: str = "dinkster.chroma",
) -> ComponentPlan[object] | None:
    planned = None if role is None else _chroma_planned(role, family_id)
    _stub_registry_probe(monkeypatch, policy, family_id, role, planned)
    return planned


def test_load_clip_type_chroma_selects_native_t5xxl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "blake3:" + "9" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_chroma_probe(monkeypatch, policy, "t5xxl")
    assert planned is not None
    token = _sdpa_route_token(120)

    selection = asyncio.run(
        policy.select(
            "dinkster.load_clip",
            {"text_encoder": _asset(digest), "type": _value("chroma")},
            ARMS,
            attention_routes={"compat@native": token},
        )
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.chroma_component_runtime_identity(
        planned,
        "t5xxl",
        FLOAT32,
        attention_policy="auto",
        attention_route_token=token,
    )
    assert selection.attention_route_token == token
    assert selection.diffusion_dtype == "unloaded"
    assert selection.text_dtype == "float32"
    assert selection.vae_dtype == "unloaded"


def test_load_clip_type_chroma_refuses_non_t5_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_chroma_probe(monkeypatch, policy, "vae")

    with pytest.raises(
        RuntimeError, match="no matching component architecture; detected Chroma vae"
    ):
        asyncio.run(
            policy.select(
                "dinkster.load_clip",
                {
                    "text_encoder": _asset("blake3:" + "a" * 64),
                    "type": _value("chroma"),
                },
                ARMS,
            )
        )


@pytest.mark.parametrize("family_id", ("dinkster.chroma", "dinkster.chroma_radiance"))
@pytest.mark.parametrize(
    "weight_dtype", [None, "default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]
)
def test_load_diffusion_model_selects_native_chroma_family(
    monkeypatch: pytest.MonkeyPatch,
    family_id: str,
    weight_dtype: str | None,
) -> None:
    digest = "blake3:" + "b" * 64
    diagnostics: list[NativePolicyDiagnostic] = []
    policy = NativeDispatchPolicy(lambda _digest: None, diagnostics.append)
    planned = _stub_chroma_probe(monkeypatch, policy, "diffusion", family_id)
    assert planned is not None
    token = _sdpa_route_token(120)
    inputs = {"diffusion_model": _asset(digest)}
    if weight_dtype is not None:
        inputs["weight_dtype"] = _value(weight_dtype)

    selection = asyncio.run(
        policy.select(
            "dinkster.load_diffusion_model",
            inputs,
            ARMS,
            attention_routes={"compat@native": token},
        )
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.chroma_component_runtime_identity(
        planned,
        "diffusion",
        BFLOAT16,
        attention_policy="auto",
        attention_route_token=token,
    )
    assert selection.attention_route_token == token
    assert selection.diffusion_dtype == "bfloat16"
    assert selection.text_dtype == selection.vae_dtype == "unloaded"
    assert diagnostics == []


def test_load_vae_selects_native_chroma_flux_kl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "blake3:" + "c" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_chroma_probe(monkeypatch, policy, "vae")
    assert planned is not None
    token = _sdpa_route_token(120)

    selection = asyncio.run(
        policy.select(
            "dinkster.load_vae",
            {"vae": _asset(digest)},
            ARMS,
            attention_routes={"compat@native": token},
        )
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.chroma_component_runtime_identity(
        planned,
        "vae",
        BFLOAT16,
        attention_policy="auto",
        attention_route_token=token,
    )
    assert selection.attention_route_token == token
    assert selection.diffusion_dtype == "unloaded"
    assert selection.text_dtype == "unloaded"
    assert selection.vae_dtype == "bfloat16"


def test_load_vae_selects_native_chroma_radiance_pixel_codec() -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)

    selection = asyncio.run(policy.select("dinkster.load_vae", {"pixel_space": _value(True)}, ARMS))

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == native_policy_module.build_runtime_identity_from_facts(
        "dinkster.chroma_radiance",
        ("family=dinkster.chroma_radiance", "component=pixel_space"),
        diffusion_dtype="unloaded",
        text_dtype="unloaded",
        vae_dtype="bfloat16",
        fp8_matmul=False,
    )
    assert selection.diffusion_dtype == "unloaded"
    assert selection.text_dtype == "unloaded"
    assert selection.vae_dtype == "bfloat16"


def _stub_anima_probe(
    monkeypatch: pytest.MonkeyPatch,
    policy: NativeDispatchPolicy,
    role: str | None,
) -> ComponentPlan[object] | None:
    planned = None if role is None else _anima_planned(role)
    _stub_registry_probe(monkeypatch, policy, "dinkster.anima", role, planned)
    return planned


def _stub_lumina2_probe(
    monkeypatch: pytest.MonkeyPatch,
    policy: NativeDispatchPolicy,
    role: str | None,
) -> ComponentPlan[object] | None:
    planned = None if role is None else _anima_planned(role)
    _stub_registry_probe(monkeypatch, policy, "dinkster.lumina2", role, planned)
    return planned


def test_load_clip_type_lumina2_selects_exact_gemma2_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "blake3:" + "4" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_lumina2_probe(monkeypatch, policy, "gemma2_2b")
    assert planned is not None

    selection = asyncio.run(
        policy.select(
            "dinkster.load_clip",
            {"text_encoder": _asset(digest), "type": _value("lumina2")},
            ARMS,
        )
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.lumina2_component_runtime_identity(
        planned, "gemma2_2b", FLOAT32
    )
    assert (selection.diffusion_dtype, selection.text_dtype, selection.vae_dtype) == (
        "unloaded",
        "float32",
        "unloaded",
    )


def test_load_clip_detects_lumina2_component_under_other_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_lumina2_probe(monkeypatch, policy, "gemma2_2b")
    assert planned is not None

    selection = asyncio.run(
        policy.select(
            "dinkster.load_clip",
            {
                "text_encoder": _asset("blake3:" + "5" * 64),
                "type": _value("stable_diffusion"),
            },
            ARMS,
        )
    )
    assert selection is not None and selection.target == "compat@native"
    assert selection.cache_tag == inference.lumina2_component_runtime_identity(
        planned, "gemma2_2b", FLOAT32
    )


def test_load_vae_selects_exact_lumina2_vae_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "blake3:" + "6" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_lumina2_probe(monkeypatch, policy, "vae")
    assert planned is not None

    selection = asyncio.run(policy.select("dinkster.load_vae", {"vae": _asset(digest)}, ARMS))

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.lumina2_component_runtime_identity(
        planned, "vae", BFLOAT16
    )
    assert (selection.diffusion_dtype, selection.text_dtype, selection.vae_dtype) == (
        "unloaded",
        "unloaded",
        "bfloat16",
    )


def test_load_diffusion_model_selects_exact_lumina2_diffusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "blake3:" + "7" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_lumina2_probe(monkeypatch, policy, "diffusion")
    assert planned is not None

    selection = asyncio.run(
        policy.select("dinkster.load_diffusion_model", {"diffusion_model": _asset(digest)}, ARMS)
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.lumina2_component_runtime_identity(
        planned, "diffusion", BFLOAT16
    )
    assert (selection.diffusion_dtype, selection.text_dtype, selection.vae_dtype) == (
        "bfloat16",
        "unloaded",
        "unloaded",
    )


@pytest.mark.parametrize("clip_type", ("anima", "stable_diffusion"))
def test_load_clip_anima_compatible_types_select_native_text_encoder(
    monkeypatch: pytest.MonkeyPatch, clip_type: str
) -> None:
    digest = "blake3:" + "6" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_anima_probe(monkeypatch, policy, "qwen3_06b")
    assert planned is not None

    selection = asyncio.run(
        policy.select(
            "dinkster.load_clip",
            {"text_encoder": _asset(digest), "type": _value(clip_type)},
            ARMS,
        )
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.anima_component_runtime_identity(
        planned, "qwen3_06b", FLOAT32
    )
    assert selection.diffusion_dtype == "unloaded"
    assert selection.text_dtype == "float32"
    assert selection.vae_dtype == "unloaded"


def test_load_clip_stable_diffusion_preserves_non_anima_owner_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_anima_probe(monkeypatch, policy, None)

    selection = asyncio.run(
        policy.select(
            "dinkster.load_clip",
            {
                "text_encoder": _asset("blake3:" + "7" * 64),
                "type": _value("stable_diffusion"),
            },
            ARMS,
        )
    )

    assert selection is not None
    assert selection.target == "compat"
    assert selection.cache_tag == "compat-tag"


def test_load_clip_type_anima_refuses_wrong_component_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_anima_probe(monkeypatch, policy, "diffusion")

    with pytest.raises(
        RuntimeError, match="no matching component architecture; detected Anima diffusion"
    ):
        asyncio.run(
            policy.select(
                "dinkster.load_clip",
                {
                    "text_encoder": _asset("blake3:" + "7" * 64),
                    "type": _value("anima"),
                },
                ARMS,
            )
        )


def test_load_diffusion_model_selects_native_anima_diffusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "blake3:" + "8" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_anima_probe(monkeypatch, policy, "diffusion")
    assert planned is not None

    selection = asyncio.run(
        policy.select("dinkster.load_diffusion_model", {"diffusion_model": _asset(digest)}, ARMS)
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.anima_component_runtime_identity(
        planned, "diffusion", BFLOAT16
    )
    assert selection.diffusion_dtype == "bfloat16"
    assert selection.text_dtype == selection.vae_dtype == "unloaded"


def _krea2_planned(role: str) -> ComponentPlan[object]:
    return ComponentPlan(
        component=role,
        path=Path(f"/{role}.safetensors"),
        config=None,
        keys={},
        dtypes={},
        quant={},
        identity_facts=(f"role={role}",),
    )


def _stub_krea2_probe(
    monkeypatch: pytest.MonkeyPatch,
    policy: NativeDispatchPolicy,
    role: str | None,
) -> ComponentPlan[object] | None:
    planned = None if role is None else _krea2_planned(role)
    _stub_registry_probe(monkeypatch, policy, "dinkster.krea2", role, planned)
    return planned


@pytest.mark.parametrize("clip_type", ("krea2", "stable_diffusion"))
def test_load_clip_krea2_compatible_types_select_native_text_encoder(
    monkeypatch: pytest.MonkeyPatch, clip_type: str
) -> None:
    digest = "blake3:" + "6" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_krea2_probe(monkeypatch, policy, "qwen3vl_4b")
    assert planned is not None

    selection = asyncio.run(
        policy.select(
            "dinkster.load_clip",
            {"text_encoder": _asset(digest), "type": _value(clip_type)},
            ARMS,
        )
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.krea2_component_runtime_identity(
        planned, "qwen3vl_4b", FLOAT32
    )
    assert selection.diffusion_dtype == "unloaded"
    assert selection.text_dtype == "float32"
    assert selection.vae_dtype == "unloaded"


def test_load_clip_stable_diffusion_preserves_non_krea2_owner_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_krea2_probe(monkeypatch, policy, None)

    selection = asyncio.run(
        policy.select(
            "dinkster.load_clip",
            {
                "text_encoder": _asset("blake3:" + "7" * 64),
                "type": _value("stable_diffusion"),
            },
            ARMS,
        )
    )

    assert selection is not None
    assert selection.target == "compat"
    assert selection.cache_tag == "compat-tag"


def test_load_clip_type_krea2_refuses_wrong_component_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_krea2_probe(monkeypatch, policy, "diffusion")

    with pytest.raises(
        RuntimeError, match="no matching component architecture; detected Krea 2 diffusion"
    ):
        asyncio.run(
            policy.select(
                "dinkster.load_clip",
                {
                    "text_encoder": _asset("blake3:" + "7" * 64),
                    "type": _value("krea2"),
                },
                ARMS,
            )
        )


def test_load_vae_refuses_krea2_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_krea2_probe(monkeypatch, policy, "qwen3vl_4b")

    with pytest.raises(
        RuntimeError, match="no matching component architecture; detected Krea 2 qwen3vl_4b"
    ):
        asyncio.run(
            policy.select(
                "dinkster.load_vae",
                {"vae": _asset("blake3:" + "7" * 64)},
                ARMS,
            )
        )


def test_load_diffusion_model_selects_native_krea2_diffusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "blake3:" + "8" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_krea2_probe(monkeypatch, policy, "diffusion")
    assert planned is not None

    selection = asyncio.run(
        policy.select("dinkster.load_diffusion_model", {"diffusion_model": _asset(digest)}, ARMS)
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.krea2_component_runtime_identity(
        planned, "diffusion", BFLOAT16
    )
    assert selection.diffusion_dtype == "bfloat16"
    assert selection.text_dtype == selection.vae_dtype == "unloaded"


def _stub_ideogram4_probe(
    monkeypatch: pytest.MonkeyPatch,
    policy: NativeDispatchPolicy,
    role: str | None,
) -> ComponentPlan[object] | None:
    planned = None if role is None else _krea2_planned(role)
    _stub_registry_probe(monkeypatch, policy, "dinkster.ideogram4", role, planned)
    return planned


@pytest.mark.parametrize("clip_type", ("ideogram4", "stable_diffusion"))
def test_load_clip_ideogram4_compatible_types_select_native_text_encoder(
    monkeypatch: pytest.MonkeyPatch, clip_type: str
) -> None:
    digest = "blake3:" + "9" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_ideogram4_probe(monkeypatch, policy, "qwen3vl_8b")
    assert planned is not None

    selection = asyncio.run(
        policy.select(
            "dinkster.load_clip",
            {"text_encoder": _asset(digest), "type": _value(clip_type)},
            ARMS,
        )
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.ideogram4_component_runtime_identity(
        planned, "qwen3vl_8b", FLOAT32
    )
    assert selection.diffusion_dtype == "unloaded"
    assert selection.text_dtype == "float32"
    assert selection.vae_dtype == "unloaded"


def test_load_clip_type_ideogram4_refuses_wrong_component_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_ideogram4_probe(monkeypatch, policy, "diffusion")

    with pytest.raises(
        RuntimeError, match="no matching component architecture; detected Ideogram 4 diffusion"
    ):
        asyncio.run(
            policy.select(
                "dinkster.load_clip",
                {
                    "text_encoder": _asset("blake3:" + "a" * 64),
                    "type": _value("ideogram4"),
                },
                ARMS,
            )
        )


def test_load_vae_refuses_ideogram4_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_ideogram4_probe(monkeypatch, policy, "qwen3vl_8b")

    with pytest.raises(
        RuntimeError, match="no matching component architecture; detected Ideogram 4 qwen3vl_8b"
    ):
        asyncio.run(
            policy.select(
                "dinkster.load_vae",
                {"vae": _asset("blake3:" + "b" * 64)},
                ARMS,
            )
        )


def test_load_diffusion_model_selects_native_ideogram4_diffusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "blake3:" + "c" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_ideogram4_probe(monkeypatch, policy, "diffusion")
    assert planned is not None

    selection = asyncio.run(
        policy.select("dinkster.load_diffusion_model", {"diffusion_model": _asset(digest)}, ARMS)
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.ideogram4_component_runtime_identity(
        planned, "diffusion", BFLOAT16
    )
    assert selection.diffusion_dtype == "bfloat16"
    assert selection.text_dtype == selection.vae_dtype == "unloaded"


def _stub_seedvr2_probe(
    monkeypatch: pytest.MonkeyPatch,
    policy: NativeDispatchPolicy,
    role: str | None,
) -> ComponentPlan[object] | None:
    planned = None if role is None else _anima_planned(role)
    _stub_registry_probe(monkeypatch, policy, "dinkster.seedvr2", role, planned)
    return planned


def test_load_diffusion_model_selects_native_seedvr2_diffusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "blake3:" + "8" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_seedvr2_probe(monkeypatch, policy, "diffusion")
    assert planned is not None

    selection = asyncio.run(
        policy.select("dinkster.load_diffusion_model", {"diffusion_model": _asset(digest)}, ARMS)
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.seedvr2_component_runtime_identity(
        planned, "diffusion", BFLOAT16
    )
    assert selection.diffusion_dtype == "bfloat16"
    assert selection.text_dtype == selection.vae_dtype == "unloaded"


def test_load_vae_selects_native_seedvr2_vae(monkeypatch: pytest.MonkeyPatch) -> None:
    digest = "blake3:" + "9" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_seedvr2_probe(monkeypatch, policy, "vae")
    assert planned is not None

    selection = asyncio.run(policy.select("dinkster.load_vae", {"vae": _asset(digest)}, ARMS))

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.seedvr2_component_runtime_identity(
        planned, "vae", FLOAT16
    )
    assert selection.diffusion_dtype == selection.text_dtype == "unloaded"
    assert selection.vae_dtype == "float16"


@pytest.mark.parametrize(
    ("node_type", "input_name", "role", "message"),
    (
        (
            "dinkster.load_diffusion_model",
            "diffusion_model",
            "vae",
            "role mismatch: detected SeedVR2 vae",
        ),
        (
            "dinkster.load_vae",
            "vae",
            "diffusion",
            "no matching component architecture; detected SeedVR2 diffusion",
        ),
    ),
)
def test_seedvr2_generic_loaders_refuse_wrong_component_role(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
    input_name: str,
    role: str,
    message: str,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_seedvr2_probe(monkeypatch, policy, role)

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(
            policy.select(
                node_type,
                {input_name: _asset("blake3:" + "a" * 64)},
                ARMS,
            )
        )


def _stub_flux2_probe(
    monkeypatch: pytest.MonkeyPatch,
    policy: NativeDispatchPolicy,
    role: str | None,
    family_id: str = "dinkster.flux2_dev",
) -> inference.Flux2PlannedComponent | None:
    planned = None if role is None else _flux2_planned(role, family_id)
    _stub_registry_probe(monkeypatch, policy, family_id, role, planned)
    return planned


def _triposplat_planned(role: str) -> inference.TripoSplatPlannedComponent:
    return inference.TripoSplatPlannedComponent(
        cast("TripoSplatComponentRole", role),
        ComponentPlan(
            component=role,
            path=Path(f"/{role}.safetensors"),
            config=None,
            keys={},
            dtypes={},
            quant={},
            identity_facts=(f"role={role}",),
        ),
    )


def _stub_triposplat_probe(
    monkeypatch: pytest.MonkeyPatch,
    policy: NativeDispatchPolicy,
    role: str | None,
) -> inference.TripoSplatPlannedComponent | None:
    planned = None if role is None else _triposplat_planned(role)
    _stub_registry_probe(monkeypatch, policy, "dinkster.triposplat", role, planned)
    return planned


@pytest.mark.parametrize(
    ("role", "family_id"),
    (
        ("mistral3_24b", "dinkster.flux2_dev"),
        ("qwen3_8b", "dinkster.flux2_klein_9b"),
        ("qwen3_4b", "dinkster.flux2_klein_4b"),
    ),
)
def test_load_clip_type_flux2_selects_native_text_encoder(
    monkeypatch: pytest.MonkeyPatch, role: str, family_id: str
) -> None:
    digest = "blake3:" + "a" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_flux2_probe(monkeypatch, policy, role, family_id)
    assert planned is not None

    selection = asyncio.run(
        policy.select(
            "dinkster.load_clip",
            {"text_encoder": _asset(digest), "type": _value("flux2")},
            ARMS,
        )
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.flux2_component_runtime_identity(planned, BFLOAT16)
    assert selection.diffusion_dtype == "unloaded"
    assert selection.text_dtype == "bfloat16"
    assert selection.vae_dtype == "unloaded"


@pytest.mark.parametrize("role", ["vae", "diffusion"])
def test_load_clip_type_flux2_refuses_non_text_assets(
    monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    family = "dinkster.flux2" if role == "vae" else "dinkster.flux2_dev"
    _stub_flux2_probe(monkeypatch, policy, role, family)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            policy.select(
                "dinkster.load_clip",
                {"text_encoder": _asset("blake3:" + "b" * 64), "type": _value("flux2")},
                ARMS,
            )
        )
    assert str(caught.value) == (
        "dinkster.load_clip: text loading: no matching component architecture; "
        f"detected Flux2 Dev {role}"
    )


def test_load_vae_selects_native_flux2_vae_in_bfloat16(monkeypatch: pytest.MonkeyPatch) -> None:
    digest = "blake3:" + "c" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_flux2_probe(monkeypatch, policy, "vae", "dinkster.flux2")
    assert planned is not None

    selection = asyncio.run(policy.select("dinkster.load_vae", {"vae": _asset(digest)}, ARMS))

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.flux2_component_runtime_identity(planned, BFLOAT16)
    assert selection.diffusion_dtype == "unloaded"
    assert selection.text_dtype == "unloaded"
    assert selection.vae_dtype == "bfloat16"


def test_load_vae_refuses_flux2_non_vae_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_flux2_probe(monkeypatch, policy, "qwen3_4b", "dinkster.flux2_klein_4b")

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(policy.select("dinkster.load_vae", {"vae": _asset("blake3:" + "d" * 64)}, ARMS))
    assert str(caught.value) == (
        "dinkster.load_vae: codec loading: no matching component architecture; "
        "detected Flux2 Klein 4B qwen3_4b"
    )


def test_load_vae_preserves_owner_fallback_when_no_family_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_flux2_probe(monkeypatch, policy, None)

    selection = asyncio.run(
        policy.select("dinkster.load_vae", {"vae": _asset("blake3:" + "e" * 64)}, ARMS)
    )

    assert selection is not None
    assert selection.target == "compat"
    assert selection.cache_tag == "compat-tag"


def test_load_diffusion_model_selects_native_flux2_diffusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "blake3:" + "f" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = _stub_flux2_probe(monkeypatch, policy, "diffusion", "dinkster.flux2_dev")
    assert planned is not None

    selection = asyncio.run(
        policy.select("dinkster.load_diffusion_model", {"diffusion_model": _asset(digest)}, ARMS)
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.flux2_component_runtime_identity(planned, BFLOAT16)
    assert selection.diffusion_dtype == "bfloat16"
    assert selection.text_dtype == selection.vae_dtype == "unloaded"


def test_load_diffusion_model_refuses_flux2_non_diffusion_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_flux2_probe(monkeypatch, policy, "vae", "dinkster.flux2")

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            policy.select(
                "dinkster.load_diffusion_model",
                {"diffusion_model": _asset("blake3:" + "1" * 64)},
                ARMS,
            )
        )
    assert str(caught.value) == (
        "dinkster.load_diffusion_model role mismatch: detected Flux2 Dev vae"
    )


def test_load_diffusion_model_selects_native_triposplat_dit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "blake3:" + "f" * 64
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_flux2_probe(monkeypatch, policy, None)
    planned = _stub_triposplat_probe(monkeypatch, policy, "dit")
    assert planned is not None

    selection = asyncio.run(
        policy.select("dinkster.load_diffusion_model", {"diffusion_model": _asset(digest)}, ARMS)
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == triposplat_component_runtime_identity(planned, BFLOAT16)
    assert selection.diffusion_dtype == "bfloat16"
    assert selection.text_dtype == selection.vae_dtype == "unloaded"


@pytest.mark.parametrize("role", ("dinov3-vision-conditioner", "gaussian-decoder"))
def test_load_diffusion_model_refuses_triposplat_non_dit_roles(
    monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_flux2_probe(monkeypatch, policy, None)
    _stub_triposplat_probe(monkeypatch, policy, role)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            policy.select(
                "dinkster.load_diffusion_model",
                {"diffusion_model": _asset("blake3:" + "1" * 64)},
                ARMS,
            )
        )
    assert str(caught.value) == (
        f"dinkster.load_diffusion_model role mismatch: detected TripoSplat {role}"
    )


def test_triposplat_policy_identity_matches_arm_recipe_identity() -> None:
    import dinkster_compat_comfy.native_arm as native_arm_module
    import dinkster_inference as inference
    from dinkster_assets import AssetRef

    planned = _triposplat_planned("dit")
    loaded = SimpleNamespace(role="dit", family_id=planned.family_id, plan=planned.plan)
    asset = AssetRef("blake3:" + "0" * 64, "dit.safetensors", 0)

    recipe = native_arm_module._component_descriptor("dinkster.triposplat").recipe(
        native_arm_module._weight_source_ref(inference, asset), loaded, "bfloat16"
    )

    assert recipe.runtime_identity == triposplat_component_runtime_identity(planned, BFLOAT16)


def test_ltxav_component_probe_memoizes_each_selected_role_but_never_a_locate_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ltxav-component.safetensors"
    path.write_bytes(b"ltxav-component")
    digest = digest_file(path)
    header_calls: list[str] = []
    plan_calls: list[str] = []
    source = object()
    planned = object()

    def load_header(_path: Path, *, asset_digest: str, asset_size: int) -> object:
        header_calls.append(asset_digest)
        assert asset_size == path.stat().st_size
        return source

    def plan_component(_source: object, *, role: str, path: Path) -> object:
        del path
        plan_calls.append(role)
        return planned

    monkeypatch.setattr(native_policy_module, "load_safetensors_header", load_header)
    monkeypatch.setattr(native_policy_module, "plan_ltxav_split_component", plan_component)
    located: list[Path | None] = [None]
    policy = NativeDispatchPolicy(lambda _digest: located[0], _ignore_diagnostic)

    assert policy._probe_ltxav_component_transaction(digest, "diffusion") is None
    assert header_calls == []

    located[0] = path
    first = policy._probe_ltxav_component_transaction(digest, "diffusion")
    second = policy._probe_ltxav_component_transaction(digest, "diffusion")
    vae = policy._probe_ltxav_component_transaction(digest, "vae")

    assert first is planned and second is planned and vae is planned
    assert header_calls == [digest, digest]
    assert plan_calls == ["diffusion", "vae"]


def test_ltxav_audio_codec_probe_memoizes_classification_but_never_a_locate_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "audio-codec.safetensors"
    path.write_bytes(b"ltxav-audio-codec")
    digest = digest_file(path)
    header_calls: list[str] = []
    source = object()
    planned = object()

    def load_header(_path: Path, *, asset_digest: str, asset_size: int) -> object:
        header_calls.append(asset_digest)
        assert asset_size == path.stat().st_size
        return source

    monkeypatch.setattr(native_policy_module, "load_safetensors_header", load_header)
    monkeypatch.setattr(
        native_policy_module,
        "plan_ltxav_split_audio_codec",
        lambda candidate, *, path: planned if candidate is source else None,
    )
    located: list[Path | None] = [None]
    policy = NativeDispatchPolicy(lambda _digest: located[0], _ignore_diagnostic)

    miss = policy._probe_ltxav_audio_codec_transaction(digest)
    assert miss.planned is None
    assert header_calls == []

    located[0] = path
    first = policy._probe_ltxav_audio_codec_transaction(digest)
    second = policy._probe_ltxav_audio_codec_transaction(digest)

    assert first.planned is planned
    assert second is first
    assert header_calls == [digest]


@pytest.mark.parametrize("recognized", (False, True))
def test_component_probe_caches_headers_not_locate_misses_and_not_registry_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recognized: bool
) -> None:
    from dinkster_inference import component_catalog
    from dinkster_inference.component_registry import ComponentRegistry
    from test_inference_component_registry import synthetic_descriptor

    descriptor = synthetic_descriptor()
    registry = ComponentRegistry()
    monkeypatch.setattr(component_catalog, "default_component_registry", lambda: registry)
    path = tmp_path / "component.safetensors"
    header = json.dumps(
        {"words.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))
    digest = digest_file(path)
    header_calls: list[str] = []

    def load_header(target: Path, *, asset_digest: str, asset_size: int) -> object:
        header_calls.append(asset_digest)
        return inference.load_safetensors_header(
            target, asset_digest=asset_digest, asset_size=asset_size
        )

    monkeypatch.setattr(native_policy_module, "load_safetensors_header", load_header)
    if recognized:
        registry.register(descriptor)
    located: list[Path | None] = [None]
    policy = NativeDispatchPolicy(lambda _digest: located[0], _ignore_diagnostic)
    assert policy._probe_components_transaction(digest) == ()
    assert header_calls == []
    located[0] = path
    first = policy._probe_components_transaction(digest)
    assert policy._probe_components_transaction(digest) is first
    assert header_calls == [digest]
    if not recognized:
        assert first == ()
        registry.register(descriptor)
        first = policy._probe_components_transaction(digest)
        assert header_calls == [digest, digest]
    assert first[0].descriptor is descriptor
    plan = first[0].plan_for("words")
    assert plan is not None
    assert plan.keys == {"weight": "words.weight"}


def test_component_probe_memoizes_malformed_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    malformed = tmp_path / "malformed.safetensors"
    malformed.write_bytes(b"not-a-safetensors-file")
    malformed_digest = digest_file(malformed)
    header_calls: list[str] = []

    def load_header(target: Path, *, asset_digest: str, asset_size: int) -> object:
        del asset_size
        header_calls.append(asset_digest)
        if target == malformed:
            raise native_policy_module.MalformedSafetensors("truncated header")
        return object()

    monkeypatch.setattr(native_policy_module, "load_safetensors_header", load_header)
    policy = NativeDispatchPolicy(
        lambda found: malformed if found == malformed_digest else None, _ignore_diagnostic
    )

    assert policy._probe_components_transaction(malformed_digest) == ()
    assert policy._probe_components_transaction(malformed_digest) == ()
    assert header_calls == [malformed_digest]


def test_component_probe_does_not_cache_quantization_failure_as_nonmatch(tmp_path: Path) -> None:
    from dinkster_inference.quantization import QuantizationError

    path = tmp_path / "bad-quantization.safetensors"
    header = json.dumps(
        {
            "__metadata__": {"_quantization_metadata": "not json"},
            "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    digest = digest_file(path)
    policy = NativeDispatchPolicy(
        lambda found: path if found == digest else None, _ignore_diagnostic
    )
    for _ in range(2):
        with pytest.raises(QuantizationError, match="malformed _quantization_metadata"):
            policy._probe_components_transaction(digest)


def _safetensors(path: Path) -> Path:
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    return path


def _fake_plan(family_id: str = "dinkster.sd15", config: object = None) -> SimpleNamespace:
    return SimpleNamespace(
        family=SimpleNamespace(id=family_id),
        identity_components=(
            ComponentPlan("clip_l", Path("clip.safetensors"), CLIP_L_TEXT_CONFIG, {}, {}, {}),
        ),
        diffusion=SimpleNamespace(config=config),
    )


def _expected_tag(
    family_id: str = "dinkster.sd15",
    *,
    fp8_matmul: bool = False,
    embedding_binding_digest: str | None = None,
) -> str:
    plan = _fake_plan(family_id)
    return build_runtime_identity(
        plan.family.id,
        plan.identity_components,
        diffusion_dtype=default_diffusion_dtype(plan.family.id),
        text_dtype=default_text_dtype(plan.family.id),
        vae_dtype=default_vae_dtype(plan.family.id),
        fp8_matmul=fp8_matmul,
        registry_token=None,
        embedding_binding_digest=embedding_binding_digest,
    )


def test_checkpoint_dispatch_rotates_with_embedding_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "model.safetensors")
    digest = digest_file(path)
    bindings = iter(("1" * 64, "2" * 64))
    monkeypatch.setattr(native_policy_module, "_embedding_binding_digest", lambda: next(bindings))
    monkeypatch.setattr(native_policy_module, "plan_native", lambda _source: _fake_plan())
    policy = NativeDispatchPolicy(lambda _digest: path, _ignore_diagnostic)

    async def scenario() -> tuple[ExecutionSelection | None, ExecutionSelection | None]:
        first = await policy.select(
            "dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS
        )
        second = await policy.select(
            "dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS
        )
        return first, second

    first, second = asyncio.run(scenario())
    assert first is not None and second is not None
    assert first.cache_tag == _expected_tag(embedding_binding_digest="1" * 64)
    assert second.cache_tag == _expected_tag(embedding_binding_digest="2" * 64)


def test_checkpoint_dispatch_consumes_authenticated_route_before_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "model.safetensors")
    digest = digest_file(path)
    token = AttentionRouteToken(
        1,
        tuple(AttentionRoute(role, "sdpa") for role in ATTENTION_ROLES),
        (("torch", "2.13.0"),),
        "dinkster.attention-kernel.v1",
        "cpu",
        None,
        "2.13.0",
        "auto",
    )
    monkeypatch.setattr(native_policy_module, "plan_native", lambda _source: _fake_plan())
    policy = NativeDispatchPolicy(lambda _digest: path, _ignore_diagnostic)

    selection = asyncio.run(
        policy.select(
            "dinkster.load_checkpoint",
            {"checkpoint": _asset(digest)},
            ARMS,
            attention_routes={"compat@native": token},
        )
    )
    assert selection is not None
    assert selection.attention_route_token == token
    assert selection.attention_policy == "auto"
    assert selection.cache_tag == build_runtime_identity(
        "dinkster.sd15",
        _fake_plan().identity_components,
        diffusion_dtype=default_diffusion_dtype("dinkster.sd15"),
        text_dtype=default_text_dtype("dinkster.sd15"),
        vae_dtype=default_vae_dtype("dinkster.sd15"),
        fp8_matmul=False,
        registry_token=None,
        attention_route_token=token,
    )


def test_concurrent_dispatch_transactions_keep_their_embedding_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "model.safetensors")
    digest = digest_file(path)
    bindings = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(native_policy_module, "_embedding_binding_digest", lambda: next(bindings))
    monkeypatch.setattr(native_policy_module, "plan_native", lambda _source: _fake_plan())

    def identity(*_args: object, **kwargs: object) -> str:
        return f"binding:{kwargs['embedding_binding_digest']}"

    monkeypatch.setattr(native_policy_module, "build_runtime_identity", identity)
    policy = NativeDispatchPolicy(lambda _digest: path, _ignore_diagnostic)

    async def scenario() -> tuple[ExecutionSelection | None, ExecutionSelection | None]:
        return await asyncio.gather(
            policy.select("dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS),
            policy.select("dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS),
        )

    first, second = asyncio.run(scenario())
    assert first is not None and second is not None
    assert {first.cache_tag, second.cache_tag} == {
        "binding:" + "a" * 64,
        "binding:" + "b" * 64,
    }


def test_checkpoint_selects_native_with_exact_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "model.safetensors")
    digest = digest_file(path)
    monkeypatch.setattr(native_policy_module, "plan_native", lambda source: _fake_plan())
    policy = NativeDispatchPolicy(
        lambda candidate: path if candidate == digest else None, _ignore_diagnostic
    )

    selection = asyncio.run(
        policy.select("dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS)
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == _expected_tag()
    assert selection.fp8_matmul is False


def test_lumina2_checkpoint_dispatch_uses_real_plan_without_embedding_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference import plan_native

    from tests.test_inference_lumina2 import _checkpoint_geometries
    from tests.test_inference_lumina2 import _Source as CheckpointSource

    path = _safetensors(tmp_path / "lumina2.safetensors")
    source = CheckpointSource(path, _checkpoint_geometries())
    digest = digest_file(path)
    plan = plan_native(source)
    monkeypatch.setattr(
        native_policy_module, "load_safetensors_header", lambda _path, **_kw: source
    )
    monkeypatch.setattr(native_policy_module, "_embedding_binding_digest", lambda: "a" * 64)
    policy = NativeDispatchPolicy(lambda _digest: path, _ignore_diagnostic)

    selection = asyncio.run(
        policy.select("dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS)
    )

    assert selection is not None and selection.target == "compat@native"
    assert selection.cache_tag == build_runtime_identity(
        plan.family.id,
        plan.identity_components,
        diffusion_dtype=BFLOAT16,
        text_dtype=FLOAT32,
        vae_dtype=BFLOAT16,
        fp8_matmul=False,
    )


def test_official_lumina2_api_workflow_translates_and_admits_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Official exported API graph with synthetic header geometry, not model execution.

    Source: workflow_templates db9d5859d09c21a2d4101a1c18f64fc2f70e4fa4,
    templates/image_netayume_lumina_t2i.json, exported by frontend 1.51.9.
    """
    import hashlib

    from dinkster_assets import AssetRef
    from dinkster_compat_comfy import (
        COMFY_INPUT_ADAPTERS,
        make_load_checkpoint_adapter,
        translate_prompt,
    )
    from dinkster_graph import GraphNode, Link
    from dinkster_inference import plan_native
    from dinkster_nodes_foundation import FOUNDATION_NODES
    from dinkster_nodes_generation import GENERATION_NODES
    from dinkster_nodes_media_io import SaveImage
    from dinkster_schema import build_schemas, comfy_alias_registry_from_wire

    from tests.test_inference_lumina2 import _checkpoint_geometries
    from tests.test_inference_lumina2 import _Source as CheckpointSource

    root = Path(__file__).parents[1]
    fixture = (root / "tests/fixtures/lumina2-workflow-api.json").read_bytes()
    assert hashlib.sha256(fixture).hexdigest() == (
        "2bfa740df6385b351ccd552e0ef9871f190968be8605785bcc1d6ae6235b9d6e"
    )
    prompt = json.loads(fixture)
    name = prompt["48:34"]["inputs"]["ckpt_name"]
    path = _safetensors(tmp_path / name)
    digest = digest_file(path)
    asset = AssetRef(digest=digest, name=name, size=path.stat().st_size)
    source = CheckpointSource(path, _checkpoint_geometries(), digest, asset.size)
    schemas = build_schemas((*FOUNDATION_NODES, *GENERATION_NODES, SaveImage))
    aliases = comfy_alias_registry_from_wire(
        json.loads((root / "packages/dinkster-nodes-foundation/comfy-aliases.json").read_text())
    )
    concat_schema = next(
        snapshot.schema
        for snapshot in aliases.source_schemas
        if snapshot.schema.node_type == "comfy.StringConcatenate"
    )
    schemas[concat_schema.node_type] = concat_schema
    translated = translate_prompt(
        prompt,
        schemas,
        input_adapters={
            **COMFY_INPUT_ADAPTERS,
            "dinkster.load_checkpoint": make_load_checkpoint_adapter(
                lambda candidate: asset if candidate == name else None
            ),
        },
    )
    nodes = translated.graph.nodes
    checkpoint = cast(GraphNode, nodes["48:34"])
    model = cast(GraphNode, nodes["48:32"])
    sampler = cast(GraphNode, nodes["48:33"])
    assert translated.targets == ("9",)
    assert checkpoint.node_type == "dinkster.load_checkpoint"
    assert checkpoint.inputs == {"checkpoint": asset.to_wire()}
    assert model.node_type == "dinkster.model_sampling_aura_flow"
    assert model.inputs == {"model": Link("48:34", "model"), "shift": 4.0}
    assert sampler.inputs["model"] == Link("48:32", "model")
    assert sampler.inputs["sampler_name"] == "res_multistep"
    assert sampler.inputs["scheduler"] == "simple"
    for node_id in ("48:50", "48:35:43"):
        assert cast(GraphNode, nodes[node_id]).inputs["clip"] == Link("48:34", "clip")
    assert cast(GraphNode, nodes["48:36"]).inputs["vae"] == Link("48:34", "vae")

    monkeypatch.setattr(
        native_policy_module, "load_safetensors_header", lambda _path, **_kw: source
    )
    policy = NativeDispatchPolicy(lambda _digest: path, _ignore_diagnostic)
    selection = asyncio.run(
        policy.select(checkpoint.node_type, {"checkpoint": _asset(digest, name)}, ARMS)
    )
    plan = plan_native(source)
    assert selection is not None and selection.target == "compat@native"
    assert selection.cache_tag == build_runtime_identity(
        plan.family.id,
        plan.identity_components,
        diffusion_dtype=BFLOAT16,
        text_dtype=FLOAT32,
        vae_dtype=BFLOAT16,
        fp8_matmul=False,
    )
    for node_id, input_name in (
        ("48:32", "model"),
        ("48:33", "model"),
        ("48:50", "clip"),
        ("48:35:43", "clip"),
        ("48:36", "vae"),
    ):
        node = cast(GraphNode, nodes[node_id])
        downstream = asyncio.run(
            policy.select(node.node_type, {input_name: _resident(node_id, selection.target)}, ARMS)
        )
        assert downstream is not None and downstream.target == selection.target


def test_lumina2_all_in_one_selects_three_independent_component_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "blake3:" + "e" * 64
    plans = (
        _anima_planned("diffusion"),
        _anima_planned("gemma2_2b"),
        _anima_planned("vae"),
    )
    components = tuple(
        zip(cast("tuple[Any, ...]", ("diffusion", "gemma2_2b", "vae")), plans, strict=True)
    )
    policy = NativeDispatchPolicy(lambda _candidate: None, _ignore_diagnostic)
    from dinkster_inference.component_catalog import default_component_registry
    from dinkster_inference.component_registry import DetectedComponents

    descriptor = default_component_registry().get("dinkster.lumina2")
    assert descriptor is not None
    monkeypatch.setattr(
        policy,
        "_probe_components_transaction",
        lambda _digest: (DetectedComponents(descriptor, components),),
    )

    diffusion = asyncio.run(
        policy.select(
            "dinkster.load_diffusion_model",
            {"diffusion_model": _asset(digest), "weight_dtype": _value("default")},
            ARMS,
        )
    )
    text = asyncio.run(
        policy.select(
            "dinkster.load_clip",
            {"text_encoder": _asset(digest), "type": _value("lumina2")},
            ARMS,
        )
    )
    vae = asyncio.run(policy.select("dinkster.load_vae", {"vae": _asset(digest)}, ARMS))

    assert diffusion is not None and text is not None and vae is not None
    assert {diffusion.target, text.target, vae.target} == {"compat@native"}
    assert diffusion.cache_tag == inference.lumina2_component_runtime_identity(
        plans[0], "diffusion", BFLOAT16
    )
    assert text.cache_tag == inference.lumina2_component_runtime_identity(
        plans[1], "gemma2_2b", FLOAT32
    )
    assert vae.cache_tag == inference.lumina2_component_runtime_identity(plans[2], "vae", BFLOAT16)


@pytest.mark.parametrize("family_id", ("dinkster.anima", "dinkster.krea2", "dinkster.future"))
def test_successful_checkpoint_plan_selects_native_without_family_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family_id: str
) -> None:
    path = _safetensors(tmp_path / "model.safetensors")
    digest = digest_file(path)
    diagnostics: list[NativePolicyDiagnostic] = []
    monkeypatch.setattr(
        native_policy_module,
        "plan_native",
        lambda _source: _fake_plan(family_id),
    )
    policy = NativeDispatchPolicy(lambda _digest: path, diagnostics.append)

    selection = asyncio.run(
        policy.select("dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS)
    )

    assert selection is not None and selection.target == "compat@native"
    assert selection.cache_tag == _expected_tag(family_id)
    assert diagnostics == []


def test_fp8_setting_flip_rotates_memoized_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "model.safetensors")
    digest = digest_file(path)
    requested = False
    calls = 0

    def plan(_source: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return _fake_plan()

    monkeypatch.setattr(native_policy_module, "plan_native", plan)
    policy = NativeDispatchPolicy(
        lambda _digest: path,
        _ignore_diagnostic,
        fp8_matmul=lambda: requested,
    )

    async def scenario() -> tuple[ExecutionSelection, ExecutionSelection]:
        nonlocal requested
        first = await policy.select(
            "dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS
        )
        requested = True
        second = await policy.select(
            "dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS
        )
        assert first is not None and second is not None
        return first, second

    first, second = asyncio.run(scenario())
    assert first.cache_tag == _expected_tag()
    assert first.fp8_matmul is False
    assert second.cache_tag == _expected_tag(fp8_matmul=True)
    assert second.fp8_matmul is True
    assert second.cache_tag != first.cache_tag
    assert calls == 2


def test_fp8_e5m2_storage_is_refused_at_header_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "e5m2.safetensors")
    digest = digest_file(path)
    component = SimpleNamespace(dtypes={"weight": SimpleNamespace(name="float8_e5m2")})
    plan = SimpleNamespace(
        family=SimpleNamespace(id="dinkster.sd15"),
        identity_components=(component,),
    )
    diagnostics: list[NativePolicyDiagnostic] = []
    monkeypatch.setattr(native_policy_module, "plan_native", lambda _source: plan)
    policy = NativeDispatchPolicy(
        lambda _digest: path,
        diagnostics.append,
        fp8_matmul=lambda: True,
    )

    selection = asyncio.run(
        policy.select("dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS)
    )

    assert selection is not None and selection.target == "compat"
    assert diagnostics[0].kind == "refused"
    assert diagnostics[0].reasons == ("fp8 matmul does not support float8_e5m2 checkpoint storage",)


@pytest.mark.parametrize(
    "node_type", ("dinkster.load_checkpoint", "dinkster.load_checkpoint_stack")
)
@pytest.mark.parametrize("native_only", (False, True))
def test_refusal_reports_reasons_and_is_memoized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node_type: str, native_only: bool
) -> None:
    path = _safetensors(tmp_path / "refused.safetensors")
    digest = digest_file(path)
    diagnostics: list[NativePolicyDiagnostic] = []
    reasons = (
        "detected checkpoint labels: example.encoder",
        "example.assembly: missing text encoder component",
    )
    calls = 0

    def refuse(source: object) -> None:
        nonlocal calls
        calls += 1
        raise NativeRefusalError(reasons)

    monkeypatch.setattr(native_policy_module, "plan_native", refuse)
    policy = NativeDispatchPolicy(lambda _digest: path, diagnostics.append)
    arms = ("provider", {"provider": "unversioned"}) if native_only else ARMS

    async def scenario() -> None:
        for _ in range(2):
            pending = policy.select(node_type, {"checkpoint": _asset(digest)}, arms)
            if native_only:
                with pytest.raises(RuntimeError) as error:
                    await pending
                assert str(error.value) == f"{node_type} cannot load checkpoint: " + "; ".join(
                    reasons
                )
            else:
                selection = await pending
                assert selection is not None and selection.target == "compat"

    asyncio.run(scenario())
    assert calls == 1
    assert diagnostics == [NativePolicyDiagnostic("refused", digest, reasons)]


@pytest.mark.parametrize("family_id", ["dinkster.flux_dev", "dinkster.flux_schnell"])
def test_flux_families_select_native(
    family_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "flux.safetensors")
    digest = digest_file(path)
    diagnostics: list[NativePolicyDiagnostic] = []
    monkeypatch.setattr(
        native_policy_module,
        "plan_native",
        lambda _source: _fake_plan(family_id),
    )
    policy = NativeDispatchPolicy(lambda _digest: path, diagnostics.append)

    selection = asyncio.run(
        policy.select("dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS)
    )

    assert selection is not None and selection.target == "compat@native"
    assert selection.cache_tag == _expected_tag(family_id)
    assert diagnostics == []


def test_malformed_safetensors_is_memoized_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "legacy.ckpt"
    path.write_bytes(b"not safetensors")
    digest = digest_file(path)
    calls = 0
    diagnostics: list[NativePolicyDiagnostic] = []

    def locate(_digest: str) -> Path:
        nonlocal calls
        calls += 1
        return path

    policy = NativeDispatchPolicy(locate, diagnostics.append)

    async def scenario() -> None:
        for _ in range(2):
            selection = await policy.select(
                "dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS
            )
            assert selection is not None and selection.target == "compat"

    asyncio.run(scenario())
    assert calls == 1
    assert len(diagnostics) == 1 and diagnostics[0].kind == "unsupported"


def test_convertible_is_non_terminal_serves_owner_and_dedupes_diagnostics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "extensionless-digest"
    path.write_bytes(b"legacy checkpoint")
    digest = digest_file(path)
    diagnostics: list[NativePolicyDiagnostic] = []
    release = asyncio.Event()
    scheduled = 0

    async def schedule(_path: Path, _logical_name: str) -> tuple[str, str | None]:
        nonlocal scheduled
        scheduled += 1
        await release.wait()
        return "transport-failure", None

    policy = NativeDispatchPolicy(
        lambda _digest: path,
        diagnostics.append,
        schedule_conversion=schedule,
    )

    async def scenario() -> None:
        for _ in range(2):
            selection = await policy.select(
                "dinkster.load_checkpoint",
                {"checkpoint": _asset(digest, "logical.ckpt")},
                ARMS,
            )
            assert selection is not None and selection.target == "compat"
        await asyncio.sleep(0)
        assert scheduled == 1
        assert policy._memo == {}  # noqa: SLF001 - proves non-terminal storage
        release.set()
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert [diagnostic.kind for diagnostic in diagnostics] == ["convertible"]


def test_conversion_transport_failure_clears_inflight_and_reschedules(
    tmp_path: Path,
) -> None:
    path = tmp_path / "asset"
    path.write_bytes(b"legacy checkpoint")
    digest = digest_file(path)
    scheduled = 0

    async def schedule(_path: Path, _logical_name: str) -> tuple[str, str | None]:
        nonlocal scheduled
        scheduled += 1
        return "transport-failure", None

    policy = NativeDispatchPolicy(
        lambda _digest: path,
        _ignore_diagnostic,
        schedule_conversion=schedule,
    )

    async def scenario() -> None:
        for expected in (1, 2):
            selection = await policy.select(
                "dinkster.load_checkpoint",
                {"checkpoint": _asset(digest, "logical.pt")},
                ARMS,
            )
            assert selection is not None and selection.target == "compat"
            await asyncio.sleep(0)
            assert scheduled == expected

    asyncio.run(scenario())


def test_conversion_refusal_memoizes_terminal_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "asset"
    path.write_bytes(b"legacy checkpoint")
    digest = digest_file(path)
    diagnostics: list[NativePolicyDiagnostic] = []
    located = 0
    scheduled = 0

    def locate(_digest: str) -> Path:
        nonlocal located
        located += 1
        return path

    async def schedule(_path: Path, _logical_name: str) -> tuple[str, str | None]:
        nonlocal scheduled
        scheduled += 1
        return "refused", "converter refused unsafe pickle globals"

    policy = NativeDispatchPolicy(
        locate,
        diagnostics.append,
        schedule_conversion=schedule,
    )

    async def scenario() -> None:
        first = await policy.select(
            "dinkster.load_checkpoint",
            {"checkpoint": _asset(digest, "logical.pth")},
            ARMS,
        )
        assert first is not None and first.target == "compat"
        await asyncio.sleep(0)
        second = await policy.select(
            "dinkster.load_checkpoint",
            {"checkpoint": _asset(digest, "renamed.ckpt")},
            ARMS,
        )
        assert second is not None and second.target == "compat"

    asyncio.run(scenario())
    assert located == 1
    assert scheduled == 1
    assert [diagnostic.kind for diagnostic in diagnostics] == [
        "convertible",
        "unsupported",
    ]


def test_discovered_sidecar_upgrades_native_with_original_asset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "extensionless-digest"
    path.write_bytes(b"legacy checkpoint")
    digest = digest_file(path)
    sidecar = tmp_path / "extensionless-digest.0123456789abcdef.dinkster.safetensors"
    planned_paths: list[Path] = []
    located: list[str] = []

    def locate(candidate: str) -> Path:
        located.append(candidate)
        return path

    def plan(source: object) -> SimpleNamespace:
        planned_paths.append(source.path)  # type: ignore[attr-defined]
        return _fake_plan()

    async def schedule(_path: Path, _logical_name: str) -> tuple[str, str | None]:
        _safetensors(sidecar)
        return "success", None

    monkeypatch.setattr(native_policy_module, "plan_native", plan)
    policy = NativeDispatchPolicy(
        locate,
        _ignore_diagnostic,
        schedule_conversion=schedule,
    )

    async def scenario() -> ExecutionSelection:
        first = await policy.select(
            "dinkster.load_checkpoint",
            {"checkpoint": _asset(digest, "original.ckpt")},
            ARMS,
        )
        assert first is not None and first.target == "compat"
        await asyncio.sleep(0)
        second = await policy.select(
            "dinkster.load_checkpoint",
            {"checkpoint": _asset(digest, "original.ckpt")},
            ARMS,
        )
        assert second is not None
        return second

    selection = asyncio.run(scenario())
    assert selection.target == "compat@native"
    assert selection.cache_tag == _expected_tag()
    assert located == [digest, digest]
    assert planned_paths == [sidecar]


@pytest.fixture
def conversion_composer() -> Iterator[ServingComposer]:
    composer = ServingComposer()
    try:
        yield composer
    finally:
        composer.composition._isolated.clear()
        asyncio.run(composer.close())


def test_conversion_scheduler_sends_only_to_advertising_worker(
    tmp_path: Path, conversion_composer: ServingComposer
) -> None:
    path = tmp_path / "model.ckpt"
    path.write_bytes(b"legacy")

    class Peer:
        def __init__(self, capable: bool) -> None:
            self.can_convert_legacy_checkpoint = capable
            self.calls = 0

        async def convert_legacy_checkpoint(
            self, _path: Path, _logical_name: str
        ) -> tuple[str, str | None]:
            self.calls += 1
            return "success", None

    older_peer = Peer(False)
    capable_peer = Peer(True)
    conversion_composer.composition._isolated.extend(cast("Any", [older_peer, capable_peer]))

    outcome = asyncio.run(
        _schedule_legacy_checkpoint_conversion(conversion_composer, path, "logical.ckpt")
    )

    assert outcome == ("success", None)
    assert older_peer.calls == 0
    assert capable_peer.calls == 1


def test_conversion_scheduler_skips_broken_peer_for_later_capable_worker(
    tmp_path: Path, conversion_composer: ServingComposer
) -> None:
    path = tmp_path / "model.ckpt"
    path.write_bytes(b"legacy")

    class BrokenPeer:
        @property
        def can_convert_legacy_checkpoint(self) -> bool:
            raise RuntimeError("session did not start")

    class CapablePeer:
        can_convert_legacy_checkpoint = True

        async def convert_legacy_checkpoint(
            self, _path: Path, _logical_name: str
        ) -> tuple[str, str | None]:
            return "success", None

    conversion_composer.composition._isolated.extend(cast("Any", [BrokenPeer(), CapablePeer()]))

    assert asyncio.run(
        _schedule_legacy_checkpoint_conversion(conversion_composer, path, "logical.ckpt")
    ) == ("success", None)


def test_conversion_scheduler_no_capable_worker_is_quiet_noop(
    tmp_path: Path, conversion_composer: ServingComposer
) -> None:
    path = tmp_path / "model.ckpt"
    path.write_bytes(b"legacy")
    older_peer = SimpleNamespace(can_convert_legacy_checkpoint=False)
    conversion_composer.composition._isolated.append(cast("Any", older_peer))

    assert asyncio.run(
        _schedule_legacy_checkpoint_conversion(conversion_composer, path, "logical.ckpt")
    ) == ("transport-failure", None)


def test_conversion_scheduler_waits_for_composer_maintenance(
    tmp_path: Path, conversion_composer: ServingComposer
) -> None:
    async def scenario() -> None:
        called = asyncio.Event()

        class Peer:
            can_convert_legacy_checkpoint = True

            async def convert_legacy_checkpoint(
                self, _path: Path, _logical_name: str
            ) -> tuple[str, str | None]:
                called.set()
                return "success", None

        conversion_composer.composition._isolated.append(cast("Any", Peer()))
        async with conversion_composer.generation_transaction():
            conversion = asyncio.create_task(
                _schedule_legacy_checkpoint_conversion(
                    conversion_composer, tmp_path / "model.ckpt", "model.ckpt"
                )
            )
            done, _ = await asyncio.wait({conversion}, timeout=0.05)
            assert not done and not called.is_set()
        assert await conversion == ("success", None)
        assert called.is_set()

    asyncio.run(scenario())


def test_missing_is_not_memoized_and_later_selects_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "appears.safetensors")
    digest = digest_file(path)
    present = False
    calls = 0
    diagnostics: list[NativePolicyDiagnostic] = []

    def locate(_digest: str) -> Path | None:
        nonlocal calls
        calls += 1
        return path if present else None

    monkeypatch.setattr(native_policy_module, "plan_native", lambda source: _fake_plan())
    policy = NativeDispatchPolicy(locate, diagnostics.append)

    async def scenario() -> None:
        nonlocal present
        first = await policy.select(
            "dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS
        )
        second = await policy.select(
            "dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS
        )
        assert first is not None and first.target == "compat"
        assert second is not None and second.target == "compat"
        present = True
        third = await policy.select(
            "dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS
        )
        assert third is not None and third.target == "compat@native"

    asyncio.run(scenario())
    assert calls == 3
    assert [diagnostic.kind for diagnostic in diagnostics] == ["missing"]


def test_concurrent_checkpoint_plans_are_single_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "shared.safetensors")
    digest = digest_file(path)
    calls = 0

    def locate(_digest: str) -> Path:
        nonlocal calls
        calls += 1
        time.sleep(0.03)
        return path

    monkeypatch.setattr(native_policy_module, "plan_native", lambda source: _fake_plan())
    policy = NativeDispatchPolicy(locate, _ignore_diagnostic)

    async def scenario() -> None:
        selections = await asyncio.gather(
            *(
                policy.select("dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS)
                for _ in range(8)
            )
        )
        assert all(
            selection is not None and selection.target == "compat@native"
            for selection in selections
        )

    asyncio.run(scenario())
    assert calls == 1


def test_cancelling_one_waiter_does_not_cancel_shared_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "shielded.safetensors")
    digest = digest_file(path)
    calls = 0

    def locate(_digest: str) -> Path:
        nonlocal calls
        calls += 1
        time.sleep(0.03)
        return path

    monkeypatch.setattr(native_policy_module, "plan_native", lambda source: _fake_plan())
    policy = NativeDispatchPolicy(locate, _ignore_diagnostic)

    async def scenario() -> None:
        cancelled = asyncio.create_task(
            policy.select("dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS)
        )
        survivor = asyncio.create_task(
            policy.select("dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS)
        )
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        selection = await survivor
        assert selection is not None and selection.target == "compat@native"

    asyncio.run(scenario())
    assert calls == 1


def test_probe_exception_does_not_poison_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "retry.safetensors")
    digest = digest_file(path)
    calls = 0

    def locate(_digest: str) -> Path:
        nonlocal calls
        calls += 1
        if calls == 1:
            return tmp_path / "vanished.safetensors"
        return path

    monkeypatch.setattr(native_policy_module, "plan_native", lambda source: _fake_plan())
    policy = NativeDispatchPolicy(locate, _ignore_diagnostic)

    async def scenario() -> None:
        with pytest.raises(OSError):
            await policy.select("dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS)
        selection = await policy.select(
            "dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS
        )
        assert selection is not None and selection.target == "compat@native"

    asyncio.run(scenario())
    assert calls == 2


def test_stale_path_fails_loudly_without_memoization(tmp_path: Path) -> None:
    expected = _safetensors(tmp_path / "expected.safetensors")
    stale = _safetensors(tmp_path / "stale.safetensors")
    stale.write_bytes(stale.read_bytes() + b"different")
    digest = digest_file(expected)
    policy = NativeDispatchPolicy(lambda _digest: stale, _ignore_diagnostic)

    with pytest.raises(AssetIntegrityError):
        asyncio.run(policy.select("dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS))


@pytest.mark.parametrize("digest", [None, 42, "not-a-digest"])
def test_missing_or_malformed_digest_is_loud(digest: object) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    with pytest.raises(RuntimeError, match="digest"):
        asyncio.run(policy.select("dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS))


_REGISTRY = TypeRegistry()
register_core_types(_REGISTRY)
register_resource_handle_type(_REGISTRY)


def _resident(resource_id: str, stamp: object = ..., *, nested: bool = False) -> Value:
    value = _REGISTRY.wrap(
        RESOURCE_HANDLE_TYPE,
        ResourceHandle(resource_id=resource_id, kind="model", owner="session"),
    )
    if stamp is not ...:
        entries = dict(value.meta.entries)
        entries[RESOURCE_PRODUCER_ARM_META_KEY] = stamp
        value = replace(value, meta=ValueMeta(entries))
    if nested:
        return Value(
            type_id=f"list<{RESOURCE_HANDLE_TYPE}>",
            fingerprint=list_fingerprint((value,)),
            meta=ValueMeta({"length": 1}),
            payload=ListPayload((value,)),
        )
    return value


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize(
    "node_type",
    (
        "dinkster.cfg_override",
        "dinkster.rescale_cfg",
        "dinkster.model_sampling_sd3",
        "dinkster.model_sampling_aura_flow",
        "dinkster.model_sampling_flux",
        "extension.adjust_model",
        "dinkster.ksampler",
        "dinkster.ksampler_advanced",
        "dinkster.basic_scheduler",
        "dinkster.beta_sampling_scheduler",
        "dinkster.sd_turbo_scheduler",
        "dinkster.sampling_percent_to_sigma",
        "dinkster.basic_guider",
        "dinkster.cfg_guider",
        "dinkster.scheduled_cfg_guider",
        "dinkster.perp_neg_guider",
        "dinkster.sampler_custom",
        "dinkster.sampler_custom_advanced",
        "dinkster.minimax_music3_text_encode",
        "dinkster.text_generate",
        "dinkster.prompt_enhance",
        "dinkster.load_lora",
        "dinkster.load_lora_model_only",
        "dinkster.vae_decode_tiled",
        "dinkster.vae_decode_audio",
        "dinkster.vae_decode_audio_tiled",
        "dinkster.vae_encode_tiled",
        "dinkster.ltxav_id_lora_reference_audio",
        "dinkster.ltxv_spatiotemporal_guidance",
        "dinkster.ltxv_modality_guidance",
        "dinkster.ltxv_duration_predictor",
        "dinkster.ltxv_dual_cfg_guider",
        "dinkster.trellis2_conditioning",
        "dinkster.pixal3d_conditioning",
        "dinkster.vae_decode_structure_trellis2",
        "dinkster.trellis2_shape_stage",
        "dinkster.trellis2_upsample_stage",
        "dinkster.vae_decode_shape_trellis",
        "dinkster.trellis2_texture_stage",
        "dinkster.vae_decode_texture_trellis",
    ),
)
def test_downstream_follows_producer_arm_default_tag(node_type: str, nested: bool) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    selection = asyncio.run(
        policy.select(
            node_type,
            {"model": _resident("m", "compat@native", nested=nested)},
            ARMS,
        )
    )
    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == "native-default"


@pytest.mark.parametrize("producer", ("provider@cpu", "provider@worker:replica-2"))
def test_downstream_producer_following_does_not_require_native_arm_name(producer: str) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    arms = ("provider", {"provider": "owner-tag", producer: "producer-tag"})
    value = _resident("model", producer)
    selection = asyncio.run(policy.select("extension.adjust_model", {"model": value}, arms))
    assert selection is not None
    assert selection.target == producer
    assert selection.cache_tag == "producer-tag"
    assert value.meta.get(RESOURCE_PRODUCER_ARM_META_KEY) == producer
    assert asyncio.run(policy.select("extension.adjust_model", {}, arms)) is None


@pytest.mark.parametrize("producer", ["compat", "compat@native"])
def test_custom_sampling_settings_do_not_override_the_guider_arm(producer: str) -> None:
    from dinkster_inference import register_inference_types
    from dinkster_inference.sampling_wire import NoiseSelection, SamplerSelection, SigmaSchedule

    registry = TypeRegistry()
    register_inference_types(registry)
    inputs = {
        "guider": _resident("guider", producer),
        "noise": registry.wrap("dinkster.noise", NoiseSelection(7)),
        "sampler": registry.wrap("dinkster.sampler", SamplerSelection("dinkster.euler", ())),
        "sigmas": registry.wrap("dinkster.sigmas", SigmaSchedule((1.0, 0.0))),
    }
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    selection = asyncio.run(policy.select("dinkster.sampler_custom_advanced", inputs, ARMS))
    assert selection is not None
    assert selection.target == producer


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize(
    "node_type",
    (
        "dinkster.ltxv_image_to_video",
        "dinkster.ltxv_image_to_video_inplace",
        "dinkster.ltxv_add_guide",
        "dinkster.ltxv_latent_upsampler",
    ),
)
def test_ltxv_media_follows_vae_producer_arm(node_type: str, nested: bool) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    selection = asyncio.run(
        policy.select(
            node_type,
            {"vae": _resident("vae", "compat@native", nested=nested)},
            ARMS,
        )
    )
    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == "native-default"


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize(
    "node_type",
    (
        "dinkster.ltxav_audio_vae_decode",
        "dinkster.ltxav_reference_audio",
        "dinkster.ltxav_id_lora_reference_audio",
    ),
)
def test_ltxav_reference_audio_follows_codec_producer_arm(node_type: str, nested: bool) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    selection = asyncio.run(
        policy.select(
            node_type,
            {"audio_vae": _resident("audio-codec", "compat@native", nested=nested)},
            ARMS,
        )
    )
    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == "native-default"


@pytest.mark.parametrize(
    ("node_type", "inputs"),
    (
        (
            "dinkster.minimax_h3_t2va_conditioning",
            {"clip": _resident("conditioner", "compat@native")},
        ),
        (
            "dinkster.minimax_h3_fl2va_conditioning",
            {
                "clip": _resident("conditioner", "compat@native"),
                "video_vae": _resident("video-vae", "compat@native"),
            },
        ),
        (
            "dinkster.minimax_h3_ref2va_conditioning",
            {
                "clip": _resident("conditioner", "compat@native"),
                "video_vae": _resident("video-vae", "compat@native"),
                "audio_vae": _resident("audio-vae", "compat@native"),
            },
        ),
        (
            "dinkster.minimax_h3_add_guide",
            {
                "vae": _resident("video-vae", "compat@native"),
                "audio_vae": _resident("audio-vae", "compat@native"),
            },
        ),
        (
            "dinkster.minimax_h3_av_encode",
            {
                "video_vae": _resident("video-vae", "compat@native"),
                "audio_vae": _resident("audio-vae", "compat@native"),
            },
        ),
        (
            "dinkster.minimax_h3_av_decode",
            {
                "video_vae": _resident("video-vae", "compat@native"),
                "audio_vae": _resident("audio-vae", "compat@native"),
            },
        ),
        (
            "dinkster.seedvr2_conditioning",
            {"model": _resident("seedvr2", "compat@native")},
        ),
    ),
)
def test_native_modality_nodes_follow_component_producer_arm(
    node_type: str,
    inputs: dict[str, Value],
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    selection = asyncio.run(policy.select(node_type, inputs, ARMS))
    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == "native-default"


def test_minimax_h3_modality_node_refuses_conflicting_component_producers() -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    with pytest.raises(RuntimeError, match="conflicting producer arms"):
        asyncio.run(
            policy.select(
                "dinkster.minimax_h3_fl2va_conditioning",
                {
                    "clip": _resident("conditioner", "compat@native"),
                    "video_vae": _resident("video-vae", "compat"),
                },
                ARMS,
            )
        )


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        ({"model": _resident("m", 12)}, "malformed"),
        ({"model": _resident("m", "other")}, "unknown"),
        (
            {
                "model": _resident("m", "compat"),
                "clip": _resident("c", "compat@native"),
            },
            "conflicting",
        ),
    ],
)
@pytest.mark.parametrize(
    "node_type",
    ("dinkster.ksampler", "dinkster.model_sampling_aura_flow", "extension.adjust_model"),
)
def test_downstream_bad_producer_stamps_are_loud(
    inputs: dict[str, Value], match: str, node_type: str
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    with pytest.raises(RuntimeError, match=match):
        asyncio.run(policy.select(node_type, inputs, ARMS))


@pytest.mark.parametrize(
    "node_type",
    tuple(NATIVE_DISPATCH_SCHEMAS),
)
def test_native_dispatch_affinity_preserves_native_body_selection(node_type: str) -> None:
    policy = NativeDispatchPolicy(
        lambda _digest: None,
        _ignore_diagnostic,
        schemas=lambda: NATIVE_DISPATCH_SCHEMAS,
    )
    selection = asyncio.run(policy.select(node_type, {}, ARMS))
    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == "native-default"

    assert asyncio.run(policy.select(node_type, {}, ("compat", {"compat": "compat-tag"}))) is None


def test_schema_dispatch_affinity_routes_without_a_policy_id_list() -> None:
    marked = NodeSchema("extension.marked", dispatch_affinity="native")
    ordinary = NodeSchema("extension.ordinary")
    former_literal = replace(
        NATIVE_DISPATCH_SCHEMAS["dinkster.create_hook_lora"], dispatch_affinity=None
    )
    schemas = {schema.node_type: schema for schema in (marked, ordinary, former_literal)}
    policy = NativeDispatchPolicy(
        lambda _digest: None,
        _ignore_diagnostic,
        schemas=lambda: schemas,
    )

    selection = asyncio.run(policy.select(marked.node_type, {}, ARMS))
    assert selection is not None
    assert selection.target == "compat@native"
    assert asyncio.run(policy.select(ordinary.node_type, {}, ARMS)) is None
    assert asyncio.run(policy.select(former_literal.node_type, {}, ARMS)) is None
    assert asyncio.run(policy.select("extension.unknown", {}, ARMS)) is None


def test_native_dispatch_schema_inventory_and_current_wire_compatibility() -> None:
    assert len(NATIVE_DISPATCH_SCHEMAS) == 11
    assert all(schema.dispatch_affinity == "native" for schema in NATIVE_DISPATCH_SCHEMAS.values())
    catalog_schemas = {
        schema.node_type: schema
        for schema in (node.schema() for node in (*NATIVE_NODES, *NATIVE_ARM_NODES))
    }
    assert {
        node_type
        for node_type, schema in catalog_schemas.items()
        if schema.dispatch_affinity == "native"
    } == set(NATIVE_DISPATCH_SCHEMAS)
    synthetic = NodeSchema("extension.synthetic", dispatch_affinity="native")

    for schema in (*NATIVE_DISPATCH_SCHEMAS.values(), synthetic):
        wire = schema_to_wire(schema)
        assert wire["dispatchAffinity"] == "native"
        assert schema_from_wire(wire) == schema
        assert schema_signature(schema) == schema_signature(replace(schema, dispatch_affinity=None))


def test_native_dispatch_affinity_is_stable_per_run_for_all_declared_schemas() -> None:
    policy = NativeDispatchPolicy(
        lambda _digest: None,
        _ignore_diagnostic,
        schemas=lambda: NATIVE_DISPATCH_SCHEMAS,
    )
    arms = (
        "compat",
        {
            "compat": "compat-tag",
            "compat@native:cuda:2": "native-tag",
            "compat@native:cuda:4": "native-tag",
        },
    )

    async def select(node_type: str, run_id: str) -> str:
        selection = await policy.select(node_type, {}, arms, run_id=run_id)
        assert selection is not None
        return selection.target

    assert {asyncio.run(select(node_type, "run-a")) for node_type in NATIVE_DISPATCH_SCHEMAS} == {
        "compat@native:cuda:2"
    }
    assert asyncio.run(select(next(iter(NATIVE_DISPATCH_SCHEMAS)), "run-b")) == (
        "compat@native:cuda:4"
    )


def test_classic_controlnet_source_ids_do_not_bypass_canonical_provider_selection() -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)

    for node_type in (
        "comfy.ControlNetLoader",
        "comfy.ControlNetApply",
        "comfy.ControlNetApplyAdvanced",
    ):
        assert asyncio.run(policy.select(node_type, {}, ARMS, run_id="control")) is None


def test_native_only_generation_provider_still_runs_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "model.safetensors")
    digest = digest_file(path)
    monkeypatch.setattr(native_policy_module, "plan_native", lambda _source: _fake_plan())
    policy = NativeDispatchPolicy(
        lambda candidate: path if candidate == digest else None, _ignore_diagnostic
    )

    async def scenario() -> None:
        assert await policy.select("dinkster.vae_decode", {"vae": _resident("v")}, ARMS) is None
        for node_type in ("dinkster.load_checkpoint", "dinkster.load_checkpoint_stack"):
            selection = await policy.select(
                node_type,
                {"checkpoint": _asset(digest)},
                ("provider", {"provider": "unversioned"}),
            )
            assert selection is not None
            assert selection.target == "provider"
            assert selection.cache_tag == _expected_tag()
        assert await policy.select("other.node", {}, ARMS) is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "node_type", ("dinkster.load_checkpoint", "dinkster.load_checkpoint_stack")
)
def test_native_only_generation_provider_refuses_unrecognized_checkpoint(
    node_type: str, tmp_path: Path
) -> None:
    path = tmp_path / "legacy.ckpt"
    path.write_bytes(b"not safetensors")
    digest = digest_file(path)
    diagnostics: list[NativePolicyDiagnostic] = []
    policy = NativeDispatchPolicy(
        lambda candidate: path if candidate == digest else None, diagnostics.append
    )

    with pytest.raises(RuntimeError) as error:
        asyncio.run(
            policy.select(
                node_type,
                {"checkpoint": _asset(digest)},
                ("provider", {"provider": "unversioned"}),
            )
        )
    assert len(diagnostics) == 1 and diagnostics[0].reasons
    assert str(error.value) == f"{node_type} cannot load checkpoint: " + "; ".join(
        diagnostics[0].reasons
    )


def test_serve_locator_uses_shared_vault_first_then_mounts(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    vault_bytes = b"vault checkpoint"
    vault_digest = digest_bytes(vault_bytes)
    with vault.writer(vault_digest) as writer:
        writer.write(vault_bytes)
        vault_path = writer.commit()
    mount_path = tmp_path / "mounted.safetensors"
    mount_path.write_bytes(b"mounted checkpoint")
    mount_digest = digest_file(mount_path)

    class Mounts:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def resolve(self, digest: str) -> Path | None:
            self.calls.append(digest)
            return mount_path if digest == mount_digest else None

    mounts = Mounts()
    locator = _native_asset_locator(vault, mounts)  # type: ignore[arg-type]
    library = ServerLibrary(vault=vault, store=LibraryStore(tmp_path / "library.sqlite"))

    assert library.vault is vault
    assert locator(vault_digest) == vault_path
    assert mounts.calls == []
    assert locator(mount_digest) == mount_path
    assert mounts.calls == [mount_digest]


def test_bare_non_ltx_refusal_keeps_owner_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _safetensors(tmp_path / "refused.safetensors")
    digest = digest_file(path)

    def refuse(_source: object) -> None:
        raise NativeRefusalError(("unsupported family",))

    monkeypatch.setattr(
        native_policy_module, "load_safetensors_header", lambda _path, **_kwargs: object()
    )
    monkeypatch.setattr(native_policy_module, "plan_native", refuse)
    policy = NativeDispatchPolicy(lambda _digest: path, _ignore_diagnostic)

    selection = asyncio.run(
        policy.select("dinkster.load_checkpoint", {"checkpoint": _asset(digest)}, ARMS)
    )

    assert selection is not None
    assert selection.target == "compat"
    assert selection.cache_tag == "compat-tag"


def test_load_clip_detects_gemma_component_despite_ltxv_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = SimpleNamespace(role="gemma3_12b", component=_anima_planned("gemma3_12b"))
    _stub_registry_probe(monkeypatch, policy, "dinkster.ltxav", "gemma3_12b", planned)
    selection = asyncio.run(
        policy.select(
            "dinkster.load_clip",
            {"text_encoder": _asset("blake3:" + "a" * 64), "type": _value("ltxv")},
            ARMS,
        )
    )
    assert selection is not None and selection.target == "compat@native"
    assert selection.cache_tag == inference.ltxav_component_runtime_identity(
        cast("Any", planned), BFLOAT16
    )


@pytest.mark.parametrize(
    "clip_type,expected_dtype",
    [
        ("ltxv", "bfloat16"),
        ("chroma", "float32"),
        ("sd3", None),
        ("mochi", None),
        ("cogvideox", None),
        ("pixart", None),
        ("stable_diffusion", None),
        ("unrecognized-profile", None),
        (None, None),
    ],
)
def test_load_clip_shared_t5_preserves_selected_text_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clip_type: str | None,
    expected_dtype: str | None,
) -> None:
    from test_inference_chroma import Source, geometrize

    path = _safetensors(tmp_path / "t5.safetensors")
    digest = digest_file(path)
    source = Source(
        geometrize(inference.t5_layout(inference.T5_XXL_CONFIG)),
        path=path,
        asset_digest=digest,
        asset_size=path.stat().st_size,
    )
    monkeypatch.setattr(
        native_policy_module, "load_safetensors_header", lambda _path, **_kwargs: source
    )
    diagnostics: list[NativePolicyDiagnostic] = []
    policy = NativeDispatchPolicy(lambda _digest: path, diagnostics.append)
    inputs = {"text_encoder": _asset(digest)}
    if clip_type is not None:
        inputs["type"] = _value(clip_type)
    selection = asyncio.run(policy.select("dinkster.load_clip", inputs, ARMS))
    assert selection is not None
    if expected_dtype is None:
        assert selection.target == "compat"
        assert selection.cache_tag == "compat-tag"
        assert selection.text_dtype is None
        assert len(diagnostics) == 1 and diagnostics[0].kind == "unsupported"
        assert "ambiguous components" in diagnostics[0].reasons[0]
        assert "Chroma t5xxl" in diagnostics[0].reasons[0]
        assert "LTX-Video t5xxl" in diagnostics[0].reasons[0]
        return
    assert diagnostics == []
    assert selection.target == "compat@native"
    if clip_type == "ltxv":
        planned = inference.plan_ltxv_split_component(source, role="t5xxl", path=path)
        expected_identity = inference.ltxv_component_runtime_identity(planned, BFLOAT16)
    else:
        planned_chroma = inference.plan_chroma_split_component(source, role="t5xxl", path=path)
        expected_identity = inference.chroma_component_runtime_identity(
            planned_chroma, "t5xxl", FLOAT32
        )
    assert selection.cache_tag == expected_identity
    assert selection.diffusion_dtype == "unloaded"
    assert selection.text_dtype == expected_dtype
    assert selection.vae_dtype == "unloaded"


@pytest.mark.parametrize(
    ("node_type", "role", "input_name", "dtype_field"),
    (
        ("dinkster.load_vae", "vae", "vae", "vae_dtype"),
        (
            "dinkster.load_diffusion_model",
            "diffusion",
            "diffusion_model",
            "diffusion_dtype",
        ),
    ),
)
def test_generic_loader_selects_native_ltxv_component(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
    role: str,
    input_name: str,
    dtype_field: str,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    config = LTXV_2B_V09_CONFIG if role == "diffusion" else LTXV_2B_V09_VAE_CONFIG
    planned = inference.LTXVStandaloneComponentPlan(
        cast("Any", role), cast("Any", replace(_anima_planned(role), config=config))
    )
    _stub_registry_probe(monkeypatch, policy, "dinkster.ltxv", role, planned)

    selection = asyncio.run(
        policy.select(node_type, {input_name: _asset("blake3:" + "d" * 64)}, ARMS)
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == inference.ltxv_component_runtime_identity(planned, BFLOAT16)
    assert getattr(selection, dtype_field) == "bfloat16"


@pytest.mark.parametrize(
    ("node_type", "input_name", "role", "dtype_field"),
    (
        ("dinkster.load_diffusion_model", "diffusion_model", "diffusion", "diffusion_dtype"),
        ("dinkster.load_vae", "vae", "vae", "vae_dtype"),
    ),
)
def test_generic_loader_selects_native_ltxav_component(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
    input_name: str,
    role: str,
    dtype_field: str,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    config = LTXAV_19B_CONFIG if role == "diffusion" else LTXAV_19B_VAE_CONFIG
    planned = native_policy_module.LTXAVStandaloneComponentPlan(
        cast("Any", role), cast("Any", replace(_anima_planned(role), config=config))
    )
    _stub_registry_probe(monkeypatch, policy, "dinkster.ltxav", role, planned)
    token = _sdpa_route_token(89)
    monkeypatch.setattr(
        policy,
        "_probe_ltxav_component_transaction",
        lambda _digest, selected_role: planned if selected_role == role else None,
    )

    selection = asyncio.run(
        policy.select(
            node_type,
            {input_name: _asset("blake3:" + "e" * 64)},
            ARMS,
            attention_routes={"compat@native": token},
        )
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == native_policy_module.ltxav_component_runtime_identity(
        planned, BFLOAT16, attention_policy="auto", attention_route_token=token
    )
    assert getattr(selection, dtype_field) == "bfloat16"


@pytest.mark.parametrize(
    ("text_role", "projection_config", "projection_from_text", "expected_roles"),
    (
        ("gemma3_12b", "single_linear", False, ("gemma3_12b", "text_projection", "connectors")),
        ("gemma3_12b", "dual_linear", False, ("gemma3_12b", "text_projection")),
        ("gemma4_12b", "dual_linear_gemma4", True, ("gemma4_12b", "text_projection")),
    ),
)
def test_dedicated_ltxav_text_loader_composes_selected_components(
    monkeypatch: pytest.MonkeyPatch,
    text_role: str,
    projection_config: str,
    projection_from_text: bool,
    expected_roles: tuple[str, ...],
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    calls: list[tuple[str, str]] = []
    identity_calls: list[tuple[str, dict[str, object]]] = []
    token = _sdpa_route_token(89)
    plans = {
        text_role: SimpleNamespace(name="gemma"),
        "text_projection": SimpleNamespace(
            name="projection",
            component=SimpleNamespace(config=projection_config),
        ),
        "connectors": SimpleNamespace(name="connectors"),
    }

    def probe(digest: str, role: str) -> object:
        calls.append((digest, role))
        if role == "text_projection" and projection_from_text and digest == checkpoint_digest:
            return None
        return plans[role]

    monkeypatch.setattr(policy, "_probe_ltxav_text_encoder_transaction", lambda _digest: text_role)
    monkeypatch.setattr(policy, "_probe_ltxav_component_transaction", probe)

    def identity(plan: object, dtype: object, **kwargs: object) -> str:
        name = cast("Any", plan).name
        dtype_name = cast("Any", dtype).name
        identity_calls.append((name, kwargs))
        return f"{name}:{dtype_name}"

    monkeypatch.setattr(native_policy_module, "ltxav_component_runtime_identity", identity)
    monkeypatch.setattr(
        native_policy_module,
        "compose_execution",
        lambda family_id, components: SimpleNamespace(
            family_id=family_id,
            components=components,
            execution_identity="ltxav-text-composition",
        ),
    )
    text_digest = "blake3:" + "f" * 64
    checkpoint_digest = "blake3:" + "1" * 64

    selection = asyncio.run(
        policy.select(
            "dinkster.load_ltxav_text_encoder",
            {
                "text_encoder": _asset(text_digest),
                "ckpt_name": _asset(checkpoint_digest),
            },
            ARMS,
            attention_routes={"compat@native": token},
        )
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == "ltxav-text-composition"
    assert selection.diffusion_dtype == "unloaded"
    assert selection.text_dtype == "bfloat16"
    assert selection.vae_dtype == "unloaded"
    expected_component_calls = len(expected_roles) + int(projection_from_text)
    assert len(calls) == expected_component_calls
    assert set(calls[:2]) == {
        (text_digest, text_role),
        (checkpoint_digest, "text_projection"),
    }
    if projection_from_text:
        assert calls[2] == (text_digest, "text_projection")
    elif "connectors" in expected_roles:
        assert calls[2] == (checkpoint_digest, "connectors")
    assert identity_calls == [
        (
            plans[role].name,
            {"attention_policy": "auto", "attention_route_token": token},
        )
        for role in expected_roles
    ]


def test_dedicated_ltxav_audio_vae_selects_native_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = cast("Any", object())
    monkeypatch.setattr(
        policy,
        "_probe_ltxav_audio_codec_transaction",
        lambda _digest: native_policy_module._LTXAVAudioCodecProbe(planned),
    )
    monkeypatch.setattr(
        native_policy_module,
        "ltxav_audio_codec_runtime_identity",
        lambda candidate: f"ltxav-audio:{candidate is planned}",
    )

    selection = asyncio.run(
        policy.select(
            "dinkster.load_ltxav_audio_vae",
            {"ckpt_name": _asset("blake3:" + "d" * 64)},
            ARMS,
        )
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == "ltxav-audio:True"
    assert selection.diffusion_dtype == "unloaded"
    assert selection.text_dtype == "unloaded"
    assert selection.vae_dtype == "float32"


def test_ltxav_latent_upscaler_loader_selects_exact_native_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    planned = cast("Any", object())
    token = _sdpa_route_token(89)
    monkeypatch.setattr(
        policy,
        "_probe_ltxav_component_transaction",
        lambda _digest, role: planned if role == "latent_upscaler" else None,
    )
    monkeypatch.setattr(
        native_policy_module,
        "ltxav_component_runtime_identity",
        lambda candidate, dtype, **_kwargs: f"ltxav-upscaler:{candidate is planned}:{dtype.name}",
    )

    selection = asyncio.run(
        policy.select(
            "dinkster.load_latent_upscale_model",
            {"model_name": _asset("blake3:" + "a" * 64)},
            ARMS,
            attention_routes={"compat@native": token},
        )
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == "ltxav-upscaler:True:bfloat16"
    assert selection.diffusion_dtype == "unloaded"
    assert selection.text_dtype == "unloaded"
    assert selection.vae_dtype == "bfloat16"


def test_non_ltx_latent_upscaler_preserves_owner_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    monkeypatch.setattr(policy, "_probe_ltxav_component_transaction", lambda *_args: None)

    selection = asyncio.run(
        policy.select(
            "dinkster.load_latent_upscale_model",
            {"model_name": _asset("blake3:" + "b" * 64)},
            ARMS,
        )
    )

    assert selection is not None
    assert selection.target == "compat"
    assert selection.cache_tag == "compat-tag"


@pytest.mark.parametrize(
    ("node_type", "role", "input_name", "match"),
    (
        (
            "dinkster.load_clip",
            "vae",
            "text_encoder",
            "no matching component architecture; detected LTX-Video vae",
        ),
        (
            "dinkster.load_vae",
            "t5xxl",
            "vae",
            "no matching component architecture; detected LTX-Video t5xxl",
        ),
        (
            "dinkster.load_diffusion_model",
            "t5xxl",
            "diffusion_model",
            "role mismatch: detected LTX-Video t5xxl",
        ),
    ),
)
def test_generic_ltxv_loaders_refuse_wrong_component_roles(
    monkeypatch: pytest.MonkeyPatch,
    node_type: str,
    role: str,
    input_name: str,
    match: str,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    _stub_registry_probe(monkeypatch, policy, "dinkster.ltxv", role, _anima_planned(role))
    inputs = {input_name: _asset("blake3:" + "e" * 64)}
    if node_type == "dinkster.load_clip":
        inputs["type"] = _value("ltxv")

    with pytest.raises(RuntimeError, match=match):
        asyncio.run(policy.select(node_type, inputs, ARMS))


@pytest.mark.parametrize("clip_type", ("ltxv", "flux2", "minimax", "unfamiliar"))
def test_load_clip_hint_does_not_block_unknown_component_fallback(clip_type: str) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)

    selection = asyncio.run(
        policy.select(
            "dinkster.load_clip",
            {"text_encoder": _asset("blake3:" + "b" * 64), "type": _value(clip_type)},
            ARMS,
        )
    )
    assert selection is not None
    assert selection.target == "compat"
    assert selection.cache_tag == "compat-tag"


def test_ltxav_text_probe_memoizes_classification_but_never_a_locate_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "encoder.safetensors"
    path.write_bytes(b"gemma3-12b-split-encoder")
    digest = digest_file(path)
    header_calls: list[str] = []
    source = object()

    def load_header(_path: Path, *, asset_digest: str, asset_size: int) -> object:
        header_calls.append(asset_digest)
        assert asset_size == path.stat().st_size
        return source

    monkeypatch.setattr(native_policy_module, "load_safetensors_header", load_header)
    monkeypatch.setattr(
        native_policy_module, "identify_ltxav_text_source", lambda _source: "gemma3_12b"
    )
    located: list[Path | None] = [None]
    policy = NativeDispatchPolicy(lambda _digest: located[0], _ignore_diagnostic)

    assert policy._probe_ltxav_text_encoder_transaction(digest) is None
    assert header_calls == []

    located[0] = path
    assert policy._probe_ltxav_text_encoder_transaction(digest) == "gemma3_12b"
    assert policy._probe_ltxav_text_encoder_transaction(digest) == "gemma3_12b"
    assert header_calls == [digest]


_TRELLIS2_SPLIT_ROLES = (
    "structure",
    "shape",
    "shape-512",
    "texture",
    "texture-512",
)


def _trellis2_split_inputs(roles: tuple[object, ...] = _TRELLIS2_SPLIT_ROLES) -> dict[str, Value]:
    inputs: dict[str, Value] = {}
    for index, role in enumerate(roles):
        member = f"components.component_{index}"
        inputs[f"{member}.component"] = _asset("blake3:" + f"{index + 1:x}" * 64)
        inputs[f"{member}.role"] = _value(role)
    return inputs


def _h3_split_inputs(digest: str, role: MiniMaxH3DiTRole) -> dict[str, Value]:
    return {
        "components.component_0.component": _asset(digest),
        "components.component_0.role": _value(role),
    }


@pytest.mark.parametrize("role", ["fl2va-dit", "ref2va-dit"])
def test_load_diffusion_components_selects_structurally_valid_h3_with_arbitrary_digest(
    role: MiniMaxH3DiTRole,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "finetune.safetensors"
    path.write_bytes(role.encode())
    digest = digest_file(path)
    layout = minimax_h3_dit_layout()
    geometries = {key: TensorGeometry(shape, FLOAT32) for key, shape in layout.keys.items()}
    header = SimpleNamespace(
        path=path,
        keys=lambda: tuple(geometries),
        entry=lambda key: SimpleNamespace(geometry=geometries[key]),
        metadata=lambda: {},
        asset_digest=digest,
        asset_size=path.stat().st_size,
    )
    monkeypatch.setattr(
        native_policy_module,
        "load_safetensors_header",
        lambda _path, **_identity: header,
    )
    runtime_versions = {"torch": "2.13.0+cu130", "dinkster-kitchen": "0.2.31"}
    policy = NativeDispatchPolicy(
        lambda found: path if found == digest else None,
        _ignore_diagnostic,
        minimax_h3_runtime_versions=lambda: runtime_versions,
    )

    selection = asyncio.run(
        policy.select("dinkster.load_diffusion_components", _h3_split_inputs(digest, role), ARMS)
    )

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == minimax_h3_dit_runtime_identity(
        asset_digest=digest,
        asset_size=path.stat().st_size,
        role=role,
        diffusion_dtype="bfloat16",
        runtime_facts=minimax_h3_dit_provider_facts(
            role,
            quantized=False,
            torch_version=runtime_versions["torch"],
            dinkster_kitchen_version=None,
        ),
    )


def test_h3_component_dispatch_binds_cache_identity_to_the_selected_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    role: MiniMaxH3DiTRole = "fl2va-dit"
    path = tmp_path / "finetune.safetensors"
    path.write_bytes(role.encode())
    digest = digest_file(path)
    layout = minimax_h3_dit_layout()
    geometries = {key: TensorGeometry(shape, FLOAT32) for key, shape in layout.keys.items()}
    header = SimpleNamespace(
        path=path,
        keys=lambda: tuple(geometries),
        entry=lambda key: SimpleNamespace(geometry=geometries[key]),
        metadata=lambda: {},
        asset_digest=digest,
        asset_size=path.stat().st_size,
    )
    monkeypatch.setattr(
        native_policy_module,
        "load_safetensors_header",
        lambda _path, **_identity: header,
    )
    policy = NativeDispatchPolicy(
        lambda found: path if found == digest else None,
        _ignore_diagnostic,
        minimax_h3_runtime_versions=lambda: {
            "torch": "2.13.0+cu130",
            "dinkster-kitchen": "0.2.31",
        },
    )
    arms = (
        "compat",
        {
            "compat": "compat-tag",
            "compat@native:cuda:0": "native-sm89",
            "compat@native:cuda:1": "native-sm120",
        },
    )
    attention_routes = {
        "compat@native:cuda:0": _sdpa_route_token(89),
        "compat@native:cuda:1": _sdpa_route_token(120),
    }

    def expected_identity(target: str) -> str:
        return minimax_h3_dit_runtime_identity(
            asset_digest=digest,
            asset_size=path.stat().st_size,
            role=role,
            diffusion_dtype="bfloat16",
            attention_route_token=attention_routes[target],
            runtime_facts=minimax_h3_dit_provider_facts(
                role,
                quantized=False,
                torch_version="2.13.0+cu130",
                dinkster_kitchen_version=None,
            ),
        )

    def select(run_id: str) -> ExecutionSelection:
        selection = asyncio.run(
            policy.select(
                "dinkster.load_diffusion_components",
                _h3_split_inputs(digest, role),
                arms,
                run_id=run_id,
                attention_routes=attention_routes,
            )
        )
        assert selection is not None
        return selection

    first = select("run-a")
    second = select("run-b")

    assert {first.target, second.target} == {"compat@native:cuda:0", "compat@native:cuda:1"}
    for selection in (first, second):
        assert selection.attention_route_token == attention_routes[selection.target]
        assert selection.cache_tag == expected_identity(selection.target)
    assert select("run-a") == first


def test_load_diffusion_components_rejects_wrong_h3_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "wrong-shape.safetensors"
    path.write_bytes(b"not-an-h3-checkpoint")
    digest = digest_file(path)
    layout = minimax_h3_dit_layout()
    geometries = {key: TensorGeometry(shape, FLOAT32) for key, shape in layout.keys.items()}
    key = "token_refiner.blocks.0.attn.qkv_proj.weight"
    geometries[key] = TensorGeometry((1, 1), FLOAT32)
    header = SimpleNamespace(
        path=path,
        keys=lambda: tuple(geometries),
        entry=lambda name: SimpleNamespace(geometry=geometries[name]),
        metadata=lambda: {},
        asset_digest=digest,
        asset_size=path.stat().st_size,
    )
    monkeypatch.setattr(
        native_policy_module,
        "load_safetensors_header",
        lambda _path, **_identity: header,
    )
    policy = NativeDispatchPolicy(
        lambda found: path if found == digest else None,
        _ignore_diagnostic,
        minimax_h3_runtime_versions=lambda: {
            "torch": "2.13.0+cu130",
            "dinkster-kitchen": "0.2.31",
        },
    )

    with pytest.raises(RuntimeError, match="geometry mismatch"):
        asyncio.run(
            policy.select(
                "dinkster.load_diffusion_components",
                _h3_split_inputs(digest, "fl2va-dit"),
                ARMS,
            )
        )


@pytest.mark.parametrize(
    "weight_dtype", [None, "default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]
)
def test_load_diffusion_components_selects_exact_five_trellis2_flows(
    monkeypatch: pytest.MonkeyPatch,
    weight_dtype: str | None,
) -> None:
    diagnostics: list[NativePolicyDiagnostic] = []
    policy = NativeDispatchPolicy(lambda _digest: None, diagnostics.append)
    plans = {
        role: ComponentPlan(role, Path(f"/{role}.safetensors"), object(), {}, {}, {})
        for role in _TRELLIS2_SPLIT_ROLES
    }
    observed: list[object] = []
    monkeypatch.setattr(
        policy,
        "_probe_trellis2_flow_transaction",
        lambda _digest, role: plans[role],
    )

    def identity(plan: object, dtype: DType) -> str:
        observed.extend((plan, dtype))
        return "native:dinkster.trellis2:" + "f" * 64

    monkeypatch.setattr(native_policy_module, "trellis2_split_model_runtime_identity", identity)
    inputs = _trellis2_split_inputs()
    if weight_dtype is not None:
        inputs["weight_dtype"] = _value(weight_dtype)

    selection = asyncio.run(policy.select("dinkster.load_diffusion_components", inputs, ARMS))

    assert selection is not None
    assert selection.target == "compat@native"
    assert selection.cache_tag == "native:dinkster.trellis2:" + "f" * 64
    plan = cast("Any", observed[0])
    assert (
        plan.structure,
        plan.shape,
        plan.shape_512,
        plan.texture,
        plan.texture_512,
    ) == tuple(plans[role] for role in _TRELLIS2_SPLIT_ROLES)
    assert observed[1] == BFLOAT16
    assert diagnostics == []


@pytest.mark.parametrize(
    ("roles", "match"),
    (
        (_TRELLIS2_SPLIT_ROLES[:-1], "missing diffusion component roles: texture-512"),
        ((*_TRELLIS2_SPLIT_ROLES, "structure"), "duplicate diffusion component role 'structure'"),
        (
            (*_TRELLIS2_SPLIT_ROLES[:-1], "foreign"),
            "unsupported diffusion component role 'foreign'",
        ),
        ((*_TRELLIS2_SPLIT_ROLES[:-1], 5), "diffusion component role must be a string"),
    ),
)
def test_load_diffusion_components_refuses_invalid_role_sets(
    monkeypatch: pytest.MonkeyPatch,
    roles: tuple[object, ...],
    match: str,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    monkeypatch.setattr(
        policy,
        "_probe_trellis2_flow_transaction",
        lambda _digest, role: ComponentPlan(
            role, Path(f"/{role}.safetensors"), object(), {}, {}, {}
        ),
    )

    with pytest.raises(RuntimeError, match=match):
        asyncio.run(
            policy.select("dinkster.load_diffusion_components", _trellis2_split_inputs(roles), ARMS)
        )


def test_load_diffusion_components_refuses_incomplete_and_mismatched_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    monkeypatch.setattr(
        policy,
        "_probe_trellis2_flow_transaction",
        lambda _digest, _role: object(),
    )
    inputs = _trellis2_split_inputs()
    del inputs["components.component_2.role"]
    with pytest.raises(RuntimeError, match="requires an asset and role"):
        asyncio.run(policy.select("dinkster.load_diffusion_components", inputs, ARMS))

    policy = NativeDispatchPolicy(lambda _digest: None, _ignore_diagnostic)
    monkeypatch.setattr(
        policy,
        "_probe_trellis2_flow_transaction",
        lambda _digest, role: None if role == "texture-512" else object(),
    )
    with pytest.raises(RuntimeError, match="texture-512.*not an exact TRELLIS.2 flow"):
        asyncio.run(
            policy.select("dinkster.load_diffusion_components", _trellis2_split_inputs(), ARMS)
        )


@pytest.mark.parametrize(
    "roles,profile", [(("clip_l",), "stable_diffusion"), (("clip_l", "clip_g"), "sdxl")]
)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_text_recipe_host_identity_matches_worker_recipe(
    monkeypatch: pytest.MonkeyPatch, roles: tuple[str, ...], profile: str, dtype: str
) -> None:
    from dinkster_inference.clip_text import CLIP_L_TEXT_CONFIG
    from dinkster_inference.text_recipes import resolve_text_recipe
    from test_inference_text_recipes import detected

    match = detected(*roles)[0]
    match = replace(
        match,
        descriptor=replace(match.descriptor, requires_text_recipe=True),
        components=tuple(
            (role, replace(plan, config=CLIP_L_TEXT_CONFIG)) for role, plan in match.components
        ),
    )
    digest = "blake3:" + "3" * 64
    binding_digest = "a" * 64
    policy = NativeDispatchPolicy(
        lambda _: None,
        _ignore_diagnostic,
        dtype_policy=lambda: {"diffusion": "auto", "textEncoder": dtype, "vae": "auto"},
    )
    monkeypatch.setattr(policy, "_probe_components_transaction", lambda _: (match,))
    monkeypatch.setattr(native_policy_module, "_embedding_binding_digest", lambda: binding_digest)
    selection = asyncio.run(
        policy.select(
            "dinkster.load_clip", {"text_encoder": _asset(digest), "type": _value(profile)}, ARMS
        )
    )
    binding = resolve_text_recipe(((match,),), profile)
    recipe = binding.recipe(
        (inference.WeightSourceRef(digest, "encoder.safetensors", 8),),
        dtype,
        embedding_binding_digest=binding_digest,
    )
    assert selection is not None and selection.target == "compat@native"
    assert selection.cache_tag == recipe.runtime_identity
    assert selection.text_dtype == dtype
    assert selection.diffusion_dtype == selection.vae_dtype == "unloaded"


@pytest.mark.parametrize(
    ("roles", "profile"),
    ((("clip_l", "clip_g"), "sdxl"), (("clip_l", "t5xxl"), "flux")),
)
def test_dual_clip_host_identity_matches_ordered_worker_recipe(
    monkeypatch: pytest.MonkeyPatch,
    roles: tuple[str, str],
    profile: str,
) -> None:
    from dinkster_inference.text_recipes import resolve_text_recipe
    from test_inference_text_recipes import detected

    matches = tuple(detected(role) for role in roles)
    configured = tuple(
        (
            replace(
                match,
                descriptor=replace(match.descriptor, requires_text_recipe=True),
                components=tuple(
                    (role, replace(plan, config=CLIP_L_TEXT_CONFIG))
                    for role, plan in match.components
                ),
            ),
        )
        for (match,), role in zip(matches, roles, strict=True)
    )
    digests = ("blake3:" + "1" * 64, "blake3:" + "2" * 64)
    probed: list[str] = []
    binding_digest = "e" * 64
    token = _sdpa_route_token(89)
    policy = NativeDispatchPolicy(lambda _: None, _ignore_diagnostic)

    def probe(digest: str) -> tuple[object, ...]:
        probed.append(digest)
        return configured[digests.index(digest)]

    monkeypatch.setattr(policy, "_probe_components_transaction", probe)
    monkeypatch.setattr(native_policy_module, "_embedding_binding_digest", lambda: binding_digest)
    inputs = {
        "text_encoder1": _asset(digests[0], "encoder-1.safetensors"),
        "text_encoder2": _asset(digests[1], "encoder-2.safetensors"),
        "device": _value("cpu"),
    }
    if profile != "sdxl":
        inputs["type"] = _value(profile)
    selection = asyncio.run(
        policy.select(
            "dinkster.load_dual_clip",
            inputs,
            ARMS,
            attention_routes={"compat@native": token},
        )
    )
    binding = resolve_text_recipe(configured, profile)
    assert selection is not None and selection.target == "compat@native"
    recipe = binding.recipe(
        tuple(
            inference.WeightSourceRef(digest, f"encoder-{index}.safetensors", 8)
            for index, digest in enumerate(digests, 1)
        ),
        selection.text_dtype or "",
        attention_policy="auto",
        attention_route_token=token,
        embedding_binding_digest=binding_digest,
    )

    assert selection.cache_tag == recipe.runtime_identity
    assert selection.text_dtype == default_text_dtype(binding.family_id).name
    assert selection.attention_route_token == token
    assert probed == list(digests)


def test_dual_clip_reversal_changes_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_inference_text_recipes import detected

    detected_by_digest = {
        "blake3:" + "1" * 64: detected("clip_l"),
        "blake3:" + "2" * 64: detected("t5xxl"),
    }
    policy = NativeDispatchPolicy(lambda _: None, _ignore_diagnostic)
    monkeypatch.setattr(
        policy,
        "_probe_components_transaction",
        lambda digest: detected_by_digest[digest],
    )

    def select(first: str, second: str) -> ExecutionSelection | None:
        return asyncio.run(
            policy.select(
                "dinkster.load_dual_clip",
                {
                    "text_encoder1": _asset(first),
                    "text_encoder2": _asset(second),
                    "type": _value("flux"),
                },
                ARMS,
            )
        )

    forward = select(*detected_by_digest)
    reverse = select(*reversed(detected_by_digest))
    assert forward is not None and reverse is not None
    assert forward.cache_tag != reverse.cache_tag


def test_dual_clip_repeated_sources_reach_recipe_resolution_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_inference_text_recipes import detected

    digest = "blake3:" + "1" * 64
    probed: list[str] = []
    diagnostics: list[NativePolicyDiagnostic] = []
    policy = NativeDispatchPolicy(lambda _: None, diagnostics.append)

    def probe(found: str) -> tuple[object, ...]:
        probed.append(found)
        return detected("clip_l")

    monkeypatch.setattr(policy, "_probe_components_transaction", probe)

    with pytest.raises(RuntimeError, match="one unambiguous clip_l"):
        asyncio.run(
            policy.select(
                "dinkster.load_dual_clip",
                {
                    "text_encoder1": _asset(digest),
                    "text_encoder2": _asset(digest),
                    "type": _value("flux"),
                },
                ARMS,
            )
        )
    assert probed == [digest, digest]
    assert [diagnostic.digest for diagnostic in diagnostics] == [digest, digest]


def test_dual_clip_unresolved_recipe_always_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_inference_text_recipes import detected

    digests = ("blake3:" + "1" * 64, "blake3:" + "2" * 64)
    diagnostics: list[NativePolicyDiagnostic] = []
    policy = NativeDispatchPolicy(lambda _: None, diagnostics.append)
    monkeypatch.setattr(
        policy,
        "_probe_components_transaction",
        lambda _digest: detected("clip_l"),
    )
    inputs = {
        "text_encoder1": _asset(digests[0]),
        "text_encoder2": _asset(digests[1]),
        "type": _value("flux"),
    }

    with pytest.raises(RuntimeError, match="no native text recipe for the ordered sources"):
        asyncio.run(policy.select("dinkster.load_dual_clip", inputs, ARMS))
    assert [diagnostic.digest for diagnostic in diagnostics] == list(digests)
    native_only = ("compat@native", {"compat@native": "native-default"})
    with pytest.raises(RuntimeError, match="no native text recipe for the ordered sources"):
        asyncio.run(policy.select("dinkster.load_dual_clip", inputs, native_only))


def test_dual_clip_resolved_recipe_requires_native_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_inference_text_recipes import detected

    policy = NativeDispatchPolicy(lambda _: None, _ignore_diagnostic)
    detected_by_digest = {
        "blake3:" + "1" * 64: detected("clip_l"),
        "blake3:" + "2" * 64: detected("clip_g"),
    }
    monkeypatch.setattr(
        policy,
        "_probe_components_transaction",
        lambda digest: detected_by_digest[digest],
    )

    with pytest.raises(RuntimeError, match="requires an available native arm"):
        asyncio.run(
            policy.select(
                "dinkster.load_dual_clip",
                {
                    "text_encoder1": _asset("blake3:" + "1" * 64),
                    "text_encoder2": _asset("blake3:" + "2" * 64),
                    "type": _value("sdxl"),
                },
                ("compat", {"compat": "compat-tag"}),
            )
        )


@pytest.mark.parametrize(
    ("inputs", "message"),
    (
        ({}, "missing its text_encoder1"),
        ({"text_encoder1": _asset("blake3:" + "1" * 64)}, "missing its text_encoder2"),
        (
            {
                "text_encoder1": _asset(3),
                "text_encoder2": _asset("blake3:" + "2" * 64),
            },
            "text_encoder1 asset has missing or malformed digest metadata",
        ),
        (
            {
                "text_encoder1": _asset("blake3:" + "1" * 64),
                "text_encoder2": _asset("blake3:" + "2" * 64),
                "type": _value(4),
            },
            "type is malformed",
        ),
    ),
)
def test_dual_clip_rejects_malformed_inputs(inputs: dict[str, Value], message: str) -> None:
    policy = NativeDispatchPolicy(lambda _: None, _ignore_diagnostic)
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(policy.select("dinkster.load_dual_clip", inputs, ARMS))


def test_unknown_text_recipe_reports_detected_components_without_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_inference_text_recipes import detected

    match = detected("clip_l")[0]
    match = replace(match, descriptor=replace(match.descriptor, requires_text_recipe=True))
    diagnostics: list[NativePolicyDiagnostic] = []
    policy = NativeDispatchPolicy(lambda _: None, diagnostics.append)
    monkeypatch.setattr(policy, "_probe_components_transaction", lambda _: (match,))
    selection = asyncio.run(
        policy.select(
            "dinkster.load_clip",
            {"text_encoder": _asset("blake3:" + "3" * 64), "type": _value("unknown")},
            ARMS,
        )
    )
    assert selection is not None and selection.target == "compat"
    assert selection.cache_tag == "compat-tag"
    assert len(diagnostics) == 1
    assert "unknown text recipe" in diagnostics[0].reasons[0]
    assert "example.text_architecture/clip_l" in diagnostics[0].reasons[0]


def test_load_latent_keeps_its_native_owner_without_model_probing() -> None:
    def unexpected_probe(_digest: str) -> Path | None:
        pytest.fail("latent loading must not probe or convert model checkpoints")

    policy = NativeDispatchPolicy(unexpected_probe, _ignore_diagnostic)
    assert asyncio.run(policy.select("dinkster.load_latent", {}, ARMS)) is None
