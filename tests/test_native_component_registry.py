"""New component registrations use the existing native binding contract."""

from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import test_native_arm as fixtures
from dinkster_assets import digest_file
from dinkster_compat_comfy import native_arm
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    ComponentBinding,
    Conditioning,
    bind_component_conditioning,
    build_runtime_identity,
    component_catalog,
    default_diffusion_dtype,
    default_text_dtype,
    default_vae_dtype,
    load_safetensors_header,
    plan_native,
)
from dinkster_inference.component_checkpoint import ComponentCheckpointPlan
from dinkster_inference.component_registry import ComponentRegistry
from dinkster_schema import ComboWidget
from test_inference_component_registry import synthetic_descriptor
from test_native_policy import ARMS, _asset, _value

from dinkster.native_policy import NativeDispatchPolicy


class SyntheticRuntime:
    def __init__(self, diffusion: object, family: object, *, runtime_identity: str) -> None:
        self.assembled = SimpleNamespace(diffusion=diffusion)
        self.family = family
        self.runtime_identity = runtime_identity

    def prepare(self, carrier: object) -> Conditioning:
        del carrier
        return Conditioning(fixtures.FakeTensor((1, 2, 3), "synthetic"))


class DerivedRuntime(SyntheticRuntime):
    pass


def test_dual_clip_schema_registers_ordered_assets_and_legacy_alias() -> None:
    from dinkster_compat_comfy.native import NATIVE_NODES, LoadDualClip

    schema = LoadDualClip.schema()
    assert schema.node_type == "dinkster.load_dual_clip"
    assert schema.aliases == ("DualCLIPLoader",)
    assert tuple(spec.id for spec in schema.inputs) == (
        "text_encoder1",
        "text_encoder2",
        "type",
        "device",
    )
    assert schema.inputs[0].type.types == schema.inputs[1].type.types == ("dinkster.asset",)
    assert LoadDualClip.CLIP_TYPES == (
        "sdxl",
        "sd3",
        "flux",
        "hunyuan_video",
        "hidream",
        "hunyuan_image",
        "hunyuan_video_15",
        "kandinsky5",
        "kandinsky5_image",
        "ltxv",
        "newbie",
        "ace",
    )
    assert schema.inputs[2].default == "sdxl"
    assert schema.inputs[2].widget == ComboWidget(options=LoadDualClip.CLIP_TYPES)
    assert schema.inputs[3].default == "default"
    assert schema.inputs[3].widget == ComboWidget(options=("default", "cpu"))
    assert LoadDualClip in NATIVE_NODES


@pytest.mark.parametrize(
    "node,input_name,kind,role,dtype",
    (
        ("diffusion_model", "diffusion_model", "model", "diffusion", "float32"),
        ("clip", "text_encoder", "text", "words", "float16"),
        ("vae", "vae", "codec", "pixels", "float32"),
    ),
)
@pytest.mark.parametrize("combined", (False, True))
def test_registered_architecture_drives_host_detection_and_dtype_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    node: str,
    input_name: str,
    kind: str,
    role: str,
    dtype: str,
    combined: bool,
) -> None:
    descriptor = replace(
        synthetic_descriptor(),
        default_diffusion_dtype=FLOAT32,
        default_text_dtype=FLOAT16,
        vae_dtypes=(FLOAT32,),
    )
    registry = ComponentRegistry()
    registry.register(descriptor)
    monkeypatch.setattr(component_catalog, "default_component_registry", lambda: registry)
    roles = descriptor.roles if combined else (role,)
    header = json.dumps(
        {
            f"{part}.weight": {
                "dtype": "BF16",
                "shape": [2, 2],
                "data_offsets": [index * 8, (index + 1) * 8],
            }
            for index, part in enumerate(roles)
        }
    ).encode()
    path = tmp_path / "synthetic.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(8 * len(roles)))
    digest = digest_file(path)
    policy = NativeDispatchPolicy(lambda found: path if found == digest else None, lambda _: None)
    inputs = {input_name: _asset(digest)}
    if kind == "text":
        inputs["type"] = _value("not_the_detected_family")
    selection = asyncio.run(policy.select(f"dinkster.load_{node}", inputs, ARMS))
    assert selection is not None
    assert selection.target == "compat@native"
    assert (
        getattr(selection, {"model": "diffusion", "text": "text", "codec": "vae"}[kind] + "_dtype")
        == dtype
    )
    source = load_safetensors_header(path, asset_digest=digest, asset_size=path.stat().st_size)
    _, selected_role, plan = registry.select(source, path, kind)
    assert selected_role == role
    assert selection.cache_tag == descriptor.component_identity(role, plan, dtype)
    assert default_diffusion_dtype(descriptor.id) == FLOAT32
    assert default_text_dtype(descriptor.id) == FLOAT16
    assert default_vae_dtype(descriptor.id) == FLOAT32
    with pytest.raises(ValueError, match="no VAE dtype supported by device set"):
        default_vae_dtype(descriptor.id, (FLOAT16,))


@pytest.mark.parametrize(
    "node,input_name,role,dtype",
    (
        ("dinkster.load_diffusion_model", "diffusion_model", "diffusion", "bfloat16"),
        ("dinkster.load_clip", "text_encoder", "gemma2_2b", "float32"),
        ("dinkster.load_vae", "vae", "vae", "bfloat16"),
        ("dinkster.load_checkpoint", "checkpoint", None, None),
    ),
)
def test_official_lumina_header_selects_each_registered_loader(
    node: str,
    input_name: str,
    role: str | None,
    dtype: str | None,
) -> None:
    from test_inference_lumina2 import _NETAYUME, _NETAYUME_BLAKE3, _NETAYUME_SIZE

    if not _NETAYUME.exists():
        pytest.skip("official NetaYume checkpoint absent")
    assert _NETAYUME.stat().st_size == _NETAYUME_SIZE
    digest = "blake3:" + _NETAYUME_BLAKE3
    source = load_safetensors_header(
        _NETAYUME,
        asset_digest=digest,
        asset_size=_NETAYUME_SIZE,
    )
    policy = NativeDispatchPolicy(
        lambda found: _NETAYUME if found == digest else None,
        lambda _diagnostic: None,
        dtype_policy=lambda: {
            "diffusion": "bfloat16",
            "textEncoder": "float32",
            "vae": "bfloat16",
        },
    )
    inputs = {input_name: _asset(digest, _NETAYUME.name)}
    if node == "dinkster.load_clip":
        inputs["type"] = _value("lumina2")
    selection = asyncio.run(policy.select(node, inputs, ARMS))
    assert selection is not None and selection.target == "compat@native"
    descriptor = component_catalog.default_component_registry().get("dinkster.lumina2")
    assert descriptor is not None
    detected = dict(descriptor.detector(source, _NETAYUME))
    assert set(detected) == {"diffusion", "gemma2_2b", "vae"}
    if role is None:
        plan = plan_native(checkpoint=source)
        assert isinstance(plan, ComponentCheckpointPlan)
        assert plan.family.id == descriptor.id
        assert set(plan.components) == set(detected)
        expected_identity = build_runtime_identity(
            plan.family.id,
            plan.identity_components,
            diffusion_dtype=BFLOAT16,
            text_dtype=FLOAT32,
            vae_dtype=BFLOAT16,
            fp8_matmul=False,
        )
    else:
        planned = detected[role]
        assert planned.component == role
        assert dtype is not None
        expected_identity = descriptor.component_identity(role, planned, dtype)
    assert selection.cache_tag == expected_identity


@pytest.mark.parametrize("requested_type", ["stable_diffusion", "unknown"])
def test_classic_text_detection_uses_an_explicit_recipe(
    monkeypatch: pytest.MonkeyPatch,
    requested_type: str,
) -> None:
    from dinkster_inference import WeightSourceRef
    from dinkster_inference.clip_text import CLIP_L_TEXT_CONFIG, clip_text_layout
    from dinkster_inference.text_recipes import resolve_text_recipe
    from test_inference_text_recipes import Header

    from dinkster import native_policy

    header = Header(clip_text_layout(CLIP_L_TEXT_CONFIG))
    registry = component_catalog.default_component_registry()
    matches = registry.detect(header, header.path)
    binding = resolve_text_recipe((matches,), "stable_diffusion")
    assert binding.components[0].role == "clip_l"
    assert binding.components[0].plan is matches[0].plan_for("clip_l")
    diagnostics: list[Any] = []
    policy = NativeDispatchPolicy(lambda _: None, diagnostics.append)
    monkeypatch.setattr(policy, "_probe_components_transaction", lambda _: matches)
    binding_digest = "b" * 64
    monkeypatch.setattr(native_policy, "_embedding_binding_digest", lambda: binding_digest)
    selection = asyncio.run(
        policy.select(
            "dinkster.load_clip",
            {"text_encoder": _asset(header.asset_digest), "type": _value(requested_type)},
            ARMS,
        )
    )
    assert selection is not None
    if requested_type == "unknown":
        assert selection.target == "compat"
        assert selection.cache_tag == "compat-tag"
        assert selection.text_dtype is None
        assert len(diagnostics) == 1 and diagnostics[0].kind == "unsupported"
        assert "unknown text recipe" in diagnostics[0].reasons[0]
    else:
        assert selection.target == "compat@native"
        assert selection.text_dtype is not None
        recipe = binding.recipe(
            (WeightSourceRef(header.asset_digest, header.path.name, header.asset_size),),
            selection.text_dtype,
            embedding_binding_digest=binding_digest,
        )
        assert selection.cache_tag == recipe.runtime_identity
        assert not diagnostics


@pytest.mark.parametrize("family_id", ("test.synthetic", "test.renamed"))
def test_registered_runtime_subclass_composes_without_family_dispatch_edits(
    monkeypatch: pytest.MonkeyPatch, family_id: str
) -> None:
    import dinkster_inference as inference

    base_descriptor = synthetic_descriptor()
    descriptor = replace(
        base_descriptor,
        family=replace(base_descriptor.family, id=family_id, display_name="Synthetic"),
        runtime_class=f"{__name__}:SyntheticRuntime",
        runtime_with_family=True,
        prepare_conditioning="prepare",
    )
    registry = ComponentRegistry()
    registry.register(descriptor)
    monkeypatch.setattr(component_catalog, "default_component_registry", lambda: registry)
    diffusion_identity = f"native:{family_id}:" + "1" * 64
    text_identity = f"native:{family_id}:" + "2" * 64
    diffusion = object()
    base = DerivedRuntime(diffusion, descriptor.family, runtime_identity=diffusion_identity)
    recipe = SimpleNamespace(
        family_id=descriptor.id,
        runtime_identity=diffusion_identity,
        sources=(SimpleNamespace(role="diffusion"),),
    )
    handle = cast("Any", SimpleNamespace(runtime=base, recipe=recipe))
    carrier = fixtures._component_conditioning_carrier()
    positive = bind_component_conditioning(
        carrier, ComponentBinding("words", descriptor.id, text_identity)
    )
    negative = bind_component_conditioning(
        fixtures._component_conditioning_carrier(),
        ComponentBinding("words", descriptor.id, text_identity),
    )
    result = native_arm.resolve_component_execution(handle, positive, negative, inference)
    assert result is not None
    runtime, rows, negative_rows = result
    assert isinstance(runtime, DerivedRuntime)
    assert runtime is not base
    assert runtime.assembled.diffusion is diffusion
    assert runtime.family is descriptor.family
    assert (
        runtime.runtime_identity
        == inference.compose_execution(
            descriptor.id, {"diffusion": diffusion_identity, "words": text_identity}
        ).execution_identity
    )
    assert len(cast("list[Any]", rows)) == len(cast("list[Any]", negative_rows)) == 1
    assert base.runtime_identity == diffusion_identity
    assert tuple(source.role for source in recipe.sources) == ("diffusion",)
    _, positive_binding = inference.split_component_conditioning(positive)
    _, negative_binding = inference.split_component_conditioning(negative)
    assert (
        positive_binding == negative_binding == ComponentBinding("words", family_id, text_identity)
    )
    assert cast("Any", negative_rows) is not cast("Any", rows)

    unrelated = SimpleNamespace(runtime=object(), recipe=recipe)
    with pytest.raises(TypeError, match="native Synthetic diffusion component"):
        native_arm.resolve_component_execution(cast("Any", unrelated), positive, [], inference)

    wrong_role = bind_component_conditioning(
        carrier, ComponentBinding("pixels", descriptor.id, text_identity)
    )
    with pytest.raises(ValueError, match="wrong component role"):
        native_arm.resolve_component_execution(handle, wrong_role, [], inference)


def test_registered_runtime_rejects_recipe_from_different_producer_before_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_inference as inference

    base_descriptor = synthetic_descriptor()
    renamed_descriptor = replace(
        base_descriptor,
        family=replace(
            base_descriptor.family,
            id="test.renamed",
            display_name="Renamed Synthetic",
        ),
        runtime_class=f"{__name__}:SyntheticRuntime",
        runtime_with_family=True,
        prepare_conditioning="prepare",
    )
    registry = ComponentRegistry()
    registry.register(renamed_descriptor)
    monkeypatch.setattr(component_catalog, "default_component_registry", lambda: registry)

    class UnevaluatedRuntime(SyntheticRuntime):
        def prepare(self, carrier: object) -> Conditioning:
            raise AssertionError(f"conditioning must remain unevaluated: {carrier!r}")

    base_recipe = fixtures._recipe()
    recipe = replace(
        base_recipe,
        sources=(inference.WeightSourceBinding("diffusion", base_recipe.sources[0].source),),
        family_id=renamed_descriptor.id,
        component_identity=("family=test.renamed", "role=diffusion"),
    )
    runtime = UnevaluatedRuntime(
        object(), base_descriptor.family, runtime_identity=recipe.runtime_identity
    )
    text_identity = "native:test.renamed:" + "2" * 64
    handle = cast("Any", SimpleNamespace(runtime=runtime, recipe=recipe))
    positive = bind_component_conditioning(
        fixtures._component_conditioning_carrier(),
        ComponentBinding("words", renamed_descriptor.id, text_identity),
    )

    with pytest.raises(
        ValueError,
        match=(
            "runtime producer family 'test.synthetic' does not match "
            "reconstruction recipe family 'test.renamed'"
        ),
    ):
        native_arm.resolve_component_execution(handle, positive, positive, inference)
