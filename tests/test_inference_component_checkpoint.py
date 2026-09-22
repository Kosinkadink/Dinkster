"""Checkpoint contracts preserve role ownership rather than guessing output slots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_inference import BFLOAT16, AssemblyError, ComponentPlan, TensorGeometry, WeightEntry
from dinkster_inference.component_catalog import default_component_registry
from dinkster_inference.component_checkpoint import (
    ComponentCheckpointPlan,
    component_source_keys,
    plan_component_checkpoint,
)
from dinkster_inference.component_registry import ComponentRegistry
from dinkster_inference.quantization import LayerQuant
from test_inference_assembly import ltxav_checkpoint
from test_inference_component_registry import Header, synthetic_descriptor


def part(role: str, *, path: Path = Path("checkpoint.safetensors")) -> ComponentPlan[object]:
    return ComponentPlan(
        role,
        path,
        "synthetic",
        {"layer.weight": f"{role}.layer.weight"},
        {"layer.weight": BFLOAT16},
        {},
        identity_facts=("asset_digest=blake3:" + "1" * 64, "asset_size=128"),
        runtime_facts=(f"operator={role}",),
    )


def test_checkpoint_roles_are_canonical_immutable_and_keep_the_original_plans() -> None:
    roles = ("diffusion", "words", "second_words", "pixels", "audio")
    descriptor = replace(
        synthetic_descriptor(),
        roles=roles,
        text_encoder_roles=("words", "second_words"),
        codec_roles=("pixels", "audio"),
    )
    components = {role: part(role) for role in roles}
    mutable_roles = list(reversed(tuple(components.items())))
    unclaimed = ["extra.weight"]
    diagnostics = ["extra source tensor"]
    plan = ComponentCheckpointPlan(
        descriptor, cast("Any", mutable_roles), cast("Any", unclaimed), cast("Any", diagnostics)
    )
    mutable_roles.clear()
    unclaimed.clear()
    diagnostics.clear()
    assert plan.family is descriptor.family
    assert tuple(plan.components) == roles
    assert plan.identity_components == tuple(components.values())
    assert all(plan.components[role] is component for role, component in components.items())
    assert plan.unclaimed == ("extra.weight",)
    assert plan.diagnostics == ("extra source tensor",)
    with pytest.raises(TypeError):
        cast("Any", plan.components)["words"] = part("words")
    with pytest.raises(FrozenInstanceError):
        cast("Any", plan).family = descriptor.family
    assert not hasattr(plan, "text")
    assert not hasattr(plan, "vae")


def test_checkpoint_keeps_optional_roles_absent() -> None:
    descriptor = synthetic_descriptor()
    diffusion = part("diffusion")
    plan = ComponentCheckpointPlan(descriptor, (("diffusion", diffusion),))
    assert dict(plan.components) == {"diffusion": diffusion}
    assert plan.identity_components == (diffusion,)


@pytest.mark.parametrize(
    "components,message",
    [
        ((("diffusion", part("diffusion")), ("diffusion", part("diffusion"))), "duplicate"),
        ((("other", part("other")),), "not declared"),
        ((("words", part("words")),), "requires.*model role"),
        ((("diffusion", part("words")),), "contains a plan for"),
    ],
)
def test_checkpoint_rejects_ambiguous_role_contracts(components: Any, message: str) -> None:
    with pytest.raises(AssemblyError, match=message):
        ComponentCheckpointPlan(synthetic_descriptor(), components)


@pytest.mark.parametrize("quantized", [False, True])
def test_checkpoint_rejects_duplicate_source_claims(quantized: bool) -> None:
    diffusion = part("diffusion")
    text = part("words")
    if quantized:
        shared_scale = "shared.weight_scale"
        diffusion = replace(
            diffusion,
            quant={
                "layer": LayerQuant(
                    "layer", "float8_e4m3fn", "diffusion.layer.weight", shared_scale
                )
            },
        )
        text = replace(
            text,
            quant={
                "layer": LayerQuant("layer", "float8_e4m3fn", "words.layer.weight", shared_scale)
            },
        )
    else:
        text = replace(text, keys=diffusion.keys)
    with pytest.raises(AssemblyError, match="claimed by both 'diffusion' and 'words'"):
        ComponentCheckpointPlan(synthetic_descriptor(), (("diffusion", diffusion), ("words", text)))


def test_equal_tensor_names_in_distinct_sources_are_not_duplicate_claims() -> None:
    diffusion = replace(part("diffusion"), keys={"layer.weight": "weight"})
    text = replace(part("words", path=Path("text.safetensors")), keys={"layer.weight": "weight"})
    plan = ComponentCheckpointPlan(
        synthetic_descriptor(), (("diffusion", diffusion), ("words", text))
    )
    assert plan.identity_components == (diffusion, text)


def test_source_claims_keep_all_quantization_artifacts() -> None:
    component = replace(
        part("diffusion"),
        quant={
            "layer": LayerQuant(
                "layer",
                "float8_e4m3fn",
                "diffusion.layer.weight",
                "scale",
                input_scale="input_scale",
                config="config",
                weight_scale_2="scale_2",
                pre_quant_scale="pre_scale",
                payloads={"extra": "payload"},
            )
        },
    )
    assert component_source_keys(component) == {
        "diffusion.layer.weight",
        "scale",
        "input_scale",
        "config",
        "scale_2",
        "pre_scale",
        "payload",
    }


@dataclass(frozen=True)
class CheckpointHeader(Header):
    path: Path = Path("checkpoint.safetensors")


@pytest.mark.parametrize("combined", [False, True])
def test_one_descriptor_resolves_combined_and_scoped_split_sources(combined: bool) -> None:
    seen: list[ComponentCheckpointPlan] = []
    descriptor = replace(
        synthetic_descriptor(),
        checkpoint_loader="example.runtime:load",
        checkpoint_validator=lambda plan: seen.append(plan),
    )
    registry = ComponentRegistry()
    registry.register(descriptor)
    if combined:
        plan = plan_component_checkpoint(
            CheckpointHeader(tuple(f"{role}.weight" for role in descriptor.roles)),
            component_registry=registry,
        )
    else:
        plan = plan_component_checkpoint(
            component_sources={
                role: CheckpointHeader((f"{role}.weight",), Path(f"{role}.safetensors"))
                for role in reversed(descriptor.roles)
            },
            component_registry=registry,
        )
    assert plan.descriptor is descriptor
    assert tuple(plan.components) == descriptor.roles
    assert seen == [plan] and seen[0] is plan
    assert plan.unclaimed == ()
    for role, component in plan.components.items():
        assert component.keys == {"weight": f"{role}.weight"}
        assert component.config == "synthetic"


def test_unknown_checkpoint_does_not_guess_a_composition() -> None:
    registry = ComponentRegistry()
    registry.register(replace(synthetic_descriptor(), checkpoint_loader="example.runtime:load"))
    with pytest.raises(AssemblyError, match=r"detected components=\(\)"):
        plan_component_checkpoint(
            CheckpointHeader(("unknown.weight",)), component_registry=registry
        )


def test_detected_components_without_output_behavior_name_the_missing_contract() -> None:
    registry = ComponentRegistry()
    registry.register(synthetic_descriptor())
    with pytest.raises(
        AssemblyError, match="test.synthetic/diffusion.*no declared checkpoint composition loader"
    ):
        plan_component_checkpoint(
            CheckpointHeader(("diffusion.weight",)), component_registry=registry
        )


def test_ltxav_all_in_one_checkpoint_has_a_generic_composition_contract() -> None:
    plan = plan_component_checkpoint(checkpoint=ltxav_checkpoint())
    assert plan.descriptor.id == "dinkster.ltxav"
    assert tuple(plan.components) == ("diffusion", "text_projection", "connectors", "vae")


def test_composition_ambiguity_does_not_depend_on_registration_order() -> None:
    first = replace(synthetic_descriptor(), checkpoint_loader="example.runtime:load")
    second = replace(first, family=replace(first.family, id="test.another"))
    for descriptors in ((first, second), (second, first)):
        registry = ComponentRegistry()
        for descriptor in descriptors:
            registry.register(descriptor)
        with pytest.raises(AssemblyError, match="multiple complete compositions"):
            plan_component_checkpoint(
                CheckpointHeader(("diffusion.weight",)), component_registry=registry
            )


def test_scoped_role_mismatch_cannot_select_a_different_detected_role() -> None:
    registry = ComponentRegistry()
    registry.register(replace(synthetic_descriptor(), checkpoint_loader="example.runtime:load"))
    with pytest.raises(AssemblyError, match="requires role 'words'; detected.*pixels"):
        plan_component_checkpoint(
            diffusion=CheckpointHeader(("diffusion.weight",)),
            component_sources={"words": CheckpointHeader(("pixels.weight",))},
            component_registry=registry,
        )


def test_explicit_split_source_takes_precedence_over_combined_role() -> None:
    registry = ComponentRegistry()
    registry.register(replace(synthetic_descriptor(), checkpoint_loader="example.runtime:load"))
    plan = plan_component_checkpoint(
        CheckpointHeader(("diffusion.weight", "words.weight")),
        diffusion=CheckpointHeader(("diffusion.weight",), Path("split.safetensors")),
        component_registry=registry,
    )
    assert plan.components["diffusion"].path == Path("split.safetensors")
    assert plan.components["words"].path == Path("checkpoint.safetensors")
    assert plan.unclaimed == ("checkpoint.safetensors:diffusion.weight",)


def test_unclaimed_source_keys_remain_visible_without_a_guessed_role() -> None:
    registry = ComponentRegistry()
    registry.register(replace(synthetic_descriptor(), checkpoint_loader="example.runtime:load"))
    plan = plan_component_checkpoint(
        CheckpointHeader(("diffusion.weight", "unknown.weight")), component_registry=registry
    )
    assert tuple(plan.components) == ("diffusion",)
    assert plan.unclaimed == ("checkpoint.safetensors:unknown.weight",)


@dataclass(frozen=True)
class GeometryHeader:
    path: Path
    geometries: Mapping[str, TensorGeometry]

    def keys(self) -> tuple[str, ...]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}


@pytest.mark.parametrize("variant", ["3b_swiglu", "7b_swiglu", "7b_mlp"])
@pytest.mark.parametrize("combined", [False, True])
def test_seedvr2_checkpoint_preserves_published_component_key_maps(
    variant: str,
    combined: bool,
) -> None:
    from dinkster_inference import SEEDVR2_CONFIGS, plan_native, seedvr2_layout, seedvr2_vae_layout
    from dinkster_inference.assembly import FLUX_DIFFUSION_PREFIX, FLUX_VAE_PREFIX

    config = next(config for config in SEEDVR2_CONFIGS if config.variant == variant)
    layouts = {"diffusion": seedvr2_layout(config), "vae": seedvr2_vae_layout()}
    prefixes = {"diffusion": FLUX_DIFFUSION_PREFIX, "vae": FLUX_VAE_PREFIX}
    sources = {
        role: GeometryHeader(
            Path(f"{role}.safetensors"),
            {key: TensorGeometry(shape, BFLOAT16) for key, shape in layout.items()},
        )
        for role, layout in layouts.items()
    }
    descriptor = default_component_registry().get("dinkster.seedvr2")
    assert descriptor is not None
    expected = {
        role: dict(descriptor.detector(source, source.path, bind_asset_identity=False))[role]
        for role, source in sources.items()
    }
    if combined:
        checkpoint = GeometryHeader(
            Path("seedvr2.safetensors"),
            {
                prefixes[role] + key: geometry
                for role, source in sources.items()
                for key, geometry in source.geometries.items()
            },
        )
        plan = plan_native(checkpoint=checkpoint)
    else:
        plan = plan_native(diffusion=sources["diffusion"], vae=sources["vae"])
    assert isinstance(plan, ComponentCheckpointPlan)
    assert plan.descriptor.id == descriptor.id
    assert tuple(plan.components) == ("diffusion", "vae")
    assert plan.unclaimed == ()
    for role, component in plan.components.items():
        assert component.config == expected[role].config
        assert dict(component.dtypes) == dict(expected[role].dtypes)
        assert dict(component.keys) == {
            key: (prefixes[role] if combined else "") + original
            for key, original in expected[role].keys.items()
        }


@pytest.mark.parametrize("real_header", [False, True])
def test_checkpoint_identity_is_independent_of_header_asset_binding(real_header: bool) -> None:
    from dinkster_inference import load_safetensors_header, runtime_component_identity
    from dinkster_inference.assembly import plan_lumina2_assembly
    from test_inference_lumina2 import (
        _NETAYUME,
        _NETAYUME_BLAKE3,
        _NETAYUME_SIZE,
        _checkpoint_geometries,
        _Source,
    )

    if real_header:
        if not _NETAYUME.exists():
            pytest.skip("official NetaYume checkpoint absent")
        bound = load_safetensors_header(
            _NETAYUME, asset_digest="blake3:" + _NETAYUME_BLAKE3, asset_size=_NETAYUME_SIZE
        )
        unbound = load_safetensors_header(_NETAYUME)
    else:
        bound = _Source(Path("lumina.safetensors"), _checkpoint_geometries())
        unbound = GeometryHeader(bound.path, bound.geometries)
    descriptor = default_component_registry().get("dinkster.lumina2")
    assert descriptor is not None
    registry = ComponentRegistry()
    for registered in default_component_registry():
        registry.register(
            replace(registered, checkpoint_loader="example.runtime:load")
            if registered is descriptor
            else registered
        )
    expected = plan_lumina2_assembly(checkpoint=bound)
    plans = [
        plan_component_checkpoint(source, component_registry=registry)
        for source in (bound, unbound)
    ]
    assert plans[0].identity_components == plans[1].identity_components
    assert plans[0].identity_components == expected.identity_components
    for plan in plans:
        assert runtime_component_identity(plan.family.id, plan.identity_components) == (
            runtime_component_identity(expected.family.id, expected.identity_components)
        )
        assert all(
            not fact.startswith(("asset_digest=", "asset_size="))
            for component in plan.identity_components
            for fact in component.identity_facts
        )
    split_bound = dict(descriptor.detector(bound, bound.path))
    for component in split_bound.values():
        assert any(fact.startswith("asset_digest=") for fact in component.identity_facts)
        assert any(fact.startswith("asset_size=") for fact in component.identity_facts)


@pytest.mark.parametrize(
    "role,section",
    [
        ("dit", "diffusion"),
        ("dinov3-vision-conditioner", "dinov3_vith"),
        ("gaussian-decoder", "gaussian_decoder"),
    ],
)
def test_real_triposplat_header_plans_keep_geometry_separate_from_asset_identity(
    role: str, section: str
) -> None:
    from dinkster_inference import FLOAT16
    from dinkster_inference.component_registry import component_plans
    from test_inference_triposplat_components import golden_shapes

    @dataclass(frozen=True)
    class BoundHeader(GeometryHeader):
        asset_digest: str = "blake3:" + "1" * 64
        asset_size: int = 1234

    source = GeometryHeader(
        Path("triposplat.safetensors"),
        {key: TensorGeometry(shape, FLOAT16) for key, shape in golden_shapes(section).items()},
    )
    bound_source = BoundHeader(source.path, source.geometries)
    descriptor = default_component_registry().get("dinkster.triposplat")
    assert descriptor is not None
    geometric = dict(descriptor.detector(source, source.path, bind_asset_identity=False))
    identified_geometry = dict(
        descriptor.detector(bound_source, source.path, bind_asset_identity=False)
    )
    bound = dict(descriptor.detector(bound_source, source.path))
    assert role in geometric and geometric == identified_geometry
    geometry_plan = component_plans(geometric[role])[0]
    bound_plan = component_plans(bound[role])[0]
    assert bound_plan == replace(
        geometry_plan,
        identity_facts=(
            *geometry_plan.identity_facts,
            f"asset_digest={bound_source.asset_digest}",
            f"asset_size={bound_source.asset_size}",
        ),
    )


def test_geometry_only_unknown_header_runs_every_registered_detector() -> None:
    source = GeometryHeader(Path("unknown.safetensors"), {})
    assert default_component_registry().detect(source, source.path, bind_asset_identity=False) == ()


@dataclass(frozen=True)
class GroupedRole:
    role: str
    identity_components: tuple[ComponentPlan[object], ...]


def test_compound_role_keeps_original_result_and_all_atomic_names() -> None:
    first, second, text = part("structure"), part("shape"), part("words")
    grouped = GroupedRole("diffusion", (first, second))
    plan = ComponentCheckpointPlan(
        synthetic_descriptor(), (("words", text), ("diffusion", grouped))
    )
    assert plan.role_plans == (("diffusion", grouped), ("words", text))
    assert plan.role_plans[0][1] is grouped
    assert plan.identity_components == (first, second, text)
    assert tuple(plan.components) == ("structure", "shape", "words")
    assert plan.components["structure"] is first
    assert plan.components["shape"] is second


def test_atomic_names_must_be_unique_across_distinct_public_roles() -> None:
    grouped = GroupedRole("diffusion", (part("words"),))
    other = part("words", path=Path("another.safetensors"))
    with pytest.raises(AssemblyError, match="duplicate checkpoint atomic component name 'words'"):
        ComponentCheckpointPlan(synthetic_descriptor(), (("diffusion", grouped), ("words", other)))


def test_checkpoint_retains_wan_tokenizer_source_without_changing_atomic_identity() -> None:
    from dinkster_inference import UMT5_XXL_CONFIG
    from dinkster_inference.assembly import Wan21StandaloneComponentPlan

    text = replace(part("umt5xxl"), config=UMT5_XXL_CONFIG)
    wrapper = Wan21StandaloneComponentPlan(
        "umt5xxl", cast("Any", text), tokenizer_source_key="text_encoders.umt5xxl.spiece_model"
    )
    descriptor = replace(
        synthetic_descriptor(),
        roles=("diffusion", "umt5xxl"),
        text_encoder_roles=("umt5xxl",),
        codec_roles=(),
    )
    diffusion = part("diffusion")
    plan = ComponentCheckpointPlan(descriptor, (("umt5xxl", wrapper), ("diffusion", diffusion)))
    assert plan.role_plans[1][1] is wrapper
    assert wrapper.tokenizer_source_key == "text_encoders.umt5xxl.spiece_model"
    assert plan.components["umt5xxl"] is text
    assert plan.identity_components == (diffusion, text)


@pytest.mark.parametrize("valid_registered", [False, True])
def test_incomplete_specialization_does_not_block_another_descriptor(
    valid_registered: bool,
) -> None:
    def incomplete(*_args: Any, **_kwargs: Any) -> tuple[tuple[str, object], ...]:
        raise AssemblyError("specific configuration requires its tokenizer")

    valid = replace(synthetic_descriptor(), checkpoint_loader="example.runtime:load")
    specialized = replace(
        valid,
        family=replace(valid.family, id="test.specific"),
        detector=incomplete,
    )
    registry = ComponentRegistry()
    registry.register(specialized)
    source = CheckpointHeader(("diffusion.weight",))
    if valid_registered:
        registry.register(valid)
        assert plan_component_checkpoint(source, component_registry=registry).descriptor is valid
    else:
        with pytest.raises(AssemblyError, match="specific configuration requires its tokenizer"):
            plan_component_checkpoint(source, component_registry=registry)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("partial_role", ["words", "pixels"])
@pytest.mark.parametrize("complete_alternative", [False, True])
def test_partial_matches_preserve_quantization_errors_without_blocking_complete_compositions(
    wrapped: bool, partial_role: str, complete_alternative: bool
) -> None:
    from dinkster_inference.quantization import QuantizationError, quantization_error_cause

    failure = QuantizationError("recognized diffusion is missing required weight_scale")

    def broken(*_args: Any, **_kwargs: Any) -> tuple[tuple[str, object], ...]:
        if wrapped:
            raise AssemblyError("invalid diffusion quantization") from failure
        raise failure

    valid = replace(synthetic_descriptor(), checkpoint_loader="example.runtime:load")
    registry = ComponentRegistry()
    registry.register(
        replace(valid, family=replace(valid.family, id="test.broken"), detector=broken)
    )
    registry.register(valid)
    keys = (f"{partial_role}.weight",)
    if complete_alternative:
        keys += ("diffusion.weight",)
    source = CheckpointHeader(keys)
    if complete_alternative:
        assert plan_component_checkpoint(source, component_registry=registry).descriptor is valid
    else:
        with pytest.raises(AssemblyError, match=f"test.synthetic/{partial_role}") as caught:
            plan_component_checkpoint(source, component_registry=registry)
        assert quantization_error_cause(caught.value) is failure
        assert "test.broken" in str(caught.value)
        assert str(failure) in str(caught.value)


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("variant", ["base", "edit", "layered"])
def test_qwen_descriptor_preserves_original_component_plans(combined: bool, variant: str) -> None:
    from dinkster_inference import (
        FLUX_DIFFUSION_PREFIX,
        FLUX_VAE_PREFIX,
        QWEN_IMAGE_CONFIG,
        QWEN_IMAGE_EDIT_2511_CONFIG,
        QWEN_IMAGE_LAYERED_CONFIG,
        QWEN_IMAGE_TEXT_PREFIX,
        plan_qwen_image_assembly,
    )
    from test_inference_assembly import (
        geometrize,
        prefixed,
        qwen_image_dit_layout,
        qwen_image_text_layout,
        source,
        wan21_vae_geometries,
    )

    config = {
        "base": QWEN_IMAGE_CONFIG,
        "edit": QWEN_IMAGE_EDIT_2511_CONFIG,
        "layered": QWEN_IMAGE_LAYERED_CONFIG,
    }[variant]
    geometries = {
        "diffusion": geometrize(qwen_image_dit_layout(config).keys, BFLOAT16),
        "qwen2_5_vl_7b": geometrize(qwen_image_text_layout(), BFLOAT16),
        "vae": wan21_vae_geometries(),
    }
    sources = {role: source(weights, f"{role}.safetensors") for role, weights in geometries.items()}
    if combined:
        sources = {
            "checkpoint": source(
                {
                    **prefixed(geometries["diffusion"], FLUX_DIFFUSION_PREFIX),
                    **prefixed(geometries["qwen2_5_vl_7b"], QWEN_IMAGE_TEXT_PREFIX),
                    **prefixed(geometries["vae"], FLUX_VAE_PREFIX),
                }
            )
        }
    expected = plan_qwen_image_assembly(**sources)
    actual = (
        plan_component_checkpoint(sources["checkpoint"])
        if combined
        else plan_component_checkpoint(component_sources=sources)
    )
    assert actual.family is expected.family
    assert actual.identity_components == expected.identity_components


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("variant", ["dev", "dev-full", "klein-4b", "klein-9b"])
def test_flux2_descriptor_preserves_original_component_plans(combined: bool, variant: str) -> None:
    from dinkster_inference import (
        FLUX2_DEV_CONFIG,
        FLUX2_KLEIN_4B_CONFIG,
        FLUX2_KLEIN_9B_CONFIG,
        FLUX_DIFFUSION_PREFIX,
        FLUX_VAE_PREFIX,
        KLEIN_QWEN3_4B_CONFIG,
        KLEIN_QWEN3_8B_CONFIG,
        MISTRAL3_24B_CONFIG,
        MISTRAL3_24B_PRUNED_CONFIG,
        plan_flux2_assembly,
    )
    from test_inference_assembly import (
        flux2_layout,
        flux2_vae_geometries,
        geometrize,
        prefixed,
        qwen_text_layout,
        source,
    )

    diffusion, text, role = {
        "dev": (FLUX2_DEV_CONFIG, MISTRAL3_24B_PRUNED_CONFIG, "mistral3_24b"),
        "dev-full": (FLUX2_DEV_CONFIG, MISTRAL3_24B_CONFIG, "mistral3_24b"),
        "klein-4b": (FLUX2_KLEIN_4B_CONFIG, KLEIN_QWEN3_4B_CONFIG, "qwen3_4b"),
        "klein-9b": (FLUX2_KLEIN_9B_CONFIG, KLEIN_QWEN3_8B_CONFIG, "qwen3_8b"),
    }[variant]
    sources = {
        "diffusion": source(geometrize(flux2_layout(diffusion)), "diffusion.safetensors"),
        role: source(prefixed(geometrize(qwen_text_layout(text)), "model."), "text.safetensors"),
        "vae": source(flux2_vae_geometries(), "vae.safetensors"),
    }
    if combined:
        checkpoint = source(
            {
                **prefixed(geometrize(flux2_layout(diffusion)), FLUX_DIFFUSION_PREFIX),
                **prefixed(
                    geometrize(qwen_text_layout(text)), f"text_encoders.{role}.transformer.model."
                ),
                **prefixed(flux2_vae_geometries(), FLUX_VAE_PREFIX),
            }
        )
        expected = plan_flux2_assembly(checkpoint=checkpoint)
        actual = plan_component_checkpoint(checkpoint)
    else:
        expected = plan_flux2_assembly(
            diffusion=sources["diffusion"], text_encoder=sources[role], vae=sources["vae"]
        )
        actual = plan_component_checkpoint(component_sources=sources)
    assert actual.family is expected.family
    assert actual.identity_components == expected.identity_components


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize(
    "profile",
    (
        "t2v-1.3b",
        "causal-ar-1.3b",
        "flow-rvs-1.3b",
        "t2v-14b",
        "humo-17b",
        "i2v-14b",
        "scail-14b",
        "scail2-14b",
        "animate2-14b-2.1",
        "animate-14b-2.2",
        "bernini-14b-2.2",
        "s2v-14b-2.2",
        "wandancer-14b-2.2",
        "flf-i2v-14b",
        "fun-control-1.3b",
        "fun-inpaint-1.3b",
        "i2v-14b-2.2",
        "fun-control-14b-2.2",
        "vace-1.3b",
        "vace-14b",
        "camera-1.3b",
        "camera-14b",
        "camera-14b-2.2",
    ),
)
def test_wan_descriptor_preserves_profiles_and_tokenizer(combined: bool, profile: str) -> None:
    from dinkster_inference import (
        WAN21_DIFFUSION_PREFIX,
        WAN21_UMT5_PREFIX,
        WAN21_VAE_PREFIX,
        plan_wan21_assembly,
    )
    from dinkster_inference.wan21_component import wan_checkpoint_assembly
    from test_inference_assembly import FakeSource, source, wan21_split_sources

    sources = wan21_split_sources(profile)
    if combined:
        prefixes = {
            "diffusion": WAN21_DIFFUSION_PREFIX,
            "umt5xxl": WAN21_UMT5_PREFIX,
            "vae": WAN21_VAE_PREFIX,
        }
        geometries = {
            (
                "text_encoders.umt5xxl.spiece_model"
                if key == "spiece_model"
                else prefixes[role] + key
            ): weights.entry(key).geometry
            for role, weights in sources.items()
            if role in prefixes
            for key in weights.keys()
        }
        checkpoint = source(geometries, "combined-wan.safetensors")
        checkpoint.extra.update(cast("FakeSource", sources["diffusion"]).extra)
        sources = {
            "checkpoint": checkpoint,
            **{role: weights for role, weights in sources.items() if role not in prefixes},
        }
    expected = plan_wan21_assembly(**sources)
    actual = plan_component_checkpoint(component_sources=sources)
    assert actual.family is expected.family
    assert actual.identity_components == expected.identity_components
    assert wan_checkpoint_assembly(actual) == expected
    assert actual.unclaimed == ()


def test_wan_descriptor_keeps_gguf_payload_and_vendored_tokenizer() -> None:
    from dinkster_inference import UMT5_XXL_CONFIG, plan_wan21_assembly
    from dinkster_inference.wan21_component import wan_checkpoint_assembly
    from test_inference_assembly import gguf_text_source, wan21_split_sources

    sources = wan21_split_sources()
    sources["umt5xxl"] = gguf_text_source(UMT5_XXL_CONFIG, "umt5.gguf")
    expected = plan_wan21_assembly(**sources)
    actual = plan_component_checkpoint(component_sources=sources)
    assert actual.identity_components == expected.identity_components
    assert actual.components["umt5xxl"].payload_source is sources["umt5xxl"]
    assert wan_checkpoint_assembly(actual) == expected
    assert expected.tokenizer_vendored and expected.tokenizer_source_key == ""


def test_wan_descriptor_claims_authenticated_legacy_text_quantization_marker() -> None:
    from dinkster_inference import FLOAT8_E4M3, FLOAT32, default_text_dtype, probe_native
    from dinkster_inference.wan21_component import WanCheckpointText
    from test_inference_assembly import FakeSource, quantize_legacy, source, wan21_split_sources

    sources = wan21_split_sources()
    text = cast("FakeSource", sources["umt5xxl"])
    layer = "encoder.block.0.layer.0.SelfAttention.q"
    sources["umt5xxl"] = source(quantize_legacy(dict(text.geometries), layer), text.path.name)

    plan = plan_component_checkpoint(component_sources=sources)
    planned_text = dict(plan.role_plans)["umt5xxl"]
    assert isinstance(planned_text, WanCheckpointText)
    assert planned_text.legacy_quantization_marker == "scaled_fp8"
    assert plan.unclaimed == ()
    assert plan.components["umt5xxl"].dtypes[layer + ".weight"] is FLOAT8_E4M3
    assert plan.components["umt5xxl"].quant[layer].format == "float8_e4m3fn"
    assert default_text_dtype("dinkster.wan21") is FLOAT32
    assert probe_native(
        diffusion=sources["diffusion"],
        t5xxl=sources["umt5xxl"],
        vae=sources["vae"],
    ).native


def test_wan_descriptor_claims_prefixed_legacy_text_quantization_marker() -> None:
    from dinkster_inference import WAN21_UMT5_PREFIX
    from dinkster_inference.wan21_component import WanCheckpointText
    from test_inference_assembly import quantize_legacy, source, wan21_combined_geometries

    layer = "encoder.block.0.layer.0.SelfAttention.q"
    checkpoint = source(
        quantize_legacy(wan21_combined_geometries(), layer, WAN21_UMT5_PREFIX),
        "wan-scaled-fp8.safetensors",
    )

    plan = plan_component_checkpoint(checkpoint)
    planned_text = dict(plan.role_plans)["umt5xxl"]
    assert isinstance(planned_text, WanCheckpointText)
    assert planned_text.legacy_quantization_marker == WAN21_UMT5_PREFIX + "scaled_fp8"
    assert plan.unclaimed == ()


def test_wan_descriptor_does_not_claim_an_unparsed_legacy_marker() -> None:
    from dinkster_inference import FLOAT32, AssemblyError
    from test_inference_assembly import FakeSource, g, source, wan21_split_sources

    sources = wan21_split_sources()
    text = cast("FakeSource", sources["umt5xxl"])
    geometries = dict(text.geometries)
    geometries["scaled_fp8"] = g((0,), FLOAT32)
    sources["umt5xxl"] = source(geometries, text.path.name)

    with pytest.raises(AssemblyError, match="scaled_fp8"):
        plan_component_checkpoint(component_sources=sources)


def test_wan_descriptor_rejects_bare_and_prefixed_legacy_markers() -> None:
    from dinkster_inference import FLOAT32, WAN21_UMT5_PREFIX, AssemblyError
    from test_inference_assembly import g, quantize_legacy, source, wan21_combined_geometries

    layer = "encoder.block.0.layer.0.SelfAttention.q"
    geometries = quantize_legacy(wan21_combined_geometries(), layer, WAN21_UMT5_PREFIX)
    geometries["scaled_fp8"] = g((0,), FLOAT32)
    checkpoint = source(geometries, "wan-ambiguous-scaled-fp8.safetensors")

    with pytest.raises(AssemblyError, match="scaled_fp8"):
        plan_component_checkpoint(checkpoint)


def test_wan_descriptor_selects_no_marker_when_bare_and_prefixed_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference import FLOAT32, WAN21_UMT5_PREFIX, wan21_component
    from dinkster_inference.assembly import plan_wan_text_component
    from dinkster_inference.wan21_component import WanCheckpointText
    from test_inference_assembly import FakeSource, g, quantize_legacy, source, wan21_split_sources

    sources = wan21_split_sources()
    text = cast("FakeSource", sources["umt5xxl"])
    layer = "encoder.block.0.layer.0.SelfAttention.q"
    geometries = quantize_legacy(dict(text.geometries), layer)
    component = plan_wan_text_component(umt5xxl=source(geometries, text.path.name))
    geometries[WAN21_UMT5_PREFIX + "scaled_fp8"] = g((0,), FLOAT32)
    ambiguous = source(geometries, text.path.name)
    monkeypatch.setattr(wan21_component, "plan_wan_text_component", lambda **_kwargs: component)

    planned = wan21_component._checkpoint_component(ambiguous, "umt5xxl")
    assert isinstance(planned, WanCheckpointText)
    assert planned.legacy_quantization_marker == ""


def test_checkpoint_rejects_auxiliary_claim_collisions() -> None:
    from dinkster_inference import UMT5_XXL_CONFIG
    from dinkster_inference.wan21_component import WanCheckpointText

    diffusion = part("diffusion")
    descriptor = replace(
        synthetic_descriptor(),
        roles=("diffusion", "umt5xxl"),
        text_encoder_roles=("umt5xxl",),
        codec_roles=(),
    )
    text_plan = ComponentPlan(
        "umt5xxl",
        diffusion.path,
        UMT5_XXL_CONFIG,
        {"weight": "umt5xxl.weight"},
        {"weight": BFLOAT16},
        {},
    )
    text = WanCheckpointText(text_plan, "diffusion.layer.weight", False)
    with pytest.raises(AssemblyError, match="claimed by both 'diffusion' and 'umt5xxl'"):
        ComponentCheckpointPlan(descriptor, (("diffusion", diffusion), ("umt5xxl", text)))


@pytest.mark.parametrize("empty_keys", [False, True])
def test_checkpoint_rejects_detector_source_path_changes(empty_keys: bool) -> None:
    component = part("diffusion", path=Path("other.safetensors"))
    if empty_keys:
        component = replace(component, keys={}, dtypes={})
    descriptor = replace(
        synthetic_descriptor(),
        checkpoint_loader="example:load",
        detector=lambda *_args, **_kwargs: (("diffusion", component),),
    )
    registry = ComponentRegistry()
    registry.register(descriptor)
    with pytest.raises(AssemblyError, match="plan changed its selected source path"):
        plan_component_checkpoint(
            CheckpointHeader(("diffusion.weight",)), component_registry=registry
        )
