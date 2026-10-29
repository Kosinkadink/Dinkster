"""Recipe semantics and ordered sources are independent of component detection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Never

import pytest
from dinkster_inference import FLOAT16, FLUX_DEV, ComponentPlan
from dinkster_inference.clip_text import (
    CLIP_G_TEXT_CONFIG,
    CLIP_L_TEXT_CONFIG,
    clip_text_layout,
)
from dinkster_inference.component_catalog import default_component_registry
from dinkster_inference.component_registry import ComponentDescriptor, DetectedComponents
from dinkster_inference.identity import build_runtime_identity_from_facts
from dinkster_inference.openclip_text import openclip_text_layout
from dinkster_inference.quantization import QuantizationError
from dinkster_inference.recipe import WeightSourceRef
from dinkster_inference.t5_text import T5_XXL_CONFIG, t5_layout
from dinkster_inference.text_components import plan_text_components
from dinkster_inference.text_recipes import (
    TextRecipeDescriptor,
    UnresolvedTextRecipe,
    default_text_recipe_registry,
    resolve_text_recipe,
)
from dinkster_inference.weights import TensorGeometry, WeightEntry
from dinkster_protocol import AttentionPolicy, AttentionRoute, AttentionRouteToken
from dinkster_protocol.attention import ATTENTION_ROLES


def detected(*roles: str) -> tuple[DetectedComponents, ...]:
    descriptor = ComponentDescriptor(
        family=replace(FLUX_DEV, id="example.text_architecture"),
        detector=lambda source, path: (),
        roles=roles,
        text_encoder_roles=roles,
        codec_roles=(),
        loader="example.module:load",
        runtime_class="example.module:Runtime",
    )
    return (
        DetectedComponents(
            descriptor,
            tuple(
                (
                    role,
                    ComponentPlan(
                        component=role,
                        path=Path(f"/source-machine/{role}.safetensors"),
                        config=role,
                        keys={"weight": f"physical.{role}.weight"},
                        dtypes={"weight": FLOAT16},
                        quant={},
                    ),
                )
                for role in roles
            ),
        ),
    )


def sources() -> tuple[WeightSourceRef, ...]:
    return tuple(
        WeightSourceRef("blake3:" + digit * 64, f"encoder-{digit}.safetensors", 8)
        for digit in ("1", "2")
    )


def test_sdxl_binding_preserves_plans_profiles_and_both_ordered_sources() -> None:
    matches = (detected("clip_g"), detected("clip_l"))
    binding = resolve_text_recipe(matches, "sdxl")
    assert [(part.source_index, part.role) for part in binding.components] == [
        (1, "clip_l"),
        (0, "clip_g"),
    ]
    assert binding.components[0].plan is matches[1][0].components[0][1]
    assert binding.components[0].plan.keys == {"weight": "physical.clip_l.weight"}
    for part in binding.components:
        assert part.profile is not None
        assert part.profile.hidden_layer == -2
        assert not part.profile.layer_norm_hidden_state
    assert binding.components[0].profile is not None
    assert binding.components[1].profile is not None
    assert binding.components[0].profile.tokenizer.pad_token == 49407
    assert binding.components[1].profile.tokenizer.pad_token == 0
    refs = sources()
    recipe = binding.recipe(refs, "float32")
    assert tuple(source.source for source in recipe.sources) == refs
    assert tuple(source.role for source in recipe.sources) == (
        "text_source_0000",
        "text_source_0001",
    )
    assert "/source-machine" not in repr(recipe)
    assert recipe.knobs.text_dtype == "float32"
    assert "text_recipe=dinkster.text_sdxl" in recipe.knobs.runtime_facts


@pytest.mark.parametrize("kind", ("sdxl", "flux"))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("configured", (False, True))
def test_host_identity_without_source_refs_matches_worker_recipe(
    kind: str, reverse: bool, configured: bool
) -> None:
    matches = (detected("clip_l"), detected("clip_g" if kind == "sdxl" else "t5xxl"))
    if reverse:
        matches = tuple(reversed(matches))
    binding = resolve_text_recipe(matches, kind)
    dtype = "bfloat16" if configured else "float32"
    policy: AttentionPolicy = "sdpa" if configured else "auto"
    token = (
        AttentionRouteToken(
            version=1,
            routes=tuple(AttentionRoute(role, "sdpa") for role in ATTENTION_ROLES),
            provider_versions=(),
            adapter_contract_revision="test-v1",
            device_kind="cpu",
            device_sm=None,
            sdpa_torch_runtime="test-torch",
            requested_policy=policy,
        )
        if configured
        else None
    )
    embedding_digest = "a" * 64 if configured else None
    knobs = binding.knobs(
        dtype,
        attention_policy=policy,
        attention_route_token=token,
        embedding_binding_digest=embedding_digest,
    )
    host_identity = build_runtime_identity_from_facts(
        binding.family_id,
        binding.component_identity,
        diffusion_dtype=knobs.diffusion_dtype,
        text_dtype=knobs.text_dtype,
        vae_dtype=knobs.vae_dtype,
        fp8_matmul=knobs.fp8_matmul,
        attention_policy=knobs.attention_policy,
        attention_route_token=knobs.attention_route_token,
        embedding_binding_digest=knobs.embedding_binding_digest,
        runtime_facts=knobs.runtime_facts,
    )
    refs = tuple(reversed(sources())) if reverse else sources()
    recipe = binding.recipe(
        refs,
        dtype,
        attention_policy=policy,
        attention_route_token=token,
        embedding_binding_digest=embedding_digest,
    )
    assert recipe.component_identity == binding.component_identity
    assert recipe.knobs == knobs
    assert recipe.runtime_identity == host_identity
    assert tuple(source.source for source in recipe.sources) == refs


def test_same_detected_bytes_have_type_selected_profiles_and_identity() -> None:
    matches = (detected("clip_l"), detected("clip_g", "t5xxl"))
    sdxl = resolve_text_recipe(matches, "sdxl")
    flux = resolve_text_recipe(matches, "flux")
    assert sdxl.components[0].plan is flux.components[0].plan
    assert sdxl.components[0].profile is not None
    assert flux.components[0].profile is not None
    assert flux.components[1].profile is not None
    assert sdxl.components[0].profile.hidden_layer == -2
    assert flux.components[0].profile.hidden_layer is None
    assert not flux.components[0].profile.projected_pooled
    assert flux.components[1].profile.tokenizer.min_length == 256
    refs = sources()
    assert sdxl.recipe(refs, "float32").sources == flux.recipe(refs, "float32").sources
    assert (
        sdxl.recipe(refs, "float32").runtime_identity
        != flux.recipe(refs, "float32").runtime_identity
    )


def test_reversing_sources_retains_the_role_assignment_and_recipe_order() -> None:
    matches = (detected("clip_l"), detected("t5xxl"))
    forward = resolve_text_recipe(matches, "flux").recipe(sources(), "float32")
    reverse = resolve_text_recipe(tuple(reversed(matches)), "flux").recipe(
        tuple(reversed(sources())), "float32"
    )
    assert forward.component_identity == reverse.component_identity
    assert forward.sources[0].source == reverse.sources[1].source
    assert forward.knobs.runtime_facts != reverse.knobs.runtime_facts


@pytest.mark.parametrize(
    "matches,kind,message",
    (
        ((detected("clip_l"),), "not_a_recipe", "unknown text recipe"),
        ((detected("clip_l"),), "flux", "one unambiguous t5xxl"),
        ((detected("clip_l"), detected("clip_l")), "flux", "one unambiguous clip_l"),
        (
            (detected("clip_l", "t5xxl"), detected("clip_g")),
            "flux",
            "discard an input source",
        ),
    ),
)
def test_unknown_missing_ambiguous_and_unused_inputs_keep_detection_evidence(
    matches: tuple[tuple[DetectedComponents, ...], ...], kind: str, message: str
) -> None:
    with pytest.raises(UnresolvedTextRecipe, match=message) as error:
        resolve_text_recipe(matches, kind)
    assert "example.text_architecture/clip_l" in str(error.value)


@pytest.mark.parametrize("kind", ("stable_diffusion", "sd3", "mochi", "cogvideox", "pixart"))
def test_unimplemented_shared_t5_recipes_leave_provider_fallback_available(kind: str) -> None:
    with pytest.raises(UnresolvedTextRecipe):
        resolve_text_recipe((detected("t5xxl"),), kind)


def test_recipe_integrity_failures_are_not_resolution_fallbacks() -> None:
    registry = default_text_recipe_registry()
    failure = ValueError("invalid quantization metadata")

    def invalid_recipe(matches: tuple[tuple[DetectedComponents, ...], ...]) -> Never:
        raise failure

    registry.register(TextRecipeDescriptor("example.invalid", invalid_recipe))
    with pytest.raises(ValueError) as error:
        resolve_text_recipe((detected("clip_l"),), "example.invalid", registry=registry)
    assert error.value is failure
    assert not isinstance(error.value, UnresolvedTextRecipe)


def test_textual_inversion_snapshot_is_part_of_recipe_identity() -> None:
    binding = resolve_text_recipe((detected("clip_l"), detected("t5xxl")), "flux")
    without_embeddings = binding.recipe(sources(), "float32")
    first = binding.recipe(sources(), "float32", embedding_binding_digest="a" * 64)
    second = binding.recipe(sources(), "float32", embedding_binding_digest="b" * 64)
    assert first.knobs.embedding_binding_digest == "a" * 64
    assert second.knobs.embedding_binding_digest == "b" * 64
    assert first.sources == second.sources == without_embeddings.sources
    assert (
        len({first.runtime_identity, second.runtime_identity, without_embeddings.runtime_identity})
        == 3
    )
    with pytest.raises(ValueError, match="embedding_binding_digest"):
        binding.recipe(sources(), "float32", embedding_binding_digest="invalid")


def test_profile_changes_are_identity_visible_and_sources_cannot_be_dropped() -> None:
    binding = resolve_text_recipe((detected("clip_l"), detected("t5xxl")), "flux")
    component = binding.components[0]
    assert component.profile is not None
    changed = replace(
        binding,
        components=(
            replace(component, profile=replace(component.profile, hidden_layer=-2)),
            binding.components[1],
        ),
    )
    assert (
        binding.recipe(sources(), "float32").runtime_identity
        != changed.recipe(sources(), "float32").runtime_identity
    )
    with pytest.raises(ValueError, match="retain every source"):
        binding.recipe(sources()[:1], "float32")


def test_registered_recipe_needs_no_architecture_or_family_admission_edit() -> None:
    matches = (detected("clip_l"), detected("t5xxl"))
    binding = resolve_text_recipe(matches, "flux")
    registry = default_text_recipe_registry()
    registry.register(
        TextRecipeDescriptor(
            "example.text_recipe",
            lambda detected_sources: replace(binding, id="example.text_recipe"),
            ("custom_text",),
        )
    )
    custom = resolve_text_recipe(matches, "custom_text", registry=registry)
    assert custom.components == binding.components
    assert (
        custom.recipe(sources(), "float32").runtime_identity
        != binding.recipe(sources(), "float32").runtime_identity
    )


@dataclass
class Header:
    layout: dict[str, tuple[int, ...]]
    path: Path = Path("/source-machine/text.safetensors")
    asset_digest: str = "blake3:" + "3" * 64
    asset_size: int = 1234

    def keys(self) -> tuple[str, ...]:
        return tuple(self.layout)

    def entry(self, key: str) -> WeightEntry:
        geometry = TensorGeometry(self.layout[key], FLOAT16)
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> dict[str, str]:
        return {}


@pytest.mark.parametrize(
    "layout,role",
    (
        (clip_text_layout(CLIP_L_TEXT_CONFIG), "clip_l"),
        (clip_text_layout(CLIP_G_TEXT_CONFIG), "clip_g"),
        (openclip_text_layout(CLIP_G_TEXT_CONFIG), "clip_g"),
        (
            {
                key.removeprefix("text_model."): value
                for key, value in clip_text_layout(CLIP_L_TEXT_CONFIG).items()
                if key != "text_projection.weight"
            },
            "clip_l",
        ),
        (t5_layout(T5_XXL_CONFIG), "t5xxl"),
    ),
)
@pytest.mark.parametrize("bind_asset_identity", (False, True))
def test_standalone_architecture_uses_existing_layout_planners(
    layout: dict[str, tuple[int, ...]], role: str, bind_asset_identity: bool
) -> None:
    source = Header(layout)
    options = {} if bind_asset_identity else {"bind_asset_identity": False}
    ((detected_role, plan),) = plan_text_components(source, source.path, **options)
    assert detected_role == plan.component == role
    assert plan.path == source.path
    assert set(plan.keys.values()) <= set(layout)
    assert set(plan.dtypes.values()) == {FLOAT16}
    assert (f"asset_digest={source.asset_digest}" in plan.identity_facts) is bind_asset_identity
    assert (f"asset_size={source.asset_size}" in plan.identity_facts) is bind_asset_identity
    registry = default_component_registry()
    descriptor = registry.get("dinkster.classic_text")
    assert descriptor is not None and descriptor.requires_text_recipe
    assert descriptor.detector is plan_text_components
    assert descriptor.model_role not in descriptor.roles
    matches = registry.detect(source, source.path, bind_asset_identity=bind_asset_identity)
    assert next(match for match in matches if match.descriptor is descriptor).components == (
        (role, plan),
    )
    recipe_descriptors = {item.id for item in registry if item.requires_text_recipe}
    assert recipe_descriptors == {
        "dinkster.ace_step_1_5",
        "dinkster.classic_text",
        "dinkster.hunyuan_image",
        "dinkster.hunyuan_video",
        "dinkster.newbie",
    }
    if not bind_asset_identity:
        from test_inference_component_checkpoint import GeometryHeader

        unbound = GeometryHeader(
            source.path, {key: source.entry(key).geometry for key in source.keys()}
        )
        assert plan_text_components(unbound, unbound.path, **options) == ((role, plan),)


@pytest.mark.parametrize("kind", ("sdxl", "flux"))
@pytest.mark.parametrize("reverse", (False, True))
def test_full_component_registry_resolves_classic_recipes_in_both_source_orders(
    kind: str, reverse: bool
) -> None:
    first = Header(clip_text_layout(CLIP_L_TEXT_CONFIG))
    second = Header(
        clip_text_layout(CLIP_G_TEXT_CONFIG) if kind == "sdxl" else t5_layout(T5_XXL_CONFIG)
    )
    headers = (second, first) if reverse else (first, second)
    registry = default_component_registry()
    matches = tuple(registry.detect(source, source.path) for source in headers)
    binding = resolve_text_recipe(matches, kind)
    assert tuple(component.role for component in binding.components) == (
        "clip_l",
        "clip_g" if kind == "sdxl" else "t5xxl",
    )
    assert tuple(component.source_index for component in binding.components) == (
        (1, 0) if reverse else (0, 1)
    )
    for component in binding.components:
        architecture = next(
            match
            for match in matches[component.source_index]
            if match.descriptor.requires_text_recipe
        )
        assert component.plan is architecture.plan_for(component.role)


def test_combined_text_architecture_retains_both_physical_prefixes_and_transforms() -> None:
    clip_l = clip_text_layout(CLIP_L_TEXT_CONFIG)
    clip_g = openclip_text_layout(CLIP_G_TEXT_CONFIG)
    source = Header(
        {
            **{f"conditioner.embedders.0.transformer.{k}": v for k, v in clip_l.items()},
            **{f"conditioner.embedders.1.model.{k}": v for k, v in clip_g.items()},
        }
    )
    plans = dict(plan_text_components(source, source.path))
    assert set(plans) == {"clip_l", "clip_g"}
    assert all(key.startswith("conditioner.embedders.0.") for key in plans["clip_l"].keys.values())
    assert all(key.startswith("conditioner.embedders.1.") for key in plans["clip_g"].keys.values())
    assert plans["clip_g"].transforms
    assert not plans["clip_l"].transforms


def test_incomplete_architecture_does_not_claim_a_component() -> None:
    source = Header({"text_model.embeddings.token_embedding.weight": (49408, 768)})
    assert plan_text_components(source, source.path) == ()


def test_text_component_preserves_nested_quantization_failure() -> None:
    class BadMetadata(Header):
        def metadata(self) -> dict[str, str]:
            return {"_quantization_metadata": "not json"}

    source = BadMetadata(clip_text_layout(CLIP_L_TEXT_CONFIG))
    with pytest.raises(QuantizationError, match="malformed _quantization_metadata"):
        plan_text_components(source, source.path)


@pytest.mark.parametrize("valid_alternative", (False, True))
def test_quantized_openclip_is_not_an_unrecognized_text_recipe(valid_alternative: bool) -> None:
    from tests.test_inference_assembly import geometrize, prefixed, quantize_legacy, source

    geometry = geometrize(openclip_text_layout(CLIP_G_TEXT_CONFIG), FLOAT16)
    quantize_legacy(geometry, "transformer.resblocks.0.mlp.c_fc")
    if valid_alternative:
        geometry.update(prefixed(geometrize(clip_text_layout(CLIP_L_TEXT_CONFIG)), "clip_l."))
    header = source(geometry)
    if valid_alternative:
        ((role, plan),) = plan_text_components(header, header.path)
        assert role == "clip_l"
        assert all(key.startswith("clip_l.") for key in plan.keys.values())
    else:
        with pytest.raises(QuantizationError, match="quantized OpenCLIP"):
            plan_text_components(header, header.path)
        with pytest.raises(QuantizationError, match="quantized OpenCLIP"):
            default_component_registry().detect(header, header.path)


@pytest.mark.parametrize("prefix", ("", "clip_l."))
def test_text_component_allows_valid_prefix_alternative_to_quantization_failure(
    prefix: str,
) -> None:
    source = Header(
        {
            "unrelated.weight_scale": (),
            **{prefix + key: shape for key, shape in clip_text_layout(CLIP_L_TEXT_CONFIG).items()},
        }
    )
    if not prefix:
        with pytest.raises(QuantizationError, match="weight_scale"):
            plan_text_components(source, source.path)
    else:
        ((role, plan),) = plan_text_components(source, source.path)
        assert role == "clip_l"
        assert all(key.startswith(prefix) for key in plan.keys.values())


@pytest.mark.parametrize(
    "roles,recipe_id,layer",
    (
        (("clip_l",), "dinkster.text_sd15", None),
        (("clip_g",), "dinkster.text_sdxl_refiner", -2),
        (("clip_l", "clip_g"), "dinkster.text_sdxl", -2),
    ),
)
def test_stable_diffusion_type_uses_detected_text_roles_and_roundtrips_canonical_recipe(
    roles: tuple[str, ...], recipe_id: str, layer: int | None
) -> None:
    matches = (detected(*roles),)
    binding = resolve_text_recipe(matches, "stable_diffusion")
    assert binding.id == recipe_id
    for part in binding.components:
        assert part.profile is not None
        assert part.profile.hidden_layer == layer
    assert resolve_text_recipe(matches, binding.id) == binding


def test_composition_cannot_omit_a_component_or_consume_one_twice() -> None:
    binding = resolve_text_recipe((detected("clip_l"), detected("t5xxl")), "flux")
    with pytest.raises(ValueError, match="consume every component"):
        replace(binding, composition_roles=("clip_l", "clip_l"))
    with pytest.raises(ValueError, match="require a composer"):
        replace(binding, composer=None)


@pytest.mark.parametrize("absent_roles", (("clip_l",), ("t5xxl",), ("clip_l", "t5xxl")))
def test_absent_generic_packing_profiles_preserve_ordered_sources_and_identity(
    absent_roles: tuple[str, ...],
) -> None:
    binding = resolve_text_recipe((detected("t5xxl"), detected("clip_l")), "flux")
    changed = replace(
        binding,
        components=tuple(
            replace(part, profile=None) if part.role in absent_roles else part
            for part in binding.components
        ),
    )
    before = binding.recipe(sources(), "float32")
    after = changed.recipe(sources(), "float32")
    assert before.sources == after.sources
    assert before.component_identity == after.component_identity
    assert before.runtime_identity != after.runtime_identity
    assert changed.recipe(sources(), "float32") == after
    profiles = {}
    for part in binding.components:
        assert part.profile is not None
        profiles[part.role] = None if part.role in absent_roles else asdict(part.profile)
    facts = [
        json.loads(fact.removeprefix("text_component="))
        for fact in after.knobs.runtime_facts
        if fact.startswith("text_component=")
    ]
    assert facts == [[1, "clip_l", profiles["clip_l"]], [0, "t5xxl", profiles["t5xxl"]]]
    for role in absent_roles:
        source_index = 1 if role == "clip_l" else 0
        assert f'text_component=[{source_index},"{role}",null]' in after.knobs.runtime_facts


@pytest.mark.parametrize(
    "kind,roles",
    (
        ("sdxl", ("clip_l", "clip_g")),
        ("flux", ("clip_l", "t5xxl")),
        ("stable_diffusion", ("clip_l",)),
        ("dinkster.text_sdxl_refiner", ("clip_g",)),
    ),
)
def test_non_null_profile_facts_and_identity_match_original_serialization(
    kind: str, roles: tuple[str, ...]
) -> None:
    binding = resolve_text_recipe(tuple(detected(role) for role in roles), kind)
    original_facts = [
        f"text_recipe={binding.id}",
        f"text_composer={binding.composer}",
        "text_composition_roles=" + json.dumps(binding.composition_roles),
    ]
    for part in binding.components:
        assert part.profile is not None
        original_facts.append(
            "text_component="
            + json.dumps(
                (part.source_index, part.role, asdict(part.profile)),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        original_facts.extend(part.plan.runtime_facts)
    recipe = binding.recipe(sources()[: len(roles)], "float32")
    original_knobs = replace(recipe.knobs, runtime_facts=tuple(original_facts))
    assert recipe.knobs.runtime_facts == tuple(original_facts)
    assert recipe.runtime_identity == replace(recipe, knobs=original_knobs).runtime_identity
