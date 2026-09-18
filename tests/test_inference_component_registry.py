"""Component registration, header selection, and reconstruction identity contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from dinkster_inference import BFLOAT16, LUMINA2, ComponentPlan, TensorGeometry, WeightEntry
from dinkster_inference.component_catalog import default_component_registry
from dinkster_inference.component_registry import ComponentDescriptor, ComponentRegistry
from dinkster_inference.recipe import WeightSourceRef
from dinkster_inference.registry import RegistryError
from dinkster_inference.weights import WeightSource


@dataclass(frozen=True)
class Header:
    names: tuple[str, ...]
    path: Path = Path("combined.safetensors")

    def keys(self) -> tuple[str, ...]:
        return self.names

    def entry(self, key: str) -> WeightEntry:
        if key not in self.names:
            raise KeyError(key)
        return WeightEntry(key, TensorGeometry((2, 2), BFLOAT16), 0, 8)

    def metadata(self) -> dict[str, str]:
        return {}


def synthetic_descriptor() -> ComponentDescriptor:
    def detect(
        source: WeightSource, path: Path, *, bind_asset_identity: bool = True
    ) -> tuple[tuple[str, ComponentPlan[str]], ...]:
        del bind_asset_identity
        components: list[tuple[str, ComponentPlan[str]]] = []
        for role in ("diffusion", "words", "pixels"):
            source_key = f"{role}.weight"
            if source_key in source.keys():
                components.append(
                    (
                        role,
                        ComponentPlan(
                            role,
                            path,
                            "synthetic",
                            {"weight": source_key},
                            {"weight": source.entry(source_key).geometry.dtype},
                            {},
                            runtime_facts=("operator=synthetic",),
                        ),
                    )
                )
        return tuple(components)

    return ComponentDescriptor(
        replace(LUMINA2, id="test.synthetic", display_name="Synthetic"),
        detect,
        ("diffusion", "words", "pixels"),
        ("words",),
        ("pixels",),
        "test_component_runtime:load",
        "test_component_runtime:Runtime",
    )


@pytest.mark.parametrize(
    "kind,role", [("model", "diffusion"), ("text", "words"), ("codec", "pixels")]
)
@pytest.mark.parametrize("requires_text_recipe", (False, True))
def test_new_descriptor_selects_split_and_combined_headers(
    kind: str, role: str, requires_text_recipe: bool
) -> None:
    registry = ComponentRegistry()
    descriptor = replace(synthetic_descriptor(), requires_text_recipe=requires_text_recipe)
    registry.register(descriptor)
    split = Header((f"{role}.weight",))
    combined = Header(("diffusion.weight", "words.weight", "pixels.weight"))
    path = Path("synthetic.safetensors")
    if kind == "text" and requires_text_recipe:
        for header in (split, combined):
            matches = registry.detect(header, path)
            assert matches[0].plan_for(role) is not None
            with pytest.raises(ValueError, match="explicit text recipe required"):
                registry.select_detected(matches, kind, family_id=descriptor.id)
        return
    selected, selected_role, split_plan = registry.select(split, path, kind)
    combined_selected, combined_role, combined_plan = registry.select(combined, path, kind)
    assert selected is descriptor and combined_selected is descriptor
    assert selected_role == combined_role == role
    assert split_plan == combined_plan
    assert split_plan.keys == {"weight": f"{role}.weight"}
    assert descriptor.sigma_space == descriptor.family.sampling


@pytest.mark.parametrize("role", ["diffusion", "words", "pixels"])
def test_recipe_matches_dispatch_identity_without_replanning(role: str) -> None:
    descriptor = synthetic_descriptor()
    plan = descriptor.detector(Header((f"{role}.weight",)), Path("weights.safetensors"))[0][1]
    loaded = SimpleNamespace(plan=plan, role=role)
    source = WeightSourceRef("blake3:" + "1" * 64, "weights.safetensors", 8)
    recipe = descriptor.recipe(source, loaded, "float32")
    assert recipe.runtime_identity == descriptor.component_identity(role, plan, "float32")
    assert recipe.knobs.diffusion_dtype == ("float32" if role == "diffusion" else "unloaded")
    assert recipe.knobs.text_dtype == ("float32" if role == "words" else "unloaded")
    assert recipe.knobs.vae_dtype == ("float32" if role == "pixels" else "unloaded")
    assert recipe.knobs.runtime_facts == ("operator=synthetic",)
    assert plan.dtypes == {"weight": BFLOAT16}


def test_generic_component_role_requires_descriptor_declaration() -> None:
    plan = object()

    def detect_controlnet(
        source: WeightSource, path: Path, *, bind_asset_identity: bool = True
    ) -> tuple[tuple[str, object], ...]:
        del source, path, bind_asset_identity
        return (("controlnet", plan),)

    descriptor = replace(synthetic_descriptor(), detector=detect_controlnet)
    registry = ComponentRegistry()
    registry.register(descriptor)
    source = Header(("controlnet.weight",))
    with pytest.raises(ValueError, match="no matching component architecture"):
        registry.select(source, source.path, "controlnet")

    declared = replace(descriptor, roles=(*descriptor.roles, "controlnet"))
    registry = ComponentRegistry()
    registry.register(declared)
    selected, role, selected_plan = registry.select(source, source.path, "controlnet")
    assert selected is declared
    assert role == "controlnet"
    assert selected_plan is plan


def test_unknown_and_wrong_role_diagnostics_do_not_assign_a_default_family(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = ComponentRegistry()
    registry.register(synthetic_descriptor())
    unknown = Header(("unfamiliar.weight",))
    assert registry.detect(unknown, Path("unknown.safetensors")) == ()
    assert "No matching component architecture" in caplog.text
    assert "detected 1 tensor keys (unfamiliar.weight)" in caplog.text
    with pytest.raises(ValueError, match=r"detected 1 tensor keys \(unfamiliar.weight\)"):
        registry.select(unknown, Path("unknown.safetensors"), "model")
    with pytest.raises(ValueError, match="detected Synthetic words"):
        registry.select(Header(("words.weight",)), Path("text.safetensors"), "model")


def test_ambiguous_geometry_requires_evidence_not_registration_order() -> None:
    descriptor = synthetic_descriptor()
    other = replace(descriptor, family=replace(descriptor.family, id="test.other"))
    header = Header(("diffusion.weight",))
    for order in ((descriptor, other), (other, descriptor)):
        registry = ComponentRegistry()
        for item in order:
            registry.register(item)
        with pytest.raises(ValueError, match="ambiguous components"):
            registry.select(header, Path("weights.safetensors"), "model")
        assert (
            registry.select(header, Path("weights.safetensors"), "model", family_id=descriptor.id)[
                0
            ]
            is descriptor
        )


def test_registration_collision_does_not_replace_existing_behavior() -> None:
    registry = ComponentRegistry()
    descriptor = synthetic_descriptor()
    registry.register(descriptor)
    with pytest.raises(RegistryError, match="already registered"):
        registry.register(descriptor)
    assert tuple(registry) == (descriptor,)


def test_assembly_discovery_uses_component_registration_without_family_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference import (
        builtin_assembly_registry,
        builtin_family_registry,
        component_catalog,
        resolve_native_assembly,
    )
    from dinkster_inference.component_checkpoint import ComponentCheckpointPlan

    descriptor = replace(
        synthetic_descriptor(), checkpoint_loader="test_component_runtime:load_checkpoint"
    )
    header = Header(("diffusion.weight", "words.weight", "pixels.weight"))
    registry = ComponentRegistry()
    registry.register(descriptor)
    monkeypatch.setattr(component_catalog, "default_component_registry", lambda: registry)
    assert builtin_family_registry().get(descriptor.id) is None
    assembly = builtin_assembly_registry().get("dinkster.components")
    assert assembly is not None and assembly.aliases == (descriptor.id,)
    resolution = resolve_native_assembly(checkpoint=header)
    assert resolution.registration == assembly
    assert isinstance(resolution.plan, ComponentCheckpointPlan)
    assert resolution.plan.descriptor is descriptor
    assert resolution.plan.identity_components == tuple(
        part for _role, part in descriptor.detector(header, header.path)
    )


def test_component_registrations_share_one_assembly_and_ambiguous_geometry_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference import (
        NativeRefusalError,
        builtin_assembly_registry,
        component_catalog,
        resolve_native_assembly,
    )

    descriptor = replace(synthetic_descriptor(), checkpoint_loader="example:load")
    registry = ComponentRegistry()
    registry.register(descriptor)
    registry.register(
        replace(
            descriptor,
            family=replace(descriptor.family, id="test.other"),
            checkpoint_loader="other:load",
        )
    )
    monkeypatch.setattr(component_catalog, "default_component_registry", lambda: registry)
    assembly = builtin_assembly_registry().get("dinkster.components")
    assert assembly is not None and assembly.aliases == (descriptor.id, "test.other")
    with pytest.raises(NativeRefusalError, match="multiple complete compositions"):
        resolve_native_assembly(checkpoint=Header(("diffusion.weight",)))


def test_loader_hint_disambiguates_shared_geometry_without_gating_detection() -> None:
    descriptor = replace(synthetic_descriptor(), text_loader_hints=("legacy_text",))
    other = replace(
        descriptor,
        family=replace(descriptor.family, id="test.other"),
        text_loader_hints=(),
    )
    registry = ComponentRegistry()
    registry.register(other)
    registry.register(descriptor)
    header = Header(("words.weight",))
    path = Path("text.safetensors")
    assert registry.select(header, path, "text", family_id="dinkster.legacy_text")[0] is descriptor
    assert registry.select(header, path, "text", family_id=other.id)[0] is other
    with pytest.raises(ValueError, match="ambiguous components"):
        registry.select(header, path, "text", family_id="dinkster.unfamiliar")
    only_other = ComponentRegistry()
    only_other.register(other)
    assert only_other.select(header, path, "text", family_id="dinkster.legacy_text")[0] is other


def test_recognized_planning_diagnostic_does_not_block_other_descriptors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import dinkster_inference as inference
    from dinkster_inference.component_catalog import _split_detector
    from dinkster_inference.families import DetectionEvidence

    descriptor = synthetic_descriptor()

    class Detector:
        def detect(self, source: WeightSource) -> DetectionEvidence:
            return DetectionEvidence("test.diagnostic", tuple(source.keys()), {})

    family = replace(descriptor.family, id="test.diagnostic", detector=Detector())

    def unavailable_plan(source: WeightSource, *, role: str, path: Path) -> None:
        del source, role, path
        raise ValueError("quantized operator configuration is missing")

    monkeypatch.setattr(inference, "plan_anima_split_component", unavailable_plan)
    registry = ComponentRegistry()
    registry.register(
        replace(
            descriptor,
            family=family,
            detector=_split_detector("plan_anima_split_component", ("diffusion",), family),
        )
    )
    registry.register(descriptor)
    selected, role, _plan = registry.select(
        Header(("diffusion.weight",)), Path("weights.safetensors"), "model"
    )
    assert selected is descriptor and role == "diffusion"
    assert "Detected Synthetic" in caplog.text
    assert "quantized operator configuration is missing" in caplog.text


def legacy_recipe_input(family_id: str, role: str) -> SimpleNamespace:
    plan = ComponentPlan(
        role,
        Path("weights.safetensors"),
        SimpleNamespace(family_id=family_id),
        {"weight": f"checkpoint.{role}.weight"},
        {"weight": BFLOAT16},
        {},
        identity_facts=("asset_digest=fixture",),
        runtime_facts=("z=last", "a=first"),
    )
    return SimpleNamespace(
        role=role,
        plan=plan,
        identity_components=(plan,),
        family_id=family_id,
        runtime=SimpleNamespace(model_role="fl2va-dit", runtime_facts=plan.runtime_facts),
    )


_LEGACY_RECIPES = cast(
    "dict[str, Any]",
    json.loads(
        (Path(__file__).parent / "fixtures" / "component_recipe_identities.json").read_text()
    ),
)


@pytest.mark.parametrize("case", _LEGACY_RECIPES["cases"])
def test_registered_recipes_preserve_legacy_identities(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:

    descriptor = default_component_registry().get(case["family"])
    assert descriptor is not None
    loaded = legacy_recipe_input(case["family"], case["role"])
    source = WeightSourceRef("blake3:" + "1" * 64, "weights.safetensors", 8)
    original_keys = dict(loaded.plan.keys)
    original_dtypes = dict(loaded.plan.dtypes)
    for dtype, expected in case["identities"].items():
        recipe = descriptor.recipe(source, loaded, dtype)
        assert recipe.runtime_identity == expected
        assert recipe.sources[0].role == case["role"]
        assert loaded.plan.keys == original_keys
        assert loaded.plan.dtypes == original_dtypes


@pytest.mark.parametrize(
    "family_id,expected_policy",
    [("dinkster.chroma", "auto"), ("dinkster.minimax_music3", "auto"), ("dinkster.ltxav", None)],
)
def test_unbound_attention_policy_preserves_component_recipe_semantics(
    family_id: str, expected_policy: str | None
) -> None:
    descriptor = default_component_registry().get(family_id)
    assert descriptor is not None
    loaded = legacy_recipe_input(family_id, "diffusion")
    source = WeightSourceRef("blake3:" + "1" * 64, "weights.safetensors", 8)
    if expected_policy is None:
        with pytest.raises(ValueError, match="named attention policy requires an authenticated"):
            descriptor.recipe(source, loaded, "bfloat16", attention_policy="sdpa")
        return
    recipe = descriptor.recipe(source, loaded, "bfloat16", attention_policy="sdpa")
    assert recipe.knobs.attention_policy == expected_policy
    assert recipe.knobs.attention_route_token is None
    assert recipe.runtime_identity == descriptor.component_identity(
        "diffusion", loaded.plan, "bfloat16", attention_policy="sdpa"
    )
