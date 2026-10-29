"""S2 reconstruction recipe and normalized overlay identity proofs."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict, replace

import pytest
from dinkster_inference import (
    FLOAT32,
    DependencyRef,
    DiffPatchRef,
    OverlayPatch,
    PatchOverlay,
    PatchTarget,
    ProviderPatchRef,
    ReconstructionRecipe,
    RuntimeKnobs,
    WeightSourceBinding,
    WeightSourceRef,
    canonical_patch_overlay,
    patch_overlay_stack_digest,
)
from dinkster_inference.recipe import GGUF_PROVIDER_PIN, ProviderPin


def source(digit: str = "1") -> WeightSourceRef:
    return WeightSourceRef(
        digest="blake3:" + digit * 64,
        name="weights.safetensors",
        size=123,
    )


def test_gguf_provider_pin_metadata_is_exact_and_frozen() -> None:
    assert GGUF_PROVIDER_PIN == ProviderPin(
        provider_id="ggml.gguf-py.v0",
        distribution="gguf",
        version="0.19.0",
        source_revision="a290ce626663dae1d54f70bce3ca6d8f67aab62f",
        wheel_sha256="70bcd10edfe697fb2dad6e40af2234b9d8ece9a41a99761405121ebda1c3c1cd",
    )
    with pytest.raises(FrozenInstanceError):
        GGUF_PROVIDER_PIN.version = "0.19.1"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("provider_id", "", "provider_id"),
        ("provider_id", "GGML.gguf-py.v0", "provider_id"),
        ("distribution", "gguf_py", "distribution"),
        ("version", "v0.19.0", "version"),
        ("source_revision", "A" * 40, "source_revision"),
        ("wheel_sha256", "0" * 63, "wheel_sha256"),
    ),
)
def test_provider_pin_constructor_requires_canonical_metadata(
    field: str, value: str, message: str
) -> None:
    values = asdict(GGUF_PROVIDER_PIN)
    values[field] = value
    with pytest.raises(ValueError, match=message):
        ProviderPin(**values)


def test_provider_pin_wire_round_trip_is_canonical_and_strict() -> None:
    wire = GGUF_PROVIDER_PIN.to_wire()
    assert wire == (
        '{"distribution":"gguf","providerId":"ggml.gguf-py.v0",'
        '"sourceRevision":"a290ce626663dae1d54f70bce3ca6d8f67aab62f",'
        '"version":"0.19.0","wheelSha256":'
        '"70bcd10edfe697fb2dad6e40af2234b9d8ece9a41a99761405121ebda1c3c1cd"}'
    )
    assert ProviderPin.from_wire(wire, expected=GGUF_PROVIDER_PIN) is not (GGUF_PROVIDER_PIN)
    assert ProviderPin.from_wire(wire, expected=GGUF_PROVIDER_PIN) == (GGUF_PROVIDER_PIN)
    for drift in (
        replace(GGUF_PROVIDER_PIN, version="0.19.1"),
        replace(GGUF_PROVIDER_PIN, source_revision="0" * 40),
        replace(GGUF_PROVIDER_PIN, wheel_sha256="0" * 64),
    ):
        assert drift != GGUF_PROVIDER_PIN
        with pytest.raises(ValueError, match="expected provider pin"):
            ProviderPin.from_wire(drift.to_wire(), expected=GGUF_PROVIDER_PIN)
    with pytest.raises(ValueError, match="canonical provider pin wire"):
        ProviderPin.from_wire(wire + " ", expected=GGUF_PROVIDER_PIN)
    with pytest.raises(ValueError, match="exact fields"):
        ProviderPin.from_wire(
            wire.removesuffix("}") + ',"capability":true}',
            expected=GGUF_PROVIDER_PIN,
        )


def overlay(
    digit: str = "1", *, strength: float = 1.0, key_map: str = "native.sd15.v1"
) -> PatchOverlay:
    return PatchOverlay.from_decoded(
        source=source(digit),
        dialect="none",
        key_map=key_map,
        strength_model=strength,
        strength_clip=0.5,
        patches=(
            OverlayPatch(
                "diffusion",
                PatchTarget("z.weight"),
                DiffPatchRef("z.diff"),
            ),
            OverlayPatch(
                "diffusion",
                PatchTarget("a.weight"),
                DiffPatchRef("a.diff"),
            ),
        ),
    )


def recipe(
    *overlays: PatchOverlay,
    dependencies: tuple[DependencyRef[ReconstructionRecipe], ...] = (),
) -> ReconstructionRecipe:
    return ReconstructionRecipe(
        sources=(WeightSourceBinding("checkpoint", source("0")),),
        family_id="dinkster.sd15",
        component_identity=("family=dinkster.sd15", "component=diffusion"),
        knobs=RuntimeKnobs(
            diffusion_dtype="float16",
            text_dtype=FLOAT32.name,
            vae_dtype=FLOAT32.name,
            fp8_matmul=False,
        ),
        overlays=overlays,
        dependencies=dependencies,
    )


def dependency(
    child_id: str,
    child: ReconstructionRecipe,
    *,
    residency_group: str = "group/main",
    scope: str = "model",
    clone_mode: str = "with-parent",
    accounting_owner: str = "parent",
) -> DependencyRef[ReconstructionRecipe]:
    return DependencyRef(
        child_id=child_id,
        child=child,
        residency_group=residency_group,
        scope=scope,  # type: ignore[arg-type]
        clone_mode=clone_mode,  # type: ignore[arg-type]
        accounting_owner=accounting_owner,
    )


def test_weight_source_is_path_free_rpc_clean_asset_identity() -> None:
    value = source()
    wire = asdict(value)
    assert "path" not in wire
    assert value.digest == "blake3:" + "1" * 64
    encoded = json.dumps(wire)
    assert '"path"' not in encoded
    assert json.loads(encoded)["provider_config"] == []
    with pytest.raises(ValueError, match="canonical blake3"):
        replace(value, digest="sha256:" + "1" * 64)


def test_out_of_tree_patch_ref_is_plain_rpc_clean_data() -> None:
    decoded = ProviderPatchRef("proof.scale", (("enabled", True), ("tensorKey", "factor")))
    patch = OverlayPatch("diffusion", PatchTarget("weight"), decoded)
    wire = asdict(patch)
    assert json.loads(json.dumps(wire))["decoded"] == {
        "provider_id": "proof.scale",
        "parameters": [["enabled", True], ["tensorKey", "factor"]],
    }
    with pytest.raises(ValueError, match="sorted and unique"):
        replace(decoded, parameters=tuple(reversed(decoded.parameters)))


def test_overlay_digest_is_canonical_and_patch_order_independent() -> None:
    first = overlay()
    reversed_input = PatchOverlay.from_decoded(
        source=first.source,
        dialect=first.dialect,
        key_map=first.key_map,
        strength_model=float(first.strength_model),
        strength_clip=float(first.strength_clip),
        patches=tuple(reversed(first.patches)),
    )
    assert first == reversed_input
    assert first.structural_digest == reversed_input.structural_digest
    assert tuple(patch.target.key for patch in first.patches) == (
        "a.weight",
        "z.weight",
    )
    document = json.loads(canonical_patch_overlay(first))
    assert document["strengthModel"] == "1"
    assert document["strengthClip"] == "0.5"


def test_overlay_digest_covers_content_strength_key_map_and_decoded_mapping() -> None:
    base = overlay()
    variants = (
        overlay("2"),
        overlay(strength=0.75),
        overlay(key_map="native.sd15.v2"),
        PatchOverlay.from_decoded(
            source=base.source,
            dialect=base.dialect,
            key_map=base.key_map,
            strength_model=1.0,
            strength_clip=0.5,
            patches=(
                OverlayPatch(
                    "diffusion",
                    PatchTarget("a.weight"),
                    DiffPatchRef("different.diff"),
                ),
            ),
        ),
    )
    assert len({base.structural_digest, *(item.structural_digest for item in variants)}) == 5
    with pytest.raises(ValueError, match="finite"):
        overlay(strength=float("nan"))


def test_overlay_stack_identity_is_order_sensitive_and_empty_is_absent() -> None:
    first = overlay("1")
    second = overlay("2")
    assert patch_overlay_stack_digest(()) is None
    assert patch_overlay_stack_digest((first, second)) != patch_overlay_stack_digest(
        (second, first)
    )
    assert recipe(first, second).runtime_identity != recipe(second, first).runtime_identity


def test_recipe_append_is_immutable_and_runtime_identity_is_reproducible() -> None:
    base = recipe()
    added = base.append_overlays((overlay(),))
    assert base.overlays == ()
    assert added.overlays != ()
    assert base.runtime_identity == recipe().runtime_identity
    assert added.runtime_identity == recipe(overlay()).runtime_identity
    with pytest.raises(TypeError, match="tuple"):
        base.append_overlays([])  # type: ignore[arg-type]


def test_recipe_requires_canonical_source_and_component_order() -> None:
    with pytest.raises(TypeError, match="sources must be a tuple"):
        replace(recipe(), sources=list(recipe().sources))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ordered by role"):
        ReconstructionRecipe(
            sources=(
                WeightSourceBinding("vae", source()),
                WeightSourceBinding("checkpoint", source("2")),
            ),
            family_id="dinkster.sd15",
            component_identity=("family=dinkster.sd15",),
            knobs=recipe().knobs,
        )
    with pytest.raises(ValueError, match="begin with its family"):
        replace(recipe(), component_identity=("family=other",))


def test_empty_dependencies_preserve_runtime_identity_bytes() -> None:
    base = recipe()
    omitted = ReconstructionRecipe(
        sources=base.sources,
        family_id=base.family_id,
        component_identity=base.component_identity,
        knobs=base.knobs,
        overlays=base.overlays,
        attachments=base.attachments,
    )
    explicit = ReconstructionRecipe(
        sources=omitted.sources,
        family_id=omitted.family_id,
        component_identity=omitted.component_identity,
        knobs=omitted.knobs,
        overlays=omitted.overlays,
        attachments=omitted.attachments,
        dependencies=(),
    )
    assert explicit.runtime_identity == omitted.runtime_identity


def test_dependency_ref_and_recipe_dependency_tuple_are_frozen() -> None:
    edge = dependency("controlnet", recipe())
    parent = recipe(dependencies=(edge,))
    with pytest.raises(FrozenInstanceError):
        edge.scope = "invocation"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        parent.dependencies = ()  # type: ignore[misc]


def test_recipe_dependency_container_and_children_are_strict() -> None:
    edge = dependency("controlnet", recipe())
    with pytest.raises(TypeError, match="dependencies must be a tuple"):
        replace(recipe(), dependencies=[edge])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple of DependencyRef"):
        replace(recipe(), dependencies=("controlnet",))  # type: ignore[arg-type]
    malformed = dependency("controlnet", "not-a-recipe")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="children must be ReconstructionRecipe"):
        replace(recipe(), dependencies=(malformed,))  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ("child_id", "residency_group", "accounting_owner"))
@pytest.mark.parametrize("bad_id", ("Uppercase", ".leading", ""))
def test_dependency_ids_are_canonical(field: str, bad_id: str) -> None:
    values = {
        "child_id": "controlnet",
        "residency_group": "group/main",
        "accounting_owner": "parent",
    }
    values[field] = bad_id
    with pytest.raises(ValueError, match=f"dependency {field} must be a canonical id"):
        DependencyRef(
            child_id=values["child_id"],
            child=recipe(),
            residency_group=values["residency_group"],
            scope="model",
            clone_mode="with-parent",
            accounting_owner=values["accounting_owner"],
        )


@pytest.mark.parametrize("field", ("child_id", "residency_group", "accounting_owner"))
def test_dependency_ids_must_be_strings(field: str) -> None:
    edge = dependency("controlnet", recipe())
    with pytest.raises(TypeError, match=f"dependency {field} must be a string"):
        replace(edge, **{field: 1})


def test_dependency_reserved_duplicate_and_accounting_refusals() -> None:
    child = recipe()
    with pytest.raises(ValueError, match="reserved"):
        dependency("parent", child)
    with pytest.raises(ValueError, match="child_ids must be unique"):
        recipe(dependencies=(dependency("aux", child), dependency("aux", child)))
    with pytest.raises(ValueError, match="accounting_owner is unknown"):
        recipe(dependencies=(dependency("aux", child, accounting_owner="missing"),))
    with pytest.raises(ValueError, match="same residency_group"):
        recipe(
            dependencies=(
                dependency("owner", child, residency_group="group/one"),
                dependency(
                    "aux",
                    child,
                    residency_group="group/two",
                    accounting_owner="owner",
                ),
            )
        )
    with pytest.raises(ValueError, match="must share accounting_owner"):
        recipe(
            dependencies=(
                dependency("owner", child),
                dependency("aux", child, accounting_owner="owner"),
            )
        )


def test_dependency_scope_and_clone_mode_are_closed() -> None:
    with pytest.raises(ValueError, match="scope is unsupported"):
        dependency("aux", recipe(), scope="session")
    with pytest.raises(ValueError, match="clone_mode is unsupported"):
        dependency("aux", recipe(), clone_mode="copy")


def test_dependency_order_changes_equality_and_runtime_identity() -> None:
    first = dependency("controlnet", recipe())
    second = dependency("upscaler", recipe(), residency_group="group/upscale")
    forward = recipe(dependencies=(first, second))
    reverse = recipe(dependencies=(second, first))
    assert forward != reverse
    assert forward.runtime_identity != reverse.runtime_identity


def test_structurally_equal_nested_dependency_graphs_reproduce_identity() -> None:
    def nested() -> ReconstructionRecipe:
        grandchild = recipe()
        child = recipe(dependencies=(dependency("adapter", grandchild),))
        return recipe(dependencies=(dependency("controlnet", child),))

    assert nested().runtime_identity == nested().runtime_identity


def test_dependency_graph_refuses_cycles_defensively() -> None:
    child = recipe()
    object.__setattr__(
        child,
        "dependencies",
        (dependency("self", child),),
    )
    with pytest.raises(ValueError, match="must be acyclic"):
        replace(child)


def test_append_overlays_preserves_dependencies() -> None:
    dependencies = (dependency("controlnet", recipe()),)
    base = recipe(dependencies=dependencies)
    assert base.append_overlays((overlay(),)).dependencies is dependencies


def test_dependency_recipe_asdict_contains_only_plain_data() -> None:
    value = asdict(
        recipe(
            dependencies=(
                dependency(
                    "controlnet",
                    recipe(dependencies=(dependency("adapter", recipe()),)),
                ),
            )
        )
    )

    def assert_plain(item: object) -> None:
        assert isinstance(item, str | int | bool | tuple | dict) or item is None
        if isinstance(item, tuple):
            for nested in item:
                assert_plain(nested)
        elif isinstance(item, dict):
            for key, nested in item.items():
                assert isinstance(key, str)
                assert_plain(nested)

    assert_plain(value)
