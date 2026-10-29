"""S2 out-of-tree patch-provider, ordering, identity, and apply proofs."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference import (
    AdapterPatch,
    LoRASpec,
    OverlayPatch,
    PatchOverlay,
    PatchTarget,
    ProviderPatchRef,
    ReconstructionRecipe,
    RuntimeKnobs,
    WeightSourceBinding,
    WeightSourceRef,
    patch_overlay_stack_digest,
)
from dinkster_inference_torch import (
    MaterializeError,
    PatchProviderExtensionEntry,
    PatchProviderSnapshot,
    apply_patches,
    build_patch_set,
    materialize_patch_provider_registry,
    materialize_value,
    remove_patch_provider_catalog_record,
    write_patch_provider_catalog,
)

FIXTURES = Path(__file__).parent
ENTRIES = (
    PatchProviderExtensionEntry("proof_b", "s2_patch_pack_b:register"),
    PatchProviderExtensionEntry("proof_a", "s2_patch_pack_a:register"),
)
MODULES = ("s2_patch_pack_a", "s2_patch_pack_b")


def _materialized(tmp_path: Path, *, candidate: bool = False):
    catalog = tmp_path / "patch-providers.json"
    key = (
        f"candidate:{uuid.uuid4().hex}"
        if candidate
        else "sha256:" + hashlib.sha256(str(tmp_path).encode("utf-8")).hexdigest()
    )
    write_patch_provider_catalog(catalog, key, ENTRIES)
    return catalog, key, materialize_patch_provider_registry(key, catalog_path=catalog)


def _source(digit: str) -> WeightSourceRef:
    return WeightSourceRef("blake3:" + digit * 64, "patch.safetensors", 4)


def _overlay(source: WeightSourceRef, decoded: ProviderPatchRef, provider: str) -> PatchOverlay:
    return PatchOverlay.from_decoded(
        source=source,
        dialect="proof",
        key_map=provider + ".v1",
        strength_model=1.0,
        strength_clip=0.0,
        patches=(OverlayPatch("diffusion", PatchTarget("weight"), decoded),),
    )


def test_out_of_tree_providers_materialize_from_opaque_worker_catalog(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(FIXTURES))
    try:
        catalog, key, materialized = _materialized(tmp_path)
        assert key.startswith("sha256:")
        assert materialized.extension_ids == ("proof_a", "proof_b")
        assert materialized.registry.ids() == ("proof_a.scale", "proof_b.shift")
        assert tuple(item.id for item in materialized.snapshot.providers) == (
            "proof_a.scale",
            "proof_b.shift",
        )
        document = json.loads(catalog.read_text(encoding="utf-8"))
        assert tuple(document["records"]) == (key,)
        assert materialize_patch_provider_registry(key, catalog_path=catalog) is materialized
    finally:
        sys.path.remove(str(FIXTURES))
        for module in MODULES:
            sys.modules.pop(module, None)


def test_candidate_is_not_cached_and_transient_record_is_removable(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(FIXTURES))
    try:
        catalog, key, first = _materialized(tmp_path, candidate=True)
        second = materialize_patch_provider_registry(key, catalog_path=catalog)
        assert second is not first
        remove_patch_provider_catalog_record(catalog, key)
        assert key not in json.loads(catalog.read_text(encoding="utf-8"))["records"]
    finally:
        sys.path.remove(str(FIXTURES))
        for module in MODULES:
            sys.modules.pop(module, None)


def test_empty_provider_snapshot_has_an_empty_registry(tmp_path: Path) -> None:
    catalog = tmp_path / "patch-providers.json"
    key = "sha256:" + hashlib.sha256(str(tmp_path).encode()).hexdigest()
    write_patch_provider_catalog(catalog, key, (), PatchProviderSnapshot())
    materialized = materialize_patch_provider_registry(key, catalog_path=catalog)
    assert materialized.registry.ids() == ()
    assert materialized.snapshot == PatchProviderSnapshot()
    assert materialized.extension_ids == ()


def test_worker_materialization_validates_provider_declarations(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(FIXTURES))
    try:
        catalog, _key, produced = _materialized(tmp_path, candidate=True)
        mismatch = f"candidate:{uuid.uuid4().hex}"
        write_patch_provider_catalog(
            catalog,
            mismatch,
            ENTRIES,
            PatchProviderSnapshot(produced.snapshot.providers[:-1]),
        )
        with pytest.raises(RuntimeError, match="declaration mismatch"):
            materialize_patch_provider_registry(mismatch, catalog_path=catalog)
    finally:
        sys.path.remove(str(FIXTURES))
        for module in MODULES:
            sys.modules.pop(module, None)


def test_two_provider_overlays_flow_decode_identity_apply_and_order(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(FIXTURES))
    try:
        _catalog, _key, materialized = _materialized(tmp_path)
        pack_a = importlib.import_module("s2_patch_pack_a")
        pack_b = importlib.import_module("s2_patch_pack_b")
        scale_ref = pack_a.decode("factor")
        shift_ref = pack_b.decode("delta")
        scale_overlay = _overlay(_source("1"), scale_ref, pack_a.PROVIDER_ID)
        shift_overlay = _overlay(_source("2"), shift_ref, pack_b.PROVIDER_ID)
        assert patch_overlay_stack_digest(
            (scale_overlay, shift_overlay)
        ) != patch_overlay_stack_digest((shift_overlay, scale_overlay))

        scale = build_patch_set(
            {PatchTarget("weight"): scale_ref},
            {"factor": torch.tensor(2.0)},
            structural_digest=scale_overlay.structural_digest,
            provider_registry=materialized.registry,
        )
        shift = build_patch_set(
            {PatchTarget("weight"): shift_ref},
            {"delta": torch.tensor(3.0)},
            structural_digest=shift_overlay.structural_digest,
            provider_registry=materialized.registry,
        )
        scale_then_shift = apply_patches(
            torch.tensor(1.0), scale.entries("weight") + shift.entries("weight")
        )
        shift_then_scale = apply_patches(
            torch.tensor(1.0), shift.entries("weight") + scale.entries("weight")
        )
        assert scale_then_shift.item() == 5.0
        assert shift_then_scale.item() == 8.0
        patch = scale.entries("weight")[0].value
        assert isinstance(patch, AdapterPatch)
        adapter: Any = patch.adapter
        assert tuple(adapter.payload_tensors()) == (adapter.factor,)
        assert adapter.rebuild_payloads((torch.tensor(4.0),)).factor.item() == 4.0
    finally:
        sys.path.remove(str(FIXTURES))
        for module in MODULES:
            sys.modules.pop(module, None)


def test_unknown_provider_refuses_loudly() -> None:
    with pytest.raises(MaterializeError, match="not registered"):
        materialize_value(ProviderPatchRef("missing.provider", ()), {})


def test_builtin_lora_behavior_stays_available_without_provider_registry() -> None:
    spec = LoRASpec("up", "down")
    tensors = {
        "up": torch.tensor([[2.0]]),
        "down": torch.tensor([[3.0]]),
    }
    value = materialize_value(spec, tensors)
    patch_set = build_patch_set(
        {PatchTarget("weight"): spec},
        tensors,
    )
    result = apply_patches(torch.tensor([[1.0]]), patch_set.entries("weight"))
    assert isinstance(value, AdapterPatch)
    assert result.item() == 7.0


def test_clone_overlay_materialize_drop_rebuild_preserves_golden_behavior() -> None:
    spec = LoRASpec("up", "down")
    lora = PatchOverlay.from_decoded(
        source=_source("3"),
        dialect="none",
        key_map="native.dinkster.proof.v1",
        strength_model=1.0,
        strength_clip=0.0,
        patches=(OverlayPatch("diffusion", PatchTarget("weight"), spec),),
    )
    base = ReconstructionRecipe(
        sources=(WeightSourceBinding("checkpoint", _source("0")),),
        family_id="dinkster.proof",
        component_identity=("family=dinkster.proof", "component=diffusion"),
        knobs=RuntimeKnobs("float32", "float32", "float32", False),
    )
    cloned = base.append_overlays((lora,))

    def materialize(recipe: ReconstructionRecipe) -> tuple[str, torch.Tensor]:
        weight = torch.tensor([[1.0]])
        for overlay in recipe.overlays:
            patch_set = build_patch_set(
                {patch.target: patch.decoded for patch in overlay.patches},
                {
                    "up": torch.tensor([[2.0]]),
                    "down": torch.tensor([[3.0]]),
                },
                strength=float(overlay.strength_model),
                structural_digest=overlay.structural_digest,
            )
            weight = apply_patches(weight, patch_set.entries("weight"))
        sample = weight @ torch.tensor([[7.0]])
        return recipe.runtime_identity, sample

    base_identity, base_sample = materialize(base)
    first_identity, first_sample = materialize(cloned)
    rebuilt_identity, rebuilt_sample = materialize(cloned)
    assert base_sample.item() == 7.0
    assert first_sample.item() == 49.0
    assert first_identity != base_identity
    assert rebuilt_identity == first_identity
    assert torch.equal(rebuilt_sample, first_sample)


def test_provider_descriptor_type_is_worker_local_not_recipe_data(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(FIXTURES))
    try:
        _catalog, _key, materialized = _materialized(tmp_path)
        provider = materialized.registry.get("proof_a.scale")
        assert provider is not None
        assert callable(provider.materialize)
        declaration = materialized.snapshot.providers[0]
        assert not any(callable(value) for _key, value in declaration.behavior_metadata)
    finally:
        sys.path.remove(str(FIXTURES))
        for module in MODULES:
            sys.modules.pop(module, None)
