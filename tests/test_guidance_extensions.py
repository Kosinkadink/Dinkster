"""Focused generalized inference-catalog proofs for guidance extensions."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from dinkster_inference import (
    GuidanceContribution,
    GuidanceEvaluationWrapperDescriptor,
    GuidancePlanAugmentationDescriptor,
    SamplerExtensionEntry,
    builtin_sampler_snapshot,
    guidance_declarations,
    materialize_inference_generation,
    release_inference_generation,
    remove_sampler_catalog_record,
    write_sampler_catalog,
)
from dinkster_inference.extensions import _inference_cache
from dinkster_protocol import GuidanceRegistrySnapshot, KeyedContribution
from test_inference_extensions import _extension_manifest, _host_manifest, _worker_env

from dinkster.compose import CompositionError, PackSpec, ServingComposer

FIXTURES = Path(__file__).parent
MODULES = ("s3_guidance_pack_a", "s3_guidance_pack_b")
ENTRIES = (
    SamplerExtensionEntry("proof_b", "s3_guidance_pack_b:register"),
    SamplerExtensionEntry("proof_a", "s3_guidance_pack_a:register"),
)


@pytest.fixture(autouse=True)
def _fixture_modules():
    sys.path.insert(0, str(FIXTURES))
    for module in MODULES:
        sys.modules.pop(module, None)
    yield
    for module in MODULES:
        sys.modules.pop(module, None)
    sys.path.remove(str(FIXTURES))


def _key(*, candidate: bool = True) -> str:
    prefix = "candidate:" if candidate else "sha256:"
    return prefix + uuid.uuid4().hex


def _materialize(catalog: Path, entries=ENTRIES, *, candidate: bool = True):
    key = _key(candidate=candidate)
    write_sampler_catalog(catalog, key, entries)
    generation = materialize_inference_generation(key, catalog_path=catalog)
    return key, generation


def _ids(snapshot: GuidanceRegistrySnapshot) -> tuple[str, ...]:
    return tuple(item.id for item in snapshot.contributions)


def test_two_packs_materialize_exact_guidance_surfaces_and_order(tmp_path: Path) -> None:
    key, generation = _materialize(tmp_path / "catalog.json")
    assert generation.sampler_snapshot == builtin_sampler_snapshot()
    assert tuple(extension for extension, _ in generation.extensions) == ("proof_a", "proof_b")
    assert _ids(generation.guidance_snapshot) == (
        "proof_b.guidance_strategy.wrapper",
        "proof_a.cfg_rescale.wrapper",
        "proof_b.guidance_strategy.pre",
        "proof_a.cfg_rescale.pre",
        "proof_b.guidance_strategy",
        "proof_b.guidance_strategy.post",
        "proof_a.cfg_rescale.post",
    )
    assert tuple(item.surface_id for item in generation.guidance_snapshot.contributions) == (
        "inference.guidance.condition-evaluation",
        "inference.guidance.condition-evaluation",
        "inference.guidance.pre-cfg",
        "inference.guidance.pre-cfg",
        "inference.guidance.strategy",
        "inference.guidance.post-cfg",
        "inference.guidance.post-cfg",
    )
    assert tuple(extension for extension, _ in generation.guidance_contributions) == (
        "proof_a",
        "proof_b",
    )
    release_inference_generation(key)


def test_activation_order_is_neutral(tmp_path: Path) -> None:
    first_key, forward = _materialize(tmp_path / "catalog.json", ENTRIES)
    first = forward.guidance_snapshot
    release_inference_generation(first_key)
    for module in MODULES:
        sys.modules.pop(module, None)
    key, reverse = _materialize(tmp_path / "catalog.json", tuple(reversed(ENTRIES)))
    assert reverse.guidance_snapshot == first
    release_inference_generation(key)


def test_isolated_worker_keeps_pack_modules_out_of_parent(tmp_path: Path) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env(tmp_path))
        try:
            await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            await composer.add_pack(
                _extension_manifest(tmp_path / "proof_b", "proof_b", "s3_guidance_pack_b")
            )
            await composer.add_pack(
                _extension_manifest(tmp_path / "proof_a", "proof_a", "s3_guidance_pack_a")
            )
            assert all(module not in sys.modules for module in MODULES)
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_attention_contribution_declares_canonical_surface() -> None:
    from dinkster_inference import AttentionGuidanceDescriptor

    contribution: GuidanceContribution[Any] = GuidanceContribution(
        attention=AttentionGuidanceDescriptor("proof.att", lambda positive, negative: positive)
    )
    declarations = guidance_declarations(contribution, attention_order=2)
    assert tuple((item.surface_id, item.id) for item in declarations) == (
        ("inference.guidance.attention", "proof.att"),
    )
    metadata = dict(declarations[0].behavior_metadata)
    assert metadata["contractVersion"] == 1
    assert metadata["order"] == 2
    assert metadata["requiresUncond"] is False
    GuidanceRegistrySnapshot(declarations)


def test_plan_augmentations_declare_canonical_surface() -> None:
    contribution: GuidanceContribution[Any] = GuidanceContribution(
        plan_augmentations=(
            GuidancePlanAugmentationDescriptor(
                "proof.plan", lambda context, plan: plan, order=4, requires_uncond=True
            ),
        )
    )
    declarations = guidance_declarations(contribution)
    assert tuple((item.surface_id, item.id) for item in declarations) == (
        ("inference.guidance.plan-augmentation", "proof.plan"),
    )
    metadata = dict(declarations[0].behavior_metadata)
    assert metadata["contractVersion"] == 1
    assert metadata["order"] == 4
    assert metadata["requiresUncond"] is True
    GuidanceRegistrySnapshot(declarations)


def test_exclusive_strategies_are_rejected() -> None:
    from s3_guidance_pack_b import GUIDANCE

    declarations = guidance_declarations(GUIDANCE)
    strategy = next(item for item in declarations if item.surface_id.endswith("strategy"))
    with pytest.raises(ValueError, match="guidance strategy is exclusive"):
        GuidanceRegistrySnapshot((replace(strategy, id="proof_a.other_strategy"), strategy))


def test_owner_aware_strategy_collision_preserves_catalog_generation(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    good_key, good = _materialize(catalog)
    old_surface = good.guidance_snapshot
    old_catalog = json.loads(catalog.read_text(encoding="utf-8"))
    collision = (
        SamplerExtensionEntry("proof_a", "s3_guidance_pack_a:register_with_strategy"),
        ENTRIES[0],
    )
    with pytest.raises(RuntimeError) as caught:
        _materialize(catalog, collision)
    diagnostic = str(caught.value)
    assert "extension=proof_a contribution=proof_a.alternate_strategy" in diagnostic
    assert "extension=proof_b contribution=proof_b.guidance_strategy" in diagnostic
    rematerialized = materialize_inference_generation(good_key, catalog_path=catalog)
    assert rematerialized.guidance_snapshot == old_surface
    current_catalog = json.loads(catalog.read_text(encoding="utf-8"))
    assert current_catalog["records"][good_key] == old_catalog["records"][good_key]
    release_inference_generation(good_key)


def test_bypass_variant_declares_exact_strategy_surface(tmp_path: Path) -> None:
    entry = SamplerExtensionEntry("proof_b", "s3_guidance_pack_b:register_bypass")
    key, generation = _materialize(tmp_path / "catalog.json", (entry,))
    strategy = next(
        item
        for item in generation.guidance_snapshot.contributions
        if item.surface_id == "inference.guidance.strategy"
    )
    assert strategy.id == "proof_b.guidance_strategy.bypass"
    assert strategy.surface_id == "inference.guidance.strategy"
    assert dict(strategy.behavior_metadata)["participation"] == "bypass-transforms"
    release_inference_generation(key)


@pytest.mark.parametrize("kind", ["missing", "extra", "reordered", "metadata"])
def test_materialization_validation_is_bidirectional(tmp_path: Path, kind: str) -> None:
    catalog = tmp_path / "catalog.json"
    _, produced = _materialize(catalog)
    expected = list(produced.extensions)
    extension, declarations = expected[0]
    values = list(declarations)
    if kind == "missing":
        values.pop()
    elif kind == "extra":
        values.append(replace(values[0], id="proof_a.extra"))
    elif kind == "reordered":
        values.reverse()
    else:
        values[0] = replace(values[0], behavior_metadata=(("contractVersion", 99),))
    expected[0] = (extension, tuple(values))
    key = _key()
    write_sampler_catalog(catalog, key, ENTRIES, expected_extensions=expected)
    with pytest.raises(RuntimeError, match="declarations changed"):
        materialize_inference_generation(key, catalog_path=catalog)
    assert all(module not in sys.modules for module in MODULES)


def test_registration_callback_raise_cleans_imported_modules(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    key = _key()
    entry = SamplerExtensionEntry("proof_a", "s3_guidance_pack_a:register_raises")
    write_sampler_catalog(catalog, key, (entry,))
    with pytest.raises(RuntimeError, match="proof_a callback raised"):
        materialize_inference_generation(key, catalog_path=catalog)
    assert "s3_guidance_pack_a" not in sys.modules


def test_candidate_not_lru_and_catalog_removal(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    key, first = _materialize(catalog)
    second = materialize_inference_generation(key, catalog_path=catalog)
    assert second is not first
    remove_sampler_catalog_record(catalog, key)
    assert key not in json.loads(catalog.read_text(encoding="utf-8"))["records"]
    release_inference_generation(key)


def test_published_generations_remain_cached_until_worker_retirement(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    baseline = set(sys.modules)
    active_keys = []
    generations = []
    for _ in range(12):
        key, generation = _materialize(catalog, candidate=False)
        active_keys.append(key)
        generations.append(generation)
    assert all(
        _inference_cache[key] is generation
        for key, generation in zip(active_keys, generations, strict=True)
    )
    assert len({id(_inference_cache[key]) for key in active_keys}) == len(active_keys)
    assert sum(name == module for name in sys.modules for module in MODULES) == len(MODULES)
    for key in active_keys:
        release_inference_generation(key)
    assert all(key not in _inference_cache for key in active_keys)
    assert all(module not in sys.modules for module in MODULES)
    for _ in range(12):
        key, generation = _materialize(catalog, candidate=False)
        assert all(module in sys.modules for module in generation.module_prefixes)
        assert materialize_inference_generation(key, catalog_path=catalog) is generation
        release_inference_generation(key)
        assert all(module not in sys.modules for module in MODULES)
    assert not ({name for name in sys.modules if name.startswith("s3_guidance_pack_")} - baseline)


def test_composer_strategy_collision_rolls_back_exact_surface(tmp_path: Path) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env(tmp_path))
        try:
            await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            await composer.add_pack(
                _extension_manifest(tmp_path / "proof_b", "proof_b", "s3_guidance_pack_b")
            )
            before_runtime = composer._runtime_seat.pin()
            before_choices = dict(composer.composition.choices)
            before_catalog = composer._sampler_catalog_path.read_bytes()
            manifest = _extension_manifest(tmp_path / "proof_a", "proof_a", "s3_guidance_pack_a")
            text = manifest.read_text(encoding="utf-8").replace(
                ':register"', ':register_with_strategy"'
            )
            manifest.write_text(text, encoding="utf-8")
            with pytest.raises(CompositionError) as caught:
                await composer.add_pack(PackSpec(manifest))
            message = str(caught.value)
            assert "inference:inference.guidance.strategy" in message
            assert "proof_a" in message and "proof_b" in message
            assert composer._runtime_seat.pin() is before_runtime
            assert composer.composition.choices == before_choices
            assert composer._sampler_catalog_path.read_bytes() == before_catalog
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_host_types_failure_after_materialization_rolls_back_generation(tmp_path: Path) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env(tmp_path))
        try:
            await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            assert not composer._sampler_catalog_path.exists()
            manifest = _extension_manifest(tmp_path / "proof_b", "proof_b", "s3_guidance_pack_b")

            def fail(_registry: object) -> None:
                raise RuntimeError("host type boom")

            with pytest.raises(RuntimeError, match="host type boom"):
                await composer.add_pack(PackSpec(manifest, host_types=fail))
            if composer._sampler_catalog_path.exists():
                catalog = json.loads(composer._sampler_catalog_path.read_text(encoding="utf-8"))
                assert catalog["records"] == {}
            assert composer._published_generation_digest is None
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_retired_digest_activation_is_refused_exactly(tmp_path: Path) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env(tmp_path))
        try:
            await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            manifest = _extension_manifest(tmp_path / "proof_b", "proof_b", "s3_guidance_pack_b")
            await composer.add_pack(PackSpec(manifest))
            digest = composer._published_generation_digest
            assert digest is not None
            await composer.remove_pack("proof_b")
            # Admitted work retains access to its published generation.
            assert composer._retired_generation_digests == {digest}
            catalog = json.loads(composer._sampler_catalog_path.read_text(encoding="utf-8"))
            assert set(catalog["records"]) == {digest}
            with pytest.raises(
                CompositionError,
                match=(
                    rf"inference generation {digest} is retired and cannot be materialized "
                    "or activated for new work"
                ),
            ):
                await composer.add_pack(PackSpec(manifest))
            assert composer._retired_generation_digests == {digest}
            assert set(
                json.loads(composer._sampler_catalog_path.read_text(encoding="utf-8"))["records"]
            ) == {digest}
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_legacy_sampler_contribution_preserves_snapshot(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    key = _key()
    entry = SamplerExtensionEntry("proof_a", "s1_sampler_pack_a:register")
    write_sampler_catalog(catalog, key, (entry,))
    generation = materialize_inference_generation(key, catalog_path=catalog)
    assert generation.sampler_snapshot.samplers[:-1] == builtin_sampler_snapshot().samplers
    assert generation.sampler_snapshot.samplers[-1].id == "proof_a.scaled_euler"
    assert generation.guidance_snapshot == GuidanceRegistrySnapshot()
    release_inference_generation(key)
    sys.modules.pop("s1_sampler_pack_a", None)


def test_malformed_duplicate_id_namespace_and_metadata() -> None:
    def callback(request: Any, next: Any) -> Any:
        return next(request)

    with pytest.raises(ValueError, match="namespaced"):
        GuidanceEvaluationWrapperDescriptor("bare", callback)
    with pytest.raises(ValueError, match=r"sorted unique config\.\*"):
        GuidanceEvaluationWrapperDescriptor(
            "proof.bad", callback, behavior_metadata=(("other", True),)
        )
    descriptor = GuidanceEvaluationWrapperDescriptor("proof.duplicate", callback)
    with pytest.raises(ValueError, match="globally unique"):
        GuidanceContribution(evaluation_wrappers=(descriptor, descriptor))
    with pytest.raises(ValueError, match="extension ids must be unique"):
        write_sampler_catalog(
            Path("unused.json"),
            "candidate:duplicate",
            (ENTRIES[0], replace(ENTRIES[0], entry_point=ENTRIES[1].entry_point)),
        )
    with pytest.raises(ValueError, match="guidance contractVersion"):
        GuidanceRegistrySnapshot(
            (
                KeyedContribution(
                    "inference.guidance.condition-evaluation",
                    "proof.bad_metadata",
                    behavior_metadata=(
                        ("contractVersion", 2),
                        ("order", 0),
                        ("requiresUncond", False),
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="guidance contractVersion"):
        GuidanceRegistrySnapshot(
            (
                KeyedContribution(
                    "inference.guidance.condition-evaluation",
                    "proof.bad_bool_version",
                    behavior_metadata=(
                        ("contractVersion", True),
                        ("order", 0),
                        ("requiresUncond", False),
                    ),
                ),
            )
        )
