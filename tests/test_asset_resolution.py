from __future__ import annotations

import concurrent.futures
import socket
import sqlite3
from dataclasses import FrozenInstanceError
from pathlib import Path

import dinkster_assets.acquire as acquisition
import dinkster_assets.identity as identity
import pytest
from dinkster_assets.identity import AssetError
from dinkster_assets.model import AssetRef
from dinkster_assets.resolution import (
    RESOLUTION_SCHEMA_VERSION,
    AdvisoryAlias,
    Artifact,
    CompatibilityContext,
    DeclaredRequirements,
    LocalMaterialization,
    LogicalModel,
    ModelVariant,
    MountMaterialization,
    ProviderMirror,
    ReferenceMapping,
    ResolutionRequest,
    ResolutionSnapshot,
    ResolutionStore,
    SourceIdentity,
    VariantArtifact,
    resolve,
)

DIGEST_A = "blake3:" + "a" * 64
DIGEST_B = "blake3:" + "b" * 64
DIGEST_C = "blake3:" + "c" * 64
KIND = "model/diffusion"
SCOPE = "workspace-a"


def logical(logical_id: str = "flux/dev", *, kind: str = KIND) -> LogicalModel:
    return LogicalModel(logical_id, "flux", kind, 1)


def variant(
    variant_id: str = "fp8-main",
    *,
    logical_id: str = "flux/dev",
    dtype: str = "float8-e4m3fn",
    quantization: str = "fp8",
    requirements: DeclaredRequirements | None = None,
    updated_at: float = 2,
) -> ModelVariant:
    return ModelVariant(
        logical_id,
        variant_id,
        dtype,
        quantization,
        "safetensors",
        "diffusion",
        requirements or DeclaredRequirements(),
        updated_at,
    )


def ref(
    digest: str = DIGEST_A,
    *,
    name: str = "model.safetensors",
    virtual_path: str = "models/model.safetensors",
) -> AssetRef:
    return AssetRef(digest, name, 123, "application/octet-stream", virtual_path)


def local(digest: str = DIGEST_A, **kwargs: str) -> LocalMaterialization:
    return LocalMaterialization(SCOPE, ref(digest, **kwargs), 3, 4)


def mirror(
    digest: str = DIGEST_A,
    *,
    provider: str = "provider-a",
    source: str = "source-a",
    state: str = "available",
    reason: str = "",
    metadata: dict[str, object] | None = None,
) -> ProviderMirror:
    return ProviderMirror(
        SourceIdentity(provider, source),
        digest,
        state,  # type: ignore[arg-type]
        reason,
        metadata or {},
        5,
        6,
    )


def request(**changes: object) -> ResolutionRequest:
    values: dict[str, object] = {
        "scope": SCOPE,
        "reference_key": "workflow:model",
        "asset_kind": KIND,
        "authorized_providers": frozenset(("provider-a", "provider-b")),
        "compatibility": CompatibilityContext(),
    }
    values.update(changes)
    return ResolutionRequest(**values)  # type: ignore[arg-type]


def snapshot(
    *,
    variants: tuple[ModelVariant, ...] | None = None,
    links: tuple[VariantArtifact, ...] | None = None,
    mirrors: tuple[ProviderMirror, ...] = (),
    locals: tuple[LocalMaterialization, ...] = (),
    mounts: tuple[MountMaterialization, ...] = (),
    aliases: tuple[AdvisoryAlias, ...] = (),
) -> ResolutionSnapshot:
    rows = variants or (variant(),)
    bindings = links or (VariantArtifact("flux/dev", "fp8-main", DIGEST_A),)
    digests = {link.digest for link in bindings}
    return ResolutionSnapshot(
        logical_models=(logical(),),
        variants=rows,
        artifacts=tuple(Artifact(digest) for digest in sorted(digests)),
        links=bindings,
        aliases=aliases,
        mirrors=mirrors,
        locals=locals,
        mounts=mounts,
    )


def seed_store(store: ResolutionStore) -> None:
    store.upsert_logical_model(logical())
    store.upsert_variant(variant())
    store.link_variant_artifact(VariantArtifact("flux/dev", "fp8-main", DIGEST_A))


def mount_local(
    scope: str = SCOPE,
    *,
    mount_id: str = "models",
    priority: int = 0,
    kind: str = KIND,
    digest: str = DIGEST_A,
    name: str = "model.safetensors",
    virtual_path: str = "mounts/models/model.safetensors",
) -> MountMaterialization:
    return MountMaterialization(
        scope,
        mount_id,
        priority,
        kind,
        AssetRef(digest, name, 123, "application/octet-stream", virtual_path),
    )


def test_mount_materialization_snapshot_is_frozen_complete_and_scope_isolated(
    tmp_path: Path,
) -> None:
    with pytest.raises(AssetError, match="asset kind must be a string"):
        mount_local(kind=None)  # type: ignore[arg-type]
    with pytest.raises(AssetError, match="virtual path must match"):
        mount_local(virtual_path="mounts/other/model.safetensors")

    store = ResolutionStore(tmp_path / "resolution.sqlite")
    rows = (
        mount_local(priority=2, virtual_path="mounts/models/z.safetensors"),
        mount_local(
            mount_id="preferred",
            priority=-1,
            virtual_path="mounts/preferred/a.safetensors",
        ),
        mount_local(
            "workspace-b",
            priority=2,
            virtual_path="mounts/models/z.safetensors",
        ),
        mount_local(
            "workspace-b",
            mount_id="preferred",
            priority=-1,
            virtual_path="mounts/preferred/a.safetensors",
        ),
    )
    store.replace_mount_snapshot((SCOPE, "workspace-b"), rows)
    assert store.snapshot(SCOPE).mounts == (rows[1], rows[0])
    assert store.snapshot("workspace-b").mounts == (rows[3], rows[2])
    store.close()


def test_mount_snapshot_validates_before_delete_rolls_back_and_prunes_stale(
    tmp_path: Path,
) -> None:
    store = ResolutionStore(tmp_path / "resolution.sqlite")
    normalized = local()
    store.upsert_local_materialization(normalized)
    original = (
        mount_local(),
        mount_local("workspace-b"),
    )
    store.replace_mount_snapshot((SCOPE, "workspace-b"), original)

    conflict = mount_local(digest=DIGEST_B)
    for invalid, message in (
        ((original[0], original[0], original[1]), "duplicate materialization"),
        ((original[0], conflict, original[1]), "duplicate materialization"),
        ((original[0],), "cover every configured scope"),
    ):
        with pytest.raises(AssetError, match=message):
            store.replace_mount_snapshot((SCOPE, "workspace-b"), invalid)
        assert store.snapshot(SCOPE).mounts == (original[0],)
    assert store.snapshot(SCOPE).locals == (normalized,)

    replacement = (
        mount_local(
            mount_id="a",
            priority=-2,
            virtual_path="mounts/a/canonical.safetensors",
        ),
        mount_local(
            mount_id="b",
            priority=4,
            virtual_path="mounts/b/duplicate.safetensors",
        ),
        mount_local(
            "workspace-b",
            mount_id="a",
            priority=-2,
            virtual_path="mounts/a/canonical.safetensors",
        ),
        mount_local(
            "workspace-b",
            mount_id="b",
            priority=4,
            virtual_path="mounts/b/duplicate.safetensors",
        ),
    )
    store.replace_mount_snapshot((SCOPE, "workspace-b"), replacement)
    assert store.snapshot(SCOPE).mounts == replacement[:2]
    assert store.snapshot(SCOPE).locals == (normalized,)
    store.close()


def test_linked_resolution_uses_only_same_kind_canonical_mount_fallback() -> None:
    foreign_scope = mount_local(
        "workspace-b",
        mount_id="foreign",
        priority=-20,
        virtual_path="mounts/foreign/model.safetensors",
    )
    wrong_kind = mount_local(
        mount_id="wrong",
        priority=-10,
        kind="model/lora",
        virtual_path="mounts/wrong/model.safetensors",
    )
    canonical = mount_local(
        mount_id="a-models",
        priority=0,
        virtual_path="mounts/a-models/model.safetensors",
    )
    later = mount_local(
        mount_id="z-models",
        priority=0,
        virtual_path="mounts/z-models/model.safetensors",
    )
    outcome = resolve(
        request(expected_digest=DIGEST_A),
        None,
        snapshot(mounts=(foreign_scope, wrong_kind, later, canonical)),
    )
    assert outcome.status == "resolved"
    assert outcome.selected is not None
    assert outcome.selected.ref == canonical.ref


def test_models_normalize_freeze_and_require_complete_exact_asset_ref() -> None:
    model = LogicalModel(" Flux/DEV ", " FLUX ", KIND, 1)
    row = ModelVariant(
        model.logical_id,
        " FP8-MAIN ",
        "FLOAT8-E4M3FN",
        "FP8",
        "SAFETENSORS",
        "DIFFUSION",
        DeclaredRequirements(hardware=frozenset(("CUDA",))),
        2,
    )
    assert (model.logical_id, model.family, row.variant_id) == (
        "flux/dev",
        "flux",
        "fp8-main",
    )
    with pytest.raises(FrozenInstanceError):
        row.variant_id = "changed"  # type: ignore[misc]
    with pytest.raises(AssetError, match="integer"):
        LocalMaterialization(SCOPE, AssetRef(DIGEST_A, "x", True, virtual_path="x"))
    with pytest.raises(AssetError, match="ref name"):
        LocalMaterialization(SCOPE, AssetRef(DIGEST_A, "", 1, virtual_path="x"))
    with pytest.raises(AssetError, match="media type"):
        LocalMaterialization(SCOPE, AssetRef(DIGEST_A, "x", 1, media_type="", virtual_path="x"))
    with pytest.raises(AssetError, match="virtual path"):
        LocalMaterialization(SCOPE, AssetRef(DIGEST_A, "x", 1, virtual_path=""))
    with pytest.raises(AssetError, match="resolver"):
        LocalMaterialization(
            SCOPE,
            AssetRef(
                DIGEST_A,
                "x",
                1,
                virtual_path="x",
                resolver=lambda _digest: None,  # type: ignore[arg-type]
            ),
        )
    stored = local()
    assert stored.ref.to_wire() == {
        "digest": DIGEST_A,
        "name": "model.safetensors",
        "size": 123,
        "mediaType": "application/octet-stream",
        "virtualPath": "models/model.safetensors",
    }


def test_compatibility_enrichment_does_not_churn_stable_variant_identity() -> None:
    before = variant(requirements=DeclaredRequirements())
    after = variant(
        requirements=DeclaredRequirements(
            loaders=frozenset(("native-flux",)),
            runtimes=frozenset(("torch",)),
            hardware=frozenset(("cuda",)),
        ),
        updated_at=9,
    )
    assert (before.logical_id, before.variant_id) == (after.logical_id, after.variant_id)
    assert before.requirements != after.requirements


def test_distinct_fp8_nvfp4_variants_and_no_implicit_sibling_preference() -> None:
    fp8 = variant()
    nvfp4 = variant(
        "nvfp4-main",
        dtype="float4-e2m1fn",
        quantization="nvfp4",
        requirements=DeclaredRequirements(hardware=frozenset(("blackwell",))),
    )
    facts = snapshot(
        variants=(fp8, nvfp4),
        links=(
            VariantArtifact("flux/dev", "fp8-main", DIGEST_A),
            VariantArtifact("flux/dev", "nvfp4-main", DIGEST_B),
        ),
        locals=(local(DIGEST_A), local(DIGEST_B, virtual_path="models/nvfp4.safetensors")),
    )
    implicit = resolve(request(logical_id="flux/dev", compatibility=None), None, facts)
    assert implicit.status == "ambiguous"
    assert {candidate.variant_id for candidate in implicit.candidates} == {
        "fp8-main",
        "nvfp4-main",
    }
    explicit = resolve(request(logical_id="flux/dev", variant_id="fp8-main"), None, facts)
    assert explicit.status == "resolved"
    assert explicit.selected is not None and explicit.selected.digest == DIGEST_A


def test_explicit_compatibility_may_leave_exactly_one_variant() -> None:
    fp8 = variant(requirements=DeclaredRequirements(hardware=frozenset(("cuda",))))
    nvfp4 = variant(
        "nvfp4-main",
        dtype="float4-e2m1fn",
        quantization="nvfp4",
        requirements=DeclaredRequirements(hardware=frozenset(("blackwell",))),
    )
    facts = snapshot(
        variants=(fp8, nvfp4),
        links=(
            VariantArtifact("flux/dev", "fp8-main", DIGEST_A),
            VariantArtifact("flux/dev", "nvfp4-main", DIGEST_B),
        ),
        locals=(local(DIGEST_A), local(DIGEST_B, virtual_path="models/nvfp4.safetensors")),
    )
    outcome = resolve(
        request(
            logical_id="flux/dev",
            compatibility=CompatibilityContext(hardware=frozenset(("cuda",))),
        ),
        None,
        facts,
    )
    assert outcome.status == "resolved"
    assert outcome.tier == "compatibility"
    assert outcome.selected is not None and outcome.selected.variant_id == "fp8-main"
    assert {candidate.variant_id for candidate in outcome.candidates} == {
        "fp8-main",
        "nvfp4-main",
    }


def test_exact_triple_mapping_and_wrong_kind_refuse_without_fallthrough() -> None:
    mapping = ReferenceMapping(SCOPE, "workflow:model", "flux/dev", "fp8-main", DIGEST_A)
    facts = snapshot(locals=(local(),))
    outcome = resolve(request(expected_digest=DIGEST_A), mapping, facts)
    assert outcome.status == "resolved" and outcome.tier == "mapping"
    assert outcome.selected is not None
    assert (
        outcome.selected.logical_id,
        outcome.selected.variant_id,
        outcome.selected.digest,
    ) == ("flux/dev", "fp8-main", DIGEST_A)
    wrong_kind = resolve(request(asset_kind="model/lora"), mapping, facts)
    assert wrong_kind.status == "incompatible"
    assert wrong_kind.reason == "mapping kind mismatch"
    no_artifacts = ResolutionSnapshot(logical_models=(logical(),))
    selected_wrong_kind = resolve(
        request(logical_id="flux/dev", asset_kind="model/lora"), None, no_artifacts
    )
    assert selected_wrong_kind.status == "incompatible"
    assert selected_wrong_kind.reason == "logical model kind mismatch"


def test_trusted_source_requires_authorized_exact_observation() -> None:
    source = SourceIdentity("provider-a", "source-a")
    facts = snapshot(mirrors=(mirror(DIGEST_A),), locals=(local(),))
    trusted = request(trusted_source=source, trusted_digest=DIGEST_A)
    assert resolve(trusted, None, facts).status == "resolved"
    absent = resolve(trusted, None, snapshot(locals=(local(),)))
    assert absent.status == "incompatible"
    assert absent.reason == "trusted source unavailable"
    unauthorized = resolve(
        request(
            authorized_providers=frozenset(("provider-b",)),
            trusted_source=source,
            trusted_digest=DIGEST_A,
        ),
        None,
        facts,
    )
    assert unauthorized.status == "incompatible"
    assert unauthorized.reason == "trusted provider unauthorized"
    offline = snapshot(
        mirrors=(mirror(DIGEST_A, state="unavailable", reason="provider offline"),),
        locals=(local(),),
    )
    unavailable = resolve(trusted, None, offline)
    assert unavailable.status == "incompatible"
    assert unavailable.reason == "trusted source unavailable"
    mapped = ReferenceMapping(SCOPE, "workflow:model", "flux/dev", "fp8-main", DIGEST_A, source)
    mapped_unavailable = resolve(request(), mapped, offline)
    assert mapped_unavailable.status == "incompatible"
    assert mapped_unavailable.reason == "trusted source unavailable"


def test_availability_and_compatibility_are_orthogonal_with_bounded_reasons() -> None:
    ready_variant = variant()
    restricted_variant = variant(
        "nvfp4-main",
        dtype="float4-e2m1fn",
        quantization="nvfp4",
        requirements=DeclaredRequirements(hardware=frozenset(("blackwell",))),
    )
    facts = snapshot(
        variants=(ready_variant, restricted_variant),
        links=(
            VariantArtifact("flux/dev", "fp8-main", DIGEST_A),
            VariantArtifact("flux/dev", "fp8-main", DIGEST_B),
            VariantArtifact("flux/dev", "fp8-main", DIGEST_C),
            VariantArtifact("flux/dev", "nvfp4-main", DIGEST_B),
        ),
        mirrors=(
            mirror(DIGEST_B),
            mirror(DIGEST_C, source="offline", state="unavailable", reason="provider offline"),
        ),
        locals=(local(DIGEST_A),),
    )
    context = CompatibilityContext(hardware=frozenset(("cuda",)))
    outcome = resolve(request(compatibility=context), None, facts)
    states = {candidate.availability for candidate in outcome.candidates}
    assert states == {"local", "downloadable", "unavailable"}
    incompatible = resolve(
        request(
            logical_id="flux/dev",
            variant_id="nvfp4-main",
            compatibility=context,
        ),
        None,
        facts,
    )
    assert incompatible.status == "incompatible"
    assert incompatible.selected is not None
    assert incompatible.selected.availability == "downloadable"
    assert incompatible.selected.compatibility == "incompatible"
    assert "hardware:blackwell" in incompatible.reason
    assert all(
        len(candidate.availability_reason) <= 256 and len(candidate.compatibility_reason) <= 256
        for candidate in outcome.candidates
    )

    long_requirement = "x" * 300
    long_variant = variant(
        requirements=DeclaredRequirements(hardware=frozenset((long_requirement,)))
    )
    long_outcome = resolve(
        request(
            logical_id="flux/dev",
            variant_id="fp8-main",
            compatibility=CompatibilityContext(),
        ),
        None,
        snapshot(variants=(long_variant,), locals=(local(),)),
    )
    assert long_outcome.status == "incompatible"
    assert len(long_outcome.reason) <= 256
    assert long_outcome.reason.startswith("missing compatibility requirements:")


def test_installed_incompatible_retains_local_ref_and_separate_reasons() -> None:
    restricted = variant(requirements=DeclaredRequirements(hardware=frozenset(("blackwell",))))
    outcome = resolve(
        request(
            logical_id="flux/dev",
            variant_id="fp8-main",
            compatibility=CompatibilityContext(hardware=frozenset(("cuda",))),
        ),
        None,
        snapshot(variants=(restricted,), locals=(local(),)),
    )
    assert outcome.status == "incompatible"
    assert outcome.selected is not None
    assert outcome.selected.availability == "local"
    assert outcome.selected.availability_reason == "local materialization"
    assert outcome.selected.compatibility == "incompatible"
    assert outcome.selected.compatibility_reason == "hardware:blackwell"
    assert outcome.selected.ref == ref()


def test_unknown_compatibility_keeps_candidate_visible_but_cannot_bind() -> None:
    facts = snapshot(locals=(local(),))
    outcome = resolve(request(expected_digest=DIGEST_A, compatibility=None), None, facts)
    assert outcome.status == "missing"
    assert outcome.tier == "expected_digest"
    assert outcome.selected is not None
    assert outcome.selected.availability == "local"
    assert outcome.selected.compatibility == "unknown"
    assert outcome.selected.compatibility_reason == "compatibility context required"
    assert outcome.selected.ref == ref()


def test_mapping_cannot_bind_an_installed_incompatible_candidate() -> None:
    restricted = variant(requirements=DeclaredRequirements(hardware=frozenset(("blackwell",))))
    mapping = ReferenceMapping(SCOPE, "workflow:model", "flux/dev", "fp8-main", DIGEST_A)
    outcome = resolve(
        request(compatibility=CompatibilityContext(hardware=frozenset(("cuda",)))),
        mapping,
        snapshot(variants=(restricted,), locals=(local(),)),
    )
    assert outcome.status == "incompatible"
    assert outcome.tier == "mapping"
    assert outcome.selected is not None
    assert outcome.selected.availability == "local"
    assert outcome.selected.compatibility == "incompatible"
    assert outcome.selected.ref == ref()


def test_multiple_compatible_siblings_remain_visible_and_ambiguous() -> None:
    sibling = variant(
        "fp16-main",
        dtype="float16",
        quantization="none",
        requirements=DeclaredRequirements(hardware=frozenset(("cuda",))),
    )
    incompatible_sibling = variant(
        "nvfp4-main",
        dtype="float4-e2m1fn",
        quantization="nvfp4",
        requirements=DeclaredRequirements(hardware=frozenset(("blackwell",))),
    )
    facts = snapshot(
        variants=(variant(), sibling, incompatible_sibling),
        links=(
            VariantArtifact("flux/dev", "fp8-main", DIGEST_A),
            VariantArtifact("flux/dev", "fp16-main", DIGEST_B),
            VariantArtifact("flux/dev", "nvfp4-main", DIGEST_C),
        ),
        locals=(
            local(DIGEST_A),
            local(DIGEST_B, virtual_path="models/fp16.safetensors"),
            local(DIGEST_C, virtual_path="models/nvfp4.safetensors"),
        ),
    )
    outcome = resolve(
        request(
            logical_id="flux/dev",
            compatibility=CompatibilityContext(hardware=frozenset(("cuda",))),
        ),
        None,
        facts,
    )
    assert outcome.status == "ambiguous"
    assert outcome.tier == "compatibility"
    assert outcome.selected is None
    assert {candidate.variant_id for candidate in outcome.candidates} == {
        "fp8-main",
        "fp16-main",
        "nvfp4-main",
    }
    assert {candidate.compatibility for candidate in outcome.candidates} == {
        "compatible",
        "incompatible",
    }


def test_authority_selects_one_compatible_shared_digest_and_keeps_sibling_visible() -> None:
    incompatible_sibling = variant(
        "nvfp4-main",
        dtype="float4-e2m1fn",
        quantization="nvfp4",
        requirements=DeclaredRequirements(hardware=frozenset(("blackwell",))),
    )
    facts = snapshot(
        variants=(variant(), incompatible_sibling),
        links=(
            VariantArtifact("flux/dev", "fp8-main", DIGEST_A),
            VariantArtifact("flux/dev", "nvfp4-main", DIGEST_A),
        ),
        mirrors=(mirror(DIGEST_A),),
        locals=(local(),),
    )
    context = CompatibilityContext(hardware=frozenset(("cuda",)))
    expected = resolve(request(expected_digest=DIGEST_A, compatibility=context), None, facts)
    assert expected.status == "resolved"
    assert expected.selected is not None and expected.selected.variant_id == "fp8-main"
    assert len(expected.candidates) == 2
    trusted = resolve(
        request(
            trusted_source=SourceIdentity("provider-a", "source-a"),
            trusted_digest=DIGEST_A,
            compatibility=context,
        ),
        None,
        facts,
    )
    assert trusted.status == "resolved"
    assert trusted.selected is not None and trusted.selected.variant_id == "fp8-main"
    assert len(trusted.candidates) == 2


def test_snapshot_freezes_sequences_and_rejects_duplicate_identities() -> None:
    mutable_models = [logical()]
    frozen = ResolutionSnapshot(logical_models=mutable_models)  # type: ignore[arg-type]
    mutable_models.clear()
    assert frozen.logical_models == (logical(),)

    with pytest.raises(AssetError, match="duplicate logical model"):
        ResolutionSnapshot(logical_models=(logical(), logical()))
    with pytest.raises(AssetError, match="duplicate variant"):
        ResolutionSnapshot(variants=(variant(), variant()))
    with pytest.raises(AssetError, match="duplicate variant artifact"):
        link = VariantArtifact("flux/dev", "fp8-main", DIGEST_A)
        ResolutionSnapshot(links=(link, link))
    with pytest.raises(AssetError, match="duplicate provider source"):
        ResolutionSnapshot(mirrors=(mirror(DIGEST_A), mirror(DIGEST_B)))
    with pytest.raises(AssetError, match="duplicate local materialization"):
        ResolutionSnapshot(locals=(local(name="first"), local(name="second", virtual_path="other")))


def test_same_digest_mirrors_and_renamed_local_merge_without_duplicate_artifacts(
    tmp_path: Path,
) -> None:
    store = ResolutionStore(tmp_path / "resolution.sqlite")
    seed_store(store)
    store.upsert_mirror(mirror(provider="provider-a", source="one"))
    store.upsert_mirror(mirror(provider="provider-b", source="two"))
    store.upsert_local_materialization(local(name="old.safetensors", virtual_path="old/model"))
    renamed = local(name="new.safetensors", virtual_path="new/model")
    store.upsert_local_materialization(renamed)
    result = store.snapshot(SCOPE)
    assert result.artifacts == (Artifact(DIGEST_A),)
    assert len(result.mirrors) == 2
    assert result.locals == (renamed,)
    store.close()


def test_same_basename_different_digest_and_hint_metadata_stay_ambiguous() -> None:
    facts = snapshot(
        links=(
            VariantArtifact("flux/dev", "fp8-main", DIGEST_A),
            VariantArtifact("flux/dev", "fp8-main", DIGEST_B),
        ),
        mirrors=(
            mirror(DIGEST_A, source="a", metadata={"displayName": "winner"}),
            mirror(
                DIGEST_B,
                source="b",
                metadata={
                    "source": "a",
                    "reference": "same.safetensors",
                    "loaderPath": "models/same.safetensors",
                    "modelType": "preferred",
                    "displayName": "preferred",
                },
            ),
        ),
        locals=(
            local(DIGEST_A, name="same.safetensors", virtual_path="a/same.safetensors"),
            local(DIGEST_B, name="same.safetensors", virtual_path="b/same.safetensors"),
        ),
    )
    forward = resolve(request(logical_id="flux/dev", variant_id="fp8-main"), None, facts)
    reverse = resolve(
        request(logical_id="flux/dev", variant_id="fp8-main"),
        None,
        ResolutionSnapshot(
            facts.logical_models,
            facts.variants,
            facts.artifacts,
            tuple(reversed(facts.links)),
            facts.aliases,
            tuple(reversed(facts.mirrors)),
            tuple(reversed(facts.locals)),
        ),
    )
    assert forward.status == reverse.status == "ambiguous"
    assert [(row.variant_id, row.digest) for row in forward.candidates] == [
        (row.variant_id, row.digest) for row in reverse.candidates
    ]


def test_ambiguous_alias_never_becomes_identity_authority() -> None:
    other = LogicalModel("flux/schnell", "flux", KIND, 1)
    facts = snapshot(
        aliases=(
            AdvisoryAlias("logical", "flux-model.safetensors", "flux/dev"),
            AdvisoryAlias("logical", "flux-model.safetensors", "flux/schnell"),
        )
    )
    facts = ResolutionSnapshot(
        facts.logical_models + (other,),
        facts.variants,
        facts.artifacts,
        facts.links,
        facts.aliases,
        facts.mirrors,
        facts.locals,
    )
    outcome = resolve(request(logical_id="flux-model.safetensors"), None, facts)
    assert outcome.status == "ambiguous"
    assert outcome.reason == "logical alias ambiguous"


def test_provider_refresh_enriches_and_drops_only_its_observations(tmp_path: Path) -> None:
    store = ResolutionStore(tmp_path / "resolution.sqlite")
    seed_store(store)
    first = mirror(metadata={"license": "old"})
    other = mirror(provider="provider-b", source="mirror-b")
    store.upsert_mirror(first)
    store.upsert_mirror(other)
    enriched = mirror(metadata={"license": "new", "family": "flux"})
    store.replace_provider_snapshot("provider-a", (enriched,))
    snap = store.snapshot(SCOPE)
    assert snap.mirrors == (enriched, other)
    assert snap.artifacts == (Artifact(DIGEST_A),)
    store.replace_provider_snapshot("provider-a", ())
    snap = store.snapshot(SCOPE)
    assert snap.mirrors == (other,)
    assert snap.artifacts == (Artifact(DIGEST_A),)
    assert snap.links == (VariantArtifact("flux/dev", "fp8-main", DIGEST_A),)
    store.close()


def test_provider_snapshot_duplicate_and_invalid_rows_roll_back(tmp_path: Path) -> None:
    store = ResolutionStore(tmp_path / "resolution.sqlite")
    original = mirror()
    store.upsert_mirror(original)
    duplicate = mirror(source="replacement")
    with pytest.raises(sqlite3.IntegrityError):
        store.replace_provider_snapshot("provider-a", (duplicate, duplicate))
    assert store.snapshot(SCOPE).mirrors == (original,)
    with pytest.raises(AssetError, match="must match"):
        store.replace_provider_snapshot("provider-a", (mirror(provider="provider-b"),))
    assert store.snapshot(SCOPE).mirrors == (original,)
    store.close()


def test_store_reopen_schema_version_mapping_and_requirement_enrichment(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resolution.sqlite"
    store = ResolutionStore(path)
    seed_store(store)
    enriched = variant(
        requirements=DeclaredRequirements(
            loaders=frozenset(("native-flux",)), hardware=frozenset(("cuda",))
        ),
        updated_at=9,
    )
    store.upsert_variant(enriched)
    mapping = ReferenceMapping(
        SCOPE,
        "workflow:model",
        "flux/dev",
        "fp8-main",
        DIGEST_A,
        SourceIdentity("provider-a", "source-a"),
        10,
    )
    store.upsert_mapping(mapping)
    replacement = ReferenceMapping(
        SCOPE, "workflow:model", "flux/dev", "fp8-main", DIGEST_B, updated_at=11
    )
    store.upsert_mapping(replacement)
    store.close()
    reopened = ResolutionStore(path)
    assert reopened.get_mapping(SCOPE, "workflow:model") == replacement
    assert reopened.get_mapping("workspace-b", "workflow:model") is None
    (stored_variant,) = reopened.snapshot(SCOPE).variants
    assert stored_variant == enriched
    reopened.close()
    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == RESOLUTION_SCHEMA_VERSION
    connection.execute(f"PRAGMA user_version = {RESOLUTION_SCHEMA_VERSION + 1}")
    connection.commit()
    connection.close()
    with pytest.raises(AssetError, match="schema version"):
        ResolutionStore(path)


def test_schema_one_migrates_transactionally_with_all_normalized_facts_preserved(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resolution.sqlite"
    store = ResolutionStore(path)
    seed_store(store)
    store.add_alias(AdvisoryAlias("logical", "Flux Dev", "flux/dev"))
    store.upsert_mirror(mirror())
    store.upsert_local_materialization(local())
    mapping = ReferenceMapping(SCOPE, "workflow:model", "flux/dev", "fp8-main", DIGEST_A)
    store.upsert_mapping(mapping)
    expected = store.snapshot(SCOPE)
    store.close()

    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE mount_materializations")
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    migrated = ResolutionStore(path)
    assert migrated.snapshot(SCOPE) == expected
    assert migrated.get_mapping(SCOPE, "workflow:model") == mapping
    assert sqlite3.connect(path).execute("PRAGMA user_version").fetchone()[0] == 2
    migrated.replace_mount_snapshot((SCOPE,), (mount_local(),))
    migrated.close()
    reopened = ResolutionStore(path)
    assert reopened.snapshot(SCOPE).mounts == (mount_local(),)
    reopened.close()


def test_store_refuses_logical_and_variant_identity_churn(tmp_path: Path) -> None:
    store = ResolutionStore(tmp_path / "resolution.sqlite")
    seed_store(store)
    with pytest.raises(AssetError, match="family/kind"):
        store.upsert_logical_model(LogicalModel("flux/dev", "sdxl", KIND, 3))
    with pytest.raises(AssetError, match="execution identity"):
        store.upsert_variant(variant(dtype="float16", quantization="none"))
    store.upsert_variant(
        variant(requirements=DeclaredRequirements(loaders=frozenset(("native-flux",))))
    )
    store.upsert_variant(variant(requirements=DeclaredRequirements(hardware=frozenset(("cuda",)))))
    (stored,) = store.snapshot(SCOPE).variants
    assert stored.requirements == DeclaredRequirements(
        loaders=frozenset(("native-flux",)), hardware=frozenset(("cuda",))
    )
    store.close()


def test_store_initialization_and_writes_are_thread_safe(tmp_path: Path) -> None:
    path = tmp_path / "resolution.sqlite"

    def open_and_close(_index: int) -> None:
        store = ResolutionStore(path)
        store.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        tuple(executor.map(open_and_close, range(4)))
    store = ResolutionStore(path)

    def write(index: int) -> None:
        store.upsert_mirror(mirror(source=f"source-{index}"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(write, range(32)))
    assert len(store.snapshot(SCOPE).mirrors) == 32
    store.close()


def test_cross_connection_identity_enrichment_and_snapshot_are_atomic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resolution.sqlite"
    first = ResolutionStore(path)
    second = ResolutionStore(path)
    first.upsert_logical_model(logical())
    first.upsert_variant(variant())

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(
                first.upsert_variant,
                variant(requirements=DeclaredRequirements(loaders=frozenset(("native-flux",)))),
            ),
            executor.submit(
                second.upsert_variant,
                variant(requirements=DeclaredRequirements(hardware=frozenset(("cuda",)))),
            ),
        )
        for future in futures:
            future.result()
    (stored,) = first.snapshot(SCOPE).variants
    assert stored.requirements == DeclaredRequirements(
        loaders=frozenset(("native-flux",)), hardware=frozenset(("cuda",))
    )

    def refresh(index: int) -> None:
        digest = "blake3:" + f"{index:064x}"
        first.replace_provider_snapshot("provider-a", (mirror(digest, source="current"),))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(lambda: [refresh(index) for index in range(1, 20)])
        snapshots = [second.snapshot(SCOPE) for _index in range(20)]
        writer.result()
    for snap in snapshots:
        artifact_digests = {artifact.digest for artifact in snap.artifacts}
        assert all(row.digest in artifact_digests for row in snap.mirrors)
    first.close()
    second.close()


def test_partial_version_zero_schema_recovers_idempotently(tmp_path: Path) -> None:
    path = tmp_path / "resolution.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE artifacts (digest TEXT PRIMARY KEY)")
    connection.commit()
    connection.close()
    store = ResolutionStore(path)
    seed_store(store)
    assert store.snapshot(SCOPE).links == (VariantArtifact("flux/dev", "fp8-main", DIGEST_A),)
    store.close()


def test_resolution_is_pure_and_performs_no_implicit_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "resolution.sqlite"
    store = ResolutionStore(path)
    seed_store(store)
    store.upsert_local_materialization(local())
    facts = store.snapshot(SCOPE)
    before = path.read_bytes()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("resolution attempted external IO")

    with monkeypatch.context() as guard:
        guard.setattr(AssetRef, "open", forbidden)
        guard.setattr(AssetRef, "read_bytes", forbidden)
        guard.setattr(AssetRef, "local_path", forbidden)
        guard.setattr(Path, "open", forbidden)
        guard.setattr(socket, "socket", forbidden)
        guard.setattr(identity, "digest_file", forbidden)
        guard.setattr(identity, "digest_bytes", forbidden)
        guard.setattr(acquisition, "acquire_need", forbidden)
        outcome = resolve(request(expected_digest=DIGEST_A), None, facts)
    assert outcome.status == "resolved"
    assert before == path.read_bytes()
    store.close()
