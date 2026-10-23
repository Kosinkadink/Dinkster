from __future__ import annotations

import sqlite3
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from dinkster_assets.identity import AssetError
from dinkster_assets.model import AssetRef
from dinkster_assets.resolution import (
    RESOLUTION_SCHEMA_VERSION,
    AdvisoryAlias,
    Artifact,
    DeclaredRequirements,
    LocalMaterialization,
    LogicalModel,
    ModelVariant,
    MountMaterialization,
    ProviderMirror,
    ResolutionSnapshot,
    ResolutionStore,
    SourceIdentity,
    VariantArtifact,
)

DIGEST_A = "blake3:" + "a" * 64
DIGEST_B = "blake3:" + "b" * 64
KIND = "model/diffusion"
SCOPE = "workspace-a"


def logical(logical_id: str = "flux/dev") -> LogicalModel:
    return LogicalModel(logical_id, "flux", KIND, 1)


def variant(
    variant_id: str = "fp8-main", *, requirements: DeclaredRequirements | None = None
) -> ModelVariant:
    return ModelVariant(
        "flux/dev",
        variant_id,
        "float8-e4m3fn",
        "fp8",
        "safetensors",
        "diffusion",
        requirements or DeclaredRequirements(),
        2,
    )


def ref(digest: str = DIGEST_A, *, virtual_path: str = "models/model.safetensors") -> AssetRef:
    return AssetRef(digest, "model.safetensors", 123, "application/octet-stream", virtual_path)


def mount(
    scope: str = SCOPE, *, mount_id: str = "models", priority: int = 0, digest: str = DIGEST_A
) -> MountMaterialization:
    return MountMaterialization(
        scope,
        mount_id,
        priority,
        KIND,
        ref(digest, virtual_path=f"mounts/{mount_id}/model.safetensors"),
    )


def seed(store: ResolutionStore) -> None:
    store.upsert_logical_model(logical())
    store.upsert_variant(variant())
    store.upsert_artifact(Artifact(DIGEST_A))
    store.link_variant_artifact(VariantArtifact("flux/dev", "fp8-main", DIGEST_A))


def test_catalog_models_normalize_freeze_and_validate_asset_refs() -> None:
    model = LogicalModel(" Flux/DEV ", " FLUX ", KIND, 1)
    row = variant()
    assert (model.logical_id, model.family, row.variant_id) == ("flux/dev", "flux", "fp8-main")
    with pytest.raises(FrozenInstanceError):
        row.variant_id = "changed"  # type: ignore[misc]
    with pytest.raises(AssetError, match="virtual path must match"):
        MountMaterialization(SCOPE, "models", 0, KIND, ref(DIGEST_B, virtual_path="models/x"))


def test_snapshot_rejects_duplicate_catalog_facts() -> None:
    row = variant()
    with pytest.raises(AssetError, match="duplicate variant identity"):
        ResolutionSnapshot(variants=(row, row))
    with pytest.raises(AssetError, match="invalid row"):
        ResolutionSnapshot(artifacts=(DIGEST_A,))  # type: ignore[arg-type]


def test_store_snapshot_preserves_catalog_provider_and_local_facts(tmp_path: Path) -> None:
    store = ResolutionStore(tmp_path / "resolution.sqlite")
    seed(store)
    store.add_alias(AdvisoryAlias("logical", "Flux Dev", "flux/dev"))
    mirror = ProviderMirror(SourceIdentity("provider-a", "source-a"), DIGEST_A, "available")
    store.upsert_mirror(mirror)
    local = LocalMaterialization(SCOPE, ref(), 3, 4)
    store.upsert_local_materialization(local)

    facts = store.snapshot(SCOPE)
    assert facts.logical_models == (logical(),)
    assert facts.variants == (variant(),)
    assert facts.artifacts == (Artifact(DIGEST_A),)
    assert facts.links == (VariantArtifact("flux/dev", "fp8-main", DIGEST_A),)
    assert facts.aliases == (AdvisoryAlias("logical", "Flux Dev", "flux/dev"),)
    assert facts.mirrors == (mirror,)
    assert facts.locals == (local,)
    store.close()


def test_mount_snapshot_is_scope_isolated_ordered_and_atomic(tmp_path: Path) -> None:
    store = ResolutionStore(tmp_path / "resolution.sqlite")
    rows = (
        mount(priority=2),
        mount(mount_id="preferred", priority=-1),
        mount("workspace-b", priority=2),
        mount("workspace-b", mount_id="preferred", priority=-1),
    )
    store.replace_mount_snapshot((SCOPE, "workspace-b"), rows)
    assert store.snapshot(SCOPE).mounts == (rows[1], rows[0])
    assert store.snapshot("workspace-b").mounts == (rows[3], rows[2])

    with pytest.raises(AssetError, match="duplicate materialization"):
        store.replace_mount_snapshot((SCOPE, "workspace-b"), (*rows, rows[0]))
    assert store.snapshot(SCOPE).mounts == (rows[1], rows[0])
    store.close()


def test_provider_snapshot_replaces_only_one_provider(tmp_path: Path) -> None:
    store = ResolutionStore(tmp_path / "resolution.sqlite")
    original = ProviderMirror(SourceIdentity("provider-a", "source-a"), DIGEST_A, "available")
    other = ProviderMirror(SourceIdentity("provider-b", "source-b"), DIGEST_B, "available")
    store.upsert_mirror(original)
    store.upsert_mirror(other)
    replacement = ProviderMirror(
        SourceIdentity("provider-a", "source-c"), DIGEST_B, "unavailable", "offline"
    )
    store.replace_provider_snapshot("provider-a", (replacement,))
    assert store.snapshot(SCOPE).mirrors == (replacement, other)
    store.close()


def test_schema_version_and_legacy_mapping_table_are_compatible(tmp_path: Path) -> None:
    path = tmp_path / "resolution.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version = 1")
    connection.execute(
        "CREATE TABLE mappings (scope TEXT, reference_key TEXT, logical_id TEXT, variant_id TEXT, "
        "digest TEXT, trusted_provider_id TEXT, trusted_source_id TEXT, updated_at REAL)"
    )
    connection.execute(
        "INSERT INTO mappings VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)",
        (SCOPE, "workflow:model", "flux/dev", "fp8-main", DIGEST_A, 1),
    )
    connection.commit()
    connection.close()

    store = ResolutionStore(path)
    assert store._schema_version() == RESOLUTION_SCHEMA_VERSION
    assert store.snapshot(SCOPE) == ResolutionSnapshot()
    store.close()
