from __future__ import annotations

from pathlib import Path

import pytest
from dinkster_assets.identity import AssetError
from dinkster_assets.model import AssetRef
from dinkster_assets.resolution import (
    MountMaterialization,
    ProviderMirror,
    ResolutionStore,
    SourceIdentity,
)

DIGEST_A = "blake3:" + "a" * 64
DIGEST_B = "blake3:" + "b" * 64
KIND = "model/diffusion"
SCOPE = "workspace-a"


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
