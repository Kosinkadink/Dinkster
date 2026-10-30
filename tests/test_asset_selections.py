"""Source-named selections use only live, exact-kind directory grants."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dinkster_api.v1 import (
    AssetError,
    AssetRef,
    LocalAssetRef,
    asset_selection_names,
    resolve_asset_selection,
)
from dinkster_assets import digest_bytes


def grants(tmp_path: Path, *rows: tuple[str, Path]) -> dict[str, str]:
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {
                        "id": f"grant-{i}",
                        "kind": kind,
                        "root": str(root),
                        "index": str(root / "missing-index.json"),
                        "priority": i,
                    }
                    for i, (kind, root) in enumerate(rows)
                ]
            }
        )
    )
    return {"DINKSTER_MOUNTS_SNAPSHOT": str(snapshot)}


def seed(root: Path, name: str, data: bytes = b"fixture") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("change", ("replace", "revoke"))
def test_indexed_selection_uses_existing_identity_and_live_authority(
    tmp_path: Path, change: str
) -> None:
    from dinkster_assets import MountDef, MountTable

    root = tmp_path / "vae"
    path = seed(root, "nested/tiny.safetensors", b"first-value")
    snapshot = tmp_path / "mounts.json"
    table = MountTable(snapshot)
    table.add(MountDef(id="vae", path=root), kind="model/vae")
    table.scan("vae")
    env = {"DINKSTER_MOUNTS_SNAPSHOT": str(snapshot)}
    selected = resolve_asset_selection("model/vae", "nested/tiny.safetensors", env=env)
    assert isinstance(selected, AssetRef)
    assert selected.digest == digest_bytes(b"first-value")
    assert selected.virtual_path == "mounts/vae/nested/tiny.safetensors"
    assert selected.read_bytes() == b"first-value"
    if change == "replace":
        stat = path.stat()
        path.unlink()
        path.write_bytes(b"other-value")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    else:
        table.remove("vae")
    with pytest.raises(AssetError):
        selected.read_bytes()


def test_directory_modes_preserve_source_discovery_and_ranked_resolution(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    seed(first, "model_index.json")
    seed(first, "vendor/model/model_index.json")
    seed(first, "dataset/metadata.json")
    seed(first, "dataset/nested/model_index.json")
    seed(first, "dataset/nested/metadata.json")
    seed(second, "vendor/model/model_index.json", b"later")
    seed(second, "shards/part.safetensors")
    seed(second, "not-dataset/part.SAFETENSORS")
    env = grants(tmp_path, ("model/diffusers", first), ("model/diffusers", second))
    assert asset_selection_names("model/diffusers", env=env) == (
        ".",
        "dataset/nested",
        "vendor/model",
    )
    assert resolve_asset_selection("model/diffusers", "vendor/model", env=env) == (
        first / "vendor/model"
    )
    assert resolve_asset_selection("model/diffusers", ".", env=env) == first
    env = grants(tmp_path, ("directory/dataset", first), ("directory/dataset", second))
    assert asset_selection_names("directory/dataset", env=env) == ("dataset", "shards")
    seed(first, "shards/shard_0.pkl")
    assert resolve_asset_selection("directory/dataset", "shards", env=env) == first / "shards"
    (first / "shards/shard_0.pkl").unlink()
    (first / "shards").rmdir()
    env = grants(tmp_path, ("directory/input", first))
    assert asset_selection_names("directory/input", env=env) == (
        "dataset",
        "dataset/nested",
        "vendor",
        "vendor/model",
    )
    with pytest.raises(AssetError):
        resolve_asset_selection("directory/input", ".", env=env)


def test_vae_pairs_stay_in_one_root_and_file_refs_revalidate_grants(tmp_path: Path) -> None:
    first, second, vaes = tmp_path / "first", tmp_path / "second", tmp_path / "vaes"
    seed(first, "taesd_encoder.safetensors", b"encoder")
    seed(second, "taesd_decoder.safetensors", b"decoder")
    seed(first, "taef2_encoder.safetensors")
    seed(first, "taef2_decoder.safetensors")
    seed(first, "taehv.safetensors")
    seed(first, "nested/taesdxl_encoder.safetensors")
    seed(first, "nested/taesdxl_decoder.safetensors")
    seed(vaes, "nested/real.safetensors")
    env = grants(
        tmp_path, ("model/vae-approx", first), ("model/vae-approx", second), ("model/vae", vaes)
    )
    assert asset_selection_names("model/vae", env=env) == (
        "nested/real.safetensors",
        "taehv.safetensors",
        "taef2",
        "pixel_space",
    )
    assert resolve_asset_selection("model/vae", "pixel_space", env=env) is None
    with pytest.raises(AssetError):
        resolve_asset_selection("model/vae", "taesd", env=env)
    seed(first, "taesd_decoder.safetensors", b"first-decoder")
    pair = resolve_asset_selection("model/vae", "taesd", env=env)
    assert isinstance(pair, tuple)
    assert all(isinstance(member, LocalAssetRef) for member in pair)
    assert [member.read_bytes() for member in pair] == [b"encoder", b"first-decoder"]
    file = resolve_asset_selection("model/vae", "nested/real.safetensors", env=env)
    assert isinstance(file, LocalAssetRef)
    assert file.local_path() == vaes / "nested/real.safetensors"
    grants(tmp_path, ("model/checkpoint", first))
    for ref in (*pair, file):
        with pytest.raises(AssetError):
            ref.local_path()
    assert asset_selection_names("model/vae", env=env) == ("pixel_space",)


def test_video_vae_names_follow_exact_source_category_routing(tmp_path: Path) -> None:
    ordinary, approx, fallback = tmp_path / "vae", tmp_path / "approx", tmp_path / "fallback"
    seed(ordinary, "taehv.safetensors", b"ordinary collision")
    seed(approx, "taehv.safetensors", b"preferred video")
    seed(fallback, "taehv.safetensors", b"fallback video")
    seed(approx, "taehv_custom.safetensors", b"prefix only")
    env = grants(
        tmp_path,
        ("model/vae", ordinary),
        ("model/vae-approx", approx),
        ("model/vae-approx", fallback),
    )
    selected = resolve_asset_selection("model/vae", "taehv.safetensors", env=env)
    assert isinstance(selected, LocalAssetRef)
    assert selected.read_bytes() == b"preferred video"
    assert "taehv_custom.safetensors" in asset_selection_names("model/vae", env=env)
    with pytest.raises(AssetError, match="not granted"):
        resolve_asset_selection("model/vae", "taehv_custom.safetensors", env=env)
    seed(ordinary, "taehv_custom.safetensors", b"ordinary prefix")
    selected = resolve_asset_selection("model/vae", "taehv_custom.safetensors", env=env)
    assert isinstance(selected, LocalAssetRef)
    assert selected.read_bytes() == b"ordinary prefix"
    grants(tmp_path, ("model/vae", ordinary))
    with pytest.raises(AssetError, match="not granted"):
        resolve_asset_selection("model/vae", "taehv.safetensors", env=env)


@pytest.mark.parametrize("name", ["../outside", "/outside", "C:/outside", "a\\b", "a/../b", "a//b"])
def test_unsafe_selection_names_never_open_files(tmp_path: Path, name: str) -> None:
    env = grants(tmp_path, ("model/diffusers", tmp_path))
    with pytest.raises(AssetError):
        resolve_asset_selection("model/diffusers", name, env=env)


def test_no_filesystem_or_library_fallback_and_changed_bytes_fail_closed(tmp_path: Path) -> None:
    path = seed(tmp_path / "vae", "real.safetensors", b"first")
    env = {"DINKSTER_ASSET_ROOT": str(path.parent), "DINKSTER_ASSET_VAULT": str(path.parent)}
    assert asset_selection_names("model/vae", env=env) == ("pixel_space",)
    assert asset_selection_names("directory/input", env=env) == ()
    with pytest.raises(AssetError):
        asset_selection_names("model/checkpoint", env=env)
    env = grants(tmp_path, ("model/vae", path.parent))
    ref = resolve_asset_selection("model/vae", path.name, env=env)
    assert isinstance(ref, LocalAssetRef)
    path.write_bytes(b"other")
    with pytest.raises(AssetError):
        ref.read_bytes()


@pytest.mark.skipif(os.name != "nt", reason="Windows native file sharing")
def test_local_selection_refuses_active_writers_and_excludes_writes_until_close(
    tmp_path: Path,
) -> None:
    path = seed(tmp_path / "vae", "real.safetensors", b"first")
    env = grants(tmp_path, ("model/vae", path.parent))
    selected = resolve_asset_selection("model/vae", path.name, env=env)
    assert isinstance(selected, LocalAssetRef)
    with path.open("r+b") as writer:
        stat = path.stat()
        for data in (b"other", b"third"):
            writer.seek(0)
            writer.write(data)
            writer.flush()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            with pytest.raises(AssetError):
                resolve_asset_selection("model/vae", path.name, env=env)
            with pytest.raises(AssetError):
                selected.read_bytes()
    current = resolve_asset_selection("model/vae", path.name, env=env)
    assert isinstance(current, LocalAssetRef)
    with current.open() as reader:
        with pytest.raises(OSError):
            with path.open("r+b"):
                pass
        assert reader.read() == b"third"
    path.write_bytes(b"after")
    with pytest.raises(AssetError):
        current.read_bytes()


@pytest.mark.skipif(os.name != "nt", reason="Windows native file revisions")
def test_same_stat_rewrite_invalidates_local_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_assets import model, mounts

    path = seed(tmp_path / "vae", "real.safetensors", b"first")
    stat = path.stat()
    signature = model.local_file_signature
    monkeypatch.setattr(
        model, "local_file_signature", lambda _stat, source: signature(stat, source)
    )
    monkeypatch.setattr(mounts, "local_file_signature", model.local_file_signature)
    env = grants(tmp_path, ("model/vae", path.parent))
    ref = resolve_asset_selection("model/vae", path.name, env=env)
    assert isinstance(ref, LocalAssetRef)
    assert ref.read_bytes() == b"first"
    path.write_bytes(b"other")
    with pytest.raises(AssetError):
        ref.read_bytes()


def test_unsupported_revision_mount_waits_for_background_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_assets import MountDef, MountTable, mounts

    monkeypatch.setattr(
        mounts,
        "mount_revision_capability",
        lambda _root: {"supported": False, "filesystem": "exFAT", "reason": "no file USN"},
    )
    root = tmp_path / "vae"
    path = seed(root, "real.safetensors", b"weights")
    env = grants(tmp_path, ("model/vae", root))
    with pytest.raises(AssetError) as refused:
        resolve_asset_selection("model/vae", path.name, env=env)
    message = str(refused.value)
    for detail in (
        "grant-0",
        "exFAT",
        "file revision tracking unavailable",
        "background index",
        "verified AssetRef",
        "missingEntryCount",
    ):
        assert detail in message
    snapshot = Path(env["DINKSTER_MOUNTS_SNAPSHOT"])
    table = MountTable(snapshot)
    table.add(MountDef(id="vae", path=root), kind="model/vae")
    with pytest.raises(AssetError, match="background index"):
        table.bind_name(path.name, "model/vae")
    table.scan("vae")
    selected = resolve_asset_selection("model/vae", path.name, env=env)
    assert isinstance(selected, AssetRef)
    assert selected.digest == digest_bytes(b"weights")
    assert selected.read_bytes() == b"weights"
    named = table.bind_name(path.name, "model/vae")
    assert isinstance(named, AssetRef)
    assert named.read_bytes() == b"weights"
    if os.name == "nt":
        assert table.descriptors()[0]["localBindingCapability"] == {
            "supported": False,
            "filesystem": "exFAT",
            "reason": "no file USN",
        }
    table.remove("vae")
    for ref in (selected, named):
        with pytest.raises(AssetError):
            ref.read_bytes()


@pytest.mark.parametrize("verified", [False, True])
def test_existing_file_refs_keep_verification_and_require_live_exact_grants(
    tmp_path: Path, verified: bool
) -> None:
    path = seed(tmp_path / "vae", "real.safetensors", b"weights")
    env = grants(tmp_path, ("model/vae", path.parent))
    local = resolve_asset_selection("model/vae", path.name, env=env)
    assert isinstance(local, LocalAssetRef)

    class Resolver:
        def resolve(self, digest: str) -> Path | None:
            return path if digest == digest_bytes(b"weights") else None

    ref = (
        AssetRef(digest_bytes(b"weights"), path.name, 7, resolver=Resolver()) if verified else local
    )
    assert resolve_asset_selection("model/vae", ref, env=env) is ref
    assert ref.read_bytes() == b"weights"
    assert ("digest" in ref.to_wire()) is verified
    with pytest.raises(AssetError):
        resolve_asset_selection("model/diffusers", ref, env=env)
    shadow = seed(tmp_path / "higher-priority", path.name, b"other-weights")
    grants(tmp_path, ("model/vae", shadow.parent), ("model/vae", path.parent))
    assert resolve_asset_selection("model/vae", ref, env=env) is ref
    grants(tmp_path, ("model/checkpoint", path.parent))
    with pytest.raises(AssetError):
        resolve_asset_selection("model/vae", ref, env=env)


def test_directory_selection_rejects_links_outside_subtree_and_cycles(tmp_path: Path) -> None:
    root, outside = tmp_path / "grant", tmp_path / "outside"
    seed(root, "model/model_index.json")
    seed(root, "sibling/weights.bin")
    seed(outside, "model_index.json")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    env = grants(tmp_path, ("model/diffusers", root))
    assert "escape" not in asset_selection_names("model/diffusers", env=env)
    (root / "model" / "weights.bin").symlink_to(root / "sibling/weights.bin")
    with pytest.raises(AssetError):
        resolve_asset_selection("model/diffusers", "model", env=env)
    (root / "model" / "weights.bin").unlink()
    (root / "model" / "cycle").symlink_to(root / "model", target_is_directory=True)
    with pytest.raises(AssetError):
        resolve_asset_selection("model/diffusers", "model", env=env)
