"""Source-named selections use only live, exact-kind directory grants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dinkster_api.v1 import (
    AssetError,
    LocalAssetRef,
    asset_selection_names,
    resolve_asset_selection,
)


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
