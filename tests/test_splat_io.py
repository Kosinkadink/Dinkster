"""Native asset-backed gaussian splat I/O contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import dinkster_nodes_media_io.splat as splat_module
import numpy
import pytest
from dinkster_assets import (
    AssetError,
    AssetRef,
    AssetVault,
    MountSnapshotResolver,
    classify_media,
    digest_bytes,
)
from dinkster_nodes_media_io import LoadGaussianSplat, SaveGaussianSplat
from dinkster_values import render_splat_ply

from tests.test_splat_codec import splat_value


def _mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "out"
    root.mkdir()
    index = root / ".dinkster-asset-index.json"
    index.write_text("{}", "utf-8")
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {
                        "id": "comfy-output",
                        "root": str(root),
                        "index": str(index),
                        "mode": "readwrite",
                    }
                ]
            }
        ),
        "utf-8",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return root, snapshot


def _bound(ref: AssetRef, snapshot: Path) -> AssetRef:
    return AssetRef(
        ref.digest,
        ref.name,
        ref.size,
        ref.media_type,
        ref.virtual_path,
        MountSnapshotResolver(snapshot),
    )


def test_save_then_load_round_trips_the_splat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, snapshot = _mount(tmp_path, monkeypatch)
    value = splat_value(gaussians=6)
    data = render_splat_ply(value)
    saved = SaveGaussianSplat.execute(splat=value)
    ref = cast(AssetRef, saved["splat"])
    assert ref.media_type == "model/ply"
    assert ref.name.endswith(".ply")
    assert ref.size == len(data)
    written = root / "3d" / ref.name
    assert written.read_bytes() == data
    assert ref.digest == digest_bytes(data)
    loaded = cast("dict[str, Any]", LoadGaussianSplat.execute(splat=_bound(ref, snapshot)))
    for key in ("positions", "scales", "rotations", "opacities", "sh"):
        assert numpy.allclose(loaded["splat"][key], value[key], atol=1e-5), key


def test_saved_splat_passes_the_media_upload_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _snapshot = _mount(tmp_path, monkeypatch)
    saved = SaveGaussianSplat.execute(splat=splat_value())
    ref = cast(AssetRef, saved["splat"])
    classification = classify_media((root / "3d" / ref.name).read_bytes())
    assert (classification.kind, classification.media_type, classification.extension) == (
        "media/model3d",
        "model/ply",
        "ply",
    )


def test_save_honors_explicit_target_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _snapshot = _mount(tmp_path, monkeypatch)
    saved = SaveGaussianSplat.execute(
        splat=splat_value(),
        target={"mount": "comfy-output", "prefix": "splats/scene"},
    )
    ref = cast(AssetRef, saved["splat"])
    assert (root / "splats" / ref.name).is_file()


def test_save_refuses_malformed_splat_values() -> None:
    with pytest.raises(ValueError, match="exactly positions"):
        SaveGaussianSplat.execute(splat={"positions": numpy.zeros((1, 1, 3), numpy.float32)})


def test_save_requires_configured_mounts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DINKSTER_MOUNTS_SNAPSHOT", raising=False)
    with pytest.raises(AssetError, match="DINKSTER_MOUNTS_SNAPSHOT"):
        SaveGaussianSplat.execute(splat=splat_value())


def test_save_bounds_serialized_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mount(tmp_path, monkeypatch)
    monkeypatch.setattr(splat_module, "MAX_SPLAT_PLY_BYTES", 16)
    with pytest.raises(ValueError, match="output limit"):
        SaveGaussianSplat.execute(splat=splat_value())


def test_load_refuses_non_splat_assets(tmp_path: Path) -> None:
    payload = b"not a splat"
    digest = digest_bytes(payload)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
    ref = AssetRef(digest, "bad.ply", len(payload), "model/ply", resolver=vault)
    with pytest.raises(ValueError, match="cannot decode 'bad.ply' as a gaussian splat"):
        LoadGaussianSplat.execute(splat=ref)
