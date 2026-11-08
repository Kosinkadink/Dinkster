"""Native asset-backed gaussian splat I/O contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import dinkster_3d.save3d as save_module
import numpy
import pytest
from dinkster_3d.save3d import SaveGaussianSplat
from dinkster_3d.splat_files import File3DToSplat, SplatToFile3D
from dinkster_assets import (
    AssetError,
    AssetRef,
    AssetVault,
    MountSnapshotResolver,
    classify_media,
    digest_bytes,
)
from dinkster_schema.reporting import capture_saved_artifacts
from dinkster_values import render_splat_ply

from tests.test_splat_codec import splat_value

pytestmark = pytest.mark.usefixtures("saved_artifact_context")


@pytest.fixture(autouse=True)
def produced_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))


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
                ],
                "outputMount": "comfy-output",
            }
        ),
        "utf-8",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    return root, snapshot


def test_save_then_load_round_trips_the_splat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, snapshot = _mount(tmp_path, monkeypatch)
    value = splat_value(gaussians=6)
    data = render_splat_ply(value)
    model = cast("dict[str, object]", SplatToFile3D.execute(splat=value)["model_3d"])
    with capture_saved_artifacts() as claims:
        saved = SaveGaussianSplat.execute(model_3d=model, viewport_state={})
    assert saved["model_3d"] is model
    (claim,) = claims
    ref = AssetRef.from_wire(claim, MountSnapshotResolver(snapshot))
    assert ref.name == "ComfyUI_00001.ply"
    assert ref.media_type == cast(AssetRef, model["source"]).media_type
    assert ref.size == len(data)
    written = root / "3d" / ref.name
    assert written.read_bytes() == data
    assert ref.digest == digest_bytes(data)
    loaded = cast(
        Any,
        File3DToSplat.execute(model_3d={"format": "ply", "source": ref})["splat"],
    )
    for key in ("positions", "scales", "rotations", "opacities", "sh"):
        numpy.testing.assert_allclose(getattr(loaded, key).cpu().numpy(), value[key], atol=1e-5)


def test_saved_splat_passes_the_media_upload_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, snapshot = _mount(tmp_path, monkeypatch)
    model = cast("dict[str, object]", SplatToFile3D.execute(splat=splat_value())["model_3d"])
    with capture_saved_artifacts() as claims:
        SaveGaussianSplat.execute(model_3d=model, viewport_state={})
    (claim,) = claims
    ref = AssetRef.from_wire(claim, MountSnapshotResolver(snapshot))
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
    model = cast("dict[str, object]", SplatToFile3D.execute(splat=splat_value())["model_3d"])
    SaveGaussianSplat.execute(
        model_3d=model,
        viewport_state={},
        mount="comfy-output",
        filename_prefix="splats/scene",
    )
    assert (root / "splats/scene_00001.ply").read_bytes() == cast(
        AssetRef, model["source"]
    ).read_bytes()


def test_file_conversion_refuses_malformed_splat_values() -> None:
    with pytest.raises(ValueError, match="exactly positions"):
        SplatToFile3D.execute(splat={"positions": numpy.zeros((1, 1, 3), numpy.float32)})


def test_save_requires_configured_mounts(monkeypatch: pytest.MonkeyPatch) -> None:
    model = cast("dict[str, object]", SplatToFile3D.execute(splat=splat_value())["model_3d"])
    monkeypatch.delenv("DINKSTER_MOUNTS_SNAPSHOT", raising=False)
    with pytest.raises(AssetError, match="DINKSTER_MOUNTS_SNAPSHOT"):
        SaveGaussianSplat.execute(model_3d=model, viewport_state={})


def test_save_bounds_published_file_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _snapshot = _mount(tmp_path, monkeypatch)
    model = cast("dict[str, object]", SplatToFile3D.execute(splat=splat_value())["model_3d"])
    monkeypatch.setattr(save_module, "ENCODED_MEDIA_LIMIT_BYTES", 16)
    with pytest.raises(AssetError, match="limit"):
        SaveGaussianSplat.execute(model_3d=model, viewport_state={})
    assert not list(root.rglob("*.ply"))


def test_load_refuses_non_splat_assets(tmp_path: Path) -> None:
    payload = b"not a splat"
    digest = digest_bytes(payload)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(payload)
        writer.commit()
    ref = AssetRef(digest, "bad.ply", len(payload), "model/ply", resolver=vault)
    with pytest.raises(ValueError, match="PLY"):
        File3DToSplat.execute(model_3d={"format": "ply", "source": ref})
