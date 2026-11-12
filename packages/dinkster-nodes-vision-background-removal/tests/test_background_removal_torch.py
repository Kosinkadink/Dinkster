"""CPU contracts for admitted background models and foreground masks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from dinkster_api.v1 import (
    AssetRef,
    LocalAssetRef,
)
from dinkster_assets import AssetError, digest_file
from dinkster_vision_background_removal import background_removal
from dinkster_vision_background_removal.background_removal import (
    LoadBackgroundRemovalModel,
)


@pytest.mark.parametrize("local", (False, True))
@pytest.mark.parametrize("valid", (False, True))
def test_loader_preserves_admitted_model_identity_and_rejects_invalid_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local: bool, valid: bool
) -> None:
    path = tmp_path / "foreground.safetensors"
    path.write_bytes(b"controlled checkpoint")

    class Resolver:
        def resolve(self, digest: str) -> Path:
            return path

        def resolve_local(self, binding: str) -> tuple[Path, tuple[int, int, int, int, int]]:
            stat = path.stat()
            return path, (
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                stat.st_dev,
                stat.st_ino,
            )

        def reconcile_local(self, binding: str) -> AssetRef | None:
            return None

    resolver = Resolver()
    asset = (
        LocalAssetRef("host-binding", path.name, path.stat().st_size, path.name, resolver=resolver)
        if local
        else AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=resolver)
    )
    model = object()

    def load(filename: str) -> object:
        assert filename == str(path)
        return model if valid else None

    monkeypatch.setattr(background_removal, "initialize_inference_process", lambda: None)
    monkeypatch.setattr(
        background_removal,
        "importlib",
        SimpleNamespace(import_module=lambda name: SimpleNamespace(load=load)),
    )
    if valid:
        assert LoadBackgroundRemovalModel.execute(bg_removal_name=asset)["bg_model"] is model
    else:
        with pytest.raises(RuntimeError) as error:
            LoadBackgroundRemovalModel.execute(bg_removal_name=asset)
        assert str(error.value) == (
            "ERROR: background model file is invalid and does not contain a valid "
            "background removal model."
        )


@pytest.mark.parametrize(
    "asset", ("foreground.safetensors", AssetRef("blake3:" + "b" * 64, "missing", 1))
)
def test_loader_rejects_unadmitted_input_before_inference_initialization(
    monkeypatch: pytest.MonkeyPatch, asset: object
) -> None:
    def forbidden() -> None:
        pytest.fail("unadmitted path initialized inference")

    monkeypatch.setattr(background_removal, "initialize_inference_process", forbidden)
    with pytest.raises((TypeError, AssetError)):
        LoadBackgroundRemovalModel.execute(bg_removal_name=asset)
