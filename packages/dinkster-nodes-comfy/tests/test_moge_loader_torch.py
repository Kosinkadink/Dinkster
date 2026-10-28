"""Real checkpoint bytes at the MoGe loader's granted file boundary."""

from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import AssetRef, LocalAssetRef, initialize_inference_process
from dinkster_assets import AssetError, digest_file
from dinkster_assets.model import LocalFileSignature, local_file_signature
from dinkster_nodes_comfy.moge_loader import LoadMoGeModel
from safetensors.torch import save_file


@pytest.mark.parametrize("format", ["safetensors", "pt"])
@pytest.mark.parametrize("local", [False, True])
def test_moge_loader_preserves_checkpoint_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format: str, local: bool
) -> None:
    initialize_inference_process()
    from dinkster_inference.ldm.moge import model

    class LoadedModel:
        def __init__(self, state: dict[str, torch.Tensor]) -> None:
            self.state = state

    monkeypatch.setattr(model, "MoGeModel", LoadedModel)
    state = {
        "encoder.weight": torch.tensor([[1.25, -3.5], [7.0, 0.125]], dtype=torch.float16),
        "mask_threshold": torch.tensor(0.375),
        "indices": torch.tensor([4, 1, 9], dtype=torch.int64),
    }
    path = tmp_path / f"geometry.{format}"
    if format == "safetensors":
        save_file(state, path)
    else:
        torch.save(state, path)

    class Resolver:
        def resolve(self, digest: str) -> Path:
            return path

        def resolve_local(self, binding: str) -> tuple[Path, LocalFileSignature]:
            return path, local_file_signature(path.stat())

        def reconcile_local(self, binding: str) -> None:
            return None

    asset: AssetRef | LocalAssetRef
    if local:
        asset = LocalAssetRef(
            "granted", path.name, path.stat().st_size, path.name, resolver=Resolver()
        )
    else:
        asset = AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=Resolver())
    loaded = LoadMoGeModel.execute(model_name=asset)["MOGE_MODEL"]
    assert isinstance(loaded, LoadedModel)
    assert loaded.state.keys() == state.keys()
    for key, tensor in state.items():
        assert loaded.state[key].dtype == tensor.dtype
        assert torch.equal(loaded.state[key], tensor)


@pytest.mark.parametrize("value", ["ungranted.safetensors", Path("ungranted.pt"), None])
def test_moge_loader_rejects_ungranted_paths(value: object) -> None:
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        LoadMoGeModel.execute(model_name=value)


def test_moge_loader_rejects_revoked_binding() -> None:
    asset = LocalAssetRef("revoked", "missing.pt", 4, "geometry/missing.pt")
    with pytest.raises(AssetError, match="unavailable, revoked or changed"):
        LoadMoGeModel.execute(model_name=asset)
