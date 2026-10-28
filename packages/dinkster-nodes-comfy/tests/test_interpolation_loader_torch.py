"""Real checkpoint loading through the public interpolation node."""

from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import AssetRef, initialize_inference_process
from dinkster_assets import digest_file
from dinkster_nodes_comfy.interpolation_loader import FrameInterpolationModelLoader
from safetensors.torch import save_file


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, _digest: str) -> Path:
        return self.path


@pytest.mark.parametrize("kind", ("film", "rife", "raw-rife"))
def test_real_checkpoint_loads_exact_weights_and_configuration(tmp_path: Path, kind: str) -> None:
    initialize_inference_process()
    from dinkster_inference.model_patcher import CoreModelPatcher
    from dinkster_nodes_comfy.interpolation_models import film_model, rife_model

    source = film_model() if kind == "film" else rife_model(6, (8, 10, 12, 14, 16))
    expected = {}
    for index, (key, value) in enumerate(source.state_dict().items()):
        expected[key] = torch.full_like(value, (index % 11 - 5) / 1024)
    stored = expected
    if kind == "raw-rife":
        stored = {
            "module." + key.replace("blocks.", "block"): value for key, value in expected.items()
        }
        stored["module.teacher.discarded"] = torch.tensor([73.0])
        stored["module.caltime.discarded"] = torch.tensor([-19.0])
    path = tmp_path / "interpolation.safetensors"
    save_file(stored, path)
    digest = digest_file(path)
    asset = AssetRef(digest, path.name, path.stat().st_size, resolver=_Resolver(path))
    result = FrameInterpolationModelLoader.execute(model_name=asset)["INTERP_MODEL"]
    assert isinstance(result, CoreModelPatcher)
    assert not result.model.training
    assert result.model.get_dtype() == torch.float32
    actual = result.model.state_dict()
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert digest_file(path) == digest
    if kind != "film":
        assert result.model.encode.cnn3.out_channels == 6
        assert [block.conv0[1][0].out_channels for block in result.model.blocks] == [
            8,
            10,
            12,
            14,
            16,
        ]


def test_unrecognized_checkpoint_has_reference_error(tmp_path: Path) -> None:
    initialize_inference_process()
    path = tmp_path / "unknown.safetensors"
    save_file({"unrelated.weight": torch.tensor([[1.0, -3.0]])}, path)
    asset = AssetRef(digest_file(path), path.name, path.stat().st_size, resolver=_Resolver(path))
    with pytest.raises(ValueError, match="^Unrecognized frame interpolation model format$"):
        FrameInterpolationModelLoader.execute(model_name=asset)


def test_unbound_checkpoint_is_rejected() -> None:
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        FrameInterpolationModelLoader.execute(model_name="checkpoint.safetensors")
