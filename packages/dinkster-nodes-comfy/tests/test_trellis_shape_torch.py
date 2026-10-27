"""Dense Trellis occupancy creates batch-grouped sparse sampling inputs."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_nodes_comfy.trellis_shape import Trellis2ShapeStage
from dinkster_values import decode_latent, encode_latent


@pytest.mark.parametrize("resolution", (32, 33))
@pytest.mark.parametrize("project", (False, True))
@pytest.mark.parametrize("portable", (False, True))
def test_sparse_batch_association_and_stage_dispatch(
    monkeypatch: pytest.MonkeyPatch, resolution: int, project: bool, portable: bool
) -> None:
    occupancy = torch.zeros(2, 3, 4, resolution)
    occupancy[0, 2, 1, 3] = -2
    occupancy[0, 0, 3, 1] = 5
    occupancy[1, 1, 0, 2] = 1
    before = occupancy.clone()
    expected_coords = torch.tensor([[0, 0, 3, 1], [0, 2, 1, 3], [1, 1, 0, 2]], dtype=torch.int32)
    tokens = torch.tensor([[[7.0, 11.0]]])
    pack = {"projection marker": object()}
    positive = [[tokens, {"keep": "positive", **({"proj_feat_pack": pack} if project else {})}]]
    negative = [[-tokens, {"keep": "negative"}], "unchanged"]
    if project:
        initialize_inference_process()
        model = importlib.import_module("dinkster_inference.ldm.trellis2.model")

        def projection(actual: Any, stage: str, *, coords: Any, coord_resolution: int) -> Any:
            assert actual is pack
            assert stage == ("shape_512" if resolution == 32 else "shape_1024")
            assert coord_resolution == resolution
            torch.testing.assert_close(coords, expected_coords)
            return coords[:, 1:].float() + (100 if stage == "shape_512" else 200)

        monkeypatch.setattr(model, "compute_stage_proj_feats", projection)
    data: Any = occupancy
    if portable:
        data = decode_latent(encode_latent({"samples": occupancy}))["samples"]
    output = Trellis2ShapeStage.execute(
        positive=positive, negative=negative, voxel=SimpleNamespace(data=data)
    )
    latent = output["LATENT"]
    assert isinstance(latent, dict)
    torch.testing.assert_close(latent["coords"], expected_coords)
    torch.testing.assert_close(latent["coord_counts"], torch.tensor([2, 1]))
    assert latent["samples"].shape == (2, 32, 2, 1)
    assert torch.count_nonzero(latent["samples"]) == 0
    assert latent["coord_resolution"] == resolution and latent["type"] == "trellis2"
    assert latent["model_frame"] == ("y_up" if project else "z_up")
    for name in ("positive", "negative"):
        result: Any = output[name]
        assert result[0][1]["keep"] == name
        torch.testing.assert_close(result[0][1]["trellis2_coords"], expected_coords)
        torch.testing.assert_close(result[0][1]["trellis2_coord_counts"], torch.tensor([2, 1]))
        assert result[0][1]["trellis2_generation_mode"] == (
            "shape_generation_512" if resolution == 32 else "shape_generation"
        )
        if project:
            torch.testing.assert_close(
                result[0][1]["trellis2_proj_feats"],
                expected_coords[:, 1:].float() + (100 if resolution == 32 else 200),
            )
        else:
            assert "trellis2_proj_feats" not in result[0][1]
    assert output["negative"][1] == "unchanged"
    assert "trellis2_coords" not in positive[0][1] and "trellis2_coords" not in negative[0][1]
    torch.testing.assert_close(occupancy, before)


@pytest.mark.parametrize("missing_first", (False, True))
def test_empty_or_noncontiguous_occupancy_errors(missing_first: bool) -> None:
    data = torch.zeros(2, 2, 2, 2)
    if missing_first:
        data[1, 0, 0, 0] = 1
    with pytest.raises(
        ValueError, match="Non-contiguous Trellis2 batch ids" if missing_first else "can't be empty"
    ):
        Trellis2ShapeStage.execute(positive=[], negative=[], voxel=SimpleNamespace(data=data))
