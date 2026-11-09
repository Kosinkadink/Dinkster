"""Texture-stage sparse batches and metadata survive ordinary native pack hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    LATENT_TYPE,
    VOXEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.trellis import Trellis2TextureStage

from dinkster.compose import PackSpec, ServingComposer


def stage_inputs(*, padded: bool = True) -> dict[str, Any]:
    # Hand-derived: batch zero has two tokens; batch one has one. The third
    # padded token in each batch must not become a sparse shape feature.
    coords = torch.tensor([[1, 1, 2, 3], [0, 3, 2, 1], [0, 0, 1, 2]], dtype=torch.int32)
    samples = (
        torch.arange(12).reshape(2, 2, 3, 1).float()
        if padded
        else torch.tensor([[0.0, 3.0], [1.0, 4.0], [6.0, 9.0]])
    )
    return {
        "positive": [[torch.ones(1, 2, 3), {"keep": "positive"}]],
        "negative": [[torch.zeros(1, 2, 3), {"keep": "negative"}]],
        "shape_latent": {"samples": samples, "coords": coords, "coord_resolution": 8},
    }


@pytest.mark.parametrize("padded", (False, True))
def test_texture_stage_preserves_sparse_layout_and_inputs(padded: bool) -> None:
    inputs = stage_inputs(padded=padded)
    before = inputs["shape_latent"]["samples"].clone()
    result = Trellis2TextureStage.execute(**inputs)
    latent = result["LATENT"]
    assert latent["samples"].shape == (2, 32, 2, 1)
    assert latent["samples"].dtype == torch.float32
    assert torch.count_nonzero(latent["samples"]) == 0
    assert latent["coords"] is inputs["shape_latent"]["coords"]
    assert torch.equal(latent["coord_counts"], torch.tensor([2, 1]))
    assert latent["coord_resolution"] == 8
    assert latent["model_frame"] == "z_up"
    expected = torch.tensor([[0.0, 3.0], [1.0, 4.0], [6.0, 9.0]])
    for name in ("positive", "negative"):
        entry = result[name][0]
        assert entry[0] is inputs[name][0][0]
        assert entry[1]["keep"] == name
        assert entry[1]["trellis2_generation_mode"] == "texture_generation"
        assert torch.equal(entry[1]["trellis2_shape_slat"], expected)
        assert torch.equal(entry[1]["trellis2_coord_counts"], torch.tensor([2, 1]))
        assert entry[1]["trellis2_model_frame"] == "z_up"
        assert inputs[name][0][1] == {"keep": name}
    assert torch.equal(inputs["shape_latent"]["samples"], before)


@pytest.mark.parametrize("resolution", (None, 8))
def test_texture_stage_projection_and_frame_override(resolution: int | None) -> None:
    inputs = stage_inputs()
    if resolution is None:
        del inputs["shape_latent"]["coord_resolution"]
    inputs["shape_latent"]["model_frame"] = "authored_frame"
    # Hand-derived: constant maps project the same two channel values at every
    # coordinate. A different fallback map exposes selecting the wrong stage.
    inputs["positive"][0][1]["proj_feat_pack"] = {
        "mesh_scale": torch.ones(1),
        "transform_matrix": torch.eye(4).unsqueeze(0),
        "camera_angle_x": torch.ones(1),
        "feature_map": torch.full((1, 2, 2, 2), 99.0),
        "stages": {
            "tex_1024": {
                "feature_map": torch.tensor([2.0, 7.0]).reshape(1, 2, 1, 1).expand(1, 2, 2, 2),
                "image_resolution": 4,
            }
        },
    }
    result = Trellis2TextureStage.execute(**inputs)
    assert result["LATENT"]["model_frame"] == "authored_frame"
    for name in ("positive", "negative"):
        metadata = result[name][0][1]
        assert metadata["trellis2_model_frame"] == "authored_frame"
        if resolution is None:
            assert "trellis2_proj_feats" not in metadata
            assert "coord_resolution" not in result["LATENT"]
        else:
            assert torch.equal(metadata["trellis2_proj_feats"], torch.tensor([[2.0, 7.0]] * 3))
    del inputs["shape_latent"]["model_frame"]
    assert Trellis2TextureStage.execute(**inputs)["LATENT"]["model_frame"] == "y_up"


@pytest.mark.parametrize(
    ("coords", "message"),
    (
        (torch.zeros(3, 3), "shape"),
        (torch.zeros(0, 4), "empty"),
        (torch.tensor([[-1, 0, 0, 0]]), "non-negative"),
        (torch.tensor([[0, 0, 0, 0], [2, 0, 0, 0]]), "Non-contiguous"),
    ),
)
def test_texture_stage_rejects_invalid_coord_batches(coords: Any, message: str) -> None:
    inputs = stage_inputs()
    inputs["shape_latent"]["coords"] = coords
    with pytest.raises(ValueError, match=message):
        Trellis2TextureStage.execute(**inputs)


class TextureSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="trellis-fixture.source",
            outputs=(
                OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("voxel", TypeExpr.concrete(VOXEL_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        occupancy = torch.zeros(2, 3, 4, 8)
        occupancy[0, 0, 3, 1] = 5
        occupancy[0, 2, 1, 3] = -2
        occupancy[1, 1, 0, 2] = 1
        return cls.outputs(
            positive=[[torch.ones(1, 2, 3), {"keep": "positive"}]],
            negative=[[torch.zeros(1, 2, 3), {"keep": "negative"}]],
            voxel={"data": occupancy},
        )


class ShapeFeatures(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="trellis-fixture.features",
            inputs=(InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),),
            outputs=(OutputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),),
        )

    @classmethod
    def execute(cls, *, latent: Any) -> Any:
        latent = materialize_latent(latent)
        torch.testing.assert_close(
            latent["coords"],
            torch.tensor([[0, 0, 3, 1], [0, 2, 1, 3], [1, 1, 0, 2]], dtype=torch.int32),
        )
        torch.testing.assert_close(latent["coord_counts"], torch.tensor([2, 1]))
        assert latent["samples"].shape == (2, 32, 2, 1)
        assert torch.count_nonzero(latent["samples"]) == 0
        samples = torch.full_like(latent["samples"], 9999)
        samples[0, :, 0, 0] = torch.arange(32) + 11
        samples[0, :, 1, 0] = torch.arange(32) + 101
        samples[1, :, 0, 0] = torch.arange(32) + 1001
        return cls.outputs(latent={**latent, "samples": samples})


class TextureInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="trellis-fixture.inspect",
            inputs=(
                InputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, positive: Any, negative: Any, latent: object) -> Any:
        latent = materialize_latent(latent)
        assert latent["samples"].shape == (2, 32, 2, 1)
        assert torch.count_nonzero(latent["samples"]) == 0
        assert torch.equal(latent["coord_counts"], torch.tensor([2, 1]))
        expected_coords = torch.tensor(
            [[0, 0, 3, 1], [0, 2, 1, 3], [1, 1, 0, 2]], dtype=torch.int32
        )
        torch.testing.assert_close(latent["coords"], expected_coords, rtol=0, atol=0)
        expected_features = torch.arange(32)[None] + torch.tensor([[11], [101], [1001]])
        for name, conditioning in (("positive", positive), ("negative", negative)):
            conditioning = materialize_latent(conditioning)
            assert conditioning[0][1]["keep"] == name
            torch.testing.assert_close(
                conditioning[0][1]["trellis2_shape_slat"],
                expected_features.float(),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                conditioning[0][1]["trellis2_coords"], expected_coords, rtol=0, atol=0
            )
            assert conditioning[0][1]["trellis2_generation_mode"] == "texture_generation"
        return cls.outputs(summary="native shape coordinates retain sampled feature association")


@pytest.mark.parametrize("in_process", (False, True))
def test_native_manifest_texture_stage_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "trellis_fixture.py").write_text(
            "from test_trellis_torch import TextureSource, ShapeFeatures, TextureInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (TextureSource, ShapeFeatures, TextureInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "trellis.toml"
        fixture.write_text(
            '[pack]\nname="trellis-fixture"\npublisher="dinkster"\n'
            'namespaces=["trellis-fixture"]\n[pack.entry]\n'
            'nodes="trellis_fixture:NODES"\ntypes="trellis_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "trellis-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("trellis-fixture.source"),
                    "shape": GraphNode(
                        "Trellis2ShapeStage",
                        {name: Link("source", name) for name in ("positive", "negative", "voxel")},
                    ),
                    "features": GraphNode(
                        "trellis-fixture.features", {"latent": Link("shape", "LATENT")}
                    ),
                    "texture": GraphNode(
                        "Trellis2TextureStage",
                        {
                            "positive": Link("shape", "positive"),
                            "negative": Link("shape", "negative"),
                            "shape_latent": Link("features", "latent"),
                        },
                    ),
                    "inspect": GraphNode(
                        "trellis-fixture.inspect",
                        {
                            "positive": Link("texture", "positive"),
                            "negative": Link("texture", "negative"),
                            "latent": Link("texture", "LATENT"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "native shape coordinates retain sampled feature association"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
