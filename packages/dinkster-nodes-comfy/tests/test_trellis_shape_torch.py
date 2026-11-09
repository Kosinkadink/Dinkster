"""Dense Trellis occupancy creates batch-grouped sparse sampling inputs."""

from __future__ import annotations

import asyncio
import importlib
import os
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    LATENT_TYPE,
    VAE_TYPE,
    VOXEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
    materialize_mesh,
    materialize_shape_subdivides,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.trellis_shape import Trellis2ShapeStage, VaeDecodeShapeTrellis
from dinkster_values import decode_latent, encode_latent

from dinkster.compose import PackSpec, ServingComposer


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


class ShapeSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="shape-fixture.source",
            outputs=(
                OutputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
                OutputSpec("voxel", TypeExpr.concrete(VOXEL_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        occupancy = torch.zeros(2, 3, 4, 33)
        occupancy[0, 2, 1, 3] = -2
        occupancy[0, 0, 3, 1] = 5
        occupancy[1, 1, 0, 2] = 1
        return cls.outputs(
            positive=[[torch.tensor([[[7.0, 11.0]]]), {"keep": "positive"}]],
            negative=[[torch.tensor([[[-3.0, -13.0]]]), {"keep": "negative"}]],
            voxel={"data": occupancy, "voxel_colors": None, "resolution": (3, 4, 33)},
        )


class ShapeInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="shape-fixture.inspect",
            inputs=(
                InputSpec("positive", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("negative", TypeExpr.concrete(CONDITIONING_TYPE)),
                InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, positive: Any, negative: Any, latent: Any) -> dict[str, object]:
        latent = materialize_latent(latent)
        coords = torch.tensor([[0, 0, 3, 1], [0, 2, 1, 3], [1, 1, 0, 2]], dtype=torch.int32)
        counts = torch.tensor([2, 1])
        torch.testing.assert_close(latent["coords"], coords)
        torch.testing.assert_close(latent["coord_counts"], counts)
        assert latent["samples"].shape == (2, 32, 2, 1)
        assert torch.count_nonzero(latent["samples"]) == 0
        assert latent["coord_resolution"] == 33 and latent["model_frame"] == "z_up"
        for name, conditioning, tokens in (
            ("positive", positive, [7.0, 11.0]),
            ("negative", negative, [-3.0, -13.0]),
        ):
            conditioning = materialize_latent(conditioning)
            torch.testing.assert_close(conditioning[0][0], torch.tensor([[tokens]]))
            extras = conditioning[0][1]
            assert extras["keep"] == name
            assert extras["trellis2_generation_mode"] == "shape_generation"
            assert "trellis2_proj_feats" not in extras
            torch.testing.assert_close(extras["trellis2_coords"], coords)
            torch.testing.assert_close(extras["trellis2_coord_counts"], counts)
        return cls.outputs(summary="batch-grouped shape conditioning and latent")


@pytest.mark.parametrize("in_process", (False, True))
def test_ordinary_manifest_shape_stage_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"shape_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from test_trellis_shape_torch import ShapeSource, ShapeInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (ShapeSource, ShapeInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname = "shape-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["shape-fixture"]\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
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
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="shape-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("shape-fixture.source"),
                    "shape": GraphNode(
                        "Trellis2ShapeStage",
                        {name: Link("source", name) for name in ("positive", "negative", "voxel")},
                    ),
                    "inspect": GraphNode(
                        "shape-fixture.inspect",
                        {
                            "positive": Link("shape", "positive"),
                            "negative": Link("shape", "negative"),
                            "latent": Link("shape", "LATENT"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "batch-grouped shape conditioning and latent"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())


def decode_samples(*, counted: bool = True) -> Any:
    values = torch.zeros(2, 32, 2, 1)
    values[0, 0, :, 0] = torch.tensor([1.0, 3.0])
    values[1, 0, :, 0] = torch.tensor([5.0, 999.0 if counted else 7.0])
    return {
        "samples": values,
        "coords": torch.tensor(
            [[1, 5, 6, 7], [0, 1, 2, 3], [0, 3, 4, 5]]
            if counted
            else [[0, 1, 2, 3], [0, 3, 4, 5], [1, 5, 6, 7], [1, 7, 8, 9]],
            dtype=torch.int32,
        ),
        **({"coord_counts": torch.tensor([2, 1]), "coord_resolution": 64} if counted else {}),
        "model_frame": "z_up",
    }


class ShapeDecoder:
    vae_dtype = torch.float16

    def __init__(self) -> None:
        self.first_stage_model = self
        self.resolution = torch.tensor(1536)
        self.prepared: Any = None
        self.inputs: list[Any] = []

    def prepare_decode(self, shape: Any, *, memory_required: int) -> None:
        self.prepared = (shape, memory_required)

    def decode_shape_slat(self, value: Any, resolution: int) -> Any:
        sparse = importlib.import_module("dinkster_inference.ldm.trellis2.vae")
        self.inputs.append(value)
        assert value.feats.dtype == torch.float16 and value.coords.dtype == torch.int32
        assert resolution in (1024, 1536)
        meshes = []
        for index in range(value.shape[0]):
            count = int((value.coords[:, 0] == index).sum())
            vertices = torch.tensor([[1.0, 2.0, 3.0]]).repeat(count + 2, 1).double()
            faces = torch.tensor([[0, 1, 2]]).repeat(count, 1)
            meshes.append((vertices, faces))
        stages = [
            sparse.SparseTensor(
                feats=value.feats[:, :2] + index,
                coords=value.coords.clone(),
                shape=torch.Size([value.shape[0], 2]),
                scale=(Fraction(1, 3), Fraction(2), Fraction(5, 7)),
                spatial_cache={"must-not-cross-wire": object()},
            )
            for index in range(2)
        ]
        return meshes, stages


@pytest.mark.parametrize("counted", (False, True))
@pytest.mark.parametrize("portable", (False, True))
def test_shape_decode_mesh_frames_sparse_stages_and_ragged_counts(
    counted: bool, portable: bool
) -> None:
    latent = decode_samples(counted=counted)
    before = latent["coords"].clone()
    values_before = latent["samples"].clone()
    vae = ShapeDecoder()
    argument = decode_latent(encode_latent(latent)) if portable else latent
    output = VaeDecodeShapeTrellis.execute(samples=argument, vae=vae)
    mesh = materialize_mesh(output["mesh"])
    expected_resolution = 1024 if counted else 1536
    assert vae.prepared == (
        (2, 32, 2, 1),
        2 * 1024**3 + expected_resolution**2 * 5 // 4 * (896 * 2 + 108),
    )
    expected_values = torch.tensor([1.0, 3.0, 5.0] if counted else [1.0, 3.0, 5.0, 7.0])
    expected_features = (expected_values * 5.972266 + 0.781296).half()
    torch.testing.assert_close(
        torch.cat([item.feats[:, 0] for item in vae.inputs]), expected_features, rtol=0, atol=0
    )
    assert len(vae.inputs) == (2 if counted else 1)
    assert mesh.vertices.dtype == torch.float32 and mesh.faces.dtype == torch.int32
    torch.testing.assert_close(mesh.vertices[:, :3], torch.tensor([1.0, 3.0, -2.0]).expand(2, 3, 3))
    if counted:
        assert mesh.vertex_counts.tolist() == [4, 3] and mesh.face_counts.tolist() == [2, 1]
        assert (mesh.vertices[1, 3:] == 0).all() and (mesh.faces[1, 1:] == 0).all()
    else:
        assert mesh.vertex_counts is None and mesh.face_counts is None
    stages = materialize_shape_subdivides(output["shape_subdivides"])
    assert len(stages) == 2
    expected_coords = before[[1, 2, 0]] if counted else before
    for index, stage in enumerate(stages):
        assert stage.shape == torch.Size([2, 2]) and stage._spatial_cache == {}
        assert stage._scale == (
            (Fraction(1),) * 3 if counted else (Fraction(1, 3), Fraction(2), Fraction(5, 7))
        )
        torch.testing.assert_close(stage.feats[:, 0], expected_features + index, rtol=0, atol=0)
        torch.testing.assert_close(stage.coords, expected_coords)
    torch.testing.assert_close(latent["coords"], before, rtol=0, atol=0)
    torch.testing.assert_close(latent["samples"], values_before, rtol=0, atol=0)


class ShapeDecodeSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "shape-decode-fixture.source",
            outputs=(
                OutputSpec("samples", TypeExpr.concrete(LATENT_TYPE)),
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(samples=decode_samples(), vae=ShapeDecoder())


@pytest.mark.parametrize("in_process", (False, True))
def test_ordinary_shape_decoder_hosts_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"shape_decode_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from test_trellis_shape_torch import ShapeDecodeSource\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (ShapeDecodeSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture.toml"
        fixture.write_text(
            '[pack]\nname="shape-decode-fixture"\npublisher="dinkster"\n'
            'namespaces=["shape-decode-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
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
                        worker_group=None if in_process else "shape-decode-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("shape-decode-fixture.source"),
                    "decode": GraphNode(
                        "VaeDecodeShapeTrellis",
                        {name: Link("source", name) for name in ("samples", "vae")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["decode"])
                mesh = materialize_mesh(result.outputs["decode"]["mesh"].resolve())
                stages = materialize_shape_subdivides(
                    result.outputs["decode"]["shape_subdivides"].resolve()
                )
                assert mesh.vertex_counts.tolist() == [4, 3]
                torch.testing.assert_close(mesh.vertices[1, 0], torch.tensor([1.0, 3.0, -2.0]))
                assert len(stages) == 2 and stages[0].shape == torch.Size([2, 2])
                torch.testing.assert_close(
                    stages[0].coords,
                    torch.tensor([[0, 1, 2, 3], [0, 3, 4, 5], [1, 5, 6, 7]], dtype=torch.int32),
                )
                assert all(stage._spatial_cache == {} for stage in stages)
        finally:
            await composer.close()

    asyncio.run(scenario())
