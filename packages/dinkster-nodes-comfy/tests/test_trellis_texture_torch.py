"""Texture decoding retains sparse ordering, subdivision scales and material channels."""

from __future__ import annotations

import asyncio
import importlib
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_TYPE,
    SHAPE_SUBDIVIDES_TYPE,
    VAE_TYPE,
    VOXEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_voxel,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy import register_inference_types
from dinkster_nodes_comfy.trellis import VaeDecodeTextureTrellis
from dinkster_values import TypeRegistry, decode_latent, encode_latent

from dinkster.compose import PackSpec, ServingComposer


def texture_inputs(*, counted: bool = True) -> tuple[Any, Any]:
    initialize_inference_process()
    sparse = importlib.import_module("dinkster_inference.ldm.trellis2.vae")
    tensor = torch.arange(128).reshape(2, 32, 2, 1).float() / 128
    coords = torch.tensor([[1, 5, 7, 11], [0, 2, 3, 4], [0, 6, 8, 9]], dtype=torch.int32)
    latent = {"samples": tensor, "coords": coords, "coord_resolution": 96, "model_frame": "z_up"}
    if counted:
        latent["coord_counts"] = torch.tensor([2, 1])
    else:
        latent["samples"] = tensor[:1]
        latent["coords"] = coords[1:]
    stages = [
        sparse.SparseTensor(
            feats=torch.full((len(coords), 4), float(index + 3)),
            coords=coords.clone(),
            shape=torch.Size([2, 4]),
            scale=(Fraction(1, 3), Fraction(2, 5), Fraction(index + 1, 7)),
        )
        for index in range(2)
    ]
    for stage in stages:
        stage._spatial_cache["discard"] = "not transported"
    return latent, stages


class TextureVAE:
    vae_dtype = torch.float16

    def __init__(self, counted: bool = True, width: int = 4, maximum: int = 11) -> None:
        self.first_stage_model = self
        self.counted, self.width, self.maximum = counted, width, maximum
        self.prepared: Any = None

    def prepare_decode(self, shape: Any, *, memory_required: int) -> None:
        self.prepared = (shape, memory_required)
        assert tuple(shape) == ((2, 32, 2, 1) if self.counted else (1, 32, 2, 1))
        assert memory_required == 2 * 1024**3 + 3 * (896 * 2 + 27 * 4)

    def decode_tex_slat(self, value: Any, stages: Any) -> Any:
        formats = importlib.import_module("dinkster_inference.latent_formats")
        tensor = torch.arange(128).reshape(2, 32, 2, 1).float() / 128
        expected = torch.stack([tensor[0, :, 0, 0], tensor[0, :, 1, 0]])
        expected_coords = [[0, 2, 3, 4], [0, 6, 8, 9]]
        if self.counted:
            expected = torch.cat([expected, tensor[1, :, 0, 0][None]])
            expected_coords.append([1, 5, 7, 11])
        expected = formats.Trellis2TexSLAT().process_out(expected).half()
        torch.testing.assert_close(value.feats, expected, rtol=0, atol=0)
        torch.testing.assert_close(value.coords, torch.tensor(expected_coords, dtype=torch.int32))
        assert value.feats.device.type == "cpu" and value.feats.dtype == self.vae_dtype
        for index, stage in enumerate(stages):
            assert stage.shape == torch.Size([2, 4]) and stage._spatial_cache == {}
            assert stage._scale == (Fraction(1, 3), Fraction(2, 5), Fraction(index + 1, 7))
            torch.testing.assert_close(stage.feats, torch.full((3, 4), float(index + 3)))
        coords = torch.tensor([[0, 2, 3, 5], [1, 7, 9, self.maximum]], dtype=torch.int32)
        if self.width == 3:
            coords = coords[:, 1:]
        elif self.width == 0:
            coords = coords[:0]
        colors = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1, 0, 0.25]])
        sparse = importlib.import_module("dinkster_inference.ldm.trellis2.vae")
        # The output coordinate width is intentionally independent of input batches.
        return sparse.SparseTensor(
            feats=colors[: len(coords)], coords=coords, shape=torch.Size([2, 6])
        )


def check_voxel(value: Any) -> None:
    voxel = materialize_voxel(value)
    assert voxel.resolution == 1536
    torch.testing.assert_close(
        voxel.data, torch.tensor([[0, 2, 5, 1532], [1, 7, 11, 1526]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        voxel.colors,
        torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1, 0, 0.25]]),
    )


@pytest.mark.parametrize("counted", (False, True))
@pytest.mark.parametrize("portable", (False, True))
def test_texture_decode_transport_and_sparse_order(counted: bool, portable: bool) -> None:
    latent, stages = texture_inputs(counted=counted)
    before = latent["samples"].clone()
    if portable:
        latent = decode_latent(encode_latent(latent))
        registry = TypeRegistry()
        register_inference_types(registry)
        spec = registry.spec(SHAPE_SUBDIVIDES_TYPE)
        stages = spec.decode(spec.encode(stages))
    output = VaeDecodeTextureTrellis.execute(
        samples=latent, vae=TextureVAE(counted), shape_subdivides=stages
    )
    check_voxel(output["voxel_colors"])
    if not portable:
        torch.testing.assert_close(latent["samples"], before)
        assert stages[0]._spatial_cache == {"discard": "not transported"}


@pytest.mark.parametrize(
    "width,maximum,resolution", [(4, 255, 256), (3, 256, 512), (4, 2048, 2049), (0, 11, 1024)]
)
@pytest.mark.parametrize("frame", ("y_up", "z_up"))
def test_texture_resolution_inference_and_coordinate_frame(
    width: int, maximum: int, resolution: int, frame: str
) -> None:
    latent, stages = texture_inputs()
    del latent["coord_resolution"]
    latent["model_frame"] = frame
    result = VaeDecodeTextureTrellis.execute(
        samples=latent, vae=TextureVAE(width=width, maximum=maximum), shape_subdivides=stages
    )
    voxel = materialize_voxel(result["voxel_colors"])
    assert voxel.resolution == resolution and voxel.colors.shape == (0 if width == 0 else 2, 6)
    expected = torch.tensor([[0, 2, 3, 5], [1, 7, 9, maximum]], dtype=torch.int32)
    if frame == "z_up":
        expected = torch.tensor(
            [[0, 2, 5, resolution - 4], [1, 7, maximum, resolution - 10]], dtype=torch.int32
        )
    if width == 3:
        expected = expected[:, 1:]
    if width == 0:
        expected = expected[:0]
    torch.testing.assert_close(voxel.data, expected)


class TextureSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="texture-fixture.source",
            outputs=(
                OutputSpec("samples", TypeExpr.concrete(LATENT_TYPE)),
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                OutputSpec("shape_subdivides", TypeExpr.concrete(SHAPE_SUBDIVIDES_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        latent, stages = texture_inputs()
        return cls.outputs(samples=latent, vae=TextureVAE(), shape_subdivides=stages)


class TextureInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="texture-fixture.inspect",
            inputs=(InputSpec("voxel", TypeExpr.concrete(VOXEL_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, voxel: Any) -> Any:
        check_voxel(voxel)
        return cls.outputs(summary="six channels and two sparse batches")


@pytest.mark.parametrize("in_process", (False, True))
def test_ordinary_manifest_texture_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "trellis_texture_fixture.py").write_text(
            "from test_trellis_texture_torch import TextureSource, TextureInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (TextureSource, TextureInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "texture.toml"
        fixture.write_text(
            '[pack]\nname="texture-fixture"\npublisher="dinkster"\nnamespaces=["texture-fixture"]\n[pack.entry]\nnodes="trellis_texture_fixture:NODES"\ntypes="trellis_texture_fixture:register_inference_types"\n',
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
                        worker_group=None if in_process else "texture-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("texture-fixture.source"),
                    "decode": GraphNode(
                        "VaeDecodeTextureTrellis",
                        {
                            name: Link("source", name)
                            for name in ("samples", "vae", "shape_subdivides")
                        },
                    ),
                    "inspect": GraphNode(
                        "texture-fixture.inspect", {"voxel": Link("decode", "voxel_colors")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "six channels and two sparse batches"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
