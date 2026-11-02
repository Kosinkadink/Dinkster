"""Structure VAE batching and dense occupancy survive both ordinary hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_TYPE,
    VAE_TYPE,
    VOXEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_voxel,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.trellis_structure import VaeDecodeStructureTrellis2
from dinkster_values import decode_latent, encode_latent

from dinkster.compose import PackSpec, ServingComposer


def samples() -> Any:
    value = torch.zeros(3, 12, 1, 1, 1, dtype=torch.float64)
    value[:, 0, 0, 0, 0] = torch.tensor([1, 3, 5])
    value[:, 7, 0, 0, 0] = torch.tensor([7, 11, 15])
    value[:, 8:] = 999
    return value


class StructureVAE:
    vae_dtype = torch.float16

    def __init__(self) -> None:
        self.first_stage_model = self

    def prepare_decode(self, shape: Any) -> int:
        assert shape == (3, 8, 1, 1, 1)
        return 2

    def decode_structure(self, value: Any) -> Any:
        assert value.dtype == torch.float16 and value.shape[1] == 8
        output = torch.full((len(value), 1, 64, 64, 64), -1.0)
        for i in range(len(value)):
            z, x = int(value[i, 0, 0, 0, 0]), int(value[i, 7, 0, 0, 0])
            output[i, 0, z, 9, x] = 0.001
            output[i, 0, z + 1, 17, x + 2] = 0
            output[i, 0, z + 2, 23, x + 4] = -0.001
        return output


def check_voxel(voxel: Any, resolution: int) -> None:
    data = materialize_voxel(voxel).data
    assert data.dtype == torch.float32 and data.shape == (3, resolution, resolution, resolution)
    expected = (
        [[0, 1, 9, 7], [1, 3, 9, 11], [2, 5, 9, 15]]
        if resolution == 64
        else [[0, 0, 4, 3], [1, 1, 4, 5], [2, 2, 4, 7]]
    )
    torch.testing.assert_close(torch.nonzero(data), torch.tensor(expected))
    assert data.sum().item() == 3


@pytest.mark.parametrize("resolution", ("32", "64"))
@pytest.mark.parametrize("portable", (False, True))
def test_threshold_pooling_and_partial_batch(resolution: str, portable: bool) -> None:
    tensor = samples()
    before = tensor.clone()
    latent: Any = {"samples": tensor}
    if portable:
        latent = decode_latent(encode_latent(latent))
    result = VaeDecodeStructureTrellis2.execute(
        samples=latent, vae=StructureVAE(), resolution=resolution
    )
    check_voxel(result["voxel"], int(resolution))
    torch.testing.assert_close(tensor, before)


class StructureSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="structure-fixture.source",
            outputs=(
                OutputSpec("samples", TypeExpr.concrete(LATENT_TYPE)),
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Any:
        return cls.outputs(samples={"samples": samples()}, vae=StructureVAE())


class StructureInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="structure-fixture.inspect",
            inputs=(InputSpec("voxel", TypeExpr.concrete(VOXEL_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, voxel: Any) -> Any:
        check_voxel(voxel, 32)
        return cls.outputs(summary="three pooled occupancy voxels")


@pytest.mark.parametrize("in_process", (False, True))
def test_ordinary_manifest_structure_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        (tmp_path / "trellis_structure_fixture.py").write_text(
            "from test_trellis_structure_torch import StructureSource, StructureInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (StructureSource, StructureInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "structure.toml"
        fixture.write_text(
            '[pack]\nname="structure-fixture"\npublisher="dinkster"\n'
            'namespaces=["structure-fixture"]\n[pack.entry]\n'
            'nodes="trellis_structure_fixture:NODES"\n'
            'types="trellis_structure_fixture:register_inference_types"\n',
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
                        worker_group=None if in_process else "structure-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("structure-fixture.source"),
                    "decode": GraphNode(
                        "VaeDecodeStructureTrellis2",
                        {"samples": Link("source", "samples"), "vae": Link("source", "vae")},
                    ),
                    "inspect": GraphNode(
                        "structure-fixture.inspect", {"voxel": Link("decode", "voxel")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "three pooled occupancy voxels"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
