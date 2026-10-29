"""Gaussian counts and passthrough through native CPU execution."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    SPLAT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.gaussian_splat import GetSplatCount

from dinkster.compose import PackSpec, ServingComposer


def _source(case: str) -> SimpleNamespace:
    size = 0 if case == "empty" else 3
    return SimpleNamespace(
        positions=torch.arange(18, dtype=torch.float64).reshape(2, 3, 3)[:, :size],
        scales=torch.ones(2, size, 3),
        rotations=torch.tensor([1.0, 0, 0, 0]).expand(2, size, 4),
        opacities=torch.ones(2, size, 1),
        sh=torch.zeros(2, size, 1, 3),
        counts=(
            torch.tensor([1, 3])
            if case == "nonuniform"
            else torch.zeros(2, dtype=torch.int32)
            if case == "zero-counts"
            else None
        ),
    )


@pytest.mark.parametrize(
    "case,expected", [("uniform", 6), ("nonuniform", 4), ("empty", 0), ("zero-counts", 0)]
)
def test_splat_count_preserves_input_and_counts_only_valid_rows(case: str, expected: int) -> None:
    source = _source(case)
    result = GetSplatCount.execute(splat=source)
    assert result["splat"] is source
    assert result["count"] == expected
    assert source.positions.flatten().tolist() == ([] if case == "empty" else list(range(18)))


class SplatSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "splat-count-source.source",
            inputs=(InputSpec("case", TypeExpr.concrete(CORE_STRING)),),
            outputs=(OutputSpec("splat", TypeExpr.concrete(SPLAT_TYPE)),),
        )

    @classmethod
    def execute(cls, *, case: str) -> dict[str, object]:
        return cls.outputs(splat=_source(case))


@pytest.mark.parametrize("in_process", [True, False])
def test_splat_count_executes_through_the_ordinary_manifest(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        module = f"splat_count_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_get_splat_count_torch import SplatSource\nNODES = (SplatSource,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="splat-count-source"\npublisher="dinkster"\n'
            'namespaces=["splat-count-source"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            }
        )
        try:
            for path in (manifest, fixture):
                await composer.add_pack(PackSpec(manifest=path, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            for case, expected in (
                ("uniform", 6),
                ("nonuniform", 4),
                ("empty", 0),
                ("zero-counts", 0),
            ):
                graph = Graph(
                    {
                        "source": GraphNode("splat-count-source.source", {"case": case}),
                        "count": GraphNode("GetSplatCount", {"splat": Link("source", "splat")}),
                    }
                )
                for _attempt in range(2):
                    result = await engine.run(graph, ["count"])
                    assert result.outputs["count"]["count"].resolve() == expected
                    assert result.outputs["count"]["splat"].type_id == SPLAT_TYPE
        finally:
            await composer.close()

    asyncio.run(scenario())
