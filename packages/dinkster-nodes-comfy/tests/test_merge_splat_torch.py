"""CPU merge geometry and execution through the ordinary native manifest."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dinkster_api.v1 import (
    SPLAT_TYPE,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_splat,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.gaussian_splat import MergeSplat

from dinkster.compose import PackSpec, ServingComposer


def _sources() -> tuple[SimpleNamespace, SimpleNamespace]:
    def value(offset: int, size: int, degree: int, counts: torch.Tensor | None) -> SimpleNamespace:
        return SimpleNamespace(
            positions=torch.arange(2 * size * 3, dtype=torch.float64).reshape(2, size, 3) + offset,
            scales=torch.full((2, size, 3), offset + 1, dtype=torch.float32),
            rotations=torch.tensor([1.0, 0, 0, 0] if offset == 10 else [0.0, 1, 0, 0])
            .expand(2, size, 4)
            .clone(),
            opacities=torch.full((2, size, 1), offset / 100, dtype=torch.float32),
            sh=torch.arange(2 * size * degree * 3, dtype=torch.float32).reshape(2, size, degree, 3)
            + offset,
            counts=counts,
        )

    return value(10, 3, 1, torch.tensor([1, 3], dtype=torch.int32)), value(40, 2, 4, None)


def _assert_merged(value: object) -> None:
    result = materialize_splat(value)
    assert result.positions.dtype == torch.float64
    assert result.scales.dtype == torch.float32
    assert result.counts.dtype == torch.int64
    assert result.counts.tolist() == [3, 5]
    assert result.positions.tolist() == [
        [[10, 11, 12], [40, 41, 42], [43, 44, 45], [0, 0, 0], [0, 0, 0]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27], [46, 47, 48], [49, 50, 51]],
    ]
    for batch, first_size, valid_size in ((0, 1, 3), (1, 3, 5)):
        assert result.scales[batch, :first_size].tolist() == [[11] * 3] * first_size
        assert result.scales[batch, first_size:valid_size].tolist() == [[41] * 3] * 2
        assert result.rotations[batch, :valid_size].tolist() == (
            [[1, 0, 0, 0]] * first_size + [[0, 1, 0, 0]] * 2
        )
        torch.testing.assert_close(
            result.opacities[batch, :valid_size, 0],
            torch.tensor([0.1] * first_size + [0.4] * 2),
            rtol=0,
            atol=0,
        )
        assert result.sh[batch, :first_size, 1:].count_nonzero().item() == 0
        expected_dc = [
            [10 + (batch * 3 + index) * 3 + axis for axis in range(3)]
            for index in range(first_size)
        ]
        assert result.sh[batch, :first_size, 0].tolist() == expected_dc
        expected_sh = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) + 40 + batch * 24
        torch.testing.assert_close(
            result.sh[batch, first_size:valid_size], expected_sh, rtol=0, atol=0
        )
        for name in ("positions", "scales", "rotations", "opacities", "sh"):
            assert getattr(result, name)[batch, valid_size:].count_nonzero().item() == 0


def test_merge_splat_truncates_real_counts_pads_sh_and_preserves_sources() -> None:
    first, second = _sources()
    snapshots = [
        {key: item.clone() if item is not None else None for key, item in vars(value).items()}
        for value in (first, second)
    ]
    # The body consumes mapping order, not lexical slot order.
    result = MergeSplat.execute(splats={"splat10": first, "splat2": second, "splat31": None})
    _assert_merged(result["splat"])
    for value, snapshot in zip((first, second), snapshots, strict=True):
        for name, before in snapshot.items():
            after = getattr(value, name)
            if before is None:
                assert after is None
            else:
                torch.testing.assert_close(after, before, rtol=0, atol=0)


@pytest.mark.parametrize("case,size", [("uniform", 5), ("empty", 0), ("zero-counts", 0)])
def test_merge_splat_uniform_lengths_omit_counts(case: str, size: int) -> None:
    first, second = _sources()
    first.counts = None
    if case == "empty":
        for value in (first, second):
            for name in ("positions", "scales", "rotations", "opacities", "sh"):
                setattr(value, name, getattr(value, name)[:, :0])
    elif case == "zero-counts":
        first.counts = second.counts = torch.zeros(2, dtype=torch.int64)
    merged = MergeSplat.execute(splats={"splat0": first, "splat1": second})["splat"]
    result = materialize_splat(merged)
    assert result.counts is None
    assert result.positions.shape == (2, size, 3)
    assert result.sh.shape == (2, size, 4, 3)
    if case == "uniform":
        assert result.positions[0, :, 0].tolist() == [10, 13, 16, 40, 43]
        assert result.positions[1, :, 0].tolist() == [19, 22, 25, 46, 49]


def test_merge_splat_rejects_missing_inputs_and_batch_mismatch() -> None:
    with pytest.raises(ValueError, match="connect at least one splat"):
        MergeSplat.execute(splats={"splat0": None})
    first, second = _sources()
    for name in ("positions", "scales", "rotations", "opacities", "sh"):
        setattr(second, name, getattr(second, name)[:1])
    with pytest.raises(ValueError, match=r"batch size mismatch \(2 vs 1\)"):
        MergeSplat.execute(splats={"splat0": first, "splat1": second})


class SplatSources(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "splat-merge-source.source",
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(SPLAT_TYPE)) for name in ("first", "second")
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        first, second = _sources()
        return cls.outputs(first=first, second=second)


@pytest.mark.parametrize("in_process", [True, False])
def test_merge_splat_executes_through_the_ordinary_manifest(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        module = f"splat_merge_fixture_{in_process}"
        (tmp_path / f"{module}.py").write_text(
            "from dinkster_api.v1 import register_inference_types\n"
            "from test_merge_splat_torch import SplatSources\nNODES = (SplatSources,)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="splat-merge-source"\npublisher="dinkster"\n'
            'namespaces=["splat-merge-source"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            }
        )
        try:
            for path in (Path(__file__).resolve().parents[1] / "dinkster-pack.toml", fixture):
                await composer.add_pack(PackSpec(manifest=path, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                {
                    "source": GraphNode("splat-merge-source.source"),
                    "merge": GraphNode(
                        "MergeSplat",
                        {
                            "splats.splat0": Link("source", "first"),
                            "splats.splat1": Link("source", "second"),
                        },
                    ),
                }
            )
            for _attempt in range(2):
                result = await engine.run(graph, ["merge"])
                output = result.outputs["merge"]["splat"]
                assert output.type_id == SPLAT_TYPE
                _assert_merged(output.resolve())
        finally:
            await composer.close()

    asyncio.run(scenario())
