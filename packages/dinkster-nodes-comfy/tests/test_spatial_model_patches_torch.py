"""CPU behavior and hosted transport coverage for spatial model patches."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_FLOAT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.spatial_model_patches import (
    HyperTile,
    PatchModelAddDownscale,
    TomePatchModel,
)

from dinkster.compose import PackSpec, ServingComposer


class Sampling:
    def percent_to_sigma(self, percent: float) -> float:
        return 1.0 - percent


class Patcher:
    def __init__(self) -> None:
        self.sampling = Sampling()
        self.attn_input: Any = None
        self.attn_output: Any = None
        self.input_patch: Any = None
        self.after_skip_patch: Any = None
        self.output_patch: Any = None

    def clone(self) -> Patcher:
        result = Patcher()
        result.sampling = self.sampling
        result.attn_input = self.attn_input
        result.attn_output = self.attn_output
        result.input_patch = self.input_patch
        result.after_skip_patch = self.after_skip_patch
        result.output_patch = self.output_patch
        return result

    def get_model_object(self, name: str) -> Any:
        assert name == "model_sampling"
        return self.sampling

    def set_model_attn1_patch(self, patch: Any) -> None:
        self.attn_input = patch

    def set_model_attn1_output_patch(self, patch: Any) -> None:
        self.attn_output = patch

    def set_model_input_block_patch(self, patch: Any) -> None:
        self.input_patch = patch

    def set_model_input_block_patch_after_skip(self, patch: Any) -> None:
        self.after_skip_patch = patch

    def set_model_output_block_patch(self, patch: Any) -> None:
        self.output_patch = patch


def test_tome_merges_and_unmerges_seeded_asymmetric_tokens_without_mutating_source() -> None:
    source = Patcher()
    patched = TomePatchModel.execute(model=source, ratio=0.3)["MODEL"]
    query = torch.arange(30, dtype=torch.float32).reshape(1, 15, 2) + 1
    torch.manual_seed(7)
    merged, key, value = patched.attn_input(
        query, query + 100, query + 200, {"original_shape": (1, 2, 3, 5)}
    )
    restored = patched.attn_output(merged, {})
    assert source.attn_input is None
    assert source.attn_output is None
    assert merged.shape == (1, 11, 2)
    assert key.shape == query.shape
    assert value.shape == query.shape
    assert restored.shape == query.shape
    assert torch.equal(
        merged[0, -4:], torch.tensor([[3.0, 4.0], [1.0, 2.0], [5.0, 6.0], [15.0, 16.0]])
    )
    assert torch.equal(restored[0, 5:10], torch.full((5, 2), 15.0).add(torch.tensor([0.0, 1.0])))


def test_tome_bypasses_zero_ratio_and_downsampled_attention() -> None:
    query = torch.arange(30, dtype=torch.float32).reshape(1, 15, 2) + 1
    for ratio, shape in ((0.0, (1, 2, 3, 5)), (0.3, (1, 2, 6, 10))):
        patched = TomePatchModel.execute(model=Patcher(), ratio=ratio)["MODEL"]
        merged, _, _ = patched.attn_input(query, query, query, {"original_shape": shape})
        restored = patched.attn_output(merged, {})
        assert merged is query
        assert restored is query


def test_hypertile_round_trips_asymmetric_tiles_and_preserves_source() -> None:
    source = Patcher()
    patched = HyperTile.execute(
        model=source, tile_size=32, swap_size=2, max_depth=0, scale_depth=False
    )["MODEL"]
    query = torch.arange(192, dtype=torch.float32).reshape(1, 96, 2)
    torch.manual_seed(3)
    tiled, key, value = patched.attn_input(
        query, query + 100, query + 200, {"original_shape": (1, 2, 8, 12)}
    )
    restored = patched.attn_output(tiled, {})
    assert source.attn_input is None
    assert source.attn_output is None
    assert tiled.shape == (6, 16, 2)
    assert key.shape == query.shape
    assert value.shape == query.shape
    assert torch.equal(restored, query)

    bypass, _, _ = patched.attn_input(query, query, query, {"original_shape": (1, 2, 7, 13)})
    assert bypass is query
    assert patched.attn_output(bypass, {}) is query


@pytest.mark.parametrize("after_skip", (False, True))
def test_downscale_respects_sigma_window_skip_branch_and_restores_skip_shape(
    after_skip: bool,
) -> None:
    source = Patcher()
    patched = PatchModelAddDownscale.execute(
        model=source,
        block_number=3,
        downscale_factor=2.0,
        start_percent=0.0,
        end_percent=0.35,
        downscale_after_skip=after_skip,
        downscale_method="nearest-exact",
        upscale_method="nearest-exact",
    )["MODEL"]
    installed = patched.after_skip_patch if after_skip else patched.input_patch
    assert source.input_patch is None
    assert source.after_skip_patch is None
    assert source.output_patch is None
    hidden = torch.arange(35, dtype=torch.float32).reshape(1, 1, 5, 7)
    active = installed(hidden, {"block": ("input", 3), "sigmas": [torch.tensor(1.0)]})
    ending = installed(
        hidden, {"block": ("input", 3), "sigmas": [torch.tensor(0.65, dtype=torch.float64)]}
    )
    wrong_block = installed(hidden, {"block": ("input", 2), "sigmas": [torch.tensor(0.8)]})
    outside_window = installed(hidden, {"block": ("input", 3), "sigmas": [torch.tensor(0.2)]})
    resized, skip = patched.output_patch(active, hidden, {})
    assert active.shape == (1, 1, 2, 4)
    assert torch.equal(active, torch.tensor([[[[7.0, 9.0, 11.0, 13.0], [21.0, 23.0, 25.0, 27.0]]]]))
    assert torch.equal(ending, active)
    assert wrong_block is hidden
    assert outside_window is hidden
    assert resized.shape == hidden.shape
    assert skip is hidden


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="spatial-fixture.model",
            inputs=(),
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(MODEL=Patcher())


class ModelResult(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="spatial-fixture.result",
            inputs=(
                InputSpec("original", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
            ),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, original: Patcher, model: Patcher) -> Mapping[str, object]:
        assert original.attn_input is None
        assert original.attn_output is None
        query = torch.arange(30, dtype=torch.float32).reshape(1, 15, 2) + 1
        torch.manual_seed(7)
        merged, _, _ = model.attn_input(query, query, query, {"original_shape": (1, 2, 3, 5)})
        restored = model.attn_output(merged, {})
        assert merged.shape == (1, 11, 2)
        assert torch.equal(
            merged[0, -4:], torch.tensor([[3.0, 4.0], [1.0, 2.0], [5.0, 6.0], [15.0, 16.0]])
        )
        assert torch.equal(
            restored[0, 5:10], torch.full((5, 2), 15.0).add(torch.tensor([0.0, 1.0]))
        )
        return cls.outputs(value=float(restored.sum()))


class HyperTileResult(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="spatial-fixture.hypertile-result",
            inputs=(
                InputSpec("original", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
            ),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, original: Patcher, model: Patcher) -> Mapping[str, object]:
        assert original.attn_input is None
        assert original.attn_output is None
        query = torch.arange(192, dtype=torch.float32).reshape(1, 96, 2)
        tiled, _, _ = model.attn_input(query, query, query, {"original_shape": (1, 2, 8, 12)})
        restored = model.attn_output(tiled, {})
        assert tiled.shape == (6, 16, 2)
        assert torch.equal(restored, query)
        return cls.outputs(value=float(restored.sum()))


class DownscaleResult(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="spatial-fixture.downscale-result",
            inputs=(
                InputSpec("original", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
            ),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, original: Patcher, model: Patcher) -> Mapping[str, object]:
        assert original.input_patch is None
        assert original.after_skip_patch is None
        assert original.output_patch is None
        hidden = torch.arange(35, dtype=torch.float32).reshape(1, 1, 5, 7)
        reduced = model.after_skip_patch(
            hidden, {"block": ("input", 3), "sigmas": [torch.tensor(1.0)]}
        )
        restored, _ = model.output_patch(reduced, hidden, {})
        assert reduced.shape == (1, 1, 2, 4)
        assert torch.equal(
            reduced, torch.tensor([[[[7.0, 9.0, 11.0, 13.0], [21.0, 23.0, 25.0, 27.0]]]])
        )
        assert restored.shape == hidden.shape
        assert torch.equal(
            restored,
            torch.tensor(
                [
                    [
                        [
                            [7.0, 7.0, 9.0, 11.0, 11.0, 13.0, 13.0],
                            [7.0, 7.0, 9.0, 11.0, 11.0, 13.0, 13.0],
                            [21.0, 21.0, 23.0, 25.0, 25.0, 27.0, 27.0],
                            [21.0, 21.0, 23.0, 25.0, 25.0, 27.0, 27.0],
                            [21.0, 21.0, 23.0, 25.0, 25.0, 27.0, 27.0],
                        ]
                    ]
                ]
            ),
        )
        return cls.outputs(value=float(restored.sum()))


FIXTURE_NODES = (ModelSource, ModelResult, HyperTileResult, DownscaleResult)


@pytest.mark.parametrize("in_process", (True,))
def test_complete_manifest_transports_an_ordinary_tome_model_graph_in_both_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            manifest = Path(__file__).parents[1] / "dinkster-pack.toml"
            (tmp_path / "spatial_fixture_sources.py").write_text(
                "from test_spatial_model_patches_torch import FIXTURE_NODES\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_comfy.device_routing import device_choices\n"
                "NODES = (*COMFY_NODES, *FIXTURE_NODES)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["spatial-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "spatial_fixture_sources:NODES")
                .replace(
                    "dinkster_nodes_comfy.device_routing:device_choices",
                    "spatial_fixture_sources:device_choices",
                )
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "spatial_fixture_sources:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("spatial-fixture.model", {}),
                    "patch": GraphNode(
                        "TomePatchModel", {"model": Link("source", "MODEL"), "ratio": 0.3}
                    ),
                    "result": GraphNode(
                        "spatial-fixture.result",
                        {
                            "original": Link("source", "MODEL"),
                            "model": Link("patch", "MODEL"),
                        },
                    ),
                    "hypertile": GraphNode(
                        "HyperTile",
                        {
                            "model": Link("source", "MODEL"),
                            "tile_size": 32,
                            "swap_size": 2,
                            "max_depth": 0,
                            "scale_depth": False,
                        },
                    ),
                    "hypertile_result": GraphNode(
                        "spatial-fixture.hypertile-result",
                        {
                            "original": Link("source", "MODEL"),
                            "model": Link("hypertile", "MODEL"),
                        },
                    ),
                    "downscale": GraphNode(
                        "PatchModelAddDownscale",
                        {
                            "model": Link("source", "MODEL"),
                            "block_number": 3,
                            "downscale_factor": 2.0,
                            "start_percent": 0.0,
                            "end_percent": 0.35,
                            "downscale_after_skip": True,
                            "downscale_method": "nearest-exact",
                            "upscale_method": "nearest-exact",
                        },
                    ),
                    "downscale_result": GraphNode(
                        "spatial-fixture.downscale-result",
                        {
                            "original": Link("source", "MODEL"),
                            "model": Link("downscale", "MODEL"),
                        },
                    ),
                }
            )
            for _ in range(2):
                result = await composer.composition.make_engine(lambda _event: None).run(
                    graph, ["result", "hypertile_result", "downscale_result"]
                )
                assert result.outputs["result"]["value"].resolve() == 465.0
                assert result.outputs["hypertile_result"]["value"].resolve() == 18336.0
                assert result.outputs["downscale_result"]["value"].resolve() == 649.0
        finally:
            await composer.close()

    asyncio.run(scenario())
