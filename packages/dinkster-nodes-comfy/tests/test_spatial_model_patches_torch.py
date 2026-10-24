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
        return Patcher()

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
    active = installed(hidden, {"block": ("input", 3), "sigmas": [torch.tensor(0.8)]})
    wrong_block = installed(hidden, {"block": ("input", 2), "sigmas": [torch.tensor(0.8)]})
    outside_window = installed(hidden, {"block": ("input", 3), "sigmas": [torch.tensor(0.2)]})
    resized, skip = patched.output_patch(active, hidden, {})
    assert active.shape == (1, 1, 2, 4)
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
            inputs=(InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, model: Patcher) -> Mapping[str, object]:
        return cls.outputs(value=float(model.attn_input is not None))


FIXTURE_NODES = (ModelSource, ModelResult)


@pytest.mark.parametrize("in_process", (True, False))
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
                "NODES = (*COMFY_NODES, *FIXTURE_NODES)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["spatial-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "spatial_fixture_sources:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "spatial_fixture_sources:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            result = await composer.composition.make_engine(lambda _event: None).run(
                Graph(
                    nodes={
                        "source": GraphNode("spatial-fixture.model", {}),
                        "patch": GraphNode(
                            "TomePatchModel", {"model": Link("source", "MODEL"), "ratio": 0.3}
                        ),
                        "result": GraphNode(
                            "spatial-fixture.result", {"model": Link("patch", "MODEL")}
                        ),
                    }
                ),
                ["result"],
            )
            assert result.outputs["result"]["value"].resolve() == 1.0
        finally:
            await composer.close()

    asyncio.run(scenario())
