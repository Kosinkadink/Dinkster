"""CPU behavior and manifest coverage for native video CFG guidance."""

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
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.video_guidance import VideoLinearCFGGuidance, VideoTriangleCFGGuidance


def _model_patcher() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    return ModelPatcher(torch.nn.Linear(2, 2), torch.device("cpu"), torch.device("cpu"))


@pytest.mark.parametrize(
    ("node", "frames", "scales"),
    (
        (VideoLinearCFGGuidance, 1, (1.25,)),
        (VideoLinearCFGGuidance, 4, (1.25, 2.4166667, 3.5833333, 4.75)),
        (VideoLinearCFGGuidance, 5, (1.25, 2.125, 3.0, 3.875, 4.75)),
        (VideoTriangleCFGGuidance, 1, (1.25,)),
        (VideoTriangleCFGGuidance, 4, (1.25, 3.5833333, 3.5833333, 1.25)),
        (VideoTriangleCFGGuidance, 5, (1.25, 3.0, 4.75, 3.0, 1.25)),
    ),
)
def test_video_cfg_callbacks_match_hand_derived_scales_and_preserve_source(
    node: type[Node], frames: int, scales: tuple[float, ...]
) -> None:
    source = _model_patcher()
    source.model_options["transformer_options"]["preserved"] = {"value": 17}
    cond = torch.arange(frames * 2, dtype=torch.float32).reshape(frames, 1, 1, 2) + 3.0
    uncond = torch.arange(frames * 2, dtype=torch.float32).reshape(frames, 1, 1, 2) - 5.0
    patched = node.execute(model=source, min_cfg=1.25)["MODEL"]
    callback = patched.model_options["sampler_cfg_function"]
    result = callback({"cond": cond, "uncond": uncond, "cond_scale": 4.75})
    expected = uncond + torch.tensor(scales).reshape(frames, 1, 1, 1) * (cond - uncond)
    torch.testing.assert_close(result, expected)
    assert result.dtype == cond.dtype
    assert result.device == cond.device
    assert patched is not source
    assert "sampler_cfg_function" not in source.model_options
    assert source.model_options["transformer_options"] == {"preserved": {"value": 17}}
    assert patched.model_options["transformer_options"] == {"preserved": {"value": 17}}


class VideoGuidanceModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-guidance-fixture.source",
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(MODEL=_model_patcher())


class VideoGuidanceModelConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="video-guidance-fixture.consumer",
            inputs=(InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, model: Any) -> Mapping[str, object]:
        cond = torch.tensor([[[[3.0]]], [[[5.0]]], [[[7.0]]]])
        uncond = torch.tensor([[[[-5.0]]], [[[-3.0]]], [[[-1.0]]]])
        result = model.model_options["sampler_cfg_function"](
            {"cond": cond, "uncond": uncond, "cond_scale": 4.0}
        )
        return cls.outputs(value=float(result.sum()))


VIDEO_GUIDANCE_FIXTURE_NODES = (VideoGuidanceModelSource, VideoGuidanceModelConsumer)


@pytest.mark.parametrize("in_process", (True,))
def test_ordinary_manifest_executes_both_video_guidance_nodes_twice_in_both_host_modes(
    tmp_path: Path, in_process: bool
) -> None:
    from dinkster.compose import PackSpec, ServingComposer

    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": str(Path(__file__).parent),
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            module = "video_guidance_fixture_" + str(in_process)
            (tmp_path / f"{module}.py").write_text(
                "from test_video_guidance_torch import VIDEO_GUIDANCE_FIXTURE_NODES\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_comfy.device_routing import device_choices\n"
                "NODES = (*COMFY_NODES, *VIDEO_GUIDANCE_FIXTURE_NODES)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["video-guidance-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", f"{module}:NODES")
                .replace(
                    "dinkster_nodes_comfy.device_routing:device_choices",
                    f"{module}:device_choices",
                )
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    f"{module}:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            engine = composer.composition.make_engine(lambda _event: None)
            for node_type, expected in (
                ("VideoLinearCFGGuidance", 51.0),
                ("VideoTriangleCFGGuidance", 39.0),
            ):
                graph = Graph(
                    nodes={
                        "source": GraphNode("video-guidance-fixture.source"),
                        "patch": GraphNode(
                            node_type, {"model": Link("source", "MODEL"), "min_cfg": 1.0}
                        ),
                        "consumer": GraphNode(
                            "video-guidance-fixture.consumer", {"model": Link("patch", "MODEL")}
                        ),
                    }
                )
                for _ in range(2):
                    result = await engine.run(graph, ["consumer"])
                    assert result.outputs["consumer"]["value"].resolve() == pytest.approx(expected)
        finally:
            await composer.close()

    asyncio.run(scenario())
