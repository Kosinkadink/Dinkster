"""CPU callback execution and ordinary manifest coverage for CFG guidance."""

from __future__ import annotations

import asyncio
import math
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
from dinkster_nodes_comfy.cfg_guidance import APG, TCFG, CFGNorm, CFGZeroStar, Mahiro, NAGuidance

from dinkster.compose import PackSpec, ServingComposer


class Patcher:
    def __init__(self) -> None:
        self.pre: Any = None
        self.post: Any = None
        self.attention: Any = None
        self.optimization_disabled = False

    def clone(self) -> Patcher:
        return Patcher()

    def set_model_sampler_pre_cfg_function(self, callback: Any) -> None:
        self.pre = callback

    def set_model_sampler_cfg_function(self, callback: Any) -> None:
        self.pre = callback

    def set_model_sampler_post_cfg_function(self, callback: Any) -> None:
        self.post = callback

    def set_model_attn1_output_patch(self, callback: Any) -> None:
        self.attention = callback

    def disable_model_cfg1_optimization(self) -> None:
        self.optimization_disabled = True


def test_cfg_guidance_callbacks_produce_asymmetric_tensor_outputs_without_source_mutation() -> None:
    source = Patcher()
    cond = torch.tensor([[[[6.0, -2.0]]]])
    uncond = torch.tensor([[[[1.0, 3.0]]]])
    x = torch.tensor([[[[2.0, 4.0]]]])
    for node, kwargs, expected in (
        (CFGZeroStar, {}, [11.0, -7.0]),
        (CFGNorm, {"strength": 0.7}, [4.2, -1.4]),
        (Mahiro, {}, [13.0, -6.0]),
    ):
        patched = node.execute(model=source, **kwargs)["MODEL"]
        result = patched.post(
            {
                "cond_scale": 2.5,
                "input": x,
                "cond_denoised": cond,
                "uncond_denoised": uncond,
                "denoised": cond * 2 - uncond,
            }
        )
        torch.testing.assert_close(result, torch.tensor(expected).reshape_as(cond))
        assert source.post is None
    pre_norm = CFGNorm.execute(model=source, strength=0.7, pre_cfg=True)["MODEL"]
    pre_result = pre_norm.pre({"cond": cond, "uncond": uncond, "cond_scale": 2.5})
    torch.testing.assert_close(pre_result, torch.tensor([8.25, -4.25]).reshape_as(cond))
    apg = APG.execute(model=source, eta=0.25, norm_threshold=2.0, momentum=0.5)["MODEL"]
    apg_result = apg.pre(
        {"conds_out": [cond, uncond, x], "sigma": torch.tensor([2.0]), "cond_scale": 2.5}
    )
    assert len(apg_result) == 3
    torch.testing.assert_close(
        apg_result[0],
        torch.tensor([3 + 0.1 * math.sqrt(2), 1 - 0.7 * math.sqrt(2)]).reshape_as(cond),
    )
    assert apg_result[1] is uncond
    assert apg_result[2] is x
    tcfg = TCFG.execute(model=source)["MODEL"]
    orthogonal_cond = torch.tensor([2.0, 0.0]).reshape_as(cond)
    orthogonal_uncond = torch.tensor([0.0, 1.0]).reshape_as(cond)
    tcfg_result = tcfg.pre(
        {
            "input": torch.zeros_like(x),
            "conds_out": [orthogonal_cond, orthogonal_uncond],
            "conds": [object(), object()],
        }
    )
    assert tcfg_result[0] is orthogonal_cond
    torch.testing.assert_close(tcfg_result[1], torch.zeros_like(uncond))
    nag = NAGuidance.execute(model=source, nag_scale=2.0, nag_alpha=0.5, nag_tau=1.5)["MODEL"]
    attention = torch.tensor([[[1.0, 2.0]], [[4.0, -3.0]]])
    before = attention.clone()
    result = nag.attention(attention.clone(), {"cond_or_uncond": [0, 1]})
    torch.testing.assert_close(result, torch.tensor([[[0.0, 2.75]], [[4.0, -3.0]]]))
    assert torch.equal(attention, before)
    assert nag.optimization_disabled


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cfg-guidance-fixture.source",
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(MODEL=Patcher())


class ModelConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="cfg-guidance-fixture.consumer",
            inputs=(InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, model: Patcher) -> Mapping[str, object]:
        args = {
            "cond_scale": 2.0,
            "input": torch.tensor([[[[2.0]]]]),
            "cond_denoised": torch.tensor([[[[5.0]]]]),
            "uncond_denoised": torch.tensor([[[[1.0]]]]),
            "denoised": torch.tensor([[[[9.0]]]]),
        }
        return cls.outputs(value=float(model.post(args).item()))


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_links_cfg_guidance_producer_and_consumer_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            module = "cfg_guidance_fixture_" + str(in_process)
            (tmp_path / f"{module}.py").write_text(
                "from test_cfg_guidance_torch import ModelConsumer, ModelSource\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, ModelSource, ModelConsumer)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["cfg-guidance-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", f"{module}:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    f"{module}:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("cfg-guidance-fixture.source"),
                    "patch": GraphNode("CFGZeroStar", {"model": Link("source", "MODEL")}),
                    "consumer": GraphNode(
                        "cfg-guidance-fixture.consumer", {"model": Link("patch", "MODEL")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                assert (await engine.run(graph, ["consumer"])).outputs["consumer"][
                    "value"
                ].resolve() == 13.0
        finally:
            await composer.close()

    asyncio.run(scenario())
