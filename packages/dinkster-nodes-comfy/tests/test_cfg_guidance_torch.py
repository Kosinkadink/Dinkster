"""CPU callback execution and ordinary manifest coverage for CFG guidance."""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
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
from dinkster_nodes_comfy.cfg_guidance import (
    APG,
    TCFG,
    CFGNorm,
    CFGZeroStar,
    Mahiro,
    NAGuidance,
    RenormCFG,
)

from dinkster.compose import PackSpec, ServingComposer


class Patcher:
    def __init__(self) -> None:
        self.model = SimpleNamespace(diffusion_model=SimpleNamespace(in_channels=1))
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


class RenormConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "cfg-guidance-fixture.renorm",
            inputs=(InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, model: Patcher) -> Mapping[str, object]:
        for timestep, expected in (
            (
                [49.0, 51.0],
                [12.5 / math.sqrt(61), 15.0 / math.sqrt(61), 7.0, -9.0, 1.0, 1.0, -3.0, 2.0],
            ),
            ([50.0, 49.0], [3.0, 4.0, 7.0, -9.0, 6.0, 8.0, -3.0, 2.0]),
        ):
            actual = model.pre(
                {
                    "cond_denoised": torch.tensor(
                        [3.0, 4.0, 7.0, -9.0, 6.0, 8.0, -3.0, 2.0]
                    ).reshape(2, 2, 1, 2),
                    "uncond_denoised": torch.tensor(
                        [1.0, 2.0, -5.0, 8.0, 11.0, 15.0, 4.0, -1.0]
                    ).reshape(2, 2, 1, 2),
                    "cond_scale": 2.0,
                    "timestep": torch.tensor(timestep),
                    "input": torch.zeros((2, 2, 1, 2)),
                }
            )
            torch.testing.assert_close(
                actual, -torch.tensor(expected).reshape(2, 2, 1, 2), rtol=1e-6, atol=0
            )
        return cls.outputs(value=2.0)


@pytest.mark.parametrize("in_process", (True,))
def test_ordinary_manifest_links_cfg_guidance_producer_and_consumer_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            module = "cfg_guidance_fixture_" + str(in_process)
            (tmp_path / f"{module}.py").write_text(
                "from test_cfg_guidance_torch import ModelConsumer, ModelSource, RenormConsumer\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (ModelSource, ModelConsumer, RenormConsumer)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "cfg-guidance-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["cfg-guidance-fixture"]\n[pack.entry]\n'
                f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            for pack in (manifest, fixture_manifest):
                await composer.add_pack(
                    PackSpec(manifest=pack, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=pack,
                        worker_group="cfg-guidance-fixture",
                        group_manifests=(manifest, fixture_manifest),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("cfg-guidance-fixture.source"),
                    "patch": GraphNode("CFGZeroStar", {"model": Link("source", "MODEL")}),
                    "consumer": GraphNode(
                        "cfg-guidance-fixture.consumer", {"model": Link("patch", "MODEL")}
                    ),
                    "renorm": GraphNode(
                        "RenormCFG",
                        {"model": Link("source", "MODEL"), "cfg_trunc": 50, "renorm_cfg": 0.5},
                    ),
                    "renorm-consumer": GraphNode(
                        "cfg-guidance-fixture.renorm", {"model": Link("renorm", "MODEL")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                outputs = (await engine.run(graph, ["consumer", "renorm-consumer"])).outputs
                assert outputs["consumer"]["value"].resolve() == 13.0
                assert outputs["renorm-consumer"]["value"].resolve() == 2.0
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("sample_count", (1, 2))
@pytest.mark.parametrize(
    ("timestep", "renorm_cfg", "eps"),
    [
        (49.0, 0.0, (5.0, 6.0)),
        (49.0, 0.5, (12.5 / math.sqrt(61), 15.0 / math.sqrt(61))),
        (50.0, 0.5, (3.0, 4.0)),
        (51.0, 0.5, (3.0, 4.0)),
    ],
)
def test_renorm_limits_guided_channels_before_truncation_only(
    timestep: float, renorm_cfg: float, eps: tuple[float, float], sample_count: int
) -> None:
    source = Patcher()
    patched = RenormCFG.execute(model=source, cfg_trunc=50, renorm_cfg=renorm_cfg)["MODEL"]
    positive = torch.tensor([3.0, 4.0, 7.0, -9.0, 6.0, 8.0, -3.0, 2.0]).reshape(2, 2, 1, 2)
    negative = torch.tensor([1.0, 2.0, -5.0, 8.0, 11.0, 15.0, 4.0, -1.0]).reshape(2, 2, 1, 2)
    expected = [*eps, 7.0, -9.0]
    expected.extend([1.0, 1.0, -3.0, 2.0] if timestep < 50 else [6.0, 8.0, -3.0, 2.0])
    value = patched.pre(
        {
            "cond_denoised": positive[:sample_count],
            "uncond_denoised": negative[:sample_count],
            "cond_scale": 2.0,
            "timestep": torch.tensor([timestep, 100 - timestep])[:sample_count],
            "input": torch.zeros((sample_count, 2, 1, 2)),
        }
    )
    torch.testing.assert_close(
        value, -torch.tensor(expected).reshape(2, 2, 1, 2)[:sample_count], rtol=1e-6, atol=0
    )
    assert positive.flatten().tolist() == [3.0, 4.0, 7.0, -9.0, 6.0, 8.0, -3.0, 2.0]
    assert negative.flatten().tolist() == [1.0, 2.0, -5.0, 8.0, 11.0, 15.0, 4.0, -1.0]
    assert patched is not source and source.pre is None
