"""CPU callback and ordinary-manifest coverage for attention guidance patches."""

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
from dinkster_nodes_comfy.attention_guidance import (
    PerturbedAttentionGuidance,
    SelfAttentionGuidance,
    SkipLayerGuidanceDiT,
    SkipLayerGuidanceDiTSimple,
)

from dinkster.compose import PackSpec, ServingComposer


class Sampling:
    def percent_to_sigma(self, percent: float) -> float:
        return 10.0 * (1.0 - percent)


class Patcher:
    def __init__(self, sampling: Sampling | None = None) -> None:
        self.sampling = sampling or Sampling()
        self.post: list[Any] = []
        self.attention: Any = None
        self.calc: Any = None
        self.disable_cfg1_optimization = False

    def clone(self) -> Patcher:
        clone = Patcher(self.sampling)
        clone.post = self.post.copy()
        clone.attention = self.attention
        clone.calc = self.calc
        clone.disable_cfg1_optimization = self.disable_cfg1_optimization
        return clone

    def get_model_object(self, name: str) -> Sampling:
        assert name == "model_sampling"
        return self.sampling

    def set_model_sampler_post_cfg_function(
        self, callback: Any, disable_cfg1_optimization: bool = False
    ) -> None:
        self.post.append(callback)
        self.disable_cfg1_optimization = disable_cfg1_optimization

    def set_model_attn1_replace(self, callback: Any, *location: object) -> None:
        assert location == ("middle", 0, 0)
        self.attention = callback

    def set_model_sampler_calc_cond_batch_function(self, callback: Any) -> None:
        self.calc = callback


class FakeAttention:
    @staticmethod
    def optimized_attention_for_device(device: torch.device) -> Any:
        def optimized(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: object
        ) -> torch.Tensor:
            return v + 2

        return optimized


class FakeModelPatcher:
    calls: list[tuple[str, str, int]] = []

    @classmethod
    def set_model_options_patch_replace(
        cls, options: dict[str, Any], patch: Any, name: str, block: str, layer: int
    ) -> dict[str, Any]:
        result = options.copy()
        result.setdefault("patches", []).append((patch, name, block, layer))
        cls.calls.append((name, block, layer))
        return result


class FakeSamplers:
    calls: list[tuple[Any, list[Any], dict[str, Any]]] = []

    @classmethod
    def calc_cond_batch(
        cls,
        model: Any,
        conds: list[Any],
        x: torch.Tensor,
        sigma: torch.Tensor,
        options: dict[str, Any],
    ) -> list[torch.Tensor]:
        cls.calls.append((model, conds, options))
        if conds == ["uncond"]:
            return [x - 4]
        if conds == ["cond"]:
            return [torch.full_like(x, 4.0)]
        if conds == ["cond", None]:
            return [torch.full_like(x, 7.0), torch.zeros_like(x)]
        if conds == [None, "uncond"]:
            return [torch.zeros_like(x), torch.full_like(x, -3.0)]
        return [torch.full_like(x, 2.0), torch.full_like(x, -2.0)]


def fake_modules() -> tuple[Any, Any, Any, Any]:
    return torch, FakeAttention, FakeModelPatcher, FakeSamplers


@pytest.fixture(autouse=True)
def use_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import dinkster_nodes_comfy.attention_guidance as guidance

    FakeModelPatcher.calls.clear()
    FakeSamplers.calls.clear()
    monkeypatch.setattr(guidance, "_inference_modules", fake_modules)


def _post_args(result: torch.Tensor, sigma: float = 5.0) -> dict[str, Any]:
    return {
        "model": "runtime-model",
        "model_options": {"transformer_options": {}},
        "cond": "cond",
        "uncond": "uncond",
        "input": torch.full_like(result, 8.0),
        "sigma": torch.tensor([sigma]),
        "cond_denoised": torch.full_like(result, 9.0),
        "uncond_denoised": torch.full_like(result, 2.0),
        "denoised": result,
    }


def test_attention_guidance_callbacks_preserve_source_and_cover_conditional_routes() -> None:
    source = Patcher()
    sag = SelfAttentionGuidance.execute(model=source, scale=0.5, blur_sigma=2.0)["MODEL"]
    assert sag is not source
    assert source.attention is None and source.post == []
    q = torch.tensor([[[1.0, -1.0]], [[2.0, 3.0]]])
    k = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    v = torch.tensor([[[3.0, 5.0]], [[7.0, 11.0]]])
    recorded = sag.attention(
        q, k, v, {"n_heads": 1, "cond_or_uncond": [0, 1], "attn_precision": None}
    )
    assert recorded.shape == v.shape
    uncond_only = sag.attention(
        q, k, v, {"n_heads": 1, "cond_or_uncond": [0], "attn_precision": None}
    )
    assert torch.equal(uncond_only, v + 2)
    result = torch.full((1, 1, 5, 5), 3.0)
    torch.testing.assert_close(sag.post[0](_post_args(result)), torch.full_like(result, 2.0))
    small = torch.ones((1, 1, 4, 4))
    assert sag.post[0](_post_args(small)) is small

    pag = PerturbedAttentionGuidance.execute(model=source, scale=2.0)["MODEL"]
    pag_result = pag.post[0](_post_args(torch.full((1, 1, 5, 5), 3.0)))
    torch.testing.assert_close(pag_result, torch.full_like(pag_result, 13.0))
    patch = FakeSamplers.calls[-1][2]["patches"][0]
    assert torch.equal(patch[0](q, k, v, {}), v)
    zero = PerturbedAttentionGuidance.execute(model=source, scale=0.0)["MODEL"]
    original = torch.full((1, 1, 5, 5), 3.0)
    assert zero.post[0](_post_args(original)) is original


def test_skip_layer_callbacks_parse_layers_include_sigma_boundaries_and_keep_empty_identity() -> (
    None
):
    source = Patcher()
    identity = SkipLayerGuidanceDiT.execute(
        model=source,
        double_layers="",
        single_layers="not-a-layer",
        scale=3.0,
        start_percent=0.1,
        end_percent=0.9,
        rescaling_scale=0.0,
    )["MODEL"]
    assert identity is source
    slg = SkipLayerGuidanceDiT.execute(
        model=source,
        double_layers="7, nine 9",
        single_layers="2",
        scale=2.0,
        start_percent=0.0,
        end_percent=1.0,
        rescaling_scale=0.0,
    )["MODEL"]
    result = torch.full((1, 1, 5, 5), 3.0)
    torch.testing.assert_close(
        slg.post[0](_post_args(result, sigma=10.0)), torch.full_like(result, 13.0)
    )
    assert FakeModelPatcher.calls == [
        ("dit", "double_block", 7),
        ("dit", "double_block", 9),
        ("dit", "single_block", 2),
    ]
    assert slg.post[0](_post_args(result, sigma=11.0)) is result

    simple = SkipLayerGuidanceDiTSimple.execute(
        model=source,
        double_layers="1",
        single_layers="3",
        start_percent=0.0,
        end_percent=1.0,
    )["MODEL"]
    routed = simple.calc(
        {
            "model": "runtime-model",
            "conds": ["cond", "uncond"],
            "input": result,
            "sigma": torch.tensor([10.0]),
            "model_options": {},
        }
    )
    torch.testing.assert_close(routed[0], torch.full_like(result, 7.0))
    torch.testing.assert_close(routed[1], torch.full_like(result, -3.0))
    fallback = simple.calc(
        {
            "model": "runtime-model",
            "conds": ["cond", "uncond"],
            "input": result,
            "sigma": torch.tensor([11.0]),
            "model_options": {},
        }
    )
    torch.testing.assert_close(fallback[0], torch.full_like(result, 2.0))
    torch.testing.assert_close(fallback[1], torch.full_like(result, -2.0))


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="attention-guidance-fixture.source",
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(MODEL=Patcher())


class ModelConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="attention-guidance-fixture.consumer",
            inputs=(InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, model: Patcher) -> Mapping[str, object]:
        assert len(model.post) == 3
        assert model.attention is not None and model.calc is not None
        return cls.outputs(value=float(len(model.post) + bool(model.attention) + bool(model.calc)))


FIXTURE_NODES = (ModelSource, ModelConsumer)


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_runs_all_attention_guidance_model_patches_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(Path(__file__).parent)})
        try:
            manifest = Path(__file__).parents[1] / "dinkster-pack.toml"
            module = f"attention_guidance_fixture_{in_process}"
            (tmp_path / f"{module}.py").write_text(
                "import dinkster_nodes_comfy.attention_guidance as guidance\n"
                "from test_attention_guidance_torch import FIXTURE_NODES, fake_modules\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "guidance._inference_modules = fake_modules\n"
                "NODES = (*COMFY_NODES, *FIXTURE_NODES)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["attention-guidance-fixture",')
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
                    "source": GraphNode("attention-guidance-fixture.source"),
                    "sag": GraphNode("SelfAttentionGuidance", {"model": Link("source", "MODEL")}),
                    "pag": GraphNode("PerturbedAttentionGuidance", {"model": Link("sag", "MODEL")}),
                    "slg": GraphNode("SkipLayerGuidanceDiT", {"model": Link("pag", "MODEL")}),
                    "simple": GraphNode(
                        "SkipLayerGuidanceDiTSimple", {"model": Link("slg", "MODEL")}
                    ),
                    "consumer": GraphNode(
                        "attention-guidance-fixture.consumer", {"model": Link("simple", "MODEL")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["value"].resolve() == 5.0
        finally:
            await composer.close()

    asyncio.run(scenario())
