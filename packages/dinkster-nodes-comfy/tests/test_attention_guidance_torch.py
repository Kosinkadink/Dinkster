"""CPU callback and ordinary-manifest coverage for attention guidance patches."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as functional
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


def _runtime() -> tuple[Any, Any, Any]:
    initialize_inference_process()
    from dinkster_inference import model_patcher, samplers
    from dinkster_inference.ldm.modules import attention

    return attention, model_patcher, samplers


def _asymmetric(shape: tuple[int, ...], start: float) -> torch.Tensor:
    return torch.arange(
        start, start + torch.tensor(shape).prod().item(), dtype=torch.float32
    ).reshape(shape)


def _blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    coordinate = torch.linspace(-4, 4, steps=9)
    density = torch.exp(-0.5 * (coordinate / sigma).pow(2))
    kernel = density / density.sum()
    kernel = torch.mm(kernel[:, None], kernel[None, :]).expand(image.shape[1], 1, 9, 9)
    return functional.conv2d(
        functional.pad(image, [4, 4, 4, 4], mode="reflect"), kernel, groups=image.shape[1]
    )


def _sag_blur(image: torch.Tensor, q: torch.Tensor, k: torch.Tensor, sigma: float) -> torch.Tensor:
    scores = (q[1] @ k[1].transpose(0, 1) / q.shape[-1] ** 0.5).softmax(dim=-1)
    mask = (scores.sum(dim=0) > 1.0).reshape(1, 1, 2, 2).to(image.dtype)
    mask = functional.interpolate(mask, image.shape[-2:])
    return _blur(image, sigma) * mask + image * (1 - mask)


def _post_args(denoised: torch.Tensor, sigma: float) -> dict[str, Any]:
    return {
        "model": "runtime-model",
        "model_options": {"transformer_options": {}},
        "cond": "cond",
        "uncond": "uncond",
        "input": _asymmetric(tuple(denoised.shape), 20.0),
        "sigma": torch.tensor([sigma]),
        "cond_denoised": _asymmetric(tuple(denoised.shape), -7.0),
        "uncond_denoised": _asymmetric(tuple(denoised.shape), 2.0),
        "denoised": denoised,
    }


def _controlled_calc(
    model: Any, conds: list[Any], x: torch.Tensor, sigma: torch.Tensor, options: dict[str, Any]
) -> list[torch.Tensor]:
    if conds == ["uncond"]:
        return [x * 0.25 - 3.0]
    if conds == ["cond"]:
        patches = options["transformer_options"]["patches_replace"]
        if "attn1" not in patches:
            return [x * 0.5 + 4.0]
        patch = patches["attn1"][("middle", 0)]
        value = patch(
            torch.ones((1, 4, 2)), torch.ones((1, 4, 2)), x.flatten(2).transpose(1, 2), {}
        )
        return [value.transpose(1, 2).reshape_as(x)]
    if conds == ["cond", None]:
        return [x * 0.5 + 4.0, torch.zeros_like(x)]
    if conds == [None, "uncond"]:
        return [torch.zeros_like(x), x * -0.25 - 2.0]
    return [x * 0.1 + 1.0, x * -0.1 - 1.0]


def test_active_callbacks_use_pinned_runtime_modules_and_preserve_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention, model_patcher, samplers = _runtime()
    original_calc = samplers.calc_cond_batch
    monkeypatch.setattr(samplers, "calc_cond_batch", _controlled_calc)
    source = Patcher()
    source_options = {"transformer_options": {"keep": object()}}
    x0 = _asymmetric((1, 1, 5, 8), 2.0)
    q = _asymmetric((2, 4, 2), -2.0)
    k = _asymmetric((2, 4, 2), 4.0)
    v = _asymmetric((2, 4, 2), 11.0)

    sag = SelfAttentionGuidance.execute(model=source, scale=0.3, blur_sigma=1.5)["MODEL"]
    assert sag is not source and source.attention is None and source.post == []
    captured = sag.attention(
        q, k, v, {"n_heads": 1, "cond_or_uncond": [0, 1], "attn_precision": None}
    )
    assert captured.shape == v.shape
    args = _post_args(x0, 5.0)
    args["model_options"] = source_options.copy()
    before_x0, before_options = x0.clone(), source_options.copy()
    actual = sag.post[0](args)
    degraded = _sag_blur(args["uncond_denoised"], q, k, 1.5)
    noised = degraded + args["input"] - args["uncond_denoised"]
    expected = args["denoised"] + (degraded - (noised * 0.25 - 3.0)) * 0.3
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(degraded, args["uncond_denoised"])
    assert torch.equal(x0, before_x0) and source_options == before_options
    optimized = sag.attention(
        q[:1], k[:1], v[:1], {"n_heads": 1, "cond_or_uncond": [0], "attn_precision": None}
    )
    expected_optimized = attention.optimized_attention_for_device(q.device)(q[:1], k[:1], v[:1], 1)
    torch.testing.assert_close(optimized, expected_optimized)
    small = _asymmetric((1, 1, 4, 8), 1.0)
    assert sag.post[0](_post_args(small, 5.0)) is small

    pag = PerturbedAttentionGuidance.execute(model=source, scale=1.7)["MODEL"]
    pag_args = _post_args(x0, 5.0)
    pag_args["model_options"] = {"transformer_options": {"existing": "preserved"}}
    pag_options = pag_args["model_options"].copy()
    pag_result = pag.post[0](pag_args)
    expected_pag = pag_args["denoised"] + (pag_args["cond_denoised"] - pag_args["input"]) * 1.7
    torch.testing.assert_close(pag_result, expected_pag)
    assert pag_args["model_options"] == pag_options
    zero = PerturbedAttentionGuidance.execute(model=source, scale=0.0)["MODEL"]
    assert zero.post[0](pag_args) is pag_args["denoised"]

    assert callable(model_patcher.set_model_options_patch_replace)
    assert original_calc is not _controlled_calc


def test_skip_layer_callbacks_cover_windows_rescaling_and_missing_uncond(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, samplers = _runtime()
    monkeypatch.setattr(samplers, "calc_cond_batch", _controlled_calc)
    source = Patcher()
    value = _asymmetric((1, 1, 5, 8), -4.0)
    slg = SkipLayerGuidanceDiT.execute(
        model=source,
        double_layers="7, 9",
        single_layers="2",
        scale=1.25,
        start_percent=0.0,
        end_percent=1.0,
        rescaling_scale=0.4,
    )["MODEL"]
    for sigma in (0.0, 10.0):
        args = _post_args(value, sigma)
        result = slg.post[0](args)
        skipped = args["input"] * 0.5 + 4.0
        guided = args["denoised"] + (args["cond_denoised"] - skipped) * 1.25
        expected = guided * (0.4 * args["cond_denoised"].std() / guided.std() + 0.6)
        torch.testing.assert_close(result, expected)
    out_of_window = _post_args(value, 11.0)
    assert slg.post[0](out_of_window) is value
    inactive = SkipLayerGuidanceDiT.execute(
        model=source,
        double_layers="1",
        single_layers="",
        scale=0.0,
        start_percent=0.0,
        end_percent=1.0,
        rescaling_scale=0.5,
    )["MODEL"]
    assert inactive.post[0](_post_args(value, 5.0)) is value

    simple = SkipLayerGuidanceDiTSimple.execute(
        model=source,
        double_layers="1",
        single_layers="3",
        start_percent=0.0,
        end_percent=1.0,
    )["MODEL"]
    active_args = {
        "model": "runtime-model",
        "conds": ["cond", "uncond"],
        "input": value,
        "sigma": torch.tensor([10.0]),
        "model_options": {"transformer_options": {}},
    }
    active = simple.calc(active_args)
    torch.testing.assert_close(active[0], value * 0.5 + 4.0)
    torch.testing.assert_close(active[1], value * -0.25 - 2.0)
    out = simple.calc({**active_args, "sigma": torch.tensor([11.0])})
    torch.testing.assert_close(out[0], value * 0.1 + 1.0)
    missing = simple.calc({**active_args, "conds": ["cond", None]})
    torch.testing.assert_close(missing[1], torch.zeros_like(value))


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "attention-guidance-fixture.source",
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(MODEL=Patcher())


class ModelConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "attention-guidance-fixture.consumer",
            inputs=(InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, model: Patcher) -> Mapping[str, object]:
        value = _asymmetric((1, 1, 5, 8), 3.0)
        q = _asymmetric((2, 4, 2), 1.0)
        k = _asymmetric((2, 4, 2), 20.0)
        v = _asymmetric((2, 4, 2), -3.0)
        model.attention(q, k, v, {"n_heads": 1, "cond_or_uncond": [0, 1], "attn_precision": None})
        sag_args = _post_args(value, 5.0)
        sag = model.post[0](sag_args)
        degraded = _sag_blur(sag_args["uncond_denoised"], q, k, 1.5)
        sag_expected = (
            value
            + (
                degraded
                - ((degraded + sag_args["input"] - sag_args["uncond_denoised"]) * 0.25 - 3.0)
            )
            * 0.3
        )
        torch.testing.assert_close(sag, sag_expected)
        pag_args = _post_args(value, 5.0)
        pag = model.post[1](pag_args)
        pag_expected = value + (pag_args["cond_denoised"] - pag_args["input"]) * 1.7
        torch.testing.assert_close(pag, pag_expected)
        slg_args = _post_args(value, 10.0)
        slg = model.post[2](slg_args)
        skipped = slg_args["input"] * 0.5 + 4.0
        slg_guided = value + (slg_args["cond_denoised"] - skipped) * 1.25
        slg_expected = slg_guided * (0.4 * slg_args["cond_denoised"].std() / slg_guided.std() + 0.6)
        torch.testing.assert_close(slg, slg_expected)
        simple_args = {
            "model": "runtime-model",
            "conds": ["cond", "uncond"],
            "input": value,
            "sigma": torch.tensor([10.0]),
            "model_options": {"transformer_options": {}},
        }
        simple = model.calc(simple_args)
        torch.testing.assert_close(simple[0], value * 0.5 + 4.0)
        torch.testing.assert_close(simple[1], value * -0.25 - 2.0)
        return cls.outputs(value=float(sag.sum() + pag.sum() + slg.sum() + simple[1].sum()))


FIXTURE_NODES = (ModelSource, ModelConsumer)


@pytest.mark.parametrize("in_process", (True, False))
def test_ordinary_manifest_executes_active_attention_guidance_callbacks_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={"PYTHONPATH": str(Path(__file__).parent), "DINKSTER_ACCELERATOR": "cpu"}
        )
        try:
            manifest = Path(__file__).parents[1] / "dinkster-pack.toml"
            module = f"attention_guidance_fixture_{in_process}"
            (tmp_path / f"{module}.py").write_text(
                "from dinkster_api.v1 import initialize_inference_process\n"
                "initialize_inference_process()\n"
                "from dinkster_inference import samplers\n"
                "from test_attention_guidance_torch import FIXTURE_NODES, _controlled_calc\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "samplers.calc_cond_batch = _controlled_calc\n"
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
                    "sag": GraphNode(
                        "SelfAttentionGuidance",
                        {"model": Link("source", "MODEL"), "scale": 0.3, "blur_sigma": 1.5},
                    ),
                    "pag": GraphNode(
                        "PerturbedAttentionGuidance",
                        {"model": Link("sag", "MODEL"), "scale": 1.7},
                    ),
                    "slg": GraphNode(
                        "SkipLayerGuidanceDiT",
                        {
                            "model": Link("pag", "MODEL"),
                            "double_layers": "7",
                            "single_layers": "2",
                            "scale": 1.25,
                            "start_percent": 0.0,
                            "end_percent": 1.0,
                            "rescaling_scale": 0.4,
                        },
                    ),
                    "simple": GraphNode(
                        "SkipLayerGuidanceDiTSimple",
                        {
                            "model": Link("slg", "MODEL"),
                            "double_layers": "1",
                            "single_layers": "3",
                            "start_percent": 0.0,
                            "end_percent": 1.0,
                        },
                    ),
                    "consumer": GraphNode(
                        "attention-guidance-fixture.consumer", {"model": Link("simple", "MODEL")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            results = []
            for _ in range(2):
                results.append(
                    (await engine.run(graph, ["consumer"])).outputs["consumer"]["value"].resolve()
                )
            assert results[0] == results[1] and results[0] != 0.0
        finally:
            await composer.close()

    asyncio.run(scenario())
