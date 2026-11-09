"""CPU behavior checks for native perpendicular negative guidance."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    GUIDER_TYPE,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.perpneg import PerpNeg, PerpNegGuider

from dinkster.compose import PackSpec, ServingComposer


def _model_patcher() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    return ModelPatcher(torch.nn.Linear(2, 2), torch.device("cpu"), torch.device("cpu"))


def test_perpneg_guider_uses_pinned_cfg_batch_callbacks_and_preserves_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_inference_process()
    from dinkster_inference import samplers

    source = _model_patcher()
    positive = torch.tensor([[[[4.0, -3.0]]]])
    negative = torch.tensor([[[[1.0, 5.0]]]])
    empty = torch.tensor([[[[-2.0, 1.0]]]])
    x = torch.tensor([[[[9.0, 7.0]]]])
    calls: list[list[Any]] = []

    def calc(model: Any, conds: list[Any], *_: Any) -> list[torch.Tensor]:
        calls.append(conds)
        return [positive, negative, empty]

    monkeypatch.setattr(samplers, "calc_cond_batch", calc)
    guider = PerpNegGuider.execute(
        model=source,
        positive=[["positive", {}]],
        negative=[["negative", {}]],
        empty_conditioning=[["empty", {}]],
        cfg=2.5,
        neg_scale=0.75,
    )["GUIDER"]
    guider.inner_model = source
    guider.conds = guider.original_conds
    trace: list[str] = []

    def pre(args: dict[str, Any]) -> list[torch.Tensor]:
        trace.append("pre")
        assert args["conds_out"] == [positive, negative, empty]
        return [prediction * 2.0 for prediction in args["conds_out"]]

    def post(args: dict[str, Any]) -> torch.Tensor:
        trace.append("post")
        assert torch.equal(args["empty_cond_denoised"], empty * 2.0)
        assert torch.equal(args["cond_denoised"], positive * 2.0)
        assert torch.equal(args["uncond_denoised"], negative * 2.0)
        return args["denoised"] + torch.tensor([[[[1.0, -2.0]]]])

    options = {
        "marker": object(),
        "sampler_pre_cfg_function": [pre],
        "sampler_post_cfg_function": [post],
    }
    before = x.clone()
    actual = guider.predict_noise(x, torch.tensor([3.0]), options)
    torch.testing.assert_close(actual, torch.tensor([[[[216.0 / 13.0, -925.0 / 26.0]]]]))
    assert trace == ["pre", "post"]
    assert len(calls) == 1 and all(condition is not None for condition in calls[0])
    assert torch.equal(x, before)
    assert source.model_options.get("sampler_cfg_function") is None


@pytest.mark.parametrize(
    ("disable", "expected_none"), ((False, (False, True, True)), (True, (False, False, False)))
)
def test_perpneg_guider_keeps_cfg_one_optimization_boundary(
    monkeypatch: pytest.MonkeyPatch, disable: bool, expected_none: tuple[bool, bool, bool]
) -> None:
    initialize_inference_process()
    from dinkster_inference import samplers

    seen: list[list[Any]] = []
    monkeypatch.setattr(
        samplers,
        "calc_cond_batch",
        lambda _model, conds, x, *_: (
            seen.append(conds) or [torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x)]
        ),
    )
    guider = PerpNegGuider.execute(
        model=_model_patcher(),
        positive=[["positive", {}]],
        negative=[["negative", {}]],
        empty_conditioning=[["empty", {}]],
        cfg=1.0,
        neg_scale=0.0,
    )["GUIDER"]
    guider.inner_model = guider.model_patcher
    guider.conds = guider.original_conds
    guider.predict_noise(
        torch.ones((1, 1, 1, 2)), torch.tensor([1.0]), {"disable_cfg1_optimization": disable}
    )
    assert tuple(value is None for value in seen[0]) == expected_none


def test_legacy_perpneg_clones_source_and_uses_empty_conditioning_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_inference_process()
    from dinkster_inference import samplers

    source = _model_patcher()
    source.extra_conds = {}
    empty = torch.tensor([[[[2.0, -1.0]]]])
    monkeypatch.setattr(samplers, "encode_model_conds", lambda *_: "encoded-empty")
    monkeypatch.setattr(samplers, "calc_cond_batch", lambda *_: [empty])
    patched = PerpNeg.execute(model=source, empty_conditioning=[[None, {}]], neg_scale=0.5)["MODEL"]
    x = torch.tensor([[[[7.0, 3.0]]]])
    pos = torch.tensor([[[[6.0, -2.0]]]])
    neg = torch.tensor([[[[1.0, 4.0]]]])
    result = patched.model_options["sampler_cfg_function"](
        {
            "model": SimpleNamespace(extra_conds={}),
            "input": x,
            "sigma": torch.tensor([1.0]),
            "model_options": {"preserved": True},
            "cond_denoised": pos,
            "uncond_denoised": neg,
            "cond_scale": 2.0,
        }
    )
    torch.testing.assert_close(result, torch.tensor([[[[-32.0 / 17.0, 178.0 / 17.0]]]]))
    assert patched is not source
    assert "sampler_cfg_function" not in source.model_options


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="perpneg-fixture.source",
            outputs=(
                OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                OutputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(model=_model_patcher(), conditioning=[[None, {}]])


class Consumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="perpneg-fixture.consumer",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("guider", TypeExpr.concrete(GUIDER_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model: Any, guider: Any) -> Mapping[str, object]:
        from dinkster_inference import samplers

        assert model is not guider.model_patcher
        assert "sampler_cfg_function" in model.model_options
        assert "sampler_cfg_function" not in guider.model_patcher.model_options
        assert guider.cfg == 2.0 and guider.neg_scale == 0.5
        positive = torch.tensor([[[[6.0, -2.0]]]])
        negative = torch.tensor([[[[1.0, 4.0]]]])
        empty = torch.tensor([[[[2.0, -1.0]]]])
        x = torch.tensor([[[[7.0, 3.0]]]])
        before = x.clone()
        sigma = torch.tensor([1.0])
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(samplers, "encode_model_conds", lambda *_: "encoded-empty")
            patch.setattr(
                samplers,
                "calc_cond_batch",
                lambda _model, conds, *_: (
                    [empty] if len(conds) == 1 else [positive, negative, empty]
                ),
            )
            guider.inner_model = guider.model_patcher.model
            guider.inner_model.extra_conds = {}
            guider.conds = guider.original_conds
            result = guider.predict_noise(x, sigma)
            torch.testing.assert_close(result, torch.tensor([[[[151.0 / 17.0, -127.0 / 17.0]]]]))
            residual = model.model_options["sampler_cfg_function"](
                {
                    "model": guider.inner_model,
                    "input": x,
                    "sigma": sigma,
                    "model_options": {"preserved": True},
                    "cond_denoised": positive,
                    "uncond_denoised": negative,
                    "cond_scale": 2.0,
                }
            )
            torch.testing.assert_close(residual, torch.tensor([[[[-32.0 / 17.0, 178.0 / 17.0]]]]))
        assert torch.equal(x, before)
        return cls.outputs(summary="perpneg")


@pytest.mark.parametrize("in_process", (True, False))
def test_perpneg_nodes_run_twice_in_both_ordinary_hosts_with_cache(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        module = f"perpneg_fixture_{int(in_process)}"
        (tmp_path / f"{module}.py").write_text(
            "from test_perpneg_torch import Consumer, Source\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES = (Source, Consumer)\n",
            encoding="ascii",
        )
        fixture_manifest = tmp_path / "fixture-dinkster-pack.toml"
        fixture_manifest.write_text(
            "[pack]\n"
            'name = "perpneg-fixture"\n'
            'publisher = "dinkster"\n'
            'namespaces = ["perpneg-fixture"]\n\n'
            "[pack.entry]\n"
            f'nodes = "{module}:NODES"\n'
            f'types = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent),)),
            }
        )
        try:
            manifests = (native_manifest, fixture_manifest)
            group = "perpneg" if not in_process else None
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=group,
                        group_manifests=manifests if group else (),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("perpneg-fixture.source"),
                    "legacy": GraphNode(
                        "PerpNeg",
                        {
                            "model": Link("source", "model"),
                            "empty_conditioning": Link("source", "conditioning"),
                            "neg_scale": 0.5,
                        },
                    ),
                    "guider": GraphNode(
                        "PerpNegGuider",
                        {
                            "model": Link("source", "model"),
                            "positive": Link("source", "conditioning"),
                            "negative": Link("source", "conditioning"),
                            "empty_conditioning": Link("source", "conditioning"),
                            "cfg": 2.0,
                            "neg_scale": 0.5,
                        },
                    ),
                    "consumer": GraphNode(
                        "perpneg-fixture.consumer",
                        {"model": Link("legacy", "MODEL"), "guider": Link("guider", "GUIDER")},
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "perpneg"
                if attempt:
                    assert set(result.cached) == set(graph.nodes)
        finally:
            await composer.close()

    asyncio.run(scenario())
