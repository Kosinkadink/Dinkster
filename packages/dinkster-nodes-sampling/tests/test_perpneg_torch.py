"""CPU behavior checks for native perpendicular negative guidance."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    initialize_inference_process,
)
from dinkster_nodes_sampling.perpneg import PerpNeg, PerpNegGuider


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
