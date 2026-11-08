from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    initialize_inference_process,
)
from dinkster_sd.attention_guidance import SkipLayerGuidanceSD3


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


def test_sd3_skip_layer_guidance_preserves_window_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, samplers = _runtime()
    monkeypatch.setattr(samplers, "calc_cond_batch", _controlled_calc)
    source = Patcher()
    value = _asymmetric((1, 1, 5, 8), -4.0)
    patched = SkipLayerGuidanceSD3.execute(
        model=source, layers="2, 7", scale=1.75, start_percent=0.2, end_percent=0.8
    )["MODEL"]
    for sigma in (2.0, 5.0, 8.0):
        args = _post_args(value, sigma)
        expected = value + (args["cond_denoised"] - (args["input"] * 0.5 + 4.0)) * 1.75
        torch.testing.assert_close(patched.post[0](args), expected, rtol=0, atol=0)
    for sigma in (1.9, 8.1):
        assert patched.post[0](_post_args(value, sigma)) is value
    assert patched is not source
    assert source.post == []
