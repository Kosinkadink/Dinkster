"""Native AV guide consumer through nonempty pinned-engine Euler sampling.

Expected-value provenance: ComfyUI comfy_extras/nodes_lt.py,
LTXVDualCFGGuider at F bc5f880dbdd43e426077f42ccec53bb577ffee9f,
and dinkster-inference samplers.py CFGGuider/KSAMPLER/cfg_function plus
k_diffusion/sampling.py sample_euler at df0ec36bf1b3350004c7c785d9a9fc866ad9f630.
Exact inputs: two video batches of shape (1,1,2,3), two audio batches
of shape (2,1,2), sigmas [1,0], video conditional/unconditional [4,1],
audio conditional/unconditional [-2,2], scales below, video mask=1,
audio mask=1 except the final spatial column=0. Hand-derived: one Euler
step ends at the denoised prediction, video=1+3*video_cfg and
audio=2-4*audio_cfg where the mask is 1; masked audio retains its source.
Random-noise cases use seed 41 and batch indices [3,1]. Independent CPU
Generator draws four samples per stream and selects draws 3,1, as specified
by sample.py prepare_noise_inner at the same inference revision. Expectations
retain float32 operation order: x-(u+(c-u)*cfg) in noise space and
x+(x-denoised)*(-1) for Euler, rather than algebraically cancelling x.
Only model loading and conditioning preparation are controlled. The actual
public node boundary, packing, mask preparation, CFG, Euler, callbacks and
unpacking run. This is not real-model or GPU parity evidence.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch
from dinkster_api.v1 import MultiStreamLatent, initialize_inference_process
from dinkster_nodes_comfy.custom_sampling import SamplerCustomAdvanced
from dinkster_nodes_comfy.ltx_dual_cfg import LTXVDualCFGGuider
from dinkster_nodes_comfy.noise import DisableNoise, RandomNoise


@pytest.mark.parametrize("video_cfg,audio_cfg", [(2.0, 5.0), (1.0, 0.0), (0.0, 1.0)])
@pytest.mark.parametrize("random_noise", [False, True])
def test_dual_cfg_native_av_nonempty_sampling(
    monkeypatch: pytest.MonkeyPatch, video_cfg: float, audio_cfg: float, random_noise: bool
) -> None:
    initialize_inference_process()
    samplers = importlib.import_module("dinkster_inference.samplers")
    helpers = importlib.import_module("dinkster_inference.sampler_helpers")
    model_sampling = importlib.import_module("dinkster_inference.model_sampling")
    from dinkster_inference.model_patcher import ModelPatcher

    video = torch.arange(12.0).reshape(2, 1, 1, 2, 3) + 13
    audio = torch.arange(8.0).reshape(2, 2, 1, 2) + 21
    audio_mask = torch.ones(2, 1, 1, 2)
    audio_mask[..., 1] = 0
    original_mask = audio_mask.clone()
    source = {
        "samples": MultiStreamLatent.from_pairs((("video", video), ("audio", audio))),
        "noise_mask": MultiStreamLatent.from_pairs(
            (("video", torch.ones_like(video)), ("audio", audio_mask))
        ),
        "batch_index": [3, 1],
        "keep": "native-av-source",
    }
    generator = torch.Generator(device="cpu").manual_seed(41)
    initial = []
    denoised = []
    expected = []
    for samples, mask, conditional, unconditional, cfg in (
        (video, torch.ones_like(video), 4.0, 1.0, video_cfg),
        (audio, audio_mask, -2.0, 2.0, audio_cfg),
    ):
        draws = [torch.randn((1, *samples.shape[1:]), generator=generator) for _ in range(4)]
        noise_values = (
            torch.cat((draws[3], draws[1])) if random_noise else torch.zeros_like(samples)
        )
        x = noise_values + samples
        model_input = x * mask + samples * (1 - mask)
        initial.append(model_input)
        u, c = model_input - unconditional, model_input - conditional
        prediction = model_input - (u + (c - u) * cfg)
        prediction = prediction * mask + samples * (1 - mask)
        denoised.append(prediction)
        expected.append(x + (x - prediction) * -1)

    def predictions(args: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        x = args["input"]
        assert tuple(x.shape) == (2, 1, 10)
        expected_input = torch.cat([v.reshape(2, 1, -1) for v in initial], dim=-1)
        assert torch.equal(x, expected_input)
        assert args["conds"][0][0]["tag"] == "positive"
        assert args["conds"][1][0]["tag"] == "negative"
        conditional = torch.tensor([4.0] * 6 + [-2.0] * 4).expand_as(x)
        unconditional = torch.tensor([1.0] * 6 + [2.0] * 4).expand_as(x)
        return conditional, unconditional

    model = torch.nn.Linear(1, 1)
    model.model_sampling = model_sampling.EPS()
    model.model_sampling.sigma_data = 1.0
    model.model_sampling.sigma_max = 10.0
    model.process_latent_in = lambda value: value
    model.process_latent_out = lambda value: value
    model.scale_latent_inpaint = lambda **values: values["latent_image"]
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    patcher.model_options["sampler_calc_cond_batch_function"] = predictions
    monkeypatch.setattr(
        helpers,
        "prepare_sampling",
        lambda patcher, shape, conds, options: (patcher.model, conds, []),
    )
    monkeypatch.setattr(samplers, "process_conds", lambda model, noise, conds, *a, **k: conds)
    positive = [[torch.ones(1), {"tag": "positive"}]]
    negative = [[torch.zeros(1), {"tag": "negative"}]]
    guider = LTXVDualCFGGuider.execute(
        model=patcher,
        positive=positive,
        negative=negative,
        video_cfg=video_cfg,
        audio_cfg=audio_cfg,
    )["GUIDER"]
    noise = (
        RandomNoise.execute(noise_seed=41)["NOISE"]
        if random_noise
        else DisableNoise.execute()["NOISE"]
    )
    output = SamplerCustomAdvanced.execute(
        noise=noise,
        guider=guider,
        sampler=samplers.ksampler("euler"),
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image=source,
    )
    for name, wanted in (("output", expected), ("denoised_output", denoised)):
        result = output[name]
        assert result["keep"] == "native-av-source"
        assert result["batch_index"] == [3, 1]
        samples = result["samples"]
        assert type(samples) is MultiStreamLatent
        assert samples.roles == ("video", "audio")
        assert torch.equal(samples.by_role("video"), wanted[0])
        assert torch.equal(samples.by_role("audio"), wanted[1])
    assert torch.equal(video, torch.arange(12.0).reshape(2, 1, 1, 2, 3) + 13)
    assert torch.equal(audio, torch.arange(8.0).reshape(2, 2, 1, 2) + 21)
    assert torch.equal(source["noise_mask"].by_role("audio"), original_mask)
    assert torch.equal(source["noise_mask"].by_role("video"), torch.ones_like(video))
    assert positive[0][1] == {"tag": "positive"}
    assert negative[0][1] == {"tag": "negative"}
    assert patcher.model_options == {
        "transformer_options": {},
        "sampler_calc_cond_batch_function": predictions,
    }
