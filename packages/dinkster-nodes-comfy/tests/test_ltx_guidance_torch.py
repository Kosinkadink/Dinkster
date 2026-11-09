"""CPU contracts for LTXV model-guidance callbacks and audio tokens."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_nodes_comfy.ltx_guidance import (
    LTXVModalityGuidance,
    LTXVReferenceAudio,
    LTXVSpatioTemporalGuidance,
)


class Sampling:
    def percent_to_sigma(self, percent: float) -> float:
        return 12.0 * (1.0 - percent)


class Model:
    def __init__(self) -> None:
        self.callbacks: list[Any] = []

    def clone(self) -> Model:
        result = Model()
        result.callbacks = self.callbacks.copy()
        return result

    def get_model_object(self, name: str) -> Sampling:
        assert name == "model_sampling"
        return Sampling()

    def set_model_sampler_post_cfg_function(self, callback: Any) -> None:
        self.callbacks.append(callback)


class AudioVae:
    audio_sample_rate = 16000

    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        self.inputs.append(waveform.clone())
        return torch.arange(waveform.shape[0] * 3 * 5 * 2.0).reshape(waveform.shape[0], 3, 5, 2)


def guidance_args(sigma: float) -> dict[str, Any]:
    return {
        "model": object(),
        "sigma": torch.tensor([sigma], dtype=torch.float64),
        "input": torch.tensor([1.0, 3.0, 9.0]),
        "denoised": torch.tensor([2.0, -3.0, 7.0]),
        "cond_denoised": torch.tensor([5.0, 2.0, -1.0]),
        "cond": [{"model_conds": {"ref_audio": "identity", "text": "keep"}, "other": 4}],
        "model_options": {"transformer_options": {"keep": 7}, "other": "preserved"},
    }


@pytest.mark.parametrize("source_rate", [16000, 8000])
def test_reference_audio_token_axes_and_source_preservation(source_rate: int) -> None:
    initialize_inference_process()
    from dinkster_inference.audio import resample

    model, vae = Model(), AudioVae()
    waveform = torch.arange(2 * 2 * 17.0).reshape(2, 2, 17) / 100
    before = waveform.clone()
    positive, negative = [[torch.ones(1), {"keep": "p"}]], [[torch.zeros(1), {"keep": "n"}]]
    output = LTXVReferenceAudio.execute(
        model=model,
        positive=positive,
        negative=negative,
        reference_audio={"waveform": waveform, "sample_rate": source_rate},
        audio_vae=vae,
        identity_guidance_scale=2.5,
        start_percent=0.25,
        end_percent=0.75,
    )
    expected_waveform = waveform if source_rate == 16000 else resample(waveform, 8000, 16000)
    assert torch.equal(vae.inputs[0], expected_waveform.movedim(1, -1))
    expected_tokens = torch.stack(
        [
            torch.tensor(
                [
                    [base + frame * 2 + freq for base in (0, 10, 20) for freq in (0, 1)]
                    for frame in range(5)
                ]
            )
            + batch * 30
            for batch in range(2)
        ]
    ).float()
    for name, label in (("positive", "p"), ("negative", "n")):
        assert output[name][0][1]["keep"] == label
        assert torch.equal(output[name][0][1]["ref_audio"]["tokens"], expected_tokens)
    assert torch.equal(waveform, before)
    assert positive[0][1] == {"keep": "p"} and negative[0][1] == {"keep": "n"}
    assert not model.callbacks and len(output["MODEL"].callbacks) == 1


def test_reference_audio_vae_without_rate_uses_44100() -> None:
    vae = AudioVae()
    waveform = torch.arange(22.0).reshape(1, 2, 11)
    LTXVReferenceAudio.execute(
        model=Model(),
        positive=[],
        negative=[],
        reference_audio={"waveform": waveform, "sample_rate": 44100},
        audio_vae=SimpleNamespace(encode=vae.encode),
        identity_guidance_scale=0.0,
        start_percent=0.0,
        end_percent=1.0,
    )
    assert torch.equal(vae.inputs[0], waveform.movedim(1, -1))


@pytest.mark.parametrize(
    "sigma,active", [(9.001, False), (9.0, True), (5.0, True), (3.0, True), (2.999, False)]
)
@pytest.mark.parametrize("kind", ["reference", "temporal", "modality"])
def test_guidance_closed_windows_and_unmodified_inputs(
    monkeypatch, sigma: float, active: bool, kind: str
) -> None:
    initialize_inference_process()
    from dinkster_inference import samplers

    def prediction(model, conditions, samples, timestep, options):
        assert options["other"] == "preserved"
        transformer = options["transformer_options"]
        assert transformer["keep"] == 7
        if kind == "reference":
            assert conditions == [[{"model_conds": {"text": "keep"}, "other": 4}]]
        elif kind == "temporal":
            assert transformer["stg_self_attn_blocks"] == frozenset({2, 5, 29})
        else:
            assert transformer["a2v_cross_attn"] is False and transformer["v2a_cross_attn"] is False
        return (torch.tensor([-1.0, 4.0, 2.0]),)

    monkeypatch.setattr(samplers, "calc_cond_batch", prediction)
    model = Model()
    common = {"model": model, "start_percent": 0.25, "end_percent": 0.75}
    if kind == "reference":
        output = LTXVReferenceAudio.execute(
            **common,
            positive=[],
            negative=[],
            reference_audio={"waveform": torch.ones(1, 1, 7), "sample_rate": 16000},
            audio_vae=AudioVae(),
            identity_guidance_scale=2.5,
        )
    elif kind == "temporal":
        output = LTXVSpatioTemporalGuidance.execute(**common, scale=2.5, blocks="2, 29, -5,2")
    else:
        output = LTXVModalityGuidance.execute(**common, modality_scale=3.5)
    args = guidance_args(sigma)
    old_conditioning, old_options = (
        copy.deepcopy(args["cond"]),
        copy.deepcopy(args["model_options"]),
    )
    result = output["MODEL"].callbacks[-1](args)
    expected = torch.tensor([17.0, -8.0, -0.5]) if active else args["denoised"]
    assert torch.equal(result, expected)
    assert args["cond"] == old_conditioning and args["model_options"] == old_options
    assert not model.callbacks


@pytest.mark.parametrize(
    "kind", ["reference", "zero-temporal", "empty-temporal", "modality", "near-one-modality"]
)
def test_disabled_guidance_avoids_extra_sampler_pass(monkeypatch, kind: str) -> None:
    initialize_inference_process()
    from dinkster_inference import samplers

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled guidance must not evaluate another prediction")

    monkeypatch.setattr(samplers, "calc_cond_batch", forbidden)
    common = {"model": Model(), "start_percent": 0.25, "end_percent": 0.75}
    if kind == "reference":
        output = LTXVReferenceAudio.execute(
            **common,
            positive=[],
            negative=[],
            reference_audio={"waveform": torch.ones(1, 1, 7), "sample_rate": 16000},
            audio_vae=AudioVae(),
            identity_guidance_scale=0.0,
        )
    elif kind.endswith("temporal"):
        output = LTXVSpatioTemporalGuidance.execute(
            **common,
            scale=0.0 if kind == "zero-temporal" else 2.5,
            blocks="29" if kind == "zero-temporal" else "none",
        )
    else:
        output = LTXVModalityGuidance.execute(
            **common,
            modality_scale=1.0 if kind == "modality" else 1.0 + 1e-10,
        )
    args = guidance_args(5.0)
    assert output["MODEL"].callbacks[-1](args) is args["denoised"]


def test_guidance_nodes_stack_without_replacing_earlier_patches(monkeypatch) -> None:
    initialize_inference_process()
    from dinkster_inference import samplers

    monkeypatch.setattr(
        samplers, "calc_cond_batch", lambda *args: (torch.tensor([-1.0, 4.0, 2.0]),)
    )
    source = Model()
    window = {"start_percent": 0.25, "end_percent": 0.75}
    temporal = LTXVSpatioTemporalGuidance.execute(model=source, scale=1.5, blocks="2,29", **window)[
        "MODEL"
    ]
    coupled = LTXVModalityGuidance.execute(model=temporal, modality_scale=2.0, **window)["MODEL"]
    output = LTXVReferenceAudio.execute(
        model=coupled,
        positive=[],
        negative=[],
        reference_audio={"waveform": torch.ones(1, 1, 7), "sample_rate": 16000},
        audio_vae=AudioVae(),
        identity_guidance_scale=0.5,
        **window,
    )["MODEL"]
    args = guidance_args(5.0)
    for callback in output.callbacks:
        args["denoised"] = callback(args)
    assert torch.equal(args["denoised"], torch.tensor([20.0, -9.0, -2.0]))
    assert not source.callbacks
    assert len(temporal.callbacks) == 1 and len(coupled.callbacks) == 2
    assert args["model_options"] == {"transformer_options": {"keep": 7}, "other": "preserved"}
