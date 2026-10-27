"""CPU contracts for InfiniteTalk conditioning and existing model patch wrappers."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import AUDIO_ENCODER_OUTPUT_TYPE, TypeRegistry, register_inference_types
from dinkster_nodes_comfy.wan_infinite_talk import WanInfiniteTalkToVideo


class Model:
    def __init__(self) -> None:
        self.model_options: dict[str, Any] = {"transformer_options": {"preserved": 17}}
        self.patches: dict[str, Any] = {}
        self.wrappers: dict[str, Any] = {}

    def clone(self) -> Model:
        return copy.deepcopy(self)

    def model_dtype(self) -> torch.dtype:
        return torch.float64

    def add_wrapper_with_key(self, kind: str, key: str, wrapper: Any) -> None:
        self.wrappers[key] = wrapper

    def set_model_patch(self, patch: Any, name: str) -> None:
        self.patches[name] = patch


class AudioProjection:
    seq_len = 5

    def __call__(self, first: torch.Tensor, latter: torch.Tensor) -> torch.Tensor:
        return torch.cat((first.mean((2, 3)), latter.mean((2, 3))), dim=1).unsqueeze(2)


class VAE:
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image[::4, ::8, ::8, 0].unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1, 1)


def _audio(frames: int = 10, offset: float = 0.0) -> dict[str, Any]:
    ramp = torch.arange(frames, dtype=torch.float32).reshape(1, frames, 1)
    first = ramp + torch.tensor([offset, offset + 10])
    return {
        "encoded_audio": first,
        "encoded_audio_all_layers": (torch.full_like(first, -777), first, first + 100),
        "audio_samples": frames * 320,
    }


def _inputs(**values: Any) -> dict[str, Any]:
    return {
        "mode": "single_speaker",
        "model": Model(),
        "model_patch": SimpleNamespace(model=SimpleNamespace(audio_proj=AudioProjection())),
        "positive": [[torch.tensor([2.0]), {"preserved": "positive"}]],
        "negative": [[torch.tensor([7.0]), {"preserved": "negative"}]],
        "vae": VAE(),
        "width": 48,
        "height": 32,
        "length": 5,
        "audio_encoder_output_1": _audio(),
        "motion_frame_count": 3,
        "audio_scale": 0.0,
        "start_image": torch.full((1, 32, 48, 4), 0.75),
        **values,
    }


def test_single_speaker_public_audio_roundtrip_and_motion_conditioning() -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    audio = _audio()
    spec = registry.spec(AUDIO_ENCODER_OUTPUT_TYPE)
    inputs = _inputs(audio_encoder_output_1=spec.decode(spec.encode(audio)))
    result = WanInfiniteTalkToVideo.execute(**inputs)
    model = result["model"]
    embedding = model.model_options["transformer_options"]["audio_embeds"]
    assert embedding.dtype == torch.float64
    assert embedding.shape == (1, 2, 1, 2)
    torch.testing.assert_close(
        embedding[0, :, 0],
        torch.tensor([[51.35, 61.35], [55.0625, 65.0625]], dtype=torch.float64),
    )
    assert model is not inputs["model"]
    assert inputs["model"].model_options == {"transformer_options": {"preserved": 17}}
    assert inputs["model"].patches == inputs["model"].wrappers == {}
    assert model.patches["attn2_patch"].audio_scale == 0.0
    assert "attn1_patch" not in model.patches
    wrapper = model.wrappers["infinite_talk_outer_sample"]
    assert wrapper.is_extend is False
    assert torch.all(wrapper.motion_frames_latent == 0.75)
    assert result["trim_image"] == 0
    assert result["latent"]["samples"].shape == (1, 16, 2, 4, 6)
    assert result["latent"]["samples"].count_nonzero() == 0
    for key in ("positive", "negative"):
        metadata = result[key][0][1]
        assert torch.all(metadata["concat_latent_image"][:, :, 0] == 0.75)
        assert torch.all(metadata["concat_latent_image"][:, :, 1] == 0.5)
        assert torch.equal(metadata["concat_mask"].flatten(), torch.tensor([0.0] * 24 + [1.0] * 24))
        assert inputs[key][0][1] == {"preserved": key}
    assert torch.all(inputs["start_image"] == 0.75)
    assert torch.all(audio["encoded_audio_all_layers"][0] == -777)


def test_two_speakers_use_sequential_audio_regions_masks_and_previous_tail() -> None:
    first = {"encoded_audio_all_layers": [torch.full((1, 8, 2), value) for value in (-99, 2, 8)]}
    second = {
        "encoded_audio_all_layers": [torch.full((1, 12, 2), value) for value in (-99, 20, 40)]
    }
    # Floating point features match the admitted audio encoder output contract.
    for audio in (first, second):
        audio["encoded_audio_all_layers"] = [
            value.float() for value in audio["encoded_audio_all_layers"]
        ]
    mask_1 = torch.zeros((1, 32, 48))
    mask_1[:, :16, :16] = 1
    mask_2 = torch.ones((1, 32, 48))
    previous = torch.arange(7, dtype=torch.float32).reshape(7, 1, 1, 1).expand(7, 32, 48, 3)
    inputs = _inputs(
        mode="two_speakers",
        start_image=None,
        previous_frames=previous,
        audio_encoder_output_1=first,
        **{"mode.audio_encoder_output_2": second, "mode.mask_1": mask_1, "mode.mask_2": mask_2},
    )
    result = WanInfiniteTalkToVideo.execute(**inputs)
    model = result["model"]
    embedding = model.model_options["transformer_options"]["audio_embeds"]
    # Window 4..8 lies at the sequential speaker boundary, not mixed at time zero.
    torch.testing.assert_close(
        embedding[0, :, :, 0], torch.tensor([[2.0, 18.0], [0.625, 26.25]], dtype=torch.float64)
    )
    assert result["trim_image"] == 3
    wrapper = model.wrappers["infinite_talk_outer_sample"]
    assert wrapper.is_extend is True
    assert torch.all(wrapper.motion_frames_latent == 4)
    targets = model.patches["attn1_patch"].ref_target_masks
    assert targets.dtype == torch.bool
    assert torch.equal(
        targets, torch.tensor([[True, False, False, False, False, False], [True] * 6])
    )
    assert result["positive"] is inputs["positive"]
    assert result["negative"] is inputs["negative"]
    assert torch.equal(previous[:, 0, 0, 0], torch.arange(7, dtype=torch.float32))
    assert torch.all(second["encoded_audio_all_layers"][2] == 40)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"previous_frames": torch.zeros(2, 32, 48, 3)}, "Not enough previous frames"),
        ({"audio_encoder_output_2": _audio()}, "Masks must be provided"),
        (
            {"mask_1": torch.ones(1, 32, 48), "mask_2": torch.ones(1, 32, 48)},
            "Second audio encoder",
        ),
    ],
)
def test_invalid_motion_and_speaker_inputs(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WanInfiniteTalkToVideo.execute(**_inputs(**changes))
