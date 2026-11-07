"""CPU behavior for native ComfyUI noise nodes."""

from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    MultiStreamLatent,
)
from dinkster_nodes_sampling.noise import AddNoise, DisableNoise, RandomNoise


def test_random_noise_uses_seeded_batch_indices_and_disable_noise_preserves_layout() -> None:
    latent = {
        "samples": torch.zeros((2, 1, 2, 2), dtype=torch.float32),
        "batch_index": [7, 3],
    }
    noise = RandomNoise.execute(noise_seed=42)["NOISE"].generate_noise(latent)
    # ComfyUI Noise_Random, F bc5f880d: seed=42, float32 zeros (2,1,2,2),
    # batch_index=[7,3]; generate_noise delegates to comfy.sample.prepare_noise.
    assert noise.flatten().tolist() == [
        -0.049903348088264465,
        0.5263369679450989,
        -0.008498823270201683,
        0.7290605902671814,
        1.110290288925171,
        -1.6897989511489868,
        -0.9889599084854126,
        0.9579718112945557,
    ]
    disabled = DisableNoise.execute()["NOISE"].generate_noise(latent)
    assert disabled.dtype == latent["samples"].dtype
    assert disabled.device == latent["samples"].device
    assert torch.equal(disabled, torch.zeros_like(latent["samples"]))


def test_disable_noise_uses_pinned_cpu_nested_noise_preparation() -> None:
    from dinkster_api.v1 import initialize_inference_process

    initialize_inference_process()
    from dinkster_inference.nested_tensor import NestedTensor

    samples = NestedTensor([torch.ones((1, 1, 2, 2), dtype=torch.float64)])
    disabled = DisableNoise.execute()["NOISE"].generate_noise({"samples": samples})
    assert disabled.is_nested
    tensor = disabled.unbind()[0]
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float64
    assert torch.equal(tensor, torch.zeros_like(tensor))


@pytest.mark.parametrize("noise_node", (RandomNoise, DisableNoise))
def test_public_noise_preserves_asymmetric_av_roles_shapes_and_dtypes(
    noise_node: type[Any],
) -> None:
    video = torch.zeros((1, 3, 2, 2, 3), dtype=torch.float64)
    audio = torch.zeros((1, 2, 4), dtype=torch.float32)
    samples = MultiStreamLatent.from_pairs((("video", video), ("audio", audio)))
    noise = (
        noise_node.execute(noise_seed=42)["NOISE"]
        if noise_node is RandomNoise
        else noise_node.execute()["NOISE"]
    ).generate_noise({"samples": samples, "batch_index": [3]})
    assert type(noise) is MultiStreamLatent
    assert noise.roles == ("video", "audio")
    assert noise.by_role("video").shape == video.shape
    assert noise.by_role("video").dtype == video.dtype
    assert noise.by_role("audio").shape == audio.shape
    assert noise.by_role("audio").dtype == audio.dtype


class _Sampling:
    def noise_scaling(
        self, scale: torch.Tensor, noise: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        return latent + noise * scale


class _Model:
    def get_model_object(self, name: str) -> object:
        return {
            "model_sampling": _Sampling(),
            "process_latent_in": lambda value: value + 10,
            "process_latent_out": lambda value: torch.nan_to_num(value * 2),
        }[name]


def test_add_noise_preserves_metadata_skips_empty_input_and_uses_sigma_endpoints() -> None:
    latent = {"samples": torch.zeros((1, 1, 1, 2)), "note": "preserved"}
    value = AddNoise.execute(
        model=_Model(),
        noise=DisableNoise.execute()["NOISE"],
        sigmas=torch.tensor([4.0, 1.5]),
        latent_image=latent,
    )["LATENT"]
    assert value["note"] == "preserved"
    assert torch.equal(value["samples"], torch.zeros((1, 1, 1, 2)))
    assert (
        AddNoise.execute(
            model=_Model(),
            noise=DisableNoise.execute()["NOISE"],
            sigmas=torch.tensor([]),
            latent_image=latent,
        )["LATENT"]
        is latent
    )


def test_add_noise_uses_absolute_endpoint_delta_and_transforms_nonzero_latents() -> None:
    latent = {"samples": torch.tensor([[[[1.0, -2.0]]]]), "note": "kept"}

    class Noise:
        def generate_noise(self, _: object) -> torch.Tensor:
            return torch.tensor([[[[2.0, -3.0]]]])

    value = AddNoise.execute(
        model=_Model(), noise=Noise(), sigmas=torch.tensor([1.0, 4.0]), latent_image=latent
    )["LATENT"]
    assert value["note"] == "kept"
    assert torch.equal(value["samples"], torch.tensor([[[[34.0, -2.0]]]]))
    one_sigma = AddNoise.execute(
        model=_Model(), noise=Noise(), sigmas=torch.tensor([3.0]), latent_image=latent
    )["LATENT"]
    assert torch.equal(one_sigma["samples"], torch.tensor([[[[34.0, -2.0]]]]))


@pytest.mark.parametrize("empty", (False, True))
def test_add_noise_handles_native_av_arithmetic_and_preserves_source(empty: bool) -> None:
    class Model:
        def get_model_object(self, name: str) -> object:
            return {
                "model_sampling": _Sampling(),
                "process_latent_in": lambda value: value + 10,
                "process_latent_out": lambda value: value * 2,
            }[name]

    video = torch.zeros((1, 1, 2, 2, 3), dtype=torch.float64)
    audio = torch.zeros((1, 1, 4), dtype=torch.float32)
    if not empty:
        audio[0, 0] = torch.tensor([3.0, float("nan"), float("inf"), -float("inf")])
    original_audio = audio.clone()
    samples = MultiStreamLatent.from_pairs((("video", video), ("audio", audio)))
    source = {"samples": samples, "note": "kept"}
    result = AddNoise.execute(
        model=Model(),
        noise=DisableNoise.execute()["NOISE"],
        sigmas=torch.tensor([4.0, 1.0]),
        latent_image=source,
    )["LATENT"]
    assert result["note"] == "kept"
    assert result["samples"].roles == ("video", "audio")
    # Hand-derived: zero noise, shift both streams iff any stream is nonzero,
    # then multiply by two. Mixed empty/nonempty streams detect per-stream shifts.
    shift = 0 if empty else 10
    torch.testing.assert_close(result["samples"].by_role("video"), (video + shift) * 2)
    expected_audio = torch.zeros_like(audio) if empty else torch.tensor([[[26.0, 0.0, 0.0, 0.0]]])
    torch.testing.assert_close(result["samples"].by_role("audio"), expected_audio)
    torch.testing.assert_close(audio, original_audio, equal_nan=True)
    assert source["samples"] is samples
