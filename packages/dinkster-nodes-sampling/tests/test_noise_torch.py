"""CPU behavior for native ComfyUI noise nodes."""

from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    LATENT_TYPE,
    MultiStreamLatent,
    materialize_latent,
)
from dinkster_sampling.noise import DisableNoise, RandomNoise


def test_sampling_manifest_registers_consumer_codecs_without_unrelated_types() -> None:
    from dinkster_workers import load_manifest
    from dinkster_workers.host import load_pack

    manifest = load_manifest(Path(__file__).resolve().parents[1] / "dinkster-pack.toml")
    _, registry, _, _ = load_pack(manifest)
    required = {"MODEL", "CONDITIONING", "LATENT", "SAMPLER", "SIGMAS", "NOISE", "GUIDER"}
    assert required <= set(registry.type_ids())
    assert all(registry.spec(type_id).declared_codec for type_id in required)
    assert not {"PHOTOMAKER", "INTERP_MODEL", "VAE", "CLIP", "IMAGE", "MASK", "SVG"} & set(
        registry.type_ids()
    )
    samples = torch.tensor([[[[1.25, -3.5, 8.0], [0.125, 7.0, -0.75]]]], dtype=torch.float64)
    codec = registry.spec(LATENT_TYPE)
    decoded = materialize_latent(
        codec.decode(codec.encode({"samples": samples, "batch_index": [7]}))
    )
    assert decoded["batch_index"] == [7]
    assert decoded["samples"].dtype == torch.float64
    torch.testing.assert_close(decoded["samples"], samples, rtol=0, atol=0)


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
