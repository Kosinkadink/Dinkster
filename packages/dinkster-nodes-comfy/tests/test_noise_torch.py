"""CPU behavior for native ComfyUI noise nodes."""

import torch
from dinkster_nodes_comfy.noise import AddNoise, DisableNoise, RandomNoise


def test_random_noise_uses_seeded_batch_indices_and_disable_noise_preserves_layout() -> None:
    latent = {
        "samples": torch.zeros((2, 1, 2, 2), dtype=torch.float32),
        "batch_index": [7, 3],
    }
    noise = RandomNoise.execute(noise_seed=42)["NOISE"].generate_noise(latent)
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
