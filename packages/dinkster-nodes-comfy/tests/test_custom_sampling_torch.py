"""CPU behavior for native custom sampler nodes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from dinkster_nodes_comfy import custom_sampling


class _Sample:
    def __init__(self) -> None:
        self.noise_args: tuple[object, ...] | None = None

    def fix_empty_latent_channels(
        self, _model: object, samples: torch.Tensor, *_ratios: object
    ) -> torch.Tensor:
        return samples + 1

    def prepare_noise(self, samples: torch.Tensor, seed: int, batch_index: object) -> torch.Tensor:
        self.noise_args = (seed, batch_index)
        return samples * 0 + torch.tensor([[[[2.0, -3.0]]], [[[5.0, 7.0]]]])

    def prepare_empty_noise(self, samples: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(samples)

    def sample_custom(
        self,
        model: object,
        noise: torch.Tensor,
        cfg: float,
        sampler: object,
        sigmas: torch.Tensor,
        positive: object,
        negative: object,
        samples: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        del model, sampler, sigmas, positive, negative
        callback = kwargs["callback"]
        assert callable(callback)
        callback(0, samples - noise, samples, 2)
        assert cfg == 2.5 and kwargs["noise_mask"] is not None and kwargs["seed"] == 19
        return samples + noise


class _Model:
    def __init__(self) -> None:
        self.model = SimpleNamespace(process_latent_out=lambda tensor: tensor * 10)


class _LatentFormat:
    latent_channels = 1
    latent_dimensions = 2
    spacial_downscale_ratio = 8
    temporal_downscale_ratio = 1

    def fix_empty_latent(self, samples: torch.Tensor) -> torch.Tensor:
        return samples


class _ModelSampling:
    sigma_max = 1.0

    def noise_scaling(
        self, sigma: torch.Tensor, noise: torch.Tensor, latent: torch.Tensor, _maximum: bool
    ) -> torch.Tensor:
        return latent + noise * sigma

    def inverse_noise_scaling(self, _sigma: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        return samples


class _RealModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.latent_format = _LatentFormat()
        self.model_sampling = _ModelSampling()

    def get_dtype(self) -> torch.dtype:
        return torch.float32

    def get_model_object(self, name: str) -> object:
        return getattr(self, name)

    def memory_required(self, *_args: object, **_kwargs: object) -> int:
        return 0

    def extra_conds(self, **_kwargs: object) -> dict[str, object]:
        return {}

    def scale_latent_inpaint(self, **values: torch.Tensor) -> torch.Tensor:
        return values["latent_image"]

    def process_latent_in(self, samples: torch.Tensor) -> torch.Tensor:
        return samples

    def process_latent_out(self, samples: torch.Tensor) -> torch.Tensor:
        return samples * 3


def test_sampler_custom_uses_actual_cpu_cfg_guider_and_sampler(monkeypatch: Any) -> None:
    from dinkster_api.v1 import initialize_inference_process

    initialize_inference_process()
    from dinkster_inference import samplers
    from dinkster_inference.model_patcher import ModelPatcher

    patcher = ModelPatcher(_RealModel(), torch.device("cpu"), torch.device("cpu"))
    expected = torch.tensor([[[[3.0, -7.0]]], [[[11.0, 5.0]]]])
    original = samplers.calc_cond_batch
    monkeypatch.setattr(
        samplers,
        "calc_cond_batch",
        lambda _model, conditions, _samples, _sigma, _options: [
            expected if condition is not None else torch.zeros_like(expected)
            for condition in conditions
        ],
    )
    try:
        output = custom_sampling.SamplerCustom.execute(
            model=patcher,
            add_noise=False,
            noise_seed=99,
            cfg=1.0,
            positive=[],
            negative=[],
            sampler=samplers.ksampler("euler"),
            sigmas=torch.tensor([1.0, 0.0]),
            latent_image={
                "samples": torch.tensor([[[[2.0, -4.0]]], [[[8.0, 1.0]]]]),
                "batch_index": [6, 2],
                "noise_mask": torch.tensor([[[[1.0, 0.0]]]]),
                "origin": "real-cpu",
            },
        )
    finally:
        monkeypatch.setattr(samplers, "calc_cond_batch", original)
    assert output["output"]["batch_index"] == [6, 2]
    assert output["output"]["origin"] == "real-cpu"
    expected_masked = torch.tensor([[[[9.0, -12.0]]], [[[33.0, 3.0]]]])
    torch.testing.assert_close(output["output"]["samples"], expected_masked)
    torch.testing.assert_close(output["denoised_output"]["samples"], expected_masked)


def test_sampler_custom_preserves_source_metadata_and_returns_callback_denoised_tensor(
    monkeypatch: Any,
) -> None:
    sample = _Sample()
    monkeypatch.setattr(custom_sampling, "_runtime", lambda: sample)
    latent = {
        "samples": torch.tensor([[[[1.0, -4.0]]], [[[3.0, 8.0]]]]),
        "batch_index": [9, 2],
        "noise_mask": torch.tensor([[[[0.0, 1.0]]]]),
        "downscale_ratio_spacial": 8,
        "downscale_ratio_temporal": 2,
        "source": "kept",
    }
    output = custom_sampling.SamplerCustom.execute(
        model=_Model(),
        add_noise=True,
        noise_seed=19,
        cfg=2.5,
        positive=object(),
        negative=object(),
        sampler=object(),
        sigmas=torch.tensor([3.0, 1.0, 0.0]),
        latent_image=latent,
    )
    assert sample.noise_args == (19, [9, 2])
    assert output["output"]["source"] == "kept"
    assert output["output"]["batch_index"] == [9, 2]
    assert "downscale_ratio_spacial" not in output["output"]
    torch.testing.assert_close(
        output["output"]["samples"], torch.tensor([[[[4.0, -6.0]]], [[[9.0, 16.0]]]])
    )
    torch.testing.assert_close(
        output["denoised_output"]["samples"], torch.tensor([[[[0.0, 0.0]]], [[[-10.0, 20.0]]]])
    )


def test_advanced_sampler_uses_the_guider_and_preserves_batch_metadata(monkeypatch: Any) -> None:
    sample = _Sample()
    monkeypatch.setattr(custom_sampling, "_runtime", lambda: sample)

    class Noise:
        seed = 41

        def generate_noise(self, latent: dict[str, object]) -> torch.Tensor:
            assert latent["batch_index"] == [4, 1]
            return torch.tensor([[[[4.0, -2.0]]], [[[3.0, 9.0]]]])

    class Guider:
        model_patcher = _Model()

        def sample(
            self,
            noise: torch.Tensor,
            samples: torch.Tensor,
            sampler: object,
            sigmas: torch.Tensor,
            **kwargs: object,
        ) -> torch.Tensor:
            del sampler, sigmas
            assert kwargs["denoise_mask"] is not None and kwargs["seed"] == 41
            kwargs["callback"](0, samples - noise, samples, 1)
            return samples + noise * 2

    output = custom_sampling.SamplerCustomAdvanced.execute(
        noise=Noise(),
        guider=Guider(),
        sampler=object(),
        sigmas=torch.tensor([2.0, 0.0]),
        latent_image={
            "samples": torch.tensor([[[[2.0, 5.0]]], [[[7.0, -1.0]]]]),
            "batch_index": [4, 1],
            "noise_mask": torch.ones((1, 1, 1, 2)),
            "label": "source",
        },
    )
    assert output["output"]["batch_index"] == [4, 1]
    assert output["output"]["label"] == "source"
    torch.testing.assert_close(
        output["output"]["samples"], torch.tensor([[[[11.0, 2.0]]], [[[14.0, 18.0]]]])
    )
    torch.testing.assert_close(
        output["denoised_output"]["samples"], torch.tensor([[[[-10.0, 80.0]]], [[[50.0, -90.0]]]])
    )
