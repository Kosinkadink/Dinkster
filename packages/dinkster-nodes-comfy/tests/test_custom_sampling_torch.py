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
        callback({"denoised": samples - noise})
        assert cfg == 2.5 and kwargs["noise_mask"] is not None and kwargs["seed"] == 19
        return samples + noise


class _Preview:
    def prepare_callback(self, _model: object, steps: int, output: dict[str, object]) -> Any:
        assert steps in (1, 2)

        def callback(state: dict[str, torch.Tensor]) -> None:
            output["x0"] = state["denoised"]

        return callback


class _Model:
    def __init__(self) -> None:
        self.model = SimpleNamespace(process_latent_out=lambda tensor: tensor * 10)


def test_sampler_custom_preserves_source_metadata_and_returns_callback_denoised_tensor(
    monkeypatch: Any,
) -> None:
    sample = _Sample()
    monkeypatch.setattr(custom_sampling, "_runtime", lambda: (sample, _Preview()))
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
    monkeypatch.setattr(custom_sampling, "_runtime", lambda: (sample, _Preview()))

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
            kwargs["callback"]({"denoised": samples - noise})
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
