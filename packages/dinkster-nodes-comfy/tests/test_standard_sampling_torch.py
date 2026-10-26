"""CPU execution behavior for standard sampler wrappers."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from dinkster_native.standard_sampling import StandardKSampler, StandardKSamplerAdvanced


def _install_sampling_runtime(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fix_empty_latent_channels(
        model: object, latent: torch.Tensor, spatial: object, temporal: object
    ) -> torch.Tensor:
        assert model == "model"
        assert spatial == 8
        assert temporal == 4
        return latent + 0.25

    def prepare_noise(latent: torch.Tensor, seed: int, batch_index: object) -> torch.Tensor:
        assert batch_index == (2, 0)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(latent.shape, generator=generator, dtype=latent.dtype)

    def prepare_empty_noise(latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(latent)

    def sample(
        model: object,
        noise: torch.Tensor,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: object,
        negative: object,
        latent: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        calls.append(
            {
                "model": model,
                "noise": noise.clone(),
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive": positive,
                "negative": negative,
                **kwargs,
            }
        )
        return latent + noise * cfg + steps

    modules = {
        "dinkster_inference.sample": SimpleNamespace(
            fix_empty_latent_channels=fix_empty_latent_channels,
            prepare_noise=prepare_noise,
            prepare_empty_noise=prepare_empty_noise,
            sample=sample,
        )
    }
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: modules.get(name) or real_import(name),
    )
    return calls


def _latent() -> dict[str, object]:
    return {
        "samples": torch.tensor(
            [[[[0.125, -1.5, 2.25], [3.5, -0.75, 0.625]]]], dtype=torch.float32
        ),
        "batch_index": (2, 0),
        "noise_mask": torch.tensor([[[[1.0, 0.0, 0.5], [0.25, 0.75, 1.0]]]]),
        "downscale_ratio_spacial": 8,
        "downscale_ratio_temporal": 4,
        "preserved": "source metadata",
    }


def test_ksampler_uses_seeded_noise_and_preserves_source_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_sampling_runtime(monkeypatch)
    source = _latent()
    original = source["samples"].clone()

    result = StandardKSampler.execute(
        model="model",
        seed=739,
        steps=3,
        cfg=1.25,
        sampler_name="dinkster.euler",
        scheduler="dinkster.normal",
        positive="positive",
        negative="negative",
        latent_image=source,
        denoise=0.4,
    )["latent"]

    generator = torch.Generator(device="cpu").manual_seed(739)
    expected_noise = torch.randn(original.shape, generator=generator)
    expected = original + 0.25 + expected_noise * 1.25 + 3
    torch.testing.assert_close(result["samples"], expected)
    assert torch.equal(source["samples"], original)
    assert result["preserved"] == "source metadata"
    assert "downscale_ratio_spacial" not in result
    assert "downscale_ratio_temporal" not in result
    assert len(calls) == 1
    call = calls[0]
    assert call["model"] == "model"
    torch.testing.assert_close(call["noise"], expected_noise)
    assert call["steps"] == 3
    assert call["cfg"] == 1.25
    assert call["sampler_name"] == "euler"
    assert call["scheduler"] == "normal"
    assert call["positive"] == "positive"
    assert call["negative"] == "negative"
    assert call["denoise"] == 0.4
    assert call["disable_noise"] is False
    assert call["start_step"] is None
    assert call["last_step"] is None
    assert call["force_full_denoise"] is False
    assert call["noise_mask"] is source["noise_mask"]
    assert call["callback"] is None
    assert call["disable_pbar"] is True
    assert call["seed"] == 739


def test_ksampler_advanced_disables_noise_and_forwards_step_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_sampling_runtime(monkeypatch)
    source = _latent()
    original = source["samples"].clone()

    result = StandardKSamplerAdvanced.execute(
        model="model",
        add_noise="disable",
        noise_seed=918,
        steps=5,
        cfg=0.75,
        sampler_name="dinkster.dpmpp_2m",
        scheduler="dinkster.karras",
        positive="positive",
        negative="negative",
        latent_image=source,
        start_at_step=1,
        end_at_step=4,
        return_with_leftover_noise="enable",
    )["latent"]

    torch.testing.assert_close(result["samples"], original + 0.25 + 5)
    call = calls[0]
    assert torch.equal(call["noise"], torch.zeros_like(original))
    assert call["sampler_name"] == "dpmpp_2m"
    assert call["scheduler"] == "karras"
    assert call["denoise"] == 1.0
    assert call["disable_noise"] is True
    assert call["start_step"] == 1
    assert call["last_step"] == 4
    assert call["force_full_denoise"] is False
    assert call["seed"] == 918

    StandardKSamplerAdvanced.execute(
        model="model",
        add_noise="enable",
        noise_seed=918,
        steps=5,
        cfg=0.75,
        sampler_name="dinkster.dpmpp_2m",
        scheduler="dinkster.karras",
        positive="positive",
        negative="negative",
        latent_image=source,
        start_at_step=1,
        end_at_step=4,
        return_with_leftover_noise="disable",
    )
    assert calls[1]["force_full_denoise"] is True
