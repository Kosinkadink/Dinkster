"""Hunyuan latent upscale dimension ordering and metadata."""

from types import SimpleNamespace

import torch
from dinkster_hunyuan.latent_upscale import HunyuanVideo15LatentUpscaleWithModel


def test_hunyuan_zero_dimensions_returns_the_original_public_latent() -> None:
    source = {"samples": torch.ones((1, 2, 3, 5)), "metadata": "kept"}
    result = HunyuanVideo15LatentUpscaleWithModel.execute(
        model=object(),
        samples=source,
        upscale_method="bilinear",
        width=0,
        height=0,
        crop="disabled",
    )["LATENT"]
    assert result is source


def test_hunyuan_one_zero_dimension_uses_width_height_in_the_core_order(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    class Model:
        def resample_latent(self, latent: torch.Tensor) -> torch.Tensor:
            return latent + 3

    monkeypatch.setattr(
        "dinkster_hunyuan.latent_upscale.initialize_inference_process", lambda: None
    )
    monkeypatch.setattr(
        "dinkster_hunyuan.latent_upscale.importlib",
        SimpleNamespace(
            import_module=lambda name: SimpleNamespace(
                common_upscale=lambda tensor, width, height, _method, _crop: (
                    calls.append((width, height)) or tensor.new_ones((1, 2, height, width))
                )
            )
        ),
    )
    source = {"samples": torch.zeros((1, 2, 3, 5), dtype=torch.float64), "noise_mask": object()}
    output = HunyuanVideo15LatentUpscaleWithModel.execute(
        model=Model(), samples=source, upscale_method="bilinear", width=0, height=96, crop="center"
    )["LATENT"]
    assert calls == [(10, 6)]
    assert set(output) == {"samples"}
    assert output["samples"].shape == (1, 2, 6, 10)
    assert output["samples"].dtype is torch.float32
