from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_misc_models.sensenova_options import SenseNovaSamplingOptions


class Patcher:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            model_config=SimpleNamespace(
                sampling_settings={"shift": 0.75, "noise_scale": 1.75, "multiplier": 17}
            )
        )
        self.sampling: Any = None

    def clone(self) -> Patcher:
        result = Patcher()
        result.model = self.model
        result.sampling = self.sampling
        return result

    def add_object_patch(self, name: str, value: Any) -> None:
        assert name == "model_sampling"
        self.sampling = value


@pytest.mark.parametrize("shift", (1.0, 0.25, 3.5))
def test_sensenova_options_preserves_source_and_configured_noise_scale(shift: float) -> None:
    original = Patcher()
    patched = SenseNovaSamplingOptions.execute(model=original, shift=shift)["MODEL"]
    assert patched is not original
    assert original.sampling is None
    assert original.model.model_config.sampling_settings == {
        "shift": 0.75,
        "noise_scale": 1.75,
        "multiplier": 17,
    }
    sampling = patched.sampling
    base = torch.tensor([0.0, 0.17, 0.63, 1.0])
    expected = shift * base / (1.0 + (shift - 1.0) * base)
    torch.testing.assert_close(sampling.sigma((1.0 - base) * 1000), expected)
    assert len(sampling.sigmas) == 1001
    assert sampling.noise_scale == 1.75
    assert sampling.percent_to_sigma(0.0) == 1.0
    assert sampling.percent_to_sigma(1.0) == 0.0
    assert sampling.percent_to_sigma(0.4) == pytest.approx(shift * 0.6 / (1 + (shift - 1) * 0.6))
    noise = torch.full((1, 2, 33, 65), 2.0)
    latent = torch.full_like(noise, -3.0)
    scale = math.sqrt(2 * 3 / 64) * 1.75
    torch.testing.assert_close(
        sampling.noise_scaling(torch.tensor([0.4]), noise, latent),
        0.4 * scale * noise + 0.6 * latent,
    )
