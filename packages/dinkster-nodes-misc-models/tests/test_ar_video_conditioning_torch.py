from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    initialize_inference_process,
    materialize_latent,
)
from dinkster_misc_models.ar_video import ARVideoI2V


class Encoder:
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        assert pixels.shape == (1, 16, 32, 3)
        return pixels[:, ::8, ::8, :1].permute(0, 3, 1, 2).unsqueeze(2)


def image() -> torch.Tensor:
    return torch.arange(2 * 4 * 8 * 4, dtype=torch.float32).reshape(2, 4, 8, 4) / 128


def model() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    patcher = ModelPatcher(torch.nn.Linear(2, 3), torch.device("cpu"), torch.device("cpu"))
    patcher.model_options["transformer_options"] = {
        "preserved": 19,
        "ar_config": {"frames": 3, "initial_latent": "replace-me"},
    }
    return patcher


def check(patched: Any, latent: object, frames: int) -> None:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    assert type(patched) is ModelPatcher
    assert patched.model_options["transformer_options"]["preserved"] == 19
    ar_config = patched.model_options["transformer_options"]["ar_config"]
    assert ar_config["frames"] == 3
    # Independent bilinear samples at output (y,x)=(0,0),(0,8),...,(8,24).
    expected = torch.tensor([[0, 6.5, 14.5, 22.5], [52, 58.5, 66.5, 74.5]]) / 128
    encoded = ar_config["initial_latent"]
    torch.testing.assert_close(encoded, expected.reshape(1, 1, 1, 2, 4), rtol=0, atol=0)
    assert encoded.dtype == torch.float32 and encoded.device.type == "cpu"
    samples = materialize_latent(latent)["samples"]
    assert samples.shape == (2, 16, frames, 2, 4)
    assert samples.dtype == torch.float32 and samples.device.type == "cpu"
    assert not samples.count_nonzero()


@pytest.mark.parametrize(("length", "frames"), ((1, 1), (8, 2), (9, 3)))
def test_start_image_conditions_clone_without_modifying_sources(length: int, frames: int) -> None:
    source, pixels = model(), image()
    before = pixels.clone()
    result = ARVideoI2V.execute(
        model=source,
        vae=Encoder(),
        start_image=pixels,
        width=32,
        height=16,
        length=length,
        batch_size=2,
    )
    assert result["MODEL"] is not source
    check(result["MODEL"], result["LATENT"], frames)
    assert source.model_options["transformer_options"] == {
        "preserved": 19,
        "ar_config": {"frames": 3, "initial_latent": "replace-me"},
    }
    assert torch.equal(pixels, before)
