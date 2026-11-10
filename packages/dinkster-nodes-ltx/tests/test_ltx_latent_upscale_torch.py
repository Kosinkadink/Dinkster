"""LTX latent upscaling normalization and source metadata."""

from types import SimpleNamespace

import torch
from dinkster_ltx.latent_upscale import LTXVLatentUpsampler


def test_ltx_normalizes_around_the_model_and_removes_noise_mask(monkeypatch) -> None:
    calls: list[object] = []

    class Statistics:
        def un_normalize(self, value: torch.Tensor) -> torch.Tensor:
            calls.append("un")
            return value + 10

        def normalize(self, value: torch.Tensor) -> torch.Tensor:
            calls.append("norm")
            return value - 4

    class Model:
        load_device = torch.device("cpu")
        model = staticmethod(lambda value: calls.append("model") or value * 2)

        @staticmethod
        def model_dtype() -> torch.dtype:
            return torch.float64

    management = SimpleNamespace(
        load_models_gpu=lambda models, memory_required: calls.append((models, memory_required)),
        intermediate_device=lambda: torch.device("cpu"),
    )
    monkeypatch.setattr("dinkster_ltx.latent_upscale.initialize_inference_process", lambda: None)
    monkeypatch.setattr(
        "dinkster_ltx.latent_upscale.importlib",
        SimpleNamespace(import_module=lambda _name: management),
    )
    source = {
        "samples": torch.tensor([[[[2.0]]]], dtype=torch.float32),
        "noise_mask": object(),
        "keep": 7,
    }
    vae = SimpleNamespace(first_stage_model=SimpleNamespace(per_channel_statistics=Statistics()))
    output = LTXVLatentUpsampler.execute(samples=source, upscale_model=Model(), vae=vae)["LATENT"]
    assert calls[0][1] == 3000.0
    assert calls[1:] == ["un", "model", "norm"]
    assert output["samples"].dtype is torch.float32
    assert torch.equal(output["samples"], torch.tensor([[[[20.0]]]]))
    assert output["keep"] == 7 and "noise_mask" not in output
