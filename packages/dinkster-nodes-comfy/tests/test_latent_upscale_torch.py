"""CPU contracts for native latent upscale consumers."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.latent_upscale import (
    HunyuanVideo15LatentUpscaleWithModel,
    LTXVLatentUpsampler,
)


def test_registry_and_schemas_are_core_compatible_without_torch_runtime() -> None:
    assert [
        node.schema().node_type
        for node in COMFY_NODES
        if node
        in {
            HunyuanVideo15LatentUpscaleWithModel,
            LTXVLatentUpsampler,
        }
    ] == ["HunyuanVideo15LatentUpscaleWithModel", "LTXVLatentUpsampler"]
    hunyuan = HunyuanVideo15LatentUpscaleWithModel.schema()
    assert [input.id for input in hunyuan.inputs] == [
        "model",
        "samples",
        "upscale_method",
        "width",
        "height",
        "crop",
    ]
    assert hunyuan.inputs[2].default == "bilinear"
    assert [input.id for input in LTXVLatentUpsampler.schema().inputs] == [
        "samples",
        "upscale_model",
        "vae",
    ]


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
        "dinkster_nodes_comfy.latent_upscale.initialize_inference_process", lambda: None
    )
    monkeypatch.setattr(
        "dinkster_nodes_comfy.latent_upscale.importlib.import_module",
        lambda name: SimpleNamespace(
            common_upscale=lambda tensor, width, height, _method, _crop: (
                calls.append((width, height)) or tensor.new_ones((1, 2, height, width))
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
    monkeypatch.setattr(
        "dinkster_nodes_comfy.latent_upscale.initialize_inference_process", lambda: None
    )
    monkeypatch.setattr(
        "dinkster_nodes_comfy.latent_upscale.importlib.import_module", lambda _name: management
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
