from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    initialize_inference_process,
)
from dinkster_qwen.image_control_patches import QwenImageDiffsynthControlnet


class _Sampling:
    def percent_to_sigma(self, value: float) -> float:
        return 10.0 - value


class _Diffusion(torch.nn.Module):
    def __init__(self, blocks: int = 4) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(blocks))


class _Host(torch.nn.Module):
    def __init__(self, blocks: int = 4) -> None:
        super().__init__()
        self.model_sampling = _Sampling()
        self.diffusion_model = _Diffusion(blocks)


def _patcher(model: torch.nn.Module) -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    return ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))


class _Vae:
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        channels = image.movedim(-1, 1)
        return torch.cat(
            (channels, torch.zeros_like(channels[:, :1]).expand(-1, 13, -1, -1)), dim=1
        )

    def spacial_compression_encode(self) -> int:
        return 1


class _DiffSynth(torch.nn.Module):
    additional_in_dim = 4

    def process_input_latent_image(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def control_block(
        self, _image: torch.Tensor, encoded: torch.Tensor, block: int
    ) -> torch.Tensor:
        return encoded + block + 1


class _ZImage(torch.nn.Module):
    additional_in_dim = 0
    n_control_layers = 3

    def forward(
        self, _text: torch.Tensor, _image: torch.Tensor, _pe: torch.Tensor, _vec: torch.Tensor
    ) -> torch.Tensor:
        return torch.full((1, 4, 3), 2.0)

    def forward_control_block(
        self, layer: int, state: torch.Tensor, *_args: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.full_like(state, layer + 1.0), state + 10.0

    def forward_noise_refiner_block(
        self, block: int, state: torch.Tensor, *_args: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.full_like(state, block + 3.0), state + 20.0


class _QwenFunControl(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.control_blocks = torch.nn.ModuleList((torch.nn.Identity(), torch.nn.Identity()))

    def init_stream(
        self, pristine: torch.Tensor, _control: torch.Tensor, _prefix_len: int
    ) -> torch.Tensor:
        return pristine

    def step(
        self, index: int, stream: torch.Tensor, *_args: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return stream + index + 1, torch.full_like(stream, index + 4.0)


def _installed(patcher: Any, name: str) -> Any:
    return patcher.model_options["transformer_options"]["patches"][name][0]


def test_diffsynth_dispatch_slices_rgb_scales_and_preserves_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_inference_process()
    import dinkster_inference.ldm.lumina.controlnet as z_runtime
    import dinkster_inference.ldm.qwen_image21.model as qwen_runtime

    monkeypatch.setattr(qwen_runtime, "QwenImage21FunControl", _QwenFunControl)
    monkeypatch.setattr(z_runtime, "ZImage_Control", _ZImage)
    source = _patcher(_Host())
    control = _patcher(_DiffSynth())
    image = torch.tensor([[[[1.0, 2.0, 3.0, 99.0]]]])
    patched = QwenImageDiffsynthControlnet.execute(
        model=source,
        model_patch=control,
        vae=_Vae(),
        image=image,
        strength=0.25,
    )["MODEL"]
    patch = _installed(patched, "double_block")
    values = torch.zeros(1, 17, 1, 1)
    result = patch(
        {
            "transformer_options": {"sigmas": torch.tensor([9.5])},
            "x": torch.zeros(1, 1, 1, 1),
            "img": values,
            "block_index": 1,
        }
    )
    assert source.model_options == {"transformer_options": {}}
    assert result["img"] is values
    assert torch.equal(values[0, :4, 0, 0], torch.tensor([0.75, 1.0, 1.25, 0.5]))
    untouched = torch.full_like(values, 5.0)
    patch(
        {
            "transformer_options": {"sigmas": torch.tensor([8.0])},
            "x": torch.zeros(1, 1, 1, 1),
            "img": untouched,
            "block_index": 1,
        }
    )
    assert torch.equal(untouched, torch.full_like(untouched, 5.0))
    assert patch.models() == [control]
    assert patch.to(torch.device("cpu")) is patch
