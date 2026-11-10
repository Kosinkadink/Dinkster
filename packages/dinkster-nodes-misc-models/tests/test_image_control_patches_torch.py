from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    initialize_inference_process,
)
from dinkster_misc_models.image_control_patches import ZImageFunControlnet


@pytest.fixture(autouse=True)
def initialize_runtime() -> None:
    initialize_inference_process()


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


class _ZImageInpaint(_ZImage):
    additional_in_dim = 1


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


class _QwenLatentFormat:
    latent_channels = 16

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


def _installed(patcher: Any, name: str) -> Any:
    return patcher.model_options["transformer_options"]["patches"][name][0]


def test_zimage_dispatch_uses_real_model_patcher_for_normal_and_refiner_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_inference.ldm.lumina.controlnet as z_runtime
    import dinkster_inference.ldm.qwen_image21.model as qwen_runtime

    monkeypatch.setattr(qwen_runtime, "QwenImage21FunControl", _QwenFunControl)
    monkeypatch.setattr(z_runtime, "ZImage_Control", _ZImage)
    source = _patcher(_Host())
    control = _patcher(_ZImage())
    patched = ZImageFunControlnet.execute(
        model=source,
        model_patch=control,
        vae=_Vae(),
        image=torch.ones(1, 2, 2, 4),
        strength=0.25,
    )["MODEL"]
    normal, refiner = _installed(patched, "double_block"), _installed(patched, "noise_refiner")
    assert normal is refiner
    assert source.model_options == {"transformer_options": {}}
    common = {
        "transformer_options": {"sigmas": torch.tensor([9.5])},
        "x": torch.zeros(1, 3, 2, 2),
        "img_input": torch.zeros(1, 4, 3),
        "txt": torch.ones(1, 1, 3),
        "pe": torch.ones(1, 1, 3),
        "vec": torch.ones(1, 3),
    }
    image = torch.zeros(1, 4, 3)
    output = normal({**common, "img": image, "block_index": 0})
    assert "img" not in output and "txt" not in output
    assert torch.equal(image, torch.full_like(image, 0.25))
    normal({**common, "img": image, "block_index": 10})
    assert torch.equal(image, torch.full_like(image, 0.75))
    refiner_image = torch.zeros(1, 4, 3)
    refiner({**common, "img": refiner_image, "block_index": 2, "block_type": "noise_refiner"})
    assert torch.equal(refiner_image, torch.full_like(refiner_image, 1.25))
    normal.to(torch.device("cpu"))
    assert normal.temp_data is None
    assert [input.id for input in ZImageFunControlnet.define_schema().inputs] == [
        "model",
        "model_patch",
        "vae",
        "strength",
        "image",
        "inpaint_image",
        "mask",
        "start_percent",
        "end_percent",
    ]


def test_zimage_inpaint_materializes_and_inverts_the_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_inference.ldm.lumina.controlnet as z_runtime
    import dinkster_inference.ldm.qwen_image21.model as qwen_runtime

    monkeypatch.setattr(qwen_runtime, "QwenImage21FunControl", _QwenFunControl)
    monkeypatch.setattr(z_runtime, "ZImage_Control", _ZImageInpaint)
    patched = ZImageFunControlnet.execute(
        model=_patcher(_Host()),
        model_patch=_patcher(_ZImageInpaint()),
        vae=_Vae(),
        image=torch.full((1, 2, 2, 4), 0.25),
        inpaint_image=torch.full((1, 2, 2, 4), 0.75),
        mask=torch.zeros(1, 2, 2),
    )["MODEL"]
    patch = _installed(patched, "double_block")
    assert patch.encoded_image.shape == (1, 33, 2, 2)
    assert torch.equal(patch.mask, torch.ones(1, 1, 1, 2, 2))


def test_qwen_prepare_keeps_raw_mask_for_active_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_inference.latent_formats as latent_formats
    import dinkster_inference.ldm.lumina.controlnet as z_runtime
    import dinkster_inference.ldm.qwen_image21.model as qwen_runtime

    monkeypatch.setattr(qwen_runtime, "QwenImage21FunControl", _QwenFunControl)
    monkeypatch.setattr(z_runtime, "ZImage_Control", _ZImage)
    monkeypatch.setattr(latent_formats, "QwenImage21", _QwenLatentFormat)
    patched = ZImageFunControlnet.execute(
        model=_patcher(_Host()),
        model_patch=_patcher(_QwenFunControl()),
        vae=_Vae(),
        image=torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]),
        inpaint_image=torch.tensor([[[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]]]),
        mask=torch.tensor([[[0.0, 1.0]]]),
    )["MODEL"]
    fun = patched.model_options["transformer_options"]["patches_replace"]["dit"][
        (
            "single_block",
            0,
        )
    ].control_patch
    result = fun.diffusion_model_wrapper(
        lambda *_args: _args[5],
        torch.zeros(1, 1, 1, 2),
        None,
        None,
        None,
        None,
        {"sigmas": torch.tensor([9.5])},
    )
    assert fun.active and result["sigmas"].item() == 9.5
    assert fun.control is not None
    assert torch.equal(fun.control[0, :3, 0], torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))
    assert torch.equal(fun.control[0, 16, 0], torch.tensor([1.0, 0.0]))
    assert torch.equal(
        fun.control[0, 17:20, 0], torch.tensor([[10.0, 0.5], [20.0, 0.5], [30.0, 0.5]])
    )
