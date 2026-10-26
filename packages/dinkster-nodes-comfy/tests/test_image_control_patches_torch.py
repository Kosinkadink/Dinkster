"""CPU tests for Qwen, Z-Image, and USO image-control patches."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from dinkster_nodes_comfy.image_control_patches import (
    QwenImageDiffsynthControlnet,
    USOStyleReference,
    ZImageFunControlnet,
)


class _Sampling:
    def percent_to_sigma(self, value: float) -> float:
        return 10.0 - value


class _Model:
    def __init__(self) -> None:
        self.sampling = _Sampling()
        self.double: Any = None
        self.refiner: Any = None
        self.post: Any = None

    def clone(self) -> _Model:
        return _Model()

    def get_model_object(self, name: str) -> _Sampling:
        assert name == "model_sampling"
        return self.sampling

    def set_model_double_block_patch(self, patch: Any) -> None:
        self.double = patch

    def set_model_noise_refiner_patch(self, patch: Any) -> None:
        self.refiner = patch

    def set_model_post_input_patch(self, patch: Any) -> None:
        self.post = patch


class _Vae:
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image.movedim(-1, 1)


class _Qwen:
    additional_in_dim = 0

    def process_input_latent_image(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.movedim(1, -1).flatten(1, 2)

    def control_block(self, image: torch.Tensor, encoded: torch.Tensor, _: int) -> torch.Tensor:
        return encoded + 2


def test_qwen_patch_uses_asymmetric_values_without_mutating_source() -> None:
    source = _Model()
    image = torch.zeros(1, 2, 2, 3)
    patched = QwenImageDiffsynthControlnet.execute(
        model=source,
        model_patch=SimpleNamespace(model=_Qwen()),
        vae=_Vae(),
        image=image.numpy(),
        strength=0.25,
    )["MODEL"]
    assert source.double is None
    values = torch.full((1, 4, 3), 3.0)
    output = patched.double(
        {"transformer_options": {"sigmas": torch.tensor([8.0])}, "img": values, "block_index": 1}
    )
    assert output["img"].shape == (1, 4, 3)
    assert torch.equal(output["img"], torch.full((1, 4, 3), 3.5))


def test_zimage_installs_both_patch_points_and_schema_ports_are_native() -> None:
    model = _Model()
    patched = ZImageFunControlnet.execute(
        model=model,
        model_patch=SimpleNamespace(model=SimpleNamespace(n_control_layers=3)),
        vae=_Vae(),
        image=torch.ones(1, 2, 2, 3).numpy(),
    )["MODEL"]
    assert patched.double is patched.refiner
    assert [item.name for item in ZImageFunControlnet.define_schema().inputs] == [
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


def test_uso_patch_prepends_projected_tokens_and_preserves_inputs() -> None:
    model = _Model()
    projector = torch.nn.Identity()
    vision = SimpleNamespace(
        all_hidden_states=torch.arange(1 * 21 * 2 * 3, dtype=torch.float32).reshape(1, 21, 2, 3),
        penultimate_hidden_states=torch.full((1, 2, 3), 99.0),
    )
    patched = USOStyleReference.execute(
        model=model, model_patch=SimpleNamespace(model=projector), clip_vision_output=vision
    )["MODEL"]
    text = torch.tensor([[[7.0, 8.0, 9.0]]])
    ids = torch.ones(1, 1, 3)
    output = patched.post({"txt": text, "txt_ids": ids})
    assert model.post is None
    assert output["txt"].shape == (3, 3, 3)
    assert torch.equal(output["txt"][0, 2], torch.tensor([99.0, 99.0, 99.0]))
    assert torch.equal(output["txt_ids"][:, :-1], torch.zeros(1, 2, 3))
    assert torch.equal(output["txt_ids"][:, -1:], ids)
