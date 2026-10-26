"""CPU behavior coverage for native image-control model patches."""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DINKSTER_ACCELERATOR"] = "cpu"

import pytest
import torch
from dinkster_api.v1 import (
    MODEL_TYPE,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_nodes_comfy.image_control_patches import (
    QwenImage21FunControlBlockPatch,
    QwenImageDiffsynthControlnet,
    USOStyleReference,
    ZImageFunControlnet,
)


@pytest.fixture(autouse=True)
def _initialize_runtime() -> None:
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


class _Projector(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.sum(dim=0)


class ImageControlModelSource(Node):
    """Ordinary host fixture with the native public MODEL contract."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="image-control-fixture.model-source",
            inputs=(),
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(MODEL=_patcher(_Host()))


def _installed(patcher: Any, name: str) -> Any:
    return patcher.model_options["transformer_options"]["patches"][name][0]


def test_diffsynth_dispatch_slices_rgb_scales_and_preserves_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_qwen_fun_control_replaces_blocks_restores_inactive_path_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_inference.ldm.lumina.controlnet as z_runtime
    import dinkster_inference.ldm.qwen_image21.model as qwen_runtime

    monkeypatch.setattr(qwen_runtime, "QwenImage21FunControl", _QwenFunControl)
    monkeypatch.setattr(z_runtime, "ZImage_Control", _ZImage)
    source = _patcher(_Host())
    control = _patcher(_QwenFunControl())
    patched = ZImageFunControlnet.execute(
        model=source,
        model_patch=control,
        vae=_Vae(),
        image=torch.ones(1, 2, 2, 3),
        strength=0.5,
    )["MODEL"]
    replacements = patched.model_options["transformer_options"]["patches_replace"]["dit"]
    first = replacements[("single_block", 0)]
    assert isinstance(first, QwenImage21FunControlBlockPatch)
    assert sorted(key[1] for key in replacements) == [0, 2]
    assert source.model_options == {"transformer_options": {}}
    fun = first.control_patch
    fun.active = True
    fun.control = torch.full((1, 129, 1, 2), 3.0)
    original = torch.ones(1, 2, 3)
    output = first(
        {
            "img": original,
            "prefix_len": 0,
            "mod": None,
            "pe": None,
            "attn_fn": None,
            "transformer_options": {},
        },
        {"original_block": lambda args: {"img": args["img"] + 2.0}},
    )
    assert torch.equal(original, torch.ones_like(original))
    assert torch.equal(output["img"], torch.full_like(output["img"], 5.0))
    inactive_options = {"patches_replace": {"dit": replacements}, "sigmas": torch.tensor([8.0])}
    forwarded = fun.diffusion_model_wrapper(
        lambda *_args: _args[5], torch.zeros(1, 1, 1, 2), None, None, None, None, inactive_options
    )
    assert forwarded["patches_replace"]["dit"] == {}
    fun.to(torch.device("cpu"))
    fun.cleanup()
    assert fun.control is fun.stream is fun.pristine is None
    assert not fun.active


def test_uso_reference_projects_stacked_features_and_preserves_source() -> None:
    source = _patcher(_Host())
    projector = _patcher(_Projector())
    vision = SimpleNamespace(
        all_hidden_states=torch.arange(1 * 21 * 2 * 3, dtype=torch.float32).reshape(1, 21, 2, 3),
        penultimate_hidden_states=torch.full((1, 2, 3), 99.0),
    )
    patched = USOStyleReference.execute(
        model=source, model_patch=projector, clip_vision_output=vision
    )["MODEL"]
    patch = _installed(patched, "post_input")
    text, ids = torch.tensor([[[7.0, 8.0, 9.0]]]), torch.ones(1, 1, 3)
    output = patch({"txt": text, "txt_ids": ids})
    expected = (
        vision.all_hidden_states[:, -20]
        + vision.all_hidden_states[:, -11]
        + vision.penultimate_hidden_states
    )
    assert source.model_options == {"transformer_options": {}}
    assert output["txt"].shape == (1, 3, 3)
    assert torch.equal(output["txt"][:, :2], expected)
    assert torch.equal(output["txt"][:, 2:], text)
    assert torch.equal(output["txt_ids"][:, :2], torch.zeros(1, 2, 3))
    assert torch.equal(output["txt_ids"][:, 2:], ids)
    assert patch.models() == [projector]
    assert patch.to(torch.device("cpu")) is patch


def test_image_control_fixture_uses_native_public_model_type() -> None:
    schema = ImageControlModelSource.define_schema()
    assert schema.outputs[0].id == "MODEL"
    assert ImageControlModelSource.execute()["MODEL"].__class__.__name__ == "ModelPatcher"
