from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    AssetRef,
    initialize_inference_process,
)
from dinkster_assets import digest_file
from dinkster_sd import photomaker
from dinkster_sd.photomaker import PhotoMakerEncode, PhotoMakerLoader


class IdentityModel:
    load_device = torch.device("cpu")

    def __init__(self) -> None:
        self.pixels: Any = None
        self.mask: Any = None

    def __call__(self, *, id_pixel_values: Any, prompt_embeds: Any, class_tokens_mask: Any) -> Any:
        self.pixels = id_pixel_values
        self.mask = class_tokens_mask
        return prompt_embeds + class_tokens_mask[..., None] * 7


class Clip:
    def __init__(self) -> None:
        self.tokens = {
            "g": [[(11, 1.0, 0), (12, 0.4, 2), (13, 0.9, 3), (99, 1.0, -1)]],
            "l": [[(21, 0.3, 2), (22, 0.7, 2), (23, 0.6, 4), (98, 1.0, -1)]],
        }
        self.cond = torch.arange(77 * 3, dtype=torch.float32).reshape(1, 77, 3) / 13
        self.pooled = torch.tensor([[2.0, -3.0]])
        self.received: Any = None

    def tokenize(self, text: str, *, return_word_ids: bool) -> Any:
        assert return_word_ids
        return self.tokens

    def encode_from_tokens(self, tokens: Any, *, return_pooled: bool) -> Any:
        assert return_pooled
        self.received = tokens
        return self.cond, self.pooled


@pytest.mark.parametrize("text", ("portrait photomaker sunset", "portrait without marker"))
def test_encode_removes_selected_word_and_preserves_padding_and_pooled_identity(text: str) -> None:
    model = IdentityModel()
    clip = Clip()
    original = {key: [list(chunk) for chunk in chunks] for key, chunks in clip.tokens.items()}
    image = torch.linspace(0, 1, 32 * 48 * 3).reshape(1, 32, 48, 3)
    result: Any = PhotoMakerEncode.execute(photomaker=model, image=image, clip=clip, text=text)[
        "CONDITIONING"
    ]
    assert result[0][1]["pooled_output"] is clip.pooled
    assert clip.tokens == original
    if "photomaker" in text:
        assert clip.received == {
            "g": [[(11, 1.0, 0), (13, 0.9, 3), (99, 1.0, -1), (99, 1.0, -1)]],
            "l": [[(23, 0.6, 4), (98, 1.0, -1), (98, 1.0, -1), (98, 1.0, -1)]],
        }
        assert model.mask.shape == (1, 77)
        assert model.mask.nonzero().tolist() == [[0, 1]]
        expected = clip.cond.clone()
        expected[:, 1] += 7
        torch.testing.assert_close(result[0][0], expected, rtol=0, atol=0)
        assert model.pixels.shape == (1, 1, 3, 224, 224)
        assert model.pixels.dtype == torch.float32
    else:
        assert result[0][0] is clip.cond
        assert model.mask is None
        # F uses index=-1 when absent, removing matching end-token word IDs.
        assert clip.received["g"][0] == original["g"][0]
        assert clip.received["l"][0] == original["l"][0]


def test_fusion_preserves_unselected_tokens_and_uses_ordered_valid_ids() -> None:
    fusion: Any = photomaker._fusion_module(torch.nn)(3)
    with torch.no_grad():
        for index, parameter in enumerate(fusion.parameters()):
            parameter.copy_(
                torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
                * 0.017
                + (index + 1) * 0.031
            )
    prompt = torch.tensor([[[1.0, -2.0, 3.0], [7.0, 0.5, -4.0], [-1.0, 8.0, 2.0]]])
    ids = torch.tensor([[[[0.2, -0.4, 0.8]], [[1.3, 2.1, -0.7]], [[99.0, 99.0, 99.0]]]])
    mask = torch.tensor([[True, False, True]])
    before = prompt.clone()

    def mlp(x: Any, module: Any) -> Any:
        normalized = torch.nn.functional.layer_norm(
            x, (x.shape[-1],), module.layernorm.weight, module.layernorm.bias
        )
        hidden = torch.nn.functional.gelu(
            torch.nn.functional.linear(normalized, module.fc1.weight, module.fc1.bias)
        )
        value = torch.nn.functional.linear(hidden, module.fc2.weight, module.fc2.bias)
        return value + x if module.use_residual else value

    selected = before[:, [0, 2]].reshape(2, 3)
    combined = mlp(torch.cat((selected, ids[0, :2, 0]), dim=-1), fusion.mlp1) + selected
    expected = torch.nn.functional.layer_norm(
        mlp(combined, fusion.mlp2),
        (3,),
        fusion.layer_norm.weight,
        fusion.layer_norm.bias,
    )
    result = fusion(prompt, ids.to(torch.float64), mask)
    assert result.data_ptr() == prompt.data_ptr()
    torch.testing.assert_close(result[:, 1], before[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(result[:, [0, 2]].reshape(2, 3), expected, rtol=0, atol=0)


@pytest.mark.parametrize("nested", (True, False))
def test_loader_safely_reads_admitted_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool
) -> None:
    initialize_inference_process()
    data = {"visual_projection_2.weight": torch.tensor([[1.0, -3.0], [2.0, 4.0]])}
    path = tmp_path / "identity.pt"
    torch.save({"id_encoder": data} if nested else data, path)

    class Resolver:
        def resolve(self, digest: str) -> Path:
            return path

    class Model:
        state: Any = None

        def load_state_dict(self, state: Any) -> None:
            self.state = state

    model = Model()
    monkeypatch.setattr(photomaker, "_id_encoder", lambda: model)
    asset = AssetRef(
        digest=digest_file(path),
        name=path.name,
        size=path.stat().st_size,
        resolver=Resolver(),
    )
    result = PhotoMakerLoader.execute(photomaker_model_name=asset)
    assert result["PHOTOMAKER"] is model
    assert list(model.state) == ["visual_projection_2.weight"]
    torch.testing.assert_close(
        model.state["visual_projection_2.weight"], data["visual_projection_2.weight"]
    )


def test_identity_constructor_and_projection_order_use_pinned_vision_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    clip_model = importlib.import_module("dinkster_inference.clip_model")
    monkeypatch.setattr(management, "text_encoder_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(management, "text_encoder_offload_device", lambda: torch.device("meta"))
    monkeypatch.setattr(management, "text_encoder_dtype", lambda device: torch.float32)
    with torch.device("meta"):
        model = photomaker._id_encoder()
    assert isinstance(model, clip_model.CLIPVisionModelProjection)
    assert model.load_device == torch.device("cpu")
    state = model.state_dict()
    assert state["visual_projection.weight"].shape == (768, 1024)
    assert state["visual_projection_2.weight"].shape == (1280, 1024)
    assert state["fuse_module.mlp1.fc1.weight"].shape == (2048, 4096)
    assert state["fuse_module.mlp1.fc2.weight"].shape == (2048, 2048)
    assert state["fuse_module.mlp2.layernorm.weight"].shape == (2048,)
    assert state["fuse_module.layer_norm.weight"].shape == (2048,)
    assert "vision_model.encoder.layers.23.self_attn.q_proj.weight" in state
    assert all(parameter.device.type == "meta" for parameter in model.parameters())

    class Vision(torch.nn.Module):
        def forward(self, pixels: Any) -> Any:
            assert pixels.shape == (2, 3, 4, 6)
            return (None, None, pixels[:, 0, 0, 0, None].expand(2, 1024))

    class Projection(torch.nn.Module):
        def __init__(self, size: int, offset: float) -> None:
            super().__init__()
            self.size = size
            self.offset = offset

        def forward(self, features: Any) -> Any:
            return features[:, : self.size] + self.offset

    class Fusion(torch.nn.Module):
        def forward(self, prompts: Any, ids: Any, mask: Any) -> Any:
            assert ids.shape == (1, 2, 1, 2048)
            torch.testing.assert_close(ids[0, 0, 0, :768], torch.full((768,), 3.0))
            torch.testing.assert_close(ids[0, 1, 0, :768], torch.full((768,), 7.0))
            torch.testing.assert_close(ids[0, 0, 0, 768:], torch.full((1280,), -1.0))
            torch.testing.assert_close(ids[0, 1, 0, 768:], torch.full((1280,), 3.0))
            assert mask.nonzero().tolist() == [[0, 2]]
            return prompts

    class SecondProjection(torch.nn.Module):
        def forward(self, features: Any) -> Any:
            return features[:, :1].expand(2, 1280) - 2.0

    model.vision_model = Vision()
    model.visual_projection = Projection(768, 2.0)
    model.visual_projection_2 = SecondProjection()
    model.fuse_module = Fusion()
    pixels = torch.stack((torch.ones(3, 4, 6), torch.full((3, 4, 6), 5.0))).unsqueeze(0)
    prompts = torch.arange(3 * 2048, dtype=torch.float32).reshape(1, 3, 2048)
    assert model(pixels, prompts, torch.tensor([[False, False, True]])) is prompts
