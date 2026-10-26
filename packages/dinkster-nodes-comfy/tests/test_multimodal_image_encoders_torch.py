"""Qwen multimodal encoder CPU checks independent of reference appending."""

from __future__ import annotations

from typing import Any

import torch
from dinkster_nodes_comfy.mage_image_encoders import TextEncodeMageFlowEdit
from dinkster_nodes_comfy.ming_image_encoders import TextEncodeMingImageEdit
from dinkster_nodes_comfy.qwen_image_encoders import (
    QwenImage21Cache,
    TextEncodeQwenImage21,
    TextEncodeQwenImageEdit,
    TextEncodeQwenImageEditPlus,
)


class Clip:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def tokenize(self, text: str, **options: Any) -> dict[str, Any]:
        self.calls.append((text, options))
        return {"text": text, **options}

    def encode_from_tokens_scheduled(self, tokens: dict[str, Any]) -> list[list[Any]]:
        return [[torch.ones((1, 2)), {"tokens": tokens}]]


def test_ming_without_images_preserves_prompt_and_has_no_reference_metadata() -> None:
    clip = Clip()
    result = TextEncodeMingImageEdit.execute(clip=clip, prompt="describe", images={})
    assert result["CONDITIONING"] == [
        [torch.ones((1, 2)), {"tokens": {"text": "describe", "images": []}}]
    ]


def test_mage_without_images_uses_each_default_axis_and_null_negative_prompt() -> None:
    clip = Clip()
    result = TextEncodeMageFlowEdit.execute(
        clip=clip, prompt="edit", negative_prompt="", width=0, height=512, batch_size=2
    )
    assert result["latent"]["samples"].shape == (2, 128, 32, 64)
    assert clip.calls == [("edit", {"images": []}), (" ", {"images": []})]


def test_qwen_edit_encoders_preserve_no_image_token_contracts() -> None:
    clip = Clip()
    edit = TextEncodeQwenImageEdit.execute(clip=clip, prompt="edit")["CONDITIONING"]
    plus = TextEncodeQwenImageEditPlus.execute(clip=clip, prompt="plus")["CONDITIONING"]
    assert edit[0][1]["tokens"] == {"text": "edit", "images": []}
    assert plus[0][1]["tokens"]["text"] == "plus"
    assert plus[0][1]["tokens"]["images"] == []
    assert plus[0][1]["tokens"]["llama_template"] == TextEncodeQwenImageEditPlus.llama_template


def test_qwen_image21_empty_images_prevent_empty_text_and_create_default_latent() -> None:
    clip = Clip()
    result = TextEncodeQwenImage21.execute(clip=clip, prompt="", negative_prompt="", resolution=0)
    assert result["latent"]["samples"].shape == (1, 64, 64, 64)
    assert [call[1] for call in clip.calls] == [
        {"images": [], "keep_vision": True, "prevent_empty_text": True},
        {"images": [], "keep_vision": True, "prevent_empty_text": True},
    ]


def test_qwen_cache_clones_model_and_replaces_only_its_cache_option() -> None:
    class Model:
        def __init__(self) -> None:
            self.model_options = {"transformer_options": {"other": "keep"}}

        def clone(self) -> Any:
            clone = Model()
            clone.model_options = {
                "transformer_options": self.model_options["transformer_options"].copy()
            }
            return clone

    source = Model()
    patched = QwenImage21Cache.execute(model=source, device="cpu", dtype="int8")["MODEL"]
    assert patched is not source
    assert source.model_options == {"transformer_options": {"other": "keep"}}
    assert patched.model_options["transformer_options"] == {
        "other": "keep",
        "qwen_image21_cache": {"device": "cpu", "dtype": "int8"},
    }
