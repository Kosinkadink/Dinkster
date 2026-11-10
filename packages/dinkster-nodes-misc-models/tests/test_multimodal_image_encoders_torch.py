from __future__ import annotations

from typing import Any

import torch
from dinkster_misc_models.mage_image_encoders import TextEncodeMageFlowEdit
from dinkster_misc_models.ming_image_encoders import TextEncodeMingImageEdit


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
    conditioning: Any = result["CONDITIONING"]
    assert len(conditioning) == 1
    torch.testing.assert_close(conditioning[0][0], torch.ones((1, 2)))
    assert conditioning[0][1] == {"tokens": {"text": "describe", "images": []}}


def test_mage_without_images_uses_each_default_axis_and_null_negative_prompt() -> None:
    clip = Clip()
    result = TextEncodeMageFlowEdit.execute(
        clip=clip, prompt="edit", negative_prompt="", width=0, height=512, batch_size=2
    )
    assert result["latent"]["samples"].shape == (2, 128, 32, 64)
    assert clip.calls == [("edit", {"images": []}), (" ", {"images": []})]
