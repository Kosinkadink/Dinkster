from __future__ import annotations

from typing import Any

import torch
from dinkster_misc_models.clip_model_specific import CLIPTextEncodeKandinsky5


class KandinskyClip:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def tokenize(self, text: str) -> dict[str, object]:
        self.calls.append(text)
        return {
            "clip_l": ["clip-l", text],
            "qwen25_7b": ["qwen", text],
            "other": {"prompt": text},
        }

    def encode_from_tokens_scheduled(self, tokens: dict[str, object]) -> list[list[Any]]:
        return [
            [
                torch.tensor([[1.0, 2.0]], dtype=torch.float64),
                {"tokens": tokens, "source": "first"},
            ],
            [torch.tensor([[3.0]], dtype=torch.float32), {"tokens": tokens, "source": "second"}],
        ]


def test_kandinsky_text_overwrites_only_qwen_stream() -> None:
    clip = KandinskyClip()
    result = CLIPTextEncodeKandinsky5.execute(
        clip=clip, clip_l="wide river", qwen25_7b="night city"
    )["CONDITIONING"]
    assert clip.calls == ["wide river", "night city"]
    for tensor, metadata in result:
        assert metadata["tokens"] == {
            "clip_l": ["clip-l", "wide river"],
            "qwen25_7b": ["qwen", "night city"],
            "other": {"prompt": "wide river"},
        }
        assert metadata["source"] in {"first", "second"}
        assert tensor.dtype in {torch.float64, torch.float32}
    assert torch.equal(result[0][0], torch.tensor([[1.0, 2.0]], dtype=torch.float64))
    assert torch.equal(result[1][0], torch.tensor([[3.0]], dtype=torch.float32))
