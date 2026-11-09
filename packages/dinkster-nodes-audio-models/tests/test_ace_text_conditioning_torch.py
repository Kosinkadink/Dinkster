from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_audio_models.ace_text_conditioning import (
    TextEncodeAceStepAudio,
    TextEncodeAceStepAudio15,
)


class AsymmetricClip:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.last_conditioning: list[list[Any]] = []

    def tokenize(self, tags: str, **options: object) -> dict[str, object]:
        self.calls.append((tags, options))
        return {"tags": tags, **options}

    def encode_from_tokens_scheduled(self, tokens: dict[str, object]) -> list[list[Any]]:
        self.last_conditioning = [
            [torch.tensor([[1.0, 3.0]], dtype=torch.float64), {"source": "first", **tokens}],
            [torch.tensor([[2.0]], dtype=torch.float32), {"source": "second", **tokens}],
        ]
        return self.last_conditioning


def test_ace_text_encoders_delegate_exact_options_and_copy_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dinkster_audio_models.ace_text_conditioning.initialize_inference_process",
        lambda: None,
    )
    ace = AsymmetricClip()
    result = TextEncodeAceStepAudio.execute(
        clip=ace, tags="soft drums", lyrics="first\nsecond", lyrics_strength=3.25
    )["CONDITIONING"]
    assert ace.calls == [("soft drums", {"lyrics": "first\nsecond"})]
    assert [entry[1]["lyrics_strength"] for entry in result] == [3.25, 3.25]
    assert [entry[1]["source"] for entry in result] == ["first", "second"]
    assert all(
        entry[1] is not original[1]
        for entry, original in zip(result, ace.last_conditioning, strict=True)
    )
    assert all(
        entry[0] is original[0]
        for entry, original in zip(result, ace.last_conditioning, strict=True)
    )

    ace15 = AsymmetricClip()
    result15 = TextEncodeAceStepAudio15.execute(
        clip=ace15,
        tags="bright bass",
        lyrics="la",
        seed=0xFFFFFFFFFFFFFFFF,
        bpm=171,
        duration=93.7,
        timesignature="6",
        language="yue",
        keyscale="Bb minor",
        generate_audio_codes=False,
        cfg_scale=7.4,
        temperature=1.23,
        top_p=0.43,
        top_k=19,
        min_p=0.017,
    )["CONDITIONING"]
    assert ace15.calls == [
        (
            "bright bass",
            {
                "lyrics": "la",
                "bpm": 171,
                "duration": 93.7,
                "timesignature": 6,
                "language": "yue",
                "keyscale": "Bb minor",
                "seed": 0xFFFFFFFFFFFFFFFF,
                "generate_audio_codes": False,
                "cfg_scale": 7.4,
                "temperature": 1.23,
                "top_p": 0.43,
                "top_k": 19,
                "min_p": 0.017,
            },
        )
    ]
    assert [entry[1]["source"] for entry in result15] == ["first", "second"]
    assert result15[0][0].dtype is torch.float64 and result15[1][0].dtype is torch.float32
    assert all(
        entry[0] is original[0]
        for entry, original in zip(result15, ace15.last_conditioning, strict=True)
    )
