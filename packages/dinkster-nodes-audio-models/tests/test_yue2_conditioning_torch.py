from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import torch
from dinkster_audio_models.yue2_conditioning import YuE2GenerateMusic


class YuE2Clip:
    def __init__(self) -> None:
        self.tokenize_calls: list[tuple[str, dict[str, object]]] = []
        self.generate_calls: list[tuple[dict[str, object], dict[str, object]]] = []
        self.decode_calls: list[object] = []

    def tokenize(self, style: str, **options: object) -> dict[str, object]:
        self.tokenize_calls.append((style, options))
        return {"style": style, **options}

    def generate(self, tokens: dict[str, object], **options: object) -> tuple[str, int]:
        self.generate_calls.append((tokens, options))
        return (tokens["style"], options["seed"])

    def decode(self, generated: object) -> str:
        self.decode_calls.append(generated)
        return f"ABC:{generated!r}"

    def encode_from_tokens_scheduled(self, tokens: dict[str, object]) -> list[list[Any]]:
        return [
            [torch.tensor([[1.0, 3.0]], dtype=torch.float64), {"yue2_frames": 13, **tokens}],
            [torch.tensor([[2.0]], dtype=torch.float32), {"yue2_frames": 29, **tokens}],
        ]


@pytest.fixture
def yue2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dinkster_audio_models.yue2_conditioning.initialize_inference_process", lambda: None
    )
    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference.text_encoders.yue2",
        types.SimpleNamespace(FRAMES_PER_SECOND=2.0),
    )


@pytest.mark.parametrize(
    ("abc", "mode", "cfg_scale", "duration", "expected_mode", "expected_cfg", "expected_tokens"),
    (
        ("X:1", "melody", None, 2.25, "melody", 1.0, 4),
        ("  \n\t", "melody", None, 2.25, "off", 1.01, 4),
        ("", "full", 0.0, -1.0, "off", 0.0, 1),
        ("X:1", "full", None, 2.75, "full", 1.0, 6),
    ),
)
def test_yue2_music_preserves_observable_conditioning_and_duration(
    yue2: None,
    abc: str,
    mode: str,
    cfg_scale: float | None,
    duration: float,
    expected_mode: str,
    expected_cfg: float,
    expected_tokens: int,
) -> None:
    clip = YuE2Clip()
    result = YuE2GenerateMusic.execute(
        clip=clip,
        style="electric piano",
        lyrics="verse",
        abc=abc,
        seed=31,
        mode=mode,
        max_duration=duration,
        temperature=1.4,
        top_p=0.85,
        top_k=51,
        repetition_penalty=1.25,
        cfg_scale=cfg_scale,
    )
    expected = {
        "lyrics": "verse",
        "cot": expected_mode,
        "seed": 31,
        "abc": abc,
        "max_tokens": expected_tokens,
        "temperature": 1.4,
        "top_p": 0.85,
        "top_k": 51,
        "repetition_penalty": 1.25,
        "cfg_scale": expected_cfg,
    }
    assert clip.tokenize_calls == [("electric piano", expected)]
    assert result["seconds"] == 6.5
    conditioning = result["CONDITIONING"]
    assert [entry[1]["yue2_frames"] for entry in conditioning] == [13, 29]
    assert [entry[1]["max_tokens"] for entry in conditioning] == [expected_tokens, expected_tokens]
    assert conditioning[0][0].dtype is torch.float64
    assert conditioning[1][0].dtype is torch.float32
