"""CPU YuE2 ABC and music conditioning behavior."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import torch
from dinkster_nodes_comfy.yue2_conditioning import YuE2GenerateABC, YuE2GenerateMusic


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
        "dinkster_nodes_comfy.yue2_conditioning.initialize_inference_process", lambda: None
    )
    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference.text_encoders.yue2",
        types.SimpleNamespace(FRAMES_PER_SECOND=2.0),
    )


def test_yue2_abc_schema_and_generation_parameters(yue2: None) -> None:
    schema = YuE2GenerateABC.define_schema()
    assert schema.node_type == "YuE2GenerateABC"
    assert schema.category == "model/conditioning/yue2"
    assert schema.search_terms == ("yue2", "music", "abc", "notation")
    assert [input.id for input in schema.inputs] == [
        "clip",
        "style",
        "lyrics",
        "seed",
        "mode",
        "max_abc_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "penalty_window",
    ]
    assert [input.default for input in schema.inputs[5:]] == [8192, 0.7, 0.9, 30, 1.005, 100]
    assert schema.inputs[10].doc == "Number of recent ABC tokens used to penalize repetition."
    assert schema.outputs[0].id == "abc"

    clip = YuE2Clip()
    result = YuE2GenerateABC.execute(
        clip=clip,
        style="orchestral",
        lyrics="one\ntwo",
        seed=0xFFFFFFFFFFFFFFFF,
        mode="melody",
        max_abc_tokens=17,
        temperature=1.35,
        top_p=0.41,
        top_k=7,
        repetition_penalty=1.7,
        penalty_window=9,
    )
    assert clip.tokenize_calls == [
        (
            "orchestral",
            {
                "lyrics": "one\ntwo",
                "cot": "melody",
                "seed": 0xFFFFFFFFFFFFFFFF,
                "max_tokens": 17,
                "penalty_window": 9,
            },
        )
    ]
    assert clip.generate_calls == [
        (
            {"style": "orchestral", **clip.tokenize_calls[0][1]},
            {
                "max_length": 17,
                "temperature": 1.35,
                "top_p": 0.41,
                "top_k": 7,
                "repetition_penalty": 1.7,
                "seed": 0xFFFFFFFFFFFFFFFF,
            },
        )
    ]
    assert clip.decode_calls == [("orchestral", 0xFFFFFFFFFFFFFFFF)]
    assert result == {"abc": "ABC:('orchestral', 18446744073709551615)"}


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


def test_yue2_music_schema_preserves_reference_order_defaults_and_optional_cfg() -> None:
    schema = YuE2GenerateMusic.define_schema()
    assert schema.node_type == "YuE2GenerateMusic"
    assert schema.category == "model/conditioning/yue2"
    assert schema.search_terms == ("yue2", "music", "audio", "abc")
    assert [input.id for input in schema.inputs] == [
        "clip",
        "style",
        "lyrics",
        "abc",
        "seed",
        "mode",
        "max_duration",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "cfg_scale",
    ]
    assert [input.default for input in schema.inputs[3:]] == [
        "",
        0,
        None,
        360.0,
        1.0,
        0.95,
        100,
        1.2,
        None,
    ]
    assert schema.inputs[11].required is False
    assert [output.id for output in schema.outputs] == ["CONDITIONING", "seconds"]
