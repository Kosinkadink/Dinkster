from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_sd.clip_sd3 import CLIPTextEncodeSD3


class TripleEncoder:
    def tokenize(self, text: str) -> dict[str, list[list[int]]]:
        return {
            key: [[offset + ord(character)] for character in text] or [[offset]]
            for key, offset in (("g", 1000), ("l", 2000), ("t5xxl", 3000), ("extra", 4000))
        }

    def encode_from_tokens_scheduled(self, tokens: dict[str, list[list[int]]]) -> list[list[Any]]:
        token_streams = {key: value.copy() for key, value in tokens.items()}
        tensor = torch.tensor([[sum(value[0] for value in stream) for stream in tokens.values()]])
        return [
            [tensor, {"start_percent": 0.25, "token_streams": token_streams}],
            [tensor + 1, {"start_percent": 0.75, "token_streams": token_streams}],
        ]


def _tokens(
    *, clip_l: str, clip_g: str, t5xxl: str, empty_padding: str
) -> dict[str, list[list[int]]]:
    encoder = TripleEncoder()
    result = CLIPTextEncodeSD3.execute(
        clip=encoder,
        clip_l=clip_l,
        clip_g=clip_g,
        t5xxl=t5xxl,
        empty_padding=empty_padding,
    )["CONDITIONING"][0][1]["token_streams"]
    return result


@pytest.mark.parametrize(
    ("clip_l", "clip_g", "expected_g", "expected_l"),
    (
        ("a", "wxyz", [[1119], [1120], [1121], [1122]], [[2097], [2000], [2000], [2000]]),
        ("wxyz", "a", [[1097], [1000], [1000], [1000]], [[2119], [2120], [2121], [2122]]),
    ),
)
def test_sd3_aligns_asymmetric_clip_streams_in_both_directions(
    clip_l: str, clip_g: str, expected_g: list[list[int]], expected_l: list[list[int]]
) -> None:
    tokens = _tokens(clip_l=clip_l, clip_g=clip_g, t5xxl="t5", empty_padding="none")
    assert tokens["g"] == expected_g
    assert tokens["l"] == expected_l
    assert tokens["t5xxl"] == [[3116], [3053]]
    assert tokens["extra"] == ([[4119], [4120], [4121], [4122]] if clip_g == "wxyz" else [[4097]])


@pytest.mark.parametrize(
    ("empty_padding", "expected"),
    (
        ("none", {"g": [], "l": [], "t5xxl": [], "extra": [[4000]]}),
        ("empty_prompt", {"g": [[1000]], "l": [[2000]], "t5xxl": [[3000]], "extra": [[4000]]}),
    ),
)
def test_sd3_empty_padding_selects_empty_stream_or_empty_prompt(
    empty_padding: str, expected: dict[str, list[list[int]]]
) -> None:
    assert _tokens(clip_l="", clip_g="", t5xxl="", empty_padding=empty_padding) == expected
