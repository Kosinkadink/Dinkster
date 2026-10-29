"""Lumina Image 2.0 Gemma 2 prompt policy."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .gemma_text import GEMMA2_LUMINA_2B_CONFIG
from .prompt_tokens import Chunk, TokenizerProfile, pack_spans, tokenize_prompt

LUMINA2_SYSTEM_PROMPTS = {
    "superior": (
        "You are an assistant designed to generate superior images with the superior "
        "degree of image-text alignment based on textual prompts or user prompts."
    ),
    "alignment": (
        "You are an assistant designed to generate high-quality images with the highest "
        "degree of image-text alignment based on textual prompts."
    ),
}

_LUMINA2_PROFILE = TokenizerProfile(
    max_length=99_999_999,
    start_token=GEMMA2_LUMINA_2B_CONFIG.bos_token_id,
    end_token=None,
    pad_token=GEMMA2_LUMINA_2B_CONFIG.pad_token_id,
    pad_to_max_length=False,
    min_length=1,
)
_END_OF_TURN = "<end_of_turn>"
_END_OF_TURN_SPLIT = re.compile(f"({re.escape(_END_OF_TURN)})")


@dataclass(frozen=True)
class Lumina2PromptTokens:
    ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.ids or len(self.ids) != len(self.attention_mask):
            raise ValueError("Lumina2 ids and attention mask must be non-empty and equal")
        if len(self.ids) != len(self.weights):
            raise ValueError("Lumina2 ids and weights must be equal length")
        if self.ids[0] != GEMMA2_LUMINA_2B_CONFIG.bos_token_id:
            raise ValueError("Lumina2 rows must begin with BOS")
        if any(value not in (0, 1) for value in self.attention_mask):
            raise ValueError("Lumina2 attention mask values must be binary integers")


def lumina2_system_prompt(user_prompt: str, system_prompt: str) -> str:
    """Compose the exact CLIPTextEncodeLumina2 prompt string."""
    if type(user_prompt) is not str:
        raise TypeError("user_prompt must be a string")
    try:
        system = LUMINA2_SYSTEM_PROMPTS[system_prompt]
    except KeyError:
        raise ValueError(f"unknown Lumina2 system prompt {system_prompt!r}") from None
    return f"{system} <Prompt Start> {user_prompt}"


def _chunk_values(chunk: Chunk) -> tuple[tuple[int, ...], tuple[float, ...]]:
    ids: list[int] = []
    weights: list[float] = []
    for token in chunk:
        if not isinstance(token.unit, int):
            raise ValueError("Lumina2 prompts resolve no embedding directives")
        ids.append(token.unit)
        weights.append(token.weight)
    return tuple(ids), tuple(weights)


def tokenize_lumina2_prompt(
    text: str,
    *,
    encode: Callable[[str], Sequence[int]],
) -> Lumina2PromptTokens:
    """Apply ComfyUI's unbounded weighted Gemma 2 tokenization."""

    def encode_word(word: str) -> Sequence[int]:
        if _END_OF_TURN not in word:
            return encode(word)
        ids: list[int] = []
        for part in _END_OF_TURN_SPLIT.split(word):
            if not part:
                continue
            if part == _END_OF_TURN:
                ids.append(GEMMA2_LUMINA_2B_CONFIG.end_of_turn_token_id)
            else:
                ids.extend(encode(part))
        # ComfyUI slices one token after special-token extraction, where
        # SentencePiece did not add BOS.
        return ids[1:]

    spans = tokenize_prompt(text, encode_word=encode_word).spans
    (chunk,) = pack_spans(spans, _LUMINA2_PROFILE)
    ids, weights = _chunk_values(chunk)
    return Lumina2PromptTokens(ids, (1,) * len(ids), weights)


__all__ = [
    "LUMINA2_SYSTEM_PROMPTS",
    "Lumina2PromptTokens",
    "lumina2_system_prompt",
    "tokenize_lumina2_prompt",
]
