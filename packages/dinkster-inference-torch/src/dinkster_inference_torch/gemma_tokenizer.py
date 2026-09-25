"""Gemma SentencePiece tokenizers for native text conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Protocol, cast

from dinkster_inference import MEBIBYTE

#: SentencePiece piece count. The transformer vocabulary is larger
#: (262208): the trailing rows back multimodal soft tokens such as
#: <image_soft_token> 262144, which the prompt policy maps without
#: the SentencePiece model knowing them.
GEMMA_SENTENCEPIECE_VOCAB_SIZE = 262144

LUMINA2_TOKENIZER_ATTRIBUTE = "_dinkster_lumina2_sentencepiece"
LUMINA2_TOKENIZER_BYTE_CAP = 8 * MEBIBYTE


class _SentencePieceProcessor(Protocol):
    def get_piece_size(self) -> int: ...

    def pad_id(self) -> int: ...

    def eos_id(self) -> int: ...

    def bos_id(self) -> int: ...

    def unk_id(self) -> int: ...

    def encode(
        self, text: str, *, out_type: type[int], add_bos: bool, add_eos: bool
    ) -> list[int]: ...


class _JsonEncoding(Protocol):
    ids: list[int]


class _JsonTokenizer(Protocol):
    def get_vocab_size(self, with_added_tokens: bool = True) -> int: ...

    def token_to_id(self, token: str) -> int | None: ...

    def encode(self, text: str, add_special_tokens: bool = True) -> _JsonEncoding: ...


class GemmaTokenizerError(ValueError):
    """The checkpoint tokenizer payload is not the official Gemma vocabulary."""


class GemmaSentencePieceTokenizer:
    """Strict adapter over an embedded Gemma SentencePiece model."""

    def __init__(
        self,
        model: bytes,
        *,
        expected_vocab_size: int = GEMMA_SENTENCEPIECE_VOCAB_SIZE,
    ) -> None:
        if type(model) is not bytes or not model:
            raise GemmaTokenizerError("Gemma tokenizer model must be nonempty bytes")
        if type(expected_vocab_size) is not int or expected_vocab_size < 1:
            raise GemmaTokenizerError("Gemma tokenizer vocabulary size must be a positive integer")
        try:
            processor = cast(
                "_SentencePieceProcessor",
                importlib.import_module("sentencepiece").SentencePieceProcessor(
                    model_proto=model, add_bos=False, add_eos=False
                ),
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise GemmaTokenizerError(f"invalid Gemma SentencePiece model: {error}") from None
        facts = (
            processor.get_piece_size(),
            processor.pad_id(),
            processor.eos_id(),
            processor.bos_id(),
            processor.unk_id(),
        )
        if facts != (expected_vocab_size, 0, 1, 2, 3):
            raise GemmaTokenizerError(
                f"Gemma SentencePiece model must have {expected_vocab_size} pieces and "
                "special ids pad=0, eos=1, bos=2, unk=3"
            )
        self._processor = processor

    def encode(self, text: str) -> Sequence[int]:
        if type(text) is not str:
            raise TypeError("Gemma tokenizer input must be a string")
        return self._processor.encode(text, out_type=int, add_bos=False, add_eos=False)


class GemmaJsonTokenizer:
    """Strict adapter over the tokenizer JSON embedded in Gemma 4 assets."""

    def __init__(self, model: bytes) -> None:
        if type(model) is not bytes or not model:
            raise GemmaTokenizerError("Gemma tokenizer JSON must be nonempty bytes")
        try:
            encoded = model.decode("utf-8")
            processor = cast(
                "_JsonTokenizer", importlib.import_module("tokenizers").Tokenizer.from_str(encoded)
            )
        except (AttributeError, UnicodeDecodeError, ValueError) as error:
            raise GemmaTokenizerError(f"invalid Gemma tokenizer JSON: {error}") from None
        facts = (
            processor.get_vocab_size(with_added_tokens=True),
            processor.token_to_id("<pad>"),
            processor.token_to_id("<eos>"),
            processor.token_to_id("<bos>"),
            processor.token_to_id("<unk>"),
        )
        if facts != (GEMMA_SENTENCEPIECE_VOCAB_SIZE, 0, 1, 2, 3):
            raise GemmaTokenizerError(
                "Gemma tokenizer JSON must have 262144 tokens and special ids "
                "pad=0, eos=1, bos=2, unk=3"
            )
        self._processor = processor

    def encode(self, text: str) -> Sequence[int]:
        if type(text) is not str:
            raise TypeError("Gemma tokenizer input must be a string")
        return self._processor.encode(text, add_special_tokens=False).ids


__all__ = [
    "GEMMA_SENTENCEPIECE_VOCAB_SIZE",
    "LUMINA2_TOKENIZER_ATTRIBUTE",
    "LUMINA2_TOKENIZER_BYTE_CAP",
    "GemmaJsonTokenizer",
    "GemmaSentencePieceTokenizer",
    "GemmaTokenizerError",
]
