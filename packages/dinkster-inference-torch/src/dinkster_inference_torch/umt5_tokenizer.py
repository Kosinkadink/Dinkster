"""UMT5 SentencePiece tokenizer used by Wan text conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Protocol, cast

UMT5_SENTENCEPIECE_VOCAB_SIZE = 256000


class _SentencePieceProcessor(Protocol):
    def get_piece_size(self) -> int: ...

    def pad_id(self) -> int: ...

    def eos_id(self) -> int: ...

    def bos_id(self) -> int: ...

    def unk_id(self) -> int: ...

    def encode(
        self, text: str, *, out_type: type[int], add_bos: bool, add_eos: bool
    ) -> list[int]: ...


class Umt5TokenizerError(ValueError):
    """The checkpoint tokenizer payload is not the official UMT5 vocabulary."""


class Umt5SentencePieceTokenizer:
    """Strict adapter over the SentencePiece model embedded in Wan checkpoints."""

    def __init__(self, model: bytes) -> None:
        if type(model) is not bytes or not model:
            raise Umt5TokenizerError("UMT5 tokenizer model must be nonempty bytes")
        try:
            processor = cast(
                "_SentencePieceProcessor",
                importlib.import_module("sentencepiece").SentencePieceProcessor(
                    model_proto=model, add_bos=False, add_eos=False
                ),
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise Umt5TokenizerError(f"invalid UMT5 SentencePiece model: {error}") from None
        facts = (
            processor.get_piece_size(),
            processor.pad_id(),
            processor.eos_id(),
            processor.bos_id(),
            processor.unk_id(),
        )
        if facts != (UMT5_SENTENCEPIECE_VOCAB_SIZE, 0, 1, 2, 3):
            raise Umt5TokenizerError(
                "UMT5 SentencePiece model must have 256000 pieces and special ids "
                "pad=0, eos=1, bos=2, unk=3"
            )
        self._processor = processor

    def encode(self, text: str) -> Sequence[int]:
        if type(text) is not str:
            raise TypeError("UMT5 tokenizer input must be a string")
        return self._processor.encode(text, out_type=int, add_bos=False, add_eos=False)


__all__ = [
    "UMT5_SENTENCEPIECE_VOCAB_SIZE",
    "Umt5SentencePieceTokenizer",
    "Umt5TokenizerError",
]
