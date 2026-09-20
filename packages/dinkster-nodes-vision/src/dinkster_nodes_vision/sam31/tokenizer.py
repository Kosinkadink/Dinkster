"""Offline CLIP byte-pair tokenizer for SAM 3.1 text prompts."""

from __future__ import annotations

import gzip
import hashlib
import unicodedata
from functools import lru_cache
from importlib import resources

CLIP_BOS = 49406
CLIP_EOS = 49407
CLIP_VOCAB_SIZE = 49408
TOKEN_COUNT = 32
TOKENIZER_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
TOKENIZER_TEXT_SHA256 = "67603cfda2e032ad77b5f8808af37789d590db664b26df8705d2bf8b3c553fc8"

_SOT = "<|startoftext|>"
_EOT = "<|endoftext|>"
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _tokenizer_data() -> bytes:
    compressed = (
        resources.files("dinkster_nodes_vision.sam31")
        .joinpath("data/bpe_simple_vocab_16e6.txt.gz")
        .read_bytes()
    )
    digest = hashlib.sha256(compressed).hexdigest()
    if digest != TOKENIZER_SHA256:
        raise ValueError(f"SAM 3.1 tokenizer hash mismatch: {digest}")
    data = gzip.decompress(compressed)
    digest = hashlib.sha256(data).hexdigest()
    if digest != TOKENIZER_TEXT_SHA256:
        raise ValueError(f"SAM 3.1 tokenizer content hash mismatch: {digest}")
    return data


def _bytes_to_unicode() -> dict[int, str]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    code_points = byte_values[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            code_points.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, (chr(value) for value in code_points), strict=True))


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(part.lower() for part in normalized.split())


def _scan_words(text: str) -> list[str]:
    words: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if text.startswith(_SOT, index):
            words.append(_SOT)
            index += len(_SOT)
            continue
        if text.startswith(_EOT, index):
            words.append(_EOT)
            index += len(_EOT)
            continue
        if char == "'":
            contraction = next(
                (value for value in _CONTRACTIONS if text.startswith(value, index)),
                None,
            )
            if contraction is not None:
                words.append(contraction)
                index += len(contraction)
                continue
        category = unicodedata.category(char)
        if category.startswith("L"):
            end = index + 1
            while end < len(text) and unicodedata.category(text[end]).startswith("L"):
                end += 1
        elif category.startswith("N"):
            end = index + 1
        else:
            end = index + 1
            while end < len(text):
                other = text[end]
                other_category = unicodedata.category(other)
                if other.isspace() or other_category.startswith(("L", "N")):
                    break
                end += 1
        words.append(text[index:end])
        index = end
    return words


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(word, word[1:], strict=False))


class ClipBpe:
    def __init__(self) -> None:
        merge_lines = _tokenizer_data().decode("utf-8").split("\n")
        merges = [tuple(line.split()) for line in merge_lines[1 : 49152 - 256 - 2 + 1]]
        byte_encoder = _bytes_to_unicode()
        vocabulary = list(byte_encoder.values())
        vocabulary.extend(value + "</w>" for value in byte_encoder.values())
        vocabulary.extend("".join(pair) for pair in merges)
        vocabulary.extend((_SOT, _EOT))
        self._encoder = dict(zip(vocabulary, range(len(vocabulary)), strict=True))
        self._ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._byte_encoder = byte_encoder
        self._cache: dict[str, str] = {_SOT: _SOT, _EOT: _EOT}
        if len(self._encoder) != CLIP_VOCAB_SIZE:
            raise ValueError(f"SAM 3.1 tokenizer has {len(self._encoder)} tokens")

    def _bpe(self, token: str) -> str:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = (*token[:-1], token[-1] + "</w>")
        pairs = _pairs(word)
        while pairs:
            bigram = min(pairs, key=lambda pair: self._ranks.get(pair, float("inf")))
            if bigram not in self._ranks:
                break
            first, second = bigram
            merged: list[str] = []
            index = 0
            while index < len(word):
                try:
                    found = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:found])
                index = found
                if index < len(word) - 1 and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            pairs = _pairs(word)
        result = " ".join(word)
        self._cache[token] = result
        return result

    def encode(self, text: str) -> tuple[int, ...]:
        tokens: list[int] = []
        for word in _scan_words(_normalize(text)):
            encoded = "".join(self._byte_encoder[value] for value in word.encode("utf-8"))
            tokens.extend(self._encoder[piece] for piece in self._bpe(encoded).split(" "))
        return tuple(tokens)

    def batches(self, text: str) -> tuple[tuple[int, ...], ...]:
        tokens = self.encode(text)
        sections = [tokens[index : index + TOKEN_COUNT - 2] for index in range(0, len(tokens), 30)]
        if not sections:
            sections = [()]
        return tuple(
            (CLIP_BOS, *section, CLIP_EOS, *((0,) * (TOKEN_COUNT - len(section) - 2)))
            for section in sections
        )


@lru_cache(maxsize=1)
def load_tokenizer() -> ClipBpe:
    return ClipBpe()


__all__ = [
    "CLIP_BOS",
    "CLIP_EOS",
    "CLIP_VOCAB_SIZE",
    "TOKENIZER_SHA256",
    "TOKENIZER_TEXT_SHA256",
    "ClipBpe",
    "load_tokenizer",
]
