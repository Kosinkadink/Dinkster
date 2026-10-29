"""Self-contained CLIP byte-level BPE tokenizer - no transformers.

ComfyUI tokenizes through Hugging Face ``CLIPTokenizer`` loaded from
its vendored ``comfy/sd1_tokenizer/`` data (comfy/sd1_clip.py
SDTokenizer @ b78cec87). This module reimplements that exact pipeline
on the stdlib so the torch-free package tokenizes identically:

- Normalization is the tokenizer's non-ftfy path (transformers
  4.57.3 tokenization_clip.py: ``BasicTokenizer(strip_accents=False,
  do_split_on_punc=False)`` with lowercasing and CJK-character
  padding, then NFC). ftfy is NOT part of the pinned behavior: the
  audited reference environment does not install it, and pinning one
  deterministic normalization beats varying with an optional import.
- Word splitting replicates the CLIP regex
  ``<|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|
  [\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+`` (IGNORECASE) as an
  explicit scanner - the stdlib ``re`` has no ``\\p`` classes.
- Byte-level BPE runs over the vendored vocab/merges (gzipped copies
  of ComfyUI's sd1_tokenizer data; provenance hashes below), with the
  reference's ``</w>`` end-of-word marker and lowest-rank-first merge
  loop.

``encode`` returns the INNER token ids - what the reference slices
out of Hugging Face output as ``input_ids[1:-1]`` - so BOS/EOS
placement stays where it belongs, in the chunk packer
(clip_tokenize). Goldens pin this module against the executed
reference tokenizer, never a re-derivation
(tests/goldens/clip_tokenizer_goldens.json).
"""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache

from .vendored import read_vendored

#: Special-token ids from the vendored vocabulary.
CLIP_BOS = 49406
CLIP_EOS = 49407
CLIP_VOCAB_SIZE = 49408

#: sha256 of the UNCOMPRESSED vendored files, byte-identical to
#: ComfyUI's comfy/sd1_tokenizer/vocab.json and merges.txt @ b78cec87.
CLIP_VOCAB_SHA256 = "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"
CLIP_MERGES_SHA256 = "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"

_SOT = "<|startoftext|>"
_EOT = "<|endoftext|>"
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _bytes_to_unicode() -> dict[int, str]:
    """The GPT-2/CLIP reversible byte->unicode table (transformers
    tokenization_clip.py bytes_to_unicode)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs), strict=True))


def _is_control(char: str) -> bool:
    if char in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(char).startswith("C")


def _is_whitespace(char: str) -> bool:
    if char in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(char) == "Zs"


def _is_cjk(cp: int) -> bool:
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


def normalize_text(text: str) -> str:
    """The reference's non-ftfy normalization: drop control chars,
    collapse whitespace to spaces, pad CJK characters, NFC, lowercase
    per whitespace token (no accent stripping, no punctuation
    splitting), rejoin with single spaces."""
    cleaned: list[str] = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        cleaned.append(" " if _is_whitespace(char) else char)
    padded: list[str] = []
    for char in cleaned:
        if _is_cjk(ord(char)):
            padded.append(f" {char} ")
        else:
            padded.append(char)
    normalized = unicodedata.normalize("NFC", "".join(padded))
    return " ".join(token.lower() for token in normalized.split())


def _scan_words(text: str) -> list[str]:
    """Replicates the CLIP splitting regex on normalized text:
    special-token literals, contraction suffixes, letter runs, single
    digits, and runs of everything else, in that alternation order."""
    words: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char == " " or char.isspace():
            i += 1
            continue
        if text.startswith(_SOT, i):
            words.append(_SOT)
            i += len(_SOT)
            continue
        if text.startswith(_EOT, i):
            words.append(_EOT)
            i += len(_EOT)
            continue
        if char == "'":
            matched = next((c for c in _CONTRACTIONS if text.startswith(c, i)), None)
            if matched is not None:
                words.append(matched)
                i += len(matched)
                continue
        category = unicodedata.category(char)
        if category.startswith("L"):
            j = i + 1
            while j < n and unicodedata.category(text[j]).startswith("L"):
                j += 1
            words.append(text[i:j])
            i = j
            continue
        if category.startswith("N"):
            words.append(char)
            i += 1
            continue
        j = i + 1
        while j < n:
            other = text[j]
            if other == " " or other.isspace():
                break
            other_cat = unicodedata.category(other)
            if other_cat.startswith("L") or other_cat.startswith("N"):
                break
            j += 1
        words.append(text[i:j])
        i = j
    return words


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(word, word[1:], strict=False))


def _split_specials(text: str) -> list[str]:
    """Splits RAW text on exact special-token literals, before any
    normalization - Hugging Face extracts added tokens case-sensitively
    from the unnormalized input, so ``!!<|endoftext|>`` keeps the
    literal intact while the surrounding text is tokenized normally."""
    parts: list[str] = []
    rest = text
    while rest:
        hits = [(idx, lit) for lit in (_SOT, _EOT) if (idx := rest.find(lit)) != -1]
        if not hits:
            parts.append(rest)
            break
        idx, lit = min(hits)
        if idx:
            parts.append(rest[:idx])
        parts.append(lit)
        rest = rest[idx + len(lit) :]
    return parts


class ClipBpe:
    """The vendored-vocabulary CLIP BPE encoder. Construction parses
    ~1.5 MB of data; share one instance via :func:`load_clip_bpe`.

    The only mutable state is the merge cache (the reference keeps an
    identical unbounded per-token cache); results are pure functions
    of the input text.
    """

    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]) -> None:
        self._encoder = vocab
        self._ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._byte_encoder = _bytes_to_unicode()
        self._unk = vocab[_EOT]
        self._cache: dict[str, str] = {_SOT: _SOT, _EOT: _EOT}

    @classmethod
    def from_vendored_data(cls) -> ClipBpe:
        vocab: dict[str, int] = json.loads(read_vendored("clip_vocab.json.gz", CLIP_VOCAB_SHA256))
        merge_text = read_vendored("clip_merges.txt.gz", CLIP_MERGES_SHA256).decode("utf-8")
        # The reference slices lines [1 : 49152-256-2+1] - skip the
        # version header, keep exactly the merges the vocab covers.
        lines = merge_text.strip().split("\n")[1 : 49152 - 256 - 2 + 1]
        merges: list[tuple[str, str]] = []
        for line in lines:
            first, second = line.split()
            merges.append((first, second))
        return cls(vocab, merges)

    def _bpe(self, token: str) -> str:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self._ranks.get(pair, float("inf")))
            if bigram not in self._ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        merged = " ".join(word)
        self._cache[token] = merged
        return merged

    def encode(self, text: str) -> tuple[int, ...]:
        """Text -> inner token ids (no BOS/EOS). Identical to the
        reference's ``tokenizer(text)["input_ids"][1:-1]``."""
        ids: list[int] = []
        for part in _split_specials(text):
            if part in (_SOT, _EOT):
                ids.append(self._encoder[part])
                continue
            for word in _scan_words(normalize_text(part)):
                encoded = "".join(self._byte_encoder[b] for b in word.encode("utf-8"))
                for bpe_token in self._bpe(encoded).split(" "):
                    ids.append(self._encoder.get(bpe_token, self._unk))
        return tuple(ids)


@lru_cache(maxsize=1)
def load_clip_bpe() -> ClipBpe:
    """The shared vendored-data instance (parsing is done once)."""
    return ClipBpe.from_vendored_data()


__all__ = [
    "CLIP_BOS",
    "CLIP_EOS",
    "CLIP_MERGES_SHA256",
    "CLIP_VOCAB_SHA256",
    "CLIP_VOCAB_SIZE",
    "ClipBpe",
    "load_clip_bpe",
    "normalize_text",
]
