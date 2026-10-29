"""Self-contained Qwen2 byte-level BPE and Ovis prompt policy.

ComfyUI's Ovis route uses ``transformers.Qwen2Tokenizer`` over the
vendored Qwen 2.5 vocabulary (``comfy/text_encoders/qwen25_tokenizer``
@ b78cec87). This is the same tokenizer without a transformers runtime:
NFC normalization, leftmost-longest added-token extraction, Qwen's
Unicode pre-tokenization expression, GPT-2 byte mapping, and rank-ordered
BPE over SHA-pinned compressed package data.

The Ovis wrapper is intentionally a raw-text API. ComfyUI calls
``tokenize_with_weights(..., disable_weights=True)`` after applying its
fixed template, so parentheses and ``:weight`` syntax remain literal
text. The resulting sequence is right-padded to 284 with token 151643,
and an attention mask accompanies it. ``slice_start`` reproduces
``OvisTEModel.encode_token_weights``: find the first ``background`` token
(4004), include the following colon (25), and keep output from that colon.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

from .ideogram4_text import format_ideogram4_prompt
from .krea2_text import format_krea2_prompt
from .qwen_text import KLEIN_QWEN3_8B_CONFIG, OVIS_QWEN3_2B_CONFIG, Z_IMAGE_QWEN3_4B_CONFIG
from .vendored import read_vendored

QWEN_PAD = 151643
QWEN_IM_START = 151644
QWEN_IM_END = 151645
QWEN_VOCAB_SHA256 = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
QWEN_MERGES_SHA256 = "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"

_ADDED_TOKENS = {
    "<|endoftext|>": 151643,
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|object_ref_start|>": 151646,
    "<|object_ref_end|>": 151647,
    "<|box_start|>": 151648,
    "<|box_end|>": 151649,
    "<|quad_start|>": 151650,
    "<|quad_end|>": 151651,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|vision_pad|>": 151654,
    "<|image_pad|>": 151655,
    "<|video_pad|>": 151656,
    "<tool_call>": 151657,
    "</tool_call>": 151658,
    "<|fim_prefix|>": 151659,
    "<|fim_middle|>": 151660,
    "<|fim_suffix|>": 151661,
    "<|fim_pad|>": 151662,
    "<|repo_name|>": 151663,
    "<|file_sep|>": 151664,
    "<tool_response>": 151665,
    "</tool_response>": 151666,
    "<think>": 151667,
    "</think>": 151668,
}
_ADDED_BY_LENGTH = tuple(sorted(_ADDED_TOKENS, key=len, reverse=True))
_ADDED_TOKEN_TEXT = {token_id: token for token, token_id in _ADDED_TOKENS.items()}
_CONTRACTIONS = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")


def _bytes_to_unicode() -> dict[int, str]:
    values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    codepoints = values[:]
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, codepoints), strict=True))


def _letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


def _number(char: str) -> bool:
    return unicodedata.category(char).startswith("N")


def _symbol(char: str) -> bool:
    return not char.isspace() and not _letter(char) and not _number(char)


def _qwen_pretokens(text: str, *, digit_run: int = 1) -> list[str]:
    r"""Scan Qwen2Tokenizer's PRETOKENIZE_REGEX without ``regex``.

    Alternation order and the unusual trailing-whitespace lookahead are
    preserved. Unicode category L/N stand in for ``\p{L}``/``\p{N}``;
    Python's ``isspace`` is the stdlib equivalent used for ``\s`` here.

    ``digit_run`` is the maximum digits per number piece: 1 for Qwen's
    ``\p{N}``, 3 for the Llama-style ``\p{N}{1,3}`` shared by the
    Mistral tekken route (tekken_bpe.py).
    """
    out: list[str] = []
    index = 0
    size = len(text)
    while index < size:
        lowered = text[index : index + 4].lower()
        contraction = next(
            (value for value in _CONTRACTIONS if lowered.startswith(value)),
            None,
        )
        if contraction is not None:
            out.append(text[index : index + len(contraction)])
            index += len(contraction)
            continue

        start = index
        if _letter(text[index]):
            index += 1
            while index < size and _letter(text[index]):
                index += 1
            out.append(text[start:index])
            continue
        if (
            text[index] not in "\r\n"
            and not _letter(text[index])
            and not _number(text[index])
            and index + 1 < size
            and _letter(text[index + 1])
        ):
            index += 2
            while index < size and _letter(text[index]):
                index += 1
            out.append(text[start:index])
            continue
        if _number(text[index]):
            end = index + 1
            while end < size and end - index < digit_run and _number(text[end]):
                end += 1
            out.append(text[index:end])
            index = end
            continue

        symbol_start = index
        if text[index] == " " and index + 1 < size and _symbol(text[index + 1]):
            index += 1
        if index < size and _symbol(text[index]):
            index += 1
            while index < size and _symbol(text[index]):
                index += 1
            while index < size and text[index] in "\r\n":
                index += 1
            out.append(text[symbol_start:index])
            continue
        index = start

        if text[index].isspace():
            whitespace_end = index
            last_newline = -1
            while whitespace_end < size and text[whitespace_end].isspace():
                if text[whitespace_end] in "\r\n":
                    last_newline = whitespace_end
                whitespace_end += 1
            if last_newline >= index:
                out.append(text[index : last_newline + 1])
                index = last_newline + 1
                continue
            if whitespace_end == size:
                out.append(text[index:whitespace_end])
                index = whitespace_end
                continue
            if whitespace_end - index > 1:
                out.append(text[index : whitespace_end - 1])
                index = whitespace_end - 1
                continue
            out.append(text[index:whitespace_end])
            index = whitespace_end
            continue

        # The regex alternatives are exhaustive. Keep an explicit
        # single-codepoint fallback so a future Unicode category cannot
        # turn a malformed prompt into an infinite loop.
        out.append(text[index])
        index += 1
    return out


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(word, word[1:], strict=False))


def merge_word(word: tuple[str, ...], ranks: Mapping[tuple[str, str], int]) -> tuple[str, ...]:
    """Run the byte-level BPE merge loop over one pre-tokenized word."""
    pairs = _pairs(word)
    if not pairs:
        return word
    while True:
        pair = min(pairs, key=lambda item: ranks.get(item, 1 << 60))
        if pair not in ranks:
            break
        first, second = pair
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
            if index + 1 < len(word) and word[index + 1] == second:
                merged.append(first + second)
                index += 2
            else:
                merged.append(first)
                index += 1
        word = tuple(merged)
        if len(word) == 1:
            break
        pairs = _pairs(word)
    return word


class QwenBpe:
    """Qwen2Tokenizer's vocabulary and byte-level BPE model."""

    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]) -> None:
        self._vocab = vocab
        self._tokens = {token_id: token for token, token_id in vocab.items()}
        self._ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._bytes = _bytes_to_unicode()
        self._unicode_bytes = {char: value for value, char in self._bytes.items()}
        self._cache: dict[str, tuple[str, ...]] = {}

    @classmethod
    def from_vendored_data(cls) -> QwenBpe:
        vocab: dict[str, int] = json.loads(read_vendored("qwen_vocab.json.gz", QWEN_VOCAB_SHA256))
        lines = read_vendored("qwen_merges.txt.gz", QWEN_MERGES_SHA256).decode("utf-8").splitlines()
        merges: list[tuple[str, str]] = []
        for index, line in enumerate(lines):
            line = line.strip()
            if not line or (index == 0 and line.startswith("#version:")):
                continue
            first, second = line.split()
            merges.append((first, second))
        return cls(vocab, merges)

    def _bpe(self, token: str) -> tuple[str, ...]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = merge_word(tuple(token), self._ranks)
        self._cache[token] = word
        return word

    def _plain(self, text: str) -> list[int]:
        normalized = unicodedata.normalize("NFC", text)
        ids: list[int] = []
        for pretoken in _qwen_pretokens(normalized):
            encoded = "".join(self._bytes[value] for value in pretoken.encode("utf-8"))
            for token in self._bpe(encoded):
                token_id = self._vocab.get(token)
                if token_id is None:
                    raise ValueError(f"Qwen vocabulary has no BPE token {token!r}")
                ids.append(token_id)
        return ids

    def encode(self, text: str) -> list[int]:
        """Tokenize raw text, extracting Qwen's 26 added tokens first."""
        ids: list[int] = []
        plain_start = 0
        index = 0
        while index < len(text):
            added = next(
                (token for token in _ADDED_BY_LENGTH if text.startswith(token, index)),
                None,
            )
            if added is None:
                index += 1
                continue
            if plain_start < index:
                ids.extend(self._plain(text[plain_start:index]))
            ids.append(_ADDED_TOKENS[added])
            index += len(added)
            plain_start = index
        if plain_start < len(text):
            ids.extend(self._plain(text[plain_start:]))
        return ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "replace",
    ) -> str:
        """Decode Qwen token IDs through the inverse byte-level vocabulary."""
        if errors not in ("strict", "ignore", "replace"):
            raise ValueError("decode errors must be 'strict', 'ignore', or 'replace'")
        return self.decode_bytes(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        ).decode("utf-8", errors=errors)

    def decode_bytes(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> bytes:
        """Decode token IDs without splitting incomplete UTF-8 sequences."""
        if type(skip_special_tokens) is not bool:
            raise TypeError("skip_special_tokens must be an exact boolean")
        output = bytearray()
        for token_id in token_ids:
            if type(token_id) is not int or token_id < 0:
                raise ValueError("Qwen token IDs must be non-negative exact integers")
            special = _ADDED_TOKEN_TEXT.get(token_id)
            if special is not None:
                if not skip_special_tokens:
                    output.extend(special.encode())
                continue
            token = self._tokens.get(token_id)
            if token is None:
                raise ValueError(f"Qwen vocabulary has no token ID {token_id}")
            try:
                output.extend(self._unicode_bytes[char] for char in token)
            except KeyError as error:
                raise ValueError(f"Qwen token {token!r} is outside the byte alphabet") from error
        return bytes(output)


@lru_cache(maxsize=1)
def load_qwen_bpe() -> QwenBpe:
    return QwenBpe.from_vendored_data()


@dataclass(frozen=True)
class OvisPromptTokens:
    """One templated Ovis sequence and its model/slicing policy."""

    ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    slice_start: int

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.attention_mask):
            raise ValueError("token ids and attention mask must have equal length")
        if not 0 <= self.slice_start < len(self.ids):
            raise ValueError("slice_start must point inside the token sequence")


@dataclass(frozen=True)
class ZImagePromptTokens:
    """One Qwen3-4B row under Z-Image's fixed chat template."""

    ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.ids or len(self.ids) != len(self.attention_mask):
            raise ValueError("Z-Image token IDs and attention mask must be non-empty and equal")
        if any(value != 1 for value in self.attention_mask):
            raise ValueError("Z-Image's unpadded token row must be fully attended")


def sd_tokenizer_segments(templated: str) -> list[str]:
    """Replicate SDTokenizer's raw-text preprocessing with weights disabled.

    SDTokenizer applies an escape/unescape round trip even when prompt
    weighting is disabled, removing the backslash from escaped
    parentheses. It also splits whitespace-delimited ``embedding:``
    references into independent tokenizer calls before checking whether
    an embedding directory exists; the split alone changes token
    boundaries (comfy/sd1_clip.py @ b78cec87).
    """
    templated = templated.replace("\\)", "\0\1").replace("\\(", "\0\2")
    templated = templated.replace("\0\1", ")").replace("\0\2", "(")
    split = re.split(r"(?<=\s)embedding:", templated)
    segments = [split[0], *(f"embedding:{part}" for part in split[1:])]
    return [segment for segment in segments if segment]


@dataclass(frozen=True)
class Krea2PromptTokens:
    """One Qwen3-VL-4B row under Krea 2's fixed chat template."""

    ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        ids = tuple(self.ids)
        mask = tuple(self.attention_mask)
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "attention_mask", mask)
        if not ids or len(ids) != len(mask):
            raise ValueError("Krea 2 token IDs and attention mask must be non-empty and equal")
        if any(type(token_id) is not int or token_id < 0 for token_id in ids):
            raise ValueError("Krea 2 token IDs must be non-negative integers")
        if any(type(value) is not int or value not in (0, 1) for value in mask):
            raise ValueError("Krea 2 attention mask values must be binary integers")
        boundary = mask.index(0) if 0 in mask else len(ids)
        if boundary == 0 or any(mask[boundary:]):
            raise ValueError("Krea 2 attention masks attend a non-empty prefix only")


@dataclass(frozen=True)
class Ideogram4PromptTokens:
    """One Qwen3-VL-8B row under the Ideogram 4 chat template."""

    ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.ids or len(self.ids) != len(self.attention_mask):
            raise ValueError("Ideogram 4 token IDs and attention mask must be non-empty and equal")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.ids):
            raise ValueError("Ideogram 4 token IDs must be non-negative integers")
        if any(type(value) is not int or value not in (0, 1) for value in self.attention_mask):
            raise ValueError("Ideogram 4 attention mask values must be binary integers")


def tokenize_ideogram4_prompt(
    text: str, *, tokenizer: QwenBpe | None = None
) -> Ideogram4PromptTokens:
    """Apply the ComfyUI Ideogram 4 Qwen3-VL prompt policy."""

    templated = format_ideogram4_prompt(text).text
    bpe = tokenizer or load_qwen_bpe()
    ids = tuple(
        token_id for segment in sd_tokenizer_segments(templated) for token_id in bpe.encode(segment)
    )
    first_pad = ids.index(QWEN_PAD) if QWEN_PAD in ids else len(ids)
    return Ideogram4PromptTokens(ids, (1,) * first_pad + (0,) * (len(ids) - first_pad))


def tokenize_krea2_prompt(text: str, *, tokenizer: QwenBpe | None = None) -> Krea2PromptTokens:
    """Apply the exact ComfyUI Krea 2 Qwen3-VL prompt policy.

    Preformatted prompts (starting with ``<|im_start|>``) bypass the
    template; the SDTokenizer escape round trip and embedding split
    apply to the final text either way. Krea 2 tokenizes with
    ``thinking=True``, so no empty think block is appended, and rows
    are never padded. The reference treats the first pad-token ID as
    end-of-sequence, masking it and everything after it; the row always
    opens with ``<|im_start|>``, so the mask is a ones prefix.
    """

    templated = format_krea2_prompt(text).text
    bpe = tokenizer or load_qwen_bpe()
    ids = tuple(
        token_id for segment in sd_tokenizer_segments(templated) for token_id in bpe.encode(segment)
    )
    first_pad = ids.index(QWEN_PAD) if QWEN_PAD in ids else len(ids)
    mask = (1,) * first_pad + (0,) * (len(ids) - first_pad)
    return Krea2PromptTokens(ids, mask)


def tokenize_z_image_prompt(text: str, *, tokenizer: QwenBpe | None = None) -> ZImagePromptTokens:
    """Apply the exact ComfyUI Z-Image Qwen3-4B prompt policy."""
    templated = Z_IMAGE_QWEN3_4B_CONFIG.prompt_template.format(text)
    ids = tuple((tokenizer or load_qwen_bpe()).encode(templated))
    return ZImagePromptTokens(ids, (1,) * len(ids))


@dataclass(frozen=True)
class Flux2KleinPromptTokens:
    """One Qwen3 row under Flux2 Klein's fixed chat template."""

    ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.attention_mask):
            raise ValueError("token ids and attention mask must have equal length")
        if len(self.ids) < KLEIN_QWEN3_8B_CONFIG.min_tokens:
            raise ValueError("Klein rows are padded to at least 512 tokens")


def tokenize_flux2_klein_prompt(
    text: str, *, tokenizer: QwenBpe | None = None
) -> Flux2KleinPromptTokens:
    """Apply the exact ComfyUI Flux2 Klein Qwen3 prompt policy.

    The 4B and 8B towers share one tokenizer and policy: fixed chat
    template, right-padding to 512 tokens with 151643, and attention
    masked from the first pad token onward, mirroring the reference
    mask construction (comfy/sd1_clip.py process_tokens @ b78cec87).
    """
    config = KLEIN_QWEN3_8B_CONFIG
    bpe = tokenizer or load_qwen_bpe()
    segments = sd_tokenizer_segments(config.prompt_template.format(text))
    ids = [token_id for segment in segments for token_id in bpe.encode(segment)]
    if len(ids) < config.min_tokens:
        ids.extend([config.pad_token_id] * (config.min_tokens - len(ids)))
    try:
        first_pad = ids.index(config.pad_token_id)
    except ValueError:
        first_pad = len(ids)
    mask = [1] * first_pad + [0] * (len(ids) - first_pad)
    return Flux2KleinPromptTokens(tuple(ids), tuple(mask))


def tokenize_ovis_prompt(text: str, *, tokenizer: QwenBpe | None = None) -> OvisPromptTokens:
    """Apply the fixed Ovis template with weighting disabled."""
    config = OVIS_QWEN3_2B_CONFIG
    segments = sd_tokenizer_segments(config.prompt_template.format(text))
    bpe = tokenizer or load_qwen_bpe()
    ids = [token_id for segment in segments for token_id in bpe.encode(segment)]
    if len(ids) < config.min_tokens:
        ids.extend([config.pad_token_id] * (config.min_tokens - len(ids)))
    try:
        first_pad = ids.index(config.pad_token_id)
    except ValueError:
        first_pad = len(ids)
    mask = [1] * first_pad + [0] * (len(ids) - first_pad)
    marker_id = config.slice_marker_id
    if marker_id is None:
        raise ValueError("Ovis prompt policy requires slice marker tokens")
    try:
        slice_start = ids.index(marker_id)
    except ValueError as error:
        raise ValueError("Ovis template marker token 4004 is missing") from error
    if slice_start + 1 < len(ids) and ids[slice_start + 1] == config.slice_marker_suffix_id:
        slice_start += 1
    return OvisPromptTokens(tuple(ids), tuple(mask), slice_start)


__all__ = [
    "QWEN_IM_END",
    "QWEN_IM_START",
    "QWEN_MERGES_SHA256",
    "QWEN_PAD",
    "QWEN_VOCAB_SHA256",
    "Flux2KleinPromptTokens",
    "Ideogram4PromptTokens",
    "Krea2PromptTokens",
    "OvisPromptTokens",
    "ZImagePromptTokens",
    "QwenBpe",
    "load_qwen_bpe",
    "merge_word",
    "sd_tokenizer_segments",
    "tokenize_flux2_klein_prompt",
    "tokenize_ideogram4_prompt",
    "tokenize_krea2_prompt",
    "tokenize_ovis_prompt",
    "tokenize_z_image_prompt",
]
