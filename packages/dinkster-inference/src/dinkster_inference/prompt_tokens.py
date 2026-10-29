"""Prompt-weight grammar, directive splitting, and chunk packing.

The torch-free half of ComfyUI's ``SDTokenizer.tokenize_with_weights``
(comfy/sd1_clip.py @ b78cec87), split along the stage-1 contract
boundaries (text_encoders):

- :func:`parse_prompt_weights` is the ``(text:1.2)`` emphasis grammar
  - the reference's escape_important/parse_parentheses/token_weights
  ported verbatim, including replace-not-multiply explicit weights,
  the 1.1x nesting default, last-colon parsing, and float artifacts
  (``((x))`` is exactly ``1.1 * 1.1``).
- :func:`tokenize_prompt` turns a prompt into WeightedSpans: one span
  per reference token group, in order, so a span's INDEX is its
  reference word id minus one. Empty spans are preserved (whitespace
  words tokenize to nothing but still consume a word id); dropped
  directives (unresolvable embeddings) consume none. BPE is injected
  as a ``WordEncoder`` so this module serves any family; embedding
  directives resolve through an ``EmbeddingResolver`` returning the
  embedding's vector count, keeping tokenization free of weight files.
- :func:`pack_spans` reshapes spans into model-sized chunks - BOS/EOS
  placement, the large-word split rule, and the pad family
  (pad_to_max_length / min_length / min_padding / pad_left) exactly as
  the reference packs, with embedding references expanded to per-row
  :class:`EmbeddingSlot` units.

Deliberate divergence: a bare ``embedding:`` directive with no name
crashes the reference with IndexError
(docs/comfyui-issues/sd1-clip-bare-embedding-directive-crash.md);
here it is reported as a missing embedding instead. Everything else
is golden-pinned against the executed reference
(tests/goldens/clip_tokenizer_goldens.json).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .clip_bpe import CLIP_BOS, CLIP_EOS
from .text_encoders import EmbeddingRef, WeightedSpan

#: Text of one word -> its inner token ids (no BOS/EOS), e.g.
#: ``ClipBpe.encode``.
WordEncoder = Callable[[str], Sequence[int]]

#: Embedding name -> vector count, or None when unknown. Called during
#: tokenization (directive resolution) and packing (row expansion);
#: must be deterministic across both.
EmbeddingResolver = Callable[[str], "int | None"]

_EMBEDDING_IDENTIFIER = "embedding:"
_ESCAPED_CLOSE = "\0\1"
_ESCAPED_OPEN = "\0\2"


def _escape_important(text: str) -> str:
    return text.replace("\\)", _ESCAPED_CLOSE).replace("\\(", _ESCAPED_OPEN)


def _unescape_important(text: str) -> str:
    return text.replace(_ESCAPED_CLOSE, ")").replace(_ESCAPED_OPEN, "(")


def _parse_parentheses(string: str) -> list[str]:
    """The reference's top-level paren grouper, verbatim: unbalanced
    input follows its nesting counter, never raises."""
    result: list[str] = []
    current = ""
    nesting = 0
    for char in string:
        if char == "(":
            if nesting == 0:
                if current:
                    result.append(current)
                current = "("
            else:
                current += char
            nesting += 1
        elif char == ")":
            nesting -= 1
            if nesting == 0:
                result.append(current + ")")
                current = ""
            else:
                current += char
        else:
            current += char
    if current:
        result.append(current)
    return result


def _token_weights(string: str, current_weight: float) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for x in _parse_parentheses(string):
        weight = current_weight
        if len(x) >= 2 and x[-1] == ")" and x[0] == "(":
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx + 1 :])
                    x = x[:xx]
                except ValueError:
                    pass
            out += _token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out


def parse_prompt_weights(
    text: str, *, disable_weights: bool = False
) -> tuple[tuple[str, float], ...]:
    """Prompt text -> ``(segment, weight)`` runs with escapes resolved.

    ``(text)`` multiplies the inherited weight by 1.1; ``(text:w)``
    REPLACES it with ``w`` (nested explicit weights do not compound);
    an unparseable ``:suffix`` stays literal text at the 1.1 default;
    ``\\(`` and ``\\)`` are literal parentheses. Segments may be empty
    strings; with ``disable_weights`` the whole prompt is one
    weight-1.0 segment (escapes still resolve).
    """
    escaped = _escape_important(text)
    if disable_weights:
        parsed = [(escaped, 1.0)]
    else:
        parsed = _token_weights(escaped, 1.0)
    return tuple((_unescape_important(segment), weight) for segment, weight in parsed)


@dataclass(frozen=True)
class TokenizedPrompt:
    """Spans in reference token-group order (span index = word id - 1,
    empty spans included) plus the embedding names that failed to
    resolve and were dropped."""

    spans: tuple[WeightedSpan, ...]
    missing_embeddings: tuple[str, ...] = ()


def _split_directive_name(name: str) -> tuple[str, str]:
    """First whitespace item is the candidate name, stopped at the
    first ``<`` or ``[``; everything else is leftover prompt text."""
    pieces = name.split()
    candidate = pieces[0]
    leftover = " ".join(pieces[1:])
    match = re.search(r"[<\[]", candidate)
    if match is not None:
        leftover = candidate[match.start() :] + (f" {leftover}" if leftover else "")
        candidate = candidate[: match.start()]
    return candidate, leftover


def tokenize_prompt(
    text: str,
    *,
    encode_word: WordEncoder,
    resolve: EmbeddingResolver | None = None,
    disable_weights: bool = False,
) -> TokenizedPrompt:
    """Prompt text -> weighted spans, replicating the reference's
    token-group construction exactly (including its quirks: comma-retry
    leftover rewriting, directive names dropped when unresolvable,
    leftover text never re-scanned for directives).

    Without ``resolve``, ``embedding:`` words are plain text - the
    reference behaves identically with no embedding directory.
    """
    spans: list[WeightedSpan] = []
    missing: list[str] = []
    for segment, weight in parse_prompt_weights(text, disable_weights=disable_weights):
        split = re.split(r"(?<=\s)" + re.escape(_EMBEDDING_IDENTIFIER), segment)
        words = [split[0]] + [f"{_EMBEDDING_IDENTIFIER}{part}" for part in split[1:]]
        for word in words:
            if not word:
                continue
            if word.startswith(_EMBEDDING_IDENTIFIER) and resolve is not None:
                raw = word[len(_EMBEDDING_IDENTIFIER) :].strip("\n")
                if not raw.split():
                    # Reference crash (IndexError) - see module docstring.
                    missing.append(raw)
                    continue
                name, leftover = _split_directive_name(raw)
                rows = resolve(name) if name else None
                if rows is None:
                    stripped = name.strip(",")
                    if len(stripped) < len(name):
                        rows = resolve(stripped) if stripped else None
                        leftover = f"{name[len(stripped) :]} {leftover}"
                        if rows is not None:
                            name = stripped
                if rows is None:
                    missing.append(name)
                else:
                    spans.append(WeightedSpan((EmbeddingRef(name),), weight))
                if leftover:
                    word = leftover
                else:
                    continue
            spans.append(WeightedSpan(tuple(encode_word(word)), weight))
    return TokenizedPrompt(tuple(spans), tuple(missing))


@dataclass(frozen=True)
class TokenizerProfile:
    """One family's chunk shape - the constructor knobs of the
    reference SDTokenizer that affect packing. ``min_length`` and
    ``min_padding`` apply to the final chunk only and can exceed
    ``max_length``; ``pad_left`` prepends every pad run instead of
    appending."""

    max_length: int = 77
    start_token: int | None = CLIP_BOS
    end_token: int | None = CLIP_EOS
    pad_token: int = CLIP_EOS
    pad_to_max_length: bool = True
    min_length: int | None = None
    min_padding: int | None = None
    pad_left: bool = False
    max_word_length: int = 8
    empty_has_end: bool = True

    def __post_init__(self) -> None:
        specials = (self.start_token is not None) + (self.end_token is not None)
        if self.max_length - specials < self.max_word_length - 1:
            # Any word below the large-word split threshold must fit in
            # an empty chunk or the reference algorithm cannot progress.
            raise ValueError(
                "max_length too small for max_word_length: chunks of "
                f"{self.max_length} cannot hold sub-threshold words of "
                f"{self.max_word_length - 1} tokens plus special tokens"
            )
        if self.max_word_length < 1:
            raise ValueError("max_word_length must be at least 1")


#: SD1/SDXL CLIP-L: pads with EOS.
CLIP_L_PROFILE = TokenizerProfile()
#: SDXL CLIP-G: pads with 0 (reference pad_with_end=False).
CLIP_G_PROFILE = TokenizerProfile(pad_token=0)


@dataclass(frozen=True)
class EmbeddingSlot:
    """One vector of a textual-inversion embedding occupying one chunk
    position; the encoder substitutes the actual vector at row ``row``."""

    name: str
    row: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("embedding name must not be empty")
        if self.row < 0:
            raise ValueError("embedding row must be non-negative")


PackedUnit = int | EmbeddingSlot


@dataclass(frozen=True)
class PackedToken:
    """One chunk position: a token id or embedding vector slot, its
    emphasis weight, and the reference word id (0 for specials/pads)."""

    unit: PackedUnit
    weight: float
    word_id: int


Chunk = tuple[PackedToken, ...]


def _pad(chunk: list[PackedToken], profile: TokenizerProfile, amount: int) -> None:
    pads = [PackedToken(profile.pad_token, 1.0, 0)] * amount
    if profile.pad_left:
        chunk[:0] = pads
    else:
        chunk.extend(pads)


def pack_spans(
    spans: Sequence[WeightedSpan],
    profile: TokenizerProfile = CLIP_L_PROFILE,
    *,
    resolve: EmbeddingResolver | None = None,
) -> tuple[Chunk, ...]:
    """Spans -> model-sized chunks, the reference reshape loop exactly:
    groups of ``max_word_length`` or more tokens split across chunk
    boundaries; shorter groups move whole to the next chunk (padding
    the one they left when ``pad_to_max_length``); specials and pads
    carry weight 1.0 and word id 0.

    Spans containing :class:`EmbeddingRef` units need ``resolve`` to
    expand them into per-row slots; an unresolvable reference here is
    an error (tokenize_prompt already dropped unknown names).
    """
    groups: list[list[tuple[PackedUnit, float]]] = []
    for span in spans:
        group: list[tuple[PackedUnit, float]] = []
        for unit in span.tokens:
            if isinstance(unit, EmbeddingRef):
                rows = resolve(unit.name) if resolve is not None else None
                if rows is None:
                    raise ValueError(
                        f"embedding {unit.name!r} is not resolvable at pack "
                        "time; tokenize and pack must share one resolver"
                    )
                group.extend((EmbeddingSlot(unit.name, row), span.weight) for row in range(rows))
            else:
                group.append((unit, span.weight))
        groups.append(group)

    has_end = 1 if profile.end_token is not None else 0
    chunks: list[list[PackedToken]] = []
    chunk: list[PackedToken] = []
    if profile.start_token is not None:
        chunk.append(PackedToken(profile.start_token, 1.0, 0))
    chunks.append(chunk)
    for i, group in enumerate(groups):
        is_large = len(group) >= profile.max_word_length
        while group:
            if len(group) + len(chunk) > profile.max_length - has_end:
                remaining = profile.max_length - len(chunk) - has_end
                if is_large:
                    chunk.extend(
                        PackedToken(unit, weight, i + 1) for unit, weight in group[:remaining]
                    )
                    if profile.end_token is not None:
                        chunk.append(PackedToken(profile.end_token, 1.0, 0))
                    group = group[remaining:]
                else:
                    if profile.end_token is not None:
                        chunk.append(PackedToken(profile.end_token, 1.0, 0))
                    if profile.pad_to_max_length:
                        _pad(chunk, profile, remaining)
                chunk = []
                if profile.start_token is not None:
                    chunk.append(PackedToken(profile.start_token, 1.0, 0))
                chunks.append(chunk)
            else:
                chunk.extend(PackedToken(unit, weight, i + 1) for unit, weight in group)
                group = []

    if profile.end_token is not None:
        chunk.append(PackedToken(profile.end_token, 1.0, 0))
    if profile.min_padding is not None:
        _pad(chunk, profile, profile.min_padding)
    if profile.pad_to_max_length and len(chunk) < profile.max_length:
        _pad(chunk, profile, profile.max_length - len(chunk))
    if profile.min_length is not None and len(chunk) < profile.min_length:
        _pad(chunk, profile, profile.min_length - len(chunk))

    return tuple(tuple(c) for c in chunks)


def empty_chunk(profile: TokenizerProfile, length: int) -> Chunk:
    """The empty-prompt chunk of ``length`` positions - the reference
    gen_empty_tokens (comfy/sd1_clip.py @ b78cec87): start + end +
    pad fill, all weight 1.0, word id 0. Encoders batch one of these
    beside weighted chunks as the interpolation baseline."""
    units: list[int] = []
    if profile.start_token is not None:
        units.append(profile.start_token)
    if profile.empty_has_end and profile.end_token is not None:
        units.append(profile.end_token)
    units.extend([profile.pad_token] * (length - len(units)))
    return tuple(PackedToken(unit, 1.0, 0) for unit in units)


@dataclass(frozen=True)
class PromptTokenizer:
    """The stage-1 ``Tokenizer`` contract bound to a word encoder and
    optional embedding resolver (``ClipBpe.encode`` + a directory
    resolver reproduces the reference SD1/SDXL tokenizers)."""

    encode_word: WordEncoder
    resolve: EmbeddingResolver | None = None
    disable_weights: bool = False

    def tokenize(self, text: str) -> tuple[WeightedSpan, ...]:
        return tokenize_prompt(
            text,
            encode_word=self.encode_word,
            resolve=self.resolve,
            disable_weights=self.disable_weights,
        ).spans


__all__ = [
    "CLIP_G_PROFILE",
    "CLIP_L_PROFILE",
    "Chunk",
    "EmbeddingResolver",
    "EmbeddingSlot",
    "PackedToken",
    "PackedUnit",
    "PromptTokenizer",
    "TokenizedPrompt",
    "TokenizerProfile",
    "WordEncoder",
    "empty_chunk",
    "parse_prompt_weights",
    "pack_spans",
    "tokenize_prompt",
]
