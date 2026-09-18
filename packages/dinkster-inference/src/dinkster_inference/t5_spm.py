"""Self-contained T5 SentencePiece/Unigram tokenizer - no transformers.

ComfyUI tokenizes T5 prompts through Hugging Face ``T5TokenizerFast``
loaded from its vendored ``comfy/text_encoders/t5_tokenizer/``
tokenizer.json (comfy/text_encoders/flux.py T5XXLTokenizer @
b78cec87). This module reimplements that exact pipeline - tokenizers
0.22.2 semantics, verified against its source - on the stdlib:

- Added-token extraction runs FIRST, on the raw unnormalized text:
  the 103 vendored specials (``<pad>``, ``</s>``, ``<unk>``,
  ``<extra_id_0>``..``<extra_id_99>``) match leftmost-longest,
  case-sensitively, with no whitespace absorption
  (added_vocabulary.rs, MatchKind::LeftmostLongest).
- Normalization per remaining piece: the SentencePiece
  ``Precompiled`` charsmap (spm_precompiled 0.1.3 double-array trie,
  applied per extended grapheme cluster with whole-cluster lookup
  only under 6 bytes - normalizers/precompiled.rs), then
  ``Strip(right)`` over Unicode White_Space, then the regex
  ``" {2,}"`` replaced by U+2581.
- Metaspace pre-tokenization: spaces become U+2581, the piece whose
  ORIGINAL offset is 0 gets a U+2581 prefix unless it already starts
  with one (prepend_scheme "first"; prepending onto an empty piece is
  a no-op like the reference), then splits keeping each delimiter
  merged with what follows (SplitDelimiterBehavior::MergedWithNext).
- Unigram Viterbi per pre-token over the vendored vocabulary
  (models/unigram/model.rs encode_optimized): strict ``>`` best-path
  comparisons (first-written wins ties), unknown edges spanning one
  scalar scored ``min_score - 10.0`` and only added when no
  vocabulary token covers exactly that scalar, consecutive unknowns
  fused into one ``<unk>``.

``encode`` returns the INNER token ids - what the reference slices
out of Hugging Face output as ``input_ids[:-1]`` (T5 has no BOS; the
template processor's trailing EOS belongs to the chunk packer, like
CLIP's specials). Goldens pin this module against the executed
reference tokenizer, never a re-derivation
(tests/goldens/t5_tokenizer_goldens.json).
"""

from __future__ import annotations

import base64
import json
import re
from functools import lru_cache

from .graphemes import grapheme_clusters
from .prompt_tokens import TokenizerProfile
from .vendored import read_vendored

#: Special-token ids from the vendored vocabulary.
T5_PAD = 0
T5_EOS = 1
T5_UNK = 2
#: The t5xxl EMBEDDING table is padded past the 32100-entry vocab.
T5_VOCAB_SIZE = 32128

#: sha256 of the UNCOMPRESSED vendored tokenizer.json, byte-identical
#: to ComfyUI's comfy/text_encoders/t5_tokenizer/tokenizer.json
#: @ b78cec87.
T5_TOKENIZER_SHA256 = "6cd1badb7467cdab971403bbeb1945070ca385652385a2b6549f672b08089dfc"

_SPACE_PIECE = "\u2581"
_MULTISPACE = re.compile(" {2,}")
_UNK_PENALTY = 10.0

#: Rust ``char::is_whitespace`` - the Unicode White_Space property
#: (Strip normalizer semantics; NOT Python str.isspace, which also
#: accepts U+001C..U+001F).
_WHITE_SPACE = frozenset(
    chr(cp)
    for cp in (
        *range(0x0009, 0x000E),
        0x0020,
        0x0085,
        0x00A0,
        0x1680,
        *range(0x2000, 0x200B),
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
    )
)


class _DoubleArray:
    """spm_precompiled 0.1.3 DoubleArray: u32 units where bit 8 is
    has_leaf, bits 0..30 the leaf value, low byte + bit 31 the label,
    and the offset is ``(unit >> 10) << ((unit & 512) >> 6)``."""

    def __init__(self, units: list[int]) -> None:
        self._units = units

    def common_prefix_search(self, key: bytes) -> list[int]:
        units = self._units
        results: list[int] = []
        unit = units[0]
        node_pos = (unit >> 10) << ((unit & (1 << 9)) >> 6)
        for c in key:
            if c == 0:
                break
            node_pos ^= c
            unit = units[node_pos]
            if unit & 0x800000FF != c:
                return results
            node_pos ^= (unit >> 10) << ((unit & (1 << 9)) >> 6)
            if (unit >> 8) & 1:
                results.append(units[node_pos] & 0x7FFFFFFF)
        return results


class _Precompiled:
    """The SentencePiece precompiled charsmap: binary layout
    ``[u32 trie byte length][trie: u32 array][normalized string]``,
    where trie leaf values are byte offsets into the NUL-delimited
    normalized string (spm_precompiled 0.1.3 lib.rs)."""

    def __init__(self, charsmap: bytes) -> None:
        trie_size = int.from_bytes(charsmap[:4], "little")
        count = trie_size // 4
        units = [int.from_bytes(charsmap[4 + 4 * i : 8 + 4 * i], "little") for i in range(count)]
        self._trie = _DoubleArray(units)
        self._normalized = charsmap[4 + 4 * count :]

    def transform(self, chunk: str) -> str | None:
        results = self._trie.common_prefix_search(chunk.encode("utf-8"))
        if not results:
            return None
        start = results[0]
        end = self._normalized.find(b"\0", start)
        if end == -1:
            end = len(self._normalized)
        return self._normalized[start:end].decode("utf-8")

    def normalize(self, text: str) -> str:
        """normalizers/precompiled.rs @ tokenizers 0.22.2: per
        extended grapheme cluster, try the whole cluster only when it
        is under 6 UTF-8 bytes; otherwise (or on miss) transform each
        scalar independently, keeping unmatched scalars."""
        out: list[str] = []
        for grapheme in grapheme_clusters(text):
            if len(grapheme.encode("utf-8")) < 6:
                norm = self.transform(grapheme)
                if norm is not None:
                    out.append(norm)
                    continue
            for char in grapheme:
                norm = self.transform(char)
                out.append(char if norm is None else norm)
        return "".join(out)


#: Byte-trie nodes: byte -> child node, and -1 -> the leaf token id.
_TrieNode = dict[int, "int | _TrieNode"]


class _Unigram:
    """models/unigram/model.rs @ tokenizers 0.22.2, the
    encode_optimized path (deserialized models are always optimized
    and always fuse unknowns)."""

    def __init__(self, vocab: list[tuple[str, float]], unk_id: int) -> None:
        self._vocab = vocab
        self._unk_id = unk_id
        self._min_score = min(score for _, score in vocab)
        trie: _TrieNode = {}
        for token_id, (token, _) in enumerate(vocab):
            node = trie
            for byte in token.encode("utf-8"):
                child = node.setdefault(byte, {})
                assert isinstance(child, dict)
                node = child
            # Last id wins for duplicated strings, like the
            # reference's token_to_ids HashMap::insert.
            node[-1] = token_id
        self._trie = trie

    def encode(self, piece: str) -> list[int]:
        if not piece:
            return []
        data = piece.encode("utf-8")
        size = len(data)
        # best_path_ends_at[pos] = (score, starts_at | None, id).
        best: list[tuple[float, int | None, int]] = [(0.0, None, 0) for _ in range(size + 1)]
        unk_score = self._min_score - _UNK_PENALTY

        # Char boundary byte lengths, walked left to right.
        starts_at = 0
        for char in piece:
            mblen = len(char.encode("utf-8"))
            score_here = best[starts_at][0]
            has_single_node = False
            node = self._trie
            pos = starts_at
            while pos < size:
                child = node.get(data[pos])
                if not isinstance(child, dict):
                    break
                node = child
                pos += 1
                leaf = node.get(-1)
                if isinstance(leaf, int):
                    candidate = self._vocab[leaf][1] + score_here
                    target = best[pos]
                    if target[1] is None or candidate > target[0]:
                        best[pos] = (candidate, starts_at, leaf)
                    if pos - starts_at == mblen:
                        has_single_node = True
            if not has_single_node:
                candidate = unk_score + score_here
                target = best[starts_at + mblen]
                if target[1] is None or candidate > target[0]:
                    best[starts_at + mblen] = (candidate, starts_at, self._unk_id)
            starts_at += mblen

        # Backtrack, fusing consecutive unknown spans (fuse_unk).
        ids: list[int] = []
        ends_at = size
        unk_pending = False
        while ends_at > 0:
            _, node_start, node_id = best[ends_at]
            assert node_start is not None
            if node_id == self._unk_id:
                unk_pending = True
            else:
                if unk_pending:
                    ids.append(self._unk_id)
                    unk_pending = False
                ids.append(node_id)
            ends_at = node_start
        if unk_pending:
            ids.append(self._unk_id)
        ids.reverse()
        return ids


def _metaspace(piece: str, at_origin: bool) -> list[str]:
    """pre_tokenizers/metaspace.rs @ tokenizers 0.22.2 with
    replacement U+2581, prepend_scheme "first", split on: spaces
    become the replacement; the piece starting at original offset 0
    is prefixed unless it already starts with the replacement (a
    no-op on empty pieces, like NormalizedString::prepend); then
    split keeping each delimiter merged with the following text."""
    piece = piece.replace(" ", _SPACE_PIECE)
    if at_origin and piece and not piece.startswith(_SPACE_PIECE):
        piece = _SPACE_PIECE + piece
    if not piece:
        return []
    splits: list[str] = []
    start = 0
    for i in range(1, len(piece)):
        if piece[i] == _SPACE_PIECE:
            splits.append(piece[start:i])
            start = i
    splits.append(piece[start:])
    return splits


class T5SpmTokenizer:
    """The vendored-vocabulary T5 tokenizer. Construction parses
    ~2.4 MB of JSON and builds the Unigram trie; share one instance
    via :func:`load_t5_spm`. Stateless after construction; results
    are pure functions of the input text."""

    def __init__(
        self,
        precompiled: _Precompiled,
        unigram: _Unigram,
        added_tokens: dict[str, int],
    ) -> None:
        self._precompiled = precompiled
        self._unigram = unigram
        self._added = added_tokens
        # Leftmost-longest extraction: no vendored special is a
        # prefix of another, so a longest-first alternation scan is
        # exactly aho-corasick LeftmostLongest here.
        self._added_pattern = re.compile(
            "|".join(re.escape(token) for token in sorted(added_tokens, key=len, reverse=True))
        )

    @classmethod
    def from_vendored_data(cls) -> T5SpmTokenizer:
        spec = json.loads(read_vendored("t5_tokenizer.json.gz", T5_TOKENIZER_SHA256))
        normalizers = spec["normalizer"]["normalizers"]
        precompiled = _Precompiled(base64.b64decode(normalizers[0]["precompiled_charsmap"]))
        vocab = [(token, score) for token, score in spec["model"]["vocab"]]
        unigram = _Unigram(vocab, spec["model"]["unk_id"])
        added = {entry["content"]: entry["id"] for entry in spec["added_tokens"]}
        return cls(precompiled, unigram, added)

    def normalize(self, text: str) -> str:
        """The full normalizer chain for one added-token-free piece:
        Precompiled charsmap, right-strip Unicode White_Space, then
        space runs of 2+ become one U+2581."""
        text = self._precompiled.normalize(text)
        end = len(text)
        while end > 0 and text[end - 1] in _WHITE_SPACE:
            end -= 1
        return _MULTISPACE.sub(_SPACE_PIECE, text[:end])

    def pre_tokenize(self, text: str, *, at_origin: bool = True) -> list[str]:
        """Normalized pre-token pieces for one added-token-free
        piece of raw text."""
        return _metaspace(self.normalize(text), at_origin)

    def encode(self, text: str) -> tuple[int, ...]:
        """Text -> inner token ids (no trailing EOS). Identical to
        the reference's ``tokenizer(text)["input_ids"][:-1]``."""
        ids: list[int] = []
        cursor = 0
        for match in self._added_pattern.finditer(text) if self._added else ():
            piece = text[cursor : match.start()]
            if piece:
                ids.extend(self._encode_piece(piece, at_origin=cursor == 0))
            ids.append(self._added[match.group()])
            cursor = match.end()
        tail = text[cursor:]
        if tail:
            ids.extend(self._encode_piece(tail, at_origin=cursor == 0))
        return tuple(ids)

    def _encode_piece(self, piece: str, *, at_origin: bool) -> list[int]:
        out: list[int] = []
        for pretoken in self.pre_tokenize(piece, at_origin=at_origin):
            out.extend(self._unigram.encode(pretoken))
        return out


@lru_cache(maxsize=1)
def load_t5_spm() -> T5SpmTokenizer:
    """The shared vendored-data instance (parsing is done once)."""
    return T5SpmTokenizer.from_vendored_data()


#: Flux T5-XXL chunk shape (comfy/text_encoders/flux.py
#: T5XXLTokenizer @ b78cec87): no start token, EOS 1, pads with 0
#: (pad_with_end=False), no per-chunk padding, the final chunk padded
#: to at least 256, and a max_length so large packing never splits.
#: SD3's T5 uses min_length=77; that profile ships with the SD3
#: family. Golden-pinned against the executed reference
#: (tests/goldens/t5_tokenizer_goldens.json "weighted").
T5_XXL_FLUX_PROFILE = TokenizerProfile(
    max_length=99999999,
    start_token=None,
    end_token=T5_EOS,
    pad_token=T5_PAD,
    pad_to_max_length=False,
    min_length=256,
)

#: LTXV's T5 pads to at least 128 tokens (comfy/text_encoders/lt.py
#: LTXVT5Tokenizer @ b78cec87); otherwise the Flux profile.
T5_XXL_LTXV_PROFILE = TokenizerProfile(
    max_length=99999999,
    start_token=None,
    end_token=T5_EOS,
    pad_token=T5_PAD,
    pad_to_max_length=False,
    min_length=128,
)

#: PixArt/Chroma T5-XXL uses no minimum padding. Its weighted-prompt
#: baseline removes EOS and is therefore all pad tokens.
T5_XXL_PIXART_PROFILE = TokenizerProfile(
    max_length=99999999,
    start_token=None,
    end_token=T5_EOS,
    pad_token=T5_PAD,
    pad_to_max_length=False,
    min_length=1,
    empty_has_end=False,
)


__all__ = [
    "T5_EOS",
    "T5_PAD",
    "T5_TOKENIZER_SHA256",
    "T5_UNK",
    "T5_VOCAB_SIZE",
    "T5_XXL_FLUX_PROFILE",
    "T5_XXL_LTXV_PROFILE",
    "T5_XXL_PIXART_PROFILE",
    "T5SpmTokenizer",
    "load_t5_spm",
]
