r"""Self-contained Mistral tekken BPE and the Flux2 dev prompt policy.

ComfyUI's Flux2 dev route reads the ``tekken_model`` blob embedded in
the Mistral3-Small text-encoder checkpoint and converts it with
transformers' MistralConverter (comfy/text_encoders/flux.py
``load_mistral_tokenizer`` @ b78cec87). This is that tokenizer without
a transformers runtime. Conversion facts mirrored exactly:

- The converter's default split pattern is used; the blob's own tekken
  v11 pattern is never passed (see
  docs/comfyui-issues/flux2-tekken-tokenizer.md). The default is Qwen's
  expression with up to three digits per number piece (``\p{N}{1,3}``),
  so the scanner is shared with qwen_bpe.
- Regular token ids are the blob rank plus ``default_num_special_tokens``
  (1000); entries with rank >= ``default_vocab_size`` - 1000 are dropped.
- The 1000 special tokens occupy ids 0..999 and are extracted from raw
  text leftmost-longest before pretokenization.
- Merges are derived tiktoken-style from the kept vocabulary: for every
  multi-byte token, each two-piece split whose halves are both kept
  becomes a merge, ordered locally by piece ranks and globally by
  whole-token rank. An exact vocabulary hit bypasses the merge loop
  (tokenizers' ``ignore_merges=True``).
- No normalizer runs; in particular there is no NFC pass.

The vendored copy of the blob pins the exact bytes of the
``tekken_model`` tensor in the official
``flux2/mistral_3_small_flux2_bf16.safetensors`` release
(https://huggingface.co/Comfy-Org/flux2-dev/) and serves detection-free
consumers such as tests; runtime assembly hands checkpoint bytes in
directly.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from functools import lru_cache

from .qwen_bpe import (
    _bytes_to_unicode,  # pyright: ignore[reportPrivateUsage]
    _qwen_pretokens,  # pyright: ignore[reportPrivateUsage]
    merge_word,
    sd_tokenizer_segments,
)
from .qwen_text import MISTRAL3_24B_CONFIG
from .vendored import read_vendored

TEKKEN_FLUX2_VENDORED = "flux2_tekken.json.gz"
TEKKEN_FLUX2_SHA256 = "6e2501687ccd0e1f30f36319eaf2b46958b897811e246cd8eb5d385b9e3de7d1"

TEKKEN_BOS = 1


class TekkenBpe:
    """The converted tekken vocabulary and byte-level BPE model."""

    def __init__(
        self,
        vocab: dict[str, int],
        merge_priorities: dict[tuple[str, str], int],
        special_ids: dict[str, int],
    ) -> None:
        self._vocab = vocab
        self._ranks = merge_priorities
        self._special_ids = special_ids
        self._special_split = re.compile(
            "("
            + "|".join(re.escape(token) for token in sorted(special_ids, key=len, reverse=True))
            + ")"
        )
        self._bytes = _bytes_to_unicode()
        self._cache: dict[str, tuple[str, ...]] = {}

    @classmethod
    def from_tekken_bytes(cls, data: bytes) -> TekkenBpe:
        """Convert one raw tekken JSON blob.

        The rank-as-id contract below only reproduces the reference when
        both blob lists are ordered by rank, because the reference
        assigns ids by enumerating them; a blob that breaks that order
        is refused rather than converted differently.
        """
        payload = json.loads(data)
        config = payload["config"]
        num_special = int(config["default_num_special_tokens"])
        max_rank = int(config["default_vocab_size"]) - num_special

        special_ids: dict[str, int] = {}
        for position, entry in enumerate(payload["special_tokens"]):
            if entry["rank"] != position:
                raise ValueError("tekken special_tokens must be ordered by contiguous rank")
            if "token_str" not in entry:
                raise ValueError("tekken special tokens without token_str are not supported")
            special_ids[str(entry["token_str"])] = position
        if len(special_ids) != num_special:
            raise ValueError(
                f"tekken declares {num_special} special tokens, found {len(special_ids)}"
            )

        byte_encoder = _bytes_to_unicode()
        byte_ranks: dict[bytes, int] = {}
        for position, entry in enumerate(payload["vocab"]):
            if entry["rank"] != position:
                raise ValueError("tekken vocab must be ordered by contiguous rank")
            if position >= max_rank:
                break
            byte_ranks[base64.b64decode(entry["token_bytes"])] = position
        if len(byte_ranks) != max_rank:
            raise ValueError(
                f"tekken vocab holds {len(byte_ranks)} kept tokens, expected {max_rank}"
            )

        def as_chars(token: bytes) -> str:
            return "".join(byte_encoder[value] for value in token)

        vocab = {as_chars(token): rank + num_special for token, rank in byte_ranks.items()}
        merge_priorities: dict[tuple[str, str], int] = {}
        for token in byte_ranks:
            if len(token) == 1:
                continue
            local: list[tuple[int, int, bytes, bytes]] = []
            for index in range(1, len(token)):
                left, right = token[:index], token[index:]
                left_rank = byte_ranks.get(left)
                right_rank = byte_ranks.get(right)
                if left_rank is not None and right_rank is not None:
                    local.append((left_rank, right_rank, left, right))
            local.sort(key=lambda item: (item[0], item[1]))
            for _, _, left, right in local:
                merge_priorities[(as_chars(left), as_chars(right))] = len(merge_priorities)
        return cls(vocab, merge_priorities, special_ids)

    def _plain(self, text: str) -> list[int]:
        ids: list[int] = []
        for pretoken in _qwen_pretokens(text, digit_run=3):
            encoded = "".join(self._bytes[value] for value in pretoken.encode("utf-8"))
            whole = self._vocab.get(encoded)
            if whole is not None:
                ids.append(whole)
                continue
            word = self._cache.get(encoded)
            if word is None:
                word = merge_word(tuple(encoded), self._ranks)
                self._cache[encoded] = word
            for token in word:
                token_id = self._vocab.get(token)
                if token_id is None:
                    raise ValueError(f"tekken vocabulary has no BPE token {token!r}")
                ids.append(token_id)
        return ids

    def encode(self, text: str) -> list[int]:
        """Tokenize raw text, extracting the 1000 special tokens first."""
        ids: list[int] = []
        for part in self._special_split.split(text):
            if not part:
                continue
            special = self._special_ids.get(part)
            if special is not None:
                ids.append(special)
            else:
                ids.extend(self._plain(part))
        return ids


@lru_cache(maxsize=1)
def load_flux2_tekken_bpe() -> TekkenBpe:
    return TekkenBpe.from_tekken_bytes(read_vendored(TEKKEN_FLUX2_VENDORED, TEKKEN_FLUX2_SHA256))


@dataclass(frozen=True)
class Flux2DevPromptTokens:
    """One Mistral3 row under Flux2 dev's fixed instruct template."""

    ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.ids or len(self.ids) != len(self.attention_mask):
            raise ValueError("Flux2 dev token ids and attention mask must be non-empty and equal")
        if self.ids[0] != TEKKEN_BOS:
            raise ValueError("Flux2 dev rows start with <s>")
        if any(value != 1 for value in self.attention_mask):
            raise ValueError("Flux2 dev rows are unpadded and fully attended")


def tokenize_flux2_dev_prompt(text: str, *, tokenizer: TekkenBpe) -> Flux2DevPromptTokens:
    """Apply the exact ComfyUI Flux2 dev Mistral3 prompt policy.

    The reference prepends ``<s>`` and never pads: its minimum length is
    one, which the start token always satisfies, and the model-side pad
    id 0 (``<unk>``) is never produced, so the mask is all ones
    (comfy/text_encoders/flux.py Mistral3Tokenizer/Mistral3_24BModel
    @ b78cec87).
    """
    segments = sd_tokenizer_segments(MISTRAL3_24B_CONFIG.prompt_template.format(text))
    ids = [
        TEKKEN_BOS,
        *(token_id for segment in segments for token_id in tokenizer.encode(segment)),
    ]
    return Flux2DevPromptTokens(tuple(ids), (1,) * len(ids))


__all__ = [
    "TEKKEN_BOS",
    "TEKKEN_FLUX2_SHA256",
    "TEKKEN_FLUX2_VENDORED",
    "Flux2DevPromptTokens",
    "TekkenBpe",
    "load_flux2_tekken_bpe",
    "tokenize_flux2_dev_prompt",
]
