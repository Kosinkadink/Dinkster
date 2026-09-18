"""Anima's dual-tokenizer prompt policy - torch-free.

ComfyUI's AnimaTokenizer (comfy/text_encoders/anima.py @ 82f839f5)
runs the SAME prompt through two SDTokenizers: the Qwen 2.5 byte-level
BPE feeding the Qwen3-0.6B tower, and the vendored T5 SentencePiece
tokenizer whose ids and emphasis weights ride to the diffusion model's
LLM adapter. Both sides parse the ``(text:1.2)`` emphasis grammar and
split whitespace-preceded ``embedding:`` directives; the Qwen side
then forces every weight to 1.0 (the syntax is stripped but carries no
signal there), while the T5 side keeps its weights. Neither side
applies a chat template, adds a start token, or pads to a fixed
length: the Qwen row gains pad 151643 and the T5 row its trailing
EOS 1 only through the reference packer (min_length 1), so a Qwen row
is padded only when the prompt tokenizes to nothing.

The Qwen attention mask follows the reference process_tokens rule with
no end token (comfy/sd1_clip.py @ 82f839f5): a pad run OPENING the row
is masked while the text after it stays attended (a prompt can start
with a literal ``<|endoftext|>``, since no template precedes it), and
the first pad after that run ends attention - it and everything later
are masked. An empty prompt is one fully masked pad.

Both streams are golden-pinned against the executed reference
tokenizer (tests/goldens/anima_text_goldens.json).
"""

from __future__ import annotations

from dataclasses import dataclass

from .prompt_tokens import Chunk, TokenizerProfile, pack_spans, tokenize_prompt
from .qwen_bpe import QWEN_PAD, QwenBpe, load_qwen_bpe
from .t5_spm import T5_EOS, T5_PAD, T5SpmTokenizer, load_t5_spm

#: Anima's SDTokenizer shapes (comfy/text_encoders/anima.py
#: @ 82f839f5): no start token, unbounded chunk length, no per-chunk
#: padding, and the final chunk padded to at least one position. The
#: T5 side appends EOS 1 and pads with 0 (pad_with_end=False); the
#: Qwen side has no end token and pads with 151643.
_ANIMA_T5XXL_PROFILE = TokenizerProfile(
    max_length=99999999,
    start_token=None,
    end_token=T5_EOS,
    pad_token=T5_PAD,
    pad_to_max_length=False,
    min_length=1,
)
_ANIMA_QWEN_PROFILE = TokenizerProfile(
    max_length=99999999,
    start_token=None,
    end_token=None,
    pad_token=QWEN_PAD,
    pad_to_max_length=False,
    min_length=1,
)


@dataclass(frozen=True)
class AnimaPromptTokens:
    """One prompt's two token streams: the Qwen3-0.6B tower row and
    the T5-vocabulary ids and emphasis weights consumed by the
    diffusion model's LLM adapter."""

    qwen_ids: tuple[int, ...]
    qwen_attention_mask: tuple[int, ...]
    t5xxl_ids: tuple[int, ...]
    t5xxl_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.qwen_ids or len(self.qwen_ids) != len(self.qwen_attention_mask):
            raise ValueError("Anima Qwen ids and attention mask must be non-empty and equal")
        if any(value not in (0, 1) for value in self.qwen_attention_mask):
            raise ValueError("Anima attention mask values must be binary integers")
        if not self.t5xxl_ids or len(self.t5xxl_ids) != len(self.t5xxl_weights):
            raise ValueError("Anima T5 ids and weights must be non-empty and equal")
        if self.t5xxl_ids[-1] != T5_EOS:
            raise ValueError("Anima T5 rows end with EOS")


def _chunk_ids(chunk: Chunk) -> tuple[int, ...]:
    ids: list[int] = []
    for token in chunk:
        if not isinstance(token.unit, int):
            raise ValueError("Anima prompts resolve no embedding directives")
        ids.append(token.unit)
    return tuple(ids)


def _reference_mask(ids: tuple[int, ...], pad: int) -> tuple[int, ...]:
    """The reference process_tokens masking for a row with no end
    token: an opening pad run is masked without ending attention, and
    the first pad past it is masked along with everything after."""
    mask: list[int] = []
    left_pad = False
    for index, token in enumerate(ids):
        if index == 0 and token == pad:
            left_pad = True
        if left_pad:
            if token == pad:
                mask.append(0)
                continue
            left_pad = False
        if token == pad:
            mask.extend([0] * (len(ids) - index))
            break
        mask.append(1)
    return tuple(mask)


def tokenize_anima_prompt(
    text: str,
    *,
    qwen_tokenizer: QwenBpe | None = None,
    t5_tokenizer: T5SpmTokenizer | None = None,
) -> AnimaPromptTokens:
    """Apply the exact ComfyUI Anima dual-tokenizer prompt policy."""
    bpe = qwen_tokenizer or load_qwen_bpe()
    qwen_spans = tokenize_prompt(text, encode_word=bpe.encode).spans
    (qwen_chunk,) = pack_spans(qwen_spans, _ANIMA_QWEN_PROFILE)
    qwen_ids = _chunk_ids(qwen_chunk)
    mask = _reference_mask(qwen_ids, QWEN_PAD)

    spm = t5_tokenizer or load_t5_spm()
    t5_spans = tokenize_prompt(text, encode_word=spm.encode).spans
    (t5_chunk,) = pack_spans(t5_spans, _ANIMA_T5XXL_PROFILE)
    return AnimaPromptTokens(
        qwen_ids,
        mask,
        _chunk_ids(t5_chunk),
        tuple(token.weight for token in t5_chunk),
    )


__all__ = [
    "AnimaPromptTokens",
    "tokenize_anima_prompt",
]
