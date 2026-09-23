"""YuE2 score, semantic-token, and acoustic-prefix generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
from dinkster_inference_torch.attention import AttentionKernel, select_attention
from dinkster_inference_torch.model_prefetch import (
    close_prefetch_queue,
    make_prefetch_queue,
    prefetch_queue_pop,
)
from dinkster_inference_torch.operations import INITLESS, Operations
from dinkster_inference_torch.qwen_text import QwenBlock, QwenTextModel
from tokenizers import Tokenizer

from .model import yue2_qwen_config

EOD = 151_643
ABC_START, ABC_END = 151_847, 151_848
MUSIC_START, MUSIC_END = 151_851, 151_852
CODEC_OFFSET, CODEC_SIZE = 151_853, 32_768
CONTEXT = 24_576
INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": (
        "Generate a melody-only ABC transcription without chord symbols, then generate music "
        "with codec tokens from the given conditions."
    ),
    "full": (
        "Generate a chord-annotated ABC transcription, then generate music with codec tokens "
        "from the given conditions."
    ),
}


def _rope_positions(
    head_dim: int,
    positions: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numerator = torch.arange(0, head_dim, 2, device=positions.device).float()
    inverse = 1.0 / (theta ** (numerator / head_dim))
    frequencies = positions.float()[..., None] * inverse
    embedding = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(1)
    sine = embedding.sin()
    half = head_dim // 2
    return embedding.cos(), sine[..., :half], -sine[..., half:]


@dataclass(frozen=True)
class YuE2Prompt:
    prefix: tuple[int, ...]
    negative: tuple[int, ...]
    abc_ids: tuple[int, ...]
    mode: str


def prompt_tokens(
    tokenizer: Tokenizer,
    style: str,
    lyrics: str,
    mode: str,
    abc: str = "",
) -> YuE2Prompt:
    if mode not in INSTRUCTIONS:
        raise ValueError("YuE2 mode must be off, melody, or full")
    prompt = f"{INSTRUCTIONS[mode]}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"
    return YuE2Prompt(
        (EOD, *tokenizer.encode(prompt).ids, ABC_START),
        (EOD, *tokenizer.encode(INSTRUCTIONS[mode]).ids),
        tuple(tokenizer.encode(abc).ids),
        mode,
    )


def distribution(
    logits: torch.Tensor,
    history: list[int],
    step: int,
    phase: Literal["abc", "semantic"],
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    penalty_window: int,
    min_tokens: int,
    legacy_off: bool = False,
) -> torch.Tensor:
    scores = logits.clone() if legacy_off else logits.float().clone()
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, -torch.inf)
    if phase == "abc":
        allowed[..., :EOD] = 0
    else:
        allowed[..., CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[..., end] = 0
    scores += allowed
    if step < min_tokens:
        scores[..., end] = -torch.inf
    if repetition_penalty != 1.0 and history:
        recent = torch.tensor([history[-penalty_window:]], dtype=torch.long, device=scores.device)
        counts = torch.zeros_like(scores)
        counts.scatter_add_(-1, recent, torch.ones_like(recent, dtype=scores.dtype))
        penalty = repetition_penalty**counts
        scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    if temperature == 0:
        return scores
    scores /= temperature
    threshold = scores.topk(min(top_k, scores.shape[-1])).values[..., -1, None]
    scores.masked_fill_(scores < threshold, -torch.inf)
    if top_p < 1:
        values, indices = scores.sort(descending=True)
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > top_p
        removed[..., : 3 if legacy_off else 1] = False
        values.masked_fill_(removed, -torch.inf)
        scores = values.scatter(-1, indices, values)
    return scores


def chunk_ranges(
    frames: int, prefix_tokens: int, context: int = CONTEXT
) -> tuple[tuple[int, int], ...]:
    size = (context - prefix_tokens - 3) // 2
    if frames < 1 or size < 1:
        raise ValueError("YuE2 needs tokens and context for at least one acoustic frame")
    return tuple((start, min(start + size, frames)) for start in range(0, frames, size))


class YuE2TextBackbone(QwenTextModel):
    def __init__(
        self,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel | None = None,
    ) -> None:
        config = yue2_qwen_config()
        kernel = select_attention("qwen").kernel if attention_kernel is None else attention_kernel
        super().__init__(config, operations=operations, attention_kernel=kernel)
        self.lm_head = operations.linear(config.hidden_size, config.vocab_size, bias=False)


class YuE2TextModel(torch.nn.Module):
    def __init__(
        self,
        _config: object = None,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel | None = None,
    ) -> None:
        super().__init__()
        self.model = YuE2TextBackbone(
            operations=operations,
            attention_kernel=attention_kernel,
        )

    def _prefill(
        self,
        prefixes: list[tuple[int, ...]],
        capacity: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        device = self.model.embed_tokens.weight.device
        length = max(map(len, prefixes))
        ids = torch.tensor(
            [[0] * (length - len(prefix)) + list(prefix) for prefix in prefixes],
            device=device,
            dtype=torch.long,
        )
        visible = torch.zeros((len(prefixes), capacity), device=device, dtype=torch.bool)
        for index, prefix in enumerate(prefixes):
            visible[index, length - len(prefix) : length] = True
        hidden = self.model.embed_tokens(ids)
        causal = torch.full(
            (length, length),
            torch.finfo(hidden.dtype).min / 4,
            device=device,
            dtype=hidden.dtype,
        ).triu_(1)
        mask = causal[None, None] + (~visible[:, None, None, :length]).to(hidden.dtype) * (
            torch.finfo(hidden.dtype).min / 4
        )
        positions = visible[:, :length].cumsum(-1).sub_(1).clamp_min_(0)
        frequencies = _rope_positions(
            self.model.config.head_dim,
            positions,
            self.model.config.rope_theta,
        )
        caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        prefetch = make_prefetch_queue(self.model.layers)
        try:
            for layer in self.model.layers:
                prefetch_queue_pop(prefetch, layer)
                hidden, key, value = cast("QwenBlock", layer).forward_causal(
                    hidden, mask, frequencies, None, 0
                )
                cache_key = key.new_empty((key.shape[0], key.shape[1], capacity, key.shape[3]))
                cache_value = value.new_empty(
                    (value.shape[0], value.shape[1], capacity, value.shape[3])
                )
                cache_key[:, :, :length].copy_(key)
                cache_value[:, :, :length].copy_(value)
                caches.append((cache_key, cache_value))
            prefetch_queue_pop(prefetch, None)
        finally:
            close_prefetch_queue(prefetch)
        logits = self.model.lm_head(self.model.norm(hidden)[:, -1])
        return logits, caches, visible

    def _decode(
        self,
        token: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        visible: torch.Tensor,
        cache_position: int,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.model.embed_tokens(token)
        frequencies = _rope_positions(
            self.model.config.head_dim,
            positions,
            self.model.config.rope_theta,
        )
        mask = (~visible[:, None, None, : cache_position + 1]).to(hidden.dtype) * (
            torch.finfo(hidden.dtype).min / 4
        )
        prefetch = make_prefetch_queue(self.model.layers)
        try:
            for index, layer in enumerate(self.model.layers):
                prefetch_queue_pop(prefetch, layer)
                hidden, _key, _value = cast("QwenBlock", layer).forward_causal(
                    hidden, mask, frequencies, caches[index], cache_position
                )
            prefetch_queue_pop(prefetch, None)
        finally:
            close_prefetch_queue(prefetch)
        return self.model.lm_head(self.model.norm(hidden)[:, -1])

    def generate(
        self,
        prefix: tuple[int, ...],
        *,
        seed: int,
        max_tokens: int,
        phase: Literal["abc", "semantic"],
        negative: tuple[int, ...] | None = None,
        cfg_scale: float = 1.0,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        penalty_window: int,
        min_tokens: int,
        legacy_off: bool = False,
    ) -> tuple[list[int], bool]:
        prefixes = [prefix] if cfg_scale == 1.0 else [prefix, negative or ()]
        prefix_length = max(map(len, prefixes))
        if prefix_length + max_tokens > CONTEXT:
            raise ValueError("YuE2 prompt plus generation exceeds the model context")
        logits, caches, visible = self._prefill(prefixes, prefix_length + max_tokens)
        generator = torch.Generator(device=logits.device).manual_seed(seed)
        history: list[int] = []
        end = ABC_END if phase == "abc" else MUSIC_END
        for step in range(max_tokens):
            guided = (
                logits if cfg_scale == 1.0 else logits[1:] + cfg_scale * (logits[:1] - logits[1:])
            )
            scores = distribution(
                guided,
                history,
                step,
                phase,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                penalty_window=penalty_window,
                min_tokens=min_tokens,
                legacy_off=legacy_off,
            )
            next_id = (
                scores.argmax(-1, keepdim=True)
                if temperature == 0
                else torch.multinomial(scores.softmax(-1), 1, generator=generator)
            )
            token = int(next_id.item())
            if token == end:
                return history, False
            history.append(token)
            if step + 1 < max_tokens:
                visible[:, prefix_length + step] = True
                decode = next_id.expand(len(prefixes), 1)
                positions = torch.tensor(
                    [[len(item) + step] for item in prefixes],
                    device=decode.device,
                    dtype=torch.long,
                )
                logits = self._decode(
                    decode,
                    caches,
                    visible,
                    prefix_length + step,
                    positions,
                )
        return history, True

    def acoustic_conditioning(
        self,
        prefix: tuple[int, ...],
        tokens: list[int],
    ) -> tuple[torch.Tensor, tuple[tuple[int, int, int, int], ...]]:
        ranges = chunk_ranges(len(tokens), len(prefix))
        chunks: list[tuple[int, int, int, int]] = []
        outputs: list[torch.Tensor] = []
        offset = 0
        for start, end in ranges:
            ids = (*prefix, *tokens[start:end], MUSIC_END)
            _logits, caches, _visible = self._prefill([ids], len(ids))
            stacked = torch.stack(
                [torch.stack((key, value), dim=2) for key, value in caches], dim=2
            )
            output = stacked.permute(0, 4, 2, 3, 1, 5).flatten(2)
            outputs.append(output)
            chunks.append((start, end, offset, offset + len(ids)))
            offset += len(ids)
        return torch.cat(outputs, dim=1), tuple(chunks)


def tokenizer_from_bytes(data: bytes) -> Tokenizer:
    return Tokenizer.from_str(data.decode("utf-8"))


__all__ = [
    "YuE2Prompt",
    "YuE2TextModel",
    "chunk_ranges",
    "distribution",
    "prompt_tokens",
    "tokenizer_from_bytes",
]
