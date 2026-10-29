"""Native MiniMax Music 3 autoregressive text-to-music conditioner."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from dinkster_inference import (
    AUDIO_CODE_OFFSET,
    C0_VOCAB_SIZE,
    DEFAULT_CFG_SCALE,
    DEFAULT_TOP_K,
    MAX_AUDIO_FRAMES,
    MAX_PROMPT_TOKENS,
    SPECIAL_TOKEN_IDS,
    MiniMaxMusic3TextConfig,
    QwenTextConfig,
    build_music_prompt,
    derive_music_seed,
)
from tokenizers import Tokenizer

from .attention import AttentionKernel, select_attention
from .model_prefetch import close_prefetch_queue, make_prefetch_queue, prefetch_queue_pop
from .operations import INITLESS, Operations, module_compute_device
from .quant_linear import linear_input_act
from .qwen_text import QwenTextModel

_DEFAULT_ATTENTION = select_attention("qwen").kernel


def _qwen_config(config: MiniMaxMusic3TextConfig) -> QwenTextConfig:
    return QwenTextConfig(
        architecture="minimax_music3",
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=MAX_PROMPT_TOKENS + MAX_AUDIO_FRAMES + 1,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        qkv_bias=False,
        qk_norm=True,
        prompt_template="{}",
        min_tokens=1,
        pad_token_id=0,
        attention_head_dim=config.head_dim,
        merged_qkv=config.merged_qkv,
        merged_mlp=config.merged_mlp,
    )


def sample_top_k(
    logits: torch.Tensor,
    top_k: int,
    generator: torch.Generator,
) -> torch.Tensor:
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    top_k = min(top_k, values.shape[-1])
    threshold = torch.topk(values, top_k, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    probabilities = torch.nan_to_num(torch.softmax(values, dim=-1), nan=0.0)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


class MiniMaxMusic3RvqAttention(torch.nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: MiniMaxMusic3TextConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.heads = config.decoder_num_heads
        self.head_dim = config.hidden_size // self.heads
        if config.decoder_merged_qkv:
            self.qkv_proj = operations.linear(
                config.hidden_size, 3 * config.hidden_size, bias=False
            )
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            self.qkv_proj = None
            self.q_proj = operations.linear(config.hidden_size, config.hidden_size, bias=False)
            self.k_proj = operations.linear(config.hidden_size, config.hidden_size, bias=False)
            self.v_proj = operations.linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = operations.linear(config.hidden_size, config.hidden_size, bias=False)
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        if self.qkv_proj is not None:
            query, key, value = self.qkv_proj(hidden).chunk(3, dim=-1)
        else:
            assert self.q_proj is not None and self.k_proj is not None and self.v_proj is not None
            query, key, value = (
                self.q_proj(hidden),
                self.k_proj(hidden),
                self.v_proj(hidden),
            )
        shape = (batch, length, self.heads, self.head_dim)
        query = query.reshape(shape).transpose(1, 2)
        key = key.reshape(shape).transpose(1, 2)
        value = value.reshape(shape).transpose(1, 2)
        mask = torch.full(
            (length, length),
            torch.finfo(query.dtype).min,
            device=query.device,
            dtype=query.dtype,
        ).triu_(1)
        output = self._attention_kernel(query, key, value, mask=mask)
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, width))


class MiniMaxMusic3RvqMlp(torch.nn.Module):
    def __init__(self, config: MiniMaxMusic3TextConfig, *, operations: Operations) -> None:
        super().__init__()
        if config.decoder_merged_mlp:
            self.gate_up_proj = operations.linear(
                config.hidden_size, 2 * config.decoder_intermediate_size, bias=False
            )
            self.gate_proj = self.up_proj = None
        else:
            self.gate_up_proj = None
            self.gate_proj = operations.linear(
                config.hidden_size, config.decoder_intermediate_size, bias=False
            )
            self.up_proj = operations.linear(
                config.hidden_size, config.decoder_intermediate_size, bias=False
            )
        self.down_proj = operations.linear(
            config.decoder_intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.gate_up_proj is not None:
            return linear_input_act(
                self.down_proj,
                self.gate_up_proj(hidden),
                "swiglu",
            )
        assert self.gate_proj is not None and self.up_proj is not None
        gate, value = self.gate_proj(hidden), self.up_proj(hidden)
        return self.down_proj(F.silu(gate) * value)


class MiniMaxMusic3RvqBlock(torch.nn.Module):
    def __init__(
        self,
        config: MiniMaxMusic3TextConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.input_layernorm = operations.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MiniMaxMusic3RvqAttention(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.post_attention_layernorm = operations.rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = MiniMaxMusic3RvqMlp(config, operations=operations)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden))
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class MiniMaxMusic3RvqDecoder(torch.nn.Module):
    def __init__(
        self,
        config: MiniMaxMusic3TextConfig,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__()
        self.projection = operations.linear(config.hidden_size, config.hidden_size, bias=False)
        self.pos_embedding = operations.embedding(16, config.hidden_size)
        self.audio_heads = torch.nn.ModuleList(
            operations.linear(config.hidden_size, config.audio_vocab_size, bias=False)
            for _ in range(config.audio_num_codebooks - 1)
        )
        self.layers = torch.nn.ModuleList(
            MiniMaxMusic3RvqBlock(config, operations=operations, attention_kernel=attention_kernel)
            for _ in range(config.decoder_num_layers)
        )
        self.norm = operations.rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, sequence: torch.Tensor, *, prefetched: bool = False) -> torch.Tensor:
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        hidden = sequence + self.pos_embedding(positions).unsqueeze(0)
        prefetch = None if prefetched else make_prefetch_queue(self.layers)
        try:
            for layer in self.layers:
                prefetch_queue_pop(prefetch, layer)
                hidden = layer(hidden)
            prefetch_queue_pop(prefetch, None)
        finally:
            close_prefetch_queue(prefetch)
        return self.norm(hidden)


class _MiniMaxMusic3Qwen(QwenTextModel):
    embed_tokens: torch.nn.Embedding
    embed_tokens_prefill: torch.nn.Embedding
    embed_tokens_audio: torch.nn.Embedding
    lm_head: torch.nn.Linear
    lm_head_pruned: torch.nn.Linear
    audio_extra_embedding: torch.nn.Embedding
    audio_decoder: MiniMaxMusic3RvqDecoder


class MiniMaxMusic3TextModel(torch.nn.Module):
    def __init__(
        self,
        config: MiniMaxMusic3TextConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = _MiniMaxMusic3Qwen(
            _qwen_config(config),
            operations=operations,
            attention_kernel=attention_kernel,
        )
        if config.pruned:
            del self.model.embed_tokens
            self.model.embed_tokens_prefill = operations.embedding(
                AUDIO_CODE_OFFSET, config.hidden_size
            )
            self.model.embed_tokens_audio = operations.embedding(C0_VOCAB_SIZE, config.hidden_size)
            self.model.lm_head_pruned = operations.linear(
                config.hidden_size, C0_VOCAB_SIZE + 1, bias=False
            )
        else:
            self.model.lm_head = operations.linear(
                config.hidden_size, config.vocab_size, bias=False
            )
        self.model.audio_extra_embedding = operations.embedding(
            config.audio_vocab_size * (config.audio_num_codebooks - 1),
            config.hidden_size,
        )
        self.model.audio_decoder = MiniMaxMusic3RvqDecoder(
            config,
            operations=operations,
            attention_kernel=attention_kernel,
        )

    def _guided_c0(self, logits: torch.Tensor, cfg_scale: float, top_k: int) -> torch.Tensor:
        conditioned = logits[0:1].float()
        unconditioned = logits[1:2].float()
        guided = unconditioned + (conditioned - unconditioned) * cfg_scale
        threshold = torch.topk(conditioned, top_k, dim=-1).values[..., -1, None]
        return guided.masked_fill(conditioned < threshold, -float("inf"))

    def _depth_codes(
        self,
        hidden: torch.Tensor,
        c0: torch.Tensor,
        c0_embed: torch.Tensor,
        generator: torch.Generator,
        cfg_scale: float,
        top_k: int,
        *,
        prefetched: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoder = self.model.audio_decoder
        sequence = [decoder.projection(hidden).unsqueeze(1)]
        sequence.append(decoder.projection(c0_embed).unsqueeze(1))
        codes = [c0]
        hidden_parts: list[torch.Tensor] = []
        for index in range(1, self.config.audio_num_codebooks):
            decoder_input = torch.cat(sequence, dim=1)
            output = decoder(decoder_input, prefetched=prefetched)[:, -1]
            hidden_parts.append(output[:1].detach())
            logits = decoder.audio_heads[index - 1](output)
            conditioned = logits[:1].float()
            unconditioned = logits[1:2].float()
            code = sample_top_k(
                unconditioned + (conditioned - unconditioned) * cfg_scale,
                top_k,
                generator,
            ).repeat(2)
            codes.append(code)
            if index < self.config.audio_num_codebooks - 1:
                embedding = self.model.audio_extra_embedding(
                    code + (index - 1) * self.config.audio_vocab_size
                )
                sequence.append(decoder.projection(embedding).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

    def _embed_c0(self, codes: torch.Tensor) -> torch.Tensor:
        if self.config.pruned:
            return self.model.embed_tokens_audio(codes)
        return self.model.embed_tokens(codes + AUDIO_CODE_OFFSET)

    def _embed_audio_frame(self, codes: torch.Tensor) -> torch.Tensor:
        c0 = self._embed_c0(codes[:, 0])
        offsets = (
            torch.arange(self.config.audio_num_codebooks - 1, device=codes.device)
            * self.config.audio_vocab_size
        )
        extra = self.model.audio_extra_embedding(codes[:, 1:] + offsets.unsqueeze(0)).sum(dim=1)
        return ((c0 + extra) * (self.config.audio_num_codebooks**-0.5)).unsqueeze(1)

    def _sample_c0(
        self,
        hidden: torch.Tensor,
        cfg_scale: float,
        top_k: int,
        generator: torch.Generator,
        vocab_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.config.pruned:
            guided = self._guided_c0(self.model.lm_head_pruned(hidden).float(), cfg_scale, top_k)
            code = sample_top_k(guided, top_k, generator)
            stop_token = 0
            offset = 1
        else:
            assert vocab_mask is not None
            logits = self.model.lm_head(hidden).float().masked_fill(vocab_mask, -float("inf"))
            guided = self._guided_c0(logits, cfg_scale, top_k).masked_fill(
                vocab_mask, -float("inf")
            )
            code = sample_top_k(guided, top_k, generator)
            stop_token = SPECIAL_TOKEN_IDS["<|audio_end|>"]
            offset = AUDIO_CODE_OFFSET
        return torch.where(code == stop_token, 0, code - offset), code, stop_token

    def generate(
        self,
        input_ids: torch.Tensor,
        seed: int,
        max_audio_frames: int,
        *,
        compute_dtype: torch.dtype,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        top_k: int = DEFAULT_TOP_K,
    ) -> torch.Tensor:
        prompt_tokens = input_ids.shape[1]
        if prompt_tokens > MAX_PROMPT_TOKENS:
            raise ValueError(
                f"MiniMax Music 3 prompt has {prompt_tokens} tokens; maximum is {MAX_PROMPT_TOKENS}"
            )
        decode_limit = min(int(max_audio_frames), MAX_AUDIO_FRAMES)
        if decode_limit < 1:
            raise ValueError("MiniMax Music 3 requires at least one audio frame")
        if not 1 <= top_k <= C0_VOCAB_SIZE:
            raise ValueError(f"MiniMax Music 3 top_k must be in [1, {C0_VOCAB_SIZE}]")
        embedding = (
            self.model.embed_tokens_prefill if self.config.pruned else self.model.embed_tokens
        )
        device = module_compute_device(embedding)
        input_ids = input_ids.to(device)
        unconditioned = input_ids.clone()
        unconditioned[:, 1:-2] = SPECIAL_TOKEN_IDS["<|audio_cfg|>"]
        text_ids = torch.cat((input_ids, unconditioned), dim=0)
        text_embeds = embedding(text_ids).to(dtype=compute_dtype)
        cache = self.model.allocate_causal_cache(
            2,
            prompt_tokens + decode_limit + 1,
            dtype=compute_dtype,
            fixed=True,
        )
        output, _new = self.model.forward_causal(
            None,
            cache,
            embeds=text_embeds,
        )
        last_hidden = output[:, -1]
        cache_position = prompt_tokens
        generator = torch.Generator(device=device).manual_seed(derive_music_seed(seed, "ar"))
        vocab_mask = None
        if not self.config.pruned:
            vocab_mask = torch.ones(self.config.vocab_size, dtype=torch.bool, device=device)
            vocab_mask[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + C0_VOCAB_SIZE] = False
            vocab_mask[SPECIAL_TOKEN_IDS["<|audio_end|>"]] = False

        hidden_frames: list[torch.Tensor] = []
        pending_code: torch.Tensor | None = None
        pending_event: torch.cuda.Event | None = None
        pending_hidden: torch.Tensor | None = None
        stop_token: int | None = None
        cuda_device = device.type == "cuda"
        decoder = self.model.audio_decoder
        depth_modules = torch.nn.ModuleList((decoder, self.model.audio_extra_embedding))
        graph_enabled = cuda_device
        capture_stream: torch.cuda.Stream | None = None
        graph: torch.cuda.CUDAGraph | None = None
        graph_warmed = False
        depth_hidden_input = torch.empty_like(last_hidden)
        depth_c0_input = torch.empty((last_hidden.shape[0],), dtype=torch.long, device=device)
        depth_c0_embed_input = torch.empty_like(last_hidden)
        depth_codes_output = torch.empty(
            (last_hidden.shape[0], self.config.audio_num_codebooks),
            dtype=torch.long,
            device=device,
        )
        depth_hidden_output = torch.empty(
            (1, last_hidden.shape[-1] * (self.config.audio_num_codebooks - 1)),
            dtype=compute_dtype,
            device=device,
        )

        try:
            for frame_index in range(decode_limit + 1):
                if pending_code is not None:
                    if pending_event is not None:
                        pending_event.synchronize()
                    if int(pending_code.item()) == stop_token:
                        pending_hidden = None
                        break
                    if pending_hidden is not None:
                        hidden_frames.append(pending_hidden)
                        if len(hidden_frames) >= decode_limit:
                            break
                c0, code_or_stop, stop_token = self._sample_c0(
                    last_hidden, cfg_scale, top_k, generator, vocab_mask
                )
                if pending_code is None:
                    pending_code = torch.empty_like(
                        code_or_stop,
                        device="cpu",
                        pin_memory=cuda_device,
                    )
                    if cuda_device:
                        pending_event = torch.cuda.Event()
                pending_code.copy_(code_or_stop, non_blocking=cuda_device)
                if pending_event is not None:
                    pending_event.record()
                c0 = c0.repeat(2)
                c0_embed = self._embed_c0(c0)
                feedback_codes = depth_codes_output
                depth_hidden = depth_hidden_output
                if graph_enabled:
                    depth_hidden_input.copy_(last_hidden)
                    depth_c0_input.copy_(c0)
                    depth_c0_embed_input.copy_(c0_embed)
                    prefetch = make_prefetch_queue((depth_modules,))
                    if prefetch is None:
                        graph_enabled = False
                    else:
                        try:
                            fully_prefetched = prefetch_queue_pop(prefetch, depth_modules)
                            if fully_prefetched:

                                def depth_core() -> None:
                                    codes, hidden = self._depth_codes(
                                        depth_hidden_input,
                                        depth_c0_input,
                                        depth_c0_embed_input,
                                        generator,
                                        cfg_scale,
                                        top_k,
                                        prefetched=True,
                                    )
                                    depth_codes_output.copy_(codes)
                                    depth_hidden_output.copy_(hidden)

                                if capture_stream is None:
                                    capture_stream = torch.cuda.Stream(device=device)
                                current_stream = torch.cuda.current_stream(device)
                                if graph is not None:
                                    graph.replay()
                                elif graph_warmed:
                                    graph = torch.cuda.CUDAGraph()
                                    graph.register_generator_state(generator)
                                    capture_stream.wait_stream(current_stream)
                                    with torch.cuda.graph(
                                        graph,
                                        stream=capture_stream,
                                        capture_error_mode="thread_local",
                                    ):
                                        depth_core()
                                    current_stream.wait_stream(capture_stream)
                                    graph.replay()
                                else:
                                    capture_stream.wait_stream(current_stream)
                                    with torch.cuda.stream(capture_stream):
                                        depth_core()
                                    current_stream.wait_stream(capture_stream)
                                    graph_warmed = True
                                feedback_codes = depth_codes_output
                                depth_hidden = depth_hidden_output
                            else:
                                graph_enabled = False
                        finally:
                            close_prefetch_queue(prefetch)
                if not graph_enabled:
                    feedback_codes, depth_hidden = self._depth_codes(
                        last_hidden, c0, c0_embed, generator, cfg_scale, top_k
                    )
                if frame_index > 0:
                    pending_hidden = torch.cat((last_hidden[:1].detach(), depth_hidden), dim=-1)[
                        0
                    ].clone()
                feedback = self._embed_audio_frame(feedback_codes).to(dtype=compute_dtype)
                output, _new = self.model.forward_causal(
                    None,
                    cache,
                    cache_position=cache_position,
                    embeds=feedback,
                )
                last_hidden = output[:, -1]
                cache_position += 1

            if pending_hidden is not None and len(hidden_frames) < decode_limit:
                if pending_event is not None:
                    pending_event.synchronize()
                assert pending_code is not None and stop_token is not None
                if int(pending_code.item()) != stop_token:
                    hidden_frames.append(pending_hidden)
            if not hidden_frames:
                raise ValueError("MiniMax Music 3 generated zero audio frames")
            return torch.stack(hidden_frames).cpu()
        finally:
            if graph is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    graph.reset()


def tokenize_music_prompt(tokenizer: Tokenizer, caption: str, lyrics: str) -> Sequence[int]:
    return tokenizer.encode(build_music_prompt(caption, lyrics), add_special_tokens=False).ids


__all__ = [
    "MiniMaxMusic3TextModel",
    "sample_top_k",
    "tokenize_music_prompt",
]
