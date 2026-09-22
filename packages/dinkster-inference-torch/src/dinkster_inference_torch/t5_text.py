"""T5 and UMT5 text encoders: native torch architecture + encoding.

Faithful port of the reference T5 encoder (comfy/text_encoders/t5.py
T5LayerNorm/T5Attention/T5LayerSelfAttention/T5DenseGatedActDense/
T5DenseActDense/T5LayerFF/T5Block/T5Stack/T5 @ b78cec87) and the
Flux encoding policy around it (comfy/text_encoders/sd3_clip.py
T5XXLModel, comfy/text_encoders/flux.py FluxClipModel), constructed
from a torch-free :class:`~dinkster_inference.t5_text.T5Config` through
the typed :class:`~dinkster_inference_torch.operations.Operations` seam.
State-dict keys are IDENTICAL to the reference: shared.weight,
encoder.block.N.layer.0.SelfAttention.{q,k,v,o}.weight,
encoder.block.N.layer.0.SelfAttention.relative_attention_bias.weight,
encoder.block.N.layer.{0,1}.layer_norm.weight,
encoder.block.N.layer.1.DenseReluDense.{wi_0,wi_1|wi,wo}.weight,
encoder.final_layer_norm.weight.

Faithfulness notes:

- T5 attention is UNSCALED (Mesh-TF init folds the scale into the
  weights). The reference multiplies k by ``sqrt(head_dim)`` so the
  downstream 1/sqrt(head_dim) attention cancels it; here SDPA runs
  with ``scale=1.0`` directly - bit-identical, since sqrt(64)=8 is a
  power of two and that multiply is exact.
- :class:`T5LayerNorm` keeps the reference's exact RMS math (variance
  in the INPUT dtype, no float32 upcast) instead of
  ``F.rms_norm``, whose accumulation dtype differs in fp16/bf16.
- Only block 0 owns the relative-attention-bias table for classic T5;
  UMT5 owns one table per block. Each owning block computes its bias,
  exactly following the reference's ``past_bias`` replacement.

Scope pins (everything else is ledgered in ROADMAP "Native
inference", never silently dropped):

- Classic T5 preserves the Flux policy of no attention mask. UMT5
  follows Wan's policy: padding is excluded from attention and masked
  output rows are zeroed.
- Per-call hidden-layer selection captures post-block state and applies
  final_layer_norm, matching the normalized intermediate policy in
  ComfyUI 25dfc16f's T5Stack. Omitting it returns the final hidden state.
Deliberate preservation: no ``inference_mode``/``no_grad`` anywhere
(training program, docs/native-inference-plan.md 3.1); residual adds
are out-of-place (the reference mutates in place, mathematically
identical) so autograd and torch.compile see plain dataflow.

Like the reference SDClipModel, the ENCODER casts conditioning to
float32; the bare model runs in whatever dtype its parameters hold.
T5 exposes no pooled output - Flux takes its pooled vector from
CLIP-L (:func:`compose_flux_conditioning`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from dinkster_inference.prompt_tokens import (
    Chunk,
    EmbeddingSlot,
    TokenizerProfile,
    empty_chunk,
    pack_spans,
)
from dinkster_inference.t5_spm import T5_XXL_FLUX_PROFILE
from dinkster_inference.t5_text import UMT5_XXL_WAN_PROFILE, T5Config
from dinkster_inference.text_encoders import Conditioning, WeightedSpan

from ._conditioning_layout import declare_text_conditioning, declared_token_count
from .attention import AttentionKernel, select_attention
from .clip_text import apply_span_weights
from .operations import INITLESS, Operations, bound_compute_device
from .quant import FP8_DTYPES

_DEFAULT_T5_ATTENTION = select_attention("t5").kernel

if TYPE_CHECKING:
    from .module_residency import ResidencyBinding

__all__ = [
    "T5Attention",
    "T5Block",
    "T5DenseActDense",
    "T5DenseGatedActDense",
    "T5EncodeError",
    "T5LayerFF",
    "T5LayerNorm",
    "T5LayerSelfAttention",
    "T5Stack",
    "T5TextEncoder",
    "T5TextModel",
    "compose_flux_conditioning",
    "relative_position_bucket",
]


class T5EncodeError(ValueError):
    """Encoding could not proceed; the message names why."""


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    # comfy/text_encoders/t5.py activations["gelu_pytorch_tanh"]
    # @ b78cec87.
    return torch.nn.functional.gelu(x, approximate="tanh")


#: Resolved at construction time (bind-time, compile discipline);
#: keys mirror the reference activations table.
_ACTIVATIONS: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu_pytorch_tanh": _gelu_tanh,
    "relu": torch.nn.functional.relu,
}


class T5LayerNorm(torch.nn.Module):
    """Reference T5LayerNorm @ b78cec87: RMS norm, weight only, no
    bias, no mean subtraction. The variance is computed in the INPUT
    dtype (the reference does not upcast), so this stays a manual
    implementation rather than ``F.rms_norm``, whose internal
    accumulation dtype would diverge in reduced precision."""

    _residency: ResidencyBinding | None = None

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        # The reference allocates a raw empty Parameter outside its
        # operations seam too; the value always comes from the
        # checkpoint.
        self.weight = torch.nn.Parameter(torch.empty(hidden_size))
        self.variance_epsilon = eps

    def bind_residency(self, binding: ResidencyBinding) -> None:
        """Bind post-assembly routed access without changing RMS math."""
        self._residency = binding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        binding = self._residency
        if binding is not None and not binding.mechanism.is_loaded(binding.unit):
            with binding.lease() as lease:
                return lease.get("weight", dtype=x.dtype) * x
        return self.weight.to(dtype=x.dtype) * x


def relative_position_bucket(
    relative_position: torch.Tensor,
    *,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    """Reference T5Attention._relative_position_bucket @ b78cec87,
    bidirectional (this is an encoder; the reference always passes
    bidirectional=True): sign claims half the buckets, half of each
    side maps exact small offsets, the rest bins logarithmically up
    to ``max_distance``."""
    num_buckets //= 2
    relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
    relative_position = torch.abs(relative_position)

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    return relative_buckets + torch.where(is_small, relative_position, relative_position_if_large)


class T5Attention(torch.nn.Module):
    """Reference T5Attention @ b78cec87: bias-free q/k/v/o
    projections, UNSCALED multi-head SDPA (``scale=1.0``; see module
    docstring), block 0 additionally owning the relative-position
    bias table."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: T5Config,
        *,
        relative_attention_bias: bool,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_T5_ATTENTION,
    ) -> None:
        super().__init__()
        model, inner = config.d_model, config.inner_dim
        self.q = operations.linear(model, inner, bias=False)
        self.k = operations.linear(model, inner, bias=False)
        self.v = operations.linear(model, inner, bias=False)
        self.o = operations.linear(inner, model, bias=False)
        self.num_heads = config.num_heads
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.relative_attention_bias: torch.nn.Embedding | None = None
        if relative_attention_bias:
            self.relative_attention_bias = operations.embedding(
                config.relative_attention_num_buckets, config.num_heads
            )

    def compute_bias(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Reference compute_bias @ b78cec87 for self-attention
        (query_length == key_length): the (1, heads, length, length)
        additive bias later blocks reuse."""
        assert self.relative_attention_bias is not None
        context_position = torch.arange(length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(length, dtype=torch.long, device=device)[None, :]
        bucket = relative_position_bucket(
            memory_position - context_position,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(bucket).to(dtype=dtype)
        return values.permute(2, 0, 1).unsqueeze(0).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        past_bias: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq, _ = x.shape
        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(seq, x.device, x.dtype)
        mask = past_bias if attention_mask is None else attention_mask
        if attention_mask is not None and past_bias is not None:
            mask = attention_mask + past_bias

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq, self.num_heads, -1).transpose(1, 2)

        out = self._attention_kernel(
            split(self.q(x)),
            split(self.k(x)),
            split(self.v(x)),
            mask=mask,
            scale=1.0,
        )
        return (
            self.o(out.transpose(1, 2).reshape(batch, seq, -1)),
            past_bias,
        )


class T5LayerSelfAttention(torch.nn.Module):
    """Reference T5LayerSelfAttention @ b78cec87: pre-norm residual
    attention block."""

    def __init__(
        self,
        config: T5Config,
        *,
        relative_attention_bias: bool,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_T5_ATTENTION,
    ) -> None:
        super().__init__()
        self.SelfAttention = T5Attention(
            config,
            relative_attention_bias=relative_attention_bias,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        past_bias: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        output, past_bias = self.SelfAttention(self.layer_norm(x), past_bias, attention_mask)
        return x + output, past_bias


class T5DenseGatedActDense(torch.nn.Module):
    """Reference T5DenseGatedActDense @ b78cec87: act(wi_0) * wi_1
    -> wo, all bias-free."""

    def __init__(
        self,
        config: T5Config,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        model, ff = config.d_model, config.d_ff
        self.wi_0 = operations.linear(model, ff, bias=False)
        self.wi_1 = operations.linear(model, ff, bias=False)
        self.wo = operations.linear(ff, model, bias=False)
        self.act = _ACTIVATIONS[config.dense_act_fn]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(self.act(self.wi_0(x)) * self.wi_1(x))


class T5DenseActDense(torch.nn.Module):
    """Reference T5DenseActDense @ b78cec87: act(wi) -> wo, both
    bias-free (the non-gated "old" T5 feed-forward)."""

    def __init__(
        self,
        config: T5Config,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.wi = operations.linear(config.d_model, config.d_ff, bias=False)
        self.wo = operations.linear(config.d_ff, config.d_model, bias=False)
        self.act = _ACTIVATIONS[config.dense_act_fn]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(self.act(self.wi(x)))


class T5LayerFF(torch.nn.Module):
    """Reference T5LayerFF @ b78cec87: pre-norm residual
    feed-forward block; gated or plain per the config."""

    def __init__(
        self,
        config: T5Config,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.DenseReluDense: T5DenseGatedActDense | T5DenseActDense = (
            T5DenseGatedActDense(config, operations=operations)
            if config.is_gated_act
            else T5DenseActDense(config, operations=operations)
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.DenseReluDense(self.layer_norm(x))


class T5Block(torch.nn.Module):
    """Reference T5Block @ b78cec87: layer.0 self-attention, layer.1
    feed-forward (the ModuleList indices ARE the state-dict names)."""

    def __init__(
        self,
        config: T5Config,
        *,
        relative_attention_bias: bool,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_T5_ATTENTION,
    ) -> None:
        super().__init__()
        self.layer = torch.nn.ModuleList(
            [
                T5LayerSelfAttention(
                    config,
                    relative_attention_bias=relative_attention_bias,
                    operations=operations,
                    attention_kernel=attention_kernel,
                ),
                T5LayerFF(config, operations=operations),
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        past_bias: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x, past_bias = self.layer[0](x, past_bias, attention_mask)
        return self.layer[1](x), past_bias


class T5Stack(torch.nn.Module):
    """T5 blocks with normalized post-block hidden-state capture."""

    def __init__(
        self,
        config: T5Config,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_T5_ATTENTION,
    ) -> None:
        super().__init__()
        self.block = torch.nn.ModuleList(
            T5Block(
                config,
                relative_attention_bias=(config.model_type == "umt5" or i == 0),
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for i in range(config.num_layers)
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        hidden_layer: int | None = None,
    ) -> torch.Tensor:
        if hidden_layer is not None and type(hidden_layer) is not int:
            raise ValueError("hidden_layer must be an integer or None")
        capture = hidden_layer
        if capture is not None:
            if capture < 0:
                capture += len(self.block)
            if not 0 <= capture < len(self.block):
                raise ValueError(
                    f"hidden layer {hidden_layer} is out of range for {len(self.block)} layers"
                )
        additive_mask: torch.Tensor | None = None
        if attention_mask is not None:
            batch, length = attention_mask.shape
            additive_mask = 1.0 - attention_mask.to(x.dtype).reshape(batch, 1, 1, length)
            additive_mask = additive_mask.expand(batch, 1, length, length)
            additive_mask = additive_mask.masked_fill(
                additive_mask.to(torch.bool), -torch.finfo(x.dtype).max
            )
        past_bias: torch.Tensor | None = None
        intermediate: torch.Tensor | None = None
        for index, block in enumerate(self.block):
            x, past_bias = block(x, past_bias, additive_mask)
            if index == capture:
                intermediate = x.clone()
        final = self.final_layer_norm(x)
        return final if intermediate is None else self.final_layer_norm(intermediate)


class T5TextModel(torch.nn.Module):
    """Reference T5 @ b78cec87 with token embedding moved to the
    caller: it consumes pre-substituted embedding rows (the
    textual-inversion path), so the token-id forward has no caller
    here. ``shared`` is the only embedding module the reference
    constructs; checkpoints' duplicate ``encoder.embed_tokens.weight``
    is dropped at load (the reference loads strict=False)."""

    def __init__(
        self,
        config: T5Config,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_T5_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        self.shared = operations.embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config, operations=operations, attention_kernel=attention_kernel)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.shared(token_ids)
        if self.shared.weight.dtype in FP8_DTYPES:
            # Reference t5.py forward "Fix for fp8 T5 base"
            # @ b78cec87 (gated there on the checkpoint dtype being
            # outside fp32/fp16/bf16): fp8 storage can hold NaN
            # encodings, and one poisoned embedding row would sink the
            # whole encoding. Scrub after dequantization.
            embeds = torch.nan_to_num(embeds)
        return embeds

    def forward(
        self,
        embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        hidden_layer: int | None = None,
    ) -> torch.Tensor:
        return self.encoder(embeds, attention_mask, hidden_layer=hidden_layer)


#: Textual-inversion vectors by name: a (rows, d_model) tensor, or
#: None when unknown (t5xxl-keyed embeddings in the reference).
_EmbeddingLookup = Callable[[str], torch.Tensor | None]


class T5TextEncoder:
    """The reference encode policy over packed chunks for T5/UMT5
    (sd3_clip.T5XXLModel via ClipTokenWeightEncoder @ b78cec87):
    batch the sections (plus one empty-prompt chunk when any weight
    deviates from 1.0), substitute textual-inversion rows, forward
    once, interpolate weighted positions
    against the empty encoding, concatenate chunks along the sequence
    axis. T5 has no pooled output; ``Conditioning.pooled`` is None.

    ``encode`` packs spans with this encoder's profile (Flux: no BOS, EOS 1, pad 0 to
    at least 256, unbounded single chunk) and encodes the chunks.
    Conditioning is float32 like the reference. ``model_type="umt5"``
    defaults to Wan's attention mask and zero-out-masked policy;
    classic T5 defaults to unmasked. ``attention_masked`` and
    ``zero_out_masked`` override those defaults per model: LTXV runs
    classic T5 masked through EOS without zeroing the padded rows
    (lt.LTXVT5XXL @ b78cec87)."""

    def __init__(
        self,
        model: T5TextModel,
        *,
        profile: TokenizerProfile | None = None,
        embeddings: _EmbeddingLookup | None = None,
        attention_masked: bool | None = None,
        zero_out_masked: bool | None = None,
    ) -> None:
        if profile is None:
            profile = (
                UMT5_XXL_WAN_PROFILE if model.config.model_type == "umt5" else T5_XXL_FLUX_PROFILE
            )
        if attention_masked is None:
            attention_masked = model.config.model_type == "umt5"
        if zero_out_masked is None:
            zero_out_masked = model.config.model_type == "umt5"
        if zero_out_masked and not attention_masked:
            raise T5EncodeError("zero_out_masked needs attention_masked; there is no mask to apply")
        if profile.pad_left:
            raise T5EncodeError(
                "left-padded profiles are not supported by the T5"
                " encoder (ROADMAP: Native inference)"
            )
        self.model = model
        self.profile = profile
        self.embeddings = embeddings
        self.attention_masked = attention_masked
        self.zero_out_masked = zero_out_masked

    def _rows(self, name: str) -> int | None:
        if self.embeddings is None:
            return None
        vectors = self.embeddings(name)
        return None if vectors is None else vectors.shape[0]

    def _vectors(self, name: str) -> torch.Tensor:
        vectors = None if self.embeddings is None else self.embeddings(name)
        if vectors is None:
            raise T5EncodeError(f"embedding {name!r} is not resolvable at encode time")
        model_dim = self.model.config.d_model
        if vectors.ndim != 2 or vectors.shape[1] != model_dim:
            raise T5EncodeError(
                f"embedding {name!r} has shape {tuple(vectors.shape)};"
                f" expected (rows, {model_dim}). The reference drops"
                " such embeddings with a warning; Dinkster refuses instead."
            )
        return vectors

    def encode(
        self,
        spans: Sequence[WeightedSpan],
        *,
        hidden_layer: int | None = None,
        min_padding: int | None = None,
        min_length: int | None = None,
    ) -> Conditioning[torch.Tensor]:
        profile = self.profile
        if min_padding is not None:
            profile = replace(profile, min_padding=min_padding)
        if min_length is not None:
            profile = replace(profile, min_length=min_length)
        return self.encode_chunks(
            pack_spans(spans, profile, resolve=self._rows), hidden_layer=hidden_layer
        )

    def encode_chunks(
        self, chunks: Sequence[Chunk], *, hidden_layer: int | None = None
    ) -> Conditioning[torch.Tensor]:
        if not chunks:
            raise ValueError("no chunks to encode")
        length = len(chunks[0])
        if any(len(chunk) != length for chunk in chunks):
            raise ValueError("chunks must share one length")
        if hidden_layer is not None and type(hidden_layer) is not int:
            raise ValueError("hidden_layer must be an integer or None")
        # SDClipModel.set_clip_options resets an oversized override to last.
        if hidden_layer is not None and abs(hidden_layer) > self.model.config.num_layers:
            hidden_layer = None

        sections = len(chunks)
        has_weights = any(packed.weight != 1.0 for chunk in chunks for packed in chunk)
        batch = list(chunks)
        if has_weights:
            batch.append(empty_chunk(self.profile, length))

        weight = self.model.shared.weight
        device = bound_compute_device(self.model.shared) or weight.device
        ids = torch.tensor(
            [
                [packed.unit if isinstance(packed.unit, int) else 0 for packed in chunk]
                for chunk in batch
            ],
            dtype=torch.long,
            device=device,
        )
        embeds = self.model.embed_tokens(ids)
        for row, chunk in enumerate(batch):
            for pos, packed in enumerate(chunk):
                if isinstance(packed.unit, EmbeddingSlot):
                    vectors = self._vectors(packed.unit.name)
                    embeds[row, pos] = vectors[packed.unit.row].to(
                        device=device, dtype=embeds.dtype
                    )

        attention_mask: torch.Tensor | None = None
        if self.attention_masked:
            attention_mask = torch.zeros_like(ids)
            for row, chunk in enumerate(batch):
                eos = False
                for pos, packed in enumerate(chunk):
                    if not eos:
                        attention_mask[row, pos] = 1
                    if isinstance(packed.unit, int) and packed.unit == self.profile.end_token:
                        eos = True

        if hidden_layer is None:
            z_all = (
                self.model(embeds) if attention_mask is None else self.model(embeds, attention_mask)
            ).float()
        else:
            z_all = self.model(embeds, attention_mask, hidden_layer=hidden_layer).float()
        if attention_mask is not None and self.zero_out_masked:
            z_all = z_all * attention_mask.unsqueeze(-1)
        z = z_all[:sections]
        if has_weights:
            weights = torch.tensor(
                [[packed.weight for packed in chunk] for chunk in chunks],
                dtype=z.dtype,
                device=z.device,
            ).unsqueeze(-1)
            z = apply_span_weights(z, weights, z_all[-1])
        token_count = sections * length
        if attention_mask is not None and not self.zero_out_masked:
            # Without zero-out, padded rows keep model output, so a consumer
            # cannot recover the real length from the tensor; declare the
            # through-EOS count so runtimes can rebuild the attention mask.
            # That count only describes a contiguous prefix, so padding
            # before the final chunk's tail is unrepresentable - refuse it.
            flat = attention_mask[:sections].reshape(-1)
            token_count = int(flat.sum().item())
            if int(flat[:token_count].sum().item()) != token_count:
                raise T5EncodeError(
                    "masked encoding without zero-out declares a contiguous"
                    " token prefix; the attention mask has padding before"
                    " the last attended token, which that count cannot"
                    " represent"
                )
        return declare_text_conditioning(
            Conditioning(z.reshape(1, sections * length, -1), None),
            token_count,
        )


def compose_flux_conditioning(
    t5: Conditioning[torch.Tensor], clip_l: Conditioning[torch.Tensor]
) -> Conditioning[torch.Tensor]:
    """FluxClipModel.encode_token_weights @ b78cec87: the T5 sequence
    is the conditioning, CLIP-L's raw pooled vector rides beside it
    (Flux's SDClipModel is constructed return_projected_pooled=False).
    """
    if clip_l.pooled is None:
        raise T5EncodeError(
            "Flux conditioning needs CLIP-L's pooled vector; the CLIP-L encoding carried none"
        )
    conditioning = Conditioning(t5.embeddings, clip_l.pooled)
    token_count = declared_token_count(t5)
    if token_count is None:
        return conditioning
    return declare_text_conditioning(conditioning, token_count)


def compose_flux_t5_conditioning(
    t5: Conditioning[torch.Tensor],
) -> Conditioning[torch.Tensor]:
    """Supply the neutral CLIP-L vector for a Flux recipe containing only T5."""
    pooled = torch.zeros(
        (t5.embeddings.shape[0], 768),
        dtype=t5.embeddings.dtype,
        device=t5.embeddings.device,
    )
    conditioning = Conditioning(t5.embeddings, pooled)
    token_count = declared_token_count(t5)
    if token_count is None:
        return conditioning
    return declare_text_conditioning(conditioning, token_count)
