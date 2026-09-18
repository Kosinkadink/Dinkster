"""SD1/SDXL CLIP text model: the native torch architecture + encoding.

Faithful port of the reference text model (comfy/clip_model.py
CLIPAttention/CLIPMLP/CLIPLayer/CLIPEncoder/CLIPEmbeddings/
CLIPTextModel_ /CLIPTextModel @ b78cec87) and the encoding policy
around it (comfy/sd1_clip.py ClipTokenWeightEncoder/SDClipModel,
comfy/sdxl_clip.py SDXLClipModel), constructed from a torch-free
:class:`~dinkster_inference.clip_text.ClipTextConfig` through the typed
:class:`~dinkster_inference_torch.operations.Operations` seam. State-dict
keys are IDENTICAL to the reference: text_model.embeddings.*,
text_model.encoder.layers.N.*, text_model.final_layer_norm.*,
text_projection.weight.

Scope pins (everything else is ledgered in ROADMAP "Native
inference", never silently dropped):

- Causal attention only, via PyTorch SDPA ``is_causal=True`` - the
  additive causal mask the reference always builds. The
  attention-mask input mode (``enable_attention_masks``) and
  ``zero_out_masked`` are OFF for every supported SD1/SDXL text
  encoder and are not ported.
- One selected hidden layer (``intermediate_output`` as an int,
  negative indices resolved like the reference); the ``"all"`` /
  list-of-layers stacking modes have no SD1/SDXL consumer.
- No left-padded profiles (``pad_left``): neither CLIP profile pads
  left, and the reference's left-pad masking interacts with the
  unported attention-mask mode.
- Textual-inversion vectors whose width mismatches the model refuse
  with :class:`ClipEncodeError` - a deliberate divergence from the
  reference, which warns, drops the embedding, and pads the chunk
  (comfy/sd1_clip.py process_tokens); silent conditioning drift is a
  worse failure mode than a loud error.

Deliberate preservation: no ``inference_mode``/``no_grad`` anywhere
(training program, docs/native-inference-plan.md 3.1); residual adds
are out-of-place so autograd and torch.compile see plain dataflow.

Like the reference SDClipModel (which forwards at fixed
``dtype=torch.float32`` and floats its outputs), the ENCODER casts
conditioning and pooled results to float32; the bare model runs in
whatever dtype its parameters hold.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import torch
from dinkster_inference.clip_text import ClipTextConfig
from dinkster_inference.prompt_tokens import (
    CLIP_L_PROFILE,
    Chunk,
    EmbeddingSlot,
    TokenizerProfile,
    empty_chunk,
    pack_spans,
)
from dinkster_inference.text_encoders import Conditioning, WeightedSpan

from ._conditioning_layout import declare_text_conditioning, declared_token_count
from .attention import AttentionKernel, select_attention
from .operations import INITLESS, Operations, bound_compute_device

_DEFAULT_CLIP_ATTENTION = select_attention("clip").kernel

__all__ = [
    "ClipAttention",
    "ClipEmbeddings",
    "ClipEncodeError",
    "ClipEncodePolicy",
    "ClipEncoder",
    "ClipLayer",
    "ClipMlp",
    "ClipTextEncoder",
    "ClipTextModel",
    "ClipTextOutput",
    "ClipTextTransformer",
    "EmbeddingLookup",
    "SD1_CLIP_L_POLICY",
    "SDXL_CLIP_POLICY",
    "apply_span_weights",
    "compose_sdxl_conditioning",
]


class ClipEncodeError(ValueError):
    """Encoding could not proceed; the message names why."""


def _quick_gelu(x: torch.Tensor) -> torch.Tensor:
    # comfy/clip_model.py ACTIVATIONS["quick_gelu"] @ b78cec87.
    return x * torch.sigmoid(1.702 * x)


#: Resolved at construction time (bind-time, compile discipline);
#: keys mirror the reference ACTIVATIONS table for supported models.
_ACTIVATIONS: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "quick_gelu": _quick_gelu,
    "gelu": torch.nn.functional.gelu,
}


class ClipAttention(torch.nn.Module):
    """Reference CLIPAttention @ b78cec87: separate q/k/v/out
    projections, multi-head SDPA. Always causal - the reference
    builds the additive causal mask unconditionally for text."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        embed_dim: int,
        heads: int,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_CLIP_ATTENTION,
    ) -> None:
        super().__init__()
        if embed_dim % heads:
            raise ValueError(f"embed_dim {embed_dim} is not divisible by heads {heads}")
        self.heads = heads
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        self.q_proj = operations.linear(embed_dim, embed_dim, bias=True)
        self.k_proj = operations.linear(embed_dim, embed_dim, bias=True)
        self.v_proj = operations.linear(embed_dim, embed_dim, bias=True)
        self.out_proj = operations.linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, width = x.shape

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq, self.heads, -1).transpose(1, 2)

        out = self._attention_kernel(
            split(self.q_proj(x)),
            split(self.k_proj(x)),
            split(self.v_proj(x)),
            causal=True,
        )
        return self.out_proj(out.transpose(1, 2).reshape(batch, seq, width))


class ClipMlp(torch.nn.Module):
    """Reference CLIPMLP @ b78cec87; activation fixed at bind time."""

    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        activation: str,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.fc1 = operations.linear(embed_dim, intermediate_size, bias=True)
        self.activation = _ACTIVATIONS[activation]
        self.fc2 = operations.linear(intermediate_size, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class ClipLayer(torch.nn.Module):
    """Reference CLIPLayer @ b78cec87: pre-norm attention and MLP
    residual blocks."""

    def __init__(
        self,
        config: ClipTextConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_CLIP_ATTENTION,
    ) -> None:
        super().__init__()
        self.layer_norm1 = operations.layer_norm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = ClipAttention(
            config.hidden_size,
            config.num_attention_heads,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.layer_norm2 = operations.layer_norm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = ClipMlp(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            operations=operations,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class ClipEncoder(torch.nn.Module):
    """Reference CLIPEncoder @ b78cec87, restricted to the int
    ``intermediate_output`` mode SD1/SDXL use."""

    def __init__(
        self,
        config: ClipTextConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_CLIP_ATTENTION,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            ClipLayer(config, operations=operations, attention_kernel=attention_kernel)
            for _ in range(config.num_hidden_layers)
        )

    def forward(
        self, x: torch.Tensor, hidden_layer: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (final hidden state, the state captured after layer
        ``hidden_layer`` or None). Negative indices count from the
        end like the reference (-2 is the penultimate layer)."""
        capture: int | None = None
        if hidden_layer is not None:
            capture = hidden_layer + len(self.layers) if hidden_layer < 0 else hidden_layer
            if not 0 <= capture < len(self.layers):
                raise ValueError(
                    f"hidden layer {hidden_layer} is out of range for {len(self.layers)} layers"
                )
        intermediate: torch.Tensor | None = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == capture:
                intermediate = x.clone()
        return x, intermediate


class ClipEmbeddings(torch.nn.Module):
    """Reference CLIPEmbeddings @ b78cec87 as a parameter container:
    the transformer consumes pre-substituted embedding rows (the
    textual-inversion path), so the reference's token-id forward has
    no caller here. The reference never overrides its vocab_size
    default of 49408 from the config dict; the torch-free config
    carries the value explicitly."""

    def __init__(
        self,
        config: ClipTextConfig,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.token_embedding = operations.embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = operations.embedding(
            config.max_position_embeddings, config.hidden_size
        )


@dataclass(frozen=True)
class ClipTextOutput:
    """Reference CLIPTextModel forward tuple, named. ``hidden`` is the
    selected intermediate layer (None when not requested); ``pooled``
    the raw EOS-position vector; ``projected`` text_projection(pooled)
    - the reference returns both and the encode policy picks."""

    last_hidden: torch.Tensor
    hidden: torch.Tensor | None
    pooled: torch.Tensor
    projected: torch.Tensor


class ClipTextTransformer(torch.nn.Module):
    """Reference CLIPTextModel_ @ b78cec87 with token embedding and
    EOS location moved to the caller: it consumes pre-substituted
    embedding rows (the textual-inversion path) and explicit EOS
    indices instead of re-deriving them from token ids."""

    def __init__(
        self,
        config: ClipTextConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_CLIP_ATTENTION,
    ) -> None:
        super().__init__()
        self.embeddings = ClipEmbeddings(config, operations=operations)
        self.encoder = ClipEncoder(config, operations=operations, attention_kernel=attention_kernel)
        self.final_layer_norm = operations.layer_norm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        embeds: torch.Tensor,
        eos_index: torch.Tensor,
        *,
        hidden_layer: int | None = None,
        layer_norm_hidden_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        table_size = self.embeddings.position_embedding.num_embeddings
        if embeds.shape[1] > table_size:
            raise ValueError(
                f"sequence of {embeds.shape[1]} exceeds the position table of {table_size}"
            )
        positions = self.embeddings.position_embedding(
            torch.arange(embeds.shape[1], device=embeds.device)
        )
        x = embeds + positions.to(dtype=embeds.dtype)
        x, hidden = self.encoder(x, hidden_layer)
        x = self.final_layer_norm(x)
        if hidden is not None and layer_norm_hidden_state:
            hidden = self.final_layer_norm(hidden)
        pooled = x[
            torch.arange(x.shape[0], device=x.device),
            eos_index.to(device=x.device),
        ]
        return x, hidden, pooled


class ClipTextModel(torch.nn.Module):
    """Reference CLIPTextModel @ b78cec87: the transformer plus the
    bias-free hidden->hidden text projection (the config JSON's
    projection_dim is ignored there too). SD1 checkpoints do not ship
    text_projection.weight; load accordingly (strict on the rest)."""

    def __init__(
        self,
        config: ClipTextConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_CLIP_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        self.text_model = ClipTextTransformer(
            config, operations=operations, attention_kernel=attention_kernel
        )
        self.text_projection = operations.linear(config.hidden_size, config.hidden_size, bias=False)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.text_model.embeddings.token_embedding(token_ids)

    def forward(
        self,
        embeds: torch.Tensor,
        eos_index: torch.Tensor,
        *,
        hidden_layer: int | None = None,
        layer_norm_hidden_state: bool = True,
    ) -> ClipTextOutput:
        last, hidden, pooled = self.text_model(
            embeds,
            eos_index,
            hidden_layer=hidden_layer,
            layer_norm_hidden_state=layer_norm_hidden_state,
        )
        return ClipTextOutput(last, hidden, pooled, self.text_projection(pooled))


@dataclass(frozen=True)
class ClipEncodePolicy:
    """Which representation a family reads off the text model - the
    reference SDClipModel constructor knobs (layer/layer_idx,
    layer_norm_hidden_state, return_projected_pooled)."""

    hidden_layer: int | None = None
    layer_norm_hidden_state: bool = True
    projected_pooled: bool = True


#: SD1 checkpoints: final hidden state, RAW pooled (the reference's
#: SD1CheckpointClipModel passes return_projected_pooled=False -
#: text_projection is absent from these checkpoints).
SD1_CLIP_L_POLICY = ClipEncodePolicy(projected_pooled=False)

#: SDXL, both towers: penultimate hidden layer without the final
#: layer norm, projected pooled (only CLIP-G's pooled is consumed).
SDXL_CLIP_POLICY = ClipEncodePolicy(hidden_layer=-2, layer_norm_hidden_state=False)


#: Textual-inversion vectors by name: a (rows, hidden) tensor, or
#: None when unknown. The torch-side counterpart of the torch-free
#: EmbeddingResolver (which only reports row counts).
EmbeddingLookup = Callable[[str], torch.Tensor | None]


def apply_span_weights(
    z: torch.Tensor, weights: torch.Tensor, z_empty: torch.Tensor
) -> torch.Tensor:
    """The reference emphasis interpolation (ClipTokenWeightEncoder
    @ b78cec87): ``(z - z_empty) * weight + z_empty`` per position,
    with weight 1.0 positions returned EXACTLY (the reference skips
    them; float round-trip through the arithmetic would drift)."""
    return torch.where(weights == 1.0, z, (z - z_empty) * weights + z_empty)


def _eos_index(chunk: Chunk, profile: TokenizerProfile) -> int:
    """Where the reference pools: the first non-left-padded token
    equal to the end token (process_tokens' num_tokens - 1), falling
    back to the last position when no EOS exists."""
    cmp = profile.pad_token if profile.end_token is None else profile.end_token
    for i, packed in enumerate(chunk):
        if isinstance(packed.unit, int) and packed.unit == cmp:
            return i
    return len(chunk) - 1


class ClipTextEncoder:
    """The reference SDClipModel.encode_token_weights policy over
    packed chunks: batch the sections (plus one empty-prompt chunk
    when any weight deviates from 1.0), substitute textual-inversion
    rows, forward once, interpolate weighted positions against the
    empty encoding, and concatenate chunks along the sequence axis.
    The pooled vector comes from the FIRST chunk only.

    ``encode`` packs spans with this encoder's profile and encodes the chunks.
    Conditioning and pooled outputs are float32 like the reference.
    """

    def __init__(
        self,
        model: ClipTextModel,
        *,
        profile: TokenizerProfile = CLIP_L_PROFILE,
        policy: ClipEncodePolicy = SD1_CLIP_L_POLICY,
        embeddings: EmbeddingLookup | None = None,
    ) -> None:
        if profile.pad_left:
            raise ClipEncodeError(
                "left-padded profiles are not supported by the CLIP"
                " encoder (ROADMAP: Native inference)"
            )
        self.model = model
        self.profile = profile
        self.policy = policy
        self.embeddings = embeddings

    def _rows(self, name: str) -> int | None:
        if self.embeddings is None:
            return None
        vectors = self.embeddings(name)
        return None if vectors is None else vectors.shape[0]

    def _vectors(self, name: str) -> torch.Tensor:
        vectors = None if self.embeddings is None else self.embeddings(name)
        if vectors is None:
            raise ClipEncodeError(f"embedding {name!r} is not resolvable at encode time")
        hidden = self.model.config.hidden_size
        if vectors.ndim != 2 or vectors.shape[1] != hidden:
            raise ClipEncodeError(
                f"embedding {name!r} has shape {tuple(vectors.shape)};"
                f" expected (rows, {hidden}). The reference drops such"
                " embeddings with a warning; Dinkster refuses instead."
            )
        return vectors

    def encode(
        self,
        spans: Sequence[WeightedSpan],
        *,
        hidden_layer: int | None = None,
    ) -> Conditioning[torch.Tensor]:
        policy = self.policy
        if hidden_layer is not None:
            policy = replace(
                policy,
                hidden_layer=(
                    hidden_layer
                    if abs(hidden_layer) <= self.model.config.num_hidden_layers
                    else None
                ),
            )
        return self._encode_chunks(
            pack_spans(spans, self.profile, resolve=self._rows),
            policy,
        )

    def encode_chunks(self, chunks: Sequence[Chunk]) -> Conditioning[torch.Tensor]:
        return self._encode_chunks(chunks, self.policy)

    def _encode_chunks(
        self,
        chunks: Sequence[Chunk],
        policy: ClipEncodePolicy,
    ) -> Conditioning[torch.Tensor]:
        if not chunks:
            raise ValueError("no chunks to encode")
        length = len(chunks[0])
        if any(len(chunk) != length for chunk in chunks):
            raise ValueError("chunks must share one length")

        sections = len(chunks)
        has_weights = any(packed.weight != 1.0 for chunk in chunks for packed in chunk)
        batch = list(chunks)
        if has_weights:
            batch.append(empty_chunk(self.profile, length))

        weight = self.model.text_model.embeddings.token_embedding.weight
        device = (
            bound_compute_device(self.model.text_model.embeddings.token_embedding) or weight.device
        )
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

        eos = torch.tensor(
            [_eos_index(chunk, self.profile) for chunk in batch],
            dtype=torch.long,
            device=device,
        )
        out = self.model(
            embeds,
            eos,
            hidden_layer=policy.hidden_layer,
            layer_norm_hidden_state=policy.layer_norm_hidden_state,
        )
        z_all = (out.last_hidden if out.hidden is None else out.hidden).float()
        pooled = (out.projected if policy.projected_pooled else out.pooled).float()
        first_pooled = pooled[0:1]

        z = z_all[:sections]
        if has_weights:
            weights = torch.tensor(
                [[packed.weight for packed in chunk] for chunk in chunks],
                dtype=z.dtype,
                device=z.device,
            ).unsqueeze(-1)
            z = apply_span_weights(z, weights, z_all[-1])
        cond = z.reshape(1, sections * length, -1)
        return declare_text_conditioning(
            Conditioning(cond, first_pooled),
            sections * length,
        )


def compose_sdxl_conditioning(
    clip_l: Conditioning[torch.Tensor], clip_g: Conditioning[torch.Tensor]
) -> Conditioning[torch.Tensor]:
    """SDXLClipModel.encode_token_weights @ b78cec87: cut both towers
    to the shorter sequence, concatenate along the feature axis
    (768 + 1280 = 2048), keep CLIP-G's pooled."""
    cut = min(clip_l.embeddings.shape[1], clip_g.embeddings.shape[1])
    conditioning = Conditioning(
        torch.cat([clip_l.embeddings[:, :cut], clip_g.embeddings[:, :cut]], dim=-1),
        clip_g.pooled,
    )
    clip_l_tokens = declared_token_count(clip_l)
    clip_g_tokens = declared_token_count(clip_g)
    if clip_l_tokens is None or clip_g_tokens is None:
        return conditioning
    return declare_text_conditioning(conditioning, min(clip_l_tokens, clip_g_tokens))
