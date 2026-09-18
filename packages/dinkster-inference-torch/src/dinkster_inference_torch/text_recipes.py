"""Native text recipe assembly and encoding over independently planned sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, BinaryIO

import torch
from dinkster_inference import AttentionPolicy, AttentionRouteToken
from dinkster_inference.clip_bpe import load_clip_bpe
from dinkster_inference.clip_text import ClipTextConfig
from dinkster_inference.component_registry import execution_symbol
from dinkster_inference.prompt_tokens import PromptTokenizer
from dinkster_inference.sources import SafetensorsSource
from dinkster_inference.t5_spm import load_t5_spm
from dinkster_inference.t5_text import T5Config
from dinkster_inference.text_encoders import Conditioning
from dinkster_inference.text_recipes import TextRecipeBinding

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .attention import AttentionStatus, resolve_role_attention
from .clip_text import ClipEncodePolicy, ClipTextEncoder, ClipTextModel, EmbeddingLookup
from .conditioning_adapters import basic_conditioning_to_carrier
from .t5_text import T5TextEncoder, T5TextModel


@dataclass(frozen=True)
class LoadedTextRecipe:
    binding: TextRecipeBinding
    module: torch.nn.ModuleDict
    attention_status: tuple[AttentionStatus, ...]

    @property
    def identity_components(self) -> tuple[Any, ...]:
        return tuple(part.plan for part in self.binding.components)


def assemble_text_recipe(
    binding: TextRecipeBinding,
    *,
    compute_dtype: torch.dtype,
    sources: tuple[SafetensorsSource, ...],
    source_files: tuple[BinaryIO, ...],
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> LoadedTextRecipe:
    """Materialize a verified binding using its ordered, already opened source files."""
    if len(sources) != len(source_files) or {
        part.source_index for part in binding.components
    } != set(range(len(sources))):
        raise ValueError("text assembly requires every ordered source and its open file")
    modules: dict[str, torch.nn.Module] = {}
    statuses: list[AttentionStatus] = []
    for part in binding.components:
        if isinstance(part.plan.config, ClipTextConfig):
            attention = resolve_role_attention("clip", attention_policy, attention_route_token)
            builder = partial(ClipTextModel, attention_kernel=attention.kernel)
        elif isinstance(part.plan.config, T5Config):
            attention = resolve_role_attention("t5", attention_policy, attention_route_token)
            builder = partial(T5TextModel, attention_kernel=attention.kernel)
        else:
            raise ValueError(f"text assembly has no builder for {type(part.plan.config).__name__}")
        modules[part.role] = _load_component(
            part.plan,
            builder,
            compute_dtype=compute_dtype,
            fp8_matmul=False,
            source=sources[part.source_index],
            source_file=source_files[part.source_index],
        )
        statuses.append(attention.status)
    return LoadedTextRecipe(binding, torch.nn.ModuleDict(modules), tuple(statuses))


class TextRecipeRuntime:
    """Encode using explicit profiles and composition, never a diffusion-family switch."""

    text_conditioning_carrier = staticmethod(basic_conditioning_to_carrier)

    def __init__(
        self,
        loaded: LoadedTextRecipe,
        *,
        embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    ) -> None:
        self.loaded = loaded
        self._compose = (
            None if loaded.binding.composer is None else execution_symbol(loaded.binding.composer)
        )
        self._encoders: dict[str, ClipTextEncoder | T5TextEncoder] = {}
        self._tokenizers: dict[str, PromptTokenizer] = {}
        for part in loaded.binding.components:
            model = loaded.module[part.role]
            lookup = (embedding_lookups or {}).get(part.role)
            profile = part.profile
            if profile is None:
                raise ValueError("CLIP/T5 text runtime requires a tokenizer packing profile")
            if isinstance(model, ClipTextModel):
                encoder = ClipTextEncoder(
                    model,
                    profile=profile.tokenizer,
                    policy=ClipEncodePolicy(
                        profile.hidden_layer,
                        profile.layer_norm_hidden_state,
                        profile.projected_pooled,
                    ),
                    embeddings=lookup,
                )
                tokenize = load_clip_bpe().encode
            elif isinstance(model, T5TextModel):
                encoder = T5TextEncoder(
                    model,
                    profile=profile.tokenizer,
                    attention_masked=profile.attention_masked,
                    zero_out_masked=profile.zero_out_masked,
                    embeddings=lookup,
                )
                tokenize = load_t5_spm().encode
            else:
                raise ValueError(f"text recipe has no encoder for {type(model).__name__}")
            self._encoders[part.role] = encoder
            self._tokenizers[part.role] = PromptTokenizer(
                encode_word=tokenize,
                resolve=None if lookup is None else partial(_embedding_rows, lookup),
            )

    def encode_text(
        self,
        text: str,
        *,
        hidden_layer: int | None = None,
        min_padding: int | None = None,
        min_length: int | None = None,
    ) -> Conditioning[torch.Tensor]:
        outputs: dict[str, Conditioning[torch.Tensor]] = {}
        for role, encoder in self._encoders.items():
            spans = self._tokenizers[role].tokenize(text)
            outputs[role] = (
                encoder.encode(spans, hidden_layer=hidden_layer)
                if isinstance(encoder, ClipTextEncoder)
                else encoder.encode(spans, min_padding=min_padding, min_length=min_length)
            )
        roles = self.loaded.binding.composition_roles
        if self._compose is None:
            return outputs[roles[0]]
        return self._compose(*(outputs[role] for role in roles))


def _embedding_rows(lookup: EmbeddingLookup, name: str) -> int | None:
    vectors = lookup(name)
    return None if vectors is None else vectors.shape[0]
