"""LTX text recipe loading over verified, ordered source descriptors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import BinaryIO, cast

import torch
from dinkster_inference import (
    AttentionPolicy,
    AttentionRouteToken,
    Conditioning,
    ConditioningCarrier,
)
from dinkster_inference.component_registry import execution_symbol
from dinkster_inference.gemma_text import GemmaTextConfig, tokenize_ltx_gemma_prompt
from dinkster_inference.ltx_text_recipe import LTX_GEMMA_TOKENIZER_KEYS
from dinkster_inference.sources import SafetensorsSource
from dinkster_inference.text_recipes import TextRecipeBinding

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .attention import AttentionRole, AttentionStatus, resolve_role_attention
from .clip_text import EmbeddingLookup
from .gemma_text import GemmaTextModel, LtxDualTextProjection, LtxGemmaTextEncoder
from .gemma_tokenizer import GemmaJsonTokenizer, GemmaSentencePieceTokenizer
from .ltx_connector import LtxTextConnectors
from .ltxav_runtime import _text_carrier  # pyright: ignore[reportPrivateUsage]
from .operations import Operations
from .quant_linear import Fp8Linear, Int8Linear, Nvfp4Linear
from .text_recipes import LoadedTextRecipe


@dataclass(frozen=True)
class LoadedLtxTextRecipe(LoadedTextRecipe):
    tokenizer_model: bytes


def assemble_ltx_text_recipe(
    binding: TextRecipeBinding,
    *,
    compute_dtype: torch.dtype,
    sources: tuple[SafetensorsSource, ...],
    source_files: tuple[BinaryIO, ...],
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    attention_backends: tuple[tuple[str, AttentionRole], ...],
) -> LoadedLtxTextRecipe:
    """Strict-load every bound component using the caller's already verified open files."""
    if len(sources) != len(source_files) or {
        part.source_index for part in binding.components
    } != set(range(len(sources))):
        raise ValueError("LTX text assembly requires every ordered source and its open file")
    gemma_role = binding.composition_roles[0]
    gemma_part = next(part for part in binding.components if part.role == gemma_role)
    gemma_config = cast(GemmaTextConfig, gemma_part.plan.config)
    in_features = gemma_config.hidden_size * (gemma_config.num_hidden_layers + 1)
    tokenizer_model = sources[gemma_part.source_index].read_uint8_configuration_from_file(
        source_files[gemma_part.source_index],
        LTX_GEMMA_TOKENIZER_KEYS[gemma_role],
        limit=(64 if gemma_role == "gemma4_12b" else 8) * 1024 * 1024,
    )
    modules: dict[str, torch.nn.Module] = {}
    statuses: list[AttentionStatus] = []
    backends = dict(attention_backends)
    for part in binding.components:
        source = sources[part.source_index]
        builder: Callable[..., torch.nn.Module]
        if part.role == gemma_role:
            attention = resolve_role_attention(
                backends[part.role], attention_policy, attention_route_token
            )
            builder = partial(GemmaTextModel, attention_kernel=attention.kernel)
            statuses.append(attention.status)
        elif part.role == "connectors":
            attention = resolve_role_attention(
                backends[part.role], attention_policy, attention_route_token
            )
            builder = partial(LtxTextConnectors, attention_kernel=attention.kernel)
            statuses.append(attention.status)
        elif part.role == "text_projection":
            if part.plan.config == "single_linear":
                out_features = source.entry(part.plan.keys["weight"]).geometry.shape[0]

                def single_projection(
                    _config: object, *, operations: Operations, out_features: int = out_features
                ) -> torch.nn.Module:
                    return operations.linear(in_features, out_features, bias=False)

                builder = single_projection
            else:
                video_features = source.entry(
                    part.plan.keys["video_aggregate_embed.weight"]
                ).geometry.shape[0]
                audio_features = source.entry(
                    part.plan.keys["audio_aggregate_embed.weight"]
                ).geometry.shape[0]

                def dual_projection(
                    _config: object,
                    *,
                    operations: Operations,
                    video_features: int = video_features,
                    audio_features: int = audio_features,
                ) -> torch.nn.Module:
                    return LtxDualTextProjection(
                        in_features, video_features, audio_features, operations=operations
                    )

                builder = dual_projection
        else:
            raise ValueError(f"LTX text assembly cannot consume component {part.role!r}")
        module = _load_component(
            part.plan,
            builder,
            compute_dtype=compute_dtype,
            fp8_matmul=False,
            source=source,
            source_file=source_files[part.source_index],
        )
        if part.role == gemma_role:
            for layer in module.modules():
                if isinstance(layer, Fp8Linear | Int8Linear | Nvfp4Linear):
                    layer.compute_dtype = torch.float32
                    layer.full_precision_matmul = True
        modules[part.role] = module
    return LoadedLtxTextRecipe(
        binding, torch.nn.ModuleDict(modules), tuple(statuses), tokenizer_model
    )


class LtxTextRecipeRuntime:
    """Raw prompts through the detected Gemma stack and its LTX projection contract."""

    def __init__(
        self,
        loaded: LoadedLtxTextRecipe,
        *,
        embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    ) -> None:
        if embedding_lookups:
            raise ValueError("LTX raw-prompt encoding does not consume embedding lookups")
        self.loaded = loaded
        self._gemma_role = loaded.binding.composition_roles[0]
        gemma = cast(GemmaTextModel, loaded.module[self._gemma_role])
        projection = cast(torch.nn.Linear | LtxDualTextProjection, loaded.module["text_projection"])
        connectors = (
            cast(LtxTextConnectors, loaded.module["connectors"])
            if "connectors" in loaded.module
            else None
        )
        self._tokenizer = (
            GemmaJsonTokenizer(loaded.tokenizer_model)
            if self._gemma_role == "gemma4_12b"
            else GemmaSentencePieceTokenizer(loaded.tokenizer_model)
        )
        self._config = gemma.config
        assert loaded.binding.composer is not None
        self._encoder: LtxGemmaTextEncoder = execution_symbol(loaded.binding.composer)(
            gemma, projection, self._tokenizer, connectors=connectors
        )
        self._text_dim = (
            projection.video_aggregate_embed.out_features
            + projection.audio_aggregate_embed.out_features
            if isinstance(projection, LtxDualTextProjection)
            else projection.out_features * (2 if connectors is not None else 1)
        )

    def encode_text(
        self,
        text: str,
        *,
        hidden_layer: int | None = None,
        min_padding: int | None = None,
        min_length: int | None = None,
    ) -> Conditioning[torch.Tensor]:
        if any(value is not None for value in (hidden_layer, min_padding, min_length)):
            raise ValueError("LTX text recipes do not consume CLIP/T5 encoding overrides")
        tokens = tokenize_ltx_gemma_prompt(
            text, encode=self._tokenizer.encode, apply_template=False, config=self._config
        )
        return self._encoder.encode_tokens(tokens)

    def text_conditioning_carrier(self, value: Conditioning[torch.Tensor]) -> ConditioningCarrier:
        return _text_carrier(
            value,
            text_dim=self._text_dim,
            family_id=self.loaded.binding.family_id,
            text_stream=self._gemma_role,
        )
