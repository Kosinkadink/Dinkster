"""Original Hunyuan Video Llama3 and raw CLIP-L pooled text conditioning.

Template, weighting, mask and structural crop follow comfy/text_encoders/
hunyuan_video.py and comfy/sd1_clip.py at 25dfc16f9ac0a87991d34fbf5f02d6c25c844639.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import BinaryIO

import torch
from dinkster_inference import (
    AttentionPolicy,
    AttentionRouteToken,
    Conditioning,
    ConditioningCarrier,
    ConditioningSet,
    PayloadReference,
    make_conditioning_carrier,
)
from dinkster_inference.clip_bpe import load_clip_bpe
from dinkster_inference.clip_text import ClipTextConfig
from dinkster_inference.component_registry import execution_symbol
from dinkster_inference.llama3_text import LLAMA3_TOKENIZER_SHA256
from dinkster_inference.prompt_tokens import (
    Chunk,
    EmbeddingSlot,
    PackedToken,
    PromptTokenizer,
    TokenizerProfile,
    empty_chunk,
    pack_spans,
)
from dinkster_inference.qwen_text import QwenTextConfig
from dinkster_inference.sources import SafetensorsSource
from dinkster_inference.text_recipes import TextRecipeBinding
from dinkster_inference.vendored import read_vendored
from tokenizers import Tokenizer

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .attention import AttentionStatus, resolve_role_attention
from .clip_text import (
    ClipEncodePolicy,
    ClipTextEncoder,
    ClipTextModel,
    EmbeddingLookup,
    apply_span_weights,
)
from .conditioning_adapters import basic_conditioning_to_carrier
from .operations import bound_compute_device
from .payloads import tensor_to_payload_binding
from .qwen_text import QwenTextModel
from .text_recipes import LoadedTextRecipe

ATTENTION_MASK_METADATA = "dinkster.hunyuan_video/attention_mask"


def assemble_hunyuan_video_text(
    binding: TextRecipeBinding,
    *,
    compute_dtype: torch.dtype,
    sources: tuple[SafetensorsSource, ...],
    source_files: tuple[BinaryIO, ...],
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> LoadedTextRecipe:
    if len(sources) != len(source_files) or {
        part.source_index for part in binding.components
    } != set(range(len(sources))):
        raise ValueError("Hunyuan text assembly requires every ordered source and open file")
    modules: dict[str, torch.nn.Module] = {}
    statuses: list[AttentionStatus] = []
    for part in binding.components:
        if isinstance(part.plan.config, ClipTextConfig):
            attention = resolve_role_attention("clip", attention_policy, attention_route_token)
            builder = partial(ClipTextModel, attention_kernel=attention.kernel)
        elif isinstance(part.plan.config, QwenTextConfig):
            attention = resolve_role_attention("qwen", attention_policy, attention_route_token)
            builder = partial(QwenTextModel, attention_kernel=attention.kernel)
        else:
            raise ValueError("Hunyuan text assembly requires CLIP-L and Llama configs")
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


def hunyuan_crop(tokens: Sequence[PackedToken]) -> slice:
    """Find the last user header and its first EOT in expanded token positions."""
    start, end = 0, len(tokens)
    for index, token in enumerate(tokens):
        if token.unit == 128006 and index + 2 < len(tokens):
            if tokens[index + 1].unit == 882 and tokens[index + 2].unit == 128007:
                start, end = index + 2, -1
        if token.unit == 128009 and end == -1:
            end = index + 1
    if len(tokens) > start + 2 and tokens[start + 1].unit == 271:
        start += 2
    return slice(start, end)


@dataclass(frozen=True)
class LlamaTextEncoding:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor


class HunyuanLlamaEncoder:
    def __init__(self, model: QwenTextModel, embeddings: EmbeddingLookup | None = None) -> None:
        self.model = model
        self.embeddings = embeddings
        self.profile = TokenizerProfile(
            max_length=99999999,
            start_token=128000,
            end_token=None,
            pad_token=model.config.pad_token_id,
            pad_to_max_length=False,
            min_length=model.config.min_tokens,
            empty_has_end=False,
        )
        self.bpe = Tokenizer.from_str(
            read_vendored("llama3_tokenizer.json.gz", LLAMA3_TOKENIZER_SHA256).decode("utf-8")
        )
        self.tokenizer = PromptTokenizer(
            encode_word=self._encode_word,
            resolve=None if embeddings is None else self._rows,
        )

    def _encode_word(self, text: str) -> list[int]:
        return self.bpe.encode(text, add_special_tokens=False).ids

    def _rows(self, name: str) -> int | None:
        vectors = None if self.embeddings is None else self.embeddings(name)
        return None if vectors is None else vectors.shape[0]

    def tokenize(
        self, text: str, *, min_length: int | None = None, min_padding: int | None = None
    ) -> Chunk:
        profile = replace(
            self.profile,
            min_length=self.profile.min_length if min_length is None else min_length,
            min_padding=min_padding,
        )
        spans = self.tokenizer.tokenize(self.model.config.prompt_template.format(text))
        return pack_spans(spans, profile, resolve=self._rows)[0]

    def encode(
        self,
        text: str,
        *,
        hidden_layer: int | None = None,
        min_length: int | None = None,
        min_padding: int | None = None,
    ) -> LlamaTextEncoding:
        tokens = self.tokenize(text, min_length=min_length, min_padding=min_padding)
        has_weights = any(token.weight != 1.0 for token in tokens)
        chunks = (tokens, empty_chunk(self.profile, len(tokens))) if has_weights else (tokens,)
        device = (
            bound_compute_device(self.model.embed_tokens) or self.model.embed_tokens.weight.device
        )
        ids = torch.tensor(
            [[token.unit if isinstance(token.unit, int) else 0 for token in row] for row in chunks],
            device=device,
            dtype=torch.long,
        )
        mask_rows: list[list[int]] = []
        embeds = self.model.embed_tokens(ids)
        for row_index, row in enumerate(chunks):
            mask: list[int] = []
            ended = False
            for index, token in enumerate(row):
                if isinstance(token.unit, EmbeddingSlot):
                    vectors = None if self.embeddings is None else self.embeddings(token.unit.name)
                    if vectors is None or vectors.ndim != 2 or vectors.shape[1] != embeds.shape[-1]:
                        raise ValueError("Llama embedding must have (rows, hidden_size) shape")
                    embeds[row_index, index] = vectors[token.unit.row].to(embeds)
                    mask.append(1)
                else:
                    ended = ended or token.unit == self.profile.pad_token
                    mask.append(0 if ended else 1)
            mask_rows.append(mask)
        attention_mask = torch.tensor(mask_rows, device=device, dtype=torch.long)
        hidden = self.model(None, attention_mask, embeds=embeds, hidden_layer=hidden_layer).float()
        output = hidden[:1]
        if has_weights:
            weights = torch.tensor(
                [token.weight for token in tokens], device=hidden.device, dtype=hidden.dtype
            ).view(1, -1, 1)
            output = apply_span_weights(output, weights, hidden[-1])
        crop = hunyuan_crop(tokens)
        return LlamaTextEncoding(output[:, crop].clone(), attention_mask[:1, crop].clone())


def compose_hunyuan_video_conditioning(
    llama: LlamaTextEncoding, clip_l: Conditioning[torch.Tensor]
) -> ConditioningCarrier:
    if clip_l.pooled is None:
        raise ValueError("Hunyuan Video requires raw CLIP-L pooled output")
    carrier = basic_conditioning_to_carrier(Conditioning(llama.embeddings, clip_l.pooled))
    if bool(torch.all(llama.attention_mask == 1)):
        return carrier
    mask = tensor_to_payload_binding(
        "attention_mask", llama.attention_mask, space="conditioning-attention-mask"
    )
    record = replace(
        carrier.conditioning.records[0],
        extension_metadata=((ATTENTION_MASK_METADATA, PayloadReference(mask.reference_id)),),
    )
    return make_conditioning_carrier(ConditioningSet((record,)), (*carrier.bindings, mask))


class HunyuanVideoTextRuntime:
    def __init__(
        self,
        loaded: LoadedTextRecipe,
        *,
        embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    ) -> None:
        self.loaded = loaded
        llama, clip = loaded.module["llama"], loaded.module["clip_l"]
        if not isinstance(llama, QwenTextModel) or not isinstance(clip, ClipTextModel):
            raise ValueError("Hunyuan Video requires Llama and CLIP-L modules")
        lookups = embedding_lookups or {}
        self.llama = HunyuanLlamaEncoder(llama, lookups.get("llama"))
        clip_part = next(part for part in loaded.binding.components if part.role == "clip_l")
        if clip_part.profile is None:
            raise ValueError("CLIP-L requires its packing profile")
        self.clip = ClipTextEncoder(
            clip,
            profile=clip_part.profile.tokenizer,
            policy=ClipEncodePolicy(projected_pooled=False),
            embeddings=lookups.get("clip_l"),
        )
        self.clip_tokenizer = PromptTokenizer(
            encode_word=load_clip_bpe().encode,
            resolve=None if "clip_l" not in lookups else self._clip_rows,
        )
        assert loaded.binding.composer is not None
        self._compose = execution_symbol(loaded.binding.composer)

    def _clip_rows(self, name: str) -> int | None:
        vectors = None if self.clip.embeddings is None else self.clip.embeddings(name)
        return None if vectors is None else vectors.shape[0]

    @staticmethod
    def text_conditioning_carrier(value: ConditioningCarrier) -> ConditioningCarrier:
        return value

    def encode_text(
        self,
        text: str,
        *,
        hidden_layer: int | None = None,
        min_padding: int | None = None,
        min_length: int | None = None,
    ) -> ConditioningCarrier:
        llama = self.llama.encode(
            text, hidden_layer=hidden_layer, min_padding=min_padding, min_length=min_length
        )
        clip = self.clip.encode(self.clip_tokenizer.tokenize(text), hidden_layer=hidden_layer)
        return self._compose(llama, clip)
