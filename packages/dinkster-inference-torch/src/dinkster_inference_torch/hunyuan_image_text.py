"""Hunyuan Image Qwen2.5-VL and optional ByT5-small text conditioning."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import BinaryIO

import torch
from dinkster_inference import (
    AttentionPolicy,
    AttentionRouteToken,
    Conditioning,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    PayloadDescriptor,
    PayloadReference,
    make_conditioning_carrier,
)
from dinkster_inference.component_registry import execution_symbol
from dinkster_inference.hunyuan_image_text import (
    BYT5_TOKENIZER_HASHES,
    HUNYUAN_IMAGE_TEMPLATE,
    QWEN_TOKENIZER_CONFIG_SHA256,
)
from dinkster_inference.prompt_tokens import (
    Chunk,
    EmbeddingSlot,
    PromptTokenizer,
    TokenizerProfile,
    pack_spans,
)
from dinkster_inference.qwen_bpe import load_qwen_bpe
from dinkster_inference.qwen_image_text import QwenImageTextConfig
from dinkster_inference.sources import SafetensorsSource
from dinkster_inference.t5_text import T5Config
from dinkster_inference.text_recipes import TextRecipeBinding
from dinkster_inference.vendored import read_vendored

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .attention import AttentionKernel, AttentionStatus, resolve_role_attention
from .clip_text import EmbeddingLookup
from .operations import Operations, bound_compute_device
from .payloads import tensor_to_payload_binding
from .quant_linear import Fp8Linear, Int8Linear, Nvfp4Linear
from .qwen_image_text import QwenImageTextModel
from .t5_text import T5TextEncoder, T5TextModel
from .text_recipes import LoadedTextRecipe

ATTENTION_MASK_METADATA = "dinkster.hunyuan_image/attention_mask"
BYT5_CONDITIONING_METADATA = "dinkster.hunyuan_image/conditioning_byt5small"


def assemble_hunyuan_image_text(
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
        raise ValueError("Hunyuan Image assembly requires every ordered source and open file")
    modules: dict[str, torch.nn.Module] = {}
    statuses: list[AttentionStatus] = []
    for part in binding.components:
        if isinstance(part.plan.config, QwenImageTextConfig):
            attention = resolve_role_attention("qwen", attention_policy, attention_route_token)

            def build_qwen(
                config: QwenImageTextConfig,
                *,
                operations: Operations,
                expected: QwenImageTextConfig = part.plan.config,
                kernel: AttentionKernel = attention.kernel,
            ) -> QwenImageTextModel:
                if config != expected:
                    raise ValueError("Hunyuan Image Qwen plan changed during assembly")
                return QwenImageTextModel(
                    operations=operations,
                    attention_kernel=kernel,
                )

            builder = build_qwen
        elif isinstance(part.plan.config, T5Config):
            attention = resolve_role_attention("t5", attention_policy, attention_route_token)
            builder = partial(T5TextModel, attention_kernel=attention.kernel)
        else:
            raise ValueError("Hunyuan Image assembly requires Qwen and ByT5 configs")
        module = _load_component(
            part.plan,
            builder,
            compute_dtype=compute_dtype,
            fp8_matmul=False,
            source=sources[part.source_index],
            source_file=source_files[part.source_index],
        )
        if part.role == "qwen25_vl":
            for layer in module.modules():
                if isinstance(layer, Fp8Linear | Int8Linear | Nvfp4Linear):
                    layer.compute_dtype = torch.float32
                    layer.full_precision_matmul = True
        modules[part.role] = module
        statuses.append(attention.status)
    return LoadedTextRecipe(binding, torch.nn.ModuleDict(modules), tuple(statuses))


def extract_byt5_prompt(text: str) -> str | None:
    captures: list[str] = []
    for pattern in ('"(.*?)"', r"\u2018(.*?)\u2019", r"\u201c(.*?)\u201d"):
        captures.extend(re.findall(pattern, text))
    return None if not captures else "".join(f'Text "{value}". ' for value in captures)


class ByT5Tokenizer:
    """Hash-pinned byte tokenizer used by the Hunyuan glyph tower."""

    def __init__(self) -> None:
        names = ("added_tokens", "special_tokens_map", "tokenizer_config")
        documents = [
            json.loads(read_vendored(f"byt5_{name}.json.gz", digest))
            for name, digest in zip(names, BYT5_TOKENIZER_HASHES, strict=True)
        ]
        added, special, config = documents
        if (
            added.get("<extra_id_0>") != 259
            or special.get("eos_token", {}).get("content") != "</s>"
            or config.get("tokenizer_class") != "ByT5Tokenizer"
        ):
            raise ValueError("vendored ByT5 tokenizer contract is invalid")
        self._specials = tuple(
            sorted(
                (
                    (value["content"], int(index))
                    for index, value in config["added_tokens_decoder"].items()
                    if value["special"]
                ),
                reverse=True,
            )
        )

    def encode(self, text: str) -> list[int]:
        result: list[int] = []
        cursor = 0
        while cursor < len(text):
            special = next(
                (
                    (token, token_id)
                    for token, token_id in self._specials
                    if text.startswith(token, cursor)
                ),
                None,
            )
            if special is not None:
                token, token_id = special
                result.append(token_id)
                cursor += len(token)
                continue
            result.extend(byte + 3 for byte in text[cursor].encode("utf-8"))
            cursor += 1
        return result


@dataclass(frozen=True)
class HunyuanQwenEncoding:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor


class HunyuanQwenEncoder:
    def __init__(
        self, model: QwenImageTextModel, embeddings: EmbeddingLookup | None = None
    ) -> None:
        self.model = model
        self.embeddings = embeddings
        tokenizer_config = json.loads(
            read_vendored("qwen_tokenizer_config.json.gz", QWEN_TOKENIZER_CONFIG_SHA256)
        )
        added = tokenizer_config.get("added_tokens_decoder", {})
        if (
            added.get("151643", {}).get("content") != "<|endoftext|>"
            or added.get("151644", {}).get("content") != "<|im_start|>"
            or added.get("151645", {}).get("content") != "<|im_end|>"
        ):
            raise ValueError("vendored Qwen tokenizer contract is invalid")
        self.profile = TokenizerProfile(99999999, None, None, 151643, False, min_length=1)
        self.tokenizer = PromptTokenizer(
            encode_word=load_qwen_bpe().encode,
            resolve=None if embeddings is None else self._rows,
            disable_weights=True,
        )

    def _rows(self, name: str) -> int | None:
        vectors = None if self.embeddings is None else self.embeddings(name)
        return None if vectors is None else vectors.shape[0]

    def tokenize(self, text: str) -> Chunk:
        formatted = (
            text
            if text.startswith(("<|im_start|>", "<|start_header_id|>"))
            else HUNYUAN_IMAGE_TEMPLATE.format(text)
        )
        spans = self.tokenizer.tokenize(formatted)
        return pack_spans(spans, self.profile, resolve=self._rows)[0]

    def encode(self, text: str) -> HunyuanQwenEncoding:
        tokens = self.tokenize(text)
        chunks = (tokens,)
        device = (
            bound_compute_device(self.model.model.embed_tokens)
            or self.model.model.embed_tokens.weight.device
        )
        ids = torch.tensor(
            [[token.unit if isinstance(token.unit, int) else 0 for token in row] for row in chunks],
            device=device,
            dtype=torch.long,
        )
        embeds = self.model.model.embed(ids)
        for row_index, row in enumerate(chunks):
            for index, token in enumerate(row):
                if isinstance(token.unit, EmbeddingSlot):
                    vectors = None if self.embeddings is None else self.embeddings(token.unit.name)
                    if vectors is None or vectors.ndim != 2 or vectors.shape[1] != embeds.shape[-1]:
                        raise ValueError("Qwen embedding must have (rows, hidden_size) shape")
                    embeds[row_index, index] = vectors[token.unit.row].to(embeds)
        mask = torch.ones(ids.shape, dtype=torch.long, device=device)
        tap = self.model.model.shape.num_layers - 2
        hidden = self.model.model.tapped_embeds(embeds, mask, tap_layers=(tap,))[:, 0].float()
        plain_ids = [token.unit if isinstance(token.unit, int) else 0 for token in tokens]
        crop_start = -1
        if plain_ids[0] == 27 and len(plain_ids) > 36:
            crop_start = 36
        else:
            marker_count = 0
            for index, token_id in enumerate(plain_ids):
                if token_id == 151644 and marker_count < 2:
                    crop_start = index
                    marker_count += 1
            if len(plain_ids) > crop_start + 3 and plain_ids[crop_start + 1 : crop_start + 3] == [
                872,
                198,
            ]:
                crop_start += 3
        crop = slice(crop_start, len(tokens))
        return HunyuanQwenEncoding(hidden[:, crop].clone(), mask[:1, crop].clone())


def compose_hunyuan_image_conditioning(
    qwen: HunyuanQwenEncoding,
    byt5: Conditioning[torch.Tensor] | None,
) -> ConditioningCarrier:
    text = tensor_to_payload_binding("text", qwen.embeddings, space="conditioning-text")
    metadata: list[tuple[str, PayloadReference]] = []
    bindings = [text]
    if not bool(torch.all(qwen.attention_mask == 1)):
        mask = tensor_to_payload_binding(
            "attention_mask", qwen.attention_mask, space="conditioning-attention-mask"
        )
        bindings.append(mask)
        metadata.append((ATTENTION_MASK_METADATA, PayloadReference(mask.reference_id)))
    if byt5 is not None:
        glyph = tensor_to_payload_binding(
            "conditioning_byt5small", byt5.embeddings, space="conditioning-byt5small"
        )
        bindings.append(glyph)
        metadata.append((BYT5_CONDITIONING_METADATA, PayloadReference(glyph.reference_id)))
    record = ConditioningRecord(
        channels=(
            (
                ConditioningChannel.TEXT,
                PayloadDescriptor(
                    PayloadReference(text.reference_id), text.shape, text.dtype, text.space
                ),
            ),
        ),
        extension_metadata=tuple(metadata),
    )
    return make_conditioning_carrier(ConditioningSet((record,)), tuple(bindings))


class HunyuanImageTextRuntime:
    def __init__(
        self,
        loaded: LoadedTextRecipe,
        *,
        embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    ) -> None:
        qwen, byt5 = loaded.module["qwen25_vl"], loaded.module["byt5_small"]
        if not isinstance(qwen, QwenImageTextModel) or not isinstance(byt5, T5TextModel):
            raise ValueError("Hunyuan Image requires Qwen2.5-VL and ByT5-small modules")
        lookups = embedding_lookups or {}
        self.qwen = HunyuanQwenEncoder(qwen, lookups.get("qwen25_vl"))
        part = next(part for part in loaded.binding.components if part.role == "byt5_small")
        if part.profile is None:
            raise ValueError("ByT5-small requires its packing profile")
        self.byt5 = T5TextEncoder(
            byt5,
            profile=part.profile.tokenizer,
            attention_masked=True,
            zero_out_masked=True,
            embeddings=lookups.get("byt5_small"),
        )
        self.byt5_tokenizer = PromptTokenizer(
            encode_word=ByT5Tokenizer().encode,
            resolve=None if "byt5_small" not in lookups else self._byt5_rows,
        )
        assert loaded.binding.composer is not None
        self._compose = execution_symbol(loaded.binding.composer)

    def _byt5_rows(self, name: str) -> int | None:
        vectors = None if self.byt5.embeddings is None else self.byt5.embeddings(name)
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
        if any(value is not None for value in (hidden_layer, min_padding, min_length)):
            raise ValueError("Hunyuan Image uses fixed text encoding policy")
        qwen = self.qwen.encode(text)
        glyph_text = extract_byt5_prompt(text)
        byt5 = (
            None
            if glyph_text is None
            else self.byt5.encode(self.byt5_tokenizer.tokenize(glyph_text))
        )
        return self._compose(qwen, byt5)
