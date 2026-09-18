"""NewBie Gemma 3 sequence and Jina CLIP v2 pooled conditioning."""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import BinaryIO, Protocol, cast

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
from dinkster_inference.component_registry import execution_symbol
from dinkster_inference.gemma_text import GEMMA3_NEWBIE_4B_CONFIG, GemmaTextConfig
from dinkster_inference.jina_clip_text import JinaClipTextConfig
from dinkster_inference.newbie_text import NEWBIE_TOKENIZER_KEYS
from dinkster_inference.prompt_tokens import (
    EmbeddingSlot,
    PromptTokenizer,
    TokenizerProfile,
    pack_spans,
)
from dinkster_inference.sources import SafetensorsSource
from dinkster_inference.text_recipes import TextRecipeBinding

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .attention import AttentionStatus, resolve_role_attention
from .clip_text import EmbeddingLookup
from .conditioning_adapters import basic_conditioning_to_carrier
from .gemma_text import GemmaTextModel
from .jina_clip_text import JinaClipTextModel
from .operations import bound_compute_device
from .payloads import tensor_to_payload_binding
from .quant_linear import Fp8Linear, Int8Linear, Nvfp4Linear
from .text_recipes import LoadedTextRecipe

ATTENTION_MASK_METADATA = "dinkster.newbie/attention_mask"
TOKENIZER_BYTE_CAP = 8 * 1024 * 1024


class _SentencePiece(Protocol):
    def get_piece_size(self) -> int: ...
    def pad_id(self) -> int: ...
    def eos_id(self) -> int: ...
    def bos_id(self) -> int: ...
    def encode(
        self, text: str, *, out_type: type[int], add_bos: bool, add_eos: bool
    ) -> list[int]: ...


class NewBieSentencePiece:
    def __init__(
        self,
        model: bytes,
        *,
        size: int,
        special_ids: tuple[int, int, int],
        added_tokens: Mapping[str, int] | None = None,
    ) -> None:
        processor = cast(
            _SentencePiece,
            importlib.import_module("sentencepiece").SentencePieceProcessor(
                model_proto=model, add_bos=False, add_eos=False
            ),
        )
        if (
            processor.get_piece_size(),
            processor.bos_id(),
            processor.eos_id(),
            processor.pad_id(),
        ) != (
            size,
            *special_ids,
        ):
            raise ValueError("NewBie SentencePiece model has the wrong vocabulary or special ids")
        self._processor = processor
        self._added = dict(added_tokens or {})
        self._split = (
            None
            if not self._added
            else re.compile("(" + "|".join(re.escape(token) for token in self._added) + ")")
        )

    def encode(self, text: str) -> Sequence[int]:
        if self._split is None:
            return self._processor.encode(text, out_type=int, add_bos=False, add_eos=False)
        ids: list[int] = []
        for part in self._split.split(text):
            if not part:
                continue
            special = self._added.get(part)
            if special is None:
                ids.extend(self._processor.encode(part, out_type=int, add_bos=False, add_eos=False))
            else:
                ids.append(special)
        return ids


@dataclass(frozen=True)
class LoadedNewBieText(LoadedTextRecipe):
    tokenizers: tuple[bytes, bytes]


def assemble_newbie_text(
    binding: TextRecipeBinding,
    *,
    compute_dtype: torch.dtype,
    sources: tuple[SafetensorsSource, ...],
    source_files: tuple[BinaryIO, ...],
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
) -> LoadedNewBieText:
    if len(sources) != len(source_files) or {
        part.source_index for part in binding.components
    } != set(range(len(sources))):
        raise ValueError("NewBie assembly requires every ordered source and open file")
    modules: dict[str, torch.nn.Module] = {}
    statuses: list[AttentionStatus] = []
    tokenizer_models: dict[str, bytes] = {}
    for part in binding.components:
        source = sources[part.source_index]
        tokenizer_models[part.role] = source.read_uint8_configuration_from_file(
            source_files[part.source_index],
            NEWBIE_TOKENIZER_KEYS[part.role],
            limit=TOKENIZER_BYTE_CAP,
        )
        if isinstance(part.plan.config, GemmaTextConfig):
            attention = resolve_role_attention("qwen", attention_policy, attention_route_token)
            builder = partial(GemmaTextModel, attention_kernel=attention.kernel)
        elif isinstance(part.plan.config, JinaClipTextConfig):
            attention = resolve_role_attention("clip", attention_policy, attention_route_token)
            builder = partial(JinaClipTextModel, attention_kernel=attention.kernel)
        else:
            raise ValueError("NewBie assembly requires Gemma and Jina configs")
        module = _load_component(
            part.plan,
            builder,
            compute_dtype=compute_dtype,
            fp8_matmul=False,
            source=source,
            source_file=source_files[part.source_index],
        )
        if part.role == "gemma":
            for layer in module.modules():
                if isinstance(layer, Fp8Linear | Int8Linear | Nvfp4Linear):
                    layer.compute_dtype = torch.float32
                    layer.full_precision_matmul = True
        modules[part.role] = module
        statuses.append(attention.status)
    return LoadedNewBieText(
        binding,
        torch.nn.ModuleDict(modules),
        tuple(statuses),
        (tokenizer_models["gemma"], tokenizer_models["jina"]),
    )


def _ids_and_mask(
    text: str,
    tokenizer: NewBieSentencePiece,
    profile: TokenizerProfile,
    lookup: EmbeddingLookup | None,
    width: int,
    embedding: torch.nn.Embedding,
    *,
    disable_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def rows(name: str) -> int | None:
        vectors = None if lookup is None else lookup(name)
        return None if vectors is None else vectors.shape[0]

    packed = pack_spans(
        PromptTokenizer(
            tokenizer.encode,
            resolve=rows if lookup is not None else None,
            disable_weights=disable_weights,
        ).tokenize(text),
        profile,
        resolve=rows if lookup is not None else None,
    )[0]
    device = bound_compute_device(embedding) or embedding.weight.device
    ids = torch.tensor(
        [[token.unit if isinstance(token.unit, int) else profile.pad_token for token in packed]],
        device=device,
        dtype=torch.long,
    )
    mask = torch.tensor(
        [[0 if token.unit == profile.pad_token else 1 for token in packed]],
        device=device,
        dtype=torch.long,
    )
    embeds = embedding(ids)
    for index, token in enumerate(packed):
        if isinstance(token.unit, EmbeddingSlot):
            vectors = None if lookup is None else lookup(token.unit.name)
            if vectors is None or vectors.ndim != 2 or vectors.shape[1] != width:
                raise ValueError("NewBie embedding has the wrong shape")
            embeds[0, index] = vectors[token.unit.row].to(embeds)
    return ids, mask, embeds


def compose_newbie_conditioning(
    gemma: tuple[torch.Tensor, torch.Tensor], jina: Conditioning[torch.Tensor]
) -> ConditioningCarrier:
    carrier = basic_conditioning_to_carrier(Conditioning(gemma[0], jina.pooled))
    if bool(torch.all(gemma[1] == 1)):
        return carrier
    mask = tensor_to_payload_binding(
        "attention_mask", gemma[1], space="conditioning-attention-mask"
    )
    record = replace(
        carrier.conditioning.records[0],
        extension_metadata=((ATTENTION_MASK_METADATA, PayloadReference(mask.reference_id)),),
    )
    return make_conditioning_carrier(ConditioningSet((record,)), (*carrier.bindings, mask))


class NewBieTextRuntime:
    def __init__(
        self,
        loaded: LoadedNewBieText,
        *,
        embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    ) -> None:
        self.loaded = loaded
        self.gemma = cast(GemmaTextModel, loaded.module["gemma"])
        self.jina = cast(JinaClipTextModel, loaded.module["jina"])
        self.lookups = embedding_lookups or {}
        self.gemma_tokenizer = NewBieSentencePiece(
            loaded.tokenizers[0],
            size=262144,
            special_ids=(2, 1, 0),
            added_tokens={"<image_soft_token>": 262144, "<end_of_turn>": 106},
        )
        self.jina_tokenizer = NewBieSentencePiece(
            loaded.tokenizers[1], size=250000, special_ids=(0, 2, 1)
        )
        assert loaded.binding.composer is not None
        self._compose = execution_symbol(loaded.binding.composer)

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
            raise ValueError("NewBie uses fixed text encoding policy")
        gemma_ids, gemma_mask, gemma_embeds = _ids_and_mask(
            text,
            self.gemma_tokenizer,
            TokenizerProfile(99999999, 2, None, 0, False, min_length=1),
            self.lookups.get("gemma"),
            GEMMA3_NEWBIE_4B_CONFIG.hidden_size,
            self.gemma.embed_tokens,
            disable_weights=True,
        )
        gemma = self.gemma(gemma_ids, gemma_mask, embeds=gemma_embeds)[:, -2].float()
        jina_ids, jina_mask, jina_embeds = _ids_and_mask(
            text,
            self.jina_tokenizer,
            TokenizerProfile(8192, 0, 2, 1, False, min_length=1),
            self.lookups.get("jina"),
            self.jina.config.hidden_size,
            self.jina.model.embeddings.word_embeddings,
            disable_weights=False,
        )
        sequence, pooled = self.jina(jina_ids, jina_mask, embeds=jina_embeds)
        jina = Conditioning(sequence.float(), pooled.float())
        return self._compose((gemma, gemma_mask), jina)
