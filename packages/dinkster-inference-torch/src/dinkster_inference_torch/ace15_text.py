"""ACE-Step 1.5 structured prompts, constrained audio LM, and text carrier.

The recipe follows comfy/text_encoders/ace15.py at 25dfc16f9ac0.
Audio-code generation is text-side conditioning, not diffusion sampling.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, BinaryIO, cast

import torch
import torch.nn.functional as F
import yaml  # pyright: ignore[reportMissingTypeStubs]
from dinkster_inference import (
    AttentionPolicy,
    AttentionRouteToken,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    encode_conditioning_carrier,
    make_conditioning_carrier,
)
from dinkster_inference.ace15_text import ACE15_TEXT_ROLES
from dinkster_inference.component_registry import execution_symbol
from dinkster_inference.qwen_bpe import QwenBpe, load_qwen_bpe, sd_tokenizer_segments
from dinkster_inference.qwen_text import QwenTextConfig
from dinkster_inference.sources import SafetensorsSource
from dinkster_inference.text_recipes import TextRecipeBinding

from .assemble import _load_component  # pyright: ignore[reportPrivateUsage]
from .attention import AttentionRole, resolve_role_attention
from .clip_text import EmbeddingLookup
from .operations import bound_compute_device, bound_compute_dtype, materialized_embedding_weight
from .payloads import payload_binding_to_tensor, tensor_to_payload_binding
from .quant_linear import Fp8Linear, Int8Linear, Nvfp4Linear
from .qwen_text import QwenTextModel
from .text_recipes import LoadedTextRecipe

LM_TEMPLATE = (
    "<|im_start|>system\n# Instruction\nGenerate audio semantic tokens based on the given"
    " conditions:\n\n<|im_end|>\n<|im_start|>user\n# Caption\n{}\n\n# Lyric\n{}\n"
    "<|im_end|>\n<|im_start|>assistant\n{}\n\n<|im_end|>\n"
)
LYRICS_TEMPLATE = "# Languages\n{}\n\n# Lyric\n{}<|endoftext|><|endoftext|>"
CONDITIONER_TEMPLATE = (
    "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"
    "# Caption\n{}\n\n# Metas\n{}\n<|endoftext|>\n<|endoftext|>"
)
AUDIO_START = 151669
AUDIO_END = 215669
EOS = 151645
PAD = 151643


@dataclass(frozen=True)
class ACE15Generation:
    min_tokens: int = 1
    max_tokens: int = 1024
    seed: int = 0
    generate_audio_codes: bool = True
    cfg_scale: float = 2.0
    temperature: float = 0.85
    top_p: float | None = 0.9
    top_k: int | None = 0
    min_p: float | None = 0.0


@dataclass(frozen=True)
class ACE15Tokens:
    lm_prompt: tuple[int, ...]
    lm_prompt_negative: tuple[int, ...]
    lyrics: tuple[int, ...]
    qwen3_06b: tuple[int, ...]
    generation: ACE15Generation


@dataclass(frozen=True)
class ACE15Conditioning:
    sequence: torch.Tensor
    lyrics: torch.Tensor
    audio_codes: torch.Tensor | None = None


class ACE15TextRuntimeError(ValueError):
    """The ACE text carrier violates its exact runtime contract."""


def _metadata_think(parameters: Mapping[str, Any]) -> str:
    metas = {
        key: parameters[key]
        for key in ("bpm", "duration", "keyscale", "timesignature")
        if key in parameters
    }
    signature = metas.get("timesignature")
    if isinstance(signature, str) and signature.endswith("/4"):
        metas["timesignature"] = signature[:-2]
    metas = {
        key: int(value) if isinstance(value, str) and value.isdigit() else value
        for key, value in metas.items()
        if value not in {"unspecified", None}
    }
    serialized = yaml.dump(metas, allow_unicode=True, sort_keys=True).strip() if metas else ""
    return f"<think>\n{serialized}\n</think>"


def ace15_prompts(text: str, **parameters: Any) -> tuple[dict[str, str], ACE15Generation]:
    """Serialize the source's caption, lyrics, metadata and negative variants."""
    text = text.strip()
    negative = parameters.get("caption_negative", text).strip()
    lyrics = parameters.get("lyrics", "")
    negative_lyrics = parameters.get("lyrics_negative", lyrics)
    duration = parameters.get("duration", 120)
    if isinstance(duration, str):
        duration = float(duration.split(None, 1)[0])
    duration = math.ceil(duration)
    parameters["duration"] = duration
    generation = ACE15Generation(
        min_tokens=int(parameters.get("min_tokens", duration * 5)),
        max_tokens=int(parameters.get("max_tokens", duration * 5)),
        seed=parameters.get("seed", 0),
        generate_audio_codes=parameters.get("generate_audio_codes", True),
        cfg_scale=parameters.get("cfg_scale", 2.0),
        temperature=parameters.get("temperature", 0.85),
        top_p=parameters.get("top_p", 0.9),
        top_k=parameters.get("top_k", 0),
        min_p=parameters.get("min_p", 0.0),
    )
    negative_metas = {
        key: parameters[key + "_negative"]
        for key in ("bpm", "duration", "keyscale", "timesignature", "language", "caption")
        if key + "_negative" in parameters
    }
    if not parameters.get("use_negative_caption"):
        negative_metas.pop("caption", None)
    cap = {key: parameters.get(key, "N/A") for key in ("bpm", "timesignature", "keyscale")}
    if isinstance(cap["timesignature"], str) and cap["timesignature"].endswith("/4"):
        cap["timesignature"] = cap["timesignature"][:-2]
    cap["duration"] = f"{duration} seconds"
    language = parameters.get("language")
    return {
        "lm_prompt": LM_TEMPLATE.format(text, lyrics.strip(), _metadata_think(parameters)),
        "lm_prompt_negative": LM_TEMPLATE.format(
            negative, negative_lyrics.strip(), _metadata_think(negative_metas)
        ),
        "lyrics": LYRICS_TEMPLATE.format(language if language is not None else "", lyrics),
        "qwen3_06b": CONDITIONER_TEMPLATE.format(
            text, "\n".join(f"- {key}: {value}" for key, value in cap.items())
        ),
    }, generation


def tokenize_ace15_prompt(
    text: str, *, tokenizer: QwenBpe | None = None, **parameters: Any
) -> ACE15Tokens:
    prompts, generation = ace15_prompts(text, **parameters)
    bpe = tokenizer or load_qwen_bpe()
    minimum = parameters.get("min_length") or 1

    def encode(prompt: str) -> tuple[int, ...]:
        ids = tuple(
            token for segment in sd_tokenizer_segments(prompt) for token in bpe.encode(segment)
        )
        return ids + (PAD,) * max(0, minimum - len(ids))

    return ACE15Tokens(
        encode(prompts["lm_prompt"]),
        encode(prompts["lm_prompt_negative"]),
        encode(prompts["lyrics"]),
        encode(prompts["qwen3_06b"]),
        generation,
    )


def ace15_attention_mask(ids: Sequence[int]) -> tuple[int, ...]:
    """Mask leading pads and all positions from the first non-leading pad."""
    left = True
    ended = False
    mask: list[int] = []
    for token in ids:
        if token != PAD:
            left = False
        elif not left:
            ended = True
        mask.append(int(not left and not ended))
    return tuple(mask)


def generate_audio_codes(
    model: QwenTextModel,
    positive: Sequence[int],
    negative: Sequence[int],
    options: ACE15Generation,
) -> torch.Tensor:
    """Seeded constrained generation with source-order filters and full-vocab draws."""
    rows = [tuple(positive)]
    if options.cfg_scale != 1.0:
        rows.append(tuple(negative))
    length = max(map(len, rows))
    rows = [(PAD,) * (length - len(row)) + row for row in rows]
    device = bound_compute_device(model.embed_tokens) or model.embed_tokens.weight.device
    ids = torch.tensor(rows, device=device, dtype=torch.long)
    masks = [ace15_attention_mask(row) for row in rows]
    mask = (
        torch.tensor(masks, device=device, dtype=torch.long)
        if any(0 in row for row in masks)
        else None
    )
    dtype = bound_compute_dtype(model.embed_tokens) or model.embed_tokens.weight.dtype
    cache = model.allocate_causal_cache(
        len(rows), length + options.max_tokens, dtype=dtype, fixed=True
    )
    codes = torch.empty((options.max_tokens,), device=device, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(options.seed)
    sampling_logits = torch.empty((1, model.config.vocab_size), device=device, dtype=dtype)
    position = 0
    count = 0
    for step in range(options.max_tokens):
        hidden, _ = model.forward_causal(ids, cache, cache_position=position, attention_mask=mask)
        position += ids.shape[1]
        use_eos = options.min_tokens < step
        with materialized_embedding_weight(model.embed_tokens) as weight:
            last = hidden[:, -1].to(weight.device)
            logits = F.linear(last, weight[AUDIO_START:AUDIO_END])
            if use_eos:
                logits = torch.cat((F.linear(last, weight[EOS : EOS + 1]), logits), dim=-1)
        logits = (
            logits[1:2] + options.cfg_scale * (logits[0:1] - logits[1:2])
            if options.cfg_scale != 1.0
            else logits[0:1]
        )
        remove = torch.finfo(logits.dtype).min
        if options.top_k is not None and options.top_k > 0:
            threshold = torch.topk(logits, min(options.top_k, logits.shape[-1])).values[:, -1:]
            logits[logits < threshold] = remove
        if options.min_p is not None and options.min_p > 0:
            probabilities = logits.softmax(dim=-1)
            logits[
                probabilities < options.min_p * probabilities.max(dim=-1, keepdim=True).values
            ] = remove
        if options.top_p is not None and options.top_p < 1.0:
            sorted_logits, indices = logits.sort(descending=True)
            removed = sorted_logits.softmax(dim=-1).cumsum(dim=-1) > options.top_p
            removed[:, 1:] = removed[:, :-1].clone()
            removed[:, 0] = False
            logits[torch.zeros_like(removed).scatter_(1, indices, removed)] = remove
        if options.temperature > 0:
            filtered = logits / options.temperature
            sampling_logits.fill_(remove)
            if use_eos:
                sampling_logits[:, EOS] = filtered[:, 0]
                filtered = filtered[:, 1:]
            sampling_logits[:, AUDIO_START:AUDIO_END] = filtered
            token = torch.multinomial(sampling_logits.softmax(dim=-1), 1, generator=generator)
        else:
            token = logits.argmax(dim=-1, keepdim=True)
            token = (
                torch.where(token == 0, EOS, token + AUDIO_START - 1)
                if use_eos
                else token + AUDIO_START
            )
        if token.item() == EOS:
            break
        codes[count] = token[0, 0] - AUDIO_START
        count += 1
        ids = token.repeat(len(rows), 1)
        if mask is not None:
            mask = torch.cat(
                (mask, torch.ones((len(rows), 1), device=mask.device, dtype=mask.dtype)), dim=1
            )
    return codes[:count].clone().unsqueeze(0)


def compose_ace15_conditioning(
    sequence: torch.Tensor, lyrics: torch.Tensor, audio_codes: torch.Tensor | None
) -> ConditioningCarrier:
    text = tensor_to_payload_binding("text", sequence, space="conditioning-text")
    bindings = [
        text,
        tensor_to_payload_binding("conditioning_lyrics", lyrics, space="conditioning-lyrics"),
    ]
    metadata = {"dinkster.ace15/conditioning_lyrics": PayloadReference("conditioning_lyrics")}
    if audio_codes is not None:
        bindings.append(tensor_to_payload_binding("audio_codes", audio_codes, space="audio-codes"))
        metadata["dinkster.ace15/audio_codes"] = PayloadReference("audio_codes")
    record = ConditioningRecord(
        channels=(
            (
                ConditioningChannel.TEXT,
                PayloadDescriptor(PayloadReference("text"), text.shape, text.dtype, text.space),
            ),
        ),
        extension_metadata=tuple(metadata.items()),
    )
    return make_conditioning_carrier(ConditioningSet((record,)), tuple(bindings))


def materialize_ace15_conditioning(
    carrier: ConditioningCarrier, *, device: torch.device | str
) -> ACE15Conditioning:
    """Materialize one canonical ACE carrier on the requested device."""
    if type(carrier) is not ConditioningCarrier:
        raise TypeError("carrier must be exact ConditioningCarrier")
    encode_conditioning_carrier(carrier)
    records = carrier.conditioning.records
    if len(records) != 1:
        raise ACE15TextRuntimeError("ACE conditioning requires one record")
    record = records[0]
    if (
        record.area is not None
        or record.mask is not None
        or record.scale_vector is not None
        or record.token_layout is not None
        or type(record.schedule) is not PercentRange
        or (record.schedule.start_percent, record.schedule.end_percent) != (0.0, 1.0)
    ):
        raise ACE15TextRuntimeError("ACE conditioning requires one full unmodified record")
    channels = dict(record.channels)
    text = channels.pop(ConditioningChannel.TEXT, None)
    if text is None or channels:
        raise ACE15TextRuntimeError("ACE conditioning requires only one text channel")
    lyrics_key = "dinkster.ace15/conditioning_lyrics"
    audio_key = "dinkster.ace15/audio_codes"
    metadata = dict(record.extension_metadata)
    if set(metadata) - {lyrics_key, audio_key}:
        raise ACE15TextRuntimeError("ACE conditioning contains unknown extension metadata")
    lyrics = metadata.get(lyrics_key)
    audio = metadata.get(audio_key)
    if type(lyrics) is not PayloadReference:
        raise ACE15TextRuntimeError("ACE conditioning requires one lyrics payload")
    if audio is not None and type(audio) is not PayloadReference:
        raise ACE15TextRuntimeError("ACE audio-code metadata must be one payload")
    bindings = {binding.reference_id: binding for binding in carrier.bindings}

    def tensor(reference: PayloadReference, space: str) -> torch.Tensor:
        binding = bindings.get(reference.id)
        if binding is None or binding.space != space:
            raise ACE15TextRuntimeError(f"ACE conditioning requires {space!r} payloads")
        return payload_binding_to_tensor(binding).to(device)

    return ACE15Conditioning(
        tensor(text.reference, "conditioning-text"),
        tensor(lyrics, "conditioning-lyrics"),
        None if audio is None else tensor(audio, "audio-codes"),
    )


def assemble_ace15_text_recipe(
    binding: TextRecipeBinding,
    *,
    compute_dtype: torch.dtype,
    sources: tuple[SafetensorsSource, ...],
    source_files: tuple[BinaryIO, ...],
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    attention_backends: tuple[tuple[str, AttentionRole], ...],
) -> LoadedTextRecipe:
    if len(sources) != len(source_files) or {
        part.source_index for part in binding.components
    } != set(range(len(sources))):
        raise ValueError("ACE assembly requires every ordered source and open file")
    backends = dict(attention_backends)
    modules: dict[str, torch.nn.Module] = {}
    attention_status = []
    for part in binding.components:
        if not isinstance(part.plan.config, QwenTextConfig):
            raise ValueError("ACE text assembly requires Qwen text components")
        try:
            attention_backend = backends[part.role]
        except KeyError as error:
            raise ValueError(f"ACE text role {part.role!r} has no attention backend") from error
        attention = resolve_role_attention(
            attention_backend, attention_policy, attention_route_token
        )
        attention_status.append(attention.status)
        modules[part.role] = _load_component(
            part.plan,
            partial(QwenTextModel, attention_kernel=attention.kernel),
            compute_dtype=compute_dtype,
            fp8_matmul=False,
            source=sources[part.source_index],
            source_file=source_files[part.source_index],
        )
        for layer in modules[part.role].modules():
            if isinstance(layer, Fp8Linear | Int8Linear | Nvfp4Linear):
                layer.compute_dtype = torch.float32
                layer.full_precision_matmul = True
    return LoadedTextRecipe(binding, torch.nn.ModuleDict(modules), tuple(attention_status))


class ACE15TextRuntime:
    """Encode structured ACE parameters; generic string callers use source defaults."""

    def __init__(
        self,
        loaded: LoadedTextRecipe,
        *,
        embedding_lookups: Mapping[str, EmbeddingLookup] | None = None,
    ) -> None:
        if embedding_lookups:
            raise ValueError("ACE-Step 1.5 does not support textual inversion")
        self.loaded = loaded
        self._compose = execution_symbol(cast(str, loaded.binding.composer))

    @staticmethod
    def text_conditioning_carrier(value: ConditioningCarrier) -> ConditioningCarrier:
        return value

    def materialize_conditioning(
        self, carrier: ConditioningCarrier, *, device: torch.device | str
    ) -> ACE15Conditioning:
        return materialize_ace15_conditioning(carrier, device=device)

    def encode_text(self, text: str, **parameters: Any) -> ConditioningCarrier:
        return self.encode_tokens(tokenize_ace15_prompt(text, **parameters))

    def encode_tokens(self, tokens: ACE15Tokens) -> ConditioningCarrier:
        conditioner = cast(QwenTextModel, self.loaded.module[ACE15_TEXT_ROLES[0]])
        lm = cast(QwenTextModel, self.loaded.module[ACE15_TEXT_ROLES[1]])
        device = (
            bound_compute_device(conditioner.embed_tokens) or conditioner.embed_tokens.weight.device
        )

        def inputs(row: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.tensor([row], device=device, dtype=torch.long),
                torch.tensor([ace15_attention_mask(row)], device=device, dtype=torch.long),
            )

        sequence = conditioner(*inputs(tokens.qwen3_06b))
        # A call-local view shares routed parameters without mutating the model's
        # output policy, so simultaneous lyric/final encodes cannot race.
        lyric_model = copy.copy(conditioner)
        lyric_model.config = replace(
            conditioner.config, output_hidden_layer=0, layer_norm_hidden_state=False
        )
        lyrics = lyric_model(*inputs(tokens.lyrics))
        options = tokens.generation
        codes = None
        if options.generate_audio_codes:
            # Preserve ace15.py's call-site bound; see the indexed upstream issue.
            codes = generate_audio_codes(
                lm,
                tokens.lm_prompt,
                tokens.lm_prompt_negative,
                replace(options, max_tokens=options.min_tokens),
            )
        return self._compose(sequence, lyrics, codes)
