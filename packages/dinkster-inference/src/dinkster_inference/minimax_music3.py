"""MiniMax Music 3 profiles, geometry, and prompt policy.

The implementation is sourced from ComfyUI 345c9190497c82cff53e71fb4ae00d1e135a6542
and the maintained workflow template at
8417f4f2a8380556070721d0ff1da8285d6e5438. Official artifact identities are
pinned to Comfy-Org/MiniMax-Music-3 revision
6baad88896848433857c170ba4f05d2ea9d5f218.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .codecs import CodecDescriptor, CodecTiling
from .devices import BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, DType
from .families import EvidenceValue
from .latents import LatentDescriptor
from .weights import TensorGeometry, WeightSource

COMFYUI_SOURCE_COMMIT = "345c9190497c82cff53e71fb4ae00d1e135a6542"
WORKFLOW_TEMPLATES_SOURCE_COMMIT = "8417f4f2a8380556070721d0ff1da8285d6e5438"
OFFICIAL_ARTIFACT_REVISION = "6baad88896848433857c170ba4f05d2ea9d5f218"
_OFFICIAL_ARTIFACT_BASE = (
    f"https://huggingface.co/Comfy-Org/MiniMax-Music-3/resolve/{OFFICIAL_ARTIFACT_REVISION}/"
)
OFFICIAL_ARTIFACTS: Mapping[str, tuple[str, int, str]] = MappingProxyType(
    {
        "diffusion_models/minimax_music3_dit_fp16.safetensors": (
            _OFFICIAL_ARTIFACT_BASE + "diffusion_models/minimax_music3_dit_fp16.safetensors",
            4_914_197_682,
            "45494a2b6b69af115902ff28eaf54118d19067aa54da01000f3e3efce7ba0e34",
        ),
        "diffusion_models/minimax_music3_dit_fp32.safetensors": (
            _OFFICIAL_ARTIFACT_BASE + "diffusion_models/minimax_music3_dit_fp32.safetensors",
            9_828_345_396,
            "ab54b44bcc40ea49c44bd65fd86cade12adf1c5bd9a5aba63f0c68f438554d2d",
        ),
        "diffusion_models/minimax_music3_dit_int8_convrot.safetensors": (
            _OFFICIAL_ARTIFACT_BASE
            + "diffusion_models/minimax_music3_dit_int8_convrot.safetensors",
            2_502_161_682,
            "d6b959633e69899f99f3a92d6741c0fe79f26958a30811e50e372ef978b24d5f",
        ),
        "text_encoders/minimax_music3_text_encoder_bf16.safetensors": (
            _OFFICIAL_ARTIFACT_BASE + "text_encoders/minimax_music3_text_encoder_bf16.safetensors",
            18_472_478_038,
            "9805d045978ce917cd1e6327b5cd8b85df1a46e4c69b61c9acf3eb29891c3958",
        ),
        "text_encoders/minimax_music3_text_encoder_pruned_bf16.safetensors": (
            _OFFICIAL_ARTIFACT_BASE
            + "text_encoders/minimax_music3_text_encoder_pruned_bf16.safetensors",
            16_706_629_398,
            "e81e469c92af324dc69b77e5a11179e22542a6d628f2b205c8cf6143e510d976",
        ),
        "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors": (
            _OFFICIAL_ARTIFACT_BASE
            + "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
            9_196_611_886,
            "010b7416d2336a08c711bc22ee65849c9623069ddb7d89bec011a75699e52014",
        ),
        "vae/minimax_music3_dav.safetensors": (
            _OFFICIAL_ARTIFACT_BASE + "vae/minimax_music3_dav.safetensors",
            216_696_128,
            "2a32155b769be01445fcc2a8663b910fc9e1751e18dc1c3ec528064512d9ef0c",
        ),
    }
)

SPECIAL_TOKEN_IDS: Mapping[str, int] = MappingProxyType(
    {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|audio_cfg|>": 151654,
        "<|audio_start|>": 151669,
        "<|audio_end|>": 151670,
        "<|caption_start|>": 151671,
        "<|caption_end|>": 151672,
        "<|lyrics_start|>": 151673,
        "<|lyrics_end|>": 151674,
    }
)
AUDIO_CODE_OFFSET = 151675
AUDIO_FRAMES_PER_SECOND = 25
MAX_AUDIO_FRAMES = 9000
MAX_PROMPT_TOKENS = 5000
DEFAULT_CFG_SCALE = 1.5
DEFAULT_TOP_K = 50
C0_VOCAB_SIZE = 16384
MAX_CONDITION_FRAMES = 200
CONDITION_HOP_FRAMES = 100

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LYRIC_TAG_RE = re.compile(r"\s*(\[[^\]]+\])\s*")


def _remove_markdown_format(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", raw_line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text.replace("\u2022 ", "").replace("    ", "")


def clean_music_caption(caption: str) -> str:
    def replace_special(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(replace_special, caption)
    text = _remove_markdown_format(text)
    return re.sub(r"\n{2,}", "\n", text)


def normalize_music_lyrics(lyrics: str) -> str:
    parts = _LYRIC_TAG_RE.split(lyrics)
    text = "\n".join(part.lower() if part.startswith("[") else part for part in parts if part)
    text = text.replace(" ^ ", "\n")
    return f"[start]\n{text}"


def build_music_prompt(caption: str, lyrics: str) -> str:
    return (
        "<|im_start|><|caption_start|>"
        f"{clean_music_caption(caption)}"
        "<|caption_end|><|lyrics_start|>"
        f"{normalize_music_lyrics(lyrics)}"
        "<|lyrics_end|><|im_end|><|audio_start|>"
    )


def derive_music_seed(seed: int, *parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"minimax-ttm")
    digest.update(int(seed).to_bytes(8, "little", signed=False))
    for part in parts:
        value = str(part).encode("utf-8")
        digest.update(len(value).to_bytes(4, "little"))
        digest.update(value)
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


def minimax_music3_latent_length(audio_frames: int) -> int:
    return max(1, int(audio_frames * 44100 / 24000 * 960 / 512))


@dataclass(frozen=True)
class MiniMaxMusic3Config:
    family_id: str = "dinkster.minimax_music3"
    latent_channels: int = 128
    context_layers: int = 8
    context_width: int = 4096
    hidden_width: int = 2048
    blocks: int = 36
    attention_heads: int = 32
    attention_head_dim: int = 64
    feed_forward_width: int = 8192
    inference_dtypes: tuple[DType, DType, DType] = (FLOAT16, BFLOAT16, FLOAT32)
    memory_factor: float = 2.0

    def __post_init__(self) -> None:
        expected = (
            "dinkster.minimax_music3",
            128,
            8,
            4096,
            2048,
            36,
            32,
            64,
            8192,
            (FLOAT16, BFLOAT16, FLOAT32),
            2.0,
        )
        actual = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(
            type(value) is not type(required) or value != required
            for value, required in zip(actual, expected, strict=True)
        ):
            raise ValueError("MiniMaxMusic3Config only represents the exact supported profile")


MINIMAX_MUSIC3_CONFIG = MiniMaxMusic3Config()
MINIMAX_MUSIC3_LATENT = LatentDescriptor(
    channels=128,
    dimensions=1,
    spatial_downscale=512,
)


@dataclass(frozen=True)
class MiniMaxMusic3TextConfig:
    pruned: bool
    merged_qkv: bool
    merged_mlp: bool
    decoder_merged_qkv: bool
    decoder_merged_mlp: bool
    storage_format: Literal["floating", "int8_pruned"]
    vocab_size: int = 200000
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 10240
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    audio_vocab_size: int = 1024
    audio_num_codebooks: int = 8
    decoder_num_heads: int = 16
    decoder_intermediate_size: int = 6144
    decoder_num_layers: int = 4

    def __post_init__(self) -> None:
        merged = (
            self.merged_qkv,
            self.merged_mlp,
            self.decoder_merged_qkv,
            self.decoder_merged_mlp,
        )
        if (self.pruned and not all(merged)) or (not self.pruned and any(merged)):
            raise ValueError("MiniMax Music 3 pruned and merged text layouts must match")
        if self.storage_format not in ("floating", "int8_pruned"):
            raise ValueError("unknown MiniMax Music 3 text storage format")
        if self.storage_format == "int8_pruned" and not self.pruned:
            raise ValueError("MiniMax Music 3 INT8 storage requires the pruned text layout")


@dataclass(frozen=True)
class MiniMaxMusic3DavConfig:
    latent_channels: int = 128
    hidden_channels: int = 1024
    decoder_channels: int = 1536
    strides: tuple[int, int, int, int] = (8, 8, 4, 2)
    sample_rate: int = 44100


MINIMAX_MUSIC3_DAV_CONFIG = MiniMaxMusic3DavConfig()


def minimax_music3_diffusion_layout(
    config: MiniMaxMusic3Config = MINIMAX_MUSIC3_CONFIG,
) -> Mapping[str, tuple[int, ...]]:
    hidden = config.hidden_width
    layout: dict[str, tuple[int, ...]] = {
        "cond_layer_logits": (config.context_layers,),
        "cond_layer_scale": (1,),
        "latent_conditioners.0.weight": (hidden, config.context_width, 3),
        "latent_conditioners.0.bias": (hidden,),
        "diffusion_transformer.preprocess_conv.weight": (2304, 2304, 1),
        "diffusion_transformer.postprocess_conv.weight": (
            config.latent_channels,
            config.latent_channels,
            1,
        ),
        "diffusion_transformer.timestep_features.weight": (128, 1),
        "diffusion_transformer.to_timestep_embed.0.weight": (hidden, 256),
        "diffusion_transformer.to_timestep_embed.0.bias": (hidden,),
        "diffusion_transformer.to_timestep_embed.2.weight": (hidden, hidden),
        "diffusion_transformer.to_timestep_embed.2.bias": (hidden,),
        "diffusion_transformer.transformer.project_in.weight": (hidden, 2304),
        "diffusion_transformer.transformer.project_out.weight": (
            config.latent_channels,
            hidden,
        ),
        "diffusion_transformer.transformer.rotary_pos_emb.inv_freq": (16,),
    }
    for index in range(config.blocks):
        root = f"diffusion_transformer.transformer.layers.{index}"
        layout.update(
            {
                f"{root}.pre_norm.gamma": (hidden,),
                f"{root}.pre_norm.beta": (hidden,),
                f"{root}.self_attn.to_qkv.weight": (3 * hidden, hidden),
                f"{root}.self_attn.to_out.weight": (hidden, hidden),
                f"{root}.ff_norm.gamma": (hidden,),
                f"{root}.ff_norm.beta": (hidden,),
                f"{root}.ff.ff.0.proj.weight": (2 * config.feed_forward_width, hidden),
                f"{root}.ff.ff.0.proj.bias": (2 * config.feed_forward_width,),
                f"{root}.ff.ff.2.weight": (hidden, config.feed_forward_width),
                f"{root}.ff.ff.2.bias": (hidden,),
            }
        )
    return MappingProxyType(layout)


def minimax_music3_text_layout(config: MiniMaxMusic3TextConfig) -> Mapping[str, tuple[int, ...]]:
    hidden = config.hidden_size
    layout: dict[str, tuple[int, ...]] = {
        "model.norm.weight": (hidden,),
        "model.audio_extra_embedding.weight": (
            config.audio_vocab_size * (config.audio_num_codebooks - 1),
            hidden,
        ),
        "model.audio_decoder.projection.weight": (hidden, hidden),
        "model.audio_decoder.pos_embedding.weight": (16, hidden),
        "model.audio_decoder.norm.weight": (hidden,),
        "tokenizer_json": (11423806,),
    }
    if config.pruned:
        layout.update(
            {
                "model.embed_tokens_prefill.weight": (AUDIO_CODE_OFFSET, hidden),
                "model.embed_tokens_audio.weight": (C0_VOCAB_SIZE, hidden),
                "model.lm_head_pruned.weight": (C0_VOCAB_SIZE + 1, hidden),
            }
        )
    else:
        layout.update(
            {
                "model.embed_tokens.weight": (config.vocab_size, hidden),
                "model.lm_head.weight": (config.vocab_size, hidden),
            }
        )
    for index in range(config.num_hidden_layers):
        root = f"model.layers.{index}"
        layout[f"{root}.input_layernorm.weight"] = (hidden,)
        layout[f"{root}.post_attention_layernorm.weight"] = (hidden,)
        layout[f"{root}.self_attn.q_norm.weight"] = (config.head_dim,)
        layout[f"{root}.self_attn.k_norm.weight"] = (config.head_dim,)
        layout[f"{root}.self_attn.o_proj.weight"] = (hidden, hidden)
        layout[f"{root}.mlp.down_proj.weight"] = (hidden, config.intermediate_size)
        if config.merged_qkv:
            query = config.num_attention_heads * config.head_dim
            kv = config.num_key_value_heads * config.head_dim
            layout[f"{root}.self_attn.qkv_proj.weight"] = (query + 2 * kv, hidden)
        else:
            layout[f"{root}.self_attn.q_proj.weight"] = (
                config.num_attention_heads * config.head_dim,
                hidden,
            )
            layout[f"{root}.self_attn.k_proj.weight"] = (
                config.num_key_value_heads * config.head_dim,
                hidden,
            )
            layout[f"{root}.self_attn.v_proj.weight"] = (
                config.num_key_value_heads * config.head_dim,
                hidden,
            )
        if config.merged_mlp:
            layout[f"{root}.mlp.gate_up_proj.weight"] = (2 * config.intermediate_size, hidden)
        else:
            layout[f"{root}.mlp.gate_proj.weight"] = (config.intermediate_size, hidden)
            layout[f"{root}.mlp.up_proj.weight"] = (config.intermediate_size, hidden)
    for index in range(config.decoder_num_layers):
        root = f"model.audio_decoder.layers.{index}"
        layout[f"{root}.input_layernorm.weight"] = (hidden,)
        layout[f"{root}.post_attention_layernorm.weight"] = (hidden,)
        layout[f"{root}.self_attn.o_proj.weight"] = (hidden, hidden)
        layout[f"{root}.mlp.down_proj.weight"] = (
            hidden,
            config.decoder_intermediate_size,
        )
        if config.decoder_merged_qkv:
            layout[f"{root}.self_attn.qkv_proj.weight"] = (3 * hidden, hidden)
        else:
            for projection in ("q", "k", "v"):
                layout[f"{root}.self_attn.{projection}_proj.weight"] = (hidden, hidden)
        if config.decoder_merged_mlp:
            layout[f"{root}.mlp.gate_up_proj.weight"] = (
                2 * config.decoder_intermediate_size,
                hidden,
            )
        else:
            layout[f"{root}.mlp.gate_proj.weight"] = (
                config.decoder_intermediate_size,
                hidden,
            )
            layout[f"{root}.mlp.up_proj.weight"] = (
                config.decoder_intermediate_size,
                hidden,
            )
    for index in range(config.audio_num_codebooks - 1):
        layout[f"model.audio_decoder.audio_heads.{index}.weight"] = (
            config.audio_vocab_size,
            hidden,
        )
    return MappingProxyType(layout)


def _minimax_music3_int8_text_weights(config: MiniMaxMusic3TextConfig) -> frozenset[str]:
    weights: set[str] = set()
    for index in range(config.num_hidden_layers):
        root = f"model.layers.{index}"
        weights.update(
            {
                f"{root}.self_attn.qkv_proj.weight",
                f"{root}.self_attn.o_proj.weight",
                f"{root}.mlp.gate_up_proj.weight",
                f"{root}.mlp.down_proj.weight",
            }
        )
    for index in range(config.decoder_num_layers):
        root = f"model.audio_decoder.layers.{index}"
        weights.update(
            {
                f"{root}.self_attn.qkv_proj.weight",
                f"{root}.self_attn.o_proj.weight",
                f"{root}.mlp.gate_up_proj.weight",
                f"{root}.mlp.down_proj.weight",
            }
        )
    return frozenset(weights)


def minimax_music3_dav_layout(
    config: MiniMaxMusic3DavConfig = MINIMAX_MUSIC3_DAV_CONFIG,
) -> Mapping[str, tuple[int, ...]]:
    layout: dict[str, tuple[int, ...]] = {
        "dec_in_proj.weight": (config.hidden_channels, config.latent_channels // 2, 1),
        "dec_in_proj.bias": (config.hidden_channels,),
        "decoder.model.0.weight_g": (config.decoder_channels, 1, 1),
        "decoder.model.0.weight_v": (config.decoder_channels, config.hidden_channels, 7),
        "decoder.model.0.bias": (config.decoder_channels,),
    }
    channels = config.decoder_channels
    output_channels = channels
    for block, stride in enumerate(config.strides, start=1):
        input_channels = channels // (2 ** (block - 1))
        output_channels = channels // (2**block)
        root = f"decoder.model.{block}.block"
        layout[f"{root}.0.alpha"] = (1, input_channels, 1)
        layout[f"{root}.1.weight_g"] = (input_channels, 1, 1)
        layout[f"{root}.1.weight_v"] = (input_channels, output_channels, 2 * stride)
        layout[f"{root}.1.bias"] = (output_channels,)
        for unit in range(2, 5):
            unit_root = f"{root}.{unit}.block"
            layout[f"{unit_root}.0.alpha"] = (1, output_channels, 1)
            layout[f"{unit_root}.1.weight_g"] = (output_channels, 1, 1)
            layout[f"{unit_root}.1.weight_v"] = (output_channels, output_channels, 7)
            layout[f"{unit_root}.1.bias"] = (output_channels,)
            layout[f"{unit_root}.2.alpha"] = (1, output_channels, 1)
            layout[f"{unit_root}.3.weight_g"] = (output_channels, 1, 1)
            layout[f"{unit_root}.3.weight_v"] = (output_channels, output_channels, 1)
            layout[f"{unit_root}.3.bias"] = (output_channels,)
    layout.update(
        {
            "decoder.model.5.alpha": (1, output_channels, 1),
            "decoder.model.6.weight_g": (1, 1, 1),
            "decoder.model.6.weight_v": (1, output_channels, 7),
            "decoder.model.6.bias": (1,),
        }
    )
    return MappingProxyType(layout)


def detect_minimax_music3_text_config(
    geometries: Mapping[str, TensorGeometry],
) -> MiniMaxMusic3TextConfig:
    keys = frozenset(geometries)
    pruned = "model.embed_tokens_prefill.weight" in keys
    merged_qkv = "model.layers.0.self_attn.qkv_proj.weight" in keys
    merged_mlp = "model.layers.0.mlp.gate_up_proj.weight" in keys
    decoder_merged_qkv = "model.audio_decoder.layers.0.self_attn.qkv_proj.weight" in keys
    decoder_merged_mlp = "model.audio_decoder.layers.0.mlp.gate_up_proj.weight" in keys
    storage_format: Literal["floating", "int8_pruned"] = (
        "int8_pruned"
        if any(
            key != "tokenizer_json" and geometry.dtype is INT8
            for key, geometry in geometries.items()
        )
        else "floating"
    )
    config = MiniMaxMusic3TextConfig(
        pruned,
        merged_qkv,
        merged_mlp,
        decoder_merged_qkv,
        decoder_merged_mlp,
        storage_format,
    )
    layout = minimax_music3_text_layout(config)
    if set(geometries) != set(layout) or any(
        geometries[key].shape != shape for key, shape in layout.items()
    ):
        raise ValueError("source is not an exact MiniMax Music 3 text layout")
    int8_weights: frozenset[str] = (
        _minimax_music3_int8_text_weights(config)
        if config.storage_format == "int8_pruned"
        else frozenset()
    )
    for key, geometry in geometries.items():
        if key == "tokenizer_json":
            if geometry.dtype is not UINT8:
                raise ValueError("MiniMax Music 3 tokenizer_json must be uint8")
        elif key in int8_weights:
            if geometry.dtype is not INT8:
                raise ValueError(
                    "MiniMax Music 3 text tensor storage differs from the declared INT8 layout"
                )
        elif geometry.dtype.kind != "float":
            raise ValueError("MiniMax Music 3 text tensor storage differs from the declared layout")
    return config


@dataclass(frozen=True)
class MiniMaxMusic3Evidence:
    config: MiniMaxMusic3Config
    key_prefix: str
    matched_keys: tuple[str, ...]
    fields: Mapping[str, EvidenceValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched_keys", tuple(self.matched_keys))
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


def detect_minimax_music3(source: WeightSource) -> MiniMaxMusic3Evidence | None:
    layout = minimax_music3_diffusion_layout()
    source_geometries = {key: source.entry(key).geometry for key in source.keys()}
    for prefix in ("model.diffusion_model.", ""):
        scoped = {
            key[len(prefix) :]: geometry
            for key, geometry in source_geometries.items()
            if key.startswith(prefix)
        }
        geometries = {
            key: geometry
            for key, geometry in scoped.items()
            if not key.endswith((".weight_scale", ".comfy_quant"))
        }
        int8_weights = frozenset(
            key for key, geometry in geometries.items() if geometry.dtype is INT8
        )
        expected_quant = frozenset(
            artifact
            for weight in int8_weights
            for artifact in (
                weight.removesuffix(".weight") + ".weight_scale",
                weight.removesuffix(".weight") + ".comfy_quant",
            )
        )
        actual_quant = frozenset(set(scoped) - set(geometries))
        if any(not key.endswith(".weight") for key in int8_weights) or (
            actual_quant != expected_quant
        ):
            continue
        expected = frozenset(prefix + key for key in layout)
        if set(geometries) != set(layout):
            continue
        if any(
            geometries[key].shape != shape
            or geometries[key].dtype not in (FLOAT16, BFLOAT16, FLOAT32, INT8)
            for key, shape in layout.items()
        ):
            continue
        dtypes = frozenset(geometry.dtype for geometry in geometries.values())
        if dtypes not in (
            frozenset({FLOAT16}),
            frozenset({BFLOAT16}),
            frozenset({FLOAT32}),
            frozenset({FLOAT16, INT8}),
        ):
            continue
        if INT8 in dtypes:
            official_int8 = frozenset(
                f"diffusion_transformer.transformer.layers.{index}.{suffix}"
                for index in range(MINIMAX_MUSIC3_CONFIG.blocks)
                for suffix in (
                    "self_attn.to_qkv.weight",
                    "self_attn.to_out.weight",
                    "ff.ff.0.proj.weight",
                    "ff.ff.2.weight",
                )
            )
            if int8_weights != official_int8 or any(
                geometry.dtype is not (INT8 if key in official_int8 else FLOAT16)
                for key, geometry in geometries.items()
            ):
                continue
        return MiniMaxMusic3Evidence(
            MINIMAX_MUSIC3_CONFIG,
            prefix,
            tuple(sorted(expected)),
            {
                "blocks": MINIMAX_MUSIC3_CONFIG.blocks,
                "context_layers": MINIMAX_MUSIC3_CONFIG.context_layers,
                "context_width": MINIMAX_MUSIC3_CONFIG.context_width,
                "hidden_width": MINIMAX_MUSIC3_CONFIG.hidden_width,
                "key_prefix": prefix,
                "latent_channels": MINIMAX_MUSIC3_CONFIG.latent_channels,
            },
        )
    return None


MINIMAX_MUSIC3_DAV_DESCRIPTOR = CodecDescriptor(
    id="dinkster.minimax_music3_dav",
    display_name="MiniMax Music 3 DAV",
    kind="audio",
    latent=MINIMAX_MUSIC3_LATENT,
    supported_dtypes=frozenset({FLOAT32}),
    content_channels=2,
    supports_tiling=True,
    tiling=CodecTiling(
        decode_tile=(1536,),
        decode_overlap=(64,),
        encode_tile=(1536,),
        encode_overlap=(64,),
    ),
)


__all__ = [
    "AUDIO_CODE_OFFSET",
    "AUDIO_FRAMES_PER_SECOND",
    "C0_VOCAB_SIZE",
    "COMFYUI_SOURCE_COMMIT",
    "CONDITION_HOP_FRAMES",
    "DEFAULT_CFG_SCALE",
    "DEFAULT_TOP_K",
    "MAX_AUDIO_FRAMES",
    "MAX_CONDITION_FRAMES",
    "MAX_PROMPT_TOKENS",
    "MINIMAX_MUSIC3_CONFIG",
    "MINIMAX_MUSIC3_DAV_CONFIG",
    "MINIMAX_MUSIC3_DAV_DESCRIPTOR",
    "MINIMAX_MUSIC3_LATENT",
    "OFFICIAL_ARTIFACT_REVISION",
    "OFFICIAL_ARTIFACTS",
    "SPECIAL_TOKEN_IDS",
    "WORKFLOW_TEMPLATES_SOURCE_COMMIT",
    "MiniMaxMusic3Config",
    "MiniMaxMusic3DavConfig",
    "MiniMaxMusic3Evidence",
    "MiniMaxMusic3TextConfig",
    "build_music_prompt",
    "clean_music_caption",
    "derive_music_seed",
    "detect_minimax_music3",
    "detect_minimax_music3_text_config",
    "minimax_music3_dav_layout",
    "minimax_music3_diffusion_layout",
    "minimax_music3_latent_length",
    "minimax_music3_text_layout",
    "normalize_music_lyrics",
]
