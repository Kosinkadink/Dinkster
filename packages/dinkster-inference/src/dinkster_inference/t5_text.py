"""T5/UMT5 text encoders: torch-free config, layout, detection.

The reference builds its T5 encoder from transformers-style config
JSON (comfy/text_encoders/t5.py T5 @ b78cec87, reading
comfy/text_encoders/t5_config_xxl.json for T5-XXL).
:class:`T5Config` carries exactly the fields that construction
consumes - the JSON's decoder/dropout/initializer/token-id fields are
inert there and are not modeled.

Reference behaviors pinned as facts of the layout:

- All T5 Linears are bias-free; layer norms are T5LayerNorm (RMS,
  weight only, no bias).
- ``model_type != "umt5"`` means ONE relative-attention-bias table,
  in block 0 only; every later block reuses block 0's computed bias
  (``past_bias`` threading in T5Stack). UMT5 carries a table in every
  block.
- Checkpoints ship ``encoder.embed_tokens.weight`` as a byte-duplicate
  of ``shared.weight``; the reference constructs only ``shared`` and
  loads with strict=False, so the key is optional here (shape-checked
  when present, never required).

Detection rejects rather than guesses: head count is not derivable
from fused projection shapes, so only the exact T5-XXL, UMT5-XXL,
and Hunyuan Image ByT5-small geometries are accepted, by full layout
comparison against :func:`t5_layout`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

from .prompt_tokens import TokenizerProfile
from .vendored import read_vendored
from .weights import TensorGeometry

#: comfy/text_encoders/t5.py activations @ b78cec87.
T5_TEXT_ACTIVATIONS = ("gelu_pytorch_tanh", "relu")

#: Shipped as a duplicate of shared.weight; the reference loads with
#: strict=False and never constructs the module, so absence is legal.
T5_TEXT_OPTIONAL_KEYS = frozenset({"encoder.embed_tokens.weight"})


class T5TextDetectError(ValueError):
    """The geometry mapping is not a supported T5 text-encoder
    checkpoint; the message names what was found instead."""


@dataclass(frozen=True)
class T5Config:
    """The construction-relevant subset of the reference config JSON.

    ``model_type`` stays explicit because the reference derives the
    relative-attention layout from it (anything but ``"umt5"`` means
    a single block-0 bias table). ``layer_norm_eps`` is the JSON's
    layer_norm_epsilon; the reference hardcodes T5LayerNorm's default
    1e-6, which the JSON happens to match - kept explicit so the pin
    is visible."""

    d_model: int
    d_ff: int
    d_kv: int
    num_heads: int
    num_layers: int
    vocab_size: int
    dense_act_fn: str
    is_gated_act: bool
    model_type: str = "t5"
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        for field in (
            "d_model",
            "d_ff",
            "d_kv",
            "num_heads",
            "num_layers",
            "vocab_size",
            "relative_attention_num_buckets",
            "relative_attention_max_distance",
        ):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if self.dense_act_fn not in T5_TEXT_ACTIVATIONS:
            raise ValueError(
                f"dense_act_fn must be one of {T5_TEXT_ACTIVATIONS}, got {self.dense_act_fn!r}"
            )
        if self.model_type not in ("t5", "umt5"):
            raise ValueError("model_type must be 't5' or 'umt5'")
        if self.layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be positive")

    @property
    def inner_dim(self) -> int:
        """The fused attention projection width, ``d_kv * num_heads``
        (the reference never reads head width from shapes)."""
        return self.d_kv * self.num_heads


#: comfy/text_encoders/t5_config_xxl.json @ b78cec87.
T5_XXL_CONFIG = T5Config(
    d_model=4096,
    d_ff=10240,
    d_kv=64,
    num_heads=64,
    num_layers=24,
    vocab_size=32128,
    dense_act_fn="gelu_pytorch_tanh",
    is_gated_act=True,
)

#: comfy/text_encoders/umt5_config_xxl.json @ b78cec87.
UMT5_XXL_CONFIG = T5Config(
    d_model=4096,
    d_ff=10240,
    d_kv=64,
    num_heads=64,
    num_layers=24,
    vocab_size=256384,
    dense_act_fn="gelu_pytorch_tanh",
    is_gated_act=True,
    model_type="umt5",
)

BYT5_SMALL_GLYPH_CONFIG = T5Config(
    d_model=1472,
    d_ff=3584,
    d_kv=64,
    num_heads=6,
    num_layers=12,
    vocab_size=1510,
    dense_act_fn="gelu_pytorch_tanh",
    is_gated_act=True,
)

#: comfy/text_encoders/wan.py UMT5XXlTokenizer @ b78cec87: no BOS,
#: EOS 1, pad 0, one unbounded chunk padded to at least 512 tokens.
UMT5_XXL_WAN_PROFILE = TokenizerProfile(
    max_length=99999999,
    start_token=None,
    end_token=1,
    pad_token=0,
    pad_to_max_length=False,
    min_length=512,
)

#: T5-XL/base and the non-gated wi-only "old" T5-XXL remain outside
#: the exact detected geometries.
KNOWN_T5_CONFIGS = (T5_XXL_CONFIG, UMT5_XXL_CONFIG, BYT5_SMALL_GLYPH_CONFIG)

#: sha256 of the uncompressed vendored UMT5 SentencePiece model
#: (4,548,313 bytes). Byte-identical between the ``spiece_model``
#: tensor embedded in Comfy-Org's Wan repackaged text encoder
#: (https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged
#: split_files/text_encoders/umt5_xxl_fp16.safetensors) and the
#: upstream tokenizer (https://huggingface.co/google/umt5-xxl
#: spiece.model), so vendoring reproduces exactly what safetensors
#: checkpoints carry inline.
UMT5_SPIECE_SHA256 = "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"


@lru_cache(maxsize=1)
def load_umt5_spiece() -> bytes:
    """The vendored UMT5-XXL SentencePiece model bytes, hash-verified.

    GGUF text-encoder files carry no ``spiece_model`` tensor, so Wan
    assembly reads the tokenizer from here instead of the checkpoint."""
    return read_vendored("umt5_spiece.model.gz", UMT5_SPIECE_SHA256)


def t5_layout(config: T5Config) -> dict[str, tuple[int, ...]]:
    """The exact state-dict key/shape listing the reference T5
    encoder produces for ``config`` - the single source both
    detection (compare a header against it) and the torch tests
    (compare the constructed module against the executed reference)
    consume. Optional checkpoint extras (T5_TEXT_OPTIONAL_KEYS) are
    NOT part of the module layout and are not listed."""
    model = config.d_model
    inner = config.inner_dim
    ff = config.d_ff
    layout: dict[str, tuple[int, ...]] = {
        "shared.weight": (config.vocab_size, model),
        "encoder.final_layer_norm.weight": (model,),
    }
    for i in range(config.num_layers):
        attn = f"encoder.block.{i}.layer.0.SelfAttention."
        layout[f"{attn}q.weight"] = (inner, model)
        layout[f"{attn}k.weight"] = (inner, model)
        layout[f"{attn}v.weight"] = (inner, model)
        layout[f"{attn}o.weight"] = (model, inner)
        if config.model_type == "umt5" or i == 0:
            layout[f"{attn}relative_attention_bias.weight"] = (
                config.relative_attention_num_buckets,
                config.num_heads,
            )
        layout[f"encoder.block.{i}.layer.0.layer_norm.weight"] = (model,)
        ff_prefix = f"encoder.block.{i}.layer.1.DenseReluDense."
        if config.is_gated_act:
            layout[f"{ff_prefix}wi_0.weight"] = (ff, model)
            layout[f"{ff_prefix}wi_1.weight"] = (ff, model)
        else:
            layout[f"{ff_prefix}wi.weight"] = (ff, model)
        layout[f"{ff_prefix}wo.weight"] = (model, ff)
        layout[f"encoder.block.{i}.layer.1.layer_norm.weight"] = (model,)
    return layout


def detect_t5_config(
    geometries: Mapping[str, TensorGeometry],
) -> T5Config:
    """Classify a T5-scoped header (``encoder.*``/``shared.*`` keys,
    any checkpoint prefix already stripped) as one exact supported
    T5-family layout, or refuse loudly. Dtypes are ignored -
    checkpoints legitimately ship fp16/bf16/fp8. Quantization metadata
    must be removed by the component-scoped quantization classifier
    before detection."""
    if not geometries:
        raise T5TextDetectError("empty state dict header")
    shared = geometries.get("shared.weight")
    if shared is None:
        raise T5TextDetectError("not a T5 text-encoder state dict (no shared.weight)")
    if len(shared.shape) != 2:
        raise T5TextDetectError(f"shared.weight has rank {len(shared.shape)}, expected 2")
    vocab, model = shared.shape
    config = next(
        (
            candidate
            for candidate in KNOWN_T5_CONFIGS
            if candidate.d_model == model and candidate.vocab_size == vocab
        ),
        None,
    )
    if config is None:
        raise T5TextDetectError(
            f"unknown T5 geometry: shared embedding {vocab}x{model};"
            " head count is not derivable from fused projection shapes,"
            " so only T5-XXL, UMT5-XXL, and the registered exact ByT5-small"
            " profile are accepted"
        )

    layout = t5_layout(config)
    problems: list[str] = []
    for key, shape in layout.items():
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
        elif found.shape != shape:
            problems.append(f"{key}: expected shape {shape}, found {found.shape}")
    for key in sorted(set(geometries) - set(layout)):
        if key in T5_TEXT_OPTIONAL_KEYS:
            if geometries[key].shape != (config.vocab_size, config.d_model):
                problems.append(
                    f"{key}: expected shape"
                    f" {(config.vocab_size, config.d_model)},"
                    f" found {geometries[key].shape}"
                )
        else:
            problems.append(f"unexpected key {key}")
    if problems:
        shown = "; ".join(problems[:6])
        more = len(problems) - 6
        if more > 0:
            shown += f"; and {more} more"
        raise T5TextDetectError(
            f"geometry does not match the {config.model_type.upper()}-XXL layout: {shown}"
        )
    return config


__all__ = [
    "BYT5_SMALL_GLYPH_CONFIG",
    "KNOWN_T5_CONFIGS",
    "T5Config",
    "T5TextDetectError",
    "T5_TEXT_ACTIVATIONS",
    "T5_TEXT_OPTIONAL_KEYS",
    "T5_XXL_CONFIG",
    "UMT5_SPIECE_SHA256",
    "UMT5_XXL_CONFIG",
    "UMT5_XXL_WAN_PROFILE",
    "detect_t5_config",
    "load_umt5_spiece",
    "t5_layout",
]
