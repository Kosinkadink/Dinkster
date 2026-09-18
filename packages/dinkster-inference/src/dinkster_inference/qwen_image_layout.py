"""Exact torch-free Qwen Image DiT layout and assembly planning.

The state layouts follow ComfyUI's Qwen Image modules at commit
``76135e557da1ec7dcb270160f01e597565e3e003`` and the exact staged profiles
detected by :mod:`dinkster_inference.qwen_image`. Planning accepts only one
complete bare or ``model.diffusion_model.``-prefixed diffusion source.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from .devices import FLOAT32, DType
from .qwen_image import (
    QWEN_IMAGE_CONFIG,
    QWEN_IMAGE_EDIT_2511_CONFIG,
    QWEN_IMAGE_LAYERED_CONFIG,
    QwenImageConfig,
)
from .weights import TensorGeometry, WeightSource

_PREFIXES = ("", "model.diffusion_model.")
_TIME_INPUT_WIDTH = 256
_MLP_WIDTH = 12288
_CONFIGS = (QWEN_IMAGE_CONFIG, QWEN_IMAGE_EDIT_2511_CONFIG, QWEN_IMAGE_LAYERED_CONFIG)


class QwenImageDiTAssemblyError(ValueError):
    """The source is not one exact unpacked Qwen Image DiT layout."""


def _linear(
    keys: dict[str, tuple[int, ...]],
    name: str,
    out_features: int,
    in_features: int,
) -> None:
    keys[f"{name}.weight"] = (out_features, in_features)
    keys[f"{name}.bias"] = (out_features,)


def _qwen_image_dit_keys(config: QwenImageConfig) -> dict[str, tuple[int, ...]]:
    keys: dict[str, tuple[int, ...]] = {"txt_norm.weight": (config.text_width,)}
    _linear(
        keys,
        "time_text_embed.timestep_embedder.linear_1",
        config.hidden_width,
        _TIME_INPUT_WIDTH,
    )
    _linear(
        keys,
        "time_text_embed.timestep_embedder.linear_2",
        config.hidden_width,
        config.hidden_width,
    )
    if config.use_additional_t_cond:
        keys["time_text_embed.addition_t_embedding.weight"] = (2, config.hidden_width)
    _linear(
        keys,
        "img_in",
        config.hidden_width,
        config.patchified_input_channels,
    )
    _linear(keys, "txt_in", config.hidden_width, config.text_width)

    for index in range(config.transformer_blocks):
        root = f"transformer_blocks.{index}"
        _linear(keys, f"{root}.img_mod.1", 6 * config.hidden_width, config.hidden_width)
        _linear(keys, f"{root}.img_mlp.net.0.proj", _MLP_WIDTH, config.hidden_width)
        _linear(keys, f"{root}.img_mlp.net.2", config.hidden_width, _MLP_WIDTH)
        _linear(keys, f"{root}.txt_mod.1", 6 * config.hidden_width, config.hidden_width)
        _linear(keys, f"{root}.txt_mlp.net.0.proj", _MLP_WIDTH, config.hidden_width)
        _linear(keys, f"{root}.txt_mlp.net.2", config.hidden_width, _MLP_WIDTH)
        for norm in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            keys[f"{root}.attn.{norm}.weight"] = (config.attention_head_dim,)
        for projection in (
            "to_q",
            "to_k",
            "to_v",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_out.0",
            "to_add_out",
        ):
            _linear(
                keys,
                f"{root}.attn.{projection}",
                config.hidden_width,
                config.hidden_width,
            )

    _linear(
        keys,
        "norm_out.linear",
        2 * config.hidden_width,
        config.hidden_width,
    )
    _linear(
        keys,
        "proj_out",
        config.patch[0] * config.patch[1] * config.output_latent_channels,
        config.hidden_width,
    )
    if config.default_ref_method == "index_timestep_zero":
        keys["__index_timestep_zero__"] = (0,)
    return keys


@dataclass(frozen=True)
class QwenImageDiTLayout:
    """Complete state and stream geometry for the exact base Qwen Image DiT."""

    config: QwenImageConfig
    keys: Mapping[str, tuple[int, ...]]
    depth: int = field(default=60, init=False)
    hidden_width: int = field(default=3072, init=False)
    attention_heads: int = field(default=24, init=False)
    attention_head_dim: int = field(default=128, init=False)
    text_width: int = field(default=3584, init=False)
    time_input_width: int = field(default=256, init=False)
    time_embed_width: int = field(default=3072, init=False)
    image_patch_width: int = field(default=64, init=False)
    output_patch_width: int = field(default=64, init=False)
    patch: tuple[int, int] = field(default=(2, 2), init=False)
    attention_kind: Literal["joint_full"] = field(default="joint_full", init=False)
    stream_order: tuple[str, str] = field(default=("text", "image"), init=False)

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.config), QwenImageConfig):
            raise TypeError("layout config must be QwenImageConfig")
        if self.config not in _CONFIGS:
            raise ValueError("layout config must be an exact Qwen Image profile")
        keys_obj = cast("object", self.keys)
        if not isinstance(keys_obj, Mapping):
            raise TypeError("layout keys must be a mapping")
        frozen = dict(cast("Mapping[str, tuple[int, ...]]", keys_obj))
        expected = _qwen_image_dit_keys(self.config)
        if frozen != expected:
            raise ValueError("keys must equal the exact Qwen Image DiT layout")
        object.__setattr__(self, "keys", MappingProxyType(frozen))


def qwen_image_dit_layout(
    config: QwenImageConfig = QWEN_IMAGE_CONFIG,
) -> QwenImageDiTLayout:
    """Return one exact Qwen Image DiT state layout."""
    if not isinstance(cast("object", config), QwenImageConfig):
        raise TypeError("config must be QwenImageConfig")
    if config not in _CONFIGS:
        raise ValueError("config must be an exact Qwen Image profile")
    return QwenImageDiTLayout(config, _qwen_image_dit_keys(config))


@dataclass(frozen=True)
class QwenImageDiTAssemblyPlan:
    """Deterministic complete-key plan for one Qwen Image source."""

    layout: QwenImageDiTLayout
    source_prefix: str
    keys: Mapping[str, str]
    dtypes: Mapping[str, DType]
    claims: tuple[str, ...]
    execution_format: Literal["bfloat16", "float32"]
    family_id: str = field(default="dinkster.qwen_image", init=False)
    source_role: str = field(default="diffusion", init=False)

    def __post_init__(self) -> None:
        if self.layout != qwen_image_dit_layout(self.layout.config):
            raise ValueError("assembly plan must use an exact Qwen Image DiT layout")
        if self.source_prefix not in _PREFIXES:
            raise ValueError("Qwen Image source prefix must be bare or model.diffusion_model.")
        keys = dict(self.keys)
        dtypes = dict(self.dtypes)
        expected = {key: self.source_prefix + key for key in self.layout.keys}
        if keys != expected:
            raise ValueError("Qwen Image plan must map every layout key exactly once")
        if set(dtypes) != set(expected):
            raise ValueError("Qwen Image plan dtypes must cover every layout key")
        if any(dtype.kind != "float" for dtype in dtypes.values()):
            raise QwenImageDiTAssemblyError("Qwen Image plan weights require floating storage")
        if not isinstance(cast("object", self.claims), tuple):
            raise TypeError("Qwen Image claims must be a tuple")
        if self.claims != tuple(sorted(expected.values())):
            raise ValueError("Qwen Image claims must exactly cover every source key")
        if self.execution_format not in ("bfloat16", "float32"):
            raise QwenImageDiTAssemblyError(
                "Qwen Image execution format must be bfloat16 or float32"
            )
        object.__setattr__(self, "keys", MappingProxyType(keys))
        object.__setattr__(self, "dtypes", MappingProxyType(dtypes))


def _source_prefix(source_keys: tuple[str, ...], layout: QwenImageDiTLayout) -> str:
    if not source_keys:
        raise QwenImageDiTAssemblyError("empty Qwen Image DiT source")
    actual = set(source_keys)
    bare = set(layout.keys)
    prefixed = {"model.diffusion_model." + key for key in layout.keys}
    if actual == bare:
        return ""
    if actual == prefixed:
        return "model.diffusion_model."
    has_bare = bool(actual & bare)
    has_prefixed = bool(actual & prefixed)
    if has_bare and has_prefixed:
        raise QwenImageDiTAssemblyError(
            "mixed bare and model.diffusion_model.-prefixed Qwen Image keys"
        )
    expected = prefixed if has_prefixed else bare
    missing = tuple(sorted(expected - actual))
    foreign = tuple(sorted(actual - expected))
    if missing:
        raise QwenImageDiTAssemblyError(
            "missing required Qwen Image DiT keys: " + ", ".join(missing[:3])
        )
    raise QwenImageDiTAssemblyError(
        "foreign or leftover Qwen Image DiT keys: " + ", ".join(foreign[:3])
    )


def _entry_geometry(source: WeightSource, key: str) -> TensorGeometry:
    try:
        entry = source.entry(key)
    except KeyError as error:
        raise QwenImageDiTAssemblyError(
            f"inconsistent Qwen Image source omitted advertised key {key}"
        ) from error
    if entry.key != key:
        raise QwenImageDiTAssemblyError(
            f"inconsistent Qwen Image source returned {entry.key} for {key}"
        )
    return entry.geometry


def plan_qwen_image_dit_assembly(
    source: WeightSource,
) -> QwenImageDiTAssemblyPlan:
    """Plan one exact bare or prefixed Qwen Image DiT without payload reads."""
    source_keys = tuple(source.keys())
    if len(source_keys) != len(set(source_keys)):
        raise QwenImageDiTAssemblyError("duplicate Qwen Image DiT source keys")
    layered_keys = {prefix + "time_text_embed.addition_t_embedding.weight" for prefix in _PREFIXES}
    edit_2511_keys = {prefix + "__index_timestep_zero__" for prefix in _PREFIXES}
    config = (
        QWEN_IMAGE_LAYERED_CONFIG
        if layered_keys.intersection(source_keys)
        else QWEN_IMAGE_EDIT_2511_CONFIG
        if edit_2511_keys.intersection(source_keys)
        else QWEN_IMAGE_CONFIG
    )
    layout = qwen_image_dit_layout(config)
    prefix = _source_prefix(source_keys, layout)
    model_to_source = {key: prefix + key for key in layout.keys}
    dtypes: dict[str, DType] = {}
    for model_key, source_key in model_to_source.items():
        geometry = _entry_geometry(source, source_key)
        expected = layout.keys[model_key]
        if geometry.shape != expected:
            raise QwenImageDiTAssemblyError(
                f"geometry mismatch for {source_key}: got {geometry.shape}, expected {expected}"
            )
        if geometry.dtype.kind != "float":
            raise QwenImageDiTAssemblyError(
                f"Qwen Image weight {source_key} requires floating storage, "
                f"got {geometry.dtype.name}"
            )
        dtypes[model_key] = geometry.dtype
    storage_dtypes = set(dtypes.values())
    execution_format: Literal["bfloat16", "float32"] = (
        "float32" if storage_dtypes == {FLOAT32} else "bfloat16"
    )
    return QwenImageDiTAssemblyPlan(
        layout=layout,
        source_prefix=prefix,
        keys=model_to_source,
        dtypes=dtypes,
        claims=tuple(sorted(source_keys)),
        execution_format=execution_format,
    )


__all__ = [
    "QwenImageDiTAssemblyError",
    "QwenImageDiTAssemblyPlan",
    "QwenImageDiTLayout",
    "plan_qwen_image_dit_assembly",
    "qwen_image_dit_layout",
]
