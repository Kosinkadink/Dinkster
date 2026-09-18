"""Torch-free classic SD1.5 ControlNet geometry and application records.

Only the complete, unpacked classic SD1.5 architecture from ComfyUI
``comfy/cldm/cldm.py`` and its finite Diffusers spelling conversion are
accepted. Detection is deliberately stricter than ComfyUI's non-strict load:
missing, extra, mixed, packed, colliding, and geometry-drifted headers refuse
before any payload access. This module does not provide ControlNet execution.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

from .conditioning import PayloadReference, PercentRange
from .devices import FLOAT16, FLOAT32
from .unet import SD15_UNET_CONFIG, SDXL_UNET_CONFIG, UNetConfig, unet_layout
from .weights import TensorGeometry

ControlNetSourceLayout: TypeAlias = Literal["canonical", "diffusers"]
SDControlModeToken: TypeAlias = Literal[
    "openpose",
    "depth",
    "hed",
    "pidi",
    "scribble",
    "ted",
    "canny",
    "lineart",
    "anime_lineart",
    "mlsd",
    "normal",
    "segment",
    "tile",
    "repaint",
]
_CHILD_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_SD_CONTROL_MODE_TOKENS = frozenset(
    {
        "openpose",
        "depth",
        "hed",
        "pidi",
        "scribble",
        "ted",
        "canny",
        "lineart",
        "anime_lineart",
        "mlsd",
        "normal",
        "segment",
        "tile",
        "repaint",
    }
)
SD_CONTROL_MODE_INDEX: Mapping[SDControlModeToken, int] = MappingProxyType(
    {
        "openpose": 0,
        "depth": 1,
        "hed": 2,
        "pidi": 2,
        "scribble": 2,
        "ted": 2,
        "canny": 3,
        "lineart": 3,
        "anime_lineart": 3,
        "mlsd": 3,
        "normal": 4,
        "segment": 5,
        "tile": 6,
        "repaint": 7,
    }
)

SD15_CONTROL_RESIDUAL_SITES = tuple(
    f"sd15.unet.input_skip.{index:02d}.v1" for index in range(12)
) + ("sd15.unet.middle.v1",)
SDXL_CONTROL_RESIDUAL_SITES = tuple(
    f"sdxl.unet.input_skip.{index:02d}.v1" for index in range(9)
) + ("sdxl.unet.middle.v1",)


class ControlNetDetectError(ValueError):
    """A header is not the one supported classic SD1.5 ControlNet layout."""


@dataclass(frozen=True)
class SDControlMode:
    """One semantic selector for an SD control-provider family."""

    provider: Literal["sdxl-controlnet-union"]
    token: SDControlModeToken

    def __post_init__(self) -> None:
        if type(self.provider) is not str:
            raise TypeError("SD control mode provider must be a string")
        if self.provider != "sdxl-controlnet-union":
            raise ValueError("unknown SD control mode provider")
        if type(self.token) is not str:
            raise TypeError("SD control mode token must be a string")
        if self.token not in _SD_CONTROL_MODE_TOKENS:
            raise ValueError("unknown SD control mode token")


@dataclass(frozen=True)
class SDXLControlNetConfig:
    """Construction facts for one classic SDXL ControlNet artifact."""

    base: UNetConfig = SDXL_UNET_CONFIG
    hint_channels: int = 3

    def __post_init__(self) -> None:
        if self.base != SDXL_UNET_CONFIG:
            raise ValueError("only the standard SDXL ControlNet geometry is supported")
        if type(self.hint_channels) is not int or self.hint_channels < 1:
            raise ValueError("SDXL ControlNet hint channels must be a positive integer")


@dataclass(frozen=True)
class SDXLControlNetUnionConfig:
    """Construction facts for one xinsir SDXL ControlNet Union artifact."""

    base: UNetConfig = SDXL_UNET_CONFIG
    hint_channels: int = 3
    mode_capacity: Literal[6, 8] = 8

    def __post_init__(self) -> None:
        if self.base != SDXL_UNET_CONFIG or self.hint_channels != 3:
            raise ValueError("only the xinsir SDXL ControlNet Union geometry is supported")
        if type(self.mode_capacity) is not int or self.mode_capacity not in (6, 8):
            raise ValueError("SDXL ControlNet Union mode capacity must be exactly 6 or 8")


@dataclass(frozen=True)
class SDXLControlLoRAConfig:
    """Construction facts for SDXL Control-LoRA."""

    base: UNetConfig = SDXL_UNET_CONFIG
    hint_channels: int = 3
    rank: int = 128

    def __post_init__(self) -> None:
        if self.base != SDXL_UNET_CONFIG or self.hint_channels != 3:
            raise ValueError(
                "SDXL Control-LoRA requires SDXL base geometry and three hint channels"
            )
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("SDXL Control-LoRA rank must be a positive integer")


_SDXL_CONTROL_LORA_CONFIG = SDXLControlLoRAConfig()


@dataclass(frozen=True)
class SD15ControlNetConfig:
    """Construction facts for the one accepted classic SD1.5 ControlNet."""

    in_channels: int = 4
    model_channels: int = 320
    hint_channels: int = 3
    channel_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: tuple[int, ...] = (2, 2, 2, 2)
    transformer_depth: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 0, 0)
    transformer_depth_middle: int = 1
    context_dim: int = 768
    num_heads: int = 8

    def __post_init__(self) -> None:
        expected = (
            4,
            320,
            3,
            (1, 2, 4, 4),
            (2, 2, 2, 2),
            (1, 1, 1, 1, 1, 1, 0, 0),
            1,
            768,
            8,
        )
        actual = (
            self.in_channels,
            self.model_channels,
            self.hint_channels,
            self.channel_mult,
            self.num_res_blocks,
            self.transformer_depth,
            self.transformer_depth_middle,
            self.context_dim,
            self.num_heads,
        )
        if actual != expected:
            raise ValueError(
                f"only the classic SD1.5 ControlNet geometry is supported; got {actual!r}"
            )


@dataclass(frozen=True)
class ControlNetLayout:
    """Exact canonical model keys and block-output channels."""

    config: SD15ControlNetConfig
    keys: Mapping[str, tuple[int, ...]]
    zero_conv_channels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.config), SD15ControlNetConfig):
            raise TypeError("ControlNet layout config must be SD15ControlNetConfig")
        keys_obj = cast("object", self.keys)
        if not isinstance(keys_obj, Mapping):
            raise TypeError("ControlNet layout keys must be a mapping")
        frozen: dict[str, tuple[int, ...]] = {}
        for key, shape in cast("Mapping[object, object]", keys_obj).items():
            if not isinstance(key, str) or not key:
                raise ValueError("ControlNet layout keys must be non-empty strings")
            if not isinstance(shape, tuple):
                raise ValueError("ControlNet layout shapes must be tuples")
            shape_values = cast("tuple[object, ...]", shape)
            if any(type(dim) is not int or dim < 1 for dim in shape_values):
                raise ValueError("ControlNet layout shapes must contain positive integers")
            frozen[key] = cast("tuple[int, ...]", shape)
        object.__setattr__(self, "keys", MappingProxyType(frozen))
        if (
            not isinstance(cast("object", self.zero_conv_channels), tuple)
            or len(self.zero_conv_channels) != 12
            or any(
                type(channels) is not int or channels < 1 for channels in self.zero_conv_channels
            )
        ):
            raise ValueError("ControlNet layout needs 12 positive zero-conv channel widths")
        expected_keys, expected_channels = _sd15_controlnet_geometry(self.config)
        if frozen != expected_keys or self.zero_conv_channels != expected_channels:
            raise ValueError("ControlNet layout must equal the exact classic SD1.5 layout")


def _sd15_controlnet_geometry(
    config: SD15ControlNetConfig,
) -> tuple[dict[str, tuple[int, ...]], tuple[int, ...]]:
    shared = unet_layout(SD15_UNET_CONFIG)
    keys = {
        key: shape
        for key, shape in shared.items()
        if key.startswith(("time_embed.", "input_blocks.", "middle_block."))
    }
    hint = (
        (0, 16, config.hint_channels),
        (2, 16, 16),
        (4, 32, 16),
        (6, 32, 32),
        (8, 96, 32),
        (10, 96, 96),
        (12, 256, 96),
        (14, config.model_channels, 256),
    )
    for index, out_channels, in_channels in hint:
        keys[f"input_hint_block.{index}.weight"] = (
            out_channels,
            in_channels,
            3,
            3,
        )
        keys[f"input_hint_block.{index}.bias"] = (out_channels,)

    zero_conv_channels = (
        320,
        320,
        320,
        320,
        640,
        640,
        640,
        1280,
        1280,
        1280,
        1280,
        1280,
    )
    for index, channels in enumerate(zero_conv_channels):
        keys[f"zero_convs.{index}.0.weight"] = (channels, channels, 1, 1)
        keys[f"zero_convs.{index}.0.bias"] = (channels,)
    keys["middle_block_out.0.weight"] = (1280, 1280, 1, 1)
    keys["middle_block_out.0.bias"] = (1280,)
    return keys, zero_conv_channels


def sd15_controlnet_layout(config: SD15ControlNetConfig) -> ControlNetLayout:
    """Return the exact canonical classic SD1.5 ControlNet state layout."""
    if not isinstance(cast("object", config), SD15ControlNetConfig):
        raise TypeError("config must be SD15ControlNetConfig")
    keys, zero_conv_channels = _sd15_controlnet_geometry(config)
    return ControlNetLayout(config, keys, zero_conv_channels)


_RESNET_TO_DIFFUSERS = MappingProxyType(
    {
        "in_layers.0": "norm1",
        "in_layers.2": "conv1",
        "emb_layers.1": "time_emb_proj",
        "out_layers.0": "norm2",
        "out_layers.3": "conv2",
        "skip_connection": "conv_shortcut",
    }
)
_INPUT_BLOCK_TO_DIFFUSERS = MappingProxyType(
    {
        1: (0, 0),
        2: (0, 1),
        4: (1, 0),
        5: (1, 1),
        7: (2, 0),
        8: (2, 1),
        10: (3, 0),
        11: (3, 1),
    }
)


def _diffusers_key(canonical: str) -> str:
    basics = (
        ("time_embed.0.", "time_embedding.linear_1."),
        ("time_embed.2.", "time_embedding.linear_2."),
        ("input_blocks.0.0.", "conv_in."),
        ("middle_block_out.0.", "controlnet_mid_block."),
    )
    for old, new in basics:
        if canonical.startswith(old):
            return new + canonical[len(old) :]
    if canonical.startswith("input_hint_block."):
        index_text, parameter = canonical.removeprefix("input_hint_block.").split(".", 1)
        index = int(index_text)
        if index == 0:
            return f"controlnet_cond_embedding.conv_in.{parameter}"
        if index == 14:
            return f"controlnet_cond_embedding.conv_out.{parameter}"
        return f"controlnet_cond_embedding.blocks.{index // 2 - 1}.{parameter}"
    if canonical.startswith("zero_convs."):
        index, _, parameter = canonical.removeprefix("zero_convs.").split(".", 2)
        return f"controlnet_down_blocks.{index}.{parameter}"
    if canonical.startswith("input_blocks."):
        parts = canonical.split(".")
        block = int(parts[1])
        rest = ".".join(parts[3:])
        if block in (3, 6, 9):
            return f"down_blocks.{block // 3 - 1}.downsamplers.0.conv." + rest.removeprefix("op.")
        level, resnet = _INPUT_BLOCK_TO_DIFFUSERS[block]
        if parts[2] == "1":
            return f"down_blocks.{level}.attentions.{resnet}.{rest}"
        return _diffusers_resnet_key(f"down_blocks.{level}.resnets.{resnet}", rest)
    if canonical.startswith("middle_block."):
        parts = canonical.split(".")
        block = int(parts[1])
        rest = ".".join(parts[2:])
        if block == 1:
            return f"mid_block.attentions.0.{rest}"
        return _diffusers_resnet_key(f"mid_block.resnets.{0 if block == 0 else 1}", rest)
    raise AssertionError(f"canonical ControlNet map does not cover {canonical!r}")


def _diffusers_resnet_key(prefix: str, rest: str) -> str:
    for canonical, diffusers in _RESNET_TO_DIFFUSERS.items():
        if rest.startswith(canonical + "."):
            return prefix + "." + diffusers + rest[len(canonical) :]
    raise AssertionError(f"canonical ControlNet resnet map does not cover {rest!r}")


def normalize_sd15_controlnet(
    geometries: Mapping[str, TensorGeometry],
) -> tuple[SD15ControlNetConfig, ControlNetSourceLayout, dict[str, str]]:
    if not geometries:
        raise ControlNetDetectError("empty state dict header")
    keys = frozenset(geometries)
    if any(key.startswith("control_model.") for key in keys):
        raise ControlNetDetectError(
            "packed control_model.* checkpoints are unsupported without"
            " truthful unpacked logical geometry"
        )
    canonical_prefixes = (
        "time_embed.",
        "input_blocks.",
        "middle_block.",
        "input_hint_block.",
        "zero_convs.",
        "middle_block_out.",
    )
    diffusers_prefixes = (
        "time_embedding.",
        "conv_in.",
        "down_blocks.",
        "mid_block.",
        "controlnet_cond_embedding.",
        "controlnet_down_blocks.",
        "controlnet_mid_block.",
    )
    has_canonical = any(key.startswith(canonical_prefixes) for key in keys)
    has_diffusers = any(key.startswith(diffusers_prefixes) for key in keys)
    if has_canonical and has_diffusers:
        raise ControlNetDetectError(
            "mixed canonical and diffusers-format ControlNet keys; refusing ambiguous layout"
        )
    marker = "controlnet_cond_embedding.conv_in.weight"
    if has_diffusers and marker not in keys:
        raise ControlNetDetectError(
            f"incomplete or unknown diffusers-format ControlNet; missing {marker}"
        )

    config = SD15ControlNetConfig()
    layout = sd15_controlnet_layout(config)
    if has_diffusers:
        source_to_model = {_diffusers_key(key): key for key in layout.keys}
        source_layout: ControlNetSourceLayout = "diffusers"
    else:
        source_to_model = {key: key for key in layout.keys}
        source_layout = "canonical"
    expected_sources = set(source_to_model)
    missing = tuple(sorted(expected_sources - keys))
    leftover = tuple(sorted(keys - expected_sources))
    if missing:
        raise ControlNetDetectError("missing required ControlNet keys: " + ", ".join(missing[:3]))
    if leftover:
        raise ControlNetDetectError(
            "leftover or foreign ControlNet keys: " + ", ".join(leftover[:3])
        )
    if len(set(source_to_model.values())) != len(source_to_model):
        raise ControlNetDetectError("ambiguous ControlNet key map collision")
    for source_key, model_key in source_to_model.items():
        geometry = geometries[source_key]
        if geometry.dtype.kind != "float":
            raise ControlNetDetectError(
                f"ControlNet weight {source_key} requires floating storage,"
                f" got {geometry.dtype.name}"
            )
        actual = geometry.shape
        expected = layout.keys[model_key]
        if actual != expected:
            raise ControlNetDetectError(
                f"geometry mismatch for {source_key}: got {actual}, expected {expected}"
            )
    return config, source_layout, source_to_model


def detect_sd15_controlnet(
    geometries: Mapping[str, TensorGeometry],
) -> SD15ControlNetConfig:
    """Detect one exact canonical or Diffusers classic SD1.5 ControlNet."""
    config, _, _ = normalize_sd15_controlnet(geometries)
    return config


def sdxl_control_lora_layout(
    config: SDXLControlLoRAConfig = _SDXL_CONTROL_LORA_CONFIG,
) -> Mapping[str, tuple[int, ...]]:
    """Return the Stability AI Control-LoRA tensor layout for the declared rank."""
    if type(config) is not SDXLControlLoRAConfig:
        raise TypeError("config must be an exact SDXLControlLoRAConfig")
    base = unet_layout(config.base)
    branch_prefixes = ("time_embed.", "label_emb.", "input_blocks.", "middle_block.")
    layout: dict[str, tuple[int, ...]] = {"lora_controlnet": (0,)}
    for key, shape in base.items():
        if not key.startswith(branch_prefixes):
            continue
        if key.endswith(".weight") and len(shape) in (2, 4):
            module = key.removesuffix(".weight")
            rank = 4 if module == "input_blocks.0.0" else config.rank
            layout[f"{module}.down"] = (rank, shape[1], *shape[2:])
            layout[f"{module}.up"] = (shape[0], rank, *((1, 1) if len(shape) == 4 else ()))
        else:
            layout[key] = shape

    hint = (
        (0, 16, config.hint_channels),
        (2, 16, 16),
        (4, 32, 16),
        (6, 32, 32),
        (8, 96, 32),
        (10, 96, 96),
        (12, 256, 96),
        (14, config.base.model_channels, 256),
    )
    for index, out_channels, in_channels in hint:
        layout[f"input_hint_block.{index}.weight"] = (out_channels, in_channels, 3, 3)
        layout[f"input_hint_block.{index}.bias"] = (out_channels,)

    channels = config.base.model_channels
    input_channels = [channels]
    for level, multiplier in enumerate(config.base.channel_mult):
        for _ in range(config.base.num_res_blocks[level]):
            channels = multiplier * config.base.model_channels
            input_channels.append(channels)
        if level != len(config.base.channel_mult) - 1:
            input_channels.append(channels)
    for index, width in enumerate(input_channels):
        layout[f"zero_convs.{index}.0.weight"] = (width, width, 1, 1)
        layout[f"zero_convs.{index}.0.bias"] = (width,)
    layout["middle_block_out.0.weight"] = (channels, channels, 1, 1)
    layout["middle_block_out.0.bias"] = (channels,)
    return MappingProxyType(layout)


def detect_sdxl_control_lora(
    geometries: Mapping[str, TensorGeometry],
) -> SDXLControlLoRAConfig:
    """Detect SDXL Control-LoRA geometry, deriving rank from the weights."""
    rank_source = geometries.get("time_embed.0.down")
    if rank_source is None:
        raise ControlNetDetectError("missing required Control-LoRA key: time_embed.0.down")
    if len(rank_source.shape) != 2 or rank_source.shape[0] <= 0:
        raise ControlNetDetectError("Control-LoRA geometry mismatch for time_embed.0.down")
    config = SDXLControlLoRAConfig(rank=rank_source.shape[0])
    expected = sdxl_control_lora_layout(config)
    keys = set(geometries)
    missing = sorted(set(expected) - keys)
    leftover = sorted(keys - set(expected))
    if missing:
        raise ControlNetDetectError("missing required Control-LoRA keys: " + ", ".join(missing[:3]))
    if leftover:
        raise ControlNetDetectError(
            "leftover or foreign Control-LoRA keys: " + ", ".join(leftover[:3])
        )
    for key, shape in expected.items():
        geometry = geometries[key]
        expected_dtype = FLOAT32 if key == "lora_controlnet" else FLOAT16
        if geometry.shape != shape or geometry.dtype != expected_dtype:
            raise ControlNetDetectError(
                f"Control-LoRA geometry mismatch for {key}: got {geometry.dtype.name}"
                f" {geometry.shape}, expected {expected_dtype.name} {shape}"
            )
    return config


def _sdxl_controlnet_diffusers_source_key(canonical: str) -> str:
    if canonical.startswith("label_emb.0.0."):
        return "add_embedding.linear_1." + canonical.removeprefix("label_emb.0.0.")
    if canonical.startswith("label_emb.0.2."):
        return "add_embedding.linear_2." + canonical.removeprefix("label_emb.0.2.")
    if canonical.startswith("input_hint_block."):
        index_text, parameter = canonical.removeprefix("input_hint_block.").split(".", 1)
        index = int(index_text)
        if index == 0:
            return f"controlnet_cond_embedding.conv_in.{parameter}"
        if index == 14:
            return f"controlnet_cond_embedding.conv_out.{parameter}"
        return f"controlnet_cond_embedding.blocks.{index // 2 - 1}.{parameter}"
    if canonical.startswith("zero_convs."):
        index, _, parameter = canonical.removeprefix("zero_convs.").split(".", 2)
        return f"controlnet_down_blocks.{index}.{parameter}"
    if canonical.startswith("middle_block_out.0."):
        return "controlnet_mid_block." + canonical.removeprefix("middle_block_out.0.")
    if canonical.startswith("transformer_layes.0.attn.in_proj."):
        return canonical.replace(".in_proj.", ".in_proj_")
    if canonical.startswith(
        ("task_embedding", "transformer_layes.", "spatial_ch_projs.", "control_add_embedding.")
    ):
        return canonical
    return _diffusers_key(canonical)


def sdxl_controlnet_layout(config: SDXLControlNetConfig) -> Mapping[str, tuple[int, ...]]:
    """Return the exact canonical classic SDXL ControlNet layout."""
    if type(config) is not SDXLControlNetConfig:
        raise TypeError("config must be an exact SDXLControlNetConfig")
    union_layout = sdxl_controlnet_union_layout(
        SDXLControlNetUnionConfig(base=config.base, hint_channels=3)
    )
    union_prefixes = (
        "task_embedding",
        "transformer_layes.",
        "spatial_ch_projs.",
        "control_add_embedding.",
    )
    layout = {
        key: shape for key, shape in union_layout.items() if not key.startswith(union_prefixes)
    }
    if config.hint_channels != 3:
        layout["input_hint_block.0.weight"] = (16, config.hint_channels, 3, 3)
    return MappingProxyType(layout)


def normalize_sdxl_controlnet(
    geometries: Mapping[str, TensorGeometry],
) -> tuple[SDXLControlNetConfig, dict[str, str]]:
    """Detect native, prefixed-native, or Diffusers classic SDXL ControlNet weights."""
    keys = set(geometries)
    prefix = "control_model."
    if keys and all(key.startswith(prefix) for key in keys):
        canonical_geometries = {
            key.removeprefix(prefix): value for key, value in geometries.items()
        }
        source_for = {key.removeprefix(prefix): key for key in geometries}
    else:
        canonical_geometries = dict(geometries)
        source_for = {key: key for key in geometries}

    hint_key = "input_hint_block.0.weight"
    diffusers_hint_key = "controlnet_cond_embedding.conv_in.weight"
    hint_geometry = canonical_geometries.get(hint_key)
    if hint_geometry is None:
        hint_geometry = canonical_geometries.get(diffusers_hint_key)
    if hint_geometry is None or len(hint_geometry.shape) != 4:
        raise ControlNetDetectError("missing or invalid SDXL ControlNet hint input")
    config = SDXLControlNetConfig(hint_channels=hint_geometry.shape[1])
    layout = sdxl_controlnet_layout(config)
    if hint_key in canonical_geometries:
        canonical_to_source = {key: source_for.get(key, key) for key in layout}
    else:
        canonical_to_source = {key: _sdxl_controlnet_diffusers_source_key(key) for key in layout}
    source_to_model = {source: model for model, source in canonical_to_source.items()}
    missing = sorted(set(source_to_model) - keys)
    leftover = sorted(keys - set(source_to_model))
    if missing:
        raise ControlNetDetectError(
            "missing required SDXL ControlNet keys: " + ", ".join(missing[:3])
        )
    if leftover:
        raise ControlNetDetectError(
            "leftover or foreign SDXL ControlNet keys: " + ", ".join(leftover[:3])
        )
    if len(set(source_to_model.values())) != len(source_to_model):
        raise ControlNetDetectError("ambiguous SDXL ControlNet key map collision")
    for source, model in source_to_model.items():
        geometry = geometries[source]
        expected = layout[model]
        if geometry.dtype.kind != "float":
            raise ControlNetDetectError(
                f"SDXL ControlNet weight {source} requires floating storage,"
                f" got {geometry.dtype.name}"
            )
        if geometry.shape != expected:
            raise ControlNetDetectError(
                f"SDXL ControlNet geometry mismatch for {source}: got {geometry.dtype.name}"
                f" {geometry.shape}, expected floating {expected}"
            )
    return config, source_to_model


def sdxl_controlnet_union_layout(
    config: SDXLControlNetUnionConfig,
) -> Mapping[str, tuple[int, ...]]:
    """Return the exact canonical xinsir SDXL ControlNet Union layout."""
    if type(config) is not SDXLControlNetUnionConfig:
        raise TypeError("config must be an exact SDXLControlNetUnionConfig")
    base = unet_layout(config.base)
    prefixes = ("time_embed.", "label_emb.", "input_blocks.", "middle_block.")
    layout = {key: shape for key, shape in base.items() if key.startswith(prefixes)}
    hint = (
        (0, 16, config.hint_channels),
        (2, 16, 16),
        (4, 32, 16),
        (6, 32, 32),
        (8, 96, 32),
        (10, 96, 96),
        (12, 256, 96),
        (14, config.base.model_channels, 256),
    )
    for index, out_channels, in_channels in hint:
        layout[f"input_hint_block.{index}.weight"] = (out_channels, in_channels, 3, 3)
        layout[f"input_hint_block.{index}.bias"] = (out_channels,)
    channels = (320, 320, 320, 320, 640, 640, 640, 1280, 1280)
    for index, width in enumerate(channels):
        layout[f"zero_convs.{index}.0.weight"] = (width, width, 1, 1)
        layout[f"zero_convs.{index}.0.bias"] = (width,)
    layout["middle_block_out.0.weight"] = (1280, 1280, 1, 1)
    layout["middle_block_out.0.bias"] = (1280,)
    layout.update(
        {
            "task_embedding": (config.mode_capacity, 320),
            "transformer_layes.0.attn.in_proj.weight": (960, 320),
            "transformer_layes.0.attn.in_proj.bias": (960,),
            "transformer_layes.0.attn.out_proj.weight": (320, 320),
            "transformer_layes.0.attn.out_proj.bias": (320,),
            "transformer_layes.0.ln_1.weight": (320,),
            "transformer_layes.0.ln_1.bias": (320,),
            "transformer_layes.0.ln_2.weight": (320,),
            "transformer_layes.0.ln_2.bias": (320,),
            "transformer_layes.0.mlp.c_fc.weight": (1280, 320),
            "transformer_layes.0.mlp.c_fc.bias": (1280,),
            "transformer_layes.0.mlp.c_proj.weight": (320, 1280),
            "transformer_layes.0.mlp.c_proj.bias": (320,),
            "spatial_ch_projs.weight": (320, 320),
            "spatial_ch_projs.bias": (320,),
            "control_add_embedding.linear_1.weight": (1280, 256 * config.mode_capacity),
            "control_add_embedding.linear_1.bias": (1280,),
            "control_add_embedding.linear_2.weight": (1280, 1280),
            "control_add_embedding.linear_2.bias": (1280,),
        }
    )
    return MappingProxyType(layout)


def normalize_sdxl_controlnet_union(
    geometries: Mapping[str, TensorGeometry],
) -> tuple[SDXLControlNetUnionConfig, dict[str, str]]:
    """Detect the exact 6-mode or 8-mode xinsir SDXL Union layout."""
    task = geometries.get("task_embedding")
    if task is None or len(task.shape) != 2 or task.shape[1] != 320:
        raise ControlNetDetectError("missing or invalid SDXL ControlNet Union task_embedding")
    capacity = task.shape[0]
    if capacity not in (6, 8):
        raise ControlNetDetectError("SDXL ControlNet Union mode capacity must be 6 or 8")
    config = SDXLControlNetUnionConfig(mode_capacity=capacity)
    layout = sdxl_controlnet_union_layout(config)
    source_to_model = {_sdxl_controlnet_diffusers_source_key(key): key for key in layout}
    keys = set(geometries)
    missing = sorted(set(source_to_model) - keys)
    leftover = sorted(keys - set(source_to_model))
    if missing:
        raise ControlNetDetectError("missing required Union keys: " + ", ".join(missing[:3]))
    if leftover:
        raise ControlNetDetectError("leftover or foreign Union keys: " + ", ".join(leftover[:3]))
    if len(set(source_to_model.values())) != len(source_to_model):
        raise ControlNetDetectError("ambiguous SDXL ControlNet Union key map collision")
    for source, model in source_to_model.items():
        geometry = geometries[source]
        expected = layout[model]
        if geometry.dtype.kind != "float":
            raise ControlNetDetectError(
                f"SDXL ControlNet Union weight {source} requires floating storage,"
                f" got {geometry.dtype.name}"
            )
        if geometry.shape != expected:
            raise ControlNetDetectError(
                f"Union geometry mismatch for {source}: got {geometry.dtype.name}"
                f" {geometry.shape}, expected floating {expected}"
            )
    return config, source_to_model


@dataclass(frozen=True)
class ControlApplication:
    """One ordered control invocation declaration without tensor payloads.

    Scheduled SD1.5 control uses ``hint.id`` as the materialized hint's
    content digest.
    """

    child_id: str
    hint: PayloadReference
    strength: float
    window: PercentRange
    previous: ControlApplication | None = None
    mode: SDControlMode | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.child_id), str):
            raise TypeError("control child_id must be a string")
        if _CHILD_ID_RE.fullmatch(self.child_id) is None or self.child_id == "parent":
            raise ValueError("control child_id must be a canonical non-parent id")
        if not isinstance(cast("object", self.hint), PayloadReference):
            raise TypeError("control hint must be a PayloadReference")
        if type(self.strength) is not float:
            raise TypeError("control strength must be a float")
        if not math.isfinite(self.strength) or not 0.0 <= self.strength <= 10.0:
            raise ValueError("control strength must be finite and in [0, 10]")
        if not isinstance(cast("object", self.window), PercentRange):
            raise TypeError("control window must be a PercentRange")
        if self.previous is not None and not isinstance(
            cast("object", self.previous), ControlApplication
        ):
            raise TypeError("control previous must be a ControlApplication or None")
        if self.mode is not None and type(self.mode) is not SDControlMode:
            raise TypeError("control mode must be an exact SDControlMode or None")
        active: set[int] = set()
        current: ControlApplication | None = self
        while current is not None:
            identity = id(current)
            if identity in active:
                raise ValueError("control application chain must be acyclic")
            active.add(identity)
            current = current.previous
        child_ids: set[str] = set()
        current = self
        while current is not None:
            if current.child_id in child_ids:
                raise ValueError("control application child ids must be unique")
            child_ids.add(current.child_id)
            current = current.previous


__all__ = [
    "ControlApplication",
    "ControlNetDetectError",
    "ControlNetLayout",
    "ControlNetSourceLayout",
    "SDControlMode",
    "SDControlModeToken",
    "SD_CONTROL_MODE_INDEX",
    "SD15_CONTROL_RESIDUAL_SITES",
    "SDXL_CONTROL_RESIDUAL_SITES",
    "SD15ControlNetConfig",
    "SDXLControlNetConfig",
    "SDXLControlLoRAConfig",
    "SDXLControlNetUnionConfig",
    "detect_sd15_controlnet",
    "detect_sdxl_control_lora",
    "normalize_sdxl_controlnet_union",
    "normalize_sdxl_controlnet",
    "sd15_controlnet_layout",
    "sdxl_controlnet_layout",
    "sdxl_control_lora_layout",
    "sdxl_controlnet_union_layout",
]
