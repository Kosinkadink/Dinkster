"""Torch-free maintained Qwen Image ControlNet layouts and detection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .devices import BFLOAT16, FLOAT16, FLOAT32, DType
from .qwen_image_layout import qwen_image_dit_layout
from .weights import WeightSource

QwenImageControlKind = Literal["instantx", "instantx_inpaint", "fun"]
QwenImageDiffSynthKind = Literal["diffsynth", "diffsynth_inpaint"]


@dataclass(frozen=True, slots=True)
class QwenImageControlConfig:
    """One exact maintained Qwen Image control architecture."""

    kind: QwenImageControlKind
    input_features: int
    transformer_blocks: int
    injection_layers: tuple[int, ...]
    supported_dtypes: tuple[DType, ...] = (BFLOAT16, FLOAT16, FLOAT32)

    def __post_init__(self) -> None:
        expected = {
            "instantx": (64, 60, tuple(range(60))),
            "instantx_inpaint": (68, 60, tuple(range(60))),
            "fun": (132, 5, (0, 12, 24, 36, 48)),
        }
        if (
            self.kind not in expected
            or (self.input_features, self.transformer_blocks, self.injection_layers)
            != expected[self.kind]
            or self.supported_dtypes != (BFLOAT16, FLOAT16, FLOAT32)
        ):
            raise ValueError("unsupported Qwen Image control configuration")


QWEN_IMAGE_INSTANTX_CONTROL = QwenImageControlConfig("instantx", 64, 60, tuple(range(60)))
QWEN_IMAGE_INSTANTX_INPAINT_CONTROL = QwenImageControlConfig(
    "instantx_inpaint", 68, 60, tuple(range(60))
)
QWEN_IMAGE_FUN_CONTROL = QwenImageControlConfig("fun", 132, 5, (0, 12, 24, 36, 48))


@dataclass(frozen=True, slots=True)
class QwenImageDiffSynthConfig:
    """One exact maintained blockwise Qwen Image DiffSynth patch."""

    kind: QwenImageDiffSynthKind
    input_features: int
    transformer_blocks: int = 60
    supported_dtypes: tuple[DType, ...] = (BFLOAT16, FLOAT16, FLOAT32)

    def __post_init__(self) -> None:
        if (
            (self.kind, self.input_features) not in (("diffsynth", 64), ("diffsynth_inpaint", 68))
            or self.transformer_blocks != 60
            or self.supported_dtypes != (BFLOAT16, FLOAT16, FLOAT32)
        ):
            raise ValueError("unsupported Qwen Image DiffSynth configuration")


QWEN_IMAGE_DIFFSYNTH = QwenImageDiffSynthConfig("diffsynth", 64)
QWEN_IMAGE_DIFFSYNTH_INPAINT = QwenImageDiffSynthConfig("diffsynth_inpaint", 68)


def qwen_image_diffsynth_layout(
    config: QwenImageDiffSynthConfig,
) -> Mapping[str, tuple[int, ...]]:
    """Return the complete state layout for an exact DiffSynth patch."""
    if config is not QWEN_IMAGE_DIFFSYNTH and config is not QWEN_IMAGE_DIFFSYNTH_INPAINT:
        raise ValueError("config must be an exact Qwen Image DiffSynth profile")
    keys: dict[str, tuple[int, ...]] = {
        "img_in.weight": (3072, config.input_features),
        "img_in.bias": (3072,),
    }
    for index in range(60):
        prefix = f"controlnet_blocks.{index}"
        keys[f"{prefix}.x_rms.weight"] = (3072,)
        keys[f"{prefix}.y_rms.weight"] = (3072,)
        keys[f"{prefix}.input_proj.weight"] = (3072, 3072)
        keys[f"{prefix}.input_proj.bias"] = (3072,)
        keys[f"{prefix}.output_proj.weight"] = (3072, 3072)
        keys[f"{prefix}.output_proj.bias"] = (3072,)
    return MappingProxyType(keys)


_DIFFSYNTH_LAYOUTS = MappingProxyType(
    {
        "diffsynth": qwen_image_diffsynth_layout(QWEN_IMAGE_DIFFSYNTH),
        "diffsynth_inpaint": qwen_image_diffsynth_layout(QWEN_IMAGE_DIFFSYNTH_INPAINT),
    }
)


def detect_qwen_image_diffsynth(source: WeightSource) -> QwenImageDiffSynthConfig | None:
    """Detect an exact maintained DiffSynth patch from header geometry."""
    key_set = set(source.keys())
    if "controlnet_blocks.0.y_rms.weight" not in key_set or "img_in.weight" not in key_set:
        return None
    try:
        width = source.entry("img_in.weight").geometry.shape[1]
    except (IndexError, KeyError):
        return None
    config = (
        QWEN_IMAGE_DIFFSYNTH
        if width == 64
        else QWEN_IMAGE_DIFFSYNTH_INPAINT
        if width == 68
        else None
    )
    if config is None:
        return None
    layout = _DIFFSYNTH_LAYOUTS[config.kind]
    if key_set != set(layout):
        return None
    for key, shape in layout.items():
        geometry = source.entry(key).geometry
        if geometry.shape != shape or geometry.dtype.kind != "float":
            return None
    return config


def require_qwen_image_diffsynth_layout(
    source: WeightSource,
) -> tuple[QwenImageDiffSynthConfig, Mapping[str, tuple[int, ...]]]:
    """Return exact DiffSynth detection and layout or raise at planning."""
    config = detect_qwen_image_diffsynth(source)
    if config is None:
        raise ValueError("source is not an exact maintained Qwen Image DiffSynth checkpoint")
    return config, qwen_image_diffsynth_layout(config)


def _instantx_layout(config: QwenImageControlConfig) -> dict[str, tuple[int, ...]]:
    keys = dict(qwen_image_dit_layout().keys)
    for prefix in ("norm_out.", "proj_out."):
        keys = {key: shape for key, shape in keys.items() if not key.startswith(prefix)}
    keys["controlnet_x_embedder.weight"] = (3072, config.input_features)
    keys["controlnet_x_embedder.bias"] = (3072,)
    for index in range(60):
        keys[f"controlnet_blocks.{index}.weight"] = (3072, 3072)
        keys[f"controlnet_blocks.{index}.bias"] = (3072,)
    return keys


def _fun_layout() -> dict[str, tuple[int, ...]]:
    base = qwen_image_dit_layout().keys
    block = {
        key.removeprefix("transformer_blocks.0."): shape
        for key, shape in base.items()
        if key.startswith("transformer_blocks.0.")
    }
    keys: dict[str, tuple[int, ...]] = {
        "control_img_in.weight": (3072, 132),
        "control_img_in.bias": (3072,),
    }
    for index in range(5):
        for suffix, shape in block.items():
            keys[f"control_blocks.{index}.{suffix}"] = shape
        if index == 0:
            keys["control_blocks.0.before_proj.weight"] = (3072, 3072)
            keys["control_blocks.0.before_proj.bias"] = (3072,)
        keys[f"control_blocks.{index}.after_proj.weight"] = (3072, 3072)
        keys[f"control_blocks.{index}.after_proj.bias"] = (3072,)
    return keys


_LAYOUTS: Mapping[QwenImageControlKind, Mapping[str, tuple[int, ...]]] = MappingProxyType(
    {
        "instantx": MappingProxyType(_instantx_layout(QWEN_IMAGE_INSTANTX_CONTROL)),
        "instantx_inpaint": MappingProxyType(_instantx_layout(QWEN_IMAGE_INSTANTX_INPAINT_CONTROL)),
        "fun": MappingProxyType(_fun_layout()),
    }
)


def qwen_image_control_layout(
    config: QwenImageControlConfig,
) -> Mapping[str, tuple[int, ...]]:
    """Return the complete state layout for one exact control architecture."""
    if all(
        config is not profile
        for profile in (
            QWEN_IMAGE_INSTANTX_CONTROL,
            QWEN_IMAGE_INSTANTX_INPAINT_CONTROL,
            QWEN_IMAGE_FUN_CONTROL,
        )
    ):
        raise ValueError("config must be an exact Qwen Image control profile")
    return _LAYOUTS[config.kind]


def detect_qwen_image_control(source: WeightSource) -> QwenImageControlConfig | None:
    """Detect an exact maintained control checkpoint from header geometry."""
    keys = tuple(source.keys())
    if len(keys) != len(set(keys)):
        return None
    key_set = set(keys)
    if "control_blocks.0.after_proj.weight" in key_set and "control_img_in.weight" in key_set:
        candidates = (QWEN_IMAGE_FUN_CONTROL,)
    elif (
        "controlnet_blocks.0.weight" in key_set
        and "transformer_blocks.0.img_mlp.net.0.proj.weight" in key_set
        and "controlnet_x_embedder.weight" in key_set
    ):
        try:
            width = source.entry("controlnet_x_embedder.weight").geometry.shape[1]
        except (IndexError, KeyError):
            return None
        candidates = (
            (QWEN_IMAGE_INSTANTX_CONTROL,)
            if width == 64
            else (QWEN_IMAGE_INSTANTX_INPAINT_CONTROL,)
            if width == 68
            else ()
        )
    else:
        return None
    for config in candidates:
        layout = qwen_image_control_layout(config)
        if key_set != set(layout):
            continue
        for key, shape in layout.items():
            try:
                geometry = source.entry(key).geometry
            except KeyError:
                break
            if geometry.shape != shape or geometry.dtype.kind != "float":
                break
        else:
            return config
    return None


def require_qwen_image_control_layout(
    source: WeightSource,
) -> tuple[QwenImageControlConfig, Mapping[str, tuple[int, ...]]]:
    """Return exact detection and layout or raise at the planning boundary."""
    config = detect_qwen_image_control(source)
    if config is None:
        raise ValueError("source is not an exact maintained Qwen Image control checkpoint")
    return config, qwen_image_control_layout(config)


__all__ = [
    "QWEN_IMAGE_DIFFSYNTH",
    "QWEN_IMAGE_DIFFSYNTH_INPAINT",
    "QWEN_IMAGE_FUN_CONTROL",
    "QWEN_IMAGE_INSTANTX_CONTROL",
    "QWEN_IMAGE_INSTANTX_INPAINT_CONTROL",
    "QwenImageControlConfig",
    "QwenImageControlKind",
    "QwenImageDiffSynthConfig",
    "QwenImageDiffSynthKind",
    "detect_qwen_image_control",
    "detect_qwen_image_diffsynth",
    "qwen_image_control_layout",
    "qwen_image_diffsynth_layout",
    "require_qwen_image_control_layout",
    "require_qwen_image_diffsynth_layout",
]
