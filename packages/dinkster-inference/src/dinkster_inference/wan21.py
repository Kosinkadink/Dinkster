"""Torch-free core Wan detection, latent, sampling, and codec facts.

The header rules follow ``comfy/model_detection.py`` and
``comfy/supported_models.py`` at ComfyUI ``7dde5617``. Official Wan 2.1 core,
Animate2, Fun, camera, and VACE geometries plus the supported Wan 2.2 core,
Fun, and camera geometries are admitted. Other Wan variants remain unrecognized.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, cast

from .codecs import CodecDescriptor, CodecTiling
from .devices import BFLOAT16, FLOAT16, FLOAT32
from .families import DetectionEvidence, EvidenceValue
from .latents import LatentDescriptor
from .quantization import QuantizationError, split_quantization
from .sampling import Parameterization, SamplingDescriptor
from .spaces import FlowSigmas
from .weights import ConfigurationPayloadSource, TensorGeometry, WeightEntry, WeightSource

__all__ = [
    "WAN21_CODEC",
    "WAN21_FLOW_RVS_CODEC",
    "WAN21_ANIMATE2_SETTINGS_KEY",
    "WAN21_CAUSAL_INITIAL_LATENT_KEY",
    "WAN21_SCAIL_REPLACEMENT_KEY",
    "WAN21_CAMERA_1_3B",
    "WAN21_CAMERA_14B",
    "WAN21_LATENT",
    "WAN21_SAMPLING",
    "WAN21_SIGMAS",
    "WAN21_FLF_I2V_14B",
    "WAN21_FUN_CONTROL_1_3B",
    "WAN21_FUN_INPAINT_1_3B",
    "WAN21_ANIMATE2_14B",
    "WAN21_I2V_14B",
    "WAN21_HUMO_17B",
    "WAN21_SCAIL_14B",
    "WAN21_SCAIL2_14B",
    "WAN21_T2V_1_3B",
    "WAN21_CAUSAL_AR_1_3B",
    "WAN21_FLOW_RVS_1_3B",
    "WAN21_T2V_14B",
    "WAN21_VACE_1_3B",
    "WAN21_VACE_14B",
    "WAN22_FUN_CONTROL_5B",
    "WAN22_FUN_CONTROL_14B",
    "WAN22_FUN_INPAINT_5B",
    "WAN22_ANIMATE_14B",
    "WAN22_BERNINI_14B",
    "WAN22_DANCER_SETTINGS_KEY",
    "WAN22_S2V_14B",
    "WAN22_I2V_14B",
    "WAN22_WANDANCER_14B",
    "WAN22_LATENT",
    "WAN22_CODEC",
    "WAN22_CAMERA_14B",
    "WAN22_SAMPLING",
    "WAN22_SIGMAS",
    "WAN22_TI2V_5B",
    "Wan21Config",
    "Wan21Animate2Settings",
    "Wan21Detector",
    "Wan21PoseBlockCacheDevice",
    "Wan21PoseBlockCacheSettings",
    "Wan21PoseBlockCacheStorage",
    "Wan22DancerSettings",
    "Wan22Detector",
    "decode_wan22_dancer_settings",
    "detect_wan21",
    "detect_wan22",
    "decode_wan21_animate2_settings",
    "encode_wan21_animate2_settings",
    "encode_wan22_dancer_settings",
    "wan21_layout",
]

_FAMILY_ID = "dinkster.wan21"
_WAN22_FAMILY_ID = "dinkster.wan22"
_PREFIXES = ("model.diffusion_model.", "")
_INFERENCE_DTYPES = frozenset({FLOAT16, BFLOAT16, FLOAT32})
_ANIMATE_MARKER = "face_adapter.fuser_blocks.0.k_norm.weight"
_ANIMATE_ROOTS = ("pose_patch_embedding.", "motion_encoder.", "face_adapter.", "face_encoder.")
_HUMO_MARKER = "audio_proj.audio_proj_glob_1.layer.bias"
_S2V_MARKER = "casual_audio_encoder.encoder.final_linear.weight"
_S2V_ROOTS = (
    "trainable_cond_mask.",
    "casual_audio_encoder.",
    "cond_encoder.",
    "audio_injector.",
    "frame_packer.",
)


@dataclass(frozen=True, slots=True)
class _Profile:
    name: str
    model_type: Literal["t2v", "i2v", "ti2v"]
    parameter_count: str
    input_channels: int
    output_channels: int
    hidden_width: int
    ffn_width: int
    attention_heads: int
    layers: int
    flf_pos_embed_token_number: int | None = None
    reference_channels: int | None = None
    vace_layers: int | None = None
    camera_channels: int | None = None
    model_variant: Literal[
        "base",
        "animate",
        "animate2",
        "bernini",
        "scail",
        "scail2",
        "flow_rvs",
        "causal_ar",
        "s2v",
        "humo",
        "wandancer",
    ] = "base"


_WAN21_PROFILES = (
    _Profile("t2v-1.3b", "t2v", "1.3B", 16, 16, 1536, 8960, 12, 30),
    _Profile(
        "causal-ar-1.3b",
        "t2v",
        "1.3B",
        16,
        16,
        1536,
        8960,
        12,
        30,
        model_variant="causal_ar",
    ),
    _Profile(
        "flow-rvs-1.3b",
        "t2v",
        "1.3B",
        16,
        16,
        1536,
        8960,
        12,
        30,
        model_variant="flow_rvs",
    ),
    _Profile("t2v-14b", "t2v", "14B", 16, 16, 5120, 13824, 40, 40),
    _Profile(
        "humo-17b",
        "t2v",
        "17B",
        36,
        16,
        5120,
        13824,
        40,
        40,
        model_variant="humo",
    ),
    _Profile(
        "s2v-14b-2.2",
        "t2v",
        "14B",
        16,
        16,
        5120,
        13824,
        40,
        40,
        model_variant="s2v",
    ),
    _Profile(
        "wandancer-14b-2.2",
        "i2v",
        "14B",
        36,
        16,
        5120,
        13824,
        40,
        40,
        reference_channels=16,
        model_variant="wandancer",
    ),
    _Profile(
        "bernini-14b-2.2",
        "t2v",
        "14B",
        16,
        16,
        5120,
        13824,
        40,
        40,
        model_variant="bernini",
    ),
    _Profile("i2v-14b", "i2v", "14B", 36, 16, 5120, 13824, 40, 40),
    _Profile(
        "animate2-14b-2.1",
        "i2v",
        "14B",
        36,
        16,
        5120,
        13824,
        40,
        40,
        model_variant="animate2",
    ),
    _Profile(
        "animate-14b-2.2",
        "i2v",
        "14B",
        36,
        16,
        5120,
        13824,
        40,
        40,
        model_variant="animate",
    ),
    _Profile(
        "scail-14b",
        "i2v",
        "14B",
        20,
        16,
        5120,
        13824,
        40,
        40,
        model_variant="scail",
    ),
    _Profile(
        "scail2-14b",
        "i2v",
        "14B",
        20,
        16,
        5120,
        13824,
        40,
        40,
        model_variant="scail2",
    ),
    _Profile("flf-i2v-14b", "i2v", "14B", 36, 16, 5120, 13824, 40, 40, 514),
    _Profile("fun-control-1.3b", "i2v", "1.3B", 48, 16, 1536, 8960, 12, 30),
    _Profile(
        "fun-inpaint-1.3b",
        "i2v",
        "1.3B",
        36,
        16,
        1536,
        8960,
        12,
        30,
    ),
    _Profile(
        "camera-1.3b",
        "i2v",
        "1.3B",
        32,
        16,
        1536,
        8960,
        12,
        30,
        camera_channels=24,
    ),
    _Profile(
        "camera-14b",
        "i2v",
        "14B",
        32,
        16,
        5120,
        13824,
        40,
        40,
        camera_channels=24,
    ),
    _Profile("i2v-14b-2.2", "t2v", "14B", 36, 16, 5120, 13824, 40, 40),
    _Profile(
        "camera-14b-2.2",
        "t2v",
        "14B",
        36,
        16,
        5120,
        13824,
        40,
        40,
        camera_channels=24,
    ),
    _Profile(
        "fun-control-14b-2.2",
        "t2v",
        "14B",
        52,
        16,
        5120,
        13824,
        40,
        40,
        reference_channels=16,
    ),
    _Profile("vace-1.3b", "t2v", "1.3B", 16, 16, 1536, 8960, 12, 30, vace_layers=15),
    _Profile("vace-14b", "t2v", "14B", 16, 16, 5120, 13824, 40, 40, vace_layers=8),
)
_WAN22_PROFILES = (
    _Profile("ti2v-5b", "ti2v", "5B", 48, 48, 3072, 14336, 24, 30),
    _Profile(
        "fun-control-5b",
        "t2v",
        "5B",
        148,
        48,
        3072,
        14336,
        24,
        30,
        reference_channels=48,
    ),
    _Profile(
        "fun-inpaint-5b",
        "t2v",
        "5B",
        100,
        48,
        3072,
        14336,
        24,
        30,
    ),
)


@dataclass(frozen=True, slots=True)
class Wan21Config:
    """Construction-relevant geometry for one native Wan DiT."""

    model_type: Literal["t2v", "i2v", "ti2v"] = "t2v"
    in_channels: int = 16
    hidden_size: int = 1536
    ffn_hidden_size: int = 8960
    num_heads: int = 12
    num_layers: int = 30
    text_dim: int = 4096
    time_freq_dim: int = 256
    out_channels: int = 16
    patch_size: tuple[int, int, int] = (1, 2, 2)
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6
    flf_pos_embed_token_number: int | None = None
    vace_layers: int | None = field(default=None, repr=False)
    reference_channels: int | None = None
    camera_channels: int | None = None
    model_variant: Literal[
        "base",
        "animate",
        "animate2",
        "bernini",
        "scail",
        "scail2",
        "flow_rvs",
        "causal_ar",
        "s2v",
        "humo",
        "wandancer",
    ] = "base"

    def __post_init__(self) -> None:
        dimensions = (
            self.in_channels,
            self.hidden_size,
            self.ffn_hidden_size,
            self.num_heads,
            self.num_layers,
            self.text_dim,
            self.time_freq_dim,
            self.out_channels,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Wan dimensions and layer counts must be positive integers")
        if self.model_type not in ("t2v", "i2v", "ti2v"):
            raise ValueError("Wan model_type must be 't2v', 'i2v', or 'ti2v'")
        if self.model_variant not in (
            "base",
            "animate",
            "animate2",
            "bernini",
            "scail",
            "scail2",
            "flow_rvs",
            "causal_ar",
            "s2v",
            "humo",
            "wandancer",
        ):
            raise ValueError(
                "Wan model_variant must be 'base', 'animate', 'animate2', 'bernini', "
                "'scail', 'scail2', 'flow_rvs', 'causal_ar', 's2v', 'humo', or 'wandancer'"
            )
        if self.flf_pos_embed_token_number is not None:
            if (
                type(self.flf_pos_embed_token_number) is not int
                or self.flf_pos_embed_token_number <= 0
            ):
                raise ValueError("Wan FLF positional token count must be a positive integer")
            if self.model_type != "i2v":
                raise ValueError("Wan FLF positional embeddings require I2V geometry")
        if self.reference_channels is not None:
            if type(self.reference_channels) is not int or self.reference_channels <= 0:
                raise ValueError("Wan reference channels must be a positive integer")
            if self.reference_channels != self.out_channels:
                raise ValueError("Wan reference channels must match the output latent channels")
            if (
                self.model_variant != "wandancer"
                and self.in_channels - self.out_channels != self.out_channels * 2 + 4
            ):
                raise ValueError("Wan reference projection requires Fun control geometry")
        if self.vace_layers is not None:
            if type(self.vace_layers) is not int or self.vace_layers <= 0:
                raise ValueError("Wan VACE layer count must be a positive integer")
            if self.model_type != "t2v" or (self.in_channels, self.out_channels) != (16, 16):
                raise ValueError("Wan VACE requires 16-channel T2V geometry")
            if self.vace_layers > self.num_layers or self.num_layers % self.vace_layers:
                raise ValueError("Wan VACE layers must divide the backbone layer count")
        if self.camera_channels is not None:
            if self.camera_channels != 24:
                raise ValueError("Wan camera conditioning requires 24 channels")
            expected_camera = (32, 16) if self.model_type == "i2v" else (36, 16)
            if (self.in_channels, self.out_channels) != expected_camera:
                raise ValueError(
                    "Wan 2.1 camera requires 32/16 I2V geometry and Wan 2.2 camera "
                    "requires 36/16 T2V geometry"
                )
            if self.reference_channels is not None or self.vace_layers is not None:
                raise ValueError("Wan camera geometry cannot also be full-reference or VACE")
        if self.patch_size != (1, 2, 2):
            raise ValueError("Wan requires patch_size=(1, 2, 2)")
        if self.model_variant in ("animate", "animate2"):
            if (self.model_type, self.in_channels, self.out_channels) != ("i2v", 36, 16):
                raise ValueError("Wan Animate models require 36/16 I2V geometry")
            if self.model_variant == "animate" and (self.num_layers < 5 or self.num_layers % 5):
                raise ValueError("Wan Animate layers must be a positive multiple of five")
            if any(
                value is not None
                for value in (
                    self.flf_pos_embed_token_number,
                    self.vace_layers,
                    self.reference_channels,
                    self.camera_channels,
                )
            ):
                raise ValueError(
                    "Wan Animate models cannot also use FLF, VACE, reference, or camera geometry"
                )
        if self.model_variant in ("scail", "scail2"):
            if (self.model_type, self.in_channels, self.out_channels) != ("i2v", 20, 16):
                raise ValueError("Wan SCAIL models require 20/16 I2V geometry")
            if any(
                value is not None
                for value in (
                    self.flf_pos_embed_token_number,
                    self.vace_layers,
                    self.reference_channels,
                    self.camera_channels,
                )
            ):
                raise ValueError(
                    "Wan SCAIL models cannot also use FLF, VACE, reference, or camera geometry"
                )
        elif (self.in_channels, self.out_channels) == (20, 16):
            raise ValueError("Wan 20/16 I2V geometry requires a SCAIL model variant")
        if self.model_variant == "flow_rvs" and (
            self.model_type != "t2v"
            or (self.in_channels, self.out_channels) != (16, 16)
            or (
                self.hidden_size,
                self.ffn_hidden_size,
                self.num_heads,
                self.num_layers,
            )
            != (1536, 8960, 12, 30)
            or any(
                value is not None
                for value in (
                    self.flf_pos_embed_token_number,
                    self.vace_layers,
                    self.reference_channels,
                    self.camera_channels,
                )
            )
        ):
            raise ValueError("Wan FlowRVS requires base 16-channel T2V geometry")
        if self.model_variant == "causal_ar" and (
            self.model_type != "t2v"
            or (self.in_channels, self.out_channels) != (16, 16)
            or (
                self.hidden_size,
                self.ffn_hidden_size,
                self.num_heads,
                self.num_layers,
            )
            != (1536, 8960, 12, 30)
            or any(
                value is not None
                for value in (
                    self.flf_pos_embed_token_number,
                    self.vace_layers,
                    self.reference_channels,
                    self.camera_channels,
                )
            )
        ):
            raise ValueError("Wan CausalAR requires base 1.3B 16-channel T2V geometry")
        if self.model_variant == "bernini" and (
            self.model_type != "t2v"
            or (self.in_channels, self.out_channels) != (16, 16)
            or (
                self.hidden_size,
                self.ffn_hidden_size,
                self.num_heads,
                self.num_layers,
            )
            != (5120, 13824, 40, 40)
            or any(
                value is not None
                for value in (
                    self.flf_pos_embed_token_number,
                    self.vace_layers,
                    self.reference_channels,
                    self.camera_channels,
                )
            )
        ):
            raise ValueError("Wan Bernini requires 14B 16-channel T2V geometry")
        if self.model_variant == "s2v" and (
            self.model_type != "t2v"
            or (self.in_channels, self.out_channels) != (16, 16)
            or (
                self.hidden_size,
                self.ffn_hidden_size,
                self.num_heads,
                self.num_layers,
            )
            != (5120, 13824, 40, 40)
            or any(
                value is not None
                for value in (
                    self.flf_pos_embed_token_number,
                    self.vace_layers,
                    self.reference_channels,
                    self.camera_channels,
                )
            )
        ):
            raise ValueError("Wan 2.2 S2V requires 14B 16-channel T2V geometry")
        if self.model_variant == "humo" and (
            self.model_type != "t2v"
            or (self.in_channels, self.out_channels) != (36, 16)
            or (
                self.hidden_size,
                self.ffn_hidden_size,
                self.num_heads,
                self.num_layers,
            )
            != (5120, 13824, 40, 40)
            or any(
                value is not None
                for value in (
                    self.flf_pos_embed_token_number,
                    self.vace_layers,
                    self.reference_channels,
                    self.camera_channels,
                )
            )
        ):
            raise ValueError("Wan HuMo requires 17B 36/16 T2V geometry")
        if self.model_variant == "wandancer" and (
            self.model_type != "i2v"
            or (self.in_channels, self.out_channels) != (36, 16)
            or (
                self.hidden_size,
                self.ffn_hidden_size,
                self.num_heads,
                self.num_layers,
            )
            != (5120, 13824, 40, 40)
            or self.reference_channels != 16
            or any(
                value is not None
                for value in (
                    self.flf_pos_embed_token_number,
                    self.vace_layers,
                    self.camera_channels,
                )
            )
        ):
            raise ValueError("WanDancer requires exact 14B 36/16 I2V and reference geometry")
        expected_channels = {
            "t2v": ((16, 16), (36, 16), (52, 16), (100, 48), (148, 48)),
            "i2v": ((20, 16), (32, 16), (36, 16), (48, 16)),
            "ti2v": ((48, 48),),
        }[self.model_type]
        if (self.in_channels, self.out_channels) not in expected_channels:
            expected = " or ".join(f"{inputs}/{outputs}" for inputs, outputs in expected_channels)
            raise ValueError(
                f"Wan {self.model_type.upper()} requires input/output channels {expected}"
            )
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        head_dim = self.hidden_size // self.num_heads
        if head_dim % 2 or head_dim < 6:
            raise ValueError("Wan attention head width must be even and at least 6")
        if self.time_freq_dim % 2:
            raise ValueError("time_freq_dim must be even")
        if not isinstance(self.eps, float) or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be a finite positive float")


@dataclass(frozen=True, slots=True)
class Wan21Animate2Settings:
    pose_strength: float = 1.0
    reference_strength: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("pose_strength", self.pose_strength),
            ("reference_strength", self.reference_strength),
        ):
            if type(value) is not float:
                raise TypeError(f"{name} must be a float")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class Wan22DancerSettings:
    fps: float = 30.0
    audio_inject_scale: float = 1.0

    def __post_init__(self) -> None:
        if type(self.fps) is not float or not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("WanDancer fps must be a finite positive float")
        if (
            type(self.audio_inject_scale) is not float
            or not math.isfinite(self.audio_inject_scale)
            or self.audio_inject_scale < 0.0
        ):
            raise ValueError("WanDancer audio injection scale must be a finite non-negative float")


WAN21_ANIMATE2_SETTINGS_KEY = "dinkster.wan21/animate2"
WAN21_CAUSAL_INITIAL_LATENT_KEY = "dinkster.model-wan/causal-ar-initial-latent"
WAN21_SCAIL_REPLACEMENT_KEY = "dinkster.wan21/scail-replacement"
WAN22_DANCER_SETTINGS_KEY = "dinkster.wan22/dancer"


def encode_wan21_animate2_settings(settings: Wan21Animate2Settings) -> Mapping[str, float]:
    if type(settings) is not Wan21Animate2Settings:
        raise TypeError("settings must be exact Wan21Animate2Settings")
    return MappingProxyType(
        {
            "pose_strength": settings.pose_strength,
            "reference_strength": settings.reference_strength,
        }
    )


def decode_wan21_animate2_settings(value: object) -> Wan21Animate2Settings:
    if not isinstance(value, Mapping):
        raise TypeError("Wan Animate2 settings metadata must be a mapping")
    raw = cast("Mapping[object, object]", value)
    expected = {"pose_strength", "reference_strength"}
    if set(raw) != expected:
        unknown = sorted(str(key) for key in set(raw) - expected)
        missing = sorted(expected - set(raw))
        details: list[str] = []
        if unknown:
            details.append("unknown keys: " + ", ".join(unknown))
        if missing:
            details.append("missing keys: " + ", ".join(missing))
        raise ValueError("invalid Wan Animate2 settings metadata (" + "; ".join(details) + ")")
    pose_strength = raw["pose_strength"]
    reference_strength = raw["reference_strength"]
    if type(pose_strength) is not float or type(reference_strength) is not float:
        raise TypeError("Wan Animate2 settings values must be floats")
    return Wan21Animate2Settings(pose_strength, reference_strength)


def encode_wan22_dancer_settings(settings: Wan22DancerSettings) -> Mapping[str, float]:
    if type(settings) is not Wan22DancerSettings:
        raise TypeError("settings must be exact Wan22DancerSettings")
    return MappingProxyType(
        {
            "fps": settings.fps,
            "audio_inject_scale": settings.audio_inject_scale,
        }
    )


def decode_wan22_dancer_settings(value: object) -> Wan22DancerSettings:
    if not isinstance(value, Mapping):
        raise TypeError("WanDancer settings metadata must be a mapping")
    raw = cast("Mapping[object, object]", value)
    expected = {"fps", "audio_inject_scale"}
    if set(raw) != expected:
        raise ValueError("WanDancer settings metadata requires exactly fps and audio_inject_scale")
    fps = raw["fps"]
    audio_inject_scale = raw["audio_inject_scale"]
    if type(fps) is not float or type(audio_inject_scale) is not float:
        raise TypeError("WanDancer settings values must be floats")
    return Wan22DancerSettings(fps, audio_inject_scale)


class Wan21PoseBlockCacheDevice(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


class Wan21PoseBlockCacheStorage(StrEnum):
    DEFAULT = "default"
    INT8 = "int8"
    INT4 = "int4"


@dataclass(frozen=True, slots=True)
class Wan21PoseBlockCacheSettings:
    device: Wan21PoseBlockCacheDevice = Wan21PoseBlockCacheDevice.CPU
    memory_limit_bytes: int | None = None
    storage: Wan21PoseBlockCacheStorage = Wan21PoseBlockCacheStorage.DEFAULT

    def __post_init__(self) -> None:
        if type(cast("object", self.device)) is not Wan21PoseBlockCacheDevice:
            raise TypeError("Animate2 pose cache device must be Wan21PoseBlockCacheDevice")
        if self.memory_limit_bytes is not None and (
            type(self.memory_limit_bytes) is not int or self.memory_limit_bytes < 1
        ):
            raise ValueError("Animate2 pose cache memory limit must be a positive integer")
        if type(cast("object", self.storage)) is not Wan21PoseBlockCacheStorage:
            raise TypeError("Animate2 pose cache storage must be Wan21PoseBlockCacheStorage")

    @property
    def runtime_facts(self) -> tuple[str, ...]:
        if self.storage is Wan21PoseBlockCacheStorage.DEFAULT:
            return ()
        return (f"animate2_cache_storage={self.storage.value}",)


WAN21_T2V_1_3B = Wan21Config()
WAN21_CAUSAL_AR_1_3B = Wan21Config(model_variant="causal_ar")
WAN21_FLOW_RVS_1_3B = Wan21Config(model_variant="flow_rvs")
WAN21_T2V_14B = Wan21Config(
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
)
WAN21_HUMO_17B = Wan21Config(
    in_channels=36,
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
    model_variant="humo",
)
WAN22_BERNINI_14B = Wan21Config(
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
    model_variant="bernini",
)
WAN22_S2V_14B = Wan21Config(
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
    model_variant="s2v",
)
WAN22_WANDANCER_14B = Wan21Config(
    model_type="i2v",
    in_channels=36,
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
    reference_channels=16,
    model_variant="wandancer",
)
WAN21_I2V_14B = Wan21Config(
    model_type="i2v",
    in_channels=36,
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
)
WAN21_ANIMATE2_14B = Wan21Config(
    model_type="i2v",
    model_variant="animate2",
    in_channels=36,
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
)
WAN21_SCAIL_14B = Wan21Config(
    model_type="i2v",
    model_variant="scail",
    in_channels=20,
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
)
WAN21_SCAIL2_14B = Wan21Config(
    model_type="i2v",
    model_variant="scail2",
    in_channels=20,
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
)
WAN22_ANIMATE_14B = Wan21Config(
    model_type="i2v",
    model_variant="animate",
    in_channels=36,
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
)
WAN21_FLF_I2V_14B = Wan21Config(
    model_type="i2v",
    in_channels=36,
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
    flf_pos_embed_token_number=514,
)
WAN21_FUN_CONTROL_1_3B = Wan21Config(model_type="i2v", in_channels=48)
WAN21_FUN_INPAINT_1_3B = Wan21Config(
    model_type="i2v",
    in_channels=36,
)
WAN21_CAMERA_1_3B = Wan21Config(
    model_type="i2v",
    in_channels=32,
    camera_channels=24,
)
WAN21_CAMERA_14B = Wan21Config(
    model_type="i2v",
    in_channels=32,
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
    camera_channels=24,
)
WAN21_VACE_1_3B = Wan21Config(vace_layers=15)
WAN21_VACE_14B = Wan21Config(
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
    vace_layers=8,
)
WAN22_I2V_14B = Wan21Config(
    in_channels=36,
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
)
WAN22_CAMERA_14B = Wan21Config(
    in_channels=36,
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
    camera_channels=24,
)
WAN22_TI2V_5B = Wan21Config(
    model_type="ti2v",
    in_channels=48,
    hidden_size=3072,
    ffn_hidden_size=14336,
    num_heads=24,
    num_layers=30,
    out_channels=48,
)
WAN22_FUN_CONTROL_5B = Wan21Config(
    in_channels=148,
    hidden_size=3072,
    ffn_hidden_size=14336,
    num_heads=24,
    num_layers=30,
    out_channels=48,
    reference_channels=48,
)
WAN22_FUN_INPAINT_5B = Wan21Config(
    in_channels=100,
    hidden_size=3072,
    ffn_hidden_size=14336,
    num_heads=24,
    num_layers=30,
    out_channels=48,
)
WAN22_FUN_CONTROL_14B = Wan21Config(
    in_channels=52,
    hidden_size=5120,
    ffn_hidden_size=13824,
    num_heads=40,
    num_layers=40,
    reference_channels=16,
)


def _wan22_animate_layout(config: Wan21Config) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    head_dim = hidden // config.num_heads
    layout: dict[str, tuple[int, ...]] = {
        "pose_patch_embedding.weight": (hidden, config.out_channels, *config.patch_size),
        "pose_patch_embedding.bias": (hidden,),
        "motion_encoder.enc.net_app.convs.0.0.weight": (32, 3, 1, 1),
        "motion_encoder.enc.net_app.convs.0.1.bias": (1, 32, 1, 1),
        "motion_encoder.enc.net_app.convs.8.weight": (512, 512, 4, 4),
        "motion_encoder.dec.direction.weight": (512, 20),
        "face_encoder.conv1_local.conv.weight": (4096, 512, 3),
        "face_encoder.conv1_local.conv.bias": (4096,),
        "face_encoder.conv2.conv.weight": (1024, 1024, 3),
        "face_encoder.conv2.conv.bias": (1024,),
        "face_encoder.conv3.conv.weight": (1024, 1024, 3),
        "face_encoder.conv3.conv.bias": (1024,),
        "face_encoder.out_proj.weight": (hidden, 1024),
        "face_encoder.out_proj.bias": (hidden,),
        "face_encoder.padding_tokens": (1, 1, 1, hidden),
    }
    channels = (
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 512),
        (512, 512),
        (512, 512),
    )
    for index, (input_channels, output_channels) in enumerate(channels, start=1):
        block = f"motion_encoder.enc.net_app.convs.{index}"
        layout[f"{block}.conv1.0.weight"] = (input_channels, input_channels, 3, 3)
        layout[f"{block}.conv1.1.bias"] = (1, input_channels, 1, 1)
        layout[f"{block}.conv2.0.kernel"] = (4, 4)
        layout[f"{block}.conv2.1.weight"] = (output_channels, input_channels, 3, 3)
        layout[f"{block}.conv2.2.bias"] = (1, output_channels, 1, 1)
        layout[f"{block}.skip.0.kernel"] = (4, 4)
        layout[f"{block}.skip.1.weight"] = (output_channels, input_channels, 1, 1)
    for index in range(5):
        input_features = 512
        output_features = 20 if index == 4 else 512
        layout[f"motion_encoder.enc.fc.{index}.weight"] = (
            output_features,
            input_features,
        )
        layout[f"motion_encoder.enc.fc.{index}.bias"] = (output_features,)
    for index in range(config.num_layers // 5):
        block = f"face_adapter.fuser_blocks.{index}"
        for name, output_features in (
            ("linear1_kv", hidden * 2),
            ("linear1_q", hidden),
            ("linear2", hidden),
        ):
            layout[f"{block}.{name}.weight"] = (output_features, hidden)
            layout[f"{block}.{name}.bias"] = (output_features,)
        layout[f"{block}.q_norm.weight"] = (head_dim,)
        layout[f"{block}.k_norm.weight"] = (head_dim,)
    return layout


def wan21_layout(config: Wan21Config = WAN21_T2V_1_3B) -> dict[str, tuple[int, ...]]:
    """Return the exact Wan diffusion model-key to tensor-shape contract."""

    layout: dict[str, tuple[int, ...]] = {}

    def affine(prefix: str, out_features: int, in_features: int) -> None:
        layout[f"{prefix}.weight"] = (out_features, in_features)
        layout[f"{prefix}.bias"] = (out_features,)

    hidden = config.hidden_size
    layout["patch_embedding.weight"] = (
        hidden,
        config.in_channels,
        *config.patch_size,
    )
    layout["patch_embedding.bias"] = (hidden,)
    if config.model_variant in ("scail", "scail2"):
        layout["patch_embedding_pose.weight"] = (hidden, 20, *config.patch_size)
        layout["patch_embedding_pose.bias"] = (hidden,)
    if config.model_variant == "scail2":
        layout["patch_embedding_mask.weight"] = (hidden, 28, *config.patch_size)
        layout["patch_embedding_mask.bias"] = (hidden,)
    if config.model_variant == "s2v":
        layout["trainable_cond_mask.weight"] = (3, hidden)
        layout["casual_audio_encoder.weights"] = (1, 25, 1, 1)
        for name, out_channels, in_channels in (
            ("conv1_local", hidden, 1024),
            ("conv1_global", hidden // 4, 1024),
            ("conv2", hidden // 2, hidden // 4),
            ("conv3", hidden, hidden // 2),
        ):
            layout[f"casual_audio_encoder.encoder.{name}.conv.weight"] = (
                out_channels,
                in_channels,
                3,
            )
            layout[f"casual_audio_encoder.encoder.{name}.conv.bias"] = (out_channels,)
        affine("casual_audio_encoder.encoder.final_linear", hidden, hidden)
        layout["casual_audio_encoder.encoder.padding_tokens"] = (1, 1, 1, hidden)
        layout["cond_encoder.weight"] = (hidden, 16, *config.patch_size)
        layout["cond_encoder.bias"] = (hidden,)
        for index in range(12):
            injector = f"audio_injector.injector.{index}"
            for projection in ("q", "k", "v", "o"):
                affine(f"{injector}.{projection}", hidden, hidden)
            layout[f"{injector}.norm_q.weight"] = (hidden,)
            layout[f"{injector}.norm_k.weight"] = (hidden,)
            affine(f"audio_injector.injector_adain_layers.{index}.linear", hidden * 2, hidden)
        for name, temporal, spatial in (
            ("proj", 1, 2),
            ("proj_2x", 2, 4),
            ("proj_4x", 4, 8),
        ):
            layout[f"frame_packer.{name}.weight"] = (
                hidden,
                16,
                temporal,
                spatial,
                spatial,
            )
            layout[f"frame_packer.{name}.bias"] = (hidden,)
    if config.model_variant == "humo":
        for name, out_features, in_features in (
            ("audio_proj_glob_1", 512, 51200),
            ("audio_proj_glob_2", 512, 512),
            ("audio_proj_glob_3", 24576, 512),
        ):
            affine(f"audio_proj.{name}.layer", out_features, in_features)
        layout["audio_proj.audio_proj_glob_norm.layer.weight"] = (1536,)
        layout["audio_proj.audio_proj_glob_norm.layer.bias"] = (1536,)
    affine("text_embedding.0", hidden, config.text_dim)
    affine("text_embedding.2", hidden, hidden)
    affine("time_embedding.0", hidden, config.time_freq_dim)
    affine("time_embedding.2", hidden, hidden)
    affine("time_projection.1", hidden * 6, hidden)

    def attention_block(block: str) -> None:
        layout[f"{block}.modulation"] = (1, 6, hidden)
        for attention in ("self_attn", "cross_attn"):
            for projection in ("q", "k", "v", "o"):
                affine(f"{block}.{attention}.{projection}", hidden, hidden)
            layout[f"{block}.{attention}.norm_q.weight"] = (hidden,)
            layout[f"{block}.{attention}.norm_k.weight"] = (hidden,)
        layout[f"{block}.norm3.weight"] = (hidden,)
        layout[f"{block}.norm3.bias"] = (hidden,)
        affine(f"{block}.ffn.0", config.ffn_hidden_size, hidden)
        affine(f"{block}.ffn.2", hidden, config.ffn_hidden_size)
        if config.model_type == "i2v":
            affine(f"{block}.cross_attn.k_img", hidden, hidden)
            affine(f"{block}.cross_attn.v_img", hidden, hidden)
            layout[f"{block}.cross_attn.norm_k_img.weight"] = (hidden,)

    for index in range(config.num_layers):
        attention_block(f"blocks.{index}")
        if config.model_variant == "humo":
            wrapper = f"blocks.{index}.audio_cross_attn_wrapper"
            for projection, in_features in (
                ("q", hidden),
                ("k", 1536),
                ("v", 1536),
                ("o", hidden),
            ):
                affine(f"{wrapper}.audio_cross_attn.{projection}", hidden, in_features)
            layout[f"{wrapper}.audio_cross_attn.norm_q.weight"] = (hidden,)
            layout[f"{wrapper}.audio_cross_attn.norm_k.weight"] = (hidden,)
            layout[f"{wrapper}.norm1_audio.weight"] = (hidden,)
            layout[f"{wrapper}.norm1_audio.bias"] = (hidden,)

    if config.model_variant == "wandancer":
        layout["patch_embedding_global.weight"] = (
            hidden,
            config.in_channels,
            *config.patch_size,
        )
        layout["patch_embedding_global.bias"] = (hidden,)
        layout["img_emb_refimage.proj.0.weight"] = (1280,)
        layout["img_emb_refimage.proj.0.bias"] = (1280,)
        affine("img_emb_refimage.proj.1", 1280, 1280)
        affine("img_emb_refimage.proj.3", hidden, 1280)
        layout["img_emb_refimage.proj.4.weight"] = (hidden,)
        layout["img_emb_refimage.proj.4.bias"] = (hidden,)
        affine("head_global.head", math.prod(config.patch_size) * config.out_channels, hidden)
        layout["head_global.modulation"] = (1, 2, hidden)
        affine("music_projection", 256, 35)
        for index in range(2):
            root = f"music_encoder.{index}"
            layout[f"{root}.self_attn.in_proj_weight"] = (768, 256)
            layout[f"{root}.self_attn.in_proj_bias"] = (768,)
            affine(f"{root}.self_attn.out_proj", 256, 256)
            affine(f"{root}.linear1", 1024, 256)
            affine(f"{root}.linear2", 256, 1024)
            for norm in ("norm1", "norm2"):
                layout[f"{root}.{norm}.weight"] = (256,)
                layout[f"{root}.{norm}.bias"] = (256,)
        for index in range(8):
            injector = f"music_injector.injector.{index}"
            for projection in ("q", "k", "v", "o"):
                affine(f"{injector}.{projection}", hidden, hidden)
            layout[f"{injector}.norm_q.weight"] = (hidden,)
            layout[f"{injector}.norm_k.weight"] = (hidden,)

    if config.vace_layers is not None:
        layout["vace_patch_embedding.weight"] = (hidden, 96, *config.patch_size)
        layout["vace_patch_embedding.bias"] = (hidden,)
        for index in range(config.vace_layers):
            block = f"vace_blocks.{index}"
            attention_block(block)
            if index == 0:
                affine(f"{block}.before_proj", hidden, hidden)
            affine(f"{block}.after_proj", hidden, hidden)

    if config.model_type == "i2v":
        if config.flf_pos_embed_token_number is not None:
            layout["img_emb.emb_pos"] = (1, config.flf_pos_embed_token_number, 1280)
        layout["img_emb.proj.0.weight"] = (1280,)
        layout["img_emb.proj.0.bias"] = (1280,)
        affine("img_emb.proj.1", 1280, 1280)
        affine("img_emb.proj.3", hidden, 1280)
        layout["img_emb.proj.4.weight"] = (hidden,)
        layout["img_emb.proj.4.bias"] = (hidden,)

    if config.model_variant == "animate":
        layout.update(_wan22_animate_layout(config))

    if config.reference_channels is not None:
        layout["ref_conv.weight"] = (hidden, config.reference_channels, 2, 2)
        layout["ref_conv.bias"] = (hidden,)

    if config.camera_channels is not None:
        layout["control_adapter.conv.weight"] = (
            hidden,
            config.camera_channels * 64,
            2,
            2,
        )
        layout["control_adapter.conv.bias"] = (hidden,)
        for projection in ("conv1", "conv2"):
            layout[f"control_adapter.residual_blocks.0.{projection}.weight"] = (
                hidden,
                hidden,
                3,
                3,
            )
            layout[f"control_adapter.residual_blocks.0.{projection}.bias"] = (hidden,)

    layout["head.modulation"] = (1, 2, hidden)
    affine("head.head", math.prod(config.patch_size) * config.out_channels, hidden)
    return layout


# Cheap-preview projection constants for the Wan latent spaces
# (comfy/latent_formats.py @ 783545f6). Wan content plays at 16 fps.
_WAN21_RGB_FACTORS = (
    (-0.1299, -0.1692, 0.2932),
    (0.0671, 0.0406, 0.0442),
    (0.3568, 0.2548, 0.1747),
    (0.0372, 0.2344, 0.1420),
    (0.0313, 0.0189, -0.0328),
    (0.0296, -0.0956, -0.0665),
    (-0.3477, -0.4059, -0.2925),
    (0.0166, 0.1902, 0.1975),
    (-0.0412, 0.0267, -0.1364),
    (-0.1293, 0.0740, 0.1636),
    (0.0680, 0.3019, 0.1128),
    (0.0032, 0.0581, 0.0639),
    (-0.1251, 0.0927, 0.1699),
    (0.0060, -0.0633, 0.0005),
    (0.3477, 0.2275, 0.2950),
    (0.1984, 0.0913, 0.1861),
)

_WAN22_RGB_FACTORS = (
    (0.0119, 0.0103, 0.0046),
    (-0.1062, -0.0504, 0.0165),
    (0.0140, 0.0409, 0.0491),
    (-0.0813, -0.0677, 0.0607),
    (0.0656, 0.0851, 0.0808),
    (0.0264, 0.0463, 0.0912),
    (0.0295, 0.0326, 0.0590),
    (-0.0244, -0.0270, 0.0025),
    (0.0443, -0.0102, 0.0288),
    (-0.0465, -0.0090, -0.0205),
    (0.0359, 0.0236, 0.0082),
    (-0.0776, 0.0854, 0.1048),
    (0.0564, 0.0264, 0.0561),
    (0.0006, 0.0594, 0.0418),
    (-0.0319, -0.0542, -0.0637),
    (-0.0268, 0.0024, 0.0260),
    (0.0539, 0.0265, 0.0358),
    (-0.0359, -0.0312, -0.0287),
    (-0.0285, -0.1032, -0.1237),
    (0.1041, 0.0537, 0.0622),
    (-0.0086, -0.0374, -0.0051),
    (0.0390, 0.0670, 0.2863),
    (0.0069, 0.0144, 0.0082),
    (0.0006, -0.0167, 0.0079),
    (0.0313, -0.0574, -0.0232),
    (-0.1454, -0.0902, -0.0481),
    (0.0714, 0.0827, 0.0447),
    (-0.0304, -0.0574, -0.0196),
    (0.0401, 0.0384, 0.0204),
    (-0.0758, -0.0297, -0.0014),
    (0.0568, 0.1307, 0.1372),
    (-0.0055, -0.0310, -0.0380),
    (0.0239, -0.0305, 0.0325),
    (-0.0663, -0.0673, -0.0140),
    (-0.0416, -0.0047, -0.0023),
    (0.0166, 0.0112, -0.0093),
    (-0.0211, 0.0011, 0.0331),
    (0.1833, 0.1466, 0.2250),
    (-0.0368, 0.0370, 0.0295),
    (-0.3441, -0.3543, -0.2008),
    (-0.0479, -0.0489, -0.0420),
    (-0.0660, -0.0153, 0.0800),
    (-0.0101, 0.0068, 0.0156),
    (-0.0690, -0.0452, -0.0927),
    (-0.0145, 0.0041, 0.0015),
    (0.0421, 0.0451, 0.0373),
    (0.0504, -0.0483, -0.0356),
    (-0.0837, 0.0168, 0.0055),
)

WAN21_LATENT = LatentDescriptor(
    channels=16,
    dimensions=3,
    spatial_downscale=8,
    temporal_downscale=4,
    temporal_causal=True,
    content_fps=16.0,
    rgb_factors=_WAN21_RGB_FACTORS,
    rgb_bias=(-0.1835, -0.0868, -0.3360),
    taesd_decoder="lighttaew2_1",
)

WAN21_SIGMAS = FlowSigmas(shift=8.0)
WAN21_SAMPLING = SamplingDescriptor(
    parameterization=Parameterization.FLOW,
    sigma_min=WAN21_SIGMAS.sigma_min,
    sigma_max=WAN21_SIGMAS.sigma_max,
    shift=8.0,
)

WAN21_CODEC = CodecDescriptor(
    id="dinkster.wan21_vae",
    display_name="Wan 2.1 causal video VAE",
    kind="video",
    latent=WAN21_LATENT,
    supported_dtypes=_INFERENCE_DTYPES,
    content_channels=3,
    supports_tiling=True,
    tiling=CodecTiling(
        decode_tile=(999, 32, 32),
        decode_overlap=(1, 8, 8),
        encode_tile=(9999, 512, 512),
        encode_overlap=(1, 64, 64),
    ),
)

WAN21_FLOW_RVS_CODEC = CodecDescriptor(
    id="dinkster.wan21_flow_rvs_vae",
    display_name="Wan 2.1 FlowRVS mask VAE",
    kind="video",
    latent=WAN21_LATENT,
    supported_dtypes=_INFERENCE_DTYPES,
    content_channels=1,
    supports_tiling=True,
    tiling=CodecTiling(
        decode_tile=(999, 32, 32),
        decode_overlap=(1, 8, 8),
        encode_tile=(9999, 512, 512),
        encode_overlap=(1, 64, 64),
    ),
)

WAN22_LATENT = LatentDescriptor(
    channels=48,
    dimensions=3,
    spatial_downscale=16,
    temporal_downscale=4,
    temporal_causal=True,
    content_fps=16.0,
    rgb_factors=_WAN22_RGB_FACTORS,
    rgb_bias=(0.0317, -0.0878, -0.1388),
    taesd_decoder="lighttaew2_2",
)

WAN22_CODEC = CodecDescriptor(
    id="dinkster.wan22_vae",
    display_name="Wan 2.2 causal video VAE",
    kind="video",
    latent=WAN22_LATENT,
    supported_dtypes=_INFERENCE_DTYPES,
    content_channels=3,
    supports_tiling=True,
    tiling=CodecTiling(
        decode_tile=(999, 32, 32),
        decode_overlap=(1, 8, 8),
        encode_tile=(9999, 512, 512),
        encode_overlap=(1, 64, 64),
    ),
)

WAN22_SIGMAS = FlowSigmas(shift=8.0)
WAN22_SAMPLING = SamplingDescriptor(
    parameterization=Parameterization.FLOW,
    sigma_min=WAN22_SIGMAS.sigma_min,
    sigma_max=WAN22_SIGMAS.sigma_max,
    shift=8.0,
)


@dataclass(frozen=True, slots=True)
class _NormalizedDetectionSource:
    geometries: Mapping[str, TensorGeometry]

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometries", MappingProxyType(dict(self.geometries)))

    def keys(self) -> tuple[str, ...]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return MappingProxyType({})


def _quantized_detection_source(source: WeightSource, keys: frozenset[str]) -> WeightSource | None:
    if not any(key.endswith(".comfy_quant") for key in keys):
        return source
    try:
        geometries = {key: source.entry(key).geometry for key in keys}
    except KeyError:
        return None
    try:
        split = split_quantization(
            geometries,
            source.metadata(),
            payload_reader=(
                source.read_uint8_configuration
                if isinstance(source, ConfigurationPayloadSource)
                else None
            ),
        )
    except (KeyError, QuantizationError):
        return None
    normalized = dict(split.architecture)
    for quant in split.layers.values():
        geometry = normalized[quant.weight]
        normalized[quant.weight] = TensorGeometry(geometry.shape, FLOAT32)
    return _NormalizedDetectionSource(normalized)


def _geometry(source: WeightSource, keys: frozenset[str], key: str) -> TensorGeometry | None:
    if key not in keys:
        return None
    try:
        return source.entry(key).geometry
    except KeyError:
        return None


def _exact_geometry(
    source: WeightSource,
    keys: frozenset[str],
    key: str,
    shape: tuple[int, ...],
) -> bool:
    geometry = _geometry(source, keys, key)
    return geometry is not None and geometry.shape == shape and geometry.dtype.kind == "float"


def _block_indices(keys: frozenset[str], prefix: str, layers: int) -> frozenset[int] | None:
    root = prefix + "blocks."
    valid_text = frozenset(str(index) for index in range(layers))
    indices: set[int] = set()
    for key in keys:
        if not key.startswith(root):
            continue
        text, separator, remainder = key[len(root) :].partition(".")
        if (
            not separator
            or not remainder
            or not text.isascii()
            or not text.isdecimal()
            or text not in valid_text
        ):
            return None
        indices.add(int(text))
    return frozenset(indices)


def _match_profile(
    source: WeightSource,
    keys: frozenset[str],
    prefix: str,
    profile: _Profile,
    family_id: str,
) -> DetectionEvidence | None:
    if _block_indices(keys, prefix, profile.layers) != frozenset(range(profile.layers)):
        return None
    if profile.vace_layers is None:
        if prefix + "vace_patch_embedding.weight" in keys:
            return None
    elif _block_indices(keys, prefix + "vace_", profile.vace_layers) != frozenset(
        range(profile.vace_layers)
    ):
        return None
    camera_marker = prefix + "control_adapter.conv.weight"
    if (camera_marker in keys) != (profile.camera_channels is not None):
        return None
    has_animate_state = any(
        key.startswith(prefix + root) for key in keys for root in _ANIMATE_ROOTS
    )
    if has_animate_state != (profile.model_variant == "animate"):
        return None
    if profile.model_variant == "animate" and prefix + _ANIMATE_MARKER not in keys:
        return None
    has_scail_pose = prefix + "patch_embedding_pose.weight" in keys
    has_scail_mask = prefix + "patch_embedding_mask.weight" in keys
    if has_scail_pose != (profile.model_variant in ("scail", "scail2")):
        return None
    if has_scail_mask != (profile.model_variant == "scail2"):
        return None
    has_s2v_state = any(key.startswith(prefix + root) for key in keys for root in _S2V_ROOTS)
    if has_s2v_state != (profile.model_variant == "s2v"):
        return None
    if profile.model_variant == "s2v" and prefix + _S2V_MARKER not in keys:
        return None
    has_humo_state = any(
        key.startswith(prefix + "audio_proj.")
        or (key.startswith(prefix + "blocks.") and ".audio_cross_attn_wrapper." in key)
        for key in keys
    )
    if has_humo_state != (profile.model_variant == "humo"):
        return None
    if profile.model_variant == "humo" and prefix + _HUMO_MARKER not in keys:
        return None
    has_wandancer_state = prefix + "patch_embedding_global.weight" in keys
    if has_wandancer_state != (profile.model_variant == "wandancer"):
        return None
    if any(key.startswith(prefix + "full_ref") for key in keys):
        return None

    width = profile.hidden_width
    required: dict[str, tuple[int, ...]] = {
        "head.modulation": (1, 2, width),
        "head.head.weight": (math.prod((1, 2, 2)) * profile.output_channels, width),
        "patch_embedding.weight": (width, profile.input_channels, 1, 2, 2),
        "text_embedding.0.weight": (width, 4096),
        "time_embedding.0.weight": (width, 256),
    }
    for index in range(profile.layers):
        block = f"blocks.{index}."
        required.update(
            {
                block + "modulation": (1, 6, width),
                block + "self_attn.q.weight": (width, width),
                block + "self_attn.k.weight": (width, width),
                block + "self_attn.v.weight": (width, width),
                block + "self_attn.o.weight": (width, width),
                block + "self_attn.norm_q.weight": (width,),
                block + "self_attn.norm_k.weight": (width,),
                block + "cross_attn.q.weight": (width, width),
                block + "cross_attn.k.weight": (width, width),
                block + "cross_attn.v.weight": (width, width),
                block + "cross_attn.o.weight": (width, width),
                block + "cross_attn.norm_q.weight": (width,),
                block + "cross_attn.norm_k.weight": (width,),
                block + "ffn.0.weight": (profile.ffn_width, width),
                block + "ffn.2.weight": (width, profile.ffn_width),
            }
        )
        if profile.model_type == "i2v":
            required.update(
                {
                    block + "cross_attn.k_img.weight": (width, width),
                    block + "cross_attn.v_img.weight": (width, width),
                    block + "cross_attn.norm_k_img.weight": (width,),
                }
            )
    if profile.vace_layers is not None:
        required["vace_patch_embedding.weight"] = (width, 96, 1, 2, 2)
        for index in range(profile.vace_layers):
            block = f"vace_blocks.{index}."
            required.update(
                {
                    block + "modulation": (1, 6, width),
                    block + "self_attn.q.weight": (width, width),
                    block + "self_attn.k.weight": (width, width),
                    block + "self_attn.v.weight": (width, width),
                    block + "self_attn.o.weight": (width, width),
                    block + "self_attn.norm_q.weight": (width,),
                    block + "self_attn.norm_k.weight": (width,),
                    block + "cross_attn.q.weight": (width, width),
                    block + "cross_attn.k.weight": (width, width),
                    block + "cross_attn.v.weight": (width, width),
                    block + "cross_attn.o.weight": (width, width),
                    block + "cross_attn.norm_q.weight": (width,),
                    block + "cross_attn.norm_k.weight": (width,),
                    block + "ffn.0.weight": (profile.ffn_width, width),
                    block + "ffn.2.weight": (width, profile.ffn_width),
                    block + "after_proj.weight": (width, width),
                }
            )
            before = prefix + block + "before_proj.weight"
            if index == 0:
                required[block + "before_proj.weight"] = (width, width)
            elif before in keys:
                return None
    if profile.model_type == "i2v":
        required.update(
            {
                "img_emb.proj.0.bias": (1280,),
                "img_emb.proj.1.weight": (1280, 1280),
                "img_emb.proj.3.weight": (width, 1280),
            }
        )
        emb_pos = "img_emb.emb_pos"
        if profile.flf_pos_embed_token_number is None:
            if prefix + emb_pos in keys:
                return None
        else:
            required[emb_pos] = (1, profile.flf_pos_embed_token_number, 1280)
    elif prefix + "img_emb.proj.0.bias" in keys or prefix + "img_emb.emb_pos" in keys:
        return None
    if profile.reference_channels is None:
        if prefix + "ref_conv.weight" in keys or prefix + "ref_conv.bias" in keys:
            return None
    else:
        required["ref_conv.weight"] = (width, profile.reference_channels, 2, 2)
        required["ref_conv.bias"] = (width,)
    if profile.camera_channels is not None:
        required.update(
            {
                "control_adapter.conv.weight": (
                    width,
                    profile.camera_channels * 64,
                    2,
                    2,
                ),
                "control_adapter.conv.bias": (width,),
                "control_adapter.residual_blocks.0.conv1.weight": (
                    width,
                    width,
                    3,
                    3,
                ),
                "control_adapter.residual_blocks.0.conv1.bias": (width,),
                "control_adapter.residual_blocks.0.conv2.weight": (
                    width,
                    width,
                    3,
                    3,
                ),
                "control_adapter.residual_blocks.0.conv2.bias": (width,),
            }
        )
    if profile.model_variant == "animate":
        required.update(
            _wan22_animate_layout(
                Wan21Config(
                    model_type="i2v",
                    model_variant="animate",
                    in_channels=profile.input_channels,
                    hidden_size=profile.hidden_width,
                    ffn_hidden_size=profile.ffn_width,
                    num_heads=profile.attention_heads,
                    num_layers=profile.layers,
                    out_channels=profile.output_channels,
                )
            )
        )
    if profile.model_variant in ("scail", "scail2"):
        required.update(
            {
                "patch_embedding_pose.weight": (width, 20, 1, 2, 2),
                "patch_embedding_pose.bias": (width,),
            }
        )
    if profile.model_variant == "scail2":
        required.update(
            {
                "patch_embedding_mask.weight": (width, 28, 1, 2, 2),
                "patch_embedding_mask.bias": (width,),
            }
        )
    if profile.model_variant == "s2v":
        required.update(wan21_layout(WAN22_S2V_14B))
    if profile.model_variant == "humo":
        required.update(wan21_layout(WAN21_HUMO_17B))
    if profile.model_variant == "wandancer":
        required.update(wan21_layout(WAN22_WANDANCER_14B))

    matched: list[str] = []
    for suffix, shape in required.items():
        key = prefix + suffix
        if not _exact_geometry(source, keys, key, shape):
            return None
        matched.append(key)

    field_values: dict[str, EvidenceValue] = {
        "attention_head_dim": 128,
        "attention_heads": profile.attention_heads,
        "flf": profile.flf_pos_embed_token_number is not None,
        "full_ref": profile.reference_channels is not None,
        "ffn_width": profile.ffn_width,
        "hidden_width": profile.hidden_width,
        "input_channels": profile.input_channels,
        "key_prefix": prefix,
        "layers": profile.layers,
        "model_type": profile.model_type,
        "model_variant": profile.model_variant,
        "output_channels": profile.output_channels,
        "parameter_count": profile.parameter_count,
        "patch": "1x2x2",
        "profile": profile.name,
        "ref_conv": profile.reference_channels is not None,
    }
    if profile.flf_pos_embed_token_number is not None:
        field_values["flf_pos_embed_token_number"] = profile.flf_pos_embed_token_number
    if profile.reference_channels is not None:
        field_values["reference_channels"] = profile.reference_channels
    if profile.camera_channels is not None:
        field_values["camera"] = True
        field_values["camera_channels"] = profile.camera_channels
    if profile.vace_layers is not None:
        field_values["vace"] = True
        field_values["vace_layers"] = profile.vace_layers
        field_values["vace_mapping_step"] = profile.layers // profile.vace_layers
    fields: Mapping[str, EvidenceValue] = MappingProxyType(field_values)
    return DetectionEvidence(family_id, tuple(matched), fields)


def _transformer_metadata_profile(
    source: WeightSource,
) -> Literal[
    "default", "animate2", "bernini", "flow_rvs", "causal_ar", "s2v", "wandancer", "refuse"
]:
    metadata = source.metadata()
    model_type = metadata.get("model_type")
    if model_type in ("wanvideo_wantodance_local", "wanvideo_wantodance_global"):
        return "wandancer" if set(metadata) == {"model_type"} else "refuse"
    bernini = model_type in ("bernini_high", "bernini_low")
    if model_type is not None and not bernini:
        return "refuse"
    encoded = metadata.get("config")
    if encoded is None:
        return "bernini" if bernini else "default"
    try:
        decoded = cast("object", json.loads(encoded))
    except json.JSONDecodeError:
        return "refuse"
    if type(decoded) is not dict:
        return "refuse"
    config = cast("dict[object, object]", cast("Any", decoded))
    transformer: object = config.get("transformer", {})
    if type(transformer) is not dict:
        return "refuse"
    transformer_config = cast("dict[object, object]", cast("Any", transformer))
    if not transformer_config:
        return "bernini" if bernini else "default"
    if bernini:
        return "refuse"
    if transformer_config in (
        {"model_type": "animate2"},
        {"image_model": "wan2.1", "model_type": "animate2"},
    ):
        return "animate2"
    if transformer_config == {"image_model": "wan2.1", "model_type": "s2v"}:
        return "s2v"
    if config == {"transformer": {"model_type": "flow_rvs"}}:
        return "flow_rvs"
    if (
        config == {"transformer": {"causal_ar": True}}
        and type(transformer_config["causal_ar"]) is bool
    ):
        return "causal_ar"
    return "refuse"


def detect_wan21(source: WeightSource) -> DetectionEvidence | None:
    """Return evidence for the 16-channel Wan 2.1/2.2 architecture family."""

    metadata_profile = _transformer_metadata_profile(source)
    if metadata_profile == "refuse":
        return None
    profiles = tuple(
        profile
        for profile in _WAN21_PROFILES
        if (
            profile.model_variant == "animate2"
            and metadata_profile == "animate2"
            or profile.model_variant == "bernini"
            and metadata_profile == "bernini"
            or profile.model_variant == "flow_rvs"
            and metadata_profile == "flow_rvs"
            or profile.model_variant == "causal_ar"
            and metadata_profile == "causal_ar"
            or profile.model_variant == "s2v"
            and metadata_profile in ("default", "s2v")
            or profile.model_variant == "wandancer"
            and metadata_profile in ("default", "wandancer")
            or profile.model_variant
            not in ("animate2", "bernini", "flow_rvs", "causal_ar", "s2v", "wandancer")
            and metadata_profile == "default"
        )
    )
    keys = frozenset(source.keys())
    detection_source = _quantized_detection_source(source, keys)
    if detection_source is None:
        return None
    if detection_source is not source:
        keys = frozenset(detection_source.keys())
    matches: list[DetectionEvidence] = []
    for prefix in _PREFIXES:
        for profile in profiles:
            evidence = _match_profile(detection_source, keys, prefix, profile, _FAMILY_ID)
            if evidence is not None:
                matches.append(evidence)
    return matches[0] if len(matches) == 1 else None


def detect_wan22(source: WeightSource) -> DetectionEvidence | None:
    """Return immutable evidence for the official Wan 2.2 TI2V 5B profile."""

    if _transformer_metadata_profile(source) != "default":
        return None
    keys = frozenset(source.keys())
    detection_source = _quantized_detection_source(source, keys)
    if detection_source is None:
        return None
    if detection_source is not source:
        keys = frozenset(detection_source.keys())
    matches: list[DetectionEvidence] = []
    for prefix in _PREFIXES:
        for profile in _WAN22_PROFILES:
            evidence = _match_profile(detection_source, keys, prefix, profile, _WAN22_FAMILY_ID)
            if evidence is not None:
                matches.append(evidence)
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True, slots=True)
class Wan21Detector:
    """FamilyDetector for the 16-channel Wan 2.1/2.2 architecture family."""

    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        return detect_wan21(source)


@dataclass(frozen=True, slots=True)
class Wan22Detector:
    """FamilyDetector implementation for the official Wan 2.2 core profile."""

    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        return detect_wan22(source)
