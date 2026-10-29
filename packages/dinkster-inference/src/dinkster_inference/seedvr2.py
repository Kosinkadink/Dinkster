"""Torch-free SeedVR2 diffusion model descriptions and strict detection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .devices import BFLOAT16, FLOAT16, FLOAT32, DType
from .families import EvidenceValue
from .spaces import FlowSigmas
from .weights import TensorGeometry, WeightSource


class SeedVR2DetectError(ValueError):
    """A header is not one of the supported SeedVR2 diffusion layouts."""


@dataclass(frozen=True, slots=True)
class SeedVR2Config:
    variant: str
    width: int
    heads: int
    layers: int
    separate_layers: int
    mlp_type: str
    rope_dim: int
    rope_type: str
    vid_out_norm: bool = False
    family_id: str = "dinkster.seedvr2"
    latent_channels: int = 16
    input_channels: int = 33
    text_input_width: int = 5120
    head_dim: int = 128
    time_width: int = 256
    patch: tuple[int, int, int] = (1, 2, 2)
    window: tuple[int, int, int] = (4, 3, 3)
    norm_eps: float = 1e-5
    sampling_shift: float = 1.0
    inference_dtypes: tuple[DType, ...] = (BFLOAT16, FLOAT16, FLOAT32)
    memory_factor: float = 2.0

    def __post_init__(self) -> None:
        accepted = {
            ("3b_swiglu", 2560, 20, 32, 10, "swiglu", 128, "mmrope3d", True),
            ("7b_swiglu", 3072, 24, 36, 10, "swiglu", 64, "rope3d", False),
            ("7b_mlp", 3072, 24, 36, 36, "normal", 64, "rope3d", False),
        }
        identity = (
            self.variant,
            self.width,
            self.heads,
            self.layers,
            self.separate_layers,
            self.mlp_type,
            self.rope_dim,
            self.rope_type,
            self.vid_out_norm,
        )
        fixed = (
            self.family_id,
            self.latent_channels,
            self.input_channels,
            self.text_input_width,
            self.head_dim,
            self.time_width,
            self.patch,
            self.window,
            self.norm_eps,
            self.sampling_shift,
            self.inference_dtypes,
            self.memory_factor,
        )
        expected = (
            "dinkster.seedvr2",
            16,
            33,
            5120,
            128,
            256,
            (1, 2, 2),
            (4, 3, 3),
            1e-5,
            1.0,
            (BFLOAT16, FLOAT16, FLOAT32),
            2.0,
        )
        if identity not in accepted or any(
            type(value) is not type(required) or value != required
            for value, required in zip(fixed, expected, strict=True)
        ):
            raise ValueError("SeedVR2Config only represents a published SeedVR2 variant")


SEEDVR2_3B = SeedVR2Config("3b_swiglu", 2560, 20, 32, 10, "swiglu", 128, "mmrope3d", True)
# Official revision 10f035ad has no 7B SwiGLU checkpoint; pinned synthetic
# ComfyUI parity goldens cover this compatibility layout.
SEEDVR2_7B = SeedVR2Config("7b_swiglu", 3072, 24, 36, 10, "swiglu", 64, "rope3d")
SEEDVR2_7B_MLP = SeedVR2Config("7b_mlp", 3072, 24, 36, 36, "normal", 64, "rope3d")
SEEDVR2_CONFIGS = (SEEDVR2_3B, SEEDVR2_7B, SEEDVR2_7B_MLP)
SEEDVR2_SIGMAS = FlowSigmas(shift=SEEDVR2_3B.sampling_shift)
_PREFIXES = ("model.diffusion_model.", "")


def _linear(
    layout: dict[str, tuple[int, ...]], key: str, out: int, inp: int, *, bias: bool = True
) -> None:
    layout[f"{key}.weight"] = (out, inp)
    if bias:
        layout[f"{key}.bias"] = (out,)


def _branch(
    layout: dict[str, tuple[int, ...]], root: str, branch: str, config: SeedVR2Config
) -> None:
    width = config.width
    _linear(layout, f"{root}.attn.proj_qkv.{branch}", width * 3, width, bias=False)
    _linear(layout, f"{root}.attn.proj_out.{branch}", width, width)
    layout[f"{root}.attn.norm_q.{branch}.weight"] = (config.head_dim,)
    layout[f"{root}.attn.norm_k.{branch}.weight"] = (config.head_dim,)
    if config.mlp_type == "swiglu":
        hidden = 256 * ((int(2 * width * 4 / 3) + 255) // 256)
        for name in ("proj_in_gate", "proj_in"):
            _linear(layout, f"{root}.mlp.{branch}.{name}", hidden, width, bias=False)
        _linear(layout, f"{root}.mlp.{branch}.proj_out", width, hidden, bias=False)
    else:
        _linear(layout, f"{root}.mlp.{branch}.proj_in", width * 4, width)
        _linear(layout, f"{root}.mlp.{branch}.proj_out", width, width * 4)
    for layer in ("attn", "mlp"):
        for mode in ("shift", "scale", "gate"):
            layout[f"{root}.ada.{branch}.{layer}_{mode}"] = (width,)


def seedvr2_layout(config: SeedVR2Config) -> Mapping[str, tuple[int, ...]]:
    """Return the immutable exact state layout for one diffusion variant."""
    if config not in SEEDVR2_CONFIGS:
        raise ValueError("unsupported SeedVR2 config")
    width = config.width
    layout: dict[str, tuple[int, ...]] = {
        "positive_conditioning": (58, config.text_input_width),
        "negative_conditioning": (64, config.text_input_width),
    }
    _linear(layout, "vid_in.proj", width, config.input_channels * 4)
    _linear(layout, "txt_in", width, config.text_input_width)
    _linear(layout, "emb_in.proj_in", width, config.time_width)
    _linear(layout, "emb_in.proj_hid", width, width)
    _linear(layout, "emb_in.proj_out", width * 6, width)
    for index in range(config.layers):
        root = f"blocks.{index}"
        layout[f"{root}.attn.rope.rope.freqs"] = (21 if config.rope_type == "mmrope3d" else 10,)
        if index < config.separate_layers:
            _branch(layout, root, "vid", config)
            _branch(layout, root, "txt", config)
        else:
            _branch(layout, root, "all", config)
    _linear(layout, "vid_out.proj", config.latent_channels * 4, width)
    if config.vid_out_norm:
        layout["vid_out_norm.weight"] = (width,)
        layout["vid_out_ada.out_shift"] = (width,)
        layout["vid_out_ada.out_scale"] = (width,)
    return MappingProxyType(layout)


def detect_seedvr2_config(geometries: Mapping[str, TensorGeometry]) -> SeedVR2Config:
    if not geometries:
        raise SeedVR2DetectError("empty state dict header")
    signatures = (
        ("blocks.35.mlp.vid.proj_out.weight", SEEDVR2_7B_MLP),
        ("blocks.35.mlp.all.proj_in_gate.weight", SEEDVR2_7B),
        ("blocks.31.mlp.all.proj_in_gate.weight", SEEDVR2_3B),
    )
    config = next((candidate for key, candidate in signatures if key in geometries), None)
    if config is None:
        raise SeedVR2DetectError("missing SeedVR2 variant signature")
    expected = seedvr2_layout(config)
    problems: list[str] = []
    for key, shape in expected.items():
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
        elif found.shape != shape:
            problems.append(f"{key}: expected shape {shape}, found {found.shape}")
    problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(expected)))
    if problems:
        raise SeedVR2DetectError("invalid SeedVR2 geometry: " + "; ".join(problems[:6]))
    return config


@dataclass(frozen=True, slots=True)
class SeedVR2Evidence:
    config: SeedVR2Config
    key_prefix: str
    matched_keys: tuple[str, ...]
    fields: Mapping[str, EvidenceValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


def detect_seedvr2(source: WeightSource) -> SeedVR2Evidence | None:
    keys = tuple(source.keys())
    for prefix in _PREFIXES:
        scoped = {
            key.removeprefix(prefix): source.entry(key).geometry
            for key in keys
            if key.startswith(prefix)
        }
        try:
            config = detect_seedvr2_config(scoped)
        except SeedVR2DetectError:
            continue
        return SeedVR2Evidence(
            config,
            prefix,
            tuple(sorted(prefix + key for key in scoped)),
            {
                "variant": config.variant,
                "width": config.width,
                "layers": config.layers,
                "key_prefix": prefix,
                "sampling_shift": config.sampling_shift,
            },
        )
    return None


__all__ = [
    "SEEDVR2_3B",
    "SEEDVR2_7B",
    "SEEDVR2_7B_MLP",
    "SEEDVR2_CONFIGS",
    "SEEDVR2_SIGMAS",
    "SeedVR2Config",
    "SeedVR2DetectError",
    "SeedVR2Evidence",
    "detect_seedvr2",
    "detect_seedvr2_config",
    "seedvr2_layout",
]
