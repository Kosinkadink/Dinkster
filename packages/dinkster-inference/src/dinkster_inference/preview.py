"""Declarative sampling-preview providers.

A provider turns in-flight sampler state (a latent tensor) into a small
displayable frame. Providers are DATA, not monkey patches: a spec names
what it produces (``kind``), what it costs (``cost``), and what a latent
descriptor must declare for it to apply (``requires``), optionally pinned
to one model family. Resolution is a pure function of (descriptor,
family, mode) over the registered specs - no import-order sensitivity,
no global mutation, no implicit latent-format coupling.

The specs here are torch-free. Decoder implementations live with their
execution backend (``dinkster_inference_torch.preview``) keyed by spec id;
the sampling arm resolves a spec, then asks its backend for the matching
decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dinkster_protocol import PreviewMode, validate_preview_mode

from .latents import LatentDescriptor

PreviewKind = Literal["image", "animation", "audio", "encoded_animation"]
PreviewCost = Literal["cheap", "model"]
PreviewRequirement = Literal["rgb_factors", "taesd_decoder", "family_codec", "latent"]

PREVIEW_KINDS = ("image", "animation", "audio", "encoded_animation")
PREVIEW_COSTS = ("cheap", "model")
PREVIEW_REQUIREMENTS = ("rgb_factors", "taesd_decoder", "family_codec", "latent")


@dataclass(frozen=True)
class PreviewFrame:
    """One decoded preview frame: HWC uint8 RGB pixels (any ndarray-like
    with tobytes()), plus its dimensions."""

    rgb: object
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("preview frame dimensions must be positive")


@dataclass(frozen=True)
class PreviewClip:
    """Decoded frames of an animated preview, addressed into a fixed ring.

    ``frame_indices`` name each frame's slot in [0, ``frame_count``); the
    frontend keeps one ring of ``frame_count`` slots per node and replaces
    slots as frames arrive. ``fps`` is the ring's display rate (None when
    the family fixes no rate)."""

    frames: tuple[PreviewFrame, ...]
    frame_indices: tuple[int, ...]
    frame_count: int
    fps: float | None = None

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("a preview clip must hold at least one frame")
        if len(self.frames) != len(self.frame_indices):
            raise ValueError("preview clip frames and frame_indices must align")
        if self.frame_count <= 0:
            raise ValueError("preview clip frame_count must be positive")
        if any(index < 0 or index >= self.frame_count for index in self.frame_indices):
            raise ValueError("preview clip frame_indices must lie in [0, frame_count)")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("preview clip fps must be positive")


@dataclass(frozen=True)
class EncodedPreviewAnimation:
    """One self-contained encoded animation preview (an animated WebP, a
    short MP4), shipped as-is with its container mime; the emitter never
    re-encodes it. It replaces the node's previous frame for its stream
    rather than addressing a frame ring."""

    data: bytes
    mime: str
    width: int
    height: int

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("encoded preview animation data must be non-empty")
        if not (self.mime.startswith("image/") or self.mime.startswith("video/")):
            raise ValueError(f"unsupported encoded preview animation mime: {self.mime!r}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("encoded preview animation dimensions must be positive")


@dataclass(frozen=True)
class PreviewProviderSpec:
    """One preview decoder's declarative face.

    ``requires`` names the latent-descriptor fact the decoder consumes:
    ``rgb_factors`` (a constant projection matrix), ``taesd_decoder``
    (a named TAE-class decoder asset), ``family_codec`` (the family
    runtime's own preview codec), or ``latent`` (nothing beyond the raw
    sampler state itself). ``family_id`` pins the spec to one
    family; None applies to any family whose descriptor satisfies
    ``requires``. ``dimensions`` pins the spec to descriptors with that
    many content dimensions (3 for animated video providers); None
    applies regardless."""

    id: str
    kind: PreviewKind
    cost: PreviewCost
    requires: PreviewRequirement
    family_id: str | None = None
    dimensions: int | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("preview provider id must be non-empty")
        if self.kind not in PREVIEW_KINDS:
            raise ValueError(f"unsupported preview kind: {self.kind!r}")
        if self.cost not in PREVIEW_COSTS:
            raise ValueError(f"unsupported preview cost: {self.cost!r}")
        if self.requires not in PREVIEW_REQUIREMENTS:
            raise ValueError(f"unsupported preview requirement: {self.requires!r}")
        if self.family_id is not None and not self.family_id:
            raise ValueError("preview provider family_id must be non-empty or None")
        if self.dimensions is not None and self.dimensions not in (1, 2, 3):
            raise ValueError(f"unsupported preview dimensions pin: {self.dimensions!r}")

    def matches(
        self,
        descriptor: LatentDescriptor,
        *,
        family_id: str,
        family_codec: bool = False,
    ) -> bool:
        if self.family_id is not None and self.family_id != family_id:
            return False
        if self.dimensions is not None and descriptor.dimensions != self.dimensions:
            return False
        if self.requires == "rgb_factors":
            return descriptor.rgb_factors is not None
        if self.requires == "taesd_decoder":
            return descriptor.taesd_decoder is not None
        if self.requires == "latent":
            return True
        return family_codec


class PreviewProviderRegistry:
    """Registered provider specs plus deterministic resolution.

    ``resolve`` orders every matching spec by mode-aware preference:
    ``off`` yields nothing; ``cheap`` admits only cheap providers;
    ``quality`` and ``auto`` admit everything, model-cost first. Within
    one preference tier a family-pinned spec beats a generic one and
    ids tie-break lexicographically, so resolution never depends on
    registration order."""

    def __init__(self) -> None:
        self._specs: dict[str, PreviewProviderSpec] = {}

    def register(self, spec: PreviewProviderSpec) -> None:
        if type(spec) is not PreviewProviderSpec:
            raise TypeError("preview providers must be exact PreviewProviderSpec values")
        existing = self._specs.get(spec.id)
        if existing is not None and existing != spec:
            raise ValueError(f"preview provider {spec.id!r} is already registered differently")
        self._specs[spec.id] = spec

    def specs(self) -> tuple[PreviewProviderSpec, ...]:
        return tuple(sorted(self._specs.values(), key=lambda spec: spec.id))

    def resolve(
        self,
        descriptor: LatentDescriptor,
        *,
        family_id: str,
        mode: PreviewMode,
        kinds: tuple[PreviewKind, ...] = ("image",),
        family_codec: bool = False,
    ) -> tuple[PreviewProviderSpec, ...]:
        validate_preview_mode(mode)
        if mode == "off":
            return ()
        candidates = [
            spec
            for spec in self._specs.values()
            if spec.kind in kinds
            and (mode != "cheap" or spec.cost == "cheap")
            and spec.matches(descriptor, family_id=family_id, family_codec=family_codec)
        ]
        candidates.sort(
            key=lambda spec: (
                0 if spec.cost == "model" else 1,
                0 if spec.family_id is not None else 1,
                spec.id,
            )
        )
        return tuple(candidates)


LATENT2RGB_PROVIDER = PreviewProviderSpec(
    id="dinkster.latent2rgb",
    kind="image",
    cost="cheap",
    requires="rgb_factors",
)

TAESD_PROVIDER = PreviewProviderSpec(
    id="dinkster.taesd",
    kind="image",
    cost="model",
    requires="taesd_decoder",
)

LATENT2RGB_ANIMATION_PROVIDER = PreviewProviderSpec(
    id="dinkster.latent2rgb.animation",
    kind="animation",
    cost="cheap",
    requires="rgb_factors",
    dimensions=3,
)

TAEHV_PROVIDER = PreviewProviderSpec(
    id="dinkster.taehv",
    kind="animation",
    cost="model",
    requires="taesd_decoder",
    dimensions=3,
)

LATENT2WAVEFORM_PROVIDER = PreviewProviderSpec(
    id="dinkster.latent2waveform",
    kind="audio",
    cost="cheap",
    requires="latent",
    dimensions=1,
)

LATENT2RGB_WEBP_PROVIDER = PreviewProviderSpec(
    id="dinkster.latent2rgb.webp",
    kind="encoded_animation",
    cost="cheap",
    requires="rgb_factors",
    dimensions=3,
)

TRIPOSPLAT_SPLAT_PROVIDER = PreviewProviderSpec(
    id="dinkster.triposplat.splat",
    kind="image",
    cost="model",
    requires="family_codec",
    family_id="dinkster.triposplat",
    dimensions=1,
)


def preview_fps(descriptor: LatentDescriptor) -> float | None:
    """The display rate of one preview frame per latent timestep, derived
    from the descriptor's content rate; None when the family fixes none."""
    if descriptor.content_fps is None:
        return None
    return descriptor.content_fps / descriptor.temporal_downscale


def builtin_preview_registry() -> PreviewProviderRegistry:
    """A fresh registry holding the core providers."""
    registry = PreviewProviderRegistry()
    registry.register(LATENT2RGB_PROVIDER)
    registry.register(TAESD_PROVIDER)
    registry.register(LATENT2RGB_ANIMATION_PROVIDER)
    registry.register(LATENT2RGB_WEBP_PROVIDER)
    registry.register(TAEHV_PROVIDER)
    registry.register(LATENT2WAVEFORM_PROVIDER)
    registry.register(TRIPOSPLAT_SPLAT_PROVIDER)
    return registry


__all__ = [
    "LATENT2RGB_ANIMATION_PROVIDER",
    "LATENT2RGB_PROVIDER",
    "LATENT2RGB_WEBP_PROVIDER",
    "LATENT2WAVEFORM_PROVIDER",
    "PREVIEW_COSTS",
    "PREVIEW_KINDS",
    "PREVIEW_REQUIREMENTS",
    "EncodedPreviewAnimation",
    "PreviewClip",
    "PreviewCost",
    "PreviewFrame",
    "PreviewKind",
    "PreviewProviderRegistry",
    "PreviewProviderSpec",
    "PreviewRequirement",
    "TAEHV_PROVIDER",
    "TAESD_PROVIDER",
    "TRIPOSPLAT_SPLAT_PROVIDER",
    "builtin_preview_registry",
    "preview_fps",
]
