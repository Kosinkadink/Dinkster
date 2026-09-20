"""Model families as registrations, detection as evidence.

ComfyUI decides "what model is this" with a central ordered if-chain
plus a central ordered class list, first match wins, order silently
load-bearing (comfy/model_detection.py detect_unet_config,
comfy/supported_models.py models @ b78cec87). Supporting a new
architecture means editing both.

Here a family is a registration: its detector inspects a WeightSource
header and returns evidence (which keys matched, what was derived) or
None. The registry ranks evidence by explicit specificity - ties at the
top are an ambiguity DIAGNOSTIC, never a silent coin flip. New
architectures register from packs; the core is not edited.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol

from .devices import BFLOAT16, FLOAT32, DType
from .latents import LatentDescriptor, MultiStreamLatentDescriptor
from .sampling import SamplingDescriptor
from .weights import WeightSource

EvidenceValue = str | int | float | bool


@dataclass(frozen=True)
class DetectionEvidence:
    """Why a detector believes a checkpoint is its family.

    ``fields`` are derived architecture facts (widths, depths, channel
    counts) that feed the family's typed config; ``matched_keys`` are
    the signature keys that fired, kept for diagnostics.
    """

    family_id: str
    matched_keys: tuple[str, ...]
    fields: Mapping[str, EvidenceValue]

    def __post_init__(self) -> None:
        # Snapshot: a detector reusing/mutating its dict cannot change
        # evidence after the fact.
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


class FamilyDetector(Protocol):
    """Inspects a weight-source header; never reads payload bytes,
    never mutates anything (both are things ComfyUI's detection does)."""

    def detect(self, source: WeightSource) -> DetectionEvidence | None: ...


@dataclass(frozen=True)
class ComponentWiring:
    """Where a family's components live in a combined checkpoint and
    which text encoders it uses (ids, resolved via registries - not
    imported classes, unlike comfy/supported_models.py @ b78cec87)."""

    diffusion_prefix: str = "model.diffusion_model."
    vae_prefix: str = "first_stage_model."
    text_encoder_prefix: str = "cond_stage_model."
    text_encoders: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreviewDecoderProperties:
    """A family's quality-preview decoder kind and decoder-specific target."""

    kind: Literal["taesd", "taehv", "asset"]
    target: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("taesd", "taehv", "asset"):
            raise ValueError(f"unsupported preview decoder kind {self.kind!r}")
        if self.kind == "taehv":
            if self.target is not None:
                raise ValueError("TAEHV preview registration does not accept a target")
        elif not self.target:
            raise ValueError(f"{self.kind} preview registration requires a target")


@dataclass(frozen=True)
class EngineProperties:
    """Cross-cutting engine behavior supplied by family registration."""

    diffusion_dtype: DType = BFLOAT16
    text_dtype: DType = BFLOAT16
    vae_dtypes: tuple[DType, ...] = (BFLOAT16, FLOAT32)
    sigma_space: Literal["default", "flux"] = "default"
    regional_memory_factor: float | None = None
    clip_text_profile: Literal["none", "sd1", "sdxl"] = "none"
    adm_profile: Literal["none", "sdxl", "sdxl_refiner"] = "none"
    ipadapter_profile: Literal["none", "sd15"] = "none"
    controlnet_profile: Literal["none", "sd15", "sdxl"] = "none"
    preview_decoder: PreviewDecoderProperties | None = None
    compatibility_latent_formats: tuple[str, ...] = ()
    gguf_architecture: str | None = None

    def __post_init__(self) -> None:
        if not self.vae_dtypes:
            raise ValueError("vae_dtypes must not be empty")
        if self.regional_memory_factor is not None and self.regional_memory_factor <= 0:
            raise ValueError("regional_memory_factor must be positive")


@dataclass(frozen=True)
class ModelFamily:
    """One registered architecture family.

    ``specificity`` orders detection explicitly (higher wins; a variant
    like FluxInpaint must declare a higher value than its parent - what
    list position encodes implicitly in ComfyUI). ``memory_factor`` is
    the empirical estimation multiplier (supported_models
    memory_usage_factor @ b78cec87).
    """

    id: str
    display_name: str
    detector: FamilyDetector
    specificity: int
    latent: LatentDescriptor | MultiStreamLatentDescriptor
    sampling: SamplingDescriptor
    wiring: ComponentWiring
    supported_dtypes: frozenset[DType]
    memory_factor: float = 1.0
    aliases: tuple[str, ...] = ()
    engine: EngineProperties = field(default_factory=EngineProperties)

    def __post_init__(self) -> None:
        if self.memory_factor <= 0:
            raise ValueError("memory_factor must be positive")

    def single_stream_latent(self) -> LatentDescriptor:
        """The family's latent for single-stream execution paths.

        Multistream families are refused so a path built around one
        latent tensor can never silently process a stream pack."""
        if isinstance(self.latent, LatentDescriptor):
            return self.latent
        raise ValueError(
            f"family {self.id!r} declares a multistream latent;"
            " this path executes single-stream latents only"
        )


@dataclass(frozen=True)
class DetectionResult:
    """The registry's answer: best evidence (None = unrecognized), every
    candidate that fired (specificity-ordered), and whether the top was
    ambiguous. Ambiguity means callers surface a diagnostic naming the
    tied families instead of guessing."""

    best: DetectionEvidence | None
    candidates: tuple[DetectionEvidence, ...]
    ambiguous: tuple[str, ...] = ()


class FamilyRegistry:
    """Detection across registered families with explicit ordering."""

    def __init__(self) -> None:
        from .registry import Registry

        self._registry: Registry[ModelFamily] = Registry()

    def register(self, family: ModelFamily) -> None:
        self._registry.register(family)

    def get(self, id_or_alias: str) -> ModelFamily | None:
        return self._registry.get(id_or_alias)

    def ids(self) -> tuple[str, ...]:
        return self._registry.ids()

    def detect(self, source: WeightSource) -> DetectionResult:
        """Run every detector; rank evidence by (specificity desc, id).

        Evidence claiming a family id other than its own is a detector
        bug and raises. A specificity tie at the top is reported, and
        ``best`` stays None - unrecognized and ambiguous both refuse to
        guess."""
        scored: list[tuple[int, str, DetectionEvidence]] = []
        for family in self._registry:
            evidence = family.detector.detect(source)
            if evidence is None:
                continue
            if evidence.family_id != family.id:
                raise ValueError(
                    f"detector for {family.id!r} returned evidence for {evidence.family_id!r}"
                )
            scored.append((family.specificity, family.id, evidence))
        scored.sort(key=lambda item: (-item[0], item[1]))
        candidates = tuple(evidence for _, _, evidence in scored)
        if not scored:
            return DetectionResult(best=None, candidates=())
        top_specificity = scored[0][0]
        tied = tuple(fid for spec, fid, _ in scored if spec == top_specificity)
        if len(tied) > 1:
            return DetectionResult(best=None, candidates=candidates, ambiguous=tied)
        return DetectionResult(best=candidates[0], candidates=candidates)


__all__ = [
    "ComponentWiring",
    "DetectionEvidence",
    "DetectionResult",
    "EvidenceValue",
    "EngineProperties",
    "FamilyDetector",
    "FamilyRegistry",
    "ModelFamily",
    "PreviewDecoderProperties",
]
