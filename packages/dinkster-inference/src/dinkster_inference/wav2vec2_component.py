"""Strict Wav2Vec2 component planning and immutable identity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .assembly import ComponentPlan
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .wav2vec2 import (
    WAV2VEC2_CHINESE_BASE,
    WAV2VEC2_LARGE,
    Wav2Vec2Config,
    wav2vec2_layout,
)
from .weights import AssetIdentifiedSource, WeightSource

WAV2VEC2_COMPONENT_FAMILY_ID = "dinkster.wav2vec2"
_PREFIX = "wav2vec2."


@dataclass(frozen=True, slots=True)
class _ArtifactProfile:
    component: str
    config: Wav2Vec2Config
    markers: tuple[tuple[str, tuple[int, ...]], ...]


_LARGE_ARTIFACT = _ArtifactProfile(
    "wav2vec2-large",
    WAV2VEC2_LARGE,
    (
        ("lm_head.bias", (32,)),
        ("lm_head.weight", (32, 1024)),
    ),
)
_CHINESE_BASE_ARTIFACT = _ArtifactProfile(
    "wav2vec2-chinese-base",
    WAV2VEC2_CHINESE_BASE,
    (
        ("project_hid.bias", (256,)),
        ("project_hid.weight", (256, 768)),
        ("project_q.bias", (256,)),
        ("project_q.weight", (256, 256)),
        ("quantizer.codevectors", (1, 640, 128)),
        ("quantizer.weight_proj.bias", (640,)),
        ("quantizer.weight_proj.weight", (640, 512)),
    ),
)
_ARTIFACT_PROFILES = (_LARGE_ARTIFACT, _CHINESE_BASE_ARTIFACT)


class Wav2Vec2ComponentAssemblyError(ValueError):
    """A source is not an exact supported Wav2Vec2 component."""


def _expected_source_layout(profile: _ArtifactProfile) -> dict[str, tuple[int, ...]]:
    return {
        **{_PREFIX + key: shape for key, shape in wav2vec2_layout(profile.config).items()},
        **dict(profile.markers),
    }


def _detect_artifact_profile(source_keys: set[str]) -> _ArtifactProfile:
    expected_profiles = tuple(
        (profile, set(_expected_source_layout(profile))) for profile in _ARTIFACT_PROFILES
    )
    matches = tuple(profile for profile, expected in expected_profiles if source_keys == expected)
    if len(matches) == 1:
        return matches[0]
    details: list[str] = []
    for profile, expected in expected_profiles:
        differences: list[str] = []
        missing = sorted(expected - source_keys)
        unexpected = sorted(source_keys - expected)
        if missing:
            differences.append("missing " + ", ".join(missing))
        if unexpected:
            differences.append("unexpected " + ", ".join(unexpected))
        details.append(f"{profile.component}: " + "; ".join(differences))
    raise Wav2Vec2ComponentAssemblyError(
        "Wav2Vec2 source does not match an exact supported key layout (" + ") (".join(details) + ")"
    )


def plan_wav2vec2_component(source: WeightSource, *, path: Path) -> ComponentPlan[Wav2Vec2Config]:
    """Plan one exact official Wav2Vec2 encoder artifact."""

    if getattr(source, "path", None) != path:
        raise Wav2Vec2ComponentAssemblyError("Wav2Vec2 source path differs from selection")
    if not isinstance(source, AssetIdentifiedSource):
        raise Wav2Vec2ComponentAssemblyError("Wav2Vec2 source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise Wav2Vec2ComponentAssemblyError("Wav2Vec2 source must carry asset identity")
    source_keys = set(source.keys())
    profile = _detect_artifact_profile(source_keys)
    layout = wav2vec2_layout(profile.config)
    keys = {key: _PREFIX + key for key in layout}
    dtypes: dict[str, DType] = {}
    source_layout = _expected_source_layout(profile)
    for source_key, shape in source_layout.items():
        geometry = source.entry(source_key).geometry
        if geometry.shape != shape:
            raise Wav2Vec2ComponentAssemblyError(
                f"Wav2Vec2 {source_key} expected shape {shape}, found {geometry.shape}"
            )
        if geometry.dtype.kind != "float":
            raise Wav2Vec2ComponentAssemblyError(
                f"Wav2Vec2 {source_key} requires floating-point storage, "
                f"found {geometry.dtype.name}"
            )
        if source_key.startswith(_PREFIX):
            dtypes[source_key.removeprefix(_PREFIX)] = geometry.dtype
    return ComponentPlan(
        component=profile.component,
        path=path,
        config=profile.config,
        keys=keys,
        dtypes=dtypes,
        quant={},
        ignored=tuple(key for key, _shape in profile.markers),
        identity_facts=(
            f"asset_digest={digest}",
            f"asset_size={size}",
            f"sample_rate={profile.config.sample_rate}",
            f"layer_outputs={profile.config.num_layers + 1}",
        ),
    )


def wav2vec2_component_runtime_identity(
    plan: ComponentPlan[Wav2Vec2Config], compute_dtype: DType
) -> str:
    """Build the native identity for one Wav2Vec2 component."""

    return build_runtime_identity_from_facts(
        WAV2VEC2_COMPONENT_FAMILY_ID,
        runtime_component_identity(WAV2VEC2_COMPONENT_FAMILY_ID, (plan,)),
        diffusion_dtype="unloaded",
        text_dtype=compute_dtype.name,
        vae_dtype="unloaded",
        fp8_matmul=False,
        runtime_facts=plan.runtime_facts,
    )


__all__ = [
    "WAV2VEC2_COMPONENT_FAMILY_ID",
    "Wav2Vec2ComponentAssemblyError",
    "plan_wav2vec2_component",
    "wav2vec2_component_runtime_identity",
]
