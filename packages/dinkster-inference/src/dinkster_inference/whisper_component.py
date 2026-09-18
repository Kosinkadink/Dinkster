"""Strict Whisper Large v3 component planning and immutable identity."""

from __future__ import annotations

from pathlib import Path

from .assembly import ComponentPlan
from .devices import DType
from .identity import build_runtime_identity_from_facts, runtime_component_identity
from .weights import AssetIdentifiedSource, WeightSource
from .whisper import WHISPER_LARGE_V3, WhisperLargeV3Config, whisper_large_v3_layout

WHISPER_LARGE_V3_COMPONENT_FAMILY_ID = "dinkster.whisper-large-v3"
_ENCODER_PREFIX = "model."
_DECODER_CONTEXT = 448
_VOCAB_SIZE = 51_866


class WhisperLargeV3ComponentAssemblyError(ValueError):
    """A source is not the exact official Whisper Large v3 artifact."""


def _decoder_layout() -> dict[str, tuple[int, ...]]:
    state = WHISPER_LARGE_V3.n_audio_state
    hidden = state * 4
    layout: dict[str, tuple[int, ...]] = {
        "model.decoder.embed_positions.weight": (_DECODER_CONTEXT, state),
        "model.decoder.embed_tokens.weight": (_VOCAB_SIZE, state),
        "model.decoder.layer_norm.weight": (state,),
        "model.decoder.layer_norm.bias": (state,),
    }
    for index in range(WHISPER_LARGE_V3.n_audio_layer):
        prefix = f"model.decoder.layers.{index}"
        layer: dict[str, tuple[int, ...]] = {
            f"{prefix}.fc1.weight": (hidden, state),
            f"{prefix}.fc1.bias": (hidden,),
            f"{prefix}.fc2.weight": (state, hidden),
            f"{prefix}.fc2.bias": (state,),
            f"{prefix}.final_layer_norm.weight": (state,),
            f"{prefix}.final_layer_norm.bias": (state,),
        }
        for attention in ("self_attn", "encoder_attn"):
            attention_prefix = f"{prefix}.{attention}"
            layer.update(
                {
                    f"{attention_prefix}.q_proj.weight": (state, state),
                    f"{attention_prefix}.q_proj.bias": (state,),
                    f"{attention_prefix}.k_proj.weight": (state, state),
                    f"{attention_prefix}.v_proj.weight": (state, state),
                    f"{attention_prefix}.v_proj.bias": (state,),
                    f"{attention_prefix}.out_proj.weight": (state, state),
                    f"{attention_prefix}.out_proj.bias": (state,),
                    f"{prefix}.{attention}_layer_norm.weight": (state,),
                    f"{prefix}.{attention}_layer_norm.bias": (state,),
                }
            )
        layout.update(layer)
    return layout


def _official_artifact_layout() -> dict[str, tuple[int, ...]]:
    return {
        **{_ENCODER_PREFIX + key: shape for key, shape in whisper_large_v3_layout().items()},
        **_decoder_layout(),
    }


def plan_whisper_large_v3_component(
    source: WeightSource,
    *,
    path: Path,
) -> ComponentPlan[WhisperLargeV3Config]:
    """Plan the exact official full Whisper Large v3 artifact's encoder."""

    if getattr(source, "path", None) != path:
        raise WhisperLargeV3ComponentAssemblyError("Whisper source path differs from selection")
    if not isinstance(source, AssetIdentifiedSource):
        raise WhisperLargeV3ComponentAssemblyError("Whisper source must carry asset identity")
    digest = source.asset_digest
    size = source.asset_size
    if not digest or type(size) is not int or size < 0:
        raise WhisperLargeV3ComponentAssemblyError("Whisper source must carry asset identity")

    encoder_layout = whisper_large_v3_layout()
    decoder_layout = _decoder_layout()
    artifact_layout = _official_artifact_layout()
    source_keys = set(source.keys())
    if source_keys != set(artifact_layout):
        missing = sorted(set(artifact_layout) - source_keys)
        unexpected = sorted(source_keys - set(artifact_layout))
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise WhisperLargeV3ComponentAssemblyError(
            "Whisper source does not match the exact Large v3 artifact: " + "; ".join(details)
        )

    for source_key, shape in artifact_layout.items():
        geometry = source.entry(source_key).geometry
        if geometry.shape != shape:
            raise WhisperLargeV3ComponentAssemblyError(
                f"Whisper {source_key} expected shape {shape}, found {geometry.shape}"
            )
        if geometry.dtype.kind != "float":
            raise WhisperLargeV3ComponentAssemblyError(
                f"Whisper {source_key} requires floating-point storage, found {geometry.dtype.name}"
            )

    keys = {key: _ENCODER_PREFIX + key for key in encoder_layout}
    dtypes = {key: source.entry(source_key).geometry.dtype for key, source_key in keys.items()}
    return ComponentPlan(
        component="whisper-large-v3",
        path=path,
        config=WHISPER_LARGE_V3,
        keys=keys,
        dtypes=dtypes,
        quant={},
        ignored=tuple(sorted(decoder_layout)),
        identity_facts=(
            f"asset_digest={digest}",
            f"asset_size={size}",
            "artifact_layout=official-full-whisper-large-v3",
            "sample_rate=16000",
            "chunk_samples=480000",
            "audio_context=1500",
            "layer_outputs=33",
        ),
    )


def whisper_large_v3_component_runtime_identity(
    plan: ComponentPlan[WhisperLargeV3Config],
    compute_dtype: DType,
) -> str:
    """Build the native identity for one Whisper Large v3 component."""

    return build_runtime_identity_from_facts(
        WHISPER_LARGE_V3_COMPONENT_FAMILY_ID,
        runtime_component_identity(WHISPER_LARGE_V3_COMPONENT_FAMILY_ID, (plan,)),
        diffusion_dtype="unloaded",
        text_dtype=compute_dtype.name,
        vae_dtype="unloaded",
        fp8_matmul=False,
        runtime_facts=plan.runtime_facts,
    )


__all__ = [
    "WHISPER_LARGE_V3_COMPONENT_FAMILY_ID",
    "WhisperLargeV3ComponentAssemblyError",
    "plan_whisper_large_v3_component",
    "whisper_large_v3_component_runtime_identity",
]
