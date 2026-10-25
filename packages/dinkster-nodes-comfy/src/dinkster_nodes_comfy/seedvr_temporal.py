"""Native SeedVR2 temporal latent chunking and recombination."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_api.v1 import (
    CORE_INT,
    LATENT_TYPE,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)

_LATENT = TypeExpr.concrete(LATENT_TYPE)
_LATENT_LIST = TypeExpr.list_of(_LATENT)
_INT = TypeExpr.concrete(CORE_INT)
_LATENT_CHANNELS = 16
_SPATIAL_DOWNSAMPLE = 8
_CHUNK_GIB_PER_MPX_FRAME = 0.55
_CHUNK_RESERVED_GIB = 8.5
_CHUNK_SIGMA_GIB = 0.55
_CHUNK_SIGMA_K = 4
_MAX_FRAMES = 16384


def _integer(
    name: str, default: int, minimum: int, maximum: int, step: int | None = None
) -> InputSpec:
    return InputSpec(
        name,
        _INT,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _combo_values(
    combo_id: str, selected: str | Mapping[str, Any], values: Mapping[str, Any]
) -> Mapping[str, Any]:
    if isinstance(selected, Mapping):
        return selected
    prefix = f"{combo_id}."
    return {
        combo_id: selected,
        **{key.removeprefix(prefix): value for key, value in values.items()},
    }


def _samples(latent: object, node_name: str) -> tuple[Mapping[str, object], Any]:
    value = materialize_latent(latent)
    if not isinstance(value, Mapping):
        raise ValueError(f"{node_name}: expected a LATENT mapping.")
    mapping = cast("Mapping[str, object]", value)
    samples: Any = mapping.get("samples")
    if samples is None or not hasattr(samples, "ndim"):
        raise ValueError(f"{node_name}: latent is missing tensor samples.")
    return mapping, samples


def _crossfade_weights(overlap: int, device: object, dtype: object) -> Any:
    torch = importlib.import_module("torch")
    ramp = torch.linspace(0.0, 1.0, steps=overlap, device=device, dtype=dtype)
    ramp = ((ramp - 1.0 / 3.0) / (1.0 / 3.0)).clamp(0.0, 1.0)
    return 0.5 + 0.5 * torch.cos(torch.pi * ramp)


class SeedVR2TemporalChunk(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SeedVR2TemporalChunk",
            display_name="Split SeedVR2 Latent",
            category="model/latent/seedvr",
            description="Split a SeedVR2 video latent into overlapping temporal chunks.",
            search_terms=("seedvr2", "split", "chunk", "temporal", "video upscale", "rebatch"),
            inputs=(
                InputSpec("latent", _LATENT, doc="The VAE-encoded SeedVR2 latent to split."),
                _integer("temporal_overlap", 0, 0, _MAX_FRAMES),
            ),
            combos=(
                DynamicComboSpec(
                    "chunking_mode",
                    (
                        DynamicComboOption("auto"),
                        DynamicComboOption(
                            "manual", (_integer("frames_per_chunk", 21, 1, _MAX_FRAMES, 4),)
                        ),
                    ),
                    doc="manual uses frames_per_chunk; auto uses free device memory.",
                ),
            ),
            outputs=(
                OutputSpec("latents", _LATENT_LIST, doc="Temporal chunks in sequence order."),
                OutputSpec("temporal_overlap", _INT, doc="Effective latent-frame overlap."),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        latent: object,
        temporal_overlap: int,
        chunking_mode: str | Mapping[str, Any],
        **values: Any,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        source, samples = _samples(latent, "SeedVR2TemporalChunk")
        if samples.ndim != 5:
            raise ValueError(
                "SeedVR2TemporalChunk: expected a 5-D video latent (B, C, T, H, W); "
                f"got shape {tuple(samples.shape)}."
            )
        if samples.shape[1] != _LATENT_CHANNELS:
            raise ValueError(
                f"SeedVR2TemporalChunk: expected {_LATENT_CHANNELS} latent channels; "
                f"got {samples.shape[1]}."
            )
        if temporal_overlap < 0:
            raise ValueError(
                f"SeedVR2TemporalChunk: temporal_overlap must be >= 0; got {temporal_overlap}."
            )
        mode_values = _combo_values("chunking_mode", chunking_mode, values)
        mode = mode_values["chunking_mode"]
        if mode not in ("auto", "manual"):
            raise ValueError(f"SeedVR2TemporalChunk: unknown chunking_mode {mode!r}.")
        latent_frames = samples.shape[2]
        pixel_frames = 4 * (latent_frames - 1) + 1
        if mode == "auto":
            management = importlib.import_module("dinkster_inference.model_management")
            free_gib = management.get_free_memory(management.get_torch_device()) / (1024**3)
            mpx_per_frame = (
                samples.shape[0]
                * samples.shape[3]
                * samples.shape[4]
                * _SPATIAL_DOWNSAMPLE**2
                / 1e6
            )
            budget_gib = free_gib - _CHUNK_RESERVED_GIB - _CHUNK_SIGMA_K * _CHUNK_SIGMA_GIB
            maximum_latent = max(1, int(budget_gib / (_CHUNK_GIB_PER_MPX_FRAME * mpx_per_frame)))
            frames_per_chunk = min(4 * (maximum_latent - 1) + 1, pixel_frames)
        else:
            frames_per_chunk = mode_values["frames_per_chunk"]
            if frames_per_chunk < 1 or (frames_per_chunk - 1) % 4:
                raise ValueError(
                    "SeedVR2TemporalChunk: frames_per_chunk must be a 4n+1 pixel-frame count; "
                    f"got {frames_per_chunk}."
                )
        if pixel_frames <= frames_per_chunk:
            return cls.outputs(latents=[dict(source)], temporal_overlap=0)
        chunk_frames = (frames_per_chunk - 1) // 4 + 1
        overlap = min(temporal_overlap, chunk_frames - 1)
        step = chunk_frames - overlap
        chunks: list[dict[str, object]] = []
        for start in range(0, latent_frames, step):
            end = min(start + chunk_frames, latent_frames)
            chunks.append({**source, "samples": samples[:, :, start:end].contiguous()})
            if end >= latent_frames:
                break
        return cls.outputs(latents=chunks, temporal_overlap=overlap)


class SeedVR2TemporalMerge(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SeedVR2TemporalMerge",
            display_name="Merge SeedVR2 Latents",
            category="model/latent/seedvr",
            description="Recombine SeedVR2 temporal chunks with a Hann overlap crossfade.",
            search_terms=("seedvr2", "merge", "temporal", "hann", "crossfade"),
            inputs=(
                InputSpec(
                    "latents", _LATENT_LIST, doc="Sampled temporal chunks in sequence order."
                ),
                InputSpec("temporal_overlap", _INT, default=0, force_input=True),
            ),
            outputs=(OutputSpec("latent", _LATENT, doc="The recombined latent."),),
        )

    @classmethod
    def execute(
        cls, *, latents: Sequence[object], temporal_overlap: int | Sequence[int]
    ) -> Mapping[str, object]:
        initialize_inference_process()
        if isinstance(temporal_overlap, Sequence) and not isinstance(temporal_overlap, str):
            temporal_overlap = temporal_overlap[0]
        temporal_overlap = cast("int", temporal_overlap)
        if temporal_overlap < 0:
            raise ValueError(
                f"SeedVR2TemporalMerge: temporal_overlap must be >= 0; got {temporal_overlap}."
            )
        if not latents:
            raise ValueError("SeedVR2TemporalMerge: expected at least one latent chunk.")
        first_latent, first = _samples(latents[0], "SeedVR2TemporalMerge")
        if first.ndim != 5:
            raise ValueError(
                "SeedVR2TemporalMerge: expected 5-D video latents (B, C, T, H, W); "
                f"chunk 0 has shape {tuple(first.shape)}."
            )
        chunks = [first]
        for index, latent in enumerate(latents[1:], start=1):
            _, chunk = _samples(latent, "SeedVR2TemporalMerge")
            if chunk.shape[:2] != first.shape[:2] or chunk.shape[3:] != first.shape[3:]:
                raise ValueError(
                    f"SeedVR2TemporalMerge: chunk {index} shape {tuple(chunk.shape)} "
                    "does not match "
                    "chunk 0 outside the temporal axis."
                )
            if index < len(latents) - 1 and chunk.shape[2] != first.shape[2]:
                raise ValueError(
                    f"SeedVR2TemporalMerge: chunk {index} has {chunk.shape[2]} latent frames but "
                    f"chunk 0 has {first.shape[2]}; only the final chunk may be shorter."
                )
            chunks.append(chunk)
        output = {key: value for key, value in first_latent.items() if key != "noise_mask"}
        if len(chunks) == 1:
            return cls.outputs(latent={**output, "samples": first})
        torch = importlib.import_module("torch")
        if temporal_overlap == 0:
            return cls.outputs(latent={**output, "samples": torch.cat(chunks, dim=2)})
        chunk_frames = first.shape[2]
        step = chunk_frames - min(temporal_overlap, chunk_frames - 1)
        total_frames = step * (len(chunks) - 1) + chunks[-1].shape[2]
        batch, channels, _, height, width = first.shape
        merged = torch.empty(
            (batch, channels, total_frames, height, width), device=first.device, dtype=first.dtype
        )
        merged[:, :, :chunk_frames] = first
        filled = chunk_frames
        for index, chunk in enumerate(chunks[1:], start=1):
            start = index * step
            end = start + chunk.shape[2]
            fade = min(filled - start, chunk.shape[2])
            if fade:
                previous = _crossfade_weights(fade, chunk.device, chunk.dtype).view(
                    1, 1, fade, 1, 1
                )
                merged[:, :, start : start + fade] = merged[
                    :, :, start : start + fade
                ] * previous + chunk[:, :, :fade] * (1.0 - previous)
                merged[:, :, start + fade : end] = chunk[:, :, fade:]
            else:
                merged[:, :, start:end] = chunk
            filled = end
        return cls.outputs(latent={**output, "samples": merged})


SEEDVR_TEMPORAL_NODES = (SeedVR2TemporalChunk, SeedVR2TemporalMerge)
