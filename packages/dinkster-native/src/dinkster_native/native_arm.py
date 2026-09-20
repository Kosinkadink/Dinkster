"""Dinkster-native execution bodies for the compat pack's managed nodes.

The public schemas stay owned by :mod:`dinkster_native.native`; these
subclasses replace only execution when the host selects the ``native`` arm.
Importing this module is deliberately torch-free. Torch and the native torch
runtime are resolved only while an arm body executes, so the root engine venv
can still inspect and validate every schema without installing torch.

One checkpoint or split-source bundle has one resident identity. Its assembled
diffusion, text, and VAE modules enroll as separate per-unit residency
mechanisms; each node body places only the stage it consumes while one
process-wide lock excludes concurrent placement and advisory memory shedding.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import logging
import math
import os
import platform
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
from functools import partial, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, cast

from dinkster_assets import AssetRef, EmbeddingNameIndex, resolver_from_env
from dinkster_inference.registries import builtin_registries as _builtin_inference_registries
from dinkster_inference.runtime import uses_classic_embedding_bindings
from dinkster_inference.sampling_wire import NoiseSelection, SamplerSelection, SigmaSchedule
from dinkster_memory import AcceleratorMemoryPolicyError
from dinkster_nodes_generation import (
    GENERATION_NODES,
    MODEL3D_GENERATION_NODES,
    clean_enhanced_prompt,
    prepare_ltx2_prompt,
)
from dinkster_schema import (
    AssetWidget,
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputFamilySpec,
    InputSpec,
    MappingSource,
    Node,
    NodeSchema,
    NumberWidget,
    OutputInterface,
    OutputSpec,
    ReplacementCase,
    ReplacementRule,
    StringWidget,
    TypeExpr,
    report_progress,
)
from dinkster_workers import current_execution_context

from .legacy_sources import resolve_weight_source
from .memory_policy import native_memory_policy
from .model_aware_schedules import (
    ALIGN_YOUR_STEPS_NOISE_LEVELS,
    GITS_NOISE_LEVELS,
    OPTIMAL_STEPS_NOISE_LEVELS,
    align_your_steps_sigmas,
    gits_sigmas,
    optimal_steps_sigmas,
)
from .native import (
    SCHEDULED_HOOKS_KEY,
    WAN_CAMERA_POSES,
    ApplyZImageControlPatch,
    BerniniConditioning,
    CLIPTextEncode,
    ConcatAVLatent,
    ControlNetApply,
    ControlNetApplyAdvanced,
    ControlNetLoader,
    EmptyLatentImage,
    EmptyLTXAVLatent,
    EmptyLTXVLatent,
    EmptyMiniMaxH3AV,
    EmptyMiniMaxMusic3LatentAudio,
    InspectLatentMask,
    KSampler,
    KSamplerAdvanced,
    LoadCheckpoint,
    LoadClip,
    LoadDiffusionModel,
    LoadDualClip,
    LoadLora,
    LoadLoraModelOnly,
    LoadVae,
    LoadVision,
    LoadZImageControlPatch,
    MiniMaxH3AddGuide,
    MiniMaxH3AudioReferenceValue,
    MiniMaxH3AVDecode,
    MiniMaxH3AVEncode,
    MiniMaxH3FL2VAConditioning,
    MiniMaxH3ImageReferenceValue,
    MiniMaxH3MotionContext,
    MiniMaxH3REF2VAConditioning,
    MiniMaxH3T2VAConditioning,
    MiniMaxH3VideoReferenceValue,
    MiniMaxMusic3TextEncode,
    PreviewLatentAudio,
    PreviewLatentVisual,
    ScheduledHookKeyframes,
    ScheduledHooks,
    ScheduledLoraHook,
    SeparateAVLatent,
    SetLatentMaskFromFrames,
    SetLatentMaskFromTimeRanges,
    VAEDecode,
    VAEDecodeAudio,
    VAEDecodeAudioTiled,
    VAEEncode,
    Wan21ClipVisionEncode,
    Wan21ImageToVideo,
    Wan22FunControlToVideo,
    Wan22ImageToVideoLatent,
    WanCameraEmbedding,
    WanCameraImageToVideo,
    WanFirstLastFrameToVideo,
    WanFunControlToVideo,
    WanFunInpaintToVideo,
    WanMoveConcatTrack,
    WanMoveGenerateTracks,
    WanMoveTracksFromCoords,
    WanMoveTrackToVideo,
    WanMoveVisualizeTracks,
    WanPhantomSubjectToVideo,
    WanTrackToVideo,
    WanVaceToVideo,
    validate_lora_execution_mode,
)
from .native_catalog import NATIVE_SCHEDULING_NODE_TYPES
from .native_residency import (
    NativeComponentHandle,
    NativeResidencyBusyError,
    NativeResidencyPool,
    NativeRuntimeHandle,
    ResidencyRouteFacts,
    current_native_observer,
    default_dependency_residency_pool,
    default_native_residency,
    native_execution_span,
    select_current_device,
    select_load_device,
)
from .pool import default_pool
from .preview_emit import (
    multistream_sampling_preview_emitter,
    preview_stage,
    sampling_preview_emitter,
)
from .type_ids import comfy_type_id

if TYPE_CHECKING:
    from dinkster_inference import ContextWindowsSpec, SamplingSegment

log = logging.getLogger("dinkster.native.native_arm")
_AIMDO_HEADROOM_TARGET_ENV = "DINKSTER_AIMDO_HEADROOM_TARGET"
_EMBEDDING_FREE_RUNTIME_SOURCE_ROLE_SETS = (
    ("checkpoint", "gemma3_12b"),
    ("checkpoint", "t5xxl"),
    ("clip_vision", "diffusion", "t5xxl", "vae"),
    ("diffusion", "qwen3_4b", "vae"),
    ("diffusion", "t5xxl", "vae"),
)
_RUNTIME_SOURCE_ROLE_SETS = (
    ("checkpoint",),
    ("checkpoint", "vae"),
    ("clip_l", "diffusion", "t5xxl", "vae"),
    *_EMBEDDING_FREE_RUNTIME_SOURCE_ROLE_SETS,
)


def _not_cancelled() -> bool:
    return False


_HOOKS = TypeExpr.concrete(comfy_type_id("HOOKS"))
_HOOK_KEYFRAMES = TypeExpr.concrete(comfy_type_id("HOOK_KEYFRAMES"))
_TIMESTEPS_RANGE = TypeExpr.concrete(comfy_type_id("TIMESTEPS_RANGE"))
_CONDITIONING = TypeExpr.concrete(comfy_type_id("CONDITIONING"))
_MASK = TypeExpr.concrete(comfy_type_id("MASK"))
_FLOAT = TypeExpr.concrete("core.float")
_COMBO = TypeExpr.concrete("core.combo")
_INT = TypeExpr.concrete("core.int")
_STRING = TypeExpr.concrete("core.string")
_BOOLEAN = TypeExpr.concrete("core.boolean")
_ASSET = TypeExpr.concrete("dinkster.asset")
_DINKSTER_MODEL = TypeExpr.concrete("dinkster.model")
_DINKSTER_CLIP = TypeExpr.concrete("dinkster.clip")
_DINKSTER_CLIP_VISION = TypeExpr.concrete("dinkster.clip-vision")
_DINKSTER_VAE = TypeExpr.concrete("dinkster.vae")
_DINKSTER_CONDITIONING = TypeExpr.concrete("dinkster.conditioning")
_DINKSTER_LATENT = TypeExpr.concrete("dinkster.latent")
_DINKSTER_LATENT_LIST = TypeExpr.list_of(_DINKSTER_LATENT)
_DINKSTER_IMAGE = TypeExpr.concrete("dinkster.image")
_DINKSTER_MASK = TypeExpr.concrete("dinkster.mask")
_DINKSTER_SAMPLER = TypeExpr.concrete("dinkster.sampler")
_DINKSTER_SIGMAS = TypeExpr.concrete("dinkster.sigmas")
_DINKSTER_GUIDER = TypeExpr.concrete("dinkster.guider")
_DINKSTER_NOISE = TypeExpr.concrete("dinkster.noise")
_DINKSTER_LATENT_OPERATION = TypeExpr.concrete("dinkster.latent-operation")
_COMFY_AUDIO = TypeExpr.concrete("comfy.AUDIO")
_COMFY_LATENT_UPSCALE_MODEL = TypeExpr.concrete("comfy.LATENT_UPSCALE_MODEL")
_COMFY_MODEL_PATCH = TypeExpr.concrete("comfy.MODEL_PATCH")
_COMFY_VOXEL = TypeExpr.concrete("comfy.VOXEL")
_COMFY_MESH = TypeExpr.concrete("comfy.MESH")
_COMFY_SHAPE_SUBDIVIDES = TypeExpr.concrete("comfy.SHAPE_SUBDIVIDES")
_MODEL3D_PROVIDER_SCHEMAS = {
    node.schema().node_type: node.schema() for node in MODEL3D_GENERATION_NODES
}
_NATIVE_PROMPT_KEY = "dinkster.native/prompt"
_NATIVE_PREPARED_CONDITIONING_KEY = "dinkster.native/prepared-conditioning"
_NATIVE_HOOKS_KEY = SCHEDULED_HOOKS_KEY
_NATIVE_MASK_KEY = "dinkster.native/mask"
_NATIVE_MASK_BOUNDS_KEY = "dinkster.native/mask-bounds"
_NativeHookKeyframes = ScheduledHookKeyframes
_NativeLoraHook = ScheduledLoraHook
_NativeHooks = ScheduledHooks
_CustomNoiseValue = NoiseSelection
_CustomSigmasValue = SigmaSchedule
_DISABLE_CFG1_OPTIMIZATION = object()


@dataclass(frozen=True, slots=True)
class _CustomSamplerValue:
    descriptor: object
    options: tuple[tuple[str, object], ...] = ()
    extension_snapshot_digest: str | None = None
    extension_ids: tuple[str, ...] = ()


def sampler_wire_value(value: object) -> object:
    if type(value) is _CustomSamplerValue:
        return SamplerSelection(
            cast("Any", value.descriptor).id,
            cast("Any", value.options),
            value.extension_snapshot_digest,
            value.extension_ids,
        )
    return value


@dataclass(frozen=True, slots=True)
class _CustomGuiderValue:
    model: object
    positive: object
    negative: object | None
    cfg: float
    transforms: tuple[tuple[str, object], ...] = ()
    batching: object | None = None

    @property
    def _dinkster_resident_refs(self) -> tuple[object, ...]:
        return (self.model, *_control_resident_refs(self.positive, self.negative))


@dataclass(frozen=True, slots=True)
class _DualCFGGuiderValue:
    model: object
    cond1: object
    cond2: object
    negative: object
    cfg_conds: float
    cfg_cond2_negative: float
    nested: bool
    batching: object | None = None

    @property
    def _dinkster_resident_refs(self) -> tuple[object, ...]:
        return (
            self.model,
            *_control_resident_refs(self.cond1, self.cond2, self.negative),
        )


@dataclass(frozen=True, slots=True)
class _DualModelGuiderValue:
    model: object
    model_negative: object | None
    positive: object
    negative: object | None
    cfg: float
    batching: object | None = None

    @property
    def _dinkster_resident_refs(self) -> tuple[object, ...]:
        models = (self.model,) if self.model_negative is None else (self.model, self.model_negative)
        return (*models, *_control_resident_refs(self.positive, self.negative))


@dataclass(frozen=True, slots=True)
class _LTXAVDualGuiderValue:
    model: object
    positive: object
    negative: object
    video_cfg: float
    audio_cfg: float
    batching: object | None = None

    @property
    def _dinkster_resident_refs(self) -> tuple[object, ...]:
        return (self.model, *_control_resident_refs(self.positive, self.negative))


@dataclass(frozen=True, slots=True)
class _PerpNegGuiderValue:
    model: object
    positive: object
    negative: object
    empty: object
    cfg: float
    neg_scale: float
    batching: object | None = None

    @property
    def _dinkster_resident_refs(self) -> tuple[object, ...]:
        return (self.model, *_control_resident_refs(self.positive, self.negative, self.empty))


@dataclass(frozen=True, slots=True)
class _LatentOperationValue:
    """One latent-samples transform (comfy_extras/nodes_latent.py
    LatentOperation* @ b78cec87), carried as parameters so the math
    runs wherever the operation is applied."""

    kind: str
    params: tuple[tuple[str, float | int], ...]


_FLUX_GUIDANCE_METADATA_KEY = "dinkster-compat-comfy/flux-guidance"
_FLUX_GUIDANCE_DISABLED = "disabled"
_FLUX2_REFERENCE_LATENTS_KEY = "dinkster-model-flux2/reference-latents"
_CONTROLNET_TEXT_METADATA_KEY = "dinkster-compat-comfy/cross-attn-controlnet"
_CONTROLNET_POOLED_METADATA_KEY = "dinkster-compat-comfy/pooled-output-controlnet"
_LTX_FRAME_RATE_METADATA_KEY = "dinkster-compat-comfy/ltx-frame-rate"


@dataclass(frozen=True, slots=True)
class _NativeClipOptions:
    source: object
    hidden_layer: int | None = None
    t5_min_padding: int | None = None
    t5_min_length: int | None = None

    @property
    def _dinkster_resident_refs(self) -> tuple[object, ...]:
        return (self.source,)


def _native_clip_options(value: object) -> _NativeClipOptions:
    return value if type(value) is _NativeClipOptions else _NativeClipOptions(value)


@dataclass(frozen=True, slots=True)
class _GuidedRows:
    """In-process pairing of materialized conditioning rows with a FluxGuidance
    strength.

    Never crosses the node wire: GenerationKSampler materializes provider
    conditioning into rows before delegating to NativeKSampler, and rows carry
    no extension metadata, so the guidance rides alongside them here until
    ``_split_flux_guidance`` separates the pair again."""

    rows: object
    guidance: float | str


def _with_flux_guidance(carrier: Any, guidance: float | None) -> Any:
    """Stamp a FluxGuidance strength on every record; the last stamp wins.

    The stamp rides ``extension_metadata`` inside the canonical carrier so
    it survives the conditioning wire between workers; samplers recover and
    strip it through ``_split_flux_guidance``."""
    inference = importlib.import_module("dinkster_inference")
    if not carrier.conditioning.records:
        raise ValueError("cannot stamp FluxGuidance on empty conditioning")
    records: list[Any] = []
    for record in carrier.conditioning.records:
        metadata = dict(record.extension_metadata)
        metadata[_FLUX_GUIDANCE_METADATA_KEY] = guidance
        records.append(replace(record, extension_metadata=tuple(metadata.items())))
    return inference.make_conditioning_carrier(
        inference.ConditioningSet(tuple(records)), carrier.bindings
    )


def _split_flux_guidance(value: object) -> tuple[object, float | str | None]:
    if type(value) is _GuidedRows:
        return value.rows, value.guidance
    inference = importlib.import_module("dinkster_inference")
    if type(value) is not inference.ConditioningCarrier:
        return value, None
    carrier = cast("Any", value)
    records = carrier.conditioning.records
    missing = object()
    stamps = [
        dict(record.extension_metadata).get(_FLUX_GUIDANCE_METADATA_KEY, missing)
        for record in records
    ]
    if all(stamp is missing for stamp in stamps):
        return value, None
    if any(stamp is missing for stamp in stamps) or len(set(stamps)) != 1:
        raise ValueError("conditioning records disagree on their FluxGuidance strength")
    stamp = stamps[0]
    if stamp is None:
        guidance: float | str = _FLUX_GUIDANCE_DISABLED
    elif type(stamp) is float:
        guidance = stamp
    else:
        raise ValueError("FluxGuidance conditioning metadata must be a float or null")
    stripped: list[Any] = []
    for record in records:
        metadata = dict(record.extension_metadata)
        del metadata[_FLUX_GUIDANCE_METADATA_KEY]
        stripped.append(replace(record, extension_metadata=tuple(metadata.items())))
    return (
        inference.make_conditioning_carrier(
            inference.ConditioningSet(tuple(stripped)), carrier.bindings
        ),
        guidance,
    )


def _with_ltx_frame_rate(
    value: object,
    inference: Any,
    frame_rate: float,
    *,
    family_id: str,
    family_name: str,
) -> object:
    if type(value) is not inference.ConditioningCarrier:
        return value
    carrier = cast("Any", value)
    records = carrier.conditioning.records
    if not records:
        raise TypeError(f"conditioning must contain {family_name} text conditioning")
    updated: list[Any] = []
    for record in records:
        layout = record.token_layout
        if layout is None or layout.family_id != family_id:
            raise TypeError(f"conditioning must contain {family_name} text conditioning")
        metadata = dict(record.extension_metadata)
        metadata[_LTX_FRAME_RATE_METADATA_KEY] = frame_rate
        updated.append(replace(record, extension_metadata=tuple(metadata.items())))
    return inference.make_conditioning_carrier(
        inference.ConditioningSet(tuple(updated)), carrier.bindings
    )


def _split_ltx_frame_rate(value: object, inference: Any) -> tuple[object, float | None]:
    if type(value) is not inference.ConditioningCarrier:
        return value, None
    carrier = cast("Any", value)
    records = carrier.conditioning.records
    stamps = [
        dict(record.extension_metadata).get(_LTX_FRAME_RATE_METADATA_KEY) for record in records
    ]
    if all(stamp is None for stamp in stamps):
        return value, None
    if any(stamp is None for stamp in stamps) or any(stamp != stamps[0] for stamp in stamps[1:]):
        raise ValueError("conditioning records disagree on their LTX frame rate")
    frame_rate = stamps[0]
    if type(frame_rate) is not float or not math.isfinite(frame_rate) or frame_rate <= 0.0:
        raise ValueError("LTX frame-rate conditioning must be a positive finite float")
    stripped: list[Any] = []
    for record in records:
        metadata = dict(record.extension_metadata)
        del metadata[_LTX_FRAME_RATE_METADATA_KEY]
        stripped.append(replace(record, extension_metadata=tuple(metadata.items())))
    return (
        inference.make_conditioning_carrier(
            inference.ConditioningSet(tuple(stripped)), carrier.bindings
        ),
        frame_rate,
    )


def _component_bound_carrier(value: object, inference: Any) -> tuple[object, Any]:
    """Split conditioning into (carrier, component binding or None).

    Non-carrier values carry no binding; the caller decides whether a
    missing binding is an error."""
    if type(value) is not inference.ConditioningCarrier:
        return value, None
    return inference.split_component_conditioning(value)


def _effective_flux_guidance(
    positive_guidance: float | str | None, negative_guidance: float | str | None
) -> float | str | None:
    if (
        positive_guidance is not None
        and negative_guidance is not None
        and positive_guidance != negative_guidance
    ):
        raise ValueError("positive and negative FluxGuidance strengths must match")
    return positive_guidance if positive_guidance is not None else negative_guidance


def _native_schema(
    node_type: str,
    display_name: str,
    category: str,
    inputs: tuple[InputSpec, ...],
    outputs: tuple[OutputSpec, ...],
    alias: str,
) -> NodeSchema:
    replacement = ReplacementRule(
        from_type=f"comfy.{alias}",
        cases=(
            ReplacementCase.build(
                node_type,
                inputs={spec.id: MappingSource.copy(spec.id) for spec in inputs},
                outputs={spec.id: spec.id for spec in outputs},
            ),
        ),
    )
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category=category,
        inputs=inputs,
        outputs=outputs,
        aliases=(alias,),
        replacements=(replacement,),
        dispatch_affinity="native",
    )


class NativeCreateHookLora(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _native_schema(
            NATIVE_SCHEDULING_NODE_TYPES["NativeCreateHookLora"],
            "Create Hook LoRA",
            "comfy/advanced/hooks/create",
            (
                InputSpec(
                    "lora_name",
                    _ASSET,
                    widget=AssetWidget(("application/octet-stream",), kind="model/lora"),
                ),
                InputSpec(
                    "strength_model",
                    _FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=-20.0, max=20.0, step=0.01),
                ),
                InputSpec(
                    "strength_clip",
                    _FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=-20.0, max=20.0, step=0.01),
                ),
                InputSpec("prev_hooks", _HOOKS, required=False),
            ),
            (OutputSpec("hooks", _HOOKS),),
            "CreateHookLora",
        )

    @classmethod
    def execute(
        cls,
        *,
        lora_name: object,
        strength_model: float,
        strength_clip: float,
        prev_hooks: object | None = None,
    ) -> Mapping[str, object]:
        if not isinstance(lora_name, AssetRef):
            raise TypeError("lora_name must be a LoRA AssetRef")
        if prev_hooks is not None and not isinstance(prev_hooks, _NativeHooks):
            raise TypeError("prev_hooks must come from native hook nodes")
        previous = () if prev_hooks is None else prev_hooks.loras
        hook = _NativeLoraHook(lora_name, float(strength_model), float(strength_clip))
        return cls.outputs(hooks=_NativeHooks(previous + (hook,)))


class NativeCreateHookKeyframe(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _native_schema(
            NATIVE_SCHEDULING_NODE_TYPES["NativeCreateHookKeyframe"],
            "Create Hook Keyframe",
            "comfy/advanced/hooks/scheduling",
            (
                InputSpec(
                    "strength_mult",
                    _FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=-20.0, max=20.0, step=0.01),
                ),
                InputSpec(
                    "start_percent",
                    _FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec("prev_hook_kf", _HOOK_KEYFRAMES, required=False),
            ),
            (OutputSpec("HOOK_KF", _HOOK_KEYFRAMES),),
            "CreateHookKeyframe",
        )

    @classmethod
    def execute(
        cls,
        *,
        strength_mult: float,
        start_percent: float,
        prev_hook_kf: object | None = None,
    ) -> Mapping[str, object]:
        if not 0.0 <= start_percent <= 1.0:
            raise ValueError("start_percent must be in [0, 1]")
        if prev_hook_kf is not None and not isinstance(prev_hook_kf, _NativeHookKeyframes):
            raise TypeError("prev_hook_kf must come from native keyframe nodes")
        previous = () if prev_hook_kf is None else prev_hook_kf.points
        points = tuple(sorted(previous + ((float(start_percent), float(strength_mult)),)))
        if len({percent for percent, _ in points}) != len(points):
            raise ValueError("LoRA curve points must use unique start percentages")
        return cls.outputs(HOOK_KF=_NativeHookKeyframes(points))


class NativeSetHookKeyframes(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _native_schema(
            NATIVE_SCHEDULING_NODE_TYPES["NativeSetHookKeyframes"],
            "Set Hook Keyframes",
            "comfy/advanced/hooks/scheduling",
            (
                InputSpec("hooks", _HOOKS),
                InputSpec("hook_kf", _HOOK_KEYFRAMES, required=False),
            ),
            (OutputSpec("hooks", _HOOKS),),
            "SetHookKeyframes",
        )

    @classmethod
    def execute(cls, *, hooks: object, hook_kf: object | None = None) -> Mapping[str, object]:
        if not isinstance(hooks, _NativeHooks):
            raise TypeError("hooks must come from native hook nodes")
        if hook_kf is not None and not isinstance(hook_kf, _NativeHookKeyframes):
            raise TypeError("hook_kf must come from native keyframe nodes")
        if hook_kf is None:
            return cls.outputs(hooks=hooks)
        return cls.outputs(
            hooks=_NativeHooks(tuple(replace(hook, keyframes=hook_kf) for hook in hooks.loras))
        )


class NativeConditioningTimestepsRange(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _native_schema(
            NATIVE_SCHEDULING_NODE_TYPES["NativeConditioningTimestepsRange"],
            "Timesteps Range",
            "comfy/advanced/hooks",
            (
                InputSpec(
                    "start_percent",
                    _FLOAT,
                    required=False,
                    default=0.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
                InputSpec(
                    "end_percent",
                    _FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
            ),
            (
                OutputSpec("TIMESTEPS_RANGE", _TIMESTEPS_RANGE),
                OutputSpec("BEFORE_RANGE", _TIMESTEPS_RANGE),
                OutputSpec("AFTER_RANGE", _TIMESTEPS_RANGE),
            ),
            "ConditioningTimestepsRange",
        )

    @classmethod
    def execute(cls, *, start_percent: float, end_percent: float) -> Mapping[str, object]:
        if not 0.0 <= start_percent <= end_percent <= 1.0:
            raise ValueError("conditioning range must satisfy 0 <= start <= end <= 1")
        return cls.outputs(
            TIMESTEPS_RANGE=(float(start_percent), float(end_percent)),
            BEFORE_RANGE=(0.0, float(start_percent)),
            AFTER_RANGE=(float(end_percent), 1.0),
        )


def _condition_entries(value: object, name: str) -> list[list[object]]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be conditioning data")
    result: list[list[object]] = []
    for entry in cast("Sequence[object]", value):
        if isinstance(entry, str | bytes) or not isinstance(entry, Sequence):
            raise TypeError(f"{name} entries must be [tensor, metadata]")
        pair = list(cast("Sequence[object]", entry))
        if len(pair) != 2 or not isinstance(pair[1], Mapping):
            raise TypeError(f"{name} entries must be [tensor, metadata]")
        pair[1] = dict(cast("Mapping[object, object]", pair[1]))
        result.append(pair)
    return result


def _set_native_conditioning_properties(
    value: object,
    *,
    name: str,
    strength: float,
    set_cond_area: str,
    mask: object | None,
    hooks: object | None,
    timesteps: object | None,
) -> list[list[object]]:
    if set_cond_area not in ("default", "mask bounds"):
        raise ValueError("set_cond_area must be 'default' or 'mask bounds'")
    if not 0.0 <= strength <= 10.0:
        raise ValueError("conditioning strength must be in [0, 10]")
    if hooks is not None and not isinstance(hooks, _NativeHooks):
        raise TypeError("hooks must come from native hook nodes")
    if timesteps is None:
        start, end = 0.0, 1.0
    elif isinstance(timesteps, str | bytes) or not isinstance(timesteps, Sequence):
        raise TypeError("timesteps must come from native range nodes")
    else:
        values = cast("Sequence[object]", timesteps)
        if len(values) != 2 or any(type(item) not in (int, float) for item in values):
            raise TypeError("timesteps must come from native range nodes")
        start, end = (float(item) for item in cast("Sequence[int | float]", values))
    result = _condition_entries(value, name)
    for entry in result:
        metadata = cast("dict[object, object]", entry[1])
        metadata["strength"] = float(strength)
        metadata["start_percent"] = start
        metadata["end_percent"] = end
        if hooks is not None:
            metadata[_NATIVE_HOOKS_KEY] = hooks
        if mask is not None:
            metadata[_NATIVE_MASK_KEY] = mask
            metadata[_NATIVE_MASK_BOUNDS_KEY] = set_cond_area == "mask bounds"
    return result


class NativeConditioningSetPropertiesAndCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _native_schema(
            NATIVE_SCHEDULING_NODE_TYPES["NativeConditioningSetPropertiesAndCombine"],
            "Cond Set Props Combine",
            "comfy/advanced/hooks/cond single",
            (
                InputSpec("cond", _CONDITIONING),
                InputSpec("cond_NEW", _CONDITIONING),
                InputSpec(
                    "strength",
                    _FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "set_cond_area",
                    _COMBO,
                    required=False,
                    default="default",
                    widget=ComboWidget(options=("default", "mask bounds")),
                ),
                InputSpec("mask", _MASK, required=False),
                InputSpec("hooks", _HOOKS, required=False),
                InputSpec("timesteps", _TIMESTEPS_RANGE, required=False),
            ),
            (OutputSpec("conditioning", _CONDITIONING),),
            "ConditioningSetPropertiesAndCombine",
        )

    @classmethod
    def execute(
        cls,
        *,
        cond: object,
        cond_NEW: object,
        strength: float,
        set_cond_area: str,
        mask: object | None = None,
        hooks: object | None = None,
        timesteps: object | None = None,
    ) -> Mapping[str, object]:
        combined = _condition_entries(cond, "cond") + _set_native_conditioning_properties(
            cond_NEW,
            name="cond_NEW",
            strength=strength,
            set_cond_area=set_cond_area,
            mask=mask,
            hooks=hooks,
            timesteps=timesteps,
        )
        return cls.outputs(conditioning=combined)


class NativePairConditioningSetProperties(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _native_schema(
            NATIVE_SCHEDULING_NODE_TYPES["NativePairConditioningSetProperties"],
            "Cond Pair Set Props",
            "comfy/advanced/hooks/cond pair",
            (
                InputSpec("positive_NEW", _CONDITIONING),
                InputSpec("negative_NEW", _CONDITIONING),
                InputSpec(
                    "strength",
                    _FLOAT,
                    required=False,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
                InputSpec(
                    "set_cond_area",
                    _COMBO,
                    required=False,
                    default="default",
                    widget=ComboWidget(options=("default", "mask bounds")),
                ),
                InputSpec("mask", _MASK, required=False),
                InputSpec("hooks", _HOOKS, required=False),
                InputSpec("timesteps", _TIMESTEPS_RANGE, required=False),
            ),
            (
                OutputSpec("positive", _CONDITIONING),
                OutputSpec("negative", _CONDITIONING),
            ),
            "PairConditioningSetProperties",
        )

    @classmethod
    def execute(
        cls,
        *,
        positive_NEW: object,
        negative_NEW: object,
        strength: float,
        set_cond_area: str,
        mask: object | None = None,
        hooks: object | None = None,
        timesteps: object | None = None,
    ) -> Mapping[str, object]:
        values = {
            name: _set_native_conditioning_properties(
                value,
                name=name,
                strength=strength,
                set_cond_area=set_cond_area,
                mask=mask,
                hooks=hooks,
                timesteps=timesteps,
            )
            for name, value in (
                ("positive", positive_NEW),
                ("negative", negative_NEW),
            )
        }
        return cls.outputs(**values)


def _torch() -> Any:
    """Resolve torch only in an executing body, never at module import."""
    return importlib.import_module("torch")


def _embedding_resource(
    inference_torch: Any,
    *,
    component_roles: tuple[str, ...] = ("clip_l", "clip_g", "t5xxl"),
) -> tuple[EmbeddingNameIndex | None, Mapping[str, Any] | None]:
    """Freeze one mount namespace and share its tensors across components."""
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT")
    if not snapshot:
        return None, None
    index = EmbeddingNameIndex(snapshot)
    if index.binding_digest is None:
        return index, None
    tensors_by_component: dict[tuple[str, str], Any] = {}

    def component_lookup(component: str) -> Any:
        def lookup(name: str) -> Any | None:
            cache_key = (component, name)
            if cache_key in tensors_by_component:
                return tensors_by_component[cache_key]
            ref = index.resolve(name)
            if ref is None:
                return None
            path = ref.local_path()
            inference = importlib.import_module("dinkster_inference")
            source = inference.load_safetensors_header(path)
            keys = tuple(source.keys())
            tensors = inference_torch.load_tensors(path, keys)
            selected = tensors.get(component)
            if selected is None:
                selected_rows = [
                    tensors[key]
                    for key in keys
                    if key.startswith("bundle_emb.") and key.endswith(".string_to_param.*")
                ]
                if selected_rows:
                    selected = _torch().cat(selected_rows, dim=0)
            if selected is None:
                selected_rows = [
                    tensors[key]
                    for key in keys
                    if key.startswith("bundle_emb.") and key.endswith(f".{component}")
                ]
                if selected_rows:
                    selected = _torch().cat(selected_rows, dim=0)
            if selected is None and len(keys) == 1:
                selected = tensors[keys[0]]
            if selected is None:
                detail = "contains no tensors" if not keys else "has no unambiguous component"
                raise ValueError(f"embedding {name!r} {detail} for {component}")
            if selected.ndim == 1:
                selected = selected.unsqueeze(0)
            if selected.ndim != 2:
                raise ValueError(f"embedding {name!r} tensor must have rank 1 or 2")
            tensors_by_component[cache_key] = selected
            return selected

        return lookup

    return index, {component: component_lookup(component) for component in component_roles}


def _freeze_embedding_resource(
    *,
    component_roles: tuple[str, ...] = ("clip_l", "clip_g", "t5xxl"),
) -> tuple[EmbeddingNameIndex | None, Mapping[str, Any] | None]:
    if not os.environ.get("DINKSTER_MOUNTS_SNAPSHOT"):
        return None, None
    inference_torch = importlib.import_module("dinkster_inference_torch")
    return _embedding_resource(inference_torch, component_roles=component_roles)


def _is_accelerator_oom(error: RuntimeError, *, torch: Any, device: Any) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return (
        device.type == "mps"
        and type(error) is RuntimeError
        and str(error).startswith("MPS backend out of memory")
    )


def _direct_vae_batch_size(
    *,
    input_batch: int,
    free_memory_bytes: int,
    memory_bytes_per_item: int,
) -> int:
    return max(1, min(input_batch, free_memory_bytes // max(1, memory_bytes_per_item)))


def _run_direct_vae(
    *,
    handle: Any,
    value: Any,
    direction: str,
    operation: Any,
    codec: Any,
) -> Any:
    torch = _torch()
    estimator = getattr(codec, "memory", None)
    batch_size = value.shape[0]
    if estimator is not None and batch_size > 1:
        inference = importlib.import_module("dinkster_inference")
        dtype_name = str(getattr(codec, "compute_dtype", None) or value.dtype).removeprefix(
            "torch."
        )
        dtype = {
            "float16": inference.FLOAT16,
            "bfloat16": inference.BFLOAT16,
            "float32": inference.FLOAT32,
        }[dtype_name]
        geometry = inference.TensorGeometry((1, *value.shape[1:]), dtype)
        estimate = (
            estimator.decode_bytes(geometry)
            if direction == "decode"
            else estimator.encode_bytes(geometry)
        )
        free_memory = (
            importlib.import_module("dinkster_inference_torch")
            .get_free_memory(handle.load_device)
            .free_total
        )
        batch_size = _direct_vae_batch_size(
            input_batch=value.shape[0],
            free_memory_bytes=free_memory,
            memory_bytes_per_item=estimate,
        )

    batches = (
        (value,)
        if batch_size == value.shape[0]
        else tuple(
            value[start : start + batch_size] for start in range(0, value.shape[0], batch_size)
        )
    )
    outputs: list[Any] = []
    used_tiled_fallback = False
    for batch in batches:
        direct_oom: BaseException | None = None
        output: Any = None
        try:
            output = operation(batch)
        except RuntimeError as caught:
            if not _is_accelerator_oom(caught, torch=torch, device=handle.load_device):
                raise
            direct_oom = caught.with_traceback(None)
        if direct_oom is not None:
            output = _retry_tiled_vae_after_oom(
                handle=handle,
                value=batch,
                output_dtype=torch.float32,
                direction=direction,
                oom=direct_oom,
                codec=codec,
            )
            used_tiled_fallback = True
        outputs.append(output)
    if used_tiled_fallback and len(outputs) > 1:
        outputs = [output.to("cpu") for output in outputs]
    return outputs[0] if len(outputs) == 1 else torch.cat(tuple(outputs))


def _retry_tiled_vae_after_oom(
    *,
    handle: Any,
    value: Any,
    output_dtype: Any,
    direction: str,
    oom: BaseException,
    codec: Any | None = None,
) -> Any:
    """Retry one failed direct VAE operation through its codec tiler."""
    backend = handle.load_device.type.upper()
    log.warning(
        "WARNING: %s out of memory during direct VAE %s; retrying with tiled VAE %s",
        backend,
        direction,
        direction,
    )
    inference_torch = importlib.import_module("dinkster_inference_torch")
    inference_torch.soft_empty_cache(handle.load_device)
    if codec is None:
        codec = getattr(handle.runtime, "codec", None)
    tiled = None if codec is None else getattr(codec, f"{direction}_tiled", None)
    if tiled is None:
        raise RuntimeError(f"native VAE runtime has no codec tiled {direction} path") from oom

    try:
        # ComfyUI's intermediate device is CPU unless --gpu-only, which Dinkster does not wire.
        return tiled(
            value,
            output_device="cpu",
            dtype=output_dtype,
        )
    except importlib.import_module("dinkster_inference").TilePlanError as tiling_error:
        descriptor = getattr(codec, "descriptor", None)
        if getattr(descriptor, "supports_tiling", None) is False:
            raise oom from tiling_error
        raise


def _sampler_registry(
    inference: Any, extension_snapshot_digest: str | None
) -> tuple[Any, tuple[str, ...], str | None]:
    inference_torch = importlib.import_module("dinkster_inference_torch")
    if extension_snapshot_digest is None:
        registries = _inference_registries(inference)
        return inference_torch.torch_sampler_registry(registries.samplers), (), None
    generation = inference.materialize_inference_generation(extension_snapshot_digest)
    registries = _inference_registries(inference, extension_snapshot_digest)
    return (
        inference_torch.torch_sampler_registry(registries.samplers),
        tuple(extension_id for extension_id, _ in generation.extensions),
        extension_snapshot_digest,
    )


def _inference_registries(inference: Any, generation_key: str | None = None) -> Any:
    context = current_execution_context()
    active = getattr(context, "inference_registries", None)
    context_key = getattr(context, "extension_snapshot_digest", None)
    if active is not None and generation_key in (None, context_key):
        return active
    key = generation_key or context_key
    if key is not None:
        return inference.materialize_inference_generation(key).registries
    return _builtin_inference_registries()


def _active_inference_registries() -> Any:
    return _inference_registries(importlib.import_module("dinkster_inference"))


def _load_runtime(
    path: Path | Mapping[str, Path],
    expected_identity: str,
    fp8_matmul: bool,
    extension_snapshot_digest: str | None = None,
    embedding_resource: (tuple[EmbeddingNameIndex | None, Mapping[str, Any] | None] | None) = None,
    attention_policy: Any = "auto",
    attention_route_token: Any | None = None,
    source_assets: Mapping[str, AssetRef] | None = None,
) -> Any:
    """Construct the runtime selected by the host's component dtype policy."""
    inference = importlib.import_module("dinkster_inference")
    wiring = importlib.import_module("dinkster_inference_torch.wiring")
    context = current_execution_context()
    dtype_kwargs: dict[str, object] = {}
    if context is not None and context.diffusion_dtype is not None:
        torch = _torch()
        dtype_kwargs = {
            "diffusion_dtype": _torch_dtype(torch, context.diffusion_dtype),
            "text_dtype": _torch_dtype(torch, cast("str", context.text_dtype)),
            "vae_dtype": _torch_dtype(torch, cast("str", context.vae_dtype)),
        }
    resource = _freeze_embedding_resource() if embedding_resource is None else embedding_resource
    embedding_index, embedding_lookups = resource
    embedding_kwargs: dict[str, object] = {}
    if embedding_index is not None and embedding_index.binding_digest is not None:
        assert embedding_lookups is not None
        embedding_kwargs = {
            "embedding_lookups": embedding_lookups,
            "embedding_binding_digest": embedding_index.binding_digest,
        }
    if isinstance(path, Path):
        paths = {"checkpoint": resolve_weight_source(path)}
    else:
        paths = _canonical_runtime_sources(path)
    assets = None if source_assets is None else _canonical_runtime_sources(source_assets)
    if assets is not None and assets.keys() != paths.keys():
        raise ValueError("native runtime assets and paths must have identical roles")
    sources = {
        role: inference.load_safetensors_header(
            source_path,
            **(
                {}
                if assets is None
                else {
                    "asset_digest": assets[role].digest,
                    "asset_size": assets[role].size,
                }
            ),
        )
        for role, source_path in paths.items()
    }
    sampler_registry, _, registry_token = _sampler_registry(inference, extension_snapshot_digest)
    attention_kwargs = (
        {}
        if attention_route_token is None
        else {
            "attention_policy": attention_policy,
            "attention_route_token": attention_route_token,
        }
    )
    if registry_token is None:
        return wiring.load_runtime(
            **sources,
            expected_identity=expected_identity,
            storage_dtype_follows_compute=True,
            fp8_matmul=fp8_matmul,
            **dtype_kwargs,
            **attention_kwargs,
            **embedding_kwargs,
        )
    assert extension_snapshot_digest is not None
    guidance_executor = None
    materialize = getattr(inference, "materialize_inference_generation", None)
    generation = None if materialize is None else materialize(extension_snapshot_digest)
    if generation is not None and generation.guidance_contributions:
        inference_torch = importlib.import_module("dinkster_inference_torch")
        registry = inference_torch.GuidanceRegistry(generation.guidance_contributions)
        if registry.active:
            guidance_executor = inference_torch.GuidanceExecutor(registry)
    kwargs = {
        **sources,
        "expected_identity": expected_identity,
        "storage_dtype_follows_compute": True,
        "fp8_matmul": fp8_matmul,
        **dtype_kwargs,
        **attention_kwargs,
        "sampler_registry": sampler_registry,
        "registry_token": registry_token,
        "extension_behavior_hash": extension_snapshot_digest.removeprefix("sha256:"),
        **embedding_kwargs,
    }
    if guidance_executor is not None:
        kwargs["guidance_executor"] = guidance_executor
    return wiring.load_runtime(**kwargs)


def _torch_dtype(torch: Any, name: str) -> Any:
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[name]
    except KeyError:
        raise RuntimeError(f"host selected unsupported runtime dtype {name!r}") from None


def _weight_storage_dtype(torch: Any, name: str) -> Any | None:
    return {
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e4m3fn_fast": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }.get(name)


def _lora_patch_weight_dtype(device: object) -> object:
    inference_torch = importlib.import_module("dinkster_inference_torch")
    return inference_torch.lora_compute_dtype(device)


def _require_fp8_matmul_support(torch: Any, device: Any) -> bool:
    inference_torch = importlib.import_module("dinkster_inference_torch")
    probe_error: Exception | None = None
    try:
        supported = bool(inference_torch.supports_fp8_matmul(device))
    except Exception as exc:  # noqa: BLE001 - normalize capability failures
        supported = False
        probe_error = exc
    if supported:
        return True
    capability = "unavailable"
    if getattr(device, "type", None) == "cuda":
        try:
            properties = torch.cuda.get_device_properties(device)
            capability = f"SM {properties.major}.{properties.minor}"
        except Exception:  # noqa: BLE001 - refusal still names unavailable capability
            pass
    message = (
        "fp8-matmul-unsupported: "
        f"device={device}; capability={capability}; "
        f"torch={getattr(torch, '__version__', 'unknown')}"
    )
    if probe_error is not None:
        message += f"; probeError={type(probe_error).__name__}: {probe_error}"
    log.warning("%s; falling back to the default matmul path", message)
    return False


def _disable_runtime_fp8_matmul(runtime: Any) -> None:
    assembled = runtime.assembled
    declared = cast("object", getattr(assembled, "components", None))
    candidates: Sequence[Any] = tuple(
        cast("Mapping[object, Any]", declared).values()
        if isinstance(declared, Mapping)
        else cast("Mapping[str, Any]", vars(assembled)).values()
    )
    components: dict[int, Any] = {
        id(module): module for module in candidates if callable(getattr(module, "modules", None))
    }
    for component in components.values():
        for module in component.modules():
            bind = getattr(module, "bind_fp8_matmul", None)
            if callable(bind):
                bind(False)


def _warn_aimdo_fallback(mode: str, gate: str, detail: str) -> str:
    """Warn about an aimdo gate failure and return it as a route fact reason."""
    if mode == "on":
        log.warning(
            "explicit --aimdo=on requested dynamic VRAM residency, but the %s "
            "gate failed: %s; falling back to eager residency",
            gate,
            detail,
        )
    else:
        log.warning(
            "automatic aimdo dynamic VRAM residency disabled by the %s gate: "
            "%s; falling back to eager residency",
            gate,
            detail,
        )
    return f"{gate} gate failed: {detail}"


def _aimdo_mode() -> str:
    mode = os.environ.get("DINKSTER_AIMDO_ARM", "auto")
    if mode not in ("auto", "on", "off"):
        raise ValueError("DINKSTER_AIMDO_ARM must be 'auto', 'on', or 'off'")
    return mode


def _upstream_default_platform(torch: Any) -> tuple[bool, str]:
    """Mirror ComfyUI main.py's default ModelPatcherDynamic gate.

    NVIDIA CUDA is admitted outside WSL. AMD ROCm requires a parsed runtime
    version of at least 7.14.
    """
    version = getattr(torch, "version", None)
    release = platform.uname().release
    if release.endswith("-Microsoft") or release.endswith("microsoft-standard-WSL2"):
        return False, f"WSL release {release!r} is excluded upstream"
    hip_version = getattr(version, "hip", None)
    if hip_version is not None:
        try:
            parts = str(hip_version).split(".")
            if len(parts) < 2:
                raise ValueError("missing minor version")
            rocm_version = tuple(map(int, parts[:2]))
        except (TypeError, ValueError):
            return (
                False,
                f"ROCm runtime {hip_version!r} is malformed; ROCm 7.14 or later is required",
            )
        if rocm_version >= (7, 14):
            return True, ""
        return False, f"ROCm runtime {hip_version!r} is below required version 7.14"
    if getattr(version, "cuda", None):
        return True, ""
    return False, "no NVIDIA CUDA or AMD ROCm runtime was reported; ROCm 7.14 or later is required"


def _apply_pending_aimdo_headroom() -> None:
    """Publish a headroom frame after activation succeeds."""
    raw = os.environ.get(_AIMDO_HEADROOM_TARGET_ENV)
    if raw is None:
        return
    try:
        target = int(raw)
        if target < 0:
            raise ValueError("negative target")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        applied = inference_torch.set_simple_vram_headroom(target)
    except Exception:  # noqa: BLE001 - headroom control cannot fail a graph
        log.warning("pending aimdo runtime headroom setter raised", exc_info=True)
        return
    if applied is True:
        os.environ.pop(_AIMDO_HEADROOM_TARGET_ENV, None)


def _aimdo_mechanism_factory(mode: str, device: Any, torch: Any) -> tuple[Any | None, str | None]:
    """Resolve AimdoWeights, or (None, reason) when a gate keeps residency eager."""
    if mode not in ("auto", "on"):
        return None, None
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        return None, _warn_aimdo_fallback(mode, "CUDA availability", repr(exc))
    if not cuda_available or device.type != "cuda":
        return None, _warn_aimdo_fallback(
            mode,
            "CUDA device",
            f"cuda_available={cuda_available}, load_device={device}",
        )
    if mode == "auto":
        try:
            supported, detail = _upstream_default_platform(torch)
        except Exception as exc:
            return None, _warn_aimdo_fallback(mode, "upstream platform", repr(exc))
        if not supported:
            return None, _warn_aimdo_fallback(mode, "upstream platform", detail)
    try:
        importlib.import_module("numpy")
    except Exception as exc:
        return None, _warn_aimdo_fallback(mode, "NumPy", repr(exc))
    try:
        aimdo = importlib.import_module("dinkster_inference_torch.aimdo_residency")
        mechanism_factory = aimdo.AimdoWeights
        ready = aimdo.ensure_visible_aimdo_devices(native_memory_policy())
    except AcceleratorMemoryPolicyError:
        raise
    except Exception as exc:
        return None, _warn_aimdo_fallback(mode, "aimdo device activation", repr(exc))
    if ready is not True:
        return None, _warn_aimdo_fallback(
            mode,
            "aimdo device activation",
            f"ensure_visible_aimdo_devices returned {ready!r} for load_device={device}",
        )
    _apply_pending_aimdo_headroom()
    return mechanism_factory, None


class _AimdoComponentFactory:
    """Keep one handle's selected Aimdo route unless its policy requires resident weights."""

    def __init__(
        self,
        mode: str,
        mechanism_factory: Any,
        eager_factory: Any,
        component_by_module: Mapping[int, str],
        resident_components: frozenset[str],
        *,
        fixed_promotion_components: frozenset[str] = frozenset(),
    ) -> None:
        self._mode = mode
        self._mechanism_factory = mechanism_factory
        self._eager_factory = eager_factory
        self._component_by_module = component_by_module
        self._resident_components = resident_components
        self._fixed_promotion_components = fixed_promotion_components
        self._dynamic_components: list[str] = []
        self._resident_served: list[str] = []

    def route_facts(self) -> ResidencyRouteFacts:
        return ResidencyRouteFacts(
            requested=self._mode,
            mechanism="aimdo" if self._dynamic_components else "eager",
            dynamic_components=tuple(self._dynamic_components),
            resident_components=tuple(self._resident_served),
        )

    def __call__(self, weights: Any, **kwargs: Any) -> Any:
        component = self._component_by_module.get(id(getattr(weights, "module", None)), "<unknown>")
        if component in self._resident_components:
            mechanism = self._eager_factory(weights, **kwargs)
            self._resident_served.append(component)
            return mechanism
        try:
            mechanism_kwargs = (
                {
                    **kwargs,
                    "fixed_promotion": True,
                    "promote_non_fp8_raw": True,
                }
                if component in self._fixed_promotion_components
                else kwargs
            )
            mechanism = self._mechanism_factory(weights, **mechanism_kwargs)
        except Exception as exc:
            exc.add_note(
                f"Aimdo construction failed for component {component!r} "
                f"(requested {self._mode}); the selected residency mechanism was not changed"
            )
            log.error(
                "Aimdo construction failed for component %r (requested %s); "
                "failing the load without changing its residency mechanism",
                component,
                self._mode,
                exc_info=True,
            )
            raise
        self._dynamic_components.append(component)
        return mechanism


def _build_runtime_handle(
    runtime: Any,
    torch: Any,
    *,
    recipe: Any,
    load_device: Any | None = None,
    patch_sets: Mapping[str, object] | None = None,
    storage_dtype: object | None = None,
    materializer: Any | None = None,
    source_resolvers: Mapping[str, object] | None = None,
) -> NativeRuntimeHandle:
    """Enroll one runtime through the production native-handle path."""
    device = select_load_device(torch) if load_device is None else torch.device(load_device)
    if recipe.knobs.fp8_matmul and not _require_fp8_matmul_support(torch, device):
        _disable_runtime_fp8_matmul(runtime)
    patch_kwargs: dict[str, Any] = (
        {}
        if not patch_sets
        else {
            "patch_weight_dtype": _lora_patch_weight_dtype(device),
            "patch_key_prefixes": {
                component: prefix for prefix, component in _lora_target_routes(runtime.assembled)
            },
        }
    )
    storage_dtypes = None if storage_dtype is None else {"diffusion": storage_dtype}
    mode = _aimdo_mode()
    mechanism_factory, fallback_reason = _aimdo_mechanism_factory(mode, device, torch)
    if mechanism_factory is None:
        eager_facts = ResidencyRouteFacts(
            requested=mode,
            mechanism="eager",
            fallback_reason=fallback_reason,
        )
        return NativeRuntimeHandle(
            runtime,
            device,
            recipe=recipe,
            patch_sets=patch_sets,
            **patch_kwargs,
            storage_dtypes=storage_dtypes,
            materializer=materializer,
            source_resolvers=source_resolvers,
            coordinator=default_native_residency(),
            residency_route_facts=lambda: eager_facts,
            _torch_module=torch,
        )
    inference_torch = importlib.import_module("dinkster_inference_torch")
    policy = getattr(runtime, "residency_policy", None)
    if policy is not None and not isinstance(policy, inference_torch.NativeResidencyPolicy):
        raise TypeError("runtime residency_policy must be a NativeResidencyPolicy or None")
    assembled = getattr(runtime, "assembled", None)
    classic_components = (
        "diffusion",
        "clip_l",
        "clip_g",
        "t5xxl",
        "umt5xxl",
        "gemma2_2b",
        "text_encoder",
        "text",
        "clip_vision",
        "vae",
    )
    policy_components = () if policy is None else policy.enrollment_components
    declared_components = getattr(assembled, "components", None)
    component_by_module = {
        id(module): component
        for component, module in (
            declared_components.items()
            if declared_components is not None
            else (
                (component, getattr(assembled, component, None))
                for component in dict.fromkeys((*classic_components, *policy_components))
            )
        )
        if module is not None
    }
    component_descriptor = _builtin_inference_registries().components.get(recipe.family_id)
    component_factory = _AimdoComponentFactory(
        mode,
        mechanism_factory,
        inference_torch.ResidentWeights,
        component_by_module,
        frozenset() if policy is None else policy.resident_components,
        fixed_promotion_components=frozenset(
            () if component_descriptor is None else component_descriptor.fixed_promotion_roles
        ),
    )
    return NativeRuntimeHandle(
        runtime,
        device,
        recipe=recipe,
        patch_sets=patch_sets,
        **patch_kwargs,
        storage_dtypes=storage_dtypes,
        materializer=materializer,
        source_resolvers=source_resolvers,
        coordinator=default_native_residency(free_memory=inference_torch.dynamic_free_memory),
        mechanism_factory=component_factory,
        residency_route_facts=component_factory.route_facts,
        _torch_module=torch,
    )


def _weight_source_ref(inference: Any, asset: AssetRef) -> Any:
    return inference.WeightSourceRef(
        digest=asset.digest,
        name=asset.name,
        size=asset.size,
        media_type=asset.media_type,
        virtual_path=asset.virtual_path,
    )


def _canonical_runtime_sources(sources: Mapping[str, Any]) -> dict[str, Any]:
    roles = tuple(sorted(sources))
    if roles not in _RUNTIME_SOURCE_ROLE_SETS:
        expected = " or ".join(str(role_set) for role_set in _RUNTIME_SOURCE_ROLE_SETS)
        raise ValueError(f"native runtime source roles must be exactly {expected}; got {roles}")
    return {role: sources[role] for role in roles}


def _runtime_recipe(
    inference: Any,
    assets: Mapping[str, AssetRef],
    sources: Mapping[str, Any],
    *,
    fp8_matmul: bool,
    extension_snapshot_digest: str | None,
    plan: Any | None = None,
    embedding_binding_digest: str | None = None,
    attention_policy: Any = "auto",
    attention_route_token: Any | None = None,
) -> Any:
    canonical_assets = _canonical_runtime_sources(assets)
    canonical_sources = _canonical_runtime_sources(sources)
    if canonical_assets.keys() != canonical_sources.keys():
        raise ValueError("native runtime assets and headers must have identical roles")
    selected_plan = (
        inference.plan_native(**canonical_sources, fp8_matmul=fp8_matmul) if plan is None else plan
    )
    context = current_execution_context()
    selected_dtypes = (
        (
            context.diffusion_dtype,
            context.text_dtype,
            context.vae_dtype,
        )
        if context is not None and context.diffusion_dtype is not None
        else (
            inference.default_diffusion_dtype(selected_plan.family.id).name,
            inference.default_text_dtype(selected_plan.family.id).name,
            inference.default_vae_dtype(selected_plan.family.id).name,
        )
    )
    assert all(isinstance(dtype, str) for dtype in selected_dtypes)
    extension_hash = (
        None
        if extension_snapshot_digest is None
        else extension_snapshot_digest.removeprefix("sha256:")
    )
    recipe = inference.ReconstructionRecipe(
        sources=tuple(
            inference.WeightSourceBinding(role, _weight_source_ref(inference, asset))
            for role, asset in canonical_assets.items()
        ),
        family_id=selected_plan.family.id,
        component_identity=inference.runtime_component_identity(
            selected_plan.family.id, selected_plan.identity_components
        ),
        knobs=inference.RuntimeKnobs(
            diffusion_dtype=selected_dtypes[0],
            text_dtype=selected_dtypes[1],
            vae_dtype=selected_dtypes[2],
            fp8_matmul=fp8_matmul,
            registry_token=extension_snapshot_digest,
            extension_behavior_hash=extension_hash,
            embedding_binding_digest=embedding_binding_digest,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        ),
    )
    return recipe


def _source_asset(source_ref: Any, source_resolvers: Mapping[str, object]) -> AssetRef:
    resolver = source_resolvers.get(source_ref.digest)
    if resolver is None:
        resolver = resolver_from_env()
    if resolver is None:
        raise RuntimeError(
            f"no worker-local asset store can resolve reconstruction source {source_ref.digest}"
        )
    return AssetRef(
        digest=source_ref.digest,
        name=source_ref.name,
        size=source_ref.size,
        media_type=source_ref.media_type,
        virtual_path=source_ref.virtual_path,
        resolver=cast("Any", resolver),
    )


def _recipe_source(recipe: Any, role: str) -> Any:
    for binding in recipe.sources:
        if binding.role == role:
            return binding.source
    raise RuntimeError(f"native reconstruction recipe has no {role!r} source")


def _recipe_bundle_name(recipe: Any) -> str:
    roles = tuple(binding.role for binding in recipe.sources)
    if len(roles) == 1:
        primary_role = roles[0]
    elif roles in _RUNTIME_SOURCE_ROLE_SETS:
        primary_role = "checkpoint" if "checkpoint" in roles else "diffusion"
    else:
        raise RuntimeError(f"native reconstruction recipe has unsupported source roles {roles}")
    return cast("str", _recipe_source(recipe, primary_role).name)


def _materialize_patch_sets(
    inference: Any,
    inference_torch: Any,
    recipe: Any,
    source_resolvers: Mapping[str, object],
) -> dict[str, object]:
    """Materialize builtin overlays into ordered per-component PatchSets.

    ProviderPatchRef materialization remains fail-loud until the first
    installable patch-provider pack triggers production catalog activation,
    as ledgered in ROADMAP. The worker-local provider seam itself is proven
    independently without making callable provider code part of the recipe.
    """
    if not recipe.overlays:
        return {}
    descriptor = _active_inference_registries().components.get(recipe.family_id)
    model_role = "diffusion" if descriptor is None else descriptor.model_role
    grouped: dict[str, dict[str, tuple[object, ...]]] = {}
    for overlay in recipe.overlays:
        asset = _source_asset(overlay.source, source_resolvers)
        path = resolve_weight_source(asset.local_path(), logical_name=asset.name)
        source = inference.load_safetensors_header(path)
        tensors = inference_torch.load_tensors(path, source.keys())
        by_component: dict[str, dict[object, object]] = {}
        for patch in overlay.patches:
            by_component.setdefault(patch.component, {})[patch.target] = patch.decoded
        for component, decoded in by_component.items():
            strength = (
                float(overlay.strength_model)
                if component in ("diffusion", model_role)
                else float(overlay.strength_clip)
            )
            if strength == 0.0:
                continue
            materialized = inference_torch.build_patch_set(
                decoded,
                tensors,
                strength=strength,
            )
            component_entries = grouped.setdefault(component, {})
            for key in materialized.keys():
                component_entries[key] = component_entries.get(key, ()) + tuple(
                    materialized.entries(key)
                )
    stack_digest = recipe.patch_stack_digest
    assert stack_digest is not None
    return {
        component: inference.PatchSet(entries, structural_digest=stack_digest)
        for component, entries in grouped.items()
    }


def _materialize_single_recipe_handle(
    recipe: Any,
    source_resolvers: Mapping[str, object],
    torch: Any,
    embedding_resource: (tuple[EmbeddingNameIndex | None, Mapping[str, Any] | None] | None) = None,
) -> NativeRuntimeHandle:
    """Resolve recipe digests, verify its planned identity, and rebuild."""
    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    recipe_binding = getattr(getattr(recipe, "knobs", None), "embedding_binding_digest", None)
    if recipe_binding is not None:
        if embedding_resource is None:
            embedding_resource = _embedding_resource(inference_torch)
        index, _ = embedding_resource
        if index is None or index.binding_digest != recipe_binding:
            raise RuntimeError("worker textual-inversion binding does not match runtime recipe")
    source_bindings = {binding.role: binding.source for binding in recipe.sources}
    if len(source_bindings) != len(recipe.sources):
        raise RuntimeError("native reconstruction recipe has duplicate source roles")
    canonical_bindings = _canonical_runtime_sources(source_bindings)
    paths: dict[str, Path] = {}
    for role, source_ref in canonical_bindings.items():
        asset = _source_asset(source_ref, source_resolvers)
        path = asset.local_path()
        paths[role] = (
            resolve_weight_source(path, logical_name=asset.name) if role == "checkpoint" else path
        )
    # A split recipe is one authority transaction: every source must resolve
    # and pass digest verification before any header is read.
    sources: dict[str, Any] = {
        role: inference.load_safetensors_header(
            path,
            asset_digest=source_bindings[role].digest,
            asset_size=source_bindings[role].size,
        )
        for role, path in paths.items()
    }
    plan = inference.plan_native(
        **sources,
        fp8_matmul=recipe.knobs.fp8_matmul,
    )
    resource = (
        (_embedding_resource(inference_torch) if embedding_resource is None else embedding_resource)
        if uses_classic_embedding_bindings(plan)
        else (None, None)
    )
    embedding_index, embedding_lookups = resource
    worker_binding = None if embedding_index is None else embedding_index.binding_digest
    if worker_binding != recipe_binding:
        raise RuntimeError("worker textual-inversion binding does not match runtime recipe")
    component_identity = inference.runtime_component_identity(
        plan.family.id, plan.identity_components
    )
    if plan.family.id != recipe.family_id or (component_identity != recipe.component_identity):
        raise RuntimeError("resolved sources no longer match the reconstruction recipe")
    sampler_registry, _, registry_token = _sampler_registry(inference, recipe.knobs.registry_token)
    load_kwargs: dict[str, object] = {
        **sources,
        "expected_identity": recipe.runtime_identity,
        "storage_dtype_follows_compute": True,
        "fp8_matmul": recipe.knobs.fp8_matmul,
        "diffusion_dtype": _torch_dtype(torch, recipe.knobs.diffusion_dtype),
        "text_dtype": _torch_dtype(torch, recipe.knobs.text_dtype),
        "vae_dtype": _torch_dtype(torch, recipe.knobs.vae_dtype),
        "patch_overlay_digests": tuple(overlay.structural_digest for overlay in recipe.overlays),
    }
    if recipe.knobs.attention_route_token is not None:
        load_kwargs.update(
            attention_policy=recipe.knobs.attention_policy,
            attention_route_token=recipe.knobs.attention_route_token,
        )
    if worker_binding is not None:
        assert embedding_lookups is not None
        load_kwargs.update(
            embedding_lookups=embedding_lookups,
            embedding_binding_digest=worker_binding,
        )
    if registry_token is not None:
        load_kwargs.update(
            sampler_registry=sampler_registry,
            registry_token=registry_token,
            extension_behavior_hash=recipe.knobs.extension_behavior_hash,
        )
        generation = inference.materialize_inference_generation(registry_token)
        if generation.guidance_contributions:
            guidance_registry = inference_torch.GuidanceRegistry(generation.guidance_contributions)
            if guidance_registry.active:
                load_kwargs["guidance_executor"] = inference_torch.GuidanceExecutor(
                    guidance_registry
                )
    runtime = inference_torch.load_runtime(**load_kwargs)
    patch_sets = _materialize_patch_sets(inference, inference_torch, recipe, source_resolvers)

    def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
        return _materialize_recipe_handle(next_recipe, next_resolvers, torch, resource)

    return _build_runtime_handle(
        runtime,
        torch,
        recipe=recipe,
        patch_sets=patch_sets,
        materializer=materializer,
        source_resolvers=source_resolvers,
    )


@dataclass(frozen=True)
class _PreparedRecipeMaterialization:
    recipe: Any
    path: tuple[str, ...]
    sources: Mapping[str, Any]
    patch_sets: Mapping[str, object]
    load_extras: Mapping[str, object]
    materialization_key: str
    estimated_cost: int


def _recipe_nodes(recipe: Any) -> dict[tuple[str, ...], Any]:
    nodes: dict[tuple[str, ...], Any] = {}

    def walk(current: Any, path: tuple[str, ...]) -> None:
        nodes[path] = current
        for dependency in current.dependencies:
            walk(dependency.child, path + (dependency.child_id,))

    walk(recipe, ())
    return nodes


def _source_key_facts(source: Any) -> tuple[object, ...]:
    return (
        source.digest,
        source.name,
        source.size,
        source.media_type,
        source.virtual_path,
    )


def _materialization_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return "native-materialization:sha256:" + hashlib.sha256(encoded).hexdigest()


def _prevalidate_recipe_materializations(
    recipe: Any,
    source_resolvers: Mapping[str, object],
    torch: Any,
    embedding_resource: tuple[EmbeddingNameIndex | None, Mapping[str, Any] | None],
) -> tuple[Any, dict[tuple[str, ...], _PreparedRecipeMaterialization]]:
    """Resolve and validate the complete graph before residency can mutate."""
    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    dependency_plan = inference.plan_dependencies(recipe)
    recipes = _recipe_nodes(recipe)
    if set(recipes) != {node.path for node in dependency_plan.nodes}:
        raise RuntimeError("native dependency plan does not match its recipe graph")

    embedding_index, _ = embedding_resource
    worker_binding = None if embedding_index is None else embedding_index.binding_digest
    nodes_by_path = {node.path: node for node in dependency_plan.nodes}
    for path, current in recipes.items():
        node = nodes_by_path[path]
        if node.runtime_identity != current.runtime_identity:
            raise RuntimeError("native dependency plan identity does not match recipe")
        if path and node.clone_mode != "with-parent":
            raise RuntimeError("native dependency clone mode 'shared' is not supported")
        if path and node.scope in ("contribution", "invocation"):
            raise RuntimeError(f"native dependency scope {node.scope!r} is not supported")
        if (
            current.knobs.embedding_binding_digest is not None
            and current.knobs.embedding_binding_digest != worker_binding
        ):
            raise RuntimeError("worker textual-inversion binding does not match runtime recipe")
        source_bindings = {binding.role: binding.source for binding in current.sources}
        if len(source_bindings) != len(current.sources):
            raise RuntimeError("native reconstruction recipe has duplicate source roles")
        _canonical_runtime_sources(source_bindings)

    validated: dict[
        tuple[str, ...],
        tuple[
            Mapping[str, Any],
            Mapping[str, object],
            Mapping[str, object],
            int,
        ],
    ] = {}
    for path, current in recipes.items():
        source_bindings = _canonical_runtime_sources(
            {binding.role: binding.source for binding in current.sources}
        )
        paths: dict[str, Path] = {}
        estimated_cost = 0
        for role, source_ref in source_bindings.items():
            asset = _source_asset(source_ref, source_resolvers)
            source_path = asset.local_path()
            paths[role] = (
                resolve_weight_source(source_path, logical_name=asset.name)
                if role == "checkpoint"
                else source_path
            )
            estimated_cost += source_ref.size
        for overlay in current.overlays:
            overlay_asset = _source_asset(overlay.source, source_resolvers)
            overlay_path = resolve_weight_source(
                overlay_asset.local_path(), logical_name=overlay_asset.name
            )
            inference.load_safetensors_header(overlay_path)
            estimated_cost += overlay.source.size
        sources = {
            role: inference.load_safetensors_header(
                source_path,
                asset_digest=source_bindings[role].digest,
                asset_size=source_bindings[role].size,
            )
            for role, source_path in paths.items()
        }
        native_plan = inference.plan_native(
            **sources,
            fp8_matmul=current.knobs.fp8_matmul,
        )
        binding = worker_binding if uses_classic_embedding_bindings(native_plan) else None
        if binding != current.knobs.embedding_binding_digest:
            raise RuntimeError("worker textual-inversion binding does not match runtime recipe")
        component_identity = inference.runtime_component_identity(
            native_plan.family.id, native_plan.identity_components
        )
        if native_plan.family.id != current.family_id or (
            component_identity != current.component_identity
        ):
            raise RuntimeError("resolved sources no longer match the reconstruction recipe")
        for dtype in (
            current.knobs.diffusion_dtype,
            current.knobs.text_dtype,
            current.knobs.vae_dtype,
        ):
            _torch_dtype(torch, dtype)
        patch_sets = _materialize_patch_sets(inference, inference_torch, current, source_resolvers)
        component_plans = {
            component.component: component
            for component in native_plan.identity_components
            if component is not None
        }
        unknown_components = set(patch_sets) - component_plans.keys()
        if unknown_components:
            raise RuntimeError(
                "native dependency patches name unknown components: "
                + ", ".join(sorted(unknown_components))
            )
        for component, patch_set in patch_sets.items():
            unknown_targets = set(patch_set.keys()) - component_plans[component].keys.keys()
            if unknown_targets:
                raise RuntimeError(
                    f"native dependency patches name unknown {component!r} targets: "
                    + ", ".join(sorted(unknown_targets))
                )
        load_extras: dict[str, object] = {}
        sampler_registry, _, registry_token = _sampler_registry(
            inference, current.knobs.registry_token
        )
        if registry_token is not None:
            load_extras.update(
                sampler_registry=sampler_registry,
                registry_token=registry_token,
                extension_behavior_hash=current.knobs.extension_behavior_hash,
            )
            generation = inference.materialize_inference_generation(registry_token)
            if generation.guidance_contributions:
                guidance_registry = inference_torch.GuidanceRegistry(
                    generation.guidance_contributions
                )
                if guidance_registry.active:
                    load_extras["guidance_executor"] = inference_torch.GuidanceExecutor(
                        guidance_registry
                    )
        validated[path] = (sources, patch_sets, load_extras, estimated_cost)

    prepared: dict[tuple[str, ...], _PreparedRecipeMaterialization] = {}
    for path in dependency_plan.load_order:
        current = recipes[path]
        sources, patch_sets, prepared_load_extras, estimated_cost = validated[path]
        node = nodes_by_path[path]
        child_keys = tuple(
            (dependency.child_id, prepared[path + (dependency.child_id,)].materialization_key)
            for dependency in current.dependencies
        )
        edge_facts = None
        if path:
            edge_facts = (
                path[-1],
                node.residency_group,
                node.scope,
                node.clone_mode,
                node.accounting_owner,
            )
        knob_facts = {
            "diffusion_dtype": current.knobs.diffusion_dtype,
            "text_dtype": current.knobs.text_dtype,
            "vae_dtype": current.knobs.vae_dtype,
            "fp8_matmul": current.knobs.fp8_matmul,
            "registry_token": current.knobs.registry_token,
            "extension_behavior_hash": current.knobs.extension_behavior_hash,
            "embedding_binding_digest": current.knobs.embedding_binding_digest,
        }
        if current.knobs.attention_route_token is not None:
            protocol = importlib.import_module("dinkster_protocol")
            knob_facts.update(
                attention_policy=current.knobs.attention_policy,
                attention_route_token=protocol.attention_route_token_to_wire(
                    current.knobs.attention_route_token
                ),
            )
        payload = {
            "path": path,
            "edge": edge_facts,
            "runtime_identity": current.runtime_identity,
            "family_id": current.family_id,
            "component_identity": current.component_identity,
            "knobs": knob_facts,
            "sources": tuple(
                (binding.role, *_source_key_facts(binding.source)) for binding in current.sources
            ),
            "overlays": tuple(
                (
                    overlay.structural_digest,
                    *_source_key_facts(overlay.source),
                    overlay.dialect,
                    overlay.key_map,
                    overlay.strength_model,
                    overlay.strength_clip,
                )
                for overlay in current.overlays
            ),
            "attachments": tuple(
                (
                    attachment.name,
                    attachment.clone,
                    attachment.device,
                    attachment.rebuild_data,
                    attachment.rebuild_entry_point,
                )
                for attachment in current.attachments
            ),
            "children": child_keys,
        }
        prepared[path] = _PreparedRecipeMaterialization(
            current,
            path,
            sources,
            patch_sets,
            prepared_load_extras,
            _materialization_digest(payload),
            estimated_cost,
        )
    return dependency_plan, prepared


def _dependency_account_identities(
    dependency_plan: Any,
    prepared: Mapping[tuple[str, ...], _PreparedRecipeMaterialization],
) -> dict[tuple[str, ...], str]:
    root_key = prepared[()].materialization_key
    accounts = {(): _materialization_digest((root_key, "root", "parent"))}
    for group in dependency_plan.residency_groups:
        parent_key = prepared[group.declaring_path].materialization_key
        owner = "parent" if group.owner_path == group.declaring_path else group.owner_path[-1]
        account = _materialization_digest((parent_key, "residency-group", group.name, owner))
        for member_path in group.member_paths:
            accounts[member_path] = account
    if set(accounts) != set(prepared):
        raise RuntimeError("native dependency accounting does not cover every node")
    return accounts


def _assemble_prevalidated_handle(
    prepared: _PreparedRecipeMaterialization,
    source_resolvers: Mapping[str, object],
    torch: Any,
    embedding_resource: tuple[EmbeddingNameIndex | None, Mapping[str, Any] | None],
) -> NativeRuntimeHandle:
    inference_torch = importlib.import_module("dinkster_inference_torch")
    recipe = prepared.recipe
    embedding_index, embedding_lookups = (
        (None, None) if recipe.knobs.embedding_binding_digest is None else embedding_resource
    )
    worker_binding = None if embedding_index is None else embedding_index.binding_digest
    load_kwargs: dict[str, object] = {
        **prepared.sources,
        "expected_identity": recipe.runtime_identity,
        "storage_dtype_follows_compute": True,
        "fp8_matmul": recipe.knobs.fp8_matmul,
        "diffusion_dtype": _torch_dtype(torch, recipe.knobs.diffusion_dtype),
        "text_dtype": _torch_dtype(torch, recipe.knobs.text_dtype),
        "vae_dtype": _torch_dtype(torch, recipe.knobs.vae_dtype),
        "patch_overlay_digests": tuple(overlay.structural_digest for overlay in recipe.overlays),
    }
    if recipe.knobs.attention_route_token is not None:
        load_kwargs.update(
            attention_policy=recipe.knobs.attention_policy,
            attention_route_token=recipe.knobs.attention_route_token,
        )
    if worker_binding is not None:
        assert embedding_lookups is not None
        load_kwargs.update(
            embedding_lookups=embedding_lookups,
            embedding_binding_digest=worker_binding,
        )
    load_kwargs.update(prepared.load_extras)
    runtime = inference_torch.load_runtime(**load_kwargs)

    def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
        return _materialize_recipe_handle(next_recipe, next_resolvers, torch, embedding_resource)

    return _build_runtime_handle(
        runtime,
        torch,
        recipe=recipe,
        patch_sets=prepared.patch_sets,
        materializer=materializer,
        source_resolvers=source_resolvers,
    )


def _materialize_recipe_handle(
    recipe: Any,
    source_resolvers: Mapping[str, object],
    torch: Any,
    embedding_resource: (tuple[EmbeddingNameIndex | None, Mapping[str, Any] | None] | None) = None,
) -> NativeRuntimeHandle:
    if not getattr(recipe, "dependencies", ()):
        return _materialize_single_recipe_handle(
            recipe, source_resolvers, torch, embedding_resource
        )
    inference_torch = importlib.import_module("dinkster_inference_torch")
    resource = (
        _embedding_resource(inference_torch) if embedding_resource is None else embedding_resource
    )
    dependency_plan, prepared = _prevalidate_recipe_materializations(
        recipe, source_resolvers, torch, resource
    )
    accounts = _dependency_account_identities(dependency_plan, prepared)
    pool: NativeResidencyPool = default_dependency_residency_pool()

    def assemble_paths(
        paths: tuple[tuple[str, ...], ...],
    ) -> dict[tuple[str, ...], NativeRuntimeHandle]:
        built: dict[tuple[str, ...], NativeRuntimeHandle] = {}
        try:
            for path in paths:
                node = prepared[path]
                token = pool.acquire(
                    node.materialization_key,
                    accounts[path],
                    node.estimated_cost,
                )
                try:
                    handle = _assemble_prevalidated_handle(node, source_resolvers, torch, resource)
                except BaseException:
                    pool.release(token)
                    raise
                handle.bind_residency_token(pool, token)
                built[path] = handle
        except BaseException as exc:
            for path in reversed(tuple(built)):
                try:
                    built[path].terminal_release()
                except BaseException as cleanup:
                    exc.add_note(
                        f"dependency construction rollback for {path!r} also failed: {cleanup!r}"
                    )
            raise
        return built

    initial = dependency_plan.select_active(frozenset()).load_order
    built = assemble_paths(initial)
    root = built.pop(())

    def materialize_conditional(
        active_conditional: frozenset[tuple[str, ...]],
    ) -> Mapping[tuple[str, ...], NativeRuntimeHandle]:
        selected = dependency_plan.select_active(active_conditional)
        existing = root.owned_dependencies
        missing = tuple(path for path in selected.load_order if path and path not in existing)
        return assemble_paths(missing)

    try:
        root.bind_dependencies(dependency_plan, built, materialize_conditional)
    except BaseException as exc:
        for handle in (root, *reversed(tuple(built.values()))):
            if handle.released:
                continue
            try:
                handle.terminal_release()
            except BaseException as cleanup:
                exc.add_note(f"dependency binding rollback also failed: {cleanup!r}")
        raise
    return root


def _logical_lora_key_map(
    inference: Any,
    handle: NativeRuntimeHandle,
    text_handle: NativeComponentHandle | None = None,
) -> dict[str, object]:
    assembled = handle.runtime.assembled
    diffusion_keys = tuple(f"diffusion_model.{key}" for key in assembled.diffusion.state_dict())
    key_map: dict[str, object] = dict(inference.native_unet_key_map(diffusion_keys))
    config = getattr(assembled.diffusion, "config", None)
    if isinstance(config, inference.UNetConfig):
        key_map.update(inference.sd_unet_diffusers_key_map(diffusion_keys, config))
    hidden_size = getattr(config, "hidden_size", None)
    if isinstance(hidden_size, int) and hidden_size > 0:
        key_map.update(inference.flux_linear1_qkv_key_map(diffusion_keys, hidden_size))
    if handle.recipe.family_id == inference.Z_IMAGE_CONFIG.family_id:
        hidden_width = getattr(config, "hidden_width", None)
        if not isinstance(hidden_width, int) or hidden_width <= 0:
            raise RuntimeError("native Z-Image runtime has no valid hidden width")
        key_map.update(inference.z_image_diffusers_key_map(diffusion_keys, hidden_width))
    clip_keys: list[str] = []
    for component, logical_component in (
        ("clip_l", "clip_l"),
        ("clip_g", "clip_g"),
        ("t5xxl", "t5xxl"),
        ("umt5xxl", "t5xxl"),
    ):
        module = getattr(assembled, component, None)
        if module is None:
            continue
        clip_keys.extend(f"{logical_component}.transformer.{key}" for key in module.state_dict())
    if text_handle is not None:
        text_recipe = text_handle.recipe
        if text_recipe is None or len(text_recipe.sources) != 1:
            raise TypeError("split Flux2 text handle has no exact retained component recipe")
        text_role = text_recipe.sources[0].role
        clip_keys.extend(
            f"{text_role}.transformer.model.{key}"
            for key in cast("Any", text_handle.module).state_dict()
        )
    key_map.update(inference.clip_lora_key_map(clip_keys))
    return key_map


def _lora_target_routes(assembled: object) -> tuple[tuple[str, str], ...]:
    routes = [
        (prefix, component)
        for prefix, component in (
            ("diffusion_model.", "diffusion"),
            ("clip_l.transformer.", "clip_l"),
            ("clip_g.transformer.", "clip_g"),
        )
        if getattr(assembled, component, None) is not None
    ]
    if getattr(assembled, "umt5xxl", None) is not None:
        routes.append(("t5xxl.transformer.", "umt5xxl"))
    elif getattr(assembled, "t5xxl", None) is not None:
        routes.append(("t5xxl.transformer.", "t5xxl"))
    declared: Mapping[str, object] | None = getattr(assembled, "components", None)
    if declared is not None:
        names = {id(module): name for name, module in declared.items()}
        return tuple(
            (prefix, names[id(getattr(assembled, component))]) for prefix, component in routes
        )
    return tuple(routes)


def _route_patch_target(
    inference: Any,
    handle: NativeRuntimeHandle,
    target: Any,
    text_role: str | None = None,
) -> tuple[str, Any]:
    for prefix, component in _lora_target_routes(handle.runtime.assembled):
        if target.key.startswith(prefix):
            return component, inference.PatchTarget(
                target.key.removeprefix(prefix), offset=target.offset
            )
    if text_role is not None:
        prefix = f"{text_role}.transformer.model."
        if target.key.startswith(prefix):
            return text_role, inference.PatchTarget(
                target.key.removeprefix(prefix), offset=target.offset
            )
    raise RuntimeError(f"decoded LoRA target {target.key!r} has no native component route")


def _native_lora_overlay(
    handle: NativeRuntimeHandle,
    lora: AssetRef,
    strength_model: float,
    strength_clip: float,
    *,
    text_handle: NativeComponentHandle | None = None,
) -> Any:
    for name, value in (
        ("strength_model", strength_model),
        ("strength_clip", strength_clip),
    ):
        if not -100.0 <= value <= 100.0:
            raise ValueError(f"{name} must be in [-100.0, 100.0], got {value}")
    inference = importlib.import_module("dinkster_inference")
    lora_path = resolve_weight_source(lora.local_path(), logical_name=lora.name)
    source = inference.load_safetensors_header(lora_path)
    geometries = {key: source.entry(key).geometry for key in source.keys()}
    key_map = _logical_lora_key_map(inference, handle, text_handle)
    decoded = inference.decode_lora(geometries, key_map)
    for diagnostic in decoded.diagnostics:
        log.warning("native LoRA %s: %s", lora.digest, diagnostic)
    if decoded.unmatched:
        log.warning(
            "native LoRA %s has %d unmatched keys: %s",
            lora.digest,
            len(decoded.unmatched),
            ", ".join(decoded.unmatched),
        )
    patches: list[object] = []
    text_role = (
        None
        if text_handle is None or text_handle.recipe is None
        else text_handle.recipe.sources[0].role
    )
    for target, patch in decoded.patches.items():
        component, routed = _route_patch_target(inference, handle, target, text_role)
        patches.append(inference.OverlayPatch(component, routed, patch))
    if not patches:
        raise RuntimeError(f"native LoRA {lora.digest} has no patch keys matching this model")
    return inference.PatchOverlay.from_decoded(
        source=_weight_source_ref(inference, lora),
        dialect=decoded.dialect,
        key_map=f"native.{handle.recipe.family_id}.v1",
        strength_model=strength_model,
        strength_clip=strength_clip,
        patches=tuple(patches),
    )


def _overlay_for_components(inference: Any, overlay: Any, components: set[str]) -> Any | None:
    patches = tuple(patch for patch in overlay.patches if patch.component in components)
    if not patches:
        return None
    return inference.PatchOverlay.from_decoded(
        source=overlay.source,
        dialect=overlay.dialect,
        key_map=overlay.key_map,
        strength_model=float(overlay.strength_model),
        strength_clip=float(overlay.strength_clip),
        patches=patches,
    )


def _native_handle(value: object, input_id: str) -> NativeRuntimeHandle:
    if isinstance(value, _NativeModelOverlay):
        value = value.handle
    if not isinstance(value, NativeRuntimeHandle):
        raise TypeError(
            f"{input_id} must be a native-arm runtime handle, got {type(value).__name__}"
        )
    value.require_active()
    return value


@dataclass(frozen=True)
class _ZImageControlBinding:
    handle: NativeComponentHandle
    image: object
    strength: float


@dataclass(frozen=True, eq=False)
class _NativeControlNetResource:
    handle: NativeComponentHandle | None
    asset_digest: str
    resource_digest: str
    source_layout: str
    descriptor: Any = None
    plan: Any = None
    hint_channels: int = 3
    mode: Any = None

    @property
    def _dinkster_resident_owner(self) -> object:
        return self if self.handle is None else self.handle


@dataclass(frozen=True, slots=True)
class _ControlHintSnapshot:
    shape: tuple[int, int, int, int]
    data: bytes
    digest: str


@dataclass(frozen=True, slots=True)
class _ClassicControlEntry:
    child_id: str
    resident_id: str
    model_digest: str
    hint: _ControlHintSnapshot


@dataclass(frozen=True, slots=True)
class _ClassicControlBinding:
    application: Any
    entries: tuple[_ClassicControlEntry, ...]
    apply_to_uncond: bool


@dataclass(frozen=True, eq=False)
class _ControlledConditioning:
    conditioning: Any
    binding: _ClassicControlBinding
    resources: tuple[_NativeControlNetResource, ...]

    @property
    def _dinkster_resident_refs(self) -> tuple[object, ...]:
        return tuple(
            resource if resource.handle is None else resource.handle for resource in self.resources
        )

    @property
    def _dinkster_resident_fingerprint(self) -> str:
        from dinkster_inference import encode_conditioning_carrier, encode_control_application
        from dinkster_values import stable_hash

        return stable_hash(
            [
                encode_conditioning_carrier(self.conditioning),
                encode_control_application(self.binding.application),
                str(self.binding.apply_to_uncond).encode("ascii"),
                *(resource.resource_digest.encode("ascii") for resource in self.resources),
            ]
        )


def _controlled_conditioning(value: object) -> _ControlledConditioning | None:
    inference = importlib.import_module("dinkster_inference")
    if isinstance(value, inference.ResidentConditioningCarrier):
        payload = cast("Any", value).payload
        if isinstance(payload, _ControlledConditioning):
            return payload
    return None


def _control_resident_refs(*values: object) -> tuple[object, ...]:
    return tuple(
        resource
        for value in values
        if (controlled := _controlled_conditioning(value)) is not None
        for resource in controlled.resources
    )


@dataclass(frozen=True, eq=False)
class _NativeModelOverlay:
    """Diffusion overlays over one shared resident runtime."""

    handle: NativeRuntimeHandle
    overlays: tuple[Any, ...]
    source_resolvers: Mapping[str, object]
    z_image_control: _ZImageControlBinding | None = None
    sampling_shift: float | None = None
    guidance_transforms: tuple[tuple[str, Any], ...] = ()
    context_windows: ContextWindowsSpec | None = None
    chroma_radiance_options: tuple[Any, ...] = ()
    sampling_cache: Any | None = None
    sampling_timeline: Any | None = None
    sampling_space: Any | None = None

    def __post_init__(self) -> None:
        if self.sampling_space is not None and self.sampling_shift is not None:
            raise ValueError("sampling space and sampling shift are mutually exclusive")
        if self.z_image_control is not None:
            self.z_image_control.handle.register_dependent(self)

    @property
    def _dinkster_resident_owner(self) -> NativeRuntimeHandle:
        return self.handle

    @property
    def _dinkster_application_identity(self) -> str:
        identity = self.handle.recipe.runtime_identity
        if any(
            contribution is _DISABLE_CFG1_OPTIMIZATION
            for _, contribution in self.guidance_transforms
        ):
            inference = importlib.import_module("dinkster_inference")
            return cast(
                "str",
                inference.extend_runtime_identity(
                    identity,
                    ("guidance.disable_cfg1_optimization=true",),
                ),
            )
        return identity

    @property
    def runtime(self) -> Any:
        return self.handle.runtime

    @property
    def recipe(self) -> Any:
        return self.handle.recipe

    @property
    def load_device(self) -> object:
        return self.handle.load_device

    def require_active(self) -> None:
        self.handle.require_active()

    def stage(self, role: str) -> Any:
        return self.handle.stage(role)


class _NativeCodecHandle:
    """Codec-only view of one compat-owned native runtime."""

    def __init__(self, handle: NativeRuntimeHandle) -> None:
        inference = importlib.import_module("dinkster_inference")
        inference.require_inference_runtime_handle(handle, "handle")
        codec = getattr(handle.runtime, "codec", None)
        descriptor = getattr(codec, "descriptor", None)
        if not isinstance(descriptor, inference.CodecDescriptor):
            raise TypeError("native runtime does not expose a CodecDescriptor")
        self._handle = handle
        self._descriptor = descriptor
        self._resource_identity = handle.recipe.runtime_identity

    @property
    def _dinkster_resident_owner(self) -> NativeRuntimeHandle:
        return self._handle

    @property
    def descriptor(self) -> Any:
        return self._descriptor

    @property
    def resource_identity(self) -> str:
        return self._resource_identity

    @property
    def load_device(self) -> object:
        return self._handle.load_device

    @property
    def accepts_batched_video(self) -> bool:
        return bool(getattr(self._handle.runtime.codec, "accepts_batched_video", False))

    @property
    def accepts_image_batch_latent(self) -> bool:
        return bool(getattr(self._handle.runtime.codec, "accepts_image_batch_latent", False))

    @property
    def manages_input_device(self) -> bool:
        return bool(getattr(self._handle.runtime.codec, "manages_input_device", False))

    def require_active(self) -> None:
        self._handle.require_active()

    def stage(self) -> Any:
        return self._handle.stage("vae")

    def _direct(self, value: Any, direction: str) -> Any:
        codec = self._handle.runtime.codec
        operation = (
            self._handle.runtime.decode_latent
            if direction == "decode"
            else self._handle.runtime.encode_content
        )
        return _run_direct_vae(
            handle=self._handle,
            value=value,
            direction=direction,
            operation=operation,
            codec=codec,
        )

    def decode_latent(self, latent: Any) -> Any:
        return self._direct(latent, "decode")

    def decode_latent_tiled(
        self,
        latent: Any,
        *,
        tile: tuple[int, ...],
        overlap: tuple[int, ...],
    ) -> Any:
        return self._handle.runtime.codec.decode_tiled(latent, tile=tile, overlap=overlap)

    def encode_content(self, content: Any) -> Any:
        return self._direct(content, "encode")

    def encode_content_tiled(
        self,
        content: Any,
        *,
        tile: tuple[int, ...],
        overlap: tuple[int, ...],
    ) -> Any:
        return self._handle.runtime.codec.encode_tiled(content, tile=tile, overlap=overlap)


class _PixelSpaceCodecHandle:
    """Stateless pixel codec published through the standard codec seam."""

    def __init__(self, resource_identity: str, compute_dtype: str) -> None:
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        torch = _torch()
        self._codec = inference_torch.PixelSpaceCodec(
            compute_dtype=_torch_dtype(torch, compute_dtype)
        )
        self._descriptor = inference.CodecDescriptor(
            id="dinkster.chroma_radiance_pixel_space",
            display_name="Chroma Radiance Pixel Space",
            kind="image",
            latent=inference.CHROMA_RADIANCE.single_stream_latent(),
            supported_dtypes=inference.CHROMA_RADIANCE.supported_dtypes,
            supports_tiling=False,
        )
        self._resource_identity = resource_identity
        self._load_device = torch.device("cpu")

    @property
    def descriptor(self) -> Any:
        return self._descriptor

    @property
    def resource_identity(self) -> str:
        return self._resource_identity

    @property
    def load_device(self) -> object:
        return self._load_device

    def require_active(self) -> None:
        return None

    @contextmanager
    def stage(self) -> Any:
        yield

    def decode_latent(self, latent: Any) -> Any:
        return self._codec.decode(latent)

    def encode_content(self, content: Any) -> Any:
        return self._codec.encode(content)


def _native_model(
    value: object, input_id: str
) -> tuple[
    NativeRuntimeHandle,
    tuple[Any, ...],
    Mapping[str, object],
    _ZImageControlBinding | None,
    float | None,
    tuple[tuple[str, Any], ...],
    ContextWindowsSpec | None,
    tuple[Any, ...],
]:
    if isinstance(value, _NativeModelOverlay):
        value.handle.require_active()
        return (
            value.handle,
            value.overlays,
            value.source_resolvers,
            value.z_image_control,
            value.sampling_shift,
            value.guidance_transforms,
            value.context_windows,
            value.chroma_radiance_options,
        )
    return _native_handle(value, input_id), (), {}, None, None, (), None, ()


def _native_model_sampling_cache(value: object) -> object | None:
    return value.sampling_cache if isinstance(value, _NativeModelOverlay) else None


def _native_model_sampling_timeline(value: object) -> object | None:
    return value.sampling_timeline if isinstance(value, _NativeModelOverlay) else None


def _native_model_sampling_space(value: object) -> Any | None:
    return value.sampling_space if isinstance(value, _NativeModelOverlay) else None


def _cfg1_optimization_setting(
    transforms: tuple[tuple[str, Any], ...],
) -> tuple[tuple[tuple[str, Any], ...], bool]:
    ordinary = tuple(
        (owner, contribution)
        for owner, contribution in transforms
        if contribution is not _DISABLE_CFG1_OPTIMIZATION
    )
    return ordinary, len(ordinary) != len(transforms)


def _sampling_space_runtime(runtime: Any, space: Any | None) -> Any:
    if space is None:
        return runtime
    inference = importlib.import_module("dinkster_inference")
    if not isinstance(runtime, inference.SamplingSpaceOverrideRuntime):
        raise TypeError("model runtime does not support a sampling-space override")
    return runtime.with_sampling_space(space)


_ExecuteP = ParamSpec("_ExecuteP")


def _bind_model_sampling_options(
    execute: Callable[_ExecuteP, Mapping[str, object]],
) -> Callable[_ExecuteP, Mapping[str, object]]:
    @wraps(execute)
    def wrapped(*args: _ExecuteP.args, **kwargs: _ExecuteP.kwargs) -> Mapping[str, object]:
        model = cast("Mapping[str, object]", kwargs).get("model")
        inference = importlib.import_module("dinkster_inference")
        if isinstance(model, inference.ApplicationChain):
            model = cast("Any", model).model
        with (
            inference.use_sampling_cache(_native_model_sampling_cache(model)),
            inference.use_sampling_timeline(_native_model_sampling_timeline(model)),
        ):
            return execute(*args, **kwargs)

    return cast("Callable[_ExecuteP, Mapping[str, object]]", wrapped)


def _context_windows_sampling_kwargs(
    runtime: Any, context_windows: ContextWindowsSpec | None, description: str
) -> dict[str, Any]:
    """Admission for KSampler context windows; the runtime revalidates its profile."""
    if context_windows is None:
        return {}
    inference = importlib.import_module("dinkster_inference")
    if not (
        isinstance(runtime, inference.ContextWindowsRuntime) and runtime.supports_context_windows
    ):
        raise ValueError(f"{description} does not support context windows")
    return {"context_windows": context_windows}


def _application_chain_model(value: object, input_id: str) -> tuple[object, tuple[Any, ...]]:
    inference = importlib.import_module("dinkster_inference")
    if not isinstance(value, inference.ApplicationChain):
        return value, ()
    chain = cast("Any", value)
    handle = inference.require_inference_runtime_handle(chain.model, input_id)
    if chain.base_model_identity != handle.recipe.runtime_identity:
        raise ValueError(f"{input_id} application chain base model identity changed")
    family_id = handle.recipe.family_id
    applications = cast("tuple[Any, ...]", chain.applications)
    for index, application in enumerate(applications):
        inference.require_inference_component_handle(
            application.handle,
            f"{input_id} application {index}",
        )
        if application.family_id != family_id:
            raise ValueError(
                f"{input_id} application {index} family_id must match the base model family"
            )
    return chain.model, applications


@contextmanager
def _staged_applications(
    applications: tuple[Any, ...],
    handle: NativeRuntimeHandle,
    *,
    stage_runtime: bool = True,
) -> Any:
    with ExitStack() as stages:
        for application in applications:
            stages.enter_context(
                application.handle.stage_with(handle, application.role)
                if stage_runtime
                else application.handle.stage()
            )
        yield


def _application_kwargs(
    applications: tuple[Any, ...],
    handle: NativeRuntimeHandle,
    latent: object,
    *,
    reserved_keys: set[str],
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    for index, application in enumerate(applications):
        component = application.handle.component
        materialized = application.materialize_application_kwargs(
            handle.runtime,
            component,
            latent,
        )
        if not isinstance(materialized, Mapping):
            raise TypeError(
                f"model application {index} materialize_application_kwargs must return a mapping"
            )
        for key, item in cast("Mapping[object, object]", materialized).items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"model application {index} kwarg names must be non-empty strings")
            if key in reserved_keys:
                raise ValueError(f"model application {index} kwarg {key!r} collides")
            if key == "sd15_attention_contributions" and key in kwargs:
                previous = kwargs[key]
                if type(previous) is not tuple or type(item) is not tuple:
                    raise TypeError("SD1.5 attention contributions must materialize as tuples")
                kwargs[key] = (*previous, *item)
                continue
            if key in kwargs:
                raise ValueError(f"model application {index} kwarg {key!r} collides")
            kwargs[key] = item
    return kwargs


@contextmanager
def _materialized_application_kwargs(
    applications: tuple[Any, ...],
    handle: NativeRuntimeHandle,
    latent: object,
    *,
    reserved_keys: set[str],
    stage_runtime: bool = True,
) -> Any:
    with _staged_applications(applications, handle, stage_runtime=stage_runtime):
        yield _application_kwargs(
            applications,
            handle,
            latent,
            reserved_keys=reserved_keys,
        )


def _overlay_targets(overlay: object, component: str) -> bool:
    patches = getattr(overlay, "patches", None)
    if patches is None:
        return True
    if component == "text":
        return any(patch.component != "diffusion" for patch in patches)
    return any(patch.component == component for patch in patches)


def _overlay_has_offsets(overlay: object) -> bool:
    patches = getattr(overlay, "patches", ())
    return any(
        getattr(getattr(patch, "target", None), "offset", None) is not None for patch in patches
    )


def _inpaint_conditioning(
    metadata: Mapping[object, object], input_id: str, torch: Any, inference: Any
) -> Any | None:
    concat_mask = metadata.get("concat_mask")
    concat_latent = metadata.get("concat_latent_image")
    if (concat_mask is None) != (concat_latent is None):
        raise ValueError(
            f"{input_id} inpaint conditioning requires both concat_mask and concat_latent_image"
        )
    if concat_mask is None:
        return None
    if not isinstance(concat_mask, torch.Tensor):
        raise TypeError(f"{input_id} concat_mask must be a torch.Tensor")
    if not isinstance(concat_latent, torch.Tensor):
        raise TypeError(f"{input_id} concat_latent_image must be a torch.Tensor")
    return inference.InpaintConditioning(mask=concat_mask, masked_image=concat_latent)


def _scheduled_inpaint(value: object, input_id: str, torch: Any, inference: Any) -> Any | None:
    resolved = tuple(
        _inpaint_conditioning(cast("Mapping[object, object]", entry[1]), input_id, torch, inference)
        for entry in _condition_entries(value, input_id)
    )
    present = tuple(item for item in resolved if item is not None)
    if not present:
        return None
    first = present[0]
    if len(present) != len(resolved) or any(
        item.mask is not first.mask or item.masked_image is not first.masked_image
        for item in present[1:]
    ):
        raise ValueError(f"{input_id} scheduled inpaint conditioning must share concat values")
    return first


def _conditioning(value: object, input_id: str, torch: Any, inference: Any) -> tuple[Any, Any]:
    """Decode text plus the narrow native SD inpaint conditioning subset."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(
            f"{input_id} conditioning must contain exactly one [tensor, metadata] entry"
        )
    entries = cast("Sequence[object]", value)
    if len(entries) != 1:
        raise ValueError(
            f"{input_id} conditioning must contain exactly one entry, got {len(entries)}"
        )
    entry = entries[0]
    if isinstance(entry, str | bytes) or not isinstance(entry, Sequence):
        raise ValueError(f"{input_id} conditioning entry must be [tensor, metadata]")
    parts = cast("Sequence[object]", entry)
    if len(parts) != 2:
        raise ValueError(
            f"{input_id} conditioning entry must have exactly two elements, got {len(parts)}"
        )
    embeddings, metadata = parts
    if type(embeddings) is inference.PreparedMultiStreamConditioning:
        if type(metadata) is not dict or metadata:
            raise ValueError(
                f"{input_id} prepared multi-stream conditioning requires empty metadata"
            )
        return embeddings, None
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(
            f"{input_id} conditioning embeddings must be a torch.Tensor, "
            f"got {type(embeddings).__name__}"
        )
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{input_id} conditioning metadata must be a mapping")
    metadata_map = cast("Mapping[object, object]", metadata)
    unsupported = set(metadata_map) - {
        "pooled_output",
        "concat_mask",
        "concat_latent_image",
        "control",
        "control_apply_to_uncond",
        _NATIVE_PROMPT_KEY,
        _NATIVE_PREPARED_CONDITIONING_KEY,
    }
    if unsupported:
        raise ValueError(
            f"{input_id} conditioning metadata is unsupported on the native arm: "
            + ", ".join(sorted(repr(key) for key in unsupported))
        )
    prepared = metadata_map.get(_NATIVE_PREPARED_CONDITIONING_KEY)
    if prepared is not None:
        if set(metadata_map) - {"control", "control_apply_to_uncond"} != {
            _NATIVE_PREPARED_CONDITIONING_KEY
        }:
            raise ValueError(f"{input_id} prepared conditioning cannot carry legacy metadata")
        if not isinstance(prepared, inference.Conditioning):
            raise TypeError(f"{input_id} prepared conditioning must be a Conditioning value")
        typed_prepared = cast("Any", prepared)
        if typed_prepared.embeddings is not embeddings:
            raise ValueError(f"{input_id} prepared conditioning embeddings are inconsistent")
        return prepared, None
    pooled = metadata_map.get("pooled_output")
    if pooled is not None and not isinstance(pooled, torch.Tensor):
        raise TypeError(
            f"{input_id} pooled_output must be a torch.Tensor when present, "
            f"got {type(pooled).__name__}"
        )
    inpaint = _inpaint_conditioning(metadata_map, input_id, torch, inference)
    return inference.Conditioning(embeddings=embeddings, pooled=pooled), inpaint


def _conditioning_classic_control(value: object, input_id: str) -> _ClassicControlBinding | None:
    if type(value) is _GuidedRows:
        value = value.rows
    controlled = _controlled_conditioning(value)
    if controlled is not None:
        return controlled.binding
    inference = importlib.import_module("dinkster_inference")
    if isinstance(value, (inference.ConditioningCarrier, inference.ResidentConditioningCarrier)):
        return None
    entries = _condition_entries(value, input_id)
    if len(entries) != 1:
        if any(
            "control" in cast("Mapping[object, object]", entry[1])
            or "control_apply_to_uncond" in cast("Mapping[object, object]", entry[1])
            for entry in entries
        ):
            raise ValueError(
                f"{input_id} classic ControlNet requires exactly one conditioning entry"
            )
        return None
    metadata = cast("Mapping[object, object]", entries[0][1])
    raw_binding = metadata.get("control")
    raw_apply_to_uncond = metadata.get("control_apply_to_uncond")
    if raw_binding is None:
        if raw_apply_to_uncond is not None:
            raise ValueError(
                f"{input_id} control_apply_to_uncond is present without a control binding"
            )
        return None
    if not isinstance(raw_binding, _ClassicControlBinding):
        raise TypeError(f"{input_id} control metadata is not a native classic ControlNet binding")
    if type(raw_apply_to_uncond) is not bool:
        raise TypeError(f"{input_id} control_apply_to_uncond must be a bool")
    if raw_apply_to_uncond is not raw_binding.apply_to_uncond:
        raise ValueError(f"{input_id} control apply-to-uncond metadata is inconsistent")
    return raw_binding


def _select_classic_control_binding(
    positive: object, negative: object
) -> _ClassicControlBinding | None:
    positive_binding = _conditioning_classic_control(positive, "positive")
    negative_binding = _conditioning_classic_control(negative, "negative")
    if positive_binding is None and negative_binding is None:
        return None
    if positive_binding is None:
        raise ValueError("negative conditioning cannot carry classic ControlNet by itself")
    if negative_binding == positive_binding:
        return positive_binding
    if negative_binding is None and positive_binding.apply_to_uncond:
        return positive_binding
    raise ValueError(
        "positive and negative classic ControlNet chains must match; use "
        "ControlNetApplyAdvanced on both conditioning inputs"
    )


def _require_classic_control_keyword(receiver: Callable[..., Any]) -> None:
    try:
        inspect.signature(receiver).bind_partial(None, control=None)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "classic ControlNet requires the selected sampler to accept the 'control' keyword"
        ) from exc


def _materialize_classic_control(
    binding: _ClassicControlBinding | None,
    *,
    latent_batch: int,
    torch: Any,
    inference: Any,
    inference_torch: Any,
    base_handle: Any = None,
    cleanup: ExitStack | None = None,
) -> tuple[Any | None, tuple[NativeComponentHandle, ...]]:
    from dinkster_inference.component_registry import execution_symbol

    if binding is None:
        return None, ()
    applications: list[Any] = []
    application = binding.application
    while application is not None:
        if type(application) is not inference.ControlApplication:
            raise TypeError("classic ControlNet application metadata is malformed")
        applications.append(application)
        application = application.previous
    applications.reverse()
    if len(applications) != len(binding.entries):
        raise ValueError("classic ControlNet application and resource chains differ in length")
    pool = default_pool()
    resolved = None
    handles: list[NativeComponentHandle] = []
    seen_handles: set[int] = set()
    materialized: dict[str, NativeComponentHandle] = {}
    for application, entry in zip(applications, binding.entries, strict=True):
        if application.child_id != entry.child_id:
            raise ValueError("classic ControlNet application and resource chains are misordered")
        prefix = "resident:"
        if not entry.resident_id.startswith(prefix) or len(entry.resident_id) == len(prefix):
            raise ValueError("classic ControlNet resident identity is malformed")
        resource = pool.get(entry.resident_id[len(prefix) :])
        if not isinstance(resource, _NativeControlNetResource):
            raise TypeError("classic ControlNet resident identity resolved to the wrong value type")
        if resource.resource_digest != entry.model_digest:
            raise ValueError("classic ControlNet assembly provenance changed after apply")
        control_handle = resource.handle or materialized.get(entry.resident_id)
        if control_handle is None:
            if base_handle is None or cleanup is None:
                raise ValueError(
                    "control component requires a sampling model and invocation lifetime"
                )
            base = base_handle.runtime.assembled
            base_source = next(
                source.source
                for source in base_handle.recipe.sources
                if source.role in ("checkpoint", "diffusion")
            )
            module = execution_symbol(resource.descriptor.loader)(
                resource.plan,
                base.compute_dtype("diffusion"),
                base=base,
                base_asset_digest=base_source.digest,
            )
            control_handle = _enroll_control_module(
                module, torch, inference_torch, discard_on_release=True
            )
            cleanup.callback(control_handle.terminal_release)
            materialized[entry.resident_id] = control_handle
        control_handle.require_active()
        shape = entry.hint.shape
        if shape[0] not in (1, latent_batch):
            raise ValueError(
                "ControlNet hint batch must be one or match the latent batch; "
                f"got hint {shape[0]} and latent {latent_batch}"
            )
        expected_bytes = math.prod(shape) * 4
        if len(entry.hint.data) != expected_bytes:
            raise ValueError("classic ControlNet hint snapshot byte length is malformed")
        hint = torch.frombuffer(bytearray(entry.hint.data), dtype=torch.float32).reshape(*shape)
        if inference_torch.sd_control_hint_digest(hint) != entry.hint.digest:
            raise ValueError("classic ControlNet hint snapshot digest changed after apply")
        conditioning_factory = (
            inference_torch.SDControlConditioning
            if resource.descriptor is None
            else execution_symbol(resource.descriptor.runtime_class)
        )
        resolved = conditioning_factory(
            application,
            control_handle.module,
            hint,
            control_handle.resource_identity,
            entry.hint.digest,
            previous=resolved,
        )
        handle_identity = id(control_handle)
        if handle_identity not in seen_handles:
            seen_handles.add(handle_identity)
            handles.append(control_handle)
    return resolved, tuple(handles)


@contextmanager
def _classic_control_context(binding: _ClassicControlBinding | None, **kwargs: Any):
    with ExitStack() as cleanup:
        yield (
            (None, ())
            if binding is None
            else _materialize_classic_control(binding, cleanup=cleanup, **kwargs)
        )


def _uses_native_scheduling(value: object) -> bool:
    for entry in _condition_entries(value, "conditioning"):
        metadata = cast("Mapping[object, object]", entry[1])
        if (
            _NATIVE_HOOKS_KEY in metadata
            or _NATIVE_MASK_KEY in metadata
            or "start_percent" in metadata
            or "end_percent" in metadata
            or metadata.get("strength", 1.0) != 1.0
        ):
            return True
    return False


def _prompt_routes(inference: Any, family: Any, text: str) -> tuple[Any, ...]:
    return tuple(
        inference.ScheduledPromptRoute(inference.EncoderStream.from_encoder_id(encoder_id), text)
        for encoder_id in family.wiring.text_encoders
    )


def _curve_segments(hooks: _NativeHooks) -> tuple[tuple[float, float, tuple[float, ...]], ...]:
    breakpoints = {0.0, 1.0}
    explicit_end = False
    for hook in hooks.loras:
        if hook.keyframes is not None:
            breakpoints.update(percent for percent, _ in hook.keyframes.points)
            explicit_end = explicit_end or any(
                percent == 1.0 for percent, _ in hook.keyframes.points
            )
    ordered = tuple(sorted(breakpoints))

    def multiplier(hook: _NativeLoraHook, percent: float) -> float:
        if hook.keyframes is None or not hook.keyframes.points:
            return 1.0
        result = hook.keyframes.points[0][1]
        for start, strength in hook.keyframes.points:
            if start > percent:
                break
            result = strength
        return result

    segments: list[tuple[float, float, tuple[float, ...]]] = []
    for index, start in enumerate(ordered):
        if index + 1 == len(ordered):
            if start == 1.0 and not explicit_end:
                continue
            end = 1.0
        else:
            boundary = ordered[index + 1]
            end = math.nextafter(boundary, 0.0) if boundary < 1.0 or explicit_end else 1.0
        segments.append((start, end, tuple(multiplier(hook, start) for hook in hooks.loras)))
    return tuple(segments)


class _StagedEncoder:
    def __init__(self, handle: NativeRuntimeHandle, encoder: object) -> None:
        self._handle = handle
        self._encoder = encoder

    def encode(self, *args: object, **kwargs: object) -> object:
        with self._handle.stage("text"):
            with _torch().no_grad():
                return cast("Any", self._encoder).encode(*args, **kwargs)


class _ScheduledTextRuntime:
    _ENCODERS = frozenset(
        {
            "_ovis_encoder",
            "_t5_encoder",
            "_clip_encoder",
            "_clip_l_encoder",
            "_clip_g_encoder",
        }
    )

    def __init__(self, handle: NativeRuntimeHandle) -> None:
        self._handle = handle

    def __getattr__(self, name: str) -> object:
        value = getattr(self._handle.runtime, name)
        if name in self._ENCODERS and value is not None:
            return _StagedEncoder(self._handle, value)
        return value

    def dispose(self) -> None:
        self._handle.terminal_release()


class _ScheduledDiffusionDeclaration:
    def dispose(self) -> None:
        return None


class _NativeScheduleState:
    def __init__(
        self,
        handle: NativeRuntimeHandle,
        inference: Any,
        inference_torch: Any,
        *,
        ordinary_overlays: tuple[Any, ...] = (),
        ordinary_resolvers: Mapping[str, object] | None = None,
    ) -> None:
        self.handle = handle
        self.inference = inference
        self.inference_torch = inference_torch
        self.executions: list[Any] = []
        self.overlays: dict[str, tuple[Any, ...]] = {}
        self.resolvers: dict[str, Any] = dict(ordinary_resolvers or {})
        self.patch_sets: dict[str, Any] = {}
        self.hooks: tuple[_NativeLoraHook, ...] = ()
        self.ordinary_overlays = ordinary_overlays
        self.declaration = inference.KeyedContribution(
            "inference.patch-providers",
            "dinkster.native.lora",
            behavior_metadata=(("contractVersion", 1),),
        )
        self.snapshot = inference_torch.PatchProviderSnapshot((self.declaration,))

    def builder(
        self,
        _base: object,
        target: object,
        overlays: tuple[Any, ...],
        cancelled: object,
    ) -> object:
        if cast("Any", cancelled)():
            raise RuntimeError("native scheduled variant construction cancelled")
        if target is self.inference.PatchTargetComponent.TEXT:
            resolvers = {
                overlay.source.digest: self.resolvers[overlay.source.digest]
                for overlay in overlays
                if overlay.source.digest in self.resolvers
            }
            return _ScheduledTextRuntime(
                self.handle.clone(cast("Any", overlays), source_resolvers=resolvers)
            )
        return _ScheduledDiffusionDeclaration()

    def register(self, overlays: tuple[Any, ...]) -> None:
        if not overlays:
            return
        digest = self.inference.patch_overlay_stack_digest(overlays)
        assert digest is not None
        existing = self.overlays.get(digest)
        if existing is not None and existing != overlays:
            raise RuntimeError("native scheduled overlay identity collision")
        self.overlays[digest] = overlays
        for overlay in overlays:
            source = overlay.source
            for hook in self.hooks:
                if hook.lora.digest == source.digest and hook.lora.resolver is not None:
                    self.resolvers[source.digest] = hook.lora.resolver

    def add_hooks(self, hooks: _NativeHooks) -> None:
        self.hooks += hooks.loras

    def resolve(self, requests: tuple[Any, ...], cancel: Any) -> tuple[Any, ...]:
        resolved: list[Any] = []
        for request in requests:
            if cancel():
                raise RuntimeError("native scheduled patch resolution cancelled")
            overlays = self.overlays.get(request.stack_digest)
            if overlays is None:
                raise RuntimeError(
                    f"native scheduled patch stack {request.stack_digest} is unknown"
                )
            patch_set = self.patch_sets.get(request.stack_digest)
            if patch_set is None:
                recipe = replace(self.handle.recipe, overlays=overlays)
                patch_sets = _materialize_patch_sets(
                    self.inference,
                    self.inference_torch,
                    recipe,
                    self.resolvers,
                )
                patch_set = patch_sets.get("diffusion")
                if patch_set is None:
                    raise RuntimeError("native scheduled LoRA stack has no diffusion patches")
                self.patch_sets[request.stack_digest] = patch_set
            resolved.append(
                self.inference_torch.ScheduledPatchResolution(
                    request,
                    self.handle.recipe.runtime_identity,
                    self.snapshot,
                    self.declaration.id,
                    self.declaration,
                    patch_set,
                )
            )
        return tuple(resolved)

    def close(self) -> None:
        error: BaseException | None = None
        for execution in self.executions:
            try:
                execution.close()
            except BaseException as caught:
                if error is None:
                    error = caught
        if error is not None:
            raise error


def _scheduled_carrier(
    value: object,
    input_id: str,
    handle: NativeRuntimeHandle,
    state: _NativeScheduleState,
) -> object:
    inference = state.inference
    carriers: list[Any] = []
    for entry_index, raw_entry in enumerate(_condition_entries(value, input_id)):
        metadata = cast("dict[object, object]", raw_entry[1])
        unsupported = set(metadata) - {
            "pooled_output",
            "start_percent",
            "end_percent",
            "strength",
            "concat_mask",
            "concat_latent_image",
            _NATIVE_PROMPT_KEY,
            _NATIVE_HOOKS_KEY,
            _NATIVE_MASK_KEY,
            _NATIVE_MASK_BOUNDS_KEY,
        }
        if unsupported:
            raise ValueError(
                f"{input_id} scheduled conditioning metadata is unsupported on the native arm: "
                + ", ".join(sorted(repr(key) for key in unsupported))
            )
        prompt_data = metadata.get(_NATIVE_PROMPT_KEY)
        if not isinstance(prompt_data, tuple):
            raise ValueError(
                f"{input_id} scheduled conditioning must originate from native CLIP Text Encode"
            )
        prompt_values = cast("tuple[object, ...]", prompt_data)
        if len(prompt_values) != 2 or any(type(item) is not str for item in prompt_values):
            raise ValueError(
                f"{input_id} scheduled conditioning must originate from native CLIP Text Encode"
            )
        text, runtime_identity = cast("tuple[str, str]", prompt_values)
        if runtime_identity != handle.recipe.runtime_identity:
            raise ValueError(f"{input_id} conditioning belongs to a different native runtime")
        start = metadata.get("start_percent", 0.0)
        end = metadata.get("end_percent", 1.0)
        strength = metadata.get("strength", 1.0)
        if any(type(item) not in (int, float) for item in (start, end, strength)):
            raise TypeError(f"{input_id} schedule and strength values must be numbers")
        numeric = cast("tuple[int | float, int | float, int | float]", (start, end, strength))
        start, end, strength = (float(item) for item in numeric)
        schedule = inference.PercentRange(start, end)
        mask = metadata.get(_NATIVE_MASK_KEY)
        mask_binding = None
        mask_descriptor = None
        if mask is not None:
            torch = _torch()
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"{input_id} mask must be a torch.Tensor")
            mask_tensor = cast("Any", mask)
            if mask_tensor.ndim < 3:
                mask_tensor = mask_tensor.unsqueeze(0)
            mask_binding = state.inference_torch.tensor_to_payload_binding(
                f"{input_id}-mask-{entry_index}",
                mask_tensor,
                space=state.inference_torch.MASK_PAYLOAD_SPACE,
            )
            mask_descriptor = inference.MaskDescriptor(
                inference.PayloadReference(mask_binding.reference_id),
                strength,
                metadata.get(_NATIVE_MASK_BOUNDS_KEY) is True,
            )
        hooks = metadata.get(_NATIVE_HOOKS_KEY)
        if hooks is not None and not isinstance(hooks, _NativeHooks):
            raise TypeError(f"{input_id} hooks did not come from native hook nodes")
        if hooks is not None:
            state.add_hooks(hooks)
            segments = _curve_segments(hooks)
        else:
            segments = ((0.0, 1.0, ()),)
        for segment_start, segment_end, multipliers in segments:
            effective = inference.intersect_ranges(
                schedule, inference.PercentRange(segment_start, segment_end)
            )
            if effective is inference.EMPTY_RANGE:
                continue
            scheduled_overlays = (
                tuple(
                    _native_lora_overlay(
                        handle,
                        hook.lora,
                        hook.strength_model * multiplier,
                        hook.strength_clip * multiplier,
                    )
                    for hook, multiplier in zip(hooks.loras, multipliers, strict=True)
                    if hook.strength_model * multiplier != 0.0
                    or hook.strength_clip * multiplier != 0.0
                )
                if hooks is not None
                else ()
            )
            overlays = state.ordinary_overlays + scheduled_overlays
            if overlays:
                state.register(overlays)
            text_stacks = (
                (inference.ScheduledPatchStack(effective, overlays),)
                if any(
                    float(overlay.strength_clip) != 0.0 and _overlay_targets(overlay, "text")
                    for overlay in overlays
                )
                else ()
            )
            diffusion_stacks = (
                (inference.ScheduledPatchStack(effective, overlays),)
                if any(
                    float(overlay.strength_model) != 0.0 and _overlay_targets(overlay, "diffusion")
                    for overlay in overlays
                )
                else ()
            )
            request = inference.ScheduledEncodeRequest(
                (
                    inference.ScheduledPrompt(
                        effective,
                        _prompt_routes(inference, handle.runtime.family, text),
                    ),
                ),
                text_patches=text_stacks,
                diffusion_patches=diffusion_stacks,
            )
            execution = inference.ScheduledExecution(inference.ScheduledVariantOwner(state.builder))
            state.executions.append(execution)
            execution_context = current_execution_context()
            carrier: Any = handle.runtime.encode_text_scheduled(
                request,
                execution=execution,
                cancelled=(
                    execution_context.cancelled if execution_context is not None else _not_cancelled
                ),
                type_registry=inference.InferenceTypeRegistry(),
            )
            if mask_descriptor is not None:
                assert mask_binding is not None
                records = tuple(
                    replace(record, mask=mask_descriptor) for record in carrier.conditioning.records
                )
                carrier = inference.make_conditioning_carrier(
                    inference.ConditioningSet(records),
                    (*carrier.bindings, mask_binding),
                )
            elif strength != 1.0:
                area = inference.AreaDescriptor(
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    inference.AreaUnits.PERCENT,
                    strength,
                )
                records = tuple(
                    replace(record, area=area) for record in carrier.conditioning.records
                )
                carrier = inference.make_conditioning_carrier(
                    inference.ConditioningSet(records), carrier.bindings
                )
            carriers.append(carrier)
    records = tuple(record for carrier in carriers for record in carrier.conditioning.records)
    bindings = tuple(binding for carrier in carriers for binding in carrier.bindings)
    return inference.make_conditioning_carrier(inference.ConditioningSet(records), bindings)


def _catalog_id(registry: Any, requested: str, kind: str) -> str:
    descriptor = registry.get(requested)
    if descriptor is None:
        raise ValueError(
            f"unknown {kind} {requested!r} (registered ids: {', '.join(registry.ids())})"
        )
    return cast("str", descriptor.id)


def _compute_dtype(runtime: Any) -> Any:
    dtype = runtime.assembled.compute_dtype("diffusion")
    if dtype is None:
        raise RuntimeError(
            f"native runtime family {runtime.family.id!r} has no assembled diffusion dtype"
        )
    return dtype


def _component_candidate_path(asset: AssetRef) -> Path:
    if asset.resolver is None:
        raise ValueError(f"component asset {asset.digest} has no resolver")
    path = asset.resolver.resolve(asset.digest)
    if path is None:
        raise ValueError(f"component asset {asset.digest} is not materializable")
    return path


def _component_descriptor(family_id: str) -> Any:
    descriptor = _active_inference_registries().components.get(family_id)
    if descriptor is None:
        raise ValueError(f"no component architecture detected for {family_id!r}")
    return descriptor


def _build_component_runtime_handle(
    descriptor: Any,
    asset: AssetRef,
    role: str,
    expected_identity: str,
    torch: Any,
    *,
    compute_dtype: str,
    as_model: bool = False,
    load_device: Any | None = None,
    required_recipe: Any | None = None,
    source_resolvers: Mapping[str, object] | None = None,
    attention_policy: Any | None = None,
    attention_route_token: Any | None = None,
    artifact_role: str | None = None,
    storage_dtype: object | None = None,
) -> Any:
    from dinkster_inference.component_registry import execution_symbol

    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    context = current_execution_context()
    if required_recipe is not None:
        attention_policy = required_recipe.knobs.attention_policy
        attention_route_token = required_recipe.knobs.attention_route_token
    elif attention_policy is None:
        attention_policy = "auto" if context is None else context.attention_policy
        attention_route_token = None if context is None else context.attention_route_token
    device = select_load_device(torch) if load_device is None else torch.device(load_device)
    load_identity = (
        expected_identity
        if required_recipe is None
        else replace(required_recipe, overlays=()).runtime_identity
    )
    load_kwargs: dict[str, Any] = {}
    attention_backend = descriptor.family.engine.attention_backend(role)
    if attention_backend is not None:
        load_kwargs.update(
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
            attention_backend=attention_backend,
        )
    if descriptor.family.engine.quantized_component_load_device:
        load_kwargs["load_device"] = device
    if artifact_role is not None:
        load_kwargs["artifact_role"] = artifact_role
    loaded = execution_symbol(descriptor.loader)(
        _component_candidate_path(asset),
        asset=asset,
        expected_role=role,
        expected_identity=load_identity,
        compute_dtype=_torch_dtype(torch, compute_dtype),
        **load_kwargs,
    )
    if descriptor.tokenizer_attribute is not None and loaded.tokenizer is not None:
        setattr(loaded.module, descriptor.tokenizer_attribute, loaded.tokenizer)
    base_recipe = descriptor.recipe(
        _weight_source_ref(inference, asset),
        loaded,
        compute_dtype,
        attention_policy=attention_policy,
        attention_route_token=attention_route_token,
    )
    label = descriptor.family.display_name
    if base_recipe.runtime_identity != load_identity:
        raise RuntimeError(f"{label} component recipe identity differs from dispatch identity")
    if required_recipe is not None and base_recipe != replace(required_recipe, overlays=()):
        raise RuntimeError(f"rebuilt {label} component recipe differs from retained recipe")
    recipe = base_recipe if required_recipe is None else required_recipe
    if recipe.runtime_identity != expected_identity:
        raise RuntimeError(f"{label} component overlay identity differs from dispatch identity")
    resolvers = dict(source_resolvers or {})
    if asset.resolver is not None:
        resolvers[asset.digest] = asset.resolver
    patch_sets = _materialize_patch_sets(inference, inference_torch, recipe, resolvers)
    if set(patch_sets) - {role}:
        raise RuntimeError(f"{label} component overlays must target only {role}")

    def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
        next_asset = _source_asset(_recipe_source(next_recipe, role), next_resolvers)
        return _build_component_runtime_handle(
            descriptor,
            next_asset,
            role,
            next_recipe.runtime_identity,
            torch,
            compute_dtype=compute_dtype,
            as_model=as_model,
            load_device=device,
            required_recipe=next_recipe,
            source_resolvers=next_resolvers,
            artifact_role=artifact_role,
            storage_dtype=storage_dtype,
        )

    if as_model:
        from dinkster_inference.component_registry import build_component_runtime

        runtime = build_component_runtime(
            descriptor, loaded, recipe.runtime_identity, _torch_dtype(torch, compute_dtype)
        )
        handle = _build_runtime_handle(
            runtime,
            torch,
            recipe=recipe,
            load_device=device,
            patch_sets=patch_sets,
            storage_dtype=storage_dtype,
            materializer=materializer,
            source_resolvers=resolvers,
        )
        if descriptor.pool_model:
            pool = default_pool()
            pool.label(handle, asset.name)
            handle.attach_pool(pool)
        return handle

    return _enroll_component_handle(
        loaded.module,
        role,
        recipe=recipe,
        device=device,
        torch=torch,
        materializer=materializer,
        source_resolvers=resolvers,
        label=asset.name,
        patch_set=patch_sets.get(role),
        aimdo_roles=descriptor.aimdo_roles,
        fixed_promotion_roles=descriptor.fixed_promotion_roles,
    )


def _enroll_component_handle(
    module: Any,
    role: str,
    *,
    recipe: Any,
    device: Any,
    torch: Any,
    materializer: Any,
    source_resolvers: Mapping[str, object],
    label: str,
    patch_set: Any = None,
    aimdo_roles: tuple[str, ...] = (),
    fixed_promotion_roles: tuple[str, ...] = (),
    runtime: object | None = None,
) -> NativeComponentHandle:
    inference_torch = importlib.import_module("dinkster_inference_torch")
    enroll_kwargs: dict[str, Any] = (
        {}
        if patch_set is None
        else {"patch_set": patch_set, "patch_weight_dtype": _lora_patch_weight_dtype(device)}
    )
    route_facts_provider: Callable[[], ResidencyRouteFacts] | None = None
    if aimdo_roles:
        mode = _aimdo_mode()
        mechanism_factory, fallback_reason = (
            _aimdo_mechanism_factory(mode, device, torch) if role in aimdo_roles else (None, None)
        )
        if mechanism_factory is None:
            route_facts = ResidencyRouteFacts(
                requested=mode,
                mechanism="eager",
                fallback_reason=fallback_reason,
                resident_components=(role,) if role not in aimdo_roles else (),
                fallback_components=(role,) if fallback_reason is not None else (),
            )

            def eager_route_facts() -> ResidencyRouteFacts:
                return route_facts

            route_facts_provider = eager_route_facts
            coordinator = default_native_residency()
        else:
            component_factory = _AimdoComponentFactory(
                mode,
                mechanism_factory,
                inference_torch.ResidentWeights,
                {id(module): role},
                frozenset(),
                fixed_promotion_components=frozenset(fixed_promotion_roles),
            )
            route_facts_provider = component_factory.route_facts
            coordinator = default_native_residency(free_memory=inference_torch.dynamic_free_memory)
            enroll_kwargs["mechanism_factory"] = component_factory
    else:
        coordinator = default_native_residency()
    mechanism = coordinator.enroll_component(
        module,
        load_device=device,
        offload_device=torch.device("cpu"),
        enroller=inference_torch.enroll_component,
        **enroll_kwargs,
    )
    handle_kwargs: dict[str, Any] = (
        {} if route_facts_provider is None else {"residency_route_facts": route_facts_provider}
    )
    if runtime is not None:
        handle_kwargs["runtime"] = runtime
    handle = NativeComponentHandle(
        module,
        mechanism,
        device,
        resource_identity=recipe.runtime_identity,
        coordinator=coordinator,
        recipe=recipe,
        materializer=materializer,
        source_resolvers=source_resolvers,
        **handle_kwargs,
    )
    pool = default_pool()
    pool.label(handle, label)
    handle.attach_pool(pool)
    return handle


def _ltxav_component_identity(
    inference: Any,
    asset: AssetRef,
    role: str,
    compute_dtype: str,
    attention_policy: Any = "auto",
    attention_route_token: Any | None = None,
) -> tuple[str, Any]:
    path = _component_candidate_path(asset)
    source = inference.load_safetensors_header(
        path,
        asset_digest=asset.digest,
        asset_size=path.stat().st_size,
    )
    planned = inference.plan_ltxav_split_component(source, role=role, path=path)
    dtype = {
        "float16": inference.FLOAT16,
        "float32": inference.FLOAT32,
        "bfloat16": inference.BFLOAT16,
    }.get(compute_dtype)
    if dtype is None:
        raise ValueError(f"unsupported LTX-2 component compute dtype {compute_dtype!r}")
    return (
        inference.ltxav_component_runtime_identity(
            planned,
            dtype,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
        ),
        planned,
    )


class _LTXAVTextHandle:
    family_id = "dinkster.ltxav"
    role = "text"

    def __init__(
        self,
        handles: tuple[NativeComponentHandle, ...],
        runtime_identity: str,
    ) -> None:
        if len(handles) not in (2, 3):
            raise ValueError("LTX-2 text composition requires Gemma and projection components")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        gemma = cast("Any", handles[0].component)
        projection = cast("Any", handles[1].component)
        connectors = None if len(handles) == 2 else cast("Any", handles[2].component)
        if type(gemma) is not inference_torch.LTXAVGemmaComponent:
            raise TypeError("LTX-2 text composition requires the Gemma component")
        self._handles = handles
        self._runtime = inference_torch.LTXAVTextRuntime(
            gemma.model,
            projection,
            gemma.tokenizer_model,
            connectors=connectors,
        )
        self.resource_identity = runtime_identity
        self.load_device = handles[0].load_device

    @property
    def _dinkster_resident_owner(self) -> NativeComponentHandle:
        return self._handles[0]

    @property
    def _dinkster_resident_refs(self) -> tuple[NativeComponentHandle, ...]:
        return self._handles[1:]

    def require_active(self) -> None:
        for handle in self._handles:
            handle.require_active()

    @contextmanager
    def stage(self):
        self.require_active()
        with ExitStack() as stages:
            for handle in self._handles:
                stages.enter_context(handle.stage())
            yield

    def encode_text(self, text: str) -> Any:
        return self._runtime.text_conditioning_carrier(self._runtime.encode_text(text))


def _build_ltxav_text_handle(
    text_encoder: AssetRef,
    checkpoint: AssetRef,
    expected_identity: str,
    torch: Any,
    *,
    compute_dtype: str,
    load_device: Any | None,
) -> _LTXAVTextHandle:
    inference = importlib.import_module("dinkster_inference")
    context = current_execution_context()
    attention_policy = "auto" if context is None else context.attention_policy
    attention_route_token = None if context is None else context.attention_route_token
    text_path = _component_candidate_path(text_encoder)
    text_source = inference.load_safetensors_header(
        text_path,
        asset_digest=text_encoder.digest,
        asset_size=text_path.stat().st_size,
    )
    text_role = inference.identify_ltxav_text_source(text_source)
    if text_role not in ("gemma3_12b", "gemma4_12b"):
        raise RuntimeError("LTX-2 text encoder asset has no supported Gemma text role")
    gemma_identity, _ = _ltxav_component_identity(
        inference,
        text_encoder,
        text_role,
        compute_dtype,
        attention_policy,
        attention_route_token,
    )
    projection_asset = checkpoint
    try:
        projection_identity, projection_plan = _ltxav_component_identity(
            inference,
            projection_asset,
            "text_projection",
            compute_dtype,
            attention_policy,
            attention_route_token,
        )
    except inference.LTXAVComponentAssemblyError:
        projection_asset = text_encoder
        projection_identity, projection_plan = _ltxav_component_identity(
            inference,
            projection_asset,
            "text_projection",
            compute_dtype,
            attention_policy,
            attention_route_token,
        )
    projection_kind = projection_plan.component.config
    if (text_role == "gemma4_12b") != (projection_kind == "dual_linear_gemma4"):
        raise RuntimeError("LTX-2 Gemma and text projection profiles differ")
    components = {
        text_role: gemma_identity,
        "text_projection": projection_identity,
    }
    connector_identity = None
    if projection_plan.component.config == "single_linear":
        connector_identity, _ = _ltxav_component_identity(
            inference,
            checkpoint,
            "connectors",
            compute_dtype,
            attention_policy,
            attention_route_token,
        )
        components["connectors"] = connector_identity
    composition = inference.compose_execution("dinkster.ltxav", components)
    if composition.execution_identity != expected_identity:
        raise RuntimeError("LTX-2 text composition differs from dispatch identity")
    handles: list[NativeComponentHandle] = []
    try:
        handles.append(
            _build_component_runtime_handle(
                _component_descriptor("dinkster.ltxav"),
                text_encoder,
                text_role,
                gemma_identity,
                torch,
                compute_dtype=compute_dtype,
                load_device=load_device,
            )
        )
        handles.append(
            _build_component_runtime_handle(
                _component_descriptor("dinkster.ltxav"),
                projection_asset,
                "text_projection",
                projection_identity,
                torch,
                compute_dtype=compute_dtype,
                load_device=load_device,
            )
        )
        if connector_identity is not None:
            handles.append(
                _build_component_runtime_handle(
                    _component_descriptor("dinkster.ltxav"),
                    checkpoint,
                    "connectors",
                    connector_identity,
                    torch,
                    compute_dtype=compute_dtype,
                    load_device=load_device,
                )
            )
    except Exception:
        for handle in reversed(handles):
            handle.terminal_release()
        raise
    return _LTXAVTextHandle(tuple(handles), expected_identity)


def _ltxav_audio_codec_recipe(inference: Any, asset: AssetRef, loaded: Any) -> Any:
    plans = loaded.plan.identity_components
    runtime_facts = tuple(sorted({fact for plan in plans for fact in plan.runtime_facts}))
    return inference.ReconstructionRecipe(
        sources=(inference.WeightSourceBinding("audio_vae", _weight_source_ref(inference, asset)),),
        family_id="dinkster.ltxav",
        component_identity=inference.runtime_component_identity("dinkster.ltxav", plans),
        knobs=inference.RuntimeKnobs(
            diffusion_dtype="unloaded",
            text_dtype="unloaded",
            vae_dtype="float32",
            fp8_matmul=False,
            runtime_facts=runtime_facts,
        ),
    )


def _build_ltxav_audio_codec_handle(
    asset: AssetRef,
    expected_identity: str,
    torch: Any,
    *,
    load_device: Any | None = None,
    required_recipe: Any | None = None,
    source_resolvers: Mapping[str, object] | None = None,
) -> NativeComponentHandle:
    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    loaded = inference_torch.load_ltxav_audio_codec(
        _component_candidate_path(asset),
        asset=asset,
        expected_identity=expected_identity,
    )
    recipe = _ltxav_audio_codec_recipe(inference, asset, loaded)
    if recipe.runtime_identity != expected_identity:
        raise RuntimeError("LTX-2 audio codec recipe identity differs from dispatch identity")
    if required_recipe is not None and recipe != required_recipe:
        raise RuntimeError("rebuilt LTX-2 audio codec recipe differs from retained recipe")
    resolvers = dict(source_resolvers or {})
    if asset.resolver is not None:
        resolvers[asset.digest] = asset.resolver
    device = select_load_device(torch) if load_device is None else torch.device(load_device)
    coordinator = default_native_residency()
    mechanism = coordinator.enroll_component(
        loaded.module,
        load_device=device,
        offload_device=torch.device("cpu"),
        enroller=inference_torch.enroll_component,
    )

    def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
        next_asset = _source_asset(_recipe_source(next_recipe, "audio_vae"), next_resolvers)
        return _build_ltxav_audio_codec_handle(
            next_asset,
            next_recipe.runtime_identity,
            torch,
            load_device=device,
            required_recipe=next_recipe,
            source_resolvers=next_resolvers,
        )

    handle = NativeComponentHandle(
        loaded.module,
        mechanism,
        device,
        resource_identity=expected_identity,
        coordinator=coordinator,
        recipe=recipe,
        materializer=materializer,
        source_resolvers=resolvers,
    )
    pool = default_pool()
    pool.label(handle, asset.name)
    handle.attach_pool(pool)
    return handle


def _component_execution_context(node_type: str) -> Any:
    context = current_execution_context()
    if context is None or context.expected_execution_identity is None:
        raise RuntimeError(f"native {node_type} ran without an expected execution identity")
    return context


def _trellis2_artifact_role_matches(asset: AssetRef, role: str) -> bool:
    inference = importlib.import_module("dinkster_inference")
    path = _component_candidate_path(asset)
    try:
        source = inference.load_safetensors_header(
            path,
            asset_digest=asset.digest,
            asset_size=asset.size,
        )
        inference.plan_trellis2_artifact(source, role=role, path=path)
    except ValueError:
        return False
    return True


def _trellis2_split_model_recipe(
    inference: Any,
    assets: Mapping[str, AssetRef],
    plan: Any,
    compute_dtype: str,
) -> Any:
    components = plan.identity_components
    runtime_facts = tuple(
        sorted({fact for component in components for fact in component.runtime_facts})
    )
    return inference.ReconstructionRecipe(
        sources=tuple(
            inference.WeightSourceBinding(role, _weight_source_ref(inference, assets[role]))
            for role in sorted(assets)
        ),
        family_id="dinkster.trellis2",
        component_identity=inference.runtime_component_identity("dinkster.trellis2", components),
        knobs=inference.RuntimeKnobs(
            diffusion_dtype=compute_dtype,
            text_dtype="unloaded",
            vae_dtype="unloaded",
            fp8_matmul=False,
            runtime_facts=runtime_facts,
        ),
    )


def _build_trellis2_split_model_handle(
    assets: Mapping[str, AssetRef],
    expected_identity: str,
    torch: Any,
    *,
    compute_dtype: str,
    required_recipe: Any | None = None,
    source_resolvers: Mapping[str, object] | None = None,
    storage_dtype: object | None = None,
) -> NativeRuntimeHandle:
    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    role_order = ("shape", "shape-512", "structure", "texture", "texture-512")
    if tuple(sorted(assets)) != role_order:
        raise ValueError(f"TRELLIS.2 split flow roles must be exactly {role_order}")
    loaded = {
        role: inference_torch.load_trellis2_flow_artifact(
            _component_candidate_path(assets[role]),
            asset=assets[role],
            expected_role=role,
            compute_dtype=_torch_dtype(torch, compute_dtype),
        )
        for role in role_order
    }
    plan = inference.Trellis2SplitModelPlan(
        structure=loaded["structure"].plan,
        shape=loaded["shape"].plan,
        shape_512=loaded["shape-512"].plan,
        texture=loaded["texture"].plan,
        texture_512=loaded["texture-512"].plan,
    )
    recipe = _trellis2_split_model_recipe(inference, assets, plan, compute_dtype)
    if recipe.runtime_identity != expected_identity:
        raise RuntimeError("TRELLIS.2 split model recipe identity differs from dispatch identity")
    if required_recipe is not None and recipe != required_recipe:
        raise RuntimeError("rebuilt TRELLIS.2 split model recipe differs from retained recipe")
    resolvers = dict(source_resolvers or {})
    for asset in assets.values():
        if asset.resolver is not None:
            resolvers[asset.digest] = asset.resolver
    assembled = inference_torch.AssembledTrellis2(
        inference_torch.Trellis2FlowBundle(
            structure=loaded["structure"].module,
            shape=loaded["shape"].module,
            shape_512=loaded["shape-512"].module,
            texture=loaded["texture"].module,
            texture_512=loaded["texture-512"].module,
        ),
        plan,
    )
    runtime = inference_torch.Trellis2DiffusionRuntime(
        assembled,
        runtime_identity=expected_identity,
        compute_dtype=_torch_dtype(torch, compute_dtype),
    )

    def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
        next_assets = {
            role: _source_asset(_recipe_source(next_recipe, role), next_resolvers)
            for role in role_order
        }
        return _build_trellis2_split_model_handle(
            next_assets,
            next_recipe.runtime_identity,
            torch,
            compute_dtype=compute_dtype,
            required_recipe=next_recipe,
            source_resolvers=next_resolvers,
            storage_dtype=storage_dtype,
        )

    return _build_runtime_handle(
        runtime,
        torch,
        recipe=recipe,
        storage_dtype=storage_dtype,
        materializer=materializer,
        source_resolvers=resolvers,
    )


def _load_detected_component(
    asset: AssetRef,
    kind: str,
    context: Any,
    *,
    load_device: Any | None = None,
    detected: Any | None = None,
    storage_dtype: object | None = None,
) -> Any:
    inference = importlib.import_module("dinkster_inference")
    registry = _active_inference_registries().components
    family_id = context.expected_execution_identity.partition(":")[2].partition(":")[0]
    if detected is None:
        path = _component_candidate_path(asset)
        source = inference.load_safetensors_header(
            path, asset_digest=asset.digest, asset_size=asset.size
        )
        descriptor, role, _plan = registry.select(source, path, kind, family_id=family_id)
    else:
        descriptor, role, _plan = registry.select_detected(detected, kind, family_id=family_id)
    dtype = {
        "model": context.diffusion_dtype,
        "text": context.text_dtype,
        "codec": context.vae_dtype,
    }[kind]
    if dtype is None:
        dtype = inference.default_vae_dtype(descriptor.id).name if kind == "codec" else "bfloat16"
    storage_kwargs: dict[str, Any] = (
        {} if storage_dtype is None else {"storage_dtype": storage_dtype}
    )
    return _build_component_runtime_handle(
        descriptor,
        asset,
        role,
        context.expected_execution_identity,
        _torch(),
        compute_dtype=dtype,
        as_model=kind == "model",
        load_device=load_device,
        attention_policy=context.attention_policy,
        attention_route_token=context.attention_route_token,
        **storage_kwargs,
    )


def build_text_recipe_handle(
    assets: tuple[AssetRef, ...],
    requested_type: str,
    expected_identity: str,
    *,
    compute_dtype: str,
    load_device: Any | None = None,
    attention_policy: Any = "auto",
    attention_route_token: Any | None = None,
    required_recipe: Any | None = None,
    source_resolvers: Mapping[str, object] | None = None,
    embedding_resource: Any | None = None,
) -> NativeComponentHandle:
    """Verify ordered sources and reconstruct their explicit text encoding recipe."""
    from dinkster_inference import PatchSet
    from dinkster_inference.component_registry import execution_symbol
    from dinkster_inference.sources import SafetensorsSource, load_safetensors_header_from_file
    from dinkster_inference.text_recipes import resolve_text_recipe

    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    torch = _torch()
    device = select_load_device(torch) if load_device is None else torch.device(load_device)
    if required_recipe is not None:
        attention_policy = required_recipe.knobs.attention_policy
        attention_route_token = required_recipe.knobs.attention_route_token
    with ExitStack() as opened:
        files = tuple(opened.enter_context(asset.open()) for asset in assets)
        sources: list[SafetensorsSource] = []
        for asset, file in zip(assets, files, strict=True):
            if os.fstat(file.fileno()).st_size != asset.size:
                raise ValueError("text component byte size differs from its asset metadata")
            source = load_safetensors_header_from_file(
                file,
                path=_component_candidate_path(asset),
                asset_digest=asset.digest,
                asset_size=asset.size,
            )
            sources.append(replace(source, configuration_file=file))
        registry = _active_inference_registries().components
        detected = tuple(registry.detect(source, source.path) for source in sources)
        binding = resolve_text_recipe(detected, requested_type)
        if any(part.profile is not None for part in binding.components):
            resource = (
                _freeze_embedding_resource(
                    component_roles=tuple(part.role for part in binding.components)
                )
                if embedding_resource is None
                else embedding_resource
            )
        else:
            resource = (None, None)
        embedding_index, embedding_lookups = resource
        base_recipe = binding.recipe(
            tuple(_weight_source_ref(inference, asset) for asset in assets),
            compute_dtype,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
            embedding_binding_digest=(
                None if embedding_index is None else embedding_index.binding_digest
            ),
        )
        if required_recipe is not None and base_recipe != replace(required_recipe, overlays=()):
            raise RuntimeError("rebuilt text encoding recipe differs from retained recipe")
        recipe = base_recipe if required_recipe is None else required_recipe
        if recipe.runtime_identity != expected_identity:
            raise RuntimeError("text encoding recipe identity differs from dispatch identity")
        load_kwargs: dict[str, object] = {}
        descriptor = registry.get(binding.family_id)
        if descriptor is not None and descriptor.family.engine.attention_backends:
            load_kwargs["attention_backends"] = descriptor.family.engine.attention_backends
        loaded = execution_symbol(binding.loader)(
            binding,
            compute_dtype=_torch_dtype(torch, compute_dtype),
            sources=tuple(sources),
            source_files=files,
            attention_policy=attention_policy,
            attention_route_token=attention_route_token,
            **load_kwargs,
        )
    runtime = execution_symbol(binding.runtime_class)(loaded, embedding_lookups=embedding_lookups)
    resolvers = dict(source_resolvers or {})
    resolvers.update(
        {asset.digest: asset.resolver for asset in assets if asset.resolver is not None}
    )
    patch_sets = cast(
        "dict[str, PatchSet[Any]]",
        _materialize_patch_sets(inference, inference_torch, recipe, resolvers),
    )
    if set(patch_sets) - set(loaded.module):
        raise RuntimeError("text overlays target components absent from the encoding recipe")
    patch_set = (
        inference.PatchSet(
            {
                f"{role}.{key}": tuple(patches.entries(key))
                for role, patches in patch_sets.items()
                for key in patches.keys()
            },
            structural_digest=recipe.patch_stack_digest,
        )
        if patch_sets
        else None
    )

    def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
        return build_text_recipe_handle(
            tuple(_source_asset(item.source, next_resolvers) for item in next_recipe.sources),
            binding.id,
            next_recipe.runtime_identity,
            compute_dtype=compute_dtype,
            load_device=device,
            required_recipe=next_recipe,
            source_resolvers=next_resolvers,
            embedding_resource=resource,
        )

    return _enroll_component_handle(
        loaded.module,
        "text",
        recipe=recipe,
        device=device,
        torch=torch,
        materializer=materializer,
        source_resolvers=resolvers,
        label=" + ".join(asset.name for asset in assets),
        patch_set=patch_set,
        runtime=runtime,
    )


class NativeLoadClip(LoadClip):
    """Load one official text encoder as an independent component."""

    @classmethod
    def execute(cls, *, text_encoder: object, type: str, device: str) -> Mapping[str, object]:
        if not isinstance(text_encoder, AssetRef):
            raise TypeError("text_encoder must be an AssetRef")
        from dinkster_inference.sources import load_safetensors_header
        from dinkster_inference.text_recipes import UnresolvedTextRecipe, resolve_text_recipe

        context = _component_execution_context("load_clip")
        torch = _torch()
        load_device = torch.device("cpu") if device == "cpu" else None
        with native_execution_span("load", "load"):
            path = _component_candidate_path(text_encoder)
            source = load_safetensors_header(
                path, asset_digest=text_encoder.digest, asset_size=text_encoder.size
            )
            detected = _active_inference_registries().components.detect(source, path)
            try:
                resolve_text_recipe((detected,), type)
            except UnresolvedTextRecipe:
                fixed = tuple(
                    match for match in detected if not match.descriptor.requires_text_recipe
                )
                if detected and not fixed:
                    raise
                handle = _load_detected_component(
                    text_encoder, "text", context, load_device=load_device, detected=fixed
                )
            else:
                handle = build_text_recipe_handle(
                    (text_encoder,),
                    type,
                    context.expected_execution_identity,
                    compute_dtype=context.text_dtype or "float32",
                    load_device=load_device,
                    attention_policy=context.attention_policy,
                    attention_route_token=context.attention_route_token,
                )
        return cls.outputs(clip=handle)


class NativeLoadDualClip(LoadDualClip):
    """Load two official text encoders as one ordered native recipe."""

    @classmethod
    def execute(
        cls,
        *,
        text_encoder1: object,
        text_encoder2: object,
        type: str,
        device: str,
    ) -> Mapping[str, object]:
        if not isinstance(text_encoder1, AssetRef):
            raise TypeError("text_encoder1 must be an AssetRef")
        if not isinstance(text_encoder2, AssetRef):
            raise TypeError("text_encoder2 must be an AssetRef")
        context = _component_execution_context("load_dual_clip")
        torch = _torch()
        load_device = torch.device("cpu") if device == "cpu" else None
        with native_execution_span("load", "load"):
            handle = build_text_recipe_handle(
                (text_encoder1, text_encoder2),
                type,
                context.expected_execution_identity,
                compute_dtype=context.text_dtype or "float32",
                load_device=load_device,
                attention_policy=context.attention_policy,
                attention_route_token=context.attention_route_token,
            )
        return cls.outputs(clip=handle)


class NativeLoadVae(LoadVae):
    """Load one official codec as an independent component."""

    @classmethod
    def execute(
        cls, *, vae: object | None = None, pixel_space: bool = False
    ) -> Mapping[str, object]:
        if type(pixel_space) is not bool:
            raise TypeError("pixel_space must be a boolean")
        context = _component_execution_context("load_vae")
        if pixel_space:
            if vae is not None:
                raise ValueError("pixel_space and vae are mutually exclusive")
            inference = importlib.import_module("dinkster_inference")
            compute_dtype = (
                context.vae_dtype or inference.default_vae_dtype(inference.CHROMA_RADIANCE.id).name
            )
            return cls.outputs(
                vae=_PixelSpaceCodecHandle(context.expected_execution_identity, compute_dtype)
            )
        if not isinstance(vae, AssetRef):
            raise TypeError("vae must be an AssetRef")
        with native_execution_span("load", "load"):
            handle = _load_detected_component(vae, "codec", context)
        return cls.outputs(vae=handle)


class NativeLoadVision(LoadVision):
    """Load one admitted vision component under native residency."""

    @classmethod
    def execute(cls, *, vision_encoder: object) -> Mapping[str, object]:
        if not isinstance(vision_encoder, AssetRef):
            raise TypeError("vision_encoder must be an AssetRef")
        context = _component_execution_context("load_vision")
        if not _trellis2_artifact_role_matches(vision_encoder, "vision"):
            raise ValueError("native vision loading requires an admitted TRELLIS.2 vision asset")
        with native_execution_span("load", "load"):
            handle = _build_component_runtime_handle(
                _component_descriptor("dinkster.trellis2"),
                vision_encoder,
                "vision",
                context.expected_execution_identity,
                _torch(),
                compute_dtype=context.text_dtype or "bfloat16",
            )
        return cls.outputs(vision=handle)


class NativeLoadDiffusionModel(LoadDiffusionModel):
    """Load one admitted diffusion component."""

    @classmethod
    def execute(cls, *, diffusion_model: object, weight_dtype: str) -> Mapping[str, object]:
        if not isinstance(diffusion_model, AssetRef):
            raise TypeError("diffusion_model must be an AssetRef")
        storage_dtype = (
            None if weight_dtype == "default" else _weight_storage_dtype(_torch(), weight_dtype)
        )
        context = current_execution_context()
        if context is None or context.expected_execution_identity is None:
            raise RuntimeError(
                "native load_diffusion_model ran without an expected execution identity"
            )
        with native_execution_span("load", "load"):
            handle = _load_detected_component(
                diffusion_model, "model", context, storage_dtype=storage_dtype
            )
        return cls.outputs(model=handle)


class NativeLoadCheckpoint(LoadCheckpoint):
    """Load a checkpoint through native checkpoint or component plans."""

    @classmethod
    def execute(cls, *, checkpoint: object) -> Mapping[str, object]:
        if not isinstance(checkpoint, AssetRef):
            raise TypeError(f"checkpoint must be an AssetRef, got {type(checkpoint).__name__}")
        path = resolve_weight_source(checkpoint.local_path(), logical_name=checkpoint.name)
        context = current_execution_context()
        if context is None:
            raise RuntimeError(
                "native load_checkpoint ran without an execution context; "
                "the dispatch host must supply the selected body identity"
            )
        expected_identity = context.expected_execution_identity
        if expected_identity is None:
            raise RuntimeError("native load_checkpoint ran without an expected execution identity")
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        source = inference.load_safetensors_header(
            path,
            asset_digest=checkpoint.digest,
            asset_size=checkpoint.size,
        )
        plan = inference.plan_native(checkpoint=source, fp8_matmul=context.fp8_matmul)
        embedding_resource = (
            _freeze_embedding_resource() if uses_classic_embedding_bindings(plan) else (None, None)
        )
        embedding_index, _ = embedding_resource
        embedding_binding_digest = (
            None if embedding_index is None else embedding_index.binding_digest
        )
        attention_kwargs: dict[str, object] = {}
        if context.attention_route_token is None:
            runtime = _load_runtime(
                path,
                expected_identity,
                context.fp8_matmul,
                context.extension_snapshot_digest,
                embedding_resource,
                source_assets={"checkpoint": checkpoint},
            )
        else:
            attention_kwargs = {
                "attention_policy": context.attention_policy,
                "attention_route_token": context.attention_route_token,
            }
            runtime = _load_runtime(
                path,
                expected_identity,
                context.fp8_matmul,
                context.extension_snapshot_digest,
                embedding_resource,
                context.attention_policy,
                context.attention_route_token,
                source_assets={"checkpoint": checkpoint},
            )
        recipe = _runtime_recipe(
            inference,
            {"checkpoint": checkpoint},
            {"checkpoint": source},
            fp8_matmul=context.fp8_matmul,
            extension_snapshot_digest=context.extension_snapshot_digest,
            plan=plan,
            embedding_binding_digest=embedding_binding_digest,
            **attention_kwargs,
        )
        if recipe.runtime_identity != expected_identity:
            raise RuntimeError("native checkpoint recipe identity does not match dispatch identity")
        source_resolvers = (
            {} if checkpoint.resolver is None else {checkpoint.digest: checkpoint.resolver}
        )

        def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
            return _materialize_recipe_handle(
                next_recipe, next_resolvers, torch, embedding_resource
            )

        handle = _build_runtime_handle(
            runtime,
            torch,
            recipe=recipe,
            materializer=materializer,
            source_resolvers=source_resolvers,
        )
        pool = default_pool()
        pool.label(handle, checkpoint.name)
        handle.attach_pool(pool)
        return cls.outputs(model=handle, clip=handle, vae=handle)


class NativeLoadModelProfile(Node):
    """Re-probe a stored model profile and project an existing loader result."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_model_profile")

    @classmethod
    def execute(
        cls,
        *,
        checkpoint: object,
        entries: str,
        output_spec: OutputInterface,
    ) -> Mapping[str, object]:
        if not isinstance(checkpoint, AssetRef):
            raise TypeError(f"checkpoint must be an AssetRef, got {type(checkpoint).__name__}")
        inference = importlib.import_module("dinkster_inference")
        path = checkpoint.local_path()
        with checkpoint.open() as verified:
            profile = inference.load_model_output_profile(
                path,
                asset_digest=checkpoint.digest,
                asset_size=os.fstat(verified.fileno()).st_size,
                stored=entries,
                handle=verified,
            )
        loaded = (
            NativeLoadCheckpoint.execute(checkpoint=checkpoint)
            if profile.kind == "checkpoint"
            else NativeLoadDiffusionModel.execute(
                diffusion_model=checkpoint,
                weight_dtype="default",
            )
        )
        return {output.id: loaded[output.id] for output in output_spec.outputs}


def _nvfp4_runtime_status(  # pyright: ignore[reportUnusedFunction]
    handle: NativeRuntimeHandle,
) -> object:
    """Return inference's immutable per-runtime NVFP4 snapshot unchanged."""
    module = importlib.import_module("dinkster_inference_torch._nvfp4_diagnostics")
    return module.nvfp4_runtime_status(handle.runtime)


def _attention_runtime_status(  # pyright: ignore[reportUnusedFunction]
    handle: NativeRuntimeHandle,
) -> object:
    """Return immutable recipe-bound attention routing evidence."""
    handle.require_active()
    inference = importlib.import_module("dinkster_inference")
    return inference.resolve_attention_runtime_status(
        handle.recipe.knobs.attention_policy,
        handle.recipe.knobs.attention_route_token,
    )


def _native_controlnet(value: object) -> _NativeControlNetResource:
    if not isinstance(value, _NativeControlNetResource):
        raise TypeError(
            f"control_net must be a native ControlNet resource, got {type(value).__name__}"
        )
    if value.handle is not None:
        value.handle.require_active()
    return value


def _snapshot_control_hint(
    image: object, torch: Any, inference_torch: Any, channels: int = 3
) -> _ControlHintSnapshot:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"ControlNet image must be a torch.Tensor, got {type(image).__name__}")
    tensor = cast("Any", image)
    allowed_channels = (1, 3) if channels == 1 else (channels,)
    if tensor.ndim != 4 or tensor.shape[-1] not in allowed_channels:
        raise ValueError(
            f"ControlNet image must be [batch x H x W x {channels}], got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] < 1 or tensor.shape[1] < 1 or tensor.shape[2] < 1:
        raise ValueError("ControlNet image batch and spatial dimensions must be positive")
    hint = (
        tensor.detach()
        .movedim(-1, 1)
        .to(device=torch.device("cpu"), dtype=torch.float32)
        .contiguous()
        .clone()
    )
    shape = cast("tuple[int, int, int, int]", tuple(int(dim) for dim in hint.shape))
    return _ControlHintSnapshot(
        shape,
        hint.numpy().tobytes(order="C"),
        inference_torch.sd_control_hint_digest(hint),
    )


def _control_child_id(
    resource: _NativeControlNetResource,
    hint: _ControlHintSnapshot,
    strength: float,
    start_percent: float,
    end_percent: float,
    previous: Any | None,
) -> str:
    identity = "\n".join(
        (
            resource.resource_digest,
            hint.digest,
            repr(strength),
            repr(start_percent),
            repr(end_percent),
            *(() if resource.mode is None else (repr(resource.mode),)),
            "" if previous is None else previous.child_id,
        )
    )
    return "classic-" + hashlib.sha256(identity.encode("ascii")).hexdigest()[:24]


def _extend_control_binding(
    previous: _ClassicControlBinding | None,
    *,
    resource: _NativeControlNetResource,
    hint: _ControlHintSnapshot,
    strength: float,
    start_percent: float,
    end_percent: float,
    apply_to_uncond: bool,
    inference: Any,
) -> _ClassicControlBinding:
    pool = default_pool()
    resident_id = "resident:" + pool.rid_for(resource)
    previous_application = None if previous is None else previous.application
    child_id = _control_child_id(
        resource, hint, strength, start_percent, end_percent, previous_application
    )
    application = inference.ControlApplication(
        child_id,
        inference.PayloadReference(hint.digest),
        strength,
        inference.PercentRange(start_percent, end_percent),
        previous_application,
        mode=resource.mode,
    )
    entry = _ClassicControlEntry(child_id, resident_id, resource.resource_digest, hint)
    return _ClassicControlBinding(
        application,
        (*(() if previous is None else previous.entries), entry),
        apply_to_uncond,
    )


def _apply_classic_control(conditioning: object, **kwargs: Any) -> list[list[object]]:
    result: list[list[object]] = []
    for raw_entry in _condition_entries(conditioning, "conditioning"):
        metadata = cast("Mapping[object, object]", raw_entry[1])
        previous = metadata.get("control")
        if previous is not None and not isinstance(previous, _ClassicControlBinding):
            raise TypeError(
                "classic ControlNet cannot chain after a non-native control value; "
                "all ControlNet apply nodes must execute on the native arm"
            )
        binding = _extend_control_binding(previous, **kwargs)
        updated = dict(metadata)
        updated["control"] = binding
        updated["control_apply_to_uncond"] = binding.apply_to_uncond
        result.append([raw_entry[0], updated])
    return result


def _enroll_control_module(
    module: Any, torch: Any, inference_torch: Any, *, discard_on_release: bool = False
) -> NativeComponentHandle:
    load_device = select_load_device(torch)
    coordinator = default_native_residency()
    mechanism = coordinator.enroll_component(
        module,
        load_device=load_device,
        offload_device=torch.device("cpu"),
        enroller=inference_torch.enroll_component,
    )
    return NativeComponentHandle(
        module,
        mechanism,
        load_device,
        resource_identity=module.resource_digest,
        coordinator=coordinator,
        discard_on_release=discard_on_release,
    )


class NativeControlNetLoader(ControlNetLoader):
    @classmethod
    def execute(cls, *, control_net_name: object) -> Mapping[str, object]:
        if not isinstance(control_net_name, AssetRef):
            raise TypeError(
                f"control_net_name must be an AssetRef, got {type(control_net_name).__name__}"
            )
        from dinkster_inference.component_registry import component_plans, execution_symbol

        path = resolve_weight_source(
            control_net_name.local_path(), logical_name=control_net_name.name
        )
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        source = inference.load_safetensors_header(
            path, asset_digest=control_net_name.digest, asset_size=control_net_name.size
        )
        descriptor, _role, plan = _active_inference_registries().components.select(
            source, path, "controlnet"
        )
        log.info(
            "Control component %s selected by tensor geometry; "
            "target-model compatibility is checked during execution, not by family name",
            descriptor.id,
        )
        torch = _torch()
        load_device = select_load_device(torch)
        controlnet_dtype = (
            torch.float32
            if load_device.type == "cpu"
            else getattr(torch, descriptor.default_diffusion_dtype.name)
        )
        handle = None
        digest = control_net_name.digest.removeprefix("blake3:")
        if not getattr(descriptor, "requires_base", False):
            load_kwargs: dict[str, object] = {}
            attention_backend = descriptor.family.engine.attention_backend("controlnet")
            if attention_backend is not None:
                context = current_execution_context()
                if descriptor.family.engine.attention_requires_route and (
                    context is None or context.attention_route_token is None
                ):
                    raise ValueError(
                        f"control component {descriptor.id} requires an authenticated "
                        "attention route"
                    )
                load_kwargs.update(
                    attention_policy=("auto" if context is None else context.attention_policy),
                    attention_route_token=(
                        None if context is None else context.attention_route_token
                    ),
                    attention_backend=attention_backend,
                )
            module = execution_symbol(descriptor.loader)(plan, controlnet_dtype, **load_kwargs)
            handle = _enroll_control_module(module, torch, inference_torch)
            digest = module.resource_digest
        resource = _NativeControlNetResource(
            handle,
            control_net_name.digest,
            digest,
            getattr(plan, "source_layout", "native"),
            descriptor,
            plan,
            getattr(
                component_plans(plan)[0].config,
                "hint_channels",
                getattr(descriptor, "hint_channels", 3),
            ),
        )
        if handle is not None:
            pool = default_pool()
            pool.label(handle, control_net_name.name)
            handle.attach_pool(pool)
        return cls.outputs(control_net=resource)


class NativeControlNetApply(ControlNetApply):
    @classmethod
    def execute(
        cls,
        *,
        conditioning: object,
        control_net: object,
        image: object,
        strength: float,
    ) -> Mapping[str, object]:
        if strength == 0.0:
            return cls.outputs(conditioning=conditioning)
        resource = _native_controlnet(control_net)
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        hint = _snapshot_control_hint(image, torch, inference_torch, resource.hint_channels)
        applied = _apply_classic_control(
            conditioning,
            resource=resource,
            hint=hint,
            strength=float(strength),
            start_percent=0.0,
            end_percent=1.0,
            apply_to_uncond=True,
            inference=inference,
        )
        return cls.outputs(conditioning=applied)


class NativeControlNetApplyAdvanced(ControlNetApplyAdvanced):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        control_net: object,
        image: object,
        strength: float,
        start_percent: float,
        end_percent: float,
        vae: object = None,
    ) -> Mapping[str, object]:
        if vae is not None:
            raise ValueError(
                "SD1.5 native ControlNetApplyAdvanced does not support the optional VAE input"
            )
        if strength == 0.0:
            return cls.outputs(positive=positive, negative=negative)
        resource = _native_controlnet(control_net)
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        hint = _snapshot_control_hint(image, torch, inference_torch, resource.hint_channels)
        return cls.outputs(
            positive=_apply_classic_control(
                positive,
                resource=resource,
                hint=hint,
                strength=float(strength),
                start_percent=float(start_percent),
                end_percent=float(end_percent),
                apply_to_uncond=False,
                inference=inference,
            ),
            negative=_apply_classic_control(
                negative,
                resource=resource,
                hint=hint,
                strength=float(strength),
                start_percent=float(start_percent),
                end_percent=float(end_percent),
                apply_to_uncond=False,
                inference=inference,
            ),
        )


class NativeLoadZImageControlPatch(LoadZImageControlPatch):
    @classmethod
    def execute(cls, *, model_patch: object) -> Mapping[str, object]:
        if not isinstance(model_patch, AssetRef):
            raise TypeError(f"model_patch must be an AssetRef, got {type(model_patch).__name__}")
        context = current_execution_context()
        if context is None:
            raise RuntimeError("native model patch loading requires an execution context")
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        source = inference.load_safetensors_header(model_patch.local_path())
        attention_kwargs: dict[str, object] = {}
        if context.attention_route_token is not None:
            attention_kwargs = {
                "attention_policy": context.attention_policy,
                "attention_route_token": context.attention_route_token,
            }
        source_keys = source.keys()
        if (
            any(key.endswith("duration_head.attention_pooler.query_tokens") for key in source_keys)
            or "attention_pooler.query_tokens" in source_keys
        ):
            identity, _ = _ltxav_component_identity(
                inference,
                model_patch,
                "duration_head",
                "float32",
            )
            handle = _build_component_runtime_handle(
                _component_descriptor("dinkster.ltxav"),
                model_patch,
                "duration_head",
                identity,
                _torch(),
                compute_dtype="float32",
            )
            return cls.outputs(model_patch=handle)
        if "audio_proj.proj1.weight" in source.keys():
            plan = inference.plan_wan21_multitalk(source, asset_digest=model_patch.digest)
            assembled = inference_torch.assemble_wan21_multitalk(plan, **attention_kwargs)
            module = assembled.patch
            resource_identity = inference.extend_runtime_identity(
                inference.build_runtime_identity_from_facts(
                    "dinkster.wan21",
                    inference.runtime_component_identity("dinkster.wan21", (plan.patch,)),
                    diffusion_dtype=inference.BFLOAT16.name,
                    text_dtype="unloaded",
                    vae_dtype="unloaded",
                    fp8_matmul=False,
                    runtime_facts=plan.patch.runtime_facts,
                ),
                (f"resource={assembled.resource_digest}",),
            )
        else:
            plan = inference.plan_z_image_control(source, asset_digest=model_patch.digest)
            assembled = inference_torch.assemble_z_image_control(plan, **attention_kwargs)
            module = assembled.control
            resource_identity = assembled.resource_digest
        torch = _torch()
        load_device = select_load_device(torch)
        coordinator = default_native_residency()
        mechanism = coordinator.enroll_component(
            module,
            load_device=load_device,
            offload_device=torch.device("cpu"),
            enroller=inference_torch.enroll_component,
        )
        handle = NativeComponentHandle(
            module,
            mechanism,
            load_device,
            resource_identity=resource_identity,
            coordinator=coordinator,
        )
        pool = default_pool()
        pool.label(handle, model_patch.name)
        handle.attach_pool(pool)
        return cls.outputs(model_patch=handle)


class NativeApplyZImageControlPatch(ApplyZImageControlPatch):
    @classmethod
    def execute(
        cls,
        *,
        model: object,
        model_patch: object,
        vae: object,
        image: object,
        strength: float,
    ) -> Mapping[str, object]:
        (
            handle,
            overlays,
            resolvers,
            existing_control,
            sampling_shift,
            guidance_transforms,
            context_windows,
            chroma_radiance_options,
        ) = _native_model(model, "model")
        if getattr(handle.runtime, "accepts_z_image_control", False) is not True:
            raise TypeError("Z-Image Fun ControlNet requires a native Z-Image model")
        if _native_handle(vae, "vae") is not handle:
            raise ValueError("model and VAE must come from the same Z-Image runtime")
        if existing_control is not None:
            raise ValueError("a Z-Image model can carry only one Fun ControlNet patch")
        if not isinstance(model_patch, NativeComponentHandle):
            raise TypeError("model_patch must be a native Z-Image control patch")
        model_patch.require_active()
        if getattr(model_patch.module, "accepts_z_image_control_binding", False) is not True:
            raise TypeError("model_patch must be a native Z-Image control patch")
        torch = _torch()
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        tensor = cast("Any", image)
        if tensor.ndim != 4 or tensor.shape[-1] < 3:
            raise ValueError("image must be an NHWC torch.Tensor with at least three channels")
        if not math.isfinite(strength) or not -10.0 <= strength <= 10.0:
            raise ValueError(f"strength must be finite and in [-10.0, 10.0], got {strength}")
        control_image = tensor[..., :3].permute(0, 3, 1, 2).detach().to("cpu").contiguous()
        binding = _ZImageControlBinding(model_patch, control_image, strength)
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                binding,
                sampling_shift,
                guidance_transforms,
                context_windows,
                chroma_radiance_options,
                sampling_cache=_native_model_sampling_cache(model),
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=_native_model_sampling_space(model),
            )
        )


def load_native_runtime_handle(assets: Mapping[str, object]) -> NativeRuntimeHandle:
    """Load a fixed-role asset bundle through the production native handle path."""
    # Resolve and verify every fixed-role asset before reading headers or
    # constructing runtime state.
    for role, asset in assets.items():
        if not isinstance(asset, AssetRef):
            raise TypeError(f"{role} must be an AssetRef, got {type(asset).__name__}")
    refs = cast("dict[str, AssetRef]", assets)
    paths = {role: refs[role].local_path() for role in assets}
    context = current_execution_context()
    if context is None:
        raise RuntimeError(
            "native runtime loading ran without an execution context; "
            "the host must supply execution knobs"
        )
    torch = _torch()
    inference = importlib.import_module("dinkster_inference")
    canonical_paths = _canonical_runtime_sources(paths)
    sources = {
        role: inference.load_safetensors_header(path) for role, path in canonical_paths.items()
    }
    plan = inference.plan_native(**sources, fp8_matmul=context.fp8_matmul)
    embedding_resource = (
        _freeze_embedding_resource() if uses_classic_embedding_bindings(plan) else (None, None)
    )
    embedding_index, _ = embedding_resource
    embedding_binding_digest = None if embedding_index is None else embedding_index.binding_digest
    attention_kwargs: dict[str, object] = {}
    if context.attention_route_token is not None:
        attention_kwargs = {
            "attention_policy": context.attention_policy,
            "attention_route_token": context.attention_route_token,
        }
    recipe = _runtime_recipe(
        inference,
        refs,
        sources,
        fp8_matmul=context.fp8_matmul,
        extension_snapshot_digest=context.extension_snapshot_digest,
        plan=plan,
        embedding_binding_digest=embedding_binding_digest,
        **attention_kwargs,
    )
    expected_identity = context.expected_execution_identity or recipe.runtime_identity
    if context.attention_route_token is None:
        runtime = _load_runtime(
            canonical_paths,
            expected_identity,
            context.fp8_matmul,
            context.extension_snapshot_digest,
            embedding_resource,
            source_assets=refs,
        )
    else:
        runtime = _load_runtime(
            canonical_paths,
            expected_identity,
            context.fp8_matmul,
            context.extension_snapshot_digest,
            embedding_resource,
            context.attention_policy,
            context.attention_route_token,
            source_assets=refs,
        )
    if recipe.runtime_identity != expected_identity:
        raise RuntimeError("native runtime recipe identity does not match dispatch identity")
    source_resolvers = {
        asset.digest: asset.resolver for asset in refs.values() if asset.resolver is not None
    }

    def materializer(next_recipe: Any, next_resolvers: Mapping[str, object]) -> Any:
        return _materialize_recipe_handle(next_recipe, next_resolvers, torch, embedding_resource)

    handle = _build_runtime_handle(
        runtime,
        torch,
        recipe=recipe,
        materializer=materializer,
        source_resolvers=source_resolvers,
    )
    pool = default_pool()
    label_source = refs["diffusion"] if "diffusion" in refs else refs["checkpoint"]
    pool.label(handle, label_source.name)
    handle.attach_pool(pool)
    return handle


def _lora_execution_mode(handle: NativeRuntimeHandle, requested: str) -> str:
    if requested == "precalculate" or callable(getattr(handle.runtime, "sample_scheduled", None)):
        return requested
    log.warning(
        "native runtime does not support scheduled LoRA patch resolution; "
        "falling back to precalculate execution mode"
    )
    return "precalculate"


def _apply_native_lora_stack(
    model: object,
    clip: object,
    loras: Sequence[tuple[object, float, float]],
    execution_mode: str,
) -> tuple[object, object]:
    validate_lora_execution_mode(execution_mode)
    (
        model_handle,
        existing,
        existing_resolvers,
        z_image_control,
        sampling_shift,
        guidance_transforms,
        context_windows,
        chroma_radiance_options,
    ) = _native_model(model, "model")
    inference = importlib.import_module("dinkster_inference")
    if model_handle.recipe.family_id == inference.MINIMAX_H3_CONFIG.family_id:
        raise ValueError("MiniMax H3 LoRAs require Load LoRA")

    active: list[tuple[AssetRef, float, float]] = []
    for lora, strength_model, strength_clip in loras:
        if not isinstance(lora, AssetRef):
            raise TypeError(f"lora must be an AssetRef, got {type(lora).__name__}")
        if strength_model != 0.0 or strength_clip != 0.0:
            active.append((lora, strength_model, strength_clip))
    if model_handle.recipe.family_id in inference.FLUX2_TEXT_ROLE_BY_FAMILY and isinstance(
        clip, NativeComponentHandle
    ):
        text_handle = load_registered_component(clip, "clip")
        if text_handle.recipe is None or (
            text_handle.recipe.family_id != model_handle.recipe.family_id
        ):
            raise ValueError("Flux2 model and clip must belong to the same family")
        return _apply_split_flux2_lora_stack(
            model,
            model_handle,
            text_handle,
            active,
            execution_mode,
            existing=existing,
            existing_resolvers=existing_resolvers,
            z_image_control=z_image_control,
            sampling_shift=sampling_shift,
            guidance_transforms=guidance_transforms,
            context_windows=context_windows,
            chroma_radiance_options=chroma_radiance_options,
        )
    clip_handle = _native_handle(clip, "clip")
    if model_handle is not clip_handle:
        raise ValueError("native Load LoRA requires model and clip from the same runtime handle")
    if not active:
        if execution_mode == "precalculate" and existing:
            clone = model_handle.clone(existing, source_resolvers=existing_resolvers)
            bundle_name = _recipe_bundle_name(model_handle.recipe)
            overlay_name = getattr(getattr(existing[-1], "source", None), "name", "LoRA stack")
            default_pool().label(clone, f"{bundle_name} + {overlay_name}")
            patched_model: object = clone
            if (
                z_image_control is not None
                or _native_model_sampling_space(model) is not None
                or sampling_shift is not None
                or guidance_transforms
                or context_windows is not None
                or chroma_radiance_options
                or _native_model_sampling_cache(model) is not None
                or _native_model_sampling_timeline(model) is not None
            ):
                patched_model = _NativeModelOverlay(
                    clone,
                    (),
                    {},
                    z_image_control,
                    sampling_shift,
                    guidance_transforms,
                    context_windows,
                    chroma_radiance_options,
                    sampling_cache=_native_model_sampling_cache(model),
                    sampling_timeline=_native_model_sampling_timeline(model),
                    sampling_space=_native_model_sampling_space(model),
                )
            return patched_model, clone
        return model, clip

    overlays = tuple(
        _native_lora_overlay(model_handle, lora, strength_model, strength_clip)
        for lora, strength_model, strength_clip in active
    )
    combined = existing + overlays
    combined_resolvers = dict(existing_resolvers)
    for lora, _, _ in active:
        if lora.resolver is not None:
            combined_resolvers[lora.digest] = lora.resolver
    has_text_patches = any(_overlay_targets(item, "text") for item in combined)
    has_offset_patches = any(_overlay_has_offsets(item) for item in combined)
    execution_mode = _lora_execution_mode(model_handle, execution_mode)
    if execution_mode == "attach" and has_text_patches:
        raise ValueError("LoRA attach mode supports diffusion-only patches")
    if execution_mode == "attach" and has_offset_patches:
        raise ValueError("LoRA offset patches require precalculate execution mode")
    if (
        execution_mode == "precalculate"
        or has_text_patches
        or has_offset_patches
        or z_image_control
    ):
        clone = model_handle.clone(combined, source_resolvers=combined_resolvers)
        bundle_name = _recipe_bundle_name(model_handle.recipe)
        default_pool().label(clone, f"{bundle_name} + {active[-1][0].name}")
        patched_model: object = clone
        if (
            z_image_control is not None
            or _native_model_sampling_space(model) is not None
            or sampling_shift is not None
            or guidance_transforms
            or context_windows is not None
            or chroma_radiance_options
            or _native_model_sampling_cache(model) is not None
            or _native_model_sampling_timeline(model) is not None
        ):
            patched_model = _NativeModelOverlay(
                clone,
                (),
                {},
                z_image_control,
                sampling_shift,
                guidance_transforms,
                context_windows,
                chroma_radiance_options,
                sampling_cache=_native_model_sampling_cache(model),
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=_native_model_sampling_space(model),
            )
        return patched_model, clone
    return (
        _NativeModelOverlay(
            model_handle,
            combined,
            combined_resolvers,
            z_image_control,
            sampling_shift,
            guidance_transforms,
            context_windows,
            chroma_radiance_options,
            sampling_cache=_native_model_sampling_cache(model),
            sampling_timeline=_native_model_sampling_timeline(model),
            sampling_space=_native_model_sampling_space(model),
        ),
        clip_handle,
    )


def _apply_split_flux2_lora_stack(
    model: object,
    model_handle: NativeRuntimeHandle,
    text_handle: NativeComponentHandle,
    active: Sequence[tuple[AssetRef, float, float]],
    execution_mode: str,
    *,
    existing: tuple[Any, ...],
    existing_resolvers: Mapping[str, object],
    z_image_control: _ZImageControlBinding | None,
    sampling_shift: float | None,
    guidance_transforms: tuple[tuple[str, Any], ...],
    context_windows: ContextWindowsSpec | None,
    chroma_radiance_options: tuple[Any, ...],
) -> tuple[object, object]:
    if not active:
        if execution_mode == "precalculate" and existing:
            clone = model_handle.clone(existing, source_resolvers=existing_resolvers)
            bundle_name = _recipe_bundle_name(model_handle.recipe)
            overlay_name = getattr(getattr(existing[-1], "source", None), "name", "LoRA stack")
            default_pool().label(clone, f"{bundle_name} + {overlay_name}")
            patched_model: object = clone
            if (
                z_image_control is not None
                or _native_model_sampling_space(model) is not None
                or sampling_shift is not None
                or guidance_transforms
                or context_windows is not None
                or chroma_radiance_options
                or _native_model_sampling_cache(model) is not None
                or _native_model_sampling_timeline(model) is not None
            ):
                patched_model = _NativeModelOverlay(
                    clone,
                    (),
                    {},
                    z_image_control,
                    sampling_shift,
                    guidance_transforms,
                    context_windows,
                    chroma_radiance_options,
                    sampling_cache=_native_model_sampling_cache(model),
                    sampling_timeline=_native_model_sampling_timeline(model),
                    sampling_space=_native_model_sampling_space(model),
                )
            return patched_model, text_handle
        return model, text_handle

    inference = importlib.import_module("dinkster_inference")
    overlays = tuple(
        _native_lora_overlay(
            model_handle,
            lora,
            strength_model,
            strength_clip,
            text_handle=text_handle,
        )
        for lora, strength_model, strength_clip in active
    )
    combined = existing + overlays
    combined_resolvers = dict(existing_resolvers)
    for lora, _, _ in active:
        if lora.resolver is not None:
            combined_resolvers[lora.digest] = lora.resolver
    text_recipe = text_handle.recipe
    assert text_recipe is not None
    text_role = text_recipe.sources[0].role
    diffusion_overlays = tuple(
        selected
        for overlay in combined
        if (selected := _overlay_for_components(inference, overlay, {"diffusion"})) is not None
    )
    text_overlays = tuple(
        selected
        for overlay in overlays
        if (selected := _overlay_for_components(inference, overlay, {text_role})) is not None
    )
    has_text_patches = bool(text_overlays)
    has_offset_patches = any(_overlay_has_offsets(item) for item in combined)
    if execution_mode == "attach" and has_text_patches:
        raise ValueError("LoRA attach mode supports diffusion-only patches")
    if execution_mode == "attach" and has_offset_patches:
        raise ValueError("LoRA offset patches require precalculate execution mode")
    if execution_mode != "precalculate" and not has_text_patches and not has_offset_patches:
        return (
            _NativeModelOverlay(
                model_handle,
                combined,
                combined_resolvers,
                z_image_control,
                sampling_shift,
                guidance_transforms,
                context_windows,
                chroma_radiance_options,
                sampling_cache=_native_model_sampling_cache(model),
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=_native_model_sampling_space(model),
            ),
            text_handle,
        )

    bundle_name = _recipe_bundle_name(model_handle.recipe)
    overlay_name = active[-1][0].name
    patched_handle = model_handle
    if diffusion_overlays:
        patched_handle = model_handle.clone(
            diffusion_overlays,
            source_resolvers=combined_resolvers,
        )
        default_pool().label(patched_handle, f"{bundle_name} + {overlay_name}")
    patched_text = text_handle
    if text_overlays:
        patched_text = text_handle.clone(
            text_overlays,
            source_resolvers=combined_resolvers,
        )
        default_pool().label(patched_text, f"{bundle_name} text + {overlay_name}")
    patched_model = patched_handle
    if (
        z_image_control is not None
        or _native_model_sampling_space(model) is not None
        or sampling_shift is not None
        or guidance_transforms
        or context_windows is not None
        or chroma_radiance_options
        or _native_model_sampling_cache(model) is not None
        or _native_model_sampling_timeline(model) is not None
    ):
        patched_model = _NativeModelOverlay(
            patched_handle,
            (),
            {},
            z_image_control,
            sampling_shift,
            guidance_transforms,
            context_windows,
            chroma_radiance_options,
            sampling_cache=_native_model_sampling_cache(model),
            sampling_timeline=_native_model_sampling_timeline(model),
            sampling_space=_native_model_sampling_space(model),
        )
    return patched_model, patched_text


def _apply_native_model_lora_stack(
    model: object,
    loras: Sequence[tuple[object, float]],
    execution_mode: str,
) -> object:
    validate_lora_execution_mode(execution_mode)
    (
        handle,
        existing,
        existing_resolvers,
        z_image_control,
        sampling_shift,
        guidance_transforms,
        context_windows,
        chroma_radiance_options,
    ) = _native_model(model, "model")
    active: list[tuple[AssetRef, float]] = []
    for lora, strength_model in loras:
        if not isinstance(lora, AssetRef):
            raise TypeError(f"lora must be an AssetRef, got {type(lora).__name__}")
        if strength_model != 0.0:
            active.append((lora, strength_model))
    if not active:
        if execution_mode == "precalculate" and existing:
            clone = handle.clone(existing, source_resolvers=existing_resolvers)
            bundle_name = _recipe_bundle_name(handle.recipe)
            overlay_name = getattr(getattr(existing[-1], "source", None), "name", "LoRA stack")
            default_pool().label(clone, f"{bundle_name} + {overlay_name}")
            if (
                z_image_control is not None
                or _native_model_sampling_space(model) is not None
                or sampling_shift is not None
                or guidance_transforms
                or context_windows is not None
                or chroma_radiance_options
                or _native_model_sampling_cache(model) is not None
                or _native_model_sampling_timeline(model) is not None
            ):
                return _NativeModelOverlay(
                    clone,
                    (),
                    {},
                    z_image_control,
                    sampling_shift,
                    guidance_transforms,
                    context_windows,
                    chroma_radiance_options,
                    sampling_cache=_native_model_sampling_cache(model),
                    sampling_timeline=_native_model_sampling_timeline(model),
                    sampling_space=_native_model_sampling_space(model),
                )
            return clone
        return model

    overlays = tuple(
        _native_lora_overlay(handle, lora, strength_model, 0.0) for lora, strength_model in active
    )
    combined = existing + overlays
    combined_resolvers = dict(existing_resolvers)
    for lora, _ in active:
        if lora.resolver is not None:
            combined_resolvers[lora.digest] = lora.resolver
    has_offset_patches = any(_overlay_has_offsets(item) for item in combined)
    execution_mode = _lora_execution_mode(handle, execution_mode)
    if execution_mode == "attach" and has_offset_patches:
        raise ValueError("LoRA offset patches require precalculate execution mode")
    if execution_mode == "precalculate" or has_offset_patches or z_image_control:
        clone = handle.clone(combined, source_resolvers=combined_resolvers)
        bundle_name = _recipe_bundle_name(handle.recipe)
        default_pool().label(clone, f"{bundle_name} + {active[-1][0].name}")
        if (
            z_image_control is not None
            or _native_model_sampling_space(model) is not None
            or sampling_shift is not None
            or guidance_transforms
            or context_windows is not None
            or chroma_radiance_options
            or _native_model_sampling_cache(model) is not None
            or _native_model_sampling_timeline(model) is not None
        ):
            return _NativeModelOverlay(
                clone,
                (),
                {},
                z_image_control,
                sampling_shift,
                guidance_transforms,
                context_windows,
                chroma_radiance_options,
                sampling_cache=_native_model_sampling_cache(model),
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=_native_model_sampling_space(model),
            )
        return clone
    return _NativeModelOverlay(
        handle,
        combined,
        combined_resolvers,
        z_image_control,
        sampling_shift,
        guidance_transforms,
        context_windows,
        chroma_radiance_options,
        sampling_cache=_native_model_sampling_cache(model),
        sampling_timeline=_native_model_sampling_timeline(model),
        sampling_space=_native_model_sampling_space(model),
    )


class NativeLoadLora(LoadLora):
    """Apply one normalized LoRA through an explicit native execution strategy."""

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        clip: object,
        lora: object,
        strength_model: float,
        strength_clip: float,
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        patched_model, patched_clip = _apply_native_lora_stack(
            model,
            clip,
            ((lora, strength_model, strength_clip),),
            execution_mode,
        )
        return cls.outputs(model=patched_model, clip=patched_clip)


class NativeLoadLoraModelOnly(LoadLoraModelOnly):
    """Apply one diffusion-only LoRA through an explicit native strategy."""

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        lora: object,
        strength_model: float,
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        return cls.outputs(
            model=_apply_native_model_lora_stack(
                model,
                ((lora, strength_model),),
                execution_mode,
            )
        )


class NativeClipTextEncode(CLIPTextEncode):
    """Encode text through the runtime and emit Comfy conditioning data."""

    @classmethod
    def execute(cls, *, text: str, clip: object) -> Mapping[str, object]:
        handle = _native_handle(clip, "clip")
        torch = _torch()
        with handle.stage("text"):
            with torch.inference_mode():
                conditioning = handle.runtime.encode_text(text)
        prepare_text = getattr(handle.runtime, "prepare_text_conditioning", None)
        if callable(prepare_text):
            inference = importlib.import_module("dinkster_inference")
            prepared = prepare_text(conditioning)
            value = inference.PreparedMultiStreamConditioning(
                handle.runtime.conditioning_identity,
                prepared,
            )
            wire: list[list[object]] = [[value, {}]]
            return cls.outputs(conditioning=wire)
        if getattr(handle.runtime, "preserves_text_conditioning", False):
            return cls.outputs(
                conditioning=[
                    [conditioning.embeddings, {_NATIVE_PREPARED_CONDITIONING_KEY: conditioning}]
                ]
            )
        metadata: dict[str, object] = {}
        if conditioning.pooled is not None:
            metadata["pooled_output"] = conditioning.pooled
        metadata[_NATIVE_PROMPT_KEY] = (text, handle.recipe.runtime_identity)
        return cls.outputs(conditioning=[[conditioning.embeddings, metadata]])


@dataclass(frozen=True, slots=True)
class _Wan21ClipVisionOutput:
    conditioning_identity: str
    embedding: object


class NativeWan21ClipVisionEncode(Wan21ClipVisionEncode):
    @classmethod
    def execute(cls, *, model: object, image: object) -> Mapping[str, object]:
        handle = _native_handle(model, "model")
        assembled = handle.runtime.assembled
        if assembled.diffusion.config.model_type != "i2v":
            raise ValueError("model must use the Wan 2.1 I2V profile")
        if assembled.clip_vision is None:
            raise ValueError("loaded Wan 2.1 I2V model has no CLIP vision encoder")
        torch = _torch()
        if type(image) is not torch.Tensor:
            raise TypeError("image must be an exact torch.Tensor")
        tensor = cast("Any", image)
        with handle.stage("vision"):
            with torch.inference_mode():
                embedding = handle.runtime.encode_vision(tensor.to(handle.load_device))
        return cls.outputs(
            clip_vision_output=_Wan21ClipVisionOutput(
                handle.runtime.conditioning_identity,
                embedding,
            )
        )


def _wan21_i2v_prepared(
    value: object,
    *,
    name: str,
    handle: NativeRuntimeHandle,
    inference: Any,
    concat_latent: Any,
    vision: Any = None,
) -> list[list[object]]:
    text = _prepared_multistream_conditioning(
        value,
        inference,
        name,
        handle.runtime.conditioning_identity,
    )
    if text is None:
        raise TypeError(f"{name} must contain Wan 2.1 text conditioning")
    prepared = handle.runtime.prepare_i2v_conditioning(text, concat_latent, vision)
    return [
        [
            inference.PreparedMultiStreamConditioning(
                handle.runtime.conditioning_identity,
                prepared,
            ),
            {},
        ]
    ]


def _wan21_clip_embedding(
    value: object,
    *,
    name: str,
    handle: NativeRuntimeHandle,
    torch: Any,
) -> Any:
    if type(value) is not _Wan21ClipVisionOutput:
        raise TypeError(f"{name} must come from native Wan 2.1 CLIP Vision Encode")
    output = value
    if output.conditioning_identity != handle.runtime.conditioning_identity:
        raise ValueError(f"{name} is incompatible with the native Wan profile")
    if type(output.embedding) is not torch.Tensor:
        raise TypeError(f"{name} embedding must be an exact torch.Tensor")
    return output.embedding


def _wan_bernini_prepared(
    value: object,
    *,
    name: str,
    handle: NativeRuntimeHandle,
    inference: Any,
    context_latents: tuple[Any, ...],
) -> list[list[object]]:
    text = _prepared_multistream_conditioning(
        value,
        inference,
        name,
        handle.runtime.conditioning_identity,
    )
    if text is None:
        raise TypeError(f"{name} must contain Wan text conditioning")
    prepared = handle.runtime.prepare_bernini_conditioning(text, context_latents)
    return [
        [
            inference.PreparedMultiStreamConditioning(
                handle.runtime.conditioning_identity,
                prepared,
            ),
            {},
        ]
    ]


class NativeBerniniConditioning(BerniniConditioning):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        source_video: object = None,
        reference_video: object = None,
        reference_images: object = None,
        ref_max_size: int = 848,
    ) -> Mapping[str, object]:
        for name, value in (
            ("width", width),
            ("height", height),
            ("ref_max_size", ref_max_size),
        ):
            if type(value) is not int or not 16 <= value <= 8192 or value % 16:
                raise ValueError(f"{name} must be a multiple of 16 between 16 and 8192")
        if type(length) is not int or not 1 <= length <= 8192 or (length - 1) % 4:
            raise ValueError("length must be 1 plus a multiple of 4 between 1 and 8192")
        if type(batch_size) is not int or not 1 <= batch_size <= 4096:
            raise ValueError("batch_size must be between 1 and 4096")

        handle = _native_handle(vae, "vae")
        config = handle.runtime.assembled.diffusion.config
        if getattr(config, "model_variant", None) != "bernini" or not callable(
            getattr(handle.runtime, "prepare_bernini_conditioning", None)
        ):
            raise ValueError("vae must come from a Wan 2.2 Bernini 14B profile")

        inference = importlib.import_module("dinkster_inference")
        torch = _torch()
        common_upscale = importlib.import_module("dinkster_inference_torch.resize").common_upscale
        reference_items: tuple[tuple[str, object], ...] = ()
        if reference_images is not None:
            if not isinstance(reference_images, Mapping):
                raise TypeError("reference_images must be a mapping")
            references = cast("Mapping[object, object]", reference_images)
            if len(references) > 8:
                raise ValueError("reference_images supports at most 8 slots")
            if any(
                type(name) is not str or not name.startswith("reference_image_")
                for name in references
            ):
                raise ValueError("reference_images keys must use the reference_image_ prefix")
            reference_items = tuple(
                (name, references[name])
                for name in sorted(cast("Mapping[str, object]", references))
                if references[name] is not None
            )

        def encode_reference(value: object, name: str) -> Any:
            image = _wan_image(value, name, torch)
            image_height = int(image.shape[1])
            image_width = int(image.shape[2])
            scale = min(ref_max_size / max(image_height, image_width), 1.0)
            resized_height = max(16, round(image_height * scale / 16) * 16)
            resized_width = max(16, round(image_width * scale / 16) * 16)
            resized = common_upscale(
                image[:, :, :, :3].movedim(-1, 1),
                resized_width,
                resized_height,
                "area",
                "disabled",
            ).movedim(1, -1)
            frames = int(resized.shape[0])
            content = resized.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
            return _wan_encode_content(
                handle=handle,
                content=content,
                expected=(
                    1,
                    16,
                    ((frames - 1) // 4) + 1,
                    resized_height // 8,
                    resized_width // 8,
                ),
                name=name,
                torch=torch,
            )

        context_latents: list[Any] = []
        if source_video is not None or reference_video is not None or reference_items:
            with handle.stage("vae", unload_before=("text",)):
                with torch.inference_mode():
                    if source_video is not None:
                        source = _wan_image(source_video, "source_video", torch)
                        resized = common_upscale(
                            source[:length, :, :, :3].movedim(-1, 1),
                            width,
                            height,
                            "area",
                            "center",
                        ).movedim(1, -1)
                        frames = int(resized.shape[0])
                        content = resized.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
                        context_latents.append(
                            _wan_encode_content(
                                handle=handle,
                                content=content,
                                expected=(
                                    1,
                                    16,
                                    ((frames - 1) // 4) + 1,
                                    height // 8,
                                    width // 8,
                                ),
                                name="source video",
                                torch=torch,
                            )
                        )
                        del content, resized, source
                    if reference_video is not None:
                        video = _wan_image(reference_video, "reference_video", torch)
                        context_latents.append(encode_reference(video[:length], "reference video"))
                        del video
                    for name, images in reference_items:
                        validated = _wan_image(images, name, torch)
                        for index in range(validated.shape[0]):
                            context_latents.append(
                                encode_reference(
                                    validated[index : index + 1],
                                    f"{name} frame {index}",
                                )
                            )
                        del validated

        prepared_positive = _wan_bernini_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            context_latents=tuple(context_latents),
        )
        prepared_negative = _wan_bernini_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            context_latents=tuple(context_latents),
        )
        latent = torch.zeros(
            (batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8),
            device="cpu",
            dtype=torch.float32,
        )
        return cls.outputs(
            positive=prepared_positive,
            negative=prepared_negative,
            latent={"samples": latent},
        )


class NativeWan21ImageToVideo(Wan21ImageToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        model: object,
        start_image: object,
        clip_vision_output: object = None,
        width: int,
        height: int,
        length: int,
        batch_size: int,
    ) -> Mapping[str, object]:
        if width < 16 or width > 16384 or width % 16 != 0:
            raise ValueError("width must be a multiple of 16 between 16 and 16384")
        if height < 16 or height > 16384 or height % 16 != 0:
            raise ValueError("height must be a multiple of 16 between 16 and 16384")
        if length < 1 or length > 16384 or (length - 1) % 4 != 0:
            raise ValueError("length must be 1 plus a multiple of 4 between 1 and 16384")
        if batch_size < 1 or batch_size > 4096:
            raise ValueError("batch_size must be between 1 and 4096")
        handle = _native_handle(model, "model")
        assembled = handle.runtime.assembled
        config = assembled.diffusion.config
        legacy_i2v = config.model_type == "i2v"
        wan22_i2v = (
            config.model_type == "t2v"
            and getattr(config, "in_channels", None) == 36
            and getattr(config, "out_channels", None) == 16
        )
        if not legacy_i2v and not wan22_i2v:
            raise ValueError("model must use a supported Wan I2V profile")
        if legacy_i2v and assembled.clip_vision is None:
            raise ValueError("loaded Wan 2.1 I2V model has no CLIP vision encoder")
        torch = _torch()
        if type(start_image) is not torch.Tensor:
            raise TypeError("start_image must be an exact torch.Tensor")
        image = cast("Any", start_image)
        if (
            image.ndim != 4
            or image.shape[0] <= 0
            or image.shape[1] <= 0
            or image.shape[2] <= 0
            or image.shape[3] < 3
            or not image.is_floating_point()
            or image.layout != torch.strided
        ):
            raise ValueError(
                "start_image must be a nonempty strided floating [frames,H,W,C>=3] tensor"
            )
        vision = None
        if legacy_i2v and clip_vision_output is not None:
            vision = _wan21_clip_embedding(
                clip_vision_output,
                name="clip_vision_output",
                handle=handle,
                torch=torch,
            )
        elif not legacy_i2v and clip_vision_output is not None:
            raise ValueError("Wan 2.2 I2V does not consume CLIP vision output")

        resized = (
            importlib.import_module("dinkster_inference_torch.resize")
            .common_upscale(
                image[:length, :, :, :3].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            )
            .movedim(1, -1)
        )
        frames = min(int(resized.shape[0]), length)
        padded = torch.full(
            (length, height, width, 3),
            0.5,
            device=resized.device,
            dtype=resized.dtype,
        )
        padded[:frames] = resized[:frames]
        content = padded.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
        with handle.stage("vae"):
            with torch.inference_mode():
                direct_oom: BaseException | None = None
                reference: Any = None
                try:
                    reference = handle.runtime.encode_content(content)
                except RuntimeError as caught:
                    if not _is_accelerator_oom(caught, torch=torch, device=handle.load_device):
                        raise
                    direct_oom = caught.with_traceback(None)
                if direct_oom is not None:
                    reference = _retry_tiled_vae_after_oom(
                        handle=handle,
                        value=content,
                        output_dtype=torch.float32,
                        direction="encode",
                        oom=direct_oom,
                    )
        latent_frames = ((length - 1) // 4) + 1
        expected_reference = (1, 16, latent_frames, height // 8, width // 8)
        if type(reference) is not torch.Tensor or tuple(reference.shape) != expected_reference:
            actual = getattr(reference, "shape", None)
            raise ValueError(
                f"Wan VAE reference latent has shape {actual}, expected {expected_reference}"
            )
        mask = torch.zeros(
            (1, 4, latent_frames, height // 8, width // 8),
            device=reference.device,
            dtype=reference.dtype,
        )
        mask[:, :, : ((frames - 1) // 4) + 1] = 1.0
        concat_latent = torch.cat((mask, reference), dim=1)
        latent = torch.zeros(
            (batch_size, 16, latent_frames, height // 8, width // 8),
            device="cpu",
            dtype=reference.dtype,
        )
        inference = importlib.import_module("dinkster_inference")
        prepared_positive = _wan21_i2v_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        prepared_negative = _wan21_i2v_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        return cls.outputs(
            positive=prepared_positive,
            negative=prepared_negative,
            latent={"samples": latent},
        )


_WAN_CAMERA_MOTIONS = {
    "Static": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "Pan Up": ((0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    "Pan Down": ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "Pan Left": ((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    "Pan Right": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "Zoom In": ((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
    "Zoom Out": ((0.0, 0.0, 0.0), (0.0, 0.0, -2.0)),
    "Anti Clockwise (ACW)": ((0.0, 0.0, -1.0), (0.0, 0.0, 0.0)),
    "ClockWise (CW)": ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
}


def _wan_camera_dimensions(*, width: int, height: int, length: int) -> None:
    if width < 16 or width > 16384 or width % 16 != 0:
        raise ValueError("width must be a multiple of 16 between 16 and 16384")
    if height < 16 or height > 16384 or height % 16 != 0:
        raise ValueError("height must be a multiple of 16 between 16 and 16384")
    if length < 1 or length > 16384 or (length - 1) % 4 != 0:
        raise ValueError("length must be 1 plus a multiple of 4 between 1 and 16384")


def _wan_camera_embedding(
    *,
    camera_pose: str,
    width: int,
    height: int,
    length: int,
    speed: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Any:
    _wan_camera_dimensions(width=width, height=height, length=length)
    if camera_pose not in WAN_CAMERA_POSES:
        raise ValueError("camera_pose is not a supported Wan camera trajectory")
    for name, value, minimum, maximum in (
        ("speed", speed, 0.0, 10.0),
        ("fx", fx, 0.0, 1.0),
        ("fy", fy, 0.0, 1.0),
        ("cx", cx, 0.0, 1.0),
        ("cy", cy, 0.0, 1.0),
    ):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")

    np = cast("Any", importlib.import_module("numpy"))
    torch = _torch()
    raw_angle, raw_translation = _WAN_CAMERA_MOTIONS[camera_pose]
    angle = np.array(raw_angle)
    translation = np.array(raw_translation).reshape(3, 1)
    extrinsics: list[Any] = []
    for index in range(length):
        theta_x, theta_y, theta_z = index / length * speed * (np.pi / 3) * angle
        rotation_x = np.array(
            (
                (1, 0, 0),
                (0, np.cos(theta_x), -np.sin(theta_x)),
                (0, np.sin(theta_x), np.cos(theta_x)),
            )
        )
        rotation_y = np.array(
            (
                (np.cos(theta_y), 0, np.sin(theta_y)),
                (0, 1, 0),
                (-np.sin(theta_y), 0, np.cos(theta_y)),
            )
        )
        rotation_z = np.array(
            (
                (np.cos(theta_z), -np.sin(theta_z), 0),
                (np.sin(theta_z), np.cos(theta_z), 0),
                (0, 0, 1),
            )
        )
        rotation = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
        offset = index / length * speed * 1.5 * translation
        extrinsics.append(np.concatenate((rotation, offset), axis=1))

    entries: list[list[float]] = []
    for extrinsic in cast("list[list[list[float]]]", np.stack(extrinsics).tolist()):
        entry: list[float] = [fx, fy, cx, cy, 0.0, 0.0]
        entry.extend(extrinsic[0])
        entry.extend(extrinsic[1])
        entry.extend(extrinsic[2])
        entry.extend((0.0, 0.0, 0.0, 1.0))
        entries.append(entry)
    camera_parameters = np.concatenate(
        (
            np.zeros((length, 1)),
            np.array([[float(value) for value in entry] for entry in entries]),
        ),
        axis=1,
    )
    focal_x = camera_parameters[:, 1].copy()
    focal_y = camera_parameters[:, 2].copy()
    sample_ratio = width / height
    pose_ratio = 1280 / 720
    if pose_ratio > sample_ratio:
        focal_x *= height * pose_ratio / width
    else:
        focal_y *= width / pose_ratio / height
    intrinsic = np.asarray(
        tuple(
            (
                focal_x[index] * width,
                focal_y[index] * height,
                camera_parameters[index, 3] * width,
                camera_parameters[index, 4] * height,
            )
            for index in range(length)
        ),
        dtype=np.float32,
    )
    camera_to_world = tuple(np.array(entry[7:]).reshape(4, 4) for entry in camera_parameters)
    world_to_camera = tuple(np.linalg.inv(matrix) for matrix in camera_to_world)
    target = np.array(
        (
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )
    absolute_to_relative = target @ world_to_camera[0]
    poses = np.array(
        (target, *(absolute_to_relative @ matrix for matrix in camera_to_world[1:])),
        dtype=np.float32,
    )
    intrinsics = torch.as_tensor(intrinsic)[None]
    c2w = torch.as_tensor(poses)[None]
    row, column = torch.meshgrid(
        torch.linspace(0, height - 1, height, dtype=c2w.dtype),
        torch.linspace(0, width - 1, width, dtype=c2w.dtype),
        indexing="ij",
    )
    column = column.reshape(1, 1, height * width).expand(1, 1, height * width) + 0.5
    row = row.reshape(1, 1, height * width).expand(1, 1, height * width) + 0.5
    focal_x_t, focal_y_t, center_x, center_y = intrinsics.chunk(4, dim=-1)
    depth = torch.ones_like(column)
    direction_x = (column - center_x) / focal_x_t * depth
    direction_y = (row - center_y) / focal_y_t * depth
    depth = depth.expand_as(direction_y)
    directions = torch.stack(
        (
            direction_x,
            direction_y,
            depth,
        ),
        dim=-1,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)
    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3][:, :, None].expand_as(rays_d)
    plucker = torch.cat((torch.cross(rays_o, rays_d, dim=-1), rays_d), dim=-1)
    embedding = plucker.reshape(1, length, height, width, 6).permute(0, 4, 1, 2, 3)
    embedding = torch.cat(
        (embedding[:, :, :1].repeat_interleave(4, dim=2), embedding[:, :, 1:]), dim=2
    )
    embedding = (
        embedding.transpose(1, 2)
        .reshape(1, embedding.shape[2] // 4, 4, 6, height, width)
        .transpose(2, 3)
        .reshape(1, embedding.shape[2] // 4, 24, height, width)
        .transpose(1, 2)
        .contiguous()
    )
    # intermediate_device() @ b78cec87 is cpu absent ComfyUI's --gpu-only
    # flag, which Dinkster does not wire.
    return embedding.to(device="cpu")


class NativeWanCameraEmbedding(WanCameraEmbedding):
    @classmethod
    def execute(
        cls,
        *,
        camera_pose: str,
        width: int,
        height: int,
        length: int,
        speed: float = 1.0,
        fx: float = 0.5,
        fy: float = 0.5,
        cx: float = 0.5,
        cy: float = 0.5,
    ) -> Mapping[str, object]:
        embedding = _wan_camera_embedding(
            camera_pose=camera_pose,
            width=width,
            height=height,
            length=length,
            speed=speed,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
        return cls.outputs(
            camera_embedding=embedding,
            width=width,
            height=height,
            length=length,
        )


def _wan_camera_prepared(
    value: object,
    *,
    name: str,
    handle: NativeRuntimeHandle,
    inference: Any,
    concat_latent: Any,
    camera_conditions: Any,
    vision: Any,
) -> list[list[object]]:
    text = _prepared_multistream_conditioning(
        value,
        inference,
        name,
        handle.runtime.conditioning_identity,
    )
    if text is None:
        raise TypeError(f"{name} must contain Wan text conditioning")
    prepared = handle.runtime.prepare_camera_conditioning(
        text,
        concat_latent,
        camera_conditions,
        vision,
    )
    return [
        [
            inference.PreparedMultiStreamConditioning(
                handle.runtime.conditioning_identity,
                prepared,
            ),
            {},
        ]
    ]


class NativeWanCameraImageToVideo(WanCameraImageToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        clip_vision_output: object = None,
        start_image: object = None,
        camera_conditions: object = None,
    ) -> Mapping[str, object]:
        _wan_camera_dimensions(width=width, height=height, length=length)
        if batch_size < 1 or batch_size > 4096:
            raise ValueError("batch_size must be between 1 and 4096")
        handle = _native_handle(vae, "vae")
        config = handle.runtime.assembled.diffusion.config
        if (
            config.camera_channels != 24
            or config.out_channels != 16
            or config.in_channels not in (32, 36)
        ):
            raise ValueError("vae must come from a supported Wan camera profile")
        torch = _torch()
        if camera_conditions is not None:
            if type(camera_conditions) is not torch.Tensor:
                raise TypeError("camera_conditions must come from Wan Camera Embedding")
            expected_camera = (1, 24, ((length - 1) // 4) + 1, height, width)
            if tuple(cast("Any", camera_conditions).shape) != expected_camera:
                raise ValueError(
                    f"camera_conditions must have shape {expected_camera}, got "
                    f"{tuple(cast('Any', camera_conditions).shape)}"
                )
        vision = None
        if config.model_type == "i2v":
            if clip_vision_output is not None:
                vision = _wan21_clip_embedding(
                    clip_vision_output,
                    name="clip_vision_output",
                    handle=handle,
                    torch=torch,
                )
        elif clip_vision_output is not None:
            raise ValueError("Wan 2.2 camera does not consume CLIP vision output")

        latent_frames = ((length - 1) // 4) + 1
        latent_shape = (1, 16, latent_frames, height // 8, width // 8)
        concat_latent = None
        if start_image is not None:
            with handle.stage("vae"):
                with torch.inference_mode():
                    reference = handle.runtime.assembled.vae.process_out(
                        torch.zeros(latent_shape, device=handle.load_device, dtype=torch.float32)
                    )
                    image = _wan_image(start_image, "start_image", torch)
                    resized = (
                        importlib.import_module("dinkster_inference_torch.resize")
                        .common_upscale(
                            image[:length, :, :, :3].movedim(-1, 1),
                            width,
                            height,
                            "bilinear",
                            "center",
                        )
                        .movedim(1, -1)
                    )
                    content = resized.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
                    encoded_frames = ((int(resized.shape[0]) - 1) // 4) + 1
                    encoded = _wan_encode_content(
                        handle=handle,
                        content=content,
                        expected=(1, 16, encoded_frames, *latent_shape[-2:]),
                        name="camera start image",
                        torch=torch,
                    )
                    reference[:, :, : encoded.shape[2]] = encoded[:, :, :latent_frames]
                    concat_latent = reference
                    if config.in_channels == 36:
                        external_mask = torch.ones(
                            (1, 1, latent_frames * 4, *latent_shape[-2:]),
                            device=reference.device,
                            dtype=reference.dtype,
                        )
                        external_mask[:, :, : int(resized.shape[0]) + 3] = 0.0
                        model_mask = _wan_flf_model_mask(external_mask, latent_frames)
                        concat_latent = torch.cat((model_mask, reference), dim=1)
        latent = torch.zeros(
            (batch_size, 16, *latent_shape[2:]),
            device="cpu",
            dtype=torch.float32,
        )
        inference = importlib.import_module("dinkster_inference")
        positive_prepared = _wan_camera_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            camera_conditions=camera_conditions,
            vision=vision,
        )
        negative_prepared = _wan_camera_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            camera_conditions=camera_conditions,
            vision=vision,
        )
        return cls.outputs(
            positive=positive_prepared,
            negative=negative_prepared,
            latent={"samples": latent},
        )


def _wan_phantom_prepared(
    value: object,
    *,
    name: str,
    handle: NativeRuntimeHandle,
    inference: Any,
    temporal_reference: Any,
) -> list[list[object]]:
    text = _prepared_multistream_conditioning(
        value,
        inference,
        name,
        handle.runtime.conditioning_identity,
    )
    if text is None:
        raise TypeError(f"{name} must contain Wan text conditioning")
    prepared = handle.runtime.prepare_phantom_conditioning(text, temporal_reference)
    return [
        [
            inference.PreparedMultiStreamConditioning(
                handle.runtime.conditioning_identity,
                prepared,
            ),
            {},
        ]
    ]


class NativeWanPhantomSubjectToVideo(WanPhantomSubjectToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        images: object = None,
    ) -> Mapping[str, object]:
        _wan_fun_dimensions(
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
        )
        handle = _native_handle(vae, "vae")
        config = handle.runtime.assembled.diffusion.config
        if (
            config.model_type != "t2v"
            or config.in_channels != 16
            or config.out_channels != 16
            or config.reference_channels is not None
            or config.vace_layers is not None
            or config.camera_channels is not None
        ):
            raise ValueError("vae must come from a supported Wan Phantom profile")

        torch = _torch()
        temporal_reference = None
        empty_reference = None
        if images is not None:
            image = _wan_image(images, "images", torch)
            resized = (
                importlib.import_module("dinkster_inference_torch.resize")
                .common_upscale(
                    image[:length, :, :, :3].movedim(-1, 1),
                    width,
                    height,
                    "bilinear",
                    "center",
                )
                .movedim(1, -1)
            )
            references: list[Any] = []
            with handle.stage("vae", unload_before=("text",)):
                with torch.inference_mode():
                    for index, frame in enumerate(resized):
                        content = (
                            frame.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(handle.load_device)
                        )
                        references.append(
                            _wan_encode_content(
                                handle=handle,
                                content=content,
                                expected=(1, 16, 1, height // 8, width // 8),
                                name=f"Phantom reference {index}",
                                torch=torch,
                            )
                        )
                    temporal_reference = torch.cat(references, dim=2)
                    empty_reference = handle.runtime.assembled.vae.process_out(
                        torch.zeros_like(temporal_reference)
                    )

        inference = importlib.import_module("dinkster_inference")
        prepared_positive = _wan_phantom_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            temporal_reference=temporal_reference,
        )
        prepared_negative_text = _wan_phantom_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            temporal_reference=temporal_reference,
        )
        prepared_negative_img_text = _wan_phantom_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            temporal_reference=empty_reference,
        )
        latent = torch.zeros(
            (batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8),
            device="cpu",
            dtype=(torch.float32 if temporal_reference is None else temporal_reference.dtype),
        )
        return cls.outputs(
            positive=prepared_positive,
            negative_text=prepared_negative_text,
            negative_img_text=prepared_negative_img_text,
            latent={"samples": latent},
        )


def _wan_image(value: object, name: str, torch: Any) -> Any:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{name} must be an exact torch.Tensor")
    image = cast("Any", value)
    if (
        image.ndim != 4
        or image.shape[0] <= 0
        or image.shape[1] <= 0
        or image.shape[2] <= 0
        or image.shape[3] < 3
        or not image.is_floating_point()
        or image.layout != torch.strided
    ):
        raise ValueError(f"{name} must be a nonempty strided floating [frames,H,W,C>=3] tensor")
    return image


def _wan_flf_model_mask(mask: Any, latent_frames: int) -> Any:
    return 1.0 - mask.reshape(1, latent_frames, 4, *mask.shape[-2:]).transpose(1, 2)


def _wan_ati_model_mask(mask: Any) -> Any:
    # Preserve ComfyUI's two complement operations; combining them changes float values.
    external_mask = -mask + 1.0
    return 1.0 - external_mask


class NativeWanTrackToVideo(WanTrackToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        tracks: str,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        temperature: float,
        topk: int,
        start_image: object,
        clip_vision_output: object = None,
    ) -> Mapping[str, object]:
        _wan_fun_dimensions(
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
        )
        if (
            type(temperature) not in (int, float)
            or not math.isfinite(temperature)
            or not 1.0 <= temperature <= 1000.0
        ):
            raise ValueError("temperature must be finite and between 1 and 1000")
        if type(topk) is not int or not 1 <= topk <= 10:
            raise ValueError("topk must be between 1 and 10")
        handle = _native_handle(vae, "vae")
        assembled = handle.runtime.assembled
        config = assembled.diffusion.config
        if config.model_type != "i2v" or config.in_channels != 36 or config.out_channels != 16:
            raise ValueError("vae must come from a supported Wan 2.1 ATI profile")
        if assembled.clip_vision is None:
            raise ValueError("loaded Wan 2.1 ATI model has no CLIP vision encoder")

        empty_tracks = False
        if type(tracks) is str:
            try:
                empty_tracks = not json.loads(tracks.replace("'", '"'))
            except json.JSONDecodeError:
                empty_tracks = True
        if empty_tracks:
            return NativeWan21ImageToVideo.execute(
                positive=positive,
                negative=negative,
                model=vae,
                start_image=start_image,
                clip_vision_output=clip_vision_output,
                width=width,
                height=height,
                length=length,
                batch_size=batch_size,
            )

        torch = _torch()
        inference_torch = importlib.import_module("dinkster_inference_torch")
        prepared_tracks = inference_torch.prepare_wan_ati_tracks(
            tracks,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
        )
        image = _wan_image(start_image, "start_image", torch)
        resized = (
            importlib.import_module("dinkster_inference_torch.resize")
            .common_upscale(
                image[:batch_size, :, :, :3].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            )
            .movedim(1, -1)
        )
        videos = torch.full(
            (resized.shape[0], length, height, width, 3),
            0.5,
            device=resized.device,
            dtype=resized.dtype,
        )
        videos[:, 0] = resized
        videos = importlib.import_module("dinkster_inference_torch.resize").resize_to_batch_size(
            videos, batch_size
        )
        latent_frames = ((length - 1) // 4) + 1
        encoded_videos: list[Any] = []
        with handle.stage("vae"):
            with torch.inference_mode():
                for index in range(batch_size):
                    content = videos[index].permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
                    direct_oom: BaseException | None = None
                    encoded: Any = None
                    try:
                        encoded = handle.runtime.encode_content(content)
                    except RuntimeError as caught:
                        if not _is_accelerator_oom(caught, torch=torch, device=handle.load_device):
                            raise
                        direct_oom = caught.with_traceback(None)
                    if direct_oom is not None:
                        encoded = _retry_tiled_vae_after_oom(
                            handle=handle,
                            value=content,
                            output_dtype=torch.float32,
                            direction="encode",
                            oom=direct_oom,
                        )
                    expected = (1, 16, latent_frames, height // 8, width // 8)
                    if type(encoded) is not torch.Tensor or tuple(encoded.shape) != expected:
                        actual = getattr(encoded, "shape", None)
                        raise ValueError(
                            f"Wan ATI video latent has shape {actual}, expected {expected}"
                        )
                    encoded_videos.append(encoded.to("cpu"))
        external_video = torch.cat(tuple(encoded_videos), dim=0)
        model_video = assembled.vae.process_in(external_video)
        ati_mask, model_motion = inference_torch.patch_wan_ati_motion(
            prepared_tracks,
            model_video,
            temperature=float(temperature),
            topk=topk,
        )
        concat_latent = torch.cat(
            (_wan_ati_model_mask(ati_mask), assembled.vae.process_out(model_motion)),
            dim=1,
        )
        vision = None
        if clip_vision_output is not None:
            vision = _wan21_clip_embedding(
                clip_vision_output,
                name="clip_vision_output",
                handle=handle,
                torch=torch,
            )
        inference = importlib.import_module("dinkster_inference")
        prepared_positive = _wan21_i2v_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        prepared_negative = _wan21_i2v_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        latent = torch.zeros(
            (batch_size, 16, latent_frames, height // 8, width // 8),
            device="cpu",
            dtype=concat_latent.dtype,
        )
        return cls.outputs(
            positive=prepared_positive,
            negative=prepared_negative,
            latent={"samples": latent},
        )


def _wan_move_tracks(
    value: object,
    name: str,
    torch: Any,
    *,
    require_visibility: bool,
) -> tuple[Any, Any | None]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a TRACKS mapping")
    mapping = cast("Mapping[object, object]", value)
    raw_track_path = mapping.get("track_path")
    if type(raw_track_path) is not torch.Tensor:
        raise TypeError(f"{name} track_path must be an exact torch.Tensor")
    track_path = cast("Any", raw_track_path)
    if (
        track_path.ndim != 3
        or track_path.shape[0] <= 0
        or track_path.shape[1] <= 0
        or track_path.shape[2] != 2
        or not track_path.is_floating_point()
        or track_path.layout != torch.strided
    ):
        raise ValueError(f"{name} track_path must be a nonempty strided floating [T,N,2] tensor")
    raw_visibility = mapping.get("track_visibility")
    if raw_visibility is None and not require_visibility:
        return track_path, None
    if type(raw_visibility) is not torch.Tensor:
        raise TypeError(f"{name} track_visibility must be an exact torch.Tensor")
    visibility = cast("Any", raw_visibility)
    if (
        visibility.ndim != 2
        or visibility.shape[0] != track_path.shape[0]
        or visibility.shape[1] not in (1, track_path.shape[1])
        or visibility.layout != torch.strided
    ):
        raise ValueError(f"{name} track_visibility must be a strided [T,N] tensor")
    return track_path, visibility


def _wan_move_visibility(
    track_mask: object,
    *,
    track_length: int,
    num_tracks: int,
    torch: Any,
) -> Any:
    if track_mask is None:
        return torch.ones((track_length, num_tracks), dtype=torch.bool)
    if type(track_mask) is not torch.Tensor:
        raise TypeError("track_mask must be an exact torch.Tensor")
    mask = cast("Any", track_mask)
    if mask.ndim != 3 or mask.shape[0] != track_length or mask.layout != torch.strided:
        raise ValueError("track_mask must be a strided [T,H,W] tensor matching the track length")
    return (mask > 0).any(dim=(1, 2)).unsqueeze(-1)


def _wan_move_parse_coords(value: object) -> list[list[Mapping[str, Any]]]:
    if type(value) is not str:
        raise TypeError("track_coords must be a string")
    try:
        parsed = json.loads(value.replace("'", '"'))
    except json.JSONDecodeError as error:
        raise ValueError("track_coords must contain valid JSON tracks") from error
    if type(parsed) is not list or not parsed:
        raise ValueError("track_coords must contain at least one track")
    tracks = cast("list[Any]", parsed)
    first = tracks[0]
    if isinstance(first, Mapping) and "x" in first:
        tracks = [tracks]
        first = tracks[0]
    first_track = cast("list[Any]", first) if isinstance(first, list) else []
    if (
        not tracks
        or not isinstance(first, list)
        or not first_track
        or not isinstance(first_track[0], Mapping)
        or "x" not in first_track[0]
    ):
        raise ValueError("track_coords must be a track or list of tracks with x/y points")
    return cast("list[list[Mapping[str, Any]]]", tracks)


class NativeWanMoveTracksFromCoords(WanMoveTracksFromCoords):
    @classmethod
    def execute(
        cls,
        *,
        track_coords: str = "[]",
        track_mask: object = None,
    ) -> Mapping[str, object]:
        torch = _torch()
        tracks_data = _wan_move_parse_coords(track_coords)
        track_length = len(tracks_data[0])
        track_list: list[list[list[float]]] = [
            [[float(track[frame]["x"]), float(track[frame]["y"])] for track in tracks_data]
            for frame in range(track_length)
        ]
        tracks = torch.tensor(track_list, dtype=torch.float32)
        visibility = _wan_move_visibility(
            track_mask,
            track_length=track_length,
            num_tracks=int(tracks.shape[1]),
            torch=torch,
        )
        return cls.outputs(
            tracks={"track_path": tracks, "track_visibility": visibility},
            track_length=track_length,
        )


class NativeWanMoveConcatTrack(WanMoveConcatTrack):
    @classmethod
    def execute(
        cls,
        *,
        tracks_1: object,
        tracks_2: object = None,
    ) -> Mapping[str, object]:
        torch = _torch()
        path_1, visibility_1 = _wan_move_tracks(
            tracks_1, "tracks_1", torch, require_visibility=True
        )
        if tracks_2 is None:
            return cls.outputs(tracks=tracks_1)
        path_2, visibility_2 = _wan_move_tracks(
            tracks_2, "tracks_2", torch, require_visibility=True
        )
        assert visibility_1 is not None and visibility_2 is not None
        return cls.outputs(
            tracks={
                "track_path": torch.cat((path_1, path_2), dim=1),
                "track_visibility": torch.cat((visibility_1, visibility_2), dim=-1),
            }
        )


class NativeWanMoveGenerateTracks(WanMoveGenerateTracks):
    @classmethod
    def execute(
        cls,
        *,
        width: int,
        height: int,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        num_frames: int,
        num_tracks: int,
        track_spread: float,
        bezier: bool = False,
        mid_x: float = 0.5,
        mid_y: float = 0.5,
        interpolation: str = "linear",
        track_mask: object = None,
    ) -> Mapping[str, object]:
        if type(width) is not int or not 16 <= width <= 4096:
            raise ValueError("width must be between 16 and 4096")
        if type(height) is not int or not 16 <= height <= 4096:
            raise ValueError("height must be between 16 and 4096")
        if type(num_frames) is not int or not 1 <= num_frames <= 1024:
            raise ValueError("num_frames must be between 1 and 1024")
        if type(num_tracks) is not int or not 1 <= num_tracks <= 100:
            raise ValueError("num_tracks must be between 1 and 100")
        points = (start_x, start_y, mid_x, mid_y, end_x, end_y)
        if any(type(value) not in (int, float) or not 0.0 <= value <= 1.0 for value in points):
            raise ValueError("track coordinates must be between 0 and 1")
        if (
            type(track_spread) not in (int, float)
            or not math.isfinite(track_spread)
            or not 0.0 <= track_spread <= 1.0
        ):
            raise ValueError("track_spread must be finite and between 0 and 1")
        if type(bezier) is not bool:
            raise TypeError("bezier must be a bool")
        if interpolation not in ("linear", "ease_in", "ease_out", "ease_in_out", "constant"):
            raise ValueError("unknown track interpolation")

        torch = _torch()
        start_x_px, start_y_px = start_x * width, start_y * height
        mid_x_px, mid_y_px = mid_x * width, mid_y * height
        end_x_px, end_y_px = end_x * width, end_y * height
        track_spread_px = track_spread * (width + height) / 2
        t = torch.linspace(0, 1, num_frames)
        if interpolation == "constant":
            interp_values = torch.zeros_like(t)
        elif interpolation == "linear":
            interp_values = t
        elif interpolation == "ease_in":
            interp_values = t**2
        elif interpolation == "ease_out":
            interp_values = 1 - (1 - t) ** 2
        else:
            interp_values = t * t * (3 - 2 * t)

        if bezier:
            t_interp = interp_values
            one_minus_t = 1 - t_interp
            x_positions = (
                one_minus_t**2 * start_x_px
                + 2 * one_minus_t * t_interp * mid_x_px
                + t_interp**2 * end_x_px
            )
            y_positions = (
                one_minus_t**2 * start_y_px
                + 2 * one_minus_t * t_interp * mid_y_px
                + t_interp**2 * end_y_px
            )
            tangent_x = 2 * one_minus_t * (mid_x_px - start_x_px) + 2 * t_interp * (
                end_x_px - mid_x_px
            )
            tangent_y = 2 * one_minus_t * (mid_y_px - start_y_px) + 2 * t_interp * (
                end_y_px - mid_y_px
            )
        else:
            x_positions = start_x_px + (end_x_px - start_x_px) * interp_values
            y_positions = start_y_px + (end_y_px - start_y_px) * interp_values
            tangent_x = torch.full_like(t, end_x_px - start_x_px)
            tangent_y = torch.full_like(t, end_y_px - start_y_px)

        track_list: list[list[list[float]]] = []
        for frame_idx in range(num_frames):
            tx = tangent_x[frame_idx].item()
            ty = tangent_y[frame_idx].item()
            length = (tx**2 + ty**2) ** 0.5
            if length > 0:
                perp_x, perp_y = -ty / length, tx / length
            else:
                perp_x, perp_y = 1.0, 0.0
            frame_tracks: list[list[float]] = []
            for track_idx in range(num_tracks):
                offset = (track_idx - (num_tracks - 1) / 2) * track_spread_px
                frame_tracks.append(
                    [
                        float(x_positions[frame_idx].item() + perp_x * offset),
                        float(y_positions[frame_idx].item() + perp_y * offset),
                    ]
                )
            track_list.append(frame_tracks)

        tracks = torch.tensor(track_list, dtype=torch.float32)
        visibility = _wan_move_visibility(
            track_mask,
            track_length=num_frames,
            num_tracks=num_tracks,
            torch=torch,
        )
        return cls.outputs(
            tracks={"track_path": tracks, "track_visibility": visibility},
            track_length=num_frames,
        )


def _wan_move_draw_gradient_polyline(
    overlay: Any,
    line_width: int,
    points: Any,
    color: tuple[int, int, int],
    opacity: float,
    image_draw: Any,
) -> None:
    draw = image_draw.Draw(overlay, "RGBA")
    points = points[::-1]
    segment_lengths: list[float] = []
    total_length = 0.0
    for index in range(len(points) - 1):
        dx = float(points[index + 1][0] - points[index][0])
        dy = float(points[index + 1][1] - points[index][1])
        length = (dx * dx + dy * dy) ** 0.5
        segment_lengths.append(length)
        total_length += length
    if total_length == 0:
        return
    accumulated_length = 0.0
    for index, (start_point, end_point) in enumerate(zip(points[:-1], points[1:], strict=True)):
        segment_length = segment_lengths[index]
        steps = max(int(segment_length), 1)
        for step in range(steps):
            current_length = accumulated_length + (step / steps) * segment_length
            ratio = current_length / total_length
            alpha = int(255 * (1 - ratio) * opacity)
            x = int(start_point[0] + (end_point[0] - start_point[0]) * step / steps)
            y = int(start_point[1] + (end_point[1] - start_point[1]) * step / steps)
            dynamic_width = max(int(line_width * (1 - ratio)), 1)
            draw.line([(x, y), (x + 1, y)], fill=(*color, alpha), width=dynamic_width)
        accumulated_length += segment_length


def _wan_move_draw_tracks(
    video: Any,
    tracks: Any,
    visibility: Any,
    *,
    track_frame: int,
    circle_size: int,
    opacity: float,
    line_width: int,
) -> list[Any]:
    np = cast("Any", importlib.import_module("numpy"))
    image = cast("Any", importlib.import_module("PIL.Image"))
    image_draw = cast("Any", importlib.import_module("PIL.ImageDraw"))
    colors: tuple[tuple[int, int, int], ...] = (
        (102, 153, 255),
        (0, 255, 255),
        (255, 255, 0),
        (255, 102, 204),
        (0, 255, 0),
    )
    video_np = video.byte().cpu().numpy()
    tracks_np = tracks[0].long().detach().cpu().numpy()
    visibility_np = visibility[0].detach().cpu().numpy()
    num_frames, height, width = video_np.shape[:3]
    num_tracks = tracks_np.shape[1]
    alpha_opacity = int(255 * opacity)
    output_frames: list[Any] = []
    for frame_index in range(num_frames):
        frame_rgb = video_np[frame_index].astype(np.float32)
        overlay = image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw_overlay = image_draw.Draw(overlay)
        polylines: list[tuple[Any, tuple[int, int, int]]] = []
        for track_index in range(num_tracks):
            if visibility_np[frame_index, track_index] == 0:
                continue
            coordinate = tracks_np[frame_index, track_index]
            color = colors[track_index % len(colors)]
            draw_overlay.ellipse(
                (
                    coordinate[0] - circle_size,
                    coordinate[1] - circle_size,
                    coordinate[0] + circle_size,
                    coordinate[1] + circle_size,
                ),
                fill=color + (alpha_opacity,),
            )
            track_coordinates = tracks_np[
                max(frame_index - track_frame, 0) : frame_index + 1, track_index
            ]
            if len(track_coordinates) > 1:
                polylines.append((track_coordinates, color))
        overlay_np = np.array(overlay)
        alpha = overlay_np[:, :, 3:4] / 255.0
        frame_rgb = overlay_np[:, :, :3] * alpha + frame_rgb * (1 - alpha)
        if polylines:
            polyline_overlay = image.new("RGBA", (width, height), (0, 0, 0, 0))
            for track_coordinates, color in polylines:
                _wan_move_draw_gradient_polyline(
                    polyline_overlay,
                    line_width,
                    track_coordinates,
                    color,
                    opacity,
                    image_draw,
                )
            polyline_np = np.array(polyline_overlay)
            alpha = polyline_np[:, :, 3:4] / 255.0
            frame_rgb = polyline_np[:, :, :3] * alpha + frame_rgb * (1 - alpha)
        output_frames.append(image.fromarray(frame_rgb.astype(np.uint8)))
    return output_frames


class NativeWanMoveVisualizeTracks(WanMoveVisualizeTracks):
    @classmethod
    def execute(
        cls,
        *,
        images: object,
        line_resolution: int,
        circle_size: int,
        opacity: float,
        line_width: int,
        tracks: object = None,
    ) -> Mapping[str, object]:
        if tracks is None:
            return cls.outputs(images=images)
        torch = _torch()
        image_tensor = _wan_image(images, "images", torch)
        if image_tensor.shape[-1] != 3:
            raise ValueError("images must have exactly three channels")
        path, visibility = _wan_move_tracks(tracks, "tracks", torch, require_visibility=True)
        assert visibility is not None
        images_in = image_tensor * 255.0
        if images_in.shape[0] != path.shape[0]:
            repeat_count = path.shape[0] // images_in.shape[0]
            images_in = images_in.repeat(repeat_count, 1, 1, 1)
        frames = _wan_move_draw_tracks(
            images_in,
            path.unsqueeze(0),
            visibility.unsqueeze(0),
            track_frame=line_resolution,
            circle_size=circle_size,
            opacity=opacity,
            line_width=line_width,
        )
        np = cast("Any", importlib.import_module("numpy"))
        output = torch.from_numpy(np.stack([np.asarray(frame) for frame in frames])).float() / 255.0
        return cls.outputs(images=output)


def _wan_move_positions(
    tracks: Any,
    visibility: Any,
    *,
    height: int,
    width: int,
    torch: Any,
) -> Any:
    frame_count, track_count, _ = tracks.shape
    positions = -torch.ones(
        track_count,
        (frame_count - 1) // 4 + 1,
        2,
        dtype=torch.long,
    )
    selected = torch.randperm(track_count)[:track_count]
    tracks = tracks[:, selected]
    visibility = visibility[:, selected]
    for frame_index in range(0, frame_count, 4):
        current_tracks = tracks[frame_index]
        current_visibility = visibility[frame_index]
        for track_index in range(track_count):
            if (
                not current_visibility[track_index]
                or current_tracks[track_index][0] < 0
                or current_tracks[track_index][1] < 0
                or current_tracks[track_index][0] >= width
                or current_tracks[track_index][1] >= height
            ):
                continue
            x, y = current_tracks[track_index]
            positions[track_index, frame_index // 4, 0] = int(y // 8)
            positions[track_index, frame_index // 4, 1] = int(x // 8)
    return positions


def _wan_move_replace_feature(
    vae_feature: Any,
    positions: Any,
    strength: float,
    torch: Any,
) -> Any:
    batch, _, _, _, _ = vae_feature.shape
    if batch != positions.shape[0]:
        raise ValueError("WanMove track and VAE feature batch sizes must match")
    track_count = positions.shape[1]
    positions = positions[:, torch.randperm(track_count)]
    current = positions[:, :, 1:, :]
    valid = (current[..., 0] >= 0) & (current[..., 1] >= 0)
    indices = valid.nonzero(as_tuple=False)
    if indices.shape[0] == 0:
        return vae_feature
    batch_index = indices[:, 0]
    track_index = indices[:, 1]
    relative_time = indices[:, 2]
    target_time = relative_time + 1
    target_height = current[batch_index, track_index, relative_time, 0].long()
    target_width = current[batch_index, track_index, relative_time, 1].long()
    source_height = positions[batch_index, track_index, 0, 0].long()
    source_width = positions[batch_index, track_index, 0, 1].long()
    source_features = vae_feature[batch_index, :, 0, source_height, source_width]
    destination_features = vae_feature[batch_index, :, target_time, target_height, target_width]
    vae_feature[batch_index, :, target_time, target_height, target_width] = (
        destination_features + (source_features - destination_features) * strength
    )
    return vae_feature


class NativeWanMoveTrackToVideo(WanMoveTrackToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        strength: float,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        start_image: object,
        tracks: object = None,
        clip_vision_output: object = None,
    ) -> Mapping[str, object]:
        _wan_fun_dimensions(
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
        )
        if (
            type(strength) not in (int, float)
            or not math.isfinite(strength)
            or not 0.0 <= strength <= 100.0
        ):
            raise ValueError("strength must be finite and between 0 and 100")
        handle = _native_handle(vae, "vae")
        assembled = handle.runtime.assembled
        config = assembled.diffusion.config
        if config.model_type != "i2v" or config.in_channels != 36 or config.out_channels != 16:
            raise ValueError("vae must come from a supported Wan 2.1 I2V profile")
        if assembled.clip_vision is None:
            raise ValueError("loaded Wan 2.1 I2V model has no CLIP vision encoder")

        torch = _torch()
        image = _wan_image(start_image, "start_image", torch)
        resized = (
            importlib.import_module("dinkster_inference_torch.resize")
            .common_upscale(
                image[:length].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            )
            .movedim(1, -1)
        )
        frames = min(int(resized.shape[0]), length)
        padded = torch.full(
            (length, height, width, resized.shape[-1]),
            0.5,
            device=resized.device,
            dtype=resized.dtype,
        )
        padded[:frames] = resized[:frames]
        content = padded[:, :, :, :3].permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
        with handle.stage("vae"):
            with torch.inference_mode():
                direct_oom: BaseException | None = None
                reference: Any = None
                try:
                    reference = handle.runtime.encode_content(content)
                except RuntimeError as caught:
                    if not _is_accelerator_oom(caught, torch=torch, device=handle.load_device):
                        raise
                    direct_oom = caught.with_traceback(None)
                if direct_oom is not None:
                    reference = _retry_tiled_vae_after_oom(
                        handle=handle,
                        value=content,
                        output_dtype=torch.float32,
                        direction="encode",
                        oom=direct_oom,
                    )
        latent_frames = ((length - 1) // 4) + 1
        expected = (1, 16, latent_frames, height // 8, width // 8)
        if type(reference) is not torch.Tensor or tuple(reference.shape) != expected:
            actual = getattr(reference, "shape", None)
            raise ValueError(f"WanMove VAE latent has shape {actual}, expected {expected}")
        reference = reference.to("cpu")
        known_mask = torch.zeros(
            (1, 4, latent_frames, height // 8, width // 8),
            device=reference.device,
            dtype=reference.dtype,
        )
        known_mask[:, :, : ((frames - 1) // 4) + 1] = 1.0
        if tracks is not None and strength > 0.0:
            track_path, visibility = _wan_move_tracks(
                tracks, "tracks", torch, require_visibility=False
            )
            track_path = track_path[:length].to("cpu")
            track_count = int(track_path.shape[1])
            if visibility is None:
                visibility = torch.ones((length, track_count), dtype=torch.bool)
            else:
                visibility = visibility[:length].to("cpu")
            positions = _wan_move_positions(
                track_path,
                visibility,
                height=height,
                width=width,
                torch=torch,
            )
            positions = importlib.import_module(
                "dinkster_inference_torch.resize"
            ).resize_to_batch_size(positions.unsqueeze(0), batch_size)
            with torch.inference_mode():
                reference = _wan_move_replace_feature(reference, positions, float(strength), torch)
        concat_latent = torch.cat((known_mask, reference), dim=1)

        vision = None
        if clip_vision_output is not None:
            vision = _wan21_clip_embedding(
                clip_vision_output,
                name="clip_vision_output",
                handle=handle,
                torch=torch,
            )
        inference = importlib.import_module("dinkster_inference")
        prepared_positive = _wan21_i2v_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        prepared_negative = _wan21_i2v_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        latent = torch.zeros(
            (batch_size, 16, latent_frames, height // 8, width // 8),
            device="cpu",
            dtype=reference.dtype,
        )
        return cls.outputs(
            positive=prepared_positive,
            negative=prepared_negative,
            latent={"samples": latent},
        )


class NativeWanFirstLastFrameToVideo(WanFirstLastFrameToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        clip_vision_start_image: object = None,
        clip_vision_end_image: object = None,
        start_image: object = None,
        end_image: object = None,
    ) -> Mapping[str, object]:
        if width < 16 or width > 16384 or width % 16 != 0:
            raise ValueError("width must be a multiple of 16 between 16 and 16384")
        if height < 16 or height > 16384 or height % 16 != 0:
            raise ValueError("height must be a multiple of 16 between 16 and 16384")
        if length < 1 or length > 16384 or (length - 1) % 4 != 0:
            raise ValueError("length must be 1 plus a multiple of 4 between 1 and 16384")
        if batch_size < 1 or batch_size > 4096:
            raise ValueError("batch_size must be between 1 and 4096")
        handle = _native_handle(vae, "vae")
        assembled = handle.runtime.assembled
        config = assembled.diffusion.config
        if config.in_channels != 36 or config.out_channels != 16:
            raise ValueError("vae must come from a supported Wan first/last-frame profile")
        wan21_i2v = config.model_type == "i2v"
        if not wan21_i2v and config.model_type != "t2v":
            raise ValueError("vae must come from a supported Wan first/last-frame profile")
        if wan21_i2v and config.flf_pos_embed_token_number != 514:
            raise ValueError("Wan 2.1 first/last-frame conditioning requires the FLF profile")
        if wan21_i2v and assembled.clip_vision is None:
            raise ValueError("loaded Wan 2.1 I2V model has no CLIP vision encoder")

        torch = _torch()
        vision = None
        if wan21_i2v:
            vision_rows = tuple(
                _wan21_clip_embedding(value, name=name, handle=handle, torch=torch)
                for name, value in (
                    ("clip_vision_start_image", clip_vision_start_image),
                    ("clip_vision_end_image", clip_vision_end_image),
                )
                if value is not None
            )
            if not vision_rows:
                raise ValueError("Wan 2.1 first/last-frame conditioning requires CLIP vision")
            vision = vision_rows[0] if len(vision_rows) == 1 else torch.cat(vision_rows, dim=-2)
        elif clip_vision_start_image is not None or clip_vision_end_image is not None:
            raise ValueError("Wan 2.2 FLF does not consume CLIP vision output")

        common_upscale = importlib.import_module("dinkster_inference_torch.resize").common_upscale
        resized_start = None
        if start_image is not None:
            image = _wan_image(start_image, "start_image", torch)
            resized_start = common_upscale(
                image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        resized_end = None
        if end_image is not None:
            image = _wan_image(end_image, "end_image", torch)
            resized_end = common_upscale(
                image[-length:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

        image = torch.full(
            (length, height, width, 3),
            0.5,
            device="cpu",
            dtype=torch.float32,
        )
        if resized_start is not None:
            image[: resized_start.shape[0]] = resized_start
        if resized_end is not None:
            image[-resized_end.shape[0] :] = resized_end
        content = image.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
        with handle.stage("vae"):
            with torch.inference_mode():
                direct_oom: BaseException | None = None
                reference: Any = None
                try:
                    reference = handle.runtime.encode_content(content)
                except RuntimeError as caught:
                    if not _is_accelerator_oom(caught, torch=torch, device=handle.load_device):
                        raise
                    direct_oom = caught.with_traceback(None)
                if direct_oom is not None:
                    reference = _retry_tiled_vae_after_oom(
                        handle=handle,
                        value=content,
                        output_dtype=torch.float32,
                        direction="encode",
                        oom=direct_oom,
                    )
        latent_frames = ((length - 1) // 4) + 1
        expected_reference = (1, 16, latent_frames, height // 8, width // 8)
        if type(reference) is not torch.Tensor or tuple(reference.shape) != expected_reference:
            actual = getattr(reference, "shape", None)
            raise ValueError(
                f"Wan VAE reference latent has shape {actual}, expected {expected_reference}"
            )
        mask = torch.ones(
            (1, 1, latent_frames * 4, height // 8, width // 8),
            device=reference.device,
            dtype=reference.dtype,
        )
        if resized_start is not None:
            mask[:, :, : resized_start.shape[0] + 3] = 0.0
        if resized_end is not None:
            mask[:, :, -resized_end.shape[0] :] = 0.0
        mask = _wan_flf_model_mask(mask, latent_frames)
        concat_latent = torch.cat((mask, reference), dim=1)

        inference = importlib.import_module("dinkster_inference")
        prepared_positive = _wan21_i2v_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        prepared_negative = _wan21_i2v_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        latent = torch.zeros(
            (batch_size, 16, latent_frames, height // 8, width // 8),
            device="cpu",
            dtype=reference.dtype,
        )
        return cls.outputs(
            positive=prepared_positive,
            negative=prepared_negative,
            latent={"samples": latent},
        )


def _wan_fun_prepared(
    value: object,
    *,
    name: str,
    handle: NativeRuntimeHandle,
    inference: Any,
    concat_latent: Any,
    concat_mask_index: int | None,
    vision: Any = None,
    reference_latent: Any = None,
) -> list[list[object]]:
    text = _prepared_multistream_conditioning(
        value,
        inference,
        name,
        handle.runtime.conditioning_identity,
    )
    if text is None:
        raise TypeError(f"{name} must contain Wan text conditioning")
    prepared = handle.runtime.prepare_fun_conditioning(
        text,
        concat_latent,
        concat_mask_index=concat_mask_index,
        vision=vision,
        reference_latent=reference_latent,
    )
    return [
        [
            inference.PreparedMultiStreamConditioning(
                handle.runtime.conditioning_identity,
                prepared,
            ),
            {},
        ]
    ]


def _wan_fun_dimensions(*, width: int, height: int, length: int, batch_size: int) -> None:
    if width < 16 or width > 16384 or width % 16 != 0:
        raise ValueError("width must be a multiple of 16 between 16 and 16384")
    if height < 16 or height > 16384 or height % 16 != 0:
        raise ValueError("height must be a multiple of 16 between 16 and 16384")
    if length < 1 or length > 16384 or (length - 1) % 4 != 0:
        raise ValueError("length must be 1 plus a multiple of 4 between 1 and 16384")
    if batch_size < 1 or batch_size > 4096:
        raise ValueError("batch_size must be between 1 and 4096")


def _wan_encode_content(
    *,
    handle: NativeRuntimeHandle,
    content: Any,
    expected: tuple[int, ...],
    name: str,
    torch: Any,
) -> Any:
    direct_oom: BaseException | None = None
    encoded: Any = None
    try:
        encoded = handle.runtime.encode_content(content)
    except RuntimeError as caught:
        if not _is_accelerator_oom(caught, torch=torch, device=handle.load_device):
            raise
        direct_oom = caught.with_traceback(None)
    if direct_oom is not None:
        encoded = _retry_tiled_vae_after_oom(
            handle=handle,
            value=content,
            output_dtype=torch.float32,
            direction="encode",
            oom=direct_oom,
        )
    if type(encoded) is not torch.Tensor or tuple(encoded.shape) != expected:
        actual = getattr(encoded, "shape", None)
        raise ValueError(f"Wan {name} latent has shape {actual}, expected {expected}")
    return encoded


def _execute_wan_fun_control(
    cls: type[WanFunControlToVideo] | type[Wan22FunControlToVideo],
    *,
    positive: object,
    negative: object,
    vae: object,
    width: int,
    height: int,
    length: int,
    batch_size: int,
    wan22: bool,
    clip_vision_output: object = None,
    ref_image: object = None,
    start_image: object = None,
    control_video: object = None,
) -> Mapping[str, object]:
    _wan_fun_dimensions(
        width=width,
        height=height,
        length=length,
        batch_size=batch_size,
    )
    handle = _native_handle(vae, "vae")
    config = handle.runtime.assembled.diffusion.config
    channels = config.out_channels
    expected_extra = channels * 2 + (4 if wan22 else 0)
    if (
        config.in_channels - channels != expected_extra
        or (wan22 and config.reference_channels != channels)
        or (not wan22 and config.reference_channels is not None)
    ):
        version = "Wan 2.2" if wan22 else "Wan 2.1"
        raise ValueError(f"vae must come from a {version} Fun control profile")
    torch = _torch()
    vision = None
    if wan22:
        if clip_vision_output is not None:
            raise ValueError("Wan 2.2 Fun control does not consume CLIP vision output")
    else:
        vision = _wan21_clip_embedding(
            clip_vision_output,
            name="clip_vision_output",
            handle=handle,
            torch=torch,
        )
    spatial_scale = 16 if channels == 48 else 8
    latent_frames = ((length - 1) // 4) + 1
    latent_shape = (1, channels, latent_frames, height // spatial_scale, width // spatial_scale)
    common_upscale = importlib.import_module("dinkster_inference_torch.resize").common_upscale

    def image_content(value: object, name: str, frames: int) -> tuple[Any, int]:
        image = _wan_image(value, name, torch)
        resized = common_upscale(
            image[:frames, :, :, :3].movedim(-1, 1),
            width,
            height,
            "bilinear",
            "center",
        ).movedim(1, -1)
        return (
            resized.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device),
            int(resized.shape[0]),
        )

    with handle.stage("vae"):
        with torch.inference_mode():
            neutral = handle.runtime.assembled.vae.process_out(
                torch.zeros(latent_shape, device=handle.load_device, dtype=torch.float32)
            )
            concat_image = neutral.repeat(1, 2, 1, 1, 1)
            start_frames = 0
            if start_image is not None:
                content, source_frames = image_content(start_image, "start_image", length)
                encoded_frames = ((source_frames - 1) // 4) + 1
                encoded = _wan_encode_content(
                    handle=handle,
                    content=content,
                    expected=(1, channels, encoded_frames, *latent_shape[-2:]),
                    name="start image",
                    torch=torch,
                )
                start_frames = source_frames
                concat_image[:, channels:, : encoded.shape[2]] = encoded[:, :, :latent_frames]
            if control_video is not None:
                content, source_frames = image_content(control_video, "control_video", length)
                encoded_frames = ((source_frames - 1) // 4) + 1
                encoded = _wan_encode_content(
                    handle=handle,
                    content=content,
                    expected=(1, channels, encoded_frames, *latent_shape[-2:]),
                    name="control video",
                    torch=torch,
                )
                concat_image[:, :channels, : encoded.shape[2]] = encoded[:, :, :latent_frames]
            reference = None
            if ref_image is not None:
                content, _ = image_content(ref_image, "ref_image", 1)
                reference = _wan_encode_content(
                    handle=handle,
                    content=content,
                    expected=(1, channels, 1, *latent_shape[-2:]),
                    name="reference image",
                    torch=torch,
                )

    mask_index = None
    concat_latent = concat_image
    if wan22:
        external_mask = torch.ones(
            (1, 1, latent_frames * 4, *latent_shape[-2:]),
            device=concat_image.device,
            dtype=concat_image.dtype,
        )
        if start_frames:
            external_mask[:, :, : start_frames + 3] = 0.0
        model_mask = _wan_flf_model_mask(external_mask, latent_frames)
        concat_latent = torch.cat(
            (concat_image[:, :channels], model_mask, concat_image[:, channels:]),
            dim=1,
        )
        mask_index = channels

    inference = importlib.import_module("dinkster_inference")
    prepared_positive = _wan_fun_prepared(
        positive,
        name="positive",
        handle=handle,
        inference=inference,
        concat_latent=concat_latent,
        concat_mask_index=mask_index,
        vision=vision,
        reference_latent=reference,
    )
    prepared_negative = _wan_fun_prepared(
        negative,
        name="negative",
        handle=handle,
        inference=inference,
        concat_latent=concat_latent,
        concat_mask_index=mask_index,
        vision=vision,
        reference_latent=reference,
    )
    latent = torch.zeros(
        (batch_size, channels, latent_frames, *latent_shape[-2:]),
        device="cpu",
        dtype=concat_latent.dtype,
    )
    return cls.outputs(
        positive=prepared_positive,
        negative=prepared_negative,
        latent={"samples": latent},
    )


class NativeWanFunControlToVideo(WanFunControlToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        clip_vision_output: object = None,
        start_image: object = None,
        control_video: object = None,
    ) -> Mapping[str, object]:
        return _execute_wan_fun_control(
            cls,
            positive=positive,
            negative=negative,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
            wan22=False,
            clip_vision_output=clip_vision_output,
            start_image=start_image,
            control_video=control_video,
        )


class NativeWan22FunControlToVideo(Wan22FunControlToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        ref_image: object = None,
        start_image: object = None,
        control_video: object = None,
    ) -> Mapping[str, object]:
        return _execute_wan_fun_control(
            cls,
            positive=positive,
            negative=negative,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
            wan22=True,
            ref_image=ref_image,
            start_image=start_image,
            control_video=control_video,
        )


class NativeWanFunInpaintToVideo(WanFunInpaintToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        clip_vision_output: object = None,
        start_image: object = None,
        end_image: object = None,
    ) -> Mapping[str, object]:
        _wan_fun_dimensions(
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
        )
        handle = _native_handle(vae, "vae")
        config = handle.runtime.assembled.diffusion.config
        channels = config.out_channels
        if config.in_channels - channels != channels + 4 or config.reference_channels is not None:
            raise ValueError("vae must come from a Wan Fun inpaint profile")
        torch = _torch()
        vision = None
        if config.model_type == "i2v":
            vision = _wan21_clip_embedding(
                clip_vision_output,
                name="clip_vision_output",
                handle=handle,
                torch=torch,
            )
        elif clip_vision_output is not None:
            raise ValueError("Wan 2.2 Fun inpaint does not consume CLIP vision output")

        common_upscale = importlib.import_module("dinkster_inference_torch.resize").common_upscale
        resized_start = None
        if start_image is not None:
            image = _wan_image(start_image, "start_image", torch)
            resized_start = common_upscale(
                image[:length, :, :, :3].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)
        resized_end = None
        if end_image is not None:
            image = _wan_image(end_image, "end_image", torch)
            resized_end = common_upscale(
                image[-length:, :, :, :3].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)
        image_device = (
            resized_start.device
            if resized_start is not None
            else resized_end.device
            if resized_end is not None
            else torch.device("cpu")
        )
        image = torch.full(
            (length, height, width, 3),
            0.5,
            device=image_device,
            dtype=torch.float32,
        )
        if resized_start is not None:
            image[: resized_start.shape[0]] = resized_start.to(image)
        if resized_end is not None:
            image[-resized_end.shape[0] :] = resized_end.to(image)
        spatial_scale = 16 if channels == 48 else 8
        latent_frames = ((length - 1) // 4) + 1
        latent_spatial = (height // spatial_scale, width // spatial_scale)
        content = image.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
        with handle.stage("vae"):
            with torch.inference_mode():
                encoded = _wan_encode_content(
                    handle=handle,
                    content=content,
                    expected=(1, channels, latent_frames, *latent_spatial),
                    name="inpaint video",
                    torch=torch,
                )
        external_mask = torch.ones(
            (1, 1, latent_frames * 4, *latent_spatial),
            device=encoded.device,
            dtype=encoded.dtype,
        )
        if resized_start is not None:
            external_mask[:, :, : resized_start.shape[0] + 3] = 0.0
        if resized_end is not None:
            external_mask[:, :, -resized_end.shape[0] :] = 0.0
        model_mask = _wan_flf_model_mask(external_mask, latent_frames)
        concat_latent = torch.cat((model_mask, encoded), dim=1)
        inference = importlib.import_module("dinkster_inference")
        prepared_positive = _wan_fun_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            concat_mask_index=0,
            vision=vision,
        )
        prepared_negative = _wan_fun_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            concat_mask_index=0,
            vision=vision,
        )
        latent = torch.zeros(
            (batch_size, channels, latent_frames, *latent_spatial),
            device="cpu",
            dtype=encoded.dtype,
        )
        return cls.outputs(
            positive=prepared_positive,
            negative=prepared_negative,
            latent={"samples": latent},
        )


def _wan_vace_prepared(
    value: object,
    *,
    name: str,
    handle: NativeRuntimeHandle,
    inference: Any,
    frames: Any,
    mask: Any,
    strength: float,
) -> list[list[object]]:
    text = _prepared_multistream_conditioning(
        value,
        inference,
        name,
        handle.runtime.conditioning_identity,
    )
    if text is None:
        raise TypeError(f"{name} must contain Wan 2.1 text conditioning")
    prepared = handle.runtime.prepare_vace_conditioning(text, frames, mask, strength)
    return [
        [
            inference.PreparedMultiStreamConditioning(
                handle.runtime.conditioning_identity,
                prepared,
            ),
            {},
        ]
    ]


class NativeWanVaceToVideo(WanVaceToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        strength: float,
        control_video: object = None,
        control_masks: object = None,
        reference_image: object = None,
    ) -> Mapping[str, object]:
        if width < 16 or width > 16384 or width % 16 != 0:
            raise ValueError("width must be a multiple of 16 between 16 and 16384")
        if height < 16 or height > 16384 or height % 16 != 0:
            raise ValueError("height must be a multiple of 16 between 16 and 16384")
        if length < 1 or length > 16384 or (length - 1) % 4 != 0:
            raise ValueError("length must be 1 plus a multiple of 4 between 1 and 16384")
        if batch_size < 1 or batch_size > 4096:
            raise ValueError("batch_size must be between 1 and 4096")
        if (
            type(strength) not in (int, float)
            or not math.isfinite(strength)
            or not 0 <= strength <= 1000
        ):
            raise ValueError("strength must be finite and between 0 and 1000")
        strength = float(strength)

        handle = _native_handle(vae, "vae")
        config = handle.runtime.assembled.diffusion.config
        if config.vace_layers is None:
            raise ValueError("vae must come from a Wan 2.1 VACE profile")

        torch = _torch()
        common_upscale = importlib.import_module("dinkster_inference_torch.resize").common_upscale
        if control_video is None:
            control = torch.full(
                (length, height, width, 3),
                0.5,
                device="cpu",
                dtype=torch.float32,
            )
        else:
            image = _wan_image(control_video, "control_video", torch)
            resized = common_upscale(
                image[:length, :, :, :3].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)
            control = torch.full(
                (length, height, width, 3),
                0.5,
                device=resized.device,
                dtype=resized.dtype,
            )
            control[: resized.shape[0]] = resized

        if control_masks is None:
            mask = torch.ones(
                (length, height, width, 1),
                device=control.device,
                dtype=control.dtype,
            )
        else:
            if type(control_masks) is not torch.Tensor:
                raise TypeError("control_masks must be an exact torch.Tensor")
            mask_input = cast("Any", control_masks)
            if mask_input.ndim == 3:
                mask_input = mask_input.unsqueeze(1)
            if (
                mask_input.ndim != 4
                or mask_input.shape[0] <= 0
                or mask_input.shape[1] != 1
                or mask_input.shape[2] <= 0
                or mask_input.shape[3] <= 0
                or not mask_input.is_floating_point()
                or mask_input.layout != torch.strided
            ):
                raise ValueError(
                    "control_masks must be a nonempty strided floating [frames,H,W] tensor"
                )
            resized_mask = common_upscale(
                mask_input[:length], width, height, "bilinear", "center"
            ).movedim(1, -1)
            mask = torch.ones(
                (length, height, width, 1),
                device=control.device,
                dtype=control.dtype,
            )
            mask[: resized_mask.shape[0]] = resized_mask.to(
                device=control.device,
                dtype=control.dtype,
            )

        centered = control - 0.5
        inactive = centered * (1.0 - mask) + 0.5
        reactive = centered * mask + 0.5

        reference_content = None
        if reference_image is not None:
            image = _wan_image(reference_image, "reference_image", torch)
            resized_reference = common_upscale(
                image[:1, :, :, :3].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)
            reference_content = (
                resized_reference.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
            )

        def encode(content: Any, expected_time: int, name: str) -> Any:
            direct_oom: BaseException | None = None
            encoded: Any = None
            try:
                encoded = handle.runtime.encode_content(content)
            except RuntimeError as caught:
                if not _is_accelerator_oom(caught, torch=torch, device=handle.load_device):
                    raise
                direct_oom = caught.with_traceback(None)
            if direct_oom is not None:
                encoded = _retry_tiled_vae_after_oom(
                    handle=handle,
                    value=content,
                    output_dtype=torch.float32,
                    direction="encode",
                    oom=direct_oom,
                )
            expected = (1, 16, expected_time, height // 8, width // 8)
            if type(encoded) is not torch.Tensor or tuple(encoded.shape) != expected:
                actual = getattr(encoded, "shape", None)
                raise ValueError(f"Wan VACE {name} latent has shape {actual}, expected {expected}")
            return encoded

        latent_length = ((length - 1) // 4) + 1
        inactive_content = inactive.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
        reactive_content = reactive.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
        with handle.stage("vae"):
            with torch.inference_mode():
                inactive_latent = encode(inactive_content, latent_length, "inactive")
                reactive_latent = encode(reactive_content, latent_length, "reactive")
                frames = torch.cat((inactive_latent, reactive_latent), dim=1)
                reference = None
                if reference_content is not None:
                    encoded_reference = encode(reference_content, 1, "reference")
                    neutral_reference = handle.runtime.assembled.vae.process_out(
                        torch.zeros_like(encoded_reference)
                    )
                    reference = torch.cat((encoded_reference, neutral_reference), dim=1)

        mask = mask.reshape(length, height // 8, 8, width // 8, 8)
        mask = mask.permute(2, 4, 0, 1, 3).reshape(64, length, height // 8, width // 8)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0),
            size=(latent_length, height // 8, width // 8),
            mode="nearest-exact",
        ).squeeze(0)

        trim_latent = 0
        if reference is not None:
            trim_latent = int(reference.shape[2])
            frames = torch.cat((reference, frames), dim=2)
            mask = torch.cat((torch.zeros_like(mask[:, :trim_latent]), mask), dim=1)
            latent_length += trim_latent
        mask = mask.unsqueeze(0)

        inference = importlib.import_module("dinkster_inference")
        prepared_positive = _wan_vace_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            frames=frames,
            mask=mask,
            strength=strength,
        )
        prepared_negative = _wan_vace_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            frames=frames,
            mask=mask,
            strength=strength,
        )
        latent = torch.zeros(
            (batch_size, 16, latent_length, height // 8, width // 8),
            device="cpu",
            dtype=frames.dtype,
        )
        return cls.outputs(
            positive=prepared_positive,
            negative=prepared_negative,
            latent={"samples": latent},
            trim_latent=trim_latent,
        )


class NativeWan22ImageToVideoLatent(Wan22ImageToVideoLatent):
    @classmethod
    def execute(
        cls,
        *,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        start_image: object = None,
    ) -> Mapping[str, object]:
        if width < 32 or width > 16384 or width % 16 != 0:
            raise ValueError("width must be a multiple of 16 between 32 and 16384")
        if height < 32 or height > 16384 or height % 16 != 0:
            raise ValueError("height must be a multiple of 16 between 32 and 16384")
        if length < 1 or length > 16384 or (length - 1) % 4 != 0:
            raise ValueError("length must be 1 plus a multiple of 4 between 1 and 16384")
        if batch_size < 1 or batch_size > 4096:
            raise ValueError("batch_size must be between 1 and 4096")
        handle = _native_handle(vae, "vae")
        if handle.runtime.assembled.diffusion.config.model_type != "ti2v":
            raise ValueError("vae must come from the Wan 2.2 TI2V profile")
        torch = _torch()
        latent_frames = ((length - 1) // 4) + 1
        latent_shape = (1, 48, latent_frames, height // 16, width // 16)
        if start_image is None:
            latent = torch.zeros(latent_shape, device="cpu")
            return cls.outputs(latent={"samples": latent})
        if type(start_image) is not torch.Tensor:
            raise TypeError("start_image must be an exact torch.Tensor")
        image = cast("Any", start_image)
        if (
            image.ndim != 4
            or image.shape[0] <= 0
            or image.shape[1] <= 0
            or image.shape[2] <= 0
            or image.shape[3] < 3
            or not image.is_floating_point()
            or image.layout != torch.strided
        ):
            raise ValueError(
                "start_image must be a nonempty strided floating [frames,H,W,C>=3] tensor"
            )
        resized = (
            importlib.import_module("dinkster_inference_torch.resize")
            .common_upscale(
                image[:length, :, :, :3].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            )
            .movedim(1, -1)
        )
        content = resized.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
        with handle.stage("vae"):
            with torch.inference_mode():
                direct_oom: BaseException | None = None
                encoded: Any = None
                try:
                    encoded = handle.runtime.encode_content(content)
                except RuntimeError as caught:
                    if not _is_accelerator_oom(caught, torch=torch, device=handle.load_device):
                        raise
                    direct_oom = caught.with_traceback(None)
                if direct_oom is not None:
                    encoded = _retry_tiled_vae_after_oom(
                        handle=handle,
                        value=content,
                        output_dtype=torch.float32,
                        direction="encode",
                        oom=direct_oom,
                    )
                expected = (
                    1,
                    48,
                    ((int(resized.shape[0]) - 1) // 4) + 1,
                    height // 16,
                    width // 16,
                )
                if type(encoded) is not torch.Tensor or tuple(encoded.shape) != expected:
                    actual = getattr(encoded, "shape", None)
                    raise ValueError(f"Wan 2.2 VAE latent has shape {actual}, expected {expected}")
                latent = torch.zeros(latent_shape, device=encoded.device, dtype=encoded.dtype)
                latent[:, :, : encoded.shape[2]] = encoded
                mask = torch.ones(
                    (1, 1, latent_frames, height // 16, width // 16),
                    device=encoded.device,
                    dtype=encoded.dtype,
                )
                mask[:, :, : encoded.shape[2]] = 0.0
                external = handle.runtime.assembled.vae.process_out(latent) * mask + latent * (
                    1.0 - mask
                )
        output = external.to("cpu").repeat(batch_size, 1, 1, 1, 1)
        output_mask = mask.to("cpu").repeat(batch_size, 1, 1, 1, 1)
        return cls.outputs(latent={"samples": output, "noise_mask": output_mask})


def _minimax_h3_video_vae_runtime(value: object, name: str = "video_vae") -> tuple[Any, Any]:
    inference = importlib.import_module("dinkster_inference")
    handle = load_registered_component(
        value, name, "video-vae", family_id=inference.MINIMAX_H3_CONFIG.family_id
    )
    torch = _torch()
    inference_torch = importlib.import_module("dinkster_inference_torch")
    assert handle.recipe is not None
    return handle, inference_torch.MiniMaxH3VideoVaeRuntime(
        handle.component,
        runtime_identity=handle.resource_identity,
        compute_dtype=_torch_dtype(torch, handle.recipe.knobs.vae_dtype),
    )


def _minimax_h3_audio_vae_runtime(value: object, name: str = "audio_vae") -> tuple[Any, Any]:
    inference = importlib.import_module("dinkster_inference")
    handle = load_registered_component(
        value, name, "audio-vae", family_id=inference.MINIMAX_H3_CONFIG.family_id
    )
    torch = _torch()
    inference_torch = importlib.import_module("dinkster_inference_torch")
    assert handle.recipe is not None
    return handle, inference_torch.MiniMaxH3AudioVaeRuntime(
        handle.component,
        runtime_identity=handle.resource_identity,
        compute_dtype=_torch_dtype(torch, handle.recipe.knobs.vae_dtype),
    )


def _latent_mask_codec_runtime(value: object, name: str) -> Any:
    inference = importlib.import_module("dinkster_inference")
    if isinstance(value, inference.LatentMaskCodecRuntime):
        return value
    if isinstance(value, NativeComponentHandle):
        recipe = value.recipe
        roles = () if recipe is None else tuple(binding.role for binding in recipe.sources)
        if roles == ("video-vae",):
            return _minimax_h3_video_vae_runtime(value, name)[1]
        if roles == ("audio-vae",):
            return _minimax_h3_audio_vae_runtime(value, name)[1]
    if isinstance(value, NativeRuntimeHandle):
        runtime = value.runtime
        for candidate in (runtime, getattr(runtime, "codec", None)):
            if isinstance(candidate, inference.LatentMaskCodecRuntime):
                return candidate
    raise TypeError(f"{name} does not declare latent mask geometry")


def _minimax_h3_conditioner_runtime(
    clip: object,
    video_vae: object | None = None,
    audio_vae: object | None = None,
) -> tuple[NativeComponentHandle, tuple[NativeComponentHandle, ...], Any]:
    inference = importlib.import_module("dinkster_inference")
    clip_handle = load_registered_component(
        clip,
        "clip",
        "qwen3vl-32b-conditioner",
        family_id=inference.MINIMAX_H3_CONFIG.family_id,
    )
    video_handle = video_runtime = None
    if video_vae is not None:
        video_handle, video_runtime = _minimax_h3_video_vae_runtime(video_vae)
    audio_handle = audio_runtime = None
    if audio_vae is not None:
        audio_handle, audio_runtime = _minimax_h3_audio_vae_runtime(audio_vae)
    assert clip_handle.recipe is not None
    runtime = importlib.import_module("dinkster_inference_torch").MiniMaxH3ConditionerRuntime(
        clip_handle.component,
        video_runtime,
        audio_runtime,
        runtime_identity=clip_handle.resource_identity,
    )
    codec_handles = tuple(handle for handle in (video_handle, audio_handle) if handle is not None)
    return clip_handle, cast("tuple[NativeComponentHandle, ...]", codec_handles), runtime


def _minimax_h3_av(value: object, torch: Any, inference: Any, name: str) -> Any:
    if not isinstance(value, Mapping) or "samples" not in value:
        raise TypeError(f"{name} must be a LATENT mapping containing 'samples'")
    streams = cast("Any", cast("Mapping[object, object]", value)["samples"])
    if type(streams) is not inference.MultiStreamLatent or streams.roles != ("video", "audio"):
        raise TypeError(f"{name} samples must have exact ordered video/audio streams")
    video, audio = cast("tuple[Any, Any]", tuple(stream.payload for stream in streams.streams))
    if type(video) is not torch.Tensor or type(audio) is not torch.Tensor:
        raise TypeError(f"{name} streams must be exact torch.Tensor values")
    if (
        not video.is_floating_point()
        or not audio.is_floating_point()
        or video.layout != torch.strided
        or audio.layout != torch.strided
    ):
        raise TypeError(f"{name} streams must be strided floating tensors")
    if (
        video.ndim != 5
        or tuple(video.shape[:2]) != (1, 24)
        or audio.ndim != 4
        or tuple(audio.shape[:3]) != (1, 32, 2)
        or min(video.shape[2:]) <= 0
        or audio.shape[3] <= 0
    ):
        raise ValueError(f"{name} must contain exact batch-one MiniMax H3 latent streams")
    return streams


def _move_multistream_latent(value: Any, device: object) -> Any:
    def move(payload: Any) -> Any:
        return payload.to(device)

    return value.map(move)


def _minimax_h3_payload(inference: Any, tensor: Any, reference_id: str) -> tuple[Any, Any]:
    descriptor = inference.PayloadDescriptor(
        inference.PayloadReference(reference_id),
        tuple(tensor.shape),
        str(tensor.dtype).removeprefix("torch."),
        "worker:minimax-h3",
    )
    return descriptor, tensor


def _minimax_h3_image_tensor(value: object, torch: Any, name: str) -> Any:
    return _minimax_h3_image_batch(value, torch, name)[:1]


def _minimax_h3_image_batch(value: object, torch: Any, name: str) -> Any:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{name} must be an exact torch.Tensor")
    tensor = cast("Any", value)
    if (
        tensor.ndim != 4
        or tensor.shape[0] <= 0
        or tensor.shape[-1] != 3
        or min(tensor.shape[1:3]) < 2
        or not tensor.is_floating_point()
        or tensor.layout != torch.strided
    ):
        raise ValueError(f"{name} must be a strided floating [batch,height,width,3] tensor")
    return tensor


def _nearest_32(value: float | int) -> int:
    return max(32, int(round(float(value) / 32.0)) * 32)


def _minimax_h3_resize(image: Any, width: int, height: int, crop: str) -> Any:
    common_upscale = importlib.import_module("dinkster_inference_torch.resize").common_upscale
    nchw = image.permute(0, 3, 1, 2)
    return common_upscale(nchw, width, height, "lanczos", crop).permute(0, 2, 3, 1)


def _minimax_h3_target_canvas(target: Any) -> tuple[int, int]:
    shape = tuple(target.by_role("video").shape)
    if len(shape) != 5 or shape[0] != 1 or shape[1] != 24:
        raise ValueError("MiniMax H3 target video must be [1,24,time,height,width]")
    return shape[4] * 16, shape[3] * 16


def _minimax_h3_reference_image(image: Any, target: Any, mode: str) -> Any:
    if mode not in ("match", "max"):
        raise ValueError("ref_image_size must be 'match' or 'max'")
    target_width, target_height = _minimax_h3_target_canvas(target)
    source_height, source_width = int(image.shape[1]), int(image.shape[2])
    if mode == "match":
        scale = min(
            1.0,
            math.sqrt((target_width * target_height) / (source_width * source_height)),
        )
    else:
        scale = min(1.0, 2048.0 / min(source_width, source_height))
    width = _nearest_32(source_width * scale)
    height = _nearest_32(source_height * scale)
    return _minimax_h3_resize(image, width, height, "disabled")


def _minimax_h3_video_frames(frames: Any, frame_count: int) -> tuple[Any, tuple[int, ...]]:
    source_height, source_width = int(frames.shape[1]), int(frames.shape[2])
    scale = min(
        768.0 / min(source_width, source_height),
        math.sqrt((768 * 1344) / (source_width * source_height)),
    )
    width = _nearest_32(source_width * scale)
    height = _nearest_32(source_height * scale)
    if source_width * source_height < width * height:
        width = _nearest_32(source_width)
        height = _nearest_32(source_height)
    adapted = _minimax_h3_resize(frames, width, height, "disabled")
    count = min(int(adapted.shape[0]), frame_count)
    if count < 5:
        raise ValueError("MiniMax H3 video references require at least 5 frames")
    count -= (count - 5) % 17
    presentation_indices = tuple(range(0, count, 12))
    return adapted[:count], presentation_indices


def _minimax_h3_audio_value(value: object, torch: Any, name: str) -> tuple[Any, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be the standard waveform/sample_rate mapping")
    audio = cast("Mapping[object, object]", value)
    if set(audio) != {"waveform", "sample_rate"}:
        raise TypeError(f"{name} must be the standard waveform/sample_rate mapping")
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if type(waveform) is not torch.Tensor:
        raise TypeError(f"{name}.waveform must be an exact torch.Tensor")
    tensor = cast("Any", waveform)
    if (
        tensor.ndim != 3
        or tensor.shape[0] <= 0
        or tensor.shape[1] != 2
        or tensor.shape[2] <= 0
        or not tensor.is_floating_point()
        or tensor.layout != torch.strided
    ):
        raise ValueError(f"{name}.waveform must be nonempty floating [batch,2,samples]")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError(f"{name}.sample_rate must be a positive integer")
    return tensor, sample_rate


def _minimax_h3_resample_audio(waveform: Any, sample_rate: int) -> Any:
    if sample_rate == 32_000:
        return waveform
    return importlib.import_module("torchaudio.functional").resample(waveform, sample_rate, 32_000)


def _minimax_h3_audio_reference(
    inference: Any,
    waveform: Any,
    sample_rate: int,
    reference_id: str,
) -> tuple[Any, Any]:
    selected = _minimax_h3_resample_audio(waveform[:1], sample_rate)
    descriptor, payload = _minimax_h3_payload(inference, selected, reference_id)
    return inference.MiniMaxH3AudioReference(descriptor, 32_000), payload


def _minimax_h3_video_reference(
    inference: Any,
    descriptors: tuple[Any, ...],
    presentation_indices: tuple[int, ...],
    audio: Any,
) -> Any:
    return inference.MiniMaxH3VideoReference(
        descriptors,
        presentation_indices,
        tuple(float(index / 24.0) for index in presentation_indices),
        audio,
    )


def _canonical_minimax_h3_references(values: Sequence[object]) -> tuple[object, ...]:
    allowed = (
        MiniMaxH3ImageReferenceValue,
        MiniMaxH3VideoReferenceValue,
        MiniMaxH3AudioReferenceValue,
    )
    if any(type(reference) not in allowed for reference in values):
        raise TypeError("references must contain exact MiniMax H3 reference values")
    images = tuple(
        reference for reference in values if type(reference) is MiniMaxH3ImageReferenceValue
    )
    videos = tuple(
        reference for reference in values if type(reference) is MiniMaxH3VideoReferenceValue
    )
    audios = tuple(
        reference for reference in values if type(reference) is MiniMaxH3AudioReferenceValue
    )
    if len(images) > 9 or len(videos) > 3 or len(audios) > 3:
        raise ValueError("REF2VA allows at most 9 images, 3 videos, and 3 audio references")
    return (*images, *videos, *audios)


def _adapt_multistream_latent(
    value: object,
    runtime: object,
    torch: Any,
    inference: Any,
    name: str,
) -> Mapping[object, object]:
    if not isinstance(value, Mapping) or "samples" not in value:
        raise TypeError(f"{name} must be a LATENT mapping containing 'samples'")
    latent = cast("Mapping[object, object]", value)
    samples = latent["samples"]
    if type(samples) is inference.MultiStreamLatent:
        return latent
    if not isinstance(runtime, inference.MultiStreamLatentAdapterRuntime):
        raise TypeError(f"{name} cannot be adapted to the model's latent streams")
    if type(samples) is not torch.Tensor:
        raise TypeError(f"{name} samples must be an exact torch.Tensor")
    adapter = cast("Any", runtime)
    adapted = adapter.adapt_multistream_latent(
        samples,
        source_spatial_downscale=latent.get("downscale_ratio_spacial"),
        source_temporal_downscale=latent.get("downscale_ratio_temporal"),
    )
    if type(adapted) is not inference.MultiStreamLatent:
        raise TypeError("latent adaptation must return an exact MultiStreamLatent")
    result = dict(latent)
    result["samples"] = adapted
    return result


def _adapt_minimax_h3_av(
    value: object,
    runtime: object,
    torch: Any,
    inference: Any,
    name: str,
) -> Any:
    adapted = _adapt_multistream_latent(value, runtime, torch, inference, name)
    return _minimax_h3_av(adapted, torch, inference, name)


def _minimax_h3_frame_count(target: Any) -> int:
    shape = tuple(target.by_role("video").shape)
    if len(shape) != 5 or shape[0] != 1 or shape[1] != 24:
        raise ValueError("MiniMax H3 target video must be [1,24,time,height,width]")
    temporal = shape[2]
    if temporal < 1:
        raise ValueError("MiniMax H3 target video latent must have a positive temporal extent")
    temporal_mapping = importlib.import_module(
        "dinkster_inference"
    ).MINIMAX_H3_VIDEO_TEMPORAL_MAPPING
    return temporal_mapping.content_extent(temporal)


def _minimax_h3_condition(
    cls: type[Node],
    conditioner_handle: NativeComponentHandle,
    codec_handles: tuple[NativeComponentHandle, ...],
    runtime: Any,
    av: Any,
    request: Any,
    negative_prompt: str | None,
    payloads: Mapping[str, Any],
) -> Mapping[str, object]:
    if negative_prompt is not None and type(negative_prompt) is not str:
        raise TypeError("negative_prompt must be a string when provided")
    negative_request = None if negative_prompt is None else replace(request, prompt=negative_prompt)
    torch = _torch()
    inference = importlib.import_module("dinkster_inference")
    frame_count = _minimax_h3_frame_count(av)
    load_av = _move_multistream_latent(av, conditioner_handle.load_device)
    context = current_execution_context()
    cancelled = context.cancelled if context is not None else _not_cancelled
    with native_execution_span(
        "condition", "condition", device=str(conditioner_handle.load_device)
    ) as span:
        parent = None if span is None else span.span_id
        with ExitStack() as stages:
            stages.enter_context(
                conditioner_handle.stage(
                    observer_stage="condition",
                    parent_span_id=parent,
                )
            )
            for handle in codec_handles:
                stages.enter_context(
                    handle.stage(
                        observer_stage="condition",
                        parent_span_id=parent,
                    )
                )
            with torch.inference_mode():
                prepared = runtime.condition(
                    request,
                    target=load_av,
                    frame_count=frame_count,
                    payloads=payloads,
                    cancelled=cancelled,
                )
                negative_prepared = (
                    None
                    if negative_request is None
                    else runtime.condition(
                        negative_request,
                        target=load_av,
                        frame_count=frame_count,
                        payloads=payloads,
                        cancelled=cancelled,
                    )
                )

    def lane(payload: object) -> list[list[object]]:
        value = inference.PreparedMultiStreamConditioning(
            conditioner_handle.resource_identity,
            payload,
        )
        return [[value, cast("dict[str, object]", {})]]

    return cls.outputs(
        positive=lane(prepared),
        negative=[] if negative_prepared is None else lane(negative_prepared),
    )


class NativeEmptyMiniMaxH3AV(EmptyMiniMaxH3AV):
    @classmethod
    def execute(cls, *, width: int, height: int, frame_count: int) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch")
        value = inference_torch.empty_minimax_h3_av(
            width=width,
            height=height,
            frame_count=frame_count,
            device="cpu",
            dtype=_torch().bfloat16,
        )
        return cls.outputs(latent={"samples": value})


class NativeEmptyMiniMaxMusic3LatentAudio(EmptyMiniMaxMusic3LatentAudio):
    @classmethod
    def execute(cls, *, seconds: float, batch_size: int) -> Mapping[str, object]:
        if type(seconds) not in (int, float) or not 0.04 <= seconds <= 360.0:
            raise ValueError("seconds must be in [0.04, 360.0]")
        if type(batch_size) is not int or not 1 <= batch_size <= 4096:
            raise ValueError("batch_size must be an integer in [1, 4096]")
        inference = importlib.import_module("dinkster_inference")
        torch = _torch()
        audio_frames = min(
            inference.MAX_AUDIO_FRAMES,
            max(1, round(seconds * inference.AUDIO_FRAMES_PER_SECOND)),
        )
        samples = torch.zeros(
            (
                batch_size,
                inference.MINIMAX_MUSIC3_CONFIG.latent_channels,
                inference.minimax_music3_latent_length(audio_frames),
            ),
            device="cpu",
        )
        return cls.outputs(
            latent={"samples": samples, "type": "audio", "downscale_ratio_temporal": 512}
        )


class NativeEmptyLTXAVLatent(EmptyLTXAVLatent):
    @classmethod
    def execute(
        cls,
        *,
        model: object,
        width: int,
        height: int,
        length: int,
        frame_rate: int,
        batch_size: int,
    ) -> Mapping[str, object]:
        handle = _native_handle(model, "model")
        for name, value in (
            ("width", width),
            ("height", height),
            ("length", length),
            ("frame_rate", frame_rate),
            ("batch_size", batch_size),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        inference = importlib.import_module("dinkster_inference")
        runtime = handle.runtime
        geometry = getattr(runtime, "component_sampling_runtime", runtime)
        video_config = getattr(geometry, "video_vae_config", None)
        audio_config = getattr(geometry, "audio_vae_config", None)
        if video_config is None:
            raise TypeError("model requires video_vae_config")
        if audio_config is None:
            raise TypeError("model requires audio_vae_config")
        for config_name, config, fields in (
            (
                "video_vae_config",
                video_config,
                ("latent_channels", "temporal_ratio", "spatial_ratio"),
            ),
            ("audio_vae_config", audio_config, ("z_channels", "latent_frequency_bins")),
        ):
            for field in fields:
                value = getattr(config, field, None)
                if type(value) is not int or value <= 0:
                    raise TypeError(f"model requires {config_name}.{field} as a positive integer")
        rate = getattr(audio_config, "latents_per_second", None)
        if (
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or not math.isfinite(rate)
            or rate <= 0
        ):
            raise TypeError("model requires positive finite audio_vae_config.latents_per_second")
        audio_length = inference.ltx_audio_latents_from_frames(
            audio_config, length, float(frame_rate)
        )
        if width < video_config.spatial_ratio or height < video_config.spatial_ratio:
            raise ValueError(
                "width and height must cover at least one video spatial downscale step"
            )
        if audio_length < 1:
            raise ValueError("length and frame_rate must produce at least one audio latent frame")
        torch = _torch()
        video = torch.zeros(
            (
                batch_size,
                video_config.latent_channels,
                (length - 1) // video_config.temporal_ratio + 1,
                height // video_config.spatial_ratio,
                width // video_config.spatial_ratio,
            ),
            device="cpu",
            dtype=torch.float32,
        )
        audio = torch.zeros(
            (
                batch_size,
                audio_config.z_channels,
                audio_length,
                audio_config.latent_frequency_bins,
            ),
            device="cpu",
            dtype=torch.float32,
        )
        streams = inference.MultiStreamLatent.from_pairs((("video", video), ("audio", audio)))
        return cls.outputs(latent={"samples": streams})


class NativeEmptyLTXVLatent(EmptyLTXVLatent):
    @classmethod
    def execute(
        cls,
        *,
        model: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
    ) -> Mapping[str, object]:
        handle = _native_handle(model, "model")
        for name, value in (
            ("width", width),
            ("height", height),
            ("length", length),
            ("batch_size", batch_size),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        inference = importlib.import_module("dinkster_inference")
        runtime = handle.runtime
        geometry = getattr(runtime, "component_sampling_runtime", runtime)
        video_config = getattr(geometry, "video_vae_config", None)
        if video_config is None:
            raise TypeError("model requires video_vae_config")
        for field in ("latent_channels", "temporal_ratio", "spatial_ratio"):
            value = getattr(video_config, field, None)
            if type(value) is not int or value <= 0:
                raise TypeError(f"model requires video_vae_config.{field} as a positive integer")
        if width < video_config.spatial_ratio or height < video_config.spatial_ratio:
            raise ValueError(
                "width and height must cover at least one video spatial downscale step"
            )
        torch = _torch()
        video = torch.zeros(
            (
                batch_size,
                video_config.latent_channels,
                (length - 1) // video_config.temporal_ratio + 1,
                height // video_config.spatial_ratio,
                width // video_config.spatial_ratio,
            ),
            device="cpu",
            dtype=torch.float32,
        )
        streams = inference.MultiStreamLatent.from_pairs((("video", video),))
        return cls.outputs(latent={"samples": streams})


class NativeMiniMaxH3T2VAConditioning(MiniMaxH3T2VAConditioning):
    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        target: object,
        prompt: str,
        negative_prompt: str | None = None,
    ) -> Mapping[str, object]:
        handle, codecs, runtime = _minimax_h3_conditioner_runtime(clip)
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        target_av = _adapt_minimax_h3_av(target, runtime, torch, inference, "target")
        return _minimax_h3_condition(
            cls,
            handle,
            codecs,
            runtime,
            target_av,
            inference.MiniMaxH3T2VARequest(prompt),
            negative_prompt,
            {},
        )


class NativeMiniMaxH3FL2VAConditioning(MiniMaxH3FL2VAConditioning):
    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        video_vae: object,
        target: object,
        prompt: str,
        negative_prompt: str | None = None,
        first_image: object = None,
        last_image: object = None,
    ) -> Mapping[str, object]:
        handle, codecs, runtime = _minimax_h3_conditioner_runtime(clip, video_vae)
        if first_image is None and last_image is None:
            raise ValueError("FL2VA requires at least one keyframe")
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        target_av = _adapt_minimax_h3_av(target, runtime, torch, inference, "target")
        target_width, target_height = _minimax_h3_target_canvas(target_av)
        keyframes: list[Any] = []
        payloads: dict[str, Any] = {}
        for ordinal, (role_name, role, value, crop) in enumerate(
            (
                (
                    "first",
                    inference.MiniMaxH3KeyframeRole.FIRST,
                    first_image,
                    "disabled",
                ),
                (
                    "last",
                    inference.MiniMaxH3KeyframeRole.LAST,
                    last_image,
                    "center",
                ),
            ),
            1,
        ):
            if value is None:
                continue
            tensor = _minimax_h3_image_tensor(value, torch, f"{role_name}_image")
            tensor = _minimax_h3_resize(tensor, target_width, target_height, crop)
            reference_id = f"fl2va:{ordinal}:keyframe:{role_name}"
            descriptor, payload = _minimax_h3_payload(inference, tensor, reference_id)
            payloads[reference_id] = payload
            keyframes.append(inference.MiniMaxH3Keyframe(role, descriptor))
        request = inference.MiniMaxH3FL2VARequest(prompt, tuple(keyframes))
        return _minimax_h3_condition(
            cls,
            handle,
            codecs,
            runtime,
            target_av,
            request,
            negative_prompt,
            payloads,
        )


class NativeMiniMaxH3REF2VAConditioning(MiniMaxH3REF2VAConditioning):
    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        video_vae: object,
        audio_vae: object,
        target: object,
        prompt: str,
        references: object,
        ref_image_size: str,
        negative_prompt: str | None = None,
    ) -> Mapping[str, object]:
        if isinstance(references, str | bytes) or not isinstance(references, Sequence):
            raise TypeError("references must be a sequence")
        values = tuple(cast("Sequence[object]", references))
        if not values:
            raise ValueError("REF2VA requires at least one reference")
        if ref_image_size not in ("match", "max"):
            raise ValueError("ref_image_size must be 'match' or 'max'")
        canonical = _canonical_minimax_h3_references(values)
        handle, codecs, runtime = _minimax_h3_conditioner_runtime(
            clip,
            video_vae,
            audio_vae,
        )
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        target_av = _adapt_minimax_h3_av(target, runtime, torch, inference, "target")
        frame_count = _minimax_h3_frame_count(target_av)
        typed: list[Any] = []
        payloads: dict[str, Any] = {}
        for ordinal, reference in enumerate(canonical, 1):
            if type(reference) is MiniMaxH3ImageReferenceValue:
                image = _minimax_h3_image_tensor(reference.image, torch, "image reference")
                image = _minimax_h3_reference_image(image, target_av, ref_image_size)
                reference_id = f"ref2va:{ordinal}:image:1"
                descriptor, payload = _minimax_h3_payload(inference, image, reference_id)
                payloads[reference_id] = payload
                typed.append(inference.MiniMaxH3ImageReference(descriptor))
            elif type(reference) is MiniMaxH3AudioReferenceValue:
                waveform, sample_rate = _minimax_h3_audio_value(
                    {"waveform": reference.waveform, "sample_rate": reference.sample_rate},
                    torch,
                    "audio reference",
                )
                reference_id = f"ref2va:{ordinal}:audio:1"
                audio_reference, payload = _minimax_h3_audio_reference(
                    inference, waveform, sample_rate, reference_id
                )
                payloads[reference_id] = payload
                typed.append(audio_reference)
            elif type(reference) is MiniMaxH3VideoReferenceValue:
                if type(reference.frames) is not torch.Tensor:
                    raise TypeError("video reference frames must be an exact torch.Tensor")
                frames = cast("Any", reference.frames)
                if (
                    frames.ndim != 4
                    or frames.shape[0] <= 0
                    or frames.shape[-1] != 3
                    or min(frames.shape[1:3]) < 2
                    or not frames.is_floating_point()
                    or frames.layout != torch.strided
                ):
                    raise ValueError(
                        "video reference frames must be strided floating [time,height,width,3]"
                    )
                frames, presentation_indices = _minimax_h3_video_frames(frames, frame_count)
                descriptors: list[Any] = []
                for frame_ordinal, frame in enumerate(frames.split(1), 1):
                    reference_id = f"ref2va:{ordinal}:video:{frame_ordinal}"
                    descriptor, payload = _minimax_h3_payload(inference, frame, reference_id)
                    payloads[reference_id] = payload
                    descriptors.append(descriptor)
                audio_reference = None
                if reference.audio is not None:
                    waveform, sample_rate = _minimax_h3_audio_value(
                        {
                            "waveform": reference.audio.waveform,
                            "sample_rate": reference.audio.sample_rate,
                        },
                        torch,
                        "video audio reference",
                    )
                    reference_id = f"ref2va:{ordinal}:video-audio:1"
                    audio_reference, payload = _minimax_h3_audio_reference(
                        inference, waveform, sample_rate, reference_id
                    )
                    payloads[reference_id] = payload
                typed.append(
                    _minimax_h3_video_reference(
                        inference,
                        tuple(descriptors),
                        presentation_indices,
                        audio_reference,
                    )
                )
        request = inference.MiniMaxH3REF2VARequest(prompt, tuple(typed))
        return _minimax_h3_condition(
            cls,
            handle,
            codecs,
            runtime,
            target_av,
            request,
            negative_prompt,
            payloads,
        )


class NativeMiniMaxH3AddGuide(MiniMaxH3AddGuide):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        latent: object,
        frame_idx: int,
        vae: object = None,
        audio_vae: object = None,
        image: object = None,
        audio: object = None,
    ) -> Mapping[str, object]:
        if image is None and audio is None:
            raise ValueError("MiniMax H3 guides require an image or audio")
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        carrier = _minimax_h3_conditioning_carrier(positive, inference, "positive")
        if carrier is None:
            raise ValueError("positive conditioning must not be empty")
        target = _minimax_h3_av(latent, torch, inference, "latent")
        target_video = target.by_role("video")
        target_audio = target.by_role("audio")
        target_frames = _minimax_h3_frame_count(target)

        frames = None
        guide_frames = 1
        if image is not None:
            if vae is None:
                raise ValueError("anchoring guide frames requires the vae input")
            frames = _minimax_h3_image_batch(image, torch, "image")
            guide_frames = int(frames.shape[0])
            if guide_frames < 5:
                guide_frames = 1
            else:
                guide_frames -= (guide_frames - 5) % 17
            frames = frames[:guide_frames]
        resolved = inference.resolve_timeline_frame_index(frame_idx, target_frames)
        if resolved + guide_frames > target_frames:
            raise ValueError(
                f"a {guide_frames} frame guide at frame_idx {frame_idx} does not fit "
                f"the target's {target_frames} frames"
            )

        streams: list[tuple[str, Any]] = []
        if frames is not None:
            video_handle, video_runtime = _minimax_h3_video_vae_runtime(vae, "vae")
            target_width, target_height = _minimax_h3_target_canvas(target)
            frames = _minimax_h3_resize(frames, target_width, target_height, "center")
            with native_execution_span(
                "condition", "encode_guide_video", device=str(video_handle.load_device)
            ) as span:
                parent = None if span is None else span.span_id
                with video_handle.stage(observer_stage="condition", parent_span_id=parent):
                    with torch.inference_mode():
                        video = video_runtime.encode_video(
                            frames.permute(3, 0, 1, 2).unsqueeze(0).to(video_handle.load_device)
                        ).to(target_video)
            streams.append(("video", video))

        if audio is not None:
            if audio_vae is None:
                raise ValueError("anchoring guide audio requires the audio_vae input")
            audio_handle, audio_runtime = _minimax_h3_audio_vae_runtime(audio_vae)
            waveform, sample_rate = _minimax_h3_audio_value(audio, torch, "audio")
            waveform = _minimax_h3_resample_audio(waveform[:1], sample_rate)
            with native_execution_span(
                "condition", "encode_guide_audio", device=str(audio_handle.load_device)
            ) as span:
                parent = None if span is None else span.span_id
                with audio_handle.stage(observer_stage="condition", parent_span_id=parent):
                    with torch.inference_mode():
                        audio_latent = audio_runtime.encode_audio(
                            inference.MiniMaxH3AudioContent(
                                waveform.to(audio_handle.load_device),
                                32_000,
                            )
                        ).to(target_audio)
            max_audio = math.floor(
                target_audio.shape[-1]
                - inference.MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.timeline_position(resolved)
            )
            if max_audio < 1:
                raise ValueError(f"frame_idx {frame_idx} is past the end of the target audio")
            if audio_latent.shape[-1] > max_audio:
                audio_latent = audio_latent[..., :max_audio].clone()
            streams.append(("audio", audio_latent))

        guide = inference.TimelineGuide(
            resolved,
            guide_frames,
            inference.MultiStreamLatent.from_pairs(streams),
        )
        prepared = inference_torch.add_minimax_h3_timeline_guide(
            carrier.payload,
            target,
            guide,
        )
        output = inference.PreparedMultiStreamConditioning(carrier.runtime_identity, prepared)
        conditioned: list[list[object]] = [[output, cast("dict[str, object]", {})]]
        return cls.outputs(positive=conditioned)


class NativeMiniMaxH3MotionContext(MiniMaxH3MotionContext):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        latent: object,
        previous_latent: object,
        context_length: int,
    ) -> Mapping[str, object]:
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        carrier = _minimax_h3_conditioning_carrier(positive, inference, "positive")
        if carrier is None:
            raise ValueError("positive conditioning must not be empty")
        target = _minimax_h3_av(latent, torch, inference, "latent")
        previous = _minimax_h3_av(previous_latent, torch, inference, "previous_latent")
        prepared, trim_time = inference_torch.add_minimax_h3_motion_context(
            carrier.payload,
            target,
            previous,
            context_length,
        )
        output = inference.PreparedMultiStreamConditioning(carrier.runtime_identity, prepared)
        conditioned: list[list[object]] = [[output, cast("dict[str, object]", {})]]
        return cls.outputs(positive=conditioned, trim_time=trim_time)


class NativeMiniMaxH3AVEncode(MiniMaxH3AVEncode):
    @classmethod
    def execute(
        cls,
        *,
        video_vae: object,
        audio_vae: object,
        frames: object,
        audio: object,
    ) -> Mapping[str, object]:
        video_handle, video_runtime = _minimax_h3_video_vae_runtime(video_vae)
        audio_handle, audio_runtime = _minimax_h3_audio_vae_runtime(audio_vae)
        torch = _torch()
        if type(frames) is not torch.Tensor:
            raise TypeError("frames must be an exact torch.Tensor")
        images = cast("Any", frames)
        if (
            images.ndim != 4
            or images.shape[0] <= 0
            or images.shape[-1] != 3
            or min(images.shape[1:3]) < 2
            or not images.is_floating_point()
            or images.layout != torch.strided
        ):
            raise ValueError("frames must be nonempty strided floating [time,height,width,3]")
        waveform, sample_rate = _minimax_h3_audio_value(audio, torch, "audio")
        if waveform.shape[0] != 1:
            raise ValueError("audio.waveform must have exact batch size one")
        waveform = _minimax_h3_resample_audio(waveform, sample_rate)
        inference = importlib.import_module("dinkster_inference")
        with native_execution_span(
            "encode", "encode_video", device=str(video_handle.load_device)
        ) as span:
            parent = None if span is None else span.span_id
            with video_handle.stage(observer_stage="encode", parent_span_id=parent):
                with torch.inference_mode():
                    video = video_runtime.encode_video(
                        images.permute(3, 0, 1, 2).unsqueeze(0).to(video_handle.load_device)
                    )
        with native_execution_span(
            "encode", "encode_audio", device=str(audio_handle.load_device)
        ) as span:
            parent = None if span is None else span.span_id
            with audio_handle.stage(observer_stage="encode", parent_span_id=parent):
                with torch.inference_mode():
                    audio_latent = audio_runtime.encode_audio(
                        inference.MiniMaxH3AudioContent(
                            waveform.to(audio_handle.load_device),
                            32_000,
                        )
                    )
        streams = inference.MultiStreamLatent.from_pairs(
            (("video", video), ("audio", audio_latent))
        )
        return cls.outputs(latent={"samples": streams})


class NativeMiniMaxH3AVDecode(MiniMaxH3AVDecode):
    @classmethod
    def execute(
        cls,
        *,
        video_vae: object,
        audio_vae: object,
        latent: object,
    ) -> Mapping[str, object]:
        video_handle, video_runtime = _minimax_h3_video_vae_runtime(video_vae)
        audio_handle, audio_runtime = _minimax_h3_audio_vae_runtime(audio_vae)
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        value = _minimax_h3_av(latent, torch, inference, "latent")
        video_latent = value.by_role("video")
        audio_latent = value.by_role("audio")
        if (
            video_latent.ndim != 5
            or video_latent.shape[0] != 1
            or video_latent.shape[1] != 24
            or audio_latent.ndim != 4
            or tuple(audio_latent.shape[:3]) != (1, 32, 2)
        ):
            raise ValueError("av must contain exact batch-one MiniMax H3 latent streams")
        with native_execution_span(
            "decode", "decode_video", device=str(video_handle.load_device)
        ) as span:
            parent = None if span is None else span.span_id
            with video_handle.stage(observer_stage="decode", parent_span_id=parent):
                with torch.inference_mode():
                    video = video_runtime.decode_video(video_latent.to(video_handle.load_device))
        with native_execution_span(
            "decode", "decode_audio", device=str(audio_handle.load_device)
        ) as span:
            parent = None if span is None else span.span_id
            with audio_handle.stage(observer_stage="decode", parent_span_id=parent):
                with torch.inference_mode():
                    audio = audio_runtime.decode_audio(audio_latent.to(audio_handle.load_device))
        if (
            type(video) is not torch.Tensor
            or video.ndim != 5
            or tuple(video.shape[:2]) != (1, 3)
            or not video.is_floating_point()
            or video.layout != torch.strided
        ):
            raise ValueError("MiniMax H3 video decode must return [1,3,time,height,width]")
        if type(audio) is not inference.MiniMaxH3AudioContent:
            raise TypeError("MiniMax H3 audio decode must return MiniMaxH3AudioContent")
        waveform = audio.waveform
        if (
            type(waveform) is not torch.Tensor
            or waveform.ndim != 3
            or tuple(waveform.shape[:2]) != (1, 2)
            or not waveform.is_floating_point()
            or waveform.layout != torch.strided
        ):
            raise ValueError("MiniMax H3 audio decode must return [1,2,samples]")
        if type(audio.sample_rate) is not int or audio.sample_rate <= 0:
            raise ValueError("MiniMax H3 audio decode must return a positive integer sample rate")
        return cls.outputs(
            frames=video[0].permute(1, 2, 3, 0).contiguous(),
            audio={"waveform": waveform, "sample_rate": audio.sample_rate},
        )


def _latent_samples(value: object, torch: Any, inference: Any, name: str) -> tuple[Any, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a LATENT mapping")
    samples = cast("Mapping[object, object]", value).get("samples")
    if type(samples) is torch.Tensor:
        return samples, None
    if type(samples) is inference.MultiStreamLatent:
        return samples, samples
    raise TypeError(f"{name} samples must be a tensor or MultiStreamLatent")


def _av_stream(value: object, torch: Any, inference: Any, role: str, name: str) -> Any:
    samples, streams = _latent_samples(value, torch, inference, name)
    payload = samples if streams is None else streams.by_role(role)
    if type(payload) is not torch.Tensor or not payload.is_floating_point():
        raise TypeError(f"{name} {role} stream must be an exact floating torch.Tensor")
    return payload


def _fit_audio_stream(
    value: Any, target: Any, torch: Any, name: str, *, pad_value: float = 0.0
) -> Any:
    if value.ndim != target.ndim or tuple(value.shape[:-1]) != tuple(target.shape[:-1]):
        raise ValueError(
            f"{name} must match the existing audio rank, batch, and channel dimensions"
        )
    target_length = target.shape[-1]
    if value.shape[-1] > target_length:
        return value[..., :target_length]
    if value.shape[-1] < target_length:
        return torch.nn.functional.pad(value, (0, target_length - value.shape[-1]), value=pad_value)
    return value


def _role_mask(
    value: object, streams: Any, role: str, payload: Any, torch: Any, inference: Any
) -> Any:
    mask = cast("Mapping[object, object]", value).get("noise_mask")
    if mask is None:
        return None
    if type(mask) is torch.Tensor:
        if streams is None or streams.roles[0] == role:
            return mask
        return torch.ones_like(payload)
    if type(mask) is inference.MultiStreamLatent:
        structural_mask = cast("Any", mask)
        if role in structural_mask.roles:
            return structural_mask.by_role(role)
        return torch.ones_like(payload)
    raise TypeError("noise_mask must be a tensor or MultiStreamLatent")


def _latent_mask_target(
    value: object,
    mapping: Any,
    torch: Any,
    inference: Any,
    name: str,
) -> tuple[dict[object, object], Any, Any, bool]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a LATENT mapping")
    metadata = dict(cast("Mapping[object, object]", value))
    samples = cast("Any", metadata.get("samples"))
    structural = type(samples) is inference.MultiStreamLatent
    streams: Any = (
        samples
        if structural
        else inference.MultiStreamLatent.from_pairs(((mapping.role, samples),))
    )
    if type(streams) is not inference.MultiStreamLatent or mapping.role not in streams.roles:
        raise ValueError(f"{name} does not contain the codec's {mapping.role!r} role")
    target = streams.by_role(mapping.role)
    if (
        type(target) is not torch.Tensor
        or not target.is_floating_point()
        or target.layout != torch.strided
    ):
        raise TypeError(f"{name} {mapping.role} role must be an exact strided floating tensor")
    return metadata, streams, target, structural


def _mask_has_role(mask: object, streams: Any, role: str, torch: Any, inference: Any) -> bool:
    if type(mask) is torch.Tensor:
        return streams.roles[0] == role
    if type(mask) is inference.MultiStreamLatent:
        return role in cast("Any", mask).roles
    if mask is None:
        return False
    raise TypeError("noise_mask must be a tensor or MultiStreamLatent")


def _set_latent_role_mask(
    metadata: dict[object, object],
    streams: Any,
    role: str,
    role_mask: Any,
    operation: str,
    structural: bool,
    torch: Any,
    inference: Any,
    inference_torch: Any,
) -> Mapping[object, object]:
    if operation not in ("replace", "max", "min", "multiply"):
        raise ValueError("mask operation must be replace, max, min, or multiply")
    new_masks = inference_torch.normalize_latent_mask(
        inference.MultiStreamLatent.from_pairs(((role, role_mask),)),
        streams,
    )
    existing = metadata.get("noise_mask")
    has_existing_role = _mask_has_role(existing, streams, role, torch, inference)
    normalized = (
        new_masks if existing is None else inference_torch.normalize_latent_mask(existing, streams)
    )
    selected = new_masks.by_role(role)
    if operation != "replace" and has_existing_role:
        current = normalized.by_role(role)
        if operation == "max":
            selected = torch.maximum(current, selected)
        elif operation == "min":
            selected = torch.minimum(current, selected)
        else:
            selected = current * selected
    masks = inference.MultiStreamLatent.from_pairs(
        (stream_role, selected if stream_role == role else normalized.by_role(stream_role))
        for stream_role in streams.roles
    )
    metadata["noise_mask"] = masks if structural else masks.by_role(role)
    return metadata


class NativeSetLatentMaskFromFrames(SetLatentMaskFromFrames):
    @classmethod
    def execute(
        cls,
        *,
        latent: object,
        vae: object,
        mask: object,
        spatial_reduction: str,
        temporal_reduction: str,
        operation: str,
    ) -> Mapping[str, object]:
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        runtime = _latent_mask_codec_runtime(vae, "vae")
        mapping = runtime.latent_mask_mapping
        metadata, streams, target, structural = _latent_mask_target(
            latent, mapping, torch, inference, "latent"
        )
        role_mask = inference_torch.content_mask_to_latent_mask(
            mask,
            target,
            mapping,
            spatial_reduction=spatial_reduction,
            temporal_reduction=temporal_reduction,
        )
        output = _set_latent_role_mask(
            metadata,
            streams,
            mapping.role,
            role_mask,
            operation,
            structural,
            torch,
            inference,
            inference_torch,
        )
        return cls.outputs(latent=output)


class NativeSetLatentMaskFromTimeRanges(SetLatentMaskFromTimeRanges):
    @classmethod
    def execute(
        cls,
        *,
        latent: object,
        vae: object,
        ranges: str,
        selected: float,
        unselected: float,
        operation: str,
    ) -> Mapping[str, object]:
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        runtime = _latent_mask_codec_runtime(vae, "vae")
        mapping = runtime.latent_mask_mapping
        metadata, streams, target, structural = _latent_mask_target(
            latent, mapping, torch, inference, "latent"
        )
        parsed = inference.parse_time_ranges(
            ranges,
            mapping.duration_seconds(target.shape[-1]),
        )
        role_mask = inference_torch.time_ranges_to_latent_mask(
            target,
            mapping,
            parsed,
            selected=selected,
            unselected=unselected,
        )
        output = _set_latent_role_mask(
            metadata,
            streams,
            mapping.role,
            role_mask,
            operation,
            structural,
            torch,
            inference,
            inference_torch,
        )
        return cls.outputs(latent=output)


def _mask_report(
    role_mask: Any,
    target: Any,
    mapping: Any,
    source: str,
) -> str:
    time_axis = 2 if mapping.spatial_downscale is not None else target.ndim - 1
    dimensions = tuple(index for index in range(role_mask.ndim) if index != time_axis)
    maximum = role_mask.amax(dim=dimensions)
    minimum = role_mask.amin(dim=dimensions)
    values = tuple(float(value) for value in maximum)
    varied = any(float(low) != high for low, high in zip(minimum, values, strict=True))
    groups = mapping.content_ranges(len(values))
    runs: list[tuple[int, int, float]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            runs.append((start, index, values[start]))
            start = index

    def value_text(value: float) -> str:
        text = f"{value:.6g}"
        label = "keep" if value == 0.0 else "generate" if value == 1.0 else "soft"
        return f"{text} ({label})"

    def time_text(value: float) -> str:
        return f"{value:.4f}".rstrip("0").rstrip(".")

    intervals = ", ".join(
        f"{time_text(groups[start][0] / mapping.content_rate_hz)}-"
        f"{time_text(groups[stop - 1][1] / mapping.content_rate_hz)}s = {value_text(value)}"
        for start, stop, value in runs
    )
    report = (
        f"{source}\n"
        f"{mapping.role}: {len(values)} latent frames, "
        f"{time_text(mapping.duration_seconds(len(values)))}s at "
        f"{mapping.content_rate_hz:g} content frames/s\n"
        f"{intervals}"
    )
    if varied:
        report += "\nwarning: values vary within timeline frames; reporting each frame's maximum"
    return report


class NativeInspectLatentMask(InspectLatentMask):
    @classmethod
    def execute(cls, *, latent: object, vae: object) -> Mapping[str, object]:
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        runtime = _latent_mask_codec_runtime(vae, "vae")
        mapping = runtime.latent_mask_mapping
        metadata, streams, target, _structural = _latent_mask_target(
            latent, mapping, torch, inference, "latent"
        )
        raw_mask = metadata.get("noise_mask")
        has_role = _mask_has_role(raw_mask, streams, mapping.role, torch, inference)
        if raw_mask is None:
            source = f"no noise mask on the latent; {mapping.role} defaults to generate"
            role_mask = torch.ones_like(target, dtype=torch.float32)
        elif not has_role:
            source = f"noise mask has no {mapping.role} role; it defaults to generate"
            role_mask = torch.ones_like(target, dtype=torch.float32)
        else:
            source = f"{mapping.role} role mask"
            role_mask = inference_torch.normalize_latent_mask(raw_mask, streams).by_role(
                mapping.role
            )
        if not bool(torch.isfinite(role_mask).all()):
            raise ValueError("latent mask values must be finite")
        if float(role_mask.amin()) < 0.0 or float(role_mask.amax()) > 1.0:
            raise ValueError("latent mask values must be within [0, 1]")
        preview = (
            inference_torch.latent_mask_to_content_mask(role_mask, target, mapping)
            if mapping.spatial_downscale is not None
            else inference_torch.latent_mask_preview(role_mask, target)
        )
        return cls.outputs(
            mask=preview,
            report=_mask_report(role_mask, target, mapping, source),
        )


class NativeConcatAVLatent(ConcatAVLatent):
    @classmethod
    def execute(cls, *, video_latent: object, audio_latent: object) -> Mapping[str, object]:
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        video_samples, video_streams = _latent_samples(
            video_latent, torch, inference, "video_latent"
        )
        if video_streams is not None and video_streams.roles not in (
            ("video",),
            ("video", "audio"),
        ):
            raise ValueError("video_latent must be single video or exact video/audio streams")
        video = video_samples if video_streams is None else video_streams.by_role("video")
        audio_samples, audio_streams = _latent_samples(
            audio_latent, torch, inference, "audio_latent"
        )
        if audio_streams is not None and audio_streams.roles != ("audio",):
            raise ValueError("audio_latent must be a single audio stream")
        audio = audio_samples if audio_streams is None else audio_streams.by_role("audio")
        if type(video) is not torch.Tensor or not video.is_floating_point():
            raise TypeError("video_latent video stream must be an exact floating torch.Tensor")
        if type(audio) is not torch.Tensor or not audio.is_floating_point():
            raise TypeError("audio_latent audio stream must be an exact floating torch.Tensor")
        if video.shape[0] != audio.shape[0]:
            raise ValueError("video and audio streams must have the same batch size")
        if video_streams is not None and "audio" in video_streams.roles:
            audio = _fit_audio_stream(audio, video_streams.by_role("audio"), torch, "audio_latent")
        video_mask = _role_mask(video_latent, video_streams, "video", video, torch, inference)
        source_audio = audio_samples if audio_streams is None else audio_streams.by_role("audio")
        audio_mask = _role_mask(
            audio_latent, audio_streams, "audio", source_audio, torch, inference
        )
        inference_torch = importlib.import_module("dinkster_inference_torch")
        if video_mask is not None:
            video_mask = inference_torch.normalize_latent_mask(
                video_mask,
                inference.MultiStreamLatent.from_pairs((("video", video),)),
            ).by_role("video")
        if audio_mask is not None:
            audio_mask = inference_torch.normalize_latent_mask(
                audio_mask,
                inference.MultiStreamLatent.from_pairs((("audio", source_audio),)),
            ).by_role("audio")
            if video_streams is not None and "audio" in video_streams.roles:
                audio_mask = _fit_audio_stream(
                    audio_mask,
                    video_streams.by_role("audio"),
                    torch,
                    "audio noise_mask",
                    pad_value=1.0,
                )
        output = dict(cast("Mapping[object, object]", video_latent))
        output.update(cast("Mapping[object, object]", audio_latent))
        output["samples"] = inference.MultiStreamLatent.from_pairs(
            (("video", video), ("audio", audio))
        )
        if video_mask is not None or audio_mask is not None:
            output["noise_mask"] = inference.MultiStreamLatent.from_pairs(
                (
                    ("video", torch.ones_like(video) if video_mask is None else video_mask),
                    ("audio", torch.ones_like(audio) if audio_mask is None else audio_mask),
                )
            )
        else:
            output.pop("noise_mask", None)
        return cls.outputs(latent=output)


class NativeSeparateAVLatent(SeparateAVLatent):
    @classmethod
    def execute(cls, *, latent: object) -> Mapping[str, object]:
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        _, streams = _latent_samples(latent, torch, inference, "latent")
        if streams is None or streams.roles != ("video", "audio"):
            raise TypeError("latent must contain exact ordered video/audio streams")
        metadata = dict(cast("Mapping[object, object]", latent))
        video = dict(metadata)
        audio = dict(metadata)
        video["samples"] = streams.by_role("video")
        audio["samples"] = streams.by_role("audio")
        if metadata.get("noise_mask") is not None:
            masks = importlib.import_module("dinkster_inference_torch").normalize_latent_mask(
                metadata["noise_mask"], streams
            )
            video["noise_mask"] = masks.by_role("video")
            audio["noise_mask"] = masks.by_role("audio")
        return cls.outputs(video_latent=video, audio_latent=audio)


def _preview_stream(
    model: object, latent: object, role: str
) -> tuple[NativeRuntimeHandle, Any, Any, Any]:
    handle = _native_handle(model, "model")
    torch = _torch()
    inference = importlib.import_module("dinkster_inference")
    if not isinstance(handle.runtime, inference.MultiStreamPreviewRuntime):
        raise TypeError("model does not provide multi-stream preview codecs")
    payload = _av_stream(latent, torch, inference, role, "latent")
    return handle, torch, inference, payload


class NativePreviewLatentVisual(PreviewLatentVisual):
    @classmethod
    def execute(cls, *, model: object, latent: object, role: str) -> Mapping[str, object]:
        handle, torch, _, payload = _preview_stream(model, latent, role)
        with native_execution_span(
            "decode", "decode_video", device=str(handle.load_device)
        ) as span:
            parent = None if span is None else span.span_id
            with handle.stage("vae", observer_stage="decode", parent_span_id=parent):
                with torch.inference_mode():
                    visual = handle.runtime.preview_visual(role, payload.to(handle.load_device))
        if type(visual) is not torch.Tensor or not visual.is_floating_point():
            raise TypeError("visual stream preview must return a floating torch.Tensor")
        if visual.ndim == 5 and visual.shape[1] == 3:
            image = visual.permute(0, 2, 3, 4, 1).flatten(0, 1)
        elif visual.ndim == 4 and visual.shape[1] == 3:
            image = visual.permute(0, 2, 3, 1)
        else:
            raise ValueError("visual stream preview must return [B,3,H,W] or [B,3,T,H,W]")
        return cls.outputs(image=image)


class NativePreviewLatentAudio(PreviewLatentAudio):
    @classmethod
    def execute(cls, *, model: object, latent: object, role: str) -> Mapping[str, object]:
        handle, torch, inference, payload = _preview_stream(model, latent, role)
        with native_execution_span(
            "decode", "decode_audio", device=str(handle.load_device)
        ) as span:
            parent = None if span is None else span.span_id
            with handle.stage("vae", observer_stage="decode", parent_span_id=parent):
                with torch.inference_mode():
                    audio = handle.runtime.preview_audio(role, payload.to(handle.load_device))
        if type(audio) is not inference.AudioPreview:
            raise TypeError("audio stream preview must return AudioPreview")
        return cls.outputs(audio={"waveform": audio.waveform, "sample_rate": audio.sample_rate})


def _prepared_multistream_carrier(value: object, inference: Any, name: str) -> Any | None:
    if value == []:
        return None
    entries = cast("list[object]", value) if type(value) is list else []
    entry = (
        cast("list[object]", entries[0]) if len(entries) == 1 and type(entries[0]) is list else []
    )
    if (
        len(entry) != 2
        or type(entry[0]) is not inference.PreparedMultiStreamConditioning
        or entry[1] != {}
    ):
        raise TypeError(f"{name} must contain exact prepared multi-stream conditioning")
    return cast("Any", entry[0])


def _prepared_multistream_conditioning(
    value: object, inference: Any, name: str, runtime_identity: str
) -> object | None:
    prepared = _prepared_multistream_carrier(value, inference, name)
    if prepared is None:
        return None
    if prepared.runtime_identity != runtime_identity:
        raise ValueError(f"{name} conditioning was prepared by a different runtime")
    return cast("object", prepared.payload)


def _minimax_h3_conditioning_carrier(value: object, inference: Any, name: str) -> Any | None:
    prepared = _prepared_multistream_carrier(value, inference, name)
    if prepared is None:
        return None
    try:
        inference.ComponentBinding(
            "conditioner",
            inference.MINIMAX_H3_CONFIG.family_id,
            prepared.runtime_identity,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} conditioning was not prepared by an official MiniMax H3 conditioner component"
        ) from error
    return prepared


def _minimax_h3_model_handle(
    value: object,
    inference: Any,
) -> NativeRuntimeHandle | None:
    if isinstance(value, _NativeModelOverlay):
        value = value.handle
    if not isinstance(value, NativeRuntimeHandle):
        return None
    model = cast("object", value.runtime)
    recipe = value.recipe
    model_type = type(model)
    exact_type_name = (
        model_type.__module__ == "dinkster_inference_torch.minimax_h3_assembly"
        and model_type.__name__ == "MiniMaxH3Model"
    )
    if recipe.family_id != inference.MINIMAX_H3_CONFIG.family_id and not exact_type_name:
        return None
    inference_torch = importlib.import_module("dinkster_inference_torch")
    if type(model) is not inference_torch.MiniMaxH3Model:
        return None
    if (
        recipe.family_id != inference.MINIMAX_H3_CONFIG.family_id
        or tuple(binding.role for binding in recipe.sources) != ("diffusion",)
        or recipe.runtime_identity != cast("Any", model).runtime_identity
    ):
        raise TypeError("MiniMax H3 sampling requires a standalone DiT component model")
    return value


def _minimax_h3_dit_runtime(
    handle: NativeRuntimeHandle,
    conditioner_identity: str,
    inference: Any,
    torch: Any,
) -> Any:
    inference_torch = importlib.import_module("dinkster_inference_torch")
    validated = _minimax_h3_model_handle(handle, inference)
    if validated is None:
        raise TypeError("MiniMax H3 sampling requires a standalone DiT component model")
    model = validated.runtime
    recipe = validated.recipe
    model_role = model.model_role.replace("-", "_")
    composition = inference.compose_execution(
        inference.MINIMAX_H3_CONFIG.family_id,
        {
            model_role: recipe.runtime_identity,
            "conditioner": conditioner_identity,
        },
    )
    return inference_torch.MiniMaxH3DiTRuntime(
        model.assembled.diffusion,
        model_role=model_role,
        runtime_identity=composition.execution_identity,
        receipt_identity=model.receipt_identity,
        compute_dtype=_torch_dtype(torch, recipe.knobs.diffusion_dtype),
        conditioning_identity=conditioner_identity,
    )


def _minimax_h3_custom_sampling_runtime(
    handle: NativeRuntimeHandle,
    positive: object,
    negative: object | None,
    inference: Any,
) -> Any | None:
    """The H3 DiT runtime for a decomposed sampling invocation, or None
    for non-H3 handles."""
    if _minimax_h3_model_handle(handle, inference) is None:
        return None
    carrier = _minimax_h3_conditioning_carrier(positive, inference, "positive")
    if carrier is None:
        raise TypeError("positive must contain prepared MiniMax H3 conditioning")
    if negative not in ([], None):
        negative_carrier = _minimax_h3_conditioning_carrier(negative, inference, "negative")
        if (
            negative_carrier is not None
            and negative_carrier.runtime_identity != carrier.runtime_identity
        ):
            raise ValueError(
                "negative conditioning was prepared by a different MiniMax H3 conditioner component"
            )
    return _minimax_h3_dit_runtime(handle, carrier.runtime_identity, inference, _torch())


def resolve_minimax_h3_component_execution(
    handle: NativeRuntimeHandle, positive: object, negative: object, inference: Any
) -> tuple[Any, object, object] | None:
    runtime = _minimax_h3_custom_sampling_runtime(handle, positive, negative, inference)
    return None if runtime is None else (runtime, positive, negative)


def _minimax_h3_schedule_runtime(handle: NativeRuntimeHandle, inference: Any) -> Any | None:
    """The H3 DiT runtime for schedule-only queries, or None for non-H3
    handles. No conditioner is in scope, so the runtime keeps the model's
    own identity instead of a composed execution identity."""
    validated = _minimax_h3_model_handle(handle, inference)
    if validated is None:
        return None
    inference_torch = importlib.import_module("dinkster_inference_torch")
    model = validated.runtime
    recipe = validated.recipe
    return inference_torch.MiniMaxH3DiTRuntime(
        model.assembled.diffusion,
        model_role=model.model_role.replace("-", "_"),
        runtime_identity=recipe.runtime_identity,
        receipt_identity=model.receipt_identity,
        compute_dtype=_torch_dtype(_torch(), recipe.knobs.diffusion_dtype),
    )


@dataclass(frozen=True, slots=True)
class _ResolvedComponentExecution:
    runtime: Any
    positive: object
    negative: object
    conditioning_prepared: bool

    def values(self) -> tuple[Any, object, object]:
        return self.runtime, self.positive, self.negative


def _component_runtime_with_options(
    base_runtime: Any,
    descriptor: Any,
    family_id: str,
    runtime_identity: str,
    compute_dtype: Any,
    sampling_shift: float | None,
    option_windows: tuple[Any, ...],
) -> Any:
    from dinkster_inference.component_registry import execution_symbol

    sampling_runtime = getattr(base_runtime, "component_sampling_runtime", base_runtime)
    runtime_matches = any(
        isinstance(sampling_runtime, execution_symbol(reference))
        for reference in (descriptor.runtime_class, *descriptor.runtime_variants)
    )
    label = descriptor.family.display_name
    if not runtime_matches or base_runtime.runtime_identity != sampling_runtime.runtime_identity:
        raise TypeError(f"model must be a native {label} diffusion component")
    if sampling_runtime.family.id != family_id:
        raise ValueError(
            f"runtime producer family {sampling_runtime.family.id!r} does not match "
            f"reconstruction recipe family {family_id!r}"
        )
    runtime_options = descriptor.execution_options(sampling_runtime, sampling_shift, option_windows)
    assembled = getattr(sampling_runtime, "assembled", None)
    module = sampling_runtime.model if assembled is None else assembled.diffusion
    with_execution_options = getattr(sampling_runtime, "with_execution_options", None)
    if callable(with_execution_options):
        runtime = with_execution_options(
            runtime_identity=runtime_identity,
            compute_dtype=compute_dtype,
            **runtime_options,
        )
    elif descriptor.runtime_with_family:
        runtime = type(sampling_runtime)(
            module,
            sampling_runtime.family,
            runtime_identity=runtime_identity,
            **runtime_options,
        )
    else:
        runtime = type(sampling_runtime)(
            module,
            runtime_identity=runtime_identity,
            compute_dtype=compute_dtype,
            **runtime_options,
        )
    if sampling_runtime is base_runtime:
        return runtime
    replace_runtime = getattr(base_runtime, "with_component_sampling_runtime", None)
    if not callable(replace_runtime):
        raise TypeError(f"native {label} checkpoint cannot replace its sampling runtime")
    return replace_runtime(runtime)


def _resolve_component_execution(
    handle: NativeRuntimeHandle,
    positive: object,
    negative: object,
    inference: Any,
    *,
    sampling_shift: float | None = None,
    option_windows: tuple[Any, ...] = (),
    negative_handle: NativeRuntimeHandle | None = None,
    image_only_negative: bool = False,
) -> _ResolvedComponentExecution | None:
    from dinkster_inference.component_registry import execution_symbol

    recipe = handle.recipe
    descriptor = _active_inference_registries().components.get(recipe.family_id)
    if descriptor is None:
        return None
    execution_options: dict[str, Any] = {}
    if negative_handle is not None:
        execution_options["negative_handle"] = negative_handle
    if image_only_negative:
        execution_options["image_only_negative"] = True
    if descriptor.execution_resolver is not None:
        execution = execution_symbol(descriptor.execution_resolver)(
            handle,
            positive,
            negative,
            inference,
            **execution_options,
        )
        if execution is not None:
            return _ResolvedComponentExecution(*execution, conditioning_prepared=True)
        if execution_options:
            raise TypeError("runtime does not support separate-model or image-only guidance")
        log.warning(
            "component execution resolver declined registered descriptor %s; "
            "using runtime without preparing conditioning",
            descriptor.id,
        )
        return _ResolvedComponentExecution(
            handle.runtime,
            positive,
            negative,
            conditioning_prepared=False,
        )
    if execution_options:
        raise ValueError(
            "component runtime does not implement separate negative-model conditioning"
        )
    if tuple(source.role for source in recipe.sources) != (descriptor.model_role,):
        runtime = handle.runtime
        if descriptor.execution_options is not None:
            sampling_runtime = getattr(runtime, "component_sampling_runtime", runtime)
            runtime = _component_runtime_with_options(
                runtime,
                descriptor,
                recipe.family_id,
                recipe.runtime_identity,
                sampling_runtime.assembled.compute_dtype("diffusion"),
                sampling_shift,
                option_windows,
            )
        return _ResolvedComponentExecution(
            runtime,
            positive,
            negative,
            conditioning_prepared=False,
        )
    base_runtime = handle.runtime
    runtime_matches = any(
        isinstance(base_runtime, execution_symbol(reference))
        for reference in (descriptor.runtime_class, *descriptor.runtime_variants)
    )
    label = descriptor.family.display_name
    if not runtime_matches or recipe.runtime_identity != base_runtime.runtime_identity:
        raise TypeError(f"model must be a native {label} diffusion component")
    if base_runtime.family.id != recipe.family_id:
        raise ValueError(
            f"runtime producer family {base_runtime.family.id!r} does not match "
            f"reconstruction recipe family {recipe.family_id!r}"
        )
    positive_carrier, positive_binding = _component_bound_carrier(positive, inference)
    if positive_binding is None:
        if descriptor.allow_unbound_conditioning:
            return _ResolvedComponentExecution(
                base_runtime,
                positive,
                negative,
                conditioning_prepared=False,
            )
        raise TypeError(f"positive must be {label} component-bound conditioning")
    conditioning_families = (recipe.family_id, *descriptor.shared_conditioning_families)
    if positive_binding.family_id not in conditioning_families:
        raise ValueError(f"positive {label} conditioning has the wrong component family")
    conditioning_roles = descriptor.conditioning_roles or descriptor.text_encoder_roles
    if positive_binding.role not in conditioning_roles:
        raise ValueError(f"positive {label} conditioning has the wrong component role")
    negative_carrier = None
    if negative not in ([], None):
        negative_carrier, negative_binding = _component_bound_carrier(negative, inference)
        if negative_binding is None:
            raise TypeError(f"negative must be {label} component-bound conditioning or empty")
        if negative_binding.family_id not in conditioning_families:
            raise ValueError(f"negative {label} conditioning has the wrong component family")
        if negative_binding.role != positive_binding.role:
            raise ValueError(f"negative {label} conditioning has the wrong component role")
        if negative_binding != positive_binding:
            raise ValueError(f"{label} conditioning lanes must share one component binding")
    composition = inference.compose_execution(
        recipe.family_id,
        {
            descriptor.model_role: recipe.runtime_identity,
            positive_binding.role: positive_binding.identity,
        },
        shared_component_families=frozenset(descriptor.shared_conditioning_families),
    )
    if descriptor.execution_options is None:
        assembled = getattr(base_runtime, "assembled", None)
        module = base_runtime.model if assembled is None else assembled.diffusion
        if descriptor.runtime_with_family:
            runtime = type(base_runtime)(
                module,
                base_runtime.family,
                runtime_identity=composition.execution_identity,
            )
        else:
            runtime = type(base_runtime)(
                module,
                runtime_identity=composition.execution_identity,
                compute_dtype=_torch_dtype(_torch(), recipe.knobs.diffusion_dtype),
            )
    else:
        runtime = _component_runtime_with_options(
            base_runtime,
            descriptor,
            recipe.family_id,
            composition.execution_identity,
            _torch_dtype(_torch(), recipe.knobs.diffusion_dtype),
            sampling_shift,
            option_windows,
        )

    def prepare(carrier: object) -> object:
        prepare_method = getattr(runtime, descriptor.prepare_conditioning, None)
        prepare_options: dict[str, object] = {}
        if descriptor.frame_rate_conditioning:
            carrier, frame_rate = _split_ltx_frame_rate(carrier, inference)
            if frame_rate is not None:
                prepare_options["frame_rate"] = frame_rate
        conditioning = (
            execution_symbol(descriptor.prepare_conditioning)(carrier, device=handle.load_device)
            if prepare_method is None
            else prepare_method(carrier, **prepare_options)
        )
        if descriptor.conditioning_format == "raw":
            return conditioning
        if descriptor.conditioning_format == "multistream":
            prepared: list[list[Any]] = [
                [
                    inference.PreparedMultiStreamConditioning(
                        runtime.conditioning_identity, conditioning
                    ),
                    {},
                ]
            ]
            return prepared
        return [[conditioning.embeddings, {_NATIVE_PREPARED_CONDITIONING_KEY: conditioning}]]

    result: tuple[Any, object, object] = (
        runtime,
        prepare(positive_carrier),
        (
            (None if descriptor.conditioning_format == "raw" else [])
            if negative_carrier is None
            else prepare(negative_carrier)
        ),
    )
    if descriptor.release_conditioning:
        try:
            handle.coordinator.advisory_unload_components(positive_binding.identity)
        except NativeResidencyBusyError:
            pass
    return _ResolvedComponentExecution(*result, conditioning_prepared=True)


def resolve_component_execution(
    handle: NativeRuntimeHandle,
    positive: object,
    negative: object,
    inference: Any,
    *,
    sampling_shift: float | None = None,
    option_windows: tuple[Any, ...] = (),
    negative_handle: NativeRuntimeHandle | None = None,
    image_only_negative: bool = False,
) -> tuple[Any, object, object] | None:
    resolved = _resolve_component_execution(
        handle,
        positive,
        negative,
        inference,
        sampling_shift=sampling_shift,
        option_windows=option_windows,
        negative_handle=negative_handle,
        image_only_negative=image_only_negative,
    )
    return None if resolved is None else resolved.values()


def resolve_ideogram4_component_execution(
    handle: NativeRuntimeHandle,
    positive: object,
    negative: object,
    inference: Any,
    *,
    negative_handle: NativeRuntimeHandle | None = None,
    image_only_negative: bool = False,
) -> tuple[Any, object, object] | None:
    recipe = handle.recipe
    family_id = inference.IDEOGRAM4_CONFIG.family_id
    if recipe.family_id != family_id:
        return None
    if tuple(source.role for source in recipe.sources) != ("diffusion",):
        return None
    inference_torch = importlib.import_module("dinkster_inference_torch")
    base_runtime = handle.runtime
    if (
        not isinstance(base_runtime, inference_torch.Ideogram4DiffusionRuntime)
        or recipe.runtime_identity != base_runtime.runtime_identity
    ):
        raise TypeError("model must be a native Ideogram 4 diffusion component")
    negative_runtime = None
    if negative_handle is not None:
        negative_recipe = negative_handle.recipe
        negative_base = negative_handle.runtime
        if (
            negative_recipe.family_id != family_id
            or tuple(source.role for source in negative_recipe.sources) != ("diffusion",)
            or not isinstance(negative_base, inference_torch.Ideogram4DiffusionRuntime)
            or negative_recipe.runtime_identity != negative_base.runtime_identity
        ):
            raise TypeError("model_negative must be a native Ideogram 4 diffusion component")
    positive_carrier, positive_binding = _component_bound_carrier(positive, inference)
    if positive_binding is None:
        raise TypeError("positive must be Ideogram 4 component-bound conditioning")
    if positive_binding.family_id != family_id or positive_binding.role != "qwen3vl_8b":
        raise ValueError("positive Ideogram 4 conditioning has the wrong component binding")
    negative_carrier = None
    if negative not in ([], None):
        negative_carrier, negative_binding = _component_bound_carrier(negative, inference)
        if negative_binding is None:
            raise TypeError("negative must be Ideogram 4 component-bound conditioning or empty")
        if negative_binding != positive_binding:
            raise ValueError("Ideogram 4 conditioning lanes must share one component binding")
    components = {
        "diffusion": recipe.runtime_identity,
        "qwen3vl_8b": positive_binding.identity,
    }
    if negative_handle is not None:
        components["negative-diffusion"] = negative_handle.recipe.runtime_identity
    composition = inference.compose_execution(family_id, components)
    torch = _torch()
    runtime = inference_torch.Ideogram4DiffusionRuntime(
        base_runtime.assembled.diffusion,
        runtime_identity=composition.execution_identity,
        compute_dtype=_torch_dtype(torch, recipe.knobs.diffusion_dtype),
    )
    if negative_handle is not None:
        negative_runtime = inference_torch.Ideogram4DiffusionRuntime(
            negative_handle.runtime.assembled.diffusion,
            runtime_identity=composition.execution_identity,
            compute_dtype=_torch_dtype(torch, negative_handle.recipe.knobs.diffusion_dtype),
        )
    conditioning = runtime.prepare_single_stream_conditioning(positive_carrier)
    rows = [[conditioning.embeddings, {_NATIVE_PREPARED_CONDITIONING_KEY: conditioning}]]
    negative_rows: object = []
    target_runtime = runtime if negative_runtime is None else negative_runtime
    uncond = (
        target_runtime.image_only_conditioning()
        if image_only_negative
        else (
            None
            if negative_carrier is None
            else target_runtime.prepare_single_stream_conditioning(negative_carrier)
        )
    )
    if uncond is not None:
        if negative_runtime is not None:
            uncond = inference_torch.RoutedConditioning(
                embeddings=uncond.embeddings,
                pooled=uncond.pooled,
                evaluation=negative_runtime.conditioning_evaluation(),
                source=uncond,
            )
        negative_rows = [[uncond.embeddings, {_NATIVE_PREPARED_CONDITIONING_KEY: uncond}]]
    return runtime, rows, negative_rows


def resolve_seedvr2_component_execution(
    handle: NativeRuntimeHandle,
    positive: object,
    negative: object,
    inference: Any,
) -> tuple[Any, object, object] | None:
    recipe = handle.recipe
    runtime = handle.runtime
    sampling_runtime = getattr(runtime, "component_sampling_runtime", runtime)
    if getattr(sampling_runtime, "runtime_identity", None) != recipe.runtime_identity:
        raise TypeError("component sampling runtime identity does not match its model handle")
    torch = _torch()
    cond, inpaint = _conditioning(positive, "positive", torch, inference)
    if getattr(cond, "branch", None) != "positive" or not isinstance(
        getattr(cond, "component_identity", None), str
    ):
        raise TypeError("positive must come from Apply SeedVR2 Conditioning")
    if cond.component_identity != recipe.runtime_identity:
        raise ValueError("positive SeedVR2 conditioning belongs to a different model")
    if inpaint is not None:
        raise ValueError("SeedVR2 conditioning cannot contain inpaint metadata")
    if negative not in ([], None):
        uncond, uncond_inpaint = _conditioning(negative, "negative", torch, inference)
        if getattr(uncond, "branch", None) != "negative" or not isinstance(
            getattr(uncond, "component_identity", None), str
        ):
            raise TypeError("negative must come from Apply SeedVR2 Conditioning")
        if uncond.component_identity != recipe.runtime_identity:
            raise ValueError("negative SeedVR2 conditioning belongs to a different model")
        if uncond_inpaint is not None:
            raise ValueError("SeedVR2 conditioning cannot contain inpaint metadata")
    return sampling_runtime, positive, negative


def _sampling_memory_requirements(runtime: Any, samples: Any) -> tuple[int, int | None]:
    estimate = getattr(runtime, "sampling_memory_requirements", None)
    return (0, None) if estimate is None else estimate(tuple(samples.shape))


def resolve_trellis2_component_execution(
    handle: NativeRuntimeHandle, positive: object, negative: object, inference: Any
) -> tuple[Any, object, object] | None:
    recipe = handle.recipe
    if recipe.family_id != inference.TRELLIS2.id:
        return None
    source_roles = tuple(source.role for source in recipe.sources)
    if source_roles not in (
        ("diffusion",),
        ("shape", "shape-512", "structure", "texture", "texture-512"),
    ):
        return None
    inference_torch = importlib.import_module("dinkster_inference_torch")
    runtime = handle.runtime
    if (
        not isinstance(runtime, inference_torch.Trellis2DiffusionRuntime)
        or recipe.runtime_identity != runtime.runtime_identity
    ):
        raise TypeError("model must be a native TRELLIS.2 diffusion component")
    carrier_type = inference.ResidentConditioningCarrier
    resource_type = inference_torch.Trellis2ConditioningResource
    positive_carrier = cast("Any", positive)
    if not isinstance(positive, carrier_type) or not isinstance(
        positive_carrier.payload, resource_type
    ):
        raise TypeError("positive must be resident TRELLIS.2 conditioning")
    positive_resource = positive_carrier.payload
    if positive_resource.guidance_role is not inference.GuidanceRole.CONDITIONAL:
        raise ValueError("positive TRELLIS.2 conditioning has the wrong guidance lane")
    negative_resource = None
    if negative not in ([], None):
        negative_carrier = cast("Any", negative)
        if not isinstance(negative, carrier_type) or not isinstance(
            negative_carrier.payload, resource_type
        ):
            raise TypeError("negative must be resident TRELLIS.2 conditioning or empty")
        negative_resource = negative_carrier.payload
        if negative_resource.guidance_role is not inference.GuidanceRole.UNCONDITIONAL:
            raise ValueError("negative TRELLIS.2 conditioning has the wrong guidance lane")
        if not positive_resource.shares_backing(negative_resource):
            raise ValueError("TRELLIS.2 conditioning lanes must share one backing resource")
        if negative_resource.stage != positive_resource.stage:
            raise ValueError("TRELLIS.2 conditioning lanes must use the same stage")

    def rows(resource: object) -> list[list[object]]:
        return [
            [
                inference.PreparedMultiStreamConditioning(
                    runtime.conditioning_identity,
                    resource,
                ),
                dict[str, object](),
            ]
        ]

    return (
        runtime,
        rows(positive_resource),
        ([] if negative_resource is None else rows(negative_resource)),
    )


def load_registered_component(
    value: object,
    name: str,
    role: str | None = None,
    *,
    family_id: str | None = None,
) -> NativeComponentHandle:
    from dinkster_inference.component_registry import execution_symbol

    registry = importlib.import_module("dinkster_native.family_registry")
    if family_id is None:
        load = registry.registered_callable(value, "native_load")
    else:
        descriptor = _builtin_inference_registries().components.get(family_id)
        reference = None if descriptor is None else descriptor.native_load
        if reference is None:
            raise TypeError(f"no declared native_load for family={family_id!r}")
        load = execution_symbol(reference)
    return load(value, name, role)


class _RegisteredComponentCodec:
    def __init__(self, value: object, codec: Any) -> None:
        registry = importlib.import_module("dinkster_native.family_registry")
        self._codec = codec
        self._decode = registry.registered_callable(value, "native_decode")
        self._encode = registry.registered_callable(value, "native_encode")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._codec, name)

    def decode_latent(self, latent: Any) -> Any:
        return self._decode(self._codec, latent)

    def encode_content(self, content: Any) -> Any:
        return self._encode(self._codec, content)


def _native_component_codec(value: object) -> Any:
    from dinkster_inference.component_registry import execution_symbol

    recipe = getattr(value, "recipe", None)
    family_id = getattr(recipe, "family_id", None)
    descriptor = (
        None if family_id is None else _active_inference_registries().components.get(family_id)
    )
    if descriptor is None or descriptor.codec_adapter is None:
        roles = tuple(source.role for source in getattr(recipe, "sources", ()))
        raise TypeError(
            f"no declared codec adapter; detected family={family_id!r}, roles={roles!r}"
        )
    return _RegisteredComponentCodec(value, execution_symbol(descriptor.codec_adapter)(value))


def _z_image_control_latent(
    handle: NativeRuntimeHandle,
    binding: _ZImageControlBinding,
    samples: Any,
    torch: Any,
    inference_torch: Any,
) -> Any:
    image = cast("Any", binding.image)
    downscale = handle.runtime.assembled.vae.config.spatial_downscale
    target_height = samples.shape[-2] * downscale
    target_width = samples.shape[-1] * downscale
    if image.shape[-2:] != (target_height, target_width):
        old_height, old_width = image.shape[-2:]
        old_aspect = old_width / old_height
        new_aspect = target_width / target_height
        x = y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        image = image.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
        image = torch.nn.functional.interpolate(
            image,
            size=(target_height, target_width),
            mode="area",
        )
    with handle.stage("vae", observer_stage="encode"):
        with torch.inference_mode():
            encoded = handle.runtime.codec.encode(image.to(handle.load_device))
    return inference_torch.latent_process_in(encoded, handle.runtime.family.latent)


def _materialize_z_image_control(
    handle: NativeRuntimeHandle,
    binding: _ZImageControlBinding,
    samples: Any,
    torch: Any,
    inference: Any,
    inference_torch: Any,
) -> Any:
    control_latent = _z_image_control_latent(handle, binding, samples, torch, inference_torch)
    hint_digest = inference_torch.z_image_control_hint_digest(control_latent)
    control_model = cast("Any", binding.handle.module)
    model_digest = control_model.resource_digest
    if model_digest is None:
        raise RuntimeError("Z-Image control patch has no assembly provenance")
    return inference_torch.ZImageControlConditioning(
        inference.ControlApplication(
            "z-image-fun",
            inference.PayloadReference(hint_digest),
            binding.strength,
            inference.PercentRange(0.0, 1.0),
        ),
        control_model,
        control_latent,
        model_digest,
        hint_digest,
    )


def _normalize_empty_latent(samples: Any, runtime: object, torch: Any) -> Any:
    """Match ComfyUI's empty-latent channel and rank normalization."""

    family = getattr(runtime, "family", None)
    if family is None:
        return samples
    single_stream_latent = getattr(family, "single_stream_latent", None)
    if single_stream_latent is None:
        return samples
    try:
        descriptor = single_stream_latent()
    except ValueError:
        return samples
    empty = bool(torch.count_nonzero(samples) == 0)
    if empty and samples.shape[1] != descriptor.channels:
        if samples.shape[1] < 1:
            raise ValueError("empty latent must have at least one channel")
        repeats = [1] * samples.ndim
        repeats[1] = math.ceil(descriptor.channels / samples.shape[1])
        samples = samples.repeat(*repeats).narrow(1, 0, descriptor.channels)
    if descriptor.dimensions == 3 and samples.ndim == 4:
        samples = samples.unsqueeze(2)
    return samples


def _batch_index_noise_inds(latent: Mapping[object, object]) -> tuple[int, ...] | None:
    """Parse a latent's ``batch_index`` into prepare_noise batch indices."""

    batch_index = latent.get("batch_index")
    if batch_index is None:
        return None
    if not isinstance(batch_index, Sequence) or isinstance(batch_index, (str, bytes)):
        raise TypeError("latent_image['batch_index'] must be a sequence of integers")
    raw_noise_inds = cast("Sequence[object]", batch_index)
    if any(type(index) is not int or index < 0 for index in raw_noise_inds):
        raise ValueError("latent_image['batch_index'] values must be nonnegative integers")
    return tuple(cast("int", index) for index in raw_noise_inds)


def _resolve_sampling_model(
    handle: NativeRuntimeHandle,
    positive: object,
    negative: object,
    inference: Any,
    *,
    sampling_shift: float | None = None,
    option_windows: tuple[Any, ...] = (),
    negative_handle: NativeRuntimeHandle | None = None,
    image_only_negative: bool = False,
) -> tuple[Any, object, object, bool, bool]:
    resolved = _resolve_component_execution(
        handle,
        positive,
        negative,
        inference,
        sampling_shift=sampling_shift,
        option_windows=option_windows,
        negative_handle=negative_handle,
        image_only_negative=image_only_negative,
    )
    if resolved is None:
        if negative_handle is not None or image_only_negative:
            raise TypeError("runtime does not support separate-model or image-only guidance")
        return handle.runtime, positive, negative, False, False
    runtime, positive, negative = resolved.values()
    descriptor = _active_inference_registries().components.get(handle.recipe.family_id)
    if (
        resolved.conditioning_prepared
        and descriptor is not None
        and descriptor.conditioning_format == "raw"
    ):
        positive = [
            [
                inference.PreparedMultiStreamConditioning(runtime.conditioning_identity, positive),
                dict[str, object](),
            ]
        ]
        negative = (
            []
            if negative is None
            else [
                [
                    inference.PreparedMultiStreamConditioning(
                        runtime.conditioning_identity, negative
                    ),
                    dict[str, object](),
                ]
            ]
        )
    return runtime, positive, negative, True, resolved.conditioning_prepared


def _prepare_ksampler_conditioning(
    value: object, name: str, runtime: Any, handle: NativeRuntimeHandle, inference: Any
) -> object:
    if not isinstance(value, inference.ConditioningCarrier):
        return value
    if isinstance(runtime, inference.MultiStreamConditioningRuntime):
        return _prepare_provider_multistream_conditioning(value, name, runtime, inference)
    if isinstance(runtime, inference.ConditioningRuntime):
        return _prepare_provider_conditioning(value, name, runtime, inference)
    if isinstance(runtime, inference.MultiStreamFamilyRuntime):
        raise TypeError(f"{name} runtime cannot prepare canonical conditioning")
    return _materialize_provider_conditioning(value, name, handle)


class NativeKSampler(KSampler):
    """Sample through the runtime's builtin Comfy-compatible catalogs."""

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: object,
        negative: object,
        latent_image: object,
        denoise: float,
        segment: SamplingSegment | None = None,
        conditioning_batching: object | None = None,
    ) -> Mapping[str, object]:
        for name, value, low, high in (
            ("seed", seed, 0, cls.MAX_SEED),
            ("steps", steps, 1, cls.MAX_STEPS),
            ("cfg", cfg, 0.0, cls.MAX_CFG),
            ("denoise", denoise, 0.0, 1.0),
        ):
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")

        model, applications = _application_chain_model(model, "model")
        inference = importlib.import_module("dinkster_inference")
        if conditioning_batching is None:
            conditioning_batching = inference.ConditioningBatching()
        elif type(conditioning_batching) is not inference.ConditioningBatching:
            raise TypeError("conditioning_batching must be an exact ConditioningBatching")
        positive, positive_guidance = _split_flux_guidance(positive)
        negative, negative_guidance = _split_flux_guidance(negative)
        positive_controlled = _controlled_conditioning(positive)
        negative_controlled = _controlled_conditioning(negative)
        if positive_controlled is not None:
            positive = positive_controlled.conditioning
        if negative_controlled is not None:
            negative = negative_controlled.conditioning
        guidance = _effective_flux_guidance(positive_guidance, negative_guidance)
        (
            handle,
            ordinary_overlays,
            ordinary_resolvers,
            z_image_control,
            sampling_shift,
            guidance_transforms,
            context_windows,
            chroma_radiance_options,
        ) = _native_model(model, "model")
        guidance_transforms, disable_cfg1_optimization = _cfg1_optimization_setting(
            guidance_transforms
        )
        runtime, positive, negative, component_execution, conditioning_prepared = (
            _resolve_sampling_model(
                handle,
                positive,
                negative,
                inference,
                sampling_shift=sampling_shift,
                option_windows=chroma_radiance_options,
            )
        )
        component_runtime = runtime if component_execution else None
        if not conditioning_prepared:
            positive = _prepare_ksampler_conditioning(
                positive, "positive", runtime, handle, inference
            )
            negative = _prepare_ksampler_conditioning(
                negative, "negative", runtime, handle, inference
            )

        def restore_control(
            value: object,
            controlled: _ControlledConditioning | None,
            input_id: str,
        ) -> object:
            if controlled is None:
                return value
            result: list[list[object]] = []
            for raw_entry in _condition_entries(value, input_id):
                metadata = dict(cast("Mapping[object, object]", raw_entry[1]))
                metadata["control"] = controlled.binding
                metadata["control_apply_to_uncond"] = controlled.binding.apply_to_uncond
                result.append([raw_entry[0], metadata])
            return result

        positive = restore_control(positive, positive_controlled, "positive")
        negative = restore_control(negative, negative_controlled, "negative")
        latent_mapping = (
            cast("Mapping[object, object]", latent_image)
            if isinstance(latent_image, Mapping)
            else None
        )
        latent_samples = None if latent_mapping is None else latent_mapping.get("samples")
        structural_latent = type(latent_samples) is inference.MultiStreamLatent
        sparse_latent = type(latent_samples) is inference.SparseLatent
        torch = _torch()
        active_runtime = component_runtime if component_runtime is not None else handle.runtime
        runtime_family = getattr(getattr(active_runtime, "family", None), "id", None)
        unload_text_before_diffusion = (
            ()
            if component_execution
            or tuple(source.role for source in handle.recipe.sources) == ("diffusion",)
            else ("text",)
        )
        classic_control_binding = _select_classic_control_binding(positive, negative)
        if classic_control_binding is not None:
            if z_image_control is not None:
                raise ValueError("classic and Z-Image ControlNet cannot be applied together")
        plain_latent = not structural_latent
        if latent_mapping is not None and isinstance(
            active_runtime,
            (inference.MultiStreamLatentAdapterRuntime, inference.MultiStreamFamilyRuntime),
        ):
            latent_mapping = _adapt_multistream_latent(
                latent_mapping, active_runtime, torch, inference, "latent_image"
            )
            latent_samples = latent_mapping["samples"]
            structural_latent = type(latent_samples) is inference.MultiStreamLatent
        if (
            latent_mapping is not None
            and structural_latent
            and isinstance(active_runtime, inference.MultiStreamFamilyRuntime)
        ):
            runtime = active_runtime
            if ordinary_overlays or ordinary_resolvers or z_image_control is not None:
                raise TypeError("multi-stream sampling does not accept model overlays")
            noise_inds = _batch_index_noise_inds(latent_mapping)
            noise_kwargs = {} if noise_inds is None else {"noise_inds": noise_inds}
            runtime_sampling_shift = _runtime_sampling_shift(runtime, sampling_shift)
            context_windows_kwargs = _context_windows_sampling_kwargs(
                runtime, context_windows, f"{runtime_family!r} multi-stream sampling"
            )
            plain_output = plain_latent and len(cast("Any", latent_samples).roles) == 1
            denoise_mask = latent_mapping.get("noise_mask")
            if denoise_mask is not None:
                if type(denoise_mask) is torch.Tensor:
                    if not plain_output:
                        denoise_mask = cast("Any", denoise_mask).to(handle.load_device)
                elif type(denoise_mask) is inference.MultiStreamLatent:
                    if not plain_output:
                        denoise_mask = _move_multistream_latent(denoise_mask, handle.load_device)
                else:
                    raise TypeError("multi-stream noise_mask must be a tensor or MultiStreamLatent")
            prepared = _prepared_multistream_conditioning(
                positive, inference, "positive", runtime.conditioning_identity
            )
            if prepared is None:
                raise TypeError("positive must contain prepared multi-stream conditioning")
            uncond = _prepared_multistream_conditioning(
                negative, inference, "negative", runtime.conditioning_identity
            )
            context = current_execution_context()
            sampler_registry, extension_ids, _ = _sampler_registry(
                inference,
                context.extension_snapshot_digest if context is not None else None,
            )
            sampler_id = _catalog_id(sampler_registry, sampler_name, "sampler")
            scheduler_id = _catalog_id(
                _inference_registries(inference).schedulers, scheduler, "scheduler"
            )
            check_custom_sampling = getattr(runtime, "check_custom_sampling", None)
            if check_custom_sampling is not None:
                check_custom_sampling(
                    inference.CustomSamplingRequest(sampler_registry.get(sampler_id), (), ()),
                    has_denoise_mask=denoise_mask is not None,
                    has_inpaint=False,
                    has_context_windows=context_windows is not None,
                    guidance=guidance,
                )

            def report_multistream_step(event: Any) -> None:
                report_progress(event.step + 1, event.total)

            load_streams = (
                cast("Any", latent_samples)
                if plain_output
                else _move_multistream_latent(cast("Any", latent_samples), handle.load_device)
            )
            preview = (
                sampling_preview_emitter(handle, stream_role=load_streams.roles[0])
                if plain_output
                else multistream_sampling_preview_emitter(handle)
            )
            run_ksampler_as_custom = runtime.run_ksampler_as_custom
            reserved_application_kwargs = {
                "conditioning",
                "cfg",
                "sampler_id",
                "scheduler_id",
                "steps",
                "denoise",
                "seed",
                "segment",
                "denoise_mask",
                "noise_inds",
                "on_step",
                "on_state",
                "cancelled",
                "observer",
                "parent_span_id",
                "sampling_shift",
                *context_windows_kwargs,
            }
            with native_execution_span("sample", "sample", device=str(handle.load_device)) as span:
                parent = None if span is None else span.span_id
                with (
                    inference.use_sampling_environment(
                        extension_ids, context.cancelled if context is not None else _not_cancelled
                    ),
                    _staged_applications(applications, handle, stage_runtime=False),
                ):
                    with torch.inference_mode():
                        application_kwargs = _application_kwargs(
                            applications,
                            handle,
                            load_streams,
                            reserved_keys=reserved_application_kwargs,
                        )
                        with (
                            handle.stage(
                                "diffusion",
                                unload_before=unload_text_before_diffusion,
                                observer_stage="sample",
                                parent_span_id=parent,
                            ),
                            preview_stage(preview),
                        ):
                            result = run_ksampler_as_custom(
                                load_streams,
                                conditioning=prepared,
                                cfg=inference.SamplingGuidance(
                                    uncond,
                                    cfg,
                                    transforms=guidance_transforms,
                                    batching=conditioning_batching,
                                    disable_cfg1_optimization=disable_cfg1_optimization,
                                ),
                                sampler_id=sampler_id,
                                scheduler_id=scheduler_id,
                                steps=steps,
                                denoise=denoise,
                                seed=seed,
                                segment=segment,
                                denoise_mask=denoise_mask,
                                sampling_shift=runtime_sampling_shift,
                                on_step=report_multistream_step,
                                on_state=(preview.on_state if preview is not None else None),
                                cancelled=context.cancelled
                                if context is not None
                                else _not_cancelled,
                                observer=current_native_observer(),
                                parent_span_id=parent,
                                **noise_kwargs,
                                **context_windows_kwargs,
                                **application_kwargs,
                            )
            if (
                type(result) is not inference.MultiStreamLatent
                or result.roles != load_streams.roles
            ):
                raise TypeError("multi-stream sampling must return the input latent streams")
            output: dict[object, object] = dict(latent_mapping)
            if isinstance(runtime, inference.MultiStreamLatentAdapterRuntime) and not plain_output:
                output.pop("downscale_ratio_spacial", None)
                output.pop("downscale_ratio_temporal", None)
            output["samples"] = result.by_role(result.roles[0]) if plain_output else result
            return cls.outputs(latent=output)
        if structural_latent:
            raise TypeError("model does not provide multi-stream sampling")
        context = current_execution_context()
        sampler_registry, extension_ids, _ = _sampler_registry(
            inference,
            context.extension_snapshot_digest if context is not None else None,
        )
        scheduled = (
            bool(ordinary_overlays)
            or _uses_native_scheduling(positive)
            or _uses_native_scheduling(negative)
        )
        schedule_state = None
        result: Any = None
        try:
            if scheduled:
                inference_torch = importlib.import_module("dinkster_inference_torch")
                schedule_state = _NativeScheduleState(
                    handle,
                    inference,
                    inference_torch,
                    ordinary_overlays=ordinary_overlays,
                    ordinary_resolvers=ordinary_resolvers,
                )
                cond = _scheduled_carrier(positive, "positive", handle, schedule_state)
                uncond = _scheduled_carrier(negative, "negative", handle, schedule_state)
                cond_inpaint = _scheduled_inpaint(positive, "positive", torch, inference)
                uncond_inpaint = _scheduled_inpaint(negative, "negative", torch, inference)
            else:
                cond, cond_inpaint = _conditioning(positive, "positive", torch, inference)
                uncond, uncond_inpaint = _conditioning(negative, "negative", torch, inference)
            if (cond_inpaint is None) != (uncond_inpaint is None):
                raise ValueError("positive and negative inpaint conditioning must both be present")
            if (
                cond_inpaint is not None
                and uncond_inpaint is not None
                and (
                    cond_inpaint.mask is not uncond_inpaint.mask
                    or cond_inpaint.masked_image is not uncond_inpaint.masked_image
                )
            ):
                raise ValueError(
                    "positive and negative inpaint conditioning must share concat values"
                )
            if not isinstance(latent_image, Mapping):
                raise TypeError("latent_image must be a mapping containing 'samples'")
            latent = cast("Mapping[object, object]", latent_image)
            noise_inds = _batch_index_noise_inds(latent)
            samples_obj = latent.get("samples")
            if sparse_latent:
                samples: Any = samples_obj
            else:
                if not isinstance(samples_obj, torch.Tensor):
                    raise TypeError(
                        "latent_image['samples'] must be a torch.Tensor or SparseLatent"
                    )
                samples = _normalize_empty_latent(cast("Any", samples_obj), active_runtime, torch)
            noise_mask_obj = latent.get("noise_mask")
            if noise_mask_obj is not None and not isinstance(noise_mask_obj, torch.Tensor):
                raise TypeError("latent_image['noise_mask'] must be a torch.Tensor")
            noise_mask = cast("Any", noise_mask_obj)
            sampler_id = _catalog_id(sampler_registry, sampler_name, "sampler")
            scheduler_id = _catalog_id(
                _inference_registries(inference).schedulers, scheduler, "scheduler"
            )
            active_runtime = component_runtime if component_runtime is not None else handle.runtime
            sampling_space = _native_model_sampling_space(model)
            active_runtime = _sampling_space_runtime(active_runtime, sampling_space)
            control_kwargs: dict[str, object] = {}
            classic_control_handles: tuple[NativeComponentHandle, ...] = ()
            if classic_control_binding is not None:
                if schedule_state is not None:
                    _require_classic_control_keyword(active_runtime.sample_scheduled)
                else:
                    for method_name in ("sample", "sample_custom"):
                        receiver = getattr(active_runtime, method_name, None)
                        if receiver is not None:
                            _require_classic_control_keyword(receiver)
            if z_image_control is not None:
                inference_torch = importlib.import_module("dinkster_inference_torch")
                control_kwargs = {
                    "control": _materialize_z_image_control(
                        handle,
                        z_image_control,
                        samples,
                        torch,
                        inference,
                        inference_torch,
                    )
                }
            # sample_scheduled has no on_state seam, and sparse latents have no preview provider.
            preview = (
                sampling_preview_emitter(handle)
                if schedule_state is None and not sparse_latent
                else None
            )
            if context_windows is not None:
                active_runtime.check_custom_sampling(
                    inference.CustomSamplingRequest(sampler_registry.get(sampler_id), (), ()),
                    has_denoise_mask=noise_mask is not None,
                    has_inpaint=cond_inpaint is not None,
                    has_context_windows=True,
                    guidance=guidance,
                )
            sampling_memory = _sampling_memory_requirements(active_runtime, samples)
            with ExitStack() as stages:
                if classic_control_binding is not None:
                    inference_torch = importlib.import_module("dinkster_inference_torch")
                    classic_control, classic_control_handles = stages.enter_context(
                        _classic_control_context(
                            classic_control_binding,
                            latent_batch=int(samples.shape[0]),
                            torch=torch,
                            inference=inference,
                            inference_torch=inference_torch,
                            base_handle=handle,
                        )
                    )
                    control_kwargs["control"] = classic_control
                stages.enter_context(
                    handle.stage(
                        "diffusion",
                        memory_required=sampling_memory[0],
                        minimum_memory=sampling_memory[1],
                    )
                )
                if z_image_control is not None:
                    stages.enter_context(z_image_control.handle.stage(observer_stage="sample"))
                for control_handle in classic_control_handles:
                    stages.enter_context(control_handle.stage(observer_stage="sample"))
                stages.enter_context(preview_stage(preview))
                sampling_context = (
                    torch.no_grad() if schedule_state is not None else torch.inference_mode()
                )
                with sampling_context:
                    with (
                        inference.use_sampling_environment(
                            extension_ids,
                            context.cancelled if context is not None else _not_cancelled,
                        ),
                    ):
                        sample: Any = None
                        custom_only = isinstance(
                            active_runtime, inference.CustomSamplingRuntime
                        ) and not callable(getattr(active_runtime, "sample", None))
                        custom_sampling_only = custom_only or bool(
                            getattr(active_runtime, "custom_sampling_only", False)
                        )
                        continue_sampling = not custom_only
                        if custom_sampling_only and (schedule_state is not None or applications):
                            raise ValueError(
                                "custom sampling runtimes do not accept model patches or schedules"
                            )
                        if custom_only:

                            def report_custom_step(event: Any) -> None:
                                report_progress(event.step + 1, event.total)

                            sampling = active_runtime.family.sampling
                            if not inference.is_flow_parameterization(sampling.parameterization):
                                raise ValueError(
                                    "custom-only KSampler runtimes require flow parameterization"
                                )
                            runtime_sampling_shift = _runtime_sampling_shift(
                                active_runtime, sampling_shift
                            )
                            space = sampling_space or inference.FlowSigmas(
                                shift=(sampling.shift if sampling_shift is None else sampling_shift)
                            )
                            custom_result = importlib.import_module(
                                "dinkster_inference_torch.sampling_execution"
                            ).run_ksampler_as_custom(
                                active_runtime,
                                (samples if sparse_latent else samples.to(handle.load_device)),
                                samplers=sampler_registry,
                                schedulers=importlib.import_module(
                                    "dinkster_inference_torch"
                                ).torch_scheduler_registry(),
                                space=space,
                                flow=True,
                                device=handle.load_device,
                                sampler_id=sampler_id,
                                scheduler_id=scheduler_id,
                                steps=steps,
                                denoise=denoise,
                                seed=seed,
                                cond=cond,
                                cfg=inference.SamplingGuidance(
                                    uncond,
                                    cfg,
                                    transforms=guidance_transforms,
                                    batching=conditioning_batching,
                                    disable_cfg1_optimization=disable_cfg1_optimization,
                                ),
                                guidance=guidance,
                                segment=segment,
                                denoise_mask=noise_mask,
                                inpaint=cond_inpaint,
                                noise_inds=noise_inds,
                                context_windows=context_windows,
                                on_step=report_custom_step,
                                on_state=(preview.on_state if preview is not None else None),
                                sample_custom_kwargs={
                                    **control_kwargs,
                                    **(
                                        {}
                                        if runtime_sampling_shift is None
                                        else {"sampling_shift": runtime_sampling_shift}
                                    ),
                                },
                                error=ValueError,
                            )
                            result = custom_result.output
                            continue_sampling = False
                        if continue_sampling:
                            sample = (
                                active_runtime.sample_scheduled
                                if schedule_state is not None
                                else active_runtime.sample
                            )
                        family_conditioning = isinstance(
                            active_runtime, inference.ConditioningRuntime
                        )
                        kwargs: dict[str, object] = {
                            "cond": cond,
                            "cfg": inference.SamplingGuidance(
                                uncond,
                                cfg,
                                transforms=guidance_transforms,
                                batching=conditioning_batching,
                                disable_cfg1_optimization=disable_cfg1_optimization,
                            ),
                            "sampler_id": sampler_id,
                            "scheduler_id": scheduler_id,
                            "steps": steps,
                            "denoise": denoise,
                            "seed": seed,
                            "guidance": guidance,
                            "denoise_mask": noise_mask,
                            "inpaint": cond_inpaint,
                            "segment": segment,
                            "schedule_device": handle.load_device,
                            **control_kwargs,
                        }
                        # Only a non-None value is forwarded: runtimes
                        # without the parameter keep working on the
                        # default path and refuse a real request with a
                        # TypeError instead of dropping it.
                        if noise_inds is not None:
                            kwargs["noise_inds"] = noise_inds
                        if context_windows is not None:
                            kwargs["context_windows"] = context_windows
                        if continue_sampling and (
                            not family_conditioning
                            or getattr(active_runtime, "sampling_compute_dtype", None) is not None
                        ):
                            kwargs["device"] = handle.load_device
                            kwargs["compute_dtype"] = _compute_dtype(handle.runtime)
                        if continue_sampling and sampling_shift is not None:
                            kwargs["sampling_shift"] = _runtime_sampling_shift(
                                active_runtime, sampling_shift
                            )
                        if schedule_state is not None:
                            kwargs["resolver"] = schedule_state.resolve
                        elif preview is not None:
                            kwargs["on_state"] = preview.on_state
                        if continue_sampling:
                            with _materialized_application_kwargs(
                                applications,
                                handle,
                                samples,
                                reserved_keys={
                                    *kwargs,
                                    "cancelled",
                                    "noise_inds",
                                    "observer",
                                    "on_state",
                                    "on_step",
                                    "parent_span_id",
                                    "resolver",
                                },
                            ) as application_kwargs:
                                if (
                                    schedule_state is not None
                                    and "sd15_attention_contributions" in application_kwargs
                                ):
                                    raise ValueError(
                                        "SD1.5 IP-Adapter does not support scheduled prompt or "
                                        "patch sampling"
                                    )
                                kwargs.update(application_kwargs)
                                result = sample(samples, **kwargs)
        finally:
            if schedule_state is not None:
                schedule_state.close()
        output: dict[object, object] = dict(cast("Mapping[object, object]", latent_image))
        output["samples"] = result
        return cls.outputs(latent=output)


class NativeKSamplerAdvanced(KSamplerAdvanced):
    """Run a partial full-schedule range through a native runtime."""

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        add_noise: str,
        noise_seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: object,
        negative: object,
        latent_image: object,
        start_at_step: int,
        end_at_step: int,
        return_with_leftover_noise: str,
        conditioning_batching: object | None = None,
    ) -> Mapping[str, object]:
        for name, value, low, high in (
            ("noise_seed", noise_seed, 0, cls.MAX_SEED),
            ("steps", steps, 1, cls.MAX_STEPS),
            ("cfg", cfg, 0.0, cls.MAX_CFG),
            ("start_at_step", start_at_step, 0, cls.MAX_STEPS),
            ("end_at_step", end_at_step, 0, cls.MAX_STEPS),
        ):
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
        if add_noise not in ("enable", "disable"):
            raise ValueError("add_noise must be 'enable' or 'disable'")
        if return_with_leftover_noise not in ("disable", "enable"):
            raise ValueError("return_with_leftover_noise must be 'disable' or 'enable'")
        effective_end = min(end_at_step, steps)
        if start_at_step >= effective_end:
            if not isinstance(latent_image, Mapping):
                raise TypeError("latent_image must be a mapping containing 'samples'")
            output = dict(cast("Mapping[object, object]", latent_image))
            output.pop("downscale_ratio_spacial", None)
            output.pop("downscale_ratio_temporal", None)
            return cls.outputs(latent=output)
        inference = importlib.import_module("dinkster_inference")
        segment = inference.SamplingSegment(
            steps=steps,
            start_step=start_at_step,
            end_step=effective_end,
            add_noise=add_noise == "enable",
            return_with_leftover_noise=return_with_leftover_noise == "enable",
        )
        result = NativeKSampler.execute(
            model=model,
            seed=noise_seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=1.0,
            segment=segment,
            conditioning_batching=conditioning_batching,
        )
        return cls.outputs(latent=result["latent"])


class NativeVAEDecode(VAEDecode):
    """Decode native NCHW content and publish Comfy's NHWC image form."""

    @classmethod
    def execute(cls, *, samples: object, vae: object) -> Mapping[str, object]:
        handle = _native_handle(vae, "vae")
        torch = _torch()
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a latent mapping")
        latent_obj = cast("Mapping[object, object]", samples).get("samples")
        if not isinstance(latent_obj, torch.Tensor):
            raise TypeError("samples['samples'] must be a torch.Tensor")
        latent = cast("Any", latent_obj)
        if len(latent.shape) not in (4, 5):
            raise ValueError(
                "samples['samples'] must be NCHW or NCTHW rank 4/5, "
                f"got shape {tuple(latent.shape)}"
            )
        with handle.stage("vae"):
            with torch.inference_mode():
                load_latent = latent.to(handle.load_device)
                image = _run_direct_vae(
                    handle=handle,
                    value=load_latent,
                    direction="decode",
                    operation=handle.runtime.decode_latent,
                    codec=getattr(handle.runtime, "codec", None),
                )
        if len(image.shape) == 5:
            if image.shape[1] != 3:
                raise ValueError(
                    f"native video VAE decode must return [B,3,T,H,W], got {tuple(image.shape)}"
                )
            image = image.permute(0, 2, 3, 4, 1).flatten(0, 1)
        elif len(image.shape) == 4:
            image = image.permute(0, 2, 3, 1)
        else:
            raise ValueError(
                f"native VAE decode must return NCHW/NCTHW rank 4/5, got {tuple(image.shape)}"
            )
        return cls.outputs(image=image)


class NativeVAEEncode(VAEEncode):
    """Convert Comfy NHWC images to native NCHW codec content."""

    @classmethod
    def execute(cls, *, pixels: object, vae: object) -> Mapping[str, object]:
        handle = _native_handle(vae, "vae")
        torch = _torch()
        if not isinstance(pixels, torch.Tensor):
            raise TypeError("pixels must be a torch.Tensor")
        tensor = cast("Any", pixels)
        if len(tensor.shape) != 4:
            raise ValueError(f"pixels must be NHWC rank 4, got shape {tuple(tensor.shape)}")
        with handle.stage("vae"):
            content = tensor.permute(0, 3, 1, 2).to(handle.load_device)
            if importlib.import_module("dinkster_native.families.wan21").is_runtime_family(
                handle.runtime.family.id
            ):
                content = content.permute(1, 0, 2, 3).unsqueeze(0)
            with torch.inference_mode():
                latent = _run_direct_vae(
                    handle=handle,
                    value=content,
                    direction="encode",
                    operation=handle.runtime.encode_content,
                    codec=getattr(handle.runtime, "codec", None),
                )
        return cls.outputs(latent={"samples": latent})


def _provider_lora_stack_family(*, model_only: bool, min_members: int) -> InputFamilySpec:
    inputs = [
        InputSpec("lora", _ASSET),
        InputSpec("strength_model", _FLOAT, required=False, default=1.0),
    ]
    if not model_only:
        inputs.append(InputSpec("strength_clip", _FLOAT, required=False, default=1.0))
    return InputFamilySpec(
        "loras",
        tuple(inputs),
        min_members=min_members,
        max_members=50,
        member_prefix="lora_",
    )


def _provider_diffusion_components_family() -> InputFamilySpec:
    return InputFamilySpec(
        "components",
        (
            InputSpec(
                "component",
                _ASSET,
                widget=AssetWidget(
                    accept=("application/octet-stream",),
                    kind="model/diffusion",
                ),
            ),
            InputSpec("role", _STRING, widget=StringWidget()),
        ),
        min_members=1,
        max_members=64,
        member_prefix="component_",
    )


def _generation_provider_text_schema(node_type: str, display_name: str) -> NodeSchema:
    return NodeSchema(
        node_type=node_type,
        display_name=display_name,
        category="text",
        inputs=(
            InputSpec("clip", _DINKSTER_CLIP, required=False, lazy=True),
            InputSpec(
                "provider",
                _COMBO,
                required=False,
                widget=ComboWidget(remote_route="/api/choices/dinkster.generation.providers"),
                advanced=True,
            ),
            InputSpec(
                "prompt",
                _STRING,
                default="",
                widget=StringWidget(multiline=True, dynamic_prompts=True),
            ),
            InputSpec("image", _DINKSTER_IMAGE, required=False),
            InputSpec("video", _DINKSTER_IMAGE, required=False),
            InputSpec("audio", _COMFY_AUDIO, required=False),
            InputSpec("max_length", _INT, default=512),
            InputSpec("thinking", _BOOLEAN, required=False, default=False),
            InputSpec("use_default_template", _BOOLEAN, required=False, default=False),
        ),
        combos=(
            DynamicComboSpec(
                "sampling_mode",
                options=(
                    DynamicComboOption(
                        "on",
                        inputs=(
                            InputSpec("temperature", _FLOAT, default=0.7),
                            InputSpec("top_k", _INT, default=64),
                            InputSpec("top_p", _FLOAT, default=0.95),
                            InputSpec("min_p", _FLOAT, default=0.05),
                            InputSpec("repetition_penalty", _FLOAT, default=1.05),
                            InputSpec("seed", _INT, default=0),
                            InputSpec(
                                "presence_penalty",
                                _FLOAT,
                                required=False,
                                default=0.0,
                            ),
                        ),
                    ),
                    DynamicComboOption("off"),
                ),
                default="on",
            ),
        ),
        outputs=(OutputSpec("generated_text", _STRING),),
    )


_GENERATION_SCHEMAS = {
    schema.node_type: schema for node in GENERATION_NODES if (schema := node.schema())
}


def _generation_provider_schema(node_type: str) -> NodeSchema:
    """Provider-side copies of the universal generation signatures."""
    owner = _GENERATION_SCHEMAS.get(node_type)
    if owner is not None:
        return owner
    if node_type == "dinkster.empty_trellis2_latent_structure":
        return NodeSchema(
            node_type=node_type,
            display_name="Empty TRELLIS.2 Latent Structure",
            category="model/latent/trellis",
            inputs=(
                InputSpec(
                    "batch_size",
                    _INT,
                    default=1,
                    widget=NumberWidget(min=1, max=4096),
                ),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type in ("dinkster.trellis2_conditioning", "dinkster.pixal3d_conditioning"):
        pixal3d = node_type == "dinkster.pixal3d_conditioning"
        inputs = [
            InputSpec("clip_vision_model", _DINKSTER_CLIP_VISION),
            InputSpec("image", _DINKSTER_IMAGE),
        ]
        if pixal3d:
            inputs.append(
                InputSpec(
                    "camera_angle_x",
                    _FLOAT,
                    default=49.13,
                    widget=NumberWidget(min=1.0, max=170.0, step=0.01),
                )
            )
        return NodeSchema(
            node_type=node_type,
            display_name="Pixal3D Conditioning" if pixal3d else "TRELLIS.2 Conditioning",
            category="model/conditioning/trellis2",
            inputs=tuple(inputs),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
            ),
        )
    if node_type == "dinkster.vae_decode_structure_trellis2":
        return NodeSchema(
            node_type=node_type,
            display_name="Decode TRELLIS.2 Structure",
            category="model/latent/trellis",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec(
                    "resolution",
                    _COMBO,
                    default="32",
                    widget=ComboWidget(options=("32", "64")),
                ),
            ),
            outputs=(OutputSpec("voxel", _COMFY_VOXEL),),
        )
    if node_type == "dinkster.trellis2_shape_stage":
        return NodeSchema(
            node_type=node_type,
            display_name="TRELLIS.2 Shape Stage",
            category="model/conditioning/trellis2",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("voxel", _COMFY_VOXEL),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.trellis2_upsample_stage":
        return NodeSchema(
            node_type=node_type,
            display_name="TRELLIS.2 Upsample Stage",
            category="model/conditioning/trellis2",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("shape_latent", _DINKSTER_LATENT),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec(
                    "target_resolution",
                    _INT,
                    default=1024,
                    widget=NumberWidget(min=1024, max=2048, step=128),
                ),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.vae_decode_shape_trellis":
        return NodeSchema(
            node_type=node_type,
            display_name="Decode TRELLIS.2 Shape",
            category="model/latent/trellis",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("vae", _DINKSTER_VAE),
            ),
            outputs=(
                OutputSpec("mesh", _COMFY_MESH),
                OutputSpec("shape_subdivides", _COMFY_SHAPE_SUBDIVIDES),
            ),
        )
    if node_type == "dinkster.trellis2_texture_stage":
        return NodeSchema(
            node_type=node_type,
            display_name="TRELLIS.2 Texture Stage",
            category="model/conditioning/trellis2",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("shape_latent", _DINKSTER_LATENT),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.vae_decode_texture_trellis":
        return NodeSchema(
            node_type=node_type,
            display_name="Decode TRELLIS.2 Texture",
            category="model/latent/trellis",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec("shape_subdivides", _COMFY_SHAPE_SUBDIVIDES),
            ),
            outputs=(OutputSpec("voxel_colors", _COMFY_VOXEL),),
        )
    if node_type == "dinkster.load_checkpoint":
        return NodeSchema(
            node_type=node_type,
            display_name="Load Checkpoint",
            category="model/loaders",
            inputs=(InputSpec("checkpoint", _ASSET),),
            outputs=(
                OutputSpec("model", _DINKSTER_MODEL),
                OutputSpec("clip", _DINKSTER_CLIP),
                OutputSpec("vae", _DINKSTER_VAE),
            ),
        )
    if node_type == "dinkster.load_checkpoint_stack":
        return NodeSchema(
            node_type=node_type,
            display_name="Load Checkpoint Stack",
            category="model/loaders",
            inputs=(
                InputSpec("checkpoint", _ASSET),
                InputSpec("stop_at_clip_layer", _INT, default=-1),
                InputSpec("execution_mode", _COMBO, required=False, default="auto"),
            ),
            input_families=(_provider_lora_stack_family(model_only=False, min_members=0),),
            outputs=(
                OutputSpec("model", _DINKSTER_MODEL),
                OutputSpec("clip", _DINKSTER_CLIP),
                OutputSpec("vae", _DINKSTER_VAE),
            ),
        )
    if node_type == "dinkster.load_diffusion_model":
        return NodeSchema(
            node_type=node_type,
            display_name="Load Diffusion Model",
            category="model/loaders",
            inputs=(
                InputSpec("diffusion_model", _ASSET),
                InputSpec("weight_dtype", _COMBO, required=False, default="default"),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.load_diffusion_components":
        return NodeSchema(
            node_type=node_type,
            display_name="Load Diffusion Components",
            category="model/loaders",
            description="Loads named diffusion components as one admitted model.",
            inputs=(InputSpec("weight_dtype", _COMBO, required=False, default="default"),),
            input_families=(_provider_diffusion_components_family(),),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.load_ltxav_text_encoder":
        return NodeSchema(
            node_type=node_type,
            display_name="Load LTX-2 Text Encoder",
            category="model/loaders",
            inputs=(
                InputSpec("text_encoder", _ASSET),
                InputSpec("ckpt_name", _ASSET),
                InputSpec("device", _COMBO, required=False, default="default"),
            ),
            outputs=(OutputSpec("clip", _DINKSTER_CLIP),),
        )
    if node_type == "dinkster.load_ltxav_audio_vae":
        return NodeSchema(
            node_type=node_type,
            display_name="Load LTX-2 Audio VAE",
            category="model/loaders",
            inputs=(InputSpec("ckpt_name", _ASSET),),
            outputs=(OutputSpec("audio_vae", _DINKSTER_VAE),),
        )
    if node_type == "dinkster.load_latent_upscale_model":
        return NodeSchema(
            node_type=node_type,
            display_name="Load Latent Upscale Model",
            category="model/loaders",
            inputs=(InputSpec("model_name", _ASSET),),
            outputs=(OutputSpec("upscale_model", _COMFY_LATENT_UPSCALE_MODEL),),
        )
    if node_type == "dinkster.ltxav_audio_vae_decode":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-2 Audio VAE Decode",
            category="model/latent/ltxv",
            inputs=(InputSpec("samples", _DINKSTER_LATENT), InputSpec("audio_vae", _DINKSTER_VAE)),
            outputs=(OutputSpec("audio", _COMFY_AUDIO),),
        )
    if node_type == "dinkster.load_lora":
        return NodeSchema(
            node_type=node_type,
            display_name="Load LoRA (Model and CLIP)",
            category="model/loaders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("clip", _DINKSTER_CLIP),
                InputSpec("lora", _ASSET),
                InputSpec("strength_model", _FLOAT, required=False, default=1.0),
                InputSpec("strength_clip", _FLOAT, required=False, default=1.0),
                InputSpec("execution_mode", _COMBO, required=False, default="auto"),
            ),
            outputs=(
                OutputSpec("model", _DINKSTER_MODEL),
                OutputSpec("clip", _DINKSTER_CLIP),
            ),
        )
    if node_type == "dinkster.load_lora_model_only":
        return NodeSchema(
            node_type=node_type,
            display_name="Load LoRA",
            category="model/loaders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("lora", _ASSET),
                InputSpec("strength_model", _FLOAT, required=False, default=1.0),
                InputSpec("execution_mode", _COMBO, required=False, default="auto"),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.apply_lora_stack":
        return NodeSchema(
            node_type=node_type,
            display_name="Apply LoRA Stack (Model and CLIP)",
            category="model/loaders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("clip", _DINKSTER_CLIP),
                InputSpec("execution_mode", _COMBO, required=False, default="auto"),
            ),
            input_families=(_provider_lora_stack_family(model_only=False, min_members=1),),
            outputs=(
                OutputSpec("model", _DINKSTER_MODEL),
                OutputSpec("clip", _DINKSTER_CLIP),
            ),
        )
    if node_type == "dinkster.apply_lora_stack_model_only":
        return NodeSchema(
            node_type=node_type,
            display_name="Apply LoRA Stack",
            category="model/loaders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("execution_mode", _COMBO, required=False, default="auto"),
            ),
            input_families=(_provider_lora_stack_family(model_only=True, min_members=1),),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.clip_text_encode":
        return NodeSchema(
            node_type=node_type,
            display_name="CLIP Text Encode",
            category="model/conditioning",
            inputs=(
                InputSpec(
                    "text",
                    _STRING,
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec("clip", _DINKSTER_CLIP),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.clip_text_encode_lumina2":
        return NodeSchema(
            node_type=node_type,
            display_name="CLIP Text Encode (Lumina 2)",
            category="model/conditioning/lumina",
            inputs=(
                InputSpec(
                    "system_prompt",
                    _COMBO,
                    required=False,
                    default="superior",
                    widget=ComboWidget(options=("superior", "alignment")),
                ),
                InputSpec(
                    "user_prompt",
                    _STRING,
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
                InputSpec("clip", _DINKSTER_CLIP),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.model_sampling_aura_flow":
        return NodeSchema(
            node_type=node_type,
            display_name="ModelSamplingAuraFlow",
            category="model/patch",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec(
                    "shift",
                    _FLOAT,
                    required=False,
                    default=1.73,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.text_generate":
        return _generation_provider_text_schema(node_type, "Generate Text")
    if node_type == "dinkster.prompt_enhance":
        return _generation_provider_text_schema(node_type, "Enhance Prompt")
    if node_type == "dinkster.clip_set_last_layer":
        return NodeSchema(
            node_type=node_type,
            display_name="CLIP Set Last Layer",
            category="model/conditioning",
            inputs=(
                InputSpec("clip", _DINKSTER_CLIP),
                InputSpec(
                    "stop_at_clip_layer",
                    _INT,
                    default=-1,
                    widget=NumberWidget(min=-24, max=-1, step=1),
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("clip", _DINKSTER_CLIP),),
        )
    if node_type == "dinkster.t5_tokenizer_options":
        return NodeSchema(
            node_type=node_type,
            display_name="T5 Tokenizer Options",
            category="model/conditioning",
            inputs=(
                InputSpec("clip", _DINKSTER_CLIP),
                InputSpec(
                    "min_padding",
                    _INT,
                    default=0,
                    widget=NumberWidget(min=0, max=10_000, step=1),
                ),
                InputSpec(
                    "min_length",
                    _INT,
                    default=0,
                    widget=NumberWidget(min=0, max=10_000, step=1),
                ),
            ),
            outputs=(OutputSpec("clip", _DINKSTER_CLIP),),
        )
    if node_type == "dinkster.clip_text_encode_controlnet":
        return NodeSchema(
            node_type=node_type,
            display_name="CLIP Text Encode (Controlnet)",
            category="model/conditioning",
            inputs=(
                InputSpec("clip", _DINKSTER_CLIP),
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
                InputSpec(
                    "text",
                    _STRING,
                    widget=StringWidget(multiline=True, dynamic_prompts=True),
                ),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.flux_guidance":
        return NodeSchema(
            node_type=node_type,
            display_name="FluxGuidance",
            category="model/conditioning",
            inputs=(
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
                InputSpec("guidance", _FLOAT, default=3.5),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.flux_disable_guidance":
        return NodeSchema(
            node_type=node_type,
            display_name="Flux Disable Guidance",
            category="model/conditioning/flux",
            inputs=(InputSpec("conditioning", _DINKSTER_CONDITIONING),),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.reference_latent":
        return NodeSchema(
            node_type=node_type,
            display_name="Set Reference Latent",
            category="model/conditioning",
            inputs=(
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
                InputSpec("latent", _DINKSTER_LATENT, required=False),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.cfg_zero_star":
        return NodeSchema(
            node_type=node_type,
            display_name="CFGZeroStar",
            category="advanced/guidance",
            inputs=(InputSpec("model", _DINKSTER_MODEL),),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.cfg_norm":
        return NodeSchema(
            node_type=node_type,
            display_name="CFGNorm",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("strength", _FLOAT, default=1.0),
                InputSpec("pre_cfg", _BOOLEAN, default=False),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.tcfg":
        return NodeSchema(
            node_type=node_type,
            display_name="Tangential Damping CFG",
            category="advanced/guidance",
            inputs=(InputSpec("model", _DINKSTER_MODEL),),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.fresca":
        return NodeSchema(
            node_type=node_type,
            display_name="FreSca",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("scale_low", _FLOAT, default=1.0),
                InputSpec("scale_high", _FLOAT, default=1.25),
                InputSpec("freq_cutoff", _INT, default=20),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.context_windows_manual":
        return NodeSchema(
            node_type=node_type,
            display_name="Context Windows (Manual)",
            category="model/patch",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("context_length", _INT, default=16),
                InputSpec("context_overlap", _INT, default=4),
                InputSpec("context_schedule", _COMBO, default="standard_static"),
                InputSpec("context_stride", _INT, default=1),
                InputSpec("closed_loop", _BOOLEAN, default=False),
                InputSpec("fuse_method", _COMBO, default="pyramid"),
                InputSpec("dim", _INT, default=0),
                InputSpec("freenoise", _BOOLEAN, default=False),
                InputSpec("causal_window_fix", _BOOLEAN, default=True),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.wan_context_windows_manual":
        return NodeSchema(
            node_type=node_type,
            display_name="WAN Context Windows (Manual)",
            category="model/patch/wan",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("context_length", _INT, default=81),
                InputSpec("context_overlap", _INT, default=30),
                InputSpec("context_schedule", _COMBO, default="standard_uniform"),
                InputSpec("context_stride", _INT, default=1),
                InputSpec("closed_loop", _BOOLEAN, default=False),
                InputSpec("fuse_method", _COMBO, default="pyramid"),
                InputSpec("freenoise", _BOOLEAN, default=True),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.ltxv_context_windows":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Context Windows",
            category="model/patch",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("context_length", _INT, default=145),
                InputSpec("context_overlap", _INT, default=40),
                InputSpec("context_schedule", _COMBO, default="standard_uniform"),
                InputSpec("context_stride", _INT, default=1),
                InputSpec("closed_loop", _BOOLEAN, default=False),
                InputSpec("fuse_method", _COMBO, default="pyramid"),
                InputSpec("freenoise", _BOOLEAN, default=True),
                InputSpec("retain_first_frame", _BOOLEAN, default=False),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.adaptive_projected_guidance":
        return NodeSchema(
            node_type=node_type,
            display_name="Adaptive Projected Guidance",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("eta", _FLOAT, default=1.0),
                InputSpec("norm_threshold", _FLOAT, default=5.0),
                InputSpec("momentum", _FLOAT, default=0.0),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.mahiro_guidance":
        return NodeSchema(
            node_type=node_type,
            display_name="Positive-Biased Guidance",
            category="advanced/guidance",
            inputs=(InputSpec("model", _DINKSTER_MODEL),),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.epsilon_scaling":
        return NodeSchema(
            node_type=node_type,
            display_name="Epsilon Scaling",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("scaling_factor", _FLOAT, default=1.005),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.cfg_override":
        return NodeSchema(
            node_type=node_type,
            display_name="CFG Override",
            category="model/sampling/guiders",
            description="Overrides CFG over a sampling-percent range.",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("cfg", _FLOAT, default=1.0),
                InputSpec("start_percent", _FLOAT, default=0.0),
                InputSpec("end_percent", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.rescale_cfg":
        return NodeSchema(
            node_type=node_type,
            display_name="RescaleCFG",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("multiplier", _FLOAT, default=0.7),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.renorm_cfg":
        return NodeSchema(
            node_type=node_type,
            display_name="RenormCFG",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("cfg_trunc", _FLOAT, default=100.0),
                InputSpec("renorm_cfg", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.temporal_score_rescaling":
        return NodeSchema(
            node_type=node_type,
            display_name="TSR - Temporal Score Rescaling",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("tsr_k", _FLOAT, default=0.95),
                InputSpec("tsr_sigma", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.nag":
        return NodeSchema(
            node_type=node_type,
            display_name="Normalized Attention Guidance",
            category="advanced/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("nag_scale", _FLOAT, default=5.0),
                InputSpec("nag_alpha", _FLOAT, default=0.5),
                InputSpec("nag_tau", _FLOAT, default=1.5),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.ltxav_conditioning":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-2 AV Conditioning",
            category="model/conditioning",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("frame_rate", _FLOAT, default=25.0),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
            ),
        )
    if node_type == "dinkster.ltxav_reference_audio":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-2 Reference Audio",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("reference_audio", _COMFY_AUDIO),
                InputSpec("audio_vae", _DINKSTER_VAE),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
            ),
        )
    if node_type == "dinkster.ltxav_id_lora_reference_audio":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Reference Audio (ID-LoRA)",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("reference_audio", _COMFY_AUDIO),
                InputSpec("audio_vae", _DINKSTER_VAE, display_name="Audio VAE"),
                InputSpec("identity_guidance_scale", _FLOAT, default=3.0),
                InputSpec("start_percent", _FLOAT, default=0.0, advanced=True),
                InputSpec("end_percent", _FLOAT, default=1.0, advanced=True),
            ),
            outputs=(
                OutputSpec("model", _DINKSTER_MODEL),
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
            ),
        )
    if node_type == "dinkster.ltxv_spatiotemporal_guidance":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Spatio-Temporal Guidance (STG)",
            category="model/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("scale", _FLOAT, default=1.0),
                InputSpec("blocks", _STRING, default="29"),
                InputSpec("start_percent", _FLOAT, default=0.0, advanced=True),
                InputSpec("end_percent", _FLOAT, default=1.0, advanced=True),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.ltxv_modality_guidance":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Modality Guidance (A/V Coupling)",
            category="model/guidance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("modality_scale", _FLOAT, default=3.0),
                InputSpec("start_percent", _FLOAT, default=0.0, advanced=True),
                InputSpec("end_percent", _FLOAT, default=1.0, advanced=True),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.ltxv_duration_predictor":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Duration Predictor",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("duration_head", _COMFY_MODEL_PATCH),
                InputSpec("frame_rate", _FLOAT, default=24.0),
                InputSpec("min_seconds", _FLOAT, default=1.0),
                InputSpec("max_seconds", _FLOAT, default=20.0),
            ),
            outputs=(OutputSpec("num_frames", _INT), OutputSpec("seconds", _FLOAT)),
        )
    if node_type == "dinkster.ltxv_conditioning":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-Video Conditioning",
            category="model/conditioning",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("frame_rate", _FLOAT, default=25.0),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
            ),
        )
    if node_type == "dinkster.ltxv_image_to_video":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-Video Image to Video",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec("image", _DINKSTER_IMAGE),
                InputSpec("width", _INT, default=768),
                InputSpec("height", _INT, default=512),
                InputSpec("length", _INT, default=97),
                InputSpec("batch_size", _INT, default=1),
                InputSpec("strength", _FLOAT, default=1.0),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.ltxv_image_to_video_inplace":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-Video Image to Video (In-place)",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec("image", _DINKSTER_IMAGE),
                InputSpec("latent", _DINKSTER_LATENT),
                InputSpec("strength", _FLOAT, default=1.0),
                InputSpec("bypass", _BOOLEAN, required=False, default=False),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.ltxv_add_guide":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-Video Add Guide",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec("latent", _DINKSTER_LATENT),
                InputSpec("image", _DINKSTER_IMAGE),
                InputSpec("frame_idx", _INT, default=0),
                InputSpec("strength", _FLOAT, default=1.0),
                InputSpec("attention_mask", _DINKSTER_MASK, required=False),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.ltxv_crop_guides":
        return NodeSchema(
            node_type=node_type,
            display_name="LTX-Video Crop Guides",
            category="model/conditioning/ltxv",
            inputs=(
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("latent", _DINKSTER_LATENT),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
                OutputSpec("latent", _DINKSTER_LATENT),
            ),
        )
    if node_type == "dinkster.ltxv_latent_upsampler":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Latent Upsampler",
            category="model/latent/ltxv",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("upscale_model", _COMFY_LATENT_UPSCALE_MODEL),
                InputSpec("vae", _DINKSTER_VAE),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.conditioning_merge":
        return NodeSchema(
            node_type=node_type,
            display_name="Conditioning Merge",
            category="model/conditioning/transform",
            combos=(
                DynamicComboSpec(
                    "mode",
                    options=(
                        DynamicComboOption(
                            "combine",
                            inputs=(
                                InputFamilySpec(
                                    "inputs",
                                    _DINKSTER_CONDITIONING,
                                    min_members=2,
                                    member_names=tuple(
                                        f"conditioning_{index}" for index in range(1, 9)
                                    ),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "average",
                            inputs=(
                                InputSpec("conditioning_to", _DINKSTER_CONDITIONING),
                                InputSpec("conditioning_from", _DINKSTER_CONDITIONING),
                                InputSpec("conditioning_to_strength", _FLOAT, default=1.0),
                            ),
                        ),
                        DynamicComboOption(
                            "concat",
                            inputs=(
                                InputSpec("conditioning_to", _DINKSTER_CONDITIONING),
                                InputSpec("conditioning_from", _DINKSTER_CONDITIONING),
                            ),
                        ),
                    ),
                    default="combine",
                ),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.conditioning_scale":
        return NodeSchema(
            node_type=node_type,
            display_name="Conditioning Scale",
            category="model/conditioning/transform",
            inputs=(
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
                InputSpec("multiplier", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.conditioning_set_area":
        return NodeSchema(
            node_type=node_type,
            display_name="Conditioning Set Area",
            category="model/conditioning/transform",
            inputs=(
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
                InputSpec("strength", _FLOAT, default=1.0),
            ),
            combos=(
                DynamicComboSpec(
                    "units",
                    options=(
                        DynamicComboOption(
                            "pixels",
                            inputs=(
                                InputSpec("width", _INT, default=64),
                                InputSpec("height", _INT, default=64),
                                InputSpec("x", _INT, default=0),
                                InputSpec("y", _INT, default=0),
                            ),
                        ),
                        DynamicComboOption(
                            "percent",
                            inputs=(
                                InputSpec("width", _FLOAT, default=1.0),
                                InputSpec("height", _FLOAT, default=1.0),
                                InputSpec("x", _FLOAT, default=0.0),
                                InputSpec("y", _FLOAT, default=0.0),
                            ),
                        ),
                        DynamicComboOption(
                            "percent-video",
                            inputs=(
                                InputSpec("width", _FLOAT, default=1.0),
                                InputSpec("height", _FLOAT, default=1.0),
                                InputSpec("temporal", _FLOAT, default=1.0),
                                InputSpec("x", _FLOAT, default=0.0),
                                InputSpec("y", _FLOAT, default=0.0),
                                InputSpec("z", _FLOAT, default=0.0),
                            ),
                        ),
                    ),
                    default="pixels",
                ),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.conditioning_set_mask":
        return NodeSchema(
            node_type=node_type,
            display_name="Conditioning Set Mask",
            category="model/conditioning/transform",
            inputs=(
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
                InputSpec("mask", _DINKSTER_MASK),
                InputSpec("strength", _FLOAT, default=1.0),
                InputSpec("set_cond_area", _COMBO, default="default"),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.conditioning_set_timestep_range":
        return NodeSchema(
            node_type=node_type,
            display_name="Conditioning Set Timestep Range",
            category="model/conditioning/transform",
            inputs=(
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
                InputSpec("start", _FLOAT, default=0.0),
                InputSpec("end", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.conditioning_zero_out":
        return NodeSchema(
            node_type=node_type,
            display_name="Conditioning Zero Out",
            category="model/conditioning/transform",
            inputs=(InputSpec("conditioning", _DINKSTER_CONDITIONING),),
            outputs=(OutputSpec("conditioning", _DINKSTER_CONDITIONING),),
        )
    if node_type == "dinkster.chroma_radiance_options":
        return NodeSchema(
            node_type=node_type,
            display_name="Chroma Radiance Options",
            category="model/patch/chroma radiance",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("preserve_wrapper", _BOOLEAN, required=False, default=True),
                InputSpec("start_sigma", _FLOAT, required=False, default=1.0, advanced=True),
                InputSpec("end_sigma", _FLOAT, required=False, default=0.0, advanced=True),
                InputSpec("nerf_tile_size", _INT, required=False, default=-1, advanced=True),
                InputSpec(
                    "force_sequential_txt_ids",
                    _BOOLEAN,
                    required=False,
                    default=False,
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.chroma_model_sampling":
        return NodeSchema(
            node_type=node_type,
            display_name="Chroma Model Sampling",
            category="model/patch/chroma",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("shift", _FLOAT, required=False, default=1.73),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.model_sampling_sd3":
        return NodeSchema(
            node_type=node_type,
            display_name="Model Sampling SD3",
            category="model/patch/stable diffusion",
            description="Applies discrete-flow sampling with a configurable shift.",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("shift", _FLOAT, default=3.0),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.model_sampling_flux":
        return NodeSchema(
            node_type=node_type,
            display_name="ModelSamplingFlux",
            category="model/patch/flux",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("max_shift", _FLOAT, default=1.15, advanced=True),
                InputSpec("base_shift", _FLOAT, default=0.5, advanced=True),
                InputSpec("width", _INT, default=1024),
                InputSpec("height", _INT, default=1024),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.empty_latent_image":
        return NodeSchema(
            node_type=node_type,
            display_name="Empty Latent Image",
            category="model/latent",
            inputs=(
                InputSpec("width", _INT, required=False, default=512),
                InputSpec("height", _INT, required=False, default=512),
                InputSpec("batch_size", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.empty_sd3_latent_image":
        return NodeSchema(
            node_type=node_type,
            display_name="Empty SD3 Latent Image",
            category="model/latent/stable diffusion",
            inputs=(
                InputSpec("width", _INT, required=False, default=1024),
                InputSpec("height", _INT, required=False, default=1024),
                InputSpec("batch_size", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.empty_chroma_radiance_latent_image":
        return NodeSchema(
            node_type=node_type,
            display_name="Empty Chroma Radiance Latent Image",
            category="model/latent/chroma radiance",
            inputs=(
                InputSpec("width", _INT, required=False, default=1024),
                InputSpec("height", _INT, required=False, default=1024),
                InputSpec("batch_size", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.empty_flux2_latent_image":
        return NodeSchema(
            node_type=node_type,
            display_name="Empty Flux 2 Latent",
            category="model/latent",
            inputs=(
                InputSpec("width", _INT, required=False, default=1024),
                InputSpec("height", _INT, required=False, default=1024),
                InputSpec("batch_size", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.empty_ltxav_latent":
        return NodeSchema(
            node_type=node_type,
            display_name="Empty LTX-2 AV Latent",
            category="latent/multi-stream",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("width", _INT, default=768),
                InputSpec("height", _INT, default=512),
                InputSpec("length", _INT, default=97),
                InputSpec("frame_rate", _INT, default=25),
                InputSpec("batch_size", _INT, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.empty_ltxv_latent":
        return NodeSchema(
            node_type=node_type,
            display_name="Empty LTX-Video Latent",
            category="latent/multi-stream",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("width", _INT, default=768),
                InputSpec("height", _INT, default=512),
                InputSpec("length", _INT, default=97),
                InputSpec("batch_size", _INT, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.ksampler":
        return NodeSchema(
            node_type=node_type,
            display_name="KSampler",
            category="model/sampling",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("seed", _INT, default=0),
                InputSpec("steps", _INT, default=20),
                InputSpec("cfg", _FLOAT, default=8.0),
                InputSpec("sampler_name", _COMBO, default="dinkster.euler"),
                InputSpec("scheduler", _COMBO, default="dinkster.simple"),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("latent_image", _DINKSTER_LATENT),
                InputSpec("denoise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.ksampler_advanced":
        return NodeSchema(
            node_type=node_type,
            display_name="KSampler (Advanced)",
            category="model/sampling",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("add_noise", _COMBO, default="enable"),
                InputSpec("noise_seed", _INT, default=0),
                InputSpec("steps", _INT, default=20),
                InputSpec("cfg", _FLOAT, default=8.0),
                InputSpec("sampler_name", _COMBO, default="dinkster.euler"),
                InputSpec("scheduler", _COMBO, default="dinkster.simple"),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("latent_image", _DINKSTER_LATENT),
                InputSpec("start_at_step", _INT, default=0),
                InputSpec("end_at_step", _INT, default=10_000),
                InputSpec("return_with_leftover_noise", _COMBO, default="disable"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.ksampler_select":
        return NodeSchema(
            node_type=node_type,
            display_name="KSamplerSelect",
            category="model/sampling/samplers",
            inputs=(InputSpec("sampler_name", _COMBO, default="dinkster.euler"),),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_dpmpp_3m_sde":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerDPMPP_3M_SDE",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("eta", _FLOAT, default=1.0, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
                InputSpec("noise_device", _COMBO, default="gpu", advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_dpmpp_2m_sde":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerDPMPP_2M_SDE",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("solver_type", _COMBO, default="midpoint"),
                InputSpec("eta", _FLOAT, default=1.0, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
                InputSpec("noise_device", _COMBO, default="gpu", advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_dpmpp_sde":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerDPMPP_SDE",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("eta", _FLOAT, default=1.0, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
                InputSpec("r", _FLOAT, default=0.5, advanced=True),
                InputSpec("noise_device", _COMBO, default="gpu", advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_dpmpp_2s_ancestral":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerDPMPP_2S_Ancestral",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("eta", _FLOAT, default=1.0),
                InputSpec("s_noise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_euler_ancestral":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerEulerAncestral",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("eta", _FLOAT, default=1.0, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_euler_ancestral_cfg_pp":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerEulerAncestralCFG++",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("eta", _FLOAT, default=1.0),
                InputSpec("s_noise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_lms":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerLMS",
            category="model/sampling/samplers",
            inputs=(InputSpec("order", _INT, default=4, advanced=True),),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_dpm_adaptative":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerDPMAdaptative",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("order", _INT, default=3, advanced=True),
                InputSpec("rtol", _FLOAT, default=0.05, advanced=True),
                InputSpec("atol", _FLOAT, default=0.0078, advanced=True),
                InputSpec("h_init", _FLOAT, default=0.05, advanced=True),
                InputSpec("pcoeff", _FLOAT, default=0.0, advanced=True),
                InputSpec("icoeff", _FLOAT, default=1.0, advanced=True),
                InputSpec("dcoeff", _FLOAT, default=0.0, advanced=True),
                InputSpec("accept_safety", _FLOAT, default=0.81, advanced=True),
                InputSpec("eta", _FLOAT, default=0.0, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_er_sde":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerER_SDE",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("solver_type", _COMBO, default="ER-SDE"),
                InputSpec("max_stage", _INT, default=3, advanced=True),
                InputSpec("eta", _FLOAT, default=1.0, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_seeds_2":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerSEEDS2",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("solver_type", _COMBO, default="phi_1"),
                InputSpec("eta", _FLOAT, default=1.0, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
                InputSpec("r", _FLOAT, default=0.5, advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.sampler_sa_solver":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerSASolver",
            category="model/sampling/samplers",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("eta", _FLOAT, default=1.0, advanced=True),
                InputSpec("sde_start_percent", _FLOAT, default=0.2, advanced=True),
                InputSpec("sde_end_percent", _FLOAT, default=0.8, advanced=True),
                InputSpec("s_noise", _FLOAT, default=1.0, advanced=True),
                InputSpec("predictor_order", _INT, default=3, advanced=True),
                InputSpec("corrector_order", _INT, default=4, advanced=True),
                InputSpec("use_pece", _BOOLEAN, default=False, advanced=True),
                InputSpec("simple_order_2", _BOOLEAN, default=False, advanced=True),
            ),
            outputs=(OutputSpec("sampler", _DINKSTER_SAMPLER),),
        )
    if node_type == "dinkster.basic_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="BasicScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("scheduler", _COMBO, default="dinkster.simple"),
                InputSpec("steps", _INT, default=20),
                InputSpec("denoise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.beta_sampling_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="BetaSamplingScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("steps", _INT, default=20),
                InputSpec("alpha", _FLOAT, default=0.6, advanced=True),
                InputSpec("beta", _FLOAT, default=0.6, advanced=True),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.sd_turbo_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="SDTurboScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("steps", _INT, default=1),
                InputSpec("denoise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.karras_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="KarrasScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", _INT, default=20),
                InputSpec("sigma_max", _FLOAT, default=14.614642, advanced=True),
                InputSpec("sigma_min", _FLOAT, default=0.0291675, advanced=True),
                InputSpec("rho", _FLOAT, default=7.0, advanced=True),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.exponential_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="ExponentialScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", _INT, default=20),
                InputSpec("sigma_max", _FLOAT, default=14.614642, advanced=True),
                InputSpec("sigma_min", _FLOAT, default=0.0291675, advanced=True),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.polyexponential_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="PolyexponentialScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", _INT, default=20),
                InputSpec("sigma_max", _FLOAT, default=14.614642, advanced=True),
                InputSpec("sigma_min", _FLOAT, default=0.0291675, advanced=True),
                InputSpec("rho", _FLOAT, default=1.0, advanced=True),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.laplace_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="LaplaceScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", _INT, default=20),
                InputSpec("sigma_max", _FLOAT, default=14.614642, advanced=True),
                InputSpec("sigma_min", _FLOAT, default=0.0291675, advanced=True),
                InputSpec("mu", _FLOAT, default=0.0, advanced=True),
                InputSpec("beta", _FLOAT, default=0.5, advanced=True),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.vp_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="VPScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", _INT, default=20),
                InputSpec("beta_d", _FLOAT, default=19.9, advanced=True),
                InputSpec("beta_min", _FLOAT, default=0.1, advanced=True),
                InputSpec("eps_s", _FLOAT, default=0.001, advanced=True),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.align_your_steps_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="AlignYourStepsScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("model_type", _COMBO, default="SD1"),
                InputSpec("steps", _INT, default=10),
                InputSpec("denoise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.gits_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="GITSScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("coeff", _FLOAT, default=1.20, advanced=True),
                InputSpec("steps", _INT, default=10),
                InputSpec("denoise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.optimal_steps_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="OptimalStepsScheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("model_type", _COMBO, default="FLUX"),
                InputSpec("steps", _INT, default=20),
                InputSpec("denoise", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.flux2_scheduler":
        return NodeSchema(
            node_type=node_type,
            display_name="Flux2Scheduler",
            category="model/sampling/schedulers",
            inputs=(
                InputSpec("steps", _INT, default=20),
                InputSpec("width", _INT, default=1024),
                InputSpec("height", _INT, default=1024),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.manual_sigmas":
        return NodeSchema(
            node_type=node_type,
            display_name="ManualSigmas",
            category="model/sampling/sigmas",
            inputs=(InputSpec("sigmas", _STRING, default="1, 0.5"),),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.split_sigmas":
        return NodeSchema(
            node_type=node_type,
            display_name="SplitSigmas",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec("sigmas", _DINKSTER_SIGMAS),
                InputSpec("step", _INT, default=0),
            ),
            outputs=(
                OutputSpec("high_sigmas", _DINKSTER_SIGMAS),
                OutputSpec("low_sigmas", _DINKSTER_SIGMAS),
            ),
        )
    if node_type == "dinkster.split_sigmas_denoise":
        return NodeSchema(
            node_type=node_type,
            display_name="SplitSigmasDenoise",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec("sigmas", _DINKSTER_SIGMAS),
                InputSpec("denoise", _FLOAT, default=1.0),
            ),
            outputs=(
                OutputSpec("high_sigmas", _DINKSTER_SIGMAS),
                OutputSpec("low_sigmas", _DINKSTER_SIGMAS),
            ),
        )
    if node_type == "dinkster.flip_sigmas":
        return NodeSchema(
            node_type=node_type,
            display_name="FlipSigmas",
            category="model/sampling/sigmas",
            inputs=(InputSpec("sigmas", _DINKSTER_SIGMAS),),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.set_first_sigma":
        return NodeSchema(
            node_type=node_type,
            display_name="SetFirstSigma",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec("sigmas", _DINKSTER_SIGMAS),
                InputSpec("sigma", _FLOAT, default=136.0),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.extend_intermediate_sigmas":
        return NodeSchema(
            node_type=node_type,
            display_name="ExtendIntermediateSigmas",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec("sigmas", _DINKSTER_SIGMAS),
                InputSpec("steps", _INT, default=2),
                InputSpec("start_at_sigma", _FLOAT, default=-1.0),
                InputSpec("end_at_sigma", _FLOAT, default=12.0),
                InputSpec("spacing", _COMBO, default="linear"),
            ),
            outputs=(OutputSpec("sigmas", _DINKSTER_SIGMAS),),
        )
    if node_type == "dinkster.sampling_percent_to_sigma":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplingPercentToSigma",
            category="model/sampling/sigmas",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("sampling_percent", _FLOAT, default=0.0),
                InputSpec("return_actual_sigma", _BOOLEAN, default=False),
            ),
            outputs=(OutputSpec("sigma_value", _FLOAT),),
        )
    if node_type == "dinkster.basic_guider":
        return NodeSchema(
            node_type=node_type,
            display_name="Basic Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("conditioning", _DINKSTER_CONDITIONING),
            ),
            outputs=(OutputSpec("guider", _DINKSTER_GUIDER),),
        )
    if node_type == "dinkster.cfg_guider":
        return NodeSchema(
            node_type=node_type,
            display_name="CFG Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("cfg", _FLOAT, default=8.0),
            ),
            outputs=(OutputSpec("guider", _DINKSTER_GUIDER),),
        )
    if node_type == "dinkster.dual_cfg_guider":
        return NodeSchema(
            node_type=node_type,
            display_name="Dual CFG Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("cond1", _DINKSTER_CONDITIONING),
                InputSpec("cond2", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("cfg_conds", _FLOAT, default=8.0),
                InputSpec("cfg_cond2_negative", _FLOAT, default=8.0),
                InputSpec("style", _COMBO, default="regular"),
            ),
            outputs=(OutputSpec("guider", _DINKSTER_GUIDER),),
        )
    if node_type == "dinkster.ltxv_dual_cfg_guider":
        return NodeSchema(
            node_type=node_type,
            display_name="LTXV Dual CFG Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("video_cfg", _FLOAT, default=3.0),
                InputSpec("audio_cfg", _FLOAT, default=7.0),
            ),
            outputs=(OutputSpec("guider", _DINKSTER_GUIDER),),
        )
    if node_type == "dinkster.perp_neg_guider":
        return NodeSchema(
            node_type=node_type,
            display_name="Perp-Neg Guider",
            category="model/sampling/guiders",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("empty_conditioning", _DINKSTER_CONDITIONING),
                InputSpec("cfg", _FLOAT, default=8.0),
                InputSpec("neg_scale", _FLOAT, default=1.0),
            ),
            outputs=(OutputSpec("guider", _DINKSTER_GUIDER),),
        )
    if node_type == "dinkster.disable_noise":
        return NodeSchema(
            node_type=node_type,
            display_name="DisableNoise",
            category="model/sampling/noise",
            outputs=(OutputSpec("noise", _DINKSTER_NOISE),),
        )
    if node_type == "dinkster.random_noise":
        return NodeSchema(
            node_type=node_type,
            display_name="RandomNoise",
            category="model/sampling/noise",
            inputs=(InputSpec("noise_seed", _INT, default=0),),
            outputs=(OutputSpec("noise", _DINKSTER_NOISE),),
        )
    if node_type == "dinkster.add_noise":
        return NodeSchema(
            node_type=node_type,
            display_name="AddNoise",
            category="model/sampling/noise",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("noise", _DINKSTER_NOISE),
                InputSpec("sigmas", _DINKSTER_SIGMAS),
                InputSpec("latent_image", _DINKSTER_LATENT),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.sampler_custom":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerCustom",
            category="model/sampling/custom",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("add_noise", _BOOLEAN, default=True),
                InputSpec("noise_seed", _INT, default=0),
                InputSpec("cfg", _FLOAT, default=8.0),
                InputSpec("positive", _DINKSTER_CONDITIONING),
                InputSpec("negative", _DINKSTER_CONDITIONING),
                InputSpec("sampler", _DINKSTER_SAMPLER),
                InputSpec("sigmas", _DINKSTER_SIGMAS),
                InputSpec("latent_image", _DINKSTER_LATENT),
            ),
            outputs=(
                OutputSpec("output", _DINKSTER_LATENT),
                OutputSpec("denoised_output", _DINKSTER_LATENT),
            ),
            emits_previews=True,
        )
    if node_type == "dinkster.sampler_custom_advanced":
        return NodeSchema(
            node_type=node_type,
            display_name="SamplerCustomAdvanced",
            category="model/sampling/custom",
            inputs=(
                InputSpec("noise", _DINKSTER_NOISE),
                InputSpec("guider", _DINKSTER_GUIDER),
                InputSpec("sampler", _DINKSTER_SAMPLER),
                InputSpec("sigmas", _DINKSTER_SIGMAS),
                InputSpec("latent_image", _DINKSTER_LATENT),
            ),
            outputs=(
                OutputSpec("output", _DINKSTER_LATENT),
                OutputSpec("denoised_output", _DINKSTER_LATENT),
            ),
            emits_previews=True,
        )
    if node_type == "dinkster.vae_decode":
        return NodeSchema(
            node_type=node_type,
            display_name="VAE Decode",
            category="model/latent",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("vae", _DINKSTER_VAE),
            ),
            outputs=(OutputSpec("image", _DINKSTER_IMAGE),),
        )
    if node_type == "dinkster.vae_decode_tiled":
        return NodeSchema(
            node_type=node_type,
            display_name="VAE Decode (Tiled)",
            category="model/latent",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec("tile_size", _INT, default=512, advanced=True),
                InputSpec("overlap", _INT, default=64, advanced=True),
                InputSpec("temporal_size", _INT, default=64, advanced=True),
                InputSpec("temporal_overlap", _INT, default=8, advanced=True),
            ),
            outputs=(OutputSpec("image", _DINKSTER_IMAGE),),
        )
    if node_type == "dinkster.vae_encode":
        return NodeSchema(
            node_type=node_type,
            display_name="VAE Encode",
            category="model/latent",
            inputs=(
                InputSpec("pixels", _DINKSTER_IMAGE),
                InputSpec("vae", _DINKSTER_VAE),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.vae_encode_tiled":
        return NodeSchema(
            node_type=node_type,
            display_name="VAE Encode (Tiled)",
            category="model/latent",
            inputs=(
                InputSpec("pixels", _DINKSTER_IMAGE),
                InputSpec("vae", _DINKSTER_VAE),
                InputSpec("tile_size", _INT, default=512, advanced=True),
                InputSpec("overlap", _INT, default=64, advanced=True),
                InputSpec("temporal_size", _INT, default=64, advanced=True),
                InputSpec("temporal_overlap", _INT, default=8, advanced=True),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.seedvr2_preprocess":
        return NodeSchema(
            node_type=node_type,
            display_name="Pre-Process SeedVR2 Input",
            category="image/pre-processors",
            inputs=(InputSpec("resized_images", _DINKSTER_IMAGE),),
            outputs=(OutputSpec("images", _DINKSTER_IMAGE),),
        )
    if node_type == "dinkster.seedvr2_postprocess":
        return NodeSchema(
            node_type=node_type,
            display_name="Post-Process SeedVR2 Output",
            category="image/post-processors",
            inputs=(
                InputSpec("images", _DINKSTER_IMAGE),
                InputSpec("original_resized_images", _DINKSTER_IMAGE),
                InputSpec("color_correction_method", _COMBO, default="lab"),
            ),
            outputs=(OutputSpec("images", _DINKSTER_IMAGE),),
        )
    if node_type == "dinkster.seedvr2_conditioning":
        return NodeSchema(
            node_type=node_type,
            display_name="Apply SeedVR2 Conditioning",
            category="model/conditioning",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("vae_conditioning", _DINKSTER_LATENT),
            ),
            outputs=(
                OutputSpec("positive", _DINKSTER_CONDITIONING),
                OutputSpec("negative", _DINKSTER_CONDITIONING),
            ),
        )
    if node_type == "dinkster.seedvr2_temporal_chunk":
        return NodeSchema(
            node_type=node_type,
            display_name="Split SeedVR2 Latent",
            category="model/latent/batch",
            inputs=(
                InputSpec("latent", _DINKSTER_LATENT),
                InputSpec("temporal_overlap", _INT, default=0),
            ),
            combos=(
                DynamicComboSpec(
                    "chunking_mode",
                    options=(
                        DynamicComboOption("auto"),
                        DynamicComboOption(
                            "manual",
                            (InputSpec("frames_per_chunk", _INT, default=21),),
                        ),
                    ),
                    default="auto",
                ),
            ),
            outputs=(
                OutputSpec("latents", _DINKSTER_LATENT_LIST),
                OutputSpec("temporal_overlap", _INT),
            ),
        )
    if node_type == "dinkster.seedvr2_temporal_merge":
        return NodeSchema(
            node_type=node_type,
            display_name="Merge SeedVR2 Latents",
            category="model/latent/batch",
            inputs=(
                InputSpec("latents", _DINKSTER_LATENT_LIST),
                InputSpec("temporal_overlap", _INT, default=0),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.combine":
        return NodeSchema(
            node_type=node_type,
            display_name="Combine Latents",
            category="model/latent/advanced",
            inputs=(
                InputSpec("samples1", _DINKSTER_LATENT),
                InputSpec("samples2", _DINKSTER_LATENT),
                InputSpec("operation", _COMBO, required=False, default="add"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.mix":
        return NodeSchema(
            node_type=node_type,
            display_name="Mix Latents",
            category="model/latent/advanced",
            inputs=(
                InputSpec("samples1", _DINKSTER_LATENT),
                InputSpec("samples2", _DINKSTER_LATENT),
                InputSpec("operation", _COMBO, required=False, default="interpolate"),
                InputSpec("factor", _FLOAT, required=False, default=0.5),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.multiply":
        return NodeSchema(
            node_type=node_type,
            display_name="Multiply Latent",
            category="model/latent/advanced",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("multiplier", _FLOAT, required=False, default=1.0),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.rotate":
        return NodeSchema(
            node_type=node_type,
            display_name="Rotate Latent",
            category="model/latent/transform",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("angle", _COMBO, required=False, default="none"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.flip":
        return NodeSchema(
            node_type=node_type,
            display_name="Flip Latent",
            category="model/latent/transform",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("axis", _COMBO, required=False, default="vertical"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.crop":
        return NodeSchema(
            node_type=node_type,
            display_name="Crop Latent",
            category="model/latent/transform",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("width", _INT, required=False, default=512),
                InputSpec("height", _INT, required=False, default=512),
                InputSpec("x", _INT, required=False, default=0),
                InputSpec("y", _INT, required=False, default=0),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.resize":
        return NodeSchema(
            node_type=node_type,
            display_name="Resize Latent",
            category="model/latent",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("method", _COMBO, required=False, default="nearest-exact"),
                InputSpec("width", _INT, required=False, default=512),
                InputSpec("height", _INT, required=False, default=512),
                InputSpec("crop", _COMBO, required=False, default="disabled"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.resize_by":
        return NodeSchema(
            node_type=node_type,
            display_name="Resize Latent By",
            category="model/latent",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("method", _COMBO, required=False, default="nearest-exact"),
                InputSpec("scale_by", _FLOAT, required=False, default=1.5),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.composite":
        return NodeSchema(
            node_type=node_type,
            display_name="Composite Latents",
            category="model/latent",
            inputs=(
                InputSpec("destination", _DINKSTER_LATENT),
                InputSpec("source", _DINKSTER_LATENT),
                InputSpec("x", _INT, required=False, default=0),
                InputSpec("y", _INT, required=False, default=0),
                InputSpec("feather", _INT, required=False, default=0),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.composite_masked":
        return NodeSchema(
            node_type=node_type,
            display_name="Composite Latents (Masked)",
            category="model/latent",
            inputs=(
                InputSpec("destination", _DINKSTER_LATENT),
                InputSpec("source", _DINKSTER_LATENT),
                InputSpec("x", _INT, required=False, default=0),
                InputSpec("y", _INT, required=False, default=0),
                InputSpec("resize_source", _BOOLEAN, required=False, default=False),
                InputSpec("mask", _DINKSTER_MASK, required=False),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.concat":
        return NodeSchema(
            node_type=node_type,
            display_name="Concat Latents",
            category="model/latent/advanced",
            inputs=(
                InputSpec("samples1", _DINKSTER_LATENT),
                InputSpec("samples2", _DINKSTER_LATENT),
                InputSpec("dim", _COMBO, required=False, default="x"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.cut":
        return NodeSchema(
            node_type=node_type,
            display_name="Cut Latent",
            category="model/latent/advanced",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("dim", _COMBO, required=False, default="x"),
                InputSpec("index", _INT, required=False, default=0),
                InputSpec("amount", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.cut_to_batch":
        return NodeSchema(
            node_type=node_type,
            display_name="Cut Latent to Batch",
            category="model/latent/advanced",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("dim", _COMBO, required=False, default="t"),
                InputSpec("slice_size", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.from_batch":
        return NodeSchema(
            node_type=node_type,
            display_name="Latent From Batch",
            category="model/latent/batch",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("batch_index", _INT, required=False, default=0),
                InputSpec("length", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.repeat":
        return NodeSchema(
            node_type=node_type,
            display_name="Repeat Latent Batch",
            category="model/latent/batch",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("amount", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.seed_behavior":
        return NodeSchema(
            node_type=node_type,
            display_name="Latent Batch Seed Behavior",
            category="model/latent/advanced",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("behavior", _COMBO, required=False, default="fixed"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.batch":
        return NodeSchema(
            node_type=node_type,
            display_name="Batch Latents",
            category="model/latent/batch",
            input_families=(
                InputFamilySpec(
                    "latents",
                    _DINKSTER_LATENT,
                    min_members=1,
                    max_members=50,
                    member_prefix="latent_",
                ),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.rebatch":
        return NodeSchema(
            node_type=node_type,
            display_name="Rebatch Latents",
            category="model/latent/batch",
            inputs=(
                InputSpec("latents", _DINKSTER_LATENT_LIST),
                InputSpec("batch_size", _INT, required=False, default=1),
            ),
            outputs=(OutputSpec("latents", _DINKSTER_LATENT_LIST),),
        )
    if node_type == "dinkster.latent.set_noise_mask":
        return NodeSchema(
            node_type=node_type,
            display_name="Set Latent Noise Mask",
            category="model/latent",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("mask", _DINKSTER_MASK),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.replace_frames":
        return NodeSchema(
            node_type=node_type,
            display_name="Replace Video Latent Frames",
            category="model/latent/batch",
            inputs=(
                InputSpec("destination", _DINKSTER_LATENT),
                InputSpec("source", _DINKSTER_LATENT, required=False),
                InputSpec("index", _INT, required=False, default=0),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.apply_operation":
        return NodeSchema(
            node_type=node_type,
            display_name="Latent Apply Operation",
            category="model/latent/advanced/operations",
            inputs=(
                InputSpec("samples", _DINKSTER_LATENT),
                InputSpec("operation", _DINKSTER_LATENT_OPERATION),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.operation_tonemap_reinhard":
        return NodeSchema(
            node_type=node_type,
            display_name="Latent Operation Tonemap Reinhard",
            category="model/latent/advanced/operations",
            inputs=(InputSpec("multiplier", _FLOAT, required=False, default=1.0),),
            outputs=(OutputSpec("operation", _DINKSTER_LATENT_OPERATION),),
        )
    if node_type == "dinkster.latent.operation_sharpen":
        return NodeSchema(
            node_type=node_type,
            display_name="Latent Operation Sharpen",
            category="model/latent/advanced/operations",
            inputs=(
                InputSpec("sharpen_radius", _INT, required=False, default=9),
                InputSpec("sigma", _FLOAT, required=False, default=1.0),
                InputSpec("alpha", _FLOAT, required=False, default=0.1),
            ),
            outputs=(OutputSpec("operation", _DINKSTER_LATENT_OPERATION),),
        )
    if node_type == "dinkster.latent.apply_operation_cfg":
        return NodeSchema(
            node_type=node_type,
            display_name="Latent Apply Operation CFG",
            category="model/latent/advanced/operations",
            inputs=(
                InputSpec("model", _DINKSTER_MODEL),
                InputSpec("operation", _DINKSTER_LATENT_OPERATION),
            ),
            outputs=(OutputSpec("model", _DINKSTER_MODEL),),
        )
    if node_type == "dinkster.latent.generate_noise":
        return NodeSchema(
            node_type=node_type,
            display_name="Generate Noise",
            category="model/latent/advanced",
            inputs=(
                InputSpec("width", _INT, required=False, default=512),
                InputSpec("height", _INT, required=False, default=512),
                InputSpec("batch_size", _INT, required=False, default=1),
                InputSpec("seed", _INT, required=False, default=123),
                InputSpec("multiplier", _FLOAT, required=False, default=1.0),
                InputSpec("constant_batch_noise", _BOOLEAN, required=False, default=False),
                InputSpec("normalize", _BOOLEAN, required=False, default=False),
                InputSpec("model", _DINKSTER_MODEL, required=False),
                InputSpec("sigmas", _DINKSTER_SIGMAS, required=False),
                InputSpec("latent_channels", _COMBO, required=False, default="4"),
                InputSpec("shape", _COMBO, required=False, default="BCHW"),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    if node_type == "dinkster.latent.inject_noise":
        return NodeSchema(
            node_type=node_type,
            display_name="Inject Noise To Latent",
            category="model/latent/advanced",
            inputs=(
                InputSpec("latents", _DINKSTER_LATENT),
                InputSpec("strength", _FLOAT, required=False, default=0.1),
                InputSpec("noise", _DINKSTER_LATENT),
                InputSpec("normalize", _BOOLEAN, required=False, default=False),
                InputSpec("average", _BOOLEAN, required=False, default=False),
                InputSpec("mask", _DINKSTER_MASK, required=False),
                InputSpec("mix_randn_amount", _FLOAT, required=False, default=0.0),
                InputSpec("seed", _INT, required=False, default=123),
            ),
            outputs=(OutputSpec("latent", _DINKSTER_LATENT),),
        )
    raise ValueError(f"unknown generation provider node type {node_type!r}")


def _require_provider_runtime(value: object, input_id: str) -> NativeRuntimeHandle:
    inference = importlib.import_module("dinkster_inference")
    inference.require_inference_runtime_handle(value, input_id)
    return _native_handle(value, input_id)


def _runtime_sampling_shift(
    runtime: object,
    sampling_shift: float | None,
) -> float | None:
    if sampling_shift is None:
        return None
    if not getattr(runtime, "supports_sampling_shift", False):
        raise TypeError("runtime does not support a sampling shift")
    return sampling_shift


def _bind_sampling_shift(
    method: Callable[..., Any], sampling_shift: float | None
) -> Callable[..., Any]:
    if sampling_shift is None:
        return method
    return partial(method, sampling_shift=sampling_shift)


@dataclass(frozen=True)
class _ShiftedCustomSamplingRuntime:
    shift: float
    runtime: object

    @property
    def _space(self) -> Any:
        inference = importlib.import_module("dinkster_inference")
        return inference.FlowSigmas(shift=self.shift)

    def custom_sampling_sigmas(
        self,
        scheduler_id: str,
        steps: int,
        denoise: float,
        *,
        device: object | None = None,
    ) -> tuple[float, ...]:
        inference = importlib.import_module("dinkster_inference")
        if device is None:
            scheduler = _inference_registries(inference).schedulers.get(scheduler_id)
            if scheduler is None:
                raise ValueError(f"unknown scheduler {scheduler_id!r}")
            return inference.sampling_sigmas(scheduler, self._space, steps, denoise=denoise)
        schedules = importlib.import_module("dinkster_inference_torch.schedules")
        scheduler = schedules.torch_scheduler_registry().get(scheduler_id)
        if scheduler is None:
            raise ValueError(f"unknown scheduler {scheduler_id!r}")
        return inference.sampling_sigmas(
            schedules.scheduler_on_device(scheduler, device),
            self._space,
            steps,
            denoise=denoise,
        )

    def custom_sampling_beta_sigmas(
        self,
        steps: int,
        alpha: float,
        beta: float,
        *,
        device: object | None = None,
    ) -> tuple[float, ...]:
        schedules = importlib.import_module("dinkster_inference_torch.schedules")
        return schedules.custom_beta_sigmas(self._space, steps, alpha, beta, device=device)

    def custom_sampling_sd_turbo_sigmas(
        self, steps: int, denoise: float, *, device: object | None = None
    ) -> tuple[float, ...]:
        schedules = importlib.import_module("dinkster_inference_torch.schedules")
        return schedules.sd_turbo_sigmas(self._space, steps, denoise, device=device)

    def custom_sampling_percent_to_sigma(
        self,
        percent: float,
        *,
        return_actual_sigma: bool,
    ) -> float:
        schedules = importlib.import_module("dinkster_inference_torch.schedules")
        return schedules.custom_percent_to_sigma(
            self._space,
            self._space.percent_to_sigma,
            percent,
            return_actual_sigma=return_actual_sigma,
        )

    def custom_sampling_add_noise(self, latent: object, noise: object, sigma: float) -> object:
        add_noise = getattr(self.runtime, "custom_sampling_add_noise", None)
        if not callable(add_noise):
            raise TypeError("model runtime does not support AddNoise")
        return add_noise(latent, noise, sigma)


def _require_custom_sampling_runtime(
    value: object, node_name: str
) -> tuple[Any, float | None, object]:
    model, applications = _application_chain_model(value, "model")
    if applications:
        raise ValueError(f"{node_name} does not accept a model application chain")
    sampling_space = _native_model_sampling_space(model)
    if sampling_space is not None:
        handle = _native_handle(model, "model")
        return _sampling_space_runtime(handle.runtime, sampling_space), None, handle.load_device
    sampling_shift = None
    if isinstance(model, _NativeModelOverlay):
        (
            handle,
            overlays,
            resolvers,
            control,
            sampling_shift,
            transforms,
            context_windows,
            radiance_options,
        ) = _native_model(model, "model")
        inference = importlib.import_module("dinkster_inference")
        runtime = handle.runtime
        if handle.recipe.family_id in (inference.CHROMA.id, inference.CHROMA_RADIANCE.id):
            if (
                overlays
                or resolvers
                or control is not None
                or transforms
                or context_windows is not None
                or (sampling_shift is None and not radiance_options)
            ):
                raise ValueError(f"{node_name} does not accept a model overlay")
            descriptor = _active_inference_registries().components.get(handle.recipe.family_id)
            if descriptor is None or descriptor.execution_options is None:
                raise TypeError("model must be a native Chroma diffusion component")
            sampling_runtime = getattr(runtime, "component_sampling_runtime", runtime)
            return (
                _component_runtime_with_options(
                    runtime,
                    descriptor,
                    handle.recipe.family_id,
                    handle.recipe.runtime_identity,
                    sampling_runtime.assembled.compute_dtype("diffusion"),
                    sampling_shift,
                    radiance_options,
                ),
                None,
                handle.load_device,
            )
        if (
            control is not None
            and handle.recipe.family_id == inference.Z_IMAGE_CONFIG.family_id
            and sampling_shift is None
            and not overlays
            and not resolvers
            and not transforms
            and context_windows is None
            and not radiance_options
        ):
            if handle.recipe.runtime_identity != runtime.runtime_identity or not isinstance(
                runtime, inference.CustomSamplingRuntime
            ):
                raise TypeError("model must be a native custom sampling runtime")
            return runtime, None, handle.load_device
        if (
            sampling_shift is not None
            and getattr(runtime, "supports_sampling_shift", False)
            and not overlays
            and not resolvers
            and control is None
            and not transforms
            and context_windows is None
            and not radiance_options
        ):
            if handle.recipe.runtime_identity != runtime.runtime_identity or not isinstance(
                runtime, inference.CustomSamplingRuntime
            ):
                raise TypeError("model must be a native custom sampling runtime")
            return (
                runtime,
                _runtime_sampling_shift(runtime, sampling_shift),
                handle.load_device,
            )
        if (
            overlays
            or resolvers
            or control is not None
            or transforms
            or context_windows is not None
            or radiance_options
        ):
            raise ValueError(f"{node_name} does not accept a model overlay")
        runtime = handle.runtime
        if sampling_shift is None:
            raise ValueError(f"{node_name} does not accept a model overlay")
        if handle.recipe.family_id == inference.LUMINA2_CONFIG.family_id:
            model = handle
        elif not isinstance(
            runtime, inference.CustomSamplingRuntime
        ) or not inference.is_flow_parameterization(runtime.family.sampling.parameterization):
            raise TypeError("ModelSamplingSD3 requires a flow custom-sampling runtime")
        else:
            return _ShiftedCustomSamplingRuntime(sampling_shift, runtime), None, handle.load_device
    handle = _require_provider_runtime(model, "model")
    inference = importlib.import_module("dinkster_inference")
    runtime = _minimax_h3_schedule_runtime(handle, inference)
    if runtime is None:
        runtime = handle.runtime
    if not isinstance(runtime, inference.CustomSamplingRuntime):
        raise TypeError(f"model family {runtime.family.id!r} does not support custom sampling")
    return runtime, sampling_shift, handle.load_device


def _require_base_custom_sampling_runtime(
    value: object, node_name: str
) -> tuple[Any, float | None, object]:
    model, _applications = _application_chain_model(value, "model")
    return _require_custom_sampling_runtime(model, node_name)


def _materialize_provider_conditioning(
    value: object, input_id: str, handle: NativeRuntimeHandle
) -> list[list[object]]:
    inference_torch = importlib.import_module("dinkster_inference_torch")
    conditioning = inference_torch.materialize_basic_conditioning(
        value,
        device=handle.load_device,
    )
    metadata: dict[str, object] = {}
    if conditioning.pooled is not None:
        metadata["pooled_output"] = conditioning.pooled
    return [[conditioning.embeddings, metadata]]


def _prepare_provider_multistream_conditioning(
    value: object, input_id: str, runtime: Any, inference: Any
) -> list[list[object]]:
    prepare = getattr(runtime, "prepare_conditioning", None)
    if not callable(prepare):
        raise TypeError(f"{input_id} runtime prepare_conditioning must be callable")
    carrier, frame_rate = _split_ltx_frame_rate(value, inference)
    payload = prepare(carrier) if frame_rate is None else prepare(carrier, frame_rate=frame_rate)
    return [
        [
            inference.PreparedMultiStreamConditioning(runtime.conditioning_identity, payload),
            {},
        ]
    ]


def _prepare_provider_conditioning(
    value: object, input_id: str, runtime: Any, inference: Any
) -> list[list[object]]:
    prepare = getattr(runtime, "prepare_single_stream_conditioning", None)
    if not callable(prepare):
        raise TypeError(f"{input_id} runtime prepare_single_stream_conditioning must be callable")
    conditioning = prepare(value)
    if not isinstance(conditioning, inference.Conditioning):
        raise TypeError(f"{input_id} runtime must prepare a Conditioning value")
    typed = cast("Any", conditioning)
    return [[typed.embeddings, {_NATIVE_PREPARED_CONDITIONING_KEY: conditioning}]]


def _generation_lora_mode(execution_mode: str) -> str:
    if execution_mode == "attach":
        raise ValueError(
            "generation LoRA providers cannot attach patches to canonical conditioning; "
            "use 'auto' or 'precalculate'"
        )
    return "precalculate" if execution_mode == "auto" else execution_mode


class GenerationEmptyTrellis2LatentStructure(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.empty_trellis2_latent_structure")

    @classmethod
    def execute(cls, *, batch_size: int = 1) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_empty_trellis2_latent_structure(batch_size=batch_size)
        )


class GenerationTrellis2Conditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.trellis2_conditioning")

    @classmethod
    def execute(cls, *, clip_vision_model: object, image: object) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_trellis2_conditioning(
                clip_vision_model=clip_vision_model,
                image=image,
                pixal3d=False,
                camera_angle_x=49.13,
            )
        )


class GenerationPixal3DConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.pixal3d_conditioning")

    @classmethod
    def execute(
        cls,
        *,
        clip_vision_model: object,
        image: object,
        camera_angle_x: float = 49.13,
    ) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_trellis2_conditioning(
                clip_vision_model=clip_vision_model,
                image=image,
                pixal3d=True,
                camera_angle_x=camera_angle_x,
            )
        )


class GenerationVaeDecodeStructureTrellis2(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vae_decode_structure_trellis2")

    @classmethod
    def execute(
        cls, *, samples: object, vae: object, resolution: str = "32"
    ) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_vae_decode_structure_trellis2(
                samples=samples,
                vae=vae,
                resolution=resolution,
            )
        )


class GenerationTrellis2ShapeStage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.trellis2_shape_stage")

    @classmethod
    def execute(cls, *, positive: object, negative: object, voxel: object) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_trellis2_shape_stage(
                positive=positive,
                negative=negative,
                voxel=voxel,
            )
        )


class GenerationTrellis2UpsampleStage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.trellis2_upsample_stage")

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        shape_latent: object,
        vae: object,
        target_resolution: int = 1024,
    ) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_trellis2_upsample_stage(
                positive=positive,
                negative=negative,
                shape_latent=shape_latent,
                vae=vae,
                target_resolution=target_resolution,
            )
        )


class GenerationVaeDecodeShapeTrellis(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vae_decode_shape_trellis")

    @classmethod
    def execute(cls, *, samples: object, vae: object) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_vae_decode_shape_trellis(samples=samples, vae=vae)
        )


class GenerationTrellis2TextureStage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.trellis2_texture_stage")

    @classmethod
    def execute(
        cls, *, positive: object, negative: object, shape_latent: object
    ) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_trellis2_texture_stage(
                positive=positive,
                negative=negative,
                shape_latent=shape_latent,
            )
        )


class GenerationVaeDecodeTextureTrellis(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vae_decode_texture_trellis")

    @classmethod
    def execute(
        cls,
        *,
        samples: object,
        vae: object,
        shape_subdivides: object,
    ) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_vae_decode_texture_trellis(
                samples=samples,
                vae=vae,
                shape_subdivides=shape_subdivides,
            )
        )


def _model3d_provider_schema(node_type: str) -> NodeSchema:
    try:
        return _MODEL3D_PROVIDER_SCHEMAS[node_type]
    except KeyError as error:
        raise RuntimeError(f"unknown model3d provider schema {node_type!r}") from error


def _enroll_auxiliary_model(
    asset: AssetRef,
    module: object,
    role: str,
    *,
    load_device: object | None = None,
) -> NativeComponentHandle:
    torch = _torch()
    device = select_load_device(torch) if load_device is None else load_device
    coordinator = default_native_residency()
    mechanism = coordinator.enroll_component(
        module,
        load_device=device,
        offload_device=torch.device("cpu"),
    )
    handle = NativeComponentHandle(
        module,
        mechanism,
        device,
        resource_identity=f"native:{role}:{asset.digest}",
        coordinator=coordinator,
    )
    pool = default_pool()
    pool.label(handle, asset.name)
    handle.attach_pool(pool)
    return handle


@dataclass(frozen=True, eq=False)
class _NativeGeometryModel:
    handle: NativeComponentHandle
    compute_dtype: Any
    version: str
    mask_threshold: float
    num_tokens_range: tuple[int, int]

    @property
    def _dinkster_resident_owner(self) -> NativeComponentHandle:
        return self.handle


@dataclass(frozen=True, eq=False)
class _NativeBackgroundRemovalModel:
    handle: NativeComponentHandle
    compute_dtype: Any
    image_size: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]

    @property
    def _dinkster_resident_owner(self) -> NativeComponentHandle:
        return self.handle


def _load_comfy_state_dict(asset: AssetRef) -> dict[str, object]:
    checkpoint = importlib.import_module("dinkster_inference_torch.checkpoint")
    state_dict = checkpoint.load_checkpoint(_component_candidate_path(asset))
    if not isinstance(state_dict, dict):
        raise TypeError(f"{asset.name} must contain a tensor state dictionary")
    return cast("dict[str, object]", state_dict)


def _auxiliary_model_dtypes(
    state_dict: Mapping[str, object],
) -> tuple[Any, Any]:
    torch = _torch()
    storage_dtype = next(
        (
            cast("Any", value).dtype
            for value in state_dict.values()
            if isinstance(value, torch.Tensor) and cast("Any", value).is_floating_point()
        ),
        torch.float32,
    )
    return storage_dtype, torch.float32


def _load_geometry_component(
    state_dict: Mapping[str, object],
) -> tuple[object, Any, str, float, tuple[int, int]]:
    torch = _torch()
    model_module = importlib.import_module("dinkster_inference_torch.moge")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    _, compute_dtype = _auxiliary_model_dtypes(state_dict)
    with torch.device("meta"):
        module = model_module.build_from_state_dict(
            dict(state_dict),
            operations=inference_torch.CastOperations(compute_dtype),
        ).eval()
    version = "v2" if hasattr(module, "encoder") else "v1"
    mask_threshold = float(getattr(module, "mask_threshold", 0.5))
    default_range = (1200, 2500 if version == "v1" else 3600)
    raw_range = getattr(module, "num_tokens_range", default_range)
    token_range = (int(raw_range[0]), int(raw_range[1]))
    return module, compute_dtype, version, mask_threshold, token_range


def _load_background_removal_component(
    state_dict: Mapping[str, object],
    load_device: Any,
) -> tuple[object, Any, int, tuple[float, ...], tuple[float, ...]]:
    torch = _torch()
    model_module = importlib.import_module("dinkster_inference_torch.birefnet")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    storage_dtype, compute_dtype = _auxiliary_model_dtypes(state_dict)
    with torch.device("meta"):
        module = (
            model_module.BiRefNet(
                operations=inference_torch.CastOperations(compute_dtype),
            )
            .to(dtype=storage_dtype)
            .eval()
        )
    module.to_empty(device=torch.device("cpu"))
    module.load_state_dict(dict(state_dict), strict=True)
    # ComfyUI birefnet.json at c67885b: config controls preprocessing, not architecture.
    return module, compute_dtype, 1024, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)


def _triangle_mesh_batch(value: object) -> object:
    inference = importlib.import_module("dinkster_inference")
    if type(value) is inference.TriangleMeshBatch:
        return value
    required = ("vertices", "faces")
    if any(not hasattr(value, name) for name in required):
        raise TypeError("mesh operation did not return a triangle mesh")
    mesh = cast("Any", value)
    return inference.TriangleMeshBatch(
        vertices=mesh.vertices,
        faces=mesh.faces,
        uvs=getattr(mesh, "uvs", None),
        vertex_colors=getattr(mesh, "vertex_colors", None),
        texture=getattr(mesh, "texture", None),
        metallic_roughness=getattr(mesh, "metallic_roughness", None),
        vertex_counts=getattr(mesh, "vertex_counts", None),
        face_counts=getattr(mesh, "face_counts", None),
        unlit=bool(getattr(mesh, "unlit", False)),
        normals=getattr(mesh, "normals", None),
        tangents=getattr(mesh, "tangents", None),
        normal_map=getattr(mesh, "normal_map", None),
        occlusion_in_mr=bool(getattr(mesh, "occlusion_in_mr", False)),
        material=getattr(mesh, "material", None),
        emissive=getattr(mesh, "emissive", None),
    )


def _native_mesh_operation(
    operation: str, *, offload_models: bool = False, **inputs: object
) -> object:
    mesh_operations = importlib.import_module("dinkster_inference_torch.mesh_operations")
    residency = default_native_residency()
    with residency.placement_pass():
        if offload_models:
            torch = _torch()
            device = select_current_device(torch)
            if device.type != "cpu":
                manager = residency.manager
                memory = manager.policy_memory(device).free_total
                loaded = sum(
                    mechanism.loaded_bytes()
                    for mechanism in manager.registered()
                    if mechanism.load_device == device
                )
                manager.free(memory + loaded, device)
                manager.empty_cache(device)
        return getattr(mesh_operations, operation)(**inputs)


class GenerationLoadGeometryModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.load_geometry_model")

    @classmethod
    def execute(cls, *, model: object) -> Mapping[str, object]:
        if not isinstance(model, AssetRef):
            raise TypeError("model must be an AssetRef")
        torch = _torch()
        load_device = select_load_device(torch)
        module, compute_dtype, version, mask_threshold, token_range = _load_geometry_component(
            _load_comfy_state_dict(model),
        )
        handle = _enroll_auxiliary_model(
            model,
            module,
            "geometry",
            load_device=load_device,
        )
        value = _NativeGeometryModel(
            handle,
            compute_dtype,
            version,
            mask_threshold,
            token_range,
        )
        return cls.outputs(model=value)


def _run_geometry_model(
    resource: _NativeGeometryModel,
    image: object,
    resolution_level: int,
    fov_x_degrees: float,
    batch_size: int,
    force_projection: bool,
    apply_mask: bool,
) -> Mapping[str, object]:
    torch = _torch()
    if type(image) is not torch.Tensor:
        raise TypeError("image must be an exact torch.Tensor")
    if not 0 <= resolution_level <= 9:
        raise ValueError("resolution_level must be in [0, 9]")
    if not 0.0 <= fov_x_degrees <= 170.0:
        raise ValueError("fov_x_degrees must be in [0.0, 170.0]")
    if not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be in [1, 64]")
    geometry = importlib.import_module("dinkster_inference_torch.moge_geometry")
    tensor = cast("Any", image)[..., :3]
    bchw = tensor.movedim(-1, -3).contiguous()
    chunks: list[Mapping[str, Any]] = []
    lo, hi = resource.num_tokens_range
    num_tokens = int(lo + (resolution_level / 9) * (hi - lo))
    fov = None if fov_x_degrees <= 0.0 else fov_x_degrees
    with resource.handle.stage(observer_stage="encode"):
        with torch.inference_mode():
            for index in range(0, bchw.shape[0], batch_size):
                source = bchw[index : index + batch_size].to(
                    resource.handle.load_device,
                    dtype=resource.compute_dtype,
                )
                raw = cast("Any", resource.handle.module).forward(
                    source,
                    num_tokens=num_tokens,
                )
                points = raw["points"].float()
                mask = raw["mask"] > resource.mask_threshold
                aspect_ratio = source.shape[-1] / source.shape[-2]
                diagonal = (1 + aspect_ratio**2) ** 0.5
                angle = torch.as_tensor(
                    60.0 if fov is None else fov,
                    device=points.device,
                    dtype=points.dtype,
                )
                requested_focal = aspect_ratio / diagonal / torch.tan(torch.deg2rad(angle / 2))

                if fov is None:
                    focal, shift = geometry.recover_focal_shift(points, mask)
                    bad = ~torch.isfinite(focal) | (focal <= 0)
                    if bool(bad.any()):
                        focal = torch.where(bad, requested_focal, focal)
                        _, shift = geometry.recover_focal_shift(points, mask, focal=focal)
                else:
                    focal = requested_focal.expand(points.shape[0])
                    _, shift = geometry.recover_focal_shift(points, mask, focal=focal)
                focal_diagonal = focal / 2 * diagonal
                half = torch.tensor(0.5, device=points.device, dtype=points.dtype)
                intrinsics = geometry.intrinsics_from_focal_center(
                    focal_diagonal / aspect_ratio,
                    focal_diagonal,
                    half,
                    half,
                )
                points[..., 2] = points[..., 2] + shift[..., None, None]
                if resource.version == "v2":
                    mask = mask & (points[..., 2] > 0)
                depth = points[..., 2].clone()
                if force_projection:
                    points = geometry.depth_map_to_point_map(depth, intrinsics=intrinsics)
                metric_scale = raw.get("metric_scale")
                if metric_scale is not None:
                    points = points * metric_scale[:, None, None, None]
                    depth = depth * metric_scale[:, None, None]
                normal = raw.get("normal")
                if apply_mask:
                    points = torch.where(
                        mask[..., None], points, torch.full_like(points, float("inf"))
                    )
                    depth = torch.where(mask, depth, torch.full_like(depth, float("inf")))
                    if normal is not None:
                        normal = torch.where(mask[..., None], normal, torch.zeros_like(normal))
                chunk: dict[str, Any] = {
                    "points": points,
                    "depth": depth,
                    "intrinsics": intrinsics,
                    "mask": mask,
                }
                if normal is not None:
                    chunk["normal"] = normal
                chunks.append(chunk)

    result: dict[str, object] = {"image": tensor.cpu()}
    for field in ("points", "depth", "intrinsics", "mask", "normal"):
        values = [chunk[field] for chunk in chunks if field in chunk]
        if values:
            result[field] = torch.cat(values, dim=0)
    return result


class GenerationEstimateGeometry(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.estimate_geometry")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        image: object,
        resolution_level: int = 9,
        fov_x_degrees: float = 0.0,
        batch_size: int = 4,
        force_projection: bool = True,
        apply_mask: bool = True,
    ) -> Mapping[str, object]:
        if type(model) is not _NativeGeometryModel:
            raise TypeError("model must be a native geometry model")
        geometry = _run_geometry_model(
            model,
            image,
            resolution_level,
            fov_x_degrees,
            batch_size,
            force_projection,
            apply_mask,
        )
        return cls.outputs(geometry=geometry)


class GenerationGeometryToFOV(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.geometry_to_fov")

    @classmethod
    def execute(
        cls, *, geometry: object, axis: str = "vertical", unit: str = "degrees"
    ) -> Mapping[str, object]:
        if not isinstance(geometry, Mapping):
            raise TypeError("geometry must be a mapping")
        geometry_fields = cast("Mapping[str, object]", geometry)
        intrinsics = geometry_fields.get("intrinsics")
        if intrinsics is None:
            raise ValueError("geometry has no intrinsics")
        matrix = cast("Any", intrinsics)
        if matrix.ndim == 3:
            matrix = matrix[0]
        horizontal = 0.5 / float(matrix[0, 0].item())
        vertical = 0.5 / float(matrix[1, 1].item())
        half_tangent = {
            "horizontal": horizontal,
            "vertical": vertical,
            "diagonal": math.hypot(horizontal, vertical),
        }[axis]
        radians = 2.0 * math.atan(half_tangent)
        fov = radians if unit == "radians" else math.degrees(radians)
        source = next(
            (
                geometry_fields[key]
                for key in ("image", "points", "depth")
                if key in geometry_fields
            ),
            None,
        )
        if source is None:
            raise ValueError("geometry has no image, points, or depth")
        focal_pixels = float(matrix[1, 1].item()) * int(cast("Any", source).shape[1])
        return cls.outputs(fov=fov, focal_pixels=focal_pixels)


class GenerationLoadBackgroundRemoval(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.load_background_removal")

    @classmethod
    def execute(cls, *, model: object) -> Mapping[str, object]:
        if not isinstance(model, AssetRef):
            raise TypeError("model must be an AssetRef")
        torch = _torch()
        load_device = select_load_device(torch)
        module, compute_dtype, image_size, image_mean, image_std = (
            _load_background_removal_component(
                _load_comfy_state_dict(model),
                load_device,
            )
        )
        if len(image_mean) != 3 or len(image_std) != 3:
            raise ValueError("background-removal normalization must have three channels")
        handle = _enroll_auxiliary_model(
            model,
            module,
            "background-removal",
            load_device=load_device,
        )
        value = _NativeBackgroundRemovalModel(
            handle,
            compute_dtype,
            image_size,
            image_mean,
            image_std,
        )
        return cls.outputs(model=value)


class GenerationRemoveBackground(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.remove_background")

    @classmethod
    def execute(cls, *, model: object, image: object) -> Mapping[str, object]:
        if type(model) is not _NativeBackgroundRemovalModel:
            raise TypeError("model must be a native background-removal model")
        resource = model
        torch = _torch()
        if type(image) is not torch.Tensor:
            raise TypeError("image must be an exact torch.Tensor")
        tensor = cast("Any", image)
        preprocess = importlib.import_module(
            "dinkster_inference_torch.image_preprocess"
        ).clip_preprocess
        with resource.handle.stage(observer_stage="encode"):
            with torch.inference_mode():
                pixels = preprocess(
                    tensor.to(resource.handle.load_device),
                    size=resource.image_size,
                    mean=resource.image_mean,
                    std=resource.image_std,
                    crop=False,
                ).to(dtype=resource.compute_dtype)
                component = cast("Any", resource.handle.module)
                if pixels.shape[0] > 1:
                    output = torch.cat(
                        [
                            component(pixel_values=pixels[index : index + 1])
                            for index in range(pixels.shape[0])
                        ],
                        dim=0,
                    )
                else:
                    output = component(pixel_values=pixels)
                output = torch.nn.functional.interpolate(
                    output,
                    size=(tensor.shape[1], tensor.shape[2]),
                    mode="bicubic",
                    antialias=False,
                )
                mask = output.sigmoid().to(device="cpu", dtype=torch.float32).squeeze(1)
        return cls.outputs(mask=mask)


class GenerationImageCropToMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.image_crop_to_mask")

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        masks: object,
        width: int = 1024,
        height: int = 1024,
        pad_factor: float = 1.0,
        grow_mask: int = 0,
        background: str = "#000000",
    ) -> Mapping[str, object]:
        image_crop = importlib.import_module("dinkster_inference_torch.image_crop")
        result = image_crop.crop_images_to_masks(
            images=images,
            masks=masks,
            width=width,
            height=height,
            pad_factor=pad_factor,
            grow_mask=grow_mask,
            background=background,
        )
        return cls.outputs(images=result)


class GenerationPreviewMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.preview_mask")

    @classmethod
    def execute(cls, *, mask: object) -> Mapping[str, object]:
        return cls.outputs(mask=mask)


class GenerationVoxelToMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.voxel_to_mesh")

    @classmethod
    def execute(
        cls, *, voxel: object, algorithm: str = "surface net", threshold: float = 0.6
    ) -> Mapping[str, object]:
        mesh = _native_mesh_operation(
            "voxel_grid_to_mesh", voxel=voxel, algorithm=algorithm, threshold=threshold
        )
        return cls.outputs(mesh=mesh)


class GenerationGetMeshInfo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.get_mesh_info")

    @classmethod
    def execute(cls, *, mesh: object) -> Mapping[str, object]:
        mesh_ops = importlib.import_module("dinkster_inference_torch.mesh")
        return cls.outputs(mesh=_triangle_mesh_batch(mesh), info=mesh_ops.mesh_info(mesh))


class GenerationRemeshMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.remesh_mesh")

    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        resolution: int = 512,
        sign_mode: str = "udf",
        qef: bool = False,
        drop_inverted_components: bool = False,
        drop_enclosed_components: bool = False,
        manifold: bool = False,
        band: float = 1.0,
        project_back: float = 0.0,
        fix_poles: bool = False,
        smooth_iters: int = 0,
        drop_small_components: float = 0.01,
        precluster_max_verts: int = 20_000_000,
    ) -> Mapping[str, object]:
        context = current_execution_context()
        result = _native_mesh_operation(
            "remesh_mesh",
            offload_models=True,
            cancelled=_not_cancelled if context is None else context.cancelled,
            mesh=_triangle_mesh_batch(mesh),
            resolution=resolution,
            sign_mode=sign_mode,
            qef=qef,
            drop_inverted_components=drop_inverted_components,
            drop_enclosed_components=drop_enclosed_components,
            manifold=manifold,
            band=band,
            project_back=project_back,
            fix_poles=fix_poles,
            smooth_iters=smooth_iters,
            drop_small_components=drop_small_components,
            precluster_max_verts=precluster_max_verts,
        )
        return cls.outputs(mesh=result)


class GenerationDecimateMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.decimate_mesh")

    @classmethod
    def execute(
        cls, *, mesh: object, target_face_count: int = 200_000, placement_mode: str = "midpoint"
    ) -> Mapping[str, object]:
        result = _native_mesh_operation(
            "decimate_mesh",
            mesh=_triangle_mesh_batch(mesh),
            target_face_count=target_face_count,
            placement_mode=placement_mode,
        )
        return cls.outputs(mesh=result)


class GenerationSmoothMeshNormals(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.smooth_mesh_normals")

    @classmethod
    def execute(cls, *, mesh: object, crease_angle: float = 180.0) -> Mapping[str, object]:
        result = _native_mesh_operation(
            "smooth_mesh_normals",
            mesh=_triangle_mesh_batch(mesh),
            crease_angle=crease_angle,
        )
        return cls.outputs(mesh=result)


class GenerationUnwrapMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.unwrap_mesh")

    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        segmenter: str = "pec",
        resolution: int = 1024,
        padding: int = 1,
        weld_distance: float = 0.0,
    ) -> Mapping[str, object]:
        result = _native_mesh_operation(
            "unwrap_mesh",
            offload_models=True,
            mesh=_triangle_mesh_batch(mesh),
            segmenter=segmenter,
            resolution=resolution,
            padding=padding,
            weld_distance=weld_distance,
        )
        return cls.outputs(mesh=result)


class GenerationPaintMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.paint_mesh")

    @classmethod
    def execute(cls, *, mesh: object, voxel_colors: object) -> Mapping[str, object]:
        result = _native_mesh_operation(
            "paint_mesh", mesh=_triangle_mesh_batch(mesh), voxel_colors=voxel_colors
        )
        return cls.outputs(mesh=result)


class GenerationBakeTextureFromVoxel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.bake_texture_from_voxel")

    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        voxel_colors: object,
        texture_size: int = 2048,
        reference_mesh: object | None = None,
    ) -> Mapping[str, object]:
        base_color, metallic, roughness = cast(
            "tuple[object, object, object]",
            _native_mesh_operation(
                "bake_texture_from_voxel",
                offload_models=True,
                mesh=_triangle_mesh_batch(mesh),
                voxel_colors=voxel_colors,
                texture_size=texture_size,
                reference_mesh=(
                    None if reference_mesh is None else _triangle_mesh_batch(reference_mesh)
                ),
            ),
        )
        return cls.outputs(base_color=base_color, metallic=metallic, roughness=roughness)


class GenerationBakeNormalMapFromMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.bake_normal_map_from_mesh")

    @classmethod
    def execute(
        cls,
        *,
        low_poly: object,
        high_poly: object,
        resolution: int = 1024,
        cage_distance: float = 0.05,
        ignore_backfaces: bool = True,
    ) -> Mapping[str, object]:
        normal_map = _native_mesh_operation(
            "bake_normal_map_from_mesh",
            low_poly=_triangle_mesh_batch(low_poly),
            high_poly=_triangle_mesh_batch(high_poly),
            resolution=resolution,
            cage_distance=cage_distance,
            ignore_backfaces=ignore_backfaces,
        )
        return cls.outputs(normal_map=normal_map)


class GenerationBakeAmbientOcclusion(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.bake_ambient_occlusion")

    @classmethod
    def execute(
        cls,
        *,
        low_poly: object,
        high_poly: object,
        resolution: int = 1024,
        samples: int = 64,
        max_distance: float = 0.5,
        strength: float = 1.0,
        bias: float = 0.01,
    ) -> Mapping[str, object]:
        occlusion = _native_mesh_operation(
            "bake_ambient_occlusion",
            low_poly=_triangle_mesh_batch(low_poly),
            high_poly=_triangle_mesh_batch(high_poly),
            resolution=resolution,
            samples=samples,
            max_distance=max_distance,
            strength=strength,
            bias=bias,
        )
        return cls.outputs(occlusion=occlusion)


class GenerationRenderUVAtlas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.render_uv_atlas")

    @classmethod
    def execute(cls, *, mesh: object, resolution: int = 1024) -> Mapping[str, object]:
        image = _native_mesh_operation(
            "render_uv_atlas", mesh=_triangle_mesh_batch(mesh), resolution=resolution
        )
        return cls.outputs(image=image)


class GenerationApplyTextureToMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.apply_texture_to_mesh")

    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        base_color: object,
        metallic: object | None = None,
        roughness: object | None = None,
        occlusion: object | None = None,
        normal_map: object | None = None,
    ) -> Mapping[str, object]:
        result = _native_mesh_operation(
            "apply_texture_to_mesh",
            mesh=_triangle_mesh_batch(mesh),
            base_color=base_color,
            metallic=metallic,
            roughness=roughness,
            occlusion=occlusion,
            normal_map=normal_map,
        )
        return cls.outputs(mesh=result)


class GenerationMeshToModel3D(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.mesh_to_model3d")

    @classmethod
    def execute(cls, *, mesh: object) -> Mapping[str, object]:
        mesh_ops = importlib.import_module("dinkster_inference_torch.mesh")
        glb = mesh_ops.mesh_item_to_glb_bytes(mesh, 0)
        if glb is None:
            raise ValueError("mesh is empty")
        return cls.outputs(model={"format": "glb", "bytes": glb})


class GenerationLoadCheckpoint(NativeLoadCheckpoint):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_checkpoint")

    @classmethod
    def execute(cls, *, checkpoint: object) -> Mapping[str, object]:
        loaded = NativeLoadCheckpoint.execute(checkpoint=checkpoint)
        handle = _require_provider_runtime(loaded["model"], "model")
        inference = importlib.import_module("dinkster_inference")
        codec = _NativeCodecHandle(handle)
        inference.require_inference_codec_handle(codec, "vae")
        return cls.outputs(
            model=handle,
            clip=loaded["clip"],
            vae=codec,
        )


class GenerationLoadControlNet(NativeControlNetLoader):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_controlnet")


def _apply_control_carrier(
    conditioning: object,
    *,
    resource: _NativeControlNetResource,
    hint: _ControlHintSnapshot,
    **kwargs: Any,
) -> Any:
    inference = importlib.import_module("dinkster_inference")
    previous = _controlled_conditioning(conditioning)
    carrier = _conditioning_carrier(
        conditioning if previous is None else previous.conditioning, "conditioning"
    )
    binding = _extend_control_binding(
        None if previous is None else previous.binding,
        resource=resource,
        hint=hint,
        inference=inference,
        **kwargs,
    )
    resources = (*(() if previous is None else previous.resources), resource)
    return inference.ResidentConditioningCarrier(
        _ControlledConditioning(carrier, binding, resources)
    )


class GenerationApplyControlNet(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.apply_controlnet")

    @classmethod
    def execute(
        cls, *, conditioning: object, control_net: object, image: object, strength: float = 1.0
    ) -> Mapping[str, object]:
        if strength == 0.0:
            return cls.outputs(conditioning=conditioning)
        resource = _native_controlnet(control_net)
        hint = _snapshot_control_hint(
            image,
            _torch(),
            importlib.import_module("dinkster_inference_torch"),
            resource.hint_channels,
        )
        return cls.outputs(
            conditioning=_apply_control_carrier(
                conditioning,
                resource=resource,
                hint=hint,
                strength=float(strength),
                start_percent=0.0,
                end_percent=1.0,
                apply_to_uncond=True,
            )
        )


class GenerationApplyControlNetAdvanced(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.apply_controlnet_advanced")

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        control_net: object,
        image: object,
        strength: float = 1.0,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        vae: object = None,
    ) -> Mapping[str, object]:
        # Pixel-space controls do not encode their hints with the supplied VAE.
        del vae
        if strength == 0.0:
            return cls.outputs(positive=positive, negative=negative)
        resource = _native_controlnet(control_net)
        hint = _snapshot_control_hint(
            image,
            _torch(),
            importlib.import_module("dinkster_inference_torch"),
            resource.hint_channels,
        )
        return cls.outputs(
            **{
                name: _apply_control_carrier(
                    value,
                    resource=resource,
                    hint=hint,
                    strength=float(strength),
                    start_percent=float(start_percent),
                    end_percent=float(end_percent),
                    apply_to_uncond=False,
                )
                for name, value in (("positive", positive), ("negative", negative))
            }
        )


class GenerationSetControlNetUnionType(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.set_controlnet_union_type")

    @classmethod
    def execute(cls, *, control_net: object, type: str = "auto") -> Mapping[str, object]:
        resource = _native_controlnet(control_net)
        inference = importlib.import_module("dinkster_inference")
        mode = (
            None
            if type == "auto"
            else inference.SDControlMode(provider="sdxl-controlnet-union", token=type.split("/")[0])
        )
        return cls.outputs(control_net=replace(resource, mode=mode))


class GenerationLoadDiffusionModel(NativeLoadDiffusionModel):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_diffusion_model")


def _diffusion_component_assets(components: Mapping[str, object]) -> Mapping[str, AssetRef]:
    members: dict[str, dict[str, object]] = {}
    for input_id, value in components.items():
        member, separator, field = input_id.partition(".")
        if not separator or field not in ("component", "role"):
            raise ValueError(f"invalid diffusion component input {input_id!r}")
        members.setdefault(member, {})[field] = value
    assets: dict[str, AssetRef] = {}
    for member, values in members.items():
        if set(values) != {"component", "role"}:
            raise ValueError(f"diffusion component member {member!r} is incomplete")
        asset = values["component"]
        role = values["role"]
        if not isinstance(asset, AssetRef):
            raise TypeError(f"diffusion component {member!r} must be an AssetRef")
        if not isinstance(role, str) or not role:
            raise TypeError(f"diffusion component {member!r} role must be a non-empty string")
        if role in assets:
            raise ValueError(f"diffusion component role {role!r} is duplicated")
        assets[role] = asset
    return assets


class GenerationLoadDiffusionComponents(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_diffusion_components")

    @classmethod
    def execute(
        cls,
        *,
        components: Mapping[str, object],
        weight_dtype: str = "default",
    ) -> Mapping[str, object]:
        torch = _torch()
        storage_dtype = _weight_storage_dtype(torch, weight_dtype)
        context = current_execution_context()
        if context is None or context.expected_execution_identity is None:
            raise RuntimeError(
                "native load_diffusion_components ran without an expected execution identity"
            )
        with native_execution_span("load", "load"):
            assets = _diffusion_component_assets(components)
            h3_roles = set(assets) & {"fl2va-dit", "ref2va-dit"}
            if len(assets) == 1 and len(h3_roles) == 1:
                artifact_role = next(iter(h3_roles))
                storage_kwargs: dict[str, Any] = (
                    {} if storage_dtype is None else {"storage_dtype": storage_dtype}
                )
                handle = _build_component_runtime_handle(
                    _component_descriptor("dinkster.minimax_h3"),
                    assets[artifact_role],
                    "diffusion",
                    context.expected_execution_identity,
                    torch,
                    compute_dtype=context.diffusion_dtype or "bfloat16",
                    as_model=True,
                    artifact_role=artifact_role,
                    **storage_kwargs,
                )
            else:
                handle = _build_trellis2_split_model_handle(
                    assets,
                    context.expected_execution_identity,
                    torch,
                    compute_dtype=context.diffusion_dtype or "bfloat16",
                    storage_dtype=storage_dtype,
                )
        return cls.outputs(model=handle)


class GenerationLoadLTXAVTextEncoder(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_ltxav_text_encoder")

    @classmethod
    def execute(
        cls,
        *,
        text_encoder: object,
        ckpt_name: object,
        device: str = "default",
    ) -> Mapping[str, object]:
        if not isinstance(text_encoder, AssetRef):
            raise TypeError("text_encoder must be an AssetRef")
        if not isinstance(ckpt_name, AssetRef):
            raise TypeError("ckpt_name must be an AssetRef")
        if device not in ("default", "cpu"):
            raise ValueError("device must be 'default' or 'cpu'")
        context = _component_execution_context("load_ltxav_text_encoder")
        torch = _torch()
        load_device = torch.device("cpu") if device == "cpu" else None
        with native_execution_span("load", "load"):
            handle = _build_ltxav_text_handle(
                text_encoder,
                ckpt_name,
                context.expected_execution_identity,
                torch,
                compute_dtype=context.text_dtype or "bfloat16",
                load_device=load_device,
            )
        return cls.outputs(clip=handle)


class GenerationLoadLTXAVAudioVAE(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_ltxav_audio_vae")

    @classmethod
    def execute(cls, *, ckpt_name: object) -> Mapping[str, object]:
        if not isinstance(ckpt_name, AssetRef):
            raise TypeError("ckpt_name must be an AssetRef")
        context = _component_execution_context("load_ltxav_audio_vae")
        if context.vae_dtype not in (None, "float32"):
            raise RuntimeError("LTX-2 audio codec requires float32 execution")
        with native_execution_span("load", "load"):
            handle = _build_ltxav_audio_codec_handle(
                ckpt_name,
                context.expected_execution_identity,
                _torch(),
            )
        return cls.outputs(audio_vae=handle)


class GenerationLoadLatentUpscaleModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_latent_upscale_model")

    @classmethod
    def execute(cls, *, model_name: object) -> Mapping[str, object]:
        if not isinstance(model_name, AssetRef):
            raise TypeError("model_name must be an AssetRef")
        context = _component_execution_context("load_latent_upscale_model")
        with native_execution_span("load", "load"):
            handle = _build_component_runtime_handle(
                _component_descriptor("dinkster.ltxav"),
                model_name,
                "latent_upscaler",
                context.expected_execution_identity,
                _torch(),
                compute_dtype=context.vae_dtype or "bfloat16",
            )
        return cls.outputs(upscale_model=handle)


class GenerationLTXAVAudioVAEDecode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxav_audio_vae_decode")

    @classmethod
    def execute(cls, *, samples: object, audio_vae: object) -> Mapping[str, object]:
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a latent mapping")
        inference = importlib.import_module("dinkster_inference")
        torch = _torch()
        latent = cast("Mapping[object, object]", samples).get("samples")
        if type(latent) is inference.MultiStreamLatent:
            streams = cast("Any", latent)
            if streams.roles != ("audio",):
                raise TypeError("samples['samples'] must contain exactly one audio stream")
            latent = streams.by_role("audio")
        if type(latent) is not torch.Tensor:
            raise TypeError("samples['samples'] must be an exact torch.Tensor")
        tensor = cast("Any", latent)
        if (
            tensor.layout is not torch.strided
            or not tensor.is_floating_point()
            or tensor.ndim != 4
            or tensor.shape[0] <= 0
            or tensor.shape[1] != 8
            or tensor.shape[2] <= 0
            or tensor.shape[3] != 16
        ):
            raise ValueError("samples['samples'] must be nonempty floating [batch,8,time,16]")
        component, load_device, stage = _ltxav_audio_codec(audio_vae)
        inference_torch = importlib.import_module("dinkster_inference_torch")
        runtime = inference_torch.LTXAVAudioCodecRuntime(component)
        with stage():
            with torch.inference_mode():
                audio = runtime.decode_audio_latent(tensor.to(load_device))
        if type(audio) is not inference.AudioPreview:
            raise TypeError("LTX-2 audio decode must return AudioPreview")
        waveform = audio.waveform
        if (
            type(waveform) is not torch.Tensor
            or waveform.layout is not torch.strided
            or not waveform.is_floating_point()
            or waveform.ndim != 3
            or waveform.shape[0] != tensor.shape[0]
            or waveform.shape[1] not in (1, 2)
            or waveform.shape[2] <= 0
        ):
            raise TypeError("LTX-2 audio decode must return nonempty floating [batch,1|2,samples]")
        return cls.outputs(audio={"waveform": waveform, "sample_rate": audio.sample_rate})


class GenerationLoadLora(NativeLoadLora):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_lora")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        clip: object,
        lora: object,
        strength_model: float,
        strength_clip: float,
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        _require_provider_runtime(model, "model")
        return NativeLoadLora.execute(
            model=model,
            clip=clip,
            lora=lora,
            strength_model=strength_model,
            strength_clip=strength_clip,
            execution_mode=_generation_lora_mode(execution_mode),
        )


class GenerationLoadLoraModelOnly(NativeLoadLoraModelOnly):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_lora_model_only")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        lora: object,
        strength_model: float,
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        _require_provider_runtime(model, "model")
        return NativeLoadLoraModelOnly.execute(
            model=model,
            lora=lora,
            strength_model=strength_model,
            execution_mode=_generation_lora_mode(execution_mode),
        )


def _ordered_lora_members(
    loras: Mapping[str, object], fields: frozenset[str]
) -> tuple[Mapping[str, object], ...]:
    members: dict[str, dict[str, object]] = {}
    for input_id, value in loras.items():
        member, separator, field = input_id.partition(".")
        if not separator or field not in fields:
            raise ValueError(f"invalid LoRA stack input {input_id!r}")
        members.setdefault(member, {})[field] = value
    for member, values in members.items():
        if "lora" not in values:
            raise ValueError(f"LoRA stack member {member!r} is missing its lora asset")
    return tuple(members.values())


def _model_clip_lora_entries(
    loras: Mapping[str, object],
) -> tuple[tuple[object, float, float], ...]:
    members = _ordered_lora_members(loras, frozenset(("lora", "strength_model", "strength_clip")))
    return tuple(
        (
            member["lora"],
            cast("float", member.get("strength_model", 1.0)),
            cast("float", member.get("strength_clip", 1.0)),
        )
        for member in members
    )


def _model_lora_entries(loras: Mapping[str, object]) -> tuple[tuple[object, float], ...]:
    members = _ordered_lora_members(loras, frozenset(("lora", "strength_model")))
    return tuple(
        (member["lora"], cast("float", member.get("strength_model", 1.0))) for member in members
    )


class GenerationLoadCheckpointStack(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_checkpoint_stack")

    @classmethod
    def execute(
        cls,
        *,
        checkpoint: object,
        loras: Mapping[str, object],
        stop_at_clip_layer: int = -1,
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        loaded = GenerationLoadCheckpoint.execute(checkpoint=checkpoint)
        model, clip = _apply_native_lora_stack(
            loaded["model"],
            loaded["clip"],
            _model_clip_lora_entries(loras),
            _generation_lora_mode(execution_mode),
        )
        vae = _NativeCodecHandle(_native_handle(clip, "clip"))
        if stop_at_clip_layer != -1:
            clip = GenerationClipSetLastLayer.execute(
                clip=clip,
                stop_at_clip_layer=stop_at_clip_layer,
            )["clip"]
        return cls.outputs(
            model=model,
            clip=clip,
            vae=vae,
        )


class GenerationApplyLoraStack(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.apply_lora_stack")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        clip: object,
        loras: Mapping[str, object],
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        _require_provider_runtime(model, "model")
        model, clip = _apply_native_lora_stack(
            model,
            clip,
            _model_clip_lora_entries(loras),
            _generation_lora_mode(execution_mode),
        )
        return cls.outputs(model=model, clip=clip)


class GenerationApplyLoraStackModelOnly(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.apply_lora_stack_model_only")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        loras: Mapping[str, object],
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        _require_provider_runtime(model, "model")
        return cls.outputs(
            model=_apply_native_model_lora_stack(
                model,
                _model_lora_entries(loras),
                _generation_lora_mode(execution_mode),
            )
        )


def _generation_input_int(
    inputs: Mapping[str, object],
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    value = inputs.get(name)
    if type(value) is not int:
        raise TypeError(f"{name.rsplit('.', 1)[-1]} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name.rsplit('.', 1)[-1]} must be in [{minimum}, {maximum}], got {value}"
        )
    return value


def _generation_input_float(
    inputs: Mapping[str, object],
    name: str,
    minimum: float,
    maximum: float,
    *,
    inclusive_minimum: bool = True,
) -> float:
    value = inputs.get(name)
    if type(value) is not float:
        raise TypeError(f"{name.rsplit('.', 1)[-1]} must be a float")
    lower_valid = value >= minimum if inclusive_minimum else value > minimum
    if not math.isfinite(value) or not lower_valid or value > maximum:
        opening = "[" if inclusive_minimum else "("
        raise ValueError(
            f"{name.rsplit('.', 1)[-1]} must be in {opening}{minimum}, {maximum}], got {value}"
        )
    return value


def _generation_sampler(inputs: Mapping[str, object], inference: Any) -> tuple[Any, int | None]:
    mode = inputs.get("sampling_mode")
    if mode == "off":
        return (
            inference.GenerationSamplerChain(
                (
                    inference.GenerationSamplerStage(
                        inference.GenerationSamplerKind.GREEDY,
                    ),
                )
            ),
            None,
        )
    if mode != "on":
        raise ValueError(f"unknown text generation sampling mode: {mode!r}")

    temperature = _generation_input_float(
        inputs, "sampling_mode.temperature", 0.0, 2.0, inclusive_minimum=False
    )
    top_k = _generation_input_int(inputs, "sampling_mode.top_k", 0, 1_000)
    top_p = _generation_input_float(inputs, "sampling_mode.top_p", 0.0, 1.0)
    min_p = _generation_input_float(inputs, "sampling_mode.min_p", 0.0, 1.0)
    repetition = _generation_input_float(
        inputs, "sampling_mode.repetition_penalty", 0.0, 5.0, inclusive_minimum=False
    )
    presence = _generation_input_float(inputs, "sampling_mode.presence_penalty", 0.0, 5.0)
    seed = _generation_input_int(inputs, "sampling_mode.seed", 0, 2**64 - 1)

    stages: list[Any] = []
    if repetition != 1.0:
        stages.append(
            inference.GenerationSamplerStage(
                inference.GenerationSamplerKind.REPETITION_PENALTY,
                repetition,
            )
        )
    if presence != 0.0:
        stages.append(
            inference.GenerationSamplerStage(
                inference.GenerationSamplerKind.PRESENCE_PENALTY,
                presence,
            )
        )
    if temperature != 1.0:
        stages.append(
            inference.GenerationSamplerStage(
                inference.GenerationSamplerKind.TEMPERATURE,
                temperature,
            )
        )
    if top_k:
        stages.append(
            inference.GenerationSamplerStage(
                inference.GenerationSamplerKind.TOP_K,
                top_k,
            )
        )
    if min_p:
        stages.append(
            inference.GenerationSamplerStage(
                inference.GenerationSamplerKind.MIN_P,
                min_p,
            )
        )
    if top_p != 1.0:
        stages.append(
            inference.GenerationSamplerStage(
                inference.GenerationSamplerKind.TOP_P,
                top_p,
            )
        )
    stages.append(
        inference.GenerationSamplerStage(
            inference.GenerationSamplerKind.MULTINOMIAL,
        )
    )
    return inference.GenerationSamplerChain(tuple(stages)), seed


def _validate_text_generation_options(inputs: Mapping[str, object]) -> None:
    for name in ("image", "video", "audio"):
        if inputs.get(name) is not None:
            raise ValueError(f"native Qwen generation does not support {name} input")
    thinking = inputs.get("thinking", False)
    if type(thinking) is not bool:
        raise TypeError("thinking must be a boolean")
    if thinking:
        raise ValueError("native Qwen generation does not support thinking mode")
    use_default_template = inputs.get("use_default_template", False)
    if type(use_default_template) is not bool:
        raise TypeError("use_default_template must be a boolean")
    if use_default_template:
        raise ValueError(
            "native Qwen generation does not support model templates; "
            "set use_default_template to false"
        )


def _run_qwen_text_generation(inputs: Mapping[str, object], prompt: str) -> str:
    if type(prompt) is not str:
        raise TypeError("prompt must be a string")
    _validate_text_generation_options(inputs)
    max_length = _generation_input_int(inputs, "max_length", 1, 32_768)
    inference = importlib.import_module("dinkster_inference")
    sampler, seed = _generation_sampler(inputs, inference)
    handle = load_registered_component(
        inputs.get("clip"),
        "clip",
        "qwen3_06b",
        family_id=inference.ANIMA_CONFIG.family_id,
    )
    context = current_execution_context()
    cancelled = _not_cancelled if context is None else context.cancelled

    with handle.stage():
        inference_torch = importlib.import_module("dinkster_inference_torch")
        provider = inference_torch.QwenGenerationProvider(
            handle.component,
            inference.load_qwen_bpe(),
            handle.resource_identity,
        )
        request = inference.GenerationRequest(
            provider.id,
            handle.resource_identity,
            prompt=prompt,
            sampler=sampler,
            stop=inference.GenerationStopConditions(max_length),
            seed=seed,
        )
        terminal = None
        with provider.generate(request, cancelled=cancelled) as stream:
            for event in stream:
                if isinstance(event, inference.GenerationTerminalEvent):
                    if terminal is not None:
                        raise RuntimeError("generation stream produced multiple terminal events")
                    terminal = event
                elif terminal is not None:
                    raise RuntimeError(
                        "generation stream produced an event after its terminal event"
                    )
                elif not isinstance(event, inference.GenerationTokenEvent):
                    raise RuntimeError("generation stream produced an unknown event")
    if terminal is None:
        raise RuntimeError("generation stream ended without a terminal event")
    if terminal.result.finish_reason is inference.GenerationFinishReason.CANCELLED:
        raise inference.SamplingCancelled("text generation cancelled")
    return terminal.result.text


class GenerationTextGenerate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.text_generate")

    @classmethod
    def check_lazy_status(
        cls,
        *,
        clip: object | None = None,
        provider: object | None = None,
        **_inputs: object,
    ) -> tuple[str, ...]:
        return ("clip",) if provider is None and clip is None else ()

    @classmethod
    def execute(cls, **inputs: object) -> Mapping[str, object]:
        prompt = inputs.get("prompt")
        if type(prompt) is not str:
            raise TypeError("prompt must be a string")
        return cls.outputs(generated_text=_run_qwen_text_generation(inputs, prompt))


class GenerationPromptEnhance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.prompt_enhance")

    @classmethod
    def check_lazy_status(
        cls,
        *,
        clip: object | None = None,
        provider: object | None = None,
        **_inputs: object,
    ) -> tuple[str, ...]:
        return ("clip",) if provider is None and clip is None else ()

    @classmethod
    def execute(cls, **inputs: object) -> Mapping[str, object]:
        prompt = inputs.get("prompt")
        if type(prompt) is not str:
            raise TypeError("prompt must be a string")
        text = _run_qwen_text_generation(inputs, prepare_ltx2_prompt(prompt))
        return cls.outputs(generated_text=clean_enhanced_prompt(text, prompt))


class NativeMiniMaxMusic3TextEncode(MiniMaxMusic3TextEncode):
    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        caption: str,
        lyrics: str,
        seed: int,
        max_duration: float,
        cfg_scale: float,
        top_k: int,
    ) -> Mapping[str, object]:
        if type(caption) is not str or type(lyrics) is not str:
            raise TypeError("caption and lyrics must be strings")
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("seed must be an integer in [0, 18446744073709551615]")
        if type(max_duration) not in (int, float) or not 0.04 <= max_duration <= 360.0:
            raise ValueError("max_duration must be in [0.04, 360.0]")
        if type(cfg_scale) not in (int, float) or not 0.0 <= cfg_scale <= 100.0:
            raise ValueError("cfg_scale must be in [0.0, 100.0]")
        if type(top_k) is not int or not 1 <= top_k <= 16384:
            raise ValueError("top_k must be an integer in [1, 16384]")
        inference = importlib.import_module("dinkster_inference")
        handle = load_registered_component(
            clip,
            "clip",
            "text",
            family_id=inference.MINIMAX_MUSIC3_CONFIG.family_id,
        )
        tokenizer = getattr(handle.component, "_dinkster_minimax_music3_tokenizer", None)
        inference_torch = importlib.import_module("dinkster_inference_torch")
        max_audio_frames = min(
            inference.MAX_AUDIO_FRAMES,
            max(1, round(max_duration * inference.AUDIO_FRAMES_PER_SECOND)),
        )
        recipe = handle.recipe
        assert recipe is not None
        runtime = inference_torch.MiniMaxMusic3TextRuntime(
            handle.component,
            tokenizer,
            compute_dtype=_torch_dtype(_torch(), recipe.knobs.text_dtype),
        )
        torch = _torch()
        with handle.stage():
            with torch.inference_mode():
                conditioning = runtime.encode_text(
                    caption,
                    lyrics,
                    seed=seed,
                    max_audio_frames=max_audio_frames,
                    cfg_scale=cfg_scale,
                    top_k=top_k,
                )
        carrier = inference_torch.minimax_music3_conditioning_to_carrier(conditioning)
        return cls.outputs(
            conditioning=inference.bind_component_conditioning(
                carrier,
                inference.ComponentBinding(
                    "text",
                    inference.MINIMAX_MUSIC3_CONFIG.family_id,
                    handle.resource_identity,
                ),
            ),
            seconds=conditioning.embeddings.shape[1] / inference.AUDIO_FRAMES_PER_SECOND,
        )


class GenerationClipTextEncode(NativeClipTextEncode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.clip_text_encode")

    @classmethod
    def execute(cls, *, text: str, clip: object) -> Mapping[str, object]:
        options = _native_clip_options(clip)
        clip = options.source
        if type(clip) is _LTXAVTextHandle:
            direct_clip = cast("Any", clip)
            inference = importlib.import_module("dinkster_inference")
            torch = _torch()
            with direct_clip.stage():
                with torch.inference_mode():
                    carrier = direct_clip.encode_text(text)
            return cls.outputs(
                conditioning=inference.bind_component_conditioning(
                    carrier,
                    inference.ComponentBinding(
                        direct_clip.role,
                        direct_clip.family_id,
                        direct_clip.resource_identity,
                    ),
                )
            )
        if isinstance(clip, NativeComponentHandle):
            inference = importlib.import_module("dinkster_inference")
            recipe = clip.recipe
            text_runtime = getattr(clip, "runtime", None)
            encode = getattr(text_runtime, "encode_text", None)
            to_carrier = getattr(text_runtime, "text_conditioning_carrier", None)
            if callable(encode) and callable(to_carrier):
                if recipe is None:
                    raise RuntimeError("native text encoding runtime has no retained recipe")
                torch = _torch()
                with clip.stage(observer_stage="condition"), torch.inference_mode():
                    conditioning = encode(
                        text,
                        hidden_layer=options.hidden_layer,
                        min_padding=options.t5_min_padding,
                        min_length=options.t5_min_length,
                    )
                    carrier = to_carrier(conditioning)
                return cls.outputs(
                    conditioning=inference.bind_component_conditioning(
                        carrier,
                        inference.ComponentBinding(
                            "text", recipe.family_id, clip.resource_identity
                        ),
                    )
                )
            registered = importlib.import_module("dinkster_native.family_registry")
            encode_text = registered.registered_callable(clip, "native_encode_text")
            return cls.outputs(conditioning=encode_text(clip, text, options))
        handle = _require_provider_runtime(clip, "clip")
        runtime = handle.runtime
        encode_text = getattr(runtime, "encode_text", None)
        if not callable(encode_text):
            raise TypeError("clip runtime does not expose text encoding")
        torch = _torch()
        with handle.stage("text"):
            with torch.inference_mode():
                if (
                    options.hidden_layer is None
                    and options.t5_min_padding is None
                    and options.t5_min_length is None
                ):
                    conditioning = encode_text(text)
                else:
                    inference_torch = importlib.import_module("dinkster_inference_torch")
                    if isinstance(runtime, inference_torch.FluxRuntime):
                        conditioning = encode_text(
                            text,
                            hidden_layer=options.hidden_layer,
                            min_padding=options.t5_min_padding,
                            min_length=options.t5_min_length,
                        )
                    elif isinstance(runtime, inference_torch.SDRuntime):
                        conditioning = encode_text(text, hidden_layer=options.hidden_layer)
                    elif isinstance(runtime, inference_torch.Wan21Runtime):
                        conditioning = encode_text(
                            text,
                            min_padding=options.t5_min_padding,
                            min_length=options.t5_min_length,
                        )
                    else:
                        conditioning = encode_text(text)
        to_carrier = getattr(runtime, "text_conditioning_carrier", None)
        if callable(to_carrier):
            return cls.outputs(conditioning=to_carrier(conditioning))
        inference_torch = importlib.import_module("dinkster_inference_torch")
        return cls.outputs(conditioning=inference_torch.basic_conditioning_to_carrier(conditioning))


class GenerationClipTextEncodeLumina2(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.clip_text_encode_lumina2")

    @classmethod
    def execute(cls, *, system_prompt: str, user_prompt: str, clip: object) -> Mapping[str, object]:
        text = importlib.import_module("dinkster_inference").lumina2_system_prompt(
            user_prompt, system_prompt
        )
        return GenerationClipTextEncode.execute(text=text, clip=clip)


class GenerationModelSamplingAuraFlow(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.model_sampling_aura_flow")

    @classmethod
    def execute(cls, *, model: object, shift: float) -> Mapping[str, object]:
        if type(shift) is not float or not math.isfinite(shift) or shift <= 0.0:
            raise ValueError("shift must be a positive finite float")
        (
            handle,
            overlays,
            resolvers,
            z_image_control,
            _,
            guidance_transforms,
            context_windows,
            chroma_radiance_options,
        ) = _native_model(model, "model")
        inference = importlib.import_module("dinkster_inference")
        space = inference.FlowSigmas(shift=shift, multiplier=1.0, timesteps=1000)
        _sampling_space_runtime(handle.runtime, space)
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                z_image_control,
                None,
                guidance_transforms,
                context_windows,
                chroma_radiance_options,
                sampling_cache=_native_model_sampling_cache(model),
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=space,
            )
        )


class GenerationClipSetLastLayer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.clip_set_last_layer")

    @classmethod
    def execute(cls, *, clip: object, stop_at_clip_layer: int) -> Mapping[str, object]:
        if type(stop_at_clip_layer) is not int or not -24 <= stop_at_clip_layer <= -1:
            raise ValueError(
                f"stop_at_clip_layer must be an integer in [-24, -1], got {stop_at_clip_layer}"
            )
        return cls.outputs(
            clip=replace(_native_clip_options(clip), hidden_layer=stop_at_clip_layer)
        )


class GenerationT5TokenizerOptions(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.t5_tokenizer_options")

    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        min_padding: int,
        min_length: int,
    ) -> Mapping[str, object]:
        for name, value in (("min_padding", min_padding), ("min_length", min_length)):
            if type(value) is not int or not 0 <= value <= 10000:
                raise ValueError(f"{name} must be an integer in [0, 10000], got {value}")
        return cls.outputs(
            clip=replace(
                _native_clip_options(clip),
                t5_min_padding=min_padding,
                t5_min_length=min_length,
            )
        )


class GenerationClipTextEncodeControlnet(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.clip_text_encode_controlnet")

    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        conditioning: object,
        text: str,
    ) -> Mapping[str, object]:
        inference = importlib.import_module("dinkster_inference")
        target = _conditioning_carrier(conditioning, "conditioning")
        control = cast(
            "Any",
            GenerationClipTextEncode.execute(clip=clip, text=text)["conditioning"],
        )
        if not control.conditioning.records:
            raise ValueError("encoded ControlNet text must contain at least one record")
        channels = dict(control.conditioning.records[0].channels)
        text_payload = channels.get(inference.ConditioningChannel.TEXT)
        if text_payload is None:
            raise ValueError("encoded ControlNet text must carry a text payload")
        pooled_payload = channels.get(inference.ConditioningChannel.POOLED)
        records: list[Any] = []
        for record in target.conditioning.records:
            metadata = dict(record.extension_metadata)
            metadata[_CONTROLNET_TEXT_METADATA_KEY] = text_payload.reference
            metadata[_CONTROLNET_POOLED_METADATA_KEY] = (
                None if pooled_payload is None else pooled_payload.reference
            )
            records.append(replace(record, extension_metadata=tuple(metadata.items())))
        return cls.outputs(
            conditioning=_rebound_conditioning_carrier(
                inference,
                records,
                (*target.bindings, *control.bindings),
            )
        )


class GenerationFluxGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.flux_guidance")

    @classmethod
    def execute(cls, *, conditioning: object, guidance: float) -> Mapping[str, object]:
        if not 0.0 <= guidance <= 100.0:
            raise ValueError(f"guidance must be in [0.0, 100.0], got {guidance}")
        inference = importlib.import_module("dinkster_inference")
        if type(conditioning) is not inference.ConditioningCarrier:
            raise TypeError("conditioning must come from a Dinkster text-encoding node")
        return cls.outputs(conditioning=_with_flux_guidance(conditioning, float(guidance)))


class GenerationFluxDisableGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.flux_disable_guidance")

    @classmethod
    def execute(cls, *, conditioning: object) -> Mapping[str, object]:
        inference: Any = importlib.import_module("dinkster_inference")
        if type(conditioning) is not inference.ConditioningCarrier:
            raise TypeError("conditioning must come from a Dinkster text-encoding node")
        return cls.outputs(conditioning=_with_flux_guidance(conditioning, None))


class GenerationReferenceLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.reference_latent")

    @classmethod
    def execute(cls, *, conditioning: object, latent: object = None) -> Mapping[str, object]:
        carrier = _conditioning_carrier(conditioning, "conditioning")
        if latent is None:
            return cls.outputs(conditioning=carrier)
        if not isinstance(latent, Mapping):
            raise TypeError("latent must be a mapping containing 'samples'")
        torch = _torch()
        samples = cast("Mapping[str, object]", latent).get("samples")
        if type(samples) is not torch.Tensor:
            raise TypeError("latent['samples'] must be an exact torch.Tensor")
        inference: Any = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        binding = inference_torch.tensor_to_payload_binding(
            "flux2-reference-latent",
            samples,
            space="flux2-reference-latent",
        )
        reference = inference.PayloadReference(binding.reference_id)
        records: list[Any] = []
        for record in carrier.conditioning.records:
            metadata: dict[str, Any] = dict(record.extension_metadata)
            existing = metadata.get(_FLUX2_REFERENCE_LATENTS_KEY, ())
            if not isinstance(existing, tuple):
                raise TypeError("reference latent metadata must be an ordered payload tuple")
            references = cast("tuple[Any, ...]", existing)
            if any(type(item) is not inference.PayloadReference for item in references):
                raise TypeError("reference latent metadata must be an ordered payload tuple")
            metadata[_FLUX2_REFERENCE_LATENTS_KEY] = (*references, reference)
            records.append(replace(record, extension_metadata=tuple(metadata.items())))
        return cls.outputs(
            conditioning=_rebound_conditioning_carrier(
                inference,
                records,
                (*carrier.bindings, binding),
            )
        )


def _model_with_guidance_transform(
    model: object, node_type: str, contribution: object
) -> _NativeModelOverlay:
    handle, overlays, resolvers, control, shift, transforms, windows, chroma_options = (
        _native_model(model, "model")
    )
    owner_id = f"{node_type}:{len(transforms)}"
    return _NativeModelOverlay(
        handle,
        overlays,
        resolvers,
        control,
        shift,
        (*transforms, (owner_id, contribution)),
        windows,
        chroma_options,
        sampling_cache=_native_model_sampling_cache(model),
        sampling_timeline=_native_model_sampling_timeline(model),
        sampling_space=_native_model_sampling_space(model),
    )


def _guidance_transform_factory(name: str, *args: object, **kwargs: object) -> object:
    module = importlib.import_module("dinkster_inference_torch.guidance_transforms")
    return getattr(module, name)(*args, **kwargs)


def _model_family_is_flow(model: object) -> bool:
    handle = _native_model(model, "model")[0]
    inference = importlib.import_module("dinkster_inference")
    family = next(
        item for item in inference.builtin_families() if item.id == handle.recipe.family_id
    )
    return bool(inference.is_flow_parameterization(family.sampling.parameterization))


class GenerationCfgZeroStar(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.cfg_zero_star")

    @classmethod
    def execute(cls, *, model: object) -> Mapping[str, object]:
        contribution = _guidance_transform_factory("cfg_zero_star")
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.cfg_zero_star", contribution)
        )


class GenerationCfgNorm(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.cfg_norm")

    @classmethod
    def execute(cls, *, model: object, strength: float, pre_cfg: bool) -> Mapping[str, object]:
        _check_bounds(("strength", strength, 0.0, 100.0))
        contribution = _guidance_transform_factory("cfg_norm", float(strength), bool(pre_cfg))
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.cfg_norm", contribution)
        )


class GenerationTCFG(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.tcfg")

    @classmethod
    def execute(cls, *, model: object) -> Mapping[str, object]:
        contribution = _guidance_transform_factory("tcfg")
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.tcfg", contribution)
        )


class GenerationFreSca(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.fresca")

    @classmethod
    def execute(
        cls, *, model: object, scale_low: float, scale_high: float, freq_cutoff: int
    ) -> Mapping[str, object]:
        _check_bounds(
            ("scale_low", scale_low, 0.0, 10.0),
            ("scale_high", scale_high, 0.0, 10.0),
            ("freq_cutoff", freq_cutoff, 1, 10_000),
        )
        contribution = _guidance_transform_factory(
            "fresca",
            scale_low=float(scale_low),
            scale_high=float(scale_high),
            freq_cutoff=int(freq_cutoff),
        )
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.fresca", contribution)
        )


class GenerationLazyCache(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.lazy_cache")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        (
            handle,
            overlays,
            resolvers,
            control,
            shift,
            transforms,
            windows,
            chroma_options,
        ) = _native_model(model, "model")
        if _native_model_sampling_cache(model) is not None:
            raise ValueError("a model can carry only one sampling cache")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        cache = inference_torch.LazyCacheConfig(
            float(reuse_threshold),
            float(start_percent),
            float(end_percent),
        )
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                control,
                shift,
                transforms,
                windows,
                chroma_options,
                sampling_cache=cache,
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=_native_model_sampling_space(model),
            )
        )


class GenerationEasyCache(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.easy_cache")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        (
            handle,
            overlays,
            resolvers,
            control,
            shift,
            transforms,
            windows,
            chroma_options,
        ) = _native_model(model, "model")
        if _native_model_sampling_cache(model) is not None:
            raise ValueError("a model can carry only one sampling cache")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        cache = inference_torch.EasyCacheConfig(
            float(reuse_threshold),
            float(start_percent),
            float(end_percent),
        )
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                control,
                shift,
                transforms,
                windows,
                chroma_options,
                sampling_cache=cache,
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=_native_model_sampling_space(model),
            )
        )


def _sampling_parameter_curve(value: object) -> object:
    inference = importlib.import_module("dinkster_inference")
    raw_points = getattr(value, "points", None)
    interpolation = getattr(value, "interpolation", None)
    if type(raw_points) is not tuple or interpolation not in ("linear", "monotone_cubic"):
        raise TypeError("sol_tau must be a dinkster.curve value")
    try:
        points: list[tuple[float, float]] = []
        for raw_point in cast("tuple[Any, ...]", raw_points):
            point = cast("tuple[Any, ...]", raw_point)
            if type(raw_point) is not tuple or len(point) != 2:
                raise TypeError
            points.append((float(point[0]), float(point[1])))
    except (TypeError, ValueError) as error:
        raise TypeError("sol_tau must be a dinkster.curve value") from error
    return inference.SamplingParameterCurve(tuple(points), interpolation)


class GenerationAttentionSchedule(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.attention_schedule")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        approximate_provider: str,
        start_percent: float,
        end_percent: float,
        conditioning_sink: str = "off",
        sol_tau: object = None,
    ) -> Mapping[str, object]:
        (
            handle,
            overlays,
            resolvers,
            control,
            shift,
            transforms,
            windows,
            chroma_options,
        ) = _native_model(model, "model")
        if _native_model_sampling_timeline(model) is not None:
            raise ValueError("a model can carry only one sampling timeline")
        raw_statuses = getattr(getattr(handle.runtime, "assembled", None), "attention_status", None)
        diffusion_statuses: list[object] = []
        if isinstance(raw_statuses, Mapping):
            for role, status in cast("Mapping[object, object]", raw_statuses).items():
                if role in ("unet", "flux"):
                    diffusion_statuses.append(status)
        if not diffusion_statuses or any(
            getattr(status, "primary", None) != approximate_provider
            for status in diffusion_statuses
        ):
            raise ValueError(
                "attention schedule provider must match the model's selected diffusion provider"
            )
        if any(
            getattr(status, "authenticated", False) is not True for status in diffusion_statuses
        ):
            raise ValueError("attention schedule requires an authenticated provider route")
        inference = importlib.import_module("dinkster_inference")
        sink_modifiers = {
            "off": None,
            "exact_kv": "sol_conditioning_exact_kv",
        }
        if conditioning_sink not in sink_modifiers:
            raise ValueError("attention schedule conditioning sink is unsupported")
        modifier_name = sink_modifiers[conditioning_sink]
        if modifier_name is not None and approximate_provider != "sol":
            raise ValueError("attention schedule conditioning sink requires the Sol provider")
        start = float(start_percent)
        end = float(end_percent)
        curve = None if sol_tau is None else _sampling_parameter_curve(sol_tau)
        schedule = inference.SamplingTimelineSchedule(
            approximate_provider,
            start,
            end,
            curve,
            (
                ()
                if modifier_name is None
                else (inference.AttentionModifierSchedule(modifier_name, start, end),)
            ),
        )
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                control,
                shift,
                transforms,
                windows,
                chroma_options,
                sampling_cache=_native_model_sampling_cache(model),
                sampling_timeline=schedule,
                sampling_space=_native_model_sampling_space(model),
            )
        )


def _model_with_context_windows(model: object, spec: ContextWindowsSpec) -> _NativeModelOverlay:
    (
        handle,
        overlays,
        resolvers,
        control,
        shift,
        transforms,
        existing,
        chroma_options,
    ) = _native_model(model, "model")
    if existing is not None:
        raise ValueError("a model can carry only one context-windows configuration")
    return _NativeModelOverlay(
        handle,
        overlays,
        resolvers,
        control,
        shift,
        transforms,
        spec,
        chroma_options,
        sampling_cache=_native_model_sampling_cache(model),
        sampling_timeline=_native_model_sampling_timeline(model),
        sampling_space=_native_model_sampling_space(model),
    )


def _context_windows_spec(
    *,
    length: int,
    overlap: int,
    schedule: str,
    stride: int,
    closed_loop: bool,
    fuse_method: str,
    dim: int,
    freenoise: bool,
    causal_anchor: bool,
    latent_retain_indices: tuple[int, ...],
) -> ContextWindowsSpec:
    inference = importlib.import_module("dinkster_inference")
    try:
        schedule_value = inference.ContextWindowSchedule(schedule)
    except ValueError:
        raise ValueError(f"unknown context schedule {schedule!r}") from None
    try:
        fuse_value = inference.ContextFuseMethod(fuse_method)
    except ValueError:
        raise ValueError(f"unknown fuse method {fuse_method!r}") from None
    if fuse_value is inference.ContextFuseMethod.RELATIVE:
        raise ValueError("relative fusing is not supported")
    return inference.ContextWindowsSpec(
        schedule_value,
        fuse_value,
        length,
        overlap,
        stride=stride,
        closed_loop=closed_loop,
        dim=dim,
        freenoise=freenoise,
        causal_anchor=causal_anchor,
        latent_retain_indices=latent_retain_indices,
    )


class GenerationContextWindowsManual(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.context_windows_manual")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        context_length: int,
        context_overlap: int,
        context_schedule: str,
        context_stride: int,
        closed_loop: bool,
        fuse_method: str,
        dim: int,
        freenoise: bool,
        causal_window_fix: bool = True,
    ) -> Mapping[str, object]:
        spec = _context_windows_spec(
            length=context_length,
            overlap=context_overlap,
            schedule=context_schedule,
            stride=context_stride,
            closed_loop=closed_loop,
            fuse_method=fuse_method,
            dim=dim,
            freenoise=freenoise,
            causal_anchor=causal_window_fix,
            latent_retain_indices=(),
        )
        return cls.outputs(model=_model_with_context_windows(model, spec))


class GenerationWanContextWindowsManual(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.wan_context_windows_manual")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        context_length: int,
        context_overlap: int,
        context_schedule: str,
        context_stride: int,
        closed_loop: bool,
        fuse_method: str,
        freenoise: bool,
    ) -> Mapping[str, object]:
        if type(context_length) is not int or type(context_overlap) is not int:
            raise TypeError("context_length and context_overlap must be ints")
        # WAN's causal VAE packs 4n+1 real frames into n+1 latent frames.
        spec = _context_windows_spec(
            length=max((context_length - 1) // 4 + 1, 1),
            overlap=max(context_overlap // 4, 0),
            schedule=context_schedule,
            stride=context_stride,
            closed_loop=closed_loop,
            fuse_method=fuse_method,
            dim=2,
            freenoise=freenoise,
            causal_anchor=True,
            latent_retain_indices=(),
        )
        return cls.outputs(model=_model_with_context_windows(model, spec))


class GenerationLTXVContextWindows(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_context_windows")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        context_length: int,
        context_overlap: int,
        context_schedule: str,
        context_stride: int,
        closed_loop: bool,
        fuse_method: str,
        freenoise: bool,
        retain_first_frame: bool,
    ) -> Mapping[str, object]:
        if type(context_length) is not int or type(context_overlap) is not int:
            raise TypeError("context_length and context_overlap must be ints")
        spec = _context_windows_spec(
            length=max((context_length - 1) // 8 + 1, 1),
            overlap=max(context_overlap // 8, 0),
            schedule=context_schedule,
            stride=context_stride,
            closed_loop=closed_loop,
            fuse_method=fuse_method,
            dim=2,
            freenoise=freenoise,
            causal_anchor=True,
            latent_retain_indices=(0,) if retain_first_frame else (),
        )
        return cls.outputs(model=_model_with_context_windows(model, spec))


class GenerationAdaptiveProjectedGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.adaptive_projected_guidance")

    @classmethod
    def execute(
        cls, *, model: object, eta: float, norm_threshold: float, momentum: float
    ) -> Mapping[str, object]:
        _check_bounds(
            ("eta", eta, -10.0, 10.0),
            ("norm_threshold", norm_threshold, 0.0, 50.0),
            ("momentum", momentum, -5.0, 1.0),
        )
        contribution = _guidance_transform_factory(
            "apg", float(eta), float(norm_threshold), float(momentum)
        )
        return cls.outputs(
            model=_model_with_guidance_transform(
                model, "dinkster.adaptive_projected_guidance", contribution
            )
        )


class GenerationMahiroGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.mahiro_guidance")

    @classmethod
    def execute(cls, *, model: object) -> Mapping[str, object]:
        contribution = _guidance_transform_factory("mahiro")
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.mahiro_guidance", contribution)
        )


class GenerationEpsilonScaling(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.epsilon_scaling")

    @classmethod
    def execute(cls, *, model: object, scaling_factor: float) -> Mapping[str, object]:
        _check_bounds(("scaling_factor", scaling_factor, 0.5, 1.5))
        contribution = _guidance_transform_factory("epsilon_scaling", float(scaling_factor))
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.epsilon_scaling", contribution)
        )


class GenerationCFGOverride(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.cfg_override")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        cfg: float,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("cfg", cfg, 0.0, 100.0),
            ("start_percent", start_percent, 0.0, 1.0),
            ("end_percent", end_percent, 0.0, 1.0),
        )
        if start_percent > end_percent:
            raise ValueError("start_percent must be less than or equal to end_percent")
        runtime, sampling_shift, _device = _require_custom_sampling_runtime(model, "CFGOverride")
        percent_to_sigma = _bind_sampling_shift(
            runtime.custom_sampling_percent_to_sigma, sampling_shift
        )
        sigma_high = percent_to_sigma(start_percent, return_actual_sigma=True)
        sigma_low = percent_to_sigma(end_percent, return_actual_sigma=True)
        contribution = _guidance_transform_factory(
            "cfg_override",
            float(cfg),
            float(sigma_low),
            float(sigma_high),
        )
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.cfg_override", contribution)
        )


class GenerationRescaleCfg(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.rescale_cfg")

    @classmethod
    def execute(cls, *, model: object, multiplier: float) -> Mapping[str, object]:
        _check_bounds(("multiplier", multiplier, 0.0, 1.0))
        contribution = _guidance_transform_factory(
            "rescale_cfg", float(multiplier), flow=_model_family_is_flow(model)
        )
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.rescale_cfg", contribution)
        )


class GenerationRenormCfg(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.renorm_cfg")

    @classmethod
    def execute(cls, *, model: object, cfg_trunc: float, renorm_cfg: float) -> Mapping[str, object]:
        _check_bounds(
            ("cfg_trunc", cfg_trunc, 0.0, 100.0),
            ("renorm_cfg", renorm_cfg, 0.0, 100.0),
        )
        contribution = _guidance_transform_factory(
            "renorm_cfg", float(cfg_trunc), float(renorm_cfg)
        )
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.renorm_cfg", contribution)
        )


class GenerationTemporalScoreRescaling(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.temporal_score_rescaling")

    @classmethod
    def execute(cls, *, model: object, tsr_k: float, tsr_sigma: float) -> Mapping[str, object]:
        _check_bounds(
            ("tsr_k", tsr_k, 0.01, 100.0),
            ("tsr_sigma", tsr_sigma, 0.01, 100.0),
        )
        contribution = _guidance_transform_factory(
            "temporal_score_rescaling",
            float(tsr_k),
            float(tsr_sigma),
            flow=_model_family_is_flow(model),
        )
        return cls.outputs(
            model=_model_with_guidance_transform(
                model, "dinkster.temporal_score_rescaling", contribution
            )
        )


class GenerationNormalizedAttentionGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.nag")

    @classmethod
    def execute(
        cls, *, model: object, nag_scale: float, nag_alpha: float, nag_tau: float
    ) -> Mapping[str, object]:
        _check_bounds(
            ("nag_scale", nag_scale, 0.0, 50.0),
            ("nag_alpha", nag_alpha, 0.0, 1.0),
            ("nag_tau", nag_tau, 1.0, 10.0),
        )
        contribution = _guidance_transform_factory(
            "nag", float(nag_scale), float(nag_alpha), float(nag_tau)
        )
        return cls.outputs(
            model=_model_with_guidance_transform(model, "dinkster.nag", contribution)
        )


def _retime_ltx_conditioning(
    positive: object,
    negative: object,
    frame_rate: float,
    *,
    accept_carriers: bool,
    payload_type_name: str,
    family_id: str,
    family_name: str,
) -> dict[str, object]:
    rate = float(frame_rate)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"frame_rate must be a positive finite number, got {frame_rate}")
    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    payload_type = getattr(inference_torch, payload_type_name)

    def retimed(value: object, name: str) -> object:
        if accept_carriers and type(value) is inference.ConditioningCarrier:
            return _with_ltx_frame_rate(
                value,
                inference,
                rate,
                family_id=family_id,
                family_name=family_name,
            )
        prepared = _prepared_multistream_carrier(value, inference, name)
        if prepared is None:
            raise TypeError(f"{name} must contain prepared multi-stream conditioning")
        if type(prepared.payload) is not payload_type:
            raise TypeError(f"{name} must contain {family_name} text conditioning")
        payload = replace(prepared.payload, frame_rate=rate)
        rows: list[list[object]] = [
            [
                inference.PreparedMultiStreamConditioning(prepared.runtime_identity, payload),
                dict[str, object](),
            ]
        ]
        return rows

    return {
        "positive": retimed(positive, "positive"),
        "negative": retimed(negative, "negative"),
    }


class GenerationLTXAVConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxav_conditioning")

    @classmethod
    def execute(
        cls, *, positive: object, negative: object, frame_rate: float
    ) -> Mapping[str, object]:
        return cls.outputs(
            **_retime_ltx_conditioning(
                positive,
                negative,
                frame_rate,
                accept_carriers=True,
                payload_type_name="LTXAVPreparedConditioning",
                family_id="dinkster.ltxav",
                family_name="LTX-2 audio-video",
            )
        )


def _ltxav_audio_codec(value: object) -> tuple[Any, Any, Any]:
    inference = importlib.import_module("dinkster_inference")
    if isinstance(value, NativeComponentHandle):
        value.require_active()
        recipe = value.recipe
        if (
            recipe is None
            or tuple(source.role for source in recipe.sources) != ("audio_vae",)
            or recipe.runtime_identity != value.resource_identity
        ):
            raise TypeError("audio_vae requires an audio_vae source and matching recipe identity")
        inference.ComponentBinding("audio_vae", recipe.family_id, value.resource_identity)
        component = cast("Any", value.component)
        audio_vae = getattr(component, "audio_vae", None)
        for method in ("encode", "decode"):
            if not callable(getattr(audio_vae, method, None)):
                raise TypeError(f"audio_vae codec requires callable audio_vae.{method}")
        config = getattr(audio_vae, "config", None)
        if (
            getattr(config, "z_channels", None) != 8
            or getattr(config, "latent_frequency_bins", None) != 16
        ):
            raise TypeError("audio_vae codec requires [batch,8,time,16] latent geometry")
        vocoder = getattr(component, "vocoder", None)
        if not callable(vocoder) or getattr(vocoder, "config", None) is None:
            raise TypeError("audio_vae codec requires a callable vocoder with config")
        return component, value.load_device, value.stage
    raise TypeError("audio_vae must provide a native audio codec component")


def _ltxav_audio_vae(value: object) -> tuple[Any, Any, Any]:
    codec, load_device, stage = _ltxav_audio_codec(value)
    return codec.audio_vae, load_device, stage


def _ltxav_reference_audio_value(value: object) -> tuple[Any, int]:
    if not isinstance(value, Mapping):
        raise TypeError("reference_audio must be the standard waveform/sample_rate mapping")
    audio = cast("Mapping[str, object]", value)
    if set(audio) != {"waveform", "sample_rate"}:
        raise TypeError("reference_audio must be the standard waveform/sample_rate mapping")
    torch = _torch()
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if type(waveform) is not torch.Tensor:
        raise TypeError("reference_audio.waveform must be an exact torch.Tensor")
    tensor = cast("Any", waveform)
    if (
        tensor.layout is not torch.strided
        or not tensor.is_floating_point()
        or tensor.ndim != 3
        or tensor.shape[0] <= 0
        or tensor.shape[1] not in (1, 2)
        or tensor.shape[2] <= 0
    ):
        raise ValueError("reference_audio.waveform must be nonempty floating [batch,1|2,samples]")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("reference_audio.sample_rate must be a positive integer")
    return tensor, sample_rate


def _attach_ltxav_reference_audio(
    positive: object, negative: object, audio_latent: Any
) -> dict[str, object]:
    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    positive_is_carrier = type(positive) is inference.ConditioningCarrier
    negative_is_carrier = type(negative) is inference.ConditioningCarrier
    if positive_is_carrier != negative_is_carrier:
        raise TypeError("positive and negative must use the same conditioning representation")
    if positive_is_carrier:
        attached = inference_torch.ltxav_reference_audio_conditioning(
            positive, negative, audio_latent
        )
        return {"positive": attached[0], "negative": attached[1]}

    positive_prepared = _prepared_multistream_carrier(positive, inference, "positive")
    negative_prepared = _prepared_multistream_carrier(negative, inference, "negative")
    if positive_prepared is None or negative_prepared is None:
        raise TypeError("positive and negative must contain prepared LTX-2 conditioning")
    if positive_prepared.runtime_identity != negative_prepared.runtime_identity:
        raise ValueError("positive and negative conditioning were prepared by different runtimes")
    audio_tokens = audio_latent.permute(0, 2, 1, 3).reshape(
        audio_latent.shape[0],
        audio_latent.shape[2],
        audio_latent.shape[1] * audio_latent.shape[3],
    )

    def attach(prepared: Any, name: str) -> list[list[object]]:
        payload = prepared.payload
        if type(payload) is not inference_torch.LTXAVPreparedConditioning:
            raise TypeError(f"{name} must contain LTX-2 audio-video text conditioning")
        if payload.reference_audio is not None:
            raise ValueError(f"{name} LTX-2 conditioning already has reference audio")
        return [
            [
                inference.PreparedMultiStreamConditioning(
                    prepared.runtime_identity,
                    replace(payload, reference_audio=audio_tokens),
                ),
                {},
            ]
        ]

    return {
        "positive": attach(positive_prepared, "positive"),
        "negative": attach(negative_prepared, "negative"),
    }


class GenerationLTXAVReferenceAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxav_reference_audio")

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        reference_audio: object,
        audio_vae: object,
    ) -> Mapping[str, object]:
        audio_latent = _encode_ltxav_reference_audio(reference_audio, audio_vae)
        return cls.outputs(**_attach_ltxav_reference_audio(positive, negative, audio_latent))


def _encode_ltxav_reference_audio(reference_audio: object, audio_vae: object) -> Any:
    waveform, sample_rate = _ltxav_reference_audio_value(reference_audio)
    module, load_device, stage = _ltxav_audio_vae(audio_vae)
    torch = _torch()
    with stage():
        with torch.inference_mode():
            audio_latent = module.encode(waveform.to(load_device), sample_rate).float()
    if (
        type(audio_latent) is not torch.Tensor
        or audio_latent.layout is not torch.strided
        or not audio_latent.is_floating_point()
        or audio_latent.ndim != 4
        or audio_latent.shape[0] != waveform.shape[0]
        or audio_latent.shape[1] != 8
        or audio_latent.shape[2] <= 0
        or audio_latent.shape[3] != 16
    ):
        raise TypeError("LTX-2 audio VAE must encode nonempty floating [batch,8,time,16]")
    return audio_latent


class GenerationLTXAVIDLoRAReferenceAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxav_id_lora_reference_audio")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        positive: object,
        negative: object,
        reference_audio: object,
        audio_vae: object,
        identity_guidance_scale: float,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("identity_guidance_scale", identity_guidance_scale, 0.0, 100.0),
            ("start_percent", start_percent, 0.0, 1.0),
            ("end_percent", end_percent, 0.0, 1.0),
        )
        handle, inference_torch, transform_index = _ltxav_guidance_runtime(model)
        audio_latent = _encode_ltxav_reference_audio(reference_audio, audio_vae)
        conditioning = _attach_ltxav_reference_audio(positive, negative, audio_latent)
        guided_model = model
        if identity_guidance_scale != 0.0:
            sigma_start = handle.runtime.custom_sampling_percent_to_sigma(
                start_percent, return_actual_sigma=False
            )
            sigma_end = handle.runtime.custom_sampling_percent_to_sigma(
                end_percent, return_actual_sigma=False
            )
            contribution = inference_torch.ltxav_identity_guidance(
                float(identity_guidance_scale),
                float(sigma_start),
                float(sigma_end),
                order=transform_index,
            )
            guided_model = _model_with_guidance_transform(
                model,
                "dinkster.ltxav_id_lora_reference_audio",
                contribution,
            )
        return cls.outputs(model=guided_model, **conditioning)


def _ltxav_guidance_runtime(model: object) -> tuple[Any, Any, int]:
    handle, _, _, _, _, transforms, _, _ = _native_model(model, "model")
    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    recipe = handle.recipe
    if "diffusion" not in tuple(source.role for source in recipe.sources):
        raise TypeError("model requires a diffusion source binding")
    inference.ComponentBinding("diffusion", recipe.family_id, recipe.runtime_identity)
    runtime = handle.runtime
    if runtime.runtime_identity != recipe.runtime_identity:
        raise ValueError("model runtime identity must match its recipe")
    for method in ("custom_sampling_percent_to_sigma", "sample_custom", "check_custom_sampling"):
        if not callable(getattr(runtime, method, None)):
            raise TypeError(f"model requires callable {method}")
    geometry = getattr(runtime, "component_sampling_runtime", runtime)
    for name, fields in (
        ("video_vae_config", ("latent_channels", "spatial_ratio", "temporal_ratio")),
        ("audio_vae_config", ("z_channels", "latent_frequency_bins")),
    ):
        config = getattr(geometry, name, None)
        for field in fields:
            dimension = getattr(config, field, None)
            if type(dimension) is not int or dimension <= 0:
                raise TypeError(f"model requires positive {name}.{field} for audio-video guidance")
    return handle, inference_torch, len(transforms)


class GenerationLTXVSpatioTemporalGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_spatiotemporal_guidance")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        scale: float,
        blocks: str,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("scale", scale, 0.0, 100.0),
            ("start_percent", start_percent, 0.0, 1.0),
            ("end_percent", end_percent, 0.0, 1.0),
        )
        if type(blocks) is not str:
            raise TypeError("blocks must be an exact string")
        block_set = frozenset(int(value) for value in re.findall(r"\d+", blocks))
        if scale == 0.0 or not block_set:
            return cls.outputs(model=model)
        handle, inference_torch, transform_index = _ltxav_guidance_runtime(model)
        sigma_start = handle.runtime.custom_sampling_percent_to_sigma(
            start_percent, return_actual_sigma=False
        )
        sigma_end = handle.runtime.custom_sampling_percent_to_sigma(
            end_percent, return_actual_sigma=False
        )
        contribution = inference_torch.ltxav_spatiotemporal_guidance(
            float(scale),
            block_set,
            float(sigma_start),
            float(sigma_end),
            lane_id=f"dinkster.ltxav.stg-perturbed:{transform_index}",
            order=transform_index,
        )
        return cls.outputs(
            model=_model_with_guidance_transform(
                model,
                "dinkster.ltxv_spatiotemporal_guidance",
                contribution,
            )
        )


class GenerationLTXVModalityGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_modality_guidance")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        modality_scale: float,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("modality_scale", modality_scale, 1.0, 100.0),
            ("start_percent", start_percent, 0.0, 1.0),
            ("end_percent", end_percent, 0.0, 1.0),
        )
        if math.isclose(modality_scale, 1.0):
            return cls.outputs(model=model)
        handle, inference_torch, transform_index = _ltxav_guidance_runtime(model)
        sigma_start = handle.runtime.custom_sampling_percent_to_sigma(
            start_percent, return_actual_sigma=False
        )
        sigma_end = handle.runtime.custom_sampling_percent_to_sigma(
            end_percent, return_actual_sigma=False
        )
        contribution = inference_torch.ltxav_modality_guidance(
            float(modality_scale),
            float(sigma_start),
            float(sigma_end),
            lane_id=f"dinkster.ltxav.modality-decoupled:{transform_index}",
            order=transform_index,
        )
        return cls.outputs(
            model=_model_with_guidance_transform(
                model,
                "dinkster.ltxv_modality_guidance",
                contribution,
            )
        )


class GenerationLTXVDurationPredictor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_duration_predictor")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        positive: object,
        duration_head: object,
        frame_rate: float,
        min_seconds: float,
        max_seconds: float,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("frame_rate", frame_rate, 1.0, 120.0),
            ("min_seconds", min_seconds, 0.5, 120.0),
            ("max_seconds", max_seconds, 0.5, 120.0),
        )
        if max_seconds < min_seconds:
            raise ValueError("max_seconds must be greater than or equal to min_seconds")
        model_handle, inference_torch, _ = _ltxav_guidance_runtime(model)
        if not isinstance(duration_head, NativeComponentHandle):
            raise TypeError("duration_head must provide the LTX-2 duration component")
        duration_head.require_active()
        recipe = duration_head.recipe
        if (
            recipe is None
            or tuple(source.role for source in recipe.sources) != ("duration_head",)
            or recipe.runtime_identity != duration_head.resource_identity
        ):
            raise TypeError(
                "duration_head requires a duration_head source and matching recipe identity"
            )

        inference = importlib.import_module("dinkster_inference")
        inference.ComponentBinding(
            "duration_head", recipe.family_id, duration_head.resource_identity
        )
        if not callable(duration_head.component):
            raise TypeError("duration_head component must be callable")
        runtime = model_handle.runtime
        if type(positive) is inference.ConditioningCarrier:
            execution = resolve_component_execution(model_handle, positive, [], inference)
            if execution is None:
                raise TypeError("positive must contain component-bound LTX-2 text conditioning")
            runtime, prepared_positive, _ = execution
            prepared = _prepared_multistream_carrier(prepared_positive, inference, "positive")
            if prepared is None:
                raise TypeError("positive must contain LTX-2 text conditioning")
            conditioning = prepared.payload
        else:
            prepared = _prepared_multistream_carrier(positive, inference, "positive")
            if prepared is None:
                raise TypeError("positive must contain LTX-2 text conditioning")
            if prepared.runtime_identity != runtime.conditioning_identity:
                raise ValueError("positive conditioning was prepared by a different runtime")
            conditioning = prepared.payload
        if type(conditioning) is not inference_torch.LTXAVPreparedConditioning:
            raise TypeError("positive must contain LTX-2 text conditioning")

        assembled = getattr(runtime, "assembled", None)
        diffusion = getattr(assembled, "diffusion", None)
        if not callable(getattr(assembled, "compute_dtype", None)):
            raise TypeError("model requires callable assembled.compute_dtype")
        if not callable(getattr(diffusion, "preprocess_text_embeds", None)):
            raise TypeError("model requires callable diffusion.preprocess_text_embeds")
        config = getattr(diffusion, "config", None)
        for field in ("cross_attention_dim", "audio_cross_attention_dim"):
            dimension = getattr(config, field, None)
            if type(dimension) is not int or dimension <= 0:
                raise TypeError(f"model requires positive diffusion.config.{field}")
        torch = _torch()
        with duration_head.stage_with(model_handle, "diffusion"):
            with torch.inference_mode():
                diffusion = runtime.assembled.diffusion
                context = conditioning.text[:1].to(
                    device=model_handle.load_device,
                    dtype=runtime.assembled.compute_dtype("diffusion"),
                )
                processed = diffusion.preprocess_text_embeds(context)
                video_tokens, audio_tokens = torch.split(
                    processed,
                    [
                        diffusion.config.cross_attention_dim,
                        diffusion.config.audio_cross_attention_dim,
                    ],
                    dim=-1,
                )
                duration_module = cast("Any", duration_head.component)
                seconds = float(
                    duration_module(video_tokens.float(), audio_tokens.float())[0].item()
                )
        frames = inference_torch.ltx_duration_frames(
            seconds,
            float(frame_rate),
            float(min_seconds),
            float(max_seconds),
        )
        return cls.outputs(num_frames=frames, seconds=seconds)


class GenerationLTXVConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_conditioning")

    @classmethod
    def execute(
        cls, *, positive: object, negative: object, frame_rate: float
    ) -> Mapping[str, object]:
        return cls.outputs(
            **_retime_ltx_conditioning(
                positive,
                negative,
                frame_rate,
                accept_carriers=True,
                payload_type_name="LTXVPreparedConditioning",
                family_id="dinkster.ltxv",
                family_name="LTX-Video",
            )
        )


def _ltxv_media_codec(value: object) -> Any:
    inference = importlib.import_module("dinkster_inference")
    if isinstance(value, NativeComponentHandle):
        codec = _native_component_codec(value)
    elif isinstance(value, NativeRuntimeHandle):
        value.require_active()
        runtime = value.runtime
        descriptor = getattr(getattr(runtime, "codec", None), "descriptor", None)
        if descriptor is None:
            raise TypeError("vae runtime requires a codec descriptor")
        for method in ("encode_content", "decode_latent"):
            if not callable(getattr(runtime, method, None)):
                raise TypeError(f"vae runtime requires callable {method}")
        codec = inference.RuntimeCodecAdapter(value, descriptor=descriptor)
    else:
        codec = value
    codec = inference.require_inference_codec_handle(codec, "vae")
    descriptor = codec.descriptor
    latent = descriptor.latent
    if (
        descriptor.kind != "video"
        or descriptor.content_channels != 3
        or latent.dimensions != 3
        or not latent.temporal_causal
    ):
        raise TypeError("vae requires an RGB causal video codec with three latent dimensions")
    for field in ("channels", "spatial_downscale", "temporal_downscale"):
        dimension = getattr(latent, field)
        if type(dimension) is not int or dimension <= 0:
            raise TypeError(f"vae requires positive latent.{field}")
    return codec


def _ltxv_media_latent(value: object, name: str) -> tuple[dict[object, object], Any, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a latent mapping")
    metadata = dict(cast("Mapping[object, object]", value))
    inference = importlib.import_module("dinkster_inference")
    torch = _torch()
    samples, streams = _latent_samples(cast("object", value), torch, inference, name)
    if streams is None:
        streams = inference.MultiStreamLatent.from_pairs((("video", samples),))
    elif streams.roles != ("video",):
        raise TypeError(f"{name} must contain the exact LTX-Video latent stream")
    video = streams.by_role("video")
    if (
        type(video) is not torch.Tensor
        or not video.is_floating_point()
        or video.ndim != 5
        or any(size <= 0 for size in video.shape)
    ):
        raise TypeError(f"{name} video stream must be nonempty floating [B,C,T,H,W]")
    return metadata, streams, video


def _ltxv_image(value: object) -> Any:
    torch = _torch()
    if type(value) is not torch.Tensor:
        raise TypeError("image must be nonempty floating [frames,height,width,channels>=3]")
    image = cast("Any", value)
    if (
        not image.is_floating_point()
        or image.ndim != 4
        or any(size <= 0 for size in image.shape[:3])
        or image.shape[3] < 3
    ):
        raise TypeError("image must be nonempty floating [frames,height,width,channels>=3]")
    return image


def _encode_ltxv_frames(codec: Any, image: Any, width: int, height: int) -> Any:
    torch = _torch()
    frames = _ltxv_image(image)
    nchw = frames[..., :3].movedim(-1, 1)
    if tuple(nchw.shape[-2:]) != (height, width):
        old_height, old_width = nchw.shape[-2:]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        nchw = nchw.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
        nchw = torch.nn.functional.interpolate(nchw, size=(height, width), mode="bilinear")
    content = nchw.movedim(0, 1).unsqueeze(0)
    with codec.stage():
        with torch.inference_mode():
            encoded = codec.encode_content(content.to(codec.load_device))
    if (
        type(encoded) is not torch.Tensor
        or not encoded.is_floating_point()
        or encoded.ndim != 5
        or encoded.shape[0] != 1
        or encoded.shape[1] != codec.descriptor.latent.channels
        or any(size <= 0 for size in encoded.shape)
    ):
        raise TypeError("LTX-Video VAE encode must return nonempty floating [1,C,T,H,W]")
    return encoded


def _repeat_ltxv_frames(encoded: Any, batch_size: int) -> Any:
    if batch_size == 1:
        return encoded
    return encoded.expand(batch_size, -1, -1, -1, -1)


def _ltxv_media_output(
    metadata: Mapping[object, object], streams: object, denoise_mask: object
) -> dict[object, object]:
    result = dict(metadata)
    result["samples"] = streams
    result["noise_mask"] = denoise_mask
    return result


class GenerationLTXVImageToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_image_to_video")

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        image: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        strength: float,
    ) -> Mapping[str, object]:
        for name, value in (
            ("width", width),
            ("height", height),
            ("length", length),
            ("batch_size", batch_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        codec = _ltxv_media_codec(vae)
        latent = codec.descriptor.latent
        if width % latent.spatial_downscale or height % latent.spatial_downscale:
            raise ValueError("width and height must be divisible by the LTX-Video spatial scale")
        latent_height = height // latent.spatial_downscale
        latent_width = width // latent.spatial_downscale
        torch = _torch()
        video = torch.zeros(
            (
                batch_size,
                latent.channels,
                (length - 1) // latent.temporal_downscale + 1,
                latent_height,
                latent_width,
            ),
            device="cpu",
            dtype=torch.float32,
        )
        inference = importlib.import_module("dinkster_inference")
        streams = inference.MultiStreamLatent.from_pairs((("video", video),))
        encoded = _repeat_ltxv_frames(_encode_ltxv_frames(codec, image, width, height), batch_size)
        conditioned, denoise_mask = importlib.import_module(
            "dinkster_inference_torch"
        ).ltxv_condition_initial_frames(streams, encoded, strength=float(strength))
        return cls.outputs(
            positive=positive,
            negative=negative,
            latent={"samples": conditioned, "noise_mask": denoise_mask},
        )


class GenerationLTXVImageToVideoInplace(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_image_to_video_inplace")

    @classmethod
    def execute(
        cls,
        *,
        vae: object,
        image: object,
        latent: object,
        strength: float,
        bypass: bool = False,
    ) -> Mapping[str, object]:
        if type(bypass) is not bool:
            raise TypeError("bypass must be a bool")
        if bypass:
            return cls.outputs(latent=latent)
        metadata, streams, video = _ltxv_media_latent(latent, "latent")
        codec = _ltxv_media_codec(vae)
        descriptor = codec.descriptor.latent
        width = video.shape[4] * descriptor.spatial_downscale
        height = video.shape[3] * descriptor.spatial_downscale
        encoded = _repeat_ltxv_frames(
            _encode_ltxv_frames(codec, image, width, height), video.shape[0]
        )
        conditioned, denoise_mask = importlib.import_module(
            "dinkster_inference_torch"
        ).ltxv_condition_initial_frames(
            streams,
            encoded,
            strength=float(strength),
            denoise_mask=metadata.get("noise_mask"),
        )
        return cls.outputs(latent=_ltxv_media_output(metadata, conditioned, denoise_mask))


class GenerationLTXVAddGuide(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_add_guide")

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        latent: object,
        image: object,
        frame_idx: int,
        strength: float,
        attention_mask: object = None,
    ) -> Mapping[str, object]:
        if type(frame_idx) is not int:
            raise TypeError("frame_idx must be an int")
        metadata, streams, video = _ltxv_media_latent(latent, "latent")
        codec = _ltxv_media_codec(vae)
        descriptor = codec.descriptor.latent
        source = _ltxv_image(image)
        frame_count = (
            (source.shape[0] - 1) // descriptor.temporal_downscale
        ) * descriptor.temporal_downscale + 1
        source = source[:frame_count]

        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        ltx_media = importlib.import_module("dinkster_inference_torch.ltx_media")
        without_rate, _ = _split_ltx_frame_rate(positive, inference)
        _, guides = ltx_media.materialize_ltxv_guides(without_rate)
        generated_frames = video.shape[2] - sum(guide.latent_shape[0] for guide in guides)
        if generated_frames <= 0:
            raise ValueError("LTX-Video guides leave no generated latent frames")
        resolved_index = frame_idx
        if resolved_index < 0:
            resolved_index = max(
                (generated_frames - 1) * descriptor.temporal_downscale + 1 + resolved_index,
                0,
            )
        causal = resolved_index == 0 or frame_count == 1
        encoded_source = source if causal else _torch().cat((source[:1], source))
        encoded_source = encoded_source[
            : (encoded_source.shape[0] - 1)
            // descriptor.temporal_downscale
            * descriptor.temporal_downscale
            + 1
        ]
        width = video.shape[4] * descriptor.spatial_downscale
        height = video.shape[3] * descriptor.spatial_downscale
        encoded = _encode_ltxv_frames(codec, encoded_source, width, height)
        if not causal:
            encoded = encoded[:, :, 1:].clone()
        encoded = _repeat_ltxv_frames(encoded, video.shape[0])

        result = inference_torch.ltxv_add_guide(
            positive,
            negative,
            streams,
            encoded,
            frame_index=frame_idx,
            strength=float(strength),
            denoise_mask=metadata.get("noise_mask"),
            attention_mask=cast("Any", attention_mask),
            scale_factors=(
                descriptor.temporal_downscale,
                descriptor.spatial_downscale,
                descriptor.spatial_downscale,
            ),
            causal_fix=causal,
        )
        return cls.outputs(
            positive=result.positive,
            negative=result.negative,
            latent=_ltxv_media_output(metadata, result.latent, result.denoise_mask),
        )


class GenerationLTXVCropGuides(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_crop_guides")

    @classmethod
    def execute(cls, *, positive: object, negative: object, latent: object) -> Mapping[str, object]:
        metadata, streams, _ = _ltxv_media_latent(latent, "latent")
        result = importlib.import_module("dinkster_inference_torch").ltxv_crop_guides(
            positive,
            negative,
            streams,
            metadata.get("noise_mask"),
        )
        return cls.outputs(
            positive=result.positive,
            negative=result.negative,
            latent=_ltxv_media_output(metadata, result.latent, result.denoise_mask),
        )


class GenerationLTXVLatentUpsampler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_latent_upsampler")

    @classmethod
    def execute(
        cls,
        *,
        samples: object,
        upscale_model: object,
        vae: object,
    ) -> Mapping[str, object]:
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a latent mapping")
        torch = _torch()
        latent = cast("Mapping[object, object]", samples).get("samples")
        tensor = cast("Any", latent)
        if (
            type(latent) is not torch.Tensor
            or tensor.layout is not torch.strided
            or not tensor.is_floating_point()
            or tensor.ndim != 5
            or tensor.shape[0] <= 0
            or tensor.shape[1] != 128
            or min(tensor.shape[2:]) <= 0
        ):
            raise ValueError(
                "samples['samples'] must be nonempty floating [batch,128,time,height,width]"
            )
        upscaler = load_registered_component(upscale_model, "upscale_model", "latent_upscaler")
        video_vae = load_registered_component(vae, "vae", "vae")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        model = upscaler.component
        vae_model = video_vae.component
        if type(model) is not inference_torch.LTXLatentUpsampler:
            raise TypeError("upscale_model must contain the LTX-2 latent upscaler")
        if not isinstance(
            vae_model,
            (inference_torch.LTXVideoVAE, inference_torch.LTXDiffusionVideoVAE),
        ):
            raise TypeError("vae must contain an LTX-2 video VAE")
        upscaler_model = cast("Any", model)
        statistics = cast("Any", vae_model).per_channel_statistics
        if video_vae.coordinator is not upscaler.coordinator:
            raise ValueError("upscale_model and vae use different residency coordinators")
        recipe = upscaler.recipe
        assert recipe is not None
        compute_dtype = _torch_dtype(torch, recipe.knobs.vae_dtype)
        with upscaler.coordinator.locked():
            with ExitStack() as stages:
                stages.enter_context(video_vae.stage())
                stages.enter_context(upscaler.stage())
                with torch.inference_mode():
                    value = tensor.to(device=upscaler.load_device, dtype=compute_dtype)
                    value = statistics.un_normalize(value)
                    value = upscaler_model(value)
                    value = statistics.normalize(value)
        output = dict(cast("Mapping[object, object]", samples))
        output["samples"] = value.to(device=tensor.device, dtype=tensor.dtype)
        output.pop("noise_mask", None)
        return cls.outputs(latent=output)


# Conditioning value ops mirror the pinned ComfyUI nodes (nodes.py at
# b78cec87): payload math runs in the payload's own dtype through NumPy, so
# the default arm stays torch-free. BF16 payloads are refused because NumPy
# cannot reproduce bf16 arithmetic bit-exactly.
_CONDITIONING_FLOAT_WIRE_DTYPES = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}
_CONDITIONING_WIRE_BY_ITEMSIZE = {2: "F16", 4: "F32", 8: "F64"}


def _conditioning_carrier(value: object, name: str) -> Any:
    inference = importlib.import_module("dinkster_inference")
    if type(value) is not inference.ConditioningCarrier:
        raise TypeError(f"{name} must come from a Dinkster conditioning node")
    return cast("Any", value)


def _conditioning_op_float(value: object, name: str, low: float, high: float) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a number")
    number = float(cast("int | float", value))
    if not low <= number <= high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
    return number


def _float_payload_array(np_mod: Any, binding: Any, name: str) -> Any:
    dtype = _CONDITIONING_FLOAT_WIRE_DTYPES.get(binding.dtype)
    if dtype is None:
        raise ValueError(
            f"{name} has a {binding.dtype} payload; conditioning math supports F16, F32, and F64"
        )
    return np_mod.frombuffer(binding.data, dtype=np_mod.dtype(dtype)).reshape(binding.shape)


def _scalar_scaled(np_mod: Any, array: Any, scalar: float) -> Any:
    # Torch CPU half kernels keep the python scalar in float32 opmath and
    # round once per element; NEP50 would first demote the scalar to float16
    # and drift one ulp from the pin.
    if array.dtype == np_mod.float16:
        product = array.astype(np_mod.float32) * np_mod.float32(scalar)
        return product.astype(np_mod.float16)
    return array * scalar


def _float_payload_binding(
    np_mod: Any, inference: Any, reference_id: str, array: Any, space: str
) -> Any:
    data = np_mod.ascontiguousarray(array)
    wire = (
        _CONDITIONING_WIRE_BY_ITEMSIZE.get(int(data.dtype.itemsize))
        if data.dtype.kind == "f"
        else None
    )
    if wire is None:
        raise ValueError(f"unsupported conditioning payload dtype: {data.dtype}")
    if data.dtype.byteorder == ">":
        data = data.astype(data.dtype.newbyteorder("<"))
    return inference.PayloadBinding(
        reference_id, tuple(int(dim) for dim in data.shape), wire, space, data.tobytes()
    )


def _reachable_conditioning_ids(inference: Any, conditioning: Any) -> set[str]:
    ids: set[str] = set()

    def scan(value: object) -> None:
        if type(value) is inference.PayloadReference:
            ids.add(cast("Any", value).id)
        elif isinstance(value, tuple):
            for item in cast("tuple[object, ...]", value):
                scan(item)
        elif isinstance(value, Mapping):
            for item in cast("Mapping[object, object]", value).values():
                scan(item)

    for record in conditioning.records:
        for _, payload in record.channels:
            ids.add(payload.reference.id)
        if record.mask is not None:
            ids.add(record.mask.payload.id)
        if record.scale_vector is not None:
            ids.add(record.scale_vector.values.reference.id)
        for _, metadata in record.extension_metadata:
            scan(metadata)
    return ids


def _rebound_conditioning_carrier(
    inference: Any, records: Sequence[Any], bindings: Sequence[Any]
) -> Any:
    """Rebuild a carrier, keeping only the bindings the records still reach."""
    conditioning = inference.ConditioningSet(tuple(records))
    reachable = _reachable_conditioning_ids(inference, conditioning)
    kept: dict[str, Any] = {}
    for binding in bindings:
        if binding.reference_id in reachable and binding.reference_id not in kept:
            kept[binding.reference_id] = binding
    return inference.make_conditioning_carrier(conditioning, tuple(kept.values()))


def _combined_conditioning(inference: Any, carriers: Sequence[Any]) -> Any:
    records = tuple(record for carrier in carriers for record in carrier.conditioning.records)
    bindings = tuple(binding for carrier in carriers for binding in carrier.bindings)
    return _rebound_conditioning_carrier(inference, records, bindings)


def _from_side_text(inference: Any, from_carrier: Any) -> tuple[Any, Any]:
    """First from-side record's channels; extra records are ignored at the pin."""
    if not from_carrier.conditioning.records:
        raise ValueError("conditioning_from must contain at least one record")
    channels = dict(from_carrier.conditioning.records[0].channels)
    text = channels.get(inference.ConditioningChannel.TEXT)
    if text is None:
        raise ValueError("conditioning_from must carry a text payload")
    return text, channels.get(inference.ConditioningChannel.POOLED)


def _averaged_conditioning(
    inference: Any, to_carrier: Any, from_carrier: Any, strength: float
) -> Any:
    np_mod = cast("Any", importlib.import_module("numpy"))
    text_from, pooled_from = _from_side_text(inference, from_carrier)
    bindings = {
        binding.reference_id: binding for binding in (*to_carrier.bindings, *from_carrier.bindings)
    }
    cond_from = _float_payload_array(np_mod, bindings[text_from.reference.id], "conditioning_from")
    if cond_from.ndim != 3:
        raise ValueError("conditioning_from text payload must be [batch, tokens, features]")
    pooled_from_array = (
        _float_payload_array(np_mod, bindings[pooled_from.reference.id], "conditioning_from")
        if pooled_from is not None
        else None
    )
    records: list[Any] = []
    new_bindings: list[Any] = []

    def rebound(tag: str, index: int, array: Any, space: str) -> Any:
        reference_id = f"compat-conditioning-average:{index}:{tag}"
        binding = _float_payload_binding(np_mod, inference, reference_id, array, space)
        new_bindings.append(binding)
        return inference.PayloadDescriptor(
            inference.PayloadReference(reference_id), binding.shape, binding.dtype, binding.space
        )

    for index, record in enumerate(to_carrier.conditioning.records):
        channels = dict(record.channels)
        text_to = channels.get(inference.ConditioningChannel.TEXT)
        if text_to is None:
            raise ValueError("conditioning_to must carry a text payload")
        t1 = _float_payload_array(np_mod, bindings[text_to.reference.id], "conditioning_to")
        if t1.ndim != 3:
            raise ValueError("conditioning_to text payload must be [batch, tokens, features]")
        t0 = cond_from[:, : t1.shape[1]]
        if t0.shape[1] < t1.shape[1]:
            # The pin pads with float32 zeros, promoting shorter non-f32
            # from-side payloads exactly as torch.cat does.
            pad = np_mod.zeros((1, t1.shape[1] - t0.shape[1], t1.shape[2]), dtype=np_mod.float32)
            t0 = np_mod.concatenate((t0, pad), axis=1)
        blended = _scalar_scaled(np_mod, t1, strength) + _scalar_scaled(np_mod, t0, 1.0 - strength)
        replaced = {
            inference.ConditioningChannel.TEXT: rebound("text", index, blended, text_to.space)
        }
        pooled_to = channels.get(inference.ConditioningChannel.POOLED)
        if pooled_from_array is not None:
            # A to-record without a pooled payload inherits the from-side one
            # before blending, matching the pin's .get default.
            if pooled_to is not None:
                base = _float_payload_array(
                    np_mod, bindings[pooled_to.reference.id], "conditioning_to"
                )
                space = pooled_to.space
            else:
                base = pooled_from_array
                space = cast("Any", pooled_from).space
            pooled = _scalar_scaled(np_mod, base, strength) + _scalar_scaled(
                np_mod, pooled_from_array, 1.0 - strength
            )
            replaced[inference.ConditioningChannel.POOLED] = rebound("pooled", index, pooled, space)
        new_channels = [
            (channel, replaced.get(channel, payload)) for channel, payload in record.channels
        ]
        if pooled_to is None and inference.ConditioningChannel.POOLED in replaced:
            new_channels.append(
                (
                    inference.ConditioningChannel.POOLED,
                    replaced[inference.ConditioningChannel.POOLED],
                )
            )
        records.append(replace(record, channels=tuple(new_channels)))
    return _rebound_conditioning_carrier(
        inference, records, (*to_carrier.bindings, *from_carrier.bindings, *new_bindings)
    )


def _concatenated_conditioning(inference: Any, to_carrier: Any, from_carrier: Any) -> Any:
    np_mod = cast("Any", importlib.import_module("numpy"))
    text_from, _ = _from_side_text(inference, from_carrier)
    bindings = {
        binding.reference_id: binding for binding in (*to_carrier.bindings, *from_carrier.bindings)
    }
    cond_from = _float_payload_array(np_mod, bindings[text_from.reference.id], "conditioning_from")
    records: list[Any] = []
    new_bindings: list[Any] = []
    for index, record in enumerate(to_carrier.conditioning.records):
        channels = dict(record.channels)
        text_to = channels.get(inference.ConditioningChannel.TEXT)
        if text_to is None:
            raise ValueError("conditioning_to must carry a text payload")
        t1 = _float_payload_array(np_mod, bindings[text_to.reference.id], "conditioning_to")
        if (
            t1.ndim < 2
            or t1.ndim != cond_from.ndim
            or t1.shape[:1] != cond_from.shape[:1]
            or t1.shape[2:] != cond_from.shape[2:]
        ):
            raise ValueError(
                f"conditioning payload shapes {tuple(t1.shape)} and "
                f"{tuple(cond_from.shape)} cannot be joined along the token axis"
            )
        joined = np_mod.concatenate((t1, cond_from), axis=1)
        reference_id = f"compat-conditioning-concat:{index}:text"
        binding = _float_payload_binding(np_mod, inference, reference_id, joined, text_to.space)
        new_bindings.append(binding)
        descriptor = inference.PayloadDescriptor(
            inference.PayloadReference(reference_id), binding.shape, binding.dtype, binding.space
        )
        new_channels = tuple(
            (channel, descriptor if channel is inference.ConditioningChannel.TEXT else payload)
            for channel, payload in record.channels
        )
        records.append(replace(record, channels=new_channels))
    return _rebound_conditioning_carrier(
        inference, records, (*to_carrier.bindings, *from_carrier.bindings, *new_bindings)
    )


def _scaled_conditioning(inference: Any, carrier: Any, multiplier: float) -> Any:
    np_mod = cast("Any", importlib.import_module("numpy"))
    bindings = {binding.reference_id: binding for binding in carrier.bindings}
    scaled_channels = (
        inference.ConditioningChannel.TEXT,
        inference.ConditioningChannel.POOLED,
    )
    records: list[Any] = []
    new_bindings: list[Any] = []
    for index, record in enumerate(carrier.conditioning.records):
        new_channels: list[Any] = []
        for channel, payload in record.channels:
            if channel not in scaled_channels:
                new_channels.append((channel, payload))
                continue
            array = _float_payload_array(np_mod, bindings[payload.reference.id], "conditioning")
            reference_id = f"compat-conditioning-scale:{index}:{channel.value}"
            binding = _float_payload_binding(
                np_mod,
                inference,
                reference_id,
                _scalar_scaled(np_mod, array, multiplier),
                payload.space,
            )
            new_bindings.append(binding)
            new_channels.append(
                (
                    channel,
                    inference.PayloadDescriptor(
                        inference.PayloadReference(reference_id),
                        binding.shape,
                        binding.dtype,
                        binding.space,
                    ),
                )
            )
        records.append(replace(record, channels=tuple(new_channels)))
    return _rebound_conditioning_carrier(inference, records, (*carrier.bindings, *new_bindings))


def _zeroed_conditioning(inference: Any, carrier: Any) -> Any:
    bindings = {binding.reference_id: binding for binding in carrier.bindings}
    zeroed_channels = (
        inference.ConditioningChannel.TEXT,
        inference.ConditioningChannel.POOLED,
    )
    records: list[Any] = []
    new_bindings: list[Any] = []

    def zeroed(payload: Any, index: int, tag: str) -> Any:
        binding = bindings[payload.reference.id]
        reference_id = f"compat-conditioning-zero:{index}:{tag}"
        new_bindings.append(
            inference.PayloadBinding(
                reference_id, binding.shape, binding.dtype, binding.space, bytes(len(binding.data))
            )
        )
        return inference.PayloadDescriptor(
            inference.PayloadReference(reference_id), binding.shape, binding.dtype, binding.space
        )

    for index, record in enumerate(carrier.conditioning.records):
        new_channels = tuple(
            (
                channel,
                zeroed(payload, index, channel.value) if channel in zeroed_channels else payload,
            )
            for channel, payload in record.channels
        )
        scale_vector = record.scale_vector
        if scale_vector is not None:
            # Mirrors the pin zeroing conditioning_scale alongside the prompt.
            scale_vector = replace(scale_vector, values=zeroed(scale_vector.values, index, "scale"))
        records.append(replace(record, channels=new_channels, scale_vector=scale_vector))
    return _rebound_conditioning_carrier(inference, records, (*carrier.bindings, *new_bindings))


class GenerationConditioningMerge(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.conditioning_merge")

    @classmethod
    def execute(cls, **inputs: object) -> Mapping[str, object]:
        inference = importlib.import_module("dinkster_inference")
        mode = inputs.get("mode")
        if mode == "combine":
            carriers = [
                _conditioning_carrier(inputs[key], f"conditioning_{index}")
                for index in range(1, 9)
                if (key := f"mode.inputs.conditioning_{index}") in inputs
            ]
            if len(carriers) < 2:
                raise ValueError("combine requires at least two conditioning inputs")
            return cls.outputs(conditioning=_combined_conditioning(inference, carriers))
        if mode in ("average", "concat"):
            to_carrier = _conditioning_carrier(
                inputs.get("mode.conditioning_to"), "conditioning_to"
            )
            from_carrier = _conditioning_carrier(
                inputs.get("mode.conditioning_from"), "conditioning_from"
            )
            if mode == "concat":
                return cls.outputs(
                    conditioning=_concatenated_conditioning(inference, to_carrier, from_carrier)
                )
            strength = _conditioning_op_float(
                inputs.get("mode.conditioning_to_strength"), "conditioning_to_strength", 0.0, 1.0
            )
            return cls.outputs(
                conditioning=_averaged_conditioning(inference, to_carrier, from_carrier, strength)
            )
        raise ValueError(f"unknown conditioning merge mode: {mode!r}")


class GenerationConditioningScale(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.conditioning_scale")

    @classmethod
    def execute(cls, *, conditioning: object, multiplier: float) -> Mapping[str, object]:
        scale = _conditioning_op_float(multiplier, "multiplier", -100.0, 100.0)
        carrier = _conditioning_carrier(conditioning, "conditioning")
        inference = importlib.import_module("dinkster_inference")
        return cls.outputs(conditioning=_scaled_conditioning(inference, carrier, scale))


class GenerationConditioningSetArea(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.conditioning_set_area")

    @classmethod
    def execute(cls, **inputs: object) -> Mapping[str, object]:
        inference = importlib.import_module("dinkster_inference")
        carrier = _conditioning_carrier(inputs.get("conditioning"), "conditioning")
        strength = _conditioning_op_float(inputs.get("strength"), "strength", 0.0, 10.0)
        units = inputs.get("units")
        if units == "pixels":

            def latent_cells(name: str, low: int) -> int:
                value = inputs.get(f"units.{name}")
                if type(value) is not int:
                    raise TypeError(f"{name} must be an integer")
                if not low <= value <= 16384:
                    raise ValueError(f"{name} must be in [{low}, 16384], got {value}")
                return value // 8

            area = inference.AreaDescriptor(
                height=latent_cells("height", 64),
                width=latent_cells("width", 64),
                y=latent_cells("y", 0),
                x=latent_cells("x", 0),
                units=inference.AreaUnits.LATENT_CELLS,
                strength=strength,
            )
        elif units == "percent":

            def fraction(name: str) -> float:
                return _conditioning_op_float(inputs.get(f"units.{name}"), name, 0.0, 1.0)

            area = inference.AreaDescriptor(
                height=fraction("height"),
                width=fraction("width"),
                y=fraction("y"),
                x=fraction("x"),
                units=inference.AreaUnits.PERCENT,
                strength=strength,
            )
        elif units == "percent-video":

            def fraction(name: str) -> float:
                return _conditioning_op_float(inputs.get(f"units.{name}"), name, 0.0, 1.0)

            area = inference.AreaDescriptor(
                height=fraction("height"),
                width=fraction("width"),
                y=fraction("y"),
                x=fraction("x"),
                units=inference.AreaUnits.PERCENT,
                strength=strength,
                temporal=fraction("temporal"),
                z=fraction("z"),
            )
        else:
            raise ValueError(f"unknown area units: {units!r}")
        records: list[Any] = []
        for record in carrier.conditioning.records:
            mask = record.mask
            # The pin's ConditioningSetArea also clears set_area_to_bounds.
            if mask is not None:
                mask = replace(mask, set_area_to_bounds=False)
            records.append(replace(record, area=area, mask=mask))
        return cls.outputs(
            conditioning=inference.make_conditioning_carrier(
                inference.ConditioningSet(tuple(records)), carrier.bindings
            )
        )


class GenerationConditioningSetMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.conditioning_set_mask")

    @classmethod
    def execute(
        cls, *, conditioning: object, mask: object, strength: float, set_cond_area: str
    ) -> Mapping[str, object]:
        if set_cond_area not in ("default", "mask bounds"):
            raise ValueError(f"unknown set_cond_area: {set_cond_area!r}")
        bounded = _conditioning_op_float(strength, "strength", 0.0, 10.0)
        carrier = _conditioning_carrier(conditioning, "conditioning")
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        torch = _torch()
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a torch.Tensor")
        mask_tensor = cast("Any", mask)
        if mask_tensor.ndim < 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        binding = inference_torch.tensor_to_payload_binding(
            "compat-conditioning-set-mask", mask_tensor, space=inference_torch.MASK_PAYLOAD_SPACE
        )
        descriptor = inference.MaskDescriptor(
            inference.PayloadReference(binding.reference_id),
            bounded,
            set_cond_area == "mask bounds",
        )
        records = tuple(replace(record, mask=descriptor) for record in carrier.conditioning.records)
        return cls.outputs(
            conditioning=_rebound_conditioning_carrier(
                inference, records, (*carrier.bindings, binding)
            )
        )


class GenerationConditioningSetTimestepRange(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.conditioning_set_timestep_range")

    @classmethod
    def execute(cls, *, conditioning: object, start: float, end: float) -> Mapping[str, object]:
        start_percent = _conditioning_op_float(start, "start", 0.0, 1.0)
        end_percent = _conditioning_op_float(end, "end", 0.0, 1.0)
        carrier = _conditioning_carrier(conditioning, "conditioning")
        inference = importlib.import_module("dinkster_inference")
        # An inverted window can never be active, matching the pin's
        # start/end sigma bounds.
        schedule = (
            inference.PercentRange(start_percent, end_percent)
            if start_percent <= end_percent
            else inference.EMPTY_RANGE
        )
        records = tuple(
            replace(record, schedule=schedule) for record in carrier.conditioning.records
        )
        return cls.outputs(
            conditioning=inference.make_conditioning_carrier(
                inference.ConditioningSet(records), carrier.bindings
            )
        )


class GenerationConditioningZeroOut(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.conditioning_zero_out")

    @classmethod
    def execute(cls, *, conditioning: object) -> Mapping[str, object]:
        carrier = _conditioning_carrier(conditioning, "conditioning")
        inference = importlib.import_module("dinkster_inference")
        return cls.outputs(conditioning=_zeroed_conditioning(inference, carrier))


class GenerationChromaRadianceOptions(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.chroma_radiance_options")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        preserve_wrapper: bool,
        start_sigma: float,
        end_sigma: float,
        nerf_tile_size: int,
        force_sequential_txt_ids: bool,
    ) -> Mapping[str, object]:
        if type(preserve_wrapper) is not bool:
            raise TypeError("preserve_wrapper must be a boolean")
        if type(force_sequential_txt_ids) is not bool:
            raise TypeError("force_sequential_txt_ids must be a boolean")
        _check_bounds(
            ("start_sigma", start_sigma, 0.0, 1.0),
            ("end_sigma", end_sigma, 0.0, 1.0),
        )
        if type(nerf_tile_size) is not int or nerf_tile_size < -1:
            raise ValueError(f"nerf_tile_size must be at least -1, got {nerf_tile_size}")
        if nerf_tile_size < 0 and not force_sequential_txt_ids:
            return cls.outputs(model=model)

        (
            handle,
            overlays,
            resolvers,
            control,
            shift,
            transforms,
            context_windows,
            existing_options,
        ) = _native_model(model, "model")
        inference = importlib.import_module("dinkster_inference")
        if handle.recipe.family_id != inference.CHROMA_RADIANCE.id:
            raise ValueError("Chroma Radiance Options requires a Chroma Radiance model")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        window = inference_torch.ChromaRadianceOptionWindow(
            inference_torch.ChromaRadianceOptions(
                None if nerf_tile_size < 0 else nerf_tile_size,
                force_sequential_txt_ids,
            ),
            float(start_sigma),
            float(end_sigma),
        )
        options = (*existing_options, window) if preserve_wrapper else (window,)
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                control,
                shift,
                transforms,
                context_windows,
                options,
                sampling_cache=_native_model_sampling_cache(model),
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=_native_model_sampling_space(model),
            )
        )


class GenerationChromaModelSampling(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.chroma_model_sampling")

    @classmethod
    def execute(cls, *, model: object, shift: float) -> Mapping[str, object]:
        model_value, applications = _application_chain_model(model, "model")
        if applications:
            raise ValueError("Chroma Model Sampling does not accept a model application chain")
        (
            handle,
            overlays,
            resolvers,
            control,
            _,
            transforms,
            context_windows,
            radiance_options,
        ) = _native_model(model_value, "model")
        inference = importlib.import_module("dinkster_inference")
        if handle.recipe.family_id not in (inference.CHROMA.id, inference.CHROMA_RADIANCE.id):
            raise ValueError("Chroma Model Sampling requires a Chroma family model")
        if type(shift) is not float or not math.isfinite(shift) or shift <= 0.0:
            raise ValueError("Chroma Model Sampling shift must be a positive finite float")
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                control,
                shift,
                transforms,
                context_windows,
                radiance_options,
                sampling_cache=_native_model_sampling_cache(model_value),
                sampling_timeline=_native_model_sampling_timeline(model_value),
            )
        )


class GenerationModelSamplingSD3(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.model_sampling_sd3")

    @classmethod
    def execute(cls, *, model: object, shift: object) -> Mapping[str, object]:
        model_value, applications = _application_chain_model(model, "model")
        if applications:
            raise ValueError("Model Sampling SD3 does not accept a model application chain")
        (
            handle,
            overlays,
            resolvers,
            control,
            _,
            transforms,
            context_windows,
            radiance_options,
        ) = _native_model(model_value, "model")
        inference = importlib.import_module("dinkster_inference")
        family = next(
            item for item in inference.builtin_families() if item.id == handle.recipe.family_id
        )
        if not inference.is_flow_parameterization(family.sampling.parameterization):
            raise ValueError("Model Sampling SD3 requires a flow model")
        if isinstance(shift, bool) or not isinstance(shift, (int, float)):
            raise ValueError("Model Sampling SD3 shift must be a positive finite float")
        shift = float(shift)
        if not math.isfinite(shift) or shift <= 0.0:
            raise ValueError("Model Sampling SD3 shift must be a positive finite float")
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                control,
                shift,
                transforms,
                context_windows,
                radiance_options,
                sampling_cache=_native_model_sampling_cache(model_value),
                sampling_timeline=_native_model_sampling_timeline(model_value),
            )
        )


class GenerationModelSamplingLTXV(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.model_sampling_ltxv")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        latent: object | None = None,
    ) -> Mapping[str, object]:
        model_value, applications = _application_chain_model(model, "model")
        if applications:
            raise ValueError("ModelSamplingLTXV does not accept a model application chain")
        handle, overlays, resolvers, control, _, transforms, windows, options = _native_model(
            model_value, "model"
        )
        inference = importlib.import_module("dinkster_inference")
        runtime = getattr(handle.runtime, "component_sampling_runtime", handle.runtime)
        if getattr(runtime.family, "sampling", None) != inference.LTX_SAMPLING:
            raise TypeError("ModelSamplingLTXV requires an LTX sampling runtime")

        tokens = inference.LTXV_SHIFT_TOKENS_HIGH
        if latent is not None:
            if not isinstance(latent, Mapping):
                raise TypeError("latent must be a mapping containing 'samples'")
            samples = cast("Mapping[object, object]", latent).get("samples")
            if type(samples) is inference.MultiStreamLatent:
                streams = cast("Any", samples)
                if "video" not in streams.roles:
                    raise TypeError("latent must contain a video stream")
                samples = streams.by_role("video")
            torch = _torch()
            if not isinstance(samples, torch.Tensor):
                raise TypeError("latent samples must be a torch.Tensor or MultiStreamLatent")
            dimensions = tuple(cast("Sequence[int]", cast("Any", samples).shape)[2:])
            if not dimensions or any(type(size) is not int or size < 1 for size in dimensions):
                raise ValueError("latent samples must have positive token dimensions")
            tokens = math.prod(dimensions)

        shift = inference.ltxv_dynamic_shift(
            tokens,
            max_shift=max_shift,
            base_shift=base_shift,
        )
        if not math.isfinite(shift) or shift <= 0.0:
            raise ValueError("ModelSamplingLTXV computed shift must be positive and finite")
        _runtime_sampling_shift(runtime, shift)
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                control,
                shift,
                transforms,
                windows,
                options,
                sampling_cache=_native_model_sampling_cache(model_value),
                sampling_timeline=_native_model_sampling_timeline(model_value),
            )
        )


class GenerationModelSamplingFlux(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.model_sampling_flux")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        max_shift: float = 1.15,
        base_shift: float = 0.5,
        width: int = 1024,
        height: int = 1024,
    ) -> Mapping[str, object]:
        for name, value in (("max_shift", max_shift), ("base_shift", base_shift)):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be in [0, 100]")
        for name, value in (("width", width), ("height", height)):
            if type(value) is not int or not 16 <= value <= 16384:
                raise ValueError(f"{name} must be an integer in [16, 16384]")
        model_value, applications = _application_chain_model(model, "model")
        if applications:
            raise ValueError("ModelSamplingFlux does not accept a model application chain")
        handle, overlays, resolvers, control, _, transforms, windows, options = _native_model(
            model_value, "model"
        )
        inference = importlib.import_module("dinkster_inference")
        slope = (max_shift - base_shift) / (4096 - 256)
        intercept = base_shift - slope * 256
        shift = (width * height / (8 * 8 * 2 * 2)) * slope + intercept
        space = inference.FluxFlowSigmas(shift=shift)
        _sampling_space_runtime(handle.runtime, space)
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                control,
                None,
                transforms,
                windows,
                options,
                sampling_cache=_native_model_sampling_cache(model_value),
                sampling_timeline=_native_model_sampling_timeline(model_value),
                sampling_space=space,
            )
        )


class GenerationEmptyLatentImage(EmptyLatentImage):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.empty_latent_image")


class GenerationEmptySD3LatentImage(Node):
    MIN_DIMENSION = 16
    MAX_DIMENSION = 16384
    MAX_BATCH = 4096

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.empty_sd3_latent_image")

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int) -> Mapping[str, object]:
        for name, value, low, high in (
            ("width", width, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("height", height, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("batch_size", batch_size, 1, cls.MAX_BATCH),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}], got {value}")
        torch = _torch()
        samples = torch.zeros((batch_size, 16, height // 8, width // 8), device="cpu")
        return cls.outputs(latent={"samples": samples, "downscale_ratio_spacial": 8})


class GenerationEmptyChromaRadianceLatentImage(Node):
    MIN_DIMENSION = 16
    MAX_DIMENSION = 16384
    MAX_BATCH = 4096

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.empty_chroma_radiance_latent_image")

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int) -> Mapping[str, object]:
        for name, value, low, high in (
            ("width", width, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("height", height, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("batch_size", batch_size, 1, cls.MAX_BATCH),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}], got {value}")
        torch = _torch()
        samples = torch.zeros((batch_size, 3, height, width), device="cpu")
        return cls.outputs(latent={"samples": samples})


class GenerationEmptyFlux2LatentImage(Node):
    MIN_DIMENSION = 16
    MAX_DIMENSION = 16384  # ComfyUI's MAX_RESOLUTION
    MAX_BATCH = 4096

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.empty_flux2_latent_image")

    @classmethod
    def execute(cls, *, width: int, height: int, batch_size: int) -> Mapping[str, object]:
        for name, value, low, high in (
            ("width", width, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("height", height, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("batch_size", batch_size, 1, cls.MAX_BATCH),
        ):
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
        torch = _torch()
        # intermediate_device() @ b78cec87 is cpu absent ComfyUI's --gpu-only
        # flag, which Dinkster does not wire.
        samples = torch.zeros((batch_size, 128, height // 16, width // 16), device="cpu")
        return cls.outputs(latent={"samples": samples, "downscale_ratio_spacial": 16})


class GenerationEmptyLTXAVLatent(NativeEmptyLTXAVLatent):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.empty_ltxav_latent")


class GenerationEmptyLTXVLatent(NativeEmptyLTXVLatent):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.empty_ltxv_latent")


class GenerationKSampler(NativeKSampler):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ksampler")

    @classmethod
    @_bind_model_sampling_options
    def execute(
        cls,
        *,
        model: object,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: object,
        negative: object,
        latent_image: object,
        denoise: float,
        segment: SamplingSegment | None = None,
        conditioning_batching: object = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        return NativeKSampler.execute(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            segment=segment,
            conditioning_batching=_conditioning_batching_value(
                conditioning_batching, max_fused_lanes
            ),
        )


class GenerationKSamplerAdvanced(NativeKSamplerAdvanced):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ksampler_advanced")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        add_noise: str,
        noise_seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: object,
        negative: object,
        latent_image: object,
        start_at_step: int,
        end_at_step: int,
        return_with_leftover_noise: str,
        conditioning_batching: object = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        for name, value, low, high in (
            ("noise_seed", noise_seed, 0, cls.MAX_SEED),
            ("steps", steps, 1, cls.MAX_STEPS),
            ("cfg", cfg, 0.0, cls.MAX_CFG),
            ("start_at_step", start_at_step, 0, cls.MAX_STEPS),
            ("end_at_step", end_at_step, 0, cls.MAX_STEPS),
        ):
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
        if add_noise not in ("enable", "disable"):
            raise ValueError("add_noise must be 'enable' or 'disable'")
        if return_with_leftover_noise not in ("disable", "enable"):
            raise ValueError("return_with_leftover_noise must be 'disable' or 'enable'")
        effective_end = min(end_at_step, steps)
        if start_at_step >= effective_end:
            if not isinstance(latent_image, Mapping):
                raise TypeError("latent_image must be a mapping containing 'samples'")
            output = dict(cast("Mapping[object, object]", latent_image))
            output.pop("downscale_ratio_spacial", None)
            output.pop("downscale_ratio_temporal", None)
            return cls.outputs(latent=output)
        inference = importlib.import_module("dinkster_inference")
        segment = inference.SamplingSegment(
            steps=steps,
            start_step=start_at_step,
            end_step=effective_end,
            add_noise=add_noise == "enable",
            return_with_leftover_noise=return_with_leftover_noise == "enable",
        )
        result = GenerationKSampler.execute(
            model=model,
            seed=noise_seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=1.0,
            segment=segment,
            conditioning_batching=conditioning_batching,
            max_fused_lanes=max_fused_lanes,
        )
        return cls.outputs(latent=result["latent"])


def _custom_sampler_value(
    sampler_name: str,
    options: Mapping[str, object] | None = None,
) -> _CustomSamplerValue:
    inference = importlib.import_module("dinkster_inference")
    context = current_execution_context()
    snapshot_digest = None if context is None else context.extension_snapshot_digest
    if context is None:
        registry, extension_ids = inference.builtin_sampler_registry(), ()
    else:
        registry, extension_ids, _behavior_hash = _sampler_registry(inference, snapshot_digest)
    sampler_id = _catalog_id(registry, sampler_name, "sampler")
    descriptor = registry.get(sampler_id)
    if descriptor is None:
        raise ValueError(f"sampler {sampler_id!r} disappeared from its registry snapshot")
    resolved = inference.resolve_options(descriptor.options, options or {})
    return _CustomSamplerValue(
        descriptor,
        tuple((spec.name, resolved[spec.name]) for spec in descriptor.options),
        extension_snapshot_digest=snapshot_digest,
        extension_ids=extension_ids,
    )


def _sde_sampler_id(noise_device: str, cpu_id: str, gpu_id: str) -> str:
    if noise_device == "cpu":
        return cpu_id
    if noise_device == "gpu":
        return gpu_id
    raise ValueError("noise_device must be 'gpu' or 'cpu'")


class GenerationKSamplerSelect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ksampler_select")

    @classmethod
    def execute(cls, *, sampler_name: str) -> Mapping[str, object]:
        return cls.outputs(sampler=_custom_sampler_value(sampler_name))


class GenerationSamplerDPMPP3MSDE(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_dpmpp_3m_sde")

    @classmethod
    def execute(
        cls,
        *,
        eta: float,
        s_noise: float,
        noise_device: str,
    ) -> Mapping[str, object]:
        sampler_id = _sde_sampler_id(
            noise_device,
            "dinkster.dpmpp_3m_sde",
            "dinkster.dpmpp_3m_sde_gpu",
        )
        return cls.outputs(
            sampler=_custom_sampler_value(sampler_id, {"eta": eta, "s_noise": s_noise})
        )


class GenerationSamplerDPMPP2MSDE(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_dpmpp_2m_sde")

    @classmethod
    def execute(
        cls,
        *,
        solver_type: str,
        eta: float,
        s_noise: float,
        noise_device: str,
    ) -> Mapping[str, object]:
        sampler_id = _sde_sampler_id(
            noise_device,
            "dinkster.dpmpp_2m_sde",
            "dinkster.dpmpp_2m_sde_gpu",
        )
        return cls.outputs(
            sampler=_custom_sampler_value(
                sampler_id,
                {"solver_type": solver_type, "eta": eta, "s_noise": s_noise},
            )
        )


class GenerationSamplerDPMPPSDE(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_dpmpp_sde")

    @classmethod
    def execute(
        cls,
        *,
        eta: float,
        s_noise: float,
        r: float,
        noise_device: str,
    ) -> Mapping[str, object]:
        sampler_id = _sde_sampler_id(
            noise_device,
            "dinkster.dpmpp_sde",
            "dinkster.dpmpp_sde_gpu",
        )
        return cls.outputs(
            sampler=_custom_sampler_value(
                sampler_id,
                {"eta": eta, "s_noise": s_noise, "r": r},
            )
        )


class GenerationSamplerDPMPP2SAncestral(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_dpmpp_2s_ancestral")

    @classmethod
    def execute(cls, *, eta: float, s_noise: float) -> Mapping[str, object]:
        return cls.outputs(
            sampler=_custom_sampler_value(
                "dinkster.dpmpp_2s_ancestral",
                {"eta": eta, "s_noise": s_noise},
            )
        )


class GenerationSamplerEulerAncestral(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_euler_ancestral")

    @classmethod
    def execute(cls, *, eta: float, s_noise: float) -> Mapping[str, object]:
        return cls.outputs(
            sampler=_custom_sampler_value(
                "dinkster.euler_ancestral",
                {"eta": eta, "s_noise": s_noise},
            )
        )


class GenerationSamplerEulerAncestralCFGPP(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_euler_ancestral_cfg_pp")

    @classmethod
    def execute(cls, *, eta: float, s_noise: float) -> Mapping[str, object]:
        return cls.outputs(
            sampler=_custom_sampler_value(
                "dinkster.euler_ancestral_cfg_pp",
                {"eta": eta, "s_noise": s_noise},
            )
        )


class GenerationSamplerLMS(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_lms")

    @classmethod
    def execute(cls, *, order: int) -> Mapping[str, object]:
        return cls.outputs(sampler=_custom_sampler_value("dinkster.lms", {"order": order}))


class GenerationSamplerDPMAdaptative(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_dpm_adaptative")

    @classmethod
    def execute(
        cls,
        *,
        order: int,
        rtol: float,
        atol: float,
        h_init: float,
        pcoeff: float,
        icoeff: float,
        dcoeff: float,
        accept_safety: float,
        eta: float,
        s_noise: float,
    ) -> Mapping[str, object]:
        return cls.outputs(
            sampler=_custom_sampler_value(
                "dinkster.dpm_adaptive",
                {
                    "order": order,
                    "rtol": rtol,
                    "atol": atol,
                    "h_init": h_init,
                    "pcoeff": pcoeff,
                    "icoeff": icoeff,
                    "dcoeff": dcoeff,
                    "accept_safety": accept_safety,
                    "eta": eta,
                    "s_noise": s_noise,
                },
            )
        )


class GenerationSamplerERSDE(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_er_sde")

    @classmethod
    def execute(
        cls,
        *,
        solver_type: str,
        max_stage: int,
        eta: float,
        s_noise: float,
    ) -> Mapping[str, object]:
        return cls.outputs(
            sampler=_custom_sampler_value(
                "dinkster.er_sde",
                {
                    "solver_type": solver_type,
                    "max_stage": max_stage,
                    "eta": eta,
                    "s_noise": s_noise,
                },
            )
        )


class GenerationSamplerSEEDS2(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_seeds_2")

    @classmethod
    def execute(
        cls,
        *,
        solver_type: str,
        eta: float,
        s_noise: float,
        r: float,
    ) -> Mapping[str, object]:
        return cls.outputs(
            sampler=_custom_sampler_value(
                "dinkster.seeds_2",
                {"solver_type": solver_type, "eta": eta, "s_noise": s_noise, "r": r},
            )
        )


class GenerationSamplerSASolver(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_sa_solver")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        eta: float,
        sde_start_percent: float,
        sde_end_percent: float,
        s_noise: float,
        predictor_order: int,
        corrector_order: int,
        use_pece: bool,
        simple_order_2: bool,
    ) -> Mapping[str, object]:
        runtime, sampling_shift, _device = _require_base_custom_sampling_runtime(
            model, "SamplerSASolver"
        )
        percent_to_sigma = _bind_sampling_shift(
            runtime.custom_sampling_percent_to_sigma,
            sampling_shift,
        )
        return cls.outputs(
            sampler=_custom_sampler_value(
                "dinkster.configured_sa_solver",
                {
                    "eta": eta,
                    "sde_start_sigma": percent_to_sigma(
                        sde_start_percent, return_actual_sigma=False
                    ),
                    "sde_end_sigma": percent_to_sigma(sde_end_percent, return_actual_sigma=False),
                    "s_noise": s_noise,
                    "predictor_order": predictor_order,
                    "corrector_order": corrector_order,
                    "use_pece": use_pece,
                    "simple_order_2": simple_order_2,
                },
            )
        )


class GenerationBasicScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.basic_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        scheduler: str,
        steps: int,
        denoise: float,
    ) -> Mapping[str, object]:
        if not 1 <= steps <= KSampler.MAX_STEPS:
            raise ValueError(f"steps must be in [1, {KSampler.MAX_STEPS}], got {steps}")
        if not 0.0 <= denoise <= 1.0:
            raise ValueError(f"denoise must be in [0.0, 1.0], got {denoise}")
        runtime, sampling_shift, device = _require_custom_sampling_runtime(model, "BasicScheduler")
        inference = importlib.import_module("dinkster_inference")
        scheduler_id = _catalog_id(
            _inference_registries(inference).schedulers, scheduler, "scheduler"
        )
        build_sigmas = _bind_sampling_shift(runtime.custom_sampling_sigmas, sampling_shift)
        return cls.outputs(
            sigmas=_CustomSigmasValue(build_sigmas(scheduler_id, steps, denoise, device=device))
        )


class GenerationBetaSamplingScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.beta_sampling_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        steps: int,
        alpha: float,
        beta: float,
    ) -> Mapping[str, object]:
        if not 1 <= steps <= KSampler.MAX_STEPS:
            raise ValueError(f"steps must be in [1, {KSampler.MAX_STEPS}], got {steps}")
        runtime, sampling_shift, device = _require_custom_sampling_runtime(
            model, "BetaSamplingScheduler"
        )
        build_sigmas = _bind_sampling_shift(runtime.custom_sampling_beta_sigmas, sampling_shift)
        return cls.outputs(
            sigmas=_CustomSigmasValue(build_sigmas(steps, alpha, beta, device=device))
        )


class GenerationSDTurboScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sd_turbo_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        steps: int,
        denoise: float,
    ) -> Mapping[str, object]:
        if not 1 <= steps <= 10:
            raise ValueError(f"steps must be in [1, 10], got {steps}")
        if not 0.0 <= denoise <= 1.0:
            raise ValueError(f"denoise must be in [0.0, 1.0], got {denoise}")
        runtime, sampling_shift, device = _require_custom_sampling_runtime(
            model, "SDTurboScheduler"
        )
        build_sigmas = _bind_sampling_shift(runtime.custom_sampling_sd_turbo_sigmas, sampling_shift)
        return cls.outputs(sigmas=_CustomSigmasValue(build_sigmas(steps, denoise, device=device)))


def _custom_sigmas_tensor(value: object) -> tuple[Any, Any]:
    if type(value) is not _CustomSigmasValue:
        raise TypeError("sigmas must come from a Dinkster sigma-schedule node")
    torch = _torch()
    return torch, torch.FloatTensor(value.values)


def _custom_sigmas_value(tensor: Any) -> _CustomSigmasValue:
    return _CustomSigmasValue(tuple(float(value) for value in tensor.tolist()))


class GenerationKarrasScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.karras_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        steps: int,
        sigma_max: float,
        sigma_min: float,
        rho: float,
    ) -> Mapping[str, object]:
        torch = _torch()
        ramp = torch.linspace(0, 1, steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return cls.outputs(sigmas=_custom_sigmas_value(torch.cat([sigmas, sigmas.new_zeros([1])])))


class GenerationExponentialScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.exponential_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        steps: int,
        sigma_max: float,
        sigma_min: float,
    ) -> Mapping[str, object]:
        torch = _torch()
        sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), steps).exp()
        return cls.outputs(sigmas=_custom_sigmas_value(torch.cat([sigmas, sigmas.new_zeros([1])])))


class GenerationPolyexponentialScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.polyexponential_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        steps: int,
        sigma_max: float,
        sigma_min: float,
        rho: float,
    ) -> Mapping[str, object]:
        torch = _torch()
        ramp = torch.linspace(1, 0, steps) ** rho
        sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
        return cls.outputs(sigmas=_custom_sigmas_value(torch.cat([sigmas, sigmas.new_zeros([1])])))


class GenerationLaplaceScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.laplace_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        steps: int,
        sigma_max: float,
        sigma_min: float,
        mu: float,
        beta: float,
    ) -> Mapping[str, object]:
        torch = _torch()
        values = torch.linspace(0, 1, steps)
        transformed = mu - beta * torch.sign(0.5 - values) * torch.log(
            1 - 2 * torch.abs(0.5 - values) + 1e-5
        )
        sigmas = torch.clamp(torch.exp(transformed), min=sigma_min, max=sigma_max)
        return cls.outputs(sigmas=_custom_sigmas_value(sigmas))


class GenerationVPScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vp_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        steps: int,
        beta_d: float,
        beta_min: float,
        eps_s: float,
    ) -> Mapping[str, object]:
        torch = _torch()
        values = torch.linspace(1, eps_s, steps)
        sigmas = torch.sqrt(torch.special.expm1(beta_d * values**2 / 2 + beta_min * values))
        return cls.outputs(sigmas=_custom_sigmas_value(torch.cat([sigmas, sigmas.new_zeros([1])])))


class GenerationAlignYourStepsScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.align_your_steps_scheduler")

    @classmethod
    def execute(cls, *, model_type: str, steps: int, denoise: float) -> Mapping[str, object]:
        if model_type not in ALIGN_YOUR_STEPS_NOISE_LEVELS:
            valid = ", ".join(sorted(ALIGN_YOUR_STEPS_NOISE_LEVELS))
            raise ValueError(f"model_type must be one of {valid}, got {model_type!r}")
        if not 1 <= steps <= 10_000:
            raise ValueError(f"steps must be in [1, 10000], got {steps}")
        if not 0.0 <= denoise <= 1.0:
            raise ValueError(f"denoise must be in [0.0, 1.0], got {denoise}")
        return cls.outputs(
            sigmas=_CustomSigmasValue(align_your_steps_sigmas(model_type, steps, denoise))
        )


class GenerationGITSScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.gits_scheduler")

    @classmethod
    def execute(cls, *, coeff: float, steps: int, denoise: float) -> Mapping[str, object]:
        if not 0.80 <= coeff <= 1.50 or round(coeff, 2) not in GITS_NOISE_LEVELS:
            valid = ", ".join(f"{key:.2f}" for key in sorted(GITS_NOISE_LEVELS))
            raise ValueError(f"coeff must round to one of {valid}, got {coeff}")
        if not 2 <= steps <= 1000:
            raise ValueError(f"steps must be in [2, 1000], got {steps}")
        if not 0.0 <= denoise <= 1.0:
            raise ValueError(f"denoise must be in [0.0, 1.0], got {denoise}")
        return cls.outputs(sigmas=_CustomSigmasValue(gits_sigmas(coeff, steps, denoise)))


class GenerationOptimalStepsScheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.optimal_steps_scheduler")

    @classmethod
    def execute(cls, *, model_type: str, steps: int, denoise: float) -> Mapping[str, object]:
        if model_type not in OPTIMAL_STEPS_NOISE_LEVELS:
            valid = ", ".join(sorted(OPTIMAL_STEPS_NOISE_LEVELS))
            raise ValueError(f"model_type must be one of {valid}, got {model_type!r}")
        if not 3 <= steps <= 1000:
            raise ValueError(f"steps must be in [3, 1000], got {steps}")
        if not 0.0 <= denoise <= 1.0:
            raise ValueError(f"denoise must be in [0.0, 1.0], got {denoise}")
        return cls.outputs(
            sigmas=_CustomSigmasValue(optimal_steps_sigmas(model_type, steps, denoise))
        )


class GenerationFlux2Scheduler(Node):
    MIN_DIMENSION = 16
    MAX_DIMENSION = 16384  # ComfyUI's MAX_RESOLUTION
    MAX_STEPS = 4096

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.flux2_scheduler")

    @classmethod
    def execute(cls, *, steps: int, width: int, height: int) -> Mapping[str, object]:
        for name, value, low, high in (
            ("steps", steps, 1, cls.MAX_STEPS),
            ("width", width, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
            ("height", height, cls.MIN_DIMENSION, cls.MAX_DIMENSION),
        ):
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
        inference = importlib.import_module("dinkster_inference")
        torch = _torch()
        mu = inference.flux2_empirical_mu(round(width * height / 256), steps)
        # The reference's generalized time SNR shift at sigma exponent 1.0;
        # the final timestep 0 yields an exact trailing 0.0 sigma.
        timesteps = torch.linspace(1, 0, steps + 1)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1) ** 1.0)
        return cls.outputs(sigmas=_custom_sigmas_value(sigmas))


class GenerationIdeogram4Scheduler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ideogram4_scheduler")

    @classmethod
    def execute(
        cls,
        *,
        steps: int,
        width: int,
        height: int,
        mu: float,
        std: float,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("steps", steps, 1, 200),
            ("width", width, 256, 8192),
            ("height", height, 256, 8192),
            ("mu", mu, -10.0, 10.0),
            ("std", std, 0.1, 5.0),
        )
        sigmas = importlib.import_module("dinkster_inference_torch").ideogram4_sigmas(
            steps, width, height, mu, std
        )
        return cls.outputs(sigmas=_custom_sigmas_value(sigmas))


class GenerationManualSigmas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.manual_sigmas")

    @classmethod
    def execute(cls, *, sigmas: str) -> Mapping[str, object]:
        values = re.findall(r"[-+]?(?:\d*\.*\d+)", sigmas)
        tensor = _torch().FloatTensor([float(value) for value in values])
        return cls.outputs(sigmas=_custom_sigmas_value(tensor))


class GenerationSplitSigmas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.split_sigmas")

    @classmethod
    def execute(cls, *, sigmas: object, step: int) -> Mapping[str, object]:
        _, tensor = _custom_sigmas_tensor(sigmas)
        return cls.outputs(
            high_sigmas=_custom_sigmas_value(tensor[: step + 1]),
            low_sigmas=_custom_sigmas_value(tensor[step:]),
        )


class GenerationSplitSigmasDenoise(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.split_sigmas_denoise")

    @classmethod
    def execute(cls, *, sigmas: object, denoise: float) -> Mapping[str, object]:
        _, tensor = _custom_sigmas_tensor(sigmas)
        steps = max(tensor.shape[-1] - 1, 0)
        total_steps = round(steps * denoise)
        return cls.outputs(
            high_sigmas=_custom_sigmas_value(tensor[:-total_steps]),
            low_sigmas=_custom_sigmas_value(tensor[-(total_steps + 1) :]),
        )


class GenerationFlipSigmas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.flip_sigmas")

    @classmethod
    def execute(cls, *, sigmas: object) -> Mapping[str, object]:
        _, tensor = _custom_sigmas_tensor(sigmas)
        if len(tensor) == 0:
            return cls.outputs(sigmas=_custom_sigmas_value(tensor))
        tensor = tensor.flip(0)
        if tensor[0] == 0:
            tensor[0] = 0.0001
        return cls.outputs(sigmas=_custom_sigmas_value(tensor))


class GenerationSetFirstSigma(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.set_first_sigma")

    @classmethod
    def execute(cls, *, sigmas: object, sigma: float) -> Mapping[str, object]:
        _, tensor = _custom_sigmas_tensor(sigmas)
        tensor[0] = sigma
        return cls.outputs(sigmas=_custom_sigmas_value(tensor))


class GenerationExtendIntermediateSigmas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.extend_intermediate_sigmas")

    @classmethod
    def execute(
        cls,
        *,
        sigmas: object,
        steps: int,
        start_at_sigma: float,
        end_at_sigma: float,
        spacing: str,
    ) -> Mapping[str, object]:
        torch, tensor = _custom_sigmas_tensor(sigmas)
        if start_at_sigma < 0:
            start_at_sigma = float("inf")
        if spacing not in ("linear", "cosine", "sine"):
            raise KeyError(spacing)
        values = torch.linspace(0, 1, steps + 1, device=tensor.device)[1:-1]
        if spacing == "cosine":
            computed_spacing = torch.sin(values * math.pi / 2)
        elif spacing == "sine":
            computed_spacing = 1 - torch.cos(values * math.pi / 2)
        else:
            computed_spacing = values
        extended_sigmas: list[Any] = []
        for index in range(len(tensor) - 1):
            sigma_current = tensor[index]
            sigma_next = tensor[index + 1]
            extended_sigmas.append(sigma_current)
            if end_at_sigma <= sigma_current <= start_at_sigma:
                interpolated = computed_spacing * (sigma_next - sigma_current) + sigma_current
                extended_sigmas.extend(interpolated.tolist())
        if len(tensor) > 0:
            extended_sigmas.append(tensor[-1])
        return cls.outputs(sigmas=_custom_sigmas_value(torch.FloatTensor(extended_sigmas)))


class GenerationSamplingPercentToSigma(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampling_percent_to_sigma")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        sampling_percent: float,
        return_actual_sigma: bool,
    ) -> Mapping[str, object]:
        if not 0.0 <= sampling_percent <= 1.0:
            raise ValueError(f"sampling_percent must be in [0.0, 1.0], got {sampling_percent}")
        runtime, sampling_shift, _device = _require_custom_sampling_runtime(
            model, "SamplingPercentToSigma"
        )
        percent_to_sigma = _bind_sampling_shift(
            runtime.custom_sampling_percent_to_sigma,
            sampling_shift,
        )
        return cls.outputs(
            sigma_value=percent_to_sigma(
                sampling_percent,
                return_actual_sigma=return_actual_sigma,
            )
        )


def _conditioning_batching_value(mode: object, max_fused_lanes: int) -> object:
    if type(mode) is not str:
        raise TypeError("conditioning_batching must be a string")
    if type(max_fused_lanes) is not int or max_fused_lanes < 1:
        raise ValueError("max_fused_lanes must be a positive integer")
    inference = importlib.import_module("dinkster_inference")
    try:
        selected = inference.ConditioningBatchingMode(mode)
    except ValueError:
        raise ValueError(f"unknown conditioning batching mode {mode!r}") from None
    return inference.ConditioningBatching(
        selected,
        max_fused_lanes=(
            max_fused_lanes
            if selected is inference.ConditioningBatchingMode.MAX_FUSED_LANES
            else None
        ),
    )


class GenerationBasicGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.basic_guider")

    @classmethod
    def execute(cls, *, model: object, conditioning: object) -> Mapping[str, object]:
        return cls.outputs(guider=_CustomGuiderValue(model, conditioning, None, 1.0))


class GenerationCFGGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.cfg_guider")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        positive: object,
        negative: object,
        cfg: float,
        conditioning_batching: str = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        if not 0.0 <= cfg <= KSampler.MAX_CFG:
            raise ValueError(f"cfg must be in [0.0, {KSampler.MAX_CFG}], got {cfg}")
        return cls.outputs(
            guider=_CustomGuiderValue(
                model,
                positive,
                negative,
                cfg,
                batching=_conditioning_batching_value(conditioning_batching, max_fused_lanes),
            )
        )


class GenerationDualCFGGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.dual_cfg_guider")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        cond1: object,
        cond2: object,
        negative: object,
        cfg_conds: float,
        cfg_cond2_negative: float,
        style: str,
        conditioning_batching: str = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("cfg_conds", cfg_conds, 0.0, KSampler.MAX_CFG),
            ("cfg_cond2_negative", cfg_cond2_negative, 0.0, KSampler.MAX_CFG),
        )
        if style not in ("regular", "nested"):
            raise ValueError("style must be 'regular' or 'nested'")
        return cls.outputs(
            guider=_DualCFGGuiderValue(
                model,
                cond1,
                cond2,
                negative,
                cfg_conds,
                cfg_cond2_negative,
                style == "nested",
                _conditioning_batching_value(conditioning_batching, max_fused_lanes),
            )
        )


class GenerationDualModelGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.dual_model_guider")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        positive: object,
        cfg: float,
        model_negative: object | None = None,
        negative: object | None = None,
        conditioning_batching: str = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        if not 0.0 <= cfg <= KSampler.MAX_CFG:
            raise ValueError(f"cfg must be in [0.0, {KSampler.MAX_CFG}], got {cfg}")
        return cls.outputs(
            guider=_DualModelGuiderValue(
                model,
                model_negative,
                positive,
                negative,
                cfg,
                _conditioning_batching_value(conditioning_batching, max_fused_lanes),
            )
        )


class GenerationScheduledCFGGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.scheduled_cfg_guider")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        positive: object,
        negative: object,
        sigmas: object,
        from_cfg: float,
        to_cfg: float,
        schedule: str,
        conditioning_batching: str = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        _check_bounds(("from_cfg", from_cfg, 0.0, KSampler.MAX_CFG))
        _check_bounds(("to_cfg", to_cfg, 0.0, KSampler.MAX_CFG))
        if schedule not in ("linear", "log", "exp", "cos"):
            raise ValueError(f"unknown scheduled CFG interpolation {schedule!r}")
        if type(sigmas) is not _CustomSigmasValue:
            raise TypeError("sigmas must come from a Dinkster sigma-schedule node")
        contribution = _guidance_transform_factory(
            "scheduled_cfg", sigmas.values, from_cfg, to_cfg, schedule
        )
        admission_cfg = from_cfg if not math.isclose(from_cfg, 1.0) else to_cfg
        return cls.outputs(
            guider=_CustomGuiderValue(
                model,
                positive,
                negative,
                admission_cfg,
                (("dinkster.scheduled_cfg_guider", contribution),),
                _conditioning_batching_value(conditioning_batching, max_fused_lanes),
            ),
            sigmas=sigmas,
        )


class GenerationLTXVDualCFGGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_dual_cfg_guider")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        positive: object,
        negative: object,
        video_cfg: float,
        audio_cfg: float,
        conditioning_batching: str = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("video_cfg", video_cfg, 0.0, KSampler.MAX_CFG),
            ("audio_cfg", audio_cfg, 0.0, KSampler.MAX_CFG),
        )
        _ltxav_guidance_runtime(model)
        return cls.outputs(
            guider=_LTXAVDualGuiderValue(
                model,
                positive,
                negative,
                video_cfg,
                audio_cfg,
                _conditioning_batching_value(conditioning_batching, max_fused_lanes),
            )
        )


class GenerationPerpNegGuider(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.perp_neg_guider")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        positive: object,
        negative: object,
        empty_conditioning: object,
        cfg: float,
        neg_scale: float,
        conditioning_batching: str = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        if not 0.0 <= cfg <= KSampler.MAX_CFG:
            raise ValueError(f"cfg must be in [0.0, {KSampler.MAX_CFG}], got {cfg}")
        if not 0.0 <= neg_scale <= 100.0:
            raise ValueError(f"neg_scale must be in [0.0, 100.0], got {neg_scale}")
        return cls.outputs(
            guider=_PerpNegGuiderValue(
                model,
                positive,
                negative,
                empty_conditioning,
                cfg,
                neg_scale,
                _conditioning_batching_value(conditioning_batching, max_fused_lanes),
            )
        )


class GenerationDisableCFG1Optimization(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.disable_cfg1_optimization")

    @classmethod
    def execute(cls, *, model: object) -> Mapping[str, object]:
        original_model = model
        model, applications = _application_chain_model(model, "model")
        handle, overlays, resolvers, control, shift, transforms, windows, chroma_options = (
            _native_model(model, "model")
        )
        if any(contribution is _DISABLE_CFG1_OPTIMIZATION for _, contribution in transforms):
            return cls.outputs(model=original_model)
        disabled = _NativeModelOverlay(
            handle,
            overlays,
            resolvers,
            control,
            shift,
            (*transforms, ("dinkster.disable_cfg1_optimization", _DISABLE_CFG1_OPTIMIZATION)),
            windows,
            chroma_options,
            sampling_cache=_native_model_sampling_cache(model),
            sampling_timeline=_native_model_sampling_timeline(model),
            sampling_space=_native_model_sampling_space(model),
        )
        if applications:
            inference = importlib.import_module("dinkster_inference")
            disabled = inference.ApplicationChain(disabled, applications)
        return cls.outputs(model=disabled)


class GenerationDisableNoise(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.disable_noise")

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(noise=_CustomNoiseValue(None))


class GenerationRandomNoise(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.random_noise")

    @classmethod
    def execute(cls, *, noise_seed: int) -> Mapping[str, object]:
        if not 0 <= noise_seed <= KSampler.MAX_SEED:
            raise ValueError(f"noise_seed must be in [0, {KSampler.MAX_SEED}], got {noise_seed}")
        return cls.outputs(noise=_CustomNoiseValue(noise_seed))


class GenerationAddNoise(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.add_noise")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        noise: object,
        sigmas: object,
        latent_image: object,
    ) -> Mapping[str, object]:
        if type(noise) is not _CustomNoiseValue:
            raise TypeError("noise must come from RandomNoise or DisableNoise")
        if type(sigmas) is not _CustomSigmasValue:
            raise TypeError("sigmas must come from a Dinkster sigma-schedule node")
        typed_noise = noise
        typed_sigmas = sigmas
        if not isinstance(latent_image, Mapping):
            raise TypeError("latent_image must be a mapping containing 'samples'")
        sigma_values = tuple(typed_sigmas.values)
        if not sigma_values:
            return cls.outputs(latent=cast("Mapping[object, object]", latent_image))
        latent = cast("Mapping[object, object]", latent_image)
        torch = _torch()
        samples = latent.get("samples")
        if type(samples) is not torch.Tensor:
            raise TypeError("AddNoise requires latent_image['samples'] to be an exact torch.Tensor")
        samples = cast("Any", samples)
        inference_torch = importlib.import_module("dinkster_inference_torch")
        seed = 0 if typed_noise.seed is None else typed_noise.seed
        generated = (
            torch.zeros_like(samples, device="cpu")
            if typed_noise.seed is None
            else inference_torch.prepare_noise(samples, seed, _batch_index_noise_inds(latent))
        )
        runtime, sampling_shift, _device = _require_base_custom_sampling_runtime(model, "AddNoise")
        custom_sampling_add_noise = getattr(runtime, "custom_sampling_add_noise", None)
        if not callable(custom_sampling_add_noise):
            raise TypeError("model runtime does not support AddNoise")
        add_noise = _bind_sampling_shift(custom_sampling_add_noise, sampling_shift)
        first_sigma = sigma_values[0]
        scale = abs(first_sigma - sigma_values[-1]) if len(sigma_values) > 1 else first_sigma
        output = dict(latent)
        output["samples"] = add_noise(samples, generated, scale)
        return cls.outputs(latent=output)


def _custom_sampling_conditioning(
    value: object,
    input_id: str,
    handle: NativeRuntimeHandle,
    inference: Any,
    torch: Any,
) -> tuple[Any, Any]:
    runtime = handle.runtime
    prepared = (
        _prepare_provider_multistream_conditioning(value, input_id, runtime, inference)
        if isinstance(runtime, inference.MultiStreamConditioningRuntime)
        else (
            _prepare_provider_conditioning(value, input_id, runtime, inference)
            if isinstance(runtime, inference.ConditioningRuntime)
            else _materialize_provider_conditioning(value, input_id, handle)
        )
    )
    return _conditioning(prepared, input_id, torch, inference)


def _custom_sampling_has_inpaint(value: object, input_id: str, inference: Any) -> bool:
    if isinstance(value, inference.ConditioningCarrier):
        return False
    return any(
        "concat_mask" in cast("Mapping[object, object]", entry[1])
        or "concat_latent_image" in cast("Mapping[object, object]", entry[1])
        for entry in _condition_entries(value, input_id)
    )


def _execute_generation_custom_sampling(
    *,
    model: object,
    model_negative: object | None = None,
    image_only_negative: bool = False,
    noise: _CustomNoiseValue,
    sampler: object,
    sigmas: _CustomSigmasValue,
    positive: object,
    negative: object | None,
    cfg: float,
    latent_image: object,
    middle: object | None = None,
    middle_scale: float = 1.0,
    nested_guidance: bool = False,
    audio_cfg: float | None = None,
    empty: object | None = None,
    neg_scale: float = 1.0,
    guider_transforms: tuple[tuple[str, object], ...] = (),
    conditioning_batching: object | None = None,
) -> tuple[dict[object, object], dict[object, object]]:
    if type(noise) is not _CustomNoiseValue:
        raise TypeError("noise must come from RandomNoise or DisableNoise")
    inference = importlib.import_module("dinkster_inference")
    if conditioning_batching is None:
        conditioning_batching = inference.ConditioningBatching()
    elif type(conditioning_batching) is not inference.ConditioningBatching:
        raise TypeError("conditioning_batching must be an exact ConditioningBatching")
    if type(sampler) is SamplerSelection:
        context = current_execution_context()
        snapshot_digest = None if context is None else context.extension_snapshot_digest
        if sampler.extension_snapshot_digest != snapshot_digest:
            raise ValueError("sampler extension snapshot changed after KSamplerSelect executed")
        resolved = _custom_sampler_value(sampler.sampler_id, dict(sampler.options))
        if resolved.extension_ids != sampler.extension_ids:
            raise ValueError("sampler extension identities changed after KSamplerSelect executed")
        sampler = resolved
    elif type(sampler) is inference.BuiltinSamplerSelection:
        selection = cast("Any", sampler)
        sampler = _custom_sampler_value(
            selection.sampler_id,
            dict(selection.options),
        )
    if type(sampler) is not _CustomSamplerValue:
        raise TypeError("sampler must come from a Dinkster sampler-selection node")
    if type(sigmas) is not _CustomSigmasValue:
        raise TypeError("sigmas must come from a Dinkster sigma-schedule node")
    classic_control_binding = (
        _conditioning_classic_control(positive, "conditioning")
        if negative is None
        else _select_classic_control_binding(positive, negative)
    )
    if empty is not None:
        empty_control_binding = _conditioning_classic_control(empty, "empty_conditioning")
        if classic_control_binding is None:
            if empty_control_binding is not None:
                raise ValueError(
                    "perp-neg classic ControlNet cannot be attached only to empty conditioning"
                )
        elif classic_control_binding.apply_to_uncond:
            if empty_control_binding not in (None, classic_control_binding):
                raise ValueError(
                    "perp-neg empty classic ControlNet chain must match the positive control"
                )
        elif empty_control_binding != classic_control_binding:
            raise ValueError(
                "perp-neg classic ControlNet must cover empty conditioning or use apply-to-uncond"
            )
    if middle is not None:
        middle_control_binding = _conditioning_classic_control(middle, "cond2")
        if classic_control_binding is None:
            if middle_control_binding is not None:
                raise ValueError("dual CFG classic ControlNet cannot be attached only to cond2")
        elif classic_control_binding.apply_to_uncond:
            if middle_control_binding not in (None, classic_control_binding):
                raise ValueError("dual CFG cond2 classic ControlNet chain must match cond1")
        elif middle_control_binding != classic_control_binding:
            raise ValueError("dual CFG classic ControlNet must cover cond2 or use apply-to-uncond")
    if (controlled := _controlled_conditioning(positive)) is not None:
        positive = controlled.conditioning
    if (controlled := _controlled_conditioning(negative)) is not None:
        negative = controlled.conditioning
    if (controlled := _controlled_conditioning(empty)) is not None:
        empty = controlled.conditioning
    if (controlled := _controlled_conditioning(middle)) is not None:
        middle = controlled.conditioning
    model, applications = _application_chain_model(model, "model")
    (
        handle,
        overlays,
        _,
        z_image_control,
        sampling_shift,
        model_guidance_transforms,
        context_windows,
        chroma_radiance_options,
    ) = _native_model(model, "model")
    model_guidance_transforms, disable_cfg1_optimization = _cfg1_optimization_setting(
        model_guidance_transforms
    )
    guidance_transforms = (*model_guidance_transforms, *guider_transforms)
    if overlays:
        raise ValueError(
            "custom sampling does not accept model overlays other than guidance transforms"
        )
    negative_handle = None
    if model_negative is not None:
        negative_model, negative_applications = _application_chain_model(
            model_negative, "model_negative"
        )
        if negative_applications:
            raise ValueError("dual-model guidance does not accept negative model applications")
        (
            negative_handle,
            negative_overlays,
            negative_resolvers,
            negative_control,
            negative_sampling_shift,
            negative_transforms,
            negative_context_windows,
            negative_radiance_options,
        ) = _native_model(negative_model, "model_negative")
        if (
            negative_overlays
            or negative_resolvers
            or negative_control is not None
            or negative_sampling_shift is not None
            or _native_model_sampling_space(negative_model) is not None
            or negative_transforms
            or negative_context_windows is not None
            or negative_radiance_options
        ):
            raise ValueError("dual-model guidance does not accept a negative model overlay")
    positive, positive_guidance = _split_flux_guidance(positive)
    if negative is not None:
        negative, negative_guidance = _split_flux_guidance(negative)
    else:
        negative_guidance = None
    guidance = _effective_flux_guidance(positive_guidance, negative_guidance)
    if middle is not None:
        middle, middle_guidance = _split_flux_guidance(middle)
        guidance = _effective_flux_guidance(guidance, middle_guidance)
    if empty is not None:
        if negative is None:
            raise ValueError("perp-neg guidance requires negative conditioning")
        empty, empty_guidance = _split_flux_guidance(empty)
        if empty_guidance is not None:
            raise ValueError("perp-neg empty conditioning does not accept FluxGuidance")
    runtime, positive, negative, component_execution, prepared_rows = _resolve_sampling_model(
        handle,
        positive,
        negative,
        inference,
        sampling_shift=sampling_shift,
        option_windows=chroma_radiance_options,
        negative_handle=negative_handle,
        image_only_negative=image_only_negative,
    )
    if prepared_rows and negative == []:
        negative = None
    if empty is not None and negative is None:
        raise ValueError("perp-neg guidance requires negative conditioning")
    if empty is not None and prepared_rows:
        empty_runtime, empty, _, _, _ = _resolve_sampling_model(
            handle,
            empty,
            None,
            inference,
            sampling_shift=sampling_shift,
            option_windows=chroma_radiance_options,
        )
        if empty_runtime.runtime_identity != runtime.runtime_identity:
            raise ValueError("empty conditioning must use the same component binding")
    if middle is not None and prepared_rows:
        middle_runtime, middle, _, _, _ = _resolve_sampling_model(
            handle,
            middle,
            None,
            inference,
            sampling_shift=sampling_shift,
            option_windows=chroma_radiance_options,
        )
        if middle_runtime.runtime_identity != runtime.runtime_identity:
            raise ValueError("cond2 must use the same component binding")
    runtime = _sampling_space_runtime(runtime, _native_model_sampling_space(model))
    if not isinstance(runtime, inference.CustomSamplingRuntime):
        raise TypeError(f"model family {runtime.family.id!r} does not support custom sampling")
    if audio_cfg is not None and getattr(runtime, "supports_audio_cfg", None) is not True:
        raise TypeError("LTXV Dual CFG Guider requires LTX-2 audio-video latent sampling")
    runtime_sampling_shift = _runtime_sampling_shift(runtime, sampling_shift)
    bound_sample_custom = _bind_sampling_shift(runtime.sample_custom, runtime_sampling_shift)
    positive_is_carrier = isinstance(positive, inference.ConditioningCarrier)
    negative_is_carrier = negative is not None and isinstance(
        negative, inference.ConditioningCarrier
    )
    if negative is not None and positive_is_carrier != negative_is_carrier:
        raise TypeError("positive and negative must use the same conditioning representation")
    empty_is_carrier = empty is not None and isinstance(empty, inference.ConditioningCarrier)
    if empty is not None and positive_is_carrier != empty_is_carrier:
        raise TypeError("positive and empty must use the same conditioning representation")
    middle_is_carrier = middle is not None and isinstance(middle, inference.ConditioningCarrier)
    if middle is not None and positive_is_carrier != middle_is_carrier:
        raise TypeError("cond1 and cond2 must use the same conditioning representation")
    if classic_control_binding is not None and z_image_control is not None:
        raise ValueError("classic and Z-Image ControlNet cannot be applied together")
    if classic_control_binding is not None:
        _require_classic_control_keyword(bound_sample_custom)
    context = current_execution_context()
    snapshot_digest = None if context is None else context.extension_snapshot_digest
    if sampler.extension_snapshot_digest != snapshot_digest:
        raise ValueError("sampler extension snapshot changed after KSamplerSelect executed")
    request = inference.CustomSamplingRequest(
        sampler.descriptor,
        cast("tuple[tuple[str, Any], ...]", sampler.options),
        sigmas.values,
        cache=cast("Any", _native_model_sampling_cache(model)),
        timeline=cast("Any", _native_model_sampling_timeline(model)),
    )
    if not isinstance(latent_image, Mapping):
        raise TypeError("latent_image must be a mapping containing 'samples'")
    latent = cast("Mapping[object, object]", latent_image)
    inference_torch = importlib.import_module("dinkster_inference_torch")
    prepare_latent_kwargs = getattr(runtime, "custom_sampling_latent_kwargs", None)
    latent_kwargs: dict[str, object] = (
        {} if prepare_latent_kwargs is None else prepare_latent_kwargs(latent)
    )
    runtime_output_role = getattr(runtime, "dense_custom_sampling_role", None)
    dense_output_role = (
        runtime_output_role
        if type(latent.get("samples")) is not inference.MultiStreamLatent
        else None
    )
    noise_mask = latent.get("noise_mask")
    has_inpaint = (
        _custom_sampling_has_inpaint(positive, "positive", inference)
        or (negative is not None and _custom_sampling_has_inpaint(negative, "negative", inference))
        or (
            empty is not None
            and _custom_sampling_has_inpaint(empty, "empty_conditioning", inference)
        )
    )
    runtime.check_custom_sampling(
        request,
        has_denoise_mask=noise_mask is not None,
        has_inpaint=has_inpaint,
        has_context_windows=context_windows is not None,
        guidance=guidance,
    )
    torch = _torch()
    # Single-stream family descriptors may still require structural execution;
    # the runtime-owned adapter declares that boundary for plain workflow latents.
    multistream_family = (
        type(runtime.family.latent) is inference.MultiStreamLatentDescriptor
        or isinstance(runtime, inference.MultiStreamLatentAdapterRuntime)
        or (
            type(latent.get("samples")) is inference.MultiStreamLatent
            and isinstance(runtime, inference.MultiStreamFamilyRuntime)
        )
    )
    sparse_family = type(latent.get("samples")) is inference.SparseLatent
    if multistream_family:
        latent = _adapt_multistream_latent(latent, runtime, torch, inference, "latent_image")
    samples = latent.get("samples")
    if multistream_family:
        if type(samples) is not inference.MultiStreamLatent:
            raise TypeError("latent_image['samples'] must be an exact MultiStreamLatent")
        samples = cast("Any", samples)
    elif sparse_family:
        if type(samples) is not inference.SparseLatent:
            raise TypeError("latent_image['samples'] must be an exact SparseLatent")
        samples = cast("Any", samples)
    else:
        if type(samples) is not torch.Tensor:
            raise TypeError("latent_image['samples'] must be an exact torch.Tensor")
        samples = cast("Any", samples)
        samples = _normalize_empty_latent(samples, runtime, torch)
        descriptor = runtime.family.single_stream_latent()
        expected_rank = descriptor.dimensions + 2
        if samples.ndim != expected_rank:
            raise ValueError(f"custom sampling requires a rank-{expected_rank} latent")
        expected_channels = descriptor.channels
        if samples.shape[1] != expected_channels:
            if bool(torch.count_nonzero(samples)):
                raise ValueError(
                    "custom sampling latent channels must match the model unless the latent "
                    "is empty"
                )
            samples = torch.zeros(
                (samples.shape[0], expected_channels, *samples.shape[2:]),
                dtype=samples.dtype,
                layout=samples.layout,
                device=samples.device,
            )
    noise_inds = _batch_index_noise_inds(latent)
    if noise_mask is not None and type(noise_mask) is not torch.Tensor:
        if not (multistream_family and type(noise_mask) is inference.MultiStreamLatent):
            raise TypeError("latent_image['noise_mask'] must be an exact torch.Tensor")
    empty_cond = None
    empty_inpaint = None
    middle_cond = None
    middle_inpaint = None
    if multistream_family:
        if positive_is_carrier:
            cond = _prepare_provider_multistream_conditioning(
                positive, "positive", runtime, inference
            )[0][0]
        else:
            cond = _prepared_multistream_carrier(positive, inference, "positive")
            if cond is None:
                raise TypeError("positive must contain prepared multi-stream conditioning")
        cond_inpaint = None
        # The carrier resolves [] to None: an empty negative means no
        # guidance lane, exactly as on the KSampler path. Malformed
        # entries raise inside the carrier helper.
        if negative is None:
            uncond = None
        elif negative_is_carrier:
            uncond = _prepare_provider_multistream_conditioning(
                negative, "negative", runtime, inference
            )[0][0]
        else:
            uncond = _prepared_multistream_carrier(negative, inference, "negative")
        uncond_inpaint = None
        if empty is not None:
            if isinstance(empty, inference.ConditioningCarrier):
                empty_cond = _prepare_provider_multistream_conditioning(
                    empty, "empty_conditioning", runtime, inference
                )[0][0]
            else:
                empty_cond = _prepared_multistream_carrier(empty, inference, "empty_conditioning")
                if empty_cond is None:
                    raise TypeError("empty_conditioning must contain multi-stream conditioning")
        if middle is not None:
            if isinstance(middle, inference.ConditioningCarrier):
                middle_cond = _prepare_provider_multistream_conditioning(
                    middle, "cond2", runtime, inference
                )[0][0]
            else:
                middle_cond = _prepared_multistream_carrier(middle, inference, "cond2")
                if middle_cond is None:
                    raise TypeError("cond2 must contain multi-stream conditioning")
    else:
        if prepared_rows:
            cond, cond_inpaint = _conditioning(positive, "positive", torch, inference)
        else:
            cond, cond_inpaint = _custom_sampling_conditioning(
                positive, "positive", handle, inference, torch
            )
        if negative is None:
            uncond = None
            uncond_inpaint = None
        elif prepared_rows:
            uncond, uncond_inpaint = _conditioning(negative, "negative", torch, inference)
        else:
            uncond, uncond_inpaint = _custom_sampling_conditioning(
                negative, "negative", handle, inference, torch
            )
        if empty is not None:
            if prepared_rows:
                empty_cond, empty_inpaint = _conditioning(
                    empty, "empty_conditioning", torch, inference
                )
            else:
                empty_cond, empty_inpaint = _custom_sampling_conditioning(
                    empty, "empty_conditioning", handle, inference, torch
                )
        if middle is not None:
            if prepared_rows:
                middle_cond, middle_inpaint = _conditioning(middle, "cond2", torch, inference)
            else:
                middle_cond, middle_inpaint = _custom_sampling_conditioning(
                    middle, "cond2", handle, inference, torch
                )
    if negative is not None and (cond_inpaint is None) != (uncond_inpaint is None):
        raise ValueError("positive and negative inpaint conditioning must both be present")
    if (
        cond_inpaint is not None
        and uncond_inpaint is not None
        and (
            cond_inpaint.mask is not uncond_inpaint.mask
            or cond_inpaint.masked_image is not uncond_inpaint.masked_image
        )
    ):
        raise ValueError("positive and negative inpaint conditioning must share concat values")
    if empty is not None and (cond_inpaint is None) != (empty_inpaint is None):
        raise ValueError("positive and empty inpaint conditioning must both be present")
    if (
        cond_inpaint is not None
        and empty_inpaint is not None
        and (
            cond_inpaint.mask is not empty_inpaint.mask
            or cond_inpaint.masked_image is not empty_inpaint.masked_image
        )
    ):
        raise ValueError("positive and empty inpaint conditioning must share concat values")
    if middle is not None and (cond_inpaint is None) != (middle_inpaint is None):
        raise ValueError("cond1 and cond2 inpaint conditioning must both be present")
    if (
        cond_inpaint is not None
        and middle_inpaint is not None
        and (
            cond_inpaint.mask is not middle_inpaint.mask
            or cond_inpaint.masked_image is not middle_inpaint.masked_image
        )
    ):
        raise ValueError("cond1 and cond2 inpaint conditioning must share concat values")
    runtime.check_custom_sampling(
        request,
        has_denoise_mask=noise_mask is not None,
        has_inpaint=cond_inpaint is not None,
        has_context_windows=context_windows is not None,
        guidance=guidance,
    )
    control_kwargs: dict[str, object] = {}
    if z_image_control is not None:
        control_kwargs["control"] = _materialize_z_image_control(
            handle,
            z_image_control,
            samples,
            torch,
            inference,
            inference_torch,
        )
    seed = 0 if noise.seed is None else noise.seed
    if multistream_family:

        def _zero_stream(stream: Any) -> Any:
            return torch.zeros_like(stream, device="cpu")

        if noise.seed is None:
            generated_noise = samples.map(_zero_stream)
        else:
            prepare_stream_noise = getattr(
                runtime, "prepare_custom_sampling_noise", inference_torch.prepare_multistream_noise
            )
            generated_noise = prepare_stream_noise(samples, seed, noise_inds)
        samples = _move_multistream_latent(samples, handle.load_device)
        if type(noise_mask) is torch.Tensor:
            noise_mask = cast("Any", noise_mask).to(handle.load_device)
        elif type(noise_mask) is inference.MultiStreamLatent:
            noise_mask = _move_multistream_latent(noise_mask, handle.load_device)
        preview = (
            sampling_preview_emitter(handle, stream_role=runtime_output_role)
            if runtime_output_role is not None
            else multistream_sampling_preview_emitter(handle)
        )
    elif sparse_family:
        support, features = inference_torch.unpack_sparse_latent(samples)
        if noise.seed is None:
            noise_features = torch.zeros_like(features, device="cpu")
        elif noise_inds is None:
            noise_features = inference_torch.prepare_noise(features, seed)
        else:
            max_points = max(support.batch_counts)
            noise_batches = inference_torch.prepare_noise(
                features.new_empty((support.batch_size, max_points, features.shape[1])),
                seed,
                noise_inds,
            )
            noise_features = torch.cat(
                tuple(
                    noise_batches[batch, :count] for batch, count in enumerate(support.batch_counts)
                )
            )
        generated_noise = inference_torch.pack_sparse_latent(support, noise_features)
        preview = None
    else:
        generated_noise = (
            torch.zeros(
                samples.shape,
                dtype=samples.dtype,
                layout=samples.layout,
                device="cpu",
            )
            if noise.seed is None
            else inference_torch.prepare_noise(samples, seed, noise_inds)
        )
        preview = sampling_preview_emitter(handle)

    sampling_cfg = max(cfg, middle_scale) if middle_cond is not None else cfg
    if audio_cfg is not None:
        if type(samples) is not inference.MultiStreamLatent:
            raise TypeError("LTXV Dual CFG Guider requires LTX-2 audio-video latent sampling")
        if samples.roles != ("video", "audio"):
            raise TypeError("LTXV Dual CFG Guider requires video and audio latent streams")
        sampling_cfg = max(cfg, audio_cfg)
        if not math.isclose(cfg, audio_cfg):
            video = samples.by_role("video")
            video_elements = math.prod(video.shape[1:])
            guidance_transforms = (
                *guidance_transforms,
                (
                    "dinkster.ltxv_dual_cfg_guider",
                    inference_torch.ltxav_dual_cfg_guidance(
                        float(cfg), float(audio_cfg), video_elements
                    ),
                ),
            )

    def report_step(event: Any) -> None:
        report_progress(event.step + 1, event.total)

    if empty_cond is not None and uncond is None:
        raise ValueError("perp-neg guidance requires negative conditioning")
    if getattr(runtime, "video_vae_config", None) == inference.LTXAV_22B_V25_VAE_CONFIG:
        inference_torch.soft_empty_cache(handle.load_device)
    sampling_memory = _sampling_memory_requirements(runtime, samples)
    stage_negative = negative_handle is not None and (
        not math.isclose(sampling_cfg, 1.0)
        or cast("Any", sampler.descriptor).needs_uncond
        or bool(guidance_transforms)
        or disable_cfg1_optimization
    )
    with (
        _classic_control_context(
            classic_control_binding,
            latent_batch=int(samples.shape[0]) if classic_control_binding is not None else 1,
            torch=torch,
            inference=inference,
            inference_torch=inference_torch,
            base_handle=handle,
        ) as (classic_control, classic_control_handles),
        handle.stage(
            "diffusion",
            memory_required=sampling_memory[0],
            minimum_memory=sampling_memory[1],
            unload_before=(() if component_execution else ("text",)),
        ),
        (
            negative_handle.stage("diffusion")
            if stage_negative and negative_handle is not None
            else nullcontext()
        ),
        ExitStack() as control_stages,
        (
            z_image_control.handle.stage(observer_stage="sample")
            if z_image_control is not None
            else nullcontext()
        ),
        preview_stage(preview),
        torch.inference_mode(),
        inference.use_sampling_environment(
            sampler.extension_ids, context.cancelled if context is not None else _not_cancelled
        ),
        _materialized_application_kwargs(
            applications,
            handle,
            samples,
            reserved_keys={
                "noise",
                "cond",
                "cfg",
                "request",
                "seed",
                "guidance",
                "denoise_mask",
                "inpaint",
                "on_step",
                "on_state",
                "capture_denoised",
                "compute_dtype",
                "sampling_shift",
                "cancelled",
                "observer",
                "parent_span_id",
            },
        ) as application_kwargs,
    ):
        if classic_control is not None:
            control_kwargs["control"] = classic_control
        for control_handle in classic_control_handles:
            control_stages.enter_context(control_handle.stage(observer_stage="sample"))
        result = bound_sample_custom(
            samples,
            noise=generated_noise,
            cond=cond,
            # Every KSampler surface wraps SamplingGuidance unconditionally,
            # and CFG++ samplers consume the scale even without an uncond
            # payload, so the wrapper survives an absent negative here too.
            cfg=(
                inference.PerpNegSamplingGuidance(
                    uncond,
                    empty_cond,
                    sampling_cfg,
                    neg_scale,
                    transforms=guidance_transforms,
                    batching=conditioning_batching,
                    disable_cfg1_optimization=disable_cfg1_optimization,
                )
                if empty_cond is not None
                else (
                    inference.DualSamplingGuidance(
                        middle_cond,
                        uncond,
                        cfg,
                        middle_scale,
                        nested_guidance,
                        transforms=guidance_transforms,
                        batching=conditioning_batching,
                        disable_cfg1_optimization=disable_cfg1_optimization,
                    )
                    if middle_cond is not None
                    else inference.SamplingGuidance(
                        uncond,
                        sampling_cfg,
                        transforms=guidance_transforms,
                        batching=conditioning_batching,
                        disable_cfg1_optimization=disable_cfg1_optimization,
                    )
                )
            ),
            request=request,
            seed=seed,
            guidance=guidance,
            denoise_mask=cast("Any", noise_mask),
            inpaint=cond_inpaint,
            context_windows=context_windows,
            on_step=report_step,
            on_state=(preview.on_state if preview is not None else None),
            **latent_kwargs,
            **control_kwargs,
            **application_kwargs,
        )
        if type(result) is not inference.CustomSamplingResult:
            raise TypeError("custom sampling runtime must return an exact CustomSamplingResult")
        if multistream_family:
            if (
                type(result.output) is not inference.MultiStreamLatent
                or result.output.roles != samples.roles
            ):
                raise TypeError("custom sampling must return the input latent streams")
            if result.denoised_output is not None and (
                type(result.denoised_output) is not inference.MultiStreamLatent
                or result.denoised_output.roles != samples.roles
            ):
                raise TypeError(
                    "custom sampling denoised output must match the input latent streams"
                )
        elif sparse_family:
            if type(result.output) is not inference.SparseLatent:
                raise TypeError("custom sampling must return an exact SparseLatent")
            if not result.output.support.same_support(samples.support):
                raise TypeError("custom sampling must preserve sparse support")
            if result.denoised_output is not None and (
                type(result.denoised_output) is not inference.SparseLatent
                or not result.denoised_output.support.same_support(samples.support)
            ):
                raise TypeError("custom sampling denoised output must preserve sparse support")

        def move_output(value: Any) -> Any:
            if multistream_family:
                if dense_output_role is not None:
                    return value.by_role(dense_output_role).to("cpu")
                return _move_multistream_latent(value, "cpu")
            if sparse_family:
                return value
            return value.to("cpu")

        result = inference.CustomSamplingResult(
            move_output(result.output),
            None if result.denoised_output is None else move_output(result.denoised_output),
        )
    output = dict(latent)
    if dense_output_role is None:
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
    output["samples"] = result.output
    if result.denoised_output is None:
        denoised_output = dict(output)
    else:
        denoised_output = dict(latent)
        denoised_output["samples"] = result.denoised_output
    return output, denoised_output


class GenerationSamplerCustom(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_custom")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        add_noise: bool,
        noise_seed: int,
        cfg: float,
        positive: object,
        negative: object,
        sampler: object,
        sigmas: object,
        latent_image: object,
        conditioning_batching: str = "auto",
        max_fused_lanes: int = 2,
    ) -> Mapping[str, object]:
        if type(add_noise) is not bool:
            raise TypeError("add_noise must be a Boolean")
        if not 0 <= noise_seed <= KSampler.MAX_SEED:
            raise ValueError(f"noise_seed must be in [0, {KSampler.MAX_SEED}], got {noise_seed}")
        if not 0.0 <= cfg <= KSampler.MAX_CFG:
            raise ValueError(f"cfg must be in [0.0, {KSampler.MAX_CFG}], got {cfg}")
        output, denoised_output = _execute_generation_custom_sampling(
            model=model,
            noise=_CustomNoiseValue(noise_seed if add_noise else None),
            sampler=cast("_CustomSamplerValue", sampler),
            sigmas=cast("_CustomSigmasValue", sigmas),
            positive=positive,
            negative=negative,
            cfg=cfg,
            latent_image=latent_image,
            conditioning_batching=_conditioning_batching_value(
                conditioning_batching, max_fused_lanes
            ),
        )
        return cls.outputs(output=output, denoised_output=denoised_output)


class GenerationSamplerCustomAdvanced(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.sampler_custom_advanced")

    @classmethod
    def execute(
        cls,
        *,
        noise: object,
        guider: object,
        sampler: object,
        sigmas: object,
        latent_image: object,
    ) -> Mapping[str, object]:
        if type(guider) is _DualCFGGuiderValue:
            output, denoised_output = _execute_generation_custom_sampling(
                model=guider.model,
                noise=cast("_CustomNoiseValue", noise),
                sampler=cast("_CustomSamplerValue", sampler),
                sigmas=cast("_CustomSigmasValue", sigmas),
                positive=guider.cond1,
                middle=guider.cond2,
                negative=guider.negative,
                cfg=guider.cfg_conds,
                middle_scale=guider.cfg_cond2_negative,
                nested_guidance=guider.nested,
                latent_image=latent_image,
                conditioning_batching=guider.batching,
            )
            return cls.outputs(output=output, denoised_output=denoised_output)
        if type(guider) is _DualModelGuiderValue:
            output, denoised_output = _execute_generation_custom_sampling(
                model=guider.model,
                model_negative=guider.model_negative,
                image_only_negative=guider.negative is None,
                noise=cast("_CustomNoiseValue", noise),
                sampler=cast("_CustomSamplerValue", sampler),
                sigmas=cast("_CustomSigmasValue", sigmas),
                positive=guider.positive,
                negative=guider.negative,
                cfg=guider.cfg,
                latent_image=latent_image,
                conditioning_batching=guider.batching,
            )
            return cls.outputs(output=output, denoised_output=denoised_output)
        if type(guider) is _LTXAVDualGuiderValue:
            output, denoised_output = _execute_generation_custom_sampling(
                model=guider.model,
                noise=cast("_CustomNoiseValue", noise),
                sampler=cast("_CustomSamplerValue", sampler),
                sigmas=cast("_CustomSigmasValue", sigmas),
                positive=guider.positive,
                negative=guider.negative,
                cfg=guider.video_cfg,
                audio_cfg=guider.audio_cfg,
                latent_image=latent_image,
                conditioning_batching=guider.batching,
            )
            return cls.outputs(output=output, denoised_output=denoised_output)
        if type(guider) is _PerpNegGuiderValue:
            output, denoised_output = _execute_generation_custom_sampling(
                model=guider.model,
                noise=cast("_CustomNoiseValue", noise),
                sampler=cast("_CustomSamplerValue", sampler),
                sigmas=cast("_CustomSigmasValue", sigmas),
                positive=guider.positive,
                negative=guider.negative,
                cfg=guider.cfg,
                latent_image=latent_image,
                empty=guider.empty,
                neg_scale=guider.neg_scale,
                conditioning_batching=guider.batching,
            )
            return cls.outputs(output=output, denoised_output=denoised_output)
        if type(guider) is not _CustomGuiderValue:
            raise TypeError("guider must come from a Dinkster guider node")
        typed_guider = guider
        output, denoised_output = _execute_generation_custom_sampling(
            model=typed_guider.model,
            noise=cast("_CustomNoiseValue", noise),
            sampler=cast("_CustomSamplerValue", sampler),
            sigmas=cast("_CustomSigmasValue", sigmas),
            positive=typed_guider.positive,
            negative=typed_guider.negative,
            cfg=typed_guider.cfg,
            latent_image=latent_image,
            guider_transforms=typed_guider.transforms,
            conditioning_batching=typed_guider.batching,
        )
        return cls.outputs(output=output, denoised_output=denoised_output)


@dataclass(frozen=True, slots=True)
class _ImpactRegionalProvider:
    model: object
    positive: object
    negative: object
    cfg: float
    sampler: _CustomSamplerValue
    sigmas: tuple[float, ...]


def _impact_regional_provider(
    value: object,
    *,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    steps: int,
    name: str,
) -> _ImpactRegionalProvider:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an Impact BASIC_PIPE tuple")
    typed_value = cast("tuple[object, ...]", value)
    if len(typed_value) != 5:
        raise TypeError(f"{name} must be an Impact BASIC_PIPE tuple")
    model, _clip, _vae, positive, negative = typed_value
    sampler = _custom_sampler_value(sampler_name)
    runtime, sampling_shift, device = _require_custom_sampling_runtime(
        model, "ImpactRegionalSampler"
    )
    inference = importlib.import_module("dinkster_inference")
    scheduler_id = _catalog_id(_inference_registries(inference).schedulers, scheduler, "scheduler")
    descriptor = cast("Any", sampler.descriptor)
    schedule_steps = steps + 1 if descriptor.discard_penultimate else steps
    build_sigmas = _bind_sampling_shift(runtime.custom_sampling_sigmas, sampling_shift)
    sigmas = build_sigmas(scheduler_id, schedule_steps, 1.0, device=device)
    if descriptor.discard_penultimate:
        sigmas = (*sigmas[:-2], sigmas[-1])
    if len(sigmas) != steps + 1:
        raise ValueError(f"{name} schedule must contain {steps + 1} sigmas, got {len(sigmas)}")
    request = inference.CustomSamplingRequest(descriptor, sampler.options, sigmas)
    runtime.check_custom_sampling(
        request,
        has_denoise_mask=True,
        has_inpaint=False,
        has_context_windows=False,
        guidance=None,
    )
    return _ImpactRegionalProvider(model, positive, negative, cfg, sampler, sigmas)


def _impact_regional_masks(
    mask: object,
    latent: object,
    overlap_factor: int,
) -> tuple[object, object]:
    torch = _torch()
    if type(mask) is not torch.Tensor:
        raise TypeError("mask must be an exact torch.Tensor")
    if not isinstance(latent, Mapping):
        raise TypeError("samples must be a LATENT mapping")
    samples = cast("Mapping[object, object]", latent).get("samples")
    if type(samples) is not torch.Tensor:
        raise TypeError("samples['samples'] must be an exact torch.Tensor")
    samples = cast("Any", samples)
    if samples.ndim != 4:
        raise ValueError("Impact regional sampling requires a rank-4 image latent")
    typed_mask = cast("Any", mask)
    combined_mask = torch.ceil(typed_mask.detach().cpu()).to(torch.int32)
    base_mask = torch.where(combined_mask == 0, 1.0, 0.0)
    source_mask = typed_mask.clone()
    if source_mask.ndim == 4:
        source_mask = source_mask.squeeze(0).squeeze(0)
    elif source_mask.ndim == 3:
        source_mask = source_mask.squeeze(0)
    if source_mask.ndim != 2:
        raise ValueError("Impact regional sampling requires a single 2D mask")
    width = source_mask.shape[1]
    height = source_mask.shape[0]
    resized = torch.nn.functional.interpolate(
        source_mask.reshape((-1, 1, height, width)).to(dtype=torch.float32),
        size=(width, height),
        mode="bilinear",
        align_corners=False,
    )
    region_mask = (
        resized
        if overlap_factor == 0
        else torch.clamp(
            torch.nn.functional.conv2d(
                resized.round(),
                torch.ones(
                    (1, 1, overlap_factor, overlap_factor),
                    device=resized.device,
                    dtype=resized.dtype,
                ),
                padding=math.ceil((overlap_factor - 1) / 2),
            ),
            0.0,
            1.0,
        )
    )
    return (
        base_mask,
        region_mask[:, :, :width, :height].round().squeeze(0).squeeze(0),
    )


def _run_impact_regional_pass(
    provider: _ImpactRegionalProvider,
    latent: Mapping[object, object],
    sigmas: tuple[float, ...],
    *,
    seed: int,
    add_noise: bool,
    mask: object | None,
) -> dict[object, object]:
    current = dict(latent)
    if mask is None:
        current.pop("noise_mask", None)
    else:
        current["noise_mask"] = mask
    output, _denoised = _execute_generation_custom_sampling(
        model=provider.model,
        noise=_CustomNoiseValue(seed if add_noise else None),
        sampler=provider.sampler,
        sigmas=_CustomSigmasValue(sigmas),
        positive=provider.positive,
        negative=provider.negative,
        cfg=provider.cfg,
        latent_image=current,
    )
    output.pop("noise_mask", None)
    return output


class GenerationImpactRegionalSampler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.compat.impact_regional_sampler")

    @classmethod
    def execute(
        cls,
        *,
        base_basic_pipe: object,
        region_basic_pipe: object,
        mask: object,
        samples: object,
        seed: int,
        steps: int,
        base_only_steps: int,
        denoise: float,
        overlap_factor: int,
        restore_latent: bool,
        base_cfg: float,
        base_sampler_name: str,
        base_scheduler: str,
        region_cfg: float,
        region_sampler_name: str,
        region_scheduler: str,
    ) -> Mapping[str, object]:
        if not 0 <= seed <= KSampler.MAX_SEED:
            raise ValueError(f"seed must be in [0, {KSampler.MAX_SEED}], got {seed}")
        if not 1 <= steps <= KSampler.MAX_STEPS:
            raise ValueError(f"steps must be in [1, {KSampler.MAX_STEPS}], got {steps}")
        if not 0 <= base_only_steps <= KSampler.MAX_STEPS:
            raise ValueError(
                f"base_only_steps must be in [0, {KSampler.MAX_STEPS}], got {base_only_steps}"
            )
        if not 0.0 < denoise <= 1.0:
            raise ValueError(f"denoise must be in (0.0, 1.0], got {denoise}")
        if not 0 <= overlap_factor <= KSampler.MAX_STEPS:
            raise ValueError(
                f"overlap_factor must be in [0, {KSampler.MAX_STEPS}], got {overlap_factor}"
            )
        if type(restore_latent) is not bool:
            raise TypeError("restore_latent must be a Boolean")
        for name, value in (("base_cfg", base_cfg), ("region_cfg", region_cfg)):
            if not 0.0 <= value <= KSampler.MAX_CFG:
                raise ValueError(f"{name} must be in [0.0, {KSampler.MAX_CFG}], got {value}")
        advanced_steps = int(steps / denoise)
        start_at_step = advanced_steps - steps
        base = _impact_regional_provider(
            base_basic_pipe,
            cfg=base_cfg,
            sampler_name=base_sampler_name,
            scheduler=base_scheduler,
            steps=advanced_steps,
            name="base_basic_pipe",
        )
        region = _impact_regional_provider(
            region_basic_pipe,
            cfg=region_cfg,
            sampler_name=region_sampler_name,
            scheduler=region_scheduler,
            steps=advanced_steps,
            name="region_basic_pipe",
        )
        base_mask, region_mask = _impact_regional_masks(mask, samples, overlap_factor)
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a LATENT mapping")
        current = dict(cast("Mapping[object, object]", samples))
        effective_base_only = min(base_only_steps, steps)
        if effective_base_only:
            current = _run_impact_regional_pass(
                base,
                current,
                base.sigmas[start_at_step : start_at_step + effective_base_only + 1],
                seed=seed,
                add_noise=True,
                mask=None,
            )
        add_noise = effective_base_only == 0
        for index in range(start_at_step + effective_base_only, advanced_steps):
            current = _run_impact_regional_pass(
                base,
                current,
                base.sigmas[index : index + 2],
                seed=seed,
                add_noise=add_noise,
                mask=base_mask,
            )
            base_latent = current
            regional = _run_impact_regional_pass(
                region,
                base_latent,
                region.sigmas[index : index + 2],
                seed=seed,
                add_noise=False,
                mask=region_mask,
            )
            if restore_latent:
                torch = _torch()
                output, destination = _plain_latent(base_latent, torch, "base latent")
                _, source = _plain_latent(regional, torch, "regional latent")
                output["samples"] = _composite_masked_tensor(
                    destination.clone(), source, 0, 0, region_mask, 8, False, torch
                )
                current = output
            else:
                current = regional
            add_noise = False
        current.pop("noise_mask", None)
        return cls.outputs(latent=current)


def _decode_minimax_music3_audio(
    samples: object,
    vae: object,
    *,
    tile_size: int | None = None,
    overlap: int | None = None,
) -> dict[str, object]:
    if not isinstance(samples, Mapping):
        raise TypeError("samples must be a latent mapping")
    torch = _torch()
    latent = cast("Mapping[object, object]", samples).get("samples")
    if not isinstance(latent, torch.Tensor):
        raise TypeError("samples['samples'] must be a torch.Tensor")
    latent = cast("Any", latent)
    if latent.is_nested:
        latent = latent.unbind()[-1]
    if latent.ndim != 3 or latent.shape[0] < 1 or latent.shape[1] != 128:
        raise ValueError("samples['samples'] must be nonempty [batch,128,frames]")
    codec = _native_component_codec(vae)
    with codec.stage():
        with torch.inference_mode():
            load_latent = latent.to(codec.load_device)
            if tile_size is not None:
                assert overlap is not None
                audio = codec.decode_latent_tiled(
                    load_latent, tile=(tile_size,), overlap=(overlap,)
                )
            else:
                direct_oom: BaseException | None = None
                audio = None
                try:
                    audio = codec.decode_latent(load_latent)
                except RuntimeError as caught:
                    if not _is_accelerator_oom(caught, torch=torch, device=codec.load_device):
                        raise
                    direct_oom = caught.with_traceback(None)
                if direct_oom is not None:
                    log.warning(
                        "WARNING: %s out of memory during direct MiniMax Music 3 DAV decode; "
                        "retrying with tiled decode",
                        codec.load_device.type.upper(),
                    )
                    importlib.import_module("dinkster_inference_torch").soft_empty_cache(
                        codec.load_device
                    )
                    audio = codec.decode_latent_tiled(load_latent, tile=(256,), overlap=(32,))
    assert audio is not None
    audio = audio.to(device="cpu", dtype=torch.float32, copy=True)
    if audio.ndim != 3 or audio.shape[0] < 1 or audio.shape[1] != 2 or audio.shape[2] < 1:
        raise ValueError("MiniMax Music 3 DAV must return nonempty [batch,2,samples] audio")
    std = torch.std(audio, dim=(1, 2), keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio /= std
    sample_rate = cast("Mapping[object, object]", samples).get("sample_rate", 44100)
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("samples sample_rate must be a positive integer")
    return {"waveform": audio, "sample_rate": sample_rate}


class NativeVAEDecodeAudio(VAEDecodeAudio):
    @classmethod
    def execute(cls, *, samples: object, vae: object) -> Mapping[str, object]:
        return cls.outputs(audio=_decode_minimax_music3_audio(samples, vae))


class NativeVAEDecodeAudioTiled(VAEDecodeAudioTiled):
    @classmethod
    def execute(
        cls, *, samples: object, vae: object, tile_size: int, overlap: int
    ) -> Mapping[str, object]:
        if type(tile_size) is not int or not 32 <= tile_size <= 8192:
            raise ValueError("tile_size must be an integer in [32, 8192]")
        if type(overlap) is not int or not 0 <= overlap <= 1024:
            raise ValueError("overlap must be an integer in [0, 1024]")
        return cls.outputs(
            audio=_decode_minimax_music3_audio(
                samples,
                vae,
                tile_size=tile_size,
                overlap=overlap,
            )
        )


class GenerationVAEDecode(NativeVAEDecode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vae_decode")

    @classmethod
    def execute(cls, *, samples: object, vae: object) -> Mapping[str, object]:
        inference = importlib.import_module("dinkster_inference")
        component_codec = isinstance(vae, NativeComponentHandle)
        codec = (
            _native_component_codec(vae)
            if component_codec
            else inference.require_inference_codec_handle(vae, "vae")
        )
        torch = _torch()
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a latent mapping")
        latent = cast("Mapping[object, object]", samples).get("samples")
        if type(latent) is inference.MultiStreamLatent:
            streams = cast("Any", latent)
            if streams.roles != ("video",):
                raise TypeError("samples['samples'] must contain exactly one video stream")
            latent = streams.by_role("video")
        if not isinstance(latent, torch.Tensor):
            raise TypeError("samples['samples'] must be a torch.Tensor")
        latent_tensor = cast("Any", latent)
        if len(latent_tensor.shape) not in (4, 5):
            raise ValueError(
                "samples['samples'] must be NCHW or NCTHW rank 4/5, "
                f"got shape {tuple(latent_tensor.shape)}"
            )
        memory_required = getattr(codec, "decode_memory_required", None)
        stage = (
            codec.stage(memory_required=memory_required(latent_tensor))
            if callable(memory_required)
            else codec.stage()
        )
        with stage:
            with torch.inference_mode():
                image = codec.decode_latent(
                    latent_tensor
                    if getattr(codec, "manages_input_device", False)
                    else latent_tensor.to(codec.load_device)
                )
        return cls.outputs(image=_generation_decoded_image(image, codec, component_codec))


def _generation_decoded_image(image: Any, codec: Any, component_codec: bool) -> Any:
    if (
        codec.descriptor.kind == "video"
        and (component_codec or getattr(codec, "accepts_image_batch_latent", False))
        and len(image.shape) == 4
    ):
        if image.shape[1] != 3:
            raise ValueError(
                f"image-capable video codec decode must return [B,3,H,W], got {tuple(image.shape)}"
            )
        image = image.permute(0, 2, 3, 1)
    elif codec.descriptor.kind == "video":
        channels = codec.descriptor.content_channels
        if len(image.shape) != 5 or image.shape[1] != channels:
            raise ValueError(
                f"video codec decode must return [B,{channels},T,H,W], got {tuple(image.shape)}"
            )
        image = image.permute(0, 2, 3, 4, 1).flatten(0, 1)
    else:
        channels = codec.descriptor.content_channels
        if len(image.shape) != 4 or image.shape[1] != channels:
            raise ValueError(
                f"image codec decode must return [B,{channels},H,W], got {tuple(image.shape)}"
            )
        image = image.permute(0, 2, 3, 1)
    return image


class GenerationVAEDecodeTiled(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vae_decode_tiled")

    @classmethod
    def execute(
        cls,
        *,
        samples: object,
        vae: object,
        tile_size: int,
        overlap: int = 64,
        temporal_size: int = 64,
        temporal_overlap: int = 8,
    ) -> Mapping[str, object]:
        values = (tile_size, overlap, temporal_size, temporal_overlap)
        if any(type(value) is not int for value in values):
            raise TypeError("tiled VAE sizes must be exact integers")
        if tile_size <= 0 or overlap < 0 or temporal_size <= 0 or temporal_overlap < 0:
            raise ValueError("tiled VAE sizes must be positive and overlaps nonnegative")
        inference = importlib.import_module("dinkster_inference")
        component_codec = isinstance(vae, NativeComponentHandle)
        codec = inference.require_inference_tiled_codec_handle(
            _native_component_codec(vae) if component_codec else vae,
            "vae",
        )
        torch = _torch()
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a latent mapping")
        latent = cast("Mapping[object, object]", samples).get("samples")
        if type(latent) is inference.MultiStreamLatent:
            streams = cast("Any", latent)
            if streams.roles != ("video",):
                raise TypeError("samples['samples'] must contain exactly one video stream")
            latent = streams.by_role("video")
        if not isinstance(latent, torch.Tensor):
            raise TypeError("samples['samples'] must be a torch.Tensor")
        latent_tensor = cast("Any", latent)
        dimensions = codec.descriptor.latent.dimensions
        expected_rank = dimensions + 2
        accepts_image_batch = getattr(codec, "accepts_image_batch_latent", False)
        if len(latent_tensor.shape) != expected_rank and not (
            accepts_image_batch and len(latent_tensor.shape) == expected_rank - 1
        ):
            raise ValueError(
                f"samples['samples'] must be rank {expected_rank}, "
                f"got shape {tuple(latent_tensor.shape)}"
            )
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_size // 2
        spatial_scale = codec.descriptor.latent.spatial_downscale
        spatial_tile = tile_size // spatial_scale
        spatial_overlap = overlap // spatial_scale
        if dimensions == 3:
            temporal_scale = codec.descriptor.latent.temporal_downscale
            temporal_size = max(2, temporal_size // temporal_scale)
            temporal_overlap = max(
                1,
                min(temporal_size // 2, temporal_overlap // temporal_scale),
            )
            tile = (temporal_size, spatial_tile, spatial_tile)
            tile_overlap = (temporal_overlap, spatial_overlap, spatial_overlap)
        elif dimensions == 2:
            tile = (spatial_tile, spatial_tile)
            tile_overlap = (spatial_overlap, spatial_overlap)
        else:
            tile = (spatial_tile,)
            tile_overlap = (spatial_overlap,)
        if any(value <= 0 for value in tile) or any(
            overlap_value >= tile_value
            for overlap_value, tile_value in zip(tile_overlap, tile, strict=True)
        ):
            raise ValueError("tiled VAE sizes do not produce a valid latent tile")
        with codec.stage():
            with torch.inference_mode():
                image = codec.decode_latent_tiled(
                    (
                        latent_tensor
                        if getattr(codec, "manages_input_device", False)
                        else latent_tensor.to(codec.load_device)
                    ),
                    tile=tile,
                    overlap=tile_overlap,
                )
        return cls.outputs(image=_generation_decoded_image(image, codec, component_codec))


class GenerationVAEEncode(NativeVAEEncode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vae_encode")

    @classmethod
    def execute(cls, *, pixels: object, vae: object) -> Mapping[str, object]:
        inference = importlib.import_module("dinkster_inference")
        component_codec = isinstance(vae, NativeComponentHandle)
        codec = (
            _native_component_codec(vae)
            if component_codec
            else inference.require_inference_codec_handle(vae, "vae")
        )
        torch = _torch()
        if not isinstance(pixels, torch.Tensor):
            raise TypeError("pixels must be a torch.Tensor")
        pixel_tensor = cast("Any", pixels)
        batched_video = getattr(codec, "accepts_batched_video", False)
        if batched_video and len(pixel_tensor.shape) == 5 and pixel_tensor.shape[-1] == 3:
            content = pixel_tensor.permute(0, 4, 1, 2, 3)
        else:
            if len(pixel_tensor.shape) != 4 or pixel_tensor.shape[-1] != 3:
                raise ValueError(
                    f"pixels must be NHWC rank 4, got shape {tuple(pixel_tensor.shape)}"
                )
            content = pixel_tensor.permute(0, 3, 1, 2)
        if (
            not batched_video
            and codec.descriptor.kind == "video"
            and (not component_codec or getattr(codec, "sequence_content", False))
        ):
            content = content.permute(1, 0, 2, 3).unsqueeze(0)
        memory_required = getattr(codec, "encode_memory_required", None)
        stage = (
            codec.stage(memory_required=memory_required(content))
            if callable(memory_required)
            else codec.stage()
        )
        with stage:
            with torch.inference_mode():
                latent = codec.encode_content(
                    content
                    if getattr(codec, "manages_input_device", False)
                    else content.to(codec.load_device)
                )
        return cls.outputs(latent={"samples": latent})


class GenerationVAEEncodeTiled(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vae_encode_tiled")

    @classmethod
    def execute(
        cls,
        *,
        pixels: object,
        vae: object,
        tile_size: int,
        overlap: int,
        temporal_size: int,
        temporal_overlap: int,
    ) -> Mapping[str, object]:
        inference = importlib.import_module("dinkster_inference")
        component_codec = isinstance(vae, NativeComponentHandle)
        codec = inference.require_inference_tiled_codec_handle(
            _native_component_codec(vae) if component_codec else vae,
            "vae",
        )
        torch = _torch()
        if not isinstance(pixels, torch.Tensor):
            raise TypeError("pixels must be a torch.Tensor")
        pixel_tensor = cast("Any", pixels)
        batched_video = getattr(codec, "accepts_batched_video", False)
        if batched_video and len(pixel_tensor.shape) == 5 and pixel_tensor.shape[-1] == 3:
            content = pixel_tensor.permute(0, 4, 1, 2, 3)
        else:
            if len(pixel_tensor.shape) != 4 or pixel_tensor.shape[-1] != 3:
                raise ValueError(
                    f"pixels must be NHWC rank 4, got shape {tuple(pixel_tensor.shape)}"
                )
            content = pixel_tensor.permute(0, 3, 1, 2)
        if (
            not batched_video
            and codec.descriptor.kind == "video"
            and (not component_codec or getattr(codec, "sequence_content", False))
        ):
            content = content.permute(1, 0, 2, 3).unsqueeze(0)
        dimensions = codec.descriptor.latent.dimensions
        if dimensions == 3:
            tile = (temporal_size, tile_size, tile_size)
            tile_overlap = (temporal_overlap, overlap, overlap)
        elif dimensions == 2:
            tile = (tile_size, tile_size)
            tile_overlap = (overlap, overlap)
        else:
            tile = (tile_size,)
            tile_overlap = (overlap,)
        with codec.stage():
            with torch.inference_mode():
                latent = codec.encode_content_tiled(
                    content
                    if getattr(codec, "manages_input_device", False)
                    else content.to(codec.load_device),
                    tile=tile,
                    overlap=tile_overlap,
                )
        return cls.outputs(latent={"samples": latent})


def _seedvr2_bthwc(value: Any, name: str) -> tuple[Any, bool]:
    if value.ndim == 4:
        return value.unsqueeze(0), True
    if value.ndim == 5:
        return value, False
    raise ValueError(f"{name}: expected 4-D or 5-D IMAGE tensor, got shape {tuple(value.shape)}")


class GenerationSeedVR2Preprocess(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.seedvr2_preprocess")

    @classmethod
    def execute(cls, *, resized_images: object) -> Mapping[str, object]:
        torch = _torch()
        if not isinstance(resized_images, torch.Tensor):
            raise TypeError("resized_images must be a torch.Tensor")
        images = cast("Any", resized_images)
        images, _ = _seedvr2_bthwc(images, "SeedVR2Preprocess")
        if images.shape[1] < 1:
            raise ValueError("SeedVR2Preprocess expected at least one frame.")
        if min(images.shape[2], images.shape[3]) < 2:
            raise ValueError("SeedVR2Preprocess: input shorter edge must be at least 2 pixels")
        if images.shape[-1] > 3:
            images = images[..., :3]
        elif images.shape[-1] != 3:
            raise ValueError("SeedVR2Preprocess: input must have at least three channels")
        images = images.permute(0, 1, 4, 2, 3)
        batch, frames, channels, height, width = images.shape
        images = images.reshape(batch * frames, channels, height, width).clamp(0.0, 1.0)
        pad_height = (16 - height % 16) % 16
        pad_width = (16 - width % 16) % 16
        if pad_height or pad_width:
            images = torch.nn.functional.pad(images, (0, pad_width, 0, pad_height))
        images = images.reshape(batch, frames, channels, height + pad_height, width + pad_width)
        if frames > 1 and (frames - 1) % 4:
            padding = images[:, -1:].repeat(1, 4 - (frames - 1) % 4, 1, 1, 1)
            images = torch.cat((images, padding), dim=1)
        return cls.outputs(images=images.permute(0, 1, 3, 4, 2).contiguous())


def _seedvr2_restore_reference_shape(decoded: Any, reference: Any) -> Any:
    if decoded.shape[0] != 1:
        return decoded
    reference_batch, reference_frames = reference.shape[:2]
    if reference_batch < 1 or decoded.shape[1] % reference_batch:
        return decoded
    decoded_frames = decoded.shape[1] // reference_batch
    if decoded_frames < reference_frames:
        return decoded
    return decoded.reshape(
        reference_batch,
        decoded_frames,
        decoded.shape[2],
        decoded.shape[3],
        decoded.shape[4],
    )


def _seedvr2_resize_reference(reference: Any, height: int, width: int, torch: Any) -> Any:
    if reference.shape[2:4] == (height, width):
        return reference
    batch, frames = reference.shape[:2]
    flat = reference.permute(0, 1, 4, 2, 3).reshape(
        batch * frames,
        reference.shape[4],
        reference.shape[2],
        reference.shape[3],
    )
    resized = torch.nn.functional.interpolate(
        flat,
        size=(height, width),
        mode="bicubic",
        antialias=flat.device.type != "mps",
    )
    return resized.reshape(batch, frames, resized.shape[1], height, width).permute(0, 1, 3, 4, 2)


def _seedvr2_color_chunk_size(flat: Any, method: str, torch: Any, inference_torch: Any) -> int:
    constants = importlib.import_module("dinkster_inference_torch.seedvr2_constants")
    multiplier = {
        "lab": constants.SEEDVR2_LAB_SCALE_MULTIPLIER,
        "wavelet": constants.SEEDVR2_WAVELET_SCALE_MULTIPLIER,
        "adain": constants.SEEDVR2_ADAIN_SCALE_MULTIPLIER,
    }[method]
    frames, channels, height, width = flat.shape
    dtype_bytes = max(flat.element_size(), constants.SEEDVR2_DTYPE_BYTES_FLOOR)
    bytes_per_frame = height * width * channels * dtype_bytes * multiplier
    if bytes_per_frame <= 0:
        return frames
    device = select_load_device(torch)
    free_memory = inference_torch.get_free_memory(device).free_total
    available = int((free_memory * constants.SEEDVR2_COLOR_MEM_HEADROOM) // bytes_per_frame)
    return max(1, min(frames, available))


def _seedvr2_color_transfer(
    decoded: Any,
    reference: Any,
    method: str,
    torch: Any,
    inference_torch: Any,
) -> Any:
    transfer = {
        "lab": inference_torch.lab_color_transfer,
        "wavelet": inference_torch.wavelet_color_transfer,
        "adain": inference_torch.adain_color_transfer,
    }[method]
    color_device = select_load_device(torch)
    output_device = decoded.device
    chunk_size = _seedvr2_color_chunk_size(decoded, method, torch, inference_torch)
    while True:
        result = None
        try:
            for start in range(0, decoded.shape[0], chunk_size):
                end = min(start + chunk_size, decoded.shape[0])
                if method == "lab":
                    for index in range(start, end):
                        output = transfer(
                            decoded[index : index + 1].to(color_device).clone(),
                            reference[index : index + 1].to(color_device).clone(),
                        ).to(output_device)
                        if result is None:
                            result = torch.empty(
                                (decoded.shape[0],) + tuple(output.shape[1:]),
                                device=output_device,
                                dtype=output.dtype,
                            )
                        result[index : index + 1].copy_(output)
                else:
                    output = transfer(
                        decoded[start:end].to(color_device),
                        reference[start:end].to(color_device),
                    ).to(output_device)
                    if result is None:
                        result = torch.empty(
                            (decoded.shape[0],) + tuple(output.shape[1:]),
                            device=output_device,
                            dtype=output.dtype,
                        )
                    result[start:end].copy_(output)
            if result is None:
                raise ValueError(
                    "SeedVR2PostProcessing: color correction requires at least one frame."
                )
            return result
        except RuntimeError as error:
            if not _is_accelerator_oom(error, torch=torch, device=color_device):
                raise
            if chunk_size <= 1:
                raise RuntimeError(
                    "SeedVR2PostProcessing: color correction OOM at one frame"
                ) from error
            chunk_size = max(1, chunk_size // 2)


class GenerationSeedVR2PostProcessing(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.seedvr2_postprocess")

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        original_resized_images: object,
        color_correction_method: str,
    ) -> Mapping[str, object]:
        torch = _torch()
        if not isinstance(images, torch.Tensor) or not isinstance(
            original_resized_images, torch.Tensor
        ):
            raise TypeError("images and original_resized_images must be torch.Tensor values")
        if color_correction_method not in ("lab", "wavelet", "adain", "none"):
            raise ValueError(
                "SeedVR2PostProcessing: unknown color_correction_method "
                f"{color_correction_method!r}"
            )
        decoded, decoded_was_4d = _seedvr2_bthwc(cast("Any", images), "SeedVR2PostProcessing")
        original = cast("Any", original_resized_images)
        alpha = original[..., 3:4] if original.shape[-1] == 4 else None
        if original.shape[-1] >= 3:
            original = original[..., :3]
        reference, _ = _seedvr2_bthwc(original, "SeedVR2PostProcessing")
        decoded = _seedvr2_restore_reference_shape(decoded, reference)
        batch = min(decoded.shape[0], reference.shape[0])
        frames = min(decoded.shape[1], reference.shape[1])
        height = min(decoded.shape[2], reference.shape[2])
        width = min(decoded.shape[3], reference.shape[3])
        decoded = decoded[:batch, :frames, :height, :width]
        if color_correction_method == "none":
            output = decoded
        else:
            reference = _seedvr2_resize_reference(reference[:batch, :frames], height, width, torch)
            decoded_flat = (
                decoded.mul(2.0)
                .sub(1.0)
                .permute(0, 1, 4, 2, 3)
                .reshape(batch * frames, decoded.shape[4], height, width)
            )
            reference_flat = (
                reference.mul(2.0)
                .sub(1.0)
                .permute(0, 1, 4, 2, 3)
                .reshape(batch * frames, reference.shape[4], height, width)
            )
            inference_torch = importlib.import_module("dinkster_inference_torch")
            corrected = _seedvr2_color_transfer(
                decoded_flat,
                reference_flat,
                color_correction_method,
                torch,
                inference_torch,
            )
            output = corrected.reshape(
                batch, frames, corrected.shape[1], corrected.shape[2], corrected.shape[3]
            ).permute(0, 1, 3, 4, 2)
            output = output.add(1.0).div(2.0).clamp(0.0, 1.0)
        if alpha is not None:
            alpha, _ = _seedvr2_bthwc(alpha, "SeedVR2PostProcessing")
            alpha = alpha[:batch, :frames, : output.shape[2], : output.shape[3]]
            output = torch.cat((output, alpha.to(device=output.device, dtype=output.dtype)), dim=-1)
        output = output[:, :, : output.shape[2] // 2 * 2, : output.shape[3] // 2 * 2]
        if decoded_was_4d:
            output = output.flatten(0, 1)
        return cls.outputs(images=output)


class GenerationSeedVR2Conditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.seedvr2_conditioning")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        vae_conditioning: object,
    ) -> Mapping[str, object]:
        model, _applications = _application_chain_model(model, "model")
        handle, *_options = _native_model(model, "model")
        torch = _torch()
        _, samples = _plain_latent(vae_conditioning, torch, "vae_conditioning")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        positive, negative = inference_torch.seedvr2_conditioning(
            samples,
            component_identity=handle.recipe.runtime_identity,
        )

        def row(conditioning: Any) -> list[list[object]]:
            return [
                [
                    conditioning.embeddings,
                    {_NATIVE_PREPARED_CONDITIONING_KEY: conditioning},
                ]
            ]

        return cls.outputs(positive=row(positive), negative=row(negative))


class GenerationSeedVR2TemporalChunk(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.seedvr2_temporal_chunk")

    @classmethod
    def execute(
        cls,
        *,
        latent: object,
        temporal_overlap: int,
        chunking_mode: object,
    ) -> Mapping[str, object]:
        torch = _torch()
        metadata, samples = _plain_latent(latent, torch, "latent")
        if samples.ndim != 5 or samples.shape[1] != 16:
            raise ValueError("SeedVR2TemporalChunk requires a [B,16,T,H,W] video latent")
        if temporal_overlap < 0:
            raise ValueError("temporal_overlap must be nonnegative")
        if not isinstance(chunking_mode, Mapping):
            raise TypeError("chunking_mode must be a dynamic-combo mapping")
        typed_mode = cast("Mapping[str, object]", chunking_mode)
        mode = typed_mode.get("chunking_mode")
        latent_frames = samples.shape[2]
        pixel_frames = 4 * (latent_frames - 1) + 1
        if mode == "auto":
            inference_torch = importlib.import_module("dinkster_inference_torch")
            constants = importlib.import_module("dinkster_inference_torch.seedvr2_constants")
            free_gib = (
                inference_torch.get_free_memory(select_load_device(torch)).free_total / 1024**3
            )
            mpx = (
                samples.shape[0]
                * samples.shape[3]
                * samples.shape[4]
                * constants.BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE**2
                / 1e6
            )
            budget = (
                free_gib
                - constants.SEEDVR2_CHUNK_RESERVED_GIB
                - constants.SEEDVR2_CHUNK_SIGMA_K * constants.SEEDVR2_CHUNK_SIGMA_GIB
            )
            maximum = max(1, int(budget / (constants.SEEDVR2_CHUNK_GIB_PER_MPX_FRAME * mpx)))
            frames_per_chunk = min(4 * (maximum - 1) + 1, pixel_frames)
        elif mode == "manual":
            frames_per_chunk = typed_mode.get("frames_per_chunk")
            if type(frames_per_chunk) is not int:
                raise TypeError("manual chunking requires integer frames_per_chunk")
            if frames_per_chunk < 1 or (frames_per_chunk - 1) % 4:
                raise ValueError("frames_per_chunk must be a 4n+1 pixel-frame count")
        else:
            raise ValueError("chunking_mode must select 'auto' or 'manual'")
        if pixel_frames <= frames_per_chunk:
            return cls.outputs(latents=[metadata], temporal_overlap=0)
        chunk_frames = (frames_per_chunk - 1) // 4 + 1
        overlap = min(temporal_overlap, chunk_frames - 1)
        step = chunk_frames - overlap
        chunks: list[dict[Any, Any]] = []
        for start in range(0, latent_frames, step):
            end = min(start + chunk_frames, latent_frames)
            chunk = dict(metadata)
            chunk["samples"] = samples[:, :, start:end].contiguous()
            chunks.append(chunk)
            if end >= latent_frames:
                break
        return cls.outputs(latents=chunks, temporal_overlap=overlap)


class GenerationSeedVR2TemporalMerge(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.seedvr2_temporal_merge")

    @classmethod
    def execute(
        cls,
        *,
        latents: object,
        temporal_overlap: int,
    ) -> Mapping[str, object]:
        if temporal_overlap < 0:
            raise ValueError("temporal_overlap must be nonnegative")
        if isinstance(latents, str | bytes) or not isinstance(latents, Sequence) or not latents:
            raise ValueError("latents must contain at least one temporal chunk")
        torch = _torch()
        typed = cast("Sequence[object]", latents)
        metadata, first = _plain_latent(typed[0], torch, "latents[0]")
        if first.ndim != 5 or first.shape[1] != 16:
            raise ValueError("SeedVR2TemporalMerge requires [B,16,T,H,W] video latents")
        chunks = [first]
        for index, value in enumerate(typed[1:], 1):
            _, chunk = _plain_latent(value, torch, f"latents[{index}]")
            if chunk.shape[:2] != first.shape[:2] or chunk.shape[3:] != first.shape[3:]:
                raise ValueError(f"latents[{index}] does not match the first chunk")
            if index < len(typed) - 1 and chunk.shape[2] != first.shape[2]:
                raise ValueError("only the final SeedVR2 temporal chunk may be shorter")
            chunks.append(chunk)
        metadata.pop("noise_mask", None)
        if len(chunks) == 1:
            metadata["samples"] = first
        elif temporal_overlap == 0:
            metadata["samples"] = torch.cat(chunks, dim=2)
        else:
            chunk_frames = first.shape[2]
            step = chunk_frames - min(temporal_overlap, chunk_frames - 1)
            total = step * (len(chunks) - 1) + chunks[-1].shape[2]
            merged = torch.empty(
                (*first.shape[:2], total, *first.shape[3:]),
                device=first.device,
                dtype=first.dtype,
            )
            merged[:, :, :chunk_frames] = first
            filled = chunk_frames
            for index, chunk in enumerate(chunks[1:], 1):
                start = index * step
                end = start + chunk.shape[2]
                fade = min(filled - start, chunk.shape[2])
                if fade > 0:
                    ramp = torch.linspace(0.0, 1.0, fade, device=chunk.device, dtype=chunk.dtype)
                    ramp = ((ramp - 1.0 / 3.0) / (1.0 / 3.0)).clamp(0.0, 1.0)
                    previous = (0.5 + 0.5 * torch.cos(torch.pi * ramp)).view(1, 1, fade, 1, 1)
                    merged[:, :, start : start + fade] = merged[
                        :, :, start : start + fade
                    ] * previous + chunk[:, :, :fade] * (1.0 - previous)
                merged[:, :, start + fade : end] = chunk[:, :, fade:]
                filled = end
            metadata["samples"] = merged
        return cls.outputs(latent=metadata)


_LATENT_RESIZE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "bislerp")


def _check_bounds(*constraints: tuple[str, float, float, float]) -> None:
    for name, value, low, high in constraints:
        if not low <= value <= high:
            raise ValueError(f"{name} must be in [{low}, {high}], got {value}")


def _check_choice(name: str, value: str, options: tuple[str, ...]) -> None:
    if value not in options:
        raise ValueError(f"{name} must be one of {options}, got {value!r}")


def _plain_latent(value: object, torch: Any, name: str) -> tuple[dict[Any, Any], Any]:
    """Unwrap a plain-tensor LATENT mapping, copying its metadata keys."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a LATENT mapping")
    mapping = cast("Mapping[Any, Any]", value)
    samples = mapping.get("samples")
    if type(samples) is not torch.Tensor:
        raise TypeError(f"{name} samples must be an exact torch.Tensor")
    return dict(mapping), samples


def _reshape_latent_to(target_shape: Any, latent: Any, repeat_batch: bool = True) -> Any:
    """Port of comfy_extras.nodes_latent.reshape_latent_to at b78cec87."""
    utils = importlib.import_module("dinkster_inference_torch.resize")
    if latent.shape[1:] != target_shape[1:]:
        latent = utils.common_upscale(
            latent, target_shape[-1], target_shape[-2], "bilinear", "center"
        )
    if repeat_batch:
        return utils.repeat_to_batch_size(latent, target_shape[0])
    return latent


def _composite_masked_tensor(
    destination: Any,
    source: Any,
    x: int,
    y: int,
    mask: Any,
    multiplier: int,
    resize_source: bool,
    torch: Any,
) -> Any:
    """Port of comfy_extras.nodes_mask.composite at b78cec87; mutates destination."""
    utils = importlib.import_module("dinkster_inference_torch.resize")
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(
            source, size=(destination.shape[-2], destination.shape[-1]), mode="bilinear"
        )
    source = utils.repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[-1] * multiplier, min(x, destination.shape[-1] * multiplier))
    y = max(-source.shape[-2] * multiplier, min(y, destination.shape[-2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[-1], top + source.shape[-2])

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(source.shape[-2], source.shape[-1]),
            mode="bilinear",
        )
        mask = utils.repeat_to_batch_size(mask, source.shape[0])

    # Only the source region overlapping the destination is written, so an
    # offset near the edge never writes out of bounds.
    visible_width, visible_height = (
        destination.shape[-1] - left + min(0, x),
        destination.shape[-2] - top + min(0, y),
    )

    mask = mask[:, :, :visible_height, :visible_width]
    if mask.ndim < source.ndim:
        mask = mask.unsqueeze(1)

    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[..., :visible_height, :visible_width]
    destination_portion = inverse_mask * destination[..., top:bottom, left:right]

    destination[..., top:bottom, left:right] = source_portion + destination_portion
    return destination


class GenerationLatentCombine(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.combine")

    @classmethod
    def execute(cls, *, samples1: object, samples2: object, operation: str) -> Mapping[str, object]:
        _check_choice("operation", operation, ("add", "subtract"))
        torch = _torch()
        out, s1 = _plain_latent(samples1, torch, "samples1")
        _, s2 = _plain_latent(samples2, torch, "samples2")
        s2 = _reshape_latent_to(s1.shape, s2)
        out["samples"] = s1 + s2 if operation == "add" else s1 - s2
        return cls.outputs(latent=out)


class GenerationLatentMix(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.mix")

    @classmethod
    def execute(
        cls, *, samples1: object, samples2: object, operation: str, factor: float
    ) -> Mapping[str, object]:
        _check_choice("operation", operation, ("interpolate", "blend"))
        _check_bounds(("factor", factor, 0.0, 1.0))
        torch = _torch()
        out, s1 = _plain_latent(samples1, torch, "samples1")
        _, s2 = _plain_latent(samples2, torch, "samples2")
        if operation == "interpolate":
            s2 = _reshape_latent_to(s1.shape, s2)
            m1 = torch.linalg.vector_norm(s1, dim=(1))
            m2 = torch.linalg.vector_norm(s2, dim=(1))
            n1 = torch.nan_to_num(s1 / m1)
            n2 = torch.nan_to_num(s2 / m2)
            t = n1 * factor + n2 * (1.0 - factor)
            mt = torch.linalg.vector_norm(t, dim=(1))
            st = torch.nan_to_num(t / mt)
            out["samples"] = st * (m1 * factor + m2 * (1.0 - factor))
        else:
            if s1.shape != s2.shape:
                s2 = importlib.import_module("dinkster_inference_torch.resize").common_upscale(
                    s2, s1.shape[3], s1.shape[2], "bicubic", crop="center"
                )
            out["samples"] = s1 * factor + s2 * (1 - factor)
        return cls.outputs(latent=out)


class GenerationLatentMultiply(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.multiply")

    @classmethod
    def execute(cls, *, samples: object, multiplier: float) -> Mapping[str, object]:
        _check_bounds(("multiplier", multiplier, -10.0, 10.0))
        torch = _torch()
        out, s1 = _plain_latent(samples, torch, "samples")
        out["samples"] = s1 * multiplier
        return cls.outputs(latent=out)


class GenerationLatentRotate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.rotate")

    @classmethod
    def execute(cls, *, samples: object, angle: str) -> Mapping[str, object]:
        _check_choice("angle", angle, ("none", "90", "180", "270"))
        torch = _torch()
        out, s1 = _plain_latent(samples, torch, "samples")
        rotate_by = {"none": 0, "90": 1, "180": 2, "270": 3}[angle]
        out["samples"] = torch.rot90(s1, k=rotate_by, dims=[3, 2])
        return cls.outputs(latent=out)


class GenerationLatentFlip(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.flip")

    @classmethod
    def execute(cls, *, samples: object, axis: str) -> Mapping[str, object]:
        _check_choice("axis", axis, ("vertical", "horizontal"))
        torch = _torch()
        out, s1 = _plain_latent(samples, torch, "samples")
        out["samples"] = torch.flip(s1, dims=[2] if axis == "vertical" else [3])
        return cls.outputs(latent=out)


class GenerationLatentCrop(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.crop")

    @classmethod
    def execute(
        cls, *, samples: object, width: int, height: int, x: int, y: int
    ) -> Mapping[str, object]:
        _check_bounds(
            ("width", width, 64, 16_384),
            ("height", height, 64, 16_384),
            ("x", x, 0, 16_384),
            ("y", y, 0, 16_384),
        )
        torch = _torch()
        out, s_in = _plain_latent(samples, torch, "samples")
        x = x // 8
        y = y // 8
        # Clamp the origin so the crop keeps at least 8 latent cells (64 px).
        if x > (s_in.shape[3] - 8):
            x = s_in.shape[3] - 8
        if y > (s_in.shape[2] - 8):
            y = s_in.shape[2] - 8
        new_height = height // 8
        new_width = width // 8
        out["samples"] = s_in[:, :, y : y + new_height, x : x + new_width]
        return cls.outputs(latent=out)


class GenerationLatentResize(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.resize")

    @classmethod
    def execute(
        cls, *, samples: object, method: str, width: int, height: int, crop: str
    ) -> Mapping[str, object]:
        _check_choice("method", method, _LATENT_RESIZE_METHODS)
        _check_choice("crop", crop, ("disabled", "center"))
        _check_bounds(("width", width, 0, 16_384), ("height", height, 0, 16_384))
        if width == 0 and height == 0:
            return cls.outputs(latent=samples)
        torch = _torch()
        out, s_in = _plain_latent(samples, torch, "samples")
        if width == 0:
            height = max(64, height)
            width = max(64, round(s_in.shape[-1] * height / s_in.shape[-2]))
        elif height == 0:
            width = max(64, width)
            height = max(64, round(s_in.shape[-2] * width / s_in.shape[-1]))
        else:
            width = max(64, width)
            height = max(64, height)
        out["samples"] = importlib.import_module("dinkster_inference_torch.resize").common_upscale(
            s_in, width // 8, height // 8, method, crop
        )
        return cls.outputs(latent=out)


class GenerationLatentResizeBy(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.resize_by")

    @classmethod
    def execute(cls, *, samples: object, method: str, scale_by: float) -> Mapping[str, object]:
        _check_choice("method", method, _LATENT_RESIZE_METHODS)
        _check_bounds(("scale_by", scale_by, 0.01, 8.0))
        torch = _torch()
        out, s_in = _plain_latent(samples, torch, "samples")
        width = round(s_in.shape[-1] * scale_by)
        height = round(s_in.shape[-2] * scale_by)
        out["samples"] = importlib.import_module("dinkster_inference_torch.resize").common_upscale(
            s_in, width, height, method, "disabled"
        )
        return cls.outputs(latent=out)


class GenerationLatentComposite(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.composite")

    @classmethod
    def execute(
        cls, *, destination: object, source: object, x: int, y: int, feather: int
    ) -> Mapping[str, object]:
        _check_bounds(
            ("x", x, 0, 16_384),
            ("y", y, 0, 16_384),
            ("feather", feather, 0, 16_384),
        )
        torch = _torch()
        out, dest = _plain_latent(destination, torch, "destination")
        _, src = _plain_latent(source, torch, "source")
        x = x // 8
        y = y // 8
        feather = feather // 8
        s = dest.clone()
        if feather == 0:
            s[:, :, y : y + src.shape[2], x : x + src.shape[3]] = src[
                :, :, : dest.shape[2] - y, : dest.shape[3] - x
            ]
        else:
            src = src[:, :, : dest.shape[2] - y, : dest.shape[3] - x]
            mask = torch.ones_like(src)
            for t in range(feather):
                if y != 0:
                    mask[:, :, t : 1 + t, :] *= (1.0 / feather) * (t + 1)
                if y + src.shape[2] < dest.shape[2]:
                    mask[:, :, mask.shape[2] - 1 - t : mask.shape[2] - t, :] *= (1.0 / feather) * (
                        t + 1
                    )
                if x != 0:
                    mask[:, :, :, t : 1 + t] *= (1.0 / feather) * (t + 1)
                if x + src.shape[3] < dest.shape[3]:
                    mask[:, :, :, mask.shape[3] - 1 - t : mask.shape[3] - t] *= (1.0 / feather) * (
                        t + 1
                    )
            rev_mask = torch.ones_like(mask) - mask
            s[:, :, y : y + src.shape[2], x : x + src.shape[3]] = (
                src * mask + s[:, :, y : y + src.shape[2], x : x + src.shape[3]] * rev_mask
            )
        out["samples"] = s
        return cls.outputs(latent=out)


class GenerationLatentCompositeMasked(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.composite_masked")

    @classmethod
    def execute(
        cls,
        *,
        destination: object,
        source: object,
        x: int,
        y: int,
        resize_source: bool,
        mask: object = None,
    ) -> Mapping[str, object]:
        _check_bounds(("x", x, 0, 16_384), ("y", y, 0, 16_384))
        torch = _torch()
        out, dest = _plain_latent(destination, torch, "destination")
        _, src = _plain_latent(source, torch, "source")
        if mask is not None and type(mask) is not torch.Tensor:
            raise TypeError("mask must be an exact torch.Tensor")
        out["samples"] = _composite_masked_tensor(
            dest.clone(), src, x, y, mask, 8, resize_source, torch
        )
        return cls.outputs(latent=out)


class GenerationLatentConcat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.concat")

    @classmethod
    def execute(cls, *, samples1: object, samples2: object, dim: str) -> Mapping[str, object]:
        _check_choice("dim", dim, ("x", "-x", "y", "-y", "t", "-t"))
        torch = _torch()
        out, s1 = _plain_latent(samples1, torch, "samples1")
        _, s2 = _plain_latent(samples2, torch, "samples2")
        s2 = importlib.import_module("dinkster_inference_torch.resize").repeat_to_batch_size(
            s2, s1.shape[0]
        )
        ordered = (s2, s1) if "-" in dim else (s1, s2)
        axis = -1 if "x" in dim else -2 if "y" in dim else -3
        out["samples"] = torch.cat(ordered, dim=axis)
        return cls.outputs(latent=out)


class GenerationLatentCut(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.cut")

    @classmethod
    def execute(cls, *, samples: object, dim: str, index: int, amount: int) -> Mapping[str, object]:
        _check_choice("dim", dim, ("x", "y", "t"))
        _check_bounds(("index", index, -16_384, 16_384), ("amount", amount, 1, 16_384))
        torch = _torch()
        out, s1 = _plain_latent(samples, torch, "samples")
        axis = s1.ndim - 1 if dim == "x" else s1.ndim - 2 if dim == "y" else s1.ndim - 3
        if index >= 0:
            index = min(index, s1.shape[axis] - 1)
            amount = min(s1.shape[axis] - index, amount)
        else:
            index = max(index, -s1.shape[axis])
            amount = min(-index, amount)
        out["samples"] = torch.narrow(s1, axis, index, amount)
        return cls.outputs(latent=out)


class GenerationLatentCutToBatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.cut_to_batch")

    @classmethod
    def execute(cls, *, samples: object, dim: str, slice_size: int) -> Mapping[str, object]:
        _check_choice("dim", dim, ("t", "x", "y"))
        _check_bounds(("slice_size", slice_size, 1, 16_384))
        torch = _torch()
        out, s1 = _plain_latent(samples, torch, "samples")
        axis = s1.ndim - 1 if dim == "x" else s1.ndim - 2 if dim == "y" else s1.ndim - 3
        if axis < 2:
            # The axis coincides with batch or channels (t on a 4D latent):
            # the latent passes through unchanged.
            return cls.outputs(latent=samples)
        s = s1.movedim(axis, 1)
        if s.shape[1] < slice_size:
            slice_size = s.shape[1]
        elif s.shape[1] % slice_size != 0:
            s = s[:, : math.floor(s.shape[1] / slice_size) * slice_size]
        new_shape = [-1, slice_size] + list(s.shape[2:])
        out["samples"] = s.reshape(new_shape).movedim(1, axis)
        return cls.outputs(latent=out)


class GenerationLatentFromBatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.from_batch")

    @classmethod
    def execute(cls, *, samples: object, batch_index: int, length: int) -> Mapping[str, object]:
        _check_bounds(
            ("batch_index", batch_index, -16_384, 16_384),
            ("length", length, 1, 64),
        )
        torch = _torch()
        out, s_in = _plain_latent(samples, torch, "samples")
        if batch_index < 0:
            batch_index = s_in.shape[0] + batch_index
        batch_index = max(0, min(s_in.shape[0] - 1, batch_index))
        length = min(s_in.shape[0] - batch_index, length)
        out["samples"] = s_in[batch_index : batch_index + length].clone()
        if "noise_mask" in out:
            masks = out["noise_mask"]
            if masks.shape[0] == 1:
                out["noise_mask"] = masks.clone()
            else:
                if masks.shape[0] < s_in.shape[0]:
                    masks = masks.repeat(math.ceil(s_in.shape[0] / masks.shape[0]), 1, 1, 1)[
                        : s_in.shape[0]
                    ]
                out["noise_mask"] = masks[batch_index : batch_index + length].clone()
        if "batch_index" not in out:
            out["batch_index"] = list(range(batch_index, batch_index + length))
        else:
            out["batch_index"] = out["batch_index"][batch_index : batch_index + length]
        return cls.outputs(latent=out)


class GenerationLatentRepeat(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.repeat")

    @classmethod
    def execute(cls, *, samples: object, amount: int) -> Mapping[str, object]:
        _check_bounds(("amount", amount, 1, 64))
        torch = _torch()
        out, s_in = _plain_latent(samples, torch, "samples")
        out["samples"] = s_in.repeat((amount,) + (1,) * (s_in.ndim - 1))
        if "noise_mask" in out and out["noise_mask"].shape[0] > 1:
            mask = out["noise_mask"]
            out["noise_mask"] = mask.repeat((amount,) + (1,) * (mask.ndim - 1))
        if "batch_index" in out:
            batch_index = out["batch_index"]
            offset = max(batch_index) - min(batch_index) + 1
            out["batch_index"] = batch_index + [
                x + i * offset for i in range(1, amount) for x in batch_index
            ]
        return cls.outputs(latent=out)


class GenerationLatentSeedBehavior(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.seed_behavior")

    @classmethod
    def execute(cls, *, samples: object, behavior: str) -> Mapping[str, object]:
        _check_choice("behavior", behavior, ("random", "fixed"))
        torch = _torch()
        out, s_in = _plain_latent(samples, torch, "samples")
        if behavior == "random":
            out.pop("batch_index", None)
        else:
            out["batch_index"] = [out.get("batch_index", [0])[0]] * s_in.shape[0]
        return cls.outputs(latent=out)


class GenerationLatentBatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.batch")

    @classmethod
    def execute(cls, *, latents: Mapping[str, object]) -> Mapping[str, object]:
        torch = _torch()
        members = list(latents.values())
        if not members:
            raise ValueError("latents requires at least one member")
        out, first = _plain_latent(members[0], torch, "latents member 1")
        tensors: list[Any] = []
        batch_index: list[int] = []
        for position, member in enumerate(members, start=1):
            mapping, s_in = _plain_latent(member, torch, f"latents member {position}")
            tensors.append(_reshape_latent_to(first.shape, s_in, repeat_batch=False))
            batch_index.extend(mapping.get("batch_index", list(range(s_in.shape[0]))))
        out["samples"] = torch.cat(tensors, dim=0)
        out["batch_index"] = batch_index
        return cls.outputs(latent=out)


def _rebatch_entry(mapping: Mapping[Any, Any], samples: Any, offset: int, torch: Any) -> Any:
    """Port of LatentRebatch.get_batch at b78cec87 (nodes_rebatch.py)."""
    shape = samples.shape
    mask = mapping.get("noise_mask")
    if mask is None:
        mask = torch.ones((shape[0], 1, shape[2] * 8, shape[3] * 8), device="cpu")
    if mask.shape[0] < samples.shape[0]:
        mask = mask.repeat((shape[0] - 1) // mask.shape[0] + 1, 1, 1, 1)[: shape[0]]
    batch_inds = mapping.get("batch_index", [x + offset for x in range(shape[0])])
    return samples, mask, batch_inds


def _rebatch_slices(indexable: Any, num: int, batch_size: int) -> Any:
    """Port of LatentRebatch.get_slices at b78cec87."""
    slices = [indexable[i * batch_size : (i + 1) * batch_size] for i in range(num)]
    if num * batch_size < len(indexable):
        return slices, indexable[num * batch_size :]
    return slices, None


def _rebatch_slice_batch(batch: Any, num: int, batch_size: int) -> Any:
    """Port of LatentRebatch.slice_batch at b78cec87."""
    result = [_rebatch_slices(x, num, batch_size) for x in batch]
    return list(zip(*result, strict=True))


def _rebatch_cat(batch1: Any, batch2: Any, torch: Any) -> Any:
    """Port of LatentRebatch.cat_batch at b78cec87."""
    if batch1[0] is None:
        return batch2
    return [
        torch.cat((b1, b2)) if torch.is_tensor(b1) else b1 + b2
        for b1, b2 in zip(batch1, batch2, strict=True)
    ]


class GenerationLatentRebatch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.rebatch")

    @classmethod
    def execute(cls, *, latents: object, batch_size: int) -> Mapping[str, object]:
        _check_bounds(("batch_size", batch_size, 1, 4_096))
        torch = _torch()
        if not isinstance(latents, Sequence) or isinstance(latents, (str, bytes)):
            raise TypeError("latents must be a list of LATENT mappings")
        entries = [
            _plain_latent(item, torch, f"latents entry {position}")
            for position, item in enumerate(cast("Sequence[object]", latents), start=1)
        ]
        output_list: list[dict[Any, Any]] = []
        current = (None, None, None)
        processed = 0
        for mapping, samples in entries:
            next_batch = _rebatch_entry(mapping, samples, processed, torch)
            processed += len(next_batch[2])
            if current[0] is None:
                current = next_batch
            elif (
                next_batch[0].shape[-1] != current[0].shape[-1]
                or next_batch[0].shape[-2] != current[0].shape[-2]
            ):
                sliced, _ = _rebatch_slice_batch(current, 1, batch_size)
                output_list.append(
                    {
                        "samples": sliced[0][0],
                        "noise_mask": sliced[1][0],
                        "batch_index": sliced[2][0],
                    }
                )
                current = next_batch
            else:
                current = _rebatch_cat(current, next_batch, torch)
            if current[0].shape[0] > batch_size:
                num = current[0].shape[0] // batch_size
                sliced, remainder = _rebatch_slice_batch(current, num, batch_size)
                for i in range(num):
                    output_list.append(
                        {
                            "samples": sliced[0][i],
                            "noise_mask": sliced[1][i],
                            "batch_index": sliced[2][i],
                        }
                    )
                current = remainder
        if current[0] is not None:
            sliced, _ = _rebatch_slice_batch(current, 1, batch_size)
            output_list.append(
                {
                    "samples": sliced[0][0],
                    "noise_mask": sliced[1][0],
                    "batch_index": sliced[2][0],
                }
            )
        for item in output_list:
            if item["noise_mask"].mean().item() == 1.0:
                del item["noise_mask"]
        return cls.outputs(latents=output_list)


class GenerationLatentSetNoiseMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.set_noise_mask")

    @classmethod
    def execute(cls, *, samples: object, mask: object) -> Mapping[str, object]:
        torch = _torch()
        out, _ = _plain_latent(samples, torch, "samples")
        if type(mask) is not torch.Tensor:
            raise TypeError("mask must be an exact torch.Tensor")
        mask_tensor = cast("Any", mask)
        out["noise_mask"] = mask_tensor.reshape(
            (-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1])
        )
        return cls.outputs(latent=out)


class GenerationLatentReplaceFrames(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.replace_frames")

    @classmethod
    def execute(
        cls, *, destination: object, index: int, source: object = None
    ) -> Mapping[str, object]:
        _check_bounds(("index", index, -16_384, 16_384))
        if source is None:
            return cls.outputs(latent=destination)
        torch = _torch()
        _, dest = _plain_latent(destination, torch, "destination")
        out, src = _plain_latent(source, torch, "source")
        dest_frames = dest.shape[2]
        source_frames = src.shape[2]
        if index < 0:
            index = dest_frames + index
        if index > dest_frames:
            log.warning(
                "index %s is out of bounds for destination latent with %s frames",
                index,
                dest_frames,
            )
            return cls.outputs(latent=destination)
        if index + source_frames > dest_frames:
            log.warning(
                "source latent with %s frames at index %s does not fit destination "
                "latent with %s frames",
                source_frames,
                index,
                dest_frames,
            )
            return cls.outputs(latent=destination)
        merged = dest.clone()
        merged[:, :, index : index + source_frames] = src
        out["samples"] = merged
        return cls.outputs(latent=out)


def _apply_latent_operation(operation: _LatentOperationValue, latent: Any) -> Any:
    """Applies the torch-layer port of the LatentOperation* closures at b78cec87."""
    transforms = importlib.import_module("dinkster_inference_torch.guidance_transforms")
    return transforms.apply_latent_operation(operation.kind, operation.params, latent)


class GenerationLatentApplyOperation(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.apply_operation")

    @classmethod
    def execute(cls, *, samples: object, operation: object) -> Mapping[str, object]:
        if type(operation) is not _LatentOperationValue:
            raise TypeError("operation must be produced by a latent operation node")
        torch = _torch()
        out, s_in = _plain_latent(samples, torch, "samples")
        out["samples"] = _apply_latent_operation(operation, s_in)
        return cls.outputs(latent=out)


class GenerationLatentOperationTonemapReinhard(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.operation_tonemap_reinhard")

    @classmethod
    def execute(cls, *, multiplier: float) -> Mapping[str, object]:
        _check_bounds(("multiplier", multiplier, 0.0, 100.0))
        return cls.outputs(
            operation=_LatentOperationValue(
                "tonemap_reinhard", (("multiplier", float(multiplier)),)
            )
        )


class GenerationLatentOperationSharpen(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.operation_sharpen")

    @classmethod
    def execute(cls, *, sharpen_radius: int, sigma: float, alpha: float) -> Mapping[str, object]:
        _check_bounds(
            ("sharpen_radius", sharpen_radius, 1, 31),
            ("sigma", sigma, 0.1, 10.0),
            ("alpha", alpha, 0.0, 5.0),
        )
        return cls.outputs(
            operation=_LatentOperationValue(
                "sharpen",
                (
                    ("sharpen_radius", int(sharpen_radius)),
                    ("sigma", float(sigma)),
                    ("alpha", float(alpha)),
                ),
            )
        )


class GenerationLatentApplyOperationCFG(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.apply_operation_cfg")

    @classmethod
    def execute(cls, *, model: object, operation: object) -> Mapping[str, object]:
        if type(operation) is not _LatentOperationValue:
            raise TypeError("operation must be produced by a latent operation node")
        # The transform-chain position becomes the descriptor's order and a
        # distinct id, so the guidance registry's (order, id) sort reproduces
        # the reference's model_options append order for chained operations.
        _, _, _, _, _, transforms, _, _ = _native_model(model, "model")
        index = len(transforms)
        contribution = _guidance_transform_factory(
            "latent_operation",
            operation.kind,
            operation.params,
            descriptor_id=f"dinkster.latent-operation.{index}",
            order=index,
        )
        return cls.outputs(
            model=_model_with_guidance_transform(
                model, "dinkster.latent.apply_operation_cfg", contribution
            )
        )


class GenerationLatentGenerateNoise(Node):
    """Port of GenerateNoise (KJNodes nodes/nodes.py @ 3f200542): one
    float32 CPU draw of the whole selected shape from a privately
    seeded generator (same stream as the reference's global
    torch.manual_seed), sigma/multiplier scaling, then optional
    normalize and constant-batch repeat, in the reference's order."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.generate_noise")

    @classmethod
    def execute(
        cls,
        *,
        width: int,
        height: int,
        batch_size: int,
        seed: int,
        multiplier: float,
        constant_batch_noise: bool,
        normalize: bool,
        model: object | None = None,
        sigmas: object | None = None,
        latent_channels: str = "4",
        shape: str = "BCHW",
    ) -> Mapping[str, object]:
        _check_bounds(
            ("width", width, 16, 4_096),
            ("height", height, 16, 4_096),
            ("batch_size", batch_size, 1, 4_096),
            ("seed", seed, 0, KSampler.MAX_SEED),
            ("multiplier", multiplier, 0.0, 4_096.0),
        )
        _check_choice("latent_channels", latent_channels, ("4", "16"))
        _check_choice("shape", shape, ("BCHW", "BCTHW", "BTCHW"))
        torch = _torch()
        channels = int(latent_channels)
        if shape == "BCHW":
            size = [batch_size, channels, height // 8, width // 8]
        elif shape == "BCTHW":
            size = [1, channels, batch_size, height // 8, width // 8]
        else:
            size = [1, batch_size, channels, height // 8, width // 8]
        generator = torch.Generator("cpu")
        generator.manual_seed(seed)
        noise = torch.randn(
            size,
            dtype=torch.float32,
            layout=torch.strided,
            generator=generator,
            device="cpu",
        )
        if sigmas is not None:
            if type(sigmas) is not _CustomSigmasValue:
                raise TypeError("sigmas must come from a Dinkster sigma-schedule node")
            if model is None:
                raise ValueError("sigma-scaled noise generation requires the model input")
            if not sigmas.values:
                raise ValueError("sigmas must contain at least one value")
            inference = importlib.import_module("dinkster_inference")
            # Overlays and transforms are accepted but irrelevant: only the
            # family's latent scale factor participates, and no overlay
            # changes the latent space.
            handle = _native_model(model, "model")[0]
            descriptor = handle.runtime.family.latent
            if type(descriptor) is not inference.LatentDescriptor:
                raise ValueError("sigma-scaled noise generation requires a plain-latent family")
            # Float32 tensor arithmetic, matching the reference bitwise.
            sigmas_tensor = torch.tensor(sigmas.values, dtype=torch.float32)
            noise *= (sigmas_tensor[0] - sigmas_tensor[-1]) / descriptor.scale_factor
        noise *= multiplier
        if normalize:
            noise = noise / noise.std()
        if constant_batch_noise:
            # The reference's noise[0].repeat(batch_size, 1, 1, 1) only
            # produces a batch for the 4D BCHW layout; on the 5D shapes it
            # scrambles channels, so those combinations refuse instead.
            if shape != "BCHW":
                raise ValueError("constant_batch_noise requires the BCHW shape")
            noise = noise[0].repeat(batch_size, 1, 1, 1)
        return cls.outputs(latent={"samples": noise})


class GenerationLatentInjectNoise(Node):
    """Port of InjectNoiseToLatent (KJNodes nodes/nodes.py @ 3f200542),
    including its metadata-dropping output: only the combined samples
    survive, exactly like the reference's fresh {"samples"} return."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.latent.inject_noise")

    @classmethod
    def execute(
        cls,
        *,
        latents: object,
        strength: float,
        noise: object,
        normalize: bool,
        average: bool,
        mask: object | None = None,
        mix_randn_amount: float = 0.0,
        seed: int = 123,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("strength", strength, 0.0, 200.0),
            ("mix_randn_amount", mix_randn_amount, 0.0, 1_000.0),
            ("seed", seed, 0, KSampler.MAX_SEED),
        )
        torch = _torch()
        _, samples_in = _plain_latent(latents, torch, "latents")
        _, noise_in = _plain_latent(noise, torch, "noise")
        samples = samples_in.clone().cpu()
        noise_t = noise_in.clone().cpu()
        if average:
            noised = (samples + noise_t) / 2
        else:
            noised = samples + noise_t * strength
        if normalize:
            noised = noised / noised.std()
        if mask is not None:
            if type(mask) is not torch.Tensor:
                raise TypeError("mask must be an exact torch.Tensor")
            if noised.ndim != 4:
                raise ValueError("mask blending requires a rank-4 latent")
            mask_tensor = cast("Any", mask)
            blend = torch.nn.functional.interpolate(
                mask_tensor.reshape((-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1])),
                size=(noised.shape[2], noised.shape[3]),
                mode="bilinear",
            )
            blend = blend.expand((-1, noised.shape[1], -1, -1))
            if blend.shape[0] < noised.shape[0]:
                blend = blend.repeat((noised.shape[0] - 1) // blend.shape[0] + 1, 1, 1, 1)[
                    : noised.shape[0]
                ]
            noised = blend * noised + (1 - blend) * samples
        if mix_randn_amount > 0:
            generator = torch.Generator("cpu")
            generator.manual_seed(seed)
            rand_noise = torch.randn(
                noised.size(),
                dtype=noised.dtype,
                layout=noised.layout,
                generator=generator,
                device="cpu",
            )
            noised = noised + (mix_randn_amount * rand_noise)
        return cls.outputs(latent={"samples": noised})


GENERATION_PROVIDER_NODES: tuple[type[Node], ...] = (
    GenerationEmptyTrellis2LatentStructure,
    GenerationTrellis2Conditioning,
    GenerationPixal3DConditioning,
    GenerationVaeDecodeStructureTrellis2,
    GenerationTrellis2ShapeStage,
    GenerationTrellis2UpsampleStage,
    GenerationVaeDecodeShapeTrellis,
    GenerationTrellis2TextureStage,
    GenerationVaeDecodeTextureTrellis,
    GenerationLoadGeometryModel,
    GenerationEstimateGeometry,
    GenerationGeometryToFOV,
    GenerationLoadBackgroundRemoval,
    GenerationRemoveBackground,
    GenerationImageCropToMask,
    GenerationPreviewMask,
    GenerationVoxelToMesh,
    GenerationGetMeshInfo,
    GenerationRemeshMesh,
    GenerationDecimateMesh,
    GenerationSmoothMeshNormals,
    GenerationUnwrapMesh,
    GenerationPaintMesh,
    GenerationBakeTextureFromVoxel,
    GenerationBakeNormalMapFromMesh,
    GenerationBakeAmbientOcclusion,
    GenerationRenderUVAtlas,
    GenerationApplyTextureToMesh,
    GenerationMeshToModel3D,
    NativeLoadModelProfile,
    GenerationLoadCheckpoint,
    GenerationLoadControlNet,
    GenerationApplyControlNet,
    GenerationApplyControlNetAdvanced,
    GenerationSetControlNetUnionType,
    GenerationLoadCheckpointStack,
    GenerationLoadDiffusionModel,
    GenerationLoadDiffusionComponents,
    GenerationLoadLTXAVTextEncoder,
    GenerationLoadLTXAVAudioVAE,
    GenerationLoadLatentUpscaleModel,
    GenerationLTXAVAudioVAEDecode,
    GenerationLoadLora,
    GenerationLoadLoraModelOnly,
    GenerationApplyLoraStack,
    GenerationApplyLoraStackModelOnly,
    GenerationClipTextEncode,
    GenerationClipTextEncodeLumina2,
    GenerationModelSamplingAuraFlow,
    GenerationTextGenerate,
    GenerationPromptEnhance,
    GenerationClipSetLastLayer,
    GenerationT5TokenizerOptions,
    GenerationClipTextEncodeControlnet,
    GenerationFluxGuidance,
    GenerationFluxDisableGuidance,
    GenerationReferenceLatent,
    GenerationCfgZeroStar,
    GenerationCfgNorm,
    GenerationTCFG,
    GenerationFreSca,
    GenerationLazyCache,
    GenerationEasyCache,
    GenerationAttentionSchedule,
    GenerationContextWindowsManual,
    GenerationWanContextWindowsManual,
    GenerationLTXVContextWindows,
    GenerationAdaptiveProjectedGuidance,
    GenerationMahiroGuidance,
    GenerationEpsilonScaling,
    GenerationCFGOverride,
    GenerationRescaleCfg,
    GenerationRenormCfg,
    GenerationTemporalScoreRescaling,
    GenerationNormalizedAttentionGuidance,
    GenerationLTXAVConditioning,
    GenerationLTXAVReferenceAudio,
    GenerationLTXAVIDLoRAReferenceAudio,
    GenerationLTXVSpatioTemporalGuidance,
    GenerationLTXVModalityGuidance,
    GenerationLTXVDurationPredictor,
    GenerationLTXVDualCFGGuider,
    GenerationLTXVConditioning,
    GenerationLTXVImageToVideo,
    GenerationLTXVImageToVideoInplace,
    GenerationLTXVAddGuide,
    GenerationLTXVCropGuides,
    GenerationLTXVLatentUpsampler,
    GenerationConditioningMerge,
    GenerationConditioningScale,
    GenerationConditioningSetArea,
    GenerationConditioningSetMask,
    GenerationConditioningSetTimestepRange,
    GenerationConditioningZeroOut,
    GenerationChromaRadianceOptions,
    GenerationChromaModelSampling,
    GenerationModelSamplingSD3,
    GenerationModelSamplingLTXV,
    GenerationModelSamplingFlux,
    GenerationEmptyLatentImage,
    GenerationEmptySD3LatentImage,
    GenerationEmptyChromaRadianceLatentImage,
    GenerationEmptyFlux2LatentImage,
    GenerationEmptyLTXAVLatent,
    GenerationEmptyLTXVLatent,
    GenerationKSampler,
    GenerationKSamplerAdvanced,
    GenerationKSamplerSelect,
    GenerationSamplerDPMPP3MSDE,
    GenerationSamplerDPMPP2MSDE,
    GenerationSamplerDPMPPSDE,
    GenerationSamplerDPMPP2SAncestral,
    GenerationSamplerEulerAncestral,
    GenerationSamplerEulerAncestralCFGPP,
    GenerationSamplerLMS,
    GenerationSamplerDPMAdaptative,
    GenerationSamplerERSDE,
    GenerationSamplerSEEDS2,
    GenerationSamplerSASolver,
    GenerationBasicScheduler,
    GenerationBetaSamplingScheduler,
    GenerationSDTurboScheduler,
    GenerationKarrasScheduler,
    GenerationExponentialScheduler,
    GenerationPolyexponentialScheduler,
    GenerationLaplaceScheduler,
    GenerationVPScheduler,
    GenerationAlignYourStepsScheduler,
    GenerationGITSScheduler,
    GenerationOptimalStepsScheduler,
    GenerationFlux2Scheduler,
    GenerationIdeogram4Scheduler,
    GenerationManualSigmas,
    GenerationSplitSigmas,
    GenerationSplitSigmasDenoise,
    GenerationFlipSigmas,
    GenerationSetFirstSigma,
    GenerationExtendIntermediateSigmas,
    GenerationSamplingPercentToSigma,
    GenerationBasicGuider,
    GenerationCFGGuider,
    GenerationDualCFGGuider,
    GenerationDualModelGuider,
    GenerationScheduledCFGGuider,
    GenerationPerpNegGuider,
    GenerationDisableCFG1Optimization,
    GenerationDisableNoise,
    GenerationRandomNoise,
    GenerationAddNoise,
    GenerationSamplerCustom,
    GenerationSamplerCustomAdvanced,
    GenerationImpactRegionalSampler,
    GenerationVAEDecode,
    GenerationVAEDecodeTiled,
    GenerationVAEEncode,
    GenerationVAEEncodeTiled,
    GenerationSeedVR2Preprocess,
    GenerationSeedVR2PostProcessing,
    GenerationSeedVR2Conditioning,
    GenerationSeedVR2TemporalChunk,
    GenerationSeedVR2TemporalMerge,
    GenerationLatentCombine,
    GenerationLatentMix,
    GenerationLatentMultiply,
    GenerationLatentRotate,
    GenerationLatentFlip,
    GenerationLatentCrop,
    GenerationLatentResize,
    GenerationLatentResizeBy,
    GenerationLatentComposite,
    GenerationLatentCompositeMasked,
    GenerationLatentConcat,
    GenerationLatentCut,
    GenerationLatentCutToBatch,
    GenerationLatentFromBatch,
    GenerationLatentRepeat,
    GenerationLatentSeedBehavior,
    GenerationLatentBatch,
    GenerationLatentRebatch,
    GenerationLatentSetNoiseMask,
    GenerationLatentReplaceFrames,
    GenerationLatentApplyOperation,
    GenerationLatentOperationTonemapReinhard,
    GenerationLatentOperationSharpen,
    GenerationLatentApplyOperationCFG,
    GenerationLatentGenerateNoise,
    GenerationLatentInjectNoise,
)


NATIVE_SCHEDULING_NODES: tuple[type[Node], ...] = tuple(
    cast(type[Node], globals()[name]) for name in NATIVE_SCHEDULING_NODE_TYPES
)


NATIVE_ARM_NODES: tuple[type[Node], ...] = (
    *NATIVE_SCHEDULING_NODES,
    GenerationClipTextEncodeLumina2,
    GenerationModelSamplingAuraFlow,
    NativeLoadModelProfile,
    NativeLoadClip,
    NativeLoadDualClip,
    NativeLoadVae,
    NativeLoadVision,
    GenerationLoadDiffusionModel,
    GenerationLoadDiffusionComponents,
    GenerationEmptyTrellis2LatentStructure,
    GenerationTrellis2Conditioning,
    GenerationPixal3DConditioning,
    GenerationVaeDecodeStructureTrellis2,
    GenerationTrellis2ShapeStage,
    GenerationTrellis2UpsampleStage,
    GenerationVaeDecodeShapeTrellis,
    GenerationTrellis2TextureStage,
    GenerationVaeDecodeTextureTrellis,
    GenerationLoadGeometryModel,
    GenerationEstimateGeometry,
    GenerationGeometryToFOV,
    GenerationLoadBackgroundRemoval,
    GenerationRemoveBackground,
    GenerationImageCropToMask,
    GenerationPreviewMask,
    GenerationVoxelToMesh,
    GenerationGetMeshInfo,
    GenerationRemeshMesh,
    GenerationDecimateMesh,
    GenerationSmoothMeshNormals,
    GenerationUnwrapMesh,
    GenerationPaintMesh,
    GenerationBakeTextureFromVoxel,
    GenerationBakeNormalMapFromMesh,
    GenerationBakeAmbientOcclusion,
    GenerationRenderUVAtlas,
    GenerationApplyTextureToMesh,
    GenerationMeshToModel3D,
    NativeEmptyMiniMaxH3AV,
    NativeEmptyMiniMaxMusic3LatentAudio,
    NativeMiniMaxMusic3TextEncode,
    NativeVAEDecodeAudio,
    NativeVAEDecodeAudioTiled,
    NativeSetLatentMaskFromFrames,
    NativeSetLatentMaskFromTimeRanges,
    NativeInspectLatentMask,
    NativeMiniMaxH3T2VAConditioning,
    NativeMiniMaxH3FL2VAConditioning,
    NativeMiniMaxH3REF2VAConditioning,
    NativeMiniMaxH3AddGuide,
    NativeMiniMaxH3MotionContext,
    NativeMiniMaxH3AVEncode,
    NativeMiniMaxH3AVDecode,
    NativeConcatAVLatent,
    NativeSeparateAVLatent,
    NativePreviewLatentVisual,
    NativePreviewLatentAudio,
    NativeWan21ClipVisionEncode,
    NativeBerniniConditioning,
    NativeWan21ImageToVideo,
    NativeWanCameraEmbedding,
    NativeWanCameraImageToVideo,
    NativeWanPhantomSubjectToVideo,
    NativeWanTrackToVideo,
    NativeWanMoveTracksFromCoords,
    NativeWanMoveConcatTrack,
    NativeWanMoveGenerateTracks,
    NativeWanMoveVisualizeTracks,
    NativeWanMoveTrackToVideo,
    NativeWanFirstLastFrameToVideo,
    NativeWanFunControlToVideo,
    NativeWan22FunControlToVideo,
    NativeWanFunInpaintToVideo,
    NativeWanVaceToVideo,
    NativeWan22ImageToVideoLatent,
    NativeLoadZImageControlPatch,
    NativeApplyZImageControlPatch,
    # Resident component handles carry producer-arm affinity: a consumer must
    # execute on the arm that produced its handle. dinkster.load_clip and
    # dinkster.load_vae build resident component handles on this arm, so their
    # generic consumers serve here with the same bodies as the default arm.
    GenerationClipSetLastLayer,
    GenerationT5TokenizerOptions,
    GenerationClipTextEncode,
    GenerationTextGenerate,
    GenerationPromptEnhance,
    GenerationClipTextEncodeControlnet,
    GenerationVAEDecode,
    GenerationVAEDecodeTiled,
    GenerationVAEEncode,
    GenerationCFGOverride,
    GenerationRescaleCfg,
    GenerationModelSamplingSD3,
    GenerationModelSamplingLTXV,
    GenerationModelSamplingFlux,
    GenerationKSampler,
    GenerationKSamplerAdvanced,
    GenerationSamplerSASolver,
    GenerationBasicScheduler,
    GenerationBetaSamplingScheduler,
    GenerationSDTurboScheduler,
    GenerationSamplingPercentToSigma,
    GenerationBasicGuider,
    GenerationLazyCache,
    GenerationEasyCache,
    GenerationAttentionSchedule,
    GenerationCFGGuider,
    GenerationDualCFGGuider,
    GenerationDualModelGuider,
    GenerationScheduledCFGGuider,
    GenerationPerpNegGuider,
    GenerationDisableCFG1Optimization,
    GenerationAddNoise,
    GenerationSamplerCustom,
    GenerationSamplerCustomAdvanced,
    GenerationImpactRegionalSampler,
    GenerationLTXAVAudioVAEDecode,
    GenerationVAEEncodeTiled,
    GenerationLTXAVReferenceAudio,
    GenerationLTXAVIDLoRAReferenceAudio,
    GenerationLTXVSpatioTemporalGuidance,
    GenerationLTXVModalityGuidance,
    GenerationLTXVDurationPredictor,
    GenerationLTXVDualCFGGuider,
    GenerationLTXVImageToVideo,
    GenerationLTXVImageToVideoInplace,
    GenerationLTXVAddGuide,
    GenerationLTXVCropGuides,
    GenerationLTXVLatentUpsampler,
    GenerationSeedVR2Conditioning,
)

__all__ = [
    "GENERATION_PROVIDER_NODES",
    "NATIVE_ARM_NODES",
    "NativeClipTextEncode",
    "NativeConcatAVLatent",
    "NativeControlNetApply",
    "NativeControlNetApplyAdvanced",
    "NativeControlNetLoader",
    "NativeConditioningSetPropertiesAndCombine",
    "NativeConditioningTimestepsRange",
    "NativeCreateHookKeyframe",
    "NativeCreateHookLora",
    "NativeEmptyLTXAVLatent",
    "NativeEmptyLTXVLatent",
    "NativeEmptyMiniMaxH3AV",
    "NativeEmptyMiniMaxMusic3LatentAudio",
    "NativeMiniMaxH3AddGuide",
    "NativeMiniMaxH3MotionContext",
    "NativeKSampler",
    "NativeKSamplerAdvanced",
    "NativeLoadCheckpoint",
    "NativeLoadModelProfile",
    "NativeLoadClip",
    "NativeLoadDualClip",
    "NativeLoadDiffusionModel",
    "NativeLoadVae",
    "NativeLoadVision",
    "NativeLoadZImageControlPatch",
    "NativeApplyZImageControlPatch",
    "NativeInspectLatentMask",
    "NativeLoadLora",
    "NativeLoadLoraModelOnly",
    "NativeMiniMaxH3AVDecode",
    "NativeMiniMaxH3AVEncode",
    "NativeMiniMaxH3FL2VAConditioning",
    "NativeMiniMaxH3REF2VAConditioning",
    "NativeMiniMaxH3T2VAConditioning",
    "NativeMiniMaxMusic3TextEncode",
    "NativePairConditioningSetProperties",
    "NativePreviewLatentAudio",
    "NativePreviewLatentVisual",
    "NativeRuntimeHandle",
    "NativeSetHookKeyframes",
    "NativeSetLatentMaskFromFrames",
    "NativeSetLatentMaskFromTimeRanges",
    "NativeSeparateAVLatent",
    "NativeVAEDecode",
    "NativeVAEDecodeAudio",
    "NativeVAEDecodeAudioTiled",
    "NativeVAEEncode",
    "NativeWan21ClipVisionEncode",
    "NativeBerniniConditioning",
    "NativeWan21ImageToVideo",
    "NativeWanCameraEmbedding",
    "NativeWanCameraImageToVideo",
    "NativeWanMoveConcatTrack",
    "NativeWanMoveGenerateTracks",
    "NativeWanMoveTracksFromCoords",
    "NativeWanMoveTrackToVideo",
    "NativeWanMoveVisualizeTracks",
    "NativeWanPhantomSubjectToVideo",
    "NativeWanTrackToVideo",
    "NativeWanFunControlToVideo",
    "NativeWan22FunControlToVideo",
    "NativeWanFunInpaintToVideo",
    "NativeWanFirstLastFrameToVideo",
    "NativeWanVaceToVideo",
    "NativeWan22ImageToVideoLatent",
    "load_native_runtime_handle",
]
