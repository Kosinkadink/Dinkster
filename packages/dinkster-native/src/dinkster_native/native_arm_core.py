# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnusedFunction=false

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
    pass

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


def _control_resident_refs(*values: object) -> tuple[object, ...]:
    from .native_arm_runtime import _control_resident_refs as resolve_refs

    return resolve_refs(*values)


__all__ = [
    "ALIGN_YOUR_STEPS_NOISE_LEVELS",
    "AcceleratorMemoryPolicyError",
    "Any",
    "ApplyZImageControlPatch",
    "AssetRef",
    "AssetWidget",
    "BerniniConditioning",
    "CLIPTextEncode",
    "Callable",
    "ComboWidget",
    "ConcatAVLatent",
    "ControlNetApply",
    "ControlNetApplyAdvanced",
    "ControlNetLoader",
    "DynamicComboOption",
    "DynamicComboSpec",
    "EmbeddingNameIndex",
    "EmptyLTXAVLatent",
    "EmptyLTXVLatent",
    "EmptyLatentImage",
    "EmptyMiniMaxH3AV",
    "EmptyMiniMaxMusic3LatentAudio",
    "ExitStack",
    "GENERATION_NODES",
    "GITS_NOISE_LEVELS",
    "InputFamilySpec",
    "InputSpec",
    "InspectLatentMask",
    "KSampler",
    "KSamplerAdvanced",
    "LoadCheckpoint",
    "LoadClip",
    "LoadDiffusionModel",
    "LoadDualClip",
    "LoadLora",
    "LoadLoraModelOnly",
    "LoadVae",
    "LoadVision",
    "LoadZImageControlPatch",
    "MODEL3D_GENERATION_NODES",
    "Mapping",
    "MappingSource",
    "MiniMaxH3AVDecode",
    "MiniMaxH3AVEncode",
    "MiniMaxH3AddGuide",
    "MiniMaxH3AudioReferenceValue",
    "MiniMaxH3FL2VAConditioning",
    "MiniMaxH3ImageReferenceValue",
    "MiniMaxH3MotionContext",
    "MiniMaxH3REF2VAConditioning",
    "MiniMaxH3T2VAConditioning",
    "MiniMaxH3VideoReferenceValue",
    "MiniMaxMusic3TextEncode",
    "NATIVE_SCHEDULING_NODE_TYPES",
    "NativeComponentHandle",
    "NativeConditioningSetPropertiesAndCombine",
    "NativeConditioningTimestepsRange",
    "NativeCreateHookKeyframe",
    "NativeCreateHookLora",
    "NativePairConditioningSetProperties",
    "NativeResidencyBusyError",
    "NativeResidencyPool",
    "NativeRuntimeHandle",
    "NativeSetHookKeyframes",
    "Node",
    "NodeSchema",
    "NoiseSelection",
    "NumberWidget",
    "OPTIMAL_STEPS_NOISE_LEVELS",
    "OutputInterface",
    "OutputSpec",
    "ParamSpec",
    "Path",
    "PreviewLatentAudio",
    "PreviewLatentVisual",
    "ReplacementCase",
    "ReplacementRule",
    "ResidencyRouteFacts",
    "SCHEDULED_HOOKS_KEY",
    "SamplerSelection",
    "ScheduledHookKeyframes",
    "ScheduledHooks",
    "ScheduledLoraHook",
    "SeparateAVLatent",
    "Sequence",
    "SetLatentMaskFromFrames",
    "SetLatentMaskFromTimeRanges",
    "SigmaSchedule",
    "StringWidget",
    "TYPE_CHECKING",
    "TypeExpr",
    "VAEDecode",
    "VAEDecodeAudio",
    "VAEDecodeAudioTiled",
    "VAEEncode",
    "WAN_CAMERA_POSES",
    "Wan21ClipVisionEncode",
    "Wan21ImageToVideo",
    "Wan22FunControlToVideo",
    "Wan22ImageToVideoLatent",
    "WanCameraEmbedding",
    "WanCameraImageToVideo",
    "WanFirstLastFrameToVideo",
    "WanFunControlToVideo",
    "WanFunInpaintToVideo",
    "WanMoveConcatTrack",
    "WanMoveGenerateTracks",
    "WanMoveTrackToVideo",
    "WanMoveTracksFromCoords",
    "WanMoveVisualizeTracks",
    "WanPhantomSubjectToVideo",
    "WanTrackToVideo",
    "WanVaceToVideo",
    "_AIMDO_HEADROOM_TARGET_ENV",
    "_ASSET",
    "_BOOLEAN",
    "_COMBO",
    "_COMFY_AUDIO",
    "_COMFY_LATENT_UPSCALE_MODEL",
    "_COMFY_MESH",
    "_COMFY_MODEL_PATCH",
    "_COMFY_SHAPE_SUBDIVIDES",
    "_COMFY_VOXEL",
    "_CONDITIONING",
    "_CONTROLNET_POOLED_METADATA_KEY",
    "_CONTROLNET_TEXT_METADATA_KEY",
    "_CustomGuiderValue",
    "_CustomNoiseValue",
    "_CustomSamplerValue",
    "_CustomSigmasValue",
    "_DINKSTER_CLIP",
    "_DINKSTER_CLIP_VISION",
    "_DINKSTER_CONDITIONING",
    "_DINKSTER_GUIDER",
    "_DINKSTER_IMAGE",
    "_DINKSTER_LATENT",
    "_DINKSTER_LATENT_LIST",
    "_DINKSTER_LATENT_OPERATION",
    "_DINKSTER_MASK",
    "_DINKSTER_MODEL",
    "_DINKSTER_NOISE",
    "_DINKSTER_SAMPLER",
    "_DINKSTER_SIGMAS",
    "_DINKSTER_VAE",
    "_DISABLE_CFG1_OPTIMIZATION",
    "_DualCFGGuiderValue",
    "_DualModelGuiderValue",
    "_EMBEDDING_FREE_RUNTIME_SOURCE_ROLE_SETS",
    "_FLOAT",
    "_FLUX2_REFERENCE_LATENTS_KEY",
    "_FLUX_GUIDANCE_DISABLED",
    "_FLUX_GUIDANCE_METADATA_KEY",
    "_GuidedRows",
    "_HOOKS",
    "_HOOK_KEYFRAMES",
    "_INT",
    "_LTXAVDualGuiderValue",
    "_LTX_FRAME_RATE_METADATA_KEY",
    "_LatentOperationValue",
    "_MASK",
    "_MODEL3D_PROVIDER_SCHEMAS",
    "_NATIVE_HOOKS_KEY",
    "_NATIVE_MASK_BOUNDS_KEY",
    "_NATIVE_MASK_KEY",
    "_NATIVE_PREPARED_CONDITIONING_KEY",
    "_NATIVE_PROMPT_KEY",
    "_NativeClipOptions",
    "_NativeHookKeyframes",
    "_NativeHooks",
    "_NativeLoraHook",
    "_PerpNegGuiderValue",
    "_RUNTIME_SOURCE_ROLE_SETS",
    "_STRING",
    "_TIMESTEPS_RANGE",
    "_active_inference_registries",
    "_builtin_inference_registries",
    "_component_bound_carrier",
    "_condition_entries",
    "_control_resident_refs",
    "_direct_vae_batch_size",
    "_effective_flux_guidance",
    "_embedding_resource",
    "_freeze_embedding_resource",
    "_inference_registries",
    "_is_accelerator_oom",
    "_native_clip_options",
    "_native_schema",
    "_not_cancelled",
    "_retry_tiled_vae_after_oom",
    "_run_direct_vae",
    "_sampler_registry",
    "_set_native_conditioning_properties",
    "_split_flux_guidance",
    "_split_ltx_frame_rate",
    "_torch",
    "_with_flux_guidance",
    "_with_ltx_frame_rate",
    "align_your_steps_sigmas",
    "annotations",
    "cast",
    "clean_enhanced_prompt",
    "comfy_type_id",
    "contextmanager",
    "current_execution_context",
    "current_native_observer",
    "dataclass",
    "default_dependency_residency_pool",
    "default_native_residency",
    "default_pool",
    "gits_sigmas",
    "hashlib",
    "importlib",
    "inspect",
    "json",
    "log",
    "logging",
    "math",
    "multistream_sampling_preview_emitter",
    "native_execution_span",
    "native_memory_policy",
    "nullcontext",
    "optimal_steps_sigmas",
    "os",
    "partial",
    "platform",
    "prepare_ltx2_prompt",
    "preview_stage",
    "re",
    "replace",
    "report_progress",
    "resolve_weight_source",
    "resolver_from_env",
    "sampler_wire_value",
    "sampling_preview_emitter",
    "select_current_device",
    "select_load_device",
    "uses_classic_embedding_bindings",
    "validate_lora_execution_mode",
    "wraps",
]
