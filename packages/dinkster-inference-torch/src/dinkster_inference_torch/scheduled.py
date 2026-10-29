"""Schedule-aware text encoding into canonical DMFC conditioning carriers."""

# This module intentionally adapts the private encoder members of the two
# concrete runtimes. Ordinary FamilyRuntime remains unchanged.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, Protocol, TypeVar, cast

import torch
from dinkster_inference import (
    EMPTY_RANGE,
    Conditioning,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRange,
    ConditioningRecord,
    ConditioningSet,
    EncoderStream,
    ExtensionInputValue,
    InferenceTypeRegistry,
    ModelFamily,
    PatchTargetComponent,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    ScheduledEncodeRequest,
    ScheduledEncodingError,
    ScheduledExecution,
    ScheduledPrompt,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    TransformTarget,
    VariantKey,
    encode_conditioning_carrier,
    intersect_ranges,
    make_conditioning_carrier,
    parse_prompt_weights,
    patch_overlay_stack_digest,
    scheduled_metadata,
)
from dinkster_inference.scheduled import register_scheduled_producer_types

from .clip_text import compose_sdxl_conditioning
from .payloads import tensor_to_payload_binding
from .t5_text import compose_flux_conditioning


class _Runtime(Protocol):
    @property
    def runtime_identity(self) -> str: ...

    @property
    def family(self) -> object: ...

    def encode_text(self, text: str) -> Conditioning[torch.Tensor]: ...


class _RuntimeParts(_Runtime, Protocol):
    _ovis_encoder: Any | None
    _t5_encoder: Any
    _t5_tokenizer: Any
    _clip_encoder: Any
    _clip_tokenizer: Any
    _tokenizer: Any
    _clip_l_encoder: Any
    _clip_g_encoder: Any


_FULL = PercentRange(0.0, 1.0)


class _ScheduledEncodingCancelled(ScheduledEncodingError):
    pass


def _check_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise _ScheduledEncodingCancelled("scheduled text encoding cancelled")


def _family_id(runtime: object) -> str:
    family_id = getattr(getattr(runtime, "family", None), "id", None)
    if not isinstance(family_id, str):
        raise ScheduledEncodingError("scheduled runtime has no family identity")
    return family_id


def _expected_streams(runtime: object) -> tuple[EncoderStream, ...]:
    if getattr(runtime, "_ovis_encoder", None) is not None:
        return (EncoderStream.OVIS_QWEN3_2B,)
    family = cast("ModelFamily", cast("_Runtime", runtime).family)
    return tuple(
        EncoderStream.from_encoder_id(encoder_id) for encoder_id in family.wiring.text_encoders
    )


def _validate_prompt(runtime: object, prompt: ScheduledPrompt) -> None:
    actual = tuple(route.stream for route in prompt.routes)
    expected = _expected_streams(runtime)
    if actual != expected:
        raise ScheduledEncodingError(
            f"family {_family_id(runtime)!r} requires scheduled routes "
            f"{tuple(item.value for item in expected)!r}, got "
            f"{tuple(item.value for item in actual)!r}"
        )
    if expected == (EncoderStream.OVIS_QWEN3_2B,):
        for _segment, weight in parse_prompt_weights(prompt.routes[0].prompt):
            if not math_is_unit(weight):
                raise ScheduledEncodingError(
                    "scheduled Ovis prompts do not support non-unit weights"
                )


def math_is_unit(value: float) -> bool:
    return value == 1.0


def _ordinary_prompt(runtime: object, prompt: ScheduledPrompt) -> str | None:
    _validate_prompt(runtime, prompt)
    values = tuple(route.prompt for route in prompt.routes)
    if values and all(value == values[0] for value in values):
        return values[0]
    return None


def _transform_streams(runtime: object, target: TransformTarget) -> tuple[EncoderStream, ...]:
    streams = _expected_streams(runtime)
    if streams == (EncoderStream.OVIS_QWEN3_2B,):
        if target is TransformTarget.POOLED:
            raise ScheduledEncodingError("Ovis has no pooled transform lane")
        return streams
    if streams == (EncoderStream.CLIP_L, EncoderStream.T5):
        return (EncoderStream.T5,) if target is TransformTarget.TEXT else (EncoderStream.CLIP_L,)
    if streams == (EncoderStream.CLIP_L, EncoderStream.CLIP_G):
        return streams if target is TransformTarget.TEXT else (EncoderStream.CLIP_G,)
    return streams


def _encode_routes(runtime: object, prompt: ScheduledPrompt) -> Conditioning[torch.Tensor]:
    """Use only existing encoder/tokenizer semantics for each exact stream."""
    _validate_prompt(runtime, prompt)
    routes = {route.stream: route.prompt for route in prompt.routes}
    expected = _expected_streams(runtime)
    parts = cast("_RuntimeParts", runtime)
    if expected == (EncoderStream.OVIS_QWEN3_2B,):
        encoder = parts._ovis_encoder
        assert encoder is not None
        return encoder.encode(routes[EncoderStream.OVIS_QWEN3_2B])
    if expected == (EncoderStream.CLIP_L, EncoderStream.T5):
        t5 = parts._t5_encoder.encode(parts._t5_tokenizer.tokenize(routes[EncoderStream.T5]))
        clip_l = parts._clip_encoder.encode(
            parts._clip_tokenizer.tokenize(routes[EncoderStream.CLIP_L])
        )
        return compose_flux_conditioning(t5, clip_l)
    tokenizer = parts._tokenizer
    clip_l = None
    if EncoderStream.CLIP_L in routes:
        clip_l = parts._clip_l_encoder.encode(tokenizer.tokenize(routes[EncoderStream.CLIP_L]))
    clip_g = None
    if EncoderStream.CLIP_G in routes:
        clip_g = parts._clip_g_encoder.encode(tokenizer.tokenize(routes[EncoderStream.CLIP_G]))
    if clip_l is not None and clip_g is not None:
        return compose_sdxl_conditioning(clip_l, clip_g)
    result = clip_l if clip_l is not None else clip_g
    assert result is not None
    return result


def _layout(runtime: object, conditioning: Conditioning[torch.Tensor]) -> TokenLayoutDescriptor:
    streams = tuple(item.value for item in _expected_streams(runtime))
    token_count = int(conditioning.embeddings.shape[1])
    if streams == (EncoderStream.CLIP_L.value, EncoderStream.T5.value):
        # Flux carries only the T5 sequence in TEXT. CLIP-L contributes
        # the separate POOLED lane and therefore is not a token segment.
        segments = (TokenSegmentDescriptor("t5", "t5", 0, token_count),)
    else:
        segments = tuple(
            TokenSegmentDescriptor(stream, stream, 0, token_count) for stream in streams
        )
    return TokenLayoutDescriptor(
        family_id=_family_id(runtime),
        version=1,
        text_streams=streams,
        segments=segments,
    )


def _carrier(
    items: tuple[
        tuple[
            Conditioning[torch.Tensor],
            ConditioningRange,
            TokenLayoutDescriptor,
            tuple[tuple[str, ExtensionInputValue], ...],
        ],
        ...,
    ],
) -> ConditioningCarrier:
    records: list[ConditioningRecord] = []
    bindings = []
    for index, (conditioning, schedule, layout, metadata) in enumerate(items):
        text_id = f"scheduled-{index}-text"
        text = tensor_to_payload_binding(
            text_id, conditioning.embeddings, space="conditioning-text"
        )
        bindings.append(text)
        channels: list[tuple[ConditioningChannel, PayloadDescriptor]] = [
            (
                ConditioningChannel.TEXT,
                PayloadDescriptor(PayloadReference(text_id), text.shape, text.dtype, text.space),
            )
        ]
        if conditioning.pooled is not None:
            pooled_id = f"scheduled-{index}-pooled"
            pooled = tensor_to_payload_binding(
                pooled_id, conditioning.pooled, space="conditioning-pooled"
            )
            bindings.append(pooled)
            channels.append(
                (
                    ConditioningChannel.POOLED,
                    PayloadDescriptor(
                        PayloadReference(pooled_id),
                        pooled.shape,
                        pooled.dtype,
                        pooled.space,
                    ),
                )
            )
        records.append(
            ConditioningRecord(
                channels=tuple(channels),
                schedule=schedule,
                token_layout=layout,
                extension_metadata=metadata,
            )
        )
    carrier = make_conditioning_carrier(ConditioningSet(tuple(records)), bindings)
    encode_conditioning_carrier(carrier)
    return carrier


def _require_compatible_transform_result(
    descriptor_id: str, before: torch.Tensor, after: object
) -> torch.Tensor:
    if not isinstance(after, torch.Tensor):
        raise ScheduledEncodingError(f"transform {descriptor_id!r} returned a non-tensor")
    if after.shape != before.shape:
        raise ScheduledEncodingError(f"transform {descriptor_id!r} changed tensor shape")
    if after.dtype != before.dtype:
        raise ScheduledEncodingError(f"transform {descriptor_id!r} changed tensor dtype")
    if after.device != before.device:
        raise ScheduledEncodingError(f"transform {descriptor_id!r} changed tensor device")
    return after


def ordinary_conditioning_carrier(runtime: _Runtime, text: str) -> ConditioningCarrier:
    """Canonical ordinary baseline used by the scheduled-equivalence proof."""
    condition = runtime.encode_text(text)
    return _carrier(((condition, _FULL, _layout(runtime, condition), ()),))


StateT = TypeVar("StateT")


def _states(values: tuple[StateT, ...]) -> tuple[StateT | None, ...]:
    return values if values else (None,)


def encode_text_scheduled(
    runtime: _Runtime,
    request: ScheduledEncodeRequest,
    *,
    execution: ScheduledExecution[object] | None = None,
    transforms: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    cancelled: Callable[[], bool] | None = None,
    type_registry: InferenceTypeRegistry,
) -> ConditioningCarrier:
    """Transactionally evaluate one normalized scheduled request."""
    if not isinstance(cast("object", request), ScheduledEncodeRequest):
        raise TypeError("request must be ScheduledEncodeRequest")
    check = cancelled or (lambda: False)
    transform_fns = transforms or {}
    for prompt in request.prompts:
        _validate_prompt(runtime, prompt)
    for stack in request.transform_stacks:
        for descriptor in stack.transforms:
            if descriptor.family_id != _family_id(runtime):
                raise ScheduledEncodingError(
                    f"transform {descriptor.id!r} targets incompatible family "
                    f"{descriptor.family_id!r}"
                )
            expected_streams = _transform_streams(runtime, descriptor.target)
            if descriptor.text_streams != expected_streams:
                raise ScheduledEncodingError(
                    f"transform {descriptor.id!r} targets incompatible "
                    f"{descriptor.target.value} layout"
                )
            if descriptor.id not in transform_fns:
                raise ScheduledEncodingError(
                    f"transform {descriptor.id!r} has no worker-local implementation"
                )
    if (request.text_patches or request.diffusion_patches) and execution is None:
        raise ScheduledEncodingError(
            "nonempty scheduled patch stacks require a scheduled execution owner"
        )
    acquired: list[VariantKey] = []
    outputs: list[
        tuple[
            Conditioning[torch.Tensor],
            ConditioningRange,
            TokenLayoutDescriptor,
            tuple[tuple[str, ExtensionInputValue], ...],
        ]
    ] = []
    try:
        _check_cancelled(check)
        register_scheduled_producer_types(type_registry)
        if request.ordinary_equivalent:
            ordinary = _ordinary_prompt(runtime, request.prompts[0])
            if ordinary is not None:
                carrier = ordinary_conditioning_carrier(runtime, ordinary)
                _check_cancelled(check)
                return carrier

        for prompt in request.prompts:
            for text_state in _states(request.text_patches):
                for diffusion_state in _states(request.diffusion_patches):
                    for transform_state in _states(request.transform_stacks):
                        ranges = [prompt.schedule]
                        if text_state is not None:
                            ranges.append(text_state.schedule)
                        if diffusion_state is not None:
                            ranges.append(diffusion_state.schedule)
                        if transform_state is not None:
                            ranges.append(transform_state.schedule)
                        effective = intersect_ranges(*ranges)
                        if effective is EMPTY_RANGE:
                            continue
                        _check_cancelled(check)
                        text_overlays = () if text_state is None else text_state.overlays
                        diffusion_overlays = (
                            () if diffusion_state is None else diffusion_state.overlays
                        )
                        text_runtime: object = runtime
                        if text_overlays:
                            assert execution is not None
                            text_runtime = execution.variants.acquire(
                                runtime,
                                base_runtime_identity=runtime.runtime_identity,
                                target=PatchTargetComponent.TEXT,
                                overlays=text_overlays,
                                cancelled=check,
                            )
                            digest = patch_overlay_stack_digest(text_overlays)
                            assert digest is not None
                            acquired.append(
                                VariantKey(
                                    runtime.runtime_identity,
                                    PatchTargetComponent.TEXT,
                                    digest,
                                )
                            )
                        if diffusion_overlays:
                            assert execution is not None
                            execution.variants.acquire(
                                runtime,
                                base_runtime_identity=runtime.runtime_identity,
                                target=PatchTargetComponent.DIFFUSION,
                                overlays=diffusion_overlays,
                                cancelled=check,
                            )
                            digest = patch_overlay_stack_digest(diffusion_overlays)
                            assert digest is not None
                            acquired.append(
                                VariantKey(
                                    runtime.runtime_identity,
                                    PatchTargetComponent.DIFFUSION,
                                    digest,
                                )
                            )
                        condition = _encode_routes(text_runtime, prompt)
                        descriptors = () if transform_state is None else transform_state.transforms
                        for descriptor in descriptors:
                            _check_cancelled(check)
                            function = transform_fns[descriptor.id]
                            if descriptor.target is TransformTarget.TEXT:
                                changed = _require_compatible_transform_result(
                                    descriptor.id,
                                    condition.embeddings,
                                    cast("object", function(condition.embeddings)),
                                )
                                condition = replace(condition, embeddings=changed)
                            else:
                                if condition.pooled is None:
                                    raise ScheduledEncodingError(
                                        f"transform {descriptor.id!r} targets absent pooled lane"
                                    )
                                changed = _require_compatible_transform_result(
                                    descriptor.id,
                                    condition.pooled,
                                    cast("object", function(condition.pooled)),
                                )
                                condition = replace(condition, pooled=changed)
                            _check_cancelled(check)
                        metadata = prompt.extension_metadata + scheduled_metadata(
                            target=_family_id(runtime),
                            text_overlays=text_overlays,
                            diffusion_overlays=diffusion_overlays,
                            transforms=descriptors,
                        )
                        outputs.append(
                            (
                                condition,
                                effective,
                                _layout(runtime, condition),
                                metadata,
                            )
                        )
        _check_cancelled(check)
        return _carrier(tuple(outputs))
    except BaseException as error:
        if execution is not None:
            if isinstance(error, _ScheduledEncodingCancelled):
                execution.close()
            else:
                cleanup_error = execution.variants.poison_many(tuple(acquired))
                if cleanup_error is not None:
                    raise cleanup_error from error
        raise


__all__ = ["encode_text_scheduled", "ordinary_conditioning_carrier"]
