"""Worker-local contracts for schedule-aware text encoding.

The values in this module are an evaluator input for the canonical
``ConditioningSet`` IR. They are not prompt syntax and never cross the public
worker protocol. Patch variants are structural ownership records in B1; a
later backend-owned adapter may materialize native handles.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar, cast

from dinkster_values import TypeRegistry

from .conditioning import (
    ConditioningRange,
    EmptyRange,
    ExtensionInputValue,
    PayloadReference,
    PercentRange,
    _freeze_extension_value,  # pyright: ignore[reportPrivateUsage]
)
from .recipe import PatchOverlay, patch_overlay_stack_digest
from .types import register_inference_types

SCHEDULED_METADATA_VERSION = 1
SCHEDULED_METADATA_PREFIX = "dinkster.inference/"
SCHEDULED_METADATA_KEYS = frozenset(
    {
        "dinkster.inference/version",
        "dinkster.inference/target",
        "dinkster.inference/text-overlay-digests",
        "dinkster.inference/text-overlay-stack-digest",
        "dinkster.inference/diffusion-overlay-digests",
        "dinkster.inference/diffusion-overlay-stack-digest",
        "dinkster.inference/transform-ids",
        "dinkster.inference/transform-digests",
        "dinkster.inference/effective-patch-state",
    }
)


class ScheduledEncodingError(ValueError):
    """A scheduled request or worker-local execution seam refused."""


InferenceTypeRegistry = TypeRegistry


def register_scheduled_producer_types(registry: InferenceTypeRegistry) -> None:
    """Register on the live registry supplied to the scheduled producer."""
    if not isinstance(cast("object", registry), TypeRegistry):
        raise TypeError("scheduled producer type_registry must be TypeRegistry")
    register_inference_types(registry)


class EncoderStream(StrEnum):
    CLIP_L = "clip_l"
    CLIP_G = "clip_g"
    T5 = "t5"
    OVIS_QWEN3_2B = "ovis_qwen3_2b"

    @classmethod
    def from_encoder_id(cls, encoder_id: str) -> EncoderStream:
        stream = encoder_id.removeprefix("dinkster.")
        if stream == "t5xxl":
            stream = cls.T5.value
        try:
            return cls(stream)
        except ValueError:
            raise ScheduledEncodingError(
                f"text encoder {encoder_id!r} has no scheduled stream"
            ) from None


class TransformTarget(StrEnum):
    TEXT = "text"
    POOLED = "pooled"


class PatchTargetComponent(StrEnum):
    TEXT = "text"
    DIFFUSION = "diffusion"


@dataclass(frozen=True)
class ScheduledPromptRoute:
    stream: EncoderStream
    prompt: str

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.stream), EncoderStream):
            raise TypeError("scheduled prompt stream must be EncoderStream")
        if not isinstance(cast("object", self.prompt), str):
            raise TypeError("scheduled prompt must be a string")


@dataclass(frozen=True)
class ScheduledPrompt:
    schedule: ConditioningRange
    routes: tuple[ScheduledPromptRoute, ...]
    extension_metadata: tuple[tuple[str, ExtensionInputValue], ...] = ()

    def __post_init__(self) -> None:
        _require_range(self.schedule)
        if not isinstance(cast("object", self.routes), tuple) or not self.routes:
            raise ValueError("a scheduled prompt needs a non-empty route tuple")
        routes_obj = cast("tuple[object, ...]", self.routes)
        if any(not isinstance(route, ScheduledPromptRoute) for route in routes_obj):
            raise TypeError("scheduled prompt routes must be ScheduledPromptRoute values")
        streams = tuple(route.stream for route in self.routes)
        if len(streams) != len(set(streams)):
            raise ValueError("scheduled prompt streams must be unique")
        _validate_metadata(self.extension_metadata)
        object.__setattr__(
            self,
            "extension_metadata",
            tuple((key, _freeze_extension_value(value)) for key, value in self.extension_metadata),
        )


@dataclass(frozen=True)
class PostEncodeTransform:
    """Identified transform descriptor; the callable stays worker-local."""

    id: str
    target: TransformTarget
    family_id: str
    text_streams: tuple[EncoderStream, ...]
    layout_version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(cast("object", self.id), str)
            or not self.id
            or self.id.strip() != self.id
        ):
            raise ValueError("transform id must be a non-empty trimmed string")
        if not isinstance(cast("object", self.target), TransformTarget):
            raise TypeError("transform target must be TransformTarget")
        if (
            not isinstance(cast("object", self.family_id), str)
            or not self.family_id
            or self.family_id.strip() != self.family_id
        ):
            raise ValueError("transform family_id must be non-empty and trimmed")
        if (
            not isinstance(cast("object", self.text_streams), tuple)
            or not self.text_streams
            or any(
                not isinstance(stream, EncoderStream)
                for stream in cast("tuple[object, ...]", self.text_streams)
            )
            or len(set(self.text_streams)) != len(self.text_streams)
        ):
            raise ValueError("transform text_streams must be non-empty and unique")
        if type(self.layout_version) is not int or self.layout_version != 1:
            raise ValueError("only scheduled transform layout version 1 is supported")

    @property
    def structural_digest(self) -> str:
        document = {
            "family": self.family_id,
            "id": self.id,
            "layoutVersion": self.layout_version,
            "streams": [stream.value for stream in self.text_streams],
            "target": self.target.value,
            "version": 1,
        }
        return hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ScheduledTransformStack:
    schedule: ConditioningRange
    transforms: tuple[PostEncodeTransform, ...]

    def __post_init__(self) -> None:
        _require_range(self.schedule)
        if not isinstance(cast("object", self.transforms), tuple) or not self.transforms:
            raise ValueError("a transform stack needs a non-empty transform tuple")
        transforms_obj = cast("tuple[object, ...]", self.transforms)
        if any(not isinstance(item, PostEncodeTransform) for item in transforms_obj):
            raise TypeError("transform stack entries must be PostEncodeTransform values")
        ids = tuple(item.id for item in self.transforms)
        if len(ids) != len(set(ids)):
            raise ValueError("transform ids must be unique within a stack")


@dataclass(frozen=True)
class ScheduledPatchStack:
    schedule: ConditioningRange
    overlays: tuple[PatchOverlay, ...]

    def __post_init__(self) -> None:
        _require_range(self.schedule)
        if not isinstance(cast("object", self.overlays), tuple) or not self.overlays:
            raise ValueError("a scheduled patch stack must be non-empty")
        overlays_obj = cast("tuple[object, ...]", self.overlays)
        if any(not isinstance(item, PatchOverlay) for item in overlays_obj):
            raise TypeError("scheduled patch entries must be PatchOverlay values")

    @property
    def stack_digest(self) -> str:
        digest = patch_overlay_stack_digest(self.overlays)
        assert digest is not None
        return digest


@dataclass(frozen=True)
class ScheduledEncodeRequest:
    prompts: tuple[ScheduledPrompt, ...]
    text_patches: tuple[ScheduledPatchStack, ...] = ()
    diffusion_patches: tuple[ScheduledPatchStack, ...] = ()
    transform_stacks: tuple[ScheduledTransformStack, ...] = ()

    def __post_init__(self) -> None:
        values = (
            ("prompts", self.prompts, ScheduledPrompt),
            ("text_patches", self.text_patches, ScheduledPatchStack),
            ("diffusion_patches", self.diffusion_patches, ScheduledPatchStack),
            ("transform_stacks", self.transform_stacks, ScheduledTransformStack),
        )
        for name, value, item_type in values:
            if not isinstance(cast("object", value), tuple):
                raise TypeError(f"scheduled {name} must be a tuple")
            value_obj = cast("tuple[object, ...]", value)
            if any(not isinstance(item, item_type) for item in value_obj):
                raise TypeError(f"scheduled {name} contains the wrong value type")
        for prompt in self.prompts:
            _validate_metadata(prompt.extension_metadata)

    @property
    def ordinary_equivalent(self) -> bool:
        return (
            len(self.prompts) == 1
            and self.prompts[0].schedule == PercentRange(0.0, 1.0)
            and not self.prompts[0].extension_metadata
            and not self.text_patches
            and not self.diffusion_patches
            and not self.transform_stacks
        )


TensorT = TypeVar("TensorT")


class ScheduledFamilyRuntime(Protocol[TensorT]):
    """Sibling scheduled seam; ordinary ``FamilyRuntime`` stays frozen."""

    def encode_text_scheduled(
        self,
        request: ScheduledEncodeRequest,
        *,
        execution: ScheduledExecution[object] | None = None,
        transforms: Mapping[str, Callable[[TensorT], TensorT]] | None = None,
        cancelled: Callable[[], bool] | None = None,
        type_registry: InferenceTypeRegistry,
    ) -> object: ...


VariantT = TypeVar("VariantT")
VariantBuilder = Callable[
    [object, PatchTargetComponent, tuple[PatchOverlay, ...], Callable[[], bool]],
    VariantT,
]


@dataclass(frozen=True)
class VariantKey:
    base_runtime_identity: str
    target: PatchTargetComponent
    patch_overlay_stack_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(cast("object", self.base_runtime_identity), str)
            or not self.base_runtime_identity
        ):
            raise ValueError("variant base runtime identity must be non-empty")
        if not isinstance(cast("object", self.target), PatchTargetComponent):
            raise TypeError("variant target must be PatchTargetComponent")
        if (
            not isinstance(cast("object", self.patch_overlay_stack_digest), str)
            or len(self.patch_overlay_stack_digest) != 64
            or any(char not in "0123456789abcdef" for char in self.patch_overlay_stack_digest)
        ):
            raise ValueError("variant patch overlay stack digest must be sha256 hex")


def _dispose(value: object) -> None:
    disposer = getattr(value, "dispose", None)
    if disposer is None:
        disposer = getattr(value, "close", None)
    if not callable(disposer):
        raise ScheduledEncodingError("a scheduled variant must provide dispose() or close()")
    disposer()


def _dispose_many(values: tuple[object, ...]) -> BaseException | None:
    first_error: BaseException | None = None
    for value in values:
        try:
            _dispose(value)
        except BaseException as error:
            if first_error is None:
                first_error = error
    return first_error


class ScheduledVariantOwner(Generic[VariantT]):
    """Execution-scoped, partitioned transactional variant owner."""

    MAX_TEST_CAPACITY = 16

    def __init__(
        self,
        builder: VariantBuilder[VariantT] | None = None,
        *,
        text_capacity: int = 1,
        diffusion_capacity: int = 1,
    ) -> None:
        for name, value in (("text", text_capacity), ("diffusion", diffusion_capacity)):
            if type(value) is not int or not 1 <= value <= self.MAX_TEST_CAPACITY:
                raise ValueError(
                    f"{name} variant capacity must be in [1, {self.MAX_TEST_CAPACITY}]"
                )
        self._builder = builder
        self._capacities = MappingProxyType(
            {
                PatchTargetComponent.TEXT: text_capacity,
                PatchTargetComponent.DIFFUSION: diffusion_capacity,
            }
        )
        self._entries: dict[PatchTargetComponent, OrderedDict[VariantKey, VariantT]] = {
            PatchTargetComponent.TEXT: OrderedDict(),
            PatchTargetComponent.DIFFUSION: OrderedDict(),
        }
        self._closed = False

    def acquire(
        self,
        base: object,
        *,
        base_runtime_identity: str,
        target: PatchTargetComponent,
        overlays: tuple[PatchOverlay, ...],
        cancelled: Callable[[], bool] | None = None,
    ) -> object:
        if not isinstance(cast("object", target), PatchTargetComponent):
            raise TypeError("scheduled variant target must be PatchTargetComponent")
        if not isinstance(cast("object", overlays), tuple) or any(
            not isinstance(item, PatchOverlay) for item in cast("tuple[object, ...]", overlays)
        ):
            raise TypeError("scheduled variant overlays must be a PatchOverlay tuple")
        if not isinstance(cast("object", base_runtime_identity), str) or not base_runtime_identity:
            raise ValueError("scheduled variant base runtime identity must be non-empty")
        if not overlays:
            return base
        if self._closed:
            raise ScheduledEncodingError("scheduled variant owner is closed")
        cancellation_seen = False

        def check() -> bool:
            nonlocal cancellation_seen
            if cancelled is not None and cancelled():
                cancellation_seen = True
            return cancellation_seen

        if check():
            self.close()
            raise ScheduledEncodingError("scheduled variant construction cancelled")
        if self._builder is None:
            raise ScheduledEncodingError(
                "nonempty scheduled patch stacks require a worker-local variant builder"
            )
        digest = patch_overlay_stack_digest(overlays)
        assert digest is not None
        key = VariantKey(base_runtime_identity, target, digest)
        entries = self._entries[target]
        existing = entries.get(key)
        if existing is not None:
            entries.move_to_end(key)
            return existing
        candidate: VariantT | None = None
        try:
            candidate = self._builder(base, target, overlays, check)
            if candidate is base:
                raise ScheduledEncodingError("a variant builder must not return the shared base")
            if any(
                candidate is owned
                for partition in self._entries.values()
                for owned in partition.values()
            ):
                candidate = None
                raise ScheduledEncodingError(
                    "a variant builder must return a newly owned candidate"
                )
            disposer = getattr(candidate, "dispose", None) or getattr(candidate, "close", None)
            if not callable(disposer):
                raise ScheduledEncodingError(
                    "a scheduled variant must provide dispose() or close()"
                )
        except BaseException as build_error:
            cleanup_error: BaseException | None = None
            if candidate is not None and candidate is not base:
                cleanup_error = _dispose_many((candidate,))
            if cancellation_seen:
                try:
                    self.close()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
            if cleanup_error is not None:
                raise cleanup_error from build_error
            raise
        if check():
            cleanup_error = _dispose_many((candidate,))
            try:
                self.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            if cleanup_error is not None:
                raise cleanup_error
            raise ScheduledEncodingError("scheduled variant construction cancelled")
        if len(entries) >= self._capacities[target]:
            _, victim = entries.popitem(last=False)
            try:
                _dispose(victim)
            except BaseException as victim_error:
                cleanup_error = _dispose_many((candidate,))
                if cleanup_error is not None:
                    raise cleanup_error from victim_error
                raise
        entries[key] = candidate
        return candidate

    def poison(self, key: VariantKey) -> None:
        if not isinstance(cast("object", key), VariantKey):
            raise TypeError("poison key must be VariantKey")
        error = self.poison_many((key,))
        if error is not None:
            raise error

    def poison_many(self, keys: tuple[VariantKey, ...]) -> BaseException | None:
        if not isinstance(cast("object", keys), tuple) or any(
            not isinstance(key, VariantKey) for key in cast("tuple[object, ...]", keys)
        ):
            raise TypeError("poison keys must be a VariantKey tuple")
        owned: list[object] = []
        for key in dict.fromkeys(keys):
            value = self._entries[key.target].pop(key, None)
            if value is not None:
                owned.append(value)
        return _dispose_many(tuple(owned))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        owned: list[object] = []
        for target in (PatchTargetComponent.TEXT, PatchTargetComponent.DIFFUSION):
            entries = self._entries[target]
            while entries:
                _, value = entries.popitem(last=False)
                owned.append(value)
        error = _dispose_many(tuple(owned))
        if error is not None:
            raise error

    end = close


@dataclass(frozen=True)
class ScheduledExecution(Generic[VariantT]):
    variants: ScheduledVariantOwner[VariantT]

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.variants), ScheduledVariantOwner):
            raise TypeError("scheduled execution variants must be ScheduledVariantOwner")

    def close(self) -> None:
        self.variants.close()

    end = close


def intersect_ranges(*ranges: ConditioningRange) -> ConditioningRange:
    result: ConditioningRange = PercentRange(0.0, 1.0)
    for value in ranges:
        _require_range(value)
        result = result.intersect(value)
    return result


def scheduled_metadata(
    *,
    target: str,
    text_overlays: tuple[PatchOverlay, ...],
    diffusion_overlays: tuple[PatchOverlay, ...],
    transforms: tuple[PostEncodeTransform, ...],
) -> tuple[tuple[str, ExtensionInputValue], ...]:
    if not text_overlays and not diffusion_overlays and not transforms:
        return ()
    text_digest = patch_overlay_stack_digest(text_overlays)
    diffusion_digest = patch_overlay_stack_digest(diffusion_overlays)
    document = {
        "version": SCHEDULED_METADATA_VERSION,
        "target": target,
        "text": text_digest,
        "diffusion": diffusion_digest,
        "transforms": [item.structural_digest for item in transforms],
    }
    effective = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    values: list[tuple[str, ExtensionInputValue]] = [
        ("dinkster.inference/version", SCHEDULED_METADATA_VERSION),
        ("dinkster.inference/target", target),
        (
            "dinkster.inference/text-overlay-digests",
            tuple(item.structural_digest for item in text_overlays),
        ),
        (
            "dinkster.inference/diffusion-overlay-digests",
            tuple(item.structural_digest for item in diffusion_overlays),
        ),
        ("dinkster.inference/transform-ids", tuple(item.id for item in transforms)),
        (
            "dinkster.inference/transform-digests",
            tuple(item.structural_digest for item in transforms),
        ),
        ("dinkster.inference/effective-patch-state", effective),
    ]
    if text_digest is not None:
        values.append(("dinkster.inference/text-overlay-stack-digest", text_digest))
    if diffusion_digest is not None:
        values.append(("dinkster.inference/diffusion-overlay-stack-digest", diffusion_digest))
    return tuple(values)


def _require_range(value: object) -> None:
    if not isinstance(value, (PercentRange, EmptyRange)):
        raise TypeError("scheduled range must be PercentRange or EMPTY_RANGE")


def _validate_metadata(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("scheduled extension metadata must be a tuple of pairs")
    seen: set[str] = set()
    for item in cast("tuple[object, ...]", value):
        if not isinstance(item, tuple):
            raise TypeError("scheduled extension metadata must contain string-keyed pairs")
        pair = cast("tuple[object, ...]", item)
        if len(pair) != 2 or not isinstance(pair[0], str):
            raise TypeError("scheduled extension metadata must contain string-keyed pairs")
        key = pair[0]
        if key.count("/") != 1:
            raise ValueError("scheduled extension metadata keys must be pack_id/key strings")
        pack_id, local_key = key.split("/", 1)
        if not pack_id or not local_key or key.strip() != key:
            raise ValueError("scheduled extension metadata keys need non-empty trimmed parts")
        if key in seen:
            raise ValueError(f"duplicate scheduled extension metadata key {key!r}")
        seen.add(key)
        if key.startswith(SCHEDULED_METADATA_PREFIX):
            raise ScheduledEncodingError(
                f"caller metadata cannot use reserved namespace {SCHEDULED_METADATA_PREFIX!r}"
            )
        _validate_metadata_value(pair[1])


def _validate_metadata_value(value: object) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("scheduled extension metadata floats must be finite")
        return
    if isinstance(value, PayloadReference):
        return
    if isinstance(value, (list, tuple)):
        for item in cast("list[object] | tuple[object, ...]", value):
            _validate_metadata_value(item)
        return
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        if any(not isinstance(key, str) for key in mapping):
            raise TypeError("scheduled extension metadata mapping keys must be strings")
        for item in mapping.values():
            _validate_metadata_value(item)
        return
    raise TypeError("scheduled extension metadata contains a non-RPC-clean value")


__all__ = [
    "EncoderStream",
    "InferenceTypeRegistry",
    "PatchTargetComponent",
    "PostEncodeTransform",
    "SCHEDULED_METADATA_KEYS",
    "SCHEDULED_METADATA_PREFIX",
    "SCHEDULED_METADATA_VERSION",
    "ScheduledEncodeRequest",
    "ScheduledEncodingError",
    "ScheduledExecution",
    "ScheduledFamilyRuntime",
    "ScheduledPatchStack",
    "ScheduledPrompt",
    "ScheduledPromptRoute",
    "ScheduledTransformStack",
    "ScheduledVariantOwner",
    "TransformTarget",
    "VariantKey",
    "intersect_ranges",
    "scheduled_metadata",
]
