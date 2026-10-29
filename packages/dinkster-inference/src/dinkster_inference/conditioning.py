"""Torch-free conditioning record contracts.

Ranges mirror ComfyUI @ f4b99bc: ``ConditioningSetTimestepRange`` and
``set_timesteps_for_conditioning`` retain percent pairs (nodes.py:287-290,
comfy/hooks.py:713-717), materialization converts each bound
(comfy/samplers.py:816-839), and activation rejects only values strictly
outside them (comfy/samplers.py:37-44). Both endpoints are therefore closed;
a zero-width range is active at its one converted sigma. ``EMPTY_RANGE`` is
the distinct, never-active result of intersecting disjoint ranges.

These values cross process and persistence boundaries only through the
separately versioned canonical carrier in ``conditioning_wire``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import TypeAlias, cast

from .guidance import ConditionScaleVector
from .spaces import SigmaSpace

CONDITIONING_RECORD_BOUNDARY = "dinkster-conditioning-carrier-v1"


def _has_type(value: object, expected: type[object] | tuple[type[object], ...]) -> bool:
    """Runtime constructor validation without weakening public annotations."""

    return isinstance(value, expected)


class ConditioningChannel(StrEnum):
    TEXT = "text"
    POOLED = "pooled"
    CONCAT_LATENT = "concat_latent"
    VISION_EMBEDDING = "vision_embedding"
    REFERENCE_VISION_EMBEDDING = "reference_vision_embedding"
    AUDIO_EMBEDDING = "audio_embedding"
    CONTROL_VIDEO = "control_video"
    REFERENCE_MOTION = "reference_motion"
    POSE_TEXT = "pose_text"
    POSE_VISION_EMBEDDING = "pose_vision_embedding"
    POSE_LATENT = "pose_latent"
    FACE_PIXELS = "face_pixels"
    CONTROL_HINT = "control_hint"
    REFERENCE_LATENT = "reference_latent"
    SCAIL_REFERENCE_LATENT = "scail_reference_latent"
    SCAIL_REFERENCE_MASK = "scail_reference_mask"
    SCAIL_DRIVING_MASK = "scail_driving_mask"
    CAMERA = "camera"


class AreaUnits(StrEnum):
    LATENT_CELLS = "latent-cells"
    PERCENT = "percent"


@dataclass(frozen=True)
class PayloadReference:
    """Opaque worker-local reference to a separately held payload."""

    id: str

    def __post_init__(self) -> None:
        if not _has_type(self.id, str):
            raise TypeError("payload reference id must be a string")
        if not self.id or self.id.strip() != self.id:
            raise ValueError("payload reference id must be a non-empty trimmed string")


@dataclass(frozen=True)
class PayloadDescriptor:
    """Declarative payload geometry; never a tensor or wire handle."""

    reference: PayloadReference
    shape: tuple[int, ...]
    dtype: str
    space: str

    def __post_init__(self) -> None:
        if not _has_type(self.reference, PayloadReference):
            raise TypeError("payload reference must be PayloadReference")
        if (
            not _has_type(self.shape, tuple)
            or not self.shape
            or any(type(dim) is not int or dim < 0 for dim in self.shape)
        ):
            raise ValueError("payload shape must contain non-negative integer dimensions")
        if any(
            not _has_type(value, str) or not value or value.strip() != value
            for value in (self.dtype, self.space)
        ):
            raise ValueError("payload dtype and space must be non-empty trimmed strings")


@dataclass(frozen=True)
class PercentRange:
    """Closed normalized percent range; equal endpoints remain active."""

    start_percent: float
    end_percent: float

    def __post_init__(self) -> None:
        if type(self.start_percent) is not float or type(self.end_percent) is not float:
            raise TypeError("range endpoints must be floats")
        if not math.isfinite(self.start_percent) or not math.isfinite(self.end_percent):
            raise ValueError("range endpoints must be finite")
        if not 0.0 <= self.start_percent <= self.end_percent <= 1.0:
            raise ValueError("range endpoints must satisfy 0 <= start <= end <= 1")

    def to_percent_pair(self) -> tuple[float, float]:
        return (self.start_percent, self.end_percent)

    def sigma_bounds(self, space: SigmaSpace) -> tuple[float, float]:
        return (
            space.percent_to_sigma(self.start_percent),
            space.percent_to_sigma(self.end_percent),
        )

    def is_active(self, sigma: float, space: SigmaSpace) -> bool:
        start_sigma, end_sigma = self.sigma_bounds(space)
        return sigma <= start_sigma and sigma >= end_sigma

    def intersect(self, other: ConditioningRange) -> ConditioningRange:
        if not _has_type(other, (PercentRange, EmptyRange)):
            raise TypeError("can only intersect another conditioning range")
        if other is EMPTY_RANGE:
            return EMPTY_RANGE
        start = max(self.start_percent, other.start_percent)
        end = min(self.end_percent, other.end_percent)
        if start > end:
            return EMPTY_RANGE
        return PercentRange(start, end)


class EmptyRange(Enum):
    """The sole algebraic empty range, distinct from every percent pair."""

    VALUE = "empty"

    def to_percent_pair(self) -> tuple[float, float]:
        raise ValueError("the empty range has no percent-pair representation")

    def sigma_bounds(self, space: SigmaSpace) -> tuple[float, float]:
        del space
        raise ValueError("the empty range has no sigma bounds")

    def is_active(self, sigma: float, space: SigmaSpace) -> bool:
        del sigma, space
        return False

    def intersect(self, other: ConditioningRange) -> EmptyRange:
        del other
        return self


EMPTY_RANGE = EmptyRange.VALUE
ConditioningRange: TypeAlias = PercentRange | EmptyRange


@dataclass(frozen=True)
class AreaDescriptor:
    """Region in ComfyUI extent-then-offset axis order.

    Percent materialization uses Python ``round`` with a minimum extent of one,
    exactly as comfy/samplers.py:760-786 @ b78cec87. Coordinates have no minimum.
    """

    height: int | float
    width: int | float
    y: int | float
    x: int | float
    units: AreaUnits
    strength: float = 1.0
    temporal: float | None = None
    z: float | None = None

    def __post_init__(self) -> None:
        values = (self.height, self.width, self.y, self.x)
        temporal_values = (self.temporal, self.z)
        if (self.temporal is None) != (self.z is None):
            raise ValueError("temporal area extent and offset must be provided together")
        if type(self.strength) is not float:
            raise TypeError("area strength must be a float")
        if not math.isfinite(self.strength) or self.strength < 0.0:
            raise ValueError("area strength must be finite and non-negative")
        if self.units is AreaUnits.LATENT_CELLS:
            if self.temporal is not None:
                raise ValueError("temporal areas use percent units")
            if any(type(value) is not int for value in values):
                raise TypeError("latent-cell area fields must be integers")
            if self.height < 1 or self.width < 1 or self.y < 0 or self.x < 0:
                raise ValueError("latent-cell extents must be positive and offsets non-negative")
        elif self.units is AreaUnits.PERCENT:
            percent_values = values + temporal_values if self.temporal is not None else values
            if any(type(value) is not float for value in percent_values):
                raise TypeError("percent area fields must be floats")
            if any(
                not math.isfinite(cast("float", value)) or not 0.0 <= cast("float", value) <= 1.0
                for value in percent_values
            ):
                raise ValueError("percent area fields must be finite values in [0, 1]")
        else:
            raise TypeError("units must be AreaUnits")

    def materialize_percent(
        self, latent_height: int, latent_width: int
    ) -> tuple[int, int, int, int]:
        if self.units is not AreaUnits.PERCENT:
            raise ValueError("only percent areas require materialization")
        if self.temporal is not None:
            raise ValueError("temporal percent areas require video materialization")
        if latent_height < 1 or latent_width < 1:
            raise ValueError("latent dimensions must be positive")
        return (
            max(1, round(cast("float", self.height) * latent_height)),
            max(1, round(cast("float", self.width) * latent_width)),
            round(cast("float", self.y) * latent_height),
            round(cast("float", self.x) * latent_width),
        )

    def materialize_percent_video(
        self, latent_temporal: int, latent_height: int, latent_width: int
    ) -> tuple[int, int, int, int, int, int]:
        if self.units is not AreaUnits.PERCENT or self.temporal is None or self.z is None:
            raise ValueError("only temporal percent areas require video materialization")
        if latent_temporal < 1 or latent_height < 1 or latent_width < 1:
            raise ValueError("latent dimensions must be positive")
        return (
            max(1, round(self.temporal * latent_temporal)),
            max(1, round(cast("float", self.height) * latent_height)),
            max(1, round(cast("float", self.width) * latent_width)),
            round(self.z * latent_temporal),
            round(cast("float", self.y) * latent_height),
            round(cast("float", self.x) * latent_width),
        )


@dataclass(frozen=True)
class MaskDescriptor:
    """Mask payload and ComfyUI ConditioningSetMask metadata."""

    payload: PayloadReference
    strength: float = 1.0
    set_area_to_bounds: bool = False

    def __post_init__(self) -> None:
        if not _has_type(self.payload, PayloadReference):
            raise TypeError("mask payload must be PayloadReference")
        if type(self.strength) is not float:
            raise TypeError("mask strength must be a float")
        if not math.isfinite(self.strength) or self.strength < 0.0:
            raise ValueError("mask strength must be finite and non-negative")
        if type(self.set_area_to_bounds) is not bool:
            raise TypeError("set_area_to_bounds must be bool")


@dataclass(frozen=True)
class RegionDescriptor:
    area: AreaDescriptor | None = None
    mask: MaskDescriptor | None = None

    def __post_init__(self) -> None:
        if self.area is not None and not _has_type(self.area, AreaDescriptor):
            raise TypeError("region area must be AreaDescriptor")
        if self.mask is not None and not _has_type(self.mask, MaskDescriptor):
            raise TypeError("region mask must be MaskDescriptor")
        if self.area is None and self.mask is None:
            raise ValueError("a region must provide an area, a mask, or both")


@dataclass(frozen=True)
class TokenSegmentDescriptor:
    name: str
    stream: str
    start_token: int
    token_count: int | None = None

    def __post_init__(self) -> None:
        if any(
            not _has_type(value, str) or not value or value.strip() != value
            for value in (self.name, self.stream)
        ):
            raise ValueError("token segment name and stream must be non-empty")
        if type(self.start_token) is not int or (
            self.token_count is not None and type(self.token_count) is not int
        ):
            raise TypeError("token segment bounds must be integers")
        if self.start_token < 0 or self.token_count is not None and self.token_count < 1:
            raise ValueError("token segment bounds must be non-negative and non-empty")


@dataclass(frozen=True)
class TokenLayoutDescriptor:
    """Versioned text-stream order and named token segments.

    Model-family materializers must call ``require_supported`` and refuse any
    family/version pair they do not explicitly implement.
    """

    family_id: str
    version: int
    text_streams: tuple[str, ...]
    segments: tuple[TokenSegmentDescriptor, ...]

    def __post_init__(self) -> None:
        if not _has_type(self.family_id, str) or not self.family_id:
            raise ValueError("token layout needs a family id")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("token layout needs a family id and positive version")
        if (
            not _has_type(self.text_streams, tuple)
            or not self.text_streams
            or any(not _has_type(stream, str) or not stream for stream in self.text_streams)
            or len(set(self.text_streams)) != len(self.text_streams)
        ):
            raise ValueError("token layout streams must be non-empty and unique")
        if (
            not _has_type(self.segments, tuple)
            or not self.segments
            or any(not _has_type(segment, TokenSegmentDescriptor) for segment in self.segments)
        ):
            raise ValueError("token layout segments must be a non-empty descriptor tuple")
        names = tuple(segment.name for segment in self.segments)
        if len(set(names)) != len(names):
            raise ValueError("token segment names must be unique")
        if any(segment.stream not in self.text_streams for segment in self.segments):
            raise ValueError("every token segment must name a declared stream")

    def require_supported(self, family_id: str, versions: Sequence[int]) -> None:
        if family_id != self.family_id or self.version not in versions:
            raise ValueError(
                f"unsupported token layout {self.family_id!r} v{self.version} "
                f"for family {family_id!r}"
            )


ExtensionScalar: TypeAlias = None | bool | int | float | str
ExtensionValue: TypeAlias = (
    ExtensionScalar
    | PayloadReference
    | tuple["ExtensionValue", ...]
    | Mapping[str, "ExtensionValue"]
)
ExtensionInputValue: TypeAlias = (
    ExtensionScalar
    | PayloadReference
    | Sequence["ExtensionInputValue"]
    | Mapping[str, "ExtensionInputValue"]
)


def _freeze_extension_value(value: object) -> ExtensionValue:
    if value is None or type(value) in (bool, int, str):
        return cast("ExtensionScalar", value)
    if type(value) is float:
        number = value
        if not math.isfinite(number):
            raise ValueError("extension metadata floats must be finite")
        return number
    if isinstance(value, PayloadReference):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_extension_value(item) for item in cast("Sequence[object]", value))
    if isinstance(value, Mapping):
        raw = cast("Mapping[object, object]", value)
        if any(not isinstance(key, str) for key in raw):
            raise TypeError("extension metadata dictionary keys must be strings")
        return MappingProxyType(
            {cast("str", key): _freeze_extension_value(item) for key, item in raw.items()}
        )
    raise TypeError("extension metadata contains a non-RPC-clean value")


def _clone_extension_value(value: ExtensionValue) -> ExtensionValue:
    if isinstance(value, PayloadReference):
        return value
    if isinstance(value, tuple):
        return tuple(_clone_extension_value(item) for item in value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _clone_extension_value(item) for key, item in value.items()})
    return value


def _validate_scale_vector(vector: ConditionScaleVector[PayloadDescriptor] | None) -> None:
    """Validate the declarative per-condition scale-vector contract.

    Axis 0 indexes the conditioning batch. Length one broadcasts across that
    batch; otherwise it must equal the materialized batch. No additional axes
    are allowed, and scale values broadcast across tokens, channels, and
    spatial dimensions. The runtime checks the materialized batch before use.
    """

    if vector is not None:
        if not _has_type(vector, ConditionScaleVector) or not _has_type(
            vector.values, PayloadDescriptor
        ):
            raise TypeError("condition scale vector must wrap a PayloadDescriptor")
        if len(vector.values.shape) != 1 or vector.values.shape[0] < 1:
            raise ValueError("condition scale vector payload must be non-empty rank 1")


@dataclass(frozen=True)
class ConditioningRecord:
    channels: tuple[tuple[ConditioningChannel, PayloadDescriptor], ...]
    area: AreaDescriptor | None = None
    mask: MaskDescriptor | None = None
    schedule: ConditioningRange = PercentRange(0.0, 1.0)
    scale_vector: ConditionScaleVector[PayloadDescriptor] | None = None
    token_layout: TokenLayoutDescriptor | None = None
    extension_metadata: tuple[tuple[str, ExtensionInputValue], ...] = ()

    def __post_init__(self) -> None:
        raw_channels = cast("object", self.channels)
        if not isinstance(raw_channels, tuple):
            raise TypeError("channels must be a tuple of (channel, descriptor) pairs")
        if not raw_channels:
            raise ValueError("conditioning record must contain at least one channel")
        channels: list[tuple[ConditioningChannel, PayloadDescriptor]] = []
        seen_channels: set[ConditioningChannel] = set()
        for raw_item in cast("tuple[object, ...]", raw_channels):
            if not isinstance(raw_item, tuple):
                raise TypeError("channels must contain pairs")
            item = cast("tuple[object, ...]", raw_item)
            if len(item) != 2:
                raise TypeError("channels must contain pairs")
            channel, payload = item
            if not isinstance(channel, ConditioningChannel) or not isinstance(
                payload, PayloadDescriptor
            ):
                raise TypeError("channel entries need ConditioningChannel and PayloadDescriptor")
            if channel in seen_channels:
                raise ValueError(f"duplicate conditioning channel {channel.value!r}")
            seen_channels.add(channel)
            channels.append((channel, payload))
        object.__setattr__(self, "channels", tuple(channels))
        if self.area is not None and not _has_type(self.area, AreaDescriptor):
            raise TypeError("area must be AreaDescriptor")
        if self.mask is not None and not _has_type(self.mask, MaskDescriptor):
            raise TypeError("mask must be MaskDescriptor")
        if not _has_type(self.schedule, (PercentRange, EmptyRange)):
            raise TypeError("schedule must be PercentRange or EMPTY_RANGE")
        if self.token_layout is not None and not _has_type(
            self.token_layout, TokenLayoutDescriptor
        ):
            raise TypeError("token_layout must be TokenLayoutDescriptor")
        _validate_scale_vector(self.scale_vector)

        raw_metadata = cast("object", self.extension_metadata)
        if not isinstance(raw_metadata, tuple):
            raise TypeError("extension_metadata must be a tuple of pairs")
        metadata: list[tuple[str, ExtensionValue]] = []
        seen_keys: set[str] = set()
        for raw_item in cast("tuple[object, ...]", raw_metadata):
            if not isinstance(raw_item, tuple):
                raise TypeError("extension_metadata must contain pairs")
            item = cast("tuple[object, ...]", raw_item)
            if len(item) != 2:
                raise TypeError("extension_metadata must contain pairs")
            key, value = item
            if not isinstance(key, str) or key.count("/") != 1:
                raise ValueError("extension metadata keys must be pack_id/key strings")
            pack_id, local_key = key.split("/", 1)
            if not pack_id or not local_key or key.strip() != key:
                raise ValueError("extension metadata keys must have non-empty trimmed parts")
            if key in seen_keys:
                raise ValueError(f"duplicate extension metadata key {key!r}")
            seen_keys.add(key)
            metadata.append((key, _freeze_extension_value(value)))
        object.__setattr__(self, "extension_metadata", tuple(metadata))

    def clone(self) -> ConditioningRecord:
        """Deep-copy metadata while sharing immutable payload descriptors."""

        area = None if self.area is None else AreaDescriptor(**self.area.__dict__)
        mask = None
        if self.mask is not None:
            mask = MaskDescriptor(
                payload=self.mask.payload,
                strength=self.mask.strength,
                set_area_to_bounds=self.mask.set_area_to_bounds,
            )
        schedule: ConditioningRange
        if self.schedule is EMPTY_RANGE:
            schedule = EMPTY_RANGE
        else:
            schedule = PercentRange(self.schedule.start_percent, self.schedule.end_percent)
        scale_vector = None
        if self.scale_vector is not None:
            scale_vector = ConditionScaleVector(self.scale_vector.values)
        token_layout = None
        if self.token_layout is not None:
            token_layout = TokenLayoutDescriptor(
                family_id=self.token_layout.family_id,
                version=self.token_layout.version,
                text_streams=tuple(self.token_layout.text_streams),
                segments=tuple(
                    TokenSegmentDescriptor(**segment.__dict__)
                    for segment in self.token_layout.segments
                ),
            )
        return ConditioningRecord(
            channels=self.channels,
            area=area,
            mask=mask,
            schedule=schedule,
            scale_vector=scale_vector,
            token_layout=token_layout,
            extension_metadata=tuple(
                (key, _clone_extension_value(cast("ExtensionValue", value)))
                for key, value in self.extension_metadata
            ),
        )

    def for_region(self, region: RegionDescriptor) -> ConditioningRecord:
        """Replace area/mask and preserve all other metadata.

        This mirrors the overwrite behavior of node_helpers.py:8-22 and
        nodes.py:180-204,238-248 @ f4b99bc. Unknown extension metadata is
        preserve-only in v1; no extension transform policy is invoked.
        """

        if not _has_type(region, RegionDescriptor):
            raise TypeError("region must be RegionDescriptor")
        cloned = self.clone()
        return ConditioningRecord(
            channels=cloned.channels,
            area=region.area,
            mask=region.mask,
            schedule=cloned.schedule,
            scale_vector=cloned.scale_vector,
            token_layout=cloned.token_layout,
            extension_metadata=cloned.extension_metadata,
        )


@dataclass(frozen=True)
class ConditioningSet:
    """Immutable ordered record collection.

    ``combine`` concatenates left then right, matching ConditioningCombine
    (nodes.py:82-93) and hooks.combine_conditioning (hooks.py:731-735) at the
    pinned ComfyUI SHA. It never merges records element-wise.
    """

    records: tuple[ConditioningRecord, ...] = ()

    def __post_init__(self) -> None:
        raw_records = cast("object", self.records)
        if not isinstance(raw_records, tuple) or any(
            not isinstance(record, ConditioningRecord)
            for record in cast("tuple[object, ...]", raw_records)
        ):
            raise TypeError("records must be a tuple of ConditioningRecord values")

    def clone(self) -> ConditioningSet:
        return ConditioningSet(tuple(record.clone() for record in self.records))

    def combine(self, other: ConditioningSet) -> ConditioningSet:
        raw_other = cast("object", other)
        if not isinstance(raw_other, ConditioningSet):
            raise TypeError("can only combine another ConditioningSet")
        return ConditioningSet(self.records + raw_other.records)

    def for_region(self, region: RegionDescriptor) -> ConditioningSet:
        return ConditioningSet(tuple(record.for_region(region) for record in self.records))


# Every builtin kind x algebra operation. Citations are the mirrored behavior,
# or "Dinkster divergence" where ComfyUI has no corresponding metadata kind.
BUILTIN_METADATA_MERGE_TABLE: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "area": MappingProxyType(
            {
                "clone": "deep copy: deliberate divergence from node_helpers.py:8-22",
                "combine": "ordered records: nodes.py:82-93",
                "for_region": "replace/clear: deliberate divergence from nodes.py:180-204",
            }
        ),
        "mask": MappingProxyType(
            {
                "clone": "deep copy: deliberate divergence from node_helpers.py:8-22",
                "combine": "ordered records: nodes.py:82-93",
                "for_region": "replace/clear: deliberate divergence from nodes.py:238-248",
            }
        ),
        "schedule": MappingProxyType(
            {
                "clone": "deep copy: deliberate divergence from node_helpers.py:8-22",
                "combine": "ordered records: nodes.py:82-93",
                "for_region": "preserve: deliberate Dinkster divergence",
            }
        ),
        "scale_vector": MappingProxyType(
            {
                "clone": "copy wrapper/share payload: deliberate Dinkster divergence",
                "combine": "ordered records: nodes.py:82-93",
                "for_region": "preserve: deliberate Dinkster divergence",
            }
        ),
        "token_layout": MappingProxyType(
            {
                "clone": "deep copy: deliberate Dinkster divergence",
                "combine": "ordered records: nodes.py:82-93",
                "for_region": "preserve: deliberate Dinkster divergence",
            }
        ),
    }
)


def _canonical_range(value: ConditioningRange) -> object:
    if value is EMPTY_RANGE:
        return {"kind": "empty"}
    return {
        "kind": "percent",
        "start": _canonical_float(value.start_percent),
        "end": _canonical_float(value.end_percent),
    }


def _canonical_payload(value: PayloadDescriptor) -> object:
    return {
        "ref": value.reference.id,
        "shape": list(value.shape),
        "dtype": value.dtype,
        "space": value.space,
    }


def _canonical_float(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _canonical_extension(value: ExtensionValue) -> object:
    if value is None:
        return {"type": "none"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        return {"type": "float", "value": _canonical_float(value)}
    if type(value) is str:
        return {"type": "str", "value": value}
    if isinstance(value, PayloadReference):
        return {"type": "payload-reference", "id": value.id}
    if isinstance(value, tuple):
        return {
            "type": "sequence",
            "items": [_canonical_extension(item) for item in value],
        }
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "entries": [[key, _canonical_extension(value[key])] for key in sorted(value)],
        }
    raise TypeError("invalid frozen extension metadata value")


def canonical_conditioning_set(value: ConditioningSet) -> str:
    """Canonical behavior serialization; not a wire codec or runtime identity.

    B1 separately decides whether token layouts enter model runtime identity.
    """

    records: list[object] = []
    for record in value.records:
        area = None
        if record.area is not None:
            area = {
                "height": record.area.height,
                "width": record.area.width,
                "y": record.area.y,
                "x": record.area.x,
                "units": record.area.units.value,
                "strength": _canonical_float(record.area.strength),
            }
            if record.area.units is AreaUnits.PERCENT:
                area.update(
                    {
                        "height": _canonical_float(cast("float", record.area.height)),
                        "width": _canonical_float(cast("float", record.area.width)),
                        "y": _canonical_float(cast("float", record.area.y)),
                        "x": _canonical_float(cast("float", record.area.x)),
                    }
                )
                if record.area.temporal is not None and record.area.z is not None:
                    area.update(
                        {
                            "temporal": _canonical_float(record.area.temporal),
                            "z": _canonical_float(record.area.z),
                        }
                    )
        mask = None
        if record.mask is not None:
            mask = {
                "payload_ref": record.mask.payload.id,
                "strength": _canonical_float(record.mask.strength),
                "set_area_to_bounds": record.mask.set_area_to_bounds,
            }
        token_layout = None
        if record.token_layout is not None:
            token_layout = {
                "family_id": record.token_layout.family_id,
                "version": record.token_layout.version,
                "text_streams": list(record.token_layout.text_streams),
                "segments": [
                    {
                        "name": segment.name,
                        "stream": segment.stream,
                        "start_token": segment.start_token,
                        "token_count": segment.token_count,
                    }
                    for segment in record.token_layout.segments
                ],
            }
        records.append(
            {
                "channels": [
                    {"id": channel.value, "payload": _canonical_payload(payload)}
                    for channel, payload in record.channels
                ],
                "area": area,
                "mask": mask,
                "schedule": _canonical_range(record.schedule),
                "scale_vector": None
                if record.scale_vector is None
                else _canonical_payload(record.scale_vector.values),
                "token_layout": token_layout,
                "extension_metadata": {
                    key: _canonical_extension(cast("ExtensionValue", metadata))
                    for key, metadata in sorted(record.extension_metadata)
                },
            }
        )
    return json.dumps(
        {"format": "dinkster-conditioning-set-v1", "records": records},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
