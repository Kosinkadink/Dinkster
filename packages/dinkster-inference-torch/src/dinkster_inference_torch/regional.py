"""Regional conditioning materialization and evaluation.

The tensor behavior is a narrow port of ``resolve_areas_and_cond_masks_multidim``,
``get_area_and_mult``, and the one-condition-list accumulation in
``_calc_cond_batch`` from ComfyUI @ b78cec87. Materialization consumes the
canonical DMFC v1 carrier once per sample; evaluation owns no cache, model
handle, or other mutable process state.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Generator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypeAlias, cast

import torch
from dinkster_inference import (
    EMPTY_RANGE,
    SCHEDULED_METADATA_KEYS,
    SCHEDULED_METADATA_PREFIX,
    SCHEDULED_METADATA_VERSION,
    Conditioning,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRange,
    ExtensionInputValue,
    GuidanceRole,
    ModelFamily,
    PatchSet,
    PayloadBinding,
    PercentRange,
    RealizedSamplingTimeline,
    SigmaSpace,
    TokenLayoutDescriptor,
    current_realized_sampling_row,
    current_realized_sampling_timeline,
    encode_conditioning_carrier,
)

from . import scaled_patches as _scaled_patches
from .denoise import FluxDenoiser
from .memory import DeviceMemory, get_free_memory, regional_working_memory
from .payloads import TensorPayloadError, payload_binding_to_tensor
from .scaled_patches import PreparedScaledPatches, ScaledPatchError
from .sd_denoise import SDDenoiser

if TYPE_CHECKING:
    from dinkster_inference import GuidanceRole, Parameterization

    from .flux import Flux
    from .unet import UNetModel


MASK_PAYLOAD_SPACE = "conditioning-mask"
SCALE_PAYLOAD_SPACE = "conditioning-scale"

_TEXT_SPACE = "conditioning-text"
_POOLED_SPACE = "conditioning-pooled"
_TENSOR_DTYPES = frozenset(("F32", "F16", "BF16"))


class RegionalConditioningError(ValueError):
    """A deterministic regional materialization or evaluation refusal."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        message = f"regional-conditioning:{code}"
        super().__init__(f"{message}: {detail}" if detail else message)


@dataclass(frozen=True)
class MaterializedRegion:
    """One frozen-container record projection for a sampling execution.

    Tensor references remain ordinary mutable torch tensors owned by the
    execution; freezing prevents field replacement, not tensor mutation.
    """

    conditioning: Conditioning[torch.Tensor]
    area: tuple[int, int, int, int] | None
    area_strength: float
    mask: torch.Tensor | None
    mask_strength: float
    schedule: ConditioningRange
    extension_metadata: tuple[tuple[str, ExtensionInputValue], ...]
    latent_shape: tuple[int, int]
    scale_vector: torch.Tensor | None = None
    patch_digest: str | None = None
    realized_active_steps: frozenset[int] | None = None
    realized_timeline_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.conditioning, Conditioning):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("conditioning must be Conditioning")
        if self.area is not None and (
            not isinstance(self.area, tuple)  # pyright: ignore[reportUnnecessaryIsInstance]
            or len(self.area) != 4
            or any(type(value) is not int for value in self.area)
            or self.area[0] < 1
            or self.area[1] < 1
            or self.area[2] < 0
            or self.area[3] < 0
        ):
            raise TypeError("area must have positive extents and non-negative integer offsets")
        if (
            type(self.area_strength) is not float
            or not math.isfinite(self.area_strength)
            or self.area_strength < 0.0
        ):
            raise TypeError("area strength must be a finite non-negative float")
        if self.mask is not None and not isinstance(self.mask, torch.Tensor):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("mask must be a torch Tensor")
        if (
            type(self.mask_strength) is not float
            or not math.isfinite(self.mask_strength)
            or self.mask_strength < 0.0
        ):
            raise TypeError("mask strength must be a finite non-negative float")
        if self.schedule is not EMPTY_RANGE and not isinstance(self.schedule, PercentRange):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("schedule must be PercentRange or EMPTY_RANGE")
        if not isinstance(self.extension_metadata, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("extension metadata must be a tuple")
        if (
            not isinstance(self.latent_shape, tuple)  # pyright: ignore[reportUnnecessaryIsInstance]
            or len(self.latent_shape) != 2
            or any(type(value) is not int or value < 1 for value in self.latent_shape)
        ):
            raise TypeError("latent shape must be a positive (height, width) integer tuple")
        if self.scale_vector is not None and (
            not isinstance(self.scale_vector, torch.Tensor)  # pyright: ignore[reportUnnecessaryIsInstance]
            or self.scale_vector.ndim != 1
            or self.scale_vector.numel() < 1
            or self.scale_vector.dtype is not torch.float32
        ):
            raise TypeError("scale vector must be a nonempty rank-1 float32 Tensor")
        if self.patch_digest is not None and (
            type(self.patch_digest) is not str
            or len(self.patch_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.patch_digest)
        ):
            raise TypeError("patch digest must be a lowercase sha256 hex string")
        if self.realized_active_steps is not None and (
            type(self.realized_active_steps) is not frozenset
            or any(type(index) is not int or index < 0 for index in self.realized_active_steps)
        ):
            raise TypeError("realized active steps must be a frozenset of non-negative integers")
        if (self.realized_active_steps is None) != (self.realized_timeline_digest is None):
            raise TypeError("realized region steps and timeline digest must be declared together")
        if self.realized_timeline_digest is not None and (
            len(self.realized_timeline_digest) != 64
            or any(
                character not in "0123456789abcdef" for character in self.realized_timeline_digest
            )
        ):
            raise TypeError("realized region timeline digest must be lowercase sha256 hex")


def realize_region_schedules(
    regions: tuple[MaterializedRegion, ...],
    timeline: RealizedSamplingTimeline,
    space: SigmaSpace,
) -> tuple[MaterializedRegion, ...]:
    """Compile conditioning PercentRanges onto one exact executed timeline."""
    return tuple(
        replace(
            region,
            realized_active_steps=frozenset(
                step.step_index
                for step in timeline.executed.steps
                if region.schedule is not EMPTY_RANGE
                and region.schedule.is_active(step.sigma, space)
            ),
            realized_timeline_digest=timeline.digest,
        )
        for region in regions
    )


def _region_schedule_is_active(
    region: MaterializedRegion,
    sigma: float,
    space: SigmaSpace,
) -> bool:
    if region.realized_active_steps is None:
        return region.schedule is not EMPTY_RANGE and region.schedule.is_active(sigma, space)
    row = current_realized_sampling_row()
    timeline = current_realized_sampling_timeline()
    if row is None or timeline is None:
        raise _refuse("sampling-timeline-row")
    if timeline.digest != region.realized_timeline_digest:
        raise _refuse("sampling-timeline-digest")
    return row.anchors.step_index in region.realized_active_steps


RegionEvaluator: TypeAlias = Callable[[MaterializedRegion, torch.Tensor, float], torch.Tensor]
GroupedRegionEvaluator: TypeAlias = Callable[
    [
        tuple[MaterializedRegion, ...],
        torch.Tensor,
        float,
        Conditioning[torch.Tensor],
        tuple[GuidanceRole, ...],
        tuple[int, ...],
    ],
    torch.Tensor,
]


def _refuse(code: str, detail: str = "") -> RegionalConditioningError:
    return RegionalConditioningError(code, detail)


def _binding_map(carrier: ConditioningCarrier) -> dict[str, PayloadBinding]:
    bindings: dict[str, PayloadBinding] = {}
    for binding in carrier.bindings:
        if binding.reference_id in bindings:
            raise _refuse("duplicate-binding", binding.reference_id)
        bindings[binding.reference_id] = binding
    return bindings


def _decode_channel(
    binding: PayloadBinding,
    *,
    descriptor_shape: tuple[int, ...],
    descriptor_dtype: str,
    descriptor_space: str,
    expected_space: str,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    if (binding.shape, binding.dtype, binding.space) != (
        descriptor_shape,
        descriptor_dtype,
        descriptor_space,
    ):
        raise _refuse("descriptor-mismatch", binding.reference_id)
    if binding.space != expected_space:
        raise _refuse("payload-space", f"expected {expected_space!r}, got {binding.space!r}")
    if binding.dtype not in _TENSOR_DTYPES:
        raise _refuse("payload-dtype", binding.dtype)
    if len(binding.shape) != rank or any(dim < 1 for dim in binding.shape):
        raise _refuse("payload-shape", repr(binding.shape))
    try:
        return payload_binding_to_tensor(binding).to(device=device)
    except TensorPayloadError as error:
        raise _refuse("payload-decode", str(error)) from None


def _validate_layout(
    family_id: str,
    layout: TokenLayoutDescriptor | None,
    *,
    token_count: int,
) -> None:
    if layout is None:
        raise _refuse("missing-token-layout")
    try:
        layout.require_supported(family_id, (1,))
    except (AttributeError, TypeError, ValueError) as error:
        raise _refuse("unsupported-token-layout", str(error)) from None
    if any(
        segment.start_token >= token_count
        or (
            segment.token_count is not None
            and segment.start_token + segment.token_count > token_count
        )
        for segment in layout.segments
    ):
        raise _refuse("unsupported-token-layout", repr(layout.segments))


def _materialize_mask(
    binding: PayloadBinding,
    *,
    latent_height: int,
    latent_width: int,
    device: torch.device,
) -> torch.Tensor:
    if binding.space != MASK_PAYLOAD_SPACE:
        raise _refuse("mask-space", binding.space)
    if binding.dtype != "F32":
        raise _refuse("mask-dtype", binding.dtype)
    if len(binding.shape) not in (2, 3) or any(dim < 1 for dim in binding.shape):
        raise _refuse("mask-shape", repr(binding.shape))
    try:
        mask = payload_binding_to_tensor(binding).to(device=device)
    except TensorPayloadError as error:
        raise _refuse("mask-decode", str(error)) from None
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[1:] != (latent_height, latent_width):
        # comfy.utils.common_upscale(..., "bilinear", "none") @ b78cec87.
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(1),
            size=(latent_height, latent_width),
            mode="bilinear",
        ).squeeze(1)
    return mask


def _scheduled_patch_digest(
    metadata: tuple[tuple[str, ExtensionInputValue], ...],
    family_id: str,
) -> str | None:
    reserved = {key: value for key, value in metadata if key.startswith(SCHEDULED_METADATA_PREFIX)}
    if not reserved:
        return None
    if len(reserved) != sum(key.startswith(SCHEDULED_METADATA_PREFIX) for key, _ in metadata):
        raise _refuse("scheduled-metadata-duplicate")
    unknown = set(reserved).difference(SCHEDULED_METADATA_KEYS)
    if unknown:
        raise _refuse("scheduled-metadata-key", repr(sorted(unknown)))
    required = {
        "dinkster.inference/version",
        "dinkster.inference/target",
        "dinkster.inference/text-overlay-digests",
        "dinkster.inference/diffusion-overlay-digests",
        "dinkster.inference/transform-ids",
        "dinkster.inference/transform-digests",
        "dinkster.inference/effective-patch-state",
    }
    if not required.issubset(reserved):
        raise _refuse("scheduled-metadata-missing", repr(sorted(required.difference(reserved))))
    if reserved["dinkster.inference/version"] != SCHEDULED_METADATA_VERSION:
        raise _refuse("scheduled-metadata-version")
    if reserved["dinkster.inference/target"] != family_id:
        raise _refuse("scheduled-metadata-target")
    for key in (
        "dinkster.inference/text-overlay-digests",
        "dinkster.inference/diffusion-overlay-digests",
        "dinkster.inference/transform-ids",
        "dinkster.inference/transform-digests",
    ):
        value = reserved[key]
        if not isinstance(value, (list, tuple)) or any(type(item) is not str for item in value):
            raise _refuse("scheduled-metadata-value", key)
    for key in (
        "dinkster.inference/text-overlay-digests",
        "dinkster.inference/diffusion-overlay-digests",
        "dinkster.inference/transform-digests",
    ):
        value = reserved[key]
        assert isinstance(value, (list, tuple))
        digests = cast("tuple[str, ...]", tuple(value))
        if any(
            len(item) != 64 or any(character not in "0123456789abcdef" for character in item)
            for item in digests
        ):
            raise _refuse("scheduled-metadata-digest", key)
    transform_ids_value = reserved["dinkster.inference/transform-ids"]
    transform_digests_value = reserved["dinkster.inference/transform-digests"]
    assert isinstance(transform_ids_value, (list, tuple))
    assert isinstance(transform_digests_value, (list, tuple))
    transform_ids = cast("tuple[str, ...]", tuple(transform_ids_value))
    transform_digests = cast("tuple[str, ...]", tuple(transform_digests_value))
    if len(transform_ids) != len(transform_digests) or any(
        not item or item.strip() != item for item in transform_ids
    ):
        raise _refuse("scheduled-metadata-transforms")
    for key in (
        "dinkster.inference/effective-patch-state",
        "dinkster.inference/text-overlay-stack-digest",
        "dinkster.inference/diffusion-overlay-stack-digest",
    ):
        value = reserved.get(key)
        if value is not None and (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise _refuse("scheduled-metadata-digest", key)
    text_value = reserved["dinkster.inference/text-overlay-digests"]
    diffusion_value = reserved["dinkster.inference/diffusion-overlay-digests"]
    assert isinstance(text_value, (list, tuple))
    assert isinstance(diffusion_value, (list, tuple))
    text = cast("tuple[str, ...]", tuple(text_value))
    diffusion = cast("tuple[str, ...]", tuple(diffusion_value))
    text_stack = cast(str | None, reserved.get("dinkster.inference/text-overlay-stack-digest"))
    digest = cast(str | None, reserved.get("dinkster.inference/diffusion-overlay-stack-digest"))
    for values, stack_digest, code in (
        (text, text_stack, "scheduled-metadata-text"),
        (diffusion, digest, "scheduled-metadata-diffusion"),
    ):
        expected_stack = (
            None
            if not values
            else hashlib.sha256(
                json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
        if stack_digest != expected_stack:
            raise _refuse(code)
    effective_document = {
        "version": SCHEDULED_METADATA_VERSION,
        "target": family_id,
        "text": text_stack,
        "diffusion": digest,
        "transforms": list(transform_digests),
    }
    expected_effective = hashlib.sha256(
        json.dumps(effective_document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if reserved["dinkster.inference/effective-patch-state"] != expected_effective:
        raise _refuse("scheduled-metadata-effective")
    return digest


def _mask_aabb(mask: torch.Tensor) -> tuple[int, int, int, int]:
    bounds = torch.max(torch.abs(mask), dim=0).values
    nonzero = torch.where(bounds)
    if nonzero[0].numel() == 0:
        return (8, 8, 0, 0)
    y, x = nonzero
    height = max(8, int(torch.max(y) - torch.min(y) + 1))
    width = max(8, int(torch.max(x) - torch.min(x) + 1))
    return (height, width, int(torch.min(y)), int(torch.min(x)))


def materialize_regions(
    carrier: ConditioningCarrier,
    family_id: str,
    latent_height: int,
    latent_width: int,
    device: torch.device | str,
) -> tuple[MaterializedRegion, ...]:
    """Decode and validate a canonical carrier once for one sample."""
    if not isinstance(carrier, ConditioningCarrier):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise _refuse("invalid-carrier")
    if not isinstance(family_id, str) or not family_id:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise _refuse("unsupported-family", repr(family_id))
    if type(latent_height) is not int or type(latent_width) is not int:
        raise _refuse("latent-shape", "dimensions must be integers")
    if latent_height < 1 or latent_width < 1:
        raise _refuse("latent-shape", "dimensions must be positive")
    try:
        target = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise _refuse("device", str(error)) from None
    try:
        encoded = encode_conditioning_carrier(carrier)
    except (TypeError, ValueError) as error:
        raise _refuse("invalid-carrier", str(error)) from None
    if not encoded.startswith(b"DMFC\x01"):
        raise _refuse("unsupported-carrier-version")
    bindings = _binding_map(carrier)
    regions: list[MaterializedRegion] = []
    for record in carrier.conditioning.records:
        channels = dict(record.channels)
        if len(channels) != len(record.channels):
            raise _refuse("duplicate-channel")
        carried = tuple(
            channel.value
            for channel in channels
            if channel not in (ConditioningChannel.TEXT, ConditioningChannel.POOLED)
        )
        if carried:
            raise _refuse("carried-only-channel", repr(carried))
        text_descriptor = channels.get(ConditioningChannel.TEXT)
        if text_descriptor is None:
            raise _refuse("missing-text")
        text_binding = bindings.get(text_descriptor.reference.id)
        if text_binding is None:
            raise _refuse("missing-binding", text_descriptor.reference.id)
        text = _decode_channel(
            text_binding,
            descriptor_shape=text_descriptor.shape,
            descriptor_dtype=text_descriptor.dtype,
            descriptor_space=text_descriptor.space,
            expected_space=_TEXT_SPACE,
            rank=3,
            device=target,
        )
        pooled_descriptor = channels.get(ConditioningChannel.POOLED)
        pooled = None
        if pooled_descriptor is not None:
            pooled_binding = bindings.get(pooled_descriptor.reference.id)
            if pooled_binding is None:
                raise _refuse("missing-binding", pooled_descriptor.reference.id)
            pooled = _decode_channel(
                pooled_binding,
                descriptor_shape=pooled_descriptor.shape,
                descriptor_dtype=pooled_descriptor.dtype,
                descriptor_space=pooled_descriptor.space,
                expected_space=_POOLED_SPACE,
                rank=2,
                device=target,
            )
            if pooled.shape[0] != text.shape[0]:
                raise _refuse("conditioning-batch", "TEXT and POOLED batches disagree")
        _validate_layout(
            family_id,
            record.token_layout,
            token_count=int(text.shape[1]),
        )
        area = None
        area_strength = 1.0
        if record.area is not None:
            if record.area.temporal is not None:
                raise _refuse(
                    "temporal-area-unsupported",
                    "the regional sampling path supports only 2D latent areas",
                )
            area_strength = record.area.strength
            area = (
                record.area.materialize_percent(latent_height, latent_width)
                if record.area.units.value == "percent"
                else cast(
                    "tuple[int, int, int, int]",
                    (record.area.height, record.area.width, record.area.y, record.area.x),
                )
            )
        mask = None
        mask_strength = 1.0
        if record.mask is not None:
            mask_binding = bindings.get(record.mask.payload.id)
            if mask_binding is None:
                raise _refuse("missing-binding", record.mask.payload.id)
            mask = _materialize_mask(
                mask_binding,
                latent_height=latent_height,
                latent_width=latent_width,
                device=target,
            )
            mask_strength = record.mask.strength
            if record.mask.set_area_to_bounds:
                area = _mask_aabb(mask)
        scale_vector = None
        if record.scale_vector is not None:
            descriptor = record.scale_vector.values
            scale_binding = bindings.get(descriptor.reference.id)
            if scale_binding is None:
                raise _refuse("missing-binding", descriptor.reference.id)
            if (scale_binding.shape, scale_binding.dtype, scale_binding.space) != (
                descriptor.shape,
                descriptor.dtype,
                descriptor.space,
            ):
                raise _refuse("descriptor-mismatch", scale_binding.reference_id)
            if (
                scale_binding.space != SCALE_PAYLOAD_SPACE
                or scale_binding.dtype != "F32"
                or len(scale_binding.shape) != 1
                or scale_binding.shape[0] < 1
            ):
                raise _refuse("scale-payload")
            try:
                scale_vector = payload_binding_to_tensor(scale_binding).to(device=target)
            except TensorPayloadError as error:
                raise _refuse("scale-decode", str(error)) from None
            if not bool(torch.isfinite(scale_vector).all().item()):
                raise _refuse("scale-finite")
        patch_digest = _scheduled_patch_digest(record.extension_metadata, family_id)
        if scale_vector is not None and patch_digest is None:
            raise _refuse("scale-without-patch")
        regions.append(
            MaterializedRegion(
                Conditioning(text, pooled),
                area,
                area_strength,
                mask,
                mask_strength,
                record.schedule,
                record.extension_metadata,
                (latent_height, latent_width),
                scale_vector,
                patch_digest,
            )
        )
    return tuple(regions)


def _check_cancel(cancel: Callable[[], bool]) -> None:
    result = cancel()
    if type(result) is not bool:
        raise _refuse("callback")
    if result:
        raise _refuse("cancelled")


def _crop_and_multiplier(
    region: MaterializedRegion,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int] | None]:
    area = region.area
    crop = x
    if area is not None:
        height, width, y, x_offset = area
        if y >= x.shape[2] or x_offset >= x.shape[3]:
            raise _refuse("area-out-of-bounds", repr(area))
        height = min(x.shape[2] - y, height)
        width = min(x.shape[3] - x_offset, width)
        if height <= 0 or width <= 0:
            raise _refuse("degenerate-area", repr(area))
        area = (height, width, y, x_offset)
        crop = x.narrow(2, y, height).narrow(3, x_offset, width)
    mask = region.mask
    if mask is not None:
        # Pinned get_area_and_mult truncates an oversized mask batch first,
        # then repeats an undersized one with integer whole-tensor repeats.
        narrowed = mask[: crop.shape[0]]
        if crop.shape[0] % narrowed.shape[0] != 0:
            raise _refuse(
                "mask-batch",
                f"mask batch {narrowed.shape[0]} does not divide latent batch {crop.shape[0]}",
            )
        if area is not None:
            narrowed = narrowed.narrow(1, area[2], area[0]).narrow(2, area[3], area[1])
        multiplier = narrowed * region.mask_strength
        multiplier = multiplier.unsqueeze(1).repeat(
            (crop.shape[0] // narrowed.shape[0], crop.shape[1], 1, 1)
        )
    else:
        multiplier = torch.ones_like(crop)
    multiplier = multiplier * region.area_strength
    if mask is None and area is not None:
        fuzz = 8
        for axis, (extent, offset, total) in enumerate(
            ((area[0], area[2], x.shape[2]), (area[1], area[3], x.shape[3]))
        ):
            radius = min(fuzz, multiplier.shape[2 + axis] // 4)
            if offset != 0:
                for index in range(radius):
                    multiplier.narrow(2 + axis, index, 1).mul_((index + 1) / radius)
            if extent + offset < total:
                for index in range(radius):
                    multiplier.narrow(2 + axis, extent - 1 - index, 1).mul_((index + 1) / radius)
    return crop, multiplier, area


@dataclass(frozen=True)
class _PreparedRegion:
    region: MaterializedRegion
    crop: torch.Tensor
    multiplier: torch.Tensor
    area: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class _GroupedItem:
    prepared: _PreparedRegion
    role: GuidanceRole


def _can_concat(first: _PreparedRegion, other: _PreparedRegion) -> bool:
    """Pinned can_concat_cond/cond_equal_size over the supported IR subset."""
    if first.crop.shape != other.crop.shape:
        return False
    first_text = first.region.conditioning.embeddings
    other_text = other.region.conditioning.embeddings
    if first_text.device != other_text.device or first_text.shape[2] != other_text.shape[2]:
        return False
    common_tokens = math.lcm(first_text.shape[1], other_text.shape[1])
    if common_tokens // min(first_text.shape[1], other_text.shape[1]) > 4:
        return False
    first_pooled = first.region.conditioning.pooled
    other_pooled = other.region.conditioning.pooled
    if (first_pooled is None) != (other_pooled is None):
        return False
    if first_pooled is not None and other_pooled is not None:
        if first_pooled.device != other_pooled.device:
            return False
        if first_pooled.shape[1:] != other_pooled.shape[1:]:
            return False
    return True


def _reference_order(prepared: list[_PreparedRegion]) -> list[_PreparedRegion]:
    """Port the reference sufficient-memory/max-concat ordering path.

    B2.1 invokes one callback per region, so it deterministically selects every
    peer compatible with the first pending item and reverses that maximal group.
    B2.3 owns grouped one-forward execution and memory-fit subgroup splitting.
    """
    pending = prepared.copy()
    ordered: list[_PreparedRegion] = []
    while pending:
        first = pending[0]
        compatible = [index for index, item in enumerate(pending) if _can_concat(first, item)]
        compatible.reverse()
        for index in compatible:
            ordered.append(pending.pop(index))
    return ordered


def evaluate_regions(
    regions: tuple[MaterializedRegion, ...],
    x: torch.Tensor,
    sigma: float,
    space: SigmaSpace,
    evaluate: RegionEvaluator,
    cancel: Callable[[], bool],
) -> torch.Tensor:
    """Evaluate one condition list and normalize overlap like ComfyUI."""
    if not isinstance(regions, tuple) or any(  # pyright: ignore[reportUnnecessaryIsInstance]
        not isinstance(region, MaterializedRegion)  # pyright: ignore[reportUnnecessaryIsInstance]
        for region in regions
    ):
        raise _refuse("invalid-regions")
    if not isinstance(x, torch.Tensor) or x.ndim != 4:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise _refuse("latent-shape", "x must be rank 4")
    if any(dimension < 1 for dimension in x.shape):
        raise _refuse("latent-shape", "x dimensions must be positive")
    if x.dtype is not torch.float32:
        raise _refuse("latent-dtype", str(x.dtype))
    if type(sigma) is not float or not math.isfinite(sigma):
        raise _refuse("sigma", repr(sigma))
    if not callable(evaluate) or not callable(cancel):
        raise _refuse("callback")
    spatial_shape = (x.shape[2], x.shape[3])
    for region in regions:
        if region.latent_shape != spatial_shape:
            raise _refuse(
                "materialized-shape",
                f"{region.latent_shape!r} != {spatial_shape!r}",
            )
        if region.conditioning.embeddings.device != x.device or (
            region.conditioning.pooled is not None and region.conditioning.pooled.device != x.device
        ):
            raise _refuse("materialized-device", "conditioning and latent devices disagree")
        if region.mask is not None and region.mask.device != x.device:
            raise _refuse("materialized-device", "mask and latent devices disagree")
        if region.scale_vector is not None or region.patch_digest is not None:
            raise _refuse("grouped-executor-required")
    _check_cancel(cancel)
    output = torch.zeros_like(x)
    counts = torch.ones_like(x) * 1e-37
    active = [
        _PreparedRegion(region, *_crop_and_multiplier(region, x))
        for region in regions
        if _region_schedule_is_active(region, sigma, space)
    ]
    for prepared in _reference_order(active):
        region = prepared.region
        crop = prepared.crop
        multiplier = prepared.multiplier
        area = prepared.area
        _check_cancel(cancel)
        value = evaluate(region, crop, sigma)
        if not isinstance(value, torch.Tensor):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise _refuse("callback-type")
        if value.shape != crop.shape:
            raise _refuse("callback-shape", f"{tuple(value.shape)} != {tuple(crop.shape)}")
        if value.dtype != crop.dtype:
            raise _refuse("callback-dtype", f"{value.dtype} != {crop.dtype}")
        if value.device != crop.device:
            raise _refuse("callback-device", f"{value.device} != {crop.device}")
        out_view = output
        count_view = counts
        if area is not None:
            out_view = out_view.narrow(2, area[2], area[0]).narrow(3, area[3], area[1])
            count_view = count_view.narrow(2, area[2], area[0]).narrow(3, area[3], area[1])
        out_view.add_(value * multiplier)
        count_view.add_(multiplier)
    output.div_(counts)
    _check_cancel(cancel)
    return output


@dataclass(frozen=True)
class GroupedRegionalResult:
    """Dormant grouped cond/uncond outputs plus measurable planning facts."""

    conditional: torch.Tensor
    unconditional: torch.Tensor
    model_calls: int
    subgroup_sizes: tuple[int, ...]
    staged_bytes: int


def _region_execution_binding(
    conditional: tuple[MaterializedRegion, ...],
    unconditional: tuple[MaterializedRegion, ...],
) -> tuple[tuple[object, ...], ...]:
    binding: list[tuple[object, ...]] = []
    for region in conditional + unconditional:
        tensors = (
            region.conditioning.embeddings,
            region.conditioning.pooled,
            region.mask,
            region.scale_vector,
        )
        binding.append(
            (
                region.patch_digest,
                *(
                    None
                    if value is None
                    else (
                        id(value),
                        value._version,  # pyright: ignore[reportPrivateUsage]
                        tuple(value.shape),
                        value.dtype,
                        value.layout,
                        value.device,
                    )
                    for value in tensors
                ),
            )
        )
    return tuple(binding)


class PreparedGroupedPatches:
    """Execution-scoped digest owners staged once across sigma evaluations."""

    def __init__(
        self,
        conditional: tuple[MaterializedRegion, ...],
        unconditional: tuple[MaterializedRegion, ...],
        x: torch.Tensor,
        family_id: str,
        space: SigmaSpace,
        model: torch.nn.Module,
        patch_sets: Mapping[str, PatchSet[torch.Tensor]],
        device: torch.device | str | None,
        dtype: torch.dtype,
        cancel: Callable[[], bool],
    ) -> None:
        execution_device = _preflight_grouped_sample(
            conditional,
            unconditional,
            x,
            family_id,
            space,
            model,
            patch_sets,
            device,
            dtype,
            cancel,
        )
        _check_cancel(cancel)
        region_binding = _region_execution_binding(conditional, unconditional)
        try:
            supplied = dict(patch_sets)
        except Exception:
            raise _refuse("patch-mapping-type") from None
        if any(
            type(key) is not str or not isinstance(value, PatchSet)  # pyright: ignore[reportUnnecessaryIsInstance]
            for key, value in supplied.items()
        ):
            raise _refuse("patch-mapping-type")
        if any(patch_set.structural_digest != digest for digest, patch_set in supplied.items()):
            raise _refuse("patch-mapping", "PatchSet structural digest does not match its key")
        expected_digests = frozenset(
            region.patch_digest
            for region in conditional + unconditional
            if region.patch_digest is not None
        )
        if frozenset(supplied) != expected_digests:
            raise _refuse("patch-mapping")
        self._conditional = conditional
        self._unconditional = unconditional
        self._region_binding = region_binding
        self._latent_descriptor = (
            tuple(x.shape),
            x.dtype,
            x.layout,
            x.device,
            tuple(x.stride()),
        )
        self._family_id = family_id
        self._space = space
        self._model = model
        self._device = execution_device
        self._dtype = dtype
        self._cancel = cancel
        self._stack = ExitStack()
        self._closed = False
        self._entered = False
        self._active = False
        self._prepared: dict[str, PreparedScaledPatches] = {}
        try:
            resolved = {
                digest: _scaled_patches._resolve_targets(  # pyright: ignore[reportPrivateUsage]
                    model, supplied[digest]
                )
                for digest in sorted(supplied)
            }
            if any(
                _scaled_patches._target_compute_placement(  # pyright: ignore[reportPrivateUsage]
                    target
                )
                != (self._device, self._dtype)
                for _revision, targets in resolved.values()
                for target in targets
            ):
                raise _scaled_patches._refuse(  # pyright: ignore[reportPrivateUsage]
                    "execution-placement"
                )
            for digest in sorted(supplied):
                _check_cancel(cancel)
                self._prepared[digest] = self._stack.enter_context(
                    PreparedScaledPatches(
                        model,
                        supplied[digest],
                        self._device,
                        dtype,
                        cancel,
                        _resolved=resolved[digest],
                    )
                )
        except ScaledPatchError as error:
            self._stack.close()
            self._closed = True
            raise _refuse("patch-preparation", error.code) from None
        except BaseException:
            self._stack.close()
            self._closed = True
            raise
        self.staged_bytes = sum(owner.staged_bytes for owner in self._prepared.values())

    @property
    def digests(self) -> frozenset[str]:
        return frozenset(self._prepared)

    def validate_execution(
        self,
        conditional: tuple[MaterializedRegion, ...],
        unconditional: tuple[MaterializedRegion, ...],
        x: torch.Tensor,
        family_id: str,
        space: SigmaSpace,
        model: torch.nn.Module,
        device: torch.device | str | None,
        dtype: torch.dtype,
        cancel: Callable[[], bool],
    ) -> None:
        execution_device = _preflight_grouped_sample(
            conditional,
            unconditional,
            x,
            family_id,
            space,
            model,
            self,
            device,
            dtype,
            cancel,
            allow_prepared=True,
        )
        if self._closed:
            raise _refuse("prepared-closed")
        if (
            conditional is not self._conditional
            or unconditional is not self._unconditional
            or family_id != self._family_id
            or space is not self._space
        ):
            raise _refuse("prepared-sample")
        latent_descriptor = (
            tuple(x.shape),
            x.dtype,
            x.layout,
            x.device,
            tuple(x.stride()),
        )
        if latent_descriptor != self._latent_descriptor:
            raise _refuse("prepared-sample")
        expected_digests = frozenset(
            region.patch_digest
            for region in conditional + unconditional
            if region.patch_digest is not None
        )
        if expected_digests != self.digests:
            raise _refuse("prepared-sample")
        region_binding = _region_execution_binding(conditional, unconditional)
        if region_binding != self._region_binding:
            raise _refuse("prepared-sample")
        if model is not self._model or execution_device != self._device or dtype is not self._dtype:
            raise _refuse("prepared-execution")
        if cancel is not self._cancel:
            raise _refuse("prepared-cancel")
        for digest, owner in self._prepared.items():
            if (
                owner.model is not self._model
                or owner.device != self._device
                or owner.dtype is not self._dtype
                or owner.cancel is not self._cancel
                or owner.patch_set.structural_digest != digest
            ):
                raise _refuse("prepared-binding")
            try:
                owner.validate_ready()
            except ScaledPatchError as error:
                raise _refuse("prepared-patch", error.code) from None

    @contextmanager
    def evaluation(
        self,
        conditional: tuple[MaterializedRegion, ...],
        unconditional: tuple[MaterializedRegion, ...],
        x: torch.Tensor,
        family_id: str,
        space: SigmaSpace,
        model: torch.nn.Module,
        device: torch.device | str | None,
        dtype: torch.dtype,
        cancel: Callable[[], bool],
    ) -> Generator[None, None, None]:
        self.validate_execution(
            conditional, unconditional, x, family_id, space, model, device, dtype, cancel
        )
        if self._active:
            raise _refuse("prepared-active")
        self._active = True
        try:
            yield
        finally:
            self._active = False

    def close(self) -> None:
        if self._closed:
            return
        if self._active:
            raise _refuse("prepared-active")
        self._stack.close()
        self._prepared.clear()
        self._closed = True

    def _owner_for(self, digest: str) -> PreparedScaledPatches:
        if self._closed:
            raise _refuse("prepared-closed")
        return self._prepared[digest]

    def __enter__(self) -> PreparedGroupedPatches:
        if self._active:
            raise _refuse("prepared-active")
        if self._closed or self._entered:
            raise _refuse("prepared-closed")
        self._entered = True
        return self

    def __exit__(self, *_exc: object) -> None:
        if not self._entered:
            raise _refuse("prepared-closed")
        self.close()
        self._entered = False


def prepare_grouped_patches(
    conditional: tuple[MaterializedRegion, ...],
    unconditional: tuple[MaterializedRegion, ...],
    x: torch.Tensor,
    family_id: str,
    space: SigmaSpace,
    model: torch.nn.Module,
    patch_sets: Mapping[str, PatchSet[torch.Tensor]],
    device: torch.device | str | None,
    dtype: torch.dtype,
    cancel: Callable[[], bool],
) -> PreparedGroupedPatches:
    """Stage every authorized digest once for one grouped sample execution."""

    return PreparedGroupedPatches(
        conditional,
        unconditional,
        x,
        family_id,
        space,
        model,
        patch_sets,
        device,
        dtype,
        cancel,
    )


def _group_can_concat(first: _PreparedRegion, other: _PreparedRegion) -> bool:
    return first.region.patch_digest == other.region.patch_digest and _can_concat(first, other)


def _repeat_group_conditioning(
    items: list[_PreparedRegion],
) -> Conditioning[torch.Tensor]:
    token_count = math.lcm(*(item.region.conditioning.embeddings.shape[1] for item in items))
    texts: list[torch.Tensor] = []
    pooled: list[torch.Tensor] = []
    for item in items:
        conditioning = item.region.conditioning
        batch = item.crop.shape[0]
        text = conditioning.embeddings
        if text.shape[0] not in (1, batch):
            raise _refuse("conditioning-batch", repr(tuple(text.shape)))
        if text.shape[0] == 1 and batch != 1:
            text = text.repeat(batch, 1, 1)
        repeat = token_count // text.shape[1]
        if repeat > 1:
            text = text.repeat(1, repeat, 1)
        texts.append(text)
        if conditioning.pooled is not None:
            value = conditioning.pooled
            if value.shape[0] not in (1, batch):
                raise _refuse("conditioning-batch", repr(tuple(value.shape)))
            if value.shape[0] == 1 and batch != 1:
                value = value.repeat(batch, 1)
            pooled.append(value)
    return Conditioning(
        torch.cat(texts, dim=0),
        None if not pooled else torch.cat(pooled, dim=0),
    )


def _group_scale(items: list[_PreparedRegion], dtype: torch.dtype) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for item in items:
        batch = item.crop.shape[0]
        scale = item.region.scale_vector
        if scale is None:
            scale = torch.ones(1, device=item.crop.device, dtype=torch.float32)
        if scale.numel() == 1:
            scale = scale.expand(batch)
        elif scale.numel() != batch:
            raise _refuse("scale-batch", f"{scale.numel()} != {batch}")
        converted = scale.to(device=item.crop.device, dtype=dtype)
        if not bool(torch.isfinite(converted).all().item()):
            raise _refuse("staged-scale-finite")
        rows.append(converted)
    return torch.cat(rows)


def _validated_free_total(measured: object) -> int:
    if (
        not isinstance(measured, DeviceMemory)
        or type(measured.free_total) is not int
        or type(measured.free_torch) is not int
        or measured.free_total < 0
        or measured.free_torch < 0
        or measured.free_torch > measured.free_total
    ):
        raise _refuse("memory-result")
    return measured.free_total


def _select_subgroup(
    pending: list[_GroupedItem],
    family: ModelFamily,
    dtype: torch.dtype,
    free_memory: Callable[[torch.device], DeviceMemory],
) -> list[_GroupedItem]:
    compatible = [
        index
        for index, item in enumerate(pending)
        if _group_can_concat(pending[0].prepared, item.prepared)
    ]
    compatible.reverse()
    indexes = compatible[:1]
    free_total = _validated_free_total(free_memory(pending[compatible[0]].prepared.crop.device))
    for divisor in range(1, len(compatible) + 1):
        candidate_indexes = compatible[: len(compatible) // divisor]
        selected = [pending[index] for index in candidate_indexes]
        batch = sum(item.prepared.crop.shape[0] for item in selected)
        required = regional_working_memory(
            family,
            batch=batch,
            height=selected[0].prepared.crop.shape[2],
            width=selected[0].prepared.crop.shape[3],
            dtype=dtype,
        )
        if required * 1.5 < free_total:
            indexes = candidate_indexes
            break
    return [pending.pop(index) for index in indexes]


def _validate_grouped_region(region: MaterializedRegion, x: torch.Tensor) -> None:
    try:
        region.__post_init__()
    except (TypeError, ValueError):
        raise _refuse("invalid-region") from None
    if region.latent_shape != (x.shape[2], x.shape[3]):
        raise _refuse("materialized-shape")
    if region.area is not None:
        height, width, y, x_offset = region.area
        if y >= x.shape[2] or x_offset >= x.shape[3] or height < 1 or width < 1:
            raise _refuse("area-out-of-bounds")
    embeddings: object = region.conditioning.embeddings
    if (
        not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            embeddings, torch.Tensor
        )
        or embeddings.layout is not torch.strided
        or not torch.is_floating_point(embeddings)
        or embeddings.ndim != 3
        or any(dimension < 1 for dimension in embeddings.shape)
        or embeddings.shape[0] not in (1, x.shape[0])
    ):
        raise _refuse("conditioning-shape")
    pooled: object = region.conditioning.pooled
    if pooled is not None and (
        not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            pooled, torch.Tensor
        )
        or pooled.layout is not torch.strided
        or not torch.is_floating_point(pooled)
        or pooled.ndim != 2
        or any(dimension < 1 for dimension in pooled.shape)
        or pooled.shape[0] not in (1, x.shape[0])
    ):
        raise _refuse("conditioning-shape")
    mask: object = region.mask
    if (
        mask is not None
        and (
            not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                mask, torch.Tensor
            )
            or mask.layout is not torch.strided
            or mask.device.type == "meta"
            or mask.dtype is not torch.float32
            or mask.ndim != 3
            or mask.shape[0] < 1
            or tuple(mask.shape[1:]) != region.latent_shape
            or x.shape[0] % min(mask.shape[0], x.shape[0]) != 0
            or not bool(torch.isfinite(mask).all().item())
        )
    ):
        raise _refuse("mask-shape")
    scale: object = region.scale_vector
    if scale is not None and (
        not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            scale, torch.Tensor
        )
        or scale.layout is not torch.strided
        or scale.device.type == "meta"
        or scale.dtype is not torch.float32
        or scale.ndim != 1
        or scale.numel() == 0
        or not bool(torch.isfinite(scale).all().item())
    ):
        raise _refuse("scale-vector")
    if isinstance(scale, torch.Tensor) and scale.numel() not in (1, x.shape[0]):
        raise _refuse("scale-batch", f"{scale.numel()} != {x.shape[0]}")
    tensors = (embeddings, pooled, mask, scale)
    if any(value is not None and value.is_inference() for value in tensors):
        raise _refuse("inference-tensor")
    if any(value is not None and value.device != x.device for value in tensors):
        raise _refuse("materialized-device")
    if region.scale_vector is not None and region.patch_digest is None:
        raise _refuse("scale-without-patch")


def _preflight_grouped_sample(
    conditional: tuple[MaterializedRegion, ...],
    unconditional: tuple[MaterializedRegion, ...],
    x: torch.Tensor,
    family_id: str,
    space: SigmaSpace,
    model: torch.nn.Module,
    patch_sets: Mapping[str, PatchSet[torch.Tensor]] | PreparedGroupedPatches,
    device: torch.device | str | None,
    compute_dtype: torch.dtype,
    cancel: Callable[[], bool],
    *,
    allow_prepared: bool = False,
) -> torch.device:
    raw_conditional: object = conditional
    raw_unconditional: object = unconditional
    raw_x: object = x
    raw_space: object = space
    raw_model: object = model
    raw_patch_sets: object = patch_sets
    if not isinstance(raw_conditional, tuple) or any(  # pyright: ignore[reportUnnecessaryIsInstance]
        not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            region, MaterializedRegion
        )
        for region in raw_conditional
    ):
        raise _refuse("invalid-conditional")
    if not isinstance(raw_unconditional, tuple) or any(  # pyright: ignore[reportUnnecessaryIsInstance]
        not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            region, MaterializedRegion
        )
        for region in raw_unconditional
    ):
        raise _refuse("invalid-unconditional")
    if (
        not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            raw_x, torch.Tensor
        )
        or raw_x.ndim != 4
    ):
        raise _refuse("latent-shape", "x must be rank 4")
    if any(dimension < 1 for dimension in x.shape):
        raise _refuse("latent-shape", "x dimensions must be positive")
    if x.layout is not torch.strided or x.is_quantized:
        raise _refuse("latent-layout")
    if x.dtype is not torch.float32:
        raise _refuse("latent-dtype", str(x.dtype))
    if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
        raw_model, torch.nn.Module
    ):
        raise _refuse("model")
    admitted_mapping = isinstance(raw_patch_sets, Mapping)
    admitted_prepared = allow_prepared and isinstance(raw_patch_sets, PreparedGroupedPatches)
    if not admitted_mapping and not admitted_prepared:
        raise _refuse("patch-mapping-type")
    if compute_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise _refuse("compute-dtype", repr(compute_dtype))
    if type(family_id) is not str or not family_id:
        raise _refuse("unsupported-family", family_id)
    if not callable(getattr(raw_space, "percent_to_sigma", None)):
        raise _refuse("sigma-space")
    if not callable(cancel):
        raise _refuse("callback")
    try:
        execution_device = x.device if device is None else torch.device(device)
    except (TypeError, ValueError, RuntimeError):
        raise _refuse("execution-device") from None
    if execution_device.type == "meta" or execution_device != x.device:
        raise _refuse("execution-device")
    for region in conditional + unconditional:
        _validate_grouped_region(region, x)
    return execution_device


def evaluate_grouped_regions(
    conditional: tuple[MaterializedRegion, ...],
    unconditional: tuple[MaterializedRegion, ...],
    x: torch.Tensor,
    sigma: float,
    space: SigmaSpace,
    family: ModelFamily,
    model: torch.nn.Module,
    evaluate: GroupedRegionEvaluator,
    patch_sets: Mapping[str, PatchSet[torch.Tensor]] | PreparedGroupedPatches,
    cancel: Callable[[], bool],
    *,
    compute_dtype: torch.dtype,
    free_memory: Callable[[torch.device], DeviceMemory] = get_free_memory,
) -> GroupedRegionalResult:
    """Evaluate active cond/uncond records in pinned memory-fit subgroups.

    This surface is deliberately dormant: B2.3b owns sampling/runtime wiring.
    Patch mappings are copied and completely preflighted before model work;
    carrier digests are lookup keys only and never load or construct patches.
    """

    if not isinstance(family, ModelFamily):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise _refuse("unsupported-family", repr(family))
    family_id = family.id
    if not callable(evaluate) or not callable(cancel) or not callable(free_memory):
        raise _refuse("callback")
    if type(sigma) is not float or not math.isfinite(sigma):
        raise _refuse("sigma")
    with ExitStack() as stack:
        if isinstance(patch_sets, PreparedGroupedPatches):
            grouped_owner = patch_sets
        else:
            grouped_owner = stack.enter_context(
                prepare_grouped_patches(
                    conditional,
                    unconditional,
                    x,
                    family_id,
                    space,
                    model,
                    patch_sets,
                    None,
                    compute_dtype,
                    cancel,
                )
            )
        stack.enter_context(
            grouped_owner.evaluation(
                conditional,
                unconditional,
                x,
                family_id,
                space,
                model,
                None,
                compute_dtype,
                cancel,
            )
        )
        _check_cancel(cancel)
        staged_bytes = grouped_owner.staged_bytes
        outputs = {
            GuidanceRole.CONDITIONAL: torch.zeros_like(x),
            GuidanceRole.UNCONDITIONAL: torch.zeros_like(x),
        }
        counts = {
            GuidanceRole.CONDITIONAL: torch.ones_like(x) * 1e-37,
            GuidanceRole.UNCONDITIONAL: torch.ones_like(x) * 1e-37,
        }
        calls = 0
        subgroup_sizes: list[int] = []
        pending: list[_GroupedItem] = []
        for role, regions in (
            (GuidanceRole.CONDITIONAL, conditional),
            (GuidanceRole.UNCONDITIONAL, unconditional),
        ):
            for region in regions:
                _check_cancel(cancel)
                if _region_schedule_is_active(region, sigma, space):
                    pending.append(
                        _GroupedItem(
                            _PreparedRegion(region, *_crop_and_multiplier(region, x)), role
                        )
                    )
        while pending:
            _check_cancel(cancel)
            subgroup = _select_subgroup(pending, family, compute_dtype, free_memory)
            prepared = [item.prepared for item in subgroup]
            _check_cancel(cancel)
            conditioning = _repeat_group_conditioning(prepared)
            grouped_x = torch.cat([item.crop for item in prepared], dim=0)
            scale = _group_scale(prepared, compute_dtype)
            digest = prepared[0].region.patch_digest
            if any(item.region.patch_digest != digest for item in prepared):
                raise _refuse("patch-group")
            regions_tuple = tuple(item.region for item in prepared)
            subgroup_roles = tuple(item.role for item in subgroup)
            batch_sizes = tuple(item.crop.shape[0] for item in prepared)
            _check_cancel(cancel)
            if digest is None:
                value = evaluate(
                    regions_tuple,
                    grouped_x,
                    sigma,
                    conditioning,
                    subgroup_roles,
                    batch_sizes,
                )
            else:
                try:
                    with grouped_owner._owner_for(digest).activate(scale):  # pyright: ignore[reportPrivateUsage]
                        _check_cancel(cancel)
                        value = evaluate(
                            regions_tuple,
                            grouped_x,
                            sigma,
                            conditioning,
                            subgroup_roles,
                            batch_sizes,
                        )
                except ScaledPatchError as error:
                    raise _refuse("patch-activation", error.code) from None
            calls += 1
            subgroup_sizes.append(len(subgroup))
            if (
                not isinstance(value, torch.Tensor)  # pyright: ignore[reportUnnecessaryIsInstance]
                or value.shape != grouped_x.shape
                or value.dtype != grouped_x.dtype
                or value.device != grouped_x.device
            ):
                raise _refuse("callback-contract")
            _check_cancel(cancel)
            offset = 0
            for grouped_item in subgroup:
                _check_cancel(cancel)
                item = grouped_item.prepared
                batch = item.crop.shape[0]
                split = value.narrow(0, offset, batch)
                offset += batch
                out_view = outputs[grouped_item.role]
                count_view = counts[grouped_item.role]
                if item.area is not None:
                    height, width, y, x_offset = item.area
                    out_view = out_view.narrow(2, y, height).narrow(3, x_offset, width)
                    count_view = count_view.narrow(2, y, height).narrow(3, x_offset, width)
                out_view.add_(split * item.multiplier)
                count_view.add_(item.multiplier)
        for role in (GuidanceRole.CONDITIONAL, GuidanceRole.UNCONDITIONAL):
            _check_cancel(cancel)
            outputs[role].div_(counts[role])
        _check_cancel(cancel)
        return GroupedRegionalResult(
            outputs[GuidanceRole.CONDITIONAL],
            outputs[GuidanceRole.UNCONDITIONAL],
            calls,
            tuple(subgroup_sizes),
            staged_bytes,
        )


def flux_region_evaluator(
    model: Flux,
    *,
    guidance: float | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> RegionEvaluator:
    """Bind classic Flux single-condition math to the regional callback."""

    def evaluate(region: MaterializedRegion, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return FluxDenoiser(
            model,
            region.conditioning,
            guidance=guidance,
            compute_dtype=compute_dtype,
        )(x, sigma)

    return evaluate


def flux_grouped_region_evaluator(
    model: Flux,
    *,
    guidance: float | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> GroupedRegionEvaluator:
    """Bind one grouped classic-Flux forward to the dormant executor."""

    def evaluate(
        regions: tuple[MaterializedRegion, ...],
        x: torch.Tensor,
        sigma: float,
        conditioning: Conditioning[torch.Tensor],
        roles: tuple[GuidanceRole, ...],
        batch_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        del regions, roles, batch_sizes
        denoiser = FluxDenoiser(
            model,
            conditioning,
            guidance=guidance,
            compute_dtype=compute_dtype,
        )
        return denoiser._grouped_conditioning_forward(x, sigma, conditioning)  # pyright: ignore[reportPrivateUsage]

    return evaluate


def sd_region_evaluator(
    model: UNetModel,
    space: SigmaSpace,
    *,
    parameterization: Parameterization,
    adm: Callable[[MaterializedRegion, GuidanceRole], torch.Tensor | None] | None = None,
    compute_dtype: torch.dtype = torch.float16,
) -> RegionEvaluator:
    """Bind SD single-condition math to the regional callback."""
    from dinkster_inference import GuidanceRole

    def evaluate(region: MaterializedRegion, x: torch.Tensor, sigma: float) -> torch.Tensor:
        adm_cond = None if adm is None else adm(region, GuidanceRole.CONDITIONAL)
        return SDDenoiser(
            model,
            space,
            region.conditioning,
            adm_cond=adm_cond,
            parameterization=parameterization,
            compute_dtype=compute_dtype,
        )(x, sigma)

    return evaluate


def sd_grouped_region_evaluator(
    model: UNetModel,
    space: SigmaSpace,
    *,
    parameterization: Parameterization,
    adm: Callable[[MaterializedRegion, GuidanceRole], torch.Tensor | None] | None = None,
    compute_dtype: torch.dtype = torch.float16,
) -> GroupedRegionEvaluator:
    """Bind one grouped SD-era forward, including per-record ADM rows."""

    def evaluate(
        regions: tuple[MaterializedRegion, ...],
        x: torch.Tensor,
        sigma: float,
        conditioning: Conditioning[torch.Tensor],
        roles: tuple[GuidanceRole, ...],
        batch_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        adm_rows: list[torch.Tensor] = []
        if adm is not None:
            for region, role, batch in zip(regions, roles, batch_sizes, strict=True):
                value = adm(region, role)
                if value is None:
                    raise _refuse("adm-missing")
                if value.shape[0] == 1 and batch != 1:
                    value = value.repeat(batch, 1)
                elif value.shape[0] != batch:
                    raise _refuse("adm-batch")
                adm_rows.append(value)
        grouped_adm = None if not adm_rows else torch.cat(adm_rows, dim=0)
        denoiser = SDDenoiser(
            model,
            space,
            conditioning,
            adm_cond=grouped_adm,
            parameterization=parameterization,
            compute_dtype=compute_dtype,
        )
        return denoiser._grouped_conditioning_forward(  # pyright: ignore[reportPrivateUsage]
            x, sigma, conditioning, grouped_adm
        )

    return evaluate


__all__ = [
    "MASK_PAYLOAD_SPACE",
    "SCALE_PAYLOAD_SPACE",
    "GroupedRegionEvaluator",
    "GroupedRegionalResult",
    "MaterializedRegion",
    "PreparedGroupedPatches",
    "RegionEvaluator",
    "RegionalConditioningError",
    "evaluate_regions",
    "evaluate_grouped_regions",
    "flux_grouped_region_evaluator",
    "flux_region_evaluator",
    "materialize_regions",
    "prepare_grouped_patches",
    "realize_region_schedules",
    "sd_grouped_region_evaluator",
    "sd_region_evaluator",
]
