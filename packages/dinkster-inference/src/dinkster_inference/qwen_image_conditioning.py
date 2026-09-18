"""Torch-free Qwen Image reference-conditioning contracts.

The contracts follow ComfyUI 2a68ce33b4c9ea6ee4283e618a74560cefb32694.
They describe immutable values and plans without allocating or executing tensors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

QwenImageReferenceMethod = Literal["index", "negative_index", "spatial_offset"]
QwenImageMaskKind = Literal["non_floating", "floating"]

_WAN21_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
_WAN21_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.916,
)
_REFERENCE_METHODS = frozenset(("index", "negative_index", "spatial_offset"))


def _positive_int_tuple(value: object, length: int, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a {length}-integer tuple")
    items = cast("tuple[object, ...]", value)
    if len(items) != length:
        raise TypeError(f"{name} must be a {length}-integer tuple")
    if any(type(item) is not int for item in items):
        raise TypeError(f"{name} entries must be integers")
    result = cast("tuple[int, ...]", items)
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} entries must be positive")
    return result


@dataclass(frozen=True)
class QwenImageWan21Normalization:
    """Exact Wan21 process-in and process-out normalization facts."""

    channels: int = 16
    scale_factor: float = 1.0
    mean: tuple[float, ...] = _WAN21_MEAN
    std: tuple[float, ...] = _WAN21_STD
    process_in_formula: str = "(latent - mean) * scale_factor / std"
    process_out_formula: str = "latent * std / scale_factor + mean"

    def __post_init__(self) -> None:
        actual = (
            self.channels,
            self.scale_factor,
            self.mean,
            self.std,
            self.process_in_formula,
            self.process_out_formula,
        )
        expected = (
            16,
            1.0,
            _WAN21_MEAN,
            _WAN21_STD,
            "(latent - mean) * scale_factor / std",
            "latent * std / scale_factor + mean",
        )
        if any(
            type(value) is not type(required) or value != required
            for value, required in zip(actual, expected, strict=True)
        ):
            raise ValueError("normalization must equal the exact Wan21 facts")

    def _scalar(self, channel: int, value: float) -> float:
        if type(channel) is not int:
            raise TypeError("channel must be an integer")
        if not 0 <= channel < self.channels:
            raise ValueError("channel must be in the Wan21 16-channel range")
        if type(value) not in (int, float):
            raise TypeError("latent scalar must be an int or float")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("latent scalar must be finite")
        return result

    def process_in_scalar(self, channel: int, value: float) -> float:
        """Apply the reference process-in formula to one scalar."""
        scalar = self._scalar(channel, value)
        return (scalar - self.mean[channel]) * self.scale_factor / self.std[channel]

    def process_out_scalar(self, channel: int, value: float) -> float:
        """Apply the reference process-out formula to one scalar."""
        scalar = self._scalar(channel, value)
        return scalar * self.std[channel] / self.scale_factor + self.mean[channel]


QWEN_IMAGE_WAN21_NORMALIZATION = QwenImageWan21Normalization()


@dataclass(frozen=True)
class QwenImageLatentSnapshot:
    """Immutable BCTHW shape of one ordered, process-in reference latent."""

    shape: tuple[int, int, int, int, int]

    def __post_init__(self) -> None:
        shape = _positive_int_tuple(cast("object", self.shape), 5, "latent shape")
        if shape[1] != 16:
            raise ValueError("Qwen Image reference latents must have 16 channels")
        object.__setattr__(self, "shape", cast("tuple[int, int, int, int, int]", shape))

    @property
    def batch(self) -> int:
        return self.shape[0]

    @property
    def frames(self) -> int:
        return self.shape[2]

    @property
    def height(self) -> int:
        return self.shape[3]

    @property
    def width(self) -> int:
        return self.shape[4]

    @property
    def token_count(self) -> int:
        return self.frames * ((self.height + 1) // 2) * ((self.width + 1) // 2)


@dataclass(frozen=True)
class QwenImageCrossAttention:
    """Shape and owner of the Qwen Image cross-attention context."""

    shape: tuple[int, int, int]
    owner: Literal["c_crossattn"] = field(default="c_crossattn", init=False)

    def __post_init__(self) -> None:
        shape = _positive_int_tuple(cast("object", self.shape), 3, "cross-attention shape")
        if shape[2] != 3584:
            raise ValueError("Qwen Image cross-attention width must be 3584")
        object.__setattr__(self, "shape", cast("tuple[int, int, int]", shape))


@dataclass(frozen=True)
class QwenImageAttentionMask:
    """Optional batch-by-text mask retained by Qwen Image conditioning."""

    shape: tuple[int, int]
    kind: QwenImageMaskKind
    owner: Literal["attention_mask"] = field(default="attention_mask", init=False)

    def __post_init__(self) -> None:
        shape = _positive_int_tuple(cast("object", self.shape), 2, "attention mask")
        if self.kind not in ("non_floating", "floating"):
            raise ValueError("Qwen Image attention mask kind must be non_floating or floating")
        object.__setattr__(self, "shape", cast("tuple[int, int]", shape))


@dataclass(frozen=True)
class QwenImageConditioning:
    """Immutable model-owned cross-attention and reference inputs."""

    cross_attention: QwenImageCrossAttention
    attention_mask: QwenImageAttentionMask | None
    references: Sequence[QwenImageLatentSnapshot]
    reference_method: QwenImageReferenceMethod = "index"

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.cross_attention), QwenImageCrossAttention):
            raise TypeError("cross_attention must be QwenImageCrossAttention")
        if self.attention_mask is not None and not isinstance(
            cast("object", self.attention_mask), QwenImageAttentionMask
        ):
            raise TypeError("attention_mask must be QwenImageAttentionMask or None")
        if self.reference_method not in _REFERENCE_METHODS:
            raise ValueError("unsupported Qwen Image reference method")
        references_obj = cast("object", self.references)
        if not isinstance(references_obj, Sequence) or isinstance(references_obj, (str, bytes)):
            raise TypeError("references must be an ordered sequence")
        references = tuple(cast("Sequence[object]", references_obj))
        if any(not isinstance(item, QwenImageLatentSnapshot) for item in references):
            raise TypeError("every reference must be QwenImageLatentSnapshot")
        typed_references = cast("tuple[QwenImageLatentSnapshot, ...]", references)
        batch, text_tokens, _ = self.cross_attention.shape
        if self.attention_mask is not None and self.attention_mask.shape != (
            batch,
            text_tokens,
        ):
            raise ValueError("attention mask must match cross-attention batch and tokens")
        if any(reference.batch != batch for reference in typed_references):
            raise ValueError("reference batch must match cross-attention batch")
        object.__setattr__(self, "references", typed_references)


@dataclass(frozen=True)
class QwenImageReferencePlacement:
    """One ordered reference's token range and process_img arguments."""

    latent: QwenImageLatentSnapshot
    index: int
    height_offset: int
    width_offset: int
    patch_height_offset: int
    patch_width_offset: int
    token_offset: int
    token_count: int

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.latent), QwenImageLatentSnapshot):
            raise TypeError("placement latent must be QwenImageLatentSnapshot")
        values = (
            self.index,
            self.height_offset,
            self.width_offset,
            self.patch_height_offset,
            self.patch_width_offset,
            self.token_offset,
            self.token_count,
        )
        if any(type(value) is not int for value in values):
            raise TypeError("reference placement values must be integers")
        if any(value < 0 for value in values[1:-1]) or self.token_count <= 0:
            raise ValueError("reference offsets must be nonnegative and count positive")
        if self.token_count != self.latent.token_count:
            raise ValueError("reference token count must match its latent shape")
        if self.patch_height_offset != (self.height_offset + 1) // 2:
            raise ValueError("patch height offset must match the 2x2 packing offset")
        if self.patch_width_offset != (self.width_offset + 1) // 2:
            raise ValueError("patch width offset must match the 2x2 packing offset")

    @property
    def temporal_id_range(self) -> tuple[int, int]:
        """Return process_img's effective inclusive temporal ID range."""
        if self.latent.frames == 1:
            return (self.index, self.index)
        return (0, self.latent.frames - 1)


def _reference_placements(
    target: QwenImageLatentSnapshot,
    references: tuple[QwenImageLatentSnapshot, ...],
    method: QwenImageReferenceMethod,
) -> tuple[QwenImageReferencePlacement, ...]:
    if any(reference.batch != target.batch for reference in references):
        raise ValueError("reference batch must match target batch")
    placements: list[QwenImageReferencePlacement] = []
    token_offset = target.token_count
    packed_height = 0
    packed_width = 0
    index = 0
    for reference in references:
        height_offset = 0
        width_offset = 0
        if method == "index":
            index += 1
        elif method == "negative_index":
            index -= 1
        else:
            index = 1
            if reference.height + packed_height > reference.width + packed_width:
                width_offset = packed_width
            else:
                height_offset = packed_height
            packed_height = max(packed_height, reference.height + height_offset)
            packed_width = max(packed_width, reference.width + width_offset)
        placement = QwenImageReferencePlacement(
            latent=reference,
            index=index,
            height_offset=height_offset,
            width_offset=width_offset,
            patch_height_offset=(height_offset + 1) // 2,
            patch_width_offset=(width_offset + 1) // 2,
            token_offset=token_offset,
            token_count=reference.token_count,
        )
        placements.append(placement)
        token_offset += reference.token_count
    return tuple(placements)


@dataclass(frozen=True)
class QwenImageReferencePlan:
    """Deterministic target-first reference packing plan."""

    target: QwenImageLatentSnapshot
    method: QwenImageReferenceMethod
    references: tuple[QwenImageReferencePlacement, ...]
    target_token_count: int
    reference_token_counts: tuple[int, ...]
    total_image_tokens: int
    flow_multiplier: float = 1.0
    flow_shift: float = 1.15

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.target), QwenImageLatentSnapshot):
            raise TypeError("plan target must be QwenImageLatentSnapshot")
        if self.method not in _REFERENCE_METHODS:
            raise ValueError("unsupported Qwen Image reference method")
        references_obj = cast("object", self.references)
        if type(references_obj) is not tuple:
            raise TypeError("plan references must be a placement tuple")
        reference_items = cast("tuple[object, ...]", references_obj)
        if any(not isinstance(item, QwenImageReferencePlacement) for item in reference_items):
            raise TypeError("plan references must be a placement tuple")
        typed_references = cast("tuple[QwenImageReferencePlacement, ...]", reference_items)
        expected_placements = _reference_placements(
            self.target,
            tuple(item.latent for item in typed_references),
            self.method,
        )
        if typed_references != expected_placements:
            raise ValueError("reference placements must equal the exact packing plan")
        if type(self.target_token_count) is not int:
            raise TypeError("target token count must be an integer")
        if type(self.reference_token_counts) is not tuple or any(
            type(count) is not int for count in self.reference_token_counts
        ):
            raise TypeError("reference token counts must be an integer tuple")
        if type(self.total_image_tokens) is not int:
            raise TypeError("total image tokens must be an integer")
        expected_counts = tuple(item.token_count for item in self.references)
        if self.target_token_count != self.target.token_count:
            raise ValueError("target token count must match its latent shape")
        if self.reference_token_counts != expected_counts:
            raise ValueError("reference token counts must preserve placement order")
        if self.total_image_tokens != self.target_token_count + sum(expected_counts):
            raise ValueError("total image tokens must cover target and references")
        if (
            type(self.flow_multiplier) is not float
            or self.flow_multiplier != 1.0
            or type(self.flow_shift) is not float
            or self.flow_shift != 1.15
        ):
            raise ValueError("plan must use the exact Qwen Image flow")


def plan_qwen_image_references(
    target: QwenImageLatentSnapshot,
    references: Sequence[QwenImageLatentSnapshot],
    method: QwenImageReferenceMethod = "index",
) -> QwenImageReferencePlan:
    """Plan target-first 2x2 token packing without tensor allocation."""
    if not isinstance(cast("object", target), QwenImageLatentSnapshot):
        raise TypeError("target must be QwenImageLatentSnapshot")
    if method not in _REFERENCE_METHODS:
        raise ValueError("unsupported Qwen Image reference method")
    references_obj = cast("object", references)
    if not isinstance(references_obj, Sequence) or isinstance(references_obj, (str, bytes)):
        raise TypeError("references must be an ordered sequence")
    snapshot = tuple(cast("Sequence[object]", references_obj))
    if any(not isinstance(item, QwenImageLatentSnapshot) for item in snapshot):
        raise TypeError("every reference must be QwenImageLatentSnapshot")
    typed_references = cast("tuple[QwenImageLatentSnapshot, ...]", snapshot)
    placements = _reference_placements(target, typed_references, method)
    counts = tuple(item.token_count for item in placements)
    return QwenImageReferencePlan(
        target=target,
        method=method,
        references=placements,
        target_token_count=target.token_count,
        reference_token_counts=counts,
        total_image_tokens=target.token_count + sum(counts),
    )


__all__ = [
    "QWEN_IMAGE_WAN21_NORMALIZATION",
    "QwenImageAttentionMask",
    "QwenImageConditioning",
    "QwenImageCrossAttention",
    "QwenImageLatentSnapshot",
    "QwenImageMaskKind",
    "QwenImageReferenceMethod",
    "QwenImageReferencePlacement",
    "QwenImageReferencePlan",
    "QwenImageWan21Normalization",
    "plan_qwen_image_references",
]
