"""Weight sources: inspect checkpoints without materializing them.

ComfyUI loads the whole state dict and then guesses (comfy/sd.py,
comfy/utils.py @ b78cec87). Dinkster's contract starts from the safetensors
reality: a checkpoint is a header (keys -> geometry + byte ranges) plus
a payload you read selectively. Detection (stage 2) needs only the
header; loading (stage 4) reads slices into planned buffers.

The prefix transforms are pure reimplementations of
comfy/utils.py state_dict_prefix_replace/state_dict_filter (@ b78cec87),
generic over the value type so they work on geometries, tensors, or
anything else keyed like a state dict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from .devices import DType

V = TypeVar("V")


@dataclass(frozen=True)
class TensorGeometry:
    """Shape and dtype without a tensor - enough for planning.

    (Concept borrowed from comfy/memory_management.py TensorGeometry
    @ b78cec87, minus the ad hoc ``any`` typing.)
    """

    shape: tuple[int, ...]
    dtype: DType

    def __post_init__(self) -> None:
        if any(d < 0 for d in self.shape):
            raise ValueError(f"shape dims must be >= 0, got {self.shape}")

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        """Byte size of a packed buffer holding this tensor. Sub-byte
        dtypes round the total up to whole bytes (never per element)."""
        return (self.numel * self.dtype.bits + 7) // 8


@dataclass(frozen=True)
class RowChunk:
    """Model tensor = one of ``parts`` equal axis-0 chunks of the
    source tensor. The reference splits OpenCLIP's fused
    ``attn.in_proj_weight``/``in_proj_bias`` into q/k/v this way
    (comfy/utils.py transformers_convert @ b78cec87); the plan
    records the split so the executor never re-derives it."""

    part: int
    parts: int

    def __post_init__(self) -> None:
        if self.parts < 2:
            raise ValueError(f"parts must be >= 2, got {self.parts}")
        if not 0 <= self.part < self.parts:
            raise ValueError(f"part must be in [0, {self.parts}), got {self.part}")


@dataclass(frozen=True)
class Transpose2D:
    """Model tensor = the rank-2 source tensor transposed. The
    reference stores OpenCLIP's ``text_projection`` as an ``x @ W``
    matrix and transposes it into the transformers Linear layout
    (comfy/utils.py clip_text_transformers_convert @ b78cec87)."""


@dataclass(frozen=True)
class LinearToConv2D:
    """Model tensor = a rank-2 linear weight reshaped to a 1x1 Conv2D
    weight. Diffusers stores AutoencoderKL mid-attention projections
    as Linear weights while the canonical SD layout uses 1x1 convs
    (comfy/diffusers_convert.py @ f4b99bc6)."""


#: How a model tensor is derived from its source tensor when a plain
#: rename is not enough. Transforms never change dtype.
TensorTransform = RowChunk | Transpose2D | LinearToConv2D


def transformed_geometry(geometry: TensorGeometry, transform: TensorTransform) -> TensorGeometry:
    """The model-side geometry a transform produces from a source
    geometry; refuses shapes the transform cannot take."""
    if isinstance(transform, RowChunk):
        if not geometry.shape:
            raise ValueError("RowChunk needs at least one axis")
        rows, rem = divmod(geometry.shape[0], transform.parts)
        if rem:
            raise ValueError(
                f"axis 0 of {geometry.shape} does not split into {transform.parts} equal chunks"
            )
        return TensorGeometry((rows, *geometry.shape[1:]), geometry.dtype)
    if isinstance(transform, LinearToConv2D):
        if len(geometry.shape) != 2:
            raise ValueError(f"LinearToConv2D needs a rank-2 tensor, got shape {geometry.shape}")
        return TensorGeometry((*geometry.shape, 1, 1), geometry.dtype)
    if len(geometry.shape) != 2:
        raise ValueError(f"Transpose2D needs a rank-2 tensor, got shape {geometry.shape}")
    return TensorGeometry((geometry.shape[1], geometry.shape[0]), geometry.dtype)


@dataclass(frozen=True)
class WeightEntry:
    """One tensor in a weight source: geometry plus its byte range."""

    key: str
    geometry: TensorGeometry
    offset: int
    nbytes: int

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("weight key must not be empty")
        if self.offset < 0 or self.nbytes < 0:
            raise ValueError("offset and nbytes must be >= 0")


class WeightSource(Protocol):
    """A checkpoint you can interrogate before you commit to reading it.

    Implementations wrap safetensors files, CAS blobs, or in-memory
    dicts. Reading payload bytes is a separate (stage 4) concern; this
    surface is what detection and planning consume.
    """

    def keys(self) -> Sequence[str]: ...

    def entry(self, key: str) -> WeightEntry: ...

    def metadata(self) -> Mapping[str, str]: ...


@runtime_checkable
class ConfigurationScalarSource(Protocol):
    """Optional payload seam for behavior-bearing scalar configuration.

    Family detection remains header-only. A planner may require this seam
    after detection when an explicitly recognized scalar tensor selects
    execution math and therefore must enter pre-load structural identity.
    Implementations return the exact finite Python-float value represented
    by one floating-point tensor element and refuse every other shape/dtype.
    """

    def read_float_scalar(self, key: str) -> float: ...


@runtime_checkable
class ConfigurationPayloadSource(Protocol):
    """Optional, bounded seam for explicit uint8 configuration tensors."""

    def read_uint8_configuration(self, key: str) -> bytes: ...


@runtime_checkable
class AssetIdentifiedSource(Protocol):
    """Optional identity seam naming the immutable asset a source fronts.

    Planning that must bind provider-artifact identity (content digest
    and byte size) into runtime identity facts narrows through this seam
    and refuses sources that cannot prove which artifact they front.
    ``None`` values mean the source is unidentified.
    """

    @property
    def asset_digest(self) -> str | None: ...

    @property
    def asset_size(self) -> int | None: ...


def filter_prefix(sd: Mapping[str, V], prefix: str, *, strip: bool = True) -> dict[str, V]:
    """Entries whose key starts with ``prefix``, optionally stripped.

    Insertion order is preserved. An empty prefix returns a copy.
    """
    if strip:
        return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    return {k: v for k, v in sd.items() if k.startswith(prefix)}


def replace_prefix(sd: Mapping[str, V], old: str, new: str) -> dict[str, V]:
    """All entries, with ``old`` swapped for ``new`` on matching keys."""
    return {(new + k[len(old) :] if k.startswith(old) else k): v for k, v in sd.items()}


def count_prefix(sd: Mapping[str, V], prefix: str) -> int:
    """How many keys start with ``prefix`` - the detection signal used
    by unet_prefix_from_state_dict (comfy/model_detection.py @ b78cec87)."""
    return sum(1 for k in sd if k.startswith(prefix))


__all__ = [
    "AssetIdentifiedSource",
    "LinearToConv2D",
    "ConfigurationScalarSource",
    "RowChunk",
    "TensorGeometry",
    "TensorTransform",
    "Transpose2D",
    "WeightEntry",
    "WeightSource",
    "count_prefix",
    "filter_prefix",
    "replace_prefix",
    "transformed_geometry",
]
