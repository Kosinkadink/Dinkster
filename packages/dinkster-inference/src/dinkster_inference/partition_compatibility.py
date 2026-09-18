"""Torch-free attention partition declarations and feasibility checks."""

from __future__ import annotations

from dataclasses import dataclass

from .devices import DType
from .process_mesh import ProcessMesh
from .sequence_partition import (
    SequencePartition,
    SequencePartitionError,
    plan_sequence_partition,
)

__all__ = [
    "ContiguousShard",
    "FullSequence",
    "PartitionCompatibility",
    "PartitionCompatibilityError",
    "Replicated",
    "RingSequenceShard",
    "UlyssesHeadScatter",
    "UlyssesRingHybrid",
    "require_attention_partition_feasibility",
]


class PartitionCompatibilityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Replicated:
    pass


@dataclass(frozen=True, slots=True)
class UlyssesHeadScatter:
    pass


@dataclass(frozen=True, slots=True)
class RingSequenceShard:
    pass


@dataclass(frozen=True, slots=True)
class UlyssesRingHybrid:
    pass


@dataclass(frozen=True, slots=True)
class FullSequence:
    pass


@dataclass(frozen=True, slots=True)
class ContiguousShard:
    sequence_dimension: int
    equal_chunk: bool = True

    def __post_init__(self) -> None:
        if type(self.sequence_dimension) is not int or self.sequence_dimension < 0:
            raise PartitionCompatibilityError("sequence_dimension must be an exact int >= 0")
        if type(self.equal_chunk) is not bool:
            raise PartitionCompatibilityError("equal_chunk must be an exact bool")


_MODE_TYPES = (Replicated, UlyssesHeadScatter, RingSequenceShard, UlyssesRingHybrid)
_EXPECTATION_TYPES = (FullSequence, ContiguousShard)


@dataclass(frozen=True, slots=True)
class PartitionCompatibility:
    """Caller-declared support, not measured provider qualification.

    Revision 1 versions both the mode and tensor-expectation vocabularies.
    """

    revision: int
    modes: tuple[Replicated | UlyssesHeadScatter | RingSequenceShard | UlyssesRingHybrid, ...]
    tensor_expectation: FullSequence | ContiguousShard
    supported_dtypes: tuple[DType, ...]
    device_kinds: tuple[str, ...]
    provides_matching_block_normalization: bool

    def __post_init__(self) -> None:
        if type(self.revision) is not int or self.revision != 1:
            raise PartitionCompatibilityError("revision must be the exact int 1")
        if type(self.modes) is not tuple or not self.modes:
            raise PartitionCompatibilityError("modes must be a nonempty tuple")
        if any(type(mode) not in _MODE_TYPES for mode in self.modes):
            raise PartitionCompatibilityError("modes must contain exact supported mode values")
        if len(set(self.modes)) != len(self.modes):
            raise PartitionCompatibilityError("modes must not contain duplicates")
        if type(self.tensor_expectation) not in _EXPECTATION_TYPES:
            raise PartitionCompatibilityError(
                "tensor_expectation must be an exact FullSequence or ContiguousShard"
            )
        if type(self.supported_dtypes) is not tuple or any(
            type(dtype) is not DType for dtype in self.supported_dtypes
        ):
            raise PartitionCompatibilityError(
                "supported_dtypes must be a tuple of exact DType values"
            )
        if type(self.device_kinds) is not tuple or any(
            type(kind) is not str or not kind for kind in self.device_kinds
        ):
            raise PartitionCompatibilityError(
                "device_kinds must be a tuple of exact nonempty strings"
            )
        if type(self.provides_matching_block_normalization) is not bool:
            raise PartitionCompatibilityError(
                "provides_matching_block_normalization must be an exact bool"
            )


def require_attention_partition_feasibility(
    head_count: int,
    sequence_length: int,
    sequence_dimension: int,
    mesh: ProcessMesh,
    compatibility: PartitionCompatibility,
    dtype: DType,
    device_kind: str,
) -> SequencePartition:
    """Require a declaration to support one candidate attention placement.

    Tensor expectations describe inputs at the partition placement boundary.
    A contiguous placement input does not imply partial K/V for Ulysses local
    attention after its head-scatter redistribution.
    """

    if type(head_count) is not int or head_count < 1:
        raise PartitionCompatibilityError("head_count must be an exact int >= 1")
    if type(sequence_length) is not int or sequence_length < 1:
        raise PartitionCompatibilityError("sequence_length must be an exact int >= 1")
    if type(sequence_dimension) is not int or sequence_dimension < 0:
        raise PartitionCompatibilityError("sequence_dimension must be an exact int >= 0")
    if type(mesh) is not ProcessMesh:
        raise PartitionCompatibilityError("mesh must be an exact ProcessMesh")
    if type(compatibility) is not PartitionCompatibility:
        raise PartitionCompatibilityError("compatibility must be an exact PartitionCompatibility")
    if type(dtype) is not DType:
        raise PartitionCompatibilityError("dtype must be an exact DType")
    if type(device_kind) is not str or not device_kind:
        raise PartitionCompatibilityError("device_kind must be an exact nonempty string")

    if mesh.sp_ulysses == 1 and mesh.sp_ring == 1:
        required_mode = Replicated()
    elif mesh.sp_ring == 1:
        required_mode = UlyssesHeadScatter()
    elif mesh.sp_ulysses == 1:
        required_mode = RingSequenceShard()
    else:
        required_mode = UlyssesRingHybrid()

    if required_mode not in compatibility.modes:
        raise PartitionCompatibilityError(
            f"partition mode {type(required_mode).__name__} is not declared"
        )
    if head_count % mesh.sp_ulysses != 0:
        raise PartitionCompatibilityError("head_count must be divisible by sp_ulysses")
    if dtype not in compatibility.supported_dtypes:
        raise PartitionCompatibilityError(f"dtype {dtype.name!r} is not supported")
    if device_kind not in compatibility.device_kinds:
        raise PartitionCompatibilityError(f"device kind {device_kind!r} is not supported")

    shard_count = mesh.sp_ulysses * mesh.sp_ring
    expectation = compatibility.tensor_expectation
    if isinstance(expectation, ContiguousShard):
        if expectation.sequence_dimension != sequence_dimension:
            raise PartitionCompatibilityError(
                "contiguous-shard sequence dimension does not match the candidate"
            )
        if not expectation.equal_chunk:
            raise PartitionCompatibilityError(
                "contiguous-shard expectation must require equal chunks"
            )
    elif shard_count != 1:
        raise PartitionCompatibilityError(
            "full-sequence tensor expectation does not support sequence partitioning"
        )

    if mesh.sp_ring > 1 and not compatibility.provides_matching_block_normalization:
        raise PartitionCompatibilityError(
            "Ring requires normalization from the same local block callable"
        )

    try:
        return plan_sequence_partition(sequence_length, shard_count)
    except SequencePartitionError as error:
        raise PartitionCompatibilityError(f"sequence partition is infeasible: {error}") from error
