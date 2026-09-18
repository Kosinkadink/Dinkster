"""Model-facing declarations for persistent sequence sharding."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

import torch
from dinkster_inference import SequencePartition, SequenceShard, plan_sequence_partition

from .attention import AttentionKernel
from .sequence_parallel_attention import SequenceParallelAttentionKernel


@dataclass(frozen=True, slots=True)
class PackedSequenceFacts:
    """Immutable boundaries and layout identity for one packed sequence."""

    sequence_length: int
    segments: tuple[tuple[int, int, str], ...]
    layout_identity: str

    def __post_init__(self) -> None:
        if type(self.sequence_length) is not int:
            raise TypeError("sequence_length must be an exact int")
        if type(self.segments) is not tuple:
            raise TypeError("segments must be an exact tuple")
        if not self.segments:
            raise ValueError("segments must be nonempty")
        expected_start = 0
        for segment in self.segments:
            if type(segment) is not tuple or len(segment) != 3:
                raise TypeError("segments must contain exact tuples of start, stop, and kind")
            start, stop, kind = segment
            if type(start) is not int or type(stop) is not int:
                raise TypeError("segment boundaries must be exact ints")
            if start != expected_start:
                raise ValueError("segments must be contiguous from zero")
            if stop <= start:
                raise ValueError("segment stop must be greater than start")
            if type(kind) is not str or not kind or "\n" in kind:
                raise TypeError("segment kind must be a non-empty newline-free exact string")
            expected_start = stop
        if expected_start != self.sequence_length:
            raise ValueError("final segment stop must equal sequence_length")
        if (
            type(self.layout_identity) is not str
            or not self.layout_identity
            or "\n" in self.layout_identity
        ):
            raise TypeError("layout_identity must be a non-empty newline-free exact string")


_PackedFactsT = TypeVar("_PackedFactsT", bound=PackedSequenceFacts, contravariant=True)


class SequenceAttentionKernelFactory(Protocol[_PackedFactsT]):
    """Construct one attention kernel from a packed sequence declaration."""

    def __call__(self, facts: _PackedFactsT, /) -> AttentionKernel: ...


SequenceGather = Callable[[torch.Tensor, SequencePartition, SequenceShard], torch.Tensor]


@dataclass(frozen=True, slots=True)
class SequenceSharding:
    """Bind one packed invocation to its local sequence shard."""

    facts: PackedSequenceFacts
    partition: SequencePartition
    shard: SequenceShard
    gather: SequenceGather

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.facts), PackedSequenceFacts):
            raise TypeError("facts must be PackedSequenceFacts")
        if type(self.partition) is not SequencePartition:
            raise TypeError("partition must be an exact SequencePartition")
        if type(self.shard) is not SequenceShard:
            raise TypeError("shard must be an exact SequenceShard")
        canonical = plan_sequence_partition(self.facts.sequence_length, self.partition.shard_count)
        if self.partition != canonical:
            raise ValueError("partition must be canonical for the packed sequence facts")
        if self.shard not in self.partition.shards:
            raise ValueError("shard must belong to the partition")
        if not callable(self.gather):
            raise TypeError("gather must be callable")
        gather = cast("object", self.gather)
        if inspect.iscoroutinefunction(gather) or inspect.iscoroutinefunction(
            type(gather).__call__
        ):
            raise TypeError("gather must be synchronous")

    def validate_kernel(self, kernel: AttentionKernel, head_count: int) -> None:
        if not isinstance(cast("object", kernel), SequenceParallelAttentionKernel):
            raise TypeError("sequence sharding requires a SequenceParallelAttentionKernel")
        sequence_kernel = cast(SequenceParallelAttentionKernel, kernel)
        layout = sequence_kernel.layout
        if layout is None:
            raise ValueError("sequence sharding requires a multi-rank kernel layout")
        if sequence_kernel.mesh.ulysses * sequence_kernel.mesh.ring != self.partition.shard_count:
            raise ValueError("kernel sequence mesh does not match the partition")
        if layout.original_token_order_reference != self.facts.layout_identity:
            raise ValueError("kernel layout identity does not match the packed facts")
        if layout.global_sequence_length != self.facts.sequence_length:
            raise ValueError("kernel layout sequence length does not match the packed facts")
        if layout.shard != self.shard:
            raise ValueError("kernel layout shard does not match the requested shard")
        if layout.head_count != head_count:
            raise ValueError("kernel layout head count does not match the model attention geometry")


__all__ = [
    "PackedSequenceFacts",
    "SequenceAttentionKernelFactory",
    "SequenceGather",
    "SequenceSharding",
]
