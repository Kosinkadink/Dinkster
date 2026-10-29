"""Rank-4 sequence-parallel attention over Ulysses and Ring exchange."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import torch
from dinkster_inference import (
    ManifestConsensusToken,
    ManifestRefusal,
    ManifestRefusalCode,
    PlacementMap,
    SequenceLayout,
    SequencePartition,
    SequencePartitionError,
    SequenceShard,
    UspMesh,
    plan_sequence_partition,
)

from .attention import AttentionKernel
from .sequence_exchange import (
    AttentionExchangeBackend,
    RingSequenceExchange,
    SequenceExchangeCompletionEvent,
    SequenceExchangeSubmissions,
    UlyssesSequenceExchange,
    UspSequenceExchange,
)


class SequenceParallelAttentionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SequenceParallelAttentionKernel:
    mesh: UspMesh
    inner: AttentionKernel
    layout: SequenceLayout | None = None
    placement: PlacementMap | None = None
    consensus_token: ManifestConsensusToken | None = None
    routed_attention_backend_identity: str | None = None
    exchange_backend_identity: str | None = None
    _exchange: AttentionExchangeBackend | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.mesh) is not UspMesh:
            raise SequenceParallelAttentionError("mesh must be an exact UspMesh")
        if not isinstance(cast("object", self.inner), AttentionKernel):
            raise SequenceParallelAttentionError("inner must implement AttentionKernel")
        active_placement = (
            PlacementMap.identity(self.mesh.process_mesh)
            if self.placement is None
            else self.placement
        )
        if type(active_placement) is not PlacementMap:
            raise SequenceParallelAttentionError("placement must be an exact PlacementMap or None")
        if active_placement.mesh_digest != self.mesh.digest:
            raise SequenceParallelAttentionError("placement mesh digest does not match the mesh")
        if len(active_placement.physical_ranks) != self.mesh.world_size:
            raise SequenceParallelAttentionError(
                "placement rank count must equal the mesh world size"
            )
        object.__setattr__(self, "placement", active_placement)
        if self.mesh.ulysses == 1 and self.mesh.ring == 1:
            if self.layout is not None:
                raise SequenceParallelAttentionError(
                    "degenerate sequence attention does not accept a SequenceLayout"
                )
            if any(
                value is not None
                for value in (
                    self.consensus_token,
                    self.routed_attention_backend_identity,
                    self.exchange_backend_identity,
                )
            ):
                raise SequenceParallelAttentionError(
                    "degenerate sequence attention does not accept distributed manifest evidence"
                )
            return
        if type(self.layout) is not SequenceLayout:
            raise SequenceParallelAttentionError(
                "multi-rank sequence attention requires an exact SequenceLayout"
            )
        if self.layout.mesh_identity != self.mesh.digest:
            raise SequenceParallelAttentionError("layout mesh identity does not match the mesh")
        if self.layout.head_count % self.mesh.ulysses:
            raise SequenceParallelAttentionError(
                "head count must be divisible by the Ulysses axis degree"
            )
        try:
            partition = plan_sequence_partition(
                self.layout.global_sequence_length, self.mesh.ulysses * self.mesh.ring
            )
        except SequencePartitionError as error:
            raise SequenceParallelAttentionError(
                "global sequence length cannot use the sequence mesh partition"
            ) from error
        if self.layout.shard not in partition.shards:
            raise SequenceParallelAttentionError(
                "layout shard must belong to the canonical sequence partition"
            )
        heads_per_rank = self.layout.head_count // self.mesh.ulysses
        ulysses_index = self.layout.shard.index // self.mesh.ring
        expected_head_start = ulysses_index * heads_per_rank
        if (self.layout.head_start, self.layout.head_stop) != (
            expected_head_start,
            expected_head_start + heads_per_rank,
        ):
            raise SequenceParallelAttentionError(
                "layout head ownership does not match its Ulysses shard"
            )
        if type(self.consensus_token) is not ManifestConsensusToken:
            raise ManifestRefusal(
                ManifestRefusalCode.FORGED_CONSENSUS_TOKEN,
                "multi-rank sequence attention requires an exact minted consensus token",
            )
        if (
            self.consensus_token.group_size != self.mesh.ulysses * self.mesh.ring
            or self.consensus_token.rank != self.layout.shard.index
        ):
            raise SequenceParallelAttentionError(
                "consensus token does not match the local sequence group rank"
            )
        for name, value in (
            ("routed attention backend identity", self.routed_attention_backend_identity),
            ("exchange backend identity", self.exchange_backend_identity),
        ):
            if type(value) is not str or not value or "\n" in value:
                raise SequenceParallelAttentionError(
                    f"{name} must be a non-empty newline-free exact string"
                )
        backend: AttentionExchangeBackend
        if self.mesh.ring == 1:
            backend = UlyssesSequenceExchange(
                self.mesh,
                active_placement,
                consensus_token=self.consensus_token,
            )
        elif self.mesh.ulysses == 1:
            backend = RingSequenceExchange(
                self.mesh,
                active_placement,
                consensus_token=self.consensus_token,
            )
        else:
            backend = UspSequenceExchange(
                self.mesh,
                active_placement,
                consensus_token=self.consensus_token,
            )
        object.__setattr__(self, "_exchange", backend)

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        output, _completions = self._attend(
            q,
            k,
            v,
            mask=mask,
            causal=causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        return output

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        causal: bool,
        scale: float | None,
        enable_gqa: bool,
    ) -> tuple[torch.Tensor, tuple[SequenceExchangeCompletionEvent, ...]]:
        if self._exchange is None:
            return (
                self.inner(
                    q,
                    k,
                    v,
                    mask=mask,
                    causal=causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                ),
                (),
            )
        if any(type(tensor) is not torch.Tensor for tensor in (q, k, v)):
            raise SequenceParallelAttentionError("q, k, and v must be exact torch.Tensor values")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise SequenceParallelAttentionError("q, k, and v must have rank 4 [B, H, S, D]")
        if q.shape != k.shape or q.shape != v.shape:
            raise SequenceParallelAttentionError(
                "multi-rank q, k, and v must have identical local shapes"
            )
        if not q.is_floating_point() or not k.is_floating_point() or not v.is_floating_point():
            raise SequenceParallelAttentionError("q, k, and v must use floating-point dtypes")
        if q.dtype != k.dtype or q.dtype != v.dtype or q.device != k.device or q.device != v.device:
            raise SequenceParallelAttentionError("q, k, and v must share one dtype and device")
        if self.mesh.ring > 1 and q.dtype is not torch.float32:
            raise SequenceParallelAttentionError(
                "multi-rank Ring attention requires float32 q, k, and v"
            )
        if type(causal) is not bool:
            raise SequenceParallelAttentionError("causal must be an exact bool")
        if type(enable_gqa) is not bool:
            raise SequenceParallelAttentionError("enable_gqa must be an exact bool")
        if mask is not None or causal:
            raise SequenceParallelAttentionError(
                "multi-rank sequence attention does not accept mask or causal attention"
            )
        if enable_gqa:
            raise SequenceParallelAttentionError(
                "multi-rank sequence attention does not accept grouped-query attention"
            )
        scale_value = cast("object", scale)
        if scale_value is not None and (
            isinstance(scale_value, bool)
            or not isinstance(scale_value, (int, float))
            or not math.isfinite(scale_value)
        ):
            raise SequenceParallelAttentionError(
                "scale must be None or a finite int or float, excluding bool"
            )
        assert self.layout is not None
        result = self._exchange.attend_submissions(
            SequenceExchangeSubmissions.from_tensors(q, k, v),
            self.layout,
            self.inner,
            scale=scale,
        )
        return result.output, result.completions

    def attend_with_completions(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> tuple[torch.Tensor, tuple[SequenceExchangeCompletionEvent, ...]]:
        return self._attend(
            q,
            k,
            v,
            mask=None,
            causal=False,
            scale=scale,
            enable_gqa=False,
        )

    def gather_sequence(
        self,
        hidden: torch.Tensor,
        partition: SequencePartition,
        shard: SequenceShard,
    ) -> torch.Tensor:
        if self._exchange is None:
            if partition.shard_count != 1 or shard != partition.shards[0]:
                raise SequenceParallelAttentionError(
                    "degenerate sequence gather requires a single canonical shard"
                )
            return hidden[:, : shard.valid_rows]
        return self._exchange.gather_sequence(hidden, partition, shard)


def gather_sequence_hidden(
    kernel: SequenceParallelAttentionKernel,
    hidden: torch.Tensor,
    partition: SequencePartition,
    shard: SequenceShard,
) -> torch.Tensor:
    """Gather valid hidden rows in canonical absolute sequence order."""
    if type(kernel) is not SequenceParallelAttentionKernel:
        raise SequenceParallelAttentionError(
            "sequence gather requires an exact SequenceParallelAttentionKernel"
        )
    return kernel.gather_sequence(hidden, partition, shard)


__all__ = [
    "SequenceParallelAttentionError",
    "SequenceParallelAttentionKernel",
    "gather_sequence_hidden",
]
