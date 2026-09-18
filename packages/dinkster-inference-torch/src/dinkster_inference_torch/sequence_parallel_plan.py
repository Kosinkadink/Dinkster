"""Frozen compiled plans for sequence-parallel attention."""

from __future__ import annotations

import hashlib
import json
from dataclasses import InitVar, asdict, dataclass
from typing import cast

from dinkster_inference import (
    CompiledPlanSlot,
    PlacementMap,
    SequenceLayout,
    SequencePartition,
    UspMesh,
)
from dinkster_inference.devices import DType
from dinkster_inference.partition_compatibility import (
    PartitionCompatibility,
    require_attention_partition_feasibility,
)

from .attention import AttentionBlockKernel, AttentionKernel
from .sequence_exchange import (
    RING_ACCUMULATION_DTYPE,
    RING_ACCUMULATION_ORDER,
    RING_MERGE_ALGORITHM,
    RING_TRAVERSAL_ORDER,
)
from .sequence_sharding import PackedSequenceFacts

USP_SLOT_NAME = "usp"
USP_EXCHANGE_ULYSSES = "dinkster.sequence-exchange.ulysses.v1"
USP_EXCHANGE_RING = "dinkster.sequence-exchange.ring.v1"
USP_EXCHANGE_HYBRID = "dinkster.sequence-exchange.ulysses-ring.v1"

_PLAN_DOMAIN = "dinkster.usp.compiled-plan.v2"
_PACKING_DOMAIN = "dinkster.usp.packing-geometry.v2"
_RANK_PLAN_DOMAIN = "dinkster.usp.rank-plan.v1"
_COMPILE_AUTHORITY: object = object()


def _digest_lines(lines: tuple[str, ...]) -> str:
    hasher = hashlib.sha256()
    for line in lines:
        hasher.update(line.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _require_identity(name: str, value: str) -> None:
    if type(value) is not str or not value or "\n" in value:
        raise ValueError(f"{name} must be a non-empty newline-free exact string")


def packing_geometry_digest(facts: PackedSequenceFacts) -> str:
    """Digest canonical packed sequence geometry."""
    if not isinstance(cast("object", facts), PackedSequenceFacts):
        raise TypeError("facts must be PackedSequenceFacts")
    lines = [
        _PACKING_DOMAIN,
        f"layout_identity={facts.layout_identity}",
        f"sequence_length={facts.sequence_length}",
    ]
    lines.extend(
        f"segment[{index}]={start}:{stop}:{kind}"
        for index, (start, stop, kind) in enumerate(facts.segments)
    )
    return _digest_lines(tuple(lines))


def exchange_backend_identity(mesh: UspMesh) -> str:
    if type(mesh) is not UspMesh:
        raise TypeError("mesh must be an exact UspMesh")
    if mesh.ulysses > 1 and mesh.ring > 1:
        return USP_EXCHANGE_HYBRID
    if mesh.ring > 1:
        return USP_EXCHANGE_RING
    return USP_EXCHANGE_ULYSSES


def ring_peer_schedules(mesh: UspMesh) -> tuple[tuple[int, ...], ...]:
    """Derive each rank's ascending-ring-step peer schedule analytically."""
    if type(mesh) is not UspMesh:
        raise TypeError("mesh must be an exact UspMesh")
    schedules: list[tuple[int, ...]] = []
    for rank in mesh.ranks:
        coordinate = mesh.coordinates(rank)
        ring_ranks = tuple(
            mesh.rank_of(type(coordinate)(coordinate.guidance, coordinate.ulysses, ring))
            for ring in range(mesh.ring)
        )
        schedules.append(
            tuple(ring_ranks[(coordinate.ring - step) % mesh.ring] for step in range(mesh.ring))
        )
    return tuple(schedules)


@dataclass(frozen=True, slots=True)
class CompiledSequenceParallelPlan:
    """One immutable, globally complete USP execution plan."""

    mesh: UspMesh
    placement: PlacementMap
    partition: SequencePartition
    layouts: tuple[SequenceLayout, ...]
    rank_device_bindings: tuple[str, ...]
    packing_geometry_digest: str
    ring_peer_schedules: tuple[tuple[int, ...], ...]
    routed_attention_backend_identity: str
    exchange_backend_identity: str
    attention_kernel_contract: str
    partition_compatibility: PartitionCompatibility
    compute_dtype: DType
    device_kind: str
    compile_authority: InitVar[object]

    def __post_init__(self, compile_authority: object) -> None:
        if compile_authority is not _COMPILE_AUTHORITY:
            raise ValueError("sequence-parallel plans are created only by the compiler")

    @property
    def facts(self) -> tuple[str, ...]:
        mesh = self.mesh.process_mesh
        placement = ",".join(
            f"{logical}:{physical}"
            for logical, physical in enumerate(self.placement.physical_ranks)
        )
        devices = ",".join(
            f"{rank}:{device}" for rank, device in enumerate(self.rank_device_bindings)
        )
        shards = ",".join(
            f"{shard.index}:{shard.start}-{shard.stop}+{shard.padded_rows}"
            for shard in self.partition.shards
        )
        ropes = ",".join(
            f"{rank}:{layout.shard.start}-{layout.shard.stop}+{layout.shard.padded_rows}"
            for rank, layout in enumerate(self.layouts)
        )
        schedules = ",".join(
            f"{rank}:{'-'.join(map(str, peers))}"
            for rank, peers in enumerate(self.ring_peer_schedules)
        )
        compatibility = asdict(self.partition_compatibility)
        compatibility["modes"] = sorted(
            type(mode).__name__ for mode in self.partition_compatibility.modes
        )
        compatibility["tensor_expectation"]["kind"] = type(
            self.partition_compatibility.tensor_expectation
        ).__name__
        return (
            "mesh.layout=guidance-tp-sp_ulysses-sp_ring.v1",
            (
                f"mesh.axes=guidance:{mesh.guidance},tp:{mesh.tp},"
                f"sp_ulysses:{mesh.sp_ulysses},sp_ring:{mesh.sp_ring}"
            ),
            f"mesh.world_size={mesh.world_size}",
            f"mesh.subgroup_digest={mesh.subgroup_digest}",
            f"placement.logical_to_physical={placement}",
            f"placement.rank_to_device={devices}",
            f"partition.global_sequence_length={self.partition.sequence_length}",
            f"partition.attention_head_count={self.layouts[0].head_count}",
            f"partition.chunk_size={self.partition.chunk}",
            f"partition.shards={shards}",
            f"partition.packing_geometry_digest={self.packing_geometry_digest}",
            f"partition.rope_slicing={ropes}",
            f"collective.traversal_contract={RING_TRAVERSAL_ORDER}",
            f"collective.ring_peer_schedules={schedules}",
            f"collective.accumulation_order={RING_ACCUMULATION_ORDER}",
            f"collective.merge_algorithm={RING_MERGE_ALGORITHM}",
            f"collective.accumulation_dtype={RING_ACCUMULATION_DTYPE}",
            f"provider.routed_attention_backend={self.routed_attention_backend_identity}",
            f"provider.exchange_backend={self.exchange_backend_identity}",
            f"provider.attention_kernel_contract={self.attention_kernel_contract}",
            "provider.partition_compatibility="
            + json.dumps(compatibility, sort_keys=True, separators=(",", ":")),
            "provider.compute_dtype="
            + json.dumps(asdict(self.compute_dtype), sort_keys=True, separators=(",", ":")),
            f"provider.device_kind={self.device_kind}",
        )

    @property
    def digest(self) -> str:
        return _digest_lines((_PLAN_DOMAIN, *self.facts))

    @property
    def rank_plan_digests(self) -> tuple[str, ...]:
        return tuple(
            _digest_lines(
                (
                    _RANK_PLAN_DOMAIN,
                    f"compiled_plan={self.digest}",
                    f"rank={rank}",
                    f"logical_rank={self.placement.logical_rank(rank)}",
                    f"device={self.rank_device_bindings[rank]}",
                    f"layout={layout.shard.start}-{layout.shard.stop}+{layout.shard.padded_rows}",
                    f"rope={layout.rope_position_offset}:{layout.shard.stop}+{layout.shard.padded_rows}",
                    "ring_peers="
                    + ",".join(
                        map(
                            str,
                            self.ring_peer_schedules[self.placement.logical_rank(rank)],
                        )
                    ),
                )
            )
            for rank, layout in enumerate(self.layouts)
        )


def compile_sequence_parallel_plan(
    *,
    mesh: UspMesh,
    placement: PlacementMap,
    partition: SequencePartition,
    packed_sequence_facts: PackedSequenceFacts,
    head_count: int,
    rank_device_bindings: tuple[str, ...],
    routed_attention_backend_identity: str,
    exchange_backend: str,
    attention_kernel: AttentionKernel | AttentionBlockKernel,
    partition_compatibility: PartitionCompatibility,
    compute_dtype: DType,
    device_kind: str,
) -> CompiledSequenceParallelPlan:
    """Check provider capability and compile rank-4 [B,H,S,D] facts before consensus.

    Block plans describe feasibility, not live Ring executor qualification.
    Protocol conformance checks the provider's interface, not its arithmetic.
    """
    if type(mesh) is not UspMesh:
        raise TypeError("mesh must be an exact UspMesh")
    if type(placement) is not PlacementMap or placement.mesh_digest != mesh.digest:
        raise ValueError("placement must exactly match the USP mesh")
    if type(partition) is not SequencePartition:
        raise TypeError("partition must be an exact SequencePartition")
    if partition.shard_count != mesh.ulysses * mesh.ring:
        raise ValueError("partition shard count must equal the sequence mesh degree")
    if not isinstance(cast("object", packed_sequence_facts), PackedSequenceFacts):
        raise TypeError("packed_sequence_facts must be PackedSequenceFacts")
    if partition.sequence_length != packed_sequence_facts.sequence_length:
        raise ValueError("partition length must match the packed sequence facts")
    if type(head_count) is not int or head_count < 1 or head_count % mesh.ulysses:
        raise ValueError("head_count must be an exact positive multiple of Ulysses degree")
    if type(rank_device_bindings) is not tuple or len(rank_device_bindings) != mesh.world_size:
        raise ValueError("rank_device_bindings must cover every physical rank")
    for device in rank_device_bindings:
        _require_identity("rank device binding", device)
    _require_identity("routed attention backend identity", routed_attention_backend_identity)
    _require_identity("exchange backend identity", exchange_backend)
    if exchange_backend != exchange_backend_identity(mesh):
        raise ValueError("exchange backend identity does not match the mesh")
    _require_identity("device kind", device_kind)
    checked_partition = require_attention_partition_feasibility(
        head_count,
        packed_sequence_facts.sequence_length,
        2,
        mesh.process_mesh,
        partition_compatibility,
        compute_dtype,
        device_kind,
    )
    if checked_partition != partition:
        raise ValueError("partition must match the feasible canonical partition")
    has_block_kernel = isinstance(attention_kernel, AttentionBlockKernel)
    if partition_compatibility.provides_matching_block_normalization and not has_block_kernel:
        raise ValueError("matching block normalization requires AttentionBlockKernel")
    if mesh.ring > 1:
        if not has_block_kernel:
            raise ValueError("Ring compilation requires AttentionBlockKernel")
        kernel_contract = "dinkster.attention-kernel.block.v1"
    else:
        if not isinstance(attention_kernel, AttentionKernel):
            raise ValueError("Ulysses compilation requires AttentionKernel")
        kernel_contract = "dinkster.attention-kernel.tensor.v1"

    heads_per_rank = head_count // mesh.ulysses
    layouts: list[SequenceLayout | None] = [None] * mesh.world_size
    for logical_rank in mesh.ranks:
        coordinate = mesh.coordinates(logical_rank)
        shard = partition.shards[coordinate.ulysses * mesh.ring + coordinate.ring]
        physical_rank = placement.physical_rank(logical_rank)
        layouts[physical_rank] = SequenceLayout(
            partition.sequence_length,
            shard,
            packed_sequence_facts.layout_identity,
            head_count,
            coordinate.ulysses * heads_per_rank,
            (coordinate.ulysses + 1) * heads_per_rank,
            shard.start,
            mesh.digest,
        )
    if any(layout is None for layout in layouts):
        raise ValueError("compiled layouts must cover every physical rank")
    return CompiledSequenceParallelPlan(
        mesh,
        placement,
        partition,
        tuple(layout for layout in layouts if layout is not None),
        rank_device_bindings,
        packing_geometry_digest(packed_sequence_facts),
        ring_peer_schedules(mesh),
        routed_attention_backend_identity,
        exchange_backend,
        kernel_contract,
        partition_compatibility,
        compute_dtype,
        device_kind,
        _COMPILE_AUTHORITY,
    )


def build_usp_compiled_plan_slot(plan: CompiledSequenceParallelPlan) -> CompiledPlanSlot:
    if type(plan) is not CompiledSequenceParallelPlan:
        raise TypeError("plan must be an exact CompiledSequenceParallelPlan")
    return CompiledPlanSlot(slot=USP_SLOT_NAME, facts=plan.facts)


__all__ = [
    "CompiledSequenceParallelPlan",
    "USP_EXCHANGE_HYBRID",
    "USP_EXCHANGE_RING",
    "USP_EXCHANGE_ULYSSES",
    "USP_SLOT_NAME",
    "build_usp_compiled_plan_slot",
    "compile_sequence_parallel_plan",
    "exchange_backend_identity",
    "packing_geometry_digest",
    "ring_peer_schedules",
]
