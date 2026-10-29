from __future__ import annotations

import os
import tempfile
from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dinkster_inference import (
    MINIMAX_H3_CONFIG,
    PlacementMap,
    SequenceLayout,
    SequenceShard,
    UspMesh,
    plan_sequence_partition,
)
from dinkster_inference_torch.attention import builtin_sdpa_kernel
from dinkster_inference_torch.sequence_parallel_attention import (
    SequenceParallelAttentionError,
    SequenceParallelAttentionKernel,
)
from gpu_test_gate import require_gpu_tests_enabled
from manifest_token import minted_consensus_token
from torch.multiprocessing.spawn import spawn

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)

_EQUIVALENCE_BOUND = 2.0e-6


@dataclass(frozen=True, slots=True)
class _Case:
    name: str
    guidance: int
    ulysses: int
    ring: int
    sequence_length: int = 12

    @property
    def world_size(self) -> int:
        return self.guidance * self.ulysses * self.ring


def _local_tensor(tensor: torch.Tensor, start: int, stop: int, padding: int) -> torch.Tensor:
    value = tensor[:, :, start:stop]
    if padding:
        value = torch.nn.functional.pad(value, (0, 0, 0, padding))
    return value.contiguous()


def _distributed_kernel(
    mesh: UspMesh,
    layout: SequenceLayout,
    placement: PlacementMap | None = None,
) -> SequenceParallelAttentionKernel:
    return SequenceParallelAttentionKernel(
        mesh,
        builtin_sdpa_kernel(),
        layout,
        placement,
        minted_consensus_token(mesh.ulysses * mesh.ring, layout.shard.index),
        "test-attention-backend",
        "test-exchange-backend",
    )


def _gather_output(
    output: torch.Tensor,
    mesh: UspMesh,
    sequence_length: int,
) -> torch.Tensor:
    rank = dist.get_rank()
    selected_ranks: tuple[int, ...] | None = None
    selected_group: dist.ProcessGroup | None = None
    for logical_ranks in mesh.sequence_groups() or tuple((rank,) for rank in mesh.ranks):
        group = None
        if len(logical_ranks) > 1:
            created = dist.new_group(ranks=sorted(logical_ranks))
            if rank in logical_ranks:
                assert isinstance(created, dist.ProcessGroup)
                group = created
        if rank in logical_ranks:
            selected_ranks = logical_ranks
            selected_group = group
    assert selected_ranks is not None
    if selected_group is None:
        gathered = [output]
        process_ranks = selected_ranks
    else:
        process_ranks = tuple(sorted(selected_ranks))
        gathered = [torch.empty_like(output) for _ in process_ranks]
        dist.all_gather(gathered, output, group=selected_group)
    by_rank = dict(zip(process_ranks, gathered, strict=True))
    partition = plan_sequence_partition(sequence_length, mesh.ulysses * mesh.ring)
    return torch.cat(
        tuple(
            by_rank[rank][:, :, : shard.valid_rows]
            for rank, shard in zip(selected_ranks, partition.shards, strict=True)
        ),
        dim=2,
    )


def _run_case(rank: int, case: _Case, rendezvous: str, queue: Any) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=case.world_size,
    )
    try:
        torch.set_num_threads(1)
        mesh = UspMesh.build(
            guidance=case.guidance,
            ulysses=case.ulysses,
            ring=case.ring,
        )
        coordinate = mesh.coordinates(rank)
        partition = plan_sequence_partition(case.sequence_length, case.ulysses * case.ring)
        shard_index = coordinate.ulysses * case.ring + coordinate.ring
        shard = partition.shards[shard_index]
        generator = torch.Generator().manual_seed(8100 + coordinate.guidance)
        shape = (
            1,
            8,
            case.sequence_length,
            8,
        )
        q, k, v = tuple(torch.randn(shape, generator=generator) for _ in range(3))
        local = tuple(
            _local_tensor(tensor, shard.start, shard.stop, shard.padded_rows)
            for tensor in (q, k, v)
        )
        heads_per_rank = q.shape[1] // case.ulysses
        layout = SequenceLayout(
            case.sequence_length,
            shard,
            f"fixed-seed-token-order:{8100 + coordinate.guidance}",
            q.shape[1],
            coordinate.ulysses * heads_per_rank,
            (coordinate.ulysses + 1) * heads_per_rank,
            shard.start,
            mesh.digest,
        )
        kernel = (
            SequenceParallelAttentionKernel(mesh, builtin_sdpa_kernel())
            if case.ulysses == case.ring == 1
            else _distributed_kernel(mesh, layout)
        )
        output, completions = kernel.attend_with_completions(*local)
        if case.ulysses == case.ring == 1:
            assert completions == ()
        else:
            assert tuple(event.slot for event in completions) == ("q", "k", "v")
        gathered = _gather_output(output, mesh, case.sequence_length)
        reference = builtin_sdpa_kernel()(q, k, v)
        delta = float((gathered - reference).abs().max().item())
        if case.ring == 1:
            assert torch.equal(gathered, reference)
        sequence_groups = mesh.sequence_groups() or tuple(
            (logical_rank,) for logical_rank in mesh.ranks
        )
        sequence_group = next(group for group in sequence_groups if rank in group)
        if rank == sequence_group[0]:
            queue.put((coordinate.guidance, delta))
    finally:
        dist.destroy_process_group()


def _execute(case: _Case) -> list[tuple[int, float]]:
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "rendezvous")
        spawn(
            _run_case,
            args=(case, rendezvous, queue),
            nprocs=case.world_size,
            join=True,
        )
    return [queue.get() for _ in range(case.guidance)]


def _run_cuda_case(rank: int, case: _Case, rendezvous: str, queue: Any) -> None:
    require_gpu_tests_enabled()
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=case.world_size,
    )
    try:
        mesh = UspMesh.build(
            guidance=case.guidance,
            ulysses=case.ulysses,
            ring=case.ring,
        )
        coordinate = mesh.coordinates(rank)
        partition = plan_sequence_partition(case.sequence_length, case.ulysses * case.ring)
        shard_index = coordinate.ulysses * case.ring + coordinate.ring
        shard = partition.shards[shard_index]
        generator = torch.Generator().manual_seed(9100 + coordinate.guidance)
        shape = (
            1,
            MINIMAX_H3_CONFIG.attention_heads,
            case.sequence_length,
            MINIMAX_H3_CONFIG.attention_head_dim,
        )
        q, k, v = tuple(
            torch.randn(shape, generator=generator).to(device=f"cuda:{rank}") for _ in range(3)
        )
        local = tuple(
            _local_tensor(tensor, shard.start, shard.stop, shard.padded_rows)
            for tensor in (q, k, v)
        )
        heads_per_rank = q.shape[1] // case.ulysses
        layout = SequenceLayout(
            case.sequence_length,
            shard,
            f"h3-gpu-seed:{9100 + coordinate.guidance}",
            q.shape[1],
            coordinate.ulysses * heads_per_rank,
            (coordinate.ulysses + 1) * heads_per_rank,
            shard.start,
            mesh.digest,
        )
        kernel = (
            SequenceParallelAttentionKernel(mesh, builtin_sdpa_kernel())
            if case.ulysses == case.ring == 1
            else _distributed_kernel(mesh, layout)
        )
        output = kernel(*local)
        gathered = _gather_output(output, mesh, case.sequence_length)
        reference = builtin_sdpa_kernel()(q, k, v)
        delta = float((gathered - reference).abs().max().item())
        sequence_groups = mesh.sequence_groups() or tuple(
            (logical_rank,) for logical_rank in mesh.ranks
        )
        sequence_group = next(group for group in sequence_groups if rank in group)
        if rank == sequence_group[0]:
            queue.put((coordinate.guidance, delta))
    finally:
        dist.destroy_process_group()


def _execute_cuda(case: _Case) -> list[tuple[int, float]]:
    require_gpu_tests_enabled()
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "cuda-rendezvous")
        spawn(
            _run_cuda_case,
            args=(case, rendezvous, queue),
            nprocs=case.world_size,
            join=True,
        )
    return [queue.get() for _ in range(case.guidance)]


@pytest.mark.parametrize(
    "case",
    (
        _Case("U1R1", 1, 1, 1),
        _Case("U2R1", 1, 2, 1),
        _Case("U1R2", 1, 1, 2),
        _Case("U4R1", 1, 4, 1),
        _Case("U2R2", 1, 2, 2),
        _Case("U1R4", 1, 1, 4),
        _Case("CFG2xU2R1", 2, 2, 1),
    ),
    ids=lambda case: case.name,
)
def test_sequence_parallel_attention_matches_single_rank_reference(case: _Case) -> None:
    results = _execute(case)

    assert len(results) == case.guidance
    assert all(delta <= _EQUIVALENCE_BOUND for _, delta in results), results


def test_degenerate_kernel_is_byte_identical_and_passes_attention_keywords() -> None:
    mesh = UspMesh.build(guidance=1, ulysses=1, ring=1)
    kernel = SequenceParallelAttentionKernel(mesh, builtin_sdpa_kernel())
    generator = torch.Generator().manual_seed(44)
    q, k, v = tuple(torch.randn(1, 4, 5, 8, generator=generator) for _ in range(3))
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    mask[..., -1] = False

    expected_mask = builtin_sdpa_kernel()(q, k, v, mask=mask, scale=0.25)
    actual_mask = kernel(q, k, v, mask=mask, scale=0.25)
    expected_causal = builtin_sdpa_kernel()(q, k, v, causal=True)
    actual_causal = kernel(q, k, v, causal=True)
    gqa_k = k[:, :2]
    gqa_v = v[:, :2]
    expected_gqa = builtin_sdpa_kernel()(q, gqa_k, gqa_v, enable_gqa=True)
    actual_gqa = kernel(q, gqa_k, gqa_v, enable_gqa=True)

    assert torch.equal(actual_mask, expected_mask)
    assert torch.equal(actual_causal, expected_causal)
    assert torch.equal(actual_gqa, expected_gqa)


def test_kernel_records_are_frozen_and_validate_construction() -> None:
    mesh = UspMesh.build(guidance=1, ulysses=1, ring=1)
    kernel = SequenceParallelAttentionKernel(mesh, builtin_sdpa_kernel())
    with pytest.raises(FrozenInstanceError):
        kernel.layout = None  # type: ignore[misc]
    with pytest.raises(SequenceParallelAttentionError, match="does not accept"):
        partition = plan_sequence_partition(4, 1)
        SequenceParallelAttentionKernel(
            mesh,
            builtin_sdpa_kernel(),
            SequenceLayout(4, partition.shards[0], "order", 4, 0, 4, 0, mesh.digest),
        )


def test_kernel_validates_placement_against_the_mesh() -> None:
    mesh = UspMesh(1, 1, 2)
    partition = plan_sequence_partition(4, 2)
    layout = SequenceLayout(4, partition.shards[0], "order", 4, 0, 4, 0, mesh.digest)

    with pytest.raises(SequenceParallelAttentionError, match="digest"):
        SequenceParallelAttentionKernel(
            mesh,
            builtin_sdpa_kernel(),
            layout,
            PlacementMap(UspMesh(1, 2, 1).process_mesh, (0, 1)),
        )


def test_multi_rank_construction_requires_canonical_partition_and_head_ownership() -> None:
    infeasible_mesh = UspMesh.build(guidance=1, ulysses=1, ring=3)
    infeasible_layout = SequenceLayout(
        4, SequenceShard(0, 0, 2, 0), "order", 4, 0, 4, 0, infeasible_mesh.digest
    )
    with pytest.raises(SequenceParallelAttentionError, match="cannot use"):
        SequenceParallelAttentionKernel(infeasible_mesh, builtin_sdpa_kernel(), infeasible_layout)

    ulysses_mesh = UspMesh.build(guidance=1, ulysses=2, ring=1)
    partition = plan_sequence_partition(4, 2)
    bad_heads = SequenceLayout(4, partition.shards[0], "order", 4, 0, 1, 0, ulysses_mesh.digest)
    with pytest.raises(SequenceParallelAttentionError, match="head ownership"):
        SequenceParallelAttentionKernel(ulysses_mesh, builtin_sdpa_kernel(), bad_heads)

    ring_mesh = UspMesh.build(guidance=1, ulysses=1, ring=2)
    noncanonical = SequenceLayout(
        4, SequenceShard(0, 0, 1, 0), "order", 4, 0, 4, 0, ring_mesh.digest
    )
    with pytest.raises(SequenceParallelAttentionError, match="canonical"):
        SequenceParallelAttentionKernel(ring_mesh, builtin_sdpa_kernel(), noncanonical)


def test_multi_rank_refusals_fire_without_collective_initialization() -> None:
    ring_mesh = UspMesh.build(guidance=1, ulysses=1, ring=2)
    partition = plan_sequence_partition(4, 2)
    ring_layout = SequenceLayout(4, partition.shards[0], "order", 4, 0, 4, 0, ring_mesh.digest)
    ring = _distributed_kernel(ring_mesh, ring_layout)
    value = torch.zeros(1, 4, 2, 8, dtype=torch.bfloat16)
    with pytest.raises(SequenceParallelAttentionError, match="requires float32"):
        ring(value, value, value)
    float_value = value.float()
    with pytest.raises(SequenceParallelAttentionError, match="mask or causal"):
        ring(float_value, float_value, float_value, causal=True)
    with pytest.raises(SequenceParallelAttentionError, match="mask or causal"):
        ring(
            float_value,
            float_value,
            float_value,
            mask=torch.ones(1, 1, 2, 2, dtype=torch.bool),
        )
    with pytest.raises(SequenceParallelAttentionError, match="grouped-query"):
        ring(float_value, float_value, float_value, enable_gqa=True)
    with pytest.raises(SequenceParallelAttentionError, match="causal must"):
        ring(float_value, float_value, float_value, causal=0)  # type: ignore[arg-type]
    with pytest.raises(SequenceParallelAttentionError, match="enable_gqa must"):
        ring(float_value, float_value, float_value, enable_gqa=0)  # type: ignore[arg-type]
    with pytest.raises(SequenceParallelAttentionError, match="scale must"):
        ring(float_value, float_value, float_value, scale="bad")  # type: ignore[arg-type]

    ulysses_mesh = UspMesh.build(guidance=1, ulysses=2, ring=1)
    bad_layout = SequenceLayout(4, partition.shards[0], "order", 3, 0, 1, 0, ulysses_mesh.digest)
    with pytest.raises(SequenceParallelAttentionError, match="divisible"):
        SequenceParallelAttentionKernel(ulysses_mesh, builtin_sdpa_kernel(), bad_layout)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4 or not dist.is_nccl_available(),
    reason="four CUDA GPUs with NCCL are required",
)
@pytest.mark.parametrize(
    "case",
    (
        _Case("U1R1", 1, 1, 1, 64),
        _Case("U2R1", 1, 2, 1, 64),
        _Case("U1R2", 1, 1, 2, 64),
        _Case("U4R1", 1, 4, 1, 64),
        _Case("U2R2", 1, 2, 2, 64),
        _Case("U1R4", 1, 1, 4, 64),
        _Case("CFG2xU2R1", 2, 2, 1, 64),
    ),
    ids=lambda case: f"gpu-{case.name}",
)
def test_cuda_h3_geometry_parity(case: _Case) -> None:
    results = _execute_cuda(case)

    print(f"{case.name} deltas={results}")
    assert len(results) == case.guidance
    assert all(delta <= _EQUIVALENCE_BOUND for _, delta in results), results
