from __future__ import annotations

import os
import sys
import tempfile
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dinkster_inference import PlacementMap, SequenceLayout, UspMesh, plan_sequence_partition
from dinkster_inference_torch import sequence_exchange as sequence_exchange_module
from dinkster_inference_torch.attention import (
    AttentionBlockKernel,
    AttentionBlockResult,
    AttentionKernel,
    builtin_sdpa_kernel,
)
from dinkster_inference_torch.sequence_exchange import (
    RING_ACCUMULATION_DTYPE,
    RING_ACCUMULATION_ORDER,
    RING_MERGE_ALGORITHM,
    RING_TRAVERSAL_ORDER,
    DenseRingBlockAttention,
    RingAttentionPlan,
    RingSequenceExchange,
    SequenceExchangeCompletionEvent,
    SequenceExchangeError,
    SequenceExchangeEvent,
    SequenceExchangeHooks,
    SequenceExchangeOrderEvent,
    SequenceExchangeResult,
    SequenceExchangeSubmission,
    SequenceExchangeSubmissions,
    UlyssesSequenceExchange,
    UspSequenceExchange,
    merge_attention_blocks,
)
from dinkster_inference_torch.sequence_parallel_attention import (
    SequenceParallelAttentionKernel,
    gather_sequence_hidden,
)
from manifest_token import minted_consensus_token
from torch.multiprocessing.spawn import ProcessRaisedException, spawn

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)

_INSTRUMENTED_EQUIVALENCE_BOUND = 2.0e-6


@dataclass(frozen=True, slots=True)
class _Case:
    name: str
    guidance: int
    ulysses: int
    ring: int
    sequence_length: int = 12
    placement_order: tuple[int, ...] | None = None
    chunked_inputs: bool = False
    dtype: torch.dtype = torch.float32
    report_all_sequence_ranks: bool = False

    @property
    def world_size(self) -> int:
        return self.guidance * self.ulysses * self.ring


class _CountingKernel:
    def __init__(self, kernel: AttentionKernel) -> None:
        self.kernel = kernel
        self.calls = 0

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
        self.calls += 1
        return self.kernel(
            q,
            k,
            v,
            mask=mask,
            causal=causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )


async def _async_exchange_hook(_event: SequenceExchangeEvent) -> None:
    return None


class _AsyncCallableHook:
    async def __call__(self, _event: SequenceExchangeEvent) -> None:
        return None


class _RecordingReadyEvent:
    def __init__(self, ordinal: int, waits: list[int]) -> None:
        self.ordinal = ordinal
        self.waits = waits

    def wait(self) -> None:
        self.waits.append(self.ordinal)


def _mesh(case: _Case) -> UspMesh:
    return UspMesh.build(
        guidance=case.guidance,
        ulysses=case.ulysses,
        ring=case.ring,
    )


def _placement(case: _Case, mesh: UspMesh) -> PlacementMap:
    if case.placement_order is None:
        return PlacementMap.identity(mesh.process_mesh)
    return PlacementMap(mesh.process_mesh, case.placement_order)


def _make_inputs(seed: int, sequence_length: int, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (1, 8, sequence_length, 8)
    return tuple(torch.randn(shape, generator=generator, dtype=dtype) for _ in range(3))


def _local_tensor(tensor: torch.Tensor, start: int, stop: int, padded_rows: int) -> torch.Tensor:
    local = tensor[:, :, start:stop]
    if padded_rows:
        local = torch.nn.functional.pad(local, (0, 0, 0, padded_rows))
    return local.contiguous()


def _chunked(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    split = max(1, tensor.shape[2] // 2)
    return tensor[:, :, :split], tensor[:, :, split:]


def _sequence_group(mesh: UspMesh, rank: int) -> tuple[int, ...]:
    groups = mesh.sequence_groups() or tuple((logical_rank,) for logical_rank in mesh.ranks)
    return next(group for group in groups if rank in group)


def _create_sequence_group(
    mesh: UspMesh, placement: PlacementMap
) -> tuple[tuple[int, ...], tuple[int, ...], dist.ProcessGroup | None]:
    logical_rank = placement.logical_rank(dist.get_rank())
    selected_ranks: tuple[int, ...] | None = None
    selected_physical_ranks: tuple[int, ...] | None = None
    selected_group: dist.ProcessGroup | None = None
    for logical_ranks in mesh.sequence_groups() or tuple((rank,) for rank in mesh.ranks):
        physical_ranks = placement.apply(logical_ranks)
        process_group = None
        if len(logical_ranks) > 1:
            created = dist.new_group(ranks=list(physical_ranks))
            if logical_rank in logical_ranks:
                assert isinstance(created, dist.ProcessGroup)
                process_group = created
        if logical_rank in logical_ranks:
            selected_ranks = logical_ranks
            selected_physical_ranks = physical_ranks
            selected_group = process_group
    assert selected_ranks is not None
    assert selected_physical_ranks is not None
    return selected_ranks, selected_physical_ranks, selected_group


def _gather_valid_output(
    output: torch.Tensor,
    mesh: UspMesh,
    placement: PlacementMap,
    partition_length: int,
) -> torch.Tensor:
    logical_ranks, physical_ranks, group = _create_sequence_group(mesh, placement)
    if len(logical_ranks) == 1:
        gathered = [output]
        process_ranks = physical_ranks
    else:
        assert group is not None
        process_ranks = tuple(sorted(physical_ranks))
        gathered = [torch.empty_like(output) for _ in process_ranks]
        dist.all_gather(gathered, output, group=group)
    by_rank = dict(zip(process_ranks, gathered, strict=True))
    partition = plan_sequence_partition(partition_length, mesh.ulysses * mesh.ring)
    valid: list[torch.Tensor] = []
    for shard_index, rank in enumerate(logical_ranks):
        shard = partition.shards[shard_index]
        valid.append(by_rank[placement.physical_rank(rank)][:, :, : shard.valid_rows])
    return torch.cat(valid, dim=2)


def _backend(case: _Case, mesh: UspMesh, placement: PlacementMap, sequence_rank: int) -> Any:
    token = minted_consensus_token(mesh.ulysses * mesh.ring, sequence_rank)
    if case.ulysses > 1 and case.ring == 1:
        return UlyssesSequenceExchange(mesh, placement, consensus_token=token)
    if case.ulysses == 1 and case.ring > 1:
        return RingSequenceExchange(mesh, placement, consensus_token=token)
    return UspSequenceExchange(mesh, placement, consensus_token=token)


def _run_case(
    rank: int,
    case: _Case,
    rendezvous: str,
    result_queue: Any,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=case.world_size,
    )
    try:
        torch.set_num_threads(1)
        mesh = _mesh(case)
        placement = _placement(case, mesh)
        logical_rank = placement.logical_rank(rank)
        coordinate = mesh.coordinates(logical_rank)
        partition = plan_sequence_partition(case.sequence_length, case.ulysses * case.ring)
        shard_index = coordinate.ulysses * case.ring + coordinate.ring
        shard = partition.shards[shard_index]
        q, k, v = _make_inputs(1701 + coordinate.guidance, case.sequence_length, case.dtype)
        local = (
            _local_tensor(q, shard.start, shard.stop, shard.padded_rows),
            _local_tensor(k, shard.start, shard.stop, shard.padded_rows),
            _local_tensor(v, shard.start, shard.stop, shard.padded_rows),
        )
        heads_per_rank = q.shape[1] // case.ulysses
        layout = SequenceLayout(
            case.sequence_length,
            shard,
            f"fixed-seed-token-order:{1701 + coordinate.guidance}",
            q.shape[1],
            coordinate.ulysses * heads_per_rank,
            (coordinate.ulysses + 1) * heads_per_rank,
            shard.start,
            mesh.digest,
        )
        backend = _backend(case, mesh, placement, shard.index)
        before: list[SequenceExchangeEvent] = []
        after: list[SequenceExchangeEvent] = []
        hooks = SequenceExchangeHooks(before.append, after.append)
        counted_kernel = _CountingKernel(builtin_sdpa_kernel())
        inputs: tuple[Any, Any, Any]
        if case.chunked_inputs:
            inputs = _chunked(local[0]), _chunked(local[1]), _chunked(local[2])
        else:
            inputs = local
        submissions = SequenceExchangeSubmissions.from_tensors(*inputs)
        result = backend.attend_submissions(submissions, layout, counted_kernel, hooks=hooks)
        output = result.output
        assert tuple((event.slot, event.chunk) for event in result.completions) == tuple(
            (item.slot, item.chunk) for item in submissions.items
        )
        assert all(event.completed.ordinal > event.ready.ordinal for event in result.completions)
        repeated = backend.attend(*inputs, layout, counted_kernel)
        assert torch.equal(output, repeated)
        assert before == after
        assert before
        assert counted_kernel.calls >= 2
        if shard.padded_rows:
            assert torch.count_nonzero(output[:, :, shard.valid_rows :]).item() == 0

        gathered = _gather_valid_output(output, mesh, placement, case.sequence_length)
        reference = builtin_sdpa_kernel()(q, k, v)
        delta = float((gathered - reference).abs().max().item())
        if case.ring == 1:
            assert torch.equal(gathered, reference)
        # Instrumented equivalence bound: float32 online-softmax decomposition
        # measured <= 3.278256e-7 on the fixed CPU/Gloo matrix below; 2e-6
        # provides 6.10x headroom without admitting a different algorithm.
        assert delta <= _INSTRUMENTED_EQUIVALENCE_BOUND

        if case.world_size > 1:
            sequence_kernel = SequenceParallelAttentionKernel(
                mesh,
                builtin_sdpa_kernel(),
                layout,
                placement,
                minted_consensus_token(mesh.ulysses * mesh.ring, shard.index),
                "test-attention-backend",
                "test-exchange-backend",
            )
            sequence_kernel(*local)
            local_hidden = torch.full((1, partition.chunk, 2), -1.0)
            local_hidden[:, : shard.valid_rows] = torch.arange(
                shard.start, shard.stop, dtype=torch.float32
            ).view(1, -1, 1)
            gathered_hidden = gather_sequence_hidden(
                sequence_kernel, local_hidden, partition, shard
            )
            expected_hidden = torch.arange(case.sequence_length, dtype=torch.float32).view(1, -1, 1)
            assert torch.equal(gathered_hidden[..., :1], expected_hidden)
            assert torch.equal(gathered_hidden[..., 1:], expected_hidden)

        sequence_group = _sequence_group(mesh, logical_rank)
        if case.report_all_sequence_ranks or logical_rank == sequence_group[0]:
            plan_order: tuple[int, ...] | None = None
            if isinstance(backend, RingSequenceExchange):
                plan_order = backend.plan.traversal_order
            elif isinstance(backend, UspSequenceExchange):
                plan_order = backend.ring_plan.traversal_order
            ring_groups = mesh.ring_groups() or tuple((rank,) for rank in mesh.ranks)
            ring_group = next(group for group in ring_groups if logical_rank in group)
            result_queue.put(
                (
                    coordinate.guidance,
                    delta,
                    plan_order,
                    placement.apply(plan_order) if plan_order is not None else (),
                    placement.apply(ring_group),
                    tuple(event.operation for event in before),
                )
            )
    finally:
        dist.destroy_process_group()


def _run_hook_failure(
    rank: int,
    rendezvous: str,
    result_queue: Any,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=3),
    )
    try:
        mesh = UspMesh.build(guidance=1, ulysses=2, ring=1)
        partition = plan_sequence_partition(12, 2)
        shard = partition.shards[rank]
        q, k, v = _make_inputs(2901, 12, torch.float32)
        local = (
            _local_tensor(q, shard.start, shard.stop, shard.padded_rows),
            _local_tensor(k, shard.start, shard.stop, shard.padded_rows),
            _local_tensor(v, shard.start, shard.stop, shard.padded_rows),
        )
        layout = SequenceLayout(
            12,
            shard,
            "hook-failure-token-order",
            8,
            rank * 4,
            (rank + 1) * 4,
            shard.start,
            mesh.digest,
        )

        def before(_event: SequenceExchangeEvent) -> None:
            if rank == 0:
                raise ValueError("hook failed locally")

        try:
            UlyssesSequenceExchange(mesh, consensus_token=minted_consensus_token(2, rank)).attend(
                *local,
                layout,
                builtin_sdpa_kernel(),
                hooks=SequenceExchangeHooks(before=before),
            )
        except SequenceExchangeError as error:
            assert rank == 0
            result_queue.put((rank, "hook", str(error)))
            if sys.platform.startswith("win"):
                raise
        except RuntimeError as error:
            assert rank == 1
            result_queue.put((rank, "peer", str(error)))
        else:
            raise AssertionError("hook failure must terminate every rank")
    finally:
        dist.destroy_process_group()


def _execute(
    case: _Case,
) -> list[
    tuple[int, float, tuple[int, ...] | None, tuple[int, ...], tuple[int, ...], tuple[str, ...]]
]:
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
    result_count = case.guidance
    if case.report_all_sequence_ranks:
        result_count *= case.ulysses * case.ring
    return [queue.get() for _ in range(result_count)]


def _ring_plan(**changes: object) -> RingAttentionPlan:
    values: dict[str, object] = {
        "merge_algorithm": RING_MERGE_ALGORITHM,
        "traversal_order": (2, 0, 1),
        "traversal_contract": RING_TRAVERSAL_ORDER,
        "accumulation_order": RING_ACCUMULATION_ORDER,
        "accumulation_dtype": RING_ACCUMULATION_DTYPE,
    }
    values.update(changes)
    return RingAttentionPlan(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "case",
    (
        _Case("U1R1", 1, 1, 1, chunked_inputs=True),
        _Case("U2R1", 1, 2, 1),
        _Case("U1R2", 1, 1, 2),
        _Case("U4R1", 1, 4, 1),
        _Case("U2R2", 1, 2, 2),
        _Case("U1R4", 1, 1, 4),
    ),
    ids=lambda case: case.name,
)
def test_sequence_exchange_matches_single_rank_reference(case: _Case) -> None:
    results = _execute(case)

    assert len(results) == 1
    assert results[0][1] <= _INSTRUMENTED_EQUIVALENCE_BOUND


def test_guidance_subgroups_exchange_independently() -> None:
    results = sorted(_execute(_Case("G2U1R2", 2, 1, 2)))

    assert [guidance for guidance, *_ in results] == [0, 1]
    assert all(delta <= _INSTRUMENTED_EQUIVALENCE_BOUND for _, delta, *_ in results)


def test_ring_uses_nonidentity_placement_with_logical_traversal() -> None:
    results = _execute(
        _Case(
            "permuted-U1R3",
            1,
            1,
            3,
            placement_order=(2, 1, 0),
            report_all_sequence_ranks=True,
        )
    )
    orders = {plan[0]: plan for _, _, plan, _, _, _ in results if plan is not None}
    physical_orders = {
        plan[0]: physical for _, _, plan, physical, _, _ in results if plan is not None
    }
    transport_orders = {transport for _, _, _, _, transport, _ in results}

    assert transport_orders == {(2, 1, 0)}
    assert orders == {
        0: (0, 2, 1),
        1: (1, 0, 2),
        2: (2, 1, 0),
    }
    assert physical_orders == {
        0: (2, 0, 1),
        1: (1, 2, 0),
        2: (0, 1, 2),
    }


def test_ulysses_uses_nonidentity_placement() -> None:
    results = _execute(_Case("permuted-U4R1", 1, 4, 1, placement_order=(2, 0, 3, 1)))

    assert results[0][1] <= _INSTRUMENTED_EQUIVALENCE_BOUND


def test_hybrid_exchange_uses_nonidentity_placement() -> None:
    results = _execute(_Case("permuted-U2R2", 1, 2, 2, placement_order=(2, 0, 3, 1)))

    assert results[0][1] <= _INSTRUMENTED_EQUIVALENCE_BOUND


def test_ulysses_only_preserves_bfloat16_reference_bytes() -> None:
    results = _execute(_Case("U2R1-bfloat16", 1, 2, 1, dtype=torch.bfloat16))

    assert results[0][1] == 0.0


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("masked", (False, True))
@pytest.mark.parametrize("scale", (None, 0.25))
def test_dense_ring_block_statistics_match_full_score_reference(
    dtype: torch.dtype, masked: bool, scale: float | None
) -> None:
    generator = torch.Generator().manual_seed(185)
    q = torch.randn(2, 4, 129, 128, generator=generator).to(dtype)
    k = torch.randn(2, 4, 127, 128, generator=generator).to(dtype)
    mask = torch.arange(k.shape[-2]).view(1, 1, 1, -1) < 120 if masked else None
    score_scale = q.shape[-1] ** -0.5 if scale is None else scale
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * score_scale
    if mask is not None:
        scores.masked_fill_(~mask, -torch.inf)
    expected_maximum = scores.amax(dim=-1, keepdim=True)
    expected_sum = torch.exp(scores - expected_maximum).sum(dim=-1, keepdim=True)

    maximum, total = sequence_exchange_module.dense_ring_block_statistics(
        q, k, mask=mask, scale=scale
    )

    assert maximum.dtype == total.dtype == torch.float32
    assert maximum.shape == total.shape == (2, 4, 129, 1)
    # Query tiling can change dot-product rounding; this is not byte-equality.
    torch.testing.assert_close(maximum, expected_maximum, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(total, expected_sum, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("masked", (False, True))
def test_dense_ring_block_statistics_ignore_ambient_autocast(
    dtype: torch.dtype, masked: bool
) -> None:
    generator = torch.Generator().manual_seed(185)
    q = torch.randn(1, 2, 65, 16, generator=generator).to(dtype)
    k = torch.randn(1, 2, 7, 16, generator=generator).to(dtype)
    mask = torch.arange(7).view(1, 1, 1, -1) < 5 if masked else None
    expected_maximum, expected_sum = sequence_exchange_module.dense_ring_block_statistics(
        q, k, mask=mask
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        maximum, total = sequence_exchange_module.dense_ring_block_statistics(q, k, mask=mask)
        assert torch.is_autocast_enabled("cpu")

    assert maximum.dtype == total.dtype == torch.float32
    assert torch.equal(maximum, expected_maximum)
    assert torch.equal(total, expected_sum)


def test_dense_ring_block_statistics_bound_score_tiles_without_autograd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = torch.randn(2, 4, 129, 16, requires_grad=True)
    k = torch.randn(2, 4, 127, 16, requires_grad=True)
    matmul = torch.matmul
    shapes: list[tuple[int, ...]] = []

    def record_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        result = matmul(left, right)
        shapes.append(tuple(result.shape))
        return result

    monkeypatch.setattr(torch, "matmul", record_matmul)
    maximum, total = sequence_exchange_module.dense_ring_block_statistics(q, k)

    assert shapes == [(2, 4, 64, 127), (2, 4, 64, 127), (2, 4, 1, 127)]
    assert not maximum.requires_grad and not total.requires_grad


def test_dense_ring_block_statistics_masked_keys_have_no_mass() -> None:
    generator = torch.Generator().manual_seed(185)
    q = torch.randn(1, 2, 65, 16, generator=generator)
    k = torch.randn(1, 2, 7, 16, generator=generator)
    mask = torch.tensor([True, False, False, False, False, False, False]).view(1, 1, 1, -1)
    k[:, :, 1:] = 1e10
    expected_maximum = torch.matmul(q, k[:, :, :1].transpose(-2, -1)) * q.shape[-1] ** -0.5

    maximum, total = sequence_exchange_module.dense_ring_block_statistics(q, k, mask=mask)
    torch.testing.assert_close(maximum, expected_maximum, rtol=2e-6, atol=2e-6)
    assert torch.equal(total, torch.ones_like(total))

    maximum, total = sequence_exchange_module.dense_ring_block_statistics(
        q, k, mask=torch.zeros_like(mask)
    )
    assert torch.isneginf(maximum).all()
    assert torch.equal(total, torch.zeros_like(total))


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("masked", (False, True))
@pytest.mark.parametrize("scale", (None, -0.25))
def test_dense_ring_block_matches_full_score_reference(
    dtype: torch.dtype, masked: bool, scale: float | None
) -> None:
    generator = torch.Generator().manual_seed(185)
    q = torch.randn(2, 4, 129, 16, generator=generator).to(dtype)
    k = torch.randn(2, 4, 127, 16, generator=generator).to(dtype)
    v = torch.randn(2, 4, 127, 12, generator=generator).to(dtype)
    mask = torch.arange(127).view(1, 1, 1, -1) < 120 if masked else None
    scores = q.float() @ k.float().transpose(-2, -1) * (0.25 if scale is None else scale)
    if mask is not None:
        scores.masked_fill_(~mask, -torch.inf)
    expected_output = scores.softmax(-1) @ v.float()
    expected_maximum = scores.amax(-1, keepdim=True)
    expected_sum = (scores - expected_maximum).exp().sum(-1, keepdim=True)

    kernel: AttentionBlockKernel = DenseRingBlockAttention()
    result = kernel.attention_block(q, k, v, mask=mask, scale=scale)

    assert result.output.shape == (2, 4, 129, 12)
    assert result.maximum.shape == result.exponential_sum.shape == (2, 4, 129, 1)
    assert (
        result.output.dtype == result.maximum.dtype == result.exponential_sum.dtype == torch.float32
    )
    torch.testing.assert_close(result.output, expected_output, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(result.maximum, expected_maximum, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(result.exponential_sum, expected_sum, rtol=2e-6, atol=2e-6)
    maximum, total = sequence_exchange_module.dense_ring_block_statistics(
        q, k, mask=mask, scale=scale
    )
    assert torch.equal(result.maximum, maximum)
    assert torch.equal(result.exponential_sum, total)


def test_dense_ring_block_capability_is_distinct_from_tensor_only_attention() -> None:
    assert isinstance(DenseRingBlockAttention(), AttentionBlockKernel)
    assert not isinstance(builtin_sdpa_kernel(), AttentionBlockKernel)
    assert not isinstance(DenseRingBlockAttention(), AttentionKernel)


def test_dense_ring_block_normalizes_before_multiplying_large_values() -> None:
    q = torch.zeros(1, 1, 65, 16)
    k = torch.zeros(1, 1, 7, 16)
    v = torch.full((1, 1, 7, 12), torch.finfo(torch.float32).max / 4)
    result = DenseRingBlockAttention().attention_block(q, k, v)
    assert torch.isfinite(result.output).all()
    torch.testing.assert_close(result.output, v[:, :, :1].expand(1, 1, 65, 12))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_dense_ring_block_large_nonoverflowing_negative_scores(dtype: torch.dtype) -> None:
    q = torch.full((1, 1, 65, 1), 1e18, dtype=dtype)
    k = torch.full((1, 1, 1, 1), -1e18, dtype=dtype)
    v = torch.full((1, 1, 1, 1), 1e30, dtype=dtype)
    kernel = DenseRingBlockAttention()
    result = kernel.attention_block(q, k, v)
    assert torch.isfinite(result.maximum).all()
    assert (result.maximum < 0).all()
    assert torch.equal(result.output, v.float().expand_as(result.output))
    assert torch.equal(result.exponential_sum, torch.ones_like(result.exponential_sum))

    empty = kernel.attention_block(q, k, v, mask=torch.zeros(1, dtype=torch.bool))
    assert torch.count_nonzero(empty.output) == 0
    assert torch.count_nonzero(empty.exponential_sum) == 0
    assert torch.isneginf(empty.maximum).all()


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_dense_ring_block_one_key_and_empty_heads(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(185)
    q = torch.randn(2, 2, 65, 16, generator=generator).to(dtype)
    k = torch.randn(2, 2, 7, 16, generator=generator).to(dtype)
    v = torch.randn(2, 2, 7, 12, generator=generator).to(dtype)
    k[:, :, 1:] = 1e10
    v[:, :, 1:] = 1e10
    mask = torch.zeros(2, 2, 1, 7, dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    result = DenseRingBlockAttention().attention_block(q, k, v, mask=mask)
    assert torch.equal(result.output[0, 0], v[0, 0, :1].float().expand(65, 12))
    assert torch.equal(result.exponential_sum[0, 0], torch.ones(65, 1))
    assert torch.count_nonzero(result.output[1]) == 0
    assert torch.count_nonzero(result.output[0, 1]) == 0
    assert torch.isneginf(result.maximum[1]).all()
    assert torch.isneginf(result.maximum[0, 1]).all()
    assert torch.count_nonzero(result.exponential_sum[1]) == 0
    assert torch.count_nonzero(result.exponential_sum[0, 1]) == 0


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("masked", (False, True))
def test_dense_ring_block_preserves_float32_under_autocast(
    dtype: torch.dtype, masked: bool
) -> None:
    generator = torch.Generator().manual_seed(185)
    q = torch.randn(1, 2, 65, 16, generator=generator).to(dtype).requires_grad_()
    k = torch.randn(1, 2, 7, 16, generator=generator).to(dtype).requires_grad_()
    v = torch.randn(1, 2, 7, 12, generator=generator).to(dtype).requires_grad_()
    mask = torch.arange(7).view(1, 1, 1, -1) < 5 if masked else None
    kernel = DenseRingBlockAttention()
    expected = kernel.attention_block(q, k, v, mask=mask)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = kernel.attention_block(q, k, v, mask=mask)
        assert torch.is_autocast_enabled("cpu")
    for actual, reference in (
        (result.output, expected.output),
        (result.maximum, expected.maximum),
        (result.exponential_sum, expected.exponential_sum),
    ):
        assert actual.dtype == torch.float32
        assert not actual.requires_grad
        assert torch.equal(actual, reference)


def test_dense_ring_block_bounds_both_matrix_products(monkeypatch: pytest.MonkeyPatch) -> None:
    q = torch.randn(2, 4, 129, 16)
    k = torch.randn(2, 4, 127, 16)
    v = torch.randn(2, 4, 127, 12)
    matmul = torch.matmul
    shapes: list[tuple[int, ...]] = []

    def record_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        result = matmul(left, right)
        shapes.append(tuple(result.shape))
        return result

    monkeypatch.setattr(torch, "matmul", record_matmul)
    DenseRingBlockAttention().attention_block(q, k, v)
    assert shapes == [
        (2, 4, 64, 127),
        (2, 4, 64, 12),
        (2, 4, 64, 127),
        (2, 4, 64, 12),
        (2, 4, 1, 127),
        (2, 4, 1, 12),
    ]


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_dense_ring_blocks_merge_with_an_empty_block(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(185)
    q = torch.randn(1, 2, 65, 16, generator=generator).to(dtype)
    k = torch.randn(1, 2, 127, 16, generator=generator).to(dtype)
    v = torch.randn(1, 2, 127, 12, generator=generator).to(dtype)
    mask = (torch.arange(127) < 80).view(1, 1, 1, -1)
    scores = q.float() @ k.float().transpose(-2, -1) * 0.25
    scores.masked_fill_(~mask, -torch.inf)
    expected = scores.softmax(-1) @ v.float()
    kernel = DenseRingBlockAttention()
    blocks = [
        kernel.attention_block(
            q, k[:, :, start:stop], v[:, :, start:stop], mask=mask[..., start:stop]
        )
        for start, stop in ((0, 43), (43, 86), (86, 127))
    ]
    maximum = torch.stack([block.maximum for block in blocks]).amax(0)
    masses = [(block.maximum - maximum).exp() * block.exponential_sum for block in blocks]
    numerator = torch.stack(
        [block.output * mass for block, mass in zip(blocks, masses, strict=True)]
    ).sum(0)
    actual = numerator / torch.stack(masses).sum(0)
    assert torch.count_nonzero(blocks[-1].output) == 0
    assert torch.count_nonzero(masses[-1]) == 0
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
@pytest.mark.parametrize("empty", (None, "first", "middle", "last", "head", "all"))
def test_merge_attention_blocks_matches_full_float64_reference(
    dtype: torch.dtype, empty: str | None
) -> None:
    generator = torch.Generator().manual_seed(185)
    q = torch.randn(2, 2, 65, 16, generator=generator).to(dtype)
    k = torch.randn(2, 2, 127, 16, generator=generator).to(dtype)
    v = torch.randn(2, 2, 127, 12, generator=generator).to(dtype)
    mask = torch.ones(2, 2, 1, 127, dtype=torch.bool)
    slices = ((0, 43), (43, 86), (86, 127))
    if empty in ("first", "middle", "last"):
        start, stop = slices[("first", "middle", "last").index(empty)]
        mask[..., start:stop] = False
    elif empty == "head":
        mask[:, 0] = False
    elif empty == "all":
        mask.fill_(False)
    scores = (q.double() @ k.double().transpose(-2, -1) * 0.25).masked_fill(~mask, -torch.inf)
    maximum = scores.amax(-1, keepdim=True)
    weights = (scores - maximum.masked_fill(torch.isneginf(maximum), 0)).exp()
    total = weights.sum(-1, keepdim=True)
    expected = weights / total.masked_fill(total == 0, 1) @ v.double()
    kernel = DenseRingBlockAttention()

    actual = merge_attention_blocks(
        kernel.attention_block(
            q, k[:, :, start:stop], v[:, :, start:stop], mask=mask[..., start:stop]
        )
        for start, stop in slices
    )

    assert torch.isfinite(actual.output).all()
    torch.testing.assert_close(actual.output.double(), expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual.maximum.double(), maximum, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual.exponential_sum.double(), total, rtol=2e-6, atol=2e-6)
    if empty == "all":
        assert (
            torch.count_nonzero(actual.output) == torch.count_nonzero(actual.exponential_sum) == 0
        )
        assert torch.isneginf(actual.maximum).all()


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
@pytest.mark.parametrize("changing_maximum", (False, True))
def test_merge_attention_blocks_avoids_unnormalized_value_overflow(
    dtype: torch.dtype, changing_maximum: bool
) -> None:
    q = torch.ones(1, 2, 65, 16, dtype=dtype)
    v = torch.full((1, 2, 65, 12), torch.finfo(dtype).max / 4, dtype=dtype)
    v[..., 6:] *= -1
    kernel = DenseRingBlockAttention()
    blocks = [
        kernel.attention_block(q, torch.full_like(q, score), v)
        for score in ((-40, 40, 0) if changing_maximum else (0, 0, 0))
    ]
    assert all(torch.isfinite(block.output).all() for block in blocks)
    if dtype != torch.float16:
        assert not torch.isfinite(blocks[0].output * blocks[0].exponential_sum).all()

    actual = merge_attention_blocks(iter(blocks))

    assert torch.isfinite(actual.output).all()
    torch.testing.assert_close(actual.output, v.float(), rtol=2e-6, atol=0)
    expected_mass = 65 if changing_maximum else 195
    torch.testing.assert_close(
        actual.exponential_sum, torch.full_like(actual.exponential_sum, expected_mass)
    )


def test_merge_attention_blocks_consumes_provider_statistics_without_retaining_blocks() -> None:
    references: list[weakref.ReferenceType[torch.Tensor]] = []

    def blocks() -> Iterator[AttentionBlockResult]:
        for index in range(4):
            assert all(reference() is None for reference in references[:-1])
            output = torch.full((1, 1, 1, 2), float(index))
            references.append(weakref.ref(output))
            yield AttentionBlockResult(
                output,
                torch.full((1, 1, 1, 1), float(index)),
                torch.full((1, 1, 1, 1), float(index + 1)),
            )

    actual = merge_attention_blocks(blocks())

    masses = torch.arange(1, 5, dtype=torch.float64) * (torch.arange(4) - 3).double().exp()
    expected = (torch.arange(4) * masses).sum() / masses.sum()
    torch.testing.assert_close(actual.output, torch.full_like(actual.output, expected.item()))
    torch.testing.assert_close(actual.maximum, torch.full_like(actual.maximum, 3))
    torch.testing.assert_close(
        actual.exponential_sum, torch.full_like(actual.exponential_sum, masses.sum().item())
    )
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("count", (1, 3))
def test_merge_attention_blocks_is_inference_only_without_input_mutation_or_aliasing(
    count: int,
) -> None:
    block = AttentionBlockResult(
        torch.ones(1, 2, 3, 4, requires_grad=True),
        torch.zeros(1, 2, 3, 1, requires_grad=True),
        torch.ones(1, 2, 3, 1, requires_grad=True),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = merge_attention_blocks([block] * count)
    for original, result in (
        (block.output, actual.output),
        (block.maximum, actual.maximum),
        (block.exponential_sum, actual.exponential_sum),
    ):
        assert result.dtype == torch.float32 and not result.requires_grad
        assert result.data_ptr() != original.data_ptr()
    assert torch.equal(block.output, torch.ones_like(block.output))
    assert torch.equal(block.maximum, torch.zeros_like(block.maximum))
    assert torch.equal(block.exponential_sum, torch.ones_like(block.exponential_sum))


@pytest.mark.parametrize(
    ("field", "tensor", "message"),
    (
        ("output", torch.ones(1, 1, 2), "nonempty"),
        ("output", torch.ones(1, 1, 0, 2), "nonempty"),
        ("maximum", torch.ones(1, 1, 1, 2), "statistics"),
        ("exponential_sum", torch.ones(1, 1, 1), "statistics"),
        ("output", torch.ones(1, 1, 1, 2, dtype=torch.float16), "float32"),
        ("maximum", torch.ones(1, 1, 1, 1, dtype=torch.bfloat16), "float32"),
        ("exponential_sum", torch.ones(1, 1, 1, 1, dtype=torch.float64), "float32"),
        ("maximum", torch.ones(1, 1, 1, 1, device="meta"), "share a device"),
        ("exponential_sum", torch.ones(1, 1, 1, 1, device="meta"), "share a device"),
        ("output", torch.ones(1, 1, 1, 3), "matching output shapes"),
    ),
)
def test_merge_attention_blocks_rejects_incompatible_geometry(
    field: str, tensor: torch.Tensor, message: str
) -> None:
    block = AttentionBlockResult(
        torch.ones(1, 1, 1, 2), torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1)
    )
    with pytest.raises(ValueError, match=message):
        merge_attention_blocks((block, replace(block, **{field: tensor})))


def test_merge_attention_blocks_requires_nonempty_coupled_results() -> None:
    with pytest.raises(ValueError, match="at least one"):
        merge_attention_blocks(())
    with pytest.raises(TypeError, match="AttentionBlockResult"):
        merge_attention_blocks((cast(AttentionBlockResult, torch.ones(1, 1, 1, 2)),))


@pytest.mark.parametrize(
    "valid",
    (
        (True, True, False, True, False, False, True, True, False),
        (True, False, True, True, False, True, True, False),
    ),
)
def test_compacted_ulysses_attention_matches_masked_float64_attention(
    valid: tuple[bool, ...],
) -> None:
    q, k, v = _make_inputs(4117, len(valid), torch.float64)
    validity = torch.tensor(valid, dtype=torch.bool)
    masked = builtin_sdpa_kernel()(q, k, v, mask=validity.view(1, 1, 1, -1))
    expected = masked.masked_fill(~validity.view(1, 1, -1, 1), 0.0)
    seen: list[tuple[int, int, int, torch.Tensor | None]] = []

    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        seen.append((q.shape[2], k.shape[2], v.shape[2], mask))
        return builtin_sdpa_kernel()(
            q, k, v, mask=mask, causal=causal, scale=scale, enable_gqa=enable_gqa
        )

    compacted = sequence_exchange_module._attend_valid_rows(  # pyright: ignore[reportPrivateUsage]
        q, k, v, validity, kernel, None
    )

    valid_rows = sum(valid)
    assert seen == [(valid_rows, valid_rows, valid_rows, None)]
    torch.testing.assert_close(compacted, expected, rtol=1.0e-12, atol=1.0e-12)
    assert compacted.shape == q.shape
    assert torch.count_nonzero(compacted[:, :, ~validity]).item() == 0


def test_ulysses_attention_without_padding_passes_full_tensors_unmasked() -> None:
    q, k, v = _make_inputs(4219, 6, torch.float64)
    seen: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] = []

    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        del causal, scale, enable_gqa
        seen.append((q, k, v, mask))
        return q

    output = sequence_exchange_module._attend_valid_rows(  # pyright: ignore[reportPrivateUsage]
        q, k, v, torch.ones(6, dtype=torch.bool), kernel, None
    )

    assert seen == [(q, k, v, None)]
    assert output is q


def test_padded_ulysses_shards_preserve_output_geometry_and_zero_queries() -> None:
    results = _execute(_Case("padded-U4R1", 1, 4, 1, sequence_length=13))

    assert results[0][1] <= _INSTRUMENTED_EQUIVALENCE_BOUND


def test_padded_tail_is_excluded_from_attention() -> None:
    results = _execute(_Case("padded-U1R4", 1, 1, 4, sequence_length=13))

    assert results[0][1] <= _INSTRUMENTED_EQUIVALENCE_BOUND


def test_ring_plan_facts_are_stable() -> None:
    assert RING_MERGE_ALGORITHM == "ring-lse-fp32.v1"
    assert RING_TRAVERSAL_ORDER == "ascending-ring-step.v1"
    assert RING_ACCUMULATION_ORDER == "sequential in ring-step order"
    assert RING_ACCUMULATION_DTYPE == "float32"


@pytest.mark.parametrize(
    "values",
    (
        (True, "ring-chunk", 0, (0,)),
        ("other", "ring-chunk", 0, (0,)),
        ("ring", True, 0, (0,)),
        ("ring", "other", 0, (0,)),
        ("ring", "ring-chunk", True, (0,)),
        ("ring", "ring-chunk", -1, (0,)),
        ("ring", "ring-chunk", 0, ()),
        ("ring", "ring-chunk", 0, [0]),
        ("ring", "ring-chunk", 0, (True,)),
    ),
)
def test_sequence_exchange_event_rejects_invalid_construction(
    values: tuple[object, object, object, object],
) -> None:
    with pytest.raises(SequenceExchangeError):
        SequenceExchangeEvent(*values)  # type: ignore[arg-type]


def test_submission_slots_and_completion_events_are_explicit_and_ordered() -> None:
    tensor = torch.zeros(1, 2, 3, 4)
    submissions = SequenceExchangeSubmissions.from_tensors(
        (tensor[:, :, :1], tensor[:, :, 1:]), tensor, tensor
    )

    assert tuple((item.slot, item.chunk, item.ready.ordinal) for item in submissions.items) == (
        ("q", 0, 0),
        ("k", 0, 1),
        ("v", 0, 2),
        ("q", 1, 3),
    )


def test_backend_waits_for_submissions_in_event_order_and_exposes_completion_handles() -> None:
    waits: list[int] = []
    tensor = torch.zeros(1, 2, 3, 4)
    submissions = SequenceExchangeSubmissions(
        tuple(
            SequenceExchangeSubmission(
                slot,
                0,
                tensor,
                _RecordingReadyEvent(index, waits),
            )
            for index, slot in enumerate(("q", "k", "v"))
        )
    )
    mesh = UspMesh.build(guidance=1, ulysses=1, ring=1)
    partition = plan_sequence_partition(3, 1)
    layout = SequenceLayout(3, partition.shards[0], "event-order", 2, 0, 2, 0, mesh.digest)
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{os.path.join(directory, 'event-rendezvous')}",
            rank=0,
            world_size=1,
        )
        try:
            result = UspSequenceExchange(mesh).attend_submissions(
                submissions, layout, builtin_sdpa_kernel()
            )
        finally:
            dist.destroy_process_group()

    assert waits == [0, 1, 2]
    assert tuple(event.completed.ordinal for event in result.completions) == (3, 3, 3)
    for event in result.completions:
        event.completed.wait()


@pytest.mark.parametrize(
    "build",
    (
        lambda: SequenceExchangeOrderEvent(True),
        lambda: SequenceExchangeSubmission(
            "other",  # type: ignore[arg-type]
            0,
            torch.zeros(1),
            SequenceExchangeOrderEvent(0),
        ),
        lambda: SequenceExchangeCompletionEvent(
            "q", 0, SequenceExchangeOrderEvent(1), SequenceExchangeOrderEvent(0)
        ),
        lambda: SequenceExchangeSubmissions(
            (SequenceExchangeSubmission("q", 0, torch.zeros(1), SequenceExchangeOrderEvent(0)),)
        ),
    ),
)
def test_submission_records_reject_invalid_construction(build: Callable[[], object]) -> None:
    with pytest.raises(SequenceExchangeError):
        build()


def _completion(
    slot: str,
    chunk: int = 0,
    *,
    ready: int = 0,
    completed: int = 1,
) -> SequenceExchangeCompletionEvent:
    return SequenceExchangeCompletionEvent(
        slot,  # type: ignore[arg-type]
        chunk,
        _RecordingReadyEvent(ready, []),
        _RecordingReadyEvent(completed, []),
    )


def _valid_completions() -> tuple[SequenceExchangeCompletionEvent, ...]:
    return (
        _completion("q", ready=0, completed=3),
        _completion("k", ready=1, completed=3),
        _completion("v", ready=2, completed=3),
    )


@pytest.mark.parametrize(
    "build",
    (
        lambda: SequenceExchangeSubmission("q", 0, torch.zeros(1), SequenceExchangeOrderEvent(0)),
        lambda: SequenceExchangeSubmissions.from_tensors(
            torch.zeros(1), torch.zeros(1), torch.zeros(1)
        ),
        lambda: _completion("q", ready=-1, completed=1),
        lambda: _completion("q", ready=0, completed=-1),
        lambda: _completion("q", ready=1, completed=1),
        lambda: SequenceExchangeResult(torch.zeros(1, 2, 3, 4), ()),
        lambda: SequenceExchangeResult(
            torch.zeros(1, 2, 3, 4),
            (_completion("q"), _completion("q"), _completion("k"), _completion("v")),
        ),
        lambda: SequenceExchangeResult(
            torch.zeros(1, 2, 3, 4), (_completion("q"), _completion("k"))
        ),
        lambda: SequenceExchangeResult(torch.zeros(1), _valid_completions()),
    ),
)
def test_submission_completion_and_result_semantics_reject_invalid_records(
    build: Callable[[], object],
) -> None:
    with pytest.raises(SequenceExchangeError):
        build()


@pytest.mark.parametrize(
    ("site", "hook"),
    (
        ("before", object()),
        ("after", 1),
        ("before", _async_exchange_hook),
        ("after", _AsyncCallableHook()),
    ),
)
def test_sequence_exchange_hooks_reject_invalid_construction(
    site: str,
    hook: object,
) -> None:
    with pytest.raises(SequenceExchangeError):
        SequenceExchangeHooks(**{site: hook})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    (
        {"merge_algorithm": True},
        {"merge_algorithm": "other"},
        {"traversal_contract": "other"},
        {"accumulation_order": "other"},
        {"accumulation_dtype": "bfloat16"},
        {"traversal_order": ()},
        {"traversal_order": [0]},
        {"traversal_order": (True,)},
        {"traversal_order": (0, 0)},
    ),
)
def test_ring_attention_plan_rejects_invalid_construction(
    changes: dict[str, object],
) -> None:
    with pytest.raises(SequenceExchangeError):
        _ring_plan(**changes)


def test_default_pg_timeout_is_none_without_initialized_group() -> None:
    # None defers to new_group's backend-specific default (10 minutes for
    # NCCL, 30 for others), preserving pre-existing subgroup semantics
    # whenever the default group's configured timeout cannot be read back.
    assert not dist.is_initialized()
    assert sequence_exchange_module._default_pg_timeout() is None  # pyright: ignore[reportPrivateUsage]


def test_hook_failure_is_rank_fatal_without_deadlock() -> None:
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "hook-failure-rendezvous")
        if sys.platform.startswith("win"):
            with pytest.raises(ProcessRaisedException) as raised:
                spawn(
                    _run_hook_failure,
                    args=(rendezvous, queue),
                    nprocs=2,
                    join=True,
                )
            assert raised.value.error_index == 0
            rank, status, message = queue.get()
            if rank != 0:
                rank, status, message = queue.get()
            assert rank == 0
            assert (status, message) == (
                "hook",
                "before hook failed for operation 'head-to-sequence' at step 0",
            )
        else:
            spawn(
                _run_hook_failure,
                args=(rendezvous, queue),
                nprocs=2,
                join=True,
            )
            results = {
                rank: (status, message) for rank, status, message in (queue.get(), queue.get())
            }
            assert results[0] == (
                "hook",
                "before hook failed for operation 'head-to-sequence' at step 0",
            )
            assert results[1][0] == "peer"


def _run_mixed_transport(rank: int, rendezvous: str, result_queue: Any) -> None:
    os.environ["DINKSTER_SINGLE_JOB_SEQUENCE_TRANSPORT"] = "nccl" if rank == 0 else "peer-copy"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.set_num_threads(1)
        case = _Case("U2R1", 1, 2, 1)
        mesh = _mesh(case)
        placement = _placement(case, mesh)
        try:
            sequence_exchange_module._SequenceExchangeGroups.create(  # pyright: ignore[reportPrivateUsage]
                mesh, placement
            )
        except SequenceExchangeError as error:
            result_queue.put((rank, "guard", str(error)))
        else:
            result_queue.put((rank, "created", ""))
    finally:
        dist.destroy_process_group()


def test_mixed_sequence_transport_is_rejected_at_group_creation() -> None:
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "mixed-transport-rendezvous")
        spawn(
            _run_mixed_transport,
            args=(rendezvous, queue),
            nprocs=2,
            join=True,
        )
    results = {rank: (status, message) for rank, status, message in (queue.get(), queue.get())}
    for status, message in results.values():
        assert status == "guard"
        assert "identical on every rank" in message


def test_sequence_transport_env_defaults_and_rejects_unknown_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DINKSTER_SINGLE_JOB_SEQUENCE_TRANSPORT", raising=False)
    assert sequence_exchange_module._ulysses_transport() == "nccl"  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_TRANSPORT", "peer-copy")
    assert sequence_exchange_module._ulysses_transport() == "peer-copy"  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setenv("DINKSTER_SINGLE_JOB_SEQUENCE_TRANSPORT", "bogus")
    with pytest.raises(SequenceExchangeError, match="'nccl' or 'peer-copy'"):
        sequence_exchange_module._ulysses_transport()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_multi_rank_ring_rejects_non_float32_before_group_creation(
    dtype: torch.dtype,
) -> None:
    mesh = UspMesh.build(guidance=1, ulysses=1, ring=2)
    partition = plan_sequence_partition(12, 2)
    shard = partition.shards[0]
    layout = SequenceLayout(
        12,
        shard,
        "fixed-token-order",
        8,
        0,
        8,
        0,
        mesh.digest,
    )
    tensor = torch.zeros(1, 8, partition.chunk, 4, dtype=dtype)

    with pytest.raises(SequenceExchangeError, match="requires float32"):
        RingSequenceExchange(mesh).attend(
            tensor,
            tensor,
            tensor,
            layout,
            builtin_sdpa_kernel(),
        )


def test_ulysses_rejects_indivisible_heads_before_group_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = UspMesh.build(guidance=1, ulysses=2, ring=1)
    partition = plan_sequence_partition(12, 2)
    shard = partition.shards[0]
    layout = SequenceLayout(
        12,
        shard,
        "fixed-token-order",
        3,
        0,
        1,
        0,
        mesh.digest,
    )
    tensor = torch.zeros(1, 3, partition.chunk, 4)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    with pytest.raises(SequenceExchangeError, match="divisible"):
        UlyssesSequenceExchange(mesh).attend(
            tensor,
            tensor,
            tensor,
            layout,
            builtin_sdpa_kernel(),
        )
