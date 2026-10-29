from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dinkster_inference import (
    ManifestRefusal,
    ManifestRefusalCode,
    PlacementMap,
    SamplingTimelineSchedule,
    UspMesh,
    build_canonical_manifest,
    plan_sequence_partition,
    prove_manifest_consensus,
    realize_sampling_timeline,
)
from dinkster_inference.devices import BFLOAT16, FLOAT32, INT8
from dinkster_inference.partition_compatibility import (
    ContiguousShard,
    FullSequence,
    PartitionCompatibility,
    UlyssesHeadScatter,
)
from dinkster_inference.sampling_timeline import use_realized_sampling_timeline
from dinkster_inference_torch.attention import (
    _COMFY_KITCHEN_INT8,  # pyright: ignore[reportPrivateUsage]
    builtin_sdpa_kernel,
    schedule_aware_attention_kernel,
)
from dinkster_inference_torch.distributed import (
    DistributedSamplingConfig,
    SequenceDigestConsensusTransport,
    gather_sequence_device_bindings,
)
from dinkster_inference_torch.minimax_h3_attention import MiniMaxH3PackedSequenceFacts
from dinkster_inference_torch.sequence_exchange import (
    RING_ACCUMULATION_DTYPE,
    RING_ACCUMULATION_ORDER,
    RING_MERGE_ALGORITHM,
    RING_TRAVERSAL_ORDER,
    DenseRingBlockAttention,
    RingAttentionPlan,
)
from dinkster_inference_torch.sequence_parallel_attention import (
    SequenceParallelAttentionError,
    SequenceParallelAttentionKernel,
)
from dinkster_inference_torch.sequence_parallel_plan import (
    CompiledSequenceParallelPlan,
    build_usp_compiled_plan_slot,
    compile_sequence_parallel_plan,
    exchange_backend_identity,
    packing_geometry_digest,
    ring_peer_schedules,
)
from dinkster_inference_torch.sequence_sharding import PackedSequenceFacts


def _facts() -> MiniMaxH3PackedSequenceFacts:
    return MiniMaxH3PackedSequenceFacts(
        13,
        (
            (0, 3, "text"),
            (3, 7, "audio"),
            (7, 13, "video"),
        ),
    )


def _plan(
    *,
    provider: str = "dense-ring:torch=2.test",
    ulysses: int = 2,
    ring: int = 2,
    **overrides: Any,
) -> CompiledSequenceParallelPlan:
    mesh = UspMesh.build(guidance=1, ulysses=ulysses, ring=ring)
    arguments: dict[str, Any] = dict(
        mesh=mesh,
        placement=PlacementMap.identity(mesh.process_mesh),
        partition=plan_sequence_partition(13, ulysses * ring),
        packed_sequence_facts=_facts(),
        head_count=8,
        rank_device_bindings=("cpu",) * mesh.world_size,
        routed_attention_backend_identity=provider,
        exchange_backend=exchange_backend_identity(mesh),
        attention_kernel=DenseRingBlockAttention(),
        partition_compatibility=DenseRingBlockAttention.partition_compatibility,
        compute_dtype=FLOAT32,
        device_kind="cpu",
    )
    return compile_sequence_parallel_plan(**(arguments | overrides))


def test_compiled_plan_slot_is_deterministic_and_binds_every_fact() -> None:
    first = _plan()
    repeated = _plan()
    changed = _plan(provider="torch-sdpa:torch=other")

    assert first == repeated
    assert first.digest == repeated.digest
    assert (
        build_usp_compiled_plan_slot(first).digest == build_usp_compiled_plan_slot(repeated).digest
    )
    assert first.digest != changed.digest
    assert (
        build_usp_compiled_plan_slot(first).digest != build_usp_compiled_plan_slot(changed).digest
    )
    assert tuple(fact.partition("=")[0] for fact in first.facts) == (
        "mesh.layout",
        "mesh.axes",
        "mesh.world_size",
        "mesh.subgroup_digest",
        "placement.logical_to_physical",
        "placement.rank_to_device",
        "partition.global_sequence_length",
        "partition.attention_head_count",
        "partition.chunk_size",
        "partition.shards",
        "partition.packing_geometry_digest",
        "partition.rope_slicing",
        "collective.traversal_contract",
        "collective.ring_peer_schedules",
        "collective.accumulation_order",
        "collective.merge_algorithm",
        "collective.accumulation_dtype",
        "provider.routed_attention_backend",
        "provider.exchange_backend",
        "provider.attention_kernel_contract",
        "provider.partition_compatibility",
        "provider.compute_dtype",
        "provider.device_kind",
    )
    assert all("\n" not in fact for fact in first.facts)
    with pytest.raises(FrozenInstanceError):
        first.partition = plan_sequence_partition(12, 4)  # type: ignore[misc]
    with pytest.raises(ValueError, match="only by the compiler"):
        CompiledSequenceParallelPlan(
            first.mesh,
            first.placement,
            first.partition,
            first.layouts,
            first.rank_device_bindings,
            first.packing_geometry_digest,
            first.ring_peer_schedules,
            first.routed_attention_backend_identity,
            first.exchange_backend_identity,
            first.attention_kernel_contract,
            first.partition_compatibility,
            first.compute_dtype,
            first.device_kind,
            object(),
        )


def test_compiler_accepts_model_declared_packing_layout() -> None:
    facts = PackedSequenceFacts(
        6,
        ((0, 2, "prompt"), (2, 6, "latent")),
        "example-model-packed-sequence.v1",
    )
    mesh = UspMesh.build(guidance=1, ulysses=2, ring=1)

    plan = compile_sequence_parallel_plan(
        mesh=mesh,
        placement=PlacementMap.identity(mesh.process_mesh),
        partition=plan_sequence_partition(6, 2),
        packed_sequence_facts=facts,
        head_count=4,
        rank_device_bindings=("device:0", "device:1"),
        routed_attention_backend_identity="example-attention.v1",
        exchange_backend=exchange_backend_identity(mesh),
        attention_kernel=builtin_sdpa_kernel(),
        partition_compatibility=PartitionCompatibility(
            1, (UlyssesHeadScatter(),), ContiguousShard(2), (FLOAT32,), ("cpu",), False
        ),
        compute_dtype=FLOAT32,
        device_kind="cpu",
    )

    assert {layout.original_token_order_reference for layout in plan.layouts} == {
        "example-model-packed-sequence.v1"
    }
    assert packing_geometry_digest(facts) != packing_geometry_digest(
        replace(facts, layout_identity="example-model-packed-sequence.v2")
    )


@pytest.mark.parametrize("ulysses", (1, 2))
def test_coupled_ring_compiles_without_enabling_tensor_executor(ulysses: int) -> None:
    plan = _plan(ulysses=ulysses)
    assert plan.attention_kernel_contract == "dinkster.attention-kernel.block.v1"
    with pytest.raises(SequenceParallelAttentionError, match="must implement AttentionKernel"):
        SequenceParallelAttentionKernel(
            plan.mesh,
            DenseRingBlockAttention(),  # type: ignore[arg-type]
            plan.layouts[0],
        )


@pytest.mark.parametrize(
    "kernel",
    (
        builtin_sdpa_kernel(),
        _COMFY_KITCHEN_INT8,
        schedule_aware_attention_kernel("sdpa", builtin_sdpa_kernel()),
        schedule_aware_attention_kernel("dinkster_kitchen_int8", _COMFY_KITCHEN_INT8),
    ),
)
def test_h3_ulysses_declaration_preserves_tensor_kernel_contract(kernel: Any) -> None:
    plan = _plan(
        ring=1,
        attention_kernel=kernel,
        partition_compatibility=kernel.partition_compatibility,
        compute_dtype=BFLOAT16,
        device_kind="cuda",
        rank_device_bindings=("cuda:0", "cuda:1"),
    )
    assert plan.attention_kernel_contract == "dinkster.attention-kernel.tensor.v1"
    assert plan.compute_dtype == BFLOAT16


@pytest.mark.parametrize("kernel", (builtin_sdpa_kernel(), _COMFY_KITCHEN_INT8))
def test_sdpa_declares_cpu_but_int8_declines_cpu(kernel: Any) -> None:
    arguments: dict[str, Any] = dict(
        ring=1,
        attention_kernel=kernel,
        partition_compatibility=kernel.partition_compatibility,
        compute_dtype=BFLOAT16,
    )
    if kernel is _COMFY_KITCHEN_INT8:
        with pytest.raises(ValueError, match="device kind 'cpu' is not supported"):
            _plan(**arguments)
    else:
        assert _plan(**arguments).device_kind == "cpu"


def test_scheduled_declaration_tracks_the_active_kernel_before_compilation() -> None:
    kernel: Any = schedule_aware_attention_kernel("dinkster_kitchen_int8", _COMFY_KITCHEN_INT8)
    timeline = realize_sampling_timeline(
        SamplingTimelineSchedule("dinkster_kitchen_int8", 0.5, 1.0), (2.0, 1.0, 0.0)
    )
    with use_realized_sampling_timeline(timeline) as activate:
        activate(0)
        sdpa_plan = _plan(
            ring=1,
            attention_kernel=kernel,
            partition_compatibility=kernel.partition_compatibility,
            compute_dtype=BFLOAT16,
        )
        assert sdpa_plan.device_kind == "cpu"
        activate(1)
        with pytest.raises(ValueError, match="device kind 'cpu' is not supported"):
            _plan(
                ring=1,
                attention_kernel=kernel,
                partition_compatibility=kernel.partition_compatibility,
                compute_dtype=BFLOAT16,
            )
    assert kernel.partition_compatibility == _COMFY_KITCHEN_INT8.partition_compatibility
    assert sdpa_plan.partition_compatibility != kernel.partition_compatibility


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"attention_kernel": builtin_sdpa_kernel()}, "requires AttentionBlockKernel"),
        ({"compute_dtype": INT8}, "dtype"),
        ({"device_kind": "meta"}, "device kind"),
        ({"head_count": 3}, "positive multiple"),
        ({"attention_kernel": object()}, "requires AttentionBlockKernel"),
        ({"partition_compatibility": None}, "exact PartitionCompatibility"),
    ),
)
def test_compiler_refuses_incompatible_provider_inputs(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _plan(**overrides)


@pytest.mark.parametrize("ulysses", (1, 2))
@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"provides_matching_block_normalization": False}, "same local block callable"),
        ({"tensor_expectation": ContiguousShard(1)}, "sequence dimension"),
        ({"tensor_expectation": ContiguousShard(2, False)}, "equal chunks"),
        ({"tensor_expectation": FullSequence()}, "full-sequence"),
        ({"modes": (UlyssesHeadScatter(),)}, "not declared"),
    ),
)
def test_compiler_checks_declaration_before_minting_plan(
    ulysses: int, changes: dict[str, Any], message: str
) -> None:
    declaration = replace(DenseRingBlockAttention.partition_compatibility, **changes)
    with pytest.raises(ValueError, match=message):
        _plan(ulysses=ulysses, partition_compatibility=declaration)


@pytest.mark.parametrize(
    "changes",
    (
        {"head_count": 16},
        {"compute_dtype": BFLOAT16},
        {"device_kind": "cuda"},
        {
            "partition_compatibility": replace(
                DenseRingBlockAttention.partition_compatibility, supported_dtypes=(FLOAT32,)
            )
        },
    ),
)
def test_checked_provider_facts_bind_plan_slot_and_all_rank_digests(
    changes: dict[str, Any],
) -> None:
    original = _plan()
    changed = _plan(**changes)
    assert original.digest != changed.digest
    assert (
        build_usp_compiled_plan_slot(original).digest
        != build_usp_compiled_plan_slot(changed).digest
    )
    assert all(
        a != b for a, b in zip(original.rank_plan_digests, changed.rank_plan_digests, strict=True)
    )


@pytest.mark.parametrize(
    "facts",
    (
        lambda: PackedSequenceFacts(2, ((0, 2, ""),), "layout.v1"),
        lambda: PackedSequenceFacts(2, ((0, 2, "tokens"),), ""),
        lambda: PackedSequenceFacts(2, ((0, 1, "tokens"),), "layout.v1"),
    ),
)
def test_packed_sequence_declaration_refuses_ambiguous_geometry(facts: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        facts()


@pytest.mark.parametrize(
    ("mesh", "expected"),
    (
        (
            UspMesh.build(guidance=1, ulysses=2, ring=3),
            ((0, 2, 1), (1, 0, 2), (2, 1, 0), (3, 5, 4), (4, 3, 5), (5, 4, 3)),
        ),
        (UspMesh.build(guidance=2, ulysses=1, ring=2), ((0, 1), (1, 0), (2, 3), (3, 2))),
    ),
)
def test_analytic_ring_schedules_match_ring_plan_traversals(
    mesh: UspMesh, expected: tuple[tuple[int, ...], ...]
) -> None:
    plans = tuple(
        RingAttentionPlan(
            RING_MERGE_ALGORITHM,
            traversal,
            RING_TRAVERSAL_ORDER,
            RING_ACCUMULATION_ORDER,
            RING_ACCUMULATION_DTYPE,
        )
        for traversal in expected
    )
    assert ring_peer_schedules(mesh) == tuple(plan.traversal_order for plan in plans)


def test_declared_rope_slices_match_imperative_slice_and_zero_append() -> None:
    plan = _plan()
    global_rope = torch.arange(13, dtype=torch.float32).view(1, 13, 1)

    for layout in plan.layouts:
        shard = layout.shard
        sliced = global_rope[:, shard.start : shard.stop]
        if shard.padded_rows:
            sliced = torch.cat(
                (sliced, torch.zeros(1, shard.padded_rows, 1)),
                dim=1,
            )
        assert layout.rope_position_offset == shard.start
        assert sliced.shape[1] == plan.partition.chunk
        assert torch.equal(sliced[:, : shard.valid_rows], global_rope[:, shard.start : shard.stop])
        if shard.padded_rows:
            assert torch.count_nonzero(sliced[:, shard.valid_rows :]).item() == 0


class _MismatchedTransport:
    def exchange_digest(self, rank: int, digest: str) -> tuple[str, ...]:
        del rank
        return digest, "f" * 64


def test_consensus_mismatch_has_no_override_and_duck_token_refuses() -> None:
    plan = _plan()
    manifest = build_canonical_manifest(
        runtime_identity=f"native:test:{'0' * 64}",
        invocation_facts=("test=refusal",),
        slots=(build_usp_compiled_plan_slot(plan),),
        rank_plan_digests=tuple((digest,) for digest in plan.rank_plan_digests[:2]),
    )
    with pytest.raises(ManifestRefusal) as mismatch:
        prove_manifest_consensus(manifest, rank=0, transport=_MismatchedTransport())
    assert mismatch.value.code is ManifestRefusalCode.CONSENSUS_DIGEST_MISMATCH

    mesh = UspMesh.build(guidance=1, ulysses=1, ring=2)
    partition = plan_sequence_partition(4, 2)
    with pytest.raises(ManifestRefusal) as forged:
        SequenceParallelAttentionKernel(
            mesh,
            builtin_sdpa_kernel(),
            plan.layouts[0].__class__(
                4,
                partition.shards[0],
                "order",
                4,
                0,
                4,
                0,
                mesh.digest,
            ),
            None,
            object(),  # type: ignore[arg-type]
            "attention",
            "exchange",
        )
    assert forged.value.code is ManifestRefusalCode.FORGED_CONSENSUS_TOKEN


def _run_digest_consensus(rank: int, rendezvous: str, mismatch: bool, queue: Any) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        config = DistributedSamplingConfig(
            rank,
            2,
            "sequence",
            f"file://{rendezvous}",
            "1" * 32,
            sequence_ulysses=2,
            sequence_ring=1,
        )
        bindings = gather_sequence_device_bindings(config, torch.device("cpu"), f"device:{rank}")
        transport = SequenceDigestConsensusTransport(config, torch.device("cpu"))
        rank_digests = tuple(
            (hashlib.sha256(f"rank:{peer}".encode()).hexdigest(),) for peer in range(2)
        )
        manifest = build_canonical_manifest(
            runtime_identity=f"native:test:{'0' * 64}",
            invocation_facts=(f"case={'mismatch-' + str(rank) if mismatch else 'agreement'}",),
            slots=(),
            rank_plan_digests=rank_digests,
        )
        try:
            token = prove_manifest_consensus(manifest, rank=rank, transport=transport)
        except ManifestRefusal as error:
            queue.put((rank, bindings, error.code.value))
        else:
            queue.put((rank, bindings, token.group_size, token.rank))
    finally:
        dist.destroy_process_group()


def _run_device_binding_case(
    rank: int,
    world_size: int,
    rendezvous: str,
    invalid_rank: int | None,
    queue: Any,
    ready: Any,
) -> None:
    # Start Gloo's connection deadline after every spawned interpreter is ready.
    ready.wait(timeout=30)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    try:
        config = DistributedSamplingConfig(
            rank,
            world_size,
            "sequence",
            f"file://{rendezvous}",
            "1" * 32,
            sequence_ulysses=2,
            sequence_ring=1,
            sequence_guidance=world_size // 2,
        )
        identity = "invalid\nidentity" if rank == invalid_rank else f"device:{rank}"
        try:
            bindings = gather_sequence_device_bindings(config, torch.device("cpu"), identity)
        except (RuntimeError, ValueError) as error:
            queue.put((rank, type(error).__name__, str(error)))
        else:
            queue.put((rank, bindings))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
@pytest.mark.parametrize("invalid_rank", (None, 0))
def test_device_bindings_cover_guidance_subgroups_and_refuse_synchronously(
    invalid_rank: int | None,
) -> None:
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    ready = context.Barrier(4)
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "manifest-device-bindings")
        processes = [
            context.Process(
                target=_run_device_binding_case,
                args=(rank, 4, rendezvous, invalid_rank, queue, ready),
            )
            for rank in range(4)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
    results = sorted(queue.get() for _ in range(4))
    if invalid_rank is None:
        assert results == [
            (rank, ("device:0", "device:1", "device:2", "device:3")) for rank in range(4)
        ]
    else:
        assert results == [
            (
                0,
                "ValueError",
                "sequence device identity must be a bounded newline-free UTF-8 exact string",
            ),
            (1, "RuntimeError", "peer sequence device identity is invalid"),
            (2, "RuntimeError", "peer sequence device identity is invalid"),
            (3, "RuntimeError", "peer sequence device identity is invalid"),
        ]


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
@pytest.mark.parametrize("mismatch", (False, True))
def test_digest_consensus_transport_over_spawned_gloo(mismatch: bool) -> None:
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "manifest-consensus")
        processes = [
            context.Process(
                target=_run_digest_consensus,
                args=(rank, rendezvous, mismatch, queue),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
    results = sorted(queue.get() for _ in range(2))
    if mismatch:
        assert results == [
            (0, ("device:0", "device:1"), ManifestRefusalCode.CONSENSUS_DIGEST_MISMATCH.value),
            (1, ("device:0", "device:1"), ManifestRefusalCode.CONSENSUS_DIGEST_MISMATCH.value),
        ]
    else:
        assert results == [
            (0, ("device:0", "device:1"), 2, 0),
            (1, ("device:0", "device:1"), 2, 1),
        ]
