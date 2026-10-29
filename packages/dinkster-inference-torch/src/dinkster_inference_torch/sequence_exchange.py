"""Replaceable Ulysses and Ring attention exchange backends."""

from __future__ import annotations

import inspect
import math
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from typing import ClassVar, Literal, Protocol, TypeAlias, cast, runtime_checkable

import torch
import torch.distributed as dist
from dinkster_inference import (
    ManifestConsensusToken,
    ManifestRefusal,
    ManifestRefusalCode,
    PlacementMap,
    SequenceLayout,
    SequencePartition,
    SequenceShard,
    UspMesh,
    plan_sequence_partition,
)
from dinkster_inference.devices import BFLOAT16, FLOAT16, FLOAT32
from dinkster_inference.partition_compatibility import (
    ContiguousShard,
    PartitionCompatibility,
    RingSequenceShard,
    UlyssesRingHybrid,
)

from .attention import AttentionBlockResult, AttentionKernel

RING_MERGE_ALGORITHM = "ring-lse-fp32.v1"
RING_TRAVERSAL_ORDER = "ascending-ring-step.v1"
RING_ACCUMULATION_ORDER = "sequential in ring-step order"
RING_ACCUMULATION_DTYPE = "float32"
_EXCHANGE_BACKENDS = frozenset(("ulysses", "ring"))
_EXCHANGE_OPERATIONS = frozenset(("head-to-sequence", "sequence-to-head", "ring-chunk"))
_SUBMISSION_SLOTS = ("q", "k", "v")


class SequenceExchangeError(RuntimeError):
    pass


def _dense_ring_score_tiles(
    q: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float | None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    score_scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    key = k.float().transpose(-2, -1)
    for queries in q.split(64, dim=-2):
        scores = torch.matmul(queries.float(), key) * score_scale
        if mask is not None:
            scores.masked_fill_(~mask, -torch.inf)
        maximum = scores.amax(dim=-1, keepdim=True)
        # An entirely masked block has no softmax mass.
        origin = maximum.masked_fill(torch.isneginf(maximum), 0.0)
        weights = torch.exp(scores - origin)
        yield weights, maximum, weights.sum(dim=-1, keepdim=True)


@torch.no_grad()
def dense_ring_block_statistics(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference-only float32 dense-QK maxima and exponential sums.

    The boolean mask covers keys, as for Ring padding. Each score temporary
    is at most [B, H, 64, K]; query tiling can change GEMM rounding. These
    statistics do not describe quantized or approximate provider scores.
    Float32 conversions and unmasked arithmetic must remain finite.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        maxima: list[torch.Tensor] = []
        sums: list[torch.Tensor] = []
        for _, maximum, total in _dense_ring_score_tiles(q, k, mask, scale):
            maxima.append(maximum)
            sums.append(total)
        return torch.cat(maxima, dim=-2), torch.cat(sums, dim=-2)


class DenseRingBlockAttention:
    """Inference-only dense float32 blocks with coupled output/statistics.

    Uses at most 64 query rows per score temporary, not a fixed total byte
    budget. Output is computed from the same weights as the normalization,
    without invoking another provider or rounding blocks back to input dtype.
    Float32 conversions and unmasked arithmetic, including scaled QK and
    weighted-value accumulation, must remain finite. This is a caller
    precondition, not a runtime overflow check; finite inputs alone do not
    guarantee it. Empty blocks mean entirely masked nonempty tensor extents.
    """

    partition_compatibility: ClassVar[PartitionCompatibility] = PartitionCompatibility(
        revision=1,
        modes=(RingSequenceShard(), UlyssesRingHybrid()),
        tensor_expectation=ContiguousShard(2),
        supported_dtypes=(FLOAT32, FLOAT16, BFLOAT16),
        device_kinds=("cpu", "cuda"),
        provides_matching_block_normalization=True,
    )

    @torch.no_grad()
    def attention_block(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> AttentionBlockResult:
        with torch.autocast(device_type=q.device.type, enabled=False):
            value = v.float()
            outputs: list[torch.Tensor] = []
            maxima: list[torch.Tensor] = []
            sums: list[torch.Tensor] = []
            for weights, maximum, total in _dense_ring_score_tiles(q, k, mask, scale):
                denominator = total.masked_fill(total == 0, 1.0)
                outputs.append(torch.matmul(weights / denominator, value))
                maxima.append(maximum)
                sums.append(total)
            return AttentionBlockResult(
                torch.cat(outputs, dim=-2),
                torch.cat(maxima, dim=-2),
                torch.cat(sums, dim=-2),
            )


@torch.no_grad()
def merge_attention_blocks(blocks: Iterable[AttentionBlockResult]) -> AttentionBlockResult:
    """Merge coupled blocks sequentially as float32 normalized means.

    Consumes a nonempty stream in caller order, retaining only the aggregate
    and current block. Blocks must describe disjoint keys for the same queries,
    with matching score scale and value basis. Coupled statistics and finite
    provider arithmetic are caller preconditions, not reconstructed here.
    Zero-mass rows remain zero output/mass with maximum -inf. Rescaled mass
    sums and weighted means must remain representable in float32.

    Input tensors are not mutated or aliased by the result. No collectives,
    provider selection, output dtype conversion or runtime activation occurs.
    """
    merged: AttentionBlockResult | None = None
    for block in blocks:
        if not isinstance(cast("object", block), AttentionBlockResult):
            raise TypeError("attention blocks must be AttentionBlockResult values")
        output, maximum, total = block.output, block.maximum, block.exponential_sum
        if output.ndim != 4 or any(size == 0 for size in output.shape):
            raise ValueError("attention block output must have nonempty [B,H,Q,Dv] shape")
        statistics_shape = (*output.shape[:-1], 1)
        if maximum.shape != statistics_shape or total.shape != statistics_shape:
            raise ValueError("attention block statistics must have [B,H,Q,1] shape")
        if any(tensor.dtype != torch.float32 for tensor in (output, maximum, total)):
            raise ValueError("attention block output and statistics must be float32")
        if maximum.device != output.device or total.device != output.device:
            raise ValueError("attention block output and statistics must share a device")
        if merged is not None and (
            output.shape != merged.output.shape or output.device != merged.output.device
        ):
            raise ValueError("attention blocks must have matching output shapes and devices")
        with torch.autocast(device_type=output.device.type, enabled=False):
            if merged is None:
                merged = AttentionBlockResult(output.clone(), maximum.clone(), total.clone())
                continue
            merged_maximum = torch.maximum(merged.maximum, maximum)
            # Empty rows otherwise evaluate exp(-inf - -inf).
            previous_factor = (
                (merged.maximum - merged_maximum)
                .exp()
                .masked_fill(merged.exponential_sum == 0, 0.0)
            )
            block_factor = (maximum - merged_maximum).exp().masked_fill(total == 0, 0.0)
            previous_mass = merged.exponential_sum * previous_factor
            block_mass = total * block_factor
            merged_total = previous_mass + block_mass
            denominator = merged_total.masked_fill(merged_total == 0, 1.0)
            merged = AttentionBlockResult(
                merged.output * (previous_mass / denominator) + output * (block_mass / denominator),
                merged_maximum,
                merged_total,
            )
    if merged is None:
        raise ValueError("at least one attention block is required")
    return merged


def _require_consensus_token(
    token: object,
    *,
    group_size: int | None = None,
    rank: int | None = None,
) -> ManifestConsensusToken:
    if type(token) is not ManifestConsensusToken:
        raise ManifestRefusal(
            ManifestRefusalCode.FORGED_CONSENSUS_TOKEN,
            "sequence collectives require an exact minted consensus token",
        )
    if (group_size is not None and token.group_size != group_size) or (
        rank is not None and token.rank != rank
    ):
        raise ManifestRefusal(
            ManifestRefusalCode.FORGED_CONSENSUS_TOKEN,
            "sequence consensus token does not match the collective plan",
        )
    return token


@dataclass(frozen=True, slots=True)
class SequenceExchangeEvent:
    backend: Literal["ulysses", "ring"]
    operation: Literal["head-to-sequence", "sequence-to-head", "ring-chunk"]
    step: int
    logical_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.backend) is not str or self.backend not in _EXCHANGE_BACKENDS:
            raise SequenceExchangeError("backend must be an exact supported str")
        if type(self.operation) is not str or self.operation not in _EXCHANGE_OPERATIONS:
            raise SequenceExchangeError("operation must be an exact supported str")
        if type(self.step) is not int or self.step < 0:
            raise SequenceExchangeError("step must be an exact int >= 0")
        if (
            type(self.logical_ranks) is not tuple
            or not self.logical_ranks
            or any(type(rank) is not int for rank in self.logical_ranks)
        ):
            raise SequenceExchangeError("logical_ranks must be a non-empty tuple of exact ints")


@dataclass(frozen=True, slots=True)
class SequenceExchangeOrderEvent:
    ordinal: int

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise SequenceExchangeError("event ordinal must be an exact int >= 0")

    def wait(self) -> None:
        return None


@runtime_checkable
class SequenceExchangeEventHandle(Protocol):
    @property
    def ordinal(self) -> int: ...

    def wait(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SequenceExchangeSubmission:
    slot: Literal["q", "k", "v"]
    chunk: int
    tensor: torch.Tensor
    ready: SequenceExchangeEventHandle

    def __post_init__(self) -> None:
        if type(self.slot) is not str or self.slot not in _SUBMISSION_SLOTS:
            raise SequenceExchangeError("submission slot must be an exact q, k, or v str")
        if type(self.chunk) is not int or self.chunk < 0:
            raise SequenceExchangeError("submission chunk must be an exact int >= 0")
        if type(self.tensor) is not torch.Tensor:
            raise SequenceExchangeError("submission tensor must be an exact torch.Tensor")
        if self.tensor.ndim != 4:
            raise SequenceExchangeError("submission tensor must have rank 4 [B, H, S, D]")
        if not callable(getattr(self.ready, "wait", None)):
            raise SequenceExchangeError("submission ready must implement the event handle")
        ordinal = getattr(self.ready, "ordinal", None)
        if type(ordinal) is not int or ordinal < 0:
            raise SequenceExchangeError("submission ready event ordinal must be an exact int >= 0")


@dataclass(frozen=True, slots=True)
class SequenceExchangeSubmissions:
    items: tuple[SequenceExchangeSubmission, ...]

    def __post_init__(self) -> None:
        if type(self.items) is not tuple or any(
            type(item) is not SequenceExchangeSubmission for item in self.items
        ):
            raise SequenceExchangeError("items must be a tuple of exact submissions")
        ordinals = tuple(item.ready.ordinal for item in self.items)
        if ordinals != tuple(sorted(ordinals)) or len(set(ordinals)) != len(ordinals):
            raise SequenceExchangeError("submission ready events must be strictly ordered")
        for slot in _SUBMISSION_SLOTS:
            chunks = tuple(item.chunk for item in self.items if item.slot == slot)
            if chunks != tuple(range(len(chunks))):
                raise SequenceExchangeError(
                    f"{slot} submissions must contain consecutive chunks from zero"
                )
            if not chunks:
                raise SequenceExchangeError(f"{slot} submissions must not be empty")

    @classmethod
    def from_tensors(
        cls,
        q: TensorChunks,
        k: TensorChunks,
        v: TensorChunks,
    ) -> SequenceExchangeSubmissions:
        items: list[SequenceExchangeSubmission] = []
        ordinal = 0
        by_slot = {
            slot: (value,) if isinstance(value, torch.Tensor) else tuple(value)
            for slot, value in (("q", q), ("k", k), ("v", v))
        }
        for chunk in range(max(map(len, by_slot.values()))):
            for slot in _SUBMISSION_SLOTS:
                chunks = by_slot[slot]
                if chunk >= len(chunks):
                    continue
                items.append(
                    SequenceExchangeSubmission(
                        slot,
                        chunk,
                        chunks[chunk],
                        SequenceExchangeOrderEvent(ordinal),
                    )
                )
                ordinal += 1
        return cls(tuple(items))


@dataclass(frozen=True, slots=True)
class SequenceExchangeCompletionEvent:
    slot: Literal["q", "k", "v"]
    chunk: int
    ready: SequenceExchangeEventHandle
    completed: SequenceExchangeEventHandle

    def __post_init__(self) -> None:
        if type(self.slot) is not str or self.slot not in _SUBMISSION_SLOTS:
            raise SequenceExchangeError("completion slot must be an exact q, k, or v str")
        if type(self.chunk) is not int or self.chunk < 0:
            raise SequenceExchangeError("completion chunk must be an exact int >= 0")
        if not callable(getattr(self.ready, "wait", None)):
            raise SequenceExchangeError("completion ready must implement the event handle")
        if not callable(getattr(self.completed, "wait", None)):
            raise SequenceExchangeError("completed must implement the event handle")
        ready_ordinal = getattr(self.ready, "ordinal", None)
        completed_ordinal = getattr(self.completed, "ordinal", None)
        if type(ready_ordinal) is not int or type(completed_ordinal) is not int:
            raise SequenceExchangeError("completion event ordinals must be exact ints")
        if ready_ordinal < 0 or completed_ordinal < 0:
            raise SequenceExchangeError("completion event ordinals must be >= 0")
        if completed_ordinal <= ready_ordinal:
            raise SequenceExchangeError("completion must follow submission readiness")


@dataclass(frozen=True, slots=True)
class SequenceExchangeResult:
    output: torch.Tensor
    completions: tuple[SequenceExchangeCompletionEvent, ...]

    def __post_init__(self) -> None:
        if type(self.output) is not torch.Tensor:
            raise SequenceExchangeError("output must be an exact torch.Tensor")
        if self.output.ndim != 4:
            raise SequenceExchangeError("output must have rank 4 [B, H, S, D]")
        if type(self.completions) is not tuple or any(
            type(event) is not SequenceExchangeCompletionEvent for event in self.completions
        ):
            raise SequenceExchangeError("completions must be a tuple of exact completion events")
        if not self.completions:
            raise SequenceExchangeError("completions must not be empty")
        pairs = tuple((event.slot, event.chunk) for event in self.completions)
        if len(set(pairs)) != len(pairs):
            raise SequenceExchangeError("completions must not contain duplicate slot chunks")
        for slot in _SUBMISSION_SLOTS:
            chunks = tuple(event.chunk for event in self.completions if event.slot == slot)
            if chunks != tuple(range(len(chunks))) or not chunks:
                raise SequenceExchangeError(
                    f"{slot} completions must contain consecutive chunks from zero"
                )


@dataclass(frozen=True, slots=True)
class SequenceExchangeHooks:
    """Synchronous rank-local hooks around exchange events.

    A raising hook is a rank-fatal error; peer ranks surface it as a
    process-group timeout, like other rank-local failures inside collective
    code. Hook success is not coordinated across ranks. Per-chunk readiness
    and completion ordering is carried by the submission result records.
    """

    before: Callable[[SequenceExchangeEvent], None] | None = None
    after: Callable[[SequenceExchangeEvent], None] | None = None

    def __post_init__(self) -> None:
        for site, hook in (("before", self.before), ("after", self.after)):
            if hook is None:
                continue
            if not callable(hook):
                raise SequenceExchangeError(f"{site} hook must be callable or None")
            if inspect.iscoroutinefunction(hook) or inspect.iscoroutinefunction(
                type(hook).__call__
            ):
                raise SequenceExchangeError(f"{site} hook must be synchronous")


@dataclass(frozen=True, slots=True)
class RingAttentionPlan:
    """Receipt-ready facts for a float32 multi-rank Ring merge."""

    merge_algorithm: Literal["ring-lse-fp32.v1"]
    traversal_order: tuple[int, ...]
    traversal_contract: Literal["ascending-ring-step.v1"]
    accumulation_order: Literal["sequential in ring-step order"]
    accumulation_dtype: Literal["float32"]

    def __post_init__(self) -> None:
        for name, value, expected in (
            ("merge_algorithm", self.merge_algorithm, RING_MERGE_ALGORITHM),
            ("traversal_contract", self.traversal_contract, RING_TRAVERSAL_ORDER),
            ("accumulation_order", self.accumulation_order, RING_ACCUMULATION_ORDER),
            ("accumulation_dtype", self.accumulation_dtype, RING_ACCUMULATION_DTYPE),
        ):
            if type(value) is not str or value != expected:
                raise SequenceExchangeError(f"{name} must equal {expected!r}")
        if (
            type(self.traversal_order) is not tuple
            or not self.traversal_order
            or any(type(rank) is not int for rank in self.traversal_order)
        ):
            raise SequenceExchangeError("traversal_order must be a non-empty tuple of exact ints")
        if len(set(self.traversal_order)) != len(self.traversal_order):
            raise SequenceExchangeError("traversal_order must not contain duplicate ranks")


TensorChunks: TypeAlias = torch.Tensor | Iterable[torch.Tensor]


class AttentionExchangeBackend(Protocol):
    """Compute distributed attention for one local padded sequence shard.

    Each tensor may arrive as one ready tensor or as independently produced
    sequence chunks. Current backends materialize chunks synchronously.
    Multi-rank Ring exchange accepts float32 inputs only; Ulysses-only exchange
    preserves the input floating dtype.
    """

    def attend(
        self,
        q: TensorChunks,
        k: TensorChunks,
        v: TensorChunks,
        layout: SequenceLayout,
        kernel: AttentionKernel,
        *,
        hooks: SequenceExchangeHooks | None = None,
    ) -> torch.Tensor: ...

    def attend_submissions(
        self,
        submissions: SequenceExchangeSubmissions,
        layout: SequenceLayout,
        kernel: AttentionKernel,
        *,
        hooks: SequenceExchangeHooks | None = None,
        scale: float | None = None,
    ) -> SequenceExchangeResult: ...

    def gather_sequence(
        self,
        hidden: torch.Tensor,
        partition: SequencePartition,
        shard: SequenceShard,
    ) -> torch.Tensor: ...


def _ulysses_transport() -> str:
    """Transport for the Ulysses all_to_all: NCCL collectives (default) or
    direct copy-engine transfers into CUDA-IPC-shared buffers. Both move the
    same bytes into the same layout, so outputs are bitwise identical."""
    value = os.environ.get("DINKSTER_SINGLE_JOB_SEQUENCE_TRANSPORT", "nccl")
    if value not in ("nccl", "peer-copy"):
        raise SequenceExchangeError(
            "DINKSTER_SINGLE_JOB_SEQUENCE_TRANSPORT must be 'nccl' or 'peer-copy'"
        )
    return value


def _default_pg_timeout() -> timedelta | None:
    """The timeout the default process group was initialized with.

    torch exposes no public read-back, so probe the backend options. Returns
    None when no backend reveals one; new_group then falls back to its own
    backend-specific default, preserving pre-existing semantics.
    """
    group = dist.group.WORLD
    if group is not None:
        for device_type in ("cpu", "cuda"):
            try:
                backend = group._get_backend(  # pyright: ignore[reportPrivateUsage]
                    torch.device(device_type)
                )
            except RuntimeError:
                continue
            options = getattr(backend, "options", None)
            timeout = getattr(options, "_timeout", None)
            if isinstance(timeout, timedelta):
                return timeout
    return None


@dataclass(frozen=True, slots=True)
class _AxisGroup:
    logical_ranks: tuple[int, ...]
    physical_ranks: tuple[int, ...]
    process_ranks: tuple[int, ...]
    process_group: dist.ProcessGroup | None

    @property
    def logical_index(self) -> int:
        return self.physical_ranks.index(dist.get_rank())


@dataclass(frozen=True, slots=True)
class _SequenceExchangeGroups:
    ulysses: _AxisGroup
    ring: _AxisGroup
    sequence: _AxisGroup
    #: Persistent copy-engine transport for the Ulysses all_to_all, built
    #: once per group creation so its CUDA-IPC slots survive across
    #: attention calls. None under the default NCCL transport.
    ulysses_peer_copy: _PeerCopyAllToAll | None

    @classmethod
    def create(cls, mesh: UspMesh, placement: PlacementMap) -> _SequenceExchangeGroups:
        if not dist.is_available() or not dist.is_initialized():
            raise SequenceExchangeError("sequence exchange requires an initialized process group")
        if dist.get_world_size() != mesh.world_size:
            raise SequenceExchangeError("process-group world size must equal the mesh world size")
        logical_rank = placement.logical_rank(dist.get_rank())
        # new_group does not inherit the default group's timeout; without an
        # explicit one the axis groups get torch's backend default (30 minutes
        # for Gloo), so a rank-fatal peer failure stalls surviving ranks for
        # 30 minutes on transports with no peer-reset detection (Gloo on
        # macOS, #714). A None timeout keeps new_group's backend default.
        timeout = _default_pg_timeout()
        selected: dict[str, _AxisGroup] = {}
        for name, groups in (
            ("ring", mesh.ring_groups()),
            ("ulysses", mesh.ulysses_groups()),
            ("sequence", mesh.sequence_groups()),
        ):
            if not groups:
                groups = tuple((rank,) for rank in mesh.ranks)
            for logical_ranks in groups:
                physical_ranks = placement.apply(logical_ranks)
                process_ranks = tuple(sorted(physical_ranks))
                process_group = None
                if len(logical_ranks) > 1:
                    process_group = dist.new_group(ranks=list(physical_ranks), timeout=timeout)
                if logical_rank in logical_ranks:
                    if len(logical_ranks) == 1:
                        joined_group = None
                    elif isinstance(process_group, dist.ProcessGroup):
                        joined_group = process_group
                    else:
                        raise SequenceExchangeError("rank did not join its sequence axis group")
                    selected[name] = _AxisGroup(
                        logical_ranks, physical_ranks, process_ranks, joined_group
                    )
        transport = _ulysses_transport()
        if dist.get_world_size() > 1:
            transports: list[str | None] = [None] * dist.get_world_size()
            dist.all_gather_object(transports, transport)
            if len(set(transports)) != 1:
                raise SequenceExchangeError(
                    "sequence transport must be identical on every rank; observed "
                    f"{sorted({str(item) for item in transports})}"
                )
        ulysses_peer_copy = None
        if transport == "peer-copy" and len(selected["ulysses"].logical_ranks) > 1:
            ulysses_peer_copy = _PeerCopyAllToAll(selected["ulysses"])
        return cls(
            ulysses=selected["ulysses"],
            ring=selected["ring"],
            sequence=selected["sequence"],
            ulysses_peer_copy=ulysses_peer_copy,
        )


def _gather_sequence_rows(
    hidden: torch.Tensor,
    partition: SequencePartition,
    shard: SequenceShard,
    groups: _SequenceExchangeGroups,
    placement: PlacementMap,
    consensus_token: object,
) -> torch.Tensor:
    canonical = plan_sequence_partition(partition.sequence_length, partition.shard_count)
    if partition != canonical or shard not in canonical.shards:
        raise SequenceExchangeError("sequence gather requires a canonical local shard")
    if type(hidden) is not torch.Tensor or hidden.ndim != 3:
        raise SequenceExchangeError("sequence gather input must be [B, S, D]")
    local_rows = shard.valid_rows + shard.padded_rows
    if hidden.shape[1] != local_rows:
        raise SequenceExchangeError("sequence gather input does not match the padded shard")
    group = groups.sequence
    if group.logical_ranks.index(placement.logical_rank(dist.get_rank())) != shard.index:
        raise SequenceExchangeError("sequence gather shard does not match the local rank")
    if len(group.logical_ranks) == 1:
        gathered = [hidden]
    else:
        _require_consensus_token(
            consensus_token,
            group_size=partition.shard_count,
            rank=shard.index,
        )
        assert group.process_group is not None
        gathered = [torch.empty_like(hidden) for _ in group.process_ranks]
        dist.all_gather(gathered, hidden, group=group.process_group)
    by_rank = dict(zip(group.process_ranks, gathered, strict=True))
    return torch.cat(
        tuple(
            by_rank[placement.physical_rank(logical_rank)][:, : canonical.shards[index].valid_rows]
            for index, logical_rank in enumerate(group.logical_ranks)
        ),
        dim=1,
    )


def _materialize_chunks(value: TensorChunks, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    parts = tuple(value)
    if not parts:
        raise SequenceExchangeError(f"{name} chunks must not be empty")
    first = parts[0]
    if any(
        part.ndim != first.ndim
        or part.shape[:2] != first.shape[:2]
        or part.shape[3:] != first.shape[3:]
        or part.dtype != first.dtype
        or part.device != first.device
        for part in parts[1:]
    ):
        raise SequenceExchangeError(f"{name} chunks must agree outside the sequence axis")
    return torch.cat(parts, dim=2)


def _materialize_submissions(
    submissions: SequenceExchangeSubmissions,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if type(submissions) is not SequenceExchangeSubmissions:
        raise SequenceExchangeError("submissions must be an exact SequenceExchangeSubmissions")
    chunks: dict[str, list[torch.Tensor]] = {slot: [] for slot in _SUBMISSION_SLOTS}
    for item in submissions.items:
        try:
            item.ready.wait()
        except Exception as error:
            raise SequenceExchangeError(
                f"submission readiness failed for {item.slot} chunk {item.chunk}"
            ) from error
        chunks[item.slot].append(item.tensor)
    values = {slot: _materialize_chunks(tuple(chunks[slot]), slot) for slot in _SUBMISSION_SLOTS}
    return values["q"], values["k"], values["v"]


def _completion_events(
    submissions: SequenceExchangeSubmissions,
) -> tuple[SequenceExchangeCompletionEvent, ...]:
    start = max(item.ready.ordinal for item in submissions.items) + 1
    return tuple(
        SequenceExchangeCompletionEvent(
            item.slot,
            item.chunk,
            item.ready,
            SequenceExchangeOrderEvent(start),
        )
        for item in submissions.items
    )


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: SequenceLayout,
    mesh: UspMesh,
    placement: PlacementMap,
) -> None:
    if type(layout) is not SequenceLayout:
        raise SequenceExchangeError("layout must be an exact SequenceLayout")
    if layout.mesh_identity != mesh.digest:
        raise SequenceExchangeError("layout mesh identity does not match the exchange mesh")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.ndim != 4:
            raise SequenceExchangeError(f"{name} must have rank 4 [B, H, S, D]")
        if not tensor.is_floating_point():
            raise SequenceExchangeError(f"{name} must use a floating-point dtype")
    if q.shape != k.shape or q.shape != v.shape:
        raise SequenceExchangeError("q, k, and v must have identical local shapes")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.device != k.device or q.device != v.device:
        raise SequenceExchangeError("q, k, and v must share one dtype and device")
    if mesh.ring > 1 and q.dtype is not torch.float32:
        raise SequenceExchangeError("multi-rank Ring exchange requires float32 q, k, and v")
    if q.shape[1] != layout.head_count:
        raise SequenceExchangeError("input head count does not match the sequence layout")
    local_rows = layout.shard.valid_rows + layout.shard.padded_rows
    if q.shape[2] != local_rows:
        raise SequenceExchangeError("input sequence rows do not match the padded shard")
    if layout.global_sequence_length < mesh.ulysses * mesh.ring:
        raise SequenceExchangeError("global sequence length is shorter than the sequence mesh")
    logical_rank = placement.logical_rank(dist.get_rank())
    sequence_groups = mesh.sequence_groups() or tuple((rank,) for rank in mesh.ranks)
    sequence_group = next(group for group in sequence_groups if logical_rank in group)
    if layout.shard.index != sequence_group.index(logical_rank):
        raise SequenceExchangeError("layout shard index does not match the mesh sequence rank")
    expected = plan_sequence_partition(layout.global_sequence_length, mesh.ulysses * mesh.ring)
    if layout.shard != expected.shards[layout.shard.index]:
        raise SequenceExchangeError("layout shard does not match the global sequence partition")
    if layout.head_count % mesh.ulysses:
        raise SequenceExchangeError("head count must be divisible by the Ulysses axis degree")
    heads_per_rank = layout.head_count // mesh.ulysses
    coordinate = mesh.coordinates(logical_rank)
    expected_head_start = coordinate.ulysses * heads_per_rank
    if (layout.head_start, layout.head_stop) != (
        expected_head_start,
        expected_head_start + heads_per_rank,
    ):
        raise SequenceExchangeError("layout head ownership does not match the Ulysses coordinate")


def _emit(
    callback: Callable[[SequenceExchangeEvent], None] | None,
    event: SequenceExchangeEvent,
    site: Literal["before", "after"],
) -> None:
    if callback is not None:
        try:
            callback(event)
        except Exception as error:
            raise SequenceExchangeError(
                f"{site} hook failed for operation {event.operation!r} at step {event.step}"
            ) from error


def _key_mask(valid: torch.Tensor) -> torch.Tensor | None:
    if bool(valid.all().item()):
        return None
    return valid.view(1, 1, 1, -1)


def _zero_padded_queries(output: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if bool(valid.all().item()):
        return output
    return output.masked_fill(~valid.view(1, 1, -1, 1), 0.0)


def _attend_valid_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid: torch.Tensor,
    kernel: AttentionKernel,
    scale: float | None,
) -> torch.Tensor:
    if bool(valid.all().item()):
        return kernel(q, k, v, mask=None, scale=scale)
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    compact_output = kernel(
        q.index_select(2, valid_indices),
        k.index_select(2, valid_indices),
        v.index_select(2, valid_indices),
        mask=None,
        scale=scale,
    )
    return torch.zeros_like(q).index_copy(2, valid_indices, compact_output)


def _synchronize_local_cuda_work() -> None:
    """Wait for every kernel and copy this process has submitted, on every
    visible device. Device synchronization is per-process: it never waits on
    other ranks' work, so cross-rank ordering still needs a barrier."""
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)


class _PeerCopyAllToAll:
    """all_to_all over direct GPU-to-GPU copies into CUDA-IPC-shared slots.

    Bitwise-inert alternative to the NCCL all_to_all on the Ulysses axis:
    each rank owns one persistent receive slot per source rank, shared once
    via CUDA IPC, and senders write their pieces straight into the
    destination slots with copy-engine transfers. No NCCL SendRecv kernels
    occupy SMs during the exchange. Requires every rank of the group to
    drive a distinct visible CUDA device (see ensure_process_group).

    Ordering contract per exchange: local synchronize + group barrier before
    the writes (so nobody overwrites a slot a peer is still reading from the
    previous exchange), then local synchronize + group barrier after (so
    every write has landed before any rank reads). Returned slots are only
    valid until the next exchange on this instance.
    """

    def __init__(self, group: _AxisGroup) -> None:
        if not torch.cuda.is_available():
            raise SequenceExchangeError("peer-copy transport requires CUDA")
        if group.process_group is None:
            raise SequenceExchangeError("peer-copy transport requires a process group")
        degree = len(group.process_ranks)
        if torch.cuda.device_count() < degree:
            raise SequenceExchangeError(
                "peer-copy transport requires at least one visible CUDA device per rank"
            )
        self.group = group
        self.process_index = group.process_ranks.index(dist.get_rank())
        self._slots_by_key: dict[
            tuple[tuple[int, ...], torch.dtype],
            tuple[list[torch.Tensor], dict[int, torch.Tensor]],
        ] = {}

    def _slots(self, template: torch.Tensor) -> tuple[list[torch.Tensor], dict[int, torch.Tensor]]:
        if not template.is_cuda:
            raise SequenceExchangeError("peer-copy transport requires CUDA tensors")
        key = (tuple(template.shape), template.dtype)
        cached = self._slots_by_key.get(key)
        if cached is not None:
            return cached
        from torch.multiprocessing.reductions import reduce_tensor

        degree = len(self.group.process_ranks)
        mine = [torch.empty_like(template) for _ in range(degree)]
        # The self slot is filled with a local copy, so no peer ever rebuilds
        # it; exporting it would leak one CUDA IPC reference per slot key.
        handles: list[object] = [
            None if index == self.process_index else reduce_tensor(mine[index])
            for index in range(degree)
        ]
        payload = (template.device.index, handles)
        gathered: list[object] = [None] * degree
        dist.all_gather_object(gathered, payload, group=self.group.process_group)
        devices = [item[0] for item in gathered if isinstance(item, tuple)]
        if len(devices) != degree or len(set(devices)) != degree:
            raise SequenceExchangeError(
                "peer-copy transport requires every rank to drive a distinct "
                f"CUDA device; observed device indices {devices}"
            )
        remote: dict[int, torch.Tensor] = {}
        for index, item in enumerate(gathered):
            if index == self.process_index:
                continue
            assert isinstance(item, tuple)
            peer_handles = item[1]
            if not isinstance(peer_handles, list) or len(peer_handles) != degree:
                raise SequenceExchangeError("peer-copy slot exchange returned malformed handles")
            rebuild, args = peer_handles[self.process_index]
            remote[index] = rebuild(*args)
        entry = (mine, remote)
        self._slots_by_key[key] = entry
        return entry

    def all_to_all(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        mine, remote = self._slots(inputs[0])
        group = self.group.process_group
        assert group is not None
        _synchronize_local_cuda_work()
        dist.barrier(group=group)
        for index, piece in enumerate(inputs):
            if index == self.process_index:
                mine[index].copy_(piece, non_blocking=True)
            else:
                remote[index].copy_(piece, non_blocking=True)
        _synchronize_local_cuda_work()
        dist.barrier(group=group)
        return list(mine)


class _UlyssesExchange:
    def __init__(
        self,
        mesh: UspMesh,
        groups: _SequenceExchangeGroups,
        consensus_token: object,
    ) -> None:
        self.mesh = mesh
        self.group = groups.ulysses
        self.consensus_token = consensus_token
        self._peer_copy = groups.ulysses_peer_copy

    def _all_to_all(
        self,
        tensors_by_logical_rank: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        if len(self.group.logical_ranks) == 1:
            return tensors_by_logical_rank
        _require_consensus_token(self.consensus_token)
        assert self.group.process_group is not None
        logical_by_rank = dict(zip(self.group.physical_ranks, tensors_by_logical_rank, strict=True))
        inputs = [logical_by_rank[rank].contiguous() for rank in self.group.process_ranks]
        if self._peer_copy is not None:
            outputs = self._peer_copy.all_to_all(inputs)
        else:
            outputs = [torch.empty_like(inputs[0]) for _ in self.group.process_ranks]
            dist.all_to_all(outputs, inputs, group=self.group.process_group)
        process_by_rank = dict(zip(self.group.process_ranks, outputs, strict=True))
        return tuple(process_by_rank[rank] for rank in self.group.physical_ranks)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hooks: SequenceExchangeHooks,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        degree = len(self.group.logical_ranks)
        if degree == 1:
            return q, k, v
        event = SequenceExchangeEvent("ulysses", "head-to-sequence", 0, self.group.logical_ranks)
        _emit(hooks.before, event, "before")
        exchanged: list[torch.Tensor] = []
        for tensor in (q, k, v):
            head_chunks = tuple(tensor.chunk(degree, dim=1))
            received = self._all_to_all(head_chunks)
            exchanged.append(torch.cat(received, dim=2))
        _emit(hooks.after, event, "after")
        return exchanged[0], exchanged[1], exchanged[2]

    def reverse(self, output: torch.Tensor, hooks: SequenceExchangeHooks) -> torch.Tensor:
        degree = len(self.group.logical_ranks)
        if degree == 1:
            return output
        event = SequenceExchangeEvent("ulysses", "sequence-to-head", 0, self.group.logical_ranks)
        _emit(hooks.before, event, "before")
        sequence_chunks = tuple(output.chunk(degree, dim=2))
        received = self._all_to_all(sequence_chunks)
        restored = torch.cat(received, dim=1)
        _emit(hooks.after, event, "after")
        return restored


class _RingExchange:
    def __init__(
        self,
        mesh: UspMesh,
        groups: _SequenceExchangeGroups,
        consensus_token: object,
    ) -> None:
        self.mesh = mesh
        self.group = groups.ring
        self.consensus_token = consensus_token

    @property
    def plan(self) -> RingAttentionPlan:
        index = self.group.logical_index
        ranks = self.group.logical_ranks
        traversal = tuple(ranks[(index - step) % len(ranks)] for step in range(len(ranks)))
        return RingAttentionPlan(
            RING_MERGE_ALGORITHM,
            traversal,
            RING_TRAVERSAL_ORDER,
            RING_ACCUMULATION_ORDER,
            RING_ACCUMULATION_DTYPE,
        )

    def _block_validity(
        self, layout: SequenceLayout, ring_index: int, device: torch.device
    ) -> torch.Tensor:
        partition = plan_sequence_partition(
            layout.global_sequence_length, self.mesh.ulysses * self.mesh.ring
        )
        values: list[bool] = []
        for ulysses_index in range(self.mesh.ulysses):
            shard = partition.shards[ulysses_index * self.mesh.ring + ring_index]
            values.extend((True,) * shard.valid_rows)
            values.extend((False,) * shard.padded_rows)
        return torch.tensor(values, dtype=torch.bool, device=device)

    def _rotate(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _require_consensus_token(self.consensus_token)
        index = self.group.logical_index
        previous_rank = self.group.physical_ranks[(index - 1) % len(self.group.physical_ranks)]
        next_rank = self.group.physical_ranks[(index + 1) % len(self.group.physical_ranks)]
        next_k = torch.empty_like(k)
        next_v = torch.empty_like(v)
        assert self.group.process_group is not None
        operations = (
            dist.P2POp(dist.irecv, next_k, previous_rank, self.group.process_group, 37001),
            dist.P2POp(dist.irecv, next_v, previous_rank, self.group.process_group, 37002),
            dist.P2POp(dist.isend, k.contiguous(), next_rank, self.group.process_group, 37001),
            dist.P2POp(dist.isend, v.contiguous(), next_rank, self.group.process_group, 37002),
        )
        for work in dist.batch_isend_irecv(list(operations)):
            work.wait()
        return next_k, next_v

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layout: SequenceLayout,
        kernel: AttentionKernel,
        hooks: SequenceExchangeHooks,
        scale: float | None,
    ) -> torch.Tensor:
        plan = self.plan
        if len(plan.traversal_order) > 1:
            _require_consensus_token(self.consensus_token)
        query_valid = self._block_validity(layout, self.group.logical_index, q.device)
        if len(plan.traversal_order) == 1:
            event = SequenceExchangeEvent("ring", "ring-chunk", 0, self.group.logical_ranks)
            _emit(hooks.before, event, "before")
            output = kernel(q, k, v, mask=_key_mask(query_valid), scale=scale)
            _emit(hooks.after, event, "after")
            return _zero_padded_queries(output, query_valid)

        running_max = torch.full(
            (*q.shape[:3], 1), -torch.inf, dtype=torch.float32, device=q.device
        )
        running_sum = torch.zeros_like(running_max)
        accumulator = torch.zeros_like(q, dtype=torch.float32)
        current_k = k
        current_v = v
        score_scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        for step, owner_rank in enumerate(plan.traversal_order):
            event = SequenceExchangeEvent("ring", "ring-chunk", step, self.group.logical_ranks)
            _emit(hooks.before, event, "before")
            owner_index = self.group.logical_ranks.index(owner_rank)
            key_valid = self._block_validity(layout, owner_index, q.device)
            mask = _key_mask(key_valid)
            chunk_output = kernel(q, current_k, current_v, mask=mask, scale=scale)
            scores = torch.matmul(q.float(), current_k.float().transpose(-2, -1)) * score_scale
            if mask is not None:
                scores = scores.masked_fill(~mask, -torch.inf)
            block_max = scores.amax(dim=-1, keepdim=True)
            block_sum = torch.exp(scores - block_max).sum(dim=-1, keepdim=True)
            merged_max = torch.maximum(running_max, block_max)
            running_factor = torch.exp(running_max - merged_max)
            block_factor = torch.exp(block_max - merged_max)
            accumulator = (
                accumulator * running_factor + chunk_output.float() * block_sum * block_factor
            )
            running_sum = running_sum * running_factor + block_sum * block_factor
            running_max = merged_max
            if step + 1 < len(plan.traversal_order):
                current_k, current_v = self._rotate(current_k, current_v)
            _emit(hooks.after, event, "after")
        output = (accumulator / running_sum).to(q.dtype)
        return _zero_padded_queries(output, query_valid)


class _BaseSequenceExchange:
    def __init__(
        self,
        mesh: UspMesh,
        placement: PlacementMap | None = None,
        *,
        consensus_token: object = None,
    ) -> None:
        if type(mesh) is not UspMesh:
            raise SequenceExchangeError("mesh must be an exact UspMesh")
        active_placement = (
            PlacementMap.identity(mesh.process_mesh) if placement is None else placement
        )
        if type(active_placement) is not PlacementMap:
            raise SequenceExchangeError("placement must be an exact PlacementMap or None")
        if active_placement.mesh_digest != mesh.digest:
            raise SequenceExchangeError("placement mesh digest does not match the exchange mesh")
        if len(active_placement.physical_ranks) != mesh.world_size:
            raise SequenceExchangeError("placement rank count must equal the mesh world size")
        self.mesh = mesh
        self.placement = active_placement
        self.consensus_token = consensus_token
        self._groups: _SequenceExchangeGroups | None = None

    def gather_sequence(
        self,
        hidden: torch.Tensor,
        partition: SequencePartition,
        shard: SequenceShard,
    ) -> torch.Tensor:
        if self._groups is None:
            raise SequenceExchangeError(
                "sequence gather is available after the first attention call"
            )
        return _gather_sequence_rows(
            hidden,
            partition,
            shard,
            self._groups,
            self.placement,
            self.consensus_token,
        )

    def _prepare_submissions(
        self,
        submissions: SequenceExchangeSubmissions,
        layout: SequenceLayout,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        _SequenceExchangeGroups,
    ]:
        ready = _materialize_submissions(submissions)
        _validate_inputs(*ready, layout, self.mesh, self.placement)
        if self.mesh.ulysses * self.mesh.ring > 1:
            _require_consensus_token(
                self.consensus_token,
                group_size=self.mesh.ulysses * self.mesh.ring,
                rank=layout.shard.index,
            )
        if self._groups is None:
            self._groups = _SequenceExchangeGroups.create(self.mesh, self.placement)
        return *ready, self._groups


class UlyssesSequenceExchange(_BaseSequenceExchange):
    def attend(
        self,
        q: TensorChunks,
        k: TensorChunks,
        v: TensorChunks,
        layout: SequenceLayout,
        kernel: AttentionKernel,
        *,
        hooks: SequenceExchangeHooks | None = None,
    ) -> torch.Tensor:
        return self.attend_submissions(
            SequenceExchangeSubmissions.from_tensors(q, k, v),
            layout,
            kernel,
            hooks=hooks,
        ).output

    def attend_submissions(
        self,
        submissions: SequenceExchangeSubmissions,
        layout: SequenceLayout,
        kernel: AttentionKernel,
        *,
        hooks: SequenceExchangeHooks | None = None,
        scale: float | None = None,
    ) -> SequenceExchangeResult:
        if self.mesh.ring != 1:
            raise SequenceExchangeError("standalone Ulysses exchange requires ring degree 1")
        ready_q, ready_k, ready_v, groups = self._prepare_submissions(submissions, layout)
        active_hooks = hooks or SequenceExchangeHooks()
        exchange = _UlyssesExchange(self.mesh, groups, self.consensus_token)
        full_q, full_k, full_v = exchange.forward(ready_q, ready_k, ready_v, active_hooks)
        partition = plan_sequence_partition(layout.global_sequence_length, self.mesh.ulysses)
        valid_values: list[bool] = []
        for shard in partition.shards:
            valid_values.extend((True,) * shard.valid_rows)
            valid_values.extend((False,) * shard.padded_rows)
        valid = torch.tensor(valid_values, dtype=torch.bool, device=full_k.device)
        local_valid = torch.ones(ready_q.shape[2], dtype=torch.bool, device=ready_q.device)
        if layout.shard.padded_rows:
            local_valid[-layout.shard.padded_rows :] = False
        output = _attend_valid_rows(full_q, full_k, full_v, valid, kernel, scale)
        return SequenceExchangeResult(
            _zero_padded_queries(exchange.reverse(output, active_hooks), local_valid),
            _completion_events(submissions),
        )


class RingSequenceExchange(_BaseSequenceExchange):
    def attend(
        self,
        q: TensorChunks,
        k: TensorChunks,
        v: TensorChunks,
        layout: SequenceLayout,
        kernel: AttentionKernel,
        *,
        hooks: SequenceExchangeHooks | None = None,
    ) -> torch.Tensor:
        return self.attend_submissions(
            SequenceExchangeSubmissions.from_tensors(q, k, v),
            layout,
            kernel,
            hooks=hooks,
        ).output

    def attend_submissions(
        self,
        submissions: SequenceExchangeSubmissions,
        layout: SequenceLayout,
        kernel: AttentionKernel,
        *,
        hooks: SequenceExchangeHooks | None = None,
        scale: float | None = None,
    ) -> SequenceExchangeResult:
        if self.mesh.ulysses != 1:
            raise SequenceExchangeError("standalone Ring exchange requires Ulysses degree 1")
        ready_q, ready_k, ready_v, groups = self._prepare_submissions(submissions, layout)
        exchange = _RingExchange(self.mesh, groups, self.consensus_token)
        return SequenceExchangeResult(
            exchange.attend(
                ready_q,
                ready_k,
                ready_v,
                layout,
                kernel,
                hooks or SequenceExchangeHooks(),
                scale,
            ),
            _completion_events(submissions),
        )

    @property
    def plan(self) -> RingAttentionPlan:
        if self._groups is None:
            raise SequenceExchangeError("ring plan is available after the first attention call")
        return _RingExchange(
            self.mesh,
            self._groups,
            self.consensus_token,
        ).plan


class UspSequenceExchange(_BaseSequenceExchange):
    def attend(
        self,
        q: TensorChunks,
        k: TensorChunks,
        v: TensorChunks,
        layout: SequenceLayout,
        kernel: AttentionKernel,
        *,
        hooks: SequenceExchangeHooks | None = None,
    ) -> torch.Tensor:
        return self.attend_submissions(
            SequenceExchangeSubmissions.from_tensors(q, k, v),
            layout,
            kernel,
            hooks=hooks,
        ).output

    def attend_submissions(
        self,
        submissions: SequenceExchangeSubmissions,
        layout: SequenceLayout,
        kernel: AttentionKernel,
        *,
        hooks: SequenceExchangeHooks | None = None,
        scale: float | None = None,
    ) -> SequenceExchangeResult:
        ready_q, ready_k, ready_v, groups = self._prepare_submissions(submissions, layout)
        active_hooks = hooks or SequenceExchangeHooks()
        ulysses = _UlyssesExchange(self.mesh, groups, self.consensus_token)
        ring = _RingExchange(self.mesh, groups, self.consensus_token)
        head_q, head_k, head_v = ulysses.forward(ready_q, ready_k, ready_v, active_hooks)
        head_output = ring.attend(head_q, head_k, head_v, layout, kernel, active_hooks, scale)
        output = ulysses.reverse(head_output, active_hooks)
        local_valid = torch.ones(output.shape[2], dtype=torch.bool, device=output.device)
        if layout.shard.padded_rows:
            local_valid[-layout.shard.padded_rows :] = False
        return SequenceExchangeResult(
            _zero_padded_queries(output, local_valid), _completion_events(submissions)
        )

    @property
    def ring_plan(self) -> RingAttentionPlan:
        if self._groups is None:
            raise SequenceExchangeError("ring plan is available after the first attention call")
        return _RingExchange(
            self.mesh,
            self._groups,
            self.consensus_token,
        ).plan


__all__ = [
    "AttentionExchangeBackend",
    "DenseRingBlockAttention",
    "RING_ACCUMULATION_DTYPE",
    "RING_ACCUMULATION_ORDER",
    "RING_MERGE_ALGORITHM",
    "RING_TRAVERSAL_ORDER",
    "RingAttentionPlan",
    "RingSequenceExchange",
    "SequenceExchangeError",
    "SequenceExchangeEventHandle",
    "SequenceExchangeCompletionEvent",
    "SequenceExchangeEvent",
    "SequenceExchangeHooks",
    "SequenceExchangeOrderEvent",
    "SequenceExchangeResult",
    "SequenceExchangeSubmission",
    "SequenceExchangeSubmissions",
    "TensorChunks",
    "UlyssesSequenceExchange",
    "UspSequenceExchange",
    "dense_ring_block_statistics",
    "merge_attention_blocks",
]
