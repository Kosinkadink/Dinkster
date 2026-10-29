"""Fixed-rank NCCL data plane for single-job sampling."""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TypeVar, cast

import torch
from dinkster_inference import (
    ConsensusTransport,
    GuidanceEvaluationPlan,
    GuidanceEvaluationRequest,
    GuidancePrediction,
    GuidancePredictions,
    GuidancePredictionSource,
    RankWeights,
    SamplingCancelled,
    plan_guidance_lanes,
    use_additional_sampling_cancellation,
)

_Result = TypeVar("_Result")
_SEQUENCE_DEVICE_IDENTITY_LIMIT = 1024
_RANK_ZERO_HEARTBEAT_SECONDS = 0.25
_RANK_ZERO_CONTROL_TIMEOUT_SECONDS = 5
_LOGGER = logging.getLogger(__name__)

_GUIDANCE_RECEIPT_DOMAIN = "domain=dinkster.distributed.guidance-receipt.v2"
_SEQUENCE_RECEIPT_DOMAIN = "domain=dinkster.distributed.sequence-receipt.v2"


@dataclass(frozen=True)
class DistributedSamplingConfig:
    rank: int
    world_size: int
    mode: str
    rendezvous: str
    token: str
    attempt: str = "direct"
    sequence_ulysses: int = 1
    sequence_ring: int = 1
    sequence_guidance: int = 1
    #: Ulysses all_to_all transport: "nccl" (default) or "peer-copy".
    #: peer-copy moves pieces with direct cross-device copies over CUDA
    #: IPC buffers instead of NCCL SendRecv kernels; it requires every
    #: rank's process to see all sequence-group devices, with rank r
    #: driving device index r. Bitwise inert: both transports deliver
    #: identical bytes.
    sequence_transport: str = "nccl"


class SequenceDigestConsensusTransport(ConsensusTransport):
    """Digest-only consensus over the sequence sideband process group."""

    def __init__(self, config: DistributedSamplingConfig, device: torch.device) -> None:
        if type(config) is not DistributedSamplingConfig or config.mode != "sequence":
            raise ValueError("sequence consensus requires an exact sequence configuration")
        if type(device) is not torch.device:
            raise TypeError("sequence consensus device must be an exact torch.device")
        if config.sequence_guidance == 1:
            self._ranks = tuple(range(config.world_size))
            self._group: object | None = None
        else:
            self._ranks, self._group = _sequence_fence_group(config)
        self._rank = self._ranks.index(config.rank)
        self._device = device

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def physical_ranks(self) -> tuple[int, ...]:
        return self._ranks

    def exchange_digest(self, rank: int, digest: str) -> tuple[str, ...]:
        if type(rank) is not int or rank != self._rank:
            raise ValueError("consensus rank does not match the sequence sideband rank")
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("consensus digest must be lowercase sha256 hex")
        local = torch.tensor(tuple(bytes.fromhex(digest)), dtype=torch.uint8, device=self._device)
        peers = [torch.empty_like(local) for _ in self._ranks]
        torch.distributed.all_gather(peers, local, group=self._group)
        return tuple(bytes(peer.tolist()).hex() for peer in peers)


def gather_sequence_device_bindings(
    config: DistributedSamplingConfig,
    device: torch.device,
    local_identity: str,
) -> tuple[str, ...]:
    """Gather physical device identities over the existing sequence sideband."""
    if type(config) is not DistributedSamplingConfig or config.mode != "sequence":
        raise ValueError("sequence device bindings require an exact sequence configuration")
    if type(device) is not torch.device:
        raise TypeError("sequence binding device must be an exact torch.device")
    encoded: bytes | None = None
    if type(local_identity) is str and local_identity and "\n" not in local_identity:
        try:
            candidate = local_identity.encode("utf-8")
        except UnicodeEncodeError:
            pass
        else:
            if len(candidate) <= _SEQUENCE_DEVICE_IDENTITY_LIMIT:
                encoded = candidate
    failed = torch.tensor(int(encoded is None), dtype=torch.int32, device=device)
    torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if bool(failed.item()):
        if encoded is None:
            raise ValueError(
                "sequence device identity must be a bounded newline-free UTF-8 exact string"
            )
        raise RuntimeError("peer sequence device identity is invalid")
    assert encoded is not None
    length = torch.tensor(len(encoded), dtype=torch.int64, device=device)
    lengths = [torch.empty_like(length) for _ in range(config.world_size)]
    torch.distributed.all_gather(lengths, length)
    width = max(int(item.item()) for item in lengths)
    payload = torch.zeros(width, dtype=torch.uint8, device=device)
    payload[: len(encoded)] = torch.tensor(tuple(encoded), dtype=torch.uint8, device=device)
    payloads = [torch.empty_like(payload) for _ in range(config.world_size)]
    torch.distributed.all_gather(payloads, payload)
    return tuple(
        bytes(peer[: int(peer_length.item())].tolist()).decode("utf-8")
        for peer, peer_length in zip(payloads, lengths, strict=True)
    )


def receipt_identity_for_domain(
    domain: str,
    family_id: str,
    integration_facts: Sequence[str],
    compute_dtype: torch.dtype,
    world_size: int,
    *,
    prefix: str,
) -> str:
    """Hash one versioned receipt domain's canonical pre-image.

    The versioned domain and family lead the declared, ordered family facts.
    Dtype and world size follow. Hardware and software-build metadata remain
    in evidence records and never enter execution identity. Every UTF-8 line
    includes a trailing newline.
    """

    if type(family_id) is not str or not family_id or ":" in family_id or "\n" in family_id:
        raise ValueError("receipt family id must be a non-empty colon-free line")
    facts = tuple(integration_facts)
    if not facts or any(type(fact) is not str or not fact or "\n" in fact for fact in facts):
        raise ValueError("receipt integration facts must be non-empty lines")
    dtype_names = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    if compute_dtype not in dtype_names:
        raise TypeError("receipt compute dtype is unsupported")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("receipt world size must be an exact int of at least one")
    if type(domain) is not str or not domain.startswith("domain=") or "\n" in domain:
        raise ValueError("receipt domain must be a versioned domain= line")
    if type(prefix) is not str or not prefix.isalpha() or not prefix.islower():
        raise ValueError("receipt identity prefix must be a lowercase word")
    lines = (
        domain,
        f"family={family_id}",
        *facts,
        f"compute_dtype={dtype_names[compute_dtype]}",
        f"world_size={world_size}",
    )
    hasher = hashlib.sha256()
    for line in lines:
        hasher.update(line.encode("utf-8"))
        hasher.update(b"\n")
    return f"{prefix}:{family_id}:{hasher.hexdigest()}"


def distributed_receipt_identity(
    domain: str,
    family_id: str,
    integration_facts: Sequence[str],
    compute_dtype: torch.dtype,
    world_size: int,
) -> str:
    """Hash execution facts, never hardware or software-build evidence."""

    if type(world_size) is not int or world_size < 2:
        raise ValueError("distributed receipt world size must be an exact int of at least two")
    facts = tuple(
        fact
        for fact in integration_facts
        if type(fact) is not str
        or not fact.startswith(("torch_version=", "attention_provider_version="))
    )
    return receipt_identity_for_domain(
        domain,
        family_id,
        facts,
        compute_dtype,
        world_size,
        prefix="distributed",
    )


def guidance_receipt_identity(
    family_id: str,
    integration_facts: Sequence[str],
    compute_dtype: torch.dtype,
    world_size: int,
) -> str:
    """Hash the canonical distributed guidance receipt pre-image."""

    return distributed_receipt_identity(
        _GUIDANCE_RECEIPT_DOMAIN,
        family_id,
        integration_facts,
        compute_dtype,
        world_size,
    )


def sequence_receipt_identity(
    family_id: str,
    integration_facts: Sequence[str],
    compute_dtype: torch.dtype,
    world_size: int,
) -> str:
    """Hash the canonical distributed sequence-parallel receipt pre-image."""

    return distributed_receipt_identity(
        _SEQUENCE_RECEIPT_DOMAIN,
        family_id,
        integration_facts,
        compute_dtype,
        world_size,
    )


_LOCK = Lock()
_config: DistributedSamplingConfig | None = None
_active_attempt: str | None = None
_active_attempt_users = 0
_sequence_fence_groups: dict[tuple[str, int, int, int], tuple[tuple[int, ...], object]] = {}
_rank_zero_control_group: torch.distributed.ProcessGroup | None = None
_rank_zero_sampling: ContextVar[bool] = ContextVar("dinkster_rank_zero_sampling", default=False)


def _destroy_process_group() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def activate_distributed_sampling_attempt(group: str, attempt: int) -> None:
    """Bind the next process group to one committed workgroup attempt."""

    global _active_attempt, _active_attempt_users
    if not group or type(attempt) is not int or attempt <= 0:
        raise ValueError("distributed sampling attempt correlation is invalid")
    correlation = f"{group}:{attempt}"
    with _LOCK:
        if _active_attempt == correlation:
            _active_attempt_users += 1
            return
        if _active_attempt is not None or _config is not None:
            raise RuntimeError("distributed sampling attempt is already active")
        _active_attempt = correlation
        _active_attempt_users = 1


def release_distributed_sampling_attempt(group: str, attempt: int) -> None:
    """Destroy all collective state owned by one released attempt."""

    global _active_attempt, _active_attempt_users, _config, _rank_zero_control_group
    correlation = f"{group}:{attempt}"
    with _LOCK:
        if _active_attempt != correlation:
            raise RuntimeError("distributed sampling release has foreign correlation")
        _active_attempt_users -= 1
        if _active_attempt_users:
            return
        rendezvous = None if _config is None else _config.rendezvous
        _destroy_process_group()
        _config = None
        _sequence_fence_groups.clear()
        _rank_zero_control_group = None
        _active_attempt = None
    if rendezvous is not None and rendezvous.startswith("file://"):
        Path(rendezvous.removeprefix("file://")).unlink(missing_ok=True)


def distributed_sampling_config() -> DistributedSamplingConfig | None:
    if _rank_zero_sampling.get():
        return None
    rank_text = os.environ.get("DINKSTER_SINGLE_JOB_RANK")
    if rank_text is None:
        return None
    if _active_attempt is None:
        return None
    try:
        rank = int(rank_text)
        world_size = int(os.environ["DINKSTER_SINGLE_JOB_WORLD_SIZE"])
        mode = os.environ["DINKSTER_SINGLE_JOB_MULTI_GPU_MODE"]
        rendezvous_base = os.environ["DINKSTER_SINGLE_JOB_RENDEZVOUS"]
        token_base = os.environ["DINKSTER_SINGLE_JOB_TOKEN"]
    except (KeyError, ValueError) as exc:
        raise RuntimeError("single-job rank environment is malformed") from exc
    sequence_ulysses_text = os.environ.get("DINKSTER_SINGLE_JOB_SEQUENCE_ULYSSES")
    sequence_ring_text = os.environ.get("DINKSTER_SINGLE_JOB_SEQUENCE_RING")
    sequence_guidance_text = os.environ.get("DINKSTER_SINGLE_JOB_SEQUENCE_GUIDANCE")
    sequence_transport = os.environ.get("DINKSTER_SINGLE_JOB_SEQUENCE_TRANSPORT", "nccl")
    if sequence_transport not in ("nccl", "peer-copy"):
        raise RuntimeError("single-job sequence transport must be nccl or peer-copy")
    if sequence_transport != "nccl" and mode != "sequence":
        raise RuntimeError("single-job sequence transport requires sequence mode")
    if mode == "sequence":
        try:
            sequence_ulysses = int(sequence_ulysses_text or "")
            sequence_ring = int(sequence_ring_text or "")
            sequence_guidance = int(
                "1" if sequence_guidance_text is None else sequence_guidance_text
            )
        except ValueError as exc:
            raise RuntimeError("single-job sequence geometry is malformed") from exc
        if (
            sequence_guidance < 1
            or sequence_ulysses < 1
            or sequence_ring < 1
            or sequence_guidance * sequence_ulysses * sequence_ring != world_size
        ):
            raise RuntimeError("single-job sequence geometry is malformed")
    else:
        if (
            sequence_ulysses_text is not None
            or sequence_ring_text is not None
            or sequence_guidance_text is not None
        ):
            raise RuntimeError("single-job sequence geometry requires sequence mode")
        sequence_guidance = 1
        sequence_ulysses = 1
        sequence_ring = 1
    if (
        world_size < 2
        or not 0 <= rank < world_size
        or mode not in ("auto", "guidance", "sequence", "window")
        or not rendezvous_base.startswith("file://")
        or len(token_base) != 32
    ):
        raise RuntimeError("single-job rank environment is malformed")
    digest = hashlib.sha256(f"{token_base}:{_active_attempt}".encode()).hexdigest()
    rendezvous = f"{rendezvous_base}.{digest}"
    return DistributedSamplingConfig(
        rank,
        world_size,
        mode,
        rendezvous,
        digest[:32],
        _active_attempt,
        sequence_ulysses,
        sequence_ring,
        sequence_guidance,
        sequence_transport,
    )


def _fence_identity(config: DistributedSamplingConfig, operation: str) -> int:
    digest = hashlib.sha256(f"{config.attempt}:{operation}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def collective_fence_identity(config: DistributedSamplingConfig, operation: str) -> int:
    """Fence identity binding one attempt to one named collective operation."""

    return _fence_identity(config, operation)


def _configure_nccl_p2p_level(config: DistributedSamplingConfig) -> None:
    """Retain same-NUMA P2P across 3+ ranks on affected AMD hosts."""
    sequence_parallel = config.mode == "sequence" and (
        config.sequence_ulysses > 1 or config.sequence_ring > 1
    )
    if config.world_size <= 2 or sequence_parallel or "NCCL_P2P_LEVEL" in os.environ:
        return
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except (OSError, UnicodeError):
        return
    vendors = {
        value.strip()
        for line in cpuinfo.splitlines()
        for key, separator, value in (line.partition(":"),)
        if separator and key.strip() == "vendor_id"
    }
    if vendors == {"AuthenticAMD"}:
        os.environ.setdefault("NCCL_P2P_LEVEL", "PHB")


def ensure_process_group() -> DistributedSamplingConfig | None:
    global _config
    config = distributed_sampling_config()
    if config is None:
        return None
    with _LOCK:
        if _config is not None:
            if _config != config:
                raise RuntimeError("single-job process group configuration changed")
            return _config
        if not torch.cuda.is_available():
            raise RuntimeError("single-job multi-GPU execution requires CUDA")
        if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
            raise RuntimeError("single-job multi-GPU execution requires torch.distributed NCCL")
        _configure_nccl_p2p_level(config)
        if config.sequence_transport == "peer-copy":
            # Every rank's process must see all group devices; rank r
            # drives device index r so peers can address each other's
            # memory for direct copies.
            if torch.cuda.device_count() < config.world_size:
                raise RuntimeError(
                    "peer-copy sequence transport requires every rank to see all "
                    "sequence-group devices (launch without per-rank device isolation)"
                )
            torch.cuda.set_device(config.rank)
        else:
            torch.cuda.set_device(0)
        torch.distributed.init_process_group(
            "nccl",
            init_method=config.rendezvous,
            rank=config.rank,
            world_size=config.world_size,
            timeout=timedelta(minutes=5),
            group_name=config.token,
        )
        atexit.register(_destroy_process_group)
        _config = config
    return config


def synchronized_sampling_call(
    action: Callable[[], _Result], device: torch.device, phase: str
) -> _Result:
    """Propagate callback failures and cancellation before peers enter another collective.

    Every sampler replica must call this at the same execution boundary.
    Model evaluations have their own data-plane error fence.
    """
    if distributed_sampling_config() is None or not torch.distributed.is_initialized():
        return action()
    error: BaseException | None = None
    result: _Result | None = None
    try:
        result = action()
    except BaseException as caught:
        error = caught
    status = torch.tensor(
        2 if isinstance(error, SamplingCancelled) else int(error is not None),
        dtype=torch.int32,
        device=device,
    )
    torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MAX)
    if error is not None:
        raise error
    if status.item() == 2:
        raise SamplingCancelled(f"peer sampling cancelled during {phase}")
    if status.item():
        raise RuntimeError(f"peer sampling {phase} failed")
    return cast("_Result", result)


def rank_zero_sampling_active() -> bool:
    return _rank_zero_sampling.get()


@contextmanager
def _use_rank_zero_sampling() -> Generator[None, None, None]:
    token = _rank_zero_sampling.set(True)
    try:
        yield
    finally:
        _rank_zero_sampling.reset(token)


def _rank_zero_control_group_for_sampling() -> torch.distributed.ProcessGroup | None:
    global _rank_zero_control_group
    if not torch.distributed.is_initialized():
        return None
    with _LOCK:
        if _rank_zero_control_group is None:
            _rank_zero_control_group = cast(
                "torch.distributed.ProcessGroup",
                torch.distributed.new_group(
                    backend="gloo",
                    timeout=timedelta(seconds=_RANK_ZERO_CONTROL_TIMEOUT_SECONDS),
                ),
            )
        return _rank_zero_control_group


def run_rank_zero_sampling(
    action: Callable[[], torch.Tensor],
    template: torch.Tensor,
    config: DistributedSamplingConfig,
) -> torch.Tensor:
    """Run one sampling stage on rank zero and broadcast its final tensor.

    A control heartbeat keeps every rank in matching short collectives while
    the action runs, so a long sample never strands peers in the terminal
    broadcast until the process-group timeout.
    """

    active = ensure_process_group()
    if active != config:
        raise RuntimeError("rank-zero sampling process group configuration changed")
    control_group = _rank_zero_control_group_for_sampling()
    action_error: BaseException | None = None
    control_error: BaseException | None = None
    result: torch.Tensor | None = None
    action_finished = Event()
    peer_failed = Event()

    def control_rank_zero() -> None:
        nonlocal control_error
        sequence = 0
        try:
            while not action_finished.wait(_RANK_ZERO_HEARTBEAT_SECONDS):
                status = torch.tensor((sequence, 0), dtype=torch.int64)
                torch.distributed.broadcast(status, src=0, group=control_group)
                sequence += 1
            terminal = (
                3
                if isinstance(action_error, SamplingCancelled)
                else 2
                if action_error is not None
                else 1
            )
            status = torch.tensor((sequence, terminal), dtype=torch.int64)
            torch.distributed.broadcast(status, src=0, group=control_group)
        except BaseException as caught:
            control_error = caught
            peer_failed.set()

    if config.rank == 0:
        control = Thread(target=control_rank_zero, name="dinkster-rank-zero-control")
        control.start()
        try:
            with (
                _use_rank_zero_sampling(),
                use_additional_sampling_cancellation(peer_failed.is_set),
            ):
                result = action()
            if (
                type(result) is not torch.Tensor
                or result.shape != template.shape
                or result.dtype != template.dtype
                or result.device != template.device
            ):
                raise RuntimeError("rank-zero sampling returned incompatible tensor metadata")
        except BaseException as caught:
            action_error = caught
        finally:
            action_finished.set()
            control.join()
        if control_error is not None:
            raise RuntimeError("rank-zero sampling control failed") from control_error
        terminal_status = (
            3
            if isinstance(action_error, SamplingCancelled)
            else 2
            if action_error is not None
            else 1
        )
    else:
        expected_sequence = 0
        while True:
            status = torch.empty(2, dtype=torch.int64)
            torch.distributed.broadcast(status, src=0, group=control_group)
            sequence, terminal_status = (int(value) for value in status.tolist())
            if sequence != expected_sequence:
                raise RuntimeError("rank-zero sampling control sequence changed")
            expected_sequence += 1
            if terminal_status:
                break
    if terminal_status != 1:
        if action_error is not None:
            raise action_error
        if terminal_status == 3:
            raise SamplingCancelled("rank-zero sampling cancelled")
        if terminal_status != 2:
            raise RuntimeError("rank-zero sampling returned invalid terminal status")
        raise RuntimeError("rank-zero sampling failed")
    output = result if result is not None else torch.empty_like(template)
    torch.distributed.broadcast(output, src=0)
    return output


class DistributedGuidanceEvaluator:
    """Evaluate assigned guidance lanes locally and reduce them canonically."""

    def __init__(
        self,
        evaluate: Callable[
            [torch.Tensor, float, GuidanceEvaluationRequest[torch.Tensor]],
            GuidancePredictions[torch.Tensor],
        ],
        *,
        guidance_group_size: int | None = None,
    ) -> None:
        config = ensure_process_group()
        if config is None:
            raise RuntimeError("distributed guidance requires a configured rank")
        self.config = config
        self.evaluate = evaluate
        if guidance_group_size is not None and (
            type(guidance_group_size) is not int
            or guidance_group_size < 1
            or config.world_size % guidance_group_size
        ):
            raise RuntimeError("distributed guidance group size is malformed")
        self.guidance_group_size = guidance_group_size
        if guidance_group_size is None and config.mode not in ("auto", "guidance"):
            _LOGGER.warning(
                "No specialized %s evaluator was selected; using shared guidance replicas",
                config.mode,
            )
        self._last_ordinal = -1

    def _begin_fence(
        self,
        ordinal: int,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> tuple[object, list[torch.Tensor], torch.Tensor]:
        if ordinal <= self._last_ordinal:
            raise RuntimeError("guidance evaluation ordinal is stale or duplicate")
        if len(shape) > 14:
            raise RuntimeError("guidance sideband supports at most 14 tensor dimensions")
        control = torch.tensor(
            (
                _fence_identity(self.config, "guidance"),
                ordinal,
                len(shape),
                *shape,
                *(0 for _ in range(14 - len(shape))),
            ),
            dtype=torch.int64,
            device=device,
        )
        controls = [torch.empty_like(control) for _ in range(self.config.world_size)]
        work = torch.distributed.all_gather(controls, control, async_op=True)
        return work, controls, control

    def _finish_fence(
        self,
        ordinal: int,
        pending: tuple[object, list[torch.Tensor], torch.Tensor],
    ) -> None:
        work, controls, control = pending
        work.wait()  # type: ignore[attr-defined]
        if bool(torch.stack(controls).ne(control).any().item()):
            raise RuntimeError("guidance sideband control disagrees")
        self._last_ordinal = ordinal

    def evaluate_request(
        self,
        x: torch.Tensor,
        sigma: float,
        request: GuidanceEvaluationRequest[torch.Tensor],
    ) -> GuidancePredictions[torch.Tensor]:
        ordinal = request.execution.model_evaluation
        fence = self._begin_fence(ordinal, tuple(x.shape), x.device)
        if self.guidance_group_size is None:
            assignments = plan_guidance_lanes(
                request.plan,
                RankWeights(tuple(1.0 for _ in range(self.config.world_size))),
            )
            local_rank = self.config.rank
        else:
            guidance_degree = self.config.world_size // self.guidance_group_size
            assignments = tuple(
                replace(assignment, rank=assignment.rank * self.guidance_group_size)
                for assignment in plan_guidance_lanes(
                    request.plan,
                    RankWeights(tuple(1.0 for _ in range(guidance_degree))),
                )
            )
            local_rank = (self.config.rank // self.guidance_group_size) * self.guidance_group_size
        local_lanes = tuple(
            request.plan.lanes[item.plan_index]
            for item in assignments
            if item.rank == local_rank and item.source is GuidancePredictionSource.MODEL
        )
        local: dict[str, torch.Tensor] = {}
        local_error: BaseException | None = None
        try:
            if local_lanes:
                lane_ids = {lane.id for lane in local_lanes}
                local_plan = GuidanceEvaluationPlan(
                    local_lanes,
                    request.plan.primary_id
                    if request.plan.primary_id in lane_ids
                    else local_lanes[0].id,
                    request.plan.unconditional_id
                    if request.plan.unconditional_id in lane_ids
                    else None,
                )
                predictions = self.evaluate(
                    x,
                    sigma,
                    replace(request, plan=local_plan),
                )
                local = {item.lane_id: item.value for item in predictions.items}
        except BaseException as error:
            local_error = error

        self._finish_fence(ordinal, fence)
        failed = torch.tensor(
            1 if local_error is not None else 0,
            dtype=torch.int32,
            device=x.device,
        )
        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
        if bool(failed.item()):
            if local_error is not None:
                raise local_error
            raise RuntimeError("peer guidance evaluation failed")

        model_assignments = tuple(
            assignment
            for assignment in assignments
            if assignment.source is GuidancePredictionSource.MODEL
        )
        assignments_by_rank = tuple(
            tuple(assignment for assignment in model_assignments if assignment.rank == rank)
            for rank in range(self.config.world_size)
        )
        width = max(len(items) for items in assignments_by_rank)
        local_values = tuple(
            local[assignment.lane_id] for assignment in assignments_by_rank[self.config.rank]
        )
        packed = torch.stack(
            (*local_values, *(torch.zeros_like(x) for _ in range(width - len(local_values))))
        )
        rank_values = [torch.empty_like(packed) for _ in range(self.config.world_size)]
        torch.distributed.all_gather(rank_values, packed)
        model_values = {}
        for rank, rank_assignments in enumerate(assignments_by_rank):
            model_values.update(
                {
                    assignment.lane_id: rank_values[rank][index]
                    for index, assignment in enumerate(rank_assignments)
                }
            )
        gathered: list[GuidancePrediction[torch.Tensor]] = []
        for assignment in assignments:
            if assignment.source is GuidancePredictionSource.SYNTHETIC_ZERO:
                gathered.append(
                    GuidancePrediction(
                        assignment.lane_id,
                        torch.zeros_like(x),
                        GuidancePredictionSource.SYNTHETIC_ZERO,
                    )
                )
                continue
            gathered.append(
                GuidancePrediction(
                    assignment.lane_id,
                    model_values[assignment.lane_id],
                    GuidancePredictionSource.MODEL,
                )
            )
        return GuidancePredictions(tuple(gathered))


def fence_sequence_evaluation(
    ordinal: int,
    value: torch.Tensor,
    *,
    lane_identity: str,
    sequence_identity: str,
) -> None:
    config = ensure_process_group()
    if config is None:
        return
    if config.mode != "sequence":
        raise RuntimeError("sequence sideband requires sequence mode")
    dtype_code = sum(ord(char) for char in str(value.dtype))
    lane_code = int.from_bytes(
        hashlib.sha256(lane_identity.encode()).digest()[:8], "big", signed=True
    )
    sequence_code = int.from_bytes(
        hashlib.sha256(sequence_identity.encode()).digest()[:8], "big", signed=True
    )
    control = torch.tensor(
        (
            _fence_identity(config, "sequence"),
            ordinal,
            value.ndim,
            *value.shape,
            dtype_code,
            config.sequence_ulysses,
            config.sequence_ring,
            lane_code,
            sequence_code,
        ),
        dtype=torch.int64,
        device=value.device,
    )
    if config.sequence_guidance == 1:
        peers = [torch.empty_like(control) for _ in range(config.world_size)]
        torch.distributed.all_gather(peers, control)
    else:
        ranks, group = _sequence_fence_group(config)
        peers = [torch.empty_like(control) for _ in ranks]
        torch.distributed.all_gather(peers, control, group=group)
    if any(not torch.equal(peer, control) for peer in peers):
        raise RuntimeError("sequence-parallel sideband control disagrees")


def _sequence_fence_group(
    config: DistributedSamplingConfig,
) -> tuple[tuple[int, ...], object]:
    key = (
        config.attempt,
        config.sequence_guidance,
        config.sequence_ulysses,
        config.sequence_ring,
    )
    cached = _sequence_fence_groups.get(key)
    if cached is not None:
        return cached
    sequence_size = config.sequence_ulysses * config.sequence_ring
    selected: tuple[tuple[int, ...], object] | None = None
    for guidance in range(config.sequence_guidance):
        ranks = tuple(range(guidance * sequence_size, (guidance + 1) * sequence_size))
        group = torch.distributed.new_group(ranks=list(ranks))
        if config.rank in ranks:
            selected = ranks, group
    if selected is None:
        raise RuntimeError("sequence rank does not belong to a guidance group")
    _sequence_fence_groups[key] = selected
    return selected


def sequence_preflight_failed(local_failed: bool, device: torch.device) -> bool:
    config = ensure_process_group()
    if config is None:
        return local_failed
    if config.mode != "sequence":
        raise RuntimeError("sequence preflight requires sequence mode")
    failed = torch.tensor(int(local_failed), dtype=torch.int32, device=device)
    torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    return bool(failed.item())


__all__ = [
    "DistributedGuidanceEvaluator",
    "DistributedSamplingConfig",
    "SequenceDigestConsensusTransport",
    "activate_distributed_sampling_attempt",
    "collective_fence_identity",
    "distributed_receipt_identity",
    "distributed_sampling_config",
    "ensure_process_group",
    "fence_sequence_evaluation",
    "gather_sequence_device_bindings",
    "guidance_receipt_identity",
    "receipt_identity_for_domain",
    "release_distributed_sampling_attempt",
    "rank_zero_sampling_active",
    "run_rank_zero_sampling",
    "sequence_preflight_failed",
    "sequence_receipt_identity",
]
