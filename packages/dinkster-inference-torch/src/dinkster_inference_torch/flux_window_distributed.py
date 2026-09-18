"""Distributed window-scattered evaluation for Flux windowed sampling.

Window scatter parallelizes only the per-step window evaluations a
windowed (MultiDiffusion-style) plan already performs. It is
circumstantial acceleration and is never counted toward the core
multi-GPU performance goals (issues #121/#298).

Every rank holds a full model replica and the full latent, evaluates
its deterministically assigned joint windows, and the group gathers
every window output so each rank performs the identical canonical
merge. The merge consumes the gathered buffers in plan-window order on
every rank, so the result does not depend on group size or on which
rank evaluated which window.

Construction of the scatter evaluator demands the group's
``ManifestConsensusToken``: the proven manifest binds the exact
composite window plan through the windowed-evaluation slot, so no
scatter collective can be issued before group-wide plan consensus.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Generic, TypeVar

import torch
from dinkster_inference import (
    CanonicalManifest,
    ManifestConsensusToken,
    ModelTokenLayout,
    OptionValue,
    RankWeights,
    TokenGridTransform,
    WindowPlanBinding,
    build_canonical_manifest,
    build_windowed_evaluation_slot,
    plan_window_units,
)

from .distributed import (
    DistributedSamplingConfig,
    collective_fence_identity,
    distributed_receipt_identity,
    ensure_process_group,
    rank_zero_sampling_active,
)
from .flux_window import (
    FluxWindowConditioningEvaluation,
    FluxWindowError,
    PreparedFluxWindowConditioning,
    PreparedFluxWindowPlan,
    crop_flux_window,
    derive_flux_window_layout,
    merge_flux_window_outputs,
)

PreparedConditionT = TypeVar("PreparedConditionT")

_WINDOW_RECEIPT_DOMAIN = "domain=dinkster.distributed.window-receipt.v2"
_STATIC_WINDOW_DERIVATION_IDENTITY = "dinkster.flux.window-derivation.caller-static.v1"

# Classic Flux windows carry no sequence partition plan and refuse
# structural passthrough rows, so their manifest binding pins those
# per-window digests to versioned absence markers.
_ABSENT_PARTITION_PLAN_DIGEST = hashlib.sha256(
    b"dinkster.flux.window.partition-plan.absent.v1\n"
).hexdigest()
_ABSENT_STRUCTURAL_ROWS_DIGEST = hashlib.sha256(
    b"dinkster.flux.window.structural-rows.absent.v1\n"
).hexdigest()

_FENCE_SHAPE_LIMIT = 12


def _dtype_code(dtype: torch.dtype) -> int:
    """One int64 fence-control slot naming the tensor dtype.

    Hashed like the fence identity: a character-sum code collides
    (float8_e4m3fnuz and float8_e5m2fnuz sum identically), which would
    let ranks with different dtypes pass the sideband agreement check.
    """
    digest = hashlib.sha256(str(dtype).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def window_receipt_identity(
    family_id: str,
    integration_facts: Sequence[str],
    compute_dtype: torch.dtype,
    world_size: int,
) -> str:
    """Hash the canonical distributed window-scatter receipt pre-image."""

    return distributed_receipt_identity(
        _WINDOW_RECEIPT_DOMAIN,
        family_id,
        integration_facts,
        compute_dtype,
        world_size,
    )


class WindowDigestConsensusTransport:
    """Digest-only consensus over the whole single-job process group.

    Window scatter runs full-replica ranks, so its consensus group is
    the entire world: ``physical_ranks`` is always every rank of the
    configured group, and the digest all-gather runs on the default
    process group.
    """

    def __init__(self, config: DistributedSamplingConfig, device: torch.device) -> None:
        if type(config) is not DistributedSamplingConfig or config.mode not in ("auto", "window"):
            raise ValueError("window consensus requires a window-eligible configuration")
        if type(device) is not torch.device:
            raise TypeError("window consensus device must be an exact torch.device")
        self._ranks = tuple(range(config.world_size))
        self._rank = config.rank
        self._device = device

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def physical_ranks(self) -> tuple[int, ...]:
        return self._ranks

    def exchange_digest(self, rank: int, digest: str) -> tuple[str, ...]:
        if type(rank) is not int or rank != self._rank:
            raise ValueError("consensus rank does not match the configured window rank")
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("consensus digest must be lowercase sha256 hex")
        local = torch.tensor(tuple(bytes.fromhex(digest)), dtype=torch.uint8, device=self._device)
        peers = [torch.empty_like(local) for _ in self._ranks]
        torch.distributed.all_gather(peers, local)
        return tuple(bytes(peer.tolist()).hex() for peer in peers)


def window_preflight_failed(local_failed: bool, device: torch.device) -> bool:
    """Group-wide preflight barrier: any rank's failure fails every rank.

    One rank's manifest-derivation failure must fail the group loudly
    before the consensus all-gather, never leave peers hanging on it.
    """

    failed = torch.tensor(int(bool(local_failed)), dtype=torch.int32, device=device)
    torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    return bool(failed.item())


def window_route_mismatch(local_window: bool, device: torch.device) -> bool:
    """Require every rank to choose the same window or generic route."""

    routes = torch.tensor(
        (int(bool(local_window)), int(not local_window)), dtype=torch.int32, device=device
    )
    torch.distributed.all_reduce(routes, op=torch.distributed.ReduceOp.MAX)
    return bool(routes[0].item() and routes[1].item())


def flux_window_plan_binding(
    prepared_plan: PreparedFluxWindowPlan,
    *,
    text_token_count: int,
) -> WindowPlanBinding:
    """Bind one prepared Flux plan's per-window digests for the manifest.

    Per-window token layouts are derived from the primary lane's
    declared token count; per-lane counts are pinned separately as
    manifest invocation facts.
    """

    if type(prepared_plan) is not PreparedFluxWindowPlan:
        raise FluxWindowError("plan binding requires an exact PreparedFluxWindowPlan")
    layout_digests = tuple(
        derive_flux_window_layout(
            text_token_count=text_token_count,
            latent_height=prepared_plan.latent_height,
            latent_width=prepared_plan.latent_width,
            patch_size=prepared_plan.patch_size,
            height_indices=window.height_indices,
            width_indices=window.width_indices,
        )[0].digest
        for window in prepared_plan.windows
    )
    window_count = len(prepared_plan.windows)
    return WindowPlanBinding(
        plan=prepared_plan.declaration,
        window_token_layout_digests=layout_digests,
        window_partition_plan_digests=(_ABSENT_PARTITION_PLAN_DIGEST,) * window_count,
        window_structural_row_digests=(_ABSENT_STRUCTURAL_ROWS_DIGEST,) * window_count,
    )


def build_flux_window_manifest(
    *,
    runtime_identity: str,
    config: DistributedSamplingConfig,
    prepared_plan: PreparedFluxWindowPlan,
    text_token_counts: tuple[int, ...],
    sampler_id: str,
    sampler_options: tuple[tuple[str, OptionValue], ...],
    seed: int,
    pre_offset_sigmas: Sequence[float],
    sigmas: Sequence[float],
) -> CanonicalManifest:
    """Freeze one windowed Flux invocation's canonical manifest.

    ``text_token_counts`` carries every evaluated lane's declared token
    count, conditional lane first. The requested and executed sigma tables
    are pinned as exact float hex so schedule identity is bound bit-for-bit.
    """

    if type(text_token_counts) is not tuple or not text_token_counts:
        raise FluxWindowError("manifest lanes require at least one declared token count")
    binding = flux_window_plan_binding(prepared_plan, text_token_count=text_token_counts[0])
    derivation_facts_digest = hashlib.sha256(
        f"static-plan={prepared_plan.declaration.digest}\n".encode()
    ).hexdigest()
    slot = build_windowed_evaluation_slot(
        derivation_identity=_STATIC_WINDOW_DERIVATION_IDENTITY,
        derivation_facts_digest=derivation_facts_digest,
        plan_bindings=(binding,),
    )
    return build_canonical_manifest(
        runtime_identity=runtime_identity,
        invocation_facts=(
            "mode=window",
            f"world_size={config.world_size}",
            f"sampler={sampler_id}",
            "sampler_options="
            + json.dumps(sampler_options, ensure_ascii=True, separators=(",", ":")),
            "sampling=custom-sigmas",
            f"seed={seed}",
            "pre_offset_sigma_table=" + ",".join(float(value).hex() for value in pre_offset_sigmas),
            "sigma_table=" + ",".join(float(value).hex() for value in sigmas),
            "text_token_counts=" + ",".join(str(count) for count in text_token_counts),
            f"joint_window_count={len(prepared_plan.windows)}",
        ),
        slots=(slot,),
        rank_plan_digests=tuple(
            (prepared_plan.declaration.digest,) for _ in range(config.world_size)
        ),
    )


class DistributedFluxWindowEvaluation(Generic[PreparedConditionT]):
    """Scatter per-window Flux evaluations across the replica group.

    Wraps one :class:`FluxWindowConditioningEvaluation` and replaces
    its serial per-window loop with a deterministic scatter/gather:
    window-to-rank assignment is the shared weighted-count partition of
    ascending window indices, so every rank derives the identical exact
    partition without negotiation. Failure handling is symmetric: a
    rank whose local evaluation fails still runs the identical
    collective schedule (fence, failure reduction) and surfaces its
    error after, so peers fail closed instead of hanging.
    """

    def __init__(
        self,
        inner: FluxWindowConditioningEvaluation[PreparedConditionT],
        consensus_token: ManifestConsensusToken,
    ) -> None:
        config = ensure_process_group()
        if config is None:
            raise RuntimeError("distributed window evaluation requires a configured rank")
        if type(inner) is not FluxWindowConditioningEvaluation:
            raise TypeError("inner evaluation must be an exact FluxWindowConditioningEvaluation")
        if type(consensus_token) is not ManifestConsensusToken:
            raise RuntimeError("window scatter requires the group's manifest consensus token")
        if consensus_token.group_size != config.world_size or consensus_token.rank != config.rank:
            raise RuntimeError("window consensus token does not cover this process group")
        if len(inner.prepared_plan.windows) < 2:
            raise FluxWindowError("window scatter requires at least two joint windows")
        self.config = config
        self.inner = inner
        self.consensus_token = consensus_token
        self._weights = RankWeights(tuple(1.0 for _ in range(config.world_size)))
        self._last_ordinal = -1

    def prepare_conditioning(
        self,
        conditioning: object,
    ) -> PreparedFluxWindowConditioning[PreparedConditionT]:
        return self.inner.prepare_conditioning(conditioning)

    def validate_layout(
        self,
        conditioning: PreparedFluxWindowConditioning[PreparedConditionT],
        layout: ModelTokenLayout,
    ) -> None:
        self.inner.validate_layout(conditioning, layout)

    @staticmethod
    def inner_calls(
        conditioning: PreparedFluxWindowConditioning[PreparedConditionT],
    ) -> tuple[tuple[ModelTokenLayout, tuple[TokenGridTransform, ...]], ...]:
        return FluxWindowConditioningEvaluation.inner_calls(conditioning)

    def batchable(
        self,
        conditions: tuple[PreparedFluxWindowConditioning[PreparedConditionT], ...],
    ) -> bool:
        return self.inner.batchable(conditions)

    def evaluate_conditioning(
        self,
        x: torch.Tensor,
        sigma: float,
        condition: PreparedFluxWindowConditioning[PreparedConditionT],
    ) -> torch.Tensor:
        if rank_zero_sampling_active():
            return self.inner.evaluate_conditioning(x, sigma, condition)
        return self._scatter_evaluate(x, sigma, (condition,), batch=False)[0]

    def evaluate_conditioning_batch(
        self,
        x: torch.Tensor,
        sigma: float,
        conditions: tuple[PreparedFluxWindowConditioning[PreparedConditionT], ...],
    ) -> tuple[torch.Tensor, ...]:
        if rank_zero_sampling_active():
            return self.inner.evaluate_conditioning_batch(x, sigma, conditions)
        return self._scatter_evaluate(x, sigma, conditions, batch=True)

    def _window_shape(self, x: torch.Tensor, window_index: int) -> tuple[int, int, int, int]:
        window = self.inner.prepared_plan.windows[window_index]
        patch = self.inner.prepared_plan.patch_size
        return (
            int(x.shape[0]),
            int(x.shape[1]),
            len(window.height_indices) * patch,
            len(window.width_indices) * patch,
        )

    def _begin_fence(
        self,
        ordinal: int,
        lane_count: int,
        x: torch.Tensor,
    ) -> tuple[object, list[torch.Tensor], torch.Tensor]:
        if ordinal <= self._last_ordinal:
            raise RuntimeError("window evaluation ordinal is stale or duplicate")
        shape = tuple(int(extent) for extent in x.shape)
        if len(shape) > _FENCE_SHAPE_LIMIT:
            raise RuntimeError(
                f"window sideband supports at most {_FENCE_SHAPE_LIMIT} tensor dimensions"
            )
        control = torch.tensor(
            (
                collective_fence_identity(self.config, "window"),
                ordinal,
                len(self.inner.prepared_plan.windows),
                lane_count,
                _dtype_code(x.dtype),
                len(shape),
                *shape,
                *(0 for _ in range(_FENCE_SHAPE_LIMIT - len(shape))),
            ),
            dtype=torch.int64,
            device=x.device,
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
            raise RuntimeError("window sideband control disagrees")
        self._last_ordinal = ordinal

    def _scatter_evaluate(
        self,
        x: torch.Tensor,
        sigma: float,
        conditions: tuple[PreparedFluxWindowConditioning[PreparedConditionT], ...],
        *,
        batch: bool,
    ) -> tuple[torch.Tensor, ...]:
        plan = self.inner.prepared_plan
        window_count = len(plan.windows)
        # Nothing may raise between here and the fence: a pre-fence
        # refusal on one rank abandons peers already blocked in the
        # fence all-gather. Invalid conditions enter the fence with a
        # zero lane count, so a lane disagreement refuses symmetrically
        # at the sideband and a group-wide invalid batch fails through
        # the failure all-reduce below.
        lane_count = len(conditions) if type(conditions) is tuple else 0
        assignments = plan_window_units(window_count, self._weights)
        ordinal = self._last_ordinal + 1
        fence = self._begin_fence(ordinal, lane_count, x)

        local_error: BaseException | None = None
        local_outputs: dict[int, tuple[torch.Tensor, ...]] = {}
        try:
            if type(conditions) is not tuple or not conditions:
                raise FluxWindowError("batch evaluation requires a non-empty condition tuple")
            for assignment in assignments:
                if assignment.rank != self.config.rank:
                    continue
                index = assignment.window_index
                evaluator = self.inner.evaluators[index]
                window_x = crop_flux_window(x, plan, index)
                if batch:
                    raw = evaluator.evaluate_conditioning_batch(
                        window_x,
                        sigma,
                        tuple(condition.windows[index] for condition in conditions),
                    )
                    if type(raw) is not tuple or len(raw) != lane_count:
                        raise FluxWindowError(
                            f"joint window {index} returned the wrong number of lane predictions"
                        )
                    outputs = raw
                else:
                    outputs = (
                        evaluator.evaluate_conditioning(
                            window_x,
                            sigma,
                            conditions[0].windows[index],
                        ),
                    )
                expected_shape = self._window_shape(x, index)
                for output in outputs:
                    if (
                        type(output) is not torch.Tensor
                        or tuple(output.shape) != expected_shape
                        or output.dtype != x.dtype
                        or output.device != x.device
                    ):
                        raise FluxWindowError(
                            f"joint window {index} returned an incompatible tensor"
                        )
                local_outputs[index] = outputs
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
            raise RuntimeError("peer window evaluation failed")

        windows_by_rank = tuple(
            tuple(assignment.window_index for assignment in assignments if assignment.rank == rank)
            for rank in range(self.config.world_size)
        )
        numels = tuple(math.prod(self._window_shape(x, index)) for index in range(window_count))
        payload_sizes = tuple(
            lane_count * sum(numels[index] for index in indices) for indices in windows_by_rank
        )
        width = max(payload_sizes)
        local_flat = torch.zeros(width, dtype=x.dtype, device=x.device)
        offset = 0
        for index in windows_by_rank[self.config.rank]:
            for lane in range(lane_count):
                value = local_outputs[index][lane].reshape(-1)
                local_flat[offset : offset + numels[index]] = value
                offset += numels[index]
        gathered = [torch.empty_like(local_flat) for _ in range(self.config.world_size)]
        torch.distributed.all_gather(gathered, local_flat)

        lane_outputs: dict[int, list[torch.Tensor]] = {}
        for rank, indices in enumerate(windows_by_rank):
            offset = 0
            for index in indices:
                shape = self._window_shape(x, index)
                lanes: list[torch.Tensor] = []
                for _lane in range(lane_count):
                    lanes.append(gathered[rank][offset : offset + numels[index]].view(shape))
                    offset += numels[index]
                lane_outputs[index] = lanes
        return tuple(
            merge_flux_window_outputs(
                plan,
                tuple(lane_outputs[index][lane] for index in range(window_count)),
            )
            for lane in range(lane_count)
        )


__all__ = [
    "DistributedFluxWindowEvaluation",
    "WindowDigestConsensusTransport",
    "build_flux_window_manifest",
    "flux_window_plan_binding",
    "window_preflight_failed",
    "window_receipt_identity",
    "window_route_mismatch",
]
