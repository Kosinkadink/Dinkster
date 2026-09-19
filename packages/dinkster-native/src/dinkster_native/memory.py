"""Reservation policy for the compat pack (DESIGN 3.10).

Materializing new resources requires explicit per-node-type policy: a
generic "asset in, resource out" heuristic would misclassify converters,
inspectors, and multi-asset nodes. Input residency is different. A resident
input's envelope declares its identity and device cost, and the child-local
pool can say whether that identity is loaded, so this observable case is
generic over every invocation. Both rules run inside the worker process and
never touch node code or devices (hazards H9/H13).

Policies:

- ``dinkster.load_checkpoint``: the selected arm determines peak incremental
  load allocation. Comfy reserves ``ceil(size * 2.6)`` RAM; native reserves
  ``ceil(size * 1.5)`` RAM and zero checkpoint VRAM because it assembles on
  CPU. Stage-time native VRAM is planned from the unloaded resident input.
  An asset that declares no byte size fails planning loudly.
- resident inputs: reserve each declared ``vram:*`` cost when the pool
  cannot prove the resource is loaded. Repeated resource ids reserve once.
  An envelope with VRAM cost but no resource id also reserves (conservative),
  but cannot be deduplicated because it has no stable identity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

from dinkster_memory import InvocationView, ReservationRequest
from dinkster_values import COST_META_KEY, RESOURCE_ID_META_KEY, iter_value_tree

from .pool import default_pool

CHECKPOINT_RAM_FACTOR = 2.6
"""Comfy-arm peak incremental RAM multiplier for load_checkpoint.

Calibrated 2026-07-27 with tools/measure_checkpoint_ram.py against
v1-5-pruned-emaonly-fp16.safetensors: 2,132,696,762 file bytes,
1,000,951,808-byte child baseline HWM, 5,305,561,088-byte child peak HWM,
measured peak/file factor 2.487724. The 2.6 policy rounds that live peak up
rather than preserving the under-reserving 2.0 estimate.

This planner request is peak incremental allocation during loading. It is not
the pool's resident RAM footprint, which is retained capacity for the
resident's lifetime. The Comfy arm remains at 2.6.
"""

NATIVE_CHECKPOINT_RAM_FACTOR = 1.5
"""Native-arm peak incremental RAM multiplier for load_checkpoint.

Calibrated against handle-inclusive native CPU assembly and per-unit enrollment
measurements in the private comfy-vibe-station research notes
(notes/research/native-checkpoint-ram-calibration.md; combined
Flux fp8 maximum 1.046415; SD1.5 recheck 1.354791; maximum across accepted
historical and current runs 1.358514). Native checkpoint loading reserves no
VRAM: assembly stays on CPU, and stage-time placement is governed from
resident-input metadata.
"""


class ReservationPlanError(Exception):
    """The invocation's envelopes do not carry enough to plan honestly."""


def plan_reservations(invocation: InvocationView) -> Sequence[ReservationRequest]:
    requests = _input_vram_reservations(invocation)
    if invocation.node_type != "dinkster.load_checkpoint":
        return tuple(requests)
    checkpoint = invocation.inputs.get("checkpoint")
    if checkpoint is None:
        return tuple(requests)  # schema validation owns reporting the missing input
    size = checkpoint.meta.get("size")
    if not isinstance(size, int) or size <= 0:
        raise ReservationPlanError(
            "checkpoint asset declares no byte size; cannot plan a load "
            "reservation (refusing to silently reserve zero)"
        )
    factor = (
        NATIVE_CHECKPOINT_RAM_FACTOR
        if getattr(invocation, "arm", None) == "native"
        else CHECKPOINT_RAM_FACTOR
    )
    requests.append(ReservationRequest(residency="ram", nbytes=math.ceil(size * factor)))
    return tuple(requests)


def _input_vram_reservations(
    invocation: InvocationView,
) -> list[ReservationRequest]:
    pool = default_pool()
    by_resource: dict[str, dict[str, int]] = {}
    anonymous: list[dict[str, int]] = []
    for value in invocation.inputs.values():
        for envelope in iter_value_tree(value):
            declared = envelope.meta.get(COST_META_KEY)
            if not isinstance(declared, Mapping):
                continue
            vram = {
                residency: nbytes
                for residency, nbytes in cast("Mapping[object, object]", declared).items()
                if isinstance(residency, str)
                and residency.startswith("vram:")
                and isinstance(nbytes, int)
                and not isinstance(nbytes, bool)
                and nbytes > 0
            }
            if not vram:
                continue
            resource_id = envelope.meta.get(RESOURCE_ID_META_KEY)
            if not isinstance(resource_id, str):
                anonymous.append(vram)
                continue
            merged = by_resource.setdefault(resource_id, {})
            for residency, nbytes in vram.items():
                # Duplicate envelopes describe one resident. A maximum is
                # conservative against stale siblings without double-counting
                # the same device allocation.
                merged[residency] = max(merged.get(residency, 0), nbytes)

    requests: list[ReservationRequest] = []
    for resource_id, costs in by_resource.items():
        if pool.is_loaded(resource_id) is True:
            continue
        requests.extend(
            ReservationRequest(residency=residency, nbytes=nbytes)
            for residency, nbytes in costs.items()
        )
    for costs in anonymous:
        requests.extend(
            ReservationRequest(residency=residency, nbytes=nbytes)
            for residency, nbytes in costs.items()
        )
    return requests
