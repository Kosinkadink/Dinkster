"""PlacementWorker: placement consumes residency (DESIGN 3.10).

RoutingWorker answers "which worker implements this node type"; this
answers the orthogonal question for homogeneous workers: "which instance
must this invocation run on". The rule is the same one admission lanes
use (hazard H12): where a node runs is a fact of the *values it receives*,
never of the node type. A sampler consuming a model resident on the
worker that owns cuda:1 must execute there - the model cannot leave, and
DeviceMap has already translated the value's residency meta into the
parent's namespace, so device strings name silicon unambiguously.

Resolution order per invocation:

1. **Owners win.** Any input declaring device residency pins the
   invocation to the worker that owns that device. Two inputs resident on
   *different* workers is an error result, not a guess - cross-worker
   transfer of resident state does not exist (yet), and routing anywhere
   would produce a confusing ResidentLookupError deep in the wrong child.
2. **Unpinned invocations go to policy** (``place``), the hook where a
   scheduler decides which GPU gets a fresh model load.
3. **Then the default worker**, when configured.

Known limit, deliberate: a resident value that declares *no* device
residency (a CPU-only model) does not pin - it routes by policy/default
and will fail in the wrong child. Owner-tagging every resident envelope
is the engine-level ResourceHandle's job, not a device-string heuristic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import cast

from dinkster_protocol import (
    Invocation,
    InvocationResult,
    LazyStatusInvocation,
    LazyStatusResult,
    LazyStatusWorker,
    NodeError,
    OnInvocationEvent,
    Worker,
)
from dinkster_values import RESOURCES_META_KEY, Value, iter_value_tree

PlacePolicy = Callable[[Invocation], str]
"""Names the worker for an invocation no input has pinned."""


def resident_devices(inputs: Mapping[str, Value]) -> frozenset[str]:
    """Return every device named by value residency metadata."""
    devices: set[str] = set()
    for top in inputs.values():
        for value in iter_value_tree(top):
            resources = value.meta.get(RESOURCES_META_KEY)
            if not isinstance(resources, Mapping):
                continue
            for instance in cast("Mapping[str, object]", resources).values():
                if isinstance(instance, str):
                    devices.add(instance)
                else:
                    devices.update(cast("Iterable[str]", instance))
    return frozenset(devices)


class PlacementWorker:
    def __init__(
        self,
        workers: Mapping[str, Worker],
        *,
        devices: Mapping[str, str],
        place: PlacePolicy | None = None,
        default: str | None = None,
    ) -> None:
        """``workers`` maps name -> Worker; ``devices`` maps parent-namespace
        device ("cuda:1") -> owning worker name. Every device owner and the
        default must name a known worker - misconfiguration fails at
        construction, not mid-workflow."""
        self._workers = dict(workers)
        for device, name in devices.items():
            if name not in self._workers:
                raise KeyError(f"device {device!r} names unknown worker {name!r}")
        if default is not None and default not in self._workers:
            raise KeyError(f"default names unknown worker {default!r}")
        self._devices = dict(devices)
        self._place = place
        self._default = default

    async def prepare(self, node_types: Sequence[str]) -> None:
        # Homogeneous by contract: every worker must be able to host every
        # node type, because residency can pin any invocation to any of them.
        for worker in self._workers.values():
            await worker.prepare(node_types)

    def _pinned_devices(self, invocation: Invocation) -> set[str]:
        return set(resident_devices(invocation.inputs))

    def _error(self, invocation: Invocation, message: str) -> InvocationResult:
        return InvocationResult(
            error=NodeError(
                node_id=invocation.node_id,
                node_type=invocation.node_type,
                message=message,
            )
        )

    def _worker_for(self, invocation: Invocation) -> Worker | InvocationResult:
        devices = self._pinned_devices(invocation)
        unowned = sorted(d for d in devices if d not in self._devices)
        if unowned:
            return self._error(
                invocation,
                f"inputs are resident on device(s) {', '.join(unowned)} that "
                "no configured worker owns",
            )
        owners = {self._devices[d] for d in devices}
        if len(owners) > 1:
            return self._error(
                invocation,
                "inputs are resident on different workers "
                f"({', '.join(sorted(owners))}); cross-worker transfer of "
                "resident state does not exist - co-locate the producers",
            )
        if owners:
            name = next(iter(owners))
        elif self._place is not None:
            name = self._place(invocation)
            if name not in self._workers:
                return self._error(invocation, f"placement policy named unknown worker {name!r}")
        elif self._default is not None:
            name = self._default
        else:
            return self._error(
                invocation,
                "no input pins this invocation and no placement policy or "
                "default worker is configured",
            )
        return self._workers[name]

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        worker = self._worker_for(invocation)
        if isinstance(worker, InvocationResult):
            return worker
        return await worker.invoke(invocation, on_event=on_event)

    async def check_lazy_status(
        self,
        invocation: LazyStatusInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> LazyStatusResult:
        ordinary = Invocation(
            invocation_id=invocation.request_id,
            node_id=invocation.node_id,
            node_type=invocation.node_type,
            inputs=invocation.available_inputs,
            effective_schema=invocation.effective_schema,
            connected_undemanded_inputs=invocation.connected_undemanded_inputs,
            executor=invocation.executor,
            arm=invocation.arm,
            expected_execution_identity=invocation.expected_execution_identity,
            fp8_matmul=invocation.fp8_matmul,
            diffusion_dtype=invocation.diffusion_dtype,
            text_dtype=invocation.text_dtype,
            vae_dtype=invocation.vae_dtype,
            attention_policy=invocation.attention_policy,
            attention_route_token=invocation.attention_route_token,
            extension_snapshot_digest=invocation.extension_snapshot_digest,
        )
        worker = self._worker_for(ordinary)
        if isinstance(worker, InvocationResult):
            assert worker.error is not None
            return LazyStatusResult(error=worker.error)
        if not callable(getattr(worker, "check_lazy_status", None)):
            return LazyStatusResult(
                error=NodeError(
                    invocation.node_id,
                    invocation.node_type,
                    "lazy-protocol-skew: placed worker does not support lazy status",
                )
            )
        return await cast("LazyStatusWorker", worker).check_lazy_status(
            invocation, on_event=on_event
        )
