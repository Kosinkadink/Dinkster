"""DispatchWorker: one node type, several implementations (stage 6).

RoutingWorker answers "which worker implements this node type" with ONE
worker; this facade sits behind a route when the answer is "it depends" -
the same logical node executable by the Comfy compat worker or a dedicated
native worker. It never decides: the engine's plan_execution hook selected
the arm BEFORE cache lookup (so the selection is part of the cache key),
and the selection rides ``Invocation.executor``. This side only follows it
and enforces the ownership contract around it:

- an invocation arriving without a selection is a wiring error, loudly;
- every resident input's owner token (RESOURCE_OWNER_META_KEY) must resolve
  to the selected arm's opaque worker-session domain;
- producer-arm provenance must match the selected arm even when several
  arms share one session and residency domain;
- every resident OUTPUT must satisfy the same two facts, or it would poison
  the cache entry the selection's tag names.

Owner resolution is injected as a callable because tokens are lifetime
facts: a restarted session has a new token and opaque domain, and the
resolver must answer from the CURRENT session, never a remembered one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
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
from dinkster_values import (
    RESOURCE_ID_META_KEY,
    RESOURCE_OWNER_META_KEY,
    RESOURCE_PRODUCER_ARM_META_KEY,
    ListPayload,
    Value,
    ValueMeta,
    list_children,
    value_resource_provenance_refs,
)

ResidencyDomain = object
ResolveOwner = Callable[[str], ResidencyDomain | None]
"""Maps a RESOURCE_OWNER_META_KEY token to the opaque session domain whose
worker lifetime currently holds it, or None for a dead/foreign token."""


@dataclass(frozen=True)
class ArmWorker:
    """Immutable same-session body selector over one worker boundary."""

    worker: Worker
    arm: str | None
    producer_arm: str | None = None

    async def prepare(self, node_types: Sequence[str]) -> None:
        await self.worker.prepare(node_types)

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        result = await self.worker.invoke(replace(invocation, arm=self.arm), on_event=on_event)
        owner = getattr(self.worker, "instance_token", None)
        if result.outputs is None or self.producer_arm is None or not isinstance(owner, str):
            return result
        return replace(
            result,
            outputs={
                name: _qualify_producer_arm(value, owner, self.producer_arm)
                for name, value in result.outputs.items()
            },
        )

    async def check_lazy_status(
        self,
        invocation: LazyStatusInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> LazyStatusResult:
        if not callable(getattr(self.worker, "check_lazy_status", None)):
            return LazyStatusResult(
                error=NodeError(
                    invocation.node_id,
                    invocation.node_type,
                    "lazy-protocol-skew: arm worker does not support lazy status",
                )
            )
        worker = cast("LazyStatusWorker", self.worker)
        return await worker.check_lazy_status(replace(invocation, arm=self.arm), on_event=on_event)


def _qualify_producer_arm(value: Value, owner: str, arm: str) -> Value:
    children = list_children(value)
    if children is not None:
        qualified = tuple(_qualify_producer_arm(child, owner, arm) for child in children)
        if qualified != children:
            payload = value.payload
            assert isinstance(payload, ListPayload)
            value = replace(value, payload=replace(payload, children=qualified))
    if (
        isinstance(value.meta.get(RESOURCE_ID_META_KEY), str)
        and value.meta.get(RESOURCE_OWNER_META_KEY) == owner
    ):
        entries = dict(value.meta.entries)
        entries[RESOURCE_PRODUCER_ARM_META_KEY] = arm
        value = replace(value, meta=ValueMeta(entries))
    return value


class DispatchWorker:
    def __init__(
        self,
        arms: Mapping[str, Worker],
        *,
        resolve_owner: ResolveOwner,
        arm_domains: Mapping[str, ResidencyDomain],
        default_arms: Mapping[str, str],
    ) -> None:
        if not arms:
            raise ValueError("DispatchWorker needs at least one arm")
        self._arms = dict(arms)
        # The schema-owning/default body owns lazy demand decisions. Alternate
        # implementations receive the resulting resolved inputs at execution.
        self._owner_arm = next(iter(self._arms))
        self._resolve_owner = resolve_owner
        self._arm_domains = dict(arm_domains)
        self._default_arms = dict(default_arms)
        if set(self._arm_domains) != set(self._arms):
            raise ValueError("DispatchWorker arm_domains must exactly match arms")
        if set(self._default_arms) != set(self._arms):
            raise ValueError("DispatchWorker default_arms must exactly match arms")
        for arm, domain in self._arm_domains.items():
            default = self._default_arms[arm]
            if default not in self._arms or self._arm_domains[default] is not domain:
                raise ValueError("DispatchWorker default_arms must name one arm in each domain")

    async def prepare(self, node_types: Sequence[str]) -> None:
        # Every arm must be ready: prepare() runs before any per-node
        # selection exists, and residency can pin an invocation to any arm.
        for worker in self._arms.values():
            await worker.prepare(node_types)

    def _error(self, invocation: Invocation, message: str) -> InvocationResult:
        return InvocationResult(
            error=NodeError(
                node_id=invocation.node_id,
                node_type=invocation.node_type,
                message=message,
            )
        )

    def _ownership_problem(
        self,
        values: Mapping[str, Value],
        target: str,
        direction: str,
        *,
        require_producer_arm: bool = True,
    ) -> str | None:
        for name, value in values.items():
            for rid, owner, producer, producer_present in value_resource_provenance_refs(value):
                if not isinstance(owner, str):
                    return (
                        f"{direction} '{name}' references resource {rid!r} "
                        "with no owner token; dispatched node types require "
                        "owner-stamped resident envelopes"
                    )
                domain = self._resolve_owner(owner)
                if domain is None:
                    return (
                        f"{direction} '{name}' references resource {rid!r} "
                        f"owned by {owner!r}, which no live arm holds - it "
                        "belongs to an earlier worker lifetime or a process "
                        "outside this dispatcher"
                    )
                target_domain = self._arm_domains[target]
                if domain is not target_domain:
                    return (
                        f"{direction} '{name}' references resource {rid!r} "
                        f"owned by another worker session, but the selection "
                        f"targets arm '{target}' - resident state cannot cross "
                        "residency domains"
                    )
                if not require_producer_arm:
                    continue
                if producer_present:
                    if not isinstance(producer, str) or producer not in self._arms:
                        return (
                            f"{direction} '{name}' references resource {rid!r} "
                            f"with malformed or unknown producer arm {producer!r}"
                        )
                    producer_arm = producer
                else:
                    producer_arm = self._default_arms[target]
                if producer_arm != target:
                    return (
                        f"{direction} '{name}' references resource {rid!r} "
                        f"produced by arm '{producer_arm}', but the selection "
                        f"targets arm '{target}' - producer-arm affinity failed"
                    )
        return None

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        target = invocation.executor
        if target is None:
            return self._error(
                invocation,
                "invocation reached the dispatcher without an execution "
                "selection; the engine's plan_execution hook must be "
                "enrolled for this node type",
            )
        worker = self._arms.get(target)
        if worker is None:
            return self._error(
                invocation,
                f"execution selection names unknown arm {target!r} "
                f"(arms: {', '.join(sorted(self._arms))})",
            )
        problem = self._ownership_problem(invocation.inputs, target, "input")
        if problem is not None:
            return self._error(invocation, problem)
        result = await worker.invoke(invocation, on_event=on_event)
        if result.outputs is not None:
            # The cache entry this result lands under is tagged with the
            # selected arm's execution identity - a resident owned by any
            # OTHER process would poison it.
            problem = self._ownership_problem(result.outputs, target, "output")
            if problem is not None:
                return self._error(invocation, problem)
        return result

    async def check_lazy_status(
        self,
        invocation: LazyStatusInvocation,
        on_event: OnInvocationEvent | None = None,
    ) -> LazyStatusResult:
        problem = self._ownership_problem(
            invocation.available_inputs,
            self._owner_arm,
            "input",
            require_producer_arm=False,
        )
        if problem is not None:
            return LazyStatusResult(
                error=NodeError(invocation.node_id, invocation.node_type, problem)
            )
        worker = self._arms[self._owner_arm]
        if not callable(getattr(worker, "check_lazy_status", None)):
            return LazyStatusResult(
                error=NodeError(
                    invocation.node_id,
                    invocation.node_type,
                    "lazy-protocol-skew: owner arm does not support lazy status",
                )
            )
        return await cast("LazyStatusWorker", worker).check_lazy_status(
            invocation, on_event=on_event
        )
