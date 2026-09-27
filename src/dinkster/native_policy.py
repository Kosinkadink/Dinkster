"""Resident-value dispatch affinity shared by composed workers."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from typing import Protocol

from dinkster_engine import ExecutionSelection
from dinkster_protocol import AttentionPolicy, AttentionRouteToken
from dinkster_values import Value, value_resource_provenance_refs


class NativeDispatchPolicy(Protocol):
    def select(
        self,
        node_type: str,
        inputs: Mapping[str, Value],
        candidates: tuple[str, Mapping[str, str]],
        **context: object,
    ) -> Awaitable[ExecutionSelection | None]: ...

    def release_run(self, run_id: str) -> None: ...


def _attention_facts(
    routes: Mapping[str, AttentionRouteToken | None],
    arm: str,
) -> tuple[AttentionPolicy, AttentionRouteToken | None]:
    token = routes.get(arm)
    return ("auto", None) if token is None else (token.requested_policy, token)


def select_resident_producer(
    inputs: Mapping[str, Value],
    cache_tags: Mapping[str, str],
    attention_routes: Mapping[str, AttentionRouteToken | None],
) -> ExecutionSelection | None:
    producer_arms: set[str] = set()
    for value in inputs.values():
        for resource_id, _owner, producer_arm, present in value_resource_provenance_refs(value):
            if not present:
                continue
            if not isinstance(producer_arm, str):
                raise RuntimeError(
                    f"resident resource {resource_id!r} has malformed producer arm stamp"
                )
            if producer_arm not in cache_tags:
                raise RuntimeError(
                    f"resident resource {resource_id!r} names unknown producer arm {producer_arm!r}"
                )
            producer_arms.add(producer_arm)
    if not producer_arms:
        return None
    if len(producer_arms) != 1:
        raise RuntimeError(
            "resident inputs have conflicting producer arms: " + ", ".join(sorted(producer_arms))
        )
    target = next(iter(producer_arms))
    policy, token = _attention_facts(attention_routes, target)
    return ExecutionSelection(
        target=target,
        cache_tag=cache_tags[target],
        attention_policy=policy,
        attention_route_token=token,
    )


__all__ = ["NativeDispatchPolicy", "select_resident_producer"]
