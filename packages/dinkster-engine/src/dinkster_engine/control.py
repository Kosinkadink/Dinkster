"""Live graph access and published values belong to one executing graph scope."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import cast

from dinkster_graph import Graph, LoopPair, graph_to_wire
from dinkster_values import Value


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze(item) for key, item in cast("dict[str, object]", value).items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in cast("list[object]", value))
    return value


class RunControlState:
    def __init__(self) -> None:
        self._values: dict[str, asyncio.Future[Mapping[str, Value]]] = {}
        self.loop_pairs: dict[str, LoopPair] = {}
        self.active = True

    def check_active(self) -> None:
        if not self.active:
            raise RuntimeError("run control scope is closed")

    def future(self, node_id: str) -> asyncio.Future[Mapping[str, Value]]:
        self.check_active()
        if node_id not in self._values:
            self._values[node_id] = asyncio.get_running_loop().create_future()
        return self._values[node_id]

    def close(self) -> None:
        self.active = False
        for future in self._values.values():
            if not future.done():
                future.cancel()
        self._values.clear()


class EngineRunContext:
    def __init__(
        self,
        node_id: str,
        inputs: Mapping[str, Value],
        state: RunControlState,
        graph: Callable[[], Graph],
        publish: Callable[[str, Mapping[str, Value]], None],
        inhibit: Callable[[Sequence[str]], None],
        override: Callable[[str, Mapping[str, object]], None],
    ) -> None:
        self.node_id = node_id
        self.inputs = MappingProxyType(dict(inputs))
        self._state = state
        self._graph = graph
        self._publish = publish
        self._inhibit = inhibit
        self._override = override

    def _check_target(self, node_id: str) -> None:
        self._state.check_active()
        if node_id not in self._graph().nodes:
            raise ValueError(f"unknown run control target {node_id!r}")

    async def get_graph(self) -> Mapping[str, object]:
        self._state.check_active()
        document = graph_to_wire(self._graph())
        document["loops"] = {
            start: {"end": pair.end_id, "body": list(pair.body)}
            for start, pair in self._state.loop_pairs.items()
        }
        return cast("Mapping[str, object]", _freeze(document))

    async def await_run_value(self, node_id: str) -> Mapping[str, Value]:
        self._check_target(node_id)
        return await asyncio.shield(self._state.future(node_id))

    async def publish_run_value(self, node_id: str, value: Mapping[str, Value]) -> None:
        self._check_target(node_id)
        future = self._state.future(node_id)
        if future.done():
            raise ValueError(f"run value for {node_id!r} was already published")
        self._publish(node_id, value)
        future.set_result(MappingProxyType(dict(value)))

    async def inhibit(self, node_ids: Sequence[str]) -> None:
        for node_id in node_ids:
            self._check_target(node_id)
        self._inhibit(node_ids)

    async def override_node(self, node_id: str, inputs: Mapping[str, object]) -> None:
        self._check_target(node_id)
        self._override(node_id, inputs)
