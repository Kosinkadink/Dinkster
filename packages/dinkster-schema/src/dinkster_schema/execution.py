"""Run-scoped control available through the hidden execution_context parameter."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from dinkster_values import Value


class RunExecutionContext(Protocol):
    @property
    def node_id(self) -> str: ...

    @property
    def inputs(self) -> Mapping[str, Value]: ...

    async def get_graph(self) -> Mapping[str, object]: ...

    async def await_run_value(self, node_id: str) -> Mapping[str, Value]: ...

    async def publish_run_value(self, node_id: str, value: Mapping[str, Value]) -> None: ...

    async def inhibit(self, node_ids: Sequence[str]) -> None: ...

    async def override_node(self, node_id: str, inputs: Mapping[str, object]) -> None: ...
