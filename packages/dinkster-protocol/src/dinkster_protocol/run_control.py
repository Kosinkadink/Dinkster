"""Suspend invocation admission while a run-scoped value is awaited."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager

from dinkster_schema.execution import RunExecutionContext
from dinkster_values import Value


class InvocationLease:
    def __init__(self, acquire: Callable[[], AbstractAsyncContextManager[None]]) -> None:
        self._acquire = acquire
        self._held: AbstractAsyncContextManager[None] | None = None
        self._active = False
        self._wait_lock = asyncio.Lock()

    async def __aenter__(self) -> InvocationLease:
        held = self._acquire()
        await held.__aenter__()
        self._held = held
        self._active = True
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self._active = False
        held, self._held = self._held, None
        if held is not None:
            await held.__aexit__(None, None, None)

    def check_active(self) -> None:
        if not self._active:
            raise RuntimeError("execution context is no longer active")

    async def _resume(self) -> None:
        self.check_active()
        held = self._acquire()
        await held.__aenter__()
        if not self._active:
            await held.__aexit__(None, None, None)
            self.check_active()
        self._held = held

    async def wait(self, context: RunExecutionContext, node_id: str) -> Mapping[str, Value]:
        async with self._wait_lock:
            self.check_active()
            held, self._held = self._held, None
            assert held is not None
            await held.__aexit__(None, None, None)
            try:
                result = await context.await_run_value(node_id)
            except asyncio.CancelledError:
                raise
            except BaseException:
                await self._resume()
                raise
            await self._resume()
            return result

    def bind(self, context: RunExecutionContext) -> LeasedRunContext:
        return LeasedRunContext(context, self)


class LeasedRunContext:
    def __init__(self, context: RunExecutionContext, lease: InvocationLease) -> None:
        self._context = context
        self._lease = lease

    @property
    def node_id(self) -> str:
        return self._context.node_id

    @property
    def inputs(self) -> Mapping[str, Value]:
        return self._context.inputs

    async def get_graph(self) -> Mapping[str, object]:
        self._lease.check_active()
        return await self._context.get_graph()

    async def await_run_value(self, node_id: str) -> Mapping[str, Value]:
        return await self._lease.wait(self._context, node_id)

    async def publish_run_value(self, node_id: str, value: Mapping[str, Value]) -> None:
        self._lease.check_active()
        await self._context.publish_run_value(node_id, value)

    async def inhibit(self, node_ids: Sequence[str]) -> None:
        self._lease.check_active()
        await self._context.inhibit(node_ids)

    async def override_node(self, node_id: str, inputs: Mapping[str, object]) -> None:
        self._lease.check_active()
        await self._context.override_node(node_id, inputs)
