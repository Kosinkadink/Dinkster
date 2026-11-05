"""Cooperative deadlines for synchronous tests that run async scenarios."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_scenario(scenario: Coroutine[Any, Any, T], *, timeout: float = 120) -> T:
    name = os.environ.get("PYTEST_CURRENT_TEST", "async scenario")

    async def bounded() -> T:
        deadline = asyncio.timeout(timeout)
        try:
            async with deadline:
                return await scenario
        except TimeoutError:
            if not deadline.expired():
                raise
            raise AssertionError(f"{name}: async scenario exceeded {timeout:g}s deadline") from None

    return asyncio.run(bounded())
