"""Cooperative deadlines for synchronous tests that run async scenarios."""

from __future__ import annotations

import asyncio
import os
import traceback
from collections.abc import Coroutine
from typing import Any, TypeVar

import pytest

T = TypeVar("T")


def run_scenario(scenario: Coroutine[Any, Any, T], *, timeout: float = 120) -> T:
    name = os.environ.get("PYTEST_CURRENT_TEST", "async scenario")

    async def bounded() -> T:
        deadline = asyncio.timeout(timeout)
        interrupted = ""
        try:
            async with deadline:
                result = await scenario
        except TimeoutError as error:
            if not deadline.expired():
                raise
            interrupted = "\n" + "".join(traceback.format_exception(error))
        else:
            if not deadline.expired():
                return result
        pytest.fail(
            f"{name}: async scenario exceeded {timeout:g}s deadline{interrupted}", pytrace=False
        )

    return asyncio.run(bounded())
