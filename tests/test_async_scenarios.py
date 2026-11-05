from __future__ import annotations

import asyncio

import pytest

from tests.async_scenarios import run_scenario


def test_deadline_names_the_test_and_cleans_up_before_next_scenario() -> None:
    cleaned_up = False

    async def hang() -> None:
        nonlocal cleaned_up
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up = True

    with pytest.raises(
        AssertionError,
        match="test_deadline_names_the_test_and_cleans_up_before_next_scenario.*deadline",
    ):
        run_scenario(hang(), timeout=0.01)
    assert cleaned_up

    async def result() -> int:
        return 37

    assert run_scenario(result()) == 37


def test_scenario_timeout_error_is_not_misreported_as_deadline() -> None:
    error = TimeoutError("upstream operation")

    async def scenario() -> None:
        raise error

    with pytest.raises(TimeoutError) as caught:
        run_scenario(scenario())
    assert caught.value is error
