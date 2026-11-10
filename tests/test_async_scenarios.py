from __future__ import annotations

import asyncio

import pytest

from tests.async_scenarios import run_scenario


@pytest.mark.parametrize("suppress_cancellation", [False, True])
def test_deadline_names_the_test_and_cleans_up_before_next_scenario(
    suppress_cancellation: bool,
) -> None:
    cleaned_up = False

    async def hang() -> None:
        nonlocal cleaned_up
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if not suppress_cancellation:
                raise
        finally:
            cleaned_up = True

    with pytest.raises(
        pytest.fail.Exception,
        match="test_deadline_names_the_test_and_cleans_up_before_next_scenario.*deadline",
    ) as caught:
        try:
            run_scenario(hang(), timeout=0.01)
        except Exception:
            pytest.fail("deadline was swallowed by an ordinary exception handler")
    assert cleaned_up
    if not suppress_cancellation:
        assert "in hang" in str(caught.value)
        assert "CancelledError" in str(caught.value)

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
