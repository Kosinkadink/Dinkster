"""Each user's budget of pack calls the editor starts on one backend."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from aiohttp import web
from dinkster_protocol.lookups import (
    LOOKUP_CALLS_PER_MINUTE,
    LOOKUP_CALLS_UNDER_WAY,
    LookupCallError,
)


@dataclass
class _Usage:
    under_way: int = 0
    started: deque[float] = field(default_factory=deque[float])


class CallBudget:
    """At most 8 calls under way and 60 new calls in any 60 seconds per user.

    ``admit`` counts one new call or raises ``LookupCallError("lookup-paused")``;
    the returned function ends it. A request that joins a call already under
    way is not a new call and is not admitted again."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._usage: dict[str, _Usage] = {}

    def admit(self, principal: str) -> Callable[[], None]:
        usage = self._usage.setdefault(principal, _Usage())
        now = self._clock()
        while usage.started and now - usage.started[0] >= 60.0:
            usage.started.popleft()
        if (
            usage.under_way >= LOOKUP_CALLS_UNDER_WAY
            or len(usage.started) >= LOOKUP_CALLS_PER_MINUTE
        ):
            raise LookupCallError("lookup-paused")
        usage.started.append(now)
        usage.under_way += 1
        ended = False

        def end() -> None:
            nonlocal ended
            if not ended:
                ended = True
                usage.under_way -= 1

        return end


CALL_BUDGET_KEY: web.AppKey[CallBudget] = web.AppKey("call_budget", CallBudget)
"""The one budget an app's editor-started pack calls draw from."""
