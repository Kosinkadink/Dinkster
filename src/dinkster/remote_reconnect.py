"""Automatic redial of configured remote workers.

A remote's connection is not its identity: the daemon owns its own
lifetime, so a network blip, a daemon restart, or a daemon that was not
yet up when the engine started must not permanently degrade the engine.
One supervisor task per configured remote watches its composed session
and redials with bounded exponential backoff:

- never composed (the startup dial failed): retries ``add_remote`` until
  the daemon appears.
- composed but dead: retries ``reattach_remote`` - dial-new-first, full
  hello revalidation, one atomic swap - so a daemon redeployed with a
  different announced surface reattaches with its new surface.

Failure stays loud and conservative: admitted invocations can rebind only to
the same daemon process within its reconnect grace, placement hints naming a
disconnected worker keep failing at submission, and the
composition report row stays "failed" while redials fail. A reattach to
a RESTARTED daemon process (changed instance token) clears the engine
result cache: cache keys fingerprint schema signature + inputs, never
implementation (hazard H4), and the new process may run different code
behind an unchanged signature - the reload precedent. A transport-only
reconnect to the same process keeps the cache.

Umbrella-owned wiring like reload_api: the composer lives in dinkster-serve,
dinkster-server knows nothing about workers, and this module is the seam.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass

from dinkster_schema import core_logger
from dinkster_server import ServerState

from .compose import PackDelta, RemoteReattachResult, ServingComposer
from .remotes import RemoteSpec

__all__ = [
    "ReconnectPolicy",
    "RemoteReconnectSupervisor",
    "announce_remote_delta",
    "apply_reattach_result",
    "backoff_delays",
]

_log = core_logger("reconnect")


@dataclass(frozen=True)
class ReconnectPolicy:
    """Backoff for one remote's redial attempts. The first attempt after
    a loss is immediate (a blip should reconnect fast); afterwards the
    wait doubles from ``initial_delay`` to ``max_delay``, stretched by up
    to ``jitter`` so a fleet of engines does not dial a recovering daemon
    in lockstep."""

    initial_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.2
    poll_interval: float = 1.0


def backoff_delays(
    policy: ReconnectPolicy, rand: Callable[[], float] = random.random
) -> Iterator[float]:
    """Yield the waits between consecutive failed redial attempts."""
    delay = policy.initial_delay
    while True:
        yield delay * (1.0 + policy.jitter * rand())
        delay = min(delay * 2.0, policy.max_delay)


async def announce_remote_delta(state: ServerState, delta: PackDelta) -> int:
    """Publish a first-time remote composition to the live surface - the
    exact announcement shape of the startup add_remote path."""
    validation = await state.prepare_replace((), (), delta.schemas, delta.packs, delta.node_packs)
    return state.replace(
        (),
        (),
        delta.schemas,
        delta.packs,
        delta.node_packs,
        execution_arms=delta.execution_arms,
        remove_choices=tuple(delta.derived_choices),
        choices={**delta.choices, **delta.derived_choices},
        lazy_choices=delta.lazy_choices,
        schema_owners=delta.schema_owners,
        choice_owners=delta.choice_owners,
        compat_skips=delta.compat_skips,
        _validation=validation,
    )


async def apply_reattach_result(state: ServerState, outcome: RemoteReattachResult) -> int:
    """Publish a reattach swap to the live surface - apply_reload's shape.

    Clears the engine result cache only when the daemon process changed:
    keys fingerprint schema signature + inputs, never implementation
    (hazard H4), so a restarted daemon could stale-hit behind an unchanged
    signature. A transport-only reconnect keeps every cached result."""
    result = outcome.result
    validation = await state.prepare_replace(
        result.removed_types,
        result.removed_packs,
        result.delta.schemas,
        result.delta.packs,
        result.delta.node_packs,
    )
    epoch = state.replace(
        result.removed_types,
        result.removed_packs,
        result.delta.schemas,
        result.delta.packs,
        result.delta.node_packs,
        execution_arms=result.delta.execution_arms,
        remove_choices=(*result.removed_choices, *result.delta.derived_choices),
        choices={**result.delta.choices, **result.delta.derived_choices},
        lazy_choices=result.delta.lazy_choices,
        schema_owners=result.delta.schema_owners,
        choice_owners=result.delta.choice_owners,
        remove_compat_skips=result.removed_compat_skips,
        compat_skips=result.delta.compat_skips,
        _validation=validation,
    )
    if not outcome.same_instance:
        clear = getattr(state.engine.cache, "clear", None)
        cleared = clear() if callable(clear) else 0
        _log.info(
            "remote worker %s daemon restarted: %d cache entries cleared",
            result.pack,
            cleared,
        )
    return epoch


class RemoteReconnectSupervisor:
    """One watch task per configured remote, run after startup composition.

    Each task polls the composed session's liveness; on loss it marks the
    composition report row failed and redials forever with backoff. The
    composer serializes every mutation, so concurrent watch tasks and
    other composition changes never interleave a swap."""

    def __init__(
        self,
        state: ServerState,
        composer: ServingComposer,
        specs: Sequence[RemoteSpec],
        keys: Sequence[str],
        *,
        policy: ReconnectPolicy | None = None,
    ) -> None:
        self._state = state
        self._composer = composer
        self._specs = tuple(specs)
        self._keys = tuple(keys)
        self._policy = policy if policy is not None else ReconnectPolicy()

    async def run(self) -> None:
        async with asyncio.TaskGroup() as group:
            for spec, key in zip(self._specs, self._keys, strict=True):
                group.create_task(self._watch(spec, key))

    async def _watch(self, spec: RemoteSpec, key: str) -> None:
        while True:
            if self._composer.remote_connected(spec.name):
                await asyncio.sleep(self._policy.poll_interval)
                continue
            if self._composer.remote_connected(spec.name) is False:
                # A composed session died; the startup-failure case
                # already narrated its own row.
                self._state.mark_pack_failed(
                    key, f"connection to remote:{spec.host}:{spec.port} lost; reconnecting"
                )
                _log.warning("remote worker %s connection lost; reconnecting", spec.name)
            await self._redial_until_connected(spec, key)

    async def _redial_until_connected(self, spec: RemoteSpec, key: str) -> None:
        delays = backoff_delays(self._policy)
        # Seed dedup from the report row so a retry failing the same way
        # as the already-recorded startup dial (or loss transition) does
        # not republish the identical failure event.
        row = self._state.composition_packs.get(key)
        last_error = row.get("error") if row is not None and row.get("state") == "failed" else None
        while True:
            try:
                async with self._composer.publication_transaction():
                    if self._composer.remote_connected(spec.name) is None:
                        delta = await self._composer.add_remote(spec)
                        publication = announce_remote_delta(self._state, delta)
                    else:
                        outcome = await self._composer.reattach_remote(spec)
                        publication = apply_reattach_result(self._state, outcome)

                    async def finish_redial(publication: Awaitable[int]) -> int:
                        epoch = await publication
                        self._state.mark_pack_announced(key, epoch)
                        _log.info("remote worker %s reattached (epoch %d)", spec.name, epoch)
                        return epoch

                    await self._composer.finish_publication(finish_redial(publication))
            except Exception as exc:
                # The row stores redacted text; compare like with like
                # (redaction is a no-op on already-redacted text).
                detail = self._state.redactor.redact_text(str(exc))
                if detail != last_error:
                    # Update the report row only when the failure mode
                    # changes, so steady retries do not spam events.
                    self._state.mark_pack_failed(key, detail)
                    _log.warning("remote worker %s redial failed: %s", spec.name, detail)
                    last_error = detail
                await asyncio.sleep(next(delays))
                continue
            return
