from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import Literal, cast

import pytest
from dinkster_assets import (
    AssetError,
    AssetVault,
    TransferStatus,
    TransportCandidate,
    TransportHealth,
    TransportKind,
    TransportResolver,
    digest_bytes,
    rank_transport_candidates,
)

ASSET_BYTES = b"transport resolver fixture" * 32
ASSET_DIGEST = digest_bytes(ASSET_BYTES)
FakeStatus = TransferStatus | Literal["stalled"]


class StaticResolver:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.calls = 0

    def resolve(self, digest: str) -> Path | None:
        self.calls += 1
        return self.path


class FakeBackend:
    def __init__(
        self,
        vault: AssetVault,
        statuses: dict[str, FakeStatus] | None = None,
        *,
        before_complete: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.vault = vault
        self.statuses = statuses or {}
        self.before_complete = before_complete
        self.attempts: list[tuple[str, float]] = []

    async def transfer(
        self,
        digest: str,
        candidate: TransportCandidate,
        *,
        stall_timeout: float,
    ) -> AsyncGenerator[TransferStatus]:
        self.attempts.append((candidate.source, stall_timeout))
        status = self.statuses.get(candidate.source, "complete")
        if status == "stalled":
            await asyncio.Event().wait()
            return
        if status == "complete":
            if self.before_complete is not None:
                await self.before_complete()
            with self.vault.writer(digest) as writer:
                writer.write(ASSET_BYTES)
                writer.commit()
        yield status


def _candidate(
    source: str,
    *,
    kind: TransportKind = "http",
    region: str = "",
    health: TransportHealth = "healthy",
) -> TransportCandidate:
    return TransportCandidate(
        kind=kind,
        source=source,
        region=region,
        health=health,
    )


def _resolver(
    tmp_path: Path,
    *,
    local: StaticResolver | None = None,
    vault: AssetVault | None = None,
    known: Sequence[TransportCandidate] = (),
    backend: FakeBackend | None = None,
    validate: Callable[[str, TransportCandidate | None, Path], bool] | None = None,
    transfer_stall_timeout: float = 0.05,
) -> tuple[TransportResolver, FakeBackend, AssetVault]:
    selected_vault = vault or AssetVault(tmp_path / "vault")
    selected_backend = backend or FakeBackend(selected_vault)

    resolver = TransportResolver(
        local or StaticResolver(),
        selected_vault,
        lambda _digest: known,
        selected_backend,
        validate or (lambda _digest, _candidate, path: path.read_bytes() == ASSET_BYTES),
        preferred_region="us-east",
        transfer_stall_timeout=transfer_stall_timeout,
    )
    return resolver, selected_backend, selected_vault


def _resolve(resolver: TransportResolver) -> Path | None:
    return asyncio.run(resolver.resolve(ASSET_DIGEST))


def test_transport_ranking_is_exact_and_stable() -> None:
    candidates = (
        _candidate("broken", health="broken"),
        _candidate("global-http"),
        _candidate("degraded", health="degraded"),
        _candidate("preferred-1", region="us-east"),
        _candidate("out-of-region", region="eu-west"),
        _candidate("preferred-2", region="us-east"),
    )
    ranked = rank_transport_candidates(candidates, preferred_region="us-east")
    assert [candidate.source for candidate in ranked] == [
        "preferred-1",
        "preferred-2",
        "global-http",
        "out-of-region",
        "degraded",
        "broken",
    ]


def test_transport_ranking_deduplicates_one_source_without_reordering() -> None:
    candidates = (
        _candidate("first", health="degraded"),
        _candidate("second", health="degraded"),
        _candidate("first", health="broken"),
    )
    assert [row.source for row in rank_transport_candidates(candidates)] == ["first", "second"]


@pytest.mark.parametrize(
    "candidate",
    [
        lambda: _candidate("", kind="http"),
        lambda: _candidate("  https://mirror.example/model"),
        lambda: _candidate("lead", health=cast("TransportHealth", "unknown")),
        lambda: _candidate("lead", kind=cast("TransportKind", "ftp")),
    ],
)
def test_transport_candidate_rejects_ambiguous_facts(
    candidate: Callable[[], TransportCandidate],
) -> None:
    with pytest.raises(AssetError):
        candidate()


def test_local_and_vault_hits_do_not_transfer(tmp_path: Path) -> None:
    local_path = tmp_path / "local.safetensors"
    local_path.write_bytes(ASSET_BYTES)
    local = StaticResolver(local_path)
    resolver, backend, vault = _resolver(
        tmp_path,
        local=local,
        known=(_candidate("https://unused.example/model"),),
    )
    assert _resolve(resolver) == local_path
    assert backend.attempts == []

    local.path = None
    with vault.writer(ASSET_DIGEST) as writer:
        writer.write(ASSET_BYTES)
        vault_path = writer.commit()
    assert _resolve(resolver) == vault_path
    assert backend.attempts == []


def test_authoritative_vault_precedes_read_only_local_fallback(tmp_path: Path) -> None:
    local_path = tmp_path / "local.safetensors"
    local_path.write_bytes(ASSET_BYTES)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(ASSET_DIGEST) as writer:
        writer.write(ASSET_BYTES)
        vault_path = writer.commit()
    resolver, backend, _ = _resolver(
        tmp_path,
        local=StaticResolver(local_path),
        vault=vault,
        backend=FakeBackend(vault),
    )
    assert _resolve(resolver) == vault_path
    assert backend.attempts == []


def test_http_stall_falls_back_to_remaining_healthy_http(tmp_path: Path) -> None:
    resolver, backend, _vault = _resolver(
        tmp_path,
        known=(
            _candidate("preferred", region="us-east"),
            _candidate("fallback"),
        ),
    )
    backend.statuses["preferred"] = "stalled"
    path = _resolve(resolver)
    assert path is not None and path.read_bytes() == ASSET_BYTES
    assert backend.attempts == [("preferred", 0.05), ("fallback", 0.05)]


def test_http_failure_falls_back_to_remaining_healthy_http(tmp_path: Path) -> None:
    resolver, backend, _vault = _resolver(
        tmp_path,
        known=(
            _candidate("preferred", region="us-east"),
            _candidate("https://mirror.example/model"),
        ),
    )
    backend.statuses["preferred"] = "failed"
    path = _resolve(resolver)
    assert path is not None and path.read_bytes() == ASSET_BYTES
    assert [source for source, _timeout in backend.attempts] == [
        "preferred",
        "https://mirror.example/model",
    ]


def test_transfer_stall_deadline_resets_on_real_progress(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")

    class ProgressBackend(FakeBackend):
        async def transfer(
            self,
            digest: str,
            candidate: TransportCandidate,
            *,
            stall_timeout: float,
        ) -> AsyncGenerator[TransferStatus]:
            self.attempts.append((candidate.source, stall_timeout))
            for _ in range(3):
                await asyncio.sleep(0.03)
                yield "progress"
            with self.vault.writer(digest) as writer:
                writer.write(ASSET_BYTES)
                writer.commit()
            yield "complete"

    backend = ProgressBackend(vault)
    resolver, _, _ = _resolver(
        tmp_path,
        vault=vault,
        known=(_candidate("slow-active-source"),),
        backend=backend,
    )
    assert _resolve(resolver) == vault.resolve(ASSET_DIGEST)


def test_known_transport_failure_returns_missing(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    backend = FakeBackend(vault)

    def unavailable(_digest: str) -> Sequence[TransportCandidate]:
        raise OSError("source list unavailable")

    resolver = TransportResolver(
        StaticResolver(),
        vault,
        unavailable,
        backend,
        lambda _digest, _candidate, path: path.read_bytes() == ASSET_BYTES,
        transfer_stall_timeout=0.05,
    )
    assert _resolve(resolver) is None
    assert backend.attempts == []


def test_reported_completion_requires_canonical_validated_vault_object(tmp_path: Path) -> None:
    resolver, backend, _vault = _resolver(
        tmp_path,
        known=(_candidate("unverified"),),
        validate=lambda _digest, _candidate, _path: False,
    )
    assert _resolve(resolver) is None
    assert backend.attempts == [("unverified", 0.05)]
    assert _vault.has(ASSET_DIGEST)
    assert _resolve(resolver) is None
    assert backend.attempts == [("unverified", 0.05), ("unverified", 0.05)]


def test_invalid_published_completion_stops_without_deleting_canonical_object(
    tmp_path: Path,
) -> None:
    resolver, backend, vault = _resolver(
        tmp_path,
        known=(_candidate("bad"), _candidate("good")),
        validate=lambda _digest, candidate, _path: (
            candidate is not None and candidate.source == "good"
        ),
    )
    assert _resolve(resolver) is None
    assert vault.resolve(ASSET_DIGEST) is not None
    assert [attempt[0] for attempt in backend.attempts] == ["bad"]


def test_one_writer_per_digest_and_waiters_reuse_result(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")

    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        waiting = asyncio.Event()

        class WaitingResolver(StaticResolver):
            def resolve(self, digest: str) -> Path | None:
                waiting.set()
                return super().resolve(digest)

        async def pause_writer() -> None:
            entered.set()
            await asyncio.wait_for(release.wait(), 2)

        backend = FakeBackend(vault, before_complete=pause_writer)
        resolvers = tuple(
            TransportResolver(
                local,
                vault,
                lambda _digest: (_candidate("only-source"),),
                backend,
                lambda _digest, _candidate, path: path.read_bytes() == ASSET_BYTES,
            )
            for local in (StaticResolver(), WaitingResolver())
        )
        first = asyncio.create_task(resolvers[0].resolve(ASSET_DIGEST))
        await asyncio.wait_for(entered.wait(), 2)
        second = asyncio.create_task(resolvers[1].resolve(ASSET_DIGEST))
        try:
            await asyncio.wait_for(waiting.wait(), 2)
            assert not second.done()
            assert len(backend.attempts) == 1
        finally:
            release.set()
        results = await asyncio.wait_for(asyncio.gather(first, second), 2)
        assert len(backend.attempts) == 1
        assert results[0] == results[1]
        assert results[0] is not None and results[0].read_bytes() == ASSET_BYTES

    asyncio.run(scenario())


def test_one_writer_per_digest_across_event_loop_threads(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    entered = threading.Event()
    release = threading.Event()
    attempts_lock = threading.Lock()
    active = 0
    maximum_active = 0

    class ThreadBackend(FakeBackend):
        async def transfer(
            self,
            digest: str,
            candidate: TransportCandidate,
            *,
            stall_timeout: float,
        ) -> AsyncGenerator[TransferStatus]:
            nonlocal active, maximum_active
            with attempts_lock:
                active += 1
                maximum_active = max(maximum_active, active)
                self.attempts.append((candidate.source, stall_timeout))
            entered.set()
            await asyncio.to_thread(release.wait)
            with self.vault.writer(digest) as writer:
                writer.write(ASSET_BYTES)
                writer.commit()
            with attempts_lock:
                active -= 1
            yield "complete"

    backend = ThreadBackend(vault)
    resolvers = tuple(
        _resolver(
            tmp_path,
            vault=vault,
            known=(_candidate("only-source"),),
            backend=backend,
            transfer_stall_timeout=2.0,
        )[0]
        for _ in range(2)
    )
    results: list[Path | None] = []
    errors: list[BaseException] = []

    def resolve(resolver: TransportResolver) -> None:
        try:
            results.append(asyncio.run(resolver.resolve(ASSET_DIGEST)))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=resolve, args=(resolvers[0],))
    second = threading.Thread(target=resolve, args=(resolvers[1],))
    first.start()
    assert entered.wait(2)
    second.start()
    try:
        second.join(0.05)
        assert second.is_alive()
        assert len(backend.attempts) == 1
    finally:
        release.set()
        first.join(2)
        second.join(2)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert len(backend.attempts) == 1
    assert maximum_active == 1
    assert results == [vault.resolve(ASSET_DIGEST), vault.resolve(ASSET_DIGEST)]
