"""Acquisition: turning a declared need into verified local bytes
(DESIGN 3.12 + roadmap "templates/asset distribution").

One function, one contract: ``acquire_need`` resolves a need in strict
cost order - already-local, packaged, LAN, remote - and every source must
verify before vault publication, so a wrong file in a pack, a lying mirror,
or a truncated download lands NOTHING. The digest the need declared is the
only authority.

Consent lives ABOVE this function on purpose. Acquisition mutates disk
and possibly touches the network, so callers (job preflight, template
install, CLI) decide per digest whether to call it at all; the function
itself never asks. A digestless need returns "unverifiable" without
touching any source - the reserved seat for a future name-guess tier
with its own explicit acceptance, never a silent same-name fallback.

Provenance accretes on success: a remote URL that actually produced
verifying bytes is worth remembering, so it merges into the store for
the next machine that consults it.
"""

from __future__ import annotations
from dinkster_values import MEBIBYTE

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .fetch import (
    DEFAULT_FETCH_TIMEOUT,
    FetchAborted,
    fetch_asset,
    fetch_public_asset,
)
from .identity import AssetError
from .model import AssetResolver
from .need import AssetNeed, PackagedSource, RemoteSource
from .provenance import ProvenanceRecord, ProvenanceStore
from .public_acquisition import PublicAcquisitionReceiptStore, PublicAcquisitionSourceV1
from .vault import AssetVault

_COPY_CHUNK_SIZE = 8 * MEBIBYTE

AcquisitionStatus = Literal["held", "acquired", "unverifiable", "failed"]


@dataclass(frozen=True)
class AcquisitionResult:
    """What happened for one need. ``path`` is set for held/acquired;
    ``detail`` is a human-facing explanation for the other outcomes."""

    need: AssetNeed
    status: AcquisitionStatus
    path: Path | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("held", "acquired")


def _packaged_file(source: PackagedSource, pack_roots: Mapping[str, Path]) -> Path | None:
    """Resolve a packaged source to a real file, confined to its pack's
    root. The path grammar already bans '..'; the containment check makes
    escape structurally impossible even through symlinks."""
    root = pack_roots.get(source.pack)
    if root is None:
        return None
    candidate = (root / source.path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def acquire_need(
    need: AssetNeed,
    vault: AssetVault,
    *,
    resolver: AssetResolver | None = None,
    pack_roots: Mapping[str, Path] | None = None,
    provenance: ProvenanceStore | None = None,
    public_sources: Sequence[PublicAcquisitionSourceV1] = (),
    receipts: PublicAcquisitionReceiptStore | None = None,
    lan_resolve: Callable[[str], Path | None] | None = None,
    materialized: Callable[[], None] | None = None,
    timeout: float = DEFAULT_FETCH_TIMEOUT,
) -> AcquisitionResult:
    """Materialize one need, verified, in cost order.

    1. Local: the vault, then ``resolver`` (mounts, libraries) - free.
    2. Packaged: copy out of an installed pack under ``pack_roots``,
       digest-verified through the vault writer.
    3. LAN: ask the bounded P2P resolver for a verified vault object.
    4. Remote: the need's URLs plus any ``provenance`` leads for the
       digest, via ``fetch_asset`` (verified the same way).

    Success from a non-local source records the winning provenance."""
    if not need.digest:
        return AcquisitionResult(
            need,
            "unverifiable",
            detail=(
                "need declares no digest; acquisition requires content "
                "identity (name-based matching is a separate, explicitly "
                "accepted tier)"
            ),
        )

    held = vault.resolve(need.digest)
    if held is None and resolver is not None:
        held = resolver.resolve(need.digest)
    if held is not None:
        return AcquisitionResult(need, "held", path=held)

    failures: list[str] = []
    for source in need.sources:
        if not isinstance(source, PackagedSource):
            continue
        file = _packaged_file(source, pack_roots or {})
        if file is None:
            failures.append(f"packaged {source.pack}:{source.path}: not present")
            continue
        try:
            with file.open("rb") as handle, vault.writer(need.digest) as writer:
                while chunk := handle.read(_COPY_CHUNK_SIZE):
                    writer.write(chunk)
                path = writer.commit()
        except (OSError, AssetError) as exc:
            failures.append(f"packaged {source.pack}:{source.path}: {exc}")
            continue
        _record_provenance(need, provenance, note=f"packaged by {source.pack}")
        if materialized is not None:
            materialized()
        return AcquisitionResult(need, "acquired", path=path)

    if lan_resolve is not None:
        try:
            path = lan_resolve(need.digest)
        except (OSError, AssetError, TimeoutError) as exc:
            failures.append(f"LAN P2P: {exc}")
        else:
            if path is not None:
                return AcquisitionResult(need, "acquired", path=path)

    urls = [source.url for source in need.sources if isinstance(source, RemoteSource)]
    if provenance is not None:
        urls.extend(url for url in provenance.sources(need.digest) if url not in urls)
    if urls:
        if receipts is not None:
            eligible = tuple(
                source
                for source in public_sources
                if source.digest == need.digest and any(url in source.listed_urls for url in urls)
            )
            attempted: set[tuple[str, int]] = set()
            for url in urls:
                for source in eligible:
                    if url not in source.listed_urls or (url, source.size_bytes) in attempted:
                        continue
                    attempted.add((url, source.size_bytes))
                    try:
                        fetched = fetch_public_asset(
                            need.digest,
                            url,
                            source.size_bytes,
                            vault,
                            timeout=timeout,
                        )
                    except FetchAborted:
                        raise
                    except (OSError, AssetError, TimeoutError):
                        continue
                    for matching in eligible:
                        if (
                            matching.size_bytes == fetched.size_bytes
                            and fetched.listed_url in matching.listed_urls
                        ):
                            try:
                                receipts.record(matching, fetched)
                            except (OSError, AssetError):
                                pass
                    _record_provenance(need, provenance)
                    if materialized is not None:
                        materialized()
                    return AcquisitionResult(need, "acquired", path=fetched.path)
        path = fetch_asset(need.digest, urls, vault, timeout=timeout)
        if path is not None:
            _record_provenance(need, provenance)
            if materialized is not None:
                materialized()
            return AcquisitionResult(need, "acquired", path=path)
        failures.append(f"remote: all {len(urls)} candidate URL(s) failed")

    if not failures:
        failures.append("no usable sources declared or known")
    return AcquisitionResult(need, "failed", detail="; ".join(failures))


def _record_provenance(need: AssetNeed, provenance: ProvenanceStore | None, note: str = "") -> None:
    if provenance is None:
        return
    try:
        provenance.add(
            ProvenanceRecord(
                digest=need.digest,
                sources=tuple(
                    source.url for source in need.sources if isinstance(source, RemoteSource)
                ),
                note=note,
            )
        )
    except OSError:
        # Verified bytes remain valid when their advisory provenance cannot persist.
        return
