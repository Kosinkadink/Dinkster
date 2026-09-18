"""Reproduce a LAN P2P download while two hosts seed the same digest."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import socket
import struct
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from dinkster_assets import (
    AssetVault,
    ProvenanceStore,
    PublicAcquisitionReceiptStore,
    PublicAcquisitionReceiptV1,
    ResolverSubscriptionStore,
    derive_p2p_descriptor,
)
from dinkster_p2p import current_lan_policy, default_p2p_settings

from dinkster.lan_p2p import LanP2PController

PAYLOAD_SIZE = 16 * 1024 * 1024 + 257
STATUS_INTERVAL_SECONDS = 0.25
MAPPING_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class ControllerFixture:
    controller: LanP2PController
    digest: str
    source: Path


def _log(event: str, **fields: object) -> None:
    print(
        json.dumps(
            {
                "utc": datetime.now(UTC).isoformat(),
                "monotonicSeconds": time.monotonic(),
                "host": socket.gethostname(),
                "event": event,
                **fields,
            },
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def _write_source(path: Path, *, payload_size: int = PAYLOAD_SIZE) -> None:
    header = json.dumps(
        {
            "weight": {
                "dtype": "U8",
                "shape": [payload_size],
                "data_offsets": [0, payload_size],
            }
        },
        separators=(",", ":"),
    ).encode()
    block = hashlib.sha256(b"Dinkster issue 1383 distinct-host fixture").digest() * 32768
    remaining = payload_size
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        while remaining:
            chunk = block[:remaining]
            handle.write(chunk)
            remaining -= len(chunk)


def _settings(*, downloads: bool = False, seeding: bool = False) -> dict[str, object]:
    return {
        **default_p2p_settings(),
        "downloadsEnabled": downloads,
        "seedingEnabled": seeding,
        "scope": "lan-only",
    }


def _prepare_controller(root: Path, *, seeding: bool) -> ControllerFixture:
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source.safetensors"
    _write_source(source)
    derived = derive_p2p_descriptor(source)
    index_path = root / "resolver.json"
    index_path.write_text(
        json.dumps(
            {
                "dinksterResolver": 1,
                "name": "distinct-host-reproduction",
                "updated": "2026-09-11T00:00:00Z",
                "entries": [
                    {
                        "digest": derived.asset_digest,
                        "name": source.name,
                        "urls": ["https://models.example/distinct-host.safetensors"],
                        "size": derived.size,
                        "license": "apache-2.0",
                        "p2p": derived.descriptor.to_wire(),
                    }
                ],
            }
        ),
        "utf-8",
    )
    indexes = ResolverSubscriptionStore(
        root / "subscriptions.json",
        ProvenanceStore(root / "provenance.json"),
    )
    subscription = indexes.subscribe(str(index_path))
    indexes.set_p2p_trust(
        subscription.id,
        trusted_for_p2p=True,
        license_authoritative=True,
    )
    receipt_path = root / "receipts.json"
    if seeding:
        public_source = indexes.public_sources()[0]
        now = time.time()
        receipt = PublicAcquisitionReceiptV1(
            version=1,
            receipt_id=hashlib.sha256(str(root).encode()).hexdigest()[:32],
            digest=public_source.digest,
            size_bytes=public_source.size_bytes,
            source_type=public_source.source_type,
            source_id=public_source.source_id,
            source_revision=public_source.source_revision,
            listed_url=public_source.listed_urls[0],
            final_url=public_source.listed_urls[0],
            fetched_at=now,
        )
        receipt_path.write_text(
            json.dumps(
                {
                    "publicAcquisitionReceipts": 1,
                    "receipts": [receipt.to_wire()],
                }
            ),
            "utf-8",
        )
    vault = AssetVault(root / "vault")
    if seeding:
        with source.open("rb") as handle, vault.writer(derived.asset_digest) as writer:
            while chunk := handle.read(1024 * 1024):
                writer.write(chunk)
            writer.commit()
    controller = LanP2PController(
        vault=vault,
        resolver_indexes=indexes,
        receipts=PublicAcquisitionReceiptStore(receipt_path),
        local_path_for=vault.resolve,
    )
    return ControllerFixture(controller, derived.asset_digest, source)


async def _controller_status(
    controller: LanP2PController,
    event: str,
    *,
    run: int | None = None,
) -> dict[str, object]:
    status = await controller.status()
    _log(event, run=run, status=status)
    return status


async def _wait_for_mapping(fixture: ControllerFixture, *, role: str) -> None:
    deadline = time.monotonic() + MAPPING_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status = await _controller_status(fixture.controller, f"{role}-status")
        lan = status.get("lan")
        if isinstance(lan, dict) and fixture.digest in lan.get("mappedDigests", []):
            return
        await asyncio.sleep(STATUS_INTERVAL_SECONDS)
    raise RuntimeError(f"{role} did not advertise the fixture digest")


async def _mapping_sources(
    controller: LanP2PController,
    digest: str,
    expected_addresses: frozenset[str],
) -> tuple[str, ...]:
    deadline = time.monotonic() + MAPPING_TIMEOUT_SECONDS
    latest: tuple[str, ...] = ()
    while time.monotonic() < deadline:
        candidates = await controller._discover(digest)
        latest = tuple(sorted(candidate.source for candidate in candidates))
        addresses = {
            split.hostname for source in latest if (split := urlsplit(source)).hostname is not None
        }
        _log("mapping-discovery", sources=latest, addresses=sorted(addresses))
        if expected_addresses <= addresses:
            return latest
        await asyncio.sleep(STATUS_INTERVAL_SECONDS)
    raise RuntimeError(
        f"expected seeder addresses {sorted(expected_addresses)}, discovered {list(latest)}"
    )


def _transfer_observation(status: dict[str, object]) -> tuple[int, bool, set[str]]:
    sidecar = status.get("sidecar")
    if not isinstance(sidecar, dict):
        return 0, False, set()
    totals = sidecar.get("totals")
    downloaded = totals.get("downloadedBytes", 0) if isinstance(totals, dict) else 0
    leases = sidecar.get("leases")
    saw_download = isinstance(leases, list) and any(
        isinstance(lease, dict) and lease.get("kind") == "download" for lease in leases
    )
    diagnostics = sidecar.get("diagnostics")
    events = diagnostics.get("events") if isinstance(diagnostics, dict) else None
    peers = {
        str(event["address"])
        for event in events or []
        if isinstance(event, dict) and event.get("kind") == "peer_connected" and "address" in event
    }
    return downloaded if isinstance(downloaded, int) else 0, saw_download, peers


def _classify_result(
    result: Path | None,
    *,
    source: Path,
    max_downloaded: int,
    saw_download: bool,
) -> str:
    if result is not None and result.read_bytes() == source.read_bytes():
        return "complete"
    if not saw_download:
        return "discovery-failed"
    if max_downloaded == 0:
        return "zero-byte-stall"
    return "failed-after-progress"


async def _run_seeder(root: Path, duration: float) -> int:
    fixture = _prepare_controller(root, seeding=True)
    policy = current_lan_policy()
    _log(
        "seeder-start",
        digest=fixture.digest,
        sizeBytes=fixture.source.stat().st_size,
        addresses=list(policy.addresses),
        durationSeconds=duration,
    )
    await fixture.controller.start(_settings(seeding=True))
    try:
        await fixture.controller.reconcile()
        await _wait_for_mapping(fixture, role="seeder")
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            await _controller_status(fixture.controller, "seeder-status")
            await asyncio.sleep(STATUS_INTERVAL_SECONDS)
    finally:
        await fixture.controller.close()
        _log("seeder-stop")
    return 0


async def _run_fetcher(root: Path, runs: int, expected_seeders: frozenset[str]) -> int:
    local_seeder = _prepare_controller(root / "local-seeder", seeding=True)
    policy = current_lan_policy()
    _log(
        "fetch-start",
        digest=local_seeder.digest,
        sizeBytes=local_seeder.source.stat().st_size,
        addresses=list(policy.addresses),
        expectedSeeders=sorted(expected_seeders),
        runs=runs,
    )
    await local_seeder.controller.start(_settings(seeding=True))
    outcomes: list[str] = []
    try:
        await local_seeder.controller.reconcile()
        await _wait_for_mapping(local_seeder, role="local-seeder")
        for run in range(1, runs + 1):
            fixture = _prepare_controller(root / f"fetch-{run:03d}", seeding=False)
            if fixture.digest != local_seeder.digest:
                raise RuntimeError("fixture digest changed between controller sessions")
            await fixture.controller.start(_settings(downloads=True))
            max_downloaded = 0
            saw_download = False
            peer_addresses: set[str] = set()
            sources: tuple[str, ...] = ()
            try:
                await fixture.controller.reconcile()
                sources = await _mapping_sources(
                    fixture.controller,
                    fixture.digest,
                    expected_seeders,
                )
                resolution = asyncio.create_task(
                    asyncio.to_thread(fixture.controller.resolve_sync, fixture.digest)
                )
                while not resolution.done():
                    status = await _controller_status(
                        fixture.controller,
                        "fetcher-status",
                        run=run,
                    )
                    downloaded, observed_download, peers = _transfer_observation(status)
                    max_downloaded = max(max_downloaded, downloaded)
                    saw_download = saw_download or observed_download
                    peer_addresses.update(peers)
                    await asyncio.sleep(STATUS_INTERVAL_SECONDS)
                result = await resolution
                final_status = await _controller_status(
                    fixture.controller,
                    "fetcher-final-status",
                    run=run,
                )
                downloaded, observed_download, peers = _transfer_observation(final_status)
                max_downloaded = max(max_downloaded, downloaded)
                saw_download = saw_download or observed_download
                peer_addresses.update(peers)
                outcome = _classify_result(
                    result,
                    source=local_seeder.source,
                    max_downloaded=max_downloaded,
                    saw_download=saw_download,
                )
            finally:
                await fixture.controller.close()
            outcomes.append(outcome)
            _log(
                "run-result",
                run=run,
                outcome=outcome,
                maxDownloadedBytes=max_downloaded,
                peerAddresses=sorted(peer_addresses),
                mappingSources=sources,
            )
    finally:
        await local_seeder.controller.close()
    _log(
        "fetch-summary",
        outcomes=outcomes,
        complete=outcomes.count("complete"),
        zeroByteStalls=outcomes.count("zero-byte-stall"),
    )
    return 0 if outcomes == ["complete"] * runs else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="role", required=True)
    seeder = subparsers.add_parser("seed", help="advertise and seed the deterministic fixture")
    seeder.add_argument("--root", type=Path, required=True)
    seeder.add_argument("--duration", type=float, default=900.0)
    fetcher = subparsers.add_parser("fetch", help="run downloads beside a local seeder")
    fetcher.add_argument("--root", type=Path, required=True)
    fetcher.add_argument("--runs", type=int, default=20)
    fetcher.add_argument("--expected-seeder", action="append", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.role == "seed":
        if args.duration <= 0:
            raise SystemExit("--duration must be positive")
        return asyncio.run(_run_seeder(args.root, args.duration))
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    return asyncio.run(_run_fetcher(args.root, args.runs, frozenset(args.expected_seeder)))


if __name__ == "__main__":
    raise SystemExit(main())
