"""Inspect public acquisition receipts and current P2P grants without networking."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from dinkster_assets import (
    P2P_GRANT_VERSION,
    AssetError,
    P2PGrantReconciler,
    ProvenanceStore,
    PublicAcquisitionReceiptStore,
    ResolverSubscriptionStore,
)


def _vault_path(root: Path, digest: str) -> Path | None:
    digest_hex = digest.removeprefix("blake3:")
    path = root / "vault" / digest_hex[:2] / digest_hex
    return path if path.is_file() else None


def _report(root: Path, *, enabled: bool, region: str) -> dict[str, object]:
    receipts = PublicAcquisitionReceiptStore(root / "public-acquisition-receipts.json")
    with tempfile.TemporaryDirectory(prefix="dinkster-p2p-diagnostics-") as temporary:
        subscriptions = ResolverSubscriptionStore(
            root / "resolver-indexes.json",
            ProvenanceStore(Path(temporary) / "provenance.json"),
            region=region,
        )
        declarations = subscriptions.public_swarm_declarations()
    reconciliation = P2PGrantReconciler(clock=time.time).reconcile(
        declarations,
        receipts.records(),
        lambda digest: _vault_path(root, digest),
        enabled=enabled,
    )
    snapshot = reconciliation.snapshot
    return {
        "version": P2P_GRANT_VERSION,
        "enabled": snapshot.enabled,
        "declarationCount": len(declarations),
        "receipts": [receipt.to_wire() for receipt in receipts.records()],
        "publicGrants": [grant.to_wire() for grant in snapshot.public_grants],
        "seedGrants": [grant.to_wire() for grant in snapshot.seed_grants],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--region", default="")
    parser.add_argument(
        "--enabled",
        action="store_true",
        help="evaluate enabled grants without saving settings or starting networking",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = _report(args.library_root, enabled=args.enabled, region=args.region)
    except (AssetError, OSError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    print(f"P2P enabled for evaluation: {str(report['enabled']).lower()}")
    print(f"Declarations: {report['declarationCount']}")
    print(f"Public acquisition receipts: {len(cast('Sequence[object]', report['receipts']))}")
    print(f"Public swarm grants: {len(cast('Sequence[object]', report['publicGrants']))}")
    print(f"Seed grants: {len(cast('Sequence[object]', report['seedGrants']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
