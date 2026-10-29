"""Print or verify deterministic BitTorrent v2 descriptor fixtures."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from .p2p_descriptor import (
    P2PDescriptorError,
    P2PDescriptorResult,
    P2PDescriptorV1,
    derive_p2p_descriptor,
    verify_p2p_descriptor,
)

_FIXTURE_FIELDS = {"assetDigest", "descriptor", "info", "pieceLayer", "size"}


def _fixture(result: P2PDescriptorResult) -> dict[str, object]:
    return {
        "assetDigest": result.asset_digest,
        "descriptor": result.descriptor.to_wire(),
        "info": result.info.hex(),
        "pieceLayer": result.piece_layer.hex(),
        "size": result.size,
    }


def _hex_bytes(value: object, field_name: str) -> bytes:
    if not isinstance(value, str):
        raise P2PDescriptorError(f"{field_name} must be a lowercase hex string")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise P2PDescriptorError(f"{field_name} must be a lowercase hex string") from exc
    if decoded.hex() != value:
        raise P2PDescriptorError(f"{field_name} must be a lowercase hex string")
    return decoded


def _load_fixture(path: Path) -> tuple[str, int, P2PDescriptorV1, bytes, bytes]:
    loaded: object = json.loads(path.read_text("utf-8"))
    if not isinstance(loaded, Mapping):
        raise P2PDescriptorError(f"fixture fields must be exactly {sorted(_FIXTURE_FIELDS)}")
    loaded_mapping = cast("Mapping[object, object]", loaded)
    if set(loaded_mapping) != _FIXTURE_FIELDS:
        raise P2PDescriptorError(f"fixture fields must be exactly {sorted(_FIXTURE_FIELDS)}")
    wire = cast("Mapping[str, object]", loaded)
    asset_digest = wire["assetDigest"]
    size = wire["size"]
    descriptor = wire["descriptor"]
    if not isinstance(asset_digest, str):
        raise P2PDescriptorError("assetDigest must be a string")
    if isinstance(size, bool) or not isinstance(size, int):
        raise P2PDescriptorError("size must be an integer")
    if not isinstance(descriptor, Mapping):
        raise P2PDescriptorError("descriptor must be an object")
    parsed = P2PDescriptorV1.from_wire(cast("Mapping[str, object]", descriptor))
    return (
        asset_digest,
        size,
        parsed,
        _hex_bytes(wire["info"], "info"),
        _hex_bytes(wire["pieceLayer"], "pieceLayer"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--verify", type=Path, metavar="FIXTURE")
    args = parser.parse_args(argv)

    if args.verify is None:
        result = derive_p2p_descriptor(args.asset)
        verify_p2p_descriptor(
            args.asset,
            result.descriptor,
            asset_digest=result.asset_digest,
            size=result.size,
            info=result.info,
            piece_layer=result.piece_layer,
        )
    else:
        asset_digest, size, descriptor, info, piece_layer = _load_fixture(args.verify)
        verify_p2p_descriptor(
            args.asset,
            descriptor,
            asset_digest=asset_digest,
            size=size,
            info=info,
            piece_layer=piece_layer,
        )
        result = P2PDescriptorResult(asset_digest, size, descriptor, info, piece_layer)
    print(json.dumps(_fixture(result), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
