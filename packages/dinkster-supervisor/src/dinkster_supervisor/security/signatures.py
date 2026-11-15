"""Canonical Ed25519 envelopes authenticated by independently pinned public keys."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from typing import cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate signed document field")
        result[key] = value
    return result


def parse_signed_json(data: bytes) -> dict[str, object]:
    value: object = json.loads(data, object_pairs_hook=_unique_fields)
    if not isinstance(value, dict):
        raise ValueError("signed document must be an object")
    return cast("dict[str, object]", value)


def sign_envelope(payload: dict[str, object], key: Ed25519PrivateKey) -> bytes:
    public = key.public_key().public_bytes_raw()
    return canonical(
        {
            "signed": payload,
            "signatures": [
                {
                    "keyId": "sha256:" + hashlib.sha256(public).hexdigest(),
                    "signature": base64.b64encode(key.sign(canonical(payload))).decode("ascii"),
                }
            ],
        }
    )


def verify_envelope(data: bytes, keys: dict[str, bytes]) -> dict[str, object]:
    """Authenticate received fields, not reconstructed fields with schema defaults."""
    envelope = parse_signed_json(data)
    if set(envelope) != {"signed", "signatures"}:
        raise ValueError("signed envelope must contain only signed and signatures")
    payload, signatures = envelope["signed"], envelope["signatures"]
    if not isinstance(payload, dict) or not isinstance(signatures, list) or not signatures:
        raise ValueError("signed envelope requires an object and nonempty signatures")
    payload = cast("dict[str, object]", payload)
    message = canonical(payload)
    seen: set[str] = set()
    authenticated = False
    for value in cast("list[object]", signatures):
        if not isinstance(value, dict):
            raise ValueError("invalid signature fields")
        row = cast("dict[str, object]", value)
        if set(row) != {"keyId", "signature"}:
            raise ValueError("invalid signature fields")
        key_id, encoded = row["keyId"], row["signature"]
        if (
            not isinstance(key_id, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", key_id) is None
            or key_id in seen
            or not isinstance(encoded, str)
        ):
            raise ValueError("invalid or repeated signing key")
        seen.add(key_id)
        signature = base64.b64decode(encoded, validate=True)
        if len(signature) != 64 or base64.b64encode(signature).decode("ascii") != encoded:
            raise ValueError("signature must be canonical base64 Ed25519 bytes")
        public = keys.get(key_id)
        if public is None:
            continue
        if "sha256:" + hashlib.sha256(public).hexdigest() != key_id:
            raise ValueError("pinned signing key identifier differs from its public bytes")
        try:
            Ed25519PublicKey.from_public_bytes(public).verify(signature, message)
        except InvalidSignature:
            continue
        authenticated = True
    if not authenticated:
        raise ValueError("no valid independently trusted signature")
    return payload
