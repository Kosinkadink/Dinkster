"""Release channel authentication does not depend on its mirror or mutable selections."""

from __future__ import annotations

import base64
import hashlib
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from dinkster_supervisor.environment_channel_signatures import sign_channel, verify_channel
from dinkster_supervisor.environment_state import EnvironmentError


def _payload() -> dict[str, object]:
    return {
        "format": "dinkster.channel/1",
        "channel": "nightly",
        "engine": {"version": "0.0.8.dev17", "release": "sha256:" + "3" * 64},
        "minimumDesktopVersion": "0.2.5",
        "extensions": {"\U00010000": "\u00e9\n", "\ue000": 9_007_199_254_740_991},
    }


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _signed(payload: dict[str, object], key: Ed25519PrivateKey) -> tuple[bytes, dict[str, bytes]]:
    public = key.public_key().public_bytes_raw()
    key_id = "sha256:" + hashlib.sha256(public).hexdigest()
    envelope = {
        "signed": payload,
        "signatures": [
            {
                "keyId": key_id,
                "signature": base64.b64encode(key.sign(_canonical(payload))).decode("ascii"),
            }
        ],
    }
    return _canonical(envelope), {key_id: public}


def test_channel_signatures_cover_received_fields_and_do_not_mutate_selection():
    key = Ed25519PrivateKey.generate()
    payload = _payload()
    independent, keys = _signed(payload, key)
    assert sign_channel(payload, "nightly", key) == independent
    verified = verify_channel(independent, "nightly", keys=keys)
    assert verified == payload
    overlay = {**verified, "frontend": {"version": "0.2.7"}}
    assert verify_channel(independent, "nightly", keys=keys) == payload
    assert overlay != payload
    assert payload == _payload()
    # No built-in test key or feed-provided key authorizes the same bytes.
    with pytest.raises(EnvironmentError):
        verify_channel(independent, "nightly")


@pytest.mark.parametrize(
    "attack",
    [
        "wrong-channel",
        "tampered-engine",
        "unknown-key",
        "unsigned",
        "duplicate-field",
        "pretty-json",
    ],
)
def test_channel_authentication_refuses_substitution_and_ambiguous_metadata(attack: str):
    key = Ed25519PrivateKey.generate()
    payload = _payload()
    data, keys = _signed(payload, key)
    channel = "nightly"
    if attack == "wrong-channel":
        channel = "stable"
    elif attack == "tampered-engine":
        data = data.replace(b"0.0.8.dev17", b"0.0.8.dev18")
    elif attack == "unknown-key":
        data, _ = _signed(payload, Ed25519PrivateKey.generate())
    elif attack == "unsigned":
        data = _canonical(payload)
    elif attack == "duplicate-field":
        data = data.replace(b'"channel":"nightly"', b'"channel":"stable","channel":"nightly"')
    elif attack == "pretty-json":
        data = json.dumps(json.loads(data), indent=2).encode("ascii")
    with pytest.raises(EnvironmentError):
        verify_channel(data, channel, keys=keys)


@pytest.mark.parametrize("number", [1.5, 9_007_199_254_740_992])
def test_channel_signer_refuses_numbers_that_readers_cannot_canonicalize_identically(
    number: object,
):
    payload = {**_payload(), "extension": number}
    with pytest.raises(EnvironmentError):
        sign_channel(payload, "nightly", Ed25519PrivateKey.generate())
