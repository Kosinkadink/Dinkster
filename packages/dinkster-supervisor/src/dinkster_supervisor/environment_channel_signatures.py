"""Authenticate release channels independently of their transport or mirror host."""

from __future__ import annotations

from typing import cast

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .environment_state import EnvironmentError
from .security.signatures import canonical, parse_signed_json, sign_envelope, verify_envelope

# Trust roots are provisioned independently of mirrors and channel contents.
CHANNEL_KEYS: dict[str, bytes] = {
    "sha256:4b5b2c78b32db1ebedc4702a92d6e4e252c17531f0e102ab82dd910be6dec7b9": bytes.fromhex(
        "9baae6ed763cd3a23a7a27d48221256da2b89d89164250d4f968ddb07f046fd4"
    ),
}


def _portable_json(value: object) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int and abs(value) <= 9_007_199_254_740_991:
        return
    if isinstance(value, list):
        for item in cast("list[object]", value):
            _portable_json(item)
        return
    if isinstance(value, dict):
        for item in cast("dict[str, object]", value).values():
            _portable_json(item)
        return
    raise EnvironmentError("channel JSON requires strings, booleans and safe integers")


def _channel_payload(document: dict[str, object], channel: str) -> None:
    if (
        channel not in {"stable", "nightly"}
        or document.get("format") != "dinkster.channel/1"
        or document.get("channel") != channel
        or "signature" in document
    ):
        raise EnvironmentError("signed channel format or requested channel differs")
    _portable_json(document)


def sign_channel(document: dict[str, object], channel: str, key: Ed25519PrivateKey) -> bytes:
    payload = dict(document)
    if payload.get("signature") is not None:
        raise EnvironmentError("cannot sign a channel with an existing legacy signature")
    payload.pop("signature", None)
    if "channel" in payload and payload["channel"] != channel:
        raise EnvironmentError("channel publication name differs from its payload")
    payload["channel"] = channel
    _channel_payload(payload, channel)
    return sign_envelope(payload, key)


def verify_channel(
    data: bytes, channel: str, *, keys: dict[str, bytes] | None = None
) -> dict[str, object]:
    """Return authenticated producer fields, never an explicit local selection overlay."""
    try:
        envelope = parse_signed_json(data)
        # Canonical transport also lets non-Python readers reject duplicate fields
        # without a second JSON parser that could disagree about signed values.
        if canonical(envelope) != data:
            raise EnvironmentError("channel envelope must use canonical ASCII JSON")
        document = verify_envelope(data, CHANNEL_KEYS if keys is None else keys)
        _channel_payload(document, channel)
        return document
    except (ValueError, TypeError, UnicodeError) as exc:
        raise EnvironmentError("channel signature verification failed") from exc
