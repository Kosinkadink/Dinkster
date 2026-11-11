"""Signed notices preserve authenticated blocks and exact decisions across restarts."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from dinkster_security.documents import SecurityError
from dinkster_security.policy import Artifact, assess
from dinkster_security.store import SecurityStore

REGISTRY = "https://registry.example.test"
NOTICE = "5b2fa364-f808-4dd2-b58e-e02187f9a2cc"
VERIFIED_AT = datetime(2026, 11, 10, 12, tzinfo=UTC)


def _key(key: Ed25519PrivateKey) -> dict[str, str]:
    public = key.public_key().public_bytes_raw()
    return {
        "keyId": "sha256:" + hashlib.sha256(public).hexdigest(),
        "publicKey": base64.b64encode(public).decode(),
    }


def _trust(key: Ed25519PrivateKey, revision: int = 1) -> dict[str, Any]:
    return {
        "format": "dinkster.security-trust/1",
        "registry": REGISTRY,
        "revision": revision,
        "issuedAt": "2026-11-01T00:00:00Z",
        "expiresAt": "2026-12-01T00:00:00Z",
        "keys": [_key(key)],
    }


def _feed(state: str = "vulnerable", revision: int = 1) -> dict[str, Any]:
    return {
        "format": "dinkster.security/1",
        "registry": REGISTRY,
        "revision": revision,
        "issuedAt": "2026-11-01T00:00:00Z",
        "expiresAt": "2026-11-02T00:00:00Z",
        "notices": [
            {
                "id": NOTICE,
                "revision": 1,
                "previousRevision": None,
                "release": {
                    "pack": "test-pack",
                    "version": "1.2.3",
                    "artifactDigest": "sha256:" + "a" * 64,
                },
                "state": state,
                "status": "active",
                "actor": "test-security-role",
                "authority": "security",
                "securityConfirmed": True,
                "changedAt": "2026-11-01T00:00:00Z",
                "reason": "test classification",
                "message": "benign signed fixture",
                "incident": "test-incident",
                "fixedRelease": None,
                "fixedLockDigest": None,
                "causes": [],
            }
        ],
        "ownershipChanges": [],
    }


def _signed(payload: dict[str, Any], key: Ed25519PrivateKey) -> bytes:
    # This fixture signs raw wire fields, independently of consumer serialization.
    message = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return json.dumps(
        {
            "signed": payload,
            "signatures": [
                {
                    "keyId": _key(key)["keyId"],
                    "signature": base64.b64encode(key.sign(message.encode("ascii"))).decode(),
                }
            ],
        }
    ).encode()


def _store(tmp_path: Path, key: Ed25519PrivateKey) -> SecurityStore:
    store = SecurityStore(tmp_path / "security.sqlite3")
    store.provision_trust(json.dumps(_trust(key)).encode())
    return store


def test_cached_notice_and_changeable_exact_decision_survive_restart_and_lift(tmp_path: Path):
    key = Ed25519PrivateKey.generate()
    store = _store(tmp_path, key)
    original = _feed()
    store.import_feed(REGISTRY, _signed(original, key), now=VERIFIED_AT)
    notice = UUID(NOTICE)
    assert store.decision(REGISTRY, notice, 1) == "pending"
    store.answer(REGISTRY, notice, 1, approved=True)
    restarted = SecurityStore(store.path)
    assert restarted.decision(REGISTRY, notice, 1) == "approved"
    restarted.answer(REGISTRY, notice, 1, approved=False)
    assert store.decision(REGISTRY, notice, 1) == "denied"
    newer = copy.deepcopy(original)
    newer["revision"] = 2
    newer["notices"][0].update(revision=2, previousRevision=1, message="new finding")
    store.import_feed(REGISTRY, _signed(newer, key), now=VERIFIED_AT)
    assert store.decision(REGISTRY, notice, 2) == "pending"
    with pytest.raises(SecurityError, match="changed"):
        store.answer(REGISTRY, notice, 1, approved=True)
    newer["revision"] = 3
    newer["notices"][0].update(revision=3, previousRevision=2, status="cleared")
    store.import_feed(REGISTRY, _signed(newer, key), now=VERIFIED_AT)
    cached, last = SecurityStore(store.path).cached(REGISTRY)
    assert cached is not None and cached.notices[0].status == "cleared"
    assert last == VERIFIED_AT
    assert store.decision(REGISTRY, notice, 1) == "denied"
    with pytest.raises(SecurityError, match="active vulnerable"):
        store.answer(REGISTRY, notice, 3, approved=True)


@pytest.mark.parametrize(
    "attack",
    [
        "forged",
        "registry",
        "older",
        "dropped",
        "identity",
        "same-revision",
        "duplicate",
        "unknown",
        "bool",
    ],
)
def test_rejected_feed_preserves_cached_block_and_verification_time(tmp_path: Path, attack: str):
    key = Ed25519PrivateKey.generate()
    store = _store(tmp_path, key)
    original = _feed("malicious", revision=4)
    accepted = store.import_feed(REGISTRY, _signed(original, key), now=VERIFIED_AT)
    candidate = copy.deepcopy(original)
    candidate["revision"] = 5
    signer = key
    if attack == "forged":
        signer = Ed25519PrivateKey.generate()
    elif attack == "registry":
        candidate["registry"] = "https://other.example.test"
    elif attack == "older":
        candidate["revision"] = 3
    elif attack == "dropped":
        candidate["notices"] = []
    elif attack == "identity":
        candidate["notices"][0]["release"]["artifactDigest"] = "sha256:" + "b" * 64
    elif attack == "same-revision":
        candidate["revision"] = 4
        del candidate["notices"][0]["fixedRelease"]
    elif attack == "unknown":
        candidate["notices"][0]["continueMalicious"] = True
    elif attack == "bool":
        candidate["revision"] = True
    data = _signed(candidate, signer)
    if attack == "duplicate":
        data = data.replace(b'"revision": 5', b'"revision": 5, "revision": 6', 1)
    with pytest.raises(SecurityError):
        store.import_feed(REGISTRY, data)
    cached, last = SecurityStore(store.path).cached(REGISTRY)
    assert cached == accepted and last == VERIFIED_AT
    with pytest.raises(SecurityError, match="active vulnerable"):
        store.answer(REGISTRY, UUID(NOTICE), 1, approved=True)


def test_rotation_and_independent_recovery_preserve_blocks_decisions_and_checkpoint(tmp_path: Path):
    key, replacement, recovery = [Ed25519PrivateKey.generate() for _ in range(3)]
    store = _store(tmp_path, key)
    document = _feed()
    store.import_feed(REGISTRY, _signed(document, key), now=VERIFIED_AT)
    store.answer(REGISTRY, UUID(NOTICE), 1, approved=False)
    with pytest.raises(SecurityError):
        store.rotate_trust(REGISTRY, _signed(_trust(replacement, 2), replacement))
    store.rotate_trust(REGISTRY, _signed(_trust(replacement, 2), key))
    with pytest.raises(SecurityError):
        store.import_feed(REGISTRY, _signed(document, key))
    store.import_feed(REGISTRY, _signed(document, replacement), now=VERIFIED_AT)
    with pytest.raises(SecurityError):
        store.rotate_trust(REGISTRY, _signed(_trust(key, 1), replacement))
    store.provision_trust(json.dumps(_trust(recovery, 3)).encode(), recovery=True)
    assert store.decision(REGISTRY, UUID(NOTICE), 1) == "denied"
    cached, last = store.cached(REGISTRY)
    assert cached is not None and cached.notices[0].state == "vulnerable"
    assert last == VERIFIED_AT
    document["revision"] = 2
    store.import_feed(REGISTRY, _signed(document, recovery), now=VERIFIED_AT)
    with pytest.raises(SecurityError, match="rolled back"):
        store.import_feed(REGISTRY, _signed(_feed(), recovery))


def test_vulnerable_execution_continues_but_new_install_requires_exact_approval(tmp_path: Path):
    key = Ed25519PrivateKey.generate()
    store = _store(tmp_path, key)
    document = store.import_feed(REGISTRY, _signed(_feed(), key), now=VERIFIED_AT)
    inventory = (Artifact(REGISTRY, document.notices[0].release),)
    assert assess(store, inventory).allowed
    assert not assess(store, inventory, installing=True).allowed
    assert not assess(store, inventory, refuse_vulnerable=True).allowed
    store.answer(REGISTRY, UUID(NOTICE), 1, approved=True)
    assert assess(SecurityStore(store.path), inventory, installing=True).allowed
    assert not assess(store, inventory, refuse_vulnerable=True).allowed
    store.answer(REGISTRY, UUID(NOTICE), 1, approved=False)
    assert not assess(SecurityStore(store.path), inventory).allowed
    other = Artifact("https://another.example.test", document.notices[0].release)
    assert assess(store, (other,)).allowed
    later = VERIFIED_AT + timedelta(days=3)
    assert assess(store, (other,), now=later).allowed
    assert not assess(store, inventory, maximum_feed_age=timedelta(days=1), now=later).allowed
    assert not assess(store, (other,), maximum_feed_age=timedelta(days=1), now=later).allowed


def test_malicious_bytes_block_rehosting_and_dependencies_without_misclassifying_consumers(
    tmp_path: Path,
):
    key = Ed25519PrivateKey.generate()
    store = _store(tmp_path, key)
    feed = _feed("malicious")
    document = store.import_feed(REGISTRY, _signed(feed, key), now=VERIFIED_AT)
    bad = document.notices[0].release
    rehosted = bad.model_copy(update={"pack": "another-pack", "version": "9.8.7"})
    good = bad.model_copy(update={"artifact_digest": "sha256:" + "b" * 64})
    inventory = (
        Artifact(REGISTRY, bad),
        Artifact("https://another.example.test", rehosted),
        Artifact(REGISTRY, good, (bad.artifact_digest,)),
        Artifact(REGISTRY, good),
    )
    result = assess(SecurityStore(store.path), inventory)
    assert not result.allowed
    assert [(item.artifact, item.state, item.decision) for item in result.findings] == [
        (inventory[0], "malicious", "blocked"),
        (inventory[1], "malicious", "blocked"),
        (inventory[2], "affected", "blocked"),
    ]
    assert assess(store, (inventory[3],)).allowed
    feed["revision"] = 2
    feed["notices"][0].update(revision=2, previousRevision=1, status="cleared")
    store.import_feed(REGISTRY, _signed(feed, key), now=VERIFIED_AT)
    assert assess(store, inventory).allowed
