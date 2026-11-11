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
from dinkster_registry import InstallError, LockedPack, Lockfile, build_artifact
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


@pytest.mark.parametrize("previous_publisher", [None, "original-publisher"])
def test_signed_ownership_change_requires_explicit_previous_publisher(
    tmp_path: Path, previous_publisher: str | None
):
    key = Ed25519PrivateKey.generate()
    store = _store(tmp_path, key)
    original = _feed("malicious")
    store.import_feed(REGISTRY, _signed(original, key), now=VERIFIED_AT)
    change = {
        "kind": "ownership_transferred",
        "id": "636d54c0-ef6b-44cb-9ee3-bf7381bf0ed8",
        "revision": 2,
        "packId": "9aa02405-70c5-4648-9bf8-01341233759a",
        "pack": "test-pack",
        "previousPublisher": previous_publisher,
        "publisher": "new-publisher",
        "actor": "test-security-role",
        "changedAt": "2026-11-10T12:00:00Z",
        "reason": "continue maintained releases",
    }
    replacement = copy.deepcopy(original)
    replacement.update(revision=2, ownershipChanges=[change])
    invalid = [
        {name: value for name, value in change.items() if name != "previousPublisher"},
        {**change, "publisher": ""},
        {**change, "publisher": None},
        {**change, "previousPublisher": "new-publisher"},
    ]
    for candidate in invalid:
        replacement["ownershipChanges"] = [candidate]
        with pytest.raises(SecurityError):
            store.import_feed(REGISTRY, _signed(replacement, key))
        cached, verified = SecurityStore(store.path).cached(REGISTRY)
        assert cached is not None and cached.revision == 1
        assert cached.notices[0].state == "malicious"
        assert verified == VERIFIED_AT
    replacement["ownershipChanges"] = [change]
    store.import_feed(REGISTRY, _signed(replacement, key), now=VERIFIED_AT)
    cached, verified = SecurityStore(store.path).cached(REGISTRY)
    assert cached is not None and cached.revision == 2
    assert cached.ownership_changes[0].previous_publisher == previous_publisher
    assert cached.ownership_changes[0].publisher == "new-publisher"
    assert cached.notices[0].state == "malicious"
    assert verified == VERIFIED_AT


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
    assert assess(store, inventory, maximum_feed_age=timedelta(days=10), now=VERIFIED_AT).allowed
    assert not assess(store, inventory, maximum_feed_age=timedelta(days=8), now=VERIFIED_AT).allowed
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


def test_installer_cached_blocks_cover_new_install_serving_and_retained_rollback(tmp_path: Path):
    from dinkster.installer import Installer

    installer = Installer(tmp_path / "installed", accelerator="cpu", runtime_probe=lambda _: ())
    source = tmp_path / "pack"
    source.mkdir()
    (source / "dinkster-pack.toml").write_text(
        '[pack]\nname = "test-pack"\n[pack.entry]\nnodes = "example:NODES"\n'
    )
    targets = []
    archives = {}
    for version in ("1.2.3", "2.0.0", "3.0.0"):
        (source / "example.py").write_text(f'NODES = []\nVERSION = "{version}"\n')
        archive = tmp_path / f"{version}.zip"
        digest = build_artifact(source, archive)
        entry = LockedPack(
            "test-pack", version, digest, "publisher", ("test-pack",), "registry:" + REGISTRY
        )
        targets.append(Lockfile.of([entry]))
        archives[version] = archive
    installer.apply(
        targets[0], {targets[0].packs[0].artifact_digest: archives["1.2.3"]}, venvs=False
    )
    installer.apply(
        targets[1], {targets[1].packs[0].artifact_digest: archives["2.0.0"]}, venvs=False
    )
    key = Ed25519PrivateKey.generate()
    store = SecurityStore(installer.root / "security.sqlite3")
    store.provision_trust(json.dumps(_trust(key)).encode())
    feed = _feed("malicious")
    feed["notices"][0]["release"]["artifactDigest"] = targets[0].packs[0].artifact_digest
    store.import_feed(REGISTRY, _signed(feed, key), now=VERIFIED_AT)
    current = installer.current_number()
    with pytest.raises(InstallError, match="security malicious"):
        installer.rollback(venvs=False)
    assert installer.current_number() == current
    assert archives["1.2.3"].is_file()
    spec = installer.packs_for_serving()[0]
    assert spec.packs is not None and spec.packs["test-pack"].version == "2.0.0"
    feed["revision"] = 2
    feed["notices"][0].update(revision=2, previousRevision=1, state="vulnerable")
    store.import_feed(REGISTRY, _signed(feed, key), now=VERIFIED_AT)
    installer.rollback(venvs=False)
    spec = installer.packs_for_serving()[0]
    assert spec.packs is not None and spec.packs["test-pack"].version == "1.2.3"
    store.answer(REGISTRY, UUID(NOTICE), 2, approved=False)
    with pytest.raises(InstallError, match="decision denied"):
        installer.packs_for_serving()
    with pytest.raises(InstallError, match="decision denied"):
        installer.apply(targets[0], venvs=False)
    feed["revision"] = 3
    feed["notices"][0].update(id="636d54c0-ef6b-44cb-9ee3-bf7381bf0ed8", revision=3)
    feed["notices"][0]["release"].update(
        version="3.0.0", artifactDigest=targets[2].packs[0].artifact_digest
    )
    # New notices are added; authenticated updates cannot drop the original block record.
    previous, _ = store.cached(REGISTRY)
    assert previous is not None
    feed["notices"].append(previous.notices[0].model_dump(mode="json", by_alias=True))
    store.import_feed(REGISTRY, _signed(feed, key), now=VERIFIED_AT)
    with pytest.raises(InstallError, match="decision pending"):
        installer.apply(
            targets[2], {targets[2].packs[0].artifact_digest: archives["3.0.0"]}, venvs=False
        )
    store.answer(REGISTRY, UUID("636d54c0-ef6b-44cb-9ee3-bf7381bf0ed8"), 3, approved=True)
    installer.apply(
        targets[2], {targets[2].packs[0].artifact_digest: archives["3.0.0"]}, venvs=False
    )
    spec = installer.packs_for_serving()[0]
    assert spec.packs is not None and spec.packs["test-pack"].version == "3.0.0"
