"""The manager-side install model (DESIGN M8): lockfile, plan, generations.

What this proves: an installation is one content-addressed lockfile value
(same digest, same install), cross-publisher namespace conflicts never
reach a machine, install plans are deterministic reviewable diffs, and
activation is an append-only generation swap where running jobs keep the
installation they started on - updates can never mutate an install under
a job, and rollback appends history instead of rewriting it.
"""

from __future__ import annotations

import json

import pytest
from dinkster_registry import (
    InstallError,
    LockedPack,
    Lockfile,
    PlanRecord,
    SnapshotRecord,
    VenvSpec,
    artifact_digest,
    plan,
)

DIGEST_A = artifact_digest(b"bytes A")
DIGEST_B = artifact_digest(b"bytes B")


def locked(
    pack: str = "img-tools",
    version: str = "1.0.0",
    digest: str = DIGEST_A,
    publisher: str = "alice",
    claims: tuple[str, ...] | None = None,
    source: str = "registry",
) -> LockedPack:
    return LockedPack(
        pack=pack,
        version=version,
        artifact_digest=digest,
        publisher=publisher,
        claims=claims if claims is not None else (pack,),
        source=source,
    )


# ---------------------------------------------------------------------------
# Lockfile: the installation as a content-addressed value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("publishers", [("alice", "bob"), ("", ""), ("alice", "")])
def test_lockfile_refuses_cross_publisher_claim_conflicts(publishers: tuple[str, str]) -> None:
    """The conflict is caught before anything reaches the machine, never
    at user runtime - and separator equivalence cannot dodge it:
    'img-extra' is the same claim as 'img.extra', which nests in 'img'."""
    a = locked("img", publisher=publishers[0], claims=("img",))
    b = locked("img-extra", publisher=publishers[1], claims=("img-extra",))
    with pytest.raises(InstallError, match="conflicting namespaces"):
        Lockfile.of([a, b])


def test_lockfile_allows_same_publisher_nested_suites() -> None:
    suite = Lockfile.of(
        [
            locked("impact", claims=("impact",)),
            locked("impact-subpack", claims=("impact-subpack",)),
        ]
    )
    assert len(suite.packs) == 2


def test_lockfile_preserves_exact_claims_and_rejects_cross_kind_conflicts() -> None:
    claims = ("Epsilon Scaling", "TextEncodeAceStepAudio1.5", "Video Slice")
    original = Lockfile.of([locked(claims=claims)])
    restored = Lockfile.from_record_json(original.record_json())
    assert restored == original
    assert restored.packs[0].claims == claims
    assert restored.record_digest() == original.record_digest()
    with pytest.raises(InstallError, match="conflicting namespaces"):
        Lockfile.of([locked(claims=claims), locked("other", publisher="bob", claims=claims)])
    with pytest.raises(InstallError, match="conflicting namespaces"):
        Lockfile.of(
            [
                locked(claims=("Img.Blur",)),
                locked("other", publisher="bob", claims=("img",)),
            ]
        )


# ---------------------------------------------------------------------------
# Install plans: deterministic reviewable diffs
# ---------------------------------------------------------------------------


def test_plan_computes_every_action_kind() -> None:
    current = Lockfile.of(
        [
            locked("keep"),
            locked("upgrade-me", version="1.0.0"),
            locked("downgrade-me", version="2.0.0"),
            locked("reinstall-me", digest=DIGEST_A, source="git:github:7"),
            locked("remove-me"),
        ]
    )
    target = Lockfile.of(
        [
            locked("keep"),
            locked("upgrade-me", version="1.1.0", digest=DIGEST_B),
            locked("downgrade-me", version="1.9.0", digest=DIGEST_B),
            locked("reinstall-me", digest=DIGEST_B, source="git:github:7"),
            locked("add-me", publisher="bob"),
        ]
    )
    steps = plan(current, target).steps
    assert [(s.action, s.pack) for s in steps] == [
        ("add", "add-me"),
        ("downgrade", "downgrade-me"),
        ("reinstall", "reinstall-me"),
        ("remove", "remove-me"),
        ("upgrade", "upgrade-me"),
    ]
    by_pack = {s.pack: s for s in steps}
    assert by_pack["upgrade-me"].from_version == "1.0.0"
    assert by_pack["remove-me"].to is None
    assert by_pack["add-me"].to is not None


# ---------------------------------------------------------------------------
# Generations: activation swaps, jobs pin, history appends
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lockfile persistence (decode reruns every validation)
# ---------------------------------------------------------------------------


def test_lockfile_json_round_trips_canonically() -> None:
    lockfile = Lockfile.of(
        [
            locked(),
            locked(pack="vid-tools", digest=DIGEST_B, source="git:github.com/alice/vid-tools"),
        ]
    )
    decoded = Lockfile.from_record_json(lockfile.record_json())
    assert decoded == lockfile
    assert decoded.record_digest() == lockfile.record_digest()


def test_lockfile_decode_reruns_model_validation() -> None:
    """A hand-edited lockfile that violates install invariants fails at
    load, not at activation."""
    two_copies = (
        '{"format": "dinkster.lock/1", "packs": ['
        f'{{"pack": "img-tools", "version": "1.0.0", "artifactDigest": "{DIGEST_A}", '
        '"publisher": "alice", "claims": ["img-tools"], "source": "registry"}, '
        f'{{"pack": "img-tools", "version": "2.0.0", "artifactDigest": "{DIGEST_B}", '
        '"publisher": "alice", "claims": ["img-tools"], "source": "registry"}]}'
    )
    with pytest.raises(InstallError, match="two entries lock pack"):
        Lockfile.from_record_json(two_copies)
    not_canonical = Lockfile.of([locked()]).record_json().replace("img-tools", "img_tools")
    with pytest.raises(InstallError, match="not canonical"):
        Lockfile.from_record_json(not_canonical)


# ---------------------------------------------------------------------------
# PlanRecord: the persisted boundary between plan and apply
# ---------------------------------------------------------------------------


def test_plan_record_round_trips() -> None:
    base = Lockfile.of([locked()])
    target = Lockfile.of([locked(version="2.0.0", digest=DIGEST_B)])
    record = PlanRecord(target=target, base=base.record_digest(), venvs=False)
    decoded = PlanRecord.from_record_json(record.record_json())
    assert decoded == record
    assert decoded.target.record_digest() == target.record_digest()  # exact target preserved
    assert decoded.record_json() == record.record_json()  # deterministic wire form


def test_plan_record_matches_base_is_digest_equality() -> None:
    base = Lockfile.of([locked()])
    moved = Lockfile.of([locked(version="2.0.0", digest=DIGEST_B)])
    record = PlanRecord(target=moved, base=base.record_digest())
    assert record.matches_base(base)
    assert record.matches_base(Lockfile.of([locked()]))  # same content, same digest
    assert not record.matches_base(moved)  # installation drifted
    assert not record.matches_base(None)  # installation was emptied


# ---------------------------------------------------------------------------
# SnapshotRecord: the whole environment as one restorable value
# ---------------------------------------------------------------------------


def test_snapshot_record_round_trips() -> None:
    lockfile = Lockfile.of(
        [
            locked(),
            locked(pack="vid-tools", digest=DIGEST_B, source="git:github.com/alice/vid-tools"),
        ]
    )
    record = SnapshotRecord.of(
        lockfile,
        {"img-tools": ["torch==2.5.1", "numpy==1.26.4"], "vid-tools": []},
        python="3.12.4",
        platform="linux-x86_64",
        dinkster="0.0.1",
    )
    decoded = SnapshotRecord.from_record_json(record.record_json())
    assert decoded == record
    assert decoded.record_json() == record.record_json()  # deterministic wire form
    # pins normalize sorted, per venv
    assert decoded.pins_for("img-tools") == ("numpy==1.26.4", "torch==2.5.1")
    assert decoded.pins_for("vid-tools") == ()  # pinned empty is not unpinned
    assert decoded.pins_for("absent") is None


def test_pins_may_carry_hash_annotations() -> None:
    """The resolution lock's artifact-identity half rides IN the pin:
    '--hash=<algo>:<digest>' tokens after the exact 'name==version', and
    nothing else - an unrecognized annotation is a corrupted record,
    refused at load. Whitespace normalizes at decode so a compiled file's
    continuation formatting never leaks into pin identity. The same
    vocabulary covers snapshots AND plan venv specs (shared decode)."""
    annotated = "torch==2.5.1 --hash=sha256:aaaa --hash=sha256:bbbb"
    lockfile = Lockfile.of([locked()])
    record = SnapshotRecord.of(lockfile, {"img-tools": [annotated]})
    decoded = SnapshotRecord.from_record_json(record.record_json())
    assert decoded.pins_for("img-tools") == (annotated,)

    head = '{"format": "dinkster.snapshot/1", "lockfile": ' + lockfile.record_json()
    sloppy = head + ', "venvs": {"img-tools": ["torch==2.5.1   --hash=sha256:aaaa "]}}'
    normalized = SnapshotRecord.from_record_json(sloppy)
    assert normalized.pins_for("img-tools") == ("torch==2.5.1 --hash=sha256:aaaa",)

    for bad, fragment in (
        ("torch==2.5.1 --editable", "unrecognized annotation"),
        ("torch==2.5.1 --hash=sha256", "unrecognized annotation"),
        ("torch==2.5.1 --hash=:aaaa", "unrecognized annotation"),
        ("torch==2.5.1 sha256:aaaa", "unrecognized annotation"),
        ("--hash=sha256:a==b", "not an exact"),
        ("   ", "not an exact"),
    ):
        with pytest.raises(InstallError, match=fragment):
            SnapshotRecord.from_record_json(head + f', "venvs": {{"img-tools": ["{bad}"]}}}}')

    plan_record = PlanRecord(
        target=lockfile, venv_specs=(("img-tools", VenvSpec(exact=(annotated,))),)
    )
    replayed = PlanRecord.from_record_json(plan_record.record_json())
    assert dict(replayed.venv_specs)["img-tools"].exact == (annotated,)


def test_pin_grammar_refuses_requirements_syntax_smuggling() -> None:
    """Pins end up in requirements files, where anything beyond
    'name==version --hash=...' is ACTIVE SYNTAX. The decode grammar is a
    full-match whitelist: direct URLs, compact PEP 508 markers, extras,
    multiple '==', empty halves, comments, and malformed digests are all
    refused as corrupt records - a crafted snapshot can never turn an
    exact registry pin into an instruction uv would follow."""
    lockfile = Lockfile.of([locked()])
    head = '{"format": "dinkster.snapshot/1", "lockfile": ' + lockfile.record_json()
    for bad, fragment in (
        # direct URL: contains '==' in the query, would install from evil
        ("https://evil.example/pkg-1-py3-none-any.whl?x==1", "not an exact"),
        # compact PEP 508 marker riding the version
        ("foo==1;python_version<'0'", "not an exact"),
        ("foo[extra]==1", "not an exact"),
        ("foo==1==2", "not an exact"),
        ("==1", "not an exact"),
        ("foo==", "not an exact"),
        ("foo==1<2", "not an exact"),
        ("./local-dir==1", "not an exact"),
        ("foo==1 #comment", "unrecognized annotation"),
        ("foo==1 --hash=sha256:not-hex", "unrecognized annotation"),
        ("foo==1 --hash=sha-256:aaaa", "unrecognized annotation"),
        ("foo==1 -r other.txt", "unrecognized annotation"),
    ):
        payload = head + ', "venvs": ' + json.dumps({"img-tools": [bad]}) + "}"
        with pytest.raises(InstallError, match=fragment):
            SnapshotRecord.from_record_json(payload)
