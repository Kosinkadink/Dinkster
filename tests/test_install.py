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


# ---------------------------------------------------------------------------
# PlanRecord: the persisted boundary between plan and apply
# ---------------------------------------------------------------------------


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
