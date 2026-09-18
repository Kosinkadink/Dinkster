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
    ComposedPack,
    CompositionGeneration,
    CompositionRecordError,
    GenerationLedger,
    InstallError,
    LockedPack,
    Lockfile,
    PlanRecord,
    Release,
    ResolvedRequirement,
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


def test_composition_generation_digest_binds_mode_packs_and_resolutions() -> None:
    packs = [
        ComposedPack("provider", "2.0.0", DIGEST_B),
        ComposedPack("consumer", "1.0.0", DIGEST_A),
    ]
    resolutions = [
        ResolvedRequirement("consumer", "host", "dinkster-pack-host/1", "dinkster-pack-host/1"),
        ResolvedRequirement("consumer", "pack", "provider>=2", f"provider@2.0.0#{DIGEST_B}"),
    ]

    generation = CompositionGeneration.of("production", packs, resolutions)
    reordered = CompositionGeneration.of(
        "production", list(reversed(packs)), list(reversed(resolutions))
    )

    assert generation == reordered
    assert generation.digest == reordered.digest
    assert generation.digest.startswith("sha256:")
    assert '"mode":"production"' in generation.record_json()
    assert generation.digest != CompositionGeneration.of("development", packs, resolutions).digest
    assert (
        generation.digest
        != CompositionGeneration.of(
            "production",
            packs,
            [ResolvedRequirement("consumer", "pack", "provider>=2", f"provider@2.0.0#{DIGEST_A}")],
        ).digest
    )


def test_production_composition_requires_artifact_pins_and_unique_receipts() -> None:
    with pytest.raises(CompositionRecordError, match="no artifact digest"):
        CompositionGeneration.of("production", [ComposedPack("consumer", "1.0.0")])
    with pytest.raises(CompositionRecordError, match="more than once"):
        CompositionGeneration.of(
            "development",
            [ComposedPack("consumer")],
            [
                ResolvedRequirement("consumer", "host", "host/1", "host/1"),
                ResolvedRequirement("consumer", "host", "host/1", "other/1"),
            ],
        )


# ---------------------------------------------------------------------------
# Lockfile: the installation as a content-addressed value
# ---------------------------------------------------------------------------


def test_lockfile_digest_is_order_independent() -> None:
    a = locked("img-tools")
    b = locked("video-tools", publisher="bob")
    assert Lockfile.of([a, b]).record_digest() == Lockfile.of([b, a]).record_digest()
    assert Lockfile.of([a, b]).packs == Lockfile.of([b, a]).packs  # normalized order


def test_lockfile_digest_tracks_content() -> None:
    base = Lockfile.of([locked()])
    assert base.record_digest() == Lockfile.of([locked()]).record_digest()
    assert base.record_digest() != Lockfile.of([locked(digest=DIGEST_B)]).record_digest()
    assert base.record_digest() != Lockfile.of([locked(source="git:github:1")]).record_digest()


def test_lockfile_refuses_duplicate_pack_identities() -> None:
    with pytest.raises(InstallError, match="one canonical name is one pack"):
        Lockfile.of([locked(), locked(digest=DIGEST_B)])


def test_lockfile_refuses_cross_publisher_claim_conflicts() -> None:
    """The conflict is caught before anything reaches the machine, never
    at user runtime - and separator equivalence cannot dodge it:
    'img-extra' is the same claim as 'img.extra', which nests in 'img'."""
    a = locked("img", claims=("img",))
    b = locked("img-extra", publisher="bob", claims=("img-extra",))
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


def test_lockfile_stores_canonical_forms_only() -> None:
    with pytest.raises(InstallError, match="not canonical"):
        Lockfile.of([locked("img_tools", claims=("img_tools",))])
    # A pure executor (empty [pack] namespaces, non-empty executes)
    # locks no claims; pack-name uniqueness holds without them.
    assert Lockfile.of([locked(claims=())]).packs[0].claims == ()
    with pytest.raises(InstallError, match="version"):
        Lockfile.of([locked(version="1.0")])
    with pytest.raises(InstallError, match="digest"):
        Lockfile.of([locked(digest="md5:nope")])


def test_locked_pack_from_release_and_source_is_metadata() -> None:
    release = Release("img-tools", "1.0.0", DIGEST_A, "alice", ("img-tools",), ("img-tools.blur",))
    from_registry = LockedPack.from_release(release)
    from_git = LockedPack.from_release(release, source="git:github:12345")
    assert from_registry.pack == from_git.pack  # one identity, whatever the source
    assert from_registry.claims == ("img-tools",)
    lockfile = Lockfile.of([from_git])
    assert lockfile.get("img_tools") is from_git  # lookup is canonical too


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


def test_plan_between_identical_lockfiles_is_empty() -> None:
    lockfile = Lockfile.of([locked()])
    assert plan(lockfile, lockfile).empty
    assert plan(Lockfile.of([]), Lockfile.of([])).empty


# ---------------------------------------------------------------------------
# Generations: activation swaps, jobs pin, history appends
# ---------------------------------------------------------------------------


def test_activation_appends_and_is_idempotent_for_identical_content() -> None:
    ledger = GenerationLedger()
    assert ledger.current is None
    first = ledger.activate(Lockfile.of([locked()]))
    assert first.number == 1
    assert ledger.activate(Lockfile.of([locked()])) is first  # same digest, no new generation
    second = ledger.activate(Lockfile.of([locked(version="1.1.0", digest=DIGEST_B)]))
    assert second.number == 2
    assert ledger.current is second


def test_running_jobs_keep_the_installation_they_started_on() -> None:
    ledger = GenerationLedger()
    old = ledger.activate(Lockfile.of([locked()]))
    pinned = ledger.pin("job-1")
    assert pinned is old
    new = ledger.activate(Lockfile.of([locked(version="2.0.0", digest=DIGEST_B)]))
    assert ledger.current is new
    assert old not in ledger.prunable()  # job-1 still holds it
    assert ledger.pin("job-1") is old  # idempotent re-pin, never migrates
    ledger.release("job-1")
    assert old in ledger.prunable()
    assert new not in ledger.prunable()  # current is never prunable


def test_rollback_appends_history_instead_of_rewriting() -> None:
    ledger = GenerationLedger()
    good = Lockfile.of([locked()])
    bad = Lockfile.of([locked(version="2.0.0", digest=DIGEST_B)])
    ledger.activate(good)
    ledger.activate(bad)
    rolled = ledger.rollback()
    assert rolled.number == 3  # a NEW generation with the old content
    assert rolled.lockfile_digest == good.record_digest()
    assert [g.number for g in ledger.generations()] == [1, 2, 3]  # the bad one stays on record


def test_rollback_requires_history() -> None:
    ledger = GenerationLedger()
    with pytest.raises(InstallError, match="roll back"):
        ledger.rollback()
    ledger.activate(Lockfile.of([locked()]))
    with pytest.raises(InstallError, match="roll back"):
        ledger.rollback()


def test_prune_refuses_current_pinned_and_unknown() -> None:
    ledger = GenerationLedger()
    ledger.activate(Lockfile.of([locked()]))
    ledger.pin("job-1")
    ledger.activate(Lockfile.of([locked(version="2.0.0", digest=DIGEST_B)]))
    with pytest.raises(InstallError, match="pinned by running jobs: job-1"):
        ledger.prune(1)
    with pytest.raises(InstallError, match="current"):
        ledger.prune(2)
    with pytest.raises(InstallError, match="no generation 9"):
        ledger.prune(9)
    ledger.release("job-1")
    pruned = ledger.prune(1)
    assert pruned.number == 1
    # Numbers are never reused after pruning.
    third = ledger.activate(Lockfile.of([locked(version="3.0.0", digest=DIGEST_A)]))
    assert third.number == 3


def test_pin_and_release_require_sane_state() -> None:
    ledger = GenerationLedger()
    with pytest.raises(InstallError, match="no generation"):
        ledger.pin("job-1")
    ledger.activate(Lockfile.of([locked()]))
    with pytest.raises(InstallError, match="holds no generation pin"):
        ledger.release("job-1")
    first = ledger.pin("job-1")
    ledger.activate(Lockfile.of([locked(version="2.0.0", digest=DIGEST_B)]))
    assert ledger.pin("job-1") is first  # job-1 stays on generation 1; never migrates


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


def test_lockfile_decode_rejects_malformed_documents() -> None:
    with pytest.raises(InstallError, match="not valid JSON"):
        Lockfile.from_record_json("{nope")
    with pytest.raises(InstallError, match="JSON object"):
        Lockfile.from_record_json("[]")
    with pytest.raises(InstallError, match="unsupported lockfile format"):
        Lockfile.from_record_json('{"format": "dinkster.lock/999", "packs": []}')
    with pytest.raises(InstallError, match="'packs' must be a list"):
        Lockfile.from_record_json('{"format": "dinkster.lock/1", "packs": {}}')
    with pytest.raises(InstallError, match="must be JSON objects"):
        Lockfile.from_record_json('{"format": "dinkster.lock/1", "packs": [42]}')
    entry = (
        '{"pack": "img-tools", "version": "1.0.0", "artifactDigest": 7, '
        '"publisher": "alice", "claims": ["img-tools"], "source": "registry"}'
    )
    with pytest.raises(InstallError, match="'artifactDigest' must be a string"):
        Lockfile.from_record_json('{"format": "dinkster.lock/1", "packs": [' + entry + "]}")
    with pytest.raises(InstallError, match="'claims' must be a list of strings"):
        Lockfile.from_record_json(
            '{"format": "dinkster.lock/1", "packs": [{"pack": "img-tools", '
            '"version": "1.0.0", "artifactDigest": "sha256:00", "publisher": '
            '"alice", "claims": [1], "source": "registry"}]}'
        )


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


def test_plan_record_base_none_means_empty_root() -> None:
    target = Lockfile.of([locked()])
    record = PlanRecord(target=target)
    assert record.base is None and record.venvs is True  # defaults
    decoded = PlanRecord.from_record_json(record.record_json())
    assert decoded.base is None
    assert decoded.matches_base(None)  # empty root still matches
    assert not decoded.matches_base(target)  # any installation is stale


def test_plan_record_matches_base_is_digest_equality() -> None:
    base = Lockfile.of([locked()])
    moved = Lockfile.of([locked(version="2.0.0", digest=DIGEST_B)])
    record = PlanRecord(target=moved, base=base.record_digest())
    assert record.matches_base(base)
    assert record.matches_base(Lockfile.of([locked()]))  # same content, same digest
    assert not record.matches_base(moved)  # installation drifted
    assert not record.matches_base(None)  # installation was emptied


def test_plan_record_decode_rejects_malformed_documents() -> None:
    with pytest.raises(InstallError, match="not valid JSON"):
        PlanRecord.from_record_json("{nope")
    with pytest.raises(InstallError, match="JSON object"):
        PlanRecord.from_record_json("[]")
    with pytest.raises(InstallError, match="unsupported plan format"):
        PlanRecord.from_record_json('{"format": "dinkster.plan/999"}')
    valid_target = Lockfile.of([locked()]).record_json()
    with pytest.raises(InstallError, match="'base' must be a string digest or null"):
        PlanRecord.from_record_json(
            '{"format": "dinkster.plan/1", "base": 7, "venvs": true, "target": '
            + valid_target
            + "}"
        )
    with pytest.raises(InstallError, match="is not a digest"):
        PlanRecord.from_record_json(
            '{"format": "dinkster.plan/1", "base": "oops", "venvs": true, '
            '"target": ' + valid_target + "}"
        )
    with pytest.raises(InstallError, match="'venvs' must be a boolean"):
        PlanRecord.from_record_json(
            '{"format": "dinkster.plan/1", "base": null, "target": ' + valid_target + "}"
        )
    with pytest.raises(InstallError, match="'target' must be a lockfile object"):
        PlanRecord.from_record_json('{"format": "dinkster.plan/1", "base": null, "venvs": true}')


def test_plan_record_target_reruns_lockfile_validation() -> None:
    """A hand-edited plan whose target violates install invariants fails
    at load, exactly like a hand-edited lockfile."""
    two_copies = (
        '{"format": "dinkster.lock/1", "packs": ['
        f'{{"pack": "img-tools", "version": "1.0.0", "artifactDigest": "{DIGEST_A}", '
        '"publisher": "alice", "claims": ["img-tools"], "source": "registry"}, '
        f'{{"pack": "img-tools", "version": "2.0.0", "artifactDigest": "{DIGEST_B}", '
        '"publisher": "alice", "claims": ["img-tools"], "source": "registry"}]}'
    )
    with pytest.raises(InstallError, match="two entries lock pack"):
        PlanRecord.from_record_json(
            '{"format": "dinkster.plan/1", "base": null, "venvs": true, "target": '
            + two_copies
            + "}"
        )


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


def test_snapshot_record_accelerator_scope_is_additive() -> None:
    """Accelerator scope (DESIGN M8, cross-platform): recorded and
    round-tripped like the other host facts, and ABSENT in older
    dinkster.snapshot/1 files - which must keep decoding, as "" (not
    recorded), format id unchanged."""
    lockfile = Lockfile.of([locked()])
    record = SnapshotRecord.of(
        lockfile,
        {"img-tools": ["torch==2.5.1"]},
        python="3.12.4",
        platform="linux-x86_64",
        dinkster="0.0.1",
        accelerator="cuda",
    )
    decoded = SnapshotRecord.from_record_json(record.record_json())
    assert decoded == record and decoded.accelerator == "cuda"

    # a pre-accelerator snapshot file: drop the key, decoding still works
    document = json.loads(record.record_json())
    del document["accelerator"]
    older = SnapshotRecord.from_record_json(json.dumps(document))
    assert older.accelerator == ""  # not recorded, never invented
    assert older.pins_for("img-tools") == ("torch==2.5.1",)

    with pytest.raises(InstallError, match="'accelerator' must be a string"):
        document["accelerator"] = 7
        SnapshotRecord.from_record_json(json.dumps(document))


def test_snapshot_record_runtime_facts_are_additive_and_advisory() -> None:
    """Runtime/toolchain facts (DESIGN M8): recorded as a sorted string
    map, OMITTED from the wire when nothing was detected (never an empty
    object), and absent in older dinkster.snapshot/1 files - which keep
    decoding as not-recorded, format id unchanged."""
    lockfile = Lockfile.of([locked()])
    record = SnapshotRecord.of(
        lockfile,
        {"img-tools": ["torch==2.5.1"]},
        accelerator="cuda",
        runtime={"driver": "550.54.14", "cuda": "12.4"},
    )
    assert record.runtime == (("cuda", "12.4"), ("driver", "550.54.14"))  # sorted
    decoded = SnapshotRecord.from_record_json(record.record_json())
    assert decoded == record and decoded.runtime == record.runtime

    # nothing detected: the key is absent from the wire, not {}
    bare = SnapshotRecord.of(lockfile, {"img-tools": ["torch==2.5.1"]}, runtime={})
    assert "runtime" not in json.loads(bare.record_json())
    assert bare.runtime == ()

    # an older file without the key decodes as not-recorded
    document = json.loads(record.record_json())
    del document["runtime"]
    older = SnapshotRecord.from_record_json(json.dumps(document))
    assert older.runtime == ()

    # malformed runtime objects are rejected loudly
    for bad in (7, ["cuda"], {"cuda": 12.4}):
        document["runtime"] = bad
        with pytest.raises(InstallError, match="'runtime' must be an object of strings"):
            SnapshotRecord.from_record_json(json.dumps(document))


def test_plan_record_accelerator_is_additive() -> None:
    """The reviewed plan pins the accelerator it displayed, so apply
    installs the dependency set the user saw. Older plan files without
    the key keep decoding (as "" = not recorded)."""
    record = PlanRecord(target=Lockfile.of([locked()]), accelerator="rocm")
    decoded = PlanRecord.from_record_json(record.record_json())
    assert decoded == record and decoded.accelerator == "rocm"

    document = json.loads(record.record_json())
    del document["accelerator"]
    older = PlanRecord.from_record_json(json.dumps(document))
    assert older.accelerator == ""

    with pytest.raises(InstallError, match="'accelerator' must be a string"):
        document["accelerator"] = 7
        PlanRecord.from_record_json(json.dumps(document))


def test_plan_record_acquire_and_derive_claims_are_additive() -> None:
    """Reproduce-written plans mark acquisition-on-apply and
    manifest-derived claims explicitly in the record - the reviewer of
    the plan file sees both. Absent keys decode as False (older plan
    files keep their refuse-missing, claims-are-real behavior), and
    False is omitted on write so older records stay byte-identical."""
    record = PlanRecord(target=Lockfile.of([locked()]), acquire=True, derive_claims=True)
    decoded = PlanRecord.from_record_json(record.record_json())
    assert decoded == record and decoded.acquire and decoded.derive_claims

    document = json.loads(record.record_json())
    assert document["acquire"] is True and document["deriveClaims"] is True
    del document["acquire"]
    del document["deriveClaims"]
    older = PlanRecord.from_record_json(json.dumps(document))
    assert older.acquire is False and older.derive_claims is False

    plain = PlanRecord(target=Lockfile.of([locked()]))
    plain_document = json.loads(plain.record_json())
    assert "acquire" not in plain_document and "deriveClaims" not in plain_document

    for key in ("acquire", "deriveClaims"):
        with pytest.raises(InstallError, match=f"{key!r} must be a boolean"):
            PlanRecord.from_record_json(json.dumps({**plain_document, key: "yes"}))


def test_plan_record_venv_specs_are_additive() -> None:
    """Restore-written plans record the venv shaping the user reviewed -
    exact snapshot pins or a cross-scope restore's portable constraints -
    so apply provisions what was seen, not a fresh range resolution.
    Absent decodes as empty (fresh ranges, the original behavior), empty
    is omitted on write, entries normalize sorted by pack, and a spec
    for a pack the target does not lock refuses at construction."""
    target = Lockfile.of([locked(), locked(pack="vid-tools", digest=DIGEST_B)])
    record = PlanRecord(
        target=target,
        venv_specs=(
            ("vid-tools", VenvSpec(constraints=("numpy==1.0",))),
            ("img-tools", VenvSpec(exact=("numpy==1.0", "torch==2.0+cu124"))),
        ),
    )
    assert [pack for pack, _ in record.venv_specs] == ["img-tools", "vid-tools"]  # normalized
    decoded = PlanRecord.from_record_json(record.record_json())
    assert decoded == record

    document = json.loads(record.record_json())
    assert document["venvSpecs"]["img-tools"] == {"exact": ["numpy==1.0", "torch==2.0+cu124"]}
    assert document["venvSpecs"]["vid-tools"] == {"constraints": ["numpy==1.0"]}

    plain = PlanRecord(target=target)
    plain_document = json.loads(plain.record_json())
    assert "venvSpecs" not in plain_document
    assert PlanRecord.from_record_json(json.dumps(plain_document)).venv_specs == ()

    with pytest.raises(InstallError, match="does not lock"):
        PlanRecord(target=target, venv_specs=(("ghost", VenvSpec()),))
    with pytest.raises(InstallError, match="'venvSpecs' must be an object"):
        PlanRecord.from_record_json(json.dumps({**plain_document, "venvSpecs": []}))
    with pytest.raises(InstallError, match="must be an object"):
        PlanRecord.from_record_json(json.dumps({**plain_document, "venvSpecs": {"img-tools": 7}}))
    with pytest.raises(InstallError, match="not an exact"):
        PlanRecord.from_record_json(
            json.dumps({**plain_document, "venvSpecs": {"img-tools": {"exact": ["numpy"]}}})
        )


def test_grouped_plan_requires_identical_per_member_exact_pins() -> None:
    target = Lockfile.of([locked(), locked(pack="vid-tools", digest=DIGEST_B)])
    groups = (("models", ("img-tools", "vid-tools")),)
    matching = PlanRecord(
        target=target,
        venv_specs=(
            ("img-tools", VenvSpec(exact=("torch==2.5.1",))),
            ("vid-tools", VenvSpec(exact=("torch==2.5.1",))),
        ),
        venv_groups=groups,
    )
    assert PlanRecord.from_record_json(matching.record_json()) == matching
    with pytest.raises(InstallError, match="contradictory per-member exact pins"):
        PlanRecord(
            target=target,
            venv_specs=(
                ("img-tools", VenvSpec(exact=("torch==2.5.1",))),
                ("vid-tools", VenvSpec(exact=("torch==2.9.1",))),
            ),
            venv_groups=groups,
        )


def test_snapshot_record_venvs_can_disagree_on_versions() -> None:
    """The multiprocess property: per-venv pins are independent facts,
    so two packs on different torch versions is representable."""
    lockfile = Lockfile.of([locked(), locked(pack="vid-tools", digest=DIGEST_B)])
    record = SnapshotRecord.of(
        lockfile, {"img-tools": ["torch==2.5.1"], "vid-tools": ["torch==2.9.1"]}
    )
    assert record.pins_for("img-tools") == ("torch==2.5.1",)
    assert record.pins_for("vid-tools") == ("torch==2.9.1",)


def test_snapshot_record_rejects_pins_for_unlocked_packs() -> None:
    with pytest.raises(InstallError, match="lockfile does not lock"):
        SnapshotRecord.of(Lockfile.of([locked()]), {"ghost": ["torch==2.0.0"]})


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


def test_plan_constraint_pins_are_version_only() -> None:
    """Cross-scope demotion strips artifact identity WITH the exactness
    it belongs to, so a persisted plan whose constraints carry hashes is
    a corrupt record - refused at decode, never silently stripped. Exact
    plan pins keep accepting annotations (same-scope replay)."""
    lockfile = Lockfile.of([locked()])
    annotated = "torch==2.5.1 --hash=sha256:aaaa"
    plan = PlanRecord(target=lockfile, venv_specs=(("img-tools", VenvSpec(exact=(annotated,))),))
    document = json.loads(plan.record_json())
    document["venvSpecs"]["img-tools"] = {"constraints": [annotated]}
    with pytest.raises(InstallError, match="version-only"):
        PlanRecord.from_record_json(json.dumps(document))
    document["venvSpecs"]["img-tools"] = {"constraints": ["torch==2.5.1"]}
    decoded = PlanRecord.from_record_json(json.dumps(document))
    assert dict(decoded.venv_specs)["img-tools"].constraints == ("torch==2.5.1",)


def test_snapshot_record_decode_rejects_malformed_documents() -> None:
    with pytest.raises(InstallError, match="not valid JSON"):
        SnapshotRecord.from_record_json("{nope")
    with pytest.raises(InstallError, match="JSON object"):
        SnapshotRecord.from_record_json("[]")
    with pytest.raises(InstallError, match="unsupported snapshot format"):
        SnapshotRecord.from_record_json('{"format": "dinkster.snapshot/999"}')
    lockfile_json = Lockfile.of([locked()]).record_json()
    head = '{"format": "dinkster.snapshot/1", "lockfile": ' + lockfile_json
    with pytest.raises(InstallError, match="'lockfile' must be a lockfile object"):
        SnapshotRecord.from_record_json('{"format": "dinkster.snapshot/1", "lockfile": 7}')
    with pytest.raises(InstallError, match="'venvs' must be an object"):
        SnapshotRecord.from_record_json(head + ', "venvs": []}')
    with pytest.raises(InstallError, match="must be a list of strings"):
        SnapshotRecord.from_record_json(head + ', "venvs": {"img-tools": [7]}}')
    with pytest.raises(InstallError, match="not an exact 'name==version' pin"):
        SnapshotRecord.from_record_json(head + ', "venvs": {"img-tools": ["torch>=2.0"]}}')
    with pytest.raises(InstallError, match="'python' must be a string"):
        SnapshotRecord.from_record_json(head + ', "venvs": {}, "python": 3}')
