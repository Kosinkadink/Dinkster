"""The registry model (DESIGN M8): grants, releases, admission.

What this proves: every recorded ComfyUI registry/manager failure is
unrepresentable in the model - separator/case spoofing of names, two
owners over one namespace, silently mutated releases, opaque review
states, repo-URL identity forks - and the admission gate is deterministic
with exactly one human-judgment exception (first claim of a free
namespace), which is explicit and reasoned, never a silent queue.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from dinkster_registry import (
    DoctorEvidence,
    Grant,
    GrantTable,
    RegistryError,
    Release,
    ReleaseIndex,
    ReleaseTemplate,
    RepoBindings,
    ReviewLog,
    Submission,
    Verdict,
    Version,
    admit,
    artifact_digest,
    canonical_json,
    record_acceptance,
    validate_artifact_digest,
    validate_version,
)

DIGEST_A = artifact_digest(b"artifact bytes A")
DIGEST_B = artifact_digest(b"artifact bytes B")


def evidence(
    pack: str = "img-tools",
    ok: bool = True,
    node_types: tuple[str, ...] = ("img-tools.blur",),
    error_codes: tuple[str, ...] = (),
) -> DoctorEvidence:
    return DoctorEvidence(pack_name=pack, ok=ok, node_types=node_types, error_codes=error_codes)


def submission(
    publisher: str = "alice",
    pack: str = "img-tools",
    namespaces: tuple[str, ...] = (),
    version: str = "1.0.0",
    digest: str = DIGEST_A,
    doctor: DoctorEvidence | None = None,
    executes: tuple[str, ...] = (),
) -> Submission:
    return Submission(
        publisher=publisher,
        pack_name=pack,
        namespaces=namespaces,
        version=version,
        artifact_digest=digest,
        evidence=doctor if doctor is not None else evidence(pack=pack),
        executes=executes,
    )


def codes(verdict: Verdict) -> set[str]:
    return {finding.code for finding in verdict.findings}


def accepted_review() -> ReviewLog:
    return ReviewLog.start("registry", "2026-07-20T00:00:00Z").advance(
        "accepted", "registry", "2026-07-20T00:00:01Z"
    )


# ---------------------------------------------------------------------------
# Versions, digests, deterministic serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["0.0.1", "1.0", "0.4.12", "1.0rc1", "2.post1"])
def test_valid_versions(text: str) -> None:
    assert validate_version(text) is None
    assert str(Version.parse(text)) == text


@pytest.mark.parametrize(
    "text",
    ["01.0.0", "1.00.0", "v1.0.0", "1.0.0-beta", "1.0+local", "1.0 ", "", "a.b.c"],
)
def test_invalid_versions(text: str) -> None:
    assert validate_version(text) is not None
    with pytest.raises(RegistryError):
        Version.parse(text)


def test_version_ordering() -> None:
    assert Version.parse("0.9.9") < Version.parse("1.0.0") < Version.parse("1.0.10")


def test_artifact_digest_grammar() -> None:
    assert validate_artifact_digest(DIGEST_A) is None
    assert validate_artifact_digest("sha256:" + "0" * 64) is not None
    assert validate_artifact_digest("blake3:XYZ") is not None
    assert validate_artifact_digest("0" * 64) is not None


def test_canonical_json_is_key_order_independent() -> None:
    assert canonical_json({"b": 1, "a": [2, 3]}) == canonical_json({"a": [2, 3], "b": 1})
    assert canonical_json({"a": 1}) == '{"a":1}'  # no whitespace leaks into digests


def test_release_record_digest_tracks_content() -> None:
    release = Release("img-tools", "1.0.0", DIGEST_A, "alice", ("img-tools",), ("img-tools.blur",))
    same = Release("img-tools", "1.0.0", DIGEST_A, "alice", ("img-tools",), ("img-tools.blur",))
    other = Release("img-tools", "1.0.0", DIGEST_B, "alice", ("img-tools",), ("img-tools.blur",))
    assert release.record_digest() == same.record_digest()
    assert release.record_digest() != other.record_digest()


def test_release_templates_ride_the_record() -> None:
    """Template descriptors are release metadata like node types: they
    ride the canonical record (so the record digest tracks them) and
    land on the release exactly as the submission's probe recorded."""
    template = ReleaseTemplate(
        id="starter",
        name="Starter",
        digest=artifact_digest(b"document bytes"),
        path="templates/starter.json",
        description="a starting point",
        tags=("video",),
        assets=("model-a",),
    )
    bare = Release("img-tools", "1.0.0", DIGEST_A, "alice", (), ())
    with_template = replace(bare, templates=(template,))
    assert bare.record_digest() != with_template.record_digest()
    assert "templates/starter.json" in with_template.record_json()

    grants = GrantTable()
    releases = ReleaseIndex()
    sub = replace(submission(), templates=(template,))
    verdict = admit(sub, grants, releases)
    release = record_acceptance(sub, verdict, accepted_review(), grants, releases)
    assert release.templates == (template,)


# ---------------------------------------------------------------------------
# Release immutability
# ---------------------------------------------------------------------------


def test_release_index_is_immutable_per_version() -> None:
    index = ReleaseIndex()
    release = Release("img-tools", "1.0.0", DIGEST_A, "alice", ("img-tools",), ())
    index.add(release)
    assert index.add(release) is release  # identical re-publication is idempotent
    with pytest.raises(RegistryError, match="already published"):
        index.add(Release("img-tools", "1.0.0", DIGEST_B, "alice", ("img-tools",), ()))
    assert index.get("img-tools", "1.0.0") is release


# ---------------------------------------------------------------------------
# Repo bindings: provenance metadata, never identity
# ---------------------------------------------------------------------------


def test_repo_binding_keys_on_immutable_id_and_refuses_forks() -> None:
    bindings = RepoBindings()
    bindings.bind("github:12345", "img-tools")
    bindings.bind("github:12345", "img-tools")  # idempotent
    assert bindings.pack_for("github:12345") == "img-tools"
    with pytest.raises(RegistryError, match="already bound"):
        bindings.bind("github:12345", "other-pack")
    with pytest.raises(RegistryError, match="immutable repository id"):
        bindings.bind("", "img-tools")


# ---------------------------------------------------------------------------
# Review lifecycle: a state machine, never an opaque status
# ---------------------------------------------------------------------------


def test_review_happy_path_with_audit_trail() -> None:
    log = ReviewLog.start("registry", "t0")
    log = log.advance("needs_review", "registry", "t1", reason="first claim of 'img'")
    log = log.advance("accepted", "moderator-1", "t2")
    log = log.advance("yanked", "moderator-2", "t3", reason="malware confirmed")
    assert log.state == "yanked"
    assert [t.to_state for t in log.transitions] == [
        "submitted",
        "needs_review",
        "accepted",
        "yanked",
    ]
    assert all(t.actor for t in log.transitions)  # every step attributed


def test_review_judgment_states_require_reasons() -> None:
    log = ReviewLog.start("registry", "t0")
    with pytest.raises(RegistryError, match="explicit reason"):
        log.advance("needs_review", "registry", "t1")
    with pytest.raises(RegistryError, match="explicit reason"):
        log.advance("rejected", "moderator", "t1")


def test_review_refuses_illegal_transitions_and_missing_actors() -> None:
    log = ReviewLog.start("registry", "t0")
    with pytest.raises(RegistryError, match="cannot move"):
        log.advance("yanked", "moderator", "t1", reason="x")  # yank requires acceptance first
    accepted = log.advance("accepted", "registry", "t1")
    with pytest.raises(RegistryError, match="cannot move"):
        accepted.advance("rejected", "moderator", "t2", reason="x")
    with pytest.raises(RegistryError, match="require an actor"):
        log.advance("accepted", "", "t1")


def test_rejection_allows_appeal_back_into_review() -> None:
    log = ReviewLog.start("registry", "t0")
    log = log.advance("rejected", "moderator", "t1", reason="suspected malware")
    log = log.advance("needs_review", "moderator", "t2", reason="publisher appeal")
    assert log.state == "needs_review"


# ---------------------------------------------------------------------------
# Grant table: manifests claim, the registry grants
# ---------------------------------------------------------------------------


def test_separator_variants_are_one_claim() -> None:
    """The ComfyUI failure: capitalization/separator drift making one pack
    look like two. Here every spelling is one grant."""
    grants = GrantTable()
    grants.grant("img-tools", "alice")
    for spelling in ("img_tools", "img.tools", "img-tools"):
        assert grants.evaluate(spelling, "alice") == "granted"
        assert grants.evaluate(spelling, "bob") == "denied-taken"
    assert grants.owner_of("img_tools") == "alice"


def test_cross_publisher_nesting_is_denied_both_directions() -> None:
    grants = GrantTable()
    grants.grant("img", "alice")
    assert grants.evaluate("img.filters", "bob") == "denied-taken"
    assert grants.evaluate("img-extra", "bob") == "denied-taken"  # separator-blind conflict
    other = GrantTable()
    other.grant("img.filters", "alice")
    assert other.evaluate("img", "bob") == "denied-taken"  # enclosing is taking too
    with pytest.raises(RegistryError, match="conflicts with grants held by alice"):
        other.grant("img", "bob")


def test_same_publisher_nesting_is_grantable_without_review() -> None:
    """The Impact-Pack/Impact-Subpack suite shape (DESIGN M8)."""
    grants = GrantTable()
    grants.grant("impact", "alice")
    assert grants.evaluate("impact.subpack", "alice") == "grantable"
    grants.grant("impact.subpack", "alice")
    assert grants.evaluate("impact.subpack", "alice") == "granted"


@pytest.mark.parametrize("claim", ["std", "std.magic", "std-extra", "comfy", "dinkster", "core.x"])
def test_reserved_roots_are_never_grantable(claim: str) -> None:
    grants = GrantTable()
    assert grants.evaluate(claim, "alice") == "denied-reserved"
    with pytest.raises(RegistryError, match="reserved"):
        grants.grant(claim, "alice")


def test_grant_is_idempotent_and_transfer_is_a_registry_operation() -> None:
    grants = GrantTable()
    first = grants.grant("img", "alice")
    assert grants.grant("img", "alice") is first
    moved = grants.transfer("img", "bob")
    assert moved.publisher == "bob"
    assert grants.owner_of("img") == "bob"
    with pytest.raises(RegistryError, match="no grant exists"):
        grants.transfer("video", "bob")


def test_transfer_cannot_split_a_nested_suite_across_publishers() -> None:
    grants = GrantTable()
    grants.grant("impact", "alice")
    grants.grant("impact.subpack", "alice")
    # ANY single transfer inside a nested suite would split ownership -
    # root-first and leaf-first are refused alike.
    with pytest.raises(RegistryError, match="split nested grants"):
        grants.transfer("impact", "bob")
    with pytest.raises(RegistryError, match="split nested grants"):
        grants.transfer("impact.subpack", "bob")
    # The suite changes hands atomically or not at all.
    moved = grants.transfer_suite("impact", "bob")
    assert {g.claim for g in moved} == {"impact", "impact-subpack"}
    assert {g.publisher for g in grants.grants()} == {"bob"}


def test_transfer_suite_from_a_leaf_cannot_orphan_its_root() -> None:
    grants = GrantTable()
    grants.grant("impact", "alice")
    grants.grant("impact.subpack", "alice")
    grants.grant("impact.subpack.extras", "alice")
    with pytest.raises(RegistryError, match="enclosing root"):
        grants.transfer_suite("impact.subpack", "bob")  # 'impact' would be stranded
    with pytest.raises(RegistryError, match="no grant exists"):
        grants.transfer_suite("video", "bob")


def test_grant_table_construction_revalidates() -> None:
    with pytest.raises(RegistryError, match="conflicts"):
        GrantTable((Grant("img", "alice"), Grant("img.filters", "bob")))


# ---------------------------------------------------------------------------
# Admission: the deterministic publish gate
# ---------------------------------------------------------------------------


def test_happy_path_first_publish_needs_review_then_records() -> None:
    grants = GrantTable()
    releases = ReleaseIndex()
    sub = submission()
    verdict = admit(sub, grants, releases)
    assert verdict.state == "needs_review"  # first claim of a free namespace
    assert codes(verdict) == {"registry.first-claim"}
    assert verdict.new_claims == ("img-tools",)
    release = record_acceptance(sub, verdict, accepted_review(), grants, releases)
    assert grants.owner_of("img-tools") == "alice"
    assert releases.get("img-tools", "1.0.0") is release
    assert release.node_types == ("img-tools.blur",)


def test_established_publisher_publishes_deterministically() -> None:
    grants = GrantTable()
    grants.grant("img-tools", "alice")
    verdict = admit(submission(version="1.1.0"), grants, ReleaseIndex())
    assert verdict.state == "accepted"
    assert verdict.new_claims == ()
    assert not verdict.already_published


def test_idempotent_republish_of_identical_bytes() -> None:
    grants = GrantTable()
    grants.grant("img-tools", "alice")
    releases = ReleaseIndex()
    sub = submission()
    record_acceptance(sub, admit(sub, grants, releases), accepted_review(), grants, releases)
    again = admit(sub, grants, releases)
    assert again.state == "accepted"
    assert again.already_published


def test_release_immutability_rejects_different_bytes_under_same_version() -> None:
    grants = GrantTable()
    grants.grant("img-tools", "alice")
    releases = ReleaseIndex()
    sub = submission()
    record_acceptance(sub, admit(sub, grants, releases), accepted_review(), grants, releases)
    verdict = admit(submission(digest=DIGEST_B), grants, releases)
    assert verdict.state == "rejected"
    assert "registry.release-immutable" in codes(verdict)


def test_namespace_spoofing_by_separator_variant_is_rejected() -> None:
    grants = GrantTable()
    grants.grant("img-tools", "alice")
    verdict = admit(
        submission(publisher="mallory", pack="img_tools", doctor=evidence(pack="img_tools")),
        grants,
        ReleaseIndex(),
    )
    assert verdict.state == "rejected"
    assert "registry.namespace-taken" in codes(verdict)


def test_pack_name_is_a_claim_through_the_same_table() -> None:
    """Claiming another publisher's namespace as a pack NAME fails the
    same way - no second, weaker path (prior-art gap)."""
    grants = GrantTable()
    grants.grant("img", "alice")
    verdict = admit(
        submission(
            publisher="mallory",
            pack="img.filters",
            doctor=evidence(pack="img.filters", node_types=("img.filters.blur",)),
        ),
        grants,
        ReleaseIndex(),
    )
    assert verdict.state == "rejected"
    assert "registry.namespace-taken" in codes(verdict)


def test_reserved_namespace_claim_is_rejected() -> None:
    verdict = admit(
        submission(namespaces=("std",), doctor=evidence(node_types=("img-tools.blur",))),
        GrantTable(),
        ReleaseIndex(),
    )
    assert verdict.state == "rejected"
    assert "registry.namespace-reserved" in codes(verdict)


def test_doctor_failure_gates_publish() -> None:
    verdict = admit(
        submission(doctor=evidence(ok=False, error_codes=("import.side-effect",))),
        GrantTable(),
        ReleaseIndex(),
    )
    assert verdict.state == "rejected"
    assert "registry.doctor-failed" in codes(verdict)
    failed = next(f for f in verdict.findings if f.code == "registry.doctor-failed")
    assert "import.side-effect" in failed.message  # the cause is named, never opaque


def test_evidence_pack_mismatch_is_spoofing() -> None:
    verdict = admit(submission(doctor=evidence(pack="other-pack")), GrantTable(), ReleaseIndex())
    assert verdict.state == "rejected"
    assert codes(verdict) == {"registry.evidence-mismatch"}


def test_uncovered_node_types_are_rejected_registry_side() -> None:
    """The registry re-checks coverage itself - never trusts that the
    publisher's local doctor did."""
    verdict = admit(
        submission(doctor=evidence(node_types=("img-tools.blur", "sneaky.node"))),
        GrantTable(),
        ReleaseIndex(),
    )
    assert verdict.state == "rejected"
    assert "registry.node-type-uncovered" in codes(verdict)


def test_executed_node_types_are_exempt_from_coverage() -> None:
    """Node types in [pack] executes are covered by their owning pack's
    claim, exactly as doctor and composition treat them - a pure executor
    announces the node types it implements without claiming them."""
    verdict = admit(
        submission(
            doctor=evidence(node_types=("img-tools.blur", "other.node")),
            executes=("other.node",),
        ),
        GrantTable(),
        ReleaseIndex(),
    )
    assert "registry.node-type-uncovered" not in codes(verdict)


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"publisher": "Alice"}, "registry.invalid-name"),
        ({"pack_name": "Img-Tools"}, "registry.invalid-name"),
        ({"version": "1.0+local"}, "registry.invalid-version"),
        ({"artifact_digest": "md5:abc"}, "registry.invalid-digest"),
    ],
)
def test_grammar_violations_reject_early(override: dict[str, str], code: str) -> None:
    verdict = admit(replace(submission(), **override), GrantTable(), ReleaseIndex())
    assert verdict.state == "rejected"
    assert code in codes(verdict)


def test_record_acceptance_requires_an_accepted_review() -> None:
    grants = GrantTable()
    releases = ReleaseIndex()
    sub = submission()
    verdict = admit(sub, grants, releases)
    pending = ReviewLog.start("registry", "t0")
    with pytest.raises(RegistryError, match="accepted review"):
        record_acceptance(sub, verdict, pending, grants, releases)
    rejected = Verdict(state="rejected")
    with pytest.raises(RegistryError, match="never recorded"):
        record_acceptance(sub, rejected, accepted_review(), grants, releases)


def test_record_acceptance_rejects_forged_missing_claims() -> None:
    grants = GrantTable()
    releases = ReleaseIndex()
    sub = submission()
    with pytest.raises(RegistryError, match="claims do not match"):
        record_acceptance(
            sub, Verdict(state="accepted", new_claims=()), accepted_review(), grants, releases
        )
    assert grants.grants() == ()
    assert releases.releases() == ()


def test_admit_never_mutates_its_inputs() -> None:
    grants = GrantTable()
    releases = ReleaseIndex()
    admit(submission(), grants, releases)
    assert grants.grants() == ()
    assert releases.releases() == ()


def test_suite_publish_with_nested_namespaces_same_publisher() -> None:
    grants = GrantTable()
    grants.grant("impact", "alice")
    releases = ReleaseIndex()
    sub = submission(
        pack="impact-subpack",
        namespaces=("impact.subpack",),
        doctor=evidence(pack="impact-subpack", node_types=("impact.subpack.detailer",)),
    )
    verdict = admit(sub, grants, releases)
    # impact-subpack and impact.subpack are ONE canonical claim nested in
    # alice's own 'impact' grant: auto-grantable, no review needed.
    assert verdict.state == "accepted"
    assert verdict.new_claims == ("impact-subpack",)
    record_acceptance(sub, verdict, accepted_review(), grants, releases)
    assert grants.owner_of("impact.subpack") == "alice"


# ---------------------------------------------------------------------------
# Doctor evidence: consumes the real report JSON (drift guard)
# ---------------------------------------------------------------------------


def test_doctor_evidence_parses_the_real_report_shape() -> None:
    from dinkster_workers.doctor import DOCTOR_REPORT_VERSION, DoctorReport, Finding

    # admission mirrors the version instead of importing it (the JSON is
    # the contract); this is the guard that keeps the mirror honest
    assert DoctorEvidence.REPORT_VERSION == DOCTOR_REPORT_VERSION

    report = DoctorReport(
        pack_name="img-tools",
        manifest_path="/packs/img-tools/dinkster-pack.toml",
        findings=(
            Finding(severity="warning", code="manifest.unpinned-dep", message="x"),
            Finding(severity="error", code="import.side-effect", message="y"),
        ),
        node_types=("img-tools.blur",),
    )
    parsed = DoctorEvidence.from_report_json(report.to_json())
    assert parsed.pack_name == "img-tools"
    assert parsed.ok is False
    assert parsed.node_types == ("img-tools.blur",)
    assert parsed.error_codes == ("import.side-effect",)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"pack": "x", "ok": true, "nodeTypes": [], "findings": []}',  # no version
        '{"reportVersion": 2, "pack": "x", "ok": true, "nodeTypes": [], '
        '"findings": []}',  # future version refused, never best-effort parsed
        '{"reportVersion": "1", "pack": "x", "ok": true, "nodeTypes": [], '
        '"findings": []}',  # version is an int, not a string
        '{"reportVersion": 1, "pack": "x"}',  # missing fields
        '{"reportVersion": 1, "pack": "x", "ok": "yes", "nodeTypes": [], '
        '"findings": []}',  # wrong types
        '{"reportVersion": 1, "pack": "x", "ok": true, "nodeTypes": [1], "findings": []}',
        '{"reportVersion": 1, "pack": "x", "ok": true, "nodeTypes": [], "findings": ["oops"]}',
    ],
)
def test_doctor_evidence_rejects_malformed_reports(text: str) -> None:
    with pytest.raises(RegistryError):
        DoctorEvidence.from_report_json(text)
