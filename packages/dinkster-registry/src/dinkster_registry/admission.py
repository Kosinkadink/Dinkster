"""The publish gate: deterministic admission with explicit exceptions.

``admit`` is a pure function over (submission, grant table, release
index): the same inputs always produce the same verdict, so publishing is
reproducible - a publisher can compute the registry's answer before
uploading anything. The doctor evidence it consumes is the doctor's
*JSON report* (the stable machine interface), not a Python import: in the
real pipeline the registry runs the identical ``diagnose()`` against the
uploaded artifact bytes in its own sandbox and feeds the result here, so
the verdict binds to the artifact digest, never to publisher assertions.

The verdict vocabulary mirrors the doctor's: findings with severity,
code, message, and fix. Human judgment appears in exactly one place -
first claim of a free namespace yields ``needs_review`` with the concrete
claim named, never a silent queue (review is a report, not a
status).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, cast

from dinkster_schema import canonical_name, claim_covers, validate_name

from .grants import GrantTable
from .model import (
    RegistryError,
    Release,
    ReleaseIndex,
    ReleaseTemplate,
    ReviewLog,
    Version,
    validate_artifact_digest,
    validate_version,
)


@dataclass(frozen=True)
class AdmissionFinding:
    """One admission diagnostic - same shape vocabulary as doctor findings."""

    severity: Literal["error", "info"]
    code: str
    message: str
    fix: str = ""


@dataclass(frozen=True)
class DoctorEvidence:
    """What the registry's own doctor run said about the artifact bytes.

    Parsed from the doctor's JSON report - the versioned machine
    interface - so the registry consumes exactly what pack CI consumes
    and the two can never disagree on vocabulary.
    """

    pack_name: str
    ok: bool
    node_types: tuple[str, ...]
    error_codes: tuple[str, ...]

    # The report shape this admission code understands. Mirrors
    # dinkster_workers.doctor.DOCTOR_REPORT_VERSION on purpose without
    # importing it: the registry consumes the JSON contract, not the
    # doctor's Python surface; the drift guard in tests/test_registry.py
    # parses a real report and catches disagreement.
    REPORT_VERSION = 1

    @classmethod
    def from_report_json(cls, text: str) -> DoctorEvidence:
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"doctor report is not valid JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise RegistryError("doctor report must be a JSON object")
        report = cast("dict[str, object]", document)
        report_version = report.get("reportVersion")
        if report_version != cls.REPORT_VERSION:
            raise RegistryError(
                f"unsupported doctor reportVersion: {report_version!r} "
                f"(this registry understands {cls.REPORT_VERSION})"
            )
        pack = report.get("pack")
        ok = report.get("ok")
        node_types = report.get("nodeTypes")
        findings = report.get("findings")
        if (
            not isinstance(pack, str)
            or not isinstance(ok, bool)
            or not isinstance(node_types, list)
            or not all(isinstance(item, str) for item in cast("list[object]", node_types))
            or not isinstance(findings, list)
        ):
            raise RegistryError(
                "doctor report is missing required fields "
                "(pack: str, ok: bool, nodeTypes: [str], findings: [...])"
            )
        error_codes: list[str] = []
        for item in cast("list[object]", findings):
            if not isinstance(item, dict):
                raise RegistryError("doctor report findings must be JSON objects")
            entry = cast("dict[str, object]", item)
            severity = entry.get("severity")
            code = entry.get("code")
            if not isinstance(severity, str) or not isinstance(code, str):
                raise RegistryError("doctor findings require severity and code strings")
            if severity == "error":
                error_codes.append(code)
        return cls(
            pack_name=pack,
            ok=ok,
            node_types=tuple(cast("list[str]", node_types)),
            error_codes=tuple(error_codes),
        )


@dataclass(frozen=True)
class Submission:
    """One publish attempt, as the registry sees it.

    ``pack_name`` and ``namespaces`` come from the manifest inside the
    uploaded artifact - claims, not grants. ``evidence`` comes from the
    registry's own doctor run over those exact bytes.
    """

    publisher: str
    pack_name: str
    namespaces: tuple[str, ...]
    version: str
    artifact_digest: str
    evidence: DoctorEvidence
    templates: tuple[ReleaseTemplate, ...] = ()
    """Template descriptors the registry's own probe read from the
    artifact's manifest - browse metadata pinned to these exact bytes,
    recorded onto the release at acceptance."""
    executes: tuple[str, ...] = ()
    """Node types the manifest enrolls as an executor ([pack] executes),
    read by the registry's own probe. Exempt from namespace coverage:
    the owning pack's claim covers them, not this pack's."""


@dataclass(frozen=True)
class Verdict:
    """The admission answer: a report, never an opaque status."""

    state: Literal["accepted", "needs_review", "rejected"]
    findings: tuple[AdmissionFinding, ...] = ()
    new_claims: tuple[str, ...] = ()
    """Canonical claims to enter the grant table upon acceptance
    (first claims and same-publisher nested claims not yet recorded)."""
    already_published: bool = False
    """True for an idempotent re-publication of identical bytes."""

    @property
    def ok(self) -> bool:
        return self.state == "accepted"


def _claims_of(submission: Submission) -> tuple[str, ...]:
    """The full claim set: the pack name IS a claim, through the same
    table as everything else - no second, weaker path to a name."""
    seen: dict[str, None] = {}
    for claim in (submission.pack_name, *submission.namespaces):
        seen.setdefault(canonical_name(claim), None)
    return tuple(seen)


def admit(submission: Submission, grants: GrantTable, releases: ReleaseIndex) -> Verdict:
    """The deterministic publish gate. Never mutates its inputs."""
    findings: list[AdmissionFinding] = []

    for label, name in (
        ("publisher id", submission.publisher),
        ("pack name", submission.pack_name),
        *(("namespace claim", claim) for claim in submission.namespaces),
    ):
        problem = validate_name(name)
        if problem is not None:
            findings.append(
                AdmissionFinding(
                    severity="error",
                    code="registry.invalid-name",
                    message=f"{label} {name!r} {problem}",
                    fix="use the closed name grammar shared by every registry identifier",
                )
            )
    version_problem = validate_version(submission.version)
    if version_problem is not None:
        findings.append(
            AdmissionFinding(
                severity="error",
                code="registry.invalid-version",
                message=f"version {submission.version!r} {version_problem}",
            )
        )
    digest_problem = validate_artifact_digest(submission.artifact_digest)
    if digest_problem is not None:
        findings.append(
            AdmissionFinding(
                severity="error",
                code="registry.invalid-digest",
                message=f"artifact digest {submission.artifact_digest!r} {digest_problem}",
            )
        )
    if findings:
        # Grammar failures make every later check unreliable; stop here.
        return Verdict(state="rejected", findings=tuple(findings))

    pack = canonical_name(submission.pack_name)
    version = str(Version.parse(submission.version))
    evidence = submission.evidence

    if canonical_name(evidence.pack_name) != pack:
        findings.append(
            AdmissionFinding(
                severity="error",
                code="registry.evidence-mismatch",
                message=(
                    f"doctor evidence is for pack {evidence.pack_name!r}, "
                    f"submission is for {submission.pack_name!r}"
                ),
                fix="the report must come from the registry's doctor run over this artifact",
            )
        )
        return Verdict(state="rejected", findings=tuple(findings))

    if not evidence.ok:
        codes = ", ".join(sorted(set(evidence.error_codes))) or "unknown"
        findings.append(
            AdmissionFinding(
                severity="error",
                code="registry.doctor-failed",
                message=f"doctor reported errors: {codes}",
                fix="run `dinkster doctor` locally - same predicate, same verdict",
            )
        )

    claims = _claims_of(submission)
    executed = set(submission.executes)
    for node_type in evidence.node_types:
        if node_type in executed:
            continue
        if not any(claim_covers(claim, node_type) for claim in claims):
            findings.append(
                AdmissionFinding(
                    severity="error",
                    code="registry.node-type-uncovered",
                    message=f"node type {node_type!r} falls under no claimed namespace",
                    fix="add the covering namespace to [pack] namespaces",
                )
            )

    existing = releases.get(pack, version)
    if existing is not None:
        if existing.artifact_digest == submission.artifact_digest:
            return Verdict(state="accepted", already_published=True)
        findings.append(
            AdmissionFinding(
                severity="error",
                code="registry.release-immutable",
                message=(
                    f"{pack} {version} is already published with digest {existing.artifact_digest}"
                ),
                fix="publish the new bytes under a new version; releases never mutate",
            )
        )

    new_claims: list[str] = []
    first_claims: list[str] = []
    for claim in claims:
        status = grants.evaluate(claim, submission.publisher)
        if status == "denied-reserved":
            findings.append(
                AdmissionFinding(
                    severity="error",
                    code="registry.namespace-reserved",
                    message=f"namespace {claim!r} is reserved and never grantable",
                )
            )
        elif status == "denied-taken":
            owner = grants.owner_of(claim)
            held = f" (held by {owner})" if owner is not None else ""
            findings.append(
                AdmissionFinding(
                    severity="error",
                    code="registry.namespace-taken",
                    message=f"namespace {claim!r} conflicts with another publisher's grant{held}",
                    fix="namespaces are owned; pick a namespace outside existing grants",
                )
            )
        elif status == "free":
            new_claims.append(claim)
            first_claims.append(claim)
            findings.append(
                AdmissionFinding(
                    severity="info",
                    code="registry.first-claim",
                    message=(
                        f"first claim of free namespace {claim!r} - explicit review "
                        f"before the grant is recorded"
                    ),
                )
            )
        elif status == "grantable":
            new_claims.append(claim)

    if any(finding.severity == "error" for finding in findings):
        return Verdict(state="rejected", findings=tuple(findings))
    if first_claims:
        return Verdict(state="needs_review", findings=tuple(findings), new_claims=tuple(new_claims))
    return Verdict(state="accepted", findings=tuple(findings), new_claims=tuple(new_claims))


def record_acceptance(
    submission: Submission,
    verdict: Verdict,
    review: ReviewLog,
    grants: GrantTable,
    releases: ReleaseIndex,
) -> Release:
    """Land an accepted publish: grants and release enter together.

    Requires the review log to have actually reached ``accepted`` - a
    needs_review verdict cannot be recorded around its review. Grant
    recording and release recording happen through the same call so no
    code path can index a release whose claims were never granted.
    """
    if verdict.state == "rejected":
        raise RegistryError("a rejected verdict is never recorded")
    if review.state != "accepted":
        raise RegistryError(f"recording requires an accepted review, got {review.state!r}")
    claims = _claims_of(submission)
    declared_new = {canonical_name(claim) for claim in verdict.new_claims}
    required_new = {
        claim for claim in claims if grants.owner_of(claim) != canonical_name(submission.publisher)
    }
    if declared_new != required_new:
        raise RegistryError(
            "accepted verdict claims do not match current grants: "
            f"expected {sorted(required_new)}, got {sorted(declared_new)}"
        )
    for claim in required_new:
        grants.grant(claim, submission.publisher)
    release = Release(
        pack=canonical_name(submission.pack_name),
        version=str(Version.parse(submission.version)),
        artifact_digest=submission.artifact_digest,
        publisher=canonical_name(submission.publisher),
        claims=claims,
        node_types=submission.evidence.node_types,
        templates=submission.templates,
    )
    return releases.add(release)


__all__ = [
    "AdmissionFinding",
    "DoctorEvidence",
    "Submission",
    "Verdict",
    "admit",
    "record_acceptance",
]
