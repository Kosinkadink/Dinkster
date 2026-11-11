"""One decision policy for installed execution and new installation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from .documents import Notice, Release
from .store import SecurityStore


@dataclass(frozen=True)
class Artifact:
    registry: str
    release: Release
    dependency_digests: tuple[str, ...] = ()


@dataclass(frozen=True)
class Finding:
    artifact: Artifact
    registry: str
    notice: Notice
    state: Literal["malicious", "vulnerable", "affected"]
    decision: Literal["pending", "approved", "denied", "blocked"]
    allowed: bool
    verified_at: datetime | None


@dataclass(frozen=True)
class Assessment:
    findings: tuple[Finding, ...]
    stale_registries: tuple[str, ...]

    @property
    def allowed(self) -> bool:
        return not self.stale_registries and all(item.allowed for item in self.findings)


def assess(
    store: SecurityStore,
    artifacts: tuple[Artifact, ...],
    *,
    installing: bool = False,
    refuse_vulnerable: bool = False,
    maximum_feed_age: timedelta | None = None,
    now: datetime | None = None,
) -> Assessment:
    """Classify immutable inventory without deleting, repairing or resuming anything.

    Malicious byte identities remain blocked when rehosted. Vulnerability advice
    instead belongs to the exact release and issuing registry. Installed vulnerable
    code continues while its notice decision is pending; new installation does not.
    """
    instant = datetime.now(UTC) if now is None else now
    if instant.utcoffset() is None:
        raise ValueError("policy time must include a timezone")
    if maximum_feed_age is not None and maximum_feed_age <= timedelta(0):
        raise ValueError("maximum feed age must be positive")
    registries = tuple(sorted(set(store.registries()) | {item.registry for item in artifacts}))
    findings: list[Finding] = []
    stale: list[str] = []
    provisioned = set(store.registries())
    for registry in registries:
        document, verified_at = store.cached(registry) if registry in provisioned else (None, None)
        if maximum_feed_age is not None and (
            verified_at is None or instant - verified_at > maximum_feed_age
        ):
            stale.append(registry)
        if document is None:
            continue
        for notice in document.notices:
            if notice.status != "active":
                continue
            for artifact in artifacts:
                exact = registry == artifact.registry and notice.release == artifact.release
                malicious_bytes = notice.state == "malicious" and (
                    notice.release.artifact_digest == artifact.release.artifact_digest
                    or notice.release.artifact_digest in artifact.dependency_digests
                )
                if not exact and not malicious_bytes:
                    continue
                state = notice.state
                if (
                    malicious_bytes
                    and notice.release.artifact_digest != artifact.release.artifact_digest
                ):
                    state = "affected"
                if state == "vulnerable":
                    decision = store.decision(registry, notice.id, notice.revision)
                    allowed = (
                        not refuse_vulnerable
                        and decision != "denied"
                        and (not installing or decision == "approved")
                    )
                else:
                    decision, allowed = "blocked", False
                findings.append(
                    Finding(artifact, registry, notice, state, decision, allowed, verified_at)
                )
    return Assessment(tuple(findings), tuple(stale))
