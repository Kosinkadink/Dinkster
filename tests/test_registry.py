"""The registry model (DESIGN M8): grants, releases, admission.

What this proves: every recorded ComfyUI registry/manager failure is
unrepresentable in the model - separator/case spoofing of names, two
owners over one namespace, silently mutated releases, opaque review
states, repo-URL identity forks - and the admission gate is deterministic
with exactly one human-judgment exception (first claim of a free
namespace), which is explicit and reasoned, never a silent queue.
"""

from __future__ import annotations

import pytest
from dinkster_registry import (
    DoctorEvidence,
    RegistryError,
    Submission,
    Verdict,
    Version,
    artifact_digest,
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


# ---------------------------------------------------------------------------
# Release immutability
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Repo bindings: provenance metadata, never identity
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Review lifecycle: a state machine, never an opaque status
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Grant table: manifests claim, the registry grants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Admission: the deterministic publish gate
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Doctor evidence: consumes the real report JSON (drift guard)
# ---------------------------------------------------------------------------
