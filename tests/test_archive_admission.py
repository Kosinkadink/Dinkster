from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest
from dinkster_registry import build_artifact
from dinkster_registry.archive_admission import (
    ProbeSettings,
    run_doctor_static,
    run_executable_probe,
    run_fast_security,
    run_ingest_integrity,
    run_manifest_admission,
)
from dinkster_schema import PackCompatibility
from dinkster_workers import ManifestError, load_manifest

FIXTURES = Path(__file__).parent / "fixtures/registry-admission"


@pytest.mark.parametrize("name", ["example-pack", "color-tools"])
def test_service_fixture_archive_contract(tmp_path: Path, name: str) -> None:
    root = FIXTURES / name
    archive = tmp_path / "pack.zip"
    digest = build_artifact(root, archive)
    data = archive.read_bytes()
    ingest = run_ingest_integrity(data, len(data), digest)
    assert ingest.findings == ()
    assert ingest.data == {"artifactDigest": digest, "size": len(data)}
    metadata = run_manifest_admission(root, name, "1.2.3")
    assert metadata.findings == ()
    assert metadata.data is not None
    assert metadata.data["dinkster"] == ">=0.9,<1"
    assert metadata.data["license"] == "Apache-2.0"
    assert metadata.data["namespaces"] == ["example" if name == "example-pack" else "color"]
    assert metadata.data["dependencies"] == []
    assert metadata.data["contracts"] == {
        "host": "dinkster-pack-host/1",
        "api": "dinkster-api/v1",
        "inference": None,
    }
    if name == "example-pack":
        assert metadata.data["templates"][0]["digest"] == (
            "sha256:ac08ce34ba4f8123618661bef2425f7028ffb9ac740578a3ee88684d2523fee8"
        )
    else:
        assert metadata.data["templates"] == []
    doctor = run_doctor_static(root)
    assert doctor.findings == ()
    assert doctor.data is not None
    assert (doctor.data["status"], doctor.data["errorCount"], doctor.data["warningCount"]) == (
        "passed",
        0,
        0,
    )
    security = run_fast_security(data)
    assert security.findings == ()
    assert security.data == {"status": "passed", "policyVersion": "registry-scan/1"}


@pytest.mark.skipif(sys.platform == "win32", reason="the admission probe requires POSIX limits")
@pytest.mark.parametrize("name", ["example-pack", "color-tools"])
def test_service_fixture_probe_contract(name: str) -> None:
    result = run_executable_probe(
        FIXTURES / name, require_isolation=os.environ.get("DINKSTER_TEST_FULL_PROBE") == "1"
    )
    assert [(finding.code, finding.effect) for finding in result.findings] == [
        ("probe.declared-observed-needs", "signal")
    ]
    assert not result.blocked and not result.review
    assert result.data is not None
    observed = {
        item["nodeType"]: (item["schemaVersion"], item["schemaSignature"])
        for item in result.data["schemas"]
    }
    assert observed == (
        {
            "example.resize": (2, "d58504507e29e052c73d756c457a17221f9cb0fe"),
            "example.save": (1, "cda82c9d491cefe0d40a172be289f90a3d077339"),
        }
        if name == "example-pack"
        else {"color.adjust": (1, "80d5efe19d0bd3ea8046438497b146d4870f67f1")}
    )
    assert all(item["role"] == "executing" for item in result.data["schemas"])


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ('name = "Example_Pack"', "canonical hyphen spelling"),
        ('name = "other-pack"', "does not match submission"),
    ],
)
def test_fixture_manifest_identity_refuses(tmp_path: Path, replacement: str, message: str) -> None:
    root = tmp_path / "pack"
    shutil.copytree(FIXTURES / "example-pack", root)
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(manifest.read_text().replace('name = "example-pack"', replacement))
    result = run_manifest_admission(root, "example-pack", "1.2.3")
    assert result.blocked and result.findings[0].code == "manifest.invalid"
    assert message in result.findings[0].message


@pytest.mark.parametrize("version", ["1.0+local", "v1.0", "not-a-version"])
def test_fixture_public_version_refuses(version: str) -> None:
    result = run_manifest_admission(FIXTURES / "example-pack", "example-pack", version)
    assert result.blocked and result.findings[0].code == "manifest.invalid"


@pytest.mark.parametrize("declaration", ['version = "1.0"', 'engine = ">=0.9"'])
def test_manifest_requires_paired_compatibility(tmp_path: Path, declaration: str) -> None:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        f'[pack]\nname = "example-pack"\n{declaration}\n[pack.entry]\nnodes = "nodes:PACK_NODES"\n'
    )
    with pytest.raises(ManifestError, match="version and engine must be strings"):
        load_manifest(manifest)


@pytest.mark.parametrize("engine", ["0.8.99", "0.9.0.dev1", "0.9.7", "1.0"])
def test_shared_compatibility_range(engine: str) -> None:
    compatibility = PackCompatibility("1.2.3", ">=0.9.0.dev1,<1")
    if engine in {"0.8.99", "1.0"}:
        with pytest.raises(ValueError, match=f"example-pack@1.2.3.*{engine}"):
            compatibility.require_engine("example-pack", engine)
    else:
        compatibility.require_engine("example-pack", engine)


def test_missing_manifest_refuses(tmp_path: Path) -> None:
    for result in (
        run_manifest_admission(tmp_path, "example-pack", "1.2.3"),
        run_doctor_static(tmp_path),
        run_executable_probe(tmp_path),
    ):
        assert result.blocked and not result.review


def test_probe_dependency_url_refuses(tmp_path: Path) -> None:
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "example-pack"\n'
        'requires = ["example @ https://example.invalid/package.whl"]\n'
        '[pack.entry]\nnodes = "nodes:PACK_NODES"\n'
    )
    result = run_executable_probe(tmp_path, settings=ProbeSettings())
    assert result.blocked and result.findings[0].code == "probe.dependencies-invalid"
