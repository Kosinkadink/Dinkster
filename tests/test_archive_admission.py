from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import venv
import zipfile
from dataclasses import replace
from pathlib import Path

import packaging
import pytest
from dinkster_registry import LockedPack, Lockfile, artifact_digest
from dinkster_registry.archive_admission import (
    ProbeSettings,
    run_doctor_static,
    run_executable_probe,
    run_fast_security,
    run_ingest_integrity,
    run_locked_probe,
    run_manifest_admission,
)
from dinkster_schema import PackCompatibility
from dinkster_workers import ManifestError, load_manifest
from dinkster_workers.pack_environment import provision_pack, venv_python
from dinkster_workers.pack_lock import (
    GenerationPackages,
    LockedDistribution,
    LockedWheel,
    PackLock,
    plan_pack,
)
from dinkster_workers.pack_wheel import build_pack_archive, wheel_filename
from dinkster_workers.provision import ProvisionError

FIXTURES = Path(__file__).parent / "fixtures/registry-admission"


def test_locked_probe_installs_only_locked_wheels_without_index_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_registry.archive_admission import StageResult
    from test_pack_environment import _generation, _wheel

    root = tmp_path / "pack"
    shutil.copytree(FIXTURES / "color-tools", root)
    wheel = _wheel(
        tmp_path,
        "locked-extra",
        "1.2",
        "locked_extra",
        "VALUE = 37\n",
        requires=("unresolved-transitive-control>=99",),
    )
    python = _generation(tmp_path / "generation/python", [])
    lock = PackLock(
        "color-tools",
        "linux-cpu",
        ">=3.12",
        ("locked-extra>=1",),
        ("http://127.0.0.1:1/unreachable-index",),
        (LockedDistribution("locked-extra", "1.2", (wheel,)),),
    )

    def probe(root, settings, require_isolation, *, python, source_roots):
        value = subprocess.check_output(
            [str(python), "-I", "-c", "import locked_extra; print(locked_extra.VALUE)"],
            text=True,
        )
        return StageResult(data={"lockedValue": int(value)})

    # Exercise acquisition and the real user provisioner, independently of the jail transport.
    monkeypatch.setattr("dinkster_registry.archive_admission._run_executable_probe", probe)
    monkeypatch.setenv("UV_NO_INDEX", "1")
    result = run_locked_probe(
        root,
        lock,
        generation_python=python,
        packs=tmp_path / "packs",
        wheel_store=tmp_path / "wheels",
        uv_cache=tmp_path / "cache",
        variant="linux-cpu",
    )
    assert not result.blocked and result.data == {"lockedValue": 37}
    assert PackLock.read(tmp_path / "packs/color-tools/pack-lock.json") == lock


@pytest.mark.skipif(sys.platform == "win32", reason="the admission probe requires POSIX limits")
@pytest.mark.parametrize("include_dependency", [False, True])
def test_locked_probe_imports_only_supplied_pack_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include_dependency: bool
) -> None:
    root = tmp_path / "pack"
    shutil.copytree(FIXTURES / "color-tools", root)
    module = root / "color_tools/__init__.py"
    module.write_text(
        "from declared_support import DESCRIPTION\n"
        + module.read_text().replace(
            'description="Adjust image color properties."', "description=DESCRIPTION"
        )
    )
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    (dependency / "declared_support.py").write_text('DESCRIPTION = "Verified dependency source"\n')
    # Keep this transport regression independent of engine-wheel construction.
    monkeypatch.setattr(
        "dinkster_registry.archive_admission.provision_pack",
        lambda *args, **kwargs: Path(sys.executable),
    )
    result = run_locked_probe(
        root,
        PackLock("color-tools", "linux-cpu", ">=3.12", (), (), ()),
        generation_python=Path(sys.executable),
        packs=tmp_path / "packs",
        wheel_store=tmp_path / "wheels",
        uv_cache=tmp_path / "cache",
        variant="linux-cpu",
        source_roots=(dependency,) if include_dependency else (),
        require_isolation=os.environ.get("DINKSTER_TEST_FULL_PROBE") == "1",
    )
    assert result.blocked is not include_dependency, result.operator_details
    if include_dependency:
        assert result.data is not None
        assert result.data["schemas"][0]["schema"]["description"] == "Verified dependency source"
    else:
        assert any(finding.code == "entry.unresolvable" for finding in result.findings)


def test_locked_admission_refuses_the_same_base_conflict_as_user_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation = tmp_path / "generation"
    venv.EnvBuilder(with_pip=False).create(generation)
    python = venv_python(generation)
    site = Path(
        subprocess.check_output(
            [str(python), "-I", "-c", "import sysconfig;print(sysconfig.get_path('purelib'))"],
            text=True,
        ).strip()
    )
    shutil.copytree(Path(packaging.__file__).parent, site / "packaging")
    metadata = site / "simpleeval-1.0.7.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: simpleeval\nVersion: 1.0.7\n")
    root = tmp_path / "pack"
    shutil.copytree(FIXTURES / "color-tools", root)
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        manifest.read_text()
        .replace(
            'name = "color-tools"',
            'name = "dinkster-utilities"\nrequires = ["simpleeval==1.0.8"]',
        )
        .replace("color_tools:PACK_NODES", "dinkster_utilities:PACK_NODES")
    )
    (root / "color_tools").rename(root / "dinkster_utilities")
    lock = PackLock(
        "dinkster-utilities",
        "linux-cpu",
        ">=3.12",
        ("simpleeval==1.0.8",),
        (),
        (LockedDistribution("simpleeval", "1.0.8", ()),),
    )
    plan = plan_pack(
        lock, GenerationPackages.inspect(python), variant="linux-cpu", own_environment_size=0
    )
    assert plan.conflicts == ("simpleeval: generation has 1.0.7, lock needs 1.0.8",)
    with pytest.raises(ProvisionError, match="generation has 1.0.7, lock needs 1.0.8") as refused:
        provision_pack(
            plan,
            generation_python=python,
            packs=generation / "packs",
            wheel_store=tmp_path / "wheels",
            uv_cache=tmp_path / "cache",
            variant="linux-cpu",
            offline=True,
        )
    result = run_locked_probe(
        root,
        lock,
        generation_python=python,
        packs=generation / "packs",
        wheel_store=tmp_path / "wheels",
        uv_cache=tmp_path / "cache",
        variant="linux-cpu",
        offline=True,
        require_isolation=False,
    )
    assert result.blocked and not result.review
    assert [(finding.code, finding.message) for finding in result.findings] == [
        ("probe.dependencies-incompatible", str(refused.value))
    ]
    assert not (generation / "packs").exists()
    assert not (tmp_path / "wheels").exists()
    assert not (tmp_path / "cache").exists()

    from scripts import check_default_pack_admission as admission

    prepared = tmp_path / "prepared"
    artifacts = prepared / "artifacts"
    artifacts.mkdir(parents=True)
    archive = tmp_path / "dinkster_utilities-1.2.3-py3-none-any.whl"
    build_pack_archive(root, archive)
    payload = archive.read_bytes()
    digest = artifact_digest(payload)
    (artifacts / f"{digest.removeprefix('blake3:')}.zip").write_bytes(payload)
    (prepared / "catalog.lock").write_text(
        Lockfile.of(
            [LockedPack("dinkster-utilities", "1.2.3", digest, "dinkster", (), "registry")]
        ).record_json()
    )
    base = "sha256:" + "f" * 64
    (prepared / "engine.json").write_text(
        json.dumps(
            {
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "engineVersion": "0.9.2",
                "baseId": base,
                "variant": "linux-cpu",
            }
        )
    )
    output = prepared / "pack-locks"
    output.mkdir()
    replace(
        lock,
        packages=(
            *lock.packages,
            LockedDistribution(
                "dinkster-utilities",
                "1.2.3",
                (
                    LockedWheel(
                        archive.name,
                        len(payload),
                        hashlib.sha256(payload).hexdigest(),
                        archive.as_uri(),
                    ),
                ),
            ),
        ),
    ).write(output / "dinkster-utilities.json")
    constraints = generation / "constraints.txt"
    constraints.write_text("")
    monkeypatch.setattr(admission, "default_pack_minimums", lambda: {"dinkster-utilities": "1"})
    with pytest.raises(ValueError) as default_refused:
        admission.check_defaults(
            admission.NamedRegistry("", "https://registry.example"),
            engine_version="0.9.2",
            prepared_sources=prepared,
            generation_python=python,
            base_id=base,
            constraints=constraints,
        )
    assert str(default_refused.value) == (
        "default dinkster-utilities@1.2.3 admission refused: "
        f"probe.dependencies-incompatible: {refused.value}"
    )


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
@pytest.mark.parametrize("name", ["example-pack", "color-tools"])
def test_service_fixture_archive_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, platform: str
) -> None:
    root = FIXTURES / name
    archive = tmp_path / wheel_filename(load_manifest(root / "dinkster-pack.toml"))
    build_pack_archive(root, archive)
    data = archive.read_bytes()
    digest = artifact_digest(data)
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
    with monkeypatch.context() as platform_context:
        platform_context.setattr(sys, "platform", platform)
        doctor = run_doctor_static(root)
    expected_findings = (
        [("environment.platform-excluded", "info", "signal")]
        if name == "example-pack" and platform == "win32"
        else []
    )
    assert [
        (item.code, item.severity, item.effect) for item in doctor.findings
    ] == expected_findings
    assert not doctor.blocked and not doctor.review
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
@pytest.mark.parametrize("broken_entry", [False, True])
def test_service_fixture_probe_contract(tmp_path: Path, name: str, broken_entry: bool) -> None:
    root = tmp_path / name
    shutil.copytree(FIXTURES / name, root)
    if broken_entry:
        manifest = root / "dinkster-pack.toml"
        manifest.write_text(manifest.read_text().replace(":PACK_NODES", ":MISSING_NODES"))
    result = run_executable_probe(
        root, require_isolation=os.environ.get("DINKSTER_TEST_FULL_PROBE") == "1"
    )
    if broken_entry:
        assert result.blocked and not result.review
        assert [(finding.code, finding.effect) for finding in result.findings] == [
            ("entry.unresolvable", "block"),
            ("probe.declared-observed-needs", "signal"),
        ]
        assert result.data == {"schemas": []}
        return
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


@pytest.mark.skipif(sys.platform == "win32", reason="the admission probe requires POSIX limits")
@pytest.mark.parametrize("times_out", [False, True])
def test_executable_probe_retains_elapsed_time(tmp_path: Path, times_out: bool) -> None:
    root = tmp_path / "pack"
    shutil.copytree(FIXTURES / "color-tools", root)
    if times_out:
        (root / "color_tools" / "__init__.py").write_text(
            "import threading\nthreading.Event().wait()\n"
        )
    settings = ProbeSettings(probe_timeout_seconds=1 if times_out else 30)
    started = time.monotonic()
    result = run_executable_probe(root, settings=settings, require_isolation=False)
    total = time.monotonic() - started
    assert result.operator_details is not None
    elapsed = result.operator_details["elapsedSeconds"]
    assert isinstance(elapsed, float) and 0 < elapsed <= total
    if times_out:
        assert elapsed >= settings.probe_timeout_seconds
        assert result.blocked
        assert [(finding.code, finding.message) for finding in result.findings] == [
            ("probe.timeout", f"Executable probe exceeded 1s (elapsed {elapsed:.6f}s).")
        ]
    else:
        assert not result.blocked and result.data is not None
        assert [schema["nodeType"] for schema in result.data["schemas"]] == ["color.adjust"]


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ('name = "example_pack"', "canonical hyphen spelling"),
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


@pytest.mark.skipif(sys.platform == "win32", reason="the admission probe requires POSIX limits")
@pytest.mark.parametrize("index_only", [False, True])
@pytest.mark.parametrize("library_range", [None, ">=1,<10", ">=2"])
def test_probe_installs_trusted_library_and_operator_dependencies(
    tmp_path: Path, library_range: str | None, index_only: bool
) -> None:
    def wheel(name: str, version: str, code: str, requires: tuple[str, ...] = ()) -> Path:
        path = tmp_path / f"{name}-{version}-py3-none-any.whl"
        dist_info = f"{name}-{version}.dist-info"
        files = {
            f"{'probe_operator_only' if name == 'torch' else name}/__init__.py": code,
            f"{dist_info}/METADATA": (
                f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                + "".join(f"Requires-Dist: {value}\n" for value in requires)
            ),
            f"{dist_info}/WHEEL": (
                "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
            ),
        }
        files[f"{dist_info}/RECORD"] = "".join(f"{entry},,\n" for entry in files) + (
            f"{dist_info}/RECORD,,\n"
        )
        with zipfile.ZipFile(path, "w") as archive:
            for entry, data in files.items():
                archive.writestr(entry, data)
        return path

    operator_name = "torch" if index_only else "probe_operator_only"
    operator_requirement = operator_name.replace("_", "-") + "==1.0"
    operator = wheel(operator_name, "1.0", "VALUE = 'operator'\n")
    trusted = wheel(
        "probe_registry_library",
        "1.0",
        "from probe_operator_only import VALUE\nVALUE = 'registry-' + VALUE\n",
        (operator_requirement,),
    )
    competitor = wheel("probe_registry_library", "9.0", "VALUE = 'wrong-index'\n")
    index = tmp_path / "simple"
    for name, artifact in (
        (operator_name.replace("_", "-"), operator),
        ("probe-registry-library", competitor),
    ):
        project = index / name
        project.mkdir(parents=True)
        (project / "index.html").write_text(f'<a href="{artifact.as_uri()}">{artifact.name}</a>\n')
    root = tmp_path / "pack"
    shutil.copytree(FIXTURES / "color-tools", root)
    manifest = root / "dinkster-pack.toml"
    requirement = (
        f"probe-registry-library{library_range}"
        if library_range is not None
        else operator_requirement
    )
    manifest.write_text(
        manifest.read_text().replace("[pack]\n", f'[pack]\nrequires = ["{requirement}"]\n')
    )
    module = root / "color_tools" / "__init__.py"
    imported = "probe_registry_library" if library_range else "probe_operator_only"
    expected = "registry-operator" if library_range else "operator"
    module.write_text(
        f"from {imported} import VALUE\nassert VALUE == {expected!r}\n" + module.read_text()
    )
    original = manifest.read_bytes()
    result = run_executable_probe(
        root,
        settings=ProbeSettings(
            probe_package_index_url=index.as_uri(),
            probe_dependency_source="index" if index_only else "cpu",
        ),
        require_isolation=False,
        dependency_wheels=(trusted,) if library_range else (),
    )
    assert manifest.read_bytes() == original
    if library_range == ">=2":
        assert result.blocked
        assert result.findings[0].code == "probe.dependencies-unresolved"
    else:
        assert not result.blocked, result.operator_details
        assert result.data is not None
        assert [schema["nodeType"] for schema in result.data["schemas"]] == ["color.adjust"]


def test_service_fixture_selected_api_refuses_unjailed_execution(tmp_path: Path) -> None:
    result = run_executable_probe(
        tmp_path, api_wheel=tmp_path / "engine.whl", require_isolation=False
    )
    assert [finding.code for finding in result.findings] == ["probe.isolation-unavailable"]


@pytest.mark.skipif(
    os.environ.get("DINKSTER_TEST_FULL_PROBE") != "1",
    reason="the full Linux archive-admission workflow runs this jailed contract",
)
@pytest.mark.parametrize("export_exists", [True, False])
def test_service_fixture_selected_api_keeps_pinned_probe(
    tmp_path: Path, export_exists: bool
) -> None:
    import dinkster_api.v1 as host_api
    import dinkster_workers

    root = tmp_path / "pack"
    shutil.copytree(FIXTURES / "color-tools", root)
    module = root / "color_tools/__init__.py"
    module.write_text(
        "from dinkster_api.v1 import SELECTED_API_DESCRIPTION\n"
        + module.read_text().replace(
            'description="Adjust image color properties."', "description=SELECTED_API_DESCRIPTION"
        )
    )
    wheel = tmp_path / "dinkster_api-99.0-py3-none-any.whl"
    files = {
        "dinkster_api/__init__.py": "",
        "dinkster_api/v1.py": (
            "from dinkster_schema import Node, NodeSchema\n"
            + ('SELECTED_API_DESCRIPTION = "Same-head API export"\n' if export_exists else "")
        ),
        "dinkster_api-99.0.dist-info/METADATA": (
            "Metadata-Version: 2.1\nName: dinkster-api\nVersion: 99.0\n"
        ),
        "dinkster_api-99.0.dist-info/WHEEL": (
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        "untrusted-startup.pth": "import sys; raise RuntimeError('API startup hook executed')\n",
    }
    workers = Path(dinkster_workers.__file__).parent
    for path in workers.rglob("*.py"):
        files["dinkster_workers/" + path.relative_to(workers).as_posix()] = path.read_text()
    files["dinkster_workers/_doctor_probe.py"] = (
        "raise RuntimeError('Selected engine replaced the pinned probe')\n"
    )
    record = "dinkster_api-99.0.dist-info/RECORD"
    files[record] = "".join(f"{entry},,\n" for entry in [*files, record])
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    index = tmp_path / "empty-index"
    index.mkdir()
    assert not hasattr(host_api, "SELECTED_API_DESCRIPTION")
    result = run_executable_probe(
        root,
        api_wheel=wheel,
        settings=ProbeSettings(
            probe_package_index_url=index.as_uri(), probe_dependency_source="index"
        ),
    )
    assert not hasattr(host_api, "SELECTED_API_DESCRIPTION")
    assert result.blocked is not export_exists, result.operator_details
    if export_exists:
        assert result.data is not None
        assert result.data["schemas"][0]["schema"]["description"] == "Same-head API export"
    else:
        assert any(finding.code == "entry.unresolvable" for finding in result.findings)
