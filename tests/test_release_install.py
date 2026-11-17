"""The release builder produces one complete, version-bound artifact set."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml
from packaging.version import Version
from test_frontend_artifact import artifact

from scripts import build_release
from scripts.build_release import (
    bind_pypi_wheels,
    build_source_archive,
    development_version,
    validate_engine_wheels,
    versioned_sources,
)
from scripts.download_frontend import verify_and_extract

ROOT = Path(__file__).resolve().parents[1]


def test_extra_nightly_requests_cannot_be_replaced_before_publication() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    assert workflow["concurrency"] == {
        "group": "wheel-release-${{ github.event_name == 'workflow_dispatch' "
        "&& format('dispatch-{0}', github.run_id) || github.ref }}",
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }
    channel = workflow["jobs"]["channel"]
    assert channel["concurrency"] == {
        "group": "${{ github.event_name == 'workflow_dispatch' "
        "&& format('environment-channel-publication-dispatch-{0}', github.run_id) "
        "|| 'environment-channel-publication' }}",
        "cancel-in-progress": False,
    }
    publication = next(
        step["run"]
        for step in channel["steps"]
        if "DINKSTER_CHANNEL_SIGNING_KEY" in step.get("env", {})
    )
    assert "with retention.publication_lock(client, bucket):" in publication
    assert (
        "python .release/retention/signed_retention.py --command "
        "python -m scripts.publish_environment_channel"
    ) in " ".join(publication.replace("\\\n", " ").split())


def _distribution_wheels(tmp_path: Path, project: str) -> tuple[Path, Path]:
    direct = tmp_path / "direct"
    subprocess.run(
        ["uv", "build", "--package", project, "--wheel", "--out-dir", str(direct)],
        cwd=ROOT,
        check=True,
    )
    source = tmp_path / "source"
    subprocess.run(
        ["uv", "build", "--package", project, "--sdist", "--out-dir", str(source)],
        cwd=ROOT,
        check=True,
    )
    extracted = tmp_path / "isolated"
    with tarfile.open(next(source.glob("*.tar.gz"))) as archive:
        archive.extractall(extracted, filter="data")
    [package] = extracted.iterdir()
    rebuilt = tmp_path / "rebuilt"
    subprocess.run(
        ["uv", "build", str(package), "--wheel", "--out-dir", str(rebuilt)],
        cwd=extracted,
        check=True,
    )
    return next(direct.glob("*.whl")), next(rebuilt.glob("*.whl"))


@pytest.mark.parametrize(
    "project",
    ["dinkster", "dinkster-api", "dinkster-engine", "dinkster-supervisor", "dinkster-workers"],
)
def test_source_distribution_rebuilds_the_complete_direct_wheel(
    tmp_path: Path, project: str
) -> None:
    direct, rebuilt = _distribution_wheels(tmp_path, project)
    with (
        zipfile.ZipFile(direct) as expected,
        zipfile.ZipFile(rebuilt) as actual,
    ):
        if project == "dinkster":
            assert expected.read("dinkster/py.typed") == b""
            assert actual.read("dinkster/py.typed") == b""
        assert set(actual.namelist()) == set(expected.namelist())
        for name in expected.namelist():
            assert actual.read(name) == expected.read(name), name
        if project in {"dinkster-api", "dinkster-supervisor"}:
            security = ROOT / "packages/dinkster-supervisor/src/dinkster_supervisor/security"
            for path in security.glob("*.py"):
                assert actual.read(f"dinkster_security/{path.name}") == path.read_bytes()
                if project == "dinkster-supervisor":
                    assert (
                        actual.read(f"dinkster_supervisor/security/{path.name}")
                        == path.read_bytes()
                    )
    if project not in {"dinkster", "dinkster-engine"}:
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                f"import sys; sys.path.insert(0, {str(rebuilt)!r}); "
                "from pathlib import Path; from dinkster_store import object_path; "
                f"root = Path({str(tmp_path / 'store')!r}); "
                "first = object_path(root, 'sha256:' + 'a' * 64, label='same-label'); "
                "second = object_path(root, 'sha256:' + 'b' * 64, label='same-label'); "
                "assert first == root / 'same-label'; "
                "assert second == root / 'same-label--2'; "
                "assert object_path(root, 'sha256:' + 'a' * 64, label='renamed') == first",
            ],
            cwd=tmp_path,
            check=True,
        )


@pytest.mark.network("pypi.org", "files.pythonhosted.org")
@pytest.mark.parametrize("project", ["dinkster-api", "dinkster-supervisor"])
def test_rebuilt_api_wheel_installs_without_workspace_sources(tmp_path: Path, project: str) -> None:
    _, rebuilt = _distribution_wheels(tmp_path, project)
    environment = tmp_path / "standalone"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(environment)],
        cwd=tmp_path,
        check=True,
    )
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-sources",
            str(rebuilt),
        ],
        cwd=tmp_path,
        check=True,
    )
    if project == "dinkster-api":
        imports = (
            "import importlib.util; "
            "assert importlib.util.find_spec('dinkster_supervisor') is None; "
            "from dinkster_api.v1 import Node; "
            "print(Node.__name__); "
        )
        expected = "Node\nSecurityHost"
    else:
        imports = (
            "from dinkster_supervisor.environment import main; "
            "from dinkster_supervisor.security.host import SecurityHost; "
            "from dinkster_supervisor.limits import SECURITY_FEED_LIMIT_BYTES; "
            "assert SECURITY_FEED_LIMIT_BYTES == 33554432; "
            "print(main.__name__, SecurityHost.__name__); "
        )
        expected = "main SecurityHost\nSecurityHost"
    result = subprocess.check_output(
        [
            str(python),
            "-I",
            "-c",
            imports + "from dinkster_security.host import SecurityHost; "
            "print(SecurityHost.__name__)",
        ],
        cwd=tmp_path,
        text=True,
    )
    assert result.strip() == expected

    if project == "dinkster-supervisor":
        alias = tmp_path / "source-security"
        shutil.copytree(
            ROOT / "packages/dinkster-supervisor/src/dinkster_supervisor/security",
            alias / "dinkster_security",
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                f"import sys; sys.path.insert(0, {str(alias)!r}); "
                "from pathlib import Path; "
                "import dinkster_security.host as host; "
                f"assert Path(host.__file__).is_relative_to({str(alias)!r}); "
                "assert host.SECURITY_FEED_LIMIT_BYTES == 33554432",
            ],
            cwd=tmp_path,
            check=True,
        )


def test_built_frontend_wheel_is_discovered_without_a_checkout(tmp_path: Path) -> None:
    source = tmp_path / "frontend-source"
    source.mkdir()
    payload, pin = artifact()
    published = source / "frontend.zip"
    published.write_bytes(payload)
    uv = shutil.which("uv")
    assert uv is not None, "release packaging requires uv"
    with versioned_sources(ROOT, "0.0.1") as staged:
        verify_and_extract(payload, pin, staged / "src/dinkster/frontend_bundle/static")
        subprocess.run(
            [uv, "build", "--package", "dinkster", "--wheel", "--out-dir", str(tmp_path)],
            cwd=staged,
            check=True,
        )
    [wheel] = tmp_path.glob("*.whl")
    installed = tmp_path / "site-packages"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
    shutil.rmtree(source)
    discovered = subprocess.check_output(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys; sys.path[:0] = {[str(installed), str(ROOT / 'src')]!r}; "
            "from dinkster import frontend; "
            f"frontend.__file__ = {str(tmp_path / 'engine/src/dinkster/frontend.py')!r}; "
            "print(frontend.discover_frontend_bundle())",
        ],
        cwd=tmp_path,
        text=True,
    ).strip()
    assert discovered == str(installed / "dinkster/frontend_bundle/static")


def test_development_versions_increase_without_resetting_at_release_tags(tmp_path: Path) -> None:
    repository = tmp_path / "source"
    subprocess.run(["git", "clone", "--shared", str(ROOT), str(repository)], check=True)
    subprocess.run(["git", "checkout", "HEAD~1"], cwd=repository, check=True)
    before = Version(development_version(repository))
    subprocess.run(["git", "tag", "v999.0.0"], cwd=repository, check=True)
    assert Version(development_version(repository)) == before
    subprocess.run(["git", "checkout", "-"], cwd=repository, check=True)
    after = Version(development_version(repository))
    assert before < after
    assert before.local is after.local is None


@pytest.mark.parametrize("tag", [None, "v0.0.13"])
def test_release_wheels_pin_the_workspace_and_frontend_without_changing_the_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tag: str | None,
) -> None:
    version = build_release.release_version(ROOT, tag) if tag else development_version(ROOT)
    if tag:
        assert version == "0.0.13"
    else:
        assert version.startswith("0.0.14.dev")
    assert Version(version).local is None
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    original_lock = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    output = tmp_path / "wheels"
    with versioned_sources(ROOT, version) as staged:
        development_pack = tomllib.loads(
            (staged / "packages/dinkster-nodes-dev/pyproject.toml").read_text()
        )["project"]
        assert development_pack["version"] == version
        for project in ("dinkster", "dinkster-api", "dinkster-engine", "dinkster-supervisor"):
            subprocess.run(
                ["uv", "build", "--wheel", "--package", project, "--out-dir", str(output)],
                cwd=staged,
                check=True,
            )
    validate_engine_wheels(list(output.glob("*.whl")))
    for wheel in output.glob("*.whl"):
        with zipfile.ZipFile(wheel) as archive:
            [metadata_path] = [
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            ]
            assert (
                f"Project-URL: Source, https://github.com/Kosinkadink/Dinkster/tree/{commit}\n"
                in archive.read(metadata_path).decode()
            )
    [wheel] = output.glob("dinkster-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        [name] = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        metadata = archive.read(name).decode()
        supervisor_version = "0.0.10" if tag else development_version(ROOT, "dinkster-supervisor")
        if not tag:
            assert supervisor_version.startswith("0.0.11.dev")
        assert Version(supervisor_version).dev == Version(version).dev
        assert f"Requires-Dist: dinkster-supervisor=={supervisor_version}\n" in metadata
        wheel_modules = {
            name
            for name in archive.namelist()
            if name.startswith("dinkster/") and name.endswith(".py")
        }
        for member in (
            "server",
            "native",
            "collab",
            "compat-comfy",
            "nodes-generation",
        ):
            namespace = "dinkster_" + member.replace("-", "_")
            source = ROOT / "packages" / f"dinkster-{member}" / "src"
            assert {
                name
                for name in archive.namelist()
                if name.startswith(namespace + "/") and name.endswith(".py")
            } == {path.relative_to(source).as_posix() for path in source.rglob("*.py")}
    [supervisor_wheel] = output.glob("dinkster_supervisor-*.whl")
    assert supervisor_wheel.name == f"dinkster_supervisor-{supervisor_version}-py3-none-any.whl"
    independent = tmp_path / "independent-supervisor"
    independent_wheel = build_release.build_supervisor_wheel(ROOT, independent, tag=tag)
    assert independent_wheel.read_bytes() == supervisor_wheel.read_bytes()
    assert json.loads((independent / "release-provenance.json").read_bytes()) == {
        "version": supervisor_version,
        "sourceCommit": commit,
        "sha256": hashlib.sha256(supervisor_wheel.read_bytes()).hexdigest(),
    }
    wheels = sorted(output.glob("*.whl"))
    (output / ".gitignore").unlink(missing_ok=True)
    constraints = output / "constraints.txt"
    constraints.write_text("fixture constraints")
    build_release.write_manifest(
        output,
        {"version": version, "workspaceWheels": [path.name for path in wheels]},
        [*wheels, constraints],
    )
    original_manifest = (output / "release-manifest.json").read_bytes()
    publications = {
        build_release.wheel_metadata(path): {
            "filename": path.name,
            "size": path.stat().st_size,
            "digests": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            "url": f"https://files.pythonhosted.org/packages/fixture/{path.name}",
            "yanked": False,
        }
        for path in wheels
    }

    def publication(url: str, *, timeout: int) -> io.BytesIO:
        parts = url.split("/")
        return io.BytesIO(json.dumps({"urls": [publications[(parts[-3], parts[-2])]]}).encode())

    monkeypatch.setattr(build_release, "urlopen", publication)
    first = next(iter(publications.values()))
    expected_hash = first["digests"]["sha256"]
    first["digests"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="PyPI wheel differs"):
        bind_pypi_wheels(output)
    assert (output / "release-manifest.json").read_bytes() == original_manifest
    first["digests"]["sha256"] = expected_hash
    bound = bind_pypi_wheels(output)
    assert {row["url"] for row in bound["artifacts"] if "url" in row} == {
        file["url"] for file in publications.values()
    }
    for line in (output / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ")
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    [api_wheel] = output.glob("dinkster_api-*.whl")
    with zipfile.ZipFile(api_wheel) as archive:
        [name] = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        api_metadata = archive.read(name).decode()
        assert "Requires-Dist: torch" not in api_metadata
        assert {
            line
            for line in api_metadata.splitlines()
            if line.startswith("Requires-Dist: dinkster-")
        } == {
            "Requires-Dist: dinkster-opentimelineio==0.19.0",
        }
        assert "dinkster_store/__init__.py" in archive.namelist()
        for member in (
            "api",
            "values",
            "schema",
            "protocol",
            "memory",
            "assets",
            "caches",
            "inference-wire",
            "image-document",
            "video",
            "workers",
            "registry",
        ):
            namespace = "dinkster_" + member.replace("-", "_")
            source = ROOT / "packages" / f"dinkster-{member}" / "src"
            assert {
                name
                for name in archive.namelist()
                if name.startswith(namespace + "/")
                and "/preview/" not in name
                and name.endswith(".py")
            } == {path.relative_to(source).as_posix() for path in source.rglob("*.py")}
        assert "dinkster_video_preview/__init__.py" in archive.namelist()
        assert (
            "dinkster-doctor = dinkster_workers.doctor:main"
            in archive.read(f"dinkster_api-{version}.dist-info/entry_points.txt").decode()
        )
    source_archive = build_source_archive(ROOT, version, tmp_path)
    with zipfile.ZipFile(source_archive) as archive:
        source_modules = {
            name.removeprefix("src/")
            for name in archive.namelist()
            if name.startswith("src/dinkster/") and name.endswith(".py")
        }
    assert "dinkster/benchmark.py" in wheel_modules
    assert source_modules == wheel_modules
    assert f"Version: {version}\n" in metadata
    assert f"Requires-Dist: dinkster-api=={version}\n" in metadata
    assert f"Requires-Dist: dinkster-engine=={version}\n" in metadata
    assert "dinkster-frontend" not in metadata
    assert "Requires-Dist: dinkster-inference==0.1.2\n" not in metadata
    assert "Requires-Dist: dinkster-inference==0.1.2; extra == 'inference'\n" in metadata
    assert "Requires-Dist: aiohttp>=3.10\n" in metadata
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys; sys.path.insert(0, {str(wheel)!r}); "
            "import dinkster; from importlib.metadata import version; "
            f"assert dinkster.__version__ == version('dinkster') == {version!r}",
        ],
        cwd=tmp_path,
        check=True,
    )
    assert hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest() == original_lock
