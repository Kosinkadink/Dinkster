from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest
from dinkster_workers import load_manifest
from dinkster_workers.doctor import diagnose_static
from dinkster_workers.pack_wheel import (
    PackArchiveError,
    build_pack_archive,
    installed_pack_manifests,
)


def wheel_pack(root: Path, *, requires: str = "demo-core>=2") -> Path:
    package = root / "src" / "demo_nodes"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("NODES = []\nVALUE = 37\n")
    (root / "dinkster-pack.toml").write_text(
        '[pack]\nname = "demo-nodes"\nversion = "1.2.3"\nengine = ">=0.0.2"\n'
        f"requires = [{json.dumps(requires)}]\n"
        '[pack.entry]\nnodes = "demo_nodes:NODES"\n'
    )
    (root / "pyproject.toml").write_text('[project]\nname = "demo-nodes"\nversion = "9.9"\n')
    (root / "locales").mkdir()
    (root / "locales" / "en.json").write_text('{"hello":"world"}\n')
    return root


def test_manifest_drives_installable_wheel_and_record(tmp_path: Path) -> None:
    root = wheel_pack(tmp_path / "source")
    destination = tmp_path / "dist" / "demo_nodes-1.2.3-py3-none-any.whl"
    digest = build_pack_archive(root, destination)
    contents = destination.read_bytes()
    assert digest == "sha256:" + hashlib.sha256(contents).hexdigest()
    assert build_pack_archive(root, destination) == digest
    assert destination.read_bytes() == contents
    with zipfile.ZipFile(destination) as archive:
        assert {Path(name).parts[0] for name in archive.namelist()} == {
            "demo_nodes", "demo_nodes-1.2.3.dist-info"
        }
        metadata = BytesParser().parsebytes(archive.read("demo_nodes-1.2.3.dist-info/METADATA"))
        assert metadata["Name"] == "demo-nodes"
        assert metadata["Version"] == "1.2.3"
        assert metadata.get_all("Requires-Dist") == ["demo-core>=2"]
        assert archive.read("demo_nodes/dinkster-pack.toml") == (root / "dinkster-pack.toml").read_bytes()
        assert archive.read("demo_nodes/locales/en.json") == b'{"hello":"world"}\n'
        record = list(csv.reader(io.StringIO(archive.read("demo_nodes-1.2.3.dist-info/RECORD").decode())))
        assert {row[0] for row in record} == set(archive.namelist())
        for name, hash_field, size in record:
            if name.endswith("/RECORD"):
                assert (hash_field, size) == ("", "")
            else:
                data = archive.read(name)
                assert size == str(len(data))
                assert hash_field == "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()

    environment = tmp_path / "environment"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True, capture_output=True)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", "--no-index", str(destination)],
        check=True, capture_output=True,
    )
    result = subprocess.check_output(
        [str(python), "-I", "-c", "import demo_nodes; from importlib.metadata import version; print(demo_nodes.VALUE, version('demo-nodes'))"],
        text=True,
    )
    assert result.strip() == "37 1.2.3"
    site_packages = Path(subprocess.check_output(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
    ).strip())
    [(distribution, manifest_path)] = list(installed_pack_manifests(site_packages))
    assert distribution.metadata["Name"] == "demo-nodes"
    assert manifest_path == site_packages / "demo_nodes" / "dinkster-pack.toml"
    assert load_manifest(manifest_path).name == "demo-nodes"
    assert importlib.metadata.Distribution.at(site_packages / "demo_nodes-1.2.3.dist-info").version == "1.2.3"


@pytest.mark.parametrize("problem", ["distribution-name", "import-name", "pth-forbidden", "entry-point-forbidden"])
def test_pack_wheel_refuses_invalid_identity_or_execution_hooks(tmp_path: Path, problem: str) -> None:
    root = wheel_pack(tmp_path / "source")
    if problem == "distribution-name":
        (root / "pyproject.toml").write_text('[project]\nname = "other"\n')
    elif problem == "import-name":
        (root / "src" / "other").mkdir()
        (root / "src" / "other" / "__init__.py").write_text("")
    elif problem == "pth-forbidden":
        (root / "src" / "demo_nodes" / "hook.pth").write_text("import os\n")
    else:
        (root / "pyproject.toml").write_text('[project]\nname = "demo-nodes"\n[project.scripts]\nrun = "demo_nodes:run"\n')
    with pytest.raises(PackArchiveError, match=f"wheel.{problem}"):
        build_pack_archive(root, tmp_path / "dist")
    assert not (tmp_path / "dist").exists()


def test_doctor_refuses_pack_requirements_but_accepts_library_names(tmp_path: Path) -> None:
    root = wheel_pack(tmp_path / "source", requires="remote.pack>=2")
    manifest = load_manifest(root / "dinkster-pack.toml")
    report = diagnose_static(root, manifest, published_packs={"remote-pack"})
    assert [f.code for f in report.findings if f.code == "wheel.requires-pack"] == ["wheel.requires-pack"]
    library_report = diagnose_static(root, manifest, published_packs={"different-pack"})
    assert not any(f.code == "wheel.requires-pack" for f in library_report.findings)
    (root / "pyproject.toml").write_text('[project]\nname = "different"\n')
    (root / "src" / "demo_nodes").rename(root / "src" / "different")
    report = diagnose_static(root, manifest)
    assert {"wheel.distribution-name", "wheel.import-name"} <= {f.code for f in report.findings}


def test_pack_archive_refuses_zip_output(tmp_path: Path) -> None:
    root = wheel_pack(tmp_path / "source")
    with pytest.raises(PackArchiveError, match="wheel.required"):
        build_pack_archive(root, tmp_path / "old.zip")


@pytest.mark.parametrize("declared_dependency", [False, True])
def test_doctor_refuses_other_pack_import_even_with_composition_dependency(
    tmp_path: Path, declared_dependency: bool
) -> None:
    root = wheel_pack(tmp_path / "source")
    path = root / "dinkster-pack.toml"
    if declared_dependency:
        path.write_text(path.read_text() + '\n[pack.dependencies]\nremote-pack = ">=2"\n')
    (root / "src" / "demo_nodes" / "__init__.py").write_text(
        "import remote_pack\nimport demo_core\nNODES = []\n"
    )
    report = diagnose_static(root, load_manifest(path), published_packs={"remote-pack"})
    findings = [f for f in report.findings if f.code == "imports.pack-implementation"]
    assert [f.message for f in findings] == ["imports remote_pack"]
