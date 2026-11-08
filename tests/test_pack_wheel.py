from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import subprocess
import sys
import threading
import zipfile
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from dinkster_registry import InstallError, Lockfile, artifact_digest
from dinkster_workers import load_manifest
from dinkster_workers.doctor import diagnose_static
from dinkster_workers.pack_environment import provision_pack
from dinkster_workers.pack_lock import GenerationPackages, PackLock, plan_pack
from dinkster_workers.pack_wheel import (
    PackArchiveError,
    build_pack_archive,
    installed_pack_manifests,
)

from dinkster.installer import http_registry_fetcher
from dinkster.registries import (
    NamedRegistry,
    RegistryConfig,
    prepare_registry_pack_locks,
    resolve_release,
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
            "demo_nodes",
            "demo_nodes-1.2.3.dist-info",
        }
        metadata = BytesParser().parsebytes(archive.read("demo_nodes-1.2.3.dist-info/METADATA"))
        assert metadata["Name"] == "demo-nodes"
        assert metadata["Version"] == "1.2.3"
        assert metadata.get_all("Requires-Dist") == ["demo-core>=2"]
        assert (
            archive.read("demo_nodes/dinkster-pack.toml")
            == (root / "dinkster-pack.toml").read_bytes()
        )
        assert archive.read("demo_nodes/locales/en.json") == b'{"hello":"world"}\n'
        record = list(
            csv.reader(io.StringIO(archive.read("demo_nodes-1.2.3.dist-info/RECORD").decode()))
        )
        assert {row[0] for row in record} == set(archive.namelist())
        for name, hash_field, size in record:
            if name.endswith("/RECORD"):
                assert (hash_field, size) == ("", "")
            else:
                data = archive.read(name)
                assert size == str(len(data))
                assert (
                    hash_field
                    == "sha256="
                    + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                )

    environment = tmp_path / "environment"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(environment)],
        check=True,
        capture_output=True,
    )
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "--no-index",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )
    result = subprocess.check_output(
        [
            str(python),
            "-I",
            "-c",
            "import demo_nodes; from importlib.metadata import version; "
            "print(demo_nodes.VALUE, version('demo-nodes'))",
        ],
        text=True,
    )
    assert result.strip() == "37 1.2.3"
    site_packages = Path(
        subprocess.check_output(
            [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True,
        ).strip()
    )
    [(distribution, manifest_path)] = list(installed_pack_manifests(site_packages))
    assert distribution.metadata["Name"] == "demo-nodes"
    assert manifest_path == site_packages / "demo_nodes" / "dinkster-pack.toml"
    assert load_manifest(manifest_path).name == "demo-nodes"
    assert (
        importlib.metadata.Distribution.at(site_packages / "demo_nodes-1.2.3.dist-info").version
        == "1.2.3"
    )


@pytest.mark.parametrize(
    "problem", ["distribution-name", "import-name", "pth-forbidden", "entry-point-forbidden"]
)
def test_pack_wheel_refuses_invalid_identity_or_execution_hooks(
    tmp_path: Path, problem: str
) -> None:
    root = wheel_pack(tmp_path / "source")
    if problem == "distribution-name":
        (root / "pyproject.toml").write_text('[project]\nname = "other"\n')
    elif problem == "import-name":
        (root / "src" / "other").mkdir()
        (root / "src" / "other" / "__init__.py").write_text("")
    elif problem == "pth-forbidden":
        (root / "src" / "demo_nodes" / "hook.pth").write_text("import os\n")
    else:
        (root / "pyproject.toml").write_text(
            '[project]\nname = "demo-nodes"\n[project.scripts]\nrun = "demo_nodes:run"\n'
        )
    with pytest.raises(PackArchiveError, match=f"wheel.{problem}"):
        build_pack_archive(root, tmp_path / "dist")
    assert not (tmp_path / "dist").exists()


def test_doctor_refuses_pack_requirements_but_accepts_library_names(tmp_path: Path) -> None:
    root = wheel_pack(tmp_path / "source", requires="remote.pack>=2")
    manifest = load_manifest(root / "dinkster-pack.toml")
    report = diagnose_static(root, manifest, published_packs={"remote-pack"})
    assert [f.code for f in report.findings if f.code == "wheel.requires-pack"] == [
        "wheel.requires-pack"
    ]
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
def test_doctor_only_permits_declared_public_pack_helpers(
    tmp_path: Path, declared_dependency: bool
) -> None:
    root = wheel_pack(tmp_path / "source")
    path = root / "dinkster-pack.toml"
    if declared_dependency:
        path.write_text(path.read_text() + '\n[pack.dependencies]\nremote-pack = ">=2"\n')
    (root / "src" / "demo_nodes" / "__init__.py").write_text(
        "import remote_pack\nimport remote_pack._private\nfrom remote_pack import _hidden\n"
        "import demo_core\nNODES = []\n"
    )
    report = diagnose_static(root, load_manifest(path), published_packs={"remote-pack"})
    findings = [f for f in report.findings if f.code == "imports.pack-implementation"]
    assert [f.message for f in findings] == ([] if declared_dependency else ["imports remote_pack"])
    assert [f.message for f in report.findings if f.code == "imports.private-module"] == [
        "imports remote_pack._private",
        "imports remote_pack",
    ]


def test_static_registry_installs_exact_pack_and_library_and_refuses_corruption(
    tmp_path: Path,
) -> None:
    root = wheel_pack(tmp_path / "source")
    (root / "src/demo_nodes/__init__.py").write_text("from demo_core import VALUE\nNODES = []\n")
    output = tmp_path / "output"
    build_pack_archive(root, output)
    pack_file = output / "demo_nodes-1.2.3-py3-none-any.whl"
    library_file = output / "demo_core-2.4-py3-none-any.whl"
    with zipfile.ZipFile(library_file, "w") as wheel:
        files = {
            "demo_core/__init__.py": b"VALUE = 83\n",
            "demo_core-2.4.dist-info/METADATA": (
                b"Metadata-Version: 2.3\nName: demo-core\nVersion: 2.4\n\n"
            ),
            "demo_core-2.4.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
            ),
        }
        record = io.StringIO()
        writer = csv.writer(record)
        for name, data in files.items():
            wheel.writestr(name, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow((name, "sha256=" + digest, str(len(data))))
        writer.writerow(("demo_core-2.4.dist-info/RECORD", "", ""))
        wheel.writestr("demo_core-2.4.dist-info/RECORD", record.getvalue())
    routes: dict[str, bytes] = {}
    requested: list[str] = []
    authorization: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requested.append(self.path)
            authorization.append(self.headers.get("Authorization"))
            data = routes.get(self.path)
            if data is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header(
                "Content-Type",
                "text/html" if "/simple/" in self.path else "application/octet-stream",
            )
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    try:
        wheel_rows = []
        for file in (pack_file, library_file):
            data = file.read_bytes()
            sha256 = hashlib.sha256(data).hexdigest()
            path = f"wheels/{sha256}/{file.name}"
            routes["/static/" + path] = data
            wheel_rows.append(
                {
                    "filename": file.name,
                    "sha256": sha256,
                    "size": len(data),
                    "url": endpoint + "/static/" + path,
                }
            )
        lock_data = json.dumps(
            {
                "format": "dinkster.pack-lock/1",
                "pack": "demo-nodes",
                "variant": "cpu",
                "requiresPython": ">=3.12",
                "requirements": ["demo-core>=2"],
                "indexes": [endpoint + "/static/cpu/simple/"],
                "packages": [{"name": "demo-core", "version": "2.4", "wheels": [wheel_rows[1]]}],
            }
        ).encode()
        lock_digest = artifact_digest(lock_data)
        lock_path = "locks/" + lock_digest.replace(":", "/") + ".json"
        routes["/static/" + lock_path] = lock_data
        base_id = "sha256:" + "b" * 64
        pack_digest = artifact_digest(pack_file.read_bytes())
        release = {
            **wheel_rows[0],
            "url": str(wheel_rows[0]["url"]).removeprefix(endpoint + "/static/"),
            "name": "demo-nodes",
            "version": "1.2.3",
            "digest": pack_digest,
            "publisher": "demo",
            "engine": ">=0.0.2",
            "library": False,
            "requires": ["demo-core>=2"],
            "installLocks": {
                "cpu": {
                    "baseId": base_id,
                    "artifactDigest": pack_digest,
                    "digest": lock_digest,
                    "size": len(lock_data),
                    "url": lock_path,
                }
            },
        }
        routes["/static/packs/index.json"] = json.dumps(
            {"schemaVersion": 1, "items": [release]}
        ).encode()
        for name, row in zip(("demo-nodes", "demo-core"), wheel_rows, strict=True):
            routes[f"/static/cpu/simple/{name}/"] = (
                f'<a href="{row["url"]}#sha256={row["sha256"]}">{row["filename"]}</a>'
            ).encode()
        registry = NamedRegistry("demo", endpoint)
        entry = resolve_release(registry, "demo-nodes", "1.2.3", engine_version="0.0.2")
        assert (
            http_registry_fetcher(endpoint, token="fixture-token")(entry) == pack_file.read_bytes()
        )
        assert authorization[-2:] == ["Bearer fixture-token", "Bearer fixture-token"]
        paths = prepare_registry_pack_locks(
            RegistryConfig((registry,)),
            Lockfile.of([entry]),
            tmp_path / "locks",
            variant="cpu",
            base_id=base_id,
            selected=registry,
        )
        lock = PackLock.read(paths[0])
        assert [(p.name, p.version) for p in lock.packages] == [
            ("demo-core", "2.4"),
            ("demo-nodes", "1.2.3"),
        ]
        assert [p.wheels[0].sha256 for p in lock.packages] == [
            wheel_rows[1]["sha256"],
            wheel_rows[0]["sha256"],
        ]
        plan = plan_pack(
            lock,
            GenerationPackages.inspect(Path(sys.executable)),
            variant="cpu",
            own_environment_size=0,
        )
        python = provision_pack(
            plan,
            generation_python=Path(sys.executable),
            packs=tmp_path / "generation/packs",
            wheel_store=tmp_path / "wheel-store",
            uv_cache=tmp_path / "uv-cache",
            variant="cpu",
        )
        actual = subprocess.check_output(
            [
                str(python),
                "-I",
                "-c",
                "import demo_nodes; from importlib.metadata import version; "
                "print(demo_nodes.VALUE, version('demo-core'), version('demo-nodes'))",
            ],
            text=True,
        )
        assert actual.strip() == "83 2.4 1.2.3"
        assert requested and all(path.startswith("/static/") for path in requested)
        routes["/static/" + str(release["url"])] = b"corrupt"
        with pytest.raises(InstallError, match="wheel.hash-mismatch"):
            http_registry_fetcher(endpoint)(entry)
        routes["/static/" + lock_path] = b"corrupt"
        with pytest.raises(InstallError, match="published lock bytes differ"):
            prepare_registry_pack_locks(
                RegistryConfig((registry,)),
                Lockfile.of([entry]),
                tmp_path / "bad-locks",
                variant="cpu",
                base_id=base_id,
                selected=registry,
            )
        assert not (tmp_path / "bad-locks").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
