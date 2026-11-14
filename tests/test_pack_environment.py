"""Real wheel and interpreter coverage for generation-layered pack dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from dinkster_store import object_path
from dinkster_workers.boundary import PROTOCOL_VERSION
from dinkster_workers.pack_environment import (
    GenerationContext,
    doctor_generation,
    fetch_pack_wheels,
    fetch_wheel,
    main,
    pack_drift,
    preflight_generation,
    provision_pack,
    rebuild_generation,
    venv_python,
)
from dinkster_workers.pack_lock import (
    GenerationPackages,
    LockedDistribution,
    LockedWheel,
    PackLock,
    normalize_requirement,
    plan_pack,
    resolve_pack_lock,
)
from dinkster_workers.provision import ProvisionError
from packaging.tags import Tag


def _wheel(
    root: Path, name: str, version: str, module: str, body: str, *, requires: tuple[str, ...] = ()
) -> LockedWheel:
    filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    path = root / filename
    info = f"{name.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{module}.py", body)
        archive.writestr(
            f"{info}/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires),
        )
        archive.writestr(
            f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        archive.writestr(f"{info}/RECORD", "")
    return LockedWheel(
        filename, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest(), path.as_uri()
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_wheel_cache_keeps_same_filename_contents_distinct_and_reuses_offline(
    tmp_path: Path, legacy: bool
) -> None:
    source, store = tmp_path / "source", tmp_path / "store"
    source.mkdir()
    first = _wheel(source, "example", "1.0", "example", "VALUE = 1\n")
    original = (source / first.filename).read_bytes()
    if legacy:
        cached = store / first.sha256 / first.filename
        cached.parent.mkdir(parents=True)
        cached.write_bytes(original)
    first_path = fetch_wheel(first, store)
    assert first_path.parent.name == (first.sha256 if legacy else "example-1.0-py3-none-any")
    second = _wheel(source, "example", "1.0", "example", "VALUE = 23\n")
    second_path = fetch_wheel(second, store)
    assert second_path.parent.name == (
        "example-1.0-py3-none-any" if legacy else "example-1.0-py3-none-any--2"
    )
    assert first_path.read_bytes() == original
    assert second_path.read_bytes() == (source / second.filename).read_bytes()
    renamed_source = source / "example-1.0-1-py3-none-any.whl"
    renamed_source.write_bytes(original)
    renamed = replace(first, filename=renamed_source.name, url=renamed_source.as_uri())
    assert fetch_wheel(renamed, store) == first_path.parent / renamed.filename
    (source / first.filename).unlink()
    assert fetch_wheel(first, store, offline=True) == first_path
    assert fetch_wheel(second, store, offline=True) == second_path
    assert first_path.read_bytes() == original


def _generation(root: Path, wheels: list[LockedWheel]) -> Path:
    """A separate interpreter prefix, like a linked python-build-standalone tree."""
    base = Path(sys.base_prefix)
    if os.name == "nt":
        root.mkdir(parents=True)
        for source in base.iterdir():
            if source.suffix in {".exe", ".dll"}:
                shutil.copy2(source, root / source.name)
        shutil.copytree(base / "DLLs", root / "DLLs")
        shutil.copytree(base / "Lib", root / "Lib", ignore=shutil.ignore_patterns("site-packages"))
        python = root / "python.exe"
    else:
        (root / "bin").mkdir(parents=True)
        shutil.copy2(Path(sys.executable).resolve(), root / "bin/python")
        lib = root / "lib"
        lib.mkdir()
        for source in (base / "lib").iterdir():
            if not source.is_dir():
                (lib / source.name).symlink_to(source)
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        stdlib = lib / version
        stdlib.mkdir()
        for source in (base / "lib" / version).iterdir():
            if source.name != "site-packages":
                (stdlib / source.name).symlink_to(source)
        python = root / "bin/python"
    site = Path(
        subprocess.check_output(
            [str(python), "-c", "import site; print(site.getsitepackages()[0])"], text=True
        ).strip()
    )
    site.mkdir(parents=True, exist_ok=True)
    import packaging

    shutil.copytree(Path(packaging.__file__).parent, site / "packaging")
    for wheel in wheels:
        with zipfile.ZipFile(
            Path(urllib.request.url2pathname(wheel.url.removeprefix("file://")))
        ) as archive:
            archive.extractall(site)
    return python


@contextmanager
def _wheel_server(
    wheel: LockedWheel, payload: bytes, outcome: str
) -> Iterator[tuple[LockedWheel, list[int]]]:
    requests: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(1)
            if outcome == "exhausted" or outcome == "disconnect" and len(requests) == 1:
                self.close_connection = True
                return
            if outcome == "503" and len(requests) == 1 or outcome == "404":
                self.send_error(int(outcome))
                return
            content = (
                b"x" * len(payload)
                if outcome == "hash"
                else payload[:-1]
                if outcome == "short"
                else payload + b"x"
                if outcome == "oversize"
                else payload
            )
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            if outcome == "partial" and len(requests) == 1:
                self.wfile.write(content[:7])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.close_connection = True
                return
            self.wfile.write(content)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield (
            replace(wheel, url=f"http://127.0.0.1:{server.server_port}/{wheel.filename}"),
            requests,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("outcome", ["disconnect", "partial", "503"])
def test_pack_install_retries_interrupted_wheel_download(tmp_path: Path, outcome: str) -> None:
    wheel = _wheel(tmp_path, "fixture-extra", "1.2", "fixture_extra", "VALUE = 23\n")
    payload = (tmp_path / wheel.filename).read_bytes()
    python = _generation(tmp_path / "generation/python", [])
    with _wheel_server(wheel, payload, outcome) as (remote, requests):
        lock = PackLock(
            "example",
            "linux-cpu",
            ">=3.12",
            ("fixture-extra==1.2",),
            (),
            (LockedDistribution("fixture-extra", "1.2", (remote,)),),
        )
        plan = plan_pack(
            lock, GenerationPackages.inspect(python), variant="linux-cpu", own_environment_size=0
        )
        installed = provision_pack(
            plan,
            generation_python=python,
            packs=tmp_path / "generation/packs",
            wheel_store=tmp_path / "store",
            uv_cache=tmp_path / "cache",
            variant="linux-cpu",
        )
        assert (
            subprocess.check_output(
                [str(installed), "-c", "import fixture_extra; print(fixture_extra.VALUE)"],
                text=True,
            ).strip()
            == "23"
        )
        assert len(requests) == 2
        assert not list((tmp_path / "store").rglob("*.partial"))


@pytest.mark.parametrize("outcome", ["exhausted", "404", "hash", "short", "oversize", "offline"])
def test_pack_download_retry_refusals_preserve_verification_and_offline(
    tmp_path: Path, outcome: str
) -> None:
    wheel = _wheel(tmp_path, "fixture-extra", "1.2", "fixture_extra", "VALUE = 23\n")
    with _wheel_server(wheel, (tmp_path / wheel.filename).read_bytes(), outcome) as (
        remote,
        requests,
    ):
        lock = PackLock(
            "example",
            "linux-cpu",
            ">=3.12",
            ("fixture-extra==1.2",),
            (),
            (LockedDistribution("fixture-extra", "1.2", (remote,)),),
        )
        plan = plan_pack(
            lock,
            GenerationPackages.inspect(Path(sys.executable)),
            variant="linux-cpu",
            own_environment_size=0,
        )
        with pytest.raises(ProvisionError) as failure:
            fetch_pack_wheels(plan, tmp_path / "store", offline=outcome == "offline")
        assert "pack example:" in str(failure.value)
        assert remote.url in str(failure.value)
        assert {
            "exhausted": "failed after 3 tries",
            "404": "HTTP Error 404",
            "hash": "wheel hash or size mismatch",
            "short": "wheel hash or size mismatch",
            "oversize": "wheel exceeds locked size",
            "offline": "offline pack wheel missing",
        }[outcome] in str(failure.value)
        assert len(requests) == (3 if outcome == "exhausted" else 0 if outcome == "offline" else 1)
        assert not list((tmp_path / "store").rglob("*.whl"))
        assert not list((tmp_path / "store").rglob("*.partial"))


@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("outcome", ["partial", "exhausted", "hash"])
def test_lock_resolution_retries_verified_dependency_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, measured: bool, outcome: str
) -> None:
    api = _wheel(tmp_path, "dinkster-api", "0.0.3", "dinkster_api", "VALUE = 3\n")
    wheel = _wheel(
        tmp_path,
        "fixture-extra",
        "1.2",
        "fixture_extra",
        "VALUE = 23\n",
        requires=("dinkster-api>=0.0.3",),
    )
    with _wheel_server(wheel, (tmp_path / wheel.filename).read_bytes(), outcome) as (
        remote,
        requests,
    ):

        def compile_lock(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            output = Path(command[command.index("--output-file") + 1])
            rows = []
            for name, version, artifact in (
                ("dinkster-api", "0.0.3", api),
                ("fixture-extra", "1.2", remote),
            ):
                row = (
                    f'[[packages]]\nname="{name}"\nversion="{version}"\n'
                    f"[[packages.wheels]]\nurl={json.dumps(artifact.url)}\n"
                    f'hashes.sha256="{artifact.sha256}"\n'
                )
                if measured or name == "dinkster-api":
                    row += f"size={artifact.size}\n"
                rows.append(row)
            output.write_text("".join(rows))
            return subprocess.CompletedProcess(command, 0, "", "")

        monkeypatch.setattr(subprocess, "run", compile_lock)
        (tmp_path / "constraints.txt").write_text("")

        def resolve() -> PackLock:
            return resolve_pack_lock(
                pack="example",
                variant="linux-cpu",
                requirements=["fixture-extra==1.2"],
                python=Path(sys.executable),
                constraints=tmp_path / "constraints.txt",
                engine=">=0.0.3",
            )

        if outcome != "partial":
            with pytest.raises(ProvisionError) as failure:
                resolve()
            assert remote.url in str(failure.value)
            assert ("failed after 3 tries" if outcome == "exhausted" else "hash mismatch") in str(
                failure.value
            )
            if outcome == "exhausted":
                assert "pack example:" in str(failure.value)
            assert len(requests) == (3 if outcome == "exhausted" else 1)
            return
        lock = resolve()
        assert lock.requirements == ("fixture-extra==1.2", "dinkster-api>=0.0.3")
        assert [package.name for package in lock.packages] == ["fixture-extra"]
        assert len(requests) == (2 if measured else 3)


@pytest.mark.parametrize("source", ["index", "multiple-indexes", "constraint-index", "local"])
def test_resolved_wheel_issuer_uses_only_explicit_unambiguous_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    artifact = _wheel(tmp_path, "fixture-extra", "1.2", "fixture_extra", "VALUE = 23\n")
    content = (tmp_path / artifact.filename).read_bytes()
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_HEAD(self) -> None:
            requests.append(self.path)
            self.send_response(200 if self.path == f"/artifacts/{artifact.filename}" else 404)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()

        def do_GET(self) -> None:
            requests.append(self.path)
            if self.path in {"/simple/fixture-extra/", "/other/fixture-extra/"}:
                payload = (
                    f'<a href="/artifacts/{artifact.filename}#sha256={artifact.sha256}">'
                    f"{artifact.filename}</a>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
            elif self.path == f"/artifacts/{artifact.filename}":
                payload = content
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
            else:
                payload = b""
                self.send_response(404)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        (tmp_path / "uv.toml").write_text(
            f'[[index]]\nurl = "{endpoint}/ambient"\n', encoding="ascii"
        )
        # The offline CI gate must still reach this test's loopback index.
        monkeypatch.delenv("UV_OFFLINE", raising=False)
        monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("UV_INDEX", endpoint + "/ambient")
        monkeypatch.setenv("UV_EXTRA_INDEX_URL", endpoint + "/ambient")
        monkeypatch.setenv("UV_FIND_LINKS", endpoint + "/ambient")
        monkeypatch.setenv("UV_CONFIG_FILE", str(tmp_path / "uv.toml"))
        constraints = tmp_path / "constraints.txt"
        constraints.write_text(
            (f"--extra-index-url {endpoint}/other\n" if source == "constraint-index" else "")
            + "fixture-extra==1.2\n"
        )
        lock = resolve_pack_lock(
            pack="example",
            variant="linux-cpu",
            requirements=["fixture-extra>=1"],
            python=Path(sys.executable),
            constraints=constraints,
            indexes=(endpoint + "/simple",)
            + ((endpoint + "/other",) if source == "multiple-indexes" else ()),
            find_links=(tmp_path,) if source == "local" else (),
        )
        wheel = lock.packages[0].wheels[0]
        assert wheel.filename == artifact.filename and wheel.sha256 == artifact.sha256
        assert wheel.registry == (endpoint + "/simple" if source == "index" else None)
        assert not any("ambient" in path for path in requests)
        assert wheel.url == (
            artifact.url if source == "local" else endpoint + "/artifacts/" + artifact.filename
        )
        encoded_wheel = json.loads(lock.to_json())["packages"][0]["wheels"][0]
        assert encoded_wheel.get("registry") == wheel.registry
        assert ("registry" in encoded_wheel) == (wheel.registry is not None)
        assert PackLock.from_json(lock.to_json()) == lock
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()


@pytest.mark.parametrize("failure", ["missing", "hash", "short", "oversize"])
def test_pack_download_failure_names_pack_url_and_verification_result(
    tmp_path: Path, failure: str
) -> None:
    wheel = _wheel(tmp_path, "fixture-extra", "1.2", "fixture_extra", "value = 23\n")
    other = _wheel(tmp_path, "another-extra", "2.3", "another_extra", "value = 47\n")
    (tmp_path / other.filename).unlink()
    source = tmp_path / wheel.filename
    if failure == "missing":
        source.unlink()
    else:
        data = source.read_bytes()
        source.write_bytes(
            b"x" * len(data)
            if failure == "hash"
            else data[:-1]
            if failure == "short"
            else data + b"x"
        )
    lock = PackLock(
        "example",
        "linux-cpu",
        ">=3.12",
        ("fixture-extra==1.2",),
        (),
        (
            LockedDistribution("fixture-extra", "1.2", (wheel,)),
            LockedDistribution("another-extra", "2.3", (other,)),
        ),
    )
    generation = tmp_path / "generations/1"
    python = _generation(generation / "python", [])
    plan = plan_pack(
        lock, GenerationPackages.inspect(python), variant="linux-cpu", own_environment_size=9000
    )
    store = tmp_path / "store"
    with pytest.raises(ProvisionError) as error:
        provision_pack(
            plan,
            generation_python=python,
            packs=generation / "packs",
            wheel_store=store,
            uv_cache=tmp_path / "cache",
            variant="linux-cpu",
        )
    message = str(error.value)
    assert f"pack example: {wheel.url}: download/verification failed:" in message
    assert f"pack example: {other.url}: download/verification failed:" in message
    assert {
        "missing": "urlopen error",
        "hash": "wheel hash or size mismatch",
        "short": "wheel hash or size mismatch",
        "oversize": "wheel exceeds locked size",
    }[failure] in message
    assert not (object_path(store, "sha256:" + wheel.sha256) / wheel.filename).exists()
    assert not list(store.rglob("*.partial"))
    assert not (generation / "packs/example").exists()


@pytest.mark.parametrize("host_path", ["PYTHONPATH", "PYTHONUSERBASE"])
def test_layered_pack_downloads_only_extra_and_rebuilds_on_new_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host_path: str
) -> None:
    torch = _wheel(tmp_path, "torch", "2.14.1", "torch", "value = 41\n")
    cv = _wheel(tmp_path, "opencv-contrib-python-headless", "5.0", "cv2", "value = 17\n")
    extra = _wheel(
        tmp_path,
        "fixture-extra",
        "1.2",
        "fixture_extra",
        "import torch, cv2\ndef node(): return torch.value + cv2.value\n",
    )
    extra = replace(extra, registry="https://issuer.example.test")
    alternative_source = tmp_path / "alternative"
    alternative_source.mkdir()
    alternative = _wheel(
        alternative_source, "fixture-extra", "1.2", "fixture_extra", "def node(): return -999\n"
    )
    lock = PackLock(
        "example",
        "linux-cpu",
        ">=3.12",
        (
            "torch>=2",
            normalize_requirement("opencv-python[ffmpeg]>=4"),
            "fixture-extra==1.2",
        ),
        (),
        tuple(
            LockedDistribution(
                name, version, (wheel, alternative) if name == "fixture-extra" else (wheel,)
            )
            for name, version, wheel in (
                ("torch", "2.14.1", torch),
                ("opencv-contrib-python-headless", "5.0", cv),
                ("fixture-extra", "1.2", extra),
            )
        ),
    )
    store = tmp_path / "wheels"
    for number in (1, 2):
        generation = tmp_path / "generations" / str(number)
        parent = _generation(generation, [torch, cv])
        with monkeypatch.context() as environment:
            environment.setenv(host_path, str(tmp_path / "host"))
            site = (
                Path(
                    subprocess.check_output(
                        [str(parent), "-c", "import site; print(site.getusersitepackages())"],
                        text=True,
                    ).strip()
                )
                if host_path == "PYTHONUSERBASE"
                else tmp_path / "host"
            )
            metadata = site / "host_only-9.0.dist-info"
            metadata.mkdir(parents=True, exist_ok=True)
            (metadata / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: host-only\nVersion: 9.0\n"
            )
            facts = GenerationPackages.inspect(parent)
        assert facts.versions["torch"] == "2.14.1"
        assert "host-only" not in facts.versions
        plan = plan_pack(lock, facts, variant="linux-cpu", own_environment_size=987654)
        assert not plan.conflicts
        assert plan.wheels == (extra,)
        python = provision_pack(
            plan,
            generation_python=parent,
            packs=generation / "packs",
            wheel_store=store,
            uv_cache=tmp_path / "cache",
            variant="linux-cpu",
        )
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import fixture_extra, torch, cv2, json, importlib.metadata as m; "
                "print(json.dumps([fixture_extra.node(), torch.__file__, cv2.__file__, "
                "json.loads(m.distribution('fixture-extra').read_text('direct_url.json'))]))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        value, torch_file, cv_file, origin = json.loads(result.stdout)
        assert value == 58
        with urllib.request.urlopen(origin["url"]) as installed_wheel:
            assert hashlib.sha256(installed_wheel.read()).hexdigest() == extra.sha256
        assert Path(torch_file).is_relative_to(generation)
        assert Path(cv_file).is_relative_to(generation)
        assert not Path(torch_file).is_relative_to(generation / "packs")
        assert not Path(cv_file).is_relative_to(generation / "packs")
        assert pack_drift(python, lock, facts) == ()
        assert PackLock.read(generation / "packs/example/pack-lock.json") == lock
        receipt = json.loads((generation / "packs/example/installed-wheels.json").read_text())
        assert receipt == {
            "format": "dinkster.installed-wheels/1",
            "wheels": [
                {
                    "filename": extra.filename,
                    "sha256": extra.sha256,
                    "size": extra.size,
                    "registry": extra.registry,
                }
            ],
            "inherited": ["opencv-contrib-python-headless", "torch"],
        }
        assert [path.name for path in store.rglob("*.whl")] == [extra.filename]
        assert not list((generation / "packs").rglob("torch*"))


@pytest.mark.parametrize("rooted", [True, False])
@pytest.mark.parametrize(
    ("requirement", "transitive", "accepted"),
    [
        ("dinkster-inference", "dinkster-inference>=0.1.1", True),
        ("dinkster-inference<0.1.2", "dinkster-inference", False),
        ("dinkster-inference", "dinkster-inference<0.1.2", False),
        ("dinkster-inference", 'dinkster-inference<0.1.2; sys_platform == "win32"', True),
        ("dinkster-inference", 'dinkster-inference<0.1.2; extra == "limited"', True),
        ("fixture-extra[limited]", 'dinkster-inference<0.1.2; extra == "limited"', False),
    ],
)
def test_plan_checks_engine_requirements_instead_of_publication_versions(
    tmp_path: Path, requirement: str, transitive: str, accepted: bool, rooted: bool
) -> None:
    private = _wheel(
        tmp_path,
        "fixture-extra",
        "1.2",
        "fixture_extra",
        "VALUE = 1\n",
        requires=(transitive,),
    )
    lock = PackLock(
        "example",
        "linux-cpu",
        ">=3.12",
        (requirement, "fixture-extra==1.2") if rooted else (requirement,),
        (),
        (
            LockedDistribution("dinkster-inference", "0.1.1", ()),
            LockedDistribution("fixture-extra", "1.2", (private,)),
        ),
    )
    facts = GenerationPackages(
        {"dinkster-inference": "0.1.2"},
        "3.12.0",
        frozenset({Tag("py3", "none", "any")}),
        {"sys_platform": "linux"},
    )
    plan = plan_pack(lock, facts, variant="linux-cpu", own_environment_size=100)
    if accepted:
        assert not plan.conflicts, plan.conflicts
        assert plan.wheels == (private,)
        store = tmp_path / "store"
        fetch_wheel(private, store)
        (tmp_path / private.filename).unlink()
        assert (
            plan_pack(
                lock,
                facts,
                variant="linux-cpu",
                own_environment_size=100,
                wheel_store=store,
                offline=True,
            )
            == plan
        )
    else:
        assert any("dinkster-inference" in conflict for conflict in plan.conflicts)


def test_plan_names_conflicts_and_own_environment_size(tmp_path: Path) -> None:
    torch = _wheel(tmp_path, "torch", "2.15", "torch", "")
    compiled = LockedWheel(
        "kernel-1-cp312-cp312-win_amd64.whl", 100, "a" * 64, "https://example.invalid/kernel.whl"
    )
    lock = PackLock(
        "example",
        "linux-cpu",
        ">=3.15",
        ("torch>=2.15",),
        (),
        (
            LockedDistribution("torch", "2.15", (torch,)),
            LockedDistribution("kernel", "1", (compiled,)),
        ),
    )
    facts = GenerationPackages(
        {"torch": "2.14.1"}, "3.14.6", frozenset({Tag("py3", "none", "any")}), {}
    )
    plan = plan_pack(lock, facts, variant="linux-cpu", own_environment_size=7654321)
    assert any("torch" in conflict and "2.14.1" in conflict for conflict in plan.conflicts)
    assert any("Python" in conflict for conflict in plan.conflicts)
    assert any(
        "kernel" in conflict and "no compatible wheel" in conflict for conflict in plan.conflicts
    )
    with pytest.raises(ProvisionError, match=r"own environment \(7654321 bytes\)"):
        plan.require_layering()


def test_custom_archive_locks_use_selected_sources_and_target_constraints(tmp_path: Path) -> None:
    from dinkster_registry import LockedPack, Lockfile, build_artifact

    pinned = _wheel(tmp_path, "fixture-extra", "1.2", "fixture_extra", "value = 13\n")
    newer = _wheel(tmp_path, "fixture-extra", "2.0", "fixture_extra", "value = 99\n")
    source = tmp_path / "selected-source"
    source.mkdir()
    manifest = source / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname="example"\nnamespaces=["example"]\npublisher="dinkster"\n'
        'requires=["fixture-extra>=1"]\n[pack.entry]\nnodes="nodes:NODES"\n'
    )
    (source / "nodes.py").write_text('raise AssertionError("pack code must not be imported")\n')
    archive = tmp_path / "selected.zip"
    identity = build_artifact(source, archive)
    artifact = tmp_path / f"{identity.removeprefix('blake3:')}.zip"
    archive.rename(artifact)
    catalog = tmp_path / "selected.lock"
    catalog.write_text(
        Lockfile.of(
            [LockedPack("example", "3.7", identity, "dinkster", ("example",))]
        ).record_json()
    )
    # The checkout is not an input: resolution must use the locked archive's requirements.
    manifest.write_text("invalid checkout TOML\n")
    generation = tmp_path / "unsealed"
    generation.mkdir()
    constraints = generation / "constraints.txt"
    constraints.write_text(f"fixture-extra==1.2 --hash=sha256:{pinned.sha256}\n")
    original = {path: path.read_bytes() for path in (catalog, artifact, constraints)}
    command = [
        sys.executable,
        "-m",
        "dinkster.generation_packs",
        "--generation",
        str(generation),
        "--catalog-lock",
        str(catalog),
        "--artifacts",
        str(tmp_path),
        "--prepare-locks",
        str(tmp_path / "resolved"),
        "--variant",
        "my-custom-cuda",
        "--find-links",
        str(tmp_path),
        "--json",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["errors"] == []
    assert response["packs"] == ["example"]
    assert len(response["locks"]) == 1
    lock = PackLock.read(Path(response["locks"][0]))
    assert lock.pack == "example" and lock.variant == "my-custom-cuda"
    assert lock.requirements == ("fixture-extra>=1",)
    assert [(package.name, package.version) for package in lock.packages] == [
        ("fixture-extra", "1.2")
    ]
    assert lock.packages[0].wheels == (pinned,)
    assert {path: path.read_bytes() for path in original} == original
    assert sorted(path.name for path in generation.iterdir()) == ["constraints.txt"]
    constraints.write_text(f"fixture-extra==2.0 --hash=sha256:{newer.sha256}\n")
    command[command.index("--prepare-locks") + 1] = str(tmp_path / "newer-target")
    updated = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert updated.returncode == 0, updated.stderr
    updated_lock = PackLock.read(Path(json.loads(updated.stdout)["locks"][0]))
    assert updated_lock.packages[0].version == "2.0"
    assert updated_lock.packages[0].wheels == (newer,)
    assert PackLock.read(Path(response["locks"][0])).packages[0].version == "1.2"
    artifact.write_bytes(b"damaged selected archive")
    command[command.index("--prepare-locks") + 1] = str(tmp_path / "refused")
    refused = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert refused.returncode == 1
    assert json.loads(refused.stdout)["errors"]
    assert not list((tmp_path / "refused").glob("*.pack-lock.json"))
    assert not (generation / "dinkster.lock").exists()


@pytest.mark.parametrize("pack", ["dinkster-test-helpers", "dinkster-test-consumer"])
def test_helper_and_consumer_locks_layer_on_later_compatible_engine(
    tmp_path: Path, pack: str
) -> None:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    engine_a = "0.0.3.dev235+g3a30be8a6f04"
    engine_b = "0.0.3.dev236+g0123456789ab"
    _wheel(wheels, "dinkster-api", engine_a, "dinkster_api", "VALUE = 1\n")
    torch = _wheel(wheels, "torch", "2.14.1", "torch", "VALUE = 41\n")
    _wheel(
        wheels,
        "dinkster-test-helpers",
        "0.0.2",
        "dinkster_test_helpers",
        "from importlib.metadata import version\n"
        "def engine_version(): return version('dinkster-api')\n",
        requires=("dinkster-api", "torch<2.15"),
    )
    _wheel(
        wheels,
        "dinkster-test-consumer",
        "0.0.4",
        "dinkster_test_consumer",
        "NODES = []\n",
        requires=("dinkster-test-helpers>=0.0.2,<1",),
    )
    constraints = tmp_path / "constraints.txt"
    constraints.write_text(f"dinkster-api=={engine_a}\ntorch==2.14.1\n")
    version = "0.0.2" if pack == "dinkster-test-helpers" else "0.0.4"
    lock = resolve_pack_lock(
        pack=pack,
        variant="linux-cpu",
        requirements=[f"{pack}=={version}"],
        python=Path(sys.executable),
        constraints=constraints,
        find_links=[wheels],
        engine=">=0.0.3.dev0,<0.1",
    )
    lock_path = tmp_path / "lock.json"
    lock.write(lock_path)
    lock = PackLock.read(lock_path)
    newer = tmp_path / "newer"
    newer.mkdir()
    api = _wheel(newer, "dinkster-api", engine_b, "dinkster_api", "VALUE = 2\n")
    parent = _generation(tmp_path / "generation/python", [api, torch])
    facts = GenerationPackages.inspect(parent)
    artifacts = {path.name: path.read_bytes() for path in wheels.iterdir()}
    plan = plan_pack(lock, facts, variant="linux-cpu", own_environment_size=9000)
    assert not plan.conflicts, plan.conflicts
    assert "dinkster-api" not in {package.name for package in lock.packages}
    for versions, refused in (
        ({**facts.versions, "torch": "2.15.0"}, "torch"),
        ({**facts.versions, "dinkster-api": "0.1.0"}, "dinkster-api"),
        (
            {key: value for key, value in facts.versions.items() if key != "dinkster-api"},
            "dinkster-api",
        ),
    ):
        bad = plan_pack(
            lock,
            replace(facts, versions=versions),
            variant="linux-cpu",
            own_environment_size=9000,
        )
        assert any(refused in conflict for conflict in bad.conflicts), bad.conflicts
    store = tmp_path / "store"
    store.mkdir()
    for wheel in plan.wheels:
        (store / wheel.filename).write_bytes(artifacts[wheel.filename])
    python = provision_pack(
        plan,
        generation_python=parent,
        packs=tmp_path / "generation/packs",
        wheel_store=store,
        uv_cache=tmp_path / "cache",
        variant="linux-cpu",
        offline=True,
    )
    if pack == "dinkster-test-consumer":
        result = subprocess.check_output(
            [
                str(python),
                "-I",
                "-c",
                "from dinkster_test_helpers import engine_version; print(engine_version())",
            ],
            text=True,
        )
        assert result.strip() == engine_b
    assert {path.name: path.read_bytes() for path in wheels.iterdir()} == artifacts


def test_transitive_api_constraints_and_legacy_exact_locks_remain_binding(tmp_path: Path) -> None:
    api = _wheel(tmp_path, "dinkster-api", "0.0.3.dev235", "dinkster_api", "value = 235\n")
    _wheel(
        tmp_path,
        "fixture-extra",
        "1.2",
        "fixture_extra",
        "value = 13\n",
        requires=("dinkster-api==0.0.3.dev235",),
    )
    with zipfile.ZipFile(tmp_path / "fixture_extra-1.2-py3-none-any.whl", "a") as archive:
        archive.writestr(
            "fixture_extra/_vendor/other-1.0.dist-info/METADATA",
            "Name: other\nVersion: 1.0\nRequires-Dist: dinkster-api==99\n",
        )
    parent = _generation(tmp_path / "generation", [api])
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("dinkster-api==0.0.3.dev235\n")
    facts = GenerationPackages.inspect(parent)
    for engine in (None, ">=0.0.3.dev0,<0.1"):
        lock = resolve_pack_lock(
            pack="example",
            variant="linux-cpu",
            requirements=["fixture-extra>=1"],
            python=parent,
            constraints=constraints,
            find_links=[tmp_path],
            engine=engine,
        )
        path = tmp_path / "lock.json"
        lock.write(path)
        restored = PackLock.read(path)
        assert restored == lock
        assert not plan_pack(
            restored, facts, variant="linux-cpu", own_environment_size=100
        ).conflicts
        newer = replace(facts, versions={**facts.versions, "dinkster-api": "0.0.3.dev240"})
        refused = plan_pack(restored, newer, variant="linux-cpu", own_environment_size=100)
        assert any("dinkster-api" in conflict for conflict in refused.conflicts)
        assert ("dinkster-api" in {package.name for package in lock.packages}) == (engine is None)


def test_local_resolution_uses_constraints_and_provision_rejects_bad_hash(tmp_path: Path) -> None:
    extra = _wheel(tmp_path, "fixture-extra", "1.2", "fixture_extra", "value = 13\n")
    _wheel(tmp_path, "fixture-extra", "2.0", "fixture_extra", "value = 99\n")
    parent = _generation(tmp_path / "generation", [])
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("fixture-extra==1.2\n")
    lock = resolve_pack_lock(
        pack="example",
        variant="linux-cpu",
        requirements=["fixture-extra>=1"],
        python=parent,
        constraints=constraints,
        find_links=[tmp_path],
    )
    assert [(package.name, package.version) for package in lock.packages] == [
        ("fixture-extra", "1.2")
    ]
    assert lock.packages[0].wheels[0].sha256 == extra.sha256
    pack_root = tmp_path / "checkout"
    pack_root.mkdir()
    (pack_root / "dinkster-pack.toml").write_text(
        '[pack]\nname="example"\nnamespaces=["example"]\nrequires=["fixture-extra>=1"]\n'
        '[pack.entry]\nnodes="example:NODES"\n'
    )
    published = tmp_path / "published.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dinkster_workers.pack_lock",
            str(pack_root),
            "--variant",
            "linux-cpu",
            "--generation-python",
            str(parent),
            "--constraints",
            str(constraints),
            "--find-links",
            str(tmp_path),
            "--output",
            str(published),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert PackLock.read(published) == lock
    bad = replace(
        lock, packages=(replace(lock.packages[0], wheels=(replace(extra, sha256="b" * 64),)),)
    )
    plan = plan_pack(
        bad, GenerationPackages.inspect(parent), variant="linux-cpu", own_environment_size=9
    )
    with pytest.raises(ProvisionError, match="hash or size mismatch"):
        provision_pack(
            plan,
            generation_python=parent,
            packs=tmp_path / "packs",
            wheel_store=tmp_path / "store",
            uv_cache=tmp_path / "cache",
            variant="linux-cpu",
        )
    assert not venv_python(tmp_path / "packs/example").exists()


def test_direct_local_wheel_wins_over_same_version_retained_wheel(tmp_path: Path) -> None:
    older, selected = tmp_path / "older", tmp_path / "selected"
    older.mkdir()
    selected.mkdir()
    stale = _wheel(older, "fixture-extra", "1.2", "fixture_extra", "value = 13\n")
    chosen = _wheel(selected, "fixture-extra", "1.2", "fixture_extra", "value = 29\n")
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("fixture-extra==1.2\n")
    lock = resolve_pack_lock(
        pack="example",
        variant="linux-cpu",
        requirements=["fixture-extra>=1", f"fixture-extra @ {chosen.url}"],
        python=Path(sys.executable),
        constraints=constraints,
        find_links=[older, selected],
    )
    assert stale.sha256 != chosen.sha256
    assert lock.packages == (LockedDistribution("fixture-extra", "1.2", (chosen,)),)
    path = tmp_path / "pack-lock.json"
    lock.write(path)
    assert PackLock.read(path) == lock
    parent = _generation(tmp_path / "generation", [stale])
    plan = plan_pack(
        lock, GenerationPackages.inspect(parent), variant="linux-cpu", own_environment_size=9
    )
    assert plan.conflicts
    with pytest.raises(ProvisionError, match="does not satisfy the requirement"):
        plan.require_layering()


def test_publisher_resolution_rejects_source_only_packages(tmp_path: Path) -> None:
    archive = tmp_path / "source_only_fixture-1.0.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr(
            "source_only_fixture-1.0/pyproject.toml",
            '[build-system]\nrequires=["hatchling"]\nbuild-backend="hatchling.build"\n'
            '[project]\nname="source-only-fixture"\nversion="1.0"\n',
        )
        source.writestr(
            "source_only_fixture-1.0/PKG-INFO",
            "Metadata-Version: 2.1\nName: source-only-fixture\nVersion: 1.0\n",
        )
        source.writestr("source_only_fixture-1.0/source_only_fixture.py", "VALUE = 1\n")
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("")
    with pytest.raises(
        ProvisionError, match="pack example: source-only-fixture==1.0: source-only closure"
    ):
        resolve_pack_lock(
            pack="example",
            variant="linux-cpu",
            requirements=[f"source-only-fixture @ {archive.as_uri()}"],
            python=Path(sys.executable),
            constraints=constraints,
            find_links=[tmp_path],
        )

    empty = PackLock("example", "linux-cpu", ">=3.12", (), (), ())
    published = tmp_path / "pack-lock.json"
    empty.write(published)
    assert PackLock.read(published) == empty
    shared = replace(empty, packages=(LockedDistribution("source-only-fixture", "1.0", ()),))
    shared.write(published)
    assert PackLock.read(published) == shared
    facts = GenerationPackages({"source-only-fixture": "1.0"}, "3.12.0", frozenset(), {})
    plan = plan_pack(shared, facts, variant="linux-cpu", own_environment_size=9)
    assert plan.conflicts == ()
    assert plan.wheels == ()


def test_generation_adapter_preflights_rebuilds_and_reports_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    extra = _wheel(tmp_path, "fixture-extra", "1.2", "fixture_extra", "value = 23\n")
    lock = PackLock(
        "example",
        "linux-cpu",
        ">=3.12",
        ("fixture-extra==1.2",),
        (),
        (LockedDistribution("fixture-extra", "1.2", (extra,)),),
    )
    old = tmp_path / "generations/1"
    (old / "packs/example").mkdir(parents=True)
    lock.write(old / "packs/example/pack-lock.json")
    current = tmp_path / "current"
    current.write_text("1\n")
    target = tmp_path / "generations/2"
    python = _generation(target / "python", [])
    constraints = target / "constraints.txt"
    constraints.write_text("packaging==26.2\n")
    context = GenerationContext(
        target,
        python,
        constraints,
        "linux-cpu",
        PROTOCOL_VERSION,
        tmp_path / "wheels",
        tmp_path / "cache",
        9000,
    )
    with pytest.raises(ProvisionError, match="source generation is required"):
        preflight_generation(context)
    plans = preflight_generation(context, source_generation=old)
    assert plans[0].wheels == (extra,)
    assert not (target / "packs").exists()
    assert not context.wheel_store.exists()
    sealed = target / "generation.json"
    sealed.write_text("{}")
    with pytest.raises(ProvisionError, match="fresh, inactive target"):
        rebuild_generation(context, plans)
    sealed.unlink()
    current.write_text("2\n")
    with pytest.raises(ProvisionError, match="fresh, inactive target"):
        rebuild_generation(context, plans)
    current.write_text("1\n")
    unrelated = target / "packs/unrelated"
    unrelated.mkdir(parents=True)
    with pytest.raises(ProvisionError, match="already has pack venvs"):
        rebuild_generation(context, plans)
    unrelated.rmdir()
    assert not context.wheel_store.exists()
    context = replace(context, offline=True)

    def refuse_network(*args, **kwargs):
        pytest.fail("offline pack rebuilding attempted a download")

    with monkeypatch.context() as patch:
        patch.setattr(urllib.request, "urlopen", refuse_network)
        with pytest.raises(ProvisionError, match="offline pack wheel missing or corrupt"):
            rebuild_generation(context, plans)
        assert not context.wheel_store.exists()
        context.wheel_store.mkdir()
        cached = context.wheel_store / extra.filename
        cached.write_bytes(b"corrupt cached pack wheel")
        with pytest.raises(ProvisionError, match="offline pack wheel missing or corrupt"):
            rebuild_generation(context, plans)
        assert cached.read_bytes() == b"corrupt cached pack wheel"
        cached.write_bytes((tmp_path / extra.filename).read_bytes())
        assert rebuild_generation(context, plans) == (
            "rebuilt pack example from its dependency lock",
        )
    assert doctor_generation(context) == ()
    assert current.read_text() == "1\n"
    assert (old / "packs/example/pack-lock.json").is_file()
    args = [
        "--generation",
        str(target),
        "--generation-python",
        str(python),
        "--constraints",
        str(constraints),
        "--variant",
        "linux-cpu",
        "--worker-protocol",
        str(PROTOCOL_VERSION),
        "--wheel-store",
        str(context.wheel_store),
        "--uv-cache",
        str(context.uv_cache),
        "--own-environment-size",
        "9000",
    ]
    assert main(["doctor", *args]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["changes"] == []
    assert result["packs"] == [
        {"pack": "example", "conflicts": [], "downloadBytes": 0, "ownEnvironmentBytes": 9000}
    ]
    assert main(["references", *args]) == 0
    assert json.loads(capsys.readouterr().out)["wheelHashes"] == [extra.sha256]
    assert main(["references", *args, "--in-process-pack", "missing"]) == 1
    assert "no recorded dependency lock" in json.loads(capsys.readouterr().out)["changes"][0]
    (target / "packs/unreadable").write_text("not a pack directory")
    assert main(["references", *args]) == 1
    assert "readable pack directories" in json.loads(capsys.readouterr().out)["changes"][0]
    (target / "packs/unreadable").unlink()
    pack_python = venv_python(target / "packs/example")
    unavailable = pack_python.with_name("unavailable")
    pack_python.rename(unavailable)
    assert main(["doctor", *args]) == 1
    assert json.loads(capsys.readouterr().out)["changes"] == [
        "example: interpreter unavailable; rebuild from pack-lock.json"
    ]
    unavailable.rename(pack_python)
    assert main(["terminal", *args, "--pack", "example"]) == 0
    terminal = json.loads(capsys.readouterr().out)["changes"][0]
    assert Path(terminal) == venv_python(target / "packs/example")
    assert main(["terminal", *args]) == 0
    scratch = Path(json.loads(capsys.readouterr().out)["changes"][0])
    assert scratch == venv_python(tmp_path / "scratch/2")
    assert scratch != python
    metadata = next((target / "packs/example").rglob("fixture_extra-1.2.dist-info")) / "METADATA"
    metadata.write_text("Name: fixture-extra\nVersion: 9.0\n")
    assert main(["doctor", *args]) == 1
    drift = json.loads(capsys.readouterr().out)["changes"]
    assert any("fixture-extra" in message and "found 9.0" in message for message in drift)
    metadata.write_text("Name: fixture-extra\nVersion: 1.2\n")
    moved = tmp_path / "relocated-generation"
    shutil.move(target, moved)
    config = moved / "packs/example/pyvenv.cfg"
    config.write_text(config.read_text().replace(str(target), str(moved)))
    result = subprocess.run(
        [
            str(venv_python(moved / "packs/example")),
            "-c",
            "import fixture_extra; print(fixture_extra.value)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "23"


def test_generation_preflight_never_silently_omits_packs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "packs/broken").mkdir(parents=True)
    target = tmp_path / "target"
    python = _generation(target / "python", [])
    constraints = target / "constraints.txt"
    constraints.write_text("")
    context = GenerationContext(
        target,
        python,
        constraints,
        "linux-cpu",
        PROTOCOL_VERSION,
        tmp_path / "wheels",
        tmp_path / "cache",
        1000,
    )
    with pytest.raises(ProvisionError, match="broken.*lock missing"):
        preflight_generation(context, source_generation=source)
    wrong_variant = PackLock("broken", "win-cpu", ">=3.12", (), (), ())
    wrong_variant.write(source / "packs/broken/pack-lock.json")
    plans = preflight_generation(context, source_generation=source)
    assert plans[0].conflicts == ("variant win-cpu does not match linux-cpu",)
    with pytest.raises(ProvisionError, match="variant win-cpu"):
        rebuild_generation(context, plans)
    assert not (target / "packs").exists()
    replacement = tmp_path / "replacement.json"
    replace(wrong_variant, variant="linux-cpu").write(replacement)
    assert not preflight_generation(context, source_generation=source, locks=[replacement])[
        0
    ].conflicts


def test_collection_lists_unreadable_locks_and_valid_sibling_wheel_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source"
    for name in ("absent", "corrupt", "healthy"):
        (source / "packs" / name).mkdir(parents=True)
    (source / "packs/corrupt/pack-lock.json").write_text("not a lock")
    extra = _wheel(tmp_path, "fixture-extra", "1.2", "fixture_extra", "value = 73\n")
    (tmp_path / extra.filename).write_bytes(b"corrupt wheel")
    PackLock(
        "healthy",
        "linux-cpu",
        ">=3.12",
        ("fixture-extra==1.2",),
        (),
        (LockedDistribution("fixture-extra", "1.2", (extra,)),),
    ).write(source / "packs/healthy/pack-lock.json")
    target = tmp_path / "target"
    python = _generation(target / "python", [])
    constraints = target / "constraints.txt"
    constraints.write_text("")
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    arguments = [
        "--generation",
        str(target),
        "--generation-python",
        str(python),
        "--constraints",
        str(constraints),
        "--variant",
        "linux-cpu",
        "--worker-protocol",
        str(PROTOCOL_VERSION),
        "--wheel-store",
        str(tmp_path / "wheels"),
        "--uv-cache",
        str(tmp_path / "cache"),
        "--own-environment-size",
        "1000",
        "--source-generation",
        str(source),
        "--in-process-pack",
        "absent",
    ]
    assert main(["collect", *arguments]) == 1
    diagnostic = capsys.readouterr().err
    assert "pack absent: recorded dependency lock missing" in diagnostic
    assert "pack corrupt:" in diagnostic and "corrupt/pack-lock.json" in diagnostic.replace(
        "\\", "/"
    )
    assert "in-process packs have no dependency lock: absent" in diagnostic
    assert f"pack healthy: {extra.filename}: wheel hash or size mismatch" in diagnostic
    observed = GenerationPackages.inspect(python)
    recorded = json.dumps(
        {"tags": [str(tag) for tag in observed.tags], "markers": dict(observed.markers)}
    )
    unavailable = list(arguments)
    unavailable[unavailable.index("--generation-python") + 1] = str(tmp_path / "absent-python")
    assert main(["collect", *unavailable, "--compatibility", recorded]) == 1
    diagnostic = capsys.readouterr().err
    assert "pack absent: recorded dependency lock missing" in diagnostic
    assert f"pack healthy: {extra.filename}: wheel hash or size mismatch" in diagnostic
    assert "cannot inspect generation interpreter" not in diagnostic
    assert {path: path.read_bytes() for path in source.rglob("*") if path.is_file()} == before
    assert not (target / "packs").exists()
    assert main(["preflight", *arguments]) == 1
    assert "pack absent: recorded dependency lock missing" in capsys.readouterr().err


def test_in_process_packs_stage_in_fresh_generation_and_terminals_use_scratch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    extra = _wheel(tmp_path, "fixture-extra", "1.2", "fixture_extra", "value = 73\n")
    api = _wheel(tmp_path, "dinkster-api", "0.0.3.dev240", "dinkster_api", "value = 240\n")
    target = tmp_path / "generations/2"
    python = _generation(target / "python", [api])
    constraints = target / "constraints.txt"
    constraints.write_text("")
    lock = PackLock(
        "in-process",
        "linux-cpu",
        ">=3.12",
        ("fixture-extra==1.2", "dinkster-api<0.1,>=0.0.3.dev0"),
        (),
        (LockedDistribution("fixture-extra", "1.2", (extra,)),),
        engine=">=0.0.3.dev0,<0.1",
    )
    paths = []
    for name in ("in-process", "shared", "isolated"):
        path = tmp_path / f"{name}.json"
        replace(lock, pack=name).write(path)
        paths.append(path)
    context = GenerationContext(
        target,
        python,
        constraints,
        "linux-cpu",
        PROTOCOL_VERSION,
        tmp_path / "wheels",
        tmp_path / "cache",
        1000,
        in_process_packs=("in-process", "shared"),
    )
    plans = preflight_generation(context, locks=paths)
    assert [plan.lock.pack for plan in plans] == ["in-process", "shared", "isolated"]
    assert [plan.wheels for plan in plans] == [(extra,), (), ()]
    assert not context.wheel_store.exists()
    assert not (target / "packs").exists()
    seal = target / "generation.json"
    seal.write_text("{}")
    with pytest.raises(ProvisionError, match="fresh, inactive target"):
        rebuild_generation(context, plans)
    assert not context.wheel_store.exists()
    seal.unlink()
    rebuild_generation(context, plans)
    for name in context.in_process_packs:
        assert {path.name for path in (target / "packs" / name).iterdir()} == {
            "pack-lock.json",
            "installed-wheels.json",
        }
    first_receipt = json.loads((target / "packs/in-process/installed-wheels.json").read_text())
    shared_receipt = json.loads((target / "packs/shared/installed-wheels.json").read_text())
    isolated_receipt = json.loads((target / "packs/isolated/installed-wheels.json").read_text())
    assert [wheel["sha256"] for wheel in first_receipt["wheels"]] == [extra.sha256]
    assert first_receipt["inherited"] == ["dinkster-api"]
    assert shared_receipt["wheels"] == isolated_receipt["wheels"] == []
    assert (
        shared_receipt["inherited"]
        == isolated_receipt["inherited"]
        == [
            "dinkster-api",
            "fixture-extra",
        ]
    )
    isolated = venv_python(target / "packs/isolated")
    result = subprocess.check_output(
        [
            str(isolated),
            "-c",
            "import fixture_extra, dinkster_api; print(fixture_extra.value, dinkster_api.value)",
        ],
        text=True,
    )
    assert result.strip() == "73 240"
    assert doctor_generation(context) == ()
    args = [
        "--generation",
        str(target),
        "--generation-python",
        str(python),
        "--constraints",
        str(constraints),
        "--variant",
        "linux-cpu",
        "--worker-protocol",
        str(PROTOCOL_VERSION),
        "--wheel-store",
        str(context.wheel_store),
        "--uv-cache",
        str(context.uv_cache),
        "--own-environment-size",
        "1000",
        "--in-process-pack",
        "in-process",
        "--in-process-pack",
        "shared",
    ]
    assert main(["terminal", *args, "--pack", "in-process"]) == 0
    scratch = Path(json.loads(capsys.readouterr().out)["changes"][0])
    assert scratch == venv_python(tmp_path / "scratch/2")
    assert scratch != python
    metadata = next((target / "python").rglob("fixture_extra-1.2.dist-info")) / "METADATA"
    metadata.write_text("Name: fixture-extra\nVersion: 9.0\n")
    assert main(["doctor", *args]) == 1
    drift = json.loads(capsys.readouterr().out)["changes"]
    assert any("in-process: fixture-extra: expected 1.2, found 9.0" in item for item in drift)


def test_collection_replaces_in_process_versions_before_layering_siblings(tmp_path: Path) -> None:
    old = _wheel(tmp_path, "fixture-extra", "1.2", "fixture_extra", "value = 13\n")
    updated = _wheel(tmp_path, "fixture-extra", "2.0", "fixture_extra", "value = 79\n")
    target = tmp_path / "generations/2"
    python = _generation(target / "python", [old])
    constraints = target / "constraints.txt"
    constraints.write_text("")
    paths = []
    for name in ("a-isolated", "z-parent"):
        path = tmp_path / f"{name}.json"
        PackLock(
            name,
            "linux-cpu",
            ">=3.12",
            ("fixture-extra==2.0",),
            (),
            (LockedDistribution("fixture-extra", "2.0", (updated,)),),
        ).write(path)
        paths.append(path)
    context = GenerationContext(
        target,
        python,
        constraints,
        "linux-cpu",
        PROTOCOL_VERSION,
        tmp_path / "wheels",
        tmp_path / "cache",
        1000,
        in_process_packs=("z-parent",),
    )
    before = {path: path.read_bytes() for path in target.rglob("*") if path.is_file()}
    plans = preflight_generation(context, locks=paths, engine_only=True)
    assert [plan.lock.pack for plan in plans] == ["z-parent", "a-isolated"]
    assert [plan.wheels for plan in plans] == [(updated,), ()]
    assert all(not plan.conflicts for plan in plans)
    assert {path: path.read_bytes() for path in target.rglob("*") if path.is_file()} == before
    assert not context.wheel_store.exists()
    assert not (target / "packs").exists()


def test_in_process_dependencies_conflict_before_downloads(tmp_path: Path) -> None:
    target = tmp_path / "generations/2"
    python = _generation(target / "python", [])
    constraints = target / "constraints.txt"
    constraints.write_text("")
    first = _wheel(tmp_path, "fixture-extra", "1.2", "fixture_extra", "value = 1\n")
    second = _wheel(tmp_path, "fixture-extra", "2.0", "fixture_extra", "value = 2\n")
    paths = []
    for name, version, wheel in (("shared", "1.2", first), ("isolated", "2.0", second)):
        path = tmp_path / f"{name}.json"
        PackLock(
            name,
            "linux-cpu",
            ">=3.12",
            (f"fixture-extra=={version}",),
            (),
            (LockedDistribution("fixture-extra", version, (wheel,)),),
        ).write(path)
        paths.append(path)
    context = GenerationContext(
        target,
        python,
        constraints,
        "linux-cpu",
        PROTOCOL_VERSION,
        tmp_path / "wheels",
        tmp_path / "cache",
        1000,
        in_process_packs=("shared",),
    )
    plans = preflight_generation(context, locks=paths)
    assert plans[1].lock.pack == "isolated"
    assert plans[1].conflicts == (
        "fixture-extra==2.0: fixture-extra==1.2 does not satisfy the requirement",
    )
    with pytest.raises(ProvisionError, match="fixture-extra==2.0.*fixture-extra==1.2"):
        rebuild_generation(context, plans)
    assert not context.wheel_store.exists()
    assert not (target / "packs").exists()


@pytest.mark.parametrize("windows_error", [False, True], ids=["native", "windows-launch-error"])
def test_doctor_rejects_unavailable_generation_even_without_packs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    windows_error: bool,
) -> None:
    if windows_error:

        def missing_interpreter(
            *args: object, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            raise FileNotFoundError("[WinError 2] The system cannot find the file specified")

        monkeypatch.setattr(subprocess, "run", missing_interpreter)
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("")
    assert (
        main(
            [
                "doctor",
                "--generation",
                str(tmp_path),
                "--generation-python",
                str(tmp_path / "missing-python"),
                "--constraints",
                str(constraints),
                "--variant",
                "linux-cpu",
                "--worker-protocol",
                str(PROTOCOL_VERSION),
                "--wheel-store",
                str(tmp_path / "wheels"),
                "--uv-cache",
                str(tmp_path / "cache"),
                "--own-environment-size",
                "1000",
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["packs"] == []
    assert "missing-python" in result["changes"][0]


@pytest.mark.parametrize("problem", ["engine", "dependency"])
def test_development_pack_build_refuses_incompatible_or_unresolvable_inputs(
    tmp_path: Path, problem: str
) -> None:
    from dinkster_workers.development_sources import build_inputs

    generation, source, index = (tmp_path / name for name in ("generation", "pack", "index"))
    for directory in (generation, source, index):
        directory.mkdir()
    (generation / "engine-manifest.json").write_text('{"wheels": []}')
    (generation / "constraints.txt").write_text("")
    (source / "example").mkdir()
    (source / "example/__init__.py").write_text("NODES = []\n")
    engine = ">=9999" if problem == "engine" else ">=0.0.2"
    requirements = [] if problem == "engine" else ["unavailable-development-helper==9999"]
    (source / "dinkster-pack.toml").write_text(
        '[pack]\nname = "example"\nversion = "1.0"\npublisher = "demo"\n'
        f'engine = "{engine}"\nrequires = {json.dumps(requirements)}\n'
        '[pack.entry]\nnodes = "example:NODES"\n'
    )
    store = tmp_path / "store"
    (store / "wheels").mkdir(parents=True)
    error = (
        "declared engine requirement is incompatible"
        if problem == "engine"
        else "dependency conflict"
    )
    with pytest.raises(ProvisionError, match=error):
        build_inputs(
            {
                "generation": str(generation),
                "store": str(store),
                "artifacts": str(tmp_path / "artifacts"),
                "libraries": [],
                "packs": [str(source)],
                "packPaths": [str(source)],
                "variant": "linux-cpu",
                "accelerator": "cpu",
                "index": index.as_uri(),
            }
        )
    assert not (tmp_path / "artifacts/development-packs.lock").exists()
    assert not (generation / "packs").exists()
