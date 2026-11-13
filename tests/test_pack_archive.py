from __future__ import annotations

import hashlib
import io
import os
import shutil
import sys
import unicodedata
import zipfile
from pathlib import Path

import pytest
from dinkster_workers import ManifestError

from dinkster import manager
from dinkster.pack_archive import (
    MAX_ENTRIES,
    MAX_EXPANDED_BYTES,
    MAX_FILE_BYTES,
    MAX_MANIFEST_BYTES,
    PackArchiveError,
    build_pack_archive,
)
from tests.async_scenarios import run_scenario
from tests.platform_support import symlink_or_skip

MANIFEST = (
    '[pack]\nname = "archive-test"\nversion = "1.0"\nengine = ">=0.9,<1"\n'
    '[pack.entry]\nnodes = "archive_test:NODES"\n'
)
WHEEL_FILENAME = "archive_test-1.0-py3-none-any.whl"


def write_pack(root: Path) -> Path:
    root.mkdir()
    (root / "dinkster-pack.toml").write_bytes(MANIFEST.encode())
    (root / "archive_test").mkdir()
    (root / "archive_test/__init__.py").write_bytes(b"NODES = []\n")
    return root


def test_archive_is_deterministic_and_has_registry_metadata(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures/registry-admission/color-tools"
    first = shutil.copytree(fixture, tmp_path / "first")
    second = shutil.copytree(fixture, tmp_path / "second")
    os.utime(second / "color_tools/__init__.py", (0, 0))
    (second / "color_tools/__init__.py").chmod(0o755)

    first_output = tmp_path / "first-output/color_tools-1.2.3-py3-none-any.whl"
    second_output = tmp_path / "second-output/color_tools-1.2.3-py3-none-any.whl"
    first_digest = build_pack_archive(first, first_output)
    second_digest = build_pack_archive(second, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert (
        first_digest
        == second_digest
        == f"sha256:{hashlib.sha256(first_output.read_bytes()).hexdigest()}"
    )
    with zipfile.ZipFile(first_output) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert archive.comment == b""
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.create_system == 3
            assert info.external_attr == 0o644 << 16
            assert info.extra == b""
            assert info.comment == b""


def test_archive_normalizes_paths_and_omits_excluded_directories(tmp_path: Path) -> None:
    pack = write_pack(tmp_path / "pack")
    decomposed = unicodedata.normalize("NFD", f"caf{chr(0xE9)}.py")
    (pack / "archive_test" / decomposed).write_text("content")
    for excluded in (
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    ):
        directory = pack / "nested" / excluded
        directory.mkdir(parents=True)
        (directory / "ignored").write_text("ignored")

    output = tmp_path / WHEEL_FILENAME
    build_pack_archive(pack, output)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert "archive_test/" + unicodedata.normalize("NFC", decomposed) in names
    assert all("nested" not in name.split("/") for name in names)


def test_archive_rejects_symlinks_and_normalized_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = write_pack(tmp_path / "pack")
    package = pack / "archive_test"
    symlink_or_skip(package / "link.py", package / "__init__.py")
    with pytest.raises(PackArchiveError, match="symlink"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)

    (package / "link.py").unlink()
    entries = list(pack.rglob("*"))
    composed = f"{chr(0xE9)}.py"
    decomposed = unicodedata.normalize("NFD", composed)
    (package / composed).write_text("one")
    (package / decomposed).write_text("two")
    # Normalization-insensitive filesystems cannot store both directory entries.
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda self, pattern: iter([*entries, package / composed, package / decomposed]),
    )
    with pytest.raises(PackArchiveError, match="collides"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)


def test_archive_validates_manifest_before_writing(tmp_path: Path) -> None:
    pack = write_pack(tmp_path / "pack")
    (pack / "dinkster-pack.toml").write_text("not toml = [")
    output = tmp_path / WHEEL_FILENAME
    with pytest.raises(ManifestError):
        build_pack_archive(pack, output)
    assert not output.exists()


def test_entry_limit_accepts_exact_and_rejects_plus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = write_pack(tmp_path / "pack")
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_ENTRIES", 5)
    build_pack_archive(pack, tmp_path / WHEEL_FILENAME)
    (pack / "archive_test/third.py").write_text("")
    with pytest.raises(PackArchiveError, match="6 entries; limit is 5"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)
    assert MAX_ENTRIES == 10_000


def test_file_manifest_expanded_and_archive_limits_are_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = write_pack(tmp_path / "pack")
    largest = pack / "largest.bin"
    largest.write_bytes(b"x" * 1024)
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_FILE_BYTES", 1024)
    build_pack_archive(pack, tmp_path / WHEEL_FILENAME)
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_FILE_BYTES", 1023)
    with pytest.raises(PackArchiveError, match="file 'archive_test/largest.bin'"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)

    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_FILE_BYTES", MAX_FILE_BYTES)
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_MANIFEST_BYTES", len(MANIFEST))
    build_pack_archive(pack, tmp_path / WHEEL_FILENAME)
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_MANIFEST_BYTES", len(MANIFEST) - 1)
    with pytest.raises(PackArchiveError, match="pack manifest"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)

    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_MANIFEST_BYTES", MAX_MANIFEST_BYTES)
    baseline = tmp_path / WHEEL_FILENAME
    build_pack_archive(pack, baseline)
    with zipfile.ZipFile(baseline) as archive:
        expanded = sum(info.file_size for info in archive.infolist())
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_EXPANDED_BYTES", expanded)
    build_pack_archive(pack, tmp_path / WHEEL_FILENAME)
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_EXPANDED_BYTES", expanded - 1)
    with pytest.raises(PackArchiveError, match="expanded archive"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)

    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_EXPANDED_BYTES", MAX_EXPANDED_BYTES)
    build_pack_archive(pack, baseline)
    archive_size = baseline.stat().st_size
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_ARCHIVE_BYTES", archive_size)
    build_pack_archive(pack, tmp_path / WHEEL_FILENAME)
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_ARCHIVE_BYTES", archive_size - 1)
    with pytest.raises(PackArchiveError, match="archive"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)


def test_path_limit_counts_normalized_utf8_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = write_pack(tmp_path / "pack")
    unicode_file = pack / "archive_test" / (chr(0xE9) * 32 + ".py")
    unicode_file.write_text("")
    normalized_bytes = len(("archive_test/" + unicode_file.name).encode("utf-8"))
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_PATH_BYTES", normalized_bytes)
    build_pack_archive(pack, tmp_path / WHEEL_FILENAME)
    longer = unicode_file.with_name(unicode_file.name + "x")
    unicode_file.rename(longer)
    with pytest.raises(PackArchiveError, match="archive path"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)


def test_cli_writes_archive_and_prints_written_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = write_pack(tmp_path / "pack")
    output = tmp_path / "output" / WHEEL_FILENAME
    monkeypatch.setattr(
        sys,
        "argv",
        ["dinkster pack", "archive", str(pack), "--output", str(output)],
    )
    manager.main()
    printed = capsys.readouterr().out.strip()
    assert printed == f"sha256:{hashlib.sha256(output.read_bytes()).hexdigest()}"


@pytest.mark.parametrize(
    ("mode", "state"),
    [
        ("raw", "queued"),
        ("locks", "queued"),
        ("locks", "scanning"),
        ("locks", "retrying"),
        ("locks", "accepted"),
        ("engine", "queued"),
        ("missing", "queued"),
        ("refused", "queued"),
        ("invalid", "rejected"),
        ("invalid", "review"),
        ("invalid", "promoting"),
        ("invalid", "unknown"),
        ("invalid", None),
        ("invalid", ["queued"]),
    ],
)
def test_publication_client_preserves_archive_and_lock_bytes(
    tmp_path: Path, mode: str, state: object
) -> None:
    import json
    import threading
    from email import policy
    from email.parser import BytesParser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from dinkster.registries import NamedRegistry, RegistryPublishError, publish_release

    archive, locks = tmp_path / "demo-1.2.3-py3-none-any.whl", tmp_path / "locks.json"
    archive.write_bytes(b"exact archive\x00\xff\r\n")
    lock_bytes = b'{"win-cpu":{"baseId":"exact binding"}}\n'
    engine = {"commit": "a" * 40, "manifestDigest": "sha256:" + "b" * 64}
    if mode != "missing":
        locks.write_bytes(lock_bytes)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(
                (
                    self.path,
                    self.headers["Content-Type"],
                    self.rfile.read(int(self.headers["Content-Length"])),
                )
            )
            self.send_response(400 if mode == "refused" else 202)
            self.end_headers()
            self.wfile.write(
                b'{"error":{"message":"lock binding refused"}}'
                if mode == "refused"
                else json.dumps({"status": state, "candidateId": "candidate-1"}).encode()
            )

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    registry = NamedRegistry("", f"http://127.0.0.1:{server.server_port}")
    try:
        if mode in {"missing", "refused", "invalid"}:
            with pytest.raises(
                RegistryPublishError,
                match={
                    "missing": "cannot read publication input",
                    "refused": "lock binding refused",
                    "invalid": "malformed receipt",
                }[mode],
            ):
                publish_release(registry, archive, "demo", "1.2.3", locks=locks)
        else:
            result = publish_release(
                registry,
                archive,
                "demo",
                "1.2.3",
                locks=locks if mode in {"locks", "engine"} else None,
                engine=engine if mode == "engine" else None,
            )
            assert result.state == state and result.candidate_id == "candidate-1"
        if mode == "missing":
            assert requests == []
            return
        assert len(requests) == 1
        path, content_type, body = requests[0]
        assert path == "/v1/publish/demo/versions/1.2.3"
        if mode == "raw":
            assert content_type == "application/octet-stream" and body == archive.read_bytes()
        else:
            message = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {content_type}\r\n\r\n".encode() + body
            )
            parts = list(message.iter_parts())
            assert [part.get_param("name", header="content-disposition") for part in parts] == [
                "archive",
                "locks",
            ] + (["engine"] if mode == "engine" else [])
            assert parts[0].get_payload(decode=True) == archive.read_bytes()
            assert parts[1].get_payload(decode=True) == lock_bytes
            if mode == "engine":
                payload = parts[2].get_payload(decode=True)
                assert isinstance(payload, bytes)
                assert json.loads(payload) == engine
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_archived_sources_prepare_catalogs_and_execute_declared_node_ids(tmp_path: Path) -> None:
    import json
    from dataclasses import replace

    from dinkster_graph import graph_from_wire
    from dinkster_registry import artifact_digest, unpack_artifact
    from dinkster_workers import load_manifest
    from dinkster_workers.catalog import read_catalog
    from dinkster_workers.doctor import prepare_catalog
    from dinkster_workers.pack_wheel import pack_manifest_path

    from dinkster.compose import CompositionError, PackSpec, ServingComposer
    from dinkster.packs import pack_info_from_manifest

    saved = tmp_path / "workflow.json"
    original = b'{"nodes":{"adjust":{"nodeType":"color.adjust","inputs":{}}}}'
    saved.write_bytes(original)
    fixture = Path(__file__).parent / "fixtures/registry-admission/color-tools"
    filename = "color_tools-1.2.3-py3-none-any.whl"
    archive = tmp_path / filename
    build_pack_archive(fixture, archive)
    digest = artifact_digest(archive.read_bytes())
    source = unpack_artifact(archive, tmp_path / "sources/color-tools", digest)
    manifest = load_manifest(pack_manifest_path(source))
    assert read_catalog(manifest) is None
    report = prepare_catalog(manifest.path, environment={"PYTHONPATH": str(source)})
    assert report.ok, report
    assert read_catalog(manifest) is not None
    repacked = tmp_path / "repacked" / filename
    build_pack_archive(manifest.root, repacked)
    assert artifact_digest(repacked.read_bytes()) == digest
    assert manifest.compatibility is not None
    info = replace(
        pack_info_from_manifest(manifest),
        version=manifest.compatibility.version,
        artifact_digest=digest,
        source=source.as_uri(),
        publisher=manifest.publisher,
    )
    specs = [
        PackSpec(
            manifest=manifest.path,
            python=sys.executable,
            env={"PYTHONPATH": str(source)},
            packs={manifest.name: info},
            require_catalog=True,
        )
    ]

    async def execute() -> None:
        composer = ServingComposer()
        try:
            ordered = composer.order_pack_entries(specs)
            composer.validate_catalogs(ordered)
            for spec in ordered:
                await composer.add_pack(spec)
            composition = composer.composition
            assert composition.node_packs["color.adjust"] == "color-tools"
            assert composition.packs["color-tools"].version == "1.2.3"
            result = await composition.make_engine(lambda _event: None).run(
                graph_from_wire(json.loads(saved.read_bytes())), ["adjust"]
            )
            assert result.diagnostics == ()
            assert result.outputs == {"adjust": {}}
            assert saved.read_bytes() == original
            (manifest.root / "changed.txt").write_text("source drift")
            with pytest.raises(CompositionError, match="missing or stale"):
                composer.validate_catalogs(ordered)
        finally:
            await composer.close()

    run_scenario(execute())


@pytest.mark.parametrize("problem", [None, "path", "case", "compression", "order", "junk"])
def test_registry_archive_admission_requires_verified_wheel(
    tmp_path: Path, problem: str | None
) -> None:
    from dinkster_registry import artifact_digest
    from dinkster_registry.archive_admission import extracted_archive, run_ingest_integrity

    root = Path(__file__).parent / "fixtures/registry-admission/color-tools"
    wheel = tmp_path / "color_tools-1.2.3-py3-none-any.whl"
    build_pack_archive(root, wheel)
    with zipfile.ZipFile(wheel) as archive:
        files = [(name, archive.read(name)) for name in archive.namelist()]
    if problem == "path":
        files.append(("../escape.py", b"escape"))
    elif problem == "case":
        files.append(("color_tools/__INIT__.py", b"collision"))
    elif problem == "order":
        files.reverse()
    elif problem == "junk":
        files.append(("build/output.py", b"ignored"))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, payload in files:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o644 << 16
            if problem == "compression":
                info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    data = output.getvalue()
    digest = artifact_digest(data)
    result = run_ingest_integrity(data, len(data), digest)
    assert result.blocked is (problem in {"path", "case", "junk"})
    if problem in {None, "compression", "order"}:
        assert result.data == {"artifactDigest": digest, "size": len(data)}
        with extracted_archive(data) as extracted:
            assert (extracted / "__init__.py").read_bytes() == (
                root / "color_tools/__init__.py"
            ).read_bytes()
        assert not extracted.exists()
        assert run_ingest_integrity(data, len(data) + 1, digest).findings[0].code == (
            "integrity.size-mismatch"
        )
        assert run_ingest_integrity(data, len(data), artifact_digest(b"other")).findings[
            0
        ].code == ("integrity.digest-mismatch")
    else:
        assert result.findings[0].code == "integrity.invalid-archive"


@pytest.mark.parametrize(
    ("filename", "payload", "code", "effect"),
    [
        ("clean.py", b"NODES = []", None, None),
        ("secret.txt", b"-----BEGIN PRIVATE KEY-----", "security.confirmed-secret", "block"),
        ("bad.txt", b"dinkster-known-bad-test-payload", "security.known-bad-file", "block"),
        ("native.bin", b"MZbinary", "security.native-binary", "review"),
        ("nested.bin", b"\x1f\x8bgzip", "security.nested-archive", "review"),
        ("nodes.pyc", b"bytecode", "security.obfuscated-code", "review"),
    ],
)
def test_registry_archive_security_preserves_block_and_review(
    tmp_path: Path, filename: str, payload: bytes, code: str | None, effect: str | None
) -> None:
    from dinkster_registry.archive_admission import run_fast_security

    root = tmp_path / "pack"
    shutil.copytree(Path(__file__).parent / "fixtures/registry-admission/color-tools", root)
    (root / "color_tools" / filename).write_bytes(payload)
    archive = tmp_path / "color_tools-1.2.3-py3-none-any.whl"
    build_pack_archive(root, archive)
    result = run_fast_security(archive.read_bytes())
    assert [(finding.code, finding.effect) for finding in result.findings] == (
        [] if code is None else [(code, effect)]
    )
    assert result.blocked is (effect == "block")
    assert result.review is (effect == "review")
    assert result.data == {
        "status": "failed" if effect == "block" else "warning" if effect else "passed",
        "policyVersion": "registry-scan/1",
    }


@pytest.mark.parametrize("engine", ["0.8.9", "0.9.0", "0.9.7", "1.0.0"])
def test_registry_manifest_admission_preserves_engine_range(tmp_path: Path, engine: str) -> None:
    from dinkster_registry.archive_admission import run_manifest_admission

    root = write_pack(tmp_path / "pack")
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        MANIFEST.replace('version = "1.0"', 'version = "2.3.0"')
        + '\n[pack.contracts]\nhost = "dinkster-pack-host/1"\napi = "dinkster-api/v1"\n'
        + '[pack.extra-requires]\ncpu=["helper>=2"]\ncuda=["cuda-helper>=3"]\n'
    )
    result = run_manifest_admission(root, "archive-test", "2.3.0", engine_version=engine)
    assert result.blocked is (engine in {"0.8.9", "1.0.0"})
    if result.blocked:
        message = result.findings[0].message
        assert "archive-test@2.3.0" in message and ">=0.9,<1" in message and engine in message
    else:
        assert result.data is not None and result.data["dinkster"] == ">=0.9,<1"
        assert result.data["pythonExtraRequirements"] == {
            "cpu": ["helper>=2"],
            "cuda": ["cuda-helper>=3"],
        }
    mismatch = run_manifest_admission(root, "archive-test", "2.3.1")
    assert mismatch.blocked and "does not match submission" in mismatch.findings[0].message


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ('license = "LicenseRef-Private"', "requires a root LICENSE"),
        ('license = "not-a-license"', "Unknown license"),
        ('repository = "http://example.org/pack"', "must be an HTTPS URL"),
        ('dinkster = "nonsense"', "Invalid specifier"),
    ],
)
def test_registry_manifest_admission_refuses_invalid_metadata(
    tmp_path: Path, declaration: str, message: str
) -> None:
    from dinkster_registry.archive_admission import run_manifest_admission

    root = write_pack(tmp_path / "pack")
    (root / "dinkster-pack.toml").write_text(
        MANIFEST.replace('version = "1.0"\nengine = ">=0.9,<1"\n', "").replace(
            "[pack]\n", f"[pack]\n{declaration}\n"
        )
        + '\n[pack.contracts]\nhost = "dinkster-pack-host/1"\napi = "dinkster-api/v1"\n'
    )
    result = run_manifest_admission(root, "archive-test", "1.0.0")
    assert result.blocked and result.findings[0].code == "manifest.invalid"
    assert message in result.findings[0].message


@pytest.mark.skipif(sys.platform == "win32", reason="the admission probe requires POSIX limits")
@pytest.mark.parametrize("problem", ["empty", "remote", "exception", "output", "timeout"])
def test_registry_executable_probe_bounds_and_sanitizes(tmp_path: Path, problem: str) -> None:
    from dinkster_registry.archive_admission import ProbeSettings, run_executable_probe

    root = write_pack(tmp_path / "pack")
    if problem == "remote":
        manifest = root / "dinkster-pack.toml"
        manifest.write_text(
            MANIFEST.replace("[pack]\n", '[pack]\nnode-source = "remote-gateway-catalog"\n')
        )
    elif problem == "exception":
        (root / "archive_test/__init__.py").write_text(
            'raise ValueError("private operator detail")\n'
        )
    elif problem == "output":
        (root / "archive_test/__init__.py").write_text(
            'import os\nos.write(1, b"x" * 100000)\nNODES = []\n'
        )
    elif problem == "timeout":
        (root / "archive_test/__init__.py").write_text("import time\ntime.sleep(2)\nNODES = []\n")
    result = run_executable_probe(
        root,
        settings=ProbeSettings(
            probe_timeout_seconds=1 if problem == "timeout" else 30,
            probe_output_limit_bytes=4096,
            probe_operator_tail_bytes=512,
        ),
        require_isolation=os.environ.get("DINKSTER_TEST_FULL_PROBE") == "1",
    )
    codes = {finding.code for finding in result.findings}
    if problem == "remote":
        assert not result.blocked and result.data == {"schemas": []}
    else:
        assert result.blocked
        expected = {
            "empty": "probe.no-schemas",
            "exception": "entry.unresolvable",
            "output": "probe.output-limit",
            "timeout": "probe.timeout",
        }[problem]
        assert expected in codes
    assert all("private operator detail" not in finding.message for finding in result.findings)
    assert result.operator_details is not None
    assert len(str(result.operator_details["stdoutTail"]).encode()) <= 512


def test_registry_probe_refuses_degraded_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_registry.archive_admission import run_executable_probe
    from dinkster_workers.sandbox import ProbeJail

    root = write_pack(tmp_path / "pack")
    monkeypatch.setattr(
        "dinkster_registry.archive_admission.detect_probe_jail",
        lambda: ProbeJail(bwrap="/usr/bin/true", user_namespaces=False),
    )
    result = run_executable_probe(root)
    assert result.blocked and result.findings[0].code == "probe.isolation-unavailable"
    assert result.operator_details == {"jail": "degraded"}
