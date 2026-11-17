"""Project pointers and generation drift are shared with non-CLI clients."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_supervisor.environment_state import (
    Channel,
    EnvironmentError,
    Generation,
    content_digest,
    read_current,
    set_current,
    writer_lock,
)


def test_writer_lock_serializes_processes_and_survives_owner_exit(tmp_path: Path):
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; from dinkster_supervisor.environment_state import writer_lock; "
        "import sys\nwith writer_lock(Path(sys.argv[1])): print('locked', flush=True)",
        str(tmp_path),
    ]
    with (
        pytest.raises(RuntimeError, match="release on error"),
        writer_lock(tmp_path, operation="update"),
    ):
        with (tmp_path / ".lock").open("rb") as metadata:
            metadata.seek(1)
            record = json.load(metadata)
        refused = subprocess.run(command, capture_output=True, text=True, timeout=10)
        assert refused.returncode != 0
        assert "another environment writer holds" in refused.stderr
        assert "operation=update" in refused.stderr
        assert f"pid={os.getpid()}" in refused.stderr
        assert f"acquiredAt={record['acquiredAt']}" in refused.stderr
        raise RuntimeError("release on error")
    assert (tmp_path / ".lock").is_file()
    assert subprocess.run(command, capture_output=True, text=True, timeout=10).stdout == "locked\n"
    holder = [*command]
    holder[2] += "; sys.stdin.read()"
    with subprocess.Popen(
        holder, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    ) as process:
        assert process.stdout is not None
        assert process.stdout.readline() == "locked\n"
        try:
            refused = subprocess.run(command, capture_output=True, text=True, timeout=10)
            assert refused.returncode != 0
        finally:
            process.kill()
            process.wait(timeout=10)
    assert subprocess.run(command, capture_output=True, text=True, timeout=10).stdout == "locked\n"


@pytest.mark.parametrize("replacement_record", [b"", b"{", b"[]"])
def test_writer_lock_does_not_keep_maintenance_identity_after_holder_changes(
    tmp_path: Path, replacement_record: bytes
):
    command = [
        sys.executable,
        "-c",
        """
import sys
import time
from pathlib import Path
from dinkster_supervisor import environment_state as state
sleep = time.sleep

def pause(seconds):
    print('waiting', flush=True)
    sys.stdin.readline()
    sleep(0.15)

state.time.sleep = pause
try:
    with state.writer_lock(Path(sys.argv[1]), operation='update'):
        print('entered', flush=True)
except state.EnvironmentError as exc:
    print(str(exc), flush=True)
""",
        str(tmp_path),
    ]
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    ) as process:
        try:
            assert process.stdin is not None and process.stdout is not None
            with writer_lock(tmp_path, operation="maintenance"):
                assert process.stdout.readline() == "waiting\n"
            with writer_lock(tmp_path, operation="update"):
                # Pause the replacement owner between truncation and metadata publication.
                with (tmp_path / ".lock").open("r+b", buffering=0) as metadata:
                    metadata.seek(1)
                    metadata.truncate(1)
                    metadata.write(replacement_record)
                process.stdin.write("continue\n")
                process.stdin.flush()
                refused = process.stdout.readline()
                assert "another environment writer holds" in refused
                assert "operation=unknown" in refused
                assert "maintenance" not in refused
                assert process.wait(timeout=10) == 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


@pytest.mark.parametrize("operation", ["maintenance", "update"])
def test_writer_lock_bounds_maintenance_wait_and_refuses_explicit_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
):
    from dinkster_supervisor import environment_state as state

    monkeypatch.setattr(state, "_MAINTENANCE_WAIT_SECONDS", 0.05)
    with writer_lock(tmp_path, operation=operation):
        with pytest.raises(EnvironmentError) as caught:
            with writer_lock(tmp_path, operation="update"):
                pytest.fail("second writer entered the transaction")
        assert f"operation={operation}" in str(caught.value)
        assert ("timed out waiting for maintenance" in str(caught.value)) == (
            operation == "maintenance"
        )
    with writer_lock(tmp_path, operation="update"):
        pass


def _generation(root: Path, number: int) -> Path:
    directory = root / "generations" / str(number)
    directory.mkdir(parents=True)
    (directory / "constraints.txt").write_bytes(b"demo==1\n")
    (directory / "python").mkdir()
    (directory / "python" / "module.py").write_bytes(b"ANSWER = 42\n")
    metadata = {
        "format": "dinkster.generation/1",
        "generation": number,
        "base": "sha256:" + "a" * 64,
        "engine": {"version": "0.3.1", "release": "sha256:" + "b" * 64},
        "constraints": "sha256:" + hashlib.sha256(b"demo==1\n").hexdigest(),
        "variant": "linux-cpu",
        "workerProtocol": 1,
        "files": {
            "constraints.txt": "sha256:" + hashlib.sha256(b"demo==1\n").hexdigest(),
            "python/module.py": "sha256:" + hashlib.sha256(b"ANSWER = 42\n").hexdigest(),
        },
    }
    path = directory / "generation.json"
    path.write_text(json.dumps(metadata))
    return path


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_environment_layout_migration_preserves_software_and_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted: bool, legacy: bool
):
    from dinkster_supervisor import environment_migration as migration

    old = _generation(tmp_path, 1)
    manifest = old.parent / "engine-manifest.json"
    manifest.write_text(json.dumps({"capabilities": [] if legacy else ["build-metadata/1"]}))
    record = Generation.read(old)
    replace(record, files=(*record.files, (manifest.name, content_digest(manifest)))).write(old)
    original = old.read_bytes()
    set_current(tmp_path, 1)
    settings = b'{"format": "dinkster.project/1", "channel": "stable"}\n'
    (tmp_path / "project.json").write_bytes(settings)
    history = tmp_path / "history"
    history.mkdir()
    snapshot = b'{"operation":"old","generation":1,"snapshot":{"metadata":{}}}\n'
    (history / "1.json").write_bytes(snapshot)
    cfg = old.parent / "packs" / "demo" / "pyvenv.cfg"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        f"home = {old.parent / 'python' / 'bin'}\ninclude-system-site-packages = false\n"
    )
    rename = migration.durable_rename
    if interrupted:

        def interrupt(source: Path, destination: Path) -> None:
            rename(source, destination)
            raise OSError("interrupted after directory publication")

        monkeypatch.setattr(migration, "durable_rename", interrupt)
        with pytest.raises(OSError, match="interrupted after"):
            migration.migrate_environment(tmp_path)
        monkeypatch.setattr(migration, "durable_rename", rename)

    assert migration.migrate_environment(tmp_path)
    build = tmp_path / "builds" / "1"
    assert (build / "build.json").read_bytes() == original
    if legacy:
        assert (build / "generation.json").read_bytes() == original
    else:
        assert not (build / "generation.json").exists()
    assert Generation.read(build / "build.json").drift(build) == ()
    assert (build / "packs/demo/pyvenv.cfg").read_text() == (
        f"home = {build / 'python' / ('' if os.name == 'nt' else 'bin')}\n"
        "include-system-site-packages = false\n"
    )
    assert (tmp_path / "environment.json").read_bytes() == settings
    assert (history / "1.json").read_bytes() == snapshot
    assert read_current(tmp_path) == 1
    assert not (tmp_path / "project.json").exists()
    assert not (tmp_path / "generations").exists()
    assert not (tmp_path / "layout-migration.json").exists()
    assert not migration.migrate_environment(tmp_path)


def test_environment_layout_migration_refuses_drift_without_renaming(tmp_path: Path):
    from dinkster_supervisor.environment_migration import migrate_environment

    old = _generation(tmp_path, 1)
    (tmp_path / "project.json").write_text('{"format": "dinkster.project/1"}')
    (old.parent / "python/module.py").write_bytes(b"changed\n")
    with pytest.raises(EnvironmentError, match="intact snapshot"):
        migrate_environment(tmp_path)
    assert old.exists()
    assert not (tmp_path / "builds").exists()
    assert not (tmp_path / "environment.json").exists()
    assert not (tmp_path / "layout-migration.json").exists()


def test_pointer_requires_a_complete_generation_and_preserves_previous_on_failure(tmp_path: Path):
    assert read_current(tmp_path) is None
    first = _generation(tmp_path, 1)
    set_current(tmp_path, 1)
    assert (tmp_path / "current").read_bytes() == b"1\n"
    assert read_current(tmp_path) == 1
    with pytest.raises(EnvironmentError, match="cannot read generation.json"):
        set_current(tmp_path, 2)
    assert read_current(tmp_path) == 1
    second = _generation(tmp_path, 2)
    second.write_text(first.read_text())
    with pytest.raises(EnvironmentError, match="number disagree"):
        set_current(tmp_path, 2)
    assert read_current(tmp_path) == 1


def test_doctor_detects_edits_missing_and_added_files_without_reporting_pack_files(tmp_path: Path):
    path = _generation(tmp_path, 7)
    root = path.parent
    generation = Generation.read(path)
    assert generation.drift(root) == ()
    (root / "packs" / "example").mkdir(parents=True)
    (root / "packs" / "example" / "pyvenv.cfg").write_text("home = python\n")
    shared = root / "inference/shared-inference"
    shared.mkdir(parents=True)
    (shared / "pyvenv.cfg").write_text("home = python\n")
    lock = shared / "pack-lock.json"
    lock.write_text("{}\n")
    generation = replace(
        generation,
        files=(
            *generation.files,
            ("inference/shared-inference/pack-lock.json", content_digest(lock)),
        ),
    )
    assert generation.drift(root) == ()
    lock.write_text('{"modified": true}\n')
    assert generation.drift(root) == ("modified: inference/shared-inference/pack-lock.json",)
    lock.write_text("{}\n")
    (root / "python" / "__pycache__").mkdir()
    (root / "python" / "__pycache__" / "module.pyc").write_bytes(b"cache")
    assert generation.drift(root) == ()
    for name in (
        "packs-extra/unexpected.py",
        "python/packs/unexpected.py",
        "python/nested/__pycache__-extra/unexpected.py",
    ):
        extra = root / name
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_bytes(b"injected")
    ignored = root / "python/nested/__pycache__/nested/cache.pyc"
    ignored.parent.mkdir(parents=True)
    ignored.write_bytes(b"cache")
    (root / "python" / "module.py").write_bytes(b"ANSWER = 99\n")
    (root / "constraints.txt").unlink()
    (root / "python" / "unexpected.py").write_bytes(b"injected")
    assert generation.drift(root) == (
        "missing: constraints.txt",
        "modified: python/module.py",
        "unexpected: packs-extra/unexpected.py",
        "unexpected: python/nested/__pycache__-extra/unexpected.py",
        "unexpected: python/packs/unexpected.py",
        "unexpected: python/unexpected.py",
    )


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "C:/escape", "a\\b"])
def test_generation_refuses_file_hashes_outside_its_tree(tmp_path: Path, name: str):
    path = _generation(tmp_path, 1)
    metadata = json.loads(path.read_text())
    metadata["files"][name] = "sha256:" + "c" * 64
    path.write_text(json.dumps(metadata))
    with pytest.raises(EnvironmentError, match="relative file path"):
        Generation.read(path)


@pytest.mark.parametrize("isolated", [None, "example", ["example", "example"], ["../escape"]])
def test_generation_inference_exclusions_round_trip_and_reject_invalid_records(
    tmp_path: Path, isolated: object
) -> None:
    path = _generation(tmp_path, 1)
    original = Generation.read(path)
    assert original.isolated_inference_packs == ()
    replace(original, isolated_inference_packs=("example", "other")).write(path)
    assert Generation.read(path).isolated_inference_packs == ("example", "other")
    metadata = json.loads(path.read_text())
    metadata["isolatedInferencePacks"] = isolated
    path.write_text(json.dumps(metadata))
    with pytest.raises(EnvironmentError, match="isolatedInferencePacks"):
        Generation.read(path)


def _channel_document() -> dict:
    return {
        "format": "dinkster.channel/1",
        "engine": {
            "version": "0.4.0.dev1",
            "release": "sha256:" + "a" * 64,
            "locations": [
                {"type": "mirror", "path": "engine/" + "a" * 64 + "/release-manifest.json"},
                {
                    "type": "github",
                    "repository": "Kosinkadink/Dinkster",
                    "tag": "main-" + "d" * 40,
                    "asset": "release-manifest.json",
                },
            ],
        },
        "base": {
            variant: {
                "id": "sha256:" + sha * 64,
                "locations": [
                    {"type": "mirror", "path": f"base/{variant}/" + sha * 64 + "/manifest.json"}
                ],
            }
            for variant, sha in (("win-cpu", "b"), ("linux-cpu", "c"))
        },
        "minimumDesktopVersion": "0.2.0",
        "signature": None,
    }


def test_channel_selects_an_exact_base_and_never_falls_back_to_cpu(tmp_path: Path):
    path = tmp_path / "main.json"
    path.write_text(json.dumps(_channel_document()))
    channel = Channel.read(path)
    assert channel.engine_version == "0.4.0.dev1"
    assert channel.engine_release == "sha256:" + "a" * 64
    assert channel.base_for("linux-cpu") == "sha256:" + "c" * 64
    assert channel.base_for("win-cpu") == "sha256:" + "b" * 64
    assert [location.kind for location in channel.engine_locations] == ["mirror", "github"]
    assert channel.engine_locations[1].tag == "main-" + "d" * 40
    assert channel.engine_locations[1].repository == "Kosinkadink/Dinkster"
    assert dict(channel.base_locations)["linux-cpu"][0].path == (
        "base/linux-cpu/" + "c" * 64 + "/manifest.json"
    )
    with pytest.raises(EnvironmentError, match="does not provide variant win-nvidia") as error:
        channel.base_for("win-nvidia")
    assert "supported published variants: linux-cpu, win-cpu" in str(error.value)
    assert "--from-scratch" in str(error.value)


@pytest.mark.parametrize(
    "path",
    [
        "engine/2026-10-31-0.4.0-bbbbbbbb/release-manifest.json",
        "engine/2026-10-31-0.4.0-aaaaaaaa/manifest.json",
        "engine/2026-10-31-0.4.0-aaaaaaaa/extra/release-manifest.json",
        "engine/2026-10-31-0.4.0-aaaaaaaa./release-manifest.json",
        "engine/2026-10-31-0.4.0-aaaaaaaa /release-manifest.json",
        "engine/2026-10-31-0.4.0-aaaaaaaa?/release-manifest.json",
        "base/linux-cpu/2026-10-31-py3.14.10-torch2.14.1-aaaaaaaa/manifest.json",
    ],
)
def test_channel_refuses_labelled_location_with_wrong_identity_or_layout(path: str):
    document = _channel_document()
    document["engine"]["locations"] = [{"type": "mirror", "path": path}]
    with pytest.raises(EnvironmentError):
        Channel.parse(document)


@pytest.mark.parametrize(
    "locations",
    [
        [],
        [{"type": "mirror", "path": "engine/latest/release-manifest.json"}],
        [{"type": "mirror", "path": "../release-manifest.json"}],
        [{"type": "https", "path": "https://example.org/release-manifest.json"}],
        [
            {
                "type": "github",
                "repository": "https://example.org/repo",
                "tag": "v1",
                "asset": "release-manifest.json",
            }
        ],
        [
            {
                "type": "github",
                "repository": "owner/repo",
                "tag": "../v1",
                "asset": "release-manifest.json",
            }
        ],
        [
            {
                "type": "github",
                "repository": "owner/repo",
                "tag": "main",
                "asset": "release-manifest.json",
            }
        ],
        [{"type": "github", "repository": "owner/repo", "tag": "v1", "asset": "other.json"}],
    ],
)
def test_channel_refuses_ambiguous_or_unsafe_manifest_locations(tmp_path: Path, locations: list):
    document = _channel_document()
    document["engine"]["locations"] = locations
    path = tmp_path / "main.json"
    path.write_text(json.dumps(document))
    with pytest.raises(EnvironmentError):
        Channel.read(path)


def test_channel_refuses_digest_only_base_and_unverified_signature(tmp_path: Path):
    path = tmp_path / "main.json"
    document = _channel_document()
    document["base"]["win-cpu"] = "sha256:" + "b" * 64
    path.write_text(json.dumps(document))
    with pytest.raises(EnvironmentError, match="id and locations"):
        Channel.read(path)
    document = _channel_document()
    document["signature"] = "not-verified"
    path.write_text(json.dumps(document))
    with pytest.raises(EnvironmentError, match="signed channel verification"):
        Channel.read(path)
