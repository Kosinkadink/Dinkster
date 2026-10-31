"""Project pointers and generation drift are shared with non-CLI clients."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from dinkster_supervisor.environment_state import (
    Channel,
    EnvironmentError,
    Generation,
    Journal,
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
    with pytest.raises(RuntimeError, match="release on error"), writer_lock(tmp_path):
        refused = subprocess.run(command, capture_output=True, text=True, timeout=10)
        assert refused.returncode != 0
        assert "another environment writer holds" in refused.stderr
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
    (root / "python" / "__pycache__").mkdir()
    (root / "python" / "__pycache__" / "module.pyc").write_bytes(b"cache")
    assert generation.drift(root) == ()
    (root / "python" / "module.py").write_bytes(b"ANSWER = 99\n")
    (root / "constraints.txt").unlink()
    (root / "python" / "unexpected.py").write_bytes(b"injected")
    assert generation.drift(root) == (
        "missing: constraints.txt",
        "modified: python/module.py",
        "unexpected: python/unexpected.py",
    )


def test_generation_preserves_in_process_hosting_and_defaults_to_isolated(tmp_path: Path):
    path = _generation(tmp_path, 1)
    assert Generation.read(path).in_process_packs == ()
    metadata = json.loads(path.read_text())
    metadata["inProcessPacks"] = ["example_pack"]
    path.write_text(json.dumps(metadata))
    generation = Generation.read(path)
    generation.write(path)
    assert Generation.read(path).in_process_packs == ("example_pack",)
    metadata["inProcessPacks"] = ["../escape"]
    path.write_text(json.dumps(metadata))
    with pytest.raises(EnvironmentError, match="array of pack ids"):
        Generation.read(path)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "C:/escape", "a\\b"])
def test_generation_refuses_file_hashes_outside_its_tree(tmp_path: Path, name: str):
    path = _generation(tmp_path, 1)
    metadata = json.loads(path.read_text())
    metadata["files"][name] = "sha256:" + "c" * 64
    path.write_text(json.dumps(metadata))
    with pytest.raises(EnvironmentError, match="relative file path"):
        Generation.read(path)


def test_journal_records_recovery_state_and_retains_failure_diagnostics(tmp_path: Path):
    path = tmp_path / "journal" / "update.json"
    Journal("update", 2, 3, "switching").write(path)
    interrupted = Journal.read(path)
    assert interrupted.incomplete
    assert (interrupted.previous, interrupted.target) == (2, 3)
    Journal("update", 2, 3, "rolled-back", "target failed health check").write(path)
    recovered = Journal.read(path)
    assert not recovered.incomplete
    assert recovered.error == "target failed health check"
    assert json.loads(path.read_text())["format"] == "dinkster.journal/1"


@pytest.mark.parametrize(
    "settings",
    [
        [],
        {},
        {"previous": "{}"},
        {"previous": "{}", "target": 42},
        {"previous": "{}", "target": "not json"},
        {"previous": "{}", "target": "{}"},
    ],
)
def test_journal_refuses_incomplete_or_invalid_project_selection(tmp_path: Path, settings: object):
    path = tmp_path / "journal.json"
    Journal("update", 1, 2, "switching").write(path)
    document = json.loads(path.read_bytes())
    document["projectSettings"] = settings
    path.write_text(json.dumps(document))
    with pytest.raises(EnvironmentError, match="project settings"):
        Journal.read(path)


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
    with pytest.raises(EnvironmentError, match="does not provide variant win-nvidia"):
        channel.base_for("win-nvidia")


def test_channel_resolves_labelled_locations_without_changing_content_identities():
    document = _channel_document()
    engine_path = "engine/2026-10-31-0.4.0.dev1+g1234-aaaaaaaa/release-manifest.json"
    base_path = "base/win-cpu/2026-10-30-py3.14.10-torch2.14.1-bbbbbbbb/manifest.json"
    document["engine"]["locations"][0]["path"] = engine_path
    document["base"]["win-cpu"]["locations"][0]["path"] = base_path
    channel = Channel.parse(document)
    assert channel.engine_release == "sha256:" + "a" * 64
    assert channel.base_for("win-cpu") == "sha256:" + "b" * 64
    assert channel.engine_locations[0].path == engine_path
    assert dict(channel.base_locations)["win-cpu"][0].path == base_path
    document["base"]["win-cpu"]["locations"][0]["path"] = base_path.replace(
        "torch2.14.1-", "torch2.14.1-cu132-"
    )
    assert Channel.parse(document).base_for("win-cpu") == "sha256:" + "b" * 64


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
