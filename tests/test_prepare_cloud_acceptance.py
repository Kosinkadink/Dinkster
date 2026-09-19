from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts import prepare_cloud_acceptance as preparation


def test_archive_combines_pinned_sources_and_relocks_before_locked_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = tmp_path / "core"
    evidence = tmp_path / "evidence"
    core.mkdir()
    evidence.mkdir()
    commands: list[list[str]] = []

    def run(command: list[str], *, cwd: Path, **_kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        if command[0] == "git":
            assert "archive" in command
            target = Path(
                next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output="))
            )
            if cwd == core:
                assert command[-1] == "a" * 40
                files = {"pyproject.toml": b"core project\n", "uv.lock": b"core lock\n"}
                mode = "w:gz"
            else:
                assert cwd == evidence
                assert command[-2:] == ["b" * 40, "packages/dinkster-acceptance"]
                files = {"packages/dinkster-acceptance/pyproject.toml": b"external package\n"}
                mode = "w"
            with tarfile.open(target, mode) as archive:
                for name, data in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
        elif command[1] == "lock":
            assert (cwd / "packages/dinkster-acceptance/pyproject.toml").read_bytes() == (
                b"external package\n"
            )
            (cwd / "uv.lock").write_bytes(b"combined lock\n")
        else:
            assert command[1] == "sync"
            assert "--locked" in command
            assert (cwd / "uv.lock").read_bytes() == b"combined lock\n"
        return subprocess.CompletedProcess(command, 0)

    def output(command: list[str], *, cwd: Path, **_kwargs: object) -> str:
        if command[0] != "git":
            assert command[0].endswith("dinkster-acceptance-import-check")
            return '{"status":"ok"}'
        assert command[:2] == ["git", "rev-parse"]
        if command[-1].endswith("^{tree}"):
            return ("c" if cwd == core else "d") * 40
        assert command[-1] == ("core-pin^{commit}" if cwd == core else "evidence-pin^{commit}")
        return ("a" if cwd == core else "b") * 40

    monkeypatch.setattr(preparation.subprocess, "run", run)
    monkeypatch.setattr(preparation, "_run", output)
    destination = tmp_path / "release.tar.gz"
    receipt = preparation.prepare(
        core,
        "core-pin",
        destination,
        evidence_root=evidence,
        evidence_revision="evidence-pin",
        uv="test-uv",
    )
    assert receipt["commit"] == "a" * 40
    assert receipt["tree"] == "c" * 40
    assert receipt["evidence_commit"] == "b" * 40
    assert receipt["evidence_tree"] == "d" * 40
    assert receipt["archive_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert json.loads(destination.with_suffix(".gz.receipt.json").read_text()) == receipt
    with tarfile.open(destination) as archive:
        contents = {}
        for name in ("pyproject.toml", "uv.lock", "packages/dinkster-acceptance/pyproject.toml"):
            member = archive.extractfile(name)
            assert member is not None
            contents[name] = member.read()
        assert contents == {
            "pyproject.toml": b"core project\n",
            "uv.lock": b"combined lock\n",
            "packages/dinkster-acceptance/pyproject.toml": b"external package\n",
        }
    assert [command[1] for command in commands if command[0] == "test-uv"] == ["lock", "sync"]
