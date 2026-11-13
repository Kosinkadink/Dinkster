from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.check_disposable_publication import check_publication


@pytest.mark.parametrize("publication_fails", [True, False])
def test_publication_cleanup_timeout_preserves_the_publication_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publication_fails: bool
) -> None:
    inputs = tmp_path / "inputs"
    for relative in (
        "bases/linux-cpu/manifest.json",
        "locks/test.pack-lock.json",
        "sources/official-packs.lock",
    ):
        path = inputs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    index = tmp_path / "index"
    index.mkdir()

    def run(command, **kwargs):
        if command[1:3] == ["rm", "--force"]:
            raise subprocess.TimeoutExpired(command, 15)
        if "scripts.publish_official_packs" in command:
            if publication_fails:
                return subprocess.CompletedProcess(command, 1, "", "entry.unresolvable")
            output = "[]"
        elif command[1] == "image":
            output = "sha256:test"
        elif "mint-token" in command:
            output = "test-token"
        elif command[1] == "exec" and "-c" in command:
            output = "[]"
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(subprocess, "run", run)
    error = RuntimeError if publication_fails else subprocess.TimeoutExpired
    match = "entry.unresolvable" if publication_fails else "timed out"
    with pytest.raises(error, match=match):
        check_publication(tmp_path, inputs, index, "registry", "publisher")
