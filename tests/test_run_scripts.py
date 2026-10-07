from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import run as launcher


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher.sys, "argv", ["run.py", "--no-browser", "--port", "4640"])
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        launcher.shutil, "which", lambda name: None if name == "nvidia-smi" else name
    )
    monkeypatch.setenv("DINKSTER_EXECUTION_PYTHON", "wrong-inherited-python")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/release_sources.json").write_text(json.dumps({"commit": "1" * 40}))
    frontend = tmp_path / ".run/frontend"
    (frontend / "packages/app/dist").mkdir(parents=True)
    (frontend / "packages/app/dist/index.html").write_text("editor")
    (tmp_path / ".run/frontend-build").write_text("1" * 40)
    calls = []
    monkeypatch.setattr(launcher, "run", lambda *args, **kwargs: calls.append(args))

    def output(*args, **kwargs):
        if args == ("node", "--version"):
            return "v22.23.2"
        if args == ("git", "status", "--porcelain"):
            return ""
        if args == ("git", "rev-parse", "HEAD"):
            return "1" * 40
        return "True"

    monkeypatch.setattr(launcher, "output", output)
    return tmp_path, calls


@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
def test_rerun_skips_frontend_build_and_selects_native_execution(checkout, monkeypatch, system):
    root, calls = checkout
    monkeypatch.setattr(launcher.platform, "system", lambda: system)
    assert launcher.main() == 0
    relative = "Scripts/python.exe" if system == "Windows" else "bin/python"
    assert launcher.os.environ["DINKSTER_EXECUTION_PYTHON"] == str(root / ".venv-torch" / relative)
    assert len(calls) == 3
    assert calls[0][0] == ("powershell.exe" if system == "Windows" else "bash")
    assert calls[1] == (str(root / ".venv" / relative), "-m", "dinkster.cli", "setup")
    assert calls[2][-5:] == (
        "--port",
        "4640",
        "--frontend-root",
        str(root / ".run/frontend/packages/app/dist"),
        "--no-browser",
    )


def test_changed_pin_fetches_checks_out_and_rebuilds(checkout):
    root, calls = checkout
    (root / "scripts/release_sources.json").write_text(json.dumps({"commit": "2" * 40}))
    assert launcher.main() == 0
    assert calls[0] == ("git", "fetch", "origin", "2" * 40)
    assert calls[1] == ("git", "checkout", "--detach", "2" * 40)
    assert calls[2][-2:] == ("install", "--frozen-lockfile")
    assert calls[3][-3:] == ("--filter", "@dinkster/app", "build")
    assert (root / ".run/frontend-build").read_text() == "2" * 40


def test_missing_bundle_rebuilds_even_with_current_stamp(checkout):
    root, calls = checkout
    (root / ".run/frontend/packages/app/dist/index.html").unlink()
    assert launcher.main() == 0
    assert calls[0][-2:] == ("install", "--frozen-lockfile")
    assert calls[1][-1] == "build"


def test_missing_prerequisite_stops_before_mutating_checkout(checkout, monkeypatch):
    _, calls = checkout
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None if name == "node" else name)
    with pytest.raises(RuntimeError, match="node is required"):
        launcher.main()
    assert calls == []


def test_nvidia_selects_cuda_environment(checkout, monkeypatch):
    root, _ = checkout
    python = root / ".venv-gpu/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(launcher.shutil, "which", lambda name: name)
    assert launcher.main() == 0
    assert launcher.os.environ["DINKSTER_EXECUTION_PYTHON"] == str(python)


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell wrapper")
@pytest.mark.parametrize("existing_uv", [False, True])
def test_shell_bootstrap_installs_or_reuses_user_local_uv(tmp_path: Path, existing_uv: bool):
    script = tmp_path / "run.sh"
    script.write_text((Path(__file__).parents[1] / "run.sh").read_text())
    tools = tmp_path / "tools"
    tools.mkdir()
    home = tmp_path / "home with spaces"
    for name in ("dirname", "env", "sh"):
        executable = shutil.which(name)
        assert executable is not None
        (tools / name).symlink_to(executable)
    fake_uv = '#!/bin/sh\nprintf "%s\\n" "$@" > "$HOME/uv-args"\n'
    local_uv = home / ".local/bin/uv"
    if existing_uv:
        local_uv.parent.mkdir(parents=True)
        local_uv.write_text(fake_uv)
        local_uv.chmod(0o755)
    else:
        curl = tools / "curl"
        curl.write_text(
            "#!/bin/sh\ncat <<'INSTALLER'\n"
            'test "$UV_NO_MODIFY_PATH" = 1 || exit 1\n'
            'mkdir -p "$UV_INSTALL_DIR"\n'
            "cat > \"$UV_INSTALL_DIR/uv\" <<'UV'\n"
            + fake_uv
            + 'UV\nchmod +x "$UV_INSTALL_DIR/uv"\nINSTALLER\n'
        )
        curl.chmod(0o755)
        for name in ("cat", "mkdir", "chmod"):
            executable = shutil.which(name)
            assert executable is not None
            (tools / name).symlink_to(executable)
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run(
        [bash, str(script), "--no-browser", "--port", "4640"],
        env={**os.environ, "HOME": str(home), "PATH": str(tools)},
        check=True,
    )
    assert (home / "uv-args").read_text().splitlines() == [
        "run",
        "--no-project",
        "--python",
        "3.12",
        "scripts/run.py",
        "--no-browser",
        "--port",
        "4640",
    ]
    assert not (home / ".profile").exists()
