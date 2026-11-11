from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
    monkeypatch.setenv("UV_PYTHON_PREFERENCE", "only-system")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
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
            return "v26.12.0"
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
    assert launcher.os.environ["DINKSTER_EXECUTION_PYTHON"] == str(root / ".venv" / relative)
    assert launcher.os.environ["UV_PYTHON_PREFERENCE"] == "only-managed"
    assert len(calls) == 2
    assert calls[0] == (sys.executable, str(root / "scripts/setup_envs.py"))
    assert calls[1][:4] == (
        str(root / ".venv" / relative),
        "-m",
        "dinkster.serve",
        "--watch-packs",
    )
    assert calls[1][6:8] == (
        "--port",
        "4640",
    )
    assert calls[1][-2:] == (
        "--frontend-root",
        str(root / ".run/frontend/packages/app/dist"),
    )
    assert calls[1][calls[1].index("--install-root") + 1] == ""


def test_source_browser_waits_for_healthy_engine(checkout, monkeypatch):
    requests = []
    opened = threading.Event()

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            body = json.dumps({"ok": len(requests) > 1}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Health)
    url = f"http://127.0.0.1:{server.server_port}"

    def browser(value):
        assert value == url
        assert len(requests) > 1
        opened.set()

    monkeypatch.setattr(launcher.webbrowser, "open", browser)
    monkeypatch.setattr(launcher.sys, "argv", ["run.py", "--port", str(server.server_port)])
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        assert launcher.main() == 0
        assert opened.wait(5)
        assert requests == ["/api/health", "/api/health"]
    finally:
        server.shutdown()
        server.server_close()
        serving.join()


def test_changed_pin_fetches_checks_out_and_rebuilds(checkout):
    root, calls = checkout
    (root / "scripts/release_sources.json").write_text(json.dumps({"commit": "2" * 40}))
    assert launcher.main() == 0
    assert calls[0] == ("git", "fetch", "origin", "2" * 40)
    assert calls[1] == ("git", "checkout", "--detach", "2" * 40)
    assert "--package=pnpm@11.28.8" in calls[2]
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


@pytest.mark.parametrize("version", ["v22.23.2", "v26.11.9", "v27.0.0"])
def test_unsupported_node_stops_before_mutating_checkout(checkout, monkeypatch, version):
    _, calls = checkout
    monkeypatch.setattr(launcher, "output", lambda *args, **kwargs: version)
    with pytest.raises(RuntimeError, match="Node.js 26.12"):
        launcher.main()
    assert calls == []


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
        "scripts/run.py",
        "--no-browser",
        "--port",
        "4640",
    ]
    assert not (home / ".profile").exists()
