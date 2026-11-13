"""Prepare and launch the browser editor from a source checkout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path = ROOT) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def output(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def open_browser_when_ready(url: str) -> None:
    for _attempt in range(600):
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=0.5) as response:
                if response.status == 200 and json.load(response)["ok"] is True:
                    webbrowser.open(url)
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=3639)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")

    for name in ("uv", "git", "node", "npm"):
        if not shutil.which(name):
            raise RuntimeError(f"{name} is required; install it and restart your terminal")
    node_version = tuple(
        int(part) for part in output("node", "--version").lstrip("v").split(".")[:2]
    )
    if not (26, 12) <= node_version < (27, 0):
        raise RuntimeError("Node.js 26.12 or newer within 26.x is required: https://nodejs.org/")

    npm = shutil.which("npm")
    assert npm is not None
    pnpm = (npm, "exec", "--yes", "--package=pnpm@11.28.8", "--", "pnpm")
    pin = json.loads((ROOT / "scripts/release_sources.json").read_text())
    frontend = ROOT / ".run/frontend"
    frontend.parent.mkdir(exist_ok=True)
    if not frontend.exists():
        run("git", "clone", "https://github.com/Kosinkadink/Dinkster-Frontend.git", str(frontend))
    if output("git", "status", "--porcelain", cwd=frontend):
        raise RuntimeError(".run/frontend has local changes; preserve them before rerunning")
    if output("git", "rev-parse", "HEAD", cwd=frontend) != pin["commit"]:
        run("git", "fetch", "origin", pin["commit"], cwd=frontend)
        run("git", "checkout", "--detach", pin["commit"], cwd=frontend)
    stamp = ROOT / ".run/frontend-build"
    if not (frontend / "packages/app/dist/index.html").is_file() or (
        not stamp.is_file() or stamp.read_text() != pin["commit"]
    ):
        run(*pnpm, "install", "--frozen-lockfile", cwd=frontend)
        run(*pnpm, "--filter", "@dinkster/app", "build", cwd=frontend)
        stamp.write_text(pin["commit"])

    develop = str(ROOT / "scripts/develop.py")
    run(
        sys.executable,
        develop,
        "setup",
        "--port",
        str(args.port),
        "--frontend-root",
        str(frontend / "packages/app/dist"),
        "--yes",
    )
    launch = [sys.executable, develop, "start", "--port", str(args.port), "--yes"]
    url = f"http://127.0.0.1:{args.port}"
    print(f"Development editor: {url}", flush=True)
    run(*launch)
    if not args.no_browser:
        open_browser_when_ready(url)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
