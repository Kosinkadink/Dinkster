"""The no-argument local launch experience."""

from __future__ import annotations

import argparse
import threading
import time
import urllib.error
import urllib.request
import webbrowser

from . import serve
from .frontend import discover_frontend_bundle
from .setup import default_roots


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be in 1..65535")
    return port


def _open_browser_when_ready(url: str) -> None:
    health_url = f"{url}/api/health"
    for _attempt in range(600):
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dinkster",
        description="Open the local Dinkster editor",
        epilog="Advanced commands remain available as `dinkster <command> --help`.",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="print the URL without opening it"
    )
    parser.add_argument("--port", type=_port, default=3639, help="loopback port (default: 3639)")
    parser.add_argument(
        "--frontend-dev",
        metavar="URL",
        help="proxy browser application requests to a Vite development server",
    )
    args = parser.parse_args(argv)

    library_root, install_root = default_roots()
    if not library_root.is_dir() or not install_root.is_dir():
        parser.error("local installation not found; run `dinkster setup` first")

    serve_args = [
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--library-root",
        str(library_root),
        "--install-root",
        str(install_root),
        "--prepare-stale-catalogs",
        "--disable-p2p",
        "--allow-mount-changes",
    ]
    if args.frontend_dev:
        serve_args.extend(("--frontend-dev", args.frontend_dev))
    else:
        bundle = discover_frontend_bundle()
        if bundle is None:
            parser.error(
                "frontend bundle not found; build sibling Dinkster-Frontend/packages/app/dist "
                "or pass --frontend-dev URL"
            )
        serve_args.extend(("--frontend-root", str(bundle)))

    url = f"http://127.0.0.1:{args.port}"
    print(f"Dinkster is available at {url}")
    if not args.no_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()
    serve.main(serve_args)
    return 0
