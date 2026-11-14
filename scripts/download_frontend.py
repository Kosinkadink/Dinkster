"""Download and verify the versioned frontend wheel without a Node toolchain."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

REPOSITORY = "Kosinkadink/Dinkster-Frontend"


def download(sources: dict, output: Path) -> None:
    url = urllib.parse.urlsplit(sources.get("url", ""))
    if (
        url.scheme != "https"
        or url.hostname != "files.pythonhosted.org"
        or url.username
        or url.password
        or url.query
        or url.fragment
        or PurePosixPath(url.path).name
        != f"dinkster_frontend-{sources.get('version')}-py3-none-any.whl"
    ):
        raise ValueError("frontend pin requires its permanent PyPI wheel URL")
    with urllib.request.urlopen(url.geturl(), timeout=60) as response:
        payload = response.read()
    verify_wheel(payload, sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(f"Verified frontend {sources['version']} with sha256:{sources['sha256']}")


def verify_wheel(payload: bytes, sources: dict) -> dict:
    if (
        sources.get("repository") != REPOSITORY
        or not isinstance(sources.get("version"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(sources.get("sha256"))) is None
        or hashlib.sha256(payload).hexdigest() != sources["sha256"]
        or type(sources.get("size")) is not int
        or len(payload) != sources["size"]
    ):
        raise ValueError("frontend wheel identity, hash or size does not match the pin")
    with zipfile.ZipFile(io.BytesIO(payload)) as wheel:
        manifest = json.loads(wheel.read("dinkster_frontend/manifest.json"))
    api = manifest.get("engineApi", {})
    if (
        manifest.get("format") != "dinkster.frontend/1"
        or manifest.get("repository") != REPOSITORY
        or manifest.get("version") != sources["version"]
        or not isinstance(api, dict)
        or type(api.get("min")) is not int
        or type(api.get("max")) is not int
        or not 1 <= api["min"] <= api["max"]
        or api != sources.get("engineApi")
    ):
        raise ValueError("frontend wheel metadata does not match the selected release")
    return manifest


def verify_and_extract(payload: bytes, sources: dict, output: Path) -> dict:
    manifest = verify_wheel(payload, sources)
    prefix = "dinkster_frontend/static/"
    with zipfile.ZipFile(io.BytesIO(payload)) as wheel:
        entries = [entry for entry in wheel.infolist() if entry.filename.startswith(prefix)]
        names = [entry.filename.removeprefix(prefix) for entry in entries]
        if "index.html" not in names or len(names) != len(set(names)):
            raise ValueError("frontend wheel must contain a unique index.html")
        for name, entry in zip(names, entries, strict=True):
            path = PurePosixPath(name)
            if (
                not name
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in entry.orig_filename
                or ":" in entry.filename
                or entry.is_dir()
                or entry.external_attr >> 16 != 0o100644
            ):
                raise ValueError(f"unsafe frontend wheel entry: {name}")
        output.mkdir(parents=True, exist_ok=False)
        for name, entry in zip(names, entries, strict=True):
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(wheel.read(entry))
    (output.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("scripts/release_sources.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        download(json.loads(args.sources.read_text(encoding="utf-8")), args.output)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
