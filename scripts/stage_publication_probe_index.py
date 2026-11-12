"""Acquire verified Linux scanner dependencies before the isolated publication job."""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import shutil
import subprocess
import tempfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import Request, url2pathname, urlopen
from zipfile import ZipFile

from dinkster_registry import Lockfile, artifact_digest, unpack_artifact
from dinkster_supervisor.environment_artifacts import validate_manifest
from dinkster_supervisor.environment_downloads import fetch_artifact, read_metadata
from dinkster_workers import load_manifest
from dinkster_workers.pack_wheel import pack_manifest_path

from scripts.build_official_packs import library_metadata
from scripts.build_release import sha256


def stage_index(
    sources: Path, output: Path, *, engine: Path | None = None, bases: Path | None = None
) -> dict:
    """Resolve CPU probe requirements independently of the native install locks."""
    output.mkdir(parents=True, exist_ok=True)
    catalog = Lockfile.from_record_json((sources / "official-packs.lock").read_text())
    requirements: set[str] = set()
    if engine is not None:
        apis = list(engine.glob("dinkster_api-*.whl"))
        if len(apis) != 1:
            raise ValueError("probe index requires exactly one selected API wheel")
        with ZipFile(apis[0]) as wheel:
            metadata = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError("selected API wheel requires one distribution")
            fields = email.parser.BytesParser().parsebytes(wheel.read(metadata[0]))
            requirements.update(fields.get_all("Requires-Dist", []))
    with tempfile.TemporaryDirectory(prefix="publication-probe-") as directory:
        work = Path(directory)
        for pack in catalog.packs:
            archive = sources / f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl"
            if artifact_digest(archive.read_bytes()) != pack.artifact_digest:
                raise ValueError(f"probe source digest mismatch: {pack.pack}")
            with ZipFile(archive) as wheel:
                is_pack = any(Path(name).name == "dinkster-pack.toml" for name in wheel.namelist())
            if is_pack:
                extracted = unpack_artifact(archive, work / pack.pack, pack.artifact_digest)
                requirements.update(
                    load_manifest(pack_manifest_path(extracted)).requires_for("cpu")
                )
            else:
                requirements.update(library_metadata(archive)[2])
        inputs = output / "requirements.in"
        inputs.write_text("\n".join(sorted(requirements)) + "\n", encoding="ascii")
        lock = output / "pylock.toml"
        subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                "--python-version",
                "3.12",
                "--python-platform",
                "x86_64-manylinux_2_28",
                "--only-binary",
                ":all:",
                "--torch-backend",
                "cpu",
                "--find-links",
                str(sources.resolve()),
                "--format",
                "pylock.toml",
                "--output-file",
                str(lock),
                str(inputs),
            ],
            check=True,
            timeout=180,
        )
    resolution = tomllib.loads(lock.read_text())

    def acquire(item: tuple[str, dict]) -> dict:
        package, wheel = item
        parts = urlsplit(wheel["url"])
        name = unquote(parts.path.rsplit("/", 1)[-1])
        destination = output / name
        if parts.scheme == "file":
            source = Path(url2pathname(parts.path)).resolve()
            if source.parent != sources.resolve():
                raise ValueError("local probe wheel must be a selected source")
            digest, size = sha256(source), source.stat().st_size
            shutil.copyfile(source, destination)
        else:
            if parts.scheme != "https":
                raise ValueError("remote probe wheels require HTTPS")
            digest = wheel["hashes"]["sha256"]
            if "size" in wheel:
                fetch_artifact(
                    (wheel["url"].rsplit("/", 1)[0],), name, destination, digest, wheel["size"]
                )
            elif not destination.exists() or sha256(destination) != digest:
                # Some CPU wheel indexes provide hashes without file sizes.
                request = Request(wheel["url"], headers={"User-Agent": "dinkster-env"})
                with urlopen(request, timeout=120) as response, destination.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            if sha256(destination) != digest:
                raise ValueError(f"probe dependency digest mismatch: {name}")
            size = destination.stat().st_size
        return {"package": package, "filename": name, "sha256": digest, "size": size}

    items = [(p["name"], w) for p in resolution.get("packages", []) for w in p["wheels"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(acquire, items))
    if bases is not None:
        if engine is None:
            raise ValueError("runtime staging requires a candidate engine")
        stage_runtime(bases, engine, output / "runtime")
    pages: dict[str, list[str]] = {}
    for row in rows:
        filename = escape(row["filename"], quote=True)
        pages.setdefault(row["package"], []).append(
            f'<a href="../../{filename}#sha256={row["sha256"]}">{filename}</a>\n'
        )
    for package, links in pages.items():
        directory = output / "simple" / package
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index.html").write_text("".join(links), encoding="ascii")
    receipt = {"distributions": len(pages), "wheels": rows}
    (output / "inventory.json").write_text(json.dumps(receipt, indent=2), encoding="ascii")
    return receipt


def stage_runtime(bases: Path, engine: Path, output: Path) -> None:
    """Keep published bytes intact beside an explicitly unpublished exact-head engine."""
    origin = "https://releases.dinkster.org"
    for source in sorted(bases.glob("*/manifest.json")):
        base = json.loads(source.read_bytes())
        variant, identity = base["variant"], base["id"]
        index_path = f"base/{variant}/index.json"
        index_bytes = read_metadata(origin, index_path)
        index = json.loads(index_bytes)
        matches = [row for row in index if row.get("id") == identity]
        if len(matches) != 1:
            raise ValueError(f"runtime base index must identify one published {variant} base")
        path = matches[0]["path"]
        destination = output / path
        if not destination.resolve().is_relative_to(output.resolve()):
            raise ValueError("runtime base path escapes its mirror")
        payload = read_metadata(origin, path + "/manifest.json")
        published = validate_manifest(payload, identity, variant)
        if published != base:
            raise ValueError("published runtime base differs from the selected base")
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_bytes(payload)
        (output / index_path).write_bytes(index_bytes)
        if variant == "linux-cpu":
            for row in published["archive"]["parts"]:
                fetch_artifact(
                    (origin + "/" + path,),
                    row["filename"],
                    destination / row["filename"],
                    row["sha256"],
                    row["size"],
                )
    payload = (engine / "release-manifest.json").read_bytes()
    identity = hashlib.sha256(payload).hexdigest()
    manifest = validate_manifest(payload, "sha256:" + identity, None)
    path = "engine/" + identity
    destination = output / path
    destination.mkdir(parents=True)
    (destination / "release-manifest.json").write_bytes(payload)
    for row in manifest["artifacts"]:
        fetch_artifact(
            (str(engine),), row["name"], destination / row["name"], row["sha256"], row["size"]
        )
    (output / "engine/index.json").write_text(
        json.dumps(
            [
                {
                    "commit": manifest["commit"],
                    "sha256": identity,
                    "version": manifest["version"],
                    "path": path,
                    "unpublishedCandidate": True,
                }
            ]
        ),
        encoding="ascii",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--bases", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            stage_index(
                args.sources.resolve(),
                args.output.resolve(),
                engine=args.engine.resolve() if args.engine else None,
                bases=args.bases.resolve() if args.bases else None,
            )
        )
    )
