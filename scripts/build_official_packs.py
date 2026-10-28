"""Build official registry source artifacts and their existing artifact lock."""

from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from dinkster_registry import LockedPack, Lockfile, build_artifact
from dinkster_registry.model import validate_version
from dinkster_schema import canonical_claim
from dinkster_workers import load_manifest


def build(root: Path, output: Path, *, version: str) -> tuple[Path, ...]:
    """Build only committed pack sources, without importing node implementations."""
    problem = validate_version(version)
    if problem is not None:
        raise ValueError(f"pack version {version!r} {problem}")
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked = subprocess.check_output(["git", "-C", str(root), "archive", "--format=tar", commit])
    output.mkdir(parents=True, exist_ok=True)
    payloads: list[Path] = []
    locked: list[LockedPack] = []
    with tempfile.TemporaryDirectory(prefix="dinkster-official-packs-") as directory:
        staged = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(tracked)) as archive:
            archive.extractall(staged, filter="data")
        packages = staged / "packages"
        roots = [
            packages / f"dinkster-nodes-{family}"
            for family in ("comfy", "foundation", "image", "media-io", "remote", "generation")
        ]
        vision = packages / "dinkster-nodes-vision"
        for manifest in sorted(vision.glob("*_pack/dinkster-pack.toml")):
            pack_root = manifest.parent
            entry = load_manifest(manifest).nodes_entry.partition(":")[0]
            namespace, _, module = entry.partition(".")
            destination = pack_root / namespace / module.replace(".", "/")
            shutil.copytree(vision / "src" / namespace / module.replace(".", "/"), destination)
            roots.append(pack_root)
        for pack_root in roots:
            manifest_path = pack_root / "dinkster-pack.toml"
            source = pack_root / "src"
            if source.is_dir():
                # Pack workers import from the artifact root, not workspace src/.
                for module in source.iterdir():
                    shutil.move(str(module), pack_root / module.name)
                source.rmdir()
                manifest_path.write_text(
                    re.sub(
                        r'(?m)^(file\s*=\s*")src/',
                        r"\1",
                        manifest_path.read_text(encoding="utf-8"),
                    ),
                    encoding="utf-8",
                )
            manifest = load_manifest(manifest_path)
            if manifest.publisher != "dinkster":
                raise ValueError(f"official pack {manifest.name!r} must declare publisher dinkster")
            payload = output / f"{manifest.name}-{version}.zip"
            digest = build_artifact(pack_root, payload)
            locked.append(
                LockedPack(
                    pack=manifest.name,
                    version=version,
                    artifact_digest=digest,
                    publisher=manifest.publisher,
                    claims=tuple(canonical_claim(claim) for claim in manifest.namespaces),
                    source=f"git:Kosinkadink/Dinkster@{commit}",
                )
            )
            payloads.append(payload)
    lock = output / "official-packs.lock"
    lock.write_text(Lockfile.of(locked).record_json() + "\n", encoding="utf-8")
    return (*payloads, lock)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Normalized public PEP 440 pack version")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for payload in build(
        Path(__file__).resolve().parents[1], args.output.resolve(), version=args.version
    ):
        print(payload)
