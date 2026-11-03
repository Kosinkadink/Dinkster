"""Prepare complete supported-base bindings before submitting official pack releases."""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote, urlsplit

from blake3 import blake3
from dinkster_registry import Lockfile, unpack_artifact
from dinkster_workers import load_manifest
from dinkster_workers.pack_lock import PackLock
from packaging.specifiers import SpecifierSet

from dinkster.registries import NamedRegistry, RegistryPublishError, publish_release
from scripts.build_release import read_base_manifest


def publish_official_packs(
    root: Path,
    sources: Path,
    base_manifests: Sequence[Path],
    pack_locks: Sequence[Path],
    output: Path,
    *,
    registry: NamedRegistry | None = None,
    preflight_registry: str | None = None,
) -> list[dict[str, object]]:
    """Validate the entire publication set before exposing maps or making a request.

    Without a registry endpoint, only prepare inputs. Public preflight never sends credentials.
    A queued receipt is not scanner acceptance.
    Native lock production and archive construction belong to their existing builders.
    """
    if output.exists():
        raise ValueError("publication output must be a new directory")
    supported = json.loads((root / "scripts/base_bundles.json").read_bytes())
    if not supported:
        raise ValueError("official publication requires supported bases")
    bases = {}
    for path in base_manifests:
        base = read_base_manifest(path)
        variant = base["variant"]
        if variant in bases or base["id"] != "sha256:" + supported.get(variant, ""):
            raise ValueError(f"unexpected or duplicate supported base: {variant}")
        bases[variant] = base
    if bases.keys() != supported.keys():
        raise ValueError("official publication requires every supported base")
    catalog = Lockfile.from_record_json((sources / "official-packs.lock").read_text())
    if not catalog.packs:
        raise ValueError("official publication requires selected packs")
    required = {
        f"{pack.pack}-{pack.version}-{variant}.pack-lock.json": (pack.pack, variant)
        for pack in catalog.packs
        for variant in bases
    }
    supplied = {path.name: path for path in pack_locks}
    if len(supplied) != len(pack_locks) or supplied.keys() != required.keys():
        raise ValueError("official publication dependency lock coverage mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="official-publication-", dir=output.parent
    ) as directory:
        work = Path(directory)
        staged = work / "maps"
        staged.mkdir()
        for pack in catalog.packs:
            archive = sources / f"{pack.pack}-{pack.version}.zip"
            extracted = unpack_artifact(archive, work / pack.pack, pack.artifact_digest)
            manifest = load_manifest(extracted / "dinkster-pack.toml")
            if (
                pack.publisher != "dinkster"
                or manifest.publisher != "dinkster"
                or manifest.name != pack.pack
                or manifest.compatibility is None
                or manifest.compatibility.version != pack.version
            ):
                raise ValueError(f"official publication source identity mismatch: {pack.pack}")
            bindings = {}
            for variant, base in sorted(bases.items()):
                name = f"{pack.pack}-{pack.version}-{variant}.pack-lock.json"
                path = supplied[name]
                lock = PackLock.read(path)
                if (lock.pack, lock.variant) != required[name]:
                    raise ValueError(
                        f"official publication dependency lock identity mismatch: {name}"
                    )
                if not SpecifierSet(lock.requires_python).contains(
                    base["python"], prereleases=True
                ):
                    raise ValueError(f"dependency lock excludes supported base Python: {name}")
                urls = [*lock.indexes, *(w.url for p in lock.packages for w in p.wheels)]
                for url in urls:
                    origin = urlsplit(url)
                    if (
                        origin.scheme not in {"https", "http"}
                        or not origin.hostname
                        or origin.username is not None
                        or origin.password is not None
                    ):
                        raise ValueError(f"dependency lock contains a nonportable URL: {name}")
                if any(
                    type(w.size) is not int or w.size < 1 for p in lock.packages for w in p.wheels
                ):
                    raise ValueError(f"dependency lock contains an invalid wheel size: {name}")
                bindings[variant] = {
                    "baseId": base["id"],
                    "artifactDigest": pack.artifact_digest,
                    "lock": json.loads(path.read_bytes()),
                }
            (staged / f"{pack.pack}-{pack.version}.json").write_text(
                json.dumps(bindings, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="ascii",
            )
        staged.rename(output)
    endpoint = registry.endpoint if registry is not None else preflight_registry
    if endpoint is not None:
        for pack in catalog.packs:
            url = (
                endpoint.rstrip("/")
                + f"/v1/packs/{quote(pack.pack, safe='')}/versions/{quote(pack.version, safe='')}"
            )
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    published = json.load(response)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue
                raise RegistryPublishError(
                    f"{pack.pack}@{pack.version}: registry lookup failed: {exc}"
                ) from exc
            except (OSError, ValueError) as exc:
                raise RegistryPublishError(
                    f"{pack.pack}@{pack.version}: registry lookup failed: {exc}"
                ) from exc
            bindings = json.loads((output / f"{pack.pack}-{pack.version}.json").read_bytes())
            expected = {}
            for variant, binding in bindings.items():
                payload = json.dumps(
                    binding["lock"], sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("ascii")
                expected[variant] = {
                    "baseId": binding["baseId"],
                    "artifactDigest": binding["artifactDigest"],
                    "digest": "blake3:" + blake3(payload).hexdigest(),
                    "size": len(payload),
                }
            actual = published.get("installLocks", {})
            if (
                published.get("packName") != pack.pack
                or published.get("version") != pack.version
                or published.get("publishedBy", {}).get("principalId") != pack.publisher
                or published.get("artifact", {}).get("digest") != pack.artifact_digest
                or published.get("artifact", {}).get("size")
                != (sources / f"{pack.pack}-{pack.version}.zip").stat().st_size
                or actual.keys() != expected.keys()
                or any(
                    {key: actual[variant].get(key) for key in binding} != binding
                    for variant, binding in expected.items()
                )
            ):
                raise RegistryPublishError(
                    f"{pack.pack}@{pack.version}: published archive or install locks differ; "
                    "bump the pack version"
                )
    result: list[dict[str, object]] = []
    for pack in catalog.packs:
        locks = output / f"{pack.pack}-{pack.version}.json"
        row: dict[str, object] = {
            "pack": pack.pack,
            "version": pack.version,
            "artifactDigest": pack.artifact_digest,
            "locks": str(locks),
        }
        if registry is not None:
            try:
                verdict = publish_release(
                    registry,
                    sources / f"{pack.pack}-{pack.version}.zip",
                    pack.pack,
                    pack.version,
                    locks=locks,
                )
            except RegistryPublishError as exc:
                raise RegistryPublishError(
                    f"{pack.pack}@{pack.version}: publication refused (not retried); "
                    f"if archive or install locks changed, bump the pack version: {exc}"
                ) from exc
            row.update(state=verdict.state, candidateId=verdict.candidate_id)
        result.append(row)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, action="append", type=Path)
    parser.add_argument("--pack-lock", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--check-published", action="store_true")
    parser.add_argument("--registry", default="https://registry.dinkster.org")
    args = parser.parse_args()
    print(
        json.dumps(
            publish_official_packs(
                Path(__file__).resolve().parents[1],
                args.sources.resolve(),
                args.base_manifest,
                args.pack_lock,
                args.output.resolve(),
                registry=NamedRegistry("", args.registry, token_env="DINKSTER_REGISTRY_TOKEN")
                if args.publish
                else None,
                preflight_registry=args.registry if args.check_published else None,
            )
        )
    )
