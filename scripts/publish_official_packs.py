"""Prepare complete supported-base bindings before submitting official pack releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote, urlsplit

from blake3 import blake3
from dinkster_registry import LockedPack, Lockfile, artifact_digest, unpack_artifact
from dinkster_workers import load_manifest
from dinkster_workers.manifest import PackDependency
from dinkster_workers.pack_lock import PackLock
from dinkster_workers.pack_wheel import pack_manifest_path
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from dinkster.registries import NamedRegistry, RegistryPublishError
from scripts.build_official_packs import library_metadata
from scripts.build_release import read_base_manifest


def _submit_set(
    registry: NamedRegistry,
    packs: Sequence[LockedPack],
    sources: Path,
    output: Path,
    engine: dict[str, str] | None,
) -> tuple[str, dict[str, tuple[str, str]]]:
    boundary = "dinkster-" + uuid.uuid4().hex
    members = []
    files = bytearray()
    for index, pack in enumerate(packs):
        field = f"wheel{index}"
        archive = sources / f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl"
        member = {
            "name": pack.pack,
            "version": pack.version,
            "archive": field,
            "locks": json.loads((output / f"{pack.pack}-{pack.version}.json").read_bytes()),
        }
        if engine is not None:
            member["engine"] = engine
        members.append(member)
        files.extend(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
            f'filename="{archive.name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode(
                "ascii"
            )
        )
        files.extend(archive.read_bytes())
        files.extend(b"\r\n")
    data = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="manifest"\r\n\r\n'.encode("ascii")
        + json.dumps({"members": members}, allow_nan=False).encode("ascii")
        + b"\r\n"
        + files
        + f"--{boundary}--\r\n".encode("ascii")
    )
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if token := registry.token():
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        registry.endpoint.rstrip("/") + "/v1/publish-sets", data=data, headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            document = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RegistryPublishError(f"batch submission failed: HTTP {exc.code}: {detail}") from exc
    except (OSError, ValueError) as exc:
        raise RegistryPublishError(f"batch submission failed: {exc}") from exc
    (output / "set.json").write_text(json.dumps(document, indent=2) + "\n", encoding="ascii")
    try:
        set_id = document["setId"]
        if (
            not isinstance(set_id, str)
            or not set_id
            or document["status"] not in {"queued", "accepted", "rejected", "expired"}
        ):
            raise ValueError
        received = document["members"]
        if not isinstance(received, list) or len(received) != len(packs):
            raise ValueError
        expected = {pack.pack: pack for pack in packs}
        candidates = {}
        for member in received:
            pack = expected.pop(member["packName"])
            candidate_id, state = member["candidateId"], member["status"]
            if (
                member["version"] != pack.version
                or member["artifactDigest"] != pack.artifact_digest
                or not isinstance(candidate_id, str)
                or not candidate_id
                or state
                not in {
                    "queued",
                    "scanning",
                    "retrying",
                    "promoting",
                    "review",
                    "accepted",
                    "rejected",
                }
            ):
                raise ValueError
            candidates[pack.pack] = candidate_id, state
        if len({candidate_id for candidate_id, _ in candidates.values()}) != len(packs):
            raise ValueError
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise RegistryPublishError(
            "registry answered batch submission with a malformed receipt"
        ) from exc
    return set_id, candidates


def _wait_active(
    registry: NamedRegistry,
    dependency: PackDependency,
    version: str | None,
    candidate_id: str | None,
    deadline: float,
    artifact_digest: str | None = None,
) -> None:
    subject = f"{dependency.pack}@{version or dependency.version}"
    if candidate_id is not None:
        subject += f" (candidate {candidate_id})"
    last = "no active public version"
    while (remaining := deadline - time.monotonic()) > 0:
        path = f"/v1/packs/{quote(dependency.pack, safe='')}/versions"
        if version is not None:
            path += "/" + quote(version, safe="")
        else:
            path += "?limit=100"
        while path:
            try:
                request = urllib.request.Request(
                    registry.endpoint.rstrip("/") + path,
                    headers={"Cache-Control": "no-cache"},
                )
                with urllib.request.urlopen(request, timeout=min(60, remaining)) as response:
                    document = json.load(response)
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise RegistryPublishError(
                        f"{subject}: activation lookup failed: {exc}"
                    ) from exc
                document = {}
            except (OSError, ValueError) as exc:
                raise RegistryPublishError(f"{subject}: activation lookup failed: {exc}") from exc
            if version is not None:
                if (
                    document.get("state") == "active"
                    and document.get("version") == version
                    and document.get("packName") == dependency.pack
                ):
                    if (
                        artifact_digest is not None
                        and document.get("artifact", {}).get("digest") != artifact_digest
                    ):
                        raise RegistryPublishError(
                            f"{subject}: public archive digest differs from built archive"
                        )
                    return
                last = document.get("stateReason") or document.get("state") or last
                break
            for release in document.get("items", []):
                if release.get("state") == "active" and dependency.accepts(release["version"]):
                    # A listing is not the public version page required by admission.
                    _wait_active(
                        registry,
                        dependency,
                        release["version"],
                        None,
                        deadline,
                    )
                    return
            cursor = document.get("nextCursor")
            path = (
                f"/v1/packs/{quote(dependency.pack, safe='')}/versions?limit=100"
                f"&cursor={quote(cursor, safe='')}"
                if cursor
                else ""
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
        if candidate_id is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            headers = {"Cache-Control": "no-cache"}
            if token := registry.token():
                headers["Authorization"] = f"Bearer {token}"
            request = urllib.request.Request(
                registry.endpoint.rstrip("/") + "/v1/publish/" + quote(candidate_id, safe=""),
                headers=headers,
            )
            try:
                with urllib.request.urlopen(request, timeout=min(60, remaining)) as response:
                    candidate = json.load(response)
            except (OSError, ValueError) as exc:
                raise RegistryPublishError(f"{subject}: candidate lookup failed: {exc}") from exc
            last = str(candidate.get("reason") or candidate.get("status"))
            if candidate.get("status") in {"rejected", "review"}:
                findings = "; ".join(
                    f"{finding.get('code')}: {finding.get('message')}"
                    for finding in candidate.get("findings", [])
                )
                raise RegistryPublishError(
                    f"{subject}: activation refused (not retried): {last}; {findings}"
                )
        time.sleep(max(0, min(2, deadline - time.monotonic())))
    raise RegistryPublishError(f"{subject}: activation timed out (not retried): {last}")


def publish_official_packs(
    root: Path,
    sources: Path,
    base_manifests: Sequence[Path],
    pack_locks: Sequence[Path],
    output: Path,
    *,
    registry: NamedRegistry | None = None,
    preflight_registry: str | None = None,
    activation_timeout: float = 600,
    engine_manifest: Path | None = None,
) -> list[dict[str, object]]:
    """Validate the entire publication set before exposing maps or making a request.

    Without a registry endpoint, only prepare inputs. Public preflight never sends credentials.
    A queued receipt is not scanner acceptance.
    Native lock production and archive construction belong to their existing builders.
    """
    if output.exists():
        raise ValueError("publication output must be a new directory")
    engine = None
    engine_version = None
    if engine_manifest is not None:
        metadata = engine_manifest.read_bytes()
        release = json.loads(metadata)
        if (
            not isinstance(release, dict)
            or release.get("format") != "dinkster.engine/1"
            or release.get("repository") != "Kosinkadink/Dinkster"
            or not isinstance(release.get("commit"), str)
            or re.fullmatch(r"[0-9a-f]{40}", release["commit"]) is None
            or not isinstance(release.get("version"), str)
        ):
            raise ValueError("publication requires an exact Dinkster engine manifest")
        engine_version = release["version"]
        engine = {
            "commit": release["commit"],
            "manifestDigest": "sha256:" + hashlib.sha256(metadata).hexdigest(),
        }
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
    libraries = {}
    for pack in catalog.packs:
        archive = sources / f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl"
        if artifact_digest(archive.read_bytes()) != pack.artifact_digest:
            raise ValueError(f"official publication archive digest mismatch: {pack.pack}")
        with zipfile.ZipFile(archive) as wheel:
            if not any(Path(name).name == "dinkster-pack.toml" for name in wheel.namelist()):
                name, version, requirements = library_metadata(archive)
                if (name, version) != (pack.pack, pack.version) or pack.claims:
                    raise ValueError(f"official library source identity mismatch: {pack.pack}")
                libraries[pack.pack] = requirements
    node_names = {pack.pack for pack in catalog.packs} - libraries.keys()
    dependencies: dict[str, tuple[PackDependency, ...]] = {}
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
            archive = sources / f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl"
            if pack.publisher != "dinkster":
                raise ValueError(f"official publication source identity mismatch: {pack.pack}")
            if pack.pack in libraries:
                dependencies[pack.pack] = ()
                requirements = libraries[pack.pack]
            else:
                extracted = unpack_artifact(archive, work / pack.pack, pack.artifact_digest)
                manifest = load_manifest(pack_manifest_path(extracted))
                dependencies[pack.pack] = manifest.dependencies
                requirements = manifest.requires
                if (
                    manifest.publisher != "dinkster"
                    or manifest.name != pack.pack
                    or manifest.compatibility is None
                    or manifest.compatibility.version != pack.version
                ):
                    raise ValueError(f"official publication source identity mismatch: {pack.pack}")
                if engine_version is not None:
                    manifest.compatibility.require_engine(pack.pack, engine_version)
            for value in requirements:
                requirement = Requirement(value)
                name = canonicalize_name(requirement.name)
                if name in node_names:
                    raise ValueError(f"wheel.requires-pack: {pack.pack} requires node pack {name}")
                if name in libraries:
                    if requirement.url:
                        raise ValueError("official library dependencies must use version ranges")
                    dependencies[pack.pack] += (PackDependency(name, str(requirement.specifier)),)
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
    selected = {pack.pack: pack for pack in catalog.packs}
    ordered: list[LockedPack] = []
    pending = list(catalog.packs)
    while pending:
        pending_names = {pack.pack for pack in pending}
        ready = [
            pack
            for pack in pending
            if all(dep.pack not in pending_names for dep in dependencies[pack.pack])
        ]
        if not ready:
            raise ValueError("official publication has cyclic pack dependencies")
        for pack in ready:
            for dep in dependencies[pack.pack]:
                if dep.pack in selected and not dep.accepts(selected[dep.pack].version):
                    raise ValueError(f"{pack.pack}: selected {dep.pack} excludes {dep.version}")
        ordered.extend(ready)
        pending = [pack for pack in pending if pack not in ready]
    public = set()
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
                != (sources / f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl")
                .stat()
                .st_size
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
            public.add(pack.pack)
    result: list[dict[str, object]] = []
    for pack in ordered if registry is not None else catalog.packs:
        locks = output / f"{pack.pack}-{pack.version}.json"
        row: dict[str, object] = {
            "pack": pack.pack,
            "version": pack.version,
            "artifactDigest": pack.artifact_digest,
            "locks": str(locks),
        }
        result.append(row)
        if registry is not None:
            row["state"] = "published" if pack.pack in public else "submitting"
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    candidates: dict[str, str] = {}
    if registry is not None:
        missing = [pack for pack in ordered if pack.pack not in public]
        missing_names = {pack.pack for pack in missing}
        for pack in missing:
            for dep in dependencies[pack.pack]:
                if dep.pack in missing_names:
                    continue
                _wait_active(
                    registry,
                    dep,
                    selected[dep.pack].version if dep.pack in selected else None,
                    None,
                    time.monotonic() + activation_timeout,
                    selected[dep.pack].artifact_digest if dep.pack in selected else None,
                )
        if missing:
            try:
                set_id, submitted = _submit_set(registry, missing, sources, output, engine)
            except RegistryPublishError as exc:
                raise RegistryPublishError(
                    "publication refused (not retried); "
                    f"if archive or install locks changed, bump the pack version: {exc}"
                ) from exc
            for row in result:
                name = str(row["pack"])
                if name in submitted:
                    candidate_id, state = submitted[name]
                    row.update(state=state, candidateId=candidate_id, setId=set_id)
                    candidates[name] = candidate_id
            (output / "result.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
        deadline = time.monotonic() + activation_timeout
        failures = []
        for row in result:
            pack = selected[str(row["pack"])]
            if pack.pack not in candidates:
                continue
            try:
                _wait_active(
                    registry,
                    PackDependency(pack.pack, "==" + pack.version),
                    pack.version,
                    candidates[pack.pack],
                    deadline,
                    pack.artifact_digest,
                )
            except RegistryPublishError as exc:
                row["reason"] = str(exc)
                failures.append(str(exc))
            else:
                row["state"] = "published"
            (output / "result.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
        if failures:
            raise RegistryPublishError("Publication incomplete:\n" + "\n".join(failures))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, action="append", type=Path)
    parser.add_argument("--pack-lock", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engine-manifest", type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--check-published", action="store_true")
    parser.add_argument("--registry", default="https://registry.dinkster.org")
    args = parser.parse_args()
    try:
        result = publish_official_packs(
            Path(__file__).resolve().parents[1],
            args.sources.resolve(),
            args.base_manifest,
            args.pack_lock,
            args.output.resolve(),
            registry=NamedRegistry("", args.registry, token_env="DINKSTER_REGISTRY_TOKEN")
            if args.publish
            else None,
            preflight_registry=args.registry if args.check_published else None,
            engine_manifest=args.engine_manifest,
        )
    except RegistryPublishError as exc:
        if args.output.exists():
            (args.output / "failure.txt").write_text(str(exc) + "\n", encoding="utf-8")
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result))
