"""Build the versioned wheel release contract from locked sources."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

from scripts.download_frontend import verify_and_extract

TAG_PATTERN = re.compile(r"v(?P<version>[0-9]+\.[0-9]+\.[0-9]+)")
TARGET_PYTHON = "3.14.10"


def project_metadata(path: Path) -> tuple[str, str]:
    project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    return project["name"], project["version"]


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def workspace_projects(root: Path) -> dict[str, Path]:
    projects = [root / "pyproject.toml", *sorted((root / "packages").glob("*/pyproject.toml"))]
    result: dict[str, Path] = {}
    for path in projects:
        name, _ = project_metadata(path)
        if name in result:
            raise ValueError(f"duplicate workspace project name: {name}")
        result[name] = path
    return result


def release_version(root: Path, tag: str) -> str:
    match = TAG_PATTERN.fullmatch(tag)
    if match is None:
        raise ValueError("release tag must match vX.Y.Z")
    version = match["version"]
    mismatches = [
        f"{name}={project_metadata(path)[1]}"
        for name, path in workspace_projects(root).items()
        if project_metadata(path)[1] != version
    ]
    if mismatches:
        raise ValueError(
            f"release tag {tag} does not match workspace metadata: {', '.join(mismatches)}"
        )
    return version


def development_version(root: Path) -> str:
    _, version = project_metadata(root / "pyproject.toml")
    major, minor, patch = (int(part) for part in version.split("."))
    timestamp = subprocess.check_output(
        ["git", "-C", str(root), "show", "-s", "--format=%ct", "HEAD"], text=True
    ).strip()
    return f"{major}.{minor}.{patch + 1}.dev{timestamp}+g{git_head(root)}"


@contextmanager
def versioned_sources(root: Path, version: str) -> Iterator[Path]:
    """Build exact workspace dependencies without changing the checkout or lock."""
    with tempfile.TemporaryDirectory(prefix="dinkster-wheel-source-") as directory:
        staged = Path(directory)
        archive = subprocess.check_output(
            ["git", "-C", str(root), "archive", "--format=tar", "HEAD"]
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(staged, filter="data")
        projects = workspace_projects(staged)
        for path in projects.values():
            text = path.read_text(encoding="utf-8")
            text = re.sub(r'(?m)^version = "[^"]+"$', f'version = "{version}"', text, count=1)
            project = tomllib.loads(text)["project"]
            dependencies = list(project.get("dependencies", []))
            for optional in project.get("optional-dependencies", {}).values():
                dependencies.extend(optional)
            for dependency in dependencies:
                match = re.fullmatch(r"([A-Za-z0-9_-]+)(\[[^]]+\])?[^;]*(;.*)?", dependency)
                if match is not None and match[1] in projects:
                    pinned = f"{match[1]}{match[2] or ''}=={version}{match[3] or ''}"
                    text = text.replace(json.dumps(dependency), json.dumps(pinned))
            if path == staged / "pyproject.toml":
                text = text.replace(
                    "dependencies = [", f'dependencies = [\n    "dinkster-frontend=={version}",', 1
                )
            path.write_text(text, encoding="utf-8")
        yield staged


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def wheel_metadata(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as wheel:
        metadata_paths = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError(f"wheel has no unique METADATA file: {path.name}")
        fields = {}
        for line in wheel.read(metadata_paths[0]).decode("utf-8").splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                fields.setdefault(key, value)
        return fields["Name"].lower(), fields["Version"]


def build_frontend_wheel(
    frontend_artifact: Path,
    version: str,
    sources: dict,
    output: Path,
    uv: str,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="dinkster-frontend-wheel-") as directory:
        project = Path(directory)
        module = project / "src/dinkster_frontend"
        module.mkdir(parents=True)
        verify_and_extract(frontend_artifact.read_bytes(), sources, module / "static")
        (module / "__init__.py").write_text(
            '"""Installed Dinkster frontend bundle."""\n\n'
            "from importlib.resources import files\n\n"
            "def bundle_path():\n"
            '    """Return the installed frontend resource directory."""\n'
            '    return files("dinkster_frontend").joinpath("static")\n',
            encoding="utf-8",
        )
        (project / "pyproject.toml").write_text(
            "[project]\n"
            'name = "dinkster-frontend"\n'
            f'version = "{version}"\n'
            'description = "Versioned Dinkster web application bundle"\n'
            'requires-python = ">=3.12"\n\n'
            "[build-system]\n"
            'requires = ["hatchling"]\n'
            'build-backend = "hatchling.build"\n\n'
            "[tool.hatch.build.targets.wheel]\n"
            'packages = ["src/dinkster_frontend"]\n',
            encoding="utf-8",
        )
        subprocess.run([uv, "build", "--wheel", "--out-dir", str(output), str(project)], check=True)
    wheels = sorted(output.glob(f"dinkster_frontend-{version}-*.whl"))
    if len(wheels) != 1:
        raise ValueError("frontend build did not produce exactly one wheel")
    with zipfile.ZipFile(frontend_artifact) as artifact, zipfile.ZipFile(wheels[0]) as wheel:
        manifest = json.loads(artifact.read("manifest.json"))
        with zipfile.ZipFile(io.BytesIO(artifact.read("bundle.zip"))) as bundle:
            expected = {f"dinkster_frontend/static/{name}" for name in bundle.namelist()}
            actual = {
                name for name in wheel.namelist() if name.startswith("dinkster_frontend/static/")
            }
            if actual != expected or any(
                wheel.read(f"dinkster_frontend/static/{name}") != bundle.read(name)
                for name in bundle.namelist()
            ):
                raise ValueError("frontend wheel files do not match the pinned bundle")
    print(f"Packaged frontend bundle sha256:{manifest['sha256']} from {manifest['commit']}")
    return wheels[0]


def build_source_archive(root: Path, version: str, output: Path) -> Path:
    archive = output / f"dinkster-source-{version}.zip"
    with tempfile.TemporaryDirectory(prefix="dinkster-source-") as directory:
        original = Path(directory) / "tracked.zip"
        subprocess.run(
            ["git", "-C", str(root), "archive", "--format=zip", f"--output={original}", "HEAD"],
            check=True,
        )
        with (
            zipfile.ZipFile(original) as source,
            zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target,
        ):
            for name in source.namelist():
                parts = tuple(part.lower() for part in Path(name).parts)
                if any(part in {"tests", "tools", "benchmarks"} for part in parts):
                    continue
                if any("benchmark" in part for part in parts):
                    continue
                target.writestr(name, source.read(name))
    return archive


def write_constraints(root: Path, output: Path, wheels: list[Path], uv: str) -> Path:
    constraints = output / "constraints.txt"
    with tempfile.NamedTemporaryFile(prefix="dinkster-external-", suffix=".txt") as temporary:
        subprocess.run(
            [
                uv,
                "export",
                "--locked",
                "--all-packages",
                "--no-dev",
                "--group",
                "engine-base",
                "--no-emit-project",
                "--no-emit-workspace",
                "--no-annotate",
                "--no-header",
                "--output-file",
                temporary.name,
            ],
            cwd=root,
            check=True,
        )
        external = Path(temporary.name).read_text(encoding="utf-8").rstrip()
    local = []
    for wheel in sorted(wheels):
        name, version = wheel_metadata(wheel)
        local.append(f"{name}=={version} --hash=sha256:{sha256(wheel)}")
    constraints.write_text(external + "\n" + "\n".join(local) + "\n", encoding="utf-8")
    return constraints


def download_external_wheels(root: Path, output: Path, uv: str) -> list[Path]:
    """Fetch the locked CPython 3.14 Windows/Linux closure without resolving."""
    from packaging.requirements import Requirement
    from packaging.tags import compatible_tags, cpython_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename

    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dinkster-wheel-export-") as directory:
        requirements = write_constraints(root, Path(directory), [], uv).read_text()
    selected: dict[str, tuple[str, int, str]] = {}
    for windows in (True, False):
        platforms = (
            ["win_amd64"]
            if windows
            else [f"manylinux_2_{n}_x86_64" for n in range(28, 16, -1)]
            + ["manylinux2014_x86_64", "linux_x86_64"]
        )
        tags = list(cpython_tags((3, 14), platforms=platforms)) + list(
            compatible_tags((3, 14), interpreter="cp314", platforms=platforms)
        )
        priority = {}
        for index, tag in enumerate(tags):
            priority.setdefault(tag, index)
        environment = {
            "implementation_name": "cpython",
            "implementation_version": TARGET_PYTHON,
            "os_name": "nt" if windows else "posix",
            "platform_machine": "AMD64" if windows else "x86_64",
            "platform_release": "",
            "platform_system": "Windows" if windows else "Linux",
            "platform_version": "",
            "python_full_version": TARGET_PYTHON,
            "platform_python_implementation": "CPython",
            "python_version": "3.14",
            "sys_platform": "win32" if windows else "linux",
            "extra": "",
        }
        for line in requirements.replace("\\\n", "").splitlines():
            if not line.strip():
                continue
            requirement = Requirement(line.split(" --hash=", 1)[0].strip())
            if requirement.url is not None or not re.fullmatch(
                r"==[^*,]+", str(requirement.specifier)
            ):
                raise ValueError(
                    f"external requirement must be an exact registry pin: {requirement.name}"
                )
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            hashes = set(re.findall(r"--hash=sha256:([0-9a-f]{64})", line))
            candidates = []
            for package in lock["package"]:
                if canonicalize_name(package["name"]) != canonicalize_name(
                    requirement.name
                ) or not requirement.specifier.contains(package["version"], prereleases=True):
                    continue
                for wheel in package.get("wheels", []):
                    name = unquote(urlsplit(wheel["url"]).path).rsplit("/", 1)[-1]
                    wheel_name, wheel_version, _, wheel_tags = parse_wheel_filename(name)
                    if wheel_name != canonicalize_name(
                        requirement.name
                    ) or not requirement.specifier.contains(wheel_version, prereleases=True):
                        raise ValueError(f"wheel identity disagrees with locked package: {name}")
                    ranks = [priority[tag] for tag in wheel_tags if tag in priority]
                    if ranks and wheel.get("hash", "").removeprefix("sha256:") in hashes:
                        candidates.append((min(ranks), name, wheel))
            if not candidates:
                raise ValueError(f"no locked cp314 wheel for {requirement.name} on {platforms[0]}")
            _, name, wheel = min(candidates, key=lambda item: (item[0], item[1]))
            identity = (wheel["url"], wheel["size"], wheel["hash"].removeprefix("sha256:"))
            if name in selected and selected[name][1:] != identity[1:]:
                raise ValueError(f"conflicting locked artifact: {name}")
            selected[name] = identity
    paths = []
    for name, (url, size, digest) in sorted(selected.items()):
        target = output / name
        if target.exists():
            raise ValueError(f"external download output already exists: {name}")
        try:
            with urlopen(url, timeout=60) as response, target.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if target.stat().st_size != size or sha256(target) != digest:
                raise ValueError(f"downloaded wheel does not match lock: {name}")
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        paths.append(target)
    return paths


def read_base_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "dinkster.base/1":
        raise ValueError(f"unsupported base manifest format: {path}")
    identity_fields = (
        "format",
        "variant",
        "platform",
        "python",
        "interpreter",
        "wheels",
        "replaces",
        "workerProtocol",
        "nvidia",
    )
    payload = {key: manifest[key] for key in identity_fields}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    if manifest.get("id") != "sha256:" + hashlib.sha256(canonical).hexdigest():
        raise ValueError(f"base manifest id does not match its payload: {path}")
    return manifest


def stage_external_wheels(
    root: Path,
    output: Path,
    candidates: list[Path],
    base_manifests: list[Path],
    target_variants: set[str],
) -> list[Path]:
    """Omit only locked artifacts proven present in every selected target base."""
    bases: dict[str, set[tuple[str, int, str]]] = {}
    for path in base_manifests:
        manifest = read_base_manifest(path)
        variant = manifest.get("variant")
        if variant not in target_variants or variant in bases:
            raise ValueError(f"unexpected or duplicate base variant {variant!r}: {path}")
        bases[variant] = (
            {(wheel["filename"], wheel["size"], wheel["sha256"]) for wheel in manifest["wheels"]}
            if manifest.get("parts")
            else set()
        )

    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    locked = {
        (
            unquote(urlsplit(wheel["url"]).path).rsplit("/", 1)[-1],
            wheel["size"],
            wheel["hash"].removeprefix("sha256:"),
        )
        for package in lock["package"]
        for wheel in package.get("wheels", [])
        if "size" in wheel and wheel["hash"].startswith("sha256:")
    }
    staged = []
    for candidate in candidates:
        identity = (candidate.name, candidate.stat().st_size, sha256(candidate))
        if candidate.suffix != ".whl" or identity not in locked:
            raise ValueError(f"external wheel does not match a locked artifact: {candidate}")
        if (
            bases
            and bases.keys() == target_variants
            and all(identity in members for members in bases.values())
        ):
            continue
        target = output / candidate.name
        if target.exists():
            raise ValueError(f"duplicate release artifact: {candidate.name}")
        shutil.copyfile(candidate, target)
        staged.append(target)
    return staged


def build(
    root: Path,
    output: Path,
    tag: str | None,
    frontend_artifact: Path,
    *,
    uv: str = "uv",
    external_wheels: list[Path] | None = None,
    base_manifests: list[Path] | None = None,
    target_variants: set[str] | None = None,
    payload_artifacts: list[Path] | None = None,
    download_external: bool = False,
) -> dict[str, object]:
    version = release_version(root, tag) if tag else development_version(root)
    sources = json.loads((root / "scripts/release_sources.json").read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise ValueError("release output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    with versioned_sources(root, version) as staged:
        subprocess.run(
            [uv, "build", "--all-packages", "--wheel", "--out-dir", str(output)],
            cwd=staged,
            check=True,
        )
    build_frontend_wheel(frontend_artifact, version, sources, output, uv)
    wheels = sorted(output.glob("*.whl"))
    expected = set(workspace_projects(root)) | {"dinkster-frontend"}
    actual = {wheel_metadata(wheel)[0] for wheel in wheels}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"wheel set mismatch: missing={missing}, extra={extra}")
    constraints = write_constraints(root, output, wheels, uv)
    with tempfile.TemporaryDirectory(prefix="dinkster-external-wheels-") as directory:
        candidates = list(external_wheels or [])
        if download_external:
            candidates.extend(download_external_wheels(root, Path(directory), uv))
        external = stage_external_wheels(
            root, output, candidates, base_manifests or [], target_variants or set()
        )
    source = build_source_archive(root, version, output)
    artifacts = sorted([*wheels, *external, constraints, source], key=lambda path: path.name)
    reserved = {path.name.casefold() for path in artifacts} | {
        "release-manifest.json",
        "sha256sums",
    }
    for candidate in payload_artifacts or []:
        name = candidate.name
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", name) is None
            or name.casefold() in reserved
            or candidate.suffix.casefold() == ".whl"
        ):
            raise ValueError(f"invalid or duplicate release payload name: {name}")
        target = output / name
        shutil.copyfile(candidate, target)
        artifacts.append(target)
        reserved.add(name.casefold())
    artifacts.sort(key=lambda path: path.name)
    boundary = (root / "packages/dinkster-workers/src/dinkster_workers/boundary.py").read_text()
    protocol = re.search(r"(?m)^PROTOCOL_VERSION = ([0-9]+)$", boundary)
    if protocol is None:
        raise ValueError("worker boundary must declare PROTOCOL_VERSION")
    manifest: dict[str, object] = {
        "format": "dinkster.engine/1",
        "repository": "Kosinkadink/Dinkster",
        "tag": tag or f"main-{git_head(root)}",
        "commit": git_head(root),
        "version": version,
        "workerProtocol": int(protocol[1]),
        "workspaceWheels": [wheel.name for wheel in wheels],
        "artifacts": [
            {"name": path.name, "sha256": sha256(path), "size": path.stat().st_size}
            for path in artifacts
        ],
    }
    manifest_path = output / "release-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksummed = [*artifacts, manifest_path]
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksummed), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="vX.Y.Z; omit for a commit-identified development build")
    parser.add_argument("--frontend-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dist"))
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--external-wheel", type=Path, action="append", default=[])
    parser.add_argument("--base-manifest", type=Path, action="append", default=[])
    parser.add_argument("--target-variant", action="append", default=[])
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--download-external-wheels", action="store_true")
    args = parser.parse_args()
    build(
        Path(__file__).resolve().parent.parent,
        args.output.resolve(),
        args.tag,
        args.frontend_artifact.resolve(),
        uv=args.uv,
        external_wheels=args.external_wheel,
        base_manifests=args.base_manifest,
        target_variants=set(args.target_variant),
        payload_artifacts=args.artifact,
        download_external=args.download_external_wheels,
    )
