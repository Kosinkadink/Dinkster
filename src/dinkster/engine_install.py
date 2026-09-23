"""Mirror-only engine materialization inside an existing install root."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from dinkster_registry import InstallError, Lockfile
from packaging.version import Version

from . import __version__
from .engine_feed import EngineManifest, Mirror, parse_channel, parse_manifest
from .installer import EngineEnvironment, Installer


def owned_path(root: Path, relative: str) -> Path:
    """Reject linked ancestors before touching an installer-owned namespace."""
    root = root.resolve()
    parts = relative.split("/")
    if any(part in {"", ".", ".."} or "\\" in part or ":" in part for part in parts):
        raise InstallError("invalid install-relative content path")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise InstallError(f"installer-owned path must not be a symlink: {path}")
    if not path.resolve().is_relative_to(root):
        raise InstallError("installer-owned content escaped its root")
    return path


def environment_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def native_cell() -> str:
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "mac-arm64"
    if platform.system() == "Windows":
        return "win-cu128"
    if platform.system() == "Linux" and platform.machine().lower() in {"x86_64", "amd64"}:
        return "linux-cu128"
    raise InstallError("select an available engine cell explicitly for this platform")


class EngineInstaller:
    """Stages complete engines, retaining Installer's pack and activation contracts."""

    def __init__(self, installer: Installer) -> None:
        self.installer = installer
        self.root = installer.root.resolve()
        self.content_root = (installer.shared_store or self.root).resolve()

    def _object(self, digest: str) -> Path:
        return owned_path(self.content_root, f"engine-objects/{digest}")

    def _base(self, base_id: str) -> Path:
        return owned_path(self.root, f"engine-bases/{base_id}")

    def _environment(self, manifest_sha256: str) -> Path:
        return owned_path(self.root, f"engine-envs/{manifest_sha256}")

    def _run(self, command: list[str]) -> None:
        env = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_INDEX", "UV_INDEX_URL"):
            env.pop(key, None)
        env.update(UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never", PYTHONNOUSERSITE="1")
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise InstallError(f"engine environment command failed: {detail or result.returncode}")

    def install(
        self,
        mirror: Mirror,
        *,
        channel: str,
        cell: str,
        activate: bool = True,
    ) -> int:
        if channel not in {"stable", "github-live"}:
            raise InstallError("engine channel must be stable or github-live")
        selected = parse_channel(mirror.get_json(f"channels/{channel}.json"))
        if selected.channel != channel:
            raise InstallError("channel document does not match the requested channel")
        if Version(selected.minimum_launcher_version) > Version(__version__):
            raise InstallError("this engine feed requires a newer Dinkster launcher")
        if cell not in selected.cells:
            raise InstallError(f"channel has no engine for cell {cell!r}")
        artifact = selected.cells[cell]
        current = self.installer.current_number()
        lockfile = self.installer.current_lockfile() or Lockfile()

        def stage() -> EngineEnvironment:
            manifest_path = self._object(artifact.sha256)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            mirror.download(artifact, manifest_path)
            manifest = parse_manifest(manifest_path.read_bytes())
            if manifest.commit != selected.commit or manifest.cell != cell:
                raise InstallError("engine manifest does not match its selected channel and cell")
            result = self._materialize(mirror, manifest, artifact.sha256)
            self.installer._serving_interpreter = environment_python(
                self._environment(result.manifest_sha256) / "execution"
            )
            return result

        number, _ = self.installer.apply(
            lockfile,
            stage_environment=stage,
            activate=activate,
            expected_current=current,
            hosting_groups=self.installer.generation_groups(current) if current is not None else (),
            hosting_in_process=(
                self.installer.generation_in_process(current) if current is not None else ()
            ),
            hosting_runtime_pins=(
                self.installer.generation_runtime_pins(current) if current is not None else {}
            ),
        )
        return number

    def _materialize(
        self, mirror: Mirror, manifest: EngineManifest, manifest_sha256: str
    ) -> EngineEnvironment:
        base = self._base(manifest.base.id)
        archive = self._object(manifest.base.archive.sha256)
        mirror.download(manifest.base.archive, archive)
        for wheel in manifest.wheels:
            mirror.download(wheel, self._object(wheel.sha256))
        if not base.exists():
            base.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="extract-", dir=base.parent) as temporary:
                unpacked = Path(temporary) / "base"
                unpacked.mkdir()
                with tarfile.open(archive, "r:*") as source:
                    source.extractall(unpacked, filter="data")
                # Python-build-standalone includes internal interpreter symlinks.
                python = unpacked / manifest.base.python
                if not python.resolve().is_relative_to(unpacked.resolve()) or not python.is_file():
                    raise InstallError("base archive has no contained Python interpreter")
                os.replace(unpacked, base)
        base_python = base / manifest.base.python
        if not base_python.resolve().is_relative_to(base) or not base_python.is_file():
            raise InstallError("base interpreter is missing or escapes its base")
        uv = owned_path(base, "tools/uv.exe" if os.name == "nt" else "tools/uv")
        if not uv.is_file():
            raise InstallError("base archive has no offline uv executable")
        self._run(
            [
                str(base_python),
                "-I",
                "-c",
                "import importlib.metadata as m,json,sys; "
                "pins=json.loads(sys.argv[1]); "
                "assert all(m.version(name)==version for name,version in pins.items()), "
                "'base packages do not match the manifest'",
                json.dumps(dict(manifest.base.packages)),
            ]
        )
        target = self._environment(manifest_sha256)
        complete = owned_path(target, "complete.json")
        if not complete.is_file():
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)
            try:
                with tempfile.TemporaryDirectory(prefix="wheelhouse-", dir=target) as temporary:
                    wheelhouse = Path(temporary)
                    for wheel in manifest.wheels:
                        source = self._object(wheel.sha256)
                        destination = wheelhouse / wheel.filename
                        try:
                            os.link(source, destination)
                        except OSError:
                            shutil.copyfile(source, destination)
                    for kind in ("control", "execution"):
                        venv = target / kind
                        self._run(
                            [
                                str(uv),
                                "--no-config",
                                "--offline",
                                "venv",
                                "--python",
                                str(base_python),
                                "--system-site-packages",
                                str(venv),
                            ]
                        )
                        requirements = wheelhouse / f"{kind}.txt"
                        requirements.write_text(
                            "".join(
                                f"{wheel.name} @ "
                                f"{(wheelhouse / wheel.filename).as_uri()} "
                                f"--hash=sha256:{wheel.sha256}\n"
                                for wheel in manifest.wheels
                                if kind in wheel.environments
                            ),
                            encoding="utf-8",
                        )
                        self._run(
                            [
                                str(uv),
                                "--no-config",
                                "--offline",
                                "pip",
                                "install",
                                "--python",
                                str(environment_python(venv)),
                                "--no-index",
                                "--find-links",
                                str(wheelhouse),
                                "--no-deps",
                                "--require-hashes",
                                "--only-binary",
                                ":all:",
                                "--link-mode",
                                "copy",
                                "--requirement",
                                str(requirements),
                            ]
                        )
                        self._run(
                            [
                                str(environment_python(venv)),
                                "-I",
                                "-m",
                                "pip",
                                "check",
                            ]
                        )
                self._run(
                    [
                        str(environment_python(target / "control")),
                        "-I",
                        "-c",
                        "import dinkster.cli, dinkster_supervisor.supervisor; "
                        "from dinkster_frontend import bundle_path; "
                        "assert bundle_path().joinpath('index.html').is_file()",
                    ]
                )
                self._run(
                    [
                        str(environment_python(target / "execution")),
                        "-I",
                        "-c",
                        "import torch, torchvision, dinkster_inference_torch; "
                        "assert torch.ones(2).sum().item() == 2",
                    ]
                )
                complete.write_text(json.dumps({"manifestSha256": manifest_sha256}))
            except BaseException:
                shutil.rmtree(target)
                raise
        objects = tuple(
            sorted(
                {
                    manifest_sha256,
                    manifest.base.archive.sha256,
                    *(wheel.sha256 for wheel in manifest.wheels),
                }
            )
        )
        environment = EngineEnvironment(
            manifest.base.id, manifest_sha256, manifest.commit, manifest.cell, objects
        )
        self.validate(environment)
        return environment

    def validate(self, environment: EngineEnvironment) -> None:
        target = self._environment(environment.manifest_sha256)
        complete = owned_path(target, "complete.json")
        if not complete.is_file() or json.loads(complete.read_text()) != {
            "manifestSha256": environment.manifest_sha256
        }:
            raise InstallError("engine environment is incomplete")
        manifest_path = self._object(environment.manifest_sha256)
        raw = manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != environment.manifest_sha256:
            raise InstallError("stored engine manifest failed SHA-256 verification")
        manifest = parse_manifest(raw)
        if (manifest.base.id, manifest.commit, manifest.cell) != (
            environment.base_id,
            environment.commit,
            environment.cell,
        ):
            raise InstallError("engine generation does not match its stored manifest")
        base = self._base(environment.base_id)
        base_python = base / manifest.base.python
        if not base_python.is_file() or not base_python.resolve().is_relative_to(base):
            raise InstallError("generation base interpreter is missing")
        for kind in ("control", "execution"):
            owned_path(target, f"{kind}/{'Scripts' if os.name == 'nt' else 'bin'}")
            python = environment_python(target / kind)
            if not python.is_file() or not (
                python.resolve().is_relative_to(target) or python.resolve().is_relative_to(base)
            ):
                raise InstallError(f"generation {kind} interpreter is missing")

    def activate(self, number: int) -> None:
        def validate() -> None:
            environment = self.installer.environment_of(number)
            if environment is None:
                raise InstallError("generation has no engine environment")
            self.validate(environment)

        self.installer.activate(number, validate=validate)

    def rollback(self) -> int:
        def validate(environment: EngineEnvironment) -> None:
            self.validate(environment)
            self.installer._serving_interpreter = environment_python(
                self._environment(environment.manifest_sha256) / "execution"
            )

        return self.installer.rollback(validate_environment=validate)

    def interpreters(self, number: int) -> tuple[Path, Path]:
        environment = self.installer.environment_of(number)
        if environment is None:
            raise InstallError("generation has no installed engine environment")
        self.validate(environment)
        root = self._environment(environment.manifest_sha256)
        return environment_python(root / "control"), environment_python(root / "execution")

    def gc_candidates(self) -> tuple[Path, ...]:
        """Inspect only explicit install namespaces, never library or model paths."""
        roots = [self.root]
        if self.installer.shared_store is not None:
            roots = [root for _, root in self.installer._registered_roots()]
        objects: set[str] = set()
        bases: set[str] = set()
        environments: set[str] = set()
        for root in roots:
            if not root.is_dir():
                raise InstallError(f"registered install root is unavailable: {root}")
            generations = owned_path(root, "generations")
            for path in generations.glob("*.json"):
                if not path.stem.isdigit():
                    continue
                owned_path(root, f"generations/{path.name}")
                record = json.loads(path.read_text())
                Lockfile.from_record_json(json.dumps(record))
                if "engine" not in record:
                    continue
                environment = EngineEnvironment.from_record(record["engine"])
                objects.update(environment.objects)
                if root.resolve() == self.root:
                    bases.add(environment.base_id)
                    environments.add(environment.manifest_sha256)
        candidates: list[Path] = []
        for parent_root, namespace, referenced in (
            (self.root, "engine-bases", bases),
            (self.root, "engine-envs", environments),
            (self.content_root, "engine-objects", objects),
        ):
            parent = owned_path(parent_root, namespace)
            if not parent.is_dir():
                continue
            for child in parent.iterdir():
                owned_path(parent_root, f"{namespace}/{child.name}")
                if child.name not in referenced:
                    candidates.append(child)
        return tuple(sorted(candidates))

    def gc(self) -> tuple[Path, ...]:
        with self.installer._locked():
            candidates = self.gc_candidates()
            for path in candidates:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            return candidates
