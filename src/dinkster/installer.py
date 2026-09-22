"""Pack installation on disk: the install model wired to a real root.

Umbrella-owned like the rest of the wiring: `dinkster_registry` owns the
pure model (lockfiles, plans, generation semantics), `dinkster_workers` owns
venv provisioning, and this module is where a host turns "make this
lockfile the installation" into directories and an atomic pointer swap.

Install root layout (everything content-addressed by artifact digest, so
staging never disturbs what a current generation references):

    artifacts/<hex>.zip   verified artifact archives
    store/<hex>/          unpacked artifacts (immutable once written)
    venvs/<hex>/<pack>/   per-artifact venvs (an upgrade stages a NEW venv)
    generations/<n>.json  lockfile records, append-only numbering
    current               the active generation number (one small file)

Activation is one ``os.replace`` of the ``current`` pointer after all
staging succeeded - a crash mid-install leaves the previous generation
active and only unreferenced staging behind, which ``gc`` reclaims.
Rollback re-activates a previous generation's lockfile as a NEW
generation (its content is still staged), history append-only.

Cross-process job pinning is deliberately not wired: this installer is
CLI-driven and a serving process reads ``current`` at startup, so an
install during a run changes nothing for that process. Live activation
under a running server folds into the pack hot-reload work.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from dinkster_registry import (
    InstallError,
    InstallPlan,
    LockedPack,
    Lockfile,
    SnapshotRecord,
    VenvSpec,
    artifact_digest,
    plan,
)
from dinkster_registry.artifact import MANIFEST_FILENAME, build_artifact, unpack_artifact
from dinkster_schema import canonical_name
from dinkster_workers import (
    PackManifest,
    detect_accelerator,
    detect_runtime,
    diagnose,
    load_manifest,
)
from dinkster_workers.doctor import render_text
from dinkster_workers.provision import (
    ProvisionError,
    bare_pin,
    ensure_group_venv,
    ensure_pack_venv,
    freeze_venv,
    hash_pins,
)
from packaging.requirements import Requirement

from . import __version__ as dinkster_version
from .compose import PackSpec
from .packs import pack_info_from_manifest
from .storelock import StoreLockTimeout, hold_lock

LOCAL_PUBLISHER = "local"
"""Publisher id for unpublished packs installed from local directories.
Identity without authority: the registry never granted anything to it,
and composition still refuses ANY cross-pack claim overlap locally, so
sharing one publisher id never weakens collision checking on this
machine."""

LOCAL_VERSION = "0.0.0"
"""Local packs carry no release version; identity is the artifact digest,
and every local edit plans as an explicit 'reinstall' step."""

Provisioner = Callable[[PackManifest, Path, "VenvSpec | None"], Path]
"""(manifest, venv_root, spec) -> venv python. ``spec`` carries a
snapshot's pins for this venv (None = resolve the manifest's ranges).
Injectable so tests and --no-venv installs skip real venv builds."""

GroupProvisioner = Callable[[Sequence[PackManifest], str, Path, "VenvSpec | None"], Path]
"""(manifests, group name, venv root, spec) -> shared venv python."""

RuntimeProbe = Callable[[str], "tuple[tuple[str, str], ...]"]
"""(accelerator) -> advisory (key, value) runtime facts for snapshot
scope. Injectable so tests never shell out to vendor tools."""

TorchCapabilityProbe = Callable[[Path | str], bool]
RuntimeVersionProbe = Callable[[Path | str], Mapping[str, str]]


@dataclass(frozen=True)
class EngineEnvironment:
    """Immutable engine content selected alongside a generation's pack lockfile."""

    base_id: str
    manifest_sha256: str
    commit: str
    cell: str
    objects: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.objects, tuple):
            raise InstallError("engine objects must be a tuple of content identifiers")
        for digest in (self.base_id, self.manifest_sha256, *self.objects):
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise InstallError("engine content identifiers must be lowercase SHA-256 digests")
        if not isinstance(self.commit, str) or re.fullmatch(r"[0-9a-f]{40}", self.commit) is None:
            raise InstallError("engine commit must be a lowercase Git commit identifier")
        if (
            not isinstance(self.cell, str)
            or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)+", self.cell) is None
        ):
            raise InstallError("engine cell must be a platform-accelerator identifier")
        if len(set(self.objects)) != len(self.objects):
            raise InstallError("engine objects must be unique content identifiers")

    def record(self) -> dict[str, object]:
        return {
            "baseId": self.base_id,
            "manifestSha256": self.manifest_sha256,
            "commit": self.commit,
            "cell": self.cell,
            "objects": list(self.objects),
        }

    @classmethod
    def from_record(cls, record: object) -> EngineEnvironment:
        if not isinstance(record, dict):
            raise InstallError("generation engine environment must be an object")
        fields = ("baseId", "manifestSha256", "commit", "cell")
        if any(not isinstance(record.get(key), str) for key in fields):
            raise InstallError("generation engine environment has missing or invalid fields")
        objects = record.get("objects")
        if not isinstance(objects, list) or any(not isinstance(item, str) for item in objects):
            raise InstallError("generation engine objects must be a list of content identifiers")
        return cls(
            record["baseId"],
            record["manifestSha256"],
            record["commit"],
            record["cell"],
            tuple(objects),
        )


def _probe_torch_capability(interpreter: Path | str) -> bool:
    result = subprocess.run(
        [str(interpreter), "-c", "import torch"], capture_output=True, text=True
    )
    return result.returncode == 0


def _probe_runtime_versions(interpreter: Path | str) -> Mapping[str, str]:
    script = (
        "import importlib.metadata,json;"
        "print(json.dumps({n:importlib.metadata.version(n) "
        "for n in ('torch','dinkster-aimdo')}))"
    )
    result = subprocess.run([str(interpreter), "-c", script], capture_output=True, text=True)
    if result.returncode != 0:
        raise InstallError(
            "cannot read exact torch/dinkster-aimdo versions from serving interpreter: "
            + (result.stderr or result.stdout).strip()
        )
    return cast("dict[str, str]", json.loads(result.stdout))


Freezer = Callable[[Path], "tuple[str, ...]"]
"""(venv python) -> sorted ``name==version`` pins. Injectable so snapshot
tests need no real venvs."""

PinHasher = Callable[[Sequence[str]], "tuple[str, ...]"]
"""(pins) -> the same pins with portable entries hash-annotated
(``name==version --hash=<algo>:<digest> ...``). A network operation
against the package index; injectable so snapshot tests never reach it."""

RegistryFetcher = Callable[[LockedPack], bytes]
"""(entry) -> the artifact archive bytes the entry pins, from a registry.
The fetcher only moves bytes; ``acquire`` verifies them against the
entry's recorded digest before anything is admitted to the store, so a
lying or stale registry can never substitute different bytes. Injectable
so tests need no network; :func:`http_registry_fetcher` is the real one."""

DEFAULT_REGISTRY_TIMEOUT = 60.0


def _is_registry_source(source: str) -> bool:
    """Whether a lockfile source names a registry release (the bare
    ``"registry"`` default, or a ``registry:``-qualified variant)."""
    return source == "registry" or source.startswith("registry:")


def http_registry_fetcher(
    endpoint: str, *, token: str = "", timeout: float = DEFAULT_REGISTRY_TIMEOUT
) -> RegistryFetcher:
    """Acquire exact bytes through the registry's download-ticket contract."""
    base = endpoint.rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise InstallError(f"registry endpoint {endpoint!r} must be an http(s) URL")

    def fetch(entry: LockedPack) -> bytes:
        headers = {"User-Agent": "dinkster-pack"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        ticket_url = f"{base}/v1/packs/{entry.pack}/versions/{entry.version}/download-tickets"
        ticket_request = urllib.request.Request(ticket_url, headers=headers, method="POST")  # noqa: S310 - scheme checked above
        try:
            with urllib.request.urlopen(ticket_request, timeout=timeout) as response:  # noqa: S310
                ticket_raw: object = json.load(response)
            if not isinstance(ticket_raw, dict):
                raise ValueError("ticket is not an object")
            ticket = cast("dict[str, object]", ticket_raw)
            url = ticket.get("url")
            digest = ticket.get("artifactDigest")
            required_headers = ticket.get("requiredHeaders", {})
            if (
                not isinstance(url, str)
                or digest != entry.artifact_digest
                or not isinstance(required_headers, dict)
                or not all(
                    isinstance(name, str) and isinstance(value, str)
                    for name, value in required_headers.items()
                )
            ):
                raise ValueError("ticket fields do not match the locked release")
            url = urllib.parse.urljoin(base + "/", url)
            if not url.startswith(("http://", "https://")):
                raise ValueError("ticket URL is not HTTP(S)")
            request = urllib.request.Request(url, headers=cast("dict[str, str]", required_headers))  # noqa: S310 - service-issued ticket URL
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return response.read()
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise InstallError(
                f"registry download of {entry.pack}@{entry.version} from {base} failed: {exc}"
            ) from exc

    return fetch


def _digest_hex(digest: str) -> str:
    """Directory-safe key: the hex half of a prefixed digest."""
    return digest.partition(":")[2]


def _is_hex_name(name: str) -> bool:
    """Whether a filename is pure lowercase hex (a digest-derived key)."""
    return bool(name) and all(char in "0123456789abcdef" for char in name)


@dataclass(frozen=True)
class _HostingTopology:
    groups: tuple[tuple[str, tuple[str, ...]], ...] = ()
    in_process: tuple[str, ...] = ()
    runtime_pins: tuple[tuple[str, str], ...] = ()


def _hosting_record(topology: _HostingTopology) -> dict[str, object]:
    return {
        "format": "dinkster.hosting/1",
        "inProcess": list(topology.in_process),
        "runtimePins": dict(topology.runtime_pins),
        "venvGroups": {name: list(members) for name, members in topology.groups},
    }


def _load_hosting_topology(root: Path, lockfile: Lockfile) -> _HostingTopology:
    """Read and validate the install root's optional venv-group policy."""
    path = root / "hosting.toml"
    if not path.is_file():
        return _HostingTopology()
    try:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise InstallError(f"{path}: invalid hosting policy: {exc}") from exc
    raw = document.get("venv-groups")
    if raw is not None and not isinstance(raw, dict):
        raise InstallError(f"{path}: [venv-groups] must be a table")
    locked = {entry.pack for entry in lockfile.packs}
    membership: dict[str, str] = {}
    groups: list[tuple[str, tuple[str, ...]]] = []
    for name, members_raw in sorted((raw or {}).items()):
        if (
            not isinstance(name, str)
            or not name
            or name[0] in ".-"
            or any(
                char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for char in name
            )
        ):
            raise InstallError(f"{path}: venv group name {name!r} is not path-safe")
        if not isinstance(members_raw, list) or not all(
            isinstance(member, str) for member in members_raw
        ):
            raise InstallError(f"{path}: venv group {name!r} must be a list of pack names")
        members = tuple(sorted(canonical_name(member) for member in members_raw))
        if len(members) < 2:
            raise InstallError(f"{path}: venv group {name!r} must contain at least 2 packs")
        if len(set(members)) != len(members):
            raise InstallError(f"{path}: venv group {name!r} repeats a pack")
        unknown = sorted(set(members) - locked)
        if unknown:
            raise InstallError(
                f"{path}: venv group {name!r} names pack(s) not in the lockfile: "
                f"{', '.join(unknown)}"
            )
        for member in members:
            previous = membership.get(member)
            if previous is not None:
                raise InstallError(
                    f"{path}: pack {member!r} belongs to both venv groups {previous!r} and {name!r}"
                )
            membership[member] = name
        groups.append((name, members))
    in_process_raw = document.get("in-process", [])
    if not isinstance(in_process_raw, list) or not all(
        isinstance(member, str) for member in in_process_raw
    ):
        raise InstallError(f"{path}: in-process must be a list of pack names")
    in_process = tuple(sorted(canonical_name(member) for member in in_process_raw))
    if len(set(in_process)) != len(in_process):
        raise InstallError(f"{path}: in-process repeats a pack")
    unknown = sorted(set(in_process) - locked)
    if unknown:
        raise InstallError(
            f"{path}: in-process names pack(s) not in the lockfile: {', '.join(unknown)}"
        )
    contradictions = sorted(set(in_process) & set(membership))
    if contradictions:
        raise InstallError(
            f"{path}: pack(s) belong to both a venv group and in-process: "
            f"{', '.join(contradictions)}"
        )
    return _HostingTopology(groups=tuple(groups), in_process=in_process)


def _load_hosting_groups(root: Path, lockfile: Lockfile) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return _load_hosting_topology(root, lockfile).groups


def _aggregate_digest(entries: Sequence[LockedPack]) -> str:
    """Digest sorted canonical pack name + artifact digest pairs."""
    payload = "".join(
        f"{entry.pack}\0{entry.artifact_digest}\n"
        for entry in sorted(entries, key=lambda item: item.pack)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _decode_hosting_record(path: Path) -> _HostingTopology:
    if not path.is_file():
        return _HostingTopology()
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise InstallError(f"{path}: corrupt generation hosting record: {exc}") from exc
    if not isinstance(raw, dict):
        raise InstallError(f"{path}: corrupt generation hosting record")
    format_name = raw.get("format")
    new_shape = format_name == "dinkster.hosting/1"
    if isinstance(format_name, str) and not new_shape:
        raise InstallError(f"{path}: unsupported generation hosting format {format_name!r}")
    groups_raw = raw.get("venvGroups", {}) if new_shape else raw
    if not isinstance(groups_raw, dict):
        raise InstallError(f"{path}: corrupt generation hosting record")
    groups: list[tuple[str, tuple[str, ...]]] = []
    for name, members_raw in groups_raw.items():
        if not isinstance(name, str) or not isinstance(members_raw, list):
            raise InstallError(f"{path}: corrupt generation hosting record")
        members_object = cast("list[object]", members_raw)
        if not all(isinstance(member, str) for member in members_object):
            raise InstallError(f"{path}: corrupt generation hosting record")
        groups.append((name, tuple(cast("list[str]", members_object))))
    in_process_raw = raw.get("inProcess", []) if new_shape else []
    pins_raw = raw.get("runtimePins", {}) if new_shape else {}
    if (
        not isinstance(in_process_raw, list)
        or not all(isinstance(member, str) for member in in_process_raw)
        or not isinstance(pins_raw, dict)
        or not all(
            isinstance(name, str) and isinstance(version, str) for name, version in pins_raw.items()
        )
    ):
        raise InstallError(f"{path}: corrupt generation hosting record")
    return _HostingTopology(
        groups=tuple(sorted(groups)),
        in_process=tuple(cast("list[str]", in_process_raw)),
        runtime_pins=tuple(sorted(cast("dict[str, str]", pins_raw).items())),
    )


def _decode_groups_record(path: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return _decode_hosting_record(path).groups


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe" if os.name == "nt" else venv_dir / "bin" / "python"


def _archive_pack(pack_dir: Path, artifacts_dir: Path) -> tuple[PackManifest, str, Path]:
    """Archive a pack tree content-addressed; the shared step behind every
    unpublished install source. The staging name is unique per call (a
    shared artifacts dir sees concurrent writers from different install
    roots), and admission is an atomic hard-link publication - the store
    only ever contains complete digest-named archives."""
    manifest = load_manifest(pack_dir / MANIFEST_FILENAME)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fd, staging_name = tempfile.mkstemp(prefix="staging-", suffix=".zip.tmp", dir=artifacts_dir)
    os.close(fd)
    staging = Path(staging_name)
    try:
        digest = build_artifact(pack_dir, staging)
        archive = artifacts_dir / f"{_digest_hex(digest)}.zip"
        try:
            os.link(staging, archive)
        except FileExistsError:
            pass
        staging.unlink()
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return manifest, digest, archive


def _unpublished_entry(manifest: PackManifest, digest: str, source: str) -> LockedPack:
    return LockedPack(
        pack=canonical_name(manifest.name),
        version=LOCAL_VERSION,
        artifact_digest=digest,
        publisher=LOCAL_PUBLISHER,
        claims=tuple(canonical_name(claim) for claim in manifest.namespaces),
        source=source,
    )


def lock_local_pack(pack_dir: Path, artifacts_dir: Path) -> tuple[LockedPack, Path]:
    """Archive a local pack directory and lock it: digest identity,
    ``local`` publisher, claims from the manifest. Returns the entry and
    the content-addressed archive path."""
    manifest, digest, archive = _archive_pack(pack_dir, artifacts_dir)
    return _unpublished_entry(manifest, digest, f"local:{pack_dir.resolve()}"), archive


def _run_git(label: str, args: Sequence[str]) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip()[-2000:]
        raise InstallError(f"git {label} failed:\n{tail}")
    return result.stdout.strip()


def lock_git_pack(
    url: str, artifacts_dir: Path, *, ref: str | None = None
) -> tuple[LockedPack, Path]:
    """Clone a git pack and lock it through the identical archive path a
    local install uses - the clone runs git, never pack code, and only the
    deterministic artifact enters the store. Source records the EXACT
    commit (``git:<url>@<commit>``): the lockfile stays reproducible even
    though the branch moves; the artifact digest is still the real pin."""
    with tempfile.TemporaryDirectory(prefix="dinkster-git-") as scratch:
        checkout = Path(scratch) / "checkout"
        _run_git("clone", ["clone", "--quiet", url, str(checkout)])
        if ref is not None:
            _run_git("checkout", ["-C", str(checkout), "checkout", "--quiet", "--detach", ref])
        commit = _run_git("rev-parse", ["-C", str(checkout), "rev-parse", "HEAD"])
        manifest, digest, archive = _archive_pack(checkout, artifacts_dir)
    return _unpublished_entry(manifest, digest, f"git:{url}@{commit}"), archive


class Installer:
    """One install root: stage content-addressed, activate atomically."""

    def __init__(
        self,
        root: Path,
        *,
        provision: Provisioner | None = None,
        group_provision: GroupProvisioner | None = None,
        freeze: Freezer | None = None,
        hasher: PinHasher | None = None,
        workspace_packages: Sequence[Path] = (),
        accelerator: str | None = None,
        runtime_probe: RuntimeProbe | None = None,
        serving_interpreter: Path | str | None = None,
        torch_capability_probe: TorchCapabilityProbe | None = None,
        runtime_version_probe: RuntimeVersionProbe | None = None,
        registry_fetch: RegistryFetcher | None = None,
        shared_store: Path | None = None,
        lock_timeout: float = 600.0,
    ) -> None:
        self.root = root
        self._workspace_packages = tuple(workspace_packages)
        self._provision = provision if provision is not None else self._default_provision
        self._group_provision = (
            group_provision if group_provision is not None else self._default_group_provision
        )
        self._freeze = freeze if freeze is not None else freeze_venv
        self._hasher = hasher if hasher is not None else hash_pins
        self._runtime_probe = runtime_probe if runtime_probe is not None else detect_runtime
        self._serving_interpreter: Path | str = serving_interpreter or os.environ.get(
            "DINKSTER_SERVING_PYTHON", sys.executable
        )
        self._torch_capability_probe = torch_capability_probe or _probe_torch_capability
        self._runtime_version_probe = runtime_version_probe or _probe_runtime_versions
        self._accelerator = accelerator
        self._registry_fetch = registry_fetch
        self._lock_timeout = lock_timeout
        # Store-location indirection: content-addressed state (artifacts,
        # store, venvs) may live in a SHARED store several install roots
        # reference, so N installs never means N copies. The root
        # remembers its store in a pointer file - dinkster-serve and every
        # later CLI invocation pick it up without re-passing flags - and
        # changing an existing pointer is an explicit migration, never a
        # silent switch on a differing flag.
        pointer = root / "shared-store"
        if shared_store is not None:
            shared = Path(shared_store).resolve()
            if pointer.is_file():
                recorded = Path(pointer.read_text().strip())
                if recorded.resolve() != shared:
                    raise InstallError(
                        f"{root} already uses the shared store {recorded}; "
                        f"refusing to silently switch to {shared} (moving a "
                        "root between stores is a migration, not a flag)"
                    )
            else:
                root.mkdir(parents=True, exist_ok=True)
                pointer.write_text(f"{shared}\n")
        elif pointer.is_file():
            shared = Path(pointer.read_text().strip())
        else:
            shared = None
        self._shared = shared
        self._content_root = shared if shared is not None else root
        (root / "generations").mkdir(parents=True, exist_ok=True)
        for name in ("artifacts", "store", "venvs"):
            (self._content_root / name).mkdir(parents=True, exist_ok=True)
        if shared is not None:
            # Registration is what makes cross-root gc sound: the store
            # knows every root whose generations pin its content.
            self._register_root()

    @property
    def accelerator(self) -> str:
        """The accelerator this root provisions for: the explicit
        constructor selection, else host detection (cached; detection is
        driver-file/PATH inspection, never a torch import or CUDA
        context). Selects each pack's ``[pack.extra-requires]`` list and
        is recorded as snapshot scope."""
        if self._accelerator is None:
            self._accelerator = detect_accelerator()
        return self._accelerator

    def _default_provision(
        self, manifest: PackManifest, venv_root: Path, spec: VenvSpec | None
    ) -> Path:
        return ensure_pack_venv(
            manifest,
            venv_root=venv_root,
            workspace_packages=self._workspace_packages,
            pinned=spec.exact if spec is not None else None,
            constraints=spec.constraints if spec is not None else (),
            accelerator=self.accelerator,
        )

    def _default_group_provision(
        self,
        manifests: Sequence[PackManifest],
        group_name: str,
        venv_root: Path,
        spec: VenvSpec | None,
    ) -> Path:
        return ensure_group_venv(
            manifests,
            group_name,
            venv_root=venv_root,
            workspace_packages=self._workspace_packages,
            pinned=spec.exact if spec is not None else None,
            constraints=spec.constraints if spec is not None else (),
            accelerator=self.accelerator,
        )

    def hosting_groups(self, lockfile: Lockfile) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """The current install-root policy, validated against ``lockfile``."""
        return _load_hosting_groups(self.root, lockfile)

    def hosting_topology(self, lockfile: Lockfile) -> _HostingTopology:
        """Current host policy, validated as one mutually exclusive topology."""
        return _load_hosting_topology(self.root, lockfile)

    # -- state ---------------------------------------------------------

    @property
    def shared_store(self) -> Path | None:
        """The shared store this root references, None when the root's
        content is private (the classic single-root layout)."""
        return self._shared

    def _register_root(self) -> None:
        """Idempotently record this root in the shared store's registry.
        The entry name is a digest of the resolved root path, the content
        the path itself - re-registration writes identical bytes, and the
        write is temp+``os.replace`` so a reader (gc in another process)
        never sees a partial entry; concurrent construction needs no
        lock."""
        assert self._shared is not None
        roots_dir = self._shared / "roots"
        roots_dir.mkdir(parents=True, exist_ok=True)
        resolved = str(self.root.resolve())
        entry = roots_dir / hashlib.sha256(resolved.encode()).hexdigest()[:16]
        if not entry.is_file():
            fd, staged = tempfile.mkstemp(prefix="root-", dir=roots_dir)
            with os.fdopen(fd, "w") as handle:
                handle.write(f"{resolved}\n")
            os.replace(staged, entry)

    def _registered_roots(self) -> tuple[tuple[Path, Path], ...]:
        """Every (registry file, install root) the shared store knows.
        Only digest-shaped entry names count - a concurrent
        registration's staging temp is never mistaken for an entry."""
        assert self._shared is not None
        roots_dir = self._shared / "roots"
        if not roots_dir.is_dir():
            return ()
        return tuple(
            (entry, Path(entry.read_text().strip()))
            for entry in sorted(roots_dir.iterdir())
            if entry.is_file() and _is_hex_name(entry.name)
        )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """The store's writer lock: staging and gc mutate the same
        content-addressed directories, possibly from several processes
        once the store is shared."""
        try:
            with hold_lock(self._content_root / ".lock", self._lock_timeout):
                yield
        except StoreLockTimeout as exc:
            raise InstallError(str(exc)) from None

    @property
    def artifacts_dir(self) -> Path:
        return self._content_root / "artifacts"

    def _generation_path(self, number: int) -> Path:
        return self.root / "generations" / f"{number}.json"

    def _generation_groups_path(self, number: int) -> Path:
        return self.root / "generations" / f"{number}.hosting.json"

    def _groups_of(self, number: int) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return _decode_groups_record(self._generation_groups_path(number))

    def _hosting_of(self, number: int) -> _HostingTopology:
        return _decode_hosting_record(self._generation_groups_path(number))

    def _write_hosting(self, number: int, topology: _HostingTopology) -> None:
        path = self._generation_groups_path(number)
        if not topology.groups and not topology.in_process:
            path.unlink(missing_ok=True)
            return
        staged = path.with_suffix(".json.tmp")
        staged.write_text(
            json.dumps(_hosting_record(topology), sort_keys=True, separators=(",", ":")) + "\n"
        )
        os.replace(staged, path)

    def _write_groups(self, number: int, groups: Sequence[tuple[str, tuple[str, ...]]]) -> None:
        self._write_hosting(number, _HostingTopology(groups=tuple(groups)))

    def current_groups(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        number = self.current_number()
        return self._groups_of(number) if number is not None else ()

    def current_in_process(self) -> tuple[str, ...]:
        """In-process placement recorded by the active generation."""
        number = self.current_number()
        return self._hosting_of(number).in_process if number is not None else ()

    def current_runtime_pins(self) -> dict[str, str]:
        """Exact serving-runtime baseline recorded by the active generation."""
        number = self.current_number()
        return dict(self._hosting_of(number).runtime_pins) if number is not None else {}

    def serving_runtime_pins(self) -> dict[str, str]:
        """Read and validate the exact baseline of the designated serve interpreter."""
        if not self._torch_capability_probe(self._serving_interpreter):
            raise InstallError("in-process hosting requires a torch-capable serving interpreter")
        pins = dict(self._runtime_version_probe(self._serving_interpreter))
        if set(pins) != {"torch", "dinkster-aimdo"}:
            raise InstallError(
                "in-process hosting requires exact torch and dinkster-aimdo runtime pins"
            )
        return pins

    def generation_groups(self, number: int) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Effective venv groups recorded for one immutable generation."""
        if not self._generation_path(number).is_file():
            raise InstallError(f"no generation {number} exists in {self.root}")
        return self._groups_of(number)

    def generation_in_process(self, number: int) -> tuple[str, ...]:
        """In-process placement recorded by one immutable generation."""
        if not self._generation_path(number).is_file():
            raise InstallError(f"no generation {number} exists in {self.root}")
        return self._hosting_of(number).in_process

    def generation_runtime_pins(self, number: int) -> dict[str, str]:
        """Exact serving-runtime baseline recorded by one generation."""
        if not self._generation_path(number).is_file():
            raise InstallError(f"no generation {number} exists in {self.root}")
        return dict(self._hosting_of(number).runtime_pins)

    def generation_numbers(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                int(path.stem)
                for path in (self.root / "generations").glob("*.json")
                if path.stem.isdigit()
            )
        )

    def current_number(self) -> int | None:
        pointer = self.root / "current"
        if not pointer.is_file():
            return None
        text = pointer.read_text().strip()
        if not text.isdigit():
            raise InstallError(f"{pointer}: corrupt current-generation pointer {text!r}")
        return int(text)

    def lockfile_of(self, number: int) -> Lockfile:
        path = self._generation_path(number)
        if not path.is_file():
            raise InstallError(f"no generation {number} exists in {self.root}")
        return Lockfile.from_record_json(path.read_text())

    def current_lockfile(self) -> Lockfile | None:
        number = self.current_number()
        return self.lockfile_of(number) if number is not None else None

    def environment_of(self, number: int) -> EngineEnvironment | None:
        """Read the engine selected by a generation; old pack-only records have none."""
        self.lockfile_of(number)
        record = json.loads(self._generation_path(number).read_text())
        return EngineEnvironment.from_record(record["engine"]) if "engine" in record else None

    def _activate_pointer(self, number: int) -> None:
        pointer_staged = self.root / "current.tmp"
        pointer_staged.write_text(f"{number}\n")
        os.replace(pointer_staged, self.root / "current")

    def activate(self, number: int, *, validate: Callable[[], None] | None = None) -> None:
        """Activate a staged generation only if its predecessor is still current.

        Validation runs under the writer lock so GC cannot remove the staged
        environment between checking it and publishing the pointer.
        """
        with self._locked():
            self.lockfile_of(number)
            self.environment_of(number)
            current = self.current_number()
            if current == number:
                return
            record = json.loads(self._generation_path(number).read_text())
            if "previousGeneration" not in record or record["previousGeneration"] != current:
                raise InstallError("staged generation is stale: the active generation changed")
            if validate is not None:
                validate()
            self._activate_pointer(number)

    # -- staging + activation -------------------------------------------

    @property
    def store_root(self) -> Path:
        """Parent of every content-addressed store directory. Public so
        live activation can classify a composed pack as managed (its
        manifest lives under here) versus a dev --pack addition."""
        return self._content_root / "store"

    def _store_dir(self, entry: LockedPack) -> Path:
        return self.store_root / _digest_hex(entry.artifact_digest)

    def _venv_root(self, entry: LockedPack) -> Path:
        """Per-artifact venv parent; ``ensure_pack_venv`` names the venv
        after the MANIFEST name inside it (which may differ from the
        canonical ``entry.pack`` in case/separators).

        A private root has one accelerator, so digest alone keys its
        venvs. A shared store serves roots provisioned for DIFFERENT
        accelerators, and a venv's contents depend on that choice (the
        [pack.extra-requires] list), so shared venvs key by
        accelerator/digest - same bytes, different resolution, different
        venv."""
        venvs = self._content_root / "venvs"
        if self._shared is not None:
            venvs = venvs / self.accelerator
        return venvs / _digest_hex(entry.artifact_digest)

    def _group_venv_root(self, entries: Sequence[LockedPack]) -> Path:
        venvs = self._content_root / "venvs"
        if self._shared is not None:
            venvs = venvs / self.accelerator
        return venvs / _aggregate_digest(entries)

    def _topology_staged(
        self,
        target: Lockfile,
        groups: Sequence[tuple[str, tuple[str, ...]]],
        in_process: Sequence[str] = (),
    ) -> bool:
        entries = {entry.pack: entry for entry in target.packs}
        grouped = {member for _, members in groups for member in members}
        for name, members in groups:
            group_entries = tuple(entries[member] for member in members)
            if not _venv_python(self._group_venv_root(group_entries) / name).is_file():
                return False
        for entry in target.packs:
            if entry.pack in grouped or entry.pack in in_process:
                continue
            manifest_path = self._store_dir(entry) / MANIFEST_FILENAME
            if not manifest_path.is_file():
                return False
            manifest = load_manifest(manifest_path)
            if not _venv_python(self._venv_root(entry) / manifest.name).is_file():
                return False
        return True

    def _aggregate_venvs_present(self, target: Lockfile) -> bool:
        """Whether a prior grouped layout exists for this target."""
        artifact_keys = {_digest_hex(entry.artifact_digest) for entry in target.packs}
        root = self._content_root / "venvs"
        if self._shared is not None:
            root = root / self.accelerator
        return root.is_dir() and any(
            child.is_dir() and child.name not in artifact_keys for child in root.iterdir()
        )

    def venv_staged(self, entry: LockedPack) -> bool:
        """Whether this artifact's venv is already materialized - plan
        display uses it to show which steps stage a NEW venv (real work,
        real disk) versus reuse a content-addressed one."""
        root = self._venv_root(entry)
        return root.is_dir() and any(root.iterdir())

    def artifact_available(self, entry: LockedPack) -> bool:
        """Whether the exact bytes this entry pins are present (unpacked
        store or archived artifact) - what apply-from-plan checks up front
        so a stale-gc'd plan refuses loudly instead of failing mid-stage."""
        if (self._store_dir(entry) / MANIFEST_FILENAME).is_file():
            return True
        return (self.artifacts_dir / f"{_digest_hex(entry.artifact_digest)}.zip").is_file()

    def acquirable(self, entry: LockedPack) -> bool:
        """Whether ``acquire`` knows how to re-fetch this entry's bytes
        from its recorded provenance: ``local:`` and ``git:`` sources
        always, registry sources only when a registry fetcher is
        configured (an endpoint is configuration, not provenance)."""
        if entry.source.startswith(("local:", "git:")):
            return True
        return _is_registry_source(entry.source) and self._registry_fetch is not None

    def acquire(self, entry: LockedPack) -> None:
        """Re-fetch the EXACT bytes a lockfile entry pins, from its
        recorded provenance, into the content-addressed artifact store.
        Idempotent (present bytes are never re-fetched) and
        digest-verified: the rebuilt or downloaded artifact must hash to
        the recorded ``artifact_digest`` or this refuses loudly - a
        moved-on local tree, rewritten git history, or lying registry is
        NEVER silently substituted.
        The store only ever gains the recorded bytes; a mismatched
        candidate is discarded with the scratch directory."""
        if self.artifact_available(entry):
            return
        source = entry.source
        if source.startswith("local:"):
            self._acquire_from_tree(entry, Path(source[len("local:") :]), source)
        elif source.startswith("git:"):
            url, sep, commit = source[len("git:") :].rpartition("@")
            if not sep or not url or not commit:
                raise InstallError(
                    f"cannot re-acquire {entry.pack}: recorded source {source!r} "
                    f"is not of the form git:<url>@<commit>"
                )
            with tempfile.TemporaryDirectory(prefix="dinkster-git-") as scratch:
                checkout = Path(scratch) / "checkout"
                _run_git("clone", ["clone", "--quiet", url, str(checkout)])
                _run_git(
                    "checkout",
                    ["-C", str(checkout), "checkout", "--quiet", "--detach", commit],
                )
                self._acquire_from_tree(entry, checkout, source)
        elif _is_registry_source(source):
            if self._registry_fetch is None:
                raise InstallError(
                    f"cannot re-acquire {entry.pack}: source {source!r} is a "
                    f"registry release but no registry endpoint is configured "
                    f"(pass --registry or set $DINKSTER_REGISTRY)"
                )
            data = self._registry_fetch(entry)
            digest = artifact_digest(data)
            if digest != entry.artifact_digest:
                raise InstallError(
                    f"registry bytes for {entry.pack} hash {digest}, but the "
                    f"lockfile pins {entry.artifact_digest}; refusing to "
                    f"substitute different bytes"
                )
            # scratch inside the artifacts dir so the final rename is atomic
            with tempfile.TemporaryDirectory(prefix="acquire-", dir=self.artifacts_dir) as scratch:
                candidate = Path(scratch) / "artifact.zip"
                candidate.write_bytes(data)
                os.replace(candidate, self.artifacts_dir / f"{_digest_hex(digest)}.zip")
        else:
            raise InstallError(
                f"cannot re-acquire {entry.pack}: source {source!r} is not "
                f"re-acquirable (unrecognized source scheme); "
                f"stage the artifact yourself"
            )

    def _acquire_from_tree(self, entry: LockedPack, pack_dir: Path, source: str) -> None:
        """Archive ``pack_dir`` and admit it to the store ONLY if it
        hashes to the entry's recorded digest."""
        if not pack_dir.is_dir():
            raise InstallError(
                f"cannot re-acquire {entry.pack}: recorded source {source!r} "
                f"no longer exists at {pack_dir}"
            )
        # scratch inside the artifacts dir so the final rename is atomic
        with tempfile.TemporaryDirectory(prefix="acquire-", dir=self.artifacts_dir) as scratch:
            candidate = Path(scratch) / "artifact.zip"
            digest = build_artifact(pack_dir, candidate)
            if digest != entry.artifact_digest:
                raise InstallError(
                    f"re-acquired bytes for {entry.pack} from {source} hash "
                    f"{digest}, but the lockfile pins {entry.artifact_digest}; "
                    f"the source has changed since capture - refusing to "
                    f"substitute different bytes"
                )
            os.replace(candidate, self.artifacts_dir / f"{_digest_hex(digest)}.zip")

    def manifest_of(self, entry: LockedPack) -> PackManifest | None:
        """The manifest of an entry's exact bytes, from the unpacked
        store when staged, else read out of the archived artifact - so
        PLAN display can show compatibility (declared platforms,
        accelerator-conditional requirements) before anything mutates.
        None when neither is present (a gc'd plan; apply refuses that
        separately)."""
        store_manifest = self._store_dir(entry) / MANIFEST_FILENAME
        if store_manifest.is_file():
            return load_manifest(store_manifest)
        archive = self.artifacts_dir / f"{_digest_hex(entry.artifact_digest)}.zip"
        if not archive.is_file():
            return None
        with tempfile.TemporaryDirectory(prefix="dinkster-manifest-") as scratch:
            with zipfile.ZipFile(archive) as bundle:
                bundle.extract(MANIFEST_FILENAME, scratch)
            return load_manifest(Path(scratch) / MANIFEST_FILENAME)

    def venv_python(self, entry: LockedPack) -> Path | None:
        """The staged venv's interpreter for this entry, or None when no
        venv is materialized. Needs the store manifest (the venv is named
        after the manifest name, which may differ from the canonical
        pack name in case/separators)."""
        number = self.current_number()
        if number is None:
            return None
        return self._venv_python_for(entry, self.lockfile_of(number), self._groups_of(number))

    def _venv_python_for(
        self,
        entry: LockedPack,
        lockfile: Lockfile,
        groups: Sequence[tuple[str, tuple[str, ...]]],
    ) -> Path | None:
        store_manifest = self._store_dir(entry) / MANIFEST_FILENAME
        if not store_manifest.is_file():
            return None
        manifest = load_manifest(store_manifest)
        group = None
        if lockfile.get(entry.pack) is not None:
            for name, members in groups:
                if entry.pack in members:
                    group_entries = tuple(lockfile.get(member) for member in members)
                    if all(member_entry is not None for member_entry in group_entries):
                        group = (
                            name,
                            tuple(member_entry for member_entry in group_entries if member_entry),
                        )
                    break
        venv_dir = (
            self._group_venv_root(group[1]) / group[0]
            if group is not None
            else self._venv_root(entry) / manifest.name
        )
        python = (
            venv_dir / "Scripts" / "python.exe" if os.name == "nt" else venv_dir / "bin" / "python"
        )
        return python if python.exists() else None

    # -- snapshots -------------------------------------------------------

    def snapshot(self, *, hashes: bool = False) -> SnapshotRecord:
        """Capture the current installation as a complete environment
        record: the generation lockfile plus a per-venv freeze of what
        range resolution actually installed. Read-only against the
        installation. Packs whose venv was never staged are recorded
        honestly unpinned (absent from the venvs map) - never given fake
        pins.

        ``hashes`` additionally annotates each freeze's portable pins
        with their artifact hashes (the resolution lock's artifact
        identity, verified on exact replay) - the one part of capture
        that talks to the package index, so it is opt-in and fails
        loudly rather than writing a partially-verifiable record."""
        number = self.current_number()
        if number is None:
            raise InstallError("nothing is installed; nothing to snapshot")
        current = self.lockfile_of(number)
        topology = self._hosting_of(number)
        groups = topology.groups
        in_process = set(topology.in_process)
        venvs: dict[str, tuple[str, ...]] = {}
        frozen: dict[Path, tuple[str, ...]] = {}
        for entry in current.packs:
            if entry.pack in in_process:
                continue
            python = self._venv_python_for(entry, current, groups)
            if python is not None:
                pins = frozen.get(python)
                if pins is None:
                    pins = self._freeze(python)
                    if hashes:
                        pins = self._hasher(pins)
                    frozen[python] = pins
                venvs[entry.pack] = pins
        return SnapshotRecord.of(
            current,
            venvs,
            python=platform.python_version(),
            platform=f"{sys.platform}-{platform.machine()}",
            dinkster=dinkster_version,
            accelerator=self.accelerator,
            # Advisory runtime facts (CUDA/driver/ROCm versions) via the
            # vendor status tool - zero VRAM, never a torch import.
            runtime=dict(self.runtime_facts()),
            venv_groups=dict(groups),
            in_process=topology.in_process,
            runtime_pins=dict(topology.runtime_pins),
        )

    def runtime_facts(self) -> tuple[tuple[str, str], ...]:
        """Advisory runtime/toolchain facts for this host's accelerator
        (CUDA/driver/ROCm versions), best-effort and possibly empty.
        What snapshot records and restore compares against - narration
        only, never a pin-reuse input."""
        return self._runtime_probe(self.accelerator)

    def venv_drift(self, entry: LockedPack, pins: tuple[str, ...]) -> tuple[str, ...] | None:
        """How a STAGED venv differs from a snapshot's pins: the sorted
        symmetric difference, or None when no venv is staged. Empty means
        the venv matches. Version identity only - a freeze reports
        ``name==version``, so hash annotations are stripped before
        comparing, never counted as drift. Reporting-only - shared
        content-addressed venvs are never silently rebuilt (a running
        worker may hold them)."""
        python = self.venv_python(entry)
        if python is None:
            return None
        actual = set(self._freeze(python))
        recorded = {bare_pin(pin) for pin in pins}
        return tuple(sorted(actual.symmetric_difference(recorded)))

    def _stage(
        self,
        target: Lockfile,
        artifacts: Mapping[str, Path],
        venvs: bool,
        venv_specs: Mapping[str, VenvSpec] | None = None,
        allow_doctor_findings: bool = False,
        hosting_groups: Sequence[tuple[str, tuple[str, ...]]] | None = None,
        in_process: Sequence[str] = (),
        runtime_pins: Mapping[str, str] | None = None,
    ) -> None:
        """Materialize every target entry; content-addressing makes this
        incremental and re-runnable (present store dirs are kept)."""
        groups = (
            tuple(hosting_groups) if hosting_groups is not None else self.hosting_groups(target)
        )
        if groups:
            self._stage_grouped(
                target,
                artifacts,
                venvs,
                venv_specs,
                allow_doctor_findings,
                groups,
                tuple(in_process),
                runtime_pins or {},
            )
            return
        for entry in target.packs:
            store = self._store_dir(entry)
            if not (store / MANIFEST_FILENAME).is_file():
                archive = artifacts.get(entry.artifact_digest)
                if archive is None:
                    candidate = self.artifacts_dir / f"{_digest_hex(entry.artifact_digest)}.zip"
                    if not candidate.is_file():
                        raise InstallError(
                            f"pack {entry.pack!r} ({entry.artifact_digest}) is not "
                            f"staged and no artifact was provided"
                        )
                    archive = candidate
                unpack_artifact(archive, store, entry.artifact_digest)
            manifest = load_manifest(store / MANIFEST_FILENAME)
            interpreter: Path | str = (
                self._serving_interpreter if entry.pack in in_process else sys.executable
            )
            if entry.pack not in in_process and venvs:
                spec = venv_specs.get(entry.pack) if venv_specs is not None else None
                interpreter = self._provision(manifest, self._venv_root(entry), spec)
            if entry.pack in in_process:
                self._validate_in_process_runtime(entry.pack, manifest, runtime_pins or {})
            report = diagnose(manifest.path, interpreter=interpreter)
            if not report.ok and entry.source.startswith(("local:", "git:")):
                rendered = render_text(report)
                if not allow_doctor_findings:
                    raise InstallError(f"doctor refused staged pack {entry.pack!r}:\n{rendered}")
                print(
                    f"WARNING: allowing doctor findings for staged pack {entry.pack!r}:\n{rendered}"
                )

    def _stage_grouped(
        self,
        target: Lockfile,
        artifacts: Mapping[str, Path],
        venvs: bool,
        venv_specs: Mapping[str, VenvSpec] | None,
        allow_doctor_findings: bool,
        groups: Sequence[tuple[str, tuple[str, ...]]],
        in_process: Sequence[str],
        runtime_pins: Mapping[str, str],
    ) -> None:
        """Stage policy groups after every member artifact is unpacked."""
        manifests: dict[str, PackManifest] = {}
        entries = {entry.pack: entry for entry in target.packs}
        for entry in target.packs:
            store = self._store_dir(entry)
            if not (store / MANIFEST_FILENAME).is_file():
                archive = artifacts.get(entry.artifact_digest)
                if archive is None:
                    candidate = self.artifacts_dir / f"{_digest_hex(entry.artifact_digest)}.zip"
                    if not candidate.is_file():
                        raise InstallError(
                            f"pack {entry.pack!r} ({entry.artifact_digest}) is not "
                            f"staged and no artifact was provided"
                        )
                    archive = candidate
                unpack_artifact(archive, store, entry.artifact_digest)
            manifests[entry.pack] = load_manifest(store / MANIFEST_FILENAME)

        member_to_group = {
            member: (name, members) for name, members in groups for member in members
        }
        interpreters: dict[str, Path | str] = {}
        if venvs:
            for name, members in groups:
                group_entries = tuple(entries[member] for member in members)
                member_specs = [
                    venv_specs.get(member) if venv_specs is not None else None for member in members
                ]
                first_spec = member_specs[0]
                if any(spec != first_spec for spec in member_specs[1:]):
                    raise InstallError(
                        f"venv group {name!r} has contradictory per-member exact "
                        f"pins or constraints in the plan"
                    )
                try:
                    interpreter = self._group_provision(
                        tuple(manifests[member] for member in members),
                        name,
                        self._group_venv_root(group_entries),
                        first_spec,
                    )
                except ProvisionError as exc:
                    raise InstallError(
                        f"dependency resolution for venv group {name!r} failed: {exc}"
                    ) from exc
                interpreters.update({member: interpreter for member in members})
            for entry in target.packs:
                if entry.pack not in member_to_group and entry.pack not in in_process:
                    spec = venv_specs.get(entry.pack) if venv_specs is not None else None
                    interpreters[entry.pack] = self._provision(
                        manifests[entry.pack], self._venv_root(entry), spec
                    )
        for entry in target.packs:
            if entry.pack in in_process:
                self._validate_in_process_runtime(entry.pack, manifests[entry.pack], runtime_pins)
            interpreter = interpreters.get(
                entry.pack,
                self._serving_interpreter if entry.pack in in_process else sys.executable,
            )
            report = diagnose(manifests[entry.pack].path, interpreter=interpreter)
            if report.ok or not entry.source.startswith(("local:", "git:")):
                continue
            rendered = render_text(report)
            if not allow_doctor_findings:
                raise InstallError(f"doctor refused staged pack {entry.pack!r}:\n{rendered}")
            print(f"WARNING: allowing doctor findings for staged pack {entry.pack!r}:\n{rendered}")

    def _validate_in_process_runtime(
        self,
        pack: str,
        manifest: PackManifest,
        baseline: Mapping[str, str],
    ) -> None:
        if not self._torch_capability_probe(self._serving_interpreter):
            raise InstallError(
                f"in-process pack {pack!r} requires a torch-capable serving interpreter"
            )
        actual = dict(self._runtime_version_probe(self._serving_interpreter))
        if actual != dict(baseline):
            raise InstallError(
                f"in-process pack {pack!r} runtime changed during staging: "
                f"expected {dict(baseline)!r}, installed {actual!r}"
            )
        for raw in manifest.requires_for(self.accelerator):
            requirement = Requirement(raw)
            name = canonical_name(requirement.name)
            if name == "dinkster-aimdo":
                raise InstallError(
                    f"in-process pack {pack!r} must not declare a dinkster-aimdo dependency; "
                    "model residency is owned by the engine tenant registry"
                )
            if name != "torch":
                continue
            expected = actual[name]
            if str(requirement.specifier) != f"=={expected}":
                raise InstallError(
                    f"in-process pack {pack!r} must pin {name} exactly to host "
                    f"version {expected}; declared {raw!r}"
                )

    def apply(
        self,
        target: Lockfile,
        artifacts: Mapping[str, Path] | None = None,
        *,
        venvs: bool = True,
        venv_specs: Mapping[str, VenvSpec] | None = None,
        allow_doctor_findings: bool = False,
        hosting_groups: Sequence[tuple[str, tuple[str, ...]]] | None = None,
        hosting_in_process: Sequence[str] | None = None,
        hosting_runtime_pins: Mapping[str, str] | None = None,
        environment: EngineEnvironment | Literal["current"] | None = "current",
        stage_environment: Callable[[], EngineEnvironment | None] | None = None,
        activate: bool = True,
        expected_current: int | Literal["any"] | None = "any",
    ) -> tuple[int, InstallPlan]:
        """Make ``target`` the installation. Returns (generation, plan).

        Idempotent for identical content (no new generation). All staging
        happens before the pointer swap; the swap itself is one atomic
        ``os.replace``. ``venv_specs`` (a snapshot's per-pack pins, exact
        for same-scope restores, portable constraints for cross-scope)
        shapes provisioning of MISSING venvs; already-staged venvs are
        never rebuilt here (they are shared, content-addressed state -
        drift against pins is reported by the restore CLI, never
        silently repaired).

        Engine selection is inherited for pack updates. Explicit ``None``
        restores a pack-only generation. ``stage_environment`` materializes
        engine content under the same writer lock as pack staging and GC;
        ``activate=False`` records it without changing the running selection.

        The whole operation holds the store's writer lock: unpack writes
        a store directory file by file, so another process's apply or gc
        must never observe (or delete) a half-staged directory - and the
        generation record must land inside the same critical section, or
        a gc between staging and recording would see freshly staged
        content as unreferenced and collect it. The idempotence check
        also re-reads state under the lock, so two concurrent identical
        applies produce one generation, not two."""
        with self._locked():
            current = self.current_lockfile()
            current_number = self.current_number()
            if expected_current != "any" and current_number != expected_current:
                raise InstallError("the active generation changed while preparing the operation")
            current_environment = (
                self.environment_of(current_number) if current_number is not None else None
            )
            selected_environment = current_environment if environment == "current" else environment
            if stage_environment is not None:
                selected_environment = stage_environment()
            explicit_topology = (
                hosting_groups is not None
                or hosting_in_process is not None
                or hosting_runtime_pins is not None
            )
            policy = _HostingTopology() if explicit_topology else self.hosting_topology(target)
            groups = tuple(hosting_groups) if hosting_groups is not None else policy.groups
            in_process = (
                tuple(hosting_in_process) if hosting_in_process is not None else policy.in_process
            )
            grouped = {member for _, members in groups for member in members}
            contradictions = sorted(grouped & set(in_process))
            if contradictions:
                raise InstallError(
                    "pack(s) belong to both a venv group and in-process: "
                    + ", ".join(contradictions)
                )
            locked = {entry.pack for entry in target.packs}
            unknown = sorted((grouped | set(in_process)) - locked)
            if unknown:
                raise InstallError(
                    "hosting topology names pack(s) not in the lockfile: " + ", ".join(unknown)
                )
            runtime_pins = (
                dict(hosting_runtime_pins)
                if hosting_runtime_pins is not None
                else dict(policy.runtime_pins)
            )
            if in_process and not runtime_pins and not explicit_topology:
                runtime_pins = self.serving_runtime_pins()
            if not in_process:
                runtime_pins = {}
            if in_process and set(runtime_pins) != {"torch", "dinkster-aimdo"}:
                raise InstallError(
                    "in-process hosting requires exact torch and dinkster-aimdo runtime pins"
                )
            topology = _HostingTopology(
                groups=groups,
                in_process=in_process,
                runtime_pins=tuple(sorted(runtime_pins.items())),
            )
            if not venvs:
                groups = ()
                topology = replace(topology, groups=())
            if (
                current is not None
                and current.record_digest() == target.record_digest()
                and selected_environment == current_environment
            ):
                assert current_number is not None
                current_topology = self._hosting_of(current_number)
                if (
                    topology == current_topology
                    and venvs
                    and (
                        (groups or self._aggregate_venvs_present(target))
                        and not self._topology_staged(target, groups, in_process)
                    )
                ):
                    self._stage(
                        target,
                        artifacts if artifacts is not None else {},
                        venvs,
                        venv_specs,
                        allow_doctor_findings,
                        groups,
                        in_process,
                        runtime_pins,
                    )
                    return current_number, InstallPlan()
                if topology == current_topology:
                    return current_number, InstallPlan()
            steps = plan(current if current is not None else Lockfile(), target)
            self._stage(
                target,
                artifacts if artifacts is not None else {},
                venvs,
                venv_specs,
                allow_doctor_findings,
                groups,
                in_process,
                runtime_pins,
            )
            numbers = self.generation_numbers()
            number = (numbers[-1] if numbers else 0) + 1
            generation_path = self._generation_path(number)
            staged = generation_path.with_suffix(".json.tmp")
            record = json.loads(target.record_json())
            if selected_environment is not None:
                record["engine"] = selected_environment.record()
            record["hosting"] = _hosting_record(topology)
            record["previousGeneration"] = current_number
            staged.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")))
            try:
                self._write_hosting(number, topology)
                os.replace(staged, generation_path)
            except BaseException:
                staged.unlink(missing_ok=True)
                self._generation_groups_path(number).unlink(missing_ok=True)
                raise
            if activate:
                self._activate_pointer(number)
        return number, steps

    def rollback(
        self,
        *,
        venvs: bool = True,
        validate_environment: Callable[[EngineEnvironment], None] | None = None,
    ) -> int:
        """Re-activate the previous active generation, as a
        new generation. Its content is still staged unless gc removed it -
        in which case this fails loudly instead of activating a hole."""
        current = self.current_number()
        if current is None:
            raise InstallError("nothing is installed; nothing to roll back")
        previous = [n for n in self.generation_numbers() if n < current]
        if not previous:
            raise InstallError("no earlier generation exists to roll back to")
        record = json.loads(self._generation_path(current).read_text())
        previous_number = record.get("previousGeneration", previous[-1])
        if (
            not isinstance(previous_number, int)
            or isinstance(previous_number, bool)
            or previous_number not in previous
        ):
            raise InstallError("no earlier activated generation exists to roll back to")
        previous_topology = self._hosting_of(previous_number)
        previous_environment = self.environment_of(previous_number)

        def restore_environment() -> EngineEnvironment | None:
            if previous_environment is not None and validate_environment is not None:
                validate_environment(previous_environment)
            return previous_environment

        number, _ = self.apply(
            self.lockfile_of(previous_number),
            venvs=venvs,
            hosting_groups=previous_topology.groups,
            hosting_in_process=previous_topology.in_process,
            hosting_runtime_pins=dict(previous_topology.runtime_pins),
            stage_environment=restore_environment,
            expected_current=current,
        )
        return number

    def _referenced_digests(self) -> set[str]:
        """Digest hexes some generation still pins. Private root: this
        root's generations. Shared store: the union over EVERY registered
        root's generations - content another install still references is
        never this root's to collect. A registered root whose directory
        is gone fails loudly (it may be unmounted, not deleted); removing
        its registry file is the explicit way to forget it."""
        if self._shared is None:
            roots: list[Path] = [self.root]
        else:
            roots = []
            for entry, root in self._registered_roots():
                if not root.is_dir():
                    raise InstallError(
                        f"registered install root {root} does not exist; gc "
                        f"cannot prove its generations reference nothing. If "
                        f"the root is gone for good, delete {entry} and re-run"
                    )
                roots.append(root)
        referenced: set[str] = set()
        for root in roots:
            generations = root / "generations"
            if not generations.is_dir():
                continue
            for path in sorted(generations.glob("*.json")):
                if not path.stem.isdigit():
                    continue
                generation = Lockfile.from_record_json(path.read_text())
                groups_path = generations / f"{path.stem}.hosting.json"
                groups = _decode_groups_record(groups_path)
                entries = {entry.pack: entry for entry in generation.packs}
                for entry_pack in generation.packs:
                    referenced.add(_digest_hex(entry_pack.artifact_digest))
                for name, members in groups:
                    if all(member in entries for member in members):
                        aggregate = _aggregate_digest(tuple(entries[member] for member in members))
                        referenced.add(aggregate)
                        referenced.add(f"{aggregate}/{name}")
        return referenced

    def _venv_dirs(self) -> tuple[tuple[str, str], ...]:
        """(store-relative path, digest hex) for every materialized venv
        dir - one level deep on a private root, accelerator/digest on a
        shared store."""
        venvs = self._content_root / "venvs"
        found: list[tuple[str, str]] = []
        if self._shared is None:
            for child in venvs.iterdir():
                if child.is_dir():
                    found.append((f"venvs/{child.name}", child.name))
        else:
            for accel in venvs.iterdir():
                if not accel.is_dir():
                    continue
                for child in accel.iterdir():
                    if child.is_dir():
                        found.append((f"venvs/{accel.name}/{child.name}", child.name))
        return tuple(found)

    def gc_candidates(self) -> tuple[str, ...]:
        """Store/venv/artifact content no generation references - what
        ``gc`` would delete, computed without deleting (the CLI shows this
        before asking to proceed). Paths are relative to the content
        root (the shared store when one is configured)."""
        referenced = self._referenced_digests()
        candidates: list[str] = []
        for child in self.store_root.iterdir():
            if child.is_dir() and child.name not in referenced:
                candidates.append(f"store/{child.name}")
        for rel, digest in self._venv_dirs():
            if digest not in referenced:
                candidates.append(rel)
                continue
            group_prefix = f"{digest}/"
            group_refs = {item for item in referenced if item.startswith(group_prefix)}
            if group_refs:
                parent = self._content_root / rel
                for child in parent.iterdir():
                    if child.is_dir() and f"{digest}/{child.name}" not in group_refs:
                        candidates.append(f"{rel}/{child.name}")
        for archive in self.artifacts_dir.glob("*.zip"):
            if archive.stem not in referenced:
                candidates.append(f"artifacts/{archive.name}")
        return tuple(sorted(candidates))

    def gc(self) -> tuple[str, ...]:
        """Delete store/venv/artifact content no generation references.
        Generation records themselves are never collected - they are the
        audit trail and they are tiny. Runs under the store's writer
        lock, and recomputes candidates inside it: on a shared store a
        preview computed before the lock could be stale against another
        root's just-applied generation."""
        with self._locked():
            removed = self.gc_candidates()
            for item in removed:
                path = self._content_root / item
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        return removed

    # -- serving ---------------------------------------------------------

    def packs_for_serving(self) -> list[PackSpec]:
        """The current generation as compose_serving specs: manifests from
        the immutable store, interpreters from the per-artifact venvs
        (host interpreter where no venv was staged).

        Each spec carries its own packs-table entry so the lockfile's
        provenance (the digest pin, install source, publisher, and - for
        registry releases - the version) reaches /api/nodes. Unpublished
        installs (local/git) omit the version rather than surface the
        ``0.0.0`` sentinel: they have no release identity, the digest is
        the real pin, and wire omission MEANS unpinned."""
        number = self.current_number()
        if number is None:
            return []
        lockfile = self.lockfile_of(number)
        topology = self._hosting_of(number)
        groups = topology.groups
        in_process = set(topology.in_process)
        membership = {member: (name, members) for name, members in groups for member in members}
        entries = {entry.pack: entry for entry in lockfile.packs}
        specs: list[PackSpec] = []
        for entry in lockfile.packs:
            store = self._store_dir(entry)
            manifest_path = store / MANIFEST_FILENAME
            if not manifest_path.is_file():
                raise InstallError(
                    f"generation content for pack {entry.pack!r} is missing from "
                    f"the store ({store}); the install root is corrupt"
                )
            manifest = load_manifest(manifest_path)
            group = membership.get(entry.pack)
            if entry.pack in in_process:
                venv_dir = None
            elif group is None:
                venv_dir = self._venv_root(entry) / manifest.name
            else:
                name, members = group
                venv_dir = (
                    self._group_venv_root(tuple(entries[member] for member in members)) / name
                )
            python = _venv_python(venv_dir) if venv_dir is not None else None
            info = replace(
                pack_info_from_manifest(manifest),
                version=entry.version if entry.publisher != LOCAL_PUBLISHER else "",
                artifact_digest=entry.artifact_digest,
                source=entry.source,
                publisher=entry.publisher,
            )
            specs.append(
                PackSpec(
                    manifest=manifest_path,
                    python=str(python) if python is not None and python.is_file() else None,
                    env={
                        "PYTHONPATH": os.pathsep.join(
                            str(self._store_dir(entries[member])) for member in group[1]
                        )
                        if group is not None
                        else str(store)
                    },
                    require_catalog=True,
                    packs={manifest.name: info},
                    worker_group=group[0] if group is not None else None,
                    group_manifests=(
                        tuple(
                            self._store_dir(entries[member]) / MANIFEST_FILENAME
                            for member in group[1]
                        )
                        if group is not None
                        else ()
                    ),
                    in_process=entry.pack in in_process,
                    runtime_pins=dict(topology.runtime_pins),
                )
            )
        return specs


__all__ = [
    "LOCAL_PUBLISHER",
    "LOCAL_VERSION",
    "Freezer",
    "GroupProvisioner",
    "Installer",
    "PinHasher",
    "Provisioner",
    "RegistryFetcher",
    "VenvSpec",
    "http_registry_fetcher",
    "lock_git_pack",
    "lock_local_pack",
]
