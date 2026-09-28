"""Pack venv provisioning: a manifest becomes a private interpreter.

``ensure_pack_venv`` creates (or reuses) a venv for a pack with ``uv`` and
returns its python executable, ready to hand to IsolatedWorker. The venv
gets the pack itself (editable, if it has a pyproject), the manifest's
``requires``, and the dinkster packages the worker host imports.

Dev-workspace note: pass ``workspace_packages`` pointing at the local
``packages/dinkster-*`` directories so the child imports the same code the
parent runs. In a released world these come from the index like anything
else and ``workspace_packages`` stays empty.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from packaging.requirements import Requirement
from packaging.utils import NormalizedName, canonicalize_name

from .interpreter import InterpreterPreflightError, preflight_interpreter
from .manifest import PackManifest

_HOST_REQUIREMENTS = (
    "dinkster-workers",
    # The execution-boundary contracts (Invocation/Worker) - the child
    # never imports the engine/scheduler itself (hazard H3 payoff).
    "dinkster-protocol",
    "dinkster-schema",
    "dinkster-values",
    # The versioned extension door (DESIGN 3.6): pack code imports
    # dinkster_api.v1, so every pack venv must carry it.
    "dinkster-api",
)
"""What the worker host imports; installed from ``workspace_packages`` when
given, otherwise expected to resolve from the index."""


class ProvisionError(Exception):
    """Creating or populating a pack venv failed."""


def _preflight_venv(python: Path) -> Path:
    try:
        preflight_interpreter(python)
    except InterpreterPreflightError as exc:
        raise ProvisionError(f"pack venv interpreter refused: {exc}") from exc
    return python


def _run(command: Sequence[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip()[-2000:]
        raise ProvisionError(f"command failed: {' '.join(command)}\n{tail}")


def ensure_pack_venv(
    manifest: PackManifest,
    *,
    venv_root: Path,
    workspace_packages: Sequence[Path] = (),
    uv: str = "uv",
    reinstall: bool = False,
    pinned: Sequence[str] | None = None,
    constraints: Sequence[str] = (),
    accelerator: str | None = None,
) -> Path:
    """Create or reuse ``venv_root/<pack name>`` and return its python.

    ``pinned`` replaces range resolution with an exact dist list (a
    snapshot's per-venv freeze): the manifest's ``requires`` ranges and
    the index host requirements are skipped and the pins installed
    verbatim, so restoring a snapshot re-creates the resolution that was
    captured, not whatever the index resolves today. Editable installs
    (workspace packages, the pack itself) still apply - freeze capture
    excludes them, so they never collide with the pins. Pins may carry
    ``--hash=<algo>:<digest>`` annotations (:func:`hash_pins`): those
    ride in a requirements file (per-requirement options cannot be CLI
    arguments) and ``uv``'s default verify-hashes mode checks every
    provided hash against the downloaded artifact while the editables
    (never hashable) install in the same command.

    ``constraints`` (ignored when ``pinned`` is given - exactness already
    won) bind versions WITHOUT adding packages: range resolution still
    decides what to install, and any resolved dist that appears in the
    constraints gets that exact version. This is how a cross-scope
    snapshot restore reuses its portable pins - a constraint for a dist
    the destination's resolution never pulls (a vendor runtime wheel from
    the source accelerator, say) is inert instead of poison.

    ``accelerator`` selects the pack's ``[pack.extra-requires]`` list to
    install next to its base ``requires`` (None installs base only).
    PEP 508 OS/python markers inside any requirement string stay
    delegated to ``uv`` - accelerator is the one conditional dimension
    markers cannot express, so it is the only one resolved here."""
    requirements = (
        manifest.requires_for(accelerator) if accelerator is not None else manifest.requires
    )
    workspace_packages = _without_duplicate_editables(workspace_packages, (manifest.root,))
    return _ensure_venv(
        venv_root / manifest.name,
        requirements=requirements,
        editable_roots=(manifest.root,),
        workspace_packages=workspace_packages,
        uv=uv,
        reinstall=reinstall,
        pinned=pinned,
        constraints=constraints,
        accelerator=accelerator,
    )


def ensure_group_venv(
    manifests: Sequence[PackManifest],
    group_name: str,
    *,
    venv_root: Path,
    workspace_packages: Sequence[Path] = (),
    uv: str = "uv",
    reinstall: bool = False,
    pinned: Sequence[str] | None = None,
    constraints: Sequence[str] = (),
    accelerator: str | None = None,
) -> Path:
    """Create one venv for ``manifests`` and return its interpreter.

    Every member's selected requirements participate in one ``uv pip
    install`` resolution, and every member root with a pyproject is
    editable-installed into that interpreter. The caller owns the
    content-addressed aggregate root; this function only names the final
    venv after the declared group."""
    requirements = tuple(
        sorted(
            {
                requirement
                for manifest in manifests
                for requirement in (
                    manifest.requires_for(accelerator)
                    if accelerator is not None
                    else manifest.requires
                )
            }
        )
    )
    editable_roots = tuple(manifest.root for manifest in manifests)
    workspace_packages = _without_duplicate_editables(workspace_packages, editable_roots)
    venv_dir = venv_root / group_name
    return _ensure_venv(
        venv_dir,
        requirements=requirements,
        editable_roots=editable_roots,
        workspace_packages=workspace_packages,
        uv=uv,
        reinstall=reinstall,
        pinned=pinned,
        constraints=constraints,
        accelerator=accelerator,
    )


def _project_inputs(root: Path) -> dict[str, object]:
    path = root / "pyproject.toml"
    if not path.is_file():
        return {"root": str(root.resolve()), "name": None, "dependencies": []}
    try:
        project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
        return {
            "root": str(root.resolve()),
            "name": project["name"],
            "dependencies": sorted(project.get("dependencies", ())),
        }
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise ProvisionError(f"cannot read provisioning inputs from {path}: {exc}") from exc


def workspace_packages_for(
    pack_dir: Path,
    *,
    host_requirements: Sequence[str] = _HOST_REQUIREMENTS,
) -> tuple[Path, ...]:
    """Find the local workspace members needed to provision ``pack_dir``."""
    pack_dir = pack_dir.resolve()
    workspace_root = None
    workspace: dict[str, object] | None = None
    for candidate in (pack_dir, *pack_dir.parents):
        project_path = candidate / "pyproject.toml"
        if not project_path.is_file():
            continue
        try:
            document = cast(
                "dict[str, object]",
                tomllib.loads(project_path.read_text(encoding="utf-8")),
            )
        except (OSError, tomllib.TOMLDecodeError):
            continue
        tool = document.get("tool")
        tool_table = cast("dict[str, object]", tool) if isinstance(tool, dict) else {}
        uv = tool_table.get("uv")
        uv_table = cast("dict[str, object]", uv) if isinstance(uv, dict) else {}
        candidate_workspace = uv_table.get("workspace")
        if isinstance(candidate_workspace, dict):
            workspace_root = candidate
            workspace = cast("dict[str, object]", candidate_workspace)
            break
    if workspace_root is None or workspace is None:
        return ()

    members = workspace.get("members", [])
    excludes = workspace.get("exclude", [])
    if not isinstance(members, list) or not all(
        isinstance(item, str) for item in cast("list[object]", members)
    ):
        return ()
    if not isinstance(excludes, list) or not all(
        isinstance(item, str) for item in cast("list[object]", excludes)
    ):
        return ()
    excluded = {
        path.resolve()
        for pattern in cast("list[str]", excludes)
        for path in workspace_root.glob(pattern)
    }
    candidates = {workspace_root.resolve()}
    candidates.update(
        path.resolve()
        for pattern in cast("list[str]", members)
        for path in workspace_root.glob(pattern)
        if path.is_dir()
    )

    projects: dict[NormalizedName, tuple[Path, tuple[NormalizedName, ...]]] = {}
    for root in sorted(candidates - excluded):
        project_path = root / "pyproject.toml"
        if not project_path.is_file():
            continue
        try:
            document = cast(
                "dict[str, object]",
                tomllib.loads(project_path.read_text(encoding="utf-8")),
            )
        except (OSError, tomllib.TOMLDecodeError):
            continue
        project = document.get("project")
        if not isinstance(project, dict):
            continue
        project_table = cast("dict[str, object]", project)
        if not isinstance(project_table.get("name"), str):
            continue
        dependencies = project_table.get("dependencies", [])
        if not isinstance(dependencies, list) or not all(
            isinstance(item, str) for item in cast("list[object]", dependencies)
        ):
            continue
        projects[canonicalize_name(cast(str, project_table["name"]))] = (
            root,
            tuple(
                canonicalize_name(requirement.name)
                for item in cast("list[str]", dependencies)
                if (requirement := Requirement(item)).marker is None
                or requirement.marker.evaluate()
            ),
        )

    pack_project = next((project for project in projects.values() if project[0] == pack_dir), None)
    if pack_project is None:
        return ()
    pending = {canonicalize_name(requirement) for requirement in host_requirements}
    pending.update(pack_project[1])
    selected: set[Path] = set()
    visited: set[NormalizedName] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        project = projects.get(name)
        if project is None:
            continue
        root, dependencies = project
        if root != pack_dir:
            selected.add(root)
        pending.update(dependencies)
    return tuple(sorted(selected))


def _without_duplicate_editables(
    workspace_packages: Sequence[Path], editable_roots: Sequence[Path]
) -> tuple[Path, ...]:
    editable_names = {_project_name(root) for root in editable_roots}
    return tuple(root for root in workspace_packages if _project_name(root) not in editable_names)


def _project_name(root: Path) -> NormalizedName | None:
    name = _project_inputs(root)["name"]
    return canonicalize_name(name) if isinstance(name, str) else None


def _missing_host_requirements(workspace_packages: Sequence[Path]) -> tuple[str, ...]:
    provided = {_project_name(root) for root in workspace_packages}
    return tuple(
        requirement
        for requirement in _HOST_REQUIREMENTS
        if canonicalize_name(requirement) not in provided
    )


def _provisioning_digest(
    *,
    requirements: Sequence[str],
    editable_roots: Sequence[Path],
    workspace_packages: Sequence[Path],
    pinned: Sequence[str] | None,
    constraints: Sequence[str],
    accelerator: str | None,
) -> str:
    payload = {
        "requirements": sorted(requirements),
        "editableRoots": sorted(
            (_project_inputs(root) for root in editable_roots),
            key=lambda item: str(item["root"]),
        ),
        "workspacePackages": sorted(
            (_project_inputs(root) for root in workspace_packages),
            key=lambda item: str(item["root"]),
        ),
        "pinned": None if pinned is None else sorted(pinned),
        "constraints": [] if pinned is not None else sorted(constraints),
        "accelerator": accelerator,
    }
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(wire).hexdigest()


def _ensure_venv(
    venv_dir: Path,
    *,
    requirements: Sequence[str],
    editable_roots: Sequence[Path],
    workspace_packages: Sequence[Path],
    uv: str,
    reinstall: bool,
    pinned: Sequence[str] | None,
    constraints: Sequence[str],
    accelerator: str | None,
) -> Path:
    """Shared create/install machinery for one-pack and grouped venvs."""
    python = venv_dir / "Scripts" / "python.exe" if os.name == "nt" else venv_dir / "bin" / "python"
    complete = venv_dir / ".dinkster-complete"
    host_requirements = () if pinned is not None else _missing_host_requirements(workspace_packages)
    digest = _provisioning_digest(
        requirements=(() if pinned is not None else (*host_requirements, *requirements)),
        editable_roots=editable_roots,
        workspace_packages=workspace_packages,
        pinned=pinned,
        constraints=constraints,
        accelerator=accelerator,
    )
    if python.exists() and complete.is_file() and not reinstall:
        try:
            if complete.read_text(encoding="utf-8").strip() == digest:
                return _preflight_venv(python)
        except OSError:
            pass

    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    try:
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        _run([uv, "venv", "--python", sys.executable, str(venv_dir)])

        install: list[str] = [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-sources",
        ]
        if workspace_packages:
            for package_dir in workspace_packages:
                install.extend(["-e", str(package_dir)])
        install.extend(host_requirements)
        for root in editable_roots:
            if (root / "pyproject.toml").exists():
                install.extend(["-e", str(root)])
        constraints_file: Path | None = None
        pins_file: Path | None = None
        if pinned is not None:
            if any(" " in pin for pin in pinned):
                # Hash-annotated pins are per-requirement OPTIONS, which only
                # a requirements file can carry - as CLI arguments uv would
                # parse "--hash=..." as its own flag.
                handle, name = tempfile.mkstemp(prefix="dinkster-pins-", suffix=".txt")
                pins_file = Path(name)
                with os.fdopen(handle, "w") as stream:
                    stream.write("\n".join(pinned) + "\n")
                install.extend(["-r", str(pins_file)])
            else:
                install.extend(pinned)
        else:
            install.extend(requirements)
            if constraints:
                handle, name = tempfile.mkstemp(prefix="dinkster-constraints-", suffix=".txt")
                constraints_file = Path(name)
                with os.fdopen(handle, "w") as stream:
                    stream.write("\n".join(constraints) + "\n")
                install.extend(["--constraint", str(constraints_file)])
        try:
            _run(install)
        finally:
            if constraints_file is not None:
                constraints_file.unlink(missing_ok=True)
            if pins_file is not None:
                pins_file.unlink(missing_ok=True)
        _preflight_venv(python)
        complete.parent.mkdir(parents=True, exist_ok=True)
        complete.write_text(digest + "\n", encoding="utf-8")
        return python
    except BaseException:
        shutil.rmtree(venv_dir, ignore_errors=True)
        raise


def bare_pin(pin: str) -> str:
    """The ``name==version`` half of a possibly hash-annotated pin. Hash
    annotations are artifact identity, not version identity - anything
    comparing against a freeze (which reports versions only) or building
    constraints (which bind versions only) compares/carries bare pins."""
    return pin.split()[0]


def partition_portable(pins: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split freeze pins into (portable, environment-specific) by the one
    signal that is data in the pin itself: a PEP 440 local version label.
    ``torch==2.5.1+cu124`` names a vendor-specific build that only exists
    on that scope's index - reusing it on another platform/accelerator
    installs bytes that cannot work (or nothing at all). A label-free
    ``pillow==10.4.0`` names a version the index serves for every
    platform it supports, so the VERSION travels even though the wheel
    differs. Never a name heuristic: no package list, no regex on names -
    a pin is environment-specific exactly when its version says so."""
    portable: list[str] = []
    environment_specific: list[str] = []
    for pin in pins:
        _, _, version = bare_pin(pin).partition("==")
        (environment_specific if "+" in version else portable).append(pin)
    return tuple(portable), tuple(environment_specific)


def hash_pins(pins: Sequence[str], *, uv: str = "uv") -> tuple[str, ...]:
    """Annotate portable pins with their artifact hashes: each becomes
    ``name==version --hash=<algo>:<digest> ...`` carrying EVERY artifact
    the index serves for that exact version (sdist + all wheels, via
    ``uv pip compile --generate-hashes``), so an annotated pin verifies
    on any platform the version supports, not just the capture host.

    Environment-specific pins (PEP 440 local version labels, e.g.
    ``+cu124``) stay bare, honestly: they name vendor-index builds the
    configured index cannot vouch for, and provisioning's verify-hashes
    mode checks provided hashes without requiring absent ones. A network
    operation (the index is queried); failure is loud, never a silent
    bare fallback - the caller asked for verifiable pins."""
    portable, environment_specific = partition_portable(tuple(bare_pin(pin) for pin in pins))
    if not portable:
        return tuple(sorted(environment_specific))
    handle, name = tempfile.mkstemp(prefix="dinkster-hash-pins-", suffix=".in")
    reqs_in = Path(name)
    out_handle, out_name = tempfile.mkstemp(prefix="dinkster-hash-pins-", suffix=".txt")
    os.close(out_handle)
    reqs_out = Path(out_name)
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write("\n".join(portable) + "\n")
        # --no-deps: the input is already a complete closure (a freeze);
        # compile only looks up each exact version's artifacts.
        _run(
            [
                uv,
                "pip",
                "compile",
                "--generate-hashes",
                "--no-deps",
                "--no-header",
                "--no-annotate",
                "-o",
                str(reqs_out),
                str(reqs_in),
            ]
        )
        annotated = _validated_compiled_pins(
            _logical_requirement_lines(reqs_out.read_text()), portable
        )
    finally:
        reqs_in.unlink(missing_ok=True)
        reqs_out.unlink(missing_ok=True)
    return tuple(sorted(annotated + list(environment_specific)))


# The snapshot pin grammar (mirrors dinkster_registry.install's decoder): an
# exact 'name==version' first token, then only '--hash=<algo>:<hex>'
# annotations. Compiled output is VALIDATED against it before it becomes
# a snapshot pin - a compile that legally succeeds but emits option
# lines, URLs, markers, or unhashed requirements (index configuration
# can cause all of these) must fail loudly here, not surface later as a
# refused or silently-weaker record.
_EXACT_PIN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    r"==(?:[0-9]+!)?[0-9A-Za-z]+(?:[.+][0-9A-Za-z]+)*"
)

_HASH_TOKEN = re.compile(r"--hash=[A-Za-z0-9_]+:[0-9a-fA-F]+")


def _canonical_identity(pin: str) -> tuple[str, str]:
    """(canonical name, version) of an exact pin - PEP 503 name folding,
    so the compiler respelling 'Pillow' as 'pillow' still matches."""
    name, _, version = pin.partition("==")
    return re.sub(r"[-_.]+", "-", name).lower(), version


def _validated_compiled_pins(lines: Sequence[str], portable: Sequence[str]) -> list[str]:
    """The compiled lines, proven to be exactly the portable input set -
    same (name, version) identities, once each, every line carrying at
    least one well-formed hash and nothing else. Any discrepancy is a
    ProvisionError: the caller asked for verifiable pins, so an
    incomplete or adulterated compile can never become a snapshot."""
    expected = {_canonical_identity(pin) for pin in portable}
    validated: dict[tuple[str, str], str] = {}
    for line in lines:
        tokens = line.split()
        if _EXACT_PIN.fullmatch(tokens[0]) is None:
            raise ProvisionError(
                f"hash compile emitted a non-pin line {line!r}; refusing to snapshot it"
            )
        annotations = tokens[1:]
        if not annotations or any(_HASH_TOKEN.fullmatch(token) is None for token in annotations):
            raise ProvisionError(
                f"hash compile emitted {line!r} without a complete set of "
                f"'--hash=<algo>:<hex>' annotations; refusing to snapshot an "
                f"unverifiable pin"
            )
        identity = _canonical_identity(tokens[0])
        if identity in validated:
            raise ProvisionError(f"hash compile emitted {tokens[0]!r} more than once")
        if identity not in expected:
            raise ProvisionError(
                f"hash compile emitted {tokens[0]!r}, which is not one of the pins being hashed"
            )
        validated[identity] = line
    missing = expected - set(validated)
    if missing:
        names = ", ".join(sorted(f"{name}=={version}" for name, version in missing))
        raise ProvisionError(f"hash compile omitted pins: {names}")
    return list(validated.values())


def _logical_requirement_lines(text: str) -> list[str]:
    """Compiled requirements as one-string-per-requirement: backslash
    continuations joined, comments/blanks dropped, whitespace collapsed -
    the pin form the snapshot model stores and validates."""
    logical: list[str] = []
    pending = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        pending += line
        logical.append(" ".join(pending.split()))
        pending = ""
    if pending:
        logical.append(" ".join(pending.split()))
    return logical


def freeze_venv(python: Path, *, uv: str = "uv") -> tuple[str, ...]:
    """The venv's exact installed dists as sorted ``name==version`` pins -
    what a snapshot records per pack venv. Editable and direct-URL
    installs (workspace packages, the pack itself) are excluded: they are
    re-established by normal provisioning on restore and their local
    paths would be meaningless on another machine."""
    result = subprocess.run(
        [uv, "pip", "freeze", "--python", str(python)], capture_output=True, text=True
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip()[-2000:]
        raise ProvisionError(f"freeze failed for {python}\n{tail}")
    pins = [
        line.strip()
        for line in result.stdout.splitlines()
        if "==" in line and not line.startswith("-e") and " @ " not in line
    ]
    return tuple(sorted(pins))
