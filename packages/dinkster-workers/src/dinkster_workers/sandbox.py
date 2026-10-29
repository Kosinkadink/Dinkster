"""Linux bubblewrap sandboxing: the hardened launcher rung (DESIGN 3.11).

This module turns a LaunchSpec plus a SandboxPolicy into a bwrap argv, and
refuses loudly when the sandbox cannot actually be built. A requested
sandbox never silently degrades to a plain subprocess: if bwrap is missing,
or the kernel blocks unprivileged user namespaces entirely (Ubuntu 24.04
ships that default via AppArmor - even bwrap's degraded mode needs to
create a user namespace when the binary is not setuid), ``launch`` raises
SandboxUnavailable carrying the distro-specific remediation instead of
running the child unjailed.

The construction adopts pyisolate's mature ``_internal/sandbox.py``:

- deny-by-default filesystem - an explicit read-only allow-list of system
  paths, nothing else visible;
- private tmpfs ``/tmp``, fresh ``/proc`` and ``/dev``;
- an unshared network namespace with optional host-proxied HTTPS egress;
- GPU device and CUDA binds only when the policy grants them;
- ``--clearenv`` plus an explicit environment allowlist delivered through
  a private one-use file descriptor, never argv;
- capability detection with a probe ladder: full user+pid namespaces,
  degraded mount-namespace-only, unavailable-with-remediation.

Policy is host/operator configuration. Pack authors never see any of this
(hazard H9); the engine cannot tell a jailed worker from a plain one (H3).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from dinkster_schema import LOG_LEVEL_ENV, LOG_OVERRIDES_ENV
from dinkster_values import GIBIBYTE, MEBIBYTE

from .accelerator import ACCELERATOR_ENV
from .egress import EGRESS_PROXY_ENV, EgressProxy
from .launch import HOST_OWNED_ENVIRONMENT, LaunchSpec


class SandboxError(Exception):
    """A sandbox policy or construction is invalid."""


class SandboxUnavailable(SandboxError):
    """A sandbox was requested but cannot be built on this machine."""


#: The ONLY system paths a sandboxed child can read. Everything else is
#: denied. Security-critical list - additions weaken the jail.
SANDBOX_SYSTEM_PATHS: tuple[str, ...] = (
    "/usr",
    "/lib",
    "/lib64",
    "/lib32",
    "/bin",
    "/sbin",
    "/opt",
    "/etc/alternatives",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/ssl",
    "/etc/ca-certificates",
    "/etc/pki",
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
    "/etc/passwd",
    "/etc/group",
    "/etc/localtime",
    "/etc/timezone",
)

#: /dev entries bound in (device nodes, not files) when the policy grants GPU.
GPU_DEVICE_PATTERNS: tuple[str, ...] = (
    "nvidia*",
    "nvidiactl",
    "nvidia-uvm",
    "nvidia-uvm-tools",
    "dri",
)

#: Roots that may never be bound wholesale - granting any of these would
#: dissolve the jail. Specific subpaths (a model folder under /home, a repo
#: checkout) are fine; the *root* is not.
FORBIDDEN_BIND_ROOTS: frozenset[str] = frozenset(
    {"/", "/etc", "/root", "/home", "/var", "/run", "/proc", "/sys", "/dev", "/tmp"}
)

_BASE_ENV_PASSTHROUGH: tuple[str, ...] = (
    "PATH",
    "LANG",
    "LC_ALL",
    # The accelerator pin must reach sandboxed workers: explicit selection
    # is authoritative for load-device choice, and a stripped pin would
    # silently fall back to auto detection inside the jail.
    ACCELERATOR_ENV,
    LOG_LEVEL_ENV,
    LOG_OVERRIDES_ENV,
)
_GPU_ENV_PASSTHROUGH: tuple[str, ...] = (
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_CUDA_ARCH_LIST",
)

DEFAULT_WORKER_FSIZE_LIMIT_BYTES = 64 * GIBIBYTE
DEFAULT_WORKER_PROCESS_LIMIT = 4096
_WORKER_ENV_PATH = "/run/dinkster/worker-env.json"
_PROBE_ENV_PATH = "/run/dinkster/probe-env.json"


@dataclass(frozen=True)
class SandboxPolicy:
    """What the operator grants a jailed pack. Deny-by-default: no network,
    no GPU, no filesystem beyond what the launch itself requires (system
    libraries, the child's interpreter, the pack root, the endpoint).

    ``ro_binds``/``rw_binds`` are absolute paths granted read-only/writable -
    model folders, an asset vault, a pack's own writable scratch. Forbidden
    roots (``/``, ``/home``, ...) are rejected at build time, not skipped:
    a policy that would dissolve the jail is an error, not a warning.

    ``env_passthrough`` names additional parent environment variables to
    forward; everything else is cleared.

    ``max_file_size_bytes`` and ``max_processes`` are finite by default.
    Memory stays under the memory governor rather than this launcher policy.
    """

    egress_allowlist: tuple[str, ...] = ()
    gpu: bool = False
    ro_binds: tuple[str, ...] = ()
    rw_binds: tuple[str, ...] = ()
    env_passthrough: tuple[str, ...] = ()
    max_file_size_bytes: int | None = DEFAULT_WORKER_FSIZE_LIMIT_BYTES
    max_processes: int | None = DEFAULT_WORKER_PROCESS_LIMIT
    protected_roots: tuple[str, ...] = ()
    protected_ro_exceptions: tuple[str, ...] = ()
    protected_rw_exceptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class BubblewrapCapability:
    """What the probe ladder found. ``user_namespaces`` distinguishes the
    full jail (user+pid namespaces) from degraded mount-namespace-only mode
    (setuid bwrap on a userns-restricted kernel)."""

    available: bool
    bwrap: str | None
    user_namespaces: bool
    detail: str


def _read_sysctl(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def _probe(bwrap: str, *namespace_args: str) -> tuple[bool, str]:
    command = [bwrap, *namespace_args, "--dev", "/dev", "--proc", "/proc"]
    for path in ("/usr", "/bin", "/lib", "/lib64"):
        if os.path.exists(path):
            command.extend(["--ro-bind", path, path])
    command.append("/usr/bin/true")
    try:
        result = subprocess.run(command, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.decode("utf-8", errors="replace").strip()


def _remediation(error: str) -> str:
    if _read_sysctl("/proc/sys/user/max_user_namespaces") == 0:
        return (
            "user namespaces are disabled (user.max_user_namespaces=0); fix: "
            "echo 'user.max_user_namespaces=15000' | "
            "sudo tee /etc/sysctl.d/99-userns.conf && sudo sysctl --system"
        )
    if _read_sysctl("/proc/sys/kernel/apparmor_restrict_unprivileged_userns") == 1:
        return (
            "AppArmor restricts unprivileged user namespaces (Ubuntu 24.04+ "
            "default); fix: sudo apt install apparmor-profiles && sudo ln -s "
            "/usr/share/apparmor/extra-profiles/bwrap-userns-restrict "
            "/etc/apparmor.d/bwrap && sudo apparmor_parser -r /etc/apparmor.d/bwrap"
        )
    return f"bwrap probe failed: {error[:300]}"


def detect_bubblewrap() -> BubblewrapCapability:
    """Probe whether a bubblewrap jail can actually be built here.

    The ladder: full isolation (user+pid namespaces), then degraded
    mount-namespace-only (possible with a setuid bwrap when unprivileged
    userns is blocked), then unavailable with a distro-specific remediation.
    Runs two short subprocess probes at worst; call it once and keep the
    result (BubblewrapLauncher does).
    """
    if sys.platform != "linux":
        return BubblewrapCapability(
            False, None, False, f"bubblewrap requires Linux (platform: {sys.platform})"
        )
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return BubblewrapCapability(
            False,
            None,
            False,
            "bwrap not found; install bubblewrap (apt/dnf install bubblewrap)",
        )
    ok, error = _probe(bwrap, "--unshare-user", "--unshare-pid")
    if ok:
        return BubblewrapCapability(True, bwrap, True, "")
    degraded_ok, degraded_error = _probe(bwrap)
    if degraded_ok:
        return BubblewrapCapability(
            True,
            bwrap,
            False,
            "degraded: mount-namespace isolation only (unprivileged user "
            "namespaces are blocked); " + _remediation(error),
        )
    detail = _remediation(error if error else degraded_error)
    return BubblewrapCapability(False, bwrap, False, detail)


def _existing(paths: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for path in paths:
        if path not in seen and os.path.exists(path):
            seen.append(path)
    return seen


def _covered_by_system(path: str) -> bool:
    return any(path == root or path.startswith(root + "/") for root in SANDBOX_SYSTEM_PATHS)


def _next_symlink_target(path: Path) -> Path | None:
    """Resolve the first symlink component without hiding later link hops."""
    if sys.platform == "win32":
        return None
    parts = path.parts
    current = Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        current /= part
        try:
            target = Path(os.readlink(current))
        except OSError:
            continue
        if not target.is_absolute():
            target = current.parent / target
        return Path(os.path.normpath(target.joinpath(*parts[index + 1 :])))
    return None


def _absolute_executable(python: str) -> str:
    return os.path.abspath(shutil.which(python) or python)


def _interpreter_binds(python: str) -> list[str]:
    """Paths the child interpreter needs beyond the system allow-list: its
    venv, every installation prefix in its symlink chain (uv-managed
    interpreters use an absolute executable link through a version alias),
    and the parent's base prefix for the same-interpreter case."""
    binds: list[str] = []
    python_path = Path(_absolute_executable(python))
    seen: set[str] = set()
    for _ in range(40):
        key = str(python_path)
        if key in seen:
            break
        seen.add(key)
        binds.append(str(python_path.parent.parent))
        target = _next_symlink_target(python_path)
        if target is None:
            break
        python_path = target
    binds.append(sys.base_prefix)
    return [
        path
        for path in _existing(binds)
        if path not in FORBIDDEN_BIND_ROOTS and not _covered_by_system(path)
    ]


def _forbidden_root_targets() -> frozenset[str]:
    """The forbidden roots by every name they answer to: the literal paths
    plus their symlink-resolved forms. On macOS the roots themselves resolve
    elsewhere (/home to /System/Volumes/Data/home, /tmp to /private/tmp), so
    a bind that resolves to such a target dissolves the jail just the same."""
    return FORBIDDEN_BIND_ROOTS | {os.path.realpath(root) for root in FORBIDDEN_BIND_ROOTS}


def _validated(paths: tuple[str, ...], *, role: str) -> list[str]:
    forbidden = _forbidden_root_targets()
    checked: list[str] = []
    for path in paths:
        normalized = os.path.abspath(path)
        resolved = os.path.realpath(normalized)
        if normalized in forbidden or resolved in forbidden:
            raise SandboxError(
                f"{role} bind '{path}' would dissolve the sandbox; grant a specific subpath instead"
            )
        checked.append(normalized)
    return _existing(checked)


def _resolved(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _paths_overlap(left: str, right: str) -> bool:
    left_path = _resolved(left)
    right_path = _resolved(right)
    return (
        left_path == right_path
        or left_path.startswith(right_path + os.sep)
        or right_path.startswith(left_path + os.sep)
    )


def _inside(path: str, root: str) -> bool:
    resolved_path = _resolved(path)
    resolved_root = _resolved(root)
    return resolved_path == resolved_root or resolved_path.startswith(resolved_root + os.sep)


def _strictly_inside(path: str, root: str) -> bool:
    return _resolved(path).startswith(_resolved(root) + os.sep)


def _validate_protected_binds(
    paths: Iterable[str],
    *,
    protected_roots: tuple[str, ...],
    exceptions: tuple[str, ...],
    role: str,
) -> None:
    for path in paths:
        for protected in protected_roots:
            if not _paths_overlap(path, protected):
                continue
            if any(
                _strictly_inside(exception, protected) and _inside(path, exception)
                for exception in exceptions
            ):
                continue
            raise SandboxError(f"{role} bind '{path}' overlaps protected root '{protected}'")


def _launch_environment(spec: LaunchSpec, policy: SandboxPolicy) -> dict[str, str]:
    passthrough = [*_BASE_ENV_PASSTHROUGH, *policy.env_passthrough]
    if policy.gpu:
        passthrough.extend(_GPU_ENV_PASSTHROUGH)
    environment = {
        name: os.environ[name]
        for name in passthrough
        if name not in HOST_OWNED_ENVIRONMENT and name in os.environ
    }
    environment.update(
        {
            "HOME": "/tmp",
            "TMPDIR": "/tmp",
            "PYTHONNOUSERSITE": "1",
            **spec.env,
        }
    )
    return environment


def _worker_command(spec: LaunchSpec, policy: SandboxPolicy) -> list[str]:
    limits = (
        ("resource.RLIMIT_FSIZE", policy.max_file_size_bytes, "max_file_size_bytes"),
        ("resource.RLIMIT_NPROC", policy.max_processes, "max_processes"),
    )
    statements = [
        "import json, os, resource, sys",
        f"_env_file = open({_WORKER_ENV_PATH!r}, encoding='utf-8')",
        "os.environ.update(json.load(_env_file))",
        "_env_file.close()",
        f"os.unlink({_WORKER_ENV_PATH!r})",
    ]
    for resource_name, value, field_name in limits:
        if value is None:
            continue
        if type(value) is not int or value <= 0:
            raise SandboxError(f"{field_name} must be a positive integer or None")
        statements.extend(
            (
                f"_soft, _hard = resource.getrlimit({resource_name})",
                f"_limit = {value}",
                "_limit = min(_limit, _soft) if _soft != resource.RLIM_INFINITY else _limit",
                "_limit = min(_limit, _hard) if _hard != resource.RLIM_INFINITY else _limit",
                f"resource.setrlimit({resource_name}, (_limit, _limit))",
            )
        )
    statements.append("os.execv(sys.executable, [sys.executable] + sys.argv[1:])")
    return [_absolute_executable(spec.python), "-c", "; ".join(statements), *spec.command[1:]]


def build_bwrap_command(
    spec: LaunchSpec,
    policy: SandboxPolicy,
    *,
    bwrap: str = "bwrap",
    user_namespaces: bool = True,
    env_fd: int = 3,
) -> list[str]:
    """Pure argv construction - everything a jail grants, inspectable and
    unit-testable without spawning anything.

    Mount order matters to bwrap (later binds punch holes in earlier ones):
    system allow-list first, launch necessities next, policy grants last.
    Environment starts empty. A private inherited file descriptor supplies a
    one-use JSON file that the pre-import worker wrapper reads and unlinks;
    environment values never appear in bwrap's process arguments.
    """
    command: list[str] = [bwrap, "--die-with-parent", "--new-session"]
    if user_namespaces:
        command.extend(["--unshare-user", "--unshare-pid"])
    command.append("--unshare-net")
    command.extend(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"])

    system_binds = _existing(SANDBOX_SYSTEM_PATHS)
    interpreter_binds = _validated(tuple(_interpreter_binds(spec.python)), role="interpreter")
    pack_root = _validated((str(spec.pack_root),), role="pack root")[0]
    endpoint_bind = _validated((str(spec.endpoint_dir),), role="endpoint")[0]
    pythonpath = spec.env.get("PYTHONPATH", "")
    if not pythonpath and "PYTHONPATH" in policy.env_passthrough:
        pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_entries = tuple(p for p in pythonpath.split(os.pathsep) if p)
    pythonpath_binds = _validated(pythonpath_entries, role="PYTHONPATH")
    policy_ro_binds = _validated(policy.ro_binds, role="read-only")
    policy_rw_binds = _validated(policy.rw_binds, role="writable")
    gpu_ro_binds: list[str] = []
    gpu_device_binds: list[str] = []
    if policy.gpu:
        gpu_ro_binds.append("/sys")
        gpu_device_binds.extend(
            dict.fromkeys(
                str(device)
                for pattern in GPU_DEVICE_PATTERNS
                for device in sorted(Path("/dev").glob(pattern))
            )
        )
        cuda_home = os.environ.get("CUDA_HOME")
        cuda_binds = tuple(
            path
            for path in _existing(p for p in ("/opt/cuda", cuda_home) if p)
            if not _covered_by_system(path)
        )
        gpu_ro_binds.extend(_validated(cuda_binds, role="CUDA"))
    implicit_rw_binds = [endpoint_bind, *gpu_device_binds]
    if spec.use_shm and os.path.exists("/dev/shm"):
        implicit_rw_binds.append("/dev/shm")
    read_only_binds = [
        *system_binds,
        *interpreter_binds,
        pack_root,
        *pythonpath_binds,
        *gpu_ro_binds,
        *policy_ro_binds,
    ]
    _validate_protected_binds(
        read_only_binds,
        protected_roots=policy.protected_roots,
        exceptions=policy.protected_ro_exceptions,
        role="read-only",
    )
    _validate_protected_binds(
        policy_rw_binds,
        protected_roots=policy.protected_roots,
        exceptions=policy.protected_rw_exceptions,
        role="writable",
    )
    _validate_protected_binds(
        implicit_rw_binds,
        protected_roots=policy.protected_roots,
        exceptions=(),
        role="writable launch",
    )
    for writable in (*implicit_rw_binds, *policy_rw_binds):
        for read_only in read_only_binds:
            if _paths_overlap(writable, read_only):
                raise SandboxError(
                    f"writable bind '{writable}' overlaps read-only bind '{read_only}'"
                )

    for path in system_binds:
        command.extend(["--ro-bind", path, path])
    for path in interpreter_binds:
        command.extend(["--ro-bind", path, path])
    command.extend(["--ro-bind", pack_root, pack_root])
    for entry in pythonpath_binds:
        command.extend(["--ro-bind", entry, entry])

    for path in gpu_ro_binds:
        command.extend(["--ro-bind", path, path])
    for path in gpu_device_binds:
        command.extend(["--dev-bind", path, path])

    if "/dev/shm" in implicit_rw_binds:
        command.extend(["--bind", "/dev/shm", "/dev/shm"])

    # The endpoint directory (unix socket) must be writable inside the jail.
    # Its ancestors may sit under the private tmpfs /tmp; create the chain
    # so the bind point exists.
    endpoint_dir = Path(endpoint_bind)
    ancestor = Path("/")
    for part in endpoint_dir.parts[1:]:
        ancestor = ancestor / part
        command.extend(["--dir", str(ancestor)])
    command.extend(["--bind", str(endpoint_dir), str(endpoint_dir)])

    for path in policy_ro_binds:
        command.extend(["--ro-bind", path, path])
    for path in policy_rw_binds:
        command.extend(["--bind", path, path])

    command.append("--clearenv")
    command.extend(["--dir", "/run", "--dir", "/run/dinkster"])
    command.extend(["--perms", "0400", "--file", str(env_fd), _WORKER_ENV_PATH])

    command.extend(_worker_command(spec, policy))
    return command


#: rlimits applied inside the probe jail before running the probe module,
#: with no fork race and no thread-unsafe preexec_fn: per-process CPU
#: seconds (a busy-looping import dies even if wall-clock supervision is
#: lost) and max bytes any file write may produce (the probe's only
#: legitimate output is its stdout JSON report).
PROBE_CPU_LIMIT_S = 300
PROBE_FSIZE_LIMIT_BYTES = 64 * MEBIBYTE

#: python -c stub that installs the environment and rlimits, then runs the
#: probe in the same interpreter so Python startup cannot inject variables
#: after the explicit environment is installed. PYTHONPATH is applied to
#: sys.path explicitly because the interpreter consumed its empty startup
#: environment before the descriptor was available. sys.argv[1:] carries
#: the original interpreter arguments
#: (-m dinkster_workers._doctor_probe <manifest>).
_PROBE_RLIMIT_STUB = (
    "import json, os, resource, sys; "
    "from dinkster_workers import _doctor_probe as _probe; "
    f"_env_file = open({_PROBE_ENV_PATH!r}, encoding='utf-8'); "
    "_environment = json.load(_env_file); _env_file.close(); "
    f"os.unlink({_PROBE_ENV_PATH!r}); "
    "os.environ.clear(); os.environ.update(_environment); "
    "_pythonpath = _environment.get('PYTHONPATH', ''); "
    "sys.path[1:1] = [os.path.abspath(_path or os.curdir) "
    "for _path in _pythonpath.split(os.pathsep)] if _pythonpath else []; "
    f"resource.setrlimit(resource.RLIMIT_CPU, ({PROBE_CPU_LIMIT_S},) * 2); "
    f"resource.setrlimit(resource.RLIMIT_FSIZE, ({PROBE_FSIZE_LIMIT_BYTES},) * 2); "
    "sys.argv = sys.argv[2:]; _probe.main()"
)


def probe_environment(environment: Mapping[str, str] | None) -> dict[str, str]:
    if environment is not None:
        return dict(environment)
    return {
        **{name: os.environ[name] for name in _BASE_ENV_PASSTHROUGH if name in os.environ},
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PYTHONNOUSERSITE": "1",
    }


def _import_surface_binds(python: str) -> list[str]:
    """The read-only paths the probe child needs to import what its parent
    can import: the interpreter's own prefixes plus every existing absolute
    ``sys.path`` entry (editable installs land their source roots there via
    .pth files, so this covers dev workspaces and deployed site-packages
    with one rule). Nothing here is writable and nothing outside it is
    visible."""
    binds = _interpreter_binds(python)
    for entry in sys.path:
        if not entry or not os.path.isabs(entry):
            continue
        normalized = os.path.normpath(entry)
        if normalized in FORBIDDEN_BIND_ROOTS:
            continue
        if _covered_by_system(normalized):
            continue
        if any(normalized == bound or normalized.startswith(bound + os.sep) for bound in binds):
            continue
        if os.path.exists(normalized):
            binds.append(normalized)
    return binds


def build_probe_bwrap_command(
    python: str,
    manifest_root: str | Path,
    *,
    bwrap: str = "bwrap",
    user_namespaces: bool = True,
    env_fd: int = 3,
) -> list[str]:
    """Pure argv construction for the doctor's import-probe jail.

    Stricter than the worker jail because a probe needs strictly less: the
    network is ALWAYS unshared (no policy knob - an import that needs the
    network is a finding, not a grant), there is no GPU, no endpoint, no
    shm, and no writable bind anywhere - the only writable surface is the
    private tmpfs /tmp. Read-only: the system allow-list, the parent's
    import surface, and the pack bytes under probe.
    """
    command: list[str] = [bwrap, "--die-with-parent", "--new-session"]
    if user_namespaces:
        command.extend(["--unshare-user", "--unshare-pid"])
    command.append("--unshare-net")
    command.extend(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"])

    for path in _existing(SANDBOX_SYSTEM_PATHS):
        command.extend(["--ro-bind", path, path])
    for path in _import_surface_binds(python):
        command.extend(["--ro-bind", path, path])
    root = os.path.normpath(str(manifest_root))
    if root in FORBIDDEN_BIND_ROOTS:
        raise SandboxError(f"probe root '{manifest_root}' would dissolve the sandbox")
    command.extend(["--ro-bind", root, root])

    command.append("--clearenv")
    command.extend(["--dir", "/run", "--dir", "/run/dinkster"])
    command.extend(["--perms", "0400", "--file", str(env_fd), _PROBE_ENV_PATH])
    return command


@dataclass(frozen=True)
class ProbeJail:
    """A validated capability to jail the doctor's import probe.

    Construct via ``detect_probe_jail()`` so an unavailable sandbox refuses
    at detection time with the remediation - a ProbeJail in hand means the
    jail can actually be built. ``wrap`` turns the plain probe argv
    (``[python, -m, dinkster_workers._doctor_probe, manifest]``) into the
    jailed equivalent: bwrap, then a bootstrap that installs the explicit
    environment and applies CPU and file-size rlimits before importing the
    probe module.
    """

    bwrap: str
    user_namespaces: bool

    def wrap(self, argv: list[str], manifest_root: Path, *, env_fd: int = 3) -> list[str]:
        jailed = build_probe_bwrap_command(
            argv[0],
            manifest_root,
            bwrap=self.bwrap,
            user_namespaces=self.user_namespaces,
            env_fd=env_fd,
        )
        jailed.extend([argv[0], "-c", _PROBE_RLIMIT_STUB, *argv[1:]])
        return jailed


def detect_probe_jail() -> ProbeJail:
    """Detect bubblewrap and return a ProbeJail, or raise SandboxUnavailable
    with the distro-specific remediation. Never degrades: a caller that
    wants a jailed probe either gets one or a loud refusal."""
    if sys.platform != "linux":
        raise SandboxUnavailable(
            "probe sandboxing requires Linux bubblewrap; this platform has no jail rung"
        )
    capability = detect_bubblewrap()
    if not capability.available or capability.bwrap is None:
        raise SandboxUnavailable(f"bubblewrap sandbox unavailable: {capability.detail}")
    return ProbeJail(
        bwrap=capability.bwrap,
        user_namespaces=capability.user_namespaces,
    )


class BubblewrapLauncher:
    """Launcher that jails the child with bubblewrap - or refuses.

    Detection runs once, on first launch (or inject a ``capability`` to
    decide it earlier / elsewhere). An unavailable sandbox raises
    SandboxUnavailable with the remediation; it never falls back to a plain
    subprocess - the operator asked for a jail, and not getting one is a
    loud failure, not a downgrade.
    """

    def __init__(
        self,
        policy: SandboxPolicy | None = None,
        *,
        capability: BubblewrapCapability | None = None,
    ) -> None:
        self._policy = policy or SandboxPolicy()
        self._capability = capability
        self._cleanup_tasks: set[asyncio.Task[None]] = set()

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    async def _close_egress_after_exit(
        self,
        process: asyncio.subprocess.Process,
        proxy: EgressProxy,
    ) -> None:
        try:
            await process.wait()
        finally:
            await proxy.close()

    async def launch(self, spec: LaunchSpec) -> asyncio.subprocess.Process:
        capability = self._capability
        if capability is None:
            capability = self._capability = detect_bubblewrap()
        if not capability.available or capability.bwrap is None:
            raise SandboxUnavailable(f"bubblewrap sandbox unavailable: {capability.detail}")
        if not capability.user_namespaces:
            raise SandboxUnavailable(
                "pack sandbox requires bubblewrap user and PID namespaces; "
                "mount-only degraded mode is refused"
            )
        environment = _launch_environment(spec, self._policy)
        environment.pop(EGRESS_PROXY_ENV, None)
        proxy: EgressProxy | None = None
        if self._policy.egress_allowlist:
            proxy = await EgressProxy.start(
                spec.endpoint_dir / "egress.sock",
                self._policy.egress_allowlist,
            )
            environment[EGRESS_PROXY_ENV] = str(proxy.socket_path)
        try:
            with tempfile.TemporaryFile() as environment_file:
                environment_file.write(json.dumps(environment).encode("utf-8"))
                environment_file.flush()
                environment_file.seek(0)
                argv = build_bwrap_command(
                    spec,
                    self._policy,
                    bwrap=capability.bwrap,
                    user_namespaces=capability.user_namespaces,
                    env_fd=environment_file.fileno(),
                )
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    env={},
                    pass_fds=(environment_file.fileno(),),
                )
        except BaseException:
            if proxy is not None:
                await proxy.close()
            raise
        if proxy is not None:
            cleanup = asyncio.create_task(self._close_egress_after_exit(process, proxy))
            self._cleanup_tasks.add(cleanup)
            cleanup.add_done_callback(self._cleanup_tasks.discard)
        return process
