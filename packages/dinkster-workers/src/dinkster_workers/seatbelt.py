"""Deny-default macOS pack workers and import probes using Seatbelt."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .egress import EGRESS_PROXY_ENV, EgressProxy
from .launch import LaunchSpec
from .sandbox import (
    PROBE_CPU_LIMIT_S,
    PROBE_FSIZE_LIMIT_BYTES,
    ProbeJail,
    SandboxError,
    SandboxPolicy,
    SandboxUnavailable,
    _absolute_executable,  # pyright: ignore[reportPrivateUsage]
    _launch_environment,  # pyright: ignore[reportPrivateUsage]
    _next_symlink_step,  # pyright: ignore[reportPrivateUsage]
    _paths_overlap,  # pyright: ignore[reportPrivateUsage]
    _validated,  # pyright: ignore[reportPrivateUsage]
    import_surface_mounts,
    interpreter_mounts,
    validate_protected_binds,
)

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
SYSTEM_READ_PATHS = (
    "/usr",
    "/System",
    "/Library/Apple/System/Library",
    "/private/etc/ssl",
    "/private/etc/localtime",
    "/private/etc/hosts",
    "/private/etc/resolv.conf",
    "/private/var/db/timezone",
)
_FORBIDDEN_ROOTS = frozenset(
    {
        "/Users",
        "/Volumes",
        "/System",
        "/Library",
        "/private",
        "/usr",
        "/Applications",
        "/private/var/folders",
        str(Path.home().resolve()),
    }
)


@dataclass(frozen=True)
class SeatbeltCapability:
    available: bool
    sandbox_exec: str | None
    detail: str


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _checked(paths: tuple[str, ...], *, role: str) -> list[str]:
    forbidden = set(_FORBIDDEN_ROOTS)
    shared_metal: list[str] = []
    if sys.platform == "darwin":
        for key in (65536, 65537, 65538):
            root = os.confstr(key)
            if root:
                forbidden.add(os.path.realpath(root))
                if key == 65538:
                    shared_metal = [
                        os.path.join(root, "com.apple.metal"),
                        os.path.join(root, "com.apple.metalfe"),
                    ]
    for path in paths:
        resolved = os.path.realpath(path)
        if (
            resolved in forbidden
            or any(root.startswith(resolved + os.sep) for root in forbidden)
            or any(_paths_overlap(path, cache) for cache in shared_metal)
        ):
            raise SandboxError(f"{role} grant '{path}' would dissolve the sandbox")
    return _validated(paths, role=role)


def build_seatbelt_profile(
    spec: LaunchSpec,
    policy: SandboxPolicy,
    *,
    scratch: Path,
    cache: Path | None = None,
    probe: bool = False,
) -> str:
    """Generate the same filesystem and process boundary for workers and doctor."""
    interpreter, aliases = interpreter_mounts(spec.python, system_paths=SYSTEM_READ_PATHS)
    if probe:
        interpreter, aliases = import_surface_mounts(spec.python, system_paths=SYSTEM_READ_PATHS)
    pythonpath = spec.env.get("PYTHONPATH", "")
    if not pythonpath and "PYTHONPATH" in policy.env_passthrough:
        pythonpath = os.environ.get("PYTHONPATH", "")
    reads = [
        *SYSTEM_READ_PATHS,
        *_checked(tuple(interpreter), role="interpreter"),
        *_checked((str(spec.pack_root),), role="pack root"),
        *_checked(tuple(p for p in pythonpath.split(os.pathsep) if p), role="PYTHONPATH"),
        *_checked(policy.ro_binds, role="read-only"),
    ]
    writes = _checked(policy.rw_binds, role="writable")
    implicit_writes = _checked((str(scratch),), role="private scratch")
    if cache is not None:
        implicit_writes += _checked((str(cache),), role="Metal cache")
    endpoint = _checked((str(spec.endpoint_dir),), role="endpoint")[0]
    validate_protected_binds(
        reads,
        protected_roots=policy.protected_roots,
        exceptions=policy.protected_ro_exceptions,
        role="read-only",
    )
    validate_protected_binds(
        writes,
        protected_roots=policy.protected_roots,
        exceptions=policy.protected_rw_exceptions,
        role="writable",
    )
    validate_protected_binds(
        [endpoint, *implicit_writes],
        protected_roots=policy.protected_roots,
        exceptions=(),
        role="writable launch",
    )
    for writable in (*writes, *implicit_writes):
        for read_only in reads:
            if _paths_overlap(writable, read_only):
                raise SandboxError(
                    f"writable grant '{writable}' overlaps read-only grant '{read_only}'"
                )

    def subpath(path: str | Path) -> str:
        return f"(subpath {_quote(os.path.realpath(path))})"

    rules = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec)",
        "(allow process-fork)",
        "(allow signal (target same-sandbox))",
        "(deny process-info*)",
        "(allow process-info* (target same-sandbox))",
        "(allow sysctl-read)",
        '(deny sysctl-read (sysctl-name-prefix "kern.proc"))',
        '(allow file-read-data (literal "/"))',
        '(allow file-read-metadata (literal "/etc") (literal "/var") (literal "/tmp"))',
        '(allow file-read* (literal "/dev/null") (literal "/dev/random") (literal "/dev/urandom"))',
        '(allow file-write-data (literal "/dev/null"))',
        "(allow ipc-posix-sem)",
        '(allow mach-lookup (global-name "com.apple.system.opendirectoryd.libinfo"))',
    ]
    for path in dict.fromkeys(reads):
        rules.append(f"(allow file-read* {subpath(path)})")
    for path in (*writes, *implicit_writes):
        rules.append(f"(allow file-read* file-write* {subpath(path)})")
    alias_paths = [link for _, link in aliases]
    for path in (*reads, *writes, *implicit_writes, endpoint):
        pending = Path(path).absolute()
        for _ in range(40):
            step = _next_symlink_step(pending)
            if step is None:
                break
            link, _, pending = step
            alias_paths.append(str(link))
    for alias in dict.fromkeys(alias_paths):
        rules.append(f"(allow file-read-metadata (literal {_quote(alias)}))")
    if not probe:
        rules.extend(
            [
                "(allow system-socket (socket-domain AF_UNIX))",
                f"(allow network-outbound (remote unix-socket {subpath(endpoint)}))",
                f"(allow network-bind (local unix-socket {subpath(scratch)}))",
            ]
        )
        names = [r"^/__KMP_REGISTERED_LIB_[0-9]+$", r"^/torch_[0-9]+_[0-9]+_[0-9]+$"]
        if spec.use_shm:
            # Seatbelt regexes do not implement counted repetition.
            names.append("^/dinkster" + "[0-9a-f]" * 16 + "$")
        for name in names:
            rules.append(
                "(allow ipc-posix-shm-read-data ipc-posix-shm-write-data "
                "ipc-posix-shm-write-create ipc-posix-shm-write-unlink "
                f"(ipc-posix-name-regex #{_quote(name)}))"
            )
    if policy.gpu and not probe:
        rules.extend(
            [
                '(allow mach-lookup (global-name "com.apple.MTLCompilerService"))',
                '(allow iokit-open (iokit-user-client-class "AGXDeviceUserClient"))',
            ]
        )
        if cache is not None:
            rules.append(
                '(allow mach-lookup (global-name "com.apple.system.opendirectoryd.membership"))'
            )
    return "\n".join(rules) + "\n"


def detect_seatbelt() -> SeatbeltCapability:
    if sys.platform != "darwin":
        return SeatbeltCapability(
            False, None, f"Seatbelt requires macOS (platform: {sys.platform})"
        )
    if not os.access(SANDBOX_EXEC, os.X_OK):
        return SeatbeltCapability(False, None, f"{SANDBOX_EXEC} is missing")
    profile = (
        "(version 1)(deny default)(allow process-exec)"
        '(allow file-read* (subpath "/usr") (subpath "/System"))'
        '(allow file-read-data (literal "/"))'
    )
    try:
        result = subprocess.run(
            [SANDBOX_EXEC, "-p", profile, "/usr/bin/true"],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return SeatbeltCapability(False, SANDBOX_EXEC, str(exc))
    detail = result.stderr.decode(errors="replace").strip()[:300]
    if result.returncode != 0:
        detail = f"sandbox-exec probe exited {result.returncode}: {detail}"
    return SeatbeltCapability(result.returncode == 0, SANDBOX_EXEC, detail)


def _private_cache(pack_root: Path) -> tuple[Path | None, str]:
    if sys.platform != "darwin":
        return None, ""
    suffix = "dinkster-" + hashlib.sha256(os.fsencode(pack_root.resolve())).hexdigest()
    try:
        base = os.confstr(65538)  # _CS_DARWIN_USER_CACHE_DIR
        if not base:
            return None, ""
        path = Path(base) / suffix
        path.mkdir(mode=0o700, exist_ok=True)
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise SandboxError(f"private Metal cache is not an owned mode-0700 directory: {path}")
        return path, suffix
    except OSError:
        # With dirhelper and the shared cache denied, an absent private cache
        # costs compilation time but cannot fall back to shared writable data.
        return None, ""


_BOOTSTRAP = r"""
import json, os, resource, sys
with os.fdopen(int(sys.argv[1]), encoding='utf-8') as stream:
    config = json.load(stream)
os.environ.clear()
os.environ.update(config['environment'])
for name, value in config['limits']:
    if value is not None:
        soft, hard = resource.getrlimit(getattr(resource, name))
        finite = [v for v in (value, soft, hard) if v != resource.RLIM_INFINITY]
        bound = min(finite)
        resource.setrlimit(getattr(resource, name), (bound, bound))
if config['cache_suffix']:
    try:
        import ctypes
        library = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
        setter = library._set_user_dir_suffix
        setter.argtypes = [ctypes.c_char_p]
        setter.restype = ctypes.c_bool
        setter(config['cache_suffix'].encode())
    except (AttributeError, OSError):
        pass
import site
site.main()
if config.get('probe_path'):
    import importlib.util
    probe_spec = importlib.util.spec_from_file_location(
        'dinkster_workers._doctor_probe', config['probe_path'])
    probe = importlib.util.module_from_spec(probe_spec)
    sys.modules[probe_spec.name] = probe
    probe_spec.loader.exec_module(probe)
sys.path[0:0] = [os.path.abspath(p)
                 for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]
if config.get('gpu'):
    try:
        import importlib.util
        if importlib.util.find_spec('torch') is not None:
            import torch
            if not torch.backends.mps.is_available():
                print('Seatbelt: MPS is unavailable inside the sandbox. GPU access grants '
                      'only MTLCompilerService and AGXDeviceUserClient; virtual GPUs '
                      'requiring AppleParavirtDeviceUserClient are not supported. '
                      'CPU execution remains available.', file=sys.stderr, flush=True)
    except Exception as error:
        print('Seatbelt: could not determine sandboxed MPS availability '
              f'({type(error).__name__}). Continuing with the requested command.',
              file=sys.stderr, flush=True)
import runpy
args = sys.argv[2:]
if config.get('probe_path'):
    api_root = os.environ.pop('DINKSTER_PROBE_API_ROOT', None)
    if api_root:
        sys.path.insert(0, api_root)
    sys.argv = args[1:]
    probe.main()
elif args[0] == '-m':
    sys.argv = args[1:]
    runpy.run_module(args[1], run_name='__main__', alter_sys=True)
elif args[0] == '-c':
    sys.argv = ['-c', *args[2:]]
    exec(compile(args[1], '<string>', 'exec'), {'__name__': '__main__'})
else:
    sys.argv = args
    sys.path.insert(0, os.path.dirname(os.path.abspath(args[0])))
    runpy.run_path(args[0], run_name='__main__')
"""


def _limits(policy: SandboxPolicy, *, probe: bool = False) -> list[tuple[str, int | None]]:
    limits: list[tuple[str, int | None]] = [
        ("RLIMIT_FSIZE", policy.max_file_size_bytes),
        ("RLIMIT_NPROC", policy.max_processes),
    ]
    if probe:
        limits = [("RLIMIT_CPU", PROBE_CPU_LIMIT_S), ("RLIMIT_FSIZE", PROBE_FSIZE_LIMIT_BYTES)]
    for name, value in limits:
        if value is not None and (type(value) is not int or value <= 0):
            raise SandboxError(f"{name} must be a positive integer or None")
    return limits


class SeatbeltLauncher:
    def __init__(
        self,
        policy: SandboxPolicy | None = None,
        *,
        capability: SeatbeltCapability | None = None,
    ) -> None:
        self._policy = policy or SandboxPolicy()
        self._capability = capability
        self._cleanup_tasks: set[asyncio.Task[None]] = set()

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    async def launch(self, spec: LaunchSpec) -> asyncio.subprocess.Process:
        capability = self._capability
        if capability is None:
            capability = self._capability = detect_seatbelt()
        if not capability.available or capability.sandbox_exec is None:
            raise SandboxUnavailable(f"Seatbelt sandbox unavailable: {capability.detail}")
        scratch = spec.endpoint_dir / "tmp"
        scratch.mkdir(mode=0o700, exist_ok=True)
        cache, suffix = _private_cache(spec.pack_root) if self._policy.gpu else (None, "")
        profile = build_seatbelt_profile(spec, self._policy, scratch=scratch, cache=cache)
        environment = _launch_environment(spec, self._policy)
        environment.update(HOME=str(scratch), TMPDIR=".", PYTHONDONTWRITEBYTECODE="1")
        environment.pop(EGRESS_PROXY_ENV, None)
        proxy = None
        if self._policy.egress_allowlist:
            proxy = await EgressProxy.start(
                spec.endpoint_dir / "egress.sock", self._policy.egress_allowlist
            )
            environment[EGRESS_PROXY_ENV] = str(proxy.socket_path)
        try:
            with tempfile.TemporaryFile() as stream:
                stream.write(
                    json.dumps(
                        {
                            "environment": environment,
                            "cache_suffix": suffix,
                            "gpu": self._policy.gpu,
                            "limits": _limits(self._policy),
                        }
                    ).encode()
                )
                stream.seek(0)
                process = await asyncio.create_subprocess_exec(
                    capability.sandbox_exec,
                    "-p",
                    profile,
                    _absolute_executable(spec.python),
                    "-I",
                    "-S",
                    "-c",
                    _BOOTSTRAP,
                    str(stream.fileno()),
                    *spec.command[1:],
                    env={},
                    pass_fds=(stream.fileno(),),
                    cwd=scratch,
                    start_new_session=True,
                )
        except BaseException:
            if proxy is not None:
                await proxy.close()
            raise
        if proxy is not None:

            async def cleanup() -> None:
                try:
                    await process.wait()
                finally:
                    await proxy.close()

            task = asyncio.create_task(cleanup())
            self._cleanup_tasks.add(task)
            task.add_done_callback(self._cleanup_tasks.discard)
        return process


@dataclass(frozen=True)
class SeatbeltProbeJail(ProbeJail):
    """Doctor's synchronous, no-network/no-GPU variant of the worker profile."""

    def wrap(
        self,
        argv: list[str],
        manifest_root: Path,
        *,
        env_fd: int = 3,
        source_roots: tuple[Path, ...] = (),
    ) -> list[str]:
        raise SandboxUnavailable(
            "Seatbelt import probes require scoped run() to manage their private scratch directory"
        )

    def run(
        self,
        argv: list[str],
        manifest_root: Path,
        environment: dict[str, str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix="dinkster-probe-") as directory:
            scratch = Path(directory)
            environment = {
                **environment,
                "HOME": directory,
                "TMPDIR": ".",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            spec = LaunchSpec(tuple(argv), environment, scratch, manifest_root, argv[0], False)
            policy = SandboxPolicy()
            profile = build_seatbelt_profile(spec, policy, scratch=scratch, probe=True)
            with tempfile.TemporaryFile() as stream:
                stream.write(
                    json.dumps(
                        {
                            "environment": environment,
                            "cache_suffix": "",
                            "limits": _limits(policy, probe=True),
                            "probe_path": str(Path(__file__).with_name("_doctor_probe.py")),
                        }
                    ).encode()
                )
                stream.seek(0)
                return subprocess.run(
                    [
                        self.bwrap,
                        "-p",
                        profile,
                        _absolute_executable(argv[0]),
                        "-I",
                        "-S",
                        "-c",
                        _BOOTSTRAP,
                        str(stream.fileno()),
                        *argv[1:],
                    ],
                    env={},
                    pass_fds=(stream.fileno(),),
                    cwd=scratch,
                    timeout=timeout,
                    capture_output=True,
                    text=True,
                    errors="replace",
                )
