"""Export a native-ABI seccomp filter for bubblewrap, without loading it here."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import tempfile
from collections.abc import Generator
from contextlib import contextmanager

# Flatpak-style kernel attack-surface reduction, not a syscall allowlist.
# Ordinary files, threads, subprocesses, shared memory and GPU ioctls remain
# available; namespace/mount policy and network reachability live in bwrap.
_DENIED_SYSCALLS = (
    "syslog",
    "uselib",
    "acct",
    "modify_ldt",
    "quotactl",
    "add_key",
    "keyctl",
    "request_key",
    "move_pages",
    "mbind",
    "get_mempolicy",
    "set_mempolicy",
    "migrate_pages",
    "unshare",
    "setns",
    "mount",
    "umount",
    "umount2",
    "pivot_root",
    "chroot",
    "fsopen",
    "fsconfig",
    "fsmount",
    "fspick",
    "open_tree",
    "move_mount",
    "mount_setattr",
    "bpf",
    "userfaultfd",
    "perf_event_open",
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "kexec_load",
    "kexec_file_load",
    "reboot",
    "swapon",
    "swapoff",
    "init_module",
    "finit_module",
    "delete_module",
    "iopl",
    "ioperm",
    "open_by_handle_at",
    "name_to_handle_at",
    "fanotify_init",
    "personality",
)


class _ArgCompare(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


@contextmanager
def seccomp_filter() -> Generator[int]:
    """Yield a descriptor for rewound BPF; a missing/old libseccomp fails closed.

    libseccomp installs an architecture guard (including x32 on x86-64).
    Only the native ABI is enabled, so a compat syscall cannot bypass rules.
    The filter is applied by bwrap after its namespace and mount setup.
    """
    try:
        lib = ctypes.CDLL("libseccomp.so.2")
    except OSError as exc:
        raise OSError("seccomp requires libseccomp.so.2; install libseccomp2/libseccomp") from exc
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.restype = None
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_ArgCompare),
    ]
    lib.seccomp_rule_add_array.restype = ctypes.c_int
    lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.seccomp_export_bpf.restype = ctypes.c_int
    context = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise OSError("libseccomp could not initialize the sandbox filter")

    def deny(name: str, error: int = errno.EPERM, comparison: _ArgCompare | None = None) -> None:
        syscall = lib.seccomp_syscall_resolve_name(name.encode("ascii"))
        if syscall == -1:
            raise OSError(f"libseccomp does not know {name}; upgrade libseccomp")
        result = lib.seccomp_rule_add_array(
            context,
            0x00050000 | error,
            syscall,
            int(comparison is not None),
            ctypes.byref(comparison) if comparison is not None else None,
        )
        if result < 0:
            raise OSError(f"libseccomp cannot filter {name}: {os.strerror(-result)}")

    try:
        for name in _DENIED_SYSCALLS:
            deny(name)
        # These pointer-based interfaces cannot be inspected by classic BPF.
        # ENOSYS lets glibc/pthread fall back to clone, and liburing to normal I/O.
        for name in ("clone3", "io_uring_setup", "io_uring_enter", "io_uring_register"):
            deny(name, errno.ENOSYS)
        # SCMP_CMP_MASKED_EQ: preserve fork/thread creation but forbid NEWUSER.
        # s390's clone ABI puts flags in argument 1, not argument 0.
        flags_arg = 1 if platform.machine() in ("s390", "s390x") else 0
        deny("clone", comparison=_ArgCompare(flags_arg, 7, 0x10000000, 0x10000000))
        # Terminal injection must not survive an inherited terminal descriptor.
        for request in (0x5412, 0x541C):  # TIOCSTI, TIOCLINUX
            deny("ioctl", comparison=_ArgCompare(1, 7, 0xFFFFFFFF, request))
        with tempfile.TemporaryFile() as exported:
            result = lib.seccomp_export_bpf(context, exported.fileno())
            if result < 0:
                raise OSError(f"libseccomp cannot export BPF: {os.strerror(-result)}")
            exported.seek(0)
            yield exported.fileno()
    finally:
        lib.seccomp_release(context)
