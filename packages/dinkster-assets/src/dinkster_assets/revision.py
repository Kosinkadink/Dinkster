"""Cheap native revision identity for unverified Windows file bindings."""

from __future__ import annotations

import ctypes
import os
import struct
from ctypes import wintypes
from pathlib import Path
from typing import BinaryIO

from .identity import AssetError


def _open_windows_file(path: Path) -> int:
    if os.name != "nt":
        raise OSError("native file handles are only available on Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel.CreateFileW.restype = wintypes.HANDLE
    name = os.path.abspath(path)
    if not name.startswith("\\\\?\\"):
        name = "\\\\?\\UNC\\" + name[2:] if name.startswith("\\\\") else "\\\\?\\" + name
    # A USN can coalesce repeated changes until the writer closes. Refuse
    # existing writers and exclude new writers for this handle's lifetime.
    handle = kernel.CreateFileW(name, 0x80000000, 5, None, 3, 0x02000000, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def _close_windows_file(handle: int) -> None:
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle(handle)


def open_local_file(path: Path) -> BinaryIO:
    """Open local bytes with Windows write exclusion retained until close."""
    if os.name != "nt":
        return path.open("rb")
    import msvcrt

    handle = _open_windows_file(path)
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY | os.O_NOINHERIT)
    except BaseException:
        _close_windows_file(handle)
        raise
    try:
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def windows_file_revision(path: Path | int) -> int:
    """Read a tracked NTFS USN without opening a volume or requiring elevation."""
    if os.name != "nt":
        raise OSError("native file USN is only available on Windows")
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.DeviceIoControl.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    )
    owned = not isinstance(path, int)
    handle = _open_windows_file(path) if isinstance(path, Path) else msvcrt.get_osfhandle(path)
    try:
        versions = ctypes.create_string_buffer(struct.pack("<HH", 2, 2))
        record = ctypes.create_string_buffer(65536)
        length = wintypes.DWORD()
        if not kernel.DeviceIoControl(
            handle, 0x900EB, versions, 4, record, len(record), ctypes.byref(length), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if length.value < 32 or struct.unpack_from("<H", record, 4)[0] != 2:
            raise OSError("unsupported file USN record")
        revision = struct.unpack_from("<q", record, 24)[0]
        if revision == 0:
            raise OSError("file USN is zero (change journal inactive)")
        return revision
    finally:
        if owned:
            _close_windows_file(handle)


def mount_revision_capability(root: Path) -> dict[str, object]:
    if os.name != "nt":
        return {"supported": True, "method": "stat"}
    filesystem = "unknown"
    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetVolumePathNameW.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
        kernel.GetVolumeInformationW.argtypes = (
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.LPWSTR,
            wintypes.DWORD,
        )
        volume = ctypes.create_unicode_buffer(32768)
        name = ctypes.create_unicode_buffer(256)
        if not kernel.GetVolumePathNameW(str(root.resolve()), volume, len(volume)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.GetVolumeInformationW(
            volume.value, None, 0, None, None, None, name, len(name)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        filesystem = name.value
        windows_file_revision(root)
    except OSError as error:
        return {
            "supported": False,
            "filesystem": filesystem,
            "reason": error.strerror or str(error),
        }
    return {"supported": True, "method": "ntfs-usn", "filesystem": filesystem}


def require_local_revision(mount_id: str, capability: dict[str, object], progress: object) -> None:
    if capability.get("supported") is not True:
        raise AssetError(
            f"mount {mount_id!r} ({capability.get('filesystem', 'unknown')}): "
            f"file revision tracking unavailable: {capability.get('reason', 'unsupported')}; "
            "wait for the background index to verify this file, then select the same name "
            f"as a verified AssetRef; scan progress: {progress}"
        )
