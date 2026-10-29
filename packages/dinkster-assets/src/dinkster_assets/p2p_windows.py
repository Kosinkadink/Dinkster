"""Handle-relative Windows filesystem operations for confined P2P storage."""

from __future__ import annotations

import ctypes
import os
import stat
from ctypes import wintypes
from pathlib import Path
from typing import NoReturn

from .p2p_usn import usn_from_record

if os.name != "nt":  # pragma: no cover - imported only by the Windows branch
    raise ImportError("p2p_windows is available only on Windows")

_ntdll = ctypes.WinDLL("ntdll")
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_OBJ_CASE_INSENSITIVE = 0x40
_OBJ_DONT_REPARSE = 0x1000
_DELETE = 0x00010000
_SYNCHRONIZE = 0x00100000
_FILE_LIST_DIRECTORY = 0x0001
_FILE_READ_DATA = 0x0001
_FILE_WRITE_DATA = 0x0002
_FILE_ADD_FILE = 0x0002
_FILE_ADD_SUBDIRECTORY = 0x0004
_FILE_TRAVERSE = 0x0020
_FILE_READ_ATTRIBUTES = 0x0080
_FILE_WRITE_ATTRIBUTES = 0x0100
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_SHARE_DELETE = 0x00000004
_FILE_SHARE_ALL = 0x00000007
_FILE_OPEN = 1
_FILE_CREATE = 2
_FILE_OPEN_IF = 3
_FILE_DIRECTORY_FILE = 0x00000001
_FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
_FILE_NON_DIRECTORY_FILE = 0x00000040
_FILE_OPEN_REPARSE_POINT = 0x00200000
_FILE_ATTRIBUTE_NORMAL = 0x00000080
_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_OPEN_EXISTING = 3
_FILE_RENAME_INFORMATION = 10
_FILE_RENAME_INFORMATION_EX = 65
_FILE_RENAME_REPLACE_IF_EXISTS = 0x01
_FILE_RENAME_POSIX_SEMANTICS = 0x02
_FILE_DISPOSITION_INFORMATION = 13
_FILE_DISPOSITION_INFORMATION_EX = 64
_FILE_DISPOSITION_DELETE = 0x01
_FILE_DISPOSITION_POSIX_SEMANTICS = 0x02
_FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE = 0x10
_FILE_BASIC_INFO = 0
_FILE_STANDARD_INFO = 1
_FILE_END_OF_FILE_INFORMATION = 20
_FILE_ID_INFO = 18
_FILE_ID_BOTH_DIRECTORY_INFO = 10
_FILE_ID_BOTH_DIRECTORY_RESTART_INFO = 11
_ERROR_INVALID_FUNCTION = 1
_ERROR_NO_MORE_FILES = 18
_ERROR_NOT_SUPPORTED = 50
_ERROR_LOCK_VIOLATION = 33
_LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
_LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
_FSCTL_READ_FILE_USN_DATA = 0x000900EB
_FSCTL_SET_SPARSE = 0x000900C4
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class _UnicodeString(ctypes.Structure):
    _fields_ = [
        ("Length", wintypes.USHORT),
        ("MaximumLength", wintypes.USHORT),
        ("Buffer", wintypes.LPWSTR),
    ]


class _IoStatusValue(ctypes.Union):
    _fields_ = [("Status", wintypes.LONG), ("Pointer", wintypes.LPVOID)]


class _IoStatusBlock(ctypes.Structure):
    _anonymous_ = ("Value",)
    _fields_ = [("Value", _IoStatusValue), ("Information", ctypes.c_size_t)]


class _ObjectAttributes(ctypes.Structure):
    _fields_ = [
        ("Length", wintypes.ULONG),
        ("RootDirectory", wintypes.HANDLE),
        ("ObjectName", ctypes.POINTER(_UnicodeString)),
        ("Attributes", wintypes.ULONG),
        ("SecurityDescriptor", wintypes.LPVOID),
        ("SecurityQualityOfService", wintypes.LPVOID),
    ]


class _ByHandleFileInformation(ctypes.Structure):
    _fields_ = [
        ("FileAttributes", wintypes.DWORD),
        ("CreationTime", wintypes.FILETIME),
        ("LastAccessTime", wintypes.FILETIME),
        ("LastWriteTime", wintypes.FILETIME),
        ("VolumeSerialNumber", wintypes.DWORD),
        ("FileSizeHigh", wintypes.DWORD),
        ("FileSizeLow", wintypes.DWORD),
        ("NumberOfLinks", wintypes.DWORD),
        ("FileIndexHigh", wintypes.DWORD),
        ("FileIndexLow", wintypes.DWORD),
    ]


class _NameInformation(ctypes.Structure):
    _fields_ = [
        ("ReplaceIfExists", wintypes.BOOLEAN),
        ("RootDirectory", wintypes.HANDLE),
        ("FileNameLength", wintypes.ULONG),
        ("FileName", wintypes.WCHAR * 1),
    ]


class _FileStandardInformation(ctypes.Structure):
    _fields_ = [
        ("AllocationSize", ctypes.c_longlong),
        ("EndOfFile", ctypes.c_longlong),
        ("NumberOfLinks", wintypes.DWORD),
        ("DeletePending", wintypes.BOOLEAN),
        ("Directory", wintypes.BOOLEAN),
    ]


class _FileBasicInformation(ctypes.Structure):
    _fields_ = [
        ("CreationTime", ctypes.c_longlong),
        ("LastAccessTime", ctypes.c_longlong),
        ("LastWriteTime", ctypes.c_longlong),
        ("ChangeTime", ctypes.c_longlong),
        ("FileAttributes", wintypes.DWORD),
    ]


class _FileId128(ctypes.Structure):
    _fields_ = [("Identifier", wintypes.BYTE * 16)]


class _FileIdInformation(ctypes.Structure):
    _fields_ = [
        ("VolumeSerialNumber", ctypes.c_ulonglong),
        ("FileId", _FileId128),
    ]


class _FileIdBothDirectoryInformation(ctypes.Structure):
    _fields_ = [
        ("NextEntryOffset", wintypes.DWORD),
        ("FileIndex", wintypes.DWORD),
        ("CreationTime", ctypes.c_longlong),
        ("LastAccessTime", ctypes.c_longlong),
        ("LastWriteTime", ctypes.c_longlong),
        ("ChangeTime", ctypes.c_longlong),
        ("EndOfFile", ctypes.c_longlong),
        ("AllocationSize", ctypes.c_longlong),
        ("FileAttributes", wintypes.DWORD),
        ("FileNameLength", wintypes.DWORD),
        ("EaSize", wintypes.DWORD),
        ("ShortNameLength", ctypes.c_byte),
        ("ShortName", wintypes.WCHAR * 12),
        ("FileId", ctypes.c_longlong),
        ("FileName", wintypes.WCHAR * 1),
    ]


class _ReadFileUsnData(ctypes.Structure):
    _fields_ = [
        ("MinMajorVersion", wintypes.WORD),
        ("MaxMajorVersion", wintypes.WORD),
    ]


class _Overlapped(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("Event", wintypes.HANDLE),
    ]


_ntdll.NtCreateFile.restype = wintypes.LONG
_ntdll.NtCreateFile.argtypes = (
    ctypes.POINTER(wintypes.HANDLE),
    wintypes.DWORD,
    ctypes.POINTER(_ObjectAttributes),
    ctypes.POINTER(_IoStatusBlock),
    ctypes.POINTER(ctypes.c_longlong),
    wintypes.ULONG,
    wintypes.ULONG,
    wintypes.ULONG,
    wintypes.ULONG,
    wintypes.LPVOID,
    wintypes.ULONG,
)
_ntdll.NtSetInformationFile.restype = wintypes.LONG
_ntdll.NtSetInformationFile.argtypes = (
    wintypes.HANDLE,
    ctypes.POINTER(_IoStatusBlock),
    wintypes.LPVOID,
    wintypes.ULONG,
    wintypes.ULONG,
)
_ntdll.NtFlushBuffersFile.restype = wintypes.LONG
_ntdll.NtFlushBuffersFile.argtypes = (
    wintypes.HANDLE,
    ctypes.POINTER(_IoStatusBlock),
)
_ntdll.RtlNtStatusToDosError.restype = wintypes.ULONG
_ntdll.RtlNtStatusToDosError.argtypes = (wintypes.LONG,)
_kernel32.CreateFileW.restype = wintypes.HANDLE
_kernel32.CreateFileW.argtypes = (
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
)
_kernel32.CloseHandle.restype = wintypes.BOOL
_kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
_kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
_kernel32.GetFileInformationByHandle.argtypes = (
    wintypes.HANDLE,
    ctypes.POINTER(_ByHandleFileInformation),
)
_kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
_kernel32.GetFileInformationByHandleEx.argtypes = (
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    wintypes.DWORD,
)
_kernel32.LockFileEx.restype = wintypes.BOOL
_kernel32.LockFileEx.argtypes = (
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(_Overlapped),
)
_kernel32.UnlockFileEx.restype = wintypes.BOOL
_kernel32.UnlockFileEx.argtypes = (
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(_Overlapped),
)
_kernel32.DeviceIoControl.restype = wintypes.BOOL
_kernel32.DeviceIoControl.argtypes = (
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    wintypes.LPVOID,
)


def _raise_status(status: int) -> NoReturn:
    error = int(_ntdll.RtlNtStatusToDosError(status))
    raise ctypes.WinError(error)


def _raise_last_error() -> NoReturn:
    raise ctypes.WinError(ctypes.get_last_error())


def _handle(value: int) -> wintypes.HANDLE:
    return wintypes.HANDLE(value)


def _information(handle: int) -> _ByHandleFileInformation:
    result = _ByHandleFileInformation()
    if not _kernel32.GetFileInformationByHandle(_handle(handle), ctypes.byref(result)):
        _raise_last_error()
    return result


def identity(handle: int) -> tuple[int, int]:
    info = _FileIdInformation()
    if not _kernel32.GetFileInformationByHandleEx(
        _handle(handle),
        _FILE_ID_INFO,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        _raise_last_error()
    return (
        info.VolumeSerialNumber,
        int.from_bytes(bytes(info.FileId.Identifier), "little"),
    )


def volume(handle: int) -> int:
    return identity(handle)[0]


def close(handle: int) -> None:
    if not _kernel32.CloseHandle(_handle(handle)):
        _raise_last_error()


def open_root(path: Path) -> int:
    access = (
        _FILE_LIST_DIRECTORY
        | _FILE_ADD_FILE
        | _FILE_ADD_SUBDIRECTORY
        | _FILE_TRAVERSE
        | _FILE_READ_ATTRIBUTES
        | _SYNCHRONIZE
    )
    raw = _kernel32.CreateFileW(
        str(path),
        access,
        _FILE_SHARE_ALL,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    value = ctypes.cast(raw, ctypes.c_void_p).value
    if value is None or value == _INVALID_HANDLE_VALUE:
        _raise_last_error()
    handle = int(value)
    info = _information(handle)
    if (
        not info.FileAttributes & _FILE_ATTRIBUTE_DIRECTORY
        or info.FileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT
    ):
        close(handle)
        raise OSError("P2P vault root must be a regular non-reparse directory")
    return handle


def _object_attributes(
    name: str, root: int, *, allow_reparse: bool
) -> tuple[object, _UnicodeString, _ObjectAttributes]:
    if not name or name in (".", "..") or "\0" in name or "/" in name or "\\" in name:
        raise ValueError("P2P storage names must contain one canonical component")
    encoded_length = len(name.encode("utf-16-le"))
    if encoded_length > 65_534:
        raise ValueError("P2P storage name is too long")
    buffer = ctypes.create_unicode_buffer(name)
    string = _UnicodeString(
        encoded_length,
        encoded_length,
        ctypes.cast(buffer, wintypes.LPWSTR),
    )
    attributes = _ObjectAttributes(
        ctypes.sizeof(_ObjectAttributes),
        _handle(root),
        ctypes.pointer(string),
        _OBJ_CASE_INSENSITIVE | (0 if allow_reparse else _OBJ_DONT_REPARSE),
        None,
        None,
    )
    return buffer, string, attributes


def _open_relative(
    root: int,
    name: str,
    *,
    directory: bool | None,
    create: bool,
    exclusive: bool,
    writable: bool,
    allow_reparse: bool = False,
    share_write: bool = True,
    delete: bool = False,
) -> int:
    keepalive = _object_attributes(name, root, allow_reparse=allow_reparse)
    attributes = keepalive[2]
    result = wintypes.HANDLE()
    status_block = _IoStatusBlock()
    access = _FILE_READ_ATTRIBUTES | _SYNCHRONIZE | (_DELETE if delete else 0)
    if directory is True:
        access |= _FILE_LIST_DIRECTORY | _FILE_ADD_FILE | _FILE_ADD_SUBDIRECTORY | _FILE_TRAVERSE
    else:
        access |= _FILE_READ_DATA
        if writable:
            access |= _FILE_WRITE_DATA | _FILE_WRITE_ATTRIBUTES
    options = _FILE_OPEN_REPARSE_POINT | _FILE_SYNCHRONOUS_IO_NONALERT
    if directory is True:
        options |= _FILE_DIRECTORY_FILE
    elif directory is False:
        options |= _FILE_NON_DIRECTORY_FILE
    disposition = _FILE_CREATE if exclusive else (_FILE_OPEN_IF if create else _FILE_OPEN)
    status = _ntdll.NtCreateFile(
        ctypes.byref(result),
        access,
        ctypes.byref(attributes),
        ctypes.byref(status_block),
        None,
        _FILE_ATTRIBUTE_NORMAL,
        _FILE_SHARE_READ | _FILE_SHARE_DELETE | (_FILE_SHARE_WRITE if share_write else 0),
        disposition,
        options,
        None,
        0,
    )
    if status < 0:
        _raise_status(status)
    value = ctypes.cast(result, ctypes.c_void_p).value
    if value is None:
        raise OSError("NtCreateFile returned a null handle")
    handle = int(value)
    info = _information(handle)
    is_directory = bool(info.FileAttributes & _FILE_ATTRIBUTE_DIRECTORY)
    if (
        (not allow_reparse and info.FileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT)
        or (directory is not None and is_directory != directory)
        or volume(handle) != volume(root)
    ):
        close(handle)
        raise OSError(
            "P2P entry must be regular non-symlink/non-reparse and on the vault filesystem"
        )
    return handle


def open_directory(root: int, name: str, *, create: bool, delete: bool = False) -> int:
    return _open_relative(
        root,
        name,
        directory=True,
        create=create,
        exclusive=False,
        writable=True,
        delete=delete,
    )


def open_file(
    root: int,
    name: str,
    *,
    writable: bool,
    exclusive: bool,
    share_write: bool = True,
    delete: bool = False,
) -> int:
    return _open_relative(
        root,
        name,
        directory=False,
        create=exclusive,
        exclusive=exclusive,
        writable=writable,
        share_write=share_write,
        delete=delete,
    )


def open_entry(root: int, name: str, *, delete: bool = False) -> int:
    return _open_relative(
        root,
        name,
        directory=None,
        create=False,
        exclusive=False,
        writable=False,
        allow_reparse=True,
        delete=delete,
    )


def _name_buffer(
    root: int,
    name: str,
    *,
    replace: bool,
    flags: int | None,
) -> ctypes.Array[ctypes.c_char]:
    encoded = name.encode("utf-16-le")
    size = _NameInformation.FileName.offset + len(encoded)
    buffer = ctypes.create_string_buffer(size)
    information = _NameInformation.from_buffer(buffer)
    information.ReplaceIfExists = replace
    information.RootDirectory = _handle(root)
    information.FileNameLength = len(encoded)
    if flags is not None:
        wintypes.ULONG.from_buffer(buffer).value = flags
    ctypes.memmove(
        ctypes.addressof(buffer) + _NameInformation.FileName.offset,
        encoded,
        len(encoded),
    )
    return buffer


def _set_name(
    handle: int,
    root: int,
    name: str,
    *,
    replace: bool,
    flags: int | None = None,
    info_class: int,
) -> None:
    buffer = _name_buffer(root, name, replace=replace, flags=flags)
    status_block = _IoStatusBlock()
    status = _ntdll.NtSetInformationFile(
        _handle(handle),
        ctypes.byref(status_block),
        buffer,
        len(buffer),
        info_class,
    )
    if status < 0:
        _raise_status(status)


def rename(handle: int, root: int, name: str, *, replace: bool) -> None:
    _set_name(
        handle,
        root,
        name,
        replace=replace,
        info_class=_FILE_RENAME_INFORMATION,
    )


def replace(handle: int, root: int, name: str) -> None:
    try:
        _set_name(
            handle,
            root,
            name,
            replace=True,
            flags=_FILE_RENAME_REPLACE_IF_EXISTS | _FILE_RENAME_POSIX_SEMANTICS,
            info_class=_FILE_RENAME_INFORMATION_EX,
        )
    except OSError as error:
        if getattr(error, "winerror", None) not in (1, 50, 87):
            raise
        rename(handle, root, name, replace=True)


def delete(handle: int) -> None:
    status_block = _IoStatusBlock()
    flags = wintypes.ULONG(
        _FILE_DISPOSITION_DELETE
        | _FILE_DISPOSITION_POSIX_SEMANTICS
        | _FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE
    )
    status = _ntdll.NtSetInformationFile(
        _handle(handle),
        ctypes.byref(status_block),
        ctypes.byref(flags),
        ctypes.sizeof(flags),
        _FILE_DISPOSITION_INFORMATION_EX,
    )
    if status < 0:
        remove = wintypes.BOOLEAN(True)
        status = _ntdll.NtSetInformationFile(
            _handle(handle),
            ctypes.byref(status_block),
            ctypes.byref(remove),
            ctypes.sizeof(remove),
            _FILE_DISPOSITION_INFORMATION,
        )
    if status < 0:
        _raise_status(status)


def entry_names(handle: int) -> list[str]:
    names: list[str] = []
    restart = True
    while True:
        buffer = ctypes.create_string_buffer(64 * 1024)
        info_class = (
            _FILE_ID_BOTH_DIRECTORY_RESTART_INFO if restart else _FILE_ID_BOTH_DIRECTORY_INFO
        )
        if not _kernel32.GetFileInformationByHandleEx(
            _handle(handle), info_class, buffer, len(buffer)
        ):
            error = ctypes.get_last_error()
            if error == _ERROR_NO_MORE_FILES:
                return names
            _raise_last_error()
        restart = False
        offset = 0
        while True:
            address = ctypes.addressof(buffer) + offset
            row = _FileIdBothDirectoryInformation.from_address(address)
            name = ctypes.wstring_at(
                address + _FileIdBothDirectoryInformation.FileName.offset,
                row.FileNameLength // 2,
            )
            if name not in (".", ".."):
                names.append(name)
            if row.NextEntryOffset == 0:
                break
            offset += row.NextEntryOffset


def stat_entry(root: int, name: str) -> os.stat_result:
    import msvcrt

    handle = open_entry(root, name)
    info = _information(handle)
    if info.FileAttributes & (_FILE_ATTRIBUTE_DIRECTORY | _FILE_ATTRIBUTE_REPARSE_POINT):
        device, inode = identity(handle)
        close(handle)
        mode = stat.S_IFLNK if info.FileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT else stat.S_IFDIR
        size = (info.FileSizeHigh << 32) | info.FileSizeLow
        return os.stat_result(
            (
                mode,
                inode,
                device,
                info.NumberOfLinks,
                0,
                0,
                size,
                0,
                0,
                0,
            )
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    try:
        descriptor = msvcrt.open_osfhandle(handle, flags)
    except BaseException:
        close(handle)
        raise
    try:
        return os.fstat(descriptor)
    finally:
        os.close(descriptor)


def allocated_size(root: int, name: str) -> int:
    handle = open_entry(root, name)
    try:
        result = _FileStandardInformation()
        if not _kernel32.GetFileInformationByHandleEx(
            _handle(handle),
            _FILE_STANDARD_INFO,
            ctypes.byref(result),
            ctypes.sizeof(result),
        ):
            _raise_last_error()
        return max(0, result.AllocationSize)
    finally:
        close(handle)


def take_file_descriptor(handle: int, flags: int) -> int:
    import msvcrt

    return msvcrt.open_osfhandle(handle, flags)


def raw_file_handle(descriptor: int) -> int:
    import msvcrt

    return msvcrt.get_osfhandle(descriptor)


def change_token(descriptor: int) -> int:
    result = _FileBasicInformation()
    handle = _handle(raw_file_handle(descriptor))
    if not _kernel32.GetFileInformationByHandleEx(
        handle,
        _FILE_BASIC_INFO,
        ctypes.byref(result),
        ctypes.sizeof(result),
    ):
        _raise_last_error()
    request = _ReadFileUsnData(2, 3)
    # A USN record includes up to a 255-character UTF-16 filename.
    buffer = ctypes.create_string_buffer(4096)
    returned = wintypes.DWORD()
    if not _kernel32.DeviceIoControl(
        handle,
        _FSCTL_READ_FILE_USN_DATA,
        ctypes.byref(request),
        ctypes.sizeof(request),
        buffer,
        len(buffer),
        ctypes.byref(returned),
        None,
    ):
        _raise_last_error()
    usn = usn_from_record(buffer.raw, returned.value)
    return (usn << 64) | (result.ChangeTime & ((1 << 64) - 1))


def flush(handle: int) -> None:
    status_block = _IoStatusBlock()
    status = _ntdll.NtFlushBuffersFile(_handle(handle), ctypes.byref(status_block))
    if status < 0:
        _raise_status(status)


def lock_file(descriptor: int, *, blocking: bool) -> _Overlapped | None:
    overlapped = _Overlapped(Offset=0xFFFFFFFE, OffsetHigh=0x7FFFFFFF)
    flags = _LOCKFILE_EXCLUSIVE_LOCK | (0 if blocking else _LOCKFILE_FAIL_IMMEDIATELY)
    if not _kernel32.LockFileEx(
        _handle(raw_file_handle(descriptor)),
        flags,
        0,
        1,
        0,
        ctypes.byref(overlapped),
    ):
        error = ctypes.get_last_error()
        if not blocking and error == _ERROR_LOCK_VIOLATION:
            return None
        raise ctypes.WinError(error)
    return overlapped


def unlock_file(descriptor: int, overlapped: _Overlapped) -> None:
    if not _kernel32.UnlockFileEx(
        _handle(raw_file_handle(descriptor)),
        0,
        1,
        0,
        ctypes.byref(overlapped),
    ):
        _raise_last_error()


def make_sparse(descriptor: int, size: int) -> None:
    handle = raw_file_handle(descriptor)
    returned = wintypes.DWORD()
    if not _kernel32.DeviceIoControl(
        _handle(handle),
        _FSCTL_SET_SPARSE,
        None,
        0,
        None,
        0,
        ctypes.byref(returned),
        None,
    ):
        error = ctypes.get_last_error()
        if error not in (_ERROR_INVALID_FUNCTION, _ERROR_NOT_SUPPORTED):
            raise ctypes.WinError(error)
    end = ctypes.c_longlong(size)
    status_block = _IoStatusBlock()
    status = _ntdll.NtSetInformationFile(
        _handle(handle),
        ctypes.byref(status_block),
        ctypes.byref(end),
        ctypes.sizeof(end),
        _FILE_END_OF_FILE_INFORMATION,
    )
    if status < 0:
        _raise_status(status)
