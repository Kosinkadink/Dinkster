"""Measure allocated fixture storage across automatic and explicit collection.

Run with the repository's development interpreter as
python -m scripts.measure_generation_cleanup --output NEW_DIRECTORY.
Only newly created disposable installations are operated on; they remain for inspection.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import subprocess
import sys
import tempfile
from ctypes import wintypes
from pathlib import Path
from threading import Event, Thread
from typing import Any

from tests.fixtures.environments.build import build, build_base, executable_release


def allocated(root: Path, store: Path) -> int:
    if os.name == "nt":

        class FileStandardInfo(ctypes.Structure):
            _fields_ = [
                ("AllocationSize", ctypes.c_longlong),
                ("EndOfFile", ctypes.c_longlong),
                ("NumberOfLinks", wintypes.DWORD),
                ("DeletePending", ctypes.c_ubyte),
                ("Directory", ctypes.c_ubyte),
            ]

        class FileIdInfo(ctypes.Structure):
            _fields_ = [("VolumeSerialNumber", ctypes.c_ulonglong), ("FileId", ctypes.c_ubyte * 16)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        kernel.GetFileInformationByHandleEx.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
        seen: set[tuple[int, bytes]] = set()
        total = 0
        for directory in (root, store):
            for parent, directories, files in os.walk(directory, onerror=raise_scan_error):
                directories[:] = [
                    name
                    for name in directories
                    if not (Path(parent) / name).is_symlink()
                    and not (Path(parent) / name).is_junction()
                ]
                for name in files:
                    path = Path(parent) / name
                    if path.is_symlink():
                        continue
                    # Share deletion so sampling never prevents quarantine or collection.
                    handle = kernel.CreateFileW(str(path), 0x80, 7, None, 3, 0x00200000, None)
                    if handle == wintypes.HANDLE(-1).value:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        file_id = FileIdInfo()
                        if not kernel.GetFileInformationByHandleEx(
                            handle, 18, ctypes.byref(file_id), ctypes.sizeof(file_id)
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        identity = (file_id.VolumeSerialNumber, bytes(file_id.FileId))
                        if identity in seen:
                            continue
                        info = FileStandardInfo()
                        if not kernel.GetFileInformationByHandleEx(
                            handle,
                            1,
                            ctypes.byref(info),
                            ctypes.sizeof(info),
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        seen.add(identity)
                        total += info.AllocationSize
                    finally:
                        kernel.CloseHandle(handle)
        return total
    result = subprocess.check_output(
        ["du", "-s", "--block-size=1", "--", str(root), str(store)],
        text=True,
        stderr=subprocess.PIPE,
    )
    return sum(int(line.split()[0]) for line in result.splitlines())


def raise_scan_error(error: OSError) -> None:
    raise error


def filesystem(output: Path) -> str:
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetVolumePathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        kernel.GetVolumePathNameW.restype = wintypes.BOOL
        kernel.GetVolumeInformationW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.LPWSTR,
            wintypes.DWORD,
        ]
        kernel.GetVolumeInformationW.restype = wintypes.BOOL
        volume = ctypes.create_unicode_buffer(32768)
        if not kernel.GetVolumePathNameW(str(output), volume, len(volume)):
            raise ctypes.WinError(ctypes.get_last_error())
        name = ctypes.create_unicode_buffer(256)
        if not kernel.GetVolumeInformationW(
            volume.value, None, 0, None, None, None, name, len(name)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return name.value
    return subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(output)], text=True
    ).strip()


def scenario(output: Path, changed_base: bool) -> dict[str, Any]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    variant = "windows-cpu" if os.name == "nt" else "linux-cpu"
    fixture = build(output, variant, port)
    root, store, mirror = (Path(fixture[key]) for key in ("root", "store", "mirror"))
    environment = {**os.environ, "XDG_DATA_HOME": str(output / "user-data")}
    command = [sys.executable, "-m", "dinkster_supervisor.environment", "--root", str(root)]

    def run(*arguments: str) -> dict[str, Any]:
        completed = subprocess.run(
            [*command, *arguments, "--json"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        return json.loads(completed.stdout)

    setup = fixture["commands"]["setup"]
    try:
        run(*setup[setup.index("setup") : -1])
        (mirror / "channels/main.json").write_bytes(Path(fixture["updatedChannel"]).read_bytes())
        run("update", "--yes")
        protected: dict[Path, bytes] = {}
        for name in ("workflows", "models", "outputs", "mounts"):
            directory = root / name
            directory.mkdir(exist_ok=True)
            path = directory / "sentinel"
            path.write_bytes(name.encode("ascii") + b" private data")
            protected[path] = path.read_bytes()
        protected.update({path: path.read_bytes() for path in (root / "history").glob("*.json")})
        base = json.loads(Path(fixture["baseManifest"]).read_bytes())
        if changed_base:
            with tempfile.TemporaryDirectory(dir=output, prefix="new-base-") as scratch:
                base = build_base(mirror, Path(scratch), variant, version="2.0")
            channel = json.loads((mirror / "channels/main.json").read_bytes())
            channel["base"][variant] = {
                "id": base["id"],
                "locations": [
                    {"type": "mirror", "path": f"base/{variant}/{base['id'][7:]}/manifest.json"}
                ],
            }
            (mirror / "channels/main.json").write_text(json.dumps(channel))
        executable_release(mirror, base, "3.0", 73)
        before = allocated(root, store)
        samples = [before]
        rejected = 0
        stopped = Event()

        def sample() -> None:
            nonlocal rejected
            while not stopped.wait(0.05):
                try:
                    samples.append(allocated(root, store))
                except (OSError, subprocess.CalledProcessError):
                    rejected += 1

        sampler = Thread(target=sample)
        sampler.start()
        try:
            updated = run("update", "--yes")
        finally:
            stopped.set()
            sampler.join()
        assert updated["current"] == 3, updated
        assert sorted(path.name for path in (root / "generations").iterdir()) == ["2", "3"]
        after = allocated(root, store)
        samples.append(after)
        assert all(path.read_bytes() == data for path, data in protected.items())
        caches = run("gc", "--yes")
        assert all(path.read_bytes() == data for path, data in protected.items())
        return {
            "changedBase": changed_base,
            "beforeUpdateAllocatedBytes": before,
            "observedPeakAllocatedBytes": max(samples),
            "afterAutomaticCollectionAllocatedBytes": after,
            "afterExplicitCacheGCAllocatedBytes": allocated(root, store),
            "samples": len(samples),
            "rejectedConcurrentScans": rejected,
            "cleanupAttempt": updated.get("cleanupAttempt"),
            "explicitCacheGC": caches,
        }
    finally:
        if (root / "current").exists():
            run("stop", "--yes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "linux" and os.name != "nt":
        parser.error("this allocation measurement requires Linux/GNU du or native Windows")
    output = args.output.absolute()
    output.mkdir()
    report = {
        "sourceCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "filesystem": filesystem(output),
        "method": (
            (
                "Windows FileStandardInfo.AllocationSize for regular files over project+store; "
                "hardlinks counted once by volume/file identity. Directory metadata excluded. "
                if os.name == "nt"
                else "GNU du allocated blocks over project+store; hardlinks counted once. "
            )
            + "Shared/reflink extents are not deduplicated. Peak samples are non-atomic, "
            "not a continuous exact peak or filesystem free-space measurement."
        ),
        "scenarios": [
            scenario(output / name, changed)
            for name, changed in (("engine-only", False), ("changed-base", True))
        ],
    }
    (output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
