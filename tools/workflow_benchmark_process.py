"""Owned server processes and external, sampled memory evidence."""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil


class NvmlDevice:
    class Memory(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]

    def __init__(self, uuid: str):
        self.uuid = uuid
        self.lib = ctypes.CDLL("nvml.dll" if os.name == "nt" else "libnvidia-ml.so.1")
        signatures = {
            "nvmlInit_v2": [],
            "nvmlShutdown": [],
            "nvmlDeviceGetHandleByUUID": [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)],
            "nvmlDeviceGetUUID": [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint],
            "nvmlDeviceGetName": [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint],
            "nvmlSystemGetDriverVersion": [ctypes.c_char_p, ctypes.c_uint],
            "nvmlDeviceGetMemoryInfo": [ctypes.c_void_p, ctypes.POINTER(self.Memory)],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int

    def call(self, name: str, *arguments: Any) -> None:
        status = getattr(self.lib, name)(*arguments)
        if status:
            raise RuntimeError(f"{name} failed: NVML status {status}")

    def __enter__(self) -> NvmlDevice:
        self.call("nvmlInit_v2")
        try:
            self.handle = ctypes.c_void_p()
            self.call("nvmlDeviceGetHandleByUUID", self.uuid.encode(), ctypes.byref(self.handle))
            actual = ctypes.create_string_buffer(96)
            self.call("nvmlDeviceGetUUID", self.handle, actual, len(actual))
            if actual.value.decode() != self.uuid:
                raise ValueError("NVML returned another device")
        except BaseException:
            self.call("nvmlShutdown")
            raise
        return self

    def identity(self) -> dict[str, str]:
        name, driver = ctypes.create_string_buffer(96), ctypes.create_string_buffer(96)
        self.call("nvmlDeviceGetName", self.handle, name, len(name))
        self.call("nvmlSystemGetDriverVersion", driver, len(driver))
        return {
            "kind": "cuda",
            "uuid": self.uuid,
            "name": name.value.decode(),
            "driver": driver.value.decode(),
        }

    def read(self) -> dict[str, int]:
        memory = self.Memory()
        self.call("nvmlDeviceGetMemoryInfo", self.handle, ctypes.byref(memory))
        result = {name: int(getattr(memory, name)) for name in ("total", "free", "used")}
        if result["total"] <= 0 or max(result["used"], result["free"]) > result["total"]:
            raise ValueError("invalid NVML reading")
        return result

    def __exit__(self, *args: Any) -> None:
        self.call("nvmlShutdown")


class OwnedServer:
    """Use a private POSIX session, including orphaned workers, never a shared server."""

    process: subprocess.Popen[bytes]
    created: float | None

    def __init__(self, command: list[str], cwd: Path, env: dict[str, str], log: Path):
        if os.name != "posix":
            raise RuntimeError("workflow HTTP runner needs POSIX session containment")
        with log.open("xb") as output:
            self.process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.created = None
        with contextlib.suppress(psutil.NoSuchProcess):
            self.created = psutil.Process(self.process.pid).create_time()

    def members(self) -> list[psutil.Process]:
        # A worker can outlive its parent; ancestry alone is not a cleanup boundary.
        members = []
        for process in psutil.process_iter():
            with contextlib.suppress(psutil.NoSuchProcess, ProcessLookupError):
                if os.getsid(process.pid) == self.process.pid:
                    if (self.created is not None and process.create_time() < self.created) or (
                        process.pid == self.process.pid and process.create_time() != self.created
                    ):
                        raise RuntimeError("session ownership changed")
                    if process.status() != psutil.STATUS_ZOMBIE:
                        members.append(process)
        return members

    def assert_listener(self, port: int) -> None:
        for process in self.members():
            with contextlib.suppress(psutil.NoSuchProcess):
                if any(
                    c.status == psutil.CONN_LISTEN
                    and c.laddr.port == port
                    and c.laddr.ip in ("127.0.0.1", "0.0.0.0")
                    for c in process.net_connections(kind="inet")
                ):
                    return
        raise RuntimeError("HTTP listener is not owned by the benchmark session")

    def stop(self) -> bool:
        for sig, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
            deadline = time.monotonic() + timeout
            signalled = set()
            while True:
                members = self.members()
                if not members:
                    self.process.wait(timeout=2)
                    return True
                for process in members:
                    with contextlib.suppress(psutil.NoSuchProcess):
                        identity = (process.pid, process.create_time())
                        if identity not in signalled:
                            process.send_signal(sig)
                            signalled.add(identity)
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
        return not self.members()


class MemorySampler:
    """Sample tree RSS and optional UUID-bound device memory without execution hooks."""

    def __init__(
        self, server: OwnedServer, path: Path, gpu_uuid: str | None, interval: float = 0.1
    ):
        self.server, self.path, self.gpu_uuid, self.interval = server, path, gpu_uuid, interval
        self.stopping, self.ready = threading.Event(), threading.Event()
        self.errors: list[str] = []
        self.state: dict[str, Any] = {
            "samples_file": path.name,
            "requested_interval_seconds": interval,
            "sampled_peaks_are_lower_bounds": True,
            "sampled_peak_tree_rss_bytes": 0,
            "sampled_peak_device_used_bytes": None,
            "allocator_peak_bytes": None,
            "unload_residual_bytes": None,
            "count": 0,
            "maximum_gap_seconds": 0.0,
        }
        self.thread = threading.Thread(target=self._run, daemon=True, name="workflow-memory")

    def _run(self) -> None:
        try:
            context = NvmlDevice(self.gpu_uuid) if self.gpu_uuid else contextlib.nullcontext()
            with context as device, self.path.open("x", buffering=1) as output:
                previous = None
                while not self.stopping.is_set():
                    start = time.monotonic()
                    rss = 0
                    processes = []
                    for process in self.server.members():
                        with contextlib.suppress(psutil.NoSuchProcess):
                            size = process.memory_info().rss
                            rss += size
                            processes.append(
                                {
                                    "pid": process.pid,
                                    "created": process.create_time(),
                                    "rss_bytes": size,
                                }
                            )
                    memory = device.read() if device else None
                    row = {
                        "monotonic_seconds": start,
                        "epoch_ns": time.time_ns(),
                        "tree_rss_bytes": rss,
                        "processes": processes,
                        "device": memory,
                        "query_seconds": time.monotonic() - start,
                    }
                    output.write(json.dumps(row) + "\n")
                    self.state["sampled_peak_tree_rss_bytes"] = max(
                        rss, self.state["sampled_peak_tree_rss_bytes"]
                    )
                    if memory:
                        self.state["sampled_peak_device_used_bytes"] = max(
                            memory["used"], self.state["sampled_peak_device_used_bytes"] or 0
                        )
                    if previous is not None:
                        self.state["maximum_gap_seconds"] = max(
                            start - previous, self.state["maximum_gap_seconds"]
                        )
                    previous = start
                    self.state["count"] += 1
                    self.state["last_sample_monotonic_seconds"] = time.monotonic()
                    self.ready.set()
                    self.stopping.wait(max(0, self.interval - (time.monotonic() - start)))
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")
        finally:
            self.ready.set()

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(5) or self.errors or not self.state["count"]:
            raise RuntimeError(f"memory sampling unavailable: {self.errors}")

    def check(self) -> None:
        if self.errors or not self.thread.is_alive():
            raise RuntimeError(f"memory sampling stopped: {self.errors}")
        if time.monotonic() - self.state["last_sample_monotonic_seconds"] > 5:
            raise RuntimeError("memory sampling is stale; stopping the workload")

    def stop(self) -> dict[str, Any]:
        self.stopping.set()
        if self.thread.ident is not None:
            self.thread.join(5)
        return {
            **self.state,
            "errors": list(self.errors),
            "cleanup_verified": not self.thread.is_alive(),
        }
