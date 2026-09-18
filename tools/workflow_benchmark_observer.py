"""Source-hashed active allocator observation in Dinkster worker interpreters."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

CONTROL_ENV = "WORKFLOW_BENCHMARK_CONTROL"
RECORDS_ENV = "WORKFLOW_BENCHMARK_RECORDS"
SOURCE_HASH_ENV = "WORKFLOW_BENCHMARK_OBSERVER_SHA256"


def _completed_free_worker(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    device_map = row.get("deviceMap")
    consumers = row.get("consumers")
    mapping = device_map.get("mapping") if isinstance(device_map, dict) else None
    qualifier = device_map.get("qualifier") if isinstance(device_map, dict) else None
    return (
        isinstance(row.get("worker"), str)
        and bool(row["worker"])
        and isinstance(row.get("workerInstance"), str)
        and bool(row["workerInstance"])
        and row.get("status") == "complete"
        and "error" not in row
        and isinstance(mapping, dict)
        and all(
            isinstance(child, str) and bool(child) and isinstance(parent, str) and bool(parent)
            for child, parent in mapping.items()
        )
        and (qualifier is None or isinstance(qualifier, str) and bool(qualifier))
        and isinstance(consumers, list)
        and all(
            isinstance(consumer, dict)
            and isinstance(consumer.get("consumer"), str)
            and bool(consumer["consumer"])
            and consumer.get("status") == "complete"
            and "error" not in consumer
            for consumer in consumers
        )
        and len({consumer["consumer"] for consumer in consumers}) == len(consumers)
    )


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _append(path: Path, row: dict[str, Any]) -> None:
    data = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)


def _read_control(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _unlink_after_handle_release(path: Path) -> None:
    for attempt in range(100):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.01)


def _allocator_snapshot(
    reset: bool = False,
) -> tuple[dict[str, int] | None, str | None, str | None, str | None]:
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            module = torch.cuda
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            module = torch.xpu
        else:
            return None, None, None, "no supported allocator is available"
        device = module.current_device()
        physical_uuid = str(module.get_device_properties(device).uuid)
        if module is torch.cuda and physical_uuid and not physical_uuid.startswith("GPU-"):
            physical_uuid = "GPU-" + physical_uuid
        if not physical_uuid:
            return None, str(device), None, "allocator device has no physical UUID"
        module.synchronize(device)
        if reset:
            module.reset_peak_memory_stats(device)
        return (
            {
                "peak_allocated_bytes": int(module.max_memory_allocated(device)),
                "peak_reserved_bytes": int(module.max_memory_reserved(device)),
                "allocated_bytes": int(module.memory_allocated(device)),
                "reserved_bytes": int(module.memory_reserved(device)),
            },
            str(device),
            physical_uuid,
            None,
        )
    except Exception as error:
        return None, None, None, f"{type(error).__name__}: {error}"


def _install() -> None:
    control_name = os.environ.get(CONTROL_ENV)
    records_name = os.environ.get(RECORDS_ENV)
    expected_hash = os.environ.get(SOURCE_HASH_ENV)
    if not control_name or not records_name or not expected_hash:
        return
    control_path = Path(control_name)
    records_path = Path(records_name)
    source_hash = _digest(Path(__file__))
    worker_instance: str | None = None
    worker_name: str | None = None
    last_command: str | None = None
    worker_lock = threading.Lock()
    registration_lock = threading.Lock()

    def emit(event: str, **fields: Any) -> None:
        _append(
            records_path,
            {
                "event": event,
                "epoch_ns": time.time_ns(),
                "pid": os.getpid(),
                "process_instance": worker_instance,
                "source_sha256": source_hash,
                **fields,
            },
        )

    emit(
        "observer_loaded" if source_hash == expected_hash else "registration_error",
        role="unclassified",
        executable=sys.executable,
        source=str(Path(__file__).resolve()),
        error=None if source_hash == expected_hash else "observer source digest differs",
    )
    if source_hash != expected_hash:
        return
    try:
        from dinkster_workers import boundary
    except ImportError:
        return
    original_read = boundary.read_frame
    original_write = boundary.write_frame

    def service_control() -> None:
        nonlocal last_command
        command = _read_control(control_path)
        command_id = command.get("command_id")
        operation = command.get("operation")
        if (
            worker_instance is None
            or not isinstance(command_id, str)
            or command_id == last_command
            or operation not in ("reset", "read", "residue")
        ):
            return
        with worker_lock:
            if command_id == last_command:
                return
            counters, logical_device, physical_uuid, error = _allocator_snapshot(
                reset=operation == "reset"
            )
            emit(
                operation + "_ack",
                command_id=command_id,
                nonce=command.get("nonce"),
                job_ref=command.get("job_ref"),
                attempt=command.get("attempt"),
                worker=worker_name,
                requested_device=command.get("device"),
                logical_device=logical_device,
                physical_device_uuid=physical_uuid,
                visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                counters=counters,
                allocator_error=error,
            )
            last_command = command_id

    def control_loop() -> None:
        while True:
            service_control()
            time.sleep(0.01)

    def register_process(name: str | None, announced: Any) -> None:
        nonlocal worker_instance, worker_name
        with registration_lock:
            if worker_instance is not None:
                return
            from dinkster_values import process_instance_token

            worker_instance = process_instance_token()
            worker_name = name
        emit(
            "registration_ack" if announced == worker_instance else "registration_error",
            role="worker",
            worker=worker_name,
            announced_process_instance=announced,
            executable=sys.executable,
            source=str(Path(__file__).resolve()),
            error=None if announced == worker_instance else "hello worker instance differs",
        )
        service_control()
        threading.Thread(
            target=control_loop, name="workflow-allocator-observer", daemon=True
        ).start()

    def register_server_local() -> None:
        while worker_instance is None:
            if "dinkster_server.app" in sys.modules:
                from dinkster_values import process_instance_token

                register_process("local", process_instance_token())
                return
            time.sleep(0.01)

    async def read_frame(reader: Any) -> Any:
        result = await original_read(reader)
        if result is None:
            return result
        header, _ = result
        if header.get("type") == "invoke":
            service_control()
            emit(
                "invocation_ack",
                nonce=_read_control(control_path).get("nonce"),
                job_ref=header.get("jobRef"),
                invocation_id=header.get("invocationId"),
                attempt=header.get("attemptId"),
                node_id=header.get("nodeId"),
                worker=worker_name,
            )
        return result

    async def write_frame(writer: Any, header: dict[str, Any], blobs: Any) -> None:
        if header.get("type") == "hello":
            worker_name_raw = header.get("pack")
            register_process(
                worker_name_raw if isinstance(worker_name_raw, str) else None,
                header.get("workerInstance"),
            )
        await original_write(writer, header, blobs)

    boundary.read_frame = read_frame
    boundary.write_frame = write_frame
    threading.Thread(
        target=register_server_local,
        name="workflow-allocator-local-registration",
        daemon=True,
    ).start()


@dataclass
class ObserverInstallation:
    source_sha256: str
    control_path: Path
    records_path: Path
    pth_path: Path

    @classmethod
    def create(cls, python: Path, directory: Path) -> ObserverInstallation:
        source = Path(__file__).resolve()
        code = "import json,sysconfig; print(json.dumps(sysconfig.get_paths()))"
        paths = json.loads(
            subprocess.check_output([str(python), "-I", "-c", code], text=True, timeout=30)
        )
        purelib = Path(paths["purelib"])
        if not purelib.is_dir():
            raise ValueError("server interpreter has no writable site-packages directory")
        pth = purelib / f"workflow_benchmark_{secrets.token_hex(12)}.pth"
        pth.write_text(str(source.parent) + "\nimport workflow_benchmark_observer\n")
        return cls(
            source_sha256=_digest(source),
            control_path=directory / "allocator-control.json",
            records_path=directory / "allocator-records.jsonl",
            pth_path=pth,
        )

    def environment(self) -> dict[str, str]:
        return {
            CONTROL_ENV: str(self.control_path),
            RECORDS_ENV: str(self.records_path),
            SOURCE_HASH_ENV: self.source_sha256,
        }

    def records(self) -> list[dict[str, Any]]:
        if not self.records_path.exists():
            return []
        return [json.loads(line) for line in self.records_path.read_text().splitlines()]

    def registrations(self) -> list[dict[str, Any]]:
        return [row for row in self.records() if row.get("event") == "registration_ack"]

    def live_registrations(self) -> list[dict[str, Any]]:
        live = []
        for row in self.registrations():
            pid = row.get("pid")
            if type(pid) is not int or pid <= 0:
                continue
            if not psutil.pid_exists(pid):
                continue
            live.append(row)
        return live

    def wait_for_registration(self, worker: str, timeout: float = 30) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            matches = [row for row in self.live_registrations() if row.get("worker") == worker]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                break
            time.sleep(0.01)
        raise RuntimeError(f"worker observer registration is missing or duplicate: {worker}")

    def command(
        self,
        operation: str,
        *,
        nonce: str,
        device: dict[str, Any],
        expected_instances: set[str],
        job_ref: str | None = None,
        attempt: int | None = None,
        timeout: float = 30,
    ) -> list[dict[str, Any]]:
        command_id = secrets.token_hex(24)
        command = {
            "command_id": command_id,
            "operation": operation,
            "nonce": nonce,
            "device": device,
            "job_ref": job_ref,
            "attempt": attempt,
        }
        temporary = self.control_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(command, sort_keys=True))
        temporary.replace(self.control_path)
        deadline = time.monotonic() + timeout
        event = operation + "_ack"
        while time.monotonic() < deadline:
            rows = [
                row
                for row in self.records()
                if row.get("command_id") == command_id and row.get("event") == event
            ]
            actual = {
                instance for row in rows if isinstance(instance := row.get("process_instance"), str)
            }
            if actual == expected_instances and len(rows) == len(expected_instances):
                return rows
            time.sleep(0.01)
        raise TimeoutError(f"workers did not acknowledge allocator {operation}: {command_id}")

    def begin(
        self, device: dict[str, Any], *, require_existing: bool = False, timeout: float = 30
    ) -> str:
        nonce = secrets.token_hex(24)
        registrations = self.live_registrations()
        instances = {
            instance
            for row in registrations
            if isinstance(instance := row.get("process_instance"), str)
        }
        if len(instances) != len(registrations):
            raise RuntimeError("worker observer registrations are duplicate or malformed")
        if require_existing and not instances:
            raise RuntimeError("warm allocator reset has no live registered workers")
        self.command(
            "reset",
            nonce=nonce,
            device=device,
            expected_instances=instances,
            timeout=timeout,
        )
        return nonce

    def close(self) -> None:
        _unlink_after_handle_release(self.pth_path)
        _unlink_after_handle_release(self.control_path)


def _valid_counters(counters: Any) -> bool:
    names = (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "allocated_bytes",
        "reserved_bytes",
    )
    return isinstance(counters, dict) and all(
        type(counters.get(name)) is int and counters[name] >= 0 for name in names
    )


def _valid_allocator_ack(row: dict[str, Any], device: dict[str, Any]) -> bool:
    return (
        row.get("allocator_error") is None
        and _valid_counters(row.get("counters"))
        and row.get("requested_device") == device
        and isinstance(row.get("logical_device"), str)
        and bool(row["logical_device"])
        and row.get("physical_device_uuid") == device.get("uuid")
    )


def validate_window(
    records: list[dict[str, Any]],
    *,
    nonce: str,
    job_ref: str,
    attempt: int | None,
    device: dict[str, Any],
    all_cached: bool,
    source_sha256: str,
) -> dict[str, Any]:
    window = [row for row in records if row.get("nonce") == nonce]
    registrations = [row for row in records if row.get("event") == "registration_ack"]
    resets = [row for row in window if row.get("event") == "reset_ack"]
    reads = [row for row in window if row.get("event") == "read_ack"]
    invocations = [row for row in window if row.get("event") == "invocation_ack"]
    problems: list[str] = []
    if any(str(row.get("event", "")).endswith("_error") for row in window):
        problems.append("worker observer reported an error")
    if any(row.get("source_sha256") != source_sha256 for row in window):
        problems.append("worker observer record has another source identity")
    registered = [row.get("process_instance") for row in registrations]
    reset_instances = [row.get("process_instance") for row in resets]
    read_instances = [row.get("process_instance") for row in reads]
    if (
        not registrations
        or len(set(registered)) != len(registered)
        or any(row.get("source_sha256") != source_sha256 for row in registrations)
        or any(row.get("event") == "registration_error" for row in records)
    ):
        problems.append("worker observer registrations are missing, duplicate, or invalid")
    if (
        not resets
        or set(reset_instances) != set(read_instances)
        or len(set(reset_instances)) != len(reset_instances)
        or len(set(read_instances)) != len(read_instances)
        or any(instance not in registered for instance in reset_instances)
    ):
        problems.append("active reset/read acknowledgements are missing, duplicate, or restarted")
    invocation_ids = [row.get("invocation_id") for row in invocations]
    if all_cached:
        if invocations:
            problems.append("cached job has unexpected worker invocation acknowledgements")
    elif not invocations:
        problems.append("non-cached job has no worker invocation acknowledgement")
    if invocations and (
        any(not isinstance(value, str) or not value for value in invocation_ids)
        or len(set(invocation_ids)) != len(invocation_ids)
        or any(row.get("process_instance") not in reset_instances for row in invocations)
        or any(
            row.get("job_ref") != job_ref or row.get("attempt") != attempt for row in invocations
        )
    ):
        problems.append("worker invocations are duplicate, unregistered, or misattributed")
    for row in resets:
        if row.get("job_ref") is not None or row.get("attempt") is not None:
            problems.append("pre-job reset acknowledgement carries terminal identity")
        if row.get("requested_device") != device:
            problems.append("pre-job reset carries another requested device")
        if not _valid_allocator_ack(row, device):
            problems.append("pre-job reset has invalid allocator or device evidence")
    for row in reads:
        if row.get("job_ref") != job_ref or row.get("attempt") != attempt:
            problems.append("terminal read belongs to another job or attempt")
        if row.get("requested_device") != device:
            problems.append("terminal read carries another requested device")
        if not _valid_allocator_ack(row, device):
            problems.append("terminal read has invalid allocator or device evidence")
    complete_allocator_evidence = bool(resets) and all(
        _valid_allocator_ack(row, device) for row in [*resets, *reads]
    )
    worker_peaks = {
        str(row["process_instance"]): {
            "peak_allocated_bytes": row["counters"]["peak_allocated_bytes"],
            "peak_reserved_bytes": row["counters"]["peak_reserved_bytes"],
        }
        for row in reads
        if complete_allocator_evidence
    }
    return {
        "nonce": nonce,
        "job_ref": job_ref,
        "attempt": attempt,
        "status": "invalid" if problems else "acknowledged",
        "raw_records": window,
        "registrations": registrations,
        "workers": sorted(str(instance) for instance in set(reset_instances)),
        "worker_peaks": worker_peaks,
        "peak_allocated_bytes": (
            sum(row["peak_allocated_bytes"] for row in worker_peaks.values())
            if worker_peaks
            else None
        ),
        "peak_reserved_bytes": (
            sum(row["peak_reserved_bytes"] for row in worker_peaks.values())
            if worker_peaks
            else None
        ),
        "problems": problems,
    }


def validate_free_response(
    response: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    request_id: str,
    nonce: str,
    device: dict[str, Any],
    windows: list[dict[str, Any]],
    source_sha256: str,
) -> dict[str, Any]:
    problems: list[str] = []
    workers = response.get("workers")
    if (
        response.get("requestId") != request_id
        or response.get("completed") is not True
        or response.get("queuePaused") is not True
        or not isinstance(workers, list)
        or not workers
    ):
        problems.append("full-free response is incomplete, stale, or did not keep the queue paused")
        workers = []
    instances = [row.get("workerInstance") for row in workers if isinstance(row, dict)]
    valid_instances = len(instances) == len(workers) and all(
        isinstance(instance, str) and instance for instance in instances
    )
    if not valid_instances:
        problems.append("full-free worker instances are missing")
    for row in workers:
        if not _completed_free_worker(row):
            problems.append("full-free worker or consumer is incomplete")
    registration_rows = [
        row
        for row in records
        if row.get("event") == "registration_ack" and row.get("source_sha256") == source_sha256
    ]
    registrations = {row.get("process_instance"): row for row in registration_rows}
    residue_rows = [
        row for row in records if row.get("event") == "residue_ack" and row.get("nonce") == nonce
    ]
    residues = {row.get("process_instance"): row for row in residue_rows}
    expected = set(instances) if valid_instances else set()
    if (
        len(registration_rows) != len(registrations)
        or len(residue_rows) != len(residues)
        or any(row.get("source_sha256") != source_sha256 for row in residue_rows)
        or set(registrations) != expected
        or set(residues) != expected
    ):
        problems.append("full-free workers differ from source-hashed observer acknowledgements")
    if any(set(window.get("workers", [])) != expected for window in windows):
        problems.append("job windows do not cover the complete full-free worker set")
    worker_names: dict[str, set[str]] = {}
    for worker in workers:
        if (
            isinstance(worker, dict)
            and isinstance(worker.get("workerInstance"), str)
            and isinstance(worker.get("worker"), str)
        ):
            worker_names.setdefault(worker["workerInstance"], set()).add(worker["worker"])
    if any(
        registration.get("worker") not in worker_names.get(instance, set())
        for instance, registration in registrations.items()
    ):
        problems.append("observer worker name differs from the full-free response")
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        instance = worker.get("workerInstance")
        residue = residues.get(instance, {})
        device_map = worker.get("deviceMap")
        if not isinstance(device_map, dict) or not isinstance(device_map.get("mapping"), dict):
            problems.append("full-free worker has no device map")
            continue
        mapping = device_map["mapping"]
        qualifier = device_map.get("qualifier")
        logical = residue.get("logical_device")
        physical = residue.get("physical_device_uuid")
        child = f"cuda:{logical}"
        parent = mapping.get(child, child if qualifier is None else None)
        if (
            not _valid_allocator_ack(residue, device)
            or physical != device.get("uuid")
            or parent != "cuda:0"
            or qualifier is not None
        ):
            problems.append("worker device map differs from physical residue evidence")
    valid_residues = not problems and all(
        _valid_allocator_ack(row, device) for row in residues.values()
    )
    residue_counters = [row["counters"] for row in residues.values()] if valid_residues else []
    return {
        "request": {"requestId": request_id},
        "response": response,
        "residue_records": list(residues.values()),
        "residual_allocated_bytes": (
            sum(row["allocated_bytes"] for row in residue_counters) if residue_counters else None
        ),
        "residual_reserved_bytes": (
            sum(row["reserved_bytes"] for row in residue_counters) if residue_counters else None
        ),
        "problems": problems,
    }


_install()
