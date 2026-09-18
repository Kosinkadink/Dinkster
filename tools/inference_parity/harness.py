"""Run and gate one pinned inference workload on ComfyUI and Dinkster.

The engine adapter is a fresh subprocess with a two-line JSON protocol. The
same process receives exactly ``warmup`` then ``real`` so model residency is
retained. This outer process owns cross-engine pin validation, process-tree
RSS and device sampling, output comparison, and deterministic records. Dinkster's
engine-side event measurements remain owned by ``dinkster.benchmark``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import signal
import statistics
import struct
import subprocess
import sys
import threading
import time
import zlib
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = 1
PHASES = ("warmup", "real")
ENGINES = ("comfyui", "dinkster")
TIMING_ENGINE_ORDER = ("comfyui", "dinkster", "dinkster", "comfyui")
TIMING_PHASES = ("cold", "resident-warmup", "warm-1", "warm-2")
TIMING_WARM_PHASES = ("warm-1", "warm-2")
TIMING_REGRESSION_WARN_RATIO = 1.10
ADAPTER_REPLY_TIMEOUT_SECONDS = 30.0
ADAPTER_EXIT_TIMEOUT_SECONDS = 30.0


class HarnessError(RuntimeError):
    """A pin, protocol, schema, or acceptance contract was refused."""


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode()


def write_json(path: Path, value: object) -> None:
    data = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise HarnessError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _validate_safetensors_contract(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Validate one exact safetensors header without importing model code."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise HarnessError(f"safetensors prefix is truncated: {path}")
        (header_length,) = struct.unpack("<Q", prefix)
        if header_length == 0 or header_length > 100_000_000:
            raise HarnessError(f"safetensors header length is invalid: {header_length}")
        if 8 + header_length > size:
            raise HarnessError("safetensors header overruns the checkpoint")
        header_bytes = stream.read(header_length)
    try:
        raw = json.loads(header_bytes, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HarnessError(f"safetensors header is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise HarnessError("safetensors header must be an object")
    payload_size = size - 8 - header_length
    ranges: list[tuple[int, int, str]] = []
    tensors: dict[str, dict[str, Any]] = {}
    dtype_bits = {
        "BOOL": 8,
        "U8": 8,
        "I8": 8,
        "I16": 16,
        "U16": 16,
        "F16": 16,
        "BF16": 16,
        "I32": 32,
        "U32": 32,
        "F32": 32,
        "F64": 64,
        "I64": 64,
        "U64": 64,
        "F8_E4M3": 8,
        "F8_E5M2": 8,
    }
    for key, item in raw.items():
        if key == "__metadata__":
            if not isinstance(item, dict) or not all(
                isinstance(name, str) and isinstance(value, str) for name, value in item.items()
            ):
                raise HarnessError("safetensors metadata must map strings to strings")
            continue
        if not isinstance(key, str) or not isinstance(item, dict):
            raise HarnessError("safetensors tensor entries must be named objects")
        if set(item) != {"data_offsets", "dtype", "shape"}:
            raise HarnessError(f"safetensors tensor {key} has an invalid schema")
        dtype = item["dtype"]
        shape = item["shape"]
        offsets = item["data_offsets"]
        if (
            dtype not in dtype_bits
            or not isinstance(shape, list)
            or not all(
                isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape
            )
        ):
            raise HarnessError(f"safetensors tensor {key} has invalid dtype/shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(
                isinstance(offset, int) and not isinstance(offset, bool) for offset in offsets
            )
        ):
            raise HarnessError(f"safetensors tensor {key} has invalid offsets")
        begin, end = offsets
        if begin < 0 or end < begin or end > payload_size:
            raise HarnessError(f"safetensors tensor {key} offsets are out of range")
        elements = math.prod(shape)
        expected_bytes = (elements * dtype_bits[dtype] + 7) // 8
        if end - begin != expected_bytes:
            raise HarnessError(f"safetensors tensor {key} byte geometry is inconsistent")
        ranges.append((begin, end, key))
        tensors[key] = item
    ranges.sort()
    expected_begin = 0
    previous_key = "payload start"
    for begin, end, key in ranges:
        if begin != expected_begin:
            relation = "overlap" if begin < expected_begin else "gap"
            raise HarnessError(
                f"safetensors tensor ranges have a {relation} after {previous_key}: {key}"
            )
        expected_begin = end
        previous_key = key
    if expected_begin != payload_size:
        raise HarnessError("safetensors tensor ranges do not cover the complete payload")
    required = contract.get("required_tensors")
    if not isinstance(required, dict):
        raise HarnessError("safetensors contract has no required_tensors")
    for key, expected in required.items():
        actual = tensors.get(key)
        if not isinstance(expected, dict) or actual is None:
            raise HarnessError(f"required safetensors tensor is absent: {key}")
        if actual["dtype"] != expected.get("dtype") or actual["shape"] != expected.get("shape"):
            raise HarnessError(
                f"required safetensors tensor geometry mismatch: {key}: "
                f"expected {expected}, got dtype={actual['dtype']} shape={actual['shape']}"
            )
    return {
        "header_length": header_length,
        "payload_size": payload_size,
        "tensor_count": len(tensors),
    }


def _png_rgba_contract(path: Path) -> dict[str, Any]:
    """Read PNG structure and alpha bytes using only the standard library."""
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HarnessError(f"fixture is not PNG: {path}")
    offset = 8
    width = height = bit_depth = color_type = interlace = None
    compressed = bytearray()
    while offset < len(data):
        if offset + 12 > len(data):
            raise HarnessError("PNG chunk is truncated")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        if len(chunk) != length or offset + 12 + length > len(data):
            raise HarnessError("PNG chunk overruns the fixture")
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if compression != 0 or filtering != 0:
                raise HarnessError("PNG uses an unsupported compression/filter method")
        elif chunk_type == b"IDAT":
            compressed.extend(chunk)
        elif chunk_type == b"IEND":
            break
        offset += 12 + length
    if (bit_depth, color_type, interlace) != (8, 6, 0) or width is None or height is None:
        raise HarnessError("fixture must be non-interlaced 8-bit RGBA PNG")
    decoded = zlib.decompress(bytes(compressed))
    stride = width * 4
    if len(decoded) != height * (stride + 1):
        raise HarnessError("PNG decoded byte count does not match IHDR")
    previous = bytearray(stride)
    alphas_below_128 = 0
    cursor = 0
    for _row in range(height):
        filter_type = decoded[cursor]
        source = decoded[cursor + 1 : cursor + 1 + stride]
        cursor += stride + 1
        current = bytearray(stride)
        for index, value in enumerate(source):
            left = current[index - 4] if index >= 4 else 0
            up = previous[index]
            upper_left = previous[index - 4] if index >= 4 else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = up
            elif filter_type == 3:
                predictor = (left + up) // 2
            elif filter_type == 4:
                p = left + up - upper_left
                distances = (abs(p - left), abs(p - up), abs(p - upper_left))
                predictor = (left, up, upper_left)[distances.index(min(distances))]
            else:
                raise HarnessError(f"PNG uses unknown filter type {filter_type}")
            current[index] = (value + predictor) & 0xFF
        alphas_below_128 += sum(alpha < 128 for alpha in current[3::4])
        previous = current
    return {
        "alpha_mask_below_128_percent": format(alphas_below_128 * 100 / (width * height), ".3f"),
        "mode": "RGBA",
        "size": f"{width}x{height}",
    }


def validate_sdxl_inpaint_preflight(
    workload: dict[str, Any], roots: dict[str, Path]
) -> dict[str, Any]:
    """Run the W0 SDXL-inpaint byte/header gates before engine imports."""
    contract = workload.get("preflight")
    if workload.get("id") != "W0-SDXL-INPAINT" or not isinstance(contract, dict):
        raise HarnessError("SDXL inpaint preflight contract is absent")
    checkpoint = roots["artifact"] / workload["execution"]["checkpoint"]
    header = _validate_safetensors_contract(checkpoint, contract["safetensors"])
    fixture = roots["artifact"] / "input" / workload["execution"]["input_image"]
    fixture_contract = _png_rgba_contract(fixture)
    if fixture_contract != contract.get("fixture"):
        raise HarnessError(
            f"inpaint fixture geometry/alpha mismatch: expected {contract.get('fixture')}, "
            f"got {fixture_contract}"
        )
    files = contract.get("source_files")
    if not isinstance(files, dict):
        raise HarnessError("SDXL inpaint source-file pins are absent")
    for relative, expected in files.items():
        path = Path(__file__).parents[2] / relative
        if digest_file(path) != expected:
            raise HarnessError(f"source-file digest mismatch: {relative}")
    interpreter = contract.get("interpreter")
    if not isinstance(interpreter, dict):
        raise HarnessError("SDXL inpaint interpreter pin is absent")
    executable = Path(interpreter.get("path", ""))
    if executable.resolve() != Path(interpreter.get("realpath", "")):
        raise HarnessError("SDXL inpaint interpreter realpath mismatch")
    expected_adapters = {
        "comfyui": "sdxl_inpaint_comfyui_adapter.py",
        "dinkster": "sdxl_inpaint_dinkster_adapter.py",
    }
    for engine, expected_adapter in expected_adapters.items():
        engine_pin = workload.get("engines", {}).get(engine, {})
        engine_python = Path(engine_pin.get("python", "").format(root=str(roots[engine])))
        if engine_python.resolve() != executable.resolve():
            raise HarnessError(f"{engine} engine does not use the pinned interpreter")
        adapter = engine_pin.get("adapter")
        if adapter != expected_adapter:
            raise HarnessError(f"{engine} engine does not use the pinned SDXL adapter")
        adapter_source = f"tools/inference_parity/{adapter}"
        if adapter_source not in files:
            raise HarnessError(f"{engine} adapter is absent from source-file pins")
    if digest_file(executable.resolve()) != interpreter.get("digest"):
        raise HarnessError("SDXL inpaint interpreter digest mismatch")
    version = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True, check=False
    )
    observed_version = (version.stdout or version.stderr).strip()
    if version.returncode or observed_version != interpreter.get("python_version"):
        raise HarnessError("SDXL inpaint interpreter version mismatch")
    torch_version = interpreter.get("torch_version")
    if not isinstance(torch_version, str):
        raise HarnessError("SDXL inpaint torch build pin is absent")
    metadata = subprocess.run(
        [
            str(executable),
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('torch'))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if metadata.returncode or metadata.stdout.strip() != torch_version:
        raise HarnessError("SDXL inpaint installed torch distribution mismatch")
    return {"fixture": fixture_contract, "safetensors": header}


def phase_requests(seed: int) -> tuple[dict[str, Any], ...]:
    """The complete protocol: exactly two calls and the same RNG seed."""
    return tuple({"phase": phase, "seed": seed} for phase in PHASES)


def timing_requests(seed: int) -> tuple[dict[str, Any], ...]:
    """One cold, one unrecorded warmup, and two recorded warm requests."""
    return tuple({"phase": phase, "seed": seed} for phase in TIMING_PHASES)


def workload_timing_requests(workload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    execution = workload.get("execution", {})
    timing = execution.get("timing")
    if timing is None:
        return timing_requests(execution.get("seed", 0))
    seed = execution["seed"]
    discarded = timing["discarded_warmups"]
    measured = timing["measured_repeats"]
    change = timing["settings_change"]
    requests: list[dict[str, Any]] = [{"phase": "cold", "seed": seed}]
    requests.extend(
        {"phase": f"discard-{index}", "record": False, "seed": seed}
        for index in range(1, discarded + 1)
    )
    requests.extend(
        {"phase": f"repeat-{index}", "scenario": "plain", "seed": seed}
        for index in range(1, measured + 1)
    )
    requests.extend(
        {
            "cfg": change["cfg"],
            "phase": f"settings-change-{index}",
            "scenario": "settings-change",
            "seed": seed,
        }
        for index in range(1, change["repeats"] + 1)
    )
    return tuple(requests)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HarnessError(f"{path}: top level must be an object")
    return value


def _required(mapping: dict[str, Any], key: str, kind: type, where: str) -> Any:
    value = mapping.get(key)
    if not isinstance(value, kind):
        raise HarnessError(f"{where}.{key} must be {kind.__name__}")
    return value


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise HarnessError(f"manifest.schema must be {SCHEMA}")
    workloads = _required(manifest, "workloads", list, "manifest")
    ids: set[str] = set()
    for index, item in enumerate(workloads):
        where = f"manifest.workloads[{index}]"
        if not isinstance(item, dict):
            raise HarnessError(f"{where} must be object")
        workload_id = _required(item, "id", str, where)
        if not workload_id.isascii() or workload_id in ids:
            raise HarnessError(f"{where}.id must be unique ASCII")
        ids.add(workload_id)
        _required(item, "purpose", str, where)
        _required(item, "graph", dict, where)
        artifacts = _required(item, "artifacts", list, where)
        execution = _required(item, "execution", dict, where)
        engines = _required(item, "engines", dict, where)
        timing = execution.get("timing")
        if timing is not None:
            if not isinstance(timing, dict):
                raise HarnessError(f"{where}.execution.timing must be object")
            for key in ("discarded_warmups", "measured_repeats"):
                value = timing.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise HarnessError(f"{where}.execution.timing.{key} must be positive integer")
            change = timing.get("settings_change")
            cfg = change.get("cfg") if isinstance(change, dict) else None
            if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not math.isfinite(cfg):
                raise HarnessError(f"{where}.execution.timing.settings_change is invalid")
            repeats = change.get("repeats")
            if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
                raise HarnessError(
                    f"{where}.execution.timing.settings_change.repeats must be at least 2"
                )
        for engine, pin in engines.items():
            if engine not in ENGINES or not isinstance(pin, dict):
                raise HarnessError(f"{where}.engines entries must be supported engine objects")
            for key in ("adapter", "commit", "python"):
                _required(pin, key, str, f"{where}.engines.{engine}")
            extensions = pin.get("extensions", [])
            if not isinstance(extensions, list):
                raise HarnessError(f"{where}.engines.{engine}.extensions must be list")
            for extension in extensions:
                if not isinstance(extension, dict):
                    raise HarnessError(
                        f"{where}.engines.{engine}.extensions entries must be objects"
                    )
                for key in ("commit", "path", "repository"):
                    _required(extension, key, str, f"{where}.engines.{engine}.extension")
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise HarnessError(f"{where}.artifacts entries must be objects")
            for key in ("role", "path", "digest"):
                _required(artifact, key, str, f"{where}.artifact")
            if not isinstance(artifact.get("bytes"), int) or artifact["bytes"] < 0:
                raise HarnessError(f"{where}.artifact.bytes must be nonnegative integer")


def validate_acceptance(acceptance: dict[str, Any]) -> None:
    if acceptance.get("schema") != SCHEMA:
        raise HarnessError(f"acceptance.schema must be {SCHEMA}")
    workloads = _required(acceptance, "workloads", dict, "acceptance")
    for workload_id, rules in workloads.items():
        if not isinstance(workload_id, str) or not isinstance(rules, dict):
            raise HarnessError("acceptance workloads must map string ids to objects")
        phases = _required(rules, "phases", dict, f"acceptance.{workload_id}")
        if tuple(sorted(phases)) != tuple(sorted(PHASES)):
            raise HarnessError(f"acceptance.{workload_id}.phases must contain warmup and real")
        status = rules.get("status", "active")
        if status not in ("active", "calibration_pending"):
            raise HarnessError(
                f"acceptance.{workload_id}.status must be active or calibration_pending"
            )


def _git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise HarnessError(f"git {' '.join(args)} failed for {root}: {result.stderr.strip()}")
    return result.stdout.strip()


def validate_pins(
    workload: dict[str, Any], engine: str, roots: dict[str, Path], acceptance_digest: str
) -> dict[str, Any]:
    engine_pin = workload["engines"][engine]
    root = roots[engine]
    actual = _git_output(root, "rev-parse", "HEAD")
    expected = engine_pin["commit"]
    if expected != "HEAD" and actual != expected:
        raise HarnessError(f"{engine} commit mismatch: expected {expected}, got {actual}")
    if _git_output(root, "status", "--porcelain"):
        raise HarnessError(f"{engine} checkout must be clean: {root}")
    extensions = []
    for extension in engine_pin.get("extensions", []):
        relative = Path(extension["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise HarnessError(f"{engine} extension path must be a contained relative path")
        extension_root = root / relative
        extension_commit = _git_output(extension_root, "rev-parse", "HEAD")
        if extension_commit != extension["commit"]:
            raise HarnessError(
                f"{engine} extension commit mismatch: expected {extension['commit']}, "
                f"got {extension_commit}"
            )
        if _git_output(extension_root, "status", "--porcelain"):
            raise HarnessError(f"{engine} extension checkout must be clean: {extension_root}")
        extensions.append({**extension, "resolved_path": str(extension_root)})
    graph = workload["graph"][engine]
    graph_kind = graph.get("root")
    if graph_kind == "artifact":
        graph_commit = graph["commit"]
        graph_path = roots["artifact"] / graph["path"]
    else:
        graph_root = (
            roots["template"]
            if graph_kind == "template"
            else (roots["dinkster"] if graph_kind == "harness" else root)
        )
        graph_commit = _git_output(graph_root, "rev-parse", "HEAD")
        if graph["commit"] != "HEAD" and graph_commit != graph["commit"]:
            raise HarnessError(
                f"{engine} graph source commit mismatch: expected"
                f" {graph['commit']}, got {graph_commit}"
            )
        graph_path = graph_root / graph["path"]
    if digest_file(graph_path) != graph["digest"]:
        raise HarnessError(f"{engine} graph digest mismatch: {graph_path}")
    if workload["acceptance_manifest"]["digest"] != acceptance_digest:
        raise HarnessError("workload acceptance-manifest digest mismatch")
    artifact_root = roots["artifact"]
    pins = []
    for artifact in workload["artifacts"]:
        path = artifact_root / artifact["path"]
        if not path.is_file() or path.stat().st_size != artifact["bytes"]:
            raise HarnessError(f"artifact size mismatch: {path}")
        actual_digest = digest_file(path)
        if actual_digest != artifact["digest"]:
            raise HarnessError(f"artifact digest mismatch: {path}")
        pins.append({**artifact, "resolved_path": str(path)})
    result = {
        "artifacts": pins,
        "engine_commit": actual,
        "graph_commit": graph_commit,
        "graph_digest": graph["digest"],
    }
    if extensions:
        result["extensions"] = extensions
    if workload.get("id") == "W0-SDXL-INPAINT":
        result["workload_digest"] = digest_bytes(canonical_bytes(workload))
        result["preflight"] = validate_sdxl_inpaint_preflight(workload, roots)
    return result


def validate_extensions_unchanged(pins: dict[str, Any], engine: str, *, activity: str) -> None:
    for extension in pins.get("extensions", []):
        root = Path(extension["resolved_path"])
        if _git_output(root, "rev-parse", "HEAD") != extension["commit"]:
            raise HarnessError(f"{engine} extension checkout changed during {activity}: {root}")
        if _git_output(root, "status", "--porcelain"):
            raise HarnessError(
                f"{engine} extension checkout became dirty during {activity}: {root}"
            )


def hardware_inventory() -> dict[str, Any]:
    from dinkster_memory import system_memory_snapshot

    memory = system_memory_snapshot()
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise HarnessError(f"nvidia-smi inventory failed: {result.stderr.strip()}")
    gpus = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise HarnessError(f"unexpected nvidia-smi inventory line: {line}")
        gpus.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
                "memory_total_mib": int(fields[3]),
                "driver": fields[4],
            }
        )
    return {
        "cpu": platform.processor() or platform.machine(),
        "gpus": gpus,
        "platform": platform.platform(),
        "ram_bytes": memory.host_total_bytes,
    }


def validate_hardware(workload: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    expected = workload["execution"]["hardware"]
    actual_gpus = inventory["gpus"]
    if len(actual_gpus) != expected["gpu_count"]:
        raise HarnessError("GPU inventory count mismatch")
    if [gpu["name"] for gpu in actual_gpus] != expected["gpu_names"]:
        raise HarnessError("GPU inventory name mismatch")
    if "gpu_uuids" in expected and [gpu["uuid"] for gpu in actual_gpus] != expected["gpu_uuids"]:
        raise HarnessError("GPU inventory UUID mismatch")
    if "driver" in expected and any(gpu["driver"] != expected["driver"] for gpu in actual_gpus):
        raise HarnessError("GPU inventory driver mismatch")
    if "memory_total_mib" in expected and any(
        gpu["memory_total_mib"] != expected["memory_total_mib"] for gpu in actual_gpus
    ):
        raise HarnessError("GPU inventory memory mismatch")
    ordinal = workload["execution"]["device_ordinal"]
    if (
        not isinstance(ordinal, int)
        or isinstance(ordinal, bool)
        or not 0 <= ordinal < len(actual_gpus)
    ):
        raise HarnessError("GPU device ordinal is invalid")
    uuid = actual_gpus[ordinal]["uuid"]
    if uuid != expected["device_uuid"]:
        raise HarnessError(f"GPU UUID mismatch: expected {expected['device_uuid']}, got {uuid}")
    return {"device_ordinal": ordinal, "device_uuid": uuid, "inventory": inventory}


def validate_timing_hardware(
    workload: dict[str, Any],
    inventory: dict[str, Any],
    device_uuid: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Bind timing to one explicit GPU without inheriting an old station topology."""
    ordinal = workload["execution"]["device_ordinal"]
    if ordinal != expected["device_ordinal"]:
        raise HarnessError("timing workload device ordinal does not match the immutable pin")
    if device_uuid != expected["device_uuid"]:
        raise HarnessError("timing CLI GPU UUID does not match the immutable pin")
    hostname = platform.node()
    if hostname != expected["hostname"]:
        raise HarnessError("timing hostname does not match the immutable pin")
    actual_gpus = inventory["gpus"]
    if actual_gpus != expected["gpus"]:
        raise HarnessError("timing GPU inventory does not match the immutable pin")
    if ordinal < 0 or ordinal >= len(actual_gpus):
        raise HarnessError(f"timing GPU ordinal {ordinal} is absent")
    actual_uuid = actual_gpus[ordinal]["uuid"]
    if actual_uuid != device_uuid:
        raise HarnessError(f"timing GPU UUID mismatch: expected {device_uuid}, got {actual_uuid}")
    return {
        "device_ordinal": ordinal,
        "device_uuid": actual_uuid,
        "inventory": {**inventory, "hostname": hostname},
    }


def require_device_idle(ordinal: int) -> None:
    """Fail closed unless the designated timing GPU has no compute process."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={ordinal}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except subprocess.TimeoutExpired as exc:
        raise HarnessError("timing GPU process query timed out") from exc
    if result.returncode:
        raise HarnessError(f"timing GPU process query failed: {result.stderr.strip()}")
    pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if pids:
        raise HarnessError(f"timing GPU {ordinal} is not idle; compute PIDs: {', '.join(pids)}")


def _tree_rss(pid: int) -> int:
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            parents[int(entry.name)] = int(fields[3])
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss[int(entry.name)] = int(line.split()[1]) * 1024
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    members = {pid}
    changed = True
    while changed:
        changed = False
        for child, parent in parents.items():
            if parent in members and child not in members:
                members.add(child)
                changed = True
    return sum(rss.get(member, 0) for member in members)


def _process_tree_pids(pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                parents[int(entry.name)] = int((entry / "stat").read_text().split()[3])
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
                continue
    members = {pid}
    changed = True
    while changed:
        changed = False
        for child, parent in parents.items():
            if parent in members and child not in members:
                members.add(child)
                changed = True
    return members


def _device_used_bytes(ordinal: int, pids: set[int]) -> int:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={ordinal}",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return 0
    if result.returncode:
        return 0
    total = 0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) in pids:
            total += int(fields[1]) * 1024 * 1024
    return total


def _sample_process(
    process: subprocess.Popen[str], ordinal: int, stop: threading.Event, peaks: dict[str, int]
) -> None:
    while not stop.wait(0.05):
        pids = _process_tree_pids(process.pid)
        peaks["ram_bytes"] = max(peaks["ram_bytes"], _tree_rss(process.pid))
        peaks["vram_bytes"] = max(peaks["vram_bytes"], _device_used_bytes(ordinal, pids))


def _runner_command(
    workload: dict[str, Any], engine: str, roots: dict[str, Path], output_dir: Path
) -> list[str]:
    adapter = workload["engines"][engine]
    python = Path(adapter["python"].format(root=str(roots[engine])))
    script = Path(__file__).with_name(adapter["adapter"])
    return [
        str(python),
        str(script),
        "--repo",
        str(roots[engine]),
        "--artifact-root",
        str(roots["artifact"]),
        "--output-dir",
        str(output_dir),
        "--workload-json",
        json.dumps(workload["execution"], sort_keys=True),
    ]


def _adapter_outputs(
    reply: dict[str, Any], workload: dict[str, Any], engine: str
) -> tuple[dict[str, str], str | None]:
    if "outputs" in reply:
        output_paths = reply["outputs"]
        if not isinstance(output_paths, dict) or not all(
            isinstance(name, str) and isinstance(path, str) for name, path in output_paths.items()
        ):
            raise HarnessError(f"{engine} adapter outputs must map names to paths")
    else:
        output_path = reply.get("output_path")
        if not isinstance(output_path, str):
            raise HarnessError(f"{engine} adapter reply has no output paths")
        output_paths = {"image": output_path}
    expected_outputs = workload["execution"].get("outputs", ["image"])
    if sorted(output_paths) != sorted(expected_outputs):
        raise HarnessError(
            f"{engine} adapter outputs mismatch: expected {expected_outputs}, "
            f"got {sorted(output_paths)}"
        )
    expected_decode_mode = workload["execution"].get("decode_modes", {}).get(engine)
    decode_mode = reply.get("decode_mode")
    if expected_decode_mode is not None and decode_mode != expected_decode_mode:
        raise HarnessError(
            f"{engine} decode_mode mismatch: expected {expected_decode_mode}, got {decode_mode}"
        )
    return output_paths, decode_mode


def _adapter_runtime(
    reply: dict[str, Any], workload: dict[str, Any], engine: str
) -> dict[str, Any]:
    torch_version = reply.get("torch", "unknown")
    if not isinstance(torch_version, str):
        raise HarnessError(f"{engine} torch must be a string")
    dependencies = workload.get("engines", {}).get(engine, {}).get("dependencies", {})
    expected_torch = dependencies.get("torch")
    if expected_torch is not None and torch_version != expected_torch:
        raise HarnessError(
            f"{engine} torch mismatch: expected {expected_torch}, got {torch_version}"
        )
    runtime: dict[str, Any] = {"torch": torch_version}
    text_parameter_dtype = reply.get("text_parameter_dtype")
    requires_text_dtype = workload["id"] in (
        "W0-HARNESS-SD15-TXT2IMG",
        "W0-SD15-INPAINT",
        "W0-SDXL-INPAINT",
        "W0-SDXL-VPRED",
    )
    if requires_text_dtype and not isinstance(text_parameter_dtype, str):
        raise HarnessError(f"{engine} text_parameter_dtype must be a string")
    if isinstance(text_parameter_dtype, str):
        expected = workload["execution"]["precision"]["text"]
        if requires_text_dtype and text_parameter_dtype != expected:
            raise HarnessError(
                f"{engine} text_parameter_dtype mismatch: expected {expected}, "
                f"got {text_parameter_dtype}"
            )
        runtime["text_parameter_dtype"] = text_parameter_dtype
    if workload["id"] == "W0-SDXL-INPAINT":
        expected_torch = workload["preflight"]["interpreter"]["torch_version"]
        if torch_version != expected_torch:
            raise HarnessError(
                f"{engine} torch mismatch: expected {expected_torch}, got {torch_version}"
            )
    for key in ("attention_backend", "gguf", "memory_policy", "python"):
        value = reply.get(key)
        expected = dependencies.get(key)
        if expected is not None and value is None:
            raise HarnessError(f"{engine} {key} is required by the dependency pin")
        if value is not None:
            if not isinstance(value, str) or not value:
                raise HarnessError(f"{engine} {key} must be a nonempty string")
            runtime[key] = value
            if expected is not None and value != expected:
                raise HarnessError(f"{engine} {key} mismatch: expected {expected}, got {value}")
    actual_dtypes = reply.get("actual_dtypes")
    if actual_dtypes is not None:
        if not isinstance(actual_dtypes, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in actual_dtypes.items()
        ):
            raise HarnessError(f"{engine} actual_dtypes must map strings to strings")
        runtime["actual_dtypes"] = actual_dtypes
    memory_flags = reply.get("memory_flags")
    if memory_flags is not None:
        if not isinstance(memory_flags, dict) or not all(
            isinstance(key, str) and isinstance(value, bool) for key, value in memory_flags.items()
        ):
            raise HarnessError(f"{engine} memory_flags must map strings to booleans")
        runtime["memory_flags"] = memory_flags
    expected_runtime = workload.get("engines", {}).get(engine, {}).get("runtime", {})
    if expected_runtime:
        normalized = dict(runtime)
        backend = normalized.get("attention_backend")
        if backend in ("pytorch-sdpa", "sdpa"):
            normalized["attention_backend"] = "sdpa"
        for key, expected in expected_runtime.items():
            if normalized.get(key) != expected:
                raise HarnessError(
                    f"{engine} runtime {key} mismatch: expected {expected}, "
                    f"got {normalized.get(key)}"
                )
        runtime["attention_backend_normalized"] = normalized.get("attention_backend")
    return runtime


def _write_adapter_logs(
    output_dir: Path, engine: str, stdout: str, stderr: str
) -> tuple[Path, Path]:
    stdout_path = Path(f"{engine}.stdout.txt")
    stderr_path = Path(f"{engine}.stderr.txt")
    (output_dir / stdout_path).write_bytes(stdout.encode("utf-8"))
    (output_dir / stderr_path).write_bytes(stderr.encode("utf-8"))
    return stdout_path, stderr_path


def _kill_process_group(process: subprocess.Popen[str]) -> str | None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return None
    except AttributeError:
        if process.returncode is not None:
            return None
        try:
            process.kill()
        except OSError as exc:
            return str(exc)
    except OSError as exc:
        return str(exc)
    return None


def _read_adapter_reply(pipe: Any, reply_lines: list[str], stdout_chunks: list[str]) -> None:
    line = pipe.readline()
    reply_lines.append(line)
    stdout_chunks.append(line)


def _adapter_reply(line: str, engine: str, phase: str) -> dict[str, Any]:
    try:
        reply = json.loads(line)
    except json.JSONDecodeError as exc:
        raise HarnessError(f"{engine} adapter returned invalid JSON during {phase}: {exc}") from exc
    if not isinstance(reply, dict):
        raise HarnessError(f"{engine} adapter reply during {phase} must be an object")
    if reply.get("phase") != phase:
        raise HarnessError(f"{engine} adapter phase protocol mismatch")
    return reply


def _reply_integer(
    reply: dict[str, Any], key: str, engine: str, phase: str, *, positive: bool
) -> int:
    value = reply.get(key)
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "a positive integer" if positive else "a non-negative integer"
        raise HarnessError(f"{engine} adapter {phase}.{key} must be {qualifier}")
    return value


def run_workload(
    workload: dict[str, Any],
    engine: str,
    roots: dict[str, Path],
    acceptance_digest: str,
    output_dir: Path,
    *,
    requests: tuple[dict[str, Any], ...] | None = None,
    timing_device_uuid: str | None = None,
    timing_hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if engine not in ENGINES:
        raise HarnessError(f"unsupported engine: {engine}")
    selected_requests = (
        phase_requests(workload["execution"]["seed"]) if requests is None else requests
    )
    selected_phases = tuple(request["phase"] for request in selected_requests)
    if len(set(selected_phases)) != len(selected_phases):
        raise HarnessError("adapter request phases must be unique")
    pins = validate_pins(workload, engine, roots, acceptance_digest)
    harness_commit = (
        pins["engine_commit"]
        if roots["dinkster"] == roots[engine]
        else _git_output(roots["dinkster"], "rev-parse", "HEAD")
    )
    if _git_output(roots["dinkster"], "status", "--porcelain"):
        raise HarnessError(f"harness checkout must be clean: {roots['dinkster']}")
    inventory = hardware_inventory()
    if timing_device_uuid is None:
        hardware = validate_hardware(workload, inventory)
    else:
        if timing_hardware is None:
            raise HarnessError("timing hardware pin is required")
        hardware = validate_timing_hardware(
            workload, inventory, timing_device_uuid, timing_hardware
        )
    command = _runner_command(workload, engine, roots, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    stdin = process.stdin
    stdout_pipe = process.stdout
    stderr_pipe = process.stderr
    observations: dict[str, Any] = {}
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stderr_reader = threading.Thread(
        target=lambda: stderr_chunks.append(stderr_pipe.read()), daemon=True
    )
    stderr_reader.start()
    reply_reader: threading.Thread | None = None
    completed = False
    primary_error: BaseException | None = None
    cleanup_message: str | None = None
    try:
        for run_order, request in enumerate(selected_requests, 1):
            phase = request["phase"]
            peaks = {"ram_bytes": 0, "vram_bytes": 0}
            stop = threading.Event()
            sampler = threading.Thread(
                target=_sample_process,
                args=(process, hardware["device_ordinal"], stop, peaks),
                daemon=True,
            )
            started_perf_counter_ns = time.perf_counter_ns()
            sampler.start()
            try:
                try:
                    stdin.write(json.dumps(request) + "\n")
                    stdin.flush()
                except (BrokenPipeError, OSError) as exc:
                    raise HarnessError(
                        f"{engine} adapter input pipe failed during {phase}"
                    ) from exc
                reply_lines: list[str] = []
                reply_reader = threading.Thread(
                    target=_read_adapter_reply,
                    args=(stdout_pipe, reply_lines, stdout_chunks),
                    daemon=True,
                )
                reply_reader.start()
                reply_reader.join(timeout=ADAPTER_REPLY_TIMEOUT_SECONDS)
                if reply_reader.is_alive():
                    raise HarnessError(f"{engine} adapter timed out during {phase}")
                line = reply_lines[0]
                ended_perf_counter_ns = time.perf_counter_ns()
            finally:
                stop.set()
                sampler.join(timeout=6)
            if sampler.is_alive():
                raise HarnessError(f"{engine} process sampler did not stop during {phase}")
            if not line:
                raise HarnessError(f"{engine} adapter exited during {phase}; see retained stderr")
            reply = _adapter_reply(line, engine, phase)
            output_paths, decode_mode = _adapter_outputs(reply, workload, engine)
            generation_ns = _reply_integer(reply, "generation_ns", engine, phase, positive=True)
            pixels = (
                workload["execution"]["width"]
                * workload["execution"]["height"]
                * workload["execution"]["batch"]
            )
            recorded = request.get("record", phase != "resident-warmup")
            if not isinstance(recorded, bool):
                raise HarnessError(f"{engine} adapter request record flag must be boolean")
            metrics: dict[str, Any] = {}
            if recorded:
                metrics = {
                    "end_to_end_ns": ended_perf_counter_ns - started_perf_counter_ns,
                    "generation_ns": generation_ns,
                    "peak_ram_bytes": peaks["ram_bytes"],
                    "peak_vram_bytes": peaks["vram_bytes"],
                    "throughput_megapixels_per_second": format(
                        (pixels / 1_000_000) / (generation_ns / 1_000_000_000), ".9f"
                    ),
                }
                if phase in ("warmup", "cold"):
                    metrics["cold_load_ns"] = _reply_integer(
                        reply, "cold_load_ns", engine, phase, positive=False
                    )
                else:
                    metrics["warm_generation_ns"] = generation_ns
                if decode_mode is not None:
                    metrics["decode_mode"] = decode_mode
                for name in (
                    "decode_ns",
                    "max_memory_allocated_bytes",
                    "max_memory_reserved_bytes",
                    "sampling_ns",
                ):
                    if name in reply:
                        metrics[name] = _reply_integer(
                            reply,
                            name,
                            engine,
                            phase,
                            positive=name in ("decode_ns", "sampling_ns"),
                        )
            outputs = {
                name: {"digest": digest_file(Path(path)), "path": path}
                for name, path in sorted(output_paths.items())
            }
            observations[phase] = {
                "metrics": metrics,
                "pass": None,
                "request": request,
                "run_order": run_order,
                "runtime": _adapter_runtime(reply, workload, engine),
            }
            if requests is not None and recorded:
                observations[phase]["wall_clock"] = {
                    "clock": "perf_counter_ns",
                    "ended": ended_perf_counter_ns,
                    "started": started_perf_counter_ns,
                }
            if "outputs" in reply:
                observations[phase]["outputs"] = outputs
            else:
                observations[phase]["output"] = outputs["image"]
        completed = True
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if not stdin.closed:
            try:
                stdin.close()
            except (BrokenPipeError, OSError):
                pass
        cleanup_errors: list[str] = []
        if not completed or (reply_reader is not None and reply_reader.is_alive()):
            if error := _kill_process_group(process):
                cleanup_errors.append(f"could not terminate adapter process group: {error}")
        try:
            process.wait(timeout=ADAPTER_EXIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            if error := _kill_process_group(process):
                cleanup_errors.append(f"could not terminate adapter process group: {error}")
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cleanup_errors.append(f"{engine} adapter could not be reaped")
            except OSError as exc:
                cleanup_errors.append(f"{engine} adapter wait failed: {exc}")
        except OSError as exc:
            cleanup_errors.append(f"{engine} adapter wait failed: {exc}")
        if error := _kill_process_group(process):
            cleanup_errors.append(f"could not terminate adapter descendants: {error}")
        if reply_reader is not None and reply_reader.is_alive():
            reply_reader.join(timeout=5)
            if reply_reader.is_alive():
                cleanup_errors.append(f"{engine} adapter reply pipe did not close")
        remainder: list[str] = []
        stdout_reader = None
        if reply_reader is None or not reply_reader.is_alive():
            stdout_reader = threading.Thread(
                target=lambda: remainder.append(stdout_pipe.read()), daemon=True
            )
            stdout_reader.start()
            stdout_reader.join(timeout=5)
        stderr_reader.join(timeout=5)
        if (stdout_reader is not None and stdout_reader.is_alive()) or stderr_reader.is_alive():
            if error := _kill_process_group(process):
                cleanup_errors.append(f"could not close adapter output pipes: {error}")
            if stdout_reader is not None:
                stdout_reader.join(timeout=5)
            stderr_reader.join(timeout=5)
        stdout_chunks.extend(remainder)
        try:
            _write_adapter_logs(output_dir, engine, "".join(stdout_chunks), "".join(stderr_chunks))
        except OSError as exc:
            cleanup_errors.append(f"could not retain {engine} adapter logs: {exc}")
        if stdout_reader is not None and stdout_reader.is_alive():
            cleanup_errors.append(f"{engine} adapter stdout pipe did not close")
        if stderr_reader.is_alive():
            cleanup_errors.append(f"{engine} adapter stderr pipe did not close")
        if cleanup_errors:
            cleanup_message = "; ".join(cleanup_errors)
            if primary_error is not None:
                primary_error.add_note(f"Adapter cleanup errors: {cleanup_message}")
    stderr = "".join(stderr_chunks)
    stdout = "".join(stdout_chunks)
    stdout_path = Path(f"{engine}.stdout.txt")
    stderr_path = Path(f"{engine}.stderr.txt")
    late_error = None
    if process.returncode:
        late_error = HarnessError(f"{engine} adapter failed ({process.returncode}): {stderr}")
    trailing_stdout = "".join(stdout_chunks[len(selected_requests) :])
    if late_error is None and trailing_stdout:
        late_error = HarnessError(f"{engine} adapter emitted trailing stdout after real phase")
    if late_error is not None:
        if cleanup_message is not None:
            late_error.add_note(f"Adapter cleanup errors: {cleanup_message}")
        raise late_error
    if cleanup_message is not None:
        raise HarnessError(cleanup_message)
    if _git_output(roots[engine], "rev-parse", "HEAD") != pins["engine_commit"]:
        raise HarnessError(f"{engine} checkout changed during adapter execution")
    if _git_output(roots[engine], "status", "--porcelain"):
        raise HarnessError(f"{engine} checkout became dirty during adapter execution")
    validate_extensions_unchanged(pins, engine, activity="adapter execution")
    if _git_output(roots["dinkster"], "rev-parse", "HEAD") != harness_commit:
        raise HarnessError("harness checkout changed during adapter execution")
    if _git_output(roots["dinkster"], "status", "--porcelain"):
        raise HarnessError("harness checkout became dirty during adapter execution")
    record = {
        "acceptance_manifest_digest": acceptance_digest,
        "engine": {"commit": pins["engine_commit"], "id": engine},
        "execution": workload["execution"],
        "hardware": hardware,
        "harness": {"commit": harness_commit, "schema": SCHEMA},
        "pins": pins,
        "stderr_digest": digest_bytes(stderr.encode()),
        "stderr_path": str(stderr_path),
        "stdout_digest": digest_bytes(stdout.encode()),
        "stdout_path": str(stdout_path),
        "stdout_protocol": "json-lines/1",
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "workload_id": workload["id"],
    }
    if requests is None:
        record.update(
            {
                "overall_pass": None,
                "real": observations["real"],
                "warmup": observations["warmup"],
            }
        )
    else:
        record["command"] = command
        record["observations"] = observations
        record["pid"] = process.pid
    return record


def _load_output(path: str) -> Any:
    import numpy as np

    return np.load(path, allow_pickle=False)


def _gaussian_filter(image: Any) -> Any:
    import numpy as np

    offsets = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2 * 1.5**2))
    kernel /= kernel.sum()
    horizontal = np.zeros_like(image, dtype=np.float64)
    padded = np.pad(image, ((0, 0), (0, 0), (5, 5), (0, 0)), mode="reflect")
    for index, weight in enumerate(kernel):
        horizontal += weight * padded[:, :, index : index + image.shape[2], :]
    result = np.zeros_like(horizontal)
    padded = np.pad(horizontal, ((0, 0), (5, 5), (0, 0), (0, 0)), mode="reflect")
    for index, weight in enumerate(kernel):
        result += weight * padded[:, index : index + image.shape[1], :, :]
    return result


def image_ssim(left: Any, right: Any) -> float:
    """Channel-aware SSIM with the standard 11x11 Gaussian window."""
    mean_left = _gaussian_filter(left)
    mean_right = _gaussian_filter(right)
    variance_left = _gaussian_filter(left * left) - mean_left * mean_left
    variance_right = _gaussian_filter(right * right) - mean_right * mean_right
    covariance = _gaussian_filter(left * right) - mean_left * mean_right
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) / (
        (mean_left**2 + mean_right**2 + c1) * (variance_left + variance_right + c2)
    )
    return float(score.mean())


def compare_output(
    baseline: dict[str, Any], candidate: dict[str, Any], rule: dict[str, Any], name: str = "image"
) -> dict[str, Any]:
    import numpy as np

    left = _load_output(baseline["path"]).astype(np.float64)
    right = _load_output(candidate["path"]).astype(np.float64)
    if left.shape != right.shape:
        return {
            "comparator": rule.get("comparator", name),
            "pass": False,
            "reason": "shape mismatch",
        }
    error = np.abs(left - right)
    observed = {
        "max_abs": format(float(error.max(initial=0)), ".12g"),
        "mean_abs": format(float(error.mean()), ".12g"),
    }
    if rule.get("comparator") == "exact-array/1":
        passed = float(observed["max_abs"]) <= float(rule["max_abs"])
        comparator = "exact-array/1"
        threshold = {"max_abs": rule["max_abs"]}
    else:
        observed["ssim"] = format(image_ssim(left, right), ".12g")
        passed = (
            float(observed["max_abs"]) <= float(rule["max_abs"])
            and float(observed["mean_abs"]) <= float(rule["mean_abs"])
            and float(observed["ssim"]) >= float(rule["ssim_minimum"])
        )
        comparator = "image-max-mean-ssim/1"
        threshold = {
            "max_abs": rule["max_abs"],
            "mean_abs": rule["mean_abs"],
            "ssim_minimum": rule["ssim_minimum"],
        }
    return {
        "baseline_digest": baseline["digest"],
        "candidate_digest": candidate["digest"],
        "comparator": comparator,
        "observed": observed,
        "pass": passed,
        "threshold": threshold,
    }


def _phase_outputs(phase: dict[str, Any]) -> dict[str, dict[str, str]]:
    outputs = phase.get("outputs")
    if outputs is None and isinstance(phase.get("output"), dict):
        outputs = {"image": phase["output"]}
    if not isinstance(outputs, dict):
        raise HarnessError("record phase outputs must be an object")
    for name, output in outputs.items():
        if not isinstance(name, str) or not isinstance(output, dict):
            raise HarnessError("record phase outputs must map names to objects")
        path = output.get("path")
        digest = output.get("digest")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise HarnessError(f"record output {name} must contain path and digest")
        if digest_file(Path(path)) != digest:
            raise HarnessError(f"persisted output digest mismatch: {path}")
    return outputs


def compare_records(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    acceptance: dict[str, Any],
    workload: dict[str, Any],
) -> dict[str, Any]:
    if baseline["workload_id"] != candidate["workload_id"]:
        raise HarnessError("record workload ids differ")
    if baseline["workload_id"] != workload.get("id"):
        raise HarnessError("record workload id does not match selected workload")
    if baseline["engine"]["id"] != "comfyui" or candidate["engine"]["id"] != "dinkster":
        raise HarnessError("compare requires comfyui baseline and dinkster candidate")
    current_digest = digest_bytes(canonical_bytes(acceptance))
    if workload["acceptance_manifest"]["digest"] != current_digest:
        raise HarnessError("workload acceptance-manifest digest mismatch")
    candidate_digest = candidate.get("acceptance_manifest_digest")
    if not isinstance(candidate_digest, str) or candidate_digest != current_digest:
        raise HarnessError("candidate record acceptance digest is stale")
    rules = acceptance["workloads"][baseline["workload_id"]]
    if rules.get("status", "active") == "calibration_pending":
        raise HarnessError("acceptance calibration is pending")
    predecessor = rules.get("calibration", {}).get("pre_calibration_acceptance_digest")
    baseline_digest = baseline.get("acceptance_manifest_digest")
    predecessor_matches = isinstance(predecessor, str) and baseline_digest == predecessor
    if not isinstance(baseline_digest, str) or (
        baseline_digest != current_digest and not predecessor_matches
    ):
        raise HarnessError(
            "baseline record acceptance digest is not current or declared predecessor"
        )
    baseline_torch = {baseline[phase]["runtime"]["torch"] for phase in PHASES}
    candidate_torch = {candidate[phase]["runtime"]["torch"] for phase in PHASES}
    if len(baseline_torch) != 1 or len(candidate_torch) != 1 or baseline_torch != candidate_torch:
        raise HarnessError("record torch versions differ")
    record_outputs: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    for role, engine, record in (
        ("baseline", "comfyui", baseline),
        ("candidate", "dinkster", candidate),
    ):
        expected_decode_mode = workload.get("execution", {}).get("decode_modes", {}).get(engine)
        if expected_decode_mode is not None:
            for phase in PHASES:
                if record[phase]["metrics"].get("decode_mode") != expected_decode_mode:
                    raise HarnessError(
                        f"{role} {phase} decode_mode does not match workload expectation"
                    )
        record_outputs[role] = {phase: _phase_outputs(record[phase]) for phase in PHASES}
        if rules.get("within_engine_repeat") == "byte-identical":
            warmup_outputs = record_outputs[role]["warmup"]
            real_outputs = record_outputs[role]["real"]
            if sorted(warmup_outputs) != sorted(real_outputs) or any(
                warmup_outputs[name]["digest"] != real_outputs[name]["digest"]
                for name in warmup_outputs
            ):
                raise HarnessError(f"{role} warmup and real outputs are not byte-identical")
    verdict: dict[str, Any] = {
        "acceptance_manifest_digest": current_digest,
        "baseline": {"commit": baseline["engine"]["commit"]},
        "candidate": {"commit": candidate["engine"]["commit"]},
        "schema": SCHEMA,
        "workload_id": baseline["workload_id"],
    }
    for phase in PHASES:
        metric_verdicts = {}
        phase_rules = rules["phases"][phase]
        for name, budget in phase_rules["metrics"].items():
            baseline_value = baseline[phase]["metrics"][name]
            candidate_value = candidate[phase]["metrics"][name]
            if name == "throughput_megapixels_per_second":
                limit = float(baseline_value) * (1 - float(budget))
                passed = float(candidate_value) >= limit
                relation = "minimum"
            else:
                limit = float(baseline_value) * (1 + float(budget))
                passed = float(candidate_value) <= limit
                relation = "maximum"
            metric_verdicts[name] = {
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
                "limit": format(limit, ".12g"),
                "pass": passed,
                "relation": relation,
                "unit": "bytes"
                if name.endswith("bytes")
                else ("ns" if name.endswith("ns") else "MP/s"),
                "value": candidate_value,
            }
        output_rules = phase_rules.get("outputs")
        if output_rules is None:
            output_rules = {"image": phase_rules["output"]}
        baseline_outputs = record_outputs["baseline"][phase]
        candidate_outputs = record_outputs["candidate"][phase]
        if sorted(baseline_outputs) != sorted(output_rules) or sorted(candidate_outputs) != sorted(
            output_rules
        ):
            raise HarnessError(f"{phase} record outputs do not match acceptance outputs")
        output_verdicts = {
            name: compare_output(
                baseline_outputs[name], candidate_outputs[name], output_rules[name], name
            )
            for name in sorted(output_rules)
        }
        phase_pass = all(item["pass"] for item in output_verdicts.values()) and all(
            item["pass"] for item in metric_verdicts.values()
        )
        verdict[phase] = {
            "metrics": metric_verdicts,
            "pass": phase_pass,
        }
        if "outputs" in phase_rules:
            verdict[phase]["output_comparisons"] = output_verdicts
        else:
            verdict[phase]["output_comparison"] = output_verdicts["image"]
    verdict["overall_pass"] = verdict["warmup"]["pass"] and verdict["real"]["pass"]
    return verdict


def validate_timing_history(history: dict[str, Any]) -> None:
    if history.get("schema") != SCHEMA:
        raise HarnessError(f"timing history schema must be {SCHEMA}")
    records = history.get("records")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise HarnessError("timing history records must be a list of objects")
    policy = history.get("policy")
    if not isinstance(policy, dict):
        raise HarnessError("timing history policy must be an object")
    if policy.get("prior_dinkster_warm_warn_ratio") != format(TIMING_REGRESSION_WARN_RATIO, ".2f"):
        raise HarnessError("timing history prior-Dinkster warning ratio is not the fixed policy")
    if policy.get("regression_status") != "WARN":
        raise HarnessError("timing history regression status is not the fixed policy")
    if policy.get("threshold_relation") != "strictly-greater-than":
        raise HarnessError("timing history threshold relation is not the fixed policy")
    hardware = history.get("hardware")
    if not isinstance(hardware, dict) or set(hardware) != {
        "device_ordinal",
        "device_uuid",
        "gpus",
        "hostname",
    }:
        raise HarnessError("timing history hardware pin is incomplete")
    if hardware["device_ordinal"] != 0 or not isinstance(hardware["device_uuid"], str):
        raise HarnessError("timing history designated device pin is invalid")
    if not isinstance(hardware["hostname"], str) or not hardware["hostname"]:
        raise HarnessError("timing history hostname pin is invalid")
    gpus = hardware["gpus"]
    if not isinstance(gpus, list) or not gpus:
        raise HarnessError("timing history must pin a nonempty exact GPU inventory")
    for index, gpu in enumerate(gpus):
        if not isinstance(gpu, dict) or set(gpu) != {
            "driver",
            "index",
            "memory_total_mib",
            "name",
            "uuid",
        }:
            raise HarnessError(f"timing history GPU {index} pin is incomplete")
        if (
            gpu["index"] != index
            or not isinstance(gpu["driver"], str)
            or not isinstance(gpu["memory_total_mib"], int)
            or gpu["memory_total_mib"] <= 0
            or not isinstance(gpu["name"], str)
            or not isinstance(gpu["uuid"], str)
        ):
            raise HarnessError(f"timing history GPU {index} pin is invalid")
    if hardware["device_ordinal"] >= len(gpus):
        raise HarnessError("timing history designated device is absent from the inventory")
    if gpus[hardware["device_ordinal"]]["uuid"] != hardware["device_uuid"]:
        raise HarnessError("timing history designated UUID is absent from the inventory pin")
    for index, record in enumerate(records):
        where = f"timing history record {index}"
        workload = record.get("workload")
        hardware = record.get("hardware")
        engines = record.get("engines")
        if record.get("schema") != SCHEMA or record.get("status") not in (
            "BASELINE",
            "PASS",
            "WARN",
        ):
            raise HarnessError(f"{where} is not an accepted schema-{SCHEMA} result")
        if not isinstance(workload, dict) or not isinstance(workload.get("contract_digest"), str):
            raise HarnessError(f"{where} has no workload contract digest")
        if (
            not isinstance(hardware, dict)
            or not isinstance(hardware.get("device_uuid"), str)
            or not isinstance(hardware.get("inventory"), dict)
        ):
            raise HarnessError(f"{where} has no device UUID")
        if not isinstance(engines, dict):
            raise HarnessError(f"{where} has no engine results")
        for engine in ENGINES:
            result = engines.get(engine)
            if not isinstance(result, dict):
                raise HarnessError(f"{where} has no {engine} result")
            if not isinstance(result.get("commit"), str) or not isinstance(
                result.get("torch"), str
            ):
                raise HarnessError(f"{where} has incomplete {engine} source/runtime identity")
            median = result.get("warm_median_ns")
            if (
                isinstance(median, bool)
                or not isinstance(median, (int, float))
                or not math.isfinite(median)
                or median <= 0
            ):
                raise HarnessError(f"{where} has invalid {engine} warm median")


def _timing_correctness(
    processes: list[dict[str, Any]], acceptance: dict[str, Any], workload: dict[str, Any]
) -> dict[str, Any]:
    rules = acceptance["workloads"].get(workload["id"])
    if not isinstance(rules, dict) or rules.get("status", "active") != "active":
        raise HarnessError("timing requires an active correctness policy")
    expected_requests = workload_timing_requests(workload)
    expected_phases = tuple(request["phase"] for request in expected_requests)
    first_observations = processes[0].get("observations")
    if not isinstance(first_observations, dict):
        raise HarnessError("timing process has no observations")
    semantic_keys = []
    reference_outputs: dict[bytes, dict[str, dict[str, Any]]] = {}
    for phase, request in zip(expected_phases, expected_requests, strict=True):
        settings = {
            key: value
            for key, value in request.items()
            if key not in ("phase", "record", "scenario")
        }
        settings.setdefault("cfg", workload.get("execution", {}).get("cfg"))
        semantic_key = canonical_bytes(settings)
        semantic_keys.append(semantic_key)
        reference_outputs.setdefault(semantic_key, _phase_outputs(first_observations[phase]))
    checks: list[dict[str, Any]] = []
    runtime_versions: set[str] = set()
    expected_text_dtype = (
        workload["execution"]["precision"]["text"]
        if workload["id"] == "W0-HARNESS-SD15-TXT2IMG"
        else None
    )
    for process_index, process in enumerate(processes, 1):
        observations = process.get("observations")
        if not isinstance(observations, dict) or tuple(observations) != expected_phases:
            raise HarnessError("timing process phases do not match the fixed request sequence")
        for phase, request, semantic_key in zip(
            expected_phases, expected_requests, semantic_keys, strict=True
        ):
            observation = observations[phase]
            runtime = observation.get("runtime")
            if not isinstance(runtime, dict) or not isinstance(runtime.get("torch"), str):
                raise HarnessError("timing observation has no torch runtime identity")
            runtime_versions.add(runtime["torch"])
            if (
                expected_text_dtype is not None
                and runtime.get("text_parameter_dtype") != expected_text_dtype
            ):
                raise HarnessError(
                    f"timing {process['engine']['id']} text parameter dtype does not match "
                    "the workload"
                )
            outputs = _phase_outputs(observation)
            phase_rules = rules["phases"]["warmup" if phase == "cold" else "real"]
            output_rules = phase_rules.get("outputs")
            if output_rules is None:
                output_rules = {"image": phase_rules["output"]}
            if sorted(outputs) != sorted(output_rules) or sorted(
                reference_outputs[semantic_key]
            ) != sorted(output_rules):
                raise HarnessError("timing outputs do not match the correctness policy")
            comparisons = {
                name: compare_output(
                    reference_outputs[semantic_key][name], outputs[name], output_rules[name], name
                )
                for name in sorted(output_rules)
            }
            if not all(comparison["pass"] for comparison in comparisons.values()):
                raise HarnessError(
                    f"timing output correctness failed for process {process_index} phase {phase}"
                )
            checks.append(
                {
                    "engine": process["engine"]["id"],
                    "outputs": comparisons,
                    "phase": phase,
                    "process_index": process_index,
                    "recorded": request.get("record", True),
                }
            )
    if len(runtime_versions) != 1:
        raise HarnessError("timing processes used different torch versions")
    return {
        "acceptance_manifest_digest": digest_bytes(canonical_bytes(acceptance)),
        "checks": checks,
        "pass": True,
        "text_parameter_dtype": expected_text_dtype,
        "torch": runtime_versions.pop(),
    }


def _timing_median(values: list[int]) -> float:
    if not values or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values
    ):
        raise HarnessError("timing samples must be positive integer nanoseconds")
    return float(statistics.median(values))


def _timing_ratio(numerator: float, denominator: float) -> str:
    if numerator <= 0 or denominator <= 0:
        raise HarnessError("timing ratio operands must be positive")
    return format(numerator / denominator, ".9f")


def _timing_distribution(values: list[int]) -> dict[str, Any]:
    median = _timing_median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "count": len(values),
        "mad_ns": float(statistics.median(deviations)),
        "max_ns": max(values),
        "median_ns": median,
        "min_ns": min(values),
        "samples_ns": values,
    }


def _summarize_repeated_timing(
    processes: list[dict[str, Any]],
    correctness: dict[str, Any],
    workload: dict[str, Any],
    roots: dict[str, Path],
    output_dir: Path,
    timestamp_utc: str,
) -> dict[str, Any]:
    if len(processes) != len(TIMING_ENGINE_ORDER):
        raise HarnessError("repeated timing requires exactly four adapter processes")
    if tuple(process["engine"]["id"] for process in processes) != TIMING_ENGINE_ORDER:
        raise HarnessError("timing engine order is not comfyui,dinkster,dinkster,comfyui")
    requests = workload_timing_requests(workload)
    phases = tuple(request["phase"] for request in requests)
    plain_phases = tuple(
        request["phase"] for request in requests if request.get("scenario") == "plain"
    )
    change_phases = tuple(
        request["phase"] for request in requests if request.get("scenario") == "settings-change"
    )
    if len(change_phases) < 2:
        raise HarnessError("repeated timing requires two settings-change requests")
    hardware_records = [process["hardware"] for process in processes]
    if any(hardware != hardware_records[0] for hardware in hardware_records[1:]):
        raise HarnessError("timing hardware changed between processes")
    harness_commits = {process["harness"]["commit"] for process in processes}
    if len(harness_commits) != 1:
        raise HarnessError("timing harness commit changed during the run")
    final_harness_commit = harness_commits.pop()
    for engine in ENGINES:
        commits = {
            process["engine"]["commit"]
            for process in processes
            if process["engine"]["id"] == engine
        }
        if len(commits) != 1:
            raise HarnessError(f"{engine} commit changed between timing processes")
    samples: dict[str, dict[str, list[int]]] = {
        engine: {
            "cold": [],
            "cold_decode": [],
            "cold_load": [],
            "cold_sampling": [],
            "plain": [],
            "plain_sampling": [],
            "settings_first": [],
            "settings_second": [],
        }
        for engine in ENGINES
    }
    memory: dict[str, dict[str, list[int]]] = {
        engine: {"allocated": [], "reserved": [], "rss": [], "vram": []} for engine in ENGINES
    }
    runtime_receipts: dict[str, dict[str, Any]] = {}
    monotonic: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    for process_index, (engine, process) in enumerate(
        zip(TIMING_ENGINE_ORDER, processes, strict=True), 1
    ):
        observations = process.get("observations")
        if not isinstance(observations, dict) or tuple(observations) != phases:
            raise HarnessError("timing process phases do not match the repeated request sequence")
        runtimes = [observations[phase]["runtime"] for phase in phases]
        if any(runtime != runtimes[0] for runtime in runtimes[1:]):
            raise HarnessError(f"{engine} runtime receipt changed between requests")
        existing = runtime_receipts.get(engine)
        if existing is not None and existing != runtimes[0]:
            raise HarnessError(f"{engine} runtime receipt changed between processes")
        runtime_receipts[engine] = runtimes[0]
        cold = observations["cold"]["metrics"]
        samples[engine]["cold"].append(cold["end_to_end_ns"])
        samples[engine]["cold_load"].append(cold["cold_load_ns"])
        samples[engine]["cold_sampling"].append(cold["sampling_ns"])
        samples[engine]["cold_decode"].append(cold["decode_ns"])
        process_plain = [observations[phase]["metrics"]["end_to_end_ns"] for phase in plain_phases]
        samples[engine]["plain"].extend(process_plain)
        samples[engine]["plain_sampling"].extend(
            observations[phase]["metrics"]["sampling_ns"] for phase in plain_phases
        )
        samples[engine]["settings_first"].append(
            observations[change_phases[0]]["metrics"]["end_to_end_ns"]
        )
        samples[engine]["settings_second"].append(
            observations[change_phases[1]]["metrics"]["end_to_end_ns"]
        )
        monotonic_descending = all(
            right < left for left, right in zip(process_plain, process_plain[1:], strict=False)
        )
        monotonic.append(
            {
                "engine": engine,
                "monotonic_descending": monotonic_descending,
                "process_index": process_index,
                "samples_ns": process_plain,
            }
        )
        for observation in observations.values():
            metrics = observation["metrics"]
            if not metrics:
                continue
            memory[engine]["rss"].append(metrics["peak_ram_bytes"])
            memory[engine]["vram"].append(metrics["peak_vram_bytes"])
            memory[engine]["allocated"].append(metrics["max_memory_allocated_bytes"])
            memory[engine]["reserved"].append(metrics["max_memory_reserved_bytes"])
        directory = f"{process_index:02d}-{engine}"
        logs.append(
            {
                "engine": engine,
                "process_index": process_index,
                "stderr_digest": process["stderr_digest"],
                "stderr_path": str(Path(directory) / process["stderr_path"]),
                "stdout_digest": process["stdout_digest"],
                "stdout_path": str(Path(directory) / process["stdout_path"]),
            }
        )
    if any(item["monotonic_descending"] for item in monotonic):
        raise HarnessError("retained warm samples still have a monotonic descending trend")
    distributions = {
        engine: {name: _timing_distribution(values) for name, values in groups.items()}
        for engine, groups in samples.items()
    }
    warm_medians = {engine: distributions[engine]["plain"]["median_ns"] for engine in ENGINES}
    process_plain_medians = [item["samples_ns"] for item in monotonic]
    escalation = _timing_escalation(
        {engine: samples[engine]["cold"] for engine in ENGINES},
        {engine: samples[engine]["plain"] for engine in ENGINES},
        [_timing_median(values) for values in process_plain_medians],
    )
    noise_fraction = max(
        distributions[engine]["plain"]["mad_ns"] / warm_medians[engine] for engine in ENGINES
    )
    warm_ratio = warm_medians["dinkster"] / warm_medians["comfyui"]
    parity_pass = warm_ratio <= 1.0 + noise_fraction
    steps = workload["execution"]["steps"]
    engine_records = {}
    for engine in ENGINES:
        sampling_median = distributions[engine]["plain_sampling"]["median_ns"]
        engine_records[engine] = {
            "commit": next(
                process["engine"]["commit"]
                for process in processes
                if process["engine"]["id"] == engine
            ),
            "memory": {
                "peak_allocated_bytes": max(memory[engine]["allocated"]),
                "peak_process_rss_bytes": max(memory[engine]["rss"]),
                "peak_process_vram_bytes": max(memory[engine]["vram"]),
                "peak_reserved_bytes": max(memory[engine]["reserved"]),
            },
            "runtime": runtime_receipts[engine],
            "sampling": {
                **distributions[engine]["plain_sampling"],
                "median_iterations_per_second": format(
                    steps / (sampling_median / 1_000_000_000), ".9f"
                ),
                "median_step_ms": format(sampling_median / steps / 1_000_000, ".9f"),
            },
            "settings_change_first": distributions[engine]["settings_first"],
            "settings_change_second": distributions[engine]["settings_second"],
            "warm_end_to_end": distributions[engine]["plain"],
            "warm_median_ns": warm_medians[engine],
            "cold": {
                "decode": distributions[engine]["cold_decode"],
                "load_to_model_ready": distributions[engine]["cold_load"],
                "request_to_reply": distributions[engine]["cold"],
                "sampling": distributions[engine]["cold_sampling"],
            },
        }
    inventory = hardware_records[0]["inventory"]
    ordinal = hardware_records[0]["device_ordinal"]
    result = {
        "acceptance": {
            "noise_fraction": format(noise_fraction, ".9f"),
            "pass": parity_pass,
            "rule": "Dinkster warm median <= ComfyUI warm median plus pooled relative MAD",
        },
        "correctness": {
            "acceptance_manifest_digest": correctness["acceptance_manifest_digest"],
            "check_count": len(correctness["checks"]),
            "pass": True,
        },
        "engines": engine_records,
        "escalation": escalation,
        "hardware": {
            "device": inventory["gpus"][ordinal],
            "hostname": inventory["hostname"],
            "platform": inventory["platform"],
            "ram_bytes": inventory["ram_bytes"],
        },
        "harness": {
            "commit": final_harness_commit,
            "external_clock": "perf_counter_ns request-to-reply wall time",
            "nvml_poll_interval_seconds": "0.05",
            "process_order": list(TIMING_ENGINE_ORDER),
            "requests_per_process": list(phases),
            "schema": SCHEMA,
        },
        "logs": logs,
        "ratios": {
            "dinkster_over_comfyui_warm_end_to_end": format(warm_ratio, ".9f"),
            "dinkster_over_comfyui_warm_sampling": _timing_ratio(
                distributions["dinkster"]["plain_sampling"]["median_ns"],
                distributions["comfyui"]["plain_sampling"]["median_ns"],
            ),
        },
        "retained_sample_trend": monotonic,
        "schema": SCHEMA,
        "status": "PASS" if parity_pass else "WARN",
        "timestamp_utc": timestamp_utc,
        "workload": {
            "artifacts": [
                {key: artifact[key] for key in ("bytes", "digest", "path", "role")}
                for artifact in workload["artifacts"]
            ],
            "contract_digest": digest_bytes(canonical_bytes(workload)),
            "id": workload["id"],
            "settings": workload["execution"],
        },
    }
    for log in logs:
        for kind in ("stderr", "stdout"):
            path = output_dir / log[f"{kind}_path"]
            if not path.is_file() or digest_file(path) != log[f"{kind}_digest"]:
                raise HarnessError(f"timing log is missing or changed: {log[f'{kind}_path']}")
    return result


def _compatible_prior_timing(
    history: dict[str, Any],
    contract_digest: str,
    inventory: dict[str, Any],
    torch_version: str,
) -> dict[str, Any] | None:
    for record in reversed(history["records"]):
        workload = record.get("workload")
        hardware = record.get("hardware")
        engines = record.get("engines")
        if (
            isinstance(workload, dict)
            and workload.get("contract_digest") == contract_digest
            and isinstance(hardware, dict)
            and hardware.get("inventory") == inventory
            and isinstance(engines, dict)
            and isinstance(engines.get("dinkster"), dict)
            and engines["dinkster"].get("torch") == torch_version
        ):
            return record
    return None


def _timing_escalation(
    cold: dict[str, list[int]],
    warm: dict[str, list[int]],
    process_warm_medians: list[float],
) -> dict[str, Any]:
    reasons: list[str] = []
    for engine in ENGINES:
        if max(warm[engine]) / min(warm[engine]) > 1.10:
            reasons.append(f"{engine} warm max/min exceeds 1.10")
        if max(cold[engine]) / min(cold[engine]) > 1.15:
            reasons.append(f"{engine} cold max/min exceeds 1.15")
    for engine, indexes in (("comfyui", (0, 3)), ("dinkster", (1, 2))):
        first, second = (process_warm_medians[index] for index in indexes)
        if max(first, second) / min(first, second) > 1.05:
            reasons.append(f"{engine} process-position warm medians differ by more than 5 percent")
    first_direction = process_warm_medians[1] / process_warm_medians[0]
    second_direction = process_warm_medians[2] / process_warm_medians[3]
    if (first_direction - 1.0) * (second_direction - 1.0) < 0:
        reasons.append("the two engine-order directions disagree on which engine is faster")
    return {"reasons": reasons, "triggered": bool(reasons)}


def summarize_timing(
    processes: list[dict[str, Any]],
    correctness: dict[str, Any],
    workload: dict[str, Any],
    history: dict[str, Any],
    roots: dict[str, Path],
    output_dir: Path,
    timestamp_utc: str,
) -> dict[str, Any]:
    if workload["execution"].get("timing") is not None:
        return _summarize_repeated_timing(
            processes, correctness, workload, roots, output_dir, timestamp_utc
        )
    if len(processes) != len(TIMING_ENGINE_ORDER):
        raise HarnessError("timing requires exactly four adapter processes")
    if tuple(process["engine"]["id"] for process in processes) != TIMING_ENGINE_ORDER:
        raise HarnessError("timing engine order is not comfyui,dinkster,dinkster,comfyui")
    hardware_records = [process["hardware"] for process in processes]
    if any(hardware != hardware_records[0] for hardware in hardware_records[1:]):
        raise HarnessError("timing hardware changed between processes")
    harness_commits: list[str] = []
    for process in processes:
        harness_record = process.get("harness")
        if not isinstance(harness_record, dict):
            raise HarnessError("timing process harness commit is missing")
        harness_commit = harness_record.get("commit")
        if not isinstance(harness_commit, str) or not harness_commit:
            raise HarnessError("timing process harness commit is missing")
        harness_commits.append(harness_commit)
    if len(set(harness_commits)) != 1:
        raise HarnessError("harness commit changed between timing processes")
    final_harness_commit = _git_output(roots["dinkster"], "rev-parse", "HEAD")
    if harness_commits[0] != final_harness_commit:
        raise HarnessError("timing process harness commit does not match final Dinkster HEAD")
    for engine in ENGINES:
        commits = {
            process["engine"]["commit"]
            for process in processes
            if process["engine"]["id"] == engine
        }
        if len(commits) != 1:
            raise HarnessError(f"{engine} commit changed between timing processes")
    cold: dict[str, list[int]] = {engine: [] for engine in ENGINES}
    warm: dict[str, list[int]] = {engine: [] for engine in ENGINES}
    process_warm_medians: list[float] = []
    logs: list[dict[str, Any]] = []
    for process_index, (engine, process) in enumerate(
        zip(TIMING_ENGINE_ORDER, processes, strict=True), 1
    ):
        observations = process.get("observations")
        if not isinstance(observations, dict) or tuple(observations) != TIMING_PHASES:
            raise HarnessError("timing process phases do not match the fixed request sequence")
        cold[engine].append(observations["cold"]["metrics"]["end_to_end_ns"])
        process_warm = [
            observations[phase]["metrics"]["end_to_end_ns"] for phase in TIMING_WARM_PHASES
        ]
        warm[engine].extend(process_warm)
        process_warm_medians.append(_timing_median(process_warm))
        directory = f"{process_index:02d}-{engine}"
        logs.append(
            {
                "engine": engine,
                "process_index": process_index,
                "stderr_digest": process["stderr_digest"],
                "stderr_path": str(Path(directory) / process["stderr_path"]),
                "stdout_digest": process["stdout_digest"],
                "stdout_path": str(Path(directory) / process["stdout_path"]),
            }
        )
    if any(len(cold[engine]) != 2 or len(warm[engine]) != 4 for engine in ENGINES):
        raise HarnessError(
            "timing did not retain exactly two cold and four warm samples per engine"
        )
    cold_medians = {engine: _timing_median(values) for engine, values in cold.items()}
    warm_medians = {engine: _timing_median(values) for engine, values in warm.items()}
    contract_digest = digest_bytes(canonical_bytes(workload))
    hardware = hardware_records[0]
    torch_version = correctness["torch"]
    prior = _compatible_prior_timing(history, contract_digest, hardware["inventory"], torch_version)
    prior_median = None
    prior_ratio = None
    regression_status = "BASELINE"
    prior_selection = "no-history" if not history["records"] else "no-compatible-record"
    if prior is not None:
        prior_selection = "compatible-record"
        prior_median = prior["engines"]["dinkster"]["warm_median_ns"]
        prior_ratio = warm_medians["dinkster"] / prior_median
        regression_status = (
            "WARN"
            if Fraction(warm_medians["dinkster"])
            > Fraction(prior_median) * Fraction(str(TIMING_REGRESSION_WARN_RATIO))
            else "PASS"
        )
    escalation = _timing_escalation(cold, warm, process_warm_medians)
    inventory = hardware["inventory"]
    ordinal = hardware["device_ordinal"]
    designated_gpu = inventory["gpus"][ordinal]
    settings = {
        key: value
        for key, value in workload["execution"].items()
        if key not in ("device_ordinal", "hardware")
    }
    engine_records: dict[str, dict[str, Any]] = {}
    for engine in ENGINES:
        first = next(process for process in processes if process["engine"]["id"] == engine)
        python = workload["engines"][engine]["python"].format(root=str(roots[engine]))
        engine_records[engine] = {
            "cold_median_ns": cold_medians[engine],
            "cold_ns": cold[engine],
            "commit": first["engine"]["commit"],
            "python": python,
            "torch": torch_version,
            "warm_median_ns": warm_medians[engine],
            "warm_ns": warm[engine],
        }
    result = {
        "correctness": {
            "acceptance_manifest_digest": correctness["acceptance_manifest_digest"],
            "check_count": len(correctness["checks"]),
            "pass": True,
            "text_parameter_dtype": correctness["text_parameter_dtype"],
        },
        "disclosures": [
            "Native inference-runtime timing through the existing adapters; not HTTP, queue, "
            "server, or product latency.",
            "The ComfyUI adapter pins normal-scheduler sigma calculation to CPU for the "
            "existing exact-work mapping.",
            "Cold means fresh process and empty application/model cache after digest "
            "validation warmed the OS page cache.",
        ],
        "engines": engine_records,
        "escalation": escalation,
        "hardware": {
            "cpu": inventory["cpu"],
            "device_index": designated_gpu["index"],
            "device_name": designated_gpu["name"],
            "device_uuid": designated_gpu["uuid"],
            "driver": designated_gpu["driver"],
            "inventory": inventory,
            "platform": inventory["platform"],
            "ram_bytes": inventory["ram_bytes"],
        },
        "harness": {
            "commit": final_harness_commit,
            "external_clock": "perf_counter_ns request-to-reply wall time",
            "job_count": 16,
            "process_count": 4,
            "process_order": list(TIMING_ENGINE_ORDER),
            "requests_per_process": list(TIMING_PHASES),
            "schema": SCHEMA,
        },
        "kind": "native-inference-runtime-wall-time",
        "logs": logs,
        "ratios": {
            "dinkster_over_comfyui_cold": _timing_ratio(
                cold_medians["dinkster"], cold_medians["comfyui"]
            ),
            "dinkster_over_comfyui_warm": _timing_ratio(
                warm_medians["dinkster"], warm_medians["comfyui"]
            ),
        },
        "regression": {
            "equality_passes": True,
            "prior_dinkster_warm_median_ns": prior_median,
            "prior_dinkster_warm_ratio": (
                None if prior_ratio is None else format(prior_ratio, ".9f")
            ),
            "prior_selection": prior_selection,
            "status": regression_status,
            "warn_if_strictly_greater_than": format(TIMING_REGRESSION_WARN_RATIO, ".2f"),
        },
        "schema": SCHEMA,
        "status": "INDETERMINATE" if escalation["triggered"] else regression_status,
        "timestamp_utc": timestamp_utc,
        "workload": {
            "artifacts": [
                {key: artifact[key] for key in ("bytes", "digest", "path", "role")}
                for artifact in workload["artifacts"]
            ],
            "contract_digest": contract_digest,
            "id": workload["id"],
            "settings": settings,
        },
    }
    for log in result["logs"]:
        for kind in ("stderr", "stdout"):
            path = output_dir / log[f"{kind}_path"]
            if not path.is_file():
                raise HarnessError(f"timing log is missing: {log[f'{kind}_path']}")
            if digest_file(path) != log[f"{kind}_digest"]:
                raise HarnessError(f"timing log digest mismatch: {log[f'{kind}_path']}")
    return result


def run_timing(
    workload: dict[str, Any],
    roots: dict[str, Path],
    acceptance: dict[str, Any],
    acceptance_digest: str,
    history: dict[str, Any],
    output_dir: Path,
    device_uuid: str,
) -> dict[str, Any]:
    validate_timing_history(history)
    output_dir.mkdir(parents=True, exist_ok=False)
    timestamp_utc = dt.datetime.now(dt.UTC).isoformat()
    requests = workload_timing_requests(workload)
    attempt: dict[str, Any] = {
        "device_uuid": device_uuid,
        "engine_order": list(TIMING_ENGINE_ORDER),
        "processes": [],
        "request_phases": [request["phase"] for request in requests],
        "schema": SCHEMA,
        "status": "RUNNING",
        "timestamp_utc": timestamp_utc,
        "workload_id": workload["id"],
    }
    attempt_path = output_dir / "attempt.json"
    write_json(attempt_path, attempt)
    try:
        timing_hardware = history["hardware"]
        hardware = validate_timing_hardware(
            workload, hardware_inventory(), device_uuid, timing_hardware
        )
        for process_index, engine in enumerate(TIMING_ENGINE_ORDER, 1):
            if _git_output(roots["dinkster"], "status", "--porcelain"):
                raise HarnessError(f"dinkster checkout must be clean: {roots['dinkster']}")
            require_device_idle(hardware["device_ordinal"])
            process_dir = output_dir / f"{process_index:02d}-{engine}"
            process = run_workload(
                workload,
                engine,
                roots,
                acceptance_digest,
                process_dir,
                requests=requests,
                timing_device_uuid=device_uuid,
                timing_hardware=timing_hardware,
            )
            process["process_index"] = process_index
            attempt["processes"].append(process)
            write_json(attempt_path, attempt)
        require_device_idle(hardware["device_ordinal"])
        processes = attempt["processes"]
        assert isinstance(processes, list)
        correctness = _timing_correctness(processes, acceptance, workload)
        result = summarize_timing(
            processes,
            correctness,
            workload,
            history,
            roots,
            output_dir,
            timestamp_utc,
        )
        attempt["correctness"] = correctness
        attempt["result"] = result
        attempt["status"] = "COMPLETE"
        write_json(attempt_path, attempt)
        write_json(output_dir / "result.json", result)
        return result
    except BaseException as exc:
        attempt["error"] = str(exc)
        attempt["status"] = "INVALID"
        write_json(attempt_path, attempt)
        raise


def _workload(manifest: dict[str, Any], workload_id: str) -> dict[str, Any]:
    for workload in manifest["workloads"]:
        if workload["id"] == workload_id:
            return workload
    raise HarnessError(f"unknown workload: {workload_id}")


def _validate_sdxl_manifest_paths(
    workload: dict[str, Any],
    acceptance: dict[str, Any],
    manifest_path: Path,
    acceptance_path: Path,
) -> None:
    if workload.get("id") != "W0-SDXL-INPAINT":
        return
    canonical_manifest = Path(__file__).with_name("workloads.json").resolve()
    canonical_acceptance = (
        Path(__file__).with_name(workload["acceptance_manifest"]["path"]).resolve()
    )
    if manifest_path.resolve() != canonical_manifest:
        raise HarnessError("SDXL inpaint requires the canonical workload manifest")
    if acceptance_path.resolve() != canonical_acceptance:
        raise HarnessError("SDXL inpaint requires its digest-pinned acceptance manifest")
    if acceptance_path.read_bytes() != canonical_bytes(acceptance):
        raise HarnessError("SDXL inpaint acceptance manifest is not canonical JSON")


def _roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "comfyui": args.comfyui_root.resolve(),
        "dinkster": args.dinkster_root.resolve(),
        "template": args.template_root.resolve(),
        "artifact": args.artifact_root.resolve(),
    }


def _records_path(records: Path | None, path: Path) -> Path:
    if records is None or path.is_absolute():
        return path
    return records / path


def _records_default() -> Path | None:
    value = os.environ.get("DINKSTER_INFERENCE_PARITY_RECORDS")
    return Path(value) if value else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for command in (run,):
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--acceptance", type=Path, required=True)
        command.add_argument("--workload", required=True)
        command.add_argument("--engine", choices=ENGINES, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--comfyui-root", type=Path, required=True)
        command.add_argument("--dinkster-root", type=Path, required=True)
        command.add_argument("--template-root", type=Path, required=True)
        command.add_argument("--artifact-root", type=Path, required=True)
        command.add_argument("--records", type=Path, default=_records_default())
    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--acceptance", type=Path, required=True)
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--workload", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--records", type=Path, default=_records_default())
    gate = sub.add_parser("gate")
    gate.add_argument("--manifest", type=Path, required=True)
    gate.add_argument("--acceptance", type=Path, required=True)
    gate.add_argument("--workload", required=True)
    gate.add_argument("--output-dir", type=Path, required=True)
    gate.add_argument("--comfyui-root", type=Path, required=True)
    gate.add_argument("--dinkster-root", type=Path, required=True)
    gate.add_argument("--template-root", type=Path, required=True)
    gate.add_argument("--artifact-root", type=Path, required=True)
    gate.add_argument("--records", type=Path, default=_records_default())
    timing = sub.add_parser("time")
    timing.add_argument("--manifest", type=Path, required=True)
    timing.add_argument("--acceptance", type=Path, required=True)
    timing.add_argument("--history", type=Path, required=True)
    timing.add_argument("--workload", required=True)
    timing.add_argument("--output-dir", type=Path, required=True)
    timing.add_argument("--device-uuid", required=True)
    timing.add_argument("--comfyui-root", type=Path, required=True)
    timing.add_argument("--dinkster-root", type=Path, required=True)
    timing.add_argument("--template-root", type=Path, required=True)
    timing.add_argument("--artifact-root", type=Path, required=True)
    timing.add_argument("--records", type=Path, default=_records_default())
    args = parser.parse_args(argv)
    try:
        acceptance = load_json(args.acceptance)
        validate_acceptance(acceptance)
        acceptance_digest = digest_bytes(canonical_bytes(acceptance))
        if args.command == "compare":
            manifest = load_json(args.manifest)
            validate_manifest(manifest)
            workload = _workload(manifest, args.workload)
            _validate_sdxl_manifest_paths(workload, acceptance, args.manifest, args.acceptance)
            verdict = compare_records(
                load_json(_records_path(args.records, args.baseline)),
                load_json(_records_path(args.records, args.candidate)),
                acceptance,
                workload,
            )
            write_json(_records_path(args.records, args.output), verdict)
            return 0 if verdict["overall_pass"] else 1
        manifest = load_json(args.manifest)
        validate_manifest(manifest)
        workload = _workload(manifest, args.workload)
        _validate_sdxl_manifest_paths(workload, acceptance, args.manifest, args.acceptance)
        roots = _roots(args)
        if args.command == "run":
            output = _records_path(args.records, args.output)
            record = run_workload(workload, args.engine, roots, acceptance_digest, output.parent)
            write_json(output, record)
            return 0
        if args.command == "time":
            output_dir = _records_path(args.records, args.output_dir)
            result = run_timing(
                workload,
                roots,
                acceptance,
                acceptance_digest,
                load_json(_records_path(args.records, args.history)),
                output_dir,
                args.device_uuid,
            )
            if workload["execution"].get("timing") is not None:
                return 0 if result["status"] == "PASS" else 1
            return 1 if result["status"] == "INDETERMINATE" else 0
        output_dir = _records_path(args.records, args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = output_dir / "comfyui.json"
        candidate_path = output_dir / "dinkster.json"
        verdict_path = output_dir / "verdict.json"
        write_json(
            baseline_path,
            run_workload(workload, "comfyui", roots, acceptance_digest, output_dir),
        )
        write_json(
            candidate_path,
            run_workload(workload, "dinkster", roots, acceptance_digest, output_dir),
        )
        verdict = compare_records(
            load_json(baseline_path), load_json(candidate_path), acceptance, workload
        )
        write_json(verdict_path, verdict)
        return 0 if verdict["overall_pass"] else 1
    except HarnessError as exc:
        print(f"inference parity gate refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
