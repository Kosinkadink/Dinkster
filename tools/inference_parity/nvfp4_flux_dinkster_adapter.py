"""One-shot Dinkster adapter for the classic split-Flux NVFP4 Ada gate."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

ADAPTER_CONTRACT = "dinkster.nvfp4-classic-flux-ada-parity-adapter/v1"
ADAPTER_CONTRACT_SHA256 = hashlib.sha256(ADAPTER_CONTRACT.encode("ascii")).hexdigest()
ROLES = ("diffusion", "clip_l", "t5xxl", "vae")
TOP_LEVEL_KEYS = {
    "schema_version",
    "engine",
    "source",
    "status_api",
    "artifacts",
    "gpu",
    "workload",
    "expected_evidence",
    "adapter",
}
WORKLOAD = {
    "prompt": "a small red robot standing in a field, clear daylight",
    "seed": 42,
    "width": 256,
    "height": 256,
    "batch": 1,
    "latent_shape": [1, 16, 32, 32],
    "sampler": "dinkster.euler",
    "scheduler": "dinkster.simple",
    "steps": 2,
    "denoise": 1.0,
    "cfg": 1.0,
    "flux_guidance": 3.5,
    "diffusion_dtype": "bfloat16",
    "text_dtype": "float32",
    "vae_dtype": "float32",
    "diffusion_residency": "explicit_cpu_offload_reload",
    "phases": ["warmup", "real"],
    "attempts_per_engine": 1,
    "retry": False,
    "discard": False,
    "substitution": False,
}
EXPECTED_EVIDENCE = {
    "nvfp4_linear_layers": 152,
    "fp8_linear_layers": 114,
    "route_pre_sm10_calls": 304,
    "dequantize_nvfp4_calls": 304,
    "f_linear_nvfp4_calls": 304,
    "native_nvfp4_calls": 0,
    "quantize_nvfp4_calls": 0,
    "scaled_mm_nvfp4_calls": 0,
    "fallback_calls": 0,
    "error_calls": 0,
    "selected_backend": "dequantize_nvfp4_plus_f_linear",
    "complete_dequantize_and_f_linear_accounting": True,
    "representative_direct_kitchen_exact_equality": True,
    "option_a_phase_local_object_identity_exact": True,
    "option_a_cross_transition_bytes_and_logical_metadata_exact": True,
    "option_a_final_pre_release_exact": True,
    "option_a_single_handle_runtime_assembled_recorder_identity": True,
    "warmup_real_output_bytes_identical": True,
    "metrics": [
        "load_seconds",
        "generation_seconds",
        "peak_rss_bytes",
        "peak_vram_bytes",
    ],
    "cleanup": [
        "runtime_released",
        "cpu_offload_complete",
        "adapter_process_exited",
        "no_residual_gpu_process",
    ],
}
SHA256_LENGTH = 64
GIT_SHA_LENGTH = 40
ASSET_DIGEST_PREFIX = "blake3:"
STATUS_MODULE = "dinkster_inference_torch._nvfp4_diagnostics"
STATUS_SYMBOL = "nvfp4_runtime_status"
SCALE_NAMES = ("weight_scale", "weight_scale_2", "input_scale", "pre_quant_scale")
PROGRESS_KEYS = {
    "schema_version",
    "status",
    "sequence",
    "operation",
    "layers_complete",
    "layers_total",
    "bytes_complete",
    "elapsed_seconds",
    "exception_type",
}


class AdapterError(RuntimeError):
    """A fail-closed adapter contract refusal."""


def _fail(message: str) -> NoReturn:
    raise AdapterError(message)


def _exact_dict(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        _fail(f"{name} keys differ")
    return value


def _string(value: Any, name: str, *, length: int | None = None) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        _fail(f"{name} must be a non-empty ASCII string")
    if length is not None and (
        len(value) != length or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} has an invalid digest")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}")
    return value


def validate_request(value: Any) -> dict[str, Any]:
    request = _exact_dict(value, TOP_LEVEL_KEYS, "request")
    if request["schema_version"] != 1 or request["engine"] != "dinkster":
        _fail("request identity differs")
    source = _exact_dict(request["source"], {"repo_path", "sha", "tree"}, "source")
    _string(source["repo_path"], "source repo_path")
    _string(source["sha"], "source sha", length=GIT_SHA_LENGTH)
    _string(source["tree"], "source tree", length=GIT_SHA_LENGTH)
    status_api = _exact_dict(
        request["status_api"], {"module", "symbol", "contract_sha256"}, "status_api"
    )
    if status_api["module"] != STATUS_MODULE or status_api["symbol"] != STATUS_SYMBOL:
        _fail("status_api must name the exact private runtime status function")
    _string(status_api["contract_sha256"], "status_api contract", length=SHA256_LENGTH)
    adapter = _exact_dict(
        request["adapter"], {"module", "symbol", "contract_sha256", "sha256"}, "adapter"
    )
    for key in ("module", "symbol"):
        _string(adapter[key], f"adapter {key}")
    for key in ("contract_sha256", "sha256"):
        _string(adapter[key], f"adapter {key}", length=SHA256_LENGTH)
    if adapter["contract_sha256"] != ADAPTER_CONTRACT_SHA256:
        _fail("adapter contract identity differs")
    artifacts = _exact_dict(request["artifacts"], set(ROLES), "artifacts")
    for role in ROLES:
        pin = _exact_dict(artifacts[role], {"path", "size", "sha256", "blake3"}, f"artifact {role}")
        _string(pin["path"], f"artifact {role} path")
        _integer(pin["size"], f"artifact {role} size", minimum=1)
        _string(pin["sha256"], f"artifact {role} sha256", length=SHA256_LENGTH)
        blake3 = _string(pin["blake3"], f"artifact {role} blake3")
        if not blake3.startswith(ASSET_DIGEST_PREFIX):
            _fail(f"artifact {role} blake3 is not canonical")
        _string(
            blake3.removeprefix(ASSET_DIGEST_PREFIX),
            f"artifact {role} blake3",
            length=SHA256_LENGTH,
        )
    gpu = _exact_dict(request["gpu"], {"index", "uuid"}, "gpu")
    _integer(gpu["index"], "gpu index")
    _string(gpu["uuid"], "gpu uuid")
    if request["workload"] != WORKLOAD:
        _fail("workload pins differ")
    evidence = _exact_dict(
        request["expected_evidence"],
        set(EXPECTED_EVIDENCE),
        "expected_evidence",
    )
    nvfp4_layers = _integer(evidence["nvfp4_linear_layers"], "expected NVFP4 layers", minimum=1)
    fp8_layers = _integer(evidence["fp8_linear_layers"], "expected FP8 layers", minimum=1)
    dequantize_calls = _integer(
        evidence["dequantize_nvfp4_calls"], "expected NVFP4 dequantize calls", minimum=1
    )
    if nvfp4_layers != 152 or fp8_layers != 114:
        _fail("classic Flux linear inventory pins differ")
    if dequantize_calls != 304 or dequantize_calls != nvfp4_layers * WORKLOAD["steps"]:
        _fail("expected NVFP4 dequantize call count differs")
    for key in (
        "route_pre_sm10_calls",
        "f_linear_nvfp4_calls",
        "native_nvfp4_calls",
        "quantize_nvfp4_calls",
        "scaled_mm_nvfp4_calls",
        "fallback_calls",
        "error_calls",
    ):
        _integer(evidence[key], f"expected evidence {key}")
    for key in (
        "complete_dequantize_and_f_linear_accounting",
        "representative_direct_kitchen_exact_equality",
        "option_a_phase_local_object_identity_exact",
        "option_a_cross_transition_bytes_and_logical_metadata_exact",
        "option_a_final_pre_release_exact",
        "option_a_single_handle_runtime_assembled_recorder_identity",
        "warmup_real_output_bytes_identical",
    ):
        if evidence[key] is not True:
            _fail(f"expected evidence {key} differs")
    if evidence != EXPECTED_EVIDENCE:
        _fail("expected evidence pins differ")
    return request


class _DigestResolver:
    def __init__(self, paths: dict[str, Path]) -> None:
        self._paths = dict(paths)

    def resolve(self, digest: str) -> Path | None:
        return self._paths.get(digest)


def _runtime_imports() -> SimpleNamespace:
    import torch  # pyright: ignore[reportMissingImports]
    from dinkster_assets import AssetRef
    from dinkster_compat_comfy.native_arm import load_native_runtime_handle
    from dinkster_compat_comfy.native_residency import NativeRuntimeHandle
    from dinkster_inference import SamplingGuidance
    from dinkster_workers.execution import ExecutionContext, use_execution_context

    return SimpleNamespace(
        torch=torch,
        AssetRef=AssetRef,
        load_native_runtime_handle=load_native_runtime_handle,
        NativeRuntimeHandle=NativeRuntimeHandle,
        SamplingGuidance=SamplingGuidance,
        ExecutionContext=ExecutionContext,
        use_execution_context=use_execution_context,
    )


def _assets(
    request: dict[str, Any], api: SimpleNamespace
) -> tuple[dict[str, Any], _DigestResolver]:
    paths = {
        request["artifacts"][role]["blake3"]: Path(request["artifacts"][role]["path"])
        for role in ROLES
    }
    resolver = _DigestResolver(paths)
    assets: dict[str, Any] = {}
    for role in ROLES:
        pin = request["artifacts"][role]
        asset = api.AssetRef(
            digest=pin["blake3"],
            name=role,
            size=pin["size"],
            resolver=resolver,
        )
        assets[role] = asset
    return assets, resolver


def _canonical_load(request: dict[str, Any], api: SimpleNamespace) -> Any:
    assets, _ = _assets(request, api)
    context = api.ExecutionContext("compat@native", None, fp8_matmul=False)
    with api.use_execution_context(context):
        handle = api.load_native_runtime_handle(assets)
    try:
        if not isinstance(handle, api.NativeRuntimeHandle):
            _fail("canonical runtime loader returned an unexpected handle")
        return handle
    except BaseException as primary:
        if isinstance(handle, api.NativeRuntimeHandle) and not handle.released:
            try:
                _terminal_release(handle)
            except BaseException as cleanup:
                primary.add_note(f"native loader cleanup also failed: {cleanup!r}")
        raise


def _status(runtime: object) -> Any:
    from dinkster_inference_torch._nvfp4_diagnostics import nvfp4_runtime_status

    return nvfp4_runtime_status(runtime)


def _raw_tensor_bytes(tensor: Any) -> bytes:
    contiguous = tensor.detach().contiguous().cpu()
    nbytes = contiguous.numel() * contiguous.element_size()
    if nbytes == 0:
        return b""
    return ctypes.string_at(contiguous.data_ptr(), nbytes)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _aggregate_digest(items: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, data in items:
        encoded = name.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _nvfp4_modules(runtime: Any) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
    nvfp4: list[tuple[str, Any]] = []
    fp8: list[tuple[str, Any]] = []
    for name, module in runtime.assembled.diffusion.named_modules():
        kind = type(module).__name__
        if kind == "Nvfp4Linear":
            nvfp4.append((name, module))
        elif kind == "Fp8Linear":
            fp8.append((name, module))
    if not nvfp4:
        _fail("assembled diffusion contains no Nvfp4Linear")
    return nvfp4, fp8


def _inventory(runtime: Any) -> dict[str, int]:
    nvfp4, fp8 = _nvfp4_modules(runtime)
    return {
        "nvfp4_linear_layers": len(nvfp4),
        "fp8_linear_layers": len(fp8),
    }


class _ProgressSink:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._started = time.monotonic()
        self._sequence = 0

    def emit(
        self,
        operation: str,
        status: str,
        *,
        layers_complete: int = 0,
        layers_total: int = 0,
        bytes_complete: int = 0,
        exception_type: str | None = None,
    ) -> None:
        if self._path is None:
            return
        self._sequence += 1
        event = {
            "schema_version": 1,
            "status": status,
            "sequence": self._sequence,
            "operation": operation,
            "layers_complete": layers_complete,
            "layers_total": layers_total,
            "bytes_complete": bytes_complete,
            "elapsed_seconds": time.monotonic() - self._started,
            "exception_type": exception_type,
        }
        if set(event) != PROGRESS_KEYS:
            raise AssertionError("progress event keys differ")
        _atomic_write(self._path, event)
        print(_canonical_bytes(event).decode("ascii").rstrip("\n"), file=sys.stderr, flush=True)


def _transition_snapshot(runtime: Any, progress: _ProgressSink | None = None) -> dict[str, Any]:
    nvfp4, _ = _nvfp4_modules(runtime)
    layers_total = len(nvfp4)
    bytes_complete = 0
    if progress is not None:
        progress.emit(
            "transition_snapshot",
            "started",
            layers_total=layers_total,
        )
    layers: dict[str, Any] = {}
    for layer_index, (name, module) in enumerate(nvfp4, start=1):
        tensors: dict[str, Any] = {}
        for tensor_name in ("weight", *SCALE_NAMES):
            tensor = getattr(module, tensor_name, None)
            if tensor is not None:
                data = _raw_tensor_bytes(tensor)
                bytes_complete += len(data)
                tensors[tensor_name] = {
                    "key": f"{name}.{tensor_name}",
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "size": len(data),
                    "sha256": _sha256(data),
                }
        layers[name] = {
            "geometry": (module.out_features, module.in_features),
            "tensors": tensors,
        }
        if progress is not None:
            progress.emit(
                "transition_snapshot",
                "progress",
                layers_complete=layer_index,
                layers_total=layers_total,
                bytes_complete=bytes_complete,
            )
    if progress is not None:
        progress.emit(
            "transition_snapshot",
            "completed",
            layers_complete=layers_total,
            layers_total=layers_total,
            bytes_complete=bytes_complete,
        )
    return layers


def _loaded_identity(runtime: Any) -> tuple[str, dict[str, Any]]:
    nvfp4, _ = _nvfp4_modules(runtime)
    objects: dict[str, Any] = {}
    for name, module in nvfp4:
        for tensor_name in ("weight", *SCALE_NAMES):
            tensor = getattr(module, tensor_name, None)
            if tensor is not None:
                objects[f"{name}.{tensor_name}"] = tensor
    digest = _aggregate_digest(
        [(key, str(id(value)).encode("ascii")) for key, value in objects.items()]
    )
    return digest, objects


def _require_same_loaded_objects(before: dict[str, Any], after: dict[str, Any]) -> None:
    if before.keys() != after.keys() or any(before[key] is not after[key] for key in before):
        _fail("NVFP4 tensor object identity changed within loaded phase")


def _object_identity_digest(objects: dict[str, Any], tensor_name: str) -> str:
    suffix = f".{tensor_name}"
    selected = [
        (key, str(id(value)).encode("ascii"))
        for key, value in objects.items()
        if key.endswith(suffix)
    ]
    if not selected:
        _fail(f"loaded NVFP4 state has no {tensor_name} objects")
    return _aggregate_digest(selected)


def _loaded_object_identity(objects: dict[str, Any]) -> dict[str, str]:
    all_scales = [
        (key, str(id(value)).encode("ascii"))
        for key, value in objects.items()
        if any(key.endswith(f".{scale_name}") for scale_name in SCALE_NAMES)
    ]
    if not all_scales:
        _fail("loaded NVFP4 state has no scale objects")
    return {
        "qdata_object_identity_sha256": _object_identity_digest(objects, "weight"),
        "block_scale_object_identity_sha256": _object_identity_digest(objects, "weight_scale"),
        "tensor_scale_object_identity_sha256": _object_identity_digest(objects, "weight_scale_2"),
        "all_scale_object_identity_sha256": _aggregate_digest(all_scales),
    }


def _snapshot_digest(entries: list[tuple[str, Any]]) -> str:
    return _aggregate_digest(
        [
            (name, json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii"))
            for name, value in entries
        ]
    )


def _transition_result(snapshot: dict[str, Any]) -> dict[str, str]:
    qdata: list[tuple[str, bytes]] = []
    scales: list[tuple[str, bytes]] = []
    keys: list[tuple[str, Any]] = []
    shapes: list[tuple[str, Any]] = []
    dtypes: list[tuple[str, Any]] = []
    geometry: list[tuple[str, Any]] = []
    for layer_name, layer in snapshot.items():
        geometry.append((layer_name, layer["geometry"]))
        for tensor_name, tensor in layer["tensors"].items():
            qualified = f"{layer_name}.{tensor_name}"
            target = qdata if tensor_name == "weight" else scales
            target.append((qualified, bytes.fromhex(tensor["sha256"])))
            keys.append((qualified, tensor["key"]))
            shapes.append((qualified, tensor["shape"]))
            dtypes.append((qualified, tensor["dtype"]))
    return {
        "packed_qdata_bytes_sha256": _aggregate_digest(qdata),
        "all_scale_bytes_sha256": _aggregate_digest(scales),
        "logical_keys_sha256": _snapshot_digest(keys),
        "logical_shapes_sha256": _snapshot_digest(shapes),
        "logical_dtypes_sha256": _snapshot_digest(dtypes),
        "logical_geometry_sha256": _snapshot_digest(geometry),
    }


def _literal_identity_sha256(value: Any) -> str:
    return _sha256(str(id(value)).encode("ascii"))


def _single_object_identity(identity: _ProcessIdentity) -> dict[str, Any]:
    return {
        "handle_identity_sha256": _literal_identity_sha256(identity.handle),
        "runtime_identity_sha256": _literal_identity_sha256(identity.runtime),
        "assembled_identity_sha256": _literal_identity_sha256(identity.assembled),
        "recorder_identity_sha256": _literal_identity_sha256(identity.recorder),
        "exact": True,
    }


@dataclass(frozen=True)
class _ProcessIdentity:
    handle: Any
    runtime: Any
    assembled: Any
    recorder: Any
    recipe_sources: tuple[tuple[str, str], ...]
    locator: tuple[tuple[str, str], ...]
    module_names: tuple[str, ...]
    baseline: dict[str, Any]

    @classmethod
    def capture(
        cls,
        handle: Any,
        request: dict[str, Any],
        progress: _ProgressSink | None = None,
    ) -> _ProcessIdentity:
        runtime = handle.runtime
        assembled = runtime.assembled
        recorder = getattr(assembled.diffusion, "_nvfp4_diagnostics", None)
        if recorder is None:
            _fail("runtime has no NVFP4 diagnostics recorder")
        recipe_sources = tuple(
            (binding.role, binding.source.digest) for binding in handle.recipe.sources
        )
        expected_sources = tuple(
            sorted((role, request["artifacts"][role]["blake3"]) for role in ROLES)
        )
        if recipe_sources != expected_sources:
            _fail("canonical recipe source bindings differ from the request")
        locator = tuple(
            sorted(
                (request["artifacts"][role]["blake3"], request["artifacts"][role]["path"])
                for role in ROLES
            )
        )
        baseline = _transition_snapshot(runtime, progress)
        return cls(
            handle,
            runtime,
            assembled,
            recorder,
            recipe_sources,
            locator,
            tuple(baseline),
            baseline,
        )

    def check(
        self,
        request: dict[str, Any],
        progress: _ProgressSink | None = None,
        *,
        transition: bool = False,
    ) -> dict[str, Any]:
        if self.handle.runtime is not self.runtime or self.runtime.assembled is not self.assembled:
            _fail("native handle/runtime/assembled identity changed")
        if getattr(self.assembled.diffusion, "_nvfp4_diagnostics", None) is not self.recorder:
            _fail("NVFP4 diagnostics recorder identity changed")
        sources = tuple(
            (binding.role, binding.source.digest) for binding in self.handle.recipe.sources
        )
        if sources != self.recipe_sources:
            _fail("canonical recipe bindings changed")
        locator = tuple(
            sorted(
                (request["artifacts"][role]["blake3"], request["artifacts"][role]["path"])
                for role in ROLES
            )
        )
        if locator != self.locator:
            _fail("request digest locator changed")
        snapshot = _transition_snapshot(self.runtime, progress)
        if tuple(snapshot) != self.module_names:
            _fail("named NVFP4 module set changed")
        if transition and snapshot != self.baseline:
            _fail("NVFP4 packed byte or semantic transition state changed")
        return snapshot

    def status(self, request: dict[str, Any], progress: _ProgressSink | None = None) -> Any:
        self.check(request, progress)
        snapshot = _status(self.runtime)
        self.check(request, progress)
        return snapshot


def _counter_delta(before: Any, after: Any, expected: int) -> dict[str, int]:
    if before.active != 0 or after.active != 0:
        _fail("NVFP4 diagnostics contain active invocations")
    if not after.completed or any(item.terminal != "success" for item in after.completed):
        _fail("NVFP4 diagnostics lack complete successful invocations")
    keys = set(before.lifetime) | set(after.lifetime)
    delta = {key: after.lifetime.get(key, 0) - before.lifetime.get(key, 0) for key in keys}
    if any(value < 0 for value in delta.values()):
        _fail("NVFP4 lifetime counters regressed")
    forbidden = (
        "quantize_error",
        "scaled_mm_error",
        "dequantize_error",
        "requantize_error",
        "route_native",
        "route_full_precision",
        "route_non_cuda",
        "route_rank",
        "route_no_quantize_backend",
        "route_no_scaled_mm_backend",
        "route_backend_fallback",
        "route_deferred_patch",
        "requantize_success",
    )
    if any(delta.get(key, 0) for key in forbidden):
        _fail("NVFP4 diagnostics report an error or ambiguous route")
    dequantize = delta.get("dequantize_success", 0)
    if dequantize != expected or delta.get("route_pre_sm10", 0) != expected:
        _fail("Ada route/dequantize accounting differs")
    if delta.get("quantize_success", 0) or delta.get("scaled_mm_success", 0):
        _fail("Ada execution used native NVFP4 kernels")
    if dequantize < len(after.completed):
        _fail("NVFP4 lifetime delta does not cover retained completed history")
    for item in after.completed:
        counters = item.counters
        if counters.get("route_pre_sm10", 0) != 1 or counters.get("dequantize_success", 0) != 1:
            _fail("retained NVFP4 invocation does not prove one Ada dequantization")
        if any(counters.get(key, 0) for key in forbidden):
            _fail("retained NVFP4 invocation contains an ambiguous route or error")
        if counters.get("quantize_success", 0) or counters.get("scaled_mm_success", 0):
            _fail("retained NVFP4 invocation used native kernels")
    return delta


def _representative(
    layer_name: str, layer: Any, captured: dict[str, Any], torch: Any
) -> dict[str, Any]:
    if set(captured) != {"input", "output"}:
        _fail("representative NVFP4 layer did not capture one runtime invocation")
    import dinkster_kitchen  # pyright: ignore[reportMissingImports]

    value = captured["input"]
    if layer.pre_quant_scale is not None:
        value = value * layer.pre_quant_scale.to(dtype=value.dtype)
    dequantized = dinkster_kitchen.dequantize_nvfp4(
        layer.weight,
        layer.weight_scale_2,
        layer.weight_scale,
        output_type=layer.compute_dtype,
    )[: layer.out_features, : layer.in_features]
    bias = None if layer.bias is None else layer.bias.to(dtype=layer.compute_dtype)
    direct = torch.nn.functional.linear(value, dequantized, bias)
    direct_bytes = _raw_tensor_bytes(direct)
    runtime_bytes = _raw_tensor_bytes(captured["output"])
    if direct_bytes != runtime_bytes:
        _fail("representative Kitchen dequantize plus F.linear output differs")
    return {
        "layer_id": layer_name,
        "direct_sha256": _sha256(direct_bytes),
        "runtime_sha256": _sha256(runtime_bytes),
        "exact": True,
    }


def _peak_rss_bytes() -> int:
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _terminal_release(handle: Any) -> None:
    if handle.released:
        _fail("native runtime was released before final cleanup")
    handle.terminal_release()
    if not handle.released:
        _fail("native runtime terminal release did not complete")


def _cleanup_runtime(handle: Any, before_release: Callable[[], None] | None = None) -> None:
    if handle.released:
        _fail("native runtime was released before final cleanup")
    cleanup_error: BaseException | None = None
    if before_release is not None and not handle.released:
        try:
            before_release()
        except BaseException as error:
            cleanup_error = error
    if not handle.released:
        try:
            _terminal_release(handle)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
            else:
                cleanup_error.add_note(f"terminal release also failed: {error!r}")
    if cleanup_error is not None:
        raise cleanup_error


def _execute(request: dict[str, Any], progress: _ProgressSink | None = None) -> dict[str, Any]:
    api = _runtime_imports()
    torch = api.torch
    torch.cuda.reset_peak_memory_stats()
    load_started = time.monotonic()
    if progress is not None:
        progress.emit("canonical_load", "started")
    handle = _canonical_load(request, api)
    load_seconds = time.monotonic() - load_started
    phases: list[dict[str, Any]] = []
    phase_identities: dict[str, Any] = {}
    released = False
    identity: _ProcessIdentity | None = None
    cross_transition_before: dict[str, str] | None = None
    final_transition: dict[str, str] | None = None
    single_identity: dict[str, Any] | None = None

    def cleanup() -> None:
        nonlocal final_transition, released

        def final_snapshot() -> None:
            nonlocal final_transition
            if identity is not None:
                final_transition = _transition_result(
                    identity.check(request, progress, transition=True)
                )

        _cleanup_runtime(handle, final_snapshot)
        released = handle.released

    try:
        if progress is not None:
            progress.emit("canonical_load", "completed")
        identity = _ProcessIdentity.capture(handle, request, progress)
        cross_transition_before = _transition_result(identity.baseline)
        single_identity = _single_object_identity(identity)
        identity.check(request, progress, transition=True)
        with handle.stage("text"):
            with torch.inference_mode():
                cond = handle.runtime.encode_text(request["workload"]["prompt"])
                uncond = handle.runtime.encode_text("")
            identity.check(request, progress, transition=True)
        handle.advisory_unload()
        identity.check(request, progress, transition=True)
        latent = torch.zeros(
            tuple(request["workload"]["latent_shape"]),
            dtype=torch.float32,
            device=handle.load_device,
        )
        expected_inventory = {
            "nvfp4_linear_layers": request["expected_evidence"]["nvfp4_linear_layers"],
            "fp8_linear_layers": request["expected_evidence"]["fp8_linear_layers"],
        }
        if expected_inventory["nvfp4_linear_layers"] != 152:
            _fail("classic Flux NVFP4 inventory pin differs")
        expected_operations = request["expected_evidence"]["dequantize_nvfp4_calls"]
        if expected_operations != (
            expected_inventory["nvfp4_linear_layers"] * request["workload"]["steps"]
        ):
            _fail("classic Flux phase operation derivation differs")
        for phase_name in ("warmup", "real"):
            identity.check(request, progress, transition=True)
            inventory = _inventory(identity.runtime)
            if inventory != expected_inventory:
                _fail("assembled linear inventory differs from request")
            nvfp4, _ = _nvfp4_modules(identity.runtime)
            layer_name, layer = nvfp4[0]
            captured: dict[str, Any] = {}

            def capture(
                _module: Any,
                inputs: tuple[Any, ...],
                output: Any,
                phase_capture: dict[str, Any] = captured,
            ) -> None:
                if not phase_capture:
                    phase_capture["input"] = inputs[0].detach().clone()
                    phase_capture["output"] = output.detach().clone()

            hook = layer.register_forward_hook(capture)
            generation_started = time.monotonic()
            try:
                with handle.stage("diffusion"):
                    loaded_before = identity.check(request, progress, transition=True)
                    identity_before, objects_before = _loaded_identity(identity.runtime)
                    before = identity.status(request, progress)
                    mechanisms_loaded = any(item.loaded_bytes() > 0 for item in handle.mechanisms)
                    with torch.inference_mode():
                        sampled = handle.runtime.sample(
                            latent,
                            cond=cond,
                            cfg=api.SamplingGuidance(uncond, request["workload"]["cfg"]),
                            sampler_id=request["workload"]["sampler"],
                            scheduler_id=request["workload"]["scheduler"],
                            steps=request["workload"]["steps"],
                            denoise=request["workload"]["denoise"],
                            seed=request["workload"]["seed"],
                            guidance=request["workload"]["flux_guidance"],
                            compute_dtype=torch.bfloat16,
                            device=handle.load_device,
                        )
                    after = identity.status(request, progress)
                    identity_after, objects_after = _loaded_identity(identity.runtime)
                    _require_same_loaded_objects(objects_before, objects_after)
                    if identity_before != identity_after:
                        _fail("NVFP4 tensor identity digest changed within loaded phase")
                    loaded_after = identity.check(request, progress, transition=True)
                    representative = _representative(layer_name, layer, captured, torch)
            finally:
                hook.remove()
            delta = _counter_delta(before, after, expected_operations)
            handle.advisory_unload()
            identity.check(request, progress, transition=True)
            diffusion_offloaded = all(item.loaded_bytes() == 0 for item in handle.mechanisms)
            with handle.stage("vae"):
                with torch.inference_mode():
                    image = handle.runtime.decode_latent(sampled).permute(0, 2, 3, 1)
                    output_bytes = _raw_tensor_bytes(image.detach().to(dtype=torch.float32))
            handle.advisory_unload()
            identity.check(request, progress, transition=True)
            all_offloaded = all(item.loaded_bytes() == 0 for item in handle.mechanisms)
            generation_seconds = time.monotonic() - generation_started
            if loaded_after != loaded_before:
                _fail("loaded-phase packed NVFP4 state changed")
            object_before = _loaded_object_identity(objects_before)
            object_after = _loaded_object_identity(objects_after)
            if object_after != object_before:
                _fail("loaded-phase NVFP4 object identity changed")
            phase_identities[phase_name] = {
                "before": object_before,
                "after": object_after,
                "exact": True,
            }
            phases.append(
                {
                    "name": phase_name,
                    "output": {"sha256": _sha256(output_bytes), "size": len(output_bytes)},
                    "inventory": inventory,
                    "adapter_lifetime_deltas": {
                        "route_pre_sm10_calls": delta["route_pre_sm10"],
                        "dequantize_nvfp4_calls": delta["dequantize_success"],
                        "f_linear_nvfp4_calls": delta["dequantize_success"],
                        "native_nvfp4_calls": delta.get("route_native", 0),
                        "quantize_nvfp4_calls": delta.get("quantize_success", 0),
                        "scaled_mm_nvfp4_calls": delta.get("scaled_mm_success", 0),
                        "fallback_calls": delta.get("route_backend_fallback", 0),
                        "error_calls": sum(
                            value for key, value in delta.items() if key.endswith("_error")
                        ),
                        "selected_backend": "dequantize_nvfp4_plus_f_linear",
                        "complete": True,
                    },
                    "representative_equality": representative,
                    "offload_reload": {
                        "cpu_offload": diffusion_offloaded and all_offloaded,
                        "reload": mechanisms_loaded,
                        "accounting_exact": (
                            diffusion_offloaded and all_offloaded and mechanisms_loaded
                        ),
                    },
                    "metrics": {
                        "load_seconds": load_seconds,
                        "generation_seconds": generation_seconds,
                        "peak_rss_bytes": _peak_rss_bytes(),
                        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                    },
                    "cleanup": {"runtime_released": True, "cpu_offload_complete": all_offloaded},
                }
            )
        if phases[0]["output"] != phases[1]["output"]:
            _fail("warmup and real output bytes differ")
    except BaseException as primary:
        try:
            cleanup()
        except BaseException as cleanup_error:
            cleanup_notes = "; ".join(getattr(cleanup_error, "__notes__", ()))
            suffix = f"; {cleanup_notes}" if cleanup_notes else ""
            primary.add_note(f"runtime cleanup also failed: {cleanup_error!r}{suffix}")
        raise
    else:
        cleanup()
    if not released:
        _fail("native runtime was not terminally released")
    if (
        cross_transition_before is None
        or final_transition is None
        or single_identity is None
        or final_transition != cross_transition_before
    ):
        _fail("final NVFP4 transition state differs")
    return {
        "schema_version": 1,
        "engine": "dinkster",
        "status": "passed",
        "adapter": request["adapter"],
        "status_api": request["status_api"],
        "phases": phases,
        "state": {
            "phase_local_loaded_object_identity": phase_identities,
            "cross_transition": {
                "before": cross_transition_before,
                "after": final_transition,
                "exact": True,
            },
            "final_pre_release": final_transition,
            "single_object_identity": single_identity,
        },
    }


def _prepare(request: dict[str, Any], progress: _ProgressSink | None = None) -> dict[str, Any]:
    api = _runtime_imports()
    load_started = time.monotonic()
    if progress is not None:
        progress.emit("canonical_load", "started")
    handle = _canonical_load(request, api)
    load_seconds = time.monotonic() - load_started
    identity: _ProcessIdentity | None = None
    inventory: dict[str, int] | None = None
    transition: dict[str, str] | None = None
    single_identity: dict[str, Any] | None = None
    snapshot_seconds: float | None = None
    try:
        if progress is not None:
            progress.emit("canonical_load", "completed")
        snapshot_started = time.monotonic()
        identity = _ProcessIdentity.capture(handle, request, progress)
        snapshot_seconds = time.monotonic() - snapshot_started
        transition = _transition_result(identity.baseline)
        inventory = _inventory(identity.runtime)
        single_identity = _single_object_identity(identity)
    except BaseException as primary:
        try:
            _cleanup_runtime(handle)
        except BaseException as cleanup_error:
            primary.add_note(f"runtime cleanup also failed: {cleanup_error!r}")
        raise
    else:
        _cleanup_runtime(handle)
    if not handle.released:
        _fail("native runtime was not terminally released")
    if (
        identity is None
        or inventory is None
        or transition is None
        or single_identity is None
        or snapshot_seconds is None
    ):
        _fail("preparation result is incomplete")
    return {
        "schema_version": 1,
        "engine": "dinkster",
        "status": "prepared",
        "adapter": request["adapter"],
        "status_api": request["status_api"],
        "inventory": inventory,
        "transition": transition,
        "single_object_identity": single_identity,
        "metrics": {
            "load_seconds": load_seconds,
            "snapshot_seconds": snapshot_seconds,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
        "cleanup": {"runtime_released": True},
    }


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _read_request(path: Path) -> dict[str, Any]:
    try:
        return validate_request(json.loads(path.read_text(encoding="ascii")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"invalid request JSON: {error}")


def _atomic_write(path: Path, value: Any) -> None:
    data = _canonical_bytes(value)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _run(
    request: dict[str, Any] | Path,
    result: Path,
    progress_path: Path | None,
    *,
    prepare_only: bool = False,
) -> None:
    progress = _ProgressSink(progress_path)
    try:
        validated = _read_request(request) if isinstance(request, Path) else request
        value = (_prepare if prepare_only else _execute)(validated, progress)
        _atomic_write(result, value)
        progress.emit("terminal", "passed")
    except BaseException as primary:
        try:
            progress.emit("terminal", "failed", exception_type=type(primary).__name__)
        except BaseException as progress_error:
            primary.add_note(f"progress persistence also failed: {progress_error!r}")
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    _run(args.request, args.result, args.progress, prepare_only=args.prepare_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
