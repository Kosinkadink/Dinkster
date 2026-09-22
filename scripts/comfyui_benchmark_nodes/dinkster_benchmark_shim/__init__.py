"""ComfyUI-side instrumentation for the Dinkster-vs-ComfyUI benchmark.

Loaded into a pinned ComfyUI checkout as a custom node package through an
extra_model_paths custom_nodes entry, so the checkout itself stays
unmodified. dinkster-evidence's scripts/benchmark_comfyui.py launches the server with that
configuration and reads the instrumentation over HTTP.

This module must stay self-contained: it executes inside the ComfyUI
process and venv, where no Dinkster package is installed. The host, driver,
device, and peak-RSS helpers therefore mirror the ones in
dinkster-evidence's scripts/benchmark_inference.py instead of importing them.

What it provides:

- ProgressRegistry's start/finish methods are wrapped at class level to
  timestamp node execution server-side. The execution loop calls them
  on the execution thread for every node regardless of whether a
  websocket client is attached (the "executing" messages are
  client-gated), and class-level wrapping survives the per-prompt
  registry rebuild that drops registered handlers. The torch device is
  synchronized before a start or finish is timestamped, so each node's
  start-to-finish interval is synchronized wall clock - the same phase
  contract the Dinkster runner measures. A cached node records only a
  finish, which lets the driver tell reuse from re-execution.
- ProgressBar.update_absolute is wrapped to record per-step boundaries
  (no synchronize, matching the Dinkster runner's dispatch-side step
  boundaries), attributed through get_executing_context(). The
  ProgressRegistry sees only ProgressBar's throttled hook calls (100ms
  minimum interval), so it cannot be the step-boundary source. Each
  event carries the sequence max because one node can report several
  progress sequences (model weight loading and sampling both tick on
  the sampler node).
- The peak of device-global used bytes (mem_get_info) is tracked at
  synchronized node boundaries and progress ticks, because ComfyUI's
  dynamic VRAM loading holds weights outside the caching allocator
  where allocator peaks cannot see them.
- DinksterBenchmarkSink records decoded IMAGE and optional AUDIO finiteness
  and shape. When armed after memory measurement, it also writes spatially
  strided float32 NPY quality evidence from one unmeasured prompt.
- /dinkster_benchmark HTTP routes expose backend identity (with the same
  explicit admission the Dinkster runner performs), the recorded events and
  sink observations, allocator telemetry, a reset, and an unload that
  releases through ComfyUI's free path and reports the residual.
"""

import asyncio
import gc
import hashlib
import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import comfy.model_management
import comfy.utils
import execution
import latent_preview
import numpy as np
import torch
from aiohttp import web
from comfy_execution.progress import ProgressRegistry
from comfy_execution.utils import get_executing_context
from PIL import Image
from server import PromptServer

_LOCK = threading.Lock()
_EVENTS: list[dict[str, object]] = []
_OBSERVATIONS: list[dict[str, object]] = []
_PROGRESS_SEEN: dict[tuple[object, object], tuple[float, float]] = {}
_PEAK_DEVICE_USED = 0
_QUALITY_CAPTURED = False
_QUALITY_CAPTURE_ARMED = False
_QUALITY_CAPTURE_SEED: int | None = None
_FIRST_STEP_CAPTURE = None
_SECOND_STEP_CAPTURE = None
_H3_FORWARD_TRACE = None
_H3_FORWARD_ORIGINAL = None
_ALLOCATOR_WINDOW: dict[str, object] | None = None
_ATTENTION_COUNTS = {
    "provider_attempts": 0,
    "provider_successes": 0,
    "provider_exceptions": 0,
    "fallback_calls": 0,
}

#: The launching driver's per-boot nonce, echoed through the identity
#: route so the driver can tell this fresh server from a stale one
#: orphaned on the same port.
_BOOT_NONCE = os.environ.get("DINKSTER_BENCHMARK_BOOT_NONCE", "")
_PROCESS_INSTANCE = f"{os.getpid()}:{time.time_ns()}:{_BOOT_NONCE}"
_ATTENTION_POLICY = os.environ.get("DINKSTER_BENCHMARK_ATTENTION_POLICY", "auto")
_QUALITY_OUTPUT_DIR = os.environ.get("DINKSTER_BENCHMARK_QUALITY_OUTPUT_DIR")
_QUALITY_SPATIAL_STRIDE = int(os.environ.get("DINKSTER_BENCHMARK_QUALITY_SPATIAL_STRIDE", "4"))

_original_prepare_callback = latent_preview.prepare_callback


def _capture_prepare_callback(model, steps, x0_output):
    callback = _original_prepare_callback(model, steps, x0_output)

    def capture(step, denoised, current, total):
        global _FIRST_STEP_CAPTURE, _SECOND_STEP_CAPTURE
        callback(step, denoised, current, total)
        if _QUALITY_CAPTURE_ARMED and _QUALITY_OUTPUT_DIR:
            if step == 1 and _FIRST_STEP_CAPTURE is None:
                _FIRST_STEP_CAPTURE = _capture_nested_tensor(
                    current,
                    Path(_QUALITY_OUTPUT_DIR),
                    "first_step",
                )
            elif step == 2 and _SECOND_STEP_CAPTURE is None:
                _SECOND_STEP_CAPTURE = _capture_nested_tensor(
                    current,
                    Path(_QUALITY_OUTPUT_DIR),
                    "second_step",
                )

    return capture


latent_preview.prepare_callback = _capture_prepare_callback


def _record_provider_call(function):
    def tracked(*args, **kwargs):
        with _LOCK:
            _ATTENTION_COUNTS["provider_attempts"] += 1
        try:
            result = function(*args, **kwargs)
        except Exception:
            with _LOCK:
                _ATTENTION_COUNTS["provider_exceptions"] += 1
            raise
        with _LOCK:
            _ATTENTION_COUNTS["provider_successes"] += 1
        return result

    return tracked


def _install_attention_tracking() -> None:
    try:
        attention = importlib.import_module("comfy.ldm.modules.attention")
    except (AttributeError, ImportError):
        return
    if _ATTENTION_POLICY == "sdpa":
        ops = getattr(getattr(attention, "comfy", None), "ops", None)
        function = getattr(ops, "scaled_dot_product_attention", None)
        if function is not None:
            ops.scaled_dot_product_attention = _record_provider_call(function)
    elif _ATTENTION_POLICY == "dinkster_kitchen_int8":
        kitchen = getattr(attention, "comfy_kitchen", None)
        for name in ("int8_attention", "int8_attention_from_prequantized"):
            function = getattr(kitchen, name, None)
            if function is not None:
                setattr(kitchen, name, _record_provider_call(function))
    elif _ATTENTION_POLICY == "sage":
        function = getattr(attention, "sageattn", None)
        if function is not None:
            attention.sageattn = _record_provider_call(function)
        pytorch_attention = getattr(attention, "attention_pytorch", None)
        sage_attention = getattr(attention, "attention_sage", None)
        sage_body = getattr(sage_attention, "__wrapped__", sage_attention)
        sage_code = getattr(sage_body, "__code__", None)
        if pytorch_attention is not None and sage_code is not None:

            def tracked_pytorch_attention(*args, **kwargs):
                if sys._getframe(1).f_code is sage_code:
                    with _LOCK:
                        _ATTENTION_COUNTS["fallback_calls"] += 1
                return pytorch_attention(*args, **kwargs)

            attention.attention_pytorch = tracked_pytorch_attention


_install_attention_tracking()


def _device_module():
    device = comfy.model_management.get_torch_device()
    if device.type == "xpu":
        return device, torch.xpu
    if device.type == "cuda":
        return device, torch.cuda
    return device, None


def _physical_device_uuid(device, module) -> str | None:
    if module is None:
        return None
    try:
        value = str(module.get_device_properties(device).uuid)
    except Exception:
        return None
    if device.type == "cuda" and value and not value.startswith("GPU-"):
        value = "GPU-" + value
    return value or None


def _synchronize() -> None:
    # Instrumentation must never break execution; an unsynchronized
    # timestamp is better than a failed prompt.
    try:
        device, module = _device_module()
        if module is not None:
            module.synchronize(device)
    except Exception:
        pass


def _device_used_bytes() -> int | None:
    """Device-global used bytes (total - free), which unlike allocator
    peaks also sees memory ComfyUI's dynamic VRAM loading holds outside
    the caching allocator."""
    try:
        device, module = _device_module()
        mem_get_info = getattr(module, "mem_get_info", None) if module is not None else None
        if mem_get_info is None:
            return None
        free, total = mem_get_info(device)
        return int(total) - int(free)
    except Exception:
        return None


def _track_peak_device_used() -> None:
    global _PEAK_DEVICE_USED
    used = _device_used_bytes()
    if used is not None:
        with _LOCK:
            if used > _PEAK_DEVICE_USED:
                _PEAK_DEVICE_USED = used


def _record(event: str, prompt_id: object, node_id: object, *, synchronize: bool) -> None:
    if synchronize:
        _synchronize()
    _track_peak_device_used()
    with _LOCK:
        _EVENTS.append(
            {"t": time.perf_counter(), "event": event, "prompt_id": prompt_id, "node": node_id}
        )


_original_start_progress = ProgressRegistry.start_progress
_original_finish_progress = ProgressRegistry.finish_progress
_original_update_absolute = comfy.utils.ProgressBar.update_absolute


def _recording_start_progress(self, node_id):
    _record("node_start", self.prompt_id, node_id, synchronize=True)
    return _original_start_progress(self, node_id)


def _recording_update_absolute(self, value, total=None, preview=None):
    # Progress is recorded here, not at ProgressRegistry.update_progress:
    # ProgressBar throttles its hook to a 100ms minimum interval, so the
    # registry only sees a subset of step ticks, while update_absolute
    # itself is called once per step on the execution thread.
    context = get_executing_context()
    if context is not None:
        maximum = float(total if total is not None else self.total)
        clamped = min(float(value), maximum)
        key = (context.prompt_id, context.node_id)
        with _LOCK:
            changed = _PROGRESS_SEEN.get(key) != (clamped, maximum)
            if changed:
                _PROGRESS_SEEN[key] = (clamped, maximum)
                _EVENTS.append(
                    {
                        "t": time.perf_counter(),
                        "event": "progress",
                        "prompt_id": context.prompt_id,
                        "node": context.node_id,
                        "value": clamped,
                        # One node can report several progress sequences
                        # (model weight loading and sampling both tick on
                        # the sampler node); the max lets the driver tell
                        # them apart.
                        "max": maximum,
                    }
                )
        if changed:
            _track_peak_device_used()
    return _original_update_absolute(self, value, total, preview)


def _recording_finish_progress(self, node_id):
    _record("node_finish", self.prompt_id, node_id, synchronize=True)
    return _original_finish_progress(self, node_id)


ProgressRegistry.start_progress = _recording_start_progress
ProgressRegistry.finish_progress = _recording_finish_progress
comfy.utils.ProgressBar.update_absolute = _recording_update_absolute


# Completion tracking for the free path the unload endpoint triggers.
# The prompt worker clears the queue flags BEFORE it runs
# unload_all_models(), PromptExecutor.reset(), and the trailing
# gc.collect() + soft_empty_cache(), so flag consumption alone does not
# mean the executor graph has been dropped.  These wrappers follow the
# worker through that sequence and bump a generation counter only when
# the trailing soft_empty_cache() after reset() finishes; the phase
# gating skips the soft_empty_cache() calls unload_all_models() makes
# internally before reset().
_UNLOAD_PHASE = "idle"  # idle -> armed -> consumed -> reset_done -> idle
_UNLOAD_GENERATION = 0

_original_get_flags = execution.PromptQueue.get_flags
_original_executor_reset = execution.PromptExecutor.reset
_original_soft_empty_cache = comfy.model_management.soft_empty_cache


def _arm_unload_tracking() -> int:
    global _UNLOAD_PHASE
    with _LOCK:
        _UNLOAD_PHASE = "armed"
        return _UNLOAD_GENERATION


def _tracking_get_flags(self, reset=True):
    global _UNLOAD_PHASE
    flags = _original_get_flags(self, reset)
    if reset and flags.get("free_memory"):
        with _LOCK:
            if _UNLOAD_PHASE == "armed":
                _UNLOAD_PHASE = "consumed"
    return flags


def _tracking_executor_reset(self):
    global _UNLOAD_PHASE
    result = _original_executor_reset(self)
    with _LOCK:
        if _UNLOAD_PHASE == "consumed":
            _UNLOAD_PHASE = "reset_done"
    return result


def _tracking_soft_empty_cache(*args, **kwargs):
    global _UNLOAD_PHASE, _UNLOAD_GENERATION
    result = _original_soft_empty_cache(*args, **kwargs)
    with _LOCK:
        if _UNLOAD_PHASE == "reset_done":
            _UNLOAD_PHASE = "idle"
            _UNLOAD_GENERATION += 1
    return result


execution.PromptQueue.get_flags = _tracking_get_flags
execution.PromptExecutor.reset = _tracking_executor_reset
comfy.model_management.soft_empty_cache = _tracking_soft_empty_cache


class DinksterBenchmarkSink:
    """Consumes decoded media and records finiteness instead of saving."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"images": ("IMAGE", {})},
            "optional": {
                "audio": ("AUDIO", {}),
                "conditioning": ("CONDITIONING", {}),
                "latent": ("LATENT", {}),
                "final_latent": ("LATENT", {}),
                "sigmas": ("SIGMAS", {}),
                "seed": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "observe"
    OUTPUT_NODE = True
    CATEGORY = "dinkster_benchmark"

    def observe(
        self,
        images,
        audio=None,
        conditioning=None,
        latent=None,
        final_latent=None,
        sigmas=None,
        seed=0,
    ):
        global _QUALITY_CAPTURED, _QUALITY_CAPTURE_ARMED, _QUALITY_CAPTURE_SEED
        observation = {
            "finite": bool(torch.isfinite(images).all().item()),
            "shape": list(images.shape),
        }
        if audio is not None:
            waveform = audio["waveform"]
            observation.update(
                {
                    "audio_finite": bool(torch.isfinite(waveform).all().item()),
                    "audio_shape": list(waveform.shape),
                    "audio_sample_rate": audio["sample_rate"],
                }
            )
        if _QUALITY_OUTPUT_DIR and _QUALITY_CAPTURE_ARMED and not _QUALITY_CAPTURED:
            output_dir = Path(_QUALITY_OUTPUT_DIR)
            capture = {
                "version": 1,
                "seed": _QUALITY_CAPTURE_SEED,
                "image": _capture_quality_tensor(
                    images,
                    output_dir / "capture_image.npy",
                    spatial_stride=_QUALITY_SPATIAL_STRIDE,
                ),
            }
            if audio is not None:
                capture["audio"] = _capture_quality_tensor(
                    audio["waveform"],
                    output_dir / "capture_audio.npy",
                    spatial_stride=None,
                )
                capture["audio_sample_rate"] = audio["sample_rate"]
                capture["raw_outputs"] = _capture_raw_outputs(
                    images,
                    audio["waveform"],
                    audio["sample_rate"],
                    output_dir,
                )
            if conditioning is not None:
                capture["text_context"] = _capture_quality_tensor(
                    conditioning[0][0],
                    output_dir / "text_context.npy",
                    spatial_stride=None,
                )
            if latent is not None:
                noise = importlib.import_module("comfy.sample").prepare_noise(
                    latent["samples"], seed
                )
                capture["initial_noise"] = _capture_nested_tensor(
                    noise,
                    output_dir,
                    "initial_noise",
                )
            if final_latent is not None:
                capture["final_latent"] = _capture_nested_tensor(
                    final_latent["samples"],
                    output_dir,
                    "final_latent",
                )
            if sigmas is not None:
                capture["sigmas"] = _capture_quality_tensor(
                    sigmas,
                    output_dir / "sigmas.npy",
                    spatial_stride=None,
                )
            if _FIRST_STEP_CAPTURE is not None:
                capture["first_step"] = _FIRST_STEP_CAPTURE
            if _SECOND_STEP_CAPTURE is not None:
                capture["second_step"] = _SECOND_STEP_CAPTURE
            if _H3_FORWARD_TRACE is not None:
                capture["forward_trace"] = _H3_FORWARD_TRACE
            observation["quality_capture"] = capture
            _QUALITY_CAPTURED = True
            _QUALITY_CAPTURE_ARMED = False
            _QUALITY_CAPTURE_SEED = None
        with _LOCK:
            _OBSERVATIONS.append(observation)
        return {}


NODE_CLASS_MAPPINGS = {"DinksterBenchmarkSink": DinksterBenchmarkSink}
NODE_DISPLAY_NAME_MAPPINGS = {"DinksterBenchmarkSink": "Dinkster Benchmark Sink"}


def _admission_problem(backend: str) -> str | None:
    """Admit the requested backend explicitly or refuse; never fall
    through to whatever accelerator torch happens to see."""
    if backend == "rocm":
        if getattr(torch.version, "hip", None) is None:
            return "this torch is not a HIP (ROCm) build"
        if not torch.cuda.is_available():
            return "torch cannot see a ROCm device on this machine"
        return None
    if backend == "cuda":
        if getattr(torch.version, "cuda", None) is None:
            return "this torch is not a CUDA build"
        if not torch.cuda.is_available():
            return "torch cannot see a CUDA device on this machine"
        return None
    if backend == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            return "torch cannot see an XPU device on this machine"
        return None
    return f"unknown backend {backend!r}"


def _backend_runtime(backend: str) -> str:
    if backend == "rocm":
        return f"hip {torch.version.hip}"
    if backend == "cuda":
        return f"cuda {torch.version.cuda}"
    build_xpu = str(getattr(torch.version, "xpu", "") or "").strip()
    return f"xpu {build_xpu or torch.__version__}"


def _device_entries(backend: str) -> list[dict[str, object]]:
    """Device identity entries, shaped exactly like the Dinkster reports'."""
    entries: list[dict[str, object]] = []
    if backend in ("rocm", "cuda"):
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            if backend == "rocm":
                architecture = str(getattr(properties, "gcnArchName", "") or "").strip()
            else:
                architecture = f"sm_{properties.major}{properties.minor}"
            entries.append(
                {
                    "index": index,
                    "name": properties.name,
                    "architecture": architecture,
                    "total_memory": int(properties.total_memory),
                }
            )
        return entries
    for index in range(torch.xpu.device_count()):
        properties = torch.xpu.get_device_properties(index)
        architecture = ""
        for attribute in ("architecture", "device_id", "platform_name"):
            value = getattr(properties, attribute, None)
            if value:
                architecture = f"{attribute}={value}"
                break
        entries.append(
            {
                "index": index,
                "name": properties.name,
                "architecture": architecture,
                "total_memory": int(properties.total_memory),
            }
        )
    return entries


def _windows_video_controllers() -> list[str]:
    command = (
        "Get-CimInstance Win32_VideoController | "
        "ForEach-Object { $_.Name + ' driver ' + $_.DriverVersion }"
    )
    try:
        output = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        return [line.strip() for line in output.splitlines() if line.strip()]
    except Exception:
        return []


def _driver_identity(backend: str) -> str:
    """Best available display-driver identity, as the Dinkster runner records."""
    parts: list[str] = []
    if backend == "xpu" and torch.xpu.device_count() > 0:
        properties = torch.xpu.get_device_properties(0)
        for attribute in ("driver_version", "platform_name"):
            value = str(getattr(properties, attribute, "") or "").strip()
            if value:
                parts.append(f"{attribute}={value}")
    if sys.platform == "win32":
        parts.extend(_windows_video_controllers())
    elif backend == "rocm":
        for label, candidate in (
            ("amdgpu kernel driver", "/sys/module/amdgpu/version"),
            ("rocm userspace", "/opt/rocm/.info/version"),
        ):
            try:
                text = Path(candidate).read_text().strip()
            except OSError:
                continue
            if text:
                parts.append(f"{label} {text}")
        if not parts and Path("/sys/module/amdgpu").is_dir():
            # The in-tree amdgpu module has no version file; for in-tree
            # builds the kernel release is the driver version.
            parts.append(f"amdgpu in-tree kernel driver, kernel {platform.uname().release}")
    elif backend == "cuda":
        try:
            text = Path("/proc/driver/nvidia/version").read_text().splitlines()[0].strip()
        except (OSError, IndexError):
            text = ""
        if text:
            parts.append(text)
    if parts:
        return "; ".join(parts)
    return "unknown (no driver identity source on this host)"


def _peak_rss_bytes() -> int | None:
    """The process's lifetime peak resident set size, when the platform
    exposes one."""
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak <= 0:
        return None
    # Linux reports ru_maxrss in kilobytes, macOS in bytes.
    return peak * 1024 if sys.platform.startswith("linux") else peak


def _comfyui_version() -> str:
    try:
        from comfyui_version import __version__

        return str(__version__)
    except Exception:
        return ""


def _quality_file(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 22), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _capture_quality_tensor(tensor, path: Path, *, spatial_stride: int | None):
    source_shape = tuple(int(value) for value in tensor.shape)
    if spatial_stride is not None:
        if len(source_shape) != 4:
            raise ValueError("spatial quality capture requires a rank-4 THWC tensor")
        captured_shape = (
            source_shape[0],
            (source_shape[1] + spatial_stride - 1) // spatial_stride,
            (source_shape[2] + spatial_stride - 1) // spatial_stride,
            source_shape[3],
        )
    else:
        captured_shape = source_shape
    if path.exists():
        raise FileExistsError(f"quality capture refuses to overwrite {path}")
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=captured_shape)
    try:
        for index in range(source_shape[0]):
            source = tensor[index]
            if spatial_stride is not None:
                source = source[::spatial_stride, ::spatial_stride, :]
            output[index] = source.detach().to(device="cpu", dtype=torch.float32).numpy()
        output.flush()
    finally:
        del output
    return {
        **_quality_file(path),
        "dtype": "float32",
        "source_shape": list(source_shape),
        "captured_shape": list(captured_shape),
        "spatial_stride": spatial_stride,
    }


def _capture_trace_tensor(tensor, path: Path):
    source_shape = tuple(int(value) for value in tensor.shape)
    rows = tensor.detach().reshape(-1, source_shape[-1])
    row_indices = (
        torch.linspace(
            0,
            rows.shape[0] - 1,
            min(64, rows.shape[0]),
            device=rows.device,
        )
        .round()
        .to(dtype=torch.long)
        .unique()
    )
    channel_indices = (
        torch.linspace(
            0,
            rows.shape[1] - 1,
            min(128, rows.shape[1]),
            device=rows.device,
        )
        .round()
        .to(dtype=torch.long)
        .unique()
    )
    sampled = rows.index_select(0, row_indices).index_select(1, channel_indices)
    return {
        **_capture_quality_tensor(sampled, path, spatial_stride=None),
        "activation_shape": list(source_shape),
        "row_indices": row_indices.cpu().tolist(),
        "channel_indices": channel_indices.cpu().tolist(),
    }


def _install_h3_forward_trace(model, output_dir: Path):
    trace = {}
    handles = []

    def capture(name, tensor):
        if name not in trace:
            trace[name] = _capture_trace_tensor(tensor, output_dir / f"trace_{name}.npy")

    def post(name):
        def hook(_module, _inputs, output):
            capture(name, torch.cat(output, dim=-1) if type(output) is tuple else output)

        return hook

    def pre(name):
        def hook(_module, inputs):
            capture(name, inputs[0])

        return hook

    handles.extend(
        (
            model.condition_proj.register_forward_hook(post("condition_projection")),
            model.token_refiner.register_forward_hook(post("token_refiner")),
            model.video_patch_proj.register_forward_pre_hook(pre("video_patch_input")),
            model.video_patch_proj.register_forward_hook(post("video_patch_projection")),
            model.audio_patch_proj.register_forward_pre_hook(pre("audio_patch_input")),
            model.audio_patch_proj.register_forward_hook(post("audio_patch_projection")),
            model.blocks[0].register_forward_pre_hook(pre("block0_input")),
            model.blocks[0].adaln_proj.register_forward_pre_hook(pre("block0_adaln_input")),
            model.blocks[0].adaln_proj.register_forward_hook(post("block0_adaln")),
            model.blocks[0].norm1.register_forward_hook(post("block0_norm1")),
            model.blocks[0].attn.qkv_proj.register_forward_hook(post("block0_qkv")),
            model.blocks[0].attn.out_proj.register_forward_hook(post("block0_attention")),
            model.blocks[0].norm2.register_forward_hook(post("block0_norm2")),
            model.blocks[0].mlp.fc1.register_forward_hook(post("block0_mlp_input")),
            model.blocks[0].mlp.fc2.register_forward_hook(post("block0_mlp_output")),
            model.blocks[0].register_forward_hook(post("block0_output")),
            model.final_layer.register_forward_pre_hook(pre("final_input")),
        )
    )

    def final_output(_module, _inputs, output):
        capture("video_head", output[0])
        capture("audio_head", output[1])

    handles.append(model.final_layer.register_forward_hook(final_output))
    return trace, handles


def _trace_h3_forward(self, *args, **kwargs):
    global _H3_FORWARD_TRACE
    assert _H3_FORWARD_ORIGINAL is not None
    if not _QUALITY_CAPTURE_ARMED or _H3_FORWARD_TRACE is not None or not _QUALITY_OUTPUT_DIR:
        return _H3_FORWARD_ORIGINAL(self, *args, **kwargs)
    trace, handles = _install_h3_forward_trace(self, Path(_QUALITY_OUTPUT_DIR))
    try:
        return _H3_FORWARD_ORIGINAL(self, *args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
        _H3_FORWARD_TRACE = trace


def _ensure_h3_forward_trace_patch():
    global _H3_FORWARD_ORIGINAL
    if _H3_FORWARD_ORIGINAL is not None:
        return
    model_class = importlib.import_module("comfy.ldm.minimax.model").MiniMaxH3Model
    _H3_FORWARD_ORIGINAL = model_class._forward
    model_class._forward = _trace_h3_forward


def _capture_nested_tensor(value, output_dir: Path, stem: str):
    tensors = value.unbind() if value.is_nested else (value,)
    roles = ("video", "audio") if len(tensors) == 2 else tuple(str(i) for i in range(len(tensors)))
    return {
        role: _capture_quality_tensor(
            tensor,
            output_dir / f"{stem}_{role}.npy",
            spatial_stride=None,
        )
        for role, tensor in zip(roles, tensors, strict=True)
    }


def _capture_raw_outputs(images, waveform, sample_rate: int, output_dir: Path):
    frames = {}
    for label, index in (
        ("first", 0),
        ("middle", int(images.shape[0]) // 2),
        ("last", int(images.shape[0]) - 1),
    ):
        path = output_dir / f"frame_{label}.png"
        pixels = (
            images[index]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(dtype=torch.uint8)
            .numpy()
        )
        Image.fromarray(pixels, mode="RGB").save(path)
        frames[label] = {**_quality_file(path), "frame_index": index}

    audio_path = output_dir / "audio.wav"
    audio = waveform[0].detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0)
    pcm = (audio.mul(32767.0).round().to(dtype=torch.int16).numpy().T).astype("<i2", copy=False)
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(int(pcm.shape[1]))
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    return {
        "frames": frames,
        "audio": {
            **_quality_file(audio_path),
            "channels": int(pcm.shape[1]),
            "sample_rate": sample_rate,
            "sample_width_bytes": 2,
        },
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sage_module_identity() -> dict[str, object]:
    try:
        module = importlib.import_module("sageattention")
        module_path = Path(module.__file__).resolve()
        distribution_name = module.__distribution__
        if not isinstance(distribution_name, str) or not distribution_name:
            raise ValueError("sageattention.__distribution__ is missing")
        distribution = importlib.metadata.distribution(distribution_name)
        files = distribution.files or ()
        authenticated = any(
            Path(distribution.locate_file(file)).resolve() == module_path for file in files
        )
        return {
            "module": "sageattention",
            "path": str(module_path),
            "distribution": distribution_name,
            "version": distribution.version,
            "authenticated": authenticated,
        }
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        return {
            "module": "sageattention",
            "path": "",
            "distribution": "",
            "version": "",
            "authenticated": False,
        }


def _attention_execution() -> dict[str, object]:
    with _LOCK:
        counts = dict(_ATTENTION_COUNTS)
    selected_calls = counts["provider_successes"]
    if _ATTENTION_POLICY == "sage":
        selected_calls += counts["fallback_calls"]
    return {
        "policy": _ATTENTION_POLICY,
        "selected_calls": selected_calls,
        **counts,
    }


def _attention_identity() -> dict[str, object]:
    args = comfy.model_management.args
    flags = {
        "sdpa": bool(getattr(args, "use_pytorch_cross_attention", False)),
        "dinkster_kitchen_int8": bool(getattr(args, "use_ck_attention", False)),
        "sage": bool(getattr(args, "use_sage_attention", False)),
    }
    selected = [policy for policy, enabled in flags.items() if enabled]
    selected_policy = selected[0] if len(selected) == 1 else "auto" if not selected else "invalid"
    versions: list[list[str | None]] = [["torch", str(torch.__version__)]]
    provider_module = None
    if _ATTENTION_POLICY == "dinkster_kitchen_int8":
        versions.append(["comfy-kitchen", _distribution_version("comfy-kitchen")])
    elif _ATTENTION_POLICY == "sage":
        provider_module = _sage_module_identity()
        versions.append(
            [
                provider_module["distribution"],
                provider_module["version"],
            ]
        )
    fallback = "sdpa" if _ATTENTION_POLICY == "sage" else None
    return {
        "requested_policy": _ATTENTION_POLICY,
        "selected_policy": selected_policy,
        "scope": ["process_global"],
        "provider_versions": versions,
        **({"provider_module": provider_module} if provider_module is not None else {}),
        "fallback": fallback,
        "fallback_conditions": (
            ["low_precision_attention_disabled", "unsupported_mask", "provider_exception"]
            if fallback is not None
            else []
        ),
    }


_routes = PromptServer.instance.routes


@_routes.get("/dinkster_benchmark/identity")
async def _identity(request):
    backend = request.rel_url.query.get("backend", "")
    problem = _admission_problem(backend)
    if problem is not None:
        return web.json_response({"error": problem}, status=400)
    return web.json_response(
        {
            "host": {
                "platform": platform.platform(),
                "os_version": platform.version(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "driver": _driver_identity(backend),
            "torch": {
                "version": str(torch.__version__),
                "backend_runtime": _backend_runtime(backend),
            },
            "devices": _device_entries(backend),
            "comfyui_version": _comfyui_version(),
            "attention": _attention_identity(),
            "boot_nonce": _BOOT_NONCE,
        }
    )


@_routes.get("/dinkster_benchmark/state")
async def _state(request):
    with _LOCK:
        events = list(_EVENTS)
        observations = list(_OBSERVATIONS)
    return web.json_response(
        {
            "events": events,
            "observations": observations,
            "attention_execution": _attention_execution(),
        }
    )


@_routes.post("/dinkster_benchmark/reset")
async def _reset(request):
    global _ALLOCATOR_WINDOW
    global _H3_FORWARD_TRACE
    global _PEAK_DEVICE_USED, _QUALITY_CAPTURED, _QUALITY_CAPTURE_ARMED, _QUALITY_CAPTURE_SEED
    payload = await request.json()
    nonce = payload.get("nonce")
    device_identity = payload.get("device")
    legacy = nonce is None and device_identity is None
    if not legacy and (
        not isinstance(nonce, str) or len(nonce) != 48 or not isinstance(device_identity, dict)
    ):
        return web.json_response({"error": "nonce and device identity are required"}, status=400)
    with _LOCK:
        _EVENTS.clear()
        _OBSERVATIONS.clear()
        _PROGRESS_SEEN.clear()
        _PEAK_DEVICE_USED = 0
        _QUALITY_CAPTURED = False
        _QUALITY_CAPTURE_ARMED = False
        _QUALITY_CAPTURE_SEED = None
        _H3_FORWARD_TRACE = None
        for name in _ATTENTION_COUNTS:
            _ATTENTION_COUNTS[name] = 0
        _ALLOCATOR_WINDOW = (
            None
            if legacy
            else {
                "nonce": nonce,
                "device": device_identity,
                "process_instance": _PROCESS_INSTANCE,
            }
        )
    device, module = _device_module()
    physical_device_uuid = _physical_device_uuid(device, module)
    if module is not None:
        # Synchronize first: resetting peak stats needs an initialized
        # device context.
        module.synchronize(device)
        module.reset_peak_memory_stats(device)
    # Seed the device-used peak with the post-reset baseline so it is
    # defined even for a cell that never executes a node.
    _track_peak_device_used()
    return (
        web.json_response({"ok": True})
        if legacy
        else web.json_response(
            {
                "event": "reset_ack",
                "nonce": nonce,
                "device": device_identity,
                "physical_device_uuid": physical_device_uuid,
                "process_instance": _PROCESS_INSTANCE,
            }
        )
    )


@_routes.post("/dinkster_benchmark/arm_quality")
async def _arm_quality(request):
    global _QUALITY_CAPTURE_ARMED, _QUALITY_CAPTURE_SEED
    payload = await request.json()
    seed = payload.get("seed")
    if not _QUALITY_OUTPUT_DIR:
        return web.json_response({"error": "quality capture is not configured"}, status=400)
    if type(seed) is not int:
        return web.json_response({"error": "seed (int) is required"}, status=400)
    _ensure_h3_forward_trace_patch()
    with _LOCK:
        if _QUALITY_CAPTURED or _QUALITY_CAPTURE_ARMED:
            return web.json_response(
                {"error": "quality capture is already armed or complete"}, status=409
            )
        _QUALITY_CAPTURE_ARMED = True
        _QUALITY_CAPTURE_SEED = seed
    return web.json_response({"ok": True, "seed": seed})


@_routes.get("/dinkster_benchmark/memory")
async def _memory(request):
    nonce = request.rel_url.query.get("nonce")
    job_ref = request.rel_url.query.get("job_ref")
    history = PromptServer.instance.prompt_queue.get_history(job_ref) if job_ref else {}
    history_entry = history.get(job_ref) if isinstance(history, dict) else None
    prompt = history_entry.get("prompt") if isinstance(history_entry, dict) else None
    attempt = (
        prompt[0]
        if isinstance(prompt, (list, tuple)) and len(prompt) > 1 and prompt[1] == job_ref
        else None
    )
    with _LOCK:
        window = dict(_ALLOCATOR_WINDOW) if _ALLOCATOR_WINDOW is not None else None
        prompt_events = [row for row in _EVENTS if row.get("prompt_id") == job_ref]
    legacy = nonce is None and job_ref is None
    if not legacy and (
        window is None
        or nonce != window.get("nonce")
        or not job_ref
        or not prompt_events
        or attempt is None
    ):
        return web.json_response(
            {"error": "allocator window is stale or has no matching job"}, status=409
        )
    device, module = _device_module()
    if module is None:
        return web.json_response({"error": f"no allocator telemetry for {device.type}"}, status=400)
    _track_peak_device_used()
    with _LOCK:
        peak_device_used = _PEAK_DEVICE_USED
    return web.json_response(
        {
            **(
                {}
                if legacy
                else {
                    "event": "read_ack",
                    "nonce": nonce,
                    "job_ref": job_ref,
                    "attempt": attempt,
                    "process_instance": _PROCESS_INSTANCE,
                    "device": window["device"],
                    "logical_device": str(device),
                    "physical_device_uuid": _physical_device_uuid(device, module),
                    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                }
            ),
            "peak_allocated_bytes": int(module.max_memory_allocated(device)),
            "peak_reserved_bytes": int(module.max_memory_reserved(device)),
            "allocated_bytes": int(module.memory_allocated(device)),
            "reserved_bytes": int(module.memory_reserved(device)),
            "peak_rss_bytes": _peak_rss_bytes(),
            "peak_device_used_bytes": peak_device_used if peak_device_used > 0 else None,
        }
    )


@_routes.post("/dinkster_benchmark/unload")
async def _unload(request):
    """Unload through ComfyUI's own free path, then measure the residual.

    Calling unload_all_models() directly offloads weights but leaves the
    executor caches and therefore the executed model modules alive; model
    code may retain forward activations on those modules (Anima keeps its
    last crossattn_emb, ~2 MiB - see Dinkster issue #841).  The free_memory
    prompt-queue flag (what POST /free sets) additionally makes the prompt
    worker drop the executor caches and gc, releasing the whole graph.
    """
    device, module = _device_module()
    if module is None:
        return web.json_response({"error": f"no allocator telemetry for {device.type}"}, status=400)
    payload = await request.json()
    nonce = payload.get("nonce")
    job_ref = payload.get("job_ref")
    drain_target = payload.get("drain_target_bytes")
    legacy = nonce is None and job_ref is None
    if legacy and (not isinstance(drain_target, int) or drain_target < 0):
        return web.json_response(
            {"error": "drain_target_bytes (non-negative int) is required"}, status=400
        )
    with _LOCK:
        window = dict(_ALLOCATOR_WINDOW) if _ALLOCATOR_WINDOW is not None else None
        prompt_events = [row for row in _EVENTS if row.get("prompt_id") == job_ref]
    if not legacy and (
        window is None
        or nonce != window.get("nonce")
        or not isinstance(job_ref, str)
        or not prompt_events
    ):
        return web.json_response(
            {"error": "allocator window is stale or has no matching job"}, status=409
        )
    queue = PromptServer.instance.prompt_queue
    baseline = _arm_unload_tracking()
    queue.set_flag("unload_models", True)
    queue.set_flag("free_memory", True)
    # Wait for the worker to finish the whole free path (flags consumed,
    # executor reset, trailing gc + soft_empty_cache), not merely to
    # consume the flags: the allocator can sit below the drain target
    # while the executor graph is still alive.
    deadline = time.monotonic() + 30.0
    while _UNLOAD_GENERATION == baseline and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    if _UNLOAD_GENERATION == baseline:
        return web.json_response(
            {"error": "prompt worker did not complete the free path"}, status=500
        )
    if legacy:
        assert isinstance(drain_target, int)
        residual_allocated = int(module.memory_allocated(device))
        while residual_allocated > drain_target and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            residual_allocated = int(module.memory_allocated(device))
        gc.collect()
        if device.type == "cuda":
            clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear_workspaces is not None:
                clear_workspaces()
        module.empty_cache()
        module.synchronize(device)
        residual_allocated = int(module.memory_allocated(device))
        return web.json_response({"residual_allocated_bytes": residual_allocated})
    module.synchronize(device)
    residual_allocated = int(module.memory_allocated(device))
    residual_reserved = int(module.memory_reserved(device))
    return web.json_response(
        {
            "event": "unload_ack",
            "nonce": nonce,
            "job_ref": job_ref,
            "process_instance": _PROCESS_INSTANCE,
            "device": window["device"],
            "residual_allocated_bytes": residual_allocated,
            "residual_reserved_bytes": residual_reserved,
        }
    )
