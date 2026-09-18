"""Run one model-family validation cell on a ROCm or XPU device.

One invocation executes one cell - a model family in one execution mode -
end to end on real hardware and writes the JSON evidence report that
dinkster_workers.backend_env.validate_family_report checks:

    .venv-rocm/bin/python scripts/family_validation.py \\
        --backend rocm --family sd15 --checkpoint /models/sd15.safetensors \\
        --json sd15-eager.json

Families: sd15, sdxl, flux (safetensors checkpoints), gguf (a GGUF
diffusion model with split text/VAE sources), and lora (an SD-era
checkpoint plus a LoRA whose runtime patch application, effect, and exact
restoration are checked against the unpatched baseline). Eager execution
is the support contract; --mode compile compiles the diffusion module in
place and compares an unguided (cfg 1.0) compiled run against the same
invocation's unguided eager run on mean image difference - guidance
multiplies benign kernel noise chaotically, so guided comparisons cannot
distinguish faithful kernels from broken ones. Compile mode requires
--eager-report pointing at a passing eager report for the same cell -
compile claims exist only relative to proven eager behavior.

Checks per run: load (family detection and device placement), text
encoding, sampling, latent decode, finite output, second-run reuse (the
rerun noise floor for the LoRA checks), and unload (allocator drains after
the modules leave the device). Peak and residual device memory are
recorded from the backend allocator. Exits nonzero when any check fails or
the report is incomplete.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import platform
import subprocess
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from dinkster_workers.backend_env import (
    FAMILY_REPORT_VERSION,
    FAMILY_RESIDUAL_CEILING_BYTES,
    FAMILY_VALIDATION_FAMILIES,
    FAMILY_VALIDATION_FAMILY_IDS,
    validate_family_report,
)


@dataclasses.dataclass(frozen=True)
class BackendAccess:
    """The per-backend torch device APIs this runner touches."""

    device: torch.device
    backend_runtime: str
    synchronize: Callable[[], None]
    reset_peak: Callable[[], None]
    peak_allocated: Callable[[], int]
    allocated: Callable[[], int]
    empty_cache: Callable[[], None]


def _admit_backend(backend: str) -> BackendAccess:
    """Admit the requested backend explicitly or exit; never fall through
    to whatever accelerator torch happens to see."""
    if backend == "rocm":
        hip = getattr(torch.version, "hip", None)
        if hip is None:
            sys.exit("error: this torch is not a HIP (ROCm) build")
        if not torch.cuda.is_available():
            sys.exit("error: torch cannot see a ROCm device on this machine")
        device = torch.device("cuda:0")
        return BackendAccess(
            device=device,
            backend_runtime=f"hip {hip}",
            synchronize=lambda: torch.cuda.synchronize(device),
            reset_peak=lambda: torch.cuda.reset_peak_memory_stats(device),
            peak_allocated=lambda: int(torch.cuda.max_memory_allocated(device)),
            allocated=lambda: int(torch.cuda.memory_allocated(device)),
            empty_cache=torch.cuda.empty_cache,
        )
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available():
        sys.exit("error: torch cannot see an XPU device on this machine")
    build_xpu = str(getattr(torch.version, "xpu", "") or "").strip()
    device = torch.device("xpu:0")
    return BackendAccess(
        device=device,
        backend_runtime=f"xpu {build_xpu or torch.__version__}",
        synchronize=lambda: torch.xpu.synchronize(device),
        reset_peak=lambda: torch.xpu.reset_peak_memory_stats(device),
        peak_allocated=lambda: int(torch.xpu.max_memory_allocated(device)),
        allocated=lambda: int(torch.xpu.memory_allocated(device)),
        empty_cache=torch.xpu.empty_cache,
    )


def _device_entries(backend: str) -> list[dict[str, object]]:
    """Device identity entries, shaped exactly like the smoke reports'."""
    entries: list[dict[str, object]] = []
    if backend == "rocm":
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            architecture = str(getattr(properties, "gcnArchName", "") or "").strip()
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
    """Best available display-driver identity, as the smoke scripts record."""
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
    if parts:
        return "; ".join(parts)
    return "unknown (no driver identity source on this host)"


def _artifact_entry(role: str, path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 22), b""):
            digest.update(chunk)
    return {
        "role": role,
        "path": str(path),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise RuntimeError(f"output shapes differ: {tuple(left.shape)} vs {tuple(right.shape)}")
    return float((left.float() - right.float()).abs().max())


def _module_dtype(module: torch.nn.Module) -> str:
    parameter = next(iter(module.parameters()), None)
    if parameter is None:
        return "no-parameters"
    return str(parameter.dtype).removeprefix("torch.")


class CellRun:
    """One family-validation cell's mutable execution state."""

    def __init__(self, arguments: argparse.Namespace, access: BackendAccess) -> None:
        self.arguments = arguments
        self.access = access
        self.checks: dict[str, dict[str, object]] = {}
        self.failed = False
        self.runtime: Any = None
        self.family_id = "load-failed"
        self.cond: Any = None
        self.uncond: Any = None
        self.sampled: torch.Tensor | None = None
        self.first_image: torch.Tensor | None = None
        self.rerun_diff = 0.0
        self.residual_allocated = 0
        self.lora_backups: list[tuple[str, Any, dict[str, Any]]] = []

    def record(self, name: str, action: Callable[[], str], *, always: bool = False) -> None:
        if self.failed and not always:
            self.checks[name] = {"ok": False, "detail": "not reached: an earlier check failed"}
            print(f"  {name:<18} NOT REACHED")
            return
        try:
            detail = action() or "ok"
            ok = True
        except Exception as error:  # the failure text is the check's result
            # The compact detail below keeps only the first and last lines;
            # for nested errors (torch inductor subprocesses, worker pools)
            # the root cause sits mid-traceback, so preserve it in the log.
            traceback.print_exc()
            lines = [line.strip() for line in str(error).splitlines() if line.strip()]
            detail = lines[0] if lines else type(error).__name__
            # Nested errors (torch inductor subprocesses, worker pools) put
            # the root cause on the last line, not the first.
            if len(lines) > 1:
                detail = f"{detail} [last line: {lines[-1]}]"
            detail = detail[:400]
            ok = False
        self.checks[name] = {"ok": ok, "detail": detail}
        if not ok:
            self.failed = True
        print(f"  {name:<18} {'ok (' + detail + ')' if ok else 'NO (' + detail + ')'}")

    # ---------------------------------------------------------------- checks

    def load(self) -> str:
        from dinkster_inference import load_gguf_weight_source, load_safetensors_header
        from dinkster_inference_torch import load_runtime

        arguments = self.arguments
        sources: dict[str, object] = {}
        if arguments.family == "gguf":
            sources["diffusion"] = load_gguf_weight_source(arguments.diffusion_gguf)
            sources["clip_l"] = load_safetensors_header(arguments.clip_l)
            if arguments.clip_g is not None:
                sources["clip_g"] = load_safetensors_header(arguments.clip_g)
            sources["vae"] = load_safetensors_header(arguments.vae)
        else:
            sources["checkpoint"] = load_safetensors_header(arguments.checkpoint)
        self.runtime = load_runtime(**sources)  # family-default dtypes
        family_id = self.runtime.assembled.family.id
        if family_id not in FAMILY_VALIDATION_FAMILY_IDS[arguments.family]:
            raise RuntimeError(
                f"detected family {family_id!r} is not a {arguments.family} cell member"
            )
        self.family_id = family_id
        placed: list[str] = []
        for field in dataclasses.fields(self.runtime.assembled):
            value = getattr(self.runtime.assembled, field.name)
            if isinstance(value, torch.nn.Module):
                value.to(self.access.device)
                placed.append(f"{field.name}={_module_dtype(value)}")
        self.access.synchronize()
        return f"{family_id}; " + ", ".join(placed)

    def encode_text(self) -> str:
        with torch.inference_mode():
            self.cond = self.runtime.encode_text(self.arguments.prompt)
            self.uncond = self.runtime.encode_text(self.arguments.negative_prompt)
        self.access.synchronize()
        return "cond and uncond encoded"

    def _sample(self, cfg: float | None = None) -> torch.Tensor:
        from dinkster_inference import SamplingGuidance

        arguments = self.arguments
        cfg_scale = arguments.cfg if cfg is None else cfg
        latent_space = self.runtime.assembled.family.single_stream_latent()
        latent = torch.zeros(
            (
                1,
                latent_space.channels,
                arguments.height // latent_space.spatial_downscale,
                arguments.width // latent_space.spatial_downscale,
            ),
            dtype=torch.float32,
            device=self.access.device,
        )
        compute_dtype = torch.bfloat16 if arguments.family == "flux" else torch.float16
        extra: dict[str, object] = {}
        if arguments.family == "flux":
            extra["guidance"] = arguments.guidance
        with torch.inference_mode():
            sampled = self.runtime.sample(
                latent,
                cond=self.cond,
                cfg=SamplingGuidance(self.uncond, cfg_scale),
                sampler_id=arguments.sampler,
                scheduler_id=arguments.scheduler,
                steps=arguments.steps,
                denoise=1.0,
                seed=arguments.seed,
                compute_dtype=compute_dtype,
                device=self.access.device,
                **extra,
            )
        self.access.synchronize()
        return sampled

    def _decode(self, sampled: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            image = self.runtime.decode_latent(sampled).permute(0, 2, 3, 1).clamp(0, 1)
        self.access.synchronize()
        return image

    def _run_image(self, cfg: float | None = None) -> torch.Tensor:
        return self._decode(self._sample(cfg=cfg))

    def sample(self) -> str:
        self.sampled = self._sample()
        return f"latent shape {tuple(self.sampled.shape)}"

    def decode(self) -> str:
        assert self.sampled is not None
        self.first_image = self._decode(self.sampled)
        self.sampled = None
        return f"image shape {tuple(self.first_image.shape)}"

    def finite_output(self) -> str:
        assert self.first_image is not None
        if not bool(torch.isfinite(self.first_image).all()):
            raise RuntimeError("decoded image contains non-finite values")
        return "all values finite"

    def second_run_reuse(self) -> str:
        assert self.first_image is not None
        second = self._run_image()
        if not bool(torch.isfinite(second).all()):
            raise RuntimeError("second-run image contains non-finite values")
        self.rerun_diff = _max_abs_diff(self.first_image, second)
        return f"max_abs_diff={self.rerun_diff:.3e} vs first run"

    def lora_apply(self) -> str:
        from dinkster_inference import (
            PatchTarget,
            clip_lora_key_map,
            decode_lora,
            load_safetensors_header,
            native_unet_key_map,
        )
        from dinkster_inference_torch import (
            ModuleStateStore,
            build_patch_set,
            load_tensors,
            lora_compute_dtype,
            patch_weights,
        )

        arguments = self.arguments
        assembled = self.runtime.assembled
        header = load_safetensors_header(arguments.lora)
        geometries = {key: header.entry(key).geometry for key in header.keys()}
        diffusion_keys = tuple(f"diffusion_model.{key}" for key in assembled.diffusion.state_dict())
        key_map: dict[str, object] = dict(native_unet_key_map(diffusion_keys))
        clip_keys: list[str] = []
        for component in ("clip_l", "clip_g"):
            module = getattr(assembled, component, None)
            if module is None:
                continue
            clip_keys.extend(f"{component}.transformer.{key}" for key in module.state_dict())
        key_map.update(clip_lora_key_map(clip_keys))
        decoded = decode_lora(geometries, key_map)
        if not decoded.patches:
            raise RuntimeError("LoRA decoded to zero patches matching this model")
        routes = ("diffusion", "clip_l", "clip_g")
        per_component: dict[str, dict[Any, object]] = {}
        for target, patch in decoded.patches.items():
            for component in routes:
                prefix = (
                    "diffusion_model." if component == "diffusion" else f"{component}.transformer."
                )
                if (
                    target.key.startswith(prefix)
                    and getattr(assembled, component, None) is not None
                ):
                    routed = PatchTarget(target.key.removeprefix(prefix), offset=target.offset)
                    per_component.setdefault(component, {})[routed] = patch
                    break
            else:
                raise RuntimeError(f"decoded LoRA target {target.key!r} has no component route")
        tensors = load_tensors(arguments.lora)
        patch_dtype = lora_compute_dtype(self.access.device)
        counts: list[str] = []
        for component in sorted(per_component):
            strength = (
                arguments.lora_strength_model
                if component == "diffusion"
                else arguments.lora_strength_clip
            )
            patch_set = build_patch_set(per_component[component], tensors, strength=strength)
            store = ModuleStateStore(getattr(assembled, component))
            backup = patch_weights(
                store,
                patch_set,
                weight_dtype=patch_dtype,
                backup_device=torch.device("cpu"),
            )
            self.lora_backups.append((component, store, backup))
            counts.append(f"{component}={len(backup)}")
        with torch.inference_mode():
            self.cond = self.runtime.encode_text(arguments.prompt)
            self.uncond = self.runtime.encode_text(arguments.negative_prompt)
        unmatched = f", unmatched_keys={len(decoded.unmatched)}" if decoded.unmatched else ""
        return f"patched weights {', '.join(counts)}{unmatched}"

    def lora_effect(self) -> str:
        assert self.first_image is not None
        patched = self._run_image()
        if not bool(torch.isfinite(patched).all()):
            raise RuntimeError("patched image contains non-finite values")
        effect = _max_abs_diff(self.first_image, patched)
        if effect <= self.rerun_diff:
            raise RuntimeError(
                f"LoRA changed the output by {effect:.3e}, within the"
                f" rerun noise floor {self.rerun_diff:.3e}"
            )
        return f"max_abs_diff={effect:.3e} beyond rerun floor {self.rerun_diff:.3e}"

    def lora_restore(self) -> str:
        from dinkster_inference_torch import restore_weights

        assert self.first_image is not None
        for _, store, backup in self.lora_backups:
            restore_weights(store, backup)
        self.lora_backups.clear()
        with torch.inference_mode():
            self.cond = self.runtime.encode_text(self.arguments.prompt)
            self.uncond = self.runtime.encode_text(self.arguments.negative_prompt)
        restored = self._run_image()
        if not bool(torch.isfinite(restored).all()):
            raise RuntimeError("restored image contains non-finite values")
        diff = _max_abs_diff(self.first_image, restored)
        if diff > self.rerun_diff:
            raise RuntimeError(
                f"restored output differs from the baseline by {diff:.3e},"
                f" beyond the rerun noise floor {self.rerun_diff:.3e}"
            )
        return f"max_abs_diff={diff:.3e} within rerun floor {self.rerun_diff:.3e}"

    def compile_parity(self) -> str:
        # Guidance amplifies benign eager-vs-compiled kernel noise by
        # roughly (1 + cfg) * sigma per euler step, so a guided low-step
        # comparison diverges chaotically even when every kernel is
        # correct (measured on CUDA fp16: single-forward max diff 5.9e-3
        # became image mean diff 2.4e-2 at cfg 7). The parity comparison
        # therefore runs unguided (cfg 1.0) and gates on the mean image
        # difference, which sits near 1e-3 for faithful kernels and
        # orders of magnitude higher for broken ones.
        eager = self._run_image(cfg=1.0)
        if not bool(torch.isfinite(eager).all()):
            raise RuntimeError("eager comparison image contains non-finite values")
        self.runtime.assembled.diffusion.compile()
        compiled = self._run_image(cfg=1.0)
        if not bool(torch.isfinite(compiled).all()):
            raise RuntimeError("compiled image contains non-finite values")
        diff = float((eager.float() - compiled.float()).abs().mean())
        peak = _max_abs_diff(eager, compiled)
        if diff > self.arguments.compile_tolerance:
            raise RuntimeError(
                f"compiled output differs from eager by mean {diff:.3e}"
                f" (max {peak:.3e}), beyond tolerance"
                f" {self.arguments.compile_tolerance:.3e}"
            )
        return (
            f"mean_abs_diff={diff:.3e} (max {peak:.3e}) within tolerance"
            f" {self.arguments.compile_tolerance:.3e}"
        )

    def unload(self) -> str:
        if self.lora_backups:
            # A failure between lora_apply and lora_restore leaves patched
            # weights behind; revert them so unload measures clean modules.
            from dinkster_inference_torch import restore_weights

            for _, store, backup in self.lora_backups:
                restore_weights(store, backup)
            self.lora_backups.clear()
        if self.runtime is not None:
            for field in dataclasses.fields(self.runtime.assembled):
                value = getattr(self.runtime.assembled, field.name)
                if isinstance(value, torch.nn.Module):
                    value.to(torch.device("cpu"))
        self.runtime = None
        self.cond = None
        self.uncond = None
        self.sampled = None
        self.first_image = None
        if self.arguments.mode == "compile":
            # torch.compile caches (inductor constants, cudagraph pools)
            # hold device tensors beyond the module references.
            import torch._dynamo as torch_dynamo

            torch_dynamo.reset()
        # The cuBLAS/hipBLAS workspace is allocated through the caching
        # allocator (32 MiB observed on torch 2.13 cu130) and counts as
        # allocated bytes without being a leak; release it so the residual
        # ceiling measures only tensors the cell failed to free.
        if self.access.device.type == "cuda":
            clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear_workspaces is not None:
                clear_workspaces()
        gc.collect()
        self.access.empty_cache()
        self.access.synchronize()
        residual = self.access.allocated()
        self.residual_allocated = residual
        if residual > FAMILY_RESIDUAL_CEILING_BYTES:
            raise RuntimeError(f"allocator still holds {residual} B after unload")
        return f"residual_allocated={residual} B"


def _eager_reference(arguments: argparse.Namespace) -> dict[str, object] | None:
    """Validate the referenced eager report and return the compile-mode
    reference to it, or exit when the gate is not met."""
    if arguments.mode == "eager":
        if arguments.eager_report is not None:
            sys.exit("error: --eager-report is only meaningful with --mode compile")
        return None
    if arguments.eager_report is None:
        sys.exit("error: --mode compile requires --eager-report from a passing eager run")
    payload = arguments.eager_report.read_bytes()
    report = json.loads(payload)
    if isinstance(report, dict) and report.get("all_ok") is not True:
        # A failed run's report is often structurally incomplete (a load
        # failure leaves family_id null, for example); naming the real gate
        # first beats surfacing those downstream symptoms.
        sys.exit(
            "error: the referenced eager run did not pass (all_ok is not"
            " true); compile mode requires a passing eager report"
        )
    problems = validate_family_report(report, accelerator=arguments.backend)
    for problem in problems:
        print(f"error: eager report: {problem}", file=sys.stderr)
    if problems:
        sys.exit(1)
    if report.get("mode") != "eager":
        sys.exit("error: --eager-report is not an eager-mode report")
    if report.get("family") != arguments.family:
        sys.exit(
            f"error: --eager-report is for family {report.get('family')!r},"
            f" not {arguments.family!r}"
        )
    return {
        "digest": hashlib.sha256(payload).hexdigest(),
        "all_ok": True,
        "family": arguments.family,
        "accelerator": arguments.backend,
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("rocm", "xpu"), required=True)
    parser.add_argument("--family", choices=FAMILY_VALIDATION_FAMILIES, required=True)
    parser.add_argument("--mode", choices=("eager", "compile"), default="eager")
    parser.add_argument("--checkpoint", type=Path, help="safetensors checkpoint")
    parser.add_argument("--diffusion-gguf", type=Path, help="GGUF diffusion model (gguf family)")
    parser.add_argument("--clip-l", type=Path, help="CLIP-L safetensors (gguf family)")
    parser.add_argument("--clip-g", type=Path, help="CLIP-G safetensors (gguf family, SDXL)")
    parser.add_argument("--vae", type=Path, help="VAE safetensors (gguf family)")
    parser.add_argument("--lora", type=Path, help="LoRA safetensors (lora family)")
    parser.add_argument("--lora-strength-model", type=float, default=1.0)
    parser.add_argument("--lora-strength-clip", type=float, default=1.0)
    parser.add_argument("--prompt", default="a photograph of an astronaut riding a horse")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=591)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--guidance", type=float, default=3.5, help="flux distilled guidance")
    parser.add_argument("--sampler", default="dinkster.euler")
    parser.add_argument("--scheduler", default="dinkster.simple")
    parser.add_argument("--compile-tolerance", type=float, default=5e-3)
    parser.add_argument("--eager-report", type=Path, help="passing eager report (compile mode)")
    parser.add_argument("--json", type=Path, default=None, help="write the JSON report here")
    arguments = parser.parse_args()
    if arguments.cfg is None:
        arguments.cfg = 1.0 if arguments.family == "flux" else 7.0
    for name in ("cfg", "guidance", "lora_strength_model", "lora_strength_clip"):
        if not math.isfinite(getattr(arguments, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if not math.isfinite(arguments.compile_tolerance) or arguments.compile_tolerance <= 0:
        parser.error("--compile-tolerance must be a finite positive number")
    if arguments.family == "gguf":
        missing = [
            name
            for name, value in (
                ("--diffusion-gguf", arguments.diffusion_gguf),
                ("--clip-l", arguments.clip_l),
                ("--vae", arguments.vae),
            )
            if value is None
        ]
        if missing:
            parser.error(f"gguf family requires {', '.join(missing)}")
    elif arguments.checkpoint is None:
        parser.error(f"{arguments.family} family requires --checkpoint")
    if arguments.family == "lora":
        if arguments.lora is None:
            parser.error("lora family requires --lora")
        if arguments.mode == "compile":
            parser.error(
                "the lora cell runs eager only; compile evidence for it"
                " waits on the plain cells' hardware runs"
            )
    return arguments


def main() -> int:
    arguments = _parse_arguments()
    access = _admit_backend(arguments.backend)
    eager_reference = _eager_reference(arguments)

    print(f"platform: {platform.platform()} ({platform.machine()})")
    print(f"python:   {platform.python_version()}")
    print(f"torch:    {torch.__version__} ({access.backend_runtime})")
    driver = _driver_identity(arguments.backend)
    print(f"driver:   {driver}")
    print(f"cell:     {arguments.family} / {arguments.mode} on {arguments.backend}")

    artifacts: list[dict[str, object]] = []
    if arguments.family == "gguf":
        artifacts.append(_artifact_entry("diffusion_gguf", arguments.diffusion_gguf))
        artifacts.append(_artifact_entry("clip_l", arguments.clip_l))
        if arguments.clip_g is not None:
            artifacts.append(_artifact_entry("clip_g", arguments.clip_g))
        artifacts.append(_artifact_entry("vae", arguments.vae))
    else:
        artifacts.append(_artifact_entry("checkpoint", arguments.checkpoint))
    if arguments.family == "lora":
        artifacts.append(_artifact_entry("lora", arguments.lora))

    run = CellRun(arguments, access)
    print("checks:")
    run.record("load", run.load)
    run.record("encode_text", run.encode_text)
    access.reset_peak()
    run.record("sample", run.sample)
    run.record("decode", run.decode)
    run.record("finite_output", run.finite_output)
    run.record("second_run_reuse", run.second_run_reuse)
    if arguments.family == "lora":
        run.record("lora_apply", run.lora_apply)
        run.record("lora_effect", run.lora_effect)
        run.record("lora_restore", run.lora_restore)
    if arguments.mode == "compile":
        run.record("compile_parity", run.compile_parity)
    peak_allocated = access.peak_allocated()
    run.record("unload", run.unload, always=True)

    all_ok = all(bool(entry["ok"]) for entry in run.checks.values())
    report: dict[str, object] = {
        "report_version": FAMILY_REPORT_VERSION,
        "accelerator": arguments.backend,
        "host": {
            "platform": platform.platform(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "driver": driver,
        "torch": {"version": str(torch.__version__), "backend_runtime": access.backend_runtime},
        "devices": _device_entries(arguments.backend),
        "family": arguments.family,
        "mode": arguments.mode,
        "family_id": run.family_id,
        "workload": {
            "prompt": arguments.prompt,
            "negative_prompt": arguments.negative_prompt,
            "sampler_id": arguments.sampler,
            "scheduler_id": arguments.scheduler,
            "seed": arguments.seed,
            "steps": arguments.steps,
            "width": arguments.width,
            "height": arguments.height,
            "cfg": arguments.cfg,
            "guidance": arguments.guidance if arguments.family == "flux" else None,
            "lora_strength_model": arguments.lora_strength_model
            if arguments.family == "lora"
            else None,
            "lora_strength_clip": arguments.lora_strength_clip
            if arguments.family == "lora"
            else None,
            "compile_tolerance": arguments.compile_tolerance
            if arguments.mode == "compile"
            else None,
        },
        "artifacts": artifacts,
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "residual_allocated_bytes": run.residual_allocated,
        },
        "checks": run.checks,
        "all_ok": all_ok,
    }
    if eager_reference is not None:
        report["eager_report"] = eager_reference

    problems = validate_family_report(report, accelerator=arguments.backend)
    for problem in problems:
        print(f"error: incomplete report: {problem}", file=sys.stderr)
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"report written: {arguments.json}")

    if not all_ok:
        print("error: a validation check failed", file=sys.stderr)
        return 1
    if problems:
        return 1
    print(f"family validation cell {arguments.family}/{arguments.mode} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
