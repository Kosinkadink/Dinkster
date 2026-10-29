"""Validate official SeedVR2 components through the production native execution path."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for source_root in sorted((REPO_ROOT / "packages").glob("*/src")):
    sys.path.insert(0, str(source_root))
sys.path.insert(0, str(REPO_ROOT / "src"))

OFFICIAL_SOURCE = "https://huggingface.co/Comfy-Org/SeedVR2"
OFFICIAL_REVISION = "10f035adc869a5b3ffc466360b869641511c0610"
OFFICIAL_3B_SHA256 = "98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660"
OFFICIAL_3B_INT8_SHA256 = "c3dec8bcc5916843a8a858572970597462e1f2dc598d6dfd818f6cd40f53a157"
OFFICIAL_VAE_SHA256 = "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"
COMFYUI_COMMIT = "8a33128f2f8c5585c57486c07de481241e70a39c"


class FixedResolver:
    def __init__(self, path: Path, digest: str) -> None:
        from dinkster_assets.integrity import verification_record
        from dinkster_assets.model import AssetResolution

        self._path = path
        self._digest = digest
        self._resolution = AssetResolution(path, verification_record(digest, path.stat()))

    def resolve(self, digest: str) -> Path | None:
        return self._path if digest == self._digest else None

    def resolve_asset(self, digest: str) -> object | None:
        return self._resolution if digest == self._digest else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _persist(path: Path, evidence: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _artifact(path: Path, expected_sha256: str) -> tuple[object, dict[str, object]]:
    from dinkster_assets import AssetRef, digest_file

    path = path.resolve(strict=True)
    size = path.stat().st_size
    sha256 = _sha256_file(path)
    if sha256 != expected_sha256:
        raise RuntimeError(
            f"{path.name} SHA256 differs: expected {expected_sha256}, found {sha256}"
        )
    digest = digest_file(path)
    return (
        AssetRef(
            digest=digest,
            name=path.name,
            size=size,
            media_type="application/x-safetensors",
            virtual_path=f"models/{path.name}",
            resolver=FixedResolver(path, digest),
        ),
        {
            "path": str(path),
            "bytes": size,
            "sha256": sha256,
            "blake3": digest,
        },
    )


def _identity(asset: object, role: str) -> tuple[str, dict[str, object]]:
    from dinkster_inference import (
        BFLOAT16,
        FLOAT16,
        load_safetensors_header,
        plan_seedvr2_split_component,
        seedvr2_component_runtime_identity,
    )

    path = asset.resolver.resolve(asset.digest)
    source = load_safetensors_header(
        path,
        asset_digest=asset.digest,
        asset_size=asset.size,
    )
    plan = plan_seedvr2_split_component(source, role=role, path=path)
    dtype = BFLOAT16 if role == "diffusion" else FLOAT16
    identity = seedvr2_component_runtime_identity(plan, role, dtype)
    return identity, {
        "role": role,
        "component": plan.component,
        "identity": identity,
        "compute_dtype": dtype.name,
        "identity_facts": list(plan.identity_facts),
        "runtime_facts": [list(fact) for fact in plan.runtime_facts],
    }


def _load_components(
    diffusion_asset: object,
    vae_asset: object,
    diffusion_identity: str,
    vae_identity: str,
) -> tuple[object, object]:
    from dinkster_compat_comfy import native_arm as arm
    from dinkster_workers import ExecutionContext
    from dinkster_workers.execution import use_execution_context

    with use_execution_context(
        ExecutionContext(
            "native",
            diffusion_identity,
            diffusion_dtype="bfloat16",
            text_dtype="unloaded",
            vae_dtype="unloaded",
        )
    ):
        model = arm.NativeLoadDiffusionModel.execute(
            diffusion_model=diffusion_asset,
            weight_dtype="default",
        )["model"]
    with use_execution_context(
        ExecutionContext(
            "native",
            vae_identity,
            diffusion_dtype="unloaded",
            text_dtype="unloaded",
            vae_dtype="float16",
        )
    ):
        vae = arm.NativeLoadVae.execute(vae=vae_asset)["vae"]
    return model, vae


def _offline_launch(
    diffusion_asset: object,
    vae_asset: object,
    diffusion_identity: str,
    vae_identity: str,
) -> dict[str, object]:
    import torch
    from dinkster_compat_comfy import native_arm as arm

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the offline launch check")
    calls: list[dict[str, object]] = []
    model_sentinel = object()
    vae_sentinel = object()

    def build_model(
        asset: object,
        expected_identity: str,
        _torch: object,
        **kwargs: object,
    ) -> object:
        calls.append(
            {
                "boundary": "_build_seedvr2_model_handle",
                "asset_digest": asset.digest,
                "identity": expected_identity,
                "kwargs": kwargs,
            }
        )
        return model_sentinel

    def build_component(
        asset: object,
        role: str,
        expected_identity: str,
        _torch: object,
        **kwargs: object,
    ) -> object:
        calls.append(
            {
                "boundary": "_build_seedvr2_component_handle",
                "asset_digest": asset.digest,
                "role": role,
                "identity": expected_identity,
                "kwargs": kwargs,
            }
        )
        return vae_sentinel

    original_model_builder = arm._build_seedvr2_model_handle
    original_component_builder = arm._build_seedvr2_component_handle
    arm._build_seedvr2_model_handle = build_model
    arm._build_seedvr2_component_handle = build_component
    try:
        model, vae = _load_components(
            diffusion_asset,
            vae_asset,
            diffusion_identity,
            vae_identity,
        )
    finally:
        arm._build_seedvr2_model_handle = original_model_builder
        arm._build_seedvr2_component_handle = original_component_builder
    if model is not model_sentinel or vae is not vae_sentinel:
        raise RuntimeError("offline launch did not stop at both SeedVR2 construction boundaries")
    expected = (
        ("_build_seedvr2_model_handle", diffusion_identity),
        ("_build_seedvr2_component_handle", vae_identity),
    )
    actual = tuple((call["boundary"], call["identity"]) for call in calls)
    if actual != expected:
        raise RuntimeError(f"unexpected offline launch boundaries: {actual!r}")
    if torch.cuda.is_initialized():
        raise RuntimeError("offline launch initialized CUDA")
    return {
        "status": "passed",
        "cuda_initialized": False,
        "boundaries": calls,
    }


@contextmanager
def _gpu1_claim(thread_id: str, purpose: str) -> Iterator[Path]:
    if os.name != "posix":
        raise RuntimeError("GPU claim locking requires POSIX flock")
    import fcntl

    lock_path = Path.home() / "gpu-claims" / "gpu1.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        lock.seek(0)
        lock.truncate()
        lock.write(
            "thread="
            + thread_id
            + " purpose="
            + purpose
            + " timestamp="
            + datetime.now(UTC).isoformat()
            + "\n"
        )
        lock.flush()
        os.fsync(lock.fileno())
        yield lock_path


def _tensor_record(tensor: Any) -> dict[str, object]:
    value = tensor.detach().float().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "sha256_float32": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "mean": float(value.mean()),
    }


def _run_case(model: object, vae: object, frames: int) -> dict[str, object]:
    import torch
    from dinkster_compat_comfy import native_arm as arm

    source = torch.linspace(0.0, 1.0, frames * 16 * 16 * 3, dtype=torch.float32).reshape(
        frames, 16, 16, 3
    )
    temporal_size = 4096 if frames == 1 else 64
    preprocessed = arm.GenerationSeedVR2Preprocess.execute(resized_images=source)["images"]
    latent = arm.GenerationVAEEncodeTiled.execute(
        pixels=preprocessed,
        vae=vae,
        tile_size=512,
        overlap=128,
        temporal_size=temporal_size,
        temporal_overlap=8,
    )["latent"]
    conditioning = arm.GenerationSeedVR2Conditioning.execute(
        model=model,
        vae_conditioning=latent,
    )
    noise = arm.GenerationRandomNoise.execute(noise_seed=1065)["noise"]
    sampler = arm.GenerationKSamplerSelect.execute(sampler_name="euler")["sampler"]
    sigmas = arm.GenerationBasicScheduler.execute(
        model=model,
        scheduler="simple",
        steps=1,
        denoise=1.0,
    )["sigmas"]
    guider = arm.GenerationCFGGuider.execute(
        model=model,
        positive=conditioning["positive"],
        negative=conditioning["negative"],
        cfg=1.0,
    )["guider"]
    sampled = arm.GenerationSamplerCustomAdvanced.execute(
        noise=noise,
        guider=guider,
        sampler=sampler,
        sigmas=sigmas,
        latent_image=latent,
    )["output"]
    decoded = arm.GenerationVAEDecodeTiled.execute(
        samples=sampled,
        vae=vae,
        tile_size=512,
        overlap=128,
        temporal_size=temporal_size,
        temporal_overlap=8,
    )["image"]
    output = arm.GenerationSeedVR2PostProcessing.execute(
        images=decoded,
        original_resized_images=source,
        color_correction_method="none",
    )["images"]
    return {
        "frames": frames,
        "preprocessed": _tensor_record(preprocessed),
        "encoded": _tensor_record(latent["samples"]),
        "sampled": _tensor_record(sampled["samples"]),
        "output": _tensor_record(output),
    }


def _gpu_launch(
    diffusion_asset: object,
    vae_asset: object,
    diffusion_identity: str,
    vae_identity: str,
    thread_id: str,
) -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("GPU validation requires CUDA_VISIBLE_DEVICES=1")
    with _gpu1_claim(thread_id, "SeedVR2-official-validation") as lock_path:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("GPU validation requires exactly physical GPU 1 to be visible")
        torch.manual_seed(1065)
        torch.cuda.manual_seed_all(1065)
        torch.backends.cudnn.benchmark = False
        properties = torch.cuda.get_device_properties(0)
        torch.cuda.reset_peak_memory_stats(0)
        model = None
        vae = None
        try:
            model, vae = _load_components(
                diffusion_asset,
                vae_asset,
                diffusion_identity,
                vae_identity,
            )
            cases = []
            for frames in (1, 5):
                torch.cuda.synchronize(0)
                started = time.perf_counter()
                first = _run_case(model, vae, frames)
                torch.cuda.synchronize(0)
                cold_seconds = time.perf_counter() - started
                started = time.perf_counter()
                second = _run_case(model, vae, frames)
                torch.cuda.synchronize(0)
                warm_seconds = time.perf_counter() - started
                if first["output"]["sha256_float32"] != second["output"]["sha256_float32"]:
                    raise RuntimeError(f"SeedVR2 {frames}-frame output was not deterministic")
                first["repeat_output_sha256_float32"] = second["output"]["sha256_float32"]
                first["cold_execution_seconds"] = cold_seconds
                first["warm_execution_seconds"] = warm_seconds
                cases.append(first)
            return {
                "status": "passed",
                "claim": str(lock_path),
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "logical_device": 0,
                "device_name": properties.name,
                "torch": torch.__version__,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "cases": cases,
            }
        finally:
            if model is not None:
                model.terminal_release()
            if vae is not None:
                vae.terminal_release()
            torch.cuda.empty_cache()


def _verify_comfy_root(comfy_root: Path) -> Path:
    root = comfy_root.resolve(strict=True)
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=root, text=True).strip()
    if head != COMFYUI_COMMIT:
        raise RuntimeError(f"comparison requires ComfyUI {COMFYUI_COMMIT}, found {head}")
    changed = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=root,
        text=True,
    ).strip()
    if changed:
        raise RuntimeError("comparison requires an unmodified ComfyUI checkout")
    return root


def _diffusion_inputs(torch: Any, device: object) -> tuple[Any, Any, Any]:
    latent = torch.linspace(-1.0, 1.0, 16 * 2 * 2, dtype=torch.float32).reshape(1, 16, 1, 2, 2)
    restored = torch.linspace(1.0, -1.0, 16 * 2 * 2, dtype=torch.float32).reshape(1, 16, 1, 2, 2)
    condition = torch.cat((restored, torch.ones((1, 1, 1, 2, 2))), dim=1)
    timestep = torch.tensor([0.5], dtype=torch.float32)
    return (
        latent.to(device=device, dtype=torch.bfloat16),
        condition.to(device=device, dtype=torch.bfloat16),
        timestep.to(device=device),
    )


def _video_source(torch: Any, frames: int) -> Any:
    return torch.linspace(
        0.0,
        1.0,
        frames * 96 * 96 * 3,
        dtype=torch.float32,
    ).reshape(frames, 96, 96, 3)


def _decode_latent(torch: Any, frames: int, device: object) -> Any:
    latent_frames = (frames + 3) // 4
    return (
        torch.linspace(
            -0.5,
            0.5,
            16 * latent_frames * 12 * 12,
            dtype=torch.float32,
        )
        .reshape(1, 16, latent_frames, 12, 12)
        .to(device=device)
    )


def _dinkster_comparison_outputs(
    diffusion_asset: object,
    vae_asset: object,
    diffusion_identity: str,
    vae_identity: str,
    torch: Any,
) -> dict[str, Any]:
    from dinkster_inference_torch import SeedVR2CodecRuntime

    model, vae = _load_components(
        diffusion_asset,
        vae_asset,
        diffusion_identity,
        vae_identity,
    )
    outputs: dict[str, Any] = {}
    try:
        device = model.load_device
        latent, condition, timestep = _diffusion_inputs(torch, device)
        with model.stage("diffusion"):
            diffusion = model.runtime.assembled.diffusion
            with diffusion.materialized_text_conditioning(
                "positive", device=device, dtype=torch.bfloat16
            ) as stored_context:
                context = stored_context.unsqueeze(0)
                with torch.inference_mode():
                    outputs["diffusion"] = (
                        diffusion(
                            latent,
                            timestep,
                            context,
                            condition=condition,
                            transformer_options={"cond_or_uncond": ["positive"]},
                        )
                        .detach()
                        .cpu()
                    )

        codec_runtime = SeedVR2CodecRuntime(
            vae.component,
            compute_dtype=torch.float16,
        )
        with vae.stage():
            for frames in (1, 5):
                source = _video_source(torch, frames)
                content = source.permute(3, 0, 1, 2).unsqueeze(0).to(device)
                latent_input = _decode_latent(torch, frames, device)
                with torch.inference_mode():
                    outputs[f"vae_encode_{frames}"] = (
                        codec_runtime.codec.encode(content).detach().cpu()
                    )
                    outputs[f"vae_encode_tiled_{frames}"] = (
                        codec_runtime.codec.encode_tiled(
                            content,
                            tile=(64, 64, 64),
                            overlap=(8, 16, 16),
                        )
                        .detach()
                        .cpu()
                    )
                    outputs[f"vae_decode_{frames}"] = (
                        codec_runtime.codec.decode(latent_input).detach().cpu()
                    )
                    outputs[f"vae_decode_tiled_{frames}"] = (
                        codec_runtime.codec.decode_tiled(
                            latent_input,
                            tile=(16, 8, 8),
                            overlap=(2, 2, 2),
                        )
                        .detach()
                        .cpu()
                    )
    finally:
        model.terminal_release()
        vae.terminal_release()
        del model, vae
        gc.collect()
        torch.cuda.empty_cache()
    return outputs


def _enable_comfy(comfy_root: Path) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(comfy_root))
    saved_argv = sys.argv
    sys.argv = [
        str(comfy_root / "main.py"),
        "--highvram",
        "--use-pytorch-cross-attention",
        "--disable-async-offload",
    ]
    try:
        import comfy.options as options

        options.enable_args_parsing()
        import comfy.model_management as model_management
        import comfy.sd as comfy_sd
        import comfy.utils as comfy_utils
    finally:
        sys.argv = saved_argv

    return comfy_sd, comfy_utils, model_management


def _comfy_comparison_outputs(
    diffusion_path: Path,
    vae_path: Path,
    comfy_root: Path,
    torch: Any,
) -> dict[str, Any]:
    comfy_sd, comfy_utils, model_management = _enable_comfy(comfy_root)
    outputs: dict[str, Any] = {}
    patcher = comfy_sd.load_diffusion_model(
        str(diffusion_path),
        model_options={"dtype": torch.bfloat16},
    )
    try:
        model_management.load_models_gpu([patcher], force_full_load=True)
        device = patcher.load_device
        latent, condition, timestep = _diffusion_inputs(torch, device)
        diffusion = patcher.model.diffusion_model
        context = diffusion.positive_conditioning.to(device=device, dtype=torch.bfloat16).unsqueeze(
            0
        )
        with torch.inference_mode():
            outputs["diffusion"] = (
                diffusion(
                    latent,
                    timestep,
                    context,
                    condition=condition,
                    transformer_options={"cond_or_uncond": ["positive"]},
                )
                .detach()
                .cpu()
            )
    finally:
        model_management.unload_all_models()
        del patcher
        gc.collect()
        torch.cuda.empty_cache()

    state, metadata = comfy_utils.load_torch_file(str(vae_path), return_metadata=True)
    vae = comfy_sd.VAE(
        sd=state,
        metadata=metadata,
        dtype=torch.float16,
    )
    del state
    try:
        device = vae.device
        for frames in (1, 5):
            source = _video_source(torch, frames)
            latent_input = _decode_latent(torch, frames, device)
            with torch.inference_mode():
                outputs[f"vae_encode_{frames}"] = vae.encode(source).detach().cpu()
                outputs[f"vae_encode_tiled_{frames}"] = (
                    vae.encode_tiled(
                        source,
                        tile_x=64,
                        tile_y=64,
                        overlap=16,
                        tile_t=64,
                        overlap_t=8,
                    )
                    .detach()
                    .cpu()
                )
                outputs[f"vae_decode_{frames}"] = (
                    vae.decode(latent_input).movedim(-1, 1).detach().cpu()
                )
                outputs[f"vae_decode_tiled_{frames}"] = (
                    vae.decode_tiled(
                        latent_input,
                        tile_x=8,
                        tile_y=8,
                        overlap=2,
                        tile_t=16,
                        overlap_t=2,
                    )
                    .movedim(-1, 1)
                    .detach()
                    .cpu()
                )
    finally:
        model_management.unload_all_models()
        del vae
        gc.collect()
        torch.cuda.empty_cache()
    return outputs


def _comparison_record(torch: Any, dinkster: Any, comfy: Any) -> dict[str, object]:
    if tuple(dinkster.shape) != tuple(comfy.shape):
        return {
            "exact": False,
            "dinkster": _tensor_record(dinkster),
            "comfy": _tensor_record(comfy),
            "error": "shape mismatch",
        }
    dinkster_float = dinkster.float()
    comfy_float = comfy.float()
    difference = (dinkster_float - comfy_float).abs()
    return {
        "exact": bool(torch.equal(dinkster, comfy)),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "dinkster": _tensor_record(dinkster),
        "comfy": _tensor_record(comfy),
    }


def _comparison_launch(
    diffusion_asset: object,
    vae_asset: object,
    diffusion_identity: str,
    vae_identity: str,
    diffusion_path: Path,
    vae_path: Path,
    comfy_root: Path,
    thread_id: str,
) -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("GPU comparison requires CUDA_VISIBLE_DEVICES=1")
    comfy_root = _verify_comfy_root(comfy_root)
    with _gpu1_claim(thread_id, "SeedVR2-current-ComfyUI-comparison") as lock_path:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("GPU comparison requires exactly physical GPU 1 to be visible")
        torch.manual_seed(1065)
        torch.cuda.manual_seed_all(1065)
        torch.backends.cudnn.benchmark = False
        properties = torch.cuda.get_device_properties(0)
        dinkster = _dinkster_comparison_outputs(
            diffusion_asset,
            vae_asset,
            diffusion_identity,
            vae_identity,
            torch,
        )
        comfy = _comfy_comparison_outputs(
            diffusion_path,
            vae_path,
            comfy_root,
            torch,
        )
        if set(dinkster) != set(comfy):
            raise RuntimeError("comparison implementations produced different case sets")
        cases = {
            name: _comparison_record(torch, dinkster[name], comfy[name])
            for name in sorted(dinkster)
        }
        return {
            "status": (
                "passed" if all(bool(case["exact"]) for case in cases.values()) else "failed"
            ),
            "claim": str(lock_path),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "logical_device": 0,
            "device_name": properties.name,
            "torch": torch.__version__,
            "comfyui_commit": COMFYUI_COMMIT,
            "comfyui_root": str(comfy_root),
            "cases": cases,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("offline", "gpu", "compare"))
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diffusion-sha256", default=OFFICIAL_3B_SHA256)
    parser.add_argument("--vae-sha256", default=OFFICIAL_VAE_SHA256)
    parser.add_argument("--thread-id", default=os.environ.get("DINKSTER_GPU_CLAIM_THREAD", ""))
    parser.add_argument("--comfy-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evidence: dict[str, object]
    if args.output.exists():
        evidence = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        evidence = {}
    evidence["official_source"] = OFFICIAL_SOURCE
    evidence["official_revision"] = OFFICIAL_REVISION
    evidence["validator"] = str(Path(__file__).relative_to(REPO_ROOT))
    try:
        diffusion_asset, diffusion_record = _artifact(args.diffusion, args.diffusion_sha256)
        diffusion_record["immutable_url"] = (
            f"{OFFICIAL_SOURCE}/resolve/{OFFICIAL_REVISION}/diffusion_models/" + args.diffusion.name
        )
        evidence["diffusion_artifact"] = diffusion_record
        _persist(args.output, evidence)
        vae_asset, vae_record = _artifact(args.vae, args.vae_sha256)
        vae_record["immutable_url"] = (
            f"{OFFICIAL_SOURCE}/resolve/{OFFICIAL_REVISION}/vae/" + args.vae.name
        )
        evidence["vae_artifact"] = vae_record
        _persist(args.output, evidence)
        diffusion_identity, diffusion_plan = _identity(diffusion_asset, "diffusion")
        vae_identity, vae_plan = _identity(vae_asset, "vae")
        evidence["diffusion_plan"] = diffusion_plan
        evidence["vae_plan"] = vae_plan
        _persist(args.output, evidence)
        if args.mode == "offline":
            evidence["offline_launch"] = _offline_launch(
                diffusion_asset,
                vae_asset,
                diffusion_identity,
                vae_identity,
            )
        else:
            if not args.thread_id:
                raise RuntimeError("GPU validation requires --thread-id")
            evidence["offline_launch"] = _offline_launch(
                diffusion_asset, vae_asset, diffusion_identity, vae_identity
            )
            _persist(args.output, evidence)
            if args.mode == "gpu":
                evidence["gpu_launch"] = _gpu_launch(
                    diffusion_asset,
                    vae_asset,
                    diffusion_identity,
                    vae_identity,
                    args.thread_id,
                )
            else:
                if args.comfy_root is None:
                    raise RuntimeError("comparison requires --comfy-root")
                evidence["comparison"] = _comparison_launch(
                    diffusion_asset,
                    vae_asset,
                    diffusion_identity,
                    vae_identity,
                    args.diffusion,
                    args.vae,
                    args.comfy_root,
                    args.thread_id,
                )
                _persist(args.output, evidence)
                if evidence["comparison"]["status"] != "passed":  # type: ignore[index]
                    raise RuntimeError("current ComfyUI comparison was not bit-exact")
    except BaseException as error:
        evidence[f"{args.mode}_error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        _persist(args.output, evidence)
        raise
    evidence.pop(f"{args.mode}_error", None)
    evidence["updated_at"] = datetime.now(UTC).isoformat()
    _persist(args.output, evidence)


if __name__ == "__main__":
    main()
