"""Run the official NetaYume Lumina2 workflow through production native nodes.

The preflight mode proves dispatch and production construction imports without
importing torch. Run it with CUDA hidden immediately before the GPU mode.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHECKPOINT_NAME = "NetaYumev35_pretrained_all_in_one.safetensors"
CHECKPOINT_BYTES = 10_620_231_237
CHECKPOINT_SHA256 = "4125cb490996ea85c8e3ba242866da02efd83a6a2c079dc8924a95eaa8327a44"
CHECKPOINT_BLAKE3 = "3e84945252f6e757b51ada84726dffd5da38994a3c3fc612b85d3efb15e4cf35"
CHECKPOINT_URL = (
    "https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0/resolve/"
    "69c29c441119d43405d8373c654b16827d3cf46e/"
    "NetaYumev35_pretrained_all_in_one.safetensors"
)
POSITIVE_PREFIX = (
    "You are an assistant designed to generate high quality anime images based on "
    "textual prompts. <Prompt Start> "
)
NEGATIVE_PROMPT = (
    "You are an assistant designed to generate low-quality images based on textual prompts "
    "<Prompt Start> blurry, worst quality, low quality, jpeg artifacts, signature, watermark, "
    "username, error, deformed hands, bad anatomy, extra limbs, poorly drawn hands, poorly "
    "drawn face, mutation, deformed, extra eyes, extra arms, extra legs, malformed limbs, fused "
    "fingers, too many fingers, long neck, cross-eyed, bad proportions, missing arms, missing "
    "legs, extra digit, fewer digits, cropped"
)
POSITIVE_PROMPT = POSITIVE_PREFIX + (
    "1girl, solo, long flowing hair, white dress, standing in a field of flowers, sunset, "
    "detailed anime illustration"
)


@dataclass(frozen=True)
class ArtifactProof:
    path: Path
    digest: str
    size: int
    verification: Any


@dataclass(frozen=True)
class FixedAssetResolver:
    proof: ArtifactProof

    def resolve(self, digest: str) -> Path | None:
        return self.proof.path if digest == self.proof.digest else None

    def resolve_asset(self, digest: str) -> object | None:
        if digest != self.proof.digest:
            return None
        from dinkster_assets.model import AssetResolution

        return AssetResolution(self.proof.path, self.proof.verification)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _write_evidence(path: Path, update: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current: dict[str, object] = {}
    if path.is_file():
        decoded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(decoded, dict):
            current = decoded
    phases = dict(current.get("phases", {})) if isinstance(current.get("phases"), dict) else {}
    next_phases = update.pop("phases", None)
    if isinstance(next_phases, dict):
        phases.update(next_phases)
    current.update(update)
    current["phases"] = phases
    current["updated_at"] = _timestamp()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _phase(path: Path, name: str, status: str, **details: object) -> None:
    _write_evidence(
        path,
        {
            "phases": {
                name: {
                    "status": status,
                    "timestamp": _timestamp(),
                    **details,
                }
            }
        },
    )


def _verify_checkpoint(path: Path) -> ArtifactProof:
    from dinkster_assets.identity import CHUNK_SIZE, DIGEST_PREFIX, new_hasher
    from dinkster_assets.integrity import verification_record

    if not path.is_file():
        raise FileNotFoundError(path)
    sha256 = hashlib.sha256()
    blake3 = new_hasher()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        while chunk := handle.read(CHUNK_SIZE):
            sha256.update(chunk)
            blake3.update(chunk)
        after = os.fstat(handle.fileno())
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError("checkpoint changed while its digests were computed")
    actual_sha256 = sha256.hexdigest()
    actual_blake3 = blake3.hexdigest()
    if after.st_size != CHECKPOINT_BYTES:
        raise RuntimeError(
            f"checkpoint size mismatch: expected {CHECKPOINT_BYTES}, got {after.st_size}"
        )
    if actual_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {CHECKPOINT_SHA256}, got {actual_sha256}"
        )
    if actual_blake3 != CHECKPOINT_BLAKE3:
        raise RuntimeError(
            f"checkpoint BLAKE3 mismatch: expected {CHECKPOINT_BLAKE3}, got {actual_blake3}"
        )
    digest = DIGEST_PREFIX + actual_blake3
    verification = verification_record(digest, after)
    if verification is None:
        raise RuntimeError("checkpoint verification could not be bound to its descriptor")
    return ArtifactProof(path, digest, after.st_size, verification)


def _asset_value(proof: ArtifactProof) -> object:
    from dinkster_values import Value, ValueMeta
    from dinkster_values.model import PyObjPayload

    return Value(
        type_id="dinkster.asset",
        fingerprint=proof.digest,
        meta=ValueMeta({"digest": proof.digest, "name": CHECKPOINT_NAME}),
        payload=PyObjPayload(None),
    )


def _asset_ref(proof: ArtifactProof) -> object:
    from dinkster_assets import AssetRef

    return AssetRef(
        digest=proof.digest,
        name=CHECKPOINT_NAME,
        size=proof.size,
        resolver=FixedAssetResolver(proof),
    )


def _select_executions(proof: ArtifactProof) -> dict[str, object]:
    from dinkster.native_policy import NativeDispatchPolicy

    diagnostics: list[object] = []
    policy = NativeDispatchPolicy(
        lambda digest: proof.path if digest == proof.digest else None,
        diagnostics.append,
    )
    asset = _asset_value(proof)
    requests = (
        (
            "checkpoint",
            "dinkster.load_checkpoint",
            {"checkpoint": asset},
            ("bfloat16", "float32", "bfloat16"),
        ),
    )
    selections: dict[str, object] = {}
    for role, node_type, inputs, expected_dtypes in requests:
        selection = asyncio.run(
            policy.select(
                node_type,
                inputs,
                ("compat", {"compat": "compat-tag", "compat@native": "native-default"}),
            )
        )
        if selection is None:
            raise RuntimeError(f"native dispatch did not select an arm for {node_type}")
        if selection.target != "compat@native":
            raise RuntimeError(
                f"native dispatch selected unexpected {node_type} target {selection.target!r}"
            )
        dtypes = (selection.diffusion_dtype, selection.text_dtype, selection.vae_dtype)
        if dtypes != expected_dtypes:
            raise RuntimeError(f"native dispatch selected unexpected {node_type} dtypes {dtypes!r}")
        selections[role] = selection
    if diagnostics:
        raise RuntimeError(f"native dispatch emitted diagnostics: {diagnostics!r}")
    return selections


def _production_boundary() -> dict[str, object]:
    from dinkster_compat_comfy.native_arm import (
        GenerationBasicScheduler,
        GenerationClipTextEncode,
        GenerationEmptyLatentImage,
        GenerationKSamplerSelect,
        GenerationLoadCheckpoint,
        GenerationModelSamplingAuraFlow,
        GenerationSamplerCustom,
        GenerationVAEDecode,
    )

    return {
        "load_checkpoint": GenerationLoadCheckpoint,
        "clip_text_encode": GenerationClipTextEncode,
        "empty_latent_image": GenerationEmptyLatentImage,
        "model_sampling_aura_flow": GenerationModelSamplingAuraFlow,
        "basic_scheduler": GenerationBasicScheduler,
        "ksampler_select": GenerationKSamplerSelect,
        "sampler_custom": GenerationSamplerCustom,
        "vae_decode": GenerationVAEDecode,
    }


def _selection_record(selection: object) -> dict[str, object]:
    return {
        "target": selection.target,
        "cache_tag": selection.cache_tag,
        "execution_arm": selection.execution_arm,
        "fp8_matmul": selection.fp8_matmul,
        "diffusion_dtype": selection.diffusion_dtype,
        "text_dtype": selection.text_dtype,
        "vae_dtype": selection.vae_dtype,
        "attention_policy": selection.attention_policy,
        "attention_route_token": (
            None
            if selection.attention_route_token is None
            else repr(selection.attention_route_token)
        ),
    }


def run_preflight(
    checkpoint: Path, evidence: Path
) -> tuple[ArtifactProof, object, dict[str, object]]:
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise RuntimeError("preflight must start before torch is imported")
    _write_evidence(
        evidence,
        {
            "schema": "dinkster.lumina2.official-evidence.v1",
            "mode": "preflight",
            "checkpoint": {
                "name": CHECKPOINT_NAME,
                "path": str(checkpoint),
                "url": CHECKPOINT_URL,
                "bytes": CHECKPOINT_BYTES,
                "sha256": CHECKPOINT_SHA256,
                "blake3": CHECKPOINT_BLAKE3,
            },
            "workflow": {
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "cfg": 4.0,
                "shift": 4.0,
                "sampler": "res_multistep",
                "scheduler": "simple",
                "seed": 1064,
                "positive": POSITIVE_PROMPT,
                "negative": NEGATIVE_PROMPT,
            },
        },
    )
    _phase(evidence, "artifact_authentication", "running")
    proof = _verify_checkpoint(checkpoint)
    _phase(
        evidence,
        "artifact_authentication",
        "success",
        bytes=proof.size,
        sha256=CHECKPOINT_SHA256,
        blake3=CHECKPOINT_BLAKE3,
    )
    _phase(evidence, "native_dispatch", "running")
    selections = _select_executions(proof)
    _phase(
        evidence,
        "native_dispatch",
        "success",
        components={role: _selection_record(selection) for role, selection in selections.items()},
    )
    _phase(evidence, "production_import", "running")
    boundary = _production_boundary()
    imported_torch = any(name == "torch" or name.startswith("torch.") for name in sys.modules)
    if imported_torch:
        raise RuntimeError("production launch boundary imported torch during preflight")
    _phase(
        evidence,
        "production_import",
        "success",
        imported_torch=False,
        cuda_initialized=False,
        nodes=sorted(boundary),
    )
    _write_evidence(evidence, {"preflight_completed_at": _timestamp()})
    return proof, selections, boundary


def _execution_context(selection: object) -> object:
    from dinkster_workers.execution import ExecutionContext

    return ExecutionContext(
        arm=selection.target,
        expected_execution_identity=selection.cache_tag,
        fp8_matmul=selection.fp8_matmul,
        diffusion_dtype=selection.diffusion_dtype,
        text_dtype=selection.text_dtype,
        vae_dtype=selection.vae_dtype,
        attention_policy=selection.attention_policy,
        attention_route_token=selection.attention_route_token,
    )


def _load_components(
    proof: ArtifactProof,
    selections: dict[str, object],
    boundary: dict[str, object],
) -> dict[str, object]:
    from dinkster_workers.execution import use_execution_context

    asset = _asset_ref(proof)
    with use_execution_context(_execution_context(selections["checkpoint"])):
        return dict(boundary["load_checkpoint"].execute(checkpoint=asset))


def _tensor_record(tensor: Any) -> dict[str, object]:
    value = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "finite": bool(value.isfinite().all().item()),
        "minimum": float(value.min().item()),
        "maximum": float(value.max().item()),
        "mean": float(value.mean().item()),
        "standard_deviation": float(value.std().item()),
    }


def run_gpu(checkpoint: Path, evidence: Path, output: Path) -> None:
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_device or "," in visible_device:
        raise RuntimeError("GPU mode requires exactly one CUDA_VISIBLE_DEVICES selector")
    proof, selections, boundary = run_preflight(checkpoint, evidence)
    _write_evidence(evidence, {"mode": "gpu", "gpu_started_at": _timestamp()})

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"GPU mode requires exactly one visible CUDA device, got {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    _phase(
        evidence,
        "gpu_admission",
        "success",
        cuda_visible_devices=visible_device,
        visible_device_count=torch.cuda.device_count(),
        visible_device_name=properties.name,
        compute_capability=f"{properties.major}.{properties.minor}",
        total_memory_bytes=properties.total_memory,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )
    torch.cuda.reset_peak_memory_stats()

    loaded: dict[str, object] | None = None
    model_handle: object | None = None
    clip_handle: object | None = None
    vae_handle: object | None = None
    timings: dict[str, float] = {}

    def timed(name: str, operation: Any) -> object:
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = operation()
        torch.cuda.synchronize()
        timings[name] = time.perf_counter() - started
        return result

    try:
        _phase(evidence, "component_load", "running")
        loaded = timed(
            "component_load_s",
            lambda: _load_components(proof, selections, boundary),
        )
        assert isinstance(loaded, dict)
        model_handle = loaded["model"]
        clip_handle = loaded["clip"]
        vae_handle = loaded["vae"]
        model_identity = model_handle.recipe.runtime_identity
        model_family = model_handle.recipe.family_id
        component_identities = {
            "model": model_identity,
            "clip": clip_handle.recipe.runtime_identity,
            "vae": vae_handle.resource_identity,
        }
        if model_family != "dinkster.lumina2":
            raise RuntimeError(f"loaded unexpected model family {model_family!r}")
        _phase(
            evidence,
            "component_load",
            "success",
            family=model_family,
            selection_identities={
                role: selection.cache_tag for role, selection in selections.items()
            },
            component_identities=component_identities,
            seconds=timings["component_load_s"],
        )

        _phase(evidence, "text_encode", "running")
        positive = timed(
            "positive_encode_s",
            lambda: boundary["clip_text_encode"].execute(
                text=POSITIVE_PROMPT,
                clip=clip_handle,
            )["conditioning"],
        )
        negative = timed(
            "negative_encode_s",
            lambda: boundary["clip_text_encode"].execute(
                text=NEGATIVE_PROMPT,
                clip=clip_handle,
            )["conditioning"],
        )
        _phase(
            evidence,
            "text_encode",
            "success",
            positive_seconds=timings["positive_encode_s"],
            negative_seconds=timings["negative_encode_s"],
        )

        sampled_model = boundary["model_sampling_aura_flow"].execute(
            model=model_handle,
            shift=4.0,
        )["model"]
        sigmas = boundary["basic_scheduler"].execute(
            model=sampled_model,
            scheduler="simple",
            steps=30,
            denoise=1.0,
        )["sigmas"]
        sampler = boundary["ksampler_select"].execute(sampler_name="res_multistep")["sampler"]
        _phase(
            evidence,
            "sampling_plan",
            "success",
            shift=4.0,
            sampler="res_multistep",
            scheduler="simple",
            sigmas=list(sigmas.values),
        )

        latent = boundary["empty_latent_image"].execute(
            width=1024,
            height=1024,
            batch_size=1,
        )["latent"]
        _phase(evidence, "sampling", "running")
        sampled = timed(
            "sampling_s",
            lambda: boundary["sampler_custom"].execute(
                model=sampled_model,
                add_noise=True,
                noise_seed=1064,
                cfg=4.0,
                positive=positive,
                negative=negative,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latent,
            )["output"],
        )
        sampled_tensor = sampled["samples"]
        sampled_record = _tensor_record(sampled_tensor)
        if not sampled_record["finite"]:
            raise RuntimeError("sampled latent contains non-finite values")
        _phase(
            evidence,
            "sampling",
            "success",
            seconds=timings["sampling_s"],
            latent=sampled_record,
        )

        _phase(evidence, "vae_decode", "running")
        image = timed(
            "vae_decode_s",
            lambda: boundary["vae_decode"].execute(samples=sampled, vae=vae_handle)["image"],
        )
        image_record = _tensor_record(image)
        if not image_record["finite"]:
            raise RuntimeError("decoded image contains non-finite values")
        pixels = (
            image.detach().float().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
        )
        if pixels.shape != (1, 1024, 1024, 3):
            raise RuntimeError(f"decoded image has unexpected shape {pixels.shape!r}")
        from PIL import Image

        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels[0], mode="RGB").save(output)
        image_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
        _phase(
            evidence,
            "vae_decode",
            "success",
            seconds=timings["vae_decode_s"],
            image=image_record,
            output=str(output),
            output_sha256=image_sha256,
            output_bytes=output.stat().st_size,
        )
        _write_evidence(
            evidence,
            {
                "timings": timings,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                "gpu_completed_at": _timestamp(),
                "status": "success",
            },
        )
    finally:
        if model_handle is not None and not model_handle.released:
            model_handle.terminal_release()
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not args.preflight and args.output is None:
        parser.error("GPU mode requires --output")
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.preflight:
            run_preflight(args.checkpoint, args.evidence)
            _write_evidence(args.evidence, {"status": "preflight-success"})
        else:
            run_gpu(args.checkpoint, args.evidence, args.output)
    except BaseException as exc:
        _write_evidence(
            args.evidence,
            {
                "status": "failure",
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            },
        )
        raise


if __name__ == "__main__":
    main()
