#!/usr/bin/env python3
"""Run official TRELLIS.2 or Pixal3D weights through the native node surface."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch  # pyright: ignore[reportMissingImports]
from dinkster_assets import AssetRef
from dinkster_compat_comfy import native_arm
from dinkster_inference import (
    BFLOAT16,
    FLOAT32,
    Trellis2ArtifactRole,
    load_safetensors_header,
    plan_trellis2_artifact,
    trellis2_artifact_runtime_identity,
)
from PIL import Image


@dataclass(frozen=True)
class Artifact:
    relative: str
    byte_size: int
    sha256: str
    digest: str
    url: str


_COMFY_REVISION = "463441b1c32829ee876e4f297dcfff533cb357a7"
_PIXAL_REVISION = "e69bd0b6c7b959661a87051187aab18e8e5abcdf"
_WORKFLOW_REVISION = "d3b4a9e89573162b005961865164c18c8ae2206b"
_INPUT_BYTES = 1_106_832
_INPUT_SHA256 = "aaf42f16e262e88bef7c947771a745e817cc0b533922a8388b51165ede091fce"
_INPUT_URL = (
    "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/"
    f"{_WORKFLOW_REVISION}/input/viking_wolf_rune_axe.png"
)
ARTIFACTS = {
    "trellis2": Artifact(
        "comfy/trellis_2_int8_convrot.safetensors",
        5_253_048_192,
        "d01952ad137213f6a868f86b6b877026276f84af5eec23069217475a0bad3a31",
        "blake3:0a1d795a88344ceeac954046c003c625a786062a3a4d7250c59348427f825064",
        "https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/"
        f"{_COMFY_REVISION}/diffusion_models/trellis_2_int8_convrot.safetensors",
    ),
    "pixal3d": Artifact(
        "comfy/pixal3d_int8_convrot.safetensors",
        5_584_555_824,
        "4621eac3b715484f79303c7152af641fe0b2b14f4d0e3d394fd6922d00f955ec",
        "blake3:ae052e95a3c2de1dc8580c1a5376c29367bd1886a9ac460e95af616dbc59295e",
        "https://huggingface.co/Comfy-Org/Pixal3D/resolve/"
        f"{_PIXAL_REVISION}/diffusion_models/pixal3d_int8_convrot.safetensors",
    ),
    "vision": Artifact(
        "comfy/dino_v3_L_naf_fp32.safetensors",
        1_215_214_176,
        "4ad2ec4e0879a5b5b04cd97325cc37da954a7b6edca5170b86510f17f2b2290f",
        "blake3:79dfbebaa70957c43791d7e4fa93edba5ddb582e65d025c85d6abd31667c5a49",
        "https://huggingface.co/Comfy-Org/Pixal3D/resolve/"
        f"{_PIXAL_REVISION}/clip_vision/dino_v3_L_naf_fp32.safetensors",
    ),
    "shape": Artifact(
        "comfy/trellis_2_shape_vae_bf16.safetensors",
        1_095_844_024,
        "de0cb4949a76c59ee5c091a995a69bcc8c51d5aeda939f0c641a50d2a72341f4",
        "blake3:2b2d9fb7c274ff67a074824d7922c55e60e625a17f3222367bd37664dac7b65b",
        "https://huggingface.co/Comfy-Org/Pixal3D/resolve/"
        f"{_PIXAL_REVISION}/vae/trellis_2_shape_vae_bf16.safetensors",
    ),
    "texture": Artifact(
        "comfy/trellis_2_texture_vae_bf16.safetensors",
        948_461_364,
        "714e5ebf094a610e12a8e3b5175c18a62f37f6ea4218acb6073644456b73ab0e",
        "blake3:2cdcca4136cc6c630096d7d7418ded58f399c55b97cb1a059a37f03b9ba7224d",
        "https://huggingface.co/Comfy-Org/Pixal3D/resolve/"
        f"{_PIXAL_REVISION}/vae/trellis_2_texture_vae_bf16.safetensors",
    ),
}


class Resolver:
    def __init__(self, digest: str, path: Path) -> None:
        self.digest = digest
        self.path = path

    def resolve(self, digest: str) -> Path | None:
        return self.path if digest == self.digest else None


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_artifacts(root: Path) -> dict[str, AssetRef]:
    assets: dict[str, AssetRef] = {}
    for name, artifact in ARTIFACTS.items():
        path = root / artifact.relative
        if path.stat().st_size != artifact.byte_size:
            raise RuntimeError(f"{name} byte size differs from its immutable pin")
        digest = sha256(path)
        if digest != artifact.sha256:
            raise RuntimeError(f"{name} sha256 differs from its immutable pin")
        assets[name] = AssetRef(
            artifact.digest,
            path.name,
            artifact.byte_size,
            resolver=Resolver(artifact.digest, path),
        )
    return assets


def identity(root: Path, asset: AssetRef, role: Trellis2ArtifactRole, dtype: Any) -> str:
    path = root / next(
        artifact.relative for artifact in ARTIFACTS.values() if artifact.digest == asset.digest
    )
    source = load_safetensors_header(
        path,
        asset_digest=asset.digest,
        asset_size=asset.size,
    )
    planned = plan_trellis2_artifact(source, role=role, path=path)
    return trellis2_artifact_runtime_identity(planned, dtype)


def rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is unavailable")


def peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def measure(stage: str, function: Any, records: list[dict[str, Any]]) -> Any:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = function()
    torch.cuda.synchronize()
    record = {
        "stage": stage,
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved": torch.cuda.max_memory_reserved(),
        "gpu_residual_allocated": torch.cuda.memory_allocated(),
        "gpu_residual_reserved": torch.cuda.memory_reserved(),
        "host_rss": rss_bytes(),
        "host_peak_rss": peak_rss_bytes(),
    }
    records.append(record)
    print(json.dumps(record), flush=True)
    return output


def trim_runtime(stage: str, records: list[dict[str, Any]]) -> None:
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    record = {
        "stage": stage,
        "gpu_residual_allocated": torch.cuda.memory_allocated(),
        "gpu_residual_reserved": torch.cuda.memory_reserved(),
        "host_rss": rss_bytes(),
        "host_peak_rss": peak_rss_bytes(),
    }
    records.append(record)
    print(json.dumps(record), flush=True)


def load_official_image(path: Path) -> torch.Tensor:
    if path.stat().st_size != _INPUT_BYTES:
        raise RuntimeError("workflow input byte size differs from its immutable pin")
    digest = sha256(path)
    if digest != _INPUT_SHA256:
        raise RuntimeError("workflow input sha256 differs from its immutable pin")
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(pixels).unsqueeze(0)


def load_condition_image(path: Path) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32 or array.shape != (1, 1024, 1024, 3):
        raise RuntimeError(f"condition image must have float32 shape [1,1024,1024,3], got {array}")
    return torch.from_numpy(array.copy())


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu()
    if tensor.is_floating_point():
        tensor = tensor.float()
    tensor = tensor.contiguous()
    raw = tensor.numpy().tobytes()
    summary: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if tensor.numel():
        values = tensor.double()
        summary.update(
            min=float(values.min()),
            max=float(values.max()),
            mean=float(values.mean()),
            std=float(values.std(correction=0)),
        )
    return summary


def canonical(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().cpu()
    return value.float().contiguous() if value.is_floating_point() else value.contiguous()


def patched_model(
    model: object,
    *,
    shift: float | None,
    cfg_start: float,
    rescale: float,
) -> object:
    value = model
    if shift is not None:
        value = native_arm.GenerationModelSamplingSD3.execute(model=value, shift=shift)["model"]
    value = native_arm.GenerationCFGOverride.execute(
        model=value,
        cfg=1.0,
        start_percent=cfg_start,
        end_percent=1.0,
    )["model"]
    return native_arm.GenerationRescaleCfg.execute(model=value, multiplier=rescale)["model"]


def sample(
    model: object,
    positive: object,
    negative: object,
    latent: object,
    *,
    seed: int,
    steps: int,
    cfg: float,
    scheduler: str,
) -> object:
    return native_arm.NativeKSampler.execute(
        model=model,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name="euler",
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent_image=latent,
        denoise=1.0,
    )["latent"]


def run_profile(
    run: int,
    *,
    model: object,
    vision: object,
    shape_vae: object,
    texture_vae: object,
    pixal3d: bool,
    direct: int,
    cascade: int | None,
    smoke: bool,
    image: torch.Tensor,
    records: list[dict[str, Any]],
    capture_tensors: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    prefix = f"run-{run}"
    condition = measure(
        f"{prefix}:condition",
        lambda: (
            native_arm.GenerationPixal3DConditioning.execute(
                clip_vision_model=vision,
                image=image,
                camera_angle_x=49.13,
            )
            if pixal3d
            else native_arm.GenerationTrellis2Conditioning.execute(
                clip_vision_model=vision,
                image=image,
            )
        ),
        records,
    )
    positive, negative = condition["positive"], condition["negative"]
    empty = native_arm.GenerationEmptyTrellis2LatentStructure.execute(batch_size=1)["latent"]
    structure_model = patched_model(model, shift=5.0, cfg_start=0.667, rescale=0.7)
    structure = measure(
        f"{prefix}:sample-structure",
        lambda: sample(
            structure_model,
            positive,
            negative,
            empty,
            seed=56,
            steps=1 if smoke else 12,
            cfg=7.5,
            scheduler="normal",
        ),
        records,
    )
    voxel = measure(
        f"{prefix}:decode-structure",
        lambda: native_arm.GenerationVaeDecodeStructureTrellis2.execute(
            samples=structure,
            vae=shape_vae,
            resolution="64" if direct == 1024 else "32",
        )["voxel"],
        records,
    )
    shape_inputs = native_arm.GenerationTrellis2ShapeStage.execute(
        positive=positive,
        negative=negative,
        voxel=voxel,
    )
    shape_model = patched_model(model, shift=None, cfg_start=0.769, rescale=0.5)
    shape = measure(
        f"{prefix}:sample-shape-{direct}",
        lambda: sample(
            shape_model,
            shape_inputs["positive"],
            shape_inputs["negative"],
            shape_inputs["latent"],
            seed=42,
            steps=1 if smoke else 20,
            cfg=7.5,
            scheduler="normal",
        ),
        records,
    )
    if cascade is not None:
        shape_inputs = measure(
            f"{prefix}:upsample-shape",
            lambda: native_arm.GenerationTrellis2UpsampleStage.execute(
                positive=shape_inputs["positive"],
                negative=shape_inputs["negative"],
                shape_latent=shape,
                vae=shape_vae,
                target_resolution=cascade,
            ),
            records,
        )
        shape = measure(
            f"{prefix}:sample-shape-{cascade}",
            lambda: sample(
                shape_model,
                shape_inputs["positive"],
                shape_inputs["negative"],
                shape_inputs["latent"],
                seed=42,
                steps=1 if smoke else 12,
                cfg=7.5,
                scheduler="simple",
            ),
            records,
        )
    decoded_shape = measure(
        f"{prefix}:decode-shape",
        lambda: native_arm.GenerationVaeDecodeShapeTrellis.execute(
            samples=shape,
            vae=shape_vae,
        ),
        records,
    )
    texture_inputs = native_arm.GenerationTrellis2TextureStage.execute(
        positive=shape_inputs["positive"],
        negative=shape_inputs["negative"],
        shape_latent=shape,
    )
    texture = measure(
        f"{prefix}:sample-texture",
        lambda: sample(
            model,
            texture_inputs["positive"],
            texture_inputs["negative"],
            texture_inputs["latent"],
            seed=43,
            steps=1 if smoke else 12,
            cfg=1.0,
            scheduler="normal",
        ),
        records,
    )
    voxel_colors = measure(
        f"{prefix}:decode-texture",
        lambda: native_arm.GenerationVaeDecodeTextureTrellis.execute(
            samples=texture,
            vae=texture_vae,
            shape_subdivides=decoded_shape["shape_subdivides"],
        )["voxel_colors"],
        records,
    )
    mesh = decoded_shape["mesh"]
    shape_sparse = shape["samples"]
    texture_sparse = texture["samples"]
    conditioning_storage = positive.payload._base
    vertex_count = (
        int(mesh.vertex_counts[0]) if mesh.vertex_counts is not None else mesh.vertices.shape[1]
    )
    face_count = int(mesh.face_counts[0]) if mesh.face_counts is not None else mesh.faces.shape[1]
    result = {
        "run": run,
        "mesh_vertices": mesh.vertex_counts.tolist()
        if mesh.vertex_counts is not None
        else [mesh.vertices.shape[1]],
        "mesh_faces": mesh.face_counts.tolist()
        if mesh.face_counts is not None
        else [mesh.faces.shape[1]],
        "texture_points": voxel_colors.support.point_count,
        "shape_input_points": shape["samples"].support.point_count,
        "subdivision_points": [
            level.support.point_count for level in decoded_shape["shape_subdivides"].levels
        ],
        "tensors": {
            "conditioning_512": tensor_summary(conditioning_storage._global_512),
            "conditioning_1024": tensor_summary(conditioning_storage._global_1024),
            "structure": tensor_summary(structure["samples"]),
            "occupancy": tensor_summary(voxel.values),
            "shape_coordinates": tensor_summary(shape_sparse.support.coordinates),
            "shape_features": tensor_summary(shape_sparse.features),
            "mesh_vertices": tensor_summary(mesh.vertices[0, :vertex_count]),
            "mesh_faces": tensor_summary(mesh.faces[0, :face_count]),
            "texture_coordinates": tensor_summary(texture_sparse.support.coordinates),
            "texture_features": tensor_summary(texture_sparse.features),
            "pbr_coordinates": tensor_summary(voxel_colors.support.coordinates),
            "pbr_features": tensor_summary(voxel_colors.features),
        },
    }
    tensors = None
    if capture_tensors:
        tensors = {
            "conditioning_512": canonical(conditioning_storage._global_512),
            "conditioning_1024": canonical(conditioning_storage._global_1024),
            "structure": canonical(structure["samples"]),
            "occupancy": canonical(voxel.values),
            "shape_coordinates": canonical(shape_sparse.support.coordinates),
            "shape_features": canonical(shape_sparse.features),
            "mesh_vertices": canonical(mesh.vertices[0, :vertex_count]),
            "mesh_faces": canonical(mesh.faces[0, :face_count]),
            "texture_coordinates": canonical(texture_sparse.support.coordinates),
            "texture_features": canonical(texture_sparse.features),
            "pbr_coordinates": canonical(voxel_colors.support.coordinates),
            "pbr_features": canonical(voxel_colors.features),
        }
    return result, tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("/home/kosin/model-artifacts/dinkster-trellis2-1083"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--condition-image", type=Path)
    parser.add_argument("--tensors-output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pixal3d", action="store_true")
    parser.add_argument("--direct", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--cascade", type=int, choices=(1024, 1536))
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.cascade is not None and args.direct != 512:
        parser.error("--cascade starts from the direct 512 stage")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    os.environ["DINKSTER_AIMDO_ARM"] = "off"
    assets = verify_artifacts(args.artifacts)
    image_path = args.image or args.artifacts / "viking_wolf_rune_axe.png"
    load_official_image(image_path)
    image = (
        load_condition_image(args.condition_image)
        if args.condition_image is not None
        else load_official_image(image_path)
    )
    records: list[dict[str, Any]] = []
    baseline = {
        "gpu_allocated": torch.cuda.memory_allocated(),
        "gpu_reserved": torch.cuda.memory_reserved(),
        "host_rss": rss_bytes(),
        "host_peak_rss": peak_rss_bytes(),
    }
    model_name = "pixal3d" if args.pixal3d else "trellis2"
    model = measure(
        "load-model",
        lambda: native_arm._build_trellis2_model_handle(
            assets[model_name],
            identity(args.artifacts, assets[model_name], "diffusion", BFLOAT16),
            torch,
            compute_dtype="bfloat16",
        ),
        records,
    )
    vision = measure(
        "load-vision",
        lambda: native_arm._build_trellis2_component_handle(
            assets["vision"],
            "vision",
            identity(args.artifacts, assets["vision"], "vision", FLOAT32),
            torch,
            compute_dtype="float32",
        ),
        records,
    )
    shape_vae = measure(
        "load-shape-vae",
        lambda: native_arm._build_trellis2_component_handle(
            assets["shape"],
            "shape-decoder",
            identity(args.artifacts, assets["shape"], "shape-decoder", BFLOAT16),
            torch,
            compute_dtype="bfloat16",
        ),
        records,
    )
    texture_vae = measure(
        "load-texture-vae",
        lambda: native_arm._build_trellis2_component_handle(
            assets["texture"],
            "texture-decoder",
            identity(args.artifacts, assets["texture"], "texture-decoder", BFLOAT16),
            torch,
            compute_dtype="bfloat16",
        ),
        records,
    )
    runs: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] | None = None
    for run in range(1, args.repeats + 1):
        run_summary, run_tensors = run_profile(
            run,
            model=model,
            vision=vision,
            shape_vae=shape_vae,
            texture_vae=texture_vae,
            pixal3d=args.pixal3d,
            direct=args.direct,
            cascade=args.cascade,
            smoke=args.smoke,
            image=image,
            records=records,
            capture_tensors=args.tensors_output is not None and run == args.repeats,
        )
        runs.append(run_summary)
        if run_tensors is not None:
            tensors = run_tensors
        trim_runtime(f"run-{run}:cleanup", records)
    summary = {
        "profile": "pixal3d" if args.pixal3d else "trellis2",
        "smoke": args.smoke,
        "direct": args.direct,
        "cascade": args.cascade,
        "repeats": args.repeats,
        "input": {
            "path": str(image_path),
            "url": _INPUT_URL,
            "bytes": _INPUT_BYTES,
            "sha256": _INPUT_SHA256,
        },
        "condition_image": None
        if args.condition_image is None
        else {
            "path": str(args.condition_image),
            "bytes": args.condition_image.stat().st_size,
            "sha256": sha256(args.condition_image),
        },
        "baseline": baseline,
        "runs": runs,
        "records": records,
    }
    if args.tensors_output is not None:
        if tensors is None:
            raise RuntimeError("tensor capture did not produce an output")
        args.tensors_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, args.tensors_output)
        summary["tensors_output"] = {
            "path": str(args.tensors_output),
            "bytes": args.tensors_output.stat().st_size,
            "sha256": sha256(args.tensors_output),
        }
    for handle in (texture_vae, shape_vae, vision, model):
        handle.terminal_release()
    del texture_vae, shape_vae, vision, model, runs, records
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    summary["cleanup"] = {
        "gpu_allocated": torch.cuda.memory_allocated(),
        "gpu_reserved": torch.cuda.memory_reserved(),
        "host_rss": rss_bytes(),
        "host_peak_rss": peak_rss_bytes(),
        "gpu_allocated_delta": torch.cuda.memory_allocated() - baseline["gpu_allocated"],
        "gpu_reserved_delta": torch.cuda.memory_reserved() - baseline["gpu_reserved"],
        "host_rss_delta": rss_bytes() - baseline["host_rss"],
    }
    output = args.output
    if output is None:
        profile = "pixal3d" if args.pixal3d else "trellis2"
        resolution = f"cascade-{args.cascade}" if args.cascade else f"direct-{args.direct}"
        mode = "smoke" if args.smoke else "official"
        output = args.artifacts / "evidence" / f"dinkster-{profile}-{resolution}-{mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
