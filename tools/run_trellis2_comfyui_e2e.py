#!/usr/bin/env python3
"""Run the pinned ComfyUI TRELLIS.2 workflow as an authoritative reference."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib
import json
import resource
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch  # pyright: ignore[reportMissingImports]
from PIL import Image

COMFYUI_REVISION = "8a33128f2f8c5585c57486c07de481241e70a39c"
WORKFLOW_REVISION = "d3b4a9e89573162b005961865164c18c8ae2206b"
COMFY_ARTIFACT_REVISION = "463441b1c32829ee876e4f297dcfff533cb357a7"
PIXAL_ARTIFACT_REVISION = "e69bd0b6c7b959661a87051187aab18e8e5abcdf"
INPUT_BYTES = 1_106_832
INPUT_SHA256 = "aaf42f16e262e88bef7c947771a745e817cc0b533922a8388b51165ede091fce"
INPUT_URL = (
    "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/"
    f"{WORKFLOW_REVISION}/input/viking_wolf_rune_axe.png"
)


@dataclass(frozen=True)
class Artifact:
    filename: str
    byte_size: int
    sha256: str
    url: str


ARTIFACTS = {
    "trellis2": Artifact(
        "trellis_2_int8_convrot.safetensors",
        5_253_048_192,
        "d01952ad137213f6a868f86b6b877026276f84af5eec23069217475a0bad3a31",
        "https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/"
        f"{COMFY_ARTIFACT_REVISION}/diffusion_models/trellis_2_int8_convrot.safetensors",
    ),
    "pixal3d": Artifact(
        "pixal3d_int8_convrot.safetensors",
        5_584_555_824,
        "4621eac3b715484f79303c7152af641fe0b2b14f4d0e3d394fd6922d00f955ec",
        "https://huggingface.co/Comfy-Org/Pixal3D/resolve/"
        f"{PIXAL_ARTIFACT_REVISION}/diffusion_models/pixal3d_int8_convrot.safetensors",
    ),
    "vision": Artifact(
        "dino_v3_L_naf_fp32.safetensors",
        1_215_214_176,
        "4ad2ec4e0879a5b5b04cd97325cc37da954a7b6edca5170b86510f17f2b2290f",
        "https://huggingface.co/Comfy-Org/Pixal3D/resolve/"
        f"{PIXAL_ARTIFACT_REVISION}/clip_vision/dino_v3_L_naf_fp32.safetensors",
    ),
    "shape": Artifact(
        "trellis_2_shape_vae_bf16.safetensors",
        1_095_844_024,
        "de0cb4949a76c59ee5c091a995a69bcc8c51d5aeda939f0c641a50d2a72341f4",
        "https://huggingface.co/Comfy-Org/Pixal3D/resolve/"
        f"{PIXAL_ARTIFACT_REVISION}/vae/trellis_2_shape_vae_bf16.safetensors",
    ),
    "texture": Artifact(
        "trellis_2_texture_vae_bf16.safetensors",
        948_461_364,
        "714e5ebf094a610e12a8e3b5175c18a62f37f6ea4218acb6073644456b73ab0e",
        "https://huggingface.co/Comfy-Org/Pixal3D/resolve/"
        f"{PIXAL_ARTIFACT_REVISION}/vae/trellis_2_texture_vae_bf16.safetensors",
    ),
    "background": Artifact(
        "birefnet.safetensors",
        444_473_596,
        "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154",
        "https://huggingface.co/Comfy-Org/BiRefNet/resolve/"
        "35767b272f2846752a3aee1259abdd4586f735c8/background_removal/birefnet.safetensors",
    ),
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_file(path: Path, byte_size: int, digest: str, name: str) -> None:
    if path.stat().st_size != byte_size:
        raise RuntimeError(f"{name} byte size differs from its immutable pin")
    if sha256(path) != digest:
        raise RuntimeError(f"{name} sha256 differs from its immutable pin")


def verify_revision(comfyui: Path) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=comfyui, text=True).strip()
    if head != COMFYUI_REVISION:
        raise RuntimeError(f"ComfyUI must be checked out at {COMFYUI_REVISION}, got {head}")
    status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"], cwd=comfyui, text=True
    )
    if status:
        raise RuntimeError("ComfyUI tracked files must be clean for authoritative execution")


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


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu()
    if tensor.is_floating_point():
        tensor = tensor.float()
    tensor = tensor.contiguous()
    values = tensor.double()
    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
    }
    if tensor.numel():
        result.update(
            min=float(values.min()),
            max=float(values.max()),
            mean=float(values.mean()),
            std=float(values.std(correction=0)),
        )
    return result


def load_image(path: Path) -> torch.Tensor:
    verify_file(path, INPUT_BYTES, INPUT_SHA256, "workflow input")
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(pixels).unsqueeze(0)


def import_comfyui(comfyui: Path, artifacts: Path) -> dict[str, Any]:
    sys.path.insert(0, str(comfyui))
    server = types.ModuleType("server")
    server.__dict__["PromptServer"] = type("PromptServer", (), {"instance": None})
    sys.modules["server"] = server
    folder_paths = importlib.import_module("folder_paths")
    nodes = importlib.import_module("nodes")
    model_management = importlib.import_module("comfy.model_management")
    background = importlib.import_module("comfy_extras.nodes_bg_removal")
    custom_sampler = importlib.import_module("comfy_extras.nodes_custom_sampler")
    images = importlib.import_module("comfy_extras.nodes_images")
    advanced = importlib.import_module("comfy_extras.nodes_model_advanced")
    trellis2 = importlib.import_module("comfy_extras.nodes_trellis2")

    model_root = str(artifacts / "comfy")
    for category in ("diffusion_models", "clip_vision", "vae", "background_removal"):
        folder_paths.add_model_folder_path(category, model_root, is_default=True)
    return {
        "nodes": nodes,
        "model_management": model_management,
        "LoadBackgroundRemovalModel": background.LoadBackgroundRemovalModel,
        "RemoveBackground": background.RemoveBackground,
        "CFGOverride": custom_sampler.CFGOverride,
        "ImageCropToMask": images.ImageCropToMask,
        "ModelSamplingSD3": advanced.ModelSamplingSD3,
        "RescaleCFG": advanced.RescaleCFG,
        "EmptyTrellis2LatentStructure": trellis2.EmptyTrellis2LatentStructure,
        "Pixal3DConditioning": trellis2.Pixal3DConditioning,
        "Trellis2Conditioning": trellis2.Trellis2Conditioning,
        "Trellis2ShapeStage": trellis2.Trellis2ShapeStage,
        "Trellis2TextureStage": trellis2.Trellis2TextureStage,
        "Trellis2UpsampleStage": trellis2.Trellis2UpsampleStage,
        "VaeDecodeShapeTrellis": trellis2.VaeDecodeShapeTrellis,
        "VaeDecodeStructureTrellis2": trellis2.VaeDecodeStructureTrellis2,
        "VaeDecodeTextureTrellis": trellis2.VaeDecodeTextureTrellis,
    }


def prepare_condition_image(
    source: dict[str, Any], image: torch.Tensor, output: Path, records: list[dict[str, Any]]
) -> None:
    background = measure(
        "load-background-removal",
        lambda: source["LoadBackgroundRemovalModel"].execute("birefnet.safetensors")[0],
        records,
    )
    mask = measure(
        "remove-background",
        lambda: source["RemoveBackground"].execute(background, image)[0],
        records,
    )
    condition_image = source["ImageCropToMask"].execute(image, mask, 1024, 1024, 1.1, 0, "#000000")[
        0
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, condition_image.detach().cpu().float().numpy(), allow_pickle=False)
    print(
        json.dumps(
            {
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
                "tensor": tensor_summary(condition_image),
            },
            indent=2,
        )
    )


def load_condition_image(path: Path) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32 or array.shape != (1, 1024, 1024, 3):
        raise RuntimeError(f"condition image must have float32 shape [1,1024,1024,3], got {array}")
    return torch.from_numpy(array.copy())


def patch_model(source: dict[str, Any], model: object, shift: float | None) -> object:
    value = model
    if shift is not None:
        value = source["ModelSamplingSD3"]().patch(value, shift)[0]
    value = source["CFGOverride"].execute(value, 1.0, 0.667 if shift else 0.769, 1.0)[0]
    return source["RescaleCFG"]().patch(value, 0.7 if shift else 0.5)[0]


def sample(
    source: dict[str, Any],
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
    return (
        source["nodes"]
        .KSampler()
        .sample(
            model,
            seed,
            steps,
            cfg,
            "euler",
            scheduler,
            positive,
            negative,
            latent,
            1.0,
        )[0]
    )


def canonical(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().cpu()
    return value.float().contiguous() if value.is_floating_point() else value.contiguous()


def run_workflow(
    source: dict[str, Any],
    *,
    pixal3d: bool,
    cascade: int,
    smoke: bool,
    condition_image: torch.Tensor,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model_name = (
        "pixal3d_int8_convrot.safetensors" if pixal3d else "trellis_2_int8_convrot.safetensors"
    )
    model = measure(
        "load-model",
        lambda: source["nodes"].UNETLoader().load_unet(model_name, "default")[0],
        records,
    )
    vision = measure(
        "load-vision",
        lambda: source["nodes"].CLIPVisionLoader().load_clip("dino_v3_L_naf_fp32.safetensors")[0],
        records,
    )
    shape_vae = measure(
        "load-shape-vae",
        lambda: source["nodes"].VAELoader().load_vae("trellis_2_shape_vae_bf16.safetensors")[0],
        records,
    )
    texture_vae = measure(
        "load-texture-vae",
        lambda: source["nodes"].VAELoader().load_vae("trellis_2_texture_vae_bf16.safetensors")[0],
        records,
    )

    condition = measure(
        "condition",
        lambda: (
            source["Pixal3DConditioning"].execute(vision, condition_image, 49.13)
            if pixal3d
            else source["Trellis2Conditioning"].execute(vision, condition_image)
        ),
        records,
    )
    positive, negative = condition[0], condition[1]
    empty = source["EmptyTrellis2LatentStructure"].execute(1)[0]
    structure_model = patch_model(source, model, 5.0)
    structure = measure(
        "sample-structure",
        lambda: sample(
            source,
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
        "decode-structure",
        lambda: source["VaeDecodeStructureTrellis2"].execute(structure, shape_vae, "32")[0],
        records,
    )
    shape_inputs = source["Trellis2ShapeStage"].execute(positive, negative, voxel)
    shape_model = patch_model(source, model, None)
    shape = measure(
        "sample-shape-512",
        lambda: sample(
            source,
            shape_model,
            shape_inputs[0],
            shape_inputs[1],
            shape_inputs[2],
            seed=42,
            steps=1 if smoke else 20,
            cfg=7.5,
            scheduler="normal",
        ),
        records,
    )
    shape_inputs = measure(
        "upsample-shape",
        lambda: source["Trellis2UpsampleStage"].execute(
            shape_inputs[0], shape_inputs[1], shape, shape_vae, cascade
        ),
        records,
    )
    shape = measure(
        f"sample-shape-{cascade}",
        lambda: sample(
            source,
            shape_model,
            shape_inputs[0],
            shape_inputs[1],
            shape_inputs[2],
            seed=42,
            steps=1 if smoke else 12,
            cfg=7.5,
            scheduler="simple",
        ),
        records,
    )
    decoded_shape = measure(
        "decode-shape",
        lambda: source["VaeDecodeShapeTrellis"].execute(shape, shape_vae),
        records,
    )
    texture_inputs = source["Trellis2TextureStage"].execute(shape_inputs[0], shape_inputs[1], shape)
    texture = measure(
        "sample-texture",
        lambda: sample(
            source,
            model,
            texture_inputs[0],
            texture_inputs[1],
            texture_inputs[2],
            seed=43,
            steps=1 if smoke else 12,
            cfg=1.0,
            scheduler="normal",
        ),
        records,
    )
    voxel_colors = measure(
        "decode-texture",
        lambda: source["VaeDecodeTextureTrellis"].execute(texture, texture_vae, decoded_shape[1])[
            0
        ],
        records,
    )

    mesh = decoded_shape[0]
    vertex_count = (
        int(mesh.vertex_counts[0]) if mesh.vertex_counts is not None else mesh.vertices.shape[1]
    )
    face_count = int(mesh.face_counts[0]) if mesh.face_counts is not None else mesh.faces.shape[1]
    shape_features = shape["samples"].squeeze(-1).transpose(1, 2).reshape(-1, 32)
    texture_features = texture["samples"].squeeze(-1).transpose(1, 2).reshape(-1, 32)
    tensors = {
        "conditioning_512": canonical(positive[0][0]),
        "conditioning_1024": canonical(positive[0][1]["embeds"]),
        "structure": canonical(structure["samples"]),
        "occupancy": canonical(voxel.data),
        "shape_coordinates": canonical(shape["coords"]),
        "shape_features": canonical(shape_features),
        "mesh_vertices": canonical(mesh.vertices[0, :vertex_count]),
        "mesh_faces": canonical(mesh.faces[0, :face_count]),
        "texture_coordinates": canonical(texture["coords"]),
        "texture_features": canonical(texture_features),
        "pbr_coordinates": canonical(voxel_colors.data),
        "pbr_features": canonical(voxel_colors.voxel_colors),
    }
    summary = {
        "mesh_vertices": vertex_count,
        "mesh_faces": face_count,
        "texture_points": int(voxel_colors.data.shape[0]),
        "shape_input_points": int(shape["coords"].shape[0]),
        "subdivision_points": [int(level.feats.shape[0]) for level in decoded_shape[1]],
        "tensors": {name: tensor_summary(tensor) for name, tensor in tensors.items()},
    }
    source["model_management"].unload_all_models()
    source["model_management"].soft_empty_cache()
    return summary, tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--condition-image", type=Path)
    parser.add_argument("--prepare-condition-image", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensors-output", type=Path)
    parser.add_argument("--pixal3d", action="store_true")
    parser.add_argument("--cascade", type=int, choices=(1024, 1536), default=1536)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    verify_revision(args.comfyui)
    for name, artifact in ARTIFACTS.items():
        verify_file(
            args.artifacts / "comfy" / artifact.filename,
            artifact.byte_size,
            artifact.sha256,
            name,
        )
    image_path = args.image or args.artifacts / "viking_wolf_rune_axe.png"
    image = load_image(image_path)
    records: list[dict[str, Any]] = []
    baseline = {
        "gpu_allocated": torch.cuda.memory_allocated(),
        "gpu_reserved": torch.cuda.memory_reserved(),
        "host_rss": rss_bytes(),
        "host_peak_rss": peak_rss_bytes(),
    }
    source = import_comfyui(args.comfyui, args.artifacts)
    if args.prepare_condition_image is not None:
        with torch.inference_mode():
            prepare_condition_image(source, image, args.prepare_condition_image, records)
        return
    if args.condition_image is None:
        parser.error("--condition-image is required unless --prepare-condition-image is used")
    condition_image = load_condition_image(args.condition_image)
    with torch.inference_mode():
        summary, tensors = run_workflow(
            source,
            pixal3d=args.pixal3d,
            cascade=args.cascade,
            smoke=args.smoke,
            condition_image=condition_image,
            records=records,
        )
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result = {
        "reference": "ComfyUI",
        "comfyui_revision": COMFYUI_REVISION,
        "profile": "pixal3d" if args.pixal3d else "trellis2",
        "smoke": args.smoke,
        "cascade": args.cascade,
        "input": {
            "path": str(image_path),
            "url": INPUT_URL,
            "bytes": INPUT_BYTES,
            "sha256": INPUT_SHA256,
        },
        "condition_image": {
            "path": str(args.condition_image),
            "bytes": args.condition_image.stat().st_size,
            "sha256": sha256(args.condition_image),
        },
        "artifacts": {
            name: {
                "url": artifact.url,
                "bytes": artifact.byte_size,
                "sha256": artifact.sha256,
            }
            for name, artifact in ARTIFACTS.items()
        },
        "baseline": baseline,
        "run": summary,
        "records": records,
        "cleanup": {
            "gpu_allocated": torch.cuda.memory_allocated(),
            "gpu_reserved": torch.cuda.memory_reserved(),
            "host_rss": rss_bytes(),
            "host_peak_rss": peak_rss_bytes(),
            "gpu_allocated_delta": torch.cuda.memory_allocated() - baseline["gpu_allocated"],
            "gpu_reserved_delta": torch.cuda.memory_reserved() - baseline["gpu_reserved"],
            "host_rss_delta": rss_bytes() - baseline["host_rss"],
        },
    }
    if args.tensors_output is not None:
        args.tensors_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, args.tensors_output)
        result["tensors_output"] = {
            "path": str(args.tensors_output),
            "bytes": args.tensors_output.stat().st_size,
            "sha256": sha256(args.tensors_output),
        }
    output = args.output or args.artifacts / "evidence" / (
        f"comfyui-{'pixal3d' if args.pixal3d else 'trellis2'}-cascade-{args.cascade}-"
        f"{'smoke' if args.smoke else 'official'}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
