"""Generate crop/mesh utility vectors from immutable ComfyUI source, never native code.

Run twice with Python 3.12.3, torch 2.13 CPU, numpy 2.5.1 and Pillow 12.0.0:
    .venv-torch/bin/python tools/gen_crop_mesh_goldens.py --comfyui PATH

Only the named source bodies are compiled. UI envelopes and intermediate placement
are inert CPU stand-ins; crop/resize math and mesh serialization are unchanged.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import struct
import subprocess
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

REFERENCE = "25dfc16f9ac0a87991d34fbf5f02d6c25c844639"
OUT = Path(__file__).resolve().parents[1] / (
    "packages/dinkster-inference-torch/tests/goldens/crop_mesh_25dfc16f.json"
)


def crop_cases() -> dict[str, dict[str, Any]]:
    image = (torch.arange(2 * 9 * 13 * 4).reshape(2, 9, 13, 4) % 271 - 8).float() / 255
    mask = torch.zeros(1, 9, 13)
    mask[:, 2:7, 3:10] = 1
    cases: dict[str, dict[str, Any]] = {}
    for name, masks in (
        ("center", mask),
        ("inverted", 1 - mask),
        ("zero", torch.zeros_like(mask)),
        ("full", torch.ones_like(mask)),
        ("soft", torch.full_like(mask, 0.5)),
        ("edge", torch.nn.functional.pad(torch.ones(1, 4, 3), (0, 10, 0, 5))),
        ("threshold", torch.linspace(0, 1, 117).reshape(1, 9, 13)),
        ("resize_mask", mask[:, ::2, ::2]),
        ("per_item", torch.cat((mask, 1 - mask))),
    ):
        for width, height in ((8, 6), (16, 24)):
            for grow in (-1, 0, 2):
                cases[f"{name}:{width}x{height}:grow{grow}"] = dict(
                    images=image.clone(),
                    masks=masks.clone(),
                    width=width,
                    height=height,
                    pad_factor=1.3,
                    grow_mask=grow,
                    background="#1763a9",
                )
    for dtype in (torch.float16, torch.float64):
        cases[str(dtype)] = dict(
            images=image.to(dtype),
            masks=mask.to(dtype),
            width=16,
            height=8,
            pad_factor=1.0,
            grow_mask=0,
            background="invalid",
        )
    cases["batch_error"] = dict(
        images=image,
        masks=mask.expand(3, -1, -1),
        width=8,
        height=8,
        pad_factor=1.0,
        grow_mask=0,
        background="#000000",
    )
    return cases


def mesh_cases() -> dict[str, dict[str, Any]]:
    base: dict[str, Any] = dict(
        vertices=torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [9.0, 9.0, 9.0]]]
        ),
        faces=torch.tensor([[[0, 1, 2], [0, 2, 1]]]),
        uvs=None,
        vertex_colors=None,
        normals=None,
        tangents=None,
        texture=None,
        metallic_roughness=None,
        normal_map=None,
        emissive=None,
        unlit=False,
        occlusion_in_mr=False,
        material=None,
        vertex_counts=None,
        face_counts=None,
    )
    cases: dict[str, dict[str, Any]] = {"bare": base}
    colors = torch.linspace(-0.2, 1.2, 16).reshape(1, 4, 4)
    texture = torch.linspace(-0.1, 1.1, 18).reshape(1, 2, 3, 3)
    cases["rgb"] = dict(base, vertex_colors=colors[..., :3])
    cases["rgba_unlit"] = dict(base, vertex_colors=colors, unlit=True)
    rich = dict(
        base,
        uvs=torch.arange(8).reshape(1, 4, 2).float() / 8,
        vertex_colors=colors,
        normals=torch.ones(1, 4, 3),
        tangents=torch.ones(1, 4, 4),
        texture=texture,
        metallic_roughness=texture.flip(1),
        normal_map=texture.flip(2),
        emissive=texture,
        occlusion_in_mr=True,
    )
    cases["all_attributes"] = rich
    cases["textured_unlit_uses_pbr"] = dict(rich, unlit=True)
    cases["maps_without_uv"] = dict(rich, uvs=None)
    cases["unlit_ignores_pbr"] = dict(rich, texture=None, unlit=True)
    cases["material"] = dict(
        rich,
        material={
            "base_color_factor": [0.1, 0.2, 0.3, 0.4],
            "metallic_factor": 0.7,
            "roughness_factor": 0.2,
            "double_sided": False,
            "normal_scale": 0.5,
            "occlusion_strength": 0.4,
            "emissive_factor": [1.5, 0.2, 0.3],
            "emissive_strength": 3.0,
        },
    )
    cases["emissive_no_texture"] = dict(
        base,
        material={
            "emissive_factor": [0.1, 0.2, 0.3],
            "emissive_strength": 2.0,
        },
    )
    cases["variable"] = dict(rich, vertex_counts=torch.tensor([3]), face_counts=torch.tensor([1]))
    cases["batch"] = {
        key: value.expand(2, *value.shape[1:]).clone() if isinstance(value, torch.Tensor) else value
        for key, value in rich.items()
    }
    cases["batch"].update(vertex_counts=torch.tensor([3, 4]), face_counts=torch.tensor([1, 2]))
    cases["empty_vertices"] = dict(base, vertices=torch.empty(1, 0, 3))
    cases["empty_faces"] = dict(base, faces=torch.empty(1, 0, 3, dtype=torch.int64))
    cases["empty_count"] = dict(
        base, vertex_counts=torch.tensor([0]), face_counts=torch.tensor([0])
    )
    cases["bad_face"] = dict(base, faces=torch.tensor([[[0, -1, 2]]]))
    cases["bad_uvs"] = dict(base, uvs=torch.zeros(1, 2, 2))
    cases["bad_tangents"] = dict(base, tangents=torch.zeros(1, 4, 3))
    cases["oversized_count_and_uvs"] = dict(
        base,
        vertex_counts=torch.tensor([6]),
        face_counts=torch.tensor([2]),
        uvs=torch.zeros(1, 5, 2),
    )
    cases["negative_count_slice"] = dict(
        rich, vertex_counts=torch.tensor([-1]), face_counts=torch.tensor([-1])
    )
    return cases


def digest(value: torch.Tensor | bytes | None) -> dict[str, Any]:
    if value is None:
        return {"empty": True}
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest(),
        }
    return {"length": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def reference_functions(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    ns: dict[str, Any] = dict(
        torch=torch,
        np=np,
        Image=Image,
        math=math,
        logging=logging,
        json=json,
        struct=struct,
        BytesIO=BytesIO,
    )
    ns["IO"] = SimpleNamespace(ComfyNode=object, NodeOutput=lambda *x, **kw: x)
    ns["UI"] = SimpleNamespace(PreviewText=lambda x: x)
    sources = {
        "comfy/utils.py": {"common_upscale", "lanczos"},
        "comfy_extras/nodes_images.py": {"_crop_image_with_mask", "ImageCropToMask"},
        "comfy_extras/nodes_save_3d.py": {
            "get_mesh_batch_item",
            "save_glb",
            "mesh_item_to_glb_bytes",
            "GetMeshInfo",
        },
    }
    hashes: dict[str, str] = {}
    for path, names in sources.items():
        source = subprocess.run(
            ["git", "-C", str(root), "show", f"{REFERENCE}:{path}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        hashes[path] = hashlib.sha256(source.encode()).hexdigest()
        definitions = [
            node
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
        ]
        if {node.name for node in definitions} != names:
            raise RuntimeError(f"Incomplete reference bodies: {path}")
        module = ast.Module(body=cast(list[ast.stmt], definitions), type_ignores=[])
        exec(compile(module, path, "exec"), ns)
    ns["comfy"] = SimpleNamespace(
        utils=SimpleNamespace(common_upscale=ns["common_upscale"]),
        model_management=SimpleNamespace(
            intermediate_device=lambda: "cpu", intermediate_dtype=lambda: torch.float32
        ),
    )
    ns["GetMeshInfo"].hidden = SimpleNamespace(unique_id=None)
    return ns, hashes


def generate(root: Path) -> dict[str, Any]:
    ns, hashes = reference_functions(root)
    crops: dict[str, Any] = {}
    meshes: dict[str, Any] = {}
    for name, kwargs in crop_cases().items():
        try:
            crops[name] = digest(ns["ImageCropToMask"].execute(**kwargs)[0])
        except ValueError as error:
            crops[name] = {"error": str(error)}
    for name, kwargs in mesh_cases().items():
        mesh = SimpleNamespace(**kwargs)
        outputs = []
        for index in range(mesh.vertices.shape[0]):
            try:
                outputs.append(digest(ns["mesh_item_to_glb_bytes"](mesh, index)))
            except ValueError as error:
                outputs.append({"error": str(error)})
        meshes[name] = {"items": outputs, "info": ns["GetMeshInfo"].execute(mesh)[1]}
    return {
        "reference": {"commit": REFERENCE, "source_sha256": hashes},
        "runtime": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pillow": Image.__version__,
        },
        "crops": crops,
        "meshes": meshes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    payload = (json.dumps(generate(args.comfyui), indent=2, sort_keys=True) + "\n").encode()
    args.output.write_bytes(payload)
    print(f"{hashlib.sha256(payload).hexdigest()}  {args.output}")


if __name__ == "__main__":
    main()
