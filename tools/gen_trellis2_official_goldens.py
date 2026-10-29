#!/usr/bin/env python3
"""Generate official-weight TRELLIS.2 goldens from the pinned Microsoft source."""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import importlib
import json
import os
import subprocess
import sys
import types
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file

REFERENCE_COMMIT = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
GENERATOR_TORCH = "2.6.0+cu124"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "TRELLIS.2"
DEFAULT_ARTIFACTS = Path("/home/kosin/model-artifacts/dinkster-trellis2-1083/source")
DEFAULT_OUTPUT = (
    ROOT
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "trellis2_official_goldens.json"
)


@dataclass(frozen=True)
class Artifact:
    relative: str
    byte_size: int
    sha256: str
    url: str


_MICROSOFT_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
_TRELLIS1_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"
ARTIFACTS = {
    "structure-flow": Artifact(
        "ss_flow_img_dit_1_3B_64_bf16.safetensors",
        2_584_426_920,
        "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6",
        "https://huggingface.co/microsoft/TRELLIS.2-4B/resolve/"
        f"{_MICROSOFT_REVISION}/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
    ),
    "shape-512": Artifact(
        "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        2_584_574_424,
        "ec5e0917ef9b7e25ad51dffc7d19687a42019871f94239f2fa7f86264c55b70f",
        "https://huggingface.co/microsoft/TRELLIS.2-4B/resolve/"
        f"{_MICROSOFT_REVISION}/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    ),
    "shape-1024": Artifact(
        "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        2_584_574_424,
        "07cd0596f634c5adc1890023d16023afc5eed02fb84b22bb23aff5bf0030fbbd",
        "https://huggingface.co/microsoft/TRELLIS.2-4B/resolve/"
        f"{_MICROSOFT_REVISION}/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    ),
    "texture-512": Artifact(
        "slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
        2_584_672_728,
        "8371aa1c5d13be79dcd5ddfd2cf3835e902e204dc34427169a1c702828e1a94d",
        "https://huggingface.co/microsoft/TRELLIS.2-4B/resolve/"
        f"{_MICROSOFT_REVISION}/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
    ),
    "texture-1024": Artifact(
        "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
        2_584_672_728,
        "580401269059a339b8318ab9ced459a13ba63391721c83a6c383198c29e77686",
        "https://huggingface.co/microsoft/TRELLIS.2-4B/resolve/"
        f"{_MICROSOFT_REVISION}/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
    ),
    "structure-decoder": Artifact(
        "ss_dec_conv3d_16l8_fp16.safetensors",
        147_591_972,
        "1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a",
        "https://huggingface.co/microsoft/TRELLIS-image-large/resolve/"
        f"{_TRELLIS1_REVISION}/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    ),
    "shape-decoder": Artifact(
        "shape_dec_next_dc_f16c32_fp16.safetensors",
        948_490_494,
        "e3b718d3e43e4f8780e9a24ac6fff231811a67e3b058e336e10fe654c911d581",
        "https://huggingface.co/microsoft/TRELLIS.2-4B/resolve/"
        f"{_MICROSOFT_REVISION}/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
    ),
    "texture-decoder": Artifact(
        "tex_dec_next_dc_f16c32_fp16.safetensors",
        948_458_812,
        "97ea69addea2ecd9312910f5f548234665eef51c088386180b7cd5b258645e3c",
        "https://huggingface.co/microsoft/TRELLIS.2-4B/resolve/"
        f"{_MICROSOFT_REVISION}/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
    ),
}


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _verify_artifacts(root: Path) -> dict[str, dict[str, object]]:
    facts: dict[str, dict[str, object]] = {}
    for role, artifact in ARTIFACTS.items():
        path = root / artifact.relative
        if path.stat().st_size != artifact.byte_size or _digest(path) != artifact.sha256:
            raise RuntimeError(f"{role} does not match its immutable artifact pin")
        facts[role] = {
            "byte_size": artifact.byte_size,
            "sha256": artifact.sha256,
            "url": artifact.url,
        }
    return facts


def _require_source(source: Path) -> None:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != REFERENCE_COMMIT:
        raise RuntimeError(f"TRELLIS.2 source is at {revision}, expected {REFERENCE_COMMIT}")
    if torch.__version__ != GENERATOR_TORCH:
        raise RuntimeError(f"generator requires torch {GENERATOR_TORCH}, found {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("official TRELLIS.2 golden generation requires CUDA")


def _tensor(value: torch.Tensor) -> dict[str, object]:
    value = value.detach().contiguous().cpu()
    raw = value.numpy().tobytes()
    return {
        "data": base64.b64encode(zlib.compress(raw, level=9)).decode("ascii"),
        "dtype": str(value.dtype).removeprefix("torch."),
        "encoding": "base64+zlib",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(value.shape),
    }


def _release(value: object) -> None:
    del value
    gc.collect()
    torch.cuda.empty_cache()


def _configuration(artifacts: Path, stem: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((artifacts / "config" / f"{stem}.json").read_text()))


def _source_modules(source: Path) -> tuple[Any, Any, Any, Any]:
    os.environ["ATTN_BACKEND"] = "xformers"
    os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"
    sys.path.insert(0, str(source))
    sys.modules.setdefault("cumesh", types.ModuleType("cumesh"))
    convert = types.ModuleType("o_voxel.convert")
    convert.flexible_dual_grid_to_mesh = lambda *_args, **_kwargs: None
    sys.modules["o_voxel.convert"] = convert
    models = importlib.import_module("trellis2.models")
    sparse = importlib.import_module("trellis2.modules.sparse")
    vae = importlib.import_module("trellis2.models.sc_vaes.sparse_unet_vae")
    return models, sparse, vae, importlib.import_module("xformers.ops")


def _flow_inputs(*, channels: int, sparse: bool, resolution: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(1083 + channels + resolution)
    context = torch.randn((1, 17, 1024), generator=generator, dtype=torch.float32)
    timestep = torch.tensor([0.4375], dtype=torch.float32)
    if not sparse:
        return {
            "latent": torch.randn(
                (1, channels, 16, 16, 16), generator=generator, dtype=torch.float32
            ),
            "context": context,
            "timestep": timestep,
        }
    coordinates = torch.tensor(
        tuple(
            (0, index % resolution, (index // resolution) % resolution, 0) for index in range(128)
        ),
        dtype=torch.int32,
    )
    return {
        "coordinates": coordinates,
        "latent": torch.randn((128, 32), generator=generator, dtype=torch.float32),
        "shape": torch.randn((128, 32), generator=generator, dtype=torch.float32),
        "context": context,
        "timestep": timestep,
    }


def _generate_flows(
    models: Any,
    sparse_module: Any,
    artifacts: Path,
) -> dict[str, object]:
    cases = {
        "structure": ("structure-flow", 16, False, False),
        "shape-512": ("shape-512", 32, True, False),
        "shape-1024": ("shape-1024", 64, True, False),
        "texture-512": ("texture-512", 32, True, True),
        "texture-1024": ("texture-1024", 64, True, True),
    }
    results: dict[str, object] = {}
    for name, (artifact_role, resolution, is_sparse, texture) in cases.items():
        stem = Path(ARTIFACTS[artifact_role].relative).stem
        configuration = _configuration(artifacts, stem)
        model_type = getattr(models, configuration["name"])
        model = model_type(**configuration["args"])
        state = load_file(str(artifacts / ARTIFACTS[artifact_role].relative), device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        expected_missing = ["rope_phases"] if name == "structure" else []
        if missing != expected_missing or unexpected:
            raise RuntimeError(
                f"unexpected {name} state contract: missing={missing}, unexpected={unexpected}"
            )
        model.eval().cuda()
        inputs = _flow_inputs(
            channels=cast(int, configuration["args"]["in_channels"]),
            sparse=is_sparse,
            resolution=resolution,
        )
        with torch.inference_mode():
            if is_sparse:
                latent = sparse_module.SparseTensor(
                    feats=inputs["latent"].cuda(), coords=inputs["coordinates"].cuda()
                )
                concat = (
                    sparse_module.SparseTensor(
                        feats=inputs["shape"].cuda(),
                        coords=inputs["coordinates"].cuda(),
                    )
                    if texture
                    else None
                )
                output = model(
                    latent,
                    inputs["timestep"].cuda(),
                    inputs["context"].cuda(),
                    concat_cond=concat,
                ).feats
            else:
                output = model(
                    inputs["latent"].cuda(),
                    inputs["timestep"].cuda(),
                    inputs["context"].cuda(),
                )
        results[name] = {
            "artifact_role": artifact_role,
            "config": configuration,
            "inputs": {key: _tensor(value) for key, value in inputs.items()},
            "output": _tensor(output),
        }
        del model, state, output
        _release(inputs)
    return results


def _decoder_input() -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = torch.tensor(
        tuple((0, x, y, z) for x in range(2) for y in range(2) for z in range(2)),
        dtype=torch.int32,
    )
    generator = torch.Generator(device="cpu").manual_seed(2083)
    return coordinates, torch.randn((8, 32), generator=generator, dtype=torch.float32)


def _generate_decoders(
    models: Any,
    sparse_module: Any,
    vae_module: Any,
    artifacts: Path,
) -> dict[str, object]:
    structure_stem = Path(ARTIFACTS["structure-decoder"].relative).stem
    structure_config = _configuration(artifacts, structure_stem)
    structure = getattr(models, structure_config["name"])(**structure_config["args"])
    structure.load_state_dict(
        load_file(str(artifacts / ARTIFACTS["structure-decoder"].relative), device="cpu"),
        strict=True,
    )
    structure.eval().cuda()
    generator = torch.Generator(device="cpu").manual_seed(3083)
    structure_input = torch.randn((1, 8, 16, 16, 16), generator=generator)
    with torch.inference_mode():
        structure_output = structure(structure_input.cuda())
    results: dict[str, object] = {
        "structure": {
            "artifact_role": "structure-decoder",
            "config": structure_config,
            "input": _tensor(structure_input),
            "output": _tensor(structure_output),
        }
    }
    del structure, structure_output
    torch.cuda.empty_cache()

    coordinates, shape_features = _decoder_input()
    shape_stem = Path(ARTIFACTS["shape-decoder"].relative).stem
    shape_config = _configuration(artifacts, shape_stem)
    shape = getattr(models, shape_config["name"])(**shape_config["args"])
    shape.load_state_dict(
        load_file(str(artifacts / ARTIFACTS["shape-decoder"].relative), device="cpu"),
        strict=True,
    )
    shape.eval().cuda()
    shape_input = sparse_module.SparseTensor(feats=shape_features.cuda(), coords=coordinates.cuda())
    with torch.inference_mode():
        shape_output, subdivisions = vae_module.SparseUnetVaeDecoder.forward(
            shape, shape_input, return_subs=True
        )
    results["shape"] = {
        "artifact_role": "shape-decoder",
        "config": shape_config,
        "input": {"coordinates": _tensor(coordinates), "features": _tensor(shape_features)},
        "output": {
            "coordinates": _tensor(shape_output.coords),
            "features": _tensor(shape_output.feats),
            "subdivisions": [
                {"coordinates": _tensor(value.coords), "features": _tensor(value.feats)}
                for value in subdivisions
            ],
        },
    }
    del shape, shape_input, shape_output
    torch.cuda.empty_cache()

    texture_stem = Path(ARTIFACTS["texture-decoder"].relative).stem
    texture_config = _configuration(artifacts, texture_stem)
    texture = getattr(models, texture_config["name"])(**texture_config["args"])
    texture.load_state_dict(
        load_file(str(artifacts / ARTIFACTS["texture-decoder"].relative), device="cpu"),
        strict=True,
    )
    texture.eval().cuda()
    texture_generator = torch.Generator(device="cpu").manual_seed(4083)
    texture_features = torch.randn((8, 32), generator=texture_generator)
    texture_input = sparse_module.SparseTensor(
        feats=texture_features.cuda(), coords=coordinates.cuda()
    )
    with torch.inference_mode():
        texture_output = vae_module.SparseUnetVaeDecoder.forward(
            texture,
            texture_input,
            guide_subs=list(subdivisions),
        )
    results["texture"] = {
        "artifact_role": "texture-decoder",
        "config": texture_config,
        "input": {"coordinates": _tensor(coordinates), "features": _tensor(texture_features)},
        "guides": [
            {"coordinates": _tensor(value.coords), "features": _tensor(value.feats)}
            for value in subdivisions
        ],
        "output": {
            "coordinates": _tensor(texture_output.coords),
            "features": _tensor(texture_output.feats),
        },
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.source.resolve()
    artifacts = args.artifacts.resolve()
    _require_source(source)
    artifact_facts = _verify_artifacts(artifacts)
    models, sparse_module, vae_module, xformers = _source_modules(source)
    result = {
        "artifacts": artifact_facts,
        "decoders": _generate_decoders(models, sparse_module, vae_module, artifacts),
        "flows": _generate_flows(models, sparse_module, artifacts),
        "format": 1,
        "reference": {
            "attention": "xformers memory-efficient attention",
            "commit": REFERENCE_COMMIT,
            "device": torch.cuda.get_device_name(),
            "repository": "https://github.com/microsoft/TRELLIS.2",
            "sparse_convolution": "FlexGEMM",
            "torch": torch.__version__,
            "xformers": getattr(xformers, "__version__", None),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
