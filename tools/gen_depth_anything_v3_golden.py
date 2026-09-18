"""Generate a Depth Anything 3 Mono Large vector with pinned ComfyUI code.

Usage from the Dinkster repository root::

    /path/to/python tools/gen_depth_anything_v3_golden.py \
        /path/to/ComfyUI-at-e7051b0 /path/to/depth_anything_3_mono_large.safetensors

Use Python 3.12 with the package versions recorded in the generated payload.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from safetensors.torch import load_file

BASELINE = "e7051b03758a1247e3adb84a5b784ffacb9a23bd"
MODEL_SHA256 = "9b44eda5bedba5b4e125686fdb79d1db309c1b9785277576eb930f885b008f96"
TORCH_NUM_THREADS = 1
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "depth_anything_3_mono_large_e7051b0.json"


class Operations:
    Conv2d = torch.nn.Conv2d
    ConvTranspose2d = torch.nn.ConvTranspose2d
    LayerNorm = torch.nn.LayerNorm
    Linear = torch.nn.Linear


def _module(name: str, *, package: bool = False) -> ModuleType:
    module = ModuleType(name)
    if package:
        module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reference module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    **_kwargs: object,
) -> torch.Tensor:
    batch, tokens, channels = query.shape
    head_dim = channels // heads
    query = query.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )
    return output.transpose(1, 2).reshape(batch, tokens, channels)


def _common_upscale(
    samples: torch.Tensor,
    width: int,
    height: int,
    method: str,
    crop: str,
) -> torch.Tensor:
    if method != "lanczos" or crop != "disabled":
        raise ValueError("the DA3 reference must use uncropped Lanczos resizing")
    arrays = samples.movedim(1, -1).cpu().numpy()
    images = [
        Image.fromarray(np.clip(255.0 * array, 0, 255).astype(np.uint8)).resize(
            (width, height),
            resample=Image.Resampling.LANCZOS,
        )
        for array in arrays
    ]
    tensors = [torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0) for image in images]
    return torch.stack(tensors).movedim(-1, 1).to(samples.device, samples.dtype)


def _reference_model(reference: Path) -> tuple[torch.nn.Module, ModuleType]:
    comfy = _module("comfy", package=True)
    model_management = _module("comfy.model_management")
    model_management.cast_to_device = lambda value, device, dtype: value.to(  # type: ignore[attr-defined]
        device=device,
        dtype=dtype,
    )
    comfy.model_management = model_management  # type: ignore[attr-defined]
    comfy.ops = _module("comfy.ops")  # type: ignore[attr-defined]
    _module("comfy.text_encoders", package=True)
    _module("comfy.image_encoders", package=True)
    _module("comfy.ldm", package=True)
    _module("comfy.ldm.modules", package=True)
    attention = _module("comfy.ldm.modules.attention")
    attention.optimized_attention_for_device = lambda *_args, **_kwargs: _attention  # type: ignore[attr-defined]
    _module("comfy.ldm.depth_anything_3", package=True)
    selector = _module("comfy.ldm.depth_anything_3.reference_view_selector")
    selector.THRESH_FOR_REF_SELECTION = 3  # type: ignore[attr-defined]
    selector.select_reference_view = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    selector.reorder_by_reference = lambda value, _index: value  # type: ignore[attr-defined]
    selector.restore_original_order = lambda value, _index: value  # type: ignore[attr-defined]
    utils = _module("comfy.utils")
    utils.common_upscale = _common_upscale  # type: ignore[attr-defined]
    comfy.utils = utils  # type: ignore[attr-defined]

    root = reference / "comfy"
    _load("comfy.text_encoders.bert", root / "text_encoders" / "bert.py")
    _load("comfy.image_encoders.dino2", root / "image_encoders" / "dino2.py")
    _load("comfy.ldm.depth_anything_3.dpt", root / "ldm" / "depth_anything_3" / "dpt.py")
    for name in ("camera", "ray_pose", "transform"):
        stub = _module(f"comfy.ldm.depth_anything_3.{name}")
        stub.CameraDec = torch.nn.Module  # type: ignore[attr-defined]
        stub.CameraEnc = torch.nn.Module  # type: ignore[attr-defined]
        stub.get_extrinsic_from_camray = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
        stub.affine_inverse = lambda value: value  # type: ignore[attr-defined]
        stub.pose_encoding_to_extri_intri = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    model_module = _load(
        "comfy.ldm.depth_anything_3.model",
        root / "ldm" / "depth_anything_3" / "model.py",
    )
    preprocessing = _load(
        "comfy.ldm.depth_anything_3.preprocess",
        root / "ldm" / "depth_anything_3" / "preprocess.py",
    )
    model_type = cast("type[torch.nn.Module]", model_module.DepthAnything3Net)
    return model_type(
        device=torch.device("cpu"), dtype=torch.float32, operations=Operations
    ), preprocessing


def _source() -> np.ndarray:
    height, width = 48, 64
    y, x = np.mgrid[:height, :width]
    frame = np.stack(
        (
            (x * 4 + y * 2) % 256,
            (x * 2 + y * 5) % 256,
            ((x // 5) * 29 + (y // 6) * 37) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    cv2.circle(frame, (17, 23), 12, (232, 25, 91), -1)
    cv2.rectangle(frame, (35, 8), (58, 37), (18, 206, 139), -1)
    return frame


def _record(array: np.ndarray, *, dtype: np.dtype[np.generic]) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    return {
        "shape": list(contiguous.shape),
        f"{dtype.name}Base64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _check_reference(reference: Path) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != BASELINE:
        raise SystemExit(f"ComfyUI must be checked out at {BASELINE}, got {head}")
    status = subprocess.check_output(
        ["git", "-C", str(reference), "status", "--porcelain"],
        text=True,
    )
    if status:
        raise SystemExit("ComfyUI checkout must be clean")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    reference = args.reference.resolve()
    model_path = args.model.resolve()
    _check_reference(reference)
    with model_path.open("rb") as stream:
        model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if model_sha256 != MODEL_SHA256:
        raise SystemExit("model SHA-256 does not match the pinned Depth Anything 3 artifact")

    model, preprocessing = _reference_model(reference)
    stored = load_file(model_path, device="cpu")
    state = {name.removeprefix("model."): value for name, value in stored.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    source = _source()
    image = torch.from_numpy(source.astype(np.float32) / 255.0)[None]
    torch.set_num_threads(TORCH_NUM_THREADS)
    prepared = preprocessing.preprocess_image(image, process_res=504)
    with torch.inference_mode():
        predicted = model(prepared)["depth"]
        raw_depth = torch.nn.functional.interpolate(
            predicted[:, None],
            size=source.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        normalized = preprocessing.normalize_depth_min_max(raw_depth[None])[0].numpy()
    scale = 64.0 / min(normalized.shape)
    output = cv2.resize(
        normalized,
        (round(normalized.shape[1] * scale), round(normalized.shape[0] * scale)),
        interpolation=cv2.INTER_CUBIC,
    )
    output = np.repeat(np.clip(output, 0.0, 1.0)[:, :, None], 3, axis=2)

    document = {
        "baseline": BASELINE,
        "modelSha256": MODEL_SHA256,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
        "torch": torch.__version__,
        "torchNumThreads": torch.get_num_threads(),
        "source": _record(source, dtype=np.dtype(np.uint8)),
        "preparedSha256": hashlib.sha256(prepared.numpy().tobytes()).hexdigest(),
        "rawDepth": _record(raw_depth.numpy(), dtype=np.dtype(np.float32)),
        "output": _record(output, dtype=np.dtype(np.float32)),
    }
    data = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(data)
    print(f"{OUT}: sha256:{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
