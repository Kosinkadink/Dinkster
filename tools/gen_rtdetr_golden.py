"""Generate an RT-DETR v4 x-HGNet parity vector with pinned ComfyUI code.

Usage from the Dinkster repository root::

    /path/to/python tools/gen_rtdetr_golden.py \
        /path/to/ComfyUI-at-c67885b /path/to/rt_detr_v4-x-hgnet_fp16.safetensors

Use Python 3.12 with the package versions recorded in the generated payload.
The reference import also requires torchvision and safetensors.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from safetensors.torch import load_file
from torch import nn

BASELINE = "c67885b14556cf3e4e061862925282d403d09862"
MODEL_SHA256 = "581f9af9bbabb664d1891cbccd823308b176ecd409146f954dfa39af3bec2476"
MODEL_BLAKE3 = "blake3:5eaa01a6d16d654d9a4991ab1dfe489b580acc4b939cd1963ac6d12ceb9dc7f8"
TORCH_NUM_THREADS = 1
MODEL_INPUT_SIZE = 640
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "rtdetr_v4_x_hgnet_c67885b.json"


def _source() -> np.ndarray:
    height, width = 37, 53
    y, x = np.mgrid[:height, :width]
    return np.stack(
        (
            (13 * x + 23 * y + 5) % 256,
            (31 * x + 7 * y + 47) % 256,
            ((x // 5) * 29 + (y // 3) * 17 + 11) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


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


class _TorchOperations:
    Conv2d = nn.Conv2d
    Linear = nn.Linear
    LayerNorm = nn.LayerNorm


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    heads: int,
    mask: torch.Tensor | None = None,
    **_kwargs: object,
) -> torch.Tensor:
    batch, tokens, channels = query.shape
    head_width = channels // heads

    def split(tensor: torch.Tensor) -> torch.Tensor:
        return (
            tensor.unsqueeze(3)
            .reshape(batch, -1, heads, head_width)
            .permute(0, 2, 1, 3)
            .reshape(batch * heads, -1, head_width)
            .contiguous()
        )

    query, key, value = split(query), split(key), split(value)
    similarity = torch.einsum("b i d, b j d -> b i j", query, key) * head_width**-0.5
    if mask is not None:
        if mask.dtype == torch.bool:
            expanded = mask.reshape(mask.shape[0], -1)[:, None].repeat_interleave(heads, dim=0)
            similarity.masked_fill_(~expanded, -torch.finfo(similarity.dtype).max)
        else:
            mask_batch = 1 if mask.ndim == 2 else mask.shape[0]
            expanded = mask.reshape(mask_batch, -1, mask.shape[-2], mask.shape[-1])
            expanded = expanded.expand(batch, heads, -1, -1).reshape(
                -1, mask.shape[-2], mask.shape[-1]
            )
            similarity.add_(expanded)
    attention = similarity.softmax(dim=-1)
    output = torch.einsum(
        "b i j, b j d -> b i d",
        attention.to(value.dtype),
        value,
    )
    return (
        output.unsqueeze(0)
        .reshape(batch, heads, tokens, head_width)
        .permute(0, 2, 1, 3)
        .reshape(batch, tokens, channels)
    )


def _load_reference_model(reference: Path, model_path: Path) -> nn.Module:
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []  # type: ignore[attr-defined]
    model_management = types.ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: torch.device("cpu")  # type: ignore[attr-defined]
    comfy.model_management = model_management  # type: ignore[attr-defined]
    attention = types.ModuleType("comfy.ldm.modules.attention")
    attention.optimized_attention_for_device = lambda *_args, **_kwargs: _attention  # type: ignore[attr-defined]
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.ldm"] = types.ModuleType("comfy.ldm")
    sys.modules["comfy.ldm.modules"] = types.ModuleType("comfy.ldm.modules")
    sys.modules["comfy.ldm.modules.attention"] = attention

    module_path = reference / "comfy" / "ldm" / "rt_detr" / "rtdetr_v4.py"
    spec = importlib.util.spec_from_file_location("_dinkster_rtdetr_reference", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load ComfyUI RT-DETR source: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_type = cast("type[nn.Module]", module.RTv4)
    model = model_type(
        enc_h=384,
        device=torch.device("cpu"),
        dtype=torch.float32,
        operations=_TorchOperations,
    )
    state = load_file(model_path, device="cpu")
    model.load_state_dict(state, strict=True)
    return model.float().eval()


def _prepare(source: np.ndarray) -> torch.Tensor:
    frame = source.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.stack([frame])).movedim(-1, 1)
    return F.interpolate(tensor, size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), mode="bilinear")


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
        raise SystemExit("model SHA-256 does not match the pinned RT-DETR artifact")

    torch.set_num_threads(TORCH_NUM_THREADS)
    model = _load_reference_model(reference, model_path)
    source = _source()
    with torch.inference_mode():
        outputs = model._forward(_prepare(source))
        results = model.postprocess(outputs, (source.shape[1], source.shape[0]))
    result = results[0]

    document = {
        "baseline": BASELINE,
        "modelBlake3": MODEL_BLAKE3,
        "modelSha256": MODEL_SHA256,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchNumThreads": torch.get_num_threads(),
        "torchvision": torchvision.__version__,
        "source": _record(source, dtype=np.dtype(np.uint8)),
        "predLogits": _record(outputs["pred_logits"].numpy(), dtype=np.dtype(np.float32)),
        "predBoxes": _record(outputs["pred_boxes"].numpy(), dtype=np.dtype(np.float32)),
        "labels": _record(result["labels"].numpy(), dtype=np.dtype(np.int64)),
        "boxes": _record(result["boxes"].numpy(), dtype=np.dtype(np.float32)),
        "scores": _record(result["scores"].numpy(), dtype=np.dtype(np.float32)),
    }
    data = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(data)
    print(f"{OUT}: sha256:{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
