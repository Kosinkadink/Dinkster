"""Generate a BiRefNet foreground-matte vector with pinned ComfyUI code.

Usage from the Dinkster repository root::

    /path/to/python tools/gen_birefnet_golden.py \
        /path/to/ComfyUI-at-c67885b /path/to/birefnet.safetensors

Use Python 3.12 with the package versions recorded in the generated payload.
The reference import also requires torchvision and safetensors.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
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
from golden_platform import cpu_identity
from safetensors.torch import load_file
from torch import nn

BASELINE = "c67885b14556cf3e4e061862925282d403d09862"
MODEL_SHA256 = "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154"
MODEL_BLAKE3 = "blake3:03f8793ff101fb10981ee700fe276a6f481af00cb607dfafcfee46aeb8e638db"
TORCH_NUM_THREADS = 1
MODEL_INPUT_SIZE = 1024
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "birefnet_c67885b.json"


def _source() -> np.ndarray:
    height, width = 23, 31
    y, x = np.mgrid[:height, :width]
    return np.stack(
        (
            (17 * x + 11 * y + 3) % 256,
            (5 * x + 29 * y + 71) % 256,
            (37 * x + 7 * y + 19) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def _record(array: np.ndarray, *, dtype: np.dtype[np.generic], key: str) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    return {
        "shape": list(contiguous.shape),
        key: base64.b64encode(contiguous.tobytes()).decode("ascii"),
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
    Linear = nn.Linear
    Conv2d = nn.Conv2d
    BatchNorm2d = nn.BatchNorm2d
    LayerNorm = nn.LayerNorm


def _load_reference_model(reference: Path, model_path: Path) -> nn.Module:
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []  # type: ignore[attr-defined]
    ops = types.ModuleType("comfy.ops")

    def cast_to_input(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return value.to(target)

    @contextlib.contextmanager
    def cast_bias_weight_context(
        module: nn.Module,
        value: torch.Tensor,
        *,
        offloadable: bool,
    ):
        del offloadable
        weight = cast("nn.Conv2d", module).weight.to(value)
        bias = cast("nn.Conv2d", module).bias
        yield weight, None if bias is None else bias.to(value)

    ops.cast_to_input = cast_to_input  # type: ignore[attr-defined]
    ops.CastBiasWeightContext = cast_bias_weight_context  # type: ignore[attr-defined]
    comfy.ops = ops  # type: ignore[attr-defined]
    attention = types.ModuleType("comfy.ldm.modules.attention")

    def unused_attention(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("BiRefNet checkpoint unexpectedly used its unused Attention module")

    attention.optimized_attention_for_device = unused_attention  # type: ignore[attr-defined]
    sys.modules["comfy"] = comfy
    sys.modules["comfy.ops"] = ops
    sys.modules["comfy.ldm"] = types.ModuleType("comfy.ldm")
    sys.modules["comfy.ldm.modules"] = types.ModuleType("comfy.ldm.modules")
    sys.modules["comfy.ldm.modules.attention"] = attention

    module_path = reference / "comfy" / "background_removal" / "birefnet.py"
    spec = importlib.util.spec_from_file_location("_dinkster_birefnet_reference", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load ComfyUI BiRefNet source: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_type = cast("type[nn.Module]", module.BiRefNet)
    with torch.device("meta"):
        model = model_type(dtype=torch.float32, device="meta", operations=_TorchOperations)
    state = load_file(model_path, device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    return model.float().eval()


def _prepare_frame(frame: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0)
    if tensor.shape[2:] != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        tensor = F.interpolate(
            tensor,
            size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            mode="bicubic",
            antialias=True,
        )
    return torch.clip(255.0 * tensor, 0.0, 255.0).round() / 255.0


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
        raise SystemExit("model SHA-256 does not match the pinned BiRefNet artifact")

    torch.set_num_threads(TORCH_NUM_THREADS)
    model = _load_reference_model(reference, model_path)
    source = _source()
    frame = source.astype(np.float32) / 255.0
    with torch.no_grad():
        logits = model(_prepare_frame(frame))
        matte = F.interpolate(
            logits,
            size=frame.shape[:2],
            mode="bicubic",
            antialias=False,
        ).sigmoid()[0, 0]

    document = {
        "baseline": BASELINE,
        "generationCpu": cpu_identity(),
        "modelBlake3": MODEL_BLAKE3,
        "modelSha256": MODEL_SHA256,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchNumThreads": torch.get_num_threads(),
        "torchvision": torchvision.__version__,
        "source": _record(source, dtype=np.dtype(np.uint8), key="uint8Base64"),
        "matte": _record(matte.numpy(), dtype=np.dtype(np.float32), key="float32Base64"),
    }
    data = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(data)
    print(f"{OUT}: sha256:{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
