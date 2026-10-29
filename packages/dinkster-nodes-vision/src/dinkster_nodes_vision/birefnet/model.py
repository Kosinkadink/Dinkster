"""BiRefNet foreground matting over BHWC image batches."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from dinkster_api.v1 import declared_asset
from dinkster_inference_torch.birefnet import BiRefNet
from safetensors.torch import load_file

MODEL_INPUT_SIZE = 1024

_MODEL: BiRefNet | None = None
_MODEL_DIGEST = ""


def load_model() -> BiRefNet:
    """Load the declared BiRefNet checkpoint, cached per asset digest."""
    global _MODEL, _MODEL_DIGEST
    reference = declared_asset("birefnet-general")
    if _MODEL is not None and _MODEL_DIGEST == reference.digest:
        return _MODEL
    with torch.device("meta"):
        model = BiRefNet()
    state = load_file(reference.local_path(), device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    model.float().eval()
    _MODEL = model
    _MODEL_DIGEST = reference.digest
    return model


def _frames(image: object) -> list[np.ndarray]:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[1] < 1 or array.shape[2] < 1:
        raise ValueError(f"image must have non-empty BHWC shape, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("BiRefNet matting requires finite pixel values")
    array = np.clip(array, 0.0, 1.0)
    if array.shape[3] == 3:
        return list(array)
    if array.shape[3] == 1:
        return list(np.repeat(array, 3, axis=3))
    if array.shape[3] == 4:
        return list(array[..., :3])
    raise ValueError(f"image frame must have 1, 3, or 4 channels, got {array.shape}")


def prepare_frame(frame: np.ndarray) -> torch.Tensor:
    """Match ComfyUI's BiRefNet bicubic resize and byte-grid quantization."""
    tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0)
    if tensor.shape[2:] != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        tensor = F.interpolate(
            tensor,
            size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            mode="bicubic",
            antialias=True,
        )
    return torch.clip(255.0 * tensor, 0.0, 255.0).round() / 255.0


def _matte_frame(model: BiRefNet, frame: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        logits = model(prepare_frame(frame))
        logits = F.interpolate(
            logits,
            size=frame.shape[:2],
            mode="bicubic",
            antialias=False,
        )
        matte = logits.sigmoid()[0, 0].numpy(force=True)
    return np.ascontiguousarray(matte, dtype=np.float32)


def execute_matte(image: object) -> np.ndarray:
    """Estimate one immutable soft foreground matte per input frame."""
    frames = _frames(image)
    model = load_model()
    mattes = [_matte_frame(model, frame) for frame in frames]
    output = np.ascontiguousarray(np.stack(mattes), dtype=np.float32)
    immutable = np.frombuffer(output.tobytes(), dtype=np.float32).reshape(output.shape)
    return cast("np.ndarray", immutable)


__all__ = ["MODEL_INPUT_SIZE", "execute_matte", "load_model", "prepare_frame"]
