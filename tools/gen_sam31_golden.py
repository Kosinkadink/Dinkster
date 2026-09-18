"""Generate a SAM 3.1 box-prompt vector with pinned ComfyUI code.

Usage from the Dinkster repository root::

    /path/to/python tools/gen_sam31_golden.py \
        /path/to/ComfyUI-at-8dc3f3f2 \
        /path/to/sam3.1_multiplex_fp16.safetensors \
        /path/to/neon_guitarist.png

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
from typing import Any, cast

import numpy as np
import PIL
import torch
from PIL import Image
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

BASELINE = "8dc3f3f2094121c0a013e21d89136ebc331d2974"
MODEL_SHA256 = "9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03"
SOURCE_BASELINE = "0b1ef3ec90846bf82eba195ddcc30a1f5b2b6b38"
SOURCE_SHA256 = "83e63383d1715a7084afb5cf1e2e47e302869c2c944bfa62d86e769b4ff65ecc"
SOURCE_URL = (
    "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/"
    f"{SOURCE_BASELINE}/input/neon_guitarist.png"
)
TORCH_NUM_THREADS = 1
IMAGE_SIZE = 1008
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "sam31_multiplex_8dc3f3f.json"


class Operations:
    Conv2d = nn.Conv2d
    ConvTranspose2d = nn.ConvTranspose2d
    Embedding = nn.Embedding
    LayerNorm = nn.LayerNorm
    Linear = nn.Linear


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
    *,
    skip_reshape: bool = False,
    **_kwargs: object,
) -> torch.Tensor:
    if not skip_reshape:
        batch, tokens, channels = query.shape
        head_dim = channels // heads
        query = query.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
        key = key.reshape(batch, key.shape[1], heads, head_dim).transpose(1, 2)
        value = value.reshape(batch, value.shape[1], heads, head_dim).transpose(1, 2)
    else:
        batch, _, tokens, head_dim = query.shape
    output = F.scaled_dot_product_attention(query, key, value)
    return output.transpose(1, 2).reshape(batch, tokens, heads * head_dim)


def _cast_to_input(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return value.to(device=reference.device, dtype=reference.dtype)


def _rope(position: torch.Tensor, dimension: int, theta: int) -> torch.Tensor:
    scale = torch.linspace(
        0,
        (dimension - 2) / dimension,
        steps=dimension // 2,
        dtype=torch.float64,
    )
    omega = 1.0 / (theta**scale)
    angles = torch.einsum("...n,d->...nd", position.to(dtype=torch.float32), omega)
    matrix = torch.stack(
        (torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)),
        dim=-1,
    )
    return matrix.reshape(*matrix.shape[:-1], 2, 2).to(dtype=torch.float32)


def _apply_rope(value: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    pairs = value.to(dtype=frequencies.dtype).reshape(*value.shape[:-1], -1, 1, 2)
    if pairs.shape[2] != 1 and frequencies.shape[2] != 1 and pairs.shape[2] != frequencies.shape[2]:
        frequencies = frequencies[:, :, : pairs.shape[2]]
    output = frequencies[..., 0] * pairs[..., 0]
    output = output + frequencies[..., 1] * pairs[..., 1]
    return output.reshape_as(value).to(dtype=value.dtype)


class EmbedND(nn.Module):
    def __init__(self, dimension: int, theta: int, axes_dim: list[int]) -> None:
        super().__init__()
        self.dimension = dimension
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        embedded = torch.cat(
            [
                _rope(coordinates[..., index], self.axes_dim[index], self.theta)
                for index in range(coordinates.shape[-1])
            ],
            dim=-3,
        )
        return embedded.unsqueeze(1)


def _layer_norm_2d(operations: type[Operations]) -> type[nn.LayerNorm]:
    class LayerNorm2d(operations.LayerNorm):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return super().forward(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    return LayerNorm2d


def _reference_modules(reference: Path) -> tuple[type[nn.Module], type[nn.Module]]:
    comfy = _module("comfy", package=True)
    _module("comfy.ldm", package=True)
    _module("comfy.ldm.modules", package=True)
    _module("comfy.ldm.flux", package=True)
    _module("comfy.ldm.cascade", package=True)
    _module("comfy.ldm.sam3", package=True)
    model_management = _module("comfy.model_management")
    model_management.intermediate_device = lambda: torch.device("cpu")  # type: ignore[attr-defined]
    model_management.is_device_cuda = lambda device: device.type == "cuda"  # type: ignore[attr-defined]
    comfy.model_management = model_management  # type: ignore[attr-defined]
    utils = _module("comfy.utils")
    utils.common_upscale = (  # type: ignore[attr-defined]
        lambda samples, width, height, method, crop: F.interpolate(
            samples,
            size=(height, width),
            mode=method,
        )
    )
    comfy.utils = utils  # type: ignore[attr-defined]
    attention = _module("comfy.ldm.modules.attention")
    attention.optimized_attention = _attention  # type: ignore[attr-defined]
    flux_math = _module("comfy.ldm.flux.math")
    flux_math.apply_rope = lambda query, key, frequencies: (  # type: ignore[attr-defined]
        _apply_rope(query, frequencies),
        _apply_rope(key, frequencies),
    )
    flux_math.apply_rope1 = _apply_rope  # type: ignore[attr-defined]
    flux_layers = _module("comfy.ldm.flux.layers")
    flux_layers.EmbedND = lambda dim, theta, axes_dim: EmbedND(dim, theta, axes_dim)  # type: ignore[attr-defined]
    operations = _module("comfy.ops")
    operations.cast_to_input = _cast_to_input  # type: ignore[attr-defined]
    comfy.ops = operations  # type: ignore[attr-defined]
    cascade = _module("comfy.ldm.cascade.common")
    cascade.LayerNorm2d_op = _layer_norm_2d  # type: ignore[attr-defined]

    root = reference / "comfy" / "ldm" / "sam3"
    sam = _load("comfy.ldm.sam3.sam", root / "sam.py")
    tracker = _load("comfy.ldm.sam3.tracker", root / "tracker.py")
    return cast("type[nn.Module]", sam.SAM3VisionBackbone), cast(
        "type[nn.Module]",
        tracker.SAM31Tracker,
    )


def _transform_key(target: str, value: torch.Tensor) -> dict[str, torch.Tensor]:
    if target.endswith((".in_proj_weight", ".in_proj_bias")):
        base, suffix = target.rsplit(".in_proj_", 1)
        ending = ".weight" if suffix == "weight" else ".bias"
        size = value.shape[0] // 3
        return {
            base + ".q_proj" + ending: value[:size],
            base + ".k_proj" + ending: value[size : 2 * size],
            base + ".v_proj" + ending: value[2 * size :],
        }
    target = target.replace(".mlp.lin1.", ".mlp.0.")
    target = target.replace(".mlp.lin2.", ".mlp.2.")
    target = target.replace(".norm_final_attn.", ".norm_final.")
    return {target: value}


def _load_models(
    reference: Path,
    model_path: Path,
) -> tuple[nn.Module, nn.Module]:
    backbone_type, tracker_type = _reference_modules(reference)
    backbone = backbone_type(
        embed_dim=1024,
        d_model=256,
        multiplex=True,
        device=torch.device("meta"),
        dtype=torch.float32,
        operations=Operations,
    )
    tracker = tracker_type(
        device=torch.device("meta"),
        dtype=torch.float32,
        operations=Operations,
    )
    backbone_state: dict[str, torch.Tensor] = {}
    tracker_state: dict[str, torch.Tensor] = {}
    with safe_open(model_path, framework="pt", device="cpu") as stored:
        for source in stored.keys():
            if ".attn.freqs_cis" in source:
                continue
            if source.startswith("detector.backbone.vision_backbone."):
                target = source.removeprefix("detector.backbone.vision_backbone.")
                backbone_state.update(_transform_key(target, stored.get_tensor(source)))
            elif source.startswith("tracker.model."):
                target = source.removeprefix("tracker.model.")
                tracker_state.update(_transform_key(target, stored.get_tensor(source)))
    backbone.load_state_dict(backbone_state, strict=True, assign=True)
    tracker.load_state_dict(tracker_state, strict=True, assign=True)
    backbone.float().eval()
    tracker.float().eval()
    return backbone, tracker


def _segment(
    tracker: Any,
    features: list[torch.Tensor],
    *,
    box: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    high_resolution = features[:-1]
    backbone = _conditioned_feature(tracker, features[-1])
    low_resolution_masks, high_resolution_masks, _, _ = tracker._forward_sam_heads(
        backbone_features=backbone,
        point_inputs=None,
        mask_inputs=mask,
        box_inputs=box,
        high_res_features=high_resolution,
        multimask_output=False,
    )
    return cast("torch.Tensor", low_resolution_masks), cast(
        "torch.Tensor",
        high_resolution_masks,
    )


def _conditioned_feature(tracker: Any, feature: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = feature.shape
    flattened = feature.flatten(2).permute(0, 2, 1)
    flattened = flattened + _cast_to_input(tracker.interactivity_no_mem_embed, flattened)
    return flattened.view(batch, height, width, channels).permute(0, 3, 1, 2)


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


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    reference = args.reference.resolve()
    model_path = args.model.resolve()
    source_path = args.source.resolve()
    _check_reference(reference)
    if _sha256(model_path) != MODEL_SHA256:
        raise SystemExit("model SHA-256 does not match the pinned SAM 3.1 artifact")
    if _sha256(source_path) != SOURCE_SHA256:
        raise SystemExit("source SHA-256 does not match the pinned workflow input")

    torch.set_num_threads(TORCH_NUM_THREADS)
    source = np.asarray(
        Image.open(source_path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    )
    image = torch.from_numpy(source.astype(np.float32) / 255.0).movedim(-1, 0).unsqueeze(0)
    prepared = F.interpolate(image, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear")
    box = torch.tensor([[[29.0, 34.0], [243.5, 256.0]]], dtype=torch.float32) * (IMAGE_SIZE / 256)
    backbone, tracker = _load_models(reference, model_path)
    with torch.inference_mode():
        _, _, features, _ = backbone(prepared, tracker_mode="interactive")
        point_coordinates = torch.zeros((1, 1, 2), dtype=torch.float32)
        point_labels = -torch.ones((1, 1), dtype=torch.int32)
        sparse_prompt, dense_prompt = tracker.interactive_sam_prompt_encoder(
            points=(point_coordinates, point_labels),
            boxes=box,
            masks=None,
        )
        prompt_position = tracker.interactive_sam_prompt_encoder.get_dense_pe()
        first_low_resolution, first = _segment(tracker, features, box=box)
        _, refined = _segment(tracker, features, mask=first)
        restored = F.interpolate(
            refined,
            size=source.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    feature_hashes = [hashlib.sha256(feature.numpy().tobytes()).hexdigest() for feature in features]
    conditioned_hash = hashlib.sha256(
        _conditioned_feature(tracker, features[-1]).detach().numpy().tobytes()
    ).hexdigest()
    document = {
        "baseline": BASELINE,
        "box": [29.0, 34.0, 243.5, 256.0],
        "conditionedFeatureSha256": conditioned_hash,
        "featureSha256": feature_hashes,
        "firstLowResolutionSha256": hashlib.sha256(
            first_low_resolution.numpy().tobytes()
        ).hexdigest(),
        "firstPassSha256": hashlib.sha256(first.numpy().tobytes()).hexdigest(),
        "modelSha256": MODEL_SHA256,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "preparedSha256": hashlib.sha256(prepared.numpy().tobytes()).hexdigest(),
        "densePromptSha256": hashlib.sha256(dense_prompt.detach().numpy().tobytes()).hexdigest(),
        "promptPositionSha256": hashlib.sha256(
            prompt_position.detach().numpy().tobytes()
        ).hexdigest(),
        "refinedLogits": _record(restored.numpy(), dtype=np.dtype(np.float32)),
        "source": _record(source, dtype=np.dtype(np.uint8)),
        "sourceImageBaseline": SOURCE_BASELINE,
        "sourceImageSha256": SOURCE_SHA256,
        "sourceImageUrl": SOURCE_URL,
        "sparsePromptSha256": hashlib.sha256(sparse_prompt.detach().numpy().tobytes()).hexdigest(),
        "torch": torch.__version__,
        "torchNumThreads": torch.get_num_threads(),
    }
    data = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(data)
    print(f"{OUT}: sha256:{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
