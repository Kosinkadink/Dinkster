"""Generate a SAM 3.1 text-detection vector with pinned ComfyUI code.

Usage from the Dinkster repository root::

    /path/to/python tools/gen_sam31_detect_golden.py \
        /path/to/ComfyUI-at-8dc3f3f2 \
        /path/to/sam3.1_multiplex_fp16.safetensors \
        /path/to/neon_guitarist.png

Use Python 3.12 with the package versions recorded in the generated payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import PIL
import torch
import transformers
from gen_sam31_golden import (
    BASELINE,
    IMAGE_SIZE,
    MODEL_SHA256,
    SOURCE_BASELINE,
    SOURCE_SHA256,
    SOURCE_URL,
    TORCH_NUM_THREADS,
    Operations,
    _check_reference,
    _load,
    _record,
    _reference_modules,
    _segment,
    _sha256,
    _transform_key,
)
from PIL import Image
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers import CLIPTokenizer

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "sam31_detect_8dc3f3f.json"
PROMPT = "person"
TOKENIZER_COMMIT = "3bee28119e6b28e75b82b811b87b56935314e6a5"
TOKENIZER_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
TOKENIZER_URL = (
    "https://raw.githubusercontent.com/openai/CLIP/"
    f"{TOKENIZER_COMMIT}/clip/bpe_simple_vocab_16e6.txt.gz"
)
VOCAB_SHA256 = "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"
MERGES_SHA256 = "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    *,
    skip_reshape: bool = False,
    **_kwargs: object,
) -> torch.Tensor:
    if not skip_reshape:
        batch, tokens, channels = query.shape
        head_width = channels // heads
        query = query.view(batch, -1, heads, head_width).transpose(1, 2)
        key = key.view(batch, -1, heads, head_width).transpose(1, 2)
        value = value.view(batch, -1, heads, head_width).transpose(1, 2)
    else:
        batch, _, tokens, head_width = query.shape
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )
    return output.transpose(1, 2).reshape(batch, tokens, heads * head_width)


def _text_target(relative: str) -> str:
    if relative == "token_embedding.weight":
        return "embeddings.token_embedding.weight"
    if relative == "positional_embedding":
        return "embeddings.position_embedding.weight"
    if relative.startswith("ln_final."):
        return "final_layer_norm." + relative.removeprefix("ln_final.")
    target = "encoder.layers." + relative.removeprefix("transformer.resblocks.")
    target = target.replace(".ln_1.", ".layer_norm1.")
    target = target.replace(".ln_2.", ".layer_norm2.")
    target = target.replace(".attn.out_proj.", ".self_attn.out_proj.")
    target = target.replace(".mlp.c_fc.", ".mlp.fc1.")
    return target.replace(".mlp.c_proj.", ".mlp.fc2.")


def _load_reference(
    reference: Path,
    model_path: Path,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    _, tracker_type = _reference_modules(reference)
    Operations.GroupNorm = nn.GroupNorm  # type: ignore[attr-defined]
    attention = sys.modules["comfy.ldm.modules.attention"]
    attention.optimized_attention = _attention  # type: ignore[attr-defined]
    attention.optimized_attention_for_device = lambda *_args, **_kwargs: _attention  # type: ignore[attr-defined]
    operations = sys.modules["comfy.ops"]
    operations.cast_to = lambda value, **kwargs: value.to(**kwargs)  # type: ignore[attr-defined]

    detector_module = _load(
        "comfy.ldm.sam3.detector",
        reference / "comfy" / "ldm" / "sam3" / "detector.py",
    )
    clip_module = _load("comfy.clip_model", reference / "comfy" / "clip_model.py")
    with torch.device("meta"):
        detector = detector_module.SAM3Detector(
            image_model="SAM31",
            device=torch.device("meta"),
            dtype=torch.float32,
            operations=Operations,
        )
        tracker = tracker_type(
            device=torch.device("meta"),
            dtype=torch.float32,
            operations=Operations,
        )
        text_encoder = clip_module.CLIPTextModel_(
            {
                "hidden_act": "quick_gelu",
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_attention_heads": 16,
                "num_hidden_layers": 24,
                "max_position_embeddings": 32,
                "vocab_size": 49408,
                "layer_norm_eps": 1e-5,
                "eos_token_id": 49407,
            },
            torch.float32,
            torch.device("meta"),
            Operations,
        )

    detector_state: dict[str, torch.Tensor] = {}
    tracker_state: dict[str, torch.Tensor] = {}
    text_state: dict[str, torch.Tensor] = {}
    text_prefix = "detector.backbone.language_backbone.encoder."
    with safe_open(model_path, framework="pt", device="cpu") as stored:
        for source in stored.keys():
            value = stored.get_tensor(source)
            if source.startswith(text_prefix):
                relative = source.removeprefix(text_prefix)
                target = _text_target(relative)
                if target.endswith((".attn.in_proj_weight", ".attn.in_proj_bias")):
                    base, suffix = target.rsplit(".attn.in_proj_", 1)
                    ending = ".weight" if suffix == "weight" else ".bias"
                    size = value.shape[0] // 3
                    for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
                        text_state[f"{base}.self_attn.{name}{ending}"] = value[
                            index * size : (index + 1) * size
                        ]
                else:
                    text_state[target] = value
            elif source.startswith("detector."):
                detector_state.update(_transform_key(source.removeprefix("detector."), value))
            elif source.startswith("tracker.model."):
                tracker_state.update(_transform_key(source.removeprefix("tracker.model."), value))

    detector.load_state_dict(detector_state, strict=True, assign=True)
    tracker.load_state_dict(tracker_state, strict=True, assign=True)
    text_encoder.load_state_dict(text_state, strict=True, assign=True)
    detector.float().eval()
    tracker.float().eval()
    text_encoder.float().eval()
    sam = sys.modules["comfy.ldm.sam3.sam"]
    trunk = detector.backbone["vision_backbone"].trunk
    trunk.freqs_cis = sam.rope_2d(72, 72, 64, scale_pos=24 / 72)
    trunk.freqs_cis_window = sam.rope_2d(24, 24, 64)
    return detector, tracker, text_encoder


def _tokens(reference: Path) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = CLIPTokenizer.from_pretrained(reference / "comfy" / "sd1_tokenizer")
    values = cast("list[int]", tokenizer(PROMPT)["input_ids"])
    if len(values) > 32:
        raise ValueError("golden prompt exceeds one SAM 3.1 token section")
    values.extend([0] * (32 - len(values)))
    tokens = torch.tensor([values], dtype=torch.long)
    mask = torch.zeros_like(tokens)
    mask[:, : values.index(49407) + 1] = 1
    return tokens, mask


def _refine(
    detector: Any,
    tracker: Any,
    source: np.ndarray,
    coarse: torch.Tensor,
    box: torch.Tensor,
) -> torch.Tensor:
    height, width = source.shape[:2]
    left, top, right, bottom = (float(value) for value in box)
    box_width, box_height = right - left, bottom - top
    x0 = max(0, int(left - box_width * 0.1))
    y0 = max(0, int(top - box_height * 0.1))
    x1 = min(width, int(right + box_width * 0.1))
    y1 = min(height, int(bottom + box_height * 0.1))
    crop = torch.from_numpy(source[y0:y1, x0:x1].astype(np.float32) / 255.0)
    prepared = F.interpolate(
        crop.movedim(-1, 0).unsqueeze(0),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
    )
    backbone = detector.backbone["vision_backbone"]
    _, _, features, _ = backbone(prepared, tracker_mode="interactive")
    mask_height, mask_width = coarse.shape
    mask = coarse[
        int(y0 / height * mask_height) : int(y1 / height * mask_height),
        int(x0 / width * mask_width) : int(x1 / width * mask_width),
    ][None, None]
    for _ in range(2):
        mask_input = F.interpolate(
            mask,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        _, mask = _segment(tracker, features, mask=mask_input)
    refined = F.interpolate(
        mask,
        size=(y1 - y0, x1 - x0),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    coarse_full = F.interpolate(
        coarse[None, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    result = coarse_full > 0
    result[y0:y1, x0:x1] |= refined > 0
    return result.float()


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
    tokenizer_path = (
        REPO
        / "packages"
        / "dinkster-vision-sam31"
        / "src"
        / "dinkster_vision_sam31"
        / "data"
        / "bpe_simple_vocab_16e6.txt.gz"
    )
    if _sha256(tokenizer_path) != TOKENIZER_SHA256:
        raise SystemExit("bundled tokenizer does not match the pinned OpenAI CLIP artifact")
    tokenizer_root = reference / "comfy" / "sd1_tokenizer"
    if _sha256(tokenizer_root / "vocab.json") != VOCAB_SHA256:
        raise SystemExit("reference tokenizer vocabulary does not match")
    if _sha256(tokenizer_root / "merges.txt") != MERGES_SHA256:
        raise SystemExit("reference tokenizer merges do not match")

    torch.set_num_threads(TORCH_NUM_THREADS)
    source = np.asarray(
        Image.open(source_path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    )
    image = torch.from_numpy(source.astype(np.float32) / 255.0).movedim(-1, 0).unsqueeze(0)
    prepared = F.interpolate(image, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear")
    detector, tracker, text_encoder = _load_reference(reference, model_path)
    tokens, text_mask = _tokens(reference)
    with torch.inference_mode():
        token_embeddings = text_encoder.embeddings.token_embedding(tokens)
        text = text_encoder(
            None,
            text_mask,
            embeds=token_embeddings,
            num_tokens=text_mask.sum(1).tolist(),
            dtype=torch.float32,
        )[0]
        result = detector(prepared, text_embeddings=text, text_mask=text_mask)
        boxes = cast("torch.Tensor", result["boxes"])[0]
        logits = cast("torch.Tensor", result["scores"])[0]
        masks = cast("torch.Tensor", result["masks"])[0]
        top_query = int(torch.argmax(logits.sigmoid()))
        raw_box = boxes[top_query] * torch.tensor((256, 256, 256, 256))
        refined = _refine(detector, tracker, source, masks[top_query], raw_box)

    document = {
        "attentionMask": text_mask[0].tolist(),
        "baseline": BASELINE,
        "boxes": _record(boxes.numpy(), dtype=np.dtype(np.float32)),
        "coarseMask": _record(masks[top_query].numpy(), dtype=np.dtype(np.float32)),
        "logits": _record(logits.numpy(), dtype=np.dtype(np.float32)),
        "mergesSha256": MERGES_SHA256,
        "modelSha256": MODEL_SHA256,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "prompt": PROMPT,
        "refinedMask": _record(refined.numpy(), dtype=np.dtype(np.float32)),
        "source": _record(source, dtype=np.dtype(np.uint8)),
        "sourceImageBaseline": SOURCE_BASELINE,
        "sourceImageSha256": SOURCE_SHA256,
        "sourceImageUrl": SOURCE_URL,
        "textEmbedding": _record(text.numpy(), dtype=np.dtype(np.float32)),
        "tokenIds": tokens[0].tolist(),
        "tokenizerCommit": TOKENIZER_COMMIT,
        "tokenizerSha256": TOKENIZER_SHA256,
        "tokenizerUrl": TOKENIZER_URL,
        "topQuery": top_query,
        "torch": torch.__version__,
        "torchNumThreads": torch.get_num_threads(),
        "transformers": transformers.__version__,
        "vocabSha256": VOCAB_SHA256,
    }
    data = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(data)
    print(f"{OUT}: sha256:{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
