"""SeedVR2 output alignment and memory-bounded color correction."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    IMAGE_TYPE,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_image,
)


def _as_bthwc(images: Any) -> tuple[Any, bool]:
    if images.ndim == 4:
        return images.unsqueeze(0), True
    if images.ndim == 5:
        return images, False
    raise ValueError(
        f"SeedVR2PostProcessing: expected 4-D or 5-D IMAGE tensor, got shape {tuple(images.shape)}"
    )


def _color_transfer(decoded: Any, reference: Any, method: str) -> Any:
    torch = importlib.import_module("torch")
    management = importlib.import_module("dinkster_inference.model_management")
    constants = importlib.import_module("dinkster_inference.ldm.seedvr.constants")
    color = importlib.import_module("dinkster_inference.ldm.seedvr.color_fix")
    multiplier = {
        "lab": constants.SEEDVR2_LAB_SCALE_MULTIPLIER,
        "wavelet": constants.SEEDVR2_WAVELET_SCALE_MULTIPLIER,
        "adain": constants.SEEDVR2_ADAIN_SCALE_MULTIPLIER,
    }[method]
    transfer = {
        "lab": color.lab_color_transfer,
        "wavelet": color.wavelet_color_transfer,
        "adain": color.adain_color_transfer,
    }[method]
    frames, channels, height, width = decoded.shape
    dtype_bytes = max(decoded.element_size(), constants.SEEDVR2_DTYPE_BYTES_FLOOR)
    bytes_per_frame = height * width * channels * dtype_bytes * multiplier
    color_device = management.vae_device()
    chunk_size = (
        max(
            1,
            min(
                frames,
                int(
                    (
                        management.get_free_memory(color_device)
                        * constants.SEEDVR2_COLOR_MEM_HEADROOM
                    )
                    // bytes_per_frame
                ),
            ),
        )
        if bytes_per_frame > 0
        else frames
    )
    while True:
        try:
            result = None
            for start in range(0, frames, chunk_size):
                end = min(start + chunk_size, frames)
                if method == "lab":
                    outputs = None
                    for frame in range(start, end):
                        output = transfer(
                            decoded[frame : frame + 1].to(device=color_device).clone(),
                            reference[frame : frame + 1].to(device=color_device).clone(),
                        ).to(device=decoded.device)
                        if outputs is None:
                            outputs = torch.empty(
                                (end - start,) + tuple(output.shape[1:]),
                                device=decoded.device,
                                dtype=output.dtype,
                            )
                        outputs[frame - start : frame - start + 1].copy_(output)
                    if outputs is None:
                        raise ValueError(
                            "SeedVR2PostProcessing: LAB color correction "
                            "requires at least one frame."
                        )
                    output = outputs
                else:
                    output = transfer(
                        decoded[start:end].to(device=color_device),
                        reference[start:end].to(device=color_device),
                    ).to(device=decoded.device)
                if result is None:
                    result = torch.empty(
                        (frames,) + tuple(output.shape[1:]),
                        device=decoded.device,
                        dtype=output.dtype,
                    )
                result[start:end].copy_(output)
            if result is None:
                raise ValueError(
                    "SeedVR2PostProcessing: color correction requires at least one frame."
                )
            return result
        except Exception as error:
            management.raise_non_oom(error)
            if chunk_size <= 1:
                raise RuntimeError(
                    "SeedVR2PostProcessing: color correction OOM at one frame; "
                    f"color_correction_method={method}, shape={tuple(decoded.shape)}."
                ) from error
            chunk_size = max(1, chunk_size // constants.SEEDVR2_OOM_BACKOFF_DIVISOR)


class SeedVR2PostProcessing(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        image = TypeExpr.concrete(IMAGE_TYPE)
        return NodeSchema(
            node_type="SeedVR2PostProcessing",
            display_name="Post-Process SeedVR2 Output",
            category="image/post-processors",
            description=(
                "Align the generated image with the original resized image "
                "and apply color correction."
            ),
            search_terms=("seedvr2", "upscale", "color correction", "color match", "postprocess"),
            inputs=(
                InputSpec("images", image, doc="The generated image to process."),
                InputSpec(
                    "original_resized_images",
                    image,
                    doc="The original resized image before pre-processing, used as reference.",
                ),
                InputSpec(
                    "color_correction_method",
                    TypeExpr.concrete(CORE_COMBO),
                    default="lab",
                    widget=ComboWidget(("lab", "wavelet", "adain", "none")),
                    doc=(
                        "Method to match the generated image colors to the original image. "
                        "lab: transfer color in CIELAB space, preserving detail (most faithful). "
                        "wavelet: transfer low-frequency color, "
                        "keeping upscaled high-frequency detail. "
                        "adain: match per-channel mean/std (fastest, global tint). "
                        "none: skip color transfer (geometry alignment only)."
                    ),
                ),
            ),
            outputs=(OutputSpec("images", image, doc="The aligned, color-corrected image."),),
        )

    @classmethod
    def execute(
        cls, *, images: object, original_resized_images: object, color_correction_method: str
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        decoded, was_4d = _as_bthwc(materialize_image(images))
        original = materialize_image(original_resized_images)
        alpha = original[..., 3:4] if original.shape[-1] == 4 else None
        reference, _ = _as_bthwc(original[..., :3] if alpha is not None else original)
        ref_b, ref_t = reference.shape[:2]
        if decoded.shape[0] == 1 and ref_b >= 1 and decoded.shape[1] % ref_b == 0:
            decoded_t = decoded.shape[1] // ref_b
            if decoded_t >= ref_t:
                decoded = decoded.reshape(ref_b, decoded_t, *decoded.shape[2:])
        batch = min(decoded.shape[0], ref_b)
        frames = min(decoded.shape[1], ref_t)
        height = min(decoded.shape[2], reference.shape[2])
        width = min(decoded.shape[3], reference.shape[3])
        decoded = decoded[:batch, :frames, :height, :width, :]
        if color_correction_method in ("lab", "wavelet", "adain"):
            reference = reference[:batch, :frames]
            if reference.shape[2:4] != (height, width):
                transforms = importlib.import_module("torchvision.transforms.functional")
                reference_flat = reference.permute(0, 1, 4, 2, 3).reshape(
                    batch * frames, reference.shape[4], reference.shape[2], reference.shape[3]
                )
                resized = transforms.resize(
                    reference_flat,
                    size=(height, width),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=reference_flat.device.type != "mps",
                )
                reference = resized.reshape(batch, frames, resized.shape[1], height, width).permute(
                    0, 1, 3, 4, 2
                )
            decoded_raw, reference_raw = decoded.mul(2.0).sub(1.0), reference.mul(2.0).sub(1.0)
            output = _color_transfer(
                decoded_raw.permute(0, 1, 4, 2, 3).reshape(
                    batch * frames, decoded_raw.shape[4], height, width
                ),
                reference_raw.permute(0, 1, 4, 2, 3).reshape(
                    batch * frames, reference_raw.shape[4], height, width
                ),
                color_correction_method,
            )
            output = output.reshape(batch, frames, output.shape[1], height, width).permute(
                0, 1, 3, 4, 2
            )
            output = output.add(1.0).div(2.0).clamp(0.0, 1.0)
        elif color_correction_method == "none":
            output = decoded
        else:
            raise ValueError(
                "SeedVR2PostProcessing: unknown color_correction_method "
                f"{color_correction_method!r}"
            )
        if alpha is not None:
            alpha_5d, _ = _as_bthwc(alpha)
            alpha_5d = alpha_5d[
                : output.shape[0], : output.shape[1], : output.shape[2], : output.shape[3], :
            ]
            output = torch.cat(
                (output, alpha_5d.to(dtype=output.dtype, device=output.device)), dim=-1
            )
        output = output[:, :, : output.shape[-3] // 2 * 2, : output.shape[-2] // 2 * 2, :]
        if was_4d:
            output = output.reshape(-1, *output.shape[-3:])
        return cls.outputs(images=output)
