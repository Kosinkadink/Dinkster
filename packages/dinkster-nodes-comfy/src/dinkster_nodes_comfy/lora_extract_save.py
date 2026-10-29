"""Extract model differences into a saved LoRA checkpoint."""

from __future__ import annotations

import importlib
import logging
import os
import tempfile
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, cast

from dinkster_api.v1 import (
    CLIP_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_INT,
    CORE_STRING,
    MODEL_TYPE,
    AssetError,
    AssetWriter,
    ComboWidget,
    InputSpec,
    MountSnapshotWriter,
    MountWidget,
    Node,
    NodeSchema,
    NumberWidget,
    TypeExpr,
    attach_saved_artifact,
    initialize_inference_process,
)

CLAMP_QUANTILE = 0.99
_MODEL = TypeExpr.concrete(MODEL_TYPE)
_CLIP = TypeExpr.concrete(CLIP_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)


def _writer() -> AssetWriter:
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
    return AssetWriter(MountSnapshotWriter(snapshot))


def extract_lora(diff: Any, rank: int) -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    conv2d = len(diff.shape) == 4
    kernel_size = None if not conv2d else cast(tuple[int, int], diff.size()[2:4])
    conv2d_3x3 = conv2d and kernel_size != (1, 1)
    out_dim, in_dim = diff.size()[0:2]
    rank = min(rank, in_dim, out_dim)

    if conv2d:
        if conv2d_3x3:
            diff = diff.flatten(start_dim=1)
        else:
            diff = diff.squeeze()

    u, s, vh = torch.linalg.svd(diff.float())
    u = u[:, :rank]
    s = s[:rank]
    u = u @ torch.diag(s)
    vh = vh[:rank, :]

    dist = torch.cat([u.flatten(), vh.flatten()])
    hi_val = torch.quantile(dist, CLAMP_QUANTILE)
    low_val = -hi_val

    u = u.clamp(low_val, hi_val)
    vh = vh.clamp(low_val, hi_val)
    if conv2d:
        assert kernel_size is not None
        u = u.reshape(out_dim, rank, 1, 1)
        vh = vh.reshape(rank, in_dim, kernel_size[0], kernel_size[1])
    return u, vh


class LORAType(Enum):
    STANDARD = 0
    FULL_DIFF = 1


LORA_TYPES = {"standard": LORAType.STANDARD, "full_diff": LORAType.FULL_DIFF}


def calc_lora_model(
    model_diff: Any,
    rank: int,
    prefix_model: str,
    prefix_lora: str,
    output_sd: dict[str, Any],
    lora_type: LORAType | None,
    bias_diff: bool = False,
) -> dict[str, Any]:
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    utils = importlib.import_module("dinkster_inference.utils")
    management.load_models_gpu([model_diff])
    sd = model_diff.model_state_dict(filter_prefix=prefix_model)

    sd_keys = list(sd.keys())
    for key in sd_keys:
        op_keys = key.rsplit(".", 1)
        if (
            len(op_keys) < 2
            or op_keys[1] not in ["weight", "bias"]
            or (op_keys[1] == "bias" and not bias_diff)
        ):
            continue
        op = utils.get_attr(model_diff.model, op_keys[0])
        if hasattr(op, "comfy_cast_weights") and not getattr(op, "comfy_patched_weights", False):
            weight_diff = model_diff.patch_weight_to_device(
                key, model_diff.load_device, return_weight=True
            )
        else:
            weight_diff = sd[key]

        if op_keys[1] == "weight":
            if lora_type == LORAType.STANDARD:
                if weight_diff.ndim < 2:
                    if bias_diff:
                        output_sd[f"{prefix_lora}{key[len(prefix_model) : -7]}.diff"] = (
                            weight_diff.contiguous().half().cpu()
                        )
                    continue
                try:
                    out = extract_lora(weight_diff, rank)
                    output_sd[f"{prefix_lora}{key[len(prefix_model) : -7]}.lora_up.weight"] = (
                        out[0].contiguous().half().cpu()
                    )
                    output_sd[f"{prefix_lora}{key[len(prefix_model) : -7]}.lora_down.weight"] = (
                        out[1].contiguous().half().cpu()
                    )
                except BaseException:
                    logging.warning(
                        "Could not generate lora weights for key %s, "
                        "is the weight difference a zero?",
                        key,
                    )
            elif lora_type == LORAType.FULL_DIFF:
                output_sd[f"{prefix_lora}{key[len(prefix_model) : -7]}.diff"] = (
                    weight_diff.contiguous().half().cpu()
                )

        elif bias_diff and op_keys[1] == "bias":
            output_sd[f"{prefix_lora}{key[len(prefix_model) : -5]}.diff_b"] = (
                weight_diff.contiguous().half().cpu()
            )
    return output_sd


class LoraSave(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoraSave",
            display_name="Extract and Save Lora",
            category="experimental",
            search_terms=("export lora",),
            inputs=(
                InputSpec("filename_prefix", _STRING, default="loras/ComfyUI_extracted_lora"),
                InputSpec(
                    "rank",
                    TypeExpr.concrete(CORE_INT),
                    default=8,
                    advanced=True,
                    widget=NumberWidget(min=1, max=4096, step=1),
                ),
                InputSpec(
                    "lora_type",
                    TypeExpr.concrete(CORE_COMBO),
                    default="standard",
                    advanced=True,
                    widget=ComboWidget(tuple(LORA_TYPES)),
                ),
                InputSpec(
                    "bias_diff", TypeExpr.concrete(CORE_BOOLEAN), default=True, advanced=True
                ),
                InputSpec(
                    "model_diff",
                    _MODEL,
                    required=False,
                    doc="The ModelSubtract output to be converted to a lora.",
                ),
                InputSpec(
                    "text_encoder_diff",
                    _CLIP,
                    required=False,
                    doc="The CLIPSubtract output to be converted to a lora.",
                ),
                InputSpec(
                    "mount",
                    _STRING,
                    required=False,
                    default=None,
                    widget=MountWidget(),
                    adjacent_to="filename_prefix",
                ),
            ),
            idempotent=False,
            output_node=True,
        )

    @classmethod
    def execute(
        cls,
        *,
        filename_prefix: str,
        rank: int,
        lora_type: str,
        bias_diff: bool,
        model_diff: Any = None,
        text_encoder_diff: Any = None,
        mount: str | None = None,
    ) -> Mapping[str, object]:
        if model_diff is None and text_encoder_diff is None:
            return cls.outputs()

        output_sd: dict[str, Any] = {}
        selected_lora_type = LORA_TYPES.get(lora_type)
        if model_diff is not None:
            output_sd = calc_lora_model(
                model_diff,
                rank,
                "diffusion_model.",
                "diffusion_model.",
                output_sd,
                selected_lora_type,
                bias_diff=bias_diff,
            )
        if text_encoder_diff is not None:
            output_sd = calc_lora_model(
                text_encoder_diff.patcher,
                rank,
                "",
                "text_encoders.",
                output_sd,
                selected_lora_type,
                bias_diff=bias_diff,
            )

        utils = importlib.import_module("dinkster_inference.utils")
        writer = _writer()
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            utils.save_torch_file(output_sd, str(temporary_path), metadata=None)
            with temporary_path.open("rb") as stream:
                ref = writer.save_stream(
                    writer.output_target(filename_prefix, mount=mount),
                    stream,
                    suffix=".safetensors",
                    media_type="application/octet-stream",
                    limit=temporary_path.stat().st_size,
                )
                attach_saved_artifact(ref.to_wire())
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return cls.outputs()


__all__ = [
    "CLAMP_QUANTILE",
    "LORA_TYPES",
    "LORAType",
    "LoraSave",
    "calc_lora_model",
    "extract_lora",
]
