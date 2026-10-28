"""Save native model and encoder weights through the public mount writer."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO

from dinkster_api.v1 import (
    CLIP_TYPE,
    CORE_STRING,
    MODEL_TYPE,
    VAE_TYPE,
    AssetError,
    AssetWriter,
    InputSpec,
    MountSnapshotWriter,
    MountWidget,
    Node,
    NodeSchema,
    TypeExpr,
    initialize_inference_process,
)

_STRING = TypeExpr.concrete(CORE_STRING)
_MOUNT = InputSpec(
    "mount",
    _STRING,
    required=False,
    default=None,
    widget=MountWidget(),
    adjacent_to="filename_prefix",
)


def _save_tensors(
    tensors: Mapping[str, Any], metadata: dict[str, str], prefix: str, mount: str | None
) -> None:
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
    writer = AssetWriter(MountSnapshotWriter(snapshot))
    utils = importlib.import_module("dinkster_inference.utils")
    with tempfile.TemporaryDirectory(prefix="dinkster-model-save-") as directory:
        path = Path(directory) / "weights.safetensors"
        utils.save_torch_file(tensors, str(path), metadata=metadata)
        with path.open("rb") as source:
            writer.save_stream(
                writer.output_target(prefix, mount=mount),
                source,
                suffix=".safetensors",
                limit=path.stat().st_size,
            )


def _save_checkpoint(model: Any, clip: Any, vae: Any, prefix: str, mount: str | None) -> None:
    initialize_inference_process()
    snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
    if not snapshot:
        raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
    writer = AssetWriter(MountSnapshotWriter(snapshot))
    model_base = importlib.import_module("dinkster_inference.model_base")
    model_sampling = importlib.import_module("dinkster_inference.model_sampling")
    sd = importlib.import_module("dinkster_inference.sd")
    args = importlib.import_module("dinkster_inference.cli_args").args
    torch = importlib.import_module("torch")
    metadata: dict[str, str] = {}
    for model_class, architecture in (
        (model_base.SDXL_instructpix2pix, "stable-diffusion-xl-v1-edit"),
        (model_base.SDXL, "stable-diffusion-xl-v1-base"),
        (model_base.SDXLRefiner, "stable-diffusion-xl-v1-refiner"),
        (model_base.SVD_img2vid, "stable-video-diffusion-img2vid-v1"),
        (model_base.SD3, "stable-diffusion-v3-medium"),
    ):
        if isinstance(model.model, model_class):
            metadata["modelspec.architecture"] = architecture
            metadata["modelspec.sai_model_spec"] = "1.0.0"
            metadata["modelspec.implementation"] = "sgm"
            break
    extra_keys: dict[str, Any] = {}
    sampling = model.get_model_object("model_sampling")
    if isinstance(sampling, model_sampling.ModelSamplingContinuousEDM) and isinstance(
        sampling, model_sampling.V_PREDICTION
    ):
        extra_keys["edm_vpred.sigma_max"] = torch.tensor(sampling.sigma_max).float()
        extra_keys["edm_vpred.sigma_min"] = torch.tensor(sampling.sigma_min).float()
    if model.model.model_type == model_base.ModelType.EPS:
        metadata["modelspec.predict_key"] = "epsilon"
    elif model.model.model_type == model_base.ModelType.V_PREDICTION:
        metadata["modelspec.predict_key"] = "v"
        extra_keys["v_pred"] = torch.tensor([])
        if getattr(sampling, "zsnr", False):
            extra_keys["ztsnr"] = torch.tensor([])
    if not args.disable_metadata:
        metadata["prompt"] = ""
    with (
        tempfile.TemporaryDirectory(prefix="dinkster-checkpoint-save-") as directory,
        ExitStack() as streams,
    ):
        path = Path(directory) / "checkpoint.safetensors"

        def encode(stem: str, counter: int) -> BinaryIO:
            if "modelspec.architecture" in metadata:
                metadata["modelspec.title"] = f"{stem} {counter}"
            sd.save_checkpoint(
                str(path), model, clip, vae, metadata=metadata, extra_keys=extra_keys
            )
            return streams.enter_context(path.open("rb"))

        writer.save_encoded(
            writer.output_target(prefix, mount=mount),
            encode,
            suffix="_.safetensors",
            limit=sys.maxsize,
        )


class CheckpointSave(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CheckpointSave",
            display_name="Save Checkpoint",
            category="model/merging",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec("filename_prefix", _STRING, default="checkpoints/ComfyUI"),
                _MOUNT,
            ),
            output_node=True,
            idempotent=False,
            search_terms=("save model", "export checkpoint", "merge save"),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        clip: Any,
        vae: Any,
        filename_prefix: str = "checkpoints/ComfyUI",
        mount: str | None = None,
    ) -> Mapping[str, object]:
        _save_checkpoint(model, clip, vae, filename_prefix, mount)
        return cls.outputs()


class ModelSave(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ModelSave",
            category="model/merging",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("filename_prefix", _STRING, default="diffusion_models/ComfyUI"),
                _MOUNT,
            ),
            output_node=True,
            idempotent=False,
            search_terms=("export model", "checkpoint save"),
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        filename_prefix: str = "diffusion_models/ComfyUI",
        mount: str | None = None,
    ) -> Mapping[str, object]:
        _save_checkpoint(model, None, None, filename_prefix, mount)
        return cls.outputs()


class CLIPSave(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="CLIPSave",
            category="model/merging",
            inputs=(
                InputSpec("clip", TypeExpr.concrete(CLIP_TYPE)),
                InputSpec("filename_prefix", _STRING, default="clip/ComfyUI"),
                _MOUNT,
            ),
            output_node=True,
            idempotent=False,
        )

    @classmethod
    def execute(
        cls, *, clip: Any, filename_prefix: str = "clip/ComfyUI", mount: str | None = None
    ) -> Mapping[str, object]:
        initialize_inference_process()
        utils = importlib.import_module("dinkster_inference.utils")
        clip.load_model()
        state = clip.state_dict_for_saving()
        for prefix in (
            "clip_l.",
            "clip_g.",
            "clip_h.",
            "t5xxl.",
            "pile_t5xl.",
            "mt5xl.",
            "umt5xxl.",
            "t5base.",
            "gemma2_2b.",
            "llama.",
            "hydit_clip.",
            "",
        ):
            selected = {key: state.pop(key) for key in list(state) if key.startswith(prefix)}
            if not selected:
                continue
            replacements = {prefix: ""} if prefix else {}
            replacements["transformer."] = ""
            selected = utils.state_dict_prefix_replace(selected, replacements)
            target_prefix = f"{filename_prefix}_{prefix[:-1]}" if prefix else filename_prefix
            _save_tensors(selected, {"format": "pt", "prompt": ""}, target_prefix, mount)
        return cls.outputs()


class VAESave(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VAESave",
            category="model/merging",
            inputs=(
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec("filename_prefix", _STRING, default="vae/ComfyUI_vae"),
                _MOUNT,
            ),
            output_node=True,
            idempotent=False,
        )

    @classmethod
    def execute(
        cls, *, vae: Any, filename_prefix: str = "vae/ComfyUI_vae", mount: str | None = None
    ) -> Mapping[str, object]:
        initialize_inference_process()
        _save_tensors(vae.get_sd(), {"prompt": ""}, filename_prefix, mount)
        return cls.outputs()


MODEL_FILE_SAVE_NODES = (CLIPSave, VAESave)
