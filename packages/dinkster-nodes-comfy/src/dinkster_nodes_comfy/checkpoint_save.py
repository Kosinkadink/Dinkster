"""Image-only checkpoint export through the canonical mounted output writer."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from contextlib import ExitStack
from typing import Any, BinaryIO

from dinkster_api.v1 import (
    CLIP_VISION_TYPE,
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
    StringWidget,
    TypeExpr,
    initialize_inference_process,
)


class ImageOnlyCheckpointSave(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        string = TypeExpr.concrete(CORE_STRING)
        return NodeSchema(
            node_type="ImageOnlyCheckpointSave",
            category="model/merging",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("clip_vision", TypeExpr.concrete(CLIP_VISION_TYPE)),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec(
                    "filename_prefix",
                    string,
                    default="checkpoints/ComfyUI",
                    widget=StringWidget(multiline=False),
                ),
                InputSpec(
                    "mount",
                    string,
                    required=False,
                    widget=MountWidget(),
                    adjacent_to="filename_prefix",
                ),
            ),
            outputs=(),
            idempotent=False,
            output_node=True,
        )

    @classmethod
    def execute(
        cls,
        *,
        model: Any,
        clip_vision: Any,
        vae: Any,
        filename_prefix: str = "checkpoints/ComfyUI",
        mount: str | None = None,
        prompt: object = None,
        extra_pnginfo: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
        if not snapshot:
            raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
        writer = AssetWriter(MountSnapshotWriter(snapshot))
        target = writer.output_target(filename_prefix, mount=mount)
        initialize_inference_process()
        base = importlib.import_module("dinkster_inference.model_base")
        sampling = importlib.import_module("dinkster_inference.model_sampling")
        torch = importlib.import_module("torch")
        sd = importlib.import_module("dinkster_inference.sd")
        args = importlib.import_module("dinkster_inference.cli_args").args

        with tempfile.TemporaryDirectory() as folder, ExitStack() as streams:
            output_path = os.path.join(folder, "checkpoint.safetensors")

            def encode(stem: str, counter: int) -> BinaryIO:
                metadata: dict[str, str] = {}
                if isinstance(model.model, base.SDXL):
                    architecture = (
                        "stable-diffusion-xl-v1-edit"
                        if isinstance(model.model, base.SDXL_instructpix2pix)
                        else "stable-diffusion-xl-v1-base"
                    )
                elif isinstance(model.model, base.SDXLRefiner):
                    architecture = "stable-diffusion-xl-v1-refiner"
                elif isinstance(model.model, base.SVD_img2vid):
                    architecture = "stable-video-diffusion-img2vid-v1"
                elif isinstance(model.model, base.SD3):
                    architecture = "stable-diffusion-v3-medium"
                else:
                    architecture = None
                if architecture is not None:
                    metadata.update(
                        {
                            "modelspec.architecture": architecture,
                            "modelspec.sai_model_spec": "1.0.0",
                            "modelspec.implementation": "sgm",
                            "modelspec.title": f"{stem} {counter}",
                        }
                    )

                extra_keys: dict[str, Any] = {}
                model_sampling = model.get_model_object("model_sampling")
                if isinstance(model_sampling, sampling.ModelSamplingContinuousEDM) and isinstance(
                    model_sampling, sampling.V_PREDICTION
                ):
                    extra_keys["edm_vpred.sigma_max"] = torch.tensor(
                        model_sampling.sigma_max
                    ).float()
                    extra_keys["edm_vpred.sigma_min"] = torch.tensor(
                        model_sampling.sigma_min
                    ).float()
                if model.model.model_type == base.ModelType.EPS:
                    metadata["modelspec.predict_key"] = "epsilon"
                elif model.model.model_type == base.ModelType.V_PREDICTION:
                    metadata["modelspec.predict_key"] = "v"
                    extra_keys["v_pred"] = torch.tensor([])
                    if getattr(model_sampling, "zsnr", False):
                        extra_keys["ztsnr"] = torch.tensor([])
                if not args.disable_metadata:
                    metadata["prompt"] = "" if prompt is None else json.dumps(prompt)
                    if extra_pnginfo is not None:
                        metadata.update(
                            {key: json.dumps(value) for key, value in extra_pnginfo.items()}
                        )

                sd.save_checkpoint(
                    output_path,
                    model,
                    clip_vision=clip_vision,
                    vae=vae,
                    metadata=metadata,
                    extra_keys=extra_keys,
                )
                # Safetensors can replace the path, so open the completed file, not its predecessor.
                return streams.enter_context(open(output_path, "rb"))

            writer.save_encoded(
                target,
                encode,
                suffix="_.safetensors",
                limit=sys.maxsize,
            )
        return cls.outputs()
