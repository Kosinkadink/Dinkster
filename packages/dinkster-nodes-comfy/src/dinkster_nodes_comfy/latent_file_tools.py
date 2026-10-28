"""Load and save ComfyUI latent files through mounted assets."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_STRING,
    LATENT_TYPE,
    AssetError,
    AssetRef,
    AssetWidget,
    AssetWriter,
    InputSpec,
    LocalAssetRef,
    MountSnapshotWriter,
    MountWidget,
    Node,
    NodeSchema,
    OutputSpec,
    SourceFilenameSpec,
    TypeExpr,
    current_export_snapshot,
    initialize_inference_process,
    materialize_latent,
)

_ASSET = TypeExpr.concrete(ASSET_TYPE)
_LATENT = TypeExpr.concrete(LATENT_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)


class LoadLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoadLatent",
            display_name="Load Latent",
            category="model/latent",
            search_terms=("import latent", "open latent"),
            inputs=(
                InputSpec(
                    "latent",
                    _ASSET,
                    widget=AssetWidget(kind="data/latent", accept=(".latent",), allow_upload=True),
                    source_filename=SourceFilenameSpec("data/latent", "input"),
                ),
            ),
            outputs=(OutputSpec("LATENT", _LATENT),),
        )

    @classmethod
    def execute(cls, *, latent: AssetRef | LocalAssetRef) -> Mapping[str, object]:
        tensors = importlib.import_module("safetensors.torch").load_file(
            str(latent.local_path()), device="cpu"
        )
        multiplier = 1.0 if "latent_format_version_0" in tensors else 1.0 / 0.18215
        return cls.outputs(LATENT={"samples": tensors["latent_tensor"].float() * multiplier})


class SaveLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SaveLatent",
            display_name="Save Latent",
            category="model/latent",
            search_terms=("export latent",),
            inputs=(
                InputSpec("samples", _LATENT),
                InputSpec("filename_prefix", _STRING, default="latents/ComfyUI"),
                InputSpec(
                    "mount",
                    _STRING,
                    required=False,
                    default=None,
                    widget=MountWidget(),
                    adjacent_to="filename_prefix",
                ),
            ),
            outputs=(OutputSpec("samples", _LATENT),),
            idempotent=False,
            output_node=True,
        )

    @classmethod
    def execute(
        cls,
        *,
        samples: Any,
        filename_prefix: str = "latents/ComfyUI",
        mount: str | None = None,
    ) -> Mapping[str, object]:
        latent = materialize_latent(samples)
        torch = importlib.import_module("torch")
        initialize_inference_process()
        metadata = None
        if not importlib.import_module("dinkster_inference.cli_args").args.disable_metadata:
            submitted = current_export_snapshot()
            metadata = {
                "prompt": json.dumps(dict(submitted.prompt))
                if submitted is not None and submitted.prompt is not None
                else ""
            }
            if submitted is not None and submitted.extra_pnginfo is not None:
                metadata.update(
                    {key: json.dumps(value) for key, value in submitted.extra_pnginfo.items()}
                )
        data = importlib.import_module("safetensors.torch").save(
            {
                "latent_tensor": latent["samples"].contiguous(),
                "latent_format_version_0": torch.tensor([]),
            },
            metadata=metadata,
        )
        snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
        if not snapshot:
            raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
        writer = AssetWriter(MountSnapshotWriter(snapshot))
        writer.save_bytes(
            writer.output_target(filename_prefix, mount=mount),
            data,
            suffix=".latent",
        )
        return cls.outputs(samples=samples)


LATENT_FILE_NODES = (LoadLatent, SaveLatent)
