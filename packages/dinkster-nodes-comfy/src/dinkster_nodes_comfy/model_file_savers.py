"""Save native encoder weights through the public mount writer."""

from __future__ import annotations

import importlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CORE_STRING,
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
