"""Conditioning safetensors files through granted input and output mounts."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, cast

from dinkster_api.v1 import (
    ASSET_TYPE,
    CONDITIONING_TYPE,
    CORE_STRING,
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
    StringWidget,
    TypeExpr,
    initialize_inference_process,
)

_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_STRING = TypeExpr.concrete(CORE_STRING)


class ConditioningLoader(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ConditioningLoader",
            display_name="Load Conditioning",
            category="model/loaders",
            description="Loads a conditioning saved with Save Conditioning, or any safetensors "
            "file with a 'conditioning' tensor, from the embeddings folder.",
            inputs=(
                InputSpec(
                    "conditioning_name",
                    TypeExpr.concrete(ASSET_TYPE),
                    widget=AssetWidget(
                        accept=("application/octet-stream",), kind="model/embedding"
                    ),
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
        )

    @classmethod
    def execute(cls, *, conditioning_name: object) -> Mapping[str, object]:
        if not isinstance(conditioning_name, (AssetRef, LocalAssetRef)):
            raise TypeError("conditioning_name must be an AssetRef or LocalAssetRef")
        path = str(conditioning_name.local_path())
        initialize_inference_process()
        utils = importlib.import_module("dinkster_inference.utils")
        state, metadata = utils.load_torch_file(path, safe_load=True, return_metadata=True)
        conditioning = state.pop("conditioning")
        options = json.loads(
            cast(Mapping[str, Any], metadata or {}).get("conditioning_options", "{}")
        )
        lists: dict[str, dict[int, Any]] = {}
        for key, value in state.items():
            name, _, index = key.rpartition(".")
            if index.isdigit():
                lists.setdefault(name, {})[int(index)] = value
            else:
                options[key] = value
        options.update({key: [value[i] for i in sorted(value)] for key, value in lists.items()})
        return cls.outputs(CONDITIONING=[[conditioning, options]])


class SaveConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="SaveConditioning",
            display_name="Save Conditioning",
            category="model/conditioning",
            description="Saves a conditioning to the output folder as safetensors; move the file "
            "to models/embeddings to load it with Load Conditioning, e.g. to skip the text "
            "encoder.",
            search_terms=("save conditioning", "export conditioning", "cache text encoder output"),
            inputs=(
                InputSpec("conditioning", _CONDITIONING),
                InputSpec(
                    "filename_prefix",
                    _STRING,
                    default="conditioning/ComfyUI",
                    widget=StringWidget(multiline=False),
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
            outputs=(OutputSpec("conditioning", _CONDITIONING),),
            output_node=True,
            idempotent=False,
        )

    @classmethod
    def execute(
        cls,
        *,
        conditioning: Any,
        filename_prefix: str = "conditioning/ComfyUI",
        mount: str | None = None,
    ) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        safetensors = importlib.import_module("safetensors.torch")

        if len(conditioning) != 1:
            raise ValueError(
                "Save Conditioning supports a single conditioning entry, save it before combining."
            )
        tensor, options = conditioning[0]
        state = {"conditioning": tensor}
        values = {}
        for key, value in options.items():
            if isinstance(value, torch.Tensor):
                state[key] = value
            elif isinstance(value, (list, tuple)) and all(
                isinstance(t, torch.Tensor) for t in cast(Sequence[Any], value)
            ):
                state.update({f"{key}.{i}": t for i, t in enumerate(cast(Sequence[Any], value))})
            elif isinstance(value, (bool, int, float, str)):
                values[key] = value
            elif value is not None:
                raise ValueError(
                    f"Conditioning option '{key}' ({type(cast(Any, value)).__name__}) "
                    "can't be saved."
                )
        data = safetensors.save(
            {key: value.detach().to("cpu", copy=True).contiguous() for key, value in state.items()},
            metadata={"conditioning_options": json.dumps(values)},
        )
        snapshot = os.environ.get("DINKSTER_MOUNTS_SNAPSHOT", "")
        if not snapshot:
            raise AssetError("saving requires DINKSTER_MOUNTS_SNAPSHOT")
        writer = AssetWriter(MountSnapshotWriter(snapshot))
        writer.save_bytes(
            writer.output_target(filename_prefix, mount=mount),
            data,
            suffix="_.safetensors",
        )
        return cls.outputs(conditioning=conditioning)


__all__ = ["ConditioningLoader", "SaveConditioning"]
