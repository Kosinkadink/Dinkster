"""Encoded training datasets read from granted dataset folders."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_COMBO,
    LATENT_TYPE,
    AssetWidget,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    resolve_asset_selection,
)


class LoadTrainingDataset(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="LoadTrainingDataset",
            display_name="Load Training Dataset",
            category="model/training",
            description=(
                "Load encoded training dataset (latents + conditioning) "
                "from disk for use in training."
            ),
            search_terms=("import dataset", "training data"),
            inputs=(
                InputSpec(
                    "folder_name",
                    TypeExpr.concrete(CORE_COMBO),
                    widget=AssetWidget(kind="directory/dataset", selection=True),
                    doc="Saved dataset to load, from the datasets directory.",
                ),
            ),
            outputs=(
                OutputSpec(
                    "latents",
                    TypeExpr.list_of(TypeExpr.concrete(LATENT_TYPE)),
                    doc="List of latent dicts",
                ),
                OutputSpec(
                    "conditioning",
                    TypeExpr.list_of(TypeExpr.concrete(CONDITIONING_TYPE)),
                    doc="List of conditioning lists",
                ),
            ),
            idempotent=False,
        )

    @classmethod
    def execute(cls, *, folder_name: str) -> Mapping[str, object]:
        directory = resolve_asset_selection("directory/dataset", folder_name)
        if not isinstance(directory, Path):
            raise TypeError("training dataset selection requires a directory subtree")
        shards = sorted(
            path
            for path in directory.iterdir()
            if path.name.startswith("shard_") and path.name.endswith(".pkl")
        )
        if not shards:
            raise ValueError(f"No shard files found in {directory}")
        torch = importlib.import_module("torch")
        latents: list[Any] = []
        conditioning: list[Any] = []
        for path in shards:
            with path.open("rb") as handle:
                shard = torch.load(handle, weights_only=True)
            latents.extend(shard["latents"])
            conditioning.extend(shard["conditioning"])
        return cls.outputs(latents=latents, conditioning=conditioning)
