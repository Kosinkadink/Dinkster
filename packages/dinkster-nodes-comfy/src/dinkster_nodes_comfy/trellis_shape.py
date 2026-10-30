"""Trellis shape sampling setup from dense structure occupancy."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    LATENT_TYPE,
    VOXEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_voxel,
)


class Trellis2ShapeStage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        conditioning = TypeExpr.concrete(CONDITIONING_TYPE)
        return NodeSchema(
            node_type="Trellis2ShapeStage",
            display_name="Trellis2 Shape Stage",
            category="model/conditioning/trellis",
            inputs=(
                InputSpec("positive", conditioning),
                InputSpec("negative", conditioning),
                InputSpec(
                    "voxel",
                    TypeExpr.concrete(VOXEL_TYPE),
                    doc="Dense structure voxel from VaeDecodeStructureTrellis2.",
                ),
            ),
            outputs=(
                OutputSpec("positive", conditioning),
                OutputSpec("negative", conditioning),
                OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),
            ),
        )

    @classmethod
    def execute(cls, *, positive: Any, negative: Any, voxel: Any) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        decoded = materialize_voxel(voxel).data.unsqueeze(1)
        coords = torch.argwhere(decoded.bool())[:, [0, 2, 3, 4]].int()
        resolution = int(decoded.shape[-1])
        if coords.shape[0] == 0:
            raise ValueError("Trellis2 coords can't be empty")
        batch_ids = coords[:, 0].to(torch.int64)
        batch_size = int(batch_ids.max().item()) + 1
        counts = torch.bincount(batch_ids, minlength=batch_size)
        if (counts == 0).any():
            raise ValueError(
                "Non-contiguous Trellis2 batch ids in coords: "
                f"{batch_ids.unique(sorted=True).tolist()}"
            )
        max_tokens = int(counts.max().item())
        small = resolution <= 32
        extras: dict[str, Any] = {
            "trellis2_generation_mode": "shape_generation_512" if small else "shape_generation",
            "trellis2_coords": coords,
            "trellis2_coord_counts": counts,
        }
        proj_pack: Any = None
        if positive:
            entry = positive[0]
            if (
                isinstance(entry, (list, tuple))
                and len(cast(Any, entry)) >= 2
                and isinstance(entry[1], dict)
            ):
                proj_pack = cast(dict[str, Any], entry[1]).get("proj_feat_pack")
        if proj_pack is not None:
            initialize_inference_process()
            model = importlib.import_module("dinkster_inference.ldm.trellis2.model")
            extras["trellis2_proj_feats"] = model.compute_stage_proj_feats(
                proj_pack,
                "shape_512" if small else "shape_1024",
                coords=coords,
                coord_resolution=resolution,
            )
        outputs: dict[str, Any] = {}
        for name, conditioning in (("positive", positive), ("negative", negative)):
            entries: list[Any] = []
            for entry in conditioning:
                if (
                    isinstance(entry, (list, tuple))
                    and len(cast(Any, entry)) >= 2
                    and isinstance(entry[1], dict)
                ):
                    entries.append([entry[0], {**cast(dict[str, Any], entry[1]), **extras}])
                else:
                    entries.append(entry)
            outputs[name] = entries
        outputs["LATENT"] = {
            "samples": torch.zeros(batch_size, 32, max_tokens, 1),
            "coords": coords,
            "coord_counts": counts,
            "coord_resolution": resolution,
            "type": "trellis2",
            "model_frame": "y_up" if proj_pack is not None else "z_up",
        }
        return outputs
