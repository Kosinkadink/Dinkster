"""Trellis sparse shape cascade and second-pass sampling inputs."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_INT,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)


class Trellis2UpsampleStage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        conditioning = TypeExpr.concrete(CONDITIONING_TYPE)
        return NodeSchema(
            node_type="Trellis2UpsampleStage",
            display_name="Trellis2 Upsample Stage",
            category="model/conditioning/trellis",
            inputs=(
                InputSpec("positive", conditioning),
                InputSpec("negative", conditioning),
                InputSpec(
                    "shape_latent",
                    TypeExpr.concrete(LATENT_TYPE),
                    doc=(
                        "The 512-resolution shape latent output from the first "
                        "shape-stage KSampler."
                    ),
                ),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec(
                    "target_resolution",
                    TypeExpr.concrete(CORE_INT),
                    default=1024,
                    widget=NumberWidget(min=1024, max=2048, step=128),
                    doc="Voxel resolution of the upsampled shape. Higher = more detail, more VRAM.",
                ),
            ),
            outputs=(
                OutputSpec("positive", conditioning),
                OutputSpec("negative", conditioning),
                OutputSpec("LATENT", TypeExpr.concrete(LATENT_TYPE)),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        positive: Any,
        negative: Any,
        shape_latent: Any,
        vae: Any,
        target_resolution: int = 1024,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        formats = importlib.import_module("dinkster_inference.latent_formats")
        sparse = importlib.import_module("dinkster_inference.ldm.trellis2.vae")
        latent = materialize_latent(shape_latent)
        device = management.get_torch_device()
        vae.prepare_decode(latent["samples"].shape)
        samples = latent["samples"].squeeze(-1).transpose(1, 2)
        coords = latent["coords"]
        coord_counts = latent.get("coord_counts")
        items: list[tuple[Any, Any]]
        if coord_counts is None:
            items = [(samples.reshape(-1, samples.shape[-1]), coords)]
        else:
            if coord_counts.ndim != 1:
                raise ValueError(
                    f"Trellis2 coord_counts must be 1D, got shape {tuple(coord_counts.shape)}"
                )
            counts = [int(count) for count in coord_counts.tolist()]
            if any(count < 0 for count in counts):
                raise ValueError(f"Trellis2 coord_counts must be non-negative, got {counts}")
            if sum(counts) != coords.shape[0]:
                raise ValueError(
                    f"Trellis2 coord_counts total {sum(counts)} "
                    f"does not match coords rows {coords.shape[0]}"
                )
            ids = coords[:, 0].to(torch.int64)
            order = torch.argsort(ids, stable=True)
            sorted_coords = coords.index_select(0, order)
            sorted_ids = ids.index_select(0, order)
            items = []
            start = 0
            for i, count in enumerate(counts):
                coords_i = sorted_coords[start : start + count]
                if coords_i.shape[0] != count or not torch.all(
                    sorted_ids[start : start + count] == i
                ):
                    raise ValueError(
                        f"Trellis2 coords rows for batch {i} expected {count}, "
                        f"got {coords_i.shape[0]}"
                    )
                items.append((samples[i, :count], coords_i))
                start += count

        proj_pack: Any = None
        if positive:
            entry = positive[0]
            if (
                isinstance(entry, (list, tuple))
                and len(cast(Any, entry)) >= 2
                and isinstance(entry[1], dict)
            ):
                proj_pack = cast(dict[str, Any], entry[1]).get("proj_feat_pack")
        shape_format = formats.Trellis2ShapeSLAT()
        quantized: list[Any] = []
        for feats_i, coords_i in items:
            coords_i = coords_i.to(device)
            if coord_counts is not None:
                coords_i = coords_i.clone()
                coords_i[:, 0] = 0
            slat = sparse.SparseTensor(
                feats=shape_format.process_out(feats_i.to(device)), coords=coords_i
            )
            hr_coords = vae.first_stage_model.upsample_shape(
                slat.to(vae.vae_dtype), upsample_times=4
            )
            spatial = hr_coords[:, 1:].float()
            grid_res = target_resolution // 16
            if proj_pack is not None:
                spatial.add_(0.5).mul_((grid_res - 1) / 512).round_()
            else:
                spatial.add_(0.5).mul_(grid_res / 512)
            quantized.append(torch.cat([hr_coords[:, :1], spatial.int()], dim=1).unique(dim=0))

        per_sample_counts: list[int] = []
        for i, coords_i in enumerate(quantized):
            coords_i[:, 0] = i
            per_sample_counts.append(int(coords_i.shape[0]))
        coords = torch.cat(quantized, dim=0)
        if coords.shape[0] == 0:
            raise ValueError("Trellis2 coords can't be empty")
        ids = coords[:, 0].to(torch.int64)
        batch_size = int(ids.max().item()) + 1
        inferred_counts = torch.bincount(ids, minlength=batch_size)
        if (inferred_counts == 0).any():
            raise ValueError(
                f"Non-contiguous Trellis2 batch ids in coords: {ids.unique(sorted=True).tolist()}"
            )
        counts = torch.tensor(per_sample_counts, dtype=torch.int64)
        resolution = target_resolution // 16
        extras: dict[str, Any] = {
            "trellis2_generation_mode": "shape_generation",
            "trellis2_coords": coords,
            "trellis2_coord_counts": counts,
        }
        if proj_pack is not None:
            model = importlib.import_module("dinkster_inference.ldm.trellis2.model")
            extras["trellis2_proj_feats"] = model.compute_stage_proj_feats(
                proj_pack, "shape_1024", coords=coords, coord_resolution=resolution
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
            "samples": torch.zeros(batch_size, 32, int(inferred_counts.max().item()), 1),
            "coords": coords,
            "coord_counts": counts,
            "coord_resolution": resolution,
            "type": "trellis2",
            "model_frame": latent.get("model_frame", "y_up" if proj_pack is not None else "z_up"),
        }
        return outputs
