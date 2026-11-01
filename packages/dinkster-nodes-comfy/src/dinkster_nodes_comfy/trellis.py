"""Trellis sparse latent stage setup using the shared conditioning and latent values."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    LATENT_TYPE,
    SHAPE_SUBDIVIDES_TYPE,
    VAE_TYPE,
    VOXEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
    materialize_shape_subdivides,
)


def _coord_layout(coords: Any) -> tuple[int, Any, int]:
    torch = importlib.import_module("torch")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"Expected Trellis2 coords with shape [N, 4], got {tuple(coords.shape)}")
    if coords.shape[0] == 0:
        raise ValueError("Trellis2 coords can't be empty")
    batch_ids = coords[:, 0].to(torch.int64)
    if (batch_ids < 0).any():
        raise ValueError(
            f"Trellis2 batch ids must be non-negative, got {batch_ids.unique(sorted=True).tolist()}"
        )
    batch_size = int(batch_ids.max().item()) + 1
    counts = torch.bincount(batch_ids, minlength=batch_size)
    if (counts == 0).any():
        raise ValueError(
            f"Non-contiguous Trellis2 batch ids in coords: {batch_ids.unique(sorted=True).tolist()}"
        )
    return batch_size, counts, int(counts.max().item())


def flatten_sparse_samples(samples: Any, coords: Any, counts: Any) -> Any:
    torch = importlib.import_module("torch")
    samples = samples.squeeze(-1).transpose(1, 2)
    batch_ids = coords[:, 0].to(torch.int64)
    order = torch.argsort(batch_ids, stable=True)
    sorted_ids = batch_ids.index_select(0, order)
    features: list[Any] = []
    start = 0
    for index, count in enumerate(counts.tolist()):
        ids = sorted_ids[start : start + count]
        if ids.shape[0] != count or not torch.all(ids == index):
            raise ValueError(
                f"Trellis2 coords rows for batch {index} expected {count}, got {ids.shape[0]}"
            )
        features.append(samples[index, :count])
        start += count
    return torch.cat(features, dim=0)


def _conditioning_extras(conditioning: Any, extras: Mapping[str, object]) -> list[Any]:
    result: list[Any] = []
    for entry in conditioning:
        if (
            isinstance(entry, (list, tuple))
            and len(cast(Any, entry)) >= 2
            and isinstance(entry[1], dict)
        ):
            result.append([entry[0], {**cast(dict[str, Any], entry[1]), **extras}])
        else:
            result.append(entry)
    return result


class Trellis2TextureStage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        conditioning = TypeExpr.concrete(CONDITIONING_TYPE)
        latent = TypeExpr.concrete(LATENT_TYPE)
        return NodeSchema(
            node_type="Trellis2TextureStage",
            display_name="Trellis2 Texture Stage",
            category="model/conditioning/trellis",
            inputs=(
                InputSpec("positive", conditioning),
                InputSpec("negative", conditioning),
                InputSpec("shape_latent", latent),
            ),
            outputs=(
                OutputSpec("positive", conditioning),
                OutputSpec("negative", conditioning),
                OutputSpec("LATENT", latent),
            ),
        )

    @classmethod
    def execute(cls, *, positive: Any, negative: Any, shape_latent: object) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        shape = materialize_latent(shape_latent)
        coords = shape["coords"]
        resolution = shape.get("coord_resolution")
        batch_size, counts, max_tokens = _coord_layout(coords)
        shape_samples = shape["samples"]
        if shape_samples.ndim == 4:
            shape_samples = flatten_sparse_samples(shape_samples, coords, counts)

        proj_pack: Any = None
        if positive:
            entry = positive[0]
            if (
                isinstance(entry, (list, tuple))
                and len(cast(Any, entry)) >= 2
                and isinstance(entry[1], dict)
            ):
                proj_pack = cast(dict[str, Any], entry[1]).get("proj_feat_pack")
        frame = shape.get("model_frame", "y_up" if proj_pack is not None else "z_up")
        extras = {
            "trellis2_generation_mode": "texture_generation",
            "trellis2_coords": coords,
            "trellis2_coord_counts": counts,
            "trellis2_shape_slat": shape_samples,
            "trellis2_model_frame": frame,
        }
        if proj_pack is not None and resolution is not None:
            initialize_inference_process()
            model = importlib.import_module("dinkster_inference.ldm.trellis2.model")
            extras["trellis2_proj_feats"] = model.compute_stage_proj_feats(
                proj_pack, "tex_1024", coords=coords, coord_resolution=resolution
            )
        latent = {
            "samples": torch.zeros(batch_size, 32, max_tokens, 1),
            "type": "trellis2",
            "coords": coords,
            "coord_counts": counts,
            "model_frame": frame,
        }
        if resolution is not None:
            latent["coord_resolution"] = resolution
        return cls.outputs(
            positive=_conditioning_extras(positive, extras),
            negative=_conditioning_extras(negative, extras),
            LATENT=latent,
        )


class VaeDecodeTextureTrellis(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VaeDecodeTextureTrellis",
            display_name="Trellis2 VAE Decode Texture",
            category="model/latent/trellis",
            inputs=(
                InputSpec("samples", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                InputSpec(
                    "shape_subdivides",
                    TypeExpr.concrete(SHAPE_SUBDIVIDES_TYPE),
                    doc="Shape information used to guide higher-detail reconstruction "
                    "during decoding. "
                    "Helps preserve structure consistency at higher resolutions.",
                ),
            ),
            outputs=(OutputSpec("voxel_colors", TypeExpr.concrete(VOXEL_TYPE)),),
        )

    @classmethod
    def execute(cls, *, samples: Any, vae: Any, shape_subdivides: Any) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        sparse = importlib.import_module("dinkster_inference.ldm.trellis2.vae")
        formats = importlib.import_module("dinkster_inference.latent_formats")
        latent = materialize_latent(samples)
        stages = materialize_shape_subdivides(shape_subdivides)
        device = management.get_torch_device()
        bytes_per_point = 896 * management.dtype_size(vae.vae_dtype) + 27 * 4
        vae.prepare_decode(
            latent["samples"].shape,
            memory_required=2 * 1024**3 + int(stages[-1].feats.shape[0]) * bytes_per_point,
        )
        coords = latent["coords"]
        counts = latent.get("coord_counts")
        if counts is None:
            features = latent["samples"].squeeze(-1).transpose(1, 2)
            features = features.reshape(-1, features.shape[-1])
        else:
            features = flatten_sparse_samples(latent["samples"], coords, counts)
            order = torch.argsort(coords[:, 0].to(torch.int64), stable=True)
            coords = coords.index_select(0, order)
        sample = sparse.SparseTensor(
            feats=formats.Trellis2TexSLAT().process_out(features.to(device)),
            coords=coords.to(device),
        )
        stages = [
            sparse.SparseTensor(
                feats=stage.feats.to(device),
                coords=stage.coords.to(device),
                shape=stage.shape,
                scale=stage._scale,
            )
            for stage in stages
        ]
        voxel = vae.first_stage_model.decode_tex_slat(sample.to(vae.vae_dtype), stages)
        coords = voxel.coords
        resolution = latent.get("coord_resolution")
        if resolution is not None:
            resolution = int(resolution) * 16
        elif coords.numel() > 0 and coords.shape[-1] >= 3:
            spatial = coords[:, -3:] if coords.shape[-1] == 4 else coords
            max_index = int(spatial.max().item()) + 1
            resolution = next(
                (r for r in (256, 512, 1024, 1536, 2048) if r >= max_index), max_index
            )
        else:
            resolution = 1024
        if (
            latent.get("model_frame", "y_up") == "z_up"
            and coords.numel() > 0
            and coords.shape[-1] >= 3
        ):
            spatial = coords[:, -3:]
            rotated = torch.stack(
                [spatial[:, 0], spatial[:, 2], resolution - 1 - spatial[:, 1]], dim=-1
            )
            coords = (
                torch.cat([coords[:, :1], rotated], dim=-1) if coords.shape[-1] == 4 else rotated
            )
        output_device = management.intermediate_device()
        return cls.outputs(
            voxel_colors={
                "data": coords.to(output_device),
                "colors": voxel.feats.to(output_device),
                "resolution": resolution,
            }
        )
