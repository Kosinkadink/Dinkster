"""Trellis shape sampling setup from dense structure occupancy."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    GIBIBYTE,
    LATENT_TYPE,
    MESH_TYPE,
    SHAPE_SUBDIVIDES_TYPE,
    VAE_TYPE,
    VOXEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TriangleMeshBatch,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
    materialize_shape_subdivides,
    materialize_voxel,
)

from .trellis import flatten_sparse_samples


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


class VaeDecodeShapeTrellis(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="VaeDecodeShapeTrellis",
            display_name="Trellis2 VAE Decode Shape",
            category="model/latent/trellis",
            inputs=(
                InputSpec("samples", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
            ),
            outputs=(
                OutputSpec("mesh", TypeExpr.concrete(MESH_TYPE)),
                OutputSpec("shape_subdivides", TypeExpr.concrete(SHAPE_SUBDIVIDES_TYPE)),
            ),
        )

    @classmethod
    def execute(cls, *, samples: object, vae: Any) -> Mapping[str, object]:
        initialize_inference_process()
        torch = importlib.import_module("torch")
        management = importlib.import_module("dinkster_inference.model_management")
        formats = importlib.import_module("dinkster_inference.latent_formats")
        sparse = importlib.import_module("dinkster_inference.ldm.trellis2.vae")
        latent = materialize_latent(samples)
        coord_resolution = latent.get("coord_resolution")
        resolution = (
            int(coord_resolution) * 16
            if coord_resolution is not None
            else int(vae.first_stage_model.resolution.item())
        )
        values, coords = latent["samples"], latent["coords"]
        estimate = resolution * resolution * 5 // 4
        vae.prepare_decode(
            values.shape,
            memory_required=2 * GIBIBYTE
            + estimate * (896 * management.dtype_size(vae.vae_dtype) + 27 * 4),
        )
        device = management.get_torch_device()
        counts = latent.get("coord_counts")
        items: list[tuple[Any, Any]]
        if counts is None:
            features = values.squeeze(-1).transpose(1, 2)
            items = [(features.reshape(-1, features.shape[-1]), coords)]
        else:
            if counts.ndim != 1:
                raise ValueError(
                    f"Trellis2 coord_counts must be 1D, got shape {tuple(counts.shape)}"
                )
            lengths = [int(count) for count in counts.tolist()]
            if any(count < 0 for count in lengths):
                raise ValueError(f"Trellis2 coord_counts must be non-negative, got {lengths}")
            if sum(lengths) != coords.shape[0]:
                raise ValueError(
                    f"Trellis2 coord_counts total {sum(lengths)} "
                    f"does not match coords rows {coords.shape[0]}"
                )
            features = flatten_sparse_samples(values, coords, counts)
            order = torch.argsort(coords[:, 0].to(torch.int64), stable=True)
            sorted_coords = coords.index_select(0, order)
            items = list(zip(features.split(lengths), sorted_coords.split(lengths), strict=True))

        shape_format = formats.Trellis2ShapeSLAT()
        meshes: list[Any] = []
        stages_per_item: list[Any] = []
        for features, item_coords in items:
            item_coords = item_coords.to(device)
            if counts is not None:
                item_coords = item_coords.clone()
                item_coords[:, 0] = 0
            slat = sparse.SparseTensor(
                feats=shape_format.process_out(features.to(device)), coords=item_coords
            )
            mesh, stages = vae.first_stage_model.decode_shape_slat(
                slat.to(vae.vae_dtype), resolution
            )
            meshes.extend(mesh if counts is None else [mesh[0]])
            stages_per_item.append(stages)
        stages = stages_per_item[0]
        if counts is not None:
            stages = [
                sparse.SparseTensor.from_tensor_list(
                    [item[index].feats for item in stages_per_item],
                    [item[index].coords for item in stages_per_item],
                )
                for index in range(len(stages))
            ]
        vertices: list[Any] = []
        faces: list[Any] = []
        for vertex, face in meshes:
            if latent.get("model_frame", "y_up") == "z_up":
                vertex = torch.stack((vertex[..., 0], vertex[..., 2], -vertex[..., 1]), dim=-1)
            vertices.append(vertex.float().cpu())
            faces.append(face.int().cpu())
        equal = all(v.shape == vertices[0].shape for v in vertices) and all(
            f.shape == faces[0].shape for f in faces
        )
        mesh = TriangleMeshBatch(
            vertices=torch.nn.utils.rnn.pad_sequence(vertices, batch_first=True),
            faces=torch.nn.utils.rnn.pad_sequence(faces, batch_first=True),
            vertex_counts=None if equal else torch.tensor([v.shape[0] for v in vertices]),
            face_counts=None if equal else torch.tensor([f.shape[0] for f in faces]),
        )
        output_device = management.intermediate_device()
        stages = [stage.to(output_device) for stage in materialize_shape_subdivides(stages)]
        return cls.outputs(mesh=mesh, shape_subdivides=stages)
