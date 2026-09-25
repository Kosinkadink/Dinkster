"""TRELLIS.2 and Pixal3D generation-node execution."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from dinkster_inference import (
    PBR_CHANNELS,
    TRELLIS2,
    TRELLIS2_SPARSE_DECODE_ALIGNMENT_BYTES,
    TRELLIS2_SPARSE_DECODE_FIXED_BYTES,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    DenseVoxelGrid,
    InferenceComponentHandle,
    PayloadDescriptor,
    PayloadReference,
    ResidentConditioningCarrier,
    ResidentPayloadBinding,
    SparseLatent,
    SparseSubdivisionGuides,
    SparseSupport,
    SparseVolume,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    TriangleMesh,
    TriangleMeshBatch,
    make_conditioning_carrier,
    require_inference_component_handle,
)

from .sparse import make_sparse_support, pack_sparse_latent, unpack_sparse_latent
from .trellis2_assembly import Trellis2VisionModule
from .trellis2_runtime import (
    TRELLIS2_SHAPE_MEAN,
    TRELLIS2_SHAPE_STD,
    TRELLIS2_TEXTURE_MEAN,
    TRELLIS2_TEXTURE_STD,
    Trellis2ConditioningResource,
    Trellis2Frame,
    encode_trellis2_conditioning,
    make_trellis2_conditioning_resources,
    set_trellis2_conditioning_stage,
)

_ORIGIN = (-0.5, -0.5, -0.5)
_SPARSE_DECODE_FIXED_BYTES = TRELLIS2_SPARSE_DECODE_FIXED_BYTES
_SPARSE_DECODE_ALIGNMENT = TRELLIS2_SPARSE_DECODE_ALIGNMENT_BYTES
_MAX_MEMORY_REQUIRED = 2**63 - 1


def _image(value: object) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        image = torch.from_numpy(value)
    elif type(value) is torch.Tensor:
        image = value.detach()
    else:
        raise TypeError("image must be a numpy array or exact torch.Tensor")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] < 1 or image.shape[-1] < 3:
        raise ValueError("image must be a nonempty NHWC tensor with at least three channels")
    if not image.is_floating_point():
        raise TypeError("image must contain floating-point values")
    image = image[..., :3].to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(image).all()):
        raise ValueError("image must contain finite values")
    return image.clamp(0.0, 1.0)


def _source_image_digest(image: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(b"dinkster.trellis2.source-image.v1\0")
    digest.update(str(tuple(image.shape)).encode("ascii"))
    digest.update(image.numpy().tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def _component(value: object, name: str) -> InferenceComponentHandle:
    handle = require_inference_component_handle(value, name)
    if not handle.resource_identity.startswith("native:dinkster.trellis2:"):
        raise TypeError(f"{name} must be a native TRELLIS.2 component")
    return handle


def _conditioning(value: object, name: str) -> Trellis2ConditioningResource:
    if type(value) is ConditioningCarrier:
        bindings = value.bindings
        resource = (
            bindings[0].payload if len(bindings) == 1 and bindings[0].kind == "resident" else None
        )
    elif type(value) is ResidentConditioningCarrier:
        resource = value.payload
    else:
        resource = None
    if type(resource) is not Trellis2ConditioningResource:
        raise TypeError(f"{name} must be TRELLIS.2 conditioning")
    return resource


def _resident_conditioning(
    resource: Trellis2ConditioningResource,
) -> ConditioningCarrier:
    reference_id = "trellis2-prepared-conditioning"
    shape = (1,)
    dtype = "U8"
    space = "trellis2-prepared-conditioning"
    descriptor = PayloadDescriptor(PayloadReference(reference_id), shape, dtype, space)
    conditioning = ConditioningSet(
        (
            ConditioningRecord(
                channels=((ConditioningChannel.VISION_EMBEDDING, descriptor),),
                token_layout=TokenLayoutDescriptor(
                    TRELLIS2.id,
                    1,
                    ("prepared",),
                    (TokenSegmentDescriptor("prepared", "prepared", 0, 1),),
                ),
            ),
        )
    )
    binding = ResidentPayloadBinding(
        reference_id,
        shape,
        dtype,
        space,
        resource,
        resource._dinkster_resident_fingerprint,
    )
    return make_conditioning_carrier(conditioning, (binding,))


def _latent(value: object, name: str) -> SparseLatent[torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a latent mapping")
    samples = cast("Mapping[object, object]", value).get("samples")
    if type(samples) is not SparseLatent:
        raise TypeError(f"{name} samples must be a TRELLIS.2 sparse latent")
    support, features = unpack_sparse_latent(samples)
    return pack_sparse_latent(support, features)


def _frame(value: object) -> Trellis2Frame:
    if not isinstance(value, Mapping):
        raise TypeError("latent must be a mapping")
    frame = cast("Mapping[object, object]", value).get("trellis2_frame")
    if frame not in ("y_up", "z_up"):
        raise ValueError("TRELLIS.2 latent is missing its coordinate frame")
    return frame


def _denormalize(
    latent: SparseLatent[torch.Tensor], mean: tuple[float, ...], std: tuple[float, ...]
) -> SparseLatent[torch.Tensor]:
    support, features = unpack_sparse_latent(latent)
    means = features.new_tensor(mean)
    scales = features.new_tensor(std)
    return pack_sparse_latent(support, features * scales + means)


def _module_dtype(module: Any) -> torch.dtype:
    parameter = next(module.parameters())
    return cast("torch.dtype", parameter.dtype)


def _sparse_decode_memory_required(
    *,
    input_points: int,
    output_points: int,
    guide_points: int,
    element_size: int,
) -> int:
    """Conservatively size sparse decode activations without touching an accelerator."""
    values = (input_points, output_points, guide_points, element_size)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("TRELLIS.2 sparse decode geometry must contain nonnegative integers")
    if element_size not in (2, 4):
        raise ValueError("TRELLIS.2 sparse decode requires 2-byte or 4-byte floating point")

    input_bytes = input_points * 256 * 1024 * element_size
    output_bytes = output_points * 384 * element_size
    guide_bytes = guide_points * 32 * element_size
    required = _SPARSE_DECODE_FIXED_BYTES + max(input_bytes, output_bytes) + guide_bytes
    if required > _MAX_MEMORY_REQUIRED - (_SPARSE_DECODE_ALIGNMENT - 1):
        raise OverflowError("TRELLIS.2 sparse decode memory estimate exceeds int64")
    return (
        (required + _SPARSE_DECODE_ALIGNMENT - 1) // _SPARSE_DECODE_ALIGNMENT
    ) * _SPARSE_DECODE_ALIGNMENT


def _shape_decode_memory_required(sampled: SparseLatent[torch.Tensor], dtype: torch.dtype) -> int:
    return _sparse_decode_memory_required(
        input_points=sampled.support.point_count,
        output_points=0,
        guide_points=0,
        element_size=torch.empty((), dtype=dtype, device="cpu").element_size(),
    )


def _texture_decode_memory_required(
    sampled: SparseLatent[torch.Tensor],
    guides: SparseSubdivisionGuides[torch.Tensor],
    dtype: torch.dtype,
) -> int:
    guide_points = 0
    for level in guides.levels:
        if type(level.features) is not torch.Tensor or level.features.device.type != "cpu":
            raise TypeError("TRELLIS.2 subdivision guides must be CPU tensors before decode")
        guide_points += level.support.point_count
    final = guides.levels[-1]
    output_points = int(torch.count_nonzero(final.features > 0).item())
    return _sparse_decode_memory_required(
        input_points=sampled.support.point_count,
        output_points=output_points,
        guide_points=guide_points,
        element_size=torch.empty((), dtype=dtype, device="cpu").element_size(),
    )


def _move_latent(
    latent: SparseLatent[torch.Tensor], device: object, dtype: torch.dtype
) -> SparseLatent[torch.Tensor]:
    support, features = unpack_sparse_latent(latent)
    coordinates = support.coordinates.to(device=cast("Any", device))
    moved_support = make_sparse_support(
        coordinates,
        support.batch_counts,
        support.resolution,
        support.origin,
        support.voxel_size,
    )
    return pack_sparse_latent(
        moved_support,
        features.to(device=cast("Any", device), dtype=dtype),
    )


def _cpu_support(support: SparseSupport[torch.Tensor]) -> SparseSupport[torch.Tensor]:
    return make_sparse_support(
        support.coordinates.detach().to(device="cpu", dtype=torch.int32).contiguous(),
        support.batch_counts,
        support.resolution,
        support.origin,
        support.voxel_size,
    )


def _cpu_guides(
    guides: SparseSubdivisionGuides[torch.Tensor],
) -> SparseSubdivisionGuides[torch.Tensor]:
    return SparseSubdivisionGuides(
        tuple(
            SparseVolume(
                _cpu_support(level.support),
                level.features.detach().float().cpu(),
                level.channels,
            )
            for level in guides.levels
        )
    )


def _mesh_output(meshes: tuple[TriangleMesh[torch.Tensor], ...]) -> TriangleMeshBatch[torch.Tensor]:
    vertices: list[torch.Tensor] = []
    faces: list[torch.Tensor] = []
    for mesh in meshes:
        current = mesh.vertices.detach().float().cpu()
        if mesh.frame == "z_up":
            current = torch.stack((current[:, 0], current[:, 2], -current[:, 1]), dim=-1)
        vertices.append(current)
        faces.append(mesh.faces.detach().to(device="cpu", dtype=torch.int32))
    vertex_lengths = tuple(item.shape[0] for item in vertices)
    face_lengths = tuple(item.shape[0] for item in faces)
    packed_vertices = vertices[0].new_zeros((len(vertices), max(vertex_lengths), 3))
    packed_faces = faces[0].new_zeros((len(faces), max(face_lengths), 3))
    for index, (item_vertices, item_faces) in enumerate(zip(vertices, faces, strict=True)):
        packed_vertices[index, : item_vertices.shape[0]] = item_vertices
        packed_faces[index, : item_faces.shape[0]] = item_faces
    uniform = len(set(vertex_lengths)) == 1 and len(set(face_lengths)) == 1
    return TriangleMeshBatch(
        packed_vertices,
        packed_faces,
        vertex_counts=None if uniform else torch.tensor(vertex_lengths, dtype=torch.int64),
        face_counts=None if uniform else torch.tensor(face_lengths, dtype=torch.int64),
    )


def execute_empty_trellis2_latent_structure(*, batch_size: int = 1) -> Mapping[str, object]:
    if type(batch_size) is not int or not 1 <= batch_size <= 4096:
        raise ValueError("batch_size must be an integer in [1, 4096]")
    return {
        "latent": {
            "samples": torch.zeros((batch_size, 32, 16, 16, 16), dtype=torch.float32),
            "trellis2_frame": "z_up",
        }
    }


def execute_trellis2_conditioning(
    *,
    clip_vision_model: object,
    image: object,
    pixal3d: bool,
    camera_angle_x: object,
) -> Mapping[str, object]:
    if type(pixal3d) is not bool:
        raise TypeError("pixal3d must be a boolean")
    if not isinstance(camera_angle_x, int | float) or isinstance(camera_angle_x, bool):
        raise TypeError("camera_angle_x must be numeric")
    angle = float(camera_angle_x)
    if not math.isfinite(angle) or not 1.0 <= angle <= 170.0:
        raise ValueError("camera_angle_x must be finite and in [1, 170]")
    handle = _component(clip_vision_model, "clip_vision_model")
    pixels = _image(image)
    with handle.stage(), torch.inference_mode():
        module = handle.component
        if type(module) is not Trellis2VisionModule:
            raise TypeError("clip_vision_model must contain the TRELLIS.2 DINO vision module")
        positive, _negative = encode_trellis2_conditioning(
            module,
            pixels,
            pixal3d=pixal3d,
            camera_angle_x=angle,
            device=cast("Any", handle.load_device),
        )
    positive_resource, negative_resource = make_trellis2_conditioning_resources(
        positive,
        vision_identity=handle.resource_identity,
        source_image_digest=_source_image_digest(pixels),
        camera_angle_x=angle,
    )
    return {
        "positive": _resident_conditioning(positive_resource),
        "negative": _resident_conditioning(negative_resource),
    }


def execute_vae_decode_structure_trellis2(
    *, samples: object, vae: object, resolution: str = "32"
) -> Mapping[str, object]:
    if resolution not in ("32", "64"):
        raise ValueError("resolution must be '32' or '64'")
    if not isinstance(samples, Mapping):
        raise TypeError("samples must be a latent mapping")
    tensor = cast("Mapping[object, object]", samples).get("samples")
    if type(tensor) is not torch.Tensor or tuple(tensor.shape[1:]) != (32, 16, 16, 16):
        raise ValueError("samples must contain a [B,32,16,16,16] structure latent")
    handle = _component(vae, "vae")
    with handle.stage(), torch.inference_mode():
        module = cast("Any", handle.component)
        decode = getattr(module, "decode_structure", None)
        if not callable(decode):
            raise TypeError("vae must contain a TRELLIS.2 structure decoder")
        decoded = (
            cast(
                "torch.Tensor",
                decode(
                    tensor[:, :8].to(
                        device=cast("Any", handle.load_device),
                        dtype=_module_dtype(module),
                    )
                ),
            )
            > 0
        )
    target = int(resolution)
    if decoded.shape[-1] != target:
        ratio = decoded.shape[-1] // target
        decoded = F.max_pool3d(decoded.float(), ratio, ratio) > 0.5
    grid = DenseVoxelGrid(decoded.float().cpu(), ("occupancy",), "z_up")
    return {"voxel": grid}


def execute_trellis2_shape_stage(
    *, positive: object, negative: object, voxel: object
) -> Mapping[str, object]:
    positive_resource = _conditioning(positive, "positive")
    negative_resource = _conditioning(negative, "negative")
    if type(voxel) is DenseVoxelGrid:
        values = voxel.values
    else:
        values = getattr(voxel, "data", None)
        if type(values) is torch.Tensor and values.ndim == 4:
            values = values[:, None]
    if type(values) is not torch.Tensor or values.ndim != 5 or values.shape[1] != 1:
        raise TypeError("voxel must contain one batched dense occupancy channel")
    coordinates = torch.argwhere(values[:, 0].bool()).to(dtype=torch.int32)
    counts = tuple(int((coordinates[:, 0] == batch).sum()) for batch in range(values.shape[0]))
    if not coordinates.numel() or any(count == 0 for count in counts):
        raise ValueError("every TRELLIS.2 structure batch item must contain occupied voxels")
    resolution = int(values.shape[-1])
    support = make_sparse_support(
        coordinates,
        counts,
        resolution,
        _ORIGIN,
        (1.0 / resolution, 1.0 / resolution, 1.0 / resolution),
    )
    stage = "shape-512" if resolution <= 32 else "shape"
    positive_out, negative_out = set_trellis2_conditioning_stage(
        positive_resource,
        negative_resource,
        support,
        stage=stage,
    )
    latent = pack_sparse_latent(support, torch.zeros((support.point_count, 32)))
    return {
        "positive": _resident_conditioning(positive_out),
        "negative": _resident_conditioning(negative_out),
        "latent": {"samples": latent, "trellis2_frame": positive_resource.frame},
    }


def execute_trellis2_upsample_stage(
    *,
    positive: object,
    negative: object,
    shape_latent: object,
    vae: object,
    target_resolution: int = 1024,
) -> Mapping[str, object]:
    positive_resource = _conditioning(positive, "positive")
    negative_resource = _conditioning(negative, "negative")
    if type(target_resolution) is not int or not 1024 <= target_resolution <= 2048:
        raise ValueError("target_resolution must be an integer in [1024, 2048]")
    sampled = _latent(shape_latent, "shape_latent")
    handle = _component(vae, "vae")
    source_resolution = sampled.support.resolution * 16
    grid_resolution = target_resolution // 16
    high_coordinate_batches: list[torch.Tensor] = []
    with handle.stage(), torch.inference_mode():
        module = cast("Any", handle.component)
        upsample = getattr(module, "upsample_shape", None)
        if not callable(upsample):
            raise TypeError("vae must contain a TRELLIS.2 shape decoder")
        normalized = _move_latent(
            _denormalize(sampled, TRELLIS2_SHAPE_MEAN, TRELLIS2_SHAPE_STD),
            handle.load_device,
            _module_dtype(module),
        )
        normalized_coordinates = cast("torch.Tensor", normalized.support.coordinates)
        for batch_index, rows in enumerate(normalized.support.batch_slices):
            item_coordinates = normalized_coordinates[rows].clone()
            item_coordinates[:, 0] = 0
            item_support = make_sparse_support(
                item_coordinates,
                (item_coordinates.shape[0],),
                normalized.support.resolution,
                normalized.support.origin,
                normalized.support.voxel_size,
            )
            item = pack_sparse_latent(item_support, normalized.features[rows])
            high_coordinates = cast("torch.Tensor", upsample(item, upsample_times=4))
            spatial = high_coordinates[:, 1:].float().add(0.5)
            if positive_resource.pixal3d:
                spatial = (spatial * ((grid_resolution - 1) / source_resolution)).round()
            else:
                spatial = torch.floor(spatial * (grid_resolution / source_resolution))
            quantized = torch.cat((high_coordinates[:, :1], spatial), dim=1).to(torch.int32)
            quantized = torch.unique(quantized, dim=0)
            quantized[:, 0] = batch_index
            high_coordinate_batches.append(quantized.cpu())
    coordinates = torch.cat(high_coordinate_batches, dim=0)
    counts = tuple(batch.shape[0] for batch in high_coordinate_batches)
    support = make_sparse_support(
        coordinates,
        counts,
        grid_resolution,
        _ORIGIN,
        (1.0 / grid_resolution, 1.0 / grid_resolution, 1.0 / grid_resolution),
    )
    positive_out, negative_out = set_trellis2_conditioning_stage(
        positive_resource,
        negative_resource,
        support,
        stage="shape",
    )
    latent = pack_sparse_latent(support, torch.zeros((support.point_count, 32)))
    return {
        "positive": _resident_conditioning(positive_out),
        "negative": _resident_conditioning(negative_out),
        "latent": {"samples": latent, "trellis2_frame": positive_resource.frame},
    }


def execute_vae_decode_shape_trellis(*, samples: object, vae: object) -> Mapping[str, object]:
    sampled = _latent(samples, "samples")
    frame = _frame(samples)
    handle = _component(vae, "vae")
    module = cast("Any", handle.component)
    decode = getattr(module, "decode_shape", None)
    if not callable(decode):
        raise TypeError("vae must contain a TRELLIS.2 shape decoder")
    dtype = _module_dtype(module)
    memory_required = _shape_decode_memory_required(sampled, dtype)
    with handle.stage(memory_required=memory_required), torch.inference_mode():
        normalized = _move_latent(
            _denormalize(sampled, TRELLIS2_SHAPE_MEAN, TRELLIS2_SHAPE_STD),
            handle.load_device,
            dtype,
        )
        meshes, guides = cast(
            "tuple[tuple[TriangleMesh[torch.Tensor], ...], SparseSubdivisionGuides[torch.Tensor]]",
            decode(normalized, frame=frame),
        )
    return {"mesh": _mesh_output(meshes), "shape_subdivides": _cpu_guides(guides)}


def execute_trellis2_texture_stage(
    *, positive: object, negative: object, shape_latent: object
) -> Mapping[str, object]:
    positive_resource = _conditioning(positive, "positive")
    negative_resource = _conditioning(negative, "negative")
    sampled = _latent(shape_latent, "shape_latent")
    support, shape_features = unpack_sparse_latent(sampled)
    positive_out, negative_out = set_trellis2_conditioning_stage(
        positive_resource,
        negative_resource,
        support,
        stage="texture",
        shape_features=shape_features,
    )
    latent = pack_sparse_latent(support, torch.zeros_like(shape_features, device="cpu"))
    return {
        "positive": _resident_conditioning(positive_out),
        "negative": _resident_conditioning(negative_out),
        "latent": {"samples": latent, "trellis2_frame": _frame(shape_latent)},
    }


def execute_vae_decode_texture_trellis(
    *, samples: object, vae: object, shape_subdivides: object
) -> Mapping[str, object]:
    sampled = _latent(samples, "samples")
    frame = _frame(samples)
    if type(shape_subdivides) is not SparseSubdivisionGuides:
        raise TypeError("shape_subdivides must be TRELLIS.2 subdivision guides")
    handle = _component(vae, "vae")
    module = cast("Any", handle.component)
    decode = getattr(module, "decode_texture", None)
    if not callable(decode):
        raise TypeError("vae must contain a TRELLIS.2 texture decoder")
    dtype = _module_dtype(module)
    memory_required = _texture_decode_memory_required(sampled, shape_subdivides, dtype)
    with handle.stage(memory_required=memory_required), torch.inference_mode():
        normalized = _move_latent(
            _denormalize(sampled, TRELLIS2_TEXTURE_MEAN, TRELLIS2_TEXTURE_STD),
            handle.load_device,
            dtype,
        )
        guides = SparseSubdivisionGuides(
            tuple(
                SparseVolume(
                    _move_latent(
                        pack_sparse_latent(level.support, level.features),
                        handle.load_device,
                        dtype,
                    ).support,
                    level.features.to(
                        device=cast("Any", handle.load_device),
                        dtype=dtype,
                    ),
                    level.channels,
                )
                for level in shape_subdivides.levels
            )
        )
        volume = cast("SparseVolume[torch.Tensor]", decode(normalized, guides))
    support = _cpu_support(volume.support)
    coordinates = support.coordinates
    if frame == "z_up":
        spatial = coordinates[:, 1:]
        converted = torch.stack(
            (spatial[:, 0], spatial[:, 2], support.resolution - 1 - spatial[:, 1]), dim=1
        )
        coordinates = torch.cat((coordinates[:, :1], converted), dim=1)
        support = make_sparse_support(
            coordinates,
            support.batch_counts,
            support.resolution,
            _ORIGIN,
            support.voxel_size,
        )
    output = SparseVolume(
        support,
        volume.features.detach().float().cpu(),
        PBR_CHANNELS,
    )
    return {"voxel_colors": output}
