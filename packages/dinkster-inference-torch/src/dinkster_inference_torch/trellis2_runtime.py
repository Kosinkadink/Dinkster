"""Conditioning, projection, and custom sampling for TRELLIS.2 and Pixal3D."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from dinkster_inference import (
    TRELLIS2,
    TRELLIS2_SIGMAS,
    CustomSamplingResult,
    DualSamplingGuidance,
    GuidanceRole,
    ModelFamily,
    Parameterization,
    PerpNegSamplingGuidance,
    PreparedMultiStreamConditioning,
    Registry,
    SamplerDescriptor,
    SamplingCancelled,
    SamplingGuidance,
    SchedulerDescriptor,
    SigmaSpace,
    SparseLatent,
    SparseSupport,
)
from PIL import Image

from .denoise import prepare_denoise_mask, to_batch
from .operations import bound_compute_device, module_compute_device
from .parameterizations import calculate_denoised, calculate_input
from .sampling_execution import (
    CustomSamplingCfgValue,
    CustomSamplingCondValue,
    CustomSamplingLatentValue,
    SamplingAdapterContext,
    SamplingDenoiserAdapter,
    SamplingDenoiserExecution,
    SamplingExecutionInputs,
    SamplingExecutionRegistration,
    SamplingLatentAdapter,
    sampling_execution,
)
from .sampling_runtime import DenseOrSparseSamplingRuntime
from .solvers import torch_sampler_registry
from .sparse import authenticate_sparse_support, pack_sparse_latent, unpack_sparse_latent
from .trellis2_assembly import AssembledTrellis2, AssembledTrellis2Vision, Trellis2VisionModule


class Trellis2RuntimeError(ValueError):
    """The requested operation is outside the loaded TRELLIS.2 profile."""


Trellis2Stage = Literal["structure", "shape-512", "shape", "texture"]
Trellis2Frame = Literal["z_up", "y_up"]


@dataclass(frozen=True)
class Trellis2ProjectionMap:
    low_resolution: torch.Tensor
    high_resolution: torch.Tensor | None
    image_resolution: int


@dataclass(frozen=True)
class Trellis2ProjectionPack:
    stages: Mapping[str, Trellis2ProjectionMap]
    transform: torch.Tensor
    camera_angle_x: torch.Tensor
    mesh_scale: torch.Tensor


@dataclass(frozen=True)
class Trellis2Conditioning:
    """Global DINO features and the current stage's optional projected features."""

    global_512: torch.Tensor
    global_1024: torch.Tensor
    stage: Trellis2Stage = "structure"
    projected: torch.Tensor | None = None
    projection_pack: Trellis2ProjectionPack | None = None
    shape_features: torch.Tensor | None = None
    frame: Trellis2Frame = "z_up"


_PROJECTION_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)
_FRONT_VIEW_TRANSFORM = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, -2.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

TRELLIS2_SHAPE_MEAN = (
    0.781296,
    0.018091,
    -0.495192,
    -0.558457,
    1.060530,
    0.093252,
    1.518149,
    -0.933218,
    -0.732996,
    2.604095,
    -0.118341,
    -2.143904,
    0.495076,
    -2.179512,
    -2.130751,
    -0.996944,
    0.261421,
    -2.217463,
    1.260067,
    -0.150213,
    3.790713,
    1.481266,
    -1.046058,
    -1.523667,
    -0.059621,
    2.220780,
    1.621212,
    0.877230,
    0.567247,
    -3.175944,
    -3.186688,
    1.578665,
)
TRELLIS2_SHAPE_STD = (
    5.972266,
    4.706852,
    5.445010,
    5.209927,
    5.320220,
    4.547237,
    5.020802,
    5.444004,
    5.226681,
    5.683095,
    4.831436,
    5.286469,
    5.652043,
    5.367606,
    5.525084,
    4.730578,
    4.805265,
    5.124013,
    5.530808,
    5.619001,
    5.103930,
    5.417670,
    5.269677,
    5.547194,
    5.634698,
    5.235274,
    6.110351,
    5.511298,
    6.237273,
    4.879207,
    5.347008,
    5.405691,
)
TRELLIS2_TEXTURE_MEAN = (
    3.501659,
    2.212398,
    2.226094,
    0.251093,
    -0.026248,
    -0.687364,
    0.439898,
    -0.928075,
    0.029398,
    -0.339596,
    -0.869527,
    1.038479,
    -0.972385,
    0.126042,
    -1.129303,
    0.455149,
    -1.209521,
    2.069067,
    0.544735,
    2.569128,
    -0.323407,
    2.293000,
    -1.925608,
    -1.217717,
    1.213905,
    0.971588,
    -0.023631,
    0.106750,
    2.021786,
    0.250524,
    -0.662387,
    -0.768862,
)
TRELLIS2_TEXTURE_STD = (
    2.665652,
    2.743913,
    2.765121,
    2.595319,
    3.037293,
    2.291316,
    2.144656,
    2.911822,
    2.969419,
    2.501689,
    2.154811,
    3.163343,
    2.621215,
    2.381943,
    3.186697,
    3.021588,
    2.295916,
    3.234985,
    3.233086,
    2.260140,
    2.874801,
    2.810596,
    3.292720,
    2.674999,
    2.680878,
    2.372054,
    2.451546,
    2.353556,
    2.995195,
    2.379849,
    2.786195,
    2.775190,
)


@dataclass(frozen=True, slots=True)
class _Trellis2TensorSeal:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    version: int


def _seal_tensor(value: torch.Tensor) -> _Trellis2TensorSeal:
    return _Trellis2TensorSeal(tuple(value.shape), value.dtype, value.device, value._version)


def _sealed_cpu_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if type(value) is not torch.Tensor or not value.is_floating_point():
        raise TypeError(f"TRELLIS.2 {name} must be an exact floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"TRELLIS.2 {name} must contain finite values")
    return value.detach().to(device="cpu", copy=True).contiguous()


def _tensor_digest(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.contiguous().view(torch.uint8).numpy().tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def _storage_nbytes(tensors: tuple[torch.Tensor, ...]) -> int:
    storages: dict[int, int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages.setdefault(storage.data_ptr(), storage.nbytes())
    return sum(storages.values())


class _Trellis2ConditioningStorage:
    __slots__ = (
        "_camera_angle_x",
        "_global_1024",
        "_global_512",
        "_pid",
        "_pixal3d",
        "_projection_pack",
        "_resident_cost",
        "_seal",
        "_semantic_id",
        "_vision_identity",
    )

    def __init__(
        self,
        conditioning: Trellis2Conditioning,
        *,
        vision_identity: str,
        source_image_digest: str,
        camera_angle_x: float,
    ) -> None:
        if type(conditioning) is not Trellis2Conditioning or conditioning.stage != "structure":
            raise TypeError("TRELLIS.2 conditioning storage requires structure conditioning")
        if type(vision_identity) is not str or not vision_identity.startswith(
            "native:dinkster.trellis2:"
        ):
            raise ValueError("TRELLIS.2 conditioning requires a native vision identity")
        if type(source_image_digest) is not str or not source_image_digest.startswith("sha256:"):
            raise ValueError("TRELLIS.2 conditioning requires a source image sha256")
        self._global_512 = _sealed_cpu_tensor(conditioning.global_512, "512 global features")
        self._global_1024 = _sealed_cpu_tensor(conditioning.global_1024, "1024 global features")
        raw_pack = conditioning.projection_pack
        self._pixal3d = raw_pack is not None
        self._camera_angle_x = float(camera_angle_x)
        if raw_pack is None:
            self._projection_pack = None
        else:
            stages = {
                name: Trellis2ProjectionMap(
                    _sealed_cpu_tensor(stage.low_resolution, f"{name} low projection"),
                    (
                        None
                        if stage.high_resolution is None
                        else _sealed_cpu_tensor(stage.high_resolution, f"{name} high projection")
                    ),
                    stage.image_resolution,
                )
                for name, stage in raw_pack.stages.items()
            }
            self._projection_pack = Trellis2ProjectionPack(
                MappingProxyType(stages),
                _sealed_cpu_tensor(raw_pack.transform, "camera transform"),
                _sealed_cpu_tensor(raw_pack.camera_angle_x, "camera angle"),
                _sealed_cpu_tensor(raw_pack.mesh_scale, "mesh scale"),
            )
        self._vision_identity = vision_identity
        self._pid = os.getpid()
        facts = (
            "dinkster.trellis2.conditioning-resource.v1",
            vision_identity,
            source_image_digest,
            "preprocess=rgb-clamp-lanczos-imagenet-dinov3-v1",
            f"camera-angle-x={self._camera_angle_x.hex()}",
            f"pixal3d={str(self._pixal3d).lower()}",
            f"global-512={tuple(self._global_512.shape)}:{self._global_512.dtype}",
            f"global-1024={tuple(self._global_1024.shape)}:{self._global_1024.dtype}",
        )
        self._semantic_id = (
            "trellis2-conditioning:" + hashlib.sha256("\n".join(facts).encode("utf-8")).hexdigest()
        )
        tensors = tuple(self._tensors())
        self._seal = tuple((tensor, _seal_tensor(tensor)) for tensor in tensors)
        self._resident_cost = MappingProxyType({"ram": _storage_nbytes(tensors)})

    @property
    def _dinkster_resident_fingerprint(self) -> str:
        return self._semantic_id

    @property
    def _dinkster_resident_cost(self) -> Mapping[str, int]:
        return self._resident_cost

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        pack = self._projection_pack
        projections: list[torch.Tensor] = []
        if pack is not None:
            for stage in pack.stages.values():
                projections.append(stage.low_resolution)
                if stage.high_resolution is not None:
                    projections.append(stage.high_resolution)
            projections.extend((pack.transform, pack.camera_angle_x, pack.mesh_scale))
        return (self._global_512, self._global_1024, *projections)

    def validate(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError("TRELLIS.2 conditioning belongs to another process")
        for tensor, seal in self._seal:
            if _seal_tensor(tensor) != seal:
                raise RuntimeError("TRELLIS.2 conditioning storage was mutated")


class _Trellis2StageStorage:
    _stage: Trellis2Stage

    __slots__ = (
        "_base",
        "_batch_counts",
        "_resolution",
        "_resident_cost",
        "_seal",
        "_shape_features",
        "_shape_features_id",
        "_stage",
        "_support_id",
    )

    def __init__(
        self,
        base: _Trellis2ConditioningStorage,
        support: SparseSupport[torch.Tensor],
        stage: Trellis2Stage,
        shape_features: torch.Tensor | None,
    ) -> None:
        if stage not in ("shape-512", "shape", "texture"):
            raise ValueError("TRELLIS.2 sparse stage is invalid")
        authenticated = authenticate_sparse_support(support)
        if stage == "texture" and shape_features is None:
            raise ValueError("TRELLIS.2 texture stage requires shape features")
        if shape_features is not None and tuple(shape_features.shape) != (
            authenticated.point_count,
            32,
        ):
            raise ValueError("TRELLIS.2 shape features must have one 32-channel row per point")
        self._base = base
        self._support_id = authenticated.support_id
        self._batch_counts = authenticated.batch_counts
        self._resolution = authenticated.resolution
        self._stage = stage
        self._shape_features = (
            None if shape_features is None else _sealed_cpu_tensor(shape_features, "shape features")
        )
        self._shape_features_id = (
            None if self._shape_features is None else _tensor_digest(self._shape_features)
        )
        self._seal = None if self._shape_features is None else _seal_tensor(self._shape_features)
        nbytes = 0 if self._shape_features is None else _storage_nbytes((self._shape_features,))
        self._resident_cost = MappingProxyType({"ram": nbytes})

    @property
    def _dinkster_resident_refs(self) -> tuple[object, ...]:
        return (self._base,)

    @property
    def _dinkster_resident_fingerprint(self) -> str:
        shape_features = "none" if self._shape_features_id is None else self._shape_features_id
        return f"{self._base._semantic_id}:{self._stage}:{self._support_id}:{shape_features}"

    @property
    def _dinkster_resident_cost(self) -> Mapping[str, int]:
        return self._resident_cost

    def validate(self, support: SparseSupport[torch.Tensor]) -> None:
        authenticated = authenticate_sparse_support(support)
        self._base.validate()
        if (
            authenticated.support_id != self._support_id
            or authenticated.batch_counts != self._batch_counts
            or authenticated.resolution != self._resolution
        ):
            raise Trellis2RuntimeError(
                "TRELLIS.2 conditioning sparse support differs from the sampling latent"
            )
        if self._shape_features is not None and _seal_tensor(self._shape_features) != self._seal:
            raise RuntimeError("TRELLIS.2 shape conditioning storage was mutated")


class Trellis2ConditioningResource:
    """Process-affine resident conditioning backed by immutable host tensors."""

    __slots__ = ("_lane", "_storage")

    def __init__(
        self,
        storage: _Trellis2ConditioningStorage | _Trellis2StageStorage,
        lane: GuidanceRole,
    ) -> None:
        if type(storage) not in (_Trellis2ConditioningStorage, _Trellis2StageStorage):
            raise TypeError("TRELLIS.2 conditioning resource storage is invalid")
        if lane not in (GuidanceRole.CONDITIONAL, GuidanceRole.UNCONDITIONAL):
            raise ValueError("TRELLIS.2 conditioning resource lane is invalid")
        self._storage = storage
        self._lane = lane

    @property
    def _dinkster_resident_owner(self) -> object:
        return self._storage

    @property
    def _dinkster_resident_fingerprint(self) -> str:
        return self.fingerprint

    @property
    def fingerprint(self) -> str:
        return f"{self._storage._dinkster_resident_fingerprint}:{self._lane.value}"

    def shares_storage_with(self, other: Trellis2ConditioningResource) -> bool:
        return self._storage is other._storage

    @property
    def stage(self) -> Trellis2Stage:
        storage = self._storage
        return storage._stage if isinstance(storage, _Trellis2StageStorage) else "structure"

    @property
    def frame(self) -> Trellis2Frame:
        base = self._base
        return "y_up" if base._pixal3d else "z_up"

    @property
    def vision_identity(self) -> str:
        return self._base._vision_identity

    @property
    def pixal3d(self) -> bool:
        return self._base._pixal3d

    @property
    def support_id(self) -> str | None:
        storage = self._storage
        return storage._support_id if isinstance(storage, _Trellis2StageStorage) else None

    @property
    def guidance_role(self) -> GuidanceRole:
        return self._lane

    def shares_backing(self, other: object) -> bool:
        return type(other) is Trellis2ConditioningResource and self._base is other._base

    @property
    def _base(self) -> _Trellis2ConditioningStorage:
        storage = self._storage
        return storage._base if isinstance(storage, _Trellis2StageStorage) else storage


def make_trellis2_conditioning_resources(
    conditioning: Trellis2Conditioning,
    *,
    vision_identity: str,
    source_image_digest: str,
    camera_angle_x: float,
) -> tuple[Trellis2ConditioningResource, Trellis2ConditioningResource]:
    storage = _Trellis2ConditioningStorage(
        conditioning,
        vision_identity=vision_identity,
        source_image_digest=source_image_digest,
        camera_angle_x=camera_angle_x,
    )
    return (
        Trellis2ConditioningResource(storage, GuidanceRole.CONDITIONAL),
        Trellis2ConditioningResource(storage, GuidanceRole.UNCONDITIONAL),
    )


def set_trellis2_conditioning_stage(
    positive: Trellis2ConditioningResource,
    negative: Trellis2ConditioningResource,
    support: SparseSupport[torch.Tensor],
    *,
    stage: Trellis2Stage,
    shape_features: torch.Tensor | None = None,
) -> tuple[Trellis2ConditioningResource, Trellis2ConditioningResource]:
    if (
        type(positive) is not Trellis2ConditioningResource
        or type(negative) is not Trellis2ConditioningResource
    ):
        raise TypeError("TRELLIS.2 stage inputs must be conditioning resources")
    if positive._base is not negative._base:
        raise ValueError("TRELLIS.2 conditioning lanes must share one backing resource")
    if (
        positive._lane is not GuidanceRole.CONDITIONAL
        or negative._lane is not GuidanceRole.UNCONDITIONAL
    ):
        raise ValueError("TRELLIS.2 conditioning lanes are reversed")
    storage = _Trellis2StageStorage(
        positive._base,
        support,
        stage,
        shape_features,
    )
    return (
        Trellis2ConditioningResource(storage, GuidanceRole.CONDITIONAL),
        Trellis2ConditioningResource(storage, GuidanceRole.UNCONDITIONAL),
    )


def materialize_trellis2_resource(
    resource: Trellis2ConditioningResource,
    *,
    support: SparseSupport[torch.Tensor] | None,
    device: torch.device | str,
) -> Trellis2Conditioning:
    if type(resource) is not Trellis2ConditioningResource:
        raise TypeError("TRELLIS.2 conditioning payload must be a resident resource")
    storage = resource._storage
    base = resource._base
    if isinstance(storage, _Trellis2ConditioningStorage):
        if support is not None:
            raise Trellis2RuntimeError("structure conditioning cannot sample a sparse latent")
        base.validate()
        stage: Trellis2Stage = "structure"
        shape_features = None
    else:
        if support is None:
            raise Trellis2RuntimeError("sparse conditioning requires a sparse latent")
        storage.validate(support)
        stage = storage._stage
        shape_features = (
            None
            if storage._shape_features is None
            else storage._shape_features.to(device=device, copy=True)
        )
    global_512 = base._global_512.to(device=device, copy=True)
    global_1024 = base._global_1024.to(device=device, copy=True)
    if resource._lane is GuidanceRole.UNCONDITIONAL:
        global_512 = torch.zeros_like(global_512)
        global_1024 = torch.zeros_like(global_1024)
    projected = None
    pack = base._projection_pack
    if pack is not None:
        if stage == "structure":
            projected = project_trellis2_features(
                pack,
                "ss",
                dense_resolution=16,
                batch_size=global_512.shape[0],
                device=device,
            )
        else:
            assert support is not None
            projected = project_trellis2_features(
                pack,
                {
                    "shape-512": "shape_512",
                    "shape": "shape_1024",
                    "texture": "texture_1024",
                }[stage],
                coordinates=support.coordinates,
                coordinate_resolution=support.resolution,
                device=device,
            )
        if resource._lane is GuidanceRole.UNCONDITIONAL:
            projected = torch.zeros_like(projected)
    return Trellis2Conditioning(
        global_512,
        global_1024,
        stage,
        projected,
        None,
        shape_features,
        resource.frame,
    )


def _check_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise SamplingCancelled("sampling cancelled")


def build_trellis2_projection_transform(
    distance: torch.Tensor, batch_size: int, *, device: torch.device | str
) -> torch.Tensor:
    transform = torch.tensor(_FRONT_VIEW_TRANSFORM, device=device, dtype=torch.float32)
    transform = transform.unsqueeze(0).expand(batch_size, -1, -1).clone()
    if distance.ndim == 0:
        distance = distance.expand(batch_size)
    transform[:, 1, 3] = -distance.to(device=device, dtype=torch.float32)
    return transform


def _project_points(
    points: torch.Tensor,
    transform: torch.Tensor,
    camera_angle_x: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    batch, count, _ = points.shape
    homogeneous = torch.cat(
        (points, torch.ones((batch, count, 1), device=points.device, dtype=points.dtype)),
        dim=-1,
    )
    camera = torch.bmm(
        homogeneous, torch.linalg.inv(transform.float()).to(transform.dtype).transpose(-2, -1)
    )[..., :3]
    focal = (16.0 / torch.tan(camera_angle_x / 2.0) * resolution / 32.0).to(camera.dtype)
    x_pixel = focal[:, None] * camera[..., 0] / (-camera[..., 2] + 1e-8) + resolution / 2.0
    y_pixel = -focal[:, None] * camera[..., 1] / (-camera[..., 2] + 1e-8) + resolution / 2.0
    return torch.stack((x_pixel, y_pixel), dim=-1)


def _sample_projected(
    feature_map: torch.Tensor, pixels: torch.Tensor, resolution: int
) -> torch.Tensor:
    grid = ((pixels + 0.5) / resolution * 2.0 - 1.0).view(feature_map.shape[0], -1, 1, 2)
    sampled = F.grid_sample(
        feature_map,
        grid.to(feature_map.dtype),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled.squeeze(-1).transpose(1, 2)


def project_trellis2_features(
    pack: Trellis2ProjectionPack,
    stage: str,
    *,
    coordinates: torch.Tensor | None = None,
    coordinate_resolution: int | None = None,
    dense_resolution: int | None = None,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Back-project one Pixal3D stage with the reference's border sampling."""

    projection = pack.stages.get(stage)
    if projection is None:
        raise Trellis2RuntimeError(f"Pixal3D projection pack has no {stage!r} stage")
    selected_device = torch.device(
        device
        if device is not None
        else coordinates.device
        if coordinates is not None
        else pack.mesh_scale.device
    )
    transform = pack.transform.to(selected_device)
    camera_angle = pack.camera_angle_x.to(selected_device)
    mesh_scale = pack.mesh_scale.to(selected_device)
    rotation = torch.tensor(_PROJECTION_ROTATION, device=selected_device, dtype=torch.float32)
    batch_ids: torch.Tensor | None
    if coordinates is not None:
        if coordinate_resolution is None or coordinate_resolution < 1:
            raise Trellis2RuntimeError(
                "sparse projection requires a positive coordinate resolution"
            )
        batch_ids = coordinates[:, 0].long().to(selected_device)
        spatial = coordinates[:, 1:].to(selected_device, dtype=torch.float32)
        normalized = (
            spatial * 0.0
            if coordinate_resolution == 1
            else spatial / (coordinate_resolution - 1) * 2.0 - 1.0
        )
        world = normalized @ rotation.T
        world = world / mesh_scale[batch_ids, None] / 2.0
    else:
        if dense_resolution is None or dense_resolution < 1 or batch_size is None:
            raise Trellis2RuntimeError("dense projection requires resolution and batch size")
        one = torch.linspace(
            -1.0, 1.0, dense_resolution, device=selected_device, dtype=torch.float32
        )
        world = torch.stack(torch.meshgrid(one, one, one, indexing="ij"), dim=-1)
        world = (world.reshape(-1, 3) @ rotation.T)[None].expand(batch_size, -1, -1)
        world = world / mesh_scale[:, None, None] / 2.0
        batch_ids = None

    def back_project(feature_map: torch.Tensor) -> torch.Tensor:
        feature_map = feature_map.to(selected_device)
        if batch_ids is None:
            pixels = _project_points(world, transform, camera_angle, projection.image_resolution)
            return _sample_projected(feature_map, pixels, projection.image_resolution)
        output = feature_map.new_empty((world.shape[0], feature_map.shape[1]))
        for batch in range(transform.shape[0]):
            selected = batch_ids == batch
            if not bool(selected.any()):
                continue
            pixels = _project_points(
                world[selected][None],
                transform[batch : batch + 1],
                camera_angle[batch : batch + 1],
                projection.image_resolution,
            )
            output[selected] = _sample_projected(
                feature_map[batch : batch + 1], pixels, projection.image_resolution
            )[0]
        return output

    low = back_project(projection.low_resolution)
    if projection.high_resolution is None:
        return low
    return torch.cat((low, back_project(projection.high_resolution)), dim=-1)


def _lanczos_resize(image: torch.Tensor, size: int) -> torch.Tensor:
    arrays = image.movedim(1, -1).detach().cpu().float().numpy()
    resized: list[torch.Tensor] = []
    for array in arrays:
        pil = Image.fromarray(np.clip(255.0 * array, 0, 255).astype(np.uint8))
        pil = pil.resize((size, size), resample=Image.Resampling.LANCZOS)
        resized.append(torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).movedim(-1, 0))
    return torch.stack(resized).to(image.device, image.dtype)


def _upsample_naf_features(
    naf: torch.nn.Module,
    pixels: torch.Tensor,
    patches: torch.Tensor,
    source_size: int,
    target: int,
    device: torch.device,
) -> torch.Tensor:
    dtype = torch.float32
    outputs: list[torch.Tensor] = []
    for index in range(pixels.shape[0]):
        image = _lanczos_resize(pixels[index : index + 1], source_size).to(device, dtype=dtype)
        features = patches[index : index + 1].to(device, dtype=dtype)
        output = torch.empty(
            (1, features.shape[1], target, target),
            device="cpu",
            dtype=dtype,
        )
        outputs.append(naf(image, features, (target, target), output=output))
    return torch.cat(outputs)


def encode_trellis2_conditioning(
    vision: AssembledTrellis2Vision | Trellis2VisionModule,
    image: torch.Tensor,
    *,
    pixal3d: bool,
    camera_angle_x: float = 49.13,
    device: torch.device | str | None = None,
) -> tuple[Trellis2Conditioning, Trellis2Conditioning]:
    """Run the official 512/1024 DINO and optional Pixal3D NAF conditioning."""

    if type(image) is not torch.Tensor or image.ndim != 4 or image.shape[-1] < 3:
        raise TypeError("TRELLIS.2 image must be an exact nonempty NHWC torch.Tensor")
    if image.shape[0] < 1 or not image.is_floating_point():
        raise TypeError("TRELLIS.2 image must be a nonempty floating tensor")
    if pixal3d and vision.naf is None:
        raise Trellis2RuntimeError("Pixal3D conditioning requires a vision artifact with NAF")
    selected_device = torch.device(
        device
        if device is not None
        else bound_compute_device(vision.dino.embeddings.patch_embeddings)
        or module_compute_device(vision.dino)
    )
    pixels = image[..., :3].movedim(-1, 1).contiguous().float().clamp(0.0, 1.0)
    mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def encode(size: int) -> tuple[torch.Tensor, torch.Tensor]:
        resized = _lanczos_resize(pixels, size).to(selected_device)
        tokens = vision.dino(
            (resized - mean.to(selected_device)) / std.to(selected_device),
            parameter_free_norm=True,
        )
        patches = (
            tokens[:, 5:].transpose(1, 2).reshape(tokens.shape[0], 1024, size // 16, size // 16)
        )
        return tokens.detach().cpu(), patches.detach().cpu()

    global_512, patch_512 = encode(512)
    global_1024, patch_1024 = encode(1024)
    projection_pack = None
    projected = None
    frame: Trellis2Frame = "z_up"
    if pixal3d:
        naf = vision.naf
        assert naf is not None

        stages = {
            "ss": Trellis2ProjectionMap(patch_512, None, 512),
            "shape_512": Trellis2ProjectionMap(
                patch_512,
                _upsample_naf_features(naf, pixels, patch_512, 512, 512, selected_device),
                512,
            ),
            "shape_1024": Trellis2ProjectionMap(
                patch_1024,
                _upsample_naf_features(naf, pixels, patch_1024, 1024, 512, selected_device),
                1024,
            ),
            "texture_1024": Trellis2ProjectionMap(
                patch_1024,
                _upsample_naf_features(naf, pixels, patch_1024, 1024, 1024, selected_device),
                1024,
            ),
        }
        angle = math.radians(float(camera_angle_x))
        distance = 0.5 / math.tan(angle / 2.0)
        batch = image.shape[0]
        angle_tensor = torch.full((batch,), angle, dtype=torch.float32)
        distance_tensor = torch.full((batch,), distance, dtype=torch.float32)
        projection_pack = Trellis2ProjectionPack(
            stages,
            build_trellis2_projection_transform(distance_tensor, batch, device="cpu"),
            angle_tensor,
            torch.ones(batch, dtype=torch.float32),
        )
        projected = project_trellis2_features(
            projection_pack, "ss", dense_resolution=16, batch_size=batch
        )
        global_512 = global_512[:, :5]
        global_1024 = global_1024[:, :5]
        frame = "y_up"
    positive = Trellis2Conditioning(
        global_512,
        global_1024,
        projected=projected,
        projection_pack=projection_pack,
        frame=frame,
    )
    negative = replace(
        positive,
        global_512=torch.zeros_like(global_512),
        global_1024=torch.zeros_like(global_1024),
        projected=None if projected is None else torch.zeros_like(projected),
    )
    return positive, negative


def trellis2_set_sparse_stage(
    conditioning: Trellis2Conditioning,
    latent: SparseLatent[torch.Tensor],
    *,
    stage: Trellis2Stage,
    shape_features: torch.Tensor | None = None,
) -> Trellis2Conditioning:
    """Attach one authenticated sparse stage to a conditioning value."""

    support = authenticate_sparse_support(latent.support)
    if stage not in ("shape-512", "shape", "texture"):
        raise Trellis2RuntimeError("sparse stage must be shape-512, shape, or texture")
    if stage == "texture" and shape_features is None:
        raise Trellis2RuntimeError("texture conditioning requires sampled shape features")
    if shape_features is not None and tuple(shape_features.shape) != (support.point_count, 32):
        raise Trellis2RuntimeError("shape features must have one 32-channel row per sparse point")
    projected = None
    if conditioning.projection_pack is not None:
        stage_name = {
            "shape-512": "shape_512",
            "shape": "shape_1024",
            "texture": "texture_1024",
        }[stage]
        projected = project_trellis2_features(
            conditioning.projection_pack,
            stage_name,
            coordinates=support.coordinates,
            coordinate_resolution=support.resolution,
        )
    return replace(
        conditioning,
        stage=stage,
        projected=projected,
        shape_features=shape_features,
    )


@dataclass(frozen=True, slots=True)
class _ModelConditioning:
    global_features: torch.Tensor
    projected: torch.Tensor | None
    shape_features: torch.Tensor | None
    stage: Trellis2Stage


@dataclass(frozen=True)
class _Trellis2LatentContext:
    support: SparseSupport[torch.Tensor] | None
    stage: Trellis2Stage


@dataclass(frozen=True)
class _Trellis2LatentAdapter:
    def prepare(
        self,
        runtime: object,
        family: ModelFamily,
        *,
        latent: CustomSamplingLatentValue,
        noise: CustomSamplingLatentValue,
        cond: CustomSamplingCondValue,
        cfg: CustomSamplingCfgValue,
        denoise_mask: CustomSamplingLatentValue | None,
        context: SamplingAdapterContext,
        error: type[Exception],
    ) -> SamplingExecutionInputs:
        del family, error
        owner = cast("Trellis2DiffusionRuntime", runtime)
        if context.options:
            names = ", ".join(sorted(context.options))
            raise Trellis2RuntimeError(
                f"TRELLIS.2 sampling does not accept adapter options: {names}"
            )
        if type(cond) is not PreparedMultiStreamConditioning:
            raise Trellis2RuntimeError("TRELLIS.2 requires prepared conditioning")
        if cond.runtime_identity != owner.conditioning_identity:
            raise Trellis2RuntimeError("TRELLIS.2 conditioning identity differs from the runtime")
        if type(cond.payload) not in (Trellis2Conditioning, Trellis2ConditioningResource):
            raise TypeError("TRELLIS.2 conditioning payload has the wrong type")
        conditioning = cast("Trellis2Conditioning | Trellis2ConditioningResource", cond.payload)
        stage = conditioning.stage
        if isinstance(cfg, (DualSamplingGuidance, PerpNegSamplingGuidance)):
            raise Trellis2RuntimeError("TRELLIS.2 supports ordinary CFG guidance only")
        uncond_value = None
        if cfg is not None and cfg.uncond is not None:
            uncond = cfg.uncond
            if (
                type(uncond) is not PreparedMultiStreamConditioning
                or uncond.runtime_identity != owner.conditioning_identity
                or type(uncond.payload) not in (Trellis2Conditioning, Trellis2ConditioningResource)
            ):
                raise Trellis2RuntimeError("TRELLIS.2 negative conditioning is incompatible")
            uncond_value = cast(
                "Trellis2Conditioning | Trellis2ConditioningResource", uncond.payload
            )
            if uncond_value.stage != stage:
                raise Trellis2RuntimeError("TRELLIS.2 guidance lanes must use the same stage")
        guidance_cfg = (
            None
            if cfg is None
            else replace(cast("SamplingGuidance[object]", cfg), uncond=uncond_value)
        )
        sparse = type(latent) is SparseLatent
        if sparse != (type(noise) is SparseLatent):
            raise Trellis2RuntimeError("TRELLIS.2 latent and noise must share one representation")
        if sparse:
            support, latent_tensor = unpack_sparse_latent(latent)
            noise_support, noise_tensor = unpack_sparse_latent(noise)
            if not support.same_support(noise_support):
                raise Trellis2RuntimeError("TRELLIS.2 sparse noise must use the latent support")
            if stage == "structure":
                raise Trellis2RuntimeError("TRELLIS.2 sparse latent needs a sparse stage")
        else:
            support = None
            if type(latent) is not torch.Tensor or type(noise) is not torch.Tensor:
                raise Trellis2RuntimeError(
                    "TRELLIS.2 structure sampling requires tensor latent and noise"
                )
            latent_tensor = latent
            noise_tensor = noise
            if stage != "structure":
                raise Trellis2RuntimeError("TRELLIS.2 dense latent requires structure conditioning")
            if tuple(latent_tensor.shape[1:]) != (32, 16, 16, 16):
                raise Trellis2RuntimeError("TRELLIS.2 structure latent must be [B,32,16,16,16]")
        mask_tensor = None
        if denoise_mask is not None:
            if support is not None:
                mask_support, mask_tensor = unpack_sparse_latent(denoise_mask)
                if not support.same_support(mask_support):
                    raise Trellis2RuntimeError("sparse mask must use the latent support")
                mask_tensor = torch.broadcast_to(mask_tensor, latent_tensor.shape)
            elif type(denoise_mask) is torch.Tensor:
                mask_tensor = prepare_denoise_mask(denoise_mask, latent_tensor)
            else:
                raise TypeError("dense sampling requires a dense tensor mask")
        return SamplingExecutionInputs(
            latent_tensor,
            noise_tensor,
            conditioning,
            guidance_cfg,
            mask_tensor,
            _Trellis2LatentContext(support, stage),
        )

    def finish(
        self,
        inputs: SamplingExecutionInputs,
        output: torch.Tensor,
        denoised: object | None,
    ) -> CustomSamplingResult[Any]:
        context = cast("_Trellis2LatentContext", inputs.latent_context)
        if context.support is None:
            if denoised is not None and type(denoised) is not torch.Tensor:
                raise TypeError("TRELLIS.2 dense denoised state must contain a tensor")
            return CustomSamplingResult(output, denoised)
        result = pack_sparse_latent(context.support, output)
        if denoised is None:
            return CustomSamplingResult(result, None)
        if type(denoised) is not SparseLatent:
            raise TypeError("TRELLIS.2 sparse denoised state must contain a SparseLatent")
        return CustomSamplingResult(result, denoised)


class _Trellis2SamplingDenoiser:
    evaluator_identity = "dinkster.trellis2.conditioning.v1"

    def __init__(
        self,
        owner: Trellis2DiffusionRuntime,
        support: SparseSupport[torch.Tensor] | None,
        stage: Trellis2Stage,
        *,
        device: torch.device | str,
        compute_dtype: torch.dtype,
        cancelled: Callable[[], bool],
    ) -> None:
        self.owner = owner
        self.support = support
        self.stage = stage
        self.device = device
        self.compute_dtype = compute_dtype
        self.cancelled = cancelled

    def _materialize(self, value: object) -> Trellis2Conditioning:
        if type(value) is Trellis2ConditioningResource:
            return materialize_trellis2_resource(value, support=self.support, device=self.device)
        if type(value) is Trellis2Conditioning:
            return value
        raise Trellis2RuntimeError("TRELLIS.2 guidance lanes must use the same stage")

    def prepare_conditioning(self, value: object, _role: GuidanceRole) -> _ModelConditioning:
        _check_cancelled(self.cancelled)
        value = self._materialize(value)
        if value.stage != self.stage:
            raise Trellis2RuntimeError("TRELLIS.2 guidance lanes must use the same stage")
        global_features = (
            value.global_512 if value.stage in ("structure", "shape-512") else value.global_1024
        ).to(device=self.device, dtype=self.compute_dtype)
        projected = (
            None
            if value.projected is None
            else value.projected.to(device=self.device, dtype=self.compute_dtype)
        )
        shape_features = (
            None
            if value.shape_features is None
            else value.shape_features.to(device=self.device, dtype=self.compute_dtype)
        )
        return _ModelConditioning(global_features, projected, shape_features, value.stage)

    def evaluate_conditioning(
        self, value: torch.Tensor, sigma: float, condition: _ModelConditioning
    ) -> torch.Tensor:
        _check_cancelled(self.cancelled)
        support = self.support
        batch = value.shape[0] if support is None else support.batch_size
        model_input = calculate_input(Parameterization.FLOW, sigma, value).to(self.compute_dtype)
        if support is None:
            model_latent: torch.Tensor | SparseLatent[torch.Tensor] = model_input
        else:
            features = model_input
            if condition.stage == "texture":
                if condition.shape_features is None:
                    raise Trellis2RuntimeError("texture stage requires shape features")
                features = torch.cat((features, condition.shape_features), dim=-1)
            model_latent = pack_sparse_latent(support, features)
        timestep = torch.full(
            (batch,), TRELLIS2_SIGMAS.timestep(sigma), device=self.device, dtype=torch.float32
        )
        result = self.owner._model(
            condition.stage.removesuffix("-512"),
            model_latent,
            timestep,
            to_batch(condition.global_features, batch),
            projected=condition.projected,
            first_shape_pass=condition.stage == "shape-512",
            low_resolution_texture=(
                condition.stage == "texture" and support is not None and support.resolution <= 32
            ),
        )
        velocity = (
            unpack_sparse_latent(result)[1]
            if type(result) is SparseLatent
            else cast("torch.Tensor", result)
        ).float()
        return calculate_denoised(Parameterization.FLOW, sigma, velocity, value)

    def batchable(self, values: tuple[_ModelConditioning, ...]) -> bool:
        if self.support is not None or not values:
            return False
        first = next(iter(values))
        return all(
            value.stage == first.stage
            and value.global_features.shape == first.global_features.shape
            and (value.projected is None) == (first.projected is None)
            and (
                value.projected is None
                or first.projected is not None
                and value.projected.shape == first.projected.shape
            )
            for value in values[1:]
        )

    def evaluate_conditioning_batch(
        self, value: torch.Tensor, sigma: float, values: tuple[_ModelConditioning, ...]
    ) -> tuple[torch.Tensor, ...]:
        if not values or not self.batchable(values):
            raise Trellis2RuntimeError("TRELLIS.2 conditioning batch is incompatible")
        first = next(iter(values))
        batch = value.shape[0]
        model_input = torch.cat(
            (calculate_input(Parameterization.FLOW, sigma, value).to(self.compute_dtype),)
            * len(values)
        )
        timestep = torch.full(
            (batch * len(values),),
            TRELLIS2_SIGMAS.timestep(sigma),
            device=self.device,
            dtype=torch.float32,
        )
        global_features = torch.cat(
            tuple(to_batch(condition.global_features, batch) for condition in values)
        )
        projected = (
            None
            if first.projected is None
            else torch.cat(
                tuple(
                    to_batch(cast("torch.Tensor", condition.projected), batch)
                    for condition in values
                )
            )
        )
        result = cast(
            "torch.Tensor",
            self.owner._model(
                first.stage.removesuffix("-512"),
                model_input,
                timestep,
                global_features,
                projected=projected,
                first_shape_pass=first.stage == "shape-512",
                low_resolution_texture=False,
            ),
        ).float()
        repeated = torch.cat((value,) * len(values))
        return tuple(
            calculate_denoised(Parameterization.FLOW, sigma, result, repeated).chunk(len(values))
        )


def _trellis2_stage_model(owner: Trellis2DiffusionRuntime, stage: Trellis2Stage) -> torch.nn.Module:
    return {
        "structure": owner._model.structure,
        "shape-512": owner._model.shape_512,
        "shape": owner._model.shape,
        "texture": owner._model.texture,
    }[stage]


def _trellis2_device_from_inputs(runtime: object, inputs: SamplingExecutionInputs) -> torch.device:
    owner = cast("Trellis2DiffusionRuntime", runtime)
    stage = cast("_Trellis2LatentContext", inputs.latent_context).stage
    model = cast("Any", _trellis2_stage_model(owner, stage))
    return bound_compute_device(model.input_layer) or model.input_layer.weight.device


def _trellis2_denoiser(
    runtime: object, compute_dtype: torch.dtype, context: SamplingAdapterContext
) -> SamplingDenoiserExecution:
    owner = cast("Trellis2DiffusionRuntime", runtime)
    if context.inputs is None or context.device is None:
        raise RuntimeError("TRELLIS.2 sampling context is unresolved")
    latent_context = cast("_Trellis2LatentContext", context.inputs.latent_context)
    support = latent_context.support
    if support is not None:
        support = replace(support, coordinates=support.coordinates.to(context.device))
    evaluator = _Trellis2SamplingDenoiser(
        owner,
        support,
        latent_context.stage,
        device=context.device,
        compute_dtype=compute_dtype,
        cancelled=context.cancelled,
    )

    def unpack_state(value: torch.Tensor) -> object:
        return value if support is None else pack_sparse_latent(support, value)

    return SamplingDenoiserExecution(
        cast("SamplingDenoiserAdapter", evaluator),
        process_in=lambda value: value,
        process_out=lambda value: value,
        unpack_state=unpack_state,
        denoise_mask_prepared=True,
    )


def _trellis2_compute_dtype(runtime: object) -> torch.dtype:
    return cast("Trellis2DiffusionRuntime", runtime)._compute_dtype


class Trellis2DiffusionRuntime(DenseOrSparseSamplingRuntime):
    """FLOW sampling for all dense and sparse TRELLIS.2 stages."""

    sampling_error = Trellis2RuntimeError
    supports_denoised_capture = True
    sampling_execution_registration = SamplingExecutionRegistration(
        latent=cast("SamplingLatentAdapter", _Trellis2LatentAdapter()),
        denoiser=_trellis2_denoiser,
        device=lambda _runtime: None,
        device_from_inputs=_trellis2_device_from_inputs,
        compute_dtype=_trellis2_compute_dtype,
        flow=True,
    )

    def __init__(
        self,
        assembled: AssembledTrellis2,
        *,
        runtime_identity: str,
        compute_dtype: torch.dtype,
        sampler_registry: Registry[SamplerDescriptor[torch.Tensor]] | None = None,
        scheduler_registry: Registry[SchedulerDescriptor] | None = None,
    ) -> None:
        if not runtime_identity:
            raise ValueError("TRELLIS.2 runtime identity must be nonempty")
        self.assembled = assembled
        self._model = assembled.diffusion
        self._runtime_identity = runtime_identity
        self._compute_dtype = compute_dtype
        self._samplers = torch_sampler_registry(sampler_registry)
        from .schedules import torch_scheduler_registry

        self._schedulers = (
            torch_scheduler_registry() if scheduler_registry is None else scheduler_registry
        )
        self._guidance = None

    @property
    def family(self) -> ModelFamily:
        return TRELLIS2

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    @property
    def conditioning_identity(self) -> str:
        texture_512 = self._model.texture_512
        profiles = (
            self._model.structure.config.image_attention,
            self._model.shape.config.image_attention,
            self._model.shape_512.config.image_attention,
            self._model.texture.config.image_attention,
            (
                self._model.texture.config.image_attention
                if texture_512 is None
                else texture_512.config.image_attention
            ),
        )
        return "dinkster.trellis2.conditioning:v2:" + ":".join(profiles)

    def _sampling_sigma_space(self, sampling_shift: float | None) -> SigmaSpace:
        return TRELLIS2_SIGMAS

    sample_custom = cast("Any", sampling_execution)  # noqa: F811


__all__ = [
    "TRELLIS2_SHAPE_MEAN",
    "TRELLIS2_SHAPE_STD",
    "TRELLIS2_TEXTURE_MEAN",
    "TRELLIS2_TEXTURE_STD",
    "Trellis2Conditioning",
    "Trellis2ConditioningResource",
    "Trellis2DiffusionRuntime",
    "Trellis2Frame",
    "Trellis2ProjectionMap",
    "Trellis2ProjectionPack",
    "Trellis2RuntimeError",
    "Trellis2Stage",
    "build_trellis2_projection_transform",
    "encode_trellis2_conditioning",
    "make_trellis2_conditioning_resources",
    "materialize_trellis2_resource",
    "project_trellis2_features",
    "set_trellis2_conditioning_stage",
    "trellis2_set_sparse_stage",
]
