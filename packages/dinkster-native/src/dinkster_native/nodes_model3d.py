"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from .native_arm_core import (
    _MODEL3D_PROVIDER_SCHEMAS,
    Any,
    AssetRef,
    Mapping,
    NativeComponentHandle,
    Node,
    NodeSchema,
    _not_cancelled,
    _torch,
    cast,
    current_execution_context,
    dataclass,
    default_native_residency,
    default_pool,
    importlib,
    math,
    select_current_device,
    select_load_device,
)
from .native_arm_scheduling import (
    _component_candidate_path,
)
from .nodes_provider import (
    _generation_provider_schema,
)


class GenerationEmptyTrellis2LatentStructure(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.empty_trellis2_latent_structure")

    @classmethod
    def execute(cls, *, batch_size: int = 1) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_empty_trellis2_latent_structure(batch_size=batch_size)
        )


class GenerationTrellis2Conditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.trellis2_conditioning")

    @classmethod
    def execute(cls, *, clip_vision_model: object, image: object) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_trellis2_conditioning(
                clip_vision_model=clip_vision_model,
                image=image,
                pixal3d=False,
                camera_angle_x=49.13,
            )
        )


class GenerationPixal3DConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.pixal3d_conditioning")

    @classmethod
    def execute(
        cls,
        *,
        clip_vision_model: object,
        image: object,
        camera_angle_x: float = 49.13,
    ) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_trellis2_conditioning(
                clip_vision_model=clip_vision_model,
                image=image,
                pixal3d=True,
                camera_angle_x=camera_angle_x,
            )
        )


class GenerationVaeDecodeStructureTrellis2(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vae_decode_structure_trellis2")

    @classmethod
    def execute(
        cls, *, samples: object, vae: object, resolution: str = "32"
    ) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_vae_decode_structure_trellis2(
                samples=samples,
                vae=vae,
                resolution=resolution,
            )
        )


class GenerationTrellis2ShapeStage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.trellis2_shape_stage")

    @classmethod
    def execute(cls, *, positive: object, negative: object, voxel: object) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_trellis2_shape_stage(
                positive=positive,
                negative=negative,
                voxel=voxel,
            )
        )


class GenerationTrellis2UpsampleStage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.trellis2_upsample_stage")

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        shape_latent: object,
        vae: object,
        target_resolution: int = 1024,
    ) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_trellis2_upsample_stage(
                positive=positive,
                negative=negative,
                shape_latent=shape_latent,
                vae=vae,
                target_resolution=target_resolution,
            )
        )


class GenerationVaeDecodeShapeTrellis(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vae_decode_shape_trellis")

    @classmethod
    def execute(cls, *, samples: object, vae: object) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_vae_decode_shape_trellis(samples=samples, vae=vae)
        )


class GenerationTrellis2TextureStage(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.trellis2_texture_stage")

    @classmethod
    def execute(
        cls, *, positive: object, negative: object, shape_latent: object
    ) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_trellis2_texture_stage(
                positive=positive,
                negative=negative,
                shape_latent=shape_latent,
            )
        )


class GenerationVaeDecodeTextureTrellis(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vae_decode_texture_trellis")

    @classmethod
    def execute(
        cls,
        *,
        samples: object,
        vae: object,
        shape_subdivides: object,
    ) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch.trellis2_nodes")
        return cls.outputs(
            **inference_torch.execute_vae_decode_texture_trellis(
                samples=samples,
                vae=vae,
                shape_subdivides=shape_subdivides,
            )
        )


def _model3d_provider_schema(node_type: str) -> NodeSchema:
    try:
        return _MODEL3D_PROVIDER_SCHEMAS[node_type]
    except KeyError as error:
        raise RuntimeError(f"unknown model3d provider schema {node_type!r}") from error


def _enroll_auxiliary_model(
    asset: AssetRef,
    module: object,
    role: str,
    *,
    load_device: object | None = None,
) -> NativeComponentHandle:
    torch = _torch()
    device = select_load_device(torch) if load_device is None else load_device
    coordinator = default_native_residency()
    mechanism = coordinator.enroll_component(
        module,
        load_device=device,
        offload_device=torch.device("cpu"),
    )
    handle = NativeComponentHandle(
        module,
        mechanism,
        device,
        resource_identity=f"native:{role}:{asset.digest}",
        coordinator=coordinator,
    )
    pool = default_pool()
    pool.label(handle, asset.name)
    handle.attach_pool(pool)
    return handle


@dataclass(frozen=True, eq=False)
class _NativeGeometryModel:
    handle: NativeComponentHandle
    compute_dtype: Any
    version: str
    mask_threshold: float
    num_tokens_range: tuple[int, int]

    @property
    def _dinkster_resident_owner(self) -> NativeComponentHandle:
        return self.handle


@dataclass(frozen=True, eq=False)
class _NativeBackgroundRemovalModel:
    handle: NativeComponentHandle
    compute_dtype: Any
    image_size: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]

    @property
    def _dinkster_resident_owner(self) -> NativeComponentHandle:
        return self.handle


def _load_comfy_state_dict(asset: AssetRef) -> dict[str, object]:
    checkpoint = importlib.import_module("dinkster_inference_torch.checkpoint")
    state_dict = checkpoint.load_checkpoint(_component_candidate_path(asset))
    if not isinstance(state_dict, dict):
        raise TypeError(f"{asset.name} must contain a tensor state dictionary")
    return cast("dict[str, object]", state_dict)


def _auxiliary_model_dtypes(
    state_dict: Mapping[str, object],
) -> tuple[Any, Any]:
    torch = _torch()
    storage_dtype = next(
        (
            cast("Any", value).dtype
            for value in state_dict.values()
            if isinstance(value, torch.Tensor) and cast("Any", value).is_floating_point()
        ),
        torch.float32,
    )
    return storage_dtype, torch.float32


def _load_geometry_component(
    state_dict: Mapping[str, object],
) -> tuple[object, Any, str, float, tuple[int, int]]:
    torch = _torch()
    model_module = importlib.import_module("dinkster_inference_torch.moge")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    _, compute_dtype = _auxiliary_model_dtypes(state_dict)
    with torch.device("meta"):
        module = model_module.build_from_state_dict(
            dict(state_dict),
            operations=inference_torch.CastOperations(compute_dtype),
        ).eval()
    version = "v2" if hasattr(module, "encoder") else "v1"
    mask_threshold = float(getattr(module, "mask_threshold", 0.5))
    default_range = (1200, 2500 if version == "v1" else 3600)
    raw_range = getattr(module, "num_tokens_range", default_range)
    token_range = (int(raw_range[0]), int(raw_range[1]))
    return module, compute_dtype, version, mask_threshold, token_range


def _load_background_removal_component(
    state_dict: Mapping[str, object],
    load_device: Any,
) -> tuple[object, Any, int, tuple[float, ...], tuple[float, ...]]:
    torch = _torch()
    model_module = importlib.import_module("dinkster_inference_torch.birefnet")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    storage_dtype, compute_dtype = _auxiliary_model_dtypes(state_dict)
    with torch.device("meta"):
        module = (
            model_module.BiRefNet(
                operations=inference_torch.CastOperations(compute_dtype),
            )
            .to(dtype=storage_dtype)
            .eval()
        )
    module.to_empty(device=torch.device("cpu"))
    module.load_state_dict(dict(state_dict), strict=True)
    # ComfyUI birefnet.json at c67885b: config controls preprocessing, not architecture.
    return module, compute_dtype, 1024, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)


def _triangle_mesh_batch(value: object) -> object:
    inference = importlib.import_module("dinkster_inference")
    if type(value) is inference.TriangleMeshBatch:
        return value
    required = ("vertices", "faces")
    if any(not hasattr(value, name) for name in required):
        raise TypeError("mesh operation did not return a triangle mesh")
    mesh = cast("Any", value)
    return inference.TriangleMeshBatch(
        vertices=mesh.vertices,
        faces=mesh.faces,
        uvs=getattr(mesh, "uvs", None),
        vertex_colors=getattr(mesh, "vertex_colors", None),
        texture=getattr(mesh, "texture", None),
        metallic_roughness=getattr(mesh, "metallic_roughness", None),
        vertex_counts=getattr(mesh, "vertex_counts", None),
        face_counts=getattr(mesh, "face_counts", None),
        unlit=bool(getattr(mesh, "unlit", False)),
        normals=getattr(mesh, "normals", None),
        tangents=getattr(mesh, "tangents", None),
        normal_map=getattr(mesh, "normal_map", None),
        occlusion_in_mr=bool(getattr(mesh, "occlusion_in_mr", False)),
        material=getattr(mesh, "material", None),
        emissive=getattr(mesh, "emissive", None),
    )


def _native_mesh_operation(
    operation: str, *, offload_models: bool = False, **inputs: object
) -> object:
    mesh_operations = importlib.import_module("dinkster_inference_torch.mesh_operations")
    residency = default_native_residency()
    with residency.placement_pass():
        if offload_models:
            torch = _torch()
            device = select_current_device(torch)
            if device.type != "cpu":
                manager = residency.manager
                memory = manager.policy_memory(device).free_total
                loaded = sum(
                    mechanism.loaded_bytes()
                    for mechanism in manager.registered()
                    if mechanism.load_device == device
                )
                manager.free(memory + loaded, device)
                manager.empty_cache(device)
        return getattr(mesh_operations, operation)(**inputs)


class GenerationLoadGeometryModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.load_geometry_model")

    @classmethod
    def execute(cls, *, model: object) -> Mapping[str, object]:
        if not isinstance(model, AssetRef):
            raise TypeError("model must be an AssetRef")
        torch = _torch()
        load_device = select_load_device(torch)
        module, compute_dtype, version, mask_threshold, token_range = _load_geometry_component(
            _load_comfy_state_dict(model),
        )
        handle = _enroll_auxiliary_model(
            model,
            module,
            "geometry",
            load_device=load_device,
        )
        value = _NativeGeometryModel(
            handle,
            compute_dtype,
            version,
            mask_threshold,
            token_range,
        )
        return cls.outputs(model=value)


def _run_geometry_model(
    resource: _NativeGeometryModel,
    image: object,
    resolution_level: int,
    fov_x_degrees: float,
    batch_size: int,
    force_projection: bool,
    apply_mask: bool,
) -> Mapping[str, object]:
    torch = _torch()
    if type(image) is not torch.Tensor:
        raise TypeError("image must be an exact torch.Tensor")
    if not 0 <= resolution_level <= 9:
        raise ValueError("resolution_level must be in [0, 9]")
    if not 0.0 <= fov_x_degrees <= 170.0:
        raise ValueError("fov_x_degrees must be in [0.0, 170.0]")
    if not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be in [1, 64]")
    geometry = importlib.import_module("dinkster_inference_torch.moge_geometry")
    tensor = cast("Any", image)[..., :3]
    bchw = tensor.movedim(-1, -3).contiguous()
    chunks: list[Mapping[str, Any]] = []
    lo, hi = resource.num_tokens_range
    num_tokens = int(lo + (resolution_level / 9) * (hi - lo))
    fov = None if fov_x_degrees <= 0.0 else fov_x_degrees
    with resource.handle.stage(observer_stage="encode"):
        with torch.inference_mode():
            for index in range(0, bchw.shape[0], batch_size):
                source = bchw[index : index + batch_size].to(
                    resource.handle.load_device,
                    dtype=resource.compute_dtype,
                )
                raw = cast("Any", resource.handle.module).forward(
                    source,
                    num_tokens=num_tokens,
                )
                points = raw["points"].float()
                mask = raw["mask"] > resource.mask_threshold
                aspect_ratio = source.shape[-1] / source.shape[-2]
                diagonal = (1 + aspect_ratio**2) ** 0.5
                angle = torch.as_tensor(
                    60.0 if fov is None else fov,
                    device=points.device,
                    dtype=points.dtype,
                )
                requested_focal = aspect_ratio / diagonal / torch.tan(torch.deg2rad(angle / 2))

                if fov is None:
                    focal, shift = geometry.recover_focal_shift(points, mask)
                    bad = ~torch.isfinite(focal) | (focal <= 0)
                    if bool(bad.any()):
                        focal = torch.where(bad, requested_focal, focal)
                        _, shift = geometry.recover_focal_shift(points, mask, focal=focal)
                else:
                    focal = requested_focal.expand(points.shape[0])
                    _, shift = geometry.recover_focal_shift(points, mask, focal=focal)
                focal_diagonal = focal / 2 * diagonal
                half = torch.tensor(0.5, device=points.device, dtype=points.dtype)
                intrinsics = geometry.intrinsics_from_focal_center(
                    focal_diagonal / aspect_ratio,
                    focal_diagonal,
                    half,
                    half,
                )
                points[..., 2] = points[..., 2] + shift[..., None, None]
                if resource.version == "v2":
                    mask = mask & (points[..., 2] > 0)
                depth = points[..., 2].clone()
                if force_projection:
                    points = geometry.depth_map_to_point_map(depth, intrinsics=intrinsics)
                metric_scale = raw.get("metric_scale")
                if metric_scale is not None:
                    points = points * metric_scale[:, None, None, None]
                    depth = depth * metric_scale[:, None, None]
                normal = raw.get("normal")
                if apply_mask:
                    points = torch.where(
                        mask[..., None], points, torch.full_like(points, float("inf"))
                    )
                    depth = torch.where(mask, depth, torch.full_like(depth, float("inf")))
                    if normal is not None:
                        normal = torch.where(mask[..., None], normal, torch.zeros_like(normal))
                chunk: dict[str, Any] = {
                    "points": points,
                    "depth": depth,
                    "intrinsics": intrinsics,
                    "mask": mask,
                }
                if normal is not None:
                    chunk["normal"] = normal
                chunks.append(chunk)

    result: dict[str, object] = {"image": tensor.cpu()}
    for field in ("points", "depth", "intrinsics", "mask", "normal"):
        values = [chunk[field] for chunk in chunks if field in chunk]
        if values:
            result[field] = torch.cat(values, dim=0)
    return result


class GenerationEstimateGeometry(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.estimate_geometry")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        image: object,
        resolution_level: int = 9,
        fov_x_degrees: float = 0.0,
        batch_size: int = 4,
        force_projection: bool = True,
        apply_mask: bool = True,
    ) -> Mapping[str, object]:
        if type(model) is not _NativeGeometryModel:
            raise TypeError("model must be a native geometry model")
        geometry = _run_geometry_model(
            model,
            image,
            resolution_level,
            fov_x_degrees,
            batch_size,
            force_projection,
            apply_mask,
        )
        return cls.outputs(geometry=geometry)


class GenerationGeometryToFOV(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.geometry_to_fov")

    @classmethod
    def execute(
        cls, *, geometry: object, axis: str = "vertical", unit: str = "degrees"
    ) -> Mapping[str, object]:
        if not isinstance(geometry, Mapping):
            raise TypeError("geometry must be a mapping")
        geometry_fields = cast("Mapping[str, object]", geometry)
        intrinsics = geometry_fields.get("intrinsics")
        if intrinsics is None:
            raise ValueError("geometry has no intrinsics")
        matrix = cast("Any", intrinsics)
        if matrix.ndim == 3:
            matrix = matrix[0]
        horizontal = 0.5 / float(matrix[0, 0].item())
        vertical = 0.5 / float(matrix[1, 1].item())
        half_tangent = {
            "horizontal": horizontal,
            "vertical": vertical,
            "diagonal": math.hypot(horizontal, vertical),
        }[axis]
        radians = 2.0 * math.atan(half_tangent)
        fov = radians if unit == "radians" else math.degrees(radians)
        source = next(
            (
                geometry_fields[key]
                for key in ("image", "points", "depth")
                if key in geometry_fields
            ),
            None,
        )
        if source is None:
            raise ValueError("geometry has no image, points, or depth")
        focal_pixels = float(matrix[1, 1].item()) * int(cast("Any", source).shape[1])
        return cls.outputs(fov=fov, focal_pixels=focal_pixels)


class GenerationLoadBackgroundRemoval(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.load_background_removal")

    @classmethod
    def execute(cls, *, model: object) -> Mapping[str, object]:
        if not isinstance(model, AssetRef):
            raise TypeError("model must be an AssetRef")
        torch = _torch()
        load_device = select_load_device(torch)
        module, compute_dtype, image_size, image_mean, image_std = (
            _load_background_removal_component(
                _load_comfy_state_dict(model),
                load_device,
            )
        )
        if len(image_mean) != 3 or len(image_std) != 3:
            raise ValueError("background-removal normalization must have three channels")
        handle = _enroll_auxiliary_model(
            model,
            module,
            "background-removal",
            load_device=load_device,
        )
        value = _NativeBackgroundRemovalModel(
            handle,
            compute_dtype,
            image_size,
            image_mean,
            image_std,
        )
        return cls.outputs(model=value)


class GenerationRemoveBackground(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.remove_background")

    @classmethod
    def execute(cls, *, model: object, image: object) -> Mapping[str, object]:
        if type(model) is not _NativeBackgroundRemovalModel:
            raise TypeError("model must be a native background-removal model")
        resource = model
        torch = _torch()
        if type(image) is not torch.Tensor:
            raise TypeError("image must be an exact torch.Tensor")
        tensor = cast("Any", image)
        preprocess = importlib.import_module(
            "dinkster_inference_torch.image_preprocess"
        ).clip_preprocess
        with resource.handle.stage(observer_stage="encode"):
            with torch.inference_mode():
                pixels = preprocess(
                    tensor.to(resource.handle.load_device),
                    size=resource.image_size,
                    mean=resource.image_mean,
                    std=resource.image_std,
                    crop=False,
                ).to(dtype=resource.compute_dtype)
                component = cast("Any", resource.handle.module)
                if pixels.shape[0] > 1:
                    output = torch.cat(
                        [
                            component(pixel_values=pixels[index : index + 1])
                            for index in range(pixels.shape[0])
                        ],
                        dim=0,
                    )
                else:
                    output = component(pixel_values=pixels)
                output = torch.nn.functional.interpolate(
                    output,
                    size=(tensor.shape[1], tensor.shape[2]),
                    mode="bicubic",
                    antialias=False,
                )
                mask = output.sigmoid().to(device="cpu", dtype=torch.float32).squeeze(1)
        return cls.outputs(mask=mask)


class GenerationImageCropToMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.image_crop_to_mask")

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        masks: object,
        width: int = 1024,
        height: int = 1024,
        pad_factor: float = 1.0,
        grow_mask: int = 0,
        background: str = "#000000",
    ) -> Mapping[str, object]:
        image_crop = importlib.import_module("dinkster_inference_torch.image_crop")
        result = image_crop.crop_images_to_masks(
            images=images,
            masks=masks,
            width=width,
            height=height,
            pad_factor=pad_factor,
            grow_mask=grow_mask,
            background=background,
        )
        return cls.outputs(images=result)


class GenerationPreviewMask(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.preview_mask")

    @classmethod
    def execute(cls, *, mask: object) -> Mapping[str, object]:
        return cls.outputs(mask=mask)


class GenerationVoxelToMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.voxel_to_mesh")

    @classmethod
    def execute(
        cls, *, voxel: object, algorithm: str = "surface net", threshold: float = 0.6
    ) -> Mapping[str, object]:
        mesh = _native_mesh_operation(
            "voxel_grid_to_mesh", voxel=voxel, algorithm=algorithm, threshold=threshold
        )
        return cls.outputs(mesh=mesh)


class GenerationGetMeshInfo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.get_mesh_info")

    @classmethod
    def execute(cls, *, mesh: object) -> Mapping[str, object]:
        mesh_ops = importlib.import_module("dinkster_inference_torch.mesh")
        return cls.outputs(mesh=_triangle_mesh_batch(mesh), info=mesh_ops.mesh_info(mesh))


class GenerationRemeshMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.remesh_mesh")

    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        resolution: int = 512,
        sign_mode: str = "udf",
        qef: bool = False,
        drop_inverted_components: bool = False,
        drop_enclosed_components: bool = False,
        manifold: bool = False,
        band: float = 1.0,
        project_back: float = 0.0,
        fix_poles: bool = False,
        smooth_iters: int = 0,
        drop_small_components: float = 0.01,
        precluster_max_verts: int = 20_000_000,
    ) -> Mapping[str, object]:
        context = current_execution_context()
        result = _native_mesh_operation(
            "remesh_mesh",
            offload_models=True,
            cancelled=_not_cancelled if context is None else context.cancelled,
            mesh=_triangle_mesh_batch(mesh),
            resolution=resolution,
            sign_mode=sign_mode,
            qef=qef,
            drop_inverted_components=drop_inverted_components,
            drop_enclosed_components=drop_enclosed_components,
            manifold=manifold,
            band=band,
            project_back=project_back,
            fix_poles=fix_poles,
            smooth_iters=smooth_iters,
            drop_small_components=drop_small_components,
            precluster_max_verts=precluster_max_verts,
        )
        return cls.outputs(mesh=result)


class GenerationDecimateMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.decimate_mesh")

    @classmethod
    def execute(
        cls, *, mesh: object, target_face_count: int = 200_000, placement_mode: str = "midpoint"
    ) -> Mapping[str, object]:
        result = _native_mesh_operation(
            "decimate_mesh",
            mesh=_triangle_mesh_batch(mesh),
            target_face_count=target_face_count,
            placement_mode=placement_mode,
        )
        return cls.outputs(mesh=result)


class GenerationSmoothMeshNormals(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.smooth_mesh_normals")

    @classmethod
    def execute(cls, *, mesh: object, crease_angle: float = 180.0) -> Mapping[str, object]:
        result = _native_mesh_operation(
            "smooth_mesh_normals",
            mesh=_triangle_mesh_batch(mesh),
            crease_angle=crease_angle,
        )
        return cls.outputs(mesh=result)


class GenerationUnwrapMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.unwrap_mesh")

    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        segmenter: str = "pec",
        resolution: int = 1024,
        padding: int = 1,
        weld_distance: float = 0.0,
    ) -> Mapping[str, object]:
        result = _native_mesh_operation(
            "unwrap_mesh",
            offload_models=True,
            mesh=_triangle_mesh_batch(mesh),
            segmenter=segmenter,
            resolution=resolution,
            padding=padding,
            weld_distance=weld_distance,
        )
        return cls.outputs(mesh=result)


class GenerationPaintMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.paint_mesh")

    @classmethod
    def execute(cls, *, mesh: object, voxel_colors: object) -> Mapping[str, object]:
        result = _native_mesh_operation(
            "paint_mesh", mesh=_triangle_mesh_batch(mesh), voxel_colors=voxel_colors
        )
        return cls.outputs(mesh=result)


class GenerationBakeTextureFromVoxel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.bake_texture_from_voxel")

    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        voxel_colors: object,
        texture_size: int = 2048,
        reference_mesh: object | None = None,
    ) -> Mapping[str, object]:
        base_color, metallic, roughness = cast(
            "tuple[object, object, object]",
            _native_mesh_operation(
                "bake_texture_from_voxel",
                offload_models=True,
                mesh=_triangle_mesh_batch(mesh),
                voxel_colors=voxel_colors,
                texture_size=texture_size,
                reference_mesh=(
                    None if reference_mesh is None else _triangle_mesh_batch(reference_mesh)
                ),
            ),
        )
        return cls.outputs(base_color=base_color, metallic=metallic, roughness=roughness)


class GenerationBakeNormalMapFromMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.bake_normal_map_from_mesh")

    @classmethod
    def execute(
        cls,
        *,
        low_poly: object,
        high_poly: object,
        resolution: int = 1024,
        cage_distance: float = 0.05,
        ignore_backfaces: bool = True,
    ) -> Mapping[str, object]:
        normal_map = _native_mesh_operation(
            "bake_normal_map_from_mesh",
            low_poly=_triangle_mesh_batch(low_poly),
            high_poly=_triangle_mesh_batch(high_poly),
            resolution=resolution,
            cage_distance=cage_distance,
            ignore_backfaces=ignore_backfaces,
        )
        return cls.outputs(normal_map=normal_map)


class GenerationBakeAmbientOcclusion(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.bake_ambient_occlusion")

    @classmethod
    def execute(
        cls,
        *,
        low_poly: object,
        high_poly: object,
        resolution: int = 1024,
        samples: int = 64,
        max_distance: float = 0.5,
        strength: float = 1.0,
        bias: float = 0.01,
    ) -> Mapping[str, object]:
        occlusion = _native_mesh_operation(
            "bake_ambient_occlusion",
            low_poly=_triangle_mesh_batch(low_poly),
            high_poly=_triangle_mesh_batch(high_poly),
            resolution=resolution,
            samples=samples,
            max_distance=max_distance,
            strength=strength,
            bias=bias,
        )
        return cls.outputs(occlusion=occlusion)


class GenerationRenderUVAtlas(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.render_uv_atlas")

    @classmethod
    def execute(cls, *, mesh: object, resolution: int = 1024) -> Mapping[str, object]:
        image = _native_mesh_operation(
            "render_uv_atlas", mesh=_triangle_mesh_batch(mesh), resolution=resolution
        )
        return cls.outputs(image=image)


class GenerationApplyTextureToMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.apply_texture_to_mesh")

    @classmethod
    def execute(
        cls,
        *,
        mesh: object,
        base_color: object,
        metallic: object | None = None,
        roughness: object | None = None,
        occlusion: object | None = None,
        normal_map: object | None = None,
    ) -> Mapping[str, object]:
        result = _native_mesh_operation(
            "apply_texture_to_mesh",
            mesh=_triangle_mesh_batch(mesh),
            base_color=base_color,
            metallic=metallic,
            roughness=roughness,
            occlusion=occlusion,
            normal_map=normal_map,
        )
        return cls.outputs(mesh=result)


class GenerationMeshToModel3D(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.mesh_to_model3d")

    @classmethod
    def execute(cls, *, mesh: object) -> Mapping[str, object]:
        mesh_ops = importlib.import_module("dinkster_inference_torch.mesh")
        glb = mesh_ops.mesh_item_to_glb_bytes(mesh, 0)
        if glb is None:
            raise ValueError("mesh is empty")
        return cls.outputs(model={"format": "glb", "bytes": glb})


class GenerationFile3DToMesh(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _model3d_provider_schema("dinkster.file3d_to_mesh")

    @classmethod
    def execute(cls, *, model_3d: object) -> Mapping[str, object]:
        parser = importlib.import_module("dinkster_inference_torch.mesh_file_io")
        if isinstance(model_3d, AssetRef):
            path = model_3d.local_path()
            data = path.read_bytes()
            format_name = path.suffix
        elif isinstance(model_3d, Mapping):
            model_mapping = cast("Mapping[object, object]", model_3d)
            raw = model_mapping.get("bytes")
            if not isinstance(raw, bytes):
                raise TypeError("model_3d mapping must contain bytes")
            data = raw
            format_name = str(model_mapping.get("format", ""))
        elif callable(getattr(model_3d, "get_bytes", None)):
            file_value = cast("Any", model_3d)
            data = file_value.get_bytes()
            format_name = str(getattr(model_3d, "format", ""))
        else:
            raise TypeError("model_3d must be an AssetRef or 3D file value")
        return cls.outputs(
            mesh=parser.parse_mesh_file(
                data,
                format_name,
            )
        )
