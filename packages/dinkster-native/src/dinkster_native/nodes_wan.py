"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from .native_arm_core import (
    _NATIVE_PREPARED_CONDITIONING_KEY,
    _NATIVE_PROMPT_KEY,
    WAN_CAMERA_POSES,
    Any,
    BerniniConditioning,
    CLIPTextEncode,
    Mapping,
    NativeRuntimeHandle,
    Wan21ClipVisionEncode,
    Wan21ImageToVideo,
    Wan22FunControlToVideo,
    Wan22ImageToVideoLatent,
    WanCameraEmbedding,
    WanCameraImageToVideo,
    WanFirstLastFrameToVideo,
    WanFunControlToVideo,
    WanFunInpaintToVideo,
    WanMoveConcatTrack,
    WanMoveGenerateTracks,
    WanMoveTracksFromCoords,
    WanMoveTrackToVideo,
    WanMoveVisualizeTracks,
    WanPhantomSubjectToVideo,
    WanTrackToVideo,
    WanVaceToVideo,
    _is_accelerator_oom,
    _retry_tiled_vae_after_oom,
    _torch,
    cast,
    dataclass,
    importlib,
    json,
    math,
)
from .native_arm_runtime import (
    _native_handle,
)
from .nodes_minimax import _prepared_multistream_conditioning


class NativeClipTextEncode(CLIPTextEncode):
    """Encode text through the runtime and emit Comfy conditioning data."""

    @classmethod
    def execute(cls, *, text: str, clip: object) -> Mapping[str, object]:
        handle = _native_handle(clip, "clip")
        torch = _torch()
        with handle.stage("text"):
            with torch.inference_mode():
                conditioning = handle.runtime.encode_text(text)
        prepare_text = getattr(handle.runtime, "prepare_text_conditioning", None)
        if callable(prepare_text):
            inference = importlib.import_module("dinkster_inference")
            prepared = prepare_text(conditioning)
            value = inference.PreparedMultiStreamConditioning(
                handle.runtime.conditioning_identity,
                prepared,
            )
            wire: list[list[object]] = [[value, {}]]
            return cls.outputs(conditioning=wire)
        if getattr(handle.runtime, "preserves_text_conditioning", False):
            return cls.outputs(
                conditioning=[
                    [conditioning.embeddings, {_NATIVE_PREPARED_CONDITIONING_KEY: conditioning}]
                ]
            )
        metadata: dict[str, object] = {}
        if conditioning.pooled is not None:
            metadata["pooled_output"] = conditioning.pooled
        metadata[_NATIVE_PROMPT_KEY] = (text, handle.recipe.runtime_identity)
        return cls.outputs(conditioning=[[conditioning.embeddings, metadata]])


@dataclass(frozen=True, slots=True)
class _Wan21ClipVisionOutput:
    conditioning_identity: str
    embedding: object


class NativeWan21ClipVisionEncode(Wan21ClipVisionEncode):
    @classmethod
    def execute(cls, *, model: object, image: object) -> Mapping[str, object]:
        handle = _native_handle(model, "model")
        assembled = handle.runtime.assembled
        if assembled.diffusion.config.model_type != "i2v":
            raise ValueError("model must use the Wan 2.1 I2V profile")
        if assembled.clip_vision is None:
            raise ValueError("loaded Wan 2.1 I2V model has no CLIP vision encoder")
        torch = _torch()
        if type(image) is not torch.Tensor:
            raise TypeError("image must be an exact torch.Tensor")
        tensor = cast("Any", image)
        with handle.stage("vision"):
            with torch.inference_mode():
                embedding = handle.runtime.encode_vision(tensor.to(handle.load_device))
        return cls.outputs(
            clip_vision_output=_Wan21ClipVisionOutput(
                handle.runtime.conditioning_identity,
                embedding,
            )
        )


def _wan21_i2v_prepared(
    value: object,
    *,
    name: str,
    handle: NativeRuntimeHandle,
    inference: Any,
    concat_latent: Any,
    vision: Any = None,
) -> list[list[object]]:
    text = _prepared_multistream_conditioning(
        value,
        inference,
        name,
        handle.runtime.conditioning_identity,
    )
    if text is None:
        raise TypeError(f"{name} must contain Wan 2.1 text conditioning")
    prepared = handle.runtime.prepare_i2v_conditioning(text, concat_latent, vision)
    return [
        [
            inference.PreparedMultiStreamConditioning(
                handle.runtime.conditioning_identity,
                prepared,
            ),
            {},
        ]
    ]


def _wan21_clip_embedding(
    value: object,
    *,
    name: str,
    handle: NativeRuntimeHandle,
    torch: Any,
) -> Any:
    if type(value) is not _Wan21ClipVisionOutput:
        raise TypeError(f"{name} must come from native Wan 2.1 CLIP Vision Encode")
    output = value
    if output.conditioning_identity != handle.runtime.conditioning_identity:
        raise ValueError(f"{name} is incompatible with the native Wan profile")
    if type(output.embedding) is not torch.Tensor:
        raise TypeError(f"{name} embedding must be an exact torch.Tensor")
    return output.embedding


def _wan_bernini_prepared(
    value: object,
    *,
    name: str,
    handle: NativeRuntimeHandle,
    inference: Any,
    context_latents: tuple[Any, ...],
) -> list[list[object]]:
    text = _prepared_multistream_conditioning(
        value,
        inference,
        name,
        handle.runtime.conditioning_identity,
    )
    if text is None:
        raise TypeError(f"{name} must contain Wan text conditioning")
    prepared = handle.runtime.prepare_bernini_conditioning(text, context_latents)
    return [
        [
            inference.PreparedMultiStreamConditioning(
                handle.runtime.conditioning_identity,
                prepared,
            ),
            {},
        ]
    ]


class NativeBerniniConditioning(BerniniConditioning):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        source_video: object = None,
        reference_video: object = None,
        reference_images: object = None,
        ref_max_size: int = 848,
    ) -> Mapping[str, object]:
        for name, value in (
            ("width", width),
            ("height", height),
            ("ref_max_size", ref_max_size),
        ):
            if type(value) is not int or not 16 <= value <= 8192 or value % 16:
                raise ValueError(f"{name} must be a multiple of 16 between 16 and 8192")
        if type(length) is not int or not 1 <= length <= 8192 or (length - 1) % 4:
            raise ValueError("length must be 1 plus a multiple of 4 between 1 and 8192")
        if type(batch_size) is not int or not 1 <= batch_size <= 4096:
            raise ValueError("batch_size must be between 1 and 4096")

        handle = _native_handle(vae, "vae")
        config = handle.runtime.assembled.diffusion.config
        if getattr(config, "model_variant", None) != "bernini" or not callable(
            getattr(handle.runtime, "prepare_bernini_conditioning", None)
        ):
            raise ValueError("vae must come from a Wan 2.2 Bernini 14B profile")

        inference = importlib.import_module("dinkster_inference")
        torch = _torch()
        common_upscale = importlib.import_module("dinkster_inference_torch.resize").common_upscale
        reference_items: tuple[tuple[str, object], ...] = ()
        if reference_images is not None:
            if not isinstance(reference_images, Mapping):
                raise TypeError("reference_images must be a mapping")
            references = cast("Mapping[object, object]", reference_images)
            if len(references) > 8:
                raise ValueError("reference_images supports at most 8 slots")
            if any(
                type(name) is not str or not name.startswith("reference_image_")
                for name in references
            ):
                raise ValueError("reference_images keys must use the reference_image_ prefix")
            reference_items = tuple(
                (name, references[name])
                for name in sorted(cast("Mapping[str, object]", references))
                if references[name] is not None
            )

        def encode_reference(value: object, name: str) -> Any:
            image = _wan_image(value, name, torch)
            image_height = int(image.shape[1])
            image_width = int(image.shape[2])
            scale = min(ref_max_size / max(image_height, image_width), 1.0)
            resized_height = max(16, round(image_height * scale / 16) * 16)
            resized_width = max(16, round(image_width * scale / 16) * 16)
            resized = common_upscale(
                image[:, :, :, :3].movedim(-1, 1),
                resized_width,
                resized_height,
                "area",
                "disabled",
            ).movedim(1, -1)
            frames = int(resized.shape[0])
            content = resized.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
            return _wan_encode_content(
                handle=handle,
                content=content,
                expected=(
                    1,
                    16,
                    ((frames - 1) // 4) + 1,
                    resized_height // 8,
                    resized_width // 8,
                ),
                name=name,
                torch=torch,
            )

        context_latents: list[Any] = []
        if source_video is not None or reference_video is not None or reference_items:
            with handle.stage("vae", unload_before=("text",)):
                with torch.inference_mode():
                    if source_video is not None:
                        source = _wan_image(source_video, "source_video", torch)
                        resized = common_upscale(
                            source[:length, :, :, :3].movedim(-1, 1),
                            width,
                            height,
                            "area",
                            "center",
                        ).movedim(1, -1)
                        frames = int(resized.shape[0])
                        content = resized.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
                        context_latents.append(
                            _wan_encode_content(
                                handle=handle,
                                content=content,
                                expected=(
                                    1,
                                    16,
                                    ((frames - 1) // 4) + 1,
                                    height // 8,
                                    width // 8,
                                ),
                                name="source video",
                                torch=torch,
                            )
                        )
                        del content, resized, source
                    if reference_video is not None:
                        video = _wan_image(reference_video, "reference_video", torch)
                        context_latents.append(encode_reference(video[:length], "reference video"))
                        del video
                    for name, images in reference_items:
                        validated = _wan_image(images, name, torch)
                        for index in range(validated.shape[0]):
                            context_latents.append(
                                encode_reference(
                                    validated[index : index + 1],
                                    f"{name} frame {index}",
                                )
                            )
                        del validated

        prepared_positive = _wan_bernini_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            context_latents=tuple(context_latents),
        )
        prepared_negative = _wan_bernini_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            context_latents=tuple(context_latents),
        )
        latent = torch.zeros(
            (batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8),
            device="cpu",
            dtype=torch.float32,
        )
        return cls.outputs(
            positive=prepared_positive,
            negative=prepared_negative,
            latent={"samples": latent},
        )


class NativeWan21ImageToVideo(Wan21ImageToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        model: object,
        start_image: object,
        clip_vision_output: object = None,
        width: int,
        height: int,
        length: int,
        batch_size: int,
    ) -> Mapping[str, object]:
        if width < 16 or width > 16384 or width % 16 != 0:
            raise ValueError("width must be a multiple of 16 between 16 and 16384")
        if height < 16 or height > 16384 or height % 16 != 0:
            raise ValueError("height must be a multiple of 16 between 16 and 16384")
        if length < 1 or length > 16384 or (length - 1) % 4 != 0:
            raise ValueError("length must be 1 plus a multiple of 4 between 1 and 16384")
        if batch_size < 1 or batch_size > 4096:
            raise ValueError("batch_size must be between 1 and 4096")
        handle = _native_handle(model, "model")
        assembled = handle.runtime.assembled
        config = assembled.diffusion.config
        legacy_i2v = config.model_type == "i2v"
        wan22_i2v = (
            config.model_type == "t2v"
            and getattr(config, "in_channels", None) == 36
            and getattr(config, "out_channels", None) == 16
        )
        if not legacy_i2v and not wan22_i2v:
            raise ValueError("model must use a supported Wan I2V profile")
        if legacy_i2v and assembled.clip_vision is None:
            raise ValueError("loaded Wan 2.1 I2V model has no CLIP vision encoder")
        torch = _torch()
        if type(start_image) is not torch.Tensor:
            raise TypeError("start_image must be an exact torch.Tensor")
        image = cast("Any", start_image)
        if (
            image.ndim != 4
            or image.shape[0] <= 0
            or image.shape[1] <= 0
            or image.shape[2] <= 0
            or image.shape[3] < 3
            or not image.is_floating_point()
            or image.layout != torch.strided
        ):
            raise ValueError(
                "start_image must be a nonempty strided floating [frames,H,W,C>=3] tensor"
            )
        vision = None
        if legacy_i2v and clip_vision_output is not None:
            vision = _wan21_clip_embedding(
                clip_vision_output,
                name="clip_vision_output",
                handle=handle,
                torch=torch,
            )
        elif not legacy_i2v and clip_vision_output is not None:
            raise ValueError("Wan 2.2 I2V does not consume CLIP vision output")

        resized = (
            importlib.import_module("dinkster_inference_torch.resize")
            .common_upscale(
                image[:length, :, :, :3].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            )
            .movedim(1, -1)
        )
        frames = min(int(resized.shape[0]), length)
        padded = torch.full(
            (length, height, width, 3),
            0.5,
            device=resized.device,
            dtype=resized.dtype,
        )
        padded[:frames] = resized[:frames]
        content = padded.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
        with handle.stage("vae"):
            with torch.inference_mode():
                direct_oom: BaseException | None = None
                reference: Any = None
                try:
                    reference = handle.runtime.encode_content(content)
                except RuntimeError as caught:
                    if not _is_accelerator_oom(caught, torch=torch, device=handle.load_device):
                        raise
                    direct_oom = caught.with_traceback(None)
                if direct_oom is not None:
                    reference = _retry_tiled_vae_after_oom(
                        handle=handle,
                        value=content,
                        output_dtype=torch.float32,
                        direction="encode",
                        oom=direct_oom,
                    )
        latent_frames = ((length - 1) // 4) + 1
        expected_reference = (1, 16, latent_frames, height // 8, width // 8)
        if type(reference) is not torch.Tensor or tuple(reference.shape) != expected_reference:
            actual = getattr(reference, "shape", None)
            raise ValueError(
                f"Wan VAE reference latent has shape {actual}, expected {expected_reference}"
            )
        mask = torch.zeros(
            (1, 4, latent_frames, height // 8, width // 8),
            device=reference.device,
            dtype=reference.dtype,
        )
        mask[:, :, : ((frames - 1) // 4) + 1] = 1.0
        concat_latent = torch.cat((mask, reference), dim=1)
        latent = torch.zeros(
            (batch_size, 16, latent_frames, height // 8, width // 8),
            device="cpu",
            dtype=reference.dtype,
        )
        inference = importlib.import_module("dinkster_inference")
        prepared_positive = _wan21_i2v_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        prepared_negative = _wan21_i2v_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        return cls.outputs(
            positive=prepared_positive,
            negative=prepared_negative,
            latent={"samples": latent},
        )


_WAN_CAMERA_MOTIONS = {
    "Static": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "Pan Up": ((0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    "Pan Down": ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "Pan Left": ((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    "Pan Right": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "Zoom In": ((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
    "Zoom Out": ((0.0, 0.0, 0.0), (0.0, 0.0, -2.0)),
    "Anti Clockwise (ACW)": ((0.0, 0.0, -1.0), (0.0, 0.0, 0.0)),
    "ClockWise (CW)": ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
}


def _wan_camera_dimensions(*, width: int, height: int, length: int) -> None:
    if width < 16 or width > 16384 or width % 16 != 0:
        raise ValueError("width must be a multiple of 16 between 16 and 16384")
    if height < 16 or height > 16384 or height % 16 != 0:
        raise ValueError("height must be a multiple of 16 between 16 and 16384")
    if length < 1 or length > 16384 or (length - 1) % 4 != 0:
        raise ValueError("length must be 1 plus a multiple of 4 between 1 and 16384")


def _wan_camera_embedding(
    *,
    camera_pose: str,
    width: int,
    height: int,
    length: int,
    speed: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Any:
    _wan_camera_dimensions(width=width, height=height, length=length)
    if camera_pose not in WAN_CAMERA_POSES:
        raise ValueError("camera_pose is not a supported Wan camera trajectory")
    for name, value, minimum, maximum in (
        ("speed", speed, 0.0, 10.0),
        ("fx", fx, 0.0, 1.0),
        ("fy", fy, 0.0, 1.0),
        ("cx", cx, 0.0, 1.0),
        ("cy", cy, 0.0, 1.0),
    ):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")

    np = cast("Any", importlib.import_module("numpy"))
    torch = _torch()
    raw_angle, raw_translation = _WAN_CAMERA_MOTIONS[camera_pose]
    angle = np.array(raw_angle)
    translation = np.array(raw_translation).reshape(3, 1)
    extrinsics: list[Any] = []
    for index in range(length):
        theta_x, theta_y, theta_z = index / length * speed * (np.pi / 3) * angle
        rotation_x = np.array(
            (
                (1, 0, 0),
                (0, np.cos(theta_x), -np.sin(theta_x)),
                (0, np.sin(theta_x), np.cos(theta_x)),
            )
        )
        rotation_y = np.array(
            (
                (np.cos(theta_y), 0, np.sin(theta_y)),
                (0, 1, 0),
                (-np.sin(theta_y), 0, np.cos(theta_y)),
            )
        )
        rotation_z = np.array(
            (
                (np.cos(theta_z), -np.sin(theta_z), 0),
                (np.sin(theta_z), np.cos(theta_z), 0),
                (0, 0, 1),
            )
        )
        rotation = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
        offset = index / length * speed * 1.5 * translation
        extrinsics.append(np.concatenate((rotation, offset), axis=1))

    entries: list[list[float]] = []
    for extrinsic in cast("list[list[list[float]]]", np.stack(extrinsics).tolist()):
        entry: list[float] = [fx, fy, cx, cy, 0.0, 0.0]
        entry.extend(extrinsic[0])
        entry.extend(extrinsic[1])
        entry.extend(extrinsic[2])
        entry.extend((0.0, 0.0, 0.0, 1.0))
        entries.append(entry)
    camera_parameters = np.concatenate(
        (
            np.zeros((length, 1)),
            np.array([[float(value) for value in entry] for entry in entries]),
        ),
        axis=1,
    )
    focal_x = camera_parameters[:, 1].copy()
    focal_y = camera_parameters[:, 2].copy()
    sample_ratio = width / height
    pose_ratio = 1280 / 720
    if pose_ratio > sample_ratio:
        focal_x *= height * pose_ratio / width
    else:
        focal_y *= width / pose_ratio / height
    intrinsic = np.asarray(
        tuple(
            (
                focal_x[index] * width,
                focal_y[index] * height,
                camera_parameters[index, 3] * width,
                camera_parameters[index, 4] * height,
            )
            for index in range(length)
        ),
        dtype=np.float32,
    )
    camera_to_world = tuple(np.array(entry[7:]).reshape(4, 4) for entry in camera_parameters)
    world_to_camera = tuple(np.linalg.inv(matrix) for matrix in camera_to_world)
    target = np.array(
        (
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )
    absolute_to_relative = target @ world_to_camera[0]
    poses = np.array(
        (target, *(absolute_to_relative @ matrix for matrix in camera_to_world[1:])),
        dtype=np.float32,
    )
    intrinsics = torch.as_tensor(intrinsic)[None]
    c2w = torch.as_tensor(poses)[None]
    row, column = torch.meshgrid(
        torch.linspace(0, height - 1, height, dtype=c2w.dtype),
        torch.linspace(0, width - 1, width, dtype=c2w.dtype),
        indexing="ij",
    )
    column = column.reshape(1, 1, height * width).expand(1, 1, height * width) + 0.5
    row = row.reshape(1, 1, height * width).expand(1, 1, height * width) + 0.5
    focal_x_t, focal_y_t, center_x, center_y = intrinsics.chunk(4, dim=-1)
    depth = torch.ones_like(column)
    direction_x = (column - center_x) / focal_x_t * depth
    direction_y = (row - center_y) / focal_y_t * depth
    depth = depth.expand_as(direction_y)
    directions = torch.stack(
        (
            direction_x,
            direction_y,
            depth,
        ),
        dim=-1,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)
    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3][:, :, None].expand_as(rays_d)
    plucker = torch.cat((torch.cross(rays_o, rays_d, dim=-1), rays_d), dim=-1)
    embedding = plucker.reshape(1, length, height, width, 6).permute(0, 4, 1, 2, 3)
    embedding = torch.cat(
        (embedding[:, :, :1].repeat_interleave(4, dim=2), embedding[:, :, 1:]), dim=2
    )
    embedding = (
        embedding.transpose(1, 2)
        .reshape(1, embedding.shape[2] // 4, 4, 6, height, width)
        .transpose(2, 3)
        .reshape(1, embedding.shape[2] // 4, 24, height, width)
        .transpose(1, 2)
        .contiguous()
    )
    # intermediate_device() @ b78cec87 is cpu absent ComfyUI's --gpu-only
    # flag, which Dinkster does not wire.
    return embedding.to(device="cpu")


class NativeWanCameraEmbedding(WanCameraEmbedding):
    @classmethod
    def execute(
        cls,
        *,
        camera_pose: str,
        width: int,
        height: int,
        length: int,
        speed: float = 1.0,
        fx: float = 0.5,
        fy: float = 0.5,
        cx: float = 0.5,
        cy: float = 0.5,
    ) -> Mapping[str, object]:
        embedding = _wan_camera_embedding(
            camera_pose=camera_pose,
            width=width,
            height=height,
            length=length,
            speed=speed,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
        return cls.outputs(
            camera_embedding=embedding,
            width=width,
            height=height,
            length=length,
        )


def _wan_camera_prepared(
    value: object,
    *,
    name: str,
    handle: NativeRuntimeHandle,
    inference: Any,
    concat_latent: Any,
    camera_conditions: Any,
    vision: Any,
) -> list[list[object]]:
    text = _prepared_multistream_conditioning(
        value,
        inference,
        name,
        handle.runtime.conditioning_identity,
    )
    if text is None:
        raise TypeError(f"{name} must contain Wan text conditioning")
    prepared = handle.runtime.prepare_camera_conditioning(
        text,
        concat_latent,
        camera_conditions,
        vision,
    )
    return [
        [
            inference.PreparedMultiStreamConditioning(
                handle.runtime.conditioning_identity,
                prepared,
            ),
            {},
        ]
    ]


class NativeWanCameraImageToVideo(WanCameraImageToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        clip_vision_output: object = None,
        start_image: object = None,
        camera_conditions: object = None,
    ) -> Mapping[str, object]:
        _wan_camera_dimensions(width=width, height=height, length=length)
        if batch_size < 1 or batch_size > 4096:
            raise ValueError("batch_size must be between 1 and 4096")
        handle = _native_handle(vae, "vae")
        config = handle.runtime.assembled.diffusion.config
        if (
            config.camera_channels != 24
            or config.out_channels != 16
            or config.in_channels not in (32, 36)
        ):
            raise ValueError("vae must come from a supported Wan camera profile")
        torch = _torch()
        if camera_conditions is not None:
            if type(camera_conditions) is not torch.Tensor:
                raise TypeError("camera_conditions must come from Wan Camera Embedding")
            expected_camera = (1, 24, ((length - 1) // 4) + 1, height, width)
            if tuple(cast("Any", camera_conditions).shape) != expected_camera:
                raise ValueError(
                    f"camera_conditions must have shape {expected_camera}, got "
                    f"{tuple(cast('Any', camera_conditions).shape)}"
                )
        vision = None
        if config.model_type == "i2v":
            if clip_vision_output is not None:
                vision = _wan21_clip_embedding(
                    clip_vision_output,
                    name="clip_vision_output",
                    handle=handle,
                    torch=torch,
                )
        elif clip_vision_output is not None:
            raise ValueError("Wan 2.2 camera does not consume CLIP vision output")

        latent_frames = ((length - 1) // 4) + 1
        latent_shape = (1, 16, latent_frames, height // 8, width // 8)
        concat_latent = None
        if start_image is not None:
            with handle.stage("vae"):
                with torch.inference_mode():
                    reference = handle.runtime.assembled.vae.process_out(
                        torch.zeros(latent_shape, device=handle.load_device, dtype=torch.float32)
                    )
                    image = _wan_image(start_image, "start_image", torch)
                    resized = (
                        importlib.import_module("dinkster_inference_torch.resize")
                        .common_upscale(
                            image[:length, :, :, :3].movedim(-1, 1),
                            width,
                            height,
                            "bilinear",
                            "center",
                        )
                        .movedim(1, -1)
                    )
                    content = resized.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
                    encoded_frames = ((int(resized.shape[0]) - 1) // 4) + 1
                    encoded = _wan_encode_content(
                        handle=handle,
                        content=content,
                        expected=(1, 16, encoded_frames, *latent_shape[-2:]),
                        name="camera start image",
                        torch=torch,
                    )
                    reference[:, :, : encoded.shape[2]] = encoded[:, :, :latent_frames]
                    concat_latent = reference
                    if config.in_channels == 36:
                        external_mask = torch.ones(
                            (1, 1, latent_frames * 4, *latent_shape[-2:]),
                            device=reference.device,
                            dtype=reference.dtype,
                        )
                        external_mask[:, :, : int(resized.shape[0]) + 3] = 0.0
                        model_mask = _wan_flf_model_mask(external_mask, latent_frames)
                        concat_latent = torch.cat((model_mask, reference), dim=1)
        latent = torch.zeros(
            (batch_size, 16, *latent_shape[2:]),
            device="cpu",
            dtype=torch.float32,
        )
        inference = importlib.import_module("dinkster_inference")
        positive_prepared = _wan_camera_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            camera_conditions=camera_conditions,
            vision=vision,
        )
        negative_prepared = _wan_camera_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            camera_conditions=camera_conditions,
            vision=vision,
        )
        return cls.outputs(
            positive=positive_prepared,
            negative=negative_prepared,
            latent={"samples": latent},
        )


def _wan_phantom_prepared(
    value: object,
    *,
    name: str,
    handle: NativeRuntimeHandle,
    inference: Any,
    temporal_reference: Any,
) -> list[list[object]]:
    text = _prepared_multistream_conditioning(
        value,
        inference,
        name,
        handle.runtime.conditioning_identity,
    )
    if text is None:
        raise TypeError(f"{name} must contain Wan text conditioning")
    prepared = handle.runtime.prepare_phantom_conditioning(text, temporal_reference)
    return [
        [
            inference.PreparedMultiStreamConditioning(
                handle.runtime.conditioning_identity,
                prepared,
            ),
            {},
        ]
    ]


class NativeWanPhantomSubjectToVideo(WanPhantomSubjectToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        images: object = None,
    ) -> Mapping[str, object]:
        _wan_fun_dimensions(
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
        )
        handle = _native_handle(vae, "vae")
        config = handle.runtime.assembled.diffusion.config
        if (
            config.model_type != "t2v"
            or config.in_channels != 16
            or config.out_channels != 16
            or config.reference_channels is not None
            or config.vace_layers is not None
            or config.camera_channels is not None
        ):
            raise ValueError("vae must come from a supported Wan Phantom profile")

        torch = _torch()
        temporal_reference = None
        empty_reference = None
        if images is not None:
            image = _wan_image(images, "images", torch)
            resized = (
                importlib.import_module("dinkster_inference_torch.resize")
                .common_upscale(
                    image[:length, :, :, :3].movedim(-1, 1),
                    width,
                    height,
                    "bilinear",
                    "center",
                )
                .movedim(1, -1)
            )
            references: list[Any] = []
            with handle.stage("vae", unload_before=("text",)):
                with torch.inference_mode():
                    for index, frame in enumerate(resized):
                        content = (
                            frame.permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(handle.load_device)
                        )
                        references.append(
                            _wan_encode_content(
                                handle=handle,
                                content=content,
                                expected=(1, 16, 1, height // 8, width // 8),
                                name=f"Phantom reference {index}",
                                torch=torch,
                            )
                        )
                    temporal_reference = torch.cat(references, dim=2)
                    empty_reference = handle.runtime.assembled.vae.process_out(
                        torch.zeros_like(temporal_reference)
                    )

        inference = importlib.import_module("dinkster_inference")
        prepared_positive = _wan_phantom_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            temporal_reference=temporal_reference,
        )
        prepared_negative_text = _wan_phantom_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            temporal_reference=temporal_reference,
        )
        prepared_negative_img_text = _wan_phantom_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            temporal_reference=empty_reference,
        )
        latent = torch.zeros(
            (batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8),
            device="cpu",
            dtype=(torch.float32 if temporal_reference is None else temporal_reference.dtype),
        )
        return cls.outputs(
            positive=prepared_positive,
            negative_text=prepared_negative_text,
            negative_img_text=prepared_negative_img_text,
            latent={"samples": latent},
        )


def _wan_image(value: object, name: str, torch: Any) -> Any:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{name} must be an exact torch.Tensor")
    image = cast("Any", value)
    if (
        image.ndim != 4
        or image.shape[0] <= 0
        or image.shape[1] <= 0
        or image.shape[2] <= 0
        or image.shape[3] < 3
        or not image.is_floating_point()
        or image.layout != torch.strided
    ):
        raise ValueError(f"{name} must be a nonempty strided floating [frames,H,W,C>=3] tensor")
    return image


def _wan_flf_model_mask(mask: Any, latent_frames: int) -> Any:
    return 1.0 - mask.reshape(1, latent_frames, 4, *mask.shape[-2:]).transpose(1, 2)


def _wan_ati_model_mask(mask: Any) -> Any:
    # Preserve ComfyUI's two complement operations; combining them changes float values.
    external_mask = -mask + 1.0
    return 1.0 - external_mask


class NativeWanTrackToVideo(WanTrackToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        tracks: str,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        temperature: float,
        topk: int,
        start_image: object,
        clip_vision_output: object = None,
    ) -> Mapping[str, object]:
        _wan_fun_dimensions(
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
        )
        if (
            type(temperature) not in (int, float)
            or not math.isfinite(temperature)
            or not 1.0 <= temperature <= 1000.0
        ):
            raise ValueError("temperature must be finite and between 1 and 1000")
        if type(topk) is not int or not 1 <= topk <= 10:
            raise ValueError("topk must be between 1 and 10")
        handle = _native_handle(vae, "vae")
        assembled = handle.runtime.assembled
        config = assembled.diffusion.config
        if config.model_type != "i2v" or config.in_channels != 36 or config.out_channels != 16:
            raise ValueError("vae must come from a supported Wan 2.1 ATI profile")
        if assembled.clip_vision is None:
            raise ValueError("loaded Wan 2.1 ATI model has no CLIP vision encoder")

        empty_tracks = False
        if type(tracks) is str:
            try:
                empty_tracks = not json.loads(tracks.replace("'", '"'))
            except json.JSONDecodeError:
                empty_tracks = True
        if empty_tracks:
            return NativeWan21ImageToVideo.execute(
                positive=positive,
                negative=negative,
                model=vae,
                start_image=start_image,
                clip_vision_output=clip_vision_output,
                width=width,
                height=height,
                length=length,
                batch_size=batch_size,
            )

        torch = _torch()
        inference_torch = importlib.import_module("dinkster_inference_torch")
        prepared_tracks = inference_torch.prepare_wan_ati_tracks(
            tracks,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
        )
        image = _wan_image(start_image, "start_image", torch)
        resized = (
            importlib.import_module("dinkster_inference_torch.resize")
            .common_upscale(
                image[:batch_size, :, :, :3].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            )
            .movedim(1, -1)
        )
        videos = torch.full(
            (resized.shape[0], length, height, width, 3),
            0.5,
            device=resized.device,
            dtype=resized.dtype,
        )
        videos[:, 0] = resized
        videos = importlib.import_module("dinkster_inference_torch.resize").resize_to_batch_size(
            videos, batch_size
        )
        latent_frames = ((length - 1) // 4) + 1
        encoded_videos: list[Any] = []
        with handle.stage("vae"):
            with torch.inference_mode():
                for index in range(batch_size):
                    content = videos[index].permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
                    direct_oom: BaseException | None = None
                    encoded: Any = None
                    try:
                        encoded = handle.runtime.encode_content(content)
                    except RuntimeError as caught:
                        if not _is_accelerator_oom(caught, torch=torch, device=handle.load_device):
                            raise
                        direct_oom = caught.with_traceback(None)
                    if direct_oom is not None:
                        encoded = _retry_tiled_vae_after_oom(
                            handle=handle,
                            value=content,
                            output_dtype=torch.float32,
                            direction="encode",
                            oom=direct_oom,
                        )
                    expected = (1, 16, latent_frames, height // 8, width // 8)
                    if type(encoded) is not torch.Tensor or tuple(encoded.shape) != expected:
                        actual = getattr(encoded, "shape", None)
                        raise ValueError(
                            f"Wan ATI video latent has shape {actual}, expected {expected}"
                        )
                    encoded_videos.append(encoded.to("cpu"))
        external_video = torch.cat(tuple(encoded_videos), dim=0)
        model_video = assembled.vae.process_in(external_video)
        ati_mask, model_motion = inference_torch.patch_wan_ati_motion(
            prepared_tracks,
            model_video,
            temperature=float(temperature),
            topk=topk,
        )
        concat_latent = torch.cat(
            (_wan_ati_model_mask(ati_mask), assembled.vae.process_out(model_motion)),
            dim=1,
        )
        vision = None
        if clip_vision_output is not None:
            vision = _wan21_clip_embedding(
                clip_vision_output,
                name="clip_vision_output",
                handle=handle,
                torch=torch,
            )
        inference = importlib.import_module("dinkster_inference")
        prepared_positive = _wan21_i2v_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        prepared_negative = _wan21_i2v_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        latent = torch.zeros(
            (batch_size, 16, latent_frames, height // 8, width // 8),
            device="cpu",
            dtype=concat_latent.dtype,
        )
        return cls.outputs(
            positive=prepared_positive,
            negative=prepared_negative,
            latent={"samples": latent},
        )


def _wan_move_tracks(
    value: object,
    name: str,
    torch: Any,
    *,
    require_visibility: bool,
) -> tuple[Any, Any | None]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a TRACKS mapping")
    mapping = cast("Mapping[object, object]", value)
    raw_track_path = mapping.get("track_path")
    if type(raw_track_path) is not torch.Tensor:
        raise TypeError(f"{name} track_path must be an exact torch.Tensor")
    track_path = cast("Any", raw_track_path)
    if (
        track_path.ndim != 3
        or track_path.shape[0] <= 0
        or track_path.shape[1] <= 0
        or track_path.shape[2] != 2
        or not track_path.is_floating_point()
        or track_path.layout != torch.strided
    ):
        raise ValueError(f"{name} track_path must be a nonempty strided floating [T,N,2] tensor")
    raw_visibility = mapping.get("track_visibility")
    if raw_visibility is None and not require_visibility:
        return track_path, None
    if type(raw_visibility) is not torch.Tensor:
        raise TypeError(f"{name} track_visibility must be an exact torch.Tensor")
    visibility = cast("Any", raw_visibility)
    if (
        visibility.ndim != 2
        or visibility.shape[0] != track_path.shape[0]
        or visibility.shape[1] not in (1, track_path.shape[1])
        or visibility.layout != torch.strided
    ):
        raise ValueError(f"{name} track_visibility must be a strided [T,N] tensor")
    return track_path, visibility


def _wan_move_visibility(
    track_mask: object,
    *,
    track_length: int,
    num_tracks: int,
    torch: Any,
) -> Any:
    if track_mask is None:
        return torch.ones((track_length, num_tracks), dtype=torch.bool)
    if type(track_mask) is not torch.Tensor:
        raise TypeError("track_mask must be an exact torch.Tensor")
    mask = cast("Any", track_mask)
    if mask.ndim != 3 or mask.shape[0] != track_length or mask.layout != torch.strided:
        raise ValueError("track_mask must be a strided [T,H,W] tensor matching the track length")
    return (mask > 0).any(dim=(1, 2)).unsqueeze(-1)


def _wan_move_parse_coords(value: object) -> list[list[Mapping[str, Any]]]:
    if type(value) is not str:
        raise TypeError("track_coords must be a string")
    try:
        parsed = json.loads(value.replace("'", '"'))
    except json.JSONDecodeError as error:
        raise ValueError("track_coords must contain valid JSON tracks") from error
    if type(parsed) is not list or not parsed:
        raise ValueError("track_coords must contain at least one track")
    tracks = cast("list[Any]", parsed)
    first = tracks[0]
    if isinstance(first, Mapping) and "x" in first:
        tracks = [tracks]
        first = tracks[0]
    first_track = cast("list[Any]", first) if isinstance(first, list) else []
    if (
        not tracks
        or not isinstance(first, list)
        or not first_track
        or not isinstance(first_track[0], Mapping)
        or "x" not in first_track[0]
    ):
        raise ValueError("track_coords must be a track or list of tracks with x/y points")
    return cast("list[list[Mapping[str, Any]]]", tracks)


class NativeWanMoveTracksFromCoords(WanMoveTracksFromCoords):
    @classmethod
    def execute(
        cls,
        *,
        track_coords: str = "[]",
        track_mask: object = None,
    ) -> Mapping[str, object]:
        torch = _torch()
        tracks_data = _wan_move_parse_coords(track_coords)
        track_length = len(tracks_data[0])
        track_list: list[list[list[float]]] = [
            [[float(track[frame]["x"]), float(track[frame]["y"])] for track in tracks_data]
            for frame in range(track_length)
        ]
        tracks = torch.tensor(track_list, dtype=torch.float32)
        visibility = _wan_move_visibility(
            track_mask,
            track_length=track_length,
            num_tracks=int(tracks.shape[1]),
            torch=torch,
        )
        return cls.outputs(
            tracks={"track_path": tracks, "track_visibility": visibility},
            track_length=track_length,
        )


class NativeWanMoveConcatTrack(WanMoveConcatTrack):
    @classmethod
    def execute(
        cls,
        *,
        tracks_1: object,
        tracks_2: object = None,
    ) -> Mapping[str, object]:
        torch = _torch()
        path_1, visibility_1 = _wan_move_tracks(
            tracks_1, "tracks_1", torch, require_visibility=True
        )
        if tracks_2 is None:
            return cls.outputs(tracks=tracks_1)
        path_2, visibility_2 = _wan_move_tracks(
            tracks_2, "tracks_2", torch, require_visibility=True
        )
        assert visibility_1 is not None and visibility_2 is not None
        return cls.outputs(
            tracks={
                "track_path": torch.cat((path_1, path_2), dim=1),
                "track_visibility": torch.cat((visibility_1, visibility_2), dim=-1),
            }
        )


class NativeWanMoveGenerateTracks(WanMoveGenerateTracks):
    @classmethod
    def execute(
        cls,
        *,
        width: int,
        height: int,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        num_frames: int,
        num_tracks: int,
        track_spread: float,
        bezier: bool = False,
        mid_x: float = 0.5,
        mid_y: float = 0.5,
        interpolation: str = "linear",
        track_mask: object = None,
    ) -> Mapping[str, object]:
        if type(width) is not int or not 16 <= width <= 4096:
            raise ValueError("width must be between 16 and 4096")
        if type(height) is not int or not 16 <= height <= 4096:
            raise ValueError("height must be between 16 and 4096")
        if type(num_frames) is not int or not 1 <= num_frames <= 1024:
            raise ValueError("num_frames must be between 1 and 1024")
        if type(num_tracks) is not int or not 1 <= num_tracks <= 100:
            raise ValueError("num_tracks must be between 1 and 100")
        points = (start_x, start_y, mid_x, mid_y, end_x, end_y)
        if any(type(value) not in (int, float) or not 0.0 <= value <= 1.0 for value in points):
            raise ValueError("track coordinates must be between 0 and 1")
        if (
            type(track_spread) not in (int, float)
            or not math.isfinite(track_spread)
            or not 0.0 <= track_spread <= 1.0
        ):
            raise ValueError("track_spread must be finite and between 0 and 1")
        if type(bezier) is not bool:
            raise TypeError("bezier must be a bool")
        if interpolation not in ("linear", "ease_in", "ease_out", "ease_in_out", "constant"):
            raise ValueError("unknown track interpolation")

        torch = _torch()
        start_x_px, start_y_px = start_x * width, start_y * height
        mid_x_px, mid_y_px = mid_x * width, mid_y * height
        end_x_px, end_y_px = end_x * width, end_y * height
        track_spread_px = track_spread * (width + height) / 2
        t = torch.linspace(0, 1, num_frames)
        if interpolation == "constant":
            interp_values = torch.zeros_like(t)
        elif interpolation == "linear":
            interp_values = t
        elif interpolation == "ease_in":
            interp_values = t**2
        elif interpolation == "ease_out":
            interp_values = 1 - (1 - t) ** 2
        else:
            interp_values = t * t * (3 - 2 * t)

        if bezier:
            t_interp = interp_values
            one_minus_t = 1 - t_interp
            x_positions = (
                one_minus_t**2 * start_x_px
                + 2 * one_minus_t * t_interp * mid_x_px
                + t_interp**2 * end_x_px
            )
            y_positions = (
                one_minus_t**2 * start_y_px
                + 2 * one_minus_t * t_interp * mid_y_px
                + t_interp**2 * end_y_px
            )
            tangent_x = 2 * one_minus_t * (mid_x_px - start_x_px) + 2 * t_interp * (
                end_x_px - mid_x_px
            )
            tangent_y = 2 * one_minus_t * (mid_y_px - start_y_px) + 2 * t_interp * (
                end_y_px - mid_y_px
            )
        else:
            x_positions = start_x_px + (end_x_px - start_x_px) * interp_values
            y_positions = start_y_px + (end_y_px - start_y_px) * interp_values
            tangent_x = torch.full_like(t, end_x_px - start_x_px)
            tangent_y = torch.full_like(t, end_y_px - start_y_px)

        track_list: list[list[list[float]]] = []
        for frame_idx in range(num_frames):
            tx = tangent_x[frame_idx].item()
            ty = tangent_y[frame_idx].item()
            length = (tx**2 + ty**2) ** 0.5
            if length > 0:
                perp_x, perp_y = -ty / length, tx / length
            else:
                perp_x, perp_y = 1.0, 0.0
            frame_tracks: list[list[float]] = []
            for track_idx in range(num_tracks):
                offset = (track_idx - (num_tracks - 1) / 2) * track_spread_px
                frame_tracks.append(
                    [
                        float(x_positions[frame_idx].item() + perp_x * offset),
                        float(y_positions[frame_idx].item() + perp_y * offset),
                    ]
                )
            track_list.append(frame_tracks)

        tracks = torch.tensor(track_list, dtype=torch.float32)
        visibility = _wan_move_visibility(
            track_mask,
            track_length=num_frames,
            num_tracks=num_tracks,
            torch=torch,
        )
        return cls.outputs(
            tracks={"track_path": tracks, "track_visibility": visibility},
            track_length=num_frames,
        )


def _wan_move_draw_gradient_polyline(
    overlay: Any,
    line_width: int,
    points: Any,
    color: tuple[int, int, int],
    opacity: float,
    image_draw: Any,
) -> None:
    draw = image_draw.Draw(overlay, "RGBA")
    points = points[::-1]
    segment_lengths: list[float] = []
    total_length = 0.0
    for index in range(len(points) - 1):
        dx = float(points[index + 1][0] - points[index][0])
        dy = float(points[index + 1][1] - points[index][1])
        length = (dx * dx + dy * dy) ** 0.5
        segment_lengths.append(length)
        total_length += length
    if total_length == 0:
        return
    accumulated_length = 0.0
    for index, (start_point, end_point) in enumerate(zip(points[:-1], points[1:], strict=True)):
        segment_length = segment_lengths[index]
        steps = max(int(segment_length), 1)
        for step in range(steps):
            current_length = accumulated_length + (step / steps) * segment_length
            ratio = current_length / total_length
            alpha = int(255 * (1 - ratio) * opacity)
            x = int(start_point[0] + (end_point[0] - start_point[0]) * step / steps)
            y = int(start_point[1] + (end_point[1] - start_point[1]) * step / steps)
            dynamic_width = max(int(line_width * (1 - ratio)), 1)
            draw.line([(x, y), (x + 1, y)], fill=(*color, alpha), width=dynamic_width)
        accumulated_length += segment_length


def _wan_move_draw_tracks(
    video: Any,
    tracks: Any,
    visibility: Any,
    *,
    track_frame: int,
    circle_size: int,
    opacity: float,
    line_width: int,
) -> list[Any]:
    np = cast("Any", importlib.import_module("numpy"))
    image = cast("Any", importlib.import_module("PIL.Image"))
    image_draw = cast("Any", importlib.import_module("PIL.ImageDraw"))
    colors: tuple[tuple[int, int, int], ...] = (
        (102, 153, 255),
        (0, 255, 255),
        (255, 255, 0),
        (255, 102, 204),
        (0, 255, 0),
    )
    video_np = video.byte().cpu().numpy()
    tracks_np = tracks[0].long().detach().cpu().numpy()
    visibility_np = visibility[0].detach().cpu().numpy()
    num_frames, height, width = video_np.shape[:3]
    num_tracks = tracks_np.shape[1]
    alpha_opacity = int(255 * opacity)
    output_frames: list[Any] = []
    for frame_index in range(num_frames):
        frame_rgb = video_np[frame_index].astype(np.float32)
        overlay = image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw_overlay = image_draw.Draw(overlay)
        polylines: list[tuple[Any, tuple[int, int, int]]] = []
        for track_index in range(num_tracks):
            if visibility_np[frame_index, track_index] == 0:
                continue
            coordinate = tracks_np[frame_index, track_index]
            color = colors[track_index % len(colors)]
            draw_overlay.ellipse(
                (
                    coordinate[0] - circle_size,
                    coordinate[1] - circle_size,
                    coordinate[0] + circle_size,
                    coordinate[1] + circle_size,
                ),
                fill=color + (alpha_opacity,),
            )
            track_coordinates = tracks_np[
                max(frame_index - track_frame, 0) : frame_index + 1, track_index
            ]
            if len(track_coordinates) > 1:
                polylines.append((track_coordinates, color))
        overlay_np = np.array(overlay)
        alpha = overlay_np[:, :, 3:4] / 255.0
        frame_rgb = overlay_np[:, :, :3] * alpha + frame_rgb * (1 - alpha)
        if polylines:
            polyline_overlay = image.new("RGBA", (width, height), (0, 0, 0, 0))
            for track_coordinates, color in polylines:
                _wan_move_draw_gradient_polyline(
                    polyline_overlay,
                    line_width,
                    track_coordinates,
                    color,
                    opacity,
                    image_draw,
                )
            polyline_np = np.array(polyline_overlay)
            alpha = polyline_np[:, :, 3:4] / 255.0
            frame_rgb = polyline_np[:, :, :3] * alpha + frame_rgb * (1 - alpha)
        output_frames.append(image.fromarray(frame_rgb.astype(np.uint8)))
    return output_frames


class NativeWanMoveVisualizeTracks(WanMoveVisualizeTracks):
    @classmethod
    def execute(
        cls,
        *,
        images: object,
        line_resolution: int,
        circle_size: int,
        opacity: float,
        line_width: int,
        tracks: object = None,
    ) -> Mapping[str, object]:
        if tracks is None:
            return cls.outputs(images=images)
        torch = _torch()
        image_tensor = _wan_image(images, "images", torch)
        if image_tensor.shape[-1] != 3:
            raise ValueError("images must have exactly three channels")
        path, visibility = _wan_move_tracks(tracks, "tracks", torch, require_visibility=True)
        assert visibility is not None
        images_in = image_tensor * 255.0
        if images_in.shape[0] != path.shape[0]:
            repeat_count = path.shape[0] // images_in.shape[0]
            images_in = images_in.repeat(repeat_count, 1, 1, 1)
        frames = _wan_move_draw_tracks(
            images_in,
            path.unsqueeze(0),
            visibility.unsqueeze(0),
            track_frame=line_resolution,
            circle_size=circle_size,
            opacity=opacity,
            line_width=line_width,
        )
        np = cast("Any", importlib.import_module("numpy"))
        output = torch.from_numpy(np.stack([np.asarray(frame) for frame in frames])).float() / 255.0
        return cls.outputs(images=output)


def _wan_move_positions(
    tracks: Any,
    visibility: Any,
    *,
    height: int,
    width: int,
    torch: Any,
) -> Any:
    frame_count, track_count, _ = tracks.shape
    positions = -torch.ones(
        track_count,
        (frame_count - 1) // 4 + 1,
        2,
        dtype=torch.long,
    )
    selected = torch.randperm(track_count)[:track_count]
    tracks = tracks[:, selected]
    visibility = visibility[:, selected]
    for frame_index in range(0, frame_count, 4):
        current_tracks = tracks[frame_index]
        current_visibility = visibility[frame_index]
        for track_index in range(track_count):
            if (
                not current_visibility[track_index]
                or current_tracks[track_index][0] < 0
                or current_tracks[track_index][1] < 0
                or current_tracks[track_index][0] >= width
                or current_tracks[track_index][1] >= height
            ):
                continue
            x, y = current_tracks[track_index]
            positions[track_index, frame_index // 4, 0] = int(y // 8)
            positions[track_index, frame_index // 4, 1] = int(x // 8)
    return positions


def _wan_move_replace_feature(
    vae_feature: Any,
    positions: Any,
    strength: float,
    torch: Any,
) -> Any:
    batch, _, _, _, _ = vae_feature.shape
    if batch != positions.shape[0]:
        raise ValueError("WanMove track and VAE feature batch sizes must match")
    track_count = positions.shape[1]
    positions = positions[:, torch.randperm(track_count)]
    current = positions[:, :, 1:, :]
    valid = (current[..., 0] >= 0) & (current[..., 1] >= 0)
    indices = valid.nonzero(as_tuple=False)
    if indices.shape[0] == 0:
        return vae_feature
    batch_index = indices[:, 0]
    track_index = indices[:, 1]
    relative_time = indices[:, 2]
    target_time = relative_time + 1
    target_height = current[batch_index, track_index, relative_time, 0].long()
    target_width = current[batch_index, track_index, relative_time, 1].long()
    source_height = positions[batch_index, track_index, 0, 0].long()
    source_width = positions[batch_index, track_index, 0, 1].long()
    source_features = vae_feature[batch_index, :, 0, source_height, source_width]
    destination_features = vae_feature[batch_index, :, target_time, target_height, target_width]
    vae_feature[batch_index, :, target_time, target_height, target_width] = (
        destination_features + (source_features - destination_features) * strength
    )
    return vae_feature


class NativeWanMoveTrackToVideo(WanMoveTrackToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        strength: float,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        start_image: object,
        tracks: object = None,
        clip_vision_output: object = None,
    ) -> Mapping[str, object]:
        _wan_fun_dimensions(
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
        )
        if (
            type(strength) not in (int, float)
            or not math.isfinite(strength)
            or not 0.0 <= strength <= 100.0
        ):
            raise ValueError("strength must be finite and between 0 and 100")
        handle = _native_handle(vae, "vae")
        assembled = handle.runtime.assembled
        config = assembled.diffusion.config
        if config.model_type != "i2v" or config.in_channels != 36 or config.out_channels != 16:
            raise ValueError("vae must come from a supported Wan 2.1 I2V profile")
        if assembled.clip_vision is None:
            raise ValueError("loaded Wan 2.1 I2V model has no CLIP vision encoder")

        torch = _torch()
        image = _wan_image(start_image, "start_image", torch)
        resized = (
            importlib.import_module("dinkster_inference_torch.resize")
            .common_upscale(
                image[:length].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            )
            .movedim(1, -1)
        )
        frames = min(int(resized.shape[0]), length)
        padded = torch.full(
            (length, height, width, resized.shape[-1]),
            0.5,
            device=resized.device,
            dtype=resized.dtype,
        )
        padded[:frames] = resized[:frames]
        content = padded[:, :, :, :3].permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
        with handle.stage("vae"):
            with torch.inference_mode():
                direct_oom: BaseException | None = None
                reference: Any = None
                try:
                    reference = handle.runtime.encode_content(content)
                except RuntimeError as caught:
                    if not _is_accelerator_oom(caught, torch=torch, device=handle.load_device):
                        raise
                    direct_oom = caught.with_traceback(None)
                if direct_oom is not None:
                    reference = _retry_tiled_vae_after_oom(
                        handle=handle,
                        value=content,
                        output_dtype=torch.float32,
                        direction="encode",
                        oom=direct_oom,
                    )
        latent_frames = ((length - 1) // 4) + 1
        expected = (1, 16, latent_frames, height // 8, width // 8)
        if type(reference) is not torch.Tensor or tuple(reference.shape) != expected:
            actual = getattr(reference, "shape", None)
            raise ValueError(f"WanMove VAE latent has shape {actual}, expected {expected}")
        reference = reference.to("cpu")
        known_mask = torch.zeros(
            (1, 4, latent_frames, height // 8, width // 8),
            device=reference.device,
            dtype=reference.dtype,
        )
        known_mask[:, :, : ((frames - 1) // 4) + 1] = 1.0
        if tracks is not None and strength > 0.0:
            track_path, visibility = _wan_move_tracks(
                tracks, "tracks", torch, require_visibility=False
            )
            track_path = track_path[:length].to("cpu")
            track_count = int(track_path.shape[1])
            if visibility is None:
                visibility = torch.ones((length, track_count), dtype=torch.bool)
            else:
                visibility = visibility[:length].to("cpu")
            positions = _wan_move_positions(
                track_path,
                visibility,
                height=height,
                width=width,
                torch=torch,
            )
            positions = importlib.import_module(
                "dinkster_inference_torch.resize"
            ).resize_to_batch_size(positions.unsqueeze(0), batch_size)
            with torch.inference_mode():
                reference = _wan_move_replace_feature(reference, positions, float(strength), torch)
        concat_latent = torch.cat((known_mask, reference), dim=1)

        vision = None
        if clip_vision_output is not None:
            vision = _wan21_clip_embedding(
                clip_vision_output,
                name="clip_vision_output",
                handle=handle,
                torch=torch,
            )
        inference = importlib.import_module("dinkster_inference")
        prepared_positive = _wan21_i2v_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        prepared_negative = _wan21_i2v_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        latent = torch.zeros(
            (batch_size, 16, latent_frames, height // 8, width // 8),
            device="cpu",
            dtype=reference.dtype,
        )
        return cls.outputs(
            positive=prepared_positive,
            negative=prepared_negative,
            latent={"samples": latent},
        )


class NativeWanFirstLastFrameToVideo(WanFirstLastFrameToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        clip_vision_start_image: object = None,
        clip_vision_end_image: object = None,
        start_image: object = None,
        end_image: object = None,
    ) -> Mapping[str, object]:
        if width < 16 or width > 16384 or width % 16 != 0:
            raise ValueError("width must be a multiple of 16 between 16 and 16384")
        if height < 16 or height > 16384 or height % 16 != 0:
            raise ValueError("height must be a multiple of 16 between 16 and 16384")
        if length < 1 or length > 16384 or (length - 1) % 4 != 0:
            raise ValueError("length must be 1 plus a multiple of 4 between 1 and 16384")
        if batch_size < 1 or batch_size > 4096:
            raise ValueError("batch_size must be between 1 and 4096")
        handle = _native_handle(vae, "vae")
        assembled = handle.runtime.assembled
        config = assembled.diffusion.config
        if config.in_channels != 36 or config.out_channels != 16:
            raise ValueError("vae must come from a supported Wan first/last-frame profile")
        wan21_i2v = config.model_type == "i2v"
        if not wan21_i2v and config.model_type != "t2v":
            raise ValueError("vae must come from a supported Wan first/last-frame profile")
        if wan21_i2v and config.flf_pos_embed_token_number != 514:
            raise ValueError("Wan 2.1 first/last-frame conditioning requires the FLF profile")
        if wan21_i2v and assembled.clip_vision is None:
            raise ValueError("loaded Wan 2.1 I2V model has no CLIP vision encoder")

        torch = _torch()
        vision = None
        if wan21_i2v:
            vision_rows = tuple(
                _wan21_clip_embedding(value, name=name, handle=handle, torch=torch)
                for name, value in (
                    ("clip_vision_start_image", clip_vision_start_image),
                    ("clip_vision_end_image", clip_vision_end_image),
                )
                if value is not None
            )
            if not vision_rows:
                raise ValueError("Wan 2.1 first/last-frame conditioning requires CLIP vision")
            vision = vision_rows[0] if len(vision_rows) == 1 else torch.cat(vision_rows, dim=-2)
        elif clip_vision_start_image is not None or clip_vision_end_image is not None:
            raise ValueError("Wan 2.2 FLF does not consume CLIP vision output")

        common_upscale = importlib.import_module("dinkster_inference_torch.resize").common_upscale
        resized_start = None
        if start_image is not None:
            image = _wan_image(start_image, "start_image", torch)
            resized_start = common_upscale(
                image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        resized_end = None
        if end_image is not None:
            image = _wan_image(end_image, "end_image", torch)
            resized_end = common_upscale(
                image[-length:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

        image = torch.full(
            (length, height, width, 3),
            0.5,
            device="cpu",
            dtype=torch.float32,
        )
        if resized_start is not None:
            image[: resized_start.shape[0]] = resized_start
        if resized_end is not None:
            image[-resized_end.shape[0] :] = resized_end
        content = image.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
        with handle.stage("vae"):
            with torch.inference_mode():
                direct_oom: BaseException | None = None
                reference: Any = None
                try:
                    reference = handle.runtime.encode_content(content)
                except RuntimeError as caught:
                    if not _is_accelerator_oom(caught, torch=torch, device=handle.load_device):
                        raise
                    direct_oom = caught.with_traceback(None)
                if direct_oom is not None:
                    reference = _retry_tiled_vae_after_oom(
                        handle=handle,
                        value=content,
                        output_dtype=torch.float32,
                        direction="encode",
                        oom=direct_oom,
                    )
        latent_frames = ((length - 1) // 4) + 1
        expected_reference = (1, 16, latent_frames, height // 8, width // 8)
        if type(reference) is not torch.Tensor or tuple(reference.shape) != expected_reference:
            actual = getattr(reference, "shape", None)
            raise ValueError(
                f"Wan VAE reference latent has shape {actual}, expected {expected_reference}"
            )
        mask = torch.ones(
            (1, 1, latent_frames * 4, height // 8, width // 8),
            device=reference.device,
            dtype=reference.dtype,
        )
        if resized_start is not None:
            mask[:, :, : resized_start.shape[0] + 3] = 0.0
        if resized_end is not None:
            mask[:, :, -resized_end.shape[0] :] = 0.0
        mask = _wan_flf_model_mask(mask, latent_frames)
        concat_latent = torch.cat((mask, reference), dim=1)

        inference = importlib.import_module("dinkster_inference")
        prepared_positive = _wan21_i2v_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        prepared_negative = _wan21_i2v_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            vision=vision,
        )
        latent = torch.zeros(
            (batch_size, 16, latent_frames, height // 8, width // 8),
            device="cpu",
            dtype=reference.dtype,
        )
        return cls.outputs(
            positive=prepared_positive,
            negative=prepared_negative,
            latent={"samples": latent},
        )


def _wan_fun_prepared(
    value: object,
    *,
    name: str,
    handle: NativeRuntimeHandle,
    inference: Any,
    concat_latent: Any,
    concat_mask_index: int | None,
    vision: Any = None,
    reference_latent: Any = None,
) -> list[list[object]]:
    text = _prepared_multistream_conditioning(
        value,
        inference,
        name,
        handle.runtime.conditioning_identity,
    )
    if text is None:
        raise TypeError(f"{name} must contain Wan text conditioning")
    prepared = handle.runtime.prepare_fun_conditioning(
        text,
        concat_latent,
        concat_mask_index=concat_mask_index,
        vision=vision,
        reference_latent=reference_latent,
    )
    return [
        [
            inference.PreparedMultiStreamConditioning(
                handle.runtime.conditioning_identity,
                prepared,
            ),
            {},
        ]
    ]


def _wan_fun_dimensions(*, width: int, height: int, length: int, batch_size: int) -> None:
    if width < 16 or width > 16384 or width % 16 != 0:
        raise ValueError("width must be a multiple of 16 between 16 and 16384")
    if height < 16 or height > 16384 or height % 16 != 0:
        raise ValueError("height must be a multiple of 16 between 16 and 16384")
    if length < 1 or length > 16384 or (length - 1) % 4 != 0:
        raise ValueError("length must be 1 plus a multiple of 4 between 1 and 16384")
    if batch_size < 1 or batch_size > 4096:
        raise ValueError("batch_size must be between 1 and 4096")


def _wan_encode_content(
    *,
    handle: NativeRuntimeHandle,
    content: Any,
    expected: tuple[int, ...],
    name: str,
    torch: Any,
) -> Any:
    direct_oom: BaseException | None = None
    encoded: Any = None
    try:
        encoded = handle.runtime.encode_content(content)
    except RuntimeError as caught:
        if not _is_accelerator_oom(caught, torch=torch, device=handle.load_device):
            raise
        direct_oom = caught.with_traceback(None)
    if direct_oom is not None:
        encoded = _retry_tiled_vae_after_oom(
            handle=handle,
            value=content,
            output_dtype=torch.float32,
            direction="encode",
            oom=direct_oom,
        )
    if type(encoded) is not torch.Tensor or tuple(encoded.shape) != expected:
        actual = getattr(encoded, "shape", None)
        raise ValueError(f"Wan {name} latent has shape {actual}, expected {expected}")
    return encoded


def _execute_wan_fun_control(
    cls: type[WanFunControlToVideo] | type[Wan22FunControlToVideo],
    *,
    positive: object,
    negative: object,
    vae: object,
    width: int,
    height: int,
    length: int,
    batch_size: int,
    wan22: bool,
    clip_vision_output: object = None,
    ref_image: object = None,
    start_image: object = None,
    control_video: object = None,
) -> Mapping[str, object]:
    _wan_fun_dimensions(
        width=width,
        height=height,
        length=length,
        batch_size=batch_size,
    )
    handle = _native_handle(vae, "vae")
    config = handle.runtime.assembled.diffusion.config
    channels = config.out_channels
    expected_extra = channels * 2 + (4 if wan22 else 0)
    if (
        config.in_channels - channels != expected_extra
        or (wan22 and config.reference_channels != channels)
        or (not wan22 and config.reference_channels is not None)
    ):
        version = "Wan 2.2" if wan22 else "Wan 2.1"
        raise ValueError(f"vae must come from a {version} Fun control profile")
    torch = _torch()
    vision = None
    if wan22:
        if clip_vision_output is not None:
            raise ValueError("Wan 2.2 Fun control does not consume CLIP vision output")
    else:
        vision = _wan21_clip_embedding(
            clip_vision_output,
            name="clip_vision_output",
            handle=handle,
            torch=torch,
        )
    spatial_scale = 16 if channels == 48 else 8
    latent_frames = ((length - 1) // 4) + 1
    latent_shape = (1, channels, latent_frames, height // spatial_scale, width // spatial_scale)
    common_upscale = importlib.import_module("dinkster_inference_torch.resize").common_upscale

    def image_content(value: object, name: str, frames: int) -> tuple[Any, int]:
        image = _wan_image(value, name, torch)
        resized = common_upscale(
            image[:frames, :, :, :3].movedim(-1, 1),
            width,
            height,
            "bilinear",
            "center",
        ).movedim(1, -1)
        return (
            resized.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device),
            int(resized.shape[0]),
        )

    with handle.stage("vae"):
        with torch.inference_mode():
            neutral = handle.runtime.assembled.vae.process_out(
                torch.zeros(latent_shape, device=handle.load_device, dtype=torch.float32)
            )
            concat_image = neutral.repeat(1, 2, 1, 1, 1)
            start_frames = 0
            if start_image is not None:
                content, source_frames = image_content(start_image, "start_image", length)
                encoded_frames = ((source_frames - 1) // 4) + 1
                encoded = _wan_encode_content(
                    handle=handle,
                    content=content,
                    expected=(1, channels, encoded_frames, *latent_shape[-2:]),
                    name="start image",
                    torch=torch,
                )
                start_frames = source_frames
                concat_image[:, channels:, : encoded.shape[2]] = encoded[:, :, :latent_frames]
            if control_video is not None:
                content, source_frames = image_content(control_video, "control_video", length)
                encoded_frames = ((source_frames - 1) // 4) + 1
                encoded = _wan_encode_content(
                    handle=handle,
                    content=content,
                    expected=(1, channels, encoded_frames, *latent_shape[-2:]),
                    name="control video",
                    torch=torch,
                )
                concat_image[:, :channels, : encoded.shape[2]] = encoded[:, :, :latent_frames]
            reference = None
            if ref_image is not None:
                content, _ = image_content(ref_image, "ref_image", 1)
                reference = _wan_encode_content(
                    handle=handle,
                    content=content,
                    expected=(1, channels, 1, *latent_shape[-2:]),
                    name="reference image",
                    torch=torch,
                )

    mask_index = None
    concat_latent = concat_image
    if wan22:
        external_mask = torch.ones(
            (1, 1, latent_frames * 4, *latent_shape[-2:]),
            device=concat_image.device,
            dtype=concat_image.dtype,
        )
        if start_frames:
            external_mask[:, :, : start_frames + 3] = 0.0
        model_mask = _wan_flf_model_mask(external_mask, latent_frames)
        concat_latent = torch.cat(
            (concat_image[:, :channels], model_mask, concat_image[:, channels:]),
            dim=1,
        )
        mask_index = channels

    inference = importlib.import_module("dinkster_inference")
    prepared_positive = _wan_fun_prepared(
        positive,
        name="positive",
        handle=handle,
        inference=inference,
        concat_latent=concat_latent,
        concat_mask_index=mask_index,
        vision=vision,
        reference_latent=reference,
    )
    prepared_negative = _wan_fun_prepared(
        negative,
        name="negative",
        handle=handle,
        inference=inference,
        concat_latent=concat_latent,
        concat_mask_index=mask_index,
        vision=vision,
        reference_latent=reference,
    )
    latent = torch.zeros(
        (batch_size, channels, latent_frames, *latent_shape[-2:]),
        device="cpu",
        dtype=concat_latent.dtype,
    )
    return cls.outputs(
        positive=prepared_positive,
        negative=prepared_negative,
        latent={"samples": latent},
    )


class NativeWanFunControlToVideo(WanFunControlToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        clip_vision_output: object = None,
        start_image: object = None,
        control_video: object = None,
    ) -> Mapping[str, object]:
        return _execute_wan_fun_control(
            cls,
            positive=positive,
            negative=negative,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
            wan22=False,
            clip_vision_output=clip_vision_output,
            start_image=start_image,
            control_video=control_video,
        )


class NativeWan22FunControlToVideo(Wan22FunControlToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        ref_image: object = None,
        start_image: object = None,
        control_video: object = None,
    ) -> Mapping[str, object]:
        return _execute_wan_fun_control(
            cls,
            positive=positive,
            negative=negative,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
            wan22=True,
            ref_image=ref_image,
            start_image=start_image,
            control_video=control_video,
        )


class NativeWanFunInpaintToVideo(WanFunInpaintToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        clip_vision_output: object = None,
        start_image: object = None,
        end_image: object = None,
    ) -> Mapping[str, object]:
        _wan_fun_dimensions(
            width=width,
            height=height,
            length=length,
            batch_size=batch_size,
        )
        handle = _native_handle(vae, "vae")
        config = handle.runtime.assembled.diffusion.config
        channels = config.out_channels
        if config.in_channels - channels != channels + 4 or config.reference_channels is not None:
            raise ValueError("vae must come from a Wan Fun inpaint profile")
        torch = _torch()
        vision = None
        if config.model_type == "i2v":
            vision = _wan21_clip_embedding(
                clip_vision_output,
                name="clip_vision_output",
                handle=handle,
                torch=torch,
            )
        elif clip_vision_output is not None:
            raise ValueError("Wan 2.2 Fun inpaint does not consume CLIP vision output")

        common_upscale = importlib.import_module("dinkster_inference_torch.resize").common_upscale
        resized_start = None
        if start_image is not None:
            image = _wan_image(start_image, "start_image", torch)
            resized_start = common_upscale(
                image[:length, :, :, :3].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)
        resized_end = None
        if end_image is not None:
            image = _wan_image(end_image, "end_image", torch)
            resized_end = common_upscale(
                image[-length:, :, :, :3].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)
        image_device = (
            resized_start.device
            if resized_start is not None
            else resized_end.device
            if resized_end is not None
            else torch.device("cpu")
        )
        image = torch.full(
            (length, height, width, 3),
            0.5,
            device=image_device,
            dtype=torch.float32,
        )
        if resized_start is not None:
            image[: resized_start.shape[0]] = resized_start.to(image)
        if resized_end is not None:
            image[-resized_end.shape[0] :] = resized_end.to(image)
        spatial_scale = 16 if channels == 48 else 8
        latent_frames = ((length - 1) // 4) + 1
        latent_spatial = (height // spatial_scale, width // spatial_scale)
        content = image.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
        with handle.stage("vae"):
            with torch.inference_mode():
                encoded = _wan_encode_content(
                    handle=handle,
                    content=content,
                    expected=(1, channels, latent_frames, *latent_spatial),
                    name="inpaint video",
                    torch=torch,
                )
        external_mask = torch.ones(
            (1, 1, latent_frames * 4, *latent_spatial),
            device=encoded.device,
            dtype=encoded.dtype,
        )
        if resized_start is not None:
            external_mask[:, :, : resized_start.shape[0] + 3] = 0.0
        if resized_end is not None:
            external_mask[:, :, -resized_end.shape[0] :] = 0.0
        model_mask = _wan_flf_model_mask(external_mask, latent_frames)
        concat_latent = torch.cat((model_mask, encoded), dim=1)
        inference = importlib.import_module("dinkster_inference")
        prepared_positive = _wan_fun_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            concat_mask_index=0,
            vision=vision,
        )
        prepared_negative = _wan_fun_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            concat_latent=concat_latent,
            concat_mask_index=0,
            vision=vision,
        )
        latent = torch.zeros(
            (batch_size, channels, latent_frames, *latent_spatial),
            device="cpu",
            dtype=encoded.dtype,
        )
        return cls.outputs(
            positive=prepared_positive,
            negative=prepared_negative,
            latent={"samples": latent},
        )


def _wan_vace_prepared(
    value: object,
    *,
    name: str,
    handle: NativeRuntimeHandle,
    inference: Any,
    frames: Any,
    mask: Any,
    strength: float,
) -> list[list[object]]:
    text = _prepared_multistream_conditioning(
        value,
        inference,
        name,
        handle.runtime.conditioning_identity,
    )
    if text is None:
        raise TypeError(f"{name} must contain Wan 2.1 text conditioning")
    prepared = handle.runtime.prepare_vace_conditioning(text, frames, mask, strength)
    return [
        [
            inference.PreparedMultiStreamConditioning(
                handle.runtime.conditioning_identity,
                prepared,
            ),
            {},
        ]
    ]


class NativeWanVaceToVideo(WanVaceToVideo):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        strength: float,
        control_video: object = None,
        control_masks: object = None,
        reference_image: object = None,
    ) -> Mapping[str, object]:
        if width < 16 or width > 16384 or width % 16 != 0:
            raise ValueError("width must be a multiple of 16 between 16 and 16384")
        if height < 16 or height > 16384 or height % 16 != 0:
            raise ValueError("height must be a multiple of 16 between 16 and 16384")
        if length < 1 or length > 16384 or (length - 1) % 4 != 0:
            raise ValueError("length must be 1 plus a multiple of 4 between 1 and 16384")
        if batch_size < 1 or batch_size > 4096:
            raise ValueError("batch_size must be between 1 and 4096")
        if (
            type(strength) not in (int, float)
            or not math.isfinite(strength)
            or not 0 <= strength <= 1000
        ):
            raise ValueError("strength must be finite and between 0 and 1000")
        strength = float(strength)

        handle = _native_handle(vae, "vae")
        config = handle.runtime.assembled.diffusion.config
        if config.vace_layers is None:
            raise ValueError("vae must come from a Wan 2.1 VACE profile")

        torch = _torch()
        common_upscale = importlib.import_module("dinkster_inference_torch.resize").common_upscale
        if control_video is None:
            control = torch.full(
                (length, height, width, 3),
                0.5,
                device="cpu",
                dtype=torch.float32,
            )
        else:
            image = _wan_image(control_video, "control_video", torch)
            resized = common_upscale(
                image[:length, :, :, :3].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)
            control = torch.full(
                (length, height, width, 3),
                0.5,
                device=resized.device,
                dtype=resized.dtype,
            )
            control[: resized.shape[0]] = resized

        if control_masks is None:
            mask = torch.ones(
                (length, height, width, 1),
                device=control.device,
                dtype=control.dtype,
            )
        else:
            if type(control_masks) is not torch.Tensor:
                raise TypeError("control_masks must be an exact torch.Tensor")
            mask_input = cast("Any", control_masks)
            if mask_input.ndim == 3:
                mask_input = mask_input.unsqueeze(1)
            if (
                mask_input.ndim != 4
                or mask_input.shape[0] <= 0
                or mask_input.shape[1] != 1
                or mask_input.shape[2] <= 0
                or mask_input.shape[3] <= 0
                or not mask_input.is_floating_point()
                or mask_input.layout != torch.strided
            ):
                raise ValueError(
                    "control_masks must be a nonempty strided floating [frames,H,W] tensor"
                )
            resized_mask = common_upscale(
                mask_input[:length], width, height, "bilinear", "center"
            ).movedim(1, -1)
            mask = torch.ones(
                (length, height, width, 1),
                device=control.device,
                dtype=control.dtype,
            )
            mask[: resized_mask.shape[0]] = resized_mask.to(
                device=control.device,
                dtype=control.dtype,
            )

        centered = control - 0.5
        inactive = centered * (1.0 - mask) + 0.5
        reactive = centered * mask + 0.5

        reference_content = None
        if reference_image is not None:
            image = _wan_image(reference_image, "reference_image", torch)
            resized_reference = common_upscale(
                image[:1, :, :, :3].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)
            reference_content = (
                resized_reference.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
            )

        def encode(content: Any, expected_time: int, name: str) -> Any:
            direct_oom: BaseException | None = None
            encoded: Any = None
            try:
                encoded = handle.runtime.encode_content(content)
            except RuntimeError as caught:
                if not _is_accelerator_oom(caught, torch=torch, device=handle.load_device):
                    raise
                direct_oom = caught.with_traceback(None)
            if direct_oom is not None:
                encoded = _retry_tiled_vae_after_oom(
                    handle=handle,
                    value=content,
                    output_dtype=torch.float32,
                    direction="encode",
                    oom=direct_oom,
                )
            expected = (1, 16, expected_time, height // 8, width // 8)
            if type(encoded) is not torch.Tensor or tuple(encoded.shape) != expected:
                actual = getattr(encoded, "shape", None)
                raise ValueError(f"Wan VACE {name} latent has shape {actual}, expected {expected}")
            return encoded

        latent_length = ((length - 1) // 4) + 1
        inactive_content = inactive.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
        reactive_content = reactive.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
        with handle.stage("vae"):
            with torch.inference_mode():
                inactive_latent = encode(inactive_content, latent_length, "inactive")
                reactive_latent = encode(reactive_content, latent_length, "reactive")
                frames = torch.cat((inactive_latent, reactive_latent), dim=1)
                reference = None
                if reference_content is not None:
                    encoded_reference = encode(reference_content, 1, "reference")
                    neutral_reference = handle.runtime.assembled.vae.process_out(
                        torch.zeros_like(encoded_reference)
                    )
                    reference = torch.cat((encoded_reference, neutral_reference), dim=1)

        mask = mask.reshape(length, height // 8, 8, width // 8, 8)
        mask = mask.permute(2, 4, 0, 1, 3).reshape(64, length, height // 8, width // 8)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0),
            size=(latent_length, height // 8, width // 8),
            mode="nearest-exact",
        ).squeeze(0)

        trim_latent = 0
        if reference is not None:
            trim_latent = int(reference.shape[2])
            frames = torch.cat((reference, frames), dim=2)
            mask = torch.cat((torch.zeros_like(mask[:, :trim_latent]), mask), dim=1)
            latent_length += trim_latent
        mask = mask.unsqueeze(0)

        inference = importlib.import_module("dinkster_inference")
        prepared_positive = _wan_vace_prepared(
            positive,
            name="positive",
            handle=handle,
            inference=inference,
            frames=frames,
            mask=mask,
            strength=strength,
        )
        prepared_negative = _wan_vace_prepared(
            negative,
            name="negative",
            handle=handle,
            inference=inference,
            frames=frames,
            mask=mask,
            strength=strength,
        )
        latent = torch.zeros(
            (batch_size, 16, latent_length, height // 8, width // 8),
            device="cpu",
            dtype=frames.dtype,
        )
        return cls.outputs(
            positive=prepared_positive,
            negative=prepared_negative,
            latent={"samples": latent},
            trim_latent=trim_latent,
        )


class NativeWan22ImageToVideoLatent(Wan22ImageToVideoLatent):
    @classmethod
    def execute(
        cls,
        *,
        vae: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        start_image: object = None,
    ) -> Mapping[str, object]:
        if width < 32 or width > 16384 or width % 16 != 0:
            raise ValueError("width must be a multiple of 16 between 32 and 16384")
        if height < 32 or height > 16384 or height % 16 != 0:
            raise ValueError("height must be a multiple of 16 between 32 and 16384")
        if length < 1 or length > 16384 or (length - 1) % 4 != 0:
            raise ValueError("length must be 1 plus a multiple of 4 between 1 and 16384")
        if batch_size < 1 or batch_size > 4096:
            raise ValueError("batch_size must be between 1 and 4096")
        handle = _native_handle(vae, "vae")
        if handle.runtime.assembled.diffusion.config.model_type != "ti2v":
            raise ValueError("vae must come from the Wan 2.2 TI2V profile")
        torch = _torch()
        latent_frames = ((length - 1) // 4) + 1
        latent_shape = (1, 48, latent_frames, height // 16, width // 16)
        if start_image is None:
            latent = torch.zeros(latent_shape, device="cpu")
            return cls.outputs(latent={"samples": latent})
        if type(start_image) is not torch.Tensor:
            raise TypeError("start_image must be an exact torch.Tensor")
        image = cast("Any", start_image)
        if (
            image.ndim != 4
            or image.shape[0] <= 0
            or image.shape[1] <= 0
            or image.shape[2] <= 0
            or image.shape[3] < 3
            or not image.is_floating_point()
            or image.layout != torch.strided
        ):
            raise ValueError(
                "start_image must be a nonempty strided floating [frames,H,W,C>=3] tensor"
            )
        resized = (
            importlib.import_module("dinkster_inference_torch.resize")
            .common_upscale(
                image[:length, :, :, :3].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            )
            .movedim(1, -1)
        )
        content = resized.permute(3, 0, 1, 2).unsqueeze(0).to(handle.load_device)
        with handle.stage("vae"):
            with torch.inference_mode():
                direct_oom: BaseException | None = None
                encoded: Any = None
                try:
                    encoded = handle.runtime.encode_content(content)
                except RuntimeError as caught:
                    if not _is_accelerator_oom(caught, torch=torch, device=handle.load_device):
                        raise
                    direct_oom = caught.with_traceback(None)
                if direct_oom is not None:
                    encoded = _retry_tiled_vae_after_oom(
                        handle=handle,
                        value=content,
                        output_dtype=torch.float32,
                        direction="encode",
                        oom=direct_oom,
                    )
                expected = (
                    1,
                    48,
                    ((int(resized.shape[0]) - 1) // 4) + 1,
                    height // 16,
                    width // 16,
                )
                if type(encoded) is not torch.Tensor or tuple(encoded.shape) != expected:
                    actual = getattr(encoded, "shape", None)
                    raise ValueError(f"Wan 2.2 VAE latent has shape {actual}, expected {expected}")
                latent = torch.zeros(latent_shape, device=encoded.device, dtype=encoded.dtype)
                latent[:, :, : encoded.shape[2]] = encoded
                mask = torch.ones(
                    (1, 1, latent_frames, height // 16, width // 16),
                    device=encoded.device,
                    dtype=encoded.dtype,
                )
                mask[:, :, : encoded.shape[2]] = 0.0
                external = handle.runtime.assembled.vae.process_out(latent) * mask + latent * (
                    1.0 - mask
                )
        output = external.to("cpu").repeat(batch_size, 1, 1, 1, 1)
        output_mask = mask.to("cpu").repeat(batch_size, 1, 1, 1, 1)
        return cls.outputs(latent={"samples": output, "noise_mask": output_mask})
