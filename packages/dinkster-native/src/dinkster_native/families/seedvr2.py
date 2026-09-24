"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from ..native_arm_core import (
    Any,
    Mapping,
    NativeComponentHandle,
    Node,
    NodeSchema,
    Sequence,
    VAEDecodeAudio,
    VAEDecodeAudioTiled,
    _is_accelerator_oom,
    _torch,
    cast,
    importlib,
    log,
    select_load_device,
)
from ..native_arm_latent_utils import _plain_latent
from ..native_arm_runtime import (
    _application_chain_model,
    _native_model,
)
from ..nodes_provider import (
    _generation_provider_schema,
)
from ..nodes_sampling_runtime import (
    NativeVAEDecode,
    NativeVAEEncode,
    _native_component_codec,
)


def load_component(value: object, name: str, role: str | None = None) -> NativeComponentHandle:
    inference = importlib.import_module("dinkster_inference")
    handle = inference.require_inference_component_handle(value, name)
    recipe = getattr(handle, "recipe", None)
    if recipe is not None and recipe.runtime_identity != handle.resource_identity:
        raise TypeError(f"{name} component identity does not match its recipe")
    return handle


class CodecAdapter:
    sequence_content = True
    accepts_batched_video = True
    accepts_image_batch_latent = True
    manages_input_device = True

    def __init__(self, value: object) -> None:
        self._handle = load_component(value, "vae", "vae")
        recipe = getattr(self._handle, "recipe", None)
        dtype = None if recipe is None else recipe.knobs.vae_dtype
        if recipe is None:
            log.warning("SeedVR2 codec component has no reconstruction recipe; using VAE dtype")
        native = importlib.import_module("dinkster_native.native_arm")
        self._runtime = importlib.import_module("dinkster_inference_torch").SeedVR2CodecRuntime(
            self._handle.component,
            compute_dtype=None if dtype is None else native._torch_dtype(native._torch(), dtype),
        )
        self._resource_identity = self._handle.resource_identity
        self.descriptor = self._runtime.codec.descriptor
        self.load_device = self._handle.load_device

    @property
    def _dinkster_resident_owner(self) -> NativeComponentHandle:
        return self._handle

    @property
    def resource_identity(self) -> str:
        return self._resource_identity

    def require_active(self) -> None:
        self._handle.require_active()

    def stage(self) -> Any:
        return self._handle.stage()

    def decode_latent(self, latent: Any) -> Any:
        return self._runtime.decode_latent(latent)

    def decode_latent_tiled(
        self, latent: Any, *, tile: tuple[int, ...], overlap: tuple[int, ...]
    ) -> Any:
        return self._runtime.decode_latent_tiled(latent, tile=tile, overlap=overlap)

    def encode_content(self, content: Any) -> Any:
        return self._runtime.encode_content(content)

    def encode_content_tiled(
        self, content: Any, *, tile: tuple[int, ...], overlap: tuple[int, ...]
    ) -> Any:
        return self._runtime.encode_content_tiled(content, tile=tile, overlap=overlap)


def _decode_minimax_music3_audio(
    samples: object,
    vae: object,
    *,
    tile_size: int | None = None,
    overlap: int | None = None,
) -> dict[str, object]:
    if not isinstance(samples, Mapping):
        raise TypeError("samples must be a latent mapping")
    torch = _torch()
    latent = cast("Mapping[object, object]", samples).get("samples")
    inference = importlib.import_module("dinkster_inference")
    if type(latent) is inference.MultiStreamLatent:
        streams = cast("Any", latent)
        if "audio" not in streams.roles:
            raise TypeError("samples['samples'] must contain an audio stream")
        latent = streams.by_role("audio")
    if not isinstance(latent, torch.Tensor):
        raise TypeError("samples['samples'] must be a torch.Tensor")
    latent = cast("Any", latent)
    if getattr(latent, "is_nested", False):
        latent = latent.unbind()[-1]
    music3_shape = latent.ndim == 3 and latent.shape[1] == 128
    h3_shape = latent.ndim == 4 and latent.shape[1] == 32
    if latent.shape[0] < 1 or not (music3_shape or h3_shape):
        raise ValueError(
            "samples['samples'] must be nonempty [batch,128,frames] or [batch,32,time,frequency]"
        )
    codec = _native_component_codec(vae)
    with codec.stage():
        with torch.inference_mode():
            load_latent = latent.to(codec.load_device)
            if tile_size is not None:
                assert overlap is not None
                audio = codec.decode_latent_tiled(
                    load_latent, tile=(tile_size,), overlap=(overlap,)
                )
            else:
                direct_oom: BaseException | None = None
                audio = None
                try:
                    audio = codec.decode_latent(load_latent)
                except RuntimeError as caught:
                    if not _is_accelerator_oom(caught, torch=torch, device=codec.load_device):
                        raise
                    direct_oom = caught.with_traceback(None)
                if direct_oom is not None:
                    log.warning(
                        "WARNING: %s out of memory during direct MiniMax Music 3 DAV decode; "
                        "retrying with tiled decode",
                        codec.load_device.type.upper(),
                    )
                    importlib.import_module("dinkster_inference_torch").soft_empty_cache(
                        codec.load_device
                    )
                    audio = codec.decode_latent_tiled(load_latent, tile=(256,), overlap=(32,))
    assert audio is not None
    audio = audio.to(device="cpu", dtype=torch.float32, copy=True)
    if audio.ndim != 3 or audio.shape[0] < 1 or audio.shape[1] != 2 or audio.shape[2] < 1:
        raise ValueError("MiniMax Music 3 DAV must return nonempty [batch,2,samples] audio")
    std = torch.std(audio, dim=(1, 2), keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio /= std
    sample_rate = cast("Mapping[object, object]", samples).get(
        "sample_rate", getattr(codec, "sample_rate", 44100)
    )
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("samples sample_rate must be a positive integer")
    return {"waveform": audio, "sample_rate": sample_rate}


class NativeVAEDecodeAudio(VAEDecodeAudio):
    @classmethod
    def execute(cls, *, samples: object, vae: object) -> Mapping[str, object]:
        return cls.outputs(audio=_decode_minimax_music3_audio(samples, vae))


class NativeVAEDecodeAudioTiled(VAEDecodeAudioTiled):
    @classmethod
    def execute(
        cls, *, samples: object, vae: object, tile_size: int, overlap: int
    ) -> Mapping[str, object]:
        if type(tile_size) is not int or not 32 <= tile_size <= 8192:
            raise ValueError("tile_size must be an integer in [32, 8192]")
        if type(overlap) is not int or not 0 <= overlap <= 1024:
            raise ValueError("overlap must be an integer in [0, 1024]")
        return cls.outputs(
            audio=_decode_minimax_music3_audio(
                samples,
                vae,
                tile_size=tile_size,
                overlap=overlap,
            )
        )


class GenerationVAEDecode(NativeVAEDecode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vae_decode")

    @classmethod
    def execute(cls, *, samples: object, vae: object) -> Mapping[str, object]:
        inference = importlib.import_module("dinkster_inference")
        component_codec = isinstance(vae, NativeComponentHandle)
        codec = (
            _native_component_codec(vae)
            if component_codec
            else inference.require_inference_codec_handle(vae, "vae")
        )
        torch = _torch()
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a latent mapping")
        latent = cast("Mapping[object, object]", samples).get("samples")
        if type(latent) is inference.MultiStreamLatent:
            streams = cast("Any", latent)
            if "video" not in streams.roles:
                raise TypeError("samples['samples'] must contain a video stream")
            latent = streams.by_role("video")
        if not isinstance(latent, torch.Tensor):
            raise TypeError("samples['samples'] must be a torch.Tensor")
        latent_tensor = cast("Any", latent)
        if len(latent_tensor.shape) not in (4, 5):
            raise ValueError(
                "samples['samples'] must be NCHW or NCTHW rank 4/5, "
                f"got shape {tuple(latent_tensor.shape)}"
            )
        memory_required = getattr(codec, "decode_memory_required", None)
        stage = (
            codec.stage(memory_required=memory_required(latent_tensor))
            if callable(memory_required)
            else codec.stage()
        )
        with stage:
            with torch.inference_mode():
                image = codec.decode_latent(
                    latent_tensor
                    if getattr(codec, "manages_input_device", False)
                    else latent_tensor.to(codec.load_device)
                )
        return cls.outputs(image=_generation_decoded_image(image, codec, component_codec))


def _generation_decoded_image(image: Any, codec: Any, component_codec: bool) -> Any:
    if (
        codec.descriptor.kind == "video"
        and (component_codec or getattr(codec, "accepts_image_batch_latent", False))
        and len(image.shape) == 4
    ):
        if image.shape[1] != 3:
            raise ValueError(
                f"image-capable video codec decode must return [B,3,H,W], got {tuple(image.shape)}"
            )
        image = image.permute(0, 2, 3, 1)
    elif codec.descriptor.kind == "video":
        channels = codec.descriptor.content_channels
        if len(image.shape) != 5 or image.shape[1] != channels:
            raise ValueError(
                f"video codec decode must return [B,{channels},T,H,W], got {tuple(image.shape)}"
            )
        image = image.permute(0, 2, 3, 4, 1).flatten(0, 1)
    else:
        channels = codec.descriptor.content_channels
        if len(image.shape) != 4 or image.shape[1] != channels:
            raise ValueError(
                f"image codec decode must return [B,{channels},H,W], got {tuple(image.shape)}"
            )
        image = image.permute(0, 2, 3, 1)
    return image


class GenerationVAEDecodeTiled(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vae_decode_tiled")

    @classmethod
    def execute(
        cls,
        *,
        samples: object,
        vae: object,
        tile_size: int,
        overlap: int = 64,
        temporal_size: int = 64,
        temporal_overlap: int = 8,
    ) -> Mapping[str, object]:
        values = (tile_size, overlap, temporal_size, temporal_overlap)
        if any(type(value) is not int for value in values):
            raise TypeError("tiled VAE sizes must be exact integers")
        if tile_size <= 0 or overlap < 0 or temporal_size <= 0 or temporal_overlap < 0:
            raise ValueError("tiled VAE sizes must be positive and overlaps nonnegative")
        inference = importlib.import_module("dinkster_inference")
        component_codec = isinstance(vae, NativeComponentHandle)
        codec = inference.require_inference_tiled_codec_handle(
            _native_component_codec(vae) if component_codec else vae,
            "vae",
        )
        torch = _torch()
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a latent mapping")
        latent = cast("Mapping[object, object]", samples).get("samples")
        if type(latent) is inference.MultiStreamLatent:
            streams = cast("Any", latent)
            if "video" not in streams.roles:
                raise TypeError("samples['samples'] must contain a video stream")
            latent = streams.by_role("video")
        if not isinstance(latent, torch.Tensor):
            raise TypeError("samples['samples'] must be a torch.Tensor")
        latent_tensor = cast("Any", latent)
        dimensions = codec.descriptor.latent.dimensions
        expected_rank = dimensions + 2
        accepts_image_batch = getattr(codec, "accepts_image_batch_latent", False)
        if len(latent_tensor.shape) != expected_rank and not (
            accepts_image_batch and len(latent_tensor.shape) == expected_rank - 1
        ):
            raise ValueError(
                f"samples['samples'] must be rank {expected_rank}, "
                f"got shape {tuple(latent_tensor.shape)}"
            )
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_size // 2
        spatial_scale = codec.descriptor.latent.spatial_downscale
        spatial_tile = tile_size // spatial_scale
        spatial_overlap = overlap // spatial_scale
        if dimensions == 3:
            temporal_scale = codec.descriptor.latent.temporal_downscale
            temporal_size = max(2, temporal_size // temporal_scale)
            temporal_overlap = max(
                1,
                min(temporal_size // 2, temporal_overlap // temporal_scale),
            )
            tile = (temporal_size, spatial_tile, spatial_tile)
            tile_overlap = (temporal_overlap, spatial_overlap, spatial_overlap)
        elif dimensions == 2:
            tile = (spatial_tile, spatial_tile)
            tile_overlap = (spatial_overlap, spatial_overlap)
        else:
            tile = (spatial_tile,)
            tile_overlap = (spatial_overlap,)
        if any(value <= 0 for value in tile) or any(
            overlap_value >= tile_value
            for overlap_value, tile_value in zip(tile_overlap, tile, strict=True)
        ):
            raise ValueError("tiled VAE sizes do not produce a valid latent tile")
        with codec.stage():
            with torch.inference_mode():
                image = codec.decode_latent_tiled(
                    (
                        latent_tensor
                        if getattr(codec, "manages_input_device", False)
                        else latent_tensor.to(codec.load_device)
                    ),
                    tile=tile,
                    overlap=tile_overlap,
                )
        return cls.outputs(image=_generation_decoded_image(image, codec, component_codec))


class GenerationVAEEncode(NativeVAEEncode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vae_encode")

    @classmethod
    def execute(cls, *, pixels: object, vae: object) -> Mapping[str, object]:
        inference = importlib.import_module("dinkster_inference")
        component_codec = isinstance(vae, NativeComponentHandle)
        codec = (
            _native_component_codec(vae)
            if component_codec
            else inference.require_inference_codec_handle(vae, "vae")
        )
        torch = _torch()
        if not isinstance(pixels, torch.Tensor):
            raise TypeError("pixels must be a torch.Tensor")
        pixel_tensor = cast("Any", pixels)
        batched_video = getattr(codec, "accepts_batched_video", False)
        if batched_video and len(pixel_tensor.shape) == 5 and pixel_tensor.shape[-1] == 3:
            content = pixel_tensor.permute(0, 4, 1, 2, 3)
        else:
            if len(pixel_tensor.shape) != 4 or pixel_tensor.shape[-1] != 3:
                raise ValueError(
                    f"pixels must be NHWC rank 4, got shape {tuple(pixel_tensor.shape)}"
                )
            content = pixel_tensor.permute(0, 3, 1, 2)
        if (
            codec.descriptor.kind == "video"
            and len(pixel_tensor.shape) == 4
            and (not batched_video or component_codec)
            and (not component_codec or getattr(codec, "sequence_content", False))
        ):
            content = content.permute(1, 0, 2, 3).unsqueeze(0)
        memory_required = getattr(codec, "encode_memory_required", None)
        stage = (
            codec.stage(memory_required=memory_required(content))
            if callable(memory_required)
            else codec.stage()
        )
        with stage:
            with torch.inference_mode():
                latent = codec.encode_content(
                    content
                    if getattr(codec, "manages_input_device", False)
                    else content.to(codec.load_device)
                )
        return cls.outputs(latent={"samples": latent})


class GenerationVAEEncodeTiled(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.vae_encode_tiled")

    @classmethod
    def execute(
        cls,
        *,
        pixels: object,
        vae: object,
        tile_size: int,
        overlap: int,
        temporal_size: int,
        temporal_overlap: int,
    ) -> Mapping[str, object]:
        inference = importlib.import_module("dinkster_inference")
        component_codec = isinstance(vae, NativeComponentHandle)
        codec = inference.require_inference_tiled_codec_handle(
            _native_component_codec(vae) if component_codec else vae,
            "vae",
        )
        torch = _torch()
        if not isinstance(pixels, torch.Tensor):
            raise TypeError("pixels must be a torch.Tensor")
        pixel_tensor = cast("Any", pixels)
        batched_video = getattr(codec, "accepts_batched_video", False)
        if batched_video and len(pixel_tensor.shape) == 5 and pixel_tensor.shape[-1] == 3:
            content = pixel_tensor.permute(0, 4, 1, 2, 3)
        else:
            if len(pixel_tensor.shape) != 4 or pixel_tensor.shape[-1] != 3:
                raise ValueError(
                    f"pixels must be NHWC rank 4, got shape {tuple(pixel_tensor.shape)}"
                )
            content = pixel_tensor.permute(0, 3, 1, 2)
        if (
            codec.descriptor.kind == "video"
            and len(pixel_tensor.shape) == 4
            and (not batched_video or component_codec)
            and (not component_codec or getattr(codec, "sequence_content", False))
        ):
            content = content.permute(1, 0, 2, 3).unsqueeze(0)
        dimensions = codec.descriptor.latent.dimensions
        if dimensions == 3:
            tile = (temporal_size, tile_size, tile_size)
            tile_overlap = (temporal_overlap, overlap, overlap)
        elif dimensions == 2:
            tile = (tile_size, tile_size)
            tile_overlap = (overlap, overlap)
        else:
            tile = (tile_size,)
            tile_overlap = (overlap,)
        with codec.stage():
            with torch.inference_mode():
                latent = codec.encode_content_tiled(
                    content
                    if getattr(codec, "manages_input_device", False)
                    else content.to(codec.load_device),
                    tile=tile,
                    overlap=tile_overlap,
                )
        return cls.outputs(latent={"samples": latent})


def _seedvr2_bthwc(value: Any, name: str) -> tuple[Any, bool]:
    if value.ndim == 4:
        return value.unsqueeze(0), True
    if value.ndim == 5:
        return value, False
    raise ValueError(f"{name}: expected 4-D or 5-D IMAGE tensor, got shape {tuple(value.shape)}")


class GenerationSeedVR2Preprocess(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.seedvr2_preprocess")

    @classmethod
    def execute(cls, *, resized_images: object) -> Mapping[str, object]:
        torch = _torch()
        if not isinstance(resized_images, torch.Tensor):
            raise TypeError("resized_images must be a torch.Tensor")
        images = cast("Any", resized_images)
        images, _ = _seedvr2_bthwc(images, "SeedVR2Preprocess")
        if images.shape[1] < 1:
            raise ValueError("SeedVR2Preprocess expected at least one frame.")
        if min(images.shape[2], images.shape[3]) < 2:
            raise ValueError("SeedVR2Preprocess: input shorter edge must be at least 2 pixels")
        if images.shape[-1] > 3:
            images = images[..., :3]
        elif images.shape[-1] != 3:
            raise ValueError("SeedVR2Preprocess: input must have at least three channels")
        images = images.permute(0, 1, 4, 2, 3)
        batch, frames, channels, height, width = images.shape
        images = images.reshape(batch * frames, channels, height, width).clamp(0.0, 1.0)
        pad_height = (16 - height % 16) % 16
        pad_width = (16 - width % 16) % 16
        if pad_height or pad_width:
            images = torch.nn.functional.pad(images, (0, pad_width, 0, pad_height))
        images = images.reshape(batch, frames, channels, height + pad_height, width + pad_width)
        if frames > 1 and (frames - 1) % 4:
            padding = images[:, -1:].repeat(1, 4 - (frames - 1) % 4, 1, 1, 1)
            images = torch.cat((images, padding), dim=1)
        return cls.outputs(images=images.permute(0, 1, 3, 4, 2).contiguous())


def _seedvr2_restore_reference_shape(decoded: Any, reference: Any) -> Any:
    if decoded.shape[0] != 1:
        return decoded
    reference_batch, reference_frames = reference.shape[:2]
    if reference_batch < 1 or decoded.shape[1] % reference_batch:
        return decoded
    decoded_frames = decoded.shape[1] // reference_batch
    if decoded_frames < reference_frames:
        return decoded
    return decoded.reshape(
        reference_batch,
        decoded_frames,
        decoded.shape[2],
        decoded.shape[3],
        decoded.shape[4],
    )


def _seedvr2_resize_reference(reference: Any, height: int, width: int, torch: Any) -> Any:
    if reference.shape[2:4] == (height, width):
        return reference
    batch, frames = reference.shape[:2]
    flat = reference.permute(0, 1, 4, 2, 3).reshape(
        batch * frames,
        reference.shape[4],
        reference.shape[2],
        reference.shape[3],
    )
    resized = torch.nn.functional.interpolate(
        flat,
        size=(height, width),
        mode="bicubic",
        antialias=flat.device.type != "mps",
    )
    return resized.reshape(batch, frames, resized.shape[1], height, width).permute(0, 1, 3, 4, 2)


def _seedvr2_color_chunk_size(flat: Any, method: str, torch: Any, inference_torch: Any) -> int:
    constants = importlib.import_module("dinkster_inference_torch.seedvr2_constants")
    multiplier = {
        "lab": constants.SEEDVR2_LAB_SCALE_MULTIPLIER,
        "wavelet": constants.SEEDVR2_WAVELET_SCALE_MULTIPLIER,
        "adain": constants.SEEDVR2_ADAIN_SCALE_MULTIPLIER,
    }[method]
    frames, channels, height, width = flat.shape
    dtype_bytes = max(flat.element_size(), constants.SEEDVR2_DTYPE_BYTES_FLOOR)
    bytes_per_frame = height * width * channels * dtype_bytes * multiplier
    if bytes_per_frame <= 0:
        return frames
    device = select_load_device(torch)
    free_memory = inference_torch.get_free_memory(device).free_total
    available = int((free_memory * constants.SEEDVR2_COLOR_MEM_HEADROOM) // bytes_per_frame)
    return max(1, min(frames, available))


def _seedvr2_color_transfer(
    decoded: Any,
    reference: Any,
    method: str,
    torch: Any,
    inference_torch: Any,
) -> Any:
    transfer = {
        "lab": inference_torch.lab_color_transfer,
        "wavelet": inference_torch.wavelet_color_transfer,
        "adain": inference_torch.adain_color_transfer,
    }[method]
    color_device = select_load_device(torch)
    output_device = decoded.device
    chunk_size = _seedvr2_color_chunk_size(decoded, method, torch, inference_torch)
    while True:
        result = None
        try:
            for start in range(0, decoded.shape[0], chunk_size):
                end = min(start + chunk_size, decoded.shape[0])
                if method == "lab":
                    for index in range(start, end):
                        output = transfer(
                            decoded[index : index + 1].to(color_device).clone(),
                            reference[index : index + 1].to(color_device).clone(),
                        ).to(output_device)
                        if result is None:
                            result = torch.empty(
                                (decoded.shape[0],) + tuple(output.shape[1:]),
                                device=output_device,
                                dtype=output.dtype,
                            )
                        result[index : index + 1].copy_(output)
                else:
                    output = transfer(
                        decoded[start:end].to(color_device),
                        reference[start:end].to(color_device),
                    ).to(output_device)
                    if result is None:
                        result = torch.empty(
                            (decoded.shape[0],) + tuple(output.shape[1:]),
                            device=output_device,
                            dtype=output.dtype,
                        )
                    result[start:end].copy_(output)
            if result is None:
                raise ValueError(
                    "SeedVR2PostProcessing: color correction requires at least one frame."
                )
            return result
        except RuntimeError as error:
            if not _is_accelerator_oom(error, torch=torch, device=color_device):
                raise
            if chunk_size <= 1:
                raise RuntimeError(
                    "SeedVR2PostProcessing: color correction OOM at one frame"
                ) from error
            chunk_size = max(1, chunk_size // 2)


class GenerationSeedVR2PostProcessing(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.seedvr2_postprocess")

    @classmethod
    def execute(
        cls,
        *,
        images: object,
        original_resized_images: object,
        color_correction_method: str,
    ) -> Mapping[str, object]:
        torch = _torch()
        if not isinstance(images, torch.Tensor) or not isinstance(
            original_resized_images, torch.Tensor
        ):
            raise TypeError("images and original_resized_images must be torch.Tensor values")
        if color_correction_method not in ("lab", "wavelet", "adain", "none"):
            raise ValueError(
                "SeedVR2PostProcessing: unknown color_correction_method "
                f"{color_correction_method!r}"
            )
        decoded, decoded_was_4d = _seedvr2_bthwc(cast("Any", images), "SeedVR2PostProcessing")
        original = cast("Any", original_resized_images)
        alpha = original[..., 3:4] if original.shape[-1] == 4 else None
        if original.shape[-1] >= 3:
            original = original[..., :3]
        reference, _ = _seedvr2_bthwc(original, "SeedVR2PostProcessing")
        decoded = _seedvr2_restore_reference_shape(decoded, reference)
        batch = min(decoded.shape[0], reference.shape[0])
        frames = min(decoded.shape[1], reference.shape[1])
        height = min(decoded.shape[2], reference.shape[2])
        width = min(decoded.shape[3], reference.shape[3])
        decoded = decoded[:batch, :frames, :height, :width]
        if color_correction_method == "none":
            output = decoded
        else:
            reference = _seedvr2_resize_reference(reference[:batch, :frames], height, width, torch)
            decoded_flat = (
                decoded.mul(2.0)
                .sub(1.0)
                .permute(0, 1, 4, 2, 3)
                .reshape(batch * frames, decoded.shape[4], height, width)
            )
            reference_flat = (
                reference.mul(2.0)
                .sub(1.0)
                .permute(0, 1, 4, 2, 3)
                .reshape(batch * frames, reference.shape[4], height, width)
            )
            inference_torch = importlib.import_module("dinkster_inference_torch")
            corrected = _seedvr2_color_transfer(
                decoded_flat,
                reference_flat,
                color_correction_method,
                torch,
                inference_torch,
            )
            output = corrected.reshape(
                batch, frames, corrected.shape[1], corrected.shape[2], corrected.shape[3]
            ).permute(0, 1, 3, 4, 2)
            output = output.add(1.0).div(2.0).clamp(0.0, 1.0)
        if alpha is not None:
            alpha, _ = _seedvr2_bthwc(alpha, "SeedVR2PostProcessing")
            alpha = alpha[:batch, :frames, : output.shape[2], : output.shape[3]]
            output = torch.cat((output, alpha.to(device=output.device, dtype=output.dtype)), dim=-1)
        output = output[:, :, : output.shape[2] // 2 * 2, : output.shape[3] // 2 * 2]
        if decoded_was_4d:
            output = output.flatten(0, 1)
        return cls.outputs(images=output)


class GenerationSeedVR2Conditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.seedvr2_conditioning")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        vae_conditioning: object,
    ) -> Mapping[str, object]:
        model, _applications = _application_chain_model(model, "model")
        handle, *_options = _native_model(model, "model")
        torch = _torch()
        _, samples = _plain_latent(vae_conditioning, torch, "vae_conditioning")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        positive, negative = inference_torch.seedvr2_conditioning(
            samples,
            component_identity=handle.recipe.runtime_identity,
        )
        return cls.outputs(
            positive=inference_torch.seedvr2_conditioning_to_carrier(positive),
            negative=inference_torch.seedvr2_conditioning_to_carrier(negative),
        )


class GenerationSeedVR2TemporalChunk(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.seedvr2_temporal_chunk")

    @classmethod
    def execute(
        cls,
        *,
        latent: object,
        temporal_overlap: int,
        chunking_mode: object,
    ) -> Mapping[str, object]:
        torch = _torch()
        metadata, samples = _plain_latent(latent, torch, "latent")
        if samples.ndim != 5 or samples.shape[1] != 16:
            raise ValueError("SeedVR2TemporalChunk requires a [B,16,T,H,W] video latent")
        if temporal_overlap < 0:
            raise ValueError("temporal_overlap must be nonnegative")
        if not isinstance(chunking_mode, Mapping):
            raise TypeError("chunking_mode must be a dynamic-combo mapping")
        typed_mode = cast("Mapping[str, object]", chunking_mode)
        mode = typed_mode.get("chunking_mode")
        latent_frames = samples.shape[2]
        pixel_frames = 4 * (latent_frames - 1) + 1
        if mode == "auto":
            inference_torch = importlib.import_module("dinkster_inference_torch")
            constants = importlib.import_module("dinkster_inference_torch.seedvr2_constants")
            free_gib = (
                inference_torch.get_free_memory(select_load_device(torch)).free_total / 1024**3
            )
            mpx = (
                samples.shape[0]
                * samples.shape[3]
                * samples.shape[4]
                * constants.BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE**2
                / 1e6
            )
            budget = (
                free_gib
                - constants.SEEDVR2_CHUNK_RESERVED_GIB
                - constants.SEEDVR2_CHUNK_SIGMA_K * constants.SEEDVR2_CHUNK_SIGMA_GIB
            )
            maximum = max(1, int(budget / (constants.SEEDVR2_CHUNK_GIB_PER_MPX_FRAME * mpx)))
            frames_per_chunk = min(4 * (maximum - 1) + 1, pixel_frames)
        elif mode == "manual":
            frames_per_chunk = typed_mode.get("frames_per_chunk")
            if type(frames_per_chunk) is not int:
                raise TypeError("manual chunking requires integer frames_per_chunk")
            if frames_per_chunk < 1 or (frames_per_chunk - 1) % 4:
                raise ValueError("frames_per_chunk must be a 4n+1 pixel-frame count")
        else:
            raise ValueError("chunking_mode must select 'auto' or 'manual'")
        if pixel_frames <= frames_per_chunk:
            return cls.outputs(latents=[metadata], temporal_overlap=0)
        chunk_frames = (frames_per_chunk - 1) // 4 + 1
        overlap = min(temporal_overlap, chunk_frames - 1)
        step = chunk_frames - overlap
        chunks: list[dict[Any, Any]] = []
        for start in range(0, latent_frames, step):
            end = min(start + chunk_frames, latent_frames)
            chunk = dict(metadata)
            chunk["samples"] = samples[:, :, start:end].contiguous()
            chunks.append(chunk)
            if end >= latent_frames:
                break
        return cls.outputs(latents=chunks, temporal_overlap=overlap)


class GenerationSeedVR2TemporalMerge(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.seedvr2_temporal_merge")

    @classmethod
    def execute(
        cls,
        *,
        latents: object,
        temporal_overlap: int,
    ) -> Mapping[str, object]:
        if temporal_overlap < 0:
            raise ValueError("temporal_overlap must be nonnegative")
        if isinstance(latents, str | bytes) or not isinstance(latents, Sequence) or not latents:
            raise ValueError("latents must contain at least one temporal chunk")
        torch = _torch()
        typed = cast("Sequence[object]", latents)
        metadata, first = _plain_latent(typed[0], torch, "latents[0]")
        if first.ndim != 5 or first.shape[1] != 16:
            raise ValueError("SeedVR2TemporalMerge requires [B,16,T,H,W] video latents")
        chunks = [first]
        for index, value in enumerate(typed[1:], 1):
            _, chunk = _plain_latent(value, torch, f"latents[{index}]")
            if chunk.shape[:2] != first.shape[:2] or chunk.shape[3:] != first.shape[3:]:
                raise ValueError(f"latents[{index}] does not match the first chunk")
            if index < len(typed) - 1 and chunk.shape[2] != first.shape[2]:
                raise ValueError("only the final SeedVR2 temporal chunk may be shorter")
            chunks.append(chunk)
        metadata.pop("noise_mask", None)
        if len(chunks) == 1:
            metadata["samples"] = first
        elif temporal_overlap == 0:
            metadata["samples"] = torch.cat(chunks, dim=2)
        else:
            chunk_frames = first.shape[2]
            step = chunk_frames - min(temporal_overlap, chunk_frames - 1)
            total = step * (len(chunks) - 1) + chunks[-1].shape[2]
            merged = torch.empty(
                (*first.shape[:2], total, *first.shape[3:]),
                device=first.device,
                dtype=first.dtype,
            )
            merged[:, :, :chunk_frames] = first
            filled = chunk_frames
            for index, chunk in enumerate(chunks[1:], 1):
                start = index * step
                end = start + chunk.shape[2]
                fade = min(filled - start, chunk.shape[2])
                if fade > 0:
                    ramp = torch.linspace(0.0, 1.0, fade, device=chunk.device, dtype=chunk.dtype)
                    ramp = ((ramp - 1.0 / 3.0) / (1.0 / 3.0)).clamp(0.0, 1.0)
                    previous = (0.5 + 0.5 * torch.cos(torch.pi * ramp)).view(1, 1, fade, 1, 1)
                    merged[:, :, start : start + fade] = merged[
                        :, :, start : start + fade
                    ] * previous + chunk[:, :, :fade] * (1.0 - previous)
                merged[:, :, start + fade : end] = chunk[:, :, fade:]
                filled = end
            metadata["samples"] = merged
        return cls.outputs(latent=metadata)


_LATENT_RESIZE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "bislerp")
