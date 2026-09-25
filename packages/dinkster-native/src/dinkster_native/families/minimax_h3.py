"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

import hashlib

from ..family_registry import (
    load_component as _load_family_component,
)
from ..family_registry import (
    load_registered_component,
)
from ..native_arm_core import (
    _NATIVE_PREPARED_CONDITIONING_KEY,
    Any,
    ConcatAVLatent,
    EmptyLTXAVLatent,
    EmptyLTXVLatent,
    EmptyMiniMaxH3AV,
    EmptyMiniMaxMusic3LatentAudio,
    ExitStack,
    InspectLatentMask,
    Mapping,
    MiniMaxH3AddGuide,
    MiniMaxH3AudioReferenceValue,
    MiniMaxH3AVDecode,
    MiniMaxH3AVEncode,
    MiniMaxH3FL2VAConditioning,
    MiniMaxH3ImageReferenceValue,
    MiniMaxH3ImageToVideo,
    MiniMaxH3MotionContext,
    MiniMaxH3REF2VAConditioning,
    MiniMaxH3ReferenceToVideo,
    MiniMaxH3T2VAConditioning,
    MiniMaxH3VideoReferenceValue,
    NativeComponentHandle,
    NativeRuntimeHandle,
    Node,
    PreviewLatentAudio,
    PreviewLatentVisual,
    SeparateAVLatent,
    Sequence,
    SetLatentMaskFromFrames,
    SetLatentMaskFromTimeRanges,
    _component_bound_carrier,
    _not_cancelled,
    _torch,
    cast,
    current_execution_context,
    dataclass,
    importlib,
    math,
    native_execution_span,
)
from ..native_arm_runtime import (
    _native_handle,
    _NativeModelOverlay,
    _torch_dtype,
)
from .conditioning import _prepared_multistream_carrier
from .latent import _latent_samples


@dataclass(frozen=True, slots=True)
class _MiniMaxH3ResidentConditioning:
    conditioning: list[list[object]]
    owner: NativeComponentHandle
    references: tuple[NativeComponentHandle, ...]
    fingerprint: str

    @property
    def _dinkster_resident_owner(self) -> NativeComponentHandle:
        return self.owner

    @property
    def _dinkster_resident_refs(self) -> tuple[NativeComponentHandle, ...]:
        return self.references

    @property
    def _dinkster_resident_fingerprint(self) -> str:
        return self.fingerprint


def load_component(value: object, name: str, role: str | None = None) -> NativeComponentHandle:
    role_labels = {
        "qwen3vl-32b-conditioner": "conditioner",
        "video-vae": "video VAE",
        "audio-vae": "audio VAE",
    }
    expected = "component" if role is None else role_labels.get(role, "component")
    try:
        return _load_family_component(value, name, role)
    except TypeError as error:
        raise TypeError(f"{name} must be a native MiniMax H3 {expected} component") from error


def _minimax_h3_video_vae_runtime(value: object, name: str = "video_vae") -> tuple[Any, Any]:
    inference = importlib.import_module("dinkster_inference")
    handle = load_registered_component(
        value, name, "video-vae", family_id=inference.MINIMAX_H3_CONFIG.family_id
    )
    torch = _torch()
    inference_torch = importlib.import_module("dinkster_inference_torch")
    assert handle.recipe is not None
    return handle, inference_torch.MiniMaxH3VideoVaeRuntime(
        handle.component,
        runtime_identity=handle.resource_identity,
        compute_dtype=_torch_dtype(torch, handle.recipe.knobs.vae_dtype),
    )


def _minimax_h3_audio_vae_runtime(value: object, name: str = "audio_vae") -> tuple[Any, Any]:
    inference = importlib.import_module("dinkster_inference")
    handle = load_registered_component(
        value, name, "audio-vae", family_id=inference.MINIMAX_H3_CONFIG.family_id
    )
    torch = _torch()
    inference_torch = importlib.import_module("dinkster_inference_torch")
    assert handle.recipe is not None
    return handle, inference_torch.MiniMaxH3AudioVaeRuntime(
        handle.component,
        runtime_identity=handle.resource_identity,
        compute_dtype=_torch_dtype(torch, handle.recipe.knobs.vae_dtype),
    )


class CodecAdapter:
    def __init__(self, value: object) -> None:
        recipe = getattr(value, "recipe", None)
        roles = () if recipe is None else tuple(binding.role for binding in recipe.sources)
        inference = importlib.import_module("dinkster_inference")
        config = inference.MINIMAX_H3_CONFIG
        if roles == ("video-vae",):
            self._role = "video"
            self._handle, self._runtime = _minimax_h3_video_vae_runtime(value, "vae")
            self.descriptor = inference.CodecDescriptor(
                id="dinkster.minimax_h3_video_vae",
                display_name=config.video_codec_id,
                kind="video",
                latent=inference.LatentDescriptor(
                    channels=config.video_latent_channels,
                    dimensions=3,
                    spatial_downscale=config.video_spatial_downscale,
                    temporal_downscale=4,
                    temporal_causal=True,
                    content_fps=config.video_fps,
                ),
                supported_dtypes=frozenset({inference.FLOAT16, inference.FLOAT32}),
                supports_tiling=False,
            )
        elif roles == ("audio-vae",):
            self._role = "audio"
            self._handle, self._runtime = _minimax_h3_audio_vae_runtime(value, "vae")
            self.descriptor = inference.CodecDescriptor(
                id="dinkster.minimax_h3_audio_vae",
                display_name=config.audio_codec_id,
                kind="audio",
                latent=inference.LatentDescriptor(
                    channels=config.audio_latent_channels,
                    dimensions=1,
                ),
                supported_dtypes=frozenset({inference.FLOAT32}),
                content_channels=config.audio_content_channels,
                supports_tiling=False,
            )
            self.sample_rate = config.audio_sample_rate_hz
        else:
            raise TypeError(f"vae must be a native MiniMax H3 codec component, got roles={roles!r}")
        self._resource_identity = self._handle.resource_identity
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
        return self._handle.stage(clear_cache_after=True)

    def decode_latent(self, latent: Any) -> Any:
        if self._role == "video":
            return self._runtime.decode_video(latent)
        return self._runtime.decode_audio(latent).waveform

    def encode_content(self, content: Any) -> Any:
        if self._role == "video":
            return self._runtime.encode_video(content)
        inference = importlib.import_module("dinkster_inference")
        return self._runtime.encode_audio(
            inference.MiniMaxH3AudioContent(content, self.sample_rate)
        )


def _latent_mask_codec_runtime(value: object, name: str) -> Any:
    inference = importlib.import_module("dinkster_inference")
    if isinstance(value, inference.LatentMaskCodecRuntime):
        return value
    if isinstance(value, NativeComponentHandle):
        recipe = value.recipe
        roles = () if recipe is None else tuple(binding.role for binding in recipe.sources)
        if roles == ("video-vae",):
            return _minimax_h3_video_vae_runtime(value, name)[1]
        if roles == ("audio-vae",):
            return _minimax_h3_audio_vae_runtime(value, name)[1]
    if isinstance(value, NativeRuntimeHandle):
        runtime = value.runtime
        for candidate in (runtime, getattr(runtime, "codec", None)):
            if isinstance(candidate, inference.LatentMaskCodecRuntime):
                return candidate
    raise TypeError(f"{name} does not declare latent mask geometry")


def _minimax_h3_conditioner_runtime(
    clip: object,
    video_vae: object | None = None,
    audio_vae: object | None = None,
) -> tuple[NativeComponentHandle, tuple[NativeComponentHandle, ...], Any]:
    inference = importlib.import_module("dinkster_inference")
    clip_handle = load_registered_component(
        clip,
        "clip",
        "qwen3vl-32b-conditioner",
        family_id=inference.MINIMAX_H3_CONFIG.family_id,
    )
    video_handle = video_runtime = None
    if video_vae is not None:
        video_handle, video_runtime = _minimax_h3_video_vae_runtime(video_vae)
    audio_handle = audio_runtime = None
    if audio_vae is not None:
        audio_handle, audio_runtime = _minimax_h3_audio_vae_runtime(audio_vae)
    assert clip_handle.recipe is not None
    runtime = importlib.import_module("dinkster_inference_torch").MiniMaxH3ConditionerRuntime(
        clip_handle.component,
        video_runtime,
        audio_runtime,
        runtime_identity=clip_handle.resource_identity,
    )
    codec_handles = tuple(handle for handle in (video_handle, audio_handle) if handle is not None)
    return clip_handle, cast("tuple[NativeComponentHandle, ...]", codec_handles), runtime


def _minimax_h3_av(value: object, torch: Any, inference: Any, name: str) -> Any:
    if not isinstance(value, Mapping) or "samples" not in value:
        raise TypeError(f"{name} must be a LATENT mapping containing 'samples'")
    streams = cast("Any", cast("Mapping[object, object]", value)["samples"])
    if type(streams) is not inference.MultiStreamLatent or streams.roles != ("video", "audio"):
        raise TypeError(f"{name} samples must have exact ordered video/audio streams")
    video, audio = cast("tuple[Any, Any]", tuple(stream.payload for stream in streams.streams))
    if type(video) is not torch.Tensor or type(audio) is not torch.Tensor:
        raise TypeError(f"{name} streams must be exact torch.Tensor values")
    if (
        not video.is_floating_point()
        or not audio.is_floating_point()
        or video.layout != torch.strided
        or audio.layout != torch.strided
    ):
        raise TypeError(f"{name} streams must be strided floating tensors")
    if (
        video.ndim != 5
        or tuple(video.shape[:2]) != (1, 24)
        or audio.ndim != 4
        or tuple(audio.shape[:3]) != (1, 32, 2)
        or min(video.shape[2:]) <= 0
        or audio.shape[3] <= 0
    ):
        raise ValueError(f"{name} must contain exact batch-one MiniMax H3 latent streams")
    return streams


def _move_multistream_latent(value: Any, device: object) -> Any:
    def move(payload: Any) -> Any:
        return payload.to(device)

    return value.map(move)


def _minimax_h3_payload(inference: Any, tensor: Any, reference_id: str) -> tuple[Any, Any]:
    descriptor = inference.PayloadDescriptor(
        inference.PayloadReference(reference_id),
        tuple(tensor.shape),
        str(tensor.dtype).removeprefix("torch."),
        "worker:minimax-h3",
    )
    return descriptor, tensor


def _minimax_h3_tensor_bytes(tensor: Any, torch: Any, name: str) -> bytes:
    try:
        return tensor.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    except (AttributeError, RuntimeError, TypeError) as error:
        raise TypeError(f"{name} must expose stable tensor bytes") from error


def _minimax_h3_image_tensor(value: object, torch: Any, name: str) -> Any:
    return _minimax_h3_image_batch(value, torch, name)[:1]


def _minimax_h3_image_batch(value: object, torch: Any, name: str) -> Any:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{name} must be an exact torch.Tensor")
    tensor = cast("Any", value)
    if (
        tensor.ndim != 4
        or tensor.shape[0] <= 0
        or tensor.shape[-1] != 3
        or min(tensor.shape[1:3]) < 2
        or tensor.layout != torch.strided
    ):
        raise ValueError(f"{name} must be a strided floating [batch,height,width,3] tensor")
    if not tensor.is_floating_point():
        if tensor.dtype not in (torch.uint8, torch.uint16):
            raise ValueError(f"{name} must be a strided floating [batch,height,width,3] tensor")
        tensor = tensor.to(dtype=torch.float32) / float(torch.iinfo(tensor.dtype).max)
    return tensor.contiguous()


def _nearest_32(value: float | int) -> int:
    return max(32, int(round(float(value) / 32.0)) * 32)


def _minimax_h3_resize(image: Any, width: int, height: int, crop: str) -> Any:
    common_upscale = importlib.import_module("dinkster_inference_torch.resize").common_upscale
    nchw = image.permute(0, 3, 1, 2)
    return common_upscale(nchw, width, height, "lanczos", crop).permute(0, 2, 3, 1)


def _minimax_h3_target_canvas(target: Any) -> tuple[int, int]:
    shape = tuple(target.by_role("video").shape)
    if len(shape) != 5 or shape[0] != 1 or shape[1] != 24:
        raise ValueError("MiniMax H3 target video must be [1,24,time,height,width]")
    return shape[4] * 16, shape[3] * 16


def _minimax_h3_reference_image(image: Any, target: Any, mode: str) -> Any:
    if mode not in ("match", "max"):
        raise ValueError("ref_image_size must be 'match' or 'max'")
    target_width, target_height = _minimax_h3_target_canvas(target)
    source_height, source_width = int(image.shape[1]), int(image.shape[2])
    if mode == "match":
        scale = min(
            1.0,
            math.sqrt((target_width * target_height) / (source_width * source_height)),
        )
    else:
        scale = min(1.0, 2048.0 / min(source_width, source_height))
    width = _nearest_32(source_width * scale)
    height = _nearest_32(source_height * scale)
    return _minimax_h3_resize(image, width, height, "disabled")


def _minimax_h3_video_frames(frames: Any, frame_count: int) -> tuple[Any, tuple[int, ...]]:
    source_height, source_width = int(frames.shape[1]), int(frames.shape[2])
    scale = min(
        768.0 / min(source_width, source_height),
        math.sqrt((768 * 1344) / (source_width * source_height)),
    )
    width = _nearest_32(source_width * scale)
    height = _nearest_32(source_height * scale)
    if source_width * source_height < width * height:
        width = _nearest_32(source_width)
        height = _nearest_32(source_height)
    adapted = _minimax_h3_resize(frames, width, height, "disabled")
    count = min(int(adapted.shape[0]), frame_count)
    if count < 5:
        raise ValueError("MiniMax H3 video references require at least 5 frames")
    count -= (count - 5) % 17
    presentation_indices = tuple(range(0, count, 12))
    return adapted[:count], presentation_indices


def _minimax_h3_audio_value(value: object, torch: Any, name: str) -> tuple[Any, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be the standard waveform/sample_rate mapping")
    audio = cast("Mapping[object, object]", value)
    if set(audio) != {"waveform", "sample_rate"}:
        raise TypeError(f"{name} must be the standard waveform/sample_rate mapping")
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if type(waveform) is not torch.Tensor:
        raise TypeError(f"{name}.waveform must be an exact torch.Tensor")
    tensor = cast("Any", waveform)
    if (
        tensor.ndim != 3
        or tensor.shape[0] <= 0
        or tensor.shape[1] != 2
        or tensor.shape[2] <= 0
        or not tensor.is_floating_point()
        or tensor.layout != torch.strided
    ):
        raise ValueError(f"{name}.waveform must be nonempty floating [batch,2,samples]")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError(f"{name}.sample_rate must be a positive integer")
    return tensor, sample_rate


def _minimax_h3_resample_audio(waveform: Any, sample_rate: int) -> Any:
    if sample_rate == 32_000:
        return waveform
    return importlib.import_module("torchaudio.functional").resample(waveform, sample_rate, 32_000)


def _minimax_h3_audio_reference(
    inference: Any,
    waveform: Any,
    sample_rate: int,
    reference_id: str,
) -> tuple[Any, Any]:
    selected = _minimax_h3_resample_audio(waveform[:1], sample_rate)
    descriptor, payload = _minimax_h3_payload(inference, selected, reference_id)
    return inference.MiniMaxH3AudioReference(descriptor, 32_000), payload


def _minimax_h3_video_reference(
    inference: Any,
    descriptors: tuple[Any, ...],
    presentation_indices: tuple[int, ...],
    audio: Any,
) -> Any:
    return inference.MiniMaxH3VideoReference(
        descriptors,
        presentation_indices,
        tuple(float(index / 24.0) for index in presentation_indices),
        audio,
    )


def _canonical_minimax_h3_references(values: Sequence[object]) -> tuple[object, ...]:
    allowed = (
        MiniMaxH3ImageReferenceValue,
        MiniMaxH3VideoReferenceValue,
        MiniMaxH3AudioReferenceValue,
    )
    if any(type(reference) not in allowed for reference in values):
        raise TypeError("references must contain exact MiniMax H3 reference values")
    images = tuple(
        reference for reference in values if type(reference) is MiniMaxH3ImageReferenceValue
    )
    videos = tuple(
        reference for reference in values if type(reference) is MiniMaxH3VideoReferenceValue
    )
    audios = tuple(
        reference for reference in values if type(reference) is MiniMaxH3AudioReferenceValue
    )
    if len(images) > 9 or len(videos) > 3 or len(audios) > 3:
        raise ValueError("REF2VA allows at most 9 images, 3 videos, and 3 audio references")
    return (*images, *videos, *audios)


def _adapt_multistream_latent(
    value: object,
    runtime: object,
    torch: Any,
    inference: Any,
    name: str,
) -> Mapping[object, object]:
    if not isinstance(value, Mapping) or "samples" not in value:
        raise TypeError(f"{name} must be a LATENT mapping containing 'samples'")
    latent = cast("Mapping[object, object]", value)
    samples = latent["samples"]
    if type(samples) is inference.MultiStreamLatent:
        return latent
    if not isinstance(runtime, inference.MultiStreamLatentAdapterRuntime):
        raise TypeError(f"{name} cannot be adapted to the model's latent streams")
    if type(samples) is not torch.Tensor:
        raise TypeError(f"{name} samples must be an exact torch.Tensor")
    adapter = cast("Any", runtime)
    adapted = adapter.adapt_multistream_latent(
        samples,
        source_spatial_downscale=latent.get("downscale_ratio_spacial"),
        source_temporal_downscale=latent.get("downscale_ratio_temporal"),
    )
    if type(adapted) is not inference.MultiStreamLatent:
        raise TypeError("latent adaptation must return an exact MultiStreamLatent")
    result = dict(latent)
    result["samples"] = adapted
    return result


def _adapt_minimax_h3_av(
    value: object,
    runtime: object,
    torch: Any,
    inference: Any,
    name: str,
) -> Any:
    adapted = _adapt_multistream_latent(value, runtime, torch, inference, name)
    return _minimax_h3_av(adapted, torch, inference, name)


def _minimax_h3_frame_count(target: Any) -> int:
    shape = tuple(target.by_role("video").shape)
    if len(shape) != 5 or shape[0] != 1 or shape[1] != 24:
        raise ValueError("MiniMax H3 target video must be [1,24,time,height,width]")
    temporal = shape[2]
    if temporal < 1:
        raise ValueError("MiniMax H3 target video latent must have a positive temporal extent")
    temporal_mapping = importlib.import_module(
        "dinkster_inference"
    ).MINIMAX_H3_VIDEO_TEMPORAL_MAPPING
    return temporal_mapping.content_extent(temporal)


def _minimax_h3_condition(
    cls: type[Node],
    conditioner_handle: NativeComponentHandle,
    codec_handles: tuple[NativeComponentHandle, ...],
    runtime: Any,
    av: Any,
    request: Any,
    payloads: Mapping[str, Any],
) -> Mapping[str, object]:
    torch = _torch()
    inference = importlib.import_module("dinkster_inference")
    frame_count = _minimax_h3_frame_count(av)
    load_av = _move_multistream_latent(av, conditioner_handle.load_device)
    context = current_execution_context()
    cancelled = context.cancelled if context is not None else _not_cancelled
    with native_execution_span(
        "condition", "condition", device=str(conditioner_handle.load_device)
    ) as span:
        parent = None if span is None else span.span_id
        with ExitStack() as stages:
            stages.enter_context(
                conditioner_handle.stage(
                    observer_stage="condition",
                    parent_span_id=parent,
                )
            )
            for handle in codec_handles:
                stages.enter_context(
                    handle.stage(
                        observer_stage="condition",
                        parent_span_id=parent,
                    )
                )
            with torch.inference_mode():
                prepared = runtime.condition(
                    request,
                    target=load_av,
                    frame_count=frame_count,
                    payloads=payloads,
                    cancelled=cancelled,
                )
    value = inference.PreparedMultiStreamConditioning(
        conditioner_handle.resource_identity,
        prepared,
    )
    conditioning: list[list[object]] = [[value, cast("dict[str, object]", {})]]
    target_geometry = tuple((stream.role, tuple(stream.payload.shape)) for stream in av.streams)
    payload_facts = tuple(
        reference_id
        + ":"
        + hashlib.sha256(
            _minimax_h3_tensor_bytes(payload, torch, "MiniMax H3 conditioning payload")
        ).hexdigest()
        for reference_id, payload in sorted(payloads.items())
    )
    facts = (
        "dinkster.minimax-h3.conditioning.v1",
        conditioner_handle.resource_identity,
        repr(request),
        repr(target_geometry),
        str(frame_count),
        *payload_facts,
    )
    fingerprint = (
        "minimax-h3-conditioning:" + hashlib.sha256("\n".join(facts).encode("utf-8")).hexdigest()
    )
    resident = _MiniMaxH3ResidentConditioning(
        conditioning,
        conditioner_handle,
        codec_handles,
        fingerprint,
    )
    return cls.outputs(conditioning=inference.ResidentConditioningCarrier(resident))


class NativeEmptyMiniMaxH3AV(EmptyMiniMaxH3AV):
    @classmethod
    def execute(cls, *, width: int, height: int, frame_count: int) -> Mapping[str, object]:
        inference_torch = importlib.import_module("dinkster_inference_torch")
        value = inference_torch.empty_minimax_h3_av(
            width=width,
            height=height,
            frame_count=frame_count,
            device="cpu",
            dtype=_torch().bfloat16,
        )
        return cls.outputs(latent={"samples": value})


class NativeEmptyMiniMaxMusic3LatentAudio(EmptyMiniMaxMusic3LatentAudio):
    @classmethod
    def execute(cls, *, seconds: float, batch_size: int) -> Mapping[str, object]:
        if type(seconds) not in (int, float) or not 0.04 <= seconds <= 360.0:
            raise ValueError("seconds must be in [0.04, 360.0]")
        if type(batch_size) is not int or not 1 <= batch_size <= 4096:
            raise ValueError("batch_size must be an integer in [1, 4096]")
        inference = importlib.import_module("dinkster_inference")
        torch = _torch()
        audio_frames = min(
            inference.MAX_AUDIO_FRAMES,
            max(1, round(seconds * inference.AUDIO_FRAMES_PER_SECOND)),
        )
        samples = torch.zeros(
            (
                batch_size,
                inference.MINIMAX_MUSIC3_CONFIG.latent_channels,
                inference.minimax_music3_latent_length(audio_frames),
            ),
            device="cpu",
        )
        return cls.outputs(
            latent={"samples": samples, "type": "audio", "downscale_ratio_temporal": 512}
        )


class NativeEmptyLTXAVLatent(EmptyLTXAVLatent):
    @classmethod
    def execute(
        cls,
        *,
        model: object,
        width: int,
        height: int,
        length: int,
        frame_rate: int,
        batch_size: int,
    ) -> Mapping[str, object]:
        handle = _native_handle(model, "model")
        for name, value in (
            ("width", width),
            ("height", height),
            ("length", length),
            ("frame_rate", frame_rate),
            ("batch_size", batch_size),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        inference = importlib.import_module("dinkster_inference")
        runtime = handle.runtime
        geometry = getattr(runtime, "component_sampling_runtime", runtime)
        video_config = getattr(geometry, "video_vae_config", None)
        audio_config = getattr(geometry, "audio_vae_config", None)
        if video_config is None:
            raise TypeError("model requires video_vae_config")
        if audio_config is None:
            raise TypeError("model requires audio_vae_config")
        for config_name, config, fields in (
            (
                "video_vae_config",
                video_config,
                ("latent_channels", "temporal_ratio", "spatial_ratio"),
            ),
            ("audio_vae_config", audio_config, ("z_channels", "latent_frequency_bins")),
        ):
            for field in fields:
                value = getattr(config, field, None)
                if type(value) is not int or value <= 0:
                    raise TypeError(f"model requires {config_name}.{field} as a positive integer")
        rate = getattr(audio_config, "latents_per_second", None)
        if (
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or not math.isfinite(rate)
            or rate <= 0
        ):
            raise TypeError("model requires positive finite audio_vae_config.latents_per_second")
        audio_length = inference.ltx_audio_latents_from_frames(
            audio_config, length, float(frame_rate)
        )
        if width < video_config.spatial_ratio or height < video_config.spatial_ratio:
            raise ValueError(
                "width and height must cover at least one video spatial downscale step"
            )
        if audio_length < 1:
            raise ValueError("length and frame_rate must produce at least one audio latent frame")
        torch = _torch()
        video = torch.zeros(
            (
                batch_size,
                video_config.latent_channels,
                (length - 1) // video_config.temporal_ratio + 1,
                height // video_config.spatial_ratio,
                width // video_config.spatial_ratio,
            ),
            device="cpu",
            dtype=torch.float32,
        )
        audio = torch.zeros(
            (
                batch_size,
                audio_config.z_channels,
                audio_length,
                audio_config.latent_frequency_bins,
            ),
            device="cpu",
            dtype=torch.float32,
        )
        streams = inference.MultiStreamLatent.from_pairs((("video", video), ("audio", audio)))
        return cls.outputs(latent={"samples": streams})


class NativeEmptyLTXVLatent(EmptyLTXVLatent):
    @classmethod
    def execute(
        cls,
        *,
        model: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
    ) -> Mapping[str, object]:
        handle = _native_handle(model, "model")
        for name, value in (
            ("width", width),
            ("height", height),
            ("length", length),
            ("batch_size", batch_size),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        inference = importlib.import_module("dinkster_inference")
        runtime = handle.runtime
        geometry = getattr(runtime, "component_sampling_runtime", runtime)
        video_config = getattr(geometry, "video_vae_config", None)
        if video_config is None:
            raise TypeError("model requires video_vae_config")
        for field in ("latent_channels", "temporal_ratio", "spatial_ratio"):
            value = getattr(video_config, field, None)
            if type(value) is not int or value <= 0:
                raise TypeError(f"model requires video_vae_config.{field} as a positive integer")
        if width < video_config.spatial_ratio or height < video_config.spatial_ratio:
            raise ValueError(
                "width and height must cover at least one video spatial downscale step"
            )
        torch = _torch()
        video = torch.zeros(
            (
                batch_size,
                video_config.latent_channels,
                (length - 1) // video_config.temporal_ratio + 1,
                height // video_config.spatial_ratio,
                width // video_config.spatial_ratio,
            ),
            device="cpu",
            dtype=torch.float32,
        )
        streams = inference.MultiStreamLatent.from_pairs((("video", video),))
        return cls.outputs(latent={"samples": streams})


class NativeMiniMaxH3T2VAConditioning(MiniMaxH3T2VAConditioning):
    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        target: object,
        prompt: str,
    ) -> Mapping[str, object]:
        handle, codecs, runtime = _minimax_h3_conditioner_runtime(clip)
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        target_av = _adapt_minimax_h3_av(target, runtime, torch, inference, "target")
        return _minimax_h3_condition(
            cls,
            handle,
            codecs,
            runtime,
            target_av,
            inference.MiniMaxH3T2VARequest(prompt),
            {},
        )


class NativeMiniMaxH3FL2VAConditioning(MiniMaxH3FL2VAConditioning):
    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        video_vae: object,
        target: object,
        prompt: str,
        first_image: object = None,
        last_image: object = None,
    ) -> Mapping[str, object]:
        handle, codecs, runtime = _minimax_h3_conditioner_runtime(clip, video_vae)
        if first_image is None and last_image is None:
            raise ValueError("FL2VA requires at least one keyframe")
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        target_av = _adapt_minimax_h3_av(target, runtime, torch, inference, "target")
        target_width, target_height = _minimax_h3_target_canvas(target_av)
        keyframes: list[Any] = []
        payloads: dict[str, Any] = {}
        for ordinal, (role_name, role, value, crop) in enumerate(
            (
                (
                    "first",
                    inference.MiniMaxH3KeyframeRole.FIRST,
                    first_image,
                    "disabled",
                ),
                (
                    "last",
                    inference.MiniMaxH3KeyframeRole.LAST,
                    last_image,
                    "center",
                ),
            ),
            1,
        ):
            if value is None:
                continue
            tensor = _minimax_h3_image_tensor(value, torch, f"{role_name}_image")
            tensor = _minimax_h3_resize(tensor, target_width, target_height, crop)
            reference_id = f"fl2va:{ordinal}:keyframe:{role_name}"
            descriptor, payload = _minimax_h3_payload(inference, tensor, reference_id)
            payloads[reference_id] = payload
            keyframes.append(inference.MiniMaxH3Keyframe(role, descriptor))
        request = inference.MiniMaxH3FL2VARequest(prompt, tuple(keyframes))
        return _minimax_h3_condition(
            cls,
            handle,
            codecs,
            runtime,
            target_av,
            request,
            payloads,
        )


class NativeMiniMaxH3REF2VAConditioning(MiniMaxH3REF2VAConditioning):
    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        video_vae: object,
        audio_vae: object,
        target: object,
        prompt: str,
        references: object,
        ref_image_size: str,
    ) -> Mapping[str, object]:
        if isinstance(references, str | bytes) or not isinstance(references, Sequence):
            raise TypeError("references must be a sequence")
        values = tuple(cast("Sequence[object]", references))
        if not values:
            raise ValueError("REF2VA requires at least one reference")
        if ref_image_size not in ("match", "max"):
            raise ValueError("ref_image_size must be 'match' or 'max'")
        canonical = _canonical_minimax_h3_references(values)
        handle, codecs, runtime = _minimax_h3_conditioner_runtime(
            clip,
            video_vae,
            audio_vae,
        )
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        target_av = _adapt_minimax_h3_av(target, runtime, torch, inference, "target")
        frame_count = _minimax_h3_frame_count(target_av)
        typed: list[Any] = []
        payloads: dict[str, Any] = {}
        for ordinal, reference in enumerate(canonical, 1):
            if type(reference) is MiniMaxH3ImageReferenceValue:
                image = _minimax_h3_image_tensor(reference.image, torch, "image reference")
                image = _minimax_h3_reference_image(image, target_av, ref_image_size)
                reference_id = f"ref2va:{ordinal}:image:1"
                descriptor, payload = _minimax_h3_payload(inference, image, reference_id)
                payloads[reference_id] = payload
                typed.append(inference.MiniMaxH3ImageReference(descriptor))
            elif type(reference) is MiniMaxH3AudioReferenceValue:
                waveform, sample_rate = _minimax_h3_audio_value(
                    {"waveform": reference.waveform, "sample_rate": reference.sample_rate},
                    torch,
                    "audio reference",
                )
                reference_id = f"ref2va:{ordinal}:audio:1"
                audio_reference, payload = _minimax_h3_audio_reference(
                    inference, waveform, sample_rate, reference_id
                )
                payloads[reference_id] = payload
                typed.append(audio_reference)
            elif type(reference) is MiniMaxH3VideoReferenceValue:
                if type(reference.frames) is not torch.Tensor:
                    raise TypeError("video reference frames must be an exact torch.Tensor")
                frames = cast("Any", reference.frames)
                if (
                    frames.ndim != 4
                    or frames.shape[0] <= 0
                    or frames.shape[-1] != 3
                    or min(frames.shape[1:3]) < 2
                    or not frames.is_floating_point()
                    or frames.layout != torch.strided
                ):
                    raise ValueError(
                        "video reference frames must be strided floating [time,height,width,3]"
                    )
                frames, presentation_indices = _minimax_h3_video_frames(frames, frame_count)
                descriptors: list[Any] = []
                for frame_ordinal, frame in enumerate(frames.split(1), 1):
                    reference_id = f"ref2va:{ordinal}:video:{frame_ordinal}"
                    descriptor, payload = _minimax_h3_payload(inference, frame, reference_id)
                    payloads[reference_id] = payload
                    descriptors.append(descriptor)
                audio_reference = None
                if reference.audio is not None:
                    waveform, sample_rate = _minimax_h3_audio_value(
                        {
                            "waveform": reference.audio.waveform,
                            "sample_rate": reference.audio.sample_rate,
                        },
                        torch,
                        "video audio reference",
                    )
                    reference_id = f"ref2va:{ordinal}:video-audio:1"
                    audio_reference, payload = _minimax_h3_audio_reference(
                        inference, waveform, sample_rate, reference_id
                    )
                    payloads[reference_id] = payload
                typed.append(
                    _minimax_h3_video_reference(
                        inference,
                        tuple(descriptors),
                        presentation_indices,
                        audio_reference,
                    )
                )
        request = inference.MiniMaxH3REF2VARequest(prompt, tuple(typed))
        return _minimax_h3_condition(
            cls,
            handle,
            codecs,
            runtime,
            target_av,
            request,
            payloads,
        )


def _empty_minimax_h3_target(width: int, height: int, length: int) -> object:
    result = NativeEmptyMiniMaxH3AV.execute(
        width=width,
        height=height,
        frame_count=length,
    )
    return result["latent"]


class NativeMiniMaxH3ImageToVideo(MiniMaxH3ImageToVideo):
    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        vae: object,
        prompt: str,
        width: int,
        height: int,
        length: int,
        first_frame: object = None,
        last_frame: object = None,
    ) -> Mapping[str, object]:
        latent = _empty_minimax_h3_target(width, height, length)
        if first_frame is None and last_frame is None:
            result = NativeMiniMaxH3T2VAConditioning.execute(
                clip=clip,
                target=latent,
                prompt=prompt,
            )
        else:
            result = NativeMiniMaxH3FL2VAConditioning.execute(
                clip=clip,
                video_vae=vae,
                target=latent,
                prompt=prompt,
                first_image=first_frame,
                last_image=last_frame,
            )
        return cls.outputs(positive=result["conditioning"], latent=latent)


def _audio_reference(value: object, name: str) -> MiniMaxH3AudioReferenceValue:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be the standard waveform/sample_rate mapping")
    audio = cast("Mapping[object, object]", value)
    return MiniMaxH3AudioReferenceValue(
        cast("Any", audio.get("waveform")),
        cast("Any", audio.get("sample_rate")),
    )


class NativeMiniMaxH3ReferenceToVideo(MiniMaxH3ReferenceToVideo):
    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        prompt: str,
        width: int,
        height: int,
        length: int,
        ref_image_size: str,
        vae: object = None,
        audio_vae: object = None,
        ref_images: Mapping[str, object],
        ref_videos: Mapping[str, object],
        ref_video_audios: Mapping[str, object],
        ref_audios: Mapping[str, object],
    ) -> Mapping[str, object]:
        references: list[object] = [
            MiniMaxH3ImageReferenceValue(image) for image in ref_images.values()
        ]
        for name, frames in ref_videos.items():
            suffix = name.rsplit("_", 1)[-1]
            soundtrack = ref_video_audios.get(f"ref_video_audio_{suffix}")
            if soundtrack is None:
                soundtrack = ref_video_audios.get(suffix)
            references.append(
                MiniMaxH3VideoReferenceValue(
                    cast("Any", frames),
                    None
                    if soundtrack is None
                    else _audio_reference(soundtrack, f"ref_video_audio_{suffix}"),
                )
            )
        references.extend(_audio_reference(audio, name) for name, audio in ref_audios.items())
        latent = _empty_minimax_h3_target(width, height, length)
        result = NativeMiniMaxH3REF2VAConditioning.execute(
            clip=clip,
            video_vae=vae,
            audio_vae=audio_vae,
            target=latent,
            prompt=prompt,
            references=references,
            ref_image_size=ref_image_size,
        )
        return cls.outputs(positive=result["conditioning"], latent=latent)


class NativeMiniMaxH3AddGuide(MiniMaxH3AddGuide):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        latent: object,
        frame_idx: int,
        vae: object = None,
        audio_vae: object = None,
        image: object = None,
        audio: object = None,
    ) -> Mapping[str, object]:
        if image is None and audio is None:
            raise ValueError("MiniMax H3 guides require an image or audio")
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        carrier = _minimax_h3_conditioning_carrier(positive, inference, "positive")
        if carrier is None:
            raise ValueError("positive conditioning must not be empty")
        target = _minimax_h3_av(latent, torch, inference, "latent")
        target_video = target.by_role("video")
        target_audio = target.by_role("audio")
        target_frames = _minimax_h3_frame_count(target)

        frames = None
        guide_frames = 1
        if image is not None:
            if vae is None:
                raise ValueError("anchoring guide frames requires the vae input")
            frames = _minimax_h3_image_batch(image, torch, "image")
            guide_frames = int(frames.shape[0])
            if guide_frames < 5:
                guide_frames = 1
            else:
                guide_frames -= (guide_frames - 5) % 17
            frames = frames[:guide_frames]
        resolved = inference.resolve_timeline_frame_index(frame_idx, target_frames)
        if resolved + guide_frames > target_frames:
            raise ValueError(
                f"a {guide_frames} frame guide at frame_idx {frame_idx} does not fit "
                f"the target's {target_frames} frames"
            )

        streams: list[tuple[str, Any]] = []
        if frames is not None:
            video_handle, video_runtime = _minimax_h3_video_vae_runtime(vae, "vae")
            target_width, target_height = _minimax_h3_target_canvas(target)
            frames = _minimax_h3_resize(frames, target_width, target_height, "center")
            with native_execution_span(
                "condition", "encode_guide_video", device=str(video_handle.load_device)
            ) as span:
                parent = None if span is None else span.span_id
                with video_handle.stage(observer_stage="condition", parent_span_id=parent):
                    with torch.inference_mode():
                        video = video_runtime.encode_video(
                            frames.permute(3, 0, 1, 2).unsqueeze(0).to(video_handle.load_device)
                        ).to(target_video)
            streams.append(("video", video))

        if audio is not None:
            if audio_vae is None:
                raise ValueError("anchoring guide audio requires the audio_vae input")
            audio_handle, audio_runtime = _minimax_h3_audio_vae_runtime(audio_vae)
            waveform, sample_rate = _minimax_h3_audio_value(audio, torch, "audio")
            waveform = _minimax_h3_resample_audio(waveform[:1], sample_rate)
            with native_execution_span(
                "condition", "encode_guide_audio", device=str(audio_handle.load_device)
            ) as span:
                parent = None if span is None else span.span_id
                with audio_handle.stage(observer_stage="condition", parent_span_id=parent):
                    with torch.inference_mode():
                        audio_latent = audio_runtime.encode_audio(
                            inference.MiniMaxH3AudioContent(
                                waveform.to(audio_handle.load_device),
                                32_000,
                            )
                        ).to(target_audio)
            max_audio = math.floor(
                target_audio.shape[-1]
                - inference.MINIMAX_H3_VIDEO_TEMPORAL_MAPPING.timeline_position(resolved)
            )
            if max_audio < 1:
                raise ValueError(f"frame_idx {frame_idx} is past the end of the target audio")
            if audio_latent.shape[-1] > max_audio:
                audio_latent = audio_latent[..., :max_audio].clone()
            streams.append(("audio", audio_latent))

        guide = inference.TimelineGuide(
            resolved,
            guide_frames,
            inference.MultiStreamLatent.from_pairs(streams),
        )
        prepared = inference_torch.add_minimax_h3_timeline_guide(
            carrier.payload,
            target,
            guide,
        )
        output = inference.PreparedMultiStreamConditioning(carrier.runtime_identity, prepared)
        conditioned: list[list[object]] = [[output, cast("dict[str, object]", {})]]
        fingerprint = _minimax_h3_latent_fingerprint(guide.latent, torch)
        return cls.outputs(
            positive=_minimax_h3_rewrap_conditioning(
                positive,
                conditioned,
                inference,
                f"guide:{resolved}:{guide_frames}:{fingerprint}",
            )
        )


class NativeMiniMaxH3MotionContext(MiniMaxH3MotionContext):
    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        latent: object,
        previous_latent: object,
        context_length: int,
    ) -> Mapping[str, object]:
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        carrier = _minimax_h3_conditioning_carrier(positive, inference, "positive")
        if carrier is None:
            raise ValueError("positive conditioning must not be empty")
        target = _minimax_h3_av(latent, torch, inference, "latent")
        previous = _minimax_h3_av(previous_latent, torch, inference, "previous_latent")
        prepared, trim_time = inference_torch.add_minimax_h3_motion_context(
            carrier.payload,
            target,
            previous,
            context_length,
        )
        output = inference.PreparedMultiStreamConditioning(carrier.runtime_identity, prepared)
        conditioned: list[list[object]] = [[output, cast("dict[str, object]", {})]]
        fingerprint = _minimax_h3_latent_fingerprint(previous, torch)
        return cls.outputs(
            positive=_minimax_h3_rewrap_conditioning(
                positive,
                conditioned,
                inference,
                f"motion-context:{context_length}:{fingerprint}",
            ),
            trim_time=trim_time,
        )


class NativeMiniMaxH3AVEncode(MiniMaxH3AVEncode):
    @classmethod
    def execute(
        cls,
        *,
        video_vae: object,
        audio_vae: object,
        frames: object,
        audio: object,
    ) -> Mapping[str, object]:
        video_handle, video_runtime = _minimax_h3_video_vae_runtime(video_vae)
        audio_handle, audio_runtime = _minimax_h3_audio_vae_runtime(audio_vae)
        torch = _torch()
        if type(frames) is not torch.Tensor:
            raise TypeError("frames must be an exact torch.Tensor")
        images = cast("Any", frames)
        if (
            images.ndim != 4
            or images.shape[0] <= 0
            or images.shape[-1] != 3
            or min(images.shape[1:3]) < 2
            or not images.is_floating_point()
            or images.layout != torch.strided
        ):
            raise ValueError("frames must be nonempty strided floating [time,height,width,3]")
        waveform, sample_rate = _minimax_h3_audio_value(audio, torch, "audio")
        if waveform.shape[0] != 1:
            raise ValueError("audio.waveform must have exact batch size one")
        waveform = _minimax_h3_resample_audio(waveform, sample_rate)
        inference = importlib.import_module("dinkster_inference")
        with native_execution_span(
            "encode", "encode_video", device=str(video_handle.load_device)
        ) as span:
            parent = None if span is None else span.span_id
            with video_handle.stage(observer_stage="encode", parent_span_id=parent):
                with torch.inference_mode():
                    video = video_runtime.encode_video(
                        images.permute(3, 0, 1, 2).unsqueeze(0).to(video_handle.load_device)
                    )
        with native_execution_span(
            "encode", "encode_audio", device=str(audio_handle.load_device)
        ) as span:
            parent = None if span is None else span.span_id
            with audio_handle.stage(observer_stage="encode", parent_span_id=parent):
                with torch.inference_mode():
                    audio_latent = audio_runtime.encode_audio(
                        inference.MiniMaxH3AudioContent(
                            waveform.to(audio_handle.load_device),
                            32_000,
                        )
                    )
        streams = inference.MultiStreamLatent.from_pairs(
            (("video", video), ("audio", audio_latent))
        )
        return cls.outputs(latent={"samples": streams})


class NativeMiniMaxH3AVDecode(MiniMaxH3AVDecode):
    @classmethod
    def execute(
        cls,
        *,
        video_vae: object,
        audio_vae: object,
        latent: object,
    ) -> Mapping[str, object]:
        video_handle, video_runtime = _minimax_h3_video_vae_runtime(video_vae)
        audio_handle, audio_runtime = _minimax_h3_audio_vae_runtime(audio_vae)
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        value = _minimax_h3_av(latent, torch, inference, "latent")
        video_latent = value.by_role("video")
        audio_latent = value.by_role("audio")
        if (
            video_latent.ndim != 5
            or video_latent.shape[0] != 1
            or video_latent.shape[1] != 24
            or audio_latent.ndim != 4
            or tuple(audio_latent.shape[:3]) != (1, 32, 2)
        ):
            raise ValueError("av must contain exact batch-one MiniMax H3 latent streams")
        with native_execution_span(
            "decode", "decode_video", device=str(video_handle.load_device)
        ) as span:
            parent = None if span is None else span.span_id
            with video_handle.stage(observer_stage="decode", parent_span_id=parent):
                with torch.inference_mode():
                    video = video_runtime.decode_video(video_latent.to(video_handle.load_device))
        with native_execution_span(
            "decode", "decode_audio", device=str(audio_handle.load_device)
        ) as span:
            parent = None if span is None else span.span_id
            with audio_handle.stage(observer_stage="decode", parent_span_id=parent):
                with torch.inference_mode():
                    audio = audio_runtime.decode_audio(audio_latent.to(audio_handle.load_device))
        if (
            type(video) is not torch.Tensor
            or video.ndim != 5
            or tuple(video.shape[:2]) != (1, 3)
            or not video.is_floating_point()
            or video.layout != torch.strided
        ):
            raise ValueError("MiniMax H3 video decode must return [1,3,time,height,width]")
        if type(audio) is not inference.MiniMaxH3AudioContent:
            raise TypeError("MiniMax H3 audio decode must return MiniMaxH3AudioContent")
        waveform = audio.waveform
        if (
            type(waveform) is not torch.Tensor
            or waveform.ndim != 3
            or tuple(waveform.shape[:2]) != (1, 2)
            or not waveform.is_floating_point()
            or waveform.layout != torch.strided
        ):
            raise ValueError("MiniMax H3 audio decode must return [1,2,samples]")
        if type(audio.sample_rate) is not int or audio.sample_rate <= 0:
            raise ValueError("MiniMax H3 audio decode must return a positive integer sample rate")
        return cls.outputs(
            frames=video[0].permute(1, 2, 3, 0).contiguous(),
            audio={"waveform": waveform, "sample_rate": audio.sample_rate},
        )


def _av_stream(value: object, torch: Any, inference: Any, role: str, name: str) -> Any:
    samples, streams = _latent_samples(value, torch, inference, name)
    payload = samples if streams is None else streams.by_role(role)
    if type(payload) is not torch.Tensor or not payload.is_floating_point():
        raise TypeError(f"{name} {role} stream must be an exact floating torch.Tensor")
    return payload


def _fit_audio_stream(
    value: Any, target: Any, torch: Any, name: str, *, pad_value: float = 0.0
) -> Any:
    if value.ndim != target.ndim or tuple(value.shape[:-1]) != tuple(target.shape[:-1]):
        raise ValueError(
            f"{name} must match the existing audio rank, batch, and channel dimensions"
        )
    target_length = target.shape[-1]
    if value.shape[-1] > target_length:
        return value[..., :target_length]
    if value.shape[-1] < target_length:
        return torch.nn.functional.pad(value, (0, target_length - value.shape[-1]), value=pad_value)
    return value


def _role_mask(
    value: object, streams: Any, role: str, payload: Any, torch: Any, inference: Any
) -> Any:
    mask = cast("Mapping[object, object]", value).get("noise_mask")
    if mask is None:
        return None
    if type(mask) is torch.Tensor:
        if streams is None or streams.roles[0] == role:
            return mask
        return torch.ones_like(payload)
    if type(mask) is inference.MultiStreamLatent:
        structural_mask = cast("Any", mask)
        if role in structural_mask.roles:
            return structural_mask.by_role(role)
        return torch.ones_like(payload)
    raise TypeError("noise_mask must be a tensor or MultiStreamLatent")


def _latent_mask_target(
    value: object,
    mapping: Any,
    torch: Any,
    inference: Any,
    name: str,
) -> tuple[dict[object, object], Any, Any, bool]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a LATENT mapping")
    metadata = dict(cast("Mapping[object, object]", value))
    samples = cast("Any", metadata.get("samples"))
    structural = type(samples) is inference.MultiStreamLatent
    streams: Any = (
        samples
        if structural
        else inference.MultiStreamLatent.from_pairs(((mapping.role, samples),))
    )
    if type(streams) is not inference.MultiStreamLatent or mapping.role not in streams.roles:
        raise ValueError(f"{name} does not contain the codec's {mapping.role!r} role")
    target = streams.by_role(mapping.role)
    if (
        type(target) is not torch.Tensor
        or not target.is_floating_point()
        or target.layout != torch.strided
    ):
        raise TypeError(f"{name} {mapping.role} role must be an exact strided floating tensor")
    return metadata, streams, target, structural


def _mask_has_role(mask: object, streams: Any, role: str, torch: Any, inference: Any) -> bool:
    if type(mask) is torch.Tensor:
        return streams.roles[0] == role
    if type(mask) is inference.MultiStreamLatent:
        return role in cast("Any", mask).roles
    if mask is None:
        return False
    raise TypeError("noise_mask must be a tensor or MultiStreamLatent")


def _set_latent_role_mask(
    metadata: dict[object, object],
    streams: Any,
    role: str,
    role_mask: Any,
    operation: str,
    structural: bool,
    torch: Any,
    inference: Any,
    inference_torch: Any,
) -> Mapping[object, object]:
    if operation not in ("replace", "max", "min", "multiply"):
        raise ValueError("mask operation must be replace, max, min, or multiply")
    new_masks = inference_torch.normalize_latent_mask(
        inference.MultiStreamLatent.from_pairs(((role, role_mask),)),
        streams,
    )
    existing = metadata.get("noise_mask")
    has_existing_role = _mask_has_role(existing, streams, role, torch, inference)
    normalized = (
        new_masks if existing is None else inference_torch.normalize_latent_mask(existing, streams)
    )
    selected = new_masks.by_role(role)
    if operation != "replace" and has_existing_role:
        current = normalized.by_role(role)
        if operation == "max":
            selected = torch.maximum(current, selected)
        elif operation == "min":
            selected = torch.minimum(current, selected)
        else:
            selected = current * selected
    masks = inference.MultiStreamLatent.from_pairs(
        (stream_role, selected if stream_role == role else normalized.by_role(stream_role))
        for stream_role in streams.roles
    )
    metadata["noise_mask"] = masks if structural else masks.by_role(role)
    return metadata


class NativeSetLatentMaskFromFrames(SetLatentMaskFromFrames):
    @classmethod
    def execute(
        cls,
        *,
        latent: object,
        vae: object,
        mask: object,
        spatial_reduction: str,
        temporal_reduction: str,
        operation: str,
    ) -> Mapping[str, object]:
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        runtime = _latent_mask_codec_runtime(vae, "vae")
        mapping = runtime.latent_mask_mapping
        metadata, streams, target, structural = _latent_mask_target(
            latent, mapping, torch, inference, "latent"
        )
        role_mask = inference_torch.content_mask_to_latent_mask(
            mask,
            target,
            mapping,
            spatial_reduction=spatial_reduction,
            temporal_reduction=temporal_reduction,
        )
        output = _set_latent_role_mask(
            metadata,
            streams,
            mapping.role,
            role_mask,
            operation,
            structural,
            torch,
            inference,
            inference_torch,
        )
        return cls.outputs(latent=output)


class NativeSetLatentMaskFromTimeRanges(SetLatentMaskFromTimeRanges):
    @classmethod
    def execute(
        cls,
        *,
        latent: object,
        vae: object,
        ranges: str,
        selected: float,
        unselected: float,
        operation: str,
    ) -> Mapping[str, object]:
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        runtime = _latent_mask_codec_runtime(vae, "vae")
        mapping = runtime.latent_mask_mapping
        metadata, streams, target, structural = _latent_mask_target(
            latent, mapping, torch, inference, "latent"
        )
        parsed = inference.parse_time_ranges(
            ranges,
            mapping.duration_seconds(target.shape[-1]),
        )
        role_mask = inference_torch.time_ranges_to_latent_mask(
            target,
            mapping,
            parsed,
            selected=selected,
            unselected=unselected,
        )
        output = _set_latent_role_mask(
            metadata,
            streams,
            mapping.role,
            role_mask,
            operation,
            structural,
            torch,
            inference,
            inference_torch,
        )
        return cls.outputs(latent=output)


def _mask_report(
    role_mask: Any,
    target: Any,
    mapping: Any,
    source: str,
) -> str:
    time_axis = 2 if mapping.spatial_downscale is not None else target.ndim - 1
    dimensions = tuple(index for index in range(role_mask.ndim) if index != time_axis)
    maximum = role_mask.amax(dim=dimensions)
    minimum = role_mask.amin(dim=dimensions)
    values = tuple(float(value) for value in maximum)
    varied = any(float(low) != high for low, high in zip(minimum, values, strict=True))
    groups = mapping.content_ranges(len(values))
    runs: list[tuple[int, int, float]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            runs.append((start, index, values[start]))
            start = index

    def value_text(value: float) -> str:
        text = f"{value:.6g}"
        label = "keep" if value == 0.0 else "generate" if value == 1.0 else "soft"
        return f"{text} ({label})"

    def time_text(value: float) -> str:
        return f"{value:.4f}".rstrip("0").rstrip(".")

    intervals = ", ".join(
        f"{time_text(groups[start][0] / mapping.content_rate_hz)}-"
        f"{time_text(groups[stop - 1][1] / mapping.content_rate_hz)}s = {value_text(value)}"
        for start, stop, value in runs
    )
    report = (
        f"{source}\n"
        f"{mapping.role}: {len(values)} latent frames, "
        f"{time_text(mapping.duration_seconds(len(values)))}s at "
        f"{mapping.content_rate_hz:g} content frames/s\n"
        f"{intervals}"
    )
    if varied:
        report += "\nwarning: values vary within timeline frames; reporting each frame's maximum"
    return report


class NativeInspectLatentMask(InspectLatentMask):
    @classmethod
    def execute(cls, *, latent: object, vae: object) -> Mapping[str, object]:
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        runtime = _latent_mask_codec_runtime(vae, "vae")
        mapping = runtime.latent_mask_mapping
        metadata, streams, target, _structural = _latent_mask_target(
            latent, mapping, torch, inference, "latent"
        )
        raw_mask = metadata.get("noise_mask")
        has_role = _mask_has_role(raw_mask, streams, mapping.role, torch, inference)
        if raw_mask is None:
            source = f"no noise mask on the latent; {mapping.role} defaults to generate"
            role_mask = torch.ones_like(target, dtype=torch.float32)
        elif not has_role:
            source = f"noise mask has no {mapping.role} role; it defaults to generate"
            role_mask = torch.ones_like(target, dtype=torch.float32)
        else:
            source = f"{mapping.role} role mask"
            role_mask = inference_torch.normalize_latent_mask(raw_mask, streams).by_role(
                mapping.role
            )
        if not bool(torch.isfinite(role_mask).all()):
            raise ValueError("latent mask values must be finite")
        if float(role_mask.amin()) < 0.0 or float(role_mask.amax()) > 1.0:
            raise ValueError("latent mask values must be within [0, 1]")
        preview = (
            inference_torch.latent_mask_to_content_mask(role_mask, target, mapping)
            if mapping.spatial_downscale is not None
            else inference_torch.latent_mask_preview(role_mask, target)
        )
        return cls.outputs(
            mask=preview,
            report=_mask_report(role_mask, target, mapping, source),
        )


class NativeConcatAVLatent(ConcatAVLatent):
    @classmethod
    def execute(cls, *, video_latent: object, audio_latent: object) -> Mapping[str, object]:
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        video_samples, video_streams = _latent_samples(
            video_latent, torch, inference, "video_latent"
        )
        if video_streams is not None and video_streams.roles not in (
            ("video",),
            ("video", "audio"),
        ):
            raise ValueError("video_latent must be single video or exact video/audio streams")
        video = video_samples if video_streams is None else video_streams.by_role("video")
        audio_samples, audio_streams = _latent_samples(
            audio_latent, torch, inference, "audio_latent"
        )
        if audio_streams is not None and audio_streams.roles != ("audio",):
            raise ValueError("audio_latent must be a single audio stream")
        audio = audio_samples if audio_streams is None else audio_streams.by_role("audio")
        if type(video) is not torch.Tensor or not video.is_floating_point():
            raise TypeError("video_latent video stream must be an exact floating torch.Tensor")
        if type(audio) is not torch.Tensor or not audio.is_floating_point():
            raise TypeError("audio_latent audio stream must be an exact floating torch.Tensor")
        if video.shape[0] != audio.shape[0]:
            raise ValueError("video and audio streams must have the same batch size")
        if video_streams is not None and "audio" in video_streams.roles:
            audio = _fit_audio_stream(audio, video_streams.by_role("audio"), torch, "audio_latent")
        video_mask = _role_mask(video_latent, video_streams, "video", video, torch, inference)
        source_audio = audio_samples if audio_streams is None else audio_streams.by_role("audio")
        audio_mask = _role_mask(
            audio_latent, audio_streams, "audio", source_audio, torch, inference
        )
        inference_torch = importlib.import_module("dinkster_inference_torch")
        if video_mask is not None:
            video_mask = inference_torch.normalize_latent_mask(
                video_mask,
                inference.MultiStreamLatent.from_pairs((("video", video),)),
            ).by_role("video")
        if audio_mask is not None:
            audio_mask = inference_torch.normalize_latent_mask(
                audio_mask,
                inference.MultiStreamLatent.from_pairs((("audio", source_audio),)),
            ).by_role("audio")
            if video_streams is not None and "audio" in video_streams.roles:
                audio_mask = _fit_audio_stream(
                    audio_mask,
                    video_streams.by_role("audio"),
                    torch,
                    "audio noise_mask",
                    pad_value=1.0,
                )
        output = dict(cast("Mapping[object, object]", video_latent))
        output.update(cast("Mapping[object, object]", audio_latent))
        output["samples"] = inference.MultiStreamLatent.from_pairs(
            (("video", video), ("audio", audio))
        )
        if video_mask is not None or audio_mask is not None:
            output["noise_mask"] = inference.MultiStreamLatent.from_pairs(
                (
                    ("video", torch.ones_like(video) if video_mask is None else video_mask),
                    ("audio", torch.ones_like(audio) if audio_mask is None else audio_mask),
                )
            )
        else:
            output.pop("noise_mask", None)
        return cls.outputs(latent=output)


class NativeSeparateAVLatent(SeparateAVLatent):
    @classmethod
    def execute(cls, *, latent: object) -> Mapping[str, object]:
        torch = _torch()
        inference = importlib.import_module("dinkster_inference")
        _, streams = _latent_samples(latent, torch, inference, "latent")
        if streams is None or streams.roles != ("video", "audio"):
            raise TypeError("latent must contain exact ordered video/audio streams")
        metadata = dict(cast("Mapping[object, object]", latent))
        video = dict(metadata)
        audio = dict(metadata)
        video["samples"] = streams.by_role("video")
        audio["samples"] = streams.by_role("audio")
        if metadata.get("noise_mask") is not None:
            masks = importlib.import_module("dinkster_inference_torch").normalize_latent_mask(
                metadata["noise_mask"], streams
            )
            video["noise_mask"] = masks.by_role("video")
            audio["noise_mask"] = masks.by_role("audio")
        return cls.outputs(video_latent=video, audio_latent=audio)


def _preview_stream(
    model: object, latent: object, role: str
) -> tuple[NativeRuntimeHandle, Any, Any, Any]:
    handle = _native_handle(model, "model")
    torch = _torch()
    inference = importlib.import_module("dinkster_inference")
    if not isinstance(handle.runtime, inference.MultiStreamPreviewRuntime):
        raise TypeError("model does not provide multi-stream preview codecs")
    payload = _av_stream(latent, torch, inference, role, "latent")
    return handle, torch, inference, payload


class NativePreviewLatentVisual(PreviewLatentVisual):
    @classmethod
    def execute(cls, *, model: object, latent: object, role: str) -> Mapping[str, object]:
        handle, torch, _, payload = _preview_stream(model, latent, role)
        with native_execution_span(
            "decode", "decode_video", device=str(handle.load_device)
        ) as span:
            parent = None if span is None else span.span_id
            with handle.stage("vae", observer_stage="decode", parent_span_id=parent):
                with torch.inference_mode():
                    visual = handle.runtime.preview_visual(role, payload.to(handle.load_device))
        if type(visual) is not torch.Tensor or not visual.is_floating_point():
            raise TypeError("visual stream preview must return a floating torch.Tensor")
        if visual.ndim == 5 and visual.shape[1] == 3:
            image = visual.permute(0, 2, 3, 4, 1).flatten(0, 1)
        elif visual.ndim == 4 and visual.shape[1] == 3:
            image = visual.permute(0, 2, 3, 1)
        else:
            raise ValueError("visual stream preview must return [B,3,H,W] or [B,3,T,H,W]")
        return cls.outputs(image=image)


class NativePreviewLatentAudio(PreviewLatentAudio):
    @classmethod
    def execute(cls, *, model: object, latent: object, role: str) -> Mapping[str, object]:
        handle, torch, inference, payload = _preview_stream(model, latent, role)
        with native_execution_span(
            "decode", "decode_audio", device=str(handle.load_device)
        ) as span:
            parent = None if span is None else span.span_id
            with handle.stage("vae", observer_stage="decode", parent_span_id=parent):
                with torch.inference_mode():
                    audio = handle.runtime.preview_audio(role, payload.to(handle.load_device))
        if type(audio) is not inference.AudioPreview:
            raise TypeError("audio stream preview must return AudioPreview")
        return cls.outputs(audio={"waveform": audio.waveform, "sample_rate": audio.sample_rate})


def _minimax_h3_conditioning_carrier(value: object, inference: Any, name: str) -> Any | None:
    prepared = _prepared_multistream_carrier(value, inference, name)
    if prepared is None:
        return None
    try:
        inference.ComponentBinding(
            "conditioner",
            inference.MINIMAX_H3_CONFIG.family_id,
            prepared.runtime_identity,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} conditioning was not prepared by an official MiniMax H3 conditioner component"
        ) from error
    return prepared


def _minimax_h3_latent_fingerprint(value: object, torch: Any) -> str:
    digest = hashlib.sha256()
    for stream in cast("Any", value).streams:
        payload = stream.payload
        digest.update(stream.role.encode("utf-8"))
        digest.update(repr((tuple(payload.shape), payload.dtype)).encode("utf-8"))
        digest.update(_minimax_h3_tensor_bytes(payload, torch, "MiniMax H3 guide latent"))
    return digest.hexdigest()


def _minimax_h3_rewrap_conditioning(
    source: object,
    conditioning: list[list[object]],
    inference: Any,
    operation: str,
) -> object:
    if not isinstance(source, inference.ResidentConditioningCarrier):
        raise TypeError("MiniMax H3 conditioning transform requires a resident carrier")
    resident = cast("Any", source).payload
    if type(resident) is not _MiniMaxH3ResidentConditioning:
        raise TypeError("MiniMax H3 conditioning has an invalid resident payload")
    facts = (resident.fingerprint, operation)
    fingerprint = (
        "minimax-h3-conditioning:" + hashlib.sha256("\n".join(facts).encode("utf-8")).hexdigest()
    )
    return inference.ResidentConditioningCarrier(
        _MiniMaxH3ResidentConditioning(
            conditioning,
            resident.owner,
            resident.references,
            fingerprint,
        )
    )


def _minimax_h3_model_handle(
    value: object,
    inference: Any,
) -> NativeRuntimeHandle | None:
    if isinstance(value, _NativeModelOverlay):
        value = value.handle
    if not isinstance(value, NativeRuntimeHandle):
        return None
    model = cast("object", value.runtime)
    recipe = value.recipe
    model_type = type(model)
    exact_type_name = (
        model_type.__module__ == "dinkster_inference_torch.minimax_h3_assembly"
        and model_type.__name__ == "MiniMaxH3Model"
    )
    if recipe.family_id != inference.MINIMAX_H3_CONFIG.family_id and not exact_type_name:
        return None
    inference_torch = importlib.import_module("dinkster_inference_torch")
    if type(model) is not inference_torch.MiniMaxH3Model:
        return None
    if (
        recipe.family_id != inference.MINIMAX_H3_CONFIG.family_id
        or tuple(binding.role for binding in recipe.sources) != ("diffusion",)
        or recipe.runtime_identity != cast("Any", model).runtime_identity
    ):
        raise TypeError("MiniMax H3 sampling requires a standalone DiT component model")
    return value


def _minimax_h3_dit_runtime(
    handle: NativeRuntimeHandle,
    conditioner_identity: str,
    inference: Any,
    torch: Any,
) -> Any:
    inference_torch = importlib.import_module("dinkster_inference_torch")
    validated = _minimax_h3_model_handle(handle, inference)
    if validated is None:
        raise TypeError("MiniMax H3 sampling requires a standalone DiT component model")
    model = validated.runtime
    recipe = validated.recipe
    model_role = model.model_role.replace("-", "_")
    composition = inference.compose_execution(
        inference.MINIMAX_H3_CONFIG.family_id,
        {
            model_role: recipe.runtime_identity,
            "conditioner": conditioner_identity,
        },
    )
    return inference_torch.MiniMaxH3DiTRuntime(
        model.assembled.diffusion,
        model_role=model_role,
        runtime_identity=composition.execution_identity,
        receipt_identity=model.receipt_identity,
        compute_dtype=_torch_dtype(torch, recipe.knobs.diffusion_dtype),
        conditioning_identity=conditioner_identity,
    )


def _minimax_h3_custom_sampling_runtime(
    handle: NativeRuntimeHandle,
    positive: object,
    negative: object | None,
    inference: Any,
) -> Any | None:
    """The H3 DiT runtime for a decomposed sampling invocation, or None
    for non-H3 handles."""
    if _minimax_h3_model_handle(handle, inference) is None:
        return None
    carrier = _minimax_h3_conditioning_carrier(positive, inference, "positive")
    if carrier is None:
        raise TypeError("positive must contain prepared MiniMax H3 conditioning")
    if negative not in ([], None):
        negative_carrier = _minimax_h3_conditioning_carrier(negative, inference, "negative")
        if (
            negative_carrier is not None
            and negative_carrier.runtime_identity != carrier.runtime_identity
        ):
            raise ValueError(
                "negative conditioning was prepared by a different MiniMax H3 conditioner component"
            )
    return _minimax_h3_dit_runtime(handle, carrier.runtime_identity, inference, _torch())


def resolve_minimax_h3_component_execution(
    handle: NativeRuntimeHandle, positive: object, negative: object, inference: Any
) -> tuple[Any, object, object] | None:
    runtime = _minimax_h3_custom_sampling_runtime(handle, positive, negative, inference)
    return None if runtime is None else (runtime, positive, negative)


def _minimax_h3_schedule_runtime(handle: NativeRuntimeHandle, inference: Any) -> Any | None:
    """The H3 DiT runtime for schedule-only queries, or None for non-H3
    handles. No conditioner is in scope, so the runtime keeps the model's
    own identity instead of a composed execution identity."""
    validated = _minimax_h3_model_handle(handle, inference)
    if validated is None:
        return None
    inference_torch = importlib.import_module("dinkster_inference_torch")
    model = validated.runtime
    recipe = validated.recipe
    return inference_torch.MiniMaxH3DiTRuntime(
        model.assembled.diffusion,
        model_role=model.model_role.replace("-", "_"),
        runtime_identity=recipe.runtime_identity,
        receipt_identity=model.receipt_identity,
        compute_dtype=_torch_dtype(_torch(), recipe.knobs.diffusion_dtype),
    )


def resolve_ideogram4_component_execution(
    handle: NativeRuntimeHandle,
    positive: object,
    negative: object,
    inference: Any,
    *,
    negative_handle: NativeRuntimeHandle | None = None,
    image_only_negative: bool = False,
) -> tuple[Any, object, object] | None:
    recipe = handle.recipe
    family_id = inference.IDEOGRAM4_CONFIG.family_id
    if recipe.family_id != family_id:
        return None
    if tuple(source.role for source in recipe.sources) != ("diffusion",):
        return None
    inference_torch = importlib.import_module("dinkster_inference_torch")
    base_runtime = handle.runtime
    if (
        not isinstance(base_runtime, inference_torch.Ideogram4DiffusionRuntime)
        or recipe.runtime_identity != base_runtime.runtime_identity
    ):
        raise TypeError("model must be a native Ideogram 4 diffusion component")
    negative_runtime = None
    if negative_handle is not None:
        negative_recipe = negative_handle.recipe
        negative_base = negative_handle.runtime
        if (
            negative_recipe.family_id != family_id
            or tuple(source.role for source in negative_recipe.sources) != ("diffusion",)
            or not isinstance(negative_base, inference_torch.Ideogram4DiffusionRuntime)
            or negative_recipe.runtime_identity != negative_base.runtime_identity
        ):
            raise TypeError("model_negative must be a native Ideogram 4 diffusion component")
    positive_carrier, positive_binding = _component_bound_carrier(positive, inference)
    if positive_binding is None:
        raise TypeError("positive must be Ideogram 4 component-bound conditioning")
    if positive_binding.family_id != family_id or positive_binding.role != "qwen3vl_8b":
        raise ValueError("positive Ideogram 4 conditioning has the wrong component binding")
    negative_carrier = None
    if negative not in ([], None):
        negative_carrier, negative_binding = _component_bound_carrier(negative, inference)
        if negative_binding is None:
            raise TypeError("negative must be Ideogram 4 component-bound conditioning or empty")
        if negative_binding != positive_binding:
            raise ValueError("Ideogram 4 conditioning lanes must share one component binding")
    components = {
        "diffusion": recipe.runtime_identity,
        "qwen3vl_8b": positive_binding.identity,
    }
    if negative_handle is not None:
        components["negative-diffusion"] = negative_handle.recipe.runtime_identity
    composition = inference.compose_execution(family_id, components)
    torch = _torch()
    runtime = inference_torch.Ideogram4DiffusionRuntime(
        base_runtime.assembled.diffusion,
        runtime_identity=composition.execution_identity,
        compute_dtype=_torch_dtype(torch, recipe.knobs.diffusion_dtype),
    )
    if negative_handle is not None:
        negative_runtime = inference_torch.Ideogram4DiffusionRuntime(
            negative_handle.runtime.assembled.diffusion,
            runtime_identity=composition.execution_identity,
            compute_dtype=_torch_dtype(torch, negative_handle.recipe.knobs.diffusion_dtype),
        )
    conditioning = runtime.prepare_single_stream_conditioning(positive_carrier)
    rows = [[conditioning.embeddings, {_NATIVE_PREPARED_CONDITIONING_KEY: conditioning}]]
    negative_rows: object = []
    target_runtime = runtime if negative_runtime is None else negative_runtime
    uncond = (
        target_runtime.image_only_conditioning()
        if image_only_negative
        else (
            None
            if negative_carrier is None
            else target_runtime.prepare_single_stream_conditioning(negative_carrier)
        )
    )
    if uncond is not None:
        if negative_runtime is not None:
            uncond = inference_torch.RoutedConditioning(
                embeddings=uncond.embeddings,
                pooled=uncond.pooled,
                evaluation=negative_runtime.conditioning_evaluation(),
                source=uncond,
            )
        negative_rows = [[uncond.embeddings, {_NATIVE_PREPARED_CONDITIONING_KEY: uncond}]]
    return runtime, rows, negative_rows


def resolve_seedvr2_component_execution(
    handle: NativeRuntimeHandle,
    positive: object,
    negative: object,
    inference: Any,
) -> tuple[Any, object, object] | None:
    recipe = handle.recipe
    runtime = handle.runtime
    sampling_runtime = getattr(runtime, "component_sampling_runtime", runtime)
    if getattr(sampling_runtime, "runtime_identity", None) != recipe.runtime_identity:
        raise TypeError("component sampling runtime identity does not match its model handle")
    inference_torch = importlib.import_module("dinkster_inference_torch")

    def prepare(value: object, name: str, branch: str) -> object:
        conditioning = inference_torch.materialize_seedvr2_conditioning(
            value, device=handle.load_device
        )
        if conditioning.branch != branch:
            raise TypeError(f"{name} must come from Apply SeedVR2 Conditioning")
        if conditioning.component_identity != recipe.runtime_identity:
            raise ValueError(f"{name} SeedVR2 conditioning belongs to a different model")
        return [
            [
                conditioning.embeddings,
                {_NATIVE_PREPARED_CONDITIONING_KEY: conditioning},
            ]
        ]

    positive_rows = prepare(positive, "positive", "positive")
    negative_rows: object = (
        [] if negative in ([], None) else prepare(negative, "negative", "negative")
    )
    return sampling_runtime, positive_rows, negative_rows


def _sampling_memory_requirements(runtime: Any, samples: Any) -> tuple[int, int | None]:
    estimate = getattr(runtime, "sampling_memory_requirements", None)
    return (0, None) if estimate is None else estimate(tuple(samples.shape))


def resolve_trellis2_component_execution(
    handle: NativeRuntimeHandle, positive: object, negative: object, inference: Any
) -> tuple[Any, object, object] | None:
    recipe = handle.recipe
    if recipe.family_id != inference.TRELLIS2.id:
        return None
    source_roles = tuple(source.role for source in recipe.sources)
    if source_roles not in (
        ("diffusion",),
        ("shape", "shape-512", "structure", "texture", "texture-512"),
    ):
        return None
    inference_torch = importlib.import_module("dinkster_inference_torch")
    runtime = handle.runtime
    if (
        not isinstance(runtime, inference_torch.Trellis2DiffusionRuntime)
        or recipe.runtime_identity != runtime.runtime_identity
    ):
        raise TypeError("model must be a native TRELLIS.2 diffusion component")
    carrier_type = inference.ResidentConditioningCarrier
    resource_type = inference_torch.Trellis2ConditioningResource
    positive_carrier = cast("Any", positive)
    if not isinstance(positive, carrier_type) or not isinstance(
        positive_carrier.payload, resource_type
    ):
        raise TypeError("positive must be resident TRELLIS.2 conditioning")
    positive_resource = positive_carrier.payload
    if positive_resource.guidance_role is not inference.GuidanceRole.CONDITIONAL:
        raise ValueError("positive TRELLIS.2 conditioning has the wrong guidance lane")
    negative_resource = None
    if negative not in ([], None):
        negative_carrier = cast("Any", negative)
        if not isinstance(negative, carrier_type) or not isinstance(
            negative_carrier.payload, resource_type
        ):
            raise TypeError("negative must be resident TRELLIS.2 conditioning or empty")
        negative_resource = negative_carrier.payload
        if negative_resource.guidance_role is not inference.GuidanceRole.UNCONDITIONAL:
            raise ValueError("negative TRELLIS.2 conditioning has the wrong guidance lane")
        if not positive_resource.shares_backing(negative_resource):
            raise ValueError("TRELLIS.2 conditioning lanes must share one backing resource")
        if negative_resource.stage != positive_resource.stage:
            raise ValueError("TRELLIS.2 conditioning lanes must use the same stage")

    def rows(resource: object) -> list[list[object]]:
        return [
            [
                inference.PreparedMultiStreamConditioning(
                    runtime.conditioning_identity,
                    resource,
                ),
                dict[str, object](),
            ]
        ]

    return (
        runtime,
        rows(positive_resource),
        ([] if negative_resource is None else rows(negative_resource)),
    )
