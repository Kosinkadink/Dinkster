"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from ..family_registry import load_component
from ..native_arm_core import (
    Any,
    ExitStack,
    Mapping,
    NativeComponentHandle,
    NativeRuntimeHandle,
    Node,
    NodeSchema,
    _split_ltx_frame_rate,
    _torch,
    _with_ltx_frame_rate,
    cast,
    importlib,
    math,
    re,
    replace,
)
from ..native_arm_latent_utils import _check_bounds
from ..native_arm_runtime import (
    _native_model,
    _torch_dtype,
)
from ..nodes_guidance import (
    _model_with_guidance_transform,
)
from ..nodes_provider import (
    _generation_provider_schema,
)
from ..nodes_sampling_runtime import (
    _native_component_codec,
)
from .minimax_h3 import (
    _latent_samples,
    _prepared_multistream_carrier,
    load_registered_component,
    resolve_component_execution,
)


class CodecAdapter:
    sequence_content = True

    def __init__(self, value: object) -> None:
        value = load_component(value, "vae", "vae")
        recipe = value.recipe
        assert recipe is not None
        descriptor = importlib.import_module("dinkster_native.native_arm")._component_descriptor(
            recipe.family_id
        )
        latent = descriptor.family.latent
        streams: dict[str, Any] = dict(getattr(latent, "streams", ()))
        if streams:
            latent = streams.get("video")
        if latent is None or latent.dimensions != 3 or not latent.temporal_causal:
            raise TypeError("vae requires a declared causal video latent geometry")
        self._handle = value
        self._resource_identity = value.resource_identity
        module = cast("Any", value.component)
        for method in ("encode", "decode"):
            if not callable(getattr(module, method, None)):
                raise TypeError(f"vae component requires callable {method}")
        config = getattr(module, "config", None)
        for field, expected in (
            ("latent_channels", latent.channels),
            ("spatial_ratio", latent.spatial_downscale),
            ("temporal_ratio", latent.temporal_downscale),
        ):
            dimension = getattr(config, field, None)
            if type(dimension) is not int or dimension <= 0 or dimension != expected:
                raise TypeError(f"vae config.{field} must match declared video latent geometry")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        runtime_type = (
            inference_torch.LTXAVVideoCodecRuntime
            if "audio" in streams
            else inference_torch.LTXVVideoCodecRuntime
        )
        native = importlib.import_module("dinkster_native.native_arm")
        self._runtime = runtime_type(
            module, compute_dtype=native._torch_dtype(native._torch(), recipe.knobs.vae_dtype)
        )
        self.descriptor = self._runtime.codec.descriptor
        for field in ("channels", "spatial_downscale", "temporal_downscale"):
            if getattr(self.descriptor.latent, field) != getattr(latent, field):
                raise TypeError(f"vae codec requires matching declared latent.{field}")
        self.load_device = value.load_device

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
        return self._runtime.decode_latent(latent)

    def decode_latent_tiled(
        self, latent: Any, *, tile: tuple[int, ...], overlap: tuple[int, ...]
    ) -> Any:
        return self._runtime.codec.decode_tiled(latent, tile=tile, overlap=overlap)

    def encode_content(self, content: Any) -> Any:
        return self._runtime.encode_content(content)

    def encode_content_tiled(
        self, content: Any, *, tile: tuple[int, ...], overlap: tuple[int, ...]
    ) -> Any:
        return self._runtime.codec.encode_tiled(content, tile=tile, overlap=overlap)


def _retime_ltx_conditioning(
    positive: object,
    negative: object,
    frame_rate: float,
    *,
    accept_carriers: bool,
    payload_type_name: str,
    family_id: str,
    family_name: str,
) -> dict[str, object]:
    rate = float(frame_rate)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"frame_rate must be a positive finite number, got {frame_rate}")
    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    payload_type = getattr(inference_torch, payload_type_name)

    def retimed(value: object, name: str) -> object:
        if accept_carriers and type(value) is inference.ConditioningCarrier:
            return _with_ltx_frame_rate(
                value,
                inference,
                rate,
                family_id=family_id,
                family_name=family_name,
            )
        prepared = _prepared_multistream_carrier(value, inference, name)
        if prepared is None:
            raise TypeError(f"{name} must contain prepared multi-stream conditioning")
        if type(prepared.payload) is not payload_type:
            raise TypeError(f"{name} must contain {family_name} text conditioning")
        payload = replace(prepared.payload, frame_rate=rate)
        rows: list[list[object]] = [
            [
                inference.PreparedMultiStreamConditioning(prepared.runtime_identity, payload),
                dict[str, object](),
            ]
        ]
        return rows

    return {
        "positive": retimed(positive, "positive"),
        "negative": retimed(negative, "negative"),
    }


class GenerationLTXAVConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxav_conditioning")

    @classmethod
    def execute(
        cls, *, positive: object, negative: object, frame_rate: float
    ) -> Mapping[str, object]:
        return cls.outputs(
            **_retime_ltx_conditioning(
                positive,
                negative,
                frame_rate,
                accept_carriers=True,
                payload_type_name="LTXAVPreparedConditioning",
                family_id="dinkster.ltxav",
                family_name="LTX-2 audio-video",
            )
        )


def _ltxav_audio_codec(value: object) -> tuple[Any, Any, Any]:
    inference = importlib.import_module("dinkster_inference")
    if isinstance(value, NativeComponentHandle):
        value.require_active()
        recipe = value.recipe
        if (
            recipe is None
            or tuple(source.role for source in recipe.sources) != ("audio_vae",)
            or recipe.runtime_identity != value.resource_identity
        ):
            raise TypeError("audio_vae requires an audio_vae source and matching recipe identity")
        inference.ComponentBinding("audio_vae", recipe.family_id, value.resource_identity)
        component = cast("Any", value.component)
        audio_vae = getattr(component, "audio_vae", None)
        for method in ("encode", "decode"):
            if not callable(getattr(audio_vae, method, None)):
                raise TypeError(f"audio_vae codec requires callable audio_vae.{method}")
        config = getattr(audio_vae, "config", None)
        if (
            getattr(config, "z_channels", None) != 8
            or getattr(config, "latent_frequency_bins", None) != 16
        ):
            raise TypeError("audio_vae codec requires [batch,8,time,16] latent geometry")
        vocoder = getattr(component, "vocoder", None)
        if not callable(vocoder) or getattr(vocoder, "config", None) is None:
            raise TypeError("audio_vae codec requires a callable vocoder with config")
        return component, value.load_device, value.stage
    raise TypeError("audio_vae must provide a native audio codec component")


def _ltxav_audio_vae(value: object) -> tuple[Any, Any, Any]:
    codec, load_device, stage = _ltxav_audio_codec(value)
    return codec.audio_vae, load_device, stage


def _ltxav_reference_audio_value(value: object) -> tuple[Any, int]:
    if not isinstance(value, Mapping):
        raise TypeError("reference_audio must be the standard waveform/sample_rate mapping")
    audio = cast("Mapping[str, object]", value)
    if set(audio) != {"waveform", "sample_rate"}:
        raise TypeError("reference_audio must be the standard waveform/sample_rate mapping")
    torch = _torch()
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if type(waveform) is not torch.Tensor:
        raise TypeError("reference_audio.waveform must be an exact torch.Tensor")
    tensor = cast("Any", waveform)
    if (
        tensor.layout is not torch.strided
        or not tensor.is_floating_point()
        or tensor.ndim != 3
        or tensor.shape[0] <= 0
        or tensor.shape[1] not in (1, 2)
        or tensor.shape[2] <= 0
    ):
        raise ValueError("reference_audio.waveform must be nonempty floating [batch,1|2,samples]")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("reference_audio.sample_rate must be a positive integer")
    return tensor, sample_rate


def _attach_ltxav_reference_audio(
    positive: object, negative: object, audio_latent: Any
) -> dict[str, object]:
    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    positive_is_carrier = type(positive) is inference.ConditioningCarrier
    negative_is_carrier = type(negative) is inference.ConditioningCarrier
    if positive_is_carrier != negative_is_carrier:
        raise TypeError("positive and negative must use the same conditioning representation")
    if positive_is_carrier:
        attached = inference_torch.ltxav_reference_audio_conditioning(
            positive, negative, audio_latent
        )
        return {"positive": attached[0], "negative": attached[1]}

    positive_prepared = _prepared_multistream_carrier(positive, inference, "positive")
    negative_prepared = _prepared_multistream_carrier(negative, inference, "negative")
    if positive_prepared is None or negative_prepared is None:
        raise TypeError("positive and negative must contain prepared LTX-2 conditioning")
    if positive_prepared.runtime_identity != negative_prepared.runtime_identity:
        raise ValueError("positive and negative conditioning were prepared by different runtimes")
    audio_tokens = audio_latent.permute(0, 2, 1, 3).reshape(
        audio_latent.shape[0],
        audio_latent.shape[2],
        audio_latent.shape[1] * audio_latent.shape[3],
    )

    def attach(prepared: Any, name: str) -> list[list[object]]:
        payload = prepared.payload
        if type(payload) is not inference_torch.LTXAVPreparedConditioning:
            raise TypeError(f"{name} must contain LTX-2 audio-video text conditioning")
        if payload.reference_audio is not None:
            raise ValueError(f"{name} LTX-2 conditioning already has reference audio")
        return [
            [
                inference.PreparedMultiStreamConditioning(
                    prepared.runtime_identity,
                    replace(payload, reference_audio=audio_tokens),
                ),
                {},
            ]
        ]

    return {
        "positive": attach(positive_prepared, "positive"),
        "negative": attach(negative_prepared, "negative"),
    }


class GenerationLTXAVReferenceAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxav_reference_audio")

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        reference_audio: object,
        audio_vae: object,
    ) -> Mapping[str, object]:
        audio_latent = _encode_ltxav_reference_audio(reference_audio, audio_vae)
        return cls.outputs(**_attach_ltxav_reference_audio(positive, negative, audio_latent))


def _encode_ltxav_reference_audio(reference_audio: object, audio_vae: object) -> Any:
    waveform, sample_rate = _ltxav_reference_audio_value(reference_audio)
    module, load_device, stage = _ltxav_audio_vae(audio_vae)
    torch = _torch()
    with stage():
        with torch.inference_mode():
            audio_latent = module.encode(waveform.to(load_device), sample_rate).float()
    if (
        type(audio_latent) is not torch.Tensor
        or audio_latent.layout is not torch.strided
        or not audio_latent.is_floating_point()
        or audio_latent.ndim != 4
        or audio_latent.shape[0] != waveform.shape[0]
        or audio_latent.shape[1] != 8
        or audio_latent.shape[2] <= 0
        or audio_latent.shape[3] != 16
    ):
        raise TypeError("LTX-2 audio VAE must encode nonempty floating [batch,8,time,16]")
    return audio_latent


class GenerationLTXAVIDLoRAReferenceAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxav_id_lora_reference_audio")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        positive: object,
        negative: object,
        reference_audio: object,
        audio_vae: object,
        identity_guidance_scale: float,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("identity_guidance_scale", identity_guidance_scale, 0.0, 100.0),
            ("start_percent", start_percent, 0.0, 1.0),
            ("end_percent", end_percent, 0.0, 1.0),
        )
        handle, inference_torch, transform_index = _ltxav_guidance_runtime(model)
        audio_latent = _encode_ltxav_reference_audio(reference_audio, audio_vae)
        conditioning = _attach_ltxav_reference_audio(positive, negative, audio_latent)
        guided_model = model
        if identity_guidance_scale != 0.0:
            sigma_start = handle.runtime.custom_sampling_percent_to_sigma(
                start_percent, return_actual_sigma=False
            )
            sigma_end = handle.runtime.custom_sampling_percent_to_sigma(
                end_percent, return_actual_sigma=False
            )
            contribution = inference_torch.ltxav_identity_guidance(
                float(identity_guidance_scale),
                float(sigma_start),
                float(sigma_end),
                order=transform_index,
            )
            guided_model = _model_with_guidance_transform(
                model,
                "dinkster.ltxav_id_lora_reference_audio",
                contribution,
            )
        return cls.outputs(model=guided_model, **conditioning)


def _ltxav_guidance_runtime(model: object) -> tuple[Any, Any, int]:
    handle, _, _, _, _, transforms, _, _ = _native_model(model, "model")
    inference = importlib.import_module("dinkster_inference")
    inference_torch = importlib.import_module("dinkster_inference_torch")
    recipe = handle.recipe
    if "diffusion" not in tuple(source.role for source in recipe.sources):
        raise TypeError("model requires a diffusion source binding")
    inference.ComponentBinding("diffusion", recipe.family_id, recipe.runtime_identity)
    runtime = handle.runtime
    if runtime.runtime_identity != recipe.runtime_identity:
        raise ValueError("model runtime identity must match its recipe")
    for method in ("custom_sampling_percent_to_sigma", "sample_custom", "check_custom_sampling"):
        if not callable(getattr(runtime, method, None)):
            raise TypeError(f"model requires callable {method}")
    geometry = getattr(runtime, "component_sampling_runtime", runtime)
    for name, fields in (
        ("video_vae_config", ("latent_channels", "spatial_ratio", "temporal_ratio")),
        ("audio_vae_config", ("z_channels", "latent_frequency_bins")),
    ):
        config = getattr(geometry, name, None)
        for field in fields:
            dimension = getattr(config, field, None)
            if type(dimension) is not int or dimension <= 0:
                raise TypeError(f"model requires positive {name}.{field} for audio-video guidance")
    return handle, inference_torch, len(transforms)


class GenerationLTXVSpatioTemporalGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_spatiotemporal_guidance")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        scale: float,
        blocks: str,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("scale", scale, 0.0, 100.0),
            ("start_percent", start_percent, 0.0, 1.0),
            ("end_percent", end_percent, 0.0, 1.0),
        )
        if type(blocks) is not str:
            raise TypeError("blocks must be an exact string")
        block_set = frozenset(int(value) for value in re.findall(r"\d+", blocks))
        if scale == 0.0 or not block_set:
            return cls.outputs(model=model)
        handle, inference_torch, transform_index = _ltxav_guidance_runtime(model)
        sigma_start = handle.runtime.custom_sampling_percent_to_sigma(
            start_percent, return_actual_sigma=False
        )
        sigma_end = handle.runtime.custom_sampling_percent_to_sigma(
            end_percent, return_actual_sigma=False
        )
        contribution = inference_torch.ltxav_spatiotemporal_guidance(
            float(scale),
            block_set,
            float(sigma_start),
            float(sigma_end),
            lane_id=f"dinkster.ltxav.stg-perturbed:{transform_index}",
            order=transform_index,
        )
        return cls.outputs(
            model=_model_with_guidance_transform(
                model,
                "dinkster.ltxv_spatiotemporal_guidance",
                contribution,
            )
        )


class GenerationLTXVModalityGuidance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_modality_guidance")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        modality_scale: float,
        start_percent: float,
        end_percent: float,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("modality_scale", modality_scale, 1.0, 100.0),
            ("start_percent", start_percent, 0.0, 1.0),
            ("end_percent", end_percent, 0.0, 1.0),
        )
        if math.isclose(modality_scale, 1.0):
            return cls.outputs(model=model)
        handle, inference_torch, transform_index = _ltxav_guidance_runtime(model)
        sigma_start = handle.runtime.custom_sampling_percent_to_sigma(
            start_percent, return_actual_sigma=False
        )
        sigma_end = handle.runtime.custom_sampling_percent_to_sigma(
            end_percent, return_actual_sigma=False
        )
        contribution = inference_torch.ltxav_modality_guidance(
            float(modality_scale),
            float(sigma_start),
            float(sigma_end),
            lane_id=f"dinkster.ltxav.modality-decoupled:{transform_index}",
            order=transform_index,
        )
        return cls.outputs(
            model=_model_with_guidance_transform(
                model,
                "dinkster.ltxv_modality_guidance",
                contribution,
            )
        )


class GenerationLTXVDurationPredictor(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_duration_predictor")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        positive: object,
        duration_head: object,
        frame_rate: float,
        min_seconds: float,
        max_seconds: float,
    ) -> Mapping[str, object]:
        _check_bounds(
            ("frame_rate", frame_rate, 1.0, 120.0),
            ("min_seconds", min_seconds, 0.5, 120.0),
            ("max_seconds", max_seconds, 0.5, 120.0),
        )
        if max_seconds < min_seconds:
            raise ValueError("max_seconds must be greater than or equal to min_seconds")
        model_handle, inference_torch, _ = _ltxav_guidance_runtime(model)
        if not isinstance(duration_head, NativeComponentHandle):
            raise TypeError("duration_head must provide the LTX-2 duration component")
        duration_head.require_active()
        recipe = duration_head.recipe
        if (
            recipe is None
            or tuple(source.role for source in recipe.sources) != ("duration_head",)
            or recipe.runtime_identity != duration_head.resource_identity
        ):
            raise TypeError(
                "duration_head requires a duration_head source and matching recipe identity"
            )

        inference = importlib.import_module("dinkster_inference")
        inference.ComponentBinding(
            "duration_head", recipe.family_id, duration_head.resource_identity
        )
        if not callable(duration_head.component):
            raise TypeError("duration_head component must be callable")
        runtime = model_handle.runtime
        if type(positive) is inference.ConditioningCarrier:
            execution = resolve_component_execution(model_handle, positive, [], inference)
            if execution is None:
                raise TypeError("positive must contain component-bound LTX-2 text conditioning")
            runtime, prepared_positive, _ = execution
            prepared = _prepared_multistream_carrier(prepared_positive, inference, "positive")
            if prepared is None:
                raise TypeError("positive must contain LTX-2 text conditioning")
            conditioning = prepared.payload
        else:
            prepared = _prepared_multistream_carrier(positive, inference, "positive")
            if prepared is None:
                raise TypeError("positive must contain LTX-2 text conditioning")
            if prepared.runtime_identity != runtime.conditioning_identity:
                raise ValueError("positive conditioning was prepared by a different runtime")
            conditioning = prepared.payload
        if type(conditioning) is not inference_torch.LTXAVPreparedConditioning:
            raise TypeError("positive must contain LTX-2 text conditioning")

        assembled = getattr(runtime, "assembled", None)
        diffusion = getattr(assembled, "diffusion", None)
        if not callable(getattr(assembled, "compute_dtype", None)):
            raise TypeError("model requires callable assembled.compute_dtype")
        if not callable(getattr(diffusion, "preprocess_text_embeds", None)):
            raise TypeError("model requires callable diffusion.preprocess_text_embeds")
        config = getattr(diffusion, "config", None)
        for field in ("cross_attention_dim", "audio_cross_attention_dim"):
            dimension = getattr(config, field, None)
            if type(dimension) is not int or dimension <= 0:
                raise TypeError(f"model requires positive diffusion.config.{field}")
        torch = _torch()
        with duration_head.stage_with(model_handle, "diffusion"):
            with torch.inference_mode():
                diffusion = runtime.assembled.diffusion
                context = conditioning.text[:1].to(
                    device=model_handle.load_device,
                    dtype=runtime.assembled.compute_dtype("diffusion"),
                )
                processed = diffusion.preprocess_text_embeds(context)
                video_tokens, audio_tokens = torch.split(
                    processed,
                    [
                        diffusion.config.cross_attention_dim,
                        diffusion.config.audio_cross_attention_dim,
                    ],
                    dim=-1,
                )
                duration_module = cast("Any", duration_head.component)
                seconds = float(
                    duration_module(video_tokens.float(), audio_tokens.float())[0].item()
                )
        frames = inference_torch.ltx_duration_frames(
            seconds,
            float(frame_rate),
            float(min_seconds),
            float(max_seconds),
        )
        return cls.outputs(num_frames=frames, seconds=seconds)


class GenerationLTXVConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_conditioning")

    @classmethod
    def execute(
        cls, *, positive: object, negative: object, frame_rate: float
    ) -> Mapping[str, object]:
        return cls.outputs(
            **_retime_ltx_conditioning(
                positive,
                negative,
                frame_rate,
                accept_carriers=True,
                payload_type_name="LTXVPreparedConditioning",
                family_id="dinkster.ltxv",
                family_name="LTX-Video",
            )
        )


def _ltxv_media_codec(value: object) -> Any:
    inference = importlib.import_module("dinkster_inference")
    if isinstance(value, NativeComponentHandle):
        codec = _native_component_codec(value)
    elif isinstance(value, NativeRuntimeHandle):
        value.require_active()
        runtime = value.runtime
        descriptor = getattr(getattr(runtime, "codec", None), "descriptor", None)
        if descriptor is None:
            raise TypeError("vae runtime requires a codec descriptor")
        for method in ("encode_content", "decode_latent"):
            if not callable(getattr(runtime, method, None)):
                raise TypeError(f"vae runtime requires callable {method}")
        codec = inference.RuntimeCodecAdapter(value, descriptor=descriptor)
    else:
        codec = value
    codec = inference.require_inference_codec_handle(codec, "vae")
    descriptor = codec.descriptor
    latent = descriptor.latent
    if (
        descriptor.kind != "video"
        or descriptor.content_channels != 3
        or latent.dimensions != 3
        or not latent.temporal_causal
    ):
        raise TypeError("vae requires an RGB causal video codec with three latent dimensions")
    for field in ("channels", "spatial_downscale", "temporal_downscale"):
        dimension = getattr(latent, field)
        if type(dimension) is not int or dimension <= 0:
            raise TypeError(f"vae requires positive latent.{field}")
    return codec


def _ltxv_media_latent(value: object, name: str) -> tuple[dict[object, object], Any, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a latent mapping")
    metadata = dict(cast("Mapping[object, object]", value))
    inference = importlib.import_module("dinkster_inference")
    torch = _torch()
    samples, streams = _latent_samples(cast("object", value), torch, inference, name)
    if streams is None:
        streams = inference.MultiStreamLatent.from_pairs((("video", samples),))
    elif streams.roles != ("video",):
        raise TypeError(f"{name} must contain the exact LTX-Video latent stream")
    video = streams.by_role("video")
    if (
        type(video) is not torch.Tensor
        or not video.is_floating_point()
        or video.ndim != 5
        or any(size <= 0 for size in video.shape)
    ):
        raise TypeError(f"{name} video stream must be nonempty floating [B,C,T,H,W]")
    return metadata, streams, video


def _ltxv_image(value: object) -> Any:
    torch = _torch()
    if type(value) is not torch.Tensor:
        raise TypeError("image must be nonempty floating [frames,height,width,channels>=3]")
    image = cast("Any", value)
    if (
        not image.is_floating_point()
        or image.ndim != 4
        or any(size <= 0 for size in image.shape[:3])
        or image.shape[3] < 3
    ):
        raise TypeError("image must be nonempty floating [frames,height,width,channels>=3]")
    return image


def _encode_ltxv_frames(codec: Any, image: Any, width: int, height: int) -> Any:
    torch = _torch()
    frames = _ltxv_image(image)
    nchw = frames[..., :3].movedim(-1, 1)
    if tuple(nchw.shape[-2:]) != (height, width):
        old_height, old_width = nchw.shape[-2:]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        nchw = nchw.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
        nchw = torch.nn.functional.interpolate(nchw, size=(height, width), mode="bilinear")
    content = nchw.movedim(0, 1).unsqueeze(0)
    with codec.stage():
        with torch.inference_mode():
            encoded = codec.encode_content(content.to(codec.load_device))
    if (
        type(encoded) is not torch.Tensor
        or not encoded.is_floating_point()
        or encoded.ndim != 5
        or encoded.shape[0] != 1
        or encoded.shape[1] != codec.descriptor.latent.channels
        or any(size <= 0 for size in encoded.shape)
    ):
        raise TypeError("LTX-Video VAE encode must return nonempty floating [1,C,T,H,W]")
    return encoded


def _repeat_ltxv_frames(encoded: Any, batch_size: int) -> Any:
    if batch_size == 1:
        return encoded
    return encoded.expand(batch_size, -1, -1, -1, -1)


def _ltxv_media_output(
    metadata: Mapping[object, object], streams: object, denoise_mask: object
) -> dict[object, object]:
    result = dict(metadata)
    result["samples"] = streams
    result["noise_mask"] = denoise_mask
    return result


def _split_ltxv_component_binding(
    positive: object, negative: object, inference: Any
) -> tuple[object, object, object | None]:
    positive_carrier, positive_binding = inference.split_component_conditioning(positive)
    negative_carrier, negative_binding = inference.split_component_conditioning(negative)
    if positive_binding != negative_binding:
        raise ValueError("LTX-Video conditioning lanes must share one component binding")
    return positive_carrier, negative_carrier, positive_binding


def _restore_ltxv_component_binding(
    value: object, binding: object | None, inference: Any
) -> object:
    if binding is None:
        return value
    return inference.bind_component_conditioning(value, binding)


class GenerationLTXVImageToVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_image_to_video")

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        image: object,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        strength: float,
    ) -> Mapping[str, object]:
        for name, value in (
            ("width", width),
            ("height", height),
            ("length", length),
            ("batch_size", batch_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        codec = _ltxv_media_codec(vae)
        latent = codec.descriptor.latent
        if width % latent.spatial_downscale or height % latent.spatial_downscale:
            raise ValueError("width and height must be divisible by the LTX-Video spatial scale")
        latent_height = height // latent.spatial_downscale
        latent_width = width // latent.spatial_downscale
        torch = _torch()
        video = torch.zeros(
            (
                batch_size,
                latent.channels,
                (length - 1) // latent.temporal_downscale + 1,
                latent_height,
                latent_width,
            ),
            device="cpu",
            dtype=torch.float32,
        )
        inference = importlib.import_module("dinkster_inference")
        streams = inference.MultiStreamLatent.from_pairs((("video", video),))
        encoded = _repeat_ltxv_frames(_encode_ltxv_frames(codec, image, width, height), batch_size)
        conditioned, denoise_mask = importlib.import_module(
            "dinkster_inference_torch"
        ).ltxv_condition_initial_frames(streams, encoded, strength=float(strength))
        return cls.outputs(
            positive=positive,
            negative=negative,
            latent={"samples": conditioned, "noise_mask": denoise_mask},
        )


class GenerationLTXVImageToVideoInplace(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_image_to_video_inplace")

    @classmethod
    def execute(
        cls,
        *,
        vae: object,
        image: object,
        latent: object,
        strength: float,
        bypass: bool = False,
    ) -> Mapping[str, object]:
        if type(bypass) is not bool:
            raise TypeError("bypass must be a bool")
        if bypass:
            return cls.outputs(latent=latent)
        metadata, streams, video = _ltxv_media_latent(latent, "latent")
        codec = _ltxv_media_codec(vae)
        descriptor = codec.descriptor.latent
        width = video.shape[4] * descriptor.spatial_downscale
        height = video.shape[3] * descriptor.spatial_downscale
        encoded = _repeat_ltxv_frames(
            _encode_ltxv_frames(codec, image, width, height), video.shape[0]
        )
        conditioned, denoise_mask = importlib.import_module(
            "dinkster_inference_torch"
        ).ltxv_condition_initial_frames(
            streams,
            encoded,
            strength=float(strength),
            denoise_mask=metadata.get("noise_mask"),
        )
        return cls.outputs(latent=_ltxv_media_output(metadata, conditioned, denoise_mask))


class GenerationLTXVAddGuide(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_add_guide")

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        vae: object,
        latent: object,
        image: object,
        frame_idx: int,
        strength: float,
        attention_mask: object = None,
    ) -> Mapping[str, object]:
        if type(frame_idx) is not int:
            raise TypeError("frame_idx must be an int")
        metadata, streams, video = _ltxv_media_latent(latent, "latent")
        codec = _ltxv_media_codec(vae)
        descriptor = codec.descriptor.latent
        source = _ltxv_image(image)
        frame_count = (
            (source.shape[0] - 1) // descriptor.temporal_downscale
        ) * descriptor.temporal_downscale + 1
        source = source[:frame_count]

        inference = importlib.import_module("dinkster_inference")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        ltx_media = importlib.import_module("dinkster_inference_torch.ltx_media")
        positive, negative, component_binding = _split_ltxv_component_binding(
            positive, negative, inference
        )
        without_rate, _ = _split_ltx_frame_rate(positive, inference)
        _, guides = ltx_media.materialize_ltxv_guides(without_rate)
        generated_frames = video.shape[2] - sum(guide.latent_shape[0] for guide in guides)
        if generated_frames <= 0:
            raise ValueError("LTX-Video guides leave no generated latent frames")
        resolved_index = frame_idx
        if resolved_index < 0:
            resolved_index = max(
                (generated_frames - 1) * descriptor.temporal_downscale + 1 + resolved_index,
                0,
            )
        causal = resolved_index == 0 or frame_count == 1
        encoded_source = source if causal else _torch().cat((source[:1], source))
        encoded_source = encoded_source[
            : (encoded_source.shape[0] - 1)
            // descriptor.temporal_downscale
            * descriptor.temporal_downscale
            + 1
        ]
        width = video.shape[4] * descriptor.spatial_downscale
        height = video.shape[3] * descriptor.spatial_downscale
        encoded = _encode_ltxv_frames(codec, encoded_source, width, height)
        if not causal:
            encoded = encoded[:, :, 1:].clone()
        encoded = _repeat_ltxv_frames(encoded, video.shape[0])

        result = inference_torch.ltxv_add_guide(
            positive,
            negative,
            streams,
            encoded,
            frame_index=frame_idx,
            strength=float(strength),
            denoise_mask=metadata.get("noise_mask"),
            attention_mask=cast("Any", attention_mask),
            scale_factors=(
                descriptor.temporal_downscale,
                descriptor.spatial_downscale,
                descriptor.spatial_downscale,
            ),
            causal_fix=causal,
        )
        return cls.outputs(
            positive=_restore_ltxv_component_binding(result.positive, component_binding, inference),
            negative=_restore_ltxv_component_binding(result.negative, component_binding, inference),
            latent=_ltxv_media_output(metadata, result.latent, result.denoise_mask),
        )


class GenerationLTXVCropGuides(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_crop_guides")

    @classmethod
    def execute(cls, *, positive: object, negative: object, latent: object) -> Mapping[str, object]:
        metadata, streams, _ = _ltxv_media_latent(latent, "latent")
        inference = importlib.import_module("dinkster_inference")
        positive, negative, component_binding = _split_ltxv_component_binding(
            positive, negative, inference
        )
        result = importlib.import_module("dinkster_inference_torch").ltxv_crop_guides(
            positive,
            negative,
            streams,
            metadata.get("noise_mask"),
        )
        return cls.outputs(
            positive=_restore_ltxv_component_binding(result.positive, component_binding, inference),
            negative=_restore_ltxv_component_binding(result.negative, component_binding, inference),
            latent=_ltxv_media_output(metadata, result.latent, result.denoise_mask),
        )


class GenerationLTXVLatentUpsampler(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxv_latent_upsampler")

    @classmethod
    def execute(
        cls,
        *,
        samples: object,
        upscale_model: object,
        vae: object,
    ) -> Mapping[str, object]:
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a latent mapping")
        torch = _torch()
        latent = cast("Mapping[object, object]", samples).get("samples")
        tensor = cast("Any", latent)
        if (
            type(latent) is not torch.Tensor
            or tensor.layout is not torch.strided
            or not tensor.is_floating_point()
            or tensor.ndim != 5
            or tensor.shape[0] <= 0
            or tensor.shape[1] != 128
            or min(tensor.shape[2:]) <= 0
        ):
            raise ValueError(
                "samples['samples'] must be nonempty floating [batch,128,time,height,width]"
            )
        upscaler = load_registered_component(upscale_model, "upscale_model", "latent_upscaler")
        video_vae = load_registered_component(vae, "vae", "vae")
        inference_torch = importlib.import_module("dinkster_inference_torch")
        model = upscaler.component
        vae_model = video_vae.component
        if type(model) is not inference_torch.LTXLatentUpsampler:
            raise TypeError("upscale_model must contain the LTX-2 latent upscaler")
        if not isinstance(
            vae_model,
            (inference_torch.LTXVideoVAE, inference_torch.LTXDiffusionVideoVAE),
        ):
            raise TypeError("vae must contain an LTX-2 video VAE")
        upscaler_model = cast("Any", model)
        statistics = cast("Any", vae_model).per_channel_statistics
        if video_vae.coordinator is not upscaler.coordinator:
            raise ValueError("upscale_model and vae use different residency coordinators")
        recipe = upscaler.recipe
        assert recipe is not None
        compute_dtype = _torch_dtype(torch, recipe.knobs.vae_dtype)
        with upscaler.coordinator.locked():
            with ExitStack() as stages:
                stages.enter_context(video_vae.stage())
                stages.enter_context(upscaler.stage())
                with torch.inference_mode():
                    value = tensor.to(device=upscaler.load_device, dtype=compute_dtype)
                    value = statistics.un_normalize(value)
                    value = upscaler_model(value)
                    value = statistics.normalize(value)
        output = dict(cast("Mapping[object, object]", samples))
        output["samples"] = value.to(device=tensor.device, dtype=tensor.dtype)
        output.pop("noise_mask", None)
        return cls.outputs(latent=output)
