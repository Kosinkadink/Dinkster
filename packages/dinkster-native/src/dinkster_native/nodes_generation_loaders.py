"""Native-arm nodes split from the shared execution module."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from .families.ltx import _ltxav_audio_codec
from .families.minimax_h3 import (
    load_registered_component,
)
from .families.wan21 import (
    NativeClipTextEncode,
)
from .native_arm_conditioning import (
    _conditioning_carrier,
    _rebound_conditioning_carrier,
)
from .native_arm_core import (
    _CONTROLNET_POOLED_METADATA_KEY,
    _CONTROLNET_TEXT_METADATA_KEY,
    Any,
    AssetRef,
    Mapping,
    MiniMaxMusic3TextEncode,
    NativeComponentHandle,
    Node,
    NodeSchema,
    _native_clip_options,
    _not_cancelled,
    _torch,
    cast,
    clean_enhanced_prompt,
    current_execution_context,
    importlib,
    math,
    native_execution_span,
    prepare_ltx2_prompt,
    replace,
)
from .native_arm_runtime import (
    _ControlHintSnapshot,
    _controlled_conditioning,
    _ControlledConditioning,
    _native_handle,
    _native_model,
    _native_model_sampling_cache,
    _native_model_sampling_timeline,
    _NativeCodecHandle,
    _NativeControlNetResource,
    _NativeModelOverlay,
    _sampling_space_runtime,
    _torch_dtype,
    _weight_storage_dtype,
)
from .native_arm_scheduling import (
    _build_component_runtime_handle,
    _build_ltxav_audio_codec_handle,
    _build_ltxav_text_handle,
    _build_trellis2_split_model_handle,
    _component_descriptor,
    _component_execution_context,
    _LTXAVTextHandle,
)
from .nodes_loaders import (
    NativeControlNetLoader,
    NativeLoadCheckpoint,
    NativeLoadDiffusionModel,
    NativeLoadLora,
    NativeLoadLoraModelOnly,
    _apply_native_lora_stack,
    _apply_native_model_lora_stack,
    _extend_control_binding,
    _native_controlnet,
    _snapshot_control_hint,
)
from .nodes_provider import (
    _generation_lora_mode,
    _generation_provider_schema,
    _require_provider_runtime,
)


class GenerationLoadCheckpoint(NativeLoadCheckpoint):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_checkpoint")

    @classmethod
    def execute(cls, *, checkpoint: object) -> Mapping[str, object]:
        loaded = NativeLoadCheckpoint.execute(checkpoint=checkpoint)
        handle = _require_provider_runtime(loaded["model"], "model")
        inference = importlib.import_module("dinkster_inference")
        codec = _NativeCodecHandle(handle)
        inference.require_inference_codec_handle(codec, "vae")
        return cls.outputs(
            model=handle,
            clip=loaded["clip"],
            vae=codec,
        )


class GenerationLoadControlNet(NativeControlNetLoader):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_controlnet")


def _apply_control_carrier(
    conditioning: object,
    *,
    resource: _NativeControlNetResource,
    hint: _ControlHintSnapshot,
    **kwargs: Any,
) -> Any:
    inference = importlib.import_module("dinkster_inference")
    previous = _controlled_conditioning(conditioning)
    carrier = _conditioning_carrier(
        conditioning if previous is None else previous.conditioning, "conditioning"
    )
    binding = _extend_control_binding(
        None if previous is None else previous.binding,
        resource=resource,
        hint=hint,
        inference=inference,
        **kwargs,
    )
    resources = (*(() if previous is None else previous.resources), resource)
    return inference.ResidentConditioningCarrier(
        _ControlledConditioning(carrier, binding, resources)
    )


class GenerationApplyControlNet(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.apply_controlnet")

    @classmethod
    def execute(
        cls, *, conditioning: object, control_net: object, image: object, strength: float = 1.0
    ) -> Mapping[str, object]:
        if strength == 0.0:
            return cls.outputs(conditioning=conditioning)
        resource = _native_controlnet(control_net)
        hint = _snapshot_control_hint(
            image,
            _torch(),
            importlib.import_module("dinkster_inference_torch"),
            resource.hint_channels,
        )
        return cls.outputs(
            conditioning=_apply_control_carrier(
                conditioning,
                resource=resource,
                hint=hint,
                strength=float(strength),
                start_percent=0.0,
                end_percent=1.0,
                apply_to_uncond=True,
            )
        )


class GenerationApplyControlNetAdvanced(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.apply_controlnet_advanced")

    @classmethod
    def execute(
        cls,
        *,
        positive: object,
        negative: object,
        control_net: object,
        image: object,
        strength: float = 1.0,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        vae: object = None,
    ) -> Mapping[str, object]:
        # Pixel-space controls do not encode their hints with the supplied VAE.
        del vae
        if strength == 0.0:
            return cls.outputs(positive=positive, negative=negative)
        resource = _native_controlnet(control_net)
        hint = _snapshot_control_hint(
            image,
            _torch(),
            importlib.import_module("dinkster_inference_torch"),
            resource.hint_channels,
        )
        return cls.outputs(
            **{
                name: _apply_control_carrier(
                    value,
                    resource=resource,
                    hint=hint,
                    strength=float(strength),
                    start_percent=float(start_percent),
                    end_percent=float(end_percent),
                    apply_to_uncond=False,
                )
                for name, value in (("positive", positive), ("negative", negative))
            }
        )


class GenerationSetControlNetUnionType(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.set_controlnet_union_type")

    @classmethod
    def execute(cls, *, control_net: object, type: str = "auto") -> Mapping[str, object]:
        resource = _native_controlnet(control_net)
        inference = importlib.import_module("dinkster_inference")
        mode = (
            None
            if type == "auto"
            else inference.SDControlMode(provider="sdxl-controlnet-union", token=type.split("/")[0])
        )
        return cls.outputs(control_net=replace(resource, mode=mode))


class GenerationLoadDiffusionModel(NativeLoadDiffusionModel):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_diffusion_model")


def _diffusion_component_assets(components: Mapping[str, object]) -> Mapping[str, AssetRef]:
    members: dict[str, dict[str, object]] = {}
    for input_id, value in components.items():
        member, separator, field = input_id.partition(".")
        if not separator or field not in ("component", "role"):
            raise ValueError(f"invalid diffusion component input {input_id!r}")
        members.setdefault(member, {})[field] = value
    assets: dict[str, AssetRef] = {}
    for member, values in members.items():
        if set(values) != {"component", "role"}:
            raise ValueError(f"diffusion component member {member!r} is incomplete")
        asset = values["component"]
        role = values["role"]
        if not isinstance(asset, AssetRef):
            raise TypeError(f"diffusion component {member!r} must be an AssetRef")
        if not isinstance(role, str) or not role:
            raise TypeError(f"diffusion component {member!r} role must be a non-empty string")
        if role in assets:
            raise ValueError(f"diffusion component role {role!r} is duplicated")
        assets[role] = asset
    return assets


class GenerationLoadDiffusionComponents(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_diffusion_components")

    @classmethod
    def execute(
        cls,
        *,
        components: Mapping[str, object],
        weight_dtype: str = "default",
    ) -> Mapping[str, object]:
        torch = _torch()
        storage_dtype = _weight_storage_dtype(torch, weight_dtype)
        context = current_execution_context()
        if context is None or context.expected_execution_identity is None:
            raise RuntimeError(
                "native load_diffusion_components ran without an expected execution identity"
            )
        with native_execution_span("load", "load"):
            assets = _diffusion_component_assets(components)
            h3_roles = set(assets) & {"fl2va-dit", "ref2va-dit"}
            if len(assets) == 1 and len(h3_roles) == 1:
                artifact_role = next(iter(h3_roles))
                storage_kwargs: dict[str, Any] = (
                    {} if storage_dtype is None else {"storage_dtype": storage_dtype}
                )
                handle = _build_component_runtime_handle(
                    _component_descriptor("dinkster.minimax_h3"),
                    assets[artifact_role],
                    "diffusion",
                    context.expected_execution_identity,
                    torch,
                    compute_dtype=context.diffusion_dtype or "bfloat16",
                    as_model=True,
                    artifact_role=artifact_role,
                    **storage_kwargs,
                )
            else:
                handle = _build_trellis2_split_model_handle(
                    assets,
                    context.expected_execution_identity,
                    torch,
                    compute_dtype=context.diffusion_dtype or "bfloat16",
                    storage_dtype=storage_dtype,
                )
        return cls.outputs(model=handle)


class GenerationLoadLTXAVTextEncoder(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_ltxav_text_encoder")

    @classmethod
    def execute(
        cls,
        *,
        text_encoder: object,
        ckpt_name: object,
        device: str = "default",
    ) -> Mapping[str, object]:
        if not isinstance(text_encoder, AssetRef):
            raise TypeError("text_encoder must be an AssetRef")
        if not isinstance(ckpt_name, AssetRef):
            raise TypeError("ckpt_name must be an AssetRef")
        if device not in ("default", "cpu"):
            raise ValueError("device must be 'default' or 'cpu'")
        context = _component_execution_context("load_ltxav_text_encoder")
        torch = _torch()
        load_device = torch.device("cpu") if device == "cpu" else None
        with native_execution_span("load", "load"):
            handle = _build_ltxav_text_handle(
                text_encoder,
                ckpt_name,
                context.expected_execution_identity,
                torch,
                compute_dtype=context.text_dtype or "bfloat16",
                load_device=load_device,
            )
        return cls.outputs(clip=handle)


class GenerationLoadLTXAVAudioVAE(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_ltxav_audio_vae")

    @classmethod
    def execute(cls, *, ckpt_name: object) -> Mapping[str, object]:
        if not isinstance(ckpt_name, AssetRef):
            raise TypeError("ckpt_name must be an AssetRef")
        context = _component_execution_context("load_ltxav_audio_vae")
        if context.vae_dtype not in (None, "float32"):
            raise RuntimeError("LTX-2 audio codec requires float32 execution")
        with native_execution_span("load", "load"):
            handle = _build_ltxav_audio_codec_handle(
                ckpt_name,
                context.expected_execution_identity,
                _torch(),
            )
        return cls.outputs(audio_vae=handle)


class GenerationLoadLatentUpscaleModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_latent_upscale_model")

    @classmethod
    def execute(cls, *, model_name: object) -> Mapping[str, object]:
        if not isinstance(model_name, AssetRef):
            raise TypeError("model_name must be an AssetRef")
        context = _component_execution_context("load_latent_upscale_model")
        with native_execution_span("load", "load"):
            handle = _build_component_runtime_handle(
                _component_descriptor("dinkster.ltxav"),
                model_name,
                "latent_upscaler",
                context.expected_execution_identity,
                _torch(),
                compute_dtype=context.vae_dtype or "bfloat16",
            )
        return cls.outputs(upscale_model=handle)


class GenerationLTXAVAudioVAEDecode(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.ltxav_audio_vae_decode")

    @classmethod
    def execute(cls, *, samples: object, audio_vae: object) -> Mapping[str, object]:
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a latent mapping")
        inference = importlib.import_module("dinkster_inference")
        torch = _torch()
        latent = cast("Mapping[object, object]", samples).get("samples")
        if type(latent) is inference.MultiStreamLatent:
            streams = cast("Any", latent)
            if streams.roles != ("audio",):
                raise TypeError("samples['samples'] must contain exactly one audio stream")
            latent = streams.by_role("audio")
        if type(latent) is not torch.Tensor:
            raise TypeError("samples['samples'] must be an exact torch.Tensor")
        tensor = cast("Any", latent)
        if (
            tensor.layout is not torch.strided
            or not tensor.is_floating_point()
            or tensor.ndim != 4
            or tensor.shape[0] <= 0
            or tensor.shape[1] != 8
            or tensor.shape[2] <= 0
            or tensor.shape[3] != 16
        ):
            raise ValueError("samples['samples'] must be nonempty floating [batch,8,time,16]")
        component, load_device, stage = _ltxav_audio_codec(audio_vae)
        inference_torch = importlib.import_module("dinkster_inference_torch")
        runtime = inference_torch.LTXAVAudioCodecRuntime(component)
        with stage():
            with torch.inference_mode():
                audio = runtime.decode_audio_latent(tensor.to(load_device))
        if type(audio) is not inference.AudioPreview:
            raise TypeError("LTX-2 audio decode must return AudioPreview")
        waveform = audio.waveform
        if (
            type(waveform) is not torch.Tensor
            or waveform.layout is not torch.strided
            or not waveform.is_floating_point()
            or waveform.ndim != 3
            or waveform.shape[0] != tensor.shape[0]
            or waveform.shape[1] not in (1, 2)
            or waveform.shape[2] <= 0
        ):
            raise TypeError("LTX-2 audio decode must return nonempty floating [batch,1|2,samples]")
        return cls.outputs(
            audio={"waveform": waveform.detach().to("cpu"), "sample_rate": audio.sample_rate}
        )


class GenerationLoadLora(NativeLoadLora):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_lora")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        clip: object,
        lora: object,
        strength_model: float,
        strength_clip: float,
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        _require_provider_runtime(model, "model")
        return NativeLoadLora.execute(
            model=model,
            clip=clip,
            lora=lora,
            strength_model=strength_model,
            strength_clip=strength_clip,
            execution_mode=_generation_lora_mode(execution_mode),
        )


class GenerationLoadLoraModelOnly(NativeLoadLoraModelOnly):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_lora_model_only")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        lora: object,
        strength_model: float,
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        _require_provider_runtime(model, "model")
        return NativeLoadLoraModelOnly.execute(
            model=model,
            lora=lora,
            strength_model=strength_model,
            execution_mode=_generation_lora_mode(execution_mode),
        )


def _ordered_lora_members(
    loras: Mapping[str, object], fields: frozenset[str]
) -> tuple[Mapping[str, object], ...]:
    members: dict[str, dict[str, object]] = {}
    for input_id, value in loras.items():
        member, separator, field = input_id.partition(".")
        if not separator or field not in fields:
            raise ValueError(f"invalid LoRA stack input {input_id!r}")
        members.setdefault(member, {})[field] = value
    for member, values in members.items():
        if "lora" not in values:
            raise ValueError(f"LoRA stack member {member!r} is missing its lora asset")
    return tuple(members.values())


def _model_clip_lora_entries(
    loras: Mapping[str, object],
) -> tuple[tuple[object, float, float], ...]:
    members = _ordered_lora_members(loras, frozenset(("lora", "strength_model", "strength_clip")))
    return tuple(
        (
            member["lora"],
            cast("float", member.get("strength_model", 1.0)),
            cast("float", member.get("strength_clip", 1.0)),
        )
        for member in members
    )


def _model_lora_entries(loras: Mapping[str, object]) -> tuple[tuple[object, float], ...]:
    members = _ordered_lora_members(loras, frozenset(("lora", "strength_model")))
    return tuple(
        (member["lora"], cast("float", member.get("strength_model", 1.0))) for member in members
    )


class GenerationLoadCheckpointStack(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.load_checkpoint_stack")

    @classmethod
    def execute(
        cls,
        *,
        checkpoint: object,
        loras: Mapping[str, object],
        stop_at_clip_layer: int = -1,
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        loaded = GenerationLoadCheckpoint.execute(checkpoint=checkpoint)
        model, clip = _apply_native_lora_stack(
            loaded["model"],
            loaded["clip"],
            _model_clip_lora_entries(loras),
            _generation_lora_mode(execution_mode),
        )
        vae = _NativeCodecHandle(_native_handle(clip, "clip"))
        if stop_at_clip_layer != -1:
            clip = GenerationClipSetLastLayer.execute(
                clip=clip,
                stop_at_clip_layer=stop_at_clip_layer,
            )["clip"]
        return cls.outputs(
            model=model,
            clip=clip,
            vae=vae,
        )


class GenerationApplyLoraStack(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.apply_lora_stack")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        clip: object,
        loras: Mapping[str, object],
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        _require_provider_runtime(model, "model")
        model, clip = _apply_native_lora_stack(
            model,
            clip,
            _model_clip_lora_entries(loras),
            _generation_lora_mode(execution_mode),
        )
        return cls.outputs(model=model, clip=clip)


class GenerationApplyLoraStackModelOnly(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.apply_lora_stack_model_only")

    @classmethod
    def execute(
        cls,
        *,
        model: object,
        loras: Mapping[str, object],
        execution_mode: str = "auto",
    ) -> Mapping[str, object]:
        _require_provider_runtime(model, "model")
        return cls.outputs(
            model=_apply_native_model_lora_stack(
                model,
                _model_lora_entries(loras),
                _generation_lora_mode(execution_mode),
            )
        )


def _generation_input_int(
    inputs: Mapping[str, object],
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    value = inputs.get(name)
    if type(value) is not int:
        raise TypeError(f"{name.rsplit('.', 1)[-1]} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name.rsplit('.', 1)[-1]} must be in [{minimum}, {maximum}], got {value}"
        )
    return value


def _generation_input_float(
    inputs: Mapping[str, object],
    name: str,
    minimum: float,
    maximum: float,
    *,
    inclusive_minimum: bool = True,
) -> float:
    value = inputs.get(name)
    if type(value) is not float:
        raise TypeError(f"{name.rsplit('.', 1)[-1]} must be a float")
    lower_valid = value >= minimum if inclusive_minimum else value > minimum
    if not math.isfinite(value) or not lower_valid or value > maximum:
        opening = "[" if inclusive_minimum else "("
        raise ValueError(
            f"{name.rsplit('.', 1)[-1]} must be in {opening}{minimum}, {maximum}], got {value}"
        )
    return value


def _generation_sampler(inputs: Mapping[str, object], inference: Any) -> tuple[Any, int | None]:
    mode = inputs.get("sampling_mode")
    if mode == "off":
        return (
            inference.GenerationSamplerChain(
                (
                    inference.GenerationSamplerStage(
                        inference.GenerationSamplerKind.GREEDY,
                    ),
                )
            ),
            None,
        )
    if mode != "on":
        raise ValueError(f"unknown text generation sampling mode: {mode!r}")

    temperature = _generation_input_float(
        inputs, "sampling_mode.temperature", 0.0, 2.0, inclusive_minimum=False
    )
    top_k = _generation_input_int(inputs, "sampling_mode.top_k", 0, 1_000)
    top_p = _generation_input_float(inputs, "sampling_mode.top_p", 0.0, 1.0)
    min_p = _generation_input_float(inputs, "sampling_mode.min_p", 0.0, 1.0)
    repetition = _generation_input_float(
        inputs, "sampling_mode.repetition_penalty", 0.0, 5.0, inclusive_minimum=False
    )
    presence = _generation_input_float(inputs, "sampling_mode.presence_penalty", 0.0, 5.0)
    seed = _generation_input_int(inputs, "sampling_mode.seed", 0, 2**64 - 1)

    stages: list[Any] = []
    if repetition != 1.0:
        stages.append(
            inference.GenerationSamplerStage(
                inference.GenerationSamplerKind.REPETITION_PENALTY,
                repetition,
            )
        )
    if presence != 0.0:
        stages.append(
            inference.GenerationSamplerStage(
                inference.GenerationSamplerKind.PRESENCE_PENALTY,
                presence,
            )
        )
    if temperature != 1.0:
        stages.append(
            inference.GenerationSamplerStage(
                inference.GenerationSamplerKind.TEMPERATURE,
                temperature,
            )
        )
    if top_k:
        stages.append(
            inference.GenerationSamplerStage(
                inference.GenerationSamplerKind.TOP_K,
                top_k,
            )
        )
    if min_p:
        stages.append(
            inference.GenerationSamplerStage(
                inference.GenerationSamplerKind.MIN_P,
                min_p,
            )
        )
    if top_p != 1.0:
        stages.append(
            inference.GenerationSamplerStage(
                inference.GenerationSamplerKind.TOP_P,
                top_p,
            )
        )
    stages.append(
        inference.GenerationSamplerStage(
            inference.GenerationSamplerKind.MULTINOMIAL,
        )
    )
    return inference.GenerationSamplerChain(tuple(stages)), seed


def _validate_text_generation_options(inputs: Mapping[str, object]) -> None:
    for name in ("image", "video", "audio"):
        if inputs.get(name) is not None:
            raise ValueError(f"native Qwen generation does not support {name} input")
    thinking = inputs.get("thinking", False)
    if type(thinking) is not bool:
        raise TypeError("thinking must be a boolean")
    if thinking:
        raise ValueError("native Qwen generation does not support thinking mode")
    use_default_template = inputs.get("use_default_template", False)
    if type(use_default_template) is not bool:
        raise TypeError("use_default_template must be a boolean")
    if use_default_template:
        raise ValueError(
            "native Qwen generation does not support model templates; "
            "set use_default_template to false"
        )


def _run_qwen_text_generation(inputs: Mapping[str, object], prompt: str) -> str:
    if type(prompt) is not str:
        raise TypeError("prompt must be a string")
    _validate_text_generation_options(inputs)
    max_length = _generation_input_int(inputs, "max_length", 1, 32_768)
    inference = importlib.import_module("dinkster_inference")
    sampler, seed = _generation_sampler(inputs, inference)
    handle = load_registered_component(
        inputs.get("clip"),
        "clip",
        "qwen3_06b",
        family_id=inference.ANIMA_CONFIG.family_id,
    )
    context = current_execution_context()
    cancelled = _not_cancelled if context is None else context.cancelled

    with handle.stage():
        inference_torch = importlib.import_module("dinkster_inference_torch")
        provider = inference_torch.QwenGenerationProvider(
            handle.component,
            inference.load_qwen_bpe(),
            handle.resource_identity,
        )
        request = inference.GenerationRequest(
            provider.id,
            handle.resource_identity,
            prompt=prompt,
            sampler=sampler,
            stop=inference.GenerationStopConditions(max_length),
            seed=seed,
        )
        terminal = None
        with provider.generate(request, cancelled=cancelled) as stream:
            for event in stream:
                if isinstance(event, inference.GenerationTerminalEvent):
                    if terminal is not None:
                        raise RuntimeError("generation stream produced multiple terminal events")
                    terminal = event
                elif terminal is not None:
                    raise RuntimeError(
                        "generation stream produced an event after its terminal event"
                    )
                elif not isinstance(event, inference.GenerationTokenEvent):
                    raise RuntimeError("generation stream produced an unknown event")
    if terminal is None:
        raise RuntimeError("generation stream ended without a terminal event")
    if terminal.result.finish_reason is inference.GenerationFinishReason.CANCELLED:
        raise inference.SamplingCancelled("text generation cancelled")
    return terminal.result.text


class GenerationTextGenerate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.text_generate")

    @classmethod
    def check_lazy_status(
        cls,
        *,
        clip: object | None = None,
        provider: object | None = None,
        **_inputs: object,
    ) -> tuple[str, ...]:
        return ("clip",) if provider is None and clip is None else ()

    @classmethod
    def execute(cls, **inputs: object) -> Mapping[str, object]:
        prompt = inputs.get("prompt")
        if type(prompt) is not str:
            raise TypeError("prompt must be a string")
        return cls.outputs(generated_text=_run_qwen_text_generation(inputs, prompt))


class GenerationPromptEnhance(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.prompt_enhance")

    @classmethod
    def check_lazy_status(
        cls,
        *,
        clip: object | None = None,
        provider: object | None = None,
        **_inputs: object,
    ) -> tuple[str, ...]:
        return ("clip",) if provider is None and clip is None else ()

    @classmethod
    def execute(cls, **inputs: object) -> Mapping[str, object]:
        prompt = inputs.get("prompt")
        if type(prompt) is not str:
            raise TypeError("prompt must be a string")
        text = _run_qwen_text_generation(inputs, prepare_ltx2_prompt(prompt))
        return cls.outputs(generated_text=clean_enhanced_prompt(text, prompt))


class NativeMiniMaxMusic3TextEncode(MiniMaxMusic3TextEncode):
    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        caption: str,
        lyrics: str,
        seed: int,
        max_duration: float,
        cfg_scale: float,
        top_k: int,
    ) -> Mapping[str, object]:
        if type(caption) is not str or type(lyrics) is not str:
            raise TypeError("caption and lyrics must be strings")
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("seed must be an integer in [0, 18446744073709551615]")
        if type(max_duration) not in (int, float) or not 0.04 <= max_duration <= 360.0:
            raise ValueError("max_duration must be in [0.04, 360.0]")
        if type(cfg_scale) not in (int, float) or not 0.0 <= cfg_scale <= 100.0:
            raise ValueError("cfg_scale must be in [0.0, 100.0]")
        if type(top_k) is not int or not 1 <= top_k <= 16384:
            raise ValueError("top_k must be an integer in [1, 16384]")
        inference = importlib.import_module("dinkster_inference")
        handle = load_registered_component(
            clip,
            "clip",
            "text",
            family_id=inference.MINIMAX_MUSIC3_CONFIG.family_id,
        )
        tokenizer = getattr(handle.component, "_dinkster_minimax_music3_tokenizer", None)
        inference_torch = importlib.import_module("dinkster_inference_torch")
        max_audio_frames = min(
            inference.MAX_AUDIO_FRAMES,
            max(1, round(max_duration * inference.AUDIO_FRAMES_PER_SECOND)),
        )
        recipe = handle.recipe
        assert recipe is not None
        runtime = inference_torch.MiniMaxMusic3TextRuntime(
            handle.component,
            tokenizer,
            compute_dtype=_torch_dtype(_torch(), recipe.knobs.text_dtype),
        )
        torch = _torch()
        with handle.stage():
            with torch.inference_mode():
                conditioning = runtime.encode_text(
                    caption,
                    lyrics,
                    seed=seed,
                    max_audio_frames=max_audio_frames,
                    cfg_scale=cfg_scale,
                    top_k=top_k,
                )
        carrier = inference_torch.minimax_music3_conditioning_to_carrier(conditioning)
        return cls.outputs(
            conditioning=inference.bind_component_conditioning(
                carrier,
                inference.ComponentBinding(
                    "text",
                    inference.MINIMAX_MUSIC3_CONFIG.family_id,
                    handle.resource_identity,
                ),
            ),
            seconds=conditioning.embeddings.shape[1] / inference.AUDIO_FRAMES_PER_SECOND,
        )


class GenerationClipTextEncode(NativeClipTextEncode):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.clip_text_encode")

    @classmethod
    def execute(cls, *, text: str, clip: object) -> Mapping[str, object]:
        options = _native_clip_options(clip)
        clip = options.source
        if type(clip) is _LTXAVTextHandle:
            direct_clip = cast("Any", clip)
            inference = importlib.import_module("dinkster_inference")
            torch = _torch()
            with direct_clip.stage():
                with torch.inference_mode():
                    carrier = direct_clip.encode_text(text)
            return cls.outputs(
                conditioning=inference.bind_component_conditioning(
                    carrier,
                    inference.ComponentBinding(
                        direct_clip.role,
                        direct_clip.family_id,
                        direct_clip.resource_identity,
                    ),
                )
            )
        if isinstance(clip, NativeComponentHandle):
            inference = importlib.import_module("dinkster_inference")
            recipe = clip.recipe
            text_runtime = getattr(clip, "runtime", None)
            encode = getattr(text_runtime, "encode_text", None)
            to_carrier = getattr(text_runtime, "text_conditioning_carrier", None)
            if callable(encode) and callable(to_carrier):
                if recipe is None:
                    raise RuntimeError("native text encoding runtime has no retained recipe")
                torch = _torch()
                with clip.stage(observer_stage="condition"), torch.inference_mode():
                    conditioning = encode(
                        text,
                        hidden_layer=options.hidden_layer,
                        min_padding=options.t5_min_padding,
                        min_length=options.t5_min_length,
                    )
                    carrier = to_carrier(conditioning)
                return cls.outputs(
                    conditioning=inference.bind_component_conditioning(
                        carrier,
                        inference.ComponentBinding(
                            "text", recipe.family_id, clip.resource_identity
                        ),
                    )
                )
            registered = importlib.import_module("dinkster_native.family_registry")
            encode_text = registered.registered_callable(clip, "native_encode_text")
            return cls.outputs(conditioning=encode_text(clip, text, options))
        handle = _require_provider_runtime(clip, "clip")
        runtime = handle.runtime
        encode_text = getattr(runtime, "encode_text", None)
        if not callable(encode_text):
            raise TypeError("clip runtime does not expose text encoding")
        torch = _torch()
        with handle.stage("text"):
            with torch.inference_mode():
                if (
                    options.hidden_layer is None
                    and options.t5_min_padding is None
                    and options.t5_min_length is None
                ):
                    conditioning = encode_text(text)
                else:
                    inference_torch = importlib.import_module("dinkster_inference_torch")
                    if isinstance(runtime, inference_torch.FluxRuntime):
                        conditioning = encode_text(
                            text,
                            hidden_layer=options.hidden_layer,
                            min_padding=options.t5_min_padding,
                            min_length=options.t5_min_length,
                        )
                    elif isinstance(runtime, inference_torch.SDRuntime):
                        conditioning = encode_text(text, hidden_layer=options.hidden_layer)
                    elif isinstance(runtime, inference_torch.Wan21Runtime):
                        conditioning = encode_text(
                            text,
                            min_padding=options.t5_min_padding,
                            min_length=options.t5_min_length,
                        )
                    else:
                        conditioning = encode_text(text)
        to_carrier = getattr(runtime, "text_conditioning_carrier", None)
        if callable(to_carrier):
            return cls.outputs(conditioning=to_carrier(conditioning))
        inference_torch = importlib.import_module("dinkster_inference_torch")
        return cls.outputs(conditioning=inference_torch.basic_conditioning_to_carrier(conditioning))


class GenerationClipTextEncodeLumina2(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.clip_text_encode_lumina2")

    @classmethod
    def execute(cls, *, system_prompt: str, user_prompt: str, clip: object) -> Mapping[str, object]:
        text = importlib.import_module("dinkster_inference").lumina2_system_prompt(
            user_prompt, system_prompt
        )
        return GenerationClipTextEncode.execute(text=text, clip=clip)


class GenerationModelSamplingAuraFlow(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.model_sampling_aura_flow")

    @classmethod
    def execute(cls, *, model: object, shift: float) -> Mapping[str, object]:
        if type(shift) is not float or not math.isfinite(shift) or shift <= 0.0:
            raise ValueError("shift must be a positive finite float")
        (
            handle,
            overlays,
            resolvers,
            z_image_control,
            _,
            guidance_transforms,
            context_windows,
            chroma_radiance_options,
        ) = _native_model(model, "model")
        inference = importlib.import_module("dinkster_inference")
        space = inference.FlowSigmas(shift=shift, multiplier=1.0, timesteps=1000)
        _sampling_space_runtime(handle.runtime, space)
        return cls.outputs(
            model=_NativeModelOverlay(
                handle,
                overlays,
                resolvers,
                z_image_control,
                None,
                guidance_transforms,
                context_windows,
                chroma_radiance_options,
                sampling_cache=_native_model_sampling_cache(model),
                sampling_timeline=_native_model_sampling_timeline(model),
                sampling_space=space,
            )
        )


class GenerationClipSetLastLayer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.clip_set_last_layer")

    @classmethod
    def execute(cls, *, clip: object, stop_at_clip_layer: int) -> Mapping[str, object]:
        if type(stop_at_clip_layer) is not int or not -24 <= stop_at_clip_layer <= -1:
            raise ValueError(
                f"stop_at_clip_layer must be an integer in [-24, -1], got {stop_at_clip_layer}"
            )
        return cls.outputs(
            clip=replace(_native_clip_options(clip), hidden_layer=stop_at_clip_layer)
        )


class GenerationT5TokenizerOptions(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.t5_tokenizer_options")

    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        min_padding: int,
        min_length: int,
    ) -> Mapping[str, object]:
        for name, value in (("min_padding", min_padding), ("min_length", min_length)):
            if type(value) is not int or not 0 <= value <= 10000:
                raise ValueError(f"{name} must be an integer in [0, 10000], got {value}")
        return cls.outputs(
            clip=replace(
                _native_clip_options(clip),
                t5_min_padding=min_padding,
                t5_min_length=min_length,
            )
        )


class GenerationClipTextEncodeControlnet(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return _generation_provider_schema("dinkster.clip_text_encode_controlnet")

    @classmethod
    def execute(
        cls,
        *,
        clip: object,
        conditioning: object,
        text: str,
    ) -> Mapping[str, object]:
        inference = importlib.import_module("dinkster_inference")
        target = _conditioning_carrier(conditioning, "conditioning")
        control = cast(
            "Any",
            GenerationClipTextEncode.execute(clip=clip, text=text)["conditioning"],
        )
        if not control.conditioning.records:
            raise ValueError("encoded ControlNet text must contain at least one record")
        channels = dict(control.conditioning.records[0].channels)
        text_payload = channels.get(inference.ConditioningChannel.TEXT)
        if text_payload is None:
            raise ValueError("encoded ControlNet text must carry a text payload")
        pooled_payload = channels.get(inference.ConditioningChannel.POOLED)
        records: list[Any] = []
        for record in target.conditioning.records:
            metadata = dict(record.extension_metadata)
            metadata[_CONTROLNET_TEXT_METADATA_KEY] = text_payload.reference
            metadata[_CONTROLNET_POOLED_METADATA_KEY] = (
                None if pooled_payload is None else pooled_payload.reference
            )
            records.append(replace(record, extension_metadata=tuple(metadata.items())))
        return cls.outputs(
            conditioning=_rebound_conditioning_carrier(
                inference,
                records,
                (*target.bindings, *control.bindings),
            )
        )
