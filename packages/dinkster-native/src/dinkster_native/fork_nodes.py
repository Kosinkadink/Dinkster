"""Execution nodes backed directly by the Dinkster-owned ComfyUI fork."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from dinkster_assets import AssetRef
from dinkster_nodes_generation.nodes import (
    CLIPTextEncode,
    EmptyLatentImage,
    KSampler,
    LoadCheckpoint,
    LoadDiffusionModel,
    VAEDecode,
)
from dinkster_schema import (
    AssetWidget,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

MODEL = TypeExpr.concrete("dinkster.model")
CLIP = TypeExpr.concrete("dinkster.clip")
VAE = TypeExpr.concrete("dinkster.vae")
LATENT = TypeExpr.concrete("dinkster.latent")
CONDITIONING = TypeExpr.concrete("dinkster.conditioning")
IMAGE = TypeExpr.concrete("dinkster.image")
AUDIO = TypeExpr.concrete("comfy.AUDIO")
ASSET = TypeExpr.concrete("dinkster.asset")
INT = TypeExpr.concrete("core.int")
STRING = TypeExpr.concrete("core.string")
BOOLEAN = TypeExpr.concrete("core.boolean")
COMBO = TypeExpr.concrete("core.combo")


@dataclass(frozen=True, slots=True)
class _ForkConditioning:
    conditioning: object
    owner: object
    fingerprint: str

    @property
    def _dinkster_resident_owner(self) -> object:
        return self.owner

    @property
    def _dinkster_resident_fingerprint(self) -> str:
        return self.fingerprint


def _resident_conditioning(conditioning: object, owner: object) -> object:
    import hashlib

    from dinkster_inference import ResidentConditioningCarrier

    fingerprint = "fork-conditioning:" + hashlib.sha256(str(id(owner)).encode()).hexdigest()
    return ResidentConditioningCarrier(_ForkConditioning(conditioning, owner, fingerprint))


def _unwrap_conditioning(value: object) -> object:
    payload = getattr(value, "_dinkster_resident_payload", None)
    return payload.conditioning if type(payload) is _ForkConditioning else value


def _fork_samples(samples: object) -> tuple[object, tuple[str, ...] | None]:
    from dinkster_inference import MultiStreamLatent

    if type(samples) is not MultiStreamLatent:
        return samples, None
    streams = cast("MultiStreamLatent[Any]", samples)
    nested = importlib.import_module("dinkster_comfy.nested_tensor").NestedTensor(
        tuple(streams.by_role(role) for role in streams.roles)
    )
    return nested, streams.roles


def _dinkster_samples(samples: object, roles: tuple[str, ...] | None) -> object:
    if roles is None:
        return samples
    from dinkster_inference import MultiStreamLatent

    return MultiStreamLatent[Any].from_pairs(
        tuple(zip(roles, cast("Any", samples).unbind(), strict=True))
    )


class GenerationLoadCheckpoint(LoadCheckpoint):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, checkpoint: object
    ) -> Mapping[str, object]:
        if not isinstance(checkpoint, AssetRef):
            raise TypeError("checkpoint must be an AssetRef")
        model, clip, vae, _ = importlib.import_module(
            "dinkster_comfy.sd"
        ).load_checkpoint_guess_config(
            str(checkpoint.local_path()),
            output_vae=True,
            output_clip=True,
            embedding_directory=[],
        )
        return cls.outputs(model=model, clip=clip, vae=vae)


class GenerationLoadDiffusionModel(LoadDiffusionModel):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, diffusion_model: object, weight_dtype: str
    ) -> Mapping[str, object]:
        if not isinstance(diffusion_model, AssetRef):
            raise TypeError("diffusion_model must be an AssetRef")
        torch = cast("Any", importlib.import_module("torch"))
        options: dict[str, object] = {}
        if weight_dtype in ("fp8_e4m3fn", "fp8_e4m3fn_fast"):
            options["dtype"] = torch.float8_e4m3fn
            if weight_dtype.endswith("_fast"):
                options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            options["dtype"] = torch.float8_e5m2
        model = importlib.import_module("dinkster_comfy.sd").load_diffusion_model(
            str(diffusion_model.local_path()), model_options=options
        )
        return cls.outputs(model=model)


class NativeLoadClip(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_clip",
            display_name="Load CLIP",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "text_encoder",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",), kind="model/text-encoder"
                    ),
                ),
                InputSpec(
                    "type",
                    COMBO,
                    default="minimax",
                    widget=ComboWidget(options=("stable_diffusion", "minimax")),
                ),
                InputSpec(
                    "device",
                    COMBO,
                    default="default",
                    widget=ComboWidget(options=("default", "cpu")),
                ),
            ),
            outputs=(OutputSpec("clip", CLIP),),
            aliases=("CLIPLoader",),
        )

    @classmethod
    def execute(
        cls,
        *,
        text_encoder: object,
        type: str,
        device: str = "default",
    ) -> Mapping[str, object]:
        if not isinstance(text_encoder, AssetRef):
            raise TypeError("text_encoder must be an AssetRef")
        sd = importlib.import_module("dinkster_comfy.sd")
        try:
            clip_type = getattr(sd.CLIPType, type.upper())
        except AttributeError:
            raise ValueError(f"unsupported text encoder type: {type}") from None
        options: dict[str, object] = {}
        if device == "cpu":
            torch = cast("Any", importlib.import_module("torch"))
            options["load_device"] = options["offload_device"] = torch.device("cpu")
        clip = sd.load_clip(
            ckpt_paths=[str(text_encoder.local_path())],
            embedding_directory=[],
            clip_type=clip_type,
            model_options=options,
        )
        return cls.outputs(clip=clip)


class NativeLoadVae(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_vae",
            display_name="Load VAE",
            category="model/loaders",
            inputs=(
                InputSpec(
                    "vae",
                    ASSET,
                    widget=AssetWidget(accept=("application/octet-stream",), kind="model/vae"),
                ),
                InputSpec("pixel_space", BOOLEAN, required=False, default=False, advanced=True),
            ),
            outputs=(OutputSpec("vae", VAE),),
            aliases=("VAELoader",),
        )

    @classmethod
    def execute(cls, *, vae: object, pixel_space: bool = False) -> Mapping[str, object]:
        if pixel_space:
            raise ValueError("pixel-space codecs are not supported")
        if not isinstance(vae, AssetRef):
            raise TypeError("vae must be an AssetRef")
        state, metadata = importlib.import_module("dinkster_comfy.utils").load_torch_file(
            str(vae.local_path()), return_metadata=True
        )
        loaded = importlib.import_module("dinkster_comfy.sd").VAE(sd=state, metadata=metadata)
        loaded.throw_exception_if_invalid()
        return cls.outputs(vae=loaded)


class GenerationClipTextEncode(CLIPTextEncode):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, text: str, clip: object
    ) -> Mapping[str, object]:
        direct = cast("Any", clip)
        conditioning = direct.encode_from_tokens_scheduled(direct.tokenize(text))
        return cls.outputs(conditioning=_resident_conditioning(conditioning, clip))


class GenerationEmptyLatentImage(EmptyLatentImage):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, width: int, height: int, batch_size: int
    ) -> Mapping[str, object]:
        if width % 8 or height % 8:
            raise ValueError("width and height must be divisible by 8")
        torch = cast("Any", importlib.import_module("torch"))
        return cls.outputs(
            latent={"samples": torch.zeros((batch_size, 4, height // 8, width // 8))}
        )


class GenerationKSampler(KSampler):
    MAX_SEED = 0xFFFFFFFFFFFFFFFF
    MAX_STEPS = 10_000
    MAX_CFG = 100.0

    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        *,
        model: object,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: object,
        negative: object,
        latent_image: object,
        denoise: float,
        conditioning_batching: object = "auto",
        max_fused_lanes: int = 2,
        segment: object = None,
    ) -> Mapping[str, object]:
        if conditioning_batching != "auto" or max_fused_lanes != 2:
            raise ValueError("fork sampling only supports default conditioning batching")
        if segment is not None:
            raise ValueError("segmented sampling is not supported")
        if not isinstance(latent_image, Mapping):
            raise TypeError("latent_image must be a mapping")
        sample = cast("Any", importlib.import_module("dinkster_comfy.sample"))
        model_management = cast(
            "Any", importlib.import_module("dinkster_comfy.model_management")
        )
        source = dict(cast("Mapping[object, object]", latent_image))
        latent, roles = _fork_samples(source["samples"])
        model_management.unload_model_and_clones(model)
        latent = sample.fix_empty_latent_channels(
            model,
            latent,
            source.get("downscale_ratio_spacial"),
            source.get("downscale_ratio_temporal"),
        )
        noise = sample.prepare_noise(latent, seed, source.get("batch_index"))
        output = sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name.removeprefix("dinkster."),
            scheduler.removeprefix("dinkster."),
            _unwrap_conditioning(positive),
            _unwrap_conditioning(negative),
            latent,
            denoise=denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            noise_mask=source.get("noise_mask"),
            callback=None,
            disable_pbar=True,
            seed=seed,
        )
        source["samples"] = _dinkster_samples(output, roles)
        return cls.outputs(latent=source)


class GenerationVAEDecode(VAEDecode):
    @classmethod
    def execute(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, *, samples: object, vae: object
    ) -> Mapping[str, object]:
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a latent mapping")
        latent = cast("Mapping[object, object]", samples)["samples"]
        from dinkster_inference import MultiStreamLatent

        if type(latent) is MultiStreamLatent:
            latent = cast("MultiStreamLatent[Any]", latent).by_role("video")
        image = cast("Any", vae).decode(latent)
        if len(image.shape) == 5:
            image = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])
        return cls.outputs(image=image)


def _h3_shape(width: int, height: int, frame_count: int) -> object:
    from dinkster_inference import MultiStreamLatent

    torch = cast("Any", importlib.import_module("torch"))
    while frame_count % 17 != 5:
        frame_count += 1
    video_frames = 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2
    audio_frames = round(frame_count / 24 * 40)
    device = importlib.import_module("dinkster_comfy.model_management").intermediate_device()
    return MultiStreamLatent[Any].from_pairs(
        (
            (
                "video",
                torch.zeros((1, 24, video_frames, height // 16, width // 16), device=device),
            ),
            ("audio", torch.zeros((1, 32, 2, audio_frames), device=device)),
        )
    )


class NativeEmptyMiniMaxH3AV(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_minimax_h3_av",
            display_name="Empty MiniMax H3 AV Latent",
            category="minimax h3",
            inputs=(
                InputSpec(
                    "width",
                    INT,
                    default=1344,
                    widget=NumberWidget(min=32, max=16384, step=32),
                ),
                InputSpec(
                    "height",
                    INT,
                    default=768,
                    widget=NumberWidget(min=32, max=16384, step=32),
                ),
                InputSpec(
                    "frame_count",
                    INT,
                    default=124,
                    widget=NumberWidget(min=5, max=3600, step=17),
                ),
            ),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("EmptyMiniMaxH3LatentAV",),
            dispatch_affinity="native",
        )

    @classmethod
    def execute(cls, *, width: int, height: int, frame_count: int) -> Mapping[str, object]:
        return cls.outputs(latent={"samples": _h3_shape(width, height, frame_count)})


class NativeMiniMaxH3T2VAConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.minimax_h3_t2va_conditioning",
            display_name="MiniMax H3 T2VA Conditioning",
            category="minimax h3/conditioning",
            inputs=(
                InputSpec("clip", CLIP),
                InputSpec("target", LATENT),
                InputSpec("prompt", STRING, widget=StringWidget(multiline=True)),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            dispatch_affinity="native",
        )

    @classmethod
    def execute(cls, *, clip: object, target: object, prompt: str) -> Mapping[str, object]:
        del target
        direct = cast("Any", clip)
        conditioning = direct.encode_from_tokens_scheduled(direct.tokenize(prompt, images=[]))
        return cls.outputs(conditioning=_resident_conditioning(conditioning, clip))


class NativeMiniMaxH3ImageToVideo(Node):
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
        del vae
        if first_frame is not None or last_frame is not None:
            raise ValueError("keyframed H3 conditioning is not supported")
        latent = {"samples": _h3_shape(width, height, length)}
        conditioned = NativeMiniMaxH3T2VAConditioning.execute(
            clip=clip, target=latent, prompt=prompt
        )
        return {"positive": conditioned["conditioning"], "latent": latent}


class NativeSeparateAVLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.separate_av_latent",
            display_name="Separate Audio/Video Latent",
            category="latent/multi-stream",
            inputs=(InputSpec("latent", LATENT),),
            outputs=(OutputSpec("video_latent", LATENT), OutputSpec("audio_latent", LATENT)),
        )

    @classmethod
    def execute(cls, *, latent: object) -> Mapping[str, object]:
        if not isinstance(latent, Mapping):
            raise TypeError("latent must be a mapping")
        source = dict(cast("Mapping[object, object]", latent))
        streams = cast("Any", source["samples"])
        video = dict(source)
        audio = dict(source)
        video["samples"] = streams.by_role("video")
        audio["samples"] = streams.by_role("audio")
        return cls.outputs(video_latent=video, audio_latent=audio)


class NativeVAEDecodeAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.vae_decode_audio",
            display_name="VAE Decode Audio",
            category="model/latent",
            inputs=(InputSpec("samples", LATENT), InputSpec("vae", VAE)),
            outputs=(OutputSpec("audio", AUDIO),),
            aliases=("VAEDecodeAudio",),
            dispatch_affinity="native",
        )

    @classmethod
    def execute(cls, *, samples: object, vae: object) -> Mapping[str, object]:
        if not isinstance(samples, Mapping):
            raise TypeError("samples must be a latent mapping")
        latent = cast("Mapping[object, object]", samples)["samples"]
        from dinkster_inference import MultiStreamLatent

        if type(latent) is MultiStreamLatent:
            latent = cast("MultiStreamLatent[Any]", latent).by_role("audio")
        audio = cast("Any", vae).decode(latent).movedim(-1, 1)
        torch = cast("Any", importlib.import_module("torch"))
        std = torch.std(audio, dim=(1, 2), keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio /= std
        sample_rate = getattr(
            vae,
            "audio_sample_rate_output",
            getattr(vae, "audio_sample_rate", 44100),
        )
        return cls.outputs(audio={"waveform": audio, "sample_rate": sample_rate})


FORK_NODES: tuple[type[Node], ...] = (
    GenerationLoadCheckpoint,
    GenerationLoadDiffusionModel,
    NativeLoadClip,
    NativeLoadVae,
    GenerationClipTextEncode,
    GenerationEmptyLatentImage,
    GenerationKSampler,
    GenerationVAEDecode,
    NativeEmptyMiniMaxH3AV,
    NativeMiniMaxH3T2VAConditioning,
    NativeSeparateAVLatent,
    NativeVAEDecodeAudio,
)
