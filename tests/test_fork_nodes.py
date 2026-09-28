from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from dinkster_assets import AssetRef, digest_file
from dinkster_compat_comfy.translate import CompatTranslation
from dinkster_inference import MultiStreamLatent, ResidentConditioningCarrier
from dinkster_native import fork_nodes
from dinkster_native.native import register_native_types
from dinkster_protocol import Invocation
from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

CONDITIONING = TypeExpr.concrete("dinkster.conditioning")
INT = TypeExpr.concrete("core.int")


class _ConditioningConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.conditioning-consumer",
            inputs=(InputSpec("conditioning", CONDITIONING),),
            outputs=(OutputSpec("count", INT),),
        )

    @classmethod
    def execute(cls, *, conditioning: object) -> Mapping[str, object]:
        values = list(cast("Any", conditioning))
        return cls.outputs(count=len(values))


class _Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


def _asset(path: Path) -> AssetRef:
    return AssetRef(
        digest_file(path),
        path.name,
        path.stat().st_size,
        resolver=_Resolver(path),
    )


class _FakeTensor:
    def __init__(self, data: np.ndarray[Any, Any]) -> None:
        self.data = data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def reshape(self, *shape: int) -> _FakeTensor:
        return _FakeTensor(self.data.reshape(*shape))

    def movedim(self, source: int, destination: int) -> _FakeTensor:
        return _FakeTensor(np.moveaxis(self.data, source, destination))

    def __mul__(self, value: float) -> _FakeTensor:
        return _FakeTensor(self.data * value)

    def __lt__(self, value: float) -> np.ndarray[Any, np.dtype[np.bool_]]:
        return self.data < value

    def __setitem__(self, key: Any, value: float) -> None:
        self.data[key] = value

    def __itruediv__(self, value: object) -> _FakeTensor:
        divisor = value.data if isinstance(value, _FakeTensor) else value
        self.data /= divisor
        return self


class _FakeTorch:
    inference_mode_enabled = False

    class _InferenceMode:
        def __enter__(self) -> None:
            _FakeTorch.inference_mode_enabled = True

        def __exit__(self, *args: object) -> None:
            _FakeTorch.inference_mode_enabled = False

    @staticmethod
    def inference_mode() -> _FakeTorch._InferenceMode:
        return _FakeTorch._InferenceMode()

    @staticmethod
    def zeros(shape: tuple[int, ...], device: object = None) -> _FakeTensor:
        del device
        return _FakeTensor(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def ones(shape: tuple[int, ...]) -> _FakeTensor:
        return _FakeTensor(np.ones(shape, dtype=np.float32))

    @staticmethod
    def count_nonzero(value: _FakeTensor) -> np.integer[Any]:
        return np.count_nonzero(value.data)

    @staticmethod
    def std(value: _FakeTensor, dim: tuple[int, ...], keepdim: bool) -> _FakeTensor:
        return _FakeTensor(np.std(value.data, axis=dim, keepdims=keepdim))

    @staticmethod
    def isfinite(value: _FakeTensor) -> np.ndarray[Any, np.dtype[np.bool_]]:
        return np.isfinite(value.data)


def test_native_types_preserve_resident_conditioning_codec() -> None:
    class ResidentPayload:
        _dinkster_resident_fingerprint = "test-resident-conditioning"
        _dinkster_input_value = "native-conditioning"

    registry = TypeRegistry()
    register_native_types(registry)
    carrier = ResidentConditioningCarrier(ResidentPayload())

    spec = registry.spec("dinkster.conditioning")
    assert spec.decode(spec.encode(carrier)) is carrier
    assert registry.input_object("dinkster.conditioning", carrier) == "native-conditioning"


def test_worker_unwraps_resident_conditioning_for_node_consumers() -> None:
    class ResidentPayload:
        _dinkster_resident_fingerprint = "test-worker-conditioning"
        _dinkster_input_value = (("embedding", {"pooled_output": "pooled"}),)

    registry = TypeRegistry()
    register_core_types(registry)
    register_native_types(registry)
    value = registry.wrap(
        "dinkster.conditioning",
        ResidentConditioningCarrier(ResidentPayload()),
    )
    schema = _ConditioningConsumer.schema()
    worker = InProcessWorker({schema.node_type: _ConditioningConsumer}, registry)
    result = asyncio.run(
        worker.invoke(
            Invocation(
                "invocation",
                "basic-guider",
                schema.node_type,
                {"conditioning": value},
                schema,
            )
        )
    )

    assert result.error is None, result.error
    assert result.outputs is not None
    assert result.outputs["count"].resolve() == 1


def test_compat_sampler_boundary_values_remain_process_resident() -> None:
    type_ids = ("comfy.GUIDER", "comfy.NOISE", "comfy.SAMPLER", "comfy.SIGMAS")
    translation = CompatTranslation()
    translation.opaque_types.update(type_ids)
    registry = TypeRegistry()
    translation.register_types(registry)

    for type_id in type_ids:
        value = object()
        spec = registry.spec(type_id)
        assert spec.decode(spec.encode(value)) is value


def test_fork_loaders_call_dinkster_comfy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"weights")
    asset = _asset(path)
    calls: list[tuple[object, ...]] = []
    model, clip, vae = object(), object(), object()

    class LoadedVAE:
        def __init__(self, *, sd: object, metadata: object) -> None:
            calls.append(("vae", sd, metadata))

        def throw_exception_if_invalid(self) -> None:
            calls.append(("validate-vae",))

    sd = SimpleNamespace(
        CLIPType=SimpleNamespace(H3="h3"),
        VAE=LoadedVAE,
        load_checkpoint_guess_config=lambda *args, **kwargs: (
            calls.append(("checkpoint", args, kwargs)) or (model, clip, vae, None)
        ),
        load_diffusion_model=lambda *args, **kwargs: (
            calls.append(("diffusion", args, kwargs)) or model
        ),
        load_clip=lambda *args, **kwargs: calls.append(("clip", args, kwargs)) or clip,
    )
    utils = SimpleNamespace(
        load_torch_file=lambda *args, **kwargs: (
            calls.append(("torch-file", args, kwargs)) or ({"weight": 1}, {"format": "test"})
        )
    )
    real_import = importlib.import_module
    modules = {
        "dinkster_comfy.sd": sd,
        "dinkster_comfy.utils": utils,
        "torch": _FakeTorch,
    }
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: modules.get(name) or real_import(name),
    )

    assert fork_nodes.GenerationLoadCheckpoint.execute(checkpoint=asset) == {
        "model": model,
        "clip": clip,
        "vae": vae,
    }
    assert fork_nodes.GenerationLoadDiffusionModel.execute(
        diffusion_model=asset, weight_dtype="default"
    ) == {"model": model}
    assert fork_nodes.NativeLoadClip.execute(text_encoder=asset, type="h3") == {"clip": clip}
    loaded = fork_nodes.NativeLoadVae.execute(vae=asset)["vae"]

    assert isinstance(loaded, LoadedVAE)
    assert [call[0] for call in calls] == [
        "checkpoint",
        "diffusion",
        "clip",
        "torch-file",
        "vae",
        "validate-vae",
    ]
    assert cast("dict[str, object]", calls[1][2])["model_options"] == {}
    assert cast("dict[str, object]", calls[2][2])["clip_type"] == "h3"


def test_fork_sd15_adapters_preserve_conditioning_latent_and_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: _FakeTorch if name == "torch" else real_import(name),
    )

    class Clip:
        def tokenize(self, text: str) -> tuple[str, str]:
            return ("tokens", text)

        def encode_from_tokens_scheduled(self, tokens: object) -> tuple[str, object]:
            return ("conditioning", tokens)

    clip = Clip()
    encoded = fork_nodes.GenerationClipTextEncode.execute(text="hello", clip=clip)["conditioning"]
    assert isinstance(encoded, ResidentConditioningCarrier)
    assert cast("Any", encoded._dinkster_resident_payload).conditioning == (
        "conditioning",
        ("tokens", "hello"),
    )
    registry = TypeRegistry()
    register_native_types(registry)
    assert registry.input_object("dinkster.conditioning", encoded) == (
        "conditioning",
        ("tokens", "hello"),
    )

    latent = fork_nodes.GenerationEmptyLatentImage.execute(width=32, height=24, batch_size=2)[
        "latent"
    ]
    samples = cast("Mapping[str, Any]", latent)["samples"]
    assert tuple(samples.shape) == (2, 4, 3, 4)
    assert _FakeTorch.count_nonzero(samples).item() == 0

    class VAE:
        def decode(self, value: object) -> object:
            assert value is samples
            assert _FakeTorch.inference_mode_enabled
            return _FakeTorch.zeros((1, 2, 3, 4, 5))

    image = fork_nodes.GenerationVAEDecode.execute(samples=latent, vae=VAE())["image"]
    assert tuple(cast("Any", image).shape) == (2, 3, 4, 5)
    assert not _FakeTorch.inference_mode_enabled


def test_fork_h3_adapters_preserve_stream_roles_and_media(monkeypatch: pytest.MonkeyPatch) -> None:
    model_management = SimpleNamespace(intermediate_device=lambda: "cpu")
    real_import = importlib.import_module

    def import_module(name: str) -> object:
        if name == "dinkster_comfy.model_management":
            return model_management
        if name == "torch":
            return _FakeTorch
        return real_import(name)

    monkeypatch.setattr(
        importlib,
        "import_module",
        import_module,
    )

    target = fork_nodes.NativeEmptyMiniMaxH3AV.execute(width=64, height=48, frame_count=6)["latent"]
    streams = cast("Mapping[str, MultiStreamLatent[Any]]", target)["samples"]
    assert streams.roles == ("video", "audio")
    assert tuple(streams.by_role("video").shape) == (1, 24, 7, 3, 4)
    assert tuple(streams.by_role("audio").shape) == (1, 32, 2, 37)

    class Clip:
        def tokenize(self, text: str, *, images: list[object]) -> tuple[object, ...]:
            return (text, images)

        def encode_from_tokens_scheduled(self, tokens: object) -> tuple[str, object]:
            return ("h3", tokens)

    conditioning = fork_nodes.NativeMiniMaxH3T2VAConditioning.execute(
        clip=Clip(), target=target, prompt="motion"
    )["conditioning"]
    assert isinstance(conditioning, ResidentConditioningCarrier)
    assert cast("Any", conditioning._dinkster_resident_payload).conditioning == (
        "h3",
        ("motion", []),
    )

    image_to_video = fork_nodes.NativeMiniMaxH3ImageToVideo.execute(
        clip=Clip(),
        vae=object(),
        prompt="motion",
        width=64,
        height=48,
        length=6,
    )
    assert isinstance(image_to_video["positive"], ResidentConditioningCarrier)
    generated = cast("Mapping[str, MultiStreamLatent[Any]]", image_to_video["latent"])["samples"]
    assert generated.roles == ("video", "audio")

    separated = fork_nodes.NativeSeparateAVLatent.execute(latent=target)
    video = cast("Mapping[str, object]", separated["video_latent"])["samples"]
    audio = cast("Mapping[str, object]", separated["audio_latent"])["samples"]
    assert video is streams.by_role("video")
    assert audio is streams.by_role("audio")

    class VideoVAE:
        def decode(self, value: object) -> object:
            assert value is video
            assert _FakeTorch.inference_mode_enabled
            return _FakeTorch.zeros((1, 2, 3, 4, 5))

    decoded_video = fork_nodes.GenerationVAEDecode.execute(samples=target, vae=VideoVAE())["image"]
    assert tuple(cast("Any", decoded_video).shape) == (2, 3, 4, 5)
    assert not _FakeTorch.inference_mode_enabled

    class AudioVAE:
        audio_sample_rate_output = 24_000

        def decode(self, value: object) -> object:
            assert value is audio
            assert _FakeTorch.inference_mode_enabled
            return _FakeTorch.ones((1, 4, 2)) * 10

    decoded_audio = cast(
        "Mapping[str, object]",
        fork_nodes.NativeVAEDecodeAudio.execute(samples=target, vae=AudioVAE())["audio"],
    )
    assert decoded_audio["sample_rate"] == 24_000
    assert tuple(cast("Any", decoded_audio["waveform"]).shape) == (1, 2, 4)
    assert _FakeTorch.isfinite(cast("_FakeTensor", decoded_audio["waveform"])).all()
    assert not _FakeTorch.inference_mode_enabled


def test_generation_ksampler_preserves_h3_stream_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = MultiStreamLatent.from_pairs((("video", object()), ("audio", object())))
    outputs = (object(), object())

    class NestedTensor:
        def __init__(self, streams: tuple[object, ...]) -> None:
            assert streams == (inputs.by_role("video"), inputs.by_role("audio"))

        def unbind(self) -> tuple[object, ...]:
            return outputs

    sample = SimpleNamespace(
        fix_empty_latent_channels=lambda *args: args[1],
        prepare_noise=lambda *args: object(),
        sample=lambda *args, **kwargs: NestedTensor(
            (inputs.by_role("video"), inputs.by_role("audio"))
        ),
    )
    modules = {
        "torch": _FakeTorch,
        "dinkster_comfy.nested_tensor": SimpleNamespace(NestedTensor=NestedTensor),
        "dinkster_comfy.sample": sample,
        "dinkster_comfy.model_management": SimpleNamespace(
            unload_model_and_clones=lambda model: None
        ),
    }
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: modules.get(name) or real_import(name),
    )

    result = fork_nodes.GenerationKSampler.execute(
        model=object(),
        seed=459,
        steps=5,
        cfg=7.0,
        sampler_name="dinkster.euler",
        scheduler="dinkster.normal",
        positive=object(),
        negative=object(),
        latent_image={"samples": inputs},
        denoise=1.0,
    )
    sampled = cast("Mapping[str, MultiStreamLatent[object]]", result["latent"])["samples"]
    assert sampled.roles == ("video", "audio")
    assert sampled.by_role("video") is outputs[0]
    assert sampled.by_role("audio") is outputs[1]


def test_generation_ksampler_normalizes_residency_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    model = object()
    latent = object()
    normalized = object()
    sampled = object()

    def unload_model(actual_model: object) -> None:
        events.append(("unload", actual_model))

    def fix_latent(
        actual_model: object,
        actual_latent: object,
        spatial: object,
        temporal: object,
    ) -> object:
        events.append(("fix", actual_model, actual_latent, spatial, temporal))
        return normalized

    def prepare_noise(actual_latent: object, seed: object, batch_index: object) -> object:
        events.append(("noise", actual_latent, seed, batch_index))
        return object()

    def sample_latent(*args: object, **kwargs: object) -> object:
        assert _FakeTorch.inference_mode_enabled
        events.append(("sample", args, kwargs))
        return sampled

    sample = SimpleNamespace(
        fix_empty_latent_channels=fix_latent,
        prepare_noise=prepare_noise,
        sample=sample_latent,
    )
    model_management = SimpleNamespace(unload_model_and_clones=unload_model)
    modules = {
        "torch": _FakeTorch,
        "dinkster_comfy.sample": sample,
        "dinkster_comfy.model_management": model_management,
    }
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: modules.get(name) or real_import(name),
    )

    result = fork_nodes.GenerationKSampler.execute(
        model=model,
        seed=459,
        steps=5,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        positive=object(),
        negative=object(),
        latent_image={
            "samples": latent,
            "batch_index": (3,),
            "downscale_ratio_spacial": 8,
            "downscale_ratio_temporal": 4,
        },
        denoise=1.0,
    )

    assert [event[0] for event in events] == ["unload", "fix", "noise", "sample"]
    assert events[1] == ("fix", model, latent, 8, 4)
    assert events[2][1:] == (normalized, 459, (3,))
    assert cast("Mapping[str, object]", result["latent"])["samples"] is sampled
    assert not _FakeTorch.inference_mode_enabled


def test_generation_ksampler_attaches_attention_to_a_model_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = object()
    token = object()
    events: list[tuple[object, ...]] = []

    class Model:
        def clone(self) -> Model:
            events.append(("clone", self))
            return clone

        def set_model_optimized_attention(self, function: object) -> None:
            events.append(("attach", self, function))

    model = Model()
    clone = Model()
    runtime = SimpleNamespace(for_model=lambda actual: selected if actual is token else None)
    monkeypatch.setattr(
        fork_nodes,
        "current_execution_context",
        lambda: SimpleNamespace(attention_route_token=token, attention_runtime=runtime),
    )
    modules = {
        "torch": _FakeTorch,
        "dinkster_comfy.sample": SimpleNamespace(
            fix_empty_latent_channels=lambda actual, latent, *_args: (
                events.append(("fix", actual)),
                latent,
            )[1],
            prepare_noise=lambda *_args: object(),
            sample=lambda actual, *_args, **_kwargs: (
                events.append(("sample", actual)),
                object(),
            )[1],
        ),
        "dinkster_comfy.model_management": SimpleNamespace(
            unload_model_and_clones=lambda actual: events.append(("unload", actual))
        ),
    }
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: modules.get(name) or real_import(name),
    )

    fork_nodes.GenerationKSampler.execute(
        model=model,
        seed=459,
        steps=1,
        cfg=1.0,
        sampler_name="euler",
        scheduler="normal",
        positive=object(),
        negative=object(),
        latent_image={"samples": object()},
        denoise=1.0,
    )

    assert events[:2] == [("clone", model), ("attach", clone, selected)]
    assert ("attach", model, selected) not in events
    assert ("unload", clone) in events
    assert ("fix", clone) in events
    assert ("sample", clone) in events
