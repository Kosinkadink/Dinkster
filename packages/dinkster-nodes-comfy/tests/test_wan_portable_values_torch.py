"""CPU contracts for portable Wan camera, track, and audio encoder values."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from dinkster_api.v1 import (
    AUDIO_ENCODER_OUTPUT_TYPE,
    CORE_BOOLEAN,
    CORE_STRING,
    TRACKS_TYPE,
    WAN_CAMERA_EMBEDDING_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    materialize_latent,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link

from dinkster.compose import PackSpec, ServingComposer

_CAMERA = TypeExpr.concrete(WAN_CAMERA_EMBEDDING_TYPE)
_TRACKS = TypeExpr.concrete(TRACKS_TYPE)
_AUDIO_ENCODER_OUTPUT = TypeExpr.concrete(AUDIO_ENCODER_OUTPUT_TYPE)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return bytes(value.contiguous().reshape(-1).view(torch.uint8).tolist())


def _assert_tensor(actual: object, expected: torch.Tensor) -> None:
    assert type(actual) is torch.Tensor
    assert actual.dtype is expected.dtype
    assert actual.shape == expected.shape
    assert actual.device.type == "cpu"
    assert _tensor_bytes(actual) == _tensor_bytes(expected)


def _camera() -> torch.Tensor:
    return torch.arange(30, dtype=torch.float16).reshape(3, 2, 5).transpose(0, 2)


def _tracks(include_visibility: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "track_path": torch.arange(48, dtype=torch.bfloat16).reshape(2, 3, 4, 2).transpose(1, 2),
        "name": "non-square tracks",
        "frame_indices": (3, 7),
        "metadata": {"enabled": True, "none": None},
    }
    if include_visibility:
        result["track_visibility"] = torch.tensor(
            [[[True, False, True], [False, True, False]]], dtype=torch.bool
        ).transpose(1, 2)
    return result


def _audio(layers_as_tuple: bool) -> dict[str, object]:
    layers: list[torch.Tensor] = [
        torch.arange(30, dtype=torch.float32).reshape(2, 3, 5).transpose(0, 2),
        torch.empty((2, 0, 3), dtype=torch.float16),
    ]
    return {
        "encoded_audio": torch.arange(42, dtype=torch.float32).reshape(2, 3, 7).transpose(0, 2),
        "encoded_audio_all_layers": tuple(layers) if layers_as_tuple else layers,
        "audio_samples": 16001,
        "sample_rate": 16000,
        "metadata": {"source": "cpu", "normalized": False},
    }


def _assert_tracks(actual: object, expected: Mapping[str, object]) -> None:
    assert type(actual) is dict
    assert set(actual) == set(expected)
    _assert_tensor(actual["track_path"], expected["track_path"])
    assert actual["name"] == expected["name"]
    assert actual["frame_indices"] == expected["frame_indices"]
    assert actual["metadata"] == expected["metadata"]
    if "track_visibility" in expected:
        _assert_tensor(actual["track_visibility"], expected["track_visibility"])
        assert actual["track_visibility"].dtype is torch.bool


def _assert_audio(actual: object, expected: Mapping[str, object]) -> None:
    assert type(actual) is dict
    assert set(actual) == set(expected)
    _assert_tensor(actual["encoded_audio"], expected["encoded_audio"])
    actual_layers = actual["encoded_audio_all_layers"]
    expected_layers = expected["encoded_audio_all_layers"]
    assert type(actual_layers) is type(expected_layers)
    assert len(actual_layers) == len(expected_layers)
    for received, source in zip(actual_layers, expected_layers, strict=True):
        _assert_tensor(received, source)
    assert actual["audio_samples"] == expected["audio_samples"]
    assert type(actual["audio_samples"]) is int
    assert actual["sample_rate"] == expected["sample_rate"]
    assert actual["metadata"] == expected["metadata"]


@pytest.mark.parametrize("include_visibility", (False, True))
@pytest.mark.parametrize("layers_as_tuple", (False, True))
def test_wan_portable_value_codecs_preserve_cpu_tensors_and_source_storage(
    include_visibility: bool, layers_as_tuple: bool
) -> None:
    registry = TypeRegistry()
    registered = register_inference_types(registry)
    assert registry.spec(WAN_CAMERA_EMBEDDING_TYPE) in registered
    assert registry.spec(TRACKS_TYPE) in registered
    assert registry.spec(AUDIO_ENCODER_OUTPUT_TYPE) in registered
    assert register_inference_types(registry) == registered

    camera = _camera()
    tracks = _tracks(include_visibility)
    audio = _audio(layers_as_tuple)
    source_storages = (
        camera.untyped_storage().data_ptr(),
        tracks["track_path"].untyped_storage().data_ptr(),
        *((tracks["track_visibility"].untyped_storage().data_ptr(),) if include_visibility else ()),
        audio["encoded_audio"].untyped_storage().data_ptr(),
        *(layer.untyped_storage().data_ptr() for layer in audio["encoded_audio_all_layers"]),
    )

    camera_spec = registry.spec(WAN_CAMERA_EMBEDDING_TYPE)
    tracks_spec = registry.spec(TRACKS_TYPE)
    audio_spec = registry.spec(AUDIO_ENCODER_OUTPUT_TYPE)
    decoded_camera = camera_spec.decode(camera_spec.encode(camera))
    decoded_tracks = tracks_spec.decode(tracks_spec.encode(tracks))
    decoded_audio = audio_spec.decode(audio_spec.encode(audio))
    materialized_camera = materialize_latent(decoded_camera)
    materialized_tracks = materialize_latent(decoded_tracks)
    materialized_audio = materialize_latent(decoded_audio)

    _assert_tensor(materialized_camera, camera)
    _assert_tracks(materialized_tracks, tracks)
    _assert_audio(materialized_audio, audio)
    assert camera_spec.fingerprint is not None
    assert tracks_spec.fingerprint is not None
    assert audio_spec.fingerprint is not None
    assert camera_spec.fingerprint(camera) == camera_spec.fingerprint(decoded_camera)
    assert tracks_spec.fingerprint(tracks) == tracks_spec.fingerprint(decoded_tracks)
    assert audio_spec.fingerprint(audio) == audio_spec.fingerprint(decoded_audio)
    assert materialize_latent(camera) is camera
    assert materialize_latent(tracks) is tracks
    assert materialize_latent(audio) is audio
    assert source_storages == (
        camera.untyped_storage().data_ptr(),
        tracks["track_path"].untyped_storage().data_ptr(),
        *((tracks["track_visibility"].untyped_storage().data_ptr(),) if include_visibility else ()),
        audio["encoded_audio"].untyped_storage().data_ptr(),
        *(layer.untyped_storage().data_ptr() for layer in audio["encoded_audio_all_layers"]),
    )
    _assert_tensor(camera, _camera())
    _assert_tracks(tracks, _tracks(include_visibility))
    _assert_audio(audio, _audio(layers_as_tuple))


def test_audio_encoder_output_rejects_boolean_sample_count() -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    value = _audio(layers_as_tuple=False)
    value["audio_samples"] = True

    with pytest.raises(ValueError, match="audio_samples"):
        registry.spec(AUDIO_ENCODER_OUTPUT_TYPE).encode(value)


def test_existing_wan_type_registration_is_not_replayed() -> None:
    registry = TypeRegistry()
    existing = registry.register(WAN_CAMERA_EMBEDDING_TYPE)

    register_inference_types(registry)

    assert registry.spec(WAN_CAMERA_EMBEDDING_TYPE) is existing


class WanPortableSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wan-portable-fixture.source",
            inputs=(
                InputSpec("visibility", TypeExpr.concrete(CORE_BOOLEAN)),
                InputSpec("layers_as_tuple", TypeExpr.concrete(CORE_BOOLEAN)),
            ),
            outputs=(
                OutputSpec("camera", _CAMERA),
                OutputSpec("tracks", _TRACKS),
                OutputSpec("audio", _AUDIO_ENCODER_OUTPUT),
            ),
        )

    @classmethod
    def execute(cls, *, visibility: bool, layers_as_tuple: bool) -> dict[str, object]:
        return cls.outputs(
            camera=_camera(),
            tracks=_tracks(visibility),
            audio=_audio(layers_as_tuple),
        )


class WanPortableConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wan-portable-fixture.consumer",
            inputs=(
                InputSpec("camera", _CAMERA),
                InputSpec("tracks", _TRACKS),
                InputSpec("audio", _AUDIO_ENCODER_OUTPUT),
                InputSpec("visibility", TypeExpr.concrete(CORE_BOOLEAN)),
                InputSpec("layers_as_tuple", TypeExpr.concrete(CORE_BOOLEAN)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        camera: object,
        tracks: object,
        audio: object,
        visibility: bool,
        layers_as_tuple: bool,
    ) -> dict[str, object]:
        live_camera = materialize_latent(camera)
        live_tracks = materialize_latent(tracks)
        live_audio = materialize_latent(audio)
        _assert_tensor(live_camera, _camera())
        _assert_tracks(live_tracks, _tracks(visibility))
        _assert_audio(live_audio, _audio(layers_as_tuple))
        assert materialize_latent(live_camera) is live_camera
        assert materialize_latent(live_tracks) is live_tracks
        assert materialize_latent(live_audio) is live_audio
        return cls.outputs(summary="portable Wan values verified")


@pytest.mark.parametrize("in_process", (True, False))
def test_wan_portable_values_cross_ordinary_hosts_twice_with_cache(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        source_root = Path(__file__).resolve().parents[1] / "src"
        module = "wan_portable_values_fixture"
        (tmp_path / f"{module}.py").write_text(
            "from test_wan_portable_values_torch import WanPortableConsumer, WanPortableSource\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES = (WanPortableSource, WanPortableConsumer)\n",
            encoding="ascii",
        )
        fixture_manifest = tmp_path / "wan-portable-values-dinkster-pack.toml"
        fixture_manifest.write_text(
            '[pack]\nname = "wan-portable-values-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["wan-portable-fixture"]\n\n[pack.entry]\n'
            f'nodes = "{module}:NODES"\n'
            f'types = "{module}:register_inference_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(source_root))
                ),
            },
        )
        try:
            if in_process:
                specs = (
                    PackSpec(manifest=native_manifest, in_process=True),
                    PackSpec(manifest=fixture_manifest, in_process=True),
                )
            else:
                manifests = (native_manifest, fixture_manifest)
                specs = (
                    PackSpec(
                        manifest=native_manifest,
                        worker_group="wan-portable-values",
                        group_manifests=manifests,
                    ),
                    PackSpec(
                        manifest=fixture_manifest,
                        worker_group="wan-portable-values",
                        group_manifests=manifests,
                    ),
                )
            for spec in specs:
                await composer.add_pack(spec)
            engine = composer.composition.make_engine(lambda _event: None)
            for visibility, layers_as_tuple in ((False, False), (True, True)):
                graph = Graph(
                    nodes={
                        "source": GraphNode(
                            "wan-portable-fixture.source",
                            {"visibility": visibility, "layers_as_tuple": layers_as_tuple},
                        ),
                        "consumer": GraphNode(
                            "wan-portable-fixture.consumer",
                            {
                                "camera": Link("source", "camera"),
                                "tracks": Link("source", "tracks"),
                                "audio": Link("source", "audio"),
                                "visibility": visibility,
                                "layers_as_tuple": layers_as_tuple,
                            },
                        ),
                    }
                )
                for repeat in range(2):
                    result = await engine.run(graph, ["consumer"])
                    assert (
                        result.outputs["consumer"]["summary"].resolve()
                        == "portable Wan values verified"
                    )
                    if repeat:
                        assert set(result.cached) == {"source", "consumer"}
                    else:
                        assert set(result.executed) == {"source", "consumer"}
        finally:
            await composer.close()

    asyncio.run(scenario())
