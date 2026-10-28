from __future__ import annotations

import asyncio
import ctypes
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_TYPE,
    LAYERS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_values import (
    LATENT_CODEC_MAGIC,
    EncodedLatentTensor,
    EncodedMultiStreamLatent,
    decode_latent,
    encode_latent,
    validate_latent_encoded,
)
from dinkster_values.latent_codec import _copy_address

from dinkster.compose import PackSpec, ServingComposer


def test_empty_latent_is_a_present_value_without_loading_torch() -> None:
    value: dict[str, object] = {}
    encoded = encode_latent(value)
    assert decode_latent(encoded) == {}
    validate_latent_encoded(encoded, {})
    assert materialize_latent(value) is value
    assert encode_latent(decode_latent(encoded)) == encoded
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_api.v1 import materialize_latent; "
            "from dinkster_values import encode_latent, decode_latent, validate_latent_encoded; "
            "value = {}; encoded = encode_latent(value); "
            "assert decode_latent(encoded) == {}; validate_latent_encoded(encoded, {}); "
            "assert materialize_latent(value) is value; "
            "assert not any(n == 'torch' or n.startswith('torch.') "
            "or n == 'dinkster_inference' or n.startswith('dinkster_inference.') "
            "for n in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", ({"metadata": {}}, {"noise_mask": None}, None, []))
def test_empty_latent_does_not_allow_other_sampleless_values(value: object) -> None:
    encoded = encode_latent(value)
    with pytest.raises(ValueError, match="samples mapping entry"):
        decode_latent(encoded)
    with pytest.raises(ValueError, match="samples mapping entry"):
        validate_latent_encoded(encoded, {})


class EmptyLatentSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="empty-latent-fixture.source",
            outputs=(OutputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(latent={})


class EmptyLatentInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="empty-latent-fixture.inspect",
            inputs=(InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, latent: object) -> Mapping[str, object]:
        assert type(latent) is dict and latent == {}
        assert materialize_latent(latent) is latent
        return cls.outputs(summary="present empty latent")


@pytest.mark.parametrize("in_process", (True, False))
def test_empty_required_latent_survives_hosts_and_disk_memory_cache(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        media = Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-media-io"
        (tmp_path / "empty_latent_fixture.py").write_text(
            "from tests.test_latent_codec import EmptyLatentInspect, EmptyLatentSource\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "from dinkster_nodes_media_io import MEDIA_IO_NODES, register_media_types\n"
            "from dinkster_nodes_media_io import capture_device_choices\n"
            "NODES = (*MEDIA_IO_NODES, EmptyLatentSource, EmptyLatentInspect)\n"
            "def register_types(registry):\n"
            "    register_inference_types(registry)\n"
            "    register_media_types(registry)\n",
            encoding="ascii",
        )
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            (media / "dinkster-pack.toml")
            .read_text(encoding="ascii")
            .replace("namespaces = [", 'namespaces = ["empty-latent-fixture",')
            .replace("dinkster_nodes_media_io:MEDIA_IO_NODES", "empty_latent_fixture:NODES")
            .replace(
                "dinkster_nodes_media_io:register_media_types",
                "empty_latent_fixture:register_types",
            )
            .replace(
                "dinkster_nodes_media_io:capture_device_choices",
                "empty_latent_fixture:capture_device_choices",
            ),
            encoding="ascii",
        )
        graph = Graph(
            nodes={
                "source": GraphNode("empty-latent-fixture.source"),
                "inspect": GraphNode(
                    "empty-latent-fixture.inspect", {"latent": Link("source", "latent")}
                ),
            }
        )
        for restart in range(2):
            composer = ServingComposer(
                cache_mode="layered",
                cache_dir=tmp_path / "cache",
                worker_env={"PYTHONPATH": os.pathsep.join((str(tmp_path), str(media.parents[1])))},
            )
            try:
                await composer.add_pack(
                    PackSpec(manifest, in_process=in_process, trust_reserved=True)
                )
                events = []
                engine = composer.composition.make_engine(events.append)
                if not restart:
                    for _ in range(2):
                        uncached = await engine.run(graph, ["inspect"], cache_enabled=False)
                        assert set(uncached.executed) == {"source", "inspect"}
                        assert (
                            uncached.outputs["inspect"]["summary"].resolve()
                            == "present empty latent"
                        )
                for repeat in range(2):
                    events.clear()
                    result = await engine.run(graph, ["source", "inspect"])
                    assert result.outputs["source"]["latent"].resolve() == {}
                    assert result.outputs["inspect"]["summary"].resolve() == "present empty latent"
                    if restart or repeat:
                        assert set(result.cached) == {"source", "inspect"}
                        layer = "disk" if restart and not repeat else "memory"
                        assert {
                            e.detail["cacheLayer"] for e in events if e.kind == "node_cached"
                        } == {layer}
                    else:
                        assert set(result.executed) == {"source", "inspect"}
            finally:
                await composer.close()

    asyncio.run(scenario())


def _tensor(*, fill: int, shape: tuple[int, ...]) -> EncodedLatentTensor:
    size = 1
    for dimension in shape:
        size *= dimension
    return EncodedLatentTensor("float32", shape, bytes((fill, 0, 0, 0)) * size)


def test_tensor_buffer_copy_uses_pointer_sized_lengths() -> None:
    source = ctypes.create_string_buffer(b"latent")
    assert _copy_address(ctypes.addressof(source), 6) == b"latent"
    assert _copy_address(0, 0) == b""
    assert (ctypes.c_ubyte * (2**32 + 2))._length_ == 2**32 + 2


def test_structural_latent_codec_is_deterministic_and_round_trips_streams() -> None:
    video = _tensor(fill=1, shape=(1, 2, 3))
    audio = _tensor(fill=2, shape=(1, 4))
    value = {
        "samples": EncodedMultiStreamLatent((("video", video), ("audio", audio))),
        "noise_mask": EncodedMultiStreamLatent(
            (("video", _tensor(fill=3, shape=(1, 1, 3))), ("audio", audio))
        ),
        "batch_index": (0,),
        "metadata": {"name": "value", "enabled": True},
    }
    encoded = encode_latent(value)
    reordered = encode_latent(dict(reversed(tuple(value.items()))))
    assert encoded == reordered
    assert encoded.startswith(LATENT_CODEC_MAGIC)
    validate_latent_encoded(encoded, {})

    decoded = decode_latent(encoded)
    assert isinstance(decoded, dict)
    samples = decoded["samples"]
    assert type(samples) is EncodedMultiStreamLatent
    assert samples.roles == ("video", "audio")
    assert samples.streams[0][1] == video
    assert decoded["batch_index"] == (0,)
    assert decoded["metadata"] == {"enabled": True, "name": "value"}


@pytest.mark.parametrize("reserved", ("tensor", "multi", "tuple", "map", "user"))
def test_latent_codec_escapes_reserved_mapping_keys(reserved: str) -> None:
    value = {
        "samples": _tensor(fill=1, shape=(1,)),
        "metadata": {"$": reserved, "items": {"$": "tensor"}},
    }
    assert decode_latent(encode_latent(value)) == value


def test_latent_codec_validator_rejects_old_and_corrupt_frames() -> None:
    with pytest.raises(ValueError, match="framing"):
        validate_latent_encoded(b"pickle:not-a-latent", {})
    encoded = encode_latent({"samples": _tensor(fill=1, shape=(1,))})
    with pytest.raises(ValueError, match="payload"):
        validate_latent_encoded(encoded[:-4], {})
    with pytest.raises(ValueError, match="dtype and shape"):
        validate_latent_encoded(encoded.replace(b"float32", b"float99"), {})

    prefix = LATENT_CODEC_MAGIC + b"\x01"
    malformed = {"$": "map", "items": {}, "extra": True}
    with pytest.raises(ValueError, match="mapping"):
        validate_latent_encoded(prefix + json.dumps(malformed).encode(), {})


def test_registered_layers_preserve_nested_tensors_state_and_identity() -> None:
    from dinkster_api.v1 import LAYERS_TYPE, TypeRegistry, register_inference_types

    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(LAYERS_TYPE)
    image = _tensor(fill=3, shape=(1, 2, 5, 3))
    mask = EncodedLatentTensor("bool", (1, 2, 5), bytes((0, 1, 1, 0, 1)) * 2)
    document = {
        "version": 1,
        "canvas": (19, 7),
        "layers": [
            {
                "type": "raster",
                "image": image,
                "mask": mask,
                "x": -3,
                "y": 7,
                "z_index": 9,
                "name": "front",
                "opacity": 0.375,
                "blend_mode": "multiply",
                "visible": False,
                "flip_h": True,
                "flip_v": False,
                "rotation": 31.5,
                "w": 13,
                "h": 5,
            },
            {"type": "raster", "image": _tensor(fill=7, shape=(2, 3, 1, 4)), "z_index": 2},
        ],
        "metadata": {"labels": ("front", "back"), "enabled": True},
    }
    encoded = spec.encode(document)
    assert spec.decode(encoded) == document
    assert spec.validate_encoded is not None
    spec.validate_encoded(encoded, {})
    assert spec.encode(spec.decode(encoded)) == encoded
    assert spec.fingerprint is not None
    reordered = dict(reversed(tuple(document.items())))
    assert spec.encode(reordered) == encoded
    assert spec.fingerprint(reordered) == spec.fingerprint(document)
    assert spec.fingerprint({**document, "layers": list(reversed(document["layers"]))}) != (
        spec.fingerprint(document)
    )
    assert spec.fingerprint({**document, "canvas": (7, 19)}) != spec.fingerprint(document)
    changed_image = {**document["layers"][0], "image": _tensor(fill=4, shape=image.shape)}
    assert spec.fingerprint(
        {**document, "layers": [changed_image, document["layers"][1]]}
    ) != spec.fingerprint(document)
    latent = registry.spec(LATENT_TYPE)
    assert latent.fingerprint is not None
    assert spec.fingerprint(document) != latent.fingerprint(document)
    minimal = {"version": 1, "layers": []}
    assert spec.decode(spec.encode(minimal)) == minimal


def test_registered_layers_materialize_encoded_tensors_and_preserve_live_storage() -> None:
    torch = pytest.importorskip("torch")
    from dinkster_api.v1 import TypeRegistry, register_inference_types

    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(LAYERS_TYPE)
    image = torch.arange(30, dtype=torch.float64).reshape(1, 2, 5, 3)
    mask = torch.tensor([[[False, True, True, False, True], [True, False, False, True, False]]])
    document = {"canvas": (19, 7), "layers": [{"image": image, "mask": mask, "x": -3}]}
    live = materialize_latent(document)
    assert live is document
    assert live["layers"][0]["image"] is image
    assert live["layers"][0]["mask"] is mask
    assert live["layers"][0]["image"].data_ptr() == image.data_ptr()
    decoded = materialize_latent(spec.decode(spec.encode(document)))
    assert decoded["canvas"] == (19, 7)
    assert decoded["layers"][0]["x"] == -3
    for name, expected in (("image", image), ("mask", mask)):
        actual = decoded["layers"][0][name]
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)


class LayerDocumentSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="layer-fixture.source",
            outputs=(OutputSpec("layers", TypeExpr.concrete(LAYERS_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(
            layers={
                "canvas": (19, 7),
                "layers": [
                    {"image": _tensor(fill=3, shape=(1, 2, 5, 3)), "x": -3},
                    {"mask": EncodedLatentTensor("bool", (1, 2, 3), b"\x00\x01\x01\x00\x01\x00")},
                ],
            }
        )


class LayerDocumentInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="layer-fixture.inspect",
            inputs=(InputSpec("layers", TypeExpr.concrete(LAYERS_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, layers: dict[str, object]) -> Mapping[str, object]:
        assert layers == {
            "canvas": (19, 7),
            "layers": [
                {
                    "image": EncodedLatentTensor("float32", (1, 2, 5, 3), b"\x03\x00\x00\x00" * 30),
                    "x": -3,
                },
                {"mask": EncodedLatentTensor("bool", (1, 2, 3), b"\x00\x01\x01\x00\x01\x00")},
            ],
        }
        return cls.outputs(summary="two asymmetric layers")


@pytest.mark.parametrize("in_process", (True, False))
def test_registered_layers_cross_hosts_and_persistent_cache(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        (tmp_path / "layer_fixture.py").write_text(
            "from tests.test_latent_codec import LayerDocumentSource, LayerDocumentInspect\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES = (LayerDocumentSource, LayerDocumentInspect)\n",
            encoding="ascii",
        )
        manifest = tmp_path / "dinkster-pack.toml"
        manifest.write_text(
            '[pack]\nname = "layer-fixture"\npublisher = "dinkster"\n'
            'namespaces = ["layer-fixture"]\n'
            '[pack.contracts]\nhost = "dinkster-pack-host/1"\napi = "dinkster-api/v1"\n'
            '[pack.entry]\nnodes = "layer_fixture:NODES"\n'
            'types = "layer_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        graph = Graph(
            nodes={
                "source": GraphNode("layer-fixture.source"),
                "inspect": GraphNode("layer-fixture.inspect", {"layers": Link("source", "layers")}),
            }
        )
        for restart in range(2):
            composer = ServingComposer(
                cache_mode="layered",
                cache_dir=tmp_path / "cache",
                worker_env={
                    "PYTHONPATH": os.pathsep.join(
                        (str(tmp_path), str(Path(__file__).resolve().parents[1]))
                    )
                },
            )
            try:
                await composer.add_pack(
                    PackSpec(manifest, in_process=in_process, trust_reserved=True)
                )
                events = []
                engine = composer.composition.make_engine(events.append)
                if not restart:
                    for _ in range(2):
                        result = await engine.run(graph, ["inspect"], cache_enabled=False)
                        assert set(result.executed) == {"source", "inspect"}
                        assert (
                            result.outputs["inspect"]["summary"].resolve()
                            == "two asymmetric layers"
                        )
                for repeat in range(2):
                    events.clear()
                    result = await engine.run(graph, ["source", "inspect"])
                    assert result.outputs["inspect"]["summary"].resolve() == "two asymmetric layers"
                    if restart or repeat:
                        assert set(result.cached) == {"source", "inspect"}
                        assert {
                            e.detail["cacheLayer"] for e in events if e.kind == "node_cached"
                        } == {"disk" if restart and not repeat else "memory"}
                    else:
                        assert set(result.executed) == {"source", "inspect"}
            finally:
                await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("corruption", ("buffer", "shape", "dtype"))
def test_registered_layers_reject_malformed_nested_tensor_records(corruption: str) -> None:
    from dinkster_api.v1 import LAYERS_TYPE, TypeRegistry, register_inference_types

    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(LAYERS_TYPE)
    encoded = spec.encode({"version": 1, "layers": [{"image": _tensor(fill=1, shape=(1, 2, 3))}]})
    prefix = LATENT_CODEC_MAGIC + b"\x01"
    tree = json.loads(encoded[len(prefix) :])
    record = tree["items"]["layers"][0]["items"]["image"]
    if corruption == "buffer":
        record["data"] = "AA=="
    elif corruption == "shape":
        record["shape"] = [1, -2, 3]
    else:
        record["dtype"] = "float99"
    malformed = prefix + json.dumps(tree).encode()
    assert spec.validate_encoded is not None
    with pytest.raises(ValueError, match="tensor"):
        spec.validate_encoded(malformed, {})
    with pytest.raises(ValueError, match="tensor"):
        spec.decode(malformed)
