"""Portable SVG values preserve opaque ordered source bytes."""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from dinkster_api.v1 import (
    CORE_STRING,
    SVG_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_values import (
    SVG_CODEC_MAGIC,
    SVG_CODEC_VERSION,
    decode_svg,
    encode_svg,
    svg_fingerprint,
    svg_meta,
    validate_svg_encoded,
)

from dinkster.compose import PackSpec, ServingComposer

SVG_STREAMS = (b"<svg>one</svg>", b"", b"<svg>two</svg>", b"<svg>one</svg>", b"\xff")


def test_svg_round_trip_preserves_order_duplicates_empty_and_non_text_bytes() -> None:
    value = (
        b"<svg>one</svg>",
        b"",
        b"<svg>two</svg>",
        b"<svg>one</svg>",
        b"\xff\x00<svg/>",
    )
    encoded = encode_svg(value)
    assert encoded == encode_svg(value)
    assert decode_svg(encoded) == value
    assert svg_meta(value) == {"streams": 5, "byte_size": sum(map(len, value))}
    fingerprint = svg_fingerprint(SVG_TYPE)
    assert fingerprint(value) == fingerprint(decode_svg(encoded))
    assert fingerprint(value) != fingerprint((value[2], value[1], value[0], value[3], value[4]))
    assert fingerprint(value) != fingerprint(value[:-1])


def test_svg_uses_complete_stream_bytes_without_changing_source_cursors() -> None:
    sources = [io.BytesIO(b"first"), io.BytesIO(b"second")]
    sources[0].seek(2)
    sources[1].seek(4)
    value = tuple(source.getvalue() for source in sources)
    assert encode_svg(value)
    assert value == (b"first", b"second")
    assert tuple(source.tell() for source in sources) == (2, 4)


@pytest.mark.parametrize(
    "value",
    (
        [b"<svg/>"],
        (bytearray(b"<svg/>"),),
        (memoryview(b"<svg/>"),),
        ("<svg/>",),
    ),
)
def test_svg_rejects_mutable_or_non_byte_runtime_values(value: object) -> None:
    with pytest.raises(TypeError):
        encode_svg(value)


def test_svg_rejects_malformed_frames_and_metadata() -> None:
    data = encode_svg((b"<svg/>", b""))
    truncated_stream = encode_svg((b"<svg/>",))
    with pytest.raises(ValueError, match="invalid SVG codec framing"):
        decode_svg(b"not-an-svg-frame")
    with pytest.raises(ValueError, match="unsupported SVG codec version"):
        decode_svg(
            SVG_CODEC_MAGIC + bytes((SVG_CODEC_VERSION + 1,)) + data[len(SVG_CODEC_MAGIC) + 1 :]
        )
    with pytest.raises(ValueError, match="stream length is truncated"):
        decode_svg(data[: -len(b"<svg/>") - 2])
    with pytest.raises(ValueError, match="stream is truncated"):
        decode_svg(truncated_stream[:-1])
    with pytest.raises(ValueError, match="trailing bytes"):
        decode_svg(data + b"\x00")
    with pytest.raises(ValueError, match="metadata streams"):
        validate_svg_encoded(data, {"streams": 3, "byte_size": 6})
    with pytest.raises(ValueError, match="metadata byte_size"):
        validate_svg_encoded(data, {"streams": 2, "byte_size": 7})


def test_svg_registration_preserves_the_first_authority() -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(SVG_TYPE)
    register_inference_types(registry)
    assert registry.spec(SVG_TYPE) is spec
    value = (b"<svg/>", b"", b"<svg/>")
    encoded = spec.encode(value)
    assert spec.decode(encoded) == value
    assert spec.validate_encoded is not None
    spec.validate_encoded(encoded, svg_meta(value))


def test_svg_public_type_registers_in_two_fresh_python_processes() -> None:
    script = """
import sys
from dinkster_api.v1 import SVG_TYPE, TypeRegistry, register_inference_types
from dinkster_values.builtin_types import SVG_TYPE as canonical

assert SVG_TYPE is canonical
registry = TypeRegistry()
register_inference_types(registry)
register_inference_types(registry)
spec = registry.spec(SVG_TYPE)
value = (b'<svg/>', b'', b'\\xff')
encoded = spec.encode(value)
assert spec.decode(encoded) == value
assert spec.fingerprint(value) == spec.fingerprint(spec.decode(encoded))
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
"""
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr


class SvgSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="svg-fixture.source",
            outputs=(OutputSpec("svg", TypeExpr.concrete(SVG_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(svg=SVG_STREAMS)


class SvgConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="svg-fixture.consumer",
            inputs=(InputSpec("svg", TypeExpr.concrete(SVG_TYPE)),),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, svg: object) -> Mapping[str, object]:
        assert type(svg) is tuple
        assert svg == SVG_STREAMS
        assert svg_fingerprint(SVG_TYPE)(svg) == svg_fingerprint(SVG_TYPE)(SVG_STREAMS)
        return cls.outputs(result="svg")


@pytest.mark.parametrize("in_process", (True, False))
def test_svg_crosses_native_and_own_public_type_fixture_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        native = (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        )
        (tmp_path / "svg_fixture.py").write_text(
            "from test_svg_codec import SvgConsumer, SvgSource\n"
            "from dinkster_api.v1 import register_inference_types\n"
            "NODES = (SvgSource, SvgConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="svg-fixture"\npublisher="dinkster"\n'
            'namespaces=["svg-fixture"]\n[pack.entry]\n'
            'nodes="svg_fixture:NODES"\ntypes="svg_fixture:register_inference_types"\n',
            encoding="ascii",
        )
        manifests = (native, fixture)
        composer = ServingComposer(
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(native.parent / "src"))
                ),
            },
        )
        try:
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "svg-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("svg-fixture.source"),
                    "consumer": GraphNode("svg-fixture.consumer", {"svg": Link("source", "svg")}),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for repeat in range(2):
                result = await engine.run(graph, ["source", "consumer"])
                value = result.outputs["source"]["svg"].resolve()
                assert value == SVG_STREAMS
                assert svg_fingerprint(SVG_TYPE)(value) == svg_fingerprint(SVG_TYPE)(SVG_STREAMS)
                assert result.outputs["consumer"]["result"].resolve() == "svg"
                if repeat:
                    assert set(result.cached) == {"source", "consumer"}
        finally:
            await composer.close()

    asyncio.run(scenario())
