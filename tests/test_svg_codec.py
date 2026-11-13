"""Portable SVG values preserve opaque ordered source bytes."""

from __future__ import annotations

import io
import os
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
    materialize_svg,
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
from tests.async_scenarios import run_scenario

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


def test_svg_materialization_keeps_cached_bytes_and_consumer_cursors_independent() -> None:
    value = (b"first", b"", b"first", b"\xff\x00last")
    encoded = encode_svg(value)
    streams = materialize_svg(decode_svg(encoded))
    assert all(isinstance(stream, io.BytesIO) for stream in streams)
    assert tuple(stream.read() for stream in streams) == value
    streams[0].seek(1)
    streams[0].write(b"CHANGED")
    streams[1].close()
    streams[2].seek(2)
    assert streams[2].read() == b"rst"
    reopened = materialize_svg(value)
    assert tuple(stream.tell() for stream in reopened) == (0, 0, 0, 0)
    assert tuple(stream.read() for stream in reopened) == value
    assert encode_svg(value) == encoded
    assert materialize_svg(()) == ()


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
        streams = materialize_svg(svg)
        assert tuple(stream.read() for stream in streams) == SVG_STREAMS
        streams[0].close()
        assert tuple(stream.read() for stream in materialize_svg(svg)) == SVG_STREAMS
        return cls.outputs(result="svg")


@pytest.mark.parametrize("in_process", (True, False))
def test_svg_crosses_native_and_own_public_type_fixture_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        (tmp_path / "svg_fixture.py").write_text(
            "from test_svg_codec import SvgConsumer, SvgSource\n"
            "from dinkster_api.v1 import register_media_types as register_types\n"
            "NODES = (SvgSource, SvgConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="svg-fixture"\npublisher="dinkster"\n'
            'namespaces=["svg-fixture"]\n[pack.entry]\n'
            'nodes="svg_fixture:NODES"\ntypes="svg_fixture:register_types"\n',
            encoding="ascii",
        )
        composer = ServingComposer(
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path(__file__).parent))),
            },
        )
        try:
            for manifest in (fixture,):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "svg-fixture",
                        group_manifests=() if in_process else (fixture,),
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

    run_scenario(scenario())
