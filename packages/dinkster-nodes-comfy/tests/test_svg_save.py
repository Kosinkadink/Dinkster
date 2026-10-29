"""Native SVG publication through ordinary execution hosts."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from dinkster_api.v1 import (
    CORE_STRING,
    SVG_TYPE,
    ExportSnapshot,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_svg,
)
from dinkster_assets import MountDef, MountTable
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.svg_save import SaveSVGNode

from dinkster.compose import PackSpec, ServingComposer

_FIRST = b'<svg width="7" height="3"><path d="M1 2L5 1"/></svg>'
_VALUES = (_FIRST, b"plain UTF-8 text", b"", _FIRST)


class SVGSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="svg-save-fixture.source",
            outputs=(OutputSpec("svg", TypeExpr.concrete(SVG_TYPE)),),
        )

    @classmethod
    def execute(cls):
        return cls.outputs(svg=_VALUES)


class SVGInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="svg-save-fixture.inspect",
            inputs=(InputSpec("svg", TypeExpr.concrete(SVG_TYPE)),),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
            idempotent=False,
        )

    @classmethod
    def execute(cls, *, svg):
        streams = materialize_svg(svg)
        try:
            assert tuple(stream.read() for stream in streams) == _VALUES
        finally:
            for stream in streams:
                stream.close()
        return cls.outputs(result="SVG unchanged")


def test_svg_save_schema_uses_canonical_type_and_adjacent_mount() -> None:
    schema = SaveSVGNode.define_schema()
    assert [input.id for input in schema.inputs] == ["svg", "filename_prefix", "mount"]
    assert schema.inputs[-1].adjacent_to == "filename_prefix"
    assert schema.inputs[0].type == TypeExpr.concrete(SVG_TYPE)
    assert schema.outputs[0].type == TypeExpr.concrete(SVG_TYPE)
    assert schema.output_node and not schema.idempotent


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("prompt", (None, {}, {"source": {"seed": 83}}))
def test_svg_save_publishes_ordered_streams_and_submission_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool, prompt: dict | None
) -> None:
    output, alternate = tmp_path / "output", tmp_path / "alternate"
    output.mkdir()
    alternate.mkdir()
    mounts = tmp_path / "mounts.json"
    table = MountTable(mounts, index_root=tmp_path / "index")
    for name, root in (("output", output), ("alternate", alternate)):
        table.add(MountDef(name, root, mode="readwrite"))
        table.scan(name)
    table.select_output_mount("output")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(mounts))
    extra = {"workflow": {"nodes": [{"id": 43, "title": "SVG: caf\u00e9"}]}}
    submitted = None if prompt is None else ExportSnapshot(prompt=prompt, extra_pnginfo=extra)

    async def scenario():
        module = f"svg_fixture_{tmp_path.name.replace('-', '_')}"
        (tmp_path / f"{module}.py").write_text(
            "from test_svg_save import SVGSource, SVGInspect\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (SVGSource, SVGInspect)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="svg-save-fixture"\npublisher="dinkster"\n'
            'namespaces=["svg-save-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        manifests = (native, fixture)
        paths = (str(tmp_path), str(Path(__file__).parent), *(p for p in sys.path if p))
        composer = ServingComposer(
            worker_env={
                "DINKSTER_MOUNTS_SNAPSHOT": str(mounts),
                "PYTHONPATH": os.pathsep.join(paths),
                "DINKSTER_ACCELERATOR": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
        try:
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "svg-save-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("svg-save-fixture.source"),
                    "save": GraphNode(
                        "SaveSVGNode",
                        {
                            "svg": Link("source", "svg"),
                            "filename_prefix": "chosen/vector_%batch_num%",
                            "mount": "alternate",
                        },
                    ),
                    "inspect": GraphNode("svg-save-fixture.inspect", {"svg": Link("save", "svg")}),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"], export_snapshot=submitted)
                assert result.outputs["inspect"]["result"].resolve() == "SVG unchanged"
                if attempt:
                    assert "source" in result.cached
                    assert not {"save", "inspect"} & set(result.cached)
        finally:
            await composer.close()

    asyncio.run(scenario())
    paths = sorted((alternate / "chosen").glob("*.svg"))
    assert len(paths) == 8
    metadata = {"prompt": prompt, **extra}
    for batch, value in enumerate(_VALUES):
        expected = value
        if submitted is not None and value == _FIRST:
            element = (
                "  <metadata>\n                <![CDATA[\n            "
                + json.dumps(metadata, indent=2)
                + "\n                ]]>\n            </metadata>\n            "
            )
            expected = value.replace(b"><path", (">\n" + element + "<path").encode())
        for counter in (batch + 1, batch + 2):
            assert (
                alternate / f"chosen/vector_{batch}_{counter:05d}_.svg"
            ).read_bytes() == expected
    assert not list(output.rglob("*.svg"))


def test_svg_save_rejects_invalid_utf8_without_replacing_bytes(tmp_path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    mounts = tmp_path / "mounts.json"
    table = MountTable(mounts, index_root=tmp_path / "index")
    table.add(MountDef("output", output, mode="readwrite"))
    table.scan("output")
    table.select_output_mount("output")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(mounts))
    with pytest.raises(UnicodeDecodeError):
        SaveSVGNode.execute(svg=(b"\xff",))
    assert not list(output.rglob("*.svg"))
    assert SaveSVGNode.execute(svg=())["svg"] == ()
