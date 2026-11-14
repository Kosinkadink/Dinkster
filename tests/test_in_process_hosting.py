"""H3.1 in-process hosting mechanics and tenant-contract proof."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_graph import Graph, GraphNode
from dinkster_server import PackInfo, create_app
from dinkster_workers import ManifestError

from dinkster.compose import (
    PackSpec,
    ServingComposer,
    compose_serving,
)
from tests.async_scenarios import run_scenario

PINS = {"torch": "2.13.0", "comfy-aimdo": "0.4.13"}


def _manifest(directory: Path, *, extra: str = "") -> Path:
    directory.mkdir()
    path = directory / "dinkster-pack.toml"
    path.write_text(
        '[pack]\nname = "isopack"\npublisher = "dinkster"\n'
        'type-namespaces = ["iso"]\nnamespaces = ["iso"]\n'
        f"{extra}\n[pack.entry]\n"
        'nodes = "isopack_nodes:NODES"\ntypes = "isopack_nodes:register_types"\n'
    )
    return path


def _placement_manifest(directory: Path) -> Path:
    directory.mkdir()
    (directory / "h31_placement_nodes.py").write_text(
        "from dinkster_schema import InputSpec,Node,NodeSchema,OutputSpec,TypeExpr\n"
        "STRING=TypeExpr.concrete('STRING')\n"
        "class Echo(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema(node_type='h31.echo',\n"
        "            inputs=(InputSpec('value',STRING),),\n"
        "            outputs=(OutputSpec('value',STRING),))\n"
        "    @classmethod\n"
        "    def execute(cls,*,value): return cls.outputs(value=value)\n"
        "NODES=[Echo]\n"
    )
    manifest = directory / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "h31-placement"\npublisher = "dinkster"\n'
        'namespaces = ["h31"]\n[pack.entry]\n'
        'nodes = "h31_placement_nodes:NODES"\n'
    )
    return manifest


def _composer(**kwargs: object) -> ServingComposer:
    return ServingComposer(
        torch_capable=lambda: True,
        runtime_versions=lambda: PINS,
        **kwargs,  # type: ignore[arg-type]
    )


def test_in_process_placement_is_byte_identical_for_schema_provenance_and_invocation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manifest = _placement_manifest(tmp_path / "pack")
        info = {"h31-placement": PackInfo(display_name="Placement Proof", source="test")}
        isolated = await compose_serving(
            [PackSpec(manifest, env={"PYTHONPATH": str(manifest.parent)}, packs=info)],
            include_default_packs=False,
        )
        composer = _composer()
        in_process = composer.composition
        try:
            await composer.add_pack(
                PackSpec(manifest, packs=info, in_process=True, runtime_pins=PINS)
            )

            async def surface(composition: object) -> bytes:
                current = composition  # keep the helper's type use local
                app = create_app(
                    current.make_engine,  # type: ignore[attr-defined]
                    current.schemas,  # type: ignore[attr-defined]
                    packs=current.packs,  # type: ignore[attr-defined]
                    node_packs=current.node_packs,  # type: ignore[attr-defined]
                )
                client = TestClient(TestServer(app))
                await client.start_server()
                try:
                    payload = await (await client.get("/api/nodes")).json()
                    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                finally:
                    await client.close()

            assert await surface(isolated) == await surface(in_process)
            graph = Graph(nodes={"n": GraphNode("h31.echo", {"value": "same"})})
            isolated_result = await isolated.make_engine(lambda _event: None).run(graph, ["n"])
            in_process_result = await in_process.make_engine(lambda _event: None).run(graph, ["n"])
            left = isolated_result.outputs["n"]["value"]
            right = in_process_result.outputs["n"]["value"]
            assert (left.type_id, left.resolve(), left.meta) == (
                right.type_id,
                right.resolve(),
                right.meta,
            )
            assert in_process._isolated == []

            second_root = tmp_path / "second"
            second_root.mkdir()
            (second_root / "h31_second_nodes.py").write_text(
                (manifest.parent / "h31_placement_nodes.py")
                .read_text()
                .replace("h31.echo", "h32.second")
            )
            second = second_root / "dinkster-pack.toml"
            second.write_text(
                '[pack]\nname = "h31-second"\npublisher = "dinkster"\n'
                'namespaces = ["h32"]\n'
                '[pack.entry]\nnodes = "h31_second_nodes:NODES"\n'
            )
            await composer.add_pack(PackSpec(second, in_process=True, runtime_pins=PINS))
            assert (
                composer._records["h31-placement"].domain is composer._records["h31-second"].domain
            )
            second_graph = Graph(nodes={"n": GraphNode("h32.second", {"value": "same"})})
            second_result = await composer.composition.make_engine(lambda _event: None).run(
                second_graph, ["n"]
            )
            assert second_result.outputs["n"]["value"].resolve() == "same"

            conflicting_root = tmp_path / "conflicting"
            conflicting_root.mkdir()
            (conflicting_root / "h31_second_nodes.py").write_text(
                (second_root / "h31_second_nodes.py").read_text()
            )
            conflicting = conflicting_root / "dinkster-pack.toml"
            conflicting.write_text(
                '[pack]\nname = "h31-conflicting"\npublisher = "dinkster"\n'
                'namespaces = ["h33"]\n'
                '[pack.entry]\nnodes = "h31_second_nodes:NODES"\n'
            )
            with pytest.raises(ManifestError, match="already loaded from outside the pack root"):
                await composer.add_pack(PackSpec(conflicting, in_process=True, runtime_pins=PINS))
        finally:
            await composer.close()
            await isolated.close()

    run_scenario(scenario())
