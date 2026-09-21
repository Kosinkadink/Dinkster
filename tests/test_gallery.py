"""The dev widget/socket gallery: dev.gallery.* nodes, choice routes, and
the shipped template exercise every construct the native schema wire (v11)
can express, so the frontend can verify rendering/decode against a real
server instead of only its local fixtures.

What this proves: the gallery covers each native widget descriptor and
socket shape on the wire, its choice lists ride /api/choices/* (one
populated, one legally empty), the template document ships through the
development pack table, references only real gallery node types and ports,
and the whole surface is absent unless its manifest is composed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_api.v1 import ASSET_TYPE, AssetRef, AssetVault, TypeRegistry, digest_bytes
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_dev.gallery import (
    EMPTY_CHOICE_ID,
    GALLERY_TEMPLATE_ID,
    GALLERY_TEMPLATE_NAME,
    SAMPLERS,
    SAMPLERS_CHOICE_ID,
    gallery_template_bytes,
    register_gallery_types,
)
from dinkster_schema.wire import SCHEMA_WIRE_VERSION, schema_to_wire
from dinkster_server import create_app
from dinkster_workers.manifest import load_pack_templates

from dinkster.compose import compose_serving

REPO_ROOT = Path(__file__).parent.parent
DEV_PACK_MANIFEST = REPO_ROOT / "packages" / "dinkster-nodes-dev" / "dinkster-pack.toml"


def _interface(wire: dict[str, object]) -> list[dict[str, object]]:
    entries = wire["interface"]
    assert isinstance(entries, list)
    return entries


def _entry(wire: dict[str, object], entry_id: str) -> dict[str, object]:
    for item in _interface(wire):
        if item.get("id") == entry_id:
            return item
    raise AssertionError(f"{wire['nodeType']}: no interface entry {entry_id!r}")


def test_gallery_asset_codec_uses_the_process_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"gallery asset"
    digest = digest_bytes(data)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    registry = TypeRegistry()

    register_gallery_types(registry)

    value = registry.wrap(ASSET_TYPE, {"digest": digest, "name": "gallery.bin", "size": len(data)})
    resolved = value.resolve()
    assert isinstance(resolved, AssetRef)
    assert resolved.read_bytes() == data


def test_gallery_widget_wire_coverage() -> None:
    """dev.gallery.widgets carries every native input-widget descriptor at
    the current wire: bounded NUMBERs, a seed controller, a multiline STRING, a
    per-input displayName, static COMBO, remote COMBO (route +
    refreshButton), a remote-only combo with no static options, a labeled
    BOOLEAN, and bare number/string/boolean inputs that (by convention)
    carry no widget at all."""

    async def scenario() -> None:
        composition = await compose_serving([DEV_PACK_MANIFEST])
        try:
            wire = schema_to_wire(composition.schemas["dev.gallery.widgets"])
            assert wire["schemaVersion"] == SCHEMA_WIRE_VERSION == 1
            assert wire["searchTerms"] == ["gallery", "widget zoo", "kitchen sink"]

            static = _entry(wire, "combo_static")["widget"]
            assert _entry(wire, "combo_static")["type"] == {
                "kind": "concrete",
                "types": ["core.combo"],
            }
            assert isinstance(static, dict)
            assert static["type"] == "COMBO"
            assert "remote" not in static
            assert "alpha" in static["options"]

            remote = _entry(wire, "combo_remote")["widget"]
            assert _entry(wire, "combo_remote")["type"] == {
                "kind": "concrete",
                "types": ["core.combo"],
            }
            assert isinstance(remote, dict)
            assert remote["options"] == list(SAMPLERS)
            assert remote["remote"] == {
                "route": f"/api/choices/{SAMPLERS_CHOICE_ID}",
                "refreshButton": True,
            }

            remote_empty = _entry(wire, "combo_remote_empty")["widget"]
            assert _entry(wire, "combo_remote_empty")["type"] == {
                "kind": "concrete",
                "types": ["core.combo"],
            }
            assert isinstance(remote_empty, dict)
            assert "options" not in remote_empty
            assert remote_empty["remote"] == {
                "route": f"/api/choices/{EMPTY_CHOICE_ID}",
                "refreshButton": True,
            }

            assert "widget" not in _entry(wire, "flag")
            labeled = _entry(wire, "flag_labeled")["widget"]
            assert labeled == {
                "type": "BOOLEAN",
                "labelOn": "enable",
                "labelOff": "disable",
            }

            # Primitive defaults ride InputSpec.default; v11 adds the
            # NUMBER/STRING descriptors and per-input displayName.
            count = _entry(wire, "count")
            assert count["default"] == 50
            assert count["displayName"] == "Item Count"
            assert count["widget"] == {
                "type": "NUMBER",
                "min": 1,
                "max": 100,
                "step": 1,
            }
            assert _entry(wire, "seed")["widget"] == {
                "type": "NUMBER",
                "min": 0,
                "controlAfterGenerate": "randomize",
            }
            assert _entry(wire, "strength")["widget"] == {
                "type": "NUMBER",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
            }
            bare = _entry(wire, "bare_number")
            assert "widget" not in bare
            assert "displayName" not in bare
            assert "widget" not in _entry(wire, "line")
            assert _entry(wire, "prose")["widget"] == {
                "type": "STRING",
                "multiline": True,
            }

            assets = schema_to_wire(composition.schemas["dev.gallery.assets"])
            picture = _entry(assets, "picture")
            assert picture["required"] is False
            assert picture["widget"] == {
                "type": "ASSET",
                "accept": ["image/png", "image/jpeg"],
                "kind": "media/image",
            }
            assert _entry(assets, "destination")["widget"] == {
                "type": "SAVE_TARGET",
                "suffix": ".png",
            }
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_gallery_socket_wire_coverage() -> None:
    """The socket nodes express every TypeExpr shape the wire has: unions
    of 2/3/5 members, wildcards (required and optional), lists (required,
    optional, of-union, nested), match variables, and optional outputs."""

    async def scenario() -> None:
        composition = await compose_serving([DEV_PACK_MANIFEST])
        try:
            sockets = schema_to_wire(composition.schemas["dev.gallery.sockets"])
            union2 = _entry(sockets, "union2")["type"]
            assert isinstance(union2, dict)
            assert union2["kind"] == "union"
            assert len(union2["types"]) == 2
            union4 = _entry(sockets, "union4")["type"]
            assert isinstance(union4, dict)
            assert len(union4["types"]) == 5
            assert _entry(sockets, "any_in")["type"] == {"kind": "wildcard"}
            opt_any = _entry(sockets, "opt_any")
            assert opt_any["type"] == {"kind": "wildcard"}
            assert opt_any["required"] is False

            lists = schema_to_wire(composition.schemas["dev.gallery.lists"])
            of_union = _entry(lists, "list_of_union")["type"]
            assert isinstance(of_union, dict)
            assert of_union["kind"] == "list"
            assert of_union["element"]["kind"] == "union"
            nested = _entry(lists, "list_nested")["type"]
            assert isinstance(nested, dict)
            assert nested["element"]["kind"] == "list"

            match = schema_to_wire(composition.schemas["dev.gallery.match"])
            assert _entry(match, "var_in")["type"] == {
                "kind": "variable",
                "templateId": "T",
            }
            assert _entry(match, "var_out")["type"] == {
                "kind": "variable",
                "templateId": "T",
            }

            source = schema_to_wire(composition.schemas["dev.gallery.source"])
            maybe = _entry(source, "maybe_image")
            assert maybe["optional"] is True

            exotic = schema_to_wire(composition.schemas["dev.gallery.exotic_out"])
            union_out = _entry(exotic, "union_out")["type"]
            assert isinstance(union_out, dict)
            assert union_out["kind"] == "union"
            assert _entry(exotic, "any_out")["type"] == {"kind": "wildcard"}
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_gallery_explicit_pack_surface() -> None:
    """The explicit pack serves gallery nodes, choices, and its template."""

    async def scenario() -> None:
        dev = await compose_serving([DEV_PACK_MANIFEST])
        try:
            assert "dev.gallery.widgets" in dev.schemas
            assert dev.choices[SAMPLERS_CHOICE_ID] == SAMPLERS
            assert dev.choices[EMPTY_CHOICE_ID] == ()
            pack = dev.packs["dinkster-nodes-dev"]
            assert [t.id for t in pack.templates] == [GALLERY_TEMPLATE_ID]
        finally:
            await dev.close()

        plain = await compose_serving()
        try:
            assert not any(t.startswith("dev.") for t in plain.schemas)
            # Core native inference vocabulary is on every composition
            # (stage 3b); no dev.* choice lists leak.
            assert set(plain.choices) == {
                "dinkster.detection.detect.providers",
                "dinkster.detection.segment.providers",
                "dinkster.detection.segment_text.providers",
                "dinkster.detection.track.providers",
                "dinkster.image.matte.providers",
                "dinkster.image.upscale_model.providers",
                "dinkster.preprocess.model_depth.providers",
                "dinkster.preprocess.model_edges.providers",
                "dinkster.preprocess.lineart_realistic.providers",
                "dinkster.preprocess.lineart_anime.providers",
                "dinkster.preprocess.lineart_manga.providers",
                "dinkster.preprocess.anyline.providers",
                "dinkster.preprocess.teed.providers",
                "dinkster.preprocess.mlsd.providers",
                "dinkster.samplers",
                "dinkster.schedulers",
            }
            assert set(plain.packs) == {
                "dinkster-nodes-foundation",
                "dinkster-nodes-image",
                "dinkster-nodes-media-io",
                "dinkster-nodes-remote",
                "dinkster-vision-birefnet",
                "dinkster-vision-depth-anything-v2",
                "dinkster-vision-depth-anything-v3",
                "dinkster-vision-detr",
                "dinkster-vision-efficient-sam",
                "dinkster-vision-hed",
                "dinkster-vision-rtdetr",
                "dinkster-vision-sam31",
                "dinkster-vision-upscale",
            }
        finally:
            await plain.close()

    asyncio.run(scenario())


def test_gallery_choice_and_template_endpoints() -> None:
    """End to end over HTTP: /api/choices/* answers the populated and the
    legally-empty list, /api/templates indexes the gallery under the core
    pack, and the body endpoint serves the exact declared bytes."""

    async def scenario() -> None:
        composition = await compose_serving([DEV_PACK_MANIFEST])
        client = None
        try:
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
            )
            client = TestClient(TestServer(app))
            await client.start_server()

            samplers = await (await client.get(f"/api/choices/{SAMPLERS_CHOICE_ID}")).json()
            assert samplers == list(SAMPLERS)
            empty = await (await client.get(f"/api/choices/{EMPTY_CHOICE_ID}")).json()
            assert empty == []

            listing = await (await client.get("/api/templates")).json()
            rows = {(row["pack"], row["id"]): row for row in listing["templates"]}
            row = rows[("dinkster-nodes-dev", GALLERY_TEMPLATE_ID)]
            assert row["name"] == GALLERY_TEMPLATE_NAME

            body = await client.get(
                f"/api/packs/dinkster-nodes-dev/templates/{GALLERY_TEMPLATE_ID}"
            )
            assert body.status == 200
            data = await body.read()
            assert data == gallery_template_bytes()
            assert row["digest"] == "sha256:" + hashlib.sha256(data).hexdigest()
        finally:
            if client is not None:
                await client.close()
            await composition.close()

    asyncio.run(scenario())


def test_gallery_template_references_real_ports() -> None:
    """The shipped document names only gallery node types that exist and
    only ports their schemas declare - the template can never drift from
    the nodes (the backend does not interpret documents; this pin is why
    it does not have to)."""

    async def scenario() -> None:
        composition = await compose_serving([DEV_PACK_MANIFEST])
        try:
            document = json.loads(gallery_template_bytes())
            assert document["format"] == "dinkster-workflow"
            graph = document["graphs"][document["root"]]
            nodes = graph["nodes"]
            # The frontend loader is strict by design: a graph carries
            # nets/reroutes/nextOrdinal even when trivial, and the document
            # carries a view with a position for every node (this template
            # is a visual gallery - layout is part of the deliverable).
            assert graph["nets"] == {}
            assert graph["reroutes"] == {}
            assert isinstance(graph["nextOrdinal"], int)
            assert graph["nextOrdinal"] >= 0
            view_nodes = document["view"]["graphs"][document["root"]]["nodes"]
            assert set(view_nodes) == set(nodes)
            for placement in view_nodes.values():
                position = placement["position"]
                assert isinstance(position["x"], int | float)
                assert isinstance(position["y"], int | float)
            for node in nodes.values():
                schema = composition.schemas[node["type"]]
                input_ids = {spec.id for spec in schema.inputs}
                for value_id in node.get("values", {}):
                    assert value_id in input_ids, (
                        f"{node['type']}: template value {value_id!r} is not an input"
                    )
            for link in graph["links"].values():
                source = composition.schemas[nodes[link["from"]["node"]]["type"]]
                target = composition.schemas[nodes[link["to"]["node"]]["type"]]
                assert any(o.id == link["from"]["port"] for o in source.outputs)
                assert any(i.id == link["to"]["port"] for i in target.inputs)
            # Both halves of the request: connected examples exist, and
            # every socket node also keeps unconnected ports.
            linked_targets = {
                (link["to"]["node"], link["to"]["port"]) for link in graph["links"].values()
            }
            assert ("sockets", "req_image") in linked_targets
            assert ("sockets", "opt_image") in linked_targets
            assert ("sockets", "union3") not in {(node, port) for node, port in linked_targets}
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_gallery_manifest_template_matches_module() -> None:
    """The manifest template matches the package's source bytes and metadata."""
    templates = load_pack_templates(DEV_PACK_MANIFEST, pack="dinkster-nodes-dev")
    assert len(templates) == 1
    declared = templates[0]
    assert declared.id == GALLERY_TEMPLATE_ID
    assert declared.name == GALLERY_TEMPLATE_NAME
    assert declared.data == gallery_template_bytes()


def test_gallery_nodes_execute() -> None:
    """The gallery is a rendering surface with honest execution: the
    source produces values (its optional output deliberately ABSENT), the
    match node solves T from its input, and the lists node passes its
    list through."""

    async def scenario() -> None:
        composition = await compose_serving([DEV_PACK_MANIFEST])
        try:
            engine = composition.make_engine(lambda event: None)
            graph = Graph(
                nodes={
                    "src": GraphNode("dev.gallery.source", {"size": 2}),
                    "match": GraphNode("dev.gallery.match", {"var_in": Link("src", "mask")}),
                    "lst": GraphNode(
                        "dev.gallery.lists",
                        {
                            "list_req": Link("src", "image_list"),
                            "list_of_union": Link("src", "mask_list"),
                        },
                    ),
                }
            )
            result = await engine.run(graph, ["match", "lst"])
            assert result.outputs["match"]["var_out"].type_id == "dev.gallery.mask"
            assert result.outputs["lst"]["list_out"].type_id == "list<dev.image>"
        finally:
            await composition.close()

    asyncio.run(scenario())
