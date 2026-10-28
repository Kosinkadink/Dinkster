"""Mounted SaveText behavior under the native ComfyUI interface."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_api.v1 import ComboWidget, StringWidget
from dinkster_assets import (
    AssetError,
    AssetVault,
    AssetWriter,
    LibraryStore,
    MountSnapshotWriter,
    digest_bytes,
)
from dinkster_graph import Graph, GraphNode, graph_to_wire
from dinkster_nodes_comfy.save_text import SaveText
from dinkster_protocol import AssetFolderContext
from dinkster_server import ServerLibrary, create_app

from dinkster.compose import PackSpec, ServingComposer


def test_save_text_schema_preserves_core_contract_and_appends_mount() -> None:
    schema = SaveText.schema()
    assert schema.node_type == "SaveText"
    assert schema.display_name == "Save Text"
    assert schema.category == "text"
    assert tuple(spec.id for spec in schema.inputs) == (
        "text",
        "filename_prefix",
        "format",
        "mount",
    )
    assert schema.inputs[0].force_input
    assert schema.inputs[0].widget == StringWidget(multiline=False)
    assert schema.inputs[1].default == "ComfyUI"
    assert schema.inputs[2].default == "txt"
    assert isinstance(schema.inputs[2].widget, ComboWidget)
    assert tuple(schema.inputs[2].widget.options) == ("txt", "csv", "md", "json")
    assert not schema.inputs[3].required
    assert schema.inputs[3].default is None
    assert tuple(spec.id for spec in schema.outputs) == ("text",)
    assert schema.output_node
    assert not schema.idempotent


def _snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    roots = {name: tmp_path / name for name in ("default", "alternate", "readonly")}
    for root in roots.values():
        root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {"id": "default", "root": str(roots["default"]), "mode": "readwrite"},
                    {"id": "alternate", "root": str(roots["alternate"]), "mode": "readwrite"},
                    {"id": "readonly", "root": str(roots["readonly"]), "mode": "read"},
                ],
                "outputMount": "default",
            }
        ),
        "utf-8",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    roots["snapshot"] = snapshot
    return roots


def test_save_text_writes_default_and_explicit_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _snapshot(tmp_path, monkeypatch)
    original = '{"greeting":"cafe"}\n'
    assert SaveText.execute(text=original, filename_prefix="notes/default", format="json") == {
        "text": original
    }
    assert SaveText.execute(
        text="alternate\n",
        filename_prefix="notes/%width%x%height%-%unknown%",
        format="md",
        mount="alternate",
    ) == {"text": "alternate\n"}
    assert (roots["default"] / "notes" / "default_00001.json").read_bytes() == (
        b'{\n  "greeting": "cafe"\n}'
    )
    assert (roots["alternate"] / "notes" / "1x1-%unknown%_00001.md").read_bytes() == b"alternate\n"


def test_save_text_collisions_and_security_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _snapshot(tmp_path, monkeypatch)
    SaveText.execute(text="one", filename_prefix="same")
    SaveText.execute(text="two", filename_prefix="same")
    assert (roots["default"] / "same_00001.txt").read_bytes() == b"one"
    assert (roots["default"] / "same_00002.txt").read_bytes() == b"two"
    with pytest.raises(AssetError, match="read-only"):
        SaveText.execute(text="blocked", mount="readonly")
    with pytest.raises(AssetError):
        SaveText.execute(text="blocked", filename_prefix="../escape")
    with pytest.raises(ValueError, match="Unsupported text format"):
        SaveText.execute(text="blocked", format="../evil")
    roots["snapshot"].write_text(json.dumps({"mounts": [], "outputMount": "default"}), "utf-8")
    with pytest.raises(AssetError, match="no ready mount"):
        SaveText.execute(text="revoked")


@pytest.mark.parametrize("in_process", (True, False))
def test_save_text_executes_through_the_ordinary_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    roots = _snapshot(tmp_path, monkeypatch)

    async def scenario() -> None:
        manifest = (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        )
        (tmp_path / "save_text_fixture.py").write_text(
            "from dinkster_nodes_comfy.save_text import SaveText\n"
            "NODES = (SaveText,)\n"
            "def register_types(_registry):\n"
            "    return None\n",
            encoding="ascii",
        )
        fixture_manifest = tmp_path / "dinkster-pack.toml"
        fixture_manifest.write_text(
            manifest.read_text(encoding="ascii")
            .replace("dinkster_nodes_comfy:COMFY_NODES", "save_text_fixture:NODES")
            .replace(
                "dinkster_nodes_comfy:register_inference_types", "save_text_fixture:register_types"
            ),
            encoding="ascii",
        )
        composer = ServingComposer(
            worker_env={
                "DINKSTER_MOUNTS_SNAPSHOT": os.environ["DINKSTER_MOUNTS_SNAPSHOT"],
                "PYTHONPATH": str(tmp_path),
            }
        )
        try:
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "save": GraphNode(
                        "SaveText",
                        {
                            "text": "manifest\n",
                            "filename_prefix": "manifest/%width%x%height%",
                            "format": "txt",
                        },
                    )
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for counter in range(1, 3):
                result = await engine.run(graph, ["save"])
                assert result.outputs["save"]["text"].resolve() == "manifest\n"
                assert len(result.artifacts) == 1
                artifact = result.artifacts[0]
                assert artifact.node_id == "save"
                assert artifact.virtual_path == f"mounts/default/manifest/1x1_{counter:05}.txt"
                assert artifact.digest == digest_bytes(b"manifest\n")
                assert artifact.size == len(b"manifest\n")
                assert artifact.media_type == "text/plain"
            alternate = Graph(
                nodes={
                    "save": GraphNode(
                        "SaveText",
                        {"text": "", "filename_prefix": "empty", "mount": "alternate"},
                    )
                }
            )
            result = await engine.run(alternate, ["save"])
            assert len(result.artifacts) == 1
            assert result.artifacts[0].virtual_path == "mounts/alternate/empty_00001.txt"
            assert result.artifacts[0].digest == digest_bytes(b"")
            assert result.artifacts[0].size == 0
            library = ServerLibrary(
                vault=AssetVault(tmp_path / "vault"),
                store=LibraryStore(tmp_path / "library.sqlite"),
                upload_writer=AssetWriter(MountSnapshotWriter(roots["snapshot"])),
                project_folders_for=lambda _project: AssetFolderContext(
                    ("default", "alternate"), "default"
                ),
            )
            app = create_app(
                lambda on_event: composer.composition.make_engine(on_event),
                composer.composition.schemas,
                library=library,
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                response = await client.post(
                    "/api/jobs",
                    json={
                        "clientId": "save-client",
                        "jobId": "native-save",
                        "project": "film",
                        "graph": graph_to_wire(graph),
                        "targets": ["save"],
                    },
                )
                assert response.status == 202
                submitted = await response.json()
                async with asyncio.timeout(10):
                    while True:
                        status = await (
                            await client.get("/api/jobs/save-client/native-save")
                        ).json()
                        if status["state"] in {"completed", "failed", "cancelled"}:
                            break
                        await asyncio.sleep(0)
                assert status["state"] == "completed", status
                assert (
                    status["artifacts"][0]["virtualPath"] == "mounts/default/manifest/1x1_00003.txt"
                )
                (record,) = library.store.query_assets("local", project="film")[0]
                assert record.asset.digest == digest_bytes(b"manifest\n")
                assert record.asset.virtual_path == "mounts/default/manifest/1x1_00003.txt"
                assert dict(record.origin) == {
                    "kind": "produced",
                    "job": submitted["jobRef"],
                    "node": "save",
                }
                assert library.vault.digests() == []
                assert library.store.query_assets("local", project="another") == []
            finally:
                await client.close()
        finally:
            await composer.close()

    asyncio.run(scenario())
    assert (roots["default"] / "manifest" / "1x1_00001.txt").read_bytes() == b"manifest\n"
    assert (roots["default"] / "manifest" / "1x1_00002.txt").read_bytes() == b"manifest\n"
