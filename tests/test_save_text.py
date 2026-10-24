"""Mounted SaveText behavior under the native ComfyUI interface."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from dinkster_api.v1 import ComboWidget
from dinkster_assets import AssetError
from dinkster_graph import Graph, GraphNode
from dinkster_nodes_comfy.save_text import SaveText

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
        text="alternate\n", filename_prefix="notes/alternate", format="md", mount="alternate"
    ) == {"text": "alternate\n"}
    assert (roots["default"] / "notes" / "default_00001.json").read_bytes() == (
        b'{\n  "greeting": "cafe"\n}'
    )
    assert (roots["alternate"] / "notes" / "alternate_00001.md").read_bytes() == b"alternate\n"


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
            result = await composer.composition.make_engine(lambda _event: None).run(
                Graph(
                    nodes={
                        "save": GraphNode(
                            "SaveText",
                            {"text": "manifest\n", "filename_prefix": "manifest", "format": "txt"},
                        )
                    }
                ),
                ["save"],
            )
            assert result.outputs["save"]["text"].resolve() == "manifest\n"
        finally:
            await composer.close()

    asyncio.run(scenario())
    assert (roots["default"] / "manifest_00001.txt").read_bytes() == b"manifest\n"
