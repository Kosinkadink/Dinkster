"""Granted selections feed native loaders without changing source option names."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from dinkster_api.v1 import (
    AssetRef,
    AssetWidget,
    LocalAssetRef,
    resolve_asset_selection,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_assets import digest_bytes
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy import model_selections as loaders
from dinkster_server.preflight import bind_model_names
from dinkster_workers import load_manifest


class LoadedVAE:
    def __init__(self, **kwargs: object) -> None:
        self.arguments = kwargs
        self.patcher = SimpleNamespace()
        self.validated = False

    def throw_exception_if_invalid(self) -> None:
        self.validated = True


@pytest.fixture
def granted_loaders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    roots = {
        kind: tmp_path / kind.replace("/", "-")
        for kind in ("model/vae", "model/vae-approx", "model/diffusers", "model/embedding")
    }
    for root in roots.values():
        root.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {"id": str(i), "root": str(root), "kind": kind, "index": str(root / "index")}
                    for i, (kind, root) in enumerate(roots.items())
                ]
            }
        )
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    monkeypatch.setattr(loaders, "initialize_inference_process", lambda: None)
    metadata = {"format": "test"}
    loads = []

    def load(path, *, safe_load, return_metadata=False):
        assert safe_load is True
        loads.append(Path(path))
        state_dict = {"weight": Path(path).read_bytes()}
        return (state_dict, metadata) if return_metadata else state_dict

    factory = object()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(tensor=lambda value: value))
    monkeypatch.setitem(
        sys.modules, "dinkster_inference.utils", SimpleNamespace(load_torch_file=load)
    )
    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference.sd",
        SimpleNamespace(VAE=LoadedVAE, load_vae_patcher=factory),
    )
    monkeypatch.setitem(
        sys.modules,
        "dinkster_inference.latent_formats",
        SimpleNamespace(
            QwenImage21=lambda: SimpleNamespace(latents_std=[4.0], latents_mean=[-3.0])
        ),
    )
    return roots, metadata, loads, factory


def test_native_selection_schema_roundtrip_and_registration():
    for node in loaders.MODEL_SELECTION_NODES:
        schema = node.schema()
        assert schema_from_wire(schema_to_wire(schema)) == schema
        assert len(schema.inputs) == 1
        widget = schema.inputs[0].widget
        assert isinstance(widget, AssetWidget) and widget.selection
        assert not schema.idempotent
        assert node in COMFY_NODES
    assert (
        tuple(node.schema().node_type for node in COMFY_NODES)
        == load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml").namespaces
    )


def test_preflight_keeps_source_selection_names_and_reports_missing_options(granted_loaders):
    roots, _, _, _ = granted_loaders
    path = roots["model/diffusers"] / "vendor" / "model"
    path.mkdir(parents=True)
    (path / "model_index.json").write_text("{}")
    graph = Graph(
        {
            "literal": GraphNode("VAELoader", {"vae_name": "pixel_space"}),
            "tree": GraphNode("DiffusersLoader", {"model_path": "vendor/model"}),
            "missing": GraphNode("VAELoader", {"vae_name": "not-granted.safetensors"}),
        }
    )
    schemas = {node.schema().node_type: node.schema() for node in loaders.MODEL_SELECTION_NODES}
    bound, missing = bind_model_names(graph, schemas, None)
    assert bound.nodes["literal"].inputs["vae_name"] == TypedLiteral("COMBO", "pixel_space")
    assert bound.nodes["tree"] == graph.nodes["tree"]
    assert bound.nodes["missing"] == graph.nodes["missing"]
    assert len(missing) == 1
    assert missing[0]["nodeId"] == "missing"
    assert missing[0]["name"] == "not-granted.safetensors"
    assert missing[0]["fetchable"] is False
    assert "digest" not in missing[0]


@pytest.mark.parametrize("mode", ["name", "local", "verified", "video"])
def test_single_file_vae_retains_metadata_validation_and_reload_factory(granted_loaders, mode):
    roots, metadata, loads, factory = granted_loaders
    path = roots["model/vae-approx" if mode == "video" else "model/vae"] / (
        "taehv.safetensors" if mode == "video" else "real.safetensors"
    )
    path.write_bytes(b"vae")
    value = path.name
    if mode in {"local", "verified"}:
        value = resolve_asset_selection("model/vae", value)
        assert isinstance(value, LocalAssetRef)
        if mode == "verified":

            class Resolver:
                def resolve(self, digest):
                    return path if digest == digest_bytes(b"vae") else None

            value = AssetRef(digest_bytes(b"vae"), path.name, 3, resolver=Resolver())
    vae = loaders.VAELoader.execute(vae_name=value)["VAE"]
    assert isinstance(vae, LoadedVAE)
    assert vae.arguments == {"sd": {"weight": b"vae"}, "metadata": metadata}
    assert vae.arguments["metadata"] is metadata
    assert vae.validated
    assert vae.patcher.cached_patcher_init == (factory, (str(path), metadata, None))
    assert loads == [path]


@pytest.mark.parametrize(
    "name,extras,metadata",
    [
        ("taesd", {"vae_scale": 0.18215, "vae_shift": 0.0}, None),
        ("taesdxl", {"vae_scale": 0.13025, "vae_shift": 0.0}, None),
        ("taesd3", {"vae_scale": 1.5305, "vae_shift": 0.0609}, None),
        ("taef1", {"vae_scale": 0.3611, "vae_shift": 0.1159}, None),
        ("taef2", {}, {"tae_latent_channels": 128}),
        ("taeqi2_1", {"vae_scale": 0.25, "vae_shift": -3.0}, None),
        ("pixel_space", {"pixel_space_vae": 1.0}, None),
    ],
)
def test_pair_and_literal_vae_keep_source_scaling_without_single_file_factory(
    granted_loaders, name, extras, metadata
):
    roots, _, loads, _ = granted_loaders
    expected = dict(extras)
    if name != "pixel_space":
        for member, content in (("encoder", b"enc"), ("decoder", b"dec")):
            (roots["model/vae-approx"] / f"{name}_{member}.safetensors").write_bytes(content)
            expected[f"taesd_{member}.weight"] = content
    vae = loaders.VAELoader.execute(vae_name=name)["VAE"]
    assert isinstance(vae, LoadedVAE)
    assert vae.arguments == {"sd": expected, "metadata": metadata}
    assert vae.validated
    assert not hasattr(vae.patcher, "cached_patcher_init")
    assert len(loads) == (0 if name == "pixel_space" else 2)


def test_diffusers_consumes_selected_tree_and_exact_embedding_grants(granted_loaders, monkeypatch):
    roots, _, _, _ = granted_loaders
    path = roots["model/diffusers"] / "vendor" / "model"
    path.mkdir(parents=True)
    (path / "model_index.json").write_text("{}")
    model, clip, vae = object(), object(), object()

    def load(selected, **kwargs):
        assert selected == str(path)
        assert kwargs == {
            "output_vae": True,
            "output_clip": True,
            "embedding_directory": [str(roots["model/embedding"])],
        }
        return model, clip, vae

    monkeypatch.setitem(
        sys.modules, "dinkster_inference.diffusers_load", SimpleNamespace(load_diffusers=load)
    )
    assert loaders.DiffusersLoader.execute(model_path="vendor/model") == {
        "MODEL": model,
        "CLIP": clip,
        "VAE": vae,
    }
