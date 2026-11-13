"""Granted selections feed native loaders without changing source option names."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_models import model_selections as loaders
from dinkster_server.preflight import bind_model_names


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


def test_preflight_keeps_source_selection_names_and_reports_missing_options(granted_loaders):
    roots, _, _, _ = granted_loaders
    path = roots["model/diffusers"] / "vendor" / "model"
    path.mkdir(parents=True)
    (path / "model_index.json").write_text("{}")
    graph = Graph(
        {
            "literal": GraphNode("dinkster.models.vae_loader", {"vae_name": "pixel_space"}),
            "tree": GraphNode("dinkster.models.diffusers_loader", {"model_path": "vendor/model"}),
            "missing": GraphNode(
                "dinkster.models.vae_loader", {"vae_name": "not-granted.safetensors"}
            ),
        }
    )
    schemas = {
        node.schema().node_type: node.schema()
        for node in (loaders.VAELoader, loaders.DiffusersLoader)
    }
    bound, missing = bind_model_names(graph, schemas, None)
    assert bound.nodes["literal"].inputs["vae_name"] == TypedLiteral("COMBO", "pixel_space")
    assert bound.nodes["tree"] == graph.nodes["tree"]
    assert bound.nodes["missing"] == graph.nodes["missing"]
    assert len(missing) == 1
    assert missing[0]["nodeId"] == "missing"
    assert missing[0]["name"] == "not-granted.safetensors"
    assert missing[0]["fetchable"] is False
    assert "digest" not in missing[0]


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
