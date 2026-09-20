from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

import pytest
from dinkster_assets import RemoteSource
from dinkster_graph import Graph, GraphNode
from dinkster_nodes_image import (
    IMAGE_NODES,
    MODEL_DEPTH_PROVIDER_CHOICE,
    preprocessor_choices,
)
from dinkster_nodes_image import (
    ModelDepthPreprocessor as OwnerModelDepthPreprocessor,
)
from dinkster_nodes_vision.depth_anything_v3 import (
    ModelDepthPreprocessor as DepthAnything3ModelDepthPreprocessor,
)
from dinkster_schema import ComboWidget, schema_signature, schema_to_wire
from dinkster_server.preflight import graph_provider_selections
from dinkster_workers import load_manifest

from dinkster.compose import compose_serving

ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "packages" / "dinkster-nodes-vision"
MANIFEST = PACKAGE / "dinkster_vision_depth_anything_v3_pack" / "dinkster-pack.toml"
MODEL_DIGEST = "blake3:c7c3ae1883d3ad41d64aa9ce2988f265fa3c437105fadc32ff2949b7e8f18323"


def test_model_depth_owner_schema_is_reused_without_wire_expansion() -> None:
    schema = OwnerModelDepthPreprocessor.schema()
    provider_schema = DepthAnything3ModelDepthPreprocessor.schema()
    provider = schema.input("provider")
    assert provider is not None and not provider.required and provider.hidden
    assert isinstance(provider.widget, ComboWidget)
    assert provider.widget.options == ()
    assert provider.widget.remote_route == f"/api/choices/{MODEL_DEPTH_PROVIDER_CHOICE}"
    assert preprocessor_choices()[MODEL_DEPTH_PROVIDER_CHOICE] == ()
    assert OwnerModelDepthPreprocessor in IMAGE_NODES
    assert provider_schema == schema
    assert schema_to_wire(provider_schema) == schema_to_wire(schema)
    assert schema_signature(provider_schema) == schema_signature(schema)


def test_depth_anything_v3_pack_declares_cpu_provider_and_pinned_model() -> None:
    manifest = load_manifest(MANIFEST)
    assert manifest.name == "dinkster-vision-depth-anything-v3"
    assert manifest.namespaces == ()
    assert manifest.executes == ("dinkster.preprocess.model_depth",)
    assert manifest.sandbox.gpu is False
    assert manifest.sandbox.network is False
    assert manifest.sandbox.writable_mounts is False
    assert manifest.requires == (
        "numpy==2.5.1",
        "opencv-python-headless==5.0.0.93",
        "pillow==12.0.0",
        "safetensors==0.8.0",
        "torch==2.13.0",
        "transformers==5.16.1",
    )
    assert len(manifest.vision_providers) == 1
    provider = manifest.vision_providers[0]
    assert provider.choice == MODEL_DEPTH_PROVIDER_CHOICE
    assert provider.node == "dinkster.preprocess.model_depth"
    assert provider.devices == ("cpu",)
    assert provider.dtypes == ("float32",)
    assert provider.batching == "per-image"
    assert provider.artifacts == ("depth-anything-3-mono-large",)
    assert provider.model == "depth-anything-v3"
    assert len(manifest.assets) == 1
    asset = manifest.assets[0]
    assert asset.need.digest == MODEL_DIGEST
    assert asset.need.size == 1_336_748_056
    assert "Apache-2.0" in asset.need.name
    source = asset.need.sources[0]
    assert isinstance(source, RemoteSource)
    assert source.url == (
        "https://huggingface.co/Comfy-Org/Depth-Anything-3/resolve/"
        "913d1a7ab58ddbfbf94a33bf536c4fc2ff25465d/geometry_estimation/"
        "depth_anything_3_mono_large.safetensors"
    )


def test_depth_anything_v3_wheel_contains_pack_runtime_and_manifest() -> None:
    project = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    included = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert (
        included["dinkster_vision_depth_anything_v3_pack"]
        == "dinkster_vision_depth_anything_v3_pack"
    )
    assert (
        included["src/dinkster_nodes_vision/depth_anything_v3"]
        == "dinkster_vision_depth_anything_v3_pack/dinkster_nodes_vision/depth_anything_v3"
    )


def test_depth_anything_v3_populates_owner_choice_and_scopes_model() -> None:
    async def scenario() -> None:
        composition = await compose_serving()
        try:
            assert composition.choices[MODEL_DEPTH_PROVIDER_CHOICE] == (
                "dinkster-vision-depth-anything-v2",
                "dinkster-vision-depth-anything-v3",
            )
            needs = composition.asset_catalog.needs_for_nodes(
                (),
                provider_selections={
                    ("dinkster.preprocess.model_depth", "dinkster-vision-depth-anything-v3"),
                },
            )
            assert tuple(needs) == (MODEL_DIGEST,)
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_depth_model_choice_resolves_provider_without_changing_stored_graph() -> None:
    async def scenario() -> None:
        composition = await compose_serving()
        try:
            resolver = (
                composition.make_engine(lambda _event: None).pin_execution().resolve_providers
            )
            assert resolver is not None
            graph = Graph(
                nodes={
                    "depth": GraphNode(
                        "dinkster.preprocess.model_depth",
                        {"image": "fixture", "model": "depth-anything-v3"},
                    )
                }
            )
            resolved = resolver(graph)
            assert "provider" not in graph.nodes["depth"].inputs
            assert resolved.nodes["depth"].inputs["provider"] == (
                "dinkster-vision-depth-anything-v3"
            )
            selections, linked = graph_provider_selections(resolved)
            assert linked == set()
            needs = composition.asset_catalog.needs_for_nodes(
                (),
                provider_selections=selections,
            )
            assert set(needs) == {MODEL_DIGEST}

            legacy = Graph(
                nodes={
                    "depth": GraphNode(
                        "dinkster.preprocess.model_depth",
                        {
                            "image": "fixture",
                            "model": "depth-anything-v3",
                            "provider": "dinkster-vision-depth-anything-v2",
                        },
                    )
                }
            )
            assert resolver(legacy).nodes["depth"].inputs["provider"] == (
                "dinkster-vision-depth-anything-v2"
            )
            unavailable = Graph(
                nodes={
                    "depth": GraphNode(
                        "dinkster.preprocess.model_depth",
                        {"image": "fixture", "model": "unknown-model"},
                    )
                }
            )
            with pytest.raises(
                ValueError,
                match="compatible vision-processing implementation is unavailable",
            ):
                resolver(unavailable)
        finally:
            await composition.close()

    asyncio.run(scenario())
