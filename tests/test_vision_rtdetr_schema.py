from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

from dinkster_assets import RemoteSource
from dinkster_graph import Graph, GraphNode
from dinkster_nodes_image import (
    DETECT_PROVIDER_CHOICE,
    IMAGE_NODES,
    vision_choices,
)
from dinkster_nodes_image import DetectObjects as OwnerDetectObjects
from dinkster_schema import ComboWidget, schema_signature, schema_to_wire
from dinkster_vision_rtdetr import DetectObjects as RTDETRDetectObjects
from dinkster_workers import load_manifest

from dinkster.compose import compose_serving

ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "packages" / "dinkster-vision-rtdetr"
MANIFEST = PACKAGE / "dinkster-pack.toml"
MODEL_DIGEST = "blake3:5eaa01a6d16d654d9a4991ab1dfe489b580acc4b939cd1963ac6d12ceb9dc7f8"


def test_detect_owner_schema_is_reused_without_wire_expansion() -> None:
    schema = OwnerDetectObjects.schema()
    provider_schema = RTDETRDetectObjects.schema()
    provider = schema.input("provider")
    assert provider is not None and not provider.required and provider.hidden
    assert isinstance(provider.widget, ComboWidget)
    assert provider.widget.options == ()
    assert provider.widget.remote_route == f"/api/choices/{DETECT_PROVIDER_CHOICE}"
    assert vision_choices()[DETECT_PROVIDER_CHOICE] == ()
    assert OwnerDetectObjects in IMAGE_NODES
    assert provider_schema == schema
    assert schema_to_wire(provider_schema) == schema_to_wire(schema)
    assert schema_signature(provider_schema) == schema_signature(schema)


def test_pack_declares_isolated_cpu_provider_and_pinned_fp16_model() -> None:
    manifest = load_manifest(MANIFEST)
    assert manifest.name == "dinkster-vision-rtdetr"
    assert manifest.namespaces == ()
    assert manifest.executes == ("dinkster.detection.detect",)
    assert manifest.sandbox.gpu is False
    assert manifest.sandbox.network is False
    assert manifest.sandbox.writable_mounts is False
    assert manifest.requires == (
        "numpy==2.5.1",
        "safetensors==0.8.0",
        "torch==2.13.0",
    )
    assert len(manifest.vision_providers) == 1
    provider = manifest.vision_providers[0]
    assert provider.choice == DETECT_PROVIDER_CHOICE
    assert provider.node == "dinkster.detection.detect"
    assert provider.devices == ("cpu",)
    assert provider.dtypes == ("float32",)
    assert provider.batching == "batch"
    assert provider.artifacts == ("rtdetr-v4-x-hgnet-fp16",)
    assert provider.model == "rtdetr-v4-x-hgnet"
    assert len(manifest.assets) == 1
    asset = manifest.assets[0]
    assert asset.need.digest == MODEL_DIGEST
    assert asset.need.size == 123_968_978
    assert "fp16" in asset.need.name
    source = asset.need.sources[0]
    assert isinstance(source, RemoteSource)
    assert source.url == (
        "https://huggingface.co/Comfy-Org/SDPose/resolve/"
        "f122ac7976997885e3bfeab2bb3a537a6bc250bc/"
        "diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors"
    )


def test_wheel_contains_pack_runtime_and_manifest() -> None:
    project = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    included = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["dinkster-pack.toml"] == "dinkster_vision_rtdetr_pack/dinkster-pack.toml"
    assert (
        included["src/dinkster_vision_rtdetr"]
        == "dinkster_vision_rtdetr_pack/dinkster_vision_rtdetr"
    )


def test_provider_populates_owner_choice_and_scopes_model() -> None:
    async def scenario() -> None:
        composition = await compose_serving()
        try:
            assert composition.choices[DETECT_PROVIDER_CHOICE] == (
                "dinkster-vision-detr",
                "dinkster-vision-rtdetr",
                "dinkster-vision-sam31",
            )
            assert tuple(composition.asset_catalog.needs_for_nodes(())) == ()
            needs = composition.asset_catalog.needs_for_nodes(
                (),
                provider_selections={
                    ("dinkster.detection.detect", "dinkster-vision-rtdetr"),
                },
            )
            assert tuple(needs) == (MODEL_DIGEST,)
        finally:
            await composition.close()

    asyncio.run(scenario())


def test_detection_model_choice_resolves_rtdetr_provider_without_mutating_graph() -> None:
    async def scenario() -> None:
        composition = await compose_serving()
        try:
            resolver = (
                composition.make_engine(lambda _event: None).pin_execution().resolve_providers
            )
            assert resolver is not None
            graph = Graph(
                nodes={
                    "detect": GraphNode(
                        "dinkster.detection.detect",
                        {"image": "fixture", "model": "rtdetr-v4-x-hgnet"},
                    )
                }
            )

            resolved = resolver(graph)

            assert "provider" not in graph.nodes["detect"].inputs
            assert resolved.nodes["detect"].inputs["provider"] == "dinkster-vision-rtdetr"
        finally:
            await composition.close()

    asyncio.run(scenario())
