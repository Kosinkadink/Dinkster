from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

from dinkster_nodes_image import (
    IMAGE_NODES,
    MODEL_DEPTH_PROVIDER_CHOICE,
    preprocessor_choices,
)
from dinkster_nodes_image import (
    ModelDepthPreprocessor as OwnerModelDepthPreprocessor,
)
from dinkster_nodes_vision.depth_anything_v2 import (
    ModelDepthPreprocessor as ProviderModelDepthPreprocessor,
)
from dinkster_schema import (
    ComboWidget,
    schema_signature,
)
from dinkster_workers import load_manifest

from dinkster.compose import compose_serving, default_pack_spec

ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "packages" / "dinkster-nodes-vision"
MANIFEST = PACKAGE / "dinkster_vision_depth_anything_v2_pack" / "dinkster-pack.toml"
MODEL_DIGEST = "blake3:e577785fc18ba89b5ae681d69574f975e6f61b329aa14b4a58d68f12dda00c29"


def test_model_depth_owner_schema_is_stable_and_provider_populated() -> None:
    schema = OwnerModelDepthPreprocessor.schema()
    provider = schema.input("provider")
    assert provider is not None and not provider.required and provider.hidden
    assert isinstance(provider.widget, ComboWidget)
    assert provider.widget.options == ()
    assert provider.widget.remote_route == f"/api/choices/{MODEL_DEPTH_PROVIDER_CHOICE}"
    assert preprocessor_choices()[MODEL_DEPTH_PROVIDER_CHOICE] == ()
    assert OwnerModelDepthPreprocessor in IMAGE_NODES
    assert schema_signature(ProviderModelDepthPreprocessor.schema()) == schema_signature(schema)


def test_depth_provider_declares_isolated_cpu_execution_and_pinned_model() -> None:
    manifest = load_manifest(MANIFEST)
    assert manifest.name == "dinkster-vision-depth-anything-v2"
    assert manifest.namespaces == ()
    assert manifest.executes == ("dinkster.preprocess.model_depth",)
    assert manifest.sandbox.gpu is False
    assert manifest.requires == (
        "numpy==2.5.1",
        "opencv-python-headless==5.0.0.93",
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
    assert provider.artifacts == ("depth-anything-v2-large",)
    assert provider.model == "depth-anything-v2-large"
    assert len(manifest.assets) == 1
    assert manifest.assets[0].need.digest == MODEL_DIGEST
    assert manifest.assets[0].need.size == 1_341_322_868
    assert "CC-BY-NC-4.0" in manifest.assets[0].need.name
    assert tuple(source.to_wire() for source in manifest.assets[0].need.sources) == (
        {
            "type": "remote",
            "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf/"
            "resolve/7581137eff8d4e94f6e796d3baea0e9fa79b22d2/model.safetensors",
        },
    )


def test_depth_provider_wheel_contains_pack_runtime_and_manifest() -> None:
    project = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    included = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["dinkster_vision_depth_anything_v2_pack"] == (
        "dinkster_vision_depth_anything_v2_pack"
    )
    assert included["src/dinkster_nodes_vision/depth_anything_v2"] == (
        "dinkster_vision_depth_anything_v2_pack/dinkster_nodes_vision/depth_anything_v2"
    )


def test_depth_provider_populates_owner_choice_and_scopes_its_model() -> None:
    async def scenario() -> None:
        composition = await compose_serving(
            [
                default_pack_spec("dinkster-nodes-foundation"),
                default_pack_spec("dinkster-nodes-media-io"),
                default_pack_spec("dinkster-nodes-image"),
                default_pack_spec("dinkster-vision-depth-anything-v2"),
                default_pack_spec("dinkster-vision-depth-anything-v3"),
            ],
            include_default_packs=False,
        )
        try:
            assert composition.choices[MODEL_DEPTH_PROVIDER_CHOICE] == (
                "dinkster-vision-depth-anything-v2",
                "dinkster-vision-depth-anything-v3",
            )
            needs = composition.asset_catalog.needs_for_nodes(
                (),
                provider_selections={
                    ("dinkster.preprocess.model_depth", "dinkster-vision-depth-anything-v2"),
                },
            )
            assert tuple(needs) == (MODEL_DIGEST,)
        finally:
            await composition.close()

    asyncio.run(scenario())
