from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

from dinkster_assets import RemoteSource
from dinkster_nodes_image import IMAGE_NODES, MATTE_PROVIDER_CHOICE, vision_choices
from dinkster_nodes_image import ImageMatte as OwnerImageMatte
from dinkster_nodes_vision.birefnet import ImageMatte as BiRefNetImageMatte
from dinkster_schema import ComboWidget, schema_signature, schema_to_wire
from dinkster_workers import load_manifest

from dinkster.compose import compose_serving

ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "packages" / "dinkster-nodes-vision"
MANIFEST = PACKAGE / "dinkster_vision_birefnet_pack" / "dinkster-pack.toml"
MODEL_DIGEST = "blake3:03f8793ff101fb10981ee700fe276a6f481af00cb607dfafcfee46aeb8e638db"


def test_matte_owner_schema_is_stable_and_provider_populated() -> None:
    schema = OwnerImageMatte.schema()
    provider_schema = BiRefNetImageMatte.schema()
    provider = schema.input("provider")
    assert provider is not None and not provider.required and provider.hidden
    assert isinstance(provider.widget, ComboWidget)
    assert provider.widget.options == ()
    assert provider.widget.remote_route == f"/api/choices/{MATTE_PROVIDER_CHOICE}"
    assert vision_choices()[MATTE_PROVIDER_CHOICE] == ()
    assert OwnerImageMatte in IMAGE_NODES
    assert provider_schema == schema
    assert schema_to_wire(provider_schema) == schema_to_wire(schema)
    assert schema_signature(provider_schema) == schema_signature(schema)


def test_birefnet_pack_declares_cpu_provider_and_pinned_model() -> None:
    manifest = load_manifest(MANIFEST)
    assert manifest.name == "dinkster-vision-birefnet"
    assert manifest.namespaces == ()
    assert manifest.executes == ("dinkster.image.matte",)
    assert manifest.sandbox.gpu is False
    assert manifest.sandbox.network is False
    assert manifest.sandbox.writable_mounts is False
    assert manifest.requires == (
        "dinkster-inference-torch==0.0.1",
        "numpy==2.5.1",
        "safetensors==0.8.0",
        "torch==2.13.0",
        "torchvision==0.28.0",
    )
    assert len(manifest.vision_providers) == 1
    provider = manifest.vision_providers[0]
    assert provider.choice == MATTE_PROVIDER_CHOICE
    assert provider.node == "dinkster.image.matte"
    assert provider.devices == ("cpu",)
    assert provider.dtypes == ("float32",)
    assert provider.batching == "per-image"
    assert provider.artifacts == ("birefnet-general",)
    assert len(manifest.assets) == 1
    asset = manifest.assets[0]
    assert asset.need.digest == MODEL_DIGEST
    assert asset.need.size == 444_473_596
    source = asset.need.sources[0]
    assert isinstance(source, RemoteSource)
    assert source.url == (
        "https://huggingface.co/Comfy-Org/BiRefNet/resolve/"
        "35767b272f2846752a3aee1259abdd4586f735c8/background_removal/birefnet.safetensors"
    )


def test_birefnet_wheel_contains_pack_runtime_and_manifest() -> None:
    project = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    included = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["dinkster_vision_birefnet_pack"] == "dinkster_vision_birefnet_pack"
    assert (
        included["src/dinkster_nodes_vision/birefnet"]
        == "dinkster_vision_birefnet_pack/src/dinkster_nodes_vision/birefnet"
    )


def test_birefnet_provider_populates_owner_choice_and_scopes_model() -> None:
    async def scenario() -> None:
        composition = await compose_serving()
        try:
            assert composition.choices[MATTE_PROVIDER_CHOICE] == ("dinkster-vision-birefnet",)
            needs = composition.asset_catalog.needs_for_nodes(
                (),
                provider_selections={
                    ("dinkster.image.matte", "dinkster-vision-birefnet"),
                },
            )
            assert tuple(needs) == (MODEL_DIGEST,)
        finally:
            await composition.close()

    asyncio.run(scenario())
