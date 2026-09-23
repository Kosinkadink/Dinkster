from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

from dinkster_nodes_image import (
    IMAGE_NODES,
    UPSCALE_MODEL_PROVIDER_CHOICE,
    upscale_choices,
)
from dinkster_nodes_image import (
    UpscaleWithModel as OwnerUpscaleWithModel,
)
from dinkster_nodes_vision.upscale import UpscaleWithModel as ProviderUpscaleWithModel
from dinkster_schema import ComboWidget, schema_signature
from dinkster_workers import load_manifest

from dinkster.compose import compose_serving

ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "packages" / "dinkster-nodes-vision"
MANIFEST = PACKAGE / "dinkster_vision_upscale_pack" / "dinkster-pack.toml"


def test_upscale_owner_schema_is_stable_and_provider_populated() -> None:
    schema = OwnerUpscaleWithModel.schema()
    provider = schema.input("provider")
    assert provider is not None and not provider.required and provider.hidden
    assert isinstance(provider.widget, ComboWidget)
    assert provider.widget.options == ()
    assert provider.widget.remote_route == f"/api/choices/{UPSCALE_MODEL_PROVIDER_CHOICE}"
    assert upscale_choices() == {UPSCALE_MODEL_PROVIDER_CHOICE: ()}
    assert OwnerUpscaleWithModel in IMAGE_NODES
    assert schema_signature(ProviderUpscaleWithModel.schema()) == schema_signature(schema)


def test_upscale_pack_declares_isolated_cpu_provider_without_artifacts() -> None:
    manifest = load_manifest(MANIFEST)
    assert manifest.name == "dinkster-vision-upscale"
    assert manifest.namespaces == ()
    assert manifest.executes == ("dinkster.image.upscale_model",)
    assert manifest.sandbox.gpu is False
    assert manifest.requires == (
        "numpy==2.5.1",
        "torch==2.13.0",
    )
    assert len(manifest.vision_providers) == 1
    provider = manifest.vision_providers[0]
    assert provider.choice == UPSCALE_MODEL_PROVIDER_CHOICE
    assert provider.node == "dinkster.image.upscale_model"
    assert provider.devices == ("cpu",)
    assert provider.dtypes == ("float32",)
    assert provider.batching == "batch"
    assert provider.artifacts == ()
    assert manifest.assets == ()


def test_upscale_wheel_contains_pack_runtime_and_manifest() -> None:
    project = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    included = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["dinkster_vision_upscale_pack"] == "dinkster_vision_upscale_pack"
    assert (
        included["src/dinkster_nodes_vision/upscale"]
        == "dinkster_vision_upscale_pack/src/dinkster_nodes_vision/upscale"
    )


def test_upscale_provider_populates_owner_choice_with_no_asset_needs() -> None:
    async def scenario() -> None:
        composition = await compose_serving()
        try:
            assert composition.choices[UPSCALE_MODEL_PROVIDER_CHOICE] == (
                "dinkster-vision-upscale",
            )
            needs = composition.asset_catalog.needs_for_nodes(
                (),
                provider_selections={
                    ("dinkster.image.upscale_model", "dinkster-vision-upscale"),
                },
            )
            assert tuple(needs) == ()
        finally:
            await composition.close()

    asyncio.run(scenario())
