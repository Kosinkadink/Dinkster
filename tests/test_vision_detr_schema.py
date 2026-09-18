from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

from dinkster_nodes_image import (
    DETECT_PROVIDER_CHOICE,
    IMAGE_NODES,
    vision_choices,
)
from dinkster_nodes_image import (
    DetectObjects as OwnerDetectObjects,
)
from dinkster_schema import ComboWidget, schema_signature
from dinkster_vision_detr import DetectObjects as DetrDetectObjects
from dinkster_workers import load_manifest

from dinkster.compose import compose_serving

ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "packages" / "dinkster-vision-detr"
MANIFEST = PACKAGE / "dinkster-pack.toml"
MODEL_DIGEST = "blake3:2bb221c9ab83ea68d6a66bdc4cfe7bce4c49a1784287f5923521d34d23d150a2"


def test_detect_owner_schema_is_stable_and_provider_populated() -> None:
    schema = OwnerDetectObjects.schema()
    provider = schema.input("provider")
    assert provider is not None and not provider.required and provider.hidden
    assert isinstance(provider.widget, ComboWidget)
    assert provider.widget.options == ()
    assert provider.widget.remote_route == f"/api/choices/{DETECT_PROVIDER_CHOICE}"
    assert vision_choices()[DETECT_PROVIDER_CHOICE] == ()
    assert OwnerDetectObjects in IMAGE_NODES
    assert schema_signature(DetrDetectObjects.schema()) == schema_signature(schema)


def test_detr_pack_declares_isolated_cpu_provider_and_pinned_model() -> None:
    manifest = load_manifest(MANIFEST)
    assert manifest.name == "dinkster-vision-detr"
    assert manifest.namespaces == ()
    assert manifest.executes == ("dinkster.detection.detect",)
    assert manifest.sandbox.gpu is False
    assert manifest.requires == (
        "numpy==2.5.1",
        "torch==2.13.0",
    )
    assert len(manifest.vision_providers) == 1
    provider = manifest.vision_providers[0]
    assert provider.choice == DETECT_PROVIDER_CHOICE
    assert provider.node == "dinkster.detection.detect"
    assert provider.devices == ("cpu",)
    assert provider.dtypes == ("float32",)
    assert provider.batching == "per-image"
    assert provider.artifacts == ("detr-r50",)
    assert provider.model == "detr-resnet-50"
    assert len(manifest.assets) == 1
    assert manifest.assets[0].need.digest == MODEL_DIGEST
    assert manifest.assets[0].need.size == 166_618_694


def test_detr_wheel_contains_pack_runtime_and_manifest() -> None:
    project = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    included = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["dinkster-pack.toml"] == "dinkster_vision_detr_pack/dinkster-pack.toml"
    assert included["src/dinkster_vision_detr"] == "dinkster_vision_detr_pack/dinkster_vision_detr"


def test_detr_provider_populates_owner_choice_and_scopes_its_model() -> None:
    async def scenario() -> None:
        composition = await compose_serving()
        try:
            assert composition.choices[DETECT_PROVIDER_CHOICE] == (
                "dinkster-vision-detr",
                "dinkster-vision-rtdetr",
                "dinkster-vision-sam31",
            )
            needs = composition.asset_catalog.needs_for_nodes(
                (),
                provider_selections={
                    ("dinkster.detection.detect", "dinkster-vision-detr"),
                },
            )
            assert tuple(needs) == (MODEL_DIGEST,)
        finally:
            await composition.close()

    asyncio.run(scenario())
