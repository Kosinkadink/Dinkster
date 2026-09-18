from __future__ import annotations

import asyncio
import json
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import cast

from dinkster_nodes_image import (
    IMAGE_NODES,
    MODEL_DEPTH_PROVIDER_CHOICE,
    preprocessor_choices,
)
from dinkster_nodes_image import (
    ModelDepthPreprocessor as OwnerModelDepthPreprocessor,
)
from dinkster_schema import (
    ComboWidget,
    comfy_alias_registry_from_wire,
    schema_signature,
    validate_replacement_references,
)
from dinkster_vision_depth_anything_v2 import (
    ModelDepthPreprocessor as ProviderModelDepthPreprocessor,
)
from dinkster_workers import load_manifest

from dinkster.compose import compose_serving

ROOT = Path(__file__).parent.parent
PACKAGE = ROOT / "packages" / "dinkster-vision-depth-anything-v2"
MANIFEST = PACKAGE / "dinkster-pack.toml"
INACTIVE_ALIASES = PACKAGE / "comfy-aliases.inactive.json"
CORE_IMAGE_ALIASES = ROOT / "packages" / "dinkster-nodes-image" / "comfy-aliases.json"
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
    assert manifest.comfy_aliases is None


def test_depth_aliases_are_provider_local_and_inactive() -> None:
    wire = cast("dict[str, object]", json.loads(INACTIVE_ALIASES.read_text(encoding="utf-8")))
    registry = comfy_alias_registry_from_wire(wire)
    inactive_ids = {record.id for record in registry.records}
    assert inactive_ids == {
        "comfy_alias:comfyui_controlnet_aux/AIO_Preprocessor",
        "comfy_alias:comfyui_controlnet_aux/DepthAnythingV2Preprocessor",
    }
    core_wire = cast(
        "dict[str, object]", json.loads(CORE_IMAGE_ALIASES.read_text(encoding="utf-8"))
    )
    core_records = cast("list[dict[str, object]]", core_wire["records"])
    assert inactive_ids.isdisjoint(record["id"] for record in core_records)
    source_schemas = {
        snapshot.schema.node_type: snapshot.schema for snapshot in registry.source_schemas
    }
    assert set(source_schemas) == {
        "comfy.comfyui_controlnet_aux.AIO_Preprocessor",
        "comfy.comfyui_controlnet_aux.DepthAnythingV2Preprocessor",
    }
    direct_checkpoint = source_schemas[
        "comfy.comfyui_controlnet_aux.DepthAnythingV2Preprocessor"
    ].input("ckpt_name")
    assert direct_checkpoint is not None
    assert direct_checkpoint.default == "depth_anything_v2_vitl.pth"
    assert isinstance(direct_checkpoint.widget, ComboWidget)
    assert direct_checkpoint.widget.options == (
        "depth_anything_v2_vitg.pth",
        "depth_anything_v2_vitl.pth",
        "depth_anything_v2_vitb.pth",
        "depth_anything_v2_vits.pth",
    )
    aio_selector = source_schemas["comfy.comfyui_controlnet_aux.AIO_Preprocessor"].input(
        "preprocessor"
    )
    assert aio_selector is not None and aio_selector.default == "none"
    assert isinstance(aio_selector.widget, ComboWidget)
    assert len(aio_selector.widget.options) == 47
    assert aio_selector.widget.options[0] == "none"
    assert aio_selector.widget.options[34] == "DepthAnythingV2Preprocessor"
    assert aio_selector.widget.options[41:43] == ("DWPreprocessor", "AnimalPosePreprocessor")
    target = OwnerModelDepthPreprocessor.schema()
    schemas = {**source_schemas, target.node_type: target}
    for record in registry.records:
        schemas[record.source.node_type] = replace(
            source_schemas[record.source.node_type],
            replacements=(record.replacement,),
        )
    assert validate_replacement_references(schemas) == ()

    records: dict[str, dict[str, object]] = {}
    for record in cast("list[dict[str, object]]", wire["records"]):
        source = cast("dict[str, object]", record["source"])
        records[cast("str", source["nodeClass"])] = record
    direct_replacement = cast(
        "dict[str, object]", records["DepthAnythingV2Preprocessor"]["replacement"]
    )
    direct_case = cast("list[dict[str, object]]", direct_replacement["cases"])[0]
    direct_inputs = cast("dict[str, object]", direct_case["inputs"])
    direct_provider = cast("dict[str, object]", direct_inputs["provider"])
    direct_transform = cast("dict[str, object]", direct_provider["transform"])
    aio_replacement = cast("dict[str, object]", records["AIO_Preprocessor"]["replacement"])
    aio_case = cast("list[dict[str, object]]", aio_replacement["cases"])[0]
    aio_inputs = cast("dict[str, object]", aio_case["inputs"])
    aio_provider = cast("dict[str, object]", aio_inputs["provider"])
    aio_transform = cast("dict[str, object]", aio_provider["transform"])
    assert direct_transform["map"] == {
        "depth_anything_v2_vitl.pth": "dinkster-vision-depth-anything-v2"
    }
    assert aio_transform["map"] == {
        "DepthAnythingV2Preprocessor": "dinkster-vision-depth-anything-v2"
    }


def test_depth_provider_wheel_contains_pack_runtime_and_manifest() -> None:
    project = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    included = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["dinkster-pack.toml"] == (
        "dinkster_vision_depth_anything_v2_pack/dinkster-pack.toml"
    )
    assert included["comfy-aliases.inactive.json"] == (
        "dinkster_vision_depth_anything_v2_pack/comfy-aliases.inactive.json"
    )
    assert included["src/dinkster_vision_depth_anything_v2"] == (
        "dinkster_vision_depth_anything_v2_pack/dinkster_vision_depth_anything_v2"
    )


def test_depth_provider_populates_owner_choice_and_scopes_its_model() -> None:
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
                    ("dinkster.preprocess.model_depth", "dinkster-vision-depth-anything-v2"),
                },
            )
            assert tuple(needs) == (MODEL_DIGEST,)
        finally:
            await composition.close()

    asyncio.run(scenario())
