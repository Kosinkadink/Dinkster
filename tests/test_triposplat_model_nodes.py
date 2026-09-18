from __future__ import annotations

import asyncio
import inspect
import sys
import tomllib
from pathlib import Path

from dinkster_model_triposplat import TRIPOSPLAT_MODEL_NODE_IDS, TRIPOSPLAT_MODEL_NODES
from dinkster_schema import NumberWidget, build_schemas
from dinkster_workers import load_manifest

from dinkster.compose import PackSpec, ServingComposer, default_pack_spec

PACKAGE = Path(__file__).parent.parent / "packages" / "dinkster-model-triposplat"
MANIFEST = PACKAGE / "dinkster-pack.toml"
GENERATION_MANIFEST = (
    Path(__file__).parent.parent / "packages" / "dinkster-nodes-generation" / "dinkster-pack.toml"
)


def test_triposplat_manifest_declares_native_requirements() -> None:
    manifest = load_manifest(MANIFEST)

    assert manifest.name == "dinkster-model-triposplat"
    assert manifest.schema_only == ()
    assert manifest.executes == ()
    assert manifest.capabilities == ()
    assert [(item.pack, item.version) for item in manifest.dependencies] == [
        ("dinkster-nodes-media-io", "<1,>=0.0.1")
    ]
    assert [(item.id, item.version) for item in manifest.requirements.capabilities] == [
        ("dinkster.generation.schemas", "<2,>=1.0.0")
    ]
    assert {(item.registry, item.id) for item in manifest.requirements.registry} == {
        ("dinkster.model-families", "dinkster.triposplat"),
        ("dinkster.samplers", "dinkster.euler"),
        ("dinkster.schedulers", "dinkster.simple"),
    }


def test_triposplat_package_does_not_import_the_generation_owner() -> None:
    metadata = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["dependencies"] == [
        "dinkster-api",
        "dinkster-inference",
        "dinkster-inference-torch",
        "numpy>=1.26",
        "pillow>=10",
    ]
    assert not any(
        "dinkster_nodes_generation" in path.read_text(encoding="utf-8")
        for path in (PACKAGE / "src").rglob("*.py")
    )


def test_triposplat_schemas_publish_component_and_splat_boundaries() -> None:
    schemas = build_schemas(TRIPOSPLAT_MODEL_NODES)
    assert (
        tuple(schemas)
        == TRIPOSPLAT_MODEL_NODE_IDS
        == (
            "dinkster.load_triposplat_vision_encoder",
            "dinkster.load_triposplat_decoder",
            "dinkster.triposplat_preprocess_image",
            "dinkster.triposplat_conditioning",
            "dinkster.triposplat_decode",
        )
    )
    vision_loader = schemas["dinkster.load_triposplat_vision_encoder"]
    decoder_loader = schemas["dinkster.load_triposplat_decoder"]
    assert vision_loader.inputs[0].type.types == ("dinkster.asset",)
    assert vision_loader.outputs[0].type.types == ("dinkster.triposplat_vision",)
    assert decoder_loader.inputs[0].type.types == ("dinkster.asset",)
    assert decoder_loader.outputs[0].type.types == ("dinkster.triposplat_decoder",)
    assert vision_loader.aliases == ()
    assert decoder_loader.aliases == ()

    preprocess = schemas["dinkster.triposplat_preprocess_image"]
    assert {item.id: item.type.types for item in preprocess.inputs} == {
        "image": ("dinkster.image",),
        "mask": ("dinkster.mask",),
        "erode_radius": ("core.int",),
        "size": ("core.int",),
    }
    assert preprocess.outputs[0].type.types == ("dinkster.image",)
    preprocess_inputs = {item.id: item for item in preprocess.inputs}
    assert preprocess_inputs["erode_radius"].widget == NumberWidget(min=0, max=16, step=1)
    assert preprocess_inputs["size"].widget == NumberWidget(min=256, max=4096, step=16)

    conditioning = schemas["dinkster.triposplat_conditioning"]
    assert {item.id: item.type.types for item in conditioning.inputs} == {
        "vision": ("dinkster.triposplat_vision",),
        "vae": ("dinkster.vae",),
        "image": ("dinkster.image",),
    }
    assert {item.id: item.type.types for item in conditioning.outputs} == {
        "positive": ("dinkster.conditioning",),
        "negative": ("dinkster.conditioning",),
        "latent": ("dinkster.latent",),
    }

    decode = schemas["dinkster.triposplat_decode"]
    assert {item.id: item.type.types for item in decode.inputs} == {
        "samples": ("dinkster.latent",),
        "decoder": ("dinkster.triposplat_decoder",),
        "num_gaussians": ("core.int",),
        "seed": ("core.int",),
    }
    decode_inputs = {item.id: item for item in decode.inputs}
    assert decode_inputs["num_gaussians"].widget == NumberWidget(min=32768, max=1048576, step=32)
    assert decode_inputs["seed"].widget == NumberWidget(min=0, control_after_generate="randomize")
    assert decode.outputs[0].type.types == ("dinkster.splat",)
    assert decode.outputs[0].preview is True


def test_triposplat_schemas_compose_with_universal_generation() -> None:
    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(default_pack_spec("dinkster-nodes-foundation"))
            await composer.add_pack(default_pack_spec("dinkster-nodes-media-io"))
            await composer.add_pack(
                PackSpec(GENERATION_MANIFEST, trust_reserved=True, in_process=True)
            )
            delta = await composer.add_pack(
                PackSpec(MANIFEST, trust_reserved=True, in_process=True)
            )
            assert tuple(delta.schemas) == TRIPOSPLAT_MODEL_NODE_IDS
            assert composer.incomplete_generation_removals()["dinkster-model-triposplat"] == ()
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_triposplat_bodies_keep_torch_provider_lazy() -> None:
    assert "dinkster_model_triposplat.provider" not in sys.modules
    assert tuple(
        tuple(inspect.signature(node.execute).parameters) for node in TRIPOSPLAT_MODEL_NODES
    ) == (
        ("vision_encoder",),
        ("decoder",),
        ("image", "mask", "erode_radius", "size"),
        ("vision", "vae", "image"),
        ("samples", "decoder", "num_gaussians", "seed"),
    )
