from __future__ import annotations

import asyncio
import inspect
import sys
import tomllib
from pathlib import Path

from dinkster_model_qwen_image import QWEN_IMAGE_MODEL_NODE_IDS, QWEN_IMAGE_MODEL_NODES
from dinkster_schema import NumberWidget, build_schemas
from dinkster_workers import load_manifest

from dinkster.compose import PackSpec, ServingComposer, default_pack_spec

PACKAGE = Path(__file__).parent.parent / "packages" / "dinkster-model-qwen-image"
MANIFEST = PACKAGE / "dinkster-pack.toml"
GENERATION_MANIFEST = (
    Path(__file__).parent.parent / "packages" / "dinkster-nodes-generation" / "dinkster-pack.toml"
)


def test_qwen_image_manifest_declares_native_requirements() -> None:
    manifest = load_manifest(MANIFEST)

    assert manifest.name == "dinkster-model-qwen-image"
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
        ("dinkster.model-families", "dinkster.qwen_image"),
        ("dinkster.samplers", "dinkster.euler"),
        ("dinkster.schedulers", "dinkster.simple"),
    }


def test_qwen_image_package_does_not_import_the_generation_owner() -> None:
    metadata = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["dependencies"] == [
        "dinkster-api",
        "dinkster-inference",
        "dinkster-inference-torch",
        "numpy>=1.26",
    ]
    assert not any(
        "dinkster_nodes_generation" in path.read_text(encoding="utf-8")
        for path in (PACKAGE / "src").rglob("*.py")
    )


def test_qwen_image_schemas_publish_edit_and_layered_boundaries() -> None:
    schemas = build_schemas(QWEN_IMAGE_MODEL_NODES)
    assert (
        tuple(schemas)
        == QWEN_IMAGE_MODEL_NODE_IDS
        == (
            "dinkster.load_qwen_image_control",
            "dinkster.apply_qwen_image_control",
            "dinkster.load_qwen_image_diffsynth",
            "dinkster.apply_qwen_image_diffsynth",
            "dinkster.qwen_image_edit_encode",
            "dinkster.qwen_image_edit_plus_encode",
            "dinkster.empty_qwen_image_layered_latent",
        )
    )
    control_loader = schemas["dinkster.load_qwen_image_control"]
    control_apply = schemas["dinkster.apply_qwen_image_control"]
    assert control_loader.outputs[0].type.types == ("dinkster.qwen_image_control",)
    assert {item.id: item.type.types for item in control_apply.inputs} == {
        "model": ("dinkster.model",),
        "control": ("dinkster.qwen_image_control",),
        "hint": ("dinkster.latent",),
        "strength": ("core.float",),
        "start_percent": ("core.float",),
        "end_percent": ("core.float",),
    }
    assert control_apply.outputs[0].type.types == ("dinkster.model",)

    diffsynth_loader = schemas["dinkster.load_qwen_image_diffsynth"]
    diffsynth_apply = schemas["dinkster.apply_qwen_image_diffsynth"]
    assert diffsynth_loader.outputs[0].type.types == ("dinkster.qwen_image_diffsynth",)
    assert {item.id: item.type.types for item in diffsynth_apply.inputs} == {
        "model": ("dinkster.model",),
        "patch": ("dinkster.qwen_image_diffsynth",),
        "hint": ("dinkster.latent",),
        "strength": ("core.float",),
    }
    assert diffsynth_apply.outputs[0].type.types == ("dinkster.model",)

    edit = schemas["dinkster.qwen_image_edit_encode"]
    assert edit.display_name == "TextEncodeQwenImageEdit"
    assert edit.aliases == ("TextEncodeQwenImageEdit",)
    assert {item.id: item.type.types for item in edit.inputs} == {
        "clip": ("dinkster.clip",),
        "prompt": ("core.string",),
        "vae": ("dinkster.vae",),
        "image": ("dinkster.image",),
    }
    assert {item.id for item in edit.inputs if not item.required} == {"vae", "image"}
    assert {item.id: item.type.types for item in edit.outputs} == {
        "conditioning": ("dinkster.conditioning",)
    }

    plus = schemas["dinkster.qwen_image_edit_plus_encode"]
    assert plus.display_name == "TextEncodeQwenImageEditPlus"
    assert plus.aliases == ("TextEncodeQwenImageEditPlus",)
    assert {item.id: item.type.types for item in plus.inputs} == {
        "clip": ("dinkster.clip",),
        "prompt": ("core.string",),
        "vae": ("dinkster.vae",),
        "image1": ("dinkster.image",),
        "image2": ("dinkster.image",),
        "image3": ("dinkster.image",),
    }
    assert {item.id for item in plus.inputs if not item.required} == {
        "vae",
        "image1",
        "image2",
        "image3",
    }

    layered = schemas["dinkster.empty_qwen_image_layered_latent"]
    assert layered.aliases == ("EmptyQwenImageLayeredLatentImage",)
    inputs = {item.id: item for item in layered.inputs}
    assert inputs["width"].widget == NumberWidget(min=16, max=16384, step=16)
    assert inputs["height"].widget == NumberWidget(min=16, max=16384, step=16)
    assert inputs["layers"].widget == NumberWidget(min=0, max=4096, step=1)
    assert inputs["batch_size"].widget == NumberWidget(min=1, max=4096, step=1)


def test_qwen_image_schemas_compose_with_universal_generation() -> None:
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
            assert tuple(delta.schemas) == QWEN_IMAGE_MODEL_NODE_IDS
            assert composer.incomplete_generation_removals()["dinkster-model-qwen-image"] == ()
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_qwen_image_bodies_keep_torch_provider_lazy() -> None:
    assert "dinkster_model_qwen_image.provider" not in sys.modules
    assert tuple(
        tuple(inspect.signature(node.execute).parameters) for node in QWEN_IMAGE_MODEL_NODES
    ) == (
        ("control_net",),
        ("model", "control", "hint", "strength", "start_percent", "end_percent"),
        ("model_patch",),
        ("model", "patch", "hint", "strength"),
        ("clip", "prompt", "vae", "image"),
        ("clip", "prompt", "vae", "image1", "image2", "image3"),
        ("width", "height", "layers", "batch_size"),
    )
