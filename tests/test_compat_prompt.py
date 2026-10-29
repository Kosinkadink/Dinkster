"""ComfyUI API prompt adapter: v1 prompt JSON -> native Graph + targets,
strictly at the boundary (the engine/server never learn class_type or
positional output indexes), plus the umbrella HTTP endpoint."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import AssetRef, MountDef, MountTable, digest_bytes
from dinkster_assets.integrity import digest_file_with_record as real_digest_file_with_record
from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy import (
    COMFY_INPUT_ADAPTERS,
    PromptTranslationError,
    adapt_save_image_inputs,
    extract_prompt,
    make_load_checkpoint_adapter,
    make_load_clip_adapter,
    make_load_diffusion_model_adapter,
    make_load_dual_clip_adapter,
    make_load_image_adapter,
    make_load_latent_adapter,
    make_load_lora_adapter,
    make_load_model_patch_adapter,
    make_load_vae_adapter,
    make_model_asset_inputs_adapter,
    translate_mappings,
    translate_prompt,
)
from dinkster_engine import Engine, EventListener
from dinkster_graph import Graph, GraphNode, Link, RegionNode, graph_to_wire, validate
from dinkster_protocol import AttentionPolicyConfig, CompatGateDiagnostic
from dinkster_schema import (
    AssetWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    SelectorSpec,
    TypeExpr,
    WidgetRepresentation,
    WidgetRepresentations,
    build_node_types,
    build_schemas,
)
from dinkster_server import STATE_KEY, JobQueue, Principal, create_app
from dinkster_server.queue import JobGraphAdmissionError
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker
from PIL import Image, PngImagePlugin

from dinkster.compat_api import add_comfy_compat_routes
from dinkster.mounts_api import MountService, add_mount_routes

STRING = TypeExpr.concrete("core.string")
INT = TypeExpr.concrete("core.int")
COMBO = TypeExpr.concrete("core.combo")
LATENT = TypeExpr.concrete("dinkster.latent")


class Producer(Node):
    """Two ordered outputs, so positional link indexes are observable."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.Producer",
            inputs=(InputSpec("seed", INT, required=False, default=0),),
            outputs=(OutputSpec("first", STRING), OutputSpec("second", STRING)),
            aliases=("Producer",),
        )

    @classmethod
    async def execute(cls, *, seed: int) -> Mapping[str, object]:
        return cls.outputs(first=f"first-{seed}", second=f"second-{seed}")


class Consumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.Consumer",
            inputs=(InputSpec("text", STRING),),
            outputs=(OutputSpec("out", STRING),),
            aliases=("Consumer",),
            output_node=True,
        )

    @classmethod
    async def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(out=text.upper())


class ListProducer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.ListProducer",
            inputs=(InputSpec("values", TypeExpr.list_of(INT)),),
            outputs=(OutputSpec("values", TypeExpr.list_of(INT)),),
            aliases=("ListProducer",),
        )

    @classmethod
    async def execute(cls, *, values: list[int]) -> Mapping[str, object]:
        return cls.outputs(values=values)


class PairProducer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.PairProducer",
            inputs=(
                InputSpec("left", TypeExpr.list_of(INT)),
                InputSpec("right", TypeExpr.list_of(INT)),
            ),
            outputs=(
                OutputSpec("left", TypeExpr.list_of(INT)),
                OutputSpec("right", TypeExpr.list_of(INT)),
            ),
            aliases=("PairProducer",),
        )

    @classmethod
    async def execute(cls, *, left: list[int], right: list[int]) -> Mapping[str, object]:
        return cls.outputs(left=left, right=right)


class ScaleInt(Node):
    calls: list[int] = []

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.ScaleInt",
            inputs=(InputSpec("value", INT), InputSpec("factor", INT)),
            outputs=(OutputSpec("value", INT),),
            aliases=("ScaleInt",),
        )

    @classmethod
    async def execute(cls, *, value: int, factor: int) -> Mapping[str, object]:
        cls.calls.append(value)
        return cls.outputs(value=value * factor)


class AddInts(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.AddInts",
            inputs=(InputSpec("left", INT), InputSpec("right", INT)),
            outputs=(OutputSpec("value", INT),),
            aliases=("AddInts",),
        )

    @classmethod
    async def execute(cls, *, left: int, right: int) -> Mapping[str, object]:
        return cls.outputs(value=left + right)


class ListCollector(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.ListCollector",
            inputs=(InputSpec("values", TypeExpr.list_of(INT)),),
            outputs=(OutputSpec("values", TypeExpr.list_of(INT)),),
            aliases=("ListCollector",),
            output_node=True,
        )

    @classmethod
    async def execute(cls, *, values: list[int]) -> Mapping[str, object]:
        return cls.outputs(values=values)


class MappedFanout(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.MappedFanout",
            inputs=(InputSpec("value", INT),),
            outputs=(OutputSpec("values", TypeExpr.list_of(INT)),),
            aliases=("MappedFanout",),
            output_node=True,
        )

    @classmethod
    async def execute(cls, *, value: int) -> Mapping[str, object]:
        return cls.outputs(values=[value, value])


class StringSink(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.StringSink",
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("value", STRING),),
            aliases=("StringSink",),
            output_node=True,
        )

    @classmethod
    async def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(value=value)


class LatentSink(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.LatentSink",
            inputs=(InputSpec("latent", LATENT),),
            outputs=(OutputSpec("latent", LATENT),),
            aliases=("LatentSink",),
            output_node=True,
        )

    @classmethod
    async def execute(cls, *, latent: object) -> Mapping[str, object]:
        return cls.outputs(latent=latent)


class DynamicConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.DynamicConsumer",
            combos=(
                DynamicComboSpec(
                    "combo",
                    (
                        DynamicComboOption(
                            "outer",
                            (
                                DynamicComboSpec(
                                    "subcombo",
                                    (
                                        DynamicComboOption(
                                            "inner",
                                            (InputSpec("value", INT),),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "family",
                            (
                                InputFamilySpec(
                                    "frame",
                                    INT,
                                    member_prefix="image",
                                    max_members=3,
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "names",
                            (
                                InputFamilySpec(
                                    "named",
                                    INT,
                                    member_names=("left", "right"),
                                ),
                            ),
                        ),
                        DynamicComboOption(
                            "scale dimensions",
                            (InputSpec("width", INT),),
                        ),
                    ),
                ),
            ),
            outputs=(OutputSpec("out", STRING),),
            aliases=("DynamicConsumer",),
            output_node=True,
        )

    @classmethod
    async def execute(cls, **inputs: object) -> Mapping[str, object]:
        leaf = next(
            value
            for input_id, value in inputs.items()
            if input_id not in {"combo", "combo.subcombo"}
        )
        return cls.outputs(out=str(leaf))


class FakeSaveImage(Node):
    """Stands in for the native dinkster.save_image: same node type, same
    alias, same legacy-input situation - no ComfyUI bootstrap needed to
    test the filename_prefix conversion path."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.save_image",
            inputs=(
                InputSpec("images", STRING),
                InputSpec("target", STRING, required=False, default="unset"),
            ),
            outputs=(OutputSpec("out", STRING),),
            idempotent=False,
            output_node=True,
            aliases=("SaveImage",),
        )

    @classmethod
    async def execute(cls, *, images: str, target: str) -> Mapping[str, object]:
        return cls.outputs(out=images)


class CompatMetadataSave:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    seen: list[tuple[object, object]] = []

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {
            "required": {"filename": "STRING"},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def save(
        self, filename: str, prompt: object = None, extra_pnginfo: object = None
    ) -> tuple[str]:
        self.seen.append((prompt, extra_pnginfo))
        pnginfo = PngImagePlugin.PngInfo()
        if prompt is not None:
            pnginfo.add_text("prompt", json.dumps(prompt))
        if isinstance(extra_pnginfo, Mapping):
            for key in extra_pnginfo:
                pnginfo.add_text(key, json.dumps(extra_pnginfo[key]))
        Image.new("RGB", (1, 1)).save(filename, pnginfo=pnginfo)
        return (filename,)


class CompatMetadataSaveList:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    seen: list[tuple[object, object]] = []

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {
            "required": {"filename": "STRING"},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def save(
        self, filename: list[str], prompt: object = None, extra_pnginfo: object = None
    ) -> tuple[str]:
        self.seen.append((prompt, extra_pnginfo))
        return (filename[0],)


class FakeLoadImage(Node):
    """Stands in for the native dinkster.load_image: same node type, same
    alias, same legacy-input situation - the string 'image' filename
    must become a digest-backed asset wire value at the boundary."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_image",
            inputs=(InputSpec("image", STRING),),
            outputs=(OutputSpec("out", STRING),),
            aliases=("LoadImage",),
        )

    @classmethod
    async def execute(cls, *, image: str) -> Mapping[str, object]:
        return cls.outputs(out=image)


class FakeLoadCheckpoint(Node):
    """Stands in for the native dinkster.load_checkpoint: same node type,
    same alias, same legacy-input situation - the string 'ckpt_name'
    must become a digest-backed asset wire value on the RENAMED native
    'checkpoint' input at the boundary."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_checkpoint",
            inputs=(InputSpec("checkpoint", STRING),),
            outputs=(OutputSpec("out", STRING),),
            aliases=("CheckpointLoaderSimple",),
        )

    @classmethod
    async def execute(cls, *, checkpoint: str) -> Mapping[str, object]:
        return cls.outputs(out=checkpoint)


class FakeLoadLora(Node):
    """Stands in for the native dinkster.load_lora: same node type, same
    alias, same legacy-input situation as the checkpoint - 'lora_name'
    renames to the native 'lora' asset input at the boundary while the
    strength inputs ride through untouched."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.load_lora",
            inputs=(
                InputSpec("lora", STRING),
                InputSpec("strength_model", STRING, required=False),
            ),
            outputs=(OutputSpec("out", STRING),),
            aliases=("LoraLoader",),
        )

    @classmethod
    async def execute(cls, *, lora: str, strength_model: str = "") -> Mapping[str, object]:
        return cls.outputs(out=lora)


class FakeTranslatedModelLoader(Node):
    """A translated filename-list node after model asset lowering: its
    input id stays unchanged, unlike the native replacement adapters."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.TranslatedModelLoader",
            inputs=(
                InputSpec(
                    "control_net_name",
                    TypeExpr.concrete("dinkster.asset"),
                    widget=WidgetRepresentations(
                        (
                            WidgetRepresentation(
                                "picker",
                                AssetWidget(kind="model/controlnet"),
                            ),
                            WidgetRepresentation(
                                "compact-picker",
                                AssetWidget(kind="model/controlnet"),
                            ),
                        ),
                        default="picker",
                        user_switchable=True,
                    ),
                ),
            ),
            outputs=(OutputSpec("out", STRING),),
            aliases=("TranslatedModelLoader",),
            output_node=True,
        )

    @classmethod
    async def execute(cls, *, control_net_name: object) -> Mapping[str, object]:
        return cls.outputs(out=str(control_net_name))


NODES: list[type[Node]] = [
    Producer,
    Consumer,
    ListProducer,
    PairProducer,
    ScaleInt,
    AddInts,
    ListCollector,
    MappedFanout,
    StringSink,
    LatentSink,
    DynamicConsumer,
    FakeSaveImage,
    FakeLoadImage,
    FakeLoadCheckpoint,
    FakeLoadLora,
    FakeTranslatedModelLoader,
]
SCHEMAS = build_schemas(NODES)


def ambiguous_schemas() -> dict[str, NodeSchema]:
    schemas = dict(SCHEMAS)
    for suffix in ("a", "b"):
        node_type = f"comfy.pack{suffix}.Dup"
        schemas[node_type] = NodeSchema(
            node_type=node_type,
            outputs=(OutputSpec("out", STRING),),
            aliases=("Dup",),
        )
    return schemas


# -- translate_prompt ---------------------------------------------------------


def test_prompt_translates_links_and_literals() -> None:
    prompt: dict[str, Any] = {
        "1": {"class_type": "Producer", "inputs": {"seed": 7}},
        "2": {"class_type": "Consumer", "inputs": {"text": ["1", 1]}},
    }
    translation = translate_prompt(prompt, SCHEMAS)
    assert translation.targets == ("2",)
    node = translation.graph.nodes["2"]
    assert isinstance(node, GraphNode)
    assert node.node_type == "comfy.Consumer"
    # Positional index 1 resolved to the schema's second output id.
    assert node.inputs["text"] == Link(node_id="1", output_id="second")
    assert translation.graph.nodes["1"].inputs["seed"] == 7


def test_scheduling_source_alias_lowers_to_canonical_native_id() -> None:
    from dinkster_compat_comfy.native_arm import NATIVE_SCHEDULING_NODES

    schemas = build_schemas((*NATIVE_SCHEDULING_NODES, Consumer))
    translation = translate_prompt(
        {
            "1": {"class_type": "CreateHookKeyframe", "inputs": {}},
            "2": {"class_type": "Consumer", "inputs": {"text": ["1", 0]}},
        },
        schemas,
    )

    node = translation.graph.nodes["1"]
    assert isinstance(node, GraphNode)
    assert node.node_type == "dinkster.create_hook_keyframe"


@pytest.mark.parametrize("checkpoint_name", ("sd15.safetensors", "sdxl.safetensors"))
def test_controlnet_workflow_translates_to_native_nodes_and_typed_links(
    checkpoint_name: str,
) -> None:
    from dinkster_compat_comfy.native_arm import GENERATION_PROVIDER_NODES
    from dinkster_nodes_media_io import LoadImage, SaveImage

    schemas = build_schemas((*GENERATION_PROVIDER_NODES, LoadImage, SaveImage))
    checkpoint_ref = fake_ref(checkpoint_name)
    hint_ref = fake_ref("hint.png")
    resolve = fake_resolver({checkpoint_name: checkpoint_ref, "hint.png": hint_ref})
    translated = translate_prompt(
        {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint_name},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": "positive"},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": "negative"},
            },
            "4": {"class_type": "LoadImage", "inputs": {"image": "hint.png"}},
            "5": {
                "class_type": "ControlNetLoader",
                "inputs": {"control_net_name": "canny.safetensors"},
            },
            "6": {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "control_net": ["5", 0],
                    "image": ["4", 0],
                    "strength": 0.75,
                    "start_percent": 0.1,
                    "end_percent": 0.9,
                },
            },
            "7": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 512, "height": 512, "batch_size": 1},
            },
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "seed": 29,
                    "steps": 2,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "positive": ["6", 0],
                    "negative": ["6", 1],
                    "latent_image": ["7", 0],
                    "denoise": 1.0,
                },
            },
            "9": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["8", 0], "vae": ["1", 2]},
            },
            "10": {
                "class_type": "SaveImage",
                "inputs": {"images": ["9", 0], "filename_prefix": "controlnet"},
            },
        },
        schemas,
        input_adapters={
            **COMFY_INPUT_ADAPTERS,
            "dinkster.load_checkpoint": make_load_checkpoint_adapter(resolve),
            "dinkster.load_image": make_load_image_adapter(resolve),
        },
    )

    graph = translated.graph
    checkpoint_node = graph.nodes["1"]
    loader_node = graph.nodes["5"]
    apply_node = graph.nodes["6"]
    sampler_node = graph.nodes["8"]
    assert isinstance(checkpoint_node, GraphNode)
    assert isinstance(loader_node, GraphNode)
    assert isinstance(apply_node, GraphNode)
    assert isinstance(sampler_node, GraphNode)
    assert checkpoint_node.node_type == "dinkster.load_checkpoint"
    assert checkpoint_node.inputs == {"checkpoint": checkpoint_ref.to_wire()}
    assert loader_node.node_type == "dinkster.load_controlnet"
    assert loader_node.inputs == {"control_net_name": "canny.safetensors"}
    assert apply_node.node_type == "dinkster.apply_controlnet_advanced"
    assert apply_node.inputs["control_net"] == Link("5", "control_net")
    assert apply_node.inputs["positive"] == Link("2", "conditioning")
    assert apply_node.inputs["negative"] == Link("3", "conditioning")
    assert sampler_node.inputs["positive"] == Link("6", "positive")
    assert sampler_node.inputs["negative"] == Link("6", "negative")
    latent_node = graph.nodes["7"]
    assert isinstance(latent_node, GraphNode)
    assert latent_node.node_type == "dinkster.empty_latent_image"
    assert sampler_node.inputs["latent_image"] == Link("7", "latent")
    assert translated.targets == ("10",)


def _res4lyf_sampling_schemas() -> dict[str, NodeSchema]:
    from dinkster_compat_comfy.native_arm import GENERATION_PROVIDER_NODES

    return build_schemas((*GENERATION_PROVIDER_NODES, LatentSink))


@pytest.mark.parametrize(
    ("class_type", "inputs"),
    (
        (
            "KSampler",
            {
                "model": "model",
                "seed": 17,
                "steps": 20,
                "cfg": 5.0,
                "positive": "positive",
                "negative": "negative",
                "latent_image": "latent",
                "denoise": 1.0,
            },
        ),
        (
            "KSamplerAdvanced",
            {
                "model": "model",
                "add_noise": "enable",
                "noise_seed": 17,
                "steps": 20,
                "cfg": 5.0,
                "positive": "positive",
                "negative": "negative",
                "latent_image": "latent",
                "start_at_step": 0,
                "end_at_step": 20,
                "return_with_leftover_noise": "disable",
            },
        ),
    ),
)
def test_res4lyf_ksampler_workflow_preserves_widget_names(
    class_type: str,
    inputs: dict[str, object],
) -> None:
    translated = translate_prompt(
        {
            "latent": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 768, "height": 512, "batch_size": 2},
            },
            "sampler": {
                "class_type": class_type,
                "inputs": {
                    **inputs,
                    "latent_image": ["latent", 0],
                    "sampler_name": "rk_beta",
                    "scheduler": "beta57",
                },
            },
            "sink": {"class_type": "LatentSink", "inputs": {"latent": ["sampler", 0]}},
        },
        _res4lyf_sampling_schemas(),
    )

    sampler = translated.graph.nodes["sampler"]
    assert isinstance(sampler, GraphNode)
    assert sampler.inputs["sampler_name"] == "rk_beta"
    assert sampler.inputs["scheduler"] == "beta57"
    assert sampler.inputs["latent_image"] == Link("latent", "latent")
    latent = translated.graph.nodes["latent"]
    assert isinstance(latent, GraphNode)
    assert latent.node_type == "dinkster.empty_latent_image"
    assert latent.inputs == {"width": 768, "height": 512, "batch_size": 2}
    assert translated.graph.nodes["sink"].inputs["latent"] == Link("sampler", "latent")
    assert translated.targets == ("sink",)


@pytest.mark.parametrize(
    ("class_type", "custom_inputs"),
    (
        (
            "SamplerCustom",
            {
                "model": "model",
                "add_noise": True,
                "noise_seed": 17,
                "cfg": 5.0,
                "positive": "positive",
                "negative": "negative",
            },
        ),
        (
            "SamplerCustomAdvanced",
            {"noise": "noise", "guider": "guider"},
        ),
    ),
)
def test_res4lyf_decomposed_workflow_preserves_names_and_links(
    class_type: str,
    custom_inputs: dict[str, object],
) -> None:
    translated = translate_prompt(
        {
            "latent": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 512, "height": 768, "batch_size": 1},
            },
            "select": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "res_2m"},
            },
            "schedule": {
                "class_type": "BasicScheduler",
                "inputs": {
                    "model": "model",
                    "scheduler": "bong_tangent",
                    "steps": 20,
                    "denoise": 1.0,
                },
            },
            "custom": {
                "class_type": class_type,
                "inputs": {
                    **custom_inputs,
                    "sampler": ["select", 0],
                    "sigmas": ["schedule", 0],
                    "latent_image": ["latent", 0],
                },
            },
            "sink": {"class_type": "LatentSink", "inputs": {"latent": ["custom", 0]}},
        },
        _res4lyf_sampling_schemas(),
    )

    select = translated.graph.nodes["select"]
    schedule = translated.graph.nodes["schedule"]
    custom = translated.graph.nodes["custom"]
    assert isinstance(select, GraphNode)
    assert isinstance(schedule, GraphNode)
    assert isinstance(custom, GraphNode)
    assert select.inputs["sampler_name"] == "res_2m"
    assert schedule.inputs["scheduler"] == "bong_tangent"
    assert custom.inputs["sampler"] == Link("select", "sampler")
    assert custom.inputs["sigmas"] == Link("schedule", "sigmas")
    assert custom.inputs["latent_image"] == Link("latent", "latent")
    latent = translated.graph.nodes["latent"]
    assert isinstance(latent, GraphNode)
    assert latent.node_type == "dinkster.empty_latent_image"
    assert translated.graph.nodes["sink"].inputs["latent"] == Link("custom", "output")
    assert translated.targets == ("sink",)


CUSTOM_COMBO_SCHEMA = NodeSchema(
    node_type="comfy.CustomCombo",
    inputs=(
        InputSpec("choice", COMBO),
        InputSpec("index", INT, required=False, default=0),
    ),
    input_families=(
        InputFamilySpec(
            "options",
            STRING,
            member_names=tuple(f"option{index}" for index in range(1, 101)),
        ),
    ),
    outputs=(OutputSpec("STRING", STRING), OutputSpec("INDEX", INT)),
    aliases=("CustomCombo",),
    output_node=True,
)


def test_custom_combo_prompt_nests_contiguous_literal_options() -> None:
    translated = translate_prompt(
        {
            "combo": {
                "class_type": "CustomCombo",
                "inputs": {
                    "choice": "two",
                    "index": 1,
                    "option1": "one",
                    "option2": "two",
                },
            }
        },
        {CUSTOM_COMBO_SCHEMA.node_type: CUSTOM_COMBO_SCHEMA},
        input_adapters=COMFY_INPUT_ADAPTERS,
    )
    assert translated.graph.nodes["combo"].inputs == {
        "choice": "two",
        "index": 1,
        "options.option1": "one",
        "options.option2": "two",
    }

    native = translate_prompt(
        {
            "combo": {
                "class_type": "CustomCombo",
                "inputs": {
                    "choice": "two",
                    "index": 1,
                    "options.option1": "one",
                    "options.option2": "two",
                },
            }
        },
        {CUSTOM_COMBO_SCHEMA.node_type: CUSTOM_COMBO_SCHEMA},
        input_adapters=COMFY_INPUT_ADAPTERS,
    )
    assert native.graph.nodes["combo"].inputs == translated.graph.nodes["combo"].inputs


@pytest.mark.parametrize(
    ("extra", "code"),
    [
        ({"option0": "zero"}, "prompt.custom_combo.bad_option"),
        ({"option01": "one"}, "prompt.custom_combo.bad_option"),
        ({"option101": "many"}, "prompt.custom_combo.bad_option"),
        ({"option" + "1" * 5000: "long"}, "prompt.custom_combo.bad_option"),
        ({"options.option" + "1" * 5000: "long"}, "prompt.custom_combo.bad_option"),
        ({"option2": "two"}, "prompt.custom_combo.noncontiguous"),
        ({"option1": 1}, "prompt.custom_combo.bad_value"),
        (
            {"option1": "legacy", "options.option1": "native"},
            "prompt.custom_combo.conflict",
        ),
        ({"options.option2": "two"}, "prompt.custom_combo.noncontiguous"),
    ],
)
def test_custom_combo_prompt_refuses_malformed_option_families(
    extra: dict[str, object], code: str
) -> None:
    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(
            {
                "combo": {
                    "class_type": "CustomCombo",
                    "inputs": {"choice": "one", **extra},
                }
            },
            {CUSTOM_COMBO_SCHEMA.node_type: CUSTOM_COMBO_SCHEMA},
            input_adapters=COMFY_INPUT_ADAPTERS,
        )
    assert code in {problem.code for problem in excinfo.value.problems}


def test_custom_combo_prompt_refuses_linked_options_and_unknown_inputs() -> None:
    schemas = dict(SCHEMAS)
    schemas[CUSTOM_COMBO_SCHEMA.node_type] = CUSTOM_COMBO_SCHEMA
    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(
            {
                "producer": {"class_type": "Producer", "inputs": {}},
                "combo": {
                    "class_type": "CustomCombo",
                    "inputs": {"choice": "one", "option1": ["producer", 0]},
                },
            },
            schemas,
            input_adapters=COMFY_INPUT_ADAPTERS,
        )
    assert "prompt.custom_combo.linked_option" in {
        problem.code for problem in excinfo.value.problems
    }

    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(
            {
                "producer": {"class_type": "Producer", "inputs": {}},
                "combo": {
                    "class_type": "CustomCombo",
                    "inputs": {
                        "choice": "one",
                        "options.option1": ["producer", 0],
                    },
                },
            },
            schemas,
            input_adapters=COMFY_INPUT_ADAPTERS,
        )
    assert "prompt.custom_combo.linked_option" in {
        problem.code for problem in excinfo.value.problems
    }

    custom_schemas = {CUSTOM_COMBO_SCHEMA.node_type: CUSTOM_COMBO_SCHEMA}
    translated = translate_prompt(
        {
            "combo": {
                "class_type": "CustomCombo",
                "inputs": {"choice": "one", "unrelated": "value"},
            }
        },
        custom_schemas,
        input_adapters=COMFY_INPUT_ADAPTERS,
    )
    diagnostics = validate(translated.graph, custom_schemas, translated.targets)
    assert "unknown-input" in {diagnostic.code for diagnostic in diagnostics}


def list_map_prompt(*, left: list[int], right: list[int] | None = None) -> dict[str, Any]:
    if right is None:
        return {
            "source": {"class_type": "ListProducer", "inputs": {"values": left}},
            "scale": {
                "class_type": "ScaleInt",
                "inputs": {"value": ["source", 0], "factor": 3},
            },
            "collect": {
                "class_type": "ListCollector",
                "inputs": {"values": ["scale", 0]},
            },
        }
    return {
        "source": {
            "class_type": "PairProducer",
            "inputs": {"left": left, "right": right},
        },
        "add": {
            "class_type": "AddInts",
            "inputs": {"left": ["source", 0], "right": ["source", 1]},
        },
        "collect": {
            "class_type": "ListCollector",
            "inputs": {"values": ["add", 0]},
        },
    }


def _comfy_9eaba63_map_kwargs(
    input_data_all: Mapping[str, list[object]],
) -> tuple[dict[str, object], ...]:
    """Model pinned ComfyUI list mapping empty-input boundaries.

    Commit: 9eaba63e1a9f2b27701cf0a0694aeed777da42f5
    Tree: b22b1ae1406b32cc664a16343f09e8d37a25431d
    Source: execution.py::_async_map_node_over_list lines 243-319
    """
    max_len_input = max((len(values) for values in input_data_all.values()), default=0)
    if max_len_input == 0:
        return ({},)
    return tuple(
        {
            name: values[index] if index < len(values) else values[-1]
            for name, values in input_data_all.items()
        }
        for index in range(max_len_input)
    )


def test_comfy_9eaba63_zero_input_and_all_empty_map_once_with_empty_kwargs() -> None:
    assert _comfy_9eaba63_map_kwargs({}) == ({},)
    assert _comfy_9eaba63_map_kwargs({"left": [], "right": []}) == ({},)


def test_comfy_9eaba63_mixed_empty_map_keeps_repeat_last_index_error() -> None:
    with pytest.raises(IndexError):
        _comfy_9eaba63_map_kwargs({"left": [], "right": [10]})


def test_implicit_list_map_lowers_and_executes_in_element_order() -> None:
    async def scenario() -> None:
        translation = translate_prompt(list_map_prompt(left=[3, 1, 2]), SCHEMAS)
        region = translation.graph.nodes["scale"]
        assert isinstance(region, RegionNode)
        assert region.kind == "map"
        assert region.binding == "zip"
        assert region.element_ports == ("value",)
        assert region.inputs == {"value": Link("source", "values")}
        body = region.body.nodes["scale"]
        assert isinstance(body, GraphNode)
        assert body.inputs == {
            "value": Link("$region", "value"),
            "factor": 3,
        }
        collector = translation.graph.nodes["collect"]
        assert isinstance(collector, GraphNode)
        assert collector.inputs["values"] == Link("scale", "value")

        result = await make_engine().run(translation.graph, translation.targets)
        assert result.outputs["collect"]["values"].resolve() == [9, 3, 6]

    asyncio.run(scenario())


def test_implicit_list_map_cardinality_propagates_down_chains() -> None:
    prompt = list_map_prompt(left=[1, 2])
    prompt["again"] = {
        "class_type": "ScaleInt",
        "inputs": {"value": ["scale", 0], "factor": 10},
    }
    prompt["collect"]["inputs"]["values"] = ["again", 0]
    translation = translate_prompt(prompt, SCHEMAS)
    first = translation.graph.nodes["scale"]
    second = translation.graph.nodes["again"]
    assert isinstance(first, RegionNode)
    assert isinstance(second, RegionNode)
    assert second.inputs["value"] == Link("scale", "value")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1, 2], [10, 20], [11, 22]),
        ([1, 2], [10, 20, 30, 40, 50], [11, 22, 32, 42, 52]),
        ([1], [10, 20, 30], [11, 21, 31]),
    ],
)
def test_implicit_multi_list_map_matches_upstream_repeat_final_alignment(
    left: list[int], right: list[int], expected: list[int]
) -> None:
    async def scenario() -> None:
        translation = translate_prompt(list_map_prompt(left=left, right=right), SCHEMAS)
        region = translation.graph.nodes["add"]
        assert isinstance(region, RegionNode)
        assert region.binding == "broadcast"
        assert region.element_ports == ("left", "right")
        result = await make_engine().run(translation.graph, translation.targets)
        assert result.outputs["collect"]["values"].resolve() == expected

    asyncio.run(scenario())


def test_implicit_list_map_cache_identity_is_stable_across_iterations() -> None:
    async def scenario() -> None:
        ScaleInt.calls.clear()
        engine = make_engine()
        first = translate_prompt(list_map_prompt(left=[1, 2]), SCHEMAS)
        await engine.run(first.graph, first.targets)
        assert sorted(ScaleInt.calls) == [1, 2]

        second = translate_prompt(list_map_prompt(left=[1, 3]), SCHEMAS)
        await engine.run(second.graph, second.targets)
        assert sorted(ScaleInt.calls) == [1, 2, 3]

    asyncio.run(scenario())


def test_input_is_list_consumer_is_not_lowered() -> None:
    prompt = {
        "source": {"class_type": "ListProducer", "inputs": {"values": [1, 2]}},
        "collect": {
            "class_type": "ListCollector",
            "inputs": {"values": ["source", 0]},
        },
    }
    translation = translate_prompt(prompt, SCHEMAS)
    assert isinstance(translation.graph.nodes["collect"], GraphNode)


def test_element_type_incompatible_list_edge_keeps_existing_diagnostic() -> None:
    prompt = {
        "source": {"class_type": "ListProducer", "inputs": {"values": [1, 2]}},
        "sink": {
            "class_type": "StringSink",
            "inputs": {"value": ["source", 0]},
        },
    }
    translation = translate_prompt(prompt, SCHEMAS)
    assert isinstance(translation.graph.nodes["sink"], GraphNode)
    diagnostics = validate(translation.graph, SCHEMAS, translation.targets)
    mismatch = next(diag for diag in diagnostics if diag.code == "list-into-scalar")
    assert mismatch.node_id == "sink"
    assert mismatch.input_id == "value"


def test_mapped_output_is_list_lowers_to_flatten_and_executes() -> None:
    prompt = {
        "source": {"class_type": "ListProducer", "inputs": {"values": [1, 2]}},
        "fanout": {
            "class_type": "MappedFanout",
            "inputs": {"value": ["source", 0]},
        },
        "collect": {
            "class_type": "ListCollector",
            "inputs": {"values": ["fanout", 0]},
        },
    }

    async def scenario() -> None:
        translated = translate_prompt(prompt, SCHEMAS)
        fanout = translated.graph.nodes["fanout"]
        assert isinstance(fanout, RegionNode)
        assert fanout.outputs["values"].mode == "flatten"
        assert isinstance(translated.graph.nodes["collect"], GraphNode)
        result = await make_engine().run(translated.graph, translated.targets)
        assert result.outputs["collect"]["values"].resolve() == [1, 1, 2, 2]

    asyncio.run(scenario())


def test_prompt_without_list_mapping_is_byte_identical_to_prior_graph_shape() -> None:
    prompt: dict[str, Any] = {
        "1": {"class_type": "Producer", "inputs": {"seed": 7}},
        "2": {"class_type": "Consumer", "inputs": {"text": ["1", 1]}},
    }
    expected = Graph(
        nodes={
            "1": GraphNode("comfy.Producer", {"seed": 7}),
            "2": GraphNode("comfy.Consumer", {"text": Link("1", "second")}),
        }
    )
    translated = translate_prompt(prompt, SCHEMAS).graph
    assert translated == expected
    assert graph_to_wire(translated) == graph_to_wire(expected)


def test_cyclic_prompt_still_reaches_existing_cycle_diagnostic() -> None:
    prompt = {
        "left": {
            "class_type": "ScaleInt",
            "inputs": {"value": ["right", 0], "factor": 2},
        },
        "right": {
            "class_type": "ScaleInt",
            "inputs": {"value": ["left", 0], "factor": 3},
        },
        "collect": {
            "class_type": "ListCollector",
            "inputs": {"values": ["left", 0]},
        },
    }
    translation = translate_prompt(prompt, SCHEMAS)
    diagnostics = validate(translation.graph, SCHEMAS, translation.targets)
    assert any(diagnostic.code == "cycle" for diagnostic in diagnostics)


def test_prompt_extracts_recursive_dynamic_combo_choices() -> None:
    translation = translate_prompt(
        {
            "dynamic": {
                "class_type": "DynamicConsumer",
                "inputs": {
                    "combo": "outer",
                    "combo.subcombo": "inner",
                    "combo.subcombo.value": 7,
                },
            }
        },
        SCHEMAS,
    )
    node = translation.graph.nodes["dynamic"]
    assert isinstance(node, GraphNode)
    assert node.inputs == {"combo.subcombo.value": 7}
    assert node.slot_variants == {
        "combo": "outer",
        "combo.subcombo": "inner",
    }


@pytest.mark.parametrize(
    ("choice", "family_input"),
    [
        ("family", "combo.frame.image0"),
        ("names", "combo.named.right"),
    ],
)
def test_prompt_accepts_family_in_combo_vectors(choice: str, family_input: str) -> None:
    translation = translate_prompt(
        {
            "dynamic": {
                "class_type": "DynamicConsumer",
                "inputs": {"combo": choice, family_input: 7},
            }
        },
        SCHEMAS,
    )
    node = translation.graph.nodes["dynamic"]
    assert isinstance(node, GraphNode)
    assert node.inputs == {family_input: 7}
    assert node.slot_variants == {"combo": choice}


def test_prompt_extracts_space_containing_dynamic_combo_choice() -> None:
    translation = translate_prompt(
        {
            "dynamic": {
                "class_type": "DynamicConsumer",
                "inputs": {
                    "combo": "scale dimensions",
                    "combo.width": 512,
                },
            }
        },
        SCHEMAS,
    )
    node = translation.graph.nodes["dynamic"]
    assert isinstance(node, GraphNode)
    assert node.inputs == {"combo.width": 512}
    assert node.slot_variants == {"combo": "scale dimensions"}


@pytest.mark.parametrize("choice", [3, "missing"])
def test_prompt_classifies_malformed_dynamic_combo_selectors(choice: object) -> None:
    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(
            {
                "dynamic": {
                    "class_type": "DynamicConsumer",
                    "inputs": {"combo": choice},
                }
            },
            SCHEMAS,
        )
    problem = next(
        problem for problem in excinfo.value.problems if problem.code == "prompt.bad_dynamic_choice"
    )
    assert problem.node_id == "dynamic"
    assert problem.input_id == "combo"


def test_prompt_classifies_unknown_space_containing_dynamic_combo_choice() -> None:
    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(
            {
                "dynamic": {
                    "class_type": "DynamicConsumer",
                    "inputs": {"combo": "scale dimension"},
                }
            },
            SCHEMAS,
        )
    assert any(
        problem.code == "prompt.bad_dynamic_choice"
        and problem.node_id == "dynamic"
        and problem.input_id == "combo"
        for problem in excinfo.value.problems
    )


def test_prompt_accepts_native_type_ids_and_integral_float_indexes() -> None:
    prompt: dict[str, Any] = {
        "p": {"class_type": "comfy.Producer", "inputs": {}},
        "c": {"class_type": "Consumer", "inputs": {"text": ["p", 0.0]}},
    }
    translation = translate_prompt(prompt, SCHEMAS)
    assert translation.graph.nodes["c"].inputs["text"] == Link("p", "first")


def test_prompt_two_element_literal_lists_stay_literal() -> None:
    # is_link requires [str, number]; anything else is a literal that the
    # engine's own validation will judge.
    prompt: dict[str, Any] = {
        "c": {"class_type": "Consumer", "inputs": {"text": [1, 2]}},
    }
    translation = translate_prompt(prompt, SCHEMAS)
    assert translation.graph.nodes["c"].inputs["text"] == [1, 2]


def collect_problems(
    prompt: Mapping[str, Any], schemas: Mapping[str, NodeSchema]
) -> dict[str, str]:
    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(prompt, schemas)
    return {p.code: p.message for p in excinfo.value.problems}


def test_prompt_refusals_are_anchored_and_collected() -> None:
    prompt: dict[str, Any] = {
        "a": {"class_type": "NoSuchNode", "inputs": {}},
        "b": {"class_type": "Consumer", "inputs": {"text": ["missing", 0]}},
        "c": {"class_type": "Consumer", "inputs": {"text": ["a", 5]}},
        "d/bad": {"class_type": "Consumer", "inputs": {}},
    }
    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(prompt, SCHEMAS)
    problems = {(p.code, p.node_id) for p in excinfo.value.problems}
    assert ("prompt.unknown_class_type", "a") in problems
    assert ("prompt.dangling_link", "b") in problems
    assert ("prompt.bad_node_id", "d/bad") in problems
    # "c" links to "a", whose own resolution already failed - no second
    # problem is stacked on top of it (index range cannot be checked).
    assert not any(code == "prompt.bad_output_index" for code, _ in problems)


def test_prompt_output_index_out_of_range() -> None:
    prompt: dict[str, Any] = {
        "p": {"class_type": "Producer", "inputs": {}},
        "c": {"class_type": "Consumer", "inputs": {"text": ["p", 2]}},
    }
    problems = collect_problems(prompt, SCHEMAS)
    assert "prompt.bad_output_index" in problems
    assert "2 outputs" in problems["prompt.bad_output_index"]


def test_prompt_non_integral_index_refused() -> None:
    prompt: dict[str, Any] = {
        "p": {"class_type": "Producer", "inputs": {}},
        "c": {"class_type": "Consumer", "inputs": {"text": ["p", 0.5]}},
    }
    assert "prompt.bad_output_index" in collect_problems(prompt, SCHEMAS)


def test_prompt_ambiguous_class_type_lists_candidates() -> None:
    schemas = ambiguous_schemas()
    prompt: dict[str, Any] = {
        "n": {"class_type": "Dup", "inputs": {}},
        "c": {"class_type": "Consumer", "inputs": {"text": "x"}},
    }
    problems = collect_problems(prompt, schemas)
    message = problems["prompt.ambiguous_class_type"]
    assert "comfy.packa.Dup" in message and "comfy.packb.Dup" in message


def test_prompt_without_output_nodes_refused() -> None:
    prompt: dict[str, Any] = {"p": {"class_type": "Producer", "inputs": {}}}
    assert "prompt.no_output_nodes" in collect_problems(prompt, SCHEMAS)


def test_prompt_empty_refused() -> None:
    assert "prompt.empty" in collect_problems({}, SCHEMAS)


def test_prompt_reserved_link_key_literal_refused() -> None:
    prompt: dict[str, Any] = {
        "c": {
            "class_type": "Consumer",
            "inputs": {"text": {"nested": [{"$link": {"node": "x", "output": "y"}}]}},
        },
    }
    assert "prompt.reserved_key" in collect_problems(prompt, SCHEMAS)


# -- extract_prompt -----------------------------------------------------------


def test_extract_bare_prompt() -> None:
    body = {"c": {"class_type": "Consumer", "inputs": {}}}
    prompt, client_id, extra_pnginfo = extract_prompt(body)
    assert prompt is body
    assert client_id is None
    assert extra_pnginfo is None


def test_extract_wrapper_with_client_id() -> None:
    inner = {"c": {"class_type": "Consumer", "inputs": {}}}
    pnginfo = {"workflow": {"nodes": [2, 1]}, "ordered": {"z": 1, "a": 2}}
    prompt, client_id, extra_pnginfo = extract_prompt(
        {
            "prompt": inner,
            "client_id": "abc",
            "extra_data": {"extra_pnginfo": pnginfo},
        }
    )
    assert prompt is inner
    assert client_id == "abc"
    assert extra_pnginfo is pnginfo


def test_extract_prompt_node_named_prompt_stays_a_prompt() -> None:
    body = {
        "prompt": {"class_type": "Consumer", "inputs": {}},
        "other": {"class_type": "Producer", "inputs": {}},
    }
    prompt, _, extra_pnginfo = extract_prompt(body)
    assert prompt is body
    assert extra_pnginfo is None


def test_extract_rejects_non_object_body() -> None:
    with pytest.raises(PromptTranslationError):
        extract_prompt([1, 2, 3])


def test_extract_rejects_non_mapping_extra_pnginfo() -> None:
    inner = {"c": {"class_type": "Consumer", "inputs": {}}}
    with pytest.raises(PromptTranslationError) as exc_info:
        extract_prompt({"prompt": inner, "extra_data": {"extra_pnginfo": ["not", "an", "object"]}})
    assert exc_info.value.problems[0].code == "prompt.bad_extra_pnginfo"


# -- input adapters (legacy SaveImage filename_prefix) ------------------------


def save_prompt(**inputs: Any) -> dict[str, Any]:
    return {
        "1": {"class_type": "Producer", "inputs": {}},
        "2": {
            "class_type": "SaveImage",
            "inputs": {"images": ["1", 0], **inputs},
        },
    }


def test_filename_prefix_converts_to_save_target() -> None:
    translation = translate_prompt(
        save_prompt(filename_prefix="renders/scene"),
        SCHEMAS,
        input_adapters=COMFY_INPUT_ADAPTERS,
    )
    node = translation.graph.nodes["2"]
    assert node.inputs["target"] == {"mount": "comfy-output", "prefix": "renders/scene"}
    assert "filename_prefix" not in node.inputs
    assert node.inputs["images"] == Link("1", "first")
    assert translation.targets == ("2",)


def test_filename_prefix_backslashes_normalize() -> None:
    translation = translate_prompt(
        save_prompt(filename_prefix="renders\\scene"),
        SCHEMAS,
        input_adapters=COMFY_INPUT_ADAPTERS,
    )
    target = translation.graph.nodes["2"].inputs["target"]
    assert target == {"mount": "comfy-output", "prefix": "renders/scene"}


def test_filename_prefix_empty_string_gets_comfy_default() -> None:
    translation = translate_prompt(
        save_prompt(filename_prefix=""),
        SCHEMAS,
        input_adapters=COMFY_INPUT_ADAPTERS,
    )
    target = translation.graph.nodes["2"].inputs["target"]
    assert target == {"mount": "comfy-output", "prefix": "ComfyUI"}


def test_filename_prefix_absent_leaves_inputs_alone() -> None:
    translation = translate_prompt(save_prompt(), SCHEMAS, input_adapters=COMFY_INPUT_ADAPTERS)
    node = translation.graph.nodes["2"]
    assert "target" not in node.inputs  # the native schema default applies
    assert "filename_prefix" not in node.inputs


@pytest.mark.parametrize(
    ("prefix", "code"),
    [
        ("/etc/passwd", "prompt.save_target.invalid"),  # absolute
        ("a/../b", "prompt.save_target.invalid"),  # traversal
        ("a//b", "prompt.save_target.invalid"),  # empty segment
        ("%date:yyyy-MM-dd%/img", "prompt.save_target.substitution"),
        (7, "prompt.save_target.invalid"),  # not a string
    ],
)
def test_unsafe_filename_prefix_refused(prefix: Any, code: str) -> None:
    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(
            save_prompt(filename_prefix=prefix),
            SCHEMAS,
            input_adapters=COMFY_INPUT_ADAPTERS,
        )
    problems = {p.code: p for p in excinfo.value.problems}
    assert code in problems
    assert problems[code].node_id == "2"
    assert problems[code].input_id == "filename_prefix"


def test_linked_filename_prefix_refused() -> None:
    """A runtime-computed prefix cannot become a structured target at
    translation time; refuse loudly instead of half-converting."""
    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(
            save_prompt(filename_prefix=["1", 0]),
            SCHEMAS,
            input_adapters=COMFY_INPUT_ADAPTERS,
        )
    assert {p.code for p in excinfo.value.problems} == {"prompt.save_target.linked"}


def test_both_prefix_and_target_refused() -> None:
    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(
            save_prompt(filename_prefix="a", target={"mount": "comfy-output", "prefix": "b"}),
            SCHEMAS,
            input_adapters=COMFY_INPUT_ADAPTERS,
        )
    assert {p.code for p in excinfo.value.problems} == {"prompt.save_target.conflict"}


def test_without_adapters_prefix_passes_through() -> None:
    """The adapter seam is opt-in: plain translate_prompt leaves inputs
    untouched (native submissions never ride adapters)."""
    translation = translate_prompt(save_prompt(filename_prefix="a"), SCHEMAS)
    assert translation.graph.nodes["2"].inputs["filename_prefix"] == "a"


def test_adapter_unit_no_translation_machinery() -> None:
    adapted, problems = adapt_save_image_inputs("n", {"filename_prefix": "x/y"})
    assert problems == []
    assert adapted == {"target": {"mount": "comfy-output", "prefix": "x/y"}}


# -- input adapters (legacy LoadImage image filename) -------------------------


IMG_DIGEST = digest_bytes(b"png-bytes")


def load_prompt(image: Any) -> dict[str, Any]:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "Consumer", "inputs": {"text": ["1", 0]}},
    }


def fake_ref(relative: str) -> AssetRef:
    return AssetRef(
        digest=IMG_DIGEST,
        name=relative.rsplit("/", 1)[-1],
        size=9,
        media_type="image/png",
        virtual_path=f"mounts/comfy-input/{relative}",
    )


def fake_resolver(
    known: dict[str, AssetRef], seen: list[str] | None = None
) -> Callable[[str], AssetRef | None]:
    def resolve(relative: str) -> AssetRef | None:
        if seen is not None:
            seen.append(relative)
        return known.get(relative)

    return resolve


def adapters_with(resolve: Callable[[str], AssetRef | None]) -> dict[str, Any]:
    return {**COMFY_INPUT_ADAPTERS, "dinkster.load_image": make_load_image_adapter(resolve)}


def test_image_filename_converts_to_asset_wire() -> None:
    translation = translate_prompt(
        load_prompt("img.png"),
        SCHEMAS,
        input_adapters=adapters_with(fake_resolver({"img.png": fake_ref("img.png")})),
    )
    assert translation.graph.nodes["1"].inputs["image"] == {
        "digest": IMG_DIGEST,
        "name": "img.png",
        "size": 9,
        "mediaType": "image/png",
        "virtualPath": "mounts/comfy-input/img.png",
    }


def test_image_backslash_subfolder_normalizes() -> None:
    seen: list[str] = []
    known = {"sub/img.png": fake_ref("sub/img.png")}
    translate_prompt(
        load_prompt("sub\\img.png"),
        SCHEMAS,
        input_adapters=adapters_with(fake_resolver(known, seen)),
    )
    assert seen == ["sub/img.png"]


@pytest.mark.parametrize(
    "image",
    [
        "/etc/shadow.png",  # absolute
        "C:\\img.png",  # absolute (Windows drive)
        "../escape.png",  # traversal
        "a/../b.png",  # traversal
        "",  # empty
        7,  # not a string
    ],
)
def test_unsafe_or_invalid_image_refused_before_lookup(image: Any) -> None:
    seen: list[str] = []
    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(
            load_prompt(image),
            SCHEMAS,
            input_adapters=adapters_with(fake_resolver({}, seen)),
        )
    assert seen == []  # refused before any catalog lookup
    problems = {p.code: p for p in excinfo.value.problems}
    assert "prompt.load_image.invalid" in problems
    assert problems["prompt.load_image.invalid"].node_id == "1"
    assert problems["prompt.load_image.invalid"].input_id == "image"
    if image == "/etc/shadow.png":  # pin the pre-refactor diagnostic wording
        assert problems["prompt.load_image.invalid"].message == (
            "image '/etc/shadow.png' is an absolute path; legacy image "
            "names are relative to the input directory"
        )


def test_uncataloged_image_is_anchored_unresolved_problem() -> None:
    """No exact match means an explicit refusal pointing at the guess
    endpoint - never a silent same-name substitution."""
    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(
            load_prompt("nowhere.png"),
            SCHEMAS,
            input_adapters=adapters_with(fake_resolver({})),
        )
    (problem,) = excinfo.value.problems
    assert problem.code == "prompt.load_image.unresolved"
    assert problem.node_id == "1"
    assert problem.input_id == "image"
    assert "/api/assets/guess" in problem.message


def test_native_image_shapes_pass_through_untouched() -> None:
    seen: list[str] = []
    adapters = adapters_with(fake_resolver({}, seen))
    wire = fake_ref("img.png").to_wire()
    translation = translate_prompt(load_prompt(wire), SCHEMAS, input_adapters=adapters)
    assert translation.graph.nodes["1"].inputs["image"] == wire

    linked = {
        "0": {"class_type": "Producer", "inputs": {}},
        **load_prompt(["0", 0]),
    }
    translation = translate_prompt(linked, SCHEMAS, input_adapters=adapters)
    assert translation.graph.nodes["1"].inputs["image"] == Link("0", "first")
    assert seen == []  # already-native shapes never hit the resolver


def test_load_adapter_unit_no_translation_machinery() -> None:
    adapt = make_load_image_adapter(fake_resolver({"img.png": fake_ref("img.png")}))
    adapted, problems = adapt("n", {"image": "img.png"})
    assert problems == []
    assert adapted == {"image": fake_ref("img.png").to_wire()}
    untouched, problems = adapt("n", {"other": 1})
    assert problems == []
    assert untouched == {"other": 1}


# -- input adapters (legacy CheckpointLoaderSimple ckpt_name) ------------------


def ckpt_prompt(value: Any, **extra: Any) -> dict[str, Any]:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": value, **extra},
        },
        "2": {"class_type": "Consumer", "inputs": {"text": ["1", 0]}},
    }


def ckpt_ref(relative: str) -> AssetRef:
    return AssetRef(
        digest=IMG_DIGEST,
        name=relative.rsplit("/", 1)[-1],
        size=9,
        virtual_path=f"mounts/comfy-models/checkpoints/{relative}",
    )


def ckpt_adapters_with(
    resolve: Callable[[str], AssetRef | None],
) -> dict[str, Any]:
    return {
        **COMFY_INPUT_ADAPTERS,
        "dinkster.load_checkpoint": make_load_checkpoint_adapter(resolve),
    }


def test_ckpt_name_converts_and_renames_to_checkpoint() -> None:
    """The legacy ckpt_name string becomes an asset wire value on the
    RENAMED native 'checkpoint' input; the legacy key disappears."""
    ref = ckpt_ref("sd15/model.safetensors")
    translation = translate_prompt(
        ckpt_prompt("sd15\\model.safetensors"),  # backslash normalizes too
        SCHEMAS,
        input_adapters=ckpt_adapters_with(fake_resolver({"sd15/model.safetensors": ref})),
    )
    inputs = translation.graph.nodes["1"].inputs
    assert "ckpt_name" not in inputs
    assert inputs["checkpoint"] == ref.to_wire()


def test_ckpt_name_and_checkpoint_together_is_conflict() -> None:
    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(
            ckpt_prompt("a.safetensors", checkpoint={"digest": "x"}),
            SCHEMAS,
            input_adapters=ckpt_adapters_with(fake_resolver({})),
        )
    (problem,) = excinfo.value.problems
    assert problem.code == "prompt.load_checkpoint.conflict"
    assert problem.node_id == "1"
    assert problem.input_id == "ckpt_name"


@pytest.mark.parametrize(
    "value",
    [
        ["0", 0],  # a link: computed names cannot convert at translation time
        {"digest": "x"},  # asset wire under the LEGACY key, not a legacy shape
        "/abs/model.safetensors",  # absolute
        "C:\\model.safetensors",  # absolute (Windows drive)
        "../escape.safetensors",  # traversal
        "",  # empty
        7,  # not a string
    ],
)
def test_invalid_ckpt_name_refused_before_lookup(value: Any) -> None:
    """Unlike LoadImage (same input id), the checkpoint input RENAMES, so
    links and mappings under the legacy key are refused - native shapes
    belong on 'checkpoint' directly."""
    seen: list[str] = []
    prompt = ckpt_prompt(value)
    if value == ["0", 0]:
        prompt = {"0": {"class_type": "Producer", "inputs": {}}, **prompt}
    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(
            prompt, SCHEMAS, input_adapters=ckpt_adapters_with(fake_resolver({}, seen))
        )
    assert seen == []  # refused before any catalog lookup
    problems = {p.code: p for p in excinfo.value.problems}
    assert "prompt.load_checkpoint.invalid" in problems
    assert problems["prompt.load_checkpoint.invalid"].node_id == "1"
    assert problems["prompt.load_checkpoint.invalid"].input_id == "ckpt_name"


def test_uncataloged_ckpt_name_is_anchored_unresolved_problem() -> None:
    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(
            ckpt_prompt("nowhere.safetensors"),
            SCHEMAS,
            input_adapters=ckpt_adapters_with(fake_resolver({})),
        )
    (problem,) = excinfo.value.problems
    assert problem.code == "prompt.load_checkpoint.unresolved"
    assert problem.node_id == "1"
    assert problem.input_id == "ckpt_name"
    assert "/api/assets/guess" in problem.message
    assert "derived ComfyUI checkpoints roots" in problem.message


def test_native_checkpoint_input_never_hits_the_adapter() -> None:
    seen: list[str] = []
    adapters = ckpt_adapters_with(fake_resolver({}, seen))
    wire = ckpt_ref("a.safetensors").to_wire()
    prompt = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"checkpoint": wire},
        },
        "2": {"class_type": "Consumer", "inputs": {"text": ["1", 0]}},
    }
    translation = translate_prompt(prompt, SCHEMAS, input_adapters=adapters)
    assert translation.graph.nodes["1"].inputs["checkpoint"] == wire
    assert seen == []


def test_ckpt_adapter_unit_no_translation_machinery() -> None:
    ref = ckpt_ref("a.safetensors")
    adapt = make_load_checkpoint_adapter(fake_resolver({"a.safetensors": ref}))
    adapted, problems = adapt("n", {"ckpt_name": "a.safetensors"})
    assert problems == []
    assert adapted == {"checkpoint": ref.to_wire()}
    untouched, problems = adapt("n", {"other": 1})
    assert problems == []
    assert untouched == {"other": 1}


# -- input adapters (legacy lora/vae/clip/unet model names) --------------------
#
# All four are instances of the checkpoint recipe (_make_model_name_adapter),
# so the deep per-case coverage above (traversal, absolute paths, non-strings,
# refusal-before-lookup) proves the shared machinery once; these tests pin the
# per-loader parameterization (input ids, problem codes, subtrees) plus the
# behavior unique to a loader (VAELoader's unported arms, extra inputs riding
# through untouched).

MODEL_ADAPTER_CASES = [
    pytest.param(
        make_load_model_patch_adapter,
        "name",
        "model_patch",
        "prompt.load_model_patch",
        "model_patches",
        id="model-patch",
    ),
    pytest.param(
        make_load_lora_adapter,
        "lora_name",
        "lora",
        "prompt.load_lora",
        "loras",
        id="lora",
    ),
    pytest.param(
        make_load_vae_adapter,
        "vae_name",
        "vae",
        "prompt.load_vae",
        "vae",
        id="vae",
    ),
    pytest.param(
        make_load_clip_adapter,
        "clip_name",
        "text_encoder",
        "prompt.load_clip",
        "text_encoders",
        id="clip",
    ),
    pytest.param(
        make_load_diffusion_model_adapter,
        "unet_name",
        "diffusion_model",
        "prompt.load_diffusion_model",
        "diffusion_models",
        id="unet",
    ),
]


def model_ref(subtree: str, relative: str) -> AssetRef:
    return AssetRef(
        digest=IMG_DIGEST,
        name=relative.rsplit("/", 1)[-1],
        size=9,
        virtual_path=f"mounts/comfy-models/{subtree}/{relative}",
    )


@pytest.mark.parametrize(
    ("builder", "legacy_input", "native_input", "code_prefix", "subtree"),
    MODEL_ADAPTER_CASES,
)
def test_model_name_converts_and_renames(
    builder: Callable[..., Any],
    legacy_input: str,
    native_input: str,
    code_prefix: str,
    subtree: str,
) -> None:
    ref = model_ref(subtree, "sub/model.safetensors")
    adapt = builder(fake_resolver({"sub/model.safetensors": ref}))
    adapted, problems = adapt("n", {legacy_input: "sub\\model.safetensors", "extra": 3})
    assert problems == []
    # Legacy key disappears, native key carries the wire value, and any
    # other inputs (strengths, type, device, weight_dtype) ride through.
    assert adapted == {native_input: ref.to_wire(), "extra": 3}


def test_dual_clip_adapter_preserves_order_and_selectors() -> None:
    first = model_ref("text_encoders", "clip_l.safetensors")
    second = model_ref("text_encoders", "clip_g.safetensors")
    seen: list[str] = []
    adapt = make_load_dual_clip_adapter(
        fake_resolver(
            {"clip_l.safetensors": first, "clip_g.safetensors": second},
            seen,
        )
    )

    adapted, problems = adapt(
        "n",
        {
            "clip_name1": "clip_l.safetensors",
            "clip_name2": "clip_g.safetensors",
            "type": "sdxl",
            "device": "cpu",
        },
    )

    assert problems == []
    assert adapted == {
        "text_encoder1": first.to_wire(),
        "text_encoder2": second.to_wire(),
        "type": "sdxl",
        "device": "cpu",
    }
    assert seen == ["clip_l.safetensors", "clip_g.safetensors"]


def test_dual_clip_adapter_preserves_native_assets_and_links() -> None:
    first = model_ref("text_encoders", "clip_l.safetensors").to_wire()
    second = Link("source", "asset")
    adapt = make_load_dual_clip_adapter(fake_resolver({}))

    inputs = {"text_encoder1": first, "text_encoder2": second, "type": "flux"}
    adapted, problems = adapt("n", inputs)

    assert problems == []
    assert adapted is inputs


def test_dual_clip_adapter_reports_each_legacy_field_independently() -> None:
    second = model_ref("text_encoders", "t5.safetensors")
    adapt = make_load_dual_clip_adapter(fake_resolver({"t5.safetensors": second}))

    adapted, problems = adapt(
        "n",
        {"clip_name1": {"digest": "not-legacy"}, "clip_name2": "t5.safetensors"},
    )

    assert [problem.code for problem in problems] == ["prompt.load_dual_clip.clip_name1.invalid"]
    assert problems[0].input_id == "clip_name1"
    assert adapted == {"text_encoder2": second.to_wire()}

    adapted, problems = adapt("n", {"clip_name1": "missing.safetensors"})
    assert [problem.code for problem in problems] == ["prompt.load_dual_clip.clip_name1.unresolved"]
    assert problems[0].input_id == "clip_name1"
    assert adapted == {}


@pytest.mark.parametrize(
    ("builder", "legacy_input", "native_input", "code_prefix", "subtree"),
    MODEL_ADAPTER_CASES,
)
def test_model_name_conflict_and_invalid_and_unresolved(
    builder: Callable[..., Any],
    legacy_input: str,
    native_input: str,
    code_prefix: str,
    subtree: str,
) -> None:
    seen: list[str] = []
    adapt = builder(fake_resolver({}, seen))
    # Both legacy and native present: conflict.
    _, problems = adapt("n", {legacy_input: "a.safetensors", native_input: {"digest": "x"}})
    assert [p.code for p in problems] == [f"{code_prefix}.conflict"]
    assert problems[0].input_id == legacy_input
    # A link or mapping under the legacy key: invalid, refused pre-lookup.
    for bad in (Link("0", "out"), {"digest": "x"}, "../escape", 7):
        _, problems = adapt("n", {legacy_input: bad})
        assert [p.code for p in problems] == [f"{code_prefix}.invalid"], bad
    assert seen == []
    # An uncataloged exact name: anchored unresolved problem, no guessing.
    _, problems = adapt("n", {legacy_input: "nowhere.safetensors"})
    assert [p.code for p in problems] == [f"{code_prefix}.unresolved"]
    assert "/api/assets/guess" in problems[0].message
    assert subtree in problems[0].message
    assert seen == ["nowhere.safetensors"]


def test_translated_model_asset_adapter_preserves_legacy_prompt_names() -> None:
    checkpoint = model_ref("checkpoints", "sd15/model.safetensors")
    lora = model_ref("loras", "style.safetensors")
    seen: list[str] = []
    adapt = make_model_asset_inputs_adapter(
        {
            "checkpoint": (
                fake_resolver({"sd15/model.safetensors": checkpoint}, seen),
                "checkpoints",
            ),
            "lora": (fake_resolver({"style.safetensors": lora}, seen), "loras"),
        }
    )

    adapted, problems = adapt(
        "n",
        {
            "checkpoint": "sd15\\model.safetensors",
            "lora": lora.to_wire(),
            "strength": 0.5,
        },
    )
    assert problems == []
    assert adapted == {
        "checkpoint": checkpoint.to_wire(),
        "lora": lora.to_wire(),
        "strength": 0.5,
    }
    assert seen == ["sd15/model.safetensors"]

    _, problems = adapt("n", {"checkpoint": "../escape"})
    assert [problem.code for problem in problems] == ["prompt.model_asset.invalid"]
    _, problems = adapt("n", {"lora": "missing.safetensors"})
    assert [problem.code for problem in problems] == ["prompt.model_asset.unresolved"]


@pytest.mark.parametrize(
    "name",
    [
        "taesd",
        "taesdxl",
        "taesd3",
        "taef1",
        "taef2",
        "taehv.safetensors",  # video TAE: stem matches exactly
        "lighttaew2_2.safetensors",
    ],
)
def test_vae_unported_arms_refuse_honestly(name: str) -> None:
    """VAELoader's taesd/vae_approx vocabularies are not single files under
    vae/; they refuse before any catalog lookup."""
    seen: list[str] = []
    adapt = make_load_vae_adapter(fake_resolver({}, seen))
    _, problems = adapt("n", {"vae_name": name})
    assert [p.code for p in problems] == ["prompt.load_vae.unported"]
    assert problems[0].input_id == "vae_name"
    assert seen == []


def test_vae_pixel_space_selector_translates_without_asset_lookup() -> None:
    seen: list[str] = []
    adapt = make_load_vae_adapter(fake_resolver({}, seen))

    adapted, problems = adapt("n", {"vae_name": "pixel_space"})

    assert problems == []
    assert adapted == {"pixel_space": True}
    assert seen == []


def test_vae_tae_lookalike_file_still_resolves() -> None:
    """The video-TAE match is stem-exact (reference splitext dispatch):
    taehv2.safetensors is a real file v1 loads from vae/, so it converts."""
    ref = model_ref("vae", "taehv2.safetensors")
    adapt = make_load_vae_adapter(fake_resolver({"taehv2.safetensors": ref}))
    adapted, problems = adapt("n", {"vae_name": "taehv2.safetensors"})
    assert problems == []
    assert adapted == {"vae": ref.to_wire()}


def test_lora_name_translates_through_prompt_with_alias() -> None:
    """End to end through translate_prompt: the legacy LoraLoader alias
    resolves to dinkster.load_lora and lora_name converts on the way in,
    with strength inputs untouched."""
    ref = model_ref("loras", "style.safetensors")
    prompt: dict[str, Any] = {
        "1": {
            "class_type": "LoraLoader",
            "inputs": {"lora_name": "style.safetensors", "strength_model": "0.8"},
        },
        "2": {"class_type": "Consumer", "inputs": {"text": ["1", 0]}},
    }
    translation = translate_prompt(
        prompt,
        SCHEMAS,
        input_adapters={
            **COMFY_INPUT_ADAPTERS,
            "dinkster.load_lora": make_load_lora_adapter(fake_resolver({"style.safetensors": ref})),
        },
    )
    inputs = translation.graph.nodes["1"].inputs
    assert "lora_name" not in inputs
    assert inputs["lora"] == ref.to_wire()
    assert inputs["strength_model"] == "0.8"


# -- HTTP endpoint ------------------------------------------------------------


def make_engine(on_event: EventListener | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=SCHEMAS,
        registry=registry,
        worker=InProcessWorker(build_node_types(NODES), registry),
        cache=MemoryLRUCache(),
        on_event=on_event,
    )


async def make_client(*, mounts: MountService | None = None) -> TestClient:
    app = create_app(make_engine, SCHEMAS)
    if mounts is not None:
        add_mount_routes(app, mounts)
    add_comfy_compat_routes(app)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def test_endpoint_applies_server_attention_default() -> None:
    async def scenario() -> None:
        app = create_app(make_engine, SCHEMAS, attention_policy="flash")
        add_comfy_compat_routes(app)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post(
                "/api/compat/comfy/prompt",
                json={"client_id": "attention-client", "prompt": PROMPT},
            )
            assert response.status == 202
            submitted = await response.json()
            job = app[STATE_KEY].queue.job_for_run(submitted["jobRef"])
            assert job is not None
            assert job.attention_config == AttentionPolicyConfig("flash")
        finally:
            await client.close()

    asyncio.run(scenario())


async def make_legacy_export_client(node: type[object]) -> TestClient:
    translation = translate_mappings({node.__name__: node})
    nodes = translation.node_classes
    schemas = build_schemas(nodes)

    def make_export_engine(on_event: EventListener | None = None) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        translation.register_types(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    app = create_app(make_export_engine, schemas)
    add_comfy_compat_routes(app)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def wait_for_http_job(client: TestClient, client_id: str, job_id: str) -> dict[str, Any]:
    job: dict[str, Any] = {}
    for _ in range(100):
        status = await client.get(f"/api/jobs/{client_id}/{job_id}")
        job = await status.json()
        if job["state"] in ("completed", "failed", "cancelled"):
            return job
        await asyncio.sleep(0.02)
    return job


class _Authenticator:
    def __init__(self, principal: Principal) -> None:
        self.principal = principal

    async def authenticate(self, token: str) -> Principal | None:
        return self.principal if token == "accepted" else None


PROMPT: dict[str, Any] = {
    "1": {"class_type": "Producer", "inputs": {"seed": 3}},
    "2": {"class_type": "Consumer", "inputs": {"text": ["1", 0]}},
}


class AdmissionBool(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.admission-bool",
            inputs=(InputSpec("value", TypeExpr.concrete("core.boolean")),),
            outputs=(OutputSpec("value", TypeExpr.concrete("core.boolean")),),
            aliases=("AdmissionBool",),
        )

    @classmethod
    def execute(cls, *, value: bool) -> Mapping[str, object]:
        return cls.outputs(value=value)


class AdmissionBranch(Node):
    executed: list[str] = []

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.admission-branch",
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("value", STRING),),
            aliases=("AdmissionBranch",),
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        cls.executed.append(value)
        return cls.outputs(value=value)


class AdmissionSwitch(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.admission-switch",
            inputs=(
                InputSpec("switch", TypeExpr.concrete("core.boolean")),
                InputSpec("on_false", STRING, lazy=True),
                InputSpec("on_true", STRING, lazy=True),
            ),
            outputs=(OutputSpec("value", STRING),),
            aliases=("AdmissionSwitch",),
            selector=SelectorSpec("switch", {"false": "on_false", "true": "on_true"}),
        )

    @classmethod
    def check_lazy_status(
        cls, *, switch: bool, on_false: str | None, on_true: str | None
    ) -> tuple[str, ...] | None:
        del cls
        if switch and on_true is None:
            return ("on_true",)
        if not switch and on_false is None:
            return ("on_false",)
        return None

    @classmethod
    def execute(
        cls, *, switch: bool, on_false: str | None, on_true: str | None
    ) -> Mapping[str, object]:
        return cls.outputs(value=on_true if switch else on_false)


class AdmissionOutput(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.admission-output",
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("value", STRING),),
            aliases=("AdmissionOutput",),
            output_node=True,
        )

    @classmethod
    def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(value=value)


ADMISSION_NODES = (AdmissionBool, AdmissionBranch, AdmissionSwitch, AdmissionOutput)
ADMISSION_SCHEMAS = build_schemas(ADMISSION_NODES)


def make_admission_engine(on_event: EventListener | None = None) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    return Engine(
        schemas=ADMISSION_SCHEMAS,
        registry=registry,
        worker=InProcessWorker(build_node_types(ADMISSION_NODES), registry),
        cache=MemoryLRUCache(),
        on_event=on_event,
    )


def admission_graph(*, stored: object = Link("selector", "value")) -> Graph:
    return Graph(
        {
            "selector": GraphNode("test.admission-bool", {"value": True}),
            "on_false": GraphNode("test.admission-branch", {"value": "OFF"}),
            "on_true": GraphNode("test.admission-branch", {"value": "ON"}),
            "switch": GraphNode(
                "test.admission-switch",
                {
                    "switch": stored,
                    "on_false": Link("on_false", "value"),
                    "on_true": Link("on_true", "value"),
                },
            ),
            "output": GraphNode("test.admission-output", {"value": Link("switch", "value")}),
        }
    )


def admission_prompt(*, stored: object | None = None) -> dict[str, Any]:
    stored = ["selector", 0] if stored is None else stored
    return {
        "selector": {"class_type": "AdmissionBool", "inputs": {"value": True}},
        "on_false": {
            "class_type": "AdmissionBranch",
            "inputs": {"value": "OFF"},
        },
        "on_true": {
            "class_type": "AdmissionBranch",
            "inputs": {"value": "ON"},
        },
        "switch": {
            "class_type": "AdmissionSwitch",
            "inputs": {
                "switch": stored,
                "on_false": ["on_false", 0],
                "on_true": ["on_true", 0],
            },
        },
        "output": {
            "class_type": "AdmissionOutput",
            "inputs": {"value": ["switch", 0]},
        },
    }


def assert_admission_replay(replay: dict[str, Any], *, warm: bool) -> None:
    demands = [
        event
        for event in replay["events"]
        if event["type"] == "node_event" and event["event"] == "lazy_demand"
    ]
    assert [(event["nodeId"], event["data"]) for event in demands] == [
        (
            "switch",
            {
                "round": 1,
                "status": "waiting",
                "requestedInputs": ["on_true"],
                "newInputs": ["on_true"],
                "demandedInputs": ["on_true"],
                "producerNodes": ["on_true"],
            },
        ),
        (
            "switch",
            {
                "round": 2,
                "status": "ready",
                "requestedInputs": [],
                "newInputs": [],
                "demandedInputs": ["on_true"],
                "producerNodes": [],
            },
        ),
    ]
    assert not any(
        event["type"] == "node_started" and event.get("nodeId") == "on_false"
        for event in replay["events"]
    )
    assert (
        any(
            event["type"] == "node_cached" and event.get("nodeId") == "on_true"
            for event in replay["events"]
        )
        is warm
    )


def test_queue_selector_admission_preserves_region_runtime_selectors() -> None:
    async def scenario() -> None:
        queue = JobQueue(make_admission_engine())
        accepted = queue.submit("direct", "linked", admission_graph(), ["output"])
        assert accepted.graph == admission_graph()

        with pytest.raises(
            JobGraphAdmissionError,
            match="stored selector residue would rewrite the submitted graph or targets",
        ):
            queue.submit("direct", "stored", admission_graph(stored=True), ["output"])

        region_graph = Graph(
            {
                "region": RegionNode(
                    kind="map",
                    body=Graph(
                        {
                            "switch": GraphNode(
                                "test.admission-switch",
                                {"switch": True, "on": "ON"},
                            )
                        }
                    ),
                )
            }
        )
        accepted_region = queue.submit("direct", "region", region_graph, ["region"])
        assert accepted_region.graph == region_graph
        await queue.close()

    asyncio.run(scenario())


def test_native_http_computed_selector_admission_cold_and_warm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        AdmissionBranch.executed.clear()
        app = create_app(make_admission_engine, ADMISSION_SCHEMAS)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            submissions: list[dict[str, Any]] = []
            for job_id in ("native-cold", "native-warm"):
                response = await client.post(
                    "/api/jobs",
                    json={
                        "clientId": "native-selector",
                        "jobId": job_id,
                        "graph": graph_to_wire(admission_graph()),
                        "targets": ["output"],
                    },
                )
                assert response.status == 202
                submitted = await response.json()
                submissions.append(submitted)
                completed = await wait_for_http_job(client, "native-selector", job_id)
                assert completed["state"] == "completed"
                job = app[STATE_KEY].queue.job_for_run(submitted["jobRef"])
                assert job is not None and job.result is not None
                assert job.result.outputs["output"]["value"].resolve() == "ON"

            assert AdmissionBranch.executed == ["ON"]
            for index, submitted in enumerate(submissions):
                replay = await (
                    await client.get(f"/api/jobs/by-ref/{submitted['jobRef']}/events")
                ).json()
                assert_admission_replay(replay, warm=index == 1)

            malformed = admission_graph(stored=True)
            switch = malformed.nodes["switch"]
            assert isinstance(switch, GraphNode)
            malformed = Graph(
                {**malformed.nodes, "switch": GraphNode(switch.node_type, {"switch": True})}
            )
            response = await client.post(
                "/api/jobs",
                json={
                    "clientId": "native-selector",
                    "jobId": "native-malformed",
                    "graph": graph_to_wire(malformed),
                    "targets": ["output"],
                },
            )
            assert response.status == 400
            assert (await response.json())["error"] == "selector-lowering"

            def refuse_residue(*_args: object, **_kwargs: object) -> None:
                raise JobGraphAdmissionError("stored selector residue")

            monkeypatch.setattr(app[STATE_KEY].queue, "submit", refuse_residue)
            residue = await client.post(
                "/api/jobs",
                json={
                    "clientId": "native-selector",
                    "jobId": "native-residue",
                    "graph": graph_to_wire(admission_graph()),
                    "targets": ["output"],
                },
            )
            assert residue.status == 400
            assert await residue.json() == {
                "error": "selector-admission",
                "message": "stored selector residue",
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_compat_http_computed_selector_admission_cold_and_warm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        AdmissionBranch.executed.clear()
        app = create_app(make_admission_engine, ADMISSION_SCHEMAS)
        add_comfy_compat_routes(app)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            submissions: list[dict[str, Any]] = []
            for _ in range(2):
                response = await client.post(
                    "/api/compat/comfy/prompt",
                    json={"client_id": "compat-selector", "prompt": admission_prompt()},
                )
                assert response.status == 202
                submitted = await response.json()
                submissions.append(submitted)
                completed = await wait_for_http_job(client, "compat-selector", submitted["jobId"])
                assert completed["state"] == "completed"
                job = app[STATE_KEY].queue.job_for_run(submitted["jobRef"])
                assert job is not None and job.result is not None
                assert job.result.outputs["output"]["value"].resolve() == "ON"

            assert AdmissionBranch.executed == ["ON"]
            for index, submitted in enumerate(submissions):
                replay = await (
                    await client.get(f"/api/jobs/by-ref/{submitted['jobRef']}/events")
                ).json()
                assert_admission_replay(replay, warm=index == 1)

            malformed = admission_prompt(stored=True)
            malformed["switch"]["inputs"] = {"switch": True}
            response = await client.post(
                "/api/compat/comfy/prompt",
                json={"client_id": "compat-selector", "prompt": malformed},
            )
            assert response.status == 400
            assert (await response.json())["problems"][0]["code"] == "prompt.missing_branch"

            def refuse_residue(*_args: object, **_kwargs: object) -> None:
                raise JobGraphAdmissionError("stored selector residue")

            monkeypatch.setattr(app[STATE_KEY].queue, "submit", refuse_residue)
            residue = await client.post(
                "/api/compat/comfy/prompt",
                json={"client_id": "compat-selector", "prompt": admission_prompt()},
            )
            assert residue.status == 400
            assert await residue.json() == {"error": "stored selector residue"}

            def refuse_conflict(*_args: object, **_kwargs: object) -> None:
                raise ValueError("job key conflict")

            monkeypatch.setattr(app[STATE_KEY].queue, "submit", refuse_conflict)
            conflict = await client.post(
                "/api/compat/comfy/prompt",
                json={"client_id": "compat-selector", "prompt": admission_prompt()},
            )
            assert conflict.status == 409
            assert await conflict.json() == {"error": "job key conflict"}
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_compat_save_materializes_exact_submitted_metadata(tmp_path: Path) -> None:
    async def scenario() -> None:
        CompatMetadataSave.seen.clear()
        client = await make_legacy_export_client(CompatMetadataSave)
        output = tmp_path / "metadata.png"
        prompt = {
            "save": {
                "class_type": "CompatMetadataSave",
                "inputs": {"filename": str(output)},
            }
        }
        # Keep the extra metadata order deliberately non-sorted. Upstream
        # serializes each value with plain json.dumps and preserves insertion.
        extra_pnginfo = {
            "workflow": {"nodes": [2, 1], "links": {"z": 0, "a": 1}},
            "parameters": {"beta": 2, "alpha": 1},
        }
        try:
            response = await client.post(
                "/api/compat/comfy/prompt",
                json={
                    "prompt": prompt,
                    "client_id": "metadata-client",
                    "extra_data": {"extra_pnginfo": extra_pnginfo},
                },
            )
            assert response.status == 202
            submitted = await response.json()
            job = await wait_for_http_job(client, "metadata-client", submitted["jobId"])
            assert job["state"] == "completed"
            assert CompatMetadataSave.seen == [(prompt, extra_pnginfo)]
            with Image.open(output) as image:
                assert image.info == {
                    "prompt": json.dumps(prompt),
                    "workflow": json.dumps(extra_pnginfo["workflow"]),
                    "parameters": json.dumps(extra_pnginfo["parameters"]),
                }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_compat_save_without_extra_data_receives_none(tmp_path: Path) -> None:
    async def scenario() -> None:
        CompatMetadataSave.seen.clear()
        client = await make_legacy_export_client(CompatMetadataSave)
        output = tmp_path / "prompt-only.png"
        prompt = {
            "save": {
                "class_type": "CompatMetadataSave",
                "inputs": {"filename": str(output)},
            }
        }
        try:
            response = await client.post(
                "/api/compat/comfy/prompt",
                json={"prompt": prompt, "client_id": "prompt-only"},
            )
            submitted = await response.json()
            job = await wait_for_http_job(client, "prompt-only", submitted["jobId"])
            assert job["state"] == "completed"
            assert CompatMetadataSave.seen == [(prompt, None)]
            with Image.open(output) as image:
                assert image.info == {"prompt": json.dumps(prompt)}
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_compat_save_input_is_list_wraps_export_snapshot(tmp_path: Path) -> None:
    async def scenario() -> None:
        CompatMetadataSaveList.seen.clear()
        client = await make_legacy_export_client(CompatMetadataSaveList)
        prompt = {
            "save": {
                "class_type": "CompatMetadataSaveList",
                "inputs": {"filename": [str(tmp_path / "unused.png")]},
            }
        }
        extra_pnginfo = {"workflow": {"nodes": []}}
        try:
            response = await client.post(
                "/api/compat/comfy/prompt",
                json={
                    "prompt": prompt,
                    "client_id": "list-hidden",
                    "extra_data": {"extra_pnginfo": extra_pnginfo},
                },
            )
            submitted = await response.json()
            job = await wait_for_http_job(client, "list-hidden", submitted["jobId"])
            assert job["state"] == "completed"
            assert CompatMetadataSaveList.seen == [([prompt], [extra_pnginfo])]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_submits_prompt_as_native_job() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.post(
                "/api/compat/comfy/prompt", json={"prompt": PROMPT, "client_id": "me"}
            )
            assert resp.status == 202
            body = await resp.json()
            assert body["clientId"] == "me"
            assert body["scope"] == "local"
            assert body["targets"] == ["2"]
            graph = body["graph"]
            assert graph["nodes"]["2"]["nodeType"] == "comfy.Consumer"
            assert graph["nodes"]["2"]["inputs"]["text"] == {
                "$link": {"node": "1", "output": "first"}
            }
            # The job runs as an ordinary native job to completion.
            job: dict[str, Any] = {}
            for _ in range(100):
                status = await client.get(f"/api/jobs/me/{body['jobId']}")
                job = await status.json()
                if job["state"] in ("completed", "failed", "cancelled"):
                    break
                await asyncio.sleep(0.02)
            assert job["state"] == "completed"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_accepts_flat_recursive_dynamic_combo_vectors() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.post(
                "/api/compat/comfy/prompt",
                json={
                    "dynamic": {
                        "class_type": "DynamicConsumer",
                        "inputs": {
                            "combo": "outer",
                            "combo.subcombo": "inner",
                            "combo.subcombo.value": 7,
                        },
                    }
                },
            )
            assert resp.status == 202
            body = await resp.json()
            node = body["graph"]["nodes"]["dynamic"]
            assert node["inputs"] == {"combo.subcombo.value": 7}
            assert node["slotVariants"] == {
                "combo": "outer",
                "combo.subcombo": "inner",
            }
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("choice", "family_input"),
    [
        ("family", "combo.frame.image0"),
        ("names", "combo.named.right"),
    ],
)
def test_endpoint_accepts_family_in_combo_vectors(choice: str, family_input: str) -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.post(
                "/api/compat/comfy/prompt",
                json={
                    "dynamic": {
                        "class_type": "DynamicConsumer",
                        "inputs": {"combo": choice, family_input: 7},
                    }
                },
            )
            assert resp.status == 202
            body = await resp.json()
            node = body["graph"]["nodes"]["dynamic"]
            assert node["inputs"] == {family_input: 7}
            assert node["slotVariants"] == {"combo": choice}
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_resolves_x_dinkster_scope_and_stamps_principal() -> None:
    async def scenario() -> None:
        principal = Principal(
            "alice",
            {
                "a": frozenset({"jobs:submit"}),
                "b": frozenset({"jobs:submit"}),
            },
            kind="agent",
        )
        app = create_app(make_engine, SCHEMAS, authenticator=_Authenticator(principal))
        add_comfy_compat_routes(app)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            auth = {"Authorization": "Bearer accepted"}
            ambiguous = await client.post("/api/compat/comfy/prompt", json=PROMPT, headers=auth)
            assert ambiguous.status == 400
            denied = await client.post(
                "/api/compat/comfy/prompt",
                json=PROMPT,
                headers={**auth, "X-Dinkster-Scope": "c"},
            )
            assert denied.status == 403
            accepted = await client.post(
                "/api/compat/comfy/prompt",
                json=PROMPT,
                headers={**auth, "X-Dinkster-Scope": "a"},
            )
            wire = await accepted.json()
            assert accepted.status == 202
            assert wire["scope"] == "a"
            assert wire["submittedBy"] == {"principalId": "alice", "kind": "agent"}
            job = app[STATE_KEY].queue.job_for_run(wire["jobRef"])
            assert job is not None
            assert (job.scope, job.principal_id, job.principal_kind) == ("a", "alice", "agent")
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_dry_run_translates_without_submitting() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.post("/api/compat/comfy/prompt?dryRun=1", json=PROMPT)
            assert resp.status == 200
            body = await resp.json()
            assert set(body) == {"graph", "targets"}
            listing = await client.get("/api/jobs")
            assert (await listing.json())["jobs"] == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_empty_latent_source_and_native_names_lower_identically() -> None:
    from dinkster_compat_comfy.native_arm import GenerationEmptyLatentImage

    nodes = (GenerationEmptyLatentImage, LatentSink)
    schemas = build_schemas(nodes)
    assert "comfy.EmptyLatentImage" not in schemas

    def engine(on_event: EventListener | None = None) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        app = create_app(engine, schemas)
        add_comfy_compat_routes(app)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            results = []
            for name in ("EmptyLatentImage", "dinkster.empty_latent_image"):
                response = await client.post(
                    "/api/compat/comfy/prompt?dryRun=1",
                    json={
                        "latent": {
                            "class_type": name,
                            "inputs": {"width": 768, "height": 512, "batch_size": 2},
                        },
                        "sink": {
                            "class_type": "LatentSink",
                            "inputs": {"latent": ["latent", 0]},
                        },
                    },
                )
                assert response.status == 200, await response.text()
                results.append(await response.json())
            assert results[0] == results[1]
            graph = results[0]["graph"]
            assert graph["nodes"]["latent"]["nodeType"] == "dinkster.empty_latent_image"
            assert graph["nodes"]["latent"]["inputs"] == {
                "width": 768,
                "height": 512,
                "batch_size": 2,
            }
            assert results[0]["targets"] == ["sink"]
            listing = await client.get("/api/jobs")
            assert (await listing.json())["jobs"] == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_dry_run_returns_typed_problem_for_skipped_class() -> None:
    diagnostic = CompatGateDiagnostic(
        code="compat.dynamic.unsupported",
        source_node="SkippedNode",
        reason="SkippedNode: unsupported V3 dynamic input kind COMFY_FUTURE_V3",
        source_generation="v3",
        path_kind="declared",
        input_id="value",
        input_path=("value",),
        lazy=False,
        input_is_list=False,
        output_is_list=False,
        raw_link=False,
        accept_all=False,
    )

    async def scenario() -> None:
        app = create_app(
            make_engine,
            SCHEMAS,
            compat_skips={"comfy": {"SkippedNode": diagnostic}},
        )
        add_comfy_compat_routes(app)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            for class_type in ("SkippedNode", "comfy.SkippedNode"):
                response = await client.post(
                    "/api/compat/comfy/prompt?dryRun=1",
                    json={"skipped": {"class_type": class_type, "inputs": {}}},
                )
                assert response.status == 400
                body = await response.json()
                assert body["problems"] == [
                    {
                        "code": diagnostic.code,
                        "message": diagnostic.reason,
                        "nodeId": "skipped",
                        "inputId": "value",
                        "compatDiagnostic": {
                            **diagnostic.to_wire(),
                            "schemaEpoch": 1,
                            "extensionSnapshotDigest": app[
                                STATE_KEY
                            ].engine.extension_snapshot_digest,
                        },
                    }
                ]
            listing = await client.get("/api/jobs")
            assert (await listing.json())["jobs"] == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_applies_save_image_adapter() -> None:
    """The HTTP boundary converts legacy filename_prefix - the stored/
    executed graph carries the structured target, never the raw string."""

    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.post(
                "/api/compat/comfy/prompt?dryRun=1",
                json=save_prompt(filename_prefix="renders/scene"),
            )
            assert resp.status == 200
            body = await resp.json()
            inputs = body["graph"]["nodes"]["2"]["inputs"]
            assert inputs["target"] == {"mount": "comfy-output", "prefix": "renders/scene"}
            assert "filename_prefix" not in inputs

            bad = await client.post(
                "/api/compat/comfy/prompt",
                json=save_prompt(filename_prefix="../escape"),
            )
            assert bad.status == 400
            problems = (await bad.json())["problems"]
            assert problems[0]["code"] == "prompt.save_target.invalid"
            assert problems[0]["inputId"] == "filename_prefix"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_resolves_image_through_comfy_input_mount(tmp_path: Path) -> None:
    """The HTTP boundary swaps a legacy image filename for the mounted
    file's digest identity - the stored graph carries no path, and the
    digest is the actual bytes' digest."""
    root = tmp_path / "input"
    root.mkdir()
    (root / "img.png").write_bytes(b"png-bytes")
    table = MountTable(tmp_path / "snap.json")
    table.add(MountDef(id="comfy-input", path=root))
    table.scan("comfy-input")

    async def scenario() -> None:
        client = await make_client(mounts=MountService(table))
        try:
            resp = await client.post(
                "/api/compat/comfy/prompt?dryRun=1", json=load_prompt("img.png")
            )
            assert resp.status == 200
            image = (await resp.json())["graph"]["nodes"]["1"]["inputs"]["image"]
            assert image["digest"] == digest_bytes(b"png-bytes")
            assert image["virtualPath"] == "mounts/comfy-input/img.png"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_resolves_ckpt_name_through_category_mount(
    tmp_path: Path,
) -> None:
    """The HTTP boundary swaps a legacy ckpt_name for the mounted file's
    digest identity on the renamed 'checkpoint' input - exact-path under
    exactly one derived checkpoint root."""
    root = tmp_path / "models" / "checkpoints"
    (root / "sd15").mkdir(parents=True)
    (root / "sd15" / "model.safetensors").write_bytes(b"weights")
    table = MountTable(tmp_path / "snap.json")
    mount_id = "comfy-model-checkpoints-1"
    table.add(
        MountDef(id=mount_id, path=root),
        source="derived",
        kind="model/checkpoint",
    )
    table.scan(mount_id)

    async def scenario() -> None:
        client = await make_client(mounts=MountService(table))
        try:
            resp = await client.post(
                "/api/compat/comfy/prompt?dryRun=1",
                json=ckpt_prompt("sd15/model.safetensors"),
            )
            assert resp.status == 200
            inputs = (await resp.json())["graph"]["nodes"]["1"]["inputs"]
            assert "ckpt_name" not in inputs
            checkpoint = inputs["checkpoint"]
            assert checkpoint["digest"] == digest_bytes(b"weights")
            assert (
                checkpoint["virtualPath"]
                == "mounts/comfy-model-checkpoints-1/sd15/model.safetensors"
            )
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_uses_indexed_model_while_scan_continues_and_names_unindexed_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "models" / "checkpoints"
    root.mkdir(parents=True)
    cached = root / "cached.safetensors"
    cached.write_bytes(b"cached weights")
    table = MountTable(
        tmp_path / "library" / "mounts.json",
        index_root=tmp_path / "library" / "asset-indexes",
    )
    mount_id = "comfy-model-checkpoints-1"
    table.add(
        MountDef(id=mount_id, path=root),
        source="derived",
        kind="model/checkpoint",
    )
    table.scan(mount_id)

    slow = root / "slow.safetensors"
    slow.write_bytes(b"slow weights")
    import dinkster_assets.library as library_module

    hashing = Event()
    release = Event()

    def held_digest(path: Path):
        if path == slow:
            hashing.set()
            assert release.wait(5)
        return real_digest_file_with_record(path)

    monkeypatch.setattr(library_module, "digest_file_with_record", held_digest)

    async def scenario() -> None:
        client = await make_client(mounts=MountService(table))
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(table.scan, mount_id, progress_interval=0)
                assert await asyncio.to_thread(hashing.wait, 5)

                ready = await client.post(
                    "/api/compat/comfy/prompt?dryRun=1",
                    json=ckpt_prompt("cached.safetensors"),
                )
                assert ready.status == 200
                checkpoint = (await ready.json())["graph"]["nodes"]["1"]["inputs"]["checkpoint"]
                assert checkpoint["digest"] == digest_bytes(b"cached weights")

                pending = await client.post(
                    "/api/compat/comfy/prompt?dryRun=1",
                    json=ckpt_prompt("slow.safetensors"),
                )
                assert pending.status == 400
                problem = (await pending.json())["problems"][0]
                assert "slow.safetensors" in problem["message"]
                assert "got str" not in problem["message"]

                release.set()
                assert future.result(timeout=5) == 2
        finally:
            release.set()
            await client.close()

    asyncio.run(scenario())


def test_endpoint_resolves_translated_model_asset_input_without_renaming(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models" / "controlnet"
    model = root / "canny.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"control weights")
    table = MountTable(tmp_path / "snap.json")
    mount_id = "comfy-model-controlnet-1"
    table.add(
        MountDef(id=mount_id, path=root),
        source="derived",
        kind="model/controlnet",
    )
    table.scan(mount_id)

    async def scenario() -> None:
        client = await make_client(mounts=MountService(table))
        try:
            response = await client.post(
                "/api/compat/comfy/prompt?dryRun=1",
                json={
                    "load": {
                        "class_type": "TranslatedModelLoader",
                        "inputs": {"control_net_name": "canny.safetensors"},
                    }
                },
            )
            assert response.status == 200
            asset = (await response.json())["graph"]["nodes"]["load"]["inputs"]["control_net_name"]
            assert asset["digest"] == digest_bytes(b"control weights")
            assert asset["virtualPath"] == ("mounts/comfy-model-controlnet-1/canny.safetensors")
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_refuses_same_model_name_in_two_category_roots(
    tmp_path: Path,
) -> None:
    table = MountTable(tmp_path / "snap.json")
    for index, payload in ((1, b"first"), (2, b"second")):
        root = tmp_path / f"checkpoints-{index}"
        root.mkdir()
        (root / "same.safetensors").write_bytes(payload)
        mount_id = f"comfy-model-checkpoints-{index}"
        table.add(
            MountDef(id=mount_id, path=root),
            source="derived",
            kind="model/checkpoint",
        )
        table.scan(mount_id)

    async def scenario() -> None:
        client = await make_client(mounts=MountService(table))
        try:
            response = await client.post(
                "/api/compat/comfy/prompt?dryRun=1",
                json=ckpt_prompt("same.safetensors"),
            )
            assert response.status == 400
            problems = (await response.json())["problems"]
            assert [problem["code"] for problem in problems] == [
                "prompt.load_checkpoint.unresolved"
            ]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_without_mounts_refuses_image_filename() -> None:
    """No mounts service means no catalog to resolve against: the legacy
    filename is refused loudly, never passed through as a raw string."""

    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.post("/api/compat/comfy/prompt", json=load_prompt("img.png"))
            assert resp.status == 400
            problems = (await resp.json())["problems"]
            assert problems[0]["code"] == "prompt.load_image.unresolved"
            assert problems[0]["nodeId"] == "1"
            assert problems[0]["inputId"] == "image"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_endpoint_translation_failure_returns_anchored_problems() -> None:
    async def scenario() -> None:
        client = await make_client()
        try:
            resp = await client.post(
                "/api/compat/comfy/prompt",
                json={"n": {"class_type": "NoSuchNode", "inputs": {}}},
            )
            assert resp.status == 400
            body = await resp.json()
            codes = {p["code"] for p in body["problems"]}
            assert "prompt.unknown_class_type" in codes
            assert body["problems"][0]["nodeId"] == "n"
        finally:
            await client.close()

    asyncio.run(scenario())


# -- legacy conditioning adapters ----------------------------------------------

CONDITIONING = TypeExpr.concrete("dinkster.conditioning")


class CondSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.CondSource",
            outputs=(OutputSpec("conditioning", CONDITIONING),),
            aliases=("CondSource",),
        )

    @classmethod
    async def execute(cls) -> Mapping[str, object]:
        return cls.outputs(conditioning=None)


class CondSink(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.CondSink",
            inputs=(InputSpec("conditioning", CONDITIONING),),
            outputs=(OutputSpec("out", STRING),),
            aliases=("CondSink",),
            output_node=True,
        )

    @classmethod
    async def execute(cls, *, conditioning: object) -> Mapping[str, object]:
        return cls.outputs(out=str(conditioning))


def _conditioning_schemas() -> dict[str, NodeSchema]:
    from dinkster_nodes_generation.nodes import (
        ConditioningMerge,
        ConditioningScale,
        ConditioningSetArea,
        ConditioningSetMask,
        ConditioningSetTimestepRange,
        ConditioningZeroOut,
    )

    return build_schemas(
        (
            CondSource,
            CondSink,
            ConditioningMerge,
            ConditioningScale,
            ConditioningSetArea,
            ConditioningSetMask,
            ConditioningSetTimestepRange,
            ConditioningZeroOut,
        )
    )


def _conditioning_prompt(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "a": {"class_type": "CondSource", "inputs": {}},
        "b": {"class_type": "CondSource", "inputs": {}},
        "op": node,
        "sink": {"class_type": "CondSink", "inputs": {"conditioning": ["op", 0]}},
    }


def test_legacy_conditioning_combine_translates_to_merge_combo() -> None:
    schemas = _conditioning_schemas()
    translated = translate_prompt(
        _conditioning_prompt(
            {
                "class_type": "ConditioningCombine",
                "inputs": {"conditioning_1": ["a", 0], "conditioning_2": ["b", 0]},
            }
        ),
        schemas,
        input_adapters=COMFY_INPUT_ADAPTERS,
    )
    node = translated.graph.nodes["op"]
    assert isinstance(node, GraphNode)
    assert node.node_type == "dinkster.conditioning_merge"
    assert node.slot_variants == {"mode": "combine"}
    assert node.inputs == {
        "mode.inputs.conditioning_1": Link("a", "conditioning"),
        "mode.inputs.conditioning_2": Link("b", "conditioning"),
    }
    assert translated.targets == ("sink",)

    native = translate_prompt(
        _conditioning_prompt(
            {
                "class_type": "dinkster.conditioning_merge",
                "inputs": {
                    "mode": "combine",
                    "mode.inputs.conditioning_1": ["a", 0],
                    "mode.inputs.conditioning_2": ["b", 0],
                },
            }
        ),
        schemas,
        input_adapters=COMFY_INPUT_ADAPTERS,
    )
    assert native.graph.nodes["op"] == node


def test_legacy_conditioning_average_and_concat_translate_to_merge_combo() -> None:
    schemas = _conditioning_schemas()
    averaged = translate_prompt(
        _conditioning_prompt(
            {
                "class_type": "ConditioningAverage",
                "inputs": {
                    "conditioning_to": ["a", 0],
                    "conditioning_from": ["b", 0],
                    "conditioning_to_strength": 0.25,
                },
            }
        ),
        schemas,
        input_adapters=COMFY_INPUT_ADAPTERS,
    )
    node = averaged.graph.nodes["op"]
    assert isinstance(node, GraphNode)
    assert node.node_type == "dinkster.conditioning_merge"
    assert node.slot_variants == {"mode": "average"}
    assert node.inputs == {
        "mode.conditioning_to": Link("a", "conditioning"),
        "mode.conditioning_from": Link("b", "conditioning"),
        "mode.conditioning_to_strength": 0.25,
    }

    concatenated = translate_prompt(
        _conditioning_prompt(
            {
                "class_type": "ConditioningConcat",
                "inputs": {"conditioning_to": ["a", 0], "conditioning_from": ["b", 0]},
            }
        ),
        schemas,
        input_adapters=COMFY_INPUT_ADAPTERS,
    )
    node = concatenated.graph.nodes["op"]
    assert isinstance(node, GraphNode)
    assert node.slot_variants == {"mode": "concat"}
    assert node.inputs == {
        "mode.conditioning_to": Link("a", "conditioning"),
        "mode.conditioning_from": Link("b", "conditioning"),
    }


def test_legacy_conditioning_set_area_translates_units_variants() -> None:
    schemas = _conditioning_schemas()
    pixels = translate_prompt(
        _conditioning_prompt(
            {
                "class_type": "ConditioningSetArea",
                "inputs": {
                    "conditioning": ["a", 0],
                    "width": 128,
                    "height": 256,
                    "x": 8,
                    "y": 16,
                    "strength": 0.75,
                },
            }
        ),
        schemas,
        input_adapters=COMFY_INPUT_ADAPTERS,
    )
    node = pixels.graph.nodes["op"]
    assert isinstance(node, GraphNode)
    assert node.node_type == "dinkster.conditioning_set_area"
    assert node.slot_variants == {"units": "pixels"}
    assert node.inputs == {
        "conditioning": Link("a", "conditioning"),
        "strength": 0.75,
        "units.width": 128,
        "units.height": 256,
        "units.x": 8,
        "units.y": 16,
    }

    percent = translate_prompt(
        _conditioning_prompt(
            {
                "class_type": "ConditioningSetAreaPercentage",
                "inputs": {
                    "conditioning": ["a", 0],
                    "width": 0.5,
                    "height": 0.25,
                    "x": 0.125,
                    "y": 0.0,
                    "strength": 1.0,
                },
            }
        ),
        schemas,
        input_adapters=COMFY_INPUT_ADAPTERS,
    )
    node = percent.graph.nodes["op"]
    assert isinstance(node, GraphNode)
    assert node.node_type == "dinkster.conditioning_set_area"
    assert node.slot_variants == {"units": "percent"}
    assert node.inputs["units.width"] == 0.5
    assert node.inputs["units.y"] == 0.0

    video = translate_prompt(
        _conditioning_prompt(
            {
                "class_type": "ConditioningSetAreaPercentageVideo",
                "inputs": {
                    "conditioning": ["a", 0],
                    "width": 0.5,
                    "height": 0.25,
                    "temporal": 0.75,
                    "x": 0.125,
                    "y": 0.0,
                    "z": 0.25,
                    "strength": 1.0,
                },
            }
        ),
        schemas,
        input_adapters=COMFY_INPUT_ADAPTERS,
    )
    node = video.graph.nodes["op"]
    assert isinstance(node, GraphNode)
    assert node.node_type == "dinkster.conditioning_set_area"
    assert node.slot_variants == {"units": "percent-video"}
    assert node.inputs == {
        "conditioning": Link("a", "conditioning"),
        "strength": 1.0,
        "units.width": 0.5,
        "units.height": 0.25,
        "units.temporal": 0.75,
        "units.x": 0.125,
        "units.y": 0.0,
        "units.z": 0.25,
    }


@pytest.mark.parametrize(
    ("class_type", "inputs", "code"),
    [
        (
            "ConditioningCombine",
            {"conditioning_1": ["a", 0], "mode": "combine"},
            "prompt.conditioning_merge.conflict",
        ),
        (
            "ConditioningAverage",
            {
                "conditioning_to": ["a", 0],
                "mode.conditioning_from": ["b", 0],
                "conditioning_to_strength": 0.5,
            },
            "prompt.conditioning_merge.conflict",
        ),
        (
            "ConditioningSetArea",
            {"conditioning": ["a", 0], "units.width": 128, "strength": 1.0},
            "prompt.conditioning_set_area.conflict",
        ),
    ],
)
def test_legacy_conditioning_refuses_mixed_legacy_and_native_keys(
    class_type: str, inputs: dict[str, Any], code: str
) -> None:
    with pytest.raises(PromptTranslationError) as excinfo:
        translate_prompt(
            _conditioning_prompt({"class_type": class_type, "inputs": inputs}),
            _conditioning_schemas(),
            input_adapters=COMFY_INPUT_ADAPTERS,
        )
    assert code in {problem.code for problem in excinfo.value.problems}


def test_legacy_flat_conditioning_aliases_translate_without_adapters() -> None:
    schemas = _conditioning_schemas()
    translated = translate_prompt(
        {
            "a": {"class_type": "CondSource", "inputs": {}},
            "scale": {
                "class_type": "ConditioningMultiply",
                "inputs": {"conditioning": ["a", 0], "multiplier": 2.0},
            },
            "mask": {
                "class_type": "ConditioningSetMask",
                "inputs": {
                    "conditioning": ["scale", 0],
                    "mask": ["a", 0],
                    "strength": 0.65,
                    "set_cond_area": "mask bounds",
                },
            },
            "range": {
                "class_type": "ConditioningSetTimestepRange",
                "inputs": {"conditioning": ["mask", 0], "start": 0.2, "end": 0.7},
            },
            "zero": {
                "class_type": "ConditioningZeroOut",
                "inputs": {"conditioning": ["range", 0]},
            },
            "sink": {"class_type": "CondSink", "inputs": {"conditioning": ["zero", 0]}},
        },
        schemas,
        input_adapters=COMFY_INPUT_ADAPTERS,
    )
    graph = translated.graph
    scale_node = graph.nodes["scale"]
    mask_node = graph.nodes["mask"]
    range_node = graph.nodes["range"]
    zero_node = graph.nodes["zero"]
    assert isinstance(scale_node, GraphNode)
    assert isinstance(mask_node, GraphNode)
    assert isinstance(range_node, GraphNode)
    assert isinstance(zero_node, GraphNode)
    assert scale_node.node_type == "dinkster.conditioning_scale"
    assert scale_node.inputs == {"conditioning": Link("a", "conditioning"), "multiplier": 2.0}
    assert mask_node.node_type == "dinkster.conditioning_set_mask"
    assert mask_node.inputs == {
        "conditioning": Link("scale", "conditioning"),
        "mask": Link("a", "conditioning"),
        "strength": 0.65,
        "set_cond_area": "mask bounds",
    }
    assert range_node.node_type == "dinkster.conditioning_set_timestep_range"
    assert range_node.inputs == {
        "conditioning": Link("mask", "conditioning"),
        "start": 0.2,
        "end": 0.7,
    }
    assert zero_node.node_type == "dinkster.conditioning_zero_out"
    assert zero_node.inputs == {"conditioning": Link("range", "conditioning")}


def test_endpoint_load_latent_source_and_native_inputs_lower_identically(tmp_path: Path) -> None:
    from dinkster_compat_comfy.native import LoadLatent

    root = tmp_path / "output"
    root.mkdir()
    (root / "sample.latent").write_bytes(b"latent")
    table = MountTable(tmp_path / "mounts.json")
    table.add(MountDef(id="comfy-output", path=root))
    table.scan("comfy-output")
    ref = table.ref("mounts/comfy-output/sample.latent")
    nodes = (LoadLatent, LatentSink)
    schemas = build_schemas(nodes)

    def engine(on_event: EventListener | None = None) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        app = create_app(engine, schemas)
        add_mount_routes(app, MountService(table))
        add_comfy_compat_routes(app)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            results = []
            for name, inputs in (
                ("LoadLatent", {"latent": "sample.latent"}),
                ("comfy.LoadLatent", {"latent": ref.to_wire()}),
                ("dinkster.load_latent", {"asset": ref.to_wire()}),
            ):
                response = await client.post(
                    "/api/compat/comfy/prompt?dryRun=1",
                    json={
                        "source": {"class_type": name, "inputs": inputs},
                        "sink": {
                            "class_type": "LatentSink",
                            "inputs": {"latent": ["source", 0]},
                        },
                    },
                )
                assert response.status == 200, await response.text()
                results.append(await response.json())
            assert results[0] == results[1] == results[2]
            graph = results[0]["graph"]
            assert graph["nodes"]["source"]["nodeType"] == "dinkster.load_latent"
            assert graph["nodes"]["source"]["inputs"] == {"asset": ref.to_wire()}
            assert results[0]["targets"] == ["sink"]
            translated = translate_prompt(
                {
                    "source": {"class_type": "LoadLatent", "inputs": {"asset": ref.to_wire()}},
                    "sink": {"class_type": "LatentSink", "inputs": {"latent": ["source", 0]}},
                },
                schemas,
            )
            assert translated.graph.nodes["sink"].inputs == {"latent": Link("source", "samples")}
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("raw", ["../sample.latent", "/sample.latent", "C:\\sample.latent", 7])
def test_load_latent_invalid_names_keep_legacy_error_location(raw: object) -> None:
    adapter = make_load_latent_adapter(fake_resolver({}))
    inputs = {"latent": raw}
    adapted, problems = adapter("source", inputs)
    assert adapted == inputs
    assert [(p.code, p.node_id, p.input_id) for p in problems] == [
        ("prompt.load_latent.invalid", "source", "latent")
    ]


def test_load_latent_exact_resolution_and_native_values() -> None:
    ref = fake_ref("sample.latent")
    seen: list[str] = []
    adapter = make_load_latent_adapter(fake_resolver({"sub/sample.latent": ref}, seen))
    adapted, problems = adapter("source", {"latent": "sub\\sample.latent"})
    assert not problems
    assert adapted == {"asset": ref.to_wire()}
    assert seen == ["sub/sample.latent"]
    for raw in (ref, ref.to_wire(), Link("asset", "out"), None):
        inputs: dict[str, object] = {"latent": raw}
        adapted, problems = adapter("source", inputs)
        assert not problems
        assert adapted == {"asset": ref.to_wire() if isinstance(raw, AssetRef) else raw}
        assert inputs == {"latent": raw}
    native: dict[str, object] = {"asset": ref.to_wire()}
    assert adapter("source", native) == (native, [])
    assert seen == ["sub/sample.latent"]
    _, problems = adapter("source", {"latent": "sample.latent"})
    assert [(p.code, p.node_id, p.input_id) for p in problems] == [
        ("prompt.load_latent.unresolved", "source", "latent")
    ]
    assert "comfy-output" in problems[0].message
    _, problems = adapter("source", {"latent": ref.to_wire(), "asset": ref.to_wire()})
    assert problems[0].code == "prompt.load_latent.invalid"
