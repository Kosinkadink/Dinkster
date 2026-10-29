"""Compatibility lowering for Ultimate SD Upscale workflows."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest
from dinkster_assets import AssetRef, register_asset_type
from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy import (
    UltimateSDUpscaleCarrier,
    UltimateSDUpscaleNoUpscaleCarrier,
    UpscaleModelLoaderCarrier,
    make_model_asset_inputs_adapter,
    translate_prompt,
)
from dinkster_compat_comfy.prompt import PromptTranslationError
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, RegionNode, TypedLiteral, graph_to_wire, validate
from dinkster_nodes_foundation import FOUNDATION_NODES
from dinkster_nodes_generation import GENERATION_NODES
from dinkster_nodes_image import IMAGE_NODES
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_server.queue import Job, JobQueue
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

IMAGE = TypeExpr.concrete("dinkster.image")
MODEL = TypeExpr.concrete("dinkster.model")
CONDITIONING = TypeExpr.concrete("dinkster.conditioning")
VAE = TypeExpr.concrete("dinkster.vae")
ASSET = TypeExpr.concrete("dinkster.asset")


class CompatInputs(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.usdu_inputs",
            aliases=("USDUInputs",),
            outputs=(
                OutputSpec("image", IMAGE),
                OutputSpec("model", MODEL),
                OutputSpec("positive", CONDITIONING),
                OutputSpec("negative", CONDITIONING),
                OutputSpec("vae", VAE),
            ),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        raise RuntimeError("schema-only test node")


class CompatImageOutput(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.usdu_output",
            aliases=("USDUOutput",),
            inputs=(InputSpec("image", IMAGE),),
            outputs=(),
            output_node=True,
        )

    @classmethod
    def execute(cls, *, image: object) -> Mapping[str, object]:
        del image
        return cls.outputs()


class CompatImageUpscaleWithModel(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.ImageUpscaleWithModel",
            aliases=("ImageUpscaleWithModel",),
            inputs=(InputSpec("upscale_model", ASSET), InputSpec("image", IMAGE)),
            outputs=(OutputSpec("IMAGE", IMAGE),),
        )

    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError("schema-only test node")


TEST_NODES = (
    CompatInputs,
    CompatImageOutput,
    CompatImageUpscaleWithModel,
    UltimateSDUpscaleCarrier,
    UltimateSDUpscaleNoUpscaleCarrier,
    UpscaleModelLoaderCarrier,
)
SCHEMAS = build_schemas((*FOUNDATION_NODES, *IMAGE_NODES, *GENERATION_NODES, *TEST_NODES))
ASSET_REF = AssetRef(
    digest=f"blake3:{'a' * 64}",
    name="4x-UltraSharp.pth",
    size=123,
    media_type="application/octet-stream",
    virtual_path="mounts/upscalers/4x-UltraSharp.pth",
)


def _base_inputs() -> dict[str, object]:
    return {
        "model": ["source", 1],
        "positive": ["source", 2],
        "negative": ["source", 3],
        "vae": ["source", 4],
        "seed": 123,
        "steps": 17,
        "cfg": 6.5,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 0.35,
        "mode_type": "Chess",
        "tile_width": 640,
        "tile_height": 384,
        "mask_blur": 7,
        "tile_padding": 24,
        "seam_fix_mode": "Band Pass",
        "seam_fix_denoise": 0.75,
        "seam_fix_width": 48,
        "seam_fix_mask_blur": 5,
        "seam_fix_padding": 12,
        "force_uniform_tiles": False,
        "tiled_decode": False,
        "batch_size": 1,
    }


def _prompt(
    *,
    no_upscale: bool = False,
    input_overrides: Mapping[str, object] | None = None,
    loader_value: object = None,
) -> dict[str, object]:
    inputs = _base_inputs()
    if no_upscale:
        inputs["upscaled_image"] = ["source", 0]
    else:
        inputs.update(
            image=["source", 0],
            upscale_by=1.75,
            upscale_model=["loader", 0],
        )
    inputs.update(input_overrides or {})
    prompt: dict[str, object] = {
        "source": {"class_type": "USDUInputs", "inputs": {}},
        "usdu": {
            "class_type": ("UltimateSDUpscaleNoUpscale" if no_upscale else "UltimateSDUpscale"),
            "inputs": inputs,
        },
        "output": {"class_type": "USDUOutput", "inputs": {"image": ["usdu", 0]}},
    }
    if not no_upscale:
        prompt["loader"] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": ASSET_REF.to_wire() if loader_value is None else loader_value},
        }
    return prompt


def _node(graph: Graph, node_id: str) -> GraphNode:
    node = graph.nodes[node_id]
    assert isinstance(node, GraphNode)
    return node


def _fold(graph: Graph, node_id: str) -> RegionNode:
    node = graph.nodes[node_id]
    assert isinstance(node, RegionNode)
    return node


def _problem_codes(prompt: Mapping[str, object]) -> set[str]:
    with pytest.raises(PromptTranslationError) as captured:
        translate_prompt(prompt, SCHEMAS)
    return {problem.code for problem in captured.value.problems}


def test_carriers_match_the_upstream_prompt_surfaces() -> None:
    regular = UltimateSDUpscaleCarrier.schema()
    no_upscale = UltimateSDUpscaleNoUpscaleCarrier.schema()
    loader = UpscaleModelLoaderCarrier.schema()

    assert regular.search_visibility == no_upscale.search_visibility == "hidden"
    assert loader.search_visibility == "hidden"
    assert regular.aliases == ("UltimateSDUpscale",)
    assert no_upscale.aliases == ("UltimateSDUpscaleNoUpscale",)
    assert loader.aliases == ("UpscaleModelLoader",)
    assert [item.id for item in regular.inputs] == [
        "image",
        "model",
        "positive",
        "negative",
        "vae",
        "upscale_by",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "denoise",
        "upscale_model",
        "mode_type",
        "tile_width",
        "tile_height",
        "mask_blur",
        "tile_padding",
        "seam_fix_mode",
        "seam_fix_denoise",
        "seam_fix_width",
        "seam_fix_mask_blur",
        "seam_fix_padding",
        "force_uniform_tiles",
        "tiled_decode",
        "batch_size",
    ]
    assert [item.id for item in no_upscale.inputs] == [
        "upscaled_image" if item.id == "image" else item.id
        for item in regular.inputs
        if item.id not in {"upscale_by", "upscale_model"}
    ]
    defaults = {item.id: item.default for item in regular.inputs}
    expected_defaults = {
        "upscale_by": 2.0,
        "seed": 0,
        "steps": 20,
        "cfg": 8.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "denoise": 0.2,
        "mode_type": "Linear",
        "tile_width": 512,
        "tile_height": 512,
        "mask_blur": 8,
        "tile_padding": 32,
        "seam_fix_mode": "None",
        "seam_fix_denoise": 1.0,
        "seam_fix_width": 64,
        "seam_fix_mask_blur": 8,
        "seam_fix_padding": 16,
        "force_uniform_tiles": True,
        "tiled_decode": False,
        "batch_size": 1,
    }
    assert {input_id: defaults[input_id] for input_id in expected_defaults} == expected_defaults
    assert loader.inputs[0].id == "model_name"


def test_regular_alias_expands_to_upscale_and_two_chained_folds() -> None:
    translation = translate_prompt(_prompt(), SCHEMAS)
    graph = translation.graph

    assert translation.targets == ("output",)
    assert "loader" not in graph.nodes
    assert all(
        not (
            isinstance(node, GraphNode)
            and node.node_type.startswith("dinkster.compat.ultimate_sd_upscale")
        )
        for node in graph.nodes.values()
    )
    assert _node(graph, "usdu__usdu_input_info").node_type == "dinkster.image.info"
    for dimension in ("width", "height"):
        expression = _node(graph, f"usdu__usdu_target_{dimension}")
        assert expression.node_type == "dinkster.math.expression"
        assert expression.inputs["expression"] == "ceil(a * b / 64) * 64"
        assert expression.inputs["values.b"] == TypedLiteral("core.float", 1.75)
        assert expression.output_members == {}

    upscale = _node(graph, "usdu__usdu_model_upscale")
    assert upscale.node_type == "dinkster.image.upscale_model"
    assert upscale.inputs == {
        "image": Link("source", "image"),
        "upscale_model": ASSET_REF.to_wire(),
        "provider": "dinkster-vision-upscale",
        "tile_size": 512,
        "overlap": 32,
    }
    canvas = _node(graph, "usdu__usdu_canvas")
    assert canvas.node_type == "dinkster.image.resize"
    assert canvas.inputs["interpolation"] == "lanczos"
    assert canvas.slot_variants == {
        "target": "dimensions",
        "mode": "stretch",
        "divisibility": "none",
    }

    redraw_plan = _node(graph, "usdu__usdu_redraw_plan")
    assert redraw_plan.inputs["phase"] == "redraw"
    assert redraw_plan.inputs["mode"] == "chess"
    assert redraw_plan.inputs["tile_width"] == 640
    assert redraw_plan.inputs["tile_height"] == 384
    assert redraw_plan.inputs["mask_blur"] == 7
    assert redraw_plan.inputs["tile_padding"] == 24
    assert redraw_plan.inputs["force_uniform_tiles"] is False

    seam_plan = _node(graph, "usdu__usdu_seam_plan")
    assert seam_plan.inputs["phase"] == "seam_fix"
    assert seam_plan.inputs["seam_fix_mode"] == "band_pass"
    assert seam_plan.inputs["seam_fix_width"] == 48
    assert seam_plan.inputs["seam_fix_mask_blur"] == 5
    assert seam_plan.inputs["seam_fix_padding"] == 12

    redraw = _fold(graph, "usdu__usdu_redraw")
    seam = _fold(graph, "usdu")
    for fold in (redraw, seam):
        assert fold.kind == "fold"
        assert fold.binding == "zip"
        assert fold.element_ports == (
            "region",
            "crop",
            "sample_width",
            "sample_height",
            "mask_kind",
            "mask_blur",
        )
        assert fold.state_ports == ("canvas",)
        assert tuple(fold.outputs) == ("canvas",)
        assert all(isinstance(node, GraphNode) for node in fold.body.nodes.values())
        assert {
            node.node_type for node in fold.body.nodes.values() if isinstance(node, GraphNode)
        } == {
            "dinkster.string_to_combo",
            "dinkster.mask.tile_blend",
            "dinkster.image.crop",
            "dinkster.region.info",
            "dinkster.image.resize",
            "dinkster.vae_encode",
            "dinkster.ksampler",
            "dinkster.vae_decode",
            "dinkster.image.composite",
        }
        sampler = _node(fold.body, "sample_latent")
        assert sampler.inputs["seed"] == Link("$region", "seed")
        assert sampler.inputs["sampler_name"] == Link("$region", "sampler_name")
        assert sampler.inputs["scheduler"] == Link("$region", "scheduler")
        assert _node(fold.body, "sample").inputs["interpolation"] == "lanczos"
        assert _node(fold.body, "restore").inputs["interpolation"] == "lanczos"

    assert redraw.inputs["canvas"] == Link("usdu__usdu_canvas", "image")
    assert redraw.inputs["seed"] == seam.inputs["seed"] == 123
    assert redraw.inputs["denoise"] == seam.inputs["denoise"] == 0.35
    assert redraw.inputs["sampler_name"] == seam.inputs["sampler_name"] == "dinkster.euler"
    assert redraw.inputs["scheduler"] == seam.inputs["scheduler"] == "dinkster.normal"
    assert seam.inputs["canvas"] == Link("usdu__usdu_redraw", "canvas")
    assert _node(graph, "output").inputs["image"] == Link("usdu", "canvas")

    diagnostics = validate(graph, SCHEMAS, translation.targets)
    assert [item for item in diagnostics if item.severity == "error"] == []


def test_imported_seam_fix_denoise_normalizes_to_ordinary_denoise() -> None:
    first = translate_prompt(
        _prompt(input_overrides={"denoise": 0.4, "seam_fix_denoise": 0.1}),
        SCHEMAS,
    ).graph
    second = translate_prompt(
        _prompt(input_overrides={"denoise": 0.4, "seam_fix_denoise": 0.9}),
        SCHEMAS,
    ).graph

    assert first == second
    assert _fold(first, "usdu__usdu_redraw").inputs["denoise"] == 0.4
    assert _fold(first, "usdu").inputs["denoise"] == 0.4


def test_loader_filename_resolves_at_the_compat_prompt_boundary() -> None:
    adapter = make_model_asset_inputs_adapter(
        {
            "model_name": (
                lambda name: ASSET_REF if name == "4x-UltraSharp.pth" else None,
                "upscale_models",
            )
        }
    )

    graph = translate_prompt(
        _prompt(loader_value="4x-UltraSharp.pth"),
        SCHEMAS,
        input_adapters={UpscaleModelLoaderCarrier.schema().node_type: adapter},
    ).graph

    assert _node(graph, "usdu__usdu_model_upscale").inputs["upscale_model"] == ASSET_REF.to_wire()


@pytest.mark.parametrize(
    ("mode", "seam_mode", "expected_folds"),
    [
        ("Linear", "None", {"usdu"}),
        ("None", "Half Tile", {"usdu"}),
        ("None", "None", set()),
        ("Chess", "Half Tile + Intersections", {"usdu__usdu_redraw", "usdu"}),
    ],
)
def test_mode_literals_select_only_the_required_folds(
    mode: str, seam_mode: str, expected_folds: set[str]
) -> None:
    graph = translate_prompt(
        _prompt(input_overrides={"mode_type": mode, "seam_fix_mode": seam_mode}),
        SCHEMAS,
    ).graph
    folds = {node_id for node_id, node in graph.nodes.items() if isinstance(node, RegionNode)}
    assert folds == expected_folds
    assert "loader" not in graph.nodes
    assert _node(graph, "usdu__usdu_model_upscale").node_type == "dinkster.image.upscale_model"
    if not expected_folds:
        assert _node(graph, "usdu").node_type == "dinkster.image.resize"
        assert _node(graph, "output").inputs["image"] == Link("usdu", "image")


def test_no_upscale_alias_uses_input_canvas_and_can_be_a_passthrough() -> None:
    redraw = translate_prompt(
        _prompt(no_upscale=True, input_overrides={"seam_fix_mode": "None"}),
        SCHEMAS,
    ).graph
    assert "usdu__usdu_model_upscale" not in redraw.nodes
    assert "usdu__usdu_target_width" not in redraw.nodes
    assert _node(redraw, "usdu__usdu_canvas_info").inputs["image"] == Link("source", "image")
    assert _fold(redraw, "usdu").inputs["canvas"] == Link("source", "image")

    passthrough = translate_prompt(
        _prompt(
            no_upscale=True,
            input_overrides={"mode_type": "None", "seam_fix_mode": "None"},
        ),
        SCHEMAS,
    ).graph
    assert "usdu" not in passthrough.nodes
    assert "usdu__usdu_canvas_info" not in passthrough.nodes
    assert _node(passthrough, "output").inputs["image"] == Link("source", "image")


def test_upscale_model_loader_and_image_upscale_pair_lower_together() -> None:
    prompt = {
        "source": {"class_type": "USDUInputs", "inputs": {}},
        "loader": {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": ASSET_REF.to_wire()},
        },
        "upscale": {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": ["loader", 0], "image": ["source", 0]},
        },
        "output": {"class_type": "USDUOutput", "inputs": {"image": ["upscale", 0]}},
    }
    graph = translate_prompt(prompt, SCHEMAS).graph
    assert "loader" not in graph.nodes
    upscale = _node(graph, "upscale")
    assert upscale.node_type == "dinkster.image.upscale_model"
    assert upscale.inputs["upscale_model"] == ASSET_REF.to_wire()
    assert upscale.inputs["provider"] == "dinkster-vision-upscale"
    assert _node(graph, "output").inputs["image"] == Link("upscale", "image")


def test_image_upscale_pair_can_feed_no_upscale_refinement() -> None:
    refine_inputs = _base_inputs()
    refine_inputs["upscaled_image"] = ["upscale", 0]
    refine_inputs["seam_fix_mode"] = "None"
    prompt = {
        "source": {"class_type": "USDUInputs", "inputs": {}},
        "loader": {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": ASSET_REF.to_wire()},
        },
        "upscale": {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": ["loader", 0], "image": ["source", 0]},
        },
        "refine": {
            "class_type": "UltimateSDUpscaleNoUpscale",
            "inputs": refine_inputs,
        },
        "output": {"class_type": "USDUOutput", "inputs": {"image": ["refine", 0]}},
    }

    graph = translate_prompt(prompt, SCHEMAS).graph

    assert _node(graph, "upscale").node_type == "dinkster.image.upscale_model"
    assert _node(graph, "refine__usdu_canvas_info").inputs["image"] == Link("upscale", "image")
    assert _fold(graph, "refine").inputs["canvas"] == Link("upscale", "image")
    assert _node(graph, "output").inputs["image"] == Link("refine", "canvas")


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"tiled_decode": True}, "prompt.usdu.tiled_decode"),
        ({"tiled_decode": ["source", 0]}, "prompt.usdu.tiled_decode"),
        ({"batch_size": 2}, "prompt.usdu.batch_size"),
        ({"batch_size": 1.0}, "prompt.usdu.batch_size"),
        ({"sampler_name": "unknown"}, "prompt.usdu.unknown_choice"),
        ({"sampler_name": ["source", 0]}, "prompt.usdu.nonliteral_choice"),
        ({"scheduler": "unknown"}, "prompt.usdu.unknown_choice"),
        ({"mode_type": "Spiral"}, "prompt.usdu.mode"),
        ({"mode_type": ["source", 0]}, "prompt.usdu.mode"),
        ({"seam_fix_mode": "Corners"}, "prompt.usdu.seam_mode"),
        ({"seam_fix_mode": ["source", 0]}, "prompt.usdu.seam_mode"),
        ({"upscale_model": ["source", 0]}, "prompt.usdu.upscale_model_source"),
    ],
)
def test_unsupported_controls_fail_closed(overrides: Mapping[str, object], code: str) -> None:
    assert code in _problem_codes(_prompt(input_overrides=overrides))


@pytest.mark.parametrize("loader_value", ["4x-UltraSharp.pth", ["source", 0], {}])
def test_nonliteral_or_invalid_upscale_model_names_fail_closed(loader_value: object) -> None:
    assert "prompt.usdu.upscale_model_literal" in _problem_codes(_prompt(loader_value=loader_value))


def test_unused_loader_and_generated_id_collision_fail_closed() -> None:
    unused = _prompt(no_upscale=True)
    unused["loader"] = {
        "class_type": "UpscaleModelLoader",
        "inputs": {"model_name": ASSET_REF.to_wire()},
    }
    assert "prompt.usdu.unused_upscale_model_loader" in _problem_codes(unused)

    collision = _prompt()
    collision["usdu__usdu_canvas"] = {"class_type": "USDUInputs", "inputs": {}}
    assert "prompt.usdu.generated_id_collision" in _problem_codes(collision)


def test_unrecognized_carrier_inputs_fail_closed() -> None:
    unexpected_usdu = _prompt(input_overrides={"custom_sampler": ["source", 0]})
    assert "prompt.usdu.unexpected_input" in _problem_codes(unexpected_usdu)

    unexpected_loader = _prompt()
    loader = unexpected_loader["loader"]
    assert isinstance(loader, dict)
    inputs = loader["inputs"]
    assert isinstance(inputs, dict)
    inputs["category"] = "upscale_models"
    assert "prompt.usdu.unexpected_input" in _problem_codes(unexpected_loader)


def test_lowering_is_deterministic() -> None:
    first = translate_prompt(_prompt(), SCHEMAS)
    second = translate_prompt(_prompt(), SCHEMAS)
    assert first.targets == second.targets
    assert graph_to_wire(first.graph) == graph_to_wire(second.graph)


async def _wait_for_state(job: Job, state: str) -> None:
    async with asyncio.timeout(5.0):
        while job.state != state:
            await asyncio.sleep(0.005)


def test_carrier_reaching_the_native_queue_fails_execution() -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_asset_type(registry)
        nodes = (UpscaleModelLoaderCarrier,)
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
        )
        queue = JobQueue(engine)
        queue.start()
        job = queue.submit(
            "compat",
            "carrier",
            Graph(
                nodes={
                    "loader": GraphNode(
                        UpscaleModelLoaderCarrier.schema().node_type,
                        {"model_name": ASSET_REF.to_wire()},
                    )
                }
            ),
            ["loader"],
        )
        await _wait_for_state(job, "failed")
        assert job.error is not None
        assert job.error["kind"] == "execution"
        assert job.error["nodeId"] == "loader"
        assert "must be lowered before execution" in job.error["message"]
        await queue.close()

    asyncio.run(scenario())
