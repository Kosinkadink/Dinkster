"""Lower Ultimate SD Upscale carriers into native tiled-refine graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

from dinkster_assets import ASSET_TYPE, AssetError, AssetRef
from dinkster_graph import Graph, GraphNode, Link, RegionNode, RegionOutput, TypedLiteral
from dinkster_graph.model import PORTS_NODE_ID
from dinkster_schema import (
    AssetWidget,
    ComboOption,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)
from dinkster_values import CORE_BOOLEAN, CORE_COMBO, CORE_FLOAT, CORE_INT, CORE_STRING

ULTIMATE_SD_UPSCALE = "dinkster.compat.ultimate_sd_upscale"
ULTIMATE_SD_UPSCALE_NO_UPSCALE = "dinkster.compat.ultimate_sd_upscale_no_upscale"
UPSCALE_MODEL_LOADER = "dinkster.compat.upscale_model_loader"

ASSET = TypeExpr.concrete(ASSET_TYPE)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
COMBO = TypeExpr.concrete(CORE_COMBO)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
INT = TypeExpr.concrete(CORE_INT)
STRING = TypeExpr.concrete(CORE_STRING)
IMAGE = TypeExpr.concrete("dinkster.image")
MASK = TypeExpr.concrete("dinkster.mask")
REGION = TypeExpr.concrete("dinkster.region")
MODEL = TypeExpr.concrete("dinkster.model")
CONDITIONING = TypeExpr.concrete("dinkster.conditioning")
VAE = TypeExpr.concrete("dinkster.vae")

_REDRAW_MODES = ("Linear", "Chess", "None")
_SEAM_MODES = ("None", "Band Pass", "Half Tile", "Half Tile + Intersections")
_REDRAW_VALUES = {"Linear": "linear", "Chess": "chess", "None": "none"}
_SEAM_VALUES = {
    "None": "none",
    "Band Pass": "band_pass",
    "Half Tile": "half_tile",
    "Half Tile + Intersections": "half_tile_intersections",
}
_DEFAULTS: Mapping[str, object] = {
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
_ELEMENT_PORTS = (
    "region",
    "crop",
    "sample_width",
    "sample_height",
    "mask_kind",
    "mask_blur",
)
_UPSCALE_PROVIDER = "dinkster-vision-upscale"


def _combo(input_id: str, options: Sequence[str], default: str) -> InputSpec:
    return InputSpec(
        input_id,
        COMBO,
        default=default,
        widget=ComboWidget(options=tuple(ComboOption(value=value) for value in options)),
    )


def _number(
    input_id: str,
    input_type: TypeExpr,
    default: int | float,
    *,
    minimum: int | float,
    maximum: int | float,
    step: int | float,
) -> InputSpec:
    return InputSpec(
        input_id,
        input_type,
        default=default,
        widget=NumberWidget(min=minimum, max=maximum, step=step),
    )


def _refine_inputs(*, include_upscale: bool) -> tuple[InputSpec, ...]:
    image_id = "image" if include_upscale else "upscaled_image"
    inputs: list[InputSpec] = [
        InputSpec(image_id, IMAGE),
        InputSpec("model", MODEL),
        InputSpec("positive", CONDITIONING),
        InputSpec("negative", CONDITIONING),
        InputSpec("vae", VAE),
    ]
    if include_upscale:
        inputs.append(_number("upscale_by", FLOAT, 2.0, minimum=0.05, maximum=4.0, step=0.05))
    inputs.extend(
        (
            _number("seed", INT, 0, minimum=0, maximum=2**53 - 1, step=1),
            _number("steps", INT, 20, minimum=1, maximum=10_000, step=1),
            _number("cfg", FLOAT, 8.0, minimum=0.0, maximum=100.0, step=0.1),
            _combo("sampler_name", ("euler",), "euler"),
            _combo("scheduler", ("simple",), "simple"),
            _number("denoise", FLOAT, 0.2, minimum=0.0, maximum=1.0, step=0.01),
        )
    )
    if include_upscale:
        inputs.append(InputSpec("upscale_model", ASSET))
    inputs.extend(
        (
            _combo("mode_type", _REDRAW_MODES, "Linear"),
            _number("tile_width", INT, 512, minimum=64, maximum=8192, step=8),
            _number("tile_height", INT, 512, minimum=64, maximum=8192, step=8),
            _number("mask_blur", INT, 8, minimum=0, maximum=64, step=1),
            _number("tile_padding", INT, 32, minimum=0, maximum=8192, step=8),
            _combo("seam_fix_mode", _SEAM_MODES, "None"),
            _number("seam_fix_denoise", FLOAT, 1.0, minimum=0.0, maximum=1.0, step=0.01),
            _number("seam_fix_width", INT, 64, minimum=0, maximum=8192, step=8),
            _number("seam_fix_mask_blur", INT, 8, minimum=0, maximum=64, step=1),
            _number("seam_fix_padding", INT, 16, minimum=0, maximum=8192, step=8),
            InputSpec("force_uniform_tiles", BOOLEAN, default=True),
            InputSpec("tiled_decode", BOOLEAN, default=False),
            _number("batch_size", INT, 1, minimum=1, maximum=4096, step=1),
        )
    )
    return tuple(inputs)


class _Carrier(Node):
    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError(f"{cls.schema().node_type} must be lowered before execution")


class UltimateSDUpscaleCarrier(_Carrier):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type=ULTIMATE_SD_UPSCALE,
            display_name="Ultimate SD Upscale Compatibility Carrier",
            category="compatibility/comfyui",
            inputs=_refine_inputs(include_upscale=True),
            outputs=(OutputSpec("image", IMAGE),),
            aliases=("UltimateSDUpscale",),
            search_visibility="hidden",
        )


class UltimateSDUpscaleNoUpscaleCarrier(_Carrier):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type=ULTIMATE_SD_UPSCALE_NO_UPSCALE,
            display_name="Ultimate SD Upscale No-Upscale Compatibility Carrier",
            category="compatibility/comfyui",
            inputs=_refine_inputs(include_upscale=False),
            outputs=(OutputSpec("image", IMAGE),),
            aliases=("UltimateSDUpscaleNoUpscale",),
            search_visibility="hidden",
        )


class UpscaleModelLoaderCarrier(_Carrier):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type=UPSCALE_MODEL_LOADER,
            display_name="Upscale Model Loader Compatibility Carrier",
            category="compatibility/comfyui",
            inputs=(
                InputSpec(
                    "model_name",
                    ASSET,
                    widget=AssetWidget(
                        accept=("application/octet-stream",),
                        kind="model/upscaler",
                    ),
                ),
            ),
            outputs=(OutputSpec("upscale_model", ASSET),),
            aliases=("UpscaleModelLoader",),
            search_visibility="hidden",
        )


USDU_CARRIER_NODES: tuple[type[Node], ...] = (
    UltimateSDUpscaleCarrier,
    UltimateSDUpscaleNoUpscaleCarrier,
    UpscaleModelLoaderCarrier,
)


@dataclass(frozen=True)
class UsduProblem:
    code: str
    message: str
    node_id: str
    input_id: str = ""


@dataclass(frozen=True)
class UsduResult:
    graph: Graph
    problems: tuple[UsduProblem, ...] = ()


@dataclass(frozen=True)
class _Expansion:
    node_id: str
    regular: bool
    values: Mapping[str, object]
    redraw_mode: str
    seam_mode: str
    sampler_name: str
    scheduler: str
    upscale_asset: Mapping[str, object] | None


def _problem(code: str, message: str, node_id: str, input_id: str = "") -> UsduProblem:
    return UsduProblem(f"prompt.usdu.{code}", message, node_id, input_id)


def _input_values(
    node_id: str,
    node: GraphNode,
    *,
    regular: bool,
) -> tuple[dict[str, object], list[UsduProblem]]:
    required = {
        "image" if regular else "upscaled_image",
        "model",
        "positive",
        "negative",
        "vae",
    }
    if regular:
        required.add("upscale_model")
    allowed = required | set(_DEFAULTS)
    if not regular:
        allowed.remove("upscale_by")
    values = dict(_DEFAULTS)
    values.update(node.inputs)
    problems = [
        _problem("missing_input", f"required input {input_id!r} is absent", node_id, input_id)
        for input_id in sorted(required)
        if input_id not in node.inputs
    ]
    problems.extend(
        _problem("unexpected_input", f"input {input_id!r} is not supported", node_id, input_id)
        for input_id in sorted(set(node.inputs) - allowed)
    )
    for input_id in sorted(required - {"upscale_model"}):
        if input_id in node.inputs and not isinstance(node.inputs[input_id], Link):
            problems.append(
                _problem(
                    "nonliteral_data",
                    f"{input_id} must be linked from a typed node output",
                    node_id,
                    input_id,
                )
            )
    return values, problems


def _native_choice(
    value: object,
    *,
    input_id: str,
    node_id: str,
    schemas: Mapping[str, NodeSchema],
) -> tuple[str | None, UsduProblem | None]:
    if type(value) is not str:
        return None, _problem(
            "nonliteral_choice",
            f"{input_id} must be a literal string",
            node_id,
            input_id,
        )
    schema = schemas.get("dinkster.ksampler")
    spec = None if schema is None else schema.input(input_id)
    widget = None if spec is None else spec.widget
    options = () if not isinstance(widget, ComboWidget) else widget.options
    native_value = f"dinkster.{value}"
    native_options = {
        option.value if isinstance(option, ComboOption) else option for option in options
    }
    if native_value not in native_options:
        return None, _problem(
            "unknown_choice",
            f"{input_id} value {value!r} has no native KSampler counterpart",
            node_id,
            input_id,
        )
    return native_value, None


def _asset_literal(
    loaders: Mapping[str, GraphNode],
    link: object,
    *,
    node_id: str,
    input_id: str,
) -> tuple[Mapping[str, object] | None, str | None, UsduProblem | None]:
    if (
        not isinstance(link, Link)
        or link.node_id not in loaders
        or link.output_id != "upscale_model"
    ):
        return (
            None,
            None,
            _problem(
                "upscale_model_source",
                "upscale_model must be fed by an UpscaleModelLoader with a literal model name",
                node_id,
                input_id,
            ),
        )
    loader = loaders[link.node_id]
    value = loader.inputs.get("model_name")
    if isinstance(value, Link) or not isinstance(value, Mapping):
        return (
            None,
            link.node_id,
            _problem(
                "upscale_model_literal",
                "UpscaleModelLoader model_name must resolve to a literal asset",
                link.node_id,
                "model_name",
            ),
        )
    try:
        asset = AssetRef.from_wire(cast("Mapping[str, object]", value)).to_wire()
    except AssetError as exc:
        return (
            None,
            link.node_id,
            _problem(
                "upscale_model_literal",
                f"UpscaleModelLoader model_name is not a valid asset: {exc}",
                link.node_id,
                "model_name",
            ),
        )
    return asset, link.node_id, None


def _is_image_upscale(node: GraphNode, schemas: Mapping[str, NodeSchema]) -> bool:
    schema = schemas.get(node.node_type)
    return node.node_type == "comfy.ImageUpscaleWithModel" or (
        schema is not None and "ImageUpscaleWithModel" in schema.aliases
    )


def _outer_links(node: GraphNode | RegionNode) -> tuple[Link, ...]:
    return tuple(value for value in node.inputs.values() if isinstance(value, Link))


def _replace_link(node: GraphNode | RegionNode, old: Link, new: Link) -> GraphNode | RegionNode:
    inputs = {key: new if value == old else value for key, value in node.inputs.items()}
    return replace(node, inputs=inputs)


def _rewrite_link(nodes: dict[str, GraphNode | RegionNode], old: Link, new: Link) -> None:
    for node_id, node in tuple(nodes.items()):
        nodes[node_id] = _replace_link(node, old, new)


def _port(port_id: str) -> Link:
    return Link(PORTS_NODE_ID, port_id)


def _float_expression_value(value: object) -> Link | TypedLiteral:
    if isinstance(value, Link):
        return value
    return TypedLiteral(CORE_FLOAT, value)


def _refine_fold(
    plan_id: str,
    canvas: Link,
    *,
    width: Link,
    height: Link,
    values: Mapping[str, object],
    sampler_name: str,
    scheduler: str,
    denoise: object,
) -> RegionNode:
    body = Graph(
        nodes={
            "kind": GraphNode("dinkster.string_to_combo", {"string": _port("mask_kind")}),
            "mask": GraphNode(
                "dinkster.mask.tile_blend",
                {
                    "width": _port("width"),
                    "height": _port("height"),
                    "region": _port("region"),
                    "kind": Link("kind", "choice"),
                    "blur": _port("mask_blur"),
                },
            ),
            "crop": GraphNode(
                "dinkster.image.crop",
                {
                    "image": _port("canvas"),
                    "source.region": _port("crop"),
                    "source.mask": Link("mask", "mask"),
                    "padding": 0,
                    "rounding": "expand",
                },
                slot_variants={"source": "region", "outside": "clip"},
            ),
            "crop_info": GraphNode(
                "dinkster.region.info",
                {"region": _port("crop"), "integer_rounding": "expand"},
            ),
            "sample": GraphNode(
                "dinkster.image.resize",
                {
                    "image": Link("crop", "image"),
                    "target.width": _port("sample_width"),
                    "target.height": _port("sample_height"),
                    "interpolation": "lanczos",
                },
                slot_variants={
                    "target": "dimensions",
                    "mode": "stretch",
                    "divisibility": "none",
                },
            ),
            "encode": GraphNode(
                "dinkster.vae_encode",
                {"pixels": Link("sample", "image"), "vae": _port("vae")},
            ),
            "sample_latent": GraphNode(
                "dinkster.ksampler",
                {
                    "model": _port("model"),
                    "seed": _port("seed"),
                    "steps": _port("steps"),
                    "cfg": _port("cfg"),
                    "sampler_name": _port("sampler_name"),
                    "scheduler": _port("scheduler"),
                    "positive": _port("positive"),
                    "negative": _port("negative"),
                    "latent_image": Link("encode", "latent"),
                    "denoise": _port("denoise"),
                },
            ),
            "decode": GraphNode(
                "dinkster.vae_decode",
                {"samples": Link("sample_latent", "latent"), "vae": _port("vae")},
            ),
            "restore": GraphNode(
                "dinkster.image.resize",
                {
                    "image": Link("decode", "image"),
                    "target.width": Link("crop_info", "integer_width"),
                    "target.height": Link("crop_info", "integer_height"),
                    "interpolation": "lanczos",
                },
                slot_variants={
                    "target": "dimensions",
                    "mode": "stretch",
                    "divisibility": "none",
                },
            ),
            "composite": GraphNode(
                "dinkster.image.composite",
                {
                    "destination": _port("canvas"),
                    "source": Link("restore", "image"),
                    "x": Link("crop_info", "integer_x"),
                    "y": Link("crop_info", "integer_y"),
                    "blend_mode": "normal",
                    "factor": 1.0,
                    "mask": Link("crop", "mask"),
                },
                slot_variants={"source_resize": "none", "mask": "mask"},
            ),
        }
    )
    ports = {
        "region": REGION,
        "crop": REGION,
        "sample_width": INT,
        "sample_height": INT,
        "mask_kind": STRING,
        "mask_blur": INT,
        "canvas": IMAGE,
        "width": INT,
        "height": INT,
        "model": MODEL,
        "positive": CONDITIONING,
        "negative": CONDITIONING,
        "vae": VAE,
        "seed": INT,
        "steps": INT,
        "cfg": FLOAT,
        "sampler_name": COMBO,
        "scheduler": COMBO,
        "denoise": FLOAT,
    }
    return RegionNode(
        kind="fold",
        body=body,
        ports=ports,
        inputs={
            "region": Link(plan_id, "regions"),
            "crop": Link(plan_id, "crops"),
            "sample_width": Link(plan_id, "sample_widths"),
            "sample_height": Link(plan_id, "sample_heights"),
            "mask_kind": Link(plan_id, "mask_kinds"),
            "mask_blur": Link(plan_id, "mask_blurs"),
            "canvas": canvas,
            "width": width,
            "height": height,
            "model": values["model"],
            "positive": values["positive"],
            "negative": values["negative"],
            "vae": values["vae"],
            "seed": values["seed"],
            "steps": values["steps"],
            "cfg": values["cfg"],
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
        },
        element_ports=_ELEMENT_PORTS,
        state_ports=("canvas",),
        outputs={"canvas": RegionOutput(Link("composite", "image"), mode="state")},
    )


def _plan_inputs(
    width: Link,
    height: Link,
    values: Mapping[str, object],
    *,
    phase: str,
    mode: str,
) -> Mapping[str, object]:
    inputs: dict[str, object] = {
        "width": width,
        "height": height,
        "tile_width": values["tile_width"],
        "tile_height": values["tile_height"],
        "phase": phase,
        "force_uniform_tiles": values["force_uniform_tiles"],
    }
    if phase == "redraw":
        inputs.update(
            mode=mode,
            mask_blur=values["mask_blur"],
            tile_padding=values["tile_padding"],
        )
    else:
        inputs.update(
            seam_fix_mode=mode,
            seam_fix_width=values["seam_fix_width"],
            seam_fix_mask_blur=values["seam_fix_mask_blur"],
            seam_fix_padding=values["seam_fix_padding"],
        )
    return inputs


def _generated_id(node_id: str, suffix: str) -> str:
    return f"{node_id}__usdu_{suffix}"


def _needed_ids(expansion: _Expansion) -> tuple[str, ...]:
    ids: list[str] = []
    if expansion.regular:
        ids.extend(
            (
                _generated_id(expansion.node_id, "input_info"),
                _generated_id(expansion.node_id, "target_width"),
                _generated_id(expansion.node_id, "target_height"),
                _generated_id(expansion.node_id, "model_upscale"),
            )
        )
        if expansion.redraw_mode != "none" or expansion.seam_mode != "none":
            ids.append(_generated_id(expansion.node_id, "canvas"))
    elif expansion.redraw_mode != "none" or expansion.seam_mode != "none":
        ids.append(_generated_id(expansion.node_id, "canvas_info"))
    if expansion.redraw_mode != "none":
        ids.append(_generated_id(expansion.node_id, "redraw_plan"))
        if expansion.seam_mode != "none":
            ids.append(_generated_id(expansion.node_id, "redraw"))
    if expansion.seam_mode != "none":
        ids.append(_generated_id(expansion.node_id, "seam_plan"))
    return tuple(ids)


def _expand(
    nodes: dict[str, GraphNode | RegionNode],
    expansion: _Expansion,
) -> None:
    node_id = expansion.node_id
    current = nodes[node_id]
    assert isinstance(current, GraphNode)
    values = dict(expansion.values)
    values.update(current.inputs)
    redraw = expansion.redraw_mode != "none"
    seam = expansion.seam_mode != "none"

    if not expansion.regular and not redraw and not seam:
        del nodes[node_id]
        _rewrite_link(
            nodes,
            Link(node_id, "image"),
            cast("Link", values["upscaled_image"]),
        )
        return

    if expansion.regular:
        input_info_id = _generated_id(node_id, "input_info")
        target_width_id = _generated_id(node_id, "target_width")
        target_height_id = _generated_id(node_id, "target_height")
        model_upscale_id = _generated_id(node_id, "model_upscale")
        nodes[input_info_id] = GraphNode("dinkster.image.info", {"image": values["image"]})
        nodes[target_width_id] = GraphNode(
            "dinkster.math.expression",
            {
                "expression": "ceil(a * b / 64) * 64",
                "values.a": Link(input_info_id, "width"),
                "values.b": _float_expression_value(values["upscale_by"]),
            },
        )
        nodes[target_height_id] = GraphNode(
            "dinkster.math.expression",
            {
                "expression": "ceil(a * b / 64) * 64",
                "values.a": Link(input_info_id, "height"),
                "values.b": _float_expression_value(values["upscale_by"]),
            },
        )
        assert expansion.upscale_asset is not None
        nodes[model_upscale_id] = GraphNode(
            "dinkster.image.upscale_model",
            {
                "image": values["image"],
                "upscale_model": expansion.upscale_asset,
                "provider": _UPSCALE_PROVIDER,
                "tile_size": 512,
                "overlap": 32,
            },
        )
        width = Link(target_width_id, "int")
        height = Link(target_height_id, "int")
        resize_id = _generated_id(node_id, "canvas") if redraw or seam else node_id
        nodes[resize_id] = GraphNode(
            "dinkster.image.resize",
            {
                "image": Link(model_upscale_id, "image"),
                "target.width": width,
                "target.height": height,
                "interpolation": "lanczos",
            },
            slot_variants={
                "target": "dimensions",
                "mode": "stretch",
                "divisibility": "none",
            },
        )
        canvas = Link(resize_id, "image")
    else:
        info_id = _generated_id(node_id, "canvas_info")
        canvas = cast("Link", values["upscaled_image"])
        nodes[info_id] = GraphNode("dinkster.image.info", {"image": canvas})
        width = Link(info_id, "width")
        height = Link(info_id, "height")

    if redraw:
        plan_id = _generated_id(node_id, "redraw_plan")
        nodes[plan_id] = GraphNode(
            "dinkster.image.tile_refine_plan",
            _plan_inputs(width, height, values, phase="redraw", mode=expansion.redraw_mode),
        )
        fold_id = _generated_id(node_id, "redraw") if seam else node_id
        nodes[fold_id] = _refine_fold(
            plan_id,
            canvas,
            width=width,
            height=height,
            values=values,
            sampler_name=expansion.sampler_name,
            scheduler=expansion.scheduler,
            denoise=values["denoise"],
        )
        canvas = Link(fold_id, "canvas")

    if seam:
        plan_id = _generated_id(node_id, "seam_plan")
        nodes[plan_id] = GraphNode(
            "dinkster.image.tile_refine_plan",
            _plan_inputs(width, height, values, phase="seam_fix", mode=expansion.seam_mode),
        )
        # Upstream serializes seam_fix_denoise but samples both passes with denoise.
        # Normalize imported workflows here instead of carrying a dead native control.
        nodes[node_id] = _refine_fold(
            plan_id,
            canvas,
            width=width,
            height=height,
            values=values,
            sampler_name=expansion.sampler_name,
            scheduler=expansion.scheduler,
            denoise=values["denoise"],
        )

    if redraw or seam:
        _rewrite_link(nodes, Link(node_id, "image"), Link(node_id, "canvas"))


def lower_ultimate_sd_upscale(
    graph: Graph,
    schemas: Mapping[str, NodeSchema],
) -> UsduResult:
    """Expand fully literal Ultimate SD Upscale carriers or refuse them."""
    graph_nodes = dict(graph.nodes)
    loaders = {
        node_id: node
        for node_id, node in graph_nodes.items()
        if isinstance(node, GraphNode) and node.node_type == UPSCALE_MODEL_LOADER
    }
    expansions: list[_Expansion] = []
    consumed_loader_inputs: set[tuple[str, str]] = set()
    problems: list[UsduProblem] = []

    for loader_id, loader in loaders.items():
        problems.extend(
            _problem(
                "unexpected_input",
                f"input {input_id!r} is not supported",
                loader_id,
                input_id,
            )
            for input_id in sorted(set(loader.inputs) - {"model_name"})
        )

    for node_id, candidate in graph_nodes.items():
        if not isinstance(candidate, GraphNode) or candidate.node_type not in {
            ULTIMATE_SD_UPSCALE,
            ULTIMATE_SD_UPSCALE_NO_UPSCALE,
        }:
            continue
        regular = candidate.node_type == ULTIMATE_SD_UPSCALE
        values, input_problems = _input_values(node_id, candidate, regular=regular)
        problems.extend(input_problems)

        tiled_decode = values["tiled_decode"]
        if type(tiled_decode) is not bool or tiled_decode:
            problems.append(
                _problem(
                    "tiled_decode",
                    "tiled_decode must be the literal value false",
                    node_id,
                    "tiled_decode",
                )
            )
        batch_size = values["batch_size"]
        if type(batch_size) is not int or batch_size != 1:
            problems.append(
                _problem(
                    "batch_size",
                    "batch_size must be the literal integer 1",
                    node_id,
                    "batch_size",
                )
            )
        mode = values["mode_type"]
        seam_mode = values["seam_fix_mode"]
        if type(mode) is not str or mode not in _REDRAW_VALUES:
            problems.append(
                _problem(
                    "mode",
                    f"mode_type must be one of {list(_REDRAW_VALUES)}",
                    node_id,
                    "mode_type",
                )
            )
        if type(seam_mode) is not str or seam_mode not in _SEAM_VALUES:
            problems.append(
                _problem(
                    "seam_mode",
                    f"seam_fix_mode must be one of {list(_SEAM_VALUES)}",
                    node_id,
                    "seam_fix_mode",
                )
            )

        sampler, sampler_problem = _native_choice(
            values["sampler_name"],
            input_id="sampler_name",
            node_id=node_id,
            schemas=schemas,
        )
        scheduler, scheduler_problem = _native_choice(
            values["scheduler"],
            input_id="scheduler",
            node_id=node_id,
            schemas=schemas,
        )
        if sampler_problem is not None:
            problems.append(sampler_problem)
        if scheduler_problem is not None:
            problems.append(scheduler_problem)

        upscale_asset: Mapping[str, object] | None = None
        if regular and "upscale_model" in values:
            upscale_asset, loader_id, asset_problem = _asset_literal(
                loaders,
                values["upscale_model"],
                node_id=node_id,
                input_id="upscale_model",
            )
            if loader_id is not None:
                consumed_loader_inputs.add((loader_id, node_id))
            if asset_problem is not None:
                problems.append(asset_problem)

        if (
            not input_problems
            and type(mode) is str
            and mode in _REDRAW_VALUES
            and type(seam_mode) is str
            and seam_mode in _SEAM_VALUES
            and sampler is not None
            and scheduler is not None
            and (not regular or upscale_asset is not None)
        ):
            expansions.append(
                _Expansion(
                    node_id,
                    regular,
                    values,
                    _REDRAW_VALUES[mode],
                    _SEAM_VALUES[seam_mode],
                    sampler,
                    scheduler,
                    upscale_asset,
                )
            )

    image_upscales: list[tuple[str, GraphNode, Mapping[str, object], str]] = []
    for node_id, candidate in graph_nodes.items():
        if not isinstance(candidate, GraphNode) or not _is_image_upscale(candidate, schemas):
            continue
        asset, loader_id, asset_problem = _asset_literal(
            loaders,
            candidate.inputs.get("upscale_model"),
            node_id=node_id,
            input_id="upscale_model",
        )
        if loader_id is None:
            continue
        consumed_loader_inputs.add((loader_id, node_id))
        if asset_problem is not None:
            problems.append(asset_problem)
            continue
        image = candidate.inputs.get("image")
        if image is None:
            problems.append(
                _problem("missing_input", "required input 'image' is absent", node_id, "image")
            )
            continue
        assert asset is not None
        schema = schemas.get(candidate.node_type)
        output_id = "image" if schema is None else schema.outputs[0].id
        image_upscales.append((node_id, candidate, asset, output_id))

    for loader_id in loaders:
        recognized_consumers = {
            consumer_id
            for current_loader, consumer_id in consumed_loader_inputs
            if current_loader == loader_id
        }
        actual_consumers = {
            consumer_id
            for consumer_id, node in graph_nodes.items()
            if any(link.node_id == loader_id for link in _outer_links(node))
        }
        if actual_consumers - recognized_consumers:
            problems.append(
                _problem(
                    "upscale_model_consumer",
                    "UpscaleModelLoader feeds a node that the compatibility pass cannot expand",
                    loader_id,
                    "model_name",
                )
            )
        if not actual_consumers:
            problems.append(
                _problem(
                    "unused_upscale_model_loader",
                    "UpscaleModelLoader is not consumed by a supported upscale node",
                    loader_id,
                    "model_name",
                )
            )

    occupied = set(graph_nodes)
    generated: set[str] = set()
    for expansion in expansions:
        for generated_id in _needed_ids(expansion):
            if generated_id in occupied or generated_id in generated:
                problems.append(
                    _problem(
                        "generated_id_collision",
                        f"generated node id {generated_id!r} already exists",
                        expansion.node_id,
                    )
                )
            generated.add(generated_id)

    if problems:
        return UsduResult(graph, tuple(problems))

    nodes: dict[str, GraphNode | RegionNode] = dict(graph_nodes)
    for node_id, _candidate, asset, output_id in image_upscales:
        image = cast(GraphNode, nodes[node_id]).inputs["image"]
        nodes[node_id] = GraphNode(
            "dinkster.image.upscale_model",
            {
                "image": image,
                "upscale_model": asset,
                "provider": _UPSCALE_PROVIDER,
                "tile_size": 512,
                "overlap": 32,
            },
        )
        if output_id != "image":
            _rewrite_link(nodes, Link(node_id, output_id), Link(node_id, "image"))

    for expansion in expansions:
        _expand(nodes, expansion)
    for loader_id in loaders:
        del nodes[loader_id]

    surviving = [
        node_id
        for node_id, node in nodes.items()
        if isinstance(node, GraphNode)
        and node.node_type
        in {
            ULTIMATE_SD_UPSCALE,
            ULTIMATE_SD_UPSCALE_NO_UPSCALE,
            UPSCALE_MODEL_LOADER,
        }
    ]
    if surviving:
        return UsduResult(
            graph,
            tuple(
                _problem(
                    "carrier_survived",
                    "compatibility carrier could not be fully expanded",
                    node_id,
                )
                for node_id in surviving
            ),
        )
    return UsduResult(Graph(nodes))


__all__ = [
    "ULTIMATE_SD_UPSCALE",
    "ULTIMATE_SD_UPSCALE_NO_UPSCALE",
    "UPSCALE_MODEL_LOADER",
    "USDU_CARRIER_NODES",
    "UltimateSDUpscaleCarrier",
    "UltimateSDUpscaleNoUpscaleCarrier",
    "UpscaleModelLoaderCarrier",
    "UsduProblem",
    "UsduResult",
    "lower_ultimate_sd_upscale",
]
