"""M3 compat pack (DESIGN 3.10): v1 nodes become honest Dinkster nodes with
id-keyed outputs and envelope types; the engine never learns v1 existed.
Translation is pure - these tests need no ComfyUI, torch, or GPU."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib
import sys
from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
from typing import cast

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_compat_comfy import (
    CompatError,
    CompatTranslation,
    comfy_type_id,
    translate_mappings,
    translate_node,
    translate_prompt,
    translate_type,
)
from dinkster_engine import Engine, EngineEvent, ExecutionError
from dinkster_graph import Graph, GraphNode, Link, RegionNode, validate
from dinkster_protocol import ExportSnapshot, LazyStatusInvocation
from dinkster_schema import (
    BooleanWidget,
    ColorWidget,
    ComboWidget,
    DynamicComboSpec,
    DynamicSlotSpec,
    ElaborationError,
    InputFamilySpec,
    InputSpec,
    MultiComboWidget,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    SelectorSpec,
    StringWidget,
    TypeExpr,
    build_node_types,
    build_schemas,
    elaborate,
    schema_from_wire,
    schema_signature,
    schema_to_wire,
)
from dinkster_values import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    TypeRegistry,
    prepare_image_array_encoding,
    register_core_types,
)
from dinkster_workers import (
    ExecutionContext,
    InProcessWorker,
    current_execution_context,
)
from dinkster_workers.execution import use_execution_context

# --- fake v1 classes: the shapes real ComfyUI nodes use -----------------


class V1Blend:
    """Typical v1 node: required + optional inputs, combo, opaque types."""

    CATEGORY = "image/blend"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206 - v1 shape, deliberately untyped
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "mode": (["normal", "multiply", "screen"],),
            },
            "optional": {
                "label": ("STRING", {"default": ""}),
            },
        }

    def blend(self, image_a, image_b, factor, mode, label=""):  # noqa: ANN001, ANN201
        return ({"a": image_a, "b": image_b, "f": factor, "m": mode},)


class V1MultiOut:
    CATEGORY = "loaders"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {"required": {"name": ("STRING", {"default": "x"})}}

    def load(self, name):  # noqa: ANN001, ANN201
        return (f"model:{name}", f"clip:{name}", f"vae:{name}")


class V1Collide:
    """Two unnamed IMAGE outputs: positional v1 needs generated unique ids."""

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "dup"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {"required": {"x": ("IMAGE",)}}

    def dup(self, x):  # noqa: ANN001, ANN201
        return (x, x)


class V1Save:
    """OUTPUT_NODE with ui-dict result and hidden inputs."""

    CATEGORY = "output"
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {
            "required": {"text": ("STRING", {"default": "hi"})},
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    saved: list[tuple[object, object, object]] = []

    def save(self, text, prompt=..., unique_id=...):  # noqa: ANN001, ANN201
        V1Save.saved.append((text, prompt, unique_id))
        return {"ui": {"texts": [text]}}


class V1Changed:
    RETURN_TYPES = ("INT",)
    FUNCTION = "roll"
    rolls = 0

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {"required": {"seed": ("INT", {"default": 0})}}

    @classmethod
    def IS_CHANGED(cls, seed):  # noqa: ANN001, ANN206
        return float("nan")

    def roll(self, seed):  # noqa: ANN001, ANN201
        type(self).rolls += 1
        return (seed + 1,)


# --- translation: schema shape ------------------------------------------


def test_translate_type_rules() -> None:
    assert translate_type("INT")[0].types == (CORE_INT,)
    assert translate_type("FLOAT")[0].types == (CORE_FLOAT,)
    assert translate_type(["a", "b"])[0].types == (CORE_COMBO,)
    assert translate_type("*")[0].kind == "wildcard"
    expr, opaque_types = translate_type("MODEL")
    assert expr.types == ("comfy.MODEL",) and opaque_types == ("comfy.MODEL",)


def test_translate_type_comma_unions() -> None:
    expr, opaque_types = translate_type("INT,FLOAT,STRING,BOOLEAN")
    assert expr == TypeExpr.union(CORE_INT, CORE_FLOAT, CORE_STRING, CORE_BOOLEAN)
    assert opaque_types == ()

    expr, opaque_types = translate_type("MODEL,CLIP")
    assert expr == TypeExpr.union("comfy.MODEL", "comfy.CLIP")
    assert opaque_types == ("comfy.MODEL", "comfy.CLIP")

    assert translate_type("INT,*") == (TypeExpr.wildcard(), ())
    assert translate_type("INT,INT") == (TypeExpr.concrete(CORE_INT), ())
    assert translate_type("INT, INT ") == (TypeExpr.concrete(CORE_INT), ())
    assert translate_type("INT , FLOAT") == (
        TypeExpr.union(CORE_INT, CORE_FLOAT),
        (),
    )
    with pytest.raises(CompatError, match="empty v1 type declaration"):
        translate_type("")
    with pytest.raises(CompatError, match="empty v1 type declaration"):
        translate_type(",")

    class AnyType(str):
        def __eq__(self, other: object) -> bool:
            return True

        __hash__ = None  # type: ignore[assignment]

    assert translate_type(AnyType("INT,FLOAT")) == (
        TypeExpr.union(CORE_INT, CORE_FLOAT),
        (),
    )


def test_comma_typed_node_translation_and_wire_round_trip() -> None:
    class V1CommaTypes:
        RETURN_TYPES = ("MODEL,CLIP",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "resident": ("MODEL,CLIP", {"default": "configured"}),
                }
            }

        def run(self, resident):  # noqa: ANN001, ANN201
            return (resident,)

    translation = CompatTranslation()
    schema = translate_node("CommaTypes", V1CommaTypes, translation).schema()
    expected = TypeExpr.union("comfy.MODEL", "comfy.CLIP")
    assert schema.inputs[0].type == expected
    assert schema.inputs[0].default == "configured"
    assert schema.inputs[0].widget is None
    assert schema.outputs[0].type == expected
    assert schema.occupies == ("gpu",)
    assert translation.opaque_types == {"comfy.MODEL", "comfy.CLIP"}
    assert schema_from_wire(schema_to_wire(schema)) == schema


def test_blend_schema_translation() -> None:
    translation = CompatTranslation()
    node = translate_node("Blend", V1Blend, translation, display_name="Blend Images")
    schema = node.schema()
    assert schema.node_type == "comfy.Blend"
    assert schema.display_name == "Blend Images"
    assert schema.category == "comfy/image/blend"
    by_id = {spec.id: spec for spec in schema.inputs}
    assert set(by_id) == {"image_a", "image_b", "factor", "mode", "label"}
    assert by_id["image_a"].required and by_id["image_a"].type.types == ("comfy.IMAGE",)
    assert by_id["factor"].default == 0.5 and not by_id["factor"].required
    assert by_id["mode"].default == "normal"  # combo defaults to first choice
    # The choice list survives as a static dropdown (wire v9 COMBO) on the
    # concrete core.combo socket; its vocabulary remains presentation only.
    widget = by_id["mode"].widget
    assert isinstance(widget, ComboWidget)
    assert widget.options == ("normal", "multiply", "screen")
    assert not widget.remote_route and not widget.refresh_button
    assert by_id["factor"].widget == NumberWidget(min=0.0, max=1.0)
    assert by_id["label"].default == "" and not by_id["label"].required
    assert [out.id for out in schema.outputs] == ["image"]
    assert schema.idempotent
    assert translation.opaque_types == {"comfy.IMAGE"}


def test_v3_remote_options_untrusted_route_is_whole_node_classified_skip() -> None:
    class V3Remote:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"

        def run(self, choice):  # noqa: ANN001, ANN201
            return (choice,)

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "choice": (
                        "COMBO",
                        {
                            "options": ["a"],
                            "remote": {
                                "route": "/internal/files/output",
                                "refresh_button": True,
                                "control_after_refresh": "first",
                            },
                        },
                    )
                }
            }

    translation = translate_mappings({"Remote": V3Remote})
    assert translation.node_classes == []
    assert "trusted registered Dinkster choice id" in translation.skipped["Remote"]


def test_ksampler_numeric_metadata_translates_as_presentation_only() -> None:
    class KSampler:
        RETURN_TYPES = ("LATENT",)
        FUNCTION = "sample"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "seed": (
                        "INT",
                        {
                            "default": 0,
                            "min": 0,
                            "max": 0xFFFFFFFFFFFFFFFF,
                            "control_after_generate": True,
                        },
                    ),
                    "steps": (
                        "INT",
                        {"default": 20, "min": 1, "max": 10_000, "display": "slider"},
                    ),
                    "cfg": (
                        "FLOAT",
                        {
                            "default": 8.0,
                            "min": 0.0,
                            "max": 100.0,
                            "round": 0.01,
                            "display": "SLIDER",
                        },
                    ),
                    "denoise": (
                        "FLOAT",
                        {
                            "default": 1.0,
                            "min": 0.0,
                            "max": 1.0,
                            "step": 0.01,
                            "display": "number",
                        },
                    ),
                }
            }

        def sample(self, seed, steps, cfg, denoise):  # noqa: ANN001, ANN201
            return ({"seed": seed, "steps": steps, "cfg": cfg, "denoise": denoise},)

    schema = translate_node("KSampler", KSampler, CompatTranslation()).schema()
    by_id = {item.id: item for item in schema.inputs}
    assert by_id["seed"].default == 0
    assert by_id["seed"].widget == NumberWidget(min=0, control_after_generate="randomize")
    assert by_id["steps"].widget == NumberWidget(min=1, max=10_000, display="slider")
    assert by_id["cfg"].widget == NumberWidget(min=0.0, max=100.0, round=0.01)
    assert by_id["denoise"].widget == NumberWidget(min=0.0, max=1.0, step=0.01, display="number")

    wire = schema_to_wire(schema)
    interface = {item["id"]: item for item in wire["interface"]}  # type: ignore[index]
    assert interface["seed"]["widget"] == {
        "type": "NUMBER",
        "min": 0,
        "controlAfterGenerate": "randomize",
    }
    assert interface["denoise"]["widget"] == {
        "type": "NUMBER",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "display": "number",
    }

    without_widgets = translate_node("KSampler", KSampler, CompatTranslation()).schema()
    without_widgets = dataclasses.replace(
        without_widgets,
        inputs=tuple(dataclasses.replace(item, widget=None) for item in without_widgets.inputs),
    )
    assert schema_signature(schema) == schema_signature(without_widgets)


def test_v1_widget_v19_declarations_are_preserved_without_inference() -> None:
    class WidgetFacts:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "prompt": (
                        "STRING",
                        {
                            "multiline": True,
                            "placeholder": "Describe an image",
                            "dynamicPrompts": False,
                            "tooltip": "Prompt text",
                            "display_name": "Prompt",
                            "forceInput": True,
                            "advanced": True,
                        },
                    ),
                    "single_line": ("STRING", {"multiline": False}),
                    "plain": ("STRING", {}),
                    "invalid_mode": ("STRING", {"multiline": "yes"}),
                    "choice": (
                        ["a", "b"],
                        {"control_after_generate": True},
                    ),
                    "color": ("COLOR", {"default": "#123456"}),
                    "opaque": ("COLORS",),
                }
            }

        def run(  # noqa: ANN001, ANN201
            self, prompt, single_line, plain, invalid_mode, choice, color, opaque
        ):
            return (prompt,)

    translation = CompatTranslation()
    schema = translate_node("WidgetFacts", WidgetFacts, translation).schema()
    by_id = {item.id: item for item in schema.inputs}
    assert by_id["prompt"].widget == StringWidget(
        multiline=True,
        placeholder="Describe an image",
        dynamic_prompts=False,
    )
    assert by_id["prompt"].doc == "Prompt text"
    assert by_id["prompt"].display_name == "Prompt"
    assert by_id["prompt"].force_input is True
    assert by_id["prompt"].advanced is True
    assert by_id["single_line"].widget == StringWidget(multiline=False)
    assert by_id["plain"].widget is None
    assert by_id["invalid_mode"].widget is None
    assert by_id["choice"].widget == ComboWidget(
        options=("a", "b"), control_after_generate="randomize"
    )
    assert by_id["color"].type == TypeExpr.concrete("core.string")
    assert by_id["color"].widget == ColorWidget()
    assert by_id["opaque"].type == TypeExpr.concrete("comfy.COLORS")
    assert "comfy.COLOR" not in translation.opaque_types
    assert "comfy.COLORS" in translation.opaque_types

    class ColorOutput:
        RETURN_TYPES = ("COLOR",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {}}

        def run(self):  # noqa: ANN201
            return ("#123456",)

    class ColorInput:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"color": ("COLOR",)}}

        def run(self, color):  # noqa: ANN001, ANN201
            return (color,)

    output_translation = CompatTranslation()
    output_node = translate_node("ColorOutput", ColorOutput, output_translation)
    input_node = translate_node("ColorInput", ColorInput, output_translation)
    output_schema = output_node.schema()
    input_schema = input_node.schema()
    assert output_schema.outputs[0].type == TypeExpr.concrete(CORE_STRING)
    assert output_schema.outputs[0].type == input_schema.inputs[0].type
    assert (
        "widget"
        not in cast("list[dict[str, object]]", schema_to_wire(output_schema)["interface"])[0]
    )
    graph = Graph(
        nodes={
            "source": GraphNode(output_schema.node_type, {}),
            "sink": GraphNode(
                input_schema.node_type,
                {"color": Link("source", output_schema.outputs[0].id)},
            ),
        }
    )
    assert (
        validate(
            graph,
            {
                output_schema.node_type: output_schema,
                input_schema.node_type: input_schema,
            },
            ["sink"],
        )
        == []
    )
    assert "comfy.COLOR" not in output_translation.opaque_types

    class UnknownFacts(WidgetFacts):
        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "prompt": (
                        "STRING",
                        {"placeholder": 1, "dynamicPrompts": "yes", "multiline": "yes"},
                    ),
                    "mixed": (
                        "STRING",
                        {"placeholder": 1, "dynamicPrompts": False, "multiline": False},
                    ),
                    "choice": (["a", "b"], {"control_after_generate": "cycle"}),
                }
            }

    unknown = translate_node("UnknownFacts", UnknownFacts, CompatTranslation()).schema()
    assert unknown.inputs[0].widget is None
    assert unknown.inputs[1].widget == StringWidget(multiline=False, dynamic_prompts=False)
    assert unknown.inputs[2].widget == ComboWidget(options=("a", "b"))


def test_ksampler_advanced_filters_unusable_numeric_metadata() -> None:
    class StringSubclass(str):
        pass

    class KSamplerAdvanced:
        RETURN_TYPES = ("LATENT",)
        FUNCTION = "sample"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "noise_seed": (
                        "INT",
                        {
                            "default": 0,
                            "min": -sys.maxsize,
                            "max": sys.maxsize,
                            "step": 1,
                            "control_after_generate": "increment",
                        },
                    ),
                    "start_at_step": (
                        "INT",
                        {"default": 0, "min": True, "max": 10_000, "step": 0.5},
                    ),
                    "end_at_step": (
                        "INT",
                        {"default": 10_000, "min": 20, "max": 10, "step": 1},
                    ),
                    "cfg": (
                        "FLOAT",
                        {
                            "default": 8.0,
                            "min": float("nan"),
                            "max": float("inf"),
                            "step": -0.5,
                            "control_after_generate": True,
                        },
                    ),
                    "plain": (
                        "FLOAT",
                        {
                            "default": 2.5,
                            "min": Decimal("0.5"),
                            "max": Fraction(3, 2),
                            "step": False,
                        },
                    ),
                    "safe_float_int": (
                        "INT",
                        {
                            "default": 0,
                            "min": -float(2**53 - 1),
                            "max": float(2**53 - 1),
                        },
                    ),
                    "unsafe_float_int": (
                        "INT",
                        {
                            "default": 0,
                            "min": -float(2**53),
                            "max": float(2**53),
                            "step": 1,
                        },
                    ),
                    "false_control": (
                        "INT",
                        {"default": 0, "control_after_generate": False},
                    ),
                    "numeric_control": (
                        "INT",
                        {"default": 0, "control_after_generate": 1},
                    ),
                    "unsupported_control": (
                        "INT",
                        {"default": 0, "control_after_generate": "cycle"},
                    ),
                    "subclass_control": (
                        "INT",
                        {
                            "default": 0,
                            "control_after_generate": StringSubclass("increment"),
                        },
                    ),
                }
            }

        def sample(  # noqa: ANN001, ANN201
            self,
            noise_seed,
            start_at_step,
            end_at_step,
            cfg,
            plain,
            safe_float_int,
            unsafe_float_int,
        ):
            return (
                (
                    noise_seed,
                    start_at_step,
                    end_at_step,
                    cfg,
                    plain,
                    safe_float_int,
                    unsafe_float_int,
                ),
            )

    schema = translate_node("KSamplerAdvanced", KSamplerAdvanced, CompatTranslation()).schema()
    by_id = {item.id: item for item in schema.inputs}
    assert by_id["noise_seed"].widget == NumberWidget(step=1, control_after_generate="increment")
    assert by_id["start_at_step"].widget == NumberWidget(max=10_000)
    assert by_id["end_at_step"].widget == NumberWidget(step=1)
    assert by_id["cfg"].widget is None
    assert by_id["plain"].type == TypeExpr.concrete(CORE_FLOAT)
    assert by_id["plain"].default == 2.5
    assert by_id["plain"].widget is None
    assert by_id["safe_float_int"].widget == NumberWidget(
        min=-float(2**53 - 1), max=float(2**53 - 1)
    )
    assert by_id["unsafe_float_int"].widget == NumberWidget(step=1)
    assert by_id["false_control"].widget is None
    assert by_id["numeric_control"].widget is None
    assert by_id["unsupported_control"].widget is None
    assert by_id["subclass_control"].widget is None

    KSamplerAdvanced.INPUT_IS_LIST = True  # type: ignore[attr-defined]
    listed = translate_node("KSamplerAdvanced", KSamplerAdvanced, CompatTranslation()).schema()
    assert all(item.widget is None for item in listed.inputs)


def test_combo_widgets_use_string_keys_and_omit_empty_or_listed_choices() -> None:
    """Numeric choices use string wire keys; empty and list inputs have no widget."""

    class V1Edges:
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "steps": ([8, 16, 32],),
                    "ckpt_name": ([],),
                }
            }

        def run(self, steps, ckpt_name):  # noqa: ANN001, ANN201
            return (steps,)

    class V1BatchCombo:
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"
        INPUT_IS_LIST = True

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"mode": (["a", "b"],)}}

        def run(self, mode):  # noqa: ANN001, ANN201
            return (len(mode),)

    translation = CompatTranslation()
    edges = translate_node("Edges", V1Edges, translation).schema()
    assert edges.inputs[0].widget == ComboWidget(options=("8", "16", "32"))
    assert edges.inputs[0].default == "8"
    assert edges.inputs[1].widget is None
    assert all(spec.type == TypeExpr.concrete(CORE_COMBO) for spec in edges.inputs)
    batch = translate_node("BatchCombo", V1BatchCombo, translation).schema()
    (mode,) = batch.inputs
    assert mode.type == TypeExpr.list_of(TypeExpr.concrete(CORE_COMBO))
    assert mode.widget is None


@pytest.mark.parametrize("listed", [False, True])
@pytest.mark.parametrize("v3", [False, True])
def test_numeric_combo_wire_keys_restore_original_choice_types(listed: bool, v3: bool) -> None:
    class NumericCombo:
        RETURN_TYPES = ("COMBO",)
        FUNCTION = "run"
        INPUT_IS_LIST = listed
        OUTPUT_IS_LIST = (listed,)

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            options = ["auto", 8, 10, 2.5]
            declaration = (
                ("COMBO", {"options": options, "default": 10}) if v3 else (options, {"default": 10})
            )
            return {"required": {"depth": declaration}}

        def run(self, depth):  # noqa: ANN001, ANN201
            if listed:
                assert depth == [10, 2.5]
                assert type(depth[0]) is int and type(depth[1]) is float
            else:
                assert type(depth) is int and depth == 10
            return (depth,)

    node = translate_node("NumericCombo", NumericCombo, CompatTranslation())
    spec = node.schema().inputs[0]
    assert spec.default == (["10"] if listed else "10")
    assert spec.widget == (None if listed else ComboWidget(options=("auto", "8", "10", "2.5")))
    result = node.execute(depth=["10", "2.5"] if listed else "10")
    assert isinstance(result, Mapping)
    assert result == {"combo": ["10", "2.5"] if listed else "10"}
    registry = TypeRegistry()
    register_core_types(registry)
    output_type = "list<core.combo>" if listed else CORE_COMBO
    assert registry.wrap(output_type, result["combo"]).resolve() == result["combo"]


@pytest.mark.parametrize("options", [[8, "8"], [float("nan")], [float("inf")]])
def test_numeric_combo_rejects_ambiguous_or_nonfinite_choices(options: list[object]) -> None:
    class InvalidCombo:
        RETURN_TYPES = ("COMBO",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"depth": (options,)}}

        def run(self, depth):  # noqa: ANN001, ANN201
            return (depth,)

    with pytest.raises(CompatError, match="ambiguous|finite"):
        translate_node("InvalidCombo", InvalidCombo, CompatTranslation())


def test_v1_multicombo_translates_exact_list_contract_and_refuses_lookalikes() -> None:
    declared_default = ["beta", "alpha", "beta"]

    class V1MultiCombo:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"
        INPUT_IS_LIST = False

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "providers": (
                        ["beta", "alpha", "beta"],
                        {
                            "multiselect": True,
                            "multi_select": {"placeholder": "Pick", "chip": False},
                            "default": declared_default,
                        },
                    )
                }
            }

        def run(self, providers):  # noqa: ANN001, ANN201
            return (",".join(providers),)

    schema = translate_node("MultiCombo", V1MultiCombo, CompatTranslation()).schema()
    (providers,) = schema.inputs
    assert providers.type == TypeExpr.list_of(TypeExpr.concrete(CORE_COMBO))
    assert providers.default == ["beta", "alpha", "beta"]
    assert providers.required is False
    assert providers.widget == MultiComboWidget(
        options=("beta", "alpha", "beta"),
        placeholder="Pick",
        chip=False,
    )
    wire_before_mutation = schema_to_wire(schema)
    signature_before_mutation = schema_signature(schema)
    declared_default.append("pack-mutated")
    assert providers.default == ["beta", "alpha", "beta"]
    assert schema_to_wire(schema) == wire_before_mutation
    assert schema_signature(schema) == signature_before_mutation

    def translated_with(config: Mapping[str, object], options: object = None) -> str:
        class Lookalike:
            RETURN_TYPES = ("STRING",)
            FUNCTION = "run"

            @classmethod
            def INPUT_TYPES(cls):  # noqa: ANN206
                return {
                    "required": {
                        "providers": (
                            ["a", "b"] if options is None else options,
                            config,
                        )
                    }
                }

        return translate_mappings({"Lookalike": Lookalike}).skipped["Lookalike"]

    for config, options, match in (
        ({"multiselect": True}, None, "multi_select"),
        ({"multi_select": {}}, None, "multiselect"),
        ({"multiselect": "yes", "multi_select": {}}, None, "multiselect"),
        ({"multiselect": True, "multi_select": {}, "default": "a"}, None, "default"),
        (
            {"multiselect": True, "multi_select": {}, "control_after_generate": True},
            None,
            "control_after_generate",
        ),
        ({"multiselect": True, "multi_select": {}}, ["a", 1], "options"),
        ({"multiselect": True, "multi_select": {}}, [], "static options or a remote source"),
        ({"multiselect": True, "multi_select": {"unknown": True}}, None, "multi_select"),
    ):
        assert match in translated_with(config, options)

    V1MultiCombo.INPUT_IS_LIST = True  # type: ignore[attr-defined]
    try:
        skipped = translate_mappings({"MultiCombo": V1MultiCombo}).skipped["MultiCombo"]
        assert "INPUT_IS_LIST" in skipped
    finally:
        V1MultiCombo.INPUT_IS_LIST = False


def test_boolean_combo_translates_to_labeled_toggle() -> None:
    """A two-option combo that is a disguised boolean (enable/disable and
    kin, either order, any case) becomes an honest core.boolean wearing
    the original strings as toggle labels; execute() maps the boolean
    back onto the v1 vocabulary, and a legacy prompt still sending the
    string passes through verbatim. v1 BOOLEAN inputs keep their
    label_on/label_off config as the same widget."""

    class V1Noise:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "add_noise": (["enable", "disable"],),
                    "invert": (["OFF", "ON"], {"default": "ON"}),
                    "tidy": ("BOOLEAN", {"default": True, "label_on": "tidy"}),
                    "plain": ("BOOLEAN", {"default": False}),
                }
            }

        def run(self, add_noise, invert, tidy, plain):  # noqa: ANN001, ANN201
            assert isinstance(add_noise, str) and isinstance(invert, str)
            return (f"{add_noise}/{invert}/{tidy}/{plain}",)

    translation = CompatTranslation()
    node = translate_node("Noise", V1Noise, translation)
    by_id = {spec.id: spec for spec in node.schema().inputs}
    add_noise = by_id["add_noise"]
    assert add_noise.type == TypeExpr.concrete(CORE_BOOLEAN)
    assert add_noise.widget == BooleanWidget(label_on="enable", label_off="disable")
    assert add_noise.default is True  # first option "enable" is truthy
    invert = by_id["invert"]  # reversed order, uppercase, explicit default
    assert invert.type == TypeExpr.concrete(CORE_BOOLEAN)
    assert invert.widget == BooleanWidget(label_on="ON", label_off="OFF")
    assert invert.default is True
    assert by_id["tidy"].widget == BooleanWidget(label_on="tidy")
    assert by_id["plain"].widget is None  # label-less boolean: type is enough
    # Honest booleans convert back to the v1 strings; a legacy prompt's
    # original string passes through verbatim.
    out = node.execute(add_noise=False, invert=True, tidy=True, plain=False)
    assert isinstance(out, Mapping) and out["string"] == "disable/ON/True/False"
    out = node.execute(add_noise="enable", invert="OFF", tidy=False, plain=True)
    assert isinstance(out, Mapping) and out["string"] == "enable/OFF/False/True"


def test_two_option_combos_that_are_not_booleans_stay_combos() -> None:
    class V1Pair:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "mode": (["fast", "slow"],),
                    "mixed": (["enable", "off"],),  # mismatched pair
                }
            }

        def run(self, mode, mixed):  # noqa: ANN001, ANN201
            return (mode,)

    translation = CompatTranslation()
    schema = translate_node("Pair", V1Pair, translation).schema()
    for spec in schema.inputs:
        assert spec.type == TypeExpr.concrete(CORE_COMBO)
        assert isinstance(spec.widget, ComboWidget)


def test_return_names_and_collision_ids() -> None:
    translation = CompatTranslation()
    named = translate_node("Loader", V1MultiOut, translation)
    assert [out.id for out in named.schema().outputs] == ["model", "clip", "vae"]
    collided = translate_node("Dup", V1Collide, translation)
    assert [out.id for out in collided.schema().outputs] == ["image", "image_2"]


def test_output_node_and_is_changed_are_never_cached() -> None:
    translation = CompatTranslation()
    assert translate_node("Save", V1Save, translation).schema().idempotent is False
    changed = translate_node("Roll", V1Changed, translation)
    assert changed.schema().idempotent is False

    async def scenario() -> None:
        V1Changed.rolls = 0
        registry = TypeRegistry()
        register_core_types(registry)
        translation.register_types(registry)
        engine = Engine(
            schemas=build_schemas((changed,)),
            registry=registry,
            worker=InProcessWorker(build_node_types((changed,)), registry),
            cache=MemoryLRUCache(),
        )
        graph = Graph({"roll": GraphNode(changed.schema().node_type, {"seed": 3})})
        first = await engine.run(graph, ["roll"])
        second = await engine.run(graph, ["roll"])
        assert first.executed == ("roll",) and second.executed == ("roll",)
        assert first.cached == () and second.cached == ()
        assert V1Changed.rolls == 2

    asyncio.run(scenario())


def test_translation_records_alias_and_output_node() -> None:
    """The v1 class_type survives as a resolvable alias (the prompt adapter's
    lookup key), and OUTPUT_NODE survives as the output_node target hint -
    namespaced or not."""
    translation = CompatTranslation()
    core = translate_node("Blend", V1Blend, translation).schema()
    assert core.node_type == "comfy.Blend" and core.aliases == ("Blend",)
    assert core.output_node is False
    packed = translate_node("Save", V1Save, translation, namespace="mypack").schema()
    assert packed.node_type == "comfy.mypack.Save" and packed.aliases == ("Save",)
    assert packed.output_node is True


def test_comfy_prompt_translates_through_translated_schemas() -> None:
    """End to end at the boundary: a ComfyUI API prompt resolves class_type
    through translated aliases and positional link indexes through the
    translated (id-keyed) outputs."""
    from dinkster_compat_comfy import translate_prompt

    translation = translate_mappings({"Loader": V1MultiOut, "Save": V1Save})
    schemas = {cls.schema().node_type: cls.schema() for cls in translation.node_classes}
    prompt = {
        "1": {"class_type": "Loader", "inputs": {"name": "sd15"}},
        "2": {"class_type": "Save", "inputs": {"text": ["1", 2]}},
    }
    result = translate_prompt(prompt, schemas)
    assert result.targets == ("2",)
    save = result.graph.nodes["2"]
    assert isinstance(save, GraphNode)
    assert save.node_type == "comfy.Save"
    # Positional output 2 of V1MultiOut is the RETURN_NAMES id "vae".
    assert save.inputs["text"] == Link(node_id="1", output_id="vae")


def test_only_filter_rejects_unknown_names() -> None:
    with pytest.raises(CompatError, match="unknown v1 node names"):
        translate_mappings({"Blend": V1Blend}, only=["Blend", "Nope"])


# --- execution: calling convention adaptation ----------------------------


def compat_engine(translation: CompatTranslation) -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    translation.register_types(registry)
    nodes = translation.node_classes
    return Engine(
        schemas=build_schemas(nodes),
        registry=registry,
        worker=InProcessWorker(build_node_types(nodes), registry),
        cache=MemoryLRUCache(),
    )


def test_v1_lazy_hook_is_instance_bound_and_receives_execution_inputs() -> None:
    class LazySwitch:
        RETURN_TYPES = ("INT",)
        FUNCTION = "execute"
        seen: list[tuple[object, object, str | None, tuple[str | None, ...]]] = []

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "selector": (["Off", "On"],),
                    "on_true": ("INT", {"lazy": True}),
                    "on_false": ("INT", {"lazy": True}),
                }
            }

        def check_lazy_status(  # noqa: ANN001, ANN201
            self, selector, on_true=None, on_false=None
        ):
            del on_false
            context = current_execution_context()
            type(self).seen.append(
                (
                    selector,
                    on_true,
                    context.node_id if context is not None else None,
                    (
                        context.diffusion_dtype if context is not None else None,
                        context.text_dtype if context is not None else None,
                        context.vae_dtype if context is not None else None,
                    ),
                )
            )
            return ["on_true"] if selector == "On" and on_true is None else []

        def execute(self, selector, on_true, on_false):  # noqa: ANN001, ANN201
            return (on_true if selector == "On" else on_false,)

    translation = translate_mappings({"LazySwitch": LazySwitch})
    (node_class,) = translation.node_classes
    schemas = build_schemas((node_class,))
    schema = next(iter(schemas.values()))
    registry = TypeRegistry()
    register_core_types(registry)
    translation.register_types(registry)
    result = asyncio.run(
        InProcessWorker(build_node_types((node_class,)), registry).check_lazy_status(
            LazyStatusInvocation(
                request_id="lazy-hook",
                node_id="switch",
                node_type=schema.node_type,
                available_inputs={"selector": registry.wrap(CORE_BOOLEAN, True)},
                connected_undemanded_inputs=("on_true",),
                effective_schema=schema,
                expected_execution_identity="native:dtype-test",
                diffusion_dtype="float16",
                text_dtype="float32",
                vae_dtype="bfloat16",
            )
        )
    )

    on_true = schema.input("on_true")
    assert on_true is not None and on_true.lazy is True
    assert result.requested_inputs == ("on_true",)
    assert LazySwitch.seen == [("On", None, "switch", ("float16", "float32", "bfloat16"))]


def test_async_lazy_hooks_await_in_pack_and_execution_contexts() -> None:
    pack_active = False

    @contextlib.contextmanager
    def pack_context():  # noqa: ANN202 - local worker context fixture
        nonlocal pack_active
        assert not pack_active
        pack_active = True
        try:
            yield
        finally:
            pack_active = False

    class AsyncLazyNode(Node):
        behavior = "request"
        seen: list[tuple[bool, str | None]] = []

        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.async-lazy",
                inputs=(
                    InputSpec("selector", TypeExpr.concrete(CORE_BOOLEAN)),
                    InputSpec("value", TypeExpr.concrete(CORE_INT), lazy=True),
                ),
                outputs=(OutputSpec("value", TypeExpr.concrete(CORE_INT)),),
            )

        @classmethod
        async def check_lazy_status(cls, *, selector: bool, value: int | None) -> object:
            context = current_execution_context()
            cls.seen.append((pack_active, context.node_id if context else None))
            await asyncio.sleep(0)
            context = current_execution_context()
            cls.seen.append((pack_active, context.node_id if context else None))
            if cls.behavior == "error":
                raise RuntimeError("async lazy failure")
            if cls.behavior == "malformed":
                return "value"
            if selector and value is None:
                return ["value"]
            return None

        @classmethod
        def execute(cls, *, selector: bool, value: int | None) -> Mapping[str, object]:
            del selector
            return cls.outputs(value=value)

    registry = TypeRegistry()
    register_core_types(registry)
    worker = InProcessWorker(
        build_node_types((AsyncLazyNode,)), registry, pack_context=pack_context
    )
    schema = AsyncLazyNode.schema()

    def invocation(request_id: str, *, with_value: bool = False) -> LazyStatusInvocation:
        available = {"selector": registry.wrap(CORE_BOOLEAN, True)}
        connected = ("value",)
        if with_value:
            available["value"] = registry.wrap(CORE_INT, 7)
            connected = ()
        return LazyStatusInvocation(
            request_id=request_id,
            node_id="async-lazy",
            node_type=schema.node_type,
            available_inputs=available,
            connected_undemanded_inputs=connected,
            effective_schema=schema,
        )

    first = asyncio.run(worker.check_lazy_status(invocation("round-1")))
    second = asyncio.run(worker.check_lazy_status(invocation("round-2", with_value=True)))
    assert first.requested_inputs == ("value",)
    assert second.requested_inputs == ()
    assert AsyncLazyNode.seen == [(True, "async-lazy")] * 4

    AsyncLazyNode.behavior = "error"
    failed = asyncio.run(worker.check_lazy_status(invocation("error")))
    assert failed.error is not None
    assert failed.error.message == "lazy-hook-failed: async lazy failure"
    assert "RuntimeError: async lazy failure" in failed.error.traceback

    AsyncLazyNode.behavior = "malformed"
    malformed = asyncio.run(worker.check_lazy_status(invocation("malformed")))
    assert malformed.error is not None
    assert malformed.error.message == (
        "lazy-hook-failed: check_lazy_status must return a list or tuple"
    )


def test_translated_async_v1_lazy_hook_returns_awaitable_to_worker() -> None:
    class AsyncLazy:
        RETURN_TYPES = ("INT",)
        FUNCTION = "execute"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206 - v1 shape
            return {"required": {"value": ("INT", {"lazy": True})}}

        async def check_lazy_status(self, value):  # noqa: ANN001, ANN201
            await asyncio.sleep(0)
            return ["value"] if value is None else []

        def execute(self, value):  # noqa: ANN001, ANN201
            return (value,)

    translation = translate_mappings({"AsyncLazy": AsyncLazy})
    assert "AsyncLazy" not in translation.skipped
    (node_class,) = translation.node_classes
    schema = node_class.schema()
    registry = TypeRegistry()
    register_core_types(registry)
    translation.register_types(registry)
    result = asyncio.run(
        InProcessWorker(build_node_types((node_class,)), registry).check_lazy_status(
            LazyStatusInvocation(
                request_id="translated-async-lazy",
                node_id="translated",
                node_type=schema.node_type,
                available_inputs={},
                connected_undemanded_inputs=("value",),
                effective_schema=schema,
            )
        )
    )
    assert result.error is None
    assert result.requested_inputs == ("value",)


def test_v1_whole_list_lazy_sync_preserves_demand_projection_and_cache() -> None:
    class ListSource:
        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("values",)
        OUTPUT_IS_LIST = (True,)
        FUNCTION = "execute"
        calls = 0

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {}

        @classmethod
        def execute(cls):  # noqa: ANN206
            cls.calls += 1
            return ([1, 2, 3],)

    class WholeListLazy:
        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("total",)
        FUNCTION = "execute"
        INPUT_IS_LIST = True
        seen: list[tuple[object, object]] = []
        calls = 0

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {"values": ("INT", {"lazy": True})},
                "optional": {"spare": ("INT", {"lazy": True})},
            }

        def check_lazy_status(self, values, spare="omitted"):  # noqa: ANN001, ANN201
            type(self).seen.append((values, spare))
            if values == (None,):
                return ["values", "values"]
            if values == [1, 2, 3]:
                return ["values"]
            return None

        def execute(self, values, spare="omitted"):  # noqa: ANN001, ANN201
            assert spare == "omitted"
            type(self).calls += 1
            return (sum(values),)

    class NoDemand(WholeListLazy):
        seen: list[tuple[object, object]] = []
        calls = 0

        def check_lazy_status(self, values, spare="omitted"):  # noqa: ANN001, ANN201
            type(self).seen.append((values, spare))
            return None

        def execute(self, values, spare="omitted"):  # noqa: ANN001, ANN201
            assert values == (None,)
            assert spare == "omitted"
            type(self).calls += 1
            return (0,)

    async def scenario() -> None:
        translation = translate_mappings(
            {
                "ListSource": ListSource,
                "WholeListLazy": WholeListLazy,
                "NoDemand": NoDemand,
            }
        )
        assert translation.skipped == {}
        schemas = build_schemas(translation.node_classes)
        lazy_schema = schemas["comfy.WholeListLazy"]
        assert lazy_schema.input("values") == InputSpec(
            "values",
            TypeExpr.list_of(TypeExpr.concrete(CORE_INT)),
            lazy=True,
        )
        registry = TypeRegistry()
        register_core_types(registry)
        translation.register_types(registry)
        nodes = translation.node_classes
        events: list[EngineEvent] = []
        engine = Engine(
            schemas=build_schemas(nodes),
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=events.append,
        )
        demanded_graph = Graph(
            {
                "source": GraphNode("comfy.ListSource"),
                "consumer": GraphNode("comfy.WholeListLazy", {"values": Link("source", "values")}),
            }
        )
        cold = await engine.run(demanded_graph, ["consumer"])
        warm = await engine.run(demanded_graph, ["consumer"])
        assert cold.outputs["consumer"]["total"].resolve() == 6
        assert warm.outputs["consumer"]["total"].resolve() == 6
        assert cold.executed == ("source", "consumer")
        assert warm.cached == ("source", "consumer")
        assert ListSource.calls == 1
        assert WholeListLazy.calls == 1
        assert WholeListLazy.seen == [
            ((None,), "omitted"),
            ([1, 2, 3], "omitted"),
            ((None,), "omitted"),
            ([1, 2, 3], "omitted"),
        ]

        no_demand_graph = Graph(
            {
                "unused": GraphNode("comfy.ListSource"),
                "consumer": GraphNode("comfy.NoDemand", {"values": Link("unused", "values")}),
            }
        )
        no_demand = await engine.run(no_demand_graph, ["consumer"])
        assert no_demand.outputs["consumer"]["total"].resolve() == 0
        assert no_demand.executed == ("consumer",)
        assert NoDemand.seen == [((None,), "omitted")]
        assert NoDemand.calls == 1
        demands = [
            cast("Mapping[str, object]", event.detail["data"])
            for event in events
            if event.kind == "node_event" and event.detail.get("name") == "lazy_demand"
        ]
        assert [demand["status"] for demand in demands] == [
            "waiting",
            "ready",
            "waiting",
            "ready",
            "ready",
        ]
        assert [demand["newInputs"] for demand in demands] == [
            ["values"],
            [],
            ["values"],
            [],
            [],
        ]
        assert demands[0]["producerNodes"] == ["source"]
        assert demands[-1]["requestedInputs"] == []

    asyncio.run(scenario())


def test_v3_whole_list_lazy_async_failure_cancellation_and_non_idempotent_retry() -> None:
    class ListSource:
        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("values",)
        OUTPUT_IS_LIST = (True,)
        FUNCTION = "execute"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {}

        @classmethod
        def execute(cls):  # noqa: ANN206
            return ([4, 5],)

    class AsyncV3WholeList:
        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("total",)
        FUNCTION = "execute"
        INPUT_IS_LIST = True
        behavior = "error"
        calls = 0
        cleaned = 0
        started = asyncio.Event()
        release = asyncio.Event()

        @classmethod
        def GET_NODE_INFO_V1(cls):  # noqa: ANN206, N802
            return {"name": "AsyncV3WholeList"}

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"values": ("INT", {"lazy": True})}}

        @classmethod
        def IS_CHANGED(cls, values):  # noqa: ANN001, ANN206, N802
            return float("nan")

        async def check_lazy_status(self, values):  # noqa: ANN001, ANN201
            if values != (None,):
                return None
            if type(self).behavior == "error":
                raise RuntimeError("whole-list hook failed")
            if type(self).behavior == "cancel":
                type(self).started.set()
                try:
                    await type(self).release.wait()
                finally:
                    type(self).cleaned += 1
            await asyncio.sleep(0)
            return ["values"]

        def execute(self, values):  # noqa: ANN001, ANN201
            type(self).calls += 1
            return (sum(values),)

    async def scenario() -> None:
        translation = translate_mappings(
            {"ListSource": ListSource, "AsyncV3WholeList": AsyncV3WholeList}
        )
        assert translation.skipped == {}
        engine = compat_engine(translation)
        graph = Graph(
            {
                "source": GraphNode("comfy.ListSource"),
                "consumer": GraphNode(
                    "comfy.AsyncV3WholeList", {"values": Link("source", "values")}
                ),
            }
        )

        with pytest.raises(ExecutionError, match="whole-list hook failed"):
            await engine.run(graph, ["consumer"])

        AsyncV3WholeList.behavior = "cancel"
        cancelled = asyncio.create_task(engine.run(graph, ["consumer"]))
        await asyncio.wait_for(AsyncV3WholeList.started.wait(), 2)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert AsyncV3WholeList.cleaned == 1
        assert not engine._inflight

        AsyncV3WholeList.behavior = "ready"
        first = await engine.run(graph, ["consumer"])
        second = await engine.run(graph, ["consumer"])
        assert first.outputs["consumer"]["total"].resolve() == 9
        assert second.outputs["consumer"]["total"].resolve() == 9
        assert AsyncV3WholeList.calls == 2
        assert "consumer" in first.executed and "consumer" in second.executed
        assert "consumer" not in first.cached and "consumer" not in second.cached
        assert not engine._inflight

    asyncio.run(scenario())


def test_v1_tuple_outputs_become_id_keyed_values() -> None:
    async def scenario() -> None:
        translation = translate_mappings({"Loader": V1MultiOut})
        engine = compat_engine(translation)
        graph = Graph(nodes={"l": GraphNode("comfy.Loader", {"name": "sd15"})})
        result = await engine.run(graph, ["l"])
        outs = result.outputs["l"]
        assert outs["model"].resolve() == "model:sd15"
        assert outs["clip"].resolve() == "clip:sd15"
        assert outs["vae"].resolve() == "vae:sd15"
        # Opaque values are real envelopes: typed and fingerprinted.
        assert outs["model"].type_id == comfy_type_id("MODEL")
        assert outs["model"].fingerprint

    asyncio.run(scenario())


def test_output_node_runs_with_hidden_inputs_defaulted() -> None:
    async def scenario() -> None:
        V1Save.saved.clear()
        translation = translate_mappings({"Save": V1Save})
        engine = compat_engine(translation)
        graph = Graph(nodes={"s": GraphNode("comfy.Save", {"text": "hello"})})
        await engine.run(graph, ["s"])
        assert V1Save.saved == [("hello", None, "s")]

    asyncio.run(scenario())


def test_v1_unique_id_receives_submitted_prompt_node_id() -> None:
    class V1UniqueId:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"
        OUTPUT_NODE = True

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {}, "hidden": {"unique_id": "UNIQUE_ID"}}

        def run(self, unique_id=...):  # noqa: ANN001, ANN201
            return (unique_id,)

    async def scenario() -> None:
        translation = translate_mappings({"UniqueId": V1UniqueId})
        lowered = translate_prompt(
            {
                "submitted-prompt-node-17": {
                    "class_type": "UniqueId",
                    "inputs": {},
                }
            },
            build_schemas(translation.node_classes),
        )
        result = await compat_engine(translation).run(lowered.graph, lowered.targets)
        assert (
            result.outputs["submitted-prompt-node-17"]["string"].resolve()
            == "submitted-prompt-node-17"
        )

    asyncio.run(scenario())


def test_unique_id_input_is_list_receives_wrapped_node_id() -> None:
    seen: list[object] = []

    class V1UniqueIdList:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"
        INPUT_IS_LIST = True

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {}, "hidden": {"unique_id": "UNIQUE_ID"}}

        def run(self, unique_id=...):  # noqa: ANN001, ANN201
            seen.append(unique_id)
            return ("ok",)

    async def scenario() -> None:
        translation = translate_mappings({"UniqueIdList": V1UniqueIdList})
        engine = compat_engine(translation)
        graph = Graph(nodes={"submitted-list-node": GraphNode("comfy.UniqueIdList", {})})
        result = await engine.run(graph, ["submitted-list-node"])
        assert seen == [["submitted-list-node"]]
        assert result.outputs["submitted-list-node"]["string"].resolve() == "ok"

    asyncio.run(scenario())


def test_prompt_hidden_stays_none_and_unique_id_direct_call_falls_back_to_none() -> None:
    seen: list[tuple[object, object]] = []

    class V1HiddenFallback:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {},
                "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
            }

        def run(self, prompt=..., unique_id=...):  # noqa: ANN001, ANN201
            seen.append((prompt, unique_id))
            return ("ok",)

    node = translate_node("HiddenFallback", V1HiddenFallback, CompatTranslation())
    node.execute()
    assert seen == [(None, None)]


def test_non_output_wrapper_keeps_export_hidden_values_none() -> None:
    seen: list[tuple[object, object]] = []

    class V1NonOutputHidden:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }

        def run(self, prompt=..., extra_pnginfo=...):  # noqa: ANN001, ANN201
            seen.append((prompt, extra_pnginfo))
            return ("ok",)

    async def scenario() -> None:
        translation = translate_mappings({"NonOutputHidden": V1NonOutputHidden})
        assert translation.node_classes[0].schema().idempotent is True
        result = await compat_engine(translation).run(
            Graph(nodes={"non-output": GraphNode("comfy.NonOutputHidden", {})}),
            ["non-output"],
            export_snapshot=ExportSnapshot(
                prompt={"save": {"class_type": "Save", "inputs": {}}},
                extra_pnginfo={"workflow": {"nodes": []}},
            ),
        )
        assert result.outputs["non-output"]["string"].resolve() == "ok"

    asyncio.run(scenario())
    assert seen == [(None, None)]


def test_output_wrappers_do_not_share_extra_pnginfo_mutations() -> None:
    observed: list[object] = []

    class V1MutatingOutput:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"
        OUTPUT_NODE = True

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {}, "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"}}

        def run(self, extra_pnginfo=...):  # noqa: ANN001, ANN201
            assert isinstance(extra_pnginfo, dict)
            extra_pnginfo["mutated"] = True
            return ("done",)

    class V1ObservingOutput:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"
        OUTPUT_NODE = True

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {"dependency": "STRING"},
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
            }

        def run(self, dependency, extra_pnginfo=...):  # noqa: ANN001, ANN201
            observed.append(extra_pnginfo)
            return (dependency,)

    async def scenario() -> None:
        translation = translate_mappings(
            {"MutatingOutput": V1MutatingOutput, "ObservingOutput": V1ObservingOutput}
        )
        graph = Graph(
            nodes={
                "mutate": GraphNode("comfy.MutatingOutput", {}),
                "observe": GraphNode(
                    "comfy.ObservingOutput",
                    {"dependency": Link("mutate", "string")},
                ),
            }
        )
        await compat_engine(translation).run(
            graph,
            ["observe"],
            export_snapshot=ExportSnapshot(
                prompt={"observe": {"class_type": "ObservingOutput", "inputs": {}}},
                extra_pnginfo={"workflow": {"nodes": []}},
            ),
        )

    asyncio.run(scenario())
    assert observed == [{"workflow": {"nodes": []}}]


def test_unique_id_nodes_are_not_shared_across_engine_cache_keys() -> None:
    class V1UniqueCacheProbe:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {}, "hidden": {"unique_id": "UNIQUE_ID"}}

        def run(self, unique_id=...):  # noqa: ANN001, ANN201
            return (unique_id,)

    async def scenario() -> None:
        translation = translate_mappings({"UniqueCacheProbe": V1UniqueCacheProbe})
        assert translation.node_classes[0].schema().idempotent is False
        engine = compat_engine(translation)
        graph = Graph(
            nodes={
                "cache-node-a": GraphNode("comfy.UniqueCacheProbe", {}),
                "cache-node-b": GraphNode("comfy.UniqueCacheProbe", {}),
            }
        )
        result = await engine.run(graph, ["cache-node-a", "cache-node-b"])
        assert result.outputs["cache-node-a"]["string"].resolve() == "cache-node-a"
        assert result.outputs["cache-node-b"]["string"].resolve() == "cache-node-b"
        assert result.cached == ()

    asyncio.run(scenario())


def test_opaque_values_flow_between_compat_nodes() -> None:
    """Unknown value types use the default boundary without media conversion."""

    class OpaqueBlend(V1Blend):
        RETURN_TYPES = ("OPAQUE",)
        RETURN_NAMES = ("image",)

        @classmethod
        def INPUT_TYPES(cls):  # noqa: N802
            inputs = super().INPUT_TYPES()
            inputs["required"]["image_a"] = ("OPAQUE",)
            inputs["required"]["image_b"] = ("OPAQUE",)
            return inputs

    class OpaqueDup(V1Collide):
        RETURN_TYPES = ("OPAQUE", "OPAQUE")
        RETURN_NAMES = ("image", "other_image")

        @classmethod
        def INPUT_TYPES(cls):  # noqa: N802
            return {"required": {"x": ("OPAQUE",)}}

    async def scenario() -> None:
        translation = translate_mappings({"Blend": OpaqueBlend, "Dup": OpaqueDup})
        engine = compat_engine(translation)
        graph = Graph(
            nodes={
                "b": GraphNode(
                    "comfy.Blend",
                    {
                        "image_a": "pretend-image-1",
                        "image_b": "pretend-image-2",
                        "factor": 0.25,
                        "mode": "multiply",
                    },
                ),
                "d": GraphNode("comfy.Dup", {"x": Link("b", "image")}),
            }
        )
        result = await engine.run(graph, ["d"])
        blended: Mapping[str, object] = result.outputs["d"]["image"].resolve()  # type: ignore[assignment]
        assert blended == {
            "a": "pretend-image-1",
            "b": "pretend-image-2",
            "f": 0.25,
            "m": "multiply",
        }

    asyncio.run(scenario())


def test_translated_image_and_mask_share_the_image_array_codec() -> None:
    """comfy.IMAGE and comfy.MASK cross as npy bytes (the shared image-array
    codec), never pickle, so each stays one value type with its native
    spelling (dinkster.image / dinkster.mask)."""

    class V1MaskInvert:
        RETURN_TYPES = ("MASK",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"mask": ("MASK",)}}

        def run(self, mask):  # noqa: ANN001, ANN201
            return (1.0 - mask,)

    translation = translate_mappings({"Blend": V1Blend, "InvertMask": V1MaskInvert})
    registry = TypeRegistry()
    register_core_types(registry)
    translation.register_types(registry)
    for type_id in ("comfy.IMAGE", "comfy.MASK"):
        assert registry.spec(type_id).prepare_buffer_encoding is prepare_image_array_encoding


# --- execution: INPUT_IS_LIST / OUTPUT_IS_LIST as list<T> sockets --------


class V1Batch:
    """INPUT_IS_LIST: v1 calls the function once with every input a list."""

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("total", "count")
    FUNCTION = "run"
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {"required": {"xs": ("INT",), "scale": ("INT", {"default": 2})}}

    def run(self, xs, scale):  # noqa: ANN001, ANN201
        return (sum(xs) * scale[0], len(xs))


class V1Split:
    """OUTPUT_IS_LIST on one of two outputs: mixed list/scalar returns."""

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("parts", "count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {"required": {"n": ("INT", {"default": 3})}}

    def run(self, n):  # noqa: ANN001, ANN201
        return (list(range(n)), n)


class V1Double:
    """Both flags: list in, list out - v1's whole-list-through shape."""

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("doubled",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {"required": {"xs": ("INT",)}}

    def run(self, xs):  # noqa: ANN001, ANN201
        return ([x * 2 for x in xs],)


def test_zero_input_compat_node_executes_once() -> None:
    class V1ZeroInput:
        calls = 0
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {}}

        def run(self):  # noqa: ANN201
            type(self).calls += 1
            return (7,)

    async def scenario() -> None:
        translation = translate_mappings({"ZeroInput": V1ZeroInput})
        result = await compat_engine(translation).run(
            Graph(nodes={"zero": GraphNode("comfy.ZeroInput", {})}),
            ["zero"],
        )
        assert result.outputs["zero"]["int"].resolve() == 7
        assert V1ZeroInput.calls == 1

    asyncio.run(scenario())


def test_list_flags_translate_schema_shapes() -> None:
    translation = CompatTranslation()
    schema = translate_node("Batch", V1Batch, translation).schema()
    int_list = TypeExpr.list_of(TypeExpr.concrete(CORE_INT))
    # INPUT_IS_LIST wraps every input; widget defaults arrive length-1
    # wrapped in v1, so the declared default translates to [default].
    assert [spec.type for spec in schema.inputs] == [int_list, int_list]
    assert schema.inputs[1].default == [2]
    assert [out.type for out in schema.outputs] == [
        TypeExpr.concrete(CORE_INT),
        TypeExpr.concrete(CORE_INT),
    ]
    split = translate_node("Split", V1Split, translation).schema()
    assert split.outputs[0].type == int_list
    assert split.outputs[1].type == TypeExpr.concrete(CORE_INT)


def test_input_is_list_calls_once_with_whole_lists() -> None:
    async def scenario() -> None:
        translation = translate_mappings({"Batch": V1Batch})
        engine = compat_engine(translation)
        graph = Graph(nodes={"b": GraphNode("comfy.Batch", {"xs": [1, 2, 3]})})
        result = await engine.run(graph, ["b"])
        outs = result.outputs["b"]
        assert outs["total"].resolve() == 12  # sum * default scale [2]
        assert outs["count"].resolve() == 3

    asyncio.run(scenario())


def test_output_is_list_values_flow_as_native_lists() -> None:
    async def scenario() -> None:
        translation = translate_mappings({"Split": V1Split, "Double": V1Double})
        engine = compat_engine(translation)
        graph = Graph(
            nodes={
                "s": GraphNode("comfy.Split", {"n": 3}),
                "d": GraphNode("comfy.Double", {"xs": Link("s", "parts")}),
            }
        )
        result = await engine.run(graph, ["d"])
        parts = result.outputs["d"]["doubled"]
        assert parts.resolve() == [0, 2, 4]
        # The envelope carries the canonical runtime list type id.
        assert parts.type_id == "list<core.int>"

    asyncio.run(scenario())


def test_v1_fanout_scalar_mapping_and_list_aggregation_execute_end_to_end() -> None:
    class V1Scale:
        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("value",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"value": ("INT",), "factor": ("INT",)}}

        def run(self, value, factor):  # noqa: ANN001, ANN201
            return (value * factor,)

    class V1Collect:
        INPUT_IS_LIST = True
        OUTPUT_IS_LIST = (True,)
        OUTPUT_NODE = True
        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("values",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"values": ("INT",)}}

        def run(self, values):  # noqa: ANN001, ANN201
            return (values,)

    async def scenario() -> None:
        from dinkster_compat_comfy import translate_prompt

        translation = translate_mappings({"Split": V1Split, "Scale": V1Scale, "Collect": V1Collect})
        schemas = build_schemas(translation.node_classes)
        prompt = {
            "split": {"class_type": "Split", "inputs": {"n": 3}},
            "scale": {
                "class_type": "Scale",
                "inputs": {"value": ["split", 0], "factor": 4},
            },
            "collect": {
                "class_type": "Collect",
                "inputs": {"values": ["scale", 0]},
            },
        }
        translated = translate_prompt(prompt, schemas)
        assert isinstance(translated.graph.nodes["scale"], RegionNode)
        result = await compat_engine(translation).run(translated.graph, translated.targets)
        assert result.outputs["collect"]["values"].resolve() == [0, 4, 8]

    asyncio.run(scenario())


def test_core_list_wave_split_tiles_map_and_whole_list_merge() -> None:
    class SplitImageToTileList:
        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("tiles",)
        OUTPUT_IS_LIST = (True,)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"image": ("INT",)}}

        def run(self, image):  # noqa: ANN001, ANN201
            return ([image * 10 + index for index in range(image + 1)],)

    class TileConsumer:
        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("tile",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"tile": ("INT",)}}

        def run(self, tile):  # noqa: ANN001, ANN201
            return (tile + 100,)

    class ImageMergeTileList:
        INPUT_IS_LIST = True
        OUTPUT_NODE = True
        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("merged",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"tiles": ("INT",)}}

        def run(self, tiles):  # noqa: ANN001, ANN201
            return (",".join(str(tile) for tile in tiles),)

    async def scenario() -> None:
        from dinkster_compat_comfy import translate_prompt

        translation = translate_mappings(
            {
                "Split": V1Split,
                "SplitImageToTileList": SplitImageToTileList,
                "TileConsumer": TileConsumer,
                "ImageMergeTileList": ImageMergeTileList,
            }
        )
        schemas = build_schemas(translation.node_classes)
        prompt = {
            "images": {"class_type": "Split", "inputs": {"n": 3}},
            "tiles": {
                "class_type": "SplitImageToTileList",
                "inputs": {"image": ["images", 0]},
            },
            "mapped": {
                "class_type": "TileConsumer",
                "inputs": {"tile": ["tiles", 0]},
            },
            "merge": {
                "class_type": "ImageMergeTileList",
                "inputs": {"tiles": ["mapped", 0]},
            },
        }
        translated = translate_prompt(prompt, schemas)
        tiles = translated.graph.nodes["tiles"]
        assert isinstance(tiles, RegionNode)
        assert tiles.outputs["tiles"].mode == "flatten"
        result = await compat_engine(translation).run(translated.graph, translated.targets)
        assert result.outputs["merge"]["merged"].resolve() == "100,110,111,120,121,122"

    asyncio.run(scenario())


@pytest.mark.parametrize("node_name", ["RebatchImages", "RebatchLatents"])
def test_core_list_wave_whole_list_rebatch_in_and_out(node_name: str) -> None:
    class Rebatch:
        INPUT_IS_LIST = True
        OUTPUT_IS_LIST = (True,)
        OUTPUT_NODE = True
        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("batches",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"items": ("INT",), "batch_size": ("INT",)}}

        def run(self, items, batch_size):  # noqa: ANN001, ANN201
            size = batch_size[0]
            return ([sum(items[index : index + size]) for index in range(0, len(items), size)],)

    async def scenario() -> None:
        from dinkster_compat_comfy import translate_prompt

        translation = translate_mappings({"Split": V1Split, node_name: Rebatch})
        schemas = build_schemas(translation.node_classes)
        translated = translate_prompt(
            {
                "items": {"class_type": "Split", "inputs": {"n": 5}},
                "rebatch": {
                    "class_type": node_name,
                    "inputs": {"items": ["items", 0], "batch_size": [2]},
                },
            },
            schemas,
        )
        assert isinstance(translated.graph.nodes["rebatch"], GraphNode)
        result = await compat_engine(translation).run(translated.graph, translated.targets)
        assert result.outputs["rebatch"]["batches"].resolve() == [1, 5, 4]

    asyncio.run(scenario())


def test_core_list_wave_shuffle_preserves_aligned_multi_output_order() -> None:
    class PairSource:
        OUTPUT_IS_LIST = (True, True)
        RETURN_TYPES = ("INT", "STRING")
        RETURN_NAMES = ("images", "texts")
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"count": ("INT",)}}

        def run(self, count):  # noqa: ANN001, ANN201
            return (list(range(count)), [f"caption-{index}" for index in range(count)])

    class ShuffleImageTextDataset:
        INPUT_IS_LIST = True
        OUTPUT_IS_LIST = (True, True)
        OUTPUT_NODE = True
        RETURN_TYPES = ("INT", "STRING")
        RETURN_NAMES = ("images", "texts")
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"images": ("INT",), "texts": ("STRING",)}}

        def run(self, images, texts):  # noqa: ANN001, ANN201
            order = (2, 0, 1)
            return ([images[index] for index in order], [texts[index] for index in order])

    async def scenario() -> None:
        from dinkster_compat_comfy import translate_prompt

        translation = translate_mappings(
            {"PairSource": PairSource, "ShuffleImageTextDataset": ShuffleImageTextDataset}
        )
        schemas = build_schemas(translation.node_classes)
        translated = translate_prompt(
            {
                "source": {"class_type": "PairSource", "inputs": {"count": 3}},
                "shuffle": {
                    "class_type": "ShuffleImageTextDataset",
                    "inputs": {"images": ["source", 0], "texts": ["source", 1]},
                },
            },
            schemas,
        )
        result = await compat_engine(translation).run(translated.graph, translated.targets)
        assert result.outputs["shuffle"]["images"].resolve() == [2, 0, 1]
        assert result.outputs["shuffle"]["texts"].resolve() == [
            "caption-2",
            "caption-0",
            "caption-1",
        ]

    asyncio.run(scenario())


def test_core_list_wave_temporal_chunk_map_and_merge_flattens_in_order() -> None:
    class SeedVR2TemporalChunk:
        OUTPUT_IS_LIST = (True,)
        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("latents",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"latent": ("INT",)}}

        def run(self, latent):  # noqa: ANN001, ANN201
            return ([latent * 10, latent * 10 + 1],)

    class ChunkConsumer:
        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("latent",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"latent": ("INT",)}}

        def run(self, latent):  # noqa: ANN001, ANN201
            return (latent + 1,)

    class SeedVR2TemporalMerge:
        INPUT_IS_LIST = True
        OUTPUT_NODE = True
        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("video",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"latents": ("INT",)}}

        def run(self, latents):  # noqa: ANN001, ANN201
            return ("/".join(str(latent) for latent in latents),)

    async def scenario() -> None:
        from dinkster_compat_comfy import translate_prompt

        translation = translate_mappings(
            {
                "Split": V1Split,
                "SeedVR2TemporalChunk": SeedVR2TemporalChunk,
                "ChunkConsumer": ChunkConsumer,
                "SeedVR2TemporalMerge": SeedVR2TemporalMerge,
            }
        )
        schemas = build_schemas(translation.node_classes)
        translated = translate_prompt(
            {
                "videos": {"class_type": "Split", "inputs": {"n": 3}},
                "chunks": {
                    "class_type": "SeedVR2TemporalChunk",
                    "inputs": {"latent": ["videos", 0]},
                },
                "mapped": {
                    "class_type": "ChunkConsumer",
                    "inputs": {"latent": ["chunks", 0]},
                },
                "merge": {
                    "class_type": "SeedVR2TemporalMerge",
                    "inputs": {"latents": ["mapped", 0]},
                },
            },
            schemas,
        )
        chunks = translated.graph.nodes["chunks"]
        assert isinstance(chunks, RegionNode)
        assert chunks.outputs["latents"].mode == "flatten"
        result = await compat_engine(translation).run(translated.graph, translated.targets)
        assert result.outputs["merge"]["video"].resolve() == "1/2/11/12/21/22"

    asyncio.run(scenario())


def test_core_list_wave_mapped_dataset_loader_flattens_aligned_outputs() -> None:
    class LoadImageTextDataSetFromFolder:
        OUTPUT_IS_LIST = (True, True)
        OUTPUT_NODE = True
        RETURN_TYPES = ("INT", "STRING")
        RETURN_NAMES = ("images", "texts")
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"folder": ("INT",)}}

        def run(self, folder):  # noqa: ANN001, ANN201
            images = [folder * 10 + index for index in range(folder + 1)]
            texts = [f"{folder}:{index}" for index in range(folder + 1)]
            return (images, texts)

    async def scenario() -> None:
        from dinkster_compat_comfy import translate_prompt

        translation = translate_mappings(
            {"Split": V1Split, "LoadImageTextDataSetFromFolder": LoadImageTextDataSetFromFolder}
        )
        schemas = build_schemas(translation.node_classes)
        translated = translate_prompt(
            {
                "folders": {"class_type": "Split", "inputs": {"n": 2}},
                "load": {
                    "class_type": "LoadImageTextDataSetFromFolder",
                    "inputs": {"folder": ["folders", 0]},
                },
            },
            schemas,
        )
        loader = translated.graph.nodes["load"]
        assert isinstance(loader, RegionNode)
        assert [output.mode for output in loader.outputs.values()] == ["flatten", "flatten"]
        result = await compat_engine(translation).run(translated.graph, translated.targets)
        assert result.outputs["load"]["images"].resolve() == [0, 10, 11]
        assert result.outputs["load"]["texts"].resolve() == ["0:0", "1:0", "1:1"]

    asyncio.run(scenario())


def test_core_list_wave_wandancer_emits_three_aligned_lists_once() -> None:
    class WanDancerPadKeyframesList:
        OUTPUT_IS_LIST = (True, True, True)
        OUTPUT_NODE = True
        RETURN_TYPES = ("INT", "INT", "STRING")
        RETURN_NAMES = ("keyframes_sequence", "keyframes_mask", "audio_segment")
        FUNCTION = "run"
        calls = 0

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"segments": ("INT",)}}

        @classmethod
        def run(cls, segments):  # noqa: ANN001, ANN206
            cls.calls += 1
            return (
                list(range(segments)),
                [index % 2 for index in range(segments)],
                [f"audio-{index}" for index in range(segments)],
            )

    async def scenario() -> None:
        from dinkster_compat_comfy import translate_prompt

        translation = translate_mappings({"WanDancerPadKeyframesList": WanDancerPadKeyframesList})
        schemas = build_schemas(translation.node_classes)
        translated = translate_prompt(
            {
                "pad": {
                    "class_type": "WanDancerPadKeyframesList",
                    "inputs": {"segments": 3},
                }
            },
            schemas,
        )
        assert isinstance(translated.graph.nodes["pad"], GraphNode)
        WanDancerPadKeyframesList.calls = 0
        result = await compat_engine(translation).run(translated.graph, translated.targets)
        assert WanDancerPadKeyframesList.calls == 1
        assert result.outputs["pad"]["keyframes_sequence"].resolve() == [0, 1, 2]
        assert result.outputs["pad"]["keyframes_mask"].resolve() == [0, 1, 0]
        assert result.outputs["pad"]["audio_segment"].resolve() == [
            "audio-0",
            "audio-1",
            "audio-2",
        ]

    asyncio.run(scenario())


def test_v1_mapping_result_with_expand_refuses_loudly() -> None:
    class V1Expand:
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"n": ("INT", {"default": 1})}}

        def run(self, n):  # noqa: ANN001, ANN201
            return {"result": (n,), "expand": {"nodes": {}}}

    node = translate_node("Expand", V1Expand, CompatTranslation())
    with pytest.raises(CompatError) as exc_info:
        node.execute(n=1)
    assert str(exc_info.value) == (
        "Expand: v1 result requested graph expansion, which compat does not support "
        "(docs/compat-porting-recipes.md, 'Graph expansion dispositions')"
    )


def test_v1_mapping_result_with_none_expand_refuses_by_key_presence() -> None:
    class V1ExpandNone:
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"n": ("INT", {"default": 1})}}

        def run(self, n):  # noqa: ANN001, ANN201
            return {"expand": None, "result": (n,)}

    node = translate_node("ExpandNone", V1ExpandNone, CompatTranslation())
    with pytest.raises(CompatError, match="v1 result requested graph expansion"):
        node.execute(n=1)


def test_v1_scalar_execution_blocker_refuses_loudly() -> None:
    class ExecutionBlocker:
        pass

    class V1Blocked:
        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("value",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"n": ("INT", {"default": 1})}}

        def run(self, n):  # noqa: ANN001, ANN201, ARG002
            return (ExecutionBlocker(),)

    node = translate_node("Blocked", V1Blocked, CompatTranslation())
    with pytest.raises(CompatError) as exc_info:
        node.execute(n=1)
    assert str(exc_info.value) == (
        "Blocked: output 'value' returned an ExecutionBlocker, which compat does not "
        "support (docs/compat-porting-recipes.md, 'ExecutionBlocker to Dinkster absence "
        "semantics')"
    )


def test_v1_output_is_list_execution_blocker_refuses_loudly() -> None:
    class ExecutionBlocker:
        pass

    class V1BlockedList:
        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("values",)
        OUTPUT_IS_LIST = (True,)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"n": ("INT", {"default": 1})}}

        def run(self, n):  # noqa: ANN001, ANN201
            return ([n, ExecutionBlocker()],)

    node = translate_node("BlockedList", V1BlockedList, CompatTranslation())
    with pytest.raises(CompatError, match="output 'values' returned an ExecutionBlocker"):
        node.execute(n=1)


def test_v1_ui_result_mapping_still_unwraps_positional_result() -> None:
    class V1UIResult:
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"n": ("INT", {"default": 1})}}

        def run(self, n):  # noqa: ANN001, ANN201
            return {"ui": {"texts": [str(n)]}, "result": (n + 1,)}

    node = translate_node("UIResult", V1UIResult, CompatTranslation())
    assert node.execute(n=1) == {"int": 2}


def test_flagged_output_returning_non_list_raises_at_cause() -> None:
    class V1Liar:
        RETURN_TYPES = ("INT",)
        OUTPUT_IS_LIST = (True,)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"n": ("INT", {"default": 1})}}

        def run(self, n):  # noqa: ANN001, ANN201
            return (n,)  # scalar where a list was promised

    node = translate_node("Liar", V1Liar, CompatTranslation())
    with pytest.raises(CompatError, match="OUTPUT_IS_LIST"):
        node.execute(n=1)


def test_output_is_list_flag_tolerance() -> None:
    class V1ShortFlags:
        """v1 would silently drop the unflagged trailing output; short
        declarations mean False for the rest."""

        RETURN_TYPES = ("INT", "INT")
        OUTPUT_IS_LIST = (True,)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"n": ("INT", {"default": 1})}}

        def run(self, n):  # noqa: ANN001, ANN201
            return ([n], n)

    schema = translate_node("Short", V1ShortFlags, CompatTranslation()).schema()
    assert schema.outputs[0].type == TypeExpr.list_of(TypeExpr.concrete(CORE_INT))
    assert schema.outputs[1].type == TypeExpr.concrete(CORE_INT)

    class V1BadFlags:
        RETURN_TYPES = ("INT",)
        OUTPUT_IS_LIST = True  # not a sequence
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"n": ("INT",)}}

        def run(self, n):  # noqa: ANN001, ANN201
            return ([n],)

    with pytest.raises(CompatError, match="OUTPUT_IS_LIST must be a tuple"):
        translate_node("Bad", V1BadFlags, CompatTranslation())


def test_input_is_list_hidden_inputs_arrive_wrapped() -> None:
    seen: list[object] = []

    class V1HiddenBatch:
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"
        INPUT_IS_LIST = True

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {"xs": ("INT",)},
                "hidden": {"unique_id": "UNIQUE_ID"},
            }

        def run(self, xs, unique_id=...):  # noqa: ANN001, ANN201
            seen.append(unique_id)
            return (sum(xs),)

    node = translate_node("HiddenBatch", V1HiddenBatch, CompatTranslation())
    node.execute(xs=[1, 2])
    # A direct call has no invocation context, so the fallback remains None.
    assert seen == [[None]]


# --- V3 io.ComfyNode classes riding NODE_CLASS_MAPPINGS ------------------
# ComfyUI registers V3 nodes into the same mapping (nodes.py); their
# classproperties fake the v1 surface and FUNCTION names
# EXECUTE_NORMALIZED, which always returns io.NodeOutput. These fakes
# mirror comfy_api/latest/_io.py @ 947c2749 exactly where compat looks:
# the base-class NAME (detection) and the result/ui/expand/
# block_execution attributes (unwrapping).


class _NodeOutputInternal:
    """Same name as comfy_api.internal._NodeOutputInternal on purpose:
    compat detects NodeOutput by base-class name, never by import."""


class FakeNodeOutput(_NodeOutputInternal):
    def __init__(self, *args, ui=None, expand=None, block_execution=None):  # noqa: ANN002, ANN001, ANN204
        self.args = args
        self.ui = ui
        self.expand = expand
        self.block_execution = block_execution

    @property
    def result(self):  # noqa: ANN201 - mirrors the reference property
        return self.args if len(self.args) > 0 else None


class V3PrimitiveInt:
    """The v1-shim shape of a V3 primitive (nodes_primitive.py Int)."""

    CATEGORY = "utilities/primitive"
    RETURN_TYPES = ("INT",)
    FUNCTION = "EXECUTE_NORMALIZED"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {"required": {"value": ("INT", {"default": 0})}}

    @classmethod
    def EXECUTE_NORMALIZED(cls, value):  # noqa: ANN001, ANN206
        return FakeNodeOutput(value)


class V3CustomCombo:
    CATEGORY = "utilities"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("STRING", "INDEX")
    FUNCTION = "EXECUTE_NORMALIZED"
    _ACCEPT_ALL_INPUTS = True
    OUTPUT_IS_LIST = (False, False)
    seen_kwargs: dict[str, object] = {}

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {"required": {"choice": ("COMBO", {"multiselect": False, "options": []})}}

    @classmethod
    def GET_NODE_INFO_V1(cls):  # noqa: ANN206
        return {"name": "CustomCombo"}

    @classmethod
    def execute(cls, choice, index=0, **kwargs):  # noqa: ANN001, ANN003, ANN206
        cls.seen_kwargs = kwargs
        return FakeNodeOutput(choice, index)

    @classmethod
    def EXECUTE_NORMALIZED(cls, *args, **kwargs):  # noqa: ANN002, ANN003, ANN206
        return cls.execute(*args, **kwargs)


def test_v3_custom_combo_translates_and_executes_closed_option_family() -> None:
    translation = CompatTranslation()
    node = translate_node("CustomCombo", V3CustomCombo, translation)
    schema = node.schema()
    assert [item.id for item in schema.inputs] == ["choice", "index"]
    assert schema.inputs[0].type == TypeExpr.concrete(CORE_COMBO)
    assert schema.inputs[1] == InputSpec(
        "index", TypeExpr.concrete(CORE_INT), required=False, default=0
    )
    (family,) = schema.input_families
    assert family.id == "options"
    assert family.type == TypeExpr.concrete(CORE_STRING)
    assert family.member_names == tuple(f"option{index}" for index in range(1, 101))
    assert node.execute(
        choice="second",
        index=1,
        options={"option1": "first", "option2": "second"},
    ) == {"STRING": "second", "INDEX": 1}
    assert V3CustomCombo.seen_kwargs == {"options": {"option1": "first", "option2": "second"}}
    assert translation.skipped == {}


def test_v3_custom_combo_runtime_family_refuses_open_or_malformed_values() -> None:
    node = translate_node("CustomCombo", V3CustomCombo, CompatTranslation())
    with pytest.raises(CompatError, match="contiguous"):
        node.execute(choice="x", options={"option1": "x", "option3": "z"})
    for options in (
        {"option101": "x"},
        {"option0": "x"},
        {"option1": "x", "option101": "y"},
    ):
        with pytest.raises(CompatError, match="contiguous"):
            node.execute(choice="x", options=options)
    with pytest.raises(CompatError, match="must be a string"):
        node.execute(choice="x", options={"option1": 1})
    with pytest.raises(CompatError, match="undeclared execution inputs"):
        node.execute(choice="x", arbitrary="value")


def test_v3_custom_combo_lookalikes_and_generic_accept_all_stay_skipped() -> None:
    class EqualEmptyList:
        def __eq__(self, other: object) -> bool:
            return other == []

    class WrongOptions(V3CustomCombo):
        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "choice": (
                        "COMBO",
                        {"multiselect": False, "options": ["fixed"]},
                    )
                }
            }

    translation = translate_mappings({"CustomCombo": WrongOptions, "AcceptAll": V3CustomCombo})
    assert "accept_all_inputs" in translation.skipped["CustomCombo"]
    assert "accept_all_inputs" in translation.skipped["AcceptAll"]

    class EqualityOptions(V3CustomCombo):
        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "choice": (
                        "COMBO",
                        {"multiselect": False, "options": EqualEmptyList()},
                    )
                }
            }

    equality_options = translate_mappings({"CustomCombo": EqualityOptions})
    assert "accept_all_inputs" in equality_options.skipped["CustomCombo"]

    namespaced = translate_mappings({"CustomCombo": V3CustomCombo}, namespace="custom_pack")
    assert "accept_all_inputs" in namespaced.skipped["custom_pack.CustomCombo"]

    class ListedOutput(V3CustomCombo):
        OUTPUT_IS_LIST = (False, True)

    listed = translate_mappings({"CustomCombo": ListedOutput})
    assert "accept_all_inputs" in listed.skipped["CustomCombo"]

    class NonBooleanAcceptAll(V3CustomCombo):
        _ACCEPT_ALL_INPUTS = 1

    non_boolean = translate_mappings({"CustomCombo": NonBooleanAcceptAll})
    assert "accept_all_inputs" in non_boolean.skipped["CustomCombo"]

    class FalseyOutputFlag(V3CustomCombo):
        OUTPUT_IS_LIST = (False, 0)

    falsey_output = translate_mappings({"CustomCombo": FalseyOutputFlag})
    assert "accept_all_inputs" in falsey_output.skipped["CustomCombo"]

    class FalseyInputList(V3CustomCombo):
        INPUT_IS_LIST = 0

    falsey_input_list = translate_mappings({"CustomCombo": FalseyInputList})
    assert "accept_all_inputs" in falsey_input_list.skipped["CustomCombo"]

    class EqualString:
        def __init__(self, value: str) -> None:
            self.value = value

        def __eq__(self, other: object) -> bool:
            return self.value == other

    class EqualityReturnType(V3CustomCombo):
        RETURN_TYPES = (EqualString("STRING"), EqualString("INT"))

    equality_return_type = translate_mappings({"CustomCombo": EqualityReturnType})
    assert "accept_all_inputs" in equality_return_type.skipped["CustomCombo"]

    class EqualityReturnName(V3CustomCombo):
        RETURN_NAMES = (EqualString("STRING"), EqualString("INDEX"))

    equality_return_name = translate_mappings({"CustomCombo": EqualityReturnName})
    assert "accept_all_inputs" in equality_return_name.skipped["CustomCombo"]

    class BooleanIndex(V3CustomCombo):
        @classmethod
        def execute(  # type: ignore[override]
            cls,
            choice,
            index=False,
            **kwargs,  # noqa: ANN001, ANN003
        ):  # noqa: ANN206
            return FakeNodeOutput(choice, index)

    boolean_index = translate_mappings({"CustomCombo": BooleanIndex})
    assert "accept_all_inputs" in boolean_index.skipped["CustomCombo"]


class V3UniqueId:
    """Pinned V3 shim hidden shape from ComfyUI 947c2749.

    comfy_api/latest/_io.py get_v1_info emits (hidden.value,), Hidden.unique_id
    is the string UNIQUE_ID, and execution.py's V3 hidden branch supplies the
    real unique id. INPUT_TYPES is read directly in-process, without JSON.
    """

    CATEGORY = "test"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "EXECUTE_NORMALIZED"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: ANN206
        return {"required": {}, "hidden": {"unique_id": ("UNIQUE_ID",)}}

    @classmethod
    def EXECUTE_NORMALIZED(cls, unique_id=...):  # noqa: ANN206
        return FakeNodeOutput(unique_id)


def test_v3_shim_unique_id_tuple_receives_submitted_node_id() -> None:
    async def scenario() -> None:
        translation = translate_mappings({"V3UniqueId": V3UniqueId})
        engine = compat_engine(translation)
        graph = Graph(nodes={"v3-node-23": GraphNode("comfy.V3UniqueId", {})})
        result = await engine.run(graph, ["v3-node-23"])
        assert result.outputs["v3-node-23"]["string"].resolve() == "v3-node-23"
        assert translation.node_classes[0].schema().idempotent is False

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "hidden_type",
    [
        ("UNIQUE_ID", "x"),
        ["UNIQUE_ID"],
        ("PROMPT",),
        "PROMPT",
    ],
)
def test_only_closed_unique_id_hidden_shapes_are_synthesized(hidden_type: object) -> None:
    seen: list[object] = []

    class HiddenShape:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {}, "hidden": {"value": hidden_type}}

        def run(self, value=...):  # noqa: ANN001, ANN201
            seen.append(value)
            return ("ok",)

    async def scenario() -> None:
        translation = translate_mappings({"HiddenShape": HiddenShape})
        node = translation.node_classes[0]
        assert node.schema().idempotent is True
        await compat_engine(translation).run(
            Graph(nodes={"negative-shape": GraphNode("comfy.HiddenShape", {})}),
            ["negative-shape"],
        )

    asyncio.run(scenario())
    assert seen == [None]


def test_v3_node_output_unwraps_to_id_keyed_values() -> None:
    """The user-blocking bug: every V3 core node (primitives first)
    failed at execute with 'returned NodeOutput, expected a tuple'.
    The NodeOutput shape must unwrap to the v1 tuple."""

    async def scenario() -> None:
        translation = translate_mappings({"PrimitiveInt": V3PrimitiveInt})
        engine = compat_engine(translation)
        graph = Graph(nodes={"i": GraphNode("comfy.PrimitiveInt", {"value": 7})})
        result = await engine.run(graph, ["i"])
        assert result.outputs["i"]["int"].resolve() == 7

    asyncio.run(scenario())


def test_v3_ui_only_node_output_is_a_valid_empty_result() -> None:
    """NodeOutput(ui=...) with no positional results is the V3 output-
    node convention (SaveImage and kin): ui drops like the v1 ui-dict
    half, zero declared outputs means an empty value map."""

    class V3Save:
        RETURN_TYPES = ()
        FUNCTION = "EXECUTE_NORMALIZED"
        OUTPUT_NODE = True

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"text": ("STRING", {"default": ""})}}

        @classmethod
        def EXECUTE_NORMALIZED(cls, text):  # noqa: ANN001, ANN206
            return FakeNodeOutput(ui={"texts": [text]})

    node = translate_node("Save", V3Save, CompatTranslation())
    assert node.execute(text="hi") == {}


def test_v3_class_clone_receives_hidden_values_without_execute_kwargs() -> None:
    prepared: list[object] = []
    executed: list[object] = []

    class V3PreparedSave:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "EXECUTE_NORMALIZED"
        OUTPUT_NODE = True
        hidden: dict[str, object] = {}

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {"text": ("STRING",)},
                "hidden": {
                    "prompt": ("PROMPT",),
                    "extra_pnginfo": ("EXTRA_PNGINFO",),
                    "unique_id": ("UNIQUE_ID",),
                },
            }

        @classmethod
        def PREPARE_CLASS_CLONE(cls, v3_data):  # noqa: ANN001, ANN206
            prepared.append(v3_data)
            hidden = v3_data["hidden_inputs"]

            class Prepared(cls):
                pass

            Prepared.hidden = hidden
            return Prepared

        @classmethod
        def EXECUTE_NORMALIZED(cls, text):  # noqa: ANN001, ANN206
            executed.append((text, cls.hidden))
            return FakeNodeOutput(text)

    node = translate_node("PreparedSave", V3PreparedSave, CompatTranslation())
    snapshot = ExportSnapshot(prompt={"save": {}}, extra_pnginfo={"workflow": {}})
    with use_execution_context(
        ExecutionContext(
            arm=None,
            expected_execution_identity=None,
            node_id="save-17",
            export_snapshot=snapshot,
        )
    ):
        assert node.execute(text="ready") == {"string": "ready"}

    expected = {
        "PROMPT": snapshot.prompt,
        "EXTRA_PNGINFO": snapshot.extra_pnginfo,
        "UNIQUE_ID": "save-17",
    }
    assert prepared == [{"hidden_inputs": expected}]
    assert executed == [("ready", expected)]


def test_v3_expansion_and_blocking_refuse_loudly() -> None:
    class V3Expand:
        RETURN_TYPES = ("INT",)
        FUNCTION = "EXECUTE_NORMALIZED"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"n": ("INT", {"default": 0})}}

        @classmethod
        def EXECUTE_NORMALIZED(cls, n):  # noqa: ANN001, ANN206
            return FakeNodeOutput(n, expand={"nodes": {}})

    class V3Blocked:
        RETURN_TYPES = ("INT",)
        FUNCTION = "EXECUTE_NORMALIZED"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"n": ("INT", {"default": 0})}}

        @classmethod
        def EXECUTE_NORMALIZED(cls, n):  # noqa: ANN001, ANN206
            return FakeNodeOutput(block_execution="upstream said no")

    expand_node = translate_node("Expand", V3Expand, CompatTranslation())
    with pytest.raises(CompatError, match="expansion"):
        expand_node.execute(n=1)
    blocked_node = translate_node("Blocked", V3Blocked, CompatTranslation())
    with pytest.raises(CompatError, match="upstream said no"):
        blocked_node.execute(n=1)


def test_v3_output_count_mismatch_still_raises() -> None:
    """Unwrapping must not weaken the arity check: a NodeOutput with the
    wrong number of positional results is still a loud error."""

    class V3Short:
        RETURN_TYPES = ("INT", "INT")
        FUNCTION = "EXECUTE_NORMALIZED"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"n": ("INT", {"default": 0})}}

        @classmethod
        def EXECUTE_NORMALIZED(cls, n):  # noqa: ANN001, ANN206
            return FakeNodeOutput(n)

    node = translate_node("Short", V3Short, CompatTranslation())
    with pytest.raises(CompatError, match="returned 1 values"):
        node.execute(n=1)


def test_async_v3_node_executes_and_unwraps_after_await() -> None:
    """FUNCTION = EXECUTE_NORMALIZED_ASYNC follows the ordinary worker loop."""

    class V3Async:
        RETURN_TYPES = ("INT",)
        FUNCTION = "EXECUTE_NORMALIZED_ASYNC"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"n": ("INT", {"default": 0})}}

        @classmethod
        async def EXECUTE_NORMALIZED_ASYNC(cls, n):  # noqa: ANN001, ANN206
            await asyncio.sleep(0)
            return FakeNodeOutput(n + 1)

    async def scenario() -> None:
        translation = translate_mappings({"Async": V3Async})
        result = await compat_engine(translation).run(
            Graph(nodes={"async": GraphNode("comfy.Async", {"n": 7})}),
            ["async"],
        )
        assert result.outputs["async"]["int"].resolve() == 8

    asyncio.run(scenario())


def test_async_v1_failure_keeps_exception_fidelity() -> None:
    class V1AsyncError:
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"n": ("INT", {"default": 0})}}

        async def run(self, n):  # noqa: ANN001, ANN201, ARG002
            await asyncio.sleep(0)
            raise RuntimeError("upstream async failure")

    async def scenario() -> None:
        translation = translate_mappings({"AsyncError": V1AsyncError})
        with pytest.raises(ExecutionError) as exc_info:
            await compat_engine(translation).run(
                Graph(nodes={"async": GraphNode("comfy.AsyncError", {"n": 1})}),
                ["async"],
            )
        assert exc_info.value.error.message == "upstream async failure"
        assert "in run" in exc_info.value.error.traceback
        assert "RuntimeError: upstream async failure" in exc_info.value.error.traceback

    asyncio.run(scenario())


def test_async_v1_cancellation_runs_cleanup_and_allows_retry() -> None:
    started = asyncio.Event()
    cleaned = asyncio.Event()
    release = asyncio.Event()

    class V1AsyncResource:
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"
        active = False

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"n": ("INT", {"default": 0})}}

        async def run(self, n):  # noqa: ANN001, ANN201
            if type(self).active:
                raise RuntimeError("resource leaked across invocation")
            type(self).active = True
            started.set()
            try:
                await release.wait()
                return (n + 1,)
            finally:
                type(self).active = False
                cleaned.set()

    async def scenario() -> None:
        translation = translate_mappings({"AsyncResource": V1AsyncResource})
        engine = compat_engine(translation)
        graph = Graph(nodes={"async": GraphNode("comfy.AsyncResource", {"n": 4})})
        task = asyncio.create_task(engine.run(graph, ["async"]))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(cleaned.wait(), 1)
        assert V1AsyncResource.active is False

        release.set()
        result = await engine.run(graph, ["async"])
        assert result.outputs["async"]["int"].resolve() == 5
        assert V1AsyncResource.active is False

    asyncio.run(scenario())


def test_async_v1_list_mapping_preserves_output_order() -> None:
    class V1AsyncScale:
        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("value",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"value": ("INT",)}}

        async def run(self, value):  # noqa: ANN001, ANN201
            await asyncio.sleep((2 - value) * 0.001)
            return (value * 10,)

    class V1Collect:
        INPUT_IS_LIST = True
        OUTPUT_NODE = True
        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("values",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"values": ("INT",)}}

        def run(self, values):  # noqa: ANN001, ANN201
            return (",".join(str(value) for value in values),)

    async def scenario() -> None:
        translation = translate_mappings(
            {"Split": V1Split, "AsyncScale": V1AsyncScale, "Collect": V1Collect}
        )
        schemas = build_schemas(translation.node_classes)
        translated = translate_prompt(
            {
                "split": {"class_type": "Split", "inputs": {"n": 3}},
                "scale": {
                    "class_type": "AsyncScale",
                    "inputs": {"value": ["split", 0]},
                },
                "collect": {
                    "class_type": "Collect",
                    "inputs": {"values": ["scale", 0]},
                },
            },
            schemas,
        )
        assert isinstance(translated.graph.nodes["scale"], RegionNode)
        result = await compat_engine(translation).run(translated.graph, translated.targets)
        assert result.outputs["collect"]["values"].resolve() == "0,10,20"

    asyncio.run(scenario())


# --- V3 dynamic declarations through the v1 shim -----------------------


class _V3Type:
    def __init__(self, io_type: str) -> None:
        self.io_type = io_type


class _V3Template:
    def __init__(self, template_id: str, *allowed_types: str) -> None:
        self.template_id = template_id
        self.allowed_types = [_V3Type(io_type) for io_type in allowed_types]


class _V3Output:
    def __init__(self, template: _V3Template) -> None:
        self.template = template


class _V3Schema:
    def __init__(self, *outputs: _V3Output) -> None:
        self.outputs = outputs


def test_v3_switch_lazy_matchtype_markers_keep_selector_lowering() -> None:
    class ComfySwitchNode:
        RETURN_TYPES = ("COMFY_MATCHTYPE_V3",)
        RETURN_NAMES = ("output",)
        FUNCTION = "run"
        SCHEMA = _V3Schema(_V3Output(_V3Template("switch", "*")))

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            template = {"template_id": "switch", "allowed_types": "*"}
            return {
                "required": {
                    "switch": ("BOOLEAN", {}),
                },
                "optional": {
                    "on_false": (
                        "COMFY_MATCHTYPE_V3",
                        {"template": template, "lazy": True},
                    ),
                    "on_true": (
                        "COMFY_MATCHTYPE_V3",
                        {"template": template, "lazy": True},
                    ),
                }
            }

        @classmethod
        def check_lazy_status(cls, switch, on_false=None, on_true=None):  # noqa: ANN001, ANN202
            if switch and on_true is None:
                return ["on_true"]
            if not switch and on_false is None:
                return ["on_false"]

        def run(self, switch, on_false, on_true):  # noqa: ANN001, ANN201
            return (on_true if switch else on_false,)

    translation = CompatTranslation()
    node_class = translate_node("ComfySwitchNode", ComfySwitchNode, translation)
    schema = node_class.schema()
    assert schema.selector == SelectorSpec("switch", {"false": "on_false", "true": "on_true"})
    assert [spec.lazy for spec in schema.inputs] == [False, True, True]
    assert [spec.required for spec in schema.inputs] == [True, False, False]

    registry = TypeRegistry()
    register_core_types(registry)
    translation.register_types(registry)
    worker = InProcessWorker(build_node_types((node_class,)), registry)

    def invocation(request_id: str, *, with_selected_value: bool = False) -> LazyStatusInvocation:
        available = {"switch": registry.wrap(CORE_BOOLEAN, True)}
        connected = ("on_false", "on_true")
        if with_selected_value:
            available["on_true"] = registry.wrap(CORE_INT, 7)
            connected = ("on_false",)
        return LazyStatusInvocation(
            request_id=request_id,
            node_id="switch",
            node_type=schema.node_type,
            available_inputs=available,
            connected_undemanded_inputs=connected,
            effective_schema=schema,
        )

    waiting = asyncio.run(worker.check_lazy_status(invocation("waiting")))
    ready = asyncio.run(worker.check_lazy_status(invocation("ready", with_selected_value=True)))
    assert waiting.requested_inputs == ("on_true",)
    assert ready.error is None
    assert ready.requested_inputs == ()


def test_v3_matchtype_input_output_preserves_variable_relation() -> None:
    class V3Switch:
        RETURN_TYPES = ("COMFY_MATCHTYPE_V3",)
        RETURN_NAMES = ("output",)
        FUNCTION = "run"
        SCHEMA = _V3Schema(_V3Output(_V3Template("switch", "INT", "COMBO", "MODEL")))

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            template = {"template_id": "switch", "allowed_types": "INT,COMBO,MODEL"}
            return {
                "required": {
                    "on_false": ("COMFY_MATCHTYPE_V3", {"template": template}),
                    "on_true": ("COMFY_MATCHTYPE_V3", {"template": template}),
                }
            }

        def run(self, on_false, on_true):  # noqa: ANN001, ANN201
            return (on_true if on_true is not None else on_false,)

    translation = CompatTranslation()
    schema = translate_node("Switch", V3Switch, translation).schema()
    expected = TypeExpr.variable("switch", (CORE_INT, CORE_COMBO, "comfy.MODEL"))
    assert [spec.type for spec in schema.inputs] == [expected, expected]
    assert schema.outputs[0].type == expected
    assert translation.opaque_types == {"comfy.MODEL"}


def test_v1_shim_admits_lazy_whole_list_matchtype_template() -> None:
    class WholeListMatch:
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"
        INPUT_IS_LIST = True

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "values": (
                        "COMFY_MATCHTYPE_V3",
                        {
                            "template": {
                                "template_id": "item",
                                "allowed_types": "INT,STRING",
                            },
                            "lazy": True,
                        },
                    )
                }
            }

        @classmethod
        def check_lazy_status(cls, values):  # noqa: ANN001, ANN206
            return [] if values != (None,) else ["values"]

        def run(self, values):  # noqa: ANN001, ANN201
            return (len(values),)

    schema = translate_node("WholeListMatch", WholeListMatch, CompatTranslation()).schema()
    assert schema.inputs == (
        InputSpec(
            "values",
            TypeExpr.list_of(TypeExpr.variable("item", (CORE_INT, CORE_STRING))),
            lazy=True,
        ),
    )
    assert schema.selector is None


def test_v3_matchtype_any_and_one_sided_templates() -> None:
    class V3InputOnly:
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "optional": {
                    "value": (
                        "COMFY_MATCHTYPE_V3",
                        {"template": {"template_id": "any", "allowed_types": "*"}},
                    )
                }
            }

        def run(self, value=None):  # noqa: ANN001, ANN201
            return (0 if value is None else value,)

    class V3OutputOnly:
        RETURN_TYPES = ("COMFY_MATCHTYPE_V3",)
        FUNCTION = "run"
        SCHEMA = _V3Schema(_V3Output(_V3Template("out", "FLOAT", "STRING")))

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {}}

        def run(self):  # noqa: ANN201
            return (1.0,)

    input_schema = translate_node("InputOnly", V3InputOnly, CompatTranslation()).schema()
    assert input_schema.inputs[0].type == TypeExpr.variable("any")
    output_schema = translate_node("OutputOnly", V3OutputOnly, CompatTranslation()).schema()
    assert output_schema.outputs[0].type == TypeExpr.variable("out", (CORE_FLOAT, CORE_STRING))


def test_v3_unrecoverable_matchtype_output_is_skipped() -> None:
    class V3BrokenOutput:
        RETURN_TYPES = ("COMFY_MATCHTYPE_V3",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {}}

        def run(self):  # noqa: ANN201
            return (None,)

    translation = translate_mappings({"BrokenOutput": V3BrokenOutput})
    assert not translation.node_classes
    assert "template is unrecoverable" in translation.skipped["BrokenOutput"]
    assert not any(type_id.startswith("comfy.COMFY_") for type_id in translation.opaque_types)


def _autogrow_config(
    member_type: object = "INT",
    member_config: Mapping[str, object] | None = None,
    *,
    min_members: int = 0,
    max_members: int = 4,
) -> dict[str, object]:
    return {
        "template": {
            "input": {"required": {"member": (member_type, member_config or {})}},
            "prefix": "item_",
            "min": min_members,
            "max": max_members,
        }
    }


def test_v3_prefix_autogrow_translates_and_executes_nested_dict() -> None:
    class V3Autogrow:
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"
        seen: list[dict[str, object]] = []

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"items": ("COMFY_AUTOGROW_V3", _autogrow_config(min_members=0))}}

        def run(self, items):  # noqa: ANN001, ANN201
            V3Autogrow.seen.append(items)
            return (len(items),)

    node = translate_node("Autogrow", V3Autogrow, CompatTranslation())
    schema = node.schema()
    assert schema.inputs == ()
    (family,) = schema.input_families
    assert family.type == TypeExpr.concrete(CORE_INT)
    assert family.min_members == 0
    assert family.max_members == 4

    V3Autogrow.seen.clear()
    assert node.execute(items={"second": 20, "first": 10}) == {"int": 2}
    assert V3Autogrow.seen == [{"item_0": 20, "item_1": 10}]
    assert node.execute() == {"int": 0}
    assert V3Autogrow.seen[-1] == {}


def test_v3_matchtype_autogrow_lists_share_output_variable_and_execute() -> None:
    template = {"template_id": "type", "allowed_types": "INT,STRING"}

    class V3CreateList:
        RETURN_TYPES = ("COMFY_MATCHTYPE_V3",)
        RETURN_NAMES = ("list",)
        OUTPUT_IS_LIST = (True,)
        INPUT_IS_LIST = True
        FUNCTION = "run"
        SCHEMA = _V3Schema(_V3Output(_V3Template("type", "INT", "STRING")))
        seen: list[dict[str, object]] = []

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "inputs": (
                        "COMFY_AUTOGROW_V3",
                        _autogrow_config(
                            "COMFY_MATCHTYPE_V3",
                            {"template": template},
                            min_members=0,
                            max_members=8,
                        ),
                    )
                }
            }

        @classmethod
        def run(cls, inputs):  # noqa: ANN001, ANN206
            cls.seen.append(inputs)
            output: list[object] = []
            for member in inputs.values():
                output.extend(member)
            return FakeNodeOutput(output)

    translation = CompatTranslation()
    node = translate_node("CreateList", V3CreateList, translation)
    schema = node.schema()
    variable = TypeExpr.variable("type", (CORE_INT, CORE_STRING))
    list_variable = TypeExpr.list_of(variable)
    assert schema.inputs == ()
    assert schema.input_families[0].type == list_variable
    assert schema.input_families[0].min_members == 0
    assert schema.input_families[0].max_members == 8
    assert schema.outputs[0].type == list_variable

    V3CreateList.seen.clear()
    assert node.execute(inputs={"second": [2, 3], "first": [1]}) == {"list": [2, 3, 1]}
    assert V3CreateList.seen == [{"item_0": [2, 3], "item_1": [1]}]
    assert node.execute() == {"list": []}
    assert V3CreateList.seen[-1] == {}

    async def scenario() -> None:
        runtime = translate_mappings({"CreateList": V3CreateList, "Split": V1Split})
        engine = compat_engine(runtime)
        graph = Graph(
            nodes={
                "a": GraphNode("comfy.Split", {"n": 2}),
                "b": GraphNode("comfy.Split", {"n": 3}),
                "list": GraphNode(
                    "comfy.CreateList",
                    {
                        "inputs.first": Link("a", "parts"),
                        "inputs.second": Link("b", "parts"),
                    },
                ),
            }
        )
        result = await engine.run(graph, ["list"])
        value = result.outputs["list"]["list"]
        assert value.type_id == "list<core.int>"
        assert value.resolve() == [0, 1, 0, 1, 2]

    asyncio.run(scenario())


def test_v3_matchtype_autogrow_anytype_is_unconstrained() -> None:
    class V3AnyList:
        RETURN_TYPES = ("COMFY_MATCHTYPE_V3",)
        OUTPUT_IS_LIST = (True,)
        INPUT_IS_LIST = True
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            template = {"template_id": "any", "allowed_types": "*"}
            return {
                "required": {
                    "items": (
                        "COMFY_AUTOGROW_V3",
                        _autogrow_config("COMFY_MATCHTYPE_V3", {"template": template}),
                    )
                }
            }

        @classmethod
        def GET_NODE_INFO_V1(cls):  # noqa: N802, ANN206 - V3 shim API
            return {"output_matchtypes": ["any"]}

        def run(self, items):  # noqa: ANN001, ANN201
            return ([],)

    schema = translate_node("AnyList", V3AnyList, CompatTranslation()).schema()
    expected = TypeExpr.list_of(TypeExpr.variable("any"))
    assert schema.input_families[0].type == expected
    assert schema.outputs[0].type == expected


def test_v3_names_autogrow_translates_and_nested_templates_stay_classified() -> None:
    class V3Names:
        RETURN_TYPES = ()
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            config = _autogrow_config()
            template = config["template"]
            assert isinstance(template, dict)
            template.pop("prefix")
            template.pop("max")
            template["names"] = ["left", "right"]
            return {"required": {"items": ("COMFY_AUTOGROW_V3", config)}}

        def run(self, items):  # noqa: ANN001, ANN201
            return ()

    def nested_node(marker: str, config: Mapping[str, object] | None = None):
        class V3Nested(V3Names):
            @classmethod
            def INPUT_TYPES(cls):  # noqa: ANN206
                return {
                    "required": {
                        "items": (
                            "COMFY_AUTOGROW_V3",
                            _autogrow_config(marker, config),
                        )
                    }
                }

        return V3Nested

    translation = translate_mappings(
        {
            "Names": V3Names,
            "DynamicCombo": nested_node("COMFY_DYNAMICCOMBO_V3"),
            "DynamicSlot": nested_node("COMFY_DYNAMICSLOT_V3"),
            "NestedAutogrow": nested_node("COMFY_AUTOGROW_V3"),
            "BrokenMatch": nested_node("COMFY_MATCHTYPE_V3"),
        }
    )
    names_node = next(
        node for node in translation.node_classes if node.schema().node_type == "comfy.Names"
    )
    family = names_node.schema().input_families[0]
    assert family.member_names == ("left", "right")
    assert family.max_members == 2
    assert names_node.execute(items={"right": 2, "left": 1}) == {}
    for name in ("DynamicCombo", "DynamicSlot", "NestedAutogrow"):
        assert "unsupported nested dynamic marker" in translation.skipped[name]
    assert "MatchType lacks a template mapping" in translation.skipped["BrokenMatch"]
    assert not any(type_id.startswith("comfy.COMFY_") for type_id in translation.opaque_types)


def test_translation_skips_export_returns_classified_dynamic_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class V3Nested:
        RETURN_TYPES = ()
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "items": (
                        "COMFY_AUTOGROW_V3",
                        _autogrow_config("COMFY_DYNAMICSLOT_V3"),
                    )
                }
            }

        def run(self, items):  # noqa: ANN001, ANN201
            return ()

    translation = translate_mappings({"Nested": V3Nested})
    from dinkster_compat_comfy import bootstrap

    monkeypatch.setattr(
        bootstrap,
        "load_comfyui_nodes",
        lambda *, required=(): translation,
    )
    sys.modules.pop("dinkster_compat_comfy.entry", None)
    try:
        entry = importlib.import_module("dinkster_compat_comfy.entry")
        assert entry.translation_skips() == {"Nested": translation.diagnostics["Nested"]}
        diagnostic = entry.translation_skips()["Nested"]
        assert "nested dynamic marker" in diagnostic.reason
        assert diagnostic.code == "compat.dynamic.unsupported"
        assert diagnostic.input_path == ("items", "member")
        assert diagnostic.path_kind == "dynamic-family"
    finally:
        sys.modules.pop("dinkster_compat_comfy.entry", None)


def test_nested_autogrow_diagnostic_preserves_full_dynamic_family_path() -> None:
    class V3NestedAutogrow:
        RETURN_TYPES = ()
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "outer": (
                        "COMFY_DYNAMICCOMBO_V3",
                        _dynamic_combo_config(
                            [
                                (
                                    "selected",
                                    {
                                        "required": {
                                            "grow": (
                                                "COMFY_AUTOGROW_V3",
                                                _autogrow_config("COMFY_DYNAMICSLOT_V3"),
                                            )
                                        }
                                    },
                                )
                            ]
                        ),
                    )
                }
            }

        def run(self, outer):  # noqa: ANN001, ANN201
            return ()

    diagnostic = translate_mappings({"NestedAutogrow": V3NestedAutogrow}).diagnostics[
        "NestedAutogrow"
    ]
    assert diagnostic.input_path == ("outer", "grow", "member")
    assert diagnostic.input_id == "member"
    assert diagnostic.path_kind == "dynamic-family"


def test_v3_combo_string_translates_static_options_and_default() -> None:
    class V3Combo:
        RETURN_TYPES = ("COMBO",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "mode": (
                        "COMBO",
                        {"options": ["nearest", "bilinear"], "default": "bilinear"},
                    )
                }
            }

        def run(self, mode):  # noqa: ANN001, ANN201
            return (mode,)

    node = translate_node("Combo", V3Combo, CompatTranslation())
    spec = node.schema().inputs[0]
    assert spec.type == TypeExpr.concrete(CORE_COMBO)
    assert spec.default == "bilinear"
    assert spec.widget == ComboWidget(options=("nearest", "bilinear"))
    assert node.schema().outputs[0].type == TypeExpr.concrete(CORE_COMBO)


def test_v3_combo_string_disguised_boolean_executes_original_string() -> None:
    class V3Toggle:
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"
        seen: list[str] = []

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "enabled": (
                        "COMBO",
                        {"options": ["disable", "enable"], "default": "disable"},
                    )
                }
            }

        def run(self, enabled):  # noqa: ANN001, ANN201
            V3Toggle.seen.append(enabled)
            return (enabled,)

    node = translate_node("Toggle", V3Toggle, CompatTranslation())
    spec = node.schema().inputs[0]
    assert spec.type == TypeExpr.concrete(CORE_BOOLEAN)
    assert spec.default is False
    assert spec.widget == BooleanWidget(label_on="enable", label_off="disable")
    V3Toggle.seen.clear()
    assert node.execute(enabled=True) == {"string": "enable"}
    assert V3Toggle.seen == ["enable"]


def test_v3_unsupported_markers_skip_without_poisoning_other_nodes() -> None:
    def marker_node(marker: str, config: dict[str, object]):
        class MarkerNode:
            RETURN_TYPES = ()
            FUNCTION = "run"

            @classmethod
            def INPUT_TYPES(cls):  # noqa: ANN206
                return {"required": {"dynamic": (marker, config)}}

            def run(self, dynamic):  # noqa: ANN001, ANN201
                return ()

        return MarkerNode

    translation = translate_mappings(
        {
            "Good": V1MultiOut,
            "Match": marker_node(
                "COMFY_MATCHTYPE_V3",
                {"template": {"template_id": "value", "allowed_types": "INT"}},
            ),
            "Autogrow": marker_node("COMFY_AUTOGROW_V3", _autogrow_config(min_members=0)),
            "DynamicCombo": marker_node("COMFY_DYNAMICCOMBO_V3", {"options": []}),
            "DynamicSlot": marker_node(
                "COMFY_DYNAMICSLOT_V3",
                {"slotType": "choice", "inputs": {}, "forceInput": True},
            ),
        }
    )
    assert [node.schema().node_type for node in translation.node_classes] == [
        "comfy.Good",
        "comfy.Match",
        "comfy.Autogrow",
        "comfy.DynamicSlot",
    ]
    assert "required but has zero options" in translation.skipped["DynamicCombo"]
    slot_node = translation.node_classes[-1]
    assert isinstance(slot_node.schema().slots[0], DynamicSlotSpec)
    assert not any(type_id.startswith("comfy.COMFY_") for type_id in translation.opaque_types)


def _dynamic_combo_config(
    options: list[tuple[str, Mapping[str, object]]], *, default: str | None = None
) -> dict[str, object]:
    config: dict[str, object] = {
        "options": [{"key": key, "inputs": inputs} for key, inputs in options]
    }
    if default is not None:
        config["default"] = default
    return config


def test_v3_dynamic_combo_recursive_translation_and_zero_option_matrix() -> None:
    nested_match = {"template": {"template_id": "shared", "allowed_types": "INT,STRING"}}
    config = _dynamic_combo_config(
        [
            (
                "outer",
                {
                    "required": {
                        "sub": (
                            "COMFY_DYNAMICCOMBO_V3",
                            _dynamic_combo_config(
                                [
                                    (
                                        "inner",
                                        {
                                            "required": {
                                                "value": (
                                                    "COMFY_MATCHTYPE_V3",
                                                    nested_match,
                                                )
                                            }
                                        },
                                    )
                                ]
                            ),
                        )
                    }
                },
            )
        ],
        default="outer",
    )

    class RecursiveCombo:
        RETURN_TYPES = ("COMFY_MATCHTYPE_V3",)
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls) -> object:
            return {"required": {"combo": ("COMFY_DYNAMICCOMBO_V3", config)}}

        @classmethod
        def GET_NODE_INFO_V1(cls):  # noqa: N802, ANN206
            return {"output_matchtypes": ["shared"]}

        def run(self, combo):  # noqa: ANN001, ANN201
            return (combo["sub"]["value"],)

    node = translate_node("RecursiveCombo", RecursiveCombo, CompatTranslation())
    combo = node.schema().combos[0]
    assert isinstance(combo, DynamicComboSpec)
    assert combo.default == "outer"
    nested = combo.options[0].inputs[0]
    assert isinstance(nested, DynamicComboSpec)
    leaf = nested.options[0].inputs[0]
    assert isinstance(leaf, InputSpec)
    assert leaf.type == TypeExpr.variable("shared", (CORE_INT, CORE_STRING))
    assert node.schema().outputs[0].type == leaf.type

    class RequiredEmpty(RecursiveCombo):
        @classmethod
        def INPUT_TYPES(cls) -> object:
            return {"required": {"combo": ("COMFY_DYNAMICCOMBO_V3", {"options": []})}}

    class OptionalEmpty(RequiredEmpty):
        RETURN_TYPES = ()

        @classmethod
        def INPUT_TYPES(cls) -> object:
            return {"optional": {"combo": ("COMFY_DYNAMICCOMBO_V3", {"options": []})}}

    translation = translate_mappings(
        {"RequiredEmpty": RequiredEmpty, "OptionalEmpty": OptionalEmpty}
    )
    assert "required but has zero options" in translation.skipped["RequiredEmpty"]
    optional = translation.node_classes[0].schema().combos[0]
    assert optional.options == () and optional.required is False


def test_v3_combo_in_combo_executes_with_construct_scoped_selector_submission() -> None:
    config = _dynamic_combo_config(
        [
            (
                "option4",
                {
                    "required": {
                        "subcombo": (
                            "COMFY_DYNAMICCOMBO_V3",
                            _dynamic_combo_config(
                                [("opt1", {"required": {"float_x": ("FLOAT", {})}})]
                            ),
                        )
                    }
                },
            )
        ]
    )

    class DCTestShape:
        RETURN_TYPES = ("FLOAT",)
        FUNCTION = "run"
        seen: list[dict[str, object]] = []

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"combo": ("COMFY_DYNAMICCOMBO_V3", config)}}

        @classmethod
        def run(cls, combo):  # noqa: ANN001, ANN206
            cls.seen.append(combo)
            return (combo["subcombo"]["float_x"],)

    async def scenario() -> None:
        DCTestShape.seen.clear()
        engine = compat_engine(translate_mappings({"DCTestShape": DCTestShape}))
        graph = Graph(
            nodes={
                "n": GraphNode(
                    "comfy.DCTestShape",
                    {"combo.subcombo.float_x": 1.25},
                    slot_variants={
                        "combo": "option4",
                        "combo.subcombo": "opt1",
                    },
                )
            }
        )
        result = await engine.run(graph, ["n"])
        assert result.outputs["n"]["float"].resolve() == 1.25
        assert DCTestShape.seen == [
            {
                "combo": "option4",
                "subcombo": {"subcombo": "opt1", "float_x": 1.25},
            }
        ]

    asyncio.run(scenario())


def test_v3_family_in_combo_and_empty_family_lower_upstream_names() -> None:
    family = _autogrow_config(min_members=0, max_members=3)
    config = _dynamic_combo_config(
        [("batch", {"required": {"frame": ("COMFY_AUTOGROW_V3", family)}})]
    )

    class FamilyCombo:
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"
        seen: list[dict[str, object]] = []

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"mode": ("COMFY_DYNAMICCOMBO_V3", config)}}

        @classmethod
        def run(cls, mode):  # noqa: ANN001, ANN206
            cls.seen.append(mode)
            return (len(mode["frame"]),)

    node = translate_node("FamilyCombo", FamilyCombo, CompatTranslation())
    assert node.execute(mode="batch", **{"mode.frame.second": 2, "mode.frame.first": 1}) == {
        "int": 2
    }
    assert FamilyCombo.seen[-1] == {
        "mode": "batch",
        "frame": {"item_0": 2, "item_1": 1},
    }
    assert node.execute(mode="batch") == {"int": 0}
    assert FamilyCombo.seen[-1]["frame"] == {}


def test_v3_nested_list_boolean_family_restores_upstream_vocabulary() -> None:
    family = _autogrow_config(["enable", "disable"], min_members=0)
    combo = _dynamic_combo_config(
        [
            ("batch", {"required": {"flags": ("COMFY_AUTOGROW_V3", family)}}),
            (
                "alternate",
                {
                    "required": {
                        "flags": (
                            "COMFY_AUTOGROW_V3",
                            _autogrow_config(["yes", "no"], min_members=0),
                        )
                    }
                },
            ),
        ]
    )

    class ListedFlags:
        RETURN_TYPES = ()
        FUNCTION = "run"
        INPUT_IS_LIST = True
        seen: list[dict[str, object]] = []

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"mode": ("COMFY_DYNAMICCOMBO_V3", combo)}}

        @classmethod
        def run(cls, mode):  # noqa: ANN001, ANN206
            cls.seen.append(mode)
            return ()

    node = translate_node("ListedFlags", ListedFlags, CompatTranslation())
    family_spec = node.schema().combos[0].options[0].inputs[0]
    assert isinstance(family_spec, InputFamilySpec)
    assert family_spec.type == TypeExpr.list_of(TypeExpr.concrete(CORE_BOOLEAN))
    node.execute(mode="batch", **{"mode.flags.first": [True, False]})
    assert ListedFlags.seen == [
        {
            "mode": ["batch"],
            "flags": {"item_0": ["enable", "disable"]},
        }
    ]


def test_v3_top_level_list_boolean_family_restores_upstream_vocabulary() -> None:
    family = _autogrow_config(["enable", "disable"], min_members=0)

    class ListedFlags:
        RETURN_TYPES = ()
        FUNCTION = "run"
        INPUT_IS_LIST = True
        seen: list[dict[str, object]] = []

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"flags": ("COMFY_AUTOGROW_V3", family)}}

        @classmethod
        def run(cls, flags):  # noqa: ANN001, ANN206
            cls.seen.append(flags)
            return ()

    node = translate_node("ListedFlags", ListedFlags, CompatTranslation())
    node.execute(flags={"first": [True, False]})
    assert ListedFlags.seen == [{"item_0": ["enable", "disable"]}]


def test_v3_nested_resident_inputs_mark_gpu_occupancy() -> None:
    combo = _dynamic_combo_config([("load", {"required": {"model": ("MODEL", {})}})])

    class NestedResident:
        RETURN_TYPES = ()
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"mode": ("COMFY_DYNAMICCOMBO_V3", combo)}}

        @classmethod
        def run(cls, mode):  # noqa: ANN001, ANN206
            return ()

    schema = translate_node("NestedResident", NestedResident, CompatTranslation()).schema()
    assert schema.occupies == ("gpu",)

    class OpenSlotResident(NestedResident):
        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "optional": {
                    "source": (
                        "COMFY_DYNAMICSLOT_V3",
                        {"slotType": "MODEL", "inputs": {}},
                    )
                }
            }

    slot_schema = translate_node("OpenSlotResident", OpenSlotResident, CompatTranslation()).schema()
    assert slot_schema.occupies == ("gpu",)


def test_failed_nodes_rollback_opaque_types_and_composite_markers() -> None:
    class FailsAfterOpaque:
        RETURN_TYPES = ()
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "opaque": ("FAILED_OPAQUE", {}),
                    "mode": ("COMFY_DYNAMICCOMBO_V3", {"options": []}),
                }
            }

        def run(self, **kwargs):  # noqa: ANN003, ANN201
            return ()

    class CompositeMarker(FailsAfterOpaque):
        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"value": ("INT,COMFY_DYNAMICSLOT_V3", {})}}

    translation = CompatTranslation()
    translation.opaque_types.add("comfy.KEEP")
    result = translate_mappings(
        {"FailsAfterOpaque": FailsAfterOpaque, "CompositeMarker": CompositeMarker},
        translation=translation,
    )
    assert set(result.skipped) == {"FailsAfterOpaque", "CompositeMarker"}
    assert result.opaque_types == {"comfy.KEEP"}
    assert "dynamic type marker" in result.skipped["CompositeMarker"]

    class MatchAllowedMarker(FailsAfterOpaque):
        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {
                    "value": (
                        "COMFY_MATCHTYPE_V3",
                        {
                            "template": {
                                "template_id": "value",
                                "allowed_types": "INT,COMFY_DYNAMICCOMBO_V3",
                            }
                        },
                    )
                }
            }

    marker_result = translate_mappings({"MatchAllowedMarker": MatchAllowedMarker})
    assert "dynamic marker" in marker_result.skipped["MatchAllowedMarker"]
    assert not any("COMFY_" in atom for atom in marker_result.opaque_types)

    with pytest.raises(CompatError, match="required but has zero options"):
        translate_mappings(
            {"FailsAfterOpaque": FailsAfterOpaque},
            only=["FailsAfterOpaque"],
            translation=translation,
        )
    assert translation.opaque_types == {"comfy.KEEP"}


def test_v3_dynamic_slot_maps_open_form_and_lowers_dependents() -> None:
    class SlotNode:
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"
        seen: list[dict[str, object] | None] = []

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "optional": {
                    "source": (
                        "COMFY_DYNAMICSLOT_V3",
                        {
                            "slotType": "INT",
                            "inputs": {"required": {"scale": ("INT", {})}},
                            "forceInput": True,
                        },
                    )
                }
            }

        @classmethod
        def run(cls, source=None):  # noqa: ANN001, ANN206
            cls.seen.append(source)
            if source is None:
                return (0,)
            return (source["source"] * source["scale"],)

    node = translate_node("SlotNode", SlotNode, CompatTranslation())
    slot = node.schema().slots[0]
    assert slot.slot_type == TypeExpr.concrete(CORE_INT)
    assert slot.required is False and slot.force_input is True

    async def scenario() -> None:
        SlotNode.seen.clear()
        engine = compat_engine(translate_mappings({"SlotNode": SlotNode}))
        graph = Graph(
            nodes={
                "n": GraphNode(
                    "comfy.SlotNode",
                    {"source": 3, "source.scale": 4},
                )
            }
        )
        result = await engine.run(graph, ["n"])
        assert result.outputs["n"]["int"].resolve() == 12
        assert SlotNode.seen == [{"source": 3, "scale": 4}]

        absent = await engine.run(
            Graph(nodes={"absent": GraphNode("comfy.SlotNode", {})}),
            ["absent"],
        )
        assert absent.outputs["absent"]["int"].resolve() == 0
        assert SlotNode.seen == [{"source": 3, "scale": 4}, None]

    asyncio.run(scenario())


def test_v3_list_calling_convention_reaches_dynamic_nested_values() -> None:
    combo = _dynamic_combo_config([("selected", {"required": {"value": ("INT", {})}})])

    class ListedDynamics:
        RETURN_TYPES = ()
        FUNCTION = "run"
        INPUT_IS_LIST = True
        seen: list[tuple[object, object]] = []

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {
                "required": {"mode": ("COMFY_DYNAMICCOMBO_V3", combo)},
                "optional": {
                    "source": (
                        "COMFY_DYNAMICSLOT_V3",
                        {"slotType": "STRING", "inputs": {}},
                    )
                },
            }

        @classmethod
        def run(cls, mode, source):  # noqa: ANN001, ANN206
            cls.seen.append((mode, source))
            return ()

    node = translate_node("ListedDynamics", ListedDynamics, CompatTranslation())
    assert node.schema().slots[0].slot_type == TypeExpr.list_of(TypeExpr.concrete(CORE_STRING))
    node.execute(
        mode="selected",
        source=["socket"],
        **{"mode.value": [3]},
    )
    assert ListedDynamics.seen == [
        (
            {"mode": ["selected"], "value": [3]},
            {"source": ["socket"]},
        )
    ]


def test_v3_names_family_subset_min_and_identifier_failures_are_loud() -> None:
    config = _autogrow_config(min_members=1)
    template = config["template"]
    assert isinstance(template, dict)
    template.pop("prefix")
    template.pop("max")
    template["names"] = ["left", "right"]

    class Names:
        RETURN_TYPES = ()
        FUNCTION = "run"
        seen: list[dict[str, object]] = []

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"items": ("COMFY_AUTOGROW_V3", config)}}

        @classmethod
        def run(cls, items):  # noqa: ANN001, ANN206
            cls.seen.append(items)
            return ()

    schema = translate_node("Names", Names, CompatTranslation()).schema()
    assert [item.id for item in elaborate(schema, ["items.right"]).inputs] == ["items.right"]
    with pytest.raises(ElaborationError, match="needs >= 1"):
        elaborate(schema, [])
    Names.seen.clear()
    translate_node("NamesRun", Names, CompatTranslation()).execute(items={"right": 2})
    assert Names.seen == [{"right": 2}]
    translate_node("NamesOrdered", Names, CompatTranslation()).execute(
        items={"right": 2, "left": 1}
    )
    assert list(Names.seen[-1].items()) == [("left", 1), ("right", 2)]

    combo = _dynamic_combo_config(
        [("chosen", {"required": {"items": ("COMFY_AUTOGROW_V3", config)}})]
    )

    class NestedNames:
        RETURN_TYPES = ()
        FUNCTION = "run"
        seen: list[dict[str, object]] = []

        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"mode": ("COMFY_DYNAMICCOMBO_V3", combo)}}

        @classmethod
        def run(cls, mode):  # noqa: ANN001, ANN206
            cls.seen.append(mode["items"])
            return ()

    NestedNames.seen.clear()
    translate_node("NestedNames", NestedNames, CompatTranslation()).execute(
        mode="chosen",
        **{"mode.items.right": 2, "mode.items.left": 1},
    )
    assert list(NestedNames.seen[-1].items()) == [("left", 1), ("right", 2)]

    bad = _dynamic_combo_config([("bad.\u00e9", {"required": {}})])

    class BadKey(Names):
        @classmethod
        def INPUT_TYPES(cls):  # noqa: ANN206
            return {"required": {"mode": ("COMFY_DYNAMICCOMBO_V3", bad)}}

    translation = translate_mappings({"BadKey": BadKey})
    assert "must match [!-~]+( [!-~]+)*" in translation.skipped["BadKey"]


def test_v3_grouped_autogrow_lazy_and_rawlink_stay_classified_skips() -> None:
    grouped = _autogrow_config()
    template = grouped["template"]
    assert isinstance(template, dict)
    template["input"] = {"required": {"left": ("INT", {}), "right": ("INT", {})}}

    class Grouped:
        RETURN_TYPES = ()
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls) -> object:
            return {"required": {"items": ("COMFY_AUTOGROW_V3", grouped)}}

        def run(self, items):  # noqa: ANN001, ANN201
            return ()

    class Lazy(Grouped):
        @classmethod
        def INPUT_TYPES(cls) -> object:
            return {"required": {"value": ("INT", {"lazy": True})}}

    class RawLink(Grouped):
        @classmethod
        def INPUT_TYPES(cls) -> object:
            return {"required": {"value": ("INT", {"rawLink": True})}}

    class AcceptAll(Grouped):
        _ACCEPT_ALL_INPUTS = True

        @classmethod
        def INPUT_TYPES(cls) -> object:
            return {"required": {"value": ("INT", {})}}

    translation = translate_mappings(
        {
            "Grouped": Grouped,
            "Lazy": Lazy,
            "RawLink": RawLink,
            "AcceptAll": AcceptAll,
        }
    )
    assert "expected exactly one nested input" in translation.skipped["Grouped"]
    assert "lazy inputs require check_lazy_status" in translation.skipped["Lazy"]
    assert "unsupported rawLink semantics" in translation.skipped["RawLink"]
    assert "accept_all_inputs" in translation.skipped["AcceptAll"]
    assert translation.diagnostics["Lazy"].code == "compat.lazy.missing-hook"
    assert translation.diagnostics["Lazy"].lazy is True
    assert translation.diagnostics["RawLink"].code == "compat.raw-link.unsupported"
    assert translation.diagnostics["RawLink"].input_path == ("value",)
    assert translation.diagnostics["RawLink"].raw_link is True
    assert translation.diagnostics["AcceptAll"].code == "compat.accept-all.unsupported"
    assert translation.diagnostics["AcceptAll"].accept_all is True


def test_compat_gate_diagnostic_matrix_preserves_exact_source_facts_and_reasons() -> None:
    class V1Dynamic:
        RETURN_TYPES = ()
        FUNCTION = "run"

        @classmethod
        def INPUT_TYPES(cls) -> object:
            return {"required": {"value": ("COMFY_FUTURE_V3", {})}}

        def run(self, value):  # noqa: ANN001, ANN201
            return ()

    class V3Dynamic(V1Dynamic):
        INPUT_IS_LIST = 0
        OUTPUT_IS_LIST = (False, 0)

        @classmethod
        def GET_NODE_INFO_V1(cls) -> object:  # noqa: N802
            return {"name": "V3Dynamic"}

    class MalformedLazy(V1Dynamic):
        @classmethod
        def INPUT_TYPES(cls) -> object:
            return {"required": {"value": ("INT", {"lazy": 1})}}

    class MalformedOutputList(V1Dynamic):
        OUTPUT_IS_LIST = "false"

        @classmethod
        def INPUT_TYPES(cls) -> object:
            return {"required": {"value": ("INT", {})}}

        RETURN_TYPES = ("INT",)

    class RootDynamicCombo(V1Dynamic):
        @classmethod
        def INPUT_TYPES(cls) -> object:
            return {"required": {"mode": ("COMFY_DYNAMICCOMBO_V3", {"options": []})}}

    class RaisingDescriptor:
        def __get__(self, instance: object, owner: type | None = None) -> object:
            raise RuntimeError("OUTPUT_IS_LIST descriptor must not run after refusal")

    class DescriptorAfterRefusal(V1Dynamic):
        OUTPUT_IS_LIST = RaisingDescriptor()

    translation = translate_mappings(
        {
            "V1Dynamic": V1Dynamic,
            "V3Dynamic": V3Dynamic,
            "MalformedLazy": MalformedLazy,
            "MalformedOutputList": MalformedOutputList,
            "RootDynamicCombo": RootDynamicCombo,
            "DescriptorAfterRefusal": DescriptorAfterRefusal,
        }
    )

    v1 = translation.diagnostics["V1Dynamic"]
    assert v1.reason == "V1Dynamic: unsupported V3 dynamic input kind COMFY_FUTURE_V3"
    assert (v1.code, v1.source_generation, v1.path_kind) == (
        "compat.dynamic.unsupported",
        "v1",
        "declared",
    )
    assert (v1.input_id, v1.input_path) == ("value", ("value",))
    assert (v1.input_is_list, v1.output_is_list, v1.accept_all) == (False, False, False)
    assert (v1.lazy, v1.raw_link) == (False, False)

    v3 = translation.diagnostics["V3Dynamic"]
    assert v3.source_generation == "v3"
    assert v3.input_is_list is None
    assert v3.output_is_list is None

    malformed_lazy = translation.diagnostics["MalformedLazy"]
    assert malformed_lazy.reason == "MalformedLazy: input 'value' lazy must be a Boolean"
    assert malformed_lazy.code == "compat.lazy.malformed"
    assert malformed_lazy.lazy is None
    assert malformed_lazy.raw_link is False
    assert malformed_lazy.input_path == ("value",)

    output_list = translation.diagnostics["MalformedOutputList"]
    assert output_list.reason == "MalformedOutputList: OUTPUT_IS_LIST must be a tuple"
    assert output_list.code == "compat.output-list.malformed"
    assert output_list.output_is_list is None

    root_dynamic = translation.diagnostics["RootDynamicCombo"]
    assert root_dynamic.reason == "V3 DynamicCombo input 'mode' is required but has zero options"
    assert root_dynamic.input_path == ("mode",)
    assert root_dynamic.path_kind == "declared"

    descriptor = translation.diagnostics["DescriptorAfterRefusal"]
    assert descriptor.reason == (
        "DescriptorAfterRefusal: unsupported V3 dynamic input kind COMFY_FUTURE_V3"
    )
    assert descriptor.output_is_list is None
