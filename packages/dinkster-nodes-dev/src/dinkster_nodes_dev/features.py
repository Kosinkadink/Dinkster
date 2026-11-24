"""Small executable examples of optional node presentation contracts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import cast

import numpy as np
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    CURVE_TYPE,
    HISTOGRAM_TYPE,
    IMAGE_TYPE,
    PNG_CONTAINER_VERSION,
    BooleanWidget,
    Choice,
    ComboWidget,
    Count,
    Curve,
    CurveWidget,
    CustomWidgetDescriptor,
    Described,
    Formula,
    Group,
    Input,
    Item,
    Members,
    MirrorSpec,
    MirrorTolerance,
    MultiComboWidget,
    Node,
    NodeSchema,
    Numbered,
    NumberWidget,
    Option,
    Output,
    Record,
    Section,
    SlotValue,
    StringWidget,
    TypeChoice,
    TypeExpr,
    TypeRegistry,
    Variant,
    WidgetRepresentation,
    WidgetRepresentations,
    decode_image_array,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    image_input,
    render_image_png,
    validate_image_encoded,
)

STRING = TypeExpr.concrete(CORE_STRING)
INT = TypeExpr.concrete(CORE_INT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
STRINGS = TypeExpr.list_of(STRING)
IMAGE = TypeExpr.concrete(IMAGE_TYPE)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
CURVE = TypeExpr.concrete(CURVE_TYPE)
HISTOGRAM = TypeExpr.concrete(HISTOGRAM_TYPE)


def register_feature_types(registry: TypeRegistry) -> None:
    """Use the shared portable IMAGE contract without requiring inference."""
    if IMAGE_TYPE not in registry:
        registry.register(
            IMAGE_TYPE,
            encode=encode_image_array,
            decode=decode_image_array,
            fingerprint=image_array_fingerprint(IMAGE_TYPE),
            meta=image_array_meta,
            input_convert=image_input,
            validate_encoded=validate_image_encoded,
            validate_encoded_buffer=validate_image_encoded,
            asset_references_complete=True,
        )
        registry.register_rendition(
            IMAGE_TYPE,
            "png",
            mime="image/png",
            render=render_image_png,
            version=PNG_CONTAINER_VERSION,
        )


class FeatureSlot(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.features.slot",
            display_name="Feature: Dynamic Slot",
            category="dev/features",
            output_node=True,
            description="Specialize the source as text or an integer, then inspect its echo.",
            body=(
                TypeChoice(
                    "source",
                    variants=(Variant("text", STRING), Variant("integer", INT)),
                ),
                Output("text", STRING),
            ),
        )

    @classmethod
    def execute(cls, *, source: SlotValue) -> Mapping[str, object]:
        return cls.outputs(text=f"{source.variant}: {source.value}")


class FeatureOutputCount(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.features.output_count",
            display_name="Feature: Output Count",
            category="dev/features",
            output_node=True,
            description="Change count to grow or shrink outputs containing successive integers.",
            body=(
                Input("count", INT, default=3, widget=NumberWidget(min=1, max=8)),
                Input("start", INT, default=10),
                Group(
                    "numbers",
                    value=Output("number", INT),
                    keys=Numbered(""),
                    member_label=Formula('"number " + str(index() + 1)'),
                    grows=Count(Formula("count")),
                    min_members=1,
                    max_members=8,
                ),
            ),
        )

    @classmethod
    def execute(cls, *, count: int, start: int, numbers: Members[Item]) -> Mapping[str, object]:
        return {"numbers": {member.key: start + int(member.key) for member in numbers}}


class FeatureConditionalWidgets(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.features.conditional",
            display_name="Feature: Conditional Widgets",
            category="dev/features",
            output_node=True,
            description="Toggle details to hide or show the note without discarding its value.",
            body=(
                Input(
                    "details",
                    BOOLEAN,
                    default=True,
                    widget=BooleanWidget(label_on="Show details", label_off="Hide details"),
                ),
                Input(
                    "note",
                    STRING,
                    default="Keep this note",
                    widget=StringWidget(),
                    visible=Formula("details"),
                ),
                Output("text", STRING),
            ),
        )

    @classmethod
    def execute(cls, *, details: bool, note: str) -> Mapping[str, object]:
        return cls.outputs(text=f"details={details}; note={note}")


class FeatureWidgetKinds(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.features.widget_kinds",
            display_name="Feature: Widget Kinds",
            category="dev/features",
            output_node=True,
            description="Edit selections, Markdown, colors and a list; open help with a button.",
            body=(
                Input(
                    "selections",
                    TypeExpr.list_of(TypeExpr.concrete(CORE_COMBO)),
                    default=["red", "blue"],
                    widget=MultiComboWidget(options=("red", "green", "blue")),
                ),
                Input(
                    "markdown",
                    STRING,
                    default="# Try Markdown\n\n**Edit me.**",
                    widget=CustomWidgetDescriptor("MARKDOWN"),
                ),
                Input(
                    "colors",
                    STRINGS,
                    default=["#ff6600", "#2255aa"],
                    widget=CustomWidgetDescriptor("COLORS"),
                ),
                Input(
                    "button",
                    STRING,
                    default="",
                    widget=CustomWidgetDescriptor(
                        "BUTTON", {"display_name": "Open node help", "command": "node.help"}
                    ),
                ),
                Input("items", STRINGS, default=["first", "second"], widget=StringWidget()),
                Output("text", STRING),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        selections: Sequence[str],
        markdown: str,
        colors: Sequence[str],
        button: str,
        items: Sequence[str],
    ) -> Mapping[str, object]:
        return cls.outputs(
            text=json.dumps(
                {
                    "selections": list(selections),
                    "markdown": markdown,
                    "colors": list(colors),
                    "button": button,
                    "items": list(items),
                },
                sort_keys=True,
            )
        )


class FeatureRepresentations(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.features.representations",
            display_name="Feature: Widget Representations",
            category="dev/features",
            output_node=True,
            description="Switch the same stored text between line and multiline editors.",
            body=(
                Input(
                    "text",
                    STRING,
                    default="One value, two views",
                    widget=WidgetRepresentations(
                        (
                            WidgetRepresentation(
                                "line", StringWidget(multiline=False), "Single line"
                            ),
                            WidgetRepresentation(
                                "multiline", StringWidget(multiline=True), "Multiline"
                            ),
                        ),
                        default="line",
                        user_switchable=True,
                    ),
                ),
                Output("text", STRING),
            ),
        )

    @classmethod
    def execute(cls, *, text: str) -> Mapping[str, object]:
        return cls.outputs(text=text)


class FeatureOutputDescriptors(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.features.output_descriptors",
            display_name="Feature: Output Descriptors",
            category="dev/features",
            output_node=True,
            description="Each output listed in the description returns its row number.",
            body=(Described("entries", kinds=(Output("integer", INT),), max_entries=8),),
        )

    @classmethod
    def execute(cls, *, entries: Mapping[str, object]) -> Mapping[str, object]:
        return {"entries": {entry: index for index, entry in enumerate(entries, start=1)}}


class FeatureImageSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.features.image_source",
            display_name="Feature: Mirror Image Source",
            category="dev/features",
            description="A small RGB gradient supplies browser-resident imagery after running.",
            body=(Output("image", IMAGE, preview=True),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        x = np.linspace(0.0, 1.0, 96, dtype=np.float32)
        y = np.linspace(0.0, 1.0, 64, dtype=np.float32)
        image = np.empty((1, 64, 96, 3), dtype=np.float32)
        image[0, :, :, 0] = x
        image[0, :, :, 1] = y[:, None]
        image[0, :, :, 2] = 0.25
        return cls.outputs(image=image)


class FeatureMirror(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.features.mirror",
            display_name="Feature: Mirror Estimate",
            category="dev/features",
            output_node=True,
            description="Change gain for a browser estimate; run for authoritative pixels.",
            body=(
                Input("image", IMAGE),
                Input(
                    "gain",
                    FLOAT,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=2.0, step=0.05, display="slider"),
                ),
                Output("image", IMAGE, preview=True),
            ),
            mirror=MirrorSpec(
                kind="glsl",
                precision="bounded",
                tolerance=MirrorTolerance(per_channel=1.0 / 255.0),
                source="""#version 300 es
precision highp float;
uniform sampler2D u_image;
uniform float gain;
out vec4 color;
void main() {
    vec4 pixel = texelFetch(u_image, ivec2(gl_FragCoord.xy), 0);
    color = vec4(clamp(pixel.rgb * gain, 0.0, 1.0), pixel.a);
}
""",
            ),
        )

    @classmethod
    def execute(cls, *, image: np.ndarray, gain: float) -> Mapping[str, object]:
        if image.ndim not in (3, 4) or image.shape[-1] not in (3, 4):
            raise ValueError("the mirror example needs RGB or RGBA image data")
        result = image.copy()
        result[..., :3] = np.clip(image[..., :3] * gain, 0.0, 1.0)
        return cls.outputs(image=result)


class FeatureLayers(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.features.layers",
            display_name="Feature: Layers",
            category="dev/features",
            output_node=True,
            description=(
                "Add, reorder and remove layers inside a section that closes; each layer "
                "has a name and a strength."
            ),
            body=(
                Input("title", STRING, default="Layers", widget=StringWidget(multiline=False)),
                Section(
                    "layers_section",
                    label="Layers",
                    rows=(
                        Group(
                            "layers",
                            member=(
                                Input(
                                    "name",
                                    STRING,
                                    default="layer",
                                    widget=StringWidget(multiline=False),
                                ),
                                Input(
                                    "strength",
                                    FLOAT,
                                    default=1.0,
                                    widget=NumberWidget(min=0.0, max=2.0, step=0.05),
                                ),
                            ),
                            keys=Numbered(""),
                            member_label=Formula('"layer " + str(index() + 1)'),
                            grows="button",
                            max_members=8,
                            reorderable=True,
                            add_label="Add layer",
                            label="Layers",
                        ),
                    ),
                ),
                Output("text", STRING),
            ),
        )

    @classmethod
    def execute(cls, *, title: str, layers: Members[Record]) -> Mapping[str, object]:
        parts = [f"{layer['name']} {_number(layer['strength']):.2f}" for layer in layers]
        return cls.outputs(text=f"{title}: " + ", ".join(parts))


class FeatureOptionFields(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.features.option_fields",
            display_name="Feature: Fields Under an Option",
            category="dev/features",
            output_node=True,
            description=(
                "A field under a choice's option and a field under a type-led variant; "
                "each field's output gives nothing while its option or variant is not chosen."
            ),
            body=(
                Choice(
                    "mode",
                    label="Mode",
                    options=(
                        Option(
                            "fixed",
                            label="Fixed amount",
                            rows=(
                                Input(
                                    "amount",
                                    FLOAT,
                                    default=0.8,
                                    widget=NumberWidget(min=0.0, max=1.0, step=0.05),
                                ),
                            ),
                        ),
                        Option("none", label="No amount"),
                    ),
                    default="fixed",
                ),
                TypeChoice(
                    "source",
                    variants=(
                        Variant(
                            "number",
                            FLOAT,
                            rows=(
                                Input(
                                    "scale",
                                    FLOAT,
                                    default=1.5,
                                    widget=NumberWidget(min=0.0, max=4.0, step=0.25),
                                ),
                            ),
                        ),
                        Variant("text", STRING),
                    ),
                    required=False,
                ),
                Output("text", STRING),
            ),
        )

    @classmethod
    def execute(cls, *, mode: Record, source: SlotValue | None) -> Mapping[str, object]:
        parts = [f"mode={mode.key}"]
        if "amount" in mode:
            parts.append(f"amount={_number(mode['amount']):.2f}")
        if source is None:
            parts.append("source=none")
        else:
            parts.append(f"source={source.variant}:{source.value}")
            if "scale" in source.options:
                parts.append(f"scale={_number(source.options['scale']):.2f}")
        return cls.outputs(text="; ".join(parts))


class FeatureRequiredValue(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.features.required_value",
            display_name="Feature: Required Value",
            category="dev/features",
            output_node=True,
            description=(
                "One required number with no default; the node is skipped when its wire "
                "gives nothing."
            ),
            body=(Input("value", FLOAT), Output("text", STRING)),
        )

    @classmethod
    def execute(cls, *, value: float) -> Mapping[str, object]:
        return cls.outputs(text=f"value={value:.2f}")


_HISTOGRAM_SHAPES: dict[str, Callable[[int], int]] = {
    "rising": lambda index: index * 4,
    "falling": lambda index: (255 - index) * 4,
    "middle": lambda index: max(0, 1024 - abs(index - 128) * 12),
}


class FeatureHistogramShape(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.features.histogram_shape",
            display_name="Feature: Histogram Shape",
            category="dev/features",
            description="A 256-bin histogram of a chosen shape, to feed a curve's histogram.",
            body=(
                Input(
                    "shape",
                    TypeExpr.concrete(CORE_COMBO),
                    default="rising",
                    widget=ComboWidget(options=tuple(_HISTOGRAM_SHAPES)),
                ),
                Output("histogram", HISTOGRAM),
            ),
        )

    @classmethod
    def execute(cls, *, shape: str) -> Mapping[str, object]:
        return cls.outputs(histogram=[_HISTOGRAM_SHAPES[shape](index) for index in range(256)])


class FeatureCurveHistograms(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.features.curve_histograms",
            display_name="Feature: Curves With Histograms",
            category="dev/features",
            output_node=True,
            description="A group of curves; each curve has its own histogram input.",
            body=(
                Group(
                    "curves",
                    member=(
                        Input(
                            "curve",
                            CURVE,
                            default={
                                "points": [[0.0, 0.0], [1.0, 1.0]],
                                "interpolation": "monotone_cubic",
                            },
                            widget=CurveWidget(histogram="histogram"),
                        ),
                        Input("histogram", HISTOGRAM, required=False),
                    ),
                    keys=Numbered(""),
                    member_label=Formula('"curve " + str(index() + 1)'),
                    grows="button",
                    min_members=2,
                    max_members=4,
                    add_label="Add curve",
                    label="Curves",
                ),
                Output("text", STRING),
            ),
        )

    @classmethod
    def execute(cls, *, curves: Members[Record]) -> Mapping[str, object]:
        parts: list[str] = []
        for member in curves:
            curve = cast("Curve", member["curve"])
            histogram = cast("Sequence[int] | None", member.get("histogram"))
            bins = "no histogram" if histogram is None else f"{len(histogram)} bins"
            parts.append(f"{member.key}: {len(curve.points)} points, {bins}")
        return cls.outputs(text="; ".join(parts))


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


FEATURE_NODES = [
    FeatureSlot,
    FeatureOutputCount,
    FeatureConditionalWidgets,
    FeatureWidgetKinds,
    FeatureRepresentations,
    FeatureOutputDescriptors,
    FeatureImageSource,
    FeatureMirror,
    FeatureLayers,
    FeatureOptionFields,
    FeatureRequiredValue,
    FeatureHistogramShape,
    FeatureCurveHistograms,
]
