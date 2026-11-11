"""Small executable examples of optional node presentation contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    IMAGE_TYPE,
    PNG_CONTAINER_VERSION,
    BooleanWidget,
    ConditionalWidgetGroup,
    CustomWidgetDescriptor,
    DynamicSlotSpec,
    InputSpec,
    MirrorSpec,
    MirrorTolerance,
    MultiComboWidget,
    Node,
    NodeSchema,
    NumberWidget,
    OutputCountSpec,
    OutputDescriptorsSpec,
    OutputFamilySpec,
    OutputInterface,
    OutputSpec,
    SlotValue,
    SlotVariant,
    StringWidget,
    TypeExpr,
    TypeRegistry,
    WidgetRepresentation,
    WidgetRepresentations,
    decode_image_array,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    image_input,
    output_descriptor_entries,
    render_image_png,
    validate_image_encoded,
)

STRING = TypeExpr.concrete(CORE_STRING)
INT = TypeExpr.concrete(CORE_INT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
STRINGS = TypeExpr.list_of(STRING)
IMAGE = TypeExpr.concrete(IMAGE_TYPE)
FLOAT = TypeExpr.concrete(CORE_FLOAT)


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
            slots=(
                DynamicSlotSpec(
                    "source",
                    variants=(
                        SlotVariant("text", STRING),
                        SlotVariant("integer", INT),
                    ),
                ),
            ),
            outputs=(OutputSpec("text", STRING),),
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
            inputs=(
                InputSpec("count", INT, default=3, widget=NumberWidget(min=1, max=8)),
                InputSpec("start", INT, default=10),
            ),
            output_families=(
                OutputFamilySpec("numbers", INT, 1, 8, count=OutputCountSpec("count")),
            ),
        )

    @classmethod
    def execute(
        cls, *, count: int, start: int, output_spec: OutputInterface
    ) -> Mapping[str, object]:
        return cls.outputs(
            numbers={suffix: start + int(suffix) for suffix in output_spec.family("numbers")}
        )


class FeatureConditionalWidgets(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.features.conditional",
            display_name="Feature: Conditional Widgets",
            category="dev/features",
            output_node=True,
            description="Toggle details to hide or show the note without discarding its value.",
            inputs=(
                InputSpec(
                    "details",
                    BOOLEAN,
                    default=True,
                    widget=BooleanWidget(label_on="Show details", label_off="Hide details"),
                ),
                InputSpec("note", STRING, default="Keep this note", widget=StringWidget()),
            ),
            widget_groups=(ConditionalWidgetGroup("details", (True,), ("note",)),),
            outputs=(OutputSpec("text", STRING),),
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
            inputs=(
                InputSpec(
                    "selections",
                    TypeExpr.list_of(TypeExpr.concrete(CORE_COMBO)),
                    default=["red", "blue"],
                    widget=MultiComboWidget(options=("red", "green", "blue")),
                ),
                InputSpec(
                    "markdown",
                    STRING,
                    default="# Try Markdown\n\n**Edit me.**",
                    widget=CustomWidgetDescriptor("MARKDOWN"),
                ),
                InputSpec(
                    "colors",
                    STRINGS,
                    default=["#ff6600", "#2255aa"],
                    widget=CustomWidgetDescriptor("COLORS"),
                ),
                InputSpec(
                    "button",
                    STRING,
                    default="",
                    widget=CustomWidgetDescriptor(
                        "BUTTON", {"display_name": "Open node help", "command": "node.help"}
                    ),
                ),
                InputSpec("items", STRINGS, default=["first", "second"], widget=StringWidget()),
            ),
            outputs=(OutputSpec("text", STRING),),
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
            inputs=(
                InputSpec(
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
            ),
            outputs=(OutputSpec("text", STRING),),
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
            description="Edit outputs: integers return row numbers; text returns names.",
            inputs=(
                InputSpec(
                    "entries",
                    STRING,
                    default='{"entries":[{"id":"answer","name":"Answer","type":"integer"}]}',
                    widget=StringWidget(multiline=True),
                ),
            ),
            output_descriptors=OutputDescriptorsSpec(
                "entries",
                (
                    OutputSpec("integer", INT),
                    OutputSpec("text", STRING),
                ),
                8,
            ),
        )

    @classmethod
    def execute(cls, *, entries: str, output_spec: OutputInterface) -> Mapping[str, object]:
        return {
            output.id: index if entry["type"] == "integer" else entry["name"]
            for index, (output, entry) in enumerate(
                zip(output_spec.outputs, output_descriptor_entries(entries), strict=True), start=1
            )
        }


class FeatureImageSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.features.image_source",
            display_name="Feature: Mirror Image Source",
            category="dev/features",
            description="A small RGB gradient supplies browser-resident imagery after running.",
            outputs=(OutputSpec("image", IMAGE, preview=True),),
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
            inputs=(
                InputSpec("image", IMAGE),
                InputSpec(
                    "gain",
                    FLOAT,
                    default=0.5,
                    widget=NumberWidget(min=0.0, max=2.0, step=0.05, display="slider"),
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True),),
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


FEATURE_NODES = [
    FeatureSlot,
    FeatureOutputCount,
    FeatureConditionalWidgets,
    FeatureWidgetKinds,
    FeatureRepresentations,
    FeatureOutputDescriptors,
    FeatureImageSource,
    FeatureMirror,
]
