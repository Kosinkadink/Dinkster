"""Fields that declare ``automatic``: a value the host works out from a
formula or a lookup until the user fixes one."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping

from dinkster_api.v1 import (
    ASSET_TYPE,
    CLIP_TYPE,
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    MODEL_TYPE,
    AssetRef,
    AssetWidget,
    Button,
    Formula,
    Group,
    Input,
    Lookup,
    Node,
    NodeSchema,
    Numbered,
    NumberWidget,
    Output,
    Row,
    SetEach,
    TypeExpr,
    lookup,
)

FLOAT = TypeExpr.concrete(CORE_FLOAT)
INT = TypeExpr.concrete(CORE_INT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)


class AutomaticFrameRate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.automatic.frame_rate",
            display_name="Automatic: Frame Rate",
            category="dev/automatic",
            output_node=True,
            description=(
                "Frame rate follows rate / divisor until a value is typed. "
                "A divisor of 0 cannot be worked out, so the default, 24, is used."
            ),
            body=(
                Input("rate", FLOAT, default=30.0, widget=NumberWidget(min=0, max=1000)),
                Input("divisor", INT, default=1, widget=NumberWidget(min=0, max=100)),
                Input(
                    "frame_rate",
                    FLOAT,
                    default=24.0,
                    automatic=Formula("rate / divisor"),
                    widget=NumberWidget(min=1, max=240),
                ),
                Output("fps", FLOAT),
            ),
        )

    @classmethod
    def execute(cls, *, rate: float, divisor: int, frame_rate: float) -> Mapping[str, object]:
        return cls.outputs(fps=frame_rate)


@lookup
def dev_source_rate(rate: float, delay: float) -> float:
    """A quarter of ``rate`` after ``delay`` seconds; a negative rate fails."""
    time.sleep(delay)
    if rate < 0:
        raise ValueError("a negative rate has no source rate")
    return rate / 4


class AutomaticLookupRate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.automatic.lookup_rate",
            display_name="Automatic: Looked-up Frame Rate",
            category="dev/automatic",
            output_node=True,
            description=(
                "Frame rate is a lookup's answer, a quarter of rate. A negative rate "
                "makes the lookup fail; a delay over 10 seconds runs out of time. "
                "Either gives the default, 24."
            ),
            body=(
                Input("rate", FLOAT, default=96.0, widget=NumberWidget(min=-1000, max=1000)),
                Input("delay", FLOAT, default=0.0, widget=NumberWidget(min=0, max=30)),
                Input(
                    "frame_rate",
                    FLOAT,
                    default=24.0,
                    automatic=Lookup(dev_source_rate, args=("rate", "delay")),
                    widget=NumberWidget(min=1, max=240),
                ),
                Output("fps", FLOAT),
            ),
        )

    @classmethod
    def execute(cls, *, rate: float, delay: float, frame_rate: float) -> Mapping[str, object]:
        return cls.outputs(fps=frame_rate)


@lookup
def dev_video_rate(video: AssetRef) -> float:
    """The frame rate a video's file name gives as ``<rate>fps``, else 30."""
    found = re.search(r"(\d+(?:\.\d+)?)fps", video.name)
    return float(found.group(1)) if found else 30.0


class AutomaticVideoRate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.automatic.video_rate",
            display_name="Automatic: Frame Rate from a Video",
            category="dev/automatic",
            output_node=True,
            description=(
                "Frame rate is a lookup's answer for the picked video: the rate its "
                "file name gives as <rate>fps, else 30. With no video picked it is "
                "the default, 24."
            ),
            body=(
                Input(
                    "video",
                    TypeExpr.concrete(ASSET_TYPE),
                    required=False,
                    widget=AssetWidget(accept=("video/mp4", "video/webm"), kind="media/video"),
                ),
                Input(
                    "frame_rate",
                    FLOAT,
                    default=24.0,
                    automatic=Lookup(dev_video_rate, args=("video",)),
                    widget=NumberWidget(min=1, max=240),
                ),
                Output("fps", FLOAT),
            ),
        )

    @classmethod
    def execute(cls, *, frame_rate: float, video: object = None) -> Mapping[str, object]:
        return cls.outputs(fps=frame_rate)


class AutomaticLoraStack(Node):
    """The node master plan's example 18: each lora's clip strength follows
    its strength until the user separates them and fixes one."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "dev.automatic.lora_stack",
            display_name="Automatic: Lora Stack",
            category="dev/automatic",
            description="Clip strength follows strength until a value is typed.",
            body=(
                Input("model", TypeExpr.concrete(MODEL_TYPE), required=False),
                Input("clip", TypeExpr.concrete(CLIP_TYPE), required=False),
                Input(
                    "separate_clip",
                    BOOLEAN,
                    default=False,
                    label="Separate model and clip strength",
                ),
                Group(
                    "loras",
                    grows="button",
                    keys=Numbered("lora_", start=1),
                    member=(
                        Row(
                            rows=(
                                Input("on", BOOLEAN, default=True),
                                Input(
                                    "lora",
                                    TypeExpr.concrete(ASSET_TYPE),
                                    pin=False,
                                    widget=AssetWidget(kind="model/lora"),
                                ),
                                Input(
                                    "strength",
                                    FLOAT,
                                    default=1.0,
                                    pin=False,
                                    label=Formula('"Model" if separate_clip else "Strength"'),
                                    widget=NumberWidget(step=0.05, round=0.01),
                                ),
                                Input(
                                    "clip_strength",
                                    FLOAT,
                                    default=1.0,
                                    automatic=Formula("strength"),
                                    pin=False,
                                    visible=Formula("separate_clip"),
                                    label="Clip",
                                    widget=NumberWidget(step=0.05, round=0.01),
                                ),
                            )
                        ),
                    ),
                ),
                Button(
                    "toggle_all",
                    "Toggle all",
                    steps=(SetEach("loras", "on", Formula("not all(loras.on)")),),
                ),
                Output("MODEL", TypeExpr.concrete(MODEL_TYPE)),
                Output("CLIP", TypeExpr.concrete(CLIP_TYPE)),
            ),
        )

    @classmethod
    def execute(cls, **values: object) -> Mapping[str, object]:
        return cls.outputs(MODEL=values.get("model"), CLIP=values.get("clip"))


AUTOMATIC_NODES = [
    AutomaticFrameRate,
    AutomaticLookupRate,
    AutomaticVideoRate,
    AutomaticLoraStack,
]
