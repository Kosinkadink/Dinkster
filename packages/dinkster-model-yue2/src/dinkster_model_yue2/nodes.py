"""YuE2 workflow nodes."""

from __future__ import annotations

from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
)

CLIP = TypeExpr.concrete("dinkster.clip")
CONDITIONING = TypeExpr.concrete("dinkster.conditioning")
LATENT = TypeExpr.concrete("dinkster.latent")
AUDIO = TypeExpr.concrete("dinkster.audio")
VAE = TypeExpr.concrete("dinkster.vae")
STRING = TypeExpr.concrete(CORE_STRING)
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
COMBO = TypeExpr.concrete(CORE_COMBO)


def _text(name: str, *, default: str = "") -> InputSpec:
    return InputSpec(name, STRING, default=default, widget=StringWidget(multiline=True))


class YuE2GenerateABC(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.yue2_generate_abc",
            display_name="YuE2 Generate ABC",
            category="model/conditioning/yue2",
            inputs=(
                InputSpec("clip", CLIP),
                _text("style"),
                _text("lyrics"),
                InputSpec("seed", INT, default=0, widget=NumberWidget(min=0)),
                InputSpec(
                    "mode", COMBO, default="full", widget=ComboWidget(options=("full", "melody"))
                ),
                InputSpec(
                    "max_abc_tokens", INT, default=8192, widget=NumberWidget(min=1, max=20000)
                ),
                InputSpec(
                    "temperature",
                    FLOAT,
                    default=0.7,
                    widget=NumberWidget(min=0.0, max=5.0, step=0.05),
                ),
                InputSpec(
                    "top_p", FLOAT, default=0.9, widget=NumberWidget(min=0.01, max=1.0, step=0.01)
                ),
                InputSpec("top_k", INT, default=30, widget=NumberWidget(min=1, max=32768)),
                InputSpec(
                    "repetition_penalty",
                    FLOAT,
                    default=1.005,
                    widget=NumberWidget(min=0.01, max=10.0, step=0.005),
                ),
                InputSpec(
                    "penalty_window", INT, default=100, widget=NumberWidget(min=1, max=20000)
                ),
            ),
            outputs=(OutputSpec("abc", STRING),),
        )

    @classmethod
    def execute(cls, **inputs: object) -> Mapping[str, object]:
        from .provider import execute_generate_abc

        return execute_generate_abc(**inputs)


class YuE2GenerateMusic(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.yue2_generate_music",
            display_name="YuE2 Generate Music",
            category="model/conditioning/yue2",
            inputs=(
                InputSpec("clip", CLIP),
                _text("style"),
                _text("lyrics"),
                _text("abc"),
                InputSpec("seed", INT, default=0, widget=NumberWidget(min=0)),
                InputSpec(
                    "mode", COMBO, default="full", widget=ComboWidget(options=("full", "melody"))
                ),
                InputSpec(
                    "max_duration",
                    FLOAT,
                    default=360.0,
                    widget=NumberWidget(min=0.04, max=900.0, step=0.04),
                ),
                InputSpec(
                    "temperature",
                    FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=5.0, step=0.05),
                ),
                InputSpec(
                    "top_p", FLOAT, default=0.95, widget=NumberWidget(min=0.01, max=1.0, step=0.01)
                ),
                InputSpec("top_k", INT, default=100, widget=NumberWidget(min=1, max=32768)),
                InputSpec(
                    "repetition_penalty",
                    FLOAT,
                    default=1.2,
                    widget=NumberWidget(min=0.01, max=10.0, step=0.01),
                ),
                InputSpec(
                    "cfg_scale",
                    FLOAT,
                    required=False,
                    default=None,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("conditioning", CONDITIONING), OutputSpec("seconds", FLOAT)),
        )

    @classmethod
    def execute(cls, **inputs: object) -> Mapping[str, object]:
        from .provider import execute_generate_music

        return execute_generate_music(**inputs)


class EmptyYuE2LatentAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.empty_yue2_latent_audio",
            display_name="Empty YuE2 Latent Audio",
            category="model/latent/yue2",
            inputs=(
                InputSpec(
                    "seconds",
                    FLOAT,
                    default=120.0,
                    widget=NumberWidget(min=0.04, max=1000.0, step=0.04),
                ),
                InputSpec("batch_size", INT, default=1, widget=NumberWidget(min=1, max=4096)),
            ),
            outputs=(OutputSpec("latent", LATENT),),
        )

    @classmethod
    def execute(cls, seconds: float, batch_size: int) -> Mapping[str, object]:
        from .provider import execute_empty_latent

        return execute_empty_latent(seconds=seconds, batch_size=batch_size)


class YuE2DecodeAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.yue2_decode_audio",
            display_name="YuE2 Decode Audio",
            category="model/latent/yue2",
            inputs=(InputSpec("samples", LATENT), InputSpec("vae", VAE)),
            outputs=(OutputSpec("audio", AUDIO),),
        )

    @classmethod
    def execute(cls, *, samples: object, vae: object) -> Mapping[str, object]:
        from .provider import execute_decode_audio

        return execute_decode_audio(samples=samples, vae=vae)


YUE2_MODEL_NODES: tuple[type[Node], ...] = (
    YuE2GenerateABC,
    YuE2GenerateMusic,
    EmptyYuE2LatentAudio,
    YuE2DecodeAudio,
)
YUE2_MODEL_NODE_IDS = tuple(node.schema().node_type for node in YUE2_MODEL_NODES)

__all__ = ["YUE2_MODEL_NODE_IDS", "YUE2_MODEL_NODES"]
