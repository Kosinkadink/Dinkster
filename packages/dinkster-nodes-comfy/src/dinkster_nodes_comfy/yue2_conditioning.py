"""Native YuE2 ABC and music conditioning."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
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
    initialize_inference_process,
)

_CATEGORY = "model/conditioning/yue2"
_CLIP = TypeExpr.concrete(CLIP_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_STRING = TypeExpr.concrete(CORE_STRING)
_MODES = ("full", "melody")
_MODE_DOC = (
    "full: generates melody and chords; melody: generates melody only, recommended for covers."
)


class YuE2GenerateABC(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="YuE2GenerateABC",
            display_name="YuE2 Generate ABC",
            category=_CATEGORY,
            description=(
                "Generates the ABC notation of a song from style and lyrics. Connect abc to YuE2 "
                "Generate Music node."
            ),
            search_terms=("yue2", "music", "abc", "notation"),
            inputs=(
                InputSpec("clip", _CLIP),
                InputSpec(
                    "style", _STRING, widget=StringWidget(multiline=True, dynamic_prompts=True)
                ),
                InputSpec(
                    "lyrics", _STRING, widget=StringWidget(multiline=True, dynamic_prompts=True)
                ),
                InputSpec(
                    "seed",
                    _INT,
                    default=0,
                    widget=NumberWidget(
                        min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate="fixed"
                    ),
                ),
                InputSpec("mode", _COMBO, widget=ComboWidget(_MODES), doc=_MODE_DOC),
                InputSpec(
                    "max_abc_tokens",
                    _INT,
                    default=8192,
                    widget=NumberWidget(min=1, max=20000),
                    advanced=True,
                ),
                InputSpec(
                    "temperature",
                    _FLOAT,
                    default=0.7,
                    widget=NumberWidget(min=0.0, max=5.0, step=0.05),
                    advanced=True,
                ),
                InputSpec(
                    "top_p",
                    _FLOAT,
                    default=0.9,
                    widget=NumberWidget(min=0.01, max=1.0, step=0.01),
                    advanced=True,
                ),
                InputSpec(
                    "top_k",
                    _INT,
                    default=30,
                    widget=NumberWidget(min=1, max=32768),
                    advanced=True,
                ),
                InputSpec(
                    "repetition_penalty",
                    _FLOAT,
                    default=1.005,
                    widget=NumberWidget(min=0.01, max=10.0, step=0.005),
                    advanced=True,
                ),
                InputSpec(
                    "penalty_window",
                    _INT,
                    default=100,
                    widget=NumberWidget(min=1, max=20000),
                    advanced=True,
                    doc="Number of recent ABC tokens used to penalize repetition.",
                ),
            ),
            outputs=(OutputSpec("abc", _STRING, display_name="abc"),),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        style: str,
        lyrics: str,
        seed: int,
        mode: str,
        max_abc_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 30,
        repetition_penalty: float = 1.005,
        penalty_window: int = 100,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        tokens = clip.tokenize(
            style,
            lyrics=lyrics,
            cot=mode,
            seed=seed,
            max_tokens=max_abc_tokens,
            penalty_window=penalty_window,
        )
        generated = clip.generate(
            tokens,
            max_length=max_abc_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        return cls.outputs(abc=clip.decode(generated))


class YuE2GenerateMusic(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="YuE2GenerateMusic",
            display_name="YuE2 Generate Music",
            category=_CATEGORY,
            description=(
                "Generates music tokens and acoustic conditioning from style, lyrics, and an ABC "
                "notation. Provide the generated seconds to the Empty YuE2 Latent Audio node. "
                "An empty ABC input ignores the selected mode."
            ),
            search_terms=("yue2", "music", "audio", "abc"),
            inputs=(
                InputSpec("clip", _CLIP),
                InputSpec(
                    "style", _STRING, widget=StringWidget(multiline=True, dynamic_prompts=True)
                ),
                InputSpec(
                    "lyrics", _STRING, widget=StringWidget(multiline=True, dynamic_prompts=True)
                ),
                InputSpec(
                    "abc",
                    _STRING,
                    default="",
                    widget=StringWidget(multiline=True),
                    doc=(
                        "Connect the ABC generator or supply an edited score. Leave empty to use "
                        "off mode automatically."
                    ),
                ),
                InputSpec(
                    "seed",
                    _INT,
                    default=0,
                    widget=NumberWidget(
                        min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate="fixed"
                    ),
                ),
                InputSpec("mode", _COMBO, widget=ComboWidget(_MODES), doc=_MODE_DOC),
                InputSpec(
                    "max_duration",
                    _FLOAT,
                    default=360.0,
                    widget=NumberWidget(min=0.04, max=900.0, step=0.04),
                    doc=(
                        "Maximum duration in seconds. Automatically reduced for long prompts; "
                        "generation can stop earlier."
                    ),
                ),
                InputSpec(
                    "temperature",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=5.0, step=0.05),
                    advanced=True,
                ),
                InputSpec(
                    "top_p",
                    _FLOAT,
                    default=0.95,
                    widget=NumberWidget(min=0.01, max=1.0, step=0.01),
                    advanced=True,
                ),
                InputSpec(
                    "top_k",
                    _INT,
                    default=100,
                    widget=NumberWidget(min=1, max=32768),
                    advanced=True,
                ),
                InputSpec(
                    "repetition_penalty",
                    _FLOAT,
                    default=1.2,
                    widget=NumberWidget(min=0.01, max=10.0, step=0.01),
                    advanced=True,
                ),
                InputSpec(
                    "cfg_scale",
                    _FLOAT,
                    default=None,
                    required=False,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.01),
                    advanced=True,
                    doc=(
                        "Autoregressive guidance for style and lyrics. 1.0 disables CFG, "
                        "matching the ABC workflow. Use 1.01 to match the original off-mode "
                        "guidance."
                    ),
                ),
            ),
            outputs=(
                OutputSpec("CONDITIONING", _CONDITIONING),
                OutputSpec("seconds", _FLOAT, display_name="seconds"),
            ),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        style: str,
        lyrics: str,
        seed: int,
        mode: str,
        max_duration: float,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        abc: str = "",
        cfg_scale: float | None = None,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        yue2 = importlib.import_module("dinkster_inference.text_encoders.yue2")
        if not abc.strip():
            mode = "off"
        if cfg_scale is None:
            cfg_scale = 1.01 if mode == "off" else 1.0
        tokens = clip.tokenize(
            style,
            lyrics=lyrics,
            cot=mode,
            seed=seed,
            abc=abc,
            max_tokens=max(1, round(max_duration * yue2.FRAMES_PER_SECOND)),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            cfg_scale=cfg_scale,
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return cls.outputs(
            CONDITIONING=conditioning,
            seconds=conditioning[0][1]["yue2_frames"] / yue2.FRAMES_PER_SECOND,
        )


YUE2_CONDITIONING_NODES = (YuE2GenerateABC, YuE2GenerateMusic)
