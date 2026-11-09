"""Native ACE Step audio text conditioning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
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

from .conditioning import conditioning_set_values

_CATEGORY = "model/conditioning/ace"
_CLIP = TypeExpr.concrete(CLIP_TYPE)
_CONDITIONING = TypeExpr.concrete(CONDITIONING_TYPE)
_BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
_COMBO = TypeExpr.concrete(CORE_COMBO)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_INT = TypeExpr.concrete(CORE_INT)
_STRING = TypeExpr.concrete(CORE_STRING)
_LANGUAGES = (
    "ar",
    "az",
    "bg",
    "bn",
    "ca",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "id",
    "is",
    "it",
    "ja",
    "ko",
    "la",
    "lt",
    "ms",
    "ne",
    "nl",
    "no",
    "pa",
    "pl",
    "pt",
    "ro",
    "ru",
    "sa",
    "sk",
    "sr",
    "sv",
    "sw",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "uk",
    "ur",
    "vi",
    "yue",
    "zh",
    "unknown",
)
_KEYS = (
    "C",
    "C#",
    "Db",
    "D",
    "D#",
    "Eb",
    "E",
    "F",
    "F#",
    "Gb",
    "G",
    "G#",
    "Ab",
    "A",
    "A#",
    "Bb",
    "B",
)
_KEYSCALES = tuple(f"{root} {quality}" for quality in ("major", "minor") for root in _KEYS)


def _text(name: str) -> InputSpec:
    return InputSpec(name, _STRING, widget=StringWidget(multiline=True, dynamic_prompts=True))


class TextEncodeAceStepAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TextEncodeAceStepAudio",
            category=_CATEGORY,
            inputs=(
                InputSpec("clip", _CLIP),
                _text("tags"),
                _text("lyrics"),
                InputSpec(
                    "lyrics_strength",
                    _FLOAT,
                    default=1.0,
                    widget=NumberWidget(min=0.0, max=10.0, step=0.01),
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls, *, clip: Any, tags: str, lyrics: str, lyrics_strength: float = 1.0
    ) -> Mapping[str, object]:
        initialize_inference_process()
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(tags, lyrics=lyrics))
        return cls.outputs(
            CONDITIONING=conditioning_set_values(conditioning, {"lyrics_strength": lyrics_strength})
        )


class TextEncodeAceStepAudio15(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="TextEncodeAceStepAudio1.5",
            category=_CATEGORY,
            inputs=(
                InputSpec("clip", _CLIP),
                _text("tags"),
                _text("lyrics"),
                InputSpec(
                    "seed",
                    _INT,
                    default=0,
                    widget=NumberWidget(
                        min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate="fixed"
                    ),
                ),
                InputSpec("bpm", _INT, default=120, widget=NumberWidget(min=10, max=300)),
                InputSpec(
                    "duration",
                    _FLOAT,
                    default=120.0,
                    widget=NumberWidget(min=0.0, max=2000.0, step=0.1),
                ),
                InputSpec(
                    "timesignature",
                    _COMBO,
                    default="2",
                    widget=ComboWidget(("2", "3", "4", "6")),
                ),
                InputSpec("language", _COMBO, default="en", widget=ComboWidget(_LANGUAGES)),
                InputSpec(
                    "keyscale",
                    _COMBO,
                    default="C major",
                    widget=ComboWidget(_KEYSCALES),
                ),
                InputSpec(
                    "generate_audio_codes",
                    _BOOLEAN,
                    default=True,
                    advanced=True,
                    doc=(
                        "Enable the LLM that generates audio codes. This can be slow but will "
                        "increase the quality of the generated audio. Turn this off if you are "
                        "giving the model an audio reference."
                    ),
                ),
                InputSpec(
                    "cfg_scale",
                    _FLOAT,
                    default=2.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=100.0, step=0.1),
                ),
                InputSpec(
                    "temperature",
                    _FLOAT,
                    default=0.85,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=2.0, step=0.01),
                ),
                InputSpec(
                    "top_p",
                    _FLOAT,
                    default=0.9,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=2000.0, step=0.01),
                ),
                InputSpec(
                    "top_k",
                    _INT,
                    default=0,
                    advanced=True,
                    widget=NumberWidget(min=0, max=100),
                ),
                InputSpec(
                    "min_p",
                    _FLOAT,
                    default=0.0,
                    advanced=True,
                    widget=NumberWidget(min=0.0, max=1.0, step=0.001),
                ),
            ),
            outputs=(OutputSpec("CONDITIONING", _CONDITIONING),),
        )

    @classmethod
    def execute(
        cls,
        *,
        clip: Any,
        tags: str,
        lyrics: str,
        seed: int = 0,
        bpm: int = 120,
        duration: float = 120.0,
        timesignature: str = "2",
        language: str = "en",
        keyscale: str = "C major",
        generate_audio_codes: bool = True,
        cfg_scale: float = 2.0,
        temperature: float = 0.85,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
    ) -> Mapping[str, object]:
        initialize_inference_process()
        tokens = clip.tokenize(
            tags,
            lyrics=lyrics,
            bpm=bpm,
            duration=duration,
            timesignature=int(timesignature),
            language=language,
            keyscale=keyscale,
            seed=seed,
            generate_audio_codes=generate_audio_codes,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        return cls.outputs(CONDITIONING=clip.encode_from_tokens_scheduled(tokens))


ACE_TEXT_CONDITIONING_NODES = (TextEncodeAceStepAudio, TextEncodeAceStepAudio15)
