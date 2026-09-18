"""Torch-free Ideogram 4 profile, layout, text, and detection parity."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT32,
    IDEOGRAM4,
    IDEOGRAM4_CONFIG,
    IDEOGRAM4_SIGMAS,
    IDEOGRAM4_TAP_LAYERS,
    IDEOGRAM4_TEXT_CONFIG,
    NATIVE_WIRED_FAMILY_IDS,
    FlowSigmas,
    Ideogram4Config,
    Ideogram4DetectError,
    Ideogram4PromptTokens,
    Ideogram4TextConfig,
    Ideogram4TextDetectError,
    Parameterization,
    TensorGeometry,
    WeightEntry,
    builtin_families,
    builtin_family_registry,
    detect_ideogram4,
    detect_ideogram4_config,
    detect_ideogram4_text_config,
    format_ideogram4_prompt,
    ideogram4_language_layout,
    ideogram4_layout,
    ideogram4_text_layout,
    tokenize_ideogram4_prompt,
)

GOLDENS = json.loads(
    (
        Path(__file__).parent.parent
        / "packages"
        / "dinkster-inference-torch"
        / "tests"
        / "goldens"
        / "ideogram4_goldens.json"
    ).read_text()
)


class HeaderSource:
    def __init__(self, shapes: Mapping[str, tuple[int, ...]]) -> None:
        self.shapes = dict(shapes)

    def keys(self) -> Sequence[str]:
        return tuple(self.shapes)

    def entry(self, key: str) -> WeightEntry:
        geometry = TensorGeometry(self.shapes[key], BFLOAT16)
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}


def diffusion_geometries() -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, BFLOAT16) for key, shape in ideogram4_layout().items()}


def text_geometries() -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, BFLOAT16) for key, shape in ideogram4_text_layout().items()}


def test_config_pins_the_published_profile() -> None:
    config = IDEOGRAM4_CONFIG
    assert config == Ideogram4Config()
    assert (
        config.hidden_size,
        config.layers,
        config.attention_heads,
        config.attention_head_dim,
        config.intermediate_size,
        config.adaln_dim,
    ) == (4608, 34, 18, 256, 12288, 512)
    assert (config.latent_channels, config.ae_channels, config.patch) == (128, 32, (2, 2))
    assert config.text_width == 53248
    assert config.rope_theta == 5_000_000.0
    assert config.rope_dims == (24, 20, 20)
    assert config.memory_factor == 11.6
    assert config.inference_dtypes == (BFLOAT16, FLOAT32)
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if isinstance(value, str):
            replacement = value + "x"
        elif isinstance(value, tuple):
            replacement = (*value, value[-1])
        else:
            replacement = value + 1
        with pytest.raises(ValueError, match="published Ideogram 4"):
            dataclasses.replace(config, **{field.name: replacement})


def test_diffusion_layout_matches_executed_reference() -> None:
    actual = sorted((key, list(shape)) for key, shape in ideogram4_layout().items())
    assert len(actual) == 458
    assert actual == [(key, list(shape)) for key, shape in GOLDENS["layout"]]
    assert detect_ideogram4_config(diffusion_geometries()) is IDEOGRAM4_CONFIG
    with pytest.raises(TypeError):
        ideogram4_layout()["input_proj.weight"] = (1,)  # type: ignore[index]


def test_diffusion_detection_is_strict_and_prefix_aware() -> None:
    geometries = diffusion_geometries()
    geometries["input_proj.weight"] = TensorGeometry((4608, 127), BFLOAT16)
    with pytest.raises(Ideogram4DetectError, match="input_proj.weight: expected shape"):
        detect_ideogram4_config(geometries)
    geometries = diffusion_geometries()
    del geometries["layers.33.feed_forward.w3.weight"]
    with pytest.raises(Ideogram4DetectError, match="missing layers.33"):
        detect_ideogram4_config(geometries)
    geometries = diffusion_geometries()
    geometries["unexpected.weight"] = TensorGeometry((1,), BFLOAT16)
    with pytest.raises(Ideogram4DetectError, match="unexpected key"):
        detect_ideogram4_config(geometries)

    shapes = {key: shape for key, shape in ideogram4_layout().items()}
    bare = detect_ideogram4(HeaderSource(shapes))
    assert bare is not None and bare.key_prefix == ""
    prefixed = detect_ideogram4(
        HeaderSource({f"model.diffusion_model.{key}": shape for key, shape in shapes.items()})
    )
    assert prefixed is not None and prefixed.key_prefix == "model.diffusion_model."
    assert detect_ideogram4(HeaderSource({"foreign.weight": (4, 4)})) is None


def test_text_profile_and_layout_match_executed_reference() -> None:
    config = IDEOGRAM4_TEXT_CONFIG
    assert config == Ideogram4TextConfig()
    assert (
        config.vocab_size,
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    ) == (151936, 4096, 12288, 36, 32, 8, 128)
    assert config.tap_layers == IDEOGRAM4_TAP_LAYERS
    assert config.stack_width == 53248
    assert sorted((key, list(shape)) for key, shape in ideogram4_text_layout().items()) == [
        (key, list(shape)) for key, shape in GOLDENS["text_layout"]
    ]
    assert len(ideogram4_text_layout()) == 750
    assert len(ideogram4_language_layout()) == 398
    assert detect_ideogram4_text_config(text_geometries()) is config

    bad = text_geometries()
    bad["model.layers.35.mlp.down_proj.weight"] = TensorGeometry((4095, 12288), BFLOAT16)
    with pytest.raises(Ideogram4TextDetectError, match="expected shape"):
        detect_ideogram4_text_config(bad)
    missing = text_geometries()
    del missing["model.visual.deepstack_merger_list.2.linear_fc2.weight"]
    with pytest.raises(Ideogram4TextDetectError, match="missing model.visual"):
        detect_ideogram4_text_config(missing)


def test_prompt_template_and_tokenizer_match_executed_reference() -> None:
    formatted = format_ideogram4_prompt("paint a red fox")
    assert formatted.text == (
        "<|im_start|>user\npaint a red fox<|im_end|>\n<|im_start|>assistant\n"
    )
    assert formatted.preformatted is False
    raw = "<|im_start|>literal"
    assert format_ideogram4_prompt(raw).text == raw
    assert format_ideogram4_prompt(raw).preformatted is True
    for case in GOLDENS["tokenizer"]:
        tokens = tokenize_ideogram4_prompt(case["text"])
        assert list(tokens.ids) == case["ids"]
        assert list(tokens.attention_mask) == case["attention_mask"]
        assert set(case["weights"]) == {1.0}


def test_prompt_tokens_refuse_malformed_values() -> None:
    with pytest.raises(ValueError, match="non-empty and equal"):
        Ideogram4PromptTokens((), ())
    with pytest.raises(ValueError, match="non-negative"):
        Ideogram4PromptTokens((1, -1), (1, 1))
    with pytest.raises(ValueError, match="binary"):
        Ideogram4PromptTokens((1, 2), (1, 2))


def test_family_catalog_pins_sampling_latent_dtype_and_wiring() -> None:
    assert IDEOGRAM4.id == "dinkster.ideogram4"
    assert IDEOGRAM4.id in {family.id for family in builtin_families()}
    assert IDEOGRAM4.id in builtin_family_registry().ids()
    assert IDEOGRAM4.id not in NATIVE_WIRED_FAMILY_IDS
    assert IDEOGRAM4.sampling.parameterization is Parameterization.FLOW
    assert IDEOGRAM4_SIGMAS == FlowSigmas(shift=1.0)
    assert IDEOGRAM4.sampling.sigma_min == 0.001
    assert IDEOGRAM4.sampling.sigma_max == 1.0
    assert IDEOGRAM4.single_stream_latent().channels == 128
    assert IDEOGRAM4.single_stream_latent().spatial_downscale == 16
    assert IDEOGRAM4.supported_dtypes == frozenset({BFLOAT16, FLOAT32})
    assert IDEOGRAM4.memory_factor == 11.6
    assert IDEOGRAM4.wiring.text_encoders == ("dinkster.qwen3vl_8b",)
    assert math.isclose(IDEOGRAM4.sampling.shift, 1.0)
