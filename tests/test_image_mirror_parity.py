from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
from dinkster_nodes_image.adjust import (
    ADJUST_MIRROR_PER_CHANNEL_TOLERANCE,
    ADJUST_MIRROR_SOURCE,
    ADJUST_OPERATIONS,
    ImageAdjust,
)
from dinkster_nodes_image.filters import (
    FILTER_MIRROR_PER_CHANNEL_TOLERANCE,
    FILTER_MIRROR_SOURCE,
    ImageFilter,
)

from tools.generate_image_adjust_mirror_vector import build_frames, build_vector
from tools.generate_image_filter_mirror_vector import (
    EXPECTED_REGENERATION_ATOL,
)
from tools.generate_image_filter_mirror_vector import (
    build_frames as build_filter_frames,
)
from tools.generate_image_filter_mirror_vector import (
    build_vector as build_filter_vector,
)

VECTOR_PATH = Path(__file__).parent / "fixtures" / "mirror-parity" / "image_adjust_v1.json"
FILTER_VECTOR_PATH = Path(__file__).parent / "fixtures" / "mirror-parity" / "image_filter_v1.json"


def _array(record: dict[str, Any]) -> np.ndarray:
    return np.asarray(cast("list[float]", record["values"]), dtype=np.float32).reshape(
        cast("list[int]", record["shape"])
    )


def test_adjust_mirror_vector_is_current_and_executable() -> None:
    vector = cast("dict[str, Any]", json.loads(VECTOR_PATH.read_text(encoding="utf-8")))
    # The corpus uses only correctly rounded binary32 arithmetic, so
    # regeneration must be bit-stable on every platform.
    assert vector == build_vector()
    assert vector["node_type"] == ImageAdjust.schema().node_type
    frames = build_frames()
    for name, record in cast("dict[str, dict[str, Any]]", vector["frames"]).items():
        np.testing.assert_array_equal(frames[name], _array(record), strict=True)
    for case in cast("list[dict[str, Any]]", vector["cases"]):
        result = ImageAdjust.execute(
            image=frames[cast("str", case["frame"])],
            **cast("dict[str, Any]", case["inputs"]),
        )
        actual = cast("np.ndarray", result["image"])
        np.testing.assert_array_equal(
            actual,
            _array(cast("dict[str, Any]", case["expected"])),
            err_msg=cast("str", case["id"]),
            strict=True,
        )


def test_adjust_mirror_declaration_matches_shader_contract() -> None:
    schema = ImageAdjust.schema()
    mirror = schema.mirror
    assert mirror is not None
    assert mirror.kind == "glsl"
    assert mirror.precision == "bounded"
    assert mirror.tolerance is not None
    assert mirror.tolerance.per_channel == ADJUST_MIRROR_PER_CHANNEL_TOLERANCE
    assert mirror.source == ADJUST_MIRROR_SOURCE

    vector = cast("dict[str, Any]", json.loads(VECTOR_PATH.read_text(encoding="utf-8")))
    assert vector["mirror_per_channel_tolerance"] == ADJUST_MIRROR_PER_CHANNEL_TOLERANCE

    lines = ADJUST_MIRROR_SOURCE.splitlines()
    assert lines[0] == "#version 300 es"
    # GLSL ES 3.00 predeclares samplers as lowp and texel reads inherit the
    # sampler's precision, so parity requires an explicit highp sampler.
    assert "precision highp sampler2D;" in lines
    # The shader's uniforms cover the static image input, dynamic selector,
    # and the union of inputs materialized by its active option.
    declared = {
        match.group(2): match.group(1)
        for line in lines
        if (match := re.fullmatch(r"uniform (\w+) (\w+);", line)) is not None
    }
    assert declared == {
        "u_image": "sampler2D",
        "operation": "int",
        "factor": "float",
        "mean": "float",
        "standard_deviation": "float",
    }
    assert tuple(spec.id for spec in schema.inputs) == ("image",)
    assert len(schema.combos) == 1
    operation = schema.combos[0]
    assert operation.id == "operation"
    assert tuple(option.key for option in operation.options) == ADJUST_OPERATIONS
    assert {spec.id for option in operation.options for spec in option.inputs} == {
        "factor",
        "mean",
        "standard_deviation",
    }
    assert "texelFetch(u_image, ivec2(gl_FragCoord.xy), 0)" in ADJUST_MIRROR_SOURCE


def _without_expected(vector: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(vector)
    stripped["cases"] = [
        {key: value for key, value in cast("dict[str, Any]", case).items() if key != "expected"}
        for case in cast("list[dict[str, Any]]", vector["cases"])
    ]
    return stripped


def test_filter_mirror_vector_is_current_and_executable() -> None:
    vector = cast("dict[str, Any]", json.loads(FILTER_VECTOR_PATH.read_text(encoding="utf-8")))
    regenerated = build_filter_vector()
    # Unlike the adjust corpus, expected outputs are not bit-stable under
    # regeneration across platforms (np.exp and einsum reduction order move
    # by ulps), so they are compared within the generator's documented drift
    # bound; everything else must match exactly.
    assert _without_expected(vector) == _without_expected(regenerated)
    committed_cases = cast("list[dict[str, Any]]", vector["cases"])
    regenerated_cases = cast("list[dict[str, Any]]", regenerated["cases"])
    for committed_case, regenerated_case in zip(committed_cases, regenerated_cases, strict=True):
        committed = _array(cast("dict[str, Any]", committed_case["expected"]))
        current = _array(cast("dict[str, Any]", regenerated_case["expected"]))
        assert committed.shape == current.shape, committed_case["id"]
        np.testing.assert_allclose(
            current,
            committed,
            rtol=0.0,
            atol=EXPECTED_REGENERATION_ATOL,
            err_msg=cast("str", committed_case["id"]),
            strict=True,
        )
    assert vector["node_type"] == ImageFilter.schema().node_type
    frames = build_filter_frames()
    for name, record in cast("dict[str, dict[str, Any]]", vector["frames"]).items():
        np.testing.assert_array_equal(frames[name], _array(record), strict=True)


def test_filter_mirror_declaration_matches_shader_contract() -> None:
    schema = ImageFilter.schema()
    mirror = schema.mirror
    assert mirror is not None
    assert mirror.kind == "glsl"
    assert mirror.precision == "bounded"
    assert mirror.tolerance is not None
    assert mirror.tolerance.per_channel == FILTER_MIRROR_PER_CHANNEL_TOLERANCE
    assert mirror.source == FILTER_MIRROR_SOURCE
    assert mirror.applies is not None
    assert dict(mirror.applies) == {"operation": ("gaussian_blur", "sharpen")}

    vector = cast("dict[str, Any]", json.loads(FILTER_VECTOR_PATH.read_text(encoding="utf-8")))
    assert vector["mirror_per_channel_tolerance"] == FILTER_MIRROR_PER_CHANNEL_TOLERANCE
    # Every corpus case exercises an operation inside the applies scope.
    for case in cast("list[dict[str, Any]]", vector["cases"]):
        operation_value = cast("dict[str, Any]", case["inputs"])["operation"]
        assert operation_value in mirror.applies["operation"], case["id"]

    lines = FILTER_MIRROR_SOURCE.splitlines()
    assert lines[0] == "#version 300 es"
    # GLSL ES 3.00 predeclares samplers as lowp and texel reads inherit the
    # sampler's precision, so parity requires an explicit highp sampler.
    assert "precision highp sampler2D;" in lines
    declared = {
        match.group(2): match.group(1)
        for line in lines
        if (match := re.fullmatch(r"uniform (\w+) (\w+);", line)) is not None
    }
    assert declared == {
        "u_image": "sampler2D",
        "operation": "int",
        "radius": "int",
        "sigma": "float",
        "strength": "float",
    }
    assert tuple(spec.id for spec in schema.inputs) == ("image",)
    assert len(schema.combos) == 1
    operation = schema.combos[0]
    assert operation.id == "operation"
    # applies keys name declared combos and applies values name declared
    # options, in declared order (the shader's operation branches index that
    # order within the applicable subset).
    option_keys = tuple(option.key for option in operation.options)
    applicable = mirror.applies["operation"]
    assert applicable == option_keys[: len(applicable)]
    # The shader's non-image uniforms cover exactly the selector plus the
    # union of inputs materialized by the applicable options.
    applicable_inputs = {
        spec.id
        for option in operation.options
        if option.key in applicable
        for spec in option.inputs
    }
    assert set(declared) == {"u_image", "operation", *applicable_inputs}
    assert "texelFetch(u_image, center, 0)" in FILTER_MIRROR_SOURCE
