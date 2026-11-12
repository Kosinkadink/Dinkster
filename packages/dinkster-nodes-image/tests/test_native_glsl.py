"""Retained native GLSL dynamic interface and typed uniform execution."""

from __future__ import annotations

import numpy as np
import pytest
from dinkster_api.v1 import Curve
from dinkster_image import NODES as IMAGE_NODES
from dinkster_image.ported.glsl import NativeGlslShader


def _image(batch: int = 1, height: int = 2, width: int = 3) -> np.ndarray:
    values = np.arange(batch * height * width * 3, dtype=np.float32)
    return values.reshape(batch, height, width, 3) / max(1, values.size - 1)


def test_exact_glsl_schema_preserves_source_members_and_output_positions() -> None:
    from dinkster_schema import elaborate, schema_from_wire, schema_to_wire

    schema = NativeGlslShader.schema()
    assert NativeGlslShader in IMAGE_NODES
    assert schema_from_wire(schema_to_wire(schema)) == schema
    effective = elaborate(
        schema,
        ("images.image4", "images.image0", "images.image2", "floats.u_float3"),
        slot_variants={"size_mode": "custom"},
    )
    assert [(spec.id, spec.required) for spec in effective.inputs] == [
        ("fragment_shader", True),
        ("size_mode.width", True),
        ("size_mode.height", True),
        ("images.image4", False),
        ("images.image0", True),
        ("images.image2", False),
        ("floats.u_float3", False),
    ]


def test_exact_glsl_compacts_source_members_and_keeps_empty_uniform_defaults() -> None:
    source = """#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D u_image0;
uniform sampler2D u_image1;
uniform sampler2D u_curve0;
uniform float u_float0;
uniform float u_float1;
uniform int u_int0;
uniform int u_int1;
uniform bool u_bool0;
uniform bool u_bool1;
in vec2 v_texCoord;
layout(location=0) out vec4 fragColor0;
layout(location=1) out vec4 fragColor1;
layout(location=2) out vec4 fragColor2;
layout(location=3) out vec4 fragColor3;
void main() {
    fragColor0 = texture(u_image0, v_texCoord);
    fragColor1 = vec4(texture(u_image1, v_texCoord).rgb, u_float1);
    fragColor2 = vec4(float(u_int1)/10.0, u_bool1 ? 0.6 : 0.1,
        u_float0 + float(u_int0) + (u_bool0 ? 1.0 : 0.0), 1.0);
    fragColor3 = vec4(texture(u_curve0, vec2(0.5, 0.5)).rrr, 1.0);
}
"""
    try:
        result = NativeGlslShader.execute(
            fragment_shader=source,
            images={"image4": _image(2), "image0": None, "image2": _image(2) * 0.3},
            floats={"u_float3": None, "u_float1": 0.7},
            ints={"u_int4": None, "u_int2": 9},
            bools={"u_bool5": None, "u_bool0": True},
            curves={"u_curve0": None, "u_curve3": Curve(((0.0, 0.0), (1.0, 1.0)))},
        )
    except RuntimeError as exc:
        if str(exc).startswith("GLSL shader execution is unavailable:"):
            pytest.skip(str(exc))
        raise
    assert list(result) == ["IMAGE0", "IMAGE1", "IMAGE2", "IMAGE3"]
    expected = np.ones((2, 2, 3, 4), dtype=np.float32)
    expected[..., :3] = _image(2)
    np.testing.assert_allclose(np.asarray(result["IMAGE0"]), expected, rtol=0, atol=1e-6)
    expected[..., :3] *= 0.3
    expected[..., 3] = 0.7
    np.testing.assert_allclose(np.asarray(result["IMAGE1"]), expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(result["IMAGE2"]),
        np.broadcast_to([0.9, 0.6, 0.0, 1.0], expected.shape),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result["IMAGE3"]),
        np.broadcast_to([0.5, 0.5, 0.5, 1.0], expected.shape),
        rtol=0,
        atol=1e-6,
    )
