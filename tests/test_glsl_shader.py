from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from dinkster_api.v1 import Curve
from dinkster_image import glsl as glsl_module
from dinkster_image.glsl import (
    DEFAULT_FRAGMENT_SHADER,
    GlslShader,
    NativeGlslShader,
)


def _image(batch: int = 1, height: int = 2, width: int = 3) -> np.ndarray:
    values = np.arange(batch * height * width * 3, dtype=np.float32)
    return values.reshape(batch, height, width, 3) / max(1, values.size - 1)


def _outputs(batch: int, height: int, width: int) -> tuple[np.ndarray, ...]:
    return tuple(
        np.full((batch, height, width, 4), index / 4, dtype=np.float32) for index in range(4)
    )


def _run_angle_renderer_or_skip(
    request: Path,
    expected_shape: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        return glsl_module._run_isolated_renderer(request, expected_shape)
    except RuntimeError as exc:
        if str(exc).startswith("GLSL shader execution is unavailable:"):
            pytest.skip(str(exc))
        raise


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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"images": {}}, "at least one"),
        ({"images": {"u_image5": _image()}}, "unknown members"),
        (
            {"images": {"u_image0": _image(), "u_image1": _image(batch=2)}},
            "matching batch sizes",
        ),
        ({"images": {"u_image0": _image()}, "size_mode": "future"}, "unknown GLSL size mode"),
        (
            {"images": {"u_image0": _image()}, "floats": {"u_float0": float("nan")}},
            "must be finite",
        ),
        (
            {"images": {"u_image0": _image()}, "ints": {"u_int0": 1.5}},
            "must be an integer",
        ),
        (
            {"images": {"u_image0": _image()}, "bools": {"u_bool0": 1}},
            "must be a boolean",
        ),
    ],
)
def test_execute_refuses_malformed_or_unbounded_inputs(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        GlslShader.execute(**kwargs)  # type: ignore[arg-type]


def test_angle_child_renders_mrt_multipass_and_typed_uniforms(tmp_path: Path) -> None:
    missing = [name for name in ("comfy_angle", "OpenGL") if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(f"ANGLE dependencies unavailable: {', '.join(missing)}")
    source = """#version 300 es
#pragma passes 2
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D u_image0;
uniform sampler2D u_image1;
uniform sampler2D u_curve0;
uniform vec2 u_resolution;
uniform float u_float0;
uniform int u_int0;
uniform bool u_bool0;
uniform int u_pass;
in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;
layout(location = 1) out vec4 fragColor1;
layout(location = 2) out vec4 fragColor2;
layout(location = 3) out vec4 fragColor3;
void main() {
    vec4 base = u_pass == 0 ? texture(u_image1, v_texCoord) : texture(u_image0, v_texCoord);
    float typed = u_float0 + float(u_int0) * 0.01 + (u_bool0 ? 0.04 : 0.0);
    fragColor0 = base + vec4(typed + (u_pass == 0 ? 0.0 : 0.1), 0.0, 0.0, 0.0);
    fragColor1 = vec4(texture(u_curve0, vec2(0.5, 0.5)).r);
    fragColor2 = vec4(float(u_int0), float(u_bool0), float(u_pass), 1.0);
    fragColor3 = vec4(u_resolution / 10.0, float(u_pass), 1.0);
}
"""
    image = np.asarray(
        [[[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], [[0.6, 0.7, 0.8], [0.9, 1.0, 0.2]]]],
        dtype=np.float32,
    )
    request = glsl_module._write_request(
        tmp_path,
        source=source,
        width=2,
        height=2,
        images=[("u_image1", image)],
        floats=[("u_float0", 0.2)],
        ints=[("u_int0", 3)],
        bools=[("u_bool0", True)],
        curves=[("u_curve0", np.linspace(0, 1, 256, dtype=np.float32))],
    )
    image0, image1, image2, image3 = _run_angle_renderer_or_skip(request, (1, 2, 2, 4))
    expected0 = np.ones((1, 2, 2, 4), dtype=np.float32)
    expected0[..., :3] = image
    expected0[..., 0] += 0.64
    np.testing.assert_allclose(image0, expected0, rtol=0, atol=2e-6)
    np.testing.assert_allclose(image1, 0.5, rtol=0, atol=2e-3)
    expected2 = np.broadcast_to([3.0, 1.0, 1.0, 1.0], image2.shape)
    expected3 = np.broadcast_to([0.2, 0.2, 1.0, 1.0], image3.shape)
    np.testing.assert_allclose(image2, expected2, rtol=0, atol=1e-6)
    np.testing.assert_allclose(image3, expected3, rtol=0, atol=1e-6)


def test_angle_child_expands_single_channel_images_to_grayscale(tmp_path: Path) -> None:
    missing = [name for name in ("comfy_angle", "OpenGL") if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(f"ANGLE dependencies unavailable: {', '.join(missing)}")
    grayscale = np.asarray([[[[0.1], [0.7]]]], dtype=np.float32)
    request = glsl_module._write_request(
        tmp_path,
        source=DEFAULT_FRAGMENT_SHADER,
        width=2,
        height=1,
        images=[("u_image0", grayscale)],
        floats=[],
        ints=[],
        bools=[],
        curves=[],
    )

    image0, _image1, _image2, _image3 = _run_angle_renderer_or_skip(request, (1, 1, 2, 4))

    expected = np.ones((1, 1, 2, 4), dtype=np.float32)
    expected[..., :3] = grayscale
    np.testing.assert_allclose(image0, expected, rtol=0, atol=1e-6)
