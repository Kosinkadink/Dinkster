from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np
import pytest
from dinkster_nodes_foundation import Curve
from dinkster_nodes_image import DEFAULT_FRAGMENT_SHADER, IMAGE_NODES, GlslShader
from dinkster_nodes_image import glsl as glsl_module
from dinkster_schema import use_reporter
from numpy.lib.format import write_array_header_1_0


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


def test_glsl_shader_schema_is_fixed_bounded_and_discoverable() -> None:
    schema = GlslShader.schema()
    assert GlslShader in IMAGE_NODES
    assert schema.node_type == "dinkster.image.glsl_shader"
    assert schema.output_node is True
    assert schema.emits_previews is True
    assert schema.idempotent is False
    assert schema.input("fragment_shader").widget.multiline is True  # type: ignore[union-attr]
    assert [output.id for output in schema.outputs] == ["image0", "image1", "image2", "image3"]
    assert [output.preview for output in schema.outputs] == [True, False, False, False]
    families = {family.id: family for family in schema.input_families}
    assert families["images"].member_names == tuple(f"u_image{index}" for index in range(5))
    assert families["floats"].member_names == tuple(f"u_float{index}" for index in range(20))
    assert families["ints"].member_names == tuple(f"u_int{index}" for index in range(20))
    assert families["bools"].member_names == tuple(f"u_bool{index}" for index in range(10))
    assert families["curves"].member_names == tuple(f"u_curve{index}" for index in range(4))
    assert [option.key for option in schema.combos[0].options] == ["from_input", "custom"]


def test_execute_projects_exact_uniforms_previews_and_child_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: dict[str, object] = {}

    def render(
        request: Path,
        expected_shape: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw = json.loads(request.read_text(encoding="utf-8"))
        captured_request.update(cast("dict[str, object]", raw))
        assert expected_shape == (2, 4, 5, 4)
        first = np.load(request.parent / "input-0.npy", allow_pickle=False)
        curve = np.load(request.parent / "curve-0.npy", allow_pickle=False)
        np.testing.assert_array_equal(first, _image(batch=2))
        np.testing.assert_allclose(curve, np.linspace(0, 1, 256), rtol=0, atol=1e-7)
        return cast(
            "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]",
            _outputs(2, 4, 5),
        )

    monkeypatch.setattr(glsl_module, "_run_isolated_renderer", render)
    reports: list[tuple[str, Mapping[str, object], bytes | None]] = []
    with use_reporter(lambda name, data, blob: reports.append((name, data, blob))):
        result = GlslShader.execute(
            fragment_shader=DEFAULT_FRAGMENT_SHADER,
            size_mode="custom",
            width=5,
            height=4,
            images={"u_image1": _image(batch=2)},
            floats={"u_float0": 0.25},
            ints={"u_int0": 3},
            bools={"u_bool0": True},
            curves={"u_curve0": Curve(((0.0, 0.0), (1.0, 1.0)))},
        )

    assert list(result) == ["image0", "image1", "image2", "image3"]
    np.testing.assert_array_equal(result["image3"], _outputs(2, 4, 5)[3])
    assert captured_request["width"] == 5
    assert captured_request["height"] == 4
    assert captured_request["floats"] == {"u_float0": 0.25}
    assert captured_request["ints"] == {"u_int0": 3}
    assert captured_request["bools"] == {"u_bool0": True}
    preview = next(item for item in reports if item[0] == "preview")
    assert preview[1]["stream"] == "glsl-input-u_image1"
    assert preview[1]["mime"] == "image/png"
    assert preview[2] is not None and cast("bytes", preview[2]).startswith(b"\x89PNG")
    state = next(item[1] for item in reports if item[0] == "dinkster.glsl.state")
    assert state == {
        "width": 5,
        "height": 4,
        "inputs": [{"name": "u_image1", "stream": "glsl-input-u_image1"}],
        "floats": {"u_float0": 0.25},
        "ints": {"u_int0": 3},
        "bools": {"u_bool0": True},
        "curves": {"u_curve0": pytest.approx(np.linspace(0, 1, 256).tolist())},
    }


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


def test_execute_refuses_oversized_source_before_starting_child() -> None:
    with pytest.raises(ValueError, match="65536"):
        GlslShader.execute(fragment_shader="x" * 65_537, images={"u_image0": _image()})


@pytest.mark.parametrize(
    ("status", "error_type", "message"),
    [
        (
            {"ok": False, "kind": "unavailable", "message": "no ANGLE wheel"},
            RuntimeError,
            "execution is unavailable: no ANGLE wheel",
        ),
        (
            {"ok": False, "kind": "shader", "message": "syntax error"},
            ValueError,
            "compilation failed: syntax error",
        ),
        (
            {"ok": False, "kind": "runtime", "message": "context lost"},
            RuntimeError,
            "execution failed: context lost",
        ),
    ],
)
def test_parent_translates_child_refusals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
        return subprocess.CompletedProcess([], 2, "", "")

    monkeypatch.setattr("dinkster_nodes_image.glsl.subprocess.run", run)
    with pytest.raises(error_type, match=message):
        glsl_module._run_isolated_renderer(request, (1, 1, 1, 4))


def test_parent_translates_child_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")

    def timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["python"], 30)

    monkeypatch.setattr("dinkster_nodes_image.glsl.subprocess.run", timeout)
    with pytest.raises(RuntimeError, match="exceeded 30 seconds"):
        glsl_module._run_isolated_renderer(request, (1, 1, 1, 4))


def test_parent_refuses_unbounded_child_output_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        (tmp_path / "status.json").write_text('{"ok":true}', encoding="utf-8")
        with (tmp_path / "output-0.npy").open("wb") as file:
            write_array_header_1_0(
                file,
                {"descr": "<f4", "fortran_order": False, "shape": (1_000_000_000, 4)},
            )
            file.write(bytes(16))
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr("dinkster_nodes_image.glsl.subprocess.run", run)
    monkeypatch.setattr(
        np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("invalid child output reached np.load"),
    )
    with pytest.raises(RuntimeError, match="invalid output 0"):
        glsl_module._run_isolated_renderer(request, (1, 1, 1, 4))


@pytest.mark.skipif(os.name == "nt", reason="Windows denies replacing an open file")
def test_parent_reads_from_validated_descriptor_when_output_path_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "output-0.npy"
    expected = np.ones((1, 1, 1, 4), dtype=np.float32)
    np.save(path, expected, allow_pickle=False)
    replacement = tmp_path / "replacement.npy"
    np.save(replacement, np.full(expected.shape, 9, dtype=np.float32), allow_pickle=False)
    read_header = np.lib.format.read_array_header_1_0

    def read_then_replace(file: BinaryIO, *, max_header_size: int):
        header = read_header(file, max_header_size=max_header_size)
        replacement.replace(path)
        return header

    monkeypatch.setattr(np.lib.format, "read_array_header_1_0", read_then_replace)

    output = glsl_module._load_child_output(path, index=0, expected_shape=expected.shape)

    np.testing.assert_array_equal(output, expected)
    assert output.flags.writeable is True


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
