"""First-party GLSL image rendering through a bounded child process."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    CurveWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputFamilySpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    StringWidget,
    TypeExpr,
    render_image_png,
    report_event,
    report_preview,
)

from .support import MAX_DIMENSION, check_output_size, image_array, resize_array

IMAGE = TypeExpr.concrete("dinkster.image")
CURVE = TypeExpr.concrete("dinkster.curve")
FLOAT = TypeExpr.concrete(CORE_FLOAT)
INT = TypeExpr.concrete(CORE_INT)
BOOLEAN = TypeExpr.concrete(CORE_BOOLEAN)
STRING = TypeExpr.concrete(CORE_STRING)

MAX_GLSL_SOURCE_BYTES = 65_536
MAX_GLSL_PASSES = 32
MAX_GLSL_IMAGES = 5
MAX_GLSL_FLOATS = 20
MAX_GLSL_INTS = 20
MAX_GLSL_BOOLS = 10
MAX_GLSL_CURVES = 4
GLSL_CURVE_SAMPLES = 256
GLSL_RENDER_TIMEOUT_SECONDS = 30
GLSL_PREVIEW_MAX_DIMENSION = 1024
MAX_NPY_HEADER_BYTES = 1024

DEFAULT_FRAGMENT_SHADER = """#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D u_image0;
uniform vec2 u_resolution;

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

void main() {
    fragColor0 = texture(u_image0, v_texCoord);
}
"""


def _number(value: object, *, subject: str, integer: bool = False) -> int | float:
    if integer:
        if type(value) is not int:
            raise TypeError(f"{subject} must be an integer")
        return value
    if type(value) not in (int, float):
        raise TypeError(f"{subject} must be a number")
    result = float(cast("int | float", value))
    if not math.isfinite(result):
        raise ValueError(f"{subject} must be finite")
    return result


def _ordered_family(
    values: Mapping[str, object], names: tuple[str, ...], *, subject: str
) -> list[tuple[str, object]]:
    unknown = set(values).difference(names)
    if unknown:
        raise ValueError(f"{subject} has unknown members: {', '.join(sorted(unknown))}")
    return [(name, values[name]) for name in names if name in values]


def _curve_samples(value: object, *, subject: str) -> np.ndarray:
    evaluate = getattr(value, "evaluate", None)
    if not callable(evaluate):
        raise TypeError(f"{subject} must be a curve")
    samples = np.asarray(
        [evaluate(index / (GLSL_CURVE_SAMPLES - 1)) for index in range(GLSL_CURVE_SAMPLES)],
        dtype=np.float32,
    )
    if samples.shape != (GLSL_CURVE_SAMPLES,) or not np.all(np.isfinite(samples)):
        raise ValueError(f"{subject} produced non-finite samples")
    return samples


def _preview_image(image: np.ndarray) -> np.ndarray:
    height, width = (int(image.shape[1]), int(image.shape[2]))
    scale = min(1.0, GLSL_PREVIEW_MAX_DIMENSION / max(width, height))
    if scale == 1.0:
        return image[:1]
    return resize_array(
        image[:1],
        max(1, round(width * scale)),
        max(1, round(height * scale)),
        "bilinear",
    )


def _write_request(
    root: Path,
    *,
    source: str,
    width: int,
    height: int,
    images: list[tuple[str, np.ndarray]],
    floats: list[tuple[str, float]],
    ints: list[tuple[str, int]],
    bools: list[tuple[str, bool]],
    curves: list[tuple[str, np.ndarray]],
) -> Path:
    input_files: list[dict[str, str]] = []
    for index, (name, image) in enumerate(images):
        filename = f"input-{index}.npy"
        np.save(root / filename, image, allow_pickle=False)
        input_files.append({"name": name, "file": filename})
    curve_files: list[dict[str, str]] = []
    for index, (name, curve) in enumerate(curves):
        filename = f"curve-{index}.npy"
        np.save(root / filename, curve, allow_pickle=False)
        curve_files.append({"name": name, "file": filename})
    request = root / "request.json"
    request.write_text(
        json.dumps(
            {
                "source": source,
                "width": width,
                "height": height,
                "images": input_files,
                "floats": dict(floats),
                "ints": dict(ints),
                "bools": dict(bools),
                "curves": curve_files,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return request


def _load_child_output(
    path: Path,
    *,
    index: int,
    expected_shape: tuple[int, int, int, int],
) -> np.ndarray:
    expected_bytes = math.prod(expected_shape) * np.dtype(np.float32).itemsize
    try:
        size = path.stat().st_size
        if size <= expected_bytes or size > expected_bytes + MAX_NPY_HEADER_BYTES:
            raise ValueError("file size does not match the bounded output")
        with path.open("rb") as file:
            version = np.lib.format.read_magic(file)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    file, max_header_size=MAX_NPY_HEADER_BYTES
                )
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    file, max_header_size=MAX_NPY_HEADER_BYTES
                )
            else:
                raise ValueError(f"unsupported npy version {version}")
            header_bytes = file.tell()
            if (
                shape != expected_shape
                or fortran_order
                or dtype != np.dtype(np.float32)
                or size != header_bytes + expected_bytes
            ):
                raise ValueError("array header does not match the bounded output")
            payload = bytearray(expected_bytes)
            view = memoryview(payload)
            cursor = 0
            while cursor < expected_bytes:
                read = file.readinto(view[cursor:])
                if not read:
                    break
                cursor += read
            if cursor != expected_bytes or file.read(1):
                raise ValueError("array payload does not match the bounded output")
    except (EOFError, OSError, ValueError) as exc:
        raise RuntimeError(f"GLSL shader process returned invalid output {index}") from exc
    return np.frombuffer(payload, dtype=np.float32).reshape(expected_shape)


def _run_isolated_renderer(
    request: Path,
    expected_shape: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = request.parent
    try:
        result = subprocess.run(
            [sys.executable, "-m", "dinkster_nodes_image.glsl_process", str(request)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=GLSL_RENDER_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"GLSL shader execution exceeded {GLSL_RENDER_TIMEOUT_SECONDS} seconds"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"GLSL shader process could not start: {exc}") from exc

    status_path = root / "status.json"
    if not status_path.is_file():
        detail = result.stderr.strip() or f"child exited with status {result.returncode}"
        raise RuntimeError(f"GLSL shader process failed: {detail}")
    try:
        raw_status: object = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("GLSL shader process returned malformed status") from exc
    if not isinstance(raw_status, dict):
        raise RuntimeError("GLSL shader process returned malformed status")
    status = cast("dict[str, object]", raw_status)
    if type(status.get("ok")) is not bool:
        raise RuntimeError("GLSL shader process returned malformed status")
    if not status["ok"]:
        kind = status.get("kind")
        detail = status.get("message")
        if type(detail) is not str or not detail:
            detail = result.stderr.strip() or "unknown child failure"
        if kind == "unavailable":
            raise RuntimeError(f"GLSL shader execution is unavailable: {detail}")
        if kind == "shader":
            raise ValueError(f"GLSL shader compilation failed: {detail}")
        raise RuntimeError(f"GLSL shader execution failed: {detail}")
    if result.returncode != 0:
        raise RuntimeError(f"GLSL shader process exited with status {result.returncode}")

    outputs: list[np.ndarray] = []
    for index in range(4):
        path = root / f"output-{index}.npy"
        outputs.append(_load_child_output(path, index=index, expected_shape=expected_shape))
    return cast("tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]", tuple(outputs))


def _family_template(
    type_expr: TypeExpr,
    default: object,
    widget: CurveWidget | NumberWidget | None,
) -> tuple[InputSpec, ...]:
    return (
        InputSpec(
            "value",
            type_expr,
            required=False,
            default=default,
            widget=widget,
        ),
    )


class GlslShader(Node):
    """Render a GLSL ES 3.00 fragment shader outside the engine process."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        curve_default = {
            "interpolation": "monotone_cubic",
            "points": [
                {"position": 0.0, "value": 0.0},
                {"position": 1.0, "value": 1.0},
            ],
        }
        return NodeSchema(
            node_type="dinkster.image.glsl_shader",
            display_name="GLSL Shader",
            category="image/shader",
            description=(
                "Render a bounded GLSL ES 3.00 fragment shader with isolated native execution."
            ),
            inputs=(
                InputSpec(
                    "fragment_shader",
                    STRING,
                    required=False,
                    default=DEFAULT_FRAGMENT_SHADER,
                    widget=StringWidget(multiline=True),
                ),
            ),
            combos=(
                DynamicComboSpec(
                    "size_mode",
                    (
                        DynamicComboOption("from_input"),
                        DynamicComboOption(
                            "custom",
                            (
                                InputSpec(
                                    "width",
                                    INT,
                                    required=False,
                                    default=512,
                                    widget=NumberWidget(min=1, max=MAX_DIMENSION, step=1),
                                ),
                                InputSpec(
                                    "height",
                                    INT,
                                    required=False,
                                    default=512,
                                    widget=NumberWidget(min=1, max=MAX_DIMENSION, step=1),
                                ),
                            ),
                        ),
                    ),
                    default="from_input",
                ),
            ),
            input_families=(
                InputFamilySpec(
                    "images",
                    IMAGE,
                    min_members=1,
                    member_names=tuple(f"u_image{index}" for index in range(MAX_GLSL_IMAGES)),
                    display_name="Images",
                ),
                InputFamilySpec(
                    "floats",
                    _family_template(FLOAT, 0.0, NumberWidget(step=0.01)),
                    member_names=tuple(f"u_float{index}" for index in range(MAX_GLSL_FLOATS)),
                    display_name="Float uniforms",
                ),
                InputFamilySpec(
                    "ints",
                    _family_template(INT, 0, NumberWidget(step=1)),
                    member_names=tuple(f"u_int{index}" for index in range(MAX_GLSL_INTS)),
                    display_name="Integer uniforms",
                ),
                InputFamilySpec(
                    "bools",
                    _family_template(BOOLEAN, False, None),
                    member_names=tuple(f"u_bool{index}" for index in range(MAX_GLSL_BOOLS)),
                    display_name="Boolean uniforms",
                ),
                InputFamilySpec(
                    "curves",
                    _family_template(CURVE, curve_default, CurveWidget()),
                    member_names=tuple(f"u_curve{index}" for index in range(MAX_GLSL_CURVES)),
                    display_name="Curve uniforms",
                ),
            ),
            outputs=tuple(
                OutputSpec(
                    f"image{index}",
                    IMAGE,
                    doc=f"Shader output fragColor{index}.",
                    preview=index == 0,
                    alpha_policy="create_if_missing",
                )
                for index in range(4)
            ),
            output_node=True,
            emits_previews=True,
            idempotent=False,
            search_terms=("shader", "fragment shader", "GLSLShader", "WebGL"),
        )

    @classmethod
    def execute(
        cls,
        *,
        fragment_shader: str = DEFAULT_FRAGMENT_SHADER,
        size_mode: str = "from_input",
        width: int = 512,
        height: int = 512,
        images: Mapping[str, object],
        floats: Mapping[str, object] | None = None,
        ints: Mapping[str, object] | None = None,
        bools: Mapping[str, object] | None = None,
        curves: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        if type(fragment_shader) is not str:
            raise TypeError("fragment_shader must be a string")
        if len(fragment_shader.encode("utf-8")) > MAX_GLSL_SOURCE_BYTES:
            raise ValueError(f"fragment_shader exceeds {MAX_GLSL_SOURCE_BYTES} UTF-8 bytes")
        image_values = [
            (name, image_array(value, subject=name))
            for name, value in _ordered_family(
                images,
                tuple(f"u_image{index}" for index in range(MAX_GLSL_IMAGES)),
                subject="GLSL image inputs",
            )
        ]
        if not image_values:
            raise ValueError("GLSL Shader requires at least one input image")
        batch_size = int(image_values[0][1].shape[0])
        if any(int(image.shape[0]) != batch_size for _, image in image_values):
            raise ValueError("GLSL image inputs must have matching batch sizes")
        if size_mode == "from_input":
            output_height, output_width = (int(value) for value in image_values[0][1].shape[1:3])
        elif size_mode == "custom":
            if type(width) is not int or type(height) is not int:
                raise TypeError("custom GLSL width and height must be integers")
            output_width, output_height = width, height
        else:
            raise ValueError(f"unknown GLSL size mode: {size_mode}")
        check_output_size((batch_size, output_height, output_width, 16))

        float_values = [
            (name, cast("float", _number(value, subject=name)))
            for name, value in _ordered_family(
                floats or {},
                tuple(f"u_float{index}" for index in range(MAX_GLSL_FLOATS)),
                subject="GLSL float uniforms",
            )
        ]
        int_values = [
            (name, cast("int", _number(value, subject=name, integer=True)))
            for name, value in _ordered_family(
                ints or {},
                tuple(f"u_int{index}" for index in range(MAX_GLSL_INTS)),
                subject="GLSL integer uniforms",
            )
        ]
        bool_values: list[tuple[str, bool]] = []
        for name, value in _ordered_family(
            bools or {},
            tuple(f"u_bool{index}" for index in range(MAX_GLSL_BOOLS)),
            subject="GLSL boolean uniforms",
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be a boolean")
            bool_values.append((name, value))
        curve_values = [
            (name, _curve_samples(value, subject=name))
            for name, value in _ordered_family(
                curves or {},
                tuple(f"u_curve{index}" for index in range(MAX_GLSL_CURVES)),
                subject="GLSL curve uniforms",
            )
        ]

        preview_streams: list[dict[str, str]] = []
        for name, image in image_values:
            stream = f"glsl-input-{name}"
            preview = _preview_image(image)
            report_preview(
                render_image_png(preview),
                mime="image/png",
                width=int(preview.shape[2]),
                height=int(preview.shape[1]),
                stream=stream,
            )
            preview_streams.append({"name": name, "stream": stream})
        report_event(
            "dinkster.glsl.state",
            {
                "width": output_width,
                "height": output_height,
                "inputs": preview_streams,
                "floats": dict(float_values),
                "ints": dict(int_values),
                "bools": dict(bool_values),
                "curves": {name: values.tolist() for name, values in curve_values},
            },
        )

        with tempfile.TemporaryDirectory(prefix="dinkster-glsl-") as temporary:
            request = _write_request(
                Path(temporary),
                source=fragment_shader,
                width=output_width,
                height=output_height,
                images=image_values,
                floats=float_values,
                ints=int_values,
                bools=bool_values,
                curves=curve_values,
            )
            outputs = _run_isolated_renderer(
                request,
                (batch_size, output_height, output_width, 4),
            )
        return cls.outputs(**{f"image{index}": output for index, output in enumerate(outputs)})


GLSL_NODES: tuple[type[Node], ...] = (GlslShader,)


__all__ = ["DEFAULT_FRAGMENT_SHADER", "GLSL_NODES", "GlslShader"]
