from __future__ import annotations

import ctypes
import ctypes.util
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNAVAILABLE_EXIT = 77
EGL_PLATFORM_ANGLE_ANGLE = 0x3202
EGL_PLATFORM_ANGLE_TYPE_ANGLE = 0x3203
EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE = 0x3450
EGL_MESA_PLATFORM_SURFACELESS = 0x31DD
VERTEX_SOURCE = """#version 300 es
void main() {
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
"""


class AngleUnavailable(RuntimeError):
    pass


def _load_angle() -> tuple[Any, Any]:
    try:
        comfy_angle = importlib.import_module("comfy_angle")
        egl_path = str(comfy_angle.get_egl_path())
        gles_path = str(comfy_angle.get_glesv2_path())
        if sys.platform == "win32":
            angle_dir = str(comfy_angle.get_lib_dir())
            os.add_dll_directory(angle_dir)
            os.environ["PATH"] = angle_dir + os.pathsep + os.environ.get("PATH", "")
        mode = 0 if sys.platform == "win32" else ctypes.RTLD_GLOBAL
        ctypes.CDLL(egl_path, mode=mode)
        ctypes.CDLL(gles_path, mode=mode)
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        if sys.platform != "linux":
            original_find_library = ctypes.util.find_library

            def find_angle_library(name: str) -> str | None:
                if name == "EGL":
                    return egl_path
                if name == "GLESv2":
                    return gles_path
                return original_find_library(name)

            ctypes.util.find_library = find_angle_library
        opengl = importlib.import_module("OpenGL")
        opengl.USE_ACCELERATE = False
        return importlib.import_module("OpenGL.EGL"), importlib.import_module("OpenGL.GLES3")
    except (ImportError, OSError, AttributeError) as exc:
        raise AngleUnavailable(f"ANGLE runtime unavailable: {exc}") from exc


def _egl_attribs(egl: Any, *values: int) -> Any:
    attrs = [*values, egl.EGL_NONE]
    return (ctypes.c_int32 * len(attrs))(*attrs)


def _initialize_display(egl: Any) -> tuple[Any, int, int]:
    failures: list[str] = []
    display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
    if display:
        major, minor = ctypes.c_int32(), ctypes.c_int32()
        try:
            if egl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
                return display, major.value, minor.value
        except Exception as exc:
            failures.append(f"default display: {exc}")

    platform = importlib.import_module("OpenGL.platform")
    egl_library = platform.PLATFORM.EGL
    get_proc = egl_library.eglGetProcAddress
    get_proc.restype = ctypes.c_void_p
    get_proc.argtypes = [ctypes.c_char_p]
    pointer = get_proc(b"eglGetPlatformDisplayEXT")
    if pointer:
        function_type = ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        get_platform_display = function_type(pointer)
        strategies = [
            ("surfaceless", EGL_MESA_PLATFORM_SURFACELESS, None),
            (
                "ANGLE Vulkan",
                EGL_PLATFORM_ANGLE_ANGLE,
                _egl_attribs(
                    egl,
                    EGL_PLATFORM_ANGLE_TYPE_ANGLE,
                    EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE,
                ),
            ),
        ]
        for name, platform_id, attributes in strategies:
            raw_display = get_platform_display(platform_id, None, attributes)
            if not raw_display:
                failures.append(f"{name}: no display")
                continue
            candidate = ctypes.cast(raw_display, egl.EGLDisplay)
            major, minor = ctypes.c_int32(), ctypes.c_int32()
            try:
                if egl.eglInitialize(candidate, ctypes.byref(major), ctypes.byref(minor)):
                    return candidate, major.value, minor.value
                failures.append(f"{name}: initialization failed")
            except Exception as exc:
                failures.append(f"{name}: {exc}")
    else:
        failures.append("eglGetPlatformDisplayEXT unavailable")
    raise AngleUnavailable("no headless EGL display (" + "; ".join(failures) + ")")


def _gl_text(gl: Any, name: int) -> str:
    value = gl.glGetString(name)
    if not value:
        return "Unknown"
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return ctypes.string_at(value).decode(errors="replace")


class AngleRenderer:
    def __init__(self, egl: Any, gl: Any) -> None:
        self.egl = egl
        self.gl = gl
        self.display: Any | None = None
        self.surface: Any | None = None
        self.context: Any | None = None
        self.vao: int | None = None
        try:
            self.display, egl_major, egl_minor = _initialize_display(egl)
            if not egl.eglBindAPI(egl.EGL_OPENGL_ES_API):
                raise AngleUnavailable("eglBindAPI rejected OpenGL ES")
            config = egl.EGLConfig()
            config_count = ctypes.c_int32()
            config_attributes = _egl_attribs(
                egl,
                egl.EGL_RENDERABLE_TYPE,
                egl.EGL_OPENGL_ES3_BIT,
                egl.EGL_SURFACE_TYPE,
                egl.EGL_PBUFFER_BIT,
                egl.EGL_RED_SIZE,
                8,
                egl.EGL_GREEN_SIZE,
                8,
                egl.EGL_BLUE_SIZE,
                8,
                egl.EGL_ALPHA_SIZE,
                8,
            )
            if (
                not egl.eglChooseConfig(
                    self.display,
                    config_attributes,
                    ctypes.byref(config),
                    1,
                    ctypes.byref(config_count),
                )
                or config_count.value < 1
            ):
                raise AngleUnavailable("no EGL OpenGL ES 3 pbuffer configuration")
            self.surface = egl.eglCreatePbufferSurface(
                self.display,
                config,
                _egl_attribs(egl, egl.EGL_WIDTH, 64, egl.EGL_HEIGHT, 64),
            )
            if not self.surface:
                raise AngleUnavailable("EGL pbuffer creation failed")
            self.context = egl.eglCreateContext(
                self.display,
                config,
                egl.EGL_NO_CONTEXT,
                _egl_attribs(egl, egl.EGL_CONTEXT_CLIENT_VERSION, 3),
            )
            if not self.context:
                raise AngleUnavailable("EGL OpenGL ES 3 context creation failed")
            if not egl.eglMakeCurrent(self.display, self.surface, self.surface, self.context):
                raise AngleUnavailable("EGL context activation failed")
            version = _gl_text(gl, gl.GL_VERSION)
            if "OpenGL ES 3" not in version:
                raise AngleUnavailable(f"OpenGL ES 3 required, found {version}")
            self.vao = cast("int", gl.glGenVertexArrays(1))
            gl.glBindVertexArray(self.vao)
            self.description = (
                f"EGL {egl_major}.{egl_minor}; {_gl_text(gl, gl.GL_RENDERER)}; "
                f"{_gl_text(gl, gl.GL_VENDOR)}; {version}"
            )
        except AngleUnavailable:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise AngleUnavailable(f"EGL/GLES initialization failed: {exc}") from exc

    def _compile_shader(self, source: str, shader_type: int) -> int:
        gl = self.gl
        shader = cast("int", gl.glCreateShader(shader_type))
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            detail = gl.glGetShaderInfoLog(shader)
            gl.glDeleteShader(shader)
            raise RuntimeError(f"ANGLE shader compilation failed: {detail!r}")
        return shader

    def _program(self, fragment_source: str) -> int:
        gl = self.gl
        vertex = self._compile_shader(VERTEX_SOURCE, gl.GL_VERTEX_SHADER)
        try:
            fragment = self._compile_shader(fragment_source, gl.GL_FRAGMENT_SHADER)
        except Exception:
            gl.glDeleteShader(vertex)
            raise
        program = cast("int", gl.glCreateProgram())
        gl.glAttachShader(program, vertex)
        gl.glAttachShader(program, fragment)
        gl.glLinkProgram(program)
        gl.glDeleteShader(vertex)
        gl.glDeleteShader(fragment)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            detail = gl.glGetProgramInfoLog(program)
            gl.glDeleteProgram(program)
            raise RuntimeError(f"ANGLE program linking failed: {detail!r}")
        return program

    def render(
        self,
        fragment_source: str,
        image: np.ndarray,
        scalars: dict[str, tuple[str, int | float]],
    ) -> np.ndarray:
        gl = self.gl
        height, width, channels = image.shape
        rgba = np.zeros((height, width, 4), dtype=np.float32)
        rgba[:, :, :channels] = image
        program = self._program(fragment_source)
        framebuffer: int | None = None
        input_texture: int | None = None
        output_texture: int | None = None
        try:
            gl.glUseProgram(program)
            input_texture = cast("int", gl.glGenTextures(1))
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, input_texture)
            self._configure_texture()
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RGBA32F,
                width,
                height,
                0,
                gl.GL_RGBA,
                gl.GL_FLOAT,
                np.ascontiguousarray(rgba),
            )
            sampler = gl.glGetUniformLocation(program, "u_image")
            if sampler < 0:
                raise RuntimeError("ANGLE shader omitted required u_image uniform")
            gl.glUniform1i(sampler, 0)

            for name, (scalar_type, value) in scalars.items():
                location = gl.glGetUniformLocation(program, name)
                if location < 0:
                    raise RuntimeError(f"ANGLE shader omitted required {name} uniform")
                if scalar_type == "float":
                    gl.glUniform1f(location, float(value))
                else:
                    gl.glUniform1i(location, int(value))

            output_texture = cast("int", gl.glGenTextures(1))
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, output_texture)
            self._configure_texture()
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RGBA32F,
                width,
                height,
                0,
                gl.GL_RGBA,
                gl.GL_FLOAT,
                None,
            )
            framebuffer = cast("int", gl.glGenFramebuffers(1))
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER,
                gl.GL_COLOR_ATTACHMENT0,
                gl.GL_TEXTURE_2D,
                output_texture,
                0,
            )
            gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])
            if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                raise AngleUnavailable("ANGLE cannot render to an RGBA32F framebuffer")

            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, input_texture)
            gl.glViewport(0, 0, width, height)
            gl.glDisable(gl.GL_BLEND)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_SCISSOR_TEST)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            output = np.empty((height, width, 4), dtype=np.float32)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_FLOAT, output)
            error = gl.glGetError()
            if error != gl.GL_NO_ERROR:
                raise AngleUnavailable(f"ANGLE float rendering failed with GL error 0x{error:04x}")
            return output
        finally:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            gl.glUseProgram(0)
            if input_texture is not None:
                gl.glDeleteTextures(1, [input_texture])
            if output_texture is not None:
                gl.glDeleteTextures(1, [output_texture])
            if framebuffer is not None:
                gl.glDeleteFramebuffers(1, [framebuffer])
            gl.glDeleteProgram(program)

    def _configure_texture(self) -> None:
        gl = self.gl
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

    def close(self) -> None:
        if self.display is None:
            return
        try:
            if self.context is not None and self.surface is not None:
                self.egl.eglMakeCurrent(self.display, self.surface, self.surface, self.context)
                if self.vao is not None:
                    self.gl.glDeleteVertexArrays(1, [self.vao])
                self.egl.eglMakeCurrent(
                    self.display,
                    self.egl.EGL_NO_SURFACE,
                    self.egl.EGL_NO_SURFACE,
                    self.egl.EGL_NO_CONTEXT,
                )
                self.egl.eglDestroyContext(self.display, self.context)
            if self.surface is not None:
                self.egl.eglDestroySurface(self.display, self.surface)
            self.egl.eglTerminate(self.display)
        finally:
            self.display = None


def _load_corpus(name: str) -> dict[str, Any]:
    path = ROOT / "tests" / "fixtures" / "mirror-parity" / name
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _run_cases(
    renderer: AngleRenderer,
    corpus: dict[str, Any],
    source: str,
    operations: tuple[str, ...],
) -> tuple[int, float]:
    frames = cast("dict[str, dict[str, Any]]", corpus["frames"])
    tolerance = cast("float", corpus["mirror_per_channel_tolerance"])
    max_delta = 0.0
    count = 0
    for case in cast("list[dict[str, Any]]", corpus["cases"]):
        frame_record = frames[cast("str", case["frame"])]
        frame = np.asarray(frame_record["values"], dtype=np.float32).reshape(frame_record["shape"])
        expected_record = cast("dict[str, Any]", case["expected"])
        expected = np.asarray(expected_record["values"], dtype=np.float32).reshape(
            expected_record["shape"]
        )
        inputs = cast("dict[str, Any]", case["inputs"])
        operation = cast("str", inputs["operation"])
        for batch_index in range(frame.shape[0]):
            if corpus["node_type"] == "dinkster.image.adjust":
                scalars = {
                    "operation": ("int", operations.index(operation)),
                    "factor": ("float", cast("float", inputs.get("factor", 1.0))),
                    "mean": ("float", cast("float", inputs.get("mean", 0.5))),
                    "standard_deviation": (
                        "float",
                        cast("float", inputs.get("standard_deviation", 0.5)),
                    ),
                }
            else:
                scalars = {
                    "operation": ("int", operations.index(operation)),
                    "radius": ("int", cast("int", inputs["radius"])),
                    "sigma": ("float", cast("float", inputs["sigma"])),
                    "strength": ("float", cast("float", inputs.get("strength", 1.0))),
                }
            actual = renderer.render(source, frame[batch_index], scalars)
            wanted = expected[batch_index]
            visible = actual[:, :, : wanted.shape[2]]
            if not np.isfinite(visible).all():
                raise AssertionError(
                    f"{case['id']} batch {batch_index}: ANGLE returned non-finite values"
                )
            delta = float(np.max(np.abs(visible - wanted)))
            max_delta = max(max_delta, delta)
            if delta > tolerance:
                raise AssertionError(
                    f"{case['id']} batch {batch_index}: max delta {delta} exceeds {tolerance}"
                )
        count += 1
    return count, max_delta


def main() -> None:
    egl, gl = _load_angle()
    image_module = importlib.import_module("dinkster_nodes_image")
    adjust_module = importlib.import_module("dinkster_nodes_image.adjust")
    filter_module = importlib.import_module("dinkster_nodes_image.filters")
    declared = {
        schema.node_type: schema.mirror
        for node in image_module.IMAGE_NODES
        if (schema := node.schema()).mirror is not None and schema.mirror.kind == "glsl"
    }
    expected_types = {"dinkster.image.adjust", "dinkster.image.filter"}
    if set(declared) != expected_types:
        raise AssertionError(
            f"ANGLE corpus coverage does not match declared GLSL mirrors: {set(declared)}"
        )

    renderer = AngleRenderer(egl, gl)
    try:
        adjust = _load_corpus("image_adjust_v1.json")
        adjust_count, adjust_delta = _run_cases(
            renderer,
            adjust,
            adjust_module.ADJUST_MIRROR_SOURCE,
            tuple(adjust_module.ADJUST_OPERATIONS),
        )
        image_filter = _load_corpus("image_filter_v1.json")
        filter_count, filter_delta = _run_cases(
            renderer,
            image_filter,
            filter_module.FILTER_MIRROR_SOURCE,
            ("gaussian_blur", "sharpen"),
        )
        print(renderer.description)
        print(
            f"ANGLE mirror parity passed: {adjust_count + filter_count} cases; "
            f"max delta {max(adjust_delta, filter_delta):.9g}"
        )
    finally:
        renderer.close()


if __name__ == "__main__":
    try:
        main()
    except AngleUnavailable as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(UNAVAILABLE_EXIT) from exc
