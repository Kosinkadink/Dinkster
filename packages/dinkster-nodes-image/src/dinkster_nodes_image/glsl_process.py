"""Child-process ANGLE renderer for the first-party GLSL Shader node."""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np

EGL_PLATFORM_ANGLE_ANGLE = 0x3202
EGL_PLATFORM_ANGLE_TYPE_ANGLE = 0x3203
EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE = 0x3450
EGL_MESA_PLATFORM_SURFACELESS = 0x31DD
MAX_OUTPUTS = 4
MAX_PASSES = 32
MAX_IMAGES = 5

VERTEX_SOURCE = """#version 300 es
out vec2 v_texCoord;
void main() {
    vec2 vertices[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    v_texCoord = vertices[gl_VertexID] * 0.5 + 0.5;
    gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
}
"""


class AngleUnavailable(RuntimeError):
    pass


class ShaderError(ValueError):
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
        opengl = cast("Any", importlib.import_module("OpenGL"))
        opengl.USE_ACCELERATE = False
        return importlib.import_module("OpenGL.EGL"), importlib.import_module("OpenGL.GLES3")
    except (ImportError, OSError, AttributeError) as exc:
        raise AngleUnavailable(f"ANGLE runtime unavailable on {sys.platform}: {exc}") from exc


def _egl_attribs(egl: Any, *values: int) -> Any:
    attrs = [*values, egl.EGL_NONE]
    return (ctypes.c_int32 * len(attrs))(*attrs)


def _initialize_display(egl: Any) -> Any:
    failures: list[str] = []
    display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
    if display:
        major, minor = ctypes.c_int32(), ctypes.c_int32()
        try:
            if egl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
                return display
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
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p
        )
        get_platform_display = function_type(pointer)
        strategies = (
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
        )
        for name, platform_id, attributes in strategies:
            raw_display = get_platform_display(platform_id, None, attributes)
            if not raw_display:
                failures.append(f"{name}: no display")
                continue
            candidate = ctypes.cast(raw_display, egl.EGLDisplay)
            major, minor = ctypes.c_int32(), ctypes.c_int32()
            try:
                if egl.eglInitialize(candidate, ctypes.byref(major), ctypes.byref(minor)):
                    return candidate
                failures.append(f"{name}: initialization failed")
            except Exception as exc:
                failures.append(f"{name}: {exc}")
    else:
        failures.append("eglGetPlatformDisplayEXT unavailable")
    raise AngleUnavailable("no headless EGL display (" + "; ".join(failures) + ")")


def _gl_text(gl: Any, value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if value:
        return ctypes.string_at(value).decode(errors="replace")
    return "unknown error"


class AngleRenderer:
    def __init__(self, egl: Any, gl: Any) -> None:
        self.egl = egl
        self.gl = gl
        self.display: Any | None = None
        self.surface: Any | None = None
        self.context: Any | None = None
        self.vao: int | None = None
        try:
            self.display = _initialize_display(egl)
            if not egl.eglBindAPI(egl.EGL_OPENGL_ES_API):
                raise AngleUnavailable("eglBindAPI rejected OpenGL ES")
            config = egl.EGLConfig()
            config_count = ctypes.c_int32()
            attributes = _egl_attribs(
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
                    attributes,
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
            version = _gl_text(gl, gl.glGetString(gl.GL_VERSION))
            if "OpenGL ES 3" not in version:
                raise AngleUnavailable(f"OpenGL ES 3 required, found {version}")
            self.vao = cast("int", gl.glGenVertexArrays(1))
            gl.glBindVertexArray(self.vao)
        except AngleUnavailable:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise AngleUnavailable(f"EGL/GLES initialization failed: {exc}") from exc

    def _shader(self, source: str, shader_type: int) -> int:
        gl = self.gl
        shader = cast("int", gl.glCreateShader(shader_type))
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            detail = _gl_text(gl, gl.glGetShaderInfoLog(shader))
            gl.glDeleteShader(shader)
            raise ShaderError(detail)
        return shader

    def _program(self, source: str) -> int:
        gl = self.gl
        vertex = self._shader(VERTEX_SOURCE, gl.GL_VERTEX_SHADER)
        try:
            fragment = self._shader(source, gl.GL_FRAGMENT_SHADER)
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
            detail = _gl_text(gl, gl.glGetProgramInfoLog(program))
            gl.glDeleteProgram(program)
            raise ShaderError(detail)
        return program

    def render(
        self,
        source: str,
        width: int,
        height: int,
        images: list[tuple[str, np.ndarray]],
        floats: dict[str, float],
        ints: dict[str, int],
        bools: dict[str, bool],
        curves: list[tuple[str, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gl = self.gl
        matches = [int(value) for value in re.findall(r"\bfragColor(\d+)\b", source)]
        if matches and max(matches) >= MAX_OUTPUTS:
            raise ShaderError(f"fragColor outputs are limited to fragColor0-{MAX_OUTPUTS - 1}")
        output_count = max(matches, default=0) + 1
        pass_match = re.search(r"^\s*#pragma\s+passes\s+(\d+)\s*$", source, re.MULTILINE)
        passes = int(pass_match.group(1)) if pass_match else 1
        if not 1 <= passes <= MAX_PASSES:
            raise ShaderError(f"#pragma passes must be between 1 and {MAX_PASSES}")
        batch_size = int(images[0][1].shape[0])
        program = self._program(source)
        input_textures: list[int] = []
        curve_textures: list[int] = []
        output_textures: list[int] = []
        ping_textures: list[int] = []
        ping_framebuffers: list[int] = []
        framebuffer: int | None = None
        outputs = [np.empty((batch_size, height, width, 4), dtype=np.float32) for _ in range(4)]
        try:
            gl.glUseProgram(program)
            framebuffer = cast("int", gl.glGenFramebuffers(1))
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
            draw_buffers: list[int] = []
            for index in range(output_count):
                texture = cast("int", gl.glGenTextures(1))
                output_textures.append(texture)
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
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
                attachment = gl.GL_COLOR_ATTACHMENT0 + index
                gl.glFramebufferTexture2D(
                    gl.GL_FRAMEBUFFER, attachment, gl.GL_TEXTURE_2D, texture, 0
                )
                draw_buffers.append(attachment)
            gl.glDrawBuffers(output_count, draw_buffers)
            if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                raise AngleUnavailable("ANGLE cannot render to an RGBA32F framebuffer")

            if passes > 1:
                for _index in range(2):
                    texture = cast("int", gl.glGenTextures(1))
                    ping_textures.append(texture)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
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
                    ping = cast("int", gl.glGenFramebuffers(1))
                    ping_framebuffers.append(ping)
                    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, ping)
                    gl.glFramebufferTexture2D(
                        gl.GL_FRAMEBUFFER,
                        gl.GL_COLOR_ATTACHMENT0,
                        gl.GL_TEXTURE_2D,
                        texture,
                        0,
                    )
                    gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])
                    if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                        raise AngleUnavailable("ANGLE cannot create a multipass framebuffer")

            for name, _image in images:
                unit = int(name.removeprefix("u_image"))
                texture = cast("int", gl.glGenTextures(1))
                input_textures.append(texture)
                gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
                self._configure_texture()
                location = gl.glGetUniformLocation(program, name)
                if location >= 0:
                    gl.glUniform1i(location, unit)
            resolution = gl.glGetUniformLocation(program, "u_resolution")
            if resolution >= 0:
                gl.glUniform2f(resolution, float(width), float(height))
            for name, value in floats.items():
                location = gl.glGetUniformLocation(program, name)
                if location >= 0:
                    gl.glUniform1f(location, value)
            for name, value in ints.items():
                location = gl.glGetUniformLocation(program, name)
                if location >= 0:
                    gl.glUniform1i(location, value)
            for name, value in bools.items():
                location = gl.glGetUniformLocation(program, name)
                if location >= 0:
                    gl.glUniform1i(location, 1 if value else 0)
            for name, values in curves:
                texture = cast("int", gl.glGenTextures(1))
                curve_textures.append(texture)
                unit = MAX_IMAGES + int(name.removeprefix("u_curve"))
                gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
                self._configure_texture()
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D,
                    0,
                    gl.GL_R32F,
                    len(values),
                    1,
                    0,
                    gl.GL_RED,
                    gl.GL_FLOAT,
                    np.ascontiguousarray(values),
                )
                location = gl.glGetUniformLocation(program, name)
                if location >= 0:
                    gl.glUniform1i(location, unit)

            pass_location = gl.glGetUniformLocation(program, "u_pass")
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glViewport(0, 0, width, height)
            gl.glDisable(gl.GL_BLEND)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_SCISSOR_TEST)
            for batch in range(batch_size):
                for index, (name, image) in enumerate(images):
                    unit = int(name.removeprefix("u_image"))
                    rgba = np.ones((*image.shape[1:3], 4), dtype=np.float32)
                    channels = int(image.shape[3])
                    pixels = image[batch, ::-1]
                    if channels == 1:
                        rgba[:, :, :3] = pixels
                    else:
                        rgba[:, :, :channels] = pixels
                    gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, input_textures[index])
                    gl.glTexImage2D(
                        gl.GL_TEXTURE_2D,
                        0,
                        gl.GL_RGBA32F,
                        int(image.shape[2]),
                        int(image.shape[1]),
                        0,
                        gl.GL_RGBA,
                        gl.GL_FLOAT,
                        np.ascontiguousarray(rgba),
                    )
                for render_pass in range(passes):
                    last = render_pass == passes - 1
                    gl.glBindFramebuffer(
                        gl.GL_FRAMEBUFFER,
                        framebuffer if last else ping_framebuffers[render_pass % 2],
                    )
                    if last:
                        gl.glDrawBuffers(output_count, draw_buffers)
                    else:
                        gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])
                    if pass_location >= 0:
                        gl.glUniform1i(pass_location, render_pass)
                    if render_pass > 0:
                        gl.glActiveTexture(gl.GL_TEXTURE0)
                        gl.glBindTexture(
                            gl.GL_TEXTURE_2D,
                            ping_textures[(render_pass - 1) % 2],
                        )
                    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
                    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                    gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
                for index, _texture in enumerate(output_textures):
                    gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0 + index)
                    frame = np.empty((height, width, 4), dtype=np.float32)
                    gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_FLOAT, frame)
                    outputs[index][batch] = frame[::-1]
                for index in range(output_count, MAX_OUTPUTS):
                    outputs[index][batch].fill(0.0)
                error = gl.glGetError()
                if error != gl.GL_NO_ERROR:
                    raise RuntimeError(f"ANGLE rendering failed with GL error 0x{error:04x}")
            return cast("tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]", tuple(outputs))
        finally:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            gl.glUseProgram(0)
            if input_textures:
                gl.glDeleteTextures(len(input_textures), input_textures)
            if curve_textures:
                gl.glDeleteTextures(len(curve_textures), curve_textures)
            if output_textures:
                gl.glDeleteTextures(len(output_textures), output_textures)
            if ping_textures:
                gl.glDeleteTextures(len(ping_textures), ping_textures)
            if framebuffer is not None:
                gl.glDeleteFramebuffers(1, [framebuffer])
            if ping_framebuffers:
                gl.glDeleteFramebuffers(len(ping_framebuffers), ping_framebuffers)
            gl.glDeleteProgram(program)

    def _configure_texture(self) -> None:
        gl = self.gl
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
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
            self.surface = None
            self.context = None
            self.vao = None


def _fail(root: Path, kind: str, message: str) -> NoReturn:
    (root / "status.json").write_text(
        json.dumps({"ok": False, "kind": kind, "message": message}, separators=(",", ":")),
        encoding="utf-8",
    )
    raise SystemExit(2)


def _load_request(
    path: Path,
) -> tuple[
    str,
    int,
    int,
    list[tuple[str, np.ndarray]],
    dict[str, float],
    dict[str, int],
    dict[str, bool],
    list[tuple[str, np.ndarray]],
]:
    root = path.parent
    raw_value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict):
        raise ValueError("request root must be an object")
    raw = cast("dict[str, object]", raw_value)
    source, width, height = raw.get("source"), raw.get("width"), raw.get("height")
    if type(source) is not str or type(width) is not int or type(height) is not int:
        raise ValueError("request source and dimensions are malformed")
    image_records = raw.get("images")
    curve_records = raw.get("curves")
    if not isinstance(image_records, list) or not image_records:
        raise ValueError("request images must be a non-empty list")
    if not isinstance(curve_records, list):
        raise ValueError("request curves must be a list")
    images: list[tuple[str, np.ndarray]] = []
    for raw_record in cast("list[object]", image_records):
        if not isinstance(raw_record, dict):
            raise ValueError("request image entry must be an object")
        record = cast("dict[str, object]", raw_record)
        name, filename = record.get("name"), record.get("file")
        if type(name) is not str or type(filename) is not str:
            raise ValueError("request image name and file must be strings")
        images.append((name, np.load(root / filename, allow_pickle=False)))
    curves: list[tuple[str, np.ndarray]] = []
    for raw_record in cast("list[object]", curve_records):
        if not isinstance(raw_record, dict):
            raise ValueError("request curve entry must be an object")
        record = cast("dict[str, object]", raw_record)
        name, filename = record.get("name"), record.get("file")
        if type(name) is not str or type(filename) is not str:
            raise ValueError("request curve name and file must be strings")
        curves.append((name, np.load(root / filename, allow_pickle=False)))
    raw_floats, raw_ints, raw_bools = raw.get("floats"), raw.get("ints"), raw.get("bools")
    float_records = (
        cast("dict[object, object]", raw_floats) if isinstance(raw_floats, dict) else None
    )
    int_records = cast("dict[object, object]", raw_ints) if isinstance(raw_ints, dict) else None
    bool_records = cast("dict[object, object]", raw_bools) if isinstance(raw_bools, dict) else None
    if float_records is None or not all(
        type(name) is str and type(value) in (int, float) for name, value in float_records.items()
    ):
        raise ValueError("request floats must be a numeric object")
    if int_records is None or not all(
        type(name) is str and type(value) is int for name, value in int_records.items()
    ):
        raise ValueError("request ints must be an integer object")
    if bool_records is None or not all(
        type(name) is str and type(value) is bool for name, value in bool_records.items()
    ):
        raise ValueError("request bools must be a boolean object")
    floats = {str(name): float(cast("int | float", value)) for name, value in float_records.items()}
    ints = {str(name): cast("int", value) for name, value in int_records.items()}
    bools = {str(name): cast("bool", value) for name, value in bool_records.items()}
    return (
        source,
        width,
        height,
        images,
        floats,
        ints,
        bools,
        curves,
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m dinkster_nodes_image.glsl_process REQUEST.json")
    request = Path(sys.argv[1]).resolve()
    root = request.parent
    renderer: AngleRenderer | None = None
    try:
        source, width, height, images, floats, ints, bools, curves = _load_request(request)
        egl, gl = _load_angle()
        renderer = AngleRenderer(egl, gl)
        outputs = renderer.render(source, width, height, images, floats, ints, bools, curves)
        for index, output in enumerate(outputs):
            np.save(root / f"output-{index}.npy", output, allow_pickle=False)
        (root / "status.json").write_text(
            json.dumps({"ok": True}, separators=(",", ":")), encoding="utf-8"
        )
    except AngleUnavailable as exc:
        _fail(root, "unavailable", str(exc))
    except ShaderError as exc:
        _fail(root, "shader", str(exc))
    except Exception as exc:
        _fail(root, "runtime", str(exc))
    finally:
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
