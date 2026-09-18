"""Source-preserving conversion between ComfyUI VideoInput and portable VIDEO."""

from __future__ import annotations

import importlib
import io
import os
import tempfile
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
from dinkster_assets import AssetRef, resolver_from_env
from dinkster_assets.value import admit_video_source, bind_video_value
from dinkster_values import (
    TypeRegistry,
    coerce_video,
    decode_video,
    edit_video,
    effective_video_facts,
    encode_video,
    validate_video_encoded,
    video_fingerprint,
    video_meta,
    video_source,
)
from dinkster_values.video_edits import mapping
from dinkster_video import assemble_video, disassemble_video, save_video_stream
from dinkster_video.runtime import source_trim_window

from .audio import TorchAudio

VIDEO_V1_NAME = "VIDEO"


def _numpy(value: Any) -> np.ndarray:
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)


def _native(obj: object) -> dict[str, object]:
    if isinstance(obj, _VideoValue):
        return obj.value
    if isinstance(obj, Mapping):
        return coerce_video(cast("Mapping[str, object]", obj))
    impl = cast(Any, importlib.import_module("comfy_api.input_impl"))
    if type(obj) is impl.VideoFromFile:
        upstream = cast(Any, obj)
        source = upstream.get_stream_source()
        if isinstance(source, io.BytesIO):
            value = admit_video_source(source.getvalue())
        elif isinstance(source, str):
            value = admit_video_source(Path(source))
        else:
            raise TypeError("unsupported VideoFromFile source")
        start, duration = upstream.get_active_trim_window()
        if start or duration:
            value = edit_video(value, {"trim": {"start_time": start, "duration": duration}})
        # The pinned ComfyUI class has no public crop getter.
        crop = getattr(upstream, "_VideoFromFile__crop", None)
        if crop is not None:
            value = edit_video(
                value, {"crop": dict(zip(("x", "y", "width", "height"), crop, strict=True))}
            )
        return value
    if type(obj) is getattr(impl, "VideoFromComponents", None):
        upstream = cast(Any, obj)
        components = upstream.get_components()
        images = _numpy(components.images)
        if components.alpha is not None:
            images = np.concatenate((images[..., :3], _numpy(components.alpha)[..., None]), axis=-1)
        audio = components.audio
        if audio is not None:
            audio = {**audio, "waveform": _numpy(audio["waveform"])}
        return assemble_video(
            images,
            fps=components.frame_rate,
            audio=audio,
            bit_depth=str(upstream.get_bit_depth()),
            color_space=getattr(upstream, "get_color_space", lambda: "sRGB")(),
        )
    raise TypeError(
        "unsupported VideoInput subclass; boundary conversion must not encode or discard edits"
    )


class _VideoValue:
    def __init__(self, value: object) -> None:
        self.value = bind_video_value(value, resolver_from_env())

    def get_dimensions(self) -> tuple[int, int]:
        facts = effective_video_facts(self.value)
        return cast(int, facts["width"]), cast(int, facts["height"])

    def get_duration(self) -> float:
        duration = effective_video_facts(self.value)["duration"]
        if duration is None:
            raise ValueError("VIDEO duration is unknown")
        return float(cast(Fraction, duration))

    def get_frame_count(self) -> int:
        count = effective_video_facts(self.value)["frame_count"]
        if count is None:
            raise ValueError("VIDEO frame count is unknown")
        return cast(int, count)

    def get_frame_rate(self) -> Fraction:
        fps = effective_video_facts(self.value)["fps"]
        if fps is None:
            raise ValueError("VIDEO frame rate is unknown")
        return cast(Fraction, fps)

    def get_bit_depth(self) -> int:
        return cast(int, mapping(self.value["probe"], "probe")["bit_depth"])

    def get_color_space(self) -> str:
        return str(mapping(self.value["probe"], "probe")["color_space"])

    def get_container_format(self) -> str | None:
        return cast("str | None", mapping(self.value["probe"], "probe")["container"])

    def get_active_trim_window(self) -> tuple[float, float]:
        start, duration = source_trim_window(self.value)
        return float(start), float(duration) if duration is not None else 0.0

    def get_stream_source(self) -> str | io.BytesIO:
        source = video_source(self.value)
        if isinstance(source, bytes):
            return io.BytesIO(source)
        if isinstance(source, AssetRef):
            return str(source.local_path())
        raise TypeError("VIDEO source has no local file binding")

    def get_components(self) -> object:
        value = bind_video_value(self.value, resolver_from_env(), for_audio_extraction=True)
        parts = disassemble_video(value)
        torch = cast(Any, importlib.import_module("torch"))
        latest = cast(Any, importlib.import_module("comfy_api.latest"))
        images = cast(np.ndarray, parts["images"])
        audio = parts["audio"]
        if audio is not None:
            audio = TorchAudio(mapping(audio, "audio"))
        return latest.Types.VideoComponents(
            images=torch.from_numpy(images[..., :3]),
            frame_rate=self.get_frame_rate(),
            audio=audio,
            alpha=torch.from_numpy(images[..., 3]) if images.shape[-1] == 4 else None,
        )

    def as_trimmed(
        self, start_time: float, duration: float, strict_duration: bool = False
    ) -> _VideoValue:
        return type(self)(
            edit_video(
                self.value,
                {
                    "trim": {"start_time": start_time, "duration": duration},
                    "strict_duration": strict_duration,
                },
            )
        )

    def as_cropped(self, x: int, y: int, width: int, height: int) -> _VideoValue:
        return type(self)(
            edit_video(self.value, {"crop": {"x": x, "y": y, "width": width, "height": height}})
        )

    def save_to(
        self,
        path: str | BinaryIO,
        format: object = "auto",
        codec: object = "auto",
        metadata: Mapping[str, object] | None = None,
        bit_depth: int | None = None,
        crf: int | None = None,
        color_space: str | None = None,
        preset: str | None = None,
    ) -> None:
        if bit_depth is not None and bit_depth != self.get_bit_depth():
            raise ValueError("changing carried VIDEO precision requires an explicit conversion")
        if color_space is not None and color_space != self.get_color_space():
            raise ValueError("changing carried VIDEO color space requires an explicit conversion")
        if preset is not None:
            raise ValueError("VIDEO encoder presets are not supported")
        kind, name = str(getattr(format, "value", format)), str(getattr(codec, "value", codec))
        if isinstance(path, str):
            temporary: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=Path(path).parent, prefix=".dinkster-video-", delete=False
                ) as handle:
                    temporary = handle.name
                    save_video_stream(
                        self.value,
                        cast(BinaryIO, handle),
                        container=kind,
                        codec=name,
                        crf=crf,
                        metadata=metadata,
                    )
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
        else:
            save_video_stream(
                self.value, path, container=kind, codec=name, crf=crf, metadata=metadata
            )


def _upstream_video(value: object) -> object:
    impl = cast(Any, importlib.import_module("comfy_api.input_impl"))

    class Video(_VideoValue, impl.VideoFromFile):
        pass

    return Video(value)


def register_video_type(registry: TypeRegistry, type_id: str) -> None:
    fingerprint = video_fingerprint(type_id)
    registry.register(
        type_id,
        coerce=lambda obj: obj if isinstance(obj, _VideoValue) else _upstream_video(_native(obj)),
        encode=lambda obj: encode_video(_native(obj)),
        decode=lambda data: _upstream_video(decode_video(data)),
        fingerprint=lambda obj: fingerprint(_native(obj)),
        meta=lambda obj: video_meta(_native(obj)),
        validate_encoded_buffer=validate_video_encoded,
    )
