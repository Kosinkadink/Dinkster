"""Server-side audio and webcam capture through injected providers.

The pinned ComfyUI RecordAudio and WebcamCapture capture in the browser at
queue time and upload the result; the server side only loads that uploaded
file. Dinkster captures on a server-side device instead, through a provider the
deployment injects via DINKSTER_AUDIO_CAPTURE_PROVIDER and
DINKSTER_VIDEO_CAPTURE_PROVIDER ("module:attribute" naming a zero-argument
factory). Providers are loaded lazily per execution, never during schema
discovery, so discovering the node surface touches no device and tests run
against fake providers without hardware.
"""
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    CORE_INT,
    ComboWidget,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
)

from .audio import MAX_SAMPLE_RATE
from .image import IMAGE_TYPE, MAX_IMAGE_ARRAY_BYTES, MAX_IMAGE_DIMENSION
from .video import AUDIO_TYPE, MAX_DECODED_AUDIO_BYTES

AUDIO = TypeExpr.concrete(AUDIO_TYPE)
IMAGE = TypeExpr.concrete(IMAGE_TYPE)
COMBO = TypeExpr.concrete(CORE_COMBO)
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)

AUDIO_CAPTURE_PROVIDER_ENV = "DINKSTER_AUDIO_CAPTURE_PROVIDER"
VIDEO_CAPTURE_PROVIDER_ENV = "DINKSTER_VIDEO_CAPTURE_PROVIDER"

AUDIO_DEVICE_CHOICES_ID = "dinkster.devices.audio_inputs"
VIDEO_DEVICE_CHOICES_ID = "dinkster.devices.video_inputs"

MAX_RECORD_SECONDS = 3600.0
MAX_CAPTURE_TIMEOUT_SECONDS = 600.0

_DEVICE_DOC = "Capture device id. Empty selects the first available device."


class CaptureError(ValueError):
    """A capture request failed. Providers raise the typed subclasses so
    workflows can distinguish permission, discovery, cancellation, and
    timeout failures; any other provider misbehavior is reported as this
    base class."""


class CapturePermissionDenied(CaptureError):
    """The operating system or device denied access to the capture device."""


class CaptureDeviceNotFound(CaptureError):
    """The requested capture device does not exist."""


class CaptureCancelled(CaptureError):
    """The capture was cancelled before it completed."""


class CaptureTimeout(CaptureError):
    """The device produced no result within the allowed time."""


@dataclass(frozen=True)
class AudioCaptureDevice:
    """One recordable audio device as declared by a provider.

    An empty sample_rates tuple means the device accepts any rate."""

    id: str
    label: str
    sample_rates: tuple[int, ...] = ()
    max_channels: int = 2


@dataclass(frozen=True)
class VideoCaptureDevice:
    """One frame-capturable video device as declared by a provider."""

    id: str
    label: str
    max_width: int = MAX_IMAGE_DIMENSION
    max_height: int = MAX_IMAGE_DIMENSION


class AudioCaptureProvider(Protocol):
    """Duck-typed audio capture backend.

    record() returns a float32 array shaped (channels, samples) with at
    least one sample and never more than round(seconds * sample_rate);
    a device may stop early and return fewer. Failures are raised as the
    typed CaptureError subclasses."""

    def devices(self) -> Sequence[AudioCaptureDevice]: ...

    def record(
        self, device_id: str, *, seconds: float, sample_rate: int, channels: int
    ) -> np.ndarray: ...


class VideoCaptureProvider(Protocol):
    """Duck-typed video frame-capture backend.

    capture_frame() returns a uint8 array shaped (height, width, 3); a
    nonzero requested width or height is binding. Failures are raised as
    the typed CaptureError subclasses."""

    def devices(self) -> Sequence[VideoCaptureDevice]: ...

    def capture_frame(
        self, device_id: str, *, width: int, height: int, timeout_seconds: float
    ) -> np.ndarray: ...


def _load_provider(env_var: str, capture_method: str) -> Any:
    spec = os.environ.get(env_var, "")
    if not spec:
        raise CaptureError(
            f"no capture provider is configured; set {env_var} to "
            "module:attribute naming a zero-argument provider factory"
        )
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise CaptureError(f"{env_var} must be module:attribute, got {spec!r}")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as error:
        raise CaptureError(
            f"capture provider {spec!r} from {env_var} failed to load: {error}"
        ) from error
    provider = factory()
    for name in ("devices", capture_method):
        if not callable(getattr(provider, name, None)):
            raise CaptureError(f"capture provider {spec!r} does not implement {name}()")
    return provider


def _capture_device_ids(env_var: str, capture_method: str) -> tuple[str, ...]:
    """Enumerate the configured provider's device ids for the choices route.

    An unset provider is an empty list, not an error: deployments without
    capture hardware still compose this pack, and the dropdown simply has
    nothing to offer. A configured provider that fails to load or enumerate
    raises, which the choices route reports as a fetch error."""
    if not os.environ.get(env_var, ""):
        return ()
    provider = _load_provider(env_var, capture_method)
    return tuple(str(device.id) for device in provider.devices())


def capture_device_choices() -> Mapping[str, Any]:
    """Lazy choice providers for the capture device dropdowns, invoked once
    per /api/choices fetch and never at startup or discovery."""
    return {
        AUDIO_DEVICE_CHOICES_ID: lambda: _capture_device_ids(AUDIO_CAPTURE_PROVIDER_ENV, "record"),
        VIDEO_DEVICE_CHOICES_ID: lambda: _capture_device_ids(
            VIDEO_CAPTURE_PROVIDER_ENV, "capture_frame"
        ),
    }


def _pick_device(devices: Iterable[Any], device_id: str, kind: str) -> Any:
    declared = tuple(devices)
    if not declared:
        raise CaptureDeviceNotFound(f"no {kind} capture devices are available")
    if not device_id:
        return declared[0]
    for device in declared:
        if device.id == device_id:
            return device
    available = ", ".join(repr(device.id) for device in declared)
    raise CaptureDeviceNotFound(
        f"{kind} capture device {device_id!r} not found; available: {available}"
    )


def _validated_waveform(recorded: object, *, channels: int, sample_limit: int) -> np.ndarray:
    if not isinstance(recorded, np.ndarray) or recorded.dtype != np.float32:
        raise CaptureError("audio capture provider must return a float32 numpy array")
    if recorded.ndim != 2 or recorded.shape[0] != channels:
        raise CaptureError(
            f"recorded waveform must be shaped ({channels}, samples), got {recorded.shape}"
        )
    samples = recorded.shape[1]
    if samples < 1:
        raise CaptureError("audio capture provider returned an empty recording")
    if samples > sample_limit:
        raise CaptureError(
            f"audio capture provider returned {samples} samples, above the "
            f"{sample_limit} the requested duration allows"
        )
    if not np.all(np.isfinite(recorded)):
        raise CaptureError("audio capture provider returned non-finite samples")
    return np.ascontiguousarray(recorded)


def _validated_dimension(name: str, value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_IMAGE_DIMENSION
    ):
        raise ValueError(
            f"{name} must be an integer in 0..{MAX_IMAGE_DIMENSION} (0 = device native)"
        )
    return value


def _validated_frame(frame: object, *, width: int, height: int) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise CaptureError("video capture provider must return a uint8 numpy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise CaptureError(f"captured frame must be shaped (height, width, 3), got {frame.shape}")
    frame_height, frame_width = int(frame.shape[0]), int(frame.shape[1])
    if frame_height < 1 or frame_width < 1:
        raise CaptureError("video capture provider returned an empty frame")
    if frame_width > MAX_IMAGE_DIMENSION or frame_height > MAX_IMAGE_DIMENSION:
        raise CaptureError(f"captured frame exceeds {MAX_IMAGE_DIMENSION} pixels per side")
    if width and frame_width != width:
        raise CaptureError(f"requested width {width} but the provider captured {frame_width}")
    if height and frame_height != height:
        raise CaptureError(f"requested height {height} but the provider captured {frame_height}")
    if frame_width * frame_height * 3 * np.dtype(np.float32).itemsize > MAX_IMAGE_ARRAY_BYTES:
        raise CaptureError(f"captured frame exceeds the {MAX_IMAGE_ARRAY_BYTES}-byte image budget")
    return frame


class RecordAudio(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.record_audio",
            display_name="Record Audio",
            category="audio",
            description="Record audio from a server-side capture device.",
            inputs=(
                InputSpec(
                    "device",
                    COMBO,
                    required=False,
                    default="",
                    doc=_DEVICE_DOC,
                    widget=ComboWidget(
                        remote_route=f"/api/choices/{AUDIO_DEVICE_CHOICES_ID}",
                        refresh_button=True,
                    ),
                ),
                InputSpec(
                    "seconds",
                    FLOAT,
                    required=False,
                    default=10.0,
                    widget=NumberWidget(min=0.1, max=MAX_RECORD_SECONDS, step=0.1),
                ),
                InputSpec(
                    "sample_rate",
                    INT,
                    required=False,
                    default=44_100,
                    widget=NumberWidget(min=1, max=MAX_SAMPLE_RATE, step=1),
                    advanced=True,
                ),
                InputSpec(
                    "channels",
                    INT,
                    required=False,
                    default=1,
                    widget=NumberWidget(min=1, max=2, step=1),
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("audio", AUDIO, preview=True),),
            idempotent=False,
            search_terms=("record audio", "microphone", "capture audio", "audio input"),
        )

    @classmethod
    def execute(
        cls,
        *,
        device: object = "",
        seconds: float = 10.0,
        sample_rate: object = 44_100,
        channels: object = 1,
    ) -> Mapping[str, object]:
        if not isinstance(device, str):
            raise ValueError(f"device must be a string id, got {type(device).__name__}")
        length = float(seconds)
        if not math.isfinite(length) or not 0.0 < length <= MAX_RECORD_SECONDS:
            raise ValueError(
                f"seconds must be finite, positive, and at most {MAX_RECORD_SECONDS}, "
                f"got {seconds!r}"
            )
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or not 1 <= sample_rate <= MAX_SAMPLE_RATE
        ):
            raise ValueError(f"sample_rate must be an integer in 1..{MAX_SAMPLE_RATE}")
        if isinstance(channels, bool) or not isinstance(channels, int) or channels not in (1, 2):
            raise ValueError("channels must be the integer 1 (mono) or 2 (stereo)")
        sample_limit = int(round(length * sample_rate))
        if sample_limit < 1:
            raise ValueError("seconds and sample_rate must allow at least one sample")
        if sample_limit * channels * np.dtype(np.float32).itemsize > MAX_DECODED_AUDIO_BYTES:
            raise ValueError(
                f"recording exceeds the {MAX_DECODED_AUDIO_BYTES}-byte waveform budget; "
                "reduce seconds, sample_rate, or channels"
            )
        provider = _load_provider(AUDIO_CAPTURE_PROVIDER_ENV, "record")
        selected = _pick_device(provider.devices(), device, "audio")
        if selected.sample_rates and sample_rate not in selected.sample_rates:
            supported = ", ".join(str(rate) for rate in selected.sample_rates)
            raise CaptureError(
                f"audio capture device {selected.id!r} does not support {sample_rate} Hz; "
                f"supported rates: {supported}"
            )
        if channels > selected.max_channels:
            raise CaptureError(
                f"audio capture device {selected.id!r} records at most "
                f"{selected.max_channels} channel(s)"
            )
        recorded = provider.record(
            selected.id,
            seconds=length,
            sample_rate=int(sample_rate),
            channels=int(channels),
        )
        waveform = _validated_waveform(recorded, channels=channels, sample_limit=sample_limit)
        return cls.outputs(
            audio={"waveform": waveform[np.newaxis, ...], "sample_rate": int(sample_rate)}
        )


class WebcamCapture(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="dinkster.webcam_capture",
            display_name="Webcam Capture",
            category="image/io",
            description="Capture one frame from a server-side camera device.",
            inputs=(
                InputSpec(
                    "device",
                    COMBO,
                    required=False,
                    default="",
                    doc=_DEVICE_DOC,
                    widget=ComboWidget(
                        remote_route=f"/api/choices/{VIDEO_DEVICE_CHOICES_ID}",
                        refresh_button=True,
                    ),
                ),
                InputSpec(
                    "width",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=MAX_IMAGE_DIMENSION, step=1),
                ),
                InputSpec(
                    "height",
                    INT,
                    required=False,
                    default=0,
                    widget=NumberWidget(min=0, max=MAX_IMAGE_DIMENSION, step=1),
                ),
                InputSpec(
                    "timeout_seconds",
                    FLOAT,
                    required=False,
                    default=10.0,
                    widget=NumberWidget(min=0.1, max=MAX_CAPTURE_TIMEOUT_SECONDS, step=0.1),
                    advanced=True,
                ),
            ),
            outputs=(OutputSpec("image", IMAGE, preview=True),),
            idempotent=False,
            search_terms=("webcam", "camera", "capture image", "camera frame"),
        )

    @classmethod
    def execute(
        cls,
        *,
        device: object = "",
        width: object = 0,
        height: object = 0,
        timeout_seconds: float = 10.0,
    ) -> Mapping[str, object]:
        if not isinstance(device, str):
            raise ValueError(f"device must be a string id, got {type(device).__name__}")
        width = _validated_dimension("width", width)
        height = _validated_dimension("height", height)
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or not 0.0 < timeout <= MAX_CAPTURE_TIMEOUT_SECONDS:
            raise ValueError(
                f"timeout_seconds must be finite, positive, and at most "
                f"{MAX_CAPTURE_TIMEOUT_SECONDS}, got {timeout_seconds!r}"
            )
        if width and height:
            requested = width * height * 3 * np.dtype(np.float32).itemsize
            if requested > MAX_IMAGE_ARRAY_BYTES:
                raise ValueError(
                    f"requested capture resolution exceeds the "
                    f"{MAX_IMAGE_ARRAY_BYTES}-byte image budget"
                )
        provider = _load_provider(VIDEO_CAPTURE_PROVIDER_ENV, "capture_frame")
        selected = _pick_device(provider.devices(), device, "video")
        if width > selected.max_width or height > selected.max_height:
            raise CaptureError(
                f"video capture device {selected.id!r} captures at most "
                f"{selected.max_width}x{selected.max_height}"
            )
        frame = provider.capture_frame(
            selected.id,
            width=int(width),
            height=int(height),
            timeout_seconds=timeout,
        )
        validated = _validated_frame(frame, width=int(width), height=int(height))
        image = np.ascontiguousarray(validated.astype(np.float32) / 255.0)[np.newaxis, ...]
        return cls.outputs(image=image)


CAPTURE_NODES: tuple[type[Node], ...] = (RecordAudio, WebcamCapture)

__all__ = [
    "AUDIO_CAPTURE_PROVIDER_ENV",
    "AUDIO_DEVICE_CHOICES_ID",
    "CAPTURE_NODES",
    "MAX_CAPTURE_TIMEOUT_SECONDS",
    "MAX_RECORD_SECONDS",
    "VIDEO_CAPTURE_PROVIDER_ENV",
    "VIDEO_DEVICE_CHOICES_ID",
    "AudioCaptureDevice",
    "AudioCaptureProvider",
    "CaptureCancelled",
    "CaptureDeviceNotFound",
    "CaptureError",
    "CapturePermissionDenied",
    "CaptureTimeout",
    "RecordAudio",
    "VideoCaptureDevice",
    "VideoCaptureProvider",
    "WebcamCapture",
    "capture_device_choices",
]
