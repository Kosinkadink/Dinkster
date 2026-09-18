"""Native server-side capture contracts, tested with injected fake providers."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import pytest
from dinkster_nodes_media_io import RecordAudio, WebcamCapture
from dinkster_nodes_media_io.capture import (
    AUDIO_CAPTURE_PROVIDER_ENV,
    AUDIO_DEVICE_CHOICES_ID,
    VIDEO_CAPTURE_PROVIDER_ENV,
    VIDEO_DEVICE_CHOICES_ID,
    AudioCaptureDevice,
    CaptureCancelled,
    CaptureDeviceNotFound,
    CaptureError,
    CapturePermissionDenied,
    CaptureTimeout,
    VideoCaptureDevice,
    capture_device_choices,
)
from dinkster_schema import ComboWidget, TypeExpr, schema_to_wire

PROVIDER_MODULE = "capture_io_test_providers"

MIC = AudioCaptureDevice(id="mic0", label="Fake Microphone")
CAMERA = VideoCaptureDevice(id="cam0", label="Fake Camera")


class FakeAudioProvider:
    def __init__(
        self,
        devices: Sequence[AudioCaptureDevice] = (MIC,),
        record: Callable[..., np.ndarray] | None = None,
    ) -> None:
        self._devices = tuple(devices)
        self._record = record
        self.calls: list[dict[str, object]] = []

    def devices(self) -> tuple[AudioCaptureDevice, ...]:
        return self._devices

    def record(
        self, device_id: str, *, seconds: float, sample_rate: int, channels: int
    ) -> np.ndarray:
        self.calls.append(
            {
                "device_id": device_id,
                "seconds": seconds,
                "sample_rate": sample_rate,
                "channels": channels,
            }
        )
        if self._record is not None:
            return self._record(
                device_id, seconds=seconds, sample_rate=sample_rate, channels=channels
            )
        samples = int(round(seconds * sample_rate))
        return np.zeros((channels, samples), dtype=np.float32)


class FakeVideoProvider:
    def __init__(
        self,
        devices: Sequence[VideoCaptureDevice] = (CAMERA,),
        capture: Callable[..., np.ndarray] | None = None,
    ) -> None:
        self._devices = tuple(devices)
        self._capture = capture
        self.calls: list[dict[str, object]] = []

    def devices(self) -> tuple[VideoCaptureDevice, ...]:
        return self._devices

    def capture_frame(
        self, device_id: str, *, width: int, height: int, timeout_seconds: float
    ) -> np.ndarray:
        self.calls.append(
            {
                "device_id": device_id,
                "width": width,
                "height": height,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self._capture is not None:
            return self._capture(
                device_id, width=width, height=height, timeout_seconds=timeout_seconds
            )
        shape = (height or 48, width or 64, 3)
        return np.full(shape, 128, dtype=np.uint8)


@pytest.fixture
def provider_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType(PROVIDER_MODULE)
    monkeypatch.setitem(sys.modules, PROVIDER_MODULE, module)
    return module


def _install_audio(
    monkeypatch: pytest.MonkeyPatch, module: types.ModuleType, provider: FakeAudioProvider
) -> None:
    module.make_audio_provider = lambda: provider  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setenv(AUDIO_CAPTURE_PROVIDER_ENV, f"{PROVIDER_MODULE}:make_audio_provider")


def _install_video(
    monkeypatch: pytest.MonkeyPatch, module: types.ModuleType, provider: FakeVideoProvider
) -> None:
    module.make_video_provider = lambda: provider  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setenv(VIDEO_CAPTURE_PROVIDER_ENV, f"{PROVIDER_MODULE}:make_video_provider")


def _audio_result(result: object) -> dict[str, object]:
    return cast("dict[str, object]", cast("dict[str, object]", result)["audio"])


def test_record_audio_returns_typed_audio_from_injected_provider(
    monkeypatch: pytest.MonkeyPatch, provider_module: types.ModuleType
) -> None:
    def ramp(_: str, *, seconds: float, sample_rate: int, channels: int) -> np.ndarray:
        samples = int(round(seconds * sample_rate))
        base = np.linspace(-1.0, 1.0, samples, dtype=np.float32)
        return np.stack([base] * channels)

    provider = FakeAudioProvider(record=ramp)
    _install_audio(monkeypatch, provider_module, provider)
    audio = _audio_result(RecordAudio.execute(seconds=0.5, sample_rate=16_000, channels=2))
    waveform = cast("np.ndarray", audio["waveform"])
    assert audio["sample_rate"] == 16_000
    assert waveform.shape == (1, 2, 8_000)
    assert waveform.dtype == np.float32
    assert waveform.flags["C_CONTIGUOUS"]
    assert np.array_equal(waveform[0, 0], np.linspace(-1.0, 1.0, 8_000, dtype=np.float32))
    assert provider.calls == [
        {"device_id": "mic0", "seconds": 0.5, "sample_rate": 16_000, "channels": 2}
    ]


def test_record_audio_accepts_a_recording_shorter_than_requested(
    monkeypatch: pytest.MonkeyPatch, provider_module: types.ModuleType
) -> None:
    def stopped_early(_: str, *, seconds: float, sample_rate: int, channels: int) -> np.ndarray:
        del seconds
        return np.zeros((channels, sample_rate // 4), dtype=np.float32)

    _install_audio(monkeypatch, provider_module, FakeAudioProvider(record=stopped_early))
    audio = _audio_result(RecordAudio.execute(seconds=2.0, sample_rate=8_000))
    assert cast("np.ndarray", audio["waveform"]).shape == (1, 1, 2_000)


def test_record_audio_selects_the_requested_device(
    monkeypatch: pytest.MonkeyPatch, provider_module: types.ModuleType
) -> None:
    devices = (MIC, AudioCaptureDevice(id="mic1", label="Second Microphone"))
    provider = FakeAudioProvider(devices=devices)
    _install_audio(monkeypatch, provider_module, provider)
    RecordAudio.execute(device="mic1", seconds=0.25, sample_rate=8_000)
    assert provider.calls[0]["device_id"] == "mic1"


def test_record_audio_unknown_device_fails_listing_available_ids(
    monkeypatch: pytest.MonkeyPatch, provider_module: types.ModuleType
) -> None:
    _install_audio(monkeypatch, provider_module, FakeAudioProvider())
    with pytest.raises(CaptureDeviceNotFound, match=r"'missing'.*available.*'mic0'"):
        RecordAudio.execute(device="missing")


def test_record_audio_with_no_devices_fails_closed(
    monkeypatch: pytest.MonkeyPatch, provider_module: types.ModuleType
) -> None:
    _install_audio(monkeypatch, provider_module, FakeAudioProvider(devices=()))
    with pytest.raises(CaptureDeviceNotFound, match="no audio capture devices"):
        RecordAudio.execute()


def test_record_audio_enforces_declared_device_capabilities(
    monkeypatch: pytest.MonkeyPatch, provider_module: types.ModuleType
) -> None:
    limited = AudioCaptureDevice(
        id="mic0", label="Limited Microphone", sample_rates=(48_000,), max_channels=1
    )
    provider = FakeAudioProvider(devices=(limited,))
    _install_audio(monkeypatch, provider_module, provider)
    with pytest.raises(CaptureError, match="does not support 44100 Hz.*48000"):
        RecordAudio.execute(sample_rate=44_100)
    with pytest.raises(CaptureError, match="at most 1 channel"):
        RecordAudio.execute(sample_rate=48_000, channels=2)
    assert provider.calls == []


@pytest.mark.parametrize(
    "error", [CapturePermissionDenied, CaptureCancelled, CaptureTimeout, CaptureDeviceNotFound]
)
def test_record_audio_typed_provider_failures_pass_through(
    monkeypatch: pytest.MonkeyPatch,
    provider_module: types.ModuleType,
    error: type[CaptureError],
) -> None:
    def failing(*_: object, **__: object) -> np.ndarray:
        raise error("capture backend failure")

    _install_audio(monkeypatch, provider_module, FakeAudioProvider(record=failing))
    with pytest.raises(error, match="capture backend failure"):
        RecordAudio.execute()


def test_record_audio_without_configured_provider_names_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(AUDIO_CAPTURE_PROVIDER_ENV, raising=False)
    with pytest.raises(CaptureError, match=AUDIO_CAPTURE_PROVIDER_ENV):
        RecordAudio.execute()


def test_record_audio_rejects_malformed_or_unloadable_provider_specs(
    monkeypatch: pytest.MonkeyPatch, provider_module: types.ModuleType
) -> None:
    monkeypatch.setenv(AUDIO_CAPTURE_PROVIDER_ENV, "not-a-module-attribute-pair")
    with pytest.raises(CaptureError, match="must be module:attribute"):
        RecordAudio.execute()
    monkeypatch.setenv(AUDIO_CAPTURE_PROVIDER_ENV, "capture_io_module_that_does_not_exist:make")
    with pytest.raises(CaptureError, match="failed to load"):
        RecordAudio.execute()
    monkeypatch.setenv(AUDIO_CAPTURE_PROVIDER_ENV, f"{PROVIDER_MODULE}:missing_factory")
    with pytest.raises(CaptureError, match="failed to load"):
        RecordAudio.execute()
    provider_module.make_broken = lambda: object()  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setenv(AUDIO_CAPTURE_PROVIDER_ENV, f"{PROVIDER_MODULE}:make_broken")
    with pytest.raises(CaptureError, match="does not implement devices"):
        RecordAudio.execute()


def test_record_audio_rejects_out_of_domain_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    # Bounds are revalidated before provider loading, so no provider is
    # configured here and a bounds failure must not mention one.
    monkeypatch.delenv(AUDIO_CAPTURE_PROVIDER_ENV, raising=False)
    for seconds in (0.0, -1.0, float("inf"), float("nan"), 3_600.1):
        with pytest.raises(ValueError, match="seconds must be finite"):
            RecordAudio.execute(seconds=seconds)
    for sample_rate in (0, -1, 192_001, True, 44_100.5, 44_100.0, "44100"):
        with pytest.raises(ValueError, match="sample_rate must be an integer"):
            RecordAudio.execute(sample_rate=cast("int", sample_rate))
    for channels in (0, 3, True, 1.0, 2.5, "1"):
        with pytest.raises(ValueError, match="channels must be the integer 1"):
            RecordAudio.execute(channels=cast("int", channels))
    with pytest.raises(ValueError, match="device must be a string"):
        RecordAudio.execute(device=cast("str", 7))


def test_record_audio_byte_budget_fails_before_any_provider_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AUDIO_CAPTURE_PROVIDER_ENV, "capture_io_module_that_does_not_exist:make")
    with pytest.raises(ValueError, match="waveform budget"):
        RecordAudio.execute(seconds=3_600.0, sample_rate=192_000, channels=2)


def test_record_audio_rejects_misbehaving_provider_results(
    monkeypatch: pytest.MonkeyPatch, provider_module: types.ModuleType
) -> None:
    cases: list[tuple[object, str]] = [
        (np.zeros((1, 100), dtype=np.float64), "float32"),
        ([[0.0] * 100], "float32"),
        (np.zeros(100, dtype=np.float32), "shaped"),
        (np.zeros((2, 100), dtype=np.float32), "shaped"),
        (np.zeros((1, 0), dtype=np.float32), "empty recording"),
        (np.zeros((1, 8_001), dtype=np.float32), "above the 8000"),
        (np.full((1, 100), np.nan, dtype=np.float32), "non-finite"),
    ]
    for result, message in cases:
        _install_audio(
            monkeypatch,
            provider_module,
            FakeAudioProvider(record=lambda *_, result=result, **__: cast("np.ndarray", result)),
        )
        with pytest.raises(CaptureError, match=message):
            RecordAudio.execute(seconds=1.0, sample_rate=8_000, channels=1)


def test_webcam_capture_returns_bhwc_image_from_injected_provider(
    monkeypatch: pytest.MonkeyPatch, provider_module: types.ModuleType
) -> None:
    gradient = np.arange(48 * 64 * 3, dtype=np.uint32).reshape(48, 64, 3) % 256
    frame = gradient.astype(np.uint8)
    provider = FakeVideoProvider(capture=lambda *_, **__: frame)
    _install_video(monkeypatch, provider_module, provider)
    result = cast("dict[str, object]", WebcamCapture.execute())
    image = cast("np.ndarray", result["image"])
    assert image.shape == (1, 48, 64, 3)
    assert image.dtype == np.float32
    assert image.flags["C_CONTIGUOUS"]
    assert np.array_equal(image[0], frame.astype(np.float32) / 255.0)
    assert provider.calls == [
        {"device_id": "cam0", "width": 0, "height": 0, "timeout_seconds": 10.0}
    ]


def test_webcam_capture_requested_resolution_is_binding(
    monkeypatch: pytest.MonkeyPatch, provider_module: types.ModuleType
) -> None:
    provider = FakeVideoProvider()
    _install_video(monkeypatch, provider_module, provider)
    result = cast("dict[str, object]", WebcamCapture.execute(width=32, height=24))
    assert cast("np.ndarray", result["image"]).shape == (1, 24, 32, 3)
    assert provider.calls[0] == {
        "device_id": "cam0",
        "width": 32,
        "height": 24,
        "timeout_seconds": 10.0,
    }
    wrong = FakeVideoProvider(capture=lambda *_, **__: np.zeros((24, 30, 3), dtype=np.uint8))
    _install_video(monkeypatch, provider_module, wrong)
    with pytest.raises(CaptureError, match="requested width 32"):
        WebcamCapture.execute(width=32, height=24)


def test_webcam_capture_device_discovery_failures(
    monkeypatch: pytest.MonkeyPatch, provider_module: types.ModuleType
) -> None:
    _install_video(monkeypatch, provider_module, FakeVideoProvider())
    with pytest.raises(CaptureDeviceNotFound, match=r"'missing'.*available.*'cam0'"):
        WebcamCapture.execute(device="missing")
    _install_video(monkeypatch, provider_module, FakeVideoProvider(devices=()))
    with pytest.raises(CaptureDeviceNotFound, match="no video capture devices"):
        WebcamCapture.execute()


def test_webcam_capture_enforces_declared_device_capabilities(
    monkeypatch: pytest.MonkeyPatch, provider_module: types.ModuleType
) -> None:
    small = VideoCaptureDevice(id="cam0", label="Small Camera", max_width=640, max_height=480)
    provider = FakeVideoProvider(devices=(small,))
    _install_video(monkeypatch, provider_module, provider)
    with pytest.raises(CaptureError, match="at most 640x480"):
        WebcamCapture.execute(width=1_280, height=720)
    assert provider.calls == []


def test_webcam_capture_rejects_out_of_domain_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(VIDEO_CAPTURE_PROVIDER_ENV, raising=False)
    for name, value in (
        ("width", -1),
        ("width", 16_385),
        ("width", 640.5),
        ("width", "640"),
        ("height", -1),
        ("height", True),
        ("height", 480.0),
    ):
        with pytest.raises(ValueError, match=f"{name} must be an integer"):
            WebcamCapture.execute(**cast("dict[str, Any]", {name: value}))
    for timeout in (0.0, -1.0, float("inf"), float("nan"), 600.1):
        with pytest.raises(ValueError, match="timeout_seconds must be finite"):
            WebcamCapture.execute(timeout_seconds=timeout)
    with pytest.raises(ValueError, match="device must be a string"):
        WebcamCapture.execute(device=cast("str", 7))


def test_webcam_capture_pixel_budget_fails_before_any_provider_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(VIDEO_CAPTURE_PROVIDER_ENV, "capture_io_module_that_does_not_exist:make")
    with pytest.raises(ValueError, match="image budget"):
        WebcamCapture.execute(width=16_384, height=16_384)


@pytest.mark.parametrize(
    "error", [CapturePermissionDenied, CaptureCancelled, CaptureTimeout, CaptureDeviceNotFound]
)
def test_webcam_capture_typed_provider_failures_pass_through(
    monkeypatch: pytest.MonkeyPatch,
    provider_module: types.ModuleType,
    error: type[CaptureError],
) -> None:
    def failing(*_: object, **__: object) -> np.ndarray:
        raise error("capture backend failure")

    _install_video(monkeypatch, provider_module, FakeVideoProvider(capture=failing))
    with pytest.raises(error, match="capture backend failure"):
        WebcamCapture.execute()


def test_webcam_capture_rejects_misbehaving_provider_results(
    monkeypatch: pytest.MonkeyPatch, provider_module: types.ModuleType
) -> None:
    cases: list[tuple[object, str]] = [
        (np.zeros((24, 32, 3), dtype=np.float32), "uint8"),
        ([[[0] * 3] * 32] * 24, "uint8"),
        (np.zeros((24, 32), dtype=np.uint8), "shaped"),
        (np.zeros((24, 32, 4), dtype=np.uint8), "shaped"),
        (np.zeros((0, 32, 3), dtype=np.uint8), "empty frame"),
        (np.zeros((1, 16_385, 3), dtype=np.uint8), "pixels per side"),
    ]
    for result, message in cases:
        _install_video(
            monkeypatch,
            provider_module,
            FakeVideoProvider(capture=lambda *_, result=result, **__: cast("np.ndarray", result)),
        )
        with pytest.raises(CaptureError, match=message):
            WebcamCapture.execute()


def test_schema_discovery_never_touches_a_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    poison = "capture_io_module_that_must_not_import:boom"
    monkeypatch.setenv(AUDIO_CAPTURE_PROVIDER_ENV, poison)
    monkeypatch.setenv(VIDEO_CAPTURE_PROVIDER_ENV, poison)
    for node in (RecordAudio, WebcamCapture):
        schema = node.define_schema()
        assert schema_to_wire(schema)
        assert schema.idempotent is False
    # The lazy-choices factory is equally inert: building the table loads
    # no provider; only a fetch does.
    assert set(capture_device_choices()) == {AUDIO_DEVICE_CHOICES_ID, VIDEO_DEVICE_CHOICES_ID}
    assert "capture_io_module_that_must_not_import" not in sys.modules


def test_capture_device_inputs_are_remote_combos() -> None:
    """The device inputs are remote combo widgets bound to the lazy device
    routes with a refresh control; the wire value stays a plain string and
    the empty default keeps first-available selection at execute time."""
    for node, choices_id in (
        (RecordAudio, AUDIO_DEVICE_CHOICES_ID),
        (WebcamCapture, VIDEO_DEVICE_CHOICES_ID),
    ):
        schema = node.define_schema()
        device = next(spec for spec in schema.inputs if spec.id == "device")
        assert device.type == TypeExpr.concrete("core.combo")
        assert device.default == ""
        assert isinstance(device.widget, ComboWidget)
        assert device.widget.remote_route == f"/api/choices/{choices_id}"
        assert device.widget.refresh_button is True


def test_capture_device_choices_enumerate_provider_devices(
    monkeypatch: pytest.MonkeyPatch, provider_module: types.ModuleType
) -> None:
    """Each fetch enumerates the injected providers' device ids in
    declaration order; an unconfigured env var yields an empty list (a
    registered source with nothing attached, not an error)."""
    monkeypatch.delenv(AUDIO_CAPTURE_PROVIDER_ENV, raising=False)
    monkeypatch.delenv(VIDEO_CAPTURE_PROVIDER_ENV, raising=False)
    providers = capture_device_choices()
    assert providers[AUDIO_DEVICE_CHOICES_ID]() == ()
    assert providers[VIDEO_DEVICE_CHOICES_ID]() == ()

    mics = (MIC, AudioCaptureDevice(id="mic1", label="Second Microphone"))
    _install_audio(monkeypatch, provider_module, FakeAudioProvider(devices=mics))
    _install_video(monkeypatch, provider_module, FakeVideoProvider())
    assert providers[AUDIO_DEVICE_CHOICES_ID]() == ("mic0", "mic1")
    assert providers[VIDEO_DEVICE_CHOICES_ID]() == ("cam0",)


def test_capture_device_choices_propagate_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken provider surfaces from the fetch (the server maps it to a
    502) - it must not silently serve an empty device list."""
    providers = capture_device_choices()
    monkeypatch.setenv(AUDIO_CAPTURE_PROVIDER_ENV, "capture_io_module_that_must_not_import:boom")
    with pytest.raises(CaptureError):
        providers[AUDIO_DEVICE_CHOICES_ID]()
