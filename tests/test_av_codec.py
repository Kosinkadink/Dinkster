"""Torch-free proofs for the comfy.AUDIO and comfy.VIDEO byte contracts."""

from __future__ import annotations

import io
import struct
import sys
import types
from dataclasses import dataclass
from functools import cache

import numpy as np
import pytest
from dinkster_assets import AssetRef, digest_bytes
from dinkster_values import (
    TypeRegistry,
    audio_fingerprint,
    audio_meta,
    decode_audio,
    decode_video,
    encode_audio,
    encode_video,
    render_audio_wav,
    render_video_original,
    validate_video_encoded,
    video_fingerprint,
    video_from_source,
    video_meta,
    video_rendition_mime,
)

from dinkster.comfy_compose import (
    COMFY_AUDIO_TYPE,
    COMFY_VIDEO_TYPE,
    register_comfy_host_types,
)
from tests.test_video_probe import _encode


@dataclass
class FakeTensor:
    array: np.ndarray

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array


@cache
def mp4_bytes() -> bytes:
    return _encode("mp4", "libx264", "yuv420p")


@cache
def webm_bytes() -> bytes:
    return _encode("webm", "libvpx-vp9", "yuv420p")


def test_audio_round_trip_fixes_float32_bct_contract() -> None:
    waveform = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    decoded = decode_audio(encode_audio({"waveform": waveform, "sample_rate": 48_000}))
    assert isinstance(decoded, dict)
    assert decoded["sample_rate"] == 48_000
    loaded = np.asarray(decoded["waveform"])
    assert loaded.dtype == np.float32
    assert loaded.shape == (2, 3, 4)
    assert np.array_equal(loaded, waveform.astype(np.float32))


def test_audio_fingerprint_is_numpy_torch_form_independent() -> None:
    waveform = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(1, 2, 6)
    numpy_form = {"waveform": waveform, "sample_rate": 6}
    tensor_form = {"waveform": FakeTensor(waveform), "sample_rate": 6}
    fingerprint = audio_fingerprint("comfy.AUDIO")
    assert encode_audio(numpy_form) == encode_audio(tensor_form)
    assert fingerprint(numpy_form) == fingerprint(tensor_form)
    assert audio_meta(numpy_form) == {
        "sample_rate": 6,
        "shape": (1, 2, 6),
        "duration": 1.0,
    }


def test_audio_refuses_invalid_shape_rate_and_payload() -> None:
    with pytest.raises(ValueError, match="B, C, T"):
        encode_audio({"waveform": np.zeros((2, 3)), "sample_rate": 48_000})
    with pytest.raises(ValueError, match="positive uint64"):
        encode_audio({"waveform": np.zeros((1, 1, 1)), "sample_rate": 0})
    with pytest.raises(ValueError, match="shorter"):
        decode_audio(b"short")


def test_audio_wav_is_exact_pcm16_little_endian_and_sample_interleaved() -> None:
    waveform = np.array(
        [[[-2.0, -1.0, -0.5, 0.0], [0.5, 1.0, 2.0, 0.25]]],
        dtype=np.float32,
    )
    wav = render_audio_wav({"waveform": waveform, "sample_rate": 8_000})
    assert struct.unpack("<4sI4s4sIHHIIHH4sI", wav[:44]) == (
        b"RIFF",
        52,
        b"WAVE",
        b"fmt ",
        16,
        1,
        2,
        8_000,
        32_000,
        4,
        16,
        b"data",
        16,
    )
    assert struct.unpack("<8h", wav[44:]) == (
        -32768,
        16384,
        -32768,
        32767,
        -16384,
        32767,
        0,
        8192,
    )


@pytest.mark.parametrize(
    ("waveform", "sample_rate", "message"),
    [
        (np.zeros((0, 1, 1), dtype=np.float32), 1, "nonempty batch"),
        (np.zeros((1, 0, 1), dtype=np.float32), 1, "nonempty channels"),
        (np.zeros((1, 1, 0), dtype=np.float32), 1, "nonempty channels"),
        (np.array([[[np.nan]]], dtype=np.float32), 1, "finite samples"),
        (np.array([[[np.inf]]], dtype=np.float32), 1, "finite samples"),
        (np.zeros((1, 1, 1), dtype=np.float32), 1 << 32, "exceeds uint32"),
        (np.zeros((1, 2, 1), dtype=np.float32), 1 << 30, "byte rate exceeds uint32"),
    ],
)
def test_audio_wav_refuses_invalid_cardinality_samples_and_rate(
    waveform: np.ndarray, sample_rate: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        render_audio_wav({"waveform": waveform, "sample_rate": sample_rate})


def test_audio_wav_refuses_uint16_and_uint32_size_overflow_before_rendering() -> None:
    scalar = np.zeros(1, dtype=np.float32)
    too_many_channels = np.lib.stride_tricks.as_strided(
        scalar, shape=(1, 32768, 1), strides=(0, 0, 0)
    )
    with pytest.raises(ValueError, match="uint16"):
        render_audio_wav({"waveform": too_many_channels, "sample_rate": 1})

    too_many_frames = np.lib.stride_tricks.as_strided(
        scalar, shape=(1, 1, (1 << 31)), strides=(0, 0, 0)
    )
    with pytest.raises(ValueError, match="RIFF size exceeds uint32"):
        render_audio_wav({"waveform": too_many_frames, "sample_rate": 1})


@pytest.mark.parametrize("container,data", [("mp4", mp4_bytes()), ("webm", webm_bytes())])
def test_video_round_trip_preserves_raw_container_bytes(container: str, data: bytes) -> None:
    value = {"container": container, "bytes": data}
    encoded = encode_video(value)
    assert encoded.startswith(b"DINKSTER-VIDEO\x02")
    assert decode_video(encoded)["source"] == data
    assert decode_video(data)["source"] == data
    assert video_meta(value)["container"] == container
    assert video_meta(value)["byte_size"] == len(data)


def test_video_fingerprint_is_wrapper_independent() -> None:
    data = mp4_bytes()
    value = {"container": "mp4", "bytes": data}
    decoded = decode_video(encode_video(value))
    assert video_fingerprint("comfy.VIDEO")(value) == video_fingerprint("comfy.VIDEO")(decoded)


def test_video_refuses_unknown_mismatched_and_unrecognized_containers() -> None:
    with pytest.raises(ValueError, match="does not match"):
        encode_video({"container": "mov", "bytes": mp4_bytes()})
    with pytest.raises(ValueError, match="does not match"):
        encode_video({"container": "webm", "bytes": mp4_bytes()})
    with pytest.raises(ValueError):
        decode_video(b"not a container")
    with pytest.raises(ValueError):
        decode_video(b"\x00\x00\x00\x18ftypqt  quicktime")


@pytest.mark.parametrize(
    ("container", "data", "mime"),
    [("mp4", mp4_bytes(), "video/mp4"), ("webm", webm_bytes(), "video/webm")],
)
def test_video_original_rendition_preserves_validated_bytes_and_metadata_mime(
    container: str, data: bytes, mime: str
) -> None:
    value = {"container": container, "bytes": data}
    assert render_video_original(value) == data
    metadata = video_meta(value)
    assert video_rendition_mime(metadata) == mime
    validate_video_encoded(data, metadata)
    with pytest.raises(ValueError, match="metadata container"):
        video_rendition_mime({"container": "unsupported"})


def test_video_encoded_validator_refuses_metadata_that_misidentifies_bytes() -> None:
    data = webm_bytes()
    with pytest.raises(ValueError, match="container does not match"):
        validate_video_encoded(data, {"container": "mp4", "byte_size": len(data)})
    with pytest.raises(ValueError, match="byte_size does not match"):
        validate_video_encoded(data, {"container": "webm", "byte_size": len(data) + 1})
    with pytest.raises(ValueError, match="byte_size must be an int"):
        validate_video_encoded(data, {"container": "webm", "byte_size": True})


@pytest.mark.parametrize("asset_backed", [False, True])
@pytest.mark.parametrize("residency", ["ram", "ram@remote-video"])
def test_video_cost_validates_bytes_independently_of_worker_namespace(
    asset_backed: bool, residency: str
) -> None:
    data = mp4_bytes()
    video = video_from_source(data)
    if asset_backed:
        video["source"] = AssetRef(digest_bytes(data), "source.mp4", len(data)).to_wire()
    encoded = encode_video(video)
    metadata = dict(video_meta(video))
    amount = 0 if asset_backed else len(data)
    metadata["cost"] = {residency: amount}
    validate_video_encoded(encoded, metadata)
    assert metadata["cost"] == {residency: amount}
    metadata["cost"] = {residency: amount + 1}
    with pytest.raises(ValueError, match="VIDEO metadata cost"):
        validate_video_encoded(encoded, metadata)


@pytest.mark.parametrize(
    "cost",
    [
        None,
        [],
        {},
        {"disk@remote-video": 0},
        {"ram": 0, "ram@remote-video": 0},
        {"ram@remote-video": False},
        {"ram@remote-video": 0.0},
        {"ram@remote-video": "0"},
        {"ram@remote-video": -1},
    ],
)
def test_video_cost_rejects_malformed_remote_accounting(cost: object) -> None:
    data = mp4_bytes()
    video = video_from_source(data)
    video["source"] = AssetRef(digest_bytes(data), "source.mp4", len(data)).to_wire()
    metadata = {**video_meta(video), "cost": cost}
    with pytest.raises(ValueError, match="VIDEO metadata cost"):
        validate_video_encoded(encode_video(video), metadata)


def test_audio_crosses_between_compat_torch_and_torch_free_forms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_compat_comfy.audio import register_audio_type

    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = FakeTensor  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    host = TypeRegistry()
    register_comfy_host_types(host)
    worker = TypeRegistry()
    register_audio_type(worker, COMFY_AUDIO_TYPE)
    waveform = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(1, 2, 4)
    torch_form = {"waveform": FakeTensor(waveform), "sample_rate": 4}

    worker_bytes = worker.spec(COMFY_AUDIO_TYPE).encode(torch_form)
    host_form = host.spec(COMFY_AUDIO_TYPE).decode(worker_bytes)
    assert isinstance(host_form, dict)
    assert np.array_equal(host_form["waveform"], waveform)
    assert host_form["sample_rate"] == 4

    host_bytes = host.spec(COMFY_AUDIO_TYPE).encode(host_form)
    worker_form = worker.spec(COMFY_AUDIO_TYPE).decode(host_bytes)
    assert isinstance(worker_form, dict)
    assert isinstance(worker_form["waveform"], FakeTensor)
    assert np.array_equal(worker_form["waveform"].array, waveform)
    assert (
        worker.wrap(COMFY_AUDIO_TYPE, torch_form).fingerprint
        == host.wrap(COMFY_AUDIO_TYPE, host_form).fingerprint
    )


class FakeVideoInput:
    def __init__(self, source: io.BytesIO | bytes) -> None:
        self.data = source.getvalue() if isinstance(source, io.BytesIO) else source

    def save_to(self, destination: io.BytesIO) -> None:
        raise AssertionError("the VIDEO boundary must never call upstream save_to")

    def get_stream_source(self) -> io.BytesIO:
        return io.BytesIO(self.data)

    def get_active_trim_window(self) -> tuple[int, int]:
        return 0, 0


class FakeVideoFromComponents:
    def __init__(self, components: object, bit_depth: int = 8) -> None:
        self.components = components
        self.bit_depth = bit_depth

    def get_components(self) -> object:
        return self.components

    def get_bit_depth(self) -> int:
        return self.bit_depth


def test_pinned_component_video_defaults_to_srgb(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster_compat_comfy.video import register_video_type

    input_impl = types.ModuleType("comfy_api.input_impl")
    input_impl.VideoFromFile = FakeVideoInput  # pyright: ignore[reportAttributeAccessIssue]
    input_impl.VideoFromComponents = (  # pyright: ignore[reportAttributeAccessIssue]
        FakeVideoFromComponents
    )
    monkeypatch.setitem(sys.modules, "comfy_api.input_impl", input_impl)
    components = types.SimpleNamespace(
        images=FakeTensor(np.zeros((2, 4, 6, 3), dtype=np.float32)),
        alpha=None,
        audio=None,
        frame_rate=12,
    )
    value = FakeVideoFromComponents(components)
    worker = TypeRegistry()
    register_video_type(worker, COMFY_VIDEO_TYPE)

    decoded = decode_video(worker.spec(COMFY_VIDEO_TYPE).encode(value))
    assert decoded["components"]["color_space"] == "sRGB"  # type: ignore[index]


def test_video_crosses_between_upstream_family_and_torch_free_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_compat_comfy.video import register_video_type

    comfy_api = types.ModuleType("comfy_api")
    input_impl = types.ModuleType("comfy_api.input_impl")
    input_impl.VideoFromFile = FakeVideoInput  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.input_impl", input_impl)

    host = TypeRegistry()
    register_comfy_host_types(host)
    worker = TypeRegistry()
    register_video_type(worker, COMFY_VIDEO_TYPE)
    assert worker.spec(COMFY_VIDEO_TYPE).validate_encoded_buffer is validate_video_encoded
    upstream = FakeVideoInput(mp4_bytes())

    worker_bytes = worker.spec(COMFY_VIDEO_TYPE).encode(upstream)
    host_form = host.spec(COMFY_VIDEO_TYPE).decode(worker_bytes)
    assert host_form == decode_video(mp4_bytes())

    host_bytes = host.spec(COMFY_VIDEO_TYPE).encode(host_form)
    worker_form = worker.spec(COMFY_VIDEO_TYPE).decode(host_bytes)
    assert isinstance(worker_form, FakeVideoInput)
    assert worker_form.get_stream_source().getvalue() == mp4_bytes()
    assert (
        worker.wrap(COMFY_VIDEO_TYPE, upstream).fingerprint
        == host.wrap(COMFY_VIDEO_TYPE, host_form).fingerprint
    )


def test_compat_video_refuses_unknown_subclasses(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster_compat_comfy.video import register_video_type

    input_impl = types.ModuleType("comfy_api.input_impl")
    input_impl.VideoFromFile = FakeVideoInput  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "comfy_api.input_impl", input_impl)

    class UnknownVideo(FakeVideoInput):
        pass

    worker = TypeRegistry()
    register_video_type(worker, COMFY_VIDEO_TYPE)
    with pytest.raises(TypeError, match="unsupported VideoInput subclass"):
        worker.spec(COMFY_VIDEO_TYPE).encode(UnknownVideo(mp4_bytes()))


def test_av_registration_wiring_covers_host_translation_and_native() -> None:
    from dinkster_compat_comfy.native import register_native_types
    from dinkster_compat_comfy.translate import CompatTranslation

    host = TypeRegistry()
    register_comfy_host_types(host)
    assert host.spec(COMFY_AUDIO_TYPE).declared_codec
    assert host.spec(COMFY_VIDEO_TYPE).declared_codec
    assert host.spec(COMFY_VIDEO_TYPE).validate_encoded_buffer is validate_video_encoded
    audio_value = host.wrap(
        COMFY_AUDIO_TYPE,
        {"waveform": np.zeros((1, 1, 1), dtype=np.float32), "sample_rate": 1},
    )
    video_value = host.wrap(COMFY_VIDEO_TYPE, {"container": "mp4", "bytes": mp4_bytes()})
    assert [(s.kind, s.mime, s.default) for s in host.renditions_of(COMFY_AUDIO_TYPE)] == [
        ("wav", "audio/wav", True),
        ("waveform", "image/png", False),
        ("window", "audio/wav", False),
    ]
    video_spec = host.renditions_of(COMFY_VIDEO_TYPE)[0]
    assert (video_spec.kind, video_spec.default) == ("original", True)
    assert video_spec.mime_for(video_value.meta.entries) == "video/mp4"
    assert host.render(audio_value).mime == "audio/wav"
    assert host.render(video_value).data == mp4_bytes()

    translated = TypeRegistry()
    translation = CompatTranslation()
    translation.opaque_types.update({COMFY_AUDIO_TYPE, COMFY_VIDEO_TYPE})
    translation.register_types(translated)
    assert translated.spec(COMFY_AUDIO_TYPE).declared_codec
    assert translated.spec(COMFY_VIDEO_TYPE).declared_codec
    assert translated.spec(COMFY_VIDEO_TYPE).validate_encoded_buffer is validate_video_encoded

    native = TypeRegistry()
    register_native_types(native)
    assert native.spec(COMFY_AUDIO_TYPE).declared_codec
    assert native.spec(COMFY_VIDEO_TYPE).declared_codec
    assert native.spec(COMFY_VIDEO_TYPE).validate_encoded_buffer is validate_video_encoded


def test_av_preview_intent_is_explicit_on_test_owned_output_schemas() -> None:
    from dinkster_schema import Node, NodeSchema, OutputSpec, TypeExpr

    audio = TypeExpr.concrete(COMFY_AUDIO_TYPE)
    video = TypeExpr.concrete(COMFY_VIDEO_TYPE)

    class PreviewAV(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.preview_av",
                outputs=(
                    OutputSpec("audio", audio, preview=True),
                    OutputSpec("video", video, preview=True),
                ),
            )

    class OrdinaryAV(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="test.ordinary_av",
                outputs=(OutputSpec("audio", audio), OutputSpec("video", video)),
            )

    assert [output.preview for output in PreviewAV.define_schema().outputs] == [True, True]
    assert [output.preview for output in OrdinaryAV.define_schema().outputs] == [False, False]
