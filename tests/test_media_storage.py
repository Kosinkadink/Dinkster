from __future__ import annotations

import gc
import io
from multiprocessing.shared_memory import SharedMemory
from typing import Any, cast

import numpy as np
import pytest
from dinkster_values import COST_META_KEY, EncodedPayload
from dinkster_values.audio_codec import (
    audio_meta,
    decode_audio,
    encode_audio,
    render_audio_wav,
)
from dinkster_values.image_codec import (
    decode_image_array,
    encode_image_array,
    image_array_fingerprint,
    image_array_meta,
    render_image_png,
)
from dinkster_values.storage import BFLOAT16_FIELD, audio_input, encoded_storage_meta, image_input
from dinkster_values.video_codec import (
    decode_video,
    encode_video,
    validate_video_encoded,
    video_fingerprint,
    video_meta,
)
from dinkster_video import assemble_video
from PIL import Image


@pytest.mark.parametrize(
    ("dtype", "storage"),
    [("uint8", "uint8"), ("uint16", "uint16"), ("float16", "fp16"), ("float32", "fp32")],
)
def test_image_storage_roundtrip_cost_and_consumer_conversion(dtype: str, storage: str) -> None:
    array = np.arange(24, dtype=dtype).reshape(2, 2, 2, 3)
    encoded = encode_image_array(array)
    decoded = cast(np.ndarray, decode_image_array(encoded))
    np.testing.assert_array_equal(decoded, array)
    assert decoded.dtype == array.dtype
    assert image_array_meta(array)["storage_dtype"] == storage
    assert image_array_meta(array)[COST_META_KEY] == {"ram": array.nbytes}
    assert image_array_fingerprint("image")(array) == image_array_fingerprint("image")(decoded)
    normalized = cast(np.ndarray, image_input(decoded))
    expected = array.astype(np.float32)
    if dtype.startswith("uint"):
        expected /= np.iinfo(dtype).max
    np.testing.assert_array_equal(normalized, expected)
    assert normalized.dtype == np.float32
    if dtype == "float32":
        assert normalized is decoded


def test_bfloat16_storage_roundtrip_keeps_bits_until_consumed() -> None:
    bits = np.array([0, 0x3F00, 0x3F80, 0x3E01], dtype="<u2")
    array = bits.view([(BFLOAT16_FIELD, "<u2")]).reshape(1, 2, 2, 1)
    encoded = encode_image_array(array)
    decoded = cast(np.ndarray, decode_image_array(encoded))
    assert decoded.nbytes == 8
    assert image_array_meta(decoded)["storage_dtype"] == "bf16"
    np.testing.assert_array_equal(decoded.view("<u2").reshape(-1), bits)
    expected = (bits.astype(np.uint32) << 16).view(np.float32).reshape(array.shape)
    np.testing.assert_array_equal(image_input(decoded), expected)


def test_ten_bit_samples_in_uint16_remain_distinguishable() -> None:
    array = np.array([32768, 32832], dtype=np.uint16)
    decoded = decode_image_array(encode_image_array(array))
    np.testing.assert_array_equal(decoded, array)
    normalized = cast(np.ndarray, image_input(decoded))
    assert normalized[0] != normalized[1]
    assert normalized[1] - normalized[0] < 1 / 255


def test_integer_image_rendition_matches_normalized_samples() -> None:
    array = np.array([[[[0, 128, 255]]]], dtype=np.uint8)
    png = render_image_png(array)
    with Image.open(io.BytesIO(png)) as image:
        assert image.getpixel((0, 0)) == (0, 128, 255)
    assert png == render_image_png(array.astype(np.float32) / 255)


@pytest.mark.parametrize("dtype", ["int16", "float32"])
def test_audio_storage_roundtrip(dtype: str) -> None:
    array = np.arange(-12, 12, dtype=dtype).reshape(1, 2, 12)
    value = {"waveform": array, "sample_rate": 48000}
    decoded = cast(dict[str, Any], decode_audio(encode_audio(value)))
    stored = cast(dict[str, Any], decoded["source"])["pcm"]
    assert stored.dtype == array.dtype
    np.testing.assert_array_equal(stored, array)
    assert audio_meta(value)[COST_META_KEY] == {"ram": array.nbytes}
    expected = array.astype(np.float32) / 32768 if dtype == "int16" else array
    np.testing.assert_array_equal(
        cast(dict[str, Any], audio_input({"waveform": stored}))["waveform"], expected
    )


def test_pcm16_wav_rendition_preserves_all_sample_bits() -> None:
    samples = np.arange(-32768, 32768, dtype=np.int16).reshape(1, 1, -1)
    wav = render_audio_wav({"waveform": samples, "sample_rate": 48000})
    np.testing.assert_array_equal(np.frombuffer(wav[44:], dtype="<i2"), samples.reshape(-1))


def test_sliced_image_cost_includes_retained_backing_allocation() -> None:
    array = np.zeros((5, 8, 8, 3), dtype=np.uint8)
    assert image_array_meta(array[:1])[COST_META_KEY] == {"ram": array.nbytes}


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float16", "float32", "bf16"])
def test_compact_video_components_roundtrip_and_local_cost_validation(dtype: str) -> None:
    images = np.zeros((2, 4, 4, 3), dtype=[(BFLOAT16_FIELD, "<u2")] if dtype == "bf16" else dtype)
    audio = {"waveform": np.zeros((1, 2, 128), dtype=np.int16), "sample_rate": 48000}
    video = assemble_video(images, audio=audio)
    data = encode_video(video)
    decoded = decode_video(data)
    components = cast(dict[str, Any], decoded["components"])
    np.testing.assert_array_equal(components["images"], images)
    np.testing.assert_array_equal(components["audio"]["waveform"], audio["waveform"])
    assert components["images"].dtype == images.dtype
    assert video_meta(video)[COST_META_KEY] == {"ram": images.nbytes + 512}
    assert video_fingerprint("video")(video) == video_fingerprint("video")(decoded)
    meta = encoded_storage_meta(video_meta(video), len(data))
    assert meta[COST_META_KEY] == {"ram": len(data)}
    validate_video_encoded(data, meta)
    meta.pop("storage_dtype")
    validate_video_encoded(data, meta)
    meta["container"] = "mismatch"
    with pytest.raises(ValueError, match="container"):
        validate_video_encoded(data, meta)


@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize("materialize", [False, True])
def test_buffer_payload_decodes_compact_storage_before_mapping_release(
    audio: bool, materialize: bool
) -> None:
    array = np.arange(24, dtype=np.int16 if audio else np.uint8).reshape(2, 3, 4)
    encoded = (
        encode_audio({"waveform": array, "sample_rate": 48000})
        if audio
        else encode_image_array(array)
    )
    segment = SharedMemory(create=True, size=len(encoded))
    assert segment.buf is not None
    segment.buf[:] = encoded
    released: list[bool] = []

    def release() -> None:
        segment.close()
        released.append(True)

    try:
        payload = EncodedPayload.from_buffer(
            "audio" if audio else "image",
            segment.buf,
            decode_audio if audio else decode_image_array,
            "shm",
            release,
        )
        result = payload.load()
        parent = (
            cast(dict[str, Any], cast(dict[str, Any], result)["source"])["pcm"]
            if audio
            else cast(np.ndarray, result)
        )
        assert parent.dtype == array.dtype
        view = parent[0]
        if materialize:
            assert payload.data == encoded
        del parent, result, payload
        gc.collect()
        assert released == [True]
        np.testing.assert_array_equal(view, array[0])
        del view
        gc.collect()
        assert released == [True]
    finally:
        segment.close()
        segment.unlink()


@pytest.mark.parametrize("mutation", ["truncated", "trailing", "object", "bad-magic"])
def test_buffer_payload_releases_invalid_encoded_mapping(mutation: str) -> None:
    data = encode_image_array(np.arange(4, dtype=np.uint8))
    if mutation == "truncated":
        data = data[:-1]
    elif mutation == "trailing":
        data += b"extra"
    elif mutation == "bad-magic":
        data = b"not-npy" + data[7:]
    else:
        stream = io.BytesIO()
        np.save(stream, np.array([object()], dtype=object), allow_pickle=True)
        data = stream.getvalue()
    released: list[bool] = []
    payload = EncodedPayload.from_buffer(
        "image",
        memoryview(data),
        decode_image_array,
        "shm",
        lambda: released.append(True),
    )
    with pytest.raises(ValueError):
        payload.load()
    del payload
    gc.collect()
    assert released == [True]


@pytest.mark.parametrize("audio", [False, True])
def test_shared_memory_boundary_preserves_compact_storage(audio: bool) -> None:
    from dinkster_values import TypeRegistry
    from dinkster_workers import ValueCodec
    from dinkster_workers.boundary import release_segment

    from dinkster import comfy_compose

    registry = TypeRegistry()
    comfy_compose.register_comfy_host_types(registry)
    samples = np.arange(24, dtype=np.int16 if audio else np.uint8).reshape(1, 2, 4, 3)
    obj = {"waveform": samples.reshape(1, 2, 12), "sample_rate": 48000} if audio else samples
    type_id = "comfy.AUDIO" if audio else "comfy.IMAGE"
    codec = ValueCodec(registry, shm_threshold=1)
    blobs: list[bytes] = []
    segments: list[SharedMemory] = []
    consumed: list[str] = []
    source = registry.wrap(type_id, obj)
    wire, sent = codec.encode(source, blobs, segments)
    try:
        value, received = codec.decode(wire, blobs, consumed)
        assert received.transport == "shm"
        assert received.network_bytes == 0
        assert value.meta.get(COST_META_KEY) == {"ram": sent.size_bytes}
        assert value.fingerprint == source.fingerprint
        result = value.resolve()
        array = (
            cast(dict[str, Any], cast(dict[str, Any], result)["source"])["pcm"]
            if audio
            else cast(np.ndarray, result)
        )
        assert array.dtype == samples.dtype
        np.testing.assert_array_equal(array.reshape(-1), samples.reshape(-1))
    finally:
        for segment in segments:
            release_segment(segment)


@pytest.mark.parametrize("host_registration", [False, True])
@pytest.mark.parametrize("dtype", ["int16", "float32", "float16", "float64", "int32"])
@pytest.mark.parametrize("shm", [False, True])
def test_audio_storage_facts_match_canonical_payload_across_boundary(
    host_registration: bool, dtype: str, shm: bool
) -> None:
    from dinkster_nodes_media_io import AUDIO_TYPE, register_media_types
    from dinkster_values import TypeRegistry
    from dinkster_workers import ValueCodec
    from dinkster_workers.boundary import release_segment

    from dinkster import comfy_compose

    registry = TypeRegistry()
    if host_registration:
        comfy_compose.register_comfy_host_types(registry)
    else:
        register_media_types(registry)
    original = np.arange(24, dtype=dtype).reshape(1, 2, 12)
    value = registry.wrap(AUDIO_TYPE, {"waveform": original, "sample_rate": 48000})
    resolved = cast(dict[str, Any], value.resolve())
    stored = cast(dict[str, Any], resolved["source"])["pcm"]
    expected_dtype = "int16" if dtype == "int16" else "float32"
    assert stored.dtype == expected_dtype
    assert original.dtype == dtype
    assert (stored is original) == (dtype in ("int16", "float32"))
    assert value.meta.get("storage_dtype") == ("int16" if dtype == "int16" else "fp32")
    assert value.meta.get(COST_META_KEY) == {"ram": stored.nbytes}
    codec = ValueCodec(registry, shm_threshold=1 if shm else 1000000)
    blobs: list[bytes] = []
    segments: list[SharedMemory] = []
    wire, sent = codec.encode(value, blobs, segments)
    try:
        received, _ = codec.decode(wire, blobs, [])
        received_value = cast(dict[str, Any], received.resolve())
        loaded = cast(dict[str, Any], received_value["source"])["pcm"]
        assert loaded.dtype == expected_dtype
        assert received.meta.get("storage_dtype") == value.meta.get("storage_dtype")
        assert received.meta.get(COST_META_KEY) == {"ram": sent.size_bytes}
        assert received.fingerprint == value.fingerprint
        np.testing.assert_array_equal(loaded, stored)
    finally:
        for segment in segments:
            release_segment(segment)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float16", "float32"])
def test_storage_conversion_and_codec_preserve_media_semantics(dtype: str) -> None:
    from dinkster_values import annotate_image, media_semantics

    color = {"primaries": 9, "transfer": 16, "range": 2, "bit_depth": 10}
    image = annotate_image(np.zeros((2, 2, 2, 4), dtype=dtype), alpha="premultiplied", color=color)
    encoded = encode_image_array(image)
    releases: list[bool] = []
    payload = EncodedPayload.from_buffer(
        "image",
        memoryview(encoded),
        decode_image_array,
        "shm",
        lambda: releases.append(True),
    )
    decoded = payload.load()
    converted = image_input(decoded)
    assert media_semantics(decoded) == media_semantics(image)
    assert media_semantics(converted) == media_semantics(image)
    assert encode_image_array(decoded) == encoded
    assert image_array_meta(converted)["dtype"] == "float32"
    np.testing.assert_array_equal(decoded, image)
    del converted, decoded, payload
    gc.collect()
    assert releases == [True]


def test_video_components_preserve_semantic_trailer() -> None:
    from dinkster_values import annotate_image, media_semantics

    color = {"primaries": 9, "transfer": 18, "range": 2}
    images = annotate_image(np.zeros((2, 4, 4, 4), dtype=np.uint16), color=color)
    video = assemble_video(images)
    encoded = encode_video(video)
    decoded = decode_video(encoded)
    actual = cast(dict[str, Any], decoded["components"])["images"]
    assert media_semantics(actual) == media_semantics(images)
    probe = cast(dict[str, Any], decoded["probe"])
    assert [probe[key] for key in ("primaries", "transfer", "matrix", "range")] == [9, 18, 9, 1]
    assert probe["bit_depth"] == 10
    assert encode_video(decoded) == encoded


def test_inline_layer_rasters_share_array_accounting_without_duplicate_layers() -> None:
    from dinkster_nodes_image.compositor_types import (
        CompositorSourceLayer,
        LayerStack,
        layer_stack_meta,
    )

    layer = CompositorSourceLayer(
        image=np.zeros((1, 4, 4, 3), dtype=np.float32),
        mask=np.ones((1, 4, 4), dtype=np.float32),
        name="pixels",
        x=0,
        y=0,
        width=4,
        height=4,
    )
    meta = layer_stack_meta(LayerStack((layer, layer)))
    assert meta["layers"] == 2
    assert meta["storage_dtype"] == ["fp32"]
    assert layer.mask is not None
    expected = layer.image.nbytes + layer.mask.nbytes
    assert meta[COST_META_KEY] == {"ram": expected}


@pytest.mark.parametrize("field", ["primaries", "transfer", "range"])
def test_merge_rejects_incompatible_color_without_relabeling_pixels(field: str) -> None:
    from dinkster_values import annotate_image, merge_image_batches

    first = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    color = {"primaries": 1, "transfer": 13, "range": 2}
    color[field] = 255
    second = annotate_image(np.zeros((1, 2, 2, 3), dtype=np.uint16), color=color)
    with pytest.raises(ValueError, match="different color semantics"):
        merge_image_batches([first, second])


def test_merge_keeps_only_common_source_color_provenance() -> None:
    from dinkster_values import annotate_image, media_semantics, merge_image_batches

    color = {"primaries": 255, "transfer": 254, "range": 2, "matrix": 9, "bit_depth": 10}
    first = annotate_image(np.zeros((1, 2, 2, 3), dtype=np.uint16), color=color)
    second = annotate_image(np.zeros((1, 2, 2, 3), dtype=np.uint8), color={**color, "bit_depth": 8})
    merged = merge_image_batches([first, second])
    assert media_semantics(merged)["color"] == {
        key: value for key, value in color.items() if key != "bit_depth"
    }
    third = annotate_image(
        np.zeros((1, 2, 2, 3), dtype=np.float16),
        color={key: value for key, value in color.items() if key != "matrix"},
    )
    merged = merge_image_batches([first, second, third])
    assert media_semantics(merged)["color"] == {
        key: value for key, value in color.items() if key not in ("matrix", "bit_depth")
    }
    assert media_semantics(first)["color"] == color
