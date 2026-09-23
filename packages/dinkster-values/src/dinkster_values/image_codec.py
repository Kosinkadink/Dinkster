"""The standard image-array codec: npy bytes across boundaries, PNG out.

Image tensors are the first value family whose payload must be USABLE on
both sides of an interpreter boundary: produced as torch tensors in a
foreign worker (the ComfyUI compat child), rendered as PNG bytes by the
engine process - which deliberately has no torch. The default codec cannot
serve that split (pickled torch tensors only unpickle where torch lives),
so this module fixes the byte contract to numpy's ``.npy`` format: dtype,
shape and raw data, readable wherever numpy is. Each side registers the
runtime form its process needs (torch tensor in the worker, numpy array in
the engine) over the SAME bytes; fingerprints hash the raw data, so cache
identity never depends on which form a process holds (hazard H4).

numpy is a call-time requirement, not a package dependency: dinkster-values
stays dependency-free, and every interpreter that actually moves image
arrays (the engine venv, the ComfyUI venv) already ships it. Importing
this module without numpy installed is fine; calling into it is not.

Array layout contract: channels last, float values in [0, 1] -
``[H, W]``, ``[H, W, C]`` or batched ``[B, H, W, C]`` with C in {1, 2, 3, 4}.
The PNG renderer draws the FIRST batch element: a peek shows one frame;
grids and filmstrips are presentation policy, a client concern.
Mask values reuse the array bytes with ``[H, W]`` or ``[B, H, W]`` layout
and a renderer that treats the first axis as the batch.
"""

from __future__ import annotations

import importlib
import io
import json
import math
import struct
import zlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from .model import stable_hash
from .registry import BufferEncoding
from .storage import array_storage_meta, image_input, storage_array, storage_dtype

__all__ = [
    "IMAGE_BATCH_MERGER_ID",
    "IMAGE_FILE_DECODER_ID",
    "PNG_CONTAINER_VERSION",
    "annotate_image",
    "annotate_mask",
    "copy_media_semantics",
    "decode_image_array",
    "decode_image_file",
    "encode_canonical_png",
    "encode_image_array",
    "image_array_fingerprint",
    "image_array_meta",
    "mask_array_meta",
    "media_semantics",
    "merge_image_batches",
    "prepare_image_array_encoding",
    "render_image_png",
    "render_mask_png",
]

PNG_CONTAINER_VERSION = "stored-v1"


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - both venvs ship numpy
        raise RuntimeError("the image-array codec needs numpy in this interpreter") from exc
    return numpy


def _as_array(obj: object) -> Any:
    """The portable storage form, without expanding reduced-precision values."""
    return storage_array(obj)


_MEDIA_MAGIC = b"DINKSTER-MEDIA\x01"
_MEDIA_LIMIT = 4096
_DEFAULT_COLOR: dict[str, object] = {"primaries": 1, "transfer": 13, "range": 2}


def image_color(value: object = None) -> dict[str, object]:
    """Validate FFmpeg color enums without converting pixels or rejecting reserved values."""
    if value is None:
        return dict(_DEFAULT_COLOR)
    if not isinstance(value, Mapping):
        raise ValueError("image color must be an object")
    color = dict(cast("Mapping[str, object]", value))
    if set(color) - {"primaries", "transfer", "range", "matrix", "bit_depth"}:
        raise ValueError("unknown image color field")
    if not {"primaries", "transfer", "range"} <= set(color):
        raise ValueError("image color requires primaries, transfer and range")
    for key, field in color.items():
        if type(field) is not int or field < 0:
            raise ValueError(f"invalid image color {key}")
    return color


def _canonical_semantics(value: object, shape: tuple[int, ...]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("media semantics must be an object")
    metadata = dict(cast("Mapping[str, object]", value))
    if set(metadata) - {"alpha", "color", "polarity", "semantic"}:
        raise ValueError("unknown media semantic field")
    alpha = metadata.get("alpha")
    if alpha is not None:
        if alpha not in ("straight", "premultiplied", "none"):
            raise ValueError("invalid alpha mode")
        present = len(shape) in (3, 4) and shape[-1] in (2, 4)
        if (alpha != "none") != present:
            raise ValueError("alpha mode does not match image channel count")
        if alpha != "premultiplied":
            metadata.pop("alpha")
    if "color" in metadata:
        color = image_color(metadata["color"])
        metadata["color"] = color
        if color == _DEFAULT_COLOR:
            metadata.pop("color")
    if "polarity" in metadata:
        if metadata["polarity"] not in ("coverage", "transparency"):
            raise ValueError("invalid mask polarity")
        if metadata["polarity"] == "coverage":
            metadata.pop("polarity")
    if "semantic" in metadata:
        if metadata["semantic"] not in ("alpha", "selection", "other"):
            raise ValueError("invalid mask semantic")
        if metadata["semantic"] == "selection":
            metadata.pop("semantic")
    if ("polarity" in metadata or "semantic" in metadata) and (
        "alpha" in metadata or "color" in metadata
    ):
        raise ValueError("image and mask semantics cannot be combined")
    return metadata


def _shape(obj: object) -> tuple[int, ...]:
    shape = getattr(obj, "shape", None)
    return tuple(int(n) for n in (_as_array(obj).shape if shape is None else shape))


def media_semantics(obj: object) -> dict[str, object]:
    """Explicit semantics of this object; array operations do not infer new claims."""
    return _canonical_semantics(getattr(obj, "_dinkster_media", {}), _shape(obj))


def _annotate(obj: object, metadata: Mapping[str, object]) -> Any:
    canonical = _canonical_semantics(metadata, _shape(obj))
    if hasattr(obj, "detach"):
        # A new tensor view avoids changing annotations on an aliased input.
        result = cast("Any", obj).view_as(obj)
    else:
        from ._media_array import MediaArray

        result = _numpy().asarray(obj).view(MediaArray)
    result._dinkster_media = canonical
    return result


def annotate_image(
    obj: object, *, alpha: str | None = None, color: Mapping[str, object] | None = None
) -> Any:
    metadata = media_semantics(obj)
    if alpha is not None:
        metadata["alpha"] = alpha
    if color is not None:
        metadata["color"] = dict(color)
    return _annotate(obj, metadata)


def annotate_mask(obj: object, *, polarity: str = "coverage", semantic: str = "selection") -> Any:
    return _annotate(obj, {"polarity": polarity, "semantic": semantic})


def copy_media_semantics(source: object, result: object) -> Any:
    metadata = media_semantics(source)
    shape = _shape(result)
    if len(shape) not in (3, 4) or shape[-1] not in (2, 4):
        metadata.pop("alpha", None)
    return _annotate(result, metadata) if metadata else result


def _semantic_bytes(obj: object) -> bytes:
    metadata = media_semantics(obj)
    if not metadata:
        return b""
    data = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > _MEDIA_LIMIT:
        raise ValueError("media semantics exceed the metadata budget")
    return data


def _media_trailer(obj: object) -> bytes:
    data = _semantic_bytes(obj)
    return _MEDIA_MAGIC + struct.pack("<I", len(data)) + data if data else b""


def encode_image_array(obj: object) -> bytes:
    """Image array (numpy, or any torch-tensor-shaped object) -> npy bytes."""
    np = _numpy()
    buffer = io.BytesIO()
    np.save(buffer, _as_array(obj), allow_pickle=False)  # pyright: ignore[reportUnknownMemberType]
    buffer.write(_media_trailer(obj))
    return buffer.getvalue()


class _HeaderCapture:
    def __init__(self) -> None:
        self.header: bytes | None = None

    def write(self, data: bytes) -> int:
        if self.header is not None:
            raise RuntimeError("numpy wrote an unexpected second npy header chunk")
        self.header = bytes(data)
        return len(data)


class _BufferSink:
    def __init__(self, buffer: memoryview) -> None:
        self.buffer = buffer
        self.position = 0

    def write(self, data: bytes) -> int:
        end = self.position + len(data)
        if end > self.buffer.nbytes:
            raise ValueError(f"image-array encoding exceeded its {self.buffer.nbytes}-byte buffer")
        self.buffer[self.position : end] = data
        self.position = end
        return len(data)


def prepare_image_array_encoding(obj: object) -> BufferEncoding:
    """Prepare the exact npy encoding for a direct caller-owned write."""
    np = _numpy()
    array = _as_array(obj)
    if array.dtype.hasobject:
        raise ValueError("Object arrays cannot be saved when allow_pickle=False")
    dtype_class = cast("Any", type(array.dtype))
    if not getattr(dtype_class, "_legacy", True):
        raise ValueError("User-defined dtypes cannot be saved when allow_pickle=False")
    dtype = np.lib.format.drop_metadata(array.dtype)
    header_data = {
        "shape": array.shape,
        "fortran_order": bool(array.flags.f_contiguous and not array.flags.c_contiguous),
        "descr": np.lib.format.dtype_to_descr(dtype),
    }
    capture = _HeaderCapture()
    try:
        np.lib.format.write_array_header_1_0(capture, header_data)
    except ValueError:
        capture = _HeaderCapture()
        try:
            np.lib.format.write_array_header_2_0(capture, header_data)
        except UnicodeEncodeError:
            capture = _HeaderCapture()
            write_header = getattr(np.lib.format, "_write_array_header", None)
            if write_header is None:
                format_impl = importlib.import_module("numpy.lib._format_impl")
                write_header = cast("Any", format_impl)._write_array_header
            write_header(capture, header_data, (3, 0))
    if capture.header is None:  # pragma: no cover - numpy always writes a header first
        raise RuntimeError("numpy did not produce an npy header")
    header = capture.header
    trailer = _media_trailer(obj)
    size = len(header) + int(array.nbytes) + len(trailer)

    def write(buffer: memoryview) -> int:
        if buffer.nbytes != size:
            raise ValueError(
                f"image-array encoding needs a {size}-byte buffer, got {buffer.nbytes}"
            )
        sink = _BufferSink(buffer)
        np.lib.format.write_array(sink, array, allow_pickle=False)
        sink.write(trailer)
        return sink.position

    return BufferEncoding(size=size, write=write)


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate media metadata key")
        result[key] = value
    return result


def _array_header(data: bytes | memoryview) -> tuple[tuple[int, ...], bool, Any, int]:
    np = _numpy()
    stream = io.BytesIO(data[:10_032])
    version = np.lib.format.read_magic(stream)
    read_header = getattr(np.lib.format, "_read_array_header", None)
    if read_header is None:
        read_header = importlib.import_module("numpy.lib._format_impl")._read_array_header
    shape, fortran, dtype = read_header(stream, version)
    if dtype.hasobject or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("invalid media array header")
    return tuple(shape), bool(fortran), dtype, stream.tell()


def _encoded_parts(data: bytes | memoryview) -> tuple[tuple[int, ...], Any, dict[str, object]]:
    shape, _, dtype, offset = _array_header(data)
    end = offset + math.prod(shape) * int(dtype.itemsize)
    if end > len(data):
        raise ValueError("media array dimensions exceed its payload")
    if len(data) - end > _MEDIA_LIMIT + len(_MEDIA_MAGIC) + 4:
        raise ValueError("media metadata exceeds its budget")
    trailer = bytes(data[end:])
    if not trailer:
        return tuple(shape), dtype, {}
    header_size = len(_MEDIA_MAGIC) + 4
    if not trailer.startswith(_MEDIA_MAGIC) or len(trailer) < header_size:
        raise ValueError("invalid media metadata trailer")
    size = struct.unpack("<I", trailer[len(_MEDIA_MAGIC) : header_size])[0]
    if size > _MEDIA_LIMIT or len(trailer) != header_size + size:
        raise ValueError("invalid media metadata size")
    try:
        metadata = json.loads(trailer[header_size:], object_pairs_hook=_unique_pairs)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid media metadata JSON") from exc
    return tuple(shape), dtype, _canonical_semantics(metadata, tuple(shape))


def decode_image_array(data: bytes) -> object:
    """Validate header and semantics before allocating the array."""
    _, _, metadata = _encoded_parts(data)
    array = _numpy().load(io.BytesIO(data), allow_pickle=False)
    return _annotate(array, metadata) if metadata else array


def image_array_fingerprint(type_id: str) -> Callable[[object], str]:
    """A FingerprintFn hashing dtype, shape and raw data under ``type_id``.

    Faster than the default (no npy header round trip per wrap) and
    identical whichever runtime form (torch or numpy) the process holds."""

    def fingerprint(obj: object) -> str:
        np = _numpy()
        array = np.ascontiguousarray(_as_array(obj))
        semantics = _semantic_bytes(obj)
        if not semantics:
            return _array_fingerprint(type_id, array)
        return stable_hash([_array_fingerprint(type_id, array).encode(), semantics])

    return fingerprint


def _array_fingerprint(type_id: str, array: Any) -> str:
    return stable_hash(
        [
            type_id.encode("utf-8"),
            str(array.dtype).encode("utf-8"),
            repr(array.shape).encode("utf-8"),
            memoryview(array).cast("B"),
        ]
    )


def image_array_meta(obj: object) -> Mapping[str, object]:
    """Interrogable envelope meta: shape and dtype, so clients can show
    dimensions (and batch size) without touching the payload."""
    return _resident_array_meta(obj, mask=False)


def _resident_array_meta(obj: object, *, mask: bool) -> dict[str, object]:
    array = cast("Any", obj) if hasattr(obj, "dtype") and hasattr(obj, "shape") else _as_array(obj)
    storage = array_storage_meta(array)
    dtype = (
        "float32" if storage["storage_dtype"] == "bf16" else str(array.dtype).removeprefix("torch.")
    )
    return {
        **_array_meta(tuple(int(n) for n in array.shape), dtype, media_semantics(obj), mask=mask),
        **storage,
    }


def _array_meta(
    shape: tuple[int, ...], dtype: str, semantics: Mapping[str, object], *, mask: bool
) -> dict[str, object]:
    if mask:
        return {
            "shape": shape,
            "dtype": dtype,
            "polarity": semantics.get("polarity", "coverage"),
            "semantic": semantics.get("semantic", "selection"),
        }
    channels = shape[-1] if len(shape) in (3, 4) else 1
    return {
        "shape": shape,
        "dtype": dtype,
        "channels": {
            "layout": {1: "gray", 2: "gray_alpha", 3: "rgb", 4: "rgba"}.get(channels, "planes"),
            "alpha": semantics.get("alpha", "straight") if channels in (2, 4) else "none",
        },
        "color": semantics.get("color", dict(_DEFAULT_COLOR)),
    }


def mask_array_meta(obj: object) -> Mapping[str, object]:
    return _resident_array_meta(obj, mask=True)


def image_encoded_meta(data: bytes | memoryview, *, mask: bool = False) -> dict[str, object]:
    shape, dtype, semantics = _encoded_parts(data)
    kind = storage_dtype(_numpy().empty(0, dtype=dtype))
    return {
        **_array_meta(shape, "float32" if kind == "bf16" else str(dtype), semantics, mask=mask),
        "storage_dtype": kind,
    }


def validate_image_encoded(data: bytes | memoryview, meta: Mapping[str, object]) -> None:
    expected = image_encoded_meta(data, mask="polarity" in meta or "semantic" in meta)
    for key, value in expected.items():
        if key in meta and json.dumps(meta[key], sort_keys=True) != json.dumps(
            value, sort_keys=True
        ):
            raise ValueError(f"media {key} metadata does not match its payload")


IMAGE_FILE_DECODER_ID = "dinkster.image-file@3"
"""Stable identity of :func:`decode_image_file` for coerced-input cache
fingerprints (typed assets: identity = asset digest + provider identity).
Bump the ``@N`` suffix whenever the decode SEMANTICS change - same file
bytes producing a different array is a new provider identity."""

IMAGE_BATCH_MERGER_ID = "dinkster.image-batch-merge@2"
"""Stable identity of :func:`merge_image_batches`, same contract as the
decoder id: it joins merged-input cache fingerprints, so it only changes
when the merge semantics do."""


def _pillow() -> Any:
    try:
        import PIL.Image
        import PIL.ImageCms
        import PIL.ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "decoding image files needs Pillow in this interpreter "
            "(the dinkster umbrella package declares it; dinkster-values itself "
            "stays dependency-free)"
        ) from exc
    assert PIL.Image and PIL.ImageCms and PIL.ImageOps  # submodules imported for the caller
    return PIL


def decode_image_file(asset: object) -> object:
    """One image-file asset -> a float32 RGB or RGBA batch in [0, 1].

    The ``asset<comfy.IMAGE>`` decode provider (typed assets, joint
    contract 2026-07-26). ``asset`` is the base asset runtime object,
    duck-typed to its ``open()`` protocol so this module never imports
    dinkster-assets (which depends on this package).

    EXIF orientation is applied and colors are normalized to sRGB using a
    valid embedded ICC profile. Images without a usable profile are treated
    as sRGB. Source alpha is preserved as straight RGBA, including palette
    transparency; RGB sources never acquire a redundant alpha channel."""
    opener = getattr(asset, "open", None)
    if opener is None:
        raise TypeError(
            f"decode_image_file expects an asset with open(), got {type(asset).__name__}"
        )
    np = _numpy()
    pil = _pillow()
    with opener() as handle:
        try:
            with pil.Image.open(handle) as image:
                icc_profile = image.info.get("icc_profile")
                image = pil.ImageOps.exif_transpose(image)
                if image.mode.startswith("I;16"):
                    plane = np.asarray(image, dtype=np.uint16)
                    return np.repeat(plane[None, :, :, None], 3, axis=-1)
                alpha = (
                    image.convert("RGBA").getchannel("A")
                    if "A" in image.getbands() or "transparency" in image.info
                    else None
                )
                rgb = image.convert("RGB")
                if icc_profile:
                    try:
                        source = (
                            image
                            if image.mode in {"RGB", "RGBA", "RGBX", "CMYK", "LAB"}
                            else image.convert("RGB")
                        )
                        transformed = pil.ImageCms.profileToProfile(
                            source,
                            pil.ImageCms.ImageCmsProfile(io.BytesIO(icc_profile)),
                            pil.ImageCms.createProfile("sRGB"),
                            renderingIntent=pil.ImageCms.Intent.PERCEPTUAL,
                            outputMode="RGB",
                        )
                        if transformed is not None:
                            rgb = transformed
                    except (ImportError, pil.ImageCms.PyCMSError, OSError, TypeError, ValueError):
                        pass
                if alpha is not None:
                    rgb.putalpha(alpha)
        except Exception as exc:
            name = getattr(asset, "name", "") or "asset"
            raise ValueError(f"cannot decode '{name}' as an image: {exc}") from exc
    array = np.asarray(rgb, dtype=np.uint8)
    return array[None, :, :, :]


def _resize_bilinear_center(array: Any, height: int, width: int) -> Any:
    """[B, H, W, C] -> [B, height, width, C]: center-crop to the target
    aspect ratio, then bilinear-resample - numpy port of ComfyUI's
    ``common_upscale(..., "bilinear", "center")`` (torch interpolate with
    align_corners=False, edge-replicated samples)."""
    np = _numpy()
    old_height, old_width = int(array.shape[1]), int(array.shape[2])
    old_aspect = old_width / old_height
    new_aspect = width / height
    x = y = 0
    if old_aspect > new_aspect:
        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
    cropped = array[:, y : old_height - y, x : old_width - x, :]
    in_height, in_width = int(cropped.shape[1]), int(cropped.shape[2])
    if (in_height, in_width) == (height, width):
        return cropped
    ys = (np.arange(height, dtype=np.float64) + 0.5) * (in_height / height) - 0.5
    xs = (np.arange(width, dtype=np.float64) + 0.5) * (in_width / width) - 0.5
    y0 = np.floor(ys)
    x0 = np.floor(xs)
    wy = (ys - y0).astype(np.float32)
    wx = (xs - x0).astype(np.float32)
    y0i = np.clip(y0, 0, in_height - 1).astype(np.int64)  # pyright: ignore[reportUnknownMemberType]
    y1i = np.clip(y0 + 1, 0, in_height - 1).astype(np.int64)  # pyright: ignore[reportUnknownMemberType]
    x0i = np.clip(x0, 0, in_width - 1).astype(np.int64)  # pyright: ignore[reportUnknownMemberType]
    x1i = np.clip(x0 + 1, 0, in_width - 1).astype(np.int64)  # pyright: ignore[reportUnknownMemberType]
    wy = wy[None, :, None, None]
    wx = wx[None, None, :, None]
    top = cropped[:, y0i][:, :, x0i] * (1 - wx) + cropped[:, y0i][:, :, x1i] * wx
    bottom = cropped[:, y1i][:, :, x0i] * (1 - wx) + cropped[:, y1i][:, :, x1i] * wx
    return top * (1 - wy) + bottom * wy


def merge_image_batches(batches: Sequence[object]) -> object:
    """Merge decoded image batches into ONE batched array (concat on B).

    The ``comfy.IMAGE`` batch-merge provider (typed assets, user amendment
    2026-07-26): the explicit mechanism that lets a multi-select asset
    widget feed a scalar image input as one batch. Shape policy is the
    parity floor, ComfyUI's ImageBatch: later images pad their channel
    count up to the widest seen (constant 1.0, ImageBatch's alpha fill)
    and resize to the FIRST image's height/width (bilinear, center crop).
    Order is selection order; merging zero batches is an error, one batch
    passes through."""
    if not batches:
        raise ValueError("cannot merge zero image batches")
    np = _numpy()
    arrays: list[Any] = []
    semantics = [media_semantics(item) for item in batches]
    colors = [cast("Mapping[str, object]", meta.get("color", _DEFAULT_COLOR)) for meta in semantics]
    color = dict(colors[0])
    for other in colors[1:]:
        if any(other[key] != color[key] for key in ("primaries", "transfer", "range")):
            raise ValueError("cannot merge image batches with different color semantics")
        for key in ("matrix", "bit_depth"):
            if color.get(key) != other.get(key):
                color.pop(key, None)
    premultiplied = all(meta.get("alpha") == "premultiplied" for meta in semantics)
    for item, meta in zip(batches, semantics, strict=True):
        array = np.asarray(image_input(_as_array(item)))
        if array.ndim == 3:
            array = array[None, :, :, :]
        if array.ndim != 4:
            raise ValueError(f"cannot merge image batch of shape {tuple(array.shape)}")
        if not premultiplied and meta.get("alpha") == "premultiplied":
            array = array.copy()
            alpha = array[..., -1:]
            array[..., :-1] = np.divide(
                array[..., :-1], alpha, out=np.zeros_like(array[..., :-1]), where=alpha != 0
            )
        arrays.append(array)
    if len(arrays) == 1:
        return copy_media_semantics(batches[0], arrays[0])
    channels = max(int(a.shape[3]) for a in arrays)
    height, width = int(arrays[0].shape[1]), int(arrays[0].shape[2])
    merged: list[Any] = []
    for array in arrays:
        if int(array.shape[3]) < channels:
            pad = np.ones((*array.shape[:3], channels - int(array.shape[3])), dtype=np.float32)
            array = np.concatenate((array, pad), axis=3)  # pyright: ignore[reportUnknownMemberType]
        if (int(array.shape[1]), int(array.shape[2])) != (height, width):
            array = _resize_bilinear_center(array, height, width)
        merged.append(array.astype(np.float32))
    result = np.concatenate(merged, axis=0)
    return annotate_image(
        result,
        alpha="premultiplied" if premultiplied else None,
        color=color,
    )


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def _stored_zlib(data: bytes) -> bytes:
    result = bytearray(b"\x78\x01")
    if not data:
        result.extend(b"\x01\x00\x00\xff\xff")
    for offset in range(0, len(data), 65_535):
        block = data[offset : offset + 65_535]
        final = offset + len(block) == len(data)
        length = len(block)
        result.append(1 if final else 0)
        result.extend(struct.pack("<HH", length, length ^ 0xFFFF))
        result.extend(block)
    result.extend(struct.pack(">I", zlib.adler32(data)))
    return bytes(result)


def encode_canonical_png(pixels: bytes, *, width: int, height: int, color_type: int) -> bytes:
    """Encode packed 8-bit pixels in the canonical dependency-free PNG container."""
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if width < 1 or height < 1 or channels is None:
        raise ValueError("invalid canonical PNG dimensions or color type")
    row_size = width * channels
    if len(pixels) != row_size * height:
        raise ValueError("canonical PNG pixel payload has the wrong size")
    scanlines = b"".join(
        b"\0" + pixels[offset : offset + row_size] for offset in range(0, len(pixels), row_size)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", _stored_zlib(scanlines))
        + _png_chunk(b"IEND", b"")
    )


def render_image_png(obj: object) -> bytes:
    """Encode an image array ([0,1] floats; HxW, HxWxC or BxHxWxC with C in
    {1, 2, 3, 4}) as PNG bytes. Batched input renders its first element.

    Hand-rolled (zlib + struct) so the rendition costs no dependency
    beyond numpy: the browser-renderable form of an image is core
    contract (DESIGN 3.5), not an optional nicety."""
    np = _numpy()
    array = _as_array(obj)
    if array.ndim == 4:
        if array.shape[0] < 1:
            raise ValueError("cannot render an empty image batch as PNG")
        array = array[0]
    array = np.asarray(image_input(array))
    if media_semantics(obj).get("alpha") == "premultiplied":
        array = array.copy()
        alpha = array[..., -1:]
        array[..., :-1] = np.divide(
            array[..., :-1], alpha, out=np.zeros_like(array[..., :-1]), where=alpha != 0
        )
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim == 2:
        color_type = 0  # grayscale
    elif array.ndim == 3 and array.shape[2] == 2:
        color_type = 4  # grayscale with alpha
    elif array.ndim == 3 and array.shape[2] == 3:
        color_type = 2  # RGB
    elif array.ndim == 3 and array.shape[2] == 4:
        color_type = 6  # RGBA
    else:
        raise ValueError(f"cannot render shape {array.shape} as PNG")
    pixels = (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)  # pyright: ignore[reportUnknownMemberType]
    height, width = pixels.shape[0], pixels.shape[1]
    if height < 1 or width < 1:
        raise ValueError(f"cannot render shape {array.shape} as PNG")
    return encode_canonical_png(pixels.tobytes(), width=width, height=height, color_type=color_type)


def render_mask_png(obj: object) -> bytes:
    """Encode a mask array (HxW or BxHxW) as a grayscale PNG."""
    array = _as_array(image_input(obj))
    if array.ndim == 3:
        if array.shape[0] < 1:
            raise ValueError("cannot render an empty mask batch as PNG")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"cannot render mask shape {array.shape} as PNG")
    return render_image_png(array)
