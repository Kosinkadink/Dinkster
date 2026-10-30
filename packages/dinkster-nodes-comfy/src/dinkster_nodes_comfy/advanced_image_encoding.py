"""Byte encoders used by the advanced image-saving node."""

from __future__ import annotations

import importlib
import json
import os
import struct
import tempfile
import zlib
from fractions import Fraction
from io import BytesIO
from typing import Any

from dinkster_api.v1 import MEBIBYTE

from . import image_color_conversion

JsonMapping = dict[str, Any]

_FORMAT_SPECS = {
    ("png", "8-bit", 1): (255.0, "uint8", "gray", "gray"),
    ("png", "8-bit", 3): (255.0, "uint8", "rgb24", "rgb24"),
    ("png", "8-bit", 4): (255.0, "uint8", "rgba", "rgba"),
    ("png", "16-bit", 1): (65535.0, "uint16", "gray16le", "gray16be"),
    ("png", "16-bit", 3): (65535.0, "uint16", "rgb48le", "rgb48be"),
    ("png", "16-bit", 4): (65535.0, "uint16", "rgba64le", "rgba64be"),
    ("exr", "16-bit float", 1): (1.0, "float32", "grayf32le", "grayf32le"),
    ("exr", "16-bit float", 3): (1.0, "float32", "gbrpf32le", "gbrpf32le"),
    ("exr", "16-bit float", 4): (1.0, "float32", "gbrapf32le", "gbrapf32le"),
    ("exr", "32-bit float", 1): (1.0, "float32", "grayf32le", "grayf32le"),
    ("exr", "32-bit float", 3): (1.0, "float32", "gbrpf32le", "gbrpf32le"),
    ("exr", "32-bit float", 4): (1.0, "float32", "gbrapf32le", "gbrapf32le"),
}
_AVIF_COLOR_PROPERTIES = {
    "sRGB": ("BT709", "IEC61966_2_1", 1),
    "HDR": ("BT2020", "ARIB_STD_B67", 9),
    "HDR PQ": ("BT2020", "SMPTE2084", 9),
}
_HLG_A, _HLG_B, _HLG_C = 0.17883277, 0.28466892, 0.55991072928
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHROMATICITIES = {
    "Rec.709": (0.6400, 0.3300, 0.3000, 0.6000, 0.1500, 0.0600, 0.3127, 0.3290),
    "Rec.2020": (0.7080, 0.2920, 0.1700, 0.7970, 0.1310, 0.0460, 0.3127, 0.3290),
    "ACEScg": (0.7130, 0.2930, 0.1650, 0.8300, 0.1280, 0.0440, 0.32168, 0.33767),
}


def _torch() -> Any:
    return importlib.import_module("torch")


def hlg_to_linear(value: Any) -> Any:
    torch = _torch()
    if value.shape[-1] == 4:
        return torch.cat([hlg_to_linear(value[..., :3]), value[..., 3:]], dim=-1)
    low = value**2 / 3.0
    high = (torch.exp((value.clamp(min=0.5) - _HLG_C) / _HLG_A) + _HLG_B) / 12.0
    return torch.where(value <= 0.5, low, high)


def to_scene_linear(image: Any, colorspace: str) -> Any:
    if colorspace == "sRGB":
        srgb_to_linear = vars(image_color_conversion)["_srgb_to_linear"]
        if image.shape[-1] == 4:
            return _torch().cat([srgb_to_linear(image[..., :3], _torch()), image[..., 3:]], dim=-1)
        return srgb_to_linear(image, _torch())
    if colorspace == "HDR":
        return hlg_to_linear(image)
    return image


def encode_exr_preview(image: Any, colorspace: str) -> bytes:
    torch = _torch()
    pil_image = importlib.import_module("PIL.Image")
    linear = to_scene_linear(image, colorspace)
    x = torch.nan_to_num(linear.float(), nan=0.0, posinf=1.0, neginf=0.0)
    if x.shape[-1] == 1:
        x = x.expand(-1, -1, 3)
    height, width = x.shape[:2]
    x = x.movedim(-1, 0).unsqueeze(0)
    has_alpha = x.shape[1] == 4
    if has_alpha:
        alpha = x[:, 3:].clamp(0.0, 1.0)
        x = torch.cat([x[:, :3] * alpha, alpha], dim=1)
    scale = min(1.0, (1_000_000 / (width * height)) ** 0.5, 16383 / max(width, height))
    if scale < 1.0:
        x = torch.nn.functional.interpolate(
            x, size=(max(1, int(height * scale)), max(1, int(width * scale))), mode="area"
        )
    if has_alpha:
        alpha = x[:, 3:]
        x = torch.cat(
            [torch.where(alpha > 0, x[:, :3] / alpha.clamp_min(1e-12), 0.0), alpha], dim=1
        )
    linear = x[0].movedim(0, -1).clamp(0.0, 1.0)
    rgb = linear[..., :3]
    linear[..., :3] = torch.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb.pow(1 / 2.4) - 0.055)
    pixels = (linear * 255 + 0.5).to(torch.uint8).cpu().numpy()
    image = pil_image.fromarray(pixels, "RGBA" if has_alpha else "RGB")
    with BytesIO() as stream:
        image.save(stream, format="WEBP" if has_alpha else "JPEG", quality=80 if has_alpha else 85)
        return stream.getvalue()


def encode_image(image: Any, file_format: str, bit_depth: str, colorspace: str) -> bytes:
    numpy = importlib.import_module("numpy")
    av = importlib.import_module("av")
    torch = _torch()
    if image.ndim == 2:
        image = image.unsqueeze(-1)
    height, width, channels = image.shape
    spec = _FORMAT_SPECS.get((file_format, bit_depth, channels))
    if spec is None:
        raise ValueError(
            f"No {file_format}/{bit_depth} encoder for {channels}-channel images: "
            "supported channel counts are 1 (grayscale), 3 (RGB) and 4 (RGBA)."
        )
    scale, dtype_name, frame_format, stream_format = spec
    if dtype_name == "float32":
        image = to_scene_linear(image, colorspace)
        if bit_depth == "16-bit float":
            image = image.to(torch.float16)
        image_array = image.cpu().numpy().astype(numpy.float32)
    else:
        image_array = (
            (image * scale)
            .clamp(0, scale)
            .to(torch.int32)
            .cpu()
            .numpy()
            .astype(getattr(numpy, dtype_name))
        )
    codec = av.CodecContext.create(file_format, "w")
    codec.width, codec.height, codec.pix_fmt, codec.time_base = (
        width,
        height,
        stream_format,
        Fraction(1, 1),
    )
    if file_format == "exr":
        codec.options = {
            "format": "half" if bit_depth == "16-bit float" else "float",
            "compression": "zip16",
        }
    frame = av.VideoFrame.from_ndarray(image_array, format=frame_format)
    if frame_format != stream_format:
        frame = frame.reformat(format=stream_format)
    frame.pts, frame.time_base = 0, codec.time_base
    return b"".join(bytes(packet) for packet in [*codec.encode(frame), *codec.encode(None)])


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def inject_png_metadata(
    png_bytes: bytes, prompt: JsonMapping | None, extra_pnginfo: JsonMapping | None
) -> bytes:
    if not png_bytes.startswith(_PNG_SIGNATURE):
        return png_bytes
    chunks: list[bytes] = []
    if prompt is not None:
        chunks.append(
            _png_chunk(
                b"tEXt", b"prompt\x00" + json.dumps(prompt).encode("latin-1", errors="replace")
            )
        )
    if extra_pnginfo:
        chunks.extend(
            _png_chunk(
                b"tEXt",
                key.encode("latin-1")
                + b"\x00"
                + json.dumps(value).encode("latin-1", errors="replace"),
            )
            for key, value in extra_pnginfo.items()
        )
    if not chunks:
        return png_bytes
    ihdr_end = 8 + 8 + struct.unpack(">I", png_bytes[8:12])[0] + 4
    return png_bytes[:ihdr_end] + b"".join(chunks) + png_bytes[ihdr_end:]


def _exr_attribute(name: str, attr_type: str, value: bytes) -> bytes:
    return (
        name.encode()
        + b"\x00"
        + attr_type.encode()
        + b"\x00"
        + struct.pack("<i", len(value))
        + value
    )


def inject_exr_metadata(
    exr_bytes: bytes,
    prompt: JsonMapping | None,
    extra_pnginfo: JsonMapping | None,
    colorspace: str | None = None,
) -> bytes:
    if len(exr_bytes) < 8 or exr_bytes[:4] != b"\x76\x2f\x31\x01":
        return exr_bytes
    blob = b""
    if prompt is not None:
        blob += _exr_attribute("prompt", "string", json.dumps(prompt).encode())
    if extra_pnginfo:
        for key, value in extra_pnginfo.items():
            blob += _exr_attribute(key, "string", json.dumps(value).encode())
    if colorspace is not None:
        primaries = {
            "sRGB": "Rec.709",
            "HDR": "Rec.2020",
            "linear": "Rec.709",
            "linear ACEScg": "ACEScg",
        }.get(colorspace, "Rec.709")
        blob += _exr_attribute(
            "chromaticities", "chromaticities", struct.pack("<8f", *_CHROMATICITIES[primaries])
        )
    if not blob:
        return exr_bytes
    pos, window, compression = 8, None, 0
    while pos < len(exr_bytes) and exr_bytes[pos] != 0:
        name_end, type_end = (
            exr_bytes.index(b"\x00", pos),
            exr_bytes.index(b"\x00", exr_bytes.index(b"\x00", pos) + 1),
        )
        name, attr_type = exr_bytes[pos:name_end], exr_bytes[name_end + 1 : type_end]
        size = struct.unpack("<i", exr_bytes[type_end + 1 : type_end + 5])[0]
        value_start, value = type_end + 5, exr_bytes[type_end + 5 : type_end + 5 + size]
        if name == b"dataWindow" and attr_type == b"box2i":
            window = struct.unpack("<iiii", value)
        elif name == b"compression" and attr_type == b"compression":
            compression = value[0]
        pos = value_start + size
    if window is None:
        return exr_bytes
    scanlines_per_block = {
        0: 1,
        1: 1,
        2: 1,
        3: 16,
        4: 32,
        5: 16,
        6: 32,
        7: 32,
        8: 256,
        9: 256,
    }.get(compression, 1)
    _, y_min, _, y_max = window
    height = y_max - y_min + 1
    count = (height + scanlines_per_block - 1) // scanlines_per_block
    table_start, pixels = pos + 1, pos + 1 + count * 8
    offsets = struct.unpack(f"<{count}Q", exr_bytes[table_start:pixels])
    return (
        exr_bytes[:pos]
        + blob
        + exr_bytes[pos:table_start]
        + struct.pack(f"<{count}Q", *(offset + len(blob) for offset in offsets))
        + exr_bytes[pixels:]
    )


def _boxes(data: bytes | bytearray, start: int, end: int) -> list[tuple[int, int, bytes, int]]:
    result: list[tuple[int, int, bytes, int]] = []
    position = start
    while position < end:
        if position + 8 > end:
            raise ValueError("Invalid AVIF box structure.")
        size, box_type = struct.unpack_from(">I4s", data, position)
        header = 8
        if size == 1:
            if position + 16 > end:
                raise ValueError("Invalid AVIF extended-size box.")
            size, header = struct.unpack_from(">Q", data, position + 8)[0], 16
        elif size == 0:
            size = end - position
        if size < header or position + size > end:
            raise ValueError("Invalid AVIF box size.")
        result.append((position, size, box_type, header))
        position += size
    return result


def _box(box_type: bytes, payload: bytes) -> bytes:
    if len(payload) + 8 > 0xFFFFFFFF:
        raise ValueError("AVIF metadata box is too large.")
    return struct.pack(">I4s", len(payload) + 8, box_type) + payload


def _exif(metadata: JsonMapping) -> bytes:
    exif = importlib.import_module("PIL.Image").Exif()
    if "prompt" in metadata:
        exif[0x0110] = f"prompt:{json.dumps(metadata['prompt'])}"
    tag = 0x010F
    for key, value in metadata.items():
        if key != "prompt":
            exif[tag] = f"{key}:{json.dumps(value)}"
            tag -= 1
    return b"\x00\x00\x00\x00" + exif.tobytes()[6:]


def _add_exif(meta: bytes, offset: int, length: int, delta: int) -> bytes:
    if meta[4:8] != b"meta" or len(meta) < 12:
        raise ValueError("AVIF metadata requires a valid meta box.")
    children = _boxes(meta, 12, len(meta))
    by_type = {kind: (pos, size, head) for pos, size, kind, head in children}
    if not all(kind in by_type for kind in (b"pitm", b"iloc", b"iinf")):
        raise ValueError("AVIF metadata boxes are incomplete.")
    pitm_pos, pitm_size, _ = by_type[b"pitm"]
    pitm = meta[pitm_pos : pitm_pos + pitm_size]
    if pitm[8] != 0:
        raise ValueError("Unsupported AVIF primary-item format.")
    primary = struct.unpack_from(">H", pitm, 12)[0]
    iloc_pos, iloc_size, _ = by_type[b"iloc"]
    iloc = bytearray(meta[iloc_pos : iloc_pos + iloc_size])
    if iloc[8] != 0 or iloc[12] != 0x44 or iloc[13] != 0:
        raise ValueError("Unsupported AVIF item-location format.")
    count, cursor = struct.unpack_from(">H", iloc, 14)[0], 16
    ids: list[int] = []
    for _ in range(count):
        item_id, _, extents = struct.unpack_from(">HHH", iloc, cursor)
        cursor += 6
        ids.append(item_id)
        for _ in range(extents):
            struct.pack_into(">I", iloc, cursor, struct.unpack_from(">I", iloc, cursor)[0] + delta)
            cursor += 8
    if cursor != len(iloc):
        raise ValueError("Unsupported AVIF item-location entries.")
    item_id = max(ids) + 1
    if item_id > 0xFFFF or offset > 0xFFFFFFFF or length > 0xFFFFFFFF:
        raise ValueError("AVIF metadata exceeds 32-bit item limits.")
    struct.pack_into(">H", iloc, 14, count + 1)
    iloc.extend(struct.pack(">HHHII", item_id, 0, 1, offset, length))
    struct.pack_into(">I", iloc, 0, len(iloc))
    iinf_pos, iinf_size, _ = by_type[b"iinf"]
    iinf = bytearray(meta[iinf_pos : iinf_pos + iinf_size])
    if iinf[8] != 0:
        raise ValueError("Unsupported AVIF item-information format.")
    struct.pack_into(">H", iinf, 12, struct.unpack_from(">H", iinf, 12)[0] + 1)
    iinf.extend(
        _box(b"infe", b"\x02\x00\x00\x00" + struct.pack(">HH4s", item_id, 0, b"Exif") + b"\x00")
    )
    struct.pack_into(">I", iinf, 0, len(iinf))
    cdsc = _box(b"cdsc", struct.pack(">HHH", item_id, 1, primary))
    if b"iref" in by_type:
        pos, size, _ = by_type[b"iref"]
        iref = bytearray(meta[pos : pos + size])
        if iref[8] != 0:
            raise ValueError("Unsupported AVIF item-reference format.")
        iref.extend(cdsc)
        struct.pack_into(">I", iref, 0, len(iref))
    else:
        iref = _box(b"iref", b"\x00\x00\x00\x00" + cdsc)
    output = bytearray(meta[:12])
    for pos, size, kind, _ in children:
        if kind == b"iloc":
            output.extend(iloc)
        elif kind == b"iinf":
            output.extend(iinf)
            if b"iref" not in by_type:
                output.extend(iref)
        elif kind == b"iref":
            output.extend(iref)
        else:
            output.extend(meta[pos : pos + size])
    struct.pack_into(">I", output, 0, len(output))
    return bytes(output)


def _adjust_offsets(data: bytes, delta: int) -> bytes:
    output = bytearray(data)

    def visit(start: int, end: int) -> None:
        for pos, size, kind, header in _boxes(output, start, end):
            if kind in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
                visit(pos + header, pos + size)
            elif kind in (b"stco", b"co64"):
                entry_size = 4 if kind == b"stco" else 8
                count = struct.unpack_from(">I", output, pos + header + 4)[0]
                cursor = pos + header + 8
                if cursor + count * entry_size != pos + size:
                    raise ValueError("Invalid AVIF chunk-offset table.")
                for _ in range(count):
                    fmt = ">I" if entry_size == 4 else ">Q"
                    struct.pack_into(
                        fmt, output, cursor, struct.unpack_from(fmt, output, cursor)[0] + delta
                    )
                    cursor += entry_size

    visit(0, len(output))
    return bytes(output)


def _copy_file_bytes(source: Any, destination: Any, size: int) -> None:
    remaining = size
    while remaining:
        chunk = source.read(min(remaining, MEBIBYTE))
        if not chunk:
            raise ValueError("Unexpected end of AVIF file.")
        destination.write(chunk)
        remaining -= len(chunk)


def inject_avif_metadata(path: str, metadata: JsonMapping) -> None:
    if not metadata:
        return
    exif = _exif(metadata)
    file_size = os.path.getsize(path)
    top: list[tuple[int, int, bytes, int, int]] = []
    with open(path, "rb") as source:
        pos = 0
        while pos < file_size:
            source.seek(pos)
            header = source.read(8)
            if len(header) != 8:
                raise ValueError("Invalid AVIF box header.")
            size, kind = struct.unpack(">I4s", header)
            size_field, header_size = size, 8
            if size == 1:
                extended = source.read(8)
                if len(extended) != 8:
                    raise ValueError("Invalid AVIF extended-size box.")
                size, header_size = struct.unpack(">Q", extended)[0], 16
            elif size == 0:
                size = file_size - pos
            if size < header_size or pos + size > file_size:
                raise ValueError("Invalid AVIF top-level box size.")
            top.append((pos, size, kind, header_size, size_field))
            pos += size
        meta = next((box for box in top if box[2] == b"meta"), None)
        mdat = next((box for box in top if box[2] == b"mdat"), None)
        if meta is None or mdat is None or mdat[0] + mdat[1] != file_size or meta[0] > mdat[0]:
            raise ValueError("Unsupported AVIF file layout for metadata.")
        source.seek(meta[0])
        original_meta = source.read(meta[1])
    delta = len(_add_exif(original_meta, 0, len(exif), 0)) - len(original_meta)
    updated = _add_exif(original_meta, mdat[0] + mdat[1] + delta, len(exif), delta)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(path) or "."
    )
    try:
        with os.fdopen(fd, "wb") as destination, open(path, "rb") as source:
            for pos, size, kind, header, size_field in top:
                source.seek(pos)
                if kind == b"ftyp":
                    value = bytearray(source.read(size))
                    value[8:12] = (
                        b"avif"
                        if bytes(value[8:12]) == b"avis" and b"avif" in value[16:]
                        else value[8:12]
                    )
                    destination.write(value)
                elif kind == b"meta":
                    destination.write(updated)
                elif kind == b"moov":
                    destination.write(_adjust_offsets(source.read(size), delta))
                elif kind == b"mdat":
                    new_size = size + len(exif)
                    if size_field == 1:
                        destination.write(struct.pack(">I4sQ", 1, b"mdat", new_size))
                    elif size_field == 0:
                        destination.write(struct.pack(">I4s", 0, b"mdat"))
                    elif new_size <= 0xFFFFFFFF:
                        destination.write(struct.pack(">I4s", new_size, b"mdat"))
                    else:
                        raise ValueError("AVIF media-data box exceeds its 32-bit size field.")
                    source.seek(pos + header)
                    _copy_file_bytes(source, destination, size - header)
                    destination.write(exif)
                else:
                    _copy_file_bytes(source, destination, size)
        os.chmod(temporary, os.stat(path).st_mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _set_avif_color_properties(target: Any, colorspace: str) -> None:
    colors = importlib.import_module("av.video.reformatter")
    primary, transfer, matrix = _AVIF_COLOR_PROPERTIES[colorspace]
    target.color_primaries, target.color_trc, target.colorspace, target.color_range = (
        getattr(colors.ColorPrimaries, primary),
        getattr(colors.ColorTrc, transfer),
        matrix,
        colors.ColorRange.MPEG,
    )


def _avif_frame(image: Any, bit_depth: str, colorspace: str, pixel_format: str) -> Any:
    numpy, av, torch = importlib.import_module("numpy"), importlib.import_module("av"), _torch()
    if image.ndim == 2:
        image = image.unsqueeze(-1)
    channels = image.shape[-1]
    if channels not in (1, 3):
        raise ValueError(
            "AVIF saving supports 1-channel grayscale and 3-channel RGB images; "
            "PyAV's SVT-AV1 encoder does not support alpha."
        )
    if bit_depth == "10-bit YUV420":
        image_array, frame_format = (
            (image * 65535.0).clamp(0, 65535).to(torch.int32).cpu().numpy().astype(numpy.uint16),
            "gray16le" if channels == 1 else "rgb48le",
        )
    else:
        image_array, frame_format = (
            (image * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy(),
            "gray" if channels == 1 else "rgb24",
        )
    if channels == 1:
        image_array = image_array[..., 0]
    frame = av.VideoFrame.from_ndarray(image_array, format=frame_format).reformat(
        format=pixel_format, dst_colorspace=_AVIF_COLOR_PROPERTIES[colorspace][2]
    )
    _set_avif_color_properties(frame, colorspace)
    return frame


def save_avif(
    images: Any,
    output_path: str,
    bit_depth: str,
    colorspace: str,
    crf: int,
    fps: float = 1.0,
    loop_count: int | None = None,
    metadata: JsonMapping | None = None,
) -> None:
    av = importlib.import_module("av")
    if bit_depth == "auto":
        bit_depth = "10-bit YUV420" if colorspace in ("HDR", "HDR PQ") else "8-bit YUV420"
    pixel_format, options = (
        ("yuv420p10le" if bit_depth == "10-bit YUV420" else "yuv420p"),
        ({"loop": str(loop_count)} if loop_count is not None else None),
    )
    with av.open(output_path, mode="w", format="avif", options=options) as container:
        stream = container.add_stream("libsvtav1", rate=Fraction(round(fps * 1000), 1000))
        stream.width, stream.height, stream.pix_fmt, stream.options = (
            images.shape[2],
            images.shape[1],
            pixel_format,
            {"crf": str(crf), "preset": "8"},
        )
        _set_avif_color_properties(stream.codec_context, colorspace)
        for image in images:
            for packet in stream.encode(_avif_frame(image, bit_depth, colorspace, pixel_format)):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    if metadata:
        inject_avif_metadata(output_path, metadata)
