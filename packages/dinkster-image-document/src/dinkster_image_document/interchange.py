"""Native, OpenRaster, and PSD layered-document interchange."""

from __future__ import annotations
from dinkster_values import GIBIBYTE, MEBIBYTE

import io
import json
import math
import os
import warnings
import zipfile
from typing import Any, cast
from xml.etree import ElementTree as ET

import numpy as np
from dinkster_assets import AssetVault, digest_bytes
from PIL import Image

from .document import (
    IDENTITY,
    ImageDocument,
    admit_raster,
    append_raster,
    apply_commands,
    empty_document,
)
from .format import BLEND_MODES, InvalidDocument, canonical_json, ordered_layer_ids

MAX_ARCHIVE_BYTES = GIBIBYTE


def _ora_blend(mode: str) -> str:
    return "svg:src-over" if mode == "normal" else "svg:" + mode.replace("_", "-")


def _portable_raster(record: dict[str, Any], layer: dict[str, Any]) -> bool:
    if layer["kind"] != "raster" or layer["maskIds"]:
        return False
    resource = record["resources"][layer["resourceId"]]
    source = layer["sourceRect"]
    transform = layer["transform"]
    return (
        resource["mediaType"] == "image/png"
        and source == {"x": 0, "y": 0, "width": resource["width"], "height": resource["height"]}
        and {key: transform[key] for key in ("a", "b", "c", "d")}
        == {"a": 1_000_000, "b": 0, "c": 0, "d": 1_000_000}
        and transform["tx"] % 1_000_000 == 0
        and transform["ty"] % 1_000_000 == 0
    )


def _baked_layer(document: ImageDocument, identifier: str) -> bytes:
    record = document.to_record()
    layer = record["layers"][identifier]
    layer["visible"] = True
    layer["opacity"] = 65_535
    return ImageDocument.from_record(record, document.bind).render(f"layer:{identifier}").png


def _thumbnail(png: bytes) -> bytes:
    with Image.open(io.BytesIO(png)) as source:
        image = source.convert("RGBA")
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def _opacity_8(value: int) -> int:
    return round(value / 65_535 * 255)


def _ora_opacity(value: int) -> str:
    opacity = (value * 255 + 32_767) // 65_535
    if opacity == 0:
        return "0.0"
    if opacity == 255:
        return "1.0"
    # Krita truncates after multiplying by 255, so stay just above the exact ratio.
    return str(math.nextafter(opacity / 255, 1.0))


def _resource_image(document: ImageDocument, descriptor: dict[str, Any]) -> Image.Image:
    with Image.open(io.BytesIO(document.read_resource(descriptor))) as source:
        pixels = np.asarray(source.convert("RGBA"), dtype=np.int64).copy()
    alpha = pixels[:, :, 3:4]
    if descriptor["alphaMode"] == "opaque":
        alpha[:] = 255
    elif descriptor["alphaMode"] == "premultiplied":
        pixels[:, :, :3] = np.where(
            alpha == 0,
            0,
            np.minimum(255, (pixels[:, :, :3] * 255 + alpha // 2) // np.maximum(alpha, 1)),
        )
    return Image.fromarray(pixels.astype(np.uint8), "RGBA")


def _psd_mask_image(
    document: ImageDocument, record: dict[str, Any], mask: dict[str, Any]
) -> Image.Image:
    descriptor = record["resources"][mask["resourceId"]]
    source = _resource_image(document, descriptor)
    rect = mask["sourceRect"]
    source = source.crop(
        (rect["x"], rect["y"], rect["x"] + rect["width"], rect["y"] + rect["height"])
    )
    pixels = np.asarray(source, dtype=np.int64)
    if mask["channel"] == "alpha":
        value = pixels[:, :, 3] * 257
    else:
        value = (
            pixels[:, :, 0] * 13_933 + pixels[:, :, 1] * 46_871 + pixels[:, :, 2] * 4_731 + 127
        ) // 255
    if mask["invert"]:
        value = 65_535 - value
    opacity = mask["opacity"]
    value = 65_535 - opacity + (value * opacity + 32_767) // 65_535
    return Image.fromarray(((value + 128) // 257).astype(np.uint8), "L")


def _psd_standard_view_reason(record: dict[str, Any]) -> str | None:
    canvas = record["canvas"]
    if canvas.get("background") or canvas["compositing"] != "premultiplied-alpha":
        return "canvas compositing"
    for layer in record["layers"].values():
        if layer["blendMode"] in {"grain_extract", "grain_merge"}:
            return "blend mode"
        transform = layer["transform"]
        if {key: transform[key] for key in ("a", "b", "c", "d")} != {
            "a": 1_000_000,
            "b": 0,
            "c": 0,
            "d": 1_000_000,
        }:
            return "layer transform"
        if transform["tx"] % 1_000_000 or transform["ty"] % 1_000_000:
            return "layer transform"
        if layer["kind"] == "group":
            if (
                transform["tx"]
                or transform["ty"]
                or layer["maskIds"]
                or layer["clipping"] != "none"
            ):
                return "group geometry, clipping, or mask"
            continue
        if len(layer["maskIds"]) > 1:
            return "combined masks"
        if layer["maskIds"]:
            mask = record["masks"][layer["maskIds"][0]]
            if (
                any(
                    mask["transform"][key] != layer["transform"][key]
                    for key in ("a", "b", "c", "d", "tx", "ty")
                )
                or mask["sourceRect"] != layer["sourceRect"]
            ):
                return "mask geometry"
    return None


def _write_psd_with_merged_image(psd: Any, merged: Image.Image) -> bytes:
    record = getattr(psd, "_record", None)
    layer_and_mask = getattr(record, "layer_and_mask_information", None)
    layer_info = getattr(layer_and_mask, "layer_info", None)
    image_data = getattr(record, "image_data", None)
    header = getattr(record, "header", None)
    if (
        layer_info is None
        or not hasattr(layer_info, "layer_count")
        or not callable(getattr(image_data, "set_data", None))
        or not callable(getattr(record, "write", None))
        or header is None
    ):
        raise RuntimeError("unsupported psd-tools merged-image writer API")
    # psd-tools has no public setter for an external merged image or its transparency channel.
    layer_info.layer_count = -abs(layer_info.layer_count)
    raw_image_data = cast(Any, image_data)
    raw_record = cast(Any, record)
    raw_image_data.set_data([channel.tobytes() for channel in merged.split()], header)
    output = io.BytesIO()
    raw_record.write(output)
    return output.getvalue()


def export_psd(document: ImageDocument) -> bytes:
    from psd_tools import PSDImage
    from psd_tools.api.layers import Group, GroupMixin, PixelLayer
    from psd_tools.constants import BlendMode

    record = document.to_record()
    canvas = record["canvas"]
    psd = PSDImage.new("RGBA", (canvas["width"], canvas["height"]))
    reason = _psd_standard_view_reason(record)
    if not record["rootLayerIds"] or reason is not None:
        if reason is not None:
            warnings.warn(f"PSD uses a merged standard view for unsupported {reason}", stacklevel=2)
        with Image.open(io.BytesIO(document.render().png)) as merged:
            PixelLayer.frompil(merged.convert("RGBA"), psd, name="Composite")
    else:
        blend_modes = {mode.name.lower(): mode for mode in BlendMode}

        def add_layers(parent: GroupMixin, identifiers: list[str]) -> None:
            for identifier in ordered_layer_ids(record["layers"], identifiers):
                layer = record["layers"][identifier]
                if layer["kind"] == "group":
                    group = Group.new(parent, name=layer["name"])
                    group.visible = layer["visible"]
                    group.opacity = _opacity_8(layer["opacity"])
                    group.blend_mode = (
                        BlendMode.PASS_THROUGH
                        if layer.get("isolation", "isolated") == "pass-through"
                        else blend_modes[layer["blendMode"]]
                    )
                    add_layers(group, layer["childLayerIds"])
                    continue
                descriptor = record["resources"][layer["resourceId"]]
                source = _resource_image(document, descriptor)
                rect = layer["sourceRect"]
                source = source.crop(
                    (
                        rect["x"],
                        rect["y"],
                        rect["x"] + rect["width"],
                        rect["y"] + rect["height"],
                    )
                )
                transform = layer["transform"]
                pixel = PixelLayer.frompil(
                    source,
                    parent,
                    name=layer["name"],
                    left=transform["tx"] // 1_000_000,
                    top=transform["ty"] // 1_000_000,
                )
                pixel.visible = layer["visible"]
                pixel.opacity = _opacity_8(layer["opacity"])
                pixel.blend_mode = blend_modes[layer["blendMode"]]
                pixel.clipping = layer["clipping"] == "clip-to-previous"
                if layer["maskIds"]:
                    mask = record["masks"][layer["maskIds"][0]]
                    created = pixel.create_mask(
                        _psd_mask_image(document, record, mask),
                        left=transform["tx"] // 1_000_000,
                        top=transform["ty"] // 1_000_000,
                    )
                    created.disabled = not mask["enabled"]

        add_layers(psd, record["rootLayerIds"])
    with Image.open(io.BytesIO(document.render().png)) as rendered:
        return _write_psd_with_merged_image(psd, rendered.convert("RGBA"))


def export_ora(document: ImageDocument) -> bytes:
    record = document.to_record()
    canvas = record["canvas"]
    root = ET.Element(
        "image", {"w": str(canvas["width"]), "h": str(canvas["height"]), "version": "0.0.3"}
    )
    stack = ET.SubElement(root, "stack")
    merged = document.render().png
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", b"image/openraster", compress_type=zipfile.ZIP_STORED)
        archive.writestr("dinkster/document.json", document.data)
        for descriptor in record["resources"].values():
            archive.writestr(f"dinkster/{descriptor['id']}", document.read_resource(descriptor))
        ordered = ordered_layer_ids(record["layers"], record["rootLayerIds"])
        portable_modes = {
            "normal",
            "multiply",
            "screen",
            "overlay",
            "darken",
            "lighten",
            "color_dodge",
            "color_burn",
            "hard_light",
            "soft_light",
            "difference",
            "hue",
            "saturation",
            "color",
            "luminosity",
        }
        extended_compositing = (
            canvas.get("background")
            or canvas["compositing"] == "linear-premultiplied-alpha"
            or any(
                record["layers"][identifier]["clipping"] != "none"
                or record["layers"][identifier]["blendMode"] not in portable_modes
                or record["layers"][identifier].get("isolation") == "pass-through"
                for identifier in ordered
            )
        )
        if not ordered or extended_compositing:
            if extended_compositing:
                warnings.warn(
                    "OpenRaster uses a merged standard view for extended compositing; "
                    "the native extension preserves all editable layers",
                    stacklevel=2,
                )
            archive.writestr("data/composite.png", merged)
            ET.SubElement(
                stack,
                "layer",
                {
                    "name": "Composite",
                    "src": "data/composite.png",
                    "opacity": "1.0",
                    "visibility": "visible",
                    "x": "0",
                    "y": "0",
                    "composite-op": "svg:src-over",
                },
            )
        else:
            for identifier in reversed(ordered):
                layer = record["layers"][identifier]
                portable = _portable_raster(record, layer)
                path = f"data/{identifier}.png"
                archive.writestr(
                    path,
                    document.read_resource(record["resources"][layer["resourceId"]])
                    if portable
                    else _baked_layer(document, identifier),
                )
                ET.SubElement(
                    stack,
                    "layer",
                    {
                        "name": layer["name"],
                        "src": path,
                        "opacity": _ora_opacity(layer["opacity"]),
                        "visibility": "visible" if layer["visible"] else "hidden",
                        "x": str(layer["transform"]["tx"] // 1_000_000) if portable else "0",
                        "y": str(layer["transform"]["ty"] // 1_000_000) if portable else "0",
                        "composite-op": _ora_blend(layer["blendMode"]),
                    },
                )
        archive.writestr("stack.xml", ET.tostring(root, encoding="utf-8", xml_declaration=True))
        archive.writestr("mergedimage.png", merged)
        archive.writestr("Thumbnails/thumbnail.png", _thumbnail(merged))
        integrity = {
            name: digest_bytes(archive.read(name))
            for name in archive.namelist()
            if name == "stack.xml" or name.startswith("data/")
        }
        archive.writestr("dinkster/standard-digests.json", canonical_json(integrity))
    return output.getvalue()


def import_ora(data: bytes) -> ImageDocument:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > 16384 or sum(entry.file_size for entry in entries) > MAX_ARCHIVE_BYTES:
            raise InvalidDocument("OpenRaster archive exceeds resource limits")
        if len({entry.filename for entry in entries}) != len(entries):
            raise InvalidDocument("OpenRaster archive has duplicate members")
        if archive.read("mimetype") != b"image/openraster":
            raise InvalidDocument("archive is not OpenRaster")
        integrity: dict[str, str] = (
            json.loads(archive.read("dinkster/standard-digests.json"))
            if "dinkster/standard-digests.json" in archive.namelist()
            else {}
        )
        unchanged = bool(integrity) and all(
            name in archive.namelist() and digest_bytes(archive.read(name)) == digest
            for name, digest in integrity.items()
        )
        if "dinkster/document.json" in archive.namelist() and unchanged:
            document = ImageDocument(archive.read("dinkster/document.json"))
            record = document.to_record()
            for identifier, descriptor in record["resources"].items():
                resource = archive.read(f"dinkster/{identifier}")
                if (
                    digest_bytes(resource) != descriptor["digest"]
                    or len(resource) != descriptor["byteSize"]
                ):
                    raise InvalidDocument("OpenRaster native resource integrity mismatch")
                if "inline" not in descriptor:
                    root = os.environ.get("DINKSTER_ASSET_VAULT")
                    if not root:
                        raise InvalidDocument("OpenRaster resources require DINKSTER_ASSET_VAULT")
                    with AssetVault(root).writer(descriptor["digest"]) as writer:
                        writer.write(resource)
                        writer.commit()
            return document
        xml = archive.read("stack.xml")
        if len(xml) > 4 * MEBIBYTE or b"<!ENTITY" in xml or b"<!DOCTYPE" in xml:
            raise InvalidDocument("unsafe OpenRaster XML")
        root = ET.fromstring(xml)
        document = empty_document(int(root.attrib["w"]), int(root.attrib["h"]))

        def load_stack(element: ET.Element, current: ImageDocument) -> ImageDocument:
            for child in reversed(list(element)):
                if child.tag == "stack":
                    before = set(current.to_record()["layers"])
                    current = load_stack(child, current)
                    roots = [
                        identifier
                        for identifier in current.to_record()["rootLayerIds"]
                        if identifier not in before
                    ]
                    if roots:
                        current = apply_commands(
                            current,
                            [{"op": "group", "ids": roots, "name": child.get("name", "Group")}],
                        )
                        identifier = current.to_record()["rootLayerIds"][-1]
                        mode = child.get("composite-op", "svg:src-over").removeprefix("svg:")
                        current = apply_commands(
                            current,
                            [
                                {
                                    "op": "layer",
                                    "id": identifier,
                                    "changes": {
                                        "visible": child.get("visibility", "visible") != "hidden",
                                        "opacity": round(float(child.get("opacity", "1")) * 65535),
                                        "blendMode": "normal"
                                        if mode == "pass-through"
                                        else _import_blend(mode),
                                        "isolation": "pass-through"
                                        if mode == "pass-through"
                                        or child.get("isolation") == "auto"
                                        else "isolated",
                                        "transform": {
                                            **IDENTITY,
                                            "tx": round(float(child.get("x", "0")) * 1_000_000),
                                            "ty": round(float(child.get("y", "0")) * 1_000_000),
                                        },
                                    },
                                }
                            ],
                        )
                elif child.tag == "layer":
                    mode = child.get("composite-op", "svg:src-over").removeprefix("svg:")
                    current = append_raster(
                        current,
                        archive.read(child.attrib["src"]),
                        name=child.get("name", "Layer"),
                        x=float(child.get("x", "0")),
                        y=float(child.get("y", "0")),
                        opacity=float(child.get("opacity", "1")),
                        visible=child.get("visibility", "visible") != "hidden",
                        blend_mode=_import_blend(mode),
                    )
            return current

        stack = root.find("stack")
        if stack is None:
            raise InvalidDocument("OpenRaster has no stack")
        return load_stack(stack, document)


def _import_blend(mode: str) -> str:
    name = "normal" if mode == "src-over" else mode.replace("-", "_").lower()
    if name not in BLEND_MODES:
        warnings.warn(f"Unsupported layered-file blend {mode!r}; using normal", stacklevel=2)
        return "normal"
    return name


def import_psd(data: bytes) -> ImageDocument:
    from psd_tools import PSDImage
    from psd_tools.constants import ColorMode

    psd = cast(Any, PSDImage).open(io.BytesIO(data))
    if psd.depth != 8:
        warnings.warn(f"PSD {psd.depth}-bit channels are rasterized to 8-bit RGBA", stacklevel=2)
    if psd.color_mode != ColorMode.RGB:
        warnings.warn(f"PSD {psd.color_mode.name} color is rasterized to RGBA", stacklevel=2)

    def load_stack(parent: Any, current: ImageDocument) -> ImageDocument:
        for layer in parent:
            name = layer.name.removesuffix("\0")
            if layer.kind not in {"pixel", "group"}:
                warnings.warn(
                    f"PSD {layer.kind} layer {layer.name!r} is rasterized as an ordinary layer",
                    stacklevel=2,
                )
            if layer.has_vector_mask():
                warnings.warn(
                    f"PSD layer {layer.name!r} vector mask is not preserved", stacklevel=2
                )
            if layer.has_effects():
                warnings.warn(f"PSD layer {layer.name!r} effects are not rendered", stacklevel=2)
            before = set(current.to_record()["rootLayerIds"])
            if layer.is_group():
                current = load_stack(layer, current)
                roots = [key for key in current.to_record()["rootLayerIds"] if key not in before]
                if not roots:
                    continue
                current = apply_commands(current, [{"op": "group", "ids": roots, "name": name}])
            else:
                image = layer.topil()
                if image is None:
                    warnings.warn(f"PSD layer {layer.name!r} has no raster pixels", stacklevel=2)
                    continue
                output = io.BytesIO()
                image.convert("RGBA").save(output, "PNG")
                current = append_raster(
                    current, output.getvalue(), name=name, x=layer.left, y=layer.top
                )
            record = current.to_record()
            identifier = record["rootLayerIds"][-1]
            mode = str(layer.blend_mode.name).lower()
            pass_through = layer.is_group() and mode == "pass_through"
            changes = {
                "opacity": round(layer.opacity / 255 * 65535),
                "visible": layer.visible,
                "blendMode": "normal" if pass_through else _import_blend(mode),
                "clipping": "clip-to-previous" if layer.clipping else "none",
            }
            if layer.is_group():
                changes["isolation"] = "pass-through" if pass_through else "isolated"
            current = apply_commands(
                current, [{"op": "layer", "id": identifier, "changes": changes}]
            )
            if layer.mask is not None:
                mask_image = layer.mask.topil(real=False)
                if mask_image is not None:
                    left, top, _, _ = layer.mask.bbox
                    coverage = Image.new(
                        "L", (layer.width, layer.height), layer.mask.background_color
                    )
                    coverage.paste(mask_image, (left - layer.left, top - layer.top))
                    output = io.BytesIO()
                    coverage.save(output, "PNG")
                    record = current.to_record()
                    resource_id = admit_raster(record, output.getvalue())
                    current = apply_commands(
                        ImageDocument.from_record(record),
                        [
                            {
                                "op": "add_mask",
                                "ownerLayerId": identifier,
                                "mask": {
                                    "resourceId": resource_id,
                                    "enabled": not layer.mask.disabled,
                                    "channel": "luminance",
                                    "sourceRect": {
                                        "x": 0,
                                        "y": 0,
                                        "width": layer.width,
                                        "height": layer.height,
                                    },
                                    "transform": {
                                        **IDENTITY,
                                        "tx": layer.left * 1_000_000,
                                        "ty": layer.top * 1_000_000,
                                    },
                                },
                            }
                        ],
                    )
        return current

    return load_stack(psd, empty_document(psd.width, psd.height))


def load_document(data: bytes) -> ImageDocument:
    if data.startswith(b"PK"):
        return import_ora(data)
    if data.startswith(b"8BPS"):
        return import_psd(data)
    if data.startswith(b"{"):
        return ImageDocument(data)
    with Image.open(io.BytesIO(data)) as image:
        document = empty_document(*image.size)
    return append_raster(document, data)
