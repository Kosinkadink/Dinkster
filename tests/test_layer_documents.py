from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from dinkster_api.v1 import InputSpec, Node, NodeSchema, TypeExpr, register_media_types
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, TypedLiteral
from dinkster_image_document.compat import from_comfy_layers, register_comfy_layers
from dinkster_image_document.document import (
    ImageDocument,
    append_raster,
    apply_commands,
    empty_document,
    flatten,
    raster_png,
)
from dinkster_image_document.format import (
    BLEND_MODES,
    IMAGE_DOCUMENT_MEDIA_TYPE,
    INLINE_RESOURCE_LIMIT,
    InvalidDocument,
    canonical_json,
)
from dinkster_image_document.interchange import (
    export_ora,
    export_psd,
    import_ora,
    import_psd,
)
from dinkster_schema import build_node_types
from dinkster_values import (
    RenditionUnavailable,
    TypeRegistry,
    annotate_image,
    media_semantics,
    register_core_types,
)
from dinkster_workers import GroupIsolatedWorker, InProcessWorker
from dinkster_workers.boundary import ValueCodec
from dinkster_workers.saved_artifacts import SavedArtifactAuthority
from PIL import Image

from tests.async_scenarios import run_scenario

pytestmark = pytest.mark.usefixtures("saved_artifact_context")


@pytest.fixture
def layer_document_pack(tmp_path: Path) -> Path:
    (tmp_path / "layer_document_fixture.py").write_text(
        "from dinkster_api.v1 import register_media_types\n"
        "from dinkster_image.compositor import AddLayer\n"
        "from dinkster_image.layer_document import EditLayers, FlattenLayers\n"
        "from dinkster_image.layer_io import SaveLayers\n"
        "from dinkster_image.types import register_image_types\n"
        "NODES = (AddLayer, EditLayers, FlattenLayers, SaveLayers)\n"
        "def register_types(registry):\n"
        "    register_media_types(registry)\n"
        "    register_image_types(registry)\n",
        encoding="ascii",
    )
    manifest = tmp_path / "layer-document.toml"
    manifest.write_text(
        '[pack]\nname="dinkster-image"\npublisher="dinkster"\n'
        'namespaces=["dinkster.layers"]\n'
        'type-namespaces=["dinkster.layers", "comfy.LAYERS", "comfy.COMPOSITOR"]\n'
        "[pack.sandbox]\nwritable-mounts=true\n"
        '[pack.entry]\nnodes="layer_document_fixture:NODES"\n'
        'types="layer_document_fixture:register_types"\n',
        encoding="ascii",
    )
    return manifest


def document() -> ImageDocument:
    return append_raster(
        empty_document(2, 1),
        raster_png(np.array([[[1, 0, 0, 1], [0, 1, 0, 0.5]]], dtype=np.float32)),
    )


def test_comfy_layer_wire_preserves_canonical_document_and_immutability() -> None:
    value = document()
    registry = TypeRegistry()
    register_comfy_layers(registry)
    spec = registry.spec("comfy.LAYERS")
    encoded = spec.encode(value)
    assert encoded == value.data
    assert cast(ImageDocument, spec.decode(encoded)).data == encoded
    record = value.to_record()
    record["canvas"]["width"] = 100
    assert value.to_record()["canvas"]["width"] == 2


def test_document_boundary_preserves_external_and_auxiliary_references(monkeypatch) -> None:
    from dinkster_assets import AssetRef, AssetVault, digest_bytes
    from dinkster_assets.value import value_asset_references
    from dinkster_values import default_encode

    record = append_raster(
        document(), raster_png(np.full((1, 2, 3), 0.25, dtype=np.float32))
    ).to_record()
    record["resources"]["r0"].pop("inline")
    orphan = dict(record["resources"]["r0"], id="r4", digest=digest_bytes(b"orphan"), byteSize=6)
    record["resources"]["r4"] = orphan
    record["allocation"]["nextOrdinal"] = 5
    expected = ImageDocument.from_record(record)

    def forbid_io(*args, **kwargs):
        pytest.fail("encoded document admission must not read, render or publish resources")

    monkeypatch.setattr(ImageDocument, "read_resource", forbid_io)
    monkeypatch.setattr(ImageDocument, "render", forbid_io)
    monkeypatch.setattr(AssetVault, "writer", forbid_io)
    registry = TypeRegistry()
    register_comfy_layers(registry)
    wrapped = registry.wrap("comfy.LAYERS", expected)
    codec = ValueCodec(registry)
    blobs, segments = [], []
    wire, _ = codec.encode(wrapped, blobs, segments)
    auxiliary = AssetRef(digest_bytes(b"auxiliary"), "source.json", 9)
    entries = {
        **wrapped.meta.entries,
        "asset_refs": [
            auxiliary.to_wire(),
            *reversed(cast(list[object], wrapped.meta.get("asset_refs"))),
        ],
    }
    blobs[cast(int, wire["metaBlob"])] = default_encode(entries)
    received, _ = codec.decode(wire, blobs, [])
    assert cast(ImageDocument, received.resolve()).data == expected.data
    assert received.fingerprint == wrapped.fingerprint
    references = value_asset_references({"document": received})
    assert {digest: ref.size for digest, ref in references.items()} == {
        record["resources"]["r0"]["digest"]: record["resources"]["r0"]["byteSize"],
        orphan["digest"]: 6,
        auxiliary.digest: 9,
    }
    assert value_asset_references({"document": received}, complete_registry=registry) == references
    first = TypeRegistry()
    first_spec = first.register("comfy.LAYERS")
    register_comfy_layers(first)
    assert first.spec("comfy.LAYERS") is first_spec
    assert not first.spec("comfy.LAYERS").asset_references_complete


@pytest.mark.parametrize(
    "invalid",
    ["omitted", "last", "size", "digest", "conflict", "malformed", "noncanonical", "payload"],
)
def test_document_boundary_refuses_uncovered_resources_before_exposure(invalid) -> None:
    from dinkster_assets import digest_bytes
    from dinkster_values import default_encode
    from dinkster_workers.boundary import EncodedPayloadValidationError

    record = document().to_record()
    record["resources"]["r0"].pop("inline")
    record["resources"]["r2"] = dict(
        record["resources"]["r0"], id="r2", digest=digest_bytes(b"orphan"), byteSize=6
    )
    record["allocation"]["nextOrdinal"] = 3
    registry = TypeRegistry()
    register_comfy_layers(registry)
    wrapped = registry.wrap("comfy.LAYERS", ImageDocument.from_record(record))
    codec = ValueCodec(registry)
    blobs, segments = [], []
    wire, _ = codec.encode(wrapped, blobs, segments)
    metadata = dict(wrapped.meta.entries)
    refs = [dict(ref) for ref in cast(list[dict[str, object]], metadata["asset_refs"])]
    if invalid == "omitted":
        refs.clear()
    elif invalid == "last":
        refs.pop()
    elif invalid == "size":
        refs[0]["size"] = cast(int, refs[0]["size"]) + 1
    elif invalid == "digest":
        refs[0]["digest"] = digest_bytes(b"wrong")
    elif invalid == "conflict":
        refs.append({**refs[0], "size": cast(int, refs[0]["size"]) + 1})
    elif invalid == "malformed":
        metadata["asset_refs"] = "not a list"
    else:
        payload = cast(dict[str, object], wire["payload"])
        blobs[cast(int, payload["blob"])] = (
            json.dumps(record, indent=2).encode()
            if invalid == "noncanonical"
            else b"not a document"
        )
    if invalid != "malformed":
        metadata["asset_refs"] = refs
    blobs[cast(int, wire["metaBlob"])] = default_encode(metadata)
    with pytest.raises(EncodedPayloadValidationError):
        codec.decode(wire, blobs, [])


def test_document_rendition_needs_no_raster_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    record = document().to_record()
    for resource in record["resources"].values():
        resource.pop("inline")
    expected = ImageDocument.from_record(record)

    def forbid_raster(*args, **kwargs):
        pytest.fail("document retrieval must not render or read raster resources")

    monkeypatch.setattr(ImageDocument, "render", forbid_raster)
    monkeypatch.setattr(ImageDocument, "read_resource", forbid_raster)
    registry = TypeRegistry()
    register_comfy_layers(registry)
    register_comfy_layers(registry)
    spec = registry.spec("comfy.LAYERS")
    value = registry.wrap("comfy.LAYERS", spec.decode(spec.encode(expected)))
    rendition = registry.render(value, "document")
    assert rendition.mime == IMAGE_DOCUMENT_MEDIA_TYPE
    assert rendition.data == expected.data
    assert [s.kind for s in registry.renditions_of("comfy.LAYERS") if s.default] == ["document"]


@pytest.mark.parametrize("mode", BLEND_MODES)
def test_every_document_blend_renders_deterministically(mode: str) -> None:
    value = append_raster(
        document(), raster_png(np.full((1, 2, 3), 0.25)), blend_mode=mode, opacity=0.4
    )
    first = value.render()
    assert first.png == value.render().png
    assert first.pixels.dtype == np.uint16
    assert np.all(first.pixels[..., :3] <= first.pixels[..., 3:4])


def test_canvas_background_is_the_blend_backdrop() -> None:
    value = append_raster(
        empty_document(1, 1), raster_png(np.full((1, 1, 3), 0.5)), blend_mode="multiply"
    )
    value = apply_commands(
        value, [{"op": "canvas", "changes": {"background": [32768, 32768, 32768, 65535]}}]
    )
    image, mask = flatten(value)
    np.testing.assert_allclose(image, 64 / 255, atol=1 / 255, rtol=0)
    np.testing.assert_array_equal(mask, 0)


@pytest.mark.parametrize("combine", ("multiply", "add", "subtract", "intersect"))
@pytest.mark.parametrize("channel", ("alpha", "luminance"))
def test_mask_fields_groups_clipping_and_z_order(combine: str, channel: str) -> None:
    value = document()
    record = value.to_record()
    identifier = record["rootLayerIds"][0]
    resource = record["layers"][identifier]["resourceId"]
    mask = {
        "resourceId": resource,
        "sourceRect": {"x": 0, "y": 0, "width": 2, "height": 1},
        "channel": channel,
        "combineMode": combine,
        "enabled": True,
        "invert": True,
        "opacity": 32768,
    }
    masked = apply_commands(
        value,
        [
            {"op": "add_mask", "ownerLayerId": identifier, "mask": mask},
            {"op": "add_mask", "ownerLayerId": identifier, "mask": mask},
        ],
    )
    assert masked.render().png == masked.render().png
    mask_id = next(iter(masked.to_record()["masks"]))
    assert flatten(masked, f"mask:{mask_id}")[0].shape == (1, 1, 2, 3)
    disabled = apply_commands(
        masked,
        [
            {"op": "mask", "id": mid, "changes": {"enabled": False}}
            for mid in masked.to_record()["masks"]
        ],
    )
    assert disabled.render().png == value.render().png
    group = apply_commands(disabled, [{"op": "group", "ids": [identifier]}])
    assert group.render().png == value.render().png
    removed = apply_commands(group, [{"op": "remove", "id": group.to_record()["rootLayerIds"][0]}])
    assert not removed.to_record()["layers"] and not removed.to_record()["masks"]


def test_transform_components_cannot_compete_with_affine() -> None:
    record = document().to_record()
    layer = record["layers"][record["rootLayerIds"][0]]
    layer["transform"]["components"]["x"] = 2
    with pytest.raises(InvalidDocument, match="disagree"):
        ImageDocument.from_record(record)


def test_clipping_and_grouping_use_z_order_not_object_key_order() -> None:
    value = append_raster(document(), raster_png(np.ones((1, 2, 3))), z_index=10)
    record = value.to_record()
    base, clipped = record["rootLayerIds"]
    record["layers"][base]["z_index"] = -10
    record["layers"][clipped]["clipping"] = "clip-to-previous"
    record["rootLayerIds"] = [clipped, base]
    value = ImageDocument.from_record(record)
    image, _ = flatten(value)
    assert image.shape[-1] == 4
    assert image[0, 0, 1, 3] < 1
    record["layers"][clipped]["clipping"] = "none"
    value = ImageDocument.from_record(record)
    grouped = apply_commands(value, [{"op": "group", "ids": [clipped]}])
    assert grouped.render().png == value.render().png


def test_comfy_missing_canvas_and_zero_dimensions_use_source_size() -> None:
    value = from_comfy_layers(
        {"version": 1, "layers": [{"image": np.ones((1, 2, 3, 3)).tolist(), "w": 0, "h": 0}]}
    )
    assert flatten(value)[0].shape == (1, 2, 3, 3)
    assert flatten(from_comfy_layers({"version": 1, "layers": []}))[0].shape == (1, 1, 1, 4)


def large_document() -> ImageDocument:
    from dinkster_image.types import register_image_types

    value = empty_document(64, 64)
    rng = np.random.default_rng(0)
    for index in range(50):
        output = io.BytesIO()
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(output, "PNG")
        raster = output.getvalue()
        assert len(raster) > INLINE_RESOURCE_LIMIT
        value = append_raster(value, raster, name=f"Layer {index}")
    registry = TypeRegistry()
    register_image_types(registry)
    spec = registry.spec("dinkster.layers")
    assert spec.meta is not None
    meta = spec.meta(value)
    size = len(spec.encode(value)) + len(canonical_json(meta).encode())
    assert size < 1024 * 1024
    assert all("inline" not in resource for resource in value.to_record()["resources"].values())
    assert b"/tmp" not in value.data
    assert len(value.to_record()["resources"]) == 50
    print(f"50x64x64 document plus metadata: {size} bytes")
    return value


def test_ora_archive_exact_reimport_and_independent_stack_fixture() -> None:
    value = document()
    encoded = export_ora(value)
    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        assert archive.infolist()[0].filename == "mimetype"
        assert archive.infolist()[0].compress_type == zipfile.ZIP_STORED
        assert archive.read("mergedimage.png") == value.render().png
    imported = import_ora(encoded)
    assert imported.data == value.data
    assert imported.render().png == value.render().png
    fixture = io.BytesIO()
    pixels = raster_png(np.array([[[0.25, 0.5, 1, 1]]], dtype=np.float32))
    with zipfile.ZipFile(fixture, "w") as archive:
        archive.writestr("mimetype", "image/openraster")
        archive.writestr(
            "stack.xml",
            '<image w="2" h="1"><stack><layer name="Blue" src="data/a.png" x="1" y="0" '
            'opacity="1" composite-op="svg:src-over"/></stack></image>',
        )
        archive.writestr("data/a.png", pixels)
    actual = import_ora(fixture.getvalue())
    np.testing.assert_array_equal(
        flatten(actual)[0],
        np.array([[[[0, 0, 0, 0], [64 / 255, 128 / 255, 1, 1]]]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("opacity", "expected_8_bit", "expected_decimal"),
    [
        (0, 0, "0.0"),
        (1, 0, "0.0"),
        (129, 1, "0.003921568627450981"),
        (32_768, 128, "0.5019607843137256"),
        (49_151, 191, "0.7490196078431374"),
        (65_535, 255, "1.0"),
    ],
)
def test_ora_export_rounds_opacity_for_krita_without_reducing_native_precision(
    opacity: int, expected_8_bit: int, expected_decimal: str
) -> None:
    pixels = raster_png(np.ones((1, 1, 4), dtype=np.float32))
    value = append_raster(empty_document(1, 1), pixels, opacity=opacity / 65_535)

    encoded = export_ora(value)

    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        stack = ET.fromstring(archive.read("stack.xml")).find("stack")
        assert stack is not None
        emitted = list(stack)[0].attrib["opacity"]
    assert emitted == expected_decimal
    assert int(float(emitted) * 255) == expected_8_bit
    imported = import_ora(encoded)
    assert imported.data == value.data
    identifier = imported.to_record()["rootLayerIds"][0]
    assert imported.to_record()["layers"][identifier]["opacity"] == opacity


def test_ora_export_preserves_portable_layer_properties() -> None:
    def solid(width: int, height: int, rgba: tuple[int, int, int, int]) -> bytes:
        pixels = np.empty((height, width, 4), dtype=np.float32)
        pixels[:] = np.asarray(rgba, dtype=np.float32) / 255
        return raster_png(pixels)

    value = empty_document(8, 6)
    value = append_raster(value, solid(8, 6, (32, 64, 192, 64)), name="Base")
    value = append_raster(
        value,
        solid(4, 4, (224, 32, 16, 128)),
        name="Offset",
        x=-2,
        y=1,
        opacity=0.5,
        blend_mode="multiply",
    )
    value = append_raster(
        value,
        solid(2, 2, (16, 240, 64, 255)),
        name="Hidden",
        x=4,
        opacity=0.75,
        blend_mode="screen",
        visible=False,
    )
    value = append_raster(value, solid(1, 1, (0, 0, 0, 0)), name="Empty", x=6, y=5)
    encoded = export_ora(value)
    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        assert "Thumbnails/thumbnail.png" in archive.namelist()
        stack = ET.fromstring(archive.read("stack.xml")).find("stack")
        assert stack is not None
        layers = list(stack)
        assert [layer.get("name") for layer in layers] == ["Empty", "Hidden", "Offset", "Base"]
        opacity_values = [layer.get("opacity") for layer in layers]
        assert opacity_values == [
            "1.0",
            "0.7490196078431374",
            "0.5019607843137256",
            "1.0",
        ]
        # Krita converts the ORA decimal times 255 directly to an unsigned byte.
        assert [int(float(opacity) * 255) for opacity in opacity_values if opacity is not None] == [
            255,
            191,
            128,
            255,
        ]
        assert [
            {
                "visibility": layer.get("visibility"),
                "opacity": round(float(layer.get("opacity", "0")) * 65_535),
                "x": int(layer.get("x", "0")),
                "y": int(layer.get("y", "0")),
                "composite-op": layer.get("composite-op"),
            }
            for layer in layers
        ] == [
            {
                "visibility": "visible",
                "opacity": 65_535,
                "x": 6,
                "y": 5,
                "composite-op": "svg:src-over",
            },
            {
                "visibility": "hidden",
                "opacity": 49_087,
                "x": 4,
                "y": 0,
                "composite-op": "svg:screen",
            },
            {
                "visibility": "visible",
                "opacity": 32_896,
                "x": -2,
                "y": 1,
                "composite-op": "svg:multiply",
            },
            {
                "visibility": "visible",
                "opacity": 65_535,
                "x": 0,
                "y": 0,
                "composite-op": "svg:src-over",
            },
        ]
        with Image.open(io.BytesIO(archive.read(cast(str, layers[1].get("src"))))) as hidden:
            assert np.asarray(hidden.convert("RGBA"))[0, 0, 3] == 255
        standard = io.BytesIO()
        with zipfile.ZipFile(standard, "w") as stripped:
            for entry in archive.infolist():
                if not entry.filename.startswith("dinkster/"):
                    stripped.writestr(entry, archive.read(entry.filename))
    imported = import_ora(standard.getvalue())
    imported_record = imported.to_record()
    imported_layers = [
        imported_record["layers"][identifier]
        for identifier in reversed(imported_record["rootLayerIds"])
    ]
    assert [layer["name"] for layer in imported_layers] == ["Empty", "Hidden", "Offset", "Base"]
    assert [layer["visible"] for layer in imported_layers] == [True, False, True, True]
    assert [layer["opacity"] for layer in imported_layers] == [65_535, 49_087, 32_896, 65_535]
    assert [layer["transform"]["tx"] for layer in imported_layers] == [
        6_000_000,
        4_000_000,
        -2_000_000,
        0,
    ]
    assert [layer["transform"]["ty"] for layer in imported_layers] == [
        5_000_000,
        0,
        1_000_000,
        0,
    ]
    assert imported.render().png == value.render().png


def test_ora_export_bakes_only_nonportable_raster_geometry_and_masks() -> None:
    pixels = raster_png(np.ones((2, 2, 4), dtype=np.float32))
    mask = raster_png(np.zeros((2, 2), dtype=np.float32))
    value = append_raster(
        empty_document(4, 4),
        pixels,
        mask_data=mask,
        name="Baked",
        x=0.5,
        y=1,
        rotation=0.25,
        opacity=0.5,
        visible=False,
    )
    with zipfile.ZipFile(io.BytesIO(export_ora(value))) as archive:
        stack = ET.fromstring(archive.read("stack.xml")).find("stack")
        assert stack is not None
        layer = list(stack)[0]
        assert layer.attrib | {"src": "ignored"} == {
            "name": "Baked",
            "src": "ignored",
            "opacity": "0.5019607843137256",
            "visibility": "hidden",
            "x": "0",
            "y": "0",
            "composite-op": "svg:src-over",
        }
        with Image.open(io.BytesIO(archive.read(layer.attrib["src"]))) as baked:
            assert baked.size == (4, 4)
            assert np.max(np.asarray(baked)[..., 3]) > 0


def test_ora_export_uses_composite_for_empty_and_nonbaseline_stacks() -> None:
    empty = empty_document(4, 3)
    with zipfile.ZipFile(io.BytesIO(export_ora(empty))) as archive:
        stack = ET.fromstring(archive.read("stack.xml")).find("stack")
        assert stack is not None
        layers = list(stack)
        assert [layer.get("name") for layer in layers] == ["Composite"]
        assert layers[0].get("src") == "data/composite.png"
        with Image.open(io.BytesIO(archive.read("data/composite.png"))) as composite:
            assert np.max(np.asarray(composite.convert("RGBA"))) == 0
    assert import_ora(export_ora(empty)).data == empty.data

    exclusion = append_raster(
        empty_document(1, 1),
        raster_png(np.ones((1, 1, 4), dtype=np.float32)),
        blend_mode="exclusion",
    )
    with pytest.warns(UserWarning, match="merged standard view"):
        encoded = export_ora(exclusion)
    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        stack = ET.fromstring(archive.read("stack.xml")).find("stack")
        assert stack is not None
        layers = list(stack)
        assert [layer.get("composite-op") for layer in layers] == ["svg:src-over"]
        assert [name for name in archive.namelist() if name.startswith("data/")] == [
            "data/composite.png"
        ]


def test_seedream_style_document_uses_transparency_masks_and_z_index() -> None:
    background = np.zeros((1, 1, 2, 3), dtype=np.float32)
    foreground = np.ones((1, 1, 1, 3), dtype=np.float32)
    value = from_comfy_layers(
        {
            "version": 1,
            "canvas": (2, 1),
            "layers": [
                {
                    "type": "raster",
                    "image": foreground,
                    "mask": np.zeros((1, 1, 1)),
                    "x": 1,
                    "z_index": 10,
                },
                {"type": "raster", "image": background, "z_index": 0},
            ],
        }
    )
    np.testing.assert_array_equal(
        flatten(value)[0], np.array([[[[0, 0, 0], [1, 1, 1]]]], dtype=np.float32)
    )


def test_psd_real_raster_fixture() -> None:
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer

    psd = PSDImage.new("RGB", (2, 1))
    PixelLayer.frompil(Image.new("RGBA", (2, 1), (255, 0, 0, 255)), psd, name="Red")
    PixelLayer.frompil(Image.new("RGBA", (1, 1), (0, 255, 0, 255)), psd, name="Green", left=1)
    output = io.BytesIO()
    psd.save(output)
    loaded = import_psd(output.getvalue())
    assert [layer["name"] for layer in loaded.to_record()["layers"].values()] == ["Red", "Green"]
    np.testing.assert_array_equal(flatten(loaded)[0], np.array([[[[1, 0, 0], [0, 1, 0]]]]))


def test_psd_import_removes_krita_unicode_name_terminators() -> None:
    from psd_tools import PSDImage
    from psd_tools.api.layers import Group, PixelLayer

    psd = PSDImage.new("RGB", (2, 1))
    group = Group.new(psd, name="Group\0")
    PixelLayer.frompil(
        Image.new("RGBA", (2, 1), (255, 0, 0, 255)), group, name="Raster \u03a9 name\0"
    )
    output = io.BytesIO()
    psd.save(output)

    imported = import_psd(output.getvalue())
    record = imported.to_record()
    group_record = record["layers"][record["rootLayerIds"][0]]
    raster_record = record["layers"][group_record["childLayerIds"][0]]
    assert group_record["name"] == "Group"
    assert raster_record["name"] == "Raster \u03a9 name"
    np.testing.assert_array_equal(flatten(imported)[0], np.array([[[[1, 0, 0], [1, 0, 0]]]]))

    reexported = PSDImage.open(io.BytesIO(export_psd(imported)))
    assert [layer.name for layer in reexported] == ["Group"]
    assert [layer.name for layer in cast(Any, reexported[0])] == ["Raster \u03a9 name"]


def _psd_acceptance_document() -> ImageDocument:
    value = append_raster(
        empty_document(6, 4),
        raster_png(np.full((4, 6, 4), (32 / 255, 64 / 255, 1, 1), dtype=np.float32)),
        name="Backdrop",
    )
    value = append_raster(
        value,
        raster_png(np.full((2, 3, 4), (1, 32 / 255, 0, 191 / 255), dtype=np.float32)),
        mask_data=raster_png(np.zeros((2, 3), dtype=np.float32)),
        name="Masked offset",
        x=1,
        y=1,
        opacity=0.5,
        blend_mode="multiply",
    )
    value = append_raster(
        value,
        raster_png(np.full((1, 2, 4), (0, 1, 0, 1), dtype=np.float32)),
        name="Hidden",
        x=4,
        visible=False,
        blend_mode="screen",
    )
    record = value.to_record()
    masked, hidden = record["rootLayerIds"][-2:]
    value = apply_commands(
        value,
        [{"op": "group", "ids": [masked, hidden], "name": "Edits"}],
    )
    group = value.to_record()["rootLayerIds"][-1]
    return apply_commands(
        value,
        [{"op": "layer", "id": group, "changes": {"isolation": "pass-through"}}],
    )


def test_psd_export_round_trips_standard_layer_properties_and_pixels() -> None:
    from psd_tools import PSDImage

    value = append_raster(
        empty_document(4, 2),
        raster_png(np.full((2, 3, 3), (0, 0, 1), dtype=np.float32)),
        name="Backdrop",
    )
    value = append_raster(
        value,
        raster_png(np.full((1, 2, 4), (1, 0, 0, 0.5), dtype=np.float32)),
        mask_data=raster_png(np.array([[0, 0.5]], dtype=np.float32)),
        name="Masked offset",
        x=-1,
        y=1,
        opacity=0.5,
        blend_mode="multiply",
    )
    value = append_raster(
        value,
        raster_png(np.full((1, 1, 3), (0, 1, 0), dtype=np.float32)),
        name="Hidden clipped",
        x=2,
        visible=False,
        blend_mode="screen",
    )
    record = value.to_record()
    offset, hidden = record["rootLayerIds"][-2:]
    value = apply_commands(value, [{"op": "group", "ids": [offset, hidden], "name": "Edits"}])
    group = value.to_record()["rootLayerIds"][-1]
    value = apply_commands(
        value,
        [
            {
                "op": "layer",
                "id": group,
                "changes": {"isolation": "pass-through", "opacity": 49_151},
            },
            {
                "op": "layer",
                "id": hidden,
                "changes": {"clipping": "clip-to-previous"},
            },
        ],
    )

    encoded = export_psd(value)
    psd = PSDImage.open(io.BytesIO(encoded))
    with (
        Image.open(io.BytesIO(encoded)) as merged,
        Image.open(io.BytesIO(value.render().png)) as expected_merged,
    ):
        assert merged.format == "PSD"
        np.testing.assert_allclose(
            np.asarray(merged.convert("RGBA")),
            np.asarray(expected_merged.convert("RGBA")),
            atol=1,
            rtol=0,
        )
    assert [layer.name for layer in psd] == ["Backdrop", "Edits"]
    assert [layer.name for layer in cast(Any, psd[1])] == ["Masked offset", "Hidden clipped"]

    imported = import_psd(encoded)
    imported_record = imported.to_record()
    backdrop_id, group_id = imported_record["rootLayerIds"]
    imported_group = imported_record["layers"][group_id]
    masked_id, hidden_id = imported_group["childLayerIds"]
    masked = imported_record["layers"][masked_id]
    hidden_layer = imported_record["layers"][hidden_id]
    assert imported_record["layers"][backdrop_id]["name"] == "Backdrop"
    assert imported_group["name"] == "Edits"
    assert imported_group["isolation"] == "pass-through"
    assert imported_group["opacity"] == round(round(49_151 / 65_535 * 255) / 255 * 65_535)
    assert masked["name"] == "Masked offset"
    assert masked["transform"]["tx"] == -1_000_000
    assert masked["transform"]["ty"] == 1_000_000
    assert masked["blendMode"] == "multiply"
    assert masked["opacity"] == round(round(32_768 / 65_535 * 255) / 255 * 65_535)
    assert len(masked["maskIds"]) == 1
    assert hidden_layer["visible"] is False
    assert hidden_layer["blendMode"] == "screen"
    assert hidden_layer["clipping"] == "clip-to-previous"
    np.testing.assert_allclose(flatten(imported)[0], flatten(value)[0], atol=1 / 255, rtol=0)
    reexported = PSDImage.open(io.BytesIO(export_psd(imported)))
    assert [layer.name for layer in reexported] == ["Backdrop", "Edits"]


def test_psd_export_matches_fixed_editor_acceptance_contract() -> None:
    from psd_tools import PSDImage

    encoded = export_psd(_psd_acceptance_document())
    with Image.open(io.BytesIO(encoded)) as merged:
        pixels = merged.convert("RGBA").tobytes()
    assert len(pixels) == 96
    assert hashlib.sha256(pixels).hexdigest() == (
        "06590d2ccfd36d2a4b5337cde8755f7bd5332a37ce826c1a44ee21d68d029900"
    )
    psd = PSDImage.open(io.BytesIO(encoded))
    assert (psd.width, psd.height) == (6, 4)
    assert [layer.name for layer in psd] == ["Backdrop", "Edits"]
    group = cast(Any, psd[1])
    assert group.blend_mode.name == "PASS_THROUGH"
    assert [layer.name for layer in group] == ["Masked offset", "Hidden"]
    assert (group[0].left, group[0].top, group[0].opacity) == (1, 1, 128)
    assert group[0].mask is not None and not group[0].mask.disabled
    assert (group[1].left, group[1].top, group[1].visible) == (4, 0, False)


def test_psd_export_round_trips_disabled_mask_state() -> None:
    value = _psd_acceptance_document()
    record = value.to_record()
    group = record["layers"][record["rootLayerIds"][-1]]
    masked = record["layers"][group["childLayerIds"][0]]
    mask_id = masked["maskIds"][0]
    value = apply_commands(value, [{"op": "mask", "id": mask_id, "changes": {"enabled": False}}])

    imported = import_psd(export_psd(value))
    imported_record = imported.to_record()
    imported_group = imported_record["layers"][imported_record["rootLayerIds"][-1]]
    imported_masked = imported_record["layers"][imported_group["childLayerIds"][0]]
    imported_mask = imported_record["masks"][imported_masked["maskIds"][0]]
    assert imported_mask["enabled"] is False
    np.testing.assert_array_equal(flatten(imported)[0], flatten(value)[0])


@pytest.mark.parametrize("mode", [mode for mode in BLEND_MODES if not mode.startswith("grain_")])
def test_psd_export_preserves_every_representable_blend_mode(mode: str) -> None:
    value = append_raster(
        empty_document(1, 1), raster_png(np.ones((1, 1, 4), dtype=np.float32)), blend_mode=mode
    )
    imported = import_psd(export_psd(value))
    layer = imported.to_record()["layers"][imported.to_record()["rootLayerIds"][0]]
    assert layer["blendMode"] == mode


def test_psd_export_uses_merged_view_for_unsupported_structure() -> None:
    from psd_tools import PSDImage

    value = append_raster(
        empty_document(2, 1),
        raster_png(np.array([[[1, 0, 0, 0.5], [0, 1, 0, 1]]], dtype=np.float32)),
        blend_mode="grain_extract",
    )
    with pytest.warns(UserWarning, match="merged standard view"):
        encoded = export_psd(value)
    psd = PSDImage.open(io.BytesIO(encoded))
    assert [layer.name for layer in psd] == ["Composite"]
    np.testing.assert_array_equal(flatten(import_psd(encoded))[0], flatten(value)[0])

    value = append_raster(document(), raster_png(np.ones((1, 2, 4), dtype=np.float32)))
    top = value.to_record()["rootLayerIds"][-1]
    value = apply_commands(value, [{"op": "group", "ids": [top]}])
    group = value.to_record()["rootLayerIds"][-1]
    value = apply_commands(
        value,
        [{"op": "layer", "id": group, "changes": {"clipping": "clip-to-previous"}}],
    )
    with pytest.warns(UserWarning, match="group geometry, clipping, or mask"):
        encoded = export_psd(value)
    assert [layer.name for layer in PSDImage.open(io.BytesIO(encoded))] == ["Composite"]


@pytest.mark.parametrize("depth", [16, 32])
def test_psd_import_reports_rasterized_capability_gaps(
    monkeypatch: pytest.MonkeyPatch, depth: int
) -> None:
    from psd_tools import PSDImage
    from psd_tools.constants import ColorMode

    class Layer:
        def __init__(self, kind: str, name: str, vector_mask: bool) -> None:
            self.kind = kind
            self.name = name
            self.left = 0
            self.top = 0
            self.width = 1
            self.height = 1
            self.opacity = 255
            self.visible = True
            self.clipping = False
            self.blend_mode = SimpleNamespace(name="NORMAL")
            self.mask = None
            self._vector_mask = vector_mask

        def has_effects(self) -> bool:
            return False

        def has_vector_mask(self) -> bool:
            return self._vector_mask

        def is_group(self) -> bool:
            return False

        def topil(self) -> Image.Image:
            return Image.new("RGBA", (1, 1), (255, 0, 0, 255))

    class Document(list[Layer]):
        width = 1
        height = 1
        color_mode = ColorMode.CMYK

        def __init__(self, layers: list[Layer], depth: int) -> None:
            super().__init__(layers)
            self.depth = depth

    monkeypatch.setattr(
        PSDImage,
        "open",
        lambda _: Document(
            [Layer("type", "Caption", True), Layer("smartobject", "Placed", False)], depth
        ),
    )
    with pytest.warns(UserWarning) as caught:
        imported = import_psd(b"8BPS")
    messages = [str(item.message) for item in caught]
    assert any(f"{depth}-bit channels are rasterized to 8-bit RGBA" in item for item in messages)
    assert any("CMYK color is rasterized to RGBA" in item for item in messages)
    assert any(
        "type layer 'Caption' is rasterized as an ordinary layer" in item for item in messages
    )
    assert any(
        "smartobject layer 'Placed' is rasterized as an ordinary layer" in item for item in messages
    )
    assert any("layer 'Caption' vector mask is not preserved" in item for item in messages)
    assert len(imported.to_record()["rootLayerIds"]) == 2


@pytest.mark.parametrize("background", [0, 255])
def test_psd_mask_preserves_outside_bounds_background(background: int) -> None:
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer

    psd = PSDImage.new("RGB", (3, 1))
    layer = PixelLayer.frompil(Image.new("RGB", (3, 1), (255, 0, 0)), psd, name="Red")
    if layer.has_mask():
        layer.remove_mask()
    layer.create_mask(Image.new("L", (1, 1), 255), left=1, top=0)
    cast(Any, layer)._record.mask_data.background_color = background
    output = io.BytesIO()
    psd.save(output)
    rendered = np.asarray(Image.open(io.BytesIO(import_psd(output.getvalue()).render().png)))
    np.testing.assert_array_equal(rendered[..., 3], [[background, 255, background]])
    np.testing.assert_array_equal(
        rendered[..., 0], [[255 if background else 0, 255, 255 if background else 0]]
    )


def test_comfy_layer_coercion_preserves_premultiplied_color() -> None:
    color = {"primaries": 9, "transfer": 16, "range": 2}
    pixels = annotate_image(
        np.array([[[[0.5, 0, 0, 0.5]]]], np.float32), alpha="premultiplied", color=color
    )
    value = from_comfy_layers({"version": 1, "canvas": (1, 1), "layers": [{"image": pixels}]})
    actual, mask = flatten(value)
    np.testing.assert_array_equal(actual, [[[[1, 0, 0, np.float32(128 / 255)]]]])
    assert media_semantics(actual)["color"] == color
    np.testing.assert_array_equal(mask, 1 - actual[..., 3])


@pytest.mark.parametrize("opacity", [0, 128, 255])
@pytest.mark.parametrize("mask", [None, 0, 128, 255])
def test_psd_pass_through_group_preserves_backdrop_blending(opacity: int, mask: int | None) -> None:
    from psd_tools import PSDImage
    from psd_tools.api.layers import Group, PixelLayer
    from psd_tools.composite import composite
    from psd_tools.constants import BlendMode

    psd = PSDImage.new("RGB", (3, 1))
    PixelLayer.frompil(Image.new("RGB", (3, 1), (0, 0, 255)), psd, name="Blue")
    group = Group.new(psd, name="Pass through")
    group.blend_mode = BlendMode.PASS_THROUGH
    group.opacity = opacity
    layer = PixelLayer.frompil(Image.new("RGB", (1, 1), (255, 0, 0)), group, name="Multiply red")
    layer.blend_mode = BlendMode.MULTIPLY
    if mask is not None:
        group.create_mask(Image.new("L", (1, 1), mask))
    output = io.BytesIO()
    psd.save(output)
    color, _, alpha = composite(PSDImage.open(io.BytesIO(output.getvalue())), force=True)
    # Round reference float pixels; its PIL path truncates opaque 0.99999994 alpha to 254.
    reference = np.rint(np.concatenate((color, alpha), axis=-1) * 255).astype(np.uint8)
    value = import_psd(output.getvalue())
    actual = Image.open(io.BytesIO(value.render().png))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
    identifier = value.to_record()["rootLayerIds"][-1]
    assert value.to_record()["layers"][identifier]["isolation"] == "pass-through"
    moved = apply_commands(
        value,
        [
            {
                "op": "layer",
                "id": identifier,
                "changes": {
                    "transform": {
                        "a": 1_000_000,
                        "b": 0,
                        "c": 0,
                        "d": 1_000_000,
                        "tx": 1_000_000,
                        "ty": 0,
                    }
                },
            }
        ],
    )
    shifted = np.asarray(Image.open(io.BytesIO(moved.render().png)))
    np.testing.assert_array_equal(shifted[0, 1], np.asarray(reference)[0, 0])
    np.testing.assert_array_equal(shifted[0, [0, 2]], [[0, 0, 255, 255]] * 2)
    with pytest.warns(UserWarning, match="merged standard view"):
        assert import_ora(export_ora(moved)).data == moved.data


@pytest.mark.parametrize("backdrop_alpha", [0, 128, 255])
@pytest.mark.parametrize("child_alpha", [128, 255])
@pytest.mark.parametrize("opacity", [0, 32768, 65535])
def test_linear_pass_through_normal_child_matches_isolated_group(
    backdrop_alpha: int, child_alpha: int, opacity: int
) -> None:
    value = append_raster(
        empty_document(1, 1),
        raster_png(np.array([[[0, 0, 1, backdrop_alpha / 255]]], np.float32)),
    )
    value = append_raster(value, raster_png(np.array([[[1, 0, 0, child_alpha / 255]]], np.float32)))
    leaf = value.to_record()["rootLayerIds"][-1]
    value = apply_commands(value, [{"op": "group", "ids": [leaf]}])
    record = value.to_record()
    record["canvas"]["compositing"] = "linear-premultiplied-alpha"
    group = record["layers"][record["rootLayerIds"][-1]]
    group["opacity"] = opacity
    isolated = ImageDocument.from_record(record).render()
    group["isolation"] = "pass-through"
    passed = ImageDocument.from_record(record).render()
    reference = np.asarray(Image.open(io.BytesIO(isolated.png)))
    actual = np.asarray(Image.open(io.BytesIO(passed.png)))
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(passed.pixels[..., 3], isolated.pixels[..., 3])
    if backdrop_alpha == child_alpha == 255 and opacity == 32768:
        np.testing.assert_array_equal(actual, [[[188, 0, 188, 255]]])


@pytest.mark.parametrize("mask_opacity", [32768, 65535])
@pytest.mark.parametrize("translation", [0, 1])
@pytest.mark.parametrize("clipping", ["none", "clip-to-previous"])
def test_linear_pass_through_group_preserves_mask_opacity_transform_and_clipping(
    mask_opacity: int, translation: int, clipping: str
) -> None:
    value = append_raster(
        empty_document(4, 1),
        raster_png(
            np.array([[[0, 0, 1, alpha / 255] for alpha in (0, 128, 255, 128)]], np.float32)
        ),
    )
    record = value.to_record()
    mask_resource = record["layers"][record["rootLayerIds"][0]]["resourceId"]
    value = append_raster(value, raster_png(np.array([[[1, 0, 0]] * 4], np.float32)))
    leaf = value.to_record()["rootLayerIds"][-1]
    value = apply_commands(value, [{"op": "group", "ids": [leaf]}])
    identifier = value.to_record()["rootLayerIds"][-1]
    value = apply_commands(
        value,
        [
            {
                "op": "add_mask",
                "ownerLayerId": identifier,
                "mask": {
                    "resourceId": mask_resource,
                    "sourceRect": {"x": 0, "y": 0, "width": 4, "height": 1},
                    "channel": "alpha",
                    "opacity": mask_opacity,
                },
            }
        ],
    )
    record = value.to_record()
    record["canvas"]["compositing"] = "linear-premultiplied-alpha"
    group = record["layers"][identifier]
    group["opacity"] = 32768
    group["transform"]["tx"] = translation * 1_000_000
    group["clipping"] = clipping
    isolated = ImageDocument.from_record(record).render()
    group["isolation"] = "pass-through"
    passed = ImageDocument.from_record(record).render()
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(passed.png))),
        np.asarray(Image.open(io.BytesIO(isolated.png))),
    )
    np.testing.assert_array_equal(passed.pixels[..., 3], isolated.pixels[..., 3])


def test_nested_pass_through_groups_visit_each_raster_once(monkeypatch) -> None:
    import dinkster_image_document.render as renderer

    value = append_raster(empty_document(1, 1), raster_png(np.array([[[0, 0, 1]]], np.float32)))
    value = append_raster(
        value, raster_png(np.array([[[1, 0, 0]]], np.float32)), blend_mode="multiply"
    )
    for _ in range(20):
        identifier = value.to_record()["rootLayerIds"][-1]
        value = apply_commands(value, [{"op": "group", "ids": [identifier]}])
        identifier = value.to_record()["rootLayerIds"][-1]
        value = apply_commands(
            value, [{"op": "layer", "id": identifier, "changes": {"isolation": "pass-through"}}]
        )
    visits = 0
    original = renderer._composite

    def counted(*args, **kwargs):
        nonlocal visits
        visits += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(renderer, "_composite", counted)
    np.testing.assert_array_equal(flatten(value)[0], [[[[0, 0, 0]]]])
    assert visits == 2


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_lazy_document_input_retains_vault_until_hook_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    from contextlib import suppress

    from dinkster_assets.scratch import claim_inactive_produced_scratch, hold_produced_vault
    from dinkster_image.compositor_types import (
        CompositorSourceLayer,
        LayerStack,
        decode_layer_stack,
        encode_layer_stack,
        layer_stack_meta,
    )
    from dinkster_image.types import register_image_types
    from dinkster_protocol import LazyStatusInvocation
    from dinkster_values import CORE_BOOLEAN
    from dinkster_workers.host import serve_connection
    from dinkster_workers.session import BoundarySession

    root = tmp_path / "vault"
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(root))
    with closing(hold_produced_vault(root)):
        pass
    registry = TypeRegistry()
    register_core_types(registry)
    register_media_types(registry)
    register_image_types(registry)
    old = TypeRegistry()
    old.register(
        "dinkster.layers",
        encode=encode_layer_stack,
        decode=decode_layer_stack,
        meta=layer_stack_meta,
    )
    pixels = np.random.default_rng(41).random((1, 64, 64, 3), dtype=np.float32)
    source = old.wrap(
        "dinkster.layers",
        LayerStack((CompositorSourceLayer(pixels, None, "Noise", 0, 0, 64, 64),)),
    )
    blobs: list[bytes] = []
    wire, _ = ValueCodec(old, use_shm=False).encode(source, blobs, [])
    source, _ = ValueCodec(registry, use_shm=False).decode(wire, blobs, [])

    async def run() -> None:
        entered, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class MeasuredGate(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema(
                    "test.lazy-document",
                    inputs=(
                        InputSpec("condition", TypeExpr.concrete(CORE_BOOLEAN)),
                        InputSpec("value", TypeExpr.concrete("dinkster.layers"), lazy=True),
                    ),
                )

            @classmethod
            async def check_lazy_status(cls, *, condition: bool, value: object) -> tuple[str, ...]:
                assert isinstance(value, ImageDocument)
                resource = next(iter(value.to_record()["resources"].values()))
                assert "inline" not in resource
                assert value.read_resource(resource) == raster_png(pixels[0])
                entered.set()
                try:
                    await release.wait()
                    if outcome == "error":
                        raise ValueError("controlled lazy hook failure")
                    return ()
                finally:
                    settled.set()

        worker = InProcessWorker(build_node_types([MeasuredGate]), registry)
        accepted: asyncio.Future[asyncio.Task[None]] = asyncio.get_running_loop().create_future()

        async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = asyncio.current_task()
            assert task is not None
            accepted.set_result(task)
            await serve_connection(
                reader,
                writer,
                pack_name="lazy-document",
                worker=worker,
                schemas=worker.schemas,
                planner=None,
                consumers={},
                codec=ValueCodec(registry, use_shm=False),
            )

        server = await asyncio.start_server(accept, "127.0.0.1", 0)
        reader, writer = await asyncio.open_connection(*server.sockets[0].getsockname())
        host_task = await accepted
        session = BoundarySession(
            registry, role="test", pack="lazy-document", codec=ValueCodec(registry, use_shm=False)
        )
        try:
            await session.begin(reader, writer, timeout=5)
            schema = MeasuredGate.define_schema()
            request = asyncio.create_task(
                session.check_lazy_status(
                    LazyStatusInvocation(
                        "lazy-document",
                        "gate",
                        schema.node_type,
                        {"condition": registry.wrap(CORE_BOOLEAN, True), "value": source},
                        (),
                        schema,
                    )
                )
            )
            await asyncio.wait_for(entered.wait(), 5)
            with claim_inactive_produced_scratch(root / ".produced-collection") as inactive:
                assert not inactive
            if outcome == "cancel":
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                await asyncio.wait_for(settled.wait(), 5)
            else:
                release.set()
                result = await asyncio.wait_for(request, 5)
                if outcome == "error":
                    assert result.error is not None
                    assert "controlled lazy hook failure" in result.error.message
                else:
                    assert result.error is None and result.requested_inputs == ()
            with claim_inactive_produced_scratch(root / ".produced-collection") as inactive:
                assert inactive
        finally:
            release.set()
            with suppress(Exception):
                await session.send({"type": "shutdown"}, [])
            await session.close()
            await asyncio.wait_for(host_task, 5)
            server.close()
            await server.wait_closed()

    asyncio.run(run())


@pytest.mark.parametrize(
    "case", ["modern", "legacy", "missing", "truncated", "corrupt", "host-missing"]
)
def test_isolated_document_rendition_adopts_all_resources_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, layer_document_pack: Path
) -> None:
    from dinkster_assets import AssetError, AssetVault, digest_bytes
    from dinkster_assets.produced import ProducedAssetInventory
    from dinkster_image.compositor_types import (
        CompositorSourceLayer,
        LayerStack,
        decode_layer_stack,
        encode_layer_stack,
        layer_stack_meta,
    )
    from dinkster_image.types import register_image_types

    producer = AssetVault(tmp_path / "producer")
    host = AssetVault(tmp_path / "host")
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(host.root))
    registry = TypeRegistry()
    register_core_types(registry)
    register_media_types(registry)
    register_image_types(registry)
    codec = ValueCodec(registry, use_shm=False)
    pixels = np.random.default_rng(41).random((1, 64, 64, 3), dtype=np.float32)
    expected = raster_png(pixels[0])
    if case == "legacy":
        old = TypeRegistry()
        old.register(
            "dinkster.layers",
            encode=encode_layer_stack,
            decode=decode_layer_stack,
            meta=layer_stack_meta,
        )
        source = old.wrap(
            "dinkster.layers",
            LayerStack((CompositorSourceLayer(pixels, None, "Noise", 0, 0, 64, 64),)),
        )
        blobs, segments = [], []
        wire, _ = ValueCodec(old, use_shm=False).encode(source, blobs, segments)
        source, _ = codec.decode(wire, blobs, [])
    else:
        digest = digest_bytes(expected)
        with producer.writer(digest) as writer:
            writer.write(expected)
            writer.commit()
        if case != "host-missing":
            with host.writer(digest) as writer:
                writer.write(expected)
                writer.commit()
        path = producer.resolve(digest)
        assert path is not None
        if case == "missing":
            path.unlink()
        elif case == "truncated":
            path.write_bytes(expected[:-1])
        elif case == "corrupt":
            path.write_bytes(expected[:-1] + bytes([expected[-1] ^ 1]))
        record = empty_document(64, 64).to_record()
        record["allocation"]["nextOrdinal"] = 1
        record["resources"]["r0"] = {
            "id": "r0",
            "kind": "raster",
            "digest": digest,
            "byteSize": len(expected),
            "mediaType": "image/png",
            "width": 64,
            "height": 64,
            "colorSpace": "srgb",
            "channelDepth": 8,
            "alphaMode": "opaque",
        }
        source = registry.wrap("dinkster.layers", ImageDocument.from_record(record))

    before_blobs, before_segments = [], []
    before, _ = codec.encode(source, before_blobs, before_segments)
    fingerprint = source.fingerprint

    async def run() -> None:
        group = GroupIsolatedWorker(
            "rendition-test",
            [layer_document_pack],
            registry,
            extra_env={"DINKSTER_ASSET_VAULT": str(producer.root)},
        )
        await group.start()
        try:
            worker = group.members["dinkster-image"]
            if case in ("missing", "truncated", "corrupt", "host-missing"):
                error = AssetError if case == "host-missing" else RenditionUnavailable
                with pytest.raises(error):
                    await worker.render_rendition(source, "document", {})
                assert (
                    await worker.resolve_rendition_mime(
                        source.type_id, "document", source.meta.entries
                    )
                    == "application/vnd.dinkster.image-document+json"
                )
                return
            rendered = await worker.render_rendition(source, "document", {})
        finally:
            await group.close()
        record = ImageDocument(rendered.data).to_record()
        resources = list(record["resources"].values())
        assert len(resources) == 1
        descriptor = resources[0]
        assert "inline" not in descriptor
        assert descriptor["digest"] == digest_bytes(expected)
        assert descriptor["byteSize"] == len(expected)
        path = host.resolve(descriptor["digest"])
        assert path is not None
        assert path.read_bytes() == expected
        inventory = ProducedAssetInventory(host.root)
        if case == "legacy":
            assert [entry.digest for entry in inventory.entries()] == [descriptor["digest"]]
        else:
            with pytest.raises(AssetError, match="inventory is unavailable"):
                inventory.entries()

    asyncio.run(run())
    after_blobs, after_segments = [], []
    after, _ = codec.encode(source, after_blobs, after_segments)
    assert (after, after_blobs) == (before, before_blobs)
    assert source.fingerprint == fingerprint
    assert not registry.spec("dinkster.layers").asset_references_complete


@pytest.mark.parametrize("case", ["legacy", "unused", "host-missing"])
def test_isolated_native_save_adopts_migrated_resources_and_retains_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, layer_document_pack: Path
) -> None:
    import time

    from dinkster_assets import AssetVault, LibraryStore, digest_bytes
    from dinkster_assets.writer import AssetWriter, MountSnapshotWriter
    from dinkster_engine import RunResult
    from dinkster_image.compositor_types import (
        CompositorSourceLayer,
        LayerStack,
        decode_layer_stack,
        encode_layer_stack,
        layer_stack_meta,
    )
    from dinkster_image.layer_io import SaveLayers
    from dinkster_image.types import register_image_types
    from dinkster_protocol import Invocation
    from dinkster_server import ServerLibrary
    from dinkster_server.queue import Job, JobKey
    from dinkster_values import SAVE_TARGET_TYPE
    from dinkster_values.asset_folders import AssetFolderContext

    producer, host = AssetVault(tmp_path / "producer"), AssetVault(tmp_path / "host")
    output = tmp_path / "output"
    output.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps({"mounts": [{"id": "out", "root": str(output), "mode": "readwrite"}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(host.root))
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    registry = TypeRegistry()
    register_core_types(registry)
    register_media_types(registry)
    register_image_types(registry)
    pixels = np.random.default_rng(545).random((1, 64, 64, 3), dtype=np.float32)
    expected = raster_png(pixels[0])
    digest = digest_bytes(expected)
    old = TypeRegistry()
    old.register(
        "dinkster.layers",
        encode=encode_layer_stack,
        decode=decode_layer_stack,
        meta=layer_stack_meta,
    )
    original = old.wrap(
        "dinkster.layers",
        LayerStack((CompositorSourceLayer(pixels, None, "Noise", 0, 0, 64, 64),)),
    )
    blobs, segments = [], []
    wire, _ = ValueCodec(old, use_shm=False).encode(original, blobs, segments)
    codec = ValueCodec(registry, use_shm=False)
    source, _ = codec.decode(wire, blobs, [])
    if case != "legacy":
        for vault in (producer,) if case == "host-missing" else (producer, host):
            with vault.writer(digest) as writer:
                writer.write(expected)
                writer.commit()
        record = empty_document(64, 64).to_record()
        record["allocation"]["nextOrdinal"] = 1
        record["resources"]["r0"] = {
            "id": "r0",
            "kind": "raster",
            "digest": digest,
            "byteSize": len(expected),
            "mediaType": "image/png",
            "width": 64,
            "height": 64,
            "colorSpace": "srgb",
            "channelDepth": 8,
            "alphaMode": "opaque",
        }
        source = registry.wrap("dinkster.layers", ImageDocument.from_record(record))
    before_blobs, before_segments = [], []
    before, _ = codec.encode(source, before_blobs, before_segments)

    async def run() -> None:
        group = GroupIsolatedWorker(
            "native-save-test",
            [layer_document_pack],
            registry,
            extra_env={
                "DINKSTER_ASSET_VAULT": str(producer.root),
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
            },
            aimdo_arm="off",
        )
        await group.start()
        try:
            schema = SaveLayers.define_schema()
            started = time.time_ns()
            result = await group.members["dinkster-image"].invoke(
                Invocation(
                    "native-save",
                    "save",
                    schema.node_type,
                    {
                        "layers": source,
                        "target": registry.wrap(
                            SAVE_TARGET_TYPE, {"mount": "out", "prefix": "kept"}
                        ),
                        "format": registry.wrap("COMBO", "native"),
                    },
                    schema,
                )
            )
            if case == "host-missing":
                assert result.error is not None
                assert host.resolve(digest) is None
                return
            assert result.error is None
            assert result.outputs is not None
            artifacts = SavedArtifactAuthority(snapshot).capture_mounted(
                result.saved_artifact_claims, "save", started
            )
            assert len(artifacts) == 1
            assert result.outputs["document"].fingerprint == artifacts[0].digest
            with closing(LibraryStore(tmp_path / "library.sqlite")) as store:
                library = ServerLibrary(
                    host,
                    store,
                    upload_writer=AssetWriter(MountSnapshotWriter(snapshot)),
                    project_folders_for=lambda _: AssetFolderContext(("out",), "out"),
                )
                library.record_saved_outputs(
                    Job(JobKey("audit", "save"), Graph(nodes={}), (), 0, "native-save", "audit"),
                    RunResult(
                        "native-save", {"save": result.outputs}, (), (), (), artifacts=artifacts
                    ),
                )
            with closing(LibraryStore(tmp_path / "library.sqlite")) as store:
                dependencies = store.get_dependencies(artifacts[0].digest)
                assert dependencies is not None
                assert [(item["digest"], item["byteSize"]) for item in dependencies] == [
                    (digest, len(expected))
                ]
                with store.hold_asset_references() as roots:
                    assert {digest, artifacts[0].digest} <= roots
        finally:
            await group.close()
        held = host.resolve(digest)
        assert held is not None
        assert held.read_bytes() == expected
        if case == "unused":
            assert not (host.root / ".produced-assets.sqlite").exists()

    asyncio.run(run())
    after_blobs, after_segments = [], []
    after, _ = codec.encode(source, after_blobs, after_segments)
    assert (after, after_blobs) == (before, before_blobs)
    assert not registry.spec("dinkster.layers").asset_references_complete


@pytest.mark.parametrize("large", [False, True])
def test_isolated_worker_shm_flatten_matches_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, large: bool, layer_document_pack: Path
) -> None:
    from dinkster_image.compositor import AddLayer
    from dinkster_image.layer_document import FlattenLayers
    from dinkster_image.types import register_image_types

    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(tmp_path / "vault"))

    async def run() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_media_types(registry)
        register_image_types(registry)
        group = GroupIsolatedWorker(
            "layer-test",
            [layer_document_pack],
            registry,
            shm_threshold=1,
        )
        await group.start()
        worker = group.members["dinkster-image"]
        try:
            value = large_document() if large else document()
            engine = Engine(
                schemas=dict(worker.schemas),
                registry=registry,
                worker=worker,
                cache=MemoryLRUCache(),
            )
            edited = await engine.run(
                Graph(
                    nodes={
                        "edit": GraphNode(
                            "dinkster.layers.edit",
                            {
                                "layers": TypedLiteral("dinkster.layers", value.to_record()),
                                "commands": "[]",
                            },
                        )
                    }
                ),
                ["edit"],
            )
            assert (
                cast(ImageDocument, edited.outputs["edit"]["layers"].resolve()).data == value.data
            )
            if large:
                return
            source = np.random.default_rng(0).random((1, 64, 64, 3), dtype=np.float32)
            built = await engine.run(
                Graph(
                    nodes={
                        "add": GraphNode(
                            "dinkster.layers.add",
                            {"image": TypedLiteral("IMAGE", source.tolist())},
                        )
                    }
                ),
                ["add"],
            )
            published = cast(ImageDocument, built.outputs["add"]["layers"].resolve())
            resource = next(iter(published.to_record()["resources"].values()))
            assert "inline" not in resource
            expected = cast(ImageDocument, AddLayer.execute(image=source)["layers"])
            assert published.data == expected.data
            np.testing.assert_array_equal(flatten(published)[0], flatten(expected)[0])
            graph = Graph(
                nodes={
                    "flatten": GraphNode(
                        "dinkster.layers.flatten",
                        {"layers": TypedLiteral("dinkster.layers", value.to_record())},
                    )
                }
            )
            result = await engine.run(graph, ["flatten"])
            for name, expected in FlattenLayers.execute(layers=value).items():
                actual = result.outputs["flatten"][name].resolve()
                np.testing.assert_array_equal(actual, expected)
        finally:
            await group.close()

    run_scenario(run())
