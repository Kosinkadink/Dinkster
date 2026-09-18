"""Deterministic ImageDocument rendering, cache provenance, and native node coverage."""

from __future__ import annotations

import asyncio
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import AssetRef, AssetVault, LibraryStore, digest_bytes
from dinkster_image_document import (
    RENDERER_CONTRACT,
    decode_document,
    parse_selector,
    render_document,
)
from dinkster_nodes_media_io.image_document import RenderImageDocument
from dinkster_server.auth import LOCAL_PRINCIPAL, PRINCIPAL_KEY
from dinkster_server.library import ServerLibrary, add_library_routes
from dinkster_server.preflight import asset_preflight
from PIL import Image

MEDIA_TYPE = "application/vnd.dinkster.image-document+json"
IDENTITY = {"a": 1_000_000, "b": 0, "c": 0, "d": 1_000_000, "tx": 0, "ty": 0}


def _png(pixels: list[list[tuple[int, int, int, int]]]) -> bytes:
    output = BytesIO()
    Image.fromarray(np.asarray(pixels, dtype=np.uint8), "RGBA").save(output, "PNG")
    return output.getvalue()


def _resource(resource_id: str, data: bytes, width: int, height: int) -> dict[str, object]:
    return {
        "id": resource_id,
        "kind": "raster",
        "digest": digest_bytes(data),
        "byteSize": len(data),
        "mediaType": "image/png",
        "width": width,
        "height": height,
        "colorSpace": "srgb",
        "channelDepth": 8,
        "alphaMode": "straight",
    }


def _layer(
    layer_id: str,
    resource_id: str,
    width: int,
    height: int,
    *,
    blend: str = "normal",
    opacity: int = 65_535,
    transform: dict[str, int] | None = None,
    mask_ids: list[str] | None = None,
    clipping: str = "none",
) -> dict[str, object]:
    return {
        "id": layer_id,
        "kind": "raster",
        "name": layer_id,
        "visible": True,
        "opacity": opacity,
        "transform": transform or IDENTITY,
        "blendMode": blend,
        "clipping": clipping,
        "maskIds": mask_ids or [],
        "resourceId": resource_id,
        "sourceRect": {"x": 0, "y": 0, "width": width, "height": height},
    }


def _document(
    width: int,
    height: int,
    layers: dict[str, dict[str, object]],
    resources: dict[str, dict[str, object]],
    roots: list[str],
    *,
    masks: dict[str, dict[str, object]] | None = None,
    next_ordinal: int = 4,
) -> dict[str, object]:
    return {
        "format": "dinkster-image",
        "formatVersion": 1,
        "lineage": "render-test",
        "canvas": {
            "width": width,
            "height": height,
            "colorSpace": "srgb",
            "channelDepth": 8,
            "compositing": "premultiplied-alpha",
        },
        "allocation": {"nextOrdinal": next_ordinal},
        "rootLayerIds": roots,
        "layers": layers,
        "masks": masks or {},
        "resources": resources,
    }


def _canonical(document: dict[str, object]) -> bytes:
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _decoded_png(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        return np.asarray(image.convert("RGBA"))


def _render(document: dict[str, object], resources: dict[str, bytes], selector: str = "composite"):
    parsed = decode_document(_canonical(document))
    by_digest = {digest_bytes(data): data for data in resources.values()}
    return render_document(parsed, by_digest.__getitem__, parse_selector(selector))


def test_reference_render_is_byte_stable_and_identity_preserving() -> None:
    source = _png([[(255, 0, 0, 255), (0, 255, 0, 128)]])
    document = _document(
        2,
        1,
        {"l1": _layer("l1", "r0", 2, 1)},
        {"r0": _resource("r0", source, 2, 1)},
        ["l1"],
        next_ordinal=2,
    )
    first = _render(document, {"r0": source})
    second = _render(document, {"r0": source})
    assert first.png == second.png
    assert digest_bytes(first.png) == (
        "blake3:2a5bb8dce85d085838350fa7541ec9d1e0f582b06e92b8009a78a2e2eaa67f22"
    )
    assert _decoded_png(first.png).tolist() == [[[255, 0, 0, 255], [0, 255, 0, 128]]]


@pytest.mark.parametrize(
    ("blend", "expected"),
    [
        ("normal", (200, 100, 50, 255)),
        ("multiply", (78, 59, 39, 255)),
        ("screen", (222, 191, 211, 255)),
        ("overlay", (157, 127, 167, 255)),
        ("darken", (100, 100, 50, 255)),
        ("lighten", (200, 150, 200, 255)),
    ],
)
def test_reference_blend_modes(blend: str, expected: tuple[int, int, int, int]) -> None:
    backdrop = _png([[(100, 150, 200, 255)]])
    source = _png([[(200, 100, 50, 255)]])
    document = _document(
        1,
        1,
        {
            "l2": _layer("l2", "r0", 1, 1),
            "l3": _layer("l3", "r1", 1, 1, blend=blend),
        },
        {
            "r0": _resource("r0", backdrop, 1, 1),
            "r1": _resource("r1", source, 1, 1),
        },
        ["l2", "l3"],
    )
    rendered = _render(document, {"r0": backdrop, "r1": source})
    assert tuple(_decoded_png(rendered.png)[0, 0]) == expected


def test_masks_transforms_and_selectors_have_closed_semantics() -> None:
    source = _png([[(255, 0, 0, 255), (255, 0, 0, 255)]])
    mask_data = _png([[(0, 0, 0, 0), (255, 255, 255, 128)]])
    mask = {
        "id": "m3",
        "kind": "raster",
        "ownerLayerId": "l2",
        "enabled": True,
        "invert": False,
        "opacity": 65_535,
        "transform": IDENTITY,
        "combineMode": "multiply",
        "channel": "alpha",
        "resourceId": "r1",
        "sourceRect": {"x": 0, "y": 0, "width": 2, "height": 1},
    }
    document = _document(
        2,
        1,
        {"l2": _layer("l2", "r0", 2, 1, mask_ids=["m3"])},
        {
            "r0": _resource("r0", source, 2, 1),
            "r1": _resource("r1", mask_data, 2, 1),
        },
        ["l2"],
        masks={"m3": mask},
    )
    resources = {"r0": source, "r1": mask_data}
    composite = _render(document, resources)
    assert _decoded_png(composite.png).tolist() == [[[0, 0, 0, 0], [255, 0, 0, 128]]]
    selected = _render(document, resources, "mask:m3")
    assert _decoded_png(selected.png).tolist() == [[[0, 0, 0, 255], [128, 128, 128, 255]]]
    with pytest.raises(ValueError, match="selected layer"):
        _render(document, resources, "layer:missing")


def test_affine_translation_and_clipping_use_deterministic_pixel_centers() -> None:
    base = _png([[(255, 0, 0, 255), (0, 0, 0, 0)]])
    top = _png([[(0, 255, 0, 255), (0, 255, 0, 255)]])
    document = _document(
        2,
        1,
        {
            "l2": _layer(
                "l2",
                "r0",
                2,
                1,
                transform={**IDENTITY, "tx": 1_000_000},
            ),
            "l3": _layer("l3", "r1", 2, 1, clipping="clip-to-previous"),
        },
        {
            "r0": _resource("r0", base, 2, 1),
            "r1": _resource("r1", top, 2, 1),
        },
        ["l2", "l3"],
    )
    rendered = _render(document, {"r0": base, "r1": top})
    assert _decoded_png(rendered.png).tolist() == [[[0, 0, 0, 0], [0, 255, 0, 255]]]


def test_group_layers_render_children_before_applying_the_group_transform() -> None:
    source = _png([[(255, 0, 0, 255)]])
    group = {
        "id": "l2",
        "kind": "group",
        "name": "group",
        "visible": True,
        "opacity": 65_535,
        "transform": {**IDENTITY, "tx": 1_000_000},
        "blendMode": "normal",
        "clipping": "none",
        "maskIds": [],
        "childLayerIds": ["l1"],
    }
    document = _document(
        2,
        1,
        {"l1": _layer("l1", "r0", 1, 1), "l2": group},
        {"r0": _resource("r0", source, 1, 1)},
        ["l2"],
        next_ordinal=3,
    )
    rendered = _render(document, {"r0": source})
    assert _decoded_png(rendered.png).tolist() == [[[0, 0, 0, 0], [255, 0, 0, 255]]]


def test_sampling_coordinates_stay_64_bit_above_windows_int32_boundary() -> None:
    pixels = [[(index % 256, 0, 0, 255) for index in range(1_074)]]
    source = _png(pixels)
    document = _document(
        1_074,
        1,
        {"l1": _layer("l1", "r0", 1_074, 1)},
        {"r0": _resource("r0", source, 1_074, 1)},
        ["l1"],
        next_ordinal=2,
    )
    rendered = _render(document, {"r0": source})
    assert np.array_equal(_decoded_png(rendered.png), np.asarray(pixels, dtype=np.uint8))


def test_reference_profile_decodes_premultiplied_resource_files() -> None:
    source = _png([[(128, 0, 0, 128)]])
    resource = _resource("r0", source, 1, 1)
    resource["alphaMode"] = "premultiplied"
    document = _document(
        1,
        1,
        {"l1": _layer("l1", "r0", 1, 1)},
        {"r0": resource},
        ["l1"],
        next_ordinal=2,
    )
    rendered = _render(document, {"r0": source})
    assert _decoded_png(rendered.png).tolist() == [[[255, 0, 0, 128]]]


def test_reference_profile_rejects_canvas_work_and_resource_admission_excesses() -> None:
    source = _png([[(255, 0, 0, 255)]])
    resource = _resource("r0", source, 1, 1)
    oversized_canvas = _document(
        4_096,
        1_025,
        {"l1": _layer("l1", "r0", 1, 1)},
        {"r0": resource},
        ["l1"],
        next_ordinal=2,
    )
    with pytest.raises(ValueError, match="canvas exceeds"):
        _render(oversized_canvas, {"r0": source})

    layers = {f"l{ordinal}": _layer(f"l{ordinal}", "r0", 1, 1) for ordinal in range(1, 18)}
    oversized_work = _document(
        2_048,
        2_048,
        layers,
        {"r0": resource},
        list(layers),
        next_ordinal=18,
    )
    with pytest.raises(ValueError, match="work limit"):
        _render(oversized_work, {"r0": source})

    oversized_resource = _document(
        1,
        1,
        {"l1": _layer("l1", "r0", 1, 1)},
        {"r0": {**resource, "byteSize": 512 * 1024 * 1024}},
        ["l1"],
        next_ordinal=2,
    )
    with pytest.raises(ValueError, match="memory limit"):
        _render(oversized_resource, {"r0": source})


async def _client(tmp_path: Path) -> tuple[TestClient, ServerLibrary]:
    @web.middleware
    async def local_principal(request: web.Request, handler):  # type: ignore[no-untyped-def]
        request[PRINCIPAL_KEY] = LOCAL_PRINCIPAL
        return await handler(request)

    library = ServerLibrary(
        vault=AssetVault(tmp_path / "vault"),
        store=LibraryStore(tmp_path / "library.sqlite"),
    )
    app = web.Application(middlewares=[local_principal])
    add_library_routes(app, library)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, library


def test_render_endpoint_persists_output_and_provenance_cache(tmp_path: Path) -> None:
    async def scenario() -> None:
        client, library = await _client(tmp_path)
        source = _png([[(12, 34, 56, 255)]])
        try:
            upload = await client.post(
                "/api/assets/media?scope=local&kind=media/image&name=source.png",
                data=source,
                headers={"Content-Type": "image/png"},
            )
            source_digest = (await upload.json())["asset"]["digest"]
            document = _document(
                1,
                1,
                {"l1": _layer("l1", "r0", 1, 1)},
                {"r0": {**_resource("r0", source, 1, 1), "digest": source_digest}},
                ["l1"],
                next_ordinal=2,
            )
            body = _canonical(document)
            document_digest = digest_bytes(body)
            adopted = await client.post(
                "/api/assets/image-document?scope=local",
                data=body,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert adopted.status == 201, await adopted.text()
            rendered = await client.post(
                f"/api/assets/{document_digest}/render?scope=local",
                json={"selector": "composite"},
            )
            assert rendered.status == 201, await rendered.text()
            first = await rendered.json()
            assert first["cached"] is False
            assert first["provenance"]["documentDigest"] == document_digest
            assert first["provenance"]["selector"] == "composite"
            assert first["provenance"]["rendererContract"] == RENDERER_CONTRACT
            output = await client.get(f"/api/assets/{first['asset']['digest']}")
            assert output.status == 200
            assert digest_bytes(await output.read()) == first["asset"]["digest"]
            repeated = await client.post(
                f"/api/assets/{document_digest}/render?scope=local",
                json={"selector": "composite"},
            )
            assert repeated.status == 200
            second = await repeated.json()
            assert second["cached"] is True
            assert second["asset"] == first["asset"]
            assert library.store.get_derivation(first["cacheKey"]) == first["provenance"]
            wrong_scope = await client.post(
                f"/api/assets/{document_digest}/render?scope=other",
                json={"selector": "composite"},
            )
            assert wrong_scope.status == 409
            assert (await wrong_scope.json())["error"]["code"] == (
                "asset.image_document.dependency_missing"
            )
        finally:
            await client.close()

    asyncio.run(scenario())


def test_native_node_resolves_document_children_with_the_bound_resolver(tmp_path: Path) -> None:
    source = _png([[(90, 80, 70, 255)]])
    document = _document(
        1,
        1,
        {"l1": _layer("l1", "r0", 1, 1)},
        {"r0": _resource("r0", source, 1, 1)},
        ["l1"],
        next_ordinal=2,
    )
    body = _canonical(document)
    vault = AssetVault(tmp_path / "vault")
    for data in (source, body):
        with vault.writer(digest_bytes(data)) as writer:
            writer.write(data)
            writer.commit()
    reference = AssetRef(
        digest=digest_bytes(body),
        name="document.dinkster-image.json",
        size=len(body),
        media_type=MEDIA_TYPE,
        resolver=vault,
    )
    output = RenderImageDocument.execute(document=reference)["image"]
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 1, 1, 3)
    assert np.allclose(output[0, 0, 0], [90 / 255, 80 / 255, 70 / 255])


def test_preflight_expands_trusted_document_dependencies(tmp_path: Path) -> None:
    document_bytes = b"trusted document"
    document_digest = digest_bytes(document_bytes)
    child_digest = digest_bytes(b"missing raster")
    library = ServerLibrary(
        vault=AssetVault(tmp_path / "vault"),
        store=LibraryStore(tmp_path / "library.sqlite"),
    )
    try:
        with library.vault.writer(document_digest) as writer:
            writer.write(document_bytes)
            writer.commit()
        library.store.put_dependencies(
            document_digest,
            [{"resourceId": "r0", "digest": child_digest}],
        )
        plan = asset_preflight(library, {document_digest: "Document"}, set(), {})
        assert [entry["digest"] for entry in plan] == [child_digest]
        assert plan[0]["name"] == "r0"
    finally:
        library.store.close()


def test_v2_render_api_and_flatten_are_pixel_identical(tmp_path: Path) -> None:
    from dinkster_image_document.document import append_raster, empty_document, flatten, raster_png

    async def scenario() -> None:
        client, _ = await _client(tmp_path)
        document = append_raster(
            empty_document(2, 1),
            raster_png(np.array([[[1, 0, 0, 1], [0, 1, 0, 0.5]]], dtype=np.float32)),
        )
        try:
            adopted = await client.post(
                "/api/assets/image-document?scope=local",
                data=document.data,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert adopted.status == 201, await adopted.text()
            rendered = await client.post(
                f"/api/assets/{digest_bytes(document.data)}/render?scope=local",
                json={"selector": "composite"},
            )
            assert rendered.status == 201, await rendered.text()
            result = await rendered.json()
            response = await client.get(f"/api/assets/{result['asset']['digest']}")
            assert response.status == 200
            data = await response.read()
            assert data == document.render().png
            with Image.open(BytesIO(data)) as image:
                pixels = np.asarray(image, dtype=np.float32)[None] / 255
            actual, transparency = flatten(document)
            np.testing.assert_array_equal(actual, pixels)
            np.testing.assert_array_equal(transparency, 1 - pixels[..., 3])
        finally:
            await client.close()

    asyncio.run(scenario())


def test_render_api_and_flatten_node_rendition_are_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_caches import MemoryLRUCache
    from dinkster_engine import Engine, EventListener
    from dinkster_graph import Graph, GraphNode, Link, graph_to_wire
    from dinkster_nodes_image import AddLayer, FlattenLayers, register_image_types
    from dinkster_nodes_media_io import LoadImage, register_media_types
    from dinkster_schema import build_node_types, build_schemas
    from dinkster_server import create_app
    from dinkster_values import TypeRegistry, register_core_types
    from dinkster_workers import InProcessWorker

    nodes = [LoadImage, AddLayer, FlattenLayers]
    schemas = build_schemas(nodes)
    library = ServerLibrary(
        vault=AssetVault(tmp_path / "vault"),
        store=LibraryStore(tmp_path / "library.sqlite"),
    )
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(library.vault.root))

    def factory(on_event: EventListener) -> Engine:
        registry = TypeRegistry()
        register_core_types(registry)
        register_media_types(registry)
        register_image_types(registry)
        return Engine(
            schemas=schemas,
            registry=registry,
            worker=InProcessWorker(build_node_types(nodes), registry),
            cache=MemoryLRUCache(),
            on_event=on_event,
        )

    async def scenario() -> None:
        client = TestClient(TestServer(create_app(factory, schemas, library=library)))
        await client.start_server()
        source = _png(
            [
                [(255, 0, 0, 255), (0, 255, 0, 128), (0, 0, 255, 0), (4, 8, 12, 64)],
                [(15, 31, 63, 255), (127, 95, 63, 192), (9, 7, 5, 1), (1, 2, 3, 254)],
                [(20, 40, 60, 80), (80, 60, 40, 20), (3, 5, 7, 9), (250, 240, 230, 220)],
            ]
        )
        try:
            uploaded = await client.post(
                "/api/assets/media?scope=local&kind=media/image&name=source.png",
                data=source,
                headers={"Content-Type": "image/png"},
            )
            assert uploaded.status == 201, await uploaded.text()
            asset = (await uploaded.json())["asset"]
            graph = Graph(
                nodes={
                    "load": GraphNode("dinkster.load_image", {"image": asset}),
                    "layers": GraphNode(
                        "dinkster.layers.add",
                        {
                            "image": Link("load", "image"),
                            "mask": Link("load", "mask"),
                        },
                    ),
                    "flatten": GraphNode(
                        "dinkster.layers.flatten", {"layers": Link("layers", "layers")}
                    ),
                }
            )
            submitted = await client.post(
                "/api/jobs",
                json={
                    "clientId": "c1",
                    "jobId": "j1",
                    "graph": graph_to_wire(graph),
                    "targets": ["flatten"],
                },
            )
            assert submitted.status == 202, await submitted.text()
            async with asyncio.timeout(5):
                while True:
                    status = await (await client.get("/api/jobs/c1/j1")).json()
                    if status["state"] in ("completed", "failed", "cancelled"):
                        assert status["state"] == "completed", status
                        break
                    await asyncio.sleep(0.01)

            values = "/api/values?clientId=c1&jobId=j1"
            document_response = await client.get(
                values + "&nodeId=layers&outputId=layers&rendition=document"
            )
            assert document_response.status == 200, await document_response.text()
            document = await document_response.read()
            adopted = await client.post(
                "/api/assets/image-document?scope=local",
                data=document,
                headers={"Content-Type": MEDIA_TYPE},
            )
            assert adopted.status == 201, await adopted.text()
            rendered = await client.post(
                f"/api/assets/{digest_bytes(document)}/render?scope=local",
                json={"selector": "composite"},
            )
            assert rendered.status == 201, await rendered.text()
            render_asset = (await rendered.json())["asset"]
            render_response = await client.get(f"/api/assets/{render_asset['digest']}")
            assert render_response.status == 200, await render_response.text()

            flatten_response = await client.get(
                values + "&nodeId=flatten&outputId=image&rendition=png"
            )
            assert flatten_response.status == 200, await flatten_response.text()
            assert await flatten_response.read() == await render_response.read()
        finally:
            await client.close()

    asyncio.run(scenario())
