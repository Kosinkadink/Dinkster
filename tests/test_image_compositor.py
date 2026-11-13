from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from dinkster_api.v1 import (
    TypeRegistry,
)
from dinkster_image.compositor_types import (
    CompositorLayer,
    CompositorRecipe,
    CompositorSourceLayer,
    CompositorTransform,
    source_layer_fingerprint,
)
from dinkster_image.types import register_image_types
from dinkster_image_document.document import ImageDocument

Captured = tuple[str, Mapping[str, object], bytes | None]
_LAYERS_MAGIC = b"DINKSTER-LAYERS\x00\x01"


def _solid(
    color: tuple[float, ...],
    *,
    width: int = 1,
    height: int = 1,
    batch: int = 1,
) -> np.ndarray:
    return np.broadcast_to(
        np.asarray(color, dtype=np.float32),
        (batch, height, width, len(color)),
    ).copy()


def _source(
    color: tuple[float, ...] = (1.0, 0.0, 0.0),
    *,
    name: str = "Layer",
    width: int = 1,
    height: int = 1,
    x: float = 0.0,
    y: float = 0.0,
    opacity: float = 1.0,
    blend_mode: str = "normal",
    mask: np.ndarray | None = None,
) -> CompositorSourceLayer:
    return CompositorSourceLayer(
        image=_solid(color, width=width, height=height),
        mask=mask,
        name=name,
        x=x,
        y=y,
        width=width,
        height=height,
        opacity=opacity,
        blend_mode=blend_mode,
    )


def _edit(
    source: CompositorSourceLayer,
    source_index: int,
    fingerprint: str,
    **overrides: object,
) -> CompositorLayer:
    values: dict[str, object] = {
        "id": f"layer-{source_index}-{fingerprint[:8]}",
        "source_index": source_index,
        "name": source.name,
        "visible": source.visible,
        "opacity": source.opacity,
        "blend_mode": source.blend_mode,
        "transform": CompositorTransform(
            source.x,
            source.y,
            source.width,
            source.height,
            source.rotation,
        ),
        "flip_horizontal": source.flip_horizontal,
        "flip_vertical": source.flip_vertical,
    }
    values.update(overrides)
    return CompositorLayer(**values)  # type: ignore[arg-type]


def _recipe(
    sources: tuple[CompositorSourceLayer, ...],
    *,
    width: int,
    height: int,
    layers: tuple[CompositorLayer, ...] | None = None,
    background_color: str = "#000000",
    background_opacity: float = 0.0,
    background_visible: bool = False,
) -> CompositorRecipe:
    fingerprints = tuple(source_layer_fingerprint(source) for source in sources)
    return CompositorRecipe(
        version=1,
        input_fingerprints=fingerprints,
        canvas_width=width,
        canvas_height=height,
        background_color=background_color,
        background_opacity=background_opacity,
        background_visible=background_visible,
        layers=(
            tuple(_edit(source, index, fingerprints[index]) for index, source in enumerate(sources))
            if layers is None
            else layers
        ),
    )


def _registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_image_types(registry)
    register_image_types(registry)
    return registry


def test_pack_compositor_executes_tensor_layers_through_separate_worker(tmp_path) -> None:
    from pathlib import Path

    from dinkster_assets import AssetVault, LibraryStore, digest_bytes
    from dinkster_caches import MemoryLRUCache
    from dinkster_engine import Engine, ExecutionRuntime
    from dinkster_graph import Graph, GraphNode, Link
    from dinkster_image import register_types
    from dinkster_inference_wire import register_inference_types
    from dinkster_protocol import Invocation
    from dinkster_server.library import ServerLibrary
    from dinkster_values import EncodedPayload, Value, ValueMeta, register_core_types
    from dinkster_workers import IsolatedWorker

    from tests.async_scenarios import run_scenario

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_inference_types(registry)
        register_types(registry)
        worker = IsolatedWorker(
            Path(__file__).resolve().parents[1]
            / "packages/dinkster-nodes-image/dinkster-pack.toml",
            registry,
        )
        library = ServerLibrary(
            vault=AssetVault(tmp_path / "vault"), store=LibraryStore(tmp_path / "library.sqlite")
        )
        await worker.start()
        try:
            for compositor, width in (
                ({"version": 2, "documentDigest": None, "commands": []}, 7),
                ({}, 7),
                (
                    {
                        "version": 2,
                        "documentDigest": None,
                        "commands": [{"op": "canvas", "changes": {"width": 5}}],
                    },
                    5,
                ),
            ):
                graph = Graph(
                    nodes={
                        "source": GraphNode(
                            "dinkster.image.empty_image",
                            {"width": 7, "height": 3, "batch_size": 1, "color": 0xFF0000},
                        ),
                        "layer": GraphNode(
                            "dinkster.image.add_layer",
                            {"image": Link("source", "IMAGE"), "name": "Red"},
                        ),
                        "create": GraphNode(
                            "dinkster.image.image_compositor",
                            {"layers": Link("layer", "LAYERS"), "compositor": compositor},
                        ),
                    }
                )
                engine = Engine(
                    schemas=dict(worker.schemas),
                    registry=registry,
                    worker=worker,
                    pin_execution=lambda: ExecutionRuntime(
                        worker=worker, publish_image_documents=library.publish_image_documents
                    ),
                    cache=MemoryLRUCache(),
                )
                result = await engine.run(graph, ["create"])
                assert len(result.artifacts) == 1
                artifact = result.artifacts[0]
                retained = library.vault.resolve(artifact.digest)
                assert retained is not None
                original = retained.read_bytes()
                assert digest_bytes(original) == artifact.digest
                assert ImageDocument(original).to_record()["canvas"]["width"] == 7
                image = result.outputs["create"]["IMAGE"].resolve()
                np.testing.assert_array_equal(image, _solid((1.0, 0.0, 0.0), width=width, height=3))
                np.testing.assert_array_equal(
                    result.outputs["create"]["MASK"].resolve(), np.zeros((1, 3, width))
                )

            schema = worker.schemas["dinkster.image.image_compositor"]
            for index, (data, reason) in enumerate(
                (
                    (
                        b'{"version":2,"documentDigest":null,"commands":{}}',
                        "compositor commands must be an array",
                    ),
                    (
                        b'{"version":2,"documentDigest":"bad","commands":[]}',
                        "invalid compositor document digest",
                    ),
                )
            ):
                refused = await worker.invoke(
                    Invocation(
                        f"invalid-{index}",
                        "create",
                        schema.node_type,
                        {
                            "layers": registry.wrap(
                                "LAYERS", {"version": 1, "canvas": [7, 3], "layers": []}
                            ),
                            "compositor": Value(
                                "COMPOSITOR",
                                f"invalid-{index}",
                                ValueMeta(),
                                EncodedPayload("COMPOSITOR", data, None),
                            ),
                        },
                        schema,
                    )
                )
                assert refused.error is not None
                assert (
                    "encoded payload validation failed for type 'COMPOSITOR'"
                    in refused.error.message
                )
                assert reason in refused.error.message
        finally:
            await worker.close()
            library.store.close()

    run_scenario(scenario())
