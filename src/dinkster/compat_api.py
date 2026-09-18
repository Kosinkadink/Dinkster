"""ComfyUI compat HTTP surface: POST /api/compat/comfy/prompt.

Umbrella-owned glue, deliberately outside dinkster-server (which stays
protocol-only and comfy-free) and outside the compat translator (which
stays pure). The endpoint accepts ComfyUI's headless API prompt format -
either the bare prompt object or the ``{"prompt": ..., "client_id": ...}``
wrapper ComfyUI's ``/prompt`` takes - translates it through
dinkster_compat_comfy.prompt at the boundary, and submits the resulting
native graph as an ordinary job. The stored/executed representation is
always the native graph; the response includes it (plus derived targets)
so callers can see exactly what their prompt became.

``?dryRun=1`` translates and returns the native graph without submitting.

Translation runs the compat input adapters: the static table from
dinkster_compat_comfy plus the contextual LoadImage and LoadCheckpoint
adapters, bound here to the app's mount catalogs ('comfy-input' for
legacy image filenames, ordered category-specific model mounts for
legacy model filenames - both become digest-backed asset references;
the native graph never carries a path).

Translation failures return 400 with anchored problems:
``{"error": ..., "problems": [{"code", "message", "nodeId"?, "inputId"?}]}``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from aiohttp import web
from dinkster_assets import (
    ASSET_TYPE,
    KIND_MODEL_DIFFUSION,
    KIND_MODEL_LORA,
    KIND_MODEL_TEXT_ENCODER,
    KIND_MODEL_VAE,
    AssetError,
    AssetRef,
)
from dinkster_compat_comfy import (
    COMFY_INPUT_ADAPTERS,
    InputAdapter,
    PromptTranslationError,
    extract_prompt,
    make_load_checkpoint_adapter,
    make_load_clip_adapter,
    make_load_diffusion_model_adapter,
    make_load_dual_clip_adapter,
    make_load_image_adapter,
    make_load_latent_adapter,
    make_load_lora_adapter,
    make_load_model_patch_adapter,
    make_load_vae_adapter,
    make_load_vision_adapter,
    make_model_asset_inputs_adapter,
    translate_prompt,
)
from dinkster_compat_comfy.prompt import build_alias_index
from dinkster_compat_comfy.schema_snapshot import core_schema_snapshot
from dinkster_compat_comfy.translate import MODEL_FILE_CATEGORIES
from dinkster_graph import graph_to_wire
from dinkster_protocol import ExportSnapshot
from dinkster_schema import AssetWidget, NodeSchema, TypeExpr, WidgetRepresentations
from dinkster_server import (
    LOCAL_PRINCIPAL,
    STATE_KEY,
    job_to_wire,
    principal_for,
    resolve_scope,
)
from dinkster_server.queue import JobGraphAdmissionError

from .mounts_api import MOUNTS_KEY, MountService

DEFAULT_CLIENT_ID = "comfy-compat"

_LEGACY_INPUT_MOUNT = "comfy-input"
"""The derived mount serve.py grants over ComfyUI's input directory -
exactly where classic LoadImage filenames pointed."""

_CHECKPOINT_KIND = "model/checkpoint"


def add_comfy_compat_routes(app: web.Application) -> None:
    app.router.add_post("/api/compat/comfy/prompt", handle_comfy_prompt)
    app.router.add_get("/api/compat/comfy/schemas", handle_comfy_schemas)


async def handle_comfy_schemas(request: web.Request) -> web.Response:
    """Advertise source interfaces separately from the executable node catalog."""
    state = request.app[STATE_KEY]
    snapshot = core_schema_snapshot()
    index = build_alias_index(state.schemas)
    nodes: dict[str, object] = {}
    for node_type, schema in snapshot["schemas"].items():
        candidates = sorted(
            {
                target
                for alias in (node_type, *schema.get("aliases", ()))
                for target in index.get(alias, ())
                if not target.startswith("comfy.")
            }
        )
        nodes[node_type] = {
            "schema": schema,
            "nativeNodeTypes": candidates,
            "importable": len(candidates) == 1,
        }
    return web.json_response(
        {
            "sourceRepository": snapshot["sourceRepository"],
            "sourceCommit": snapshot["sourceCommit"],
            "schemaWireVersion": snapshot["schemaWireVersion"],
            "schemaEpoch": state.schema_epoch,
            "nodes": nodes,
            "skipped": snapshot["skipped"],
        }
    )


def _mount_resolver(service: MountService | None, base: str) -> Callable[[str], AssetRef | None]:
    """Exact-path lookup under ``mounts/<base>/`` in the mount catalog.
    Returns None for anything not cataloged (including when no mounts
    service is registered at all) - the adapter turns that into an
    anchored refusal, never a silent pass-through."""

    def resolve(relative: str) -> AssetRef | None:
        if service is None:
            return None
        try:
            return service.table.ref(f"mounts/{base}/{relative}")
        except AssetError:
            return None

    return resolve


def _kind_resolver(service: MountService | None, kind: str) -> Callable[[str], AssetRef | None]:
    """Resolve a legacy relative name in exactly one live category root."""

    def resolve(relative: str) -> AssetRef | None:
        if service is None:
            return None
        matches: list[AssetRef] = []
        for mount_id in service.table.mounts_for_kind(kind):
            try:
                matches.append(service.table.ref(f"mounts/{mount_id}/{relative}"))
            except AssetError:
                continue
        return matches[0] if len(matches) == 1 else None

    return resolve


def _input_adapters(
    app: web.Application, schemas: Mapping[str, NodeSchema]
) -> dict[str, InputAdapter]:
    service = app.get(MOUNTS_KEY)
    adapters = dict(COMFY_INPUT_ADAPTERS)
    adapters["dinkster.load_image"] = make_load_image_adapter(
        _mount_resolver(service, _LEGACY_INPUT_MOUNT)
    )
    adapters["dinkster.load_latent"] = make_load_latent_adapter(
        _mount_resolver(service, "comfy-output")
    )
    adapters["dinkster.load_checkpoint"] = make_load_checkpoint_adapter(
        _kind_resolver(service, _CHECKPOINT_KIND)
    )
    lora_adapter = make_load_lora_adapter(_kind_resolver(service, KIND_MODEL_LORA))
    adapters["dinkster.load_lora"] = lora_adapter
    adapters["dinkster.load_lora_model_only"] = lora_adapter
    adapters["dinkster.load_z_image_control_patch"] = make_load_model_patch_adapter(
        _kind_resolver(service, "model/patch")
    )
    adapters["dinkster.load_vae"] = make_load_vae_adapter(_kind_resolver(service, KIND_MODEL_VAE))
    adapters["dinkster.load_clip"] = make_load_clip_adapter(
        _kind_resolver(service, KIND_MODEL_TEXT_ENCODER)
    )
    adapters["dinkster.load_dual_clip"] = make_load_dual_clip_adapter(
        _kind_resolver(service, KIND_MODEL_TEXT_ENCODER)
    )
    adapters["dinkster.load_vision"] = make_load_vision_adapter(
        _kind_resolver(service, "model/clip-vision")
    )
    adapters["dinkster.load_diffusion_model"] = make_load_diffusion_model_adapter(
        _kind_resolver(service, KIND_MODEL_DIFFUSION)
    )
    # Translated model loaders retain their original input ids, but their
    # schema now advertises digest-backed assets instead of scalar combos.
    # Preserve legacy Comfy API prompts by converting literal filenames at
    # this boundary. Kind->category is one-to-one in the finite registry;
    # unknown kinds and already-native mappings/links are never guessed.
    categories_by_kind = {
        descriptor.kind: (category, descriptor)
        for category, descriptor in MODEL_FILE_CATEGORIES.items()
    }
    if len(categories_by_kind) != len(MODEL_FILE_CATEGORIES):
        raise RuntimeError("model-file asset kinds must map to exactly one category")
    asset_type = TypeExpr.concrete(ASSET_TYPE)
    for node_type, schema in schemas.items():
        if node_type in adapters:
            continue
        resolvers: dict[str, tuple[Callable[[str], AssetRef | None], str]] = {}
        for spec in schema.inputs:
            widget = spec.widget
            if isinstance(widget, WidgetRepresentations):
                # Model validation requires every asset representation to
                # share one kind, so presentation order/default cannot alter
                # legacy filename resolution at this compatibility boundary.
                widget = widget.representations[0].widget
            if spec.type != asset_type or not isinstance(widget, AssetWidget):
                continue
            category_entry = categories_by_kind.get(widget.kind)
            if category_entry is None:
                continue
            category, descriptor = category_entry
            resolvers[spec.id] = (
                _kind_resolver(service, descriptor.kind),
                category,
            )
        if resolvers:
            adapters[node_type] = make_model_asset_inputs_adapter(resolvers)
    return adapters


async def handle_comfy_prompt(request: web.Request) -> web.Response:
    state = request.app[STATE_KEY]
    principal = principal_for(request)
    scope = resolve_scope(
        principal,
        "jobs:submit",
        None if principal is LOCAL_PRINCIPAL else request.headers.get("X-Dinkster-Scope"),
    )
    try:
        body: Any = await request.json()
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": f"invalid JSON: {exc}"}),
            content_type="application/json",
        ) from exc
    try:
        prompt, client_id, extra_pnginfo = extract_prompt(body)
        skipped_classes = {}
        ambiguous_skips: set[str] = set()
        for (pack_id, skipped_node_id), diagnostic in state.compat_skips.items():
            for source_name in {
                skipped_node_id,
                diagnostic.source_node,
                f"{pack_id}.{skipped_node_id}",
            }:
                if source_name in skipped_classes:
                    ambiguous_skips.add(source_name)
                else:
                    skipped_classes[source_name] = diagnostic
        for source_name in ambiguous_skips:
            skipped_classes.pop(source_name, None)
        translation = translate_prompt(
            prompt,
            state.schemas,
            input_adapters=_input_adapters(request.app, state.schemas),
            skipped_classes=skipped_classes,
        )
    except PromptTranslationError as exc:
        problems = [problem.to_wire() for problem in exc.problems]
        for problem in problems:
            diagnostic = problem.get("compatDiagnostic")
            if isinstance(diagnostic, dict):
                diagnostic["schemaEpoch"] = state.schema_epoch
                diagnostic["extensionSnapshotDigest"] = state.engine.extension_snapshot_digest
        raise web.HTTPBadRequest(
            text=json.dumps(
                {
                    "error": str(exc),
                    "problems": problems,
                }
            ),
            content_type="application/json",
        ) from exc

    graph_wire = graph_to_wire(translation.graph)
    targets = list(translation.targets)
    if request.rel_url.query.get("dryRun") in ("1", "true"):
        return web.json_response({"graph": graph_wire, "targets": targets})

    job_id = uuid.uuid4().hex
    try:
        job = state.queue.submit(
            client_id or DEFAULT_CLIENT_ID,
            job_id,
            translation.graph,
            targets,
            scope=scope,
            principal_id=principal.principal_id,
            principal_kind=principal.kind,
            export_snapshot=ExportSnapshot(prompt=prompt, extra_pnginfo=extra_pnginfo),
            attention_config=state.attention_default,
        )
    except JobGraphAdmissionError as exc:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": str(exc)}), content_type="application/json"
        ) from exc
    except ValueError as exc:
        raise web.HTTPConflict(
            text=json.dumps({"error": str(exc)}), content_type="application/json"
        ) from exc
    wire = job_to_wire(job, None, state.engine.registry)
    wire["graph"] = graph_wire
    wire["targets"] = targets
    return web.json_response(wire, status=202)
