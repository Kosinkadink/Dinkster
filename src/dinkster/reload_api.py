"""Pack reload coordination and dev-mode mutation endpoints (DESIGN 3.9).

POST /api/packs/{packId}/reload restarts one pack's isolated worker from
its spec - a fresh import in a fresh process, the same code path as
startup - and swaps its slice of the live surface in one epoch bump. The
long-standing ComfyUI developer demand (HotReloadHack's job) with the
soundness objection dissolved: no in-process module-cache surgery, ever.
DELETE /api/packs/{packId} is the removal half: retract the pack's slice
and stop its worker, no replacement - same seam, empty delta.

Umbrella-owned wiring, like compat_api: the composer lives in dinkster-serve,
dinkster-server knows nothing about workers, and this module is the seam
between them. The HTTP mutation routes are registered only under --dev.
Production uses the same internal coordinator for install activation and
pack-owned schema source changes without exposing those routes.

Failure semantics: a reload that fails (manifest error, worker death,
schema refusal, namespace violation) leaves the OLD worker serving and
returns 409 with the error - strictly better than a dead pack. The engine
result cache is cleared on success: cache keys fingerprint schema
signature + inputs, never implementation (hazard H4), so a code change
behind an unchanged signature would stale-hit forever.
"""

from __future__ import annotations

import json
import traceback

from aiohttp import web
from dinkster_schema import core_logger
from dinkster_server import STATE_KEY, PathRedactor, ServerState

from .compose import PackSpec, ServingComposer, UnknownPackError

__all__ = ["add_reload_routes", "apply_reload", "apply_remove"]

_log = core_logger("reload")


async def apply_reload(
    state: ServerState,
    composer: ServingComposer,
    name: str,
    spec: PackSpec | None = None,
) -> dict[str, object]:
    """Reload one pack and publish the swap to the live surface.

    The single reload coordinator: the HTTP endpoint, file and schema
    watchers, and live activation all call this, so there is exactly one
    sequence of swap-then-announce (one epoch bump, one schema_changed
    after /api/nodes serves the new surface - the settled ordering
    contract). ``spec`` replaces the pack's recorded spec (activation's
    new-generation path); omitted reloads in place. Raises whatever
    :meth:`ServingComposer.reload_pack` raises; on any failure the old
    worker keeps serving and nothing here ran.
    """
    async with composer.publication_transaction():
        result = await composer.reload_pack(name, spec)

        async def publish() -> dict[str, object]:
            validation = await state.prepare_replace(
                result.removed_types,
                result.removed_packs,
                result.delta.schemas,
                result.delta.packs,
                result.delta.node_packs,
            )
            epoch = state.replace(
                result.removed_types,
                result.removed_packs,
                result.delta.schemas,
                result.delta.packs,
                result.delta.node_packs,
                execution_arms=result.delta.execution_arms,
                remove_choices=(*result.removed_choices, *result.delta.derived_choices),
                choices={**result.delta.choices, **result.delta.derived_choices},
                lazy_choices=result.delta.lazy_choices,
                schema_owners=result.delta.schema_owners,
                choice_owners=result.delta.choice_owners,
                remove_compat_skips=result.removed_compat_skips,
                compat_skips=result.delta.compat_skips,
                _validation=validation,
            )
            # The composition report tracks the LIVE state of each pack; a
            # reloaded pack is announced at the new epoch.
            for pack in result.reloaded_packs or (result.pack,):
                state.mark_pack_announced(pack, epoch)
            dropped = sorted(t for t in result.removed_types if t not in result.delta.schemas)
            # Worker reload invalidates cached results (the HotReloadHack
            # precedent): keys fingerprint signature + inputs, not code, so an
            # implementation change behind an unchanged signature would
            # stale-hit. Keys are opaque hashes - per-pack scoping is
            # impossible - so the whole cache goes on any worker replacement.
            clear = getattr(state.engine.cache, "clear", None)
            cleared = clear() if callable(clear) else 0
            _log.info(
                "pack %s reloaded: epoch %d, %d node type(s), %d dropped, %d cache entries cleared",
                result.pack,
                epoch,
                len(result.delta.schemas),
                len(dropped),
                cleared,
            )
            return {
                "pack": result.pack,
                "epoch": epoch,
                "nodes": sorted(result.delta.schemas),
                "removedNodes": dropped,
                "cacheCleared": cleared,
            }

        return await composer.finish_publication(publish())


async def apply_remove(
    state: ServerState, composer: ServingComposer, name: str
) -> dict[str, object]:
    """Remove one pack and publish the retraction to the live surface.

    The removal half of :func:`apply_reload`, same coordinator shape:
    composer first (routes drop, worker closes), then ONE
    ``state.replace`` with an empty delta - removed types vanish from
    /api/nodes, exclusive pack-table ids leave, one epoch bump, one
    schema_changed after the surface serves (the settled ordering
    contract). The composition report keeps the row as "removed".

    The result cache clears for the same H4 reason as reload: keys
    fingerprint signature + inputs, never implementation, so the
    remove-edit-recompose dev flow would stale-hit a re-added type whose
    signature never changed.
    """
    async with composer.publication_transaction():
        result = await composer.remove_pack(name)

        async def publish() -> dict[str, object]:
            validation = await state.prepare_replace(
                result.removed_types,
                result.removed_packs,
                {},
                {},
                {},
            )
            epoch = state.replace(
                result.removed_types,
                result.removed_packs,
                {},
                {},
                {},
                execution_arms=result.execution_arms,
                remove_choices=(*result.removed_choices, *result.derived_choices),
                choices=result.derived_choices,
                choice_owners=result.choice_owners,
                remove_compat_skips=result.removed_compat_skips,
                _validation=validation,
            )
            state.mark_pack_removed(result.pack, epoch)
            clear = getattr(state.engine.cache, "clear", None)
            cleared = clear() if callable(clear) else 0
            _log.info(
                "pack %s removed: epoch %d, %d node type(s) retracted, %d cache entries cleared",
                result.pack,
                epoch,
                len(result.removed_types),
                cleared,
            )
            return {
                "pack": result.pack,
                "epoch": epoch,
                "removedNodes": sorted(result.removed_types),
                "removedPacks": sorted(result.removed_packs),
                "cacheCleared": cleared,
            }

        return await composer.finish_publication(publish())


def add_reload_routes(
    app: web.Application,
    composer: ServingComposer,
    *,
    debug_errors: bool = False,
    redactor: PathRedactor | None = None,
) -> None:
    active_redactor = redactor if redactor is not None else PathRedactor()

    def _clean(text: str) -> str:
        return text if debug_errors else active_redactor.redact_text(text)

    async def handle_reload(request: web.Request) -> web.Response:
        name = request.match_info["pack_id"]
        state = request.app[STATE_KEY]
        try:
            payload = await apply_reload(state, composer, name)
        except UnknownPackError as exc:
            raise web.HTTPNotFound(
                text=json.dumps({"error": "unknown-pack", "detail": _clean(str(exc))}),
                content_type="application/json",
            ) from exc
        except Exception as exc:
            # The old worker keeps serving; nothing changed server-side.
            _log.exception("pack %s reload failed", name)
            body = {
                "error": "reload-failed",
                "type": type(exc).__name__,
                "detail": _clean(str(exc)),
                "traceback": _clean(traceback.format_exc()),
            }
            raise web.HTTPConflict(
                text=json.dumps(body),
                content_type="application/json",
            ) from exc
        return web.json_response(payload)

    async def handle_remove(request: web.Request) -> web.Response:
        name = request.match_info["pack_id"]
        state = request.app[STATE_KEY]
        try:
            payload = await apply_remove(state, composer, name)
        except UnknownPackError as exc:
            raise web.HTTPNotFound(
                text=json.dumps({"error": "unknown-pack", "detail": _clean(str(exc))}),
                content_type="application/json",
            ) from exc
        except Exception as exc:
            # No validation phase exists on removal, so anything here is
            # host wiring trouble; the surface may be partially retracted
            # only if state.replace itself refused - loud either way.
            _log.exception("pack %s removal failed", name)
            body = {
                "error": "remove-failed",
                "type": type(exc).__name__,
                "detail": _clean(str(exc)),
                "traceback": _clean(traceback.format_exc()),
            }
            raise web.HTTPConflict(
                text=json.dumps(body),
                content_type="application/json",
            ) from exc
        return web.json_response(payload)

    app.router.add_post("/api/packs/{pack_id}/reload", handle_reload)
    app.router.add_delete("/api/packs/{pack_id}", handle_remove)
