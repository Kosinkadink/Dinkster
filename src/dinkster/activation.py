"""Live pack activation: reconcile the served surface with the install
root's current generation (DESIGN 3.9's production follow-up).

``dinkster-pack`` mutates the installation with plan/apply discipline, but a
running ``dinkster-serve --install-root`` keeps serving the generation it
composed at startup. This module closes that gap with whole-generation
staging and one publication:

- ``GET /api/install/activation`` computes the reconciliation PLAN
  without touching anything - which managed packs a live activation
  would add, remove, reload, or leave alone. Plan/apply discipline on
  the wire: the read side shows exactly what the write side would do.
- ``POST /api/install/activation`` starts and validates every worker in a
  private composer. Only a complete valid generation is published. Any
  missing, incompatible, or failed pack closes the staged workers and leaves
  the previous surface, worker topology, snapshot, and epoch intact.

The POST body may pin ``{"generation": N}`` - the number the GET plan
reported. If ``dinkster-pack apply`` lands a NEWER generation between plan
and apply, the pinned request gets 409 ``generation-changed`` instead of
silently reconciling to a surface the caller never previewed. Omitting
the pin means "converge to whatever is current" - the delivery-hook
flow, where the confirmation already happened at ``dinkster-pack apply``
time and re-planning is exactly the point.

Registered whenever an install root serves - NOT dev-gated like the
reload/removal endpoints, because activation takes no input and can only
move the surface toward what ``dinkster-pack``'s own confirmed plan/apply
already put on disk. The mutation was authorized at apply time;
activation is its delivery.

Managed versus dev: a composed pack is MANAGED exactly when its spec's
manifest lives under the install root's content-addressed store.
Explicit ``--pack`` dev additions and the ComfyUI compat workers are
invisible to activation - it never touches them. Store directories are
digest-addressed, so spec equality is content identity: same path means
same bytes, and a version bump means a new store directory, classified
as a reload with the new spec.

Diagnostics are deterministic and ordered by pack name. Activations serialize;
a concurrent request gets 409 rather than interleaving two reconciliations.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web
from dinkster_schema import core_logger
from dinkster_server import STATE_KEY, ServerState

from .compose import CompositionError, PackSpec, ServingComposer, resolve_manifest_path
from .installer import Installer

__all__ = ["ActivationPlan", "add_activation_routes", "compute_plan"]

_log = core_logger("activation")


@dataclass(frozen=True)
class ActivationPlan:
    """What one activation would do, per managed pack name."""

    generation: int | None
    add: tuple[str, ...]
    remove: tuple[str, ...]
    reload: tuple[str, ...]
    unchanged: tuple[str, ...]

    def to_wire(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "add": list(self.add),
            "remove": list(self.remove),
            "reload": list(self.reload),
            "unchanged": list(self.unchanged),
        }


def _spec_name(spec: PackSpec) -> str:
    """A packs_for_serving spec carries exactly its manifest's pack-table
    entry; the key is the pack name without re-parsing the manifest."""
    assert spec.packs is not None and len(spec.packs) == 1
    return next(iter(spec.packs))


def compute_plan(
    composer: ServingComposer, installer: Installer
) -> tuple[ActivationPlan, dict[str, PackSpec]]:
    """Diff the live managed packs against the current generation.

    Returns the plan plus the generation's specs by name (what apply
    needs for the add/reload halves). Dev packs - composed specs whose
    manifest is outside the store - are excluded from the diff entirely.
    """
    new_specs = {_spec_name(spec): spec for spec in installer.packs_for_serving()}
    store_root = installer.store_root.resolve()
    managed = {
        name: spec
        for name, spec in composer.pack_specs().items()
        if Path(spec.manifest).resolve().is_relative_to(store_root)
    }
    add = tuple(sorted(name for name in new_specs if name not in managed))
    remove = tuple(sorted(name for name in managed if name not in new_specs))
    reload_ = tuple(
        sorted(
            name for name, spec in managed.items() if name in new_specs and new_specs[name] != spec
        )
    )
    unchanged = tuple(
        sorted(
            name for name, spec in managed.items() if name in new_specs and new_specs[name] == spec
        )
    )
    return (
        ActivationPlan(
            generation=installer.current_number(),
            add=add,
            remove=remove,
            reload=reload_,
            unchanged=unchanged,
        ),
        new_specs,
    )


async def apply_activation(
    state: ServerState, composer: ServingComposer, installer: Installer
) -> dict[str, object]:
    """Stage, validate, and atomically publish one complete generation."""
    plan, new_specs = compute_plan(composer, installer)
    if not (plan.add or plan.remove or plan.reload):
        return {
            "generation": plan.generation,
            "epoch": state.schema_epoch,
            "added": [],
            "removed": [],
            "reloaded": [],
            "unchanged": list(plan.unchanged),
            "failed": {},
            "diagnostics": [],
            "rolledBack": False,
        }

    current_specs = composer.pack_specs()
    store_root = installer.store_root.resolve()
    desired = {
        name: spec
        for name, spec in current_specs.items()
        if not Path(spec.manifest).resolve().is_relative_to(store_root)
    }
    desired.update(new_specs)

    async with composer.generation_transaction():
        staged = composer.spawn_empty(composition_mode="production")
        diagnostics_by_name: dict[str, dict[str, str]] = {}
        adopted = False

        def record_failure(name: str, exc: Exception) -> None:
            message = str(exc)
            if isinstance(exc, (FileNotFoundError, ModuleNotFoundError)) or (
                message.startswith("pack manifest not found:")
                or "ModuleNotFoundError" in message
                or "No module named" in message
            ):
                kind = "missing-pack"
            elif isinstance(exc, (CompositionError, ValueError)):
                kind = "incompatible-pack"
            else:
                kind = "failed-pack"
            diagnostics_by_name[name] = {"pack": name, "kind": kind, "error": message}
            _log.error("activation: staging pack %s failed: %s", name, exc)

        try:
            valid: list[PackSpec] = []
            names_by_manifest: dict[Path, str] = {}
            for name in sorted(desired):
                spec = desired[name]
                try:
                    path = resolve_manifest_path(spec.manifest)
                except Exception as exc:
                    record_failure(name, exc)
                    continue
                valid.append(spec)
                names_by_manifest[path.resolve()] = name
            try:
                ordered = staged.order_pack_entries(valid)
            except Exception as exc:
                record_failure("generation", exc)
                ordered = ()
            for spec in ordered:
                path = resolve_manifest_path(spec.manifest)
                name = names_by_manifest[path.resolve()]
                try:
                    delta = await staged.add_pack(spec)
                    if delta.pack != name:
                        raise CompositionError(
                            f"generation expected pack {name!r}, manifest declared {delta.pack!r}"
                        )
                except Exception as exc:
                    record_failure(name, exc)
            try:
                staged.validate_complete_generation()
            except Exception as exc:
                record_failure("generation", exc)
            diagnostics = [diagnostics_by_name[name] for name in sorted(diagnostics_by_name)]
            if diagnostics:
                await staged.close()
                failed = {item["pack"]: item["error"] for item in diagnostics}
                return {
                    "generation": plan.generation,
                    "epoch": state.schema_epoch,
                    "added": [],
                    "removed": [],
                    "reloaded": [],
                    "unchanged": list(plan.unchanged),
                    "failed": failed,
                    "diagnostics": diagnostics,
                    "rolledBack": True,
                    "rollbackEvidence": {
                        "epoch": state.schema_epoch,
                        "extensionSnapshotDigest": state.engine.extension_snapshot_digest,
                        "packs": sorted(current_specs),
                    },
                }

            composition = staged.composition

            async def publish() -> int:
                nonlocal adopted
                validation = await state.prepare_generation(
                    composition.schemas,
                    composition.packs,
                    composition.node_packs,
                )
                epoch = state.publish_generation(
                    composition.schemas,
                    composition.packs,
                    composition.node_packs,
                    execution_arms=composition.execution_arms,
                    choices=composition.choices,
                    lazy_choices=composition.lazy_choices,
                    schema_owners=composition.schema_owners,
                    choice_owners=composition.choice_owners,
                    compat_skips=composition.compat_skips,
                    _validation=validation,
                )
                old = composer.adopt(staged)
                adopted = True
                for name in plan.remove:
                    state.mark_pack_removed(name, epoch)
                for name in (*plan.reload, *plan.add):
                    state.mark_pack_announced(name, epoch)
                clear = getattr(state.engine.cache, "clear", None)
                if callable(clear):
                    clear()
                # Fail-loud retirement policy: the old generation is no longer
                # admitted, and its workers are retired immediately. An in-flight
                # run pinned to one may fail rather than drift generations.
                await old.close()
                # Retirement awaits, so an old invocation that was already
                # returning could repopulate a legacy key after the first clear.
                if callable(clear):
                    clear()
                return epoch

            await composer.finish_publication(publish())
        except BaseException:
            if not adopted:
                await staged.close()
            raise

    _log.info(
        "activation: generation %s live - %d added, %d removed, "
        "%d reloaded, %d unchanged (epoch %d)",
        plan.generation,
        len(plan.add),
        len(plan.remove),
        len(plan.reload),
        len(plan.unchanged),
        state.schema_epoch,
    )
    return {
        "generation": plan.generation,
        "epoch": state.schema_epoch,
        "added": list(plan.add),
        "removed": list(plan.remove),
        "reloaded": list(plan.reload),
        "unchanged": list(plan.unchanged),
        "failed": {},
        "diagnostics": [],
        "rolledBack": False,
    }


def _conflict(error: str, detail: str) -> web.HTTPConflict:
    return web.HTTPConflict(
        text=json.dumps({"error": error, "detail": detail}),
        content_type="application/json",
    )


async def _read_pinned_generation(request: web.Request) -> int | None:
    """The optional {"generation": N} pin from the POST body. An empty
    body is the unpinned delivery-hook flow; anything else must be a JSON
    object whose generation, if present, is an integer."""
    raw = await request.read()
    if not raw:
        return None
    try:
        body = json.loads(raw)
    except ValueError:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "bad-request", "detail": "body must be JSON"}),
            content_type="application/json",
        ) from None
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(
            text=json.dumps({"error": "bad-request", "detail": "body must be a JSON object"}),
            content_type="application/json",
        )
    pinned = body.get("generation")
    if pinned is None:
        return None
    if not isinstance(pinned, int) or isinstance(pinned, bool):
        raise web.HTTPBadRequest(
            text=json.dumps(
                {
                    "error": "bad-request",
                    "detail": "generation must be an integer",
                }
            ),
            content_type="application/json",
        )
    return pinned


def add_activation_routes(
    app: web.Application, composer: ServingComposer, installer: Installer
) -> None:
    busy = asyncio.Lock()

    async def handle_plan(request: web.Request) -> web.Response:
        plan, _ = compute_plan(composer, installer)
        return web.json_response(plan.to_wire())

    async def handle_activate(request: web.Request) -> web.Response:
        pinned = await _read_pinned_generation(request)
        if busy.locked():
            raise _conflict(
                "activation-in-progress",
                "another activation is being applied; retry when it finishes",
            )
        async with busy:
            state = request.app[STATE_KEY]
            if state.composition_progress is not None:
                # Startup composition still driving: the plan would list
                # pending packs as adds and race the drive tasks into
                # duplicate-compose failures. The surface converges to
                # the current generation on its own; activation only
                # makes sense once it has settled.
                raise _conflict(
                    "composition-in-progress",
                    "startup composition has not finished; retry after composition_complete",
                )
            current = installer.current_number()
            if pinned is not None and pinned != current:
                raise _conflict(
                    "generation-changed",
                    f"planned against generation {pinned} but "
                    f"{current} is current; re-fetch the plan",
                )
            payload = await apply_activation(state, composer, installer)
        return web.json_response(payload)

    app.router.add_get("/api/install/activation", handle_plan)
    app.router.add_post("/api/install/activation", handle_activate)
