"""Probe a legacy ComfyUI pack for ``dinkster port`` (child process only).

``dinkster port`` (DESIGN 3.8: porting is the endorsed path, tooling makes
it cheap) needs a ComfyUI pack's declared schemas to generate a native
pack skeleton - and reading them means importing arbitrary pack code,
which never happens in the CLI's own process. This module runs in a
disposable subprocess, the same quarantine posture as the doctor probe
and the legacy loader, and reuses the loader end to end: ``load_legacy_pack``
imports with ComfyUI's own mechanics and translates v1 mappings through
the exact translate.py rules the compat worker applies, so the port tool
can never drift from what compat actually runs.

Both ComfyUI node APIs port. v1 ``NODE_CLASS_MAPPINGS`` translate as the
worker does; V3 ``comfy_entrypoint`` packs (which the legacy *runtime*
deliberately refuses - porting is their path) resolve the extension the
way ComfyUI's own loader does (await the entrypoint, ``on_load``, then
``get_node_list``) and each ``define_schema()`` converts through
translate_v3.py. Mixed packs - v1 mappings plus a V3 entrypoint, which
upstream ComfyUI silently half-ignores - probe both halves.

On top of each translated node's real ``NodeSchema`` (serialized with the
real wire encoder - one schema representation everywhere), the probe
records the porting facts translation deliberately erases:

- the execute function's name and source (the author's porting reference),
- combo choice lists (UI affordance, dropped from type identity),
- hidden input names (excluded from schemas by translation),
- the INPUT_IS_LIST / is_input_list calling convention,
- V3 search aliases (search synonyms have no Dinkster schema home).

Usage (parent side runs this; see ``dinkster.port``)::

    python -m dinkster_compat_comfy.port_probe <pack-path> <out-json-path>

The JSON report is written to the named file, never stdout: pack imports
print whatever they like, and the transport must not care.

Environment contract: ``DINKSTER_COMFYUI_ROOT`` (bootstrap.py) - packs get
the same import environment the legacy loader gives them.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import textwrap
import types
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from dinkster_schema import schema_to_wire

from .bootstrap import bootstrap_comfyui, ensure_prompt_server
from .legacy import load_legacy_pack, pack_sys_module_name
from .translate import CompatError, CompatTranslation, iter_v1_inputs
from .translate_v3 import translate_v3_schema

PORT_PROBE_VERSION = 2


def _node_extras(v1_class: type) -> dict[str, object]:
    """Porting facts translate.py drops on purpose, best-effort: a pack
    whose INPUT_TYPES() fails on the second call still ports, just with
    an emptier reference section."""
    extras: dict[str, object] = {
        "function": str(getattr(v1_class, "FUNCTION", "") or ""),
        "inputIsList": bool(getattr(v1_class, "INPUT_IS_LIST", False)),
        "hidden": [],
        "choices": {},
        "source": "",
    }
    try:
        raw = v1_class.INPUT_TYPES()  # type: ignore[attr-defined]
        if isinstance(raw, Mapping):
            raw_mapping = cast("Mapping[str, object]", raw)
            hidden = raw_mapping.get("hidden")
            if isinstance(hidden, Mapping):
                extras["hidden"] = [str(k) for k in cast("Mapping[str, object]", hidden)]
            choices: dict[str, list[object]] = {}
            for name, v1_type, _config, _required in iter_v1_inputs(raw_mapping):
                if isinstance(v1_type, (list, tuple)):
                    choices[name] = [
                        entry if isinstance(entry, (str, int, float, bool)) else str(entry)
                        for entry in cast("Sequence[object]", v1_type)
                    ]
            extras["choices"] = choices
    except Exception:  # noqa: BLE001 - INPUT_TYPES() runs pack code
        pass
    function_name = cast("str", extras["function"])
    if function_name:
        try:
            fn = getattr(v1_class, function_name, None)
            if fn is not None:
                extras["source"] = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError):
            pass
    return extras


def _resolve_v3_nodes(module: types.ModuleType) -> list[type]:
    """Resolve a V3 pack's node classes the way ComfyUI's loader does:
    call (and await, if async) ``comfy_entrypoint``, then ``on_load``,
    then ``get_node_list``. Duck-typed on purpose - real comfy_api
    objects in the probe subprocess, plain fakes in tests."""

    async def collect() -> list[type]:
        entrypoint = module.comfy_entrypoint
        extension = entrypoint()
        if inspect.isawaitable(extension):
            extension = await extension
        on_load = getattr(extension, "on_load", None)
        if callable(on_load):
            loaded = on_load()
            if inspect.isawaitable(loaded):
                await loaded
        get_node_list = getattr(extension, "get_node_list", None)
        if not callable(get_node_list):
            raise CompatError("comfy_entrypoint did not return an extension with get_node_list")
        node_list = get_node_list()
        if inspect.isawaitable(node_list):
            node_list = await node_list
        if not isinstance(node_list, list):
            raise CompatError("get_node_list did not return a list of node classes")
        return cast("list[type]", node_list)

    return asyncio.run(collect())


def _v3_node_extras(node_class: type, schema: object) -> dict[str, object]:
    """The V3 porting facts translate_v3.py drops on purpose."""
    extras: dict[str, object] = {
        "function": "execute",
        "inputIsList": bool(getattr(schema, "is_input_list", False)),
        "hidden": [
            str(getattr(h, "value", h))
            for h in cast("Sequence[object]", getattr(schema, "hidden", ()) or ())
        ],
        "choices": {},
        "source": "",
    }
    choices: dict[str, list[object]] = {}
    for input_obj in cast("Sequence[object]", getattr(schema, "inputs", ()) or ()):
        options = getattr(input_obj, "options", None)
        input_id = str(getattr(input_obj, "id", "") or "")
        if input_id and isinstance(options, (list, tuple)):
            choices[input_id] = [
                entry if isinstance(entry, (str, int, float, bool)) else str(entry)
                for entry in cast("Sequence[object]", options)
            ]
    extras["choices"] = choices
    search_aliases = cast("Sequence[object]", getattr(schema, "search_aliases", ()) or ())
    if search_aliases:
        extras["searchAliases"] = [str(a) for a in search_aliases]
    try:
        # getattr resolves the classmethod to a bound method, which
        # inspect.getsource handles - same pattern as the v1 extras.
        fn = getattr(node_class, "execute", None)
        if fn is not None:
            extras["source"] = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        pass
    return extras


def _probe_v3(
    module: types.ModuleType,
    pack_id: str,
    translation: CompatTranslation,
    skipped: dict[str, str],
) -> list[dict[str, object]]:
    """Probe a V3 entrypoint's nodes into port entries. Per-node faults
    land in ``skipped`` with reasons (one exotic node must not take the
    pack down); an entrypoint that cannot resolve at all raises."""
    nodes: list[dict[str, object]] = []
    for node_class in _resolve_v3_nodes(module):
        name = getattr(node_class, "__name__", repr(node_class))
        try:
            # Prefer GET_SCHEMA - ComfyUI's own resolution path, which
            # finalizes (default output ids, None->[] lists) and validates
            # id uniqueness. Plain duck-typed fakes fall back to
            # define_schema and translate_v3 tolerates the raw shape.
            get_schema = getattr(node_class, "GET_SCHEMA", None)
            define_schema = getattr(node_class, "define_schema", None)
            if callable(get_schema):
                schema = get_schema()
            elif callable(define_schema):
                schema = define_schema()
            else:
                raise CompatError("V3 node class has no define_schema classmethod")
            node_id = str(getattr(schema, "node_id", "") or "")
            if node_id:
                name = node_id
            dinkster_schema = translate_v3_schema(schema, translation, namespace=pack_id)
            entry: dict[str, object] = {
                "sourceName": name,
                "sourceApi": "v3",
                "schema": schema_to_wire(dinkster_schema),
            }
            entry.update(_v3_node_extras(node_class, schema))
            nodes.append(entry)
        except CompatError as exc:
            skipped[name] = str(exc)
        except Exception as exc:  # noqa: BLE001 - define_schema runs pack code
            skipped[name] = f"{type(exc).__name__}: {exc}"
    return nodes


def probe_pack(path: Path, *, server_instance: object | None = None) -> dict[str, object]:
    """Import + translate one pack and return the port report as plain
    JSON-shaped data. Pure with respect to ComfyUI: callers that need the
    real environment bootstrap first (``main``); tests probe synthesized
    packs directly."""
    translation = CompatTranslation()
    report = load_legacy_pack(path, translation, server_instance=server_instance)
    nodes: list[dict[str, object]] = []
    v3_skipped: dict[str, str] = {}
    v3_error = ""
    if report.status == "loaded":
        module = sys.modules[pack_sys_module_name(path)]
        # status == "loaded" guarantees the loader saw non-empty mappings.
        mappings = cast("Mapping[str, type]", getattr(module, "NODE_CLASS_MAPPINGS", {}))
        translated = [
            name for name in mappings if f"{report.pack_id}.{name}" not in translation.skipped
        ]
        for v1_name, node_class in zip(translated, translation.node_classes, strict=True):
            entry: dict[str, object] = {
                "sourceName": v1_name,
                "sourceApi": "v1",
                "schema": schema_to_wire(node_class.define_schema()),
            }
            entry.update(_node_extras(mappings[v1_name]))
            nodes.append(entry)
    if report.status in ("loaded", "v3-entrypoint"):
        # v3-entrypoint means the import succeeded and the pack is pure V3;
        # loaded + an entrypoint is the mixed case ComfyUI half-ignores.
        module = sys.modules.get(pack_sys_module_name(path))
        if module is not None and callable(getattr(module, "comfy_entrypoint", None)):
            try:
                nodes.extend(_probe_v3(module, report.pack_id, translation, v3_skipped))
            except Exception as exc:  # noqa: BLE001 - entrypoint runs pack code
                v3_error = f"{type(exc).__name__}: {exc}"
    payload: dict[str, object] = {
        "portVersion": PORT_PROBE_VERSION,
        "report": asdict(report),
        "nodes": nodes,
        "opaqueTypes": sorted(translation.opaque_types),
    }
    if v3_skipped:
        payload["v3Skipped"] = v3_skipped
    if v3_error:
        payload["v3Error"] = v3_error
    return payload


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(
            "usage: python -m dinkster_compat_comfy.port_probe <pack-path> <out-json-path>",
            file=sys.stderr,
        )
        return 2
    pack_path = Path(args[0])
    out_path = Path(args[1])
    bootstrap_comfyui()
    server_instance = ensure_prompt_server()
    payload = probe_pack(pack_path, server_instance=server_instance)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry
    raise SystemExit(main())
