"""dinkster doctor's disposable import probe (DESIGN 3.9).

Run as ``python -m dinkster_workers._doctor_probe <manifest>`` in a subprocess,
never in the doctor's own process: importing a pack executes arbitrary code,
and the whole point of doctor's import checks is to observe what that import
does (output, threads, failures) without letting it touch the host. The
probe prints one JSON report to stdout and always exits 0 when the probe
itself worked - a broken pack is a *finding* in the report, not a probe
crash. (Underscore-private on purpose: packs importing this is exactly the
kind of thing doctor flags.)
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

_MAX_CAPTURED_LOGS = 200
_MAX_EXCEPTION_DETAIL = 2048


def _exception_detail(exc: BaseException) -> str:
    message = str(exc)
    environment_values = sorted(
        {value for value in os.environ.values() if len(value) >= 4},
        key=len,
        reverse=True,
    )
    for value in environment_values:
        message = message.replace(value, "<redacted>")
    if len(message) > _MAX_EXCEPTION_DETAIL:
        message = message[:_MAX_EXCEPTION_DETAIL] + "..."
    return f"{type(exc).__name__}: {message}"


class _LogCapture(logging.Handler):
    """Captures ``dinkster.*`` records emitted during pack import, structurally.

    Attached to the ``dinkster`` logger with propagation disabled for the
    duration of the import: legitimate ``pack_logger`` output is a separate,
    structured observation (``import_logs``), not raw stderr - otherwise the
    stdlib's last-resort handler would print the records into the redirected
    stderr and doctor would misclassify proper logging as side-effect output.
    """

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[dict[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if len(self.records) >= _MAX_CAPTURED_LOGS:
            return
        self.records.append(
            {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage()[:500],
            }
        )


@contextlib.contextmanager
def _capture_dinkster_logs(into: list[dict[str, str]]):
    dinkster_logger = logging.getLogger("dinkster")
    capture = _LogCapture()
    prev_propagate = dinkster_logger.propagate
    prev_level = dinkster_logger.level
    dinkster_logger.addHandler(capture)
    dinkster_logger.propagate = False
    dinkster_logger.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        dinkster_logger.removeHandler(capture)
        dinkster_logger.propagate = prev_propagate
        dinkster_logger.setLevel(prev_level)
        into.extend(capture.records)


def _cuda_initialized() -> bool:
    """True when pack import initialized a CUDA context - the expensive
    side effect worth flagging: an initialized context holds hundreds of
    MiB of VRAM per worker process for its whole lifetime (measured ~450
    MiB idle on an RTX 4090, torch 2.9/cu13; ~1 GiB with eager module
    loading), before any real work runs. Import-time torch is fine; what
    must stay lazy is the first CUDA op. Checked via sys.modules so packs
    that never pull torch in cost nothing here."""
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:  # noqa: BLE001 - stub/broken torch means no context
        return False


def probe(manifest_path: str, *, part: str | None = None) -> dict[str, Any]:
    from dinkster_protocol import EXTENSION_SCOPES

    from dinkster_workers.manifest import (
        add_pack_root_to_import_path,
        load_manifest,
        resolve_entry,
        validate_registration_claims,
    )

    manifest = load_manifest(manifest_path, part=part)
    add_pack_root_to_import_path(manifest, sys.path)

    report: dict[str, Any] = {
        "node_source": manifest.node_source,
        "entry_error": None,
        "entry_path": "",
        "interpreter": str(Path(sys.executable).absolute()),
        "import_ms": 0.0,
        "import_stdout": "",
        "import_stderr": "",
        "import_logs": [],
        "threads_started": 0,
        "cuda_initialized_on_import": False,
        "nodes": [],
        "schema_errors": [],
        "duplicate_node_types": [],
        "nodes_entry_problem": None,
        "schema_atoms": [],
        "registered_type_ids": [],
        "fallback_codec_type_ids": [],
        "unregistered_schema_atoms": [],
        "replacement_errors": [],
        "types_error": None,
        "extension_entry_errors": {},
    }

    threads_before = threading.active_count()
    out, err = io.StringIO(), io.StringIO()
    started = time.monotonic()
    try:
        with (
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
            _capture_dinkster_logs(cast("list[dict[str, str]]", report["import_logs"])),
        ):
            validate_registration_claims(manifest)
            nodes_obj = resolve_entry(manifest.nodes_entry)
            types_fn = resolve_entry(manifest.types_entry) if manifest.types_entry else None
            for scope in EXTENSION_SCOPES:
                entry = getattr(manifest.extension.entries, scope)
                if entry is None:
                    continue
                try:
                    resolve_entry(entry)
                except BaseException as exc:  # noqa: BLE001 - every accepted entry is probed
                    report["extension_entry_errors"][scope] = _exception_detail(exc)
    except BaseException as exc:  # noqa: BLE001 - pack code may raise anything
        report["entry_error"] = _exception_detail(exc)
        report["import_stdout"] = out.getvalue()[:4000]
        report["import_stderr"] = err.getvalue()[:4000]
        return report
    nodes_module = sys.modules.get(manifest.nodes_entry.partition(":")[0])
    nodes_module_file = getattr(nodes_module, "__file__", None)
    if nodes_module_file is not None:
        report["entry_path"] = str(Path(nodes_module_file).resolve())
    report["import_ms"] = (time.monotonic() - started) * 1000.0
    report["import_stdout"] = out.getvalue()[:4000]
    report["import_stderr"] = err.getvalue()[:4000]
    report["threads_started"] = max(threading.active_count() - threads_before, 0)
    report["cuda_initialized_on_import"] = _cuda_initialized()

    from dinkster_schema import Node, NodeSchema, claim_covers, validate_replacement_references
    from dinkster_schema.model import schema_type_atoms

    if not isinstance(nodes_obj, (list, tuple)):
        report["nodes_entry_problem"] = (
            f"entry.nodes must name a sequence of Node classes, got {type(nodes_obj).__name__}"
        )
        return report

    seen_types: dict[str, str] = {}
    pack_schemas: dict[str, NodeSchema] = {}
    atoms: set[str] = set()
    entries = tuple(cast("list[object] | tuple[object, ...]", nodes_obj))
    for cls in entries:
        cls_name = getattr(cls, "__name__", repr(cls))
        if not (isinstance(cls, type) and issubclass(cls, Node)):
            report["nodes_entry_problem"] = (
                f"entry.nodes contains {cls_name}, which is not a Node subclass"
            )
            continue
        try:
            schema = cls.schema()
        except BaseException as exc:  # noqa: BLE001
            report["schema_errors"].append(f"{cls_name}: {_exception_detail(exc)}")
            continue
        if (
            manifest.publisher != "dinkster"
            and schema.node_type not in manifest.executes
            and not claim_covers(manifest.name, schema.node_type)
        ):
            report["schema_errors"].append(
                f"node {schema.node_type!r} must use the pack-name prefix"
            )
        if schema.node_type in seen_types:
            report["duplicate_node_types"].append(
                f"{schema.node_type} (classes {seen_types[schema.node_type]} and {cls_name})"
            )
        else:
            seen_types[schema.node_type] = cls_name
            pack_schemas[schema.node_type] = schema
        atoms |= schema_type_atoms(schema)
        report["nodes"].append(
            {
                "node_type": schema.node_type,
                "version": schema.version,
                "description": schema.description,
                "class_name": cls_name,
                "execute_async": inspect.iscoroutinefunction(inspect.unwrap(cls.execute)),
            }
        )
    report["schema_atoms"] = sorted(atoms)
    # Static-id checks for pack-declared replacement rules: references into
    # schemas this pack ships are verifiable here; predecessors/targets from
    # other (possibly never-installed) packs are legitimately unchecked.
    report["replacement_errors"] = [
        problem.message for problem in validate_replacement_references(pack_schemas)
    ]

    from dinkster_values import TypeRegistry, register_core_types

    registry = TypeRegistry()
    register_core_types(registry)
    core_ids = set(registry.type_ids())
    if types_fn is not None and not callable(types_fn):
        report["types_error"] = (
            f"entry.types must name a callable taking a TypeRegistry, got {type(types_fn).__name__}"
        )
    else:
        try:
            with registry.registering_pack(
                manifest.name,
                manifest.type_namespaces,
                allow_bare=manifest.publisher == "dinkster",
                uses=manifest.uses_types,
            ):
                if types_fn is not None:
                    types_fn(registry)
        except BaseException as exc:  # noqa: BLE001
            report["types_error"] = _exception_detail(exc)
    used = set(registry.used_type_ids(manifest.name))
    pack_ids = [tid for tid in registry.type_ids() if tid not in core_ids | used]
    report["registered_type_ids"] = pack_ids
    report["fallback_codec_type_ids"] = [
        tid for tid in pack_ids if not registry.spec(tid).declared_codec
    ]
    report["unregistered_schema_atoms"] = [atom for atom in sorted(atoms) if atom not in registry]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            from types import SimpleNamespace

            from dinkster_protocol.lookups import lookup_functions, lookup_marks

            from .catalog import worker_declarations
            from .host import load_choices, load_extension_contributions, load_skips

            choices = load_choices(manifest)
            report["catalog"] = worker_declarations(
                SimpleNamespace(
                    schemas=pack_schemas,
                    combo_choices=choices.static,
                    lazy_choice_ids=tuple(choices.lazy),
                    compat_skips=load_skips(manifest),
                    body_arms=dict(manifest.arms),
                    extension_contributions=load_extension_contributions(manifest),
                    lookup_marks=lookup_marks(lookup_functions(pack_schemas)),
                    renditions=registry.registered_renditions(),
                    # Match worker ownership: dependency codecs are usable here
                    # but are not declarations owned by this pack.
                    registered_types=[
                        tid
                        for tid in pack_ids
                        if not any(
                            dependency.pack == registry.owner_of(tid)
                            for dependency in manifest.dependencies
                        )
                    ],
                )
            )
            report["catalog"]["residentTypes"] = [
                tid for tid in registry.type_ids() if registry.spec(tid).may_hold_resident
            ]
            report["catalog"]["types"] = {
                "typeIds": pack_ids,
                "assetDecoders": list(registry.asset_decoder_targets()),
                "batchMerges": [
                    tid for tid in registry.type_ids() if registry.batch_merge_for(tid) is not None
                ],
                "equivalences": {
                    tid: other
                    for tid in registry.type_ids()
                    if (other := registry.equivalent_type(tid)) is not None
                },
            }
    except Exception as exc:
        report.pop("catalog", None)
        report["schema_errors"].append(f"catalog: {_exception_detail(exc)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--part")
    args = parser.parse_args()
    print(json.dumps(probe(args.manifest, part=args.part)))


if __name__ == "__main__":
    main()
