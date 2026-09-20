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

import contextlib
import importlib
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


def _atoms_of(expr: Any) -> set[str]:
    """Atom type ids referenced by a TypeExpr, recursing through lists.

    Wildcards and the variable's own template id contribute nothing; a
    variable's allowed-atoms constraint does.
    """
    atoms: set[str] = set(expr.types)
    if expr.element is not None:
        atoms |= _atoms_of(expr.element)
    return atoms


def _schema_atoms(schema: Any) -> set[str]:
    from dinkster_schema import DynamicComboSpec, InputFamilySpec, InputSpec

    atoms: set[str] = set()

    def visit(entries: Any) -> None:
        nonlocal atoms
        for entry in entries:
            if isinstance(entry, InputSpec):
                atoms |= _atoms_of(entry.type)
            elif isinstance(entry, InputFamilySpec):
                visit(entry.template)
            elif isinstance(entry, DynamicComboSpec):
                for option in entry.options:
                    visit(option.inputs)
            else:
                if entry.slot_type is not None:
                    atoms |= _atoms_of(entry.slot_type)
                visit(entry.inputs)
                for variant in entry.variants or ():
                    atoms |= _atoms_of(variant.type)
                    visit(variant.inputs)

    visit((*schema.inputs, *schema.input_families, *schema.combos, *schema.slots))
    for spec in (*schema.outputs, *schema.output_families):
        atoms |= _atoms_of(spec.type)
    if schema.output_descriptors is not None:
        for choice in schema.output_descriptors.choices:
            atoms |= _atoms_of(choice.type)
    return atoms


def probe(manifest_path: str) -> dict[str, Any]:
    from dinkster_workers.manifest import (
        add_pack_root_to_import_path,
        load_manifest,
        resolve_entry,
    )

    manifest = load_manifest(manifest_path)
    add_pack_root_to_import_path(manifest, sys.path)

    report: dict[str, Any] = {
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
            nodes_obj = resolve_entry(manifest.nodes_entry)
            types_fn = resolve_entry(manifest.types_entry) if manifest.types_entry else None
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

    from dinkster_schema import Node, NodeSchema, validate_replacement_references

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
        if schema.node_type in seen_types:
            report["duplicate_node_types"].append(
                f"{schema.node_type} (classes {seen_types[schema.node_type]} and {cls_name})"
            )
        else:
            seen_types[schema.node_type] = cls_name
            pack_schemas[schema.node_type] = schema
        atoms |= _schema_atoms(schema)
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
    if types_fn is not None:
        if not callable(types_fn):
            report["types_error"] = (
                f"entry.types must name a callable taking a TypeRegistry, "
                f"got {type(types_fn).__name__}"
            )
        else:
            try:
                types_fn(registry)
            except BaseException as exc:  # noqa: BLE001
                report["types_error"] = _exception_detail(exc)
    pack_ids = [tid for tid in registry.type_ids() if tid not in core_ids]
    report["registered_type_ids"] = pack_ids
    report["fallback_codec_type_ids"] = [
        tid for tid in pack_ids if not registry.spec(tid).declared_codec
    ]
    report["unregistered_schema_atoms"] = [atom for atom in sorted(atoms) if atom not in registry]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            from types import SimpleNamespace

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
                    renditions=registry.registered_renditions(),
                )
            )
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
            inference_entry = manifest.extension.entries.inference
            if inference_entry is not None:
                import tempfile
                from dataclasses import asdict

                inference = importlib.import_module("dinkster_inference.extensions")
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "inference.json"
                    key = "candidate:doctor"
                    inference.write_sampler_catalog(
                        path,
                        key,
                        (inference.SamplerExtensionEntry(manifest.name, inference_entry),),
                    )
                    generation = inference.materialize_inference_generation(key, catalog_path=path)
                    report["catalog"]["inferenceContributions"] = [
                        asdict(item) for item in generation.extensions[0][1]
                    ]
    except Exception as exc:
        report.pop("catalog", None)
        report["schema_errors"].append(f"catalog: {_exception_detail(exc)}")
    return report


def main() -> None:
    print(json.dumps(probe(sys.argv[1])))


if __name__ == "__main__":
    main()
