"""Build and verify behavioral confidence receipts for ComfyUI translations.

The harness reads only JSON and NumPy artifacts. It never imports workflow
fixtures or custom-node packages. Executed references must identify either the
pinned ComfyUI core revision or an exact ecosystem commit.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import zipfile
from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal, DecimalException, Inexact, localcontext
from pathlib import Path, PurePosixPath
from typing import Any, cast

FORMAT = "dinkster-comfy-confidence-receipt/1"
COMFYUI_REFERENCE_REVISION = "b78cec87"
COMFYUI_REFERENCE_REVISIONS = frozenset(
    (
        COMFYUI_REFERENCE_REVISION,
        "c67885b1",
        "8a33128f",
        "15eb748b",
        "f00bfd610cb001381603669e2cc01160ae37aaf3",
    )
)
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_ARRAY_ELEMENTS = 16 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_ITEMS = 100_000
MAX_DECIMAL_PRECISION = 100_000

_COMPARATORS = frozenset({"exact-value/1", "numeric-value/1", "exact-array/1", "numeric-array/1"})
_DATA_KINDS = frozenset({"value", "image", "mask", "tensor"})
_MAPPING_KINDS = frozenset({"op", "family"})
_TIERS = frozenset({"exact", "parametric", "equivalent", "grouped"})
_REFERENCE_KINDS = frozenset({"comfyui-pinned", "ecosystem-pinned", "static"})
_TARGET_KINDS = frozenset({"node", "group"})
_TOLERANCE_OPERATORS = frozenset({"<="})
_NUMERIC_METRICS = frozenset({"max_abs", "mean_abs", "mismatch_fraction"})
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_PACK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class ConfidenceReceiptError(ValueError):
    """A receipt or artifact is malformed, unsafe, stale, or inconsistent."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ConfidenceReceiptError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ConfidenceReceiptError(f"non-finite JSON number: {value}")


def canonical_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise ConfidenceReceiptError(f"receipt is not finite JSON data: {error}") from error
    return (encoded + "\n").encode("ascii")


def _object(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfidenceReceiptError(f"{where} must be an object with string keys")
    return cast("dict[str, Any]", value)


def _array(value: object, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfidenceReceiptError(f"{where} must be an array")
    return cast("list[Any]", value)


def _fields(
    value: object,
    where: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    obj = _object(value, where)
    allowed = required | (optional or set())
    missing = sorted(required - set(obj))
    unknown = sorted(set(obj) - allowed)
    if missing:
        raise ConfidenceReceiptError(f"{where} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ConfidenceReceiptError(f"{where} has unknown fields: {', '.join(unknown)}")
    return obj


def _string(value: object, where: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ConfidenceReceiptError(
            f"{where} must be a non-empty string of at most {maximum} chars"
        )
    return value


def _finite_number(value: object, where: str) -> float:
    if type(value) not in (int, float):
        raise ConfidenceReceiptError(f"{where} must be a finite number")
    numeric = cast("int | float", value)
    try:
        finite = math.isfinite(numeric)
    except OverflowError:
        finite = False
    if not finite:
        raise ConfidenceReceiptError(f"{where} must be a finite number")
    return float(numeric)


def _validate_json(value: object, where: str, *, decimal_numbers: bool = False) -> None:
    remaining = MAX_JSON_ITEMS
    stack: list[tuple[object, int, str]] = [(value, 0, where)]
    while stack:
        item, depth, location = stack.pop()
        remaining -= 1
        if remaining < 0:
            raise ConfidenceReceiptError(f"{where} exceeds the {MAX_JSON_ITEMS}-item limit")
        if depth > MAX_JSON_DEPTH:
            raise ConfidenceReceiptError(f"{where} exceeds the {MAX_JSON_DEPTH}-level limit")
        if item is None or type(item) in (bool, int, str):
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise ConfidenceReceiptError(f"{location} must be finite")
            continue
        if decimal_numbers and isinstance(item, Decimal):
            if not item.is_finite():
                raise ConfidenceReceiptError(f"{location} must be finite")
            continue
        if isinstance(item, list):
            stack.extend(
                (child, depth + 1, f"{location}[{index}]") for index, child in enumerate(item)
            )
            continue
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            stack.extend((child, depth + 1, f"{location}.{key}") for key, child in item.items())
            continue
        raise ConfidenceReceiptError(f"{location} is not JSON data")


def _validate_mapping(value: object) -> dict[str, Any]:
    mapping = _fields(
        value,
        "receipt.mapping",
        {"registryId", "mappingKind", "tier", "source", "target"},
    )
    registry_id = _string(mapping["registryId"], "receipt.mapping.registryId")
    if registry_id.startswith("comfy_alias:"):
        registry_kind = "alias"
    elif registry_id.startswith("comfy_group:"):
        registry_kind = "group"
    else:
        raise ConfidenceReceiptError(
            "receipt.mapping.registryId must start with comfy_alias: or comfy_group:"
        )
    mapping_kind = _string(mapping["mappingKind"], "receipt.mapping.mappingKind")
    if mapping_kind not in _MAPPING_KINDS:
        raise ConfidenceReceiptError("receipt.mapping.mappingKind must be op or family")
    tier = _string(mapping["tier"], "receipt.mapping.tier")
    if tier not in _TIERS:
        raise ConfidenceReceiptError(f"unsupported receipt confidence tier: {tier}")
    if (registry_kind == "group") != (tier == "grouped"):
        raise ConfidenceReceiptError(
            "group receipts require grouped confidence and alias receipts forbid it"
        )

    source = _fields(
        mapping["source"],
        "receipt.mapping.source",
        {"pack", "name", "revision", "referenceKind"},
    )
    pack = _string(source["pack"], "receipt.mapping.source.pack", maximum=128)
    if _PACK_ID.fullmatch(pack) is None:
        raise ConfidenceReceiptError("receipt.mapping.source.pack is not a canonical pack id")
    name = _string(source["name"], "receipt.mapping.source.name", maximum=256)
    revision = _string(source["revision"], "receipt.mapping.source.revision", maximum=128)
    reference_kind = _string(source["referenceKind"], "receipt.mapping.source.referenceKind")
    if reference_kind not in _REFERENCE_KINDS:
        raise ConfidenceReceiptError(
            "source referenceKind must be comfyui-pinned, ecosystem-pinned, or static"
        )
    if pack == "comfy-core" and revision not in COMFYUI_REFERENCE_REVISIONS:
        revisions = ", ".join(sorted(COMFYUI_REFERENCE_REVISIONS))
        raise ConfidenceReceiptError(f"comfy-core receipts must use one of revisions: {revisions}")
    if reference_kind == "comfyui-pinned" and (
        pack != "comfy-core" or revision not in COMFYUI_REFERENCE_REVISIONS
    ):
        raise ConfidenceReceiptError(
            "executed references are allowed only for a pinned ComfyUI core revision"
        )
    if reference_kind == "ecosystem-pinned" and (
        pack == "comfy-core" or _COMMIT.fullmatch(revision) is None
    ):
        raise ConfidenceReceiptError(
            "ecosystem-pinned references require a non-core pack and full commit"
        )
    expected_id = f"comfy_{registry_kind}:{pack}/{name}"
    if registry_id != expected_id:
        raise ConfidenceReceiptError(f"receipt.mapping.registryId must be {expected_id!r}")

    target = _fields(mapping["target"], "receipt.mapping.target", {"kind", "id"})
    target_kind = _string(target["kind"], "receipt.mapping.target.kind")
    if target_kind not in _TARGET_KINDS:
        raise ConfidenceReceiptError("receipt.mapping.target.kind must be node or group")
    _string(target["id"], "receipt.mapping.target.id")
    return mapping


def _validate_comparison(value: object, tier: str) -> dict[str, Any]:
    comparison = _fields(
        value,
        "receipt.comparison",
        {"comparator", "dataKind"},
        {"tolerances"},
    )
    comparator = _string(comparison["comparator"], "receipt.comparison.comparator")
    data_kind = _string(comparison["dataKind"], "receipt.comparison.dataKind")
    if comparator not in _COMPARATORS:
        raise ConfidenceReceiptError(f"unsupported comparator: {comparator}")
    if data_kind not in _DATA_KINDS:
        raise ConfidenceReceiptError(f"unsupported comparison data kind: {data_kind}")
    value_comparator = comparator in {"exact-value/1", "numeric-value/1"}
    if value_comparator != (data_kind == "value"):
        raise ConfidenceReceiptError("value data and value comparators must be used together")

    tolerance_wires = _array(comparison.get("tolerances", []), "receipt.comparison.tolerances")
    tolerances: list[tuple[str, str, float]] = []
    for index, value in enumerate(tolerance_wires):
        where = f"receipt.comparison.tolerances[{index}]"
        tolerance = _fields(value, where, {"metric", "operator", "value"})
        metric = _string(tolerance["metric"], f"{where}.metric")
        operator = _string(tolerance["operator"], f"{where}.operator")
        threshold = _finite_number(tolerance["value"], f"{where}.value")
        if metric not in _NUMERIC_METRICS:
            raise ConfidenceReceiptError(f"unsupported tolerance metric: {metric}")
        if operator not in _TOLERANCE_OPERATORS:
            raise ConfidenceReceiptError(f"unsupported tolerance operator: {operator}")
        if threshold < 0 or (metric == "mismatch_fraction" and threshold > 1):
            raise ConfidenceReceiptError(f"invalid tolerance threshold for {metric}: {threshold}")
        tolerances.append((metric, operator, threshold))
    if len({metric for metric, _, _ in tolerances}) != len(tolerances):
        raise ConfidenceReceiptError("receipt comparison has duplicate tolerance metrics")
    numeric = comparator in {"numeric-value/1", "numeric-array/1"}
    if numeric != bool(tolerances) or (not numeric and "tolerances" in comparison):
        raise ConfidenceReceiptError(
            "numeric comparators require tolerances and exact comparators forbid them"
        )
    if tier == "exact" and comparator not in {"exact-value/1", "exact-array/1"}:
        raise ConfidenceReceiptError("exact confidence requires an exact comparator")
    if tier == "equivalent" and not numeric:
        raise ConfidenceReceiptError("equivalent confidence requires a numeric comparator")
    return comparison


def _validate_artifact(value: object, where: str) -> dict[str, Any]:
    artifact = _fields(value, where, {"path", "bytes", "sha256"})
    relative = _string(artifact["path"], f"{where}.path")
    _relative_parts(relative)
    size = artifact["bytes"]
    if type(size) is not int or not 0 < size <= MAX_ARTIFACT_BYTES:
        raise ConfidenceReceiptError(
            f"{where}.bytes must be an integer from 1 through {MAX_ARTIFACT_BYTES}"
        )
    digest = _string(artifact["sha256"], f"{where}.sha256")
    if _SHA256.fullmatch(digest) is None:
        raise ConfidenceReceiptError(f"{where}.sha256 must be a lowercase sha256 digest")
    return artifact


def _validate_observed(value: object, comparator: str) -> dict[str, Any]:
    observed = _object(value, "receipt.observed")
    if comparator == "exact-value/1":
        expected = {"equal"}
    elif comparator == "exact-array/1":
        expected = {"dtype_equal", "shape_equal", "values_equal"}
    else:
        expected = {"shape_equal", "max_abs", "mean_abs", "mismatch_fraction"}
    if set(observed) != expected:
        raise ConfidenceReceiptError("receipt.observed fields do not match the declared comparator")
    if comparator.startswith("exact-"):
        if any(type(item) is not bool for item in observed.values()):
            raise ConfidenceReceiptError("exact comparator observations must be booleans")
    else:
        if type(observed["shape_equal"]) is not bool:
            raise ConfidenceReceiptError("numeric shape_equal observation must be boolean")
        for metric in _NUMERIC_METRICS:
            item = observed[metric]
            if item is not None:
                _finite_number(item, f"receipt.observed.{metric}")
    return observed


def _recorded_pass(observed: dict[str, Any], comparison: dict[str, Any]) -> bool:
    comparator = comparison["comparator"]
    if comparator == "exact-value/1":
        return cast("bool", observed["equal"])
    if comparator == "exact-array/1":
        if observed["values_equal"] and not (observed["shape_equal"] and observed["dtype_equal"]):
            raise ConfidenceReceiptError(
                "exact array equality requires matching shape and dtype observations"
            )
        return cast("bool", observed["values_equal"])
    shape_equal = cast("bool", observed["shape_equal"])
    metric_values = [observed[metric] for metric in _NUMERIC_METRICS]
    if shape_equal != all(value is not None for value in metric_values):
        raise ConfidenceReceiptError(
            "numeric observations require all metrics exactly when shapes match"
        )
    if not shape_equal:
        return False
    passed = True
    for tolerance in comparison["tolerances"]:
        actual = cast("float", observed[tolerance["metric"]])
        passed = passed and actual <= float(tolerance["value"])
    return passed


def validate_receipt(value: object) -> dict[str, Any]:
    """Validate the closed receipt schema without reading artifact files."""
    receipt = _fields(
        value,
        "receipt",
        {
            "format",
            "caseId",
            "mapping",
            "parameters",
            "comparison",
            "artifacts",
            "observed",
            "pass",
        },
    )
    if receipt["format"] != FORMAT:
        raise ConfidenceReceiptError(f"unsupported confidence receipt format: {receipt['format']}")
    case_id = _string(receipt["caseId"], "receipt.caseId", maximum=256)
    if _CASE_ID.fullmatch(case_id) is None:
        raise ConfidenceReceiptError("receipt.caseId is not canonical")
    mapping = _validate_mapping(receipt["mapping"])
    parameters = _object(receipt["parameters"], "receipt.parameters")
    _validate_json(parameters, "receipt.parameters")
    comparison = _validate_comparison(receipt["comparison"], cast("str", mapping["tier"]))
    artifacts = _fields(receipt["artifacts"], "receipt.artifacts", {"reference", "native"})
    reference = _validate_artifact(artifacts["reference"], "receipt.artifacts.reference")
    native = _validate_artifact(artifacts["native"], "receipt.artifacts.native")
    if reference["path"] == native["path"]:
        raise ConfidenceReceiptError("reference and native artifacts must use distinct paths")
    observed = _validate_observed(receipt["observed"], cast("str", comparison["comparator"]))
    if type(receipt["pass"]) is not bool:
        raise ConfidenceReceiptError("receipt.pass must be a boolean")
    if receipt["pass"] is not _recorded_pass(observed, comparison):
        raise ConfidenceReceiptError("receipt.pass does not match the recorded observations")
    if len(canonical_bytes(receipt)) > MAX_RECEIPT_BYTES:
        raise ConfidenceReceiptError(f"receipt exceeds the {MAX_RECEIPT_BYTES}-byte limit")
    return receipt


def _identity(value: os.stat_result) -> tuple[int, ...]:
    identity = (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if os.name == "nt":
        return identity
    return (*identity, value.st_ctime_ns)


def _read_bounded_regular(
    path: Path,
    expected: os.stat_result,
    maximum: int,
    label: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or _identity(before) != _identity(expected):
                raise ConfidenceReceiptError(f"{label} changed before reading: {path}")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                data = stream.read(maximum + 1)
            after = os.fstat(descriptor)
            current = path.lstat()
            if _identity(before) != _identity(after) or _identity(after) != _identity(current):
                raise ConfidenceReceiptError(f"{label} changed while reading: {path}")
        finally:
            os.close(descriptor)
    except ConfidenceReceiptError:
        raise
    except OSError as error:
        raise ConfidenceReceiptError(f"cannot read {label} {path}: {error}") from error
    if len(data) != expected.st_size or len(data) > maximum:
        raise ConfidenceReceiptError(f"{label} changed size while reading: {path}")
    return data


def load_receipt(path: Path) -> dict[str, Any]:
    """Load one canonical receipt with duplicate-key and size checks."""
    try:
        path_stat = path.lstat()
    except OSError as error:
        raise ConfidenceReceiptError(f"cannot inspect receipt {path}: {error}") from error
    if not stat.S_ISREG(path_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
        raise ConfidenceReceiptError("receipt must be an ordinary file")
    if not 0 < path_stat.st_size <= MAX_RECEIPT_BYTES:
        raise ConfidenceReceiptError(
            f"receipt size must be from 1 through {MAX_RECEIPT_BYTES} bytes"
        )
    encoded = _read_bounded_regular(path, path_stat, MAX_RECEIPT_BYTES, "receipt")
    try:
        value = json.loads(
            encoded.decode("ascii"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except ConfidenceReceiptError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ConfidenceReceiptError(f"receipt is not valid JSON: {error}") from error
    receipt = validate_receipt(value)
    if canonical_bytes(receipt) != encoded:
        raise ConfidenceReceiptError("receipt is not canonical JSON")
    return receipt


def write_receipt(path: Path, value: object) -> None:
    """Atomically write one canonical validated receipt."""
    receipt = validate_receipt(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_bytes(receipt))
    os.replace(temporary, path)


def _relative_parts(relative: str) -> tuple[str, ...]:
    if "\\" in relative:
        raise ConfidenceReceiptError(f"artifact path must use POSIX separators: {relative}")
    path = PurePosixPath(relative)
    parts = path.parts
    if path.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise ConfidenceReceiptError(f"artifact path is not contained: {relative}")
    if path.as_posix() != relative:
        raise ConfidenceReceiptError(f"artifact path is not canonical: {relative}")
    return parts


def _is_reparse_point(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _read_artifact(root: Path, relative: str) -> tuple[bytes, tuple[int, int]]:
    parts = _relative_parts(relative)
    try:
        root_stat = root.lstat()
    except OSError as error:
        raise ConfidenceReceiptError(f"cannot inspect artifact root {root}: {error}") from error
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_ISLNK(root_stat.st_mode)
        or _is_reparse_point(root_stat)
    ):
        raise ConfidenceReceiptError("artifact root must be an ordinary directory")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_path = root.joinpath(*parts).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ConfidenceReceiptError(f"cannot resolve artifact path: {error}") from error
    if not resolved_path.is_relative_to(resolved_root):
        raise ConfidenceReceiptError(f"artifact path is not contained: {relative}")
    path = root
    try:
        for part in parts[:-1]:
            path = path / part
            item_stat = path.lstat()
            if (
                not stat.S_ISDIR(item_stat.st_mode)
                or stat.S_ISLNK(item_stat.st_mode)
                or _is_reparse_point(item_stat)
            ):
                raise ConfidenceReceiptError(f"artifact directory is not ordinary: {path}")
        path = path / parts[-1]
        path_stat = path.lstat()
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or stat.S_ISLNK(path_stat.st_mode)
            or _is_reparse_point(path_stat)
        ):
            raise ConfidenceReceiptError(f"artifact is not an ordinary file: {path}")
        if not 0 < path_stat.st_size <= MAX_ARTIFACT_BYTES:
            raise ConfidenceReceiptError(
                f"artifact size must be from 1 through {MAX_ARTIFACT_BYTES}: {path}"
            )
        data = _read_bounded_regular(path, path_stat, MAX_ARTIFACT_BYTES, "artifact")
    except ConfidenceReceiptError:
        raise
    except OSError as error:
        raise ConfidenceReceiptError(f"cannot read artifact {path}: {error}") from error
    return data, (path_stat.st_dev, path_stat.st_ino)


def _descriptor(root: Path, relative: str) -> tuple[dict[str, object], bytes, tuple[int, int]]:
    data, identity = _read_artifact(root, relative)
    return (
        {
            "path": relative,
            "bytes": len(data),
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        },
        data,
        identity,
    )


def _require_distinct_artifacts(reference: tuple[int, int], native: tuple[int, int]) -> None:
    if reference == native:
        raise ConfidenceReceiptError("reference and native artifacts must be distinct files")


def _json_artifact(data: bytes, where: str) -> object:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=Decimal,
        )
    except ConfidenceReceiptError:
        raise
    except (DecimalException, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ConfidenceReceiptError(f"{where} is not valid JSON: {error}") from error
    _validate_json(value, where, decimal_numbers=True)
    return value


def _json_number(value: object, where: str) -> Decimal:
    if type(value) is int:
        return Decimal(value)
    if isinstance(value, Decimal) and value.is_finite():
        return value
    raise ConfidenceReceiptError(f"{where} must be a finite JSON number")


def _array_artifact(data: bytes, where: str) -> Any:
    import numpy as np

    try:
        loaded = np.load(io.BytesIO(data), allow_pickle=False)
    except (EOFError, OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise ConfidenceReceiptError(f"{where} is not a safe NumPy array: {error}") from error
    if not isinstance(loaded, np.ndarray):
        if hasattr(loaded, "close"):
            loaded.close()
        raise ConfidenceReceiptError(f"{where} must be one .npy array")
    if loaded.ndim > 8 or loaded.size > MAX_ARRAY_ELEMENTS:
        raise ConfidenceReceiptError(f"{where} exceeds the array geometry limits")
    if loaded.dtype.kind not in "biuf" or loaded.dtype.hasobject:
        raise ConfidenceReceiptError(f"{where} must use a real numeric or boolean dtype")
    if not bool(np.isfinite(loaded).all()):
        raise ConfidenceReceiptError(f"{where} contains non-finite values")
    return loaded


def _exact_json(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        right_object = cast("dict[object, object]", right)
        return left.keys() == right_object.keys() and all(
            _exact_json(value, right_object[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        right_array = cast("list[object]", right)
        return len(left) == len(right_array) and all(
            _exact_json(a, b) for a, b in zip(left, right_array, strict=True)
        )
    return left == right


def _normalize_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _decimal_difference(left: Decimal, right: Decimal) -> Decimal:
    if left == right:
        return Decimal(0)
    parts = [value.as_tuple() for value in (left, right) if not value.is_zero()]
    minimum_exponent = min(cast("int", part.exponent) for part in parts)
    precision = (
        max(len(part.digits) + cast("int", part.exponent) - minimum_exponent for part in parts) + 1
    )
    if precision > MAX_DECIMAL_PRECISION:
        raise ConfidenceReceiptError("numeric JSON difference exceeds float64")
    context = Context(prec=precision, Emin=MIN_EMIN, Emax=MAX_EMAX)
    context.traps[Inexact] = True
    try:
        with localcontext(context):
            return abs(left - right)
    except DecimalException as exception:
        raise ConfidenceReceiptError("numeric JSON difference exceeds float64") from exception


def _compare(
    reference_data: bytes,
    native_data: bytes,
    comparison: dict[str, Any],
) -> tuple[dict[str, object], bool]:
    import numpy as np

    comparator = comparison["comparator"]
    if comparator in {"exact-value/1", "numeric-value/1"}:
        reference_value = _json_artifact(reference_data, "reference value artifact")
        native_value = _json_artifact(native_data, "native value artifact")
        if comparator == "numeric-value/1":
            reference_number = _json_number(reference_value, "reference value artifact")
            native_number = _json_number(native_value, "native value artifact")
            try:
                decimal_error = _decimal_difference(reference_number, native_number)
                error = float(decimal_error)
            except (DecimalException, OverflowError, ValueError) as exception:
                raise ConfidenceReceiptError(
                    "numeric JSON difference exceeds float64"
                ) from exception
            if not math.isfinite(error) or (decimal_error != 0 and error == 0):
                raise ConfidenceReceiptError("numeric JSON difference exceeds float64")
            error = _normalize_zero(error)
            observed_value: dict[str, object] = {
                "shape_equal": True,
                "max_abs": error,
                "mean_abs": error,
                "mismatch_fraction": 0.0 if error == 0.0 else 1.0,
            }
            return observed_value, _recorded_pass(observed_value, comparison)
        equal = _exact_json(reference_value, native_value)
        return {"equal": equal}, equal

    reference = _array_artifact(reference_data, "reference array artifact")
    native = _array_artifact(native_data, "native array artifact")
    shape_equal = reference.shape == native.shape
    if comparator == "exact-array/1":
        dtype_equal = reference.dtype == native.dtype
        values_equal = bool(shape_equal and dtype_equal and np.array_equal(reference, native))
        observed = {
            "dtype_equal": dtype_equal,
            "shape_equal": shape_equal,
            "values_equal": values_equal,
        }
        return observed, values_equal

    if reference.dtype.kind != "f" or native.dtype.kind != "f":
        raise ConfidenceReceiptError("numeric-array/1 requires floating-point arrays")
    observed_numeric: dict[str, object] = {
        "shape_equal": shape_equal,
        "max_abs": None,
        "mean_abs": None,
        "mismatch_fraction": None,
    }
    if not shape_equal:
        return observed_numeric, False
    with np.errstate(over="ignore", invalid="ignore"):
        error = np.abs(reference.astype(np.float64) - native.astype(np.float64))
    if not bool(np.isfinite(error).all()):
        raise ConfidenceReceiptError("numeric comparison produced a non-finite error")
    if error.size == 0:
        max_abs = mean_abs = mismatch_fraction = 0.0
    else:
        max_abs = _normalize_zero(float(error.max()))
        mean_abs = _normalize_zero(float(error.mean()))
        mismatch_fraction = _normalize_zero(float(np.count_nonzero(error) / error.size))
    observed_numeric.update(
        {
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "mismatch_fraction": mismatch_fraction,
        }
    )
    return observed_numeric, _recorded_pass(observed_numeric, comparison)


def create_receipt(
    *,
    case_id: str,
    mapping: dict[str, object],
    parameters: dict[str, object],
    comparison: dict[str, object],
    artifact_root: Path,
    reference_path: str,
    native_path: str,
) -> dict[str, object]:
    """Compare two artifacts and return a validated deterministic receipt."""
    validated_mapping = _validate_mapping(mapping)
    _validate_json(parameters, "receipt.parameters")
    validated_comparison = _validate_comparison(comparison, cast("str", mapping["tier"]))
    reference_descriptor, reference_data, reference_identity = _descriptor(
        artifact_root, reference_path
    )
    native_descriptor, native_data, native_identity = _descriptor(artifact_root, native_path)
    _require_distinct_artifacts(reference_identity, native_identity)
    observed, passed = _compare(reference_data, native_data, validated_comparison)
    receipt: dict[str, object] = {
        "format": FORMAT,
        "caseId": case_id,
        "mapping": validated_mapping,
        "parameters": parameters,
        "comparison": validated_comparison,
        "artifacts": {
            "reference": reference_descriptor,
            "native": native_descriptor,
        },
        "observed": observed,
        "pass": passed,
    }
    return cast("dict[str, object]", validate_receipt(receipt))


def verify_receipt(receipt: object, artifact_root: Path) -> dict[str, Any]:
    """Verify artifact hashes and reproduce the recorded comparison exactly."""
    validated = validate_receipt(receipt)
    artifacts = cast("dict[str, dict[str, object]]", validated["artifacts"])
    loaded: dict[str, bytes] = {}
    identities: dict[str, tuple[int, int]] = {}
    for role in ("reference", "native"):
        expected = artifacts[role]
        actual, data, identity = _descriptor(artifact_root, cast("str", expected["path"]))
        if actual != expected:
            raise ConfidenceReceiptError(f"{role} artifact hash or size does not match receipt")
        loaded[role] = data
        identities[role] = identity
    _require_distinct_artifacts(identities["reference"], identities["native"])
    observed, passed = _compare(
        loaded["reference"],
        loaded["native"],
        cast("dict[str, Any]", validated["comparison"]),
    )
    if not _exact_json(observed, validated["observed"]) or passed is not validated["pass"]:
        raise ConfidenceReceiptError("receipt comparison result does not reproduce")
    return validated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = verify_receipt(load_receipt(args.receipt), args.artifact_root)
    except ConfidenceReceiptError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(canonical_bytes(receipt).decode("ascii"), end="")
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
