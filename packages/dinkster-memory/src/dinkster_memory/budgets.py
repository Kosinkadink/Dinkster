"""Strict persisted memory-budget configuration.

``memory.toml`` supplies operator defaults for the same residency-class
budgets accepted by ``dinkster-serve --memory-budget``. A missing file means
no defaults; a present malformed file fails loudly instead of silently
running without the intended memory limits.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from dinkster_values.limits import GIBIBYTE, MEBIBYTE

__all__ = ["BudgetsError", "load_budgets", "parse_budgets", "parse_size"]

_SIZE_SUFFIXES = {"k": 1024, "m": MEBIBYTE, "g": GIBIBYTE, "t": 1024 * GIBIBYTE}


class BudgetsError(Exception):
    """A persisted memory budget is malformed."""


def parse_size(text: str) -> int:
    """Parse a byte count with an optional binary K/M/G/T suffix.

    Zero is a valid size: the governor accepts budgets >= 0, and a zero
    budget is the operator's "deny every reservation on this device"
    knob (and what --memory-budget accepted before this module existed).
    """
    size = text.strip().lower()
    if not size:
        raise ValueError("size must not be empty")
    factor = _SIZE_SUFFIXES.get(size[-1], 1)
    digits = size[:-1] if factor != 1 else size
    if not digits.isdigit():
        raise ValueError("size must be a byte count with an optional K/M/G/T suffix")
    return int(digits) * factor


def parse_budgets(data: object, source: str) -> dict[str, int]:
    """Validate a TOML-decoded ``[budgets]`` mapping."""
    if not isinstance(data, Mapping):
        raise BudgetsError(f"{source}: config must be a table")
    top = cast("Mapping[object, object]", data)
    unknown_top = {str(key) for key in top} - {"budgets"}
    if unknown_top:
        raise BudgetsError(
            f"{source}: unknown top-level keys {sorted(unknown_top)} "
            f"(everything lives under [budgets])"
        )
    table = top.get("budgets", {})
    if not isinstance(table, Mapping):
        raise BudgetsError(f"{source}: 'budgets' must be a table")

    budgets: dict[str, int] = {}
    for device_raw, size_raw in cast("Mapping[object, object]", table).items():
        if (
            not isinstance(device_raw, str)
            or not device_raw
            or device_raw != device_raw.strip()
            or any(character.isspace() for character in device_raw)
        ):
            raise BudgetsError(
                f"{source}: budget device keys must be non-empty strings "
                f"without whitespace, got {device_raw!r}"
            )
        where = f"{source}: budget for {device_raw!r}"
        if isinstance(size_raw, bool):
            raise BudgetsError(f"{where}: size must be a string or integer bytes")
        if isinstance(size_raw, int):
            if size_raw < 0:
                raise BudgetsError(f"{where}: integer bytes must be >= 0")
            budgets[device_raw] = size_raw
            continue
        if not isinstance(size_raw, str):
            raise BudgetsError(f"{where}: size must be a string or integer bytes")
        try:
            budgets[device_raw] = parse_size(size_raw)
        except ValueError as exc:
            raise BudgetsError(f"{where}: {exc}") from None
    return budgets


def load_budgets(path: Path) -> dict[str, int]:
    """Load memory budgets; a missing file means no persisted defaults."""
    if not path.exists():
        return {}
    try:
        data = tomllib.loads(path.read_text("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise BudgetsError(f"{path}: invalid TOML: {exc}") from exc
    return parse_budgets(data, str(path))
