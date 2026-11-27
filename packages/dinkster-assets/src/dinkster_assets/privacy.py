"""Invocation-local credential checks before managed bytes reach a file.

This protects literal credential bytes, not arbitrary transformations or writes
outside the managed asset API. It is not a Python sandbox.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import cast

from .identity import AssetError

_secret_values: ContextVar[tuple[str, ...]] = ContextVar("asset_secret_values", default=())


@contextmanager
def protect_secret_outputs(values: Iterable[str]) -> Generator[None]:
    """Protect nested and copied-context asset writes for the active invocation."""
    token = _secret_values.set(tuple(dict.fromkeys((*_secret_values.get(), *values))))
    try:
        yield
    finally:
        _secret_values.reset(token)


def check_secret_text(text: str) -> None:
    if any(value and value in text for value in _secret_values.get()):
        raise AssetError("Secret-bearing content cannot be saved")


def secret_output_protection_active() -> bool:
    return bool(_secret_values.get())


def check_secret_data(data: object) -> None:
    """Check literal text and bytes in a value's portable fields before retention."""
    if not secret_output_protection_active():
        return
    patterns = tuple(value.encode("utf-8") for value in _secret_values.get() if value)
    seen: set[int] = set()

    def check(item: object) -> None:
        if isinstance(item, str):
            check_secret_text(item)
        elif isinstance(item, (bytes, bytearray, memoryview)):
            encoded = bytes(cast("bytes | bytearray | memoryview[int]", item))
            if any(value in encoded for value in patterns):
                raise AssetError("Secret-bearing content cannot be saved")
        elif id(item) not in seen:
            seen.add(id(item))
            if isinstance(item, Mapping):
                for key, value in cast("Mapping[object, object]", item).items():
                    check(key)
                    check(value)
            elif isinstance(item, Sequence):
                for value in cast("Sequence[object]", item):
                    check(value)

    check(data)


class SecretOutputStream:
    """Withhold a boundary tail so a split credential never reaches staging files."""

    def __init__(self) -> None:
        self._patterns: tuple[bytes, ...] = ()
        self._tail = b""
        self._refused = False

    def write(self, data: bytes) -> bytes:
        if self._refused:
            raise AssetError("Secret-bearing content cannot be saved")
        self._patterns = tuple(
            dict.fromkeys(
                (
                    *self._patterns,
                    *(value.encode("utf-8") for value in _secret_values.get() if value),
                )
            )
        )
        if not self._patterns:
            return data
        combined = self._tail + data
        if any(value in combined for value in self._patterns):
            self._refused = True
            self._tail = b""
            self._patterns = ()
            raise AssetError("Secret-bearing content cannot be saved")
        tail_size = max(len(value) - 1 for value in self._patterns)
        split = max(0, len(combined) - tail_size)
        self._tail = combined[split:]
        return combined[:split]

    def finish(self) -> bytes:
        if self._refused:
            raise AssetError("Secret-bearing content cannot be saved")
        tail, self._tail = self._tail, b""
        self._patterns = ()
        return tail
