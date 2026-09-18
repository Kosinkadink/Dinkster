import sys
from typing import Any

from dinkster_native import resident as _implementation
from dinkster_native.resident import *  # pyright: ignore[reportWildcardImportFromLibrary]  # noqa: F403


def __getattr__(name: str) -> Any:
    return getattr(_implementation, name)


sys.modules[__name__] = _implementation
