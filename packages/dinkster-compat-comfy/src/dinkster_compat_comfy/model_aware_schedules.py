import sys
from typing import Any

from dinkster_native import model_aware_schedules as _implementation
from dinkster_native.model_aware_schedules import *  # pyright: ignore[reportWildcardImportFromLibrary]  # noqa: F403


def __getattr__(name: str) -> Any:
    return getattr(_implementation, name)


sys.modules[__name__] = _implementation
