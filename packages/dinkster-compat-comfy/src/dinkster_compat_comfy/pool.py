import sys
from typing import Any

from dinkster_native import pool as _implementation
from dinkster_native.pool import *  # pyright: ignore[reportWildcardImportFromLibrary]  # noqa: F403

from .comfy_execution import model_unload

_implementation.configure_compat_unload(model_unload)


def __getattr__(name: str) -> Any:
    return getattr(_implementation, name)


sys.modules[__name__] = _implementation
