"""Torch-free process bootstrap for dinkster-aimdo residency."""

from __future__ import annotations

import importlib
import logging
import sys

log = logging.getLogger("dinkster.workers.aimdo")


def bootstrap_aimdo(
    enabled: bool,
    *,
    simple_vram_headroom: int | None = None,
) -> tuple[bool, int | None]:
    """Initialize dinkster-aimdo before torch and return success plus applied headroom."""
    effective_headroom = simple_vram_headroom
    if not enabled:
        return False, effective_headroom
    if "torch" in sys.modules:
        log.warning(
            "aimdo bootstrap was requested too late because torch is already "
            "imported; process continues without successful aimdo bootstrap"
        )
        return False, effective_headroom
    try:
        control = importlib.import_module("dinkster_aimdo.control")
    except ImportError:
        log.warning(
            "dinkster_aimdo.control is unavailable; process continues without "
            "successful aimdo bootstrap"
        )
        return False, effective_headroom
    try:
        if simple_vram_headroom is None:
            initialized = control.init()  # type: ignore[attr-defined]
        else:
            try:
                initialized = control.init(  # type: ignore[attr-defined]
                    simple_vram_headroom=simple_vram_headroom
                )
            except TypeError as exc:
                if "simple_vram_headroom" not in str(exc):
                    raise
                log.warning(
                    "dinkster_aimdo.control.init() does not accept "
                    "simple_vram_headroom (dinkster-aimdo older than 0.4.10 "
                    "installed); requested %d bytes of VRAM headroom are NOT "
                    "applied and runtime headroom is disarmed for this "
                    "process - upgrade dinkster-aimdo to restore reservation "
                    "budgets; retrying init() without the argument",
                    simple_vram_headroom,
                )
                effective_headroom = None
                initialized = control.init()  # type: ignore[attr-defined]
    except Exception:
        log.warning(
            "dinkster_aimdo.control.init() raised; process continues without "
            "successful aimdo bootstrap",
            exc_info=True,
        )
        return False, effective_headroom
    if initialized is not True:
        log.warning(
            "dinkster_aimdo.control.init() returned %r; process continues without "
            "successful aimdo bootstrap",
            initialized,
        )
        return False, effective_headroom
    if effective_headroom is None:
        log.info("dinkster_aimdo.control.init() completed successfully")
    else:
        log.info(
            "dinkster_aimdo.control.init() completed successfully with "
            "simple_vram_headroom=%d bytes",
            effective_headroom,
        )
    return True, effective_headroom


__all__ = ["bootstrap_aimdo"]
