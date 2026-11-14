"""Standardized logging: named origins, host-owned verbosity.

The contract under test (DESIGN 3.6 adjacent - operational, not wire):

- Origin is the logger NAME: core subsystems are ``dinkster.<subsystem>``,
  packs are ``dinkster.pack.<pack_name>`` - stated explicitly, never guessed.
- ``configure_logging`` is host policy: one formatted stderr handler on the
  ``dinkster`` tree, per-origin overrides, idempotent reconfiguration. Packs
  get ``pack_logger`` through ``dinkster_api.v1``; configuration is
  deliberately NOT exported through the door.
- The environment (``DINKSTER_LOG_LEVEL`` / ``DINKSTER_LOG``) is how verbosity
  crosses process boundaries: worker children inherit it, so one host
  setting reaches every pack process with origins intact.
"""

from __future__ import annotations

import io
import logging

import pytest
from dinkster_schema import (
    LOG_LEVEL_ENV,
    LOG_OVERRIDES_ENV,
    configure_logging,
    configure_logging_from_env,
    core_logger,
    pack_logger,
)
from dinkster_schema.log import ROOT_LOGGER_NAME, _overridden_loggers


@pytest.fixture(autouse=True)
def _reset_dinkster_logging():
    """Restore the dinkster logger tree after each test."""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    saved = (list(root.handlers), root.level, root.propagate)
    saved_overrides = _overridden_loggers.copy()
    manager = logging.Logger.manager
    saved_children = {
        name: logging.getLogger(name).level
        for name in list(manager.loggerDict)
        if name.startswith(ROOT_LOGGER_NAME + ".")
        and isinstance(manager.loggerDict[name], logging.Logger)
    }
    yield
    root.handlers[:], root.level, root.propagate = saved[0], saved[1], saved[2]
    for name, logger in list(manager.loggerDict.items()):
        if name.startswith(ROOT_LOGGER_NAME + ".") and isinstance(logger, logging.Logger):
            logger.setLevel(saved_children.get(name, logging.NOTSET))
    _overridden_loggers.clear()
    _overridden_loggers.update(saved_overrides)


# -- names are origins ------------------------------------------------------


# -- host configuration ------------------------------------------------------


def test_output_carries_level_origin_and_message() -> None:
    stream = io.StringIO()
    configure_logging("info", stream=stream)
    pack_logger("mypack").warning("model %s missing", "sdxl")
    line = stream.getvalue()
    assert "WARNING" in line
    assert "dinkster.pack.mypack" in line
    assert "model sdxl missing" in line


def test_reconfiguration_replaces_instead_of_stacking() -> None:
    first, second = io.StringIO(), io.StringIO()
    configure_logging("debug", overrides={"dinkster.pack.a": "error"}, stream=first)
    configure_logging("debug", stream=second)
    pack_logger("a").info("after reset")
    # One handler, the new stream, and the stale override is gone.
    assert "after reset" not in first.getvalue()
    assert second.getvalue().count("after reset") == 1


# -- environment (the cross-process channel) ---------------------------------


def test_env_configuration_level_and_overrides(capsys) -> None:
    configure_logging_from_env(
        {
            LOG_LEVEL_ENV: "warning",
            LOG_OVERRIDES_ENV: "dinkster.pack.mypack=debug, dinkster.engine=error",
        }
    )
    pack_logger("mypack").debug("pack debug")
    core_logger("engine").warning("engine warn")
    core_logger("server").warning("server warn")
    err = capsys.readouterr().err
    assert "pack debug" in err
    assert "engine warn" not in err
    assert "server warn" in err


# -- the extension door ------------------------------------------------------
