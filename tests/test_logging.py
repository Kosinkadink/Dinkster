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
from dinkster_schema.log import ROOT_LOGGER_NAME


@pytest.fixture(autouse=True)
def _reset_dinkster_logging():
    """Restore the dinkster logger tree after each test."""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    saved = (list(root.handlers), root.level, root.propagate)
    manager = logging.Logger.manager
    saved_children = {
        name: logging.getLogger(name).level
        for name in list(manager.loggerDict)
        if name.startswith(ROOT_LOGGER_NAME + ".")
        and isinstance(manager.loggerDict[name], logging.Logger)
    }
    yield
    root.handlers[:], root.level, root.propagate = saved[0], saved[1], saved[2]
    for name, level in saved_children.items():
        logging.getLogger(name).setLevel(level)


# -- names are origins ------------------------------------------------------


def test_core_and_pack_logger_names() -> None:
    assert core_logger("server").name == "dinkster.server"
    assert core_logger("server.queue").name == "dinkster.server.queue"
    assert pack_logger("mypack").name == "dinkster.pack.mypack"


def test_loggers_are_plain_stdlib_loggers() -> None:
    assert type(pack_logger("mypack")) is logging.getLogger("x").__class__
    assert pack_logger("mypack") is logging.getLogger("dinkster.pack.mypack")


@pytest.mark.parametrize("bad", ["", " ", "a b", ".x", "x.", "a..b", " x"])
def test_malformed_names_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        core_logger(bad)
    with pytest.raises(ValueError):
        pack_logger(bad)


def test_core_logger_cannot_squat_the_pack_namespace() -> None:
    with pytest.raises(ValueError, match="reserved for packs"):
        core_logger("pack")
    with pytest.raises(ValueError, match="reserved for packs"):
        core_logger("pack.sneaky")


# -- host configuration ------------------------------------------------------


def test_output_carries_level_origin_and_message() -> None:
    stream = io.StringIO()
    configure_logging("info", stream=stream)
    pack_logger("mypack").warning("model %s missing", "sdxl")
    line = stream.getvalue()
    assert "WARNING" in line
    assert "dinkster.pack.mypack" in line
    assert "model sdxl missing" in line


def test_default_level_filters_debug() -> None:
    stream = io.StringIO()
    configure_logging("info", stream=stream)
    core_logger("engine").debug("chatty")
    core_logger("engine").info("kept")
    assert "chatty" not in stream.getvalue()
    assert "kept" in stream.getvalue()


def test_per_origin_override_is_independent() -> None:
    stream = io.StringIO()
    configure_logging(
        "warning",
        overrides={"dinkster.pack.verbose": "debug"},
        stream=stream,
    )
    pack_logger("verbose").debug("wanted")
    pack_logger("other").info("unwanted")
    output = stream.getvalue()
    assert "wanted" in output
    assert "unwanted" not in output


def test_reconfiguration_replaces_instead_of_stacking() -> None:
    first, second = io.StringIO(), io.StringIO()
    configure_logging("debug", overrides={"dinkster.pack.a": "error"}, stream=first)
    configure_logging("debug", stream=second)
    pack_logger("a").info("after reset")
    # One handler, the new stream, and the stale override is gone.
    assert "after reset" not in first.getvalue()
    assert second.getvalue().count("after reset") == 1


def test_foreign_handlers_on_dinkster_are_left_alone() -> None:
    root = logging.getLogger(ROOT_LOGGER_NAME)
    foreign = logging.NullHandler()
    root.addHandler(foreign)
    configure_logging("info", stream=io.StringIO())
    configure_logging("info", stream=io.StringIO())
    assert foreign in root.handlers
    root.removeHandler(foreign)


def test_records_do_not_propagate_to_the_root_logger() -> None:
    configure_logging("info", stream=io.StringIO())
    assert logging.getLogger(ROOT_LOGGER_NAME).propagate is False


def test_unknown_level_and_foreign_override_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging("loud")
    with pytest.raises(ValueError, match="outside the 'dinkster' tree"):
        configure_logging("info", overrides={"urllib3": "debug"})


def test_bad_override_leaves_configuration_untouched() -> None:
    stream = io.StringIO()
    configure_logging("info", stream=stream)
    with pytest.raises(ValueError):
        configure_logging("info", overrides={"not-dinkster": "debug"})
    core_logger("engine").info("still routed")
    assert "still routed" in stream.getvalue()


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


def test_env_absent_means_defaults(capsys) -> None:
    configure_logging_from_env({})
    core_logger("engine").info("default info")
    assert "default info" in capsys.readouterr().err


@pytest.mark.parametrize("raw", ["nolevel", "=debug", "name=", "a=b=c,"])
def test_env_malformed_entries_raise(raw: str) -> None:
    if raw == "a=b=c,":
        # partition keeps the rest in the level -> unknown level.
        with pytest.raises(ValueError):
            configure_logging_from_env({LOG_OVERRIDES_ENV: raw})
    else:
        with pytest.raises(ValueError, match="malformed|unknown"):
            configure_logging_from_env({LOG_OVERRIDES_ENV: raw})


# -- the extension door ------------------------------------------------------


def test_pack_api_exposes_logging_but_not_configuration() -> None:
    import dinkster_api.v1 as api

    assert api.pack_logger is pack_logger
    assert not hasattr(api, "configure_logging")
    assert not hasattr(api, "configure_logging_from_env")
    assert not hasattr(api, "core_logger")
