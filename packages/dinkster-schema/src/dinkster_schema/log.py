"""Standardized logging: one namespace, stdlib all the way down.

There is no Dinkster logger class. Core code and packs both use plain
``logging.Logger`` objects; the convention is the NAME, and the name is the
origin:

- core subsystems log under ``dinkster.<subsystem>`` (``core_logger("server")``)
- packs log under ``dinkster.pack.<pack_name>`` (``pack_logger("mypack")``)

Because origin is the logger name, everything the stdlib already does works
unchanged: per-origin verbosity (silence one noisy pack, debug another),
``%(name)s`` in any format string, filters, and third-party handlers. A pack
names itself explicitly - the same rule as pack-defined event names and
node_type prefixes: identity is declared, never guessed from stack frames.

``configure_logging()`` is HOST policy (the CLI, the worker service, a test
harness): it installs exactly one formatted stderr handler on the ``dinkster``
logger and sets ``propagate = False`` so records never double-print through
the root logger. Packs must never call it - a library that installs handlers
hijacks the host's output. It is deliberately not exported through
``dinkster_api.v1``; ``pack_logger`` is.

Cross-process: worker subprocesses inherit the host's stderr, so a worker
service that calls ``configure_logging()`` (it does, from the
``DINKSTER_LOG_LEVEL`` / ``DINKSTER_LOG`` environment the host passes through)
lands pack log lines on the host console with their origin name intact -
no forwarding machinery, and no way for a pack's logging to be mistaken
for core's.

Logging is chatter, like reported events: nothing correctness-critical may
live in a log line, and node authors should still prefer ``report_*`` for
anything a frontend renders (progress, previews) - logs are for humans and
operators, events are for machines.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from typing import TextIO

from .capture import forward_log_record
from .reporting import suppress_capture

ROOT_LOGGER_NAME = "dinkster"
PACK_LOGGER_PREFIX = "dinkster.pack"

LOG_LEVEL_ENV = "DINKSTER_LOG_LEVEL"
"""Default level for the whole ``dinkster`` tree, e.g. ``debug``."""

LOG_OVERRIDES_ENV = "DINKSTER_LOG"
"""Comma-separated per-origin overrides, e.g.
``dinkster.pack.mypack=debug,dinkster.server=warning``."""

_LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_DINKSTER_HANDLER_MARK = "_dinkster_standard_handler"
_overridden_loggers: set[str] = set()


class _StandardHandler(logging.StreamHandler):  # pyright: ignore[reportMissingTypeArgument]
    """The stderr handler plus execution-log forwarding: while a node
    executes, records on the ``dinkster`` tree also become attributed
    execution log events (see capture.py). The terminal write happens
    inside ``suppress_capture()`` so the stderr proxy never re-captures
    the formatted line this handler just wrote."""

    def emit(self, record: logging.LogRecord) -> None:
        with suppress_capture():
            super().emit(record)
        try:
            forward_log_record(record)
        except Exception:
            # Forwarding is chatter: a malformed record (bad %-args, a
            # broken __str__) must not raise into whatever logged it. The
            # terminal emit above already diagnosed the record via the
            # stdlib's own error handling; a second diagnostic here would
            # only add noise (and handleError itself may raise).
            pass


def _parse_level(level: str) -> int:
    resolved = _LEVELS.get(level.strip().lower())
    if resolved is None:
        raise ValueError(f"unknown log level: {level!r} (expected one of {sorted(_LEVELS)})")
    return resolved


def _validate_name(name: str, what: str) -> None:
    if not name or name != name.strip():
        raise ValueError(f"{what} must be a non-empty name: {name!r}")
    if any(ch.isspace() for ch in name):
        raise ValueError(f"{what} may not contain whitespace: {name!r}")
    if name.startswith(".") or name.endswith(".") or ".." in name:
        raise ValueError(f"{what} has empty dotted segments: {name!r}")


def core_logger(subsystem: str) -> logging.Logger:
    """The logger for a core subsystem: ``dinkster.<subsystem>``.

    Core code only. ``subsystem`` is the package's short name ("server",
    "engine", "workers", "assets"); dotted children ("server.queue") are
    fine when a subsystem wants finer origins.
    """
    _validate_name(subsystem, "subsystem")
    if subsystem == "pack" or subsystem.startswith("pack."):
        raise ValueError(f"'{PACK_LOGGER_PREFIX}' is reserved for packs; use pack_logger()")
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{subsystem}")


def pack_logger(pack_name: str) -> logging.Logger:
    """The logger for a pack: ``dinkster.pack.<pack_name>``.

    ``pack_name`` is the pack's manifest name, stated explicitly - the same
    convention as pack-defined event names. Get it once at module level:
    ``_log = pack_logger("mypack")``.
    """
    _validate_name(pack_name, "pack name")
    return logging.getLogger(f"{PACK_LOGGER_PREFIX}.{pack_name}")


def configure_logging(
    level: str = "info",
    *,
    overrides: Mapping[str, str] | None = None,
    stream: TextIO | None = None,
) -> None:
    """Install the standard handler on the ``dinkster`` tree. Host policy only.

    One formatted handler (time, level, origin name, message) goes on the
    ``dinkster`` logger; ``propagate`` is disabled so root-logger configuration
    elsewhere in the process never double-prints Dinkster lines. Idempotent:
    calling again replaces the previous configuration (level, overrides,
    stream) instead of stacking handlers.

    ``overrides`` maps logger names to levels for per-origin verbosity -
    ``{"dinkster.pack.mypack": "debug", "dinkster.server": "warning"}``. Names
    outside the ``dinkster`` tree are rejected: this function configures Dinkster,
    not the application's root logging. Handlers an embedding application
    installed on ``dinkster`` itself are left alone; only the handler this
    function previously installed is replaced.
    """
    # Validate everything before mutating any logging state, so a bad
    # override never leaves a half-applied configuration.
    parsed_level = _parse_level(level)
    parsed_overrides: dict[str, int] = {}
    for name, override_level in (overrides or {}).items():
        if name != ROOT_LOGGER_NAME and not name.startswith(ROOT_LOGGER_NAME + "."):
            raise ValueError(f"override outside the '{ROOT_LOGGER_NAME}' tree: {name!r}")
        parsed_overrides[name] = _parse_level(override_level)

    root = logging.getLogger(ROOT_LOGGER_NAME)
    handler = _StandardHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    setattr(handler, _DINKSTER_HANDLER_MARK, True)
    for existing in list(root.handlers):
        if getattr(existing, _DINKSTER_HANDLER_MARK, False):
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(parsed_level)
    root.propagate = False
    # Reset overrides from a previous configuration so reconfiguring
    # replaces rather than accumulates per-origin levels.
    for stale in _overridden_loggers - parsed_overrides.keys():
        if stale != ROOT_LOGGER_NAME:
            logging.getLogger(stale).setLevel(logging.NOTSET)
    _overridden_loggers.clear()
    for name, parsed in parsed_overrides.items():
        logging.getLogger(name).setLevel(parsed)
        _overridden_loggers.add(name)


def configure_logging_from_env(environ: Mapping[str, str]) -> None:
    """``configure_logging`` driven by ``DINKSTER_LOG_LEVEL`` / ``DINKSTER_LOG``.

    The worker service calls this at startup: subprocess launches inherit
    the host environment, so raising verbosity on the host CLI reaches
    every pack process without new plumbing. Absent variables mean the
    defaults; a malformed value raises rather than being silently ignored.
    """
    overrides: dict[str, str] = {}
    raw = environ.get(LOG_OVERRIDES_ENV, "")
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, override_level = entry.partition("=")
        if not sep or not name or not override_level:
            raise ValueError(
                f"malformed {LOG_OVERRIDES_ENV} entry: {entry!r} (expected name=level)"
            )
        overrides[name.strip()] = override_level.strip()
    configure_logging(
        environ.get(LOG_LEVEL_ENV, "info"),
        overrides=overrides,
    )
