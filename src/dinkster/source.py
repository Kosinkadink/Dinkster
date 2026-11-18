"""Select explicit Git engines, packs, libraries or frontend assets through bundled Git."""

from __future__ import annotations

import importlib
from pathlib import Path

from .setup import default_project


def main(argv: list[str] | None = None) -> int:
    environment = importlib.import_module("dinkster_supervisor.environment")
    parser = environment.argument_parser("source")
    parser.prog = "dinkster source"
    parser.allow_abbrev = False
    parser.add_argument(
        "--root", type=Path, default=default_project(), help="managed environment root"
    )
    parser.set_defaults(operation="source")
    args = parser.parse_args(argv)
    args.root = args.root.expanduser().resolve()
    return environment.run(args)
