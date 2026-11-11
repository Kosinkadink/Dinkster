"""Build explicitly selected development sources with the generation installer."""

from __future__ import annotations

import importlib
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    environment = importlib.import_module("dinkster_supervisor.environment")
    development = importlib.import_module("dinkster_supervisor.environment_development")

    parser = environment.argument_parser("develop")
    parser.prog = "dinkster develop"
    parser.description = development.DEVELOPMENT_DESCRIPTION
    parser.allow_abbrev = False
    parser.add_argument("--root", type=Path, required=True, help="development project root")
    parser.set_defaults(operation="develop")
    args = parser.parse_args(argv)
    args.root = args.root.expanduser().resolve()
    return environment.run(args)
