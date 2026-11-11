"""Build explicitly selected development sources with the generation installer."""

from __future__ import annotations

from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from dinkster_supervisor.environment import argument_parser, run
    from dinkster_supervisor.environment_development import DEVELOPMENT_DESCRIPTION

    parser = argument_parser("develop")
    parser.prog = "dinkster develop"
    parser.description = DEVELOPMENT_DESCRIPTION
    parser.allow_abbrev = False
    parser.add_argument("--root", type=Path, required=True, help="development project root")
    parser.set_defaults(operation="develop")
    args = parser.parse_args(argv)
    args.root = args.root.expanduser().resolve()
    return run(args)
