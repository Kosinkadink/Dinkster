"""Convert one legacy torch checkpoint with Dinkster's canonical safe converter."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dinkster_compat_comfy.legacy_sources import convert_legacy_checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely convert a legacy .ckpt/.pt/.pth file to safetensors."
    )
    parser.add_argument("source", type=Path, help="Legacy checkpoint to convert.")
    parser.add_argument("output", type=Path, help="Safetensors output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = convert_legacy_checkpoint(args.source, args.output)
    print(
        f"converted {args.source} -> {args.output} "
        f"(dropped_keys={len(result.dropped_keys)}, "
        f"output_bytes={result.output_bytes}, "
        f"output_sha256={result.output_sha256})"
    )


if __name__ == "__main__":
    main()
