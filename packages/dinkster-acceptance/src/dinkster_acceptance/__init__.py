from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def manifest_path() -> str:
    try:
        packaged = files("dinkster_acceptance_pack").joinpath("dinkster-pack.toml")
    except ModuleNotFoundError:
        packaged = Path(__file__).resolve().parents[2] / "dinkster-pack.toml"
    return str(packaged)


def print_manifest_path() -> None:
    print(manifest_path())


__all__ = ["manifest_path"]
