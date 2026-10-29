from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENT_PATTERN = re.compile(r"dinkster-inference==([0-9]+\.[0-9]+\.[0-9]+)")


def pinned_requirement(repo_root: Path = REPO_ROOT) -> tuple[str, str]:
    project = tomllib.loads(
        repo_root.joinpath("packages/dinkster-native/pyproject.toml").read_text()
    )
    for requirement in project["project"]["dependencies"]:
        match = REQUIREMENT_PATTERN.fullmatch(requirement)
        if match is not None:
            return match.group(0), match.group(1)
    raise RuntimeError("dinkster-native has no exact dinkster-inference release requirement")


def _module_tree(package_root: Path, module: str) -> ast.Module:
    return ast.parse(package_root.joinpath(*module.split(".")).with_suffix(".py").read_text())


def _definition(
    tree: ast.Module,
    name: str,
    kind: type[ast.AST] | tuple[type[ast.AST], ...],
) -> ast.AST:
    for node in tree.body:
        if isinstance(node, kind) and getattr(node, "name", None) == name:
            return node
    raise RuntimeError(f"installed inference is missing {name}")


def verify_source_contracts(package_root: Path) -> None:
    sd = _module_tree(package_root, "sd")
    for name in ("load_checkpoint_guess_config", "load_diffusion_model", "load_clip"):
        _definition(sd, name, ast.FunctionDef)
    _definition(sd, "VAE", ast.ClassDef)
    sample = _module_tree(package_root, "sample")
    for name in ("sample", "prepare_noise", "fix_empty_latent_channels"):
        _definition(sample, name, ast.FunctionDef)


def verify_installed(pin: str) -> None:
    distribution = importlib.metadata.distribution("dinkster-inference")
    if distribution.version != pin:
        raise RuntimeError(
            f"installed inference version {distribution.version!r} does not match pin {pin}"
        )

    package_root = Path(distribution.locate_file("dinkster_inference"))
    verify_source_contracts(package_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-requirement", action="store_true")
    args = parser.parse_args()
    requirement, pin = pinned_requirement()
    if args.print_requirement:
        print(requirement)
        return 0
    verify_installed(pin)
    print(f"verified installed dinkster-inference {pin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
