from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENT_PATTERN = re.compile(r"dinkster-inference==([0-9]+\.[0-9]+\.[0-9]+)")


def pinned_requirement(repo_root: Path = REPO_ROOT) -> tuple[str, str]:
    root_project = repo_root / "pyproject.toml"
    document = tomllib.loads(root_project.read_text())
    extra = document.get("project", {}).get("optional-dependencies", {}).get("inference", [])
    if len(extra) != 1 or (match := REQUIREMENT_PATTERN.fullmatch(extra[0])) is None:
        raise RuntimeError(
            "dinkster[inference] has no exact dinkster-inference release requirement"
        )
    pins: list[tuple[Path, str]] = []
    paths = {
        root_project,
        *repo_root.glob("packages/*/pyproject.toml"),
        *repo_root.glob("packages/**/*pack.toml"),
    }
    for path in sorted(paths):
        data = tomllib.loads(path.read_text())
        project = data.get("project", {})
        pack = data.get("pack", {})
        uv = data.get("tool", {}).get("uv", {})
        groups = [
            project.get("dependencies", []),
            *project.get("optional-dependencies", {}).values(),
            *data.get("dependency-groups", {}).values(),
            pack.get("requires", []),
            *pack.get("extra-requires", {}).values(),
            uv.get("constraint-dependencies", []),
            uv.get("override-dependencies", []),
        ]
        for group in groups:
            for text in group:
                if not isinstance(text, str):
                    continue
                requirement = Requirement(text)
                if canonicalize_name(requirement.name) == "dinkster-inference" and (
                    requirement.specifier or requirement.url
                ):
                    pins.append((path, text))
    if pins != [(root_project, match.group(0))]:
        locations = ", ".join(str(path.relative_to(repo_root)) for path, _ in pins)
        raise RuntimeError(
            f"inference must have one authored pin in dinkster[inference]: {locations}"
        )
    return match.group(0), match.group(1)


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
