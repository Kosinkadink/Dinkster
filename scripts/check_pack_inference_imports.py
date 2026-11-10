"""Check first-party pack coupling declarations without importing node code."""

from __future__ import annotations

import argparse
import ast
import json
import tomllib
from importlib.metadata import version
from pathlib import Path


def inference_imports(source: Path) -> set[str]:
    """Collect direct and literal dynamic imports from the pack's owned source."""
    imports: set[str] = set()
    for path in source.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        dynamic_functions = {"__import__"}
        dynamic_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dynamic_modules.update(
                    alias.asname or alias.name for alias in node.names if alias.name == "importlib"
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
                dynamic_functions.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                if node.module == "dinkster_inference":
                    modules.extend(f"dinkster_inference.{alias.name}" for alias in node.names)
                elif node.module is not None:
                    modules.append(node.module)
            elif isinstance(node, ast.Call) and node.args:
                function = node.func
                dynamic = (isinstance(function, ast.Name) and function.id in dynamic_functions) or (
                    isinstance(function, ast.Attribute)
                    and function.attr == "import_module"
                    and isinstance(function.value, ast.Name)
                    and function.value.id in dynamic_modules
                )
                argument = node.args[0]
                if (
                    dynamic
                    and isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                ):
                    modules.append(argument.value)
        for module in modules:
            if module == "dinkster_inference":
                imports.add("__init__")
            elif module.startswith("dinkster_inference."):
                imports.add(module.removeprefix("dinkster_inference."))
    return imports


def check_pack(pack: Path, tested_version: str) -> None:
    """Reject missing, stale or unmeasured inference coupling declarations."""
    manifest = tomllib.loads((pack / "dinkster-pack.toml").read_text(encoding="utf-8"))
    declaration = manifest["pack"].get("inference", {})
    if declaration.get("tested-version") != tested_version:
        raise ValueError(f"{pack.name}: inference tested-version must be {tested_version}")
    measured = sorted(inference_imports(pack / "src"))
    if declaration.get("imports") != measured:
        raise ValueError(f"{pack.name}: inference imports must be {measured!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    table = args.root / "packages/dinkster-schema/src/dinkster_schema/comfy_node_ids.json"
    packs = {row["pack"] for row in json.loads(table.read_text(encoding="utf-8"))["nodes"]}
    tested_version = version("dinkster-inference")
    for manifest in sorted((args.root / "packages").glob("dinkster-nodes-*/dinkster-pack.toml")):
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
        if data["pack"]["name"] in packs:
            check_pack(manifest.parent, tested_version)


if __name__ == "__main__":
    main()
