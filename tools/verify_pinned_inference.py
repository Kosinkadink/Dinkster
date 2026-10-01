from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENT_PATTERN = re.compile(
    r"dinkster-inference @ git\+https://github\.com/Kosinkadink/"
    r"dinkster-inference\.git@([0-9a-f]{40})"
)


def pinned_requirement(repo_root: Path = REPO_ROOT) -> tuple[str, str]:
    requirements = []
    for relative in ("scripts/setup_envs.sh", "scripts/setup_envs.ps1"):
        match = REQUIREMENT_PATTERN.search(repo_root.joinpath(relative).read_text())
        if match is None:
            raise RuntimeError(f"{relative} has no exact dinkster-inference requirement")
        requirements.append((match.group(0), match.group(1)))
    if requirements[0] != requirements[1]:
        raise RuntimeError("setup_envs.sh and setup_envs.ps1 pin different inference commits")
    return requirements[0]


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


def _require_strings(node: ast.AST, *values: str) -> None:
    strings = {child.value for child in ast.walk(node) if isinstance(child, ast.Constant)}
    missing = set(values) - strings
    if missing:
        raise RuntimeError(f"installed inference is missing contract strings: {sorted(missing)}")


def verify_source_contracts(package_root: Path) -> None:
    window_plan = _module_tree(package_root, "window_plan")
    for name in (
        "IntegerAffineIndexMap",
        "KindAxisMap",
        "LayerWindow",
        "MediaAxis",
        "MergeDeclaration",
        "ProportionalRangeIndexMap",
        "WindowIndexList",
        "WindowKind",
        "WindowPlanLayer",
        "WindowWeightKind",
        "WindowWeightProfile",
        "compile_window_plan",
    ):
        _definition(window_plan, name, (ast.ClassDef, ast.FunctionDef))

    window_execution = _module_tree(package_root, "window_execution")
    _definition(window_execution, "WindowTensorLayout", ast.ClassDef)
    executor = _definition(window_execution, "WindowPlanExecutor", ast.ClassDef)
    methods = {
        child.name for child in cast_class(executor).body if isinstance(child, ast.FunctionDef)
    }
    missing_methods = {"evaluate_window", "window_latent"} - methods
    if missing_methods:
        raise RuntimeError(
            f"installed inference window executor is missing methods: {sorted(missing_methods)}"
        )

    sd = _module_tree(package_root, "sd")
    _definition(sd, "load_diffusion_model", ast.FunctionDef)
    _require_strings(
        _definition(sd, "load_diffusion_model_state_dict", ast.FunctionDef),
        "assign_loaded_weights",
        "gguf_residency",
    )
    _require_strings(_definition(sd, "load_clip", ast.FunctionDef), "gguf_residency")

    gguf = _module_tree(package_root, "gguf")
    for name, kind in (
        ("GGUFBalancedResidency", ast.ClassDef),
        ("apply_gguf_residency", ast.FunctionDef),
        ("load_gguf_state_dict", ast.FunctionDef),
    ):
        _definition(gguf, name, kind)
    _require_strings(gguf, "memory", "balanced", "eager")
    _definition(_module_tree(package_root, "gguf_ops"), "GGUFOps", ast.ClassDef)


def cast_class(node: ast.AST) -> ast.ClassDef:
    if not isinstance(node, ast.ClassDef):
        raise TypeError("expected class definition")
    return node


def verify_installed(pin: str) -> None:
    distribution = importlib.metadata.distribution("dinkster-inference")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    installed = direct_url.get("vcs_info", {}).get("commit_id")
    if installed != pin:
        raise RuntimeError(f"installed inference commit {installed!r} does not match pin {pin}")

    package_root = Path(distribution.locate_file("dinkster_inference"))
    verify_source_contracts(package_root)

    window_plan = importlib.import_module("dinkster_inference.window_plan")
    patcher_extension = importlib.import_module("dinkster_inference.patcher_extension")
    if window_plan.MediaAxis("width", 4).extent != 4:
        raise RuntimeError("installed inference window plan smoke failed")
    wrappers = patcher_extension.WrappersMP
    if wrappers.CALC_COND_BATCH != "calc_cond_batch" or wrappers.WINDOW_EXECUTE != "window_execute":
        raise RuntimeError("installed inference multi-GPU wrapper contract is incompatible")


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
