"""Target-owned workflow adapter for testing lifecycle source and journal boundaries."""

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from .project_requirements import load_project_requirements

parser = argparse.ArgumentParser()
parser.add_argument("operation")
parser.add_argument("--project-root", type=Path)
parser.add_argument("--source-generation", type=Path)
parser.add_argument("--generation", type=Path)
parser.add_argument("--catalog-lock", type=Path)
parser.add_argument("--artifacts", type=Path)
parser.add_argument("--variant")
parser.add_argument("--json", action="store_true")
parser.add_argument("--yes", action="store_true")
args = parser.parse_args()
assert sys.flags.isolated and sys.dont_write_bytecode
assert importlib.metadata.version("fixture-engine") == "2.0", "planner must use target engine"
assert args.source_generation.is_dir()
assert json.loads(args.catalog_lock.read_bytes())["packs"] == []
assert args.artifacts.is_dir()
requirements, inputs = load_project_requirements(args.project_root)
errors = []
problem = os.environ.get("DINKSTER_FIXTURE_WORKFLOW_PROBLEM")
if problem == "refuse":
    errors.append("saved pack version is unavailable in selected registry catalog")
elif problem == "revision":
    inputs = ["changed-workflow-revision"]
elif problem == "catalog":
    args.catalog_lock.write_text(args.catalog_lock.read_text() + " ")
document = {
    "requirements": requirements.to_wire(),
    "inputs": list(inputs),
    "plan": {"packs": [], "assets": [], "errors": errors},
}
if args.operation == "project-apply":
    assert args.yes and (args.generation / "generation.json").is_file()
    assert json.load(sys.stdin) == document
print(json.dumps(document))
sys.exit(bool(errors))
