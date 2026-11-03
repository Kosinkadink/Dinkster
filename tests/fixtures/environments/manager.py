"""Installed fixture engine's empty registry and catalog command contract."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root")
parser.add_argument("command")
parser.add_argument("--catalog-lock", type=Path)
parser.add_argument("--artifacts", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument("--installed-lock", type=Path)
parser.add_argument("--generation", type=Path)
parser.add_argument("--variant")
parser.add_argument("--base-id")
parser.add_argument("--allow-missing", action="store_true")
parser.add_argument("--retain-sources", action="store_true")
parser.add_argument("--registry-only", action="store_true")
args = parser.parse_args()
problem = os.environ.get("DINKSTER_FIXTURE_REGISTRY_PROBLEM")
if problem == "refuse":
    parser.error("fixture-pack@1.0 declares >=9; target engine is excluded")
if args.command == "prepare-registry-sources":
    if args.installed_lock:
        assert json.loads(args.installed_lock.read_text())["packs"] == []
    args.catalog_lock.write_text('{"format":"dinkster.lock/1","packs":[]}\n')
    args.artifacts.mkdir()
    print(
        json.dumps(
            {
                "engineVersion": "wrong"
                if problem == "version"
                else importlib.metadata.version("fixture-engine"),
                "catalogLock": str(args.catalog_lock),
                "artifacts": str(args.artifacts),
                "packs": [],
                "unavailable": {} if args.installed_lock else {"fixture-pack": "registry offline"},
            }
        )
    )
elif args.command == "prepare-registry-locks":
    args.output.mkdir()
    print(json.dumps({"locks": [str(args.catalog_lock)] if problem == "path" else []}))
elif args.command != "prepare-catalogs":
    parser.error("unsupported fixture command")
