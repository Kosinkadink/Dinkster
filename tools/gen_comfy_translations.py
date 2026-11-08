"""Generate the offline ComfyUI import index from the moved packs' declarations.

Run ``python -m tools.gen_comfy_translations`` in the workspace environment;
``--check`` verifies the committed snapshot without changing it.
"""

from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path

from dinkster_schema import Node, NodeSchema
from dinkster_schema.first_party import collect_comfy_translations
from dinkster_workers import load_manifest
from dinkster_workers.manifest import resolve_entry

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src/dinkster/comfy-translations.json"


def generate(root: Path = ROOT) -> bytes:
    scope = json.loads((root / "tools/node_comparison_scope.json").read_text())
    selected = set(scope["migratedPacks"])
    identities = json.loads(files("dinkster_schema").joinpath("comfy_node_ids.json").read_text())
    expected = {row["from"] for row in identities["nodes"] if row["pack"] in selected}
    packs: list[tuple[str, str, str, list[NodeSchema]]] = []
    found: set[str] = set()
    for path in sorted((root / "packages").glob("dinkster-nodes-*/dinkster-pack.toml")):
        manifest = load_manifest(path)
        if manifest.name not in selected:
            continue
        if manifest.name in found:
            raise ValueError(f"duplicate pack manifest: {manifest.name}")
        found.add(manifest.name)
        if manifest.compatibility is None:
            raise ValueError(f"pack version missing: {manifest.name}")
        nodes = resolve_entry(manifest.nodes_entry)
        if not isinstance(nodes, (list, tuple)) or not all(
            isinstance(node, type) and issubclass(node, Node) for node in nodes
        ):
            raise ValueError(f"node entry must be a sequence of Node classes: {manifest.name}")
        packs.append(
            (
                manifest.publisher,
                manifest.name,
                manifest.compatibility.version,
                [node.schema() for node in nodes],
            )
        )
    if found != selected:
        raise ValueError(f"missing pack manifests: {sorted(selected - found)}")
    index = collect_comfy_translations(packs)
    # Round-trip JSON is also the boundary consumed by the offline client.
    data = json.dumps(index, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    actual = {item["rule"]["from"] for item in json.loads(data)["items"]}
    if actual != expected:
        raise ValueError(
            f"translation coverage: missing {sorted(expected - actual)}, "
            f"extra {sorted(actual - expected)}"
        )
    return data.encode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = generate()
    if args.check:
        if OUTPUT.read_bytes() != data:
            raise SystemExit("offline ComfyUI translations are stale; regenerate the snapshot")
    else:
        OUTPUT.write_bytes(data)


if __name__ == "__main__":
    main()
