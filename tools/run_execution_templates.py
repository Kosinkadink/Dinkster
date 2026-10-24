"""Serve native control nodes and capture their production registry contract.

This smoke harness intentionally uses the production ServingComposer path.
It keeps the template documents separate from the catalog proof: missing
heavy template implementations must be represented by an explicit test pack,
never by changing a source workflow or aliasing a production node type.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from aiohttp import ClientSession, web
from dinkster_server import create_app

from dinkster.compose import PackSpec, compose_serving

CONTROL_TYPES = frozenset(
    {"ComfySwitchNode", "StartLoop", "EndLoop", "LoopIteration", "LoopProgress", "LoopResult"}
)
PACKS = (
    "packages/dinkster-nodes-foundation/dinkster-pack.toml",
    "packages/dinkster-nodes-dev/dinkster-pack.toml",
    "packages/dinkster-nodes-comfy/dinkster-pack.toml",
)


async def _catalog(catalog_path: Path) -> dict[str, object]:
    specs = tuple(PackSpec(Path(path), trust_reserved=True, in_process=True) for path in PACKS)
    composition = await compose_serving(specs, include_default_packs=False)
    app = create_app(
        composition.make_engine,
        composition.schemas,
        packs=composition.packs,
        node_packs=composition.node_packs,
        choices=composition.choices,
        lazy_choices=composition.lazy_choices,
        schema_owners=composition.schema_owners,
        choice_owners=composition.choice_owners,
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        socket = next(iter(site._server.sockets))  # type: ignore[union-attr]
        port = socket.getsockname()[1]
        async with ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/api/nodes?wire=40") as response:
                response.raise_for_status()
                payload = await response.json()
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
        return payload
    finally:
        await runner.cleanup()
        await composition.close()


def _six_reference_rows(reference: dict[str, object]) -> dict[str, object]:
    rows = reference["nodes"]
    assert isinstance(rows, list)
    selected = [
        row for row in rows if isinstance(row, dict) and row.get("class_name") in CONTROL_TYPES
    ]
    if {row["class_name"] for row in selected} != CONTROL_TYPES:
        raise RuntimeError("reference dump does not contain all six native control rows")
    return {"source_commit": reference["source_commit"], "nodes": selected}


async def _run(
    catalog_path: Path,
    reference_path: Path,
    selected_reference_path: Path,
    selected_catalog_path: Path,
) -> None:
    catalog = await _catalog(catalog_path)
    nodes = catalog.get("nodes")
    if not isinstance(nodes, dict):
        raise RuntimeError("/api/nodes did not return a node map")
    missing = CONTROL_TYPES - nodes.keys()
    if missing:
        raise RuntimeError(f"served catalog omitted native controls: {sorted(missing)}")
    packs = catalog.get("packs")
    if not isinstance(packs, dict) or "dinkster-nodes-comfy" not in packs:
        raise RuntimeError("served catalog omitted dinkster-nodes-comfy pack provenance")
    selected_reference_path.write_text(
        json.dumps(
            _six_reference_rows(json.loads(reference_path.read_text())), indent=2, ensure_ascii=True
        )
        + "\n"
    )
    selected_catalog = dict(catalog)
    selected_catalog["nodes"] = {name: nodes[name] for name in sorted(CONTROL_TYPES)}
    selected_catalog_path.write_text(
        json.dumps(selected_catalog, indent=2, ensure_ascii=True) + "\n"
    )
    print(f"catalog={catalog_path}")
    print("controls=" + ",".join(sorted(CONTROL_TYPES)))
    print("pack=dinkster-nodes-comfy")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--selected-reference", type=Path, required=True)
    parser.add_argument("--selected-catalog", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(_run(args.catalog, args.reference, args.selected_reference, args.selected_catalog))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
