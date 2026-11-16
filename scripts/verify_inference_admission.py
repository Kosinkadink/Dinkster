"""Check inference-admission fixtures through a managed server's public HTTP API.

Install dinkster-models plus tests/fixtures/inference-admission/{inspector,refused,
versionconflict} with the public develop command first. The checkpoint directory
must contain v1-5-pruned-emaonly-fp16.safetensors. No sampling parity is measured.
"""

import argparse
import json
import math
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = args.output.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    current = (root / "current").read_text().strip()
    generation = root / "generations" / current
    record = json.loads((generation / "inference-process.json").read_text())
    assert set(record["members"]) == {"dinkster-models", "inspector"}, record
    reasons = {name: item["reason"] for name, item in record["refusals"].items()}
    assert "pack.sandbox.network" in reasons["refused"]
    assert all(text in reasons["versionconflict"] for text in ("humanize", "4.12.3", "4.13.0"))
    runtime = json.loads((root / "runtime" / current / "main.json").read_text())
    base = runtime["url"]
    client = "admission-" + uuid.uuid4().hex

    def call(route, data=None):
        request = Request(
            base + route,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=120) as response:
                return response.status, json.load(response)
        except HTTPError as response:
            return response.code, json.load(response)

    status, mounts = call("/api/mounts")
    assert status == 200, (status, mounts)
    existing = next((row for row in mounts["mounts"] if row["id"] == "checkpoint-proof"), None)
    directory = str(args.checkpoint_directory.resolve())
    if existing is None:
        status, added = call("/api/mounts", {"id": "checkpoint-proof", "path": directory})
        assert status == 201, (status, added)
    else:
        assert existing["path"] == directory, existing
    checkpoint = None
    scan_started = time.monotonic()
    while time.monotonic() - scan_started < 120:
        status, entries = call("/api/mounts/checkpoint-proof/entries")
        assert status == 200, (status, entries)
        checkpoint = next(
            (
                item
                for item in entries["entries"]
                if item["name"] == "v1-5-pruned-emaonly-fp16.safetensors"
            ),
            None,
        )
        if checkpoint is not None:
            break
        time.sleep(1)
    assert checkpoint is not None, entries
    assert checkpoint["digest"] == (
        "blake3:4c50ebc6e2a5cb19e8d19626d5ede1fb64755562085ce7383d86c72d1d03eb7e"
    ), checkpoint
    status, catalog = call("/api/nodes")
    assert status == 200, (status, catalog)
    (evidence / "admission-public-catalog.json").write_text(json.dumps(catalog))
    for name in ("refused", "versionconflict"):
        assert catalog["nodes"][name + ".loaded_model"]["unavailableReason"] == reasons[name]
        assert "unavailableReason" not in catalog["nodes"][name + ".plain"]
        request = {
            "clientId": client,
            "jobId": name,
            "project": "default",
            "graph": {"nodes": {"bad": {"nodeType": name + ".loaded_model", "inputs": {}}}},
            "targets": ["bad"],
        }
        status, response = call("/api/jobs", request)
        assert status == 400, (name, status, response)
        assert response["message"] == reasons[name]
        status, _ = call(f"/api/jobs/{client}/{name}")
        assert status == 404
        print(name, "refused before queueing:", reasons[name], flush=True)
    nodes = {
        "plain": {"nodeType": "refused.plain", "inputs": {}},
        "private": {"nodeType": "versionconflict.plain", "inputs": {}},
        "checkpoint": {
            "nodeType": "dinkster.models.checkpoint_loader_simple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "inspect": {
            "nodeType": "inspector.loaded_model",
            "inputs": {"model": {"$link": {"node": "checkpoint", "output": "MODEL"}}},
        },
    }
    request = {
        "clientId": client,
        "jobId": "loaded",
        "project": "default",
        "graph": {"nodes": nodes},
        "targets": ["plain", "private", "inspect"],
    }
    (evidence / "admission-public-loaded-request.json").write_text(json.dumps(request, indent=2))
    status, submitted = call("/api/jobs", request)
    assert status == 202, (status, submitted)
    started = time.monotonic()
    while time.monotonic() - started < 300:
        status, result = call(f"/api/jobs/{client}/loaded")
        assert status == 200, (status, result)
        if result["state"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(1)
    (evidence / "admission-public-loaded-result.json").write_text(json.dumps(result, indent=2))
    assert result["state"] == "completed", result.get("error", result["state"])
    outputs = result["outputs"]
    assert outputs["plain"]["value"]["value"] == 37
    assert outputs["private"]["version"]["value"] == "4.12.3"
    summary = json.loads(outputs["inspect"]["summary"]["value"])
    assert summary["dependency"] == "4.13.0", summary
    assert summary["parameters"] > 800_000_000, summary
    assert math.isfinite(summary["sample"]), summary
    pids = (summary["pid"], outputs["plain"]["pid"]["value"], outputs["private"]["pid"]["value"])
    assert len(set(pids)) == 3, pids
    assert set(result["executed"]) == set(nodes), result["executed"]
    summary.update(isolatedPids=list(pids[1:]), generation=current, executed=result["executed"])
    (evidence / "admission-public-membership.json").write_text(json.dumps(record, indent=2))
    (evidence / "admission-public-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
