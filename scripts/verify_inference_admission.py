"""Check captured inference-admission fixtures through a managed server's HTTP API.

Install dinkster-models plus tests/fixtures/inference-admission/{inspector,refused}
with the public develop command before running this check. The checkpoint folder
must contain v1-5-pruned-emaonly-fp16.safetensors. No sampling parity is measured.
"""

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

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
assert "pack.sandbox.network" in record["refusals"]["refused"]["reason"]
runtime = json.loads((root / "runtime" / current / "main.json").read_text())
base = runtime["url"]


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
if existing is None:
    status, added = call(
        "/api/mounts", {"id": "checkpoint-proof", "path": str(args.checkpoint_directory.resolve())}
    )
    assert status == 201, (status, added)
else:
    assert existing["path"] == str(args.checkpoint_directory.resolve()), existing
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
assert (
    checkpoint["digest"]
    == "blake3:4c50ebc6e2a5cb19e8d19626d5ede1fb64755562085ce7383d86c72d1d03eb7e"
), checkpoint
status, catalog = call("/api/nodes")
(evidence / "admission-public-catalog.json").write_text(json.dumps(catalog))
assert (
    catalog["nodes"]["refused.loaded_model"]["unavailableReason"]
    == record["refusals"]["refused"]["reason"]
)
assert "unavailableReason" not in catalog["nodes"]["refused.plain"]

for name, nodes, targets, expected in [
    ("refusal", {"bad": {"nodeType": "refused.loaded_model", "inputs": {}}}, ["bad"], 400),
    ("plain", {"plain": {"nodeType": "refused.plain", "inputs": {}}}, ["plain"], 202),
    (
        "loaded",
        {
            "checkpoint": {
                "nodeType": "dinkster.models.checkpoint_loader_simple",
                "inputs": {"ckpt_name": checkpoint},
            },
            "inspect": {
                "nodeType": "inspector.loaded_model",
                "inputs": {"model": {"$link": {"node": "checkpoint", "output": "MODEL"}}},
            },
        },
        ["inspect"],
        202,
    ),
]:
    request = {
        "clientId": "admission-proof",
        "jobId": name,
        "project": "default",
        "graph": {"nodes": nodes},
        "targets": targets,
    }
    (evidence / f"admission-public-{name}-request.json").write_text(json.dumps(request, indent=2))
    status, response = call("/api/jobs", request)
    assert status == expected, (name, status, response)
    if name == "refusal":
        assert response["message"] == record["refusals"]["refused"]["reason"]
        status, _ = call("/api/jobs/admission-proof/refusal")
        assert status == 404
        print("refusal: exact network-grant reason; no queued job", flush=True)
        continue
    started = time.monotonic()
    while time.monotonic() - started < 300:
        status, result = call("/api/jobs/admission-proof/" + name)
        if result["state"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(1)
    (evidence / f"admission-public-{name}-result.json").write_text(json.dumps(result, indent=2))
    assert result["state"] == "completed", result
    print(name, result.get("outputs"), flush=True)
(evidence / "admission-public-membership.json").write_text(json.dumps(record, indent=2))
