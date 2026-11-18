"""Check a registry-installed multipart fixture through the public engine API.

Install the published tests/fixtures/multipart-inference wheel and dinkster-models
through the public managed pack command first. This checks process placement and
loaded-model exchange, not sampling parity or the earlier admission controls.
"""

import argparse
import json
import math
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from dinkster_registry import Lockfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = args.output.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    generation = root / "builds" / (root / "current").read_text().strip()
    lock = Lockfile.from_record_json((generation / "dinkster.lock").read_text())
    entry = lock.get("multipart-model")
    assert entry is not None and entry.source.startswith("registry:http"), entry
    assert sum(pack.pack == entry.pack for pack in lock.packs) == 1
    record = json.loads((generation / "inference-process.json").read_text())
    assert "multipart-model~shared" in record["members"], record
    assert "multipart-model" not in record["members"], record
    assert "dinkster-models" in record["members"], record
    runtime = json.loads((root / "runtime" / generation.name / "main.json").read_text())
    base = runtime["url"]

    def call(url, data=None):
        request = Request(
            url,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=120) as response:
            return response.status, json.load(response)

    registry = entry.source.removeprefix("registry:").rstrip("/")
    status, listing = call(registry + "/static/packs/multipart-model/index.json")
    assert status == 200 and listing["name"] == entry.pack, listing
    versions = [row for row in listing["items"] if row["version"] == entry.version]
    assert len(versions) == 1, versions
    assert any(wheel["digest"] == entry.artifact_digest for wheel in versions[0]["wheels"])
    (evidence / "published-version.json").write_text(json.dumps(versions[0], indent=2))
    status, mounts = call(base + "/api/mounts")
    assert status == 200, mounts
    directory = str(args.checkpoint_directory.resolve())
    existing = next((row for row in mounts["mounts"] if row["id"] == "multipart-proof"), None)
    if existing is None:
        status, added = call(base + "/api/mounts", {"id": "multipart-proof", "path": directory})
        assert status == 201, added
    else:
        assert existing["path"] == directory, existing
    checkpoint = None
    entries = None
    started = time.monotonic()
    while time.monotonic() - started < 120:
        status, entries = call(base + "/api/mounts/multipart-proof/entries")
        assert status == 200, entries
        checkpoint = next(
            (
                row
                for row in entries["entries"]
                if row["name"] == "v1-5-pruned-emaonly-fp16.safetensors"
            ),
            None,
        )
        if checkpoint is not None:
            break
        time.sleep(1)
    assert checkpoint is not None, ("Checkpoint scan timed out", entries)
    assert checkpoint["digest"] == (
        "blake3:4c50ebc6e2a5cb19e8d19626d5ede1fb64755562085ce7383d86c72d1d03eb7e"
    ), checkpoint
    client = "multipart-" + uuid.uuid4().hex
    nodes = {
        "plain": {"nodeType": "multipart-model.plain", "inputs": {}},
        "checkpoint": {
            "nodeType": "dinkster.models.checkpoint_loader_simple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "inspect": {
            "nodeType": "multipart-model.loaded_model",
            "inputs": {
                "model": {"$link": {"node": "checkpoint", "output": "MODEL"}},
                "value": {"$link": {"node": "plain", "output": "value"}},
                "producer_pid": {"$link": {"node": "plain", "output": "pid"}},
            },
        },
    }
    request = {
        "clientId": client,
        "jobId": "loaded",
        "project": "default",
        "graph": {"nodes": nodes},
        "targets": ["plain", "inspect"],
        "cacheEnabled": False,
    }
    (evidence / "request.json").write_text(json.dumps(request, indent=2))
    status, submitted = call(base + "/api/jobs", request)
    assert status == 202, submitted
    result = None
    started = time.monotonic()
    while time.monotonic() - started < 300:
        status, result = call(base + f"/api/jobs/{client}/loaded")
        assert status == 200, result
        if result["state"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(1)
    (evidence / "result.json").write_text(json.dumps(result, indent=2))
    assert result is not None, "Job polling timed out before receiving a response"
    assert result["state"] == "completed", result
    assert set(result["executed"]) == set(nodes) and not result["cached"], result
    outputs = result["outputs"]
    assert outputs["plain"]["value"]["value"] == 37, outputs
    summary = json.loads(outputs["inspect"]["summary"]["value"])
    assert summary["value"] == 37 and summary["parameters"] > 800_000_000, summary
    assert math.isfinite(summary["sample"]), summary
    assert summary["producerPid"] == outputs["plain"]["pid"]["value"], summary
    assert summary["pid"] != summary["producerPid"], summary
    summary.update(
        pack=entry.pack,
        version=entry.version,
        artifactDigest=entry.artifact_digest,
        source=entry.source,
        generation=generation.name,
        membership=record,
    )
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
