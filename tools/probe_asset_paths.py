"""Assert shipped asset paths against a running CPU dinkster-serve.

Start the server with a disposable library, --allow-mount-changes, and the
default installed packs. Pass an empty disposable directory as --mount-dir.
The probe leaves uploaded immutable bytes in that disposable library.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dinkster_assets import digest_bytes
from dinkster_graph import Graph, GraphNode, graph_to_wire


def probe(base: str, mount_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}

    def request(
        method: str,
        path: str,
        expected: int,
        *,
        body: object = None,
        data: bytes | None = None,
    ) -> Any:
        headers = {"Content-Type": "application/octet-stream"}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        call = urllib.request.Request(base + path, data=data, headers=headers, method=method)
        try:
            response = urllib.request.urlopen(call, timeout=20)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            payload = response.read()
            assert response.status == expected, (method, path, response.status, payload)
        counts[str(expected)] = counts.get(str(expected), 0) + 1
        return json.loads(payload) if payload else None

    deadline = time.monotonic() + 60
    while True:
        nodes = request("GET", "/api/nodes", 200)["nodes"]
        if "dinkster.load_image" in nodes:
            break
        assert time.monotonic() < deadline, "media I/O pack did not announce Load Image"
        time.sleep(0.1)

    payload = f"asset probe: opaque workflow bytes {time.time_ns()}".encode()
    digest = digest_bytes(payload)
    assert request("POST", "/api/assets", 201, data=payload)["digest"] == digest
    assert request("POST", "/api/assets", 200, data=payload)["digest"] == digest
    counts["upload"] = 1
    counts["dedupe"] = 1

    mount_dir.mkdir(parents=True, exist_ok=True)
    assert not list(mount_dir.iterdir()), "mount directory must be empty"
    model = mount_dir / "unhashed-model.safetensors"
    model.write_bytes(b"asset probe: mounted model bytes")
    model_digest = digest_bytes(model.read_bytes())
    request(
        "POST",
        "/api/mounts",
        201,
        body={"id": "asset-probe", "path": str(mount_dir), "mode": "readwrite"},
    )
    record_id: str | None = None
    try:
        deadline = time.monotonic() + 30
        while True:
            entries = request(
                "GET", "/api/mounts/asset-probe/entries?recursive=true&limit=50", 200
            )["entries"]
            if entries:
                break
            assert time.monotonic() < deadline, "mount scan did not finish"
            time.sleep(0.1)
        assert len(entries) == 1 and entries[0]["digest"] == model_digest, entries
        counts["mountScan"] = 1
        guesses = request(
            "POST", "/api/assets/guess", 200, body={"names": [model.name, "absent.bin"]}
        )["matches"]
        assert guesses[0]["candidates"][0]["digest"] == model_digest, guesses
        assert guesses[1]["candidates"] == [], guesses
        counts["nameGuess"] = 1
        guesses = request(
            "POST",
            "/api/assets/guess",
            200,
            body={"names": [model.name], "digestHints": {model.name: model_digest}},
        )["matches"]
        assert guesses[0]["candidates"][0]["confidence"] == "digest", guesses
        counts["digestGuess"] = 1

        record = request(
            "POST",
            "/api/library",
            201,
            body={
                "scope": "local",
                "name": "asset-probe",
                "digest": digest,
                "mediaType": "application/vnd.dinkster.workflow+json",
            },
        )
        record_id = record["id"]
        assert request("GET", f"/api/library/{record_id}?scope=local", 200)["digest"] == digest
        updated = request(
            "PATCH",
            f"/api/library/{record_id}",
            200,
            body={"scope": "local", "revision": record["revision"], "name": "renamed-probe"},
        )
        assert updated["name"] == "renamed-probe" and updated["revision"] == 2
        request(
            "PATCH",
            f"/api/library/{record_id}",
            409,
            body={"scope": "local", "revision": 1, "name": "stale"},
        )
        request("DELETE", f"/api/library/{record_id}?scope=local", 204)
        record_id = None
        counts["libraryCrud"] = 1

        missing = digest_bytes(b"asset probe: intentionally absent image")
        graph = Graph(
            nodes={
                "image": GraphNode(
                    "dinkster.load_image",
                    {"image": {"digest": missing, "name": "absent.png", "size": 0}},
                )
            }
        )
        selection = request(
            "POST",
            "/api/jobs",
            409,
            body={
                "clientId": "asset-probe",
                "jobId": "missing",
                "targets": ["image"],
                "graph": graph_to_wire(graph),
            },
        )
        assert selection["error"] == "assets-missing", selection
        assert selection["assets"][0]["digest"] == missing, selection
        counts["assetsMissing"] = 1
    finally:
        if record_id is not None:
            request("DELETE", f"/api/library/{record_id}?scope=local", 204)
        request("DELETE", "/api/mounts/asset-probe", 200)
        model.unlink()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--mount-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(probe(args.url.rstrip("/"), args.mount_dir.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
