"""Submit the pinned Comfy SD1.5 acceptance graph to an installed Apple Silicon engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

CHECKPOINT_SHA256 = "e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916"
CHECKPOINT_BYTES = 2132696762
# tests/fixtures/macos-sd15.json is independent of the installed engine's starter templates.
STARTER_SHA256_BY_COMMIT = {
    "2f880079bb3b48867795db4f7ed45633fa25e374": (
        "35afda805252f88399f0f1966315d671b232b8c4e883261e6f21ae5a2936e9c7"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--starter", required=True, type=Path)
    parser.add_argument(
        "--environment-status",
        required=True,
        type=Path,
        help="retained dinkster-env --root PROJECT --json status output",
    )
    parser.add_argument(
        "--release-manifest",
        required=True,
        type=Path,
        help="published release-manifest.json whose digest matches the installed generation",
    )
    parser.add_argument(
        "--starter-commit",
        required=True,
        help="exact landed installed-engine commit from the packet's status receipt",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.absolute()
    output_owned = False
    images = output / "images"

    def save(name: str, value: object) -> None:
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def request(route: str, body: object | None = None) -> dict:
        payload = None if body is None else json.dumps(body).encode()
        with urlopen(
            Request(
                args.url.rstrip("/") + route,
                data=payload,
                headers={"Content-Type": "application/json"},
            ),
            timeout=60,
        ) as response:
            return json.load(response)

    receipt: dict = {
        "status": "incomplete",
        "host": platform.node(),
        "platform": platform.platform(),
    }
    try:
        output.mkdir()
        output_owned = True
        save("receipt.json", receipt)
        images.mkdir()
        address = urlsplit(args.url)
        if (
            address.scheme != "http"
            or address.hostname != "127.0.0.1"
            or address.path not in ("", "/")
        ):
            raise ValueError("--url must be the installed engine's HTTP loopback address")
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise ValueError("this receipt requires a native Apple Silicon Mac")
        checkpoint = args.checkpoint.resolve(strict=True)
        with checkpoint.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        receipt["checkpoint"] = {"bytes": checkpoint.stat().st_size, "sha256": digest}
        if checkpoint.stat().st_size != CHECKPOINT_BYTES or digest != CHECKPOINT_SHA256:
            raise ValueError("checkpoint must match the recorded official SD1.5 fp16 bytes")
        starter_bytes = args.starter.read_bytes()
        receipt["starterSha256"] = hashlib.sha256(starter_bytes).hexdigest()
        receipt["starterCommit"] = args.starter_commit
        expected = STARTER_SHA256_BY_COMMIT.get(args.starter_commit)
        if expected is None or receipt["starterSha256"] != expected:
            raise ValueError(
                "starter bytes do not match the tracked starter "
                "approved for the installed-engine commit"
            )
        status = json.loads(args.environment_status.read_bytes())
        save("environment-status.json", status)
        if (
            status.get("format") != "dinkster.environment-result/1"
            or status.get("operation") != "status"
            or status.get("status") != "clean"
            or type(status.get("current")) is not int
            or status["current"] < 1
        ):
            raise ValueError("installed status must identify a clean active generation")
        project = Path(status["root"])
        number = status["current"]
        if not project.is_absolute() or (project / "current").read_text().strip() != str(number):
            raise ValueError("installed status no longer matches the active generation")
        generation = json.loads(
            (project / "generations" / str(number) / "generation.json").read_bytes()
        )
        save("generation.json", generation)
        if (
            generation.get("format") != "dinkster.generation/1"
            or generation.get("generation") != number
            or generation.get("source", "published") != "published"
        ):
            raise ValueError("installed generation must identify a published release")
        release = generation["engine"]["release"]
        if (
            not isinstance(release, str)
            or not release.startswith("sha256:")
            or len(release) != 71
            or any(char not in "0123456789abcdef" for char in release[7:])
        ):
            raise ValueError("installed generation has no valid release digest")
        manifest_bytes = args.release_manifest.read_bytes()
        (output / "release-manifest.json").write_bytes(manifest_bytes)
        manifest = json.loads(manifest_bytes)
        receipt["installedCommit"] = manifest.get("commit")
        receipt["installedGeneration"] = number
        receipt["installedRelease"] = release
        if (
            "sha256:" + hashlib.sha256(manifest_bytes).hexdigest() != release
            or manifest.get("format") != "dinkster.engine/1"
            or manifest.get("commit") != args.starter_commit
        ):
            raise ValueError("installed release does not match the starter source commit")
        engines = [
            row
            for row in status.get("engines", [])
            if row.get("url", "").rstrip("/") == args.url.rstrip("/")
            and row.get("generation") == number
            and row.get("state") == "ready"
        ]
        if len(engines) != 1:
            raise ValueError("installed status must bind the URL to one ready generation")
        records = [
            json.loads(path.read_bytes())
            for path in (project / "runtime" / str(number)).glob("*.json")
        ]
        instances = [
            row.get("instance")
            for row in records
            if row.get("url") == engines[0]["url"]
            and row.get("name") == engines[0]["name"]
            and row.get("generation") == number
        ]
        if len(instances) != 1 or not isinstance(instances[0], str) or not instances[0]:
            raise ValueError("installed generation has no unique runtime instance")
        supervisor = request("/supervisor/status")
        save("supervisor-status.json", supervisor)
        if supervisor.get("state") != "ready" or supervisor.get("instance") != instances[0]:
            raise ValueError("live supervisor does not match the installed generation")
        receipt["installedStatusMatched"] = True
        starter = json.loads(starter_bytes)
        graph = starter["graphs"][starter["root"]]
        catalog = request("/api/nodes")
        save("catalog.json", catalog)
        for node in graph["nodes"].values():
            schema = catalog["nodes"].get(node["type"])
            if schema is None:
                raise ValueError(f"installed catalog has no node type {node['type']!r}")
            inputs = {port["id"] for port in schema["interface"] if port["role"] == "input"}
            if not set(node["values"]).issubset(inputs):
                raise ValueError(f"graph values do not match catalog inputs for {node['type']!r}")
        for link in graph["links"].values():
            for endpoint, role in (("from", "output"), ("to", "input")):
                node = graph["nodes"][link[endpoint]["node"]]
                ports = catalog["nodes"][node["type"]]["interface"]
                if not any(
                    port["role"] == role and port["id"] == link[endpoint]["port"] for port in ports
                ):
                    raise ValueError(
                        f"graph link does not match catalog {role} for {node['type']!r}"
                    )
        for mount_id, path, mode, kind in (
            ("mac-readiness-models", checkpoint.parent, "read", "model/checkpoint"),
            ("mac-readiness-output", images, "readwrite", "directory/output"),
        ):
            save(
                mount_id + ".json",
                request(
                    "/api/mounts",
                    {
                        "id": mount_id,
                        "path": str(path),
                        "mode": mode,
                        "kind": kind,
                    },
                ),
            )
        resolved = request(
            "/api/mounts/resolve-path",
            {
                "path": str(checkpoint),
                "project": "default",
                "scope": "local",
            },
        )
        save("checkpoint-resolution.json", resolved)
        if resolved.get("status") != "resolved":
            raise RuntimeError("engine refused the checkpoint's granted path")
        nodes = {
            key: {"nodeType": node["type"], "inputs": dict(node["values"])}
            for key, node in graph["nodes"].items()
        }
        for link in graph["links"].values():
            nodes[link["to"]["node"]]["inputs"][link["to"]["port"]] = {
                "$link": {"node": link["from"]["node"], "output": link["from"]["port"]},
            }
        nodes["checkpoint"]["inputs"]["ckpt_name"] = resolved["asset"]
        job = {
            "clientId": "mac-readiness",
            "jobId": str(uuid.uuid4()),
            "graph": {"nodes": nodes},
            "targets": ["save", "checkpoint"],
            "cacheEnabled": False,
        }
        save("request.json", job)
        accepted = request("/api/jobs", job)
        save("accepted.json", accepted)
        if any(accepted.get(key) != job[key] for key in ("clientId", "jobId")):
            raise RuntimeError("engine accepted another job's identity")
        identity = accepted["jobRef"]
        receipt["jobRef"] = identity
        receipt.update(clientId=job["clientId"], jobId=job["jobId"])
        route = "/api/jobs/by-ref/" + quote(identity, safe="")
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            history = request(route)
            save("history.json", history)
            if history.get("jobRef") != identity or any(
                history.get(key) != job[key] for key in ("clientId", "jobId")
            ):
                raise RuntimeError("engine returned another job's history")
            if history.get("state") in ("completed", "failed", "cancelled"):
                break
            time.sleep(1)
        else:
            raise TimeoutError(
                "job did not finish within 15 minutes; retained identity is not resubmitted"
            )
        save("events.json", request(route + "/events?after=0"))
        if history["state"] != "completed":
            raise RuntimeError("engine job did not complete successfully")
        if history.get("cacheEnabled") is not False or history.get("cached") != []:
            raise RuntimeError("engine did not confirm an uncached execution")
        if not {"sampler", "decode", "save"}.issubset(history.get("executed", [])):
            raise RuntimeError("engine did not execute sampling, decoding and saving")
        device = history["outputs"]["checkpoint"]["MODEL"]["meta"]["resources"]["gpu"]
        receipt["engineDevice"] = device
        if device != "mps":
            raise RuntimeError(f"engine reported device {device!r}, not mps")
        files = []
        for path in sorted(images.rglob("*.png")):
            with path.open("rb") as handle:
                image_digest = hashlib.file_digest(handle, "sha256").hexdigest()
            files.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": image_digest,
                }
            )
        if not files:
            raise RuntimeError("completed job saved no PNG output bytes")
        receipt.update(status="pass", outputs=files)
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        print(json.dumps(receipt, sort_keys=True), file=sys.stderr)
        raise
    finally:
        if output_owned:
            save("receipt.json", receipt)


if __name__ == "__main__":
    main()
