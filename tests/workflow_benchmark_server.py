"""Small real HTTP peer for testing benchmark transport, evidence and cleanup."""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def main() -> None:
    system, directory, port, mode = sys.argv[1:]
    output = Path(directory)
    jobs: dict[str, dict] = {}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value: object) -> None:
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            route = urlsplit(self.path)
            if route.path in ("/object_info", "/api/composition"):
                self.reply({"composing": False})
            elif route.path == "/api/mounts":
                self.reply({"mounts": [{"state": "ready"}]})
            elif "/journal" in route.path:
                after = int(parse_qs(route.query).get("after", ["0"])[0])
                identity = route.path.split("/")[-2]
                stream_id = "execution-run/" + identity
                job_ref = identity
                if mode == "wrong-dinkster-job":
                    job_ref = "wrong-job"
                elif mode == "wrong-dinkster-stream":
                    stream_id = "execution-run/wrong-job"
                records = [
                    {
                        "streamId": stream_id,
                        "seq": 1,
                        "name": "node_started",
                        "payload": {"arm": "test@native"},
                    },
                    {
                        "streamId": stream_id,
                        "seq": 2,
                        "name": "job_state",
                        "payload": {"jobRef": job_ref, "state": "completed"},
                    },
                ]
                self.reply(
                    {
                        "records": [r for r in records if r["seq"] > after],
                        "latestSeq": 2,
                        "coalescedBelow": 0,
                    }
                )
            else:
                identity = route.path.rsplit("/", 1)[-1]
                history = jobs[identity]
                self.reply({identity: history} if system == "comfyui" else history)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "dryRun" in self.path:
                self.reply({"graph": body["prompt"], "targets": []})
                return
            with (output / "received.jsonl").open("a") as stream:
                stream.write(json.dumps(body) + "\n")
            if mode == "lost-response":
                self.close_connection = True
                return
            identity = str(len(jobs) + 1)
            history = {
                "jobRef": identity,
                "scope": "local",
                "state": "completed",
                "status": {"completed": True, "status_str": "success"},
                "prompt": [0, identity, body["prompt"], {}, []],
            }
            if system == "comfyui":
                for node, name in (
                    ("48:32", "shift"),
                    ("48:33", "cfg"),
                    ("48:33", "denoise"),
                ):
                    history["prompt"][2][node]["inputs"][name] = float(
                        history["prompt"][2][node]["inputs"][name]
                    )
            if mode == "failed":
                history.update(state="failed", status={"completed": False, "status_str": "error"})
            elif mode == "timeout":
                history.update(
                    state="running", status={"completed": False, "status_str": "running"}
                )
            elif mode == "exit":
                os._exit(3)
            elif mode == "wrong-comfyui-job":
                history["prompt"][1] = "wrong-job"
            elif mode == "wrong-comfyui-graph":
                history["prompt"][2] = {"wrong": {"class_type": "Other", "inputs": {}}}
            elif mode == "missing-comfyui-prompt":
                del history["prompt"]
            elif mode == "malformed-comfyui-prompt":
                history["prompt"] = {}
            elif mode != "no-output":
                (output / "images" / f"{identity}.bin").write_bytes(json.dumps(body).encode())
            jobs[identity] = history
            self.reply({"prompt_id": identity, "jobRef": identity})

    HTTPServer(("127.0.0.1", int(port)), Handler).serve_forever()


if __name__ == "__main__":
    main()
