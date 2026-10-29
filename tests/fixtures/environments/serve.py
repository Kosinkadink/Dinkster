"""Small HTTP engine installed into the fixture wheel, behind the real supervisor."""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--host", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--generation", required=True)
args = parser.parse_args()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/health":
            body = {"ok": True}
        elif self.path == "/api/queue":
            queue = Path("queue.json")
            body = (
                json.loads(queue.read_bytes()) if queue.exists() else {"running": [], "queued": []}
            )
        elif self.path == "/api/cuda":
            body = {"cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}
        elif self.path == "/api/generation":
            body = {"generation": args.generation}
        else:
            self.send_error(404)
            return
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        pass


ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
