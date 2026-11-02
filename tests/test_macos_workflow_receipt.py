import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from dinkster_graph import graph_from_wire, validate
from dinkster_nodes_comfy.clip import CLIPTextEncode
from dinkster_nodes_comfy.image_file_tools import SaveImage
from dinkster_nodes_comfy.latent import EmptyLatentImage
from dinkster_nodes_comfy.model_file_loaders import CheckpointLoaderSimple
from dinkster_nodes_comfy.standard_sampling import KSampler
from dinkster_nodes_comfy.vae import VAEDecode
from dinkster_schema import build_schemas, schema_to_wire

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("receipt", ROOT / "tools/macos_workflow_receipt.py")
assert SPEC is not None and SPEC.loader is not None
receipt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(receipt)
STARTER_COMMIT = "2f880079bb3b48867795db4f7ed45633fa25e374"
STARTER_SHA256 = "53efb255c2dd8743c57d25f0ad82d1ecb3a29784d95176719507637325dca768"
STARTER_BYTES = (ROOT / "tests/fixtures/macos-sd15.json").read_bytes()
SCHEMAS = build_schemas(
    (CheckpointLoaderSimple, CLIPTextEncode, EmptyLatentImage, KSampler, VAEDecode, SaveImage)
)
CATALOG = {"nodes": {name: schema_to_wire(schema) for name, schema in SCHEMAS.items()}}


def installed_status(root, commit=STARTER_COMMIT):
    project = root / "project"
    generation = project / "generations/2"
    generation.mkdir(parents=True)
    manifest = json.dumps({"format": "dinkster.engine/1", "commit": commit}).encode()
    digest = hashlib.sha256(manifest).hexdigest()
    (root / "release-manifest.json").write_bytes(manifest)
    (project / "current").write_text("2\n")
    (generation / "generation.json").write_text(
        json.dumps(
            {
                "format": "dinkster.generation/1",
                "generation": 2,
                "engine": {"release": "sha256:" + digest},
            }
        )
    )
    engine = {
        "name": "main",
        "generation": 2,
        "state": "ready",
        "url": "http://127.0.0.1:8123",
    }
    runtime = project / "runtime/2"
    runtime.mkdir(parents=True)
    (runtime / "main.json").write_text(
        json.dumps(engine | {"instance": "instance-2", "shutdownToken": "not-public"})
    )
    status = root / "status.json"
    status.write_text(
        json.dumps(
            {
                "format": "dinkster.environment-result/1",
                "operation": "status",
                "status": "clean",
                "current": 2,
                "root": str(project),
                "engines": [engine],
            }
        )
    )
    return status


class MacWorkflowReceiptTests(unittest.TestCase):
    def test_graph_preserves_settings_and_matches_installed_comfy_ports(self):
        # Public catalog of installed engine 2f880079, measured on Apple Silicon.
        expected_ports = {
            "CheckpointLoaderSimple": ({"ckpt_name"}, {"MODEL", "CLIP", "VAE"}),
            "CLIPTextEncode": ({"text", "clip"}, {"CONDITIONING"}),
            "EmptyLatentImage": ({"width", "height", "batch_size"}, {"LATENT"}),
            "KSampler": (
                {
                    "model",
                    "seed",
                    "steps",
                    "cfg",
                    "sampler_name",
                    "scheduler",
                    "positive",
                    "negative",
                    "latent_image",
                    "denoise",
                },
                {"LATENT"},
            ),
            "VAEDecode": ({"samples", "vae"}, {"IMAGE"}),
            "SaveImage": ({"images", "filename_prefix", "mount"}, {"images"}),
        }
        for name, (inputs, outputs) in expected_ports.items():
            self.assertEqual({port.id for port in SCHEMAS[name].inputs}, inputs)
            self.assertEqual({port.id for port in SCHEMAS[name].outputs}, outputs)
        nodes = json.loads(STARTER_BYTES)["graphs"]["g0"]["nodes"]
        self.assertEqual(nodes["latent"]["values"], {"width": 512, "height": 512, "batch_size": 1})
        self.assertEqual(
            nodes["positive"]["values"], {"text": "a cinematic landscape, detailed lighting"}
        )
        self.assertEqual(nodes["negative"]["values"], {"text": "blurry, low quality"})
        self.assertEqual(
            nodes["sampler"]["values"],
            {
                "seed": 91,
                "steps": 20,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        )

    def test_receipt_requires_completed_uncached_mps_job_and_saved_bytes(self):
        for device, state, cached, foreign in (
            ("mps", "completed", [], None),
            ("cpu", "completed", [], None),
            ("mps", "failed", [], None),
            ("mps", "completed", ["sampler"], None),
            ("mps", "completed", [], "accepted"),
            ("mps", "completed", [], "history"),
            ("mps", "completed", [], "no-output"),
            ("mps", "completed", [], "memory-foreign"),
            ("mps", "completed", [], "memory-empty"),
            ("mps", "completed", [], "memory-duplicate"),
            ("mps", "completed", [], "memory-zero"),
            ("mps", "completed", [], "mount-fresh"),
            ("mps", "completed", [], "mount-race"),
            ("mps", "completed", [], "mount-path"),
            ("mps", "completed", [], "mount-mode"),
            ("mps", "completed", [], "mount-kind"),
            ("mps", "completed", [], "mount-failed"),
            ("mps", "completed", [], "mount-timeout"),
            ("mps", "completed", [], "mount-restart"),
        ):
            with (
                self.subTest(device=device, state=state, foreign=foreign),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                checkpoint = root / "model.safetensors"
                checkpoint.write_bytes(b"test checkpoint")
                starter = root / "starter.json"
                starter.write_bytes(STARTER_BYTES)
                status = installed_status(root)
                output = root / "receipt"
                asset = {
                    "digest": "sha256:resolved",
                    "name": checkpoint.name,
                    "size": 15,
                }
                submitted = []
                grants = [
                    {
                        "id": "mac-readiness-models",
                        "path": str(checkpoint.parent),
                        "mode": "read",
                        "kind": "model/checkpoint",
                        "state": "ready",
                    },
                    {
                        "id": "mac-readiness-output",
                        "path": "retained-failed-evidence",
                        "state": "pending",
                    },
                ]
                if foreign in ("mount-fresh", "mount-race"):
                    grants.pop(0)
                elif foreign in ("mount-path", "mount-mode", "mount-kind"):
                    grants[0][foreign.removeprefix("mount-")] = "conflicting"
                elif foreign in ("mount-failed", "mount-timeout"):
                    grants[0]["state"] = "failed" if foreign == "mount-failed" else "pending"
                mount_failure = foreign in (
                    "mount-path",
                    "mount-mode",
                    "mount-kind",
                    "mount-failed",
                    "mount-timeout",
                    "mount-restart",
                )
                success = (
                    device == "mps"
                    and state == "completed"
                    and not cached
                    and foreign in (None, "mount-fresh", "mount-race")
                )
                clock = iter(range(0, 10000, 301 if foreign == "mount-timeout" else 1))

                def respond(
                    request,
                    timeout,
                    *,
                    asset=asset,
                    submitted=submitted,
                    output=output,
                    state=state,
                    device=device,
                    cached=cached,
                    foreign=foreign,
                    grants=grants,
                ):
                    route = request.full_url.removeprefix("http://127.0.0.1:8123")
                    if route == "/supervisor/status":
                        body = {"state": "ready", "instance": "instance-2"}
                    elif route == "/api/nodes":
                        body = CATALOG
                    elif route == "/api/mounts?project=default":
                        body = {
                            "project": "default",
                            "mounts": [dict(row) for row in grants],
                            "restartRequired": foreign == "mount-restart",
                        }
                        if foreign != "mount-timeout":
                            for row in grants:
                                if row["id"] != "mac-readiness-output":
                                    row["state"] = "ready"
                    elif route == "/api/mounts":
                        body = json.loads(request.data)
                        self.assertFalse(any(row["id"] == body["id"] for row in grants))
                        body["state"] = "pending"
                        grants.append(body)
                        if foreign == "mount-race":
                            raise HTTPError(request.full_url, 409, "duplicate", Message(), None)
                    elif route == "/api/mounts/resolve-path":
                        self.assertTrue(
                            all(
                                row["state"] == "ready"
                                for row in grants
                                if row["id"] != "mac-readiness-output"
                            )
                        )
                        body = {"status": "resolved", "asset": asset}
                    elif route == "/api/jobs":
                        submitted.append(json.loads(request.data))
                        if foreign != "no-output":
                            (output / "images/result.png").write_bytes(b"actual output bytes")
                        body = {
                            "jobRef": "server-assigned",
                            "clientId": "mac-readiness",
                            "jobId": "foreign" if foreign == "accepted" else submitted[0]["jobId"],
                        }
                    elif route == "/api/jobs/by-ref/server-assigned":
                        body = {
                            "jobRef": "server-assigned",
                            "clientId": "mac-readiness",
                            "jobId": "foreign" if foreign == "history" else submitted[0]["jobId"],
                            "state": state,
                            "cached": cached,
                            "cacheEnabled": False,
                            "executed": ["sampler", "decode", "save"],
                            "outputs": {
                                "checkpoint": {
                                    "MODEL": {
                                        "meta": {
                                            "resourceId": "resident:checkpoint-model",
                                        }
                                    }
                                }
                            },
                        }
                    elif route == "/memory/status?details=1":
                        row = {
                            "itemId": "foreign"
                            if foreign == "memory-foreign"
                            else "checkpoint-model",
                            "displayName": "resident checkpoint",
                            "bytesByResidency": {}
                            if foreign == "memory-empty"
                            else {f"vram:{device}": 0 if foreign == "memory-zero" else 12345},
                        }
                        body = {
                            "consumerDetails": {
                                "dinkster-nodes-comfy:resident-values-0": [row, row]
                                if foreign == "memory-duplicate"
                                else [row],
                                "other-pack:resident-values-0": [
                                    dict(row, itemId="checkpoint-model")
                                ],
                            }
                        }
                    else:
                        body = {}
                    return io.BytesIO(json.dumps(body).encode())

                with (
                    patch(
                        "sys.argv",
                        [
                            "receipt",
                            "--url",
                            "http://127.0.0.1:8123",
                            "--checkpoint",
                            str(checkpoint),
                            "--starter",
                            str(starter),
                            "--starter-commit",
                            STARTER_COMMIT,
                            "--environment-status",
                            str(status),
                            "--release-manifest",
                            str(root / "release-manifest.json"),
                            "--output",
                            str(output),
                        ],
                    ),
                    patch.object(receipt.platform, "system", return_value="Darwin"),
                    patch.object(receipt.platform, "machine", return_value="arm64"),
                    patch.object(receipt, "CHECKPOINT_BYTES", checkpoint.stat().st_size),
                    patch.object(
                        receipt,
                        "CHECKPOINT_SHA256",
                        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    ),
                    patch.object(receipt, "urlopen", side_effect=respond),
                    patch.object(receipt.time, "sleep"),
                    patch.object(receipt.time, "monotonic", side_effect=clock.__next__),
                    patch("sys.stderr", new_callable=io.StringIO),
                ):
                    if success:
                        receipt.main()
                    else:
                        with self.assertRaises(RuntimeError):
                            receipt.main()
                report = json.loads((output / "receipt.json").read_text())
                if mount_failure:
                    self.assertEqual(report["status"], "failed")
                    self.assertEqual(submitted, [])
                    self.assertIn("mount", report["error"])
                    continue
                if foreign != "accepted":
                    self.assertEqual(report["jobRef"], "server-assigned")
                self.assertEqual(
                    report["status"],
                    "pass" if success else "failed",
                )
                job = submitted[0]
                self.assertFalse(job["cacheEnabled"])
                self.assertEqual(
                    job["graph"]["nodes"]["save"]["inputs"]["mount"],
                    "mac-readiness-comfy-output",
                )
                self.assertEqual(
                    next(row for row in grants if row["id"] == "mac-readiness-output"),
                    {
                        "id": "mac-readiness-output",
                        "path": "retained-failed-evidence",
                        "state": "pending",
                    },
                )
                self.assertEqual(job["graph"]["nodes"]["checkpoint"]["inputs"]["ckpt_name"], asset)
                self.assertEqual(
                    job["graph"]["nodes"]["save"]["inputs"]["images"],
                    {"$link": {"node": "decode", "output": "IMAGE"}},
                )
                diagnostics = validate(graph_from_wire(job["graph"]), SCHEMAS, job["targets"])
                self.assertEqual([item for item in diagnostics if item.severity == "error"], [])
                if report["status"] == "pass":
                    self.assertEqual(report["engineDevice"], "mps")
                    self.assertEqual(
                        report["engineResident"]["resourceId"], "resident:checkpoint-model"
                    )
                    self.assertTrue((output / "memory-status.json").is_file())
                    self.assertTrue(report["installedStatusMatched"])
                    self.assertEqual(report["installedCommit"], STARTER_COMMIT)
                    self.assertEqual(report["installedGeneration"], 2)
                    self.assertNotIn(
                        "not-public",
                        "".join(path.read_text() for path in output.glob("*.json")),
                    )
                    self.assertEqual(report["starterCommit"], STARTER_COMMIT)
                    self.assertEqual(report["starterSha256"], STARTER_SHA256)
                    self.assertEqual(
                        job["graph"]["nodes"]["sampler"]["nodeType"],
                        "KSampler",
                    )
                    self.assertEqual(job["graph"]["nodes"]["sampler"]["inputs"]["steps"], 20)
                    self.assertEqual(
                        report["outputs"][0]["sha256"],
                        hashlib.sha256(b"actual output bytes").hexdigest(),
                    )

    def test_preflight_failures_retain_diagnostics_without_overwriting_existing_output(
        self,
    ):
        for failure in ("checkpoint", "starter", "existing-output"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                checkpoint = root / "checkpoint"
                checkpoint.write_bytes(b"wrong weights")
                starter = root / "starter.json"
                starter.write_text("not JSON")
                output = root / "receipt"
                if failure == "existing-output":
                    output.mkdir()
                    (output / "receipt.json").write_text("preserved")
                with (
                    patch(
                        "sys.argv",
                        [
                            "receipt",
                            "--url",
                            "http://127.0.0.1:8123",
                            "--checkpoint",
                            str(checkpoint),
                            "--starter",
                            str(starter),
                            "--starter-commit",
                            STARTER_COMMIT,
                            "--environment-status",
                            str(root / "missing-status.json"),
                            "--release-manifest",
                            str(root / "release-manifest.json"),
                            "--output",
                            str(output),
                        ],
                    ),
                    patch.object(receipt.platform, "system", return_value="Darwin"),
                    patch.object(receipt.platform, "machine", return_value="arm64"),
                    patch.object(receipt, "CHECKPOINT_BYTES", checkpoint.stat().st_size),
                    patch.object(
                        receipt,
                        "CHECKPOINT_SHA256",
                        "wrong"
                        if failure == "checkpoint"
                        else hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    ),
                    patch("sys.stderr", new_callable=io.StringIO) as stderr,
                    self.assertRaises((ValueError, FileExistsError)),
                ):
                    receipt.main()
                diagnostic = json.loads(stderr.getvalue())
                self.assertEqual(diagnostic["status"], "failed")
                self.assertTrue(diagnostic["error"])
                if failure == "existing-output":
                    self.assertEqual((output / "receipt.json").read_text(), "preserved")
                else:
                    self.assertEqual(json.loads((output / "receipt.json").read_text()), diagnostic)

    def test_only_exact_tracked_starter_bytes_may_reach_engine(self):
        for alteration in (
            "type",
            "link",
            "value",
            "commit",
            "installed-commit",
            "missing-status",
            "generation",
            "manifest",
            "url",
            "instance",
            "catalog-type",
            "catalog-input",
            "catalog-output",
            None,
        ):
            with (
                self.subTest(alteration=alteration),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                checkpoint = root / "checkpoint"
                checkpoint.write_bytes(b"test checkpoint")
                starter = root / "starter.json"
                document = json.loads(STARTER_BYTES)
                graph = document["graphs"]["g0"]
                if alteration == "type":
                    graph["nodes"]["sampler"]["type"] = "dinkster.empty_latent_image"
                elif alteration == "link":
                    graph["links"]["l7"]["from"]["node"] = "latent"
                elif alteration == "value":
                    graph["nodes"]["sampler"]["values"]["steps"] = 0
                starter.write_bytes(
                    json.dumps(document).encode()
                    if alteration in ("type", "link", "value")
                    else STARTER_BYTES
                )
                status = installed_status(
                    root,
                    "1" * 40 if alteration == "installed-commit" else STARTER_COMMIT,
                )
                if alteration == "missing-status":
                    status.unlink()
                elif alteration == "generation":
                    (root / "project/current").write_text("3\n")
                elif alteration == "manifest":
                    (root / "release-manifest.json").write_text("{}")
                elif alteration == "url":
                    document = json.loads(status.read_bytes())
                    document["engines"][0]["url"] = "http://127.0.0.1:9999"
                    status.write_text(json.dumps(document))
                output = root / "receipt"

                class ReachedEngine(Exception):
                    pass

                def respond(request, timeout, *, alteration=alteration):
                    if request.full_url.endswith("/supervisor/status"):
                        return io.BytesIO(
                            json.dumps(
                                {
                                    "state": "ready",
                                    "instance": "foreign"
                                    if alteration == "instance"
                                    else "instance-2",
                                }
                            ).encode()
                        )
                    if request.full_url.endswith("/api/nodes"):
                        catalog = json.loads(json.dumps(CATALOG))
                        if alteration == "catalog-type":
                            del catalog["nodes"]["KSampler"]
                        elif alteration in ("catalog-input", "catalog-output"):
                            role = alteration.removeprefix("catalog-")
                            catalog["nodes"]["KSampler"]["interface"] = [
                                port
                                for port in catalog["nodes"]["KSampler"]["interface"]
                                if port["role"] != role
                            ]
                        return io.BytesIO(json.dumps(catalog).encode())
                    raise ReachedEngine()

                with (
                    patch(
                        "sys.argv",
                        [
                            "receipt",
                            "--url",
                            "http://127.0.0.1:8123",
                            "--checkpoint",
                            str(checkpoint),
                            "--starter",
                            str(starter),
                            "--starter-commit",
                            "0" * 40 if alteration == "commit" else STARTER_COMMIT,
                            "--environment-status",
                            str(status),
                            "--release-manifest",
                            str(root / "release-manifest.json"),
                            "--output",
                            str(output),
                        ],
                    ),
                    patch.object(receipt.platform, "system", return_value="Darwin"),
                    patch.object(receipt.platform, "machine", return_value="arm64"),
                    patch.object(receipt, "CHECKPOINT_BYTES", checkpoint.stat().st_size),
                    patch.object(
                        receipt,
                        "CHECKPOINT_SHA256",
                        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    ),
                    patch.object(receipt, "urlopen", side_effect=respond),
                    patch("sys.stderr", new_callable=io.StringIO),
                    self.assertRaises(
                        ReachedEngine
                        if alteration is None
                        else FileNotFoundError
                        if alteration == "missing-status"
                        else ValueError
                    ),
                ):
                    receipt.main()
                report = json.loads((output / "receipt.json").read_text())
                self.assertEqual(report["status"], "failed")
                if alteration in ("type", "link", "value", "commit"):
                    self.assertIn("tracked starter", report["error"])
                if alteration == "installed-commit":
                    self.assertEqual(report["installedCommit"], "1" * 40)
                    self.assertIn("installed release", report["error"])


if __name__ == "__main__":
    unittest.main()
