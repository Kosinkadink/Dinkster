from __future__ import annotations

import hashlib
import http.client
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import venv
from pathlib import Path
from urllib.request import Request, urlopen

import pytest
from dinkster_supervisor.environment_process import cancel_pending_process, prepare_process
from dinkster_supervisor.environment_runtime import (
    engine_assignments,
    engine_status,
    require_idle,
    start_engine,
    stop_engine,
)
from dinkster_supervisor.environment_state import EnvironmentError, interpreter
from test_environment_generations import _workspace_tree

SERVE = """
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer as BaseServer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--host', required=True)
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--bound-endpoint', type=Path)
parser.add_argument('--generation', required=True)
parser.add_argument('--library-root')
parser.add_argument('--engine-state-root')
parser.add_argument('--prepare-stale-catalogs', action='store_true')
parser.add_argument('--allow-mount-changes', action='store_true')
parser.add_argument('--sandbox-packs', action='store_true')
parser.add_argument('--allow-origin', action='append')
args = parser.parse_args()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/health':
            body = {'ok': True}
        elif self.path == '/api/queue/idle':
            with open('queue.json') as source:
                body = json.load(source)
        elif self.path == '/api/pid':
            body = {'pid': os.getpid()}
        elif self.path == '/api/cuda':
            body = {'cuda': os.environ.get('CUDA_VISIBLE_DEVICES')}
        elif self.path == '/api/environment':
            body = json.loads(os.environ['DINKSTER_ENGINE_IDENTITY'])
        elif self.path == '/api/engine-state':
            body = {'root': args.engine_state_root}
        elif self.path == '/api/sandbox':
            body = {'sandboxPacks': args.sandbox_packs}
        elif self.path == '/api/generation':
            body = {'generation': args.generation}
        else:
            self.send_error(404)
            return
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        pass

class ThreadingHTTPServer(BaseServer):
    def server_activate(self):
        super().server_activate()
        if args.bound_endpoint is not None:
            pending = args.bound_endpoint.with_suffix('.tmp')
            pending.write_text(json.dumps({
                'format': 'dinkster.bound-endpoint/1', 'port': self.server_address[1],
            }))
            pending.replace(args.bound_endpoint)

ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def macos_runtime_owners(root: Path) -> tuple[set[str], set[int]]:
    if sys.platform != "darwin":
        return set(), set()
    uid = getattr(os, "getuid")()  # noqa: B009 - Windows stubs omit this macOS-only API.
    jobs = set()
    for lifecycle in (root / "runtime").glob("*/*.process/*"):
        label = "org.dinkster.supervisor." + hashlib.sha256(str(lifecycle).encode()).hexdigest()
        result = subprocess.run(
            ["/bin/launchctl", "print", f"user/{uid}/{label}"],
            capture_output=True,
        )
        if result.returncode == 0:
            jobs.add(label)
    processes = subprocess.run(["pgrep", "-f", str(root)], capture_output=True, text=True)
    assert processes.returncode in {0, 1}
    return jobs, {int(pid) for pid in processes.stdout.split()}


def make_generation(root: Path, number: int = 1, *, fake_serve: bool = True) -> None:
    generation = root / "builds" / str(number)
    site = _workspace_tree(generation / "python")
    manifest = json.dumps(
        {
            "format": "dinkster.engine/1",
            "version": "test" if fake_serve else "0.0.20",
            "workerProtocol": 1,
            "commit": "d" * 40,
            "capabilities": [
                "bound-endpoint/1",
                *(["library-helper/1", "library-overlap/1"] if not fake_serve else []),
            ],
            **({"libraryApi": 1} if not fake_serve else {}),
            "artifacts": [
                {
                    "name": "constraints.txt",
                    "size": 5,
                    "sha256": hashlib.sha256(b"test\n").hexdigest(),
                }
            ],
        }
    )
    (generation / "engine-manifest.json").write_text(manifest)
    (generation / "build.json").write_text(
        json.dumps(
            {
                "format": "dinkster.generation/1",
                "generation": number,
                "base": "sha256:" + "a" * 64,
                "engine": {
                    "version": "test" if fake_serve else "0.0.20",
                    "release": "sha256:" + hashlib.sha256(manifest.encode()).hexdigest(),
                },
                "constraints": "sha256:" + "c" * 64,
                "variant": "linux-cpu",
                "workerProtocol": 1,
                "files": {
                    "constraints.txt": "sha256:" + "c" * 64,
                    "engine-manifest.json": "sha256:"
                    + hashlib.sha256(manifest.encode()).hexdigest(),
                },
            }
        )
    )
    (generation / "constraints.txt").write_text("test\n")
    if fake_serve:
        package = root / "dinkster"
        package.mkdir(exist_ok=True)
        (package / "__init__.py").write_text("")
        (package / "serve.py").write_text(SERVE)
        (site / "000-project.pth").write_text(str(root.resolve()) + "\n")
    (root / "queue.json").write_text('{"idle": true}')
    if not (root / "current").exists():
        (root / "current").write_text(f"{number}\n")


@pytest.mark.parametrize("helper", [False, True])
def test_managed_state_selection_requires_helper_capability(
    tmp_path: Path, helper: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_supervisor.environment_history import prepare_entry, publish_entry

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "account"))
    for number in (1, 3, 2, 4, 5, 6, 7):
        make_generation(tmp_path, number)
        if helper:
            tree = tmp_path / f"builds/{number}"
            manifest = tree / "engine-manifest.json"
            value = json.loads(manifest.read_text())
            value["capabilities"].append("library-helper/1")
            value["libraryApi"] = 1
            manifest.write_text(json.dumps(value))
            build = tree / "build.json"
            record = json.loads(build.read_text())
            engine_number = 2 if number in (4, 6) else number
            record["engine"] = {
                "version": {1: "0.0.8", 3: "0.0.10", 7: "0.0.7"}.get(engine_number, "0.0.9"),
                "release": "sha256:" + hashlib.sha256(str(engine_number).encode()).hexdigest(),
            }
            record["files"]["engine-manifest.json"] = (
                "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
            )
            record["constraints"] = record["files"]["constraints.txt"] = (
                "sha256:" + hashlib.sha256((tree / "constraints.txt").read_bytes()).hexdigest()
            )
            build.write_text(json.dumps(record))
            publish_entry(
                tmp_path,
                prepare_entry(tmp_path, tree, operation=f"install-{number}", reason="setup"),
            )
    selections: list[str | None] = []
    for number in (1, 3, 2, 3, 4, 4, 5, 6, 7):
        port = free_port()
        try:
            start_engine(tmp_path, number, "main", port)
            with urlopen(f"http://127.0.0.1:{port}/api/engine-state", timeout=10) as response:
                selected = json.load(response)["root"]
            selections.append(selected)
            if helper:
                assert Path(selected).parent == tmp_path.resolve() / "engine-state/main"
                assert Path(selected).is_dir()
            else:
                assert selected is None
        finally:
            stop_engine(tmp_path, "main", generation=number)
        if helper:
            marker = tmp_path / "engine-state/main/state.json"
            record = json.loads(marker.read_text())
            if len(selections) > 1:
                assert record["future"] == {"preserve": [4, 9]}
            record["future"] = {"preserve": [4, 9]}
            marker.write_text(json.dumps(record))
    assert selections[0] == selections[1]
    assert selections[2] == selections[3]
    assert selections[4] == selections[5] == selections[6]
    if helper:
        assert selections[0] != selections[2]
        assert selections[3] != selections[4]
        assert selections[6] != selections[7]
        assert selections[7] != selections[8]
    else:
        assert not (tmp_path / "engine-state").exists()


def test_overlapping_managed_state_preserves_live_work_and_reuses_stopped_selection(
    tmp_path: Path,
) -> None:
    from dinkster_store.library_lock import engine_state

    state, library = tmp_path / "state", tmp_path / "library"
    library.mkdir()
    engine = "sha256:" + "a" * 64
    script = """
import sys
from pathlib import Path
from dinkster_store.library_lock import engine_lock, engine_state, LibraryInUse
state, library, previous = map(Path, sys.argv[1:])
engine = 'sha256:' + 'a' * 64
with engine_state(state, library, 2, engine=engine, readers=(engine,)) as selected:
    assert selected != previous
    (selected / 'work').write_bytes(b'candidate work')
    try:
        with engine_lock(state, 'competing writer', filename=f'.{previous.name}.lock'):
            raise AssertionError('live working state lost its exclusive lock')
    except LibraryInUse:
        pass
    print(selected, flush=True)
"""
    with engine_state(state, library, 1, engine=engine, readers=(engine,)) as previous:
        (previous / "work").write_bytes(b"old client work")
        completed = subprocess.run(
            [sys.executable, "-c", script, str(state), str(library), str(previous)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert completed.returncode == 0, completed.stderr
        selected = Path(completed.stdout.strip())
        assert selected.parent == state
        assert (selected / "work").read_bytes() == b"candidate work"
        assert (previous / "work").read_bytes() == b"old client work"
        with engine_state(state, library, 3, engine=engine, readers=(engine,)) as resumed:
            assert resumed == selected
            assert (resumed / "work").read_bytes() == b"candidate work"
        assert (previous / "work").read_bytes() == b"old client work"


@pytest.mark.parametrize("created_before_refusal", [False, True])
def test_managed_state_retries_refusal_without_library(
    tmp_path: Path, created_before_refusal: bool
) -> None:
    from dinkster_store.library_lock import engine_state

    state, library = tmp_path / "state", tmp_path / "library"
    engine = "sha256:" + "a" * 64
    if created_before_refusal:
        library.mkdir()
    with pytest.raises(RuntimeError, match="record migration refused"):
        with engine_state(state, library, 7, engine=engine, readers=(engine,)) as selected:
            sentinel = selected / "private-work"
            sentinel.write_bytes(b"unfinished work")
            raise RuntimeError("record migration refused")
    marker = state / "state.json"
    record = json.loads(marker.read_text())
    record["future"] = {"preserve": [4, 9]}
    marker.write_text(json.dumps(record))
    if created_before_refusal:
        library.rmdir()
    assert not library.exists()

    with engine_state(state, library, 7, engine=engine, readers=(engine,)) as retried:
        assert retried == state / record["directory"]
        assert (retried / "private-work").read_bytes() == b"unfinished work"
        assert json.loads(marker.read_text()) == record
        assert not library.exists()
    assert not (state / ".state.pending").exists()


@pytest.mark.parametrize(
    "fault", ["unbound", "snapshot-type", "escape", "other-library", "missing-other-library"]
)
def test_managed_state_refuses_unknown_or_incompatible_selection(
    tmp_path: Path, fault: str
) -> None:
    from dinkster_store.library_lock import engine_state

    state, library = tmp_path / "state", tmp_path / "library"
    state.mkdir()
    library.mkdir()
    retained = state / "working-0123456789abcdef0123456789abcdef"
    retained.mkdir()
    sentinel = retained / "private-work"
    sentinel.write_bytes(b"unfinished newer work")
    record: dict[str, object] = {
        "format": "dinkster.engine-state-selection/1",
        "library": str(library),
        "lastSnapshot": 8,
        "directory": retained.name,
        "engine": "sha256:" + "a" * 64,
    }
    if fault == "snapshot-type":
        record["lastSnapshot"] = True
    elif fault == "escape":
        record["directory"] = "../outside"
    elif fault in {"other-library", "missing-other-library"}:
        other = tmp_path / "other-library"
        if fault == "other-library":
            other.mkdir()
        record["library"] = str(other)
    if fault != "unbound":
        (state / "state.json").write_text(json.dumps(record))
    with (
        pytest.raises(ValueError),
        engine_state(
            state, library, 7, engine="sha256:" + "a" * 64, readers=("sha256:" + "a" * 64,)
        ),
    ):
        pytest.fail("unsafe state selection was admitted")
    assert sentinel.read_bytes() == b"unfinished newer work"
    if fault == "unbound":
        assert not (state / "state.json").exists()
    else:
        assert json.loads((state / "state.json").read_text()) == record
    assert not (state / ".state.pending").exists()


def test_managed_environment_identity_survives_restart_and_snapshot_change(tmp_path: Path):
    identities = []
    for label in ("first", "second"):
        root = tmp_path / label
        root.mkdir()
        make_generation(root)
        make_generation(root, 2)
        identity = None
        for snapshot, name in ((1, "main"), (1, "main"), (2, "preview")):
            port = free_port()
            try:
                start_engine(root, snapshot, name, port)
                with urlopen(f"http://127.0.0.1:{port}/api/environment", timeout=10) as response:
                    reported = json.load(response)
                if identity is None:
                    identity = reported["environment"]
                    assert len(identity) == 32
                    assert identity != "main"
                assert reported == {
                    "environment": identity,
                    "snapshot": str(snapshot),
                    "identityVersion": 1,
                }
                assert (root / "environment-id").read_text().strip() == identity
            finally:
                stop_engine(root, name, generation=snapshot)
        identities.append(identity)
        (root / "environment-id").write_text("broken\n")
        with pytest.raises(EnvironmentError, match="invalid environment identity"):
            start_engine(root, 2, "main", free_port())
        assert (root / "environment-id").read_text() == "broken\n"
    assert identities[0] != identities[1]


def test_managed_environment_refuses_hardlinked_identity(tmp_path: Path):
    original = tmp_path / "original-id"
    original.write_bytes(b"0123456789abcdef" * 2 + b"\n")
    root = tmp_path / "environment"
    root.mkdir()
    make_generation(root)
    identity = root / "environment-id"
    identity.hardlink_to(original)
    try:
        with pytest.raises(EnvironmentError, match="environment identity must not be linked"):
            start_engine(root, 1, "main", free_port())
        assert identity.read_bytes() == original.read_bytes() == b"0123456789abcdef" * 2 + b"\n"
    finally:
        stop_engine(root, "main", generation=1)


@pytest.mark.skipif(sys.platform != "darwin", reason="requires native launchd coalition accounting")
def test_macos_stop_proves_detached_descendant_exit_without_stopping_other_processes(
    tmp_path: Path,
):
    from dinkster_supervisor.environment_runtime import generation_runtime_reason

    make_generation(tmp_path)
    serve = tmp_path / "dinkster/serve.py"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(30)
        detached = (
            "import os,signal,socket\n"
            "if os.fork(): os._exit(0)\n"
            "os.setsid()\n"
            "if os.fork(): os._exit(0)\n"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            f"connection=socket.create_connection(('127.0.0.1',{listener.getsockname()[1]}))\n"
            "connection.sendall(b'alive')\n"
            "connection.settimeout(120)\n"
            "while connection.recv(1) not in (b'', b'x'): pass\n"
        )
        serve.write_text(
            SERVE.replace(
                "args = parser.parse_args()",
                "args = parser.parse_args()\nimport subprocess,sys\n"
                f"subprocess.run([sys.executable,'-c',{detached!r}],check=True)",
            )
        )
        outsider = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(120)"])
        connection = None
        try:
            start_engine(tmp_path, 1, "main", free_port())
            assert not list((tmp_path / "runtime").rglob("launchd.plist"))
            connection, _ = listener.accept()
            connection.settimeout(30)
            assert connection.recv(5) == b"alive"
            assert generation_runtime_reason(tmp_path, 1) is not None
            stop_engine(tmp_path, "main", generation=1)
            assert connection.recv(1) == b""
            assert outsider.poll() is None
            receipts = list((tmp_path / "runtime/1/main.process").glob("*/state.json"))
            assert len(receipts) == 1
            receipt = json.loads(receipts[0].read_text())
            assert receipt["state"] == "stopped" and receipt["descendantsStopped"] is True
            assert generation_runtime_reason(tmp_path, 1) is None
        finally:
            try:
                stop_engine(tmp_path, "main", generation=1)
            finally:
                if connection is not None:
                    try:
                        connection.sendall(b"x")
                    except OSError:
                        pass
                    connection.close()
                outsider.terminate()
                outsider.wait(timeout=10)


@pytest.mark.parametrize("in_process", [False, True])
def test_managed_node_exclusions_refuse_execution_and_restore_without_reinstall(
    tmp_path: Path, in_process: bool
) -> None:
    import asyncio

    import aiohttp
    from dinkster_graph import Graph, GraphNode, graph_to_wire
    from dinkster_registry import LockedPack, Lockfile, build_artifact
    from dinkster_workers.doctor import prepare_catalog
    from test_compose_dispatch import _write_pack
    from test_serve import _job_finish, _peek_value

    from dinkster.generation_packs import stage_generation_packs

    manifest = _write_pack(
        tmp_path / "source", "exclusion", "exclusion.blocked", own_type="exclusion.allowed"
    )
    manifest.write_text(
        manifest.read_text().replace("[pack.entry]", 'version="1.0.0"\nengine=">=0"\n[pack.entry]')
    )
    archive = tmp_path / "pack.zip"
    digest = build_artifact(manifest.parent, archive)
    generation = tmp_path / "builds/1"
    generation.mkdir(parents=True)
    stage_generation_packs(
        generation,
        Lockfile.of([LockedPack("exclusion", "1.0.0", digest, "dinkster", ("exclusion",))]),
        {"exclusion": archive},
    )
    make_generation(tmp_path, fake_serve=False)
    record_path = generation / "build.json"
    if in_process:
        record_path.write_text(
            json.dumps(json.loads(record_path.read_text()) | {"inProcessPacks": ["exclusion"]})
        )
    else:
        venv.EnvBuilder(with_pip=False, system_site_packages=True, symlinks=os.name != "nt").create(
            generation / "packs/exclusion"
        )
    source = generation / "pack-sources/exclusion"
    assert prepare_catalog(
        source / "dinkster-pack.toml", environment={"PYTHONPATH": str(source)}
    ).ok
    retained = {
        str(p.relative_to(generation)): p.read_bytes()
        for p in generation.rglob("*")
        if p.is_file() and not {"python", "packs", "__pycache__"}.intersection(p.parts)
    }
    port = free_port()
    project = tmp_path / "environment.json"
    project.write_text(
        json.dumps(
            {
                "format": "dinkster.project/1",
                "channel": "nightly",
                "variant": "linux-cpu",
                "mirrors": [],
                "port": port,
            }
        )
    )
    command = [
        sys.executable,
        "-m",
        "dinkster_supervisor.environment",
        "--root",
        str(tmp_path),
        "nodes",
    ]

    def configure(*arguments: str) -> None:
        before = project.read_bytes()
        planned = subprocess.run(
            [*command, *arguments, "--plan", "--json"], capture_output=True, text=True
        )
        assert planned.returncode == 0, planned.stdout + planned.stderr
        assert project.read_bytes() == before
        applied = subprocess.run(
            [*command, *arguments, "--yes", "--json"], capture_output=True, text=True
        )
        assert applied.returncode == 0, applied.stdout + applied.stderr

    async def probe(excluded: bool, suffix: str) -> None:
        base = f"http://127.0.0.1:{port}"
        async with aiohttp.ClientSession(trust_env=False) as client:
            async with client.get(base + "/api/nodes") as response:
                catalog = (await response.json())["nodes"]
            assert set(catalog) == {"exclusion.blocked", "exclusion.allowed"}
            reason = catalog["exclusion.blocked"].get("unavailableReason")
            assert bool(reason) is excluded
            assert "unavailableReason" not in catalog["exclusion.allowed"]
            for node_type, output in (
                ("exclusion.blocked", "exclusion:probe"),
                ("exclusion.allowed", "own:probe"),
            ):
                job = node_type + suffix
                async with client.post(
                    base + "/api/jobs",
                    json={
                        "clientId": "e2e",
                        "jobId": job,
                        "graph": graph_to_wire(
                            Graph(nodes={"save": GraphNode(node_type, {"value": "probe"})})
                        ),
                        "targets": ["save"],
                    },
                ) as response:
                    body = await response.json()
                    if excluded and node_type == "exclusion.blocked":
                        assert response.status == 400, body
                        assert body["error"] == "node-unavailable" and body["message"] == reason
                        continue
                    assert response.status == 202, body
                assert (await _job_finish(client, base, job))["state"] == "completed"
                assert await _peek_value(client, base, job, "value") == output

    try:
        configure("--exclude", "exclusion.blocked")
        for suffix in ("first", "restart"):
            start_engine(tmp_path, 1, "main", port)
            asyncio.run(probe(True, suffix))
            before = project.read_bytes()
            refused = subprocess.run([*command, "--yes", "--json"], capture_output=True, text=True)
            assert refused.returncode != 0 and "stop all environment engines" in refused.stdout
            assert project.read_bytes() == before
            stop_engine(tmp_path, "main")
        configure("--exclude", "exclusion.unknown")
        with pytest.raises(
            EnvironmentError, match="unknown excluded node types: exclusion.unknown"
        ):
            start_engine(tmp_path, 1, "main", port)
        stop_engine(tmp_path, "main")
        configure()
        start_engine(tmp_path, 1, "main", port)
        asyncio.run(probe(False, "restored"))
        assert json.loads(project.read_text())["excludedNodeTypes"] == []
        assert all((generation / name).read_bytes() == value for name, value in retained.items())
    finally:
        stop_engine(tmp_path, "main")


@pytest.mark.parametrize(
    "policy", [None, {}, "exclusion.blocked", [True], [""], [" spaced"], ["same", "same"]]
)
def test_node_exclusions_refuse_malformed_settings_without_saving(
    tmp_path: Path, policy: object
) -> None:
    project = tmp_path / "environment.json"
    original = json.dumps(
        {
            "format": "dinkster.project/1",
            "channel": "nightly",
            "variant": "linux-cpu",
            "mirrors": [],
            "port": free_port(),
            "excludedNodeTypes": policy,
        }
    ).encode()
    project.write_bytes(original)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dinkster_supervisor.environment",
            "--root",
            str(tmp_path),
            "nodes",
            "--yes",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "excludedNodeTypes" in result.stdout
    assert project.read_bytes() == original


@pytest.mark.parametrize("requirement", ["hidden-models", "unknown-schema", "library-helper"])
def test_incompatible_library_refuses_before_engine_start(tmp_path: Path, requirement: str) -> None:
    root = tmp_path / "environment"
    make_generation(root)
    library = root / "dinkster-library"
    library.mkdir()
    if requirement == "hidden-models":
        (library / "project-folders.json").write_text('{"hiddenModels": {"art": ["model"]}}')
        message = "requires Dinkster 0.0.15"
    elif requirement == "library-helper":
        (library / "library-format.json").write_text(
            json.dumps(
                {
                    "format": "dinkster.library-format/1",
                    "libraryApi": 1,
                    "requiredCapabilities": ["library-helper/1"],
                }
            )
        )
        message = "requires library-helper/1"
    else:
        with sqlite3.connect(library / "library.sqlite") as database:
            database.execute("CREATE TABLE alembic_version (version_num TEXT)")
            database.execute("INSERT INTO alembic_version VALUES ('future_revision')")
        message = "requires a Dinkster version supporting library schema future_revision"
    before = {path.name: path.read_bytes() for path in library.iterdir()}
    with pytest.raises(EnvironmentError, match=message) as refusal:
        start_engine(root, 1, "main", free_port())
    if requirement == "hidden-models":
        assert "Unhide models in projects art using the current engine" in str(refusal.value)
    assert engine_status(root) == ()
    assert {path.name: path.read_bytes() for path in library.iterdir()} == before
    assert (root / "current").read_text().strip() == "1"


def test_engine_start_accepts_annotated_migration_without_importing_assets(tmp_path: Path) -> None:
    root = tmp_path / "environment"
    make_generation(root)
    assets = root / "dinkster_assets"
    migrations = assets / "migrations/versions"
    migrations.mkdir(parents=True)
    (assets / "__init__.py").write_text("raise RuntimeError('assets imported by preflight')\n")
    (migrations / "initial.py").write_text("revision: str = '0001_asset_records'\n")
    library = root / "dinkster-library"
    library.mkdir()
    with sqlite3.connect(library / "library.sqlite") as database:
        database.execute("CREATE TABLE alembic_version (version_num TEXT)")
        database.execute("INSERT INTO alembic_version VALUES ('0001_asset_records')")
    before = (library / "library.sqlite").read_bytes()
    try:
        record = start_engine(root, 1, "main", free_port())
        with urlopen(str(record["url"]) + "/api/health", timeout=10) as response:
            assert json.load(response)["ok"]
        assert (library / "library.sqlite").read_bytes() == before
    finally:
        stop_engine(root, "main")


@pytest.mark.parametrize("https", [False, True])
def test_managed_public_listener_persists_bindings_tls_and_browser_boundary(
    tmp_path: Path, https: bool
) -> None:
    import asyncio
    import ssl

    import aiohttp
    from dinkster_assets import digest_bytes
    from test_media_authority import _jpeg, _media_headers, _media_path

    if https:
        from dinkster_registry import LockedPack, Lockfile, build_artifact
        from dinkster_workers.doctor import prepare_catalog
        from test_serve import write_shutdown_pack

        from dinkster.generation_packs import stage_generation_packs

        manifest = write_shutdown_pack(tmp_path / "source", 57)
        archive = tmp_path / "pack.zip"
        digest = build_artifact(manifest.parent, archive)
        pack = "shutdown-pack-57"
        lock = Lockfile.of([LockedPack(pack, "1.2.3", digest, "dinkster", ("shutdown57",))])
        generation = tmp_path / "builds/1"
        generation.mkdir(parents=True)
        stage_generation_packs(generation, lock, {pack: archive})
        source = generation / "pack-sources" / pack
        assert prepare_catalog(
            source / "dinkster-pack.toml", environment={"PYTHONPATH": str(source)}
        ).ok
        make_generation(tmp_path, fake_serve=False)
        record = json.loads((generation / "build.json").read_text())
        record["inProcessPacks"] = [pack]
        (generation / "build.json").write_text(json.dumps(record))
    else:
        make_generation(tmp_path)
    port = free_port()
    settings = {
        "format": "dinkster.project/1",
        "channel": "nightly",
        "variant": "linux-cpu",
        "mirrors": [],
        "port": port,
    }
    (tmp_path / "environment.json").write_text(json.dumps(settings))
    addresses = ["::1", "127.0.0.1"] if https else ["::1"]
    command = [
        sys.executable,
        "-m",
        "dinkster_supervisor.environment",
        "--root",
        str(tmp_path),
        "listener",
    ]
    for address in addresses:
        command.extend(["--address", address])
    command.extend(["--allow-origin", "https://allowed.example", "--allow-host", "engine.example"])
    certificate = Path(__file__).parent / "tls/service-cert.pem"
    if https:
        command.extend(
            [
                "--tls-certificate",
                str(certificate),
                "--tls-key",
                str(certificate.with_name("service-key.pem")),
            ]
        )
    planned = subprocess.run([*command, "--plan", "--json"], capture_output=True, text=True)
    assert planned.returncode == 0, planned.stdout + planned.stderr
    assert json.loads((tmp_path / "environment.json").read_text()) == settings
    applied = subprocess.run([*command, "--yes", "--json"], capture_output=True, text=True)
    assert applied.returncode == 0, applied.stdout + applied.stderr
    image = _jpeg()
    image_limit = len(image) + 2
    upload_command = [
        sys.executable,
        "-m",
        "dinkster_supervisor.environment",
        "--root",
        str(tmp_path),
        "uploads",
        "--generic",
        "64",
        "--image",
        str(image_limit),
    ]
    if https:
        original = (tmp_path / "environment.json").read_bytes()
        preview = subprocess.run(
            [*upload_command, "--plan", "--json"], capture_output=True, text=True
        )
        assert preview.returncode == 0, preview.stdout + preview.stderr
        assert (tmp_path / "environment.json").read_bytes() == original
        saved = subprocess.run([*upload_command, "--yes", "--json"], capture_output=True, text=True)
        assert saved.returncode == 0, saved.stdout + saved.stderr
        assert json.loads((tmp_path / "environment.json").read_text())["uploadLimits"] == {
            "generic": 64,
            "media/image": image_limit,
        }

    staging: Path | None = None
    if https:
        selected = tmp_path.parent / (tmp_path.name + "-scratch")
        selected.mkdir()
        configured = subprocess.run(
            [
                sys.executable,
                "-m",
                "dinkster_supervisor.environment",
                "--root",
                str(tmp_path),
                "scratch",
                "--directory",
                str(selected),
                "--yes",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert configured.returncode == 0, configured.stdout + configured.stderr
        assert Path(json.loads(configured.stdout)["scratchRoot"]).is_dir()

    async def probe() -> None:
        context = ssl.create_default_context(cafile=str(certificate)) if https else None
        async with aiohttp.ClientSession(trust_env=False) as client:
            for address in addresses:
                host = f"[{address}]" if ":" in address else address
                url = f"{'https' if https else 'http'}://{host}:{port}"
                options = {"ssl": context, "server_hostname": "localhost"} if https else {}
                async with client.get(url + "/api/health", **options) as response:
                    assert response.status == 200
                    assert (await response.json())["ok"] is True
                if https:
                    deadline = time.monotonic() + 10
                    while True:
                        async with client.get(url + "/api/mounts", **options) as response:
                            assert response.status == 200
                            imports = next(
                                row
                                for row in (await response.json())["mounts"]
                                if row["id"] == "imports"
                            )
                        if imports["state"] not in {"pending", "scanning"}:
                            break
                        assert time.monotonic() < deadline, imports
                        await asyncio.sleep(0.02)
                    assert imports["state"] == "ready", imports

                    async def chunks(body: bytes):
                        yield body[:7]
                        if len(body) == image_limit - 1:
                            assert staging is not None
                            for _ in range(1000):
                                if list(staging.glob(".media-upload-*")):
                                    break
                                await asyncio.sleep(0.01)
                            else:
                                raise AssertionError("upload did not use selected staging")
                        await asyncio.sleep(0)
                        yield body[7:]

                    async with client.get(url + "/api/assets/upload-limits", **options) as response:
                        assert response.status == 200
                        assert await response.json() == {
                            "generic": 64,
                            "media/image": image_limit,
                            "media/audio": 1024**3,
                            "media/video": 1024**3,
                            "media/model3d": 1024**3,
                        }
                    for streamed in (False, True):
                        for size in (63, 64, 65):
                            body = bytes([size]) * size
                            async with client.post(
                                url + "/api/assets",
                                data=chunks(body) if streamed else body,
                                **options,
                            ) as response:
                                assert response.status in ((413,) if size > 64 else (200, 201))
                            async with client.get(
                                url + "/api/assets/" + digest_bytes(body), **options
                            ) as response:
                                assert response.status == (404 if size > 64 else 200)
                        for padding in (1, 2, 3):
                            body = image[:-2] + b"\x00" * padding + image[-2:]
                            async with client.post(
                                url
                                + _media_path("local", "media/image", f"boundary-{padding}.jpg"),
                                data=chunks(body) if streamed else body,
                                headers=_media_headers("image/jpeg"),
                                **options,
                            ) as response:
                                assert response.status in ((413,) if padding > 2 else (200, 201))
                            async with client.get(
                                url + "/api/assets/" + digest_bytes(body), **options
                            ) as response:
                                assert response.status == (404 if padding > 2 else 200)
                    async with client.ws_connect(
                        url + "/api/events", origin="https://allowed.example", **options
                    ) as websocket:
                        await websocket.ping()
                    with pytest.raises(aiohttp.WSServerHandshakeError) as denied:
                        await client.ws_connect(
                            url + "/api/events", origin="https://denied.example", **options
                        )
                    assert denied.value.status == 403
                async with client.get(
                    url + "/api/health", headers={"Origin": "https://allowed.example"}, **options
                ) as response:
                    assert response.status == 200
                    assert (
                        response.headers["Access-Control-Allow-Origin"] == "https://allowed.example"
                    )
                async with client.get(
                    url + "/api/health", headers={"Origin": "https://denied.example"}, **options
                ) as response:
                    assert response.status == 403
                    assert "Access-Control-Allow-Origin" not in response.headers
                async with client.get(
                    url + "/api/health", headers={"Host": "unlisted.example"}, **options
                ) as response:
                    assert response.status == 421
                async with client.post(url + "/supervisor/shutdown", **options) as response:
                    assert response.status == 403
            if not https:
                with socket.socket() as sock:
                    assert sock.connect_ex(("127.0.0.1", port)) != 0
            else:
                with pytest.raises(aiohttp.ClientError):
                    await client.get(f"http://127.0.0.1:{port}/api/health")

    try:
        for _ in range(2):
            start_engine(tmp_path, 1, "main", port)
            if https:
                upload_roots = list((tmp_path / "dinkster-library/.engine-uploads").iterdir())
                assert len(upload_roots) == 1
                staging = upload_roots[0]
                assert staging.is_dir()
                original = (tmp_path / "environment.json").read_bytes()
                refused = subprocess.run(
                    [*upload_command, "--yes", "--json"], capture_output=True, text=True
                )
                assert refused.returncode != 0
                assert "stop all environment engines" in refused.stdout
                assert (tmp_path / "environment.json").read_bytes() == original
            asyncio.run(probe())
            if https:
                assert staging is not None and list(staging.iterdir()) == []
                refused_scratch = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "dinkster_supervisor.environment",
                        "--root",
                        str(tmp_path),
                        "scratch",
                        "--yes",
                        "--json",
                    ],
                    capture_output=True,
                    text=True,
                )
                assert refused_scratch.returncode != 0
                assert "stop all environment engines" in refused_scratch.stdout
            stop_engine(tmp_path, "main")
        if https:
            reset = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dinkster_supervisor.environment",
                    "--root",
                    str(tmp_path),
                    "uploads",
                    "--yes",
                    "--json",
                ],
                capture_output=True,
                text=True,
            )
            assert reset.returncode == 0, reset.stdout + reset.stderr
            assert json.loads((tmp_path / "environment.json").read_text())["uploadLimits"] == {}
    finally:
        stop_engine(tmp_path, "main")


@pytest.mark.parametrize("grouped", [False, True])
def test_managed_pack_selection_preserves_software_and_restores_ordinary_launch(
    tmp_path: Path, grouped: bool
) -> None:
    import asyncio

    import aiohttp
    from dinkster_graph import Graph, GraphNode, graph_to_wire
    from dinkster_registry import LockedPack, Lockfile, build_artifact
    from dinkster_workers.doctor import prepare_catalog
    from test_serve import _job_finish, _peek_value, write_shutdown_pack

    from dinkster.generation_packs import stage_generation_packs

    generation = tmp_path / "builds/1"
    generation.mkdir(parents=True)
    partner_import = tmp_path / "partner-imported"
    entries = []
    archives = {}
    for index in (61, 62, 63, 64):
        manifest = write_shutdown_pack(tmp_path / f"source-{index}", index)
        text = manifest.read_text().replace(
            "[pack.entry]", 'version="1.2.3"\nengine=">=0"\n[pack.entry]'
        )
        if index == 61:
            text += '\n[pack.capabilities]\n"selection.provider"="1.0.0"\n'
        elif index == 62:
            text += (
                '\n[pack.dependencies]\nshutdown-pack-61=">=1,<2"\n'
                '[pack.requirements.capabilities]\n"selection.provider"=">=1,<2"\n'
            )
        elif index == 63:
            text = text.replace(
                "[pack.entry]", 'node-source="remote-gateway-catalog"\n[pack.entry]'
            )
        else:
            text += '[pack.requirements.capabilities]\n"selection.provider"=">=1,<2"\n'
        manifest.write_text(text)
        (manifest.parent / f"shutdown_pack_{index}.py").write_text(
            (
                f"from pathlib import Path\nPath({str(partner_import)!r}).touch()\n"
                if index == 63
                else ""
            )
            + "from dinkster_schema import Node, NodeSchema, Output, TypeExpr\n"
            "class Probe(Node):\n"
            "    @classmethod\n"
            "    def define_schema(cls):\n"
            f"        return NodeSchema(node_type='shutdown{index}.probe', "
            "body=(Output('value', TypeExpr.concrete('INT')),))\n"
            "    @classmethod\n"
            "    def execute(cls):\n"
            f"        return cls.outputs(value={index})\n"
            "NODES = (Probe,)\n"
        )
        archive = tmp_path / f"{index}.zip"
        digest = build_artifact(manifest.parent, archive)
        name = f"shutdown-pack-{index}"
        entries.append(LockedPack(name, "1.2.3", digest, "dinkster", (f"shutdown{index}",)))
        archives[name] = archive
    stage_generation_packs(generation, Lockfile.of(entries), archives)
    make_generation(tmp_path, fake_serve=False)
    record = json.loads((generation / "build.json").read_text())
    if grouped:
        for entry in entries:
            venv.EnvBuilder(
                with_pip=False, system_site_packages=True, symlinks=os.name != "nt"
            ).create(generation / "packs" / entry.pack)
        (generation / "inference-process.json").write_text(
            json.dumps(
                {
                    "format": "dinkster.inference-process/1",
                    "members": [entry.pack for entry in entries],
                    "environment": "shutdown-pack-63",
                }
            )
        )
    else:
        record["inProcessPacks"] = [entry.pack for entry in entries]
    (generation / "build.json").write_text(json.dumps(record))
    for entry in entries:
        source = generation / "pack-sources" / entry.pack
        assert prepare_catalog(
            source / "dinkster-pack.toml", environment={"PYTHONPATH": str(source)}
        ).ok
    retained = {
        str(path.relative_to(generation)): path.read_bytes()
        for path in generation.rglob("*")
        if path.is_file()
        and "python" not in path.parts
        and "packs" not in path.parts
        and "__pycache__" not in path.parts
    }
    port = free_port()
    settings = {
        "format": "dinkster.project/1",
        "channel": "nightly",
        "variant": "linux-cpu",
        "mirrors": [],
        "port": port,
    }
    project = tmp_path / "environment.json"
    project.write_text(json.dumps(settings))
    command = [sys.executable, "-m", "dinkster_supervisor.environment", "--root", str(tmp_path)]

    def configure(*arguments: str) -> None:
        original = project.read_bytes()
        preview = subprocess.run(
            [*command, "packs", *arguments, "--plan", "--json"], capture_output=True, text=True
        )
        assert preview.returncode == 0, preview.stdout + preview.stderr
        assert project.read_bytes() == original
        applied = subprocess.run(
            [*command, "packs", *arguments, "--yes", "--json"], capture_output=True, text=True
        )
        assert applied.returncode == 0, applied.stdout + applied.stderr

    async def probe(selected: set[int], *, offline: bool = False) -> None:
        base = f"http://127.0.0.1:{port}"
        async with aiohttp.ClientSession(trust_env=False) as client:
            async with asyncio.timeout(60):
                while True:
                    async with client.get(base + "/api/composition") as response:
                        report = await response.json()
                    if "composing" not in report:
                        break
                    await asyncio.sleep(0.05)
            async with client.get(base + "/api/nodes") as response:
                catalog = await response.json()
                assert set(catalog["nodes"]) == {f"shutdown{index}.probe" for index in selected}, (
                    catalog
                )
                assert response.headers["X-Dinkster-Runtime-Network"] == (
                    "offline" if offline else "online"
                )
                if offline:
                    assert "connect-src 'self'" in response.headers["Content-Security-Policy"]
                    assert "offline runtime" in str(catalog["nodes"]["shutdown63.probe"])
            for index in (61, 62, 63, 64):
                job = f"selection-{index}"
                async with client.post(
                    base + "/api/jobs",
                    json={
                        "clientId": "e2e",
                        "jobId": job,
                        "graph": graph_to_wire(
                            Graph(nodes={"save": GraphNode(f"shutdown{index}.probe", {})})
                        ),
                        "targets": ["save"],
                    },
                ) as response:
                    status = response.status
                    body = await response.json()
                if offline and index == 63:
                    assert status == 400 and body["error"] == "node-unavailable", body
                    assert "network online" in body["message"]
                elif index in selected:
                    assert status == 202, body
                    assert (await _job_finish(client, base, job))["state"] == "completed"
                    assert await _peek_value(client, base, job, "value") == index
                else:
                    assert status == 202, body
                    failed = await _job_finish(client, base, job)
                    assert failed["state"] == "failed", failed
                    assert "unknown-node-type" in str(failed["error"]), failed
            async with client.get(
                base + "/api/health", headers={"Origin": "https://unapproved.example"}
            ) as response:
                assert response.status == 403

    try:
        for arguments, selected in (
            (("--none",), set()),
            (("--only", "shutdown-pack-61"), {61}),
            (("--exclude-partners",), {61, 62, 64}),
        ):
            configure(*arguments)
            start_engine(tmp_path, 1, "main", port, selected_packs=True)
            asyncio.run(probe(selected))
            with pytest.raises(EnvironmentError, match="pack selection changed"):
                start_engine(tmp_path, 1, "main", port)
            original = project.read_bytes()
            refused = subprocess.run(
                [*command, "packs", "--none", "--yes", "--json"], capture_output=True, text=True
            )
            assert refused.returncode != 0 and "stop all environment engines" in refused.stdout
            assert project.read_bytes() == original
            stop_engine(tmp_path, "main")
        for arguments, diagnostic in (
            (("--only", "unknown-pack"), "absent from snapshot"),
            (("--only", "shutdown-pack-62"), "requires pack"),
            (("--only", "shutdown-pack-64"), "requires capability"),
        ):
            configure(*arguments)
            with pytest.raises(EnvironmentError, match=diagnostic):
                start_engine(tmp_path, 1, "main", port, selected_packs=True)
            stop_engine(tmp_path, "main")
        preview = subprocess.run(
            [*command, "network", "offline", "--plan", "--json"], capture_output=True, text=True
        )
        assert preview.returncode == 0, preview.stdout + preview.stderr
        assert "runtimeOffline" not in json.loads(project.read_text())
        partner_import.unlink(missing_ok=True)
        for mode in ("offline", "online"):
            applied = subprocess.run(
                [*command, "network", mode, "--yes", "--json"], capture_output=True, text=True
            )
            assert applied.returncode == 0, applied.stdout + applied.stderr
            if mode == "offline":
                from dinkster_workers.catalog import catalog_path

                partner_catalog = catalog_path(
                    generation / "pack-sources/shutdown-pack-63/dinkster-pack.toml"
                )
                saved_catalog = partner_catalog.read_bytes()
                partner_catalog.unlink()
                try:
                    with pytest.raises(EnvironmentError, match="network online"):
                        start_engine(tmp_path, 1, "main", port)
                    assert not partner_import.exists()
                finally:
                    stop_engine(tmp_path, "main")
                    partner_catalog.write_bytes(saved_catalog)
            for _restart in range(2 if mode == "offline" else 1):
                running = start_engine(tmp_path, 1, "main", port)
                assert running.get("runtimeOffline", False) == (mode == "offline")
                asyncio.run(probe({61, 62, 63, 64}, offline=mode == "offline"))
                assert partner_import.exists() == (mode == "online")
                original = project.read_bytes()
                refused = subprocess.run(
                    [*command, "network", "online", "--yes", "--json"],
                    capture_output=True,
                    text=True,
                )
                assert refused.returncode != 0 and "stop all environment engines" in refused.stdout
                assert project.read_bytes() == original
                stop_engine(tmp_path, "main")
        assert {
            key: value
            for key, value in json.loads(project.read_text()).items()
            if key not in {"packSelection", "runtimeOffline"}
        } == settings
        assert (tmp_path / "current").read_text() == "1\n"
        assert all((generation / name).read_bytes() == data for name, data in retained.items())
    finally:
        stop_engine(tmp_path, "main")


@pytest.mark.parametrize(
    ("policy", "arguments"),
    [
        ([], []),
        ({"allow": "shutdown-pack-61"}, []),
        ({"allow": [True]}, []),
        ({"allow": ["Shutdown_pack_61"]}, []),
        ({"allow": ["shutdown-pack-61", "shutdown-pack-61"]}, []),
        ({"excludePartners": 1}, []),
        ({"unknown": True}, []),
        ({}, ["--only", "Shutdown_pack_61"]),
        ({}, ["--only", "shutdown-pack-61", "--only", "shutdown-pack-61"]),
    ],
)
def test_pack_selection_refuses_invalid_settings_without_saving(
    tmp_path: Path, policy: object, arguments: list[str]
) -> None:
    project = tmp_path / "environment.json"
    original = json.dumps(
        {
            "format": "dinkster.project/1",
            "channel": "nightly",
            "variant": "linux-cpu",
            "mirrors": [],
            "port": 8188,
            "packSelection": policy,
        }
    )
    project.write_text(original)
    refused = subprocess.run(
        [
            sys.executable,
            "-m",
            "dinkster_supervisor.environment",
            "--root",
            str(tmp_path),
            "packs",
            *arguments,
            "--yes",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert refused.returncode != 0
    assert "packSelection" in refused.stdout
    assert project.read_text() == original


@pytest.mark.parametrize(
    ("policy", "arguments"),
    [
        ({}, ["--generic", "0"]),
        ({}, ["--image", "-1"]),
        ([], []),
        ({"generic": True}, []),
        ({"media/image": "64"}, []),
        ({"media/audio": 0}, []),
        ({"media/unknown": 64}, []),
    ],
)
def test_upload_policy_refuses_invalid_settings_without_saving(
    tmp_path: Path, policy: object, arguments: list[str]
) -> None:
    project = tmp_path / "environment.json"
    original = json.dumps(
        {
            "format": "dinkster.project/1",
            "channel": "nightly",
            "variant": "linux-cpu",
            "mirrors": [],
            "port": 8188,
            "uploadLimits": policy,
        }
    )
    project.write_text(original)
    refused = subprocess.run(
        [
            sys.executable,
            "-m",
            "dinkster_supervisor.environment",
            "--root",
            str(tmp_path),
            "uploads",
            *arguments,
            "--yes",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert refused.returncode != 0
    assert "uploadLimits" in refused.stdout
    assert project.read_text() == original


@pytest.mark.parametrize(
    ("arguments", "diagnostic"),
    [
        (["--address", "0.0.0.0"], "specific interface address"),
        (["--address", "not-an-address"], "numeric IPv4 or IPv6"),
        (["--allow-host", "bad:80"], "without ports"),
        (["--allow-origin", "https://[broken"], "invalid allowed origin"),
        (["--tls-certificate", "missing.pem"], "both certificate and key"),
        (
            ["--tls-certificate", "missing.pem", "--tls-key", "missing-key.pem"],
            "cannot load HTTPS certificate/key",
        ),
    ],
)
def test_listener_refuses_invalid_settings_without_saving(
    tmp_path: Path, arguments: list[str], diagnostic: str
) -> None:
    original = json.dumps(
        {
            "format": "dinkster.project/1",
            "channel": "nightly",
            "variant": "linux-cpu",
            "mirrors": [],
            "port": 8188,
        }
    )
    project = tmp_path / "environment.json"
    project.write_text(original)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dinkster_supervisor.environment",
            "--root",
            str(tmp_path),
            "listener",
            *arguments,
            "--yes",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert diagnostic in result.stdout + result.stderr
    assert project.read_text() == original


@pytest.mark.parametrize("delayed_exit", [False, True])
def test_public_listener_binding_failure_releases_other_addresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delayed_exit: bool
) -> None:
    make_generation(tmp_path)
    if delayed_exit:
        wrapper = (
            "import time\n"
            "from dinkster_supervisor.supervisor import main\n"
            "try:\n"
            "    main()\n"
            "except Exception:\n"
            "    time.sleep(1)\n"
            "    raise\n"
        )
        if sys.platform == "darwin":
            from dinkster_supervisor.macos_process_family import launch_supervisor

            def launch_job(command, *args):
                launch_supervisor([command[0], "-c", wrapper, *command[3:]], *args)

            monkeypatch.setattr(
                "dinkster_supervisor.macos_process_family.launch_supervisor", launch_job
            )
        else:
            popen = subprocess.Popen

            def launch(command, **kwargs):
                return popen([command[0], "-c", wrapper, *command[3:]], **kwargs)

            monkeypatch.setattr("dinkster_supervisor.environment_runtime.subprocess.Popen", launch)
    with socket.create_server(("127.0.0.1", 0)) as occupied:
        port = occupied.getsockname()[1]
        (tmp_path / "environment.json").write_text(
            json.dumps({"publicListener": {"addresses": ["::1", "127.0.0.1"]}})
        )
        with pytest.raises(EnvironmentError, match="cannot bind public listener 127.0.0.1"):
            start_engine(tmp_path, 1, "main", port)
        with socket.create_server(("::1", port), family=socket.AF_INET6):
            pass
        assert engine_status(tmp_path)[0]["state"] == "stopped"


def test_run_uses_active_interpreter_and_preserves_command_io_and_exit(tmp_path: Path) -> None:
    make_generation(tmp_path, fake_serve=False)
    make_generation(tmp_path, 2, fake_serve=False)
    (tmp_path / "current").write_text("2\n")
    command = [
        sys.executable,
        "-m",
        "dinkster_supervisor.environment",
        "--root",
        str(tmp_path),
        "run",
        "--",
        "python",
        "-c",
        "import os,sys; print(sys.prefix); print(input()); "
        "print(os.environ['DINKSTER_EXECUTION_PYTHON'], file=sys.stderr); sys.exit(37)",
    ]
    result = subprocess.run(command, input="tool input\n", capture_output=True, text=True)
    target = tmp_path / "builds/2/python"
    assert result.returncode == 37
    assert result.stdout.splitlines() == [str(target), "tool input"]
    assert Path(result.stderr.strip()).is_relative_to(target)


@pytest.mark.parametrize(
    ("exclusions", "composition_succeeds"), [(True, False), (True, True), (False, True)]
)
def test_managed_node_exclusions_wait_for_terminal_composition(
    tmp_path: Path, exclusions: bool, composition_succeeds: bool
) -> None:
    make_generation(tmp_path)
    if exclusions:
        (tmp_path / "environment.json").write_text(
            json.dumps({"excludedNodeTypes": ["exclusion.unknown"]})
        )
    health = """counter = Path('health-requests')
            with counter.open('ab') as requests:
                requests.write(b'x')
            count = counter.stat().st_size
            if count < 3:
                body = {'ok': True, 'composition': {'done': 1, 'total': 2, 'phase': 'packs'}}
            elif COMPOSITION_SUCCEEDS:
                body = {'ok': True}
            else:
                print('unknown excluded node types: exclusion.unknown', flush=True)
                os._exit(37)""".replace("COMPOSITION_SUCCEEDS", repr(composition_succeeds))
    (tmp_path / "dinkster/serve.py").write_text(SERVE.replace("body = {'ok': True}", health))
    try:
        if composition_succeeds:
            assert start_engine(tmp_path, 1, "main", free_port())["state"] == "ready"
        else:
            with pytest.raises(EnvironmentError, match="unknown excluded node types"):
                start_engine(tmp_path, 1, "main", free_port())
        assert ((tmp_path / "health-requests").stat().st_size >= 3) is exclusions
    finally:
        stop_engine(tmp_path, "main")


def test_start_waits_for_cold_engine_readiness(tmp_path: Path) -> None:
    make_generation(tmp_path)
    (tmp_path / "dinkster/serve.py").write_text("import time\ntime.sleep(35)\n" + SERVE)
    try:
        result = start_engine(tmp_path, 1, "main", free_port())
        assert result["state"] == "ready"
        with urlopen(str(result["url"]) + "/api/health", timeout=5) as response:  # noqa: S310
            assert response.status == 200
    finally:
        stop_engine(tmp_path, "main", generation=1)


@pytest.mark.parametrize(("supervisor_exit", "output_size"), [(False, 7), (True, 70000)])
def test_start_failure_preserves_exit_code_and_current_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, supervisor_exit: bool, output_size: int
) -> None:
    make_generation(tmp_path)
    failure = (
        f"import sys; print('x' * {output_size}, flush=True); "
        "print('startup diagnostic', flush=True); "
        "print('startup stderr', file=sys.stderr, flush=True); sys.exit(37)"
    )
    if supervisor_exit:
        popen = subprocess.Popen

        def launch(command, **kwargs):
            return popen([sys.executable, "-c", failure], **kwargs)

        if sys.platform == "darwin":
            from dinkster_supervisor.macos_process_family import launch_supervisor

            def launch_job(command, *args):
                launch_supervisor([sys.executable, "-c", failure], *args)

            monkeypatch.setattr(
                "dinkster_supervisor.macos_process_family.launch_supervisor", launch_job
            )
        else:
            monkeypatch.setattr("dinkster_supervisor.environment_runtime.subprocess.Popen", launch)
    else:
        (tmp_path / "dinkster/serve.py").write_text(failure)
    runtime = tmp_path / "runtime/1"
    runtime.mkdir(parents=True)
    (runtime / "main.log").write_text("old launch output must not appear\n")
    with pytest.raises(EnvironmentError) as raised:
        start_engine(tmp_path, 1, "main", free_port())
    message = str(raised.value)
    assert "exit code 37" in message
    assert "startup diagnostic" in message
    assert "startup stderr" in message
    assert "old launch output" not in message
    assert ("[earlier output truncated]" in message) is (output_size > 64 * 1024)
    assert len(message) < 68 * 1024


@pytest.mark.parametrize("host_contamination", [False, True])
def test_managed_generation_persists_folder_grant_across_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host_contamination: bool,
    shared_model_folders: dict[str, Path],
) -> None:
    from dinkster_registry import LockedPack, Lockfile, build_artifact
    from dinkster_workers.doctor import prepare_catalog
    from test_serve import write_shutdown_pack

    from dinkster.generation_packs import stage_generation_packs

    monkeypatch.delenv("DINKSTER_LIBRARY_ROOT", raising=False)
    root = tmp_path / "project"
    generation = root / "builds/1"
    generation.mkdir(parents=True)
    manifest = write_shutdown_pack(tmp_path / "source", 57)
    archive = tmp_path / "pack.zip"
    digest = build_artifact(manifest.parent, archive)
    pack = "shutdown-pack-57"
    lock = Lockfile.of([LockedPack(pack, "1.2.3", digest, "dinkster", ("shutdown57",))])
    stage_generation_packs(generation, lock, {pack: archive})
    source = generation / "pack-sources" / pack
    assert prepare_catalog(
        source / "dinkster-pack.toml", environment={"PYTHONPATH": str(source)}
    ).ok
    make_generation(root, fake_serve=False)
    if host_contamination:
        shadow = root / "dinkster"
        shadow.mkdir()
        (shadow / "__init__.py").write_text("raise RuntimeError('host engine shadow imported')\n")
        monkeypatch.setenv("PYTHONPATH", str(root))
    record = json.loads((generation / "build.json").read_text())
    record["inProcessPacks"] = [pack]
    (generation / "build.json").write_text(json.dumps(record))
    granted = tmp_path / "user files"
    granted.mkdir()
    port = free_port()
    url = f"http://127.0.0.1:{port}/api/mounts"
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<title>Explicit frontend</title>")
    try:
        start_engine(root, 1, "main", port, frontend_root=str(frontend))
        with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:  # noqa: S310
            assert b"<title>Explicit frontend</title>" in response.read()
        with pytest.raises(EnvironmentError, match="stop the engine first"):
            start_engine(root, 1, "main", port, frontend_dev="http://127.0.0.1:5173")
        with urlopen(url, timeout=5) as response:  # noqa: S310
            initial = json.load(response)
        assert initial["mountChangesAllowed"] is True
        rows = {row["id"]: row for row in initial["mounts"]}
        assert set(rows) == {"imports", "output", *shared_model_folders}
        for mount_id, path in shared_model_folders.items():
            assert rows[mount_id]["path"] == str(path)
            assert rows[mount_id]["mode"] == "readwrite"
            assert rows[mount_id]["source"] == "shared-models"
        assert initial["outputMount"] == "output"
        request = Request(
            url,
            data=json.dumps({"id": "chosen", "path": str(granted)}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:  # noqa: S310
            assert response.status == 201
            assert json.load(response)["mode"] == "read"
        stop_engine(root, "main")
        start_engine(root, 1, "main", port)
        with urlopen(url, timeout=5) as response:  # noqa: S310
            listing = json.load(response)
        assert listing["mountChangesAllowed"] is True
        assert {row["id"] for row in listing["mounts"]} == {
            "imports",
            "output",
            "chosen",
            *shared_model_folders,
        }
        selected = next(row for row in listing["mounts"] if row["id"] == "chosen")
        assert selected["path"] == str(granted)
        assert selected["mode"] == "read"
        assert listing["restartRequired"] is False
    finally:
        stop_engine(root, "main")


@pytest.mark.parametrize("frontend_kind", ["frontendRoot", "frontendDev"])
@pytest.mark.parametrize("explicit_override", [False, True])
def test_managed_development_frontend_survives_restart_and_serves_edits(
    tmp_path: Path, frontend_kind: str, explicit_override: bool
) -> None:
    import asyncio
    import threading

    import aiohttp
    from aiohttp import web
    from dinkster_registry import LockedPack, Lockfile, build_artifact
    from dinkster_workers.doctor import prepare_catalog
    from test_serve import write_shutdown_pack

    from dinkster.generation_packs import stage_generation_packs

    root = tmp_path / "project"
    generation = root / "builds/1"
    generation.mkdir(parents=True)
    manifest = write_shutdown_pack(tmp_path / "pack", 58)
    archive = tmp_path / "pack.zip"
    digest = build_artifact(manifest.parent, archive)
    lock = Lockfile.of(
        [LockedPack("shutdown-pack-58", "1.2.3", digest, "dinkster", ("shutdown58",))]
    )
    stage_generation_packs(generation, lock, {"shutdown-pack-58": archive})
    source = generation / "pack-sources/shutdown-pack-58"
    assert prepare_catalog(
        source / "dinkster-pack.toml", environment={"PYTHONPATH": str(source)}
    ).ok
    make_generation(root, fake_serve=False)
    record = json.loads((generation / "build.json").read_text())
    record["inProcessPacks"] = ["shutdown-pack-58"]
    (generation / "build.json").write_text(json.dumps(record))
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    page = frontend / "index.html"
    page.write_text("<html>selected development frontend</html>")
    loop = asyncio.new_event_loop()
    disconnected = threading.Event()
    outside_contacted = threading.Event()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()

    async def serve() -> web.AppRunner:
        async def index(_request: web.Request) -> web.Response:
            return web.Response(text=page.read_text(), content_type="text/html")

        async def websocket(request: web.Request) -> web.WebSocketResponse:
            socket = web.WebSocketResponse(
                protocols=("vite-hmr",) if request.path == "/hmr" else ()
            )
            await socket.prepare(request)
            await socket.send_json({"type": "connected"})
            async for message in socket:
                if message.type == aiohttp.WSMsgType.TEXT:
                    await socket.send_str(f"updated:{message.data}")
                elif message.type == aiohttp.WSMsgType.BINARY:
                    await socket.send_bytes(b"preview:" + message.data)
            disconnected.set()
            return socket

        async def redirect(_request: web.Request) -> web.Response:
            raise web.HTTPFound(f"http://localhost:{dev_port}/outside")

        async def path(request: web.Request) -> web.Response:
            if request.path == "/outside":
                outside_contacted.set()
            return web.Response(text=f"approved:{request.path}")

        app = web.Application()
        app.router.add_get("/", index)
        app.router.add_get("/hmr", websocket)
        app.router.add_get("/plain-hmr", websocket)
        app.router.add_get("/redirect", redirect)
        app.router.add_get("/{path:.*}", path)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", dev_port).start()
        return runner

    dev_port = free_port()
    runner = asyncio.run_coroutine_threadsafe(serve(), loop).result(timeout=10)
    selection = str(frontend) if frontend_kind == "frontendRoot" else f"http://127.0.0.1:{dev_port}"
    saved_selection = (
        (
            {"frontendDev": "http://192.0.2.1:5173"}
            if frontend_kind == "frontendRoot"
            else {"frontendRoot": str(tmp_path / "unselected-frontend")}
        )
        if explicit_override
        else {frontend_kind: selection}
    )
    (generation / "development-sources.json").write_text(
        json.dumps({"format": "dinkster.development-sources/1", "sources": [], **saved_selection})
    )
    port = free_port()

    async def exchange(path: str, protocol: str | None) -> None:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as client:
            async with client.ws_connect(
                f"http://127.0.0.1:{port}/{path}", protocols=("other", "vite-hmr")
            ) as socket:
                assert socket.protocol == protocol
                assert await socket.receive_json() == {"type": "connected"}
                await socket.send_str("edited module")
                assert (await socket.receive()).data == "updated:edited module"
                await socket.send_bytes(b"frame")
                assert (await socket.receive()).data == b"preview:frame"
            async with client.get(f"http://127.0.0.1:{port}/http://192.0.2.1/x") as response:
                assert await response.text() == "approved:/http://192.0.2.1/x"
            with pytest.raises(aiohttp.WSServerHandshakeError):
                await client.ws_connect(f"http://127.0.0.1:{port}/redirect")

    try:
        for text in ("first selection", "edited frontend after restart"):
            page.write_text(f"<html>{text}</html>")
            start_engine(
                root,
                1,
                "main",
                port,
                frontend_root=selection
                if explicit_override and frontend_kind == "frontendRoot"
                else None,
                frontend_dev=selection
                if explicit_override and frontend_kind == "frontendDev"
                else None,
            )
            with pytest.raises(EnvironmentError, match="stop the engine first"):
                start_engine(root, 1, "main", port, frontend_root=str(tmp_path / "changed"))
            with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:  # noqa: S310
                assert response.read().decode() == f"<html>{text}</html>"
            if frontend_kind == "frontendDev":
                for path, protocol in (("hmr", "vite-hmr"), ("plain-hmr", None)):
                    disconnected.clear()
                    asyncio.run(exchange(path, protocol))
                    assert disconnected.wait(5), "dev-server socket survived client disconnect"
                assert not outside_contacted.is_set(), "redirect contacted an unapproved origin"
            if explicit_override:
                from dinkster_supervisor.environment_transactions import (
                    begin_activation,
                    recover_activation,
                )

                journal = begin_activation(root, 2, "update")
                stop_engine(root, "main")
                recover_activation(root, journal)
                with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:  # noqa: S310
                    assert response.read().decode() == f"<html>{text}</html>"
            stop_engine(root, "main")
    finally:
        stop_engine(root, "main")
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()


@pytest.mark.parametrize(
    "selection",
    [
        {"frontendDev": "http://192.0.2.1:5173"},
        {"frontendDev": "http://localhost:5173/path", "allowFrontendDevRemote": True},
        {"frontendRoot": "/bundle", "frontendDev": "http://localhost:5173"},
    ],
)
def test_development_frontend_refuses_unsafe_saved_selection_before_start(
    tmp_path: Path, selection: dict[str, object]
) -> None:
    root = tmp_path / "project"
    make_generation(root)
    (root / "builds/1/development-sources.json").write_text(
        json.dumps({"format": "dinkster.development-sources/1", "sources": [], **selection})
    )
    with pytest.raises(EnvironmentError, match="frontend"):
        start_engine(root, 1, "main", free_port())
    assert engine_status(root) == ()
    assert not (root / "runtime/1").exists()


@pytest.mark.parametrize(
    ("redirecting_launcher", "health_body_timeout"),
    [(False, False), (True, False), (False, True)],
)
def test_start_status_idempotence_and_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    redirecting_launcher: bool,
    health_body_timeout: bool,
) -> None:
    monkeypatch.setattr(
        "dinkster_supervisor.environment_runtime.secrets.token_urlsafe", lambda _: "-token"
    )
    if health_body_timeout:
        original_read = http.client.HTTPResponse.read

        def read(response, *args, **kwargs):
            if response.geturl().endswith("/api/health"):
                monkeypatch.setattr(http.client.HTTPResponse, "read", original_read)
                raise TimeoutError("health response body stalled")
            return original_read(response, *args, **kwargs)

        monkeypatch.setattr(http.client.HTTPResponse, "read", read)
    if redirecting_launcher:
        launcher = tmp_path / "launcher.py"
        launcher.write_text(
            "import os, sys; os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
            if sys.platform == "darwin"
            else "import subprocess, sys\n"
            "with subprocess.Popen([sys.executable, *sys.argv[1:]]) as child:\n"
            "    sys.exit(child.wait())\n"
        )
        original = subprocess.Popen

        def launch(command, **kwargs):
            return original([command[0], str(launcher), *command[1:]], **kwargs)

        if sys.platform == "darwin":
            from dinkster_supervisor.macos_process_family import launch_supervisor

            def launch_job(command, *args):
                launch_supervisor([command[0], str(launcher), *command[1:]], *args)

            monkeypatch.setattr(
                "dinkster_supervisor.macos_process_family.launch_supervisor", launch_job
            )
        else:
            monkeypatch.setattr("dinkster_supervisor.environment_runtime.subprocess.Popen", launch)
    make_generation(tmp_path)
    port = free_port()
    try:
        row = start_engine(tmp_path, 1, "main", port, ("0", "GPU-aB12"))
        assert row == {
            "name": "main",
            "generation": 1,
            "state": "ready",
            "url": f"http://127.0.0.1:{port}",
            "gpus": ["0", "GPU-aB12"],
        }
        with urlopen(f"http://127.0.0.1:{port}/api/cuda") as response:  # noqa: S310
            assert json.loads(response.read()) == {"cuda": "0,GPU-aB12"}
        with urlopen(f"http://127.0.0.1:{port}/api/generation") as response:  # noqa: S310
            assert json.loads(response.read()) == {"generation": str(tmp_path / "builds/1")}
        with urlopen(f"http://127.0.0.1:{port}/supervisor/status") as response:  # noqa: S310
            supervisor_pid = json.loads(response.read())["supervisor"]["pid"]
        persisted = json.loads((tmp_path / "runtime/1/main.json").read_bytes())
        assert persisted["pid"] == supervisor_pid and supervisor_pid > 0
        with monkeypatch.context() as patch:

            def unavailable_tls(*args, **kwargs):
                raise AssertionError("loopback HTTP must not load TLS certificates")

            patch.setattr("ssl._create_default_https_context", unavailable_tls)
            assert engine_status(tmp_path) == (row,)
        assert start_engine(tmp_path, 1, "main", port, ("0", "GPU-aB12")) == row
        with pytest.raises(EnvironmentError, match="stop it first"):
            start_engine(tmp_path, 1, "main", free_port())
        require_idle(tmp_path)
    finally:
        stop_engine(tmp_path, "main")
    stopped = engine_status(tmp_path)[0]
    assert stopped["state"] == "stopped"
    assert stopped["gpus"] == ["0", "GPU-aB12"]


def test_stop_engine_closes_open_http_stream(tmp_path: Path) -> None:
    make_generation(tmp_path)
    serve = tmp_path / "dinkster/serve.py"
    serve.write_text(
        SERVE.replace(
            "        if self.path == '/api/health':",
            """        if self.path == '/api/stream':
            import time
            self.send_response(200)
            self.end_headers()
            while True:
                self.wfile.write(b'streaming')
                self.wfile.flush()
                time.sleep(0.05)
        if self.path == '/api/health':""",
        )
    )
    port = free_port()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    response = None
    try:
        start_engine(tmp_path, 1, "main", port)
        connection.request("GET", "/api/stream")
        response = connection.getresponse()
        assert response.status == 200
        assert response.read(9) == b"streaming"
        assert engine_status(tmp_path)[0]["gpus"] == []
        stop_engine(tmp_path, "main")
        assert engine_status(tmp_path)[0]["state"] == "stopped"
    finally:
        if response is not None:
            response.close()
        connection.close()
        stop_engine(tmp_path, "main")


def test_managed_engine_survives_launcher_session_exit(tmp_path: Path) -> None:
    make_generation(tmp_path)
    port = free_port()
    command = [
        sys.executable,
        "-c",
        "import json,sys; from pathlib import Path; "
        "from dinkster_supervisor.environment_runtime import start_engine; "
        "print(json.dumps(start_engine(Path(sys.argv[1]), 1, 'main', int(sys.argv[2]))))",
        str(tmp_path),
        str(port),
    ]
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name != "nt",
    ) as launcher:
        output, error = launcher.communicate(timeout=30)
        assert launcher.returncode == 0, error
    row = json.loads(output)
    try:
        if os.name != "nt":
            # Reproduce a terminal/WSL launcher ending its session after returning readiness.
            try:
                os.killpg(launcher.pid, signal.SIGHUP)
            except ProcessLookupError:
                pass
        for _ in range(5):
            assert engine_status(tmp_path) == (row,)
            with urlopen(row["url"] + "/api/health", timeout=5) as response:  # noqa: S310
                assert json.loads(response.read()) == {"ok": True}
            time.sleep(0.05)
    finally:
        stop_engine(tmp_path, "main")
    assert engine_status(tmp_path)[0]["state"] == "stopped"


@pytest.mark.parametrize("available", [False, True])
def test_doctor_sandbox_uses_selected_generation_readiness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], available: bool
) -> None:
    from dinkster_supervisor.environment import main

    make_generation(tmp_path)
    site = Path(
        subprocess.check_output(
            [
                str(interpreter(tmp_path / "builds/1")),
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            text=True,
        ).strip()
    )
    workers = site / "dinkster_workers"
    workers.mkdir()
    (workers / "__init__.py").write_text("")
    report = {
        "format": "dinkster.sandbox-readiness/1",
        "available": available,
        "detail": "selected engine prerequisite",
    }
    (workers / "doctor.py").write_text(f"import json; print(json.dumps({report!r}))\n")
    with pytest.raises(SystemExit) as exit:
        main(["--root", str(tmp_path), "doctor", "--sandbox", "--json"])
    assert exit.value.code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["sandbox"] == report
    assert engine_status(tmp_path) == ()
    (workers / "doctor.py").write_text("raise RuntimeError('unsupported probe')\n")
    with pytest.raises(SystemExit) as exit:
        main(["--root", str(tmp_path), "doctor", "--sandbox", "--json"])
    assert exit.value.code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["sandbox"]["available"] is False
    assert "Update or repair" in result["sandbox"]["detail"]


def test_sandbox_selection_reaches_engine_and_cannot_reuse_an_unsandboxed_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_generation(tmp_path)
    monkeypatch.setattr(
        "dinkster_supervisor.environment_runtime.sandbox_readiness",
        lambda *_: {"available": True, "detail": "available"},
    )
    port = free_port()
    try:
        for sandboxed in (False, True):
            start_engine(tmp_path, 1, "main", port, sandbox_packs=sandboxed)
            with urlopen(f"http://127.0.0.1:{port}/api/sandbox") as response:  # noqa: S310
                assert json.loads(response.read()) == {"sandboxPacks": sandboxed}
            with pytest.raises(EnvironmentError, match="sandbox selection changed"):
                start_engine(tmp_path, 1, "main", port, sandbox_packs=not sandboxed)
            binding = engine_assignments(tmp_path, 1)[0]
            assert binding["sandboxPacks"] is sandboxed
            stop_engine(tmp_path, "main")
        monkeypatch.setattr(
            "dinkster_supervisor.environment_runtime.sandbox_readiness",
            lambda *_: {"available": False, "detail": "install bubblewrap"},
        )
        with pytest.raises(EnvironmentError, match="install bubblewrap"):
            start_engine(tmp_path, 1, "main", port, sandbox_packs=True)
        assert engine_status(tmp_path)[0]["state"] == "stopped"
    finally:
        stop_engine(tmp_path, "main")


def test_start_engine_uses_generation_built_from_a_virtual_environment(tmp_path: Path) -> None:
    launcher = tmp_path / "launcher"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(launcher)
    python = launcher / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    root = tmp_path / "project"
    subprocess.run(
        [
            str(python),
            "-c",
            "import sys; from pathlib import Path; "
            "from test_environment_runtime import make_generation; "
            "make_generation(Path(sys.argv[1]))",
            str(root),
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(str(Path(path).resolve()) for path in sys.path if path),
        },
        check=True,
        capture_output=True,
        timeout=30,
    )
    start_engine(root, 1, "main", free_port())
    try:
        assert engine_status(root)[0]["state"] == "ready"
        with urlopen(str(engine_status(root)[0]["url"]) + "/api/generation") as response:  # noqa: S310
            assert json.loads(response.read()) == {"generation": str(root / "builds/1")}
    finally:
        stop_engine(root, "main")
    assert engine_status(root)[0]["state"] == "stopped"


def test_stop_refuses_wrong_instance(tmp_path: Path) -> None:
    make_generation(tmp_path)
    port = free_port()
    start_engine(tmp_path, 1, "main", port)
    record_path = tmp_path / "runtime" / "1" / "main.json"
    original = json.loads(record_path.read_text())
    record = dict(original)
    record["instance"] = "not-the-supervisor"
    record_path.write_text(json.dumps(record))
    try:
        with pytest.raises(EnvironmentError, match="refusing to stop"):
            stop_engine(tmp_path, "main")
    finally:
        record_path.write_text(json.dumps(original))
        stop_engine(tmp_path, "main")


@pytest.mark.skipif(os.name != "posix", reason="SIGTERM cleanup is a POSIX contract")
def test_sigterm_stops_managed_engine_and_allows_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisors: list[subprocess.Popen] = []
    if sys.platform != "darwin":
        original_popen = subprocess.Popen

        def launch(command, **kwargs):
            process = original_popen(command, **kwargs)
            if command[1:3] == ["-m", "dinkster_supervisor.supervisor"]:
                supervisors.append(process)
            return process

        monkeypatch.setattr("dinkster_supervisor.environment_runtime.subprocess.Popen", launch)

    def process_field(pid: int, field: str) -> str:
        result = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", f"{field}="],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode in {0, 1}
        return result.stdout.strip()

    def child_alive(pid: int) -> bool:
        state = process_field(pid, "stat")
        return bool(state) and not state.startswith("Z")

    monkeypatch.setenv("COLUMNS", "80")
    owners_before = macos_runtime_owners(tmp_path)
    make_generation(tmp_path)
    port = free_port()
    child_pid = 0
    child_command = ""
    try:
        row = start_engine(tmp_path, 1, "main", port)
        record = json.loads((tmp_path / "runtime/1/main.json").read_text())
        with urlopen(f"http://127.0.0.1:{port}/supervisor/status") as response:  # noqa: S310
            status = json.loads(response.read())
        assert status["instance"] == record["instance"]
        supervisor_pid = status["supervisor"]["pid"]
        assert supervisor_pid == record["pid"]
        child_pid = status["engine"]["pid"]
        child_command = process_field(child_pid, "args")
        assert str(tmp_path / "builds/1") in child_command
        lifecycle = next((tmp_path / "runtime/1/main.process").glob("*/state.json"))
        assert json.loads(lifecycle.read_text())["state"] == "running"

        os.kill(supervisor_pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while (
            child_alive(child_pid) or child_alive(supervisor_pid)
        ) and time.monotonic() < deadline:
            time.sleep(0.05)
        receipt = json.loads(lifecycle.read_text())["state"]
        assert not child_alive(child_pid), f"engine child survived SIGTERM; receipt={receipt}"
        assert not child_alive(supervisor_pid), "supervisor survived SIGTERM"
        if sys.platform == "darwin":
            label = (
                "org.dinkster.supervisor."
                + hashlib.sha256(str(lifecycle.parent).encode()).hexdigest()
            )
            uid = getattr(os, "getuid")()  # noqa: B009 - macOS-only API.
            job = subprocess.run(
                ["/bin/launchctl", "print", f"user/{uid}/{label}"],
                capture_output=True,
                text=True,
                check=True,
            )
            assert "last exit code = 0" in {line.strip() for line in job.stdout.splitlines()}
        else:
            assert supervisors[0].wait(timeout=10) == 0
        assert receipt == "stopped"
        assert engine_status(tmp_path)[0]["state"] == "stopped"
        assert start_engine(tmp_path, 1, "main", port) == row
        stop_engine(tmp_path, "main")
        assert macos_runtime_owners(tmp_path) == owners_before
    finally:
        stop_engine(tmp_path, "main")
        for process in supervisors:
            process.wait(timeout=10)
        if child_pid and child_alive(child_pid):
            assert process_field(child_pid, "args") == child_command
            os.kill(child_pid, signal.SIGTERM)
            deadline = time.monotonic() + 10
            while child_alive(child_pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not child_alive(child_pid), "owned fixture child did not exit during cleanup"


@pytest.mark.parametrize("live", [False, True])
def test_unverified_supervisor_exit_does_not_prove_idle(tmp_path: Path, live: bool) -> None:
    make_generation(tmp_path)
    with subprocess.Popen([sys.executable, "-c", "pass"]) as exited:
        assert exited.wait(timeout=10) == 0
    runtime = tmp_path / "runtime" / "1"
    runtime.mkdir(parents=True)
    record = {
        "name": "main",
        "generation": 1,
        "port": free_port(),
        "gpus": [],
        "instance": "original",
        "shutdownToken": "private",
        "pid": os.getpid() if live else exited.pid,
        "state": "stopped" if live else "ready",
    }
    record["url"] = f"http://127.0.0.1:{record['port']}"
    (runtime / "main.json").write_text(json.dumps(record))
    assert engine_status(tmp_path)[0]["state"] == "failed"
    with pytest.raises(EnvironmentError, match="cannot verify"):
        require_idle(tmp_path)
    with pytest.raises(EnvironmentError, match="not safely reachable"):
        start_engine(tmp_path, 1, "main", record["port"])


@pytest.mark.skipif(sys.platform != "linux", reason="Linux procfs process lifetime contract")
def test_supervisor_exit_during_status_read_refuses_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_generation(tmp_path)
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    ) as child:
        stat_path = Path("/proc") / str(child.pid) / "stat"
        runtime = tmp_path / "runtime/1"
        runtime.mkdir(parents=True)
        (runtime / "main.json").write_text(
            json.dumps(
                {
                    "name": "main",
                    "generation": 1,
                    "port": 1,
                    "gpus": [],
                    "instance": "exiting-supervisor",
                    "shutdownToken": "private",
                    "pid": child.pid,
                    "state": "ready",
                    "url": "http://127.0.0.1:1",
                }
            )
        )
        original = Path.read_text

        def read_text(path: Path, *args, **kwargs) -> str:
            if path != stat_path:
                return original(path, *args, **kwargs)
            with path.open(encoding="utf-8") as stream:
                assert child.stdin is not None
                child.stdin.close()
                assert child.wait(timeout=10) == 0
                return stream.read()

        monkeypatch.setattr(Path, "read_text", read_text)
        assert engine_status(tmp_path)[0]["state"] == "failed"
        with pytest.raises(EnvironmentError, match="cannot verify"):
            require_idle(tmp_path)


def test_zero_pid_recovers_identity_for_authenticated_stop(tmp_path: Path) -> None:
    make_generation(tmp_path)
    port = free_port()
    start_engine(tmp_path, 1, "main", port)
    record_path = tmp_path / "runtime" / "1" / "main.json"
    record = json.loads(record_path.read_text())
    record["pid"] = 0
    record_path.write_text(json.dumps(record))
    try:
        assert engine_status(tmp_path)[0]["state"] == "ready"
        assert json.loads(record_path.read_text())["pid"] == 0
    finally:
        stop_engine(tmp_path, "main")


def test_require_idle_refuses_running_or_malformed_queue(tmp_path: Path) -> None:
    make_generation(tmp_path)
    port = free_port()
    start_engine(tmp_path, 1, "main", port)
    try:
        (tmp_path / "queue.json").write_text('{"idle": false}')
        with pytest.raises(EnvironmentError, match="Finish or cancel"):
            require_idle(tmp_path)
        for malformed in ({"not": "a queue"}, {"idle": "true"}, {"idle": 1}):
            (tmp_path / "queue.json").write_text(json.dumps(malformed))
            with pytest.raises(EnvironmentError, match="cannot verify"):
                require_idle(tmp_path)
    finally:
        stop_engine(tmp_path, "main")


def test_generations_keep_assignments_but_take_turns_on_the_library(tmp_path: Path) -> None:
    make_generation(tmp_path, 1)
    make_generation(tmp_path, 2)
    first_port, second_port = free_port(), free_port()
    first = start_engine(tmp_path, 1, "main", first_port)
    try:
        with pytest.raises(EnvironmentError, match="already in use by engine main"):
            start_engine(tmp_path, 2, "main", second_port, ("GPU-aB12",))
        assert engine_status(tmp_path) == (first,)
        stop_engine(tmp_path, "main", generation=1)
        second = start_engine(tmp_path, 2, "main", second_port, ("GPU-aB12",))
        try:
            assert engine_status(tmp_path) == ({**first, "state": "stopped"},)
            assert engine_assignments(tmp_path, 2) == [
                {
                    "name": "main",
                    "port": second_port,
                    "gpus": ["GPU-aB12"],
                    "running": True,
                    "sandboxPacks": False,
                }
            ]
            require_idle(tmp_path)
            (tmp_path / "current").write_text("2\n")
            assert engine_status(tmp_path) == (second,)
            stop_engine(tmp_path, "main", generation=1)
            assert engine_status(tmp_path) == (second,)
            assert engine_assignments(tmp_path, 1) == [
                {
                    "name": "main",
                    "port": first_port,
                    "gpus": [],
                    "running": False,
                    "sandboxPacks": False,
                }
            ]
        finally:
            stop_engine(tmp_path, "main", generation=2)
    finally:
        stop_engine(tmp_path, "main", generation=1)


def test_cancelled_delayed_supervisor_cannot_spawn_its_engine(tmp_path: Path) -> None:
    lifecycle = tmp_path / "lifecycle"
    prepare_process(lifecycle, "cancelled-launch")
    assert not cancel_pending_process(lifecycle, "cancelled-launch")
    sentinel = tmp_path / "engine-started"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dinkster_supervisor.supervisor",
            "--instance=cancelled-launch",
            "--process-state",
            str(lifecycle),
            "--",
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
            str(sentinel),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert not sentinel.exists()


def test_failed_candidate_listener_proves_cleanup_and_can_restart(tmp_path: Path) -> None:
    make_generation(tmp_path, 1)
    make_generation(tmp_path, 2)
    port = free_port()
    prior = start_engine(tmp_path, 1, "main", port)
    try:
        with pytest.raises(EnvironmentError, match="before becoming ready"):
            start_engine(tmp_path, 2, "main", port)
        stop_engine(tmp_path, "main", generation=2)
        assert engine_status(tmp_path) == (prior,)
        stop_engine(tmp_path, "main", generation=1)
        candidate = start_engine(tmp_path, 2, "main", free_port())
        assert candidate["generation"] == 2 and candidate["state"] == "ready"
    finally:
        stop_engine(tmp_path, "main", generation=1)
        stop_engine(tmp_path, "main", generation=2)
