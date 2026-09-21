"""Persisted declarations use real doctor imports and real lazy worker execution."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_engine import ExecutionError, GraphValidationError
from dinkster_graph import Graph, GraphNode, Link, RegionNode, RegionOutput, TypedLiteral
from dinkster_registry import Lockfile
from dinkster_schema import TypeExpr
from dinkster_server import create_app
from dinkster_workers import DoctorReport, diagnose, load_manifest
from dinkster_workers.catalog import catalog_path, read_catalog
from dinkster_workers.doctor import prepare_catalog

from dinkster.compose import CompositionError, PackSpec, ServingComposer
from dinkster.installer import Installer, lock_local_pack
from dinkster.lazy_worker import LazyWorker


def write_pack(root: Path, name: str, value: int = 7) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        f'[pack]\nname = "{name}"\n[pack.sandbox]\n[pack.entry]\nnodes = "{name}_nodes:NODES"\n'
    )
    (root / f"{name}_nodes.py").write_text(
        "from dinkster_api.v1 import Node, NodeSchema, OutputSpec, TypeExpr\n"
        "class Constant(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        f'        return NodeSchema(node_type="{name}.constant", '
        'outputs=(OutputSpec("value", TypeExpr.concrete("core.int")),))\n'
        "    @classmethod\n"
        "    def execute(cls):\n"
        f"        return cls.outputs(value={value})\n"
        "NODES = [Constant]\n"
    )
    return manifest


@pytest.mark.parametrize("import_path", [None, "/api/composition", "/api/nodes"])
def test_boot_benchmark_rejects_swallowed_http_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, import_path: str | None
) -> None:
    from tools import benchmark_schema_catalog as benchmark

    injected = f"""
@serve.web.middleware
async def import_during_response(request, handler):
    response = await handler(request)
    if request.path == {import_path!r}:
        try:
            import torch
        except AssertionError:
            pass
    return response

create_app = serve.create_app
def with_import_probe(*args, **kwargs):
    app = create_app(*args, **kwargs)
    app.middlewares.append(import_during_response)
    return app
serve.create_app = with_import_probe
serve.main()
"""
    monkeypatch.setattr(benchmark, "ENTRY", benchmark.ENTRY.replace("serve.main()", injected))
    install = benchmark.install_packs(tmp_path, 0)
    if import_path is None:
        assert 0 < asyncio.run(benchmark.measure(tmp_path, install, 0)) < benchmark.BUDGET_SECONDS
    else:
        with pytest.raises(AssertionError, match="BLOCKED_EXECUTION_IMPORT torch"):
            asyncio.run(benchmark.measure(tmp_path, install, 0))


@pytest.mark.parametrize(
    ("line", "state"),
    [
        (b"BENCH_STARTED 123\n", "bound=False, composed=False"),
        (b"BENCH_BOUND 123\n", "bound=True, composed=False"),
        (b"0 node types composed\n", "bound=False, composed=True"),
    ],
)
def test_boot_timeout_retains_startup_and_cleanup_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: bytes, state: str
) -> None:
    from tools import benchmark_schema_catalog as benchmark

    error = TimeoutError("startup deadline")
    proc = SimpleNamespace(
        pid=123,
        returncode=None,
        stdout=SimpleNamespace(readline=AsyncMock(side_effect=[line, error])),
        terminate=Mock(),
        communicate=AsyncMock(return_value=(b"trailing child stack\n", None)),
    )
    monkeypatch.setattr(benchmark.asyncio, "create_subprocess_exec", AsyncMock(return_value=proc))
    monkeypatch.setattr(benchmark, "terminate_children", lambda _pid: [])
    monkeypatch.setattr(benchmark.psutil, "wait_procs", lambda _children, timeout: ([], []))
    with pytest.raises(TimeoutError, match="startup deadline") as raised:
        asyncio.run(benchmark.measure(tmp_path, tmp_path / "install", 0))
    assert raised.value is error
    notes = "\n".join(error.__notes__)
    assert f"stage=startup stdout, {state}" in notes
    assert "subprocess created" in notes
    assert line.decode().strip() in notes
    assert "remaining server output:\ntrailing child stack" in notes
    proc.terminate.assert_called_once_with()
    proc.communicate.assert_awaited_once_with()


@pytest.mark.parametrize("error", [TimeoutError("deadline"), AssertionError("blocked import")])
def test_boot_baseline_keeps_completed_samples_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    from tools import benchmark_schema_catalog as benchmark

    measure = AsyncMock(side_effect=[0.5, error])
    install = Mock(side_effect=lambda root, count: root / "install")
    monkeypatch.setattr(benchmark, "measure", measure)
    monkeypatch.setattr(benchmark, "install_packs", install)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--counts", "0", "10", "--repeats", "20"])
    with pytest.raises(type(error)) as raised:
        benchmark.main()
    assert raised.value is error
    header, completed, failed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert header["budget_s"] == benchmark.BUDGET_SECONDS
    assert completed["packs"] == failed["packs"] == 0
    assert completed["sample"] == 1 and completed["outcome"] == "passed"
    assert completed["seconds"] == 0.5 and completed["worker_starts"] == 0
    assert failed["sample"] == 2 and failed["outcome"] == "failed"
    assert failed["error"] == type(error).__name__
    assert failed["wall_s_including_cleanup"] >= 0
    assert "seconds" not in failed and "worker_starts" not in failed
    assert measure.await_count == 2
    install.assert_called_once()


@pytest.mark.parametrize("in_process", [False, True])
@pytest.mark.parametrize("selected", [0, 1])
def test_catalog_serves_before_workers_and_invocation_starts_only_owner(
    tmp_path: Path, in_process: bool, selected: int
) -> None:
    paths = [write_pack(tmp_path / name, name) for name in ("catalogfirst", "catalogsecond")]
    for path in paths:
        report = diagnose(path)
        assert report.ok, report
        assert read_catalog(load_manifest(path)) is not None

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            for path in paths:
                await composer.add_pack(
                    PackSpec(
                        path,
                        in_process=in_process,
                        env={"PYTHONPATH": str(path.parent)},
                        require_catalog=True,
                    )
                )
            workers = [record.worker for record in composer._records.values()]
            assert all(isinstance(worker, LazyWorker) and worker.cold for worker in workers)
            assert not any(worker.alive for worker in workers)
            assert "catalogfirst_nodes" not in sys.modules
            composition = composer.composition
            app = create_app(composition.make_engine, composition.schemas)
            async with TestClient(TestServer(app)) as client:
                response = await client.get("/api/composition")
                assert response.status == 200
                response = await client.get("/api/nodes")
                assert response.status == 200
                assert "catalogfirst.constant" in str(await response.json())
                assert all(worker.cold for worker in workers)
            engine = composition.make_engine(lambda event: None)
            node_type = ("catalogfirst.constant", "catalogsecond.constant")[selected]
            result = await engine.run(Graph(nodes={"first": GraphNode(node_type, {})}), ["first"])
            assert result.outputs["first"]["value"].resolve() == 7
            assert workers[selected].alive
            assert workers[1 - selected].cold and not workers[1 - selected].alive
        finally:
            await composer.close()
            sys.modules.pop("catalogfirst_nodes", None)
            sys.modules.pop("catalogsecond_nodes", None)

    asyncio.run(scenario())


def test_catalogs_distinguish_manifests_in_the_same_directory(tmp_path: Path) -> None:
    first = write_pack(tmp_path, "catalogone").rename(tmp_path / "first.toml")
    second = write_pack(tmp_path, "catalogtwo")
    first_manifest = load_manifest(first)
    second_manifest = load_manifest(second)
    assert diagnose(first).ok
    assert read_catalog(second_manifest) is None
    assert diagnose(second).ok
    first_catalog = read_catalog(first_manifest)
    second_catalog = read_catalog(second_manifest)
    assert first_catalog is not None and set(first_catalog.schemas) == {"catalogone.constant"}
    assert second_catalog is not None and set(second_catalog.schemas) == {"catalogtwo.constant"}
    assert first_catalog.source != second_catalog.source
    original = first.read_text()
    first.write_text("[invalid TOML")
    assert not diagnose(first).ok
    first.write_text(original)
    assert read_catalog(first_manifest) is None
    assert read_catalog(second_manifest) is not None


@pytest.mark.parametrize("check", [diagnose, prepare_catalog])
def test_doctor_refresh_and_source_update_invalidation(
    tmp_path: Path, check: Callable[[Path], DoctorReport]
) -> None:
    path = write_pack(tmp_path / "pack", "catalogupdate")
    assert check(path).ok
    manifest = load_manifest(path)
    assert read_catalog(manifest) is not None
    write_pack(path.parent, "catalogupdate", 19)
    assert read_catalog(manifest) is None
    assert check(path).ok
    assert read_catalog(manifest) is not None
    original = path.read_text()
    path.write_text("[invalid TOML")
    assert not check(path).ok
    path.write_text(original)
    assert read_catalog(manifest) is None
    assert check(path).ok
    (path.parent / "catalogupdate_nodes.py").write_text("raise RuntimeError('broken')\n")
    assert not check(path).ok
    assert read_catalog(manifest) is None


def test_default_doctor_prepares_native_catalogs(monkeypatch: pytest.MonkeyPatch) -> None:
    from argparse import Namespace

    from dinkster import compose, manager
    from dinkster.comfy_compose import comfy_compat_specs
    from dinkster.installer import InstallError

    monkeypatch.setattr(compose, "default_pack_specs", lambda: ())
    monkeypatch.setenv("DINKSTER_EXECUTION_PYTHON", sys.executable)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    args = Namespace(defaults=True, library_root="", accelerator="cpu")
    # First-party internal imports remain doctor findings, not invalid schemas.
    with pytest.raises(InstallError, match="unhealthy packs"):
        manager._cmd_doctor(args)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dinkster-pack",
            "--accelerator",
            "cpu",
            "prepare-catalogs",
            "--defaults",
            "--library-root",
            "",
        ],
    )
    manager.main()
    owner, provider = comfy_compat_specs()
    assert Path(provider.manifest).parent.name == "dinkster-native"
    assert Path(provider.manifest).name == "dinkster-pack.toml"
    assert provider.env["DINKSTER_COMFY_NATIVE_ONLY"] == "1"
    for spec in (owner, provider):
        catalog = read_catalog(load_manifest(spec.manifest))
        assert catalog is not None
        assert "dinkster.ksampler" in catalog.schemas
        assert any(schema.input_families for schema in catalog.schemas.values())


def write_custom_type_pack(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "customcatalog"\n[pack.sandbox]\n'
        '[pack.entry]\nnodes = "customcatalog_nodes:NODES"\n'
        'types = "customcatalog_nodes:register_types"\n'
    )
    (root / "customcatalog_nodes.py").write_text(
        "from dinkster_api.v1 import Node, NodeSchema, InputSpec, OutputSpec, TypeExpr\n"
        "def register_types(registry):\n"
        "    registry.register('customcatalog.tag', encode=str.encode, decode=bytes.decode)\n"
        "class Echo(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        t = TypeExpr.concrete('customcatalog.tag')\n"
        "        return NodeSchema(node_type='customcatalog.echo', "
        "inputs=(InputSpec('value', t),), outputs=(OutputSpec('value', t),))\n"
        "    @classmethod\n"
        "    def execute(cls, value):\n"
        "        return cls.outputs(value=value)\n"
        "NODES = [Echo]\n"
    )
    return manifest


def test_in_process_custom_type_is_available_for_first_typed_literal(tmp_path: Path) -> None:
    manifest = write_custom_type_pack(tmp_path)
    assert diagnose(manifest).ok
    catalog = read_catalog(load_manifest(manifest))
    assert catalog is not None and catalog.types.type_ids == {"customcatalog.tag"}

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(PackSpec(manifest, in_process=True, require_catalog=True))
            worker = composer._records["customcatalog"].worker
            assert worker.cold
            engine = composer.composition.make_engine(lambda event: None)
            graph = Graph(
                nodes={
                    "echo": GraphNode(
                        "customcatalog.echo", {"value": TypedLiteral("customcatalog.tag", "first")}
                    )
                }
            )
            result = await engine.run(graph, ["echo"])
            assert result.outputs["echo"]["value"].resolve() == "first"
            assert worker.alive
        finally:
            await composer.close()
            sys.modules.pop("customcatalog_nodes", None)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "stamp", ["customcatalog.other", "asset<customcatalog.tag>", "asset<list<customcatalog.tag>>"]
)
def test_catalog_preserves_cold_type_coercion_metadata(tmp_path: Path, stamp: str) -> None:
    path = write_custom_type_pack(tmp_path)
    source = path.parent / "customcatalog_nodes.py"
    source.write_text(
        source.read_text().replace(
            "class Echo(Node):\n",
            "    registry.register('dinkster.asset', encode=str.encode, decode=bytes.decode)\n"
            "    registry.register('customcatalog.other', encode=str.encode, decode=bytes.decode)\n"
            "    registry.register_type_equivalence('customcatalog.tag', 'customcatalog.other', "
            "provider_id='equivalence')\n"
            "    registry.register_asset_decoder('customcatalog.tag', "
            "provider_id='decode', decode=lambda value: value)\n"
            "    registry.register_asset_decoder('list<customcatalog.tag>', "
            "provider_id='decode-list', decode=lambda value: [value])\n"
            "    registry.register_batch_merge('customcatalog.tag', "
            "provider_id='merge', merge=lambda values: ''.join(values))\n"
            "class Echo(Node):\n",
        )
    )
    diagnose(path)
    catalog = read_catalog(load_manifest(path))
    assert catalog is not None
    assert catalog.types.asset_decoders == {"customcatalog.tag", "list<customcatalog.tag>"}
    assert catalog.types.batch_merges == {"customcatalog.tag"}
    assert dict(catalog.types.equivalences) == {
        "customcatalog.tag": "customcatalog.other",
        "customcatalog.other": "customcatalog.tag",
    }

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(PackSpec(path, in_process=True, require_catalog=True))
            assert composer._records["customcatalog"].worker.cold
            graph = Graph(
                nodes={
                    "echo": GraphNode("customcatalog.echo", {"value": TypedLiteral(stamp, "first")})
                }
            )
            result = await composer.composition.make_engine(lambda event: None).run(graph, ["echo"])
            assert result.outputs["echo"]["value"].resolve() == "first"
        finally:
            await composer.close()
            sys.modules.pop("customcatalog_nodes", None)

    asyncio.run(scenario())


@pytest.mark.parametrize("malformed", [None, {"typeIds": ["list<customcatalog.tag>"]}])
def test_incomplete_type_catalog_requires_doctor(tmp_path: Path, malformed: object) -> None:
    path = write_custom_type_pack(tmp_path)
    assert diagnose(path).ok
    stored = catalog_path(path)
    wire = json.loads(stored.read_text())
    if malformed is None:
        del wire["declarations"]["types"]
    else:
        wire["declarations"]["types"].update(malformed)
    stored.write_text(json.dumps(wire))
    assert read_catalog(load_manifest(path)) is None
    assert diagnose(path).ok
    assert read_catalog(load_manifest(path)) is not None


@pytest.mark.parametrize("path_kind", ["disconnected", "lazy", "empty-region"])
@pytest.mark.parametrize("edited", [False, True])
def test_undemanded_typed_pack_stays_cold(tmp_path: Path, path_kind: str, edited: bool) -> None:
    custom = write_custom_type_pack(tmp_path / "custom")
    needed = write_pack(tmp_path / "needed", "needed")
    source = needed.parent / "needed_nodes.py"
    source.write_text(
        source.read_text()
        + "from dinkster_api.v1 import InputSpec\n"
        + "class Choose(Node):\n"
        + "    @classmethod\n"
        + "    def define_schema(cls):\n"
        + "        return NodeSchema(node_type='needed.choose', inputs=(\n"
        + "            InputSpec('switch', TypeExpr.concrete('core.boolean')),\n"
        + "            InputSpec('on', TypeExpr.concrete('core.int'), lazy=True),\n"
        + "            InputSpec('off', TypeExpr.concrete('customcatalog.tag'), lazy=True)),\n"
        + "            outputs=(OutputSpec('value', TypeExpr.concrete('core.int')),))\n"
        + "    @classmethod\n"
        + "    def check_lazy_status(cls, switch, on=None, off=None):\n"
        + "        if switch:\n"
        + "            return ('on',) if on is None else ()\n"
        + "        return ('off',) if off is None else ()\n"
        + "    @classmethod\n"
        + "    def execute(cls, switch, on=None, off=None):\n"
        + "        return cls.outputs(value=on if switch else len(off))\n"
        + "NODES.append(Choose)\n"
    )
    for path in (custom, needed):
        diagnose(path)
        assert read_catalog(load_manifest(path)) is not None

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            for path in (custom, needed):
                await composer.add_pack(PackSpec(path, in_process=True, require_catalog=True))
            worker = composer._records["customcatalog"].worker
            if edited:
                module = custom.parent / "customcatalog_nodes.py"
                module.write_text(module.read_text() + "CHANGED = True\n")
            echo = GraphNode(
                "customcatalog.echo", {"value": TypedLiteral("customcatalog.tag", "unused")}
            )
            if path_kind == "empty-region":
                graph = Graph(
                    nodes={
                        "selected": RegionNode(
                            kind="map",
                            body=Graph(nodes={"echo": echo}),
                            ports={"item": TypeExpr.concrete("core.int")},
                            inputs={"item": []},
                            element_ports=("item",),
                            outputs={"value": RegionOutput(Link("echo", "value"), "gather")},
                        )
                    }
                )
            else:
                graph = Graph(
                    nodes={
                        "unused": echo,
                        "selected": GraphNode("needed.constant", {})
                        if path_kind == "disconnected"
                        else GraphNode(
                            "needed.choose",
                            {"switch": True, "on": 7, "off": Link("unused", "value")},
                        ),
                    }
                )
            result = await composer.composition.make_engine(lambda event: None).run(
                graph, ["selected"]
            )
            assert result.outputs["selected"]["value"].resolve() == (
                [] if path_kind == "empty-region" else 7
            )
            assert worker.cold and not worker.alive
            assert "customcatalog_nodes" not in sys.modules
            if edited:
                with pytest.RaisesGroup(pytest.RaisesExc(RuntimeError, match="changed; refresh")):
                    await composer.composition.make_engine(lambda event: None).run(
                        Graph(nodes={"echo": echo}), ["echo"]
                    )
        finally:
            await composer.close()
            sys.modules.pop("customcatalog_nodes", None)
            sys.modules.pop("needed_nodes", None)

    asyncio.run(scenario())


def test_unknown_type_on_disconnected_node_still_fails_validation(tmp_path: Path) -> None:
    path = write_custom_type_pack(tmp_path)
    assert diagnose(path).ok

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(PackSpec(path, in_process=True, require_catalog=True))
            graph = Graph(
                nodes={
                    name: GraphNode("customcatalog.echo", {"value": TypedLiteral(stamp, "value")})
                    for name, stamp in (
                        ("selected", "customcatalog.tag"),
                        ("unused", "unknown.tag"),
                    )
                }
            )
            with pytest.raises(GraphValidationError, match="unregistered type"):
                await composer.composition.make_engine(lambda event: None).run(graph, ["selected"])
            assert composer._records["customcatalog"].worker.cold
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_type_provider_starts_for_consumer_without_provider_node(tmp_path: Path) -> None:
    provider = write_custom_type_pack(tmp_path / "provider")
    consumer = write_pack(tmp_path / "consumer", "consumer")
    (consumer.parent / "consumer_nodes.py").write_text(
        (provider.parent / "customcatalog_nodes.py")
        .read_text()
        .replace("customcatalog.echo", "consumer.echo")
    )
    for path in (provider, consumer):
        diagnose(path)
        assert read_catalog(load_manifest(path)) is not None

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            for path in (provider, consumer):
                await composer.add_pack(PackSpec(path, in_process=True, require_catalog=True))
            assert all(record.worker.cold for record in composer._records.values())
            graph = Graph(
                nodes={
                    "echo": GraphNode(
                        "consumer.echo", {"value": TypedLiteral("customcatalog.tag", "first")}
                    )
                }
            )
            result = await composer.composition.make_engine(lambda event: None).run(graph, ["echo"])
            assert result.outputs["echo"]["value"].resolve() == "first"
            assert all(record.worker.alive for record in composer._records.values())
        finally:
            await composer.close()
            sys.modules.pop("customcatalog_nodes", None)
            sys.modules.pop("consumer_nodes", None)

    asyncio.run(scenario())


def test_static_authoring_findings_preserve_valid_declarations(tmp_path: Path) -> None:
    path = write_pack(tmp_path / "pack", "authoringcatalog")
    source = path.parent / "authoringcatalog_nodes.py"
    source.write_text("import dinkster_workers\n" + source.read_text())
    report = diagnose(path)
    assert not report.ok
    assert any(f.code == "imports.host-machinery" for f in report.findings)
    assert prepare_catalog(path).ok
    catalog = read_catalog(load_manifest(path))
    assert catalog is not None
    assert "authoringcatalog.constant" in catalog.schemas


@pytest.mark.parametrize("failure", ["source-change", "namespace"])
def test_catalog_preparation_rejects_invalid_runtime_declarations(
    tmp_path: Path, failure: str
) -> None:
    path = write_pack(tmp_path / "pack", "runtimecatalog")
    source = path.parent / "runtimecatalog_nodes.py"
    if failure == "source-change":
        source.write_text(
            "from pathlib import Path\n"
            "Path(__file__).write_text(Path(__file__).read_text() + '\\n')\n" + source.read_text()
        )
        expected = "catalog.source-changed"
    else:
        path.write_text(
            path.read_text().replace("[pack]\n", '[pack]\nnamespaces = ["runtimecatalog"]\n')
        )
        source.write_text(
            source.read_text().replace("runtimecatalog.constant", "unclaimed.constant")
        )
        expected = "namespace.uncovered"
    report = prepare_catalog(path)
    assert not report.ok
    assert any(f.code == expected for f in report.findings)
    assert read_catalog(load_manifest(path)) is None


def test_catalog_preparation_rejects_probe_without_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_workers import doctor

    path = write_pack(tmp_path / "pack", "runtimecatalog")
    probed = doctor._run_probe(load_manifest(path), None, None, None)
    assert isinstance(probed, dict)
    del probed["catalog"]
    monkeypatch.setattr(doctor, "_run_probe", lambda *args: probed)
    report = prepare_catalog(path)
    assert not report.ok
    assert any(f.code == "catalog.missing" for f in report.findings)
    assert read_catalog(load_manifest(path)) is None


def test_doctor_refresh_does_not_reuse_timestamp_valid_bytecode(tmp_path: Path) -> None:
    path = write_pack(tmp_path / "pack", "schemarename")
    assert diagnose(path).ok
    source = path.parent / "schemarename_nodes.py"
    before = source.stat()
    source.write_text(source.read_text().replace(".constant", ".replaced"))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert diagnose(path).ok
    catalog = read_catalog(load_manifest(path))
    assert catalog is not None
    assert "schemarename.replaced" in catalog.schemas
    assert "schemarename.constant" not in catalog.schemas


@pytest.mark.parametrize("reload_before_use", [False, True])
def test_group_starts_once_when_second_member_is_invoked(
    tmp_path: Path, reload_before_use: bool
) -> None:
    paths = tuple(write_pack(tmp_path / name, name) for name in ("groupfirst", "groupsecond"))
    for path in paths:
        assert diagnose(path).ok

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            for path in paths:
                await composer.add_pack(
                    PackSpec(
                        path,
                        worker_group="cataloggroup",
                        group_manifests=paths,
                        env={"PYTHONPATH": os.pathsep.join(str(p.parent) for p in paths)},
                        require_catalog=True,
                    )
                )
            workers = [record.worker for record in composer._records.values()]
            assert all(worker.cold for worker in workers)
            if reload_before_use:
                write_pack(paths[1].parent, "groupsecond", 11)
                with pytest.raises(CompositionError, match="catalog is missing or stale"):
                    await composer.reload_pack("groupsecond")
                assert all(worker.cold for worker in workers)
                assert diagnose(paths[1]).ok
                await composer.reload_pack("groupsecond")
                workers = [record.worker for record in composer._records.values()]
                assert all(worker.cold for worker in workers)
            engine = composer.composition.make_engine(lambda event: None)
            result = await engine.run(
                Graph(nodes={"second": GraphNode("groupsecond.constant", {})}), ["second"]
            )
            assert result.outputs["second"]["value"].resolve() == (11 if reload_before_use else 7)
            assert all(worker.alive for worker in workers)
            assert workers[0].instance_token == workers[1].instance_token
            assert len(composer._group_activations) == 1
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_install_update_refreshes_catalog_and_portable_copy_runs(tmp_path: Path) -> None:
    source = write_pack(tmp_path / "source", "installedcatalog")
    installer = Installer(tmp_path / "install", accelerator="cpu", runtime_probe=lambda _: ())
    first, _ = lock_local_pack(source.parent, installer.artifacts_dir)
    installer.apply(Lockfile.of([first]), venvs=False)
    old_spec = installer.packs_for_serving()[0]
    assert read_catalog(load_manifest(old_spec.manifest)) is not None
    write_pack(source.parent, "installedcatalog", 29)
    node_source = source.parent / "installedcatalog_nodes.py"
    node_source.write_text(node_source.read_text().replace(".constant", ".updated"))
    second, _ = lock_local_pack(source.parent, installer.artifacts_dir)
    assert first.artifact_digest != second.artifact_digest
    installer.apply(Lockfile.of([second]), venvs=False)
    spec = installer.packs_for_serving()[0]
    manifest = load_manifest(spec.manifest)
    catalog = read_catalog(manifest)
    assert catalog is not None
    assert "installedcatalog.updated" in catalog.schemas
    assert "installedcatalog.constant" not in catalog.schemas
    assert old_spec.manifest != spec.manifest
    relocated = tmp_path / "another-host"
    shutil.copytree(manifest.root, relocated)
    assert read_catalog(load_manifest(relocated / "dinkster-pack.toml")) is not None

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(
                PackSpec(relocated, env={"PYTHONPATH": str(relocated)}, require_catalog=True)
            )
            engine = composer.composition.make_engine(lambda event: None)
            result = await engine.run(
                Graph(nodes={"node": GraphNode("installedcatalog.updated", {})}), ["node"]
            )
            assert result.outputs["node"]["value"].resolve() == 29
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_missing_stale_and_post_composition_mutation_refuse_without_start(tmp_path: Path) -> None:
    path = write_pack(tmp_path / "pack", "changedcatalog")

    async def scenario() -> None:
        composer = ServingComposer()
        spec = PackSpec(path, env={"PYTHONPATH": str(path.parent)}, require_catalog=True)
        try:
            with pytest.raises(CompositionError, match="catalog is missing or stale"):
                await composer.add_pack(spec)
            assert diagnose(path).ok
            write_pack(path.parent, "changedcatalog", 42)
            with pytest.raises(CompositionError, match="catalog is missing or stale"):
                await composer.add_pack(spec)
            assert diagnose(path).ok
            await composer.add_pack(spec)
            worker = composer._records["changedcatalog"].worker
            write_pack(path.parent, "changedcatalog", 43)
            with pytest.raises(RuntimeError, match="changed; refresh"):
                await worker.ensure_started()
            assert not worker.alive
            assert worker.instance_token is None
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize("changed_schema", [False, True])
def test_dynamic_catalog_refresh_reuses_first_worker_and_keeps_live_reloads(
    tmp_path: Path, changed_schema: bool, warm_cache: bool
) -> None:
    path = write_pack(tmp_path / "pack", "dynamiccatalog")
    path.write_text(path.read_text() + 'schema_reload = "dynamiccatalog_nodes:wait_reload"\n')
    source = path.parent / "dynamiccatalog_nodes.py"
    source.write_text(
        source.read_text()
        + "import asyncio, json, os\nfrom pathlib import Path\n"
        + "payload = json.loads(Path(os.environ['CATALOG_FILE']).read_text())\n"
        + "NODES = [type('Dynamic' + str(index), (Constant,), {\n"
        + "    'define_schema': classmethod(lambda cls, name=name: NodeSchema(\n"
        + "        node_type=name, version=payload['version'],\n"
        + "        outputs=(OutputSpec('value', TypeExpr.concrete('core.int')),)))\n"
        + "}) for index, name in enumerate(payload['names'])]\n"
        + "async def wait_reload():\n    await asyncio.Event().wait()\n"
    )
    metadata = tmp_path / "catalog.json"
    metadata.write_text(json.dumps({"names": ["dynamiccatalog.constant"], "version": 1}))
    environment = {"PYTHONPATH": str(path.parent), "CATALOG_FILE": str(metadata)}
    assert diagnose(path, environment=environment).ok

    async def scenario() -> None:
        requests: asyncio.Queue[str] = asyncio.Queue()
        spec = PackSpec(path, env=environment, require_catalog=True)
        graph = Graph(nodes={"node": GraphNode("dynamiccatalog.constant", {})})
        if warm_cache:
            initial = ServingComposer(cache_mode="layered", cache_dir=tmp_path / "cache")
            try:
                await initial.add_pack(spec)
                engine = initial.composition.make_engine(lambda event: None)
                assert (await engine.run(graph, ["node"])).executed == ("node",)
                assert (await engine.run(graph, ["node"])).cached == ("node",)
            finally:
                await initial.close()
        composer = ServingComposer(
            on_schema_reload=requests.put_nowait,
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
        )
        try:
            await composer.add_pack(spec)
            lazy = composer._records["dynamiccatalog"].worker
            assert lazy.cold
            assert set(composer.composition.schemas) == {"dynamiccatalog.constant"}
            metadata.write_text(
                json.dumps(
                    {
                        "names": ["dynamiccatalog.constant", "dynamiccatalog.added"],
                        "version": 2 if changed_schema else 1,
                    }
                )
            )
            engine = composer.composition.make_engine(lambda event: None)
            task = asyncio.create_task(
                engine.run(
                    Graph(nodes={"node": GraphNode("dynamiccatalog.constant", {})}), ["node"]
                )
            )
            assert await asyncio.wait_for(requests.get(), 10) == "dynamiccatalog"
            live = await lazy.ensure_started()
            token = live.instance_token
            await composer.reload_pack("dynamiccatalog")
            assert composer._records["dynamiccatalog"].worker is live
            assert live.instance_token == token and lazy.alive
            if changed_schema:
                with pytest.raises(ExecutionError, match="schema changed"):
                    await task
                engine = composer.composition.make_engine(lambda event: None)
                result = await engine.run(
                    Graph(nodes={"node": GraphNode("dynamiccatalog.constant", {})}), ["node"]
                )
            else:
                result = await task
                if warm_cache:
                    assert result.cached == ("node",) and result.executed == ()
            assert result.outputs["node"]["value"].resolve() == 7
            assert "dynamiccatalog.added" in composer.composition.schemas
            metadata.write_text(
                json.dumps(
                    {"names": ["dynamiccatalog.constant", "dynamiccatalog.latest"], "version": 3}
                )
            )
            await composer.reload_pack("dynamiccatalog")
            replacement = composer._records["dynamiccatalog"].worker
            assert replacement.alive and replacement.instance_token != token
            assert not live.alive
            assert "dynamiccatalog.latest" in composer.composition.schemas
            assert "dynamiccatalog.added" not in composer.composition.schemas
        finally:
            await composer.close()
        assert not replacement.alive

    asyncio.run(scenario())


def test_development_pack_uses_live_discovery_despite_existing_catalog(tmp_path: Path) -> None:
    path = write_pack(tmp_path / "pack", "developmentcatalog")
    assert diagnose(path).ok
    assert read_catalog(load_manifest(path)) is not None

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(PackSpec(path, env={"PYTHONPATH": str(path.parent)}))
            worker = composer._records["developmentcatalog"].worker
            assert not isinstance(worker, LazyWorker)
            assert worker.alive
            assert "developmentcatalog.constant" in composer.composition.schemas
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_concurrent_activation_is_shared_and_close_reaps_worker(tmp_path: Path) -> None:
    path = write_pack(tmp_path / "pack", "concurrentcatalog")
    assert diagnose(path).ok

    async def scenario() -> None:
        composer = ServingComposer()
        await composer.add_pack(
            PackSpec(path, env={"PYTHONPATH": str(path.parent)}, require_catalog=True)
        )
        worker = composer._records["concurrentcatalog"].worker
        try:
            engine = composer.composition.make_engine(lambda event: None)
            graph = Graph(nodes={"node": GraphNode("concurrentcatalog.constant", {})})
            results = await asyncio.gather(engine.run(graph, ["node"]), engine.run(graph, ["node"]))
            assert all(result.outputs["node"]["value"].resolve() == 7 for result in results)
            first, second = await asyncio.gather(worker.ensure_started(), worker.ensure_started())
            assert first is second
            assert first.alive
        finally:
            await composer.close()
        assert not first.alive
        with pytest.raises(RuntimeError, match="closed"):
            await worker.ensure_started()

    asyncio.run(scenario())


def test_live_schema_mismatch_refuses_before_execution(tmp_path: Path) -> None:
    path = write_pack(tmp_path / "pack", "variantcatalog")
    source = path.parent / "variantcatalog_nodes.py"
    source.write_text(
        "import os\n"
        + source.read_text().replace(
            'node_type="variantcatalog.constant"',
            'node_type="variantcatalog." + os.environ.get("CATALOG_VARIANT", "constant")',
        )
    )
    assert diagnose(path).ok

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            await composer.add_pack(
                PackSpec(
                    path,
                    env={"PYTHONPATH": str(path.parent), "CATALOG_VARIANT": "changed"},
                    require_catalog=True,
                )
            )
            worker = composer._records["variantcatalog"].worker
            with pytest.raises(RuntimeError, match="declarations changed"):
                await worker.ensure_started()
            assert not worker.alive
            assert worker.instance_token is None
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("replicas", [(), (0, 1)])
@pytest.mark.parametrize("explicit_attention", [False, True])
def test_native_selection_starts_only_selected_provider_and_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replicas: tuple[int, ...],
    explicit_attention: bool,
) -> None:
    from dinkster_protocol import AttentionPolicyConfig

    from dinkster.native_policy import NativeDispatchPolicy

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    node_type = "dinkster.create_hook_lora"
    paths = [write_pack(tmp_path / name, name) for name in ("catalogowner", "catalognative")]
    environment = {"PYTHONPATH": str(Path(__file__).parent / "fixtures/attention_provider")}
    for index, path in enumerate(paths):
        name = path.parent.name
        source = path.parent / f"{name}_nodes.py"
        source.write_text(
            source.read_text()
            .replace(f"{name}.constant", node_type)
            .replace(
                "outputs=(OutputSpec",
                "dispatch_affinity='native', outputs=(OutputSpec",
            )
            + ("ARM_NODES = {'native': [Constant]}\n" if index else "")
        )
        path.write_text(
            f'[pack]\nname = "{name}"\n'
            + (f'executes = ["{node_type}"]\n' if index else 'namespaces = ["dinkster"]\n')
            + (f'[pack.arms]\nnative = ["{node_type}"]\n' if index else "")
            + f'[pack.sandbox]\n[pack.entry]\nnodes = "{name}_nodes:NODES"\n'
            + (f'arm_nodes = "{name}_nodes:ARM_NODES"\n' if index else "")
        )
        report = diagnose(path, environment=environment)
        assert report.ok, report

    async def scenario() -> None:
        composer_ref: list[ServingComposer] = []
        composer = ServingComposer(
            worker_env=environment,
            native_policy=NativeDispatchPolicy(
                lambda _: None,
                lambda _: None,
                schemas=lambda: composer_ref[0].composition.schemas,
            ),
        )
        composer_ref.append(composer)
        try:
            for index, path in enumerate(paths):
                await composer.add_pack(
                    PackSpec(
                        path,
                        trust_reserved=True,
                        require_catalog=True,
                        replica_cuda_indices=replicas if index else (),
                        env={
                            "PYTHONPATH": os.pathsep.join(
                                (str(path.parent), environment["PYTHONPATH"])
                            )
                        },
                    )
                )
            owner = composer._records["catalogowner"].worker
            native = composer._records["catalognative"].worker
            runtime = composer._runtime_seat.pin()
            await runtime.worker.prepare([node_type])
            assert owner.cold and native.cold
            engine = composer.composition.make_engine(lambda event: None)
            result = await engine.run(
                Graph(nodes={"node": GraphNode(node_type, {})}),
                ["node"],
                attention_config=AttentionPolicyConfig("sdpa") if explicit_attention else None,
            )
            assert result.outputs["node"]["value"].resolve() == 7
            assert owner.cold
            if replicas:
                assert native.lanes[0].worker.alive
                assert native.lanes[1].worker.cold
            else:
                assert native.alive
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("replicas", [(), (0, 1)])
def test_inference_declarations_compose_without_materializing_sampling_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replicas: tuple[int, ...]
) -> None:
    # The fixture's attention provider uses declarations only, never CUDA.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delitem(sys.modules, "s1_sampler_pack_a", raising=False)
    fixtures = Path(__file__).parent
    environment = {
        "PYTHONPATH": os.pathsep.join(
            (str(fixtures / "fixtures/attention_provider"), str(fixtures))
        )
    }
    host = tmp_path / "host.toml"
    host.write_text(
        '[pack]\nname = "sampling-host"\nnamespaces = ["dinkster", "comfy"]\n'
        '[pack.arms]\nnative = ["dinkster.ksampler"]\n'
        '[pack.entry]\nnodes = "s1_sampler_host:NODES"\n'
        'arm_nodes = "s1_sampler_host:ARM_NODES"\nchoices = "s1_sampler_host:choices"\n'
    )
    extension_dir = tmp_path / "extension"
    extension_dir.mkdir()
    extension = extension_dir / "dinkster-pack.toml"
    extension.write_text(
        '[pack]\nname = "proof_a"\n'
        '[pack.entry]\nnodes = "s1_sampler_empty:NODES"\n'
        '[pack.extension]\ninference = "s1_sampler_pack_a:register"\n'
        'privileges = ["inference"]\n'
    )
    for path in (extension, host):
        report = diagnose(path, environment=environment)
        assert report.ok, report

    async def scenario() -> None:
        composer = ServingComposer(worker_env=environment)
        try:
            await composer.add_pack(
                PackSpec(
                    host, trust_reserved=True, require_catalog=True, replica_cuda_indices=replicas
                )
            )
            await composer.add_pack(
                PackSpec(extension, require_catalog=True, replica_cuda_indices=replicas)
            )
            workers = [record.worker for record in composer._records.values()]
            assert all(worker.cold for worker in workers)
            assert "s1_scaled_euler" in composer.composition.choices["comfy.samplers"]
            runtime = composer._runtime_seat.pin()
            assert any(
                item.id == "proof_a.scaled_euler"
                for item in runtime.sampler_registry_snapshot.samplers
            )
            assert "s1_sampler_pack_a" not in sys.modules
            from dinkster_protocol import extension_behavior_hash

            digest = "sha256:" + extension_behavior_hash(runtime.extension_snapshot)
            live = await workers[0].materialize_inference_generation(digest)
            assert live[0][0] == "proof_a"
            assert workers[0].alive and workers[1].cold
            engine = composer.composition.make_engine(lambda event: None)
            result = await engine.run(
                Graph(nodes={"sample": GraphNode("dinkster.ksampler", {})}), ["sample"]
            )
            assert result.outputs["sample"]["value"].resolve() == 3.0
            assert workers[1].cold
        finally:
            await composer.close()

    asyncio.run(scenario())
