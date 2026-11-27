"""A pack may use another pack's wire type without requiring that pack.

The owner's storing is used when the owning pack is composed; otherwise
the borrowing pack's own storing carries the value. These tests compose
real packs through ``ServingComposer`` in isolated workers, so every value
crosses a process boundary and is stored by a codec.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_engine import ExecutionError
from dinkster_graph import Graph, GraphNode, Link
from dinkster_registry import Lockfile
from dinkster_server import STATE_KEY, create_app
from dinkster_workers.doctor import diagnose, prepare_catalog

from dinkster.compose import ServingComposer, compose_serving
from dinkster.installer import Installer, lock_local_pack
from dinkster.pack_spec import PackSpec
from dinkster.reload_api import apply_remove
from tests.async_scenarios import run_scenario


@pytest.fixture(autouse=True)
def clear_pack_imports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path is not None and Path(path).is_relative_to(tmp_path):
            del sys.modules[name]


OWNER_SOURCE = """
import json
from dinkster_api.v1 import Node, NodeSchema, Output, TypeExpr

def register_types(registry):
    registry.register(
        "owner.thing",
        encode=lambda thing: b"OWNER:" + json.dumps(thing).encode(),
        decode=lambda data: {**json.loads(data.removeprefix(b"OWNER:")), "storing": "owner"},
    )

class Produce(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="owner.produce",
            body=(Output("thing", TypeExpr.concrete("owner.thing")),),
        )

    @classmethod
    def execute(cls):
        return cls.outputs(thing={"width": 11, "height": 29})

NODES = [Produce]
"""

BORROWER_SOURCE = """
import json
import os
from dinkster_api.v1 import CORE_INT, CORE_STRING, Input, Node, NodeSchema, Output, TypeExpr

TAG = {tag!r}

def register_types(registry):
    registry.register(
        "owner.thing",
        encode=lambda thing: TAG.encode() + b":" + json.dumps(thing).encode(),
        decode=lambda data: {{**json.loads(data.partition(b":")[2]), "storing": TAG}},
    )

class Produce(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="{name}.produce",
            body=(Output("thing", TypeExpr.concrete("owner.thing")),),
        )

    @classmethod
    def execute(cls):
        return cls.outputs(thing={{"width": 11, "height": 29}})

class Read(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="{name}.read",
            body=(
                Input("thing", TypeExpr.concrete("owner.thing")),
                Input("offset", TypeExpr.concrete(CORE_INT), default=0),
                Output("difference", TypeExpr.concrete(CORE_INT)),
                Output("storing", TypeExpr.concrete(CORE_STRING)),
            ),
        )

    @classmethod
    def execute(cls, *, thing, offset):
        if marker := os.environ.get("BORROWED_TYPES_MARKER"):
            with open(marker, "a") as stream:
                stream.write("read ran\\n")
        # In-process values never pass through a codec.
        return cls.outputs(
            difference=thing["height"] - thing["width"] + offset,
            storing=thing.get("storing", "not stored"),
        )

NODES = [Produce, Read]
"""


def write_pack(root: Path, name: str, source: str, uses: str = "") -> Path:
    package = root / name / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(source)
    manifest = root / name / "dinkster-pack.toml"
    manifest.write_text(
        f'[pack]\nname = "{name}"\npublisher = "{name}-author"\n{uses}'
        "[pack.sandbox]\ngpu = false\nnetwork = false\nwritable-mounts = false\n"
        f'[pack.entry]\nnodes = "{name}:NODES"\ntypes = "{name}:register_types"\n'
    )
    return manifest


def write_borrower(root: Path, name: str) -> Path:
    return write_pack(
        root, name, BORROWER_SOURCE.format(name=name, tag=name), 'uses-types = ["owner.thing"]\n'
    )


async def produce_and_read(
    entries: list[Path | PackSpec],
    producer: str,
    reader: str,
    *,
    offset: int = 0,
    **compose_options: Any,
) -> tuple[int, str]:
    composition = await compose_serving(entries, include_default_packs=False, **compose_options)
    try:
        engine = composition.make_engine(lambda _event: None)
        result = await engine.run(
            Graph(
                nodes={
                    "produce": GraphNode(f"{producer}.produce"),
                    "read": GraphNode(
                        f"{reader}.read", {"thing": Link("produce", "thing"), "offset": offset}
                    ),
                }
            ),
            ["read"],
        )
        outputs = result.outputs["read"]
        return cast(int, outputs["difference"].resolve()), cast(str, outputs["storing"].resolve())
    finally:
        await composition.close()


def test_borrower_alone_passes_the_value_between_its_nodes_with_its_own_storing(
    tmp_path: Path,
) -> None:
    borrower = write_borrower(tmp_path, "borrower")

    difference, storing = run_scenario(produce_and_read([borrower], "borrower", "borrower"))

    assert (difference, storing) == (18, "borrower")


def test_borrower_alone_passes_doctor_and_installs_without_the_owner(tmp_path: Path) -> None:
    borrower = write_borrower(tmp_path, "borrower")
    report = diagnose(borrower)

    problems = [
        finding
        for finding in report.findings
        if finding.severity == "error" or finding.code.startswith("types.")
    ]
    assert problems == []

    installer = Installer(tmp_path / "install")
    target = Lockfile.of([lock_local_pack(borrower.parent, installer.artifacts_dir)[0]])
    installer.apply(target, venvs=False)
    assert installer.current_lockfile() == target


def test_owner_storing_carries_the_borrowers_value_when_the_owner_is_installed(
    tmp_path: Path,
) -> None:
    owner = write_pack(tmp_path, "owner", OWNER_SOURCE)
    borrower = write_borrower(tmp_path, "borrower")

    # The borrower is listed first: the owner's storing applies whatever the order.
    difference, storing = run_scenario(produce_and_read([borrower, owner], "borrower", "borrower"))

    assert (difference, storing) == (18, "owner")


@pytest.mark.parametrize("validation", ["decode", "unresolvable", "encoded", "input-convert"])
@pytest.mark.parametrize("container", [False, True], ids=["atom", "list"])
def test_invalid_borrowed_value_fails_on_receiver_naming_producer_and_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation: str,
    container: bool,
) -> None:
    registration = {
        "decode": "decode=decode_thing",
        "unresolvable": "decode=decode_thing",
        "encoded": 'decode=lambda data: json.loads(data.removeprefix(b"OWNER:")),\n'
        "        validate_encoded=lambda data, _meta: decode_thing(data)",
        "input-convert": 'decode=lambda data: json.loads(data.removeprefix(b"OWNER:")),\n'
        "        input_convert=check_thing",
    }[validation]
    error = "UnresolvablePayload" if validation == "unresolvable" else "ValueError"
    owner = write_pack(
        tmp_path,
        "owner",
        "from dinkster_values import UnresolvablePayload\n"
        + OWNER_SOURCE.replace(
            'decode=lambda data: {**json.loads(data.removeprefix(b"OWNER:")), "storing": "owner"}',
            registration,
        ).replace(
            "def register_types(registry):",
            "def decode_thing(data):\n"
            '    return check_thing(json.loads(data.removeprefix(b"OWNER:")))\n\n'
            "def check_thing(thing):\n"
            '    if "height" not in thing:\n'
            f'        raise {error}("height is required")\n'
            '    return {**thing, "storing": "owner"}\n\n'
            "def register_types(registry):",
        ),
    )
    source = BORROWER_SOURCE
    if container:
        source = (
            source.replace(
                'TypeExpr.concrete("owner.thing")',
                'TypeExpr.list_of(TypeExpr.concrete("owner.thing"))',
            )
            .replace(
                'thing={{"width": 11, "height": 29}}',
                'thing=[{{"width": 11, "height": 29}}]',
            )
            .replace("        if marker :=", "        thing = thing[0]\n        if marker :=")
        )
    borrower = write_pack(
        tmp_path,
        "borrower",
        source.format(name="borrower", tag="borrower"),
        'uses-types = ["owner.thing"]\n',
    )
    maker = write_pack(
        tmp_path,
        "maker",
        source.format(name="maker", tag="maker").replace(
            '{"width": 11, "height": 29}', '{"width": 11}'
        ),
        'uses-types = ["owner.thing"]\n',
    )
    marker = tmp_path / "read-ran"
    in_process = validation == "input-convert"
    entries: list[Path | PackSpec] = [
        PackSpec(path, in_process=in_process) for path in (maker, borrower, owner)
    ]

    assert run_scenario(produce_and_read(entries, "borrower", "borrower"))[0] == 18
    monkeypatch.setenv("BORROWED_TYPES_MARKER", str(marker))
    with pytest.raises(ExecutionError) as failed:
        run_scenario(
            produce_and_read(
                entries,
                "maker",
                "borrower",
                worker_env={"BORROWED_TYPES_MARKER": str(marker)},
            )
        )

    assert failed.value.error.node_id == "read"
    assert "maker made a owner.thing value that owner cannot read" in failed.value.error.message
    assert "input 'thing'" in failed.value.error.message
    assert not marker.exists()


def test_value_stored_by_another_pack_fails_at_the_wire_naming_both_packs(
    tmp_path: Path,
) -> None:
    borrower = write_borrower(tmp_path, "borrower")
    owner = write_pack(tmp_path, "owner", OWNER_SOURCE)
    marker = tmp_path / "read-ran"
    options: dict[str, Any] = {
        "cache_mode": "layered",
        "cache_dir": tmp_path / "cache",
        "worker_env": {"BORROWED_TYPES_MARKER": str(marker)},
    }
    # Without the owner, the produced value is kept on disk in the borrower's storing.
    assert run_scenario(produce_and_read([borrower], "borrower", "borrower", **options)) == (
        18,
        "borrower",
    )
    marker.unlink()

    # With the owner installed, the kept value reaches a reader that uses the
    # owner's storing; a new offset makes the reader run instead of hitting its cache.
    with pytest.raises(Exception, match="stored by pack 'borrower'.*pack 'owner'"):
        run_scenario(
            produce_and_read([borrower, owner], "borrower", "borrower", offset=1, **options)
        )
    assert not marker.exists()


def test_storing_that_cannot_load_fails_at_the_wire_without_falling_back(
    tmp_path: Path,
) -> None:
    owner = write_pack(
        tmp_path,
        "owner",
        'import os\nif "OWNER_STORING_AVAILABLE" not in os.environ:\n'
        '    raise ImportError("owner storing needs its own environment")\n' + OWNER_SOURCE,
    )
    borrower = write_borrower(tmp_path, "borrower")
    owner_spec = PackSpec(owner, env={"OWNER_STORING_AVAILABLE": "1"})

    with pytest.raises(
        Exception,
        match="stored by pack 'owner'.*pack 'borrower'.*owner storing needs its own environment",
    ):
        run_scenario(produce_and_read([borrower, owner_spec], "borrower", "borrower"))


@pytest.mark.parametrize("order", [("cousin", "borrower"), ("borrower", "cousin")])
def test_without_the_owner_the_first_named_user_stores_the_type(
    tmp_path: Path, order: tuple[str, str]
) -> None:
    manifests = {name: write_borrower(tmp_path, name) for name in order}

    for start in range(2):
        result = run_scenario(
            produce_and_read([manifests[name] for name in order], "cousin", "borrower")
        )
        assert result == (18, "borrower"), f"start {start}"


@pytest.mark.parametrize(
    ("names", "expected", "hosting"),
    [
        (("cousin",), "cousin", "isolated"),
        (("cousin", "borrower"), "borrower", "isolated"),
        (("borrower", "cousin"), "borrower", "isolated"),
        (("cousin", "owner", "borrower"), "owner", "isolated"),
        (("cousin", "borrower"), "borrower", "group"),
        (("cousin", "owner"), "owner", "catalog"),
        (("cousin",), "cousin", "parts"),
    ],
)
def test_pack_catalog_names_the_storing_used_by_its_borrowed_values(
    tmp_path: Path, names: tuple[str, ...], expected: str, hosting: str
) -> None:
    manifests = [
        write_pack(tmp_path, name, OWNER_SOURCE)
        if name == "owner"
        else write_borrower(tmp_path, name)
        for name in names
    ]
    if hosting == "catalog":
        for manifest in manifests:
            assert prepare_catalog(manifest).ok
    entries = [
        PackSpec(
            manifest,
            require_catalog=hosting == "catalog",
            worker_group="borrowed" if hosting == "group" else None,
            group_manifests=tuple(manifests) if hosting == "group" else (),
        )
        for manifest in manifests
    ]
    if hosting == "parts":
        manifest = manifests[0]
        (manifest.parent / "cousin" / "part.py").write_text(
            BORROWER_SOURCE.format(name="cousin.part", tag="cousin").replace(
                "owner.thing", "owner.other"
            )
        )
        manifest.write_text(
            manifest.read_text() + '\n[[pack.parts]]\nname="plain"\nuses-types=["owner.other"]\n'
            '[pack.parts.entry]\nnodes="cousin.part:NODES"\ntypes="cousin.part:register_types"\n'
        )
        entries.append(PackSpec(manifest, part="plain"))

    async def scenario() -> None:
        composer = ServingComposer()
        composition = composer.composition
        try:
            for entry in composer.order_pack_entries(entries):
                await composer.add_pack(entry)
            for replay in range(3 if hosting == "parts" else 2):
                if replay == 1:
                    await composer.reload_pack("cousin")
                if replay == 2:
                    await composer.remove_pack("cousin~plain")
                app = create_app(
                    composition.make_engine,
                    composition.schemas,
                    packs=composition.packs,
                    node_packs=composition.node_packs,
                )
                async with TestClient(TestServer(app)) as client:
                    catalog = await (await client.get("/api/nodes")).json()
                    storage = {"owner.thing": expected}
                    if hosting == "parts" and replay < 2:
                        storage["owner.other"] = "cousin"
                    assert catalog["packs"]["cousin"]["typeStorage"] == storage
                engine = composition.make_engine(lambda _event: None)
                result = await engine.run(
                    Graph(
                        nodes={
                            "produce": GraphNode("cousin.produce"),
                            "read": GraphNode("cousin.read", {"thing": Link("produce", "thing")}),
                        }
                    ),
                    ["read"],
                )
                assert result.outputs["read"]["storing"].resolve() == expected
                if hosting == "parts" and replay < 2:
                    result = await engine.run(
                        Graph(
                            nodes={
                                "produce": GraphNode("cousin.part.produce"),
                                "read": GraphNode(
                                    "cousin.part.read", {"thing": Link("produce", "thing")}
                                ),
                            }
                        ),
                        ["read"],
                    )
                    assert result.outputs["read"]["storing"].resolve() == "cousin"
        finally:
            await composer.close()

    run_scenario(scenario())


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("provider", ["owner", "aardvark"])
@pytest.mark.parametrize("hosting", ["in-process", "catalog", "isolated"])
def test_live_catalog_tracks_shared_host_storing_but_not_isolated_workers(
    tmp_path: Path, hosting: str, provider: str, warm: bool
) -> None:
    borrower = write_pack(
        tmp_path,
        "borrower",
        BORROWER_SOURCE.format(name="borrower", tag="borrower").replace(
            "encode=lambda",
            'input_convert=lambda thing: {**thing, "storing": TAG},\n        encode=lambda',
        ),
        'uses-types = ["owner.thing"]\n',
    )
    storage_pack = write_pack(
        tmp_path,
        provider,
        (
            OWNER_SOURCE
            if provider == "owner"
            else BORROWER_SOURCE.format(name=provider, tag=provider)
        ).replace(
            "encode=lambda",
            f'input_convert=lambda thing: {{**thing, "storing": "{provider}"}},\n'
            "        encode=lambda",
        ),
        "" if provider == "owner" else 'uses-types = ["owner.thing"]\n',
    )
    if hosting == "catalog":
        for manifest in (borrower, storage_pack):
            assert prepare_catalog(manifest).ok

    async def scenario() -> None:
        composer = ServingComposer()
        composition = composer.composition
        try:
            await composer.add_pack(
                PackSpec(
                    borrower, in_process=hosting != "isolated", require_catalog=hosting == "catalog"
                )
            )
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
            )
            async with TestClient(TestServer(app)) as client:
                catalog = await (await client.get("/api/nodes")).json()
                assert catalog["packs"]["borrower"]["typeStorage"] == {"owner.thing": "borrower"}
                graph = Graph(
                    nodes={
                        "produce": GraphNode("borrower.produce"),
                        "read": GraphNode("borrower.read", {"thing": Link("produce", "thing")}),
                    }
                )
                if warm:
                    result = await composition.make_engine(lambda _event: None).run(graph, ["read"])
                    assert result.outputs["read"]["storing"].resolve() == "borrower"
                delta = await composer.add_pack(
                    PackSpec(
                        storage_pack,
                        in_process=hosting != "isolated",
                        require_catalog=hosting == "catalog",
                    )
                )
                app[STATE_KEY].announce(delta.schemas, delta.packs, delta.node_packs)
                expected = "borrower" if hosting == "isolated" else provider
                catalog = await (await client.get("/api/nodes")).json()
                assert catalog["packs"]["borrower"]["typeStorage"] == {"owner.thing": expected}
                result = await composition.make_engine(lambda _event: None).run(graph, ["read"])
                assert result.outputs["read"]["storing"].resolve() == expected
                catalog = await (await client.get("/api/nodes")).json()
                assert catalog["packs"]["borrower"]["typeStorage"] == {"owner.thing": expected}
                await apply_remove(app[STATE_KEY], composer, provider)
                catalog = await (await client.get("/api/nodes")).json()
                assert catalog["packs"]["borrower"]["typeStorage"] == {"owner.thing": "borrower"}
                result = await composition.make_engine(lambda _event: None).run(graph, ["read"])
                assert result.outputs["read"]["storing"].resolve() == "borrower"
        finally:
            await composer.close()

    run_scenario(scenario())


def test_in_process_owner_composes_after_a_user_loaded_its_storing(tmp_path: Path) -> None:
    # Packs compose in name order: "borrower" starts first and loads the
    # owner's registration as the type's storing; "owner" then loads its
    # own types entry again.
    entries: list[Path | PackSpec] = [
        PackSpec(write_pack(tmp_path, "owner", OWNER_SOURCE), in_process=True),
        PackSpec(write_borrower(tmp_path, "borrower"), in_process=True),
    ]

    assert run_scenario(produce_and_read(entries, "owner", "borrower")) == (18, "not stored")


@pytest.mark.parametrize("loose", [False, True])
def test_same_named_modules_of_the_storing_and_using_pack_never_cross(
    tmp_path: Path, loose: bool
) -> None:
    # The owner's storing reads a helper module; the borrower has one of the
    # same name. Loaded into the borrower's worker, the owner's registration
    # must get its own helper or the run must fail naming both packs.
    helper = "helpers" if loose else "owner.helpers"
    owner = write_pack(
        tmp_path,
        "owner",
        f"from {helper} import TAG\n"
        + OWNER_SOURCE.replace('"storing": "owner"', '"storing": TAG'),
    )
    borrower = write_borrower(tmp_path, "borrower")
    for name in ("owner", "borrower"):
        directory = tmp_path / name if loose else tmp_path / name / name
        (directory / "helpers.py").write_text(f"TAG = '{name} helpers'\n")

    if not loose:
        result = run_scenario(produce_and_read([borrower, owner], "borrower", "borrower"))
        assert result == (18, "owner helpers")
        return
    with pytest.raises(
        Exception,
        match="stored by pack 'owner'.*pack 'borrower'.*top-level modules \\['helpers'\\]",
    ):
        run_scenario(produce_and_read([borrower, owner], "borrower", "borrower"))
