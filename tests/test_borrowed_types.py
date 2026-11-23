"""A pack may use another pack's wire type without requiring that pack.

The owner's storing is used when the owning pack is composed; otherwise
the borrowing pack's own storing carries the value. These tests compose
real packs through ``ServingComposer`` in isolated workers, so every value
crosses a process boundary and is stored by a codec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from dinkster_graph import Graph, GraphNode, Link
from dinkster_registry import Lockfile
from dinkster_workers.doctor import diagnose

from dinkster.compose import compose_serving
from dinkster.installer import Installer, lock_local_pack
from dinkster.pack_spec import PackSpec
from tests.async_scenarios import run_scenario

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


def test_in_process_owner_composes_after_a_user_loaded_its_storing(tmp_path: Path) -> None:
    # Packs compose in name order: "borrower" starts first and loads the
    # owner's registration as the type's storing; "owner" then loads its
    # own types entry again.
    entries: list[Path | PackSpec] = [
        PackSpec(write_pack(tmp_path, "owner", OWNER_SOURCE), in_process=True),
        PackSpec(write_borrower(tmp_path, "borrower"), in_process=True),
    ]

    assert run_scenario(produce_and_read(entries, "owner", "borrower")) == (18, "not stored")
