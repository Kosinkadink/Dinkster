"""A pack may use another pack's wire type without requiring that pack.

The owner's storing is used when the owning pack is composed; otherwise
the borrowing pack's own storing carries the value. These tests compose
real packs through ``ServingComposer`` in isolated workers, so every value
crosses a process boundary and is stored by a codec.
"""

from __future__ import annotations

from pathlib import Path

from dinkster_graph import Graph, GraphNode, Link
from dinkster_workers.doctor import diagnose

from dinkster.compose import compose_serving
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
                Output("difference", TypeExpr.concrete(CORE_INT)),
                Output("storing", TypeExpr.concrete(CORE_STRING)),
            ),
        )

    @classmethod
    def execute(cls, *, thing):
        return cls.outputs(
            difference=thing["height"] - thing["width"], storing=thing["storing"]
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


async def produce_and_read(manifests: list[Path], producer: str, reader: str) -> tuple[int, str]:
    composition = await compose_serving(manifests, include_default_packs=False)
    try:
        engine = composition.make_engine(lambda _event: None)
        result = await engine.run(
            Graph(
                nodes={
                    "produce": GraphNode(f"{producer}.produce"),
                    "read": GraphNode(f"{reader}.read", {"thing": Link("produce", "thing")}),
                }
            ),
            ["read"],
        )
        outputs = result.outputs["read"]
        return outputs["difference"].resolve(), outputs["storing"].resolve()
    finally:
        await composition.close()


def test_borrower_alone_passes_the_value_between_its_nodes_with_its_own_storing(
    tmp_path: Path,
) -> None:
    borrower = write_borrower(tmp_path, "borrower")

    difference, storing = run_scenario(produce_and_read([borrower], "borrower", "borrower"))

    assert (difference, storing) == (18, "borrower")


def test_borrower_alone_passes_doctor_without_the_owner(tmp_path: Path) -> None:
    report = diagnose(write_borrower(tmp_path, "borrower"))

    problems = [
        finding
        for finding in report.findings
        if finding.severity == "error" or finding.code.startswith("types.")
    ]
    assert problems == []


def test_owner_storing_carries_the_borrowers_value_when_the_owner_is_installed(
    tmp_path: Path,
) -> None:
    owner = write_pack(tmp_path, "owner", OWNER_SOURCE)
    borrower = write_borrower(tmp_path, "borrower")

    # The borrower is listed first: the owner's storing applies whatever the order.
    difference, storing = run_scenario(
        produce_and_read([borrower, owner], "borrower", "borrower")
    )

    assert (difference, storing) == (18, "owner")
