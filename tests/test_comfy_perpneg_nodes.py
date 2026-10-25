"""Torch-free public registration checks for perpendicular negative guidance."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.perpneg import PERPNEG_NODES


def test_perpneg_nodes_are_registered_with_expected_ports() -> None:
    manifest = tomllib.loads(
        (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        ).read_text(encoding="ascii")
    )
    assert tuple(
        (node.schema().node_type, tuple(port.id for port in node.schema().inputs))
        for node in PERPNEG_NODES
    ) == (
        ("PerpNeg", ("model", "empty_conditioning", "neg_scale")),
        (
            "PerpNegGuider",
            ("model", "positive", "negative", "empty_conditioning", "cfg", "neg_scale"),
        ),
    )
    for node in PERPNEG_NODES:
        assert node in COMFY_NODES
        assert node.schema().node_type in manifest["pack"]["namespaces"]
        assert node.schema().category == "experimental"
    assert PERPNEG_NODES[0].schema().deprecation is not None
    assert PERPNEG_NODES[0].schema().search_visibility == "deprecated"
    assert tuple(node.schema().node_type for node in COMFY_NODES) == tuple(
        manifest["pack"]["namespaces"]
    )


def test_perpneg_schema_imports_without_torch_or_inference() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_nodes_comfy.perpneg import PERPNEG_NODES\n"
            "actual = [node.schema().node_type for node in PERPNEG_NODES]\n"
            "assert actual == ['PerpNeg', 'PerpNegGuider']\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
