"""Torch-free public registration checks for perpendicular negative guidance."""

from __future__ import annotations

import tomllib
from pathlib import Path

from dinkster_sampling import NODES
from dinkster_sampling.perpneg import PerpNeg, PerpNegGuider
from dinkster_schema import claim_covers


def test_perpneg_nodes_are_registered_with_expected_ports() -> None:
    manifest = tomllib.loads(
        (
            Path(__file__).resolve().parents[1]
            / "packages/dinkster-nodes-sampling/dinkster-pack.toml"
        ).read_text(encoding="ascii")
    )
    assert tuple(
        (node.schema().node_type, tuple(port.id for port in node.schema().inputs))
        for node in (PerpNeg, PerpNegGuider)
    ) == (
        ("dinkster.sampling.perp_neg", ("model", "empty_conditioning", "neg_scale")),
        (
            "dinkster.sampling.perp_neg_guider",
            ("model", "positive", "negative", "empty_conditioning", "cfg", "neg_scale"),
        ),
    )
    for node in (PerpNeg, PerpNegGuider):
        assert NODES.count(node) == 1
        assert any(
            claim_covers(claim, node.schema().node_type) for claim in manifest["pack"]["namespaces"]
        )
        assert node.schema().category == "experimental"
    assert PerpNeg.schema().deprecation is not None
    assert PerpNeg.schema().search_visibility == "deprecated"
    assert all(
        any(
            claim_covers(claim, node.schema().node_type) for claim in manifest["pack"]["namespaces"]
        )
        for node in NODES
    )
