"""Torch-free public interfaces for native attention guidance nodes."""

from __future__ import annotations

import tomllib
from pathlib import Path

from dinkster_api.v1 import CORE_FLOAT, CORE_STRING, MODEL_TYPE, TypeExpr, schema_to_wire
from dinkster_sampling import NODES as SAMPLING_NODES
from dinkster_sampling.attention_guidance import (
    PerturbedAttentionGuidance,
    SelfAttentionGuidance,
    SkipLayerGuidanceDiT,
    SkipLayerGuidanceDiTSimple,
)
from dinkster_schema import claim_covers
from dinkster_sd import NODES as SD_NODES
from dinkster_sd.attention_guidance import SkipLayerGuidanceSD3


def test_attention_guidance_schemas_are_registered_with_fixed_core_interfaces() -> None:
    root = Path(__file__).resolve().parents[1]
    manifests = {
        pack: tomllib.loads(
            (root / f"packages/dinkster-nodes-{pack}/dinkster-pack.toml").read_text(
                encoding="ascii"
            )
        )
        for pack in ("sampling", "sd")
    }
    nodes = (
        SelfAttentionGuidance,
        PerturbedAttentionGuidance,
        SkipLayerGuidanceDiT,
        SkipLayerGuidanceSD3,
        SkipLayerGuidanceDiTSimple,
    )
    expected = (
        ("dinkster.sampling.self_attention_guidance", ("model", "scale", "blur_sigma")),
        ("dinkster.sampling.perturbed_attention_guidance", ("model", "scale")),
        (
            "dinkster.sampling.skip_layer_guidance_di_t",
            (
                "model",
                "double_layers",
                "single_layers",
                "scale",
                "start_percent",
                "end_percent",
                "rescaling_scale",
            ),
        ),
        (
            "dinkster.sd.skip_layer_guidance_sd3",
            ("model", "layers", "scale", "start_percent", "end_percent"),
        ),
        (
            "dinkster.sampling.skip_layer_guidance_di_t_simple",
            ("model", "double_layers", "single_layers", "start_percent", "end_percent"),
        ),
    )
    assert (
        tuple(
            (node.schema().node_type, tuple(port.id for port in node.schema().inputs))
            for node in nodes
        )
        == expected
    )
    for node in nodes:
        schema = node.schema()
        assert node in (SD_NODES if node is SkipLayerGuidanceSD3 else SAMPLING_NODES)
        manifest = manifests["sd" if node is SkipLayerGuidanceSD3 else "sampling"]
        assert any(
            claim_covers(namespace, schema.node_type)
            for namespace in manifest["pack"]["namespaces"]
        )
        assert schema.inputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert schema.outputs[0].type == TypeExpr.concrete(MODEL_TYPE)
        assert schema_to_wire(schema)
    assert SelfAttentionGuidance.schema().inputs[1].type == TypeExpr.concrete(CORE_FLOAT)
    assert SkipLayerGuidanceDiT.schema().inputs[1].type == TypeExpr.concrete(CORE_STRING)
