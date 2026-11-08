"""Torch-free SeedVR2 conditioning schema and discovery."""

from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    LATENT_TYPE,
    MODEL_TYPE,
    InputSpec,
    MappingSource,
    NodeSchema,
    OutputSpec,
    ReplacementCase,
    ReplacementRule,
    TypeExpr,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_seedvr import NODES
from dinkster_seedvr.seedvr_conditioning import SeedVR2Conditioning


def test_seedvr_conditioning_schema() -> None:
    schema = SeedVR2Conditioning.schema()
    assert schema == NodeSchema(
        node_type="dinkster.seedvr.seed_vr2_conditioning",
        display_name="Apply SeedVR2 Conditioning",
        category="model/conditioning",
        description="Build SeedVR2 positive/negative conditioning from a VAE latent.",
        search_terms=("seedvr2", "upscale", "conditioning", "SeedVR2Conditioning"),
        inputs=(
            InputSpec("model", TypeExpr.concrete(MODEL_TYPE), doc="The SeedVR2 model."),
            InputSpec("vae_conditioning", TypeExpr.concrete(LATENT_TYPE), display_name="latent"),
        ),
        outputs=(
            OutputSpec(
                "positive",
                TypeExpr.concrete(CONDITIONING_TYPE),
                display_name="positive",
                doc="The positive conditioning for sampling.",
            ),
            OutputSpec(
                "negative",
                TypeExpr.concrete(CONDITIONING_TYPE),
                display_name="negative",
                doc="The negative conditioning for sampling.",
            ),
        ),
        replacements=(
            ReplacementRule(
                from_type="SeedVR2Conditioning",
                cases=(
                    ReplacementCase.build(
                        "dinkster.seedvr.seed_vr2_conditioning",
                        inputs={
                            "model": MappingSource.copy("model"),
                            "vae_conditioning": MappingSource.copy("vae_conditioning"),
                        },
                        outputs={"positive": "positive", "negative": "negative"},
                    ),
                ),
            ),
        ),
    )
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert NODES.count(SeedVR2Conditioning) == 1
