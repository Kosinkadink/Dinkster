"""Cold schemas for native hook conditioning nodes."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from dinkster_api.v1 import (
    CLIP_TYPE,
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    CORE_COMBO,
    CORE_FLOAT,
    HOOKS_TYPE,
    MASK_TYPE,
    TIMESTEPS_RANGE_TYPE,
    ComboWidget,
    NumberWidget,
    TypeExpr,
)
from dinkster_models import NODES
from dinkster_models.hook_conditioning import (
    ConditioningSetDefaultCombine,
    ConditioningSetProperties,
    ConditioningSetPropertiesAndCombine,
    ConditioningTimestepsRange,
    PairConditioningCombine,
    PairConditioningSetDefaultCombine,
    PairConditioningSetProperties,
    PairConditioningSetPropertiesAndCombine,
    SetClipHooks,
)

HOOK_CONDITIONING_NODES = (
    ConditioningSetProperties,
    ConditioningSetPropertiesAndCombine,
    PairConditioningSetProperties,
    PairConditioningSetPropertiesAndCombine,
    ConditioningSetDefaultCombine,
    PairConditioningSetDefaultCombine,
    PairConditioningCombine,
    SetClipHooks,
    ConditioningTimestepsRange,
)


def test_hook_conditioning_schemas_are_torch_free_and_match_core_interfaces() -> None:
    expected = {
        "dinkster.models.conditioning_set_properties": (
            ("cond_NEW", "strength", "set_cond_area", "mask", "hooks", "timesteps"),
            ("conditioning",),
        ),
        "dinkster.models.conditioning_set_properties_and_combine": (
            ("cond", "cond_NEW", "strength", "set_cond_area", "mask", "hooks", "timesteps"),
            ("conditioning",),
        ),
        "dinkster.models.pair_conditioning_set_properties": (
            (
                "positive_NEW",
                "negative_NEW",
                "strength",
                "set_cond_area",
                "mask",
                "hooks",
                "timesteps",
            ),
            ("positive", "negative"),
        ),
        "dinkster.models.pair_conditioning_set_properties_and_combine": (
            (
                "positive",
                "negative",
                "positive_NEW",
                "negative_NEW",
                "strength",
                "set_cond_area",
                "mask",
                "hooks",
                "timesteps",
            ),
            ("positive", "negative"),
        ),
        "dinkster.models.conditioning_set_default_combine": (
            ("cond", "cond_DEFAULT", "hooks"),
            ("conditioning",),
        ),
        "dinkster.models.pair_conditioning_set_default_combine": (
            ("positive", "negative", "positive_DEFAULT", "negative_DEFAULT", "hooks"),
            ("positive", "negative"),
        ),
        "dinkster.models.pair_conditioning_combine": (
            ("positive_A", "negative_A", "positive_B", "negative_B"),
            ("positive", "negative"),
        ),
        "dinkster.models.set_clip_hooks": (
            ("clip", "apply_to_conds", "schedule_clip", "hooks"),
            ("clip",),
        ),
        "dinkster.models.conditioning_timesteps_range": (
            ("start_percent", "end_percent"),
            ("TIMESTEPS_RANGE", "BEFORE_RANGE", "AFTER_RANGE"),
        ),
    }

    schemas = {node.schema().node_type: node.schema() for node in HOOK_CONDITIONING_NODES}
    assert {
        node_type: (
            tuple(item.id for item in schema.inputs),
            tuple(item.id for item in schema.outputs),
        )
        for node_type, schema in schemas.items()
    } == expected
    conditioning = TypeExpr.concrete(CONDITIONING_TYPE)
    hooks = TypeExpr.concrete(HOOKS_TYPE)
    mask = TypeExpr.concrete(MASK_TYPE)
    timesteps = TypeExpr.concrete(TIMESTEPS_RANGE_TYPE)
    float_ = TypeExpr.concrete(CORE_FLOAT)
    boolean = TypeExpr.concrete(CORE_BOOLEAN)
    combo = TypeExpr.concrete(CORE_COMBO)

    for node in HOOK_CONDITIONING_NODES:
        assert node in NODES
    properties = schemas["dinkster.models.conditioning_set_properties"]
    assert [(item.type, item.required, item.default) for item in properties.inputs] == [
        (conditioning, True, None),
        (float_, True, 1.0),
        (combo, True, "default"),
        (mask, False, None),
        (hooks, False, None),
        (timesteps, False, None),
    ]
    for name in (
        "dinkster.models.conditioning_set_properties",
        "dinkster.models.conditioning_set_properties_and_combine",
        "dinkster.models.pair_conditioning_set_properties",
        "dinkster.models.pair_conditioning_set_properties_and_combine",
    ):
        assert {item.id for item in schemas[name].inputs if not item.required} == {
            "mask",
            "hooks",
            "timesteps",
        }
    assert properties.inputs[1].widget == NumberWidget(min=0.0, max=10.0, step=0.01)
    assert properties.inputs[2].widget == ComboWidget(("default", "mask bounds"))
    clip = schemas["dinkster.models.set_clip_hooks"]
    assert [(item.type, item.required, item.default, item.advanced) for item in clip.inputs] == [
        (TypeExpr.concrete(CLIP_TYPE), True, None, False),
        (boolean, True, True, True),
        (boolean, True, False, True),
        (hooks, False, None, False),
    ]
    assert clip.occupies == ("gpu",)
    range_ = schemas["dinkster.models.conditioning_timesteps_range"]
    assert [(item.required, item.default, item.widget) for item in range_.inputs] == [
        (True, 0.0, NumberWidget(min=0.0, max=1.0, step=0.001)),
        (True, 1.0, NumberWidget(min=0.0, max=1.0, step=0.001)),
    ]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from dinkster_models import NODES; "
            "[node.schema() for node in NODES]; "
            "import sys; assert 'torch' not in sys.modules",
        ],
        check=True,
        timeout=30,
    )


def test_hook_conditioning_cold_discovery_avoids_torch_and_inference_imports() -> None:
    manifest = tomllib.loads(
        (
            Path(__file__).resolve().parents[1]
            / "packages/dinkster-nodes-models/dinkster-pack.toml"
        ).read_text(encoding="ascii")
    )
    assert manifest["pack"]["namespaces"] == ["dinkster.models", "devices"]
    start = NODES.index(ConditioningSetProperties)
    assert NODES[start : start + len(HOOK_CONDITIONING_NODES)] == HOOK_CONDITIONING_NODES
