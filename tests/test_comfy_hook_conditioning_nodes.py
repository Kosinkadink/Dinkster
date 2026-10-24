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
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.hook_conditioning import HOOK_CONDITIONING_NODES


def test_hook_conditioning_schemas_are_torch_free_and_match_core_interfaces() -> None:
    expected = {
        "ConditioningSetProperties": (
            ("cond_NEW", "strength", "set_cond_area", "mask", "hooks", "timesteps"),
            ("conditioning",),
        ),
        "ConditioningSetPropertiesAndCombine": (
            ("cond", "cond_NEW", "strength", "set_cond_area", "mask", "hooks", "timesteps"),
            ("conditioning",),
        ),
        "PairConditioningSetProperties": (
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
        "PairConditioningSetPropertiesAndCombine": (
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
        "ConditioningSetDefaultCombine": (("cond", "cond_DEFAULT", "hooks"), ("conditioning",)),
        "PairConditioningSetDefaultCombine": (
            ("positive", "negative", "positive_DEFAULT", "negative_DEFAULT", "hooks"),
            ("positive", "negative"),
        ),
        "PairConditioningCombine": (
            ("positive_A", "negative_A", "positive_B", "negative_B"),
            ("positive", "negative"),
        ),
        "SetClipHooks": (("clip", "apply_to_conds", "schedule_clip", "hooks"), ("clip",)),
        "ConditioningTimestepsRange": (
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
        assert node in COMFY_NODES
    properties = schemas["ConditioningSetProperties"]
    assert [(item.type, item.required, item.default) for item in properties.inputs] == [
        (conditioning, True, None),
        (float_, False, 1.0),
        (combo, False, "default"),
        (mask, False, None),
        (hooks, False, None),
        (timesteps, False, None),
    ]
    assert properties.inputs[1].widget == NumberWidget(min=0.0, max=10.0, step=0.01)
    assert properties.inputs[2].widget == ComboWidget(("default", "mask bounds"))
    clip = schemas["SetClipHooks"]
    assert [(item.type, item.required, item.default, item.advanced) for item in clip.inputs] == [
        (TypeExpr.concrete(CLIP_TYPE), True, None, False),
        (boolean, False, True, True),
        (boolean, False, False, True),
        (hooks, False, None, False),
    ]
    assert clip.occupies == ("gpu",)
    range_ = schemas["ConditioningTimestepsRange"]
    assert [(item.required, item.default, item.widget) for item in range_.inputs] == [
        (False, 0.0, NumberWidget(min=0.0, max=1.0, step=0.001)),
        (False, 1.0, NumberWidget(min=0.0, max=1.0, step=0.001)),
    ]
    assert "torch" not in sys.modules


def test_hook_conditioning_cold_discovery_avoids_torch_and_inference_imports() -> None:
    manifest = tomllib.loads(
        (
            Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
        ).read_text(encoding="ascii")
    )
    node_types = [node.schema().node_type for node in HOOK_CONDITIONING_NODES]
    namespaces = manifest["pack"]["namespaces"]
    start = namespaces.index(node_types[0])
    assert namespaces[start : start + len(node_types)] == node_types
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['dinkster_inference'] = None\n"
            "sys.modules['torch'] = None\n"
            "from dinkster_nodes_comfy import COMFY_NODES\n"
            "schemas = [node.define_schema() for node in COMFY_NODES]\n"
            "assert {'ConditioningSetProperties', 'SetClipHooks', "
            "'ConditioningTimestepsRange'} <= {schema.node_type for schema in schemas}\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
