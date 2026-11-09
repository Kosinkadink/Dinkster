"""SAM3 preview declarations remain independent of inference imports."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from dinkster_api.v1 import CORE_FLOAT, IMAGE_TYPE, SAM3_TRACK_DATA_TYPE, NumberWidget, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.sam3_preview import SAM3_TrackPreview


def test_sam3_preview_public_schema_and_manifest() -> None:
    schema = SAM3_TrackPreview.define_schema()
    assert schema.node_type == "SAM3_TrackPreview"
    assert schema.display_name == "SAM3 Track Preview" and schema.category == "image/detection"
    assert schema.output_node and schema.emits_previews and schema.outputs == ()
    assert [item.id for item in schema.inputs] == ["track_data", "images", "opacity", "fps"]
    assert [item.type for item in schema.inputs] == [
        TypeExpr.concrete(SAM3_TRACK_DATA_TYPE),
        TypeExpr.concrete(IMAGE_TYPE),
        TypeExpr.concrete(CORE_FLOAT),
        TypeExpr.concrete(CORE_FLOAT),
    ]
    assert [item.required for item in schema.inputs] == [True, False, True, True]
    assert schema.inputs[2].default == 0.5 and schema.inputs[3].default == 24.0
    assert schema.inputs[2].widget == NumberWidget(min=0.0, max=1.0, step=0.05)
    assert schema.inputs[3].widget == NumberWidget(min=1.0, max=120.0, step=1.0)
    assert COMFY_NODES.count(SAM3_TrackPreview) == 1
    root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads(
        (root / "packages/dinkster-nodes-comfy/dinkster-pack.toml").read_text()
    )
    assert manifest["pack"]["namespaces"].count(schema.node_type) == 1


def test_sam3_preview_schema_import_is_cold() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_nodes_comfy.sam3_preview import SAM3_TrackPreview; "
            "SAM3_TrackPreview.define_schema(); "
            "assert not any(x == 'torch' or x.startswith(('torch.', 'dinkster_inference.')) "
            "for x in sys.modules)",
        ],
        check=True,
    )
