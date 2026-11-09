"""SAM3 preview declarations remain independent of inference imports."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from dinkster_api.v1 import CORE_FLOAT, IMAGE_TYPE, SAM3_TRACK_DATA_TYPE, NumberWidget, TypeExpr
from dinkster_schema import claim_covers
from dinkster_vision_sam3 import NODES
from dinkster_vision_sam3.sam3_preview import SAM3_TrackPreview


def test_sam3_preview_public_schema_and_manifest() -> None:
    schema = SAM3_TrackPreview.define_schema()
    assert schema.node_type == "dinkster.vision.sam3.sam3_track_preview"
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
    assert NODES.count(SAM3_TrackPreview) == 1
    root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads(
        (root / "packages/dinkster-nodes-vision-sam3/dinkster-pack.toml").read_text()
    )
    assert (
        sum(claim_covers(claim, schema.node_type) for claim in manifest["pack"]["namespaces"]) == 1
    )


def test_sam3_preview_schema_import_is_cold() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_vision_sam3.sam3_preview import SAM3_TrackPreview; "
            "SAM3_TrackPreview.define_schema(); "
            "assert not any(x == 'torch' or x.startswith(('torch.', 'dinkster_inference.')) "
            "for x in sys.modules)",
        ],
        check=True,
    )
