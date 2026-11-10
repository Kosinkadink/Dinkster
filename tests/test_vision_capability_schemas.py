from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest
from dinkster_graph import Graph, GraphNode, TypedLiteral, validate
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker

from tests.async_scenarios import run_scenario


def test_background_removal_registers_without_torch_or_inference() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_api.v1 import ASSET_TYPE, BACKGROUND_REMOVAL_TYPE, MASK_TYPE, TypeExpr\n"
            "from dinkster_vision_background_removal import NODES\n"
            "from dinkster_vision_background_removal.background_removal import "
            "BACKGROUND_REMOVAL_NODES\n"
            "from dinkster_workers import load_manifest\n"
            "manifest = load_manifest('packages/dinkster-nodes-vision-background-removal/"
            "dinkster-pack.toml')\n"
            "for node in BACKGROUND_REMOVAL_NODES:\n"
            "    assert node in NODES\n"
            "    assert node.define_schema().node_type.startswith(manifest.namespaces[0] + '.')\n"
            "loader, remove = [node.define_schema() for node in BACKGROUND_REMOVAL_NODES]\n"
            "assert loader.inputs[0].id == 'bg_removal_name'\n"
            "assert loader.inputs[0].type == TypeExpr.concrete(ASSET_TYPE)\n"
            "assert loader.inputs[0].widget.kind == 'model/background-removal'\n"
            "assert loader.outputs[0].type == TypeExpr.concrete(BACKGROUND_REMOVAL_TYPE)\n"
            "assert [port.id for port in remove.inputs] == ['bg_removal_model', 'image']\n"
            "assert remove.outputs[0].type == TypeExpr.concrete(MASK_TYPE)\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_bbox_crop_schema_registers_without_torch_or_inference() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_image import NODES\n"
            "from dinkster_image.ported.bbox_crop import CropByBBoxes\n"
            "from dinkster_workers import load_manifest\n"
            "manifest = load_manifest('packages/dinkster-nodes-image/dinkster-pack.toml')\n"
            "schema = CropByBBoxes.define_schema()\n"
            "assert CropByBBoxes in NODES\n"
            "assert schema.node_type.startswith(manifest.namespaces[0] + '.')\n"
            "assert [port.id for port in schema.inputs] == "
            "['image', 'bboxes', 'output_width', 'output_height', 'padding', 'keep_aspect']\n"
            "assert schema.inputs[1].force_input\n"
            "assert schema.inputs[1].default == {'x': 0, 'y': 0, 'width': 512, 'height': 512}\n"
            "assert [option.value for option in schema.inputs[-1].widget.options] == "
            "['stretch', 'pad']\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_native_rtdetr_schemas_register_without_torch_or_inference() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_api.v1 import BOUNDING_BOX_TYPE, IMAGE_TYPE, TypeExpr\n"
            "from dinkster_vision_rtdetr import NODES as RTDETR_NODES\n"
            "from dinkster_vision_rtdetr.rtdetr import RTDETR_detect, DrawBBoxes\n"
            "from dinkster_vision_sam3 import NODES as SAM3_NODES\n"
            "from dinkster_vision_sam3.sam3 import SAM3_TRACKING_NODES\n"
            "from dinkster_vision_sam3.sam3_detection import SAM3_Detect\n"
            "from dinkster_workers import load_manifest\n"
            "for pack, nodes, selected in (\n"
            "    ('rtdetr', RTDETR_NODES, (RTDETR_detect, DrawBBoxes)),\n"
            "    ('sam3', SAM3_NODES, (*SAM3_TRACKING_NODES, SAM3_Detect)),\n"
            "):\n"
            "    manifest = load_manifest(f'packages/dinkster-nodes-vision-{pack}/"
            "dinkster-pack.toml')\n"
            "    for node in selected:\n"
            "        schema = node.define_schema()\n"
            "        assert node in nodes\n"
            "        assert schema.node_type.startswith(manifest.namespaces[0] + '.')\n"
            "        assert schema.category == 'image/detection'\n"
            "detect = RTDETR_detect.define_schema()\n"
            "assert [p.id for p in detect.inputs] == "
            "['model', 'image', 'threshold', 'class_name', 'max_detections']\n"
            "assert detect.outputs[0].type == TypeExpr.concrete(BOUNDING_BOX_TYPE)\n"
            "options = detect.inputs[3].widget.options\n"
            "assert len(options) == 81 and options[0].value == 'all'\n"
            "assert options[-1].value == 'toothbrush'\n"
            "draw = DrawBBoxes.define_schema()\n"
            "assert [p.id for p in draw.inputs] == ['image', 'bboxes']\n"
            "assert not draw.inputs[0].required and draw.inputs[1].required\n"
            "assert draw.outputs[0].type == TypeExpr.concrete(IMAGE_TYPE)\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("pack", "module", "node_type", "input_id"),
    (
        (
            "upscale",
            "dinkster_upscale",
            "dinkster.upscale.upscale_model_loader",
            "model_name",
        ),
        (
            "vision-background-removal",
            "dinkster_vision_background_removal",
            "dinkster.vision.background.removal.load_background_removal_model",
            "bg_removal_name",
        ),
    ),
)
def test_isolated_vision_accepts_canonical_typed_literals_without_aliases(
    pack: str, module: str, node_type: str, input_id: str
) -> None:
    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        import_module(module).register_types(registry)
        manifest = (
            Path(__file__).parent.parent / f"packages/dinkster-nodes-{pack}/dinkster-pack.toml"
        )
        worker = IsolatedWorker(manifest, registry, python=sys.executable)
        await worker.start()
        try:
            for stamp in ("ASSET", "dinkster.asset"):
                graph = Graph(
                    nodes={
                        "vision": GraphNode(
                            node_type,
                            {
                                input_id: TypedLiteral(
                                    stamp,
                                    {
                                        "digest": "blake3:" + "a" * 64,
                                        "name": "upscale.pth",
                                    },
                                ),
                            },
                        ),
                    }
                )
                errors = [
                    d
                    for d in validate(graph, worker.schemas, ["vision"], known_types=registry)
                    if d.severity == "error"
                ]
                if stamp == "ASSET":
                    assert errors == []
                else:
                    assert [d.code for d in errors] == ["typed-literal-unknown-type"]
        finally:
            await worker.close()

    run_scenario(scenario())
