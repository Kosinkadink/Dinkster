"""Retained paired preview streams preserve independent batch sizes."""

from dinkster_image import NODES as IMAGE_NODES


def test_exact_image_compare_has_no_outputs_and_previews_each_nonempty_side() -> None:
    from collections.abc import Mapping
    from io import BytesIO

    import numpy as np
    from dinkster_image.ported.compare import NativeImageCompare
    from dinkster_schema import schema_from_wire, schema_to_wire, use_reporter
    from PIL import Image

    schema = NativeImageCompare.schema()
    assert NativeImageCompare in IMAGE_NODES
    assert schema.node_type == "dinkster.image.image_compare" and not schema.aliases
    assert [spec.id for spec in schema.inputs] == ["image_a", "image_b", "compare_view"]
    compare_view = schema.input("compare_view")
    assert compare_view is not None and compare_view.required
    assert schema.outputs == () and schema.output_node and schema.emits_previews
    assert schema_from_wire(schema_to_wire(schema)) == schema
    a = np.zeros((2, 2, 3, 3), dtype=np.float32)
    a[0, ..., 0] = 0.03
    a[1, ..., 1] = 0.61
    b = np.zeros((1, 3, 1, 3), dtype=np.float32)
    b[..., 2] = 0.91
    reports: list[tuple[Mapping[str, object], bytes | None]] = []

    def capture(name: str, data: Mapping[str, object], blob: bytes | None) -> None:
        assert name == "preview"
        reports.append((data, blob))

    with use_reporter(capture):
        assert NativeImageCompare.execute(image_a=a, image_b=b) == {}
        assert NativeImageCompare.execute(image_a=np.empty((0, 2, 3, 3))) == {}
    assert [(data["stream"], data["frameIndex"], data["frameCount"]) for data, _ in reports] == [
        ("image_a", 0, 2),
        ("image_a", 1, 2),
        ("image_b", 0, 1),
    ]
    for (_, blob), pixels in zip(reports, (a[0], a[1], b[0]), strict=True):
        assert blob is not None
        with Image.open(BytesIO(blob)) as decoded:
            np.testing.assert_array_equal(np.array(decoded), (pixels * 255).astype(np.uint8))
