from __future__ import annotations

import pytest
from dinkster_inference import (
    FLOAT32,
    TAESDConfig,
    TAESDDetectError,
    TensorGeometry,
    detect_taesd_config,
    taesd_descriptor,
    taesd_layout,
)


def geometries(role: str) -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, FLOAT32) for key, shape in taesd_layout(role).items()}  # type: ignore[arg-type]


def test_detects_independent_family_role_artifacts() -> None:
    assert detect_taesd_config(geometries("encoder"), family="sd15") == TAESDConfig(
        "sd15", "encoder"
    )
    assert detect_taesd_config(geometries("decoder"), family="sdxl") == TAESDConfig(
        "sdxl", "decoder"
    )
    assert taesd_descriptor("sd15").id != taesd_descriptor("sdxl").id


def test_detection_is_strict_and_wrong_width_refuses() -> None:
    data = geometries("decoder")
    data["1.weight"] = TensorGeometry((64, 8, 3, 3), FLOAT32)
    with pytest.raises(TAESDDetectError, match="wrong tensor geometry"):
        detect_taesd_config(data, family="sd15", role="decoder")


def test_layout_pins_architecture_listing() -> None:
    assert len(taesd_layout("encoder")) == 67
    assert len(taesd_layout("decoder")) == 67
    assert taesd_layout("decoder")["1.weight"] == (64, 4, 3, 3)
