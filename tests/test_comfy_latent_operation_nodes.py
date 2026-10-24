"""Latent operations preserve metadata, schedule boundaries, and frame ownership."""

from typing import Any, cast

import numpy as np
import pytest
from dinkster_nodes_comfy.latent_operations import (
    LATENT_OPERATION_NODES,
    LatentApplyOperation,
    LatentApplyOperationCFG,
    LatentOperationBlend,
    ReplaceVideoLatentFrames,
)
from test_comfy_latent_nodes import ArrayTensor


def test_latent_operation_interfaces_keep_exact_input_order_and_optional_groups() -> None:
    expected = {
        "LatentApplyOperation": ("samples", "operation"),
        "LatentApplyOperationCFG": ("model", "operation", "start_percent", "end_percent"),
        "LatentOperationBlend": ("reference", "strength"),
        "LatentOperationTonemapReinhard": ("multiplier",),
        "LatentOperationSharpen": ("sharpen_radius", "sigma", "alpha"),
        "ReplaceVideoLatentFrames": ("destination", "index", "source"),
    }
    assert {
        node.schema().node_type: tuple(port.id for port in node.schema().inputs)
        for node in LATENT_OPERATION_NODES
    } == expected
    assert not ReplaceVideoLatentFrames.schema().inputs[-1].required
    assert all(not port.required for port in LatentApplyOperationCFG.schema().inputs[-2:])


def test_apply_preserves_metadata_and_zero_strength_operation_keeps_input() -> None:
    samples = {"samples": np.array([2, -7]), "note": "keep"}
    result = cast(
        dict[str, Any],
        LatentApplyOperation.execute(samples=samples, operation=lambda latent: latent * 3)[
            "LATENT"
        ],
    )
    np.testing.assert_array_equal(result["samples"], [6, -21])
    np.testing.assert_array_equal(samples["samples"], [2, -7])
    assert result["note"] == "keep"
    operation = cast(
        Any, LatentOperationBlend.execute(reference=samples, strength=0)["LATENT_OPERATION"]
    )
    assert operation(latent=samples["samples"]) is samples["samples"]


class ScalarModel:
    def __init__(self) -> None:
        self.transform: Any = lambda args: args["conds_out"]

    def clone(self) -> "ScalarModel":
        return ScalarModel()

    def get_model_object(self, name: str) -> "ScalarModel":
        return self

    def percent_to_sigma(self, percent: float) -> float:
        return 10 * (1 - percent)

    def set_model_sampler_pre_cfg_function(self, transform: Any) -> None:
        self.transform = transform

    def predict(self, sigma: float, predictions: list[float]) -> list[float]:
        return self.transform({"sigma": np.array([sigma]), "conds_out": predictions.copy()})


@pytest.mark.parametrize(
    ("sigma", "expected"), [(8.01, [11, 3]), (8, [27, 3]), (4, [27, 3]), (3.99, [11, 3])]
)
def test_cfg_operation_applies_to_conditional_delta_inside_inclusive_schedule(
    sigma: float, expected: list[float]
) -> None:
    source = ScalarModel()
    model = cast(
        ScalarModel,
        LatentApplyOperationCFG.execute(
            model=source, operation=lambda latent: latent * 3, start_percent=0.2, end_percent=0.6
        )["MODEL"],
    )
    assert model.predict(sigma, [11, 3]) == expected
    assert source.predict(sigma, [11, 3]) == [11, 3]
    assert model.predict(6, [11]) == [33]


def test_video_replacement_uses_source_metadata_and_copies_destination_storage() -> None:
    destination = {"samples": ArrayTensor(np.arange(7).reshape(1, 1, 7, 1, 1)), "note": "dest"}
    source = {"samples": ArrayTensor(np.array([-2, -9]).reshape(1, 1, 2, 1, 1)), "note": "source"}
    result = cast(
        dict[str, Any],
        ReplaceVideoLatentFrames.execute(destination=destination, source=source, index=-2)[
            "LATENT"
        ],
    )
    np.testing.assert_array_equal(result["samples"].array.flatten(), [0, 1, 2, 3, 4, -2, -9])
    np.testing.assert_array_equal(
        cast(ArrayTensor, destination["samples"]).array.flatten(), np.arange(7)
    )
    assert result["note"] == "source"
    assert (
        ReplaceVideoLatentFrames.execute(destination=destination, source=source, index=6)["LATENT"]
        is destination
    )
    assert ReplaceVideoLatentFrames.execute(destination=destination)["LATENT"] is destination
