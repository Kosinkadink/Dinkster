from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from dinkster_nodes_image.transition import ImageTransition, _ease, _transition_mask


def _image(values: list[float], *, height: int = 1, width: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape((-1, 1, 1, 1))
    return array.repeat(height, axis=1).repeat(width, axis=2)


@pytest.mark.parametrize(
    "easing",
    (
        "linear",
        "ease_in",
        "ease_out",
        "ease_in_out",
        "bounce",
        "elastic",
        "glitchy",
        "exponential_ease_out",
    ),
)
def test_transition_easings_preserve_endpoints(easing: str) -> None:
    assert _ease(0.0, easing) == pytest.approx(0.0, abs=1e-12)
    expected_end = 1.0 + 0.1 * np.sin(40.0) if easing == "glitchy" else 1.0
    assert _ease(1.0, easing) == pytest.approx(expected_end, abs=1e-12)


@pytest.mark.parametrize(
    "transition",
    (
        "horizontal_slide",
        "vertical_slide",
        "box",
        "circle",
        "horizontal_door",
        "vertical_door",
        "fade",
    ),
)
def test_transition_shapes_preserve_zero_and_one_endpoints(transition: str) -> None:
    zero = _transition_mask(5, 7, 0.0, transition, 0.0)
    one = _transition_mask(5, 7, 1.0, transition, 0.0)
    if transition == "circle":
        assert zero.sum() == 1.0
        assert zero[2, 3] == 1.0
    else:
        np.testing.assert_array_equal(zero, 0)
    np.testing.assert_array_equal(one, 1)


def test_between_inputs_keeps_batches_and_inserts_endpoint_transitions() -> None:
    output = np.asarray(
        ImageTransition.execute(
            images={"first": _image([0, 1]), "second": _image([4, 5])},
            mode="between_inputs",
            transitioning_frames=3,
        )["image"]
    )
    np.testing.assert_allclose(output[:, 0, 0, 0], [0, 1, 1, 2.5, 4, 4, 5])


def test_kj_dynamic_transition_count_preserves_late_and_missing_inputs() -> None:
    output = ImageTransition.execute(
        images={"1": _image([1]), "3": _image([8]), "4": _image([99])},
        input_count=3,
        mode="between_inputs",
        transitioning_frames=2,
        shape_policy="resize_to_first",
    )["image"]
    np.testing.assert_array_equal(np.asarray(output)[:, 0, 0, 0], [1, 1, 0, 0, 0, 8, 8])


def test_within_batch_transitions_every_adjacent_pair() -> None:
    output = ImageTransition.execute(
        images={"batch": _image([0, 2, 6])},
        mode="within_batch",
        transitioning_frames=3,
    )["image"]
    np.testing.assert_allclose(np.asarray(output)[:, 0, 0, 0], [0, 1, 2, 2, 4, 6])


def test_join_batches_uses_start_index_and_replaces_the_first_tail() -> None:
    output = ImageTransition.execute(
        images={"first": _image([0, 1, 2, 3]), "second": _image([8, 9, 10])},
        mode="join_batches",
        start_index=2,
        transitioning_frames=2,
    )["image"]
    np.testing.assert_allclose(np.asarray(output)[:, 0, 0, 0], [0, 1, 2, 9, 10])


def test_join_batches_accepts_a_single_transition_frame() -> None:
    output = ImageTransition.execute(
        images={"first": _image([0, 1]), "second": _image([8, 9])},
        mode="join_batches",
        start_index=1,
        transitioning_frames=1,
    )["image"]
    np.testing.assert_array_equal(np.asarray(output)[:, 0, 0, 0], [0, 8, 9])


def test_reverse_changes_spatial_direction_without_changing_endpoints() -> None:
    batch = np.stack(
        (
            np.zeros((1, 4, 1), dtype=np.float32),
            np.ones((1, 4, 1), dtype=np.float32),
        ),
        axis=0,
    )
    forward = ImageTransition.execute(
        images={"batch": batch},
        mode="within_batch",
        transition="horizontal_slide",
        transitioning_frames=3,
    )["image"]
    reverse = ImageTransition.execute(
        images={"batch": batch},
        mode="within_batch",
        transition="horizontal_slide",
        transitioning_frames=3,
        reverse=True,
    )["image"]
    np.testing.assert_array_equal(np.asarray(forward)[0], batch[0])
    np.testing.assert_array_equal(np.asarray(forward)[-1], batch[1])
    np.testing.assert_array_equal(np.asarray(reverse)[0], batch[0])
    np.testing.assert_array_equal(np.asarray(reverse)[-1], batch[1])
    np.testing.assert_array_equal(np.asarray(forward)[1, 0, :, 0], [1, 1, 0, 0])
    np.testing.assert_array_equal(np.asarray(reverse)[1, 0, :, 0], [0, 0, 1, 1])


def test_transition_resize_and_blur_policies_are_explicit() -> None:
    first = np.zeros((1, 4, 4, 1), dtype=np.float32)
    second = np.ones((1, 2, 2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="dimensions must match"):
        ImageTransition.execute(images={"first": first, "second": second})
    resized = ImageTransition.execute(
        images={"first": first, "second": second},
        shape_policy="resize_to_first",
    )["image"]
    assert np.asarray(resized).shape == (4, 4, 4, 1)

    sharp = _transition_mask(7, 7, 0.5, "box", 0.0)
    blurred = _transition_mask(7, 7, 0.5, "box", 2.0)
    assert np.any((blurred > 0.0) & (blurred < 1.0))
    assert not np.array_equal(sharp, blurred)


def test_transition_modes_validate_dynamic_family_counts() -> None:
    with pytest.raises(ValueError, match="at least two"):
        ImageTransition.execute(images={"only": _image([0])})
    with pytest.raises(ValueError, match="exactly one"):
        ImageTransition.execute(images={"a": _image([0]), "b": _image([1])}, mode="within_batch")
    with pytest.raises(ValueError, match="exactly two"):
        ImageTransition.execute(
            images={"a": _image([0]), "b": _image([1]), "c": _image([2])},
            mode="join_batches",
        )
    with pytest.raises(ValueError, match="between_inputs requires at least 2"):
        ImageTransition.execute(images={"a": _image([0]), "b": _image([1])}, transitioning_frames=1)
    with pytest.raises(ValueError, match="join_batches requires at least 1"):
        ImageTransition.execute(
            images={"a": _image([0]), "b": _image([1])},
            mode="join_batches",
            transitioning_frames=0,
        )


def test_transition_dynamic_constructs_and_materialized_execution() -> None:
    schema = ImageTransition.define_schema()
    assert schema.version == 2
    assert tuple(combo.id for combo in schema.combos) == ("mode", "shape_policy")
    images = {"a": _image([0]), "b": _image([1])}
    actual = cast("Any", ImageTransition.execute)(
        images=images, mode="join_batches", **{"mode.start_index": 0}
    )
    expected = ImageTransition.execute(images=images, mode="join_batches", start_index=0)
    np.testing.assert_array_equal(actual["image"], expected["image"])
