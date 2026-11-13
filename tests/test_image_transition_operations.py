from __future__ import annotations

import numpy as np
from dinkster_image.transition import ImageTransition


def _image(values: list[float], *, height: int = 1, width: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape((-1, 1, 1, 1))
    return array.repeat(height, axis=1).repeat(width, axis=2)


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
