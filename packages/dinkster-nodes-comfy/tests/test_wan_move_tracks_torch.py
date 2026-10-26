"""CPU Wan Move track construction."""

from __future__ import annotations

import torch
from dinkster_nodes_comfy.wan_move_tracks import (
    GenerateTracks,
    WanMoveConcatTrack,
    WanMoveTracksFromCoords,
)


def test_tracks_from_single_quote_json_list_strings_and_mask_visibility() -> None:
    source = "[{'x': 2, 'y': 3}, {'x': 5, 'y': 7}]"
    second = "[{'x': 11, 'y': 13}, {'x': 17, 'y': 19}]"
    mask = torch.tensor([[[0.0, 0.0]], [[0.0, 1.0]]])
    result = WanMoveTracksFromCoords.execute(track_coords=[source, second], track_mask=mask)
    tracks = result["TRACKS"]
    assert result["track_length"] == 2
    assert tracks["track_path"].dtype is torch.float32
    assert tracks["track_visibility"].dtype is torch.bool
    torch.testing.assert_close(
        tracks["track_path"],
        torch.tensor([[[2.0, 3.0], [11.0, 13.0]], [[5.0, 7.0], [17.0, 19.0]]]),
    )
    assert torch.equal(tracks["track_visibility"], torch.tensor([[False], [True]]))


def test_tracks_from_single_track_and_invalid_json_match_reference_failure() -> None:
    result = WanMoveTracksFromCoords.execute(track_coords="[{'x': 2, 'y': 3}]")
    assert result["TRACKS"]["track_path"].shape == (1, 1, 2)
    try:
        WanMoveTracksFromCoords.execute(track_coords="invalid")
    except IndexError:
        pass
    else:
        raise AssertionError("invalid track data must retain the reference failure")


def test_generate_tracks_schema_and_non_square_perpendicular_spread() -> None:
    schema = GenerateTracks.define_schema()
    assert schema.search_terms == ("motion paths", "camera movement", "trajectory")
    assert [item.id for item in schema.inputs] == [
        "width",
        "height",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "num_frames",
        "num_tracks",
        "track_spread",
        "bezier",
        "mid_x",
        "mid_y",
        "interpolation",
        "track_mask",
    ]
    assert [item.default for item in schema.inputs[:13]] == [
        832,
        480,
        0.0,
        0.0,
        1.0,
        1.0,
        81,
        5,
        0.025,
        False,
        0.5,
        0.5,
        None,
    ]
    assert schema.inputs[13].required is False

    result = GenerateTracks.execute(
        width=200,
        height=100,
        start_x=0.0,
        start_y=0.0,
        end_x=1.0,
        end_y=0.0,
        num_frames=2,
        num_tracks=3,
        track_spread=0.1,
    )
    path = result["TRACKS"]["track_path"]
    assert path.dtype is torch.float32 and path.device.type == "cpu"
    torch.testing.assert_close(path[0], torch.tensor([[0.0, -15.0], [0.0, 0.0], [0.0, 15.0]]))
    torch.testing.assert_close(path[1], torch.tensor([[200.0, -15.0], [200.0, 0.0], [200.0, 15.0]]))


def test_generate_tracks_timing_bezier_one_frame_zero_tangent_and_mask() -> None:
    for interpolation, midpoint in (
        ("linear", 50.0),
        ("ease_in", 25.0),
        ("ease_out", 75.0),
        ("ease_in_out", 50.0),
        ("constant", 0.0),
    ):
        path = GenerateTracks.execute(
            width=100,
            height=100,
            start_x=0.0,
            start_y=0.0,
            end_x=1.0,
            end_y=0.0,
            num_frames=3,
            num_tracks=1,
            track_spread=0.0,
            interpolation=interpolation,
        )["TRACKS"]["track_path"]
        torch.testing.assert_close(path[1, 0], torch.tensor([midpoint, 0.0]))

    bezier = GenerateTracks.execute(
        width=100,
        height=200,
        start_x=0.0,
        start_y=0.0,
        end_x=1.0,
        end_y=1.0,
        mid_x=1.0,
        mid_y=0.0,
        bezier=True,
        num_frames=3,
        num_tracks=1,
        track_spread=0.0,
    )["TRACKS"]["track_path"]
    torch.testing.assert_close(bezier[1, 0], torch.tensor([75.0, 50.0]))

    zero = GenerateTracks.execute(
        width=100,
        height=100,
        start_x=0.5,
        start_y=0.5,
        end_x=0.5,
        end_y=0.5,
        num_frames=1,
        num_tracks=3,
        track_spread=0.1,
        track_mask=torch.zeros((1, 2, 2)),
    )["TRACKS"]
    torch.testing.assert_close(
        zero["track_path"][0], torch.tensor([[40.0, 50.0], [50.0, 50.0], [60.0, 50.0]])
    )
    assert torch.equal(zero["track_visibility"], torch.tensor([[False]]))


def test_concat_tracks_preserves_source_when_optional_input_absent() -> None:
    first = {
        "track_path": torch.tensor([[[1.0, 2.0]]]),
        "track_visibility": torch.tensor([[True]]),
        "extra": "preserved only on passthrough",
    }
    assert WanMoveConcatTrack.execute(tracks_1=first)["TRACKS"] is first
    second = {
        "track_path": torch.tensor([[[3.0, 4.0], [5.0, 6.0]]]),
        "track_visibility": torch.tensor([[False, True]]),
    }
    result = WanMoveConcatTrack.execute(tracks_1=first, tracks_2=second)["TRACKS"]
    torch.testing.assert_close(
        result["track_path"], torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    )
    assert torch.equal(result["track_visibility"], torch.tensor([[True, False, True]]))
    assert first["extra"] == "preserved only on passthrough"
