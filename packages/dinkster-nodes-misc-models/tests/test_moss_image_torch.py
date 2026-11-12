from __future__ import annotations

import json

from dinkster_misc_models.moss_image import (
    MOSSImageLayoutPrompt,
)


def test_moss_layout_uses_enclosing_centralized_edges_and_text_precedence() -> None:
    elements = [
        {"bbox": [123.5, 98.2, 789.4, 900.1], "desc": "subject", "color_palette": ["#abcdef"]},
        {"bbox": [0, 0, 1000, 1000], "type": "text", "desc": "caption", "text": "quoted"},
    ]
    result = MOSSImageLayoutPrompt.execute(
        prompt="  scene  ", element=elements, width=1001, height=601, character_boxes="1, 2"
    )["prompt"]
    assert json.loads(result) == {
        "summary": {"caption": "scene"},
        "items": [
            {
                "label": "character",
                "caption": "subject",
                "palette": ["#abcdef"],
                "bbox": [-226, -402, 175, 402],
            },
            {
                "label": "text",
                "caption": "caption",
                "text": "quoted",
                "bbox": [-300, -500, 301, 501],
            },
        ],
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }
    assert elements[0]["bbox"] == [123.5, 98.2, 789.4, 900.1]
    assert (
        MOSSImageLayoutPrompt.execute(prompt="  unchanged  ", element=[], width=1001, height=601)[
            "prompt"
        ]
        == "  unchanged  "
    )
