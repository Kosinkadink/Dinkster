"""Structured prompt output preserves source content and selected style."""

from dinkster_nodes_comfy.json_prompt import BuildJsonPromptIdeogram


def test_json_prompt_photo_preserves_text_elements_and_palette_order() -> None:
    elements = [{"description": "subject", "x": 0.25}]
    palette = {"second": "#ab12cd", "first": "#00ef45", "empty": "", "number": 7}
    result = BuildJsonPromptIdeogram.execute(
        element=elements,
        style="photo",
        high_level_description="  portrait  ",
        background="studio",
        aesthetics="muted",
        lighting="side",
        medium="photograph",
        color_palette=palette,
        **{"style.photo": "35mm", "style.art_style": "ignored"},
    )["prompt"]
    assert result == {
        "high_level_description": "  portrait  ",
        "style_description": {
            "aesthetics": "muted",
            "lighting": "side",
            "photo": "35mm",
            "medium": "photograph",
            "color_palette": ["#AB12CD", "#00EF45"],
        },
        "compositional_deconstruction": {"background": "studio", "elements": elements},
    }
    assert palette == {"second": "#ab12cd", "first": "#00ef45", "empty": "", "number": 7}
    assert elements == [{"description": "subject", "x": 0.25}]


def test_json_prompt_none_omits_style_and_blank_description() -> None:
    assert BuildJsonPromptIdeogram.execute(
        element="not an array",
        style="none",
        high_level_description=" \n ",
        color_palette=["#abcdef"],
    )["prompt"] == {"compositional_deconstruction": {"background": "", "elements": []}}


def test_json_prompt_art_style_omits_empty_palette_and_photo() -> None:
    assert BuildJsonPromptIdeogram.execute(
        element=[],
        style="art_style",
        medium="painting",
        **{"style.art_style": "watercolor", "style.photo": "ignored"},
    )["prompt"] == {
        "style_description": {
            "aesthetics": "",
            "lighting": "",
            "medium": "painting",
            "art_style": "watercolor",
        },
        "compositional_deconstruction": {"background": "", "elements": []},
    }
