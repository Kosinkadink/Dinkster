"""Qwen Image torch-free text contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
from dinkster_inference import BFLOAT16, TensorGeometry
from dinkster_inference.qwen_bpe import load_qwen_bpe
from dinkster_inference.qwen_image_text import (
    QWEN_IMAGE_TEXT_CONFIG,
    QwenImageOutputSelection,
    QwenImagePrompt,
    QwenImageTextConfig,
    QwenImageTextDetectError,
    QwenImageTokenPlan,
    detect_qwen_image_text_config,
    format_qwen_image_prompt,
    plan_qwen_image_token_rows,
    qwen_image_text_layout,
    select_qwen_image_output,
)


def exact_geometries() -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, BFLOAT16) for key, shape in qwen_image_text_layout().items()}


def test_qwen_image_text_config_and_special_tokens_are_exact_and_immutable() -> None:
    config = QWEN_IMAGE_TEXT_CONFIG
    assert config == QwenImageTextConfig()
    assert (
        config.architecture,
        config.vocab_size,
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.max_position_embeddings,
    ) == ("qwen2.5_vl_7b_qwen_image", 152064, 3584, 18944, 28, 28, 4, 128000)
    assert (config.head_dim, config.rms_norm_eps, config.rope_theta) == (
        128,
        1e-6,
        1_000_000.0,
    )
    assert config.rope_dims == (16, 24, 24)
    assert config.qkv_bias is True
    assert (
        config.pad_token_id,
        config.im_start_token_id,
        config.image_token_id,
        config.user_token_id,
        config.newline_token_id,
    ) == (151643, 151644, 151655, 872, 198)
    assert (
        config.vision_hidden_size,
        config.vision_output_size,
        config.vision_intermediate_size,
        config.vision_heads,
        config.vision_layers,
        config.vision_patch,
        config.vision_spatial_merge,
    ) == (1280, 3584, 3420, 16, 32, (2, 14, 14), 2)
    with pytest.raises(FrozenInstanceError):
        config.hidden_size = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="exact staged profile"):
        replace(config, hidden_size=1)
    with pytest.raises(ValueError, match="exact staged profile"):
        replace(config, qkv_bias=1)  # type: ignore[arg-type]


def test_qwen_image_text_layout_is_complete_exact_and_detectable() -> None:
    layout = qwen_image_text_layout()
    assert len(layout) == 728
    assert layout["model.embed_tokens.weight"] == (152064, 3584)
    assert layout["model.norm.weight"] == (3584,)
    assert layout["model.layers.0.self_attn.q_proj.weight"] == (3584, 3584)
    assert layout["model.layers.27.self_attn.k_proj.bias"] == (512,)
    assert layout["model.layers.27.mlp.down_proj.weight"] == (3584, 18944)
    assert layout["visual.patch_embed.proj.weight"] == (1280, 3, 2, 14, 14)
    assert layout["visual.blocks.0.attn.qkv.weight"] == (3840, 1280)
    assert layout["visual.blocks.31.mlp.down_proj.bias"] == (1280,)
    assert layout["visual.merger.ln_q.weight"] == (1280,)
    assert layout["visual.merger.mlp.0.weight"] == (5120, 5120)
    assert layout["visual.merger.mlp.2.weight"] == (3584, 5120)
    with pytest.raises(TypeError):
        layout["model.norm.weight"] = (1,)  # type: ignore[index]
    assert detect_qwen_image_text_config(exact_geometries()) is QWEN_IMAGE_TEXT_CONFIG


@pytest.mark.parametrize(
    ("key", "shape"),
    (
        ("model.embed_tokens.weight", (152064, 3583)),
        ("model.layers.0.self_attn.v_proj.bias", (511,)),
        ("visual.patch_embed.proj.weight", (1280, 3, 14, 14)),
        ("visual.blocks.31.mlp.gate_proj.weight", (3419, 1280)),
        ("visual.merger.mlp.2.weight", (3583, 5120)),
    ),
)
def test_qwen_image_text_header_refuses_wrong_geometry(key: str, shape: tuple[int, ...]) -> None:
    geometries = exact_geometries()
    geometries[key] = TensorGeometry(shape, BFLOAT16)
    with pytest.raises(QwenImageTextDetectError, match=key):
        detect_qwen_image_text_config(geometries)


def test_qwen_image_text_header_refuses_missing_extra_and_wrong_role() -> None:
    missing = exact_geometries()
    del missing["model.layers.27.mlp.down_proj.weight"]
    with pytest.raises(QwenImageTextDetectError, match="missing model.layers.27"):
        detect_qwen_image_text_config(missing)

    extra = exact_geometries()
    extra["lm_head.weight"] = TensorGeometry((152064, 3584), BFLOAT16)
    with pytest.raises(QwenImageTextDetectError, match="unexpected key lm_head.weight"):
        detect_qwen_image_text_config(extra)

    with pytest.raises(QwenImageTextDetectError, match="not a Qwen Image text role"):
        detect_qwen_image_text_config(
            {"model.embed_tokens.weight": TensorGeometry((1, 1), BFLOAT16)}
        )


def test_qwen_image_text_and_edit_templates_are_exact() -> None:
    text = format_qwen_image_prompt("paint a red fox")
    assert text.preformatted is False
    assert text.image_count == 0
    assert text.text == (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\npaint a red fox<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    image = format_qwen_image_prompt("make it blue", image_count=1)
    assert image.image_count == 1
    assert image.text == (
        "<|im_start|>system\nDescribe the key features of the input image (color, shape, "
        "size, texture, objects, background), then explain how the user's text instruction "
        "should alter or modify the image. Generate a new image that meets the user's "
        "requirements while maintaining consistency with the original input where "
        "appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|>"
        "<|vision_end|>make it blue<|im_end|>\n<|im_start|>assistant\n"
    )

    plus = format_qwen_image_prompt("combine them", image_count=3)
    assert plus.image_count == 3
    assert plus.text.count("<|image_pad|>") == 3
    assert (
        "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
        "Picture 2: <|vision_start|><|image_pad|><|vision_end|>"
        "Picture 3: <|vision_start|><|image_pad|><|vision_end|>combine them"
    ) in plus.text

    plus_one = format_qwen_image_prompt("change it", image_count=1, edit_plus=True)
    assert "Picture 1: <|vision_start|><|image_pad|><|vision_end|>change it" in plus_one.text
    plus_zero = format_qwen_image_prompt("draw it", edit_plus=True)
    assert plus_zero.image_count == 0
    assert "<|im_start|>user\ndraw it<|im_end|>" in plus_zero.text
    sparse_plus = format_qwen_image_prompt(
        "change it", image_count=1, edit_plus=True, image_slots=(2,)
    )
    assert "Picture 2: <|vision_start|><|image_pad|><|vision_end|>change it" in sparse_plus.text


@pytest.mark.parametrize("prefix", ("<|im_start|>", "<|start_header_id|>"))
def test_qwen_image_preformatted_prompt_bypasses_every_template(prefix: str) -> None:
    raw = prefix + "raw prompt"
    formatted = format_qwen_image_prompt(raw, template="invalid without a slot")
    assert formatted.text == raw
    assert formatted.preformatted is True


def test_qwen_image_custom_template_and_image_cardinality_fail_closed() -> None:
    assert format_qwen_image_prompt("cat", template="before {} after").text == ("before cat after")
    for template in (
        "missing",
        "{} twice {}",
        "{{}}",
        "{}{0}",
        "{name}",
        "broken {",
    ):
        with pytest.raises(ValueError, match="exactly one text slot"):
            format_qwen_image_prompt("cat", template=template)
    for image_count in (-1, 4, True):
        with pytest.raises(ValueError, match="zero to three images"):
            format_qwen_image_prompt("cat", image_count=image_count)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="image cardinality"):
        format_qwen_image_prompt("cat", image_count=1, template="before {} after")
    with pytest.raises(TypeError, match="edit_plus must be a bool"):
        format_qwen_image_prompt("cat", edit_plus=1)  # type: ignore[arg-type]
    for slots in ((1,), (0,), (2, 2), (3, 1)):
        with pytest.raises(ValueError, match="slots must be unique ascending"):
            format_qwen_image_prompt("cat", image_count=2, edit_plus=True, image_slots=slots)


def test_qwen_image_prompt_constructor_refuses_false_format_and_image_claims() -> None:
    with pytest.raises(ValueError, match="image cardinality"):
        QwenImagePrompt("<|im_start|>plain", 1, True)


def test_qwen_image_placeholder_substitution_plan_preserves_row_order() -> None:
    plan = plan_qwen_image_token_rows(((10, 151655, 11),), image_count=1)
    assert plan.rows == ((10, 151655, 11),)
    assert plan.image_slots == ((0, 1),)
    assert plan.image_count == 1
    with pytest.raises(FrozenInstanceError):
        plan.image_count = 0  # type: ignore[misc]

    plus = plan_qwen_image_token_rows(((151655, 10, 151655, 11, 151655),), image_count=3)
    assert plus.image_slots == ((0, 0), (0, 2), (0, 4))


@pytest.mark.parametrize(
    "plan",
    (
        (((151655, 7),), ((0, 1),), 1),
        (((151655,),), (), 0),
        (((7,),), ((0, 0),), 1),
    ),
)
def test_qwen_image_token_plan_constructor_refuses_false_placeholder_claims(
    plan: tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...], int],
) -> None:
    with pytest.raises(ValueError, match="exactly match placeholders"):
        QwenImageTokenPlan(*plan)


def test_qwen_image_one_image_template_tokenizes_to_one_ordered_placeholder() -> None:
    prompt = format_qwen_image_prompt("make it blue", image_count=1)
    row = tuple(load_qwen_bpe().encode(prompt.text))
    plan = plan_qwen_image_token_rows((row,), image_count=prompt.image_count)
    assert len(plan.image_slots) == 1
    row_index, token_index = plan.image_slots[0]
    assert row_index == 0
    assert plan.rows[row_index][token_index] == QWEN_IMAGE_TEXT_CONFIG.image_token_id


@pytest.mark.parametrize(
    ("rows", "image_count", "match"),
    (
        ((), 0, "exactly one non-empty token row"),
        (((1,), (2,)), 0, "exactly one non-empty token row"),
        (((1, True),), 0, "integer token IDs"),
        (((-1,),), 0, "inside the vocabulary"),
        (((151655,),), 0, "image cardinality"),
        (((151655, 151655),), 1, "image cardinality"),
        (((1,),), 1, "image cardinality"),
    ),
)
def test_qwen_image_token_rows_and_image_cardinality_refuse(
    rows: tuple[tuple[int, ...], ...], image_count: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        plan_qwen_image_token_rows(rows, image_count=image_count)


def test_qwen_image_output_selection_slices_template_and_retains_mask() -> None:
    row = (151644, 20, 151644, 872, 198, 30, 31, 151644)
    selected = select_qwen_image_output((row,), ((1, 1, 1, 1, 1, 1, 0, 0),))
    assert selected.slice_start == 5
    assert selected.token_rows == ((30, 31, 151644),)
    assert selected.attention_mask == ((1, 0, 0),)


def test_qwen_image_output_selection_removes_all_one_mask() -> None:
    row = (151644, 20, 151644, 872, 198, 30)
    selected = select_qwen_image_output((row,), ((1, 1, 1, 1, 1, 1),))
    assert selected.token_rows == ((30,),)
    assert selected.attention_mask is None
    direct = QwenImageOutputSelection(((30,),), ((1,),), 5)
    assert direct.attention_mask is None
    explicit = select_qwen_image_output((row,), None, template_end=2)
    assert explicit.slice_start == 2
    assert explicit.token_rows == ((151644, 872, 198, 30),)
    assert explicit.attention_mask is None


@pytest.mark.parametrize(
    ("row", "match"),
    (
        ((151644, 1), "two im-start"),
        ((151644, 1, 151644, 872), "ambiguous user boundary"),
        ((151644, 1, 151644, 198), "ambiguous user boundary"),
    ),
)
def test_qwen_image_output_selection_refuses_ambiguous_template_boundaries(
    row: tuple[int, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        select_qwen_image_output((row,), None)


def test_qwen_image_output_selection_refuses_malformed_rows_masks_and_offsets() -> None:
    row = (151644, 1, 151644, 872, 198, 30)
    with pytest.raises(ValueError, match="exactly one non-empty token row"):
        select_qwen_image_output((row, row), None)
    with pytest.raises(ValueError, match="attention mask shape"):
        select_qwen_image_output((row,), ((1, 1),))
    with pytest.raises(ValueError, match="binary"):
        select_qwen_image_output((row,), ((1, 1, 1, 1, 1, 2),))
    for template_end in (True, -2, len(row)):
        with pytest.raises(ValueError, match="template_end"):
            select_qwen_image_output((row,), None, template_end=template_end)  # type: ignore[arg-type]
