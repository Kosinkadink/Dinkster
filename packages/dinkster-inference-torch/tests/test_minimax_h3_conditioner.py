from __future__ import annotations

import pytest
import torch
from dinkster_inference.minimax_h3_conditioner import minimax_h3_conditioner_layout
from dinkster_inference_torch.minimax_h3_conditioner import (
    MiniMaxH3ConditionerModel,
    MiniMaxH3VisionModel,
)
from dinkster_inference_torch.qwen_image_text import (
    QwenImageLanguageModel,
    _language_rope,  # pyright: ignore[reportPrivateUsage]
)


def _reduced_model() -> MiniMaxH3ConditionerModel:
    language = QwenImageLanguageModel.reduced(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=3,
        num_heads=2,
        num_kv_heads=1,
        rope_dims=(2, 1, 1),
        qkv_bias=False,
        qk_norm=True,
        final_norm=False,
        rope_theta=5_000_000.0,
        interleaved_mrope=True,
    )
    visual = MiniMaxH3VisionModel.reduced(
        hidden_size=8,
        output_size=16,
        intermediate_size=12,
        heads=2,
        layers=3,
        patch=(2, 2, 2),
        merge_size=2,
        position_embeddings=16,
        deepstack_layers=(0, 1, 2),
    )
    return MiniMaxH3ConditionerModel(language=language, visual=visual)


def test_full_meta_state_is_the_exact_902_key_conditioner_layout() -> None:
    with torch.device("meta"):
        model = MiniMaxH3ConditionerModel()
    assert model.model.shape.max_position_embeddings == 262144
    assert model.model.shape.architecture == "MiniMax H3"
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    assert actual == dict(minimax_h3_conditioner_layout().keys)
    state = {key: torch.empty(shape, device="meta") for key, shape in actual.items()}
    model.load_state_dict(state, strict=True, assign=True)


def test_reduced_text_and_vision_execute_with_deepstack() -> None:
    model = _reduced_model()
    assert model.model.shape.rope_theta == 5_000_000.0
    assert model.model.shape.interleaved_mrope is True
    ids = torch.tensor(((1, 2, 3, 4, 5, 6),))
    visual_mask = torch.tensor(((False, True, True, True, True, False),))
    patches = torch.randn(16, 24)
    output = model(
        ids,
        position_ids=torch.tensor(((0, 1, 2, 3, 4, 5), (0, 1, 2, 2, 3, 4), (0, 1, 2, 3, 3, 4))),
        visual_mask=visual_mask,
        image_patches=patches,
        image_grid=torch.tensor(((1, 4, 4),)),
    )
    assert output.shape == (1, 6, 16)


def test_conditioner_moves_language_indices_to_embedding_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _reduced_model()
    seen: dict[str, torch.device] = {}

    def embed(ids: torch.Tensor) -> torch.Tensor:
        seen["ids"] = ids.device
        return torch.empty((1, ids.shape[1], 16), device="meta")

    def forward_embeds(
        embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        assert attention_mask is not None
        seen["attention_mask"] = attention_mask.device
        seen["position_ids"] = position_ids.device
        return embeds

    def bound_device(_module: torch.nn.Module) -> torch.device:
        return torch.device("meta")

    monkeypatch.setattr(
        "dinkster_inference_torch.minimax_h3_conditioner.bound_compute_device",
        bound_device,
    )
    monkeypatch.setattr(model.model, "embed", embed)
    monkeypatch.setattr(model.model, "forward_embeds", forward_embeds)
    output = model(
        torch.tensor(((1, 2, 3),)),
        torch.ones((1, 3), dtype=torch.long),
        position_ids=torch.arange(3).unsqueeze(0),
    )
    assert output.device.type == "meta"
    assert seen == {
        "ids": torch.device("meta"),
        "attention_mask": torch.device("meta"),
        "position_ids": torch.device("meta"),
    }


def test_conditioner_refuses_mismatched_visual_placeholder_count() -> None:
    model = _reduced_model()
    ids = torch.tensor(((1, 2, 3, 4, 5, 6),))
    try:
        model(
            ids,
            position_ids=torch.tensor(((0, 1, 2, 3, 4, 5), (0, 1, 2, 2, 3, 4), (0, 1, 2, 3, 3, 4))),
            visual_mask=torch.tensor(((False, True, True, False, False, False),)),
            image_patches=torch.randn(16, 24),
            image_grid=torch.tensor(((1, 4, 4),)),
        )
    except ValueError as error:
        assert "match visual placeholders" in str(error)
    else:
        raise AssertionError("mismatched visual placeholder count was accepted")


def test_qwen3_vl_interleaved_mrope_matches_pinned_axis_replacement() -> None:
    positions = torch.tensor(((0, 1, 2, 3), (0, 1, 5, 6), (0, 1, 8, 9)))
    actual = _language_rope(
        positions,
        head_dim=8,
        theta=5_000_000.0,
        rope_dims=(2, 1, 1),
        device=torch.device("cpu"),
        interleaved_mrope=True,
    )
    inverse = 1.0 / (5_000_000.0 ** (torch.arange(0, 8, 2).float() / 8))
    frequencies = (inverse[None, :, None] @ positions[:, None, :].float()).transpose(1, 2)
    interleaved = frequencies[0].clone()
    interleaved[..., 1:3:3] = frequencies[1, ..., 1:3:3]
    interleaved[..., 2:3:3] = frequencies[2, ..., 2:3:3]
    embedding = torch.cat((interleaved, interleaved), -1)
    sine = embedding.sin().unsqueeze(0)
    expected = (embedding.cos().unsqueeze(0), sine[..., :4], -sine[..., 4:])
    for found, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(found, wanted, rtol=0.0, atol=0.0)


def test_visual_conditioning_requires_integer_three_axis_positions_and_bool_mask() -> None:
    model = _reduced_model()
    ids = torch.tensor(((1, 2, 3, 4, 5, 6),))
    patches = torch.randn(16, 24)
    grid = torch.tensor(((1, 4, 4),))
    mask = torch.tensor(((False, True, True, True, True, False),))
    for kwargs in (
        {"visual_mask": mask},
        {"visual_mask": mask, "position_ids": torch.arange(6).float().unsqueeze(0)},
        {"visual_mask": mask.long(), "position_ids": torch.arange(6).repeat(3, 1)},
    ):
        try:
            model(ids, image_patches=patches, image_grid=grid, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid vision positioning was accepted")


@pytest.mark.parametrize("tower", ("language", "vision"))
def test_conditioner_tower_prefetch_closes_when_a_block_raises(
    tower: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_inference_torch import minimax_h3_conditioner, qwen_image_text

    module = qwen_image_text if tower == "language" else minimax_h3_conditioner
    queue = object()
    closed: list[object] = []

    def make_queue(_blocks: torch.nn.ModuleList) -> object:
        return queue

    def pop_queue(_queue: object, _block: torch.nn.Module | None) -> None:
        return None

    monkeypatch.setattr(module, "make_prefetch_queue", make_queue)
    monkeypatch.setattr(module, "prefetch_queue_pop", pop_queue)
    monkeypatch.setattr(module, "close_prefetch_queue", closed.append)

    class RaisingBlock(torch.nn.Module):
        def forward(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("block failed")

    if tower == "language":
        model = _reduced_model().model
        model.layers = torch.nn.ModuleList([RaisingBlock()])
        with pytest.raises(RuntimeError, match="block failed"):
            model(torch.tensor(((1, 2),)))
    else:
        model = _reduced_model().visual
        model.blocks = torch.nn.ModuleList([RaisingBlock()])
        with pytest.raises(RuntimeError, match="block failed"):
            model(torch.randn(16, 24), torch.tensor(((1, 4, 4),)))
    assert closed == [queue]
