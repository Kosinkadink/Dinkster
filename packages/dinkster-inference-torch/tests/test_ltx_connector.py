"""Native LTX text-embedding connector math and encoder-path goldens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from clip_fill import fill_state_dict, fill_value
from dinkster_inference import (
    GEMMA3_LTX_12B_CONFIG,
    LTX_TEXT_CONNECTOR_CONFIG,
    Conditioning,
    LtxConnectorConfig,
    LtxGemmaPromptTokens,
    ltx_connector_layout,
)
from dinkster_inference_torch import (
    LtxDualTextProjection,
    LtxEmbeddingsConnector,
    LtxGemmaTextEncoder,
    LtxTextConnectors,
)

GOLDENS = json.loads(
    (
        Path(__file__).parents[3] / "tests" / "goldens" / "ltx_embeddings_connector_goldens.json"
    ).read_text()
)


def dec(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"])


def tiny_config() -> LtxConnectorConfig:
    return LtxConnectorConfig(**GOLDENS["tiny"]["config"])


def build_tiny() -> LtxEmbeddingsConnector:
    connector = LtxEmbeddingsConnector(tiny_config())
    connector.load_state_dict(fill_state_dict(GOLDENS["tiny"]["state_dict"]), strict=True)
    return connector


def test_tiny_state_layout_matches_executed_reference() -> None:
    ours = sorted((key, list(value.shape)) for key, value in build_tiny().state_dict().items())
    assert ours == [(key, list(shape)) for key, shape in GOLDENS["tiny"]["state_dict"]]


def test_full_state_layout_and_strict_load_match_executed_reference() -> None:
    with torch.device("meta"):
        tower = LtxEmbeddingsConnector(LTX_TEXT_CONNECTOR_CONFIG)
        pair = LtxTextConnectors(LTX_TEXT_CONNECTOR_CONFIG)
    expected = [(key, tuple(shape)) for key, shape in GOLDENS["connector_layout"]]
    assert sorted(ltx_connector_layout(LTX_TEXT_CONNECTOR_CONFIG).items()) == expected
    assert sorted((key, tuple(value.shape)) for key, value in tower.state_dict().items()) == (
        expected
    )
    assert sorted(pair.state_dict()) == sorted(
        f"{prefix}{key}"
        for prefix in ("video_embeddings_connector.", "audio_embeddings_connector.")
        for key, _ in expected
    )
    state = {key: torch.empty(shape, device="meta") for key, shape in expected}
    tower.load_state_dict(state, strict=True, assign=True)
    del state["learnable_registers"]
    with pytest.raises(RuntimeError, match="Missing key"):
        tower.load_state_dict(state, strict=True, assign=True)


# The stored connector goldens are not bit-portable across hosts: the
# feed-forward GELU-tanh kernel rounds the sub-16-element remainder tail
# of each at::parallel_for chunk 1-2 ulp differently from the AVX2
# vectorized main loop, so a replay diverges from the golden exactly when
# divup(numel, threads) % 16 != 0 (issue #581; on a 64-thread Threadripper
# the 33024-element case drifts at 32/64 threads and the 32768-element
# cases at 24/48, so no count in the 24-64 host-default range is safe for
# all replays; counts <= 16 partition tail-free but would require pinning
# global thread state in the test). The executed
# reference reproduces the identical divergence against the stored golden
# in the same process, so this is kernel-tail drift, never port drift; the
# generator's in-process replay keeps the transcription contract bit-exact
# at generation time. Observed drift: max abs 4.768371e-07 (encoder replay
# at 24 threads; forward cases 3.576279e-07), zero sign crossings. Relative
# drift reaches 4.49e-04 on a near-zero output element, so the bound must
# be atol-dominated: rtol 0 with atol 2e-6 carries 4.2x headroom over the
# worst observed absolute drift.
CONNECTOR_GOLDEN_RTOL = 0.0
CONNECTOR_GOLDEN_ATOL = 2e-6


@pytest.mark.parametrize("case", range(len(GOLDENS["tiny"]["cases"])))
def test_forward_matches_executed_reference(case: int) -> None:
    spec = GOLDENS["tiny"]["cases"][case]
    tokens = fill_value(spec["input_fill_key"], spec["input"]["shape"])
    assert torch.equal(tokens, dec(spec["input"]))
    with torch.no_grad():
        got = build_tiny()(tokens)
    torch.testing.assert_close(
        got, dec(spec["output"]), rtol=CONNECTOR_GOLDEN_RTOL, atol=CONNECTOR_GOLDEN_ATOL
    )


def test_connector_validates_input_shape() -> None:
    connector = build_tiny()
    with pytest.raises(ValueError, match="connector tokens must be"):
        connector(torch.zeros(5, 8))
    with pytest.raises(ValueError, match="connector tokens must be"):
        connector(torch.zeros(1, 5, 12))


def _tiny_connectors() -> LtxTextConnectors:
    connectors = LtxTextConnectors(tiny_config())
    expected = {
        f"{prefix}{key}": shape
        for prefix in (
            GOLDENS["encoder"]["video_fill_prefix"],
            GOLDENS["encoder"]["audio_fill_prefix"],
        )
        for key, shape in GOLDENS["tiny"]["state_dict"]
    }
    assert sorted(connectors.state_dict()) == sorted(expected)
    connectors.load_state_dict(fill_state_dict(sorted(expected.items())), strict=True)
    return connectors


def _tiny_projection() -> torch.nn.Linear:
    ((key, shape),) = GOLDENS["encoder"]["projection_fill"]
    projection = torch.nn.Linear(shape[1], shape[0], bias=False)
    with torch.no_grad():
        projection.weight.copy_(fill_value(key, shape))
    return projection


class _StackModel:
    """Returns a pinned reference stack; records what the encoder sent."""

    def __init__(self, stack: torch.Tensor) -> None:
        self.config = GEMMA3_LTX_12B_CONFIG
        self.embed_tokens = type("Embedding", (), {"weight": torch.empty(0)})()
        self.stack = stack
        self.ids: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None

    def __call__(
        self, ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert attention_mask is not None
        self.ids = ids
        self.mask = attention_mask
        return self.stack.clone()


def _golden_prompt_tokens() -> LtxGemmaPromptTokens:
    mask = tuple(GOLDENS["encoder"]["attention_mask"])
    return LtxGemmaPromptTokens(tuple(range(2, 2 + len(mask))), mask, (0,) * len(mask))


def test_encoder_connector_path_matches_executed_reference() -> None:
    model = _StackModel(dec(GOLDENS["encoder"]["stack"]))
    encoder = LtxGemmaTextEncoder(
        cast("Any", model), _tiny_projection(), cast("Any", None), connectors=_tiny_connectors()
    )
    tokens = _golden_prompt_tokens()
    with torch.no_grad():
        got: Conditioning[torch.Tensor] = encoder.encode_tokens(tokens)
    assert model.ids is not None and model.mask is not None
    assert model.ids.tolist() == [list(tokens.ids)]
    assert model.mask.tolist() == [list(tokens.attention_mask)]
    assert got.pooled is None
    torch.testing.assert_close(
        got.embeddings,
        dec(GOLDENS["encoder"]["output"]),
        rtol=CONNECTOR_GOLDEN_RTOL,
        atol=CONNECTOR_GOLDEN_ATOL,
    )


def test_encoder_refuses_connectors_on_the_dual_projection() -> None:
    model = _StackModel(dec(GOLDENS["encoder"]["stack"]))
    dual = LtxDualTextProjection(in_features=12, video_features=4, audio_features=2)
    with pytest.raises(ValueError, match="apply only after the single projection"):
        LtxGemmaTextEncoder(
            cast("Any", model), dual, cast("Any", None), connectors=_tiny_connectors()
        )
