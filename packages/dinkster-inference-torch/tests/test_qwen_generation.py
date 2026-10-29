"""Native Qwen generation stream and session proofs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast
from unittest.mock import patch

import dinkster_inference_torch.qwen_generation as qwen_generation_module
import pytest
import torch
from dinkster_inference import (
    GenerationFinishReason,
    GenerationRequest,
    GenerationSamplerChain,
    GenerationSamplerKind,
    GenerationSamplerStage,
    GenerationSessionBusyError,
    GenerationSessionClosedError,
    GenerationStopConditions,
    GenerationTerminalEvent,
    GenerationTokenEvent,
    QwenTextConfig,
)
from dinkster_inference_torch import QwenGenerationProvider
from dinkster_inference_torch.qwen_generation import (
    _sample_token,  # pyright: ignore[reportPrivateUsage]
)

MODEL_ID = "native:qwen:test"


class _PrefetchHandle:
    def close(self) -> None:
        pass


class _PrefetchMechanism:
    def prefetch_enabled(self) -> bool:
        return True

    def prefetch(self, requests: Sequence[tuple[str, torch.dtype | None]]) -> _PrefetchHandle:
        del requests
        return _PrefetchHandle()


class _PrefetchBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = True
        self._mechanism = _PrefetchMechanism()

    def residency_prefetch(
        self,
    ) -> tuple[_PrefetchMechanism, tuple[tuple[str, torch.dtype | None], ...]] | None:
        if self.loaded:
            return None
        return self._mechanism, (("weight", None),)


class _Tokenizer:
    _prompts = {"p": [1, 2], "q": [3]}
    _pieces = {
        4: b"A",
        5: b"B",
        6: b"C",
        7: b"\xf0\x9f",
        8: b"\x98\x80",
        9: b"",
        10: b"S",
        11: b"T",
        12: b"O",
        13: b"P",
    }

    def encode(self, text: str) -> list[int]:
        return list(self._prompts.get(text, ()))

    def decode_bytes(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> bytes:
        del skip_special_tokens
        return b"".join(self._pieces[token_id] for token_id in token_ids)


class _InvalidTokenizer(_Tokenizer):
    def __init__(self, token_id: object) -> None:
        self._token_id = token_id

    def encode(self, text: str) -> list[int]:
        del text
        return cast("list[int]", [self._token_id])


class _ScriptedModel(torch.nn.Module):
    def __init__(self, tokens: list[int]) -> None:
        super().__init__()
        self.config = QwenTextConfig(
            architecture="anima_qwen3_06b",
            vocab_size=32,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            max_position_embeddings=32,
            rms_norm_eps=1e-6,
            rope_theta=10_000.0,
            qkv_bias=False,
            qk_norm=True,
            prompt_template="{}",
            min_tokens=1,
            pad_token_id=0,
        )
        self.embed_tokens = torch.nn.Embedding(32, 4)
        self.prefetch_block = _PrefetchBlock()
        self.layers = torch.nn.ModuleList((self.prefetch_block,))
        self._tokens = tokens
        self.calls: list[tuple[int, tuple[int, ...]]] = []
        self.prefetch_values: list[bool] = []

    def forward_causal(
        self,
        ids: torch.Tensor,
        cache_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        *,
        cache_position: int,
        frequencies: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        prefetch: bool,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        assert frequencies[0].shape[2] == ids.shape[1]
        self.prefetch_values.append(prefetch)
        call_ids = tuple(int(value) for value in ids[0].tolist())
        self.calls.append((cache_position, call_ids))
        length = ids.shape[1]
        key = ids.to(torch.float32).reshape(1, 1, length, 1).expand(-1, -1, -1, 4)
        value = key + 0.5
        end = cache_position + length
        cache_key_values[0][0][:, :, cache_position:end].copy_(key)
        cache_key_values[0][1][:, :, cache_position:end].copy_(value)
        hidden = torch.zeros((1, length, 4), dtype=torch.float32)
        return hidden, ((key, value),)

    def causal_frequencies(
        self, length: int, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty((1, 1, length, 4), device=device),
            torch.empty((1, 1, length, 2), device=device),
            torch.empty((1, 1, length, 2), device=device),
        )

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.full((*hidden.shape[:2], 32), -100.0)
        logits[..., self._tokens[len(self.calls) - 1]] = 100.0
        return logits


def _provider(
    tokens: list[int],
    tokenizer: _Tokenizer | None = None,
) -> tuple[QwenGenerationProvider, _ScriptedModel]:
    model = _ScriptedModel(tokens)
    with patch.object(qwen_generation_module, "ANIMA_QWEN3_06B_CONFIG", model.config):
        provider = QwenGenerationProvider(
            cast("Any", model),
            _Tokenizer() if tokenizer is None else tokenizer,
            MODEL_ID,
            eos_token_ids=(9,),
            block_tokens=4,
            max_device_blocks=8,
        )
    return provider, model


def _request(
    provider: QwenGenerationProvider,
    *,
    prompt: str = "p",
    maximum: int = 3,
    stop_texts: tuple[str, ...] = (),
    stop_tokens: tuple[int, ...] = (),
    open_session: bool = False,
    session: Any = None,
) -> GenerationRequest:
    return GenerationRequest(
        provider.id,
        MODEL_ID,
        prompt=prompt,
        stop=GenerationStopConditions(maximum, stop_texts, stop_tokens),
        open_session=open_session,
        session=session,
    )


def test_stream_is_pull_driven_and_partial_close_rolls_back() -> None:
    provider, model = _provider([4, 5, 6])
    stream = provider.generate(_request(provider), cancelled=lambda: False)

    assert model.calls == []
    assert next(stream) == GenerationTokenEvent(0, "A", 4)
    assert model.calls == [(0, (1, 2))]
    assert next(stream) == GenerationTokenEvent(1, "B", 5)
    assert model.calls == [(0, (1, 2)), (2, (4,))]
    stream.close()
    stream.close()
    assert provider.cache.session_states() == ()


def test_stateless_generation_skips_the_unused_final_cache_flush() -> None:
    provider, model = _provider([4, 5, 6])
    stream = provider.generate(_request(provider), cancelled=lambda: False)
    token_events = (next(stream), next(stream), next(stream))
    assert provider.cache.block_states() == ()
    terminal = cast(GenerationTerminalEvent, next(stream))
    with pytest.raises(StopIteration):
        next(stream)

    assert token_events == (
        GenerationTokenEvent(0, "A", 4),
        GenerationTokenEvent(1, "B", 5),
        GenerationTokenEvent(2, "C", 6),
    )
    assert terminal.result.text == "ABC"
    assert terminal.result.token_ids == (4, 5, 6)
    assert terminal.result.finish_reason is GenerationFinishReason.LENGTH
    assert terminal.result.continuation is None
    assert model.calls == [(0, (1, 2)), (2, (4,)), (3, (5,))]
    assert provider.cache.session_states() == ()
    assert cast("Any", stream)._cache_key is None
    assert cast("Any", stream)._cache_value is None
    assert cast("Any", stream)._cache_layers == ()


def test_generation_timers_stop_after_device_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _model = _provider([4, 5, 6])
    stream = provider.generate(
        _request(provider, maximum=2, open_session=True),
        cancelled=lambda: False,
    )
    stream_type = type(stream)
    original_run = stream_type._run_model  # pyright: ignore[reportPrivateUsage]
    original_timer = qwen_generation_module.time.perf_counter
    state = {"pending": False, "runs": 0, "synchronized": 0}

    def run_model(self: Any, token_ids: tuple[int, ...]) -> torch.Tensor:
        assert not state["pending"]
        logits = original_run(self, token_ids)
        state["pending"] = True
        state["runs"] += 1
        return logits

    def synchronize(_self: Any) -> None:
        if state["pending"]:
            state["pending"] = False
            state["synchronized"] += 1

    def perf_counter() -> float:
        assert not state["pending"]
        return original_timer()

    monkeypatch.setattr(stream_type, "_run_model", run_model)
    monkeypatch.setattr(stream_type, "_synchronize", synchronize)
    monkeypatch.setattr(qwen_generation_module.time, "perf_counter", perf_counter)

    events = tuple(stream)
    terminal = cast(GenerationTerminalEvent, events[-1])
    assert terminal.result.stats.prefill_time_s is not None
    assert terminal.result.stats.decode_time_s is not None
    assert state == {"pending": False, "runs": 3, "synchronized": 3}
    assert terminal.result.continuation is not None
    provider.close_session(terminal.result.continuation)


def test_session_commit_continuation_fork_and_close_preserve_exact_kv_lengths() -> None:
    provider, model = _provider([4, 5, 6, 6, 4, 4])
    first = tuple(
        provider.generate(
            _request(provider, maximum=2, open_session=True),
            cancelled=lambda: False,
        )
    )
    handle = cast(GenerationTerminalEvent, first[-1]).result.continuation
    assert handle is not None
    assert provider.cache.session_states()[0].token_count == 4
    assert model.calls[-1] == (3, (5,))

    fork = provider.fork_session(handle, token_count=3)
    assert sorted(state.token_count for state in provider.cache.session_states()) == [3, 4]
    continued = tuple(
        provider.generate(
            _request(provider, prompt="q", maximum=1, session=fork),
            cancelled=lambda: False,
        )
    )
    terminal = cast(GenerationTerminalEvent, continued[-1])
    assert terminal.result.continuation == fork
    assert model.calls[-2:] == [(3, (3,)), (4, (6,))]
    assert sorted(state.token_count for state in provider.cache.session_states()) == [4, 5]

    provider.close_session(handle)
    provider.close_session(handle)
    provider.close_session(fork)
    assert provider.cache.session_states() == ()
    with pytest.raises(GenerationSessionClosedError):
        provider.generate(_request(provider, session=handle), cancelled=lambda: False)


def test_active_session_rejects_reuse_close_and_fork() -> None:
    provider, _ = _provider([4, 5, 6])
    opened = tuple(
        provider.generate(
            _request(provider, maximum=1, open_session=True),
            cancelled=lambda: False,
        )
    )
    handle = cast(GenerationTerminalEvent, opened[-1]).result.continuation
    assert handle is not None
    stream = provider.generate(_request(provider, session=handle), cancelled=lambda: False)
    with pytest.raises(GenerationSessionBusyError):
        provider.generate(_request(provider, session=handle), cancelled=lambda: False)
    with pytest.raises(GenerationSessionBusyError):
        provider.close_session(handle)
    with pytest.raises(GenerationSessionBusyError):
        provider.fork_session(handle)
    stream.close()


def test_cancellation_rolls_back_existing_and_discards_provisional_session() -> None:
    provider, _ = _provider([4, 5, 6, 4, 5])
    opened = tuple(
        provider.generate(
            _request(provider, maximum=1, open_session=True),
            cancelled=lambda: False,
        )
    )
    handle = cast(GenerationTerminalEvent, opened[-1]).result.continuation
    assert handle is not None
    committed = provider.cache.session_states()[0].token_count
    cancel = False
    with provider.generate(
        _request(provider, prompt="q", maximum=2, session=handle),
        cancelled=lambda: cancel,
    ) as stream:
        assert isinstance(next(stream), GenerationTokenEvent)
        cancel = True
        terminal = cast(GenerationTerminalEvent, next(stream))
    assert terminal.result.finish_reason is GenerationFinishReason.CANCELLED
    assert terminal.result.continuation == handle
    assert provider.cache.session_states()[0].token_count == committed

    events = tuple(
        provider.generate(
            _request(provider, open_session=True),
            cancelled=lambda: True,
        )
    )
    terminal = cast(GenerationTerminalEvent, events[-1])
    assert terminal.result.finish_reason is GenerationFinishReason.CANCELLED
    assert terminal.result.continuation is None
    assert len(provider.cache.session_states()) == 1


def test_stream_buffers_utf8_and_stop_text_without_retracting_deltas() -> None:
    provider, _ = _provider([7, 8])
    events = tuple(
        provider.generate(
            _request(provider, maximum=2),
            cancelled=lambda: False,
        )
    )
    assert events[:2] == (
        GenerationTokenEvent(0, "", 7),
        GenerationTokenEvent(1, "\U0001f600", 8),
    )
    assert cast(GenerationTerminalEvent, events[-1]).result.text == "\U0001f600"

    provider, _ = _provider([4, 10, 11, 12, 13, 5])
    events = tuple(
        provider.generate(
            _request(provider, maximum=6, stop_texts=("STOP",)),
            cancelled=lambda: False,
        )
    )
    tokens = tuple(event for event in events if isinstance(event, GenerationTokenEvent))
    assert tuple(event.text for event in tokens) == ("A", "", "", "", "")
    terminal = cast(GenerationTerminalEvent, events[-1])
    assert terminal.result.finish_reason is GenerationFinishReason.STOP
    assert terminal.result.text == "A"


def test_eos_and_explicit_stop_tokens_have_distinct_finish_reasons() -> None:
    provider, _ = _provider([9])
    terminal = cast(
        GenerationTerminalEvent,
        tuple(provider.generate(_request(provider), cancelled=lambda: False))[-1],
    )
    assert terminal.result.finish_reason is GenerationFinishReason.EOS

    provider, _ = _provider([6])
    terminal = cast(
        GenerationTerminalEvent,
        tuple(
            provider.generate(
                _request(provider, stop_tokens=(6,)),
                cancelled=lambda: False,
            )
        )[-1],
    )
    assert terminal.result.finish_reason is GenerationFinishReason.STOP


def test_sampler_chain_preserves_declared_order_and_full_vocabulary_indices() -> None:
    chain = GenerationSamplerChain(
        (
            GenerationSamplerStage(GenerationSamplerKind.TOP_K, 2),
            GenerationSamplerStage(GenerationSamplerKind.PRESENCE_PENALTY, 10.0),
            GenerationSamplerStage(GenerationSamplerKind.GREEDY),
        )
    )
    token_id, token = _sample_token(torch.tensor([3.0, 2.0, 1.0, 0.0]), chain, (0,), None)
    assert token_id == 1
    assert token.tolist() == [1]

    stochastic = GenerationSamplerChain(
        (
            GenerationSamplerStage(GenerationSamplerKind.REPETITION_PENALTY, 1.1),
            GenerationSamplerStage(GenerationSamplerKind.PRESENCE_PENALTY, 0.2),
            GenerationSamplerStage(GenerationSamplerKind.FREQUENCY_PENALTY, 0.1),
            GenerationSamplerStage(GenerationSamplerKind.TEMPERATURE, 0.7),
            GenerationSamplerStage(GenerationSamplerKind.TOP_K, 4),
            GenerationSamplerStage(GenerationSamplerKind.TOP_P, 0.9),
            GenerationSamplerStage(GenerationSamplerKind.MIN_P, 0.01),
            GenerationSamplerStage(GenerationSamplerKind.TYPICAL_P, 0.95),
            GenerationSamplerStage(GenerationSamplerKind.MULTINOMIAL),
        )
    )
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4])
    first = _sample_token(logits, stochastic, (1, 1, 2), torch.Generator().manual_seed(7))
    second = _sample_token(logits, stochastic, (1, 1, 2), torch.Generator().manual_seed(7))
    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])


def test_provider_validates_model_binding_prompt_and_context() -> None:
    provider, _ = _provider([4])
    assert provider.capabilities.sessions
    assert provider.capabilities.ordered_sampler_chain
    with pytest.raises(ValueError, match="at least one prompt token"):
        provider.generate(_request(provider, prompt=""), cancelled=lambda: False)
    with pytest.raises(ValueError, match="different provider"):
        provider.generate(
            GenerationRequest("other.provider", MODEL_ID, prompt="p"),
            cancelled=lambda: False,
        )
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        provider.generate(
            _request(provider, stop_tokens=(32,)),
            cancelled=lambda: False,
        )


def test_provider_requires_the_complete_anima_profile() -> None:
    model = _ScriptedModel([4])
    with pytest.raises(ValueError, match="exact Anima Qwen3-0.6B or Qwen3-30B-A3B profile"):
        QwenGenerationProvider(cast("Any", model), _Tokenizer(), MODEL_ID)

    constructor = cast("Any", QwenGenerationProvider)
    with pytest.raises(TypeError, match="unexpected keyword argument '_testing_config'"):
        constructor(
            model,
            _Tokenizer(),
            MODEL_ID,
            _testing_config=model.config,
        )


def test_provider_reevaluates_prefetch_after_residency_transitions() -> None:
    provider, model = _provider([4, 4, 4])

    tuple(provider.generate(_request(provider, maximum=1), cancelled=lambda: False))
    model.prefetch_block.loaded = False
    tuple(provider.generate(_request(provider, maximum=1), cancelled=lambda: False))
    model.prefetch_block.loaded = True
    tuple(provider.generate(_request(provider, maximum=1), cancelled=lambda: False))

    assert model.prefetch_values == [False, True, False]


@pytest.mark.parametrize("token_id", (True, 1.75))
def test_provider_rejects_non_integer_prompt_token_ids(token_id: object) -> None:
    provider, model = _provider([4], _InvalidTokenizer(token_id))
    with pytest.raises(ValueError, match="IDs must be integers within the model vocabulary"):
        provider.generate(_request(provider), cancelled=lambda: False)
    assert model.calls == []
