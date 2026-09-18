"""OpenAI-compatible execution for the universal generation nodes."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

import dinkster_nodes_generation_openai.nodes as openai_nodes
import pytest
from dinkster_inference import (
    GenerationFinishReason,
    GenerationRequest,
    GenerationResult,
    GenerationSamplerKind,
    GenerationStats,
    GenerationTerminalEvent,
    OpenAICompatibility,
    OpenAIGenerationProvider,
    SamplingCancelled,
)
from dinkster_workers.execution import ExecutionContext, use_execution_context


def _inputs(**overrides: object) -> dict[str, object]:
    inputs: dict[str, object] = {
        "prompt": "A lighthouse in rain",
        "max_length": 32,
        "thinking": False,
        "use_default_template": False,
        "sampling_mode": "off",
    }
    inputs.update(overrides)
    return inputs


class _Stream:
    def __init__(self, text: str, reason: GenerationFinishReason) -> None:
        self._event = GenerationTerminalEvent(GenerationResult(text, reason, GenerationStats(0.01)))
        self.closed = False

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def __iter__(self) -> Iterator[GenerationTerminalEvent]:
        yield self._event


class _Provider:
    id = "test.openai"
    model_identity = "openai:test"

    def __init__(
        self,
        text: str = "generated",
        reason: GenerationFinishReason = GenerationFinishReason.EOS,
    ) -> None:
        self.text = text
        self.reason = reason
        self.requests: list[GenerationRequest] = []
        self.cancelled: list[Callable[[], bool]] = []
        self.streams: list[_Stream] = []

    def generate(
        self,
        request: GenerationRequest,
        *,
        cancelled: Callable[[], bool],
    ) -> _Stream:
        self.requests.append(request)
        self.cancelled.append(cancelled)
        stream = _Stream(self.text, self.reason)
        self.streams.append(stream)
        return stream


def test_openai_text_node_forwards_greedy_request_and_owns_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    monkeypatch.setattr(openai_nodes, "_provider", lambda: provider)

    result = openai_nodes.OpenAITextGenerate.execute(**_inputs())

    assert result == {"generated_text": "generated"}
    request = provider.requests[0]
    assert request.prompt == "A lighthouse in rain"
    assert request.stop.max_new_tokens == 32
    assert request.seed is None
    assert tuple(stage.kind for stage in request.sampler.stages) == (GenerationSamplerKind.GREEDY,)
    assert provider.streams[0].closed


@pytest.mark.parametrize(
    "hook",
    (
        openai_nodes.OpenAITextGenerate.check_lazy_status,
        openai_nodes.OpenAIPromptEnhance.check_lazy_status,
    ),
)
def test_openai_nodes_never_demand_the_native_clip(
    hook: Callable[..., tuple[str, ...]],
) -> None:
    assert hook(clip=None, provider=None) == ()
    assert hook(clip=None, provider="dinkster-nodes-generation-openai") == ()
    assert hook(clip=object(), provider=None) == ()


def test_openai_text_node_preserves_ordered_sampling_and_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider(reason=GenerationFinishReason.CANCELLED)
    monkeypatch.setattr(openai_nodes, "_provider", lambda: provider)

    def cancelled() -> bool:
        return True

    with use_execution_context(ExecutionContext("external", None, cancelled=cancelled)):
        with pytest.raises(SamplingCancelled, match="text generation cancelled"):
            openai_nodes.OpenAITextGenerate.execute(
                **_inputs(
                    sampling_mode="on",
                    **{
                        "sampling_mode.temperature": 0.7,
                        "sampling_mode.top_k": 64,
                        "sampling_mode.top_p": 0.95,
                        "sampling_mode.min_p": 0.05,
                        "sampling_mode.repetition_penalty": 1.05,
                        "sampling_mode.presence_penalty": 0.25,
                        "sampling_mode.seed": 42,
                    },
                )
            )

    request = provider.requests[0]
    assert tuple(stage.kind for stage in request.sampler.stages) == (
        GenerationSamplerKind.REPETITION_PENALTY,
        GenerationSamplerKind.PRESENCE_PENALTY,
        GenerationSamplerKind.TEMPERATURE,
        GenerationSamplerKind.TOP_K,
        GenerationSamplerKind.MIN_P,
        GenerationSamplerKind.TOP_P,
        GenerationSamplerKind.MULTINOMIAL,
    )
    assert request.seed == 42
    assert provider.cancelled[0] is cancelled
    assert provider.streams[0].closed


def test_standard_openai_node_refuses_nonportable_sampler_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAIGenerationProvider(
        "http://127.0.0.1:1/v1",
        "test-model",
        compatibility=OpenAICompatibility.STANDARD,
        timeout_s=1.0,
    )
    monkeypatch.setattr(openai_nodes, "_provider", lambda: provider)
    try:
        with pytest.raises(ValueError, match="complete-history semantics.*repetition_penalty"):
            openai_nodes.OpenAITextGenerate.execute(
                **_inputs(
                    sampling_mode="on",
                    **{
                        "sampling_mode.temperature": 0.7,
                        "sampling_mode.top_k": 64,
                        "sampling_mode.top_p": 0.95,
                        "sampling_mode.min_p": 0.05,
                        "sampling_mode.repetition_penalty": 1.05,
                        "sampling_mode.presence_penalty": 0.0,
                        "sampling_mode.seed": 0,
                    },
                )
            )
    finally:
        provider.close()


def test_openai_prompt_enhance_matches_native_format_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def generate(_inputs: Mapping[str, object], prompt: str) -> str:
        captured.append(prompt)
        return "<think>private</think><|channel>final\n **Assistant:** Enhanced\n\nprompt. "

    monkeypatch.setattr(openai_nodes, "_generate", generate)

    result = openai_nodes.OpenAIPromptEnhance.execute(**_inputs())

    assert result == {"generated_text": "Enhanced prompt."}
    assert captured[0].startswith("<|im_start|>system\nYou are a Creative Assistant.")
    assert '#### Example\nInput: "A woman at a coffee shop talking on the phone"' in captured[0]
    assert "<|im_start|>user\nUser Raw Input Prompt: A lighthouse in rain.<|im_end|>" in captured[0]
    assert captured[0].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.mark.parametrize("generated", ("<think>private reasoning", ""))
def test_openai_prompt_enhance_preserves_prompt_when_cleanup_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    generated: str,
) -> None:
    monkeypatch.setattr(openai_nodes, "_generate", lambda *_args: generated)

    assert openai_nodes.OpenAIPromptEnhance.execute(**_inputs()) == {
        "generated_text": "A lighthouse in rain"
    }


def test_openai_worker_reuses_and_closes_its_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    openai_nodes._close_provider()  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setenv("DINKSTER_OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("DINKSTER_OPENAI_MODEL", "test-model")
    monkeypatch.setenv("DINKSTER_OPENAI_STREAM", "false")
    monkeypatch.setenv("DINKSTER_OPENAI_TIMEOUT", "1")
    first = openai_nodes._provider()  # pyright: ignore[reportPrivateUsage]
    try:
        assert openai_nodes._provider() is first  # pyright: ignore[reportPrivateUsage]
        openai_nodes._close_provider()  # pyright: ignore[reportPrivateUsage]
        second = openai_nodes._provider()  # pyright: ignore[reportPrivateUsage]
        assert second is not first
    finally:
        openai_nodes._close_provider()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("value", "error"),
    (
        (False, TypeError),
        ("", ValueError),
        ("bad\npath", ValueError),
    ),
)
def test_openai_proxy_socket_validation(value: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="proxy socket"):
        OpenAIGenerationProvider(
            "https://api.example.test/v1",
            "test-model",
            proxy_socket=value,  # type: ignore[arg-type]
        )
