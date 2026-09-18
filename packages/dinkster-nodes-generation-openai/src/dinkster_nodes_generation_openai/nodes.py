"""OpenAI-compatible bodies for universal generation nodes."""

from __future__ import annotations

import atexit
import math
import os
import threading
from collections.abc import Mapping

from dinkster_inference import (
    GenerationFinishReason,
    GenerationRequest,
    GenerationSamplerChain,
    GenerationSamplerKind,
    GenerationSamplerStage,
    GenerationStopConditions,
    GenerationTerminalEvent,
    OpenAICompatibility,
    OpenAIGenerationProvider,
    SamplingCancelled,
)
from dinkster_nodes_generation import clean_enhanced_prompt, prepare_ltx2_prompt
from dinkster_nodes_generation.nodes import PromptEnhance, TextGenerate
from dinkster_workers import current_execution_context

_PROVIDER_ID = "dinkster.openai"
_FALSE = frozenset(("0", "false", "no", "off"))
_TRUE = frozenset(("1", "true", "yes", "on"))

_provider_lock = threading.Lock()
_worker_provider: OpenAIGenerationProvider | None = None


def _environment_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise ValueError(f"{name} must be true or false")


def _provider() -> OpenAIGenerationProvider:
    global _worker_provider
    with _provider_lock:
        if _worker_provider is None:
            timeout_raw = os.environ.get("DINKSTER_OPENAI_TIMEOUT", "300")
            try:
                timeout = float(timeout_raw)
            except ValueError as exc:
                raise ValueError("DINKSTER_OPENAI_TIMEOUT must be a number") from exc
            api_key = os.environ.get("DINKSTER_OPENAI_API_KEY") or None
            proxy_socket = os.environ.get("DINKSTER_EGRESS_PROXY") or None
            _worker_provider = OpenAIGenerationProvider(
                os.environ.get("DINKSTER_OPENAI_BASE_URL", ""),
                os.environ.get("DINKSTER_OPENAI_MODEL", ""),
                api_key=api_key,
                provider_id=_PROVIDER_ID,
                compatibility=OpenAICompatibility(
                    os.environ.get("DINKSTER_OPENAI_COMPATIBILITY", "openai")
                ),
                stream=_environment_bool("DINKSTER_OPENAI_STREAM", True),
                timeout_s=timeout,
                proxy_socket=proxy_socket,
            )
        return _worker_provider


def _close_provider() -> None:
    global _worker_provider
    with _provider_lock:
        provider = _worker_provider
        _worker_provider = None
    if provider is not None:
        provider.close()


atexit.register(_close_provider)


def _input_int(inputs: Mapping[str, object], name: str, minimum: int, maximum: int) -> int:
    value = inputs.get(name)
    if type(value) is not int:
        raise TypeError(f"{name.rsplit('.', 1)[-1]} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name.rsplit('.', 1)[-1]} must be in [{minimum}, {maximum}]")
    return value


def _input_float(
    inputs: Mapping[str, object],
    name: str,
    minimum: float,
    maximum: float,
    *,
    inclusive_minimum: bool = True,
) -> float:
    value = inputs.get(name)
    if type(value) is not float:
        raise TypeError(f"{name.rsplit('.', 1)[-1]} must be a float")
    lower_valid = value >= minimum if inclusive_minimum else value > minimum
    if not math.isfinite(value) or not lower_valid or value > maximum:
        opening = "[" if inclusive_minimum else "("
        raise ValueError(f"{name.rsplit('.', 1)[-1]} must be in {opening}{minimum}, {maximum}]")
    return value


def _sampler(inputs: Mapping[str, object]) -> tuple[GenerationSamplerChain, int | None]:
    mode = inputs.get("sampling_mode")
    if mode == "off":
        return GenerationSamplerChain((GenerationSamplerStage(GenerationSamplerKind.GREEDY),)), None
    if mode != "on":
        raise ValueError(f"unknown text generation sampling mode: {mode!r}")
    temperature = _input_float(
        inputs, "sampling_mode.temperature", 0.0, 2.0, inclusive_minimum=False
    )
    top_k = _input_int(inputs, "sampling_mode.top_k", 0, 1_000)
    top_p = _input_float(inputs, "sampling_mode.top_p", 0.0, 1.0)
    min_p = _input_float(inputs, "sampling_mode.min_p", 0.0, 1.0)
    repetition = _input_float(
        inputs, "sampling_mode.repetition_penalty", 0.0, 5.0, inclusive_minimum=False
    )
    presence = _input_float(inputs, "sampling_mode.presence_penalty", 0.0, 5.0)
    seed = _input_int(inputs, "sampling_mode.seed", 0, 2**53 - 1)
    stages: list[GenerationSamplerStage] = []
    for kind, value, neutral in (
        (GenerationSamplerKind.REPETITION_PENALTY, repetition, 1.0),
        (GenerationSamplerKind.PRESENCE_PENALTY, presence, 0.0),
        (GenerationSamplerKind.TEMPERATURE, temperature, 1.0),
        (GenerationSamplerKind.TOP_K, top_k, 0),
        (GenerationSamplerKind.MIN_P, min_p, 0.0),
        (GenerationSamplerKind.TOP_P, top_p, 1.0),
    ):
        if value != neutral:
            stages.append(GenerationSamplerStage(kind, value))
    stages.append(GenerationSamplerStage(GenerationSamplerKind.MULTINOMIAL))
    return GenerationSamplerChain(tuple(stages)), seed


def _validate_options(inputs: Mapping[str, object]) -> None:
    for name in ("image", "video", "audio"):
        if inputs.get(name) is not None:
            raise ValueError(f"OpenAI-compatible generation does not support {name} input")
    for name in ("thinking", "use_default_template"):
        value = inputs.get(name, False)
        if type(value) is not bool:
            raise TypeError(f"{name} must be a boolean")
        if value:
            raise ValueError(f"OpenAI-compatible generation does not support {name}")


def _generate(inputs: Mapping[str, object], prompt: str) -> str:
    _validate_options(inputs)
    provider = _provider()
    sampler, seed = _sampler(inputs)
    context = current_execution_context()
    cancelled = (lambda: False) if context is None else context.cancelled
    request = GenerationRequest(
        provider.id,
        provider.model_identity,
        prompt=prompt,
        sampler=sampler,
        stop=GenerationStopConditions(_input_int(inputs, "max_length", 1, 32_768)),
        seed=seed,
    )
    terminal: GenerationTerminalEvent | None = None
    with provider.generate(request, cancelled=cancelled) as stream:
        for event in stream:
            if isinstance(event, GenerationTerminalEvent):
                if terminal is not None:
                    raise RuntimeError("generation stream produced multiple terminal events")
                terminal = event
            elif terminal is not None:
                raise RuntimeError("generation stream produced an event after its terminal event")
    if terminal is None:
        raise RuntimeError("generation stream ended without a terminal event")
    if terminal.result.finish_reason is GenerationFinishReason.CANCELLED:
        raise SamplingCancelled("text generation cancelled")
    return terminal.result.text


class OpenAITextGenerate(TextGenerate):
    @classmethod
    def check_lazy_status(cls, **_inputs: object) -> tuple[str, ...]:
        return ()

    @classmethod
    def execute(cls, **inputs: object) -> Mapping[str, object]:
        prompt = inputs.get("prompt")
        if type(prompt) is not str:
            raise TypeError("prompt must be a string")
        return cls.outputs(generated_text=_generate(inputs, prompt))


class OpenAIPromptEnhance(PromptEnhance):
    @classmethod
    def check_lazy_status(cls, **_inputs: object) -> tuple[str, ...]:
        return ()

    @classmethod
    def execute(cls, **inputs: object) -> Mapping[str, object]:
        prompt = inputs.get("prompt")
        if type(prompt) is not str:
            raise TypeError("prompt must be a string")
        text = _generate(inputs, prepare_ltx2_prompt(prompt))
        return cls.outputs(generated_text=clean_enhanced_prompt(text, prompt))


OPENAI_GENERATION_NODES = (OpenAITextGenerate, OpenAIPromptEnhance)

__all__ = ["OPENAI_GENERATION_NODES", "OpenAIPromptEnhance", "OpenAITextGenerate"]
