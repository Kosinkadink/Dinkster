"""Text-encoder contracts: tokenize, weight, encode - split boundaries.

ComfyUI's text-encoder stack braids four concerns into one class
hierarchy: prompt-weighting syntax parsing, tokenization, textual-
inversion embedding loading, and the model forward pass
(comfy/sd1_clip.py SDTokenizer/SDClipModel @ b78cec87). Here each is
its own boundary:

- A Tokenizer turns prompt text into WeightedSpans - token ids plus the
  emphasis weight parsed from ``(text:1.2)`` syntax. Embedding
  references stay symbolic (EmbeddingRef) so tokenization needs no
  weights on disk.
- Conditioning carries sequence embeddings and an optional pooled vector.

How weights are *applied* to embeddings (the comfy/A1111 normalization
split) and chunk composition for long prompts are encoder
implementation policy - stage 5 builds them; the contract pins only the
data crossing the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from .patches import SizedTensor

T = TypeVar("T", bound=SizedTensor)


@dataclass(frozen=True)
class EmbeddingRef:
    """A textual-inversion embedding by name, resolved at encode time -
    tokenization never touches embedding files."""

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("embedding name must not be empty")


TokenUnit = int | EmbeddingRef


@dataclass(frozen=True)
class WeightedSpan:
    """A run of tokens sharing one emphasis weight.

    The typed form of ComfyUI's ``[(token, weight), ...]`` pair lists
    (comfy/sd1_clip.py tokenize_with_weights @ b78cec87), grouped by
    weight instead of repeating it per token. Negative weights are
    legal - ``(text:-1)`` parses and interpolates upstream.
    """

    tokens: tuple[TokenUnit, ...]
    weight: float = 1.0


class Tokenizer(Protocol):
    """Prompt text -> weighted spans. Pure with respect to its inputs;
    special tokens (BOS/EOS/pad) and chunking are the encoder's job."""

    def tokenize(self, text: str) -> tuple[WeightedSpan, ...]: ...


@dataclass(frozen=True)
class Conditioning(Generic[T]):
    """The common text-encoder result: sequence embeddings and, for
    families that use it, a pooled vector."""

    embeddings: T
    pooled: T | None = None


__all__ = [
    "Conditioning",
    "EmbeddingRef",
    "TokenUnit",
    "Tokenizer",
    "WeightedSpan",
]
