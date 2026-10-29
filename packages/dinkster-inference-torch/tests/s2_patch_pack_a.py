"""Out-of-tree S2 proof pack A: a multiplicative weight adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from dinkster_inference import AdapterPatch, ProviderPatchRef
from dinkster_inference_torch import (
    PatchProviderContribution,
    PatchProviderDescriptor,
)

PROVIDER_ID = "proof_a.scale"


@dataclass(frozen=True)
class ScaleAdapter:
    factor: torch.Tensor

    def target_shape(self, base: tuple[int, ...]) -> tuple[int, ...]:
        return base

    def calculate(
        self,
        weight: torch.Tensor,
        *,
        strength: float,
        function: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        delta = weight * (self.factor.to(weight) - 1.0) * strength
        weight += delta if function is None else function(delta)
        return weight

    def payload_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.factor,)

    def rebuild_payloads(self, replacements: Sequence[torch.Tensor]) -> ScaleAdapter:
        if len(replacements) != 1:
            raise ValueError("scale adapter needs exactly one replacement")
        return ScaleAdapter(replacements[0])


def decode(key: str) -> ProviderPatchRef:
    return ProviderPatchRef(PROVIDER_ID, (("key", key),))


def _materialize(
    decoded: ProviderPatchRef,
    tensors: Mapping[str, torch.Tensor],
    _intermediate_dtype: torch.dtype,
):
    return AdapterPatch(ScaleAdapter(tensors[str(dict(decoded.parameters)["key"])]))


PROVIDER = PatchProviderDescriptor(
    id=PROVIDER_ID,
    aliases=("proof_scale",),
    materialize=_materialize,
    behavior_metadata=(("operation", "multiply"), ("version", 1)),
)


def register() -> PatchProviderContribution:
    return PatchProviderContribution((PROVIDER,))
