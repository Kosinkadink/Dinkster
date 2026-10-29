"""Classifier-free guidance execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from dinkster_inference import cfg_needs_uncond
from dinkster_inference.sampling import TensorT

CondT = TypeVar("CondT")
CondT_contra = TypeVar("CondT_contra", contravariant=True)


def cfg_combine(
    cond_denoised: TensorT,
    uncond_denoised: TensorT,
    cond_scale: float,
) -> TensorT:
    """Combine conditional and unconditional denoised predictions."""
    return uncond_denoised + (cond_denoised - uncond_denoised) * cond_scale


class ConditionedEvaluator(Protocol[TensorT, CondT_contra]):
    """One denoised model evaluation at exactly one conditioning."""

    def __call__(self, x: TensorT, sigma: float, conditioning: CondT_contra) -> TensorT: ...


@dataclass(frozen=True)
class CfgDenoiser(Generic[TensorT, CondT]):
    """Classifier-free guidance over a single-conditioning evaluator."""

    evaluate: ConditionedEvaluator[TensorT, CondT]
    cond: CondT
    uncond: CondT | None
    cfg_scale: float

    def __call__(self, x: TensorT, sigma: float) -> TensorT:
        uncond = self.uncond
        if uncond is None or not cfg_needs_uncond(self.cfg_scale):
            return self.evaluate(x, sigma, self.cond)
        cond_denoised = self.evaluate(x, sigma, self.cond)
        uncond_denoised = self.evaluate(x, sigma, uncond)
        return cfg_combine(cond_denoised, uncond_denoised, self.cfg_scale)


__all__ = ["CfgDenoiser", "ConditionedEvaluator", "cfg_combine"]
