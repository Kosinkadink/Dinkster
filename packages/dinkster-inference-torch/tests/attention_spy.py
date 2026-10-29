from __future__ import annotations

from typing import Any

import torch
from dinkster_inference_torch import AttentionKernel


class CallableModuleKernel(torch.nn.Module):
    def __init__(self, delegate: AttentionKernel) -> None:
        super().__init__()
        self.delegate = delegate
        self.marker = torch.nn.Parameter(torch.ones(()))
        self.register_buffer("buffer_marker", torch.ones(()))
        self.calls: list[dict[str, Any]] = []

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "q_shape": tuple(q.shape),
                "k_shape": tuple(k.shape),
                "v_shape": tuple(v.shape),
                "q_stride": q.stride(),
                "k_stride": k.stride(),
                "v_stride": v.stride(),
                "q_contiguous": q.is_contiguous(),
                "k_contiguous": k.is_contiguous(),
                "v_contiguous": v.is_contiguous(),
                "mask": mask,
                "causal": causal,
                "scale": scale,
                "enable_gqa": enable_gqa,
            }
        )
        return self.delegate(
            q,
            k,
            v,
            mask=mask,
            causal=causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )


def assert_kernel_is_not_model_state(model: torch.nn.Module, kernel: CallableModuleKernel) -> None:
    assert all(module is not kernel for module in dict(model.named_modules()).values())
    assert all(
        parameter is not kernel.marker for parameter in dict(model.named_parameters()).values()
    )
    assert all(
        buffer is not kernel.buffer_marker for buffer in dict(model.named_buffers()).values()
    )
    assert all("_attention_kernel" not in key for key in model.state_dict())
