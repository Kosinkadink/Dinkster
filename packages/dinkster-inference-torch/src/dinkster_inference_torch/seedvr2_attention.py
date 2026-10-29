from collections.abc import Sequence

import torch
from torch import Tensor

from .attention import AttentionKernel


def _var_attention_qkv(
    q: Tensor, k: Tensor, v: Tensor, heads: int, skip_reshape: bool
) -> tuple[Tensor, Tensor, Tensor, int]:
    if skip_reshape:
        return q, k, v, q.shape[-1]
    total_tokens, embed_dim = q.shape
    head_dim = embed_dim // heads
    return (
        q.view(total_tokens, heads, head_dim),
        k.view(k.shape[0], heads, head_dim),
        v.view(v.shape[0], heads, head_dim),
        head_dim,
    )


def _var_attention_output(
    out: Tensor, heads: int, head_dim: int, skip_output_reshape: bool
) -> Tensor:
    if skip_output_reshape:
        return out
    return out.reshape(-1, heads * head_dim)


def var_attention_optimized_split(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    heads: int,
    cu_seqlens_q: Tensor | Sequence[int],
    cu_seqlens_k: Tensor | Sequence[int],
    *args: object,
    attention_kernel: AttentionKernel,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    **kwargs: object,
) -> Tensor:
    q, k, v, head_dim = _var_attention_qkv(q, k, v, heads, skip_reshape)

    q_split_indices = cu_seqlens_q[1:-1]
    k_split_indices = cu_seqlens_k[1:-1]
    if k.shape[0] != v.shape[0]:
        raise ValueError("cu_seqlens_k does not match v token count")

    q_splits = torch.tensor_split(q, q_split_indices, dim=0)
    k_splits = torch.tensor_split(k, k_split_indices, dim=0)
    v_splits = torch.tensor_split(v, k_split_indices, dim=0)
    if len(q_splits) != len(k_splits) or len(q_splits) != len(v_splits):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same sequence count")

    outputs: list[Tensor] = []
    for q_i, k_i, v_i in zip(q_splits, k_splits, v_splits, strict=True):
        q_i = q_i.permute(1, 0, 2).unsqueeze(0)
        k_i = k_i.permute(1, 0, 2).unsqueeze(0)
        v_i = v_i.permute(1, 0, 2).unsqueeze(0)
        out_i = attention_kernel(q_i, k_i, v_i)
        outputs.append(out_i.squeeze(0).permute(1, 0, 2))

    out = torch.cat(outputs, dim=0)
    return _var_attention_output(out, heads, head_dim, skip_output_reshape)


optimized_var_attention = var_attention_optimized_split
