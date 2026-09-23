# pyright: reportMissingImports=false
"""Capture reference image keys/values and inject them within one invocation."""

from dinkster_api.v1 import (
    AttentionContribution,
    AttentionSelector,
    AttentionWrapperDescriptor,
    InferenceContribution,
)

NODES = ()


def make(
    *,
    strength=0.5,
    reference_state=None,
    torch_version="2.13.0+cu130",
    aimdo_version="0.5.5.post2",
):
    if not 0 <= strength <= 1:
        raise ValueError("reference strength must be between zero and one")
    frozen_reference = None
    reference_digest = "capture-first-call"
    if reference_state is not None:
        import hashlib

        import torch

        if type(reference_state) is not dict or not reference_state:
            raise TypeError("reference_state must be a nonempty exact dict")
        frozen_reference = {}
        digest = hashlib.sha256()
        for key in sorted(reference_state):
            values = reference_state[key]
            if (
                type(key) is not tuple
                or len(key) != 3
                or any(type(part) is not str or not part for part in key)
                or type(values) is not tuple
                or len(values) != 2
                or any(type(value) is not torch.Tensor for value in values)
            ):
                raise TypeError("reference_state entries must map point keys to K/V tensors")
            frozen = tuple(value.detach().clone() for value in values)
            frozen_reference[key] = frozen
            digest.update("\0".join(key).encode())
            for value in frozen:
                digest.update(str(tuple(value.shape)).encode())
                digest.update(str(value.dtype).encode())
                digest.update(value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        reference_digest = digest.hexdigest()

    def reference(q, k, v, context, next):
        import torch
        import torch.nn.functional as F

        baseline = next(q, k, v)
        if strength == 0:
            return baseline
        key = (context.family, context.block, context.kind)
        bank = frozen_reference
        if bank is None:
            bank = context.state.setdefault("reference", {})
        if key not in bank and frozen_reference is None:
            bank[key] = (k.detach().clone(), v.detach().clone())
            return baseline
        if key not in bank:
            raise ValueError(f"reference state has no capture for {key!r}")
        ref_k, ref_v = bank[key]
        ref_k = ref_k.to(device=k.device, dtype=k.dtype)
        ref_v = ref_v.to(device=v.device, dtype=v.dtype)
        if ref_k.shape != k.shape or ref_v.shape != v.shape:
            raise ValueError("reference and generation attention geometry must match")
        image_ranges = {
            (span.start, span.end)
            for span in context.spans
            if span.axis == "key" and span.stream == "image"
        }
        if len(image_ranges) != 1:
            raise ValueError("reference attention requires one shared image-token range")
        start, end = image_ranges.pop()
        augmented = F.scaled_dot_product_attention(
            q,
            torch.cat((k, ref_k[:, :, start:end]), dim=2),
            torch.cat((v, ref_v[:, :, start:end]), dim=2),
        )
        result = baseline.clone()
        result[:, :, start:end] = baseline[:, :, start:end].lerp(
            augmented[:, :, start:end], strength
        )
        return result

    metadata = (
        ("config.reference-digest", reference_digest),
        ("config.strength", str(strength)),
    )
    return InferenceContribution(
        attention=AttentionContribution(
            torch_version=torch_version,
            aimdo_version=aimdo_version,
            wrappers=(
                AttentionWrapperDescriptor(
                    "proof_reference.unet",
                    AttentionSelector("unet", "output_blocks.11.1.transformer_blocks.0", "self"),
                    reference,
                    behavior_metadata=metadata,
                ),
                AttentionWrapperDescriptor(
                    "proof_reference.flux",
                    AttentionSelector("flux", "double_blocks.18", "joint"),
                    reference,
                    behavior_metadata=metadata,
                ),
            ),
        )
    )


def register():
    return make()
