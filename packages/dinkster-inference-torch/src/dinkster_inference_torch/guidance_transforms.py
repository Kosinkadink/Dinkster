"""CFG-transform guidance contributions matching executed ComfyUI reference nodes.

Each factory returns a GuidanceContribution whose math reproduces the
corresponding comfy_extras node executed at the pinned parity commit
(b78cec879b9460d5cb25228a83a942fb78d2cd24); the executed oracle lives in
tests/goldens/cfg_transform_goldens.json. Predictions on the guidance seam are
denoised values, so transforms that the reference defines in noise space
convert through the request input exactly as the reference does.

When the plan carries no model-evaluated unconditional lane (cfg scale one),
the reference evaluates transforms against calc_cond_batch's untouched zeros,
so these transforms substitute zeros rather than forcing an unconditional
evaluation; TCFG and FreSca instead skip entirely, mirroring their
``None in conds[:2]`` guard.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math

import torch
from dinkster_inference import (
    AttentionGuidanceDescriptor,
    GuidanceContractError,
    GuidanceContribution,
    GuidanceEvaluationPlan,
    GuidancePlanContext,
    GuidancePostCFGContext,
    GuidancePostCFGDescriptor,
    GuidancePreCFGContext,
    GuidancePreCFGDescriptor,
    GuidancePrediction,
    GuidancePredictions,
    GuidancePredictionSource,
    GuidanceReduceContext,
    GuidanceRole,
    GuidanceScaleDescriptor,
    GuidanceStrategyDescriptor,
    cfg_needs_uncond,
)
from dinkster_inference.guidance import GuidancePhaseParticipation

from .cfg import cfg_combine
from .guidance import _standard_plan

APG_STATE_NAMESPACE = "dinkster.apg"
"""Extension-state namespace APG uses for its cross-step momentum state.

Whatever activates the APG contribution must declare this namespace in the
sampling execution context's extension state.
"""

__all__ = [
    "APG_STATE_NAMESPACE",
    "apg",
    "apply_latent_operation",
    "cfg_norm",
    "cfg_override",
    "cfg_zero_star",
    "epsilon_scaling",
    "fresca",
    "latent_operation",
    "mahiro",
    "nag",
    "perp_neg",
    "renorm_cfg",
    "rescale_cfg",
    "scheduled_cfg",
    "tcfg",
    "temporal_score_rescaling",
    "trellis2_rescale_cfg",
]

_Predictions = GuidancePredictions[torch.Tensor]


def _values(
    context: GuidancePreCFGContext[torch.Tensor]
    | GuidanceReduceContext[torch.Tensor]
    | GuidancePostCFGContext[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Primary prediction and the unconditional prediction or zeros."""
    plan = context.request.plan
    by_id = {item.lane_id: item.value for item in context.predictions.items}
    uncond = by_id.get(plan.unconditional_id) if plan.unconditional_id is not None else None
    if uncond is None:
        uncond = torch.zeros_like(context.request.input)
    return by_id[plan.primary_id], uncond


def _model_uncond(
    context: GuidancePreCFGContext[torch.Tensor],
) -> tuple[str, torch.Tensor] | None:
    """The unconditional lane id and prediction, only when model-evaluated."""
    lane_id = context.request.plan.unconditional_id
    if lane_id is None:
        return None
    item = next(item for item in context.predictions.items if item.lane_id == lane_id)
    return (lane_id, item.value) if item.source is GuidancePredictionSource.MODEL else None


def _replace(predictions: _Predictions, lane_id: str, value: torch.Tensor) -> _Predictions:
    return GuidancePredictions(
        tuple(
            GuidancePrediction(
                item.lane_id, value if item.lane_id == lane_id else item.value, item.source
            )
            for item in predictions.items
        )
    )


def _optimized_scale(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    positive_flat = positive.reshape(positive.shape[0], -1)
    negative_flat = negative.reshape(negative.shape[0], -1)
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    st_star = dot_product / squared_norm
    return st_star.reshape([positive.shape[0]] + [1] * (positive.ndim - 1))


def cfg_zero_star() -> GuidanceContribution[torch.Tensor]:
    """CFG-Zero* rescaled guidance (comfy_extras.nodes_cfg.CFGZeroStar)."""

    def transform(context: GuidancePostCFGContext[torch.Tensor]) -> torch.Tensor:
        x = context.request.input
        cond_p, uncond_p = _values(context)
        alpha = _optimized_scale(x - cond_p, x - uncond_p)
        return (
            context.reduced
            + uncond_p * (alpha - 1.0)
            + context.cfg_scale * uncond_p * (1.0 - alpha)
        )

    return GuidanceContribution(
        post_cfg=(GuidancePostCFGDescriptor("dinkster.cfg-zero-star", transform),)
    )


def cfg_norm(strength: float, pre_cfg: bool = False) -> GuidanceContribution[torch.Tensor]:
    """Norm-preserving guidance rescale (comfy_extras.nodes_cfg.CFGNorm)."""
    metadata = (
        ("config.pre_cfg", pre_cfg),
        ("config.strength", str(strength)),
    )
    if pre_cfg:

        def reduce(context: GuidanceReduceContext[torch.Tensor]) -> torch.Tensor:
            x = context.request.input
            cond_p, uncond_p = _values(context)
            cond = x - cond_p
            uncond = x - uncond_p
            comb = uncond + context.cfg_scale * (cond - uncond)
            cond_norm = torch.linalg.vector_norm(cond, dim=1, keepdim=True)
            comb_norm = torch.linalg.vector_norm(comb, dim=1, keepdim=True)
            rescale = torch.where(
                comb_norm > 0,
                cond_norm / comb_norm.clamp_min(1e-12),
                torch.ones_like(comb_norm),
            )
            rescaled = comb * rescale
            if strength != 1.0:
                rescaled = strength * rescaled + (1.0 - strength) * comb
            return x - rescaled

        return GuidanceContribution(
            strategy=GuidanceStrategyDescriptor(
                "dinkster.cfg-norm",
                _standard_plan,
                reduce,
                participation=GuidancePhaseParticipation.COMPOSE,
                behavior_metadata=metadata,
            )
        )

    def transform(context: GuidancePostCFGContext[torch.Tensor]) -> torch.Tensor:
        cond_p, _ = _values(context)
        norm_full_cond = torch.norm(cond_p, dim=1, keepdim=True)
        norm_pred_text = torch.norm(context.reduced, dim=1, keepdim=True)
        scale = (norm_full_cond / (norm_pred_text + 1e-8)).clamp(min=0.0, max=1.0)
        return context.reduced * scale * strength

    return GuidanceContribution(
        post_cfg=(
            GuidancePostCFGDescriptor("dinkster.cfg-norm", transform, behavior_metadata=metadata),
        )
    )


def _score_tangential_damping(cond_score: torch.Tensor, uncond_score: torch.Tensor) -> torch.Tensor:
    batch_num = cond_score.shape[0]
    cond_score_flat = cond_score.reshape(batch_num, 1, -1).float()
    uncond_score_flat = uncond_score.reshape(batch_num, 1, -1).float()
    score_matrix = torch.cat((uncond_score_flat, cond_score_flat), dim=1)
    try:
        _, _, vh = torch.linalg.svd(score_matrix, full_matrices=False)
    except RuntimeError:
        _, _, vh = torch.linalg.svd(score_matrix.cpu(), full_matrices=False)
    v1 = vh[:, 0:1, :].to(uncond_score_flat.device)
    uncond_score_td = (uncond_score_flat @ v1.transpose(-2, -1)) * v1
    return uncond_score_td.reshape_as(uncond_score).to(uncond_score.dtype)


def tcfg() -> GuidanceContribution[torch.Tensor]:
    """Tangential-damping CFG (comfy_extras.nodes_tcfg.TCFG)."""

    def transform(context: GuidancePreCFGContext[torch.Tensor]) -> _Predictions:
        uncond = _model_uncond(context)
        if uncond is None:
            return context.predictions
        uncond_id, uncond_p = uncond
        x = context.request.input
        cond_p, _ = _values(context)
        uncond_td = _score_tangential_damping(x - cond_p, x - uncond_p)
        return _replace(context.predictions, uncond_id, x - uncond_td)

    return GuidanceContribution(pre_cfg=(GuidancePreCFGDescriptor("dinkster.tcfg", transform),))


def _fourier_filter(
    x: torch.Tensor, scale_low: float, scale_high: float, freq_cutoff: int
) -> torch.Tensor:
    dtype, device = x.dtype, x.device
    x = x.to(torch.float32)
    x_freq = torch.fft.fftn(x, dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
    mask = torch.ones(x_freq.shape, device=device) * scale_high
    m = mask
    for d in range(2):
        dim = len(x_freq.shape) - 2 + d
        cc = x_freq.shape[dim] // 2
        f_c = min(freq_cutoff, cc)
        m = m.narrow(dim, cc - f_c, f_c * 2)
    m[:] = scale_low
    x_freq = x_freq * mask
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered.to(dtype)


def fresca(
    scale_low: float, scale_high: float, freq_cutoff: int
) -> GuidanceContribution[torch.Tensor]:
    """Frequency-dependent guidance scaling (comfy_extras.nodes_fresca.FreSca)."""
    metadata = (
        ("config.freq_cutoff", freq_cutoff),
        ("config.scale_high", str(scale_high)),
        ("config.scale_low", str(scale_low)),
    )

    def transform(context: GuidancePreCFGContext[torch.Tensor]) -> _Predictions:
        uncond = _model_uncond(context)
        if uncond is None:
            return context.predictions
        _, uncond_p = uncond
        plan = context.request.plan
        cond_p, _ = _values(context)
        guidance = cond_p - uncond_p
        filtered_guidance = _fourier_filter(guidance, scale_low, scale_high, freq_cutoff)
        return _replace(context.predictions, plan.primary_id, filtered_guidance + uncond_p)

    return GuidanceContribution(
        pre_cfg=(
            GuidancePreCFGDescriptor("dinkster.fresca", transform, behavior_metadata=metadata),
        )
    )


def _project(v0: torch.Tensor, v1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # torch.nn.functional.normalize with eps and expand_as, kept inline so the
    # arithmetic matches the reference bit for bit.
    v1 = v1 / v1.norm(2.0, dim=[-1, -2, -3], keepdim=True).clamp_min(1e-12).expand_as(v1)
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel, v0_orthogonal


def apg(eta: float, norm_threshold: float, momentum: float) -> GuidanceContribution[torch.Tensor]:
    """Adaptive projected guidance (comfy_extras.nodes_apg.APG)."""
    metadata = (
        ("config.eta", str(eta)),
        ("config.momentum", str(momentum)),
        ("config.norm_threshold", str(norm_threshold)),
    )

    def transform(context: GuidancePreCFGContext[torch.Tensor]) -> _Predictions:
        plan = context.request.plan
        cond, uncond = _values(context)
        state = context.request.execution.extension_state[APG_STATE_NAMESPACE]
        sigma = float(context.request.sigma.reshape(-1)[0])
        prev_sigma = state.get("prev_sigma")
        stored_avg = state.get("running_avg")
        running_avg = stored_avg if isinstance(stored_avg, torch.Tensor) else None
        if isinstance(prev_sigma, float) and sigma > prev_sigma:
            running_avg = None
        state["prev_sigma"] = sigma

        guidance = cond - uncond
        if momentum != 0:
            running_avg = guidance if running_avg is None else momentum * running_avg + guidance
            state["running_avg"] = running_avg
            guidance = running_avg

        if norm_threshold > 0:
            guidance_norm = guidance.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale = torch.minimum(torch.ones_like(guidance_norm), norm_threshold / guidance_norm)
            guidance = guidance * scale

        guidance_parallel, guidance_orthogonal = _project(guidance, cond)
        modified_guidance = guidance_orthogonal + eta * guidance_parallel
        modified_cond = (uncond + modified_guidance) + (cond - uncond) / context.cfg_scale
        return _replace(context.predictions, plan.primary_id, modified_cond)

    return GuidanceContribution(
        pre_cfg=(GuidancePreCFGDescriptor("dinkster.apg", transform, behavior_metadata=metadata),)
    )


def mahiro() -> GuidanceContribution[torch.Tensor]:
    """Positive-biased guidance merge (comfy_extras.nodes_mahiro.Mahiro)."""

    def transform(context: GuidancePostCFGContext[torch.Tensor]) -> torch.Tensor:
        scale = context.cfg_scale
        cond_p, uncond_p = _values(context)
        leap = cond_p * scale
        u_leap = uncond_p * scale
        cfg = context.reduced
        merge = (leap + cfg) / 2
        normu = torch.sqrt(u_leap.abs()) * u_leap.sign()
        normm = torch.sqrt(merge.abs()) * merge.sign()
        sim = torch.nn.functional.cosine_similarity(normu, normm).mean()
        simsc = 2 * (sim + 1)
        return (simsc * cfg + (4 - simsc) * leap) / 4

    return GuidanceContribution(post_cfg=(GuidancePostCFGDescriptor("dinkster.mahiro", transform),))


def epsilon_scaling(scaling_factor: float) -> GuidanceContribution[torch.Tensor]:
    """Exposure-bias epsilon scaling (comfy_extras.nodes_eps.EpsilonScaling)."""
    if scaling_factor == 0:
        scaling_factor = 1e-9
    metadata = (("config.scaling_factor", str(scaling_factor)),)

    def transform(context: GuidancePostCFGContext[torch.Tensor]) -> torch.Tensor:
        x = context.request.input
        return x - (x - context.reduced) / scaling_factor

    return GuidanceContribution(
        post_cfg=(
            GuidancePostCFGDescriptor(
                "dinkster.epsilon-scaling", transform, behavior_metadata=metadata
            ),
        )
    )


def cfg_override(
    cfg: float,
    sigma_low: float,
    sigma_high: float,
) -> GuidanceContribution[torch.Tensor]:
    """Override CFG over one inclusive sigma interval."""
    metadata = (
        ("config.cfg", str(cfg)),
        ("config.sigma_high", str(sigma_high)),
        ("config.sigma_low", str(sigma_low)),
    )

    def effective_cfg(sigma: float, base_cfg: float) -> float:
        return cfg if sigma_low <= sigma <= sigma_high else base_cfg

    def plan(context: GuidancePlanContext[torch.Tensor]) -> GuidanceEvaluationPlan[torch.Tensor]:
        scale = effective_cfg(context.execution.current_sigma, context.cfg_scale)
        conditional = tuple(
            lane for lane in context.conditions if lane.role is GuidanceRole.CONDITIONAL
        )
        uncond = next(
            (lane for lane in context.conditions if lane.role is GuidanceRole.UNCONDITIONAL), None
        )
        if scale != 1.0 and (uncond is None or uncond.conditioning is None):
            raise GuidanceContractError(
                "CFG Override with an effective scale other than one requires "
                "unconditional conditioning"
            )
        if scale == 0.0 and uncond is not None:
            return GuidanceEvaluationPlan((uncond,), uncond.id, uncond.id)
        lanes = conditional
        if uncond is not None and (context.force_uncond or scale != 1.0):
            lanes += (uncond,)
        return GuidanceEvaluationPlan(
            lanes,
            lanes[0].id,
            uncond.id if uncond is not None and uncond in lanes else None,
        )

    def reduce(context: GuidanceReduceContext[torch.Tensor]) -> torch.Tensor:
        by_id = {item.lane_id: item.value for item in context.predictions.items}
        primary = by_id[context.request.plan.primary_id]
        unconditional_id = context.request.plan.unconditional_id
        if unconditional_id is None or unconditional_id == context.request.plan.primary_id:
            return primary
        strength = effective_cfg(context.request.execution.current_sigma, context.cfg_scale)
        if strength == 1.0:
            return primary
        if strength == 0.0:
            return by_id[unconditional_id]
        return cfg_combine(primary, by_id[unconditional_id], strength)

    return GuidanceContribution(
        strategy=GuidanceStrategyDescriptor(
            "dinkster.cfg-override",
            plan,
            reduce,
            participation=GuidancePhaseParticipation.COMPOSE,
            behavior_metadata=metadata,
            allows_unconditional_primary=True,
            compatible_post_cfg_kinds=("rescaler",),
        )
    )


def scheduled_cfg(
    sigmas: tuple[float, ...],
    from_cfg: float,
    to_cfg: float,
    schedule: str,
) -> GuidanceContribution[torch.Tensor]:
    """Per-evaluation CFG schedule from Inspire Pack's ScheduledCFGGuider."""
    if schedule not in {"linear", "log", "exp", "cos"}:
        raise ValueError(f"unknown scheduled CFG interpolation {schedule!r}")
    if len(sigmas) < 2:
        raise ValueError("scheduled CFG requires at least two sigma values")
    steps = len(sigmas) - 1
    by_sigma: dict[float, tuple[float, int]] = {}
    by_index: dict[int, tuple[float, int]] = {}
    for index, sigma in enumerate(sigmas):
        if schedule == "exp":
            if index == steps - 1:
                value = to_cfg
            elif from_cfg == to_cfg:
                value = from_cfg
            elif from_cfg == 0:
                value = to_cfg * (1 - math.exp(-5 * index / steps)) / (1 - math.exp(-5))
            elif to_cfg == 0:
                value = (
                    from_cfg * (math.exp(-5 * index / steps) - math.exp(-5)) / (1 - math.exp(-5))
                )
            else:
                log_from = math.log(from_cfg)
                log_to = math.log(to_cfg)
                value = math.exp(log_from + (log_to - log_from) * index / steps)
        elif schedule == "log":
            if index == 0:
                value = from_cfg
            elif index == steps - 1:
                value = to_cfg
            else:
                value = from_cfg + (to_cfg - from_cfg) * math.log(index + 1) / math.log(steps + 1)
        elif schedule == "cos":
            if index == 0 or index == steps - 1:
                value = from_cfg
            else:
                interpolation = (1.0 + math.cos(math.pi * 2 * (index / steps))) / 2
                value = from_cfg + (to_cfg - from_cfg) * interpolation
        else:
            value = from_cfg + (to_cfg - from_cfg) * index / steps
        scheduled = (value, index)
        by_sigma[float(sigma)] = scheduled
        by_index[index] = scheduled

    last_index = 0

    def transform(context: GuidancePlanContext[torch.Tensor]) -> float:
        nonlocal last_index
        sigma = context.execution.current_sigma
        scheduled = by_sigma.get(sigma)
        if scheduled is None:
            scheduled = by_index[last_index + 1]
            by_sigma[sigma] = scheduled
        last_index = scheduled[1]
        return scheduled[0]

    return GuidanceContribution(
        scale=(
            GuidanceScaleDescriptor(
                "dinkster.scheduled-cfg",
                transform,
                behavior_metadata=(
                    ("config.from_cfg", str(from_cfg)),
                    ("config.schedule", schedule),
                    ("config.to_cfg", str(to_cfg)),
                ),
            ),
        )
    )


def rescale_cfg(multiplier: float, *, flow: bool) -> GuidanceContribution[torch.Tensor]:
    """Rescaled classifier-free guidance (comfy_extras.nodes_model_advanced.RescaleCFG).

    ``flow`` is the reference's patch-time ``isinstance(model_sampling, CONST)``
    check: flow models rescale directly in x0 space; every other
    parameterization converts through the reference's v-space formula, which
    hard-codes 4-D latents (std over dims 1..3) exactly as the reference does.
    """
    metadata = (
        ("config.flow", flow),
        ("config.multiplier", str(multiplier)),
    )

    if flow:

        def transform(context: GuidancePostCFGContext[torch.Tensor]) -> torch.Tensor:
            x_orig = context.request.input
            cond_p, _ = _values(context)
            x_0_cfg = context.reduced
            dims = tuple(range(1, cond_p.ndim))
            ro_pos = cond_p.std(dim=dims, keepdim=True)
            ro_cfg = x_0_cfg.std(dim=dims, keepdim=True).clamp(min=1e-8)
            x_0_rescaled = x_0_cfg * (ro_pos / ro_cfg)
            x_0_final = multiplier * x_0_rescaled + (1.0 - multiplier) * x_0_cfg
            return x_orig - (x_orig - x_0_final)

        return GuidanceContribution(
            post_cfg=(
                GuidancePostCFGDescriptor(
                    "dinkster.rescale-cfg",
                    transform,
                    behavior_metadata=metadata,
                    composition_kind="rescaler",
                ),
            )
        )

    def reduce(context: GuidanceReduceContext[torch.Tensor]) -> torch.Tensor:
        # The reference's sampler_cfg_function returns a noise-space value n
        # and cfg_function emits x - n; both subtractions are replicated so
        # the emitted values stay bit-identical to the executed reference.
        x_orig = context.request.input
        cond_scale = context.cfg_scale
        cond_p, uncond_p = _values(context)
        sigma = context.request.sigma.reshape(-1)[0]
        cond_noise = x_orig - cond_p
        uncond_noise = x_orig - uncond_p
        x = x_orig / (sigma * sigma + 1.0)
        cond_v = ((x - (x_orig - cond_noise)) * (sigma**2 + 1.0) ** 0.5) / sigma
        uncond_v = ((x - (x_orig - uncond_noise)) * (sigma**2 + 1.0) ** 0.5) / sigma
        x_cfg = uncond_v + cond_scale * (cond_v - uncond_v)
        ro_pos = torch.std(cond_v, dim=(1, 2, 3), keepdim=True)
        ro_cfg = torch.std(x_cfg, dim=(1, 2, 3), keepdim=True)
        x_rescaled = x_cfg * (ro_pos / ro_cfg)
        x_final = multiplier * x_rescaled + (1.0 - multiplier) * x_cfg
        return x_orig - (x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5))

    return GuidanceContribution(
        strategy=GuidanceStrategyDescriptor(
            "dinkster.rescale-cfg",
            _standard_plan,
            reduce,
            participation=GuidancePhaseParticipation.COMPOSE,
            behavior_metadata=metadata,
        )
    )


def trellis2_rescale_cfg(
    multiplier: float, *, sigma_min: float = 1e-5
) -> GuidanceContribution[torch.Tensor]:
    """Microsoft TRELLIS.2 velocity-to-x0 guidance rescale."""
    metadata = (
        ("config.multiplier", str(multiplier)),
        ("config.sigma_min", str(sigma_min)),
    )

    def transform(context: GuidancePostCFGContext[torch.Tensor]) -> torch.Tensor:
        if multiplier <= 0.0 or context.request.plan.unconditional_id in (
            None,
            context.request.plan.primary_id,
        ):
            return context.reduced
        sigma = context.request.execution.current_sigma
        if sigma <= 0.0:
            raise GuidanceContractError(
                "TRELLIS.2 guidance rescale requires a positive executed sigma"
            )
        x_t = context.request.input
        cond_denoised, _ = _values(context)
        pred_pos = (x_t - cond_denoised) / sigma
        pred_cfg = (x_t - context.reduced) / sigma
        source_sigma = sigma_min + (1.0 - sigma_min) * sigma
        source_origin = (1.0 - sigma_min) * x_t
        x_0_pos = source_origin - source_sigma * pred_pos
        x_0_cfg = source_origin - source_sigma * pred_cfg
        dims = tuple(range(1, x_0_pos.ndim))
        std_pos = x_0_pos.std(dim=dims, keepdim=True)
        std_cfg = x_0_cfg.std(dim=dims, keepdim=True)
        x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
        x_0 = multiplier * x_0_rescaled + (1.0 - multiplier) * x_0_cfg
        pred = (source_origin - x_0) / source_sigma
        return x_t - sigma * pred

    return GuidanceContribution(
        post_cfg=(
            GuidancePostCFGDescriptor(
                "dinkster.trellis2-rescale-cfg",
                transform,
                behavior_metadata=metadata,
                composition_kind="rescaler",
            ),
        )
    )


def renorm_cfg(
    cfg_trunc: float, renorm: float, *, in_channels: int | None = None
) -> GuidanceContribution[torch.Tensor]:
    """Truncated renormalized guidance (comfy_extras.nodes_lumina2.RenormCFG).

    ``in_channels`` mirrors the reference's read of the diffusion core's input
    channel count; None takes every prediction channel, which matches the
    reference whenever the model input has at least as many channels as the
    latent. The renorm comparison collapses the batched norm tensor to a
    Python bool exactly as the reference does, so like the reference it
    supports only batch-1 predictions when ``renorm`` is positive.
    """
    metadata = (
        ("config.cfg_trunc", str(cfg_trunc)),
        ("config.in_channels", in_channels),
        ("config.renorm", str(renorm)),
    )

    def reduce(context: GuidanceReduceContext[torch.Tensor]) -> torch.Tensor:
        # The reference returns x - cfg_result and cfg_function emits
        # x - that; both subtractions are replicated for bit-identity.
        x_orig = context.request.input
        cond_p, uncond_p = _values(context)
        sigma = float(context.request.sigma.reshape(-1)[0])
        channels = cond_p.shape[1] if in_channels is None else in_channels
        cond_eps, uncond_eps = cond_p[:, :channels], uncond_p[:, :channels]
        cond_rest = cond_p[:, channels:]
        if sigma < cfg_trunc:
            half_eps = uncond_eps + context.cfg_scale * (cond_eps - uncond_eps)
            if float(renorm) > 0.0:
                ori_pos_norm = torch.linalg.vector_norm(
                    cond_eps, dim=tuple(range(1, len(cond_eps.shape))), keepdim=True
                )
                max_new_norm = ori_pos_norm * float(renorm)
                new_pos_norm = torch.linalg.vector_norm(
                    half_eps, dim=tuple(range(1, len(half_eps.shape))), keepdim=True
                )
                if new_pos_norm >= max_new_norm:
                    half_eps = half_eps * (max_new_norm / new_pos_norm)
        else:
            half_eps = cond_eps
        cfg_result = torch.cat([half_eps, cond_rest], dim=1)
        return x_orig - (x_orig - cfg_result)

    return GuidanceContribution(
        strategy=GuidanceStrategyDescriptor(
            "dinkster.renorm-cfg",
            _standard_plan,
            reduce,
            participation=GuidancePhaseParticipation.COMPOSE,
            behavior_metadata=metadata,
        )
    )


def perp_neg(neg_scale: float) -> GuidanceContribution[torch.Tensor]:
    """Perpendicular-negative guidance (comfy_extras/nodes_perpneg.py
    Guider_PerpNeg).

    Expects the three-lane plan compiled from PerpNegSamplingGuidance:
    "positive" (conditional), "negative" (unconditional), and "empty"
    (auxiliary). The plan reproduces the reference's cfg==1-style lane
    dropping: the negative lane evaluates only when ``neg_scale`` is not
    approximately zero or an unconditional prediction is forced, and the
    empty lane is dropped only when the negative lane is dropped and the
    cfg scale is approximately one. Dropped lanes reduce against
    calc_cond_batch's untouched zeros, exactly as the reference does. The
    perpendicular rejection sums over every element (one global dot
    product, not per-batch), matching the reference.

    Deliberate divergence: the reference gates its lane dropping behind
    ``model_options.get("disable_cfg1_optimization", False)`` (dropping is
    the default); Dinkster drops unconditionally and mirrors no such flag.
    The executed goldens are minted under the reference's default
    (optimization-active) path.

    Post-CFG contributions stacked with this strategy observe every
    evaluated lane, including "empty", through ``context.predictions`` -
    the same surface the reference exposes as ``empty_cond_denoised``.
    """
    metadata = (("config.neg_scale", str(neg_scale)),)

    def plan(
        context: GuidancePlanContext[torch.Tensor],
    ) -> GuidanceEvaluationPlan[torch.Tensor]:
        by_id = {lane.id: lane for lane in context.conditions}
        needs_negative = context.force_uncond or not math.isclose(neg_scale, 0.0)
        needs_empty = needs_negative or cfg_needs_uncond(context.cfg_scale)
        lanes = (by_id["positive"],)
        if needs_negative:
            lanes += (by_id["negative"],)
        if needs_empty:
            lanes += (by_id["empty"],)
        return GuidanceEvaluationPlan(lanes, "positive", "negative" if needs_negative else None)

    def reduce(context: GuidanceReduceContext[torch.Tensor]) -> torch.Tensor:
        by_id = {item.lane_id: item.value for item in context.predictions.items}

        def lane(lane_id: str) -> torch.Tensor:
            value = by_id.get(lane_id)
            return value if value is not None else torch.zeros_like(context.request.input)

        positive = by_id["positive"]
        negative = lane("negative")
        empty = lane("empty")
        pos = positive - empty
        neg = negative - empty
        perp = neg - ((torch.mul(neg, pos).sum()) / (torch.norm(pos) ** 2)) * pos
        perp_scaled = perp * neg_scale
        return empty + context.cfg_scale * (pos - perp_scaled)

    return GuidanceContribution(
        strategy=GuidanceStrategyDescriptor(
            "dinkster.perp-neg",
            plan,
            reduce,
            participation=GuidancePhaseParticipation.COMPOSE,
            behavior_metadata=metadata,
        )
    )


def nag(nag_scale: float, nag_alpha: float, nag_tau: float) -> GuidanceContribution[torch.Tensor]:
    """Normalized Attention Guidance (comfy_extras/nodes_nag.py NAGuidance).

    An attention-kind contribution: the transform rewrites the
    conditional rows of every attn1 (self-attention) output inside one
    fused conditional/unconditional forward, upstream of every CFG
    phase. Per site: guided = pos*nag_scale - neg*(nag_scale-1); L1
    norms over the last dim (keepdim, clamped to 1e-6) form ratio =
    norm(guided)/norm(pos); guided is scaled by min(ratio, nag_tau)/
    ratio and blended as guided*nag_alpha + pos*(1-nag_alpha). Exact
    reference math and op order.

    "dinkster.nag" is this contribution's reserved descriptor id.

    ``requires_uncond=True`` mirrors the reference's
    ``disable_model_cfg1_optimization()``: NAG needs the real negative
    stream evaluated even at cfg 1.

    Deliberate divergence: the reference's calc_cond_batch splits the
    conditional and unconditional lanes into separate forwards on shape
    mismatch or free-memory pressure, and its patch then silently
    no-ops, so upstream NAG application varies with VRAM. Dinkster refuses
    when the lanes cannot share one fused forward instead of silently
    skipping the rewrite.

    Scope: only the reference's no-slice path is implemented. The
    ``img_slice`` branch arises on mixed text/image token streams;
    SD-era attn1 sees image tokens only. Admitting a mixed-stream
    family requires porting that branch, including its both-lanes
    overwrite, with executed golden coverage."""
    metadata = (
        ("config.nag_alpha", str(nag_alpha)),
        ("config.nag_scale", str(nag_scale)),
        ("config.nag_tau", str(nag_tau)),
    )

    def transform(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        guided = positive * nag_scale - negative * (nag_scale - 1.0)
        eps = 1e-6
        norm_pos = torch.norm(positive, p=1, dim=-1, keepdim=True).clamp_min(eps)
        norm_guided = torch.norm(guided, p=1, dim=-1, keepdim=True).clamp_min(eps)
        ratio = norm_guided / norm_pos
        scale_factor = torch.minimum(ratio, torch.full_like(ratio, nag_tau)) / ratio
        guided_normalized = guided * scale_factor
        return guided_normalized * nag_alpha + positive * (1.0 - nag_alpha)

    return GuidanceContribution(
        attention=AttentionGuidanceDescriptor(
            "dinkster.nag",
            transform,
            requires_uncond=True,
            behavior_metadata=metadata,
        )
    )


def temporal_score_rescaling(
    tsr_k: float, tsr_sigma: float, *, flow: bool
) -> GuidanceContribution[torch.Tensor]:
    """Temporal score rescaling (comfy_extras.nodes_eps.TemporalScoreRescaling).

    ``flow`` selects the reference's sigma_to_half_log_snr branch: flow models
    use -logit(sigma), every other parameterization -log(sigma).
    """
    tsr_variance = tsr_sigma**2
    metadata = (
        ("config.flow", flow),
        ("config.tsr_k", str(tsr_k)),
        ("config.tsr_sigma", str(tsr_sigma)),
    )

    def transform(context: GuidancePostCFGContext[torch.Tensor]) -> torch.Tensor:
        denoised = context.reduced
        x = context.request.input
        sigma = context.request.sigma.reshape(-1)[0]
        if tsr_k == 1 or sigma == 0:
            return denoised
        half_log_snr = sigma.logit().neg() if flow else sigma.log().neg()
        snr = (2 * half_log_snr).exp()
        if snr == 0:
            return denoised
        posinf_mask = torch.isposinf(snr)
        rescaling_factor = (snr * tsr_variance + 1) / (snr * tsr_variance / tsr_k + 1)
        rescaling_r = torch.where(posinf_mask, tsr_k, rescaling_factor)
        alpha = sigma * half_log_snr.exp()
        return torch.lerp(x / alpha, denoised, rescaling_r)

    return GuidanceContribution(
        post_cfg=(
            GuidancePostCFGDescriptor(
                "dinkster.temporal-score-rescaling", transform, behavior_metadata=metadata
            ),
        )
    )


def _latent_operation_gaussian_kernel(
    kernel_size: int, sigma: float, device: torch.device
) -> torch.Tensor:
    """Port of comfy_extras.nodes_post_processing.gaussian_kernel at b78cec87."""
    x, y = torch.meshgrid(
        torch.linspace(-1, 1, kernel_size, device=device),
        torch.linspace(-1, 1, kernel_size, device=device),
        indexing="ij",
    )
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    return (g / g.sum()).to(torch.float32)


def apply_latent_operation(
    kind: str, params: tuple[tuple[str, float | int], ...], latent: torch.Tensor
) -> torch.Tensor:
    """Port of the comfy_extras.nodes_latent.LatentOperation* closures at b78cec87."""
    values = dict(params)
    if kind == "tonemap_reinhard":
        multiplier = values["multiplier"]
        latent_vector_magnitude = (torch.linalg.vector_norm(latent, dim=1) + 0.0000000001)[:, None]
        normalized_latent = latent / latent_vector_magnitude
        dims = list(range(1, latent_vector_magnitude.ndim))
        mean = torch.mean(latent_vector_magnitude, dim=dims, keepdim=True)
        std = torch.std(latent_vector_magnitude, dim=dims, keepdim=True)
        top = (std * 5 + mean) * multiplier
        latent_vector_magnitude *= 1.0 / top
        new_magnitude = latent_vector_magnitude / (latent_vector_magnitude + 1.0)
        new_magnitude *= top
        return normalized_latent * new_magnitude
    if kind == "sharpen":
        sharpen_radius = int(values["sharpen_radius"])
        sigma = float(values["sigma"])
        alpha = float(values["alpha"])
        luminance = (torch.linalg.vector_norm(latent, dim=1) + 1e-6)[:, None]
        normalized_latent = latent / luminance
        channels = latent.shape[1]
        kernel_size = sharpen_radius * 2 + 1
        kernel = _latent_operation_gaussian_kernel(kernel_size, sigma, luminance.device)
        center = kernel_size // 2
        kernel *= alpha * -10
        kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0
        padded_image = torch.nn.functional.pad(
            normalized_latent,
            (sharpen_radius, sharpen_radius, sharpen_radius, sharpen_radius),
            "reflect",
        )
        sharpened = torch.nn.functional.conv2d(
            padded_image,
            kernel.repeat(channels, 1, 1).unsqueeze(1),
            padding=kernel_size // 2,
            groups=channels,
        )[:, :, sharpen_radius:-sharpen_radius, sharpen_radius:-sharpen_radius]
        return luminance * sharpened
    raise ValueError(f"unknown latent operation kind {kind!r}")


def latent_operation(
    kind: str,
    params: tuple[tuple[str, float | int], ...],
    *,
    descriptor_id: str = "dinkster.latent-operation",
    order: int = 0,
) -> GuidanceContribution[torch.Tensor]:
    """Pre-CFG latent operation (comfy_extras.nodes_latent.LatentApplyOperationCFG).

    The reference applies the operation to the guidance difference when a
    model-evaluated unconditional prediction exists and to the conditional
    prediction alone otherwise (calc_cond_batch's zeros substitute at cfg
    scale one, and op(cond - 0) + 0 equals op(cond)). It never forces an
    unconditional evaluation, so requires_uncond stays False.
    """
    metadata = tuple(
        sorted(
            [("config.kind", kind)]
            + [
                (f"config.{name}", str(value) if isinstance(value, float) else value)
                for name, value in params
            ]
        )
    )

    def transform(context: GuidancePreCFGContext[torch.Tensor]) -> _Predictions:
        plan = context.request.plan
        cond_p, _ = _values(context)
        uncond = _model_uncond(context)
        if uncond is None:
            return _replace(
                context.predictions, plan.primary_id, apply_latent_operation(kind, params, cond_p)
            )
        _, uncond_p = uncond
        transformed = apply_latent_operation(kind, params, cond_p - uncond_p) + uncond_p
        return _replace(context.predictions, plan.primary_id, transformed)

    return GuidanceContribution(
        pre_cfg=(
            GuidancePreCFGDescriptor(
                descriptor_id, transform, order=order, behavior_metadata=metadata
            ),
        )
    )
