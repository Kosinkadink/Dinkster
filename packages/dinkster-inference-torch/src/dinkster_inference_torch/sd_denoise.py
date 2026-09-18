"""SD1/SDXL denoise bridge: the EPS _apply_model path plus SDXL's
ADM conditioning vectors.

SDDenoiser is the SD-era sibling of FluxDenoiser: it owns one EPS
preconditioning (calculate_input's 1/sqrt(sigma^2+1)), the discrete
sigma -> timestep table lookup (ModelSamplingDiscrete.timestep), the
compute-dtype casts, and compatible conditioning batching, mirroring
comfy/model_base.py BaseModel._apply_model plus calc_cond_batch's
model-call batching @ b78cec87 for the classic SD path. It drives
through the same run_denoise as Flux.

SD-era cross-attention conditioning is the reference's CONDCrossAttn
(comfy/conds.py @ b78cec87): unequal token counts still batch into
one forward by repeating each sequence to the least common multiple
of the lengths - exact for cross-attention because softmax-weighted
sums are invariant under uniform K/V duplication - as long as the
repeat factor stays within the reference's limit of 4; beyond it,
two forwards (the reference's can_concat refusal path). This
revives the "repeat-to-lcm cond batching" deferral at its named
trigger: the first CONDCrossAttn family.

encode_sdxl_adm / encode_sdxl_refiner_adm port SDXL.encode_adm and
SDXLRefiner.encode_adm (comfy/model_base.py @ b78cec87): the pooled
CLIP-G vector concatenated with Timestep(256) size embeddings
(height, width, crop_h, crop_w, then target sizes for the base or
the aesthetic score for the refiner). Defaults match the reference's
kwargs.get fallbacks - what a plain CLIPTextEncode + KSampler
workflow hits. The reference's unCLIP noise augmentation
(unclip_conditioning) is not ported; SD-era unCLIP checkpoints are
outside the wired families.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from dinkster_inference import (
    SD15_CONTROL_RESIDUAL_SITES,
    SD15_IPADAPTER_SITES,
    SD_CONTROL_MODE_INDEX,
    SDXL_CONTROL_RESIDUAL_SITES,
    AttentionGuidanceDescriptor,
    Conditioning,
    ContinuousEDMSigmas,
    GuidanceRole,
    Parameterization,
    SDControlMode,
    SigmaSpace,
    calculate_denoised,
    calculate_input,
)

from ._conditioning_layout import (
    CROSS_ATTN_REPEAT_LIMIT,
    cross_attn_repeat,
    declared_token_count,
    repeat_cross_attn,
)
from .controlnet import (
    SDControlProvider,
    SDControlResiduals,
    SDEffectMaskField,
    SDXLControlLoRA,
    SDXLControlNet,
    SDXLControlNetUnion,
    _snapshot_sd_effect_mask_field,  # pyright: ignore[reportPrivateUsage]
    normalize_control_hint,
)
from .denoise import DenoiseError, to_batch
from .ipadapter import (
    SD15AttentionExecutionContext,
    SD15IPAdapterConditioning,
    SD15IPAdapterExecution,
)
from .t2i_adapter import SD15T2IAdapter
from .unet import AttentionGuidanceContext, UNetModel, timestep_embedding

#: model_base.py encode_adm kwargs.get defaults @ b78cec87.
SDXL_ADM_DEFAULT_SIZE = 768
SDXL_AESTHETIC_DEFAULT = 6.0
SDXL_NEGATIVE_AESTHETIC_DEFAULT = 2.5

_BLANK_INPAINT_LATENT = (0.8223, -0.6876, 0.6364, 0.1380)


def _calculate_input(
    parameterization: Parameterization, sigma: float, noise: torch.Tensor
) -> torch.Tensor:
    """Executed input preconditioning on ComfyUI's scalar kernels.

    EPS and v-prediction over either discrete or continuous-EDM sigma spaces
    use ComfyUI's device-float32 square/root/divide expression. Other
    parameterizations retain the already-pinned portable behavior.
    """
    if parameterization in (Parameterization.EPS, Parameterization.V_PREDICTION):
        sigma_tensor = torch.tensor(sigma, device=noise.device, dtype=torch.float32)
        return noise / (sigma_tensor**2 + 1.0**2) ** 0.5
    return calculate_input(parameterization, sigma, noise)


def _calculate_denoised(
    parameterization: Parameterization,
    sigma: float,
    model_output: torch.Tensor,
    model_input: torch.Tensor,
) -> torch.Tensor:
    """Executed discrete/continuous-EDM v-pred transform on ComfyUI's float32
    tensor kernels."""
    if parameterization is Parameterization.V_PREDICTION:
        sigma_tensor = torch.tensor(sigma, device=model_output.device, dtype=torch.float32)
        sigma_data = 1.0
        return (
            model_input * sigma_data**2 / (sigma_tensor**2 + sigma_data**2)
            - model_output * sigma_tensor * sigma_data / (sigma_tensor**2 + sigma_data**2) ** 0.5
        )
    return calculate_denoised(parameterization, sigma, model_output, model_input)


def _center_upscale(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """ComfyUI common_upscale(..., bilinear, center) for NCHW."""
    old_height, old_width = value.shape[-2:]
    height, width = size
    old_aspect = old_width / old_height
    new_aspect = width / height
    x = y = 0
    if old_aspect > new_aspect:
        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
    cropped = value.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    return torch.nn.functional.interpolate(cropped, size=size, mode="bilinear")


def _resize_batch(value: torch.Tensor, batch: int) -> torch.Tensor:
    """ComfyUI resize_to_batch_size's distributed row selection."""
    source = value.shape[0]
    if source == batch:
        return value
    if batch <= 1:
        return value[:batch]
    if batch < source:
        scale = (source - 1) / (batch - 1)
        indexes = [min(round(index * scale), source - 1) for index in range(batch)]
    else:
        scale = source / batch
        indexes = [min(math.floor((index + 0.5) * scale), source - 1) for index in range(batch)]
    return value[indexes]


def inpaint_model_input(
    noise: torch.Tensor,
    *,
    denoise_mask: torch.Tensor | None,
    masked_image: torch.Tensor | None,
) -> torch.Tensor:
    """Build ComfyUI's native SD inpaint UNet input: noisy latent,
    rounded denoise mask, then masked-image latent.

    ``masked_image`` is already in model latent space, matching
    BaseModel.concat_cond's process_latent_in-before-resize order.
    With no mask the reference ignores any latent image and supplies
    an all-one mask plus its fixed blank latent constants.
    """
    if noise.dim() != 4 or noise.shape[1] != 4:
        raise DenoiseError(
            f"SD inpaint noise must be [batch x 4 x height x width], got shape {tuple(noise.shape)}"
        )
    if denoise_mask is None:
        mask = torch.ones_like(noise[:, :1])
        image = torch.ones_like(noise)
        for channel, value in enumerate(_BLANK_INPAINT_LATENT):
            image[:, channel].mul_(value)
    else:
        if masked_image is None:
            raise DenoiseError("SD inpaint with a mask needs a masked-image latent")
        mask = denoise_mask
        if mask.dim() == noise.dim():
            mask = mask[:, :1]
        spatial_dims = noise.dim() - 2
        if mask.dim() < spatial_dims:
            raise DenoiseError(f"SD inpaint mask has too few dimensions: shape {tuple(mask.shape)}")
        mask = mask.reshape((-1, 1) + tuple(mask.shape[-spatial_dims:]))
        target_size = (noise.shape[-2], noise.shape[-1])
        if mask.shape[-2:] != noise.shape[-2:]:
            mask = _center_upscale(mask, target_size)
        mask = _resize_batch(mask.round(), noise.shape[0]).to(
            device=noise.device, dtype=noise.dtype
        )
        image = masked_image
        if image.dim() != 4 or image.shape[1] != 4:
            raise DenoiseError(
                "SD inpaint masked image must be [batch x 4 x height x width],"
                f" got shape {tuple(image.shape)}"
            )
        if image.shape[1:] != noise.shape[1:]:
            image = _center_upscale(image, target_size)
        image = _resize_batch(image, noise.shape[0]).to(device=noise.device, dtype=noise.dtype)
    return torch.cat((noise, mask, image), dim=1)


def _size_embedding(value: float) -> torch.Tensor:
    """One Timestep(256) size embedding, [1 x 256] float32 (the
    reference's self.embedder(torch.Tensor([value])))."""
    return timestep_embedding(torch.tensor([float(value)]), 256)


def _adm_cat(pooled: torch.Tensor, scalars: list[float]) -> torch.Tensor:
    if pooled.dim() != 2:
        raise DenoiseError(
            f"ADM pooled must be [batch x features], got shape {tuple(pooled.shape)}"
        )
    flat = torch.cat([_size_embedding(value) for value in scalars], dim=1)
    flat = flat.to(device=pooled.device, dtype=torch.float32)
    flat = flat.repeat(pooled.shape[0], 1)
    return torch.cat((pooled.float(), flat), dim=1)


def encode_sdxl_adm(
    pooled: torch.Tensor,
    *,
    width: float = SDXL_ADM_DEFAULT_SIZE,
    height: float = SDXL_ADM_DEFAULT_SIZE,
    crop_w: float = 0.0,
    crop_h: float = 0.0,
    target_width: float | None = None,
    target_height: float | None = None,
) -> torch.Tensor:
    """SDXL.encode_adm @ b78cec87: [batch x 2816] = pooled CLIP-G
    (1280) + six Timestep(256) embeddings in reference order
    (height, width, crop_h, crop_w, target_height, target_width).
    Sizes are PIXEL values (the reference nodes pass latent * 8);
    targets default to the size, exactly like the reference."""
    if target_width is None:
        target_width = width
    if target_height is None:
        target_height = height
    return _adm_cat(pooled, [height, width, crop_h, crop_w, target_height, target_width])


def encode_sdxl_refiner_adm(
    pooled: torch.Tensor,
    *,
    width: float = SDXL_ADM_DEFAULT_SIZE,
    height: float = SDXL_ADM_DEFAULT_SIZE,
    crop_w: float = 0.0,
    crop_h: float = 0.0,
    aesthetic_score: float = SDXL_AESTHETIC_DEFAULT,
) -> torch.Tensor:
    """SDXLRefiner.encode_adm @ b78cec87: [batch x 2560] = pooled
    CLIP-G (1280) + five Timestep(256) embeddings (height, width,
    crop_h, crop_w, aesthetic_score). The reference defaults the
    score by prompt polarity (6 positive, 2.5 negative); the caller
    picks per conditioning."""
    return _adm_cat(pooled, [height, width, crop_h, crop_w, aesthetic_score])


def _checked_sd_cond(
    cond: Conditioning[torch.Tensor], what: str
) -> tuple[torch.Tensor, torch.Tensor | None]:
    context = cond.embeddings
    if context.dim() != 3:
        raise DenoiseError(
            f"{what} embeddings must be [batch x tokens x features],"
            f" got shape {tuple(context.shape)}"
        )
    token_count = declared_token_count(cond)
    if token_count is not None and context.shape[1] != token_count:
        raise DenoiseError(
            f"{what} embeddings have {context.shape[1]} token rows;"
            f" the family encoder declared {token_count}"
        )
    pooled = cond.pooled
    if pooled is not None and pooled.dim() != 2:
        raise DenoiseError(
            f"{what} pooled must be [batch x features], got shape {tuple(pooled.shape)}"
        )
    return context, pooled


SDCondition = tuple[torch.Tensor, torch.Tensor | None, str]


@dataclass(frozen=True, slots=True)
class SDControlGain:
    """Resolved effective gains in residual-site and guidance-lane order."""

    lane_ids: tuple[str, ...]
    site_lane_gains: tuple[tuple[float, ...], ...]
    effect_mask_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.lane_ids or len(self.lane_ids) != len(set(self.lane_ids)):
            raise DenoiseError("control gain lane ids must be non-empty and unique")
        if len(self.site_lane_gains) not in (
            len(SD15_CONTROL_RESIDUAL_SITES),
            len(SDXL_CONTROL_RESIDUAL_SITES),
        ):
            raise DenoiseError("control gain must cover one exact SD residual-site layout")
        for gains in self.site_lane_gains:
            if len(gains) != len(self.lane_ids):
                raise DenoiseError("control gain rows must cover every guidance lane")
            if any(type(gain) is not float or not math.isfinite(gain) for gain in gains):
                raise DenoiseError("control site/lane gains must be finite exact floats")
        if type(self.effect_mask_digests) is not tuple or any(
            type(digest) is not str or len(digest) != 64 for digest in self.effect_mask_digests
        ):
            raise DenoiseError("control effect masks must be SHA-256 declaration digests")

    def gains_for(self, site_index: int, lane_ids: tuple[str, ...]) -> tuple[float, ...]:
        by_lane = dict(zip(self.lane_ids, self.site_lane_gains[site_index], strict=True))
        try:
            return tuple(by_lane[lane_id] for lane_id in lane_ids)
        except KeyError as error:
            raise DenoiseError(
                f"control gain does not declare guidance lane {error.args[0]!r}"
            ) from error

    def active_for(self, lane_ids: tuple[str, ...]) -> bool:
        return any(
            gain != 0.0
            for site_index in range(len(self.site_lane_gains))
            for gain in self.gains_for(site_index, lane_ids)
        )


def _control_residual_sites(model: SDControlProvider) -> tuple[str, ...]:
    if type(model) in (SDXLControlLoRA, SDXLControlNet, SDXLControlNetUnion):
        return SDXL_CONTROL_RESIDUAL_SITES
    return SD15_CONTROL_RESIDUAL_SITES


def _scale_control_residual(
    value: torch.Tensor,
    gain: float | SDControlGain,
    site_index: int,
    lane_ids: tuple[str, ...],
    batch: int,
    effect_masks: tuple[SDEffectMaskField, ...],
    sites: tuple[str, ...],
) -> torch.Tensor:
    if type(gain) is float:
        return value * gain
    assert isinstance(gain, SDControlGain)
    factors = gain.gains_for(site_index, lane_ids)
    if len(factors) == 1:
        value = value * factors[0]
    else:
        per_batch = torch.tensor(factors, device=value.device, dtype=value.dtype).repeat_interleave(
            batch
        )
        value = value * per_batch.view(-1, 1, 1, 1)
    by_digest = {field.compiled.input_digest: field for field in effect_masks}
    for digest in gain.effect_mask_digests:
        field = by_digest[digest]
        if field.compiled.target_segment != sites[site_index]:
            continue
        mask = field.mask.to(device=value.device, dtype=value.dtype)
        source_batch = mask.shape[0]
        mask = mask.repeat((batch + source_batch - 1) // source_batch, 1, 1, 1)[:batch]
        if len(lane_ids) > 1:
            mask = mask.repeat(len(lane_ids), 1, 1, 1)
        value = value * mask
    return value


class SDDenoiser:
    """Single-conditioning evaluator over an assembled SD-era UNet.

    One call is BaseModel._apply_model @ b78cec87 for EPS or
    v-prediction over a discrete or continuous EDM sigma space:
    precondition the input on the reference's float32 tensor kernel, map
    sigma through the selected space's timestep function, cast input and
    conditioning to the compute dtype (timesteps stay float32), run the UNet,
    lift the output to float32, and convert EPS/v-prediction to denoised on
    the reference's float32 tensor kernel. The shared guidance layer
    may request one multi-conditioning forward when CONDCrossAttn's
    rules allow it (equal token counts, or repeat-to-lcm within factor
    4).

    ``adm_cond`` is a ready ADM vector
    (:func:`encode_sdxl_adm` / :func:`encode_sdxl_refiner_adm`) for
    ADM-conditioned models (SDXL base/refiner) and must be None for
    SD 1.5 - mismatches refuse at construction, mirroring the UNet's
    y-required-iff-adm validation. Model placement is the caller's
    business (residency seams); conditioning moves to the input's
    device per call.
    """

    def __init__(
        self,
        model: UNetModel,
        space: SigmaSpace,
        conditioning: Conditioning[torch.Tensor] | None = None,
        *,
        adm_cond: torch.Tensor | None = None,
        parameterization: Parameterization = Parameterization.EPS,
        inpaint_mask: torch.Tensor | None = None,
        inpaint_masked_image: torch.Tensor | None = None,
        control_model: SDControlProvider | tuple[SDControlProvider, ...] | None = None,
        control_hint: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        control_mode: SDControlMode | tuple[SDControlMode | None, ...] | None = None,
        control_effect_masks: tuple[tuple[SDEffectMaskField, ...], ...] | None = None,
        ipadapter: tuple[SD15IPAdapterExecution, ...] = (),
        compute_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model = model
        self.space = space
        self.parameterization = parameterization
        if model.config.in_channels not in (4, 9):
            raise DenoiseError(
                "SDDenoiser supports 4-channel base or 9-channel inpaint UNets,"
                f" got {model.config.in_channels} input channels"
            )
        if model.config.in_channels == 4 and (
            inpaint_mask is not None or inpaint_masked_image is not None
        ):
            raise DenoiseError("inpaint inputs require a 9-channel SD UNet")
        self._inpaint_mask = inpaint_mask
        self._inpaint_masked_image = inpaint_masked_image
        if (control_model is None) != (control_hint is None):
            raise DenoiseError("control model and hint must be supplied together")
        if control_model is not None and model.config.in_channels != 4:
            raise DenoiseError("ControlNet requires a 4-channel SD1.5 UNet")
        if isinstance(control_model, tuple):
            if not control_model or not isinstance(control_hint, tuple):
                raise DenoiseError("control model and hint chains must be non-empty tuples")
            if len(control_model) != len(control_hint):
                raise DenoiseError("control model and hint chains must have equal lengths")
            self._control_models = control_model
            self._control_hints = tuple(hint.detach().clone() for hint in control_hint)
        elif control_model is None:
            self._control_models = ()
            self._control_hints = ()
        else:
            if not isinstance(control_hint, torch.Tensor):
                raise DenoiseError("one control model requires one tensor hint")
            self._control_models = (control_model,)
            self._control_hints = (control_hint.detach().clone(),)
        self._control_sites = tuple(
            _control_residual_sites(model) for model in self._control_models
        )
        if self._control_sites and any(
            sites != self._control_sites[0] for sites in self._control_sites
        ):
            raise DenoiseError("control chain mixes incompatible residual site layouts")
        if isinstance(control_mode, tuple):
            modes = control_mode
        elif control_mode is None:
            modes = (None,) * len(self._control_models)
        elif self._control_models:
            modes = (control_mode,)
        else:
            modes = ()
        if len(modes) != len(self._control_models):
            raise DenoiseError("control modes must match the ControlNet chain")
        for control_provider, mode in zip(self._control_models, modes, strict=True):
            if type(control_provider) is SDXLControlNetUnion:
                if mode is not None and (
                    type(mode) is not SDControlMode or mode.provider != "sdxl-controlnet-union"
                ):
                    raise DenoiseError("SDXL ControlNet Union requires a Union mode or auto")
                if mode is not None and (
                    SD_CONTROL_MODE_INDEX[mode.token] >= control_provider.config.mode_capacity
                ):
                    raise DenoiseError("Union mode exceeds the detected artifact capacity")
            elif mode is not None:
                raise DenoiseError("non-Union SD control providers do not accept a mode")
        self._control_modes = modes
        self._control_hint_cache: dict[
            tuple[int, int, int, int, torch.device, torch.dtype, int], torch.Tensor
        ] = {}
        self._t2i_residual_cache: dict[
            tuple[int, int, int, int, int, int, torch.device, torch.dtype], SDControlResiduals
        ] = {}
        self._control_gains: tuple[float | SDControlGain, ...] = (0.0,) * len(self._control_models)
        if control_effect_masks is None:
            self._control_effect_masks = ((),) * len(self._control_models)
        else:
            if len(control_effect_masks) != len(self._control_models) or any(
                type(fields) is not tuple
                or any(type(field) is not SDEffectMaskField for field in fields)
                for fields in control_effect_masks
            ):
                raise DenoiseError("control effect-mask fields must match the ControlNet chain")
            snapshots = tuple(
                tuple(_snapshot_sd_effect_mask_field(field) for field in fields)
                for fields in control_effect_masks
            )
            if any(
                field.compiled.target_segment not in sites
                for fields, sites in zip(snapshots, self._control_sites, strict=True)
                for field in fields
            ):
                raise DenoiseError("effect-mask target does not belong to its control provider")
            self._control_effect_masks = snapshots
        if type(ipadapter) is not tuple or any(
            type(contribution) is not SD15IPAdapterExecution for contribution in ipadapter
        ):
            raise DenoiseError("IP-Adapter executions must be an exact tuple")
        if ipadapter:
            expected = tuple((site.id, site.width) for site in SD15_IPADAPTER_SITES)
            actual = model.attention_sites
            if len(actual) != len(set(actual)):
                raise DenoiseError("SD1.5 UNet declares duplicate attention sites")
            if set(actual) != set(expected):
                missing = sorted(set(expected) - set(actual))
                unknown = sorted(set(actual) - set(expected))
                details = []
                if missing:
                    details.append("missing " + ", ".join(site for site, _width in missing))
                if unknown:
                    details.append("unknown " + ", ".join(site for site, _width in unknown))
                raise DenoiseError(
                    "IP-Adapter requires the canonical 16-site SD1.5 UNet: " + "; ".join(details)
                )
        self._ipadapter = tuple(
            SD15IPAdapterExecution(
                SD15IPAdapterConditioning(
                    execution.conditioning.declaration,
                    execution.conditioning.model,
                    execution.conditioning.cond_tokens,
                    execution.conditioning.uncond_tokens,
                    execution.conditioning.model_digest,
                    execution.conditioning.token_digest,
                    execution.conditioning.mask,
                    execution.conditioning.mask_digest,
                ),
                execution.sigma_start,
                execution.sigma_end,
            )
            for execution in ipadapter
        )
        self.compute_dtype = compute_dtype
        self._conditioning = (
            None if conditioning is None else self.prepare_conditioning(conditioning, adm=adm_cond)
        )

    def set_control_gain(self, gain: float) -> None:
        if type(gain) is not float or not math.isfinite(gain):
            raise DenoiseError("control gain must be a finite exact float")
        if len(self._control_models) != 1:
            raise DenoiseError("scalar control gain requires exactly one ControlNet")
        if any(self._control_effect_masks):
            raise DenoiseError("effect-mask fields require structured control gain rows")
        self._control_gains = (gain,)

    def set_control_gains(self, gains: tuple[float, ...]) -> None:
        if len(gains) != len(self._control_models):
            raise DenoiseError("control gain count must match the ControlNet chain")
        if any(type(gain) is not float or not math.isfinite(gain) for gain in gains):
            raise DenoiseError("control gains must be finite exact floats")
        if any(self._control_effect_masks):
            raise DenoiseError("effect-mask fields require structured control gain rows")
        self._control_gains = gains

    def set_control_gain_rows(self, gains: tuple[SDControlGain, ...]) -> None:
        if len(gains) != len(self._control_models):
            raise DenoiseError("control gain count must match the ControlNet chain")
        if any(type(gain) is not SDControlGain for gain in gains):
            raise DenoiseError("control gain rows must be exact SDControlGain values")
        for gain, fields, sites in zip(
            gains, self._control_effect_masks, self._control_sites, strict=True
        ):
            if len(gain.site_lane_gains) != len(sites):
                raise DenoiseError("control gain residual-site layout does not match its provider")
            available = {field.compiled.input_digest for field in fields}
            if any(digest not in available for digest in gain.effect_mask_digests):
                raise DenoiseError("control gain row references an unresolved effect-mask field")
        self._control_gains = gains

    def prepare_conditioning(
        self,
        conditioning: object,
        *,
        adm: torch.Tensor | None = None,
        lane_id: str = "positive",
    ) -> SDCondition:
        if not isinstance(conditioning, Conditioning):
            raise DenoiseError("conditioning must be a Conditioning value")
        context, _ = _checked_sd_cond(conditioning, "conditioning")
        adm_channels = self.model.config.adm_in_channels
        if adm_channels is None and adm is not None:
            raise DenoiseError("ADM conditioning was given but this UNet is not ADM-conditioned")
        if adm_channels is not None and adm is None:
            raise DenoiseError(
                f"this UNet needs ADM conditioning (adm_in_channels = {adm_channels})"
            )
        if adm is not None and (adm.dim() != 2 or adm.shape[1] != adm_channels):
            raise DenoiseError(
                f"ADM conditioning must be [batch x {adm_channels}], got shape {tuple(adm.shape)}"
            )
        if type(lane_id) is not str or not lane_id:
            raise DenoiseError("guidance lane id must be a non-empty string")
        return context, adm, lane_id

    @staticmethod
    def batchable(conditions: tuple[SDCondition, ...]) -> bool:
        return (
            bool(conditions)
            and cross_attn_repeat([condition[0].shape[1] for condition in conditions]) is not None
        )

    def _forward(
        self,
        xc: torch.Tensor,
        sigma: float,
        contexts: list[torch.Tensor],
        adms: list[torch.Tensor | None],
        lane_ids: tuple[str, ...],
        repeats: list[int] | None = None,
        attention_guidance: AttentionGuidanceContext | None = None,
    ) -> torch.Tensor:
        """One UNet call over ``len(contexts)`` copies of ``xc``
        stacked along the batch axis; returns float32 output.
        ``repeats`` carries the CONDCrossAttn repeat-to-lcm factors
        (None = all 1)."""
        batch = xc.shape[0]
        if self.model.config.in_channels == 9:
            xc = inpaint_model_input(
                xc,
                denoise_mask=self._inpaint_mask,
                masked_image=self._inpaint_masked_image,
            )
        if len(contexts) > 1:
            xc = torch.cat([xc] * len(contexts), dim=0)
        total = xc.shape[0]
        device = xc.device
        ipadapter = SD15AttentionExecutionContext.for_sigma(
            self._ipadapter,
            sigma,
            lane_ids,
            batch,
            latent_height=xc.shape[-2],
            latent_width=xc.shape[-1],
            device=device,
            dtype=self.compute_dtype,
        )
        if isinstance(self.space, ContinuousEDMSigmas):
            sigma_tensor = torch.tensor(sigma, device=device, dtype=torch.float32)
            timesteps = (0.25 * sigma_tensor.log()).expand(total)
        else:
            timesteps = torch.full(
                (total,),
                self.space.timestep(sigma),
                device=device,
                dtype=torch.float32,
            )
        parts = []
        for i, context in enumerate(contexts):
            part = to_batch(context, batch).to(device=device, dtype=self.compute_dtype)
            factor = 1 if repeats is None else repeats[i]
            if factor > 1:
                # CONDCrossAttn.concat @ b78cec87: repeat-padding a
                # cross-attention sequence does not change the result.
                part = repeat_cross_attn(part, factor)
            parts.append(part)
        context = torch.cat(parts, dim=0)
        y = None
        if self.model.config.adm_in_channels is not None:
            y = torch.cat(
                [
                    to_batch(adm, batch).to(device=device, dtype=self.compute_dtype)
                    for adm in adms
                    if adm is not None
                ],
                dim=0,
            )
        control = None
        for control_index, (
            control_model,
            control_hint_source,
            control_gain,
            effect_masks,
        ) in enumerate(
            zip(
                self._control_models,
                self._control_hints,
                self._control_gains,
                self._control_effect_masks,
                strict=True,
            )
        ):
            if (type(control_gain) is float and control_gain == 0.0) or (
                type(control_gain) is SDControlGain and not control_gain.active_for(lane_ids)
            ):
                continue
            if type(control_model) is SD15T2IAdapter:
                control_hint = normalize_control_hint(
                    control_hint_source,
                    latent_height=xc.shape[2],
                    latent_width=xc.shape[3],
                    batch=batch,
                    device=device,
                    compute_dtype=xc.dtype,
                    expected_channels=1,
                    cache=self._control_hint_cache,
                )
                if len(contexts) > 1:
                    control_hint = control_hint.repeat(len(contexts), 1, 1, 1)
                cache_key = (
                    control_index,
                    batch,
                    len(contexts),
                    control_hint.shape[0],
                    control_hint.shape[2],
                    control_hint.shape[3],
                    control_hint.device,
                    control_hint.dtype,
                )
                residuals = self._t2i_residual_cache.get(cache_key)
                if residuals is None:
                    residuals = control_model(control_hint)
                    self._t2i_residual_cache[cache_key] = residuals
            else:
                control_hint = normalize_control_hint(
                    control_hint_source,
                    latent_height=xc.shape[2],
                    latent_width=xc.shape[3],
                    batch=batch,
                    device=device,
                    compute_dtype=xc.dtype,
                    expected_channels=(
                        control_model.config.hint_channels
                        if type(control_model) is SDXLControlNet
                        else 3
                    ),
                    cache=self._control_hint_cache,
                )
                if type(control_model) is SDXLControlLoRA:
                    if y is None:
                        raise DenoiseError("SDXL Control-LoRA requires ADM conditioning")
                    residuals = control_model(
                        xc,
                        control_hint,
                        timesteps.to(xc.dtype),
                        context,
                        y,
                    )
                elif type(control_model) is SDXLControlNetUnion:
                    if y is None:
                        raise DenoiseError("SDXL ControlNet Union requires ADM conditioning")
                    mode = self._control_modes[control_index]
                    residuals = control_model(
                        xc,
                        control_hint,
                        timesteps.to(xc.dtype),
                        context,
                        y,
                        None if mode is None else SD_CONTROL_MODE_INDEX[mode.token],
                    )
                elif type(control_model) is SDXLControlNet:
                    if y is None:
                        raise DenoiseError("SDXL ControlNet requires ADM conditioning")
                    residuals = control_model(
                        xc,
                        control_hint,
                        timesteps.to(xc.dtype),
                        context,
                        y,
                    )
                else:
                    residuals = control_model(
                        xc,
                        control_hint,
                        timesteps.to(xc.dtype),
                        context,
                    )
            sites = (
                SDXL_CONTROL_RESIDUAL_SITES
                if type(control_model) in (SDXLControlLoRA, SDXLControlNet, SDXLControlNetUnion)
                else SD15_CONTROL_RESIDUAL_SITES
            )
            current = SDControlResiduals(
                tuple(
                    _scale_control_residual(
                        value, control_gain, index, lane_ids, batch, effect_masks, sites
                    )
                    for index, value in enumerate(residuals.down)
                ),
                _scale_control_residual(
                    residuals.middle,
                    control_gain,
                    len(sites) - 1,
                    lane_ids,
                    batch,
                    effect_masks,
                    sites,
                ),
                residuals.down_channels,
                residuals.down_scales,
            )
            if control is None:
                control = current
            else:
                if (
                    control.down_channels != current.down_channels
                    or control.down_scales != current.down_scales
                ):
                    raise DenoiseError("control chain mixes incompatible residual site layouts")
                control = SDControlResiduals(
                    tuple(
                        previous + addition
                        for previous, addition in zip(control.down, current.down, strict=True)
                    ),
                    control.middle + current.middle,
                    control.down_channels,
                    control.down_scales,
                )
        if control is None:
            if ipadapter is not None:
                return self.model(
                    xc,
                    timesteps,
                    context=context,
                    y=y,
                    attention_guidance=attention_guidance,
                    ipadapter=ipadapter,
                ).float()
            return self.model(
                xc, timesteps, context=context, y=y, attention_guidance=attention_guidance
            ).float()
        if ipadapter is not None:
            return self.model(
                xc,
                timesteps,
                context=context,
                y=y,
                control=control,
                attention_guidance=attention_guidance,
                ipadapter=ipadapter,
            ).float()
        return self.model(
            xc,
            timesteps,
            context=context,
            y=y,
            control=control,
            attention_guidance=attention_guidance,
        ).float()

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        if self._conditioning is None:
            raise DenoiseError("no conditioning is bound to this evaluator")
        return self.evaluate_conditioning(x, sigma, self._conditioning)

    def evaluate_conditioning(
        self, x: torch.Tensor, sigma: float, condition: SDCondition
    ) -> torch.Tensor:
        return self.evaluate_conditioning_batch(x, sigma, (condition,))[0]

    def evaluate_conditioning_batch(
        self,
        x: torch.Tensor,
        sigma: float,
        conditions: tuple[SDCondition, ...],
        attention_guidance: AttentionGuidanceContext | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if not conditions or not self.batchable(conditions):
            raise DenoiseError("SD conditioning batch is empty or incompatible")
        parameterization = self.parameterization
        xc = _calculate_input(parameterization, sigma, x).to(self.compute_dtype)
        repeats = cross_attn_repeat([condition[0].shape[1] for condition in conditions])
        assert repeats is not None
        outputs = self._forward(
            xc,
            sigma,
            [condition[0] for condition in conditions],
            [condition[1] for condition in conditions],
            tuple(condition[2] for condition in conditions),
            repeats,
            attention_guidance,
        ).chunk(len(conditions))
        return tuple(_calculate_denoised(parameterization, sigma, output, x) for output in outputs)

    def evaluate_conditioning_batch_attention(
        self,
        x: torch.Tensor,
        sigma: float,
        conditions: tuple[SDCondition, ...],
        roles: tuple[GuidanceRole, ...],
        attention: tuple[AttentionGuidanceDescriptor[torch.Tensor], ...],
    ) -> tuple[torch.Tensor, ...]:
        """One fused forward whose attn1 outputs are rewritten by the
        attention-kind guidance descriptors. Lane i occupies batch rows
        [i*B, (i+1)*B) of the stacked forward, so the conditional and
        unconditional row ranges follow from the lane positions."""
        if len(roles) != len(conditions):
            raise DenoiseError("attention guidance roles do not match the conditioning batch")
        conditional = [
            index for index, role in enumerate(roles) if role is GuidanceRole.CONDITIONAL
        ]
        unconditional = [
            index for index, role in enumerate(roles) if role is GuidanceRole.UNCONDITIONAL
        ]
        if len(conditional) != 1 or len(unconditional) != 1:
            raise DenoiseError(
                "attention guidance requires exactly one conditional and one"
                " unconditional lane in the fused batch"
            )
        batch = x.shape[0]
        context = AttentionGuidanceContext(
            attention,
            (conditional[0] * batch, (conditional[0] + 1) * batch),
            (unconditional[0] * batch, (unconditional[0] + 1) * batch),
        )
        return self.evaluate_conditioning_batch(x, sigma, conditions, context)

    def _grouped_conditioning_forward(
        self,
        x: torch.Tensor,
        sigma: float,
        conditioning: Conditioning[torch.Tensor],
        adm: torch.Tensor | None,
    ) -> torch.Tensor:
        """One dormant B2.3a forward over already-grouped conditioning."""

        return self.evaluate_conditioning(
            x,
            sigma,
            self.prepare_conditioning(conditioning, adm=adm),
        )


__all__ = [
    "CROSS_ATTN_REPEAT_LIMIT",
    "SDXL_ADM_DEFAULT_SIZE",
    "SDXL_AESTHETIC_DEFAULT",
    "SDXL_NEGATIVE_AESTHETIC_DEFAULT",
    "SDDenoiser",
    "cross_attn_repeat",
    "encode_sdxl_adm",
    "encode_sdxl_refiner_adm",
    "inpaint_model_input",
]
