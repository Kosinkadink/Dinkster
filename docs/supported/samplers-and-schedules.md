## Samplers and schedules

62 builtin samplers available to generation nodes:

ar_video, euler, euler_cfg_pp, euler_ancestral, euler_ancestral_cfg_pp, heun, heunpp2,
exp_heun_2_x0, exp_heun_2_x0_sde, dpm_2, dpm_2_ancestral, lms, dpm_fast,
dpm_adaptive, dpmpp_2s_ancestral, dpmpp_2s_ancestral_cfg_pp, dpmpp_sde,
dpmpp_sde_gpu, dpmpp_2m, dpmpp_2m_cfg_pp, dpmpp_2m_sde, dpmpp_2m_sde_gpu,
dpmpp_2m_sde_heun, dpmpp_2m_sde_heun_gpu, dpmpp_3m_sde, dpmpp_3m_sde_gpu,
ddpm, lcm, ipndm, ipndm_v, deis, res_multistep, res_multistep_cfg_pp,
res_multistep_ancestral, res_multistep_ancestral_cfg_pp,
gradient_estimation, gradient_estimation_cfg_pp, er_sde, seeds_2, seeds_3,
sa_solver, sa_solver_pece, ddim, uni_pc, uni_pc_bh2, plus the 17 RES4LYF
beta RK engine samplers (@ 26036f64): res_2m, res_3m, res_2s, res_3s,
res_5s, res_6s, their _ode variants, deis_2m, deis_3m, deis_2m_ode,
deis_3m_ode, and rk_beta (configurable rk_type/eta/eta_substep).

RES4LYF's retired legacy-module sampler name "rk" resolves to res_2m_ode
(the name only ever ran its library defaults - rk_type res_2m, eta 0.0,
deterministic - and res_2m_ode is the beta engine's deterministic res_2m;
outputs differ from the retired legacy engine). The sibling name
"legacy_rk" is intentionally unsupported: its engine has no beta
equivalent, so workflows using it are refused as unknown samplers.

The RES4LYF RK samplers reproduce the reference's SEEDED noise path: for
run seed s the outer noise stream is seeded s + 1 and the substep stream
s + 10001, matching RES4LYF's own workflow-seed rewrite and its
noise_seed + MAX_STEPS substep derivation. The reference's unseeded
direct-wrapper default (torch.initial_seed() + 1, process-global state) is
unreproducible by construction and out of parity scope.

The `ar_video` sampler executes Wan 2.1 CausalAR through SamplerCustom. It
repeats the selected sigma schedule for each temporal block and exposes a
1-64 frame block-size option through Sampler AR Video. Inputs may be plain
video latents or a single video stream, with optional initial latent frames.
Plain inputs retain plain outputs and their latent-scale metadata.

ER-SDE exposes the current ComfyUI ER-SDE, reverse-time SDE, and ODE modes,
with eta 0-10, s_noise 0-100, and integration stages 1-3.

Eleven sigma schedules: simple, sgm_uniform, karras, exponential,
ddim_uniform, beta, normal, linear_quadratic, kl_optimal, plus the two
schedulers RES4LYF (@ 26036f64) registers globally: bong_tangent and
beta57 (beta spacing with alpha=0.5/beta=0.7).

ComfyUI-shaped KSamplerSelect, BasicScheduler, Basic Guider, CFG Guider,
Dual CFG Guider, Scheduled CFG Guider, Perp-Neg Guider, Disable CFG 1 Optimization,
RandomNoise, DisableNoise, AddNoise, Sampler SA-Solver, SamplerCustom, and
SamplerCustomAdvanced nodes provide SD1.5, SDXL, Flux,
Lumina2, Chroma, Chroma Radiance, Wan 2.1/2.2, Z-Image, MiniMax H3, and MiniMax Music 3 sampling
over exact sigma sequences. Scheduled CFG Guider also accepts Inspire Pack's
ScheduledCFGGuider workflow node and applies its linear, logarithmic,
exponential, or cosine CFG schedule to the supplied sigmas. SeedVR2 restoration
and upscaling accept exact caller-supplied discrete-flow sigma sequences through
the same nodes. Perp-Neg Guider
(perpendicular negative guidance over positive, negative, and
empty-prompt lanes) executes on SD1.5 and SDXL.
Dual CFG Guider supports regular and nested three-lane guidance. Sampler
SA-Solver exposes predictor and corrector orders, PECE, stochastic interval,
and noise controls. AddNoise applies a supplied noise source to a dense latent
over the selected sigma span while retaining latent metadata.
Each accepts the matching ComfyUI node ID as an alias and keeps the same
workflow-facing socket IDs and order. The custom path accepts plain and inpaint
conditioning where the model supports it. SamplerCustom and SamplerCustomAdvanced
also expose optional denoise mask, positive and negative inpaint conditioning,
noise indices, and context-window inputs. They reject conflicts with equivalent
latent, conditioning, or model metadata. The custom path also accepts standard SD1.5
IP-Adapter and maintained Qwen Image control applications. Component applications
are accepted when the model consumes their declared inputs. Other overlays and
controls are not supported there. SD1.5 IP-Adapter
with scheduled prompt or patch sampling is not supported.
Disable CFG 1 Optimization forces the negative-conditioning model call at CFG 1
and uses the standard CFG combination for both KSampler and custom sampling.
Compatible conditioning lanes are fused according to current free device
memory by default. KSampler and custom-sampling guidance expose overrides to
evaluate lanes separately or cap the number fused in one model call. Dense
TRELLIS.2, Qwen Image, and Ideogram 4 participate; sparse TRELLIS.2 and MiniMax
H3 remain separate because their authenticated packed layouts cannot be
lane-expanded without changing identity.
TripoSplat structural latents can be sampled through both KSampler and custom sampling.
KSampler and custom sampling install temporal or spatial context windows through
the shared sampling pipeline. Real-layout execution coverage currently includes
SD 1.5, SDXL, SDXL Refiner, MiniMax H3, LTX-Video, and Wan 2.1/2.2. Other
families accept context windows for exploratory use but remain untested until
later per-family validation finds and fixes any layout-specific issues. Flux2
and Wan CausalAR refuse context windows; Wan base profiles also refuse structural
conditioning that cannot be sliced without changing its meaning.
The sampling runtime APIs accept denoise masks for dense image, video, audio,
and multi-stream latents. Sparse sampling accepts masks on the same sparse
support. Wan CausalAR does not accept denoise masks. Distributed execution does
not require a numerical receipt or registered hardware. Sequence and
window-distributed execution require compatible geometry; missing measurement
evidence produces a diagnostic.
Scheduled SD and Flux sampling support prompt ranges, conditioning regions,
and patch schedules. Scheduled SD sampling can combine denoise masks with
temporal or spatial context windows.
Wan CausalAR, sampling timelines, LazyCache, and EasyCache run their complete
sampling stage on rank 0 in distributed jobs and broadcast the final result to
peers. Callbacks and cancellation are owned by rank 0 for those stages.
ModelSamplingFlux patches Flux dev and schnell with the geometry-dependent
exponential flow schedule. The patch applies to model sigma queries,
KSampler, KSamplerAdvanced, SamplerCustom, and SamplerCustomAdvanced without
changing the source model or its weights. Finite zero and negative computed
shifts are retained, not replaced with rational SD3 shifts.
Sampling-shift overrides are forwarded to models that support them. Other
models report an unsupported-shift error rather than ignoring the value.
Complete Flux2 and Lumina2 checkpoints honor explicit shifts in KSampler,
as do their independently loaded diffusion components.
Impact Pack's canonical one-region Regional Sampler graph imports as native
custom-sampling orchestration for mask-capable SD1.5 and SDXL runtimes. It
supports separate base and region BASIC_PIPE inputs, samplers, schedulers, and
CFG values; base-only steps; denoise; overlap growth; and optional restoration
outside the regional mask. Multi-region/CombineRegionalPrompts graphs,
variation noise, second-seed modes, recovery/additional samplers, custom
provider samplers or scheduler functions, and sigma factors other than 1 are
not supported by this translation.
Custom sampling and KSampler use the same registered distributed plans,
callbacks, masks, samplers, guidance transforms, and controls. Supported
combinations are listed under
[Server API and execution](server-api-and-execution.md#server-api-and-execution).
ComfyUI-shaped sampler option nodes cover Euler ancestral, Euler ancestral
CFG++, DPM++ 2S ancestral, DPM++ SDE, DPM++ 2M SDE, DPM++ 3M SDE, LMS,
DPM adaptive, ER-SDE, and SEEDS 2, including CPU/GPU SDE noise selection.
BetaSamplingScheduler, SDTurboScheduler, and SamplingPercentToSigma provide
the matching model-dependent ComfyUI sigma operations. ManualSigmas,
SplitSigmas, SplitSigmasDenoise, FlipSigmas, SetFirstSigma, and
ExtendIntermediateSigmas provide the matching ComfyUI sigma construction and
transformation operations.
KarrasScheduler, ExponentialScheduler, PolyexponentialScheduler,
LaplaceScheduler, and VPScheduler provide the matching standalone ComfyUI
sigma schedules. AlignYourStepsScheduler (SD1/SDXL/SVD), GITSScheduler, and
OptimalStepsScheduler (FLUX/Wan/Chroma) provide the matching table-driven
ComfyUI sigma schedules.

Sampling math is validated against executed ComfyUI goldens (pinned
upstream commit), including full-pipeline SD and Flux cases.
