## Samplers and schedules

Dinkster has one sampling engine: the sampler provided by `dinkster_inference`.
SD1.5 and MiniMax H3 use that engine through the production worker boundary.

The native KSampler exposes the sampler and scheduler choices provided by the
pinned runtime. Layered window plans and RES4LYF samplers are unsupported.

SamplerCustom and SamplerCustomAdvanced combine native noise, conditioning,
guiders, sampler choices and sigma schedules, returning sampled and denoised
latents while preserving source samples, masks and metadata.

Align Your Steps, GITS, Optimal Steps, and Ideogram 4 scheduler nodes produce
native SIGMAS tensors for model-specific schedule selection.
Flux2Scheduler uses image token count and sampling steps to shift its sigma schedule.

Basic, CFG, dual-CFG, and dual-model CFG guiders produce native GUIDER values
for custom sampling. CFG Override patches CFG over an inclusive sampling
percentage range.

Sigma schedules can be entered, split, flipped, adjusted, extended and derived
from a model sampling percentage through the native ComfyUI sigma nodes.

CFGZeroStar, CFGNorm, APG, TCFG, Mahiro, and NAGuidance patch a model's CFG or
attention callbacks for the native Comfy guidance interfaces.
RenormCFG truncates CFG by timestep and bounds guided prediction norms while
preserving additional output channels.

Perp-Neg Guider combines positive, negative, and empty conditioning with
perpendicular negative guidance. The deprecated Perp-Neg model patch remains
available for existing workflows.

Native SIGMAS tensors cross worker boundaries as canonical tensor records,
preserving their dtype, shape, bytes, and fingerprint. Legacy schedule
settings remain supported.

Native Epsilon Scaling and Temporal Score Rescaling patch post-CFG noise
predictions. FreSca filters low- and high-frequency guidance independently.
Temporal Score Rescaling supports per-sample sigma tensors, leaving zero-sigma
samples unchanged while rescaling the other samples independently.

EasyCache and LazyCache patch a model for approximate denoiser-result reuse
during a configured portion of sampling.

Native scheduler nodes provide Basic, Karras, Exponential, Polyexponential,
Laplace, VP, beta sampling, and SD Turbo sigma schedules.

Runtime samplers and noise generators can connect nodes in their holding
worker without serializing callables or losing sampler options and seeds.
Portable sampling selections remain supported. Runtime handles cannot be
resolved in another worker or after their resident state is released.

RandomNoise, DisableNoise, and AddNoise provide seeded noise, zero noise, and
model-scaled noise while preserving latent metadata and the source samples.

Native sampler factory nodes provide KSampler selection, Euler ancestral,
DPM++, LMS, adaptive DPM, ER-SDE, SA-Solver, and SEEDS-2 runtime samplers.

LCM upscale, LCM noise controls, and Euler CFG++ sampler nodes run through the
pinned native sampler runtime. Euler CFG++ supports regular and alternative
ComfyUI callback behavior.

SenseNova Sampling Options sets a cloned model's flow shift while preserving
its configured noise scale.
