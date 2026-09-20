# Sampling execution

`sampling_execution` is the single owner of a custom sampling run. It admits
the latent, resolves the sampler request, builds the sigma schedule and
Brownian noise, compiles guidance, runs the solver, reports preview and step
callbacks, observes cancellation, applies the denoise mask, and returns the
adapted result. KSampler surfaces compose a `CustomSamplingRequest` and call
the same engine; they do not define another execution path.

## Family contract

A family registers `SamplingExecutionRegistration` and binds `sample_custom`
to `sampling_execution`. Registration contains data and adapters, not sampling
policy:

- A `SamplingLatentAdapter` validates and narrows the family's latent, noise,
  conditioning, guidance, and mask values. Its `finish` method restores the
  family's public result shape. `SingleStreamLatentAdapter` is the standard
  tensor implementation.
- A `SamplingDenoiserAdapter` prepares conditioning and evaluates one or a
  compatible batch of conditioning lanes. It owns only the network call and
  model-specific conditioning arithmetic. Its stable evaluator identity lets
  the guidance engine reason about batching and distribution.
- A `SamplingDenoiserExecution` wraps that evaluator with explicit
  denoiser-coupled solver options, sampling bounds, and model-state step hook.
  The engine remains the only caller of the solver and hook; optional behavior
  is declared on this shape rather than discovered through adapter attributes.
- Device and compute-dtype resolvers select where and how the model call runs.
  The `flow` flag declares schedule parameterization. The runtime's
  `supports_denoised_capture` capability declares whether the result can carry
  a denoised preview.
- Invocation adapter context carries distilled guidance, inpaint data, context
  windows, and an immutable map of family-specific model-call options. These
  values may affect latent adaptation or the denoiser, never engine policy.
  Each family must reject unknown option names rather than silently ignoring
  them. Per-call cancellation uses the sampling environment and is not an
  adapter option.

A family may validate structural latents, pack or unpack streams, materialize
typed conditioning, select device and dtype, and adapt the final result. It may
not resolve samplers, construct schedules or noise, implement CFG, call the
solver, apply masks, own callbacks, or poll cancellation. Those are engine
invariants so every sampler surface observes identical behavior.

## Runtime adapters

Every diffusion runtime executes through `sampling_execution`. Flux2, Anima,
Chroma, Ideogram4, Lumina2, Qwen Image, SeedVR2, Z-Image, Flux, SD, and MiniMax
Music 3 use single-stream adapters. Wan21, LTXV, LTXAV, TRELLIS.2, TripoSplat,
and MiniMax H3 adapt structural latent shapes at the same seam.

Wan21 maps structural video streams, typed conditioning, context windows,
inpaint data, and masks into its adapter. LTXV, LTXAV, TRELLIS.2, and
TripoSplat use the same structural contract. MiniMax H3 packs video and audio
streams and finalizes audio scaling and capture through its latent adapter.
Distributed and windowed model evaluation stays inside denoiser adapters;
solver, schedule, noise, masking, cancellation, observer, and callback
execution stays in the engine.

Autoregressive or windowed model evaluation is a denoiser implementation, not
a second sampling run. A migration is complete only when KSampler and direct
SamplerCustom execution are bit-identical for the same request, noise, and
inputs, without changing numerical comparisons.
