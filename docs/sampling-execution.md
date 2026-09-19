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
- Device and compute-dtype resolvers select where and how the model call runs.
  The `flow` flag declares schedule parameterization, and the capture flag
  declares whether the result can carry a denoised preview.
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

## Migration order

Migration proceeds from the smallest adapter surface to the largest so each
new latent shape extends the registration contract without changing engine
semantics:

1. Flux2 uses `SingleStreamLatentAdapter`, shifted FLOW schedules, module
   placement, and its existing distilled-guidance-aware denoiser.
2. The other single-stream runtimes follow: Anima, Chroma, Ideogram4,
   Lumina2, Qwen Image, SeedVR2, Z-Image, and the wiring-owned SD runtime.
3. Wan21 introduces a structural video latent adapter. Its typed conditioning,
   context windows, inpaint data, and masks remain denoiser or latent-adapter
   inputs; its loop, cancellation, observers, and schedule move to the engine.
4. LTXV, LTXAV, TRELLIS.2, and TripoSplat follow the structural-latent contract.
5. MiniMax H3 introduces packed video/audio streams and finalization for audio
   scaling and capture. Distributed model evaluation stays inside its denoiser
   adapter; solver and callback execution stay in the engine. MiniMax Music 3
   then reuses that multi-stream shape.

Autoregressive or windowed model evaluation is a denoiser implementation, not
a second sampling run. A migration is complete only when KSampler and direct
SamplerCustom execution are bit-identical for the same request, noise, and
inputs, without changing numerical comparisons.
