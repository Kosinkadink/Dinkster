## Samplers and schedules

Dinkster has one sampling engine: the sampler provided by `dinkster_inference`.
SD1.5 and MiniMax H3 use that engine through the production worker boundary.

The native KSampler exposes the sampler and scheduler choices provided by the
pinned runtime. Layered window plans and RES4LYF samplers are unsupported.
