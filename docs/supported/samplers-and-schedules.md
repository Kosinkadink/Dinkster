## Samplers and schedules

Dinkster has one sampling engine: the sampler provided by `dinkster_comfy`.
SD1.5 and MiniMax H3 use that engine through the production worker boundary.

The native KSampler accepts optional layered window plans declared over the
temporal, height, and width media roles. The stock static temporal context
schedule, spatial tiles, wrap-around windows, and explicit repeated or
non-contiguous index lists can be combined without exposing tensor dimensions
in the graph.

RES4LYF RK Beta is available through a typed sampler selection with independent
outer-step and substep noise controls. Other sampler and schedule surfaces need
their own parity evidence before native exposure.
