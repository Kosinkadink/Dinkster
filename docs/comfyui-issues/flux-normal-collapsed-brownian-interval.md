# Flux normal schedules can collapse the Brownian interval

At ComfyUI commit `15eb748b3ec5f8a0a2d470b7fb280e2d7579f916`,
ModelSamplingFlux with max_shift=1.15, base_shift=0.5, width=4096 and
height=4096 produces a four-step normal schedule of
`[1.0, 0.9999997019767761, 0.9999994039535522, 0.9999991059303284, 0.0]`.
The positive interval is smaller than torchsde's 1e-6 Brownian grid.

Executing ComfyUI's BrownianTreeNoiseSampler with this interval, CPU float32
noise shape `(1, 2, 2, 2)`, seed 23, and SNR-adjusted query sigmas raises
`RecursionError` inside torchsde 0.2.6 `_Interval._split`.
This does not require a checkpoint or denoiser execution.

`tools/gen_model_sampling_flux_goldens.py` reproduces the error against its
pinned checkout and records it rather than claiming a successful reference
noise draw. Schedule values remain covered exactly. Other recorded geometry
and scheduler combinations include executed Brownian draws; this boundary
case is not evidence of successful upstream SDE sampling.
