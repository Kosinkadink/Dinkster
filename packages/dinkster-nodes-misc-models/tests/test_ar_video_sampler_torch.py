from typing import Any

from dinkster_api.v1 import (
    initialize_inference_process,
)
from dinkster_misc_models.ar_video import SamplerARVideo


def _assert_sampler(sampler: Any, frames: int) -> None:
    initialize_inference_process()
    from dinkster_inference.k_diffusion.sampling import sample_ar_video
    from dinkster_inference.samplers import KSAMPLER

    assert type(sampler) is KSAMPLER
    assert sampler.sampler_function is sample_ar_video
    assert sampler.extra_options == {"num_frame_per_block": frames}
    assert sampler.inpaint_options == {}


def test_ar_video_sampler_retains_independent_frame_options() -> None:
    samplers = [SamplerARVideo.execute(num_frame_per_block=n)["SAMPLER"] for n in (1, 3, 64)]
    for sampler, frames in zip(samplers, (1, 3, 64), strict=True):
        _assert_sampler(sampler, frames)
    assert samplers[0] is not samplers[1]
