"""First-party seedvr model nodes."""

from .seedvr_conditioning import SeedVR2Conditioning
from .seedvr_postprocess import SeedVR2PostProcessing
from .seedvr_preprocess import SeedVR2Preprocess
from .seedvr_temporal import SeedVR2TemporalChunk, SeedVR2TemporalMerge

NODES = (
    SeedVR2Conditioning,
    SeedVR2PostProcessing,
    SeedVR2Preprocess,
    SeedVR2TemporalChunk,
    SeedVR2TemporalMerge,
)
