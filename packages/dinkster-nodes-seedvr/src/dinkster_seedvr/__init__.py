"""First-party seedvr model nodes."""

from dinkster_api.v1 import inference_nodes
from dinkster_api.v1 import register_inference_types as register_types

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

__all__ = ["NODES", "register_types"]

inference_nodes(NODES)
