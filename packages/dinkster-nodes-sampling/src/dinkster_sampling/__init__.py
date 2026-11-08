"""First-party sampling nodes."""

from dinkster_api.v1 import inference_nodes

from .advanced_samplers import (
    SamplerEulerCFGpp,
    SamplerLCM,
    SamplerLCMUpscale,
)
from .attention_guidance import (
    PerturbedAttentionGuidance,
    SelfAttentionGuidance,
    SkipLayerGuidanceDiT,
    SkipLayerGuidanceDiTSimple,
)
from .cfg_guidance import (
    APG,
    TCFG,
    CFGNorm,
    CFGZeroStar,
    Mahiro,
    NAGuidance,
    RenormCFG,
)
from .cfg_guiders import (
    BasicGuider,
    CFGGuider,
    CFGOverride,
    DualCFGGuider,
    DualModelGuider,
)
from .custom_sampling import (
    SamplerCustom,
    SamplerCustomAdvanced,
)
from .model_specific_schedulers import (
    AlignYourStepsScheduler,
    GITSScheduler,
    OptimalStepsScheduler,
)
from .noise import (
    AddNoise,
    DisableNoise,
    RandomNoise,
)
from .noise_guidance import (
    EpsilonScaling,
    FreSca,
    TemporalScoreRescaling,
)
from .perpneg import (
    PerpNeg,
    PerpNegGuider,
)
from .sampler_factories import (
    KSamplerSelect,
    SamplerDPMAdaptative,
    SamplerDPMPP_2M_SDE,
    SamplerDPMPP_2S_Ancestral,
    SamplerDPMPP_3M_SDE,
    SamplerDPMPP_SDE,
    SamplerER_SDE,
    SamplerEulerAncestral,
    SamplerEulerAncestralCFGPP,
    SamplerLMS,
    SamplerSASolver,
    SamplerSEEDS2,
)
from .sigma_operations import (
    ExtendIntermediateSigmas,
    FlipSigmas,
    ManualSigmas,
    SamplingPercentToSigma,
    SetFirstSigma,
    SplitSigmas,
    SplitSigmasDenoise,
)
from .sigma_schedulers import (
    BasicScheduler,
    BetaSamplingScheduler,
    ExponentialScheduler,
    KarrasScheduler,
    LaplaceScheduler,
    PolyexponentialScheduler,
    SDTurboScheduler,
    VPScheduler,
)
from .standard_sampling import (
    KSampler,
    KSamplerAdvanced,
)
from .video_guidance import (
    VideoLinearCFGGuidance,
    VideoTriangleCFGGuidance,
)

NODES = (
    SelfAttentionGuidance,
    PerturbedAttentionGuidance,
    SkipLayerGuidanceDiT,
    SkipLayerGuidanceDiTSimple,
    KSampler,
    KSamplerAdvanced,
    SamplerCustom,
    SamplerCustomAdvanced,
    AlignYourStepsScheduler,
    GITSScheduler,
    OptimalStepsScheduler,
    CFGGuider,
    BasicGuider,
    DualCFGGuider,
    DualModelGuider,
    CFGOverride,
    CFGZeroStar,
    CFGNorm,
    APG,
    TCFG,
    Mahiro,
    NAGuidance,
    RenormCFG,
    PerpNeg,
    PerpNegGuider,
    VideoLinearCFGGuidance,
    VideoTriangleCFGGuidance,
    EpsilonScaling,
    TemporalScoreRescaling,
    FreSca,
    RandomNoise,
    DisableNoise,
    AddNoise,
    SplitSigmas,
    SplitSigmasDenoise,
    FlipSigmas,
    SetFirstSigma,
    ExtendIntermediateSigmas,
    SamplingPercentToSigma,
    ManualSigmas,
    BasicScheduler,
    KarrasScheduler,
    ExponentialScheduler,
    PolyexponentialScheduler,
    LaplaceScheduler,
    VPScheduler,
    BetaSamplingScheduler,
    SDTurboScheduler,
    KSamplerSelect,
    SamplerEulerAncestral,
    SamplerEulerAncestralCFGPP,
    SamplerLMS,
    SamplerDPMPP_3M_SDE,
    SamplerDPMPP_2M_SDE,
    SamplerDPMPP_SDE,
    SamplerDPMPP_2S_Ancestral,
    SamplerDPMAdaptative,
    SamplerER_SDE,
    SamplerSASolver,
    SamplerSEEDS2,
    SamplerLCMUpscale,
    SamplerLCM,
    SamplerEulerCFGpp,
)

inference_nodes(NODES)
