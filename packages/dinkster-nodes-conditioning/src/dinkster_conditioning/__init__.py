"""First-party conditioning nodes."""

from dinkster_api.v1 import inference_nodes

from .audio_encoder import (
    AudioEncoderEncode,
    AudioEncoderLoader,
)
from .clip import (
    CLIPSetLastLayer,
    CLIPTextEncode,
    CLIPVisionEncode,
)
from .clip_model_specific import (
    CLIPTextEncodeControlnet,
    T5TokenizerOptions,
)
from .conditioning import (
    ConditioningAverage,
    ConditioningCombine,
    ConditioningConcat,
    ConditioningMultiply,
    ConditioningSetArea,
    ConditioningSetAreaPercentage,
    ConditioningSetAreaPercentageVideo,
    ConditioningSetAreaStrength,
    ConditioningSetMask,
    ConditioningSetTimestepRange,
    ConditioningZeroOut,
)
from .conditioning_files import (
    ConditioningLoader,
    SaveConditioning,
)
from .controlnet import (
    ControlNetApply,
    ControlNetApplyAdvanced,
    ControlNetInpaintingAliMamaApply,
    SetUnionControlNetType,
)
from .inpaint import (
    InpaintModelConditioning,
)
from .pid_conditioning import (
    PiDConditioning,
)
from .reference_conditioning import (
    ReferenceLatent,
    ReferenceMultiLatent,
)
from .style_model import (
    StyleModelApply,
)

NODES = (
    AudioEncoderLoader,
    AudioEncoderEncode,
    CLIPTextEncode,
    CLIPSetLastLayer,
    CLIPVisionEncode,
    StyleModelApply,
    CLIPTextEncodeControlnet,
    T5TokenizerOptions,
    ConditioningAverage,
    ConditioningCombine,
    ConditioningConcat,
    ConditioningMultiply,
    ConditioningSetArea,
    ConditioningSetAreaPercentage,
    ConditioningSetAreaPercentageVideo,
    ConditioningSetAreaStrength,
    ConditioningSetMask,
    ConditioningZeroOut,
    ConditioningSetTimestepRange,
    PiDConditioning,
    ReferenceLatent,
    ReferenceMultiLatent,
    InpaintModelConditioning,
    ControlNetApply,
    ControlNetApplyAdvanced,
    SetUnionControlNetType,
    ControlNetInpaintingAliMamaApply,
    ConditioningLoader,
    SaveConditioning,
)

inference_nodes(NODES)
