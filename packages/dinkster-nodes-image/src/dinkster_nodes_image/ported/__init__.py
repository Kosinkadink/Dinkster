"""First-party ComfyUI-origin image nodes."""

from .bbox_crop import (
    CropByBBoxes,
)
from .compare import NativeImageCompare
from .compositing import (
    JoinImageWithAlpha,
    PorterDuffImageComposite,
    SplitImageWithAlpha,
)
from .glsl import NativeGlslShader
from .image_color_conversion import (
    ImageColorSpace,
    ImageSDRToHDR,
)
from .image_core import (
    EmptyImage,
    ImageBatch,
    ImageInvert,
    ImagePadForOutpaint,
    ImageScale,
    ImageScaleBy,
)
from .image_crop import (
    ImageCropToMask,
    ImageCropV2,
)
from .image_dimensions import (
    GetImageSize,
)
from .image_file_tools import (
    LoadImage,
    LoadImageMask,
    LoadImageOutput,
    Painter,
    PreviewImage,
    SaveAnimatedPNG,
    SaveAnimatedWEBP,
    SaveImage,
)
from .image_filters import (
    Canny,
    ImageBlend,
    ImageBlur,
    ImageQuantize,
    ImageScaleToTotalPixels,
    ImageSharpen,
)
from .image_histogram import (
    ImageHistogram,
)
from .image_layers import (
    AddLayer,
)
from .image_tiles import (
    ImageMergeTileList,
    SplitImageToTileList,
)
from .image_transform import (
    ImageAddNoise,
    ImageCrop,
    ImageFlip,
    ImageFromBatch,
    ImageRotate,
    ImageScaleToMaxDimension,
    ImageStitch,
    PadImageNode,
    RepeatImageBatch,
    ResizeAndPadImage,
)
from .mask import (
    CropMask,
    FeatherMask,
    GrowMask,
    ImageColorToMask,
    ImageCompositeMasked,
    ImageToMask,
    InvertMask,
    MaskComposite,
    MaskToImage,
    SolidMask,
    ThresholdMask,
)
from .mask_preview import (
    MaskPreview,
)
from .morphology import (
    ImageRGBToYUV,
    ImageYUVToRGB,
    Morphology,
)
from .post_processing import (
    BatchImagesNode,
    BatchMasksNode,
    ColorTransfer,
    ResizeImageMaskNode,
)
from .save_image_advanced import (
    SaveImageAdvanced,
)
from .svg_save import (
    SaveSVGNode,
)
from .text_overlay import (
    TextOverlay,
)

NODES = (
    CropByBBoxes,
    PorterDuffImageComposite,
    SplitImageWithAlpha,
    JoinImageWithAlpha,
    ImageSDRToHDR,
    ImageColorSpace,
    ImageScale,
    ImageScaleBy,
    ImageInvert,
    ImageBatch,
    EmptyImage,
    ImagePadForOutpaint,
    ImageCropToMask,
    ImageCropV2,
    GetImageSize,
    SaveImage,
    PreviewImage,
    SaveAnimatedWEBP,
    SaveAnimatedPNG,
    Painter,
    LoadImage,
    LoadImageMask,
    LoadImageOutput,
    ImageBlend,
    ImageBlur,
    ImageQuantize,
    ImageSharpen,
    ImageScaleToTotalPixels,
    Canny,
    ImageHistogram,
    AddLayer,
    SplitImageToTileList,
    ImageMergeTileList,
    ImageCrop,
    RepeatImageBatch,
    ImageFromBatch,
    ImageAddNoise,
    ImageStitch,
    ResizeAndPadImage,
    ImageRotate,
    ImageFlip,
    ImageScaleToMaxDimension,
    PadImageNode,
    ImageCompositeMasked,
    MaskToImage,
    ImageToMask,
    ImageColorToMask,
    SolidMask,
    InvertMask,
    CropMask,
    MaskComposite,
    FeatherMask,
    GrowMask,
    ThresholdMask,
    MaskPreview,
    Morphology,
    ImageRGBToYUV,
    ImageYUVToRGB,
    ResizeImageMaskNode,
    BatchImagesNode,
    BatchMasksNode,
    ColorTransfer,
    SaveImageAdvanced,
    SaveSVGNode,
    TextOverlay,
    NativeImageCompare,
    NativeGlslShader,
)

__all__ = ["NODES"]
