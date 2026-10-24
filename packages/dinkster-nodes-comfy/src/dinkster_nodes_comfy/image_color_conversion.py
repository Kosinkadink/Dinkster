"""Native ComfyUI image color conversion nodes."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from dinkster_api.v1 import (
    CORE_COMBO,
    CORE_FLOAT,
    IMAGE_TYPE,
    ComboOption,
    ComboWidget,
    DynamicComboOption,
    DynamicComboSpec,
    InputSpec,
    Node,
    NodeSchema,
    NumberWidget,
    OutputSpec,
    TypeExpr,
    materialize_image,
)

_COMBO = TypeExpr.concrete(CORE_COMBO)
_FLOAT = TypeExpr.concrete(CORE_FLOAT)
_IMAGE = TypeExpr.concrete(IMAGE_TYPE)
_HLG_A = 0.17883277
_HLG_B = 0.28466892
_HLG_C = 0.55991072928
_REC709_TO_REC2020 = (
    (0.6274038959346991, 0.3292830383778837, 0.0433130656874172),
    (0.0690972893582320, 0.9195403950754587, 0.0113623155663092),
    (0.0163914388751503, 0.0880133078772259, 0.8955952532476238),
)
_REC2020_TO_REC709 = (
    (1.6604910021084338, -0.5876411387885494, -0.0728498633198846),
    (-0.1245504745215905, 1.1328998971259600, -0.0083494226043695),
    (-0.0181507633549053, -0.1005788980080076, 1.1187296613629125),
)
_ACESCG_TO_REC2020 = (
    (1.0258247476660107, -0.0200531908382148, -0.0057715568277955),
    (-0.0022343695199762, 1.0045865018884792, -0.0023521323685036),
    (-0.0050133514680893, -0.0252900718107852, 1.0303034232788744),
)
_REC2020_TO_ACESCG = (
    (0.9748949779244186, 0.0195991086370050, 0.0055059134385761),
    (0.0021795627977041, 0.9955354688932213, 0.0022849683090752),
    (0.0047972396837727, 0.0245320166345895, 0.9706707436816380),
)
_REC709_LUMA = (0.2126390058715103, 0.7151686787677559, 0.0721923153607337)
_REC2020_LUMA = (0.2627, 0.6780, 0.0593)
_PQ_M1, _PQ_M2 = 2610 / 16384, 2523 / 32
_PQ_C1, _PQ_C2, _PQ_C3 = 3424 / 4096, 2413 / 128, 2392 / 128
_SDR_WHITE_NITS = 203.0
_HLG_PEAK_NITS = 1000.0
_HLG_GAMMA = 1.2
_LOGC3_A, _LOGC3_B = 5.555556, 0.052272
_LOGC3_C, _LOGC3_D = 0.247190, 0.385537
_LOGC3_E, _LOGC3_F = 5.367655, 0.092809
_LOGC3_CUT = 0.010591
_ACESCCT_SLOPE, _ACESCCT_OFFSET = 10.5402377416545, 0.0729055341958355
_ACESCCT_LOG_SCALE, _ACESCCT_LOG_OFFSET = 17.52, 9.72
_ACESCCT_CUT, _ACESCCT_LOG_CUT = 0.0078125, 0.155251141552511
_LOTTES_HDR_PARAMETERS = (
    -0.1517456472,
    -1.465687394,
    -0.2663205862,
    -1.11367178,
    -2.938027859,
    -2.948823929,
    -2.97376132,
    -2.978893757,
    -2.991618395,
    -2.956256151,
    -2.810376167,
    -2.666443586,
    -2.584222317,
    -2.532898188,
    -2.458465099,
    -2.309132099,
    -2.216364861,
    -2.167824268,
    -2.096168995,
    -2.062468767,
    -2.130614519,
    0.5,
)
_COLOR_SPACES = ("sRGB", "HDR", "HDR ACEScct", "HDR LogC3", "HDR PQ", "linear", "linear ACEScg")


def _combo(name: str, options: tuple[str, ...], default: str) -> InputSpec:
    return InputSpec(
        name,
        _COMBO,
        default=default,
        widget=ComboWidget(tuple(ComboOption(option) for option in options)),
    )


def _srgb_to_linear(value: Any, torch: Any) -> Any:
    low = value / 12.92
    high = ((value.clamp(min=0.0) + 0.055) / 1.055) ** 2.4
    return torch.where(value <= 0.04045, low, high)


def _convert_primaries(rgb: Any, matrix: tuple[tuple[float, float, float], ...], torch: Any) -> Any:
    red, green, blue = rgb.unbind(dim=-1)
    return torch.stack([red * row[0] + green * row[1] + blue * row[2] for row in matrix], dim=-1)


def _luminance(rgb: Any, weights: tuple[float, float, float]) -> Any:
    return (rgb * rgb.new_tensor(weights)).sum(dim=-1, keepdim=True)


def _tone_map(rgb: Any, weights: tuple[float, float, float], torch: Any) -> Any:
    luminance = _luminance(rgb, weights).clamp_min(0.0)
    peak = luminance.amax().clamp_min(1.0)
    scale = (1.0 + luminance / peak.square()) / (1.0 + luminance)
    return rgb * torch.where(peak > 1.0001, scale, 1.0)


def _compress_gamut(rgb: Any, weights: tuple[float, float, float], torch: Any) -> Any:
    luminance = _luminance(rgb, weights).clamp(0.0, 1.0)
    chroma = rgb - luminance
    minimum, maximum = rgb.amin(dim=-1, keepdim=True), rgb.amax(dim=-1, keepdim=True)
    upper = (1.0 - luminance) / (maximum - luminance).clamp_min(torch.finfo(rgb.dtype).tiny)
    lower = luminance / (luminance - minimum).clamp_min(torch.finfo(rgb.dtype).tiny)
    saturation = torch.minimum(upper, lower).clamp(0.0, 1.0)
    return torch.where(
        (minimum >= -1e-5) & (maximum <= 1.00001), rgb, torch.addcmul(luminance, chroma, saturation)
    ).clamp(0.0, 1.0)


def _sdr_to_hdr_lottes(rgb: Any, target_luminance: float | None, torch: Any) -> Any:
    rgb = _srgb_to_linear(rgb.clamp(0, 1), torch)
    parameters = rgb.new_tensor(_LOTTES_HDR_PARAMETERS)
    luminance = _luminance(rgb, (0.2126, 0.7152, 0.0722))
    safe_luminance = luminance.clamp_min(1e-6)
    a, d, peak, c = (
        0.7 + 0.6 * parameters[0].sigmoid(),
        0.5 + 2 * parameters[1].sigmoid(),
        1 + 7 * parameters[2].sigmoid(),
        0.5 + 2 * parameters[3].sigmoid(),
    )
    expanded = safe_luminance.pow(a) / ((1 / peak - c) * safe_luminance.pow(a * d) + c).clamp_min(
        1e-5
    )
    rational_gain = (expanded / safe_luminance).clamp_min(1e-3).log2().clamp(-1, 3.5)
    knots = torch.nn.functional.softplus(parameters[4:21]).cumsum(0)
    position = luminance.clamp(0, 1).sqrt() * 16
    index = position.long().clamp_max(15)
    curve_gain = knots[index] + (position - index) * (knots[index + 1] - knots[index])
    blend = parameters[21].clamp(0, 1)
    hdr = (
        _convert_primaries(rgb, _REC709_TO_REC2020, torch)
        * (blend * rational_gain + (1 - blend) * curve_gain).exp2()
    )
    if target_luminance is not None:
        source_white = (blend * peak.log2().clamp(-1, 3.5) + (1 - blend) * knots[-1]).exp2()
        target_white = target_luminance / _SDR_WHITE_NITS
        hdr = hdr / (
            1 + (1 / target_white - 1 / source_white) * _luminance(hdr, _REC2020_LUMA)
        ).clamp_min(1e-6)
    peak_nits = max(_HLG_PEAK_NITS, target_luminance or 0.0)
    gamma = 1.2 + 0.42 * math.log10(peak_nits / 1000.0)
    display = hdr.clamp_min(0) * (_SDR_WHITE_NITS / peak_nits)
    scene = (
        display * _luminance(display, _REC2020_LUMA).clamp_min(1e-12).pow(1 / gamma - 1)
    ).clamp(0, 1)
    return torch.where(
        scene <= 1 / 12,
        (3 * scene).sqrt(),
        _HLG_A * (12 * scene - _HLG_B).clamp_min(1e-12).log() + _HLG_C,
    )


class ImageSDRToHDR(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ImageSDRToHDR",
            display_name="Convert SDR to HDR",
            category="image/color",
            description=(
                "Expand sRGB brightness into Rec.2020 HLG HDR. Save the result with the HDR "
                "color space selected. Alpha is preserved."
            ),
            inputs=(InputSpec("image", _IMAGE, doc="SDR image(s) in sRGB."),),
            combos=(
                DynamicComboSpec(
                    "algorithm",
                    (
                        DynamicComboOption(
                            "lottes",
                            (
                                InputSpec(
                                    "target_luminance",
                                    _FLOAT,
                                    default=650.0,
                                    widget=NumberWidget(min=0.0, max=10000.0, step=1.0),
                                    doc=(
                                        "Peak-white target in nits. 0 keeps the calibrated "
                                        "rendering. Dark "
                                        "images are not normalized to this peak."
                                    ),
                                ),
                            ),
                        ),
                    ),
                    doc=(
                        "Lottes-style rational expansion blended with a calibrated monotonic "
                        "log-gain curve."
                    ),
                ),
            ),
            outputs=(OutputSpec("IMAGE", _IMAGE),),
            input_order=("image", "algorithm"),
        )

    @classmethod
    def execute(cls, *, image: Any, algorithm: Mapping[str, Any]) -> Mapping[str, object]:
        if algorithm["algorithm"] != "lottes":
            raise ValueError(f"Unsupported HDR algorithm: {algorithm['algorithm']}")
        torch = importlib.import_module("torch")
        image = materialize_image(image)
        if image.shape[-1] == 1:
            image = image.expand(*image.shape[:-1], 3)
        rgb = _sdr_to_hdr_lottes(
            image[..., :3].to(dtype=torch.float32), algorithm["target_luminance"] or None, torch
        )
        if image.shape[-1] == 4:
            rgb = torch.cat((rgb, image[..., 3:]), dim=-1)
        return cls.outputs(IMAGE=rgb.to(dtype=image.dtype, device=image.device))


class ImageColorSpace(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ImageColorSpace",
            display_name="Convert Image Color Space",
            category="image/color",
            description=(
                "Convert sRGB, linear Rec.709, HDR (Rec.2020 HLG), HDR PQ (Rec.2020 PQ), "
                "HDR LogC3, linear ACEScg, and HDR ACEScct. Straight alpha is not "
                "color-transformed."
            ),
            inputs=(
                InputSpec("image", _IMAGE),
                _combo("source", _COLOR_SPACES, "sRGB"),
                _combo("destination", _COLOR_SPACES, "sRGB"),
            ),
            outputs=(OutputSpec("IMAGE", _IMAGE),),
        )

    @classmethod
    def execute(cls, *, image: Any, source: str, destination: str) -> Mapping[str, object]:
        torch = importlib.import_module("torch")
        image = materialize_image(image)
        if source == destination:
            return cls.outputs(IMAGE=image)
        rgb = image[..., :3].float()
        if source == "sRGB":
            rgb = (
                _convert_primaries(_srgb_to_linear(rgb, torch), _REC709_TO_REC2020, torch)
                * _SDR_WHITE_NITS
            )
        elif source == "linear":
            rgb = _convert_primaries(rgb, _REC709_TO_REC2020, torch) * _SDR_WHITE_NITS
        elif source == "linear ACEScg":
            rgb = _convert_primaries(rgb, _ACESCG_TO_REC2020, torch) * _SDR_WHITE_NITS
        elif source == "HDR LogC3":
            rgb = rgb.clamp(0.0, 1.0)
            rgb = torch.where(
                rgb >= _LOGC3_E * _LOGC3_CUT + _LOGC3_F,
                (torch.pow(10.0, (rgb - _LOGC3_D) / _LOGC3_C) - _LOGC3_B) / _LOGC3_A,
                (rgb - _LOGC3_F) / _LOGC3_E,
            ).clamp_min(0.0)
            rgb = _convert_primaries(rgb, _REC709_TO_REC2020, torch) * _SDR_WHITE_NITS
        elif source == "HDR ACEScct":
            rgb = (
                _convert_primaries(
                    torch.where(
                        rgb > _ACESCCT_LOG_CUT,
                        torch.exp2(rgb * _ACESCCT_LOG_SCALE - _ACESCCT_LOG_OFFSET),
                        (rgb - _ACESCCT_OFFSET) / _ACESCCT_SLOPE,
                    ),
                    _ACESCG_TO_REC2020,
                    torch,
                )
                * _SDR_WHITE_NITS
            )
        elif source == "HDR":
            scene = torch.where(
                rgb <= 0.5,
                (rgb**2) / 3.0,
                (torch.exp((rgb.clamp(min=0.5) - _HLG_C) / _HLG_A) + _HLG_B) / 12.0,
            )
            rgb = scene * (
                _luminance(scene, _REC2020_LUMA).clamp_min(0.0).pow(_HLG_GAMMA - 1.0)
                * _HLG_PEAK_NITS
            )
        elif source == "HDR PQ":
            pq = (rgb.clamp_min(0.0).log() / _PQ_M2).expm1()
            rgb = ((pq + (1.0 - _PQ_C1)).clamp_min(0.0) / ((_PQ_C2 - _PQ_C3) - _PQ_C3 * pq)).pow(
                1.0 / _PQ_M1
            ) * 10000.0
        else:
            raise ValueError(f"Unsupported source color space: {source}")
        if destination == "linear":
            rgb = _convert_primaries(rgb / _SDR_WHITE_NITS, _REC2020_TO_REC709, torch)
        elif destination == "linear ACEScg":
            rgb = _convert_primaries(rgb / _SDR_WHITE_NITS, _REC2020_TO_ACESCG, torch)
        elif destination == "HDR LogC3":
            rgb = _convert_primaries(rgb / _SDR_WHITE_NITS, _REC2020_TO_REC709, torch).clamp_min(
                0.0
            )
            rgb = torch.where(
                rgb >= _LOGC3_CUT,
                _LOGC3_C * torch.log10(_LOGC3_A * rgb + _LOGC3_B) + _LOGC3_D,
                _LOGC3_E * rgb + _LOGC3_F,
            ).clamp(0.0, 1.0)
        elif destination == "HDR ACEScct":
            rgb = _convert_primaries(rgb / _SDR_WHITE_NITS, _REC2020_TO_ACESCG, torch)
            rgb = torch.where(
                rgb > _ACESCCT_CUT,
                (torch.log2(rgb.clamp_min(_ACESCCT_CUT)) + _ACESCCT_LOG_OFFSET)
                / _ACESCCT_LOG_SCALE,
                _ACESCCT_SLOPE * rgb + _ACESCCT_OFFSET,
            )
        elif destination == "sRGB":
            rgb = _convert_primaries(rgb / _SDR_WHITE_NITS, _REC2020_TO_REC709, torch)
            rgb = _compress_gamut(_tone_map(rgb, _REC709_LUMA, torch), _REC709_LUMA, torch)
            rgb = torch.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb.pow(1.0 / 2.4) - 0.055)
        elif destination == "HDR":
            rgb = (rgb / _HLG_PEAK_NITS).clamp_min(0.0)
            if source == "HDR PQ":
                rgb = _tone_map(rgb, _REC2020_LUMA, torch)
            rgb = rgb * _luminance(rgb, _REC2020_LUMA).clamp_min(torch.finfo(rgb.dtype).tiny).pow(
                1.0 / _HLG_GAMMA - 1.0
            )
            if source == "HDR PQ":
                rgb = _compress_gamut(rgb, _REC2020_LUMA, torch)
            rgb = torch.where(
                rgb <= 1.0 / 12.0,
                (3.0 * rgb.clamp_min(0.0)).sqrt(),
                _HLG_A * (12.0 * rgb.clamp_min(1.0 / 12.0) - _HLG_B).log() + _HLG_C,
            )
        elif destination == "HDR PQ":
            pq = (rgb.clamp_min(0.0) / 10000.0).pow(_PQ_M1)
            rgb = (
                (((_PQ_C1 - 1.0) + (_PQ_C2 - _PQ_C3) * pq) / (1.0 + _PQ_C3 * pq)).log1p() * _PQ_M2
            ).exp()
        else:
            raise ValueError(f"Unsupported destination color space: {destination}")
        if image.shape[-1] == 4:
            rgb = torch.cat((rgb, image[..., 3:]), dim=-1)
        return cls.outputs(IMAGE=rgb.to(dtype=image.dtype, device=image.device))


IMAGE_COLOR_CONVERSION_NODES = (ImageSDRToHDR, ImageColorSpace)
