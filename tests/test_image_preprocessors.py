from __future__ import annotations

import base64
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
from dinkster_nodes_image import IMAGE_NODES
from dinkster_nodes_image.preprocess import (
    PREPROCESSOR_NODES,
    BinaryPreprocessor,
    ColorHintPreprocessor,
    ContentShufflePreprocessor,
    EdgePreprocessor,
    HintImageResize,
    HintResolution,
    InpaintHintPreprocessor,
    LineartPreprocessor,
    ScribblePreprocessor,
    TileHintPreprocessor,
)

from tools.golden_platform import (
    GoldenUnavailableError,
    GoldenVariantNotFoundError,
    fetch_platform_golden,
)

GOLDEN_PATH = Path(__file__).parent / "goldens" / "preprocessors_controlnet_aux_59b1fc4.json"
GOLDEN = cast("dict[str, object]", json.loads(GOLDEN_PATH.read_text(encoding="utf-8")))
GOLDEN_CASES = cast("dict[str, object]", GOLDEN["cases"])
HINT_GOLDEN_PATH = Path(__file__).parent / "goldens" / "hint_resize_controlnet_aux_59b1fc4.json"
HINT_GOLDEN = cast("dict[str, object]", json.loads(HINT_GOLDEN_PATH.read_text(encoding="utf-8")))
ALIAS_PATH = (
    Path(__file__).parent.parent / "packages" / "dinkster-nodes-image" / "comfy-aliases.json"
)


def _decode(record: object) -> np.ndarray:
    payload = cast("dict[str, object]", record)
    shape = cast("list[int]", payload["shape"])
    encoded = cast("str", payload["uint8Base64"])
    return np.frombuffer(base64.b64decode(encoded), dtype=np.uint8).reshape(shape)


def _source() -> np.ndarray:
    return np.ascontiguousarray(_decode(GOLDEN["source"])[None].astype(np.float32) / 255.0)


def _golden(name: str) -> np.ndarray:
    return _decode(GOLDEN_CASES[name]).astype(np.float32) / 255.0


def _hint_source(name: str) -> np.ndarray:
    sources = cast("dict[str, object]", HINT_GOLDEN["sources"])
    return np.ascontiguousarray(_decode(sources[name])[None].astype(np.float32) / 255.0)


def _hint_case_document() -> dict[str, object]:
    # cv2's INTER_AREA/INTER_CUBIC resize kernels drift by one uint8 LSB
    # across platform builds (observed on darwin arm64 OpenCV 5.0.0: max
    # absolute difference 1 on 4.35-4.83% of elements in the continuous
    # cases; the INTER_NEAREST categorical/binary cases are bit-exact).
    # When an executed-reference variant of the same pinned controlnet_aux
    # baseline has been minted for the current platform/library tuple, it is
    # asserted instead of the Linux base goldens; platforms whose builds
    # match Linux bit-exactly (windows CI today) keep asserting the base.
    key = f"{sys.platform}-numpy{np.__version__}-opencv{cv2.__version__}"
    try:
        selected = fetch_platform_golden(HINT_GOLDEN_PATH, key)
    except GoldenVariantNotFoundError:
        return HINT_GOLDEN
    except GoldenUnavailableError as error:
        pytest.skip(f"platform evidence unavailable; baseline comparison skipped: {error}")
    document = cast("dict[str, object]", json.loads(selected.read_text(encoding="utf-8")))
    assert document["baseline"] == HINT_GOLDEN["baseline"]
    assert document["numpy"] == np.__version__
    assert document["opencv"] == cv2.__version__
    assert document["sources"] == HINT_GOLDEN["sources"]
    return document


def _hint_golden(name: str) -> np.ndarray:
    cases = cast("dict[str, object]", _hint_case_document()["cases"])
    return _decode(cases[name])


def test_checkpoint_free_preprocessors_match_controlnet_aux_goldens() -> None:
    image = _source()
    actual = {
        "binary_fixed": BinaryPreprocessor.execute(image=image, threshold=100, resolution=64)[
            "image"
        ],
        "binary_otsu": BinaryPreprocessor.execute(image=image, threshold=0, resolution=64)["image"],
        "canny": EdgePreprocessor.execute(image=image, resolution=64)["image"],
        "color_palette": ColorHintPreprocessor.execute(image=image, resolution=64)["image"],
        "content_shuffle": ContentShufflePreprocessor.execute(image=image, seed=7, resolution=64)[
            "image"
        ],
        "lineart": LineartPreprocessor.execute(
            image=image,
            gaussian_sigma=2.5,
            intensity_threshold=7,
            resolution=64,
        )["image"],
        "pyramid_canny": EdgePreprocessor.execute(
            image=image,
            method="pyramid_canny",
            low_threshold=64,
            high_threshold=128,
            resolution=64,
        )["image"],
        "recolor_intensity": ColorHintPreprocessor.execute(
            image=image,
            method="intensity",
            gamma=0.8,
            resolution=64,
        )["image"],
        "recolor_luminance": ColorHintPreprocessor.execute(
            image=image,
            method="luminance",
            gamma=1.2,
            resolution=64,
        )["image"],
        "scribble": ScribblePreprocessor.execute(image=image, resolution=64)["image"],
        "scribble_xdog": ScribblePreprocessor.execute(
            image=image,
            method="xdog",
            threshold=32,
            resolution=64,
        )["image"],
        "tile_guided": TileHintPreprocessor.execute(
            image=image,
            method="guided",
            scale_factor=2.0,
            blur_strength=3.0,
            radius=5,
            epsilon=0.01,
        )["image"],
        "tile_pyramid": TileHintPreprocessor.execute(image=image)["image"],
        "tile_simple": TileHintPreprocessor.execute(
            image=image,
            method="simple",
            scale_factor=2.0,
            blur_strength=3.0,
        )["image"],
    }
    assert set(actual) == set(GOLDEN_CASES)
    for name, output in actual.items():
        array = np.asarray(output)
        expected = _golden(name)
        assert array.shape == (1, *expected.shape)
        assert array.dtype == np.float32
        # OpenCV CPU kernels may differ by one quantization level across
        # architectures; a second level is a parity failure.
        np.testing.assert_allclose(array[0], expected, rtol=0, atol=1.0 / 255.0 + 1e-7)


@pytest.mark.parametrize("method", ["canny", "pyramid_canny"])
def test_edge_preprocessors_ignore_rgba_alpha(method: str) -> None:
    rgb = np.zeros((1, 16, 16, 3), dtype=np.float32)
    rgb[:, :, 8:, :] = 1.0
    transparent = np.concatenate((rgb, np.zeros((1, 16, 16, 1), dtype=np.float32)), axis=3)
    patterned = transparent.copy()
    patterned[..., 3] = np.linspace(0.0, 1.0, 16, dtype=np.float32)

    expected = EdgePreprocessor.execute(image=rgb, method=method, resolution=0)["image"]
    from_transparent = EdgePreprocessor.execute(image=transparent, method=method, resolution=0)[
        "image"
    ]
    from_patterned = EdgePreprocessor.execute(image=patterned, method=method, resolution=0)["image"]

    np.testing.assert_array_equal(from_transparent, expected)
    np.testing.assert_array_equal(from_patterned, expected)
    assert np.asarray(expected).shape[-1] == 3


def test_preprocessor_goldens_pin_the_source_revision() -> None:
    assert GOLDEN["baseline"] == "59b1fc411ede8623b2997855b8018f0b3b6cf49f"
    assert GOLDEN["opencv"] == "5.0.0"
    assert set(GOLDEN_CASES) == {
        "binary_fixed",
        "binary_otsu",
        "canny",
        "color_palette",
        "content_shuffle",
        "lineart",
        "pyramid_canny",
        "recolor_intensity",
        "recolor_luminance",
        "scribble",
        "scribble_xdog",
        "tile_guided",
        "tile_pyramid",
        "tile_simple",
    }


def test_preprocessor_nodes_are_registered_with_unique_native_ids() -> None:
    expected = {
        "dinkster.preprocess.binary",
        "dinkster.preprocess.color_hint",
        "dinkster.preprocess.content_shuffle",
        "dinkster.preprocess.edges",
        "dinkster.preprocess.hint_resize",
        "dinkster.preprocess.hint_resolution",
        "dinkster.preprocess.inpaint_hint",
        "dinkster.preprocess.lineart",
        "dinkster.preprocess.lineart_anime",
        "dinkster.preprocess.lineart_manga",
        "dinkster.preprocess.lineart_realistic",
        "dinkster.preprocess.anyline",
        "dinkster.preprocess.mlsd",
        "dinkster.preprocess.model_depth",
        "dinkster.preprocess.model_edges",
        "dinkster.preprocess.scribble",
        "dinkster.preprocess.teed",
        "dinkster.preprocess.tile_hint",
    }
    assert {node.schema().node_type for node in PREPROCESSOR_NODES} == expected
    assert all(node in IMAGE_NODES for node in PREPROCESSOR_NODES)


def test_controlnet_aux_aliases_preserve_preprocessor_parameters() -> None:
    registry = cast("dict[str, object]", json.loads(ALIAS_PATH.read_text(encoding="utf-8")))
    records: dict[str, dict[str, object]] = {}
    for record in cast("list[dict[str, object]]", registry["records"]):
        source = cast("dict[str, object]", record["source"])
        if source["pack"] == "comfyui_controlnet_aux":
            records[cast("str", source["nodeClass"])] = record
    assert set(records) == {
        "BinaryPreprocessor",
        "CannyEdgePreprocessor",
        "ColorPreprocessor",
        "DiffusionEdge_Preprocessor",
        "FakeScribblePreprocessor",
        "HEDPreprocessor",
        "HintImageEnchance",
        "ImageIntensityDetector",
        "ImageLuminanceDetector",
        "InpaintPreprocessor",
        "LineArtPreprocessor",
        "LineartStandardPreprocessor",
        "AnimeLineArtPreprocessor",
        "Manga2Anime_LineArt_Preprocessor",
        "AnyLineArtPreprocessor_aux",
        "M-LSDPreprocessor",
        "PiDiNetPreprocessor",
        "PixelPerfectResolution",
        "PyraCannyPreprocessor",
        "ScribblePreprocessor",
        "Scribble_PiDiNet_Preprocessor",
        "Scribble_XDoG_Preprocessor",
        "ShufflePreprocessor",
        "TTPlanet_TileGF_Preprocessor",
        "TTPlanet_TileSimple_Preprocessor",
        "TEEDPreprocessor",
        "TilePreprocessor",
    }

    def case(node_class: str, index: int = 0) -> dict[str, object]:
        replacement = cast("dict[str, object]", records[node_class]["replacement"])
        return cast("list[dict[str, object]]", replacement["cases"])[index]

    canny_inputs = cast("dict[str, object]", case("CannyEdgePreprocessor")["inputs"])
    pyramid_inputs = cast("dict[str, object]", case("PyraCannyPreprocessor")["inputs"])
    assert canny_inputs["method"] == {"kind": "constant", "value": "canny"}
    assert pyramid_inputs["method"] == {"kind": "constant", "value": "pyramid_canny"}
    assert pyramid_inputs["low_threshold"] == {"kind": "copy", "input": "low_threshold"}

    assert cast("dict[str, object]", case("LineartStandardPreprocessor")["inputs"])[
        "gaussian_sigma"
    ] == {"kind": "copy", "input": "guassian_sigma"}
    assert cast("dict[str, object]", case("BinaryPreprocessor")["inputs"])["threshold"] == {
        "kind": "copy",
        "input": "bin_threshold",
    }
    assert cast("dict[str, object]", case("ImageLuminanceDetector")["inputs"])["method"] == {
        "kind": "constant",
        "value": "luminance",
    }
    assert cast("dict[str, object]", case("ImageIntensityDetector")["inputs"])["method"] == {
        "kind": "constant",
        "value": "intensity",
    }

    tile_inputs = cast("dict[str, object]", case("TilePreprocessor")["inputs"])
    assert tile_inputs["iterations"] == {"kind": "copy", "input": "pyrUp_iters"}
    assert "resolution" not in tile_inputs
    guided_inputs = cast("dict[str, object]", case("TTPlanet_TileGF_Preprocessor")["inputs"])
    assert guided_inputs["epsilon"] == {"kind": "copy", "input": "eps"}
    assert "resolution" not in guided_inputs

    inpaint_cases = cast(
        "list[dict[str, object]]",
        cast("dict[str, object]", records["InpaintPreprocessor"]["replacement"])["cases"],
    )
    assert inpaint_cases[0]["when"] == {
        "kind": "valueEquals",
        "input": "black_pixel_for_xinsir_cn",
        "value": True,
    }
    assert cast("dict[str, object]", inpaint_cases[0]["inputs"])["masked_value"] == {
        "kind": "constant",
        "value": "black",
    }
    assert cast("dict[str, object]", inpaint_cases[1]["inputs"])["masked_value"] == {
        "kind": "constant",
        "value": "negative_one",
    }

    resolution_inputs = cast("dict[str, object]", case("PixelPerfectResolution")["inputs"])
    assert resolution_inputs["resize_mode"] == {
        "kind": "value",
        "input": "resize_mode",
        "transform": {
            "kind": "enumRename",
            "map": {
                "Just Resize": "stretch",
                "Crop and Resize": "fill",
                "Resize and Fill": "fit",
            },
        },
    }
    hint_resize_inputs = cast("dict[str, object]", case("HintImageEnchance")["inputs"])
    assert hint_resize_inputs["image"] == {"kind": "copy", "input": "hint_image"}
    assert hint_resize_inputs["resize_mode"] == resolution_inputs["resize_mode"]
    assert records["ShufflePreprocessor"]["confidence"] == {
        "tier": "parametric",
        "evidence": [
            "tests/test_image_preprocessors.py::test_checkpoint_free_preprocessors_match_controlnet_aux_goldens",
            "tests/test_image_preprocessors.py::test_controlnet_aux_aliases_preserve_preprocessor_parameters",
        ],
    }
    shuffle_rule = cast("dict[str, object]", records["ShufflePreprocessor"]["replacement"])
    assert "uint64 seeds above the JSON-safe integer maximum require review" in cast(
        "str", shuffle_rule["note"]
    )
    exact_records = {
        "AnimeLineArtPreprocessor",
        "AnyLineArtPreprocessor_aux",
        "DiffusionEdge_Preprocessor",
        "FakeScribblePreprocessor",
        "HEDPreprocessor",
        "LineArtPreprocessor",
        "M-LSDPreprocessor",
        "Manga2Anime_LineArt_Preprocessor",
        "PiDiNetPreprocessor",
        "Scribble_PiDiNet_Preprocessor",
        "TEEDPreprocessor",
    }
    assert {
        node_class: cast("dict[str, object]", record["confidence"])["tier"]
        for node_class, record in records.items()
        if node_class != "ShufflePreprocessor"
    } == {
        node_class: "exact" if node_class in exact_records else "equivalent"
        for node_class in set(records) - {"ShufflePreprocessor"}
    }


def test_content_shuffle_is_deterministic_for_zero_and_nonzero_seeds() -> None:
    image = _source()
    for seed in (0, 7):
        first = np.asarray(
            ContentShufflePreprocessor.execute(image=image, seed=seed, resolution=64)["image"]
        )
        second = np.asarray(
            ContentShufflePreprocessor.execute(image=image, seed=seed, resolution=64)["image"]
        )
        np.testing.assert_array_equal(first, second)
    zero = np.asarray(
        ContentShufflePreprocessor.execute(image=image, seed=0, resolution=64)["image"]
    )
    seven = np.asarray(
        ContentShufflePreprocessor.execute(image=image, seed=7, resolution=64)["image"]
    )
    assert not np.array_equal(zero, seven)


def test_inpaint_hint_resizes_and_broadcasts_masks() -> None:
    image = np.stack((_source()[0], 1.0 - _source()[0]))
    mask = np.zeros((1, 5, 7), dtype=np.float32)
    mask[:, 2:, 3:] = 1.0
    sentinel = np.asarray(InpaintHintPreprocessor.execute(image=image, mask=mask)["image"])
    black = np.asarray(
        InpaintHintPreprocessor.execute(image=image, mask=mask, masked_value="black")["image"]
    )
    assert sentinel.shape == image.shape
    assert np.any(sentinel == -1.0)
    assert np.all(sentinel[sentinel == -1.0] == -1.0)
    np.testing.assert_array_equal(black[sentinel == -1.0], 0.0)
    np.testing.assert_array_equal(black[sentinel != -1.0], image[sentinel != -1.0])


def test_inpaint_hint_enforces_the_image_operation_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dinkster_nodes_image.support.MAX_IMAGE_BYTES", 1)
    with pytest.raises(ValueError, match="image operation limit"):
        InpaintHintPreprocessor.execute(
            image=np.zeros((1, 1, 1, 3), dtype=np.float32),
            mask=np.zeros((1, 1, 1), dtype=np.float32),
        )


def test_hint_resolution_matches_pixel_perfect_rounding() -> None:
    image = np.zeros((2, 100, 400, 3), dtype=np.float32)
    assert HintResolution.execute(
        image=image,
        target_width=512,
        target_height=512,
        resize_mode="stretch",
    ) == {"resolution": 512}
    assert HintResolution.execute(
        image=image,
        target_width=512,
        target_height=512,
        resize_mode="fill",
    ) == {"resolution": 512}
    assert HintResolution.execute(
        image=image,
        target_width=512,
        target_height=512,
        resize_mode="fit",
    ) == {"resolution": 128}


def test_hint_image_resize_handles_batches_and_modes() -> None:
    image = np.concatenate((_source(), 1.0 - _source()), axis=0)
    for mode in ("stretch", "fill", "fit"):
        output = np.asarray(
            HintImageResize.execute(
                image=image,
                target_width=96,
                target_height=80,
                resize_mode=mode,
            )["image"]
        )
        assert output.shape == (2, 80, 96, 3)
        assert output.dtype == np.float32
        assert np.all((0.0 <= output) & (output <= 1.0))


@pytest.mark.parametrize(
    ("case", "source", "width", "height", "mode"),
    (
        ("continuous_stretch", "continuous", 96, 80, "stretch"),
        ("continuous_fill", "continuous", 96, 80, "fill"),
        ("continuous_fit", "continuous", 96, 80, "fit"),
        ("categorical_stretch", "categorical", 96, 80, "stretch"),
        ("binary_stretch", "binary", 166, 130, "stretch"),
    ),
)
def test_hint_image_resize_matches_controlnet_aux_goldens(
    case: str,
    source: str,
    width: int,
    height: int,
    mode: str,
) -> None:
    output = np.asarray(
        HintImageResize.execute(
            image=_hint_source(source),
            target_width=width,
            target_height=height,
            resize_mode=mode,
        )["image"]
    )
    expected = _hint_golden(case)
    assert output.shape == (1, *expected.shape)
    np.testing.assert_array_equal(np.rint(output[0] * 255.0).astype(np.uint8), expected)


def test_hint_resize_goldens_pin_the_source_revision() -> None:
    assert HINT_GOLDEN["baseline"] == "59b1fc411ede8623b2997855b8018f0b3b6cf49f"
    assert HINT_GOLDEN["opencv"] == "5.0.0"


def test_preprocessors_handle_batches_and_unscaled_resolution() -> None:
    image = np.concatenate((_source(), _source()), axis=0)
    output = EdgePreprocessor.execute(image=image, resolution=0)["image"]
    assert np.asarray(output).shape == (2, 65, 83, 3)
    uniform = np.zeros((1, 8, 8, 3), dtype=np.float32)
    pyramid = EdgePreprocessor.execute(
        image=uniform,
        method="pyramid_canny",
        resolution=64,
    )["image"]
    np.testing.assert_array_equal(pyramid, 0.0)


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (lambda: EdgePreprocessor.execute(image=_source(), resolution=32), "resolution"),
        (
            lambda: LineartPreprocessor.execute(image=_source(), gaussian_sigma=0.0),
            "gaussian_sigma",
        ),
        (
            lambda: ScribblePreprocessor.execute(image=_source(), method="unknown"),
            "unknown scribble",
        ),
        (lambda: BinaryPreprocessor.execute(image=_source(), threshold=300), "threshold"),
        (lambda: ColorHintPreprocessor.execute(image=_source(), gamma=3.0), "gamma"),
        (lambda: ContentShufflePreprocessor.execute(image=_source(), seed=-1), "seed"),
        (
            lambda: TileHintPreprocessor.execute(image=_source(), scale_factor=9.0),
            "scale_factor",
        ),
        (
            lambda: InpaintHintPreprocessor.execute(
                image=_source(),
                mask=np.zeros((2, 5, 5), dtype=np.float32),
            ),
            "mask batch",
        ),
        (
            lambda: HintImageResize.execute(
                image=_source(),
                target_width=32,
                target_height=512,
            ),
            "target dimensions",
        ),
    ),
)
def test_preprocessors_reject_invalid_parameters(call: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
