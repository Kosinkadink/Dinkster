from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("av")
pytest.importorskip("torch")

from tools import gen_comfy_source_parity_receipts as generator  # noqa: E402


def test_minimax_h3_alias_receipts_cover_each_pinned_mapping(tmp_path: Path) -> None:
    records = generator._mapping_records("comfy-core")

    outputs = generator._minimax_h3_alias_receipts(tmp_path, records)

    receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in outputs
        if path.name.endswith(".receipt.json")
    ]
    assert len(outputs) == 15
    assert {receipt["mapping"]["registryId"] for receipt in receipts} == {
        "comfy_alias:comfy-core/CLIPLoader",
        "comfy_alias:comfy-core/MiniMaxH3AddGuide",
        "comfy_alias:comfy-core/MiniMaxH3ImageToVideo",
        "comfy_alias:comfy-core/MiniMaxH3ReferenceToVideo",
        "comfy_alias:comfy-core/ResolutionSelector",
    }
    assert all(receipt["pass"] is True for receipt in receipts)


def test_controlnet_loader_trace_registry_supports_builtin_assembly_construction() -> None:
    source_calls: list[dict[str, object]] = []

    class SourceControlNetLoader:
        def load_controlnet(self, name: str) -> tuple[object]:
            source_calls.append(
                {"path": name, "role": "controlnet", "selection": "sdxl_control_lora"}
            )
            return (generator._SourceControlNet(),)  # pyright: ignore[reportPrivateUsage]

    source, native = generator._controlnet_loader_trace(  # pyright: ignore[reportPrivateUsage]
        SourceControlNetLoader,
        source_calls,
    )

    assert source["returnedControl"] is True
    assert native["returnedControl"] is True
    assert native["calls"] == [
        {
            "path": "models/controlnet/fixture-control-lora.safetensors",
            "role": "controlnet",
            "selection": "sdxl_control_lora",
        }
    ]


def test_adapter_hint_snapshot_preserves_rgb_until_runtime_normalization() -> None:
    import dinkster_inference_torch

    torch = generator.torch
    image = torch.tensor([2048.0, 1.0, -2048.0], dtype=torch.float16).reshape(1, 1, 1, 3)
    snapshot = generator.native_arm._snapshot_control_hint(
        image, torch, dinkster_inference_torch, channels=1
    )
    expected = image.movedim(-1, 1).float().contiguous()
    assert snapshot.shape == (1, 3, 1, 1)
    assert snapshot.data == expected.numpy().tobytes(order="C")
    assert snapshot.digest == dinkster_inference_torch.sd_control_hint_digest(expected)


def test_controlnet_receipts_execute_pinned_wrappers_and_native_semantics(tmp_path: Path) -> None:
    classes, loader_calls = generator._load_controlnet_reference(generator._comfy_root())
    records = generator._mapping_records()

    outputs = generator._controlnet_receipts(tmp_path, records, classes, loader_calls)

    receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in outputs
        if path.name.endswith(".receipt.json")
    ]
    assert len(outputs) == 12
    assert {receipt["mapping"]["registryId"] for receipt in receipts} == {
        "comfy_alias:comfy-core/ControlNetLoader",
        "comfy_alias:comfy-core/ControlNetApply",
        "comfy_alias:comfy-core/ControlNetApplyAdvanced",
        "comfy_alias:comfy-core/SetUnionControlNetType",
    }
    assert all(receipt["pass"] is True for receipt in receipts)
    assert all("weights" in receipt["parameters"]["scope"] for receipt in receipts[:1])


def test_video_receipt_fixture_probes_and_copies_through_v2(tmp_path: Path) -> None:
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    generator._write_video_fixture(first)
    generator._write_video_fixture(second)
    assert first.read_bytes() == second.read_bytes()
    value = generator.LoadVideoValue.execute(video=generator._asset_ref(first))["video"]
    probe = cast("dict[str, object]", cast("dict[str, object]", value)["probe"])
    assert probe["container"] == "mp4"
    assert probe["frame_count"] == 3
    assert probe["duration"] == 1
    assert generator.render_video_original(value) == first.read_bytes()
    trimmed = generator.TrimVideo.execute(video=value)["video"]
    saved = io.BytesIO()
    generator.save_video_stream(trimmed, saved)
    assert saved.getvalue() == first.read_bytes()


def test_resize_receipt_covers_all_dynamic_selections_for_image_and_mask(tmp_path: Path) -> None:
    record = generator._mapping_records("comfy-core")["ResizeImageMaskNode"]

    outputs = generator._resize_image_mask_receipt(
        tmp_path,
        record,
        generator._comfy_root(),
    )

    receipt = json.loads(outputs[-1].read_text(encoding="utf-8"))
    parameters = receipt["parameters"]
    assert parameters["scope"] == "all nine dynamic selections for IMAGE and MASK"
    assert len(parameters["selections"]) == 9
    assert len(parameters["outputShapes"]["nearest-exact"]) == 18
    assert parameters["maximumAbsoluteDifference"]["nearest-exact"] == 0.0
    assert parameters["maximumAbsoluteDifference"]["lanczos"] == 0.0
    assert receipt["pass"] is True


def test_control_aux_receipts_reject_evidence_resolution_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    implementation = "reference"
    case = "lineart-anime"
    relative = f"normal-512/{implementation}-{case}.json"
    source = generator.CONTROL_AUX_EVIDENCE / relative
    document = json.loads(source.read_text(encoding="utf-8"))
    document["resolution"] = 256
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(document), encoding="utf-8")
    checksums = {relative: hashlib.sha256(destination.read_bytes()).hexdigest()}
    monkeypatch.setattr(generator, "CONTROL_AUX_EVIDENCE", tmp_path)

    with pytest.raises(RuntimeError, match="invalid control auxiliary evidence"):
        generator._control_aux_result(implementation, case, checksums)


@pytest.mark.parametrize(
    ("node_class", "mapping_case", "target_input", "replacement_input", "error"),
    (
        (
            "LineArtPreprocessor",
            0,
            "coarse",
            {"kind": "constant", "value": False},
            "parameters do not match mapping: lineart-realistic-coarse",
        ),
        (
            "AnimeLineArtPreprocessor",
            0,
            "resolution",
            {"kind": "constant", "value": 256},
            "parameters do not match mapping: lineart-anime",
        ),
        (
            "AnyLineArtPreprocessor_aux",
            0,
            "lineart_lower_bound",
            {"kind": "copy", "input": "missing_source_input"},
            "mapping reads unavailable input: missing_source_input",
        ),
    ),
)
def test_control_aux_receipts_reject_mapping_parameter_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    node_class: str,
    mapping_case: int,
    target_input: str,
    replacement_input: dict[str, object],
    error: str,
) -> None:
    records = generator._mapping_records("comfyui_controlnet_aux")
    record = copy.deepcopy(records[node_class])
    replacement = cast("dict[str, Any]", record["replacement"])
    cases = cast("list[dict[str, Any]]", replacement["cases"])
    inputs = cast("dict[str, dict[str, object]]", cases[mapping_case]["inputs"])
    inputs[target_input] = replacement_input
    records[node_class] = record
    monkeypatch.setattr(generator, "_mapping_records", lambda _pack: records)

    with pytest.raises(RuntimeError, match=error):
        generator._control_aux_receipts(tmp_path)
