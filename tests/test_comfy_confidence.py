from __future__ import annotations

import copy
import json
import os
import subprocess
from decimal import localcontext
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tools.comfy_confidence as confidence
from tools.comfy_confidence import (
    COMFYUI_REFERENCE_REVISION,
    ConfidenceReceiptError,
    canonical_bytes,
    create_receipt,
    load_receipt,
    main,
    validate_receipt,
    verify_receipt,
    write_receipt,
)


def mapping(
    *,
    pack: str = "comfy-core",
    name: str = "ImageScale",
    revision: str = COMFYUI_REFERENCE_REVISION,
    reference_kind: str = "comfyui-pinned",
    registry_kind: str = "alias",
    mapping_kind: str = "op",
    tier: str = "exact",
    target_kind: str = "node",
) -> dict[str, Any]:
    return {
        "registryId": f"comfy_{registry_kind}:{pack}/{name}",
        "mappingKind": mapping_kind,
        "tier": tier,
        "source": {
            "pack": pack,
            "name": name,
            "revision": revision,
            "referenceKind": reference_kind,
        },
        "target": {"kind": target_kind, "id": "dinkster.image.resize"},
    }


def exact_value_receipt(tmp_path: Path, *, equal: bool = True) -> dict[str, Any]:
    reference = tmp_path / "reference.json"
    native = tmp_path / "native.json"
    reference.write_text('{"result":[1,true,"x"]}', encoding="ascii")
    native.write_text(
        '{"result":[1,true,"x"]}' if equal else '{"result":[1,false,"x"]}',
        encoding="ascii",
    )
    return create_receipt(
        case_id="core/image-scale/default",
        mapping=mapping(),
        parameters={"mode": "bicubic", "width": 64},
        comparison={"comparator": "exact-value/1", "dataKind": "value"},
        artifact_root=tmp_path,
        reference_path=reference.name,
        native_path=native.name,
    )


def test_exact_value_receipt_is_canonical_reproducible_and_hash_verified(
    tmp_path: Path,
) -> None:
    first = exact_value_receipt(tmp_path)
    second = exact_value_receipt(tmp_path)
    assert first == second
    assert first["pass"] is True
    assert first["observed"] == {"equal": True}
    artifacts = first["artifacts"]
    assert isinstance(artifacts, dict)
    assert artifacts["reference"]["sha256"].startswith("sha256:")

    path = tmp_path / "receipt.json"
    write_receipt(path, first)
    assert path.read_bytes() == canonical_bytes(first)
    assert load_receipt(path) == first
    assert verify_receipt(first, tmp_path) == first

    (tmp_path / "native.json").write_text('{"result":null}', encoding="ascii")
    with pytest.raises(ConfidenceReceiptError, match="hash or size"):
        verify_receipt(first, tmp_path)


def test_exact_receipts_record_value_and_array_failures_without_claiming_pass(
    tmp_path: Path,
) -> None:
    value = exact_value_receipt(tmp_path, equal=False)
    assert value["pass"] is False
    assert value["observed"] == {"equal": False}

    np.save(tmp_path / "reference.npy", np.array([[True, False]], dtype=np.bool_))
    np.save(tmp_path / "native.npy", np.array([[True, True]], dtype=np.bool_))
    array = create_receipt(
        case_id="ecosystem/mask",
        mapping=mapping(
            pack="custom-pack",
            name="MaskNode",
            revision="0123456789abcdef",
            reference_kind="static",
        ),
        parameters={},
        comparison={"comparator": "exact-array/1", "dataKind": "mask"},
        artifact_root=tmp_path,
        reference_path="reference.npy",
        native_path="native.npy",
    )
    assert array["pass"] is False
    assert array["observed"] == {
        "dtype_equal": True,
        "shape_equal": True,
        "values_equal": False,
    }


def test_exact_and_numeric_values_preserve_json_number_precision(tmp_path: Path) -> None:
    (tmp_path / "reference.json").write_text("0.1", encoding="ascii")
    (tmp_path / "native.json").write_text("0.10000000000000001", encoding="ascii")
    exact = create_receipt(
        case_id="core/value/decimal",
        mapping=mapping(),
        parameters={},
        comparison={"comparator": "exact-value/1", "dataKind": "value"},
        artifact_root=tmp_path,
        reference_path="reference.json",
        native_path="native.json",
    )
    assert exact["pass"] is False

    (tmp_path / "reference.json").write_text("9007199254740992", encoding="ascii")
    (tmp_path / "native.json").write_text("9007199254740993", encoding="ascii")
    numeric = create_receipt(
        case_id="core/value/integer",
        mapping=mapping(tier="parametric"),
        parameters={},
        comparison={
            "comparator": "numeric-value/1",
            "dataKind": "value",
            "tolerances": [{"metric": "max_abs", "operator": "<=", "value": 0.5}],
        },
        artifact_root=tmp_path,
        reference_path="reference.json",
        native_path="native.json",
    )
    assert numeric["pass"] is False
    assert numeric["observed"] == {
        "shape_equal": True,
        "max_abs": 1.0,
        "mean_abs": 1.0,
        "mismatch_fraction": 1.0,
    }

    (tmp_path / "reference.json").write_text("0.051", encoding="ascii")
    (tmp_path / "native.json").write_text("0", encoding="ascii")
    with localcontext() as context:
        context.prec = 1
        ambient = create_receipt(
            case_id="core/value/ambient-decimal-context",
            mapping=mapping(tier="equivalent"),
            parameters={},
            comparison={
                "comparator": "numeric-value/1",
                "dataKind": "value",
                "tolerances": [{"metric": "max_abs", "operator": "<=", "value": 0.05}],
            },
            artifact_root=tmp_path,
            reference_path="reference.json",
            native_path="native.json",
        )
    observed = ambient["observed"]
    assert isinstance(observed, dict)
    assert observed["max_abs"] == 0.051
    assert ambient["pass"] is False

    (tmp_path / "reference.json").write_text("1e999999999", encoding="ascii")
    with pytest.raises(ConfidenceReceiptError, match="exceeds float64"):
        create_receipt(
            case_id="core/value/overflow",
            mapping=mapping(tier="equivalent"),
            parameters={},
            comparison={
                "comparator": "numeric-value/1",
                "dataKind": "value",
                "tolerances": [{"metric": "max_abs", "operator": "<=", "value": 1.0}],
            },
            artifact_root=tmp_path,
            reference_path="reference.json",
            native_path="native.json",
        )

    (tmp_path / "reference.json").write_text("1e-999999999", encoding="ascii")
    (tmp_path / "native.json").write_text("0", encoding="ascii")
    with pytest.raises(ConfidenceReceiptError, match="exceeds float64"):
        create_receipt(
            case_id="core/value/underflow",
            mapping=mapping(tier="equivalent"),
            parameters={},
            comparison={
                "comparator": "numeric-value/1",
                "dataKind": "value",
                "tolerances": [{"metric": "max_abs", "operator": "<=", "value": 0.0}],
            },
            artifact_root=tmp_path,
            reference_path="reference.json",
            native_path="native.json",
        )


@pytest.mark.parametrize("data_kind", ("image", "mask", "tensor"))
def test_exact_array_comparator_covers_every_array_data_kind(
    tmp_path: Path, data_kind: str
) -> None:
    values = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    np.save(tmp_path / f"{data_kind}-reference.npy", values)
    np.save(tmp_path / f"{data_kind}-native.npy", values)
    receipt = create_receipt(
        case_id=f"core/{data_kind}",
        mapping=mapping(),
        parameters={"shape": list(values.shape)},
        comparison={"comparator": "exact-array/1", "dataKind": data_kind},
        artifact_root=tmp_path,
        reference_path=f"{data_kind}-reference.npy",
        native_path=f"{data_kind}-native.npy",
    )
    assert receipt["pass"] is True


def test_numeric_array_comparator_applies_declared_metrics_and_shape_gate(
    tmp_path: Path,
) -> None:
    np.save(tmp_path / "reference.npy", np.array([0.0, 1.0, 2.0], dtype=np.float32))
    np.save(tmp_path / "native.npy", np.array([0.0, 1.001, 2.0], dtype=np.float64))
    comparison = {
        "comparator": "numeric-array/1",
        "dataKind": "tensor",
        "tolerances": [
            {"metric": "max_abs", "operator": "<=", "value": 0.002},
            {"metric": "mean_abs", "operator": "<=", "value": 0.001},
            {"metric": "mismatch_fraction", "operator": "<=", "value": 0.4},
        ],
    }
    receipt = create_receipt(
        case_id="ecosystem/tensor/tolerant",
        mapping=mapping(
            pack="custom-pack",
            name="TensorNode",
            revision="v1.2.3",
            reference_kind="static",
            tier="equivalent",
        ),
        parameters={"scale": 1.0},
        comparison=comparison,
        artifact_root=tmp_path,
        reference_path="reference.npy",
        native_path="native.npy",
    )
    assert receipt["pass"] is True
    observed = receipt["observed"]
    assert isinstance(observed, dict)
    assert observed["shape_equal"] is True
    assert observed["max_abs"] == pytest.approx(0.001)
    assert observed["mean_abs"] == pytest.approx(0.001 / 3)
    assert observed["mismatch_fraction"] == pytest.approx(1 / 3)

    strict = copy.deepcopy(comparison)
    strict["tolerances"][0]["value"] = 0.0005
    failed = create_receipt(
        case_id="ecosystem/tensor/strict",
        mapping=mapping(
            pack="custom-pack",
            name="TensorNode",
            revision="v1.2.3",
            reference_kind="static",
            tier="equivalent",
        ),
        parameters={},
        comparison=strict,
        artifact_root=tmp_path,
        reference_path="reference.npy",
        native_path="native.npy",
    )
    assert failed["pass"] is False

    np.save(tmp_path / "native-shape.npy", np.array([[0.0, 1.0, 2.0]], dtype=np.float32))
    shape = create_receipt(
        case_id="ecosystem/tensor/shape",
        mapping=mapping(
            pack="custom-pack",
            name="TensorNode",
            revision="v1.2.3",
            reference_kind="static",
            tier="equivalent",
        ),
        parameters={},
        comparison=comparison,
        artifact_root=tmp_path,
        reference_path="reference.npy",
        native_path="native-shape.npy",
    )
    assert shape["pass"] is False
    assert shape["observed"] == {
        "shape_equal": False,
        "max_abs": None,
        "mean_abs": None,
        "mismatch_fraction": None,
    }


def test_numeric_value_comparator_applies_tolerance_to_json_scalars(tmp_path: Path) -> None:
    (tmp_path / "reference.json").write_text("0.125", encoding="ascii")
    (tmp_path / "native.json").write_text("0.1255", encoding="ascii")
    receipt = create_receipt(
        case_id="ecosystem/value/tolerant",
        mapping=mapping(
            pack="custom-pack",
            name="ValueNode",
            revision="deadbeef",
            reference_kind="static",
            tier="equivalent",
        ),
        parameters={},
        comparison={
            "comparator": "numeric-value/1",
            "dataKind": "value",
            "tolerances": [{"metric": "max_abs", "operator": "<=", "value": 0.001}],
        },
        artifact_root=tmp_path,
        reference_path="reference.json",
        native_path="native.json",
    )
    assert receipt["pass"] is True
    assert receipt["observed"] == {
        "shape_equal": True,
        "max_abs": pytest.approx(0.0005),
        "mean_abs": pytest.approx(0.0005),
        "mismatch_fraction": 1.0,
    }


def test_receipt_enforces_pinned_execution_and_static_ecosystem_sources(
    tmp_path: Path,
) -> None:
    exact_value_receipt(tmp_path)
    latest_core = mapping(revision="8a33128f")
    latest_receipt = create_receipt(
        case_id="core/latest-pinned-revision",
        mapping=latest_core,
        parameters={},
        comparison={"comparator": "exact-value/1", "dataKind": "value"},
        artifact_root=tmp_path,
        reference_path="reference.json",
        native_path="native.json",
    )
    assert latest_receipt["mapping"] == latest_core

    invalid = (
        mapping(revision="main"),
        mapping(pack="custom-pack", revision="deadbeef", reference_kind="comfyui-pinned"),
        mapping(pack="custom-pack", revision="deadbeef", reference_kind="ecosystem-pinned"),
        mapping(reference_kind="ecosystem-pinned"),
        mapping(registry_kind="group", tier="exact"),
        mapping(tier="grouped"),
    )
    for candidate in invalid:
        with pytest.raises(ConfidenceReceiptError):
            create_receipt(
                case_id="invalid/source",
                mapping=candidate,
                parameters={},
                comparison={"comparator": "exact-value/1", "dataKind": "value"},
                artifact_root=tmp_path,
                reference_path="reference.json",
                native_path="native.json",
            )

    ecosystem = mapping(
        pack="custom-pack",
        name="EdgeNode",
        revision="0123456789abcdef0123456789abcdef01234567",
        reference_kind="ecosystem-pinned",
    )
    receipt = create_receipt(
        case_id="ecosystem/edge-node",
        mapping=ecosystem,
        parameters={},
        comparison={"comparator": "exact-value/1", "dataKind": "value"},
        artifact_root=tmp_path,
        reference_path="reference.json",
        native_path="native.json",
    )
    assert receipt["mapping"] == ecosystem

    grouped = mapping(
        pack="custom-pack",
        name="resize-stack",
        revision="deadbeef",
        reference_kind="static",
        registry_kind="group",
        tier="grouped",
        target_kind="group",
    )
    receipt = create_receipt(
        case_id="group/resize-stack",
        mapping=grouped,
        parameters={},
        comparison={"comparator": "exact-value/1", "dataKind": "value"},
        artifact_root=tmp_path,
        reference_path="reference.json",
        native_path="native.json",
    )
    assert receipt["mapping"] == grouped

    vision_grouped = mapping(
        name="vision-stack",
        revision="c67885b1",
        registry_kind="group",
        tier="grouped",
        target_kind="group",
    )
    receipt = create_receipt(
        case_id="group/vision-stack",
        mapping=vision_grouped,
        parameters={},
        comparison={"comparator": "exact-value/1", "dataKind": "value"},
        artifact_root=tmp_path,
        reference_path="reference.json",
        native_path="native.json",
    )
    assert receipt["mapping"] == vision_grouped


def test_receipt_rejects_noncanonical_comparator_and_tolerance_claims(tmp_path: Path) -> None:
    base = exact_value_receipt(tmp_path)
    cases = []
    unknown = copy.deepcopy(base)
    unknown["comparison"]["unknown"] = True
    cases.append(unknown)
    tolerance_on_exact = copy.deepcopy(base)
    tolerance_on_exact["comparison"]["tolerances"] = [
        {"metric": "max_abs", "operator": "<=", "value": 0.0}
    ]
    cases.append(tolerance_on_exact)
    boolean_threshold = copy.deepcopy(base)
    boolean_threshold["mapping"]["tier"] = "equivalent"
    boolean_threshold["comparison"] = {
        "comparator": "numeric-array/1",
        "dataKind": "tensor",
        "tolerances": [{"metric": "max_abs", "operator": "<=", "value": True}],
    }
    cases.append(boolean_threshold)
    duplicate_metric = copy.deepcopy(boolean_threshold)
    duplicate_metric["comparison"]["tolerances"] = [
        {"metric": "max_abs", "operator": "<=", "value": 1.0},
        {"metric": "max_abs", "operator": "<=", "value": 0.0},
    ]
    cases.append(duplicate_metric)
    wrong_data = copy.deepcopy(base)
    wrong_data["comparison"]["dataKind"] = "image"
    cases.append(wrong_data)
    tampered_pass = copy.deepcopy(base)
    tampered_pass["pass"] = False
    cases.append(tampered_pass)
    inverted_error = copy.deepcopy(boolean_threshold)
    inverted_error["comparison"]["tolerances"][0] = {
        "metric": "max_abs",
        "operator": ">=",
        "value": 0.0,
    }
    cases.append(inverted_error)
    invalid_fraction = copy.deepcopy(boolean_threshold)
    invalid_fraction["comparison"]["tolerances"][0] = {
        "metric": "mismatch_fraction",
        "operator": "<=",
        "value": 1.1,
    }
    cases.append(invalid_fraction)
    for candidate in cases:
        with pytest.raises(ConfidenceReceiptError):
            validate_receipt(candidate)


def test_receipt_loader_rejects_duplicate_keys_noncanonical_json_and_nonfinite(
    tmp_path: Path,
) -> None:
    receipt = exact_value_receipt(tmp_path)
    path = tmp_path / "receipt.json"
    path.write_bytes(canonical_bytes(receipt))
    encoded = path.read_text(encoding="ascii")
    path.write_text(encoded.replace('"format":', '"format": "duplicate",\n  "format":', 1))
    with pytest.raises(ConfidenceReceiptError, match="duplicate"):
        load_receipt(path)

    path.write_text(json.dumps(receipt), encoding="ascii")
    with pytest.raises(ConfidenceReceiptError, match="canonical"):
        load_receipt(path)

    path.write_text('{"value":NaN}\n', encoding="ascii")
    with pytest.raises(ConfidenceReceiptError, match="non-finite"):
        load_receipt(path)

    link = tmp_path / "receipt-link.json"
    try:
        link.symlink_to(path)
    except OSError:
        pass
    else:
        with pytest.raises(ConfidenceReceiptError, match="ordinary file"):
            load_receipt(link)


def test_windows_file_identity_ignores_incomparable_creation_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("{}", encoding="ascii")
    path_stat = path.lstat()
    original_fstat = confidence.os.fstat

    class WindowsHandleStat:
        def __init__(self, value: Any) -> None:
            self.st_mode = value.st_mode
            self.st_dev = value.st_dev
            self.st_ino = value.st_ino
            self.st_size = value.st_size
            self.st_mtime_ns = value.st_mtime_ns
            self.st_ctime_ns = value.st_ctime_ns + 1

    monkeypatch.setattr(confidence.os, "name", "nt")
    monkeypatch.setattr(
        confidence.os,
        "fstat",
        lambda descriptor: WindowsHandleStat(original_fstat(descriptor)),
    )

    assert confidence._read_bounded_regular(path, path_stat, 16, "artifact") == b"{}"


def test_artifacts_require_canonical_paths_and_distinct_files(tmp_path: Path) -> None:
    receipt = exact_value_receipt(tmp_path)
    with pytest.raises(ConfidenceReceiptError, match="canonical"):
        create_receipt(
            case_id="invalid/path-alias",
            mapping=mapping(),
            parameters={},
            comparison={"comparator": "exact-value/1", "dataKind": "value"},
            artifact_root=tmp_path,
            reference_path="reference.json",
            native_path="./reference.json",
        )

    native = tmp_path / "native.json"
    native.unlink()
    os.link(tmp_path / "reference.json", native)
    with pytest.raises(ConfidenceReceiptError, match="distinct files"):
        create_receipt(
            case_id="invalid/hard-link",
            mapping=mapping(),
            parameters={},
            comparison={"comparator": "exact-value/1", "dataKind": "value"},
            artifact_root=tmp_path,
            reference_path="reference.json",
            native_path="native.json",
        )
    with pytest.raises(ConfidenceReceiptError, match="distinct files"):
        verify_receipt(receipt, tmp_path)


def test_artifact_paths_and_formats_are_bounded_and_data_only(tmp_path: Path) -> None:
    exact_value_receipt(tmp_path)
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}", encoding="ascii")
    with pytest.raises(ConfidenceReceiptError, match="contained"):
        create_receipt(
            case_id="invalid/path",
            mapping=mapping(),
            parameters={},
            comparison={"comparator": "exact-value/1", "dataKind": "value"},
            artifact_root=tmp_path,
            reference_path="../outside.json",
            native_path="native.json",
        )

    link = tmp_path / "linked.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pass
    else:
        with pytest.raises(ConfidenceReceiptError, match="contained|ordinary file"):
            create_receipt(
                case_id="invalid/link",
                mapping=mapping(),
                parameters={},
                comparison={"comparator": "exact-value/1", "dataKind": "value"},
                artifact_root=tmp_path,
                reference_path="linked.json",
                native_path="native.json",
            )

    outside_directory = tmp_path.parent / "outside-directory"
    outside_directory.mkdir()
    (outside_directory / "reference.json").write_text("{}", encoding="ascii")
    directory_link = tmp_path / "linked-directory"
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/d", "/c", "mklink", "/J", str(directory_link), str(outside_directory)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr or result.stdout
    else:
        directory_link.symlink_to(outside_directory, target_is_directory=True)
    try:
        with pytest.raises(ConfidenceReceiptError, match="contained|ordinary"):
            create_receipt(
                case_id="invalid/directory-link",
                mapping=mapping(),
                parameters={},
                comparison={"comparator": "exact-value/1", "dataKind": "value"},
                artifact_root=tmp_path,
                reference_path="linked-directory/reference.json",
                native_path="native.json",
            )
    finally:
        if os.name == "nt":
            os.rmdir(directory_link)
        else:
            directory_link.unlink()

    np.save(tmp_path / "object.npy", np.array([object()], dtype=object))
    np.save(tmp_path / "float.npy", np.array([1.0], dtype=np.float32))
    with pytest.raises(ConfidenceReceiptError, match="safe NumPy array"):
        create_receipt(
            case_id="invalid/object-array",
            mapping=mapping(),
            parameters={},
            comparison={"comparator": "exact-array/1", "dataKind": "tensor"},
            artifact_root=tmp_path,
            reference_path="object.npy",
            native_path="float.npy",
        )

    (tmp_path / "malformed.npy").write_bytes(b"x")
    with pytest.raises(ConfidenceReceiptError, match="safe NumPy array"):
        create_receipt(
            case_id="invalid/malformed-array",
            mapping=mapping(),
            parameters={},
            comparison={"comparator": "exact-array/1", "dataKind": "tensor"},
            artifact_root=tmp_path,
            reference_path="malformed.npy",
            native_path="float.npy",
        )

    np.save(tmp_path / "nan.npy", np.array([np.nan], dtype=np.float32))
    with pytest.raises(ConfidenceReceiptError, match="non-finite"):
        create_receipt(
            case_id="invalid/nan-array",
            mapping=mapping(),
            parameters={},
            comparison={"comparator": "exact-array/1", "dataKind": "tensor"},
            artifact_root=tmp_path,
            reference_path="nan.npy",
            native_path="float.npy",
        )


def test_cli_distinguishes_verified_failure_from_invalid_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt = exact_value_receipt(tmp_path, equal=False)
    path = tmp_path / "receipt.json"
    write_receipt(path, receipt)
    assert main([str(path), "--artifact-root", str(tmp_path)]) == 1
    assert '"pass": false' in capsys.readouterr().out

    (tmp_path / "native.json").write_text("null", encoding="ascii")
    assert main([str(path), "--artifact-root", str(tmp_path)]) == 2
    assert "hash or size" in capsys.readouterr().err
