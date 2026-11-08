"""Real safetensors conditioning I/O and ordinary native hosts."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_assets import AssetError, MountDef, MountTable, digest_file
from dinkster_conditioning.conditioning_files import ConditioningLoader, SaveConditioning
from safetensors import safe_open
from safetensors.torch import save_file

pytestmark = pytest.mark.usefixtures("saved_artifact_context")


def _conditioning() -> list[Any]:
    return [
        [
            torch.tensor([[[2.0, -7.0], [11.0, 0.5]]]).transpose(1, 2),
            {
                "pooled_output": torch.tensor([[19.0, -3.0]], dtype=torch.float16),
                "references": [torch.tensor([5.0]), torch.tensor([-13.0, 17.0])],
                "enabled": True,
                "count": 7,
                "weight": 0.375,
                "label": "asymmetric",
                "absent": None,
            },
        ]
    ]


def _mounts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MountTable:
    output = tmp_path / "output"
    output.mkdir()
    table = MountTable(tmp_path / "mounts.json", index_root=tmp_path / "index")
    table.add(MountDef(id="input", path=tmp_path), kind="model/embedding")
    table.add(MountDef(id="output", path=output, mode="readwrite"))
    table.scan("input")
    table.scan("output")
    table.select_output_mount("output")
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(tmp_path / "mounts.json"))
    return table


def test_save_load_roundtrip_and_source_preservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mounts = _mounts(tmp_path, monkeypatch)
    original = _conditioning()
    before = original[0][0].clone()
    assert SaveConditioning.execute(conditioning=original)["conditioning"] is original
    assert (
        SaveConditioning.execute(conditioning=original, mount="output")["conditioning"] is original
    )
    paths = sorted((tmp_path / "output/conditioning").glob("*.safetensors"))
    assert [p.name for p in paths] == ["ComfyUI_00001_.safetensors", "ComfyUI_00002_.safetensors"]
    with safe_open(paths[0], framework="pt") as saved:
        assert set(saved.keys()) == {
            "conditioning",
            "pooled_output",
            "references.0",
            "references.1",
        }
        assert json.loads(saved.metadata()["conditioning_options"]) == {
            "enabled": True,
            "count": 7,
            "weight": 0.375,
            "label": "asymmetric",
        }
    # Bind the saved file through the ordinary embedding mount, not a raw path.
    mounts.scan("input")
    asset = mounts.bind_name(paths[0].name, "model/embedding")
    loaded = ConditioningLoader.execute(conditioning_name=asset)["CONDITIONING"][0]
    torch.testing.assert_close(loaded[0], before, rtol=0, atol=0)
    assert loaded[1]["pooled_output"].dtype == torch.float16
    torch.testing.assert_close(
        loaded[1]["pooled_output"],
        torch.tensor([[19.0, -3.0]], dtype=torch.float16),
        rtol=0,
        atol=0,
    )
    assert [v.tolist() for v in loaded[1]["references"]] == [[5.0], [-13.0, 17.0]]
    assert loaded[1]["enabled"] is True and loaded[1]["label"] == "asymmetric"
    assert "absent" not in loaded[1]
    torch.testing.assert_close(original[0][0], before, rtol=0, atol=0)
    assert not original[0][0].is_contiguous()


def test_loader_sorts_numeric_indices_and_tensor_options_override_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "numbered.safetensors"
    save_file(
        {
            "conditioning": torch.tensor([[[2.0]]]),
            "items.10": torch.tensor([10.0]),
            "items.2": torch.tensor([2.0]),
            "literal.x": torch.tensor([-9.0]),
        },
        path,
        metadata={
            "conditioning_options": '{"items":"replace","literal.x":"replace","scale":0.625}'
        },
    )
    mounts = _mounts(tmp_path, monkeypatch)
    ref = mounts.bind_name(path.name, "model/embedding")
    before = digest_file(path)
    options = ConditioningLoader.execute(conditioning_name=ref)["CONDITIONING"][0][1]
    assert [v.item() for v in options["items"]] == [2.0, 10.0]
    assert options["literal.x"].item() == -9.0 and options["scale"] == 0.625
    assert digest_file(path) == before


@pytest.mark.parametrize("bad", ([], [[], []], [[torch.zeros(1), {"bad": {"nested": 1}}]]))
def test_invalid_conditioning_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: Any
) -> None:
    _mounts(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        SaveConditioning.execute(conditioning=bad)
    assert not list((tmp_path / "output").rglob("*.safetensors"))


def test_grant_and_output_authority_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mounts(tmp_path, monkeypatch)
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        ConditioningLoader.execute(conditioning_name="raw.safetensors")
    with pytest.raises(AssetError):
        SaveConditioning.execute(conditioning=_conditioning(), filename_prefix="../escape")
    with pytest.raises(AssetError):
        SaveConditioning.execute(conditioning=_conditioning(), mount="input")
    assert not list((tmp_path / "output").rglob("*.safetensors"))


class InspectConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="conditioning-fixture.inspect",
            inputs=(InputSpec("value", TypeExpr.concrete(CONDITIONING_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, value: Any) -> dict[str, object]:
        assert len(value) == 1
        torch.testing.assert_close(
            value[0][0], torch.tensor([[[2.0, 11.0], [-7.0, 0.5]]]), rtol=0, atol=0
        )
        assert value[0][1]["references"][1].tolist() == [-13.0, 17.0]
        assert value[0][1]["pooled_output"].dtype == torch.float16
        assert value[0][1]["label"] == "asymmetric"
        return cls.outputs(summary="conditioning tensors and metadata preserved")
