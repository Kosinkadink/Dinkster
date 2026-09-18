from __future__ import annotations

import json
import os
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from tools.inference_parity import lora_storage_receipts as receipts

PACKET = Path("tools/inference_parity/lora_storage_artifacts.json")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX descriptor APIs")
_pread = cast("Callable[[int, int, int], bytes]", getattr(os, "pread", None))


def _write_safetensors(path: Path, header: dict[str, object], payload: bytes = b"") -> Path:
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return path


def test_committed_packet_is_exact_and_closed() -> None:
    packet = receipts.load_packet(PACKET)

    assert packet["schema"] == 1
    assert tuple(artifact["role"] for artifact in packet["artifacts"]) == (
        "combined-sd15-checkpoint",
        "sd15-unet-lcm-lora",
    )
    no_lora = packet["graphs"]["no_lora"]
    with_lora = packet["graphs"]["with_lora"]
    assert no_lora["nodes"]["sampler"] == with_lora["nodes"]["sampler"] == "dinkster.ksampler"
    assert ["checkpoint.model", "sampler.model"] in no_lora["edges"]
    assert ["checkpoint.model", "lora.model"] in with_lora["edges"]
    assert ["lora.model", "sampler.model"] in with_lora["edges"]
    assert ["latent.latent", "sampler.latent_image"] in with_lora["edges"]
    assert ["sampler.latent", "decode.samples"] in with_lora["edges"]
    assert with_lora["inputs"]["lora.strength_model"] == 1.0
    assert no_lora["output"] == with_lora["output"] == "decode.image"
    assert packet["storage_oracle"]["operation_order"] == [
        "load owned fp32 base storage",
        "decode immutable ordered PatchSet",
        "apply every patch entry in declaration order in fp32",
        "cast exactly once to authoritative fp16 storage",
        "enroll converted storage with no deferred PatchSet",
    ]
    assert packet["physical_execution"] == "blocked_pending_independent_packet_review"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda packet: packet.update(extra=True),
        lambda packet: packet["artifacts"][0].update(revision="main"),
        lambda packet: packet["artifacts"][0].update(source_url="https://example.invalid/model"),
        lambda packet: packet["artifacts"][0].update(local_path="../escape.safetensors"),
        lambda packet: packet["artifacts"][0].update(size=True),
        lambda packet: packet["artifacts"][0].update(sha256="A" * 64),
        lambda packet: packet["licenses"][0].update(accepted=False),
        lambda packet: packet["graphs"]["with_lora"]["inputs"].update({"lora.strength_model": 0.5}),
        lambda packet: packet["graphs"]["with_lora"]["inputs"].update(
            {"lora.strength_model": True}
        ),
        lambda packet: packet["storage_oracle"].update(wiring_epoch=18),
        lambda packet: packet["storage_oracle"].update(wiring_epoch=17.0),
    ],
)
def test_packet_refuses_contract_drift(
    mutation: Callable[[dict[str, Any]], object], tmp_path: Path
) -> None:
    packet = json.loads(PACKET.read_text("utf-8"))
    mutation(packet)
    candidate = tmp_path / "packet.json"
    candidate.write_text(json.dumps(packet), "utf-8")

    with pytest.raises(receipts.ReceiptError):
        receipts.load_packet(candidate)


def test_packet_refuses_duplicate_json_keys(tmp_path: Path) -> None:
    candidate = tmp_path / "packet.json"
    candidate.write_text('{"schema":1,"schema":1}', "utf-8")

    with pytest.raises(receipts.ReceiptError, match="duplicate JSON key"):
        receipts.load_packet(candidate)


def test_artifact_path_refuses_intermediate_symlink(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(ordinary, target_is_directory=True)

    model = ordinary / "model.safetensors"
    model.write_bytes(b"model")
    with receipts._open_artifact(tmp_path, "ordinary/model.safetensors", 5):
        pass
    with pytest.raises(receipts.ReceiptError, match="without symlinks"):
        with receipts._open_artifact(tmp_path, "linked/model.safetensors", 5):
            pass


def test_artifact_path_detects_parent_directory_rebind(tmp_path: Path) -> None:
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "model.safetensors").write_bytes(b"artifact")

    with pytest.raises(receipts.ReceiptError, match="directory changed"):
        with receipts._open_artifact(root, "nested/model.safetensors", 8) as descriptor:
            assert _pread(descriptor, 8, 0) == b"artifact"
            nested.rename(root / "old-nested")
            replacement = root / "nested"
            replacement.mkdir()
            (replacement / "model.safetensors").write_bytes(b"artifact")


def test_artifact_path_detects_root_directory_rebind(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"artifact")

    with pytest.raises(receipts.ReceiptError, match="root changed"):
        with receipts._open_artifact(root, "model.safetensors", 8) as descriptor:
            assert _pread(descriptor, 8, 0) == b"artifact"
            root.rename(tmp_path / "old-root")
            root.mkdir()
            (root / "model.safetensors").write_bytes(b"artifact")


def test_artifact_path_detects_final_name_rebind(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    artifact = root / "model.safetensors"
    artifact.write_bytes(b"artifact")

    with pytest.raises(receipts.ReceiptError, match="changed"):
        with receipts._open_artifact(root, "model.safetensors", 8) as descriptor:
            assert _pread(descriptor, 8, 0) == b"artifact"
            artifact.rename(root / "old-model.safetensors")
            artifact.write_bytes(b"artifact")


def test_graph_authority_resolves_against_registered_node_schemas() -> None:
    from dinkster_compat_comfy.native import NATIVE_NODES
    from dinkster_inference import builtin_sampler_registry, builtin_scheduler_registry
    from dinkster_nodes_generation import GENERATION_NODES
    from dinkster_schema import ComboOption, ComboWidget, NumberWidget

    schemas = {
        node.schema().node_type: node.schema() for node in (*NATIVE_NODES, *GENERATION_NODES)
    }
    packet = receipts.load_packet(PACKET)
    for graph in packet["graphs"].values():
        graph_schemas = {name: schemas[node_type] for name, node_type in graph["nodes"].items()}
        for source, target in graph["edges"]:
            source_node, source_port = source.split(".", 1)
            target_node, target_port = target.split(".", 1)
            outputs = {output.id: output for output in graph_schemas[source_node].outputs}
            inputs = {input_.id: input_ for input_ in graph_schemas[target_node].inputs}
            assert source_port in outputs
            assert target_port in inputs
            assert outputs[source_port].type == inputs[target_port].type
        output_node, output_port = graph["output"].split(".", 1)
        assert output_port in {output.id for output in graph_schemas[output_node].outputs}
        for target, value in graph["inputs"].items():
            target_node, target_port = target.split(".", 1)
            inputs = {input_.id: input_ for input_ in graph_schemas[target_node].inputs}
            input_ = inputs[target_port]
            type_id = input_.type.types
            if type_id == ("dinkster.asset",):
                assert value in graph["artifact_roles"]
            elif type_id == ("core.int",):
                assert type(value) is int
            elif type_id == ("core.float",):
                assert type(value) in {int, float}
            elif type_id == ("core.string",):
                assert isinstance(value, str)
            elif type_id == ("core.combo",):
                assert isinstance(value, str)
            else:
                pytest.fail(f"unlinked literal targets non-literal type {type_id}")
            widget = input_.widget
            if isinstance(widget, ComboWidget):
                assert isinstance(value, str)
                option_values = tuple(
                    option.value if isinstance(option, ComboOption) else option
                    for option in widget.options
                )
                if target_port == "sampler_name":
                    descriptor = builtin_sampler_registry().get(value)
                    assert descriptor is not None and descriptor.id in option_values
                elif target_port == "scheduler":
                    descriptor = builtin_scheduler_registry().get(value)
                    assert descriptor is not None and descriptor.id in option_values
                else:
                    assert value in option_values
            if isinstance(widget, NumberWidget):
                assert type(value) in {int, float}
                numeric = float(value)
                minimum = widget.min
                maximum = widget.max
                assert minimum is None or numeric >= minimum
                assert maximum is None or numeric <= maximum


def test_safetensors_header_requires_contiguous_payload(tmp_path: Path) -> None:
    valid = {
        "tensor": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        "__metadata__": {"format": "pt"},
    }
    path = _write_safetensors(tmp_path / "valid.safetensors", valid, b"\x00\x00")
    header = receipts.read_safetensors_header(path.open("rb"), path.stat().st_size)
    assert header.tensors["tensor"]["dtype"] == "F16"

    invalid = {
        "tensor": {"dtype": "F16", "shape": [1], "data_offsets": [1, 3]},
        "__metadata__": {"format": "pt"},
    }
    path = _write_safetensors(tmp_path / "gap.safetensors", invalid, b"\x00\x00\x00")
    with pytest.raises(receipts.ReceiptError, match="contiguous"):
        receipts.read_safetensors_header(path.open("rb"), path.stat().st_size)


def test_role_contracts_distinguish_combined_sd15_from_unet_lora() -> None:
    combined = {
        "model.diffusion_model.input_blocks.0.0.weight": {
            "dtype": "F32",
            "shape": [320, 4, 3, 3],
            "data_offsets": [0, 46080],
        },
        "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": {
            "dtype": "F32",
            "shape": [49408, 768],
            "data_offsets": [46080, 151801856],
        },
        "first_stage_model.decoder.conv_out.weight": {
            "dtype": "F32",
            "shape": [3, 128, 3, 3],
            "data_offsets": [151801856, 151815680],
        },
    }
    lora = {
        "lora_unet_down_blocks_0_resnets_0_conv1.alpha": {
            "dtype": "F16",
            "shape": [],
            "data_offsets": [0, 2],
        },
        "lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight": {
            "dtype": "F16",
            "shape": [64, 320, 3, 3],
            "data_offsets": [2, 368642],
        },
        "lora_unet_down_blocks_0_resnets_0_conv1.lora_up.weight": {
            "dtype": "F16",
            "shape": [320, 64, 1, 1],
            "data_offsets": [368642, 409602],
        },
    }

    receipts.validate_role("combined-sd15-checkpoint", combined)
    receipts.validate_role("sd15-unet-lcm-lora", lora)
    with pytest.raises(receipts.ReceiptError):
        receipts.validate_role("combined-sd15-checkpoint", lora)
    with pytest.raises(receipts.ReceiptError):
        receipts.validate_role("sd15-unet-lcm-lora", combined)


def test_lora_role_refuses_text_encoder_and_incomplete_triplets() -> None:
    text_key = {
        "lora_te_text_model_encoder_layers_0.alpha": {
            "dtype": "F16",
            "shape": [],
            "data_offsets": [0, 2],
        }
    }
    incomplete = {
        "lora_unet_down_blocks_0_resnets_0_conv1.alpha": {
            "dtype": "F16",
            "shape": [],
            "data_offsets": [0, 2],
        }
    }

    with pytest.raises(receipts.ReceiptError, match="UNet-only"):
        receipts.validate_role("sd15-unet-lcm-lora", text_key)
    with pytest.raises(receipts.ReceiptError, match="triplet"):
        receipts.validate_role("sd15-unet-lcm-lora", incomplete)


def test_receipt_tool_has_no_model_or_gpu_execution_surface() -> None:
    source = Path(receipts.__file__).read_text("utf-8")

    assert "import torch" not in source
    assert "cuda" not in source.lower()
    assert "load_runtime" not in source
    assert "subprocess" not in source
    assert "requests" not in source
    assert "urllib" not in source
    assert os.linesep in source
