"""Test-only nodes proving pre-torch aimdo bootstrap and child-local arming."""

from __future__ import annotations

import gc
import importlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from dinkster_api.v1 import CORE_INT, InputSpec, Node, NodeSchema, OutputSpec, TypeExpr

INT = TypeExpr.concrete(CORE_INT)


class AimdoBootstrapProbe(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="aimdo.bootstrap_probe",
            display_name="Aimdo Bootstrap Probe",
            category="test",
            outputs=(OutputSpec("initialized", INT),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        # Importing this entry performs the real ComfyUI bootstrap and imports
        # torch before the probe, exactly as the production compat worker does.
        importlib.import_module("dinkster_compat_comfy.entry")
        # Resolves only in the child worker: the live test runs this pack
        # under the ComfyUI interpreter with every packages/*/src on
        # PYTHONPATH. Root pyright excludes packages/dinkster-inference-torch
        # (torch-free root venv), so it cannot resolve this import.
        from dinkster_inference_torch import probe_aimdo  # pyright: ignore[reportMissingImports]

        return cls.outputs(initialized=int(probe_aimdo().initialized))


class ComfyArgsProbe(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="comfy.args_probe",
            display_name="Comfy Args Probe",
            category="test",
            outputs=(OutputSpec("preview_size", INT),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        importlib.import_module("dinkster_compat_comfy.entry")
        cli_args = importlib.import_module("comfy.cli_args")
        return cls.outputs(preview_size=int(cli_args.args.preview_size))


class AimdoArmProbe(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="aimdo.arm_probe",
            display_name="Aimdo Arm Probe",
            category="test",
            outputs=(OutputSpec("ready", INT),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        # Host argv caused the pre-torch bootstrap before this entry loaded.
        # Import torch now, then verify the post-torch public activation API.
        importlib.import_module("torch")
        from dinkster_inference_torch.aimdo_activation import (  # pyright: ignore[reportMissingImports]
            ensure_visible_aimdo_devices,
        )

        armed = os.environ.get("DINKSTER_AIMDO_ARM") == "on"
        return cls.outputs(ready=int(armed and ensure_visible_aimdo_devices()))


class AimdoHeadroomProbe(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="aimdo.headroom_probe",
            display_name="Aimdo Headroom Probe",
            category="test",
            inputs=(InputSpec("nonce", INT),),
            outputs=(OutputSpec("native_headroom", INT), OutputSpec("pending", INT)),
        )

    @classmethod
    def execute(cls, nonce: int) -> Mapping[str, object]:
        del nonce
        import ctypes

        import torch  # pyright: ignore[reportMissingImports]
        from dinkster_aimdo import control  # pyright: ignore[reportMissingImports]
        from dinkster_compat_comfy.native_arm import _aimdo_mechanism_factory

        mechanism, fallback_reason = _aimdo_mechanism_factory("on", torch.device("cuda:0"), torch)
        if mechanism is None:
            raise RuntimeError(f"aimdo mechanism did not activate: {fallback_reason}")
        library = control.lib
        if library is None:
            raise RuntimeError("aimdo native library did not initialize")
        native_headroom = ctypes.c_int64.in_dll(library, "simple_vram_headroom").value
        pending = int("DINKSTER_AIMDO_HEADROOM_TARGET" in os.environ)
        return cls.outputs(native_headroom=native_headroom, pending=pending)


class AimdoNativeProof(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="aimdo.native_proof",
            display_name="Aimdo Native Proof",
            category="test",
            outputs=(
                OutputSpec("on_mechanisms", INT),
                OutputSpec("on_demand_paged", INT),
                OutputSpec("off_mechanisms", INT),
                OutputSpec("off_demand_paged", INT),
                OutputSpec("embeddings_equal", INT),
                OutputSpec("pooled_equal", INT),
                OutputSpec("telemetry_free", INT),
                OutputSpec("telemetry_total", INT),
                OutputSpec("raw_free", INT),
                OutputSpec("raw_total", INT),
                OutputSpec("allocator_reclaimable_bytes", INT),
                OutputSpec("dynamic_evictable_bytes", INT),
                OutputSpec("dynamic_pinned_bytes", INT),
                OutputSpec("resident_bytes", INT),
                OutputSpec("activation_headroom", INT),
            ),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        import torch  # pyright: ignore[reportMissingImports]
        from dinkster_compat_comfy.devices import vram_telemetry_snapshot
        from dinkster_compat_comfy.native_arm import _build_runtime_handle
        from dinkster_inference import (
            FLOAT32,
            ReconstructionRecipe,
            RuntimeKnobs,
            WeightSourceBinding,
            WeightSourceRef,
            default_diffusion_dtype,
            load_safetensors_header,
            plan_native,
            runtime_component_identity,
        )
        from dinkster_inference_torch.wiring import (  # pyright: ignore[reportMissingImports]
            load_runtime,
        )

        checkpoint = Path(os.environ["DINKSTER_AIMDO_TEST_CHECKPOINT"])

        def run(
            mode: str,
        ) -> tuple[
            list[bool],
            object,
            object,
            tuple[int, int, int, int, int, int, int, int, int],
        ]:
            os.environ["DINKSTER_AIMDO_ARM"] = mode
            source = load_safetensors_header(checkpoint)
            plan = plan_native(source)
            recipe = ReconstructionRecipe(
                sources=(
                    WeightSourceBinding(
                        "checkpoint",
                        WeightSourceRef(
                            os.environ["DINKSTER_AIMDO_TEST_CHECKPOINT_DIGEST"],
                            checkpoint.name,
                            checkpoint.stat().st_size,
                        ),
                    ),
                ),
                family_id=plan.family.id,
                component_identity=runtime_component_identity(
                    plan.family.id, plan.identity_components
                ),
                knobs=RuntimeKnobs(
                    diffusion_dtype=default_diffusion_dtype(plan.family.id).name,
                    text_dtype=FLOAT32.name,
                    vae_dtype=FLOAT32.name,
                    fp8_matmul=False,
                ),
            )
            runtime = load_runtime(checkpoint=source)
            handle = _build_runtime_handle(runtime, torch, recipe=recipe)
            demand_paged = [mechanism.demand_paged for mechanism in handle.mechanisms]
            telemetry = (0, 0, 0, 0, 0, 0, 0, 0, 0)
            try:
                with handle.stage("text"):
                    output = runtime.encode_text("a photo of a cat")
                embeddings = output.embeddings.detach().cpu().clone()
                pooled = None if output.pooled is None else output.pooled.detach().cpu().clone()
                if mode == "on":
                    inference_torch = importlib.import_module("dinkster_inference_torch")
                    device = torch.device("cuda:0")
                    measured = vram_telemetry_snapshot()["vram:cuda:0"]
                    resident = int(inference_torch.aimdo_resident_bytes(device))
                    assert measured.driver_free_bytes is not None
                    assert measured.allocator_reclaimable_bytes is not None
                    assert measured.dynamic_evictable_bytes is not None
                    assert measured.dynamic_pinned_bytes is not None
                    budget = int(os.environ["DINKSTER_ACCELERATOR_BUDGETS"].split("=", 1)[1])
                    telemetry = (
                        measured.free_bytes,
                        measured.total_bytes,
                        measured.driver_free_bytes,
                        measured.total_bytes,
                        measured.allocator_reclaimable_bytes,
                        measured.dynamic_evictable_bytes,
                        measured.dynamic_pinned_bytes,
                        resident,
                        max(0, measured.total_bytes - budget),
                    )
            finally:
                handle.coordinator.terminal_release(handle)
            del handle, runtime, output
            gc.collect()
            torch.cuda.empty_cache()
            return demand_paged, cast("Any", embeddings), cast("Any", pooled), telemetry

        if os.environ.get("DINKSTER_AIMDO_ARM") != "on":
            raise RuntimeError("native aimdo proof worker was not armed by host argv")
        on_flags, on_embeddings, on_pooled, telemetry = run("on")
        off_flags, off_embeddings, off_pooled, _off_telemetry = run("off")
        (
            telemetry_free,
            telemetry_total,
            raw_free,
            raw_total,
            allocator_reclaimable_bytes,
            dynamic_evictable_bytes,
            dynamic_pinned_bytes,
            resident_bytes,
            activation_headroom,
        ) = telemetry
        return cls.outputs(
            on_mechanisms=len(on_flags),
            on_demand_paged=sum(on_flags),
            off_mechanisms=len(off_flags),
            off_demand_paged=sum(off_flags),
            embeddings_equal=int(
                torch.equal(cast("Any", on_embeddings), cast("Any", off_embeddings))
            ),
            pooled_equal=int(
                on_pooled is not None
                and off_pooled is not None
                and torch.equal(cast("Any", on_pooled), cast("Any", off_pooled))
            ),
            telemetry_free=telemetry_free,
            telemetry_total=telemetry_total,
            raw_free=raw_free,
            raw_total=raw_total,
            allocator_reclaimable_bytes=allocator_reclaimable_bytes,
            dynamic_evictable_bytes=dynamic_evictable_bytes,
            dynamic_pinned_bytes=dynamic_pinned_bytes,
            resident_bytes=resident_bytes,
            activation_headroom=activation_headroom,
        )


NODES = (
    AimdoBootstrapProbe,
    ComfyArgsProbe,
    AimdoArmProbe,
    AimdoHeadroomProbe,
    AimdoNativeProof,
)
