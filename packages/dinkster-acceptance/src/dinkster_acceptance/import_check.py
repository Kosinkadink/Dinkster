from __future__ import annotations

import importlib
import importlib.abc
import json
import sys


class _RejectTestImports(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> None:
        if fullname == "pytest" or fullname == "golden_files" or fullname.startswith("test_"):
            raise ModuleNotFoundError(f"acceptance closure imported test-only module {fullname!r}")
        return None


def check_imports() -> dict[str, object]:
    blocker = _RejectTestImports()
    sys.meta_path.insert(0, blocker)
    try:
        import torch
        from dinkster_compat_comfy.native_arm import GenerationModelSamplingFlux
        from dinkster_compat_comfy.native_residency import NativeRuntimeHandle
        from dinkster_inference_torch import BrownianTreeNoise, FluxRuntime
        from dinkster_workers import RemoteWorker, load_manifest

        from . import manifest_path
        from .golden_files import load_model_sampling_flux_golden
        from .model_sampling_flux import ModelSamplingFluxAcceptance

        manifest = load_manifest(manifest_path())
        module_name, _, attribute = manifest.nodes_entry.partition(":")
        nodes = getattr(importlib.import_module(module_name), attribute)
        if ModelSamplingFluxAcceptance not in nodes:
            raise ValueError("acceptance manifest does not register its acceptance node")
        golden = load_model_sampling_flux_golden()
        schema = ModelSamplingFluxAcceptance.define_schema()
        return {
            "status": "ok",
            "pack": manifest.name,
            "node": schema.node_type,
            "comfyui_commit": golden["comfyui_commit"],
            "torch": torch.__version__,
            "imports": [
                BrownianTreeNoise.__qualname__,
                FluxRuntime.__qualname__,
                GenerationModelSamplingFlux.__qualname__,
                NativeRuntimeHandle.__qualname__,
                RemoteWorker.__qualname__,
            ],
        }
    finally:
        sys.meta_path.remove(blocker)


def main() -> None:
    print(json.dumps(check_imports(), sort_keys=True))


if __name__ == "__main__":
    main()
