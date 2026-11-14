"""ServingComposer enrollment for stage-6 execution dispatch."""

from __future__ import annotations

import asyncio
import os
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_engine import (
    EngineEvent,
    ExecutionError,
    ProviderResolutionError,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_protocol import (
    AttentionCapabilityEvidence,
    Invocation,
)
from dinkster_schema import (
    ComboOption,
    ComboWidget,
    schema_signature,
)
from dinkster_server import PackInfo
from dinkster_workers import (
    load_manifest,
)

from dinkster.compose import (
    CompositionError,
    PackSpec,
    ServingComposer,
)
from tests.async_scenarios import run_scenario


def _attention_capabilities(*, kitchen: bool = False) -> AttentionCapabilityEvidence:
    return AttentionCapabilityEvidence(
        version=1,
        device_kind="cpu",
        device_sm=None,
        sdpa_torch_runtime="2.13.0",
        adapter_contract_revision="dinkster.attention-kernel.v1",
        available_policies=("sdpa", "comfy_kitchen_int8") if kitchen else ("sdpa",),
        provider_versions=(
            (("comfy-kitchen", "0.2.31"), ("torch", "2.13.0"))
            if kitchen
            else (("torch", "2.13.0"),)
        ),
    )


def _write_pack(
    root: Path,
    name: str,
    node_type: str,
    *,
    executes: tuple[str, ...] = (),
    own_type: str | None = None,
    mismatched: bool = False,
    body_arm: str | None = None,
    schema_only: bool = False,
    lazy: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    module = name.replace("-", "_") + "_nodes"
    extra_input = ', InputSpec("extra", STRING, default="")' if mismatched else ""
    lazy_input = ", lazy=True" if lazy else ""
    lazy_hook = (
        """
    @classmethod
    def check_lazy_status(cls, *, value: str, **kwargs: object) -> tuple[str, ...]:
        return ()
"""
        if lazy
        else ""
    )
    own_class = ""
    nodes = "Echo"
    arm_class = ""
    arm_nodes = ""
    if body_arm is not None:
        arm_class = """

class AltEcho(Echo):
    @classmethod
    async def execute(cls, *, value: str, **kwargs: object) -> Mapping[str, object]:
        await __import__("asyncio").sleep(0)
        from dinkster_workers import current_execution_context
        context = current_execution_context()
        identity = context.expected_execution_identity if context else "missing"
        fp8 = context.fp8_matmul if context else "missing"
        return cls.outputs(value="alt:" + value + ":" + str(identity) + ":" + str(fp8))
"""
        arm_nodes = f"\nARM_NODES = {{{body_arm!r}: [AltEcho]}}\n"
    if own_type is not None:
        own_class = f"""

class Own(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type={own_type!r},
            inputs=(InputSpec("value", STRING),),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    async def execute(cls, *, value: str) -> Mapping[str, object]:
        return cls.outputs(value="own:" + value)
"""
        nodes += ", Own"
    (root / f"{module}.py").write_text(
        f"""from collections.abc import Mapping

from dinkster_schema import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_values import TypeRegistry

STRING = TypeExpr.concrete("STRING")

def register_types(registry: TypeRegistry) -> None:
    pass

class Echo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type={node_type!r},
            inputs=(InputSpec("value", STRING{lazy_input}){extra_input},),
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    async def execute(cls, *, value: str, **kwargs: object) -> Mapping[str, object]:
        return cls.outputs(value={name!r} + ":" + value)
{lazy_hook}
{own_class}
{arm_class}
NODES = [{nodes}]
{arm_nodes}
"""
    )
    namespace = (
        own_type.partition(".")[0]
        if own_type
        else name
        if executes
        else node_type.partition(".")[0]
    )
    executes_line = (
        "executes = [" + ", ".join(f'"{item}"' for item in executes) + "]\n" if executes else ""
    )
    schema_only_line = f'schema-only = ["{node_type}"]\n' if schema_only else ""
    arms_table = f'\n[pack.arms]\n{body_arm} = ["{node_type}"]\n' if body_arm else ""
    arm_entry = f'arm_nodes = "{module}:ARM_NODES"\n' if body_arm else ""
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        f'[pack]\nname = "{name}"\npublisher = "dinkster"\n'
        f'namespaces = ["{namespace}"]\n'
        f"{executes_line}{schema_only_line}{arms_table}\n[pack.entry]\n"
        f'nodes = "{module}:NODES"\ntypes = "{module}:register_types"\n{arm_entry}'
    )
    return manifest


def _write_vision_provider_packs(
    root: Path, *, provider_choice: str = "vision.providers"
) -> tuple[Path, Path]:
    from dinkster_assets import digest_bytes

    owner_root = root / "owner"
    provider_root = root / "provider"
    owner_root.mkdir(parents=True)
    provider_root.mkdir(parents=True)
    common = """from collections.abc import Mapping

from dinkster_api.v1 import ComboWidget, InputSpec, Node, NodeSchema, OutputSpec, TypeExpr

STRING = TypeExpr.concrete("STRING")
COMBO = TypeExpr.concrete("COMBO")

class Vision(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="vision.process",
            inputs=(
                InputSpec("value", STRING),
                InputSpec(
                    "provider",
                    COMBO,
                    required=False,
                    widget=ComboWidget(remote_route="/api/choices/vision.providers"),
                    hidden=True,
                ),
            ),
            outputs=(OutputSpec("value", STRING),),
        )
"""
    (owner_root / "vision_owner_nodes.py").write_text(
        common
        + """
    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError("vision provider required")

NODES = [Vision]

def choices() -> dict[str, tuple[str, ...]]:
    return {"vision.providers": ()}
""",
        encoding="utf-8",
    )
    owner_manifest = owner_root / "dinkster-pack.toml"
    owner_manifest.write_text(
        """[pack]
name = "vision-owner"
publisher = "dinkster"
namespaces = ["vision"]
[pack.entry]
nodes = "vision_owner_nodes:NODES"
choices = "vision_owner_nodes:choices"
""",
        encoding="utf-8",
    )

    model_bytes = b"provider-model"
    assets = provider_root / "assets"
    assets.mkdir()
    (assets / "model.bin").write_bytes(model_bytes)
    (provider_root / "vision_provider_nodes.py").write_text(
        common
        + """
    @classmethod
    def execute(cls, *, value: str, **_inputs: object) -> Mapping[str, object]:
        from dinkster_api.v1 import declared_asset
        model = declared_asset("model").read_bytes().decode("utf-8")
        return cls.outputs(value=f"{model}:{value}")

NODES = [Vision]
""",
        encoding="utf-8",
    )
    provider_manifest = provider_root / "dinkster-pack.toml"
    provider_manifest.write_text(
        f"""[pack]
name = "depth-provider"
namespaces = []
executes = ["vision.process"]
[[pack.vision-providers]]
choice = "{provider_choice}"
node = "vision.process"
devices = ["cpu"]
dtypes = ["float32"]
batching = "batch"
artifacts = ["model"]
[[pack.assets]]
id = "model"
name = "Test model"
digest = "{digest_bytes(model_bytes)}"
file = "assets/model.bin"
[pack.entry]
nodes = "vision_provider_nodes:NODES"
""",
        encoding="utf-8",
    )
    return owner_manifest, provider_manifest


def test_vision_provider_manifest_populates_choices_and_routes_execution(tmp_path: Path) -> None:
    async def scenario() -> None:
        import sys

        from dinkster_assets import AssetVault, digest_bytes

        owner_manifest, provider_manifest = _write_vision_provider_packs(tmp_path)
        vault = AssetVault(tmp_path / "vault")
        with vault.writer(digest_bytes(b"provider-model")) as writer:
            writer.write(b"provider-model")
            writer.commit()
        worker_env = _env(owner_manifest.parent, provider_manifest.parent)
        worker_env["DINKSTER_ASSET_VAULT"] = str(vault.root)
        composer = ServingComposer(worker_env=worker_env)
        try:
            owner_delta = await composer.add_pack(owner_manifest)
            assert owner_delta.derived_choices == {}
            assert composer.composition.choices["vision.providers"] == ()
            owner_signature = schema_signature(composer.composition.schemas["vision.process"])

            provider_delta = await composer.add_pack(provider_manifest)
            assert provider_delta.derived_choices == {"vision.providers": ("depth-provider",)}
            assert composer.composition.choices["vision.providers"] == ("depth-provider",)
            assert (
                schema_signature(composer.composition.schemas["vision.process"]) == owner_signature
            )
            assert load_manifest(provider_manifest).vision_providers[0].artifacts == ("model",)
            assert set(
                composer.composition.asset_catalog.needs_for_nodes(
                    (),
                    provider_selections={("vision.process", "depth-provider")},
                )
            ) == {digest_bytes(b"provider-model")}
            assert (
                composer.composition.asset_catalog.needs_for_nodes(
                    (),
                    provider_selections={("vision.process", "missing")},
                )
                == {}
            )
            assert "vision_provider_nodes" not in sys.modules
            reloaded = await composer.reload_pack("depth-provider")
            assert reloaded.delta.derived_choices == {}
            assert composer.composition.choices["vision.providers"] == ("depth-provider",)

            events: list[EngineEvent] = []
            engine = composer.composition.make_engine(events.append)
            graph = Graph(
                nodes={
                    "n": GraphNode(
                        "vision.process",
                        {"value": "input", "provider": "depth-provider"},
                    )
                }
            )
            result = await engine.run(graph, ["n"])
            assert result.outputs["n"]["value"].resolve() == "provider-model:input"

            automatic = Graph(nodes={"n": GraphNode("vision.process", {"value": "automatic"})})
            automatic_result = await engine.run(automatic, ["n"])
            assert automatic_result.outputs["n"]["value"].resolve() == ("provider-model:automatic")
            finished = next(
                event
                for event in reversed(events)
                if event.kind == "node_finished" and event.node_id == "n"
            )
            assert finished.detail["provider"] == "depth-provider"
            assert finished.detail["pack"] == "depth-provider"
            assert finished.detail["worker"] == "local"

            unavailable = Graph(
                nodes={
                    "n": GraphNode(
                        "vision.process",
                        {"value": "input", "provider": "missing"},
                    )
                }
            )
            with pytest.raises(
                ProviderResolutionError,
                match="compatible vision-processing implementation is unavailable",
            ):
                await engine.run(unavailable, ["n"])

            removal = await composer.remove_pack("depth-provider")
            assert removal.derived_choices == {"vision.providers": ()}
            assert composer.composition.choices["vision.providers"] == ()
            with pytest.raises(
                ProviderResolutionError,
                match="compatible vision-processing implementation is unavailable",
            ):
                await composer.composition.make_engine(lambda _event: None).run(graph, ["n"])
        finally:
            await composer.close()

    run_scenario(scenario())


def test_vision_provider_choice_mismatch_fails_atomically(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_manifest, provider_manifest = _write_vision_provider_packs(
            tmp_path, provider_choice="vision.other"
        )
        composer = ServingComposer(worker_env=_env(owner_manifest.parent, provider_manifest.parent))
        try:
            await composer.add_pack(owner_manifest)
            with pytest.raises(CompositionError, match="does not match.*provider input"):
                await composer.add_pack(provider_manifest)
            assert set(composer._records) == {"vision-owner"}
            assert composer.composition.choices["vision.providers"] == ()
        finally:
            await composer.close()

    run_scenario(scenario())


def _write_generation_provider_packs(root: Path) -> tuple[Path, Path, Path]:
    owner_root = root / "generation-owner"
    native_root = root / "generation-native"
    external_root = root / "generation-external"
    for directory in (owner_root, native_root, external_root):
        directory.mkdir(parents=True)
    schema = """from collections.abc import Mapping

from dinkster_api.v1 import ComboWidget, InputSpec, Node, NodeSchema, OutputSpec, TypeExpr

STRING = TypeExpr.concrete("STRING")
COMBO = TypeExpr.concrete("COMBO")

class Generate(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="fixture.generate",
            inputs=(
                InputSpec("clip", STRING, required=False, lazy=True),
                InputSpec(
                    "provider",
                    COMBO,
                    required=False,
                    widget=ComboWidget(remote_route="/api/choices/fixture.generation.providers"),
                    hidden=True,
                ),
                InputSpec("value", STRING),
            ),
            outputs=(OutputSpec("value", STRING),),
        )
"""
    (owner_root / "generation_owner.py").write_text(
        schema
        + """
    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError("generation provider required")

class Fail(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="fixture.fail",
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        raise RuntimeError("lazy clip branch executed")

class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="fixture.source",
            outputs=(OutputSpec("value", STRING),),
        )

    @classmethod
    def execute(cls, **_inputs: object) -> Mapping[str, object]:
        return cls.outputs(value="resident")

NODES = [Generate, Fail, Source]

def choices() -> dict[str, tuple[str, ...]]:
    return {"fixture.generation.providers": ()}
""",
        encoding="utf-8",
    )
    owner_manifest = owner_root / "dinkster-pack.toml"
    owner_manifest.write_text(
        """[pack]
name = "generation-owner"
publisher = "dinkster"
namespaces = ["fixture"]
schema-only = ["fixture.generate"]
[pack.capabilities]
"dinkster.generation.schemas" = "1.0.0"
[pack.entry]
nodes = "generation_owner:NODES"
choices = "generation_owner:choices"
""",
        encoding="utf-8",
    )

    (native_root / "generation_native.py").write_text(
        schema
        + """
    @classmethod
    def check_lazy_status(
        cls,
        *,
        clip: object | None = None,
        provider: object | None = None,
        **_inputs: object,
    ) -> tuple[str, ...]:
        return ("clip",) if provider is None and clip is None else ()

    @classmethod
    def execute(cls, *, clip: str, value: str, **_inputs: object) -> Mapping[str, object]:
        return cls.outputs(value=f"native:{clip}:{value}")

NODES = [Generate]
""",
        encoding="utf-8",
    )
    native_manifest = native_root / "dinkster-pack.toml"
    native_manifest.write_text(
        """[pack]
name = "generation-native"
namespaces = []
executes = ["fixture.generate"]
[pack.entry]
nodes = "generation_native:NODES"
""",
        encoding="utf-8",
    )

    (external_root / "generation_external.py").write_text(
        schema
        + """
    @classmethod
    def check_lazy_status(cls, **_inputs: object) -> tuple[str, ...]:
        return ()

    @classmethod
    def execute(cls, *, value: str, **_inputs: object) -> Mapping[str, object]:
        return cls.outputs(value=f"external:{value}")

NODES = [Generate]
""",
        encoding="utf-8",
    )
    external_manifest = external_root / "dinkster-pack.toml"
    external_manifest.write_text(
        """[pack]
name = "generation-external"
namespaces = []
executes = ["fixture.generate"]
[[pack.generation-providers]]
choice = "fixture.generation.providers"
node = "fixture.generate"
[pack.entry]
nodes = "generation_external:NODES"
""",
        encoding="utf-8",
    )
    return owner_manifest, native_manifest, external_manifest


def test_generation_provider_is_explicit_and_suppresses_lazy_native_inputs(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        owner, native, external = _write_generation_provider_packs(tmp_path)
        composer = ServingComposer(worker_env=_env(owner.parent, native.parent, external.parent))
        try:
            await composer.add_pack(owner)
            delta = await composer.add_pack(external)
            await composer.add_pack(native)
            assert delta.derived_choices == {
                "fixture.generation.providers": ("generation-external",)
            }
            provider_input = composer.composition.schemas["fixture.generate"].input("provider")
            assert provider_input is not None
            assert provider_input.widget == ComboWidget(
                options=(
                    ComboOption("builtin", "Built-in"),
                    ComboOption("generation-external", "generation-external"),
                )
            )
            events: list[EngineEvent] = []
            engine = composer.composition.make_engine(events.append)
            native_graph = Graph(
                nodes={
                    "source": GraphNode("fixture.source"),
                    "generate": GraphNode(
                        "fixture.generate",
                        {"clip": Link("source", "value"), "value": "prompt"},
                    ),
                }
            )
            native_result = await engine.run(native_graph, ["generate"])
            assert native_result.outputs["generate"]["value"].resolve() == (
                "native:resident:prompt"
            )
            native_finished = next(
                event
                for event in events
                if event.kind == "node_finished" and event.node_id == "generate"
            )
            assert native_finished.detail["provider"] == "builtin"
            builtin_graph = Graph(
                nodes={
                    "source": GraphNode("fixture.source"),
                    "generate": GraphNode(
                        "fixture.generate",
                        {
                            "clip": Link("source", "value"),
                            "provider": "builtin",
                            "value": "builtin prompt",
                        },
                    ),
                }
            )
            before_builtin = len(events)
            builtin_result = await engine.run(builtin_graph, ["generate"])
            assert builtin_result.outputs["generate"]["value"].resolve() == (
                "native:resident:builtin prompt"
            )
            builtin_finished = next(
                event
                for event in events[before_builtin:]
                if event.kind == "node_finished" and event.node_id == "generate"
            )
            assert builtin_finished.detail["provider"] == "builtin"

            external_graph = Graph(
                nodes={
                    "fail": GraphNode("fixture.fail"),
                    "generate": GraphNode(
                        "fixture.generate",
                        {
                            "clip": Link("fail", "value"),
                            "provider": "generation-external",
                            "value": "prompt",
                        },
                    ),
                }
            )
            external_result = await engine.run(external_graph, ["generate"])
            assert external_result.outputs["generate"]["value"].resolve() == "external:prompt"

            removal = await composer.remove_pack("generation-external")
            assert removal.derived_choices == {"fixture.generation.providers": ()}
            provider_input = composer.composition.schemas["fixture.generate"].input("provider")
            assert provider_input is not None
            assert provider_input.widget == ComboWidget(
                options=(ComboOption("builtin", "Built-in"),)
            )
            with pytest.raises(ExecutionError, match="unavailable generation provider"):
                await composer.composition.make_engine(lambda _event: None).run(
                    external_graph, ["generate"]
                )
        finally:
            await composer.close()

    run_scenario(scenario())


def _env(*roots: Path) -> dict[str, str]:
    return {"PYTHONPATH": os.pathsep.join(str(root) for root in roots)}


def _graph(node_type: str, value: str = "x") -> Graph:
    return Graph(nodes={"n": GraphNode(node_type, {"value": value})})


@pytest.mark.parametrize("shared_group", [True, False])
def test_crosspack_resident_models_survive_group_reload(tmp_path: Path, shared_group: bool) -> None:
    async def scenario() -> None:
        manifests = []
        for name, entry in (("source", "SOURCE_NODES"), ("merge", "MERGE_NODES")):
            root = tmp_path / name
            root.mkdir()
            manifest = root / "dinkster-pack.toml"
            manifest.write_text(
                f'[pack]\nname = "{name}"\npublisher = "dinkster"\n'
                f'namespaces = ["{name}"]\n'
                f'[pack.entry]\nnodes = "crosspack_nodes:{entry}"\n'
                'types = "dinkster_inference_wire:register_inference_types"\n'
            )
            manifests.append(manifest)
        composer = ServingComposer(worker_env=_env(Path(__file__).parent))
        try:
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest,
                        worker_group="crosspack-models" if shared_group else None,
                        group_manifests=tuple(manifests) if shared_group else (),
                    )
                )
            graph = Graph(
                nodes={
                    "a": GraphNode("source.model", {"weight": 10.0}),
                    "b": GraphNode("source.model", {"weight": -4.0}),
                    "m": GraphNode(
                        "merge.simple",
                        {"a": Link("a", "model"), "b": Link("b", "model"), "ratio": 0.25},
                    ),
                    "w": GraphNode("source.weight", {"model": Link("m", "model")}),
                    "mixed": GraphNode(
                        "merge.simple",
                        {"a": Link("a", "model"), "b": Link("m", "model"), "ratio": 0.25},
                    ),
                    "mixed_weight": GraphNode("source.weight", {"model": Link("mixed", "model")}),
                }
            )
            if not shared_group:
                with pytest.raises(ExecutionError, match="unknown producer arm"):
                    await composer.composition.make_engine(lambda event: None).run(graph, ["w"])
                return
            for reload in (False, True):
                if reload:
                    await composer.reload_pack("source")
                result = await composer.composition.make_engine(lambda event: None).run(
                    graph, ["w", "mixed_weight"]
                )
                assert result.outputs["w"]["weight"].resolve() == -0.5
                assert result.outputs["mixed_weight"]["weight"].resolve() == 2.125
        finally:
            await composer.close()

    run_scenario(scenario())


@pytest.mark.parametrize("optional_execution", [False, True])
def test_schema_only_owner_routes_only_after_provider_composes(
    tmp_path: Path, optional_execution: bool
) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        provider_root = tmp_path / "provider"
        owner_manifest = _write_pack(
            owner_root,
            "owner",
            "owned.echo",
            schema_only=True,
        )
        provider_manifest = _write_pack(
            provider_root,
            "provider",
            "owned.echo",
            executes=("owned.echo",),
            body_arm="alt",
        )
        composer = ServingComposer(worker_env=_env(owner_root, provider_root))
        try:
            owner_delta = await composer.add_pack(
                PackSpec(
                    manifest=owner_manifest,
                    optional_execution=("owned.echo",) if optional_execution else (),
                )
            )
            assert owner_delta.schemas == {}
            assert owner_delta.execution_arms == {}
            assert composer._topology["owned.echo"] == ()
            assert "owned.echo" not in composer._routing._routes
            with (
                nullcontext()
                if optional_execution
                else pytest.raises(CompositionError, match="have no execution provider")
            ):
                composer.validate_complete_generation()
            assert composer.incomplete_generation_removals() == (
                {} if optional_execution else {"owner": ("owned.echo",)}
            )
            with pytest.raises(RuntimeError, match="has no execution provider"):
                await composer._plan_execution(
                    "owned.echo", composer.composition.schemas["owned.echo"], {}
                )

            provider_delta = await composer.add_pack(provider_manifest)
            assert provider_delta.schemas == {
                "owned.echo": composer.composition.schemas["owned.echo"]
            }
            assert provider_delta.node_packs == {"owned.echo": "owner"}
            arms = composer._topology["owned.echo"]
            assert [arm.name for arm in arms] == ["provider", "provider@alt"]
            assert composer.composition.node_packs["owned.echo"] == "owner"
            composer.validate_complete_generation()

            engine = composer.composition.make_engine(lambda _event: None)
            result = await engine.run(_graph("owned.echo", "ok"), ["n"])
            assert result.outputs["n"]["value"].resolve() == "provider:ok"

            removal = await composer.remove_pack("provider")
            assert removal.removed_types == ("owned.echo",)
            assert composer._topology["owned.echo"] == ()
            assert "owned.echo" not in composer._routing._routes
            with (
                nullcontext()
                if optional_execution
                else pytest.raises(CompositionError, match="have no execution provider")
            ):
                composer.validate_complete_generation()
            assert composer.incomplete_generation_removals() == (
                {} if optional_execution else {"owner": ("owned.echo",)}
            )
        finally:
            await composer.close()

    run_scenario(scenario())


def test_schema_only_owner_reload_adds_and_retracts_its_body(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "owner"
        manifest = _write_pack(
            root,
            "owner",
            "owned.echo",
            schema_only=True,
        )
        composer = ServingComposer(worker_env=_env(root))
        try:
            await composer.add_pack(manifest)
            assert not composer._routing.has_route("owned.echo")

            _write_pack(root, "owner", "owned.echo")
            await composer.reload_pack("owner")
            assert composer._routing.has_route("owned.echo")
            engine = composer.composition.make_engine(lambda _event: None)
            result = await engine.run(_graph("owned.echo", "ok"), ["n"])
            assert result.outputs["n"]["value"].resolve() == "owner:ok"

            _write_pack(root, "owner", "owned.echo", schema_only=True)
            await composer.reload_pack("owner")
            assert not composer._routing.has_route("owned.echo")

            removed = await composer.remove_pack("owner")
            assert removed.removed_types == ()
            assert "owned.echo" not in composer.composition.schemas
        finally:
            await composer.close()

    run_scenario(scenario())


def test_claim_validation_is_atomic_for_mismatch_missing_and_core(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        owner_manifest = _write_pack(owner_root, "owner", "owned.echo")
        roots = [owner_root]
        composer = ServingComposer(worker_env={})
        try:
            composer._worker_env = _env(*roots)
            await composer.add_pack(owner_manifest)
            cases = (
                ("mismatch", "owned.echo", True, "schema signature"),
                ("missing", "absent.echo", False, "no owning pack"),
                ("coreclaim", "std.math.add_ints", False, "no owning pack"),
            )
            for name, claim, mismatch, message in cases:
                root = tmp_path / name
                roots.append(root)
                composer._worker_env = _env(*roots)
                manifest = _write_pack(root, name, claim, executes=(claim,), mismatched=mismatch)
                before = composer._topology
                with pytest.raises(CompositionError, match=message):
                    await composer.add_pack(manifest)
                assert composer._topology is before
                assert name not in composer.composition.packs
                assert name not in composer._records
        finally:
            await composer.close()

    run_scenario(scenario())


def test_owner_dependencies_and_alternative_reload_are_revalidated(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        arm_root = tmp_path / "arm"
        owner_manifest = _write_pack(owner_root, "owner", "owned.echo")
        arm_manifest = _write_pack(
            arm_root,
            "alternative",
            "owned.echo",
            executes=("owned.echo",),
        )
        composer = ServingComposer(worker_env=_env(owner_root, arm_root))
        try:
            await composer.add_pack(owner_manifest)
            await composer.add_pack(arm_manifest)
            with pytest.raises(CompositionError, match="alternative"):
                await composer.remove_pack("owner")

            _write_pack(owner_root, "owner", "owned.echo", mismatched=True)
            with pytest.raises(CompositionError, match="alternative"):
                await composer.reload_pack("owner")
            assert not composer._records["owner"].worker.schemas["owned.echo"].inputs[1:]

            _write_pack(
                arm_root,
                "alternative",
                "owned.echo",
                executes=("owned.echo",),
                mismatched=True,
            )
            with pytest.raises(CompositionError, match="schema signature"):
                await composer.reload_pack("alternative")
            assert len(composer._topology["owned.echo"]) == 2

            _write_pack(
                arm_root,
                "alternative",
                "owned.echo",
                executes=("owned.echo",),
            )
            old_topology = composer._topology
            old_route = composer._routing._routes["owned.echo"]
            await composer.reload_pack("alternative")
            reloaded_topology = composer._topology
            assert reloaded_topology is not old_topology
            assert len(old_topology["owned.echo"]) == 2
            assert len(reloaded_topology["owned.echo"]) == 2

            await composer.remove_pack("alternative")
            assert composer._topology is not reloaded_topology
            assert len(reloaded_topology["owned.echo"]) == 2
            assert [arm.name for arm in composer._topology["owned.echo"]] == ["owner"]
            assert composer._routing._routes["owned.echo"] is not old_route
            engine = composer.composition.make_engine(lambda event: None)
            result = await engine.run(_graph("owned.echo", "still"), ["n"])
            assert result.outputs["n"]["value"].resolve() == "owner:still"
        finally:
            await composer.close()

    run_scenario(scenario())


def test_execution_identity_and_dead_owner_fail_loudly(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        arm_root = tmp_path / "arm"
        manifest = _write_pack(owner_root, "owner", "owned.echo")
        arm_manifest = _write_pack(
            arm_root,
            "alternative",
            "owned.echo",
            executes=("owned.echo",),
        )
        digest = "blake3:" + "ab" * 32
        spec = PackSpec(
            manifest=manifest,
            packs={"owner": PackInfo(display_name="Owner", artifact_digest=digest)},
        )
        composer = ServingComposer(worker_env=_env(owner_root, arm_root))
        try:
            await composer.add_pack(spec)
            await composer.add_pack(arm_manifest)
            worker = composer._records["owner"].worker
            alternative = composer._records["alternative"].worker
            token = worker.instance_token
            assert token is not None
            selection = await composer._plan_execution(
                "owned.echo", composer.composition.schemas["owned.echo"], {}
            )
            assert selection is not None and selection.cache_tag == digest

            assert worker._proc is not None
            worker._proc.kill()
            await worker._proc.wait()
            for _ in range(100):
                if not worker.alive:
                    break
                await asyncio.sleep(0.01)
            assert alternative.alive
            assert [arm.name for arm in composer._topology["owned.echo"]] == [
                "owner",
                "alternative",
            ]
            assert not composer._owner_alive(token)
            with pytest.raises(RuntimeError, match="not live"):
                await composer._plan_execution(
                    "owned.echo", composer.composition.schemas["owned.echo"], {}
                )
            engine = composer.composition.make_engine(lambda event: None)
            with pytest.raises(ExecutionError):
                await engine.run(_graph("owned.echo"), ["n"])
        finally:
            await composer.close()

    run_scenario(scenario())


def test_same_session_body_arm_routes_without_selector_bleed(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "same"
        manifest = _write_pack(root, "same", "same.echo", body_arm="alt")
        composer = ServingComposer(worker_env=_env(root))
        try:
            await composer.add_pack(manifest)
            arms = composer._topology["same.echo"]
            assert [arm.name for arm in arms] == ["same", "same@alt"]
            assert arms[0].domain is arms[1].domain
            token = composer._records["same"].worker.instance_token
            assert token is not None
            assert composer._live_token_owners(composer._topology) == {token: arms[0].domain}
            assert composer._records["same"].worker.body_arms == {"alt": ("same.echo",)}
            loaded_manifest = composer._records["same"].worker._manifest
            session = composer._records["same"].worker._session
            session._body_arms = None
            with pytest.raises(CompositionError, match="omitted required bodyArms"):
                composer._validate_body_arms(loaded_manifest, composer._records["same"].worker)
            session._body_arms = {"alt": ("same.other",)}
            with pytest.raises(CompositionError, match="does not exactly match"):
                composer._validate_body_arms(loaded_manifest, composer._records["same"].worker)
            session._body_arms = {"alt": ("same.echo",)}
            route = composer._routing._routes["same.echo"]
            schema = composer.composition.schemas["same.echo"]
            registry = composer.composition._registry

            def invocation(index: int, target: str) -> Invocation:
                return Invocation(
                    invocation_id=f"inv-{index}",
                    node_id=f"n-{index}",
                    node_type="same.echo",
                    inputs={"value": registry.wrap("STRING", str(index))},
                    effective_schema=schema,
                    executor=target,
                    expected_execution_identity=f"identity-{index}",
                    fp8_matmul=True,
                )

            results = await asyncio.gather(
                *(
                    route.invoke(invocation(i, "same" if i % 2 == 0 else "same@alt"))
                    for i in range(20)
                )
            )
            assert [result.outputs["value"].resolve() for result in results if result.outputs] == [
                ("same:" + str(i) if i % 2 == 0 else f"alt:{i}:identity-{i}:True")
                for i in range(20)
            ]

            underlying = composer._records["same"].worker
            unknown = await underlying.invoke(replace(invocation(30, "same@alt"), arm="missing"))
            assert unknown.error is not None and "unknown body arm" in unknown.error.message
            survivor = await route.invoke(invocation(31, "same@alt"))
            assert survivor.outputs is not None
            assert survivor.outputs["value"].resolve() == "alt:31:identity-31:True"

            engine = composer.composition.make_engine(lambda event: None)
            result = await engine.run(_graph("same.echo", "default"), ["n"])
            assert result.outputs["n"]["value"].resolve() == "same:default"

            old_token = underlying.instance_token
            old_domain = composer._topology["same.echo"][0].domain
            await composer.reload_pack("same")
            new_arms = composer._topology["same.echo"]
            assert new_arms[0].domain is new_arms[1].domain
            assert new_arms[0].domain is not old_domain
            assert old_token is not None and not composer._owner_alive(old_token)
        finally:
            await composer.close()

    run_scenario(scenario())
