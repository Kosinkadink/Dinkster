"""ServingComposer enrollment for stage-6 execution dispatch."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from dinkster_engine import (
    EngineEvent,
    ExecutionError,
    ProviderResolutionError,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_memory import ReservationRequest
from dinkster_protocol import (
    WORKGROUP_CAPABILITY,
    WORKGROUP_DATA_PLANE_CAPABILITY,
    AbortWorkGroup,
    AttentionCapabilityEvidence,
    BeginWorkGroup,
    CancelWorkGroup,
    CommitWorkGroup,
    Invocation,
    InvocationEvent,
    InvocationResult,
    NodeError,
    PrepareReplica,
    ReleaseWorkGroup,
    ReplicaId,
    ReplicaReady,
    RunWorkUnit,
    WorkGroupCancelled,
    WorkGroupDefinition,
    WorkGroupMessage,
    WorkGroupPrepared,
    WorkGroupReleased,
    WorkUnitFailed,
    WorkUnitResult,
)
from dinkster_schema import (
    ComboOption,
    ComboWidget,
    NodeSchema,
    schema_signature,
)
from dinkster_server import PackInfo
from dinkster_values import (
    RESOURCE_ID_META_KEY,
    RESOURCE_OWNER_META_KEY,
    RESOURCE_PRODUCER_ARM_META_KEY,
    RESOURCES_META_KEY,
    Value,
    ValueMeta,
)
from dinkster_values.model import PyObjPayload
from dinkster_workers import (
    ReplicaEndpoint,
    load_manifest,
)

from dinkster.compose import (
    CompositionError,
    PackSpec,
    ServingComposer,
    compose_serving,
)
from tests.async_scenarios import run_scenario


@dataclass
class _WorkGroupAttempt:
    command: PrepareReplica
    prepared: bool = False
    active: bool = False
    activated: asyncio.Event = field(default_factory=asyncio.Event)
    run_allowed: asyncio.Event = field(default_factory=asyncio.Event)
    completion: asyncio.Future[str | None] | None = None
    invocation_id: str | None = None
    cancelled: bool = False

    def common(self) -> dict[str, object]:
        return {
            "worker": self.command.worker,
            "replica": self.command.replica,
            "group": self.command.group,
            "attempt": self.command.attempt,
            "device": self.command.device,
        }


class _TestWorkGroupHandler:
    def __init__(self) -> None:
        self._attempts: dict[tuple[object, object], _WorkGroupAttempt] = {}

    @staticmethod
    def _key(message: WorkGroupMessage) -> tuple[object, object]:
        return message.group, message.attempt

    def _attempt(self, message: WorkGroupMessage) -> _WorkGroupAttempt:
        attempt = self._attempts.get(self._key(message))
        if attempt is None:
            raise RuntimeError("workgroup command has no prepared attempt")
        prepared = attempt.command
        if (message.worker, message.replica, message.device) != (
            prepared.worker,
            prepared.replica,
            prepared.device,
        ):
            raise RuntimeError("workgroup command differs from the prepared rank")
        return attempt

    async def __call__(self, message: WorkGroupMessage) -> tuple[WorkGroupMessage, ...]:
        if type(message) is PrepareReplica:
            key = self._key(message)
            if key in self._attempts:
                raise RuntimeError("workgroup attempt was prepared twice")
            attempt = _WorkGroupAttempt(message)
            self._attempts[key] = attempt
            return (ReplicaReady(**attempt.common()),)  # type: ignore[arg-type]

        attempt = self._attempt(message)
        common = attempt.common()
        if type(message) is BeginWorkGroup:
            attempt.prepared = True
            return (WorkGroupPrepared(**common),)  # type: ignore[arg-type]
        if type(message) is CommitWorkGroup:
            attempt.active = True
            attempt.activated.set()
            return ()
        if type(message) is RunWorkUnit:
            attempt.completion = asyncio.get_running_loop().create_future()
            attempt.run_allowed.set()
            failure = await attempt.completion
            unit = {"unit": message.unit, "slot": message.slot}
            if failure is None:
                return (WorkUnitResult(**common, **unit),)  # type: ignore[arg-type]
            return (WorkUnitFailed(**common, **unit, reason=failure),)  # type: ignore[arg-type]
        if type(message) is CancelWorkGroup:
            attempt.cancelled = True
            attempt.activated.set()
            attempt.run_allowed.set()
            if attempt.completion is not None and not attempt.completion.done():
                attempt.completion.set_result(message.reason)
            return (WorkGroupCancelled(**common),)  # type: ignore[arg-type]
        if type(message) is AbortWorkGroup:
            attempt.activated.set()
            attempt.run_allowed.set()
            return ()
        if type(message) is ReleaseWorkGroup:
            self._attempts.pop(self._key(message), None)
            return (WorkGroupReleased(**common),)  # type: ignore[arg-type]
        raise RuntimeError(f"unsupported workgroup command {type(message).__name__}")

    async def before_invocation(self, invocation_id: str) -> None:
        attempts = tuple(self._attempts.values())
        if not attempts:
            return
        if len(attempts) != 1:
            raise RuntimeError("multiple workgroup attempts are active")
        attempt = attempts[0]
        await attempt.activated.wait()
        if not attempt.active:
            raise RuntimeError("workgroup attempt ended before invocation dispatch")
        await attempt.run_allowed.wait()
        attempt.invocation_id = invocation_id

    async def after_invocation(self, invocation_id: str, failure: str | None) -> None:
        attempt = next(
            (
                candidate
                for candidate in self._attempts.values()
                if candidate.invocation_id == invocation_id
            ),
            None,
        )
        if attempt is not None and attempt.completion is not None and not attempt.completion.done():
            attempt.completion.set_result(failure)

    def invocation_cancelled(self, invocation_id: str) -> bool:
        return any(
            attempt.invocation_id == invocation_id and attempt.cancelled
            for attempt in self._attempts.values()
        )

    def parent_manages_invocation_reservations(self) -> bool:
        return True


def _resource_value(resource_id: str, rank: int) -> Value:
    return Value(
        "test.resource",
        "same-logical-value",
        ValueMeta(
            {
                RESOURCE_ID_META_KEY: resource_id,
                RESOURCE_OWNER_META_KEY: f"owner-{rank}",
                RESOURCE_PRODUCER_ARM_META_KEY: f"arm-{rank}",
                RESOURCES_META_KEY: {"gpu": f"cuda:{rank}"},
            }
        ),
        PyObjPayload(f"resource-{rank}"),
    )


class _SingleJobFakeWorker:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.calls: list[Invocation] = []
        self.closed = False
        self.block = False
        self.error: NodeError | None = None
        self.exception: BaseException | None = None
        self.cancelled = False
        self.gate: asyncio.Event | None = None
        self.close_delay = 0.0

    async def invoke(self, invocation: Invocation, on_event=None) -> InvocationResult:
        self.calls.append(invocation)
        if on_event is not None:
            on_event(InvocationEvent("rank", {"rank": self.rank}))
        try:
            if self.block:
                await asyncio.Event().wait()
            if self.exception is not None:
                raise self.exception
            if self.gate is not None:
                await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.error is not None:
            return InvocationResult(error=self.error)
        return InvocationResult(
            outputs={"out": _resource_value(f"resource-{self.rank}", self.rank)}
        )

    async def close(self) -> None:
        self.closed = True
        await asyncio.sleep(self.close_delay)


class _WorkGroupFakeWorker(_SingleJobFakeWorker):
    workgroup_capabilities = frozenset({WORKGROUP_CAPABILITY, WORKGROUP_DATA_PLANE_CAPABILITY})
    schemas: Mapping[str, NodeSchema] = {}
    combo_choices: Mapping[str, tuple[str, ...]] = {}
    lazy_choice_ids: tuple[str, ...] = ()
    compat_skips: Mapping[str, object] = {}
    body_arms = None
    extension_contributions = None
    attention_capabilities = None
    attention_route_token = None

    def __init__(self, rank: int) -> None:
        super().__init__(rank)
        self.instance_token = f"worker-{rank}"
        self.handler = _TestWorkGroupHandler()
        self.unbound = 0
        self.release_failure = False
        self.run_transport_failure = False
        self.started = False
        self.start_error: BaseException | None = None

    @property
    def alive(self) -> bool:
        return self.started and not self.closed

    async def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    def bind_workgroup_endpoint(
        self,
        definition: WorkGroupDefinition,
        replica: ReplicaId,
    ) -> ReplicaEndpoint:
        queue: asyncio.Queue[WorkGroupMessage] = asyncio.Queue()

        async def send(message: WorkGroupMessage) -> None:
            if self.run_transport_failure and type(message) is RunWorkUnit:
                raise RuntimeError("run transport failed")
            replies = await self.handler(message)
            if self.release_failure and type(message) is ReleaseWorkGroup:
                raise RuntimeError("release transport failed")
            for reply in replies:
                await queue.put(reply)

        return ReplicaEndpoint.bind(definition, replica, send=send, receive=queue.get)

    def unbind_workgroup_endpoint(
        self,
        definition: WorkGroupDefinition,
        replica: ReplicaId,
    ) -> None:
        del definition, replica
        if not self.closed:
            assert not self.handler._attempts  # pyright: ignore[reportPrivateUsage]
        self.unbound += 1

    async def invoke(self, invocation: Invocation, on_event=None) -> InvocationResult:
        await self.handler.before_invocation(invocation.invocation_id)
        result = await super().invoke(invocation, on_event=on_event)
        await self.handler.after_invocation(
            invocation.invocation_id,
            None if result.error is None else result.error.message,
        )
        return result


class _Reservations:
    @asynccontextmanager
    async def reserve(self, requests: tuple[ReservationRequest, ...]):
        assert requests == ()
        yield


def _single_job_invocation(inputs: Mapping[str, Value] | None = None) -> Invocation:
    return Invocation(
        "invocation",
        "node",
        "test.node",
        {} if inputs is None else inputs,
        NodeSchema(node_type="test.node"),
    )


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


def test_reload_replaces_schema_owner_attribution(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "owner"
        manifest = _write_pack(root, "owner", "owned.echo", own_type="owned.other")
        composer = ServingComposer(worker_env=_env(root))
        try:
            await composer.add_pack(manifest)
            assert composer.composition.node_packs["owned.other"] == "owner"

            _write_pack(root, "owner", "owned.echo")
            await composer.reload_pack("owner")

            assert "owned.other" not in composer.composition.node_packs
            assert composer.composition.node_packs["owned.echo"] == "owner"
        finally:
            await composer.close()

    run_scenario(scenario())


def test_atomic_composition_refuses_schema_owner_without_provider(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "owner"
        manifest = _write_pack(
            root,
            "owner",
            "owned.echo",
            schema_only=True,
        )
        with pytest.raises(CompositionError, match="have no execution provider"):
            await compose_serving(
                (manifest,),
                include_default_packs=False,
                worker_env=_env(root),
            )

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


def test_incomplete_generation_removes_dependents_before_schema_owner(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_root = tmp_path / "owner"
        consumer_root = tmp_path / "consumer"
        owner_manifest = _write_pack(
            owner_root,
            "owner",
            "owned.echo",
            schema_only=True,
        )
        owner_manifest.write_text(
            owner_manifest.read_text().replace(
                "\n[pack.entry]",
                '\n[pack.capabilities]\n"owner.schemas" = "1.0.0"\n\n[pack.entry]',
            )
        )
        consumer_manifest = _write_pack(
            consumer_root,
            "consumer",
            "consumer.echo",
        )
        consumer_manifest.write_text(
            consumer_manifest.read_text().replace(
                "\n[pack.entry]",
                '\n[pack.requirements.capabilities]\n"owner.schemas" = ">=1,<2"\n\n[pack.entry]',
            )
        )
        composer = ServingComposer(worker_env=_env(owner_root, consumer_root))
        try:
            await composer.add_pack(owner_manifest)
            await composer.add_pack(consumer_manifest)
            assert composer.incomplete_generation_removals() == {
                "consumer": (),
                "owner": ("owned.echo",),
            }
            for pack in composer.incomplete_generation_removals():
                await composer.remove_pack(pack)
            assert composer.composition.packs == {}
        finally:
            await composer.close()

    run_scenario(scenario())


def test_incomplete_generation_rechecks_for_newly_stranded_schema_owner(tmp_path: Path) -> None:
    async def scenario() -> None:
        owner_a_root = tmp_path / "owner-a"
        owner_b_root = tmp_path / "owner-b"
        provider_root = tmp_path / "provider"
        owner_a = _write_pack(
            owner_a_root,
            "owner-a",
            "a.missing",
            schema_only=True,
        )
        owner_a.write_text(
            owner_a.read_text().replace(
                "\n[pack.entry]",
                '\n[pack.capabilities]\n"a.schemas" = "1.0.0"\n\n[pack.entry]',
            )
        )
        owner_b = _write_pack(
            owner_b_root,
            "owner-b",
            "b.echo",
            schema_only=True,
        )
        provider = _write_pack(
            provider_root,
            "provider",
            "b.echo",
            executes=("b.echo",),
            own_type="provider.echo",
        )
        provider.write_text(
            provider.read_text().replace(
                "\n[pack.entry]",
                '\n[pack.requirements.capabilities]\n"a.schemas" = ">=1,<2"\n\n[pack.entry]',
            )
        )
        composer = ServingComposer(worker_env=_env(owner_a_root, owner_b_root, provider_root))
        try:
            await composer.add_pack(owner_a)
            await composer.add_pack(owner_b)
            await composer.add_pack(provider)

            first = composer.incomplete_generation_removals()
            assert first == {"provider": (), "owner-a": ("a.missing",)}
            for pack in first:
                await composer.remove_pack(pack)

            second = composer.incomplete_generation_removals()
            assert second == {"owner-b": ("b.echo",)}
            for pack in second:
                await composer.remove_pack(pack)
            composer.validate_complete_generation()
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
