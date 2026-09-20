"""Pack combo choice lists: manifest entry -> worker enumeration -> hello ->
composition -> /api/choices/{id} (the remote half of the wire-v9 COMBO
contract).

What this proves: a pack's ``[pack.entry] choices`` callable is validated
in the worker host (grammar-valid ids, string values, dedupe), rides the
hello like schemas do, is owned exclusively per pack under its namespace
claims (the same rules as node types, including reload/remove atomicity),
and serves as a plain JSON array behind /api/choices/{id} - UI vocabulary,
never schema or execution identity.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_compat_comfy import ResidentPool
from dinkster_engine import Engine, EventListener
from dinkster_memory import FullReleaseResult
from dinkster_protocol import CompatGateDiagnostic, KeyedContribution, SamplerRegistrySnapshot
from dinkster_schema import ComboWidget, InputSpec, NodeSchema, TypeExpr
from dinkster_server import STATE_KEY, ChoiceOwnerGone, PackInfo, create_app
from dinkster_values import TypeRegistry
from dinkster_workers import InProcessWorker, ManifestError
from dinkster_workers.doctor import prepare_catalog
from dinkster_workers.host import load_choices, load_skips
from dinkster_workers.manifest import load_manifest
from dinkster_workers.session import (
    WorkerDied,
    _combo_choices_from_hello,
    _compat_skips_from_hello,
    _lazy_choice_ids_from_hello,
)

from dinkster.compose import CompositionError, PackSpec, ServingComposer, _lazy_choice_fetcher
from dinkster.reload_api import apply_reload, apply_remove

TESTS_DIR = Path(__file__).parent
WORKER_ENV = {"PYTHONPATH": str(TESTS_DIR)}
DEV_PACK_MANIFEST = TESTS_DIR.parent / "packages" / "dinkster-nodes-dev" / "dinkster-pack.toml"


def diagnostic(source_node: str, reason: str) -> CompatGateDiagnostic:
    return CompatGateDiagnostic(
        code="compat.dynamic.unsupported",
        source_node=source_node,
        reason=reason,
        source_generation="v1",
        path_kind="declared",
        input_id="value",
        input_path=("value",),
        lazy=False,
        input_is_list=False,
        output_is_list=False,
        raw_link=False,
        accept_all=False,
    )


def write_choice_manifest(
    directory: Path,
    *,
    name: str = "choicepack",
    choices: str | None = "choicepack_nodes:combo_choices",
    skips: str | None = None,
    namespaces: tuple[str, ...] = ("cp",),
    nodes: str = "choicepack_nodes:NODES",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    claims = "namespaces = [" + ", ".join(f'"{c}"' for c in namespaces) + "]\n"
    body = (
        f'[pack]\nname = "{name}"\n{claims}\n[pack.entry]\n'
        f'nodes = "{nodes}"\n'
        'types = "choicepack_nodes:register_types"\n'
    )
    if choices is not None:
        body += f'choices = "{choices}"\n'
    if skips is not None:
        body += f'skips = "{skips}"\n'
    manifest = directory / "dinkster-pack.toml"
    manifest.write_text(body)
    return manifest


# -- manifest ------------------------------------------------------------------


def test_manifest_choices_entry(tmp_path: Path) -> None:
    manifest = load_manifest(write_choice_manifest(tmp_path))
    assert manifest.choices_entry == "choicepack_nodes:combo_choices"
    assert load_manifest(write_choice_manifest(tmp_path, choices=None)).choices_entry is None

    bad = tmp_path / "bad.toml"
    bad.write_text('[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\nchoices = "no-colon"\n')
    with pytest.raises(ManifestError, match="module:attr"):
        load_manifest(bad)
    bad.write_text('[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\nchoices = 3\n')
    with pytest.raises(ManifestError, match="module:attr"):
        load_manifest(bad)


def test_manifest_skips_entry(tmp_path: Path) -> None:
    manifest = load_manifest(
        write_choice_manifest(tmp_path, skips="choicepack_nodes:translation_skips")
    )
    assert manifest.skips_entry == "choicepack_nodes:translation_skips"
    assert load_manifest(write_choice_manifest(tmp_path)).skips_entry is None

    bad = tmp_path / "bad.toml"
    for value in ('"no-colon"', "3"):
        bad.write_text(f'[pack]\nname = "p"\n[pack.entry]\nnodes = "m:N"\nskips = {value}\n')
        with pytest.raises(ManifestError, match="module:attr"):
            load_manifest(bad)


# -- worker host ---------------------------------------------------------------


def test_load_choices(tmp_path: Path) -> None:
    """The host validates at the source: grammar-valid ids, non-empty
    string values, dedupe preserving order; empty lists are legal."""
    manifest = load_manifest(write_choice_manifest(tmp_path))
    loaded = load_choices(manifest)
    assert loaded.static == {
        "cp.samplers": ("euler", "ddim", "heun"),
        "cp.empty": (),
    }
    assert loaded.lazy == {}
    absent = load_choices(load_manifest(write_choice_manifest(tmp_path, choices=None)))
    assert absent.static == {} and absent.lazy == {}


def test_load_choices_lazy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Callable mapping values load as lazy providers: announced by id,
    NEVER invoked at load (a provider that bumps the counter file proves
    it), and held to the same id grammar as static lists."""
    counter = tmp_path / "count"
    monkeypatch.setenv("CHOICEPACK_LAZY_COUNTER", str(counter))
    manifest = load_manifest(
        write_choice_manifest(tmp_path, choices="choicepack_nodes:lazy_probe_choices")
    )
    loaded = load_choices(manifest)
    assert loaded.static == {"cp.samplers": ("euler", "ddim")}
    assert set(loaded.lazy) == {"cp.devices", "cp.empty-lazy", "cp.boom", "cp.bad"}
    assert all(callable(provider) for provider in loaded.lazy.values())
    assert not counter.exists()

    bad = load_manifest(
        write_choice_manifest(tmp_path, choices="choicepack_nodes:lazy_bad_id_choices")
    )
    with pytest.raises(ManifestError, match="lowercase"):
        load_choices(bad)


def test_in_process_worker_serves_lazy_choices() -> None:
    """The in-process worker serves lazy fetches itself - one provider
    invocation per call, validated with the static grammar, unknown ids
    refused, provider failures and invalid results surfaced as errors."""
    calls: list[int] = []

    def devices() -> tuple[str, ...]:
        calls.append(1)
        return (f"dev-{len(calls)}", "dev-1")

    def boom() -> tuple[str, ...]:
        raise RuntimeError("device scan failed")

    worker = InProcessWorker(
        {},
        TypeRegistry(),
        lazy_choices={
            "cp.devices": devices,
            "cp.boom": boom,
            "cp.bad": lambda: ("ok", 3),  # type: ignore[return-value]
        },
    )
    assert worker.combo_choices == {}
    assert worker.lazy_choice_ids == ("cp.bad", "cp.boom", "cp.devices")
    assert not calls

    async def scenario() -> None:
        assert await worker.fetch_choices("cp.devices") == ("dev-1",)
        assert await worker.fetch_choices("cp.devices") == ("dev-2", "dev-1")
        assert len(calls) == 2
        with pytest.raises(ValueError, match="unknown lazy choice list"):
            await worker.fetch_choices("cp.absent")
        with pytest.raises(RuntimeError, match="device scan failed"):
            await worker.fetch_choices("cp.boom")
        with pytest.raises(ValueError, match="non-string"):
            await worker.fetch_choices("cp.bad")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ("choicepack_nodes:NOT_CALLABLE", "not callable"),
        ("choicepack_nodes:choices_not_mapping", "mapping"),
        ("choicepack_nodes:choices_bad_id", "lowercase"),
        ("choicepack_nodes:choices_bare_string", "sequence of strings"),
        ("choicepack_nodes:choices_non_string_value", "non-string or empty"),
        ("choicepack_nodes:choices_empty_value", "non-string or empty"),
        ("choicepack_nodes:choices_nul_value", "must not contain NUL"),
        ("choicepack_nodes:choices_surrogate_value", "must be valid UTF-8"),
        ("choicepack_nodes:choices_long_value", "maximum is 4096"),
        ("choicepack_nodes:choices_too_many", "maximum is 10000"),
        ("choicepack_nodes:choices_response_too_large", "response exceeds"),
    ],
)
def test_load_choices_refuses(tmp_path: Path, entry: str, match: str) -> None:
    manifest = load_manifest(write_choice_manifest(tmp_path, choices=entry))
    with pytest.raises(ManifestError, match=match):
        load_choices(manifest)


def test_load_skips_validates_mapping(tmp_path: Path) -> None:
    manifest = load_manifest(
        write_choice_manifest(tmp_path, skips="choicepack_nodes:translation_skips")
    )
    assert load_skips(manifest) == {
        "OpaqueNode": diagnostic("OpaqueNode", "unsupported dynamic marker")
    }
    assert load_skips(load_manifest(write_choice_manifest(tmp_path))) == {}

    for entry, match in (
        ("choicepack_nodes:NOT_CALLABLE", "not callable"),
        ("choicepack_nodes:skips_not_mapping", "mapping"),
        ("choicepack_nodes:skips_bad_name", "node name"),
        ("choicepack_nodes:skips_bad_reason", "reason"),
    ):
        bad = load_manifest(write_choice_manifest(tmp_path, skips=entry))
        with pytest.raises(ManifestError, match=match):
            load_skips(bad)


# -- hello parsing -------------------------------------------------------------


def test_combo_choices_from_hello() -> None:
    """The parent parses the hello strictly: absent means none, a
    well-formed table decodes to tuples, anything else fails the
    handshake loudly (a half-parsed choice list must never serve)."""
    assert _combo_choices_from_hello({}, role="pack", pack="p") == {}
    parsed = _combo_choices_from_hello(
        {"comboChoices": {"cp.samplers": ["euler", "ddim"], "cp.empty": []}},
        role="pack",
        pack="p",
    )
    assert parsed == {"cp.samplers": ("euler", "ddim"), "cp.empty": ()}

    for malformed in (
        ["cp.samplers"],  # not a mapping
        {"cp.samplers": "euler"},  # bare string values
        {"cp.samplers": ["euler", 3]},  # non-string value
        {"cp.samplers": ["euler", ""]},  # empty value
        {"": ["euler"]},  # empty id
        {"Bad Id": ["euler"]},  # noncanonical id
        {3: ["euler"]},  # non-string id
    ):
        with pytest.raises(RuntimeError, match="malformed comboChoices"):
            _combo_choices_from_hello({"comboChoices": malformed}, role="pack", pack="p")


def test_lazy_choice_ids_from_hello() -> None:
    """Same strict posture for the lazy announcement: absent means none,
    well-formed ids decode in order, and grammar violations, duplicates,
    or overlap with the same hello's static comboChoices fail the
    handshake loudly - one id has exactly one evaluation mode."""
    assert _lazy_choice_ids_from_hello({}, role="pack", pack="p", static_ids=()) == ()
    assert _lazy_choice_ids_from_hello(
        {"lazyChoiceIds": ["cp.devices", "cp.other"]},
        role="pack",
        pack="p",
        static_ids=("cp.samplers",),
    ) == ("cp.devices", "cp.other")

    for malformed in (
        "cp.devices",  # bare string, not a list
        {"cp.devices": ()},  # mapping
        [3],  # non-string entry
        [""],  # empty entry
        ["Bad Id"],  # noncanonical id
    ):
        with pytest.raises(RuntimeError, match="malformed lazyChoiceIds"):
            _lazy_choice_ids_from_hello(
                {"lazyChoiceIds": malformed}, role="pack", pack="p", static_ids=()
            )
    with pytest.raises(RuntimeError, match="duplicate lazyChoiceIds"):
        _lazy_choice_ids_from_hello(
            {"lazyChoiceIds": ["cp.devices", "cp.devices"]},
            role="pack",
            pack="p",
            static_ids=(),
        )
    with pytest.raises(RuntimeError, match="both a static comboChoices"):
        _lazy_choice_ids_from_hello(
            {"lazyChoiceIds": ["cp.samplers"]},
            role="pack",
            pack="p",
            static_ids=("cp.samplers",),
        )


def _response_boundary_values(*, over: bool = False) -> tuple[str, ...]:
    # Compact JSON size is 1 + sum(encoded string byte lengths) + 3*n for
    # quote/comma/bracket framing when these ASCII strings need no escaping.
    values = tuple(f"{index:04d}" + "x" * 4092 for index in range(511))
    tail_size = 2560 if over else 2559
    return (*values, "tail" + "x" * (tail_size - 4))


def test_combo_choice_response_boundary_is_exact() -> None:
    exact = _response_boundary_values()
    assert len(json.dumps(exact, ensure_ascii=False, separators=(",", ":")).encode()) == 2_097_152
    parsed = _combo_choices_from_hello(
        {"comboChoices": {"cp.boundary": list(exact)}}, role="pack", pack="p"
    )
    assert parsed["cp.boundary"] == exact
    with pytest.raises(RuntimeError, match="response exceeds"):
        _combo_choices_from_hello(
            {"comboChoices": {"cp.boundary": list(_response_boundary_values(over=True))}},
            role="pack",
            pack="p",
        )


@pytest.mark.parametrize(
    "values",
    (
        ("dup", "dup"),
        ("nul\0value",),
        ("\ud800",),
        ("x" * 4097,),
        tuple(f"v{index}" for index in range(10_001)),
    ),
)
def test_combo_choices_from_hello_rejects_unbounded_or_noncanonical_values(
    values: tuple[str, ...],
) -> None:
    with pytest.raises(RuntimeError, match="malformed comboChoices"):
        _combo_choices_from_hello(
            {"comboChoices": {"cp.invalid": list(values)}}, role="pack", pack="p"
        )


def test_derived_sampler_choices_are_bounded_before_publication() -> None:
    composer = ServingComposer()
    before = dict(composer.composition.choices)
    oversized = SamplerRegistrySnapshot(
        tuple(
            KeyedContribution(surface_id="sampling.sampler", id=f"cp.sampler-{index}")
            for index in range(10_001)
        )
    )
    with pytest.raises(CompositionError, match="maximum is 10000"):
        composer._validated_sampler_choices(oversized)
    assert composer.composition.choices == before


def test_compat_skips_from_hello() -> None:
    assert _compat_skips_from_hello({}, role="pack", pack="p") == {}
    expected = diagnostic("OpaqueNode", "reason")
    assert _compat_skips_from_hello(
        {"compatSkips": {"OpaqueNode": expected.to_wire()}}, role="pack", pack="p"
    ) == {"OpaqueNode": expected}
    for malformed in (["OpaqueNode"], {"": expected.to_wire()}, {"OpaqueNode": 3}):
        with pytest.raises(RuntimeError, match="malformed compatSkips"):
            _compat_skips_from_hello({"compatSkips": malformed}, role="pack", pack="p")
    mismatched = diagnostic("OtherNode", "reason")
    with pytest.raises(RuntimeError, match="sourceNode differs"):
        _compat_skips_from_hello(
            {"compatSkips": {"OpaqueNode": mismatched.to_wire()}},
            role="pack",
            pack="p",
        )


def test_compat_gate_diagnostic_is_frozen_strict_and_code_extensible() -> None:
    expected = diagnostic("OpaqueNode", "reason")
    assert CompatGateDiagnostic.from_wire(expected.to_wire()) == expected
    unknown = dict(expected.to_wire())
    unknown["code"] = "third-party.future-gate"
    assert CompatGateDiagnostic.from_wire(unknown).code == "third-party.future-gate"

    for malformed in (
        {**expected.to_wire(), "extra": False},
        {key: value for key, value in expected.to_wire().items() if key != "lazy"},
        {**expected.to_wire(), "inputIsList": 0},
        {**expected.to_wire(), "inputPath": ["value", 3]},
        {**expected.to_wire(), "inputId": "other"},
    ):
        with pytest.raises(ValueError):
            CompatGateDiagnostic.from_wire(malformed)

    with pytest.raises(AttributeError):
        expected.reason = "changed"  # type: ignore[misc]


# -- composition ---------------------------------------------------------------


def test_core_native_inference_choices() -> None:
    """The native sampler/scheduler registries are core vocabulary on
    every composition surface (stage 3b): dinkster.samplers and
    dinkster.schedulers serve the ported catalog ids with no packs added,
    and the development pack merges its lists instead of replacing them."""
    from dinkster_inference import builtin_samplers, builtin_schedulers

    async def scenario() -> None:
        for dev in (False, True):
            composer = ServingComposer(dev=dev, worker_env=WORKER_ENV)
            try:
                if dev:
                    await composer.add_pack(DEV_PACK_MANIFEST)
                choices = composer.composition.choices
                assert choices["dinkster.samplers"] == tuple(d.id for d in builtin_samplers())
                assert choices["dinkster.schedulers"] == tuple(d.id for d in builtin_schedulers())
                assert "dinkster.euler" in choices["dinkster.samplers"]
                assert "dinkster.karras" in choices["dinkster.schedulers"]
                if dev:
                    # Pack lists merge on top, leaving core lists intact.
                    assert any(cid.startswith("dev.") for cid in choices)
            finally:
                await composer.composition.close()

    asyncio.run(scenario())


def test_compose_surfaces_choices(tmp_path: Path) -> None:
    """The end-to-end happy path: worker enumerates, hello announces, the
    composer validates and merges, PackDelta carries the lists, and the
    served app answers /api/choices/{id} with a plain JSON array (404 for
    an unknown id) - remove retracts them."""

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            delta = await composer.add_pack(write_choice_manifest(tmp_path / "cp"))
            assert delta.choices == {
                "cp.samplers": ("euler", "ddim", "heun"),
                "cp.empty": (),
            }
            composition = composer.composition
            # Pack choices merge on top of the always-present core native
            # inference vocabulary (stage 3b).
            assert composition.choices == {
                **{
                    cid: composition.choices[cid]
                    for cid in ("dinkster.samplers", "dinkster.schedulers")
                },
                **delta.choices,
            }

            def make_engine(on_event: EventListener) -> Engine:
                return composition.make_engine(on_event)

            app = create_app(
                make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                choices=composition.choices,
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                resp = await client.get("/api/choices/cp.samplers")
                assert resp.status == 200
                assert await resp.json() == ["euler", "ddim", "heun"]
                # Core native inference vocabulary served over HTTP too
                # (stage 3b): namespaced descriptor ids, always present.
                resp = await client.get("/api/choices/dinkster.samplers")
                assert resp.status == 200
                assert "dinkster.euler" in await resp.json()
                resp = await client.get("/api/choices/dinkster.schedulers")
                assert resp.status == 200
                assert "dinkster.karras" in await resp.json()
                resp = await client.get("/api/choices/cp.empty")
                assert await resp.json() == []
                resp = await client.get("/api/choices/cp.unknown")
                assert resp.status == 404

                state = app[STATE_KEY]
                result = await apply_remove(state, composer, "choicepack")
                assert result["pack"] == "choicepack"
                assert set(composition.choices) == {
                    "dinkster.samplers",
                    "dinkster.schedulers",
                }
                resp = await client.get("/api/choices/cp.samplers")
                assert resp.status == 404
            finally:
                await client.close()
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


def test_compose_surfaces_lazy_choices(tmp_path: Path) -> None:
    """The lazy end-to-end path: the worker announces provider ids in its
    hello (values absent), composition claims them like static ids -
    including remote combo authority, cp.lazy-pick's route targets a lazy
    id - and each /api/choices fetch runs the provider in the worker
    exactly once: zero invocations at startup, one per fetch, no cache
    (the counter file is the cross-process invocation record). Provider
    exceptions and invalid results map to 502, an empty list is a legal
    200 [], and remove retires the routes."""

    async def scenario() -> None:
        counter = tmp_path / "count"
        composer = ServingComposer(
            worker_env={**WORKER_ENV, "CHOICEPACK_LAZY_COUNTER": str(counter)}
        )
        try:
            delta = await composer.add_pack(
                write_choice_manifest(
                    tmp_path / "cp",
                    choices="choicepack_nodes:lazy_probe_choices",
                    nodes="choicepack_nodes:LAZY_NODES",
                )
            )
            assert delta.choices == {"cp.samplers": ("euler", "ddim")}
            assert set(delta.lazy_choices) == {
                "cp.devices",
                "cp.empty-lazy",
                "cp.boom",
                "cp.bad",
            }
            composition = composer.composition
            assert set(composition.lazy_choices) == set(delta.lazy_choices)
            assert not counter.exists()  # startup and hello ran no provider

            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                resp = await client.get("/api/choices/cp.devices")
                assert resp.status == 200
                assert resp.headers["Cache-Control"] == "no-store"
                assert await resp.json() == ["dev-1", "common"]
                resp = await client.get("/api/choices/cp.devices")
                assert await resp.json() == ["dev-2", "common"]
                assert counter.read_text() == "2"

                resp = await client.get("/api/choices/cp.empty-lazy")
                assert resp.status == 200
                assert await resp.json() == []

                resp = await client.get("/api/choices/cp.boom")
                assert resp.status == 502
                assert resp.headers["Cache-Control"] == "no-store"
                assert "provider failed" in (await resp.json())["error"]
                resp = await client.get("/api/choices/cp.bad")
                assert resp.status == 502

                # The static list from the same hello serves without any
                # provider involvement.
                resp = await client.get("/api/choices/cp.samplers")
                assert await resp.json() == ["euler", "ddim"]
                assert counter.read_text() == "2"

                await apply_remove(app[STATE_KEY], composer, "choicepack")
                assert composition.lazy_choices == {}
                resp = await client.get("/api/choices/cp.devices")
                assert resp.status == 404
            finally:
                await client.close()
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


def test_lazy_choice_timeout_and_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung provider is a 504 under the server-owned budget. The late
    reply lands after the timeout, finds no pending future, and is
    discarded; the next fetch runs a fresh invocation and succeeds - one
    timeout cannot wedge the channel."""

    async def scenario() -> None:
        counter = tmp_path / "count"
        composer = ServingComposer(
            worker_env={**WORKER_ENV, "CHOICEPACK_LAZY_COUNTER": str(counter)}
        )
        try:
            await composer.add_pack(
                write_choice_manifest(
                    tmp_path / "cp",
                    namespaces=("cp", "cp2"),
                    nodes="choicepack_nodes:NODES2",
                    choices="choicepack_nodes:lazy_slow_choices",
                )
            )
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
            )
            monkeypatch.setattr("dinkster_server.app.LAZY_CHOICE_TIMEOUT_SECONDS", 0.2)
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                resp = await client.get("/api/choices/cp.slow")
                assert resp.status == 504
                assert resp.headers["Cache-Control"] == "no-store"
                assert "timed out" in (await resp.json())["error"]
                # Let the hung first invocation finish so its late reply
                # arrives (and is discarded) before the second fetch proves
                # the channel is still healthy.
                await asyncio.sleep(2.2)
                resp = await client.get("/api/choices/cp.slow")
                assert resp.status == 200
                assert await resp.json() == ["slow-2"]
                assert counter.read_text() == "2"
            finally:
                await client.close()
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


def test_lazy_choice_dead_owner_serves_503(tmp_path: Path) -> None:
    """A dead owning worker surfaces as 503 - unavailability, not provider
    fault - so a frontend refresh retries against live state."""

    async def scenario() -> None:
        counter = tmp_path / "count"
        composer = ServingComposer(
            worker_env={**WORKER_ENV, "CHOICEPACK_LAZY_COUNTER": str(counter)}
        )
        try:
            await composer.add_pack(
                write_choice_manifest(
                    tmp_path / "cp",
                    choices="choicepack_nodes:lazy_probe_choices",
                    nodes="choicepack_nodes:LAZY_NODES",
                )
            )
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                resp = await client.get("/api/choices/cp.devices")
                assert resp.status == 200

                process = composer._records["choicepack"].worker._proc
                assert process is not None
                process.kill()
                await process.wait()
                # The session notices the death when its read loop sees EOF;
                # poll briefly rather than race it.
                for _ in range(100):
                    resp = await client.get("/api/choices/cp.devices")
                    if resp.status == 503:
                        break
                    await asyncio.sleep(0.05)
                assert resp.status == 503
                assert resp.headers["Cache-Control"] == "no-store"
                assert "not connected" in (await resp.json())["error"]
            finally:
                await client.close()
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


def test_lazy_fetcher_maps_worker_death_to_owner_gone() -> None:
    """The composed fetcher translates WorkerDied into ChoiceOwnerGone so
    the server can answer 503 without knowing worker session types."""

    class DeadWorker:
        async def fetch_choices(self, choice_id: str) -> tuple[str, ...]:
            raise WorkerDied()

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            fetch = _lazy_choice_fetcher(DeadWorker(), "cp.devices", composer)
            with pytest.raises(ChoiceOwnerGone, match="not connected"):
                await fetch()
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("adopted", [False, True])
def test_lazy_choices_wait_for_full_free_before_activating_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adopted: bool
) -> None:
    manifest = write_choice_manifest(
        tmp_path / "cp",
        choices="choicepack_nodes:lazy_probe_choices",
        nodes="choicepack_nodes:LAZY_NODES",
    )
    assert prepare_catalog(manifest, environment=WORKER_ENV).ok

    async def scenario() -> None:
        entered, finish = asyncio.Event(), asyncio.Event()

        class BlockingPool(ResidentPool):
            async def full_release(self) -> FullReleaseResult:
                entered.set()
                await finish.wait()
                return await super().full_release()

        pool = BlockingPool()
        monkeypatch.setattr("dinkster.compose.default_pool", lambda: pool)
        composer = ServingComposer(
            worker_env={**WORKER_ENV, "CHOICEPACK_LAZY_COUNTER": str(tmp_path / "count")}
        )
        staged = composer.spawn_empty() if adopted else composer
        try:
            await staged.add_pack(PackSpec(manifest, require_catalog=True))
            if adopted:
                async with composer.generation_transaction():
                    old = composer.adopt(staged)
                await old.close()
            worker = composer._records["choicepack"].worker
            assert worker.cold
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
                full_free=composer.full_free,
            )
            async with TestClient(TestServer(app)) as client:
                await client.post("/api/queue/pause")
                release = asyncio.create_task(
                    client.post("/memory/free", json={"requestId": "choices"})
                )
                try:
                    await asyncio.wait_for(entered.wait(), 5)
                    fetch = asyncio.create_task(client.get("/api/choices/cp.devices"))
                    try:
                        done, _ = await asyncio.wait({fetch}, timeout=0.1)
                        assert not done
                        assert worker.cold
                        assert composer._mutate.locked()
                        assert app[STATE_KEY].queue._maintenance_active
                        finish.set()
                        body = await (await release).json()
                        assert body["completed"] is True
                        assert "choicepack" not in {item["worker"] for item in body["workers"]}
                        response = await asyncio.wait_for(fetch, 5)
                        assert response.status == 200, await response.json()
                        assert worker.alive and not worker.cold
                        assert app[STATE_KEY].queue.paused
                    finally:
                        finish.set()
                        await fetch
                finally:
                    finish.set()
                    await release
        finally:
            finish.set()
            await composer.close()
            if staged is not composer and staged._private_staging:
                await staged.close()

    asyncio.run(scenario())


def test_compose_rejects_uncovered_choice_id(tmp_path: Path) -> None:
    """A choice id outside the pack's namespace claims is refused - the
    same coverage rule as node types (but cp.pick must also be covered,
    so the pack claims both roots and only the choice id is outside)."""

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            with pytest.raises(CompositionError, match="outside the pack's declared"):
                await composer.add_pack(
                    write_choice_manifest(
                        tmp_path,
                        choices="choicepack_nodes:choices_outside_namespace",
                    )
                )
            assert set(composer.composition.choices) == {
                "dinkster.samplers",
                "dinkster.schedulers",
            }
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


def test_compose_rejects_choice_collision(tmp_path: Path) -> None:
    """Two packs may not both declare the same choice id - exclusive
    ownership, like node types. Non-reserved namespace claims are already
    exclusive, so the collision is only reachable through a SHARED
    reserved root (two trusted workers both claiming "comfy" - the compat
    shape). The failing add merges nothing."""

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            await composer.add_pack(
                PackSpec(
                    manifest=write_choice_manifest(
                        tmp_path / "a",
                        namespaces=("cp", "comfy"),
                        choices="choicepack_nodes:choices_reserved",
                        nodes="choicepack_nodes:RESERVED_NODES",
                    ),
                    trust_reserved=True,
                )
            )
            with pytest.raises(CompositionError, match="declared by both"):
                await composer.add_pack(
                    PackSpec(
                        manifest=write_choice_manifest(
                            tmp_path / "b",
                            name="choicepack2",
                            namespaces=("cp2", "comfy"),
                            nodes="choicepack_nodes:NODES2",
                            choices="choicepack_nodes:choices_reserved",
                        ),
                        trust_reserved=True,
                    )
                )
            # A lazy declaration of the same id collides with the static
            # owner identically - one id, one owner, either evaluation mode.
            with pytest.raises(CompositionError, match="declared by both"):
                await composer.add_pack(
                    PackSpec(
                        manifest=write_choice_manifest(
                            tmp_path / "c",
                            name="choicepack3",
                            namespaces=("cp2", "comfy"),
                            nodes="choicepack_nodes:NODES2",
                            choices="choicepack_nodes:choices_reserved_lazy",
                        ),
                        trust_reserved=True,
                    )
                )
            assert set(composer.composition.choices) == {
                "comfy.samplers",
                "dinkster.samplers",
                "dinkster.schedulers",
            }
            assert composer.composition.lazy_choices == {}
            assert set(composer.composition.schemas) >= {"cp.reserved-pick"}
            assert "cp2.pick" not in composer.composition.schemas
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


def test_compose_rejects_collisions_against_lazy_owner(tmp_path: Path) -> None:
    """A lazy id is an exclusive owner too: a second pack redeclaring it -
    statically or lazily - is refused, through the same shared reserved
    root that makes cross-pack choice collisions reachable. The lazy owner
    also proves remote combo authority accepts a reserved lazy id
    (cp.reserved-pick's route targets it)."""

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            await composer.add_pack(
                PackSpec(
                    manifest=write_choice_manifest(
                        tmp_path / "a",
                        namespaces=("cp", "comfy"),
                        choices="choicepack_nodes:choices_reserved_lazy",
                        nodes="choicepack_nodes:RESERVED_NODES",
                    ),
                    trust_reserved=True,
                )
            )
            assert set(composer.composition.lazy_choices) == {"comfy.samplers"}
            for choices_entry in (
                "choicepack_nodes:choices_reserved",
                "choicepack_nodes:choices_reserved_lazy",
            ):
                with pytest.raises(CompositionError, match="declared by both"):
                    await composer.add_pack(
                        PackSpec(
                            manifest=write_choice_manifest(
                                tmp_path / "b",
                                name="choicepack2",
                                namespaces=("cp2", "comfy"),
                                nodes="choicepack_nodes:NODES2",
                                choices=choices_entry,
                            ),
                            trust_reserved=True,
                        )
                    )
            assert set(composer.composition.lazy_choices) == {"comfy.samplers"}
            assert "comfy.samplers" not in composer.composition.choices
            assert "cp2.pick" not in composer.composition.schemas
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "nodes",
    ("choicepack_nodes:MISSING_REMOTE_NODES", "choicepack_nodes:NESTED_MISSING_REMOTE_NODES"),
)
def test_compose_requires_same_pack_authority_for_every_remote_route(
    tmp_path: Path, nodes: str
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            before_schemas = dict(composer.composition.schemas)
            before_choices = dict(composer.composition.choices)
            with pytest.raises(CompositionError, match="remote choice.*not registered"):
                await composer.add_pack(write_choice_manifest(tmp_path, nodes=nodes))
            assert composer.composition.schemas == before_schemas
            assert composer.composition.choices == before_choices
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


def test_reload_swaps_choices(tmp_path: Path) -> None:
    """Reload is remove-then-add for choices too: the same pack may
    re-announce its own ids (no self-collision), dropped ids retract,
    and the live server serves the new lists after ONE replace."""

    async def scenario() -> None:
        pack_dir = tmp_path / "cp"
        pack_dir.mkdir(parents=True, exist_ok=True)
        # v2's choices module lives in the pack dir, so the worker needs
        # both roots importable (the test_reload shape).
        composer = ServingComposer(
            worker_env={"PYTHONPATH": os.pathsep.join((str(TESTS_DIR), str(pack_dir)))}
        )
        try:
            await composer.add_pack(write_choice_manifest(pack_dir))
            composition = composer.composition

            def make_engine(on_event: EventListener) -> Engine:
                return composition.make_engine(on_event)

            app = create_app(
                make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                choices=composition.choices,
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                state = app[STATE_KEY]
                # v2 drops cp.empty: outside-namespace ids aside, the
                # easiest observable change is the surviving id set.
                (pack_dir / "choicepack_v2.py").write_text(
                    "from choicepack_nodes import NODES, register_types\n"
                    "def combo_choices():\n"
                    '    return {"cp.samplers": ("lcm",)}\n'
                )
                write_choice_manifest(pack_dir, choices="choicepack_v2:combo_choices")
                await apply_reload(state, composer, "choicepack")
                assert composition.choices["cp.samplers"] == ("lcm",)
                assert "cp.empty" not in composition.choices
                resp = await client.get("/api/choices/cp.samplers")
                assert await resp.json() == ["lcm"]
                resp = await client.get("/api/choices/cp.empty")
                assert resp.status == 404
            finally:
                await client.close()
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


def test_reload_swaps_lazy_choices(tmp_path: Path) -> None:
    """Reload is remove-then-add for lazy routes too: the owning pack may
    re-declare its own id - even flipping it from lazy to static - dropped
    lazy ids retire from the live server, and the swap itself runs no
    provider."""

    async def scenario() -> None:
        pack_dir = tmp_path / "cp"
        pack_dir.mkdir(parents=True, exist_ok=True)
        counter = tmp_path / "count"
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join((str(TESTS_DIR), str(pack_dir))),
                "CHOICEPACK_LAZY_COUNTER": str(counter),
            }
        )
        try:
            await composer.add_pack(
                write_choice_manifest(
                    pack_dir,
                    choices="choicepack_nodes:lazy_probe_choices",
                    nodes="choicepack_nodes:LAZY_NODES",
                )
            )
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                choices=composition.choices,
                lazy_choices=composition.lazy_choices,
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                resp = await client.get("/api/choices/cp.devices")
                assert await resp.json() == ["dev-1", "common"]

                (pack_dir / "choicepack_v2.py").write_text(
                    "from choicepack_nodes import LAZY_NODES as NODES, register_types\n"
                    "def combo_choices():\n"
                    "    return {'cp.samplers': ('lcm',),"
                    " 'cp.devices': ('static-now',)}\n"
                )
                write_choice_manifest(
                    pack_dir,
                    choices="choicepack_v2:combo_choices",
                    nodes="choicepack_v2:NODES",
                )
                await apply_reload(app[STATE_KEY], composer, "choicepack")
                assert composition.lazy_choices == {}
                assert composition.choices["cp.devices"] == ("static-now",)
                resp = await client.get("/api/choices/cp.devices")
                assert await resp.json() == ["static-now"]
                resp = await client.get("/api/choices/cp.boom")
                assert resp.status == 404
                assert counter.read_text() == "1"
            finally:
                await client.close()
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


def test_compat_skips_compose_serve_and_reload(tmp_path: Path) -> None:
    """Worker skips reach diagnostics and reload replaces the owned slice,
    so stale reasons from the previous worker cannot linger."""

    async def scenario() -> None:
        pack_dir = tmp_path / "cp"
        composer = ServingComposer(
            worker_env={"PYTHONPATH": os.pathsep.join((str(TESTS_DIR), str(pack_dir)))}
        )
        try:
            delta = await composer.add_pack(
                write_choice_manifest(pack_dir, skips="choicepack_nodes:translation_skips")
            )
            assert delta.compat_skips == {
                "choicepack": {"OpaqueNode": diagnostic("OpaqueNode", "unsupported dynamic marker")}
            }
            composition = composer.composition
            app = create_app(
                composition.make_engine,
                composition.schemas,
                packs=composition.packs,
                node_packs=composition.node_packs,
                choices=composition.choices,
                compat_skips=composition.compat_skips,
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                wire = (await (await client.get("/api/diagnostics")).json())["compatSkips"]
                assert len(wire) == 1
                assert wire[0]["packId"] == "choicepack"
                assert wire[0]["nodeId"] == "OpaqueNode"
                assert wire[0]["reason"] == "unsupported dynamic marker"
                assert wire[0]["code"] == "compat.dynamic.unsupported"
                assert wire[0]["schemaEpoch"] == 1
                assert wire[0]["extensionSnapshotDigest"]
                (pack_dir / "choicepack_v2.py").write_text(
                    "from choicepack_nodes import NODES, _diagnostic, register_types\n"
                    "def translation_skips():\n"
                    '    return {"ReplacementNode": _diagnostic("ReplacementNode", "new reason")}\n'
                )
                write_choice_manifest(
                    pack_dir,
                    skips="choicepack_v2:translation_skips",
                    nodes="choicepack_v2:NODES",
                )
                await apply_reload(app[STATE_KEY], composer, "choicepack")
                wire = (await (await client.get("/api/diagnostics")).json())["compatSkips"]
                assert len(wire) == 1
                assert wire[0]["nodeId"] == "ReplacementNode"
                assert wire[0]["reason"] == "new reason"
                assert wire[0]["schemaEpoch"] == 2
            finally:
                await client.close()
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


def test_legacy_compat_skips_use_derived_pack_id(tmp_path: Path) -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            spec = PackSpec(
                manifest=write_choice_manifest(
                    tmp_path,
                    skips="choicepack_nodes:legacy_translation_skips",
                ),
                packs={
                    "comfy": PackInfo(display_name="Comfy"),
                    "comfy.legacy": PackInfo(display_name="Legacy"),
                },
                attribute=lambda name: (
                    "comfy.legacy" if name.startswith("comfy.legacy.") else "comfy"
                ),
            )
            delta = await composer.add_pack(spec)
            assert delta.compat_skips == {
                "comfy.legacy": {
                    "OpaqueNode": diagnostic("comfy.legacy.OpaqueNode", "legacy dynamic marker")
                }
            }
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


# -- server state --------------------------------------------------------------


def test_state_announce_and_replace_choices(tmp_path: Path) -> None:
    """ServerState mirrors the composer's rules at the serving seam:
    announce refuses redefinition, replace refuses removing unknown ids
    and redefining ids not being removed in the same swap."""

    async def scenario() -> None:
        composer = ServingComposer(worker_env=WORKER_ENV)
        try:
            composition = composer.composition

            def make_engine(on_event: EventListener) -> Engine:
                return composition.make_engine(on_event)

            app = create_app(make_engine, dict(composition.schemas))
            state = app[STATE_KEY]
            delta = await composer.add_pack(write_choice_manifest(tmp_path))
            state.announce(delta.schemas, delta.packs, delta.node_packs, choices=delta.choices)
            assert state.choices["cp.samplers"] == ("euler", "ddim", "heun")
            with pytest.raises(ValueError, match="redefine choice list"):
                state.announce({}, {}, {}, choices={"cp.samplers": ("x",)})
            with pytest.raises(ValueError, match="unknown choice list"):
                state.replace((), (), {}, {}, {}, remove_choices=("cp.nope",))
            with pytest.raises(ValueError, match="not being removed"):
                state.replace((), (), {}, {}, {}, choices={"cp.samplers": ("x",)})
            epoch = state.schema_epoch
            schemas_before = dict(state.schemas)
            choices_before = dict(state.choices)
            with pytest.raises(ValueError, match="remote choice.*not registered"):
                state.replace(
                    (),
                    (),
                    {},
                    {},
                    {},
                    remove_choices=("cp.samplers",),
                )
            assert state.schema_epoch == epoch
            assert state.schemas == schemas_before
            assert state.choices == choices_before
            state.replace(
                (),
                (),
                {},
                {},
                {},
                remove_choices=("cp.samplers", "cp.empty"),
                choices={"cp.samplers": ("lcm",)},
            )
            assert state.choices == {"cp.samplers": ("lcm",)}
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


def test_state_announce_and_replace_lazy_choices() -> None:
    """Static and lazy ids share one exclusive table at the serving seam:
    construction refuses an id declared both ways, announce refuses
    redefinition in either direction, and replace retires a lazy route or
    flips it static in one swap."""

    async def scenario() -> None:
        async def fetch() -> tuple[str, ...]:
            return ("a",)

        composer = ServingComposer()
        try:
            composition = composer.composition
            with pytest.raises(ValueError, match="both static and lazy"):
                create_app(
                    composition.make_engine,
                    {},
                    choices={"cp.x": ("a",)},
                    lazy_choices={"cp.x": fetch},
                )
            app = create_app(
                composition.make_engine,
                {},
                choices={"cp.static": ("a",)},
                lazy_choices={"cp.lazy": fetch},
            )
            state = app[STATE_KEY]
            with pytest.raises(ValueError, match="redefine choice list"):
                state.announce({}, {}, {}, choices={"cp.lazy": ("x",)})
            with pytest.raises(ValueError, match="redefine choice list"):
                state.announce({}, {}, {}, lazy_choices={"cp.lazy": fetch})
            with pytest.raises(ValueError, match="redefine choice list"):
                state.announce({}, {}, {}, lazy_choices={"cp.static": fetch})
            with pytest.raises(ValueError, match="not being removed"):
                state.replace((), (), {}, {}, {}, lazy_choices={"cp.static": fetch})
            state.replace(
                (),
                (),
                {},
                {},
                {},
                remove_choices=("cp.lazy",),
                choices={"cp.lazy": ("now-static",)},
            )
            assert state.choices["cp.lazy"] == ("now-static",)
            assert state.lazy_choices == {}
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


def test_server_publication_revalidates_remote_authority_and_choice_bounds() -> None:
    async def scenario() -> None:
        remote_schema = NodeSchema(
            node_type="cp.remote",
            inputs=(
                InputSpec(
                    "choice",
                    TypeExpr.concrete("core.combo"),
                    widget=ComboWidget(remote_route="/api/choices/cp.remote"),
                ),
            ),
        )
        composer = ServingComposer()
        try:
            app = create_app(composer.composition.make_engine, schemas={})
            state = app[STATE_KEY]
            with pytest.raises(ValueError, match="remote choice.*not registered"):
                state.announce({remote_schema.node_type: remote_schema}, {}, {})
            with pytest.raises(ValueError, match="duplicate"):
                state.announce({}, {}, {}, choices={"cp.duplicate": ("x", "x")})
            for invalid in (
                ("nul\0value",),
                ("\ud800",),
                ("x" * 4097,),
                tuple(f"v{index}" for index in range(10_001)),
                _response_boundary_values(over=True),
            ):
                with pytest.raises(ValueError):
                    state.announce({}, {}, {}, choices={"cp.invalid": invalid})
            assert state.schemas == {}
            assert state.choices == {}
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


def test_replacement_retains_only_same_owner_remote_choice_authority() -> None:
    remote_schema = NodeSchema(
        node_type="cp.remote",
        inputs=(
            InputSpec(
                "choice",
                TypeExpr.concrete("core.combo"),
                widget=ComboWidget(remote_route="/api/choices/cp.remote"),
            ),
        ),
    )

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            app = create_app(composer.composition.make_engine, schemas={})
            state = app[STATE_KEY]
            state.replace(
                (),
                (),
                {},
                {},
                {},
                choices={"cp.remote": ("a",)},
                choice_owners={"cp.remote": "owner-a"},
            )
            state.announce(
                {remote_schema.node_type: remote_schema},
                {},
                {},
                schema_owners={remote_schema.node_type: "owner-a"},
            )
            state.replace(
                (remote_schema.node_type,),
                (),
                {remote_schema.node_type: remote_schema},
                {},
                {},
                schema_owners={remote_schema.node_type: "owner-a"},
            )
            epoch = state.schema_epoch
            schemas_before = dict(state.schemas)
            choices_before = dict(state.choices)
            with pytest.raises(ValueError, match="remote choice.*not registered"):
                state.replace(
                    (remote_schema.node_type,),
                    (),
                    {remote_schema.node_type: remote_schema},
                    {},
                    {},
                    schema_owners={remote_schema.node_type: "owner-b"},
                )
            assert state.schema_epoch == epoch
            assert state.schemas == schemas_before
            assert state.choices == choices_before
        finally:
            await composer.composition.close()

    asyncio.run(scenario())


def test_choice_endpoint_is_compact_bounded_and_no_store() -> None:
    async def scenario() -> None:
        values = ('quote"', "slash\\", "snowman-\u2603")
        boundary = _response_boundary_values()
        composer = ServingComposer()
        try:
            app = create_app(
                composer.composition.make_engine,
                schemas={},
                choices={"cp.values": values, "cp.boundary": boundary},
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                response = await client.get("/api/choices/cp.values")
                assert response.status == 200
                assert response.headers["Content-Type"] == "application/json; charset=utf-8"
                assert response.headers["Cache-Control"] == "no-store"
                assert await response.read() == json.dumps(
                    values, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                response = await client.get("/api/choices/cp.boundary")
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                assert len(await response.read()) == 2_097_152
                missing = await client.get("/api/choices/cp.missing")
                assert missing.status == 404
                assert missing.headers["Content-Type"] == "application/json; charset=utf-8"
                assert missing.headers["Cache-Control"] == "no-store"
            finally:
                await client.close()
        finally:
            await composer.composition.close()

    asyncio.run(scenario())
