"""Torch-free proofs for the attention/block extension contract.

Covers descriptor and selector validation, canonical declaration
projection, pin enforcement in the inference worker, backend exclusivity
per model family, behavior-identity drift, and the public API exports.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

import pytest
from dinkster_inference import (
    ATTENTION_BACKEND_SURFACE,
    ATTENTION_OUTPUT_SURFACE,
    ATTENTION_QKV_SURFACE,
    ATTENTION_WRAPPER_SURFACE,
    BLOCK_INJECTION_SURFACE,
    SamplerExtensionEntry,
    materialize_inference_generation,
    sampling_execution_context,
    write_sampler_catalog,
)
from dinkster_inference import attention as attention_module
from dinkster_inference.attention import (
    TORCH_DISTRIBUTION,
    AttentionBackendDescriptor,
    AttentionCallContext,
    AttentionContribution,
    AttentionOutputDescriptor,
    AttentionPinError,
    AttentionQKVDescriptor,
    AttentionSelector,
    AttentionTokenSpan,
    AttentionWrapperDescriptor,
    BlockInjectionDescriptor,
    attention_declarations,
    check_attention_pins,
)
from dinkster_protocol import (
    ATTENTION_SURFACES,
    ActiveExtension,
    ExtensionSnapshot,
    extension_behavior_hash,
)

FIXTURES = Path(__file__).parent
PACK_MODULES = ("attention_pack_a", "attention_pack_b")


@pytest.fixture(autouse=True)
def _fixture_modules():
    sys.path.insert(0, str(FIXTURES))
    for module in PACK_MODULES:
        sys.modules.pop(module, None)
    yield
    for module in PACK_MODULES:
        sys.modules.pop(module, None)
    sys.path.remove(str(FIXTURES))


def pack_a_contribution() -> AttentionContribution:
    import attention_pack_a

    return attention_pack_a.ATTENTION


def _entries(*modules: str) -> tuple[SamplerExtensionEntry, ...]:
    extension_ids = ("pack_a", "pack_b")
    return tuple(
        SamplerExtensionEntry(extension_ids[index], f"{module}:register")
        for index, module in enumerate(modules)
    )


def _catalog(path: Path) -> Path:
    return path / "catalog.json"


def _materialize(path: Path, entries: tuple[SamplerExtensionEntry, ...]):
    key = f"candidate:{uuid.uuid4().hex}"
    write_sampler_catalog(_catalog(path), key, entries)
    return materialize_inference_generation(key, catalog_path=_catalog(path))


def _matching_versions(name: str) -> str:
    return {"torch": "2.13.0+cpu", "dinkster-aimdo": "0.5.5"}[name]


def _span(**overrides: object) -> AttentionTokenSpan:
    values: dict[str, object] = {
        "axis": "query",
        "start": 0,
        "end": 77,
        "condition_id": "cond.prompt",
        "role": "conditional",
        "batch_start": 0,
        "batch_end": 2,
        "stream": "text",
    }
    values.update(overrides)
    return AttentionTokenSpan(**values)  # type: ignore[arg-type]


def _call_context(**overrides: object) -> AttentionCallContext:
    values: dict[str, object] = {
        "family": "flux",
        "block": "double_blocks.0",
        "kind": "joint",
        "heads": 24,
        "spatial_shape": (4096, 4096),
        "spans": (_span(),),
        "execution": sampling_execution_context((1.0, 0.0), 3),
        "state": {},
    }
    values.update(overrides)
    return AttentionCallContext(**values)  # type: ignore[arg-type]


def _replaced_metadata(item, key: str, value: object):
    return type(item)(
        item.surface_id,
        item.id,
        aliases=item.aliases,
        behavior_metadata=tuple(
            (name, value if name == key else old) for name, old in item.behavior_metadata
        ),
    )


def test_selector_matches_whole_wildcards_or_exact_names_only() -> None:
    selector = AttentionSelector(family="flux", block="double_blocks.0", kind="cross")
    assert selector.matches("flux", "double_blocks.0", "cross")
    assert not selector.matches("unet", "double_blocks.0", "cross")
    assert not selector.matches("flux", "double_blocks.1", "cross")
    assert not selector.matches("flux", "double_blocks.0", "joint")

    wildcard = AttentionSelector(family="flux")
    assert wildcard.matches("flux", "middle_block.1.transformer_blocks.0", "self")

    embedded = AttentionSelector(family="flux", block="double_blocks.0*")
    assert not embedded.matches("flux", "double_blocks.0", "self")

    with pytest.raises(ValueError, match="family must be a nonempty exact name"):
        AttentionSelector(family="")
    with pytest.raises(ValueError, match="family must be a nonempty exact name"):
        AttentionSelector(family="   ")
    with pytest.raises(ValueError, match="block must be '\\*' or a nonempty"):
        AttentionSelector(family="flux", block="")
    with pytest.raises(ValueError, match="kind must be '\\*' or one of"):
        AttentionSelector(family="flux", kind="sliding")


def test_token_spans_reject_empty_or_negative_ranges_and_unknown_vocabularies() -> None:
    assert _span(axis="key", stream="reference").end == 77
    with pytest.raises(ValueError, match="axis must be 'query' or 'key'"):
        _span(axis="value")
    with pytest.raises(ValueError, match="token range"):
        _span(start=5, end=5)
    with pytest.raises(ValueError, match="token range"):
        _span(start=-1, end=4)
    with pytest.raises(ValueError, match="batch range"):
        _span(batch_start=3, batch_end=3)
    with pytest.raises(ValueError, match="condition_id must be nonempty"):
        _span(condition_id="")
    with pytest.raises(ValueError, match="role must be nonempty"):
        _span(role="")
    with pytest.raises(ValueError, match="stream must be one of"):
        _span(stream="audio")


def test_call_context_requires_execution_context_and_mutable_state() -> None:
    context = _call_context()
    assert context.spans[0].stream == "text"
    assert context.state == {}

    with pytest.raises(TypeError, match="execution must be a SamplingExecutionContext"):
        _call_context(execution=object())
    with pytest.raises(TypeError, match="state must be a MutableMapping"):
        _call_context(state=())
    with pytest.raises(ValueError, match="spatial_shape must be two positive ints"):
        _call_context(spatial_shape=(4096,))
    with pytest.raises(ValueError, match="heads must be a positive int"):
        _call_context(heads=0)
    with pytest.raises(ValueError, match="kind must be one of"):
        _call_context(kind="*")


def test_attention_contribution_requires_explicit_pins_and_unique_ids() -> None:
    backend = AttentionBackendDescriptor("proof.backend", "flux", lambda q, k, v, context: q)

    with pytest.raises(TypeError, match="torch_version"):
        AttentionContribution(backends=(backend,))  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="torch_version must be an explicit nonempty"):
        AttentionContribution(backends=(backend,), torch_version="", aimdo_version="0.5.5")
    with pytest.raises(ValueError, match="aimdo_version must be an explicit nonempty"):
        AttentionContribution(backends=(backend,), torch_version="2.13.0", aimdo_version="  ")
    with pytest.raises(ValueError, match="attention contribution must be nonempty"):
        AttentionContribution(torch_version="2.13.0", aimdo_version="0.5.5")
    with pytest.raises(ValueError, match="ids must be unique within a contribution"):
        AttentionContribution(
            qkv=(
                AttentionQKVDescriptor(
                    "proof.same",
                    AttentionSelector(family="flux"),
                    lambda q, k, v, context: (q, k, v),
                ),
            ),
            outputs=(
                AttentionOutputDescriptor(
                    "proof.same",
                    AttentionSelector(family="flux"),
                    lambda output, context: output,
                ),
            ),
            torch_version="2.13.0",
            aimdo_version="0.5.5",
        )
    with pytest.raises(TypeError, match="terminal must be bool"):
        AttentionWrapperDescriptor(
            "proof.wrapper",
            AttentionSelector(family="flux"),
            lambda q, k, v, context, next: q,
            terminal=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="phase must be 'before' or 'after'"):
        BlockInjectionDescriptor(
            "proof.inject",
            AttentionSelector(family="flux"),
            lambda value, context: value,
            phase="around",
        )
    with pytest.raises(ValueError, match="exact nonempty family id"):
        AttentionBackendDescriptor("proof.backend", "*", lambda q, k, v, context: q)


def test_attention_declarations_project_canonical_surfaces_and_metadata() -> None:
    declarations = attention_declarations(pack_a_contribution())

    assert ATTENTION_SURFACES == (
        ATTENTION_QKV_SURFACE,
        ATTENTION_WRAPPER_SURFACE,
        ATTENTION_OUTPUT_SURFACE,
        ATTENTION_BACKEND_SURFACE,
        BLOCK_INJECTION_SURFACE,
    )
    assert tuple((item.surface_id, item.id) for item in declarations) == (
        (ATTENTION_QKV_SURFACE, "attention_a.qkv"),
        (ATTENTION_WRAPPER_SURFACE, "attention_a.wrapper"),
        (ATTENTION_OUTPUT_SURFACE, "attention_a.output"),
        (ATTENTION_BACKEND_SURFACE, "attention_a.backend.unet"),
        (BLOCK_INJECTION_SURFACE, "attention_a.inject"),
    )
    by_id = {item.id: dict(item.behavior_metadata) for item in declarations}
    qkv = by_id["attention_a.qkv"]
    assert qkv["contractVersion"] == 1
    assert qkv["order"] == 1
    assert qkv["selectorFamily"] == "unet"
    assert qkv["selectorBlock"] == "middle_block.1.transformer_blocks.0"
    assert qkv["selectorKind"] == "self"
    assert qkv["torchVersion"] == "2.13.0+cpu"
    assert qkv["aimdoVersion"] == "0.5.5"
    assert qkv["config.signed"] == "yes"
    wrapper = by_id["attention_a.wrapper"]
    assert wrapper["order"] == -2
    assert wrapper["terminal"] is True
    assert by_id["attention_a.backend.unet"]["family"] == "unet"
    assert "order" not in by_id["attention_a.backend.unet"]
    assert by_id["attention_a.inject"]["phase"] == "before"
    assert {"contractVersion", "order", "selectorFamily"} <= set(by_id["attention_a.output"])


def test_attention_pins_checked_against_installed_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_module, "_installed_version", _matching_versions)
    contribution = pack_a_contribution()
    check_attention_pins("pack_a", contribution)  # no raise

    def torch_mismatch(name: str) -> str:
        return "2.9.1" if name == TORCH_DISTRIBUTION else "0.5.5"

    monkeypatch.setattr(attention_module, "_installed_version", torch_mismatch)
    with pytest.raises(
        AttentionPinError, match="pack_a.*torch pin '2\\.13\\.0\\+cpu'.*has '2\\.9\\.1'.*reinstall"
    ):
        check_attention_pins("pack_a", contribution)

    def missing(name: str) -> str:
        return "2.13.0+cpu" if name == TORCH_DISTRIBUTION else ""

    monkeypatch.setattr(attention_module, "_installed_version", missing)
    with pytest.raises(
        AttentionPinError,
        match="dinkster-aimdo pin '0\\.5\\.5'.*has 'not installed'.*recreate",
    ):
        check_attention_pins("pack_a", contribution)


def test_materialization_projects_attention_and_enforces_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(attention_module, "_installed_version", _matching_versions)
    generation = _materialize(tmp_path, _entries("attention_pack_a"))
    assert generation.attention_contributions == (("pack_a", pack_a_contribution()),)
    declarations = generation.extensions[0][1]
    assert {item.surface_id for item in declarations} == set(ATTENTION_SURFACES)

    monkeypatch.undo()
    for module in PACK_MODULES:
        sys.modules.pop(module, None)
    with pytest.raises(AttentionPinError, match="torch pin '999\\.0\\.0'.*reinstall|recreate"):
        _materialize(
            tmp_path,
            (SamplerExtensionEntry("pack_b", "attention_pack_b:register_mismatched_pin"),),
        )


def test_two_family_backends_coexist_and_same_family_collides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(attention_module, "_installed_version", _matching_versions)

    generation = _materialize(tmp_path, _entries("attention_pack_a", "attention_pack_b"))
    backends = [
        item.id
        for _, declarations in generation.extensions
        for item in declarations
        if item.surface_id == ATTENTION_BACKEND_SURFACE
    ]
    assert backends == ["attention_a.backend.unet", "attention_b.backend.flux"]

    for module in PACK_MODULES:
        sys.modules.pop(module, None)
    with pytest.raises(
        RuntimeError,
        match=(
            "family 'unet'.*'pack_a'.*'attention_a\\.backend\\.unet'.*"
            "'pack_b'.*'attention_b\\.backend\\.unet'"
        ),
    ):
        _materialize(
            tmp_path,
            (
                SamplerExtensionEntry("pack_a", "attention_pack_a:register"),
                SamplerExtensionEntry("pack_b", "attention_pack_b:register_colliding"),
            ),
        )


def test_duplicate_attention_descriptor_ids_refuse_across_packs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(attention_module, "_installed_version", _matching_versions)
    with pytest.raises(
        RuntimeError,
        match="'attention_a\\.qkv'.*'pack_a'.*'pack_b'",
    ):
        _materialize(
            tmp_path,
            (
                SamplerExtensionEntry("pack_a", "attention_pack_a:register"),
                SamplerExtensionEntry("pack_b", "attention_pack_b:register_duplicate_id"),
            ),
        )


def test_attention_declaration_drift_fails_expected_extensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(attention_module, "_installed_version", _matching_versions)
    generation = _materialize(tmp_path, _entries("attention_pack_a"))
    declarations = generation.extensions[0][1]
    drifted = tuple(
        _replaced_metadata(item, "selectorFamily", "flux") if item.id == "attention_a.qkv" else item
        for item in declarations
    )
    key = f"candidate:{uuid.uuid4().hex}"
    write_sampler_catalog(
        _catalog(tmp_path),
        key,
        _entries("attention_pack_a"),
        expected_extensions=(("pack_a", drifted),),
    )
    with pytest.raises(RuntimeError, match="declarations changed during materialization"):
        materialize_inference_generation(key, catalog_path=_catalog(tmp_path))


def test_attention_declarations_carry_behavior_identity() -> None:
    declarations = attention_declarations(pack_a_contribution())
    stable = ExtensionSnapshot(
        extensions=(
            ActiveExtension(
                id="pack-a",
                version="1.0.0",
                package_digest="d",
                keyed_contributions=declarations,
            ),
        )
    )
    altered = ExtensionSnapshot(
        extensions=(
            ActiveExtension(
                id="pack-a",
                version="1.0.0",
                package_digest="d",
                keyed_contributions=tuple(
                    _replaced_metadata(item, "phase", "after")
                    if item.id == "attention_a.inject"
                    else item
                    for item in declarations
                ),
            ),
        )
    )
    assert extension_behavior_hash(stable) != extension_behavior_hash(altered)


def test_public_api_exports_attention_contract() -> None:
    import dinkster_api.v1 as api
    import dinkster_inference

    names = (
        "AIMDO_DISTRIBUTION",
        "ATTENTION_BACKEND_SURFACE",
        "ATTENTION_OUTPUT_SURFACE",
        "ATTENTION_QKV_SURFACE",
        "ATTENTION_WRAPPER_SURFACE",
        "BLOCK_INJECTION_SURFACE",
        "TORCH_DISTRIBUTION",
        "AttentionBackendDescriptor",
        "AttentionCallContext",
        "AttentionContribution",
        "AttentionKernelFn",
        "AttentionOutputDescriptor",
        "AttentionOutputTransform",
        "AttentionPinError",
        "AttentionQKVDescriptor",
        "AttentionQKVTransform",
        "AttentionSelector",
        "AttentionTokenSpan",
        "AttentionWrapperDescriptor",
        "AttentionWrapperFn",
        "AttentionWrapperNext",
        "BlockInjectionDescriptor",
        "BlockInjectionTransform",
        "attention_declarations",
    )
    for name in names:
        assert name in api.__all__, name
        assert getattr(api, name) is getattr(dinkster_inference, name), name
    # The pin check runs inside the inference worker; it is importable from
    # dinkster_inference but is not a pack-authoring name on the v1 door.
    assert "check_attention_pins" not in api.__all__
    assert dinkster_inference.check_attention_pins is check_attention_pins


def test_composition_refuses_attention_pin_mismatch_with_guidance(tmp_path: Path) -> None:
    from test_inference_extensions import _host_manifest, _worker_env

    from dinkster.compose import CompositionError, PackSpec, ServingComposer

    def manifest(root: Path, name: str, entry: str) -> Path:
        root.mkdir(parents=True)
        path = root / "dinkster-pack.toml"
        path.write_text(
            f'[pack]\\nname = "{name}"\\n'
            f'namespaces = ["{name}"]\\n\\n'
            '[pack.entry]\\nnodes = "s1_sampler_empty:NODES"\\n\\n'
            "[pack.extension]\\n"
            f'inference = "{entry}"\\n'
            'privileges = ["inference"]\\n',
            encoding="utf-8",
        )
        return path

    async def scenario() -> None:
        composer = ServingComposer(worker_env=_worker_env(tmp_path))
        try:
            await composer.add_pack(
                PackSpec(_host_manifest(tmp_path / "host"), trust_reserved=True)
            )
            with pytest.raises(
                CompositionError,
                match="torch pin '999\\.0\\.0'.*reinstall|recreate the execution environment",
            ):
                await composer.add_pack(
                    PackSpec(
                        manifest(
                            tmp_path / "attention",
                            "attention_b",
                            "attention_pack_b:register_mismatched_pin",
                        )
                    )
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
