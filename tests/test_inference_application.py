"""Torch-free resident-component application contract tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from dinkster_inference import (
    FLOAT16,
    FLOAT32,
    ApplicationChain,
    ComponentApplication,
    Conditioning,
    ReconstructionRecipe,
    RuntimeKnobs,
    WeightSourceBinding,
    WeightSourceRef,
    extend_runtime_identity,
)


def _recipe(family_id: str = "dinkster.qwen_image") -> ReconstructionRecipe:
    return ReconstructionRecipe(
        sources=(
            WeightSourceBinding(
                "diffusion",
                WeightSourceRef("blake3:" + "0" * 64, "diffusion.safetensors", 1),
            ),
        ),
        family_id=family_id,
        component_identity=(f"family={family_id}", "component=diffusion"),
        knobs=RuntimeKnobs(
            diffusion_dtype=FLOAT16.name,
            text_dtype=FLOAT32.name,
            vae_dtype=FLOAT32.name,
            fp8_matmul=False,
        ),
    )


class _Runtime:
    def __init__(self, recipe: ReconstructionRecipe) -> None:
        self.family = type("Family", (), {"id": recipe.family_id})()
        self.runtime_identity = recipe.runtime_identity

    def encode_text(self, text: str) -> Conditioning[Any]:
        raise NotImplementedError

    def sample(self, latent: object, **kwargs: object) -> object:
        raise NotImplementedError

    def custom_sampling_sigmas(self, *_args: object, **_kwargs: object) -> None: ...

    def custom_sampling_beta_sigmas(self, *_args: object, **_kwargs: object) -> None: ...

    def custom_sampling_sd_turbo_sigmas(self, *_args: object, **_kwargs: object) -> None: ...

    def custom_sampling_percent_to_sigma(self, *_args: object, **_kwargs: object) -> None: ...

    def check_custom_sampling(self, _request: object, **_kwargs: object) -> None: ...

    def sample_custom(self, *_args: object, **_kwargs: object) -> None: ...

    def decode_latent(self, latent: object) -> object:
        raise NotImplementedError

    def encode_content(self, content: object) -> object:
        raise NotImplementedError


class _RuntimeHandle:
    def __init__(self, recipe: ReconstructionRecipe | None = None) -> None:
        self.recipe = recipe or _recipe()
        self.runtime = _Runtime(self.recipe)
        self.load_device = "cpu"
        self.staged: list[str] = []

    def require_active(self) -> None:
        pass

    @contextmanager
    def stage(self, role: str) -> Iterator[None]:
        self.staged.append(role)
        yield


class _ComponentHandle:
    def __init__(self, identity: str) -> None:
        self._component = object()
        self.resource_identity = identity
        self.load_device = "cpu"

    @property
    def component(self) -> object:
        return self._component

    def require_active(self) -> None:
        pass

    @contextmanager
    def stage(self, *, memory_required: int = 0) -> Iterator[None]:
        del memory_required
        yield

    @contextmanager
    def stage_with(self, runtime_handle: object, role: str) -> Iterator[None]:
        del runtime_handle, role
        yield


def _materialize(runtime: object, component: object, latent: object) -> Mapping[str, object]:
    return {"control": (runtime, component, latent)}


def _application(
    runtime_handle: _RuntimeHandle,
    fact: str,
    *,
    family_id: str | None = None,
) -> ComponentApplication:
    identity = runtime_handle.recipe.runtime_identity
    component = _ComponentHandle(identity)
    return ComponentApplication(
        family_id=family_id or runtime_handle.recipe.family_id,
        role="diffusion",
        handle=component,
        application_identity=extend_runtime_identity(identity, (fact,)),
        materialize_application_kwargs=_materialize,
    )


def test_application_chain_identity_is_byte_stable_and_order_sensitive() -> None:
    model = _RuntimeHandle()
    first = _application(model, "control=first")
    second = _application(model, "control=second")

    forward = ApplicationChain(model, (first, second))
    repeated = ApplicationChain(model, (first, second))
    reversed_chain = ApplicationChain(model, (second, first))

    assert forward.chain_identity == repeated.chain_identity
    assert forward.chain_identity != reversed_chain.chain_identity
    assert forward.chain_identity.split(":")[:2] == model.recipe.runtime_identity.split(":")[:2]
    assert extend_runtime_identity(forward.chain_identity, ("invocation=test",))


def test_application_identity_rotates_with_component_and_behavior_facts() -> None:
    model = _RuntimeHandle()
    base = model.recipe.runtime_identity
    first_component = extend_runtime_identity(base, ("component=first",))
    second_component = extend_runtime_identity(base, ("component=second",))

    first = extend_runtime_identity(first_component, ("strength=0.5",))
    changed_strength = extend_runtime_identity(first_component, ("strength=0.75",))
    changed_component = extend_runtime_identity(second_component, ("strength=0.5",))

    assert len({first, changed_strength, changed_component}) == 3


def test_application_chain_append_preserves_order_and_base_model() -> None:
    model = _RuntimeHandle()
    first = _application(model, "control=first")
    second = _application(model, "control=second")
    chain = ApplicationChain(model, (first,)).append(second)

    assert chain.model is model
    assert chain.applications == (first, second)
    assert chain.base_model_identity == model.recipe.runtime_identity
    assert chain._dinkster_resident_owner is model
    assert chain._dinkster_resident_refs == (first.handle, second.handle)
    assert chain._dinkster_resident_fingerprint == chain.chain_identity


def test_application_chain_includes_declared_resident_dependency_owners() -> None:
    model = _RuntimeHandle()
    owner = object()

    class Dependency:
        _dinkster_resident_owner = owner

    application = _application(model, "control=first")
    application = ComponentApplication(
        application.family_id,
        application.role,
        application.handle,
        application.application_identity,
        application.materialize_application_kwargs,
        resident_dependencies=(Dependency(),),
    )

    assert ApplicationChain(model, (application,))._dinkster_resident_refs == (
        application.handle,
        owner,
    )


def test_application_values_are_frozen() -> None:
    model = _RuntimeHandle()
    application = _application(model, "control=first")
    chain = ApplicationChain(model, (application,))

    with pytest.raises(FrozenInstanceError):
        application.role = "text"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        chain.applications = ()  # type: ignore[misc]


def test_empty_and_nested_application_chains_are_refused() -> None:
    model = _RuntimeHandle()
    application = _application(model, "control=first")
    chain = ApplicationChain(model, (application,))

    with pytest.raises(ValueError, match="applications must be non-empty"):
        ApplicationChain(model, ())
    with pytest.raises(TypeError, match="must not be nested"):
        ApplicationChain(chain, (application,))


def test_application_chain_refuses_family_mismatch() -> None:
    model = _RuntimeHandle()
    other_family = _RuntimeHandle(_recipe("dinkster.other"))
    other_family_application = _application(other_family, "control=other")

    with pytest.raises(ValueError, match="family_id must match the base model family"):
        ApplicationChain(model, (other_family_application,))


@pytest.mark.parametrize(
    ("identity", "message"),
    (
        ("not-native", "3-part native identity"),
        ("native::" + "1" * 64, "family must be non-empty"),
        ("native:dinkster.qwen_image:extra:" + "1" * 64, "3-part native identity"),
        ("native:dinkster.qwen_image:" + "A" * 64, "lowercase sha256"),
    ),
)
def test_component_application_refuses_malformed_identity(identity: str, message: str) -> None:
    model = _RuntimeHandle()
    with pytest.raises(ValueError, match=message):
        ComponentApplication(
            family_id=model.recipe.family_id,
            role="diffusion",
            handle=_ComponentHandle(model.recipe.runtime_identity),
            application_identity=identity,
            materialize_application_kwargs=_materialize,
        )


def test_component_application_refuses_bad_fields() -> None:
    model = _RuntimeHandle()
    identity = extend_runtime_identity(model.recipe.runtime_identity, ("control=first",))
    component = _ComponentHandle(model.recipe.runtime_identity)

    with pytest.raises(ValueError, match="role must be a canonical id"):
        ComponentApplication(model.recipe.family_id, "Uppercase", component, identity, _materialize)
    with pytest.raises(ValueError, match="identity family must match family_id"):
        ComponentApplication("dinkster.other", "diffusion", component, identity, _materialize)
    with pytest.raises(ValueError, match="handle identity family must match family_id"):
        ComponentApplication(
            model.recipe.family_id,
            "diffusion",
            _ComponentHandle(identity.replace("dinkster.qwen_image", "dinkster.other")),
            identity,
            _materialize,
        )
    with pytest.raises(TypeError, match="materialize_application_kwargs must be callable"):
        ComponentApplication(
            model.recipe.family_id,
            "diffusion",
            component,
            identity,
            None,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="resident_dependencies must be a tuple"):
        ComponentApplication(
            model.recipe.family_id,
            "diffusion",
            component,
            identity,
            _materialize,
            resident_dependencies=[],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="resident dependencies must not be None"):
        ComponentApplication(
            model.recipe.family_id,
            "diffusion",
            component,
            identity,
            _materialize,
            resident_dependencies=(None,),
        )
